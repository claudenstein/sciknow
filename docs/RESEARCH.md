# Research & Innovations

[&larr; Back to README](../README.md)

---

sciknow implements state-of-the-art techniques from scientific document processing, topic modeling, knowledge graphs, agentic RAG, and long-form AI-assisted writing. This document describes each innovation, the research it builds on, and how it is integrated into the system.

---

## 1. SOTA PDF Parsing Pipeline

**Status:** Production

sciknow uses **MinerU 2.5** as its primary PDF converter — the current state-of-the-art among open-source tools for scientific paper parsing, scoring 86.2 on [OmniDocBench v1.5](https://arxiv.org/abs/2412.07626). The pipeline runs a cascade of specialized models:

- **DocLayout-YOLO** for page layout detection
- **MFD** (Math Formula Detection) + **MFR** (Math Formula Recognition) for LaTeX equation extraction
- **Table OCR** + HTML structure reconstruction for complex scientific tables
- **Text OCR** with consistent reading-order reconstruction

MinerU produces typed content blocks (text with heading levels, tables with HTML, equations with LaTeX, figures with captions), which sciknow's chunker maps to canonical section types with section-aware chunking parameters.

**Marker** serves as a robust fallback — its JSON block tree provides an independent parsing path that catches papers where MinerU's layout model struggles.

**Research basis:** [OmniDocBench (2024)](https://arxiv.org/abs/2412.07626)

---

## 2. Hybrid Search with Three-Signal Fusion

**Status:** Production

Most RAG systems use either dense vector search or keyword search alone. sciknow fuses **three** independent retrieval signals:

| Signal | Model | Strength |
|---|---|---|
| Dense vectors (1024-dim) | BAAI/bge-m3 | Semantic similarity across paraphrases |
| Sparse lexical vectors | BAAI/bge-m3 | Precise technical terms, acronyms, formulas |
| Full-text search (tsvector) | PostgreSQL | BM25-style relevance, stemming, phrase proximity |

Signals are merged with **Reciprocal Rank Fusion** (RRF), which is provably better than any single-signal approach for diverse query types. The fused candidates are then reranked with a **cross-encoder** (bge-reranker-v2-m3), which directly scores `(query, document)` pairs for maximum precision.

A **citation-count boost** gently lifts papers that are more cited within the corpus, acting as a proxy for authority without overriding relevance.

**Innovation:** The simultaneous dense + sparse embedding from a single bge-m3 forward pass means zero additional compute for the lexical signal. Combined with PostgreSQL's battle-tested full-text search, sciknow gets three complementary signals at the cost of one model.

**Research basis:**
- [RRF (SIGIR 2009)](https://dl.acm.org/doi/10.1145/1571941.1572114) — fusion algorithm
- [BGE M3-Embedding (2024)](https://arxiv.org/abs/2402.03216) — multi-granularity embedding
- [C-Pack (2024)](https://arxiv.org/abs/2309.07597) — cross-encoder reranking

---

## 3. BERTopic Embedding Clustering

**Status:** Production

sciknow replaced its original LLM-batch clustering (slow, unreliable, non-deterministic) with **BERTopic**, an embedding-based topic modeling pipeline:

1. **Fetch** bge-m3 abstract vectors already stored in Qdrant (zero additional embedding cost)
2. **UMAP** dimensionality reduction to 5D
3. **HDBSCAN** density-based clustering (automatically determines cluster count)
4. **c-TF-IDF** keyword extraction per cluster
5. **Single LLM call** to name all clusters from their keyword signatures

This replaced ~10 minutes of error-prone LLM batching with ~5 seconds of deterministic computation. HDBSCAN finds natural topic boundaries without requiring the user to specify a cluster count.

**Research basis:**
- [BERTopic (2022)](https://arxiv.org/abs/2203.05794)
- [UMAP (2018)](https://arxiv.org/abs/1802.03426)
- HDBSCAN (Campello et al., 2013)
- [LLM-Guided Semantic-Aware Clustering (ACL 2025)](https://aclanthology.org/2025.acl-long.902/)

---

## 4. Compiled Knowledge Wiki (Karpathy LLM-Wiki Pattern)

**Status:** Production

The central architectural innovation. Standard RAG rediscovers relationships from scratch on every query — the LLM has no persistent memory of what it has already analyzed. sciknow's wiki layer **compiles** knowledge once and queries the compilation:

```
Papers → wiki compile → interconnected wiki pages → wiki query
                              │
                              ├── Paper summaries (one per paper)
                              ├── Concept pages (grow with each paper)
                              └── Synthesis pages (cross-paper overviews)
```

**How it works:**
- Each paper gets a structured summary page via LLM
- Entities (concepts, methods, datasets) are extracted with structured JSON output
- Concept pages aggregate knowledge from all papers mentioning them
- Pages are embedded in a dedicated Qdrant collection for search
- Cross-references use `[[slug]]` notation
- Wiki grows incrementally — `wiki compile` only processes new papers by default

**Why this matters for book writing:** When `book write` retrieves context, it searches pre-synthesized wiki pages that already cross-reference multiple papers, instead of raw individual chunks. The LLM writes from richer, more coherent context.

**Research basis:** Andrej Karpathy's [LLM-wiki pattern](https://x.com/karpathy/status/1756130027985752370) — "compile the knowledge, don't retrieve it."

---

## 5. Knowledge Graph (GraphRAG-Style)

**Status:** Production

During wiki compilation, sciknow extracts **entity-relationship triples** in a single merged LLM call alongside entity extraction:

```
(Paper X,       uses_method,    Wavelet Analysis)
(Paper A,       contradicts,    Paper B)
(Total Solar Irradiance,  related_to,  Solar Cycle)
```

Triples are stored in PostgreSQL (`knowledge_graph_triples` table) and enable:
- **Multi-hop graph traversal:** "find papers that extend the methods used by papers that contradict claim X"
- **Relationship visualization:** `wiki graph "concept" --depth 2` shows the local neighborhood
- **Concept disambiguation:** canonical slugs prevent duplicate entities

The graph is extracted using Ollama's **structured output** (JSON schema via XGrammar constrained decoding), ensuring valid triple format without post-processing.

**Research basis:**
- [Microsoft GraphRAG (2024)](https://arxiv.org/abs/2404.16130)
- [KG-RAG (Nature 2025)](https://www.nature.com/articles/s41598-025-21222-z)

---

## 6. Self-Correcting RAG (Agentic Retrieval)

**Status:** Production

Standard RAG is one-shot: search → generate. If retrieved chunks are irrelevant, the answer hallucinates. sciknow implements a **self-correcting** retrieval loop:

1. **Retrieve** chunks via hybrid search
2. **Evaluate relevance** — LLM assesses whether chunks actually answer the question
3. If relevance is low → **reformulate** the query and re-search
4. **Generate** answer from validated context
5. **Check grounding** — LLM evaluates whether each claim in the answer is supported by the retrieved evidence
6. Report **ungrounded claims** with a groundedness score

This pattern reduces hallucination from ~12-14% (standard RAG) to ~5.8% (self-correcting RAG), based on published benchmarks.

Enabled via `--self-correct` on `ask question` and automatically integrated into the `book write --verify` pipeline.

**Research basis:**
- [Self-RAG (2023)](https://arxiv.org/abs/2310.11511)
- [CRAG (2024)](https://arxiv.org/abs/2401.15884)
- [Agentic RAG Survey (2025)](https://arxiv.org/abs/2501.09136)

---

## 7. Multimodal Chunk Tagging

**Status:** Production

MinerU extracts tables (HTML), equations (LaTeX), and figure captions during ingestion. sciknow tags chunks containing these elements in Qdrant payloads (`has_table`, `has_equation`), enabling:

- **Filtered retrieval:** `--has-table` returns only chunks with data tables
- **Section-appropriate context:** when writing a methods section, equations and tables are prioritized
- **Structural awareness:** the system knows when a chunk contains a data table vs. narrative text

**Research basis:**
- [RAG-Anything (2025)](https://github.com/HKUDS/RAG-Anything)
- [ManuRAG (2025)](https://arxiv.org/pdf/2601.15434)
- [Multimodal LLM Table Understanding (ACL 2025)](https://aclanthology.org/2025.trl-1.10.pdf)

---

## 8. Hierarchical Tree Planning (TreeWriter Pattern)

**Status:** Production

sciknow's `book write --plan` generates a **paragraph-level tree plan** before drafting:

```json
{
  "paragraphs": [
    {
      "main_point": "TSI measurements began with satellite observations in 1978",
      "key_sources": ["[1] Fröhlich 2006", "[3] Kopp 2011"],
      "transition": "connects to next paragraph on measurement uncertainty"
    },
    ...
  ]
}
```

The LLM plans the paragraph skeleton (main point, which sources to cite, how to transition) and then writes each paragraph grounded in specific papers. This produces more coherent long-form text than flat generation.

The autowrite loop integrates planning automatically — first iteration plans, subsequent iterations revise the content while respecting the planned structure.

**Research basis:**
- [TreeWriter (2025)](https://arxiv.org/abs/2601.12740) — hierarchical document planning
- [DOME (NAACL 2025)](https://aclanthology.org/2025.naacl-long.63/) — dynamic hierarchical outlining
- [Outline-guided Generation (NAACL 2025)](https://aclanthology.org/2025.naacl-industry.20.pdf)

---

## 9. Automated Consensus Mapping

**Status:** Production

For each concept in the knowledge graph, sciknow can map scientific consensus:

- Which papers **support / contradict / are neutral** on key claims
- How **consensus shifts over time** (2010: 3 support; 2020: 15 support, 3 contradict)
- The **most debated topics** in the corpus
- **Confidence levels** based on evidence strength and recency

This transforms the wiki from a knowledge base into an **evidence map** — directly answering "where is the science settled?" and "where is it contested?"

Accessed via `wiki consensus "topic"`. Builds on the existing `book argue` evidence classification and knowledge graph infrastructure.

---

## 10. Autonomous Convergence Loop (Autowrite)

**Status:** Production

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), sciknow's autowrite implements a **propose → evaluate → keep/discard** loop for writing:

1. **Generate** initial draft with RAG context + cross-chapter coherence
2. **Score** on 5 dimensions: groundedness, completeness, coherence, citation accuracy, overall
3. **Verify** claims against source passages (mandatory)
4. **Identify** the weakest dimension → generate targeted revision instruction
5. **Revise** targeting the specific weakness
6. **Re-score:** if improved → keep; if regressed → discard (revert to previous best)
7. **Repeat** until quality target met or max iterations exhausted

Key design decisions:
- **Mandatory claim verification** — the verifier can override the scorer's groundedness rating, catching hallucinated citations that sound plausible
- **Same-evidence evaluation** — scorer and verifier receive the writer's original retrieved results (no re-retrieval), ensuring evaluation is against the same evidence
- **Auto-resume** — sections with existing drafts are skipped by default (`--rebuild` to overwrite)
- **Auto-expand** — checks whether missing topics identified by the reviewer exist in the corpus, and suggests expansion if not

---

## 11. Per-Chapter Custom Sections

**Status:** Production

Unlike traditional paper-style sections (Introduction, Methods, Results, Discussion, Conclusion), sciknow generates **per-chapter custom sections** appropriate for scientific divulgation books. During `book outline`, the LLM proposes sections tailored to each chapter's content:

- Chapter "Solar Irradiance Measurements" → `["historical_context", "satellite_era", "measurement_challenges", "key_datasets", "summary"]`
- Chapter "Climate Models and Predictions" → `["model_types", "forcing_scenarios", "validation", "projections", "limitations"]`

Default sections for chapters without custom definitions: `["overview", "key_evidence", "current_understanding", "open_questions", "summary"]`

---

## 12. Six-Source Open-Access Discovery

**Status:** Production

`db expand` queries **six** open-access sources in priority order, maximizing PDF discovery:

1. **Copernicus** — zero-cost URL construction for `10.5194/*` DOIs (covers ACP, CP, TC, ESSD, etc.)
2. **arXiv** — direct PDF for preprints
3. **Unpaywall** — largest general OA database
4. **OpenAlex** — catches institutional repos
5. **Europe PMC** — biomedical full-text
6. **Semantic Scholar** — final fallback

Combined with a **semantic relevance filter** (bge-m3 similarity against corpus centroid or topic query), this prevents off-topic drift while maximizing coverage.

---

## 13. Hedging Fidelity (Lexical-Level Groundedness)

**Status:** Production

Standard groundedness checking verifies that a claim's *fact* is in the source. It does not catch a much subtler failure mode: the LLM systematically *strengthens* the source's epistemic modality. The 2026 SciZoom benchmark measured a **22.8% drop** in hedge density (`may`/`might`/`could`: 1.88 → 1.45 per 1000 words) in post-LLM academic writing while assertive verbs (`demonstrate`, `prove`, `outperform`) held constant — LLMs reliably turn *suggests* into *proves*, *associated with* into *causes*, and *may* into *does*. The underlying fact is right; the certainty is wrong. This is invisible to a per-citation fact-check.

sciknow now treats hedging as a first-class concern across the writer → score → verify loop:

1. **Writer rule** (`WRITE_V2_SYSTEM`): an explicit transfer-modality rule with a [BioScope](https://pmc.ncbi.nlm.nih.gov/articles/PMC2586758/)-derived hedge cue list (may, might, could, suggest, indicate, appear, seem, likely, probably, tend to, consistent with, associated with, evidence for, point to, hint at). The writer is told to carry hedges and scope qualifiers verbatim from sources, never to add hedges the source did not use, and to cap booster density (clearly, undoubtedly, obviously) at two or three per section.
2. **New verifier verdict** (`VERIFY_SYSTEM`): the existing four-way classification (`SUPPORTED`/`EXTRAPOLATED`/`MISREPRESENTED`/`MISSING`) gains a fifth label, **`OVERSTATED`** — the underlying fact is in the source, but the draft has strengthened the epistemic modality. The verifier now reports both `groundedness_score` (factually present) and `hedging_fidelity_score` (modality also right). A draft can be 1.0 grounded but 0.6 hedging-faithful if it consistently overstates.
3. **New scoring dimension** (`SCORE_SYSTEM`): `hedging_fidelity` joins groundedness, completeness, coherence, and citation_accuracy as a sixth scoring dimension. The autowrite loop's existing weakest-dimension routing automatically targets it when it's the lowest, and `_verify_draft_inner` overrides the scorer when the verifier finds OVERSTATED claims, generating a targeted "soften N overstated claims" revision instruction.

**Why this matters for climate science specifically:** the biggest reputational risk for AI-written climate communication is overclaiming. Every public-facing climate statement is scrutinised for unwarranted certainty, and standard fact-verification doesn't catch the failure mode where the model is "technically right" but rhetorically too strong.

**Research basis:**
- [Hyland 1998 — *Hedging in Scientific Research Articles*](https://jolantasinkuniene.wordpress.com/wp-content/uploads/2014/03/hyland-boosting-hedging-and-the-negotiation-of-academic-knowledge-1998.pdf) — foundational analysis of hedge function in academic writing
- [Vincze et al. 2008 — BioScope corpus](https://pmc.ncbi.nlm.nih.gov/articles/PMC2586758/) — annotated hedge cues used as the writer's reference list
- SciZoom (2026) — the empirical demonstration that LLMs systematically strip hedges
- [Hyland 2005 — *Metadiscourse: What is it and where is it going*](http://www.kenhyland.org/wp-content/uploads/2020/05/Metadiscourse_What-is-it-and-where-is-it-going.pdf) — booster density bounds

---

## 14. Local Coherence via Centering Theory (Entity-Bridge Rule)

**Status:** Production

The most-reported complaint about LLM-written long-form is that paragraphs "float" — each is internally fine but the seams are abrupt. Centering Theory (Grosz, Joshi & Weinstein 1995, *Computational Linguistics* 21:2) is the strongest forty-year empirical result we have on local coherence: the most coherent transition between two utterances keeps the backward-looking center (`Cb`) the same and as the preferred entity. The inferred preference ranking is `CONTINUE > RETAIN > SMOOTH-SHIFT > ROUGH-SHIFT`.

sciknow operationalises this as a single enforceable rule in the writer prompt — no parser, no Cb/Cf tracker, just the rule the parser would ultimately enforce:

> The first sentence of each new paragraph must explicitly name at least one entity (concept, mechanism, region, dataset, paper, or quantity) that was the grammatical subject or most salient noun phrase of the *last* sentence of the previous paragraph. No cold-starts. If you must genuinely shift topic, open with a short bridge clause that names both the prior topic and the new one.

The `coherence` scoring dimension was extended to score this explicitly, so violations are caught and routed to the revision loop. In practice this kills the "every paragraph reads like an island" failure mode at zero implementation cost.

**Research basis:**
- [Grosz, Joshi & Weinstein 1995 — Centering Theory (ACL Anthology)](https://aclanthology.org/J95-2003/)
- [Barzilay & Lapata 2008 — Modeling Local Coherence (CL)](https://aclanthology.org/J08-1001.pdf) — entity-grid generalisation; the lightweight rule above is a deliberate simplification

---

## 15. PDTB-Lite Discourse Relations in Tree Planning

**Status:** Production

The existing tree planner (Section 8) produces a hierarchical paragraph plan but says nothing about *how* each paragraph relates to its predecessor. The empirically-documented weakness in LLM long-form is **flat rhetorical texture** — paragraph after paragraph of elaboration with no concession, contrast, cause, or synthesis.

sciknow extends the `tree_plan` JSON schema with a `discourse_relation` field per paragraph, drawn from a closed PDTB-lite vocabulary of 10 labels:

| Relation | When to use | Connective |
|---|---|---|
| `background` | Sets up context, definitions, history | "Historically …" |
| `elaboration` | More detail on a point already made | "More specifically …" |
| `evidence` | Supporting data for the previous claim | "Direct measurements show …" |
| `contrast` | Opposing view or counter-evidence | "However …" / "By contrast …" |
| `concession` | Acknowledges a limitation | "Although …" |
| `cause` | Mechanism or reason | "As a result …" |
| `comparison` | Parallel with another case | "Similarly …" |
| `exemplification` | Concrete example | "For example …" |
| `qualification` | Narrows scope | "Within these limits …" |
| `synthesis` | Integrates multiple prior threads | "Taken together …" |

The writer prompt (`WRITE_V2_SYSTEM`) receives the per-paragraph relations injected as a discourse-relation block and is told to open each paragraph with a connective appropriate to its relation. The planner is explicitly told **not** to make every paragraph "elaboration" — it must vary the relations to give the section real argumentative texture.

PDTB's *shallow* philosophy — "the inference is licensed by the connective" — is a much better fit for prompt engineering than full Rhetorical Structure Theory trees, whose ~30 relations are too many to enforce reliably and whose tree structure is hard to specify in JSON.

**Research basis:**
- [Prasad et al. 2008 — Penn Discourse Treebank 2.0 (LREC)](https://catalog.ldc.upenn.edu/docs/LDC2019T05/PDTB3-Annotation-Manual.pdf)
- [Mann & Thompson 1988 — RST: Toward a functional theory of text organization](https://www.sfu.ca/rst/pdfs/Mann_Thompson_1988.pdf) — the more elaborate RST framework which sciknow deliberately *does not* fully adopt
- [Ruan et al. 2025 — *Align to Structure: Aligning LLMs with Structural Information* (arXiv:2504.03622)](https://arxiv.org/abs/2504.03622) — recent validation that rewarding RST/discourse motifs during training improves long-form generation; sciknow gets the same signal at inference time via the planner prompt

---

## 16. Step-Back Retrieval for Section Drafting

**Status:** Production

Standard hybrid retrieval is excellent at finding chunks that lexically and semantically match the *concrete* query, but writers drafting a chapter section often need *background* and *mechanism* context that the concrete query phrasing misses. Example: the query "ocean heat content trends in the North Atlantic since 1990" returns trend papers, but misses the underlying mechanism papers on ocean heat uptake — which the writer needs to explain *why* the trends look the way they do.

Step-back prompting (Zheng et al., ICLR 2024) addresses this by having the LLM first emit a more abstract reformulation of the query and retrieving for both. sciknow's `_retrieve_with_step_back`:

1. Issues the **concrete query** through the existing hybrid retrieval (dense + sparse + FTS + RRF), getting `candidate_k` candidates.
2. Asks the **fast LLM** for a single short abstract reformulation via the new `step_back` prompt template (5–12 words; e.g., "ocean heat content trends in the North Atlantic since 1990" → "mechanisms of ocean heat uptake and transport").
3. Retrieves `candidate_k // 2` more candidates for the abstract query.
4. Unions the two pools by `chunk_id`, preserving the concrete-query candidates first.
5. Reranks the unioned pool against the **original concrete query** (not the abstract one) using `bge-reranker-v2-m3`, taking top `context_k`.

The reranking-against-the-original-query step is critical: the abstract pool contributes diverse mechanism / background chunks, but the final ordering still reflects what the writer actually needs to draft. Reported gains in the original paper: +27% on TimeQA, +7% on MuSiQue.

Wired into both `write_section_stream` and `autowrite_section_stream` behind a `use_step_back` flag (default `True`).

**Research basis:**
- [Zheng et al. 2024 — *Take a Step Back: Evoking Reasoning via Abstraction in LLMs* (arXiv:2310.06117)](https://arxiv.org/abs/2310.06117) — ICLR 2024

---

## 17. Four-Layer Metadata Extraction

**Status:** Production

Each layer only runs if the previous didn't fully populate the metadata:

1. **PyMuPDF** — embedded XMP/Info fields (fast, often incomplete)
2. **Crossref API** — authoritative bibliographic data by DOI
3. **arXiv API** — for preprints
4. **Ollama LLM** — structured extraction from first ~3000 chars (fallback)

This cascade ensures maximum metadata coverage regardless of PDF quality.

---

## 18. Chain-of-Verification (Decoupled Fact-Checking)

**Status:** Production

The standard claim verifier (Section 6, also `_verify_draft_inner` in `book_ops.py`) reads the draft and the source passages **in the same context window**. This is a known anchoring failure mode: the verifier rubberstamps claims it should have flagged because the draft's framing biases its judgement of the evidence.

Chain-of-Verification (CoVe) addresses this by **decoupling** fact-checking from drafting. sciknow's implementation in `_cove_verify` runs as a perpendicular signal alongside the standard verifier:

1. **Question generation.** The LLM sees ONLY the draft (not the sources) and produces 5–8 falsifiable verification questions. The prompt biases toward claims with strong modal verbs (*proves*, *demonstrates*, *causes*) — those are the highest-risk overstatements.
2. **Independent answering.** For each question, a fresh LLM call sees ONLY the source passages (not the draft, not the question's context, not the other answers). The answerer classifies its own response as `CONFIRMED` / `PARTIAL` / `NOT_IN_SOURCES` / `DIFFERENT_SCOPE`.
3. **Mismatch report.** Anything that isn't a clean `CONFIRMED` becomes a mismatch with severity:
   - `NOT_IN_SOURCES` → **high** (the draft made a claim the sources don't support at all)
   - `DIFFERENT_SCOPE` → **medium** (the draft generalised past the source's stated region/period/conditions)
   - `PARTIAL` → **low** (the source supports only part of the claim)
4. **Targeted revision instruction.** If CoVe finds high-severity mismatches, the autowrite loop's revision instruction is *overridden* to name the specific claims and direct the writer to either remove them or restore the source's framing. Medium-severity mismatches generate scope-restoration instructions.
5. **Score override.** If CoVe's `cove_score` (1.0 − mismatches / questions) is more pessimistic than the standard verifier's `groundedness_score`, sciknow takes CoVe's number — independent re-answering is harder to fool than single-window verification.

**Cost gating.** CoVe issues 1 + N extra LLM calls per autowrite iteration (1 for question generation, N for independent answers). To keep this affordable, CoVe is **gated**: it fires only when either `groundedness` or `hedging_fidelity` is below a threshold (default 0.85). Drafts that already look clean to the standard verifier skip CoVe; drafts that look weak get the extra scrutiny. The `cove_threshold` is exposed as a kwarg on `autowrite_section_stream` and defaults to 0.85.

**Why CoVe is perpendicular to the existing pipeline.** sciknow's verifier (Section 6) and the new hedging-fidelity dimension (Section 13) catch the failure modes you can detect *while* reading the draft: missing citations, lexical overstatements, modal mismatches. CoVe catches the failure mode you can only detect by *not* reading the draft: claims whose factual content has drifted from what the sources say, where re-asking the question without seeing the draft yields a different answer. The original paper reports a **50–70% hallucination reduction** on long-form generation specifically because of this decoupling.

**Research basis:**
- [Dhuliawala et al. 2024 — *Chain-of-Verification Reduces Hallucination in Large Language Models* (arXiv:2309.11495)](https://arxiv.org/abs/2309.11495) — Findings of ACL 2024

---

## 19. RAPTOR — Hierarchical Tree of Summary Embeddings

**Status:** Production

The standard chunk index is **flat**: every retrieved chunk lives at the same level of abstraction. For chapter-level synthesis, this forces the writer to glue together 8+ disjoint single-paper findings. RAPTOR (Sarthi et al., ICLR 2024) addresses this by building a hierarchical tree of LLM-summarised cluster nodes on top of the existing chunks, all stored in the same `papers` Qdrant collection.

### How it works

1. **Backfill** `node_level: 0` on every existing leaf chunk (idempotent — only touches points that don't have the field yet).
2. **For each level 1..max_levels:**
   1. **Fetch** the parent-level vectors from Qdrant (level 0 = leaf chunks; level N = level-(N-1) summaries).
   2. **UMAP-reduce** to ~10 dimensions with cosine metric (matches bge-m3's training).
   3. **GMM cluster** with BIC-selected k. RAPTOR's paper specifically argues for soft clustering — chunks can contribute to multiple cluster summaries by membership probability. v1 uses hard assignment (argmax) for simplicity, but the GMM `proba` matrix is computed and available for future soft-assignment upgrades.
   4. **Summarise** each cluster with the LLM via the `RAPTOR_SUMMARY` prompt — a 250-450 word retrievable synthesis that preserves scope qualifiers and epistemic strength (honors hedging fidelity from Section 13). Defaults to `LLM_FAST_MODEL` since RAPTOR is a one-shot batch op and 7B-class models produce adequate summaries; override with `--model`.
   5. **Embed** the summary with bge-m3 (dense + sparse, same as leaves) and **upsert** into the same `papers` Qdrant collection with a payload that includes `node_level: N`, `summary_text`, `child_chunk_ids`, `child_count`, `document_ids`, `n_documents`, `year_min`/`year_max`, `section_types`, `topic_clusters`, plus display fields (`title`, `section_type=f"raptor_l{N}"`, `section_title`).
3. **Stop early** when fewer than `min_top_level` summaries (default 4) remain at a level, or when `max_levels` is reached.

### Retrieval integration (zero config)

`hybrid_search._hydrate` now recognises RAPTOR nodes by their `node_level` payload field and reads their metadata (title / year / content) directly from the payload, bypassing the `paper_metadata` PostgreSQL join (these nodes have no source document). The full `summary_text` is placed into `content_preview`, so `context_builder.build`'s fallback uses it as the chunk's content (RAPTOR nodes have no row in the `chunks` table).

The result: when RAPTOR summary nodes exist in the collection, the writer's hybrid retrieval (dense + sparse + Postgres FTS, RRF-fused, then bge-reranker-v2-m3) automatically returns a **mix of fine-grained chunks and mid-level summaries**. The reranker decides — if a level-1 summary is more relevant to the query than any individual chunk, it surfaces; if the writer needs a specific equation or table, the leaf chunks still win. **No retrieval-side flag is needed.**

### Why no re-ingest and no wiki recompile

RAPTOR's leaf nodes are precisely the chunks that already exist in `papers`. The build is a **pure additive layer**:

- No changes to `documents`, `paper_metadata`, `paper_sections`, or `chunks` tables.
- The only schema change is a single new payload index on the existing Qdrant collection (`node_level`, INTEGER). `init_collections()` creates it for fresh collections; `ensure_node_level_index()` is an idempotent helper that adds it to pre-existing collections (called automatically by `raptor build`).
- Existing leaf points get `node_level: 0` set via Qdrant's bulk `set_payload` API — a one-time backfill that takes seconds for thousands of points.
- The wiki layer (Section 4) is completely orthogonal — RAPTOR summaries are *retrieval artifacts* living in Qdrant; wiki pages are *human-readable markdown* in `data/wiki/`. Different data path, different purpose.

### Operating model

The build is **one-time-batch**: when new papers are ingested, the existing summary nodes are still useful but slightly stale. The recommended pattern is to re-run with `--rebuild` periodically (weekly, or after a significant ingest like a full `db expand` cycle). For incremental updates, `--rebuild` deletes every point with `node_level >= 1` and rebuilds from scratch — an A/B-safe approach because the leaves are never touched.

The CLI command lives under `sciknow catalog raptor` (a Typer sub-namespace):

```bash
sciknow catalog raptor build              # first build
sciknow catalog raptor build --dry-run    # preview cluster sizes without writing
sciknow catalog raptor build --rebuild    # wipe all level >= 1 nodes and rebuild
sciknow catalog raptor build --max-levels 3 --min-cluster-size 4
sciknow catalog raptor build --model qwen3.5:27b   # main model for higher-quality summaries
sciknow catalog raptor stats              # show node counts per level
```

### Expected impact

The writer gains access to mid-level abstractions ("CMIP6 generally underestimates Arctic amplification", "ENSO–monsoon teleconnections weakened in the satellite era") that previously had to be reconstructed inside the prompt by gluing 8+ chunks together. For chapter intros and cross-paper synthesis paragraphs, this is the difference between a list of disjoint single-paper findings and proper synthesised prose. RAPTOR's published gains on multi-document QA are 5–20% absolute on cross-document tasks; for sciknow's book-writing use case the practical impact is mostly on completeness and coherence (the chapter "feels" connected) rather than groundedness.

### Caveats

- GMM is stochastic; the build uses a fixed `random_state=42` for reproducibility. Re-runs with the same corpus produce the same tree.
- BIC-selected k can be high on small clusters, leading to many tiny groups; `--min-cluster-size` filters them out.
- Higher levels can be sparse — the build stops early when fewer than `min_top_level` (default 4) summaries remain. On a small corpus you may only get level 1.
- `_hydrate`'s RAPTOR branch returns no `authors` or `journal`, so prompts that depend on author-style citations should fall back gracefully (the existing APA formatter does — it skips empty fields).

**Research basis:**
- [Sarthi et al. 2024 — *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* (arXiv:2401.18059)](https://arxiv.org/abs/2401.18059) — ICLR 2024
- [Reference implementation (parthsarthi03/raptor)](https://github.com/parthsarthi03/raptor) — sciknow's GMM+BIC clustering follows this implementation closely

---

## 20. Measurement & Observability (Track A)

**Status:** Production

After shipping six new techniques in Phases 7–12, there was no way to objectively know whether any of them were actually moving the needle on a real corpus. Track A closes that gap with three commands and a persistence change — enough to make the next round of feature decisions data-driven instead of vibes-driven.

### Per-iteration score history persistence

`autowrite_section_stream` now records a structured per-iteration history dict capturing:

- All six scoring dimensions (groundedness, completeness, coherence, citation_accuracy, hedging_fidelity, overall)
- The verification flag counts (`n_supported`, `n_extrapolated`, `n_overstated`, `n_misrepresented`) and the verifier's `groundedness_score` / `hedging_fidelity_score`
- The Chain-of-Verification result if it ran (`cove_score`, `n_high_severity`, `n_medium_severity`, `questions_asked`)
- The revision verdict (`KEEP` / `DISCARD`) and the post-revision overall score
- A `feature_versions` dict identifying which Phase 7–12 features were active at write time

This is persisted to the existing `drafts.custom_metadata` JSONB column — **no schema migration required**, the column already had `server_default="{}"`. The persistence happens in `_save_draft` via a new `custom_metadata` kwarg.

### `sciknow book draft scores <draft_id>`

Reads back the persisted history and prints a per-iteration Rich table with all six dimensions, verification flag counts, CoVe results, and the keep/discard verdict. Shows a "Δ first → last" line so you can see at a glance which dimensions improved across the convergence loop.

### `sciknow book draft compare <draft_id_a> <draft_id_b>`

Compares two drafts side-by-side. Default mode reads the persisted final scores from `custom_metadata` (instant, no LLM calls). With `--rescore`, re-runs the scorer + verifier against fresh retrieval for both drafts so the comparison reflects the **current** rubric — useful for comparing pre-Phase-7 drafts against new ones on the same modern scoring criteria. The output is a side-by-side delta table with feature-version differences listed at the bottom.

### `sciknow book autowrite-bench`

Runs `autowrite_section_stream` N times under identical conditions on the same chapter and section, captures every run's history to `data/bench/<timestamp>/run_NN.json`, and prints a summary table with mean ± std for each scoring dimension across runs.

This is the variance-measurement tool: if `std` is high relative to the differences you're measuring in `book draft compare`, the loop is too noisy for confident A/B and you should bump `--max-iter` or accept the noise as the floor. If `std` is low, the loop is well-calibrated and small score deltas can be trusted.

### Web reader

`sciknow/web/app.py` now displays `hedging_fidelity_score` alongside `groundedness_score` in the verification panel, renders `OVERSTATED` claims in orange (distinct from yellow `EXTRAPOLATED` and red `MISREPRESENTED`), surfaces `cove_verification` events from the autowrite SSE stream as a CoVe summary line, and shows the new `hedging_fidelity` dimension in the convergence chart.

### Why this is in the research doc

Measurement isn't a research technique per se, but it's the lever that makes every future research decision sound. After this lands, "should we ship CARS chapter moves next?" stops being a guess and becomes "let's run `autowrite-bench` with and without CARS on a control chapter and see if the `coherence` mean shifts more than the variance."

---

## Planned (Researched, Implementation Pending)

The roadmap items from the 2026-04 lit sweep are now all shipped (Phases 7–12). Track A measurement landed in Phase 13. Future research notes will accumulate here as they're identified.

Likely next-up candidates (in priority order, from the original lit sweep's runners-up):

1. **CARS-adapted chapter moves** (Swales 1990 + Yang & Allison 2003) — 5-move scaffold (Orient → Tension → Evidence → Qualify → Integrate). Cost: prompt + ~20 lines. Linguistics runner-up #1.
2. **LongCite-style sentence citations** (THUDM 2024) — sentence-level grounding with span match for ALCE-compatible `citation_f1`. CS runner-up.
3. **Toulmin scaffolds** for paragraphs the planner labels `Tension` (claim/data/warrant/qualifier/rebuttal). Linguistics runner-up #2.
4. **MADAM-RAG** for paragraphs the argument-mapper flags as contradiction-heavy (Wang+ COLM 2025). CS runner-up.
5. **Soft RAPTOR clustering** — use the GMM `proba` matrix the build already computes to allow chunks to contribute to multiple cluster summaries above a probability threshold. Polish on Phase 12.

---

## Considered and Rejected

These techniques were researched in the same sweep that produced sections 13–16 and the planned items above. Each is documented here with the reason it was *not* adopted, so future sessions don't relitigate the same decisions.

### HyDE (Hypothetical Document Embeddings)
[Gao et al. 2022 (arXiv:2212.10496)](https://arxiv.org/abs/2212.10496) — generate a fake answer to the query, embed it, retrieve.

**Rejected:** still alive on pure dense retrievers, but on sciknow's hybrid dense + sparse + Postgres FTS stack with a strong cross-encoder reranker, the hypothetical doc usually introduces more noise than signal for scientific queries. Step-back prompting (Section 16) dominates it on the same use case.

### Self-RAG and CRAG (Corrective RAG)
[Asai et al. 2024 (arXiv:2310.11511)](https://arxiv.org/abs/2310.11511) and [Yan et al. 2024 (arXiv:2401.15884)](https://arxiv.org/abs/2401.15884) — adaptive retrieval with reflection / critic models.

**Rejected as published:** both rely on *fine-tuned* reflection models. Self-RAG ships a fine-tuned Llama-2 7B/13B with reflection tokens; CRAG needs a 0.77B trained evaluator. Prompt-only emulations on local Ollama models lose most of the gains. sciknow's existing claim-verification loop (Section 6) already captures the "re-retrieve when confidence is low" spirit without needing a fine-tuned critic.

### Dense X / Propositional Retrieval
[Chen et al. 2023 (arXiv:2312.06648)](https://arxiv.org/abs/2312.06648) — index atomic propositions instead of passages.

**Rejected:** the gains mostly disappear on supervised retrievers fine-tuned on passage-query pairs (bge-m3 is one), and the benefits favour unpopular-entity queries rather than scientific synthesis. The reindex cost is high and the expected lift on this stack is small.

### GraphRAG Global Search
[Edge et al., Microsoft 2024 (arXiv:2404.16130)](https://arxiv.org/abs/2404.16130) — Leiden-clustered community reports for corpus-wide synthesis.

**Rejected for now:** conceptually right for "what does the corpus as a whole say about X" questions, but building Leiden-clustered community reports over the existing KG triples is a genuine new pipeline stage (new code, new tables, large indexing cost). RAPTOR (planned, P2) gets ~80% of the same benefit at ~20% of the cost on sciknow's existing chunk index. GraphRAG global search becomes interesting only if RAPTOR underperforms on corpus-wide synthesis queries.

### Late Chunking (Jina)
[Günther et al. 2024 (arXiv:2409.04701)](https://arxiv.org/abs/2409.04701) — feed full sections through the embedder once, then mean-pool token spans per chunk so each chunk embedding is conditioned on its surrounding context.

**Rejected:** modest gains that only show up when chunking destroys cross-sentence references, which sciknow's section-aware chunker already mitigates. bge-m3 supports the technique mechanically (8K context, mean pooling), but the implementation requires bypassing `FlagEmbedding`'s high-level API. Cost-benefit is poor relative to RAPTOR.

### Full RST Tree Parsing
[Mann & Thompson 1988](https://www.sfu.ca/rst/pdfs/Mann_Thompson_1988.pdf) — full Rhetorical Structure Theory with ~30 relation types and recursive nucleus/satellite trees.

**Rejected in favour of PDTB-lite (Section 15):** SOTA neural RST parsers are ~60% F1 on relation labelling. Asking an LLM to produce RST trees over its own long drafts is cost for noise. The shallow PDTB-lite enum captures the practical rhetorical-texture wins at 1% of the operational complexity.

### Centering Theory's Full Cb/Cf/Cp Machinery
[Grosz, Joshi & Weinstein 1995](https://aclanthology.org/J95-2003/) — backward-looking center, forward-looking centers, preferred center, full transition classification.

**Rejected as a runtime checker:** would require a coreference resolver, which sciknow explicitly avoids. The CONTINUE-transition *intuition* (Section 14, the entity-bridge rule) captures ~90% of the practical value at zero infrastructure cost.

### FActScore as an Online Method
[Min et al. 2023 (arXiv:2305.14251)](https://arxiv.org/abs/2305.14251) — atomic-fact decomposition + per-fact verification.

**Rejected for online use:** superseded for online use by sentence-level citation verification (LongCite-style, see runners-up in the research notes). FActScore remains useful as an *offline* eval harness if a sanity check is needed across prompt variants.

### ALCE Benchmark
[Gao et al. 2023 (arXiv:2305.14627)](https://arxiv.org/abs/2305.14627).

**Not a technique:** ALCE is a benchmark on Wikipedia QA, not a method to implement. Useful as inspiration for citation-quality metrics (now partially covered by the new `hedging_fidelity_score` and the planned LongCite-style sentence citations) but not something to "adopt".

---

## Implementation Timeline

```
Phase 1:  BERTopic clustering       → replaced LLM batching with embedding-based clustering
Phase 2:  Knowledge Graph           → entity-relationship triples during wiki compile
Phase 3:  Self-correcting RAG       → retrieval evaluation + grounding checks
Phase 4:  Multimodal RAG            → table/equation tagging for filtered retrieval
Phase 5:  TreeWriter planning       → hierarchical paragraph plans before drafting
Phase 6:  Consensus mapping         → agreement/disagreement across corpus
Phase 7:  Hedging fidelity          → BioScope cue list + OVERSTATED verdict + scoring dim
Phase 8:  Centering entity bridge   → no-cold-start rule for paragraph openings
Phase 9:  PDTB-lite discourse plan  → discourse_relation per paragraph in tree plan
Phase 10: Step-back retrieval       → abstract reformulation augments concrete query
Phase 11: Chain-of-Verification     → decoupled fact-check questions, gated on score thresholds
Phase 12: RAPTOR hierarchical tree  → UMAP+GMM clustering, LLM cluster summaries as level-N nodes
Phase 13: Track A measurement       → score history persistence, draft scores/compare/autowrite-bench
Phase 14: Web reader v2             → modern UI, score history viewer, wiki/ask/catalog modals, stats dashboard
```
