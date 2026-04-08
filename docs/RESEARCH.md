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

## 13. Four-Layer Metadata Extraction

**Status:** Production

Each layer only runs if the previous didn't fully populate the metadata:

1. **PyMuPDF** — embedded XMP/Info fields (fast, often incomplete)
2. **Crossref API** — authoritative bibliographic data by DOI
3. **arXiv API** — for preprints
4. **Ollama LLM** — structured extraction from first ~3000 chars (fallback)

This cascade ensures maximum metadata coverage regardless of PDF quality.

---

## Implementation Timeline

All innovations are complete and in production:

```
Phase 1: BERTopic clustering       → replaced LLM batching with embedding-based clustering
Phase 2: Knowledge Graph           → entity-relationship triples during wiki compile
Phase 3: Self-correcting RAG       → retrieval evaluation + grounding checks
Phase 4: Multimodal RAG            → table/equation tagging for filtered retrieval
Phase 5: TreeWriter planning       → hierarchical paragraph plans before drafting
Phase 6: Consensus mapping         → agreement/disagreement across corpus
```
