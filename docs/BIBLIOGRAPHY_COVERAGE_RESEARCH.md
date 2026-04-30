# Bibliography Coverage in Retrieval-Augmented Book Writing — Research Survey

**Status:** research notes / standalone reading. Not a phase plan.
**Companion:** `docs/PHASE_56_ROADMAP.md` (the implementation track these techniques would slot into).
**Anchor commit:** `342053d` (Phase 55.V19h GUI fix; bibliography measurements taken on this commit).

---

## 1. The problem, measured

For the global-cooling book (commit 342053d, 2026-04-30):

| Stage | Count | % of corpus |
|---|---:|---:|
| Documents in corpus | 849 | 100% |
| Documents with at least one chunk indexed | 849 | 100% |
| Total chunks indexed | 32,535 | — |
| Distinct papers cited at least once across all active drafts | 336 | 39.6% |
| Entries in the deduped book bibliography (`refs.bib`) | 246 | 29.0% |
| Entries visible in the user's earlier broken-biber export | 140 | 16.5% |

**The structural gap is 246 / 849 ≈ 29% corpus utilization.** Roughly seven out of every ten documents you have ingested never reach the writer.

### Why so low? — the funnel

Each section runs **one** retrieval against a coarse query string `f"{section_type} {topic}"`. The default top-K is ~10 (hybrid dense+sparse, then cross-encoder rerank). For ~50 planned sections × top-K 10 ≈ 500 retrieval slots. After dedup across overlapping topics (a Maunder paper hits Sumerian + Classical + Medieval + LIA + Modern sections), unique papers collapse to ~336. Bibliography normalization (drop unresolvable, merge near-duplicates by DOI) drops it to 246.

Two structural facts dominate the funnel:

1. **One query per section.** No matter how much the section will discuss, only ~10 documents are eligible to be cited. A section about "the Maunder Minimum" with 8 distinct sub-topics still pulls 10 chunks total.
2. **Coarse query.** `"the_great_climate_debate Global Cooling"` is a topic anchor, not a fact-specific question. Hybrid search returns broadly-on-topic chunks, not chunks that target specific claims the section will make.

The remaining ~600 corpus papers are not "irrelevant" — many are tangential to the chapter's framing but very relevant to specific claims. They never make it past top-K because no single section's retrieval surfaces them.

---

## 2. The six families of techniques

Published literature on RAG coverage / diversity / iterative retrieval. For each family: what it does, what it gives you, what it costs, how it maps to sciknow.

### 2.1 Query expansion

Generate multiple query *variants* for the same retrieval intent, retrieve for each, fuse the results.

**HyDE — Hypothetical Document Embeddings** (Gao et al., 2022)
The LLM hallucinates a *plausible answer* to the query, embeds *that*, retrieves against it. The hallucinated answer's surface form matches what real corpus documents would say, so embedding similarity is much higher than for the bare query. Especially powerful when the user query is short, ambiguous, or in a different register from corpus documents.

**RAG-Fusion** (Raudaschl, 2023)
Generate 3–5 paraphrases of the query (different framings, different specificity), retrieve top-K for each, combine via **Reciprocal Rank Fusion**:
```
RRF(d) = Σᵢ 1 / (k + rank_i(d))
```
where `rank_i(d)` is `d`'s rank in the i-th query's results. RRF is parameter-light and well-behaved; the typical `k=60` works across domains.

**DMQR-RAG — Diverse Multi-Query Rewriting** (Yang et al., 2024, arXiv 2411.13154)
Same as RAG-Fusion but explicitly trains/prompts the rewriter to produce *semantically distant* paraphrases (different facets of the topic, not different wordings of the same facet). Reports +14.46 % P@5 lift over single-query on FreshQA.

**MQRF-RAG / RFG** (2025)
Multi-Query Rewriting *with feedback*: an initial retrieval grounds the LLM-generated paraphrases, so the paraphrases stay close to what the corpus actually contains. Beats HyDE on AmbigQA by 1.47 % EM, 3.75 % F1.

**Cost:** ~3–5× retrieval cost per section. Each retrieval is fast (50–200 ms hybrid + rerank), so the absolute wall-time hit is small.

**Mapping to sciknow:** Lowest-friction win. The retrieval substrate is already there; the only new code is a query-rewriter LLM call (cheap — 100–200 tokens) and an RRF fuser (a 20-line function). Where each section currently sees 10 chunks, it would see 30–50 unique chunks across its variants, so the writer's *eligibility set* is 3–5× larger before any other change.

### 2.2 Diversity-aware reranking

Given a candidate pool, pick the top-K that are *both* relevant and *different from each other*.

**MMR — Maximal Marginal Relevance** (Carbonell & Goldstein, 1998)
The canonical algorithm. Greedy:
```
score(d) = λ·sim(query, d) − (1−λ)·max_{s∈selected} sim(d, s)
```
- `λ = 1.0` → pure relevance (no diversity).
- `λ = 0.0` → pure diversity (ignore relevance).
- `λ ≈ 0.5–0.6` → standard sweet spot for retrieval reranking.

For book-length writing where the same paper retrieved across 8 sections wastes 7 retrieval slots, MMR is the cheapest fix. Implemented as a 30-line greedy loop over top-N candidates.

**Cluster-then-pick.** Run k-means or HDBSCAN on the top-N pool's embeddings, take top-1 per cluster. Forces topical spread. Faster than MMR for large pools, less smooth for small ones.

**Information-gain MMR.** Replace cosine-distance with mutual-information / KL-divergence between candidate and selected. More principled, more compute.

**Cost:** Negligible — added to the existing rerank step.

**Mapping to sciknow:** Drop-in for `retrieval/hybrid_search.py` after the bge-reranker step. Should ship as a single PR.

### 2.3 Iterative retrieval

Don't make all retrievals up-front. Generate, then retrieve more based on what was generated.

**ITER-RETGEN** (Shao et al., EMNLP 2023)
generate → use generation as next query → retrieve → regenerate. Loop 2–3 times. Each iteration brings in fresh sources because the generation surfaces new sub-topics that the original query didn't anticipate.

**IRCoT — Iterative Retrieval with Chain-of-Thought** (Trivedi et al., ACL 2023)
Every CoT sentence triggers its own retrieval. Fine-grained but expensive (10–20× retrievals per answer for multi-step questions).

**FLARE — Forward-Looking Active Retrieval** (Jiang et al., EMNLP 2023)
The writer streams. When token-confidence dips below a threshold (e.g., the next-token entropy is high, or a low-probability span is being emitted), pause, re-retrieve based on the about-to-be-generated sentence, resume with the new evidence injected. Closest analogue to what Phase 56 envisions for sciknow.

**Self-RAG** (Asai et al., ICLR 2024)
Trains the LLM to emit special `[Retrieve]` tokens autonomously when it needs more evidence. Unrealistic for sciknow because it requires fine-tuning, but conceptually clean.

**FAIR-RAG / Auto-RAG / Stop-RAG** (2024–2025)
Adaptive iterative refinement with learned stop conditions. More engineering than scientific novelty over the EMNLP 2023 originals.

**Cost:** 1.5–3× wall-time per section. Each iteration is a full retrieve+generate cycle; you're trading throughput for coverage.

**Mapping to sciknow:** ITER-RETGEN is the cheapest entry point. After the writer produces an initial draft, identify topics it mentioned but didn't cite; re-retrieve for each; revise. FLARE is the gold standard but requires intercepting the streaming generation, which the current `_stream_with_save` infrastructure can support but isn't built for.

### 2.4 Claim decomposition (architectural, biggest structural win)

Decompose the writing target into *atomic claims* and retrieve per-claim.

**FActScore** (Min et al., EMNLP 2023)
Originally an *evaluation* method: decompose a generation into atomic facts, evaluate each independently. Recent work applies it as a *generation* strategy.

**Proposition chunking** (Chen et al., 2024)
Index atomic propositions instead of chunks. Each retrieval surfaces a single self-contained claim, rather than a 1500-char paragraph that mixes multiple claims. Reported +20–35 % retrieval precision.

**VeriScore / VERIFASTSCORE** (Song et al., 2024 / EMNLP 2025)
Faster atomic-fact verification via batched NLI. Same atomic-decomposition primitive but engineered for production throughput.

**Plan-then-cite for long-form** (Sun et al., 2024)
Generate the plan as a list of atomic claims. Retrieve per claim. Generate per claim. Stitch. **This is the Phase 56 design.**

**Cost:** Architectural. ~5–10× more retrievals (one per claim, not one per section), but each retrieval can be smaller (top-3 instead of top-10) because the claim is specific.

**Mapping to sciknow:** Phase 56.3 is exactly this. The substrate (claim_atomize.py) already exists for the *post-hoc* verification path. Phase 56 promotes it from evaluator to planner.

### 2.5 Hierarchical retrieval

Index the corpus at multiple levels of abstraction and retrieve at all of them.

**RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval** (Sarthi et al., ICLR 2024)
Recursively cluster and summarize chunks bottom-up:
- Level 0: individual chunks (the leaves)
- Level 1: cluster summaries (grouped by embedding similarity, ~10–30 chunks per cluster)
- Level 2: super-cluster summaries (groups of level-1 nodes)
- Level 3+: continue until a single root or until clusters are too small

At retrieval time, query against all levels; take top-K across the union. The summaries pull in chunks the leaf-only search would miss because they capture themes that any single chunk doesn't fully express.

**GraphRAG** (Edge et al., 2024)
Build an entity / relation graph over the corpus, retrieve via graph community detection. Stronger for multi-hop reasoning; orthogonal to RAPTOR; heavier engineering footprint.

**Cost:** Build-time only; retrieval-time cost is similar to standard top-K because the union of "leaf top-K" + "summary top-K" is ranked together.

**Mapping to sciknow:** sciknow already has RAPTOR (per `docs/RESEARCH.md`). The *autowrite engine* doesn't currently retrieve from the RAPTOR collection by default — flipping the writer's retrieval to use leaf-∪-summary surfaces topical breadth that's already been computed. Free win once wired.

### 2.6 Coverage telemetry & feedback loops

Don't change how retrieval works — *measure* what's being missed and feed it back.

**RaPID — Retrieval-Augmented Long Text Generation with Writing Planning and Information Discovery** (March 2025, arXiv 2503.00751)
The most directly relevant 2025 paper for sciknow's use case. Their key observation: standard RAG doesn't *track* which corpus regions are unused, so coverage remains accidental.

Their fix: a coverage-tracking module that, after each section is written, embeds the section content and finds corpus chunks whose nearest neighbors among the *cited* chunks are weak. These are surfaced as candidate sources for the next section, or for a "missing topics" report.

**Underutilization metrics.** For each corpus document, compute: was any chunk of this doc cited by any section? If no, the doc is "unused." Cluster the unused docs by topic; surface the largest unused clusters as gap candidates.

**Cost:** Pure measurement layer. No retrieval changes.

**Mapping to sciknow:** Pair with the existing `book gaps` engine. Currently `book gaps` flags "missing claims"; this would extend it with "papers we have but never cited."

---

## 3. Stacked impact estimate

How each technique stacks on top of the others, expressed as expected unique-papers-cited (out of the 849 corpus). These are *estimates* informed by the literature, not measurements yet.

| Tier | Change | Coverage est. | % corpus | Cost (eng days) |
|---|---|---:|---:|---:|
| **0 (today)** | Status quo: 1 retrieval per section, top-10, no diversity, no expansion | 246 | 29 % | — |
| **0.5** | + MMR rerank (λ=0.55) on the top-30 candidate pool | 320–360 | 38–42 % | 1 |
| **1** | + Multi-query expansion (5 paraphrases via DMQR + RRF) | 480–560 | 57–66 % | 3 |
| **1+RAPTOR** | + Use leaf ∪ summary RAPTOR retrieval at writer | 540–620 | 64–73 % | 1 |
| **2** | + HyDE on coarse-query sections (cheap detector for which sections need it) | 580–650 | 68–77 % | 2 |
| **3 (Phase 56.3)** | + Per-claim atomic retrieval, top-3 per claim | 720–810 | 85–95 % | 5 |
| **4** | + FLARE-style mid-stream re-retrieval | 770–820 | 91–97 % | 8 |
| **5** | + RaPID-style coverage telemetry feeding `book gaps` | doesn't raise per-section; raises *book-level* coverage by spawning sections where the corpus has unused depth | — | 5 |

**Reading the table.** Stacking 0.5 + 1 + RAPTOR + 2 alone (≈ 7 days work) gets you to ~70 % coverage from 29 %. That's the cheapest "good enough" tier — well over double today's coverage with no architectural change.

The big jump is **Tier 3 (per-claim retrieval)**. That's where the structural ceiling moves: instead of being capped by `sections × K`, you become capped by `claims × K_per_claim`. With 4–6 claims per section × top-3 each = 12–18 retrieval slots per section, you have 2× more eligibility per section *and* each retrieval is targeted at a specific claim, so dedup across sections is much weaker.

Tiers 4 and 5 are diminishing returns in terms of coverage but help in other ways:
- Tier 4 (FLARE) raises *citation precision* alongside coverage, because mid-stream retrieval grounds each emerging sentence.
- Tier 5 (telemetry) shifts the question from "did we cite well?" to "what does the corpus cover that the book doesn't?" — informs future planning rounds.

---

## 4. Recommended order

1. **Tier 0.5 — MMR rerank.** Single-day patch in `retrieval/hybrid_search.py`. Lowest risk, immediate ~30 % coverage lift. Ship behind `MMR_LAMBDA=0.55` env var so it can be regression-tested.
2. **Tier 1+RAPTOR — Multi-query expansion + RAPTOR-aware retrieval.** ~4 days combined. Both are flagged. Together: another ~2× on coverage.
3. **Tier 2 — HyDE on coarse queries.** ~2 days. Detect "coarse" via section-plan length; HyDE only when bullets-per-plan is < 4 (specific plans don't benefit per the literature).
4. **Phase 56 ships.** Promote the per-claim retrieval phase as the next priority. Tiers 0.5 + 1 + 2 stack on top of it.
5. **Tier 4 — FLARE.** Defer until after Phase 56.3 is live and we have a coverage measurement baseline. Adds wall-time cost; needs justification from a metric.
6. **Tier 5 — RaPID-style coverage telemetry.** Backburner; pair with the next `book gaps` extension.

---

## 5. Cross-cutting concerns

### 5.1 Hedging fidelity vs. coverage

Today's drafts have `hedging_fidelity` as the bottom-ranked dimension (per the autowrite scorer logs). Wider retrieval can make this *worse* — pulling in 5× more chunks gives the writer 5× more material to over-claim or under-claim from. The Phase 56 design routes this by attaching a per-claim `hedge_strength` derived from the supporting chunks, so hedging becomes deterministic at write time.

If you ship Tiers 0.5–2 *without* Phase 56, watch the hedging score. If it regresses, add a "claim hedge floor" check before bib gets normalized.

### 5.2 Retrieval cost

Tiers 1 + 2 multiply retrieval calls per section by 5–7×. Each retrieval is fast (~150 ms total: hybrid search + rerank), so per-section retrieval phase grows from ~200 ms to ~1.2 s. Not noticeable inside a section that takes 8–15 minutes to write. Only worth measuring when iterative-retrieval (FLARE) lands, where retrieval might fire 30–50× per section.

### 5.3 Wall-time budget

| Engine | Today | Tier 0.5+1+2 | + Phase 56.3 | + FLARE |
|---|---:|---:|---:|---:|
| Wall time per section | ~8–15 min | ~8–15 min | ~12–22 min | ~18–35 min |

The 1.5–2× ceiling on Phase 56 wall-time still holds with the additional tiers because retrieval is not the bottleneck — the writer's decode is. Adding more retrievals barely moves the needle until you start re-writing (FLARE).

### 5.4 Evaluation

Before shipping any tier, freeze a baseline:
- Pick 5 sections from the global-cooling book.
- Snapshot their current cited-paper set (`drafts.sources`).
- Run each tier in isolation; record new cited-paper set.
- Diff: how many genuinely new papers? How many that the writer should have had but didn't?

Then ship the tier with the best ratio of "added relevant" to "added irrelevant." Without this measurement you'll drift toward "more retrievals = more chaos" rather than "more retrievals = more grounding."

`citation_precision` (mean NLI entailment of cited chunk vs. cited sentence) is the right top-line metric. It must hold or improve as coverage grows; if it drops, the new retrievals are noise, not signal.

---

## 6. Out of scope (consciously)

- **Re-ingesting corpus.** None of the techniques above need it. They all change how *existing* chunks are retrieved.
- **Larger / smaller embedding model.** Cross-cutting concern; unrelated to coverage strategy. Stay on bge-m3 unless an unrelated benchmark says otherwise.
- **GraphRAG.** Tempting but heavy. RAPTOR + claim-decomposition cover most of what GraphRAG offers for a science-book corpus. Revisit only if multi-hop reasoning becomes a bottleneck.
- **Long-context models.** Throwing a 200 K-context model at the problem doesn't help: the bottleneck isn't context, it's that the writer never *sees* most of the corpus to begin with. Coverage is a retrieval problem, not a context problem.

---

## 7. References

### Foundational
- Carbonell, J. & Goldstein, J. (1998). *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.* SIGIR. [PDF](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)
- Carbonell & Goldstein, MMR — modern production usage. [OpenSearch docs](https://docs.opensearch.org/latest/vector-search/specialized-operations/vector-search-mmr/)

### Query expansion
- Gao, L. et al. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE).*
- Yang, Z. et al. (2024). *DMQR-RAG: Diverse Multi-Query Rewriting for RAG.* [arXiv 2411.13154](https://arxiv.org/html/2411.13154v1)
- (2025). *Optimization of RAG Multi-Query Rewrite (MQRF-RAG).* [ACM 3728199.3728221](https://dl.acm.org/doi/10.1145/3728199.3728221)
- Raudaschl, A. (2023). *Forget RAG, the future is RAG-Fusion.*

### Iterative retrieval
- Shao, Z. et al. (EMNLP 2023). *ITER-RETGEN: Enhancing Retrieval-Augmented LLMs with Iterative Retrieval-Generation Synergy.*
- Trivedi, H. et al. (ACL 2023). *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT).*
- Jiang, Z. et al. (EMNLP 2023). *FLARE: Forward-Looking Active REtrieval.*
- Asai, A. et al. (ICLR 2024). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.*
- (2024). *Auto-RAG: Autonomous Retrieval-Augmented Generation.* [arXiv 2411.19443](https://arxiv.org/html/2411.19443v1)
- (2025). *FAIR-RAG: Faithful Adaptive Iterative Refinement.* [arXiv 2510.22344](https://arxiv.org/html/2510.22344v1)

### Claim decomposition
- Min, S. et al. (EMNLP 2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* [Meta AI](https://ai.meta.com/research/publications/factscore-fine-grained-atomic-evaluation-of-factual-precision-in-long-form-text-generation/)
- Chen, T. et al. (2024). *Dense X Retrieval — proposition chunking.*
- Song, Y. et al. (2024). *VeriScore: Evaluating the factuality of verifiable claims in long-form text generation.*
- (EMNLP 2025 findings). *VERIFASTSCORE.* [PDF](https://aclanthology.org/2025.findings-emnlp.491.pdf)

### Hierarchical retrieval
- Sarthi, P. et al. (ICLR 2024). *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.* [arXiv 2401.18059](https://arxiv.org/abs/2401.18059)
- Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.*

### Long-form book writing
- (March 2025). *RaPID: Retrieval-Augmented Long Text Generation with Writing Planning and Information Discovery.* [arXiv 2503.00751](https://arxiv.org/html/2503.00751v1)

### Surveys
- Gao, Y. et al. (2024). *Retrieval-Augmented Generation for LLMs: A Survey.* [arXiv 2312.10997](https://arxiv.org/html/2312.10997v5)
- (2025). *A Systematic Review of Key Retrieval-Augmented Generation Systems.* [arXiv 2507.18910](https://arxiv.org/html/2507.18910v1)

### LLM length-control side discussion (relevant to the abrupt-ending bug)
- *Prompt-Based One-Shot Exact Length-Controlled Generation.* [arXiv 2508.13805](https://arxiv.org/html/2508.13805v1)
- *How LLMs Know When to Stop Talking.* [Bouchard](https://www.louisbouchard.ai/how-llms-know-when-to-stop/)
- *My LLM Can't Stop Generating, How to Fix It?* [Kaitchup](https://kaitchup.substack.com/p/my-llm-cant-stop-generating-how-to)
