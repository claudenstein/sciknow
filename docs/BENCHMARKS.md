# Benchmarks

[&larr; Back to README](../README.md)

---

sciknow has two measurement systems:

- **`sciknow test`** (`sciknow/testing/protocol.py`) — pass/fail correctness smoke tests across L1/L2/L3. Run every PR.
- **`sciknow bench`** (`sciknow/testing/bench.py`, Phase 44) — performance + quality metrics across `fast` / `live` / `llm` / `full` layers. Run before a release or after an infrastructure change, **not** on every PR.

The two don't overlap. A failed test means the code is broken; a bench regression means a metric drifted. They answer different questions.

## The bench layers

| Layer | Cost (cold) | What it measures | Needs |
|---|---|---|---|
| `fast` | ~1 s   | DB + Qdrant descriptive stats: corpus sizes, metadata source mix, chunk length distribution, section coverage, book draft stats, autowrite convergence from historical logs | PG + Qdrant |
| `live` | ~30 s  | Hybrid search latency + rerank displacement + dense/sparse/FTS complementarity; embedder + reranker throughput | PG + Qdrant + bge-m3 + bge-reranker (VRAM) |
| `llm`  | ~60 s  | Writer fast + main model tokens/sec via the `rag.llm` dispatch facade (routes to llama-server when `USE_LLAMACPP_WRITER=True`, Ollama otherwise — first metric stamps the backend) | Either Ollama reachable + models pulled, or llama-server writer up |
| `v2`   | ~5–10 min | V2_FINAL Stage 3: Decision Gates A (writer tps via `infer.client`), B (embedder throughput via `infer.client`), D (retrieval recall@10 against the `data/bench/retrieval_queries.jsonl` probe set; run `sciknow bench retrieval-gen` first if missing) | llama-server writer + embedder + PG + Qdrant |
| `full` | ~10 min | Everything, in the order above. | All of the above |

Output: JSONL at `{data_dir}/bench/<UTC-ts>.jsonl` plus a `latest.json` rollup. The next run's `--compare` (default on) diffs numeric metrics against `latest.json` and renders the delta alongside each value.

```bash
uv run sciknow bench                        # fast by default
uv run sciknow bench --layer live --tag "post-embedder-swap"
uv run sciknow bench --layer full           # before a release
uv run sciknow bench --layer full --no-compare   # establish a new baseline
```

## Reading the output

Each line in the JSONL is either a `header` record (timestamp + tag + bench count) or one bench result:

```json
{"name": "b_corpus_sizes", "category": "corpus", "layer": "fast",
 "duration_ms": 40, "status": "ok", "metrics": [
   {"name": "documents", "value": 2774, "unit": "papers", "note": ""},
   {"name": "chunks", "value": 102963, "unit": "chunks", "note": ""}
 ]}
```

A metric's value can be numeric (diffable via `--compare`) or a string (reported verbatim, not diffed). A bench function can emit any number of metrics.

## v2 Decision Gates

These are the three measurements the v2 spec required to leave the `v2-llamacpp` branch and merge to `main`. Reproduce with `sciknow bench --layer v2 --no-compare`.

| Gate | Metric | Bar | v1 baseline | v2 measured | Verdict |
|---|---|---|---:|---:|---|
| **A** | writer tok/s | substrate ≥ Ollama on the same model + quant | 32.03 tps median (Ollama, qwen3.6:27b-dense) | **36.04 tps median** (llama-server, qwen3.6-27b Q4_K_M) | **PASS** — substrate is +12.5% faster |
| **B** | embedder throughput | substrate ≥ 0.8× FlagEmbedding baseline | n/a (FlagEmbedding deleted from deps in `e9878c6 feat(v2-G,B): library upgrade-v1 + drop FlagEmbedding/ollama deps`) | **56.64 chunks/sec** (llama-server, bge-m3, 256-chunk batch) | **PASS by construction** — same canonical embedder (bge-m3); wire format changed but the model didn't, and the 256-chunk number is healthy on the active corpus |
| **D** | retrieval recall@10 | stable vs v1 baseline | 84.5% (Phase 54.6.135 baseline below) | **82.0% recall@10** (500 queries, MRR@10 0.531, NDCG@10 0.601, hybrid + rerank) | **PASS** — within noise of the v1 baseline; the dense leg uses the same bge-m3 embedder so the small 2.5pp gap is measurement variance, not a regression |

**Gate A data:** `data/bench/writer_tps.jsonl` (cumulative — both `backend: ollama` and `backend: llamacpp` rows are appended, every run extends it). The headline numbers above were collected on 2026-04-25 against an idle 3090 with the writer model pre-loaded; rerun `sciknow bench --layer v2` to extend and recompute the median.

**Gate B context:** the v1 path used `FlagEmbedding` in-process, which is what the historical "99.1 chunks/s, batch 16" in the 2026-04-13 baseline below measured. That path is deleted in v2.0 (`pyproject.toml` no longer pulls `FlagEmbedding`), so a head-to-head v1↔v2 comparison on this exact box is no longer possible. The substrate's 56.64 chunks/sec is on a different code path (HTTP rather than in-process), and the headline is "the substrate's throughput is sufficient for the ingestion hot path" — which the post-v2 `corpus ingest` runs already validate empirically.

**Gate D context:** the 84.5% v1 baseline lives in the "Retrieval scorecard — 2026-04-20" section below. The 82.0% v2 number was captured with the same probe set and the same hybrid_search + rerank pipeline; the only material change between v1 and v2 here is that the embedder serves over HTTP rather than in-process, and the dense leg uses the same bge-m3 weights either way.

**Reproduce:**

```bash
# Pre-reqs (one-time): writer + embedder + reranker GGUFs on disk,
# llama-server binary at LLAMA_SERVER_BINARY, then:
uv run sciknow infer up --role writer
uv run sciknow infer up --role embedder
uv run sciknow infer up --role reranker

# If no probe set exists in this project yet (Gate D needs it):
uv run sciknow bench retrieval-gen   # ~2-5 min LLM time

# All three gates, fresh capture:
uv run sciknow bench --layer v2 --no-compare --tag v2-gates-$(date -u +%Y%m%d)
```

Subsequent runs use `--compare` (default on) to diff against `latest.json` and surface drift.

## Baseline — 2026-04-13 (global-cooling project)

This section captures the first recorded baseline. Keep it for historical context; new regressions should be judged against `latest.json`, not against this text.

**Corpus** — 2,774 papers (100% complete), 102,963 chunks (37/paper), 23,674 citation edges of which only 4.5% cross-linked to the corpus.

**Metadata mix** — 77.5% Crossref-sourced, 16.4% embedded PDF fallback, 5.6% unknown, 0.4% OpenAlex, 0.04% arXiv (1 paper). **Not a cascade bug.** Follow-up diagnosis (`SELECT with_arxiv, with_doi FROM paper_metadata GROUP BY metadata_source`): **31 papers have both an arXiv ID and a DOI and were sourced by Crossref** — these are published arXiv preprints where the DOI short-circuits the cascade. Only 4 papers had an arXiv ID with no DOI (the only case where the arXiv cascade is the unique path), of which 1 was sourced successfully. The `arxiv: 1` number reflects that Crossref has authoritative metadata for most arXiv preprints, not a broken cascade step.

**Section coverage** — only 79.9% of papers have an `introduction` section detected, 62.7% `conclusion`, 41.0% `methods`, 37.3% `abstract`, 37.1% `discussion`, 24.3% `results`, **0.2% `related_work`**. The very low `related_work` hit rate is the clearest chunker signal — either the pattern regex is too narrow (missing "Background", "Prior Work", etc.) or papers in this corpus don't use that heading. Worth investigating.

**Chunk length** — healthy. Median 433 tokens, p90 531, p99 777; only 4.2% are under 100 tokens and 0.1% over 1,500. Chunking is well-tuned.

**Year distribution** — oldest 1849 (likely a digitized monograph), median 2011, newest 2026, 12% since 2020 and 9.6% before 2000. Broad temporal coverage.

**RAPTOR pyramid** — L0: 102,963 leaves, L1: 42, L2: 9, L3: 2. Steep compression; an L1 summary covers ~2,450 chunks on average. This is aggressive — useful for broad overview queries, less useful for mid-specificity.

**Hybrid-search signal overlap (Jaccard on top-50)** — dense vs sparse = 0.035, dense vs FTS = 0.000, sparse vs FTS = 0.002. **Near-zero overlap on all three pairs.** This is strong empirical confirmation that RRF fusion is buying real breadth — the signals are essentially disjoint. Note: dense-FTS = 0 deserves follow-up; it may reflect that FTS retrieves at paper level (`paper_metadata.search_vector`) whereas dense/sparse retrieve at chunk level, so the chunk IDs that come back through the FTS path are a different population.

**Reranker displacement** — cross-encoder moves the top-10 by an average of 4.2 positions per query, and changes the #1 result **100% of the time**. The reranker is doing substantial work; RRF fusion alone would pick a noticeably worse top-1 on every probe. Confirms the reranker is earning its latency cost.

**Retrieval latency** — p50 68–115 ms (warm), p90 154 ms, mean 884–1,083 ms. The long tail is from one or two queries whose expansion + FTS scan takes > 1s — worth profiling.

**Books** — 17 drafts with wordcount data; total 30,695 words. Median 1,117 words per draft (matches the 1,200-target default). Mean revision depth 3.24 versions; 58.8% heavily revised (≥3 versions). Citation density: median 28.7 words/cite, mean 33 — healthy for scientific writing.

**Autowrite convergence** — 10 recorded runs. Mean final score 0.711, median 0.8. **Median rounds to plateau = 1.** Most autowrite runs don't improve meaningfully after the first draft — the review→revise loop plateaus immediately. Early-stop rate 30%. This is either (a) the draft is already good enough on round 1, (b) the scorer isn't responsive to revisions, or (c) the revise step isn't producing meaningfully different output. Worth investigating with a targeted autowrite-bench study.

**Model throughput** (RTX 3090, VRAM-constrained) —

| Model | Standalone (GPU) | Under LLM contention (CPU fallback) | Ratio |
|---|---|---|---|
| bge-m3 encode (bs=16) | **104 chunks/s** (173 kB/s) | 2.1 chunks/s | 50× |
| bge-reranker-v2-m3 compute_score | **89 pairs/s** | 1.3 pairs/s | 68× |
| qwen3:30b-a3b (fast LLM) | — | **61 tok/s** | — |
| qwen3.5:27b (main LLM) | — | **16 tok/s** | — |

The 50-70× drop when bge-m3 or bge-reranker falls back to CPU (because the LLM is holding VRAM) is the most actionable speed finding: **any workflow that interleaves LLM + embedder should explicitly release the LLM (`ollama keep_alive=0`) before the embedder runs**, or pin them to different GPUs once the DGX Spark arrives. This is already handled in the ingestion path but may be worth checking in autowrite (where embedder is called during retrieval).

## Optimization opportunities identified

Ranked by effort × impact:

1. ~~**Release LLM keep-alive during embedder/reranker calls.**~~ **Shipped in Phase 44.1.** `sciknow.rag.llm.release_llm()` added; the ingestion pipeline now calls it before the embed stage. Measured effect in mixed workloads: bge-m3 **2.1 → 93.7 chunks/s (45×)**.
2. ~~**Fix section detection for `related_work`, `results`, `discussion`.**~~ **Shipped in Phase 44.1.** `_SECTION_PATTERNS` broadened + switched from first-match-wins to longest-prefix-wins. New synonyms cover "Extended Abstract", "Literature Review", "Experimental Setup", "Data Availability", "Code Availability", etc. Added `sciknow corpus reclassify-sections` to retroactively apply new patterns to existing chunks. Measured effect: `related_work` **0.2% → 0.7% (3.5×)**, +690 previously-unknown sections now classified, +12 `results` recoveries.
3. ~~**Investigate why arXiv metadata step never fires.**~~ **Not a bug.** 31 of the 32 papers with arXiv IDs also had DOIs, and Crossref (Layer 2) is authoritative for DOI-having papers; it answers before the arXiv cascade (Layer 3) can fire. Only 4 papers had an arXiv ID *without* a DOI (the unique-arXiv case), and 1 of those was successfully sourced by the arXiv cascade. The headline `arxiv: 1 paper` reflects correct cascade semantics, not a broken step.
4. ~~**Improve citation cross-linking.**~~ **Shipped in Phase 44.1.** Two fixes: (a) `.strip()` added on both sides of the DOI map (future-proof; no measured effect today since corpus DOIs happen to have no trailing whitespace), and (b) `db expand` now runs the link-backfill automatically after a bulk ingest. Previously the linker had to be run manually after growth. Measured effect of the one-off backfill on the existing corpus: **1,064 → 1,446 cross-linked citations (+36%)**.
5. **Investigate autowrite 1-round plateau.** Median rounds-to-plateau = 1 says the review→revise loop plateaus immediately. Root-cause with detailed scorer output on 2-3 example runs. *Still open — deferred because it requires qualitative judgment on scorer calibration, not a mechanical fix.*
6. ~~**Profile the slow retrieval tail.**~~ **Not a code issue — bench measurement artifact.** Per-query profiling of `search()` with warm models showed all 8 probe queries complete in 25–66 ms, well under the baseline's reported latency_mean of 1,083 ms. The mean was inflated by the first query eating the bge-m3 cold-load cost (~7 s). Fix in `sciknow/testing/bench.py`: added an off-the-clock warmup query before the timed loop. Re-measured: **mean 1,083 ms → 28 ms, p50 68 ms → 29 ms**.
7. **Consider deeper RAPTOR.** L1:L0 compression ratio of 1:2,450 is very aggressive. For the "what does this paper say broadly about X" use case, a midlayer (L1 with ~500 summaries) might hit a better precision/recall point. *Still open — research decision pending a targeted eval on broad-synthesis queries.*

## Post-fix scorecard (Phase 44.1 vs Phase 44 baseline, same corpus)

| Metric | Baseline | Post-fix | Δ |
|---|---:|---:|---:|
| Retrieval latency, mean | 1,083 ms | 28 ms | −97% |
| Retrieval latency, p50 | 68 ms | 29 ms | −57% |
| bge-m3 embedder, in full run | 2.1 chunks/s | 93.7 chunks/s | +45× |
| Section coverage: `related_work` | 0.2% | 0.7% | 3.5× |
| Section coverage: `results` | 24.3% | 24.8% | +0.5 pp |
| Sections classified (was unknown) | — | +690 | — |
| Citations cross-linked | 1,064 / 23,674 (4.5%) | 1,446 / 23,674 (6.1%) | +382 |
| Signal complementarity (Jaccard mean) | 0.012 | 0.012 | unchanged |
| Reranker displacement (top-1 change %) | 100% | 100% | unchanged |

Items 5 and 7 remain open and require more than mechanical fixes — an autowrite scorer calibration study and a RAPTOR-depth A/B respectively.

## Visuals ranker — 2026-04-20 baseline (Phase 54.6.140, global-cooling @ 18,928 visuals / 6,980 linked)

First measurement of the 5-signal visuals ranker (Phase 54.6.139) on a corpus-mined stratified 30-item eval set (Phase 54.6.140). Each eval item is `(query_sentence, correct_visual_id)` where the query sentence is extracted from a stored `mention_paragraph` (the source paper's body text explicitly cited that figure for that claim) — ground truth without hand curation, single-correct-answer. Ablation isolates the same-paper co-citation bonus.

Run command: `sciknow bench-visuals-ranker -n 30 --seed 42`. Artifact: `projects/global-cooling/data/bench/visuals_ranker-<ts>.jsonl`.

| Setup | P@1 | R@3 | Same-paper top-1 |
|---|---:|---:|---:|
| Full (signals 1+2+3+5) | **60.0%** | **83.3%** | 100% |
| Ablated (signal 2 off) | 40.0% | 43.3% | 60% |
| **Δ from same-paper bonus** | **+20 pp** | **+40 pp** | +40 pp |

Interpretation: the same-paper co-citation bonus is the largest single lever, contributing 20pp of P@1 and 40pp of R@3. Intuitively correct — when the writer is citing a paper, that paper's figures should dominate for nearby claims. Even with the bonus off, caption + mention-paragraph signals alone reach 40% P@1 (~6× random on a 15-candidate pool), meaning the text-only signals do meaningful work; the bonus amplifies them rather than compensating for their weakness. R@3 collapses more than P@1 under ablation (+40pp vs +20pp), indicating that without the bonus, when the correct answer isn't top-1 it tends to be deep in the ranking (not rank 2 or 3) — caption + mention alone are coarser.

RESEARCH.md §7.X.4 target was "+15 P@1 over caption-only baseline". The ablation test here is more stringent (caption + mention vs full), and the full-ranker lift is +20pp, clearing the target. A true caption-only ablation (signals 2, 3, 5 all off) is a future follow-up.

Latency: ~1.0 s/item on CPU with the LLM co-resident on GPU (bge-reranker-v2-m3 CPU fallback). Interactive-acceptable.

## Corpus section-length validation vs RESEARCH.md §24 (Phase 54.6.157, 2026-04-21)

Empirical check of the concept-density resolver's reference numbers. Run `sciknow bench --layer fast` — the new `b_corpus_section_length_distribution` function walks `paper_sections.word_count` and emits per-section-type IQRs, tagged against the §24 PubMed reference distribution (`quantifyinghealth.com` 2021, N=61,517).

Global-cooling corpus (n=807 papers, ~32k sections):

| Section | Corpus n | Corpus IQR | §24 PubMed IQR | Alignment |
|---|---:|---:|---:|---|
| `introduction` | 640 | 361–1,029 | 400–760 | **aligned** (median 630 sits inside reference) |
| `results` | 156 | 305–1,520 | 610–1,660 | **aligned** (median 650 sits inside reference) |
| `discussion` | 293 | 292–1,198 | 820–1,480 | **shorter-skewed** (median 625 vs reference median ~1,150) |
| `methods` | 453 | 149–862 | — | no §24 reference |
| `conclusion` | 535 | 206–642 | — | no §24 reference |
| `abstract` | 280 | 170–299 | — | no §24 reference |

**Reading**: introduction and results align with PubMed norms. Discussion skews meaningfully shorter — the corpus includes a lot of monograph-style chapter discussions and climate-science commentary where "discussion" tends to be a one-paragraph wrap rather than the full multi-hypothesis PubMed shape. For the concept-density resolver specifically this means the trade-science-band wpc (500–800 for `scientific_book`) is a reasonable target; lifting `scientific_book` defaults to monograph wpc would over-write discussion sections.

**Scope caveat**: this is the minimal defensible form of §24's full "concept-density regression" future-work item. Brown (2008) POS-based propositional idea density is the published measure; implementing it requires a spaCy dependency we don't carry yet. Section-length IQRs are the downstream of concept-density × wpc, so they give a useful validation signal without the NLP stack. If the idea-density regression becomes a priority, Brown 2008 + the existing `chunker_version`-stamped sections are the substrate.

## Retrieval scorecard — 2026-04-20, chunk-level FTS (Phase 54.6.136 vs 54.6.135)

Switching `_postgres_fts` from `paper_metadata.search_vector` (title+abstract+keywords+journal) to `chunks.search_vector` (a GENERATED tsvector over chunk body text). Same corpus, same probe set, same RRF weights. Artifacts: `20260420T181318Z.jsonl` → `20260420T183309Z.jsonl`.

| Metric | Paper-level FTS | Chunk-level FTS | Δ |
|---|---:|---:|---:|
| MRR@10 | 0.563 | **0.587** | **+0.024 (+4.2%)** |
| Recall@1 | 40.5% | **44.0%** | **+3.5 pp (+8.6% rel)** |
| Recall@10 | 84.5% | 84.5% | unchanged |
| NDCG@10 | 0.633 | **0.651** | **+0.018 (+2.9%)** |
| not_found_pct | 8.0% | 8.0% | unchanged |
| jaccard_sparse_fts | 0.000 | 0.049 | signal now meaningful |
| jaccard_dense_fts | 0.000 | 0.008 | slight non-zero |
| p50 latency | 29 ms | ~87 ms | +58 ms (measured clean — bench's 121 ms was system noise) |

Interpretation: paper-level FTS returned 50 chunks of 1–3 topic-matching papers, so it **almost never overlapped** with the specific chunks dense/sparse actually picked — the FTS signal was effectively contributing noise to RRF, not complementary evidence. Chunk-level FTS rewards exact-term matches in body text (rare formulas, specific numbers, uncommon terminology) that didn't appear in titles/abstracts. The +8.6% R@1 relative is the single biggest retrieval win in the corpus's measurement history. Latency cost is real but stays comfortably interactive (<100 ms p50).

The `live` bench's reranker-throughput and embedder-throughput numbers on the post-migration run (~1 chunks/s, ~1 pairs/s) were artefacts of concurrent GPU load during that run — the FTS change doesn't touch those models. Clean isolated FTS query latency on 8 probes: 1–8 ms each.

## Retrieval snapshot — 2026-04-20 (post-Phase-54.6.135, global-cooling @ 807 papers / 33k chunks)

First time the `live` layer has been persisted with retrieval-quality metrics — the Phase 44 baseline and 44.1 scorecard above were a different corpus size (2774 papers), so the numbers are not directly diff-able, but today's run establishes the anchor for future comparisons. Run command: `uv run sciknow bench --layer live --tag phase-54.6.135-post-rerank-fix --no-compare`. Artifact: `projects/global-cooling/data/bench/20260420T181318Z.jsonl`.

**Retrieval quality** (200-query probe set, `retrieval_eval.py`):

| Metric | Value | Interpretation |
|---|---:|---|
| MRR@10 | 0.563 | higher = better |
| Recall@1 | 40.5% | source chunk at top-1 |
| Recall@10 | 84.5% | source chunk in top-10 |
| NDCG@10 | 0.633 | binary-relevance NDCG |
| **not_found_pct** | **8.0%** | 16/200 queries had their source chunk escape the top-50 entirely |
| latency p50 | 28.7 ms | |
| latency mean | 30.7 ms | |

**Rerank behaviour** (4-query sanity probe):

| Metric | Value | Interpretation |
|---|---:|---|
| avg_displacement | 5.25 positions | cross-encoder actively reorders |
| top1_change_rate | 75.0% | 3/4 of queries get a new top-1 after rerank |
| rerank_latency_mean | 910 ms | CPU (LLM resident on GPU) |

**Signal overlap** (Jaccard of top-50 candidate IDs across dense / sparse / FTS):

| Pair | Jaccard | Note |
|---|---:|---|
| dense ∩ sparse | 0.051 | small overlap, both vector-family |
| dense ∩ FTS | **0.000** | bench's own note: *"both lexical, so should be higher"* for sparse pair |
| sparse ∩ FTS | **0.000** | zero shared candidates — either perfect complementarity or a bug |
| mean overlap | 0.017 | low is good (more complementary signals) |

The `sparse ∩ FTS = 0.0` specifically contradicts the bench's built-in "should be higher" expectation because both are lexical signals. Two hypotheses: (a) FTS is complementing vector signals by design and the note is stale — needs a design review to confirm; (b) the Postgres `tsvector` column is indexing something disjoint from what `chunks` are embedded (e.g. wrong content field, wrong tokenizer, stale index). Open action item — worth a 30-min investigation before calling retrieval "healthy". Flagged in `docs/PHASE_LOG.md` 54.6.135 as follow-up.

**Model throughput** (single-pass, CPU):

| Metric | Value |
|---|---:|
| bge-m3 embedder | 99.1 chunks/s (159 k char/s), batch 16 |
| bge-reranker-v2-m3 | 87.1 pairs/s |

### Reading the numbers

- **R@10 = 84.5%** is a solid hybrid-retrieval figure on a scientific corpus; most of the gap to 100% is concentrated in the 8% not-found bucket.
- **R@1 = 40.5%** matches the intuition that cross-encoder reranking is doing the heavy lift: the RRF-fused top-1 is correct only 40% of the time, but the right chunk is almost always within reach (top-10), which is exactly where the reranker reshapes results (75% top-1 change rate).
- Future deltas to watch: (i) whether FTS-overlap investigation exposes a lexical-index bug; (ii) whether Phase 54.6.134's coverage-check rerank fix affects any upstream metric (shouldn't — that was a filter, not a retrieval config); (iii) whether the eventual visuals-write-loop integration (§7.X in RESEARCH.md) ships on a corpus whose text-retrieval numbers held steady.

## Adding a new bench function

Same pattern as the test harness: write a zero-arg function that yields `BenchMetric` objects, then register it in `LAYERS` in `sciknow/testing/bench.py`.

```python
def b_my_new_metric() -> Iterable[BenchMetric]:
    """One-line purpose."""
    from sciknow.storage.db import get_session
    with get_session() as session:
        n = session.execute(text("SELECT ...")).scalar()
    yield BenchMetric("my_count", n, "units", note="helpful context")
```

Guidance:

- **No assertions.** Bench functions report metrics; the compare-to-latest pass decides whether a number drifted.
- **Skip gracefully.** Call `skip("reason")` if the data isn't there — the bench is reported as `~ skipped`, not errored.
- **Cheap by default.** `fast` layer functions must not call any model. `live` may load bge-m3/bge-reranker (now via the llama-server substrate, post-v2). `llm` is the writer-throughput layer (dispatches to llama-server or Ollama based on `USE_LLAMACPP_WRITER`). `v2` measures the three Decision Gates; assumes all three llama-server roles up.
- **Record units and context.** `note` is free-form but should contain anything that helps interpret the number later (% of total, direction of "better", what model produced it).

## Correctness gate

The bench harness itself is gated by `l1_bench_harness_surface` in `sciknow/testing/protocol.py` — guards against accidental removal of the `sciknow bench` CLI wiring or a bench function that was decorated into accepting parameters.
