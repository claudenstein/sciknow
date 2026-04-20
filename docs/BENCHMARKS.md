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
| `llm`  | ~60 s  | Ollama fast + main model tokens/sec | Ollama reachable, both models pulled |
| `full` | ~3 min | Everything, in the order above. | All of the above |

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
2. ~~**Fix section detection for `related_work`, `results`, `discussion`.**~~ **Shipped in Phase 44.1.** `_SECTION_PATTERNS` broadened + switched from first-match-wins to longest-prefix-wins. New synonyms cover "Extended Abstract", "Literature Review", "Experimental Setup", "Data Availability", "Code Availability", etc. Added `sciknow db reclassify-sections` to retroactively apply new patterns to existing chunks. Measured effect: `related_work` **0.2% → 0.7% (3.5×)**, +690 previously-unknown sections now classified, +12 `results` recoveries.
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
- **Cheap by default.** `fast` layer functions must not call any model. `live` may load bge-m3/bge-reranker. `llm` is the only layer allowed to call Ollama.
- **Record units and context.** `note` is free-form but should contain anything that helps interpret the number later (% of total, direction of "better", what model produced it).

## Correctness gate

The bench harness itself is gated by `l1_bench_harness_surface` in `sciknow/testing/protocol.py` — guards against accidental removal of the `sciknow bench` CLI wiring or a bench function that was decorated into accepting parameters.
