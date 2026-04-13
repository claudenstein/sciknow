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

**Metadata mix** — 77.5% Crossref-sourced, 16.4% embedded PDF fallback, 5.6% unknown, 0.4% OpenAlex, **0.04% arXiv (1 paper)** — almost certainly indicates the arXiv cascade step is rarely triggering in practice. Either Crossref is already answering for arXiv papers, or the arXiv detection/lookup is too conservative.

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

1. **Release LLM keep-alive during embedder/reranker calls.** 50-70× slowdown when both want VRAM. Low-risk code change.
2. **Fix section detection for `related_work`, `results`, `discussion`.** 0.2%-24% hit rate means chunker is blind to a third of scientific structure. Extend `_SECTION_PATTERNS` in `sciknow/ingestion/chunker.py`.
3. **Investigate why arXiv metadata step never fires.** 1 of 2,774 papers went through the arXiv cascade — suggests the detection is too conservative or already-shadowed by Crossref. Trace with a sample of known arXiv papers.
4. **Improve citation cross-linking.** 4.5% linked. Either DOI-exact matching is missing partial matches, or the corpus just doesn't cite within itself often. Try title-normalized fuzzy match for unlinked citations.
5. **Investigate autowrite 1-round plateau.** Median rounds-to-plateau = 1 says the review→revise loop plateaus immediately. Root-cause with detailed scorer output on 2-3 example runs.
6. **Profile the slow retrieval tail.** p50/p90 are fine but mean is pulled up by a few >1 s queries. Add instrumentation to `search()` to log per-signal latency, then rerun the bench.
7. **Consider deeper RAPTOR.** L1:L0 compression ratio of 1:2,450 is very aggressive. For the "what does this paper say broadly about X" use case, a midlayer (L1 with ~500 summaries) might hit a better precision/recall point.

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
