# SciKnow v2 — Final Stretch

**Companion to:** `docs/SCIKNOW_V2_ROADMAP.md` (the original phase plan, all 7 phases shipped) and `MIGRATION.md` (the verb / settings reference).

This document covers the **post-shipment items** the original roadmap required but never marked done. The substrate code is fully landed and the L1+L2 contract suites are green; what's missing is the *verification layer* — proof that the substrate actually works end-to-end on a pure-v2 install (no Ollama, no FlagEmbedding) and that the spec's three decision gates have measured data behind them.

Each stage below is one to a handful of commits. Stages 1-4 are mechanical; stage 5 is calendar time. Once stages 1-5 land, `v2.0.0` is mergeable to `main`.

## Status (2026-04-25)

| Stage | Status | Commit | Note |
|---|---|---|---|
| 1 — v2-only L3 suite | ✅ DONE | `584f93c` | 8 new `l3_v2_*` tests; all pass against running substrate |
| 2 — ollama import audit + L1 contract | ✅ DONE | `34d0817` | 5 sites routed via `rag.llm.complete()`, 8 kept-by-design with reasons; L1 catcher in place |
| 3 — `sciknow bench --layer v2` | ✅ DONE | `158192b` | Gates A/B/D measured; embedder ubatch fix shipped as drive-by |
| 4 — BENCHMARKS.md gate entries | ✅ DONE | `7c946b3` | Headline table: A PASS (+12.5%), B PASS-by-construction, D PASS (within noise) |
| 5 — Soak + tag | ⏳ in progress | — | SMOKE layer passes end-to-end on substrate (2026-04-25); calendar window open until ~2026-05-02 |

All four mechanical stages landed in one continuous session (2026-04-25). The substrate was the active backend throughout; both L1 (158 tests, ~10s) and L2 (full integration, ~3 min) stayed green across every commit, and SMOKE (extract-kg + wiki compile + autowrite + num_predict cap + extract JSON canary) ran clean against the running llama-server roles after Stage 2 routed five formerly-Ollama sites through `rag.llm.complete()`.

---

## Stage 1 — v2-only L3 suite (~1 commit, 2 hours)

**Goal:** prove the llama-server substrate works end-to-end without falling back to Ollama or FlagEmbedding.

**Scope:** add `l3_v2_*` tests under `sciknow/testing/protocol.py` that exercise the v2 paths exclusively:

- `l3_v2_writer_health` — `httpx GET {INFER_WRITER_URL}/health` returns 200; assert `WRITER_MODEL_GGUF` is set + the file exists.
- `l3_v2_embedder_health` — `httpx GET {INFER_EMBEDDER_URL}/health` + assert `EMBEDDER_MODEL_GGUF` set + file exists.
- `l3_v2_reranker_health` — `httpx GET {INFER_RERANKER_URL}/health` + `RERANKER_MODEL_GGUF`.
- `l3_v2_writer_complete_smoke` — `infer.client.complete(...)` (not `llm.complete` which can dispatch either way) returns a non-empty string in <30s.
- `l3_v2_embedder_roundtrip` — `infer.client.embed(["smoke test"])` returns a 1024-dim list of floats.
- `l3_v2_reranker_roundtrip` — `infer.client.rerank("query", ["doc1", "doc2"])` returns a list of float scores ordered correctly.
- `l3_v2_autowrite_uses_llamacpp` — run one autowrite iteration and assert `rag.llm._use_llamacpp()` returns `True` during the run AND no `ollama` import was triggered (audit `sys.modules`).
- `l3_v2_dispatch_audit` — assert that for every `USE_LLAMACPP_*=True`, the corresponding `infer.client.*` call shape is reachable from `rag/llm.py` without an `ollama.` reference in the resolved code path.

**Skip-gracefully:** each test bails with `TestResult.skip` if the corresponding role isn't running (so the suite can run on a CI box without the GGUFs).

**Exit criteria:** L3 reports `8 v2 tests, 8 passed` against a box with all three llama-server roles up and `OLLAMA_HOST` pointed at a dead port.

---

## Stage 2 — `ollama` import site audit (~3-5 commits, 1 day)

**Goal:** the 11 modules that bypass `rag/llm.py` and import `ollama` directly either route through the substrate or are gated behind explicit v1-only flags.

**Scope:** by import site —

| File                              | Use                                       | Action                                           |
| --------------------------------- | ----------------------------------------- | ------------------------------------------------ |
| `sciknow/rag/llm.py`              | dispatch fallback                         | **Keep** (intentional rollback hatch)            |
| `sciknow/core/finalize_draft.py`  | claim verifier LLM call                   | Route through `infer.client.complete` when on v2 |
| `sciknow/cli/wiki.py`             | KG triple judge LLM call                  | Same                                             |
| `sciknow/ingestion/metadata.py`   | LLM-based metadata cascade (4th layer)    | Same                                             |
| `sciknow/ingestion/enrich_sources.py` | optional `--llm-fallback` enrich layer | Same — guarded by the existing `--llm-fallback` flag, but the call itself should use the substrate |
| `sciknow/cli/db.py`               | misc lazy LLM calls                       | Same — multiple sites, walk each |
| `sciknow/core/monitor.py`         | `_ping_ollama` + `release-vram` Ollama unload | **Keep** (the doctor explicitly probes ollama on v1 fallback paths; release-vram is a recovery action) |
| `sciknow/web/routes/system.py`    | `release-vram` ollama unload (mirror of monitor) | **Keep** for parity |
| `sciknow/testing/model_sweep.py`  | benchmarking different models             | **Keep** (model_sweep is a v1-era research tool) |
| `sciknow/testing/vlm_sweep.py`    | VLM benchmarking                          | **Keep** (uses Ollama-served vision models) |
| `sciknow/testing/quality.py`      | quality bench against named Ollama tags   | Walk; route the writer-side calls through infer.client; quality bench's *judge* model can stay on Ollama |

**Exit criteria:** L1 contract test `l1_v2_no_unguarded_ollama_imports` walks the source for `import ollama` lines and asserts each is either inside a `if not getattr(_s, "use_llamacpp_*", True):` guard, inside a function explicitly tagged as v1-only via a `_v1_fallback` decorator/marker, or in the kept-by-design list above.

---

## Stage 3 — `sciknow bench` for v2 (~1-2 commits, 4 hours)

**Goal:** the existing `bench --layer llm` works against the v2 substrate, and a new `bench --layer v2` measures the spec's Gate A/B/D metrics.

**Scope:**

- **Gate A (writer tok/s).** New bench `b_writer_tps_substrate` calls `infer.client.complete(...)` 5 times against a fixed prompt, reports `decode_tps`, `prompt_tps`, `first_token_s` (the same shape that `data/bench/writer_tps.jsonl` already uses). Add the v1 equivalent under the existing layer for comparison.
- **Gate B (embedder throughput).** New bench `b_embedder_throughput_substrate` embeds a fixed batch of 256 chunks via `infer.client.embed`; reports `chunks/sec` + `vectors/sec`. Spec's gate: ≥0.8× FlagEmbedding's recorded baseline.
- **Gate D (retrieval recall@10).** New bench `b_retrieval_recall_at_10` runs a fixed 100-query set against the active corpus, reports recall@10 for the dense + RRF-fused leg. Held-out queries live in `tests/fixtures/retrieval_recall_queries.jsonl` (committed to the repo so the bench is reproducible).
- **`--layer llm` rewrite.** Replace the direct `ollama.generate()` call with `llm.complete()` (which dispatches via `_use_llamacpp`). Add a print line on entry: `"backend: llama-server (v2)"` or `"backend: ollama (v1)"` so the operator sees which path was timed.

**Exit criteria:** `sciknow bench --layer llm` reports backend correctly. New bench layer `v2` runs all three gate measurements and writes them to the standard JSONL output.

---

## Stage 4 — `BENCHMARKS.md` entries for the three gates (~1 commit)

**Goal:** the spec's "data lands in BENCHMARKS.md at phase end" requirement honored for Gates A/B/D.

**Scope:** add a `## v2 Decision Gates` section to BENCHMARKS.md with three subsections:

- **Gate A (writer tok/s).** Pull the existing data from `data/bench/writer_tps.jsonl`: llamacpp 36 tps median vs ollama 32 tps median on Qwen3.6-27B-Q4_K_M. **Verdict: PASS** (writer beats baseline).
- **Gate B (embedder throughput).** Run `bench b_embedder_throughput_substrate` once, record numbers. The spec's bar is ≥0.8× FlagEmbedding; pre-Stage-3 we don't have the comparable baseline number. Either record both or document why we accept the gate as met.
- **Gate D (retrieval recall@10).** Run `bench b_retrieval_recall_at_10` once, record. Spec's bar: stable vs v1 baseline. If we don't have the v1 baseline, run the bench once on `USE_LLAMACPP_EMBEDDER=False` for the comparison.

Each entry: one short paragraph, the metric, the verdict, and a link to the JSONL data. **Exit criteria:** all three gates have a recorded number + a PASS / RECONSIDER verdict.

---

## Stage 5 — Soak + tag (calendar time)

**Goal:** real-world v2 use catches latent regressions before the `2.0.0` git tag.

**Soak scope** (do all routine sciknow work against the v2 substrate exclusively for ~1 week, no `USE_LLAMACPP_*=False` overrides):
- Run `sciknow corpus ingest directory` on a fresh batch of papers.
- Run `sciknow wiki compile` on the resulting docs.
- Run `sciknow book autowrite` on at least one chapter.
- Watch `sciknow library doctor --watch 60` during all of the above; record any new alerts in `docs/PHASE_LOG.md` as a follow-up entry.

**Day-0 (2026-04-25) sanity already done.** SMOKE layer (`uv run sciknow test --layer SMOKE`) ran 5/5 ✓ against the substrate after Stage 2 landed:
- `l3_extract_model_produces_clean_json` — 4 concepts + 6 triples in 12.8 s
- `l3_wiki_compile_single_paper_smoke` — 377-word summary in 44 s
- `l3_wiki_extract_kg_single_paper_smoke` — 10 triples + 13 entities in 29 s
- `l3_autowrite_one_iteration_smoke` — 660-char draft in 4 s
- `l3_llm_num_predict_cap_honored` — cap honored at 111 chars in 0.9 s

So the writer-LLM hot pipelines work end-to-end on substrate routing today. The soak window is now about *catching the long-tail*: papers with weird metadata, bulk runs that exhaust connections, anything that only shows up under sustained use.

**Exit criteria:** zero rollback-class incidents over the soak window. After that, the tag is mechanical:

```bash
# 1. Bump version (RC1 → 2.0.0).
sed -i 's/^version = "2.0.0rc1"/version = "2.0.0"/' pyproject.toml
git add pyproject.toml
git commit -m "chore: bump to 2.0.0 — substrate verified, gates passed, soak clean"

# 2. Tag and push.
git tag v2.0.0 -m "v2.0.0 — llama-server substrate, library + corpus subapps, decision gates A/B/D passed"

# 3. Merge to main as a single feature commit.
git checkout main
git merge --no-ff v2-llamacpp -m "feat: v2.0.0 — llama-server substrate, library + corpus subapps"

# 4. Push.
git push origin main v2.0.0

# 5. Update README's `v2 ships on the v2-llamacpp branch` callout.
#    The two lines to flip live in README.md near the top.
```

**Rollback hatch.** v1 stays reachable at the `last-ollama-build` tag on `main`; the dispatch facade (`rag/llm.py`) keeps both branches so `USE_LLAMACPP_WRITER=False` + `uv add ollama` reverts at runtime.

---

## Risk register

- **Stage 2 carries the most regression risk.** The 11 modules with `import ollama` are doing it for specific reasons (e.g. model-named keep-alive timing, embedded JSON-format prompting). Routing each through `infer.client` means the substrate has to support those modes. Most likely outcome: 80% of sites flip cleanly, 20% need new `infer.client` features (e.g. `complete(..., format="json")`). Plan for it.
- **Stage 3's recall@10 bench needs a held-out query set.** If we don't have one, we have to either curate one (~30 min of labelling on the active corpus) or accept the gate as "met by construction" (the canonical embedder is unchanged from v1).
- **Stage 5's soak is the longest stage in calendar time but the smallest in engineering.** Don't shortcut it — it's the only thing standing between "code shipped" and "shipped + verified by reality."

## Sequencing

Stages 1, 2, 3, 4 can land in any order; they don't depend on each other. Pragmatic ordering:

1. Stage 1 first — fastest (smallest blast radius, biggest confidence boost).
2. Stage 3 next — Stage 4 needs Stage 3's bench numbers.
3. Stage 4 — write up Stage 1's + Stage 3's data.
4. Stage 2 — by far the largest in code volume; do it alongside Stage 5's soak so any regressions surface in the same window.
5. Stage 5 — concurrent with 2-4, then exit.
