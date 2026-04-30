# sciknow v2 — Roadmap

**Companion to** `docs/roadmap/SCIKNOW_V2_SPEC.md`. Sequence the rebuild so each phase ships independently, can be tested independently, and either rolls forward or rolls back without disturbing the others.

**Branch model**: all v2 work on `v2-llamacpp`. v1 stays on `main`, frozen at the `last-ollama-build` tag (commit `cf91386`). Each phase below corresponds to one PR (or a small chain) merged to `v2-llamacpp`. After Phase G, `v2-llamacpp` merges to `main` and v1 retires.

**Time estimates** are calendar-realistic for one developer, not optimistic. They include benchmarking + L1/L2 regression budget, not just the implementation.

---

## Phase A — `infer/` substrate (≈2 weeks)

**Goal**: stand up `llama-server` for the writer role and route every generative call through it. Embedder and reranker stay on the v1 in-process path. This isolates the most disruptive change (replacing Ollama) from everything else.

**Scope**:
- `sciknow/infer/server.py` — subprocess manager (start/stop/health/swap). Writes PIDs + logs to `data/infer/`.
- `sciknow/infer/client.py` — `httpx` client with `/v1/chat/completions` (streaming), `/v1/completions`, `/health`, `/slots`. Returns events compatible with the existing SSE schema.
- `sciknow/cli/infer.py` — `up`, `down`, `status`, `swap`, `logs`. Default profile = writer-only on `:8090`.
- `sciknow/rag/llm.py` — replace `ollama.chat` calls with `infer.client`. Preserve the call-stats capture (eval_count, prompt_eval_count, etc.) by reading llama-server's `usage` block + `timings` payload.
- `.env` migration: add `INFER_WRITER_URL`. Keep `OLLAMA_HOST` as a no-op fallback for one phase only (warn loudly).

**Out of scope**: embedder, reranker, CoV refactor, web rebuild.

**Exit criteria**:
- L3 autowrite cycle on the toy corpus completes through llama-server with no Ollama process running.
- Writer tok/s on Qwen3.5-27B Q4_K_M ≥ Ollama baseline (no regression).
- L1 contract tests pass for the new event shapes.
- `data/sciknow.log` contains llama-server startup + shutdown lines.

**Rollback**: revert the PR; v1's Ollama path is intact in `main`.

---

## Phase B — embedder + reranker on `llama-server` (≈1.5 weeks)

**Goal**: kill the in-process model load class of bugs. Embedder and reranker move to dedicated llama-server instances.

**Scope**:
- Bring up two more llama-server processes (`:8091` embedder, `:8092` reranker) under the same `infer/server.py` manager. Default profile becomes "all three".
- `sciknow/infer/client.py` — add `embed(texts: list[str]) → list[list[float]]` against `/v1/embeddings` and `rerank(query: str, docs: list[str]) → list[float]` against `/v1/rerank`.
- `sciknow/ingestion/embedder.py` — drop `FlagEmbedding`. Call `infer.embed`. Keep the dense+sparse contract unless the chosen embedder server doesn't expose sparse — if not, split into two roles or accept dense-only and update `retrieval/hybrid.py`.
- `sciknow/retrieval/rerank.py` — drop `sentence-transformers`. Call `infer.rerank`.
- Drop `sciknow/retrieval/device.py` entirely. Drop the dual-embedder code path. Drop `core/gpu_ledger.py` and `core/vram_budget.py`.

**Out of scope**: ColBERT prefetch (kept; uses its own model, decision deferred to Phase D).

**Exit criteria**:
- L2 + L3 retrieval tests pass with no in-process torch.cuda calls (search the codebase to assert).
- Ingest throughput (chunks/min on a 100-paper sample) within 0.8× of v1 baseline. The likely loss is dense+sparse co-encoding; if it bites, run two embedder servers on CPU+GPU and round-robin.
- `pyproject.toml` no longer depends on `FlagEmbedding`, `sentence-transformers`, or `torch` directly (torch may still come transitively from MinerU; pin it but don't import it).

**Rollback**: revert; Phase A baseline still works.

---

## Phase C — autowrite simplification (≈1.5 weeks)

**Goal**: collapse the v1 autowrite scar tissue now that VRAM contention is gone.

**Scope**:
- Split `core/book_ops.py` (6.7 kLOC) into `core/book_ops.py` (CRUD-ish) + `core/autowrite.py` (the iteration engine).
- Define the v2 SSE event schema in `core/events.py`. All generators yield Pydantic instances; SSE serialisers call `model_dump_json()`.
- Remove every `_release_gpu_models()` call site (no longer meaningful).
- Remove the v1 three-tier active-version pick — replace with a real `drafts.is_active BOOLEAN` column + partial-unique index. Migration backfills `is_active = (custom_metadata->>'is_active' = 'true')` and falls back to highest-version-with-content.
- Keep batched CoV (the v1 phase 54.6.319 fix) — it's already correct.
- Keep the bibliography globaliser (`core/bibliography.py`) untouched.

**Exit criteria**:
- One autowrite cycle on the toy corpus emits ≤16 distinct event types, all schema-conforming.
- L1 contract tests cover every yield site in `autowrite.py`.
- `is_active` column is populated for every existing draft via a one-shot migration.

**Rollback**: this phase touches data shape (`is_active` column). Prepare a downgrade migration that restores `custom_metadata.is_active` from the column before reverting code.

---

## Phase D — retrieval cleanup (≈1 week)

**Goal**: one canonical retrieval path. No A/B sidecar configs, no dual-embedder.

**Scope**:
- `library init` (the renamed `db init`) creates a single Qdrant `papers` collection with the canonical embedder's dim. No `papers_qwen3e4b` sidecar.
- Drop the dual-embedder code paths in `retrieval/hybrid.py`. ColBERT prefetch stays as an opt-in flag (`SCIKNOW_USE_COLBERT=1`); default off.
- `retrieval/visuals.py` — keep CLIP image-embed sidecar for now (no llama-server CLIP yet). Document it as the second exception alongside MinerU.
- Resolve open question §9.2 from the spec: if Qwen2.5-VL through llama.cpp mmproj is stable, plan its integration as Phase E.5; otherwise stay on CLIP.

**Exit criteria**:
- One Qdrant collection per project for `papers` (plus `abstracts`, `wiki`, `visuals`).
- L2 hybrid-search test reports stable retrieval at recall@10 vs. v1 baseline.
- ColBERT off by default; L1 ColBERT-disabled regression test in place.

**Rollback**: Qdrant collection schema is a one-way change for an existing project. Document a `library reset` requirement and ship a `project import-v1` plan (deferred to Phase G).

---

## Phase E — web reader rebuild (≈3 weeks)

**Goal**: turn `web/app.py` (31.6 kLOC) into a maintainable FastAPI app.

**Scope**:
- Externalise CSS into `web/static/css/sciknow.css`. Single file, no preprocessor.
- Externalise JS into `web/static/js/<module>.js`. ~10 modules. No bundler. Each page emits `<script type="module" src="…">` tags only.
- Move every >100-line HTML f-string into a Jinja2 template under `web/templates/`. Templates can compose partials (modals, panels, sidebars).
- Split routes by resource into `web/routes/{books,drafts,papers,wiki,kg,jobs,admin}.py`. Each module ≤500 lines.
- `web/app.py` shrinks to mount points + middleware + lifespan. Target ≤2 kLOC.
- Keep the SSE wire format; only `/api/jobs/{id}/stats` polling is allowed for the task bar (the v1 32.5 architectural rule).

**Out of scope**: visual redesign, new features. This is structural only.

**Exit criteria**:
- Every page renders pixel-identically to v1 (visual regression via screenshot diff on a fixed 5-page set).
- `web/app.py` ≤ 2,000 LOC.
- L2 web-reader smoke test passes (existing tests).
- L1 contract tests assert no `<style>` or multi-line `<script>` in any template.

**Rollback**: trivially per-PR (split into ~10 PRs).

---

## Phase F — CLI reorganisation + scar removal (≈1 week)

**Goal**: shrink `cli/db.py` (11.5 kLOC) and `cli/book.py` (4.3 kLOC); audit + drop deprecated tests/alerts.

**Scope**:
- `cli/db.py` → split into `cli/library.py` (init/reset/stats/migrate/validate/snapshot) and `cli/corpus.py` (ingest/expand/enrich/cluster). Top-level subapp renames: `db` → `library`; new top-level `corpus`.
- `cli/main.py` registers the new subapps. Add deprecation shims: `sciknow db <verb>` prints a one-shot warning then dispatches to the new home for one minor release.
- Audit the 61 L1 regression tests. Each `l1_phase*` test gets a label: KEEP (still load-bearing), MIGRATE (rewrite as a contract test), DROP (the underlying behaviour no longer exists in v2). Goal: cut to ~30 contract-shaped tests.
- Audit the 27 alert codes. Drop deprecated entries; keep ACTIVE only.
- Drop `core/expand_feedback.py` if its surface is fully covered by `feedback`.

**Exit criteria**:
- `cli/library.py` + `cli/corpus.py` each ≤2 kLOC.
- L1 test count ≤35 and every test is contract-shaped (asserts schemas, wire formats, exit codes — not source greps).
- `sciknow --help` shows no `db` subapp; deprecation warning fires for one release then disappears.

**Rollback**: per-subcommand revert is fine; the test audit is irreversible (re-adding scar tests is not on the menu).

---

## Phase G — v1 import + cutover (≈1 week)

**Goal**: take a v1 project to v2 cleanly.

**Scope**:
- `sciknow project import-v1 <v1-slug> --as <v2-slug>` reads a v1 project's Postgres + Qdrant + filesystem, writes a v2 project with the canonical embedder (re-embeds chunks; this is the slow step), restores books+drafts+wiki pages, and stamps `imported_from = <v1-slug>` in the v2 project metadata.
- Cutover plan: tag `v2.0.0` on `v2-llamacpp`, merge to `main`, retire v1.
- Update README + CLAUDE.md + memory-store entries to point at v2 commands.
- Archive v1 docs into `docs/v1/` for posterity.

**Exit criteria**:
- The author's "global cooling" project (the largest existing v1 dataset) imports cleanly.
- A round-trip `library snapshot` → `library reset` → `project unarchive` works on the imported v2 project.
- README, CLAUDE.md, and the memory store reflect v2 commands.

**Rollback**: `main` retains v1 history; reverting the merge restores v1.

---

## Optional follow-ups (post-v2.0)

These are nice-to-have, not on the critical path:

- **Phase E.5 — Qwen2.5-VL via llama.cpp mmproj**: replace the CLIP visuals sidecar with a llama.cpp-native VLM if/when mmproj support is stable. Earns the "MinerU is the only exception" claim.
- **Phase H — DFlash speculative decoding**: bring Lucebox DFlash kernel to the writer profile when upstream-merged. Per the speculative-decoding memo, expect 1.6×–2.1× writer tok/s.
- **Phase I — observability consolidation**: collapse the three pulse formats into one; replace the v1 monitor snapshot screen with a dedicated `/admin/observability` page rendered server-side from the same pulse store.
- **Phase J — KG pipeline split**: extract `core/kg/` into a standalone module that the wiki-compile and the KG-overview page both depend on. Currently the KG canonicalisation logic is duplicated between `core/kg_canonicalize.py` and the wiki compile prompts.

---

## Sequencing notes

- **Phase A is the single highest-risk phase.** It changes the inference backend for every generative call. Budget time for prompt cache + tokenizer + sampling-parameter equivalence to Ollama. Run the L3 toy-corpus autowrite a dozen times before declaring Phase A green.
- **Phase B can start before Phase A merges**, since the embedder/reranker server processes don't conflict with the writer process. They can be developed on a side branch and merged after Phase A.
- **Phase E (web rebuild) is bisectable but slow.** Don't block Phases C/D on it. The web reader works as a 31 kLOC f-string for the duration; structural cleanup happens after the inference engine is stable.
- **Phase F (CLI reorg) is the only phase that breaks the user's muscle memory.** Ship the deprecation shims + a `MIGRATION.md` that maps every v1 verb to its v2 home before merging.
- **Phase G must not run until Phases A–D have been live on the author's working project for ≥2 weeks** without a roll-back-class incident.

## Calendar (best-case linear)

| Phase | Weeks | Cumulative |
|-------|-------|------------|
| A     | 2.0   | 2.0        |
| B     | 1.5   | 3.5        |
| C     | 1.5   | 5.0        |
| D     | 1.0   | 6.0        |
| E     | 3.0   | 9.0        |
| F     | 1.0   | 10.0       |
| G     | 1.0   | 11.0       |

≈11 weeks (≈3 calendar months) for a clean v2.0.0 cutover, assuming no Phase A regressions and no major MinerU/llama.cpp upstream breakage.

## Decision gates

These three gates require explicit go/no-go decisions before proceeding:

1. **End of Phase A**: did writer tok/s match or beat the Ollama baseline? If not, investigate llama-server flags (KV-cache shifting, batching, n_predict caps) before starting Phase B.
2. **End of Phase B**: did the embedder server's throughput on bulk ingest stay within 0.8× of FlagEmbedding's? If not, decide whether to (a) keep `FlagEmbedding` as an ingestion-time fallback, (b) accept the slowdown, or (c) round-robin across multiple embedder server instances.
3. **End of Phase D**: did retrieval recall@10 stay stable on a held-out 100-query set? If not, the canonical embedder choice may be wrong — consider Qwen3-Embedding-4B as the canonical instead of bge-m3 (and pay the 8 GB VRAM cost).

Each gate's data lands in `docs/benchmarks/BENCHMARKS.md` at phase end.
