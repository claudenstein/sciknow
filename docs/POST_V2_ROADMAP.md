# Post-v2 Roadmap

**Companions:** [`docs/V2_FINAL.md`](V2_FINAL.md) (the four post-shipment stages that closed v2.0.0rc2 → v2.0.0), [`docs/ROADMAP.md`](ROADMAP.md) (the catch-all open-items doc), [`docs/SCIKNOW_V2_ROADMAP.md`](SCIKNOW_V2_ROADMAP.md) (the original v1→v2 phase plan, all 7 phases shipped).

This doc is the **shipping order** for everything between today (2026-04-26, `v2.0.0rc2` on `v2-llamacpp`) and the next major version. It assumes the V2_FINAL stages 1–4 are done (they are, see the status table in `V2_FINAL.md`) and that the Stage 5 soak window will close cleanly.

When something here ships, mark it ✅ DONE with the commit ref and prune the entry — same convention as `ROADMAP.md`.

---

## Phase 0 — Close the v2.0.0 ship (this week, mechanical)

The only thing standing between us and `v2.0.0` is the calendar soak window.

| Step | Command | Notes |
|---|---|---|
| Soak (~7 days) | normal sciknow use against substrate | watch `library doctor --watch 60`; record any anomalies in `docs/PHASE_LOG.md` |
| Bump version | `sed -i 's/2.0.0rc2/2.0.0/' pyproject.toml && git commit` | RC2 → final |
| Tag | `git tag v2.0.0 -m "..."` | on `v2-llamacpp` |
| Merge | `git checkout main && git merge --no-ff v2-llamacpp` | one feature commit on main |
| Push | `git push origin main v2.0.0` | publish |
| README flip | edit the v2 callout near the top | "v2.0.0 is current on `main`" |

**Exit criteria:** tag visible on GitHub, README points at `main`, `v2-llamacpp` branch retained as historical reference but no longer the working branch. `V2_FINAL.md` Stage 5's status flips to ✅ DONE.

---

## Phase 1 — v2.1 cleanup (~1 week post-tag, low risk)

Kill the one-release deprecation hatches that v2.0 explicitly carried for compatibility. Each item is independently shippable:

1. **Drop `DENSE_EMBEDDER_MODEL` sidecar.** The dual-embedder split + per-project `<slug>_dense` Qdrant collection is gone in v2.1; `library upgrade-v1` is the single migration path.
   - Remove the validator warning in `sciknow/config.py`.
   - Remove the fallback code path in `sciknow/retrieval/`.
   - Remove `l1_v2_dual_embedder_deprecation_warning` from `sciknow/testing/protocol.py`.
   - Update `MIGRATION.md` to drop the "dual-embedder fallback" section.
   - **~200 lines deleted.**

2. **Remove `sciknow db` namespace shim.** It's been mounted with a deprecation warning for one release; v2.1 deletes the shim.
   - Drop the shim in `sciknow/cli/main.py`.
   - Update `MIGRATION.md` to drop the "old verbs still mounted" column.
   - **~50 lines deleted.**

3. **Decide on the v1 Ollama dispatch hatch in `rag/llm.py`.** Two options:
   - **Keep** through v2.1 — explicit rollback escape hatch (current state). Costs ~80 lines of dead code on the v2 path.
   - **Remove** in v2.2 once soak data has accumulated months of confidence. Drop the `_use_llamacpp()` branch and the `_get_client()` helper. Ollama can come back via a separate `sciknow.rag.ollama_legacy` module if ever needed. **~150 lines deleted.**

   Recommended: **keep through v2.1, drop in v2.2.**

**Exit criteria:** `pyproject.toml` bumps to 2.1.0; `_OLLAMA_KEPT_BY_DESIGN` shrinks by 1 (the `rag/llm.py` entry stays); `MIGRATION.md` shows a fresh v2.0→v2.1 column; the `library doctor` no longer warns about the dual-embedder split.

---

## Phase 2 — v2.2 features (`docs/ROADMAP.md` Tier 1/2 leftovers)

These are the only items in `ROADMAP.md` that are still genuinely open, don't require Spark, and pay back in real corpus quality. In effort-impact order:

1. **Per-book settings modal** (`ROADMAP.md` §5, half-day). Consolidates `target_chapter_words`, `mineru_vlm_model`, `custom_metadata` editing into one UI panel. Currently scattered across the Plan modal + Chapter modal + `.env`.

2. **bge-m3 LoRA fine-tune on synthetic Q-chunk pairs** (`ROADMAP.md` §6b #4, 1–2 days). Uses the existing probe set from the `b_retrieval_recall` bench as contrastive-loss training data. Distinct from the *reranker* LoRA in §3 Phase E. Keep base bge-m3 as fallback. Expected gain: +0.03–0.05 MRR.

3. **Autowrite stall investigation** (`ROADMAP.md` §5, depends on root cause). Phase 24 instrumented this; the next stall is the diagnostic opportunity. Not pre-planning work — be ready when one happens.

**Exit criteria:** `pyproject.toml` 2.2.0; `docs/ROADMAP.md` §5 + §6b shrink by 2 entries; LoRA checkpoint added to the `INFER_*` config surface.

---

## Phase 3 — DGX Spark unlock (gated on hardware arrival)

Detailed in [`docs/ROADMAP.md`](ROADMAP.md) §3, ordered simplest-first:

| Item | Effort | What it unlocks |
|---|---|---|
| **Phase B** — LLM host routing (3090 fast vs Spark 70B) | ~50 LoC | mixed-workload fast/main split |
| **Phase C** — vLLM backend on Spark | ~1 day | batch ops (catalog cluster, autowrite, argue, gaps) |
| **Phase E** — bge-reranker-v2-m3 QLoRA on the climate corpus | ~3 days | domain-specialised reranker |
| **Phase 47.S1** — CycleReviewer-ML-Llama3.1-8B as autowrite scorer | ~2 days | published 26.89% MAE reduction vs current LLM scorer |
| **Phase 47.S2** — iterative DPO on KEEP/DISCARD verdicts | ~1 week | self-improvement loop on the user's preference data |
| **Phase 47.S3** — fast-detect-gpt pre-publish gate | ~3 days | flag AI-generated text before export |
| **Layer 6** — domain LoRA on the writer (DPO) | ~1 week | `qwen-sciknow:32b` as the new default writer |

**Sequencing:** Phase B first (smallest), then C (biggest throughput win), then S1/E in parallel (independent), then S2 / Layer 6 last (need accumulated preference pairs).

**Do NOT** move embedder / reranker to Spark — 3090 has 3.4× more bandwidth and they're bandwidth-bound at single-batch-size.

---

## Phase 4 — data-gated learning passes (no code blocker, just dataset accumulation)

From [`docs/ROADMAP.md`](ROADMAP.md) §4b. Currently waiting on usage:

- **Layer 3 — Heuristic distillation (ERL-style).** Cluster Layer 1 lessons into generalized strategic principles; prepend unconditionally to the writer prompt. **Gate: ≥50 autowrite runs accumulated.** Effort: ~2 weeks once the data is there.
- **Layer 4 export → Layer 6 training.** Layer 4 (DPO export) shipped in Phase 32.9. Layer 6 (writer LoRA) ships when both the Spark is up AND ≥2k validated preference pairs are in `data/preferences/<book>.jsonl`. Per Wolfe 2024, 2k pairs × 3 epochs is enough for meaningful gains.

These are watch-list items: the moment the counters cross the threshold, ship them.

---

## What I'd skip / defer indefinitely

- **Re-relitigating** the [`docs/RESEARCH.md`](RESEARCH.md) §526 rejected list (HyDE, Self-RAG, GraphRAG global, full RST, etc.). Rejected with documented reasons; the literature would have to materially change to revisit.
- **Inventing v3.** v2 just shipped a backend swap; v3 should wait until there's a concrete forcing function (a serving paradigm or model class we can't bolt onto v2's abstractions). Today there isn't one.

---

## Sequencing summary

| Window | Phase | Status | Depends on |
|---|---|---|---|
| Now → +1 wk | **0** — close v2.0.0 ship | ⏳ soak open | calendar |
| +1 → +2 wk | **1** — v2.1 deprecation cleanup | pending | Phase 0 |
| Opportunistic | **2** — v2.2 features | pending | Phase 1 |
| Hardware | **3** — DGX Spark stack | gated | Spark delivery |
| Usage | **4** — Layer 3 / Layer 6 | gated | ≥50 / ≥2k runs |

This doc is intentionally short-lived. When Phase 0 closes, prune Phase 0 entirely. When v2.1 ships, prune Phase 1. The point is to keep the file honest about *what's still open*, not to accumulate historical rationale (that's `PHASE_LOG.md`'s job).

---

## How this doc is maintained

Same convention as `ROADMAP.md`:

1. When picking the next phase, check this doc first; ship the smallest open Phase that's not blocked.
2. After shipping, **delete the entry** (don't leave done items as historical clutter — Git history + `PHASE_LOG.md` are the historical record).
3. Only add new entries when something is genuinely deferred (researched + decided + not shipped yet).
4. If this file grows past ~300 lines, that's a sign work is accumulating faster than shipping it.
