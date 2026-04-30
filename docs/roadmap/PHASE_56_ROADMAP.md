# Phase 56 — Claim-atomic autowrite

**Branch**: `v2-llamacpp`
**Revert anchor**: tag `autowrite-stable-phase55-v19` (commit `417c195`) — the last single-shot writer with LLM-chosen `[N]` markers. All Phase 56 work composes on top of this; if a sub-phase regresses on prose quality or wall-time without lifting citation precision, revert to the tag.
**Status**: roadmap draft, awaiting joint review. Nothing implemented yet.
**Companion**: [`docs/roadmap/SCIKNOW_V2_ROADMAP.md`](SCIKNOW_V2_ROADMAP.md) (substrate), [`docs/research/RESEARCH.md`](../research/RESEARCH.md) §LongCite / §FActScore (citation-precision priors).

---

## TL;DR

The single-shot writer is fundamentally asked to do retrieval-time reasoning ("which of these 12 chunks supports the claim I'm about to make?") at generation time, with a coarse retrieval query and a 24 KB context block. It picks `[N]` markers that don't match the supporting chunk 17–62% of the time (measured on the global-cooling 2026-04-29 chapter). `citation_align` patches this post-hoc by NLI-rescoring every cited sentence; that's a fix-up, not a fix.

Phase 56 inverts the loop: the writer never picks `[N]`. The pipeline plans atomic claims, retrieves per claim, micro-generates each claim's sentence(s) against its pinned chunks, then stitches into prose. Citations are a deterministic function of `claim → supporting chunk`, computed by the assembler, not chosen by the writer. **No re-ingestion, no schema changes, no model swaps.** Same `papers`/`abstracts` Qdrant collections, same bge-m3 embedder, same writer/scorer/reranker, same `chunks`/`paper_metadata`/`drafts` tables.

---

## Problem statement

Measured on the four sections of the global-cooling chapter that ran on 2026-04-29 (`projects/global-cooling/data/autowrite/20260429_*_eff0773e.jsonl`):

| Section | Final score | Iters | citation_align remap rate |
|---|---|---|---|
| Sumerian/Egyptian minima | 0.80 | 3 | 55.7% |
| Classical antiquity / solar cycles | 0.62 | 3 | 62.5% |
| Medieval Maximum / Warm Period | 0.86 | 3 | 16.7% |
| Little Ice Age (Maunder/Dalton/Spörer) | 0.86 | 2 | 34.8% |

Two reproducible failure modes:
1. **Citation misalignment** — 17–62% of `[N]` markers point at a chunk that doesn't entail the cited sentence (per `citation_align` NLI). `citation_align` rewrites them, but the underlying evidence-claim pairing during writing is wrong.
2. **Hedging fidelity is the bottom-ranked dimension on every completed section.** Writers upgrade hedged claims into bare assertions (`suggests` → `shows`, `is associated with` → `causes`). The Phase 55 prompt block targeting this hasn't moved the needle.

Adjacent observation worth fixing: 3-iter loops with `KEEP` × 3 and ≤ 0.02 score delta (Sumerian, Classical, Medieval) burn ~10 min per section to confirm the draft hasn't changed. The LIA section's 2-iter early-exit (because iter-1 revision actually moved the score 0.81→0.86) shows the loop *can* converge faster. Tracked under 56.0 below.

## Diagnosis (5 root causes, ranked by contribution)

1. **Retrieval query is too coarse.** `autowrite.py:471` uses `f"{section_type} {topic}"` (slug + chapter title). All 12 chunks are vaguely on-topic for the *whole chapter*, none specifically pinned to a single factual claim.
2. **Writer is given a numbered menu and told to "cite as [N]".** `prompts.py:1087` instructs grounding but not entailment. The writer satisfies the rule by emitting *any* valid number.
3. **Sentence-level citation mandate creates filler pressure.** `prompts.py:1094` requires every factual sentence to end in `[N]`; combined with hedging-fidelity demands, the writer reaches for *something* numerically plausible.
4. **Long-context attention degrades on the source list.** 12 × ~1500-char chunks ≈ 18 KB of evidence. Standard middle-of-context dropout — markers skew toward chunks at the ends of the list.
5. **Generation order ≠ evidence order.** Mid-section claims get the same `[N]` the writer used earlier for an analogous claim, regardless of which chunk supports the new sentence.

## Design principles

- **Mechanical citation.** The writer must never choose `[N]`. Citation IDs are computed deterministically by the assembler from `claim → supporting chunk`.
- **Per-claim retrieval.** Each atomic claim retrieves its own evidence. The writer never sees a 12-chunk menu.
- **Drop, don't fabricate.** A claim with no chunk above the entailment floor is dropped from the draft, not papered over with a wrong citation.
- **Plan carries the hedge.** `hedge_strength` is a property of the claim, set at plan time from the supporting chunks. The writer transcribes; the scorer verifies deterministically.
- **No re-ingestion, ever.** This is a code-only refactor. If a sub-phase wants to touch ingestion or schema, it doesn't ship under Phase 56.

## Target architecture

```
section_plan
   │
   ▼
ClaimPlanner ─────────────► [Claim(text, scope, hedge_strength), …]
                                              │
                                              ▼ (per claim)
                                  ClaimRetriever (hybrid_search) ─► top-3 chunks
                                              │
                                              ▼
                                  ClaimWriter (micro-gen)
                                  prompt = (claim, 2-3 chunks)
                                  output = sentence(s) with <C:claim_id>
                                              │
                                              ▼
                              ┌─── claim → supporting chunk(s) ───┐
                              │                                   │
                              ▼                                   ▼
                       DraftAssembler ◄── all claim outputs ──────┘
                              │
                              ▼
                        StitchWriter (prose flow only;
                        cannot edit text inside <C:…>)
                              │
                              ▼
                       CitationRenderer
                       <C:claim_id> → [N] from claim.chunks
                              │
                              ▼
                          final draft
```

Module map (new files marked with ▶, existing in [brackets]):

- ▶ `sciknow/core/claim_plan.py` — `ClaimPlanner.plan(section_plan, topic) → ClaimPlan`
- ▶ `sciknow/retrieval/claim_retrieve.py` — `retrieve_for_claim(claim) → list[Chunk]` with per-claim caching
- ▶ `sciknow/core/claim_writer.py` — `write_claim(claim, chunks) → str` (sentences with `<C:claim_id>` placeholders)
- ▶ `sciknow/core/draft_assembler.py` — stitches claim outputs in plan order
- ▶ `sciknow/core/stitch_writer.py` — prose flow + entity bridges, claim spans frozen
- ▶ `sciknow/core/citation_render.py` — `<C:claim_id>` → `[N]` mapping + APA source list
- [`sciknow/core/autowrite.py`] — top-level orchestrator, swap to claim-pipeline behind a flag
- [`sciknow/core/citation_align.py`] — kept as a sanity-check post-pass (should be a no-op when the pipeline works); flag to fail-loud if it ever has to remap
- [`sciknow/core/claim_atomize.py`] — unchanged; remains the *post-hoc* offline atomizer for `book verify-draft`. Phase 56's `ClaimPlanner` is a pre-write planner — different timing, different output schema, do not merge.

## Data & schema

**Required changes: none.** All new state is transient within a writer run.

**Optional persistence (56.7, ship only if metrics warrant):**
- Additive Alembic migration: `drafts.claims JSONB NULL` carrying `[ClaimPlan + claim→chunk_ids]`. Enables: `book verify-draft` cross-checks, "show evidence per sentence" UI affordance, re-stitch without re-retrieval.
- Backwards-compatible — null on any pre-56.7 draft, both engines coexist.

---

## Phasing

Each sub-phase ships as one PR (or short chain) into `v2-llamacpp`. Each has a clean revert. The Phase 55 path stays available behind `--engine=v55` until 56.6 lands and metrics confirm the new path is at parity or better.

### 56.0 — Iteration loop hygiene (≈ half day)

**Goal**: stop wasting compute on `KEEP × 3` runs.
**Scope**: in `autowrite.py` (around line 1697), if iter-1 revision delta is < 0.02 AND the verdict is `KEEP`, skip iter 2/3 and emit `early_exit_no_improvement`. Belongs in 56.0 because it's independent of the rest of Phase 56 and pays back the wall-time budget the new pipeline will consume.
**Exit**: a section whose iter-1 doesn't improve completes in ~⅓ the wall time. Existing Phase 55 scoring unchanged.
**Risk**: low. Pure orchestration tweak.

### 56.1 — Smallest-viable patch (≈ 2 days)

**Goal**: capture ~60% of the citation-precision win without touching architecture.
**Scope**:
1. Replace the retrieval query in `autowrite.py:471` with `topic + " " + (section_plan or "")[:1500]` so the chunks are pinned to the section's stated content, not the chapter title.
2. Make `citation_align` mandatory (not advisory) on the initial draft and after each revision.
3. Raise `citation_align`'s `low_threshold` from 0.5 → 0.55, and add a `drop_threshold` (~ 0.4): sentences whose top entailing chunk is below `drop_threshold` have their `[N]` stripped (the sentence becomes uncited rather than wrong-cited).
4. Surface a new `citation_precision` field in `scores` events (mean entailment of cited chunk vs. cited sentence).
**Exit**: on a side-by-side run of the global-cooling chapter, `citation_precision` lifts from current ~0.50 → ≥ 0.70; remap rate drops by ≥ 30 pp.
**Rollback**: revert PR. Phase 55 path unchanged.

### 56.2 — Atomic claim plan (≈ 1 week)

**Goal**: stand up the claim planner; consume into the writer prompt as a sidebar (NOT yet driving retrieval or citations). Validate plan quality before betting the rest of the pipeline on it.
**Scope**:
- New `sciknow/core/claim_plan.py` with `Claim(text, scope, hedge_strength, plan_bullet_id)` and `ClaimPlanner.plan(section_plan_bullets, topic, prior_summaries) → ClaimPlan`. Reuses the writer model.
- New prompt `CLAIM_PLAN_SYSTEM` in `prompts.py`: "given these section plan bullets, produce N atomic claims (subject-predicate-quantifier-scope), each with a hedge strength."
- Writer sees the claim list as a sidebar in its prompt; claims influence what the writer asserts but the writer still picks `[N]`.
- L1 unit tests: claim parser, hedge inference from cue words, scope preservation.
**Exit**: claim plans for the existing 4 global-cooling sections look reasonable on inspection (joint review). Writer prompt with claim sidebar produces drafts that don't regress on the Phase 55 scorer.
**Rollback**: feature flag off; claim sidebar omitted from prompt.

### 56.3 — Per-claim retrieval (≈ 1 week)

**Goal**: replace the section-level retrieve with per-claim retrieves; writer still picks `[N]` but now from claim-pinned context.
**Scope**:
- New `sciknow/retrieval/claim_retrieve.py`. `retrieve_for_claim(claim) → list[Chunk]` calls `hybrid_search` with `claim.text + claim.scope` and returns the top-3 above an entailment floor.
- Caching layer keyed on `claim.text` (claims sharing supporting chunks shouldn't trigger duplicate retrievals).
- `autowrite.py` orchestrator: replace the single `_retrieve_with_step_back` call with `claim_retrieve(claim)` per claim, then **union** the results into the writer's source list. The writer still sees a numbered context, but every chunk in it is pinned to at least one claim.
- New telemetry event `claim_coverage` per draft: list of claims with < 3 chunks above threshold (early signal of corpus gaps; informs `book gaps`).
**Exit**: source list is now claim-pinned. `citation_precision` lifts another ≥ 10 pp on side-by-side. Wall-time regression bounded at < 1.5×.
**Rollback**: revert; 56.2 sidebar still works.

### 56.4 — Constrained micro-generation (≈ 2 weeks) — the load-bearing change

**Goal**: eliminate writer-chosen `[N]` entirely.
**Scope**:
- New `sciknow/core/claim_writer.py`: `write_claim(claim, chunks) → str`. Prompt is constrained: "produce 1-3 sentences asserting this claim, using ONLY these N chunks as evidence, ending the cited sentence with `<C:{claim_id}>`."
- New `sciknow/core/draft_assembler.py`: walks claims in plan order, calls `write_claim` per claim, concatenates into a draft.
- New `sciknow/core/citation_render.py`: walks the draft, replaces `<C:claim_id>` with `[N]` where N is the deduped index of `claim.chunks` in the final source list. Generates the APA source list deterministically.
- `autowrite.py`: behind `--engine=v56` (or `engine_version` config), routes to the new pipeline. Default still v55 until 56.6.
- L2 tests: assembled draft has zero `[N]` chosen by an LLM (assertion: every `[N]` in the final draft traces back to a `<C:claim_id>` that exists in the ClaimPlan).
**Exit**: full claim-atomic draft generated end-to-end on a single section. `citation_precision` ≥ 0.92 on the global-cooling chapter (target: structural). Remap rate from `citation_align` → near 0 (it should have nothing to fix).
**Rollback**: flag flip back to v55.

### 56.5 — Stitch pass for prose flow (≈ 1 week)

**Goal**: restore prose quality lost to claim-by-claim micro-generation.
**Scope**:
- New `sciknow/core/stitch_writer.py`: takes the assembled draft, produces a flowing version with entity-bridge transitions (prompts.py:1123 rule), paragraph breaks, and connective tissue.
- **Hard constraint**: stitch pass cannot edit text inside `<C:…>` spans (verified post-hoc by diff against pre-stitch claim outputs). If the stitch model touches a claim span, the pass fails closed and we keep the assembled draft.
- Stitch model defaults to the writer port (`:8090`); evaluate whether a smaller model (e.g. extractor `:8095`) suffices.
**Exit**: subjective prose quality on a 5-section blind A/B with v55 drafts is rated ≥ parity by joint review. Hedging_fidelity dimension on the scorer no longer the bottom-ranked dimension.
**Rollback**: skip stitch; ship the assembled draft as-is. Slightly choppier prose but citations stay correct.

### 56.6 — Default flip + Phase 55 deprecation timer (≈ 2 days)

**Goal**: make v56 the default engine.
**Scope**:
- Flip default `engine_version` to `v56`.
- Phase 55 path (`v55`) becomes opt-in via `--engine=v55`, marked for sunset in the changelog.
- Update CLI help, web UI status, and `book autowrite --help` to reflect the new default.
- README + PHASE_LOG entries.
**Exit**: 1 week of running v56 by default with no >0.05 regression on overall scorer score and a measured citation_precision gain ≥ 0.30 over the autowrite-stable-phase55-v19 anchor.
**Rollback**: flip default back to v55. v56 stays available.

### 56.7 — Optional: claim graph persistence (≈ 1 week)

**Goal**: persist the ClaimPlan + claim→chunk_ids on the draft for audit, re-stitch, and downstream tooling.
**Scope**:
- Alembic migration: `drafts.claims JSONB NULL`. Additive, no backfill.
- `book verify-draft`, `book argue`, `book gaps` made claim-aware where useful.
- Web reader UI: optional "show evidence" affordance per sentence.
**Exit**: gated on metric — ship only if 56.4-56.6 metrics show that having the claim graph at hand would unlock a measurable downstream win (e.g. `book argue` precision, `book gaps` recall). Otherwise defer to a later phase.
**Rollback**: drop the column; both engines stay working.

---

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Wall-time per section ≥ 2× v55 | medium | medium | Parallel per-claim retrieval; batched micro-gen; claim-chunk caching. Budget cap: 1.5× by 56.6. |
| Choppy prose post-stitch | medium | medium | 56.5 stitch pass; if insufficient, fall back to a "soft stitch" that allows light edits inside claim spans (relaxed constraint). |
| Plan quality is the bottleneck — bad plan → bad draft, no recovery | medium | high | Same model that already produces decent chapter outlines; 56.2 is gated on plan-quality joint review. Worst case: keep v55 single-shot as a fallback when planner confidence is low. |
| Empty corpus regions surfaced as dropped claims | high | low | This is a *feature*. `claim_coverage` telemetry feeds `book gaps`. Users see what's missing instead of getting fabricated citations. |
| Scorer behaviour changes when hedging becomes deterministic | high | low | Reweight hedging_fidelity in scorer config; keep old config as `phase55_legacy_scorer` for A/B. |
| `citation_align` becomes a no-op and we lose its safety net | low | medium | Keep it running; flag fail-loud if it ever has to remap (would indicate a pipeline bug, not normal operation). |
| Interaction with CoVe (Chain-of-Verification) becomes redundant | medium | low | CoVe was designed to catch unsupported claims at write time; the new pipeline catches them at plan time (via entailment floor). 56.4 should make CoVe largely redundant — open question for review. |

## Success metrics

Measured against `autowrite-stable-phase55-v19` baseline on the global-cooling chapter and 4 other sections from existing books.

| Metric | Baseline (v55) | Target (v56) | How measured |
|---|---|---|---|
| `citation_precision` (mean NLI entailment of cited chunk vs. cited sentence) | ~0.50 | ≥ 0.92 | NLI scoring per cited sentence |
| `citation_align` remap rate | 17-62% | ≤ 5% (structurally near 0) | `citation_align` event |
| `hedging_fidelity` scorer dimension | bottom-ranked, ~0.55 | ≥ 0.75, no longer bottom | scorer event |
| Overall scorer score | 0.62-0.86 | within ±0.05 of v55 (no regression) | scorer event |
| Wall-time per section | ~25 min | ≤ 1.5× | autowrite log |
| Subjective prose quality (blind A/B) | n/a | parity or better in joint review | manual rating on 10 paired drafts |

## Test plan

- **L1**: claim parser, citation render, draft assembler, hedge inference unit tests. Add to `sciknow/testing/protocol.py` as `l1_phase56_*` checks.
- **L2**: per-claim retrieval against the global-cooling DB; assembler integration; round-trip render.
- **L3**: full autowrite cycle on a single section under v56; then full chapter; side-by-side with v55 on the same section_plans.
- **Regression budget**: every Phase 55 L1/L2/L3 check still passes when running with `--engine=v55`.
- **Gold set**: pin 5 v55 drafts at the tag as ground truth for prose-quality A/B comparison.

## Migration / coexistence

- Drafts produced by either engine carry `engine_version` ("v55" | "v56") in their metadata for forensic clarity.
- Web reader unchanged — both engines produce drafts in the same `drafts` table shape.
- `book argue` and `book gaps` see no change unless 56.7 ships.
- The claim graph (if persisted in 56.7) is additive JSON; consumers that don't know about it ignore it.
- v55 path stays available indefinitely as fallback. Sunset decision is its own future phase, not part of 56.

## Open questions for joint review

1. **Persist the claim graph (56.7) or keep transient?** Cost: one migration + UI work. Benefit: audit trail + re-stitch + downstream tooling. Decide after 56.6 metrics.
2. **Stitch model size.** Does the writer model (Qwen3.5-32B class) need to do the stitch, or does the extractor (Qwen3.5-9B on `:8095`) suffice? Probably the latter; verify in 56.5.
3. **CoVe redundancy.** If 56.4 catches unsupported claims structurally, should CoVe stay for belt-and-suspenders or be retired to save tokens?
4. **Visuals widener interaction.** The Phase 54.6.150 retrieval-density widener and the visuals_ranker both assume a section-level evidence pool. With per-claim retrieval, do they need claim-aware variants, or do we keep them at the section level (unioned across claims)?
5. **Hedging vocabulary as data.** Should we move `prompts.py:1115`'s hedge cue list into a config so the deterministic hedging check uses the same source of truth as the planner?
6. **Failure mode when no chunk passes the entailment floor for a planned claim.** Drop the claim silently? Surface to `book gaps`? Both?
7. **Iteration loop in v56.** Does it still make sense to iterate the whole pipeline 3×, or do we iterate only the stitch pass (since claims and citations are now structural)?

## Out of scope

- Changes to chunkers, embedders, rerankers, scorers (model or weights).
- Re-ingesting any data; touching `paper_metadata`, `chunks`, Qdrant collections, or FTS triggers.
- Web UI rebuilds beyond optional "show evidence" affordance in 56.7.
- ColBERT prefetch, RAPTOR-related work, or other retrieval upgrades — tracked under [`docs/roadmap/ROADMAP.md`](ROADMAP.md).
- New scorer model (gemma stays the cross-family scorer).
- Chapter outline / book plan generator — claim planner consumes existing section_plan bullets verbatim.

---

**Next step on this doc**: joint review with the user, then convert agreed-on sub-phases into PR-shaped tasks in `docs/roadmap/ROADMAP.md`.
