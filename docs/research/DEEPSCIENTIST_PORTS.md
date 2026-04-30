# DeepScientist — Ranked Port List

[&larr; Back to README](../README.md) · [Comparison](../benchmarks/COMPARISON.md) · [Roadmap](../roadmap/ROADMAP.md)

---

Enumeration and ranking of every concrete feature from DeepScientist ([ResearAI/DeepScientist](https://github.com/ResearAI/DeepScientist), arXiv:2509.26603, ICLR 2026 top-10) considered for import into sciknow. Based on source-level reads of `memory_service.py`, `quest_layout.py`, `artifact_schemas.py`, and the paper §3.

Ranking criteria (all weighted):
- **Impact** — does it unlock new capability vs. refining an existing one?
- **Fit** — does it match sciknow's Python + Postgres + Qdrant + Ollama stack?
- **Effort** — days of focused work (no Spark required unless noted).
- **Debt** — does it obsolete existing code, or add alongside it?
- **Genre** — does it apply to both `scientific_book` AND `scientific_paper` project types?

---

## Already covered in sciknow (no port needed)

These look like DeepScientist innovations at first glance but sciknow has equivalent machinery already. Leaving them here as the "don't reinvent" list.

| # | DeepScientist feature | Where sciknow covers it | Notes |
|---|---|---|---|
| 0a | Findings Memory (`memory_service.py:14–272`) with embedding-free substring search | `autowrite_lessons` table (Phase 32.7) + `_get_relevant_lessons` (`book_ops.py:1010`) | sciknow's version uses bge-m3 embeddings — strictly better than substring. See `docs/reference/LESSONS.md`. |
| 0b | Recency decay on memory retrieval | `_get_relevant_lessons` recency formula: `2^(-age_days / 30)` | Generative Agents 2023 formula. DeepScientist uses a simpler last-modified sort. |
| 0c | Per-project isolation (quest dirs) | Phase 43 `projects/<slug>/` layout + per-project PostgreSQL DB + Qdrant collections | sciknow's isolation is stronger — the DB boundary is enforced at the connection layer, not just the filesystem. |
| 0d | LLM-as-reviewer at the end of a run | `_verify_draft_inner` + Phase 46.C `ensemble_review_stream` (NeurIPS rubric + meta-reviewer) | Ours is more structured — numeric rubric, stance rotation, mechanical fallback. |
| 0e | Markdown-with-frontmatter artifacts | Phase 30 multi-format export + `drafts.custom_metadata` JSONB | DeepScientist uses FS; we use DB. FS is portable but DB lets us query. |
| 0f | `promote_to_global` MCP tool | **Not covered** — sciknow lessons are book-scoped with a 0.7× cross-book downweight but no explicit global pool | See port #1 below. |

## Ranked ports (all Spark-optional — local-only is feasible)

Numbers: **Impact** out of 5, **Effort** in focused days. Ordered by Impact ÷ Effort descending.

### #1 · Kind taxonomy on `autowrite_lessons` — 2d, Impact 5/5
`memory_service.py:11` defines six card kinds: `papers, ideas, decisions, episodes, knowledge, templates`. sciknow's `autowrite_lessons` has a `dimension` column (scorer axis) but nothing describes *what* the lesson is about.

Concrete change:
- Alembic migration adds `kind TEXT` (default `'episode'` for legacy rows).
- `_distill_lessons_from_run` tags each extracted lesson with an inferred kind via a tiny classifier prompt or heuristic.
- `_get_relevant_lessons` gains an optional `kind=` filter.
- Immediate downstream use: `book gaps` can exclude `kind='rejected_idea'` to stop re-proposing what didn't work.

Why #1: lowest effort, unlocks two downstream features (port #2 and #7), and the migration is additive (nothing breaks).

---

### #2 · Rejected-idea memory gate on `book gaps` — 2d, Impact 4/5
Needs #1. Before `book gaps` proposes a topic, query `autowrite_lessons WHERE kind='rejected_idea' AND section_slug MATCH …` and inject matches into the generator prompt as "Do NOT re-propose these — they were tried and scored poorly because...". Matches `doc_mem.md:26–30` of DeepScientist ("review prior quest idea cards before proposing a new idea").

Why important: the Phase 44 bench showed autowrite plateaus at round 1 for many drafts. Part of that is probably "generator keeps proposing the same borderline topics." This closes the loop.

---

### #3 · Scope (`book`/`global`) + `promote_to_global` — 3d, Impact 4/5
Add `scope TEXT CHECK (scope IN ('book', 'global'))` to `autowrite_lessons`. Add a `sciknow_lessons` global table (same schema, no `book_id`). A `promote_to_global()` helper lifts any lesson meeting:
- `importance ≥ 0.8`
- `score_delta > 0`
- appeared (via embedding similarity ≥ 0.85) in ≥ 3 distinct books

into the global pool. `_get_relevant_lessons` unions both scopes, weighting global 0.7× by default (adjustable).

Why important: sciknow's multi-project isolation (Phase 43) is intentional for corpora but shouldn't apply to **writing-style lessons** ("don't overstate causality on cross-correlations" applies equally to a climate paper and a materials-science paper). This is the cross-book learning the research roadmap has been pointing at.

---

### #4 · Fidelity tier on `autowrite_runs` — 3d, Impact 4/5
Paper §3.2: three tiers (Hypothesis → Implement → Progress) with cheap-to-expensive promotion gates. For sciknow:

| Tier | Trigger | Memory writes |
|---|---|---|
| `hypothesis` | any finished autowrite run | distills lessons but marked low-importance by default |
| `verified`   | `verification.groundedness_score ≥ 0.7` AND `final_overall ≥ 0.7` | full importance; feeds into promote-to-global gate |
| `published`  | included in a `book export` artifact | importance × 1.2; becomes "canonical" knowledge |

Concrete change: `autowrite_runs.fidelity_tier` column + gate in `_distill_lessons_from_run`. Also feeds into `book autowrite --tier cheap/full` flag from the Phase 47 comparison doc.

Why important: fixes the "low-quality runs pollute memory" failure mode that our lesson loop is vulnerable to today. DeepScientist's key insight is that cheap exploration is OK *only if* durable memory stays clean.

---

### #5 · UCB acquisition function for `book gaps` / `book autowrite --tier` — 3d, Impact 3/5
Paper Eq. 2: `argmax(wu·vu + wq·vq + κ·ve)` where each candidate gets three 0–100 scores (utility, quality, exploration). This is a **hand-weighted UCB, not a GP**. Implementable in ~50 LOC.

Concrete change: when `book gaps` has multiple candidate topics OR `book autowrite` has multiple queued sections, run a tiny LLM pass to score each on (utility, quality, exploration) and pick via UCB. The exploration bonus (`κ·ve`) is what prevents the generator from always preferring its top-confidence pick.

Why #5 not higher: it adds complexity in exchange for better queue ordering, but today's generators work OK. Pays off most when users run `book autowrite --full` on a large chapter — instead of sequential, we go UCB-ordered.

---

### #6 · Typed decision artifacts with preserved failure branches — 4d, Impact 3/5
`artifact_schemas.py:16–45` defines 8 legal decision actions: `branch, prepare_branch, activate_branch, reset, stop, waive_baseline, request_user_decision`. Each is a first-class durable record with required `verdict`, `action`, `reason`.

For sciknow: when `book autowrite` DISCARDs a revision or the user manually rolls back a draft, write a `decision` row into a new `book_decisions` table. Link it to the draft via `parent_draft_id` (already exists) AND to an optional `alternative_draft_id` for side-by-side comparisons.

Why #6 not higher: overlaps with Phase 38 scoped snapshot bundles (we already preserve prior drafts). The port adds structured provenance on *why* a rollback happened, not new data; Phase 38 keeps *what*.

---

### #7 · Kind-filtered retrieval in the writer prompt — 2d, Impact 3/5
Needs #1. Rather than injecting top-K mixed lessons, the writer prompt gets segmented:

> **Things we know about this section** (kind='knowledge', 2 lessons)
> …
> **Things we've tried and rejected** (kind='rejected_idea', 1 lesson)
> …
> **Decisions we made elsewhere on this book** (kind='decision', 1 lesson)

This is what DeepScientist's MCP tools enable via the `kind=` parameter to `memory.list_recent`.

Why: the mixed-bag prompt today treats a "don't do X" lesson the same as a "here's a useful frame" lesson. The writer handles them differently if they're separated.

---

### #8 · `book scout` — recent-literature clustering before `book gaps` — 5d, Impact 3/5
This is the **Zochi "narrow-then-ideate"** pattern, not DeepScientist per se, but fits in the same spot. Before running `book gaps`, crawl the corpus for recent (last 3 years) papers, cluster via BERTopic (already installed), surface emerging sub-topics, and feed *those* into `book gaps` as prompt context.

Why here: included because DeepScientist's canonical skill anchors include `scout` (see `doc_canvas.md:42–50`), which maps directly.

---

### #9 · Memory body convention (1-context → 7-retrieval-hint) — 1d, Impact 2/5
`doc_mem.md:148–155` specifies a 7-part structure for every card: *1. context, 2. action/observation, 3. outcome, 4. interpretation, 5. boundaries, 6. evidence paths, 7. retrieval hint for future turns*.

For sciknow: update the `_distill_lessons_from_run` prompt to emit lessons in this shape. Keep the column TEXT (no schema change). The benefit is that retrieval-hint-at-the-end makes embedding-based similarity search more robust — the last sentence of a lesson becomes a synthetic query for the next run.

Low effort but the benefit is marginal without also doing #3 (scope).

---

### #10 · Frontmatter-style `evidence_paths` linking to artifact files — 3d, Impact 2/5
DeepScientist's frontmatter has `evidence_paths: [artifacts/baselines/verification_report.md]`. For sciknow: extend `drafts.sources` (JSONB list) with an optional `evidence_paths` field pointing to ensemble-review reports, citation-verify JSON, etc., generated in Phase 46.

Why low: we already store the source data (`custom_metadata.ensemble_review`, `custom_metadata.phase46_citations_added`). This port just moves it to a filesystem path that external tools can grep.

---

### #11 · Substring memory search as a fallback for embedding search — 1d, Impact 2/5
DeepScientist uses **substring search** (`memory_service.py:200`), not embeddings. We use embeddings. For a `_get_relevant_lessons` with zero embedding hits (e.g., query too short), we could fall back to PG `ILIKE '%query%'` over `lesson_text`.

Genuinely low impact because bge-m3 rarely returns zero. Listed for completeness.

---

## Deliberately not ported (marked "will not pursue" to prevent relitigating)

| # | Feature | Why not |
|---|---|---|
| N1 | Electron / npm launcher (`bin/ds.js`) | sciknow is a CLI + FastAPI web reader; no GUI shell. |
| N2 | MCP server plumbing | Tool-call protocol for external agent hosts. sciknow's generator events are the equivalent and better-typed. |
| N3 | `LabQuestGraphCanvas.tsx` graph UI | React/TypeScript. Would require a whole new frontend stack. |
| N4 | Per-quest git repo | Phase 43 per-project layout is stronger (DB + Qdrant boundaries, not just FS). |
| N5 | Seven canonical skill anchors (`scout, baseline, idea, experiment, analysis-campaign, write, finalize`) as a rigid pipeline | sciknow's generator composition is less rigid and more CLI-native. Cherry-picked `scout` in port #8 instead. |
| N6 | Hand-weighted UCB as a Gaussian process | The paper frames it as "BO" but it's a UCB; port #5 covers the practical implementation without the theory theater. |
| N7 | Bash `waive_baseline` / `reset` pipeline actions | Presuppose the quest-as-git-repo model (N4). Not a fit. |

---

## Recommended bundle — Phase 47 "Typed Compound Memory"

If we ship a focused phase, the right bundle is **#1 + #2 + #3 + #7**: typed lessons, rejected-idea gate, global scope, kind-filtered writer prompt. Total ≈ 9 days. Delivers:

- Cross-book style learning (a "don't overstate causality" lesson from a climate paper reaches the materials-science paper's writer)
- Self-correcting `book gaps` (stops re-proposing rejected topics)
- Segmented writer prompt (knowledge / rejected-ideas / decisions rendered separately)

#4 (fidelity tier) is then a natural Phase 48 follow-on once we have cross-book telemetry to calibrate against.

## Summary table

| # | Feature | Days | Impact | Notes |
|---|---|---:|---:|---|
| 1 | Kind taxonomy on `autowrite_lessons` | 2 | 5 | Foundation for #2, #7 |
| 2 | Rejected-idea gate on `book gaps` | 2 | 4 | Needs #1 |
| 3 | Scope + promote-to-global | 3 | 4 | Cross-book learning |
| 4 | Fidelity tier on runs | 3 | 4 | Keeps memory clean |
| 5 | UCB acquisition for queueing | 3 | 3 | Paper Eq. 2 |
| 6 | Typed decision artifacts | 4 | 3 | Overlaps Phase 38 |
| 7 | Kind-filtered writer prompt | 2 | 3 | Needs #1 |
| 8 | `book scout` pre-gaps pass | 5 | 3 | Zochi pattern |
| 9 | 7-part lesson body convention | 1 | 2 | Marginal w/o #3 |
| 10 | Evidence-path frontmatter | 3 | 2 | Data already stored |
| 11 | Substring fallback | 1 | 2 | Rarely triggered |
