# Compound Learning — Where Is "D" in sciknow?

[&larr; Back to README](../README.md)

---

This document answers a direct question: *"Is the MetaClaw / DeepScientist 'Findings Memory' pattern (opportunity D in `docs/COMPARISON.md`) already in sciknow?"*

**Short answer: yes.** It shipped as Phase 32.7 (2026-04-11) under the name **Compound Learning Layer 1** and has been in production since. Below is a concrete walkthrough of what's there, where, and how to inspect it.

## The pattern in one diagram

```
  ┌─────────────────┐
  │ autowrite_runs  │      one row per `book autowrite` invocation
  │ autowrite_iters │      per-iteration scores + verification + revision
  │ autowrite_retr. │      per-chunk retrieval with was_cited backfill
  └────────┬────────┘
           │  (producer fires on every run completion)
           ▼
  ┌─────────────────┐
  │  LLM pass       │      _distill_lessons_from_run()
  │  "what went     │      extracts 1–3 concrete lessons per run
  │   right/wrong?" │      using the FAST model (different head than
  │                 │      the scorer — MAR confirmation-bias mitigation)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐      bge-m3 dense (1024-dim) embedding
  │ autowrite_      │      importance + dimension + score_delta
  │  lessons        │      section_slug + book_id scope
  │                 │      recency: created_at (no scheduled decay job —
  │                 │                 read path decays on the fly)
  └────────┬────────┘
           │  (consumer fires before every next autowrite run)
           ▼
  ┌─────────────────┐
  │  _get_relevant_ │      importance × recency_decay × cosine_similarity
  │  lessons()      │      cross-book lessons downweighted 0.7×
  │                 │      top-K=5 by default
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ writer prompt   │      "Lessons from prior runs" block injected
  │ (next run)      │      into the WRITE_V2_SYSTEM template
  └─────────────────┘
```

## The four artifacts, with file + line references

1. **Telemetry tables (Layer 0 — shipped Phase 32.6)**

   | Table | Migration | SQLAlchemy model | What it stores |
   |---|---|---|---|
   | `autowrite_runs` | `migrations/versions/0011_autowrite_telemetry.py` | `storage/models.py` | one row per invocation (book, chapter, section, model, max_iter, target_score, final_overall, iterations_used, converged) |
   | `autowrite_iterations` | 0011 | models.py | scores JSONB, verification JSONB, cove JSONB, KEEP/DISCARD, weakest_dimension, overall_pre, overall_post |
   | `autowrite_retrievals` | 0011 | models.py | per-chunk retrieval — and critically the `was_cited` column, backfilled by `_finalize_autowrite_run` after parsing the final draft for `[N]` markers |

2. **Lessons table (Layer 1 — shipped Phase 32.7)**

   - **Migration**: `migrations/versions/0012_autowrite_lessons.py`
   - **Schema**: `id, book_id, chapter_id, section_slug, lesson_text, source_run_id, score_delta, embedding (REAL[]), importance, dimension, created_at`
   - **Why PG not Qdrant**: lesson tables stay small (~hundreds of rows per book); cosine in Python is fast enough and avoids the Qdrant collection bookkeeping for something that's not a chunk.

3. **Producer** (runs at the tail of every autowrite completion)

   | Function | Location | What it does |
   |---|---|---|
   | `_distill_lessons_from_run(run_id)` | `sciknow/core/book_ops.py:876` | reads the run's per-iteration trajectory, prompts the *fast* model to extract 1–3 concrete lessons, calls `_persist_lesson` for each |
   | `_embed_text_for_lessons(text)` | `book_ops.py:805` | bge-m3 dense vector (same model as chunks, so same vector space) |
   | `_persist_lesson(...)` | `book_ops.py:823` | one INSERT into `autowrite_lessons`, fail-soft |

   Trigger point: `book_ops.py:755–762`, inside the autowrite run finalization path. Fail-soft — a broken lesson pass never kills an autowrite.

4. **Consumer** (runs before every new autowrite invocation)

   | Function | Location | What it does |
   |---|---|---|
   | `_get_relevant_lessons(book_id, section_slug, query_text)` | `sciknow/core/book_ops.py:1010` | embeds `query_text`, scrolls up to 200 candidate lessons for the section_slug scope, ranks by `importance × recency × cosine` (Generative Agents formula), returns top-K (default 5). Cross-book lessons downweighted 0.7× |
   | `_autowrite_section_body` (caller) | `book_ops.py:3150–3156` | emits a `lessons_loaded` event, feeds the top-K into the writer prompt's `lessons=...` parameter |
   | `rag/prompts.py` writer | `rag/prompts.py:write_section_v2` | renders the lessons block into `WRITE_V2_SYSTEM` |

## How to inspect it live

```bash
# How many lessons exist in the active project?
psql sciknow_global_cooling -c "SELECT COUNT(*) FROM autowrite_lessons;"

# Top-10 most-important lessons, most recent first
psql sciknow_global_cooling -c "
  SELECT created_at::date, section_slug, dimension, importance,
         left(lesson_text, 80) AS excerpt
  FROM autowrite_lessons
  ORDER BY importance DESC, created_at DESC LIMIT 10;
"

# Does the web reader show lessons? Yes — the autowrite SSE stream emits
# `lessons_loaded` events which the chapter/section viewer displays.
```

Quick code probe:

```python
from sciknow.core.book_ops import _get_relevant_lessons
# returns list[str] — the exact lessons that would be injected into the
# writer prompt for this (book, section, query) combination
print(_get_relevant_lessons(book_id="<uuid>",
                              section_slug="overview",
                              query_text="what drives climate sensitivity"))
```

## What this matches in the external landscape

| System | Pattern name | Where it lives in sciknow |
|---|---|---|
| AutoResearchClaw | MetaClaw (30-day recency decay) | `autowrite_lessons` + `_get_relevant_lessons` recency formula |
| DeepScientist | Findings Memory (§3.2 of arXiv:2509.26603) | same table + consumer |
| Generative Agents (Park et al. 2023) | `importance × recency × relevance` ranking | used verbatim in `_get_relevant_lessons` |
| Reflexion (Shinn et al. 2023) | verbal-memory buffer distilled between trials | the `_distill_lessons_from_run` → writer-prompt loop |
| ERL (2026) | heuristic distillation — not just concat | the "1–3 lessons per run" cap prevents ExpeL-style prompt bloat |

## What sciknow's version does *not* have yet (vs DeepScientist's Findings Memory)

Phase 46 audit surfaced three concrete gaps worth considering for Phase 47:

1. **Kind taxonomy**. DeepScientist's memory cards are typed into `{paper, idea, decision, episode, knowledge, rejected_idea}` (see `docs/COMPARISON.md` Appendix C, port analysis). sciknow's `autowrite_lessons.dimension` column is orthogonal — it records which *scorer axis* produced the lesson (groundedness / citation_accuracy / etc.), not *what kind of thing* the lesson refers to. Adding a `kind` column would let us retrieve "idea memories" and "decision memories" separately, and specifically prevent the `book gaps` generator from re-proposing rejected ideas.

2. **Scope + promote-to-global**. Today every lesson is scoped to its book; cross-book lessons are retrieved only via the section_slug join and downweighted 0.7×. DeepScientist has an explicit `scope ∈ {quest, global}` and a promote-to-global action that copies a well-performing lesson into a shared cross-project pool. For sciknow that would mean a `sciknow_lessons` table keyed to `section_slug` (no `book_id`), and a migration gate like "importance ≥ 0.8 and score_delta > 0 across ≥ 3 books".

3. **Fidelity tier on the producer**. Today `_distill_lessons_from_run` fires on every run that reaches a `run_id`. DeepScientist restricts "durable" memory writes to verified+ runs only. We could gate strong lessons (importance ≥ 0.8) to runs with `verification.groundedness_score ≥ 0.7` + `final_overall ≥ 0.7`, so low-quality runs can't pollute the memory.

Each of these is 2–3 days of work and doesn't require DGX Spark. They're tracked in `docs/COMPARISON.md` Appendix C as Port A.

## Bottom line

**D is shipped.** sciknow's compound learning is implemented, load-bearing, and matches the MetaClaw / Findings-Memory / Reflexion lineage — the external field's vocabulary differs but the mechanism is the same. The AutoResearchClaw-flavored variant (which the Phase 46 audit called "D") is fully covered. The DeepScientist extensions (kind taxonomy, scope + promotion, fidelity tiers) are concrete upgrades but were not part of opportunity D in the original ranking.
