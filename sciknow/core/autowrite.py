"""
sciknow v2 — autowrite engine.

Phase C (v2 spec §2): the autowrite iteration engine — write → score →
verify → revise → converge — and the chapter-wide loop, extracted
from ``core/book_ops.py``. ``_AutowriteLogger`` and the dozens of
small helpers (retrieval, draft persistence, scoring, CoVe verification,
plan resolution, citation alignment) stay in ``book_ops`` because they
are also used by the non-autowrite verbs (``write_section_stream``,
``review_draft_stream``, ``revise_draft_stream``, etc.). This module
imports them at module top.

Backwards compatibility: ``core/book_ops.py`` re-exports the three
public symbols at the bottom of its file (``autowrite_section_stream``,
``autowrite_chapter_all_sections_stream``, ``_autowrite_section_body``)
so callers using ``from sciknow.core.book_ops import …`` keep working.
The L1 ``inspect.getsource(book_ops.autowrite_section_stream)`` tests
still resolve — Python sets ``__module__`` to this file but the
contract greps for content patterns inside the function bodies, which
are unchanged.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from sciknow.config import settings

# Helpers + the logger live in book_ops; we import them here so the
# moved bodies can keep their original call shape. The import is at
# module top because none of these helpers depend on any of the three
# moved functions — Python parses book_ops.py fully (all 1..N helper
# defs) BEFORE the bottom-of-file `from sciknow.core.autowrite import
# …` line that triggers this module to load, so the import graph is
# acyclic at runtime even though the names cross-reference statically.
from sciknow.core.book_ops import (  # noqa: F401
    Event,
    DEFAULT_TARGET_CHAPTER_WORDS,
    LENGTH_PRIORITY_THRESHOLD,
    _DEFAULT_BOOK_SECTION_SLUGS,
    _AutowriteLogger,
    _adjust_target_for_retrieval_density,
    _auto_summarize,
    _clean_json,
    _compute_length_score,
    _count_plan_concepts,
    _cove_verify_streaming,
    _create_autowrite_run,
    _extract_draft_final_overall,
    _finalize_autowrite_run,
    _get_book_length_target,
    _get_chapter_num_sections,
    _get_chapter_flexible_length,
    _get_chapter_sections_normalized,
    _get_prior_summaries,
    _get_relevant_lessons,
    _get_section_concept_density_target,
    _get_section_model,
    _get_section_plan,
    _get_section_target_words,
    _is_resumable_draft,
    _next_draft_version,
    _persist_autowrite_iteration,
    _persist_autowrite_retrievals,
    _release_gpu_models,
    _swap_to_phase,
    _retrieve,
    _retrieve_visuals,
    _retrieve_with_step_back,
    _save_draft,
    _score_draft_inner,
    _score_visual_citation,
    _section_target_words,
    _stream_phase,
    _stream_with_save,
    _titleify_slug,
    _update_draft_content,
    _verify_draft_inner,
)

logger = logging.getLogger("sciknow.core.autowrite")


def autowrite_section_stream(
    book_id: str,
    chapter_id: str,
    section_type: str,
    *,
    model: str | None = None,
    max_iter: int = 3,
    target_score: float = 0.85,
    auto_expand: bool = False,
    use_plan: bool = True,
    use_step_back: bool = True,
    use_cove: bool = True,
    cove_threshold: float = 0.85,
    target_words: int | None = None,
    resume_from_draft_id: str | None = None,
    include_visuals: bool = False,   # Phase 54.6.142
    force_resume: bool = False,      # Phase 55.V7
) -> Iterator[Event]:
    """Full convergence loop for one section: write -> score -> revise -> re-score.

    Yields rich events for live dashboard rendering.

    Phase 24 — thin wrapper around _autowrite_section_body. This level
    just owns the per-run log file lifecycle (open at start, close in
    a finally) and yields a log_path event so the GUI can surface where
    to tail. The body generator does the actual work and gets the log
    object passed in so it can call log.stage() at every transition.

    Phase 28 — resume_from_draft_id, when set, loads the existing
    draft's content and runs the score → verify → revise loop on it
    WITHOUT redoing the initial writing phase. The draft must be in a
    finished state (checkpoint in {initial, iteration_*_keep,
    iteration_*_discard, final, draft}) — partial states from
    interrupted runs are refused via _is_resumable_draft.

    Phase 55.V7 — `force_resume=True` bypasses the partial-state
    safety gate so an interrupted iteration_*_revising or
    writing_in_progress draft can still be resumed using the
    partially-written content. Risk: the partial content may end
    mid-sentence; the convergence loop will smooth that out in the
    next revision pass.

    Tail the log with:

        tail -f data/autowrite/latest.jsonl | jq

    The heartbeat thread inside the logger writes a "still here" line
    every 30 seconds even when the LLM is silent, so stalls in
    retrieval / model loading / inter-stage gaps are visible
    immediately instead of after the user reports a stuck run.
    """
    log = _AutowriteLogger(book_id, chapter_id, section_type)
    yield {"type": "log_path", "path": str(log.path)}
    try:
        yield from _autowrite_section_body(
            log,
            book_id, chapter_id, section_type,
            model=model, max_iter=max_iter, target_score=target_score,
            auto_expand=auto_expand, use_plan=use_plan,
            use_step_back=use_step_back, use_cove=use_cove,
            cove_threshold=cove_threshold, target_words=target_words,
            resume_from_draft_id=resume_from_draft_id,
            include_visuals=include_visuals,
            force_resume=force_resume,
        )
    except (GeneratorExit, KeyboardInterrupt):
        # Caller asked us to stop — propagate cleanly (the finally
        # block flushes the log and emits the `end` event).
        raise
    except Exception as exc:
        # Phase 55.V6 (2026-04-28) — surface silent / catastrophic
        # section failures as a `section_error` event yielded to the
        # consumer, instead of letting the exception propagate and
        # produce a "0-token end" log entry with no actionable signal.
        # Real cases caught in a full-book run on 2026-04-27/28:
        #   - planning + writing prompts both overflowed the writer's
        #     context window (16K) → HTTP 400, _stream_with_save
        #     raised → 0-token writing + 0-token end, no completed
        #     event. The CLI's per-section try/except (Phase 55.V4)
        #     caught the propagated exception, but only AFTER
        #     `_stream_with_save` had already exited cleanly via its
        #     `finally: _flush()`, so the error never made it back
        #     to the consumer with a recognisable shape. Emitting the
        #     event here gives every downstream consumer (CLI live
        #     dashboard, web SSE, log scrapers) a single uniform
        #     signal: ``{"type": "section_error", "error_type": ...,
        #     "message": ...}``. The `book autowrite --full` runner
        #     can keep going to the next section without a Python
        #     traceback in the user's terminal.
        log.event(
            "section_error",
            error_type=type(exc).__name__,
            message=str(exc)[:500],
        )
        yield {
            "type": "section_error",
            "error_type": type(exc).__name__,
            "message": str(exc)[:500],
        }
        # Don't re-raise — consumer can continue with the next section.
    finally:
        log.close()


def _autowrite_section_body(
    log: "_AutowriteLogger",
    book_id: str,
    chapter_id: str,
    section_type: str,
    *,
    model: str | None = None,
    max_iter: int = 3,
    target_score: float = 0.85,
    auto_expand: bool = False,
    use_plan: bool = True,
    use_step_back: bool = True,
    use_cove: bool = True,
    cove_threshold: float = 0.85,
    target_words: int | None = None,
    resume_from_draft_id: str | None = None,
    include_visuals: bool = False,   # Phase 54.6.142
    force_resume: bool = False,      # Phase 55.V7
) -> Iterator[Event]:
    """Phase 24 — the actual autowrite implementation, extracted from
    autowrite_section_stream so the public function can wrap it in a
    log lifecycle without indenting ~400 lines of body code.

    Receives the live _AutowriteLogger and calls log.stage(...) at
    every major transition. Token observers are passed into
    _stream_with_save so the heartbeat can show real tokens-per-second.

    Phase 28 — resume_from_draft_id, when set, loads the existing
    draft's content as the starting point and SKIPS the initial
    writing stream. The source draft must pass _is_resumable_draft
    (no partial states from interrupted runs); otherwise we yield
    an error and return without doing any LLM work.
    """
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream, complete as llm_complete
    from sciknow.rag.llm import warm_up as _llm_warm_up
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    qdrant = get_client()

    # Phase 54.6.31 — warm the writer model into VRAM before the
    # score/verify/revise loop starts so the first iteration doesn't
    # pay cold-start latency. Best-effort; silently continues if the
    # Ollama server is unreachable (autowrite will fail naturally on
    # the real call with a clearer error).
    #
    # Phase 54.6.305 — resolve to ``book_write_model`` when no explicit
    # model is passed. Previously this fell back to ``llm_model`` (the
    # MoE default), which at keep_alive=-1 pinned the *wrong* ~18 GB
    # model in VRAM. The real write call a few lines later then
    # resolves to ``book_write_model`` (via line ~3108 / 4348) and
    # either swaps (wasteful) or runs out of VRAM when the embedder
    # loads for retrieval (OOM at SentenceTransformer.to('cuda')).
    # Pinning the right model up front avoids both traps and matches
    # the writer/scorer/verifier model resolution used elsewhere.
    from sciknow.config import settings as _s
    _warm_model = model or _s.book_write_model or _s.llm_model
    _llm_warm_up(model=_warm_model, num_ctx=16384, num_batch=1024)

    # Phase 54.6.59 — scorer-role resolution. Explicit `--model` beats
    # everything (the user asked for one model end-to-end). Otherwise,
    # route scoring + rescoring through AUTOWRITE_SCORER_MODEL if set,
    # else fall through to the writer (which the model_info event
    # advertises as "writing/scoring/verification/CoVe — flagship").
    #
    # Phase 54.6.306 — the fallback used to be ``None`` → llm.stream
    # default → ``settings.llm_model``.  That was fine pre-54.6.243
    # when llm_model WAS the writer, but the writer has since moved
    # to ``book_write_model``.  Falling back to llm_model forced
    # Ollama to swap between the writer (22 GB, pinned by
    # keep_alive=-1) and the general model the moment scoring kicked
    # in — on 24 GB that swap failed with "model failed to load,
    # status code 500" because both sticky models wanted to stay
    # resident.  Matching the writer here keeps the single LLM hot
    # across writing → scoring → verification → CoVe → revision with
    # zero swaps.
    if model is not None:
        scorer_model = model
    elif _s.autowrite_scorer_model:
        scorer_model = _s.autowrite_scorer_model
    elif _s.book_write_model:
        scorer_model = _s.book_write_model
    else:
        scorer_model = None

    log.stage("loading_book")
    # Load book/chapter data
    with get_session() as session:
        from sqlalchemy import text
        book = session.execute(text("""
            SELECT id::text, title, plan FROM books WHERE id::text = :bid
        """), {"bid": book_id}).fetchone()
        ch = session.execute(text("""
            SELECT id::text, number, title, description, topic_query, topic_cluster, sections
            FROM book_chapters WHERE id::text = :cid
        """), {"cid": chapter_id}).fetchone()

    if not book or not ch:
        log.event("error", message="book or chapter not found")
        yield {"type": "error", "message": "Book or chapter not found."}
        return

    b_title, b_plan = book[1], book[2]
    ch_id, ch_num, ch_title, ch_desc, topic_query, topic_cluster, ch_sections = ch
    topic = topic_query or ch_title

    # Phase 55.V19 — resolve the section's human-readable title and
    # description from the chapter's `sections` JSON outline. The
    # retrieval query at line ~471 used to be `f"{section_type} {topic}"`
    # where section_type is the slug (`the_science_of_sunspots`). The
    # underscores hurt sparse keyword matching, and the slug doesn't
    # carry the topical intent — `the_science_of_sunspots` retrieves
    # different chunks than `The Science of Sunspots magnetic flux`.
    # Resolved with debug_verifier.py 2026-04-28: writer fabricated 34
    # citations against header-only chunks because retrieval surfaced
    # them. Better query → better evidence → less fabrication pressure.
    _section_title: str | None = None
    _section_description: str | None = None
    try:
        secs = ch_sections
        if isinstance(secs, str):
            import json as _json
            secs = _json.loads(secs) if secs else []
        if isinstance(secs, list):
            for _s in secs:
                if isinstance(_s, dict) and _s.get("slug") == section_type:
                    _section_title = _s.get("title")
                    _section_description = _s.get("description")
                    break
    except Exception as _exc:
        logger.debug("section title resolution failed for %s: %s",
                     section_type, _exc)
    # Slug→spaces fallback so a misconfigured outline still produces
    # a usable query (e.g. "the science of sunspots" beats
    # "the_science_of_sunspots" for sparse matching).
    section_title_for_query = (
        _section_title
        or section_type.replace("_", " ").strip()
    )

    # Phase 28 — load resume source NOW (before any LLM calls) so we
    # can fail fast if it's in a partial state. resume_content stays
    # None for fresh writes (the existing path). The eligibility check
    # uses _is_resumable_draft which refuses writing_in_progress,
    # iteration_*_revising, placeholder, and unknown checkpoints.
    resume_content: str | None = None
    resume_source_meta: dict | None = None
    if resume_from_draft_id:
        with get_session() as session:
            row = session.execute(text("""
                SELECT id::text, content, word_count, custom_metadata, section_type
                FROM drafts
                WHERE (id::text = :did OR id::text LIKE :prefix)
                  AND book_id::text = :bid
                  AND chapter_id::text = :cid
                LIMIT 1
            """), {
                "did": resume_from_draft_id,
                "prefix": f"{resume_from_draft_id}%",
                "bid": book_id, "cid": chapter_id,
            }).fetchone()
        if not row:
            msg = (
                f"resume source draft {resume_from_draft_id!r} not found "
                f"in this book/chapter"
            )
            log.event("error", message=msg)
            yield {"type": "error", "message": msg}
            return

        ok, reason = _is_resumable_draft(row[3], row[2])
        if not ok and not force_resume:
            msg = f"cannot resume from draft {row[0][:8]}: {reason}"
            log.event("resume_refused", reason=reason, draft_id=row[0])
            yield {"type": "error", "message": msg}
            return
        if not ok and force_resume:
            # Phase 55.V7 — caller passed force_resume=True; treat the
            # partial draft as a valid resume base. Log the override so
            # post-mortem can correlate any odd output with the bypass.
            log.event(
                "resume_force_overridden",
                reason=reason, draft_id=row[0],
                word_count=int(row[2] or 0),
            )

        resume_content = row[1] or ""
        resume_source_meta = {
            "source_draft_id": row[0],
            "source_word_count": int(row[2] or 0),
            "source_section_type": row[4],
            "resume_started_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint_warning": reason or None,
        }
        log.event(
            "resume_loaded",
            source_draft_id=row[0],
            source_word_count=int(row[2] or 0),
            warning=reason or None,
        )
        yield {
            "type": "resume_info",
            "source_draft_id": row[0],
            "source_word_count": int(row[2] or 0),
            "warning": reason or None,
        }

    # Phase 17 / 29 — resolve effective per-section length target.
    # 3-level priority:
    #   1. caller arg (target_words kwarg)
    #   2. per-section meta override (_get_section_target_words, Phase 29)
    #   3. derived from chapter target / num_sections (Phase 17 default)
    #
    # Phase 18 — also resolve the per-section plan in the same pass.
    with get_session() as session:
        if target_words is not None:
            effective_target_words = target_words
        else:
            section_override = _get_section_target_words(session, ch_id, section_type)
            if section_override is not None:
                effective_target_words = section_override
            else:
                # Phase 54.6.146 — Level 0 concept-density: when the
                # section has a plan with N bullets, target = N × wpc
                # from the project type. Bottom-up sizing; chapter/book
                # length emerges from section lengths per RESEARCH.md §24.
                concept_target = _get_section_concept_density_target(
                    session, ch_id, section_type, book_id,
                )
                if concept_target is not None:
                    effective_target_words = concept_target
                else:
                    # Phase 54.6.143 — pass chapter_id so per-chapter override fires
                    chapter_target = _get_book_length_target(session, book_id, chapter_id=ch_id)
                    num_sections = _get_chapter_num_sections(session, ch_id)
                    effective_target_words = _section_target_words(chapter_target, num_sections)
        section_plan = _get_section_plan(session, ch_id, section_type)

        # Phase 37 — per-section model override. Applied only when no
        # explicit caller model was passed (CLI --model / web form).
        # Covers every LLM call in the autowrite loop — writer, scorer,
        # verifier, CoVe — so a section tagged with the flagship model
        # gets consistent scoring too. Users who want to cut scorer
        # cost specifically can still set LLM_FAST_MODEL and leave the
        # section override unset.
        if model is None:
            section_model_override = _get_section_model(session, ch_id, section_type)
            if section_model_override:
                model = section_model_override

    # Phase 14.6 — Surface which model is going to do the writing so the
    # user never has to ask. resolved_model is what `llm.complete()` will
    # default to when called with model=None below; this mirrors the
    # `chosen_model = model or settings.llm_model` line in rag/llm.py.
    # Phase 54.6.243 — prefer `book_write_model` over the global
    # `llm_model` when set. The autowrite scorer/verifier/CoVe share
    # this model (see Phase 37 section-override logic above), which
    # matches the 54.6.243 bench showing qwen3.6:27b-dense writes
    # richer, better-cited prose than qwen3:30b-a3b. Assign back to
    # `model` (not just resolved_model) so every downstream
    # llm.stream()/complete() call below sees the override — passing
    # model=None into rag/llm.py would re-default to settings.llm_model
    # and silently ignore book_write_model.
    from sciknow.config import settings
    if model is None and settings.book_write_model:
        model = settings.book_write_model
    resolved_model = model or settings.llm_model

    # Phase 55.S1 — when the cross-family scorer is configured, score
    # / verify / CoVe / rescore route through it, not the writer. The
    # GUI's model_info renderer uses these fields to label the active
    # roles correctly so users don't see "scoring on writer" when a
    # gemma scorer is doing the work.
    _scorer_configured = (
        getattr(settings, "use_llamacpp_scorer", False)
        and bool(getattr(settings, "scorer_model_gguf", ""))
    )
    _scorer_model_label = (
        getattr(settings, "scorer_model_name", "scorer")
        if _scorer_configured else None
    )

    yield {
        "type": "model_info",
        "writer_model": resolved_model,
        "fast_model": settings.llm_fast_model,
        "scorer_model": _scorer_model_label,
        "writer_role": (
            "writing / revising"
            if _scorer_configured
            else "writing / scoring / verification / CoVe (flagship)"
        ),
        "scorer_role": (
            "scoring / verification / CoVe (cross-family)"
            if _scorer_configured else None
        ),
        "fast_role": "step-back retrieval (utility)",
    }
    yield {"type": "length_target", "target_words": effective_target_words}
    yield {"type": "progress", "stage": "setup",
           "detail": (
               f"Autowrite Ch.{ch_num}: {ch_title} — {section_type.capitalize()} · "
               f"model: {resolved_model} · target: ~{effective_target_words} words"
           )}

    # Step 1: Initial draft
    log.stage("retrieval")
    # Phase 55.V19 — section title (+ optional description) replace the
    # slug `section_type` in the retrieval query. See the title-resolve
    # block above for why.
    _retrieval_query = f"{section_title_for_query} {topic}"
    if _section_description:
        _retrieval_query = f"{_retrieval_query} {_section_description}"
    with get_session() as session:
        prior_summaries = _get_prior_summaries(session, book_id, ch_num)
        results, sources = _retrieve_with_step_back(
            session, qdrant, _retrieval_query,
            topic_cluster=topic_cluster, model=model,
            use_step_back=use_step_back,
        )

    if not results:
        log.event("error", message="no relevant passages found")
        yield {"type": "error", "message": "No relevant passages found."}
        return

    log.event("retrieval_done", n_results=len(results), n_sources=len(sources))

    # Phase 54.6.150 — retrieval-density widener (RESEARCH.md §24 §4).
    # When the section's target came from concept-density (section had
    # a plan with N bullets × wpc_midpoint), adjust wpc based on the
    # retrieved chunk count: more chunks → more words needed to cover
    # the evidence, fewer → don't pad. Lerp within the project type's
    # wpc_range so the adjustment is bounded. No-op when the target
    # came from an explicit override or the top-down chapter split
    # (honest signal only when the bottom-up path fired in the first
    # place).
    if target_words is None and resume_content is None:
        try:
            from sqlalchemy import text as _sql
            from sciknow.core.project_type import get_project_type
            with get_session() as _s:
                _plan = _get_section_plan(_s, ch_id, section_type)
                _override = _get_section_target_words(_s, ch_id, section_type)
                _bt_row = _s.execute(_sql(
                    "SELECT book_type FROM books WHERE id::text = :b LIMIT 1"
                ), {"b": book_id}).fetchone()
            _n_concepts = _count_plan_concepts(_plan)
            _book_type = (_bt_row[0] if _bt_row else None) or None
            # Only widen when the concept-density path resolved the target
            # (override absent AND plan present).
            if _override is None and _n_concepts > 0:
                _pt = get_project_type(_book_type)
                _new_target, _lerp, _explain = _adjust_target_for_retrieval_density(
                    effective_target_words,
                    _n_concepts,
                    len(results),
                    _pt.words_per_concept_range,
                )
                if _new_target != effective_target_words:
                    log.event(
                        "retrieval_density_adjust",
                        base=effective_target_words,
                        new=_new_target,
                        n_chunks=len(results),
                        n_concepts=_n_concepts,
                        lerp=round(_lerp, 3),
                    )
                    yield {
                        "type": "retrieval_density_adjust",
                        "base_target": effective_target_words,
                        "new_target": _new_target,
                        "n_chunks": len(results),
                        "n_concepts": _n_concepts,
                        "lerp": round(_lerp, 3),
                        "explanation": _explain,
                    }
                    logger.info(
                        "Phase 54.6.150 — %s/%s: %s",
                        str(ch_id)[:8], section_type, _explain,
                    )
                    effective_target_words = _new_target
        except Exception as exc:  # noqa: BLE001
            logger.debug("retrieval_density_adjust skipped: %s", exc)

    # Phase 54.6.x — flexible-length per-chapter opt-in. When the
    # chapter row has flexible_length=TRUE AND the retrieval pool is
    # rich enough (≥ FLEXIBLE_RICH_THRESHOLD chunks), the writer is
    # allowed to extend up to 2× the base target. The base target
    # remains the SCORING anchor (length score = min(1, actual /
    # target)) so a flexible chapter at 1× target still scores 1.0;
    # only the writer's prompt sees the larger ceiling. This is what
    # the user asked for: "only bigger, at most the double, only if
    # the corpus is good enough".
    FLEXIBLE_RICH_THRESHOLD = 24
    flexible_max_words: int | None = None
    try:
        with get_session() as _s_flex:
            if _get_chapter_flexible_length(_s_flex, ch_id):
                if len(results) >= FLEXIBLE_RICH_THRESHOLD:
                    flexible_max_words = int(effective_target_words * 2)
                    log.event(
                        "flexible_length_enabled",
                        base=effective_target_words,
                        max=flexible_max_words,
                        n_chunks=len(results),
                    )
                    yield {
                        "type": "flexible_length_enabled",
                        "base_target": effective_target_words,
                        "max_target": flexible_max_words,
                        "n_chunks": len(results),
                        "explanation": (
                            f"chapter is flexible-length and retrieval is "
                            f"rich ({len(results)} chunks ≥ "
                            f"{FLEXIBLE_RICH_THRESHOLD}); writer may extend "
                            f"to {flexible_max_words} words if evidence "
                            f"supports it"
                        ),
                    }
                else:
                    log.event(
                        "flexible_length_skipped",
                        base=effective_target_words,
                        n_chunks=len(results),
                        threshold=FLEXIBLE_RICH_THRESHOLD,
                        reason="retrieval too thin",
                    )
    except Exception as exc:  # noqa: BLE001
        logger.debug("flexible_length resolve skipped: %s", exc)

    # Phase 54.6.151 — digital-section soft ceiling (RESEARCH.md §24
    # Guideline 3, Delgado 2018 screen-reading comprehension penalty).
    # Checks the FINAL effective target (after optional 54.6.150 widener)
    # and emits a warning when it exceeds the comfort band. Non-blocking:
    # autowrite still proceeds at the requested target. Threshold is
    # 3,000 for most types (digital expository text) but lifted to
    # 5,000 for academic_monograph because monograph sections can
    # legitimately sit in the 3k-4k band per RESEARCH.md §24 (reader
    # is an expert; chunk templates inflate effective capacity per
    # Gobet & Clarkson 2004).
    _SOFT_CEILING_DEFAULT = 3000
    _SOFT_CEILING_BY_TYPE = {
        "academic_monograph": 5000,  # research-grade pedagogy
    }
    try:
        from sqlalchemy import text as _sql2
        with get_session() as _s2:
            _bt_row2 = _s2.execute(_sql2(
                "SELECT book_type FROM books WHERE id::text = :b LIMIT 1"
            ), {"b": book_id}).fetchone()
        _book_type2 = (_bt_row2[0] if _bt_row2 else None) or "scientific_book"
        _soft_ceiling = _SOFT_CEILING_BY_TYPE.get(_book_type2, _SOFT_CEILING_DEFAULT)
        if effective_target_words > _soft_ceiling:
            _msg = (
                f"target_words={effective_target_words:,} exceeds the "
                f"{_soft_ceiling:,}-word digital-comfort ceiling for book "
                f"type {_book_type2!r} (Delgado 2018 screen penalty; "
                f"RESEARCH.md §24 guideline 3). Consider splitting the "
                f"section for better absorption."
            )
            log.event(
                "section_length_warning",
                target=effective_target_words,
                soft_ceiling=_soft_ceiling,
                book_type=_book_type2,
            )
            yield {
                "type": "section_length_warning",
                "target": effective_target_words,
                "soft_ceiling": _soft_ceiling,
                "book_type": _book_type2,
                "explanation": _msg,
            }
            logger.info("Phase 54.6.151 — %s/%s: %s",
                        str(ch_id)[:8], section_type, _msg)
    except Exception as exc:  # noqa: BLE001
        logger.debug("section_length_warning check skipped: %s", exc)

    # Phase 32.6 — Compound learning Layer 0: open the autowrite_runs row
    # NOW (after retrieval succeeded) and persist the retrieval set with
    # source_position = SearchResult.rank. The run_id is held in a local
    # so the iteration loop and the final finalize block can reference
    # it. All telemetry helpers fail soft — a SQL error here logs and
    # returns None, the autowrite generator continues without telemetry.
    autowrite_run_id = _create_autowrite_run(
        book_id=book_id,
        chapter_id=ch_id,
        section_slug=section_type,
        model=resolved_model,
        target_words=effective_target_words,
        max_iter=max_iter,
        target_score=target_score,
        feature_versions={
            "use_plan": use_plan,
            "use_step_back": use_step_back,
            "use_cove": use_cove,
            "cove_threshold": cove_threshold,
            "phase7_hedging_fidelity": True,
            "phase8_entity_bridge": True,
            "phase9_pdtb_discourse_relations": True,
            "phase10_step_back_retrieval": True,
            "phase11_chain_of_verification": True,
            "phase12_raptor_retrieval": True,
        },
    )
    _persist_autowrite_retrievals(autowrite_run_id, results)
    # Track lifecycle outcome for the finally-block finalization. These
    # are mutated by the iteration loop and read at cleanup time.
    _telemetry_status = "running"
    _telemetry_error = None
    _telemetry_converged = False
    _telemetry_final_overall: float | None = None
    _telemetry_final_draft_id: str | None = None
    _telemetry_iterations_used = 0

    # Phase 28 — in resume mode, skip planning + writing entirely.
    # The starting content comes from the existing draft. We still
    # ran retrieval above so the score / verify / revise stages
    # have fresh source passages to evaluate against.
    paragraph_plan = None
    if resume_content is None and use_plan:
        log.stage("planning")
        yield {"type": "progress", "stage": "planning",
               "detail": "Building paragraph plan with discourse relations..."}
        # Phase 55.V1 — swap to generate phase before the writer-side
        # tree-plan call. Retrieval just left the embedder + reranker
        # hot (~1.8 GB); evict them so the writer's planning call has
        # the full 24 GB to work with.
        try:
            _swap_to_phase("generate")
        except Exception:
            pass
        try:
            sys_p, usr_p = rag_prompts.tree_plan(
                section_type, topic, results,
                book_plan=b_plan, prior_summaries=prior_summaries,
                target_words=effective_target_words,
                section_plan=section_plan,
            )
            # Phase 15.3 — stream so the token counter sees activity
            # Phase 24 — token_observer feeds the heartbeat's t/s stat
            # Phase 55.V6 (2026-04-28) — `format="json"` enforces
            # OpenAI response_format=json_object server-side. Without
            # it, Qwen3.6 routinely truncated the JSON mid-output
            # around char 3500–5000 (the writer just stopped before
            # closing brackets); ~10 sections in a full-book run
            # caught the truncation pattern. The JSON-object mode
            # makes the server constrain decoding to valid JSON, so
            # if the model ever wants to stop early it must close
            # the structure first. `_clean_json` salvage path retained
            # below as a belt-and-braces safety net.
            plan_raw = yield from _stream_phase(
                sys_p, usr_p, "planning",
                model=model, temperature=0.2, num_ctx=16384,
                token_observer=log.token, format="json",
            )
            plan_clean = re.sub(r'<think>.*?</think>\s*', '', plan_raw,
                                flags=re.DOTALL).strip()
            plan_data = json.loads(_clean_json(plan_clean), strict=False)
            paragraph_plan = plan_data.get("paragraphs") or []
            yield {"type": "tree_plan", "data": plan_data}
        except Exception as exc:
            logger.warning("Tree-plan failed in autowrite, continuing without: %s", exc)
            log.event("planning_failed", message=str(exc)[:200])
            paragraph_plan = None

    # Phase 32.7 — Layer 1: fetch top-K relevant lessons from past runs
    # for this section_slug. The query text combines the section type,
    # topic, and section plan so the embedding similarity ranks lessons
    # that addressed the same kind of writing challenge above ones from
    # an unrelated section. Returns [] on cold-start (no lessons yet).
    _lessons_query = " ".join(filter(None, [
        section_type, topic, (section_plan or "")[:500],
    ]))
    # Phase 47.3 — return kind-dicts so the writer prompt can group
    # lessons by kind (knowledge / idea / decision / paper / episode).
    # Exclude ``rejected_idea`` — that kind is for the gap-finder gate
    # (Phase 47.2), NOT for the writer (the writer shouldn't try to
    # actively AVOID writing about a topic; that's a planner-level gate).
    # Phase 55.V1 — lessons fetch embeds via bge-m3; needs the
    # embedder up. Tree-plan above left the writer hot; swap now.
    try:
        _swap_to_phase("retrieve")
    except Exception:
        pass
    relevant_lessons = _get_relevant_lessons(
        book_id, section_type, _lessons_query,
        kinds=("knowledge", "idea", "decision", "paper", "episode"),
        return_dicts=True,
    )
    # Phase 54.6.305 — the lessons fetch embeds `_lessons_query` through
    # the same bge-m3 + dense-embedder path retrieval uses, which
    # **reloads** both embedders into PyTorch after the
    # _retrieve_with_step_back release.  Without this second release,
    # ~9 GB of dense-embedder tensors linger in the Python CUDA cache
    # and Ollama partial-loads the writer (vram=12.6 GB / total=22 GB)
    # when it reloads for the writing stage, dropping decode from
    # 30+ t/s to ~4 t/s.  This call returns the GPU to a clean state
    # so the writer gets the full 22 GB it needs for 16k context.
    _release_gpu_models()
    # Phase 55.V1 — declarative phase swap. The lessons fetch is the
    # last retrieval-side work in the autowrite preamble; everything
    # below (writer warm-up, initial draft) is generation. Activate
    # the generate phase so the embedder + reranker llama-server
    # processes are evicted, freeing ~1.8 GB of VRAM for the writer
    # to use 16K context at full decode speed.
    _swap_to_phase("generate")
    if relevant_lessons:
        by_kind: dict[str, int] = {}
        for l in relevant_lessons:
            by_kind[l["kind"]] = by_kind.get(l["kind"], 0) + 1
        breakdown = ", ".join(f"{k}={n}" for k, n in sorted(by_kind.items()))
        yield {"type": "progress", "stage": "lessons",
               "detail": (
                   f"Layer 1: injecting {len(relevant_lessons)} lesson(s) "
                   f"from prior runs — {breakdown}"
               )}
        log.event("lessons_loaded",
                  count=len(relevant_lessons), by_kind=by_kind)

    # Phase 32.10 — Layer 5: fetch the book's style fingerprint, if it
    # exists. Returns None on cold-start (no approved drafts yet).
    # Independent of Layer 1: even when there are zero lessons, an
    # established style fingerprint can still anchor the writer's
    # voice to the user's prior accepted work.
    style_fingerprint_block = ""
    try:
        from sciknow.core.style_fingerprint import (
            get_style_fingerprint, format_fingerprint_for_prompt,
        )
        _fp = get_style_fingerprint(book_id) if book_id else None
        if _fp:
            style_fingerprint_block = format_fingerprint_for_prompt(_fp)
            yield {"type": "progress", "stage": "style",
                   "detail": f"Layer 5: injecting style fingerprint ({_fp.get('n_drafts_sampled', 0)} drafts)"}
            log.event("style_fingerprint_loaded",
                      n_drafts=_fp.get("n_drafts_sampled", 0))
    except Exception as exc:
        # Fail-soft: any error in fingerprint reads must NEVER block
        # the autowrite — Layer 5 is purely additive.
        logger.warning("style fingerprint load failed: %s", exc)

    # Phase 21.e — query the visuals table for tables/equations/figures
    # related to this section's topic. Inject into the writer prompt so
    # the LLM can incorporate them inline. Best-effort: if the visuals
    # table doesn't exist yet (fresh install without extract-visuals),
    # pass an empty list and the writer works as before.
    section_visuals: list[dict] = []
    try:
        with get_session() as session:
            from sqlalchemy import text as _vtext
            vis_rows = session.execute(_vtext("""
                SELECT kind, content, caption, figure_num, surrounding_text
                FROM visuals v
                JOIN documents d ON v.document_id = d.id
                WHERE d.ingestion_status = 'complete'
                  AND (v.kind IN ('table', 'equation', 'figure'))
                  AND (v.caption ILIKE :pat OR v.surrounding_text ILIKE :pat)
                ORDER BY v.kind, v.created_at
                LIMIT 8
            """), {"pat": f"%{(topic or '')[:50]}%"}).fetchall()
            section_visuals = [
                {"kind": r[0], "content": r[1], "caption": r[2],
                 "figure_num": r[3], "surrounding_text": r[4]}
                for r in vis_rows
            ]
    except Exception:
        pass  # visuals table may not exist — fail-soft

    # Phase 54.6.142 — opt-in visuals-in-writer integration. Runs the
    # 5-signal ranker (Phase 54.6.139) on the writing query using the
    # already-retrieved chunks' document_ids as `cited_doc_ids`, renders
    # a writer-facing block via format_visuals_prompt_block, and lets
    # write_section_v2 splice it into the user prompt while the paired
    # `visual_citation_block` rhetorically-gated instruction (Option D
    # from docs/RESEARCH.md §7.X) lands in the system prompt.
    ranked_visuals_for_writer: list = []
    visuals_prompt_block_text: str | None = None
    if include_visuals and results:
        try:
            cited_doc_ids = list({getattr(r, "document_id", "") for r in results
                                  if getattr(r, "document_id", "")})
            ranked_visuals_for_writer = _retrieve_visuals(
                topic or "",
                cited_doc_ids=cited_doc_ids,
                section_type=section_type,
                top_k=5,
            )
            if ranked_visuals_for_writer:
                visuals_prompt_block_text = rag_prompts.format_visuals_prompt_block(
                    ranked_visuals_for_writer,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("autowrite visuals retrieval failed (continuing without): %s", exc)
            ranked_visuals_for_writer = []
            visuals_prompt_block_text = None

    if resume_content is None:
        system, user = rag_prompts.write_section_v2(
            section_type, topic, results,
            book_plan=b_plan, prior_summaries=prior_summaries,
            paragraph_plan=paragraph_plan,
            target_words=effective_target_words,
            target_words_max=flexible_max_words,
            section_plan=section_plan,
            lessons=relevant_lessons,
            style_fingerprint_block=style_fingerprint_block,
            visuals=section_visuals,
            visuals_prompt_block=visuals_prompt_block_text,
        )
        n_visuals_msg = (
            f", {len(ranked_visuals_for_writer)} ranked figures"
            if ranked_visuals_for_writer else ""
        )
        yield {"type": "progress", "stage": "writing",
               "detail": f"Generating initial draft (~{effective_target_words} words, "
                         f"{len(section_visuals)} legacy visuals{n_visuals_msg})..."}
    else:
        # Resume mode — fast-path past the writing phase. The score
        # loop below will pick up `content` and run iterations on it.
        yield {"type": "progress", "stage": "resume",
               "detail": (
                   f"Resuming from existing draft "
                   f"({resume_source_meta['source_word_count']} words) — "
                   f"skipping initial writing, jumping straight to scoring."
               )}

    # Phase 19 — INSERT the placeholder draft BEFORE the writing loop.
    # This is what makes Stop-during-writing recoverable: we have a
    # row to UPDATE on every periodic save. Two safety properties:
    #
    #   1) version = max(existing for this section) + 1, so this new
    #      row always wins the GUI's "latest version per section" sort.
    #      Without this, an interrupted autowrite shows an OLDER draft
    #      after refresh — exactly the regression the user reported as
    #      "we reverted to an older version".
    #   2) Created with empty content; the writing loop's _stream_with_save
    #      will populate it incrementally and the try/finally guarantees
    #      a final flush even on Stop / GeneratorExit.
    draft_title = f"Ch.{ch_num} {ch_title} — {section_type.capitalize()} (autowrite)"
    feature_versions = {
        "use_plan": use_plan,
        "use_step_back": use_step_back,
        "use_cove": use_cove,
        "cove_threshold": cove_threshold,
        "phase7_hedging_fidelity": True,
        "phase8_entity_bridge": True,
        "phase9_pdtb_discourse_relations": True,
        "phase10_step_back_retrieval": True,
        "phase11_chain_of_verification": True,
        "phase12_raptor_retrieval": True,
    }
    history: list[dict] = []
    placeholder_metadata = {
        "score_history": history,
        "feature_versions": feature_versions,
        "final_overall": None,
        "max_iter": max_iter,
        "target_score": target_score,
        "target_words": effective_target_words,
        # Phase 28 — start state depends on whether we're resuming.
        # For a fresh write the placeholder is "writing_in_progress"
        # (filled in by the writing stream). For resume the
        # placeholder already has its full starting content, so we
        # mark it "initial" immediately.
        "checkpoint": "initial" if resume_content is not None else "writing_in_progress",
    }
    if resume_source_meta is not None:
        placeholder_metadata["resume_source"] = resume_source_meta
    with get_session() as session:
        draft_version = _next_draft_version(
            session, book_id, ch_id, section_type
        )
        draft_id = _save_draft(
            session, title=draft_title, book_id=book_id, chapter_id=ch_id,
            section_type=section_type, topic=topic,
            content=resume_content if resume_content is not None else "",
            sources=sources, model=model, summary=None,
            version=draft_version,
            custom_metadata=placeholder_metadata,
        )
    # Phase 32.6 — link this draft to the run row immediately so a
    # cancelled-mid-write run still has a recoverable draft pointer
    # (the periodic sweep can later flip its status to 'cancelled').
    _telemetry_final_draft_id = draft_id
    yield {"type": "checkpoint", "draft_id": draft_id,
           "stage": "resume_initial" if resume_content is not None else "placeholder",
           "word_count": len((resume_content or "").split())}
    log.event("placeholder_saved", draft_id=draft_id, version=draft_version,
              resume=resume_content is not None)

    if resume_content is None:
        # Phase 19 — incremental save during the writing stream. The
        # callback closure references draft_id + placeholder_metadata so
        # every flush goes to the same row. _stream_with_save flushes
        # every ~150 tokens or 5 seconds AND in a finally block on Stop.
        def _save_writing(text: str) -> None:
            _update_draft_content(
                draft_id, text,
                custom_metadata={**placeholder_metadata, "checkpoint": "writing_in_progress"},
            )

        log.stage("writing")
        content = yield from _stream_with_save(
            system, user, "writing",
            model=model, save_callback=_save_writing,
            token_observer=log.token,
        )

        # Promote the just-finished initial draft from "writing_in_progress"
        # to "initial". Same draft row, just updated metadata + final flush
        # to lock in the post-loop content (the streaming saves may have
        # been racing the loop's last few tokens).
        initial_metadata = {
            "score_history": history,
            "feature_versions": feature_versions,
            "final_overall": None,
            "max_iter": max_iter,
            "target_score": target_score,
            "target_words": effective_target_words,
            "checkpoint": "initial",
        }
        _update_draft_content(
            draft_id, content,
            custom_metadata=initial_metadata,
        )
        yield {"type": "checkpoint", "draft_id": draft_id, "stage": "initial",
               "word_count": len(content.split())}
        log.event("initial_draft_saved", word_count=len(content.split()))
    else:
        # Phase 28 resume path — content was loaded from the source
        # draft above. Skip straight to the iteration loop. The new
        # row already has resume_content baked in via _save_draft.
        content = resume_content
        log.event("resume_skipped_writing", word_count=len(content.split()))

    # Phase 32.6 — Layer 0: per-iteration delta tracker. Initialized
    # to the post-writing word count and updated on every KEEP verdict
    # so the next iteration's word_count_delta is computed against
    # the kept content, not a discarded revision.
    _telemetry_prev_word_count = len(content.split())

    # Phase 53 — autoreason "four conditions" refinement gate.
    # Advisory-only for this release: we emit the gate's recommendation
    # as an event + log line, but the revision loop still runs
    # regardless. After a release cycle of observation we can switch to
    # a hard skip. Gate inputs:
    #   - num_retrieval_hits: how much external verification signal we
    #     have for the scorer's groundedness dimension to actually work.
    #   - has_explicit_outline: True when use_plan triggered a sentence-
    #     plan or a chapter outline was available.
    from sciknow.core.refinement_gate import should_run_refinement
    _gate = should_run_refinement(
        section_type=section_type,
        target_words=target_words,
        num_retrieval_hits=len(results or []),
        has_explicit_outline=bool(use_plan),
    )
    if not _gate.recommend_refinement:
        reason_str = "; ".join(f"{n}: {m}" for n, m in _gate.reasons)
        yield {
            "type": "refinement_gate",
            "recommendation": "warn",
            "summary": _gate.summary(),
            "failed_conditions": [n for n, _ in _gate.reasons],
        }
        log.event(
            "refinement_gate_warning",
            failed=[n for n, _ in _gate.reasons],
            detail=reason_str[:300],
        )

    # Phase 54.6.71 (#7) — citation marker → chunk alignment. Runs once
    # on the writer's output before entering the iteration loop so the
    # scorer grades the corrected text, not the raw writer output. NLI
    # model is lazy-loaded (~440 MB one-time); remap gate is conservative
    # (claimed chunk entailment < 0.5 AND top chunk beats by ≥0.15) so
    # we never thrash on plausible citations. When align_citations can't
    # load the NLI model it degrades gracefully — content passes through.
    if results and content:
        try:
            from sciknow.core.citation_align import align_citations as _align
            _aln = _align(content, results)
            if _aln.n_remapped > 0:
                content = _aln.new_text
                log.event(
                    "citation_align",
                    sentences_scanned=_aln.n_sentences_scanned,
                    citations_checked=_aln.n_citations_checked,
                    remapped=_aln.n_remapped,
                    summary=_aln.summary(),
                )
                yield {"type": "citation_align",
                       "n_remapped": _aln.n_remapped,
                       "summary": _aln.summary()}
        except Exception as exc:
            logger.warning("citation_align skipped: %s", exc)

    for iteration in range(max_iter):
        yield {"type": "iteration_start", "iteration": iteration + 1, "max": max_iter}
        log.event("iteration_start", iteration=iteration + 1, max=max_iter)

        # Score (Fix 3: use writer's results, not re-retrieved)
        # Phase 15.3: streamed via _stream_phase so the GUI's token counter
        # stays alive during this 30+ second phase. The output is JSON; the
        # user sees it flowing instead of staring at "0 tok / 0 tok/s".
        log.stage("scoring", iteration=iteration + 1)
        yield {"type": "progress", "stage": "scoring",
               "detail": f"Scoring iteration {iteration + 1}..."}
        # Phase 54.6.320 — same VRAM-eviction guard as the verify phase.
        # bge-m3 + reranker re-loaded by the previous iteration's
        # retrieve step are still resident; the scorer's 27B model
        # would partial-load with the embedder co-resident.
        # Phase 55.V1 — replaced the v1-only `_release_gpu_models`
        # path with the v2-aware `_swap_to_phase("score")`. This
        # actually downs the embedder + reranker llama-server
        # processes (the v1 dance was a no-op in v2), AND swaps the
        # writer→scorer for cross-family scoring when configured.
        try:
            _release_gpu_models()
            _swap_to_phase("score")
        except Exception:
            pass
        # Phase 54.6.306 — when the scorer is a different model from
        # the writer (i.e. AUTOWRITE_SCORER_MODEL override is set,
        # like gemopus4), Ollama can't keep both sticky models
        # resident on a 24 GB card.  Writer is pinned with
        # keep_alive=-1 from the writing stage (~22 GB); the scorer
        # load then fails with "model failed to load, status code 500"
        # because Ollama refuses to evict a keep_alive=-1 model to make
        # room for another keep_alive=-1 model.  Explicitly release
        # the writer before the scorer load so Ollama has a clean GPU
        # to work with.  The writer will reload naturally on the next
        # writer-model call (verify / CoVe / revise) — ~10 s cost per
        # iteration, vs the alternative of the scoring phase hard-
        # failing and the iteration never converging.
        try:
            if scorer_model and settings.book_write_model and scorer_model != settings.book_write_model:
                from sciknow.rag.llm import release_llm as _release_llm
                _release_llm([settings.book_write_model])
        except Exception as exc:
            logger.debug("pre-scorer writer release failed: %s", exc)
        try:
            sys_s, usr_s = rag_prompts.score_draft(section_type, topic, content, results)
            # Phase 55.S1 — route through the dedicated scorer role
            # when configured, falling back to writer otherwise (the
            # rag.llm dispatch handles the fallback if the scorer GGUF
            # isn't loaded). The role param is plumbed through
            # _stream_phase via **kw → llm_stream → infer.client.chat_stream.
            score_raw = yield from _stream_phase(
                sys_s, usr_s, "scoring",
                model=scorer_model, temperature=0.0, num_ctx=16384,
                token_observer=log.token, role="scorer",
            )
            scores = json.loads(_clean_json(score_raw), strict=False)
        except Exception as exc:
            logger.warning("Scoring failed: %s", exc)
            log.event("scoring_failed", message=str(exc)[:200])
            scores = {"overall": 0.5, "weakest_dimension": "unknown",
                      "revision_instruction": "Improve overall quality."}
        # Symmetric release so the next stage (verify/CoVe/revise) can
        # reload the writer on a clean GPU without fighting the sticky
        # scorer.  No-op when scorer_model == writer (shared-model path).
        try:
            if scorer_model and settings.book_write_model and scorer_model != settings.book_write_model:
                from sciknow.rag.llm import release_llm as _release_llm
                _release_llm([scorer_model])
        except Exception as exc:
            logger.debug("post-scorer release failed: %s", exc)

        # Phase 54.6.79 (#6) — plan coverage as a dimension. Computes
        # NLI entailment of each atomic plan bullet against the draft;
        # if the scorer's "overall" stayed high but coverage fell, the
        # weakest-dimension logic below now picks "plan_coverage" and
        # the revision targets the missed bullets specifically. Fails
        # silently when NLI is unavailable or the plan text is empty.
        if section_plan and section_plan.strip():
            try:
                from sciknow.core.plan_coverage import (
                    compute_coverage, revision_hint_for_misses,
                )
                _cov = compute_coverage(content, section_plan)
                if _cov.n_bullets > 0:
                    scores["plan_coverage"] = _cov.coverage
                    log.event(
                        "plan_coverage",
                        coverage=round(_cov.coverage, 3),
                        n_bullets=_cov.n_bullets,
                        n_covered=_cov.n_covered,
                    )
                    yield {"type": "plan_coverage", "data": _cov.as_dict()}
                    # If plan_coverage is the weakest dimension AND it's
                    # below target, override the scorer's weakest + the
                    # revision instruction to name the missed bullets.
                    _dim_values = {
                        k: v for k, v in scores.items()
                        if isinstance(v, (int, float)) and k != "overall"
                    }
                    if _dim_values:
                        _weakest_dim = min(_dim_values, key=_dim_values.get)
                        if (_weakest_dim == "plan_coverage"
                                and _cov.coverage < target_score):
                            scores["weakest_dimension"] = "plan_coverage"
                            hint = revision_hint_for_misses(
                                _cov.missed_bullets
                            )
                            if hint:
                                scores["revision_instruction"] = hint
            except Exception as exc:
                logger.warning("plan_coverage skipped: %s", exc)

        # Phase 54.6.142 — visual-citation verify (L1 + L2) + score.
        # Only runs when include_visuals was on AND the writer had a
        # shortlist to cite from. Hallucinated markers force scores
        # ["visual_citation"] = 0 and become the weakest-dim candidate
        # for the next revision iteration.
        fig_verify: dict | None = None
        if include_visuals and ranked_visuals_for_writer:
            try:
                fig_verify = _verify_figure_refs(content, ranked_visuals_for_writer)
                vc_score = _score_visual_citation(
                    content, ranked_visuals_for_writer, fig_verify,
                )
                scores["visual_citation"] = vc_score
                log.event(
                    "visual_citation",
                    score=round(vc_score, 3),
                    n_markers=fig_verify["n_markers"],
                    n_hallucinated=fig_verify["n_hallucinated"],
                    n_low_entailment=fig_verify["n_low_entailment"],
                )
                yield {"type": "visual_citation", "data": {
                    "score": round(vc_score, 3),
                    **{k: v for k, v in fig_verify.items()
                       if k in ("n_markers", "n_hallucinated", "n_low_entailment")},
                }}
                # Hallucinated markers are a hard failure — force a
                # revise with a targeted instruction even if the other
                # dimensions scored high.
                if fig_verify["n_hallucinated"] > 0:
                    scores["weakest_dimension"] = "visual_citation"
                    bad = ", ".join(fig_verify["hallucinated_markers"][:5])
                    scores["revision_instruction"] = (
                        f"Remove or replace these hallucinated figure "
                        f"citations that do not resolve to any surfaced "
                        f"visual: {bad}. Either cite a figure from the "
                        f"provided shortlist or drop the inline reference "
                        f"and keep the claim as text."
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("visual_citation scoring skipped: %s", exc)

        # Fix 1: Run claim verification as part of scoring
        log.stage("verifying", iteration=iteration + 1)
        yield {"type": "progress", "stage": "verifying",
               "detail": "Verifying citations..."}
        # Phase 54.6.320 — free bge-m3 + reranker + ColBERT BEFORE the
        # 27B verifier call. Without this eviction, retrieval models
        # held ~10 GB of VRAM (re-loaded by an earlier autowrite
        # iteration's retrieve step), Ollama saw only ~14 GB free, and
        # the writer model partial-loaded with ~half its layers
        # spilled to CPU — decode collapsed from 30+ t/s to ~4 t/s.
        # This is the same pattern 54.6.305 fixed for the writer
        # entry path; missing here was the in-loop verify phase.
        # Phase 55.V1 — also swap to score phase for v2 substrate.
        try:
            _release_gpu_models()
            _swap_to_phase("score")
        except Exception:
            pass
        try:
            sys_v, usr_v = rag_prompts.verify_claims(content, results)
            # Phase 55.S1 — verify is a judging task, so route it
            # through the scorer role (Gemma 4 31B when configured) for
            # cross-family signal. Falls back to writer when
            # USE_LLAMACPP_SCORER=false via the rag.llm safety net.
            # Bonus: keeps llama-server's prompt cache warm across the
            # score → verify → CoVe block on the scorer side, saving
            # ~1-2s of prefill on each successive call.
            verify_raw = yield from _stream_phase(
                sys_v, usr_v, "verifying",
                model=scorer_model, temperature=0.0, num_ctx=16384,
                token_observer=log.token, role="scorer",
            )
            vdata = json.loads(_clean_json(verify_raw), strict=False)
            groundedness = vdata.get("groundedness_score", 1.0)
            hedging = vdata.get("hedging_fidelity_score", 1.0)

            # Merge groundedness into scores — if verification finds issues,
            # it can override the scorer's groundedness and force a revision.
            if groundedness < scores.get("groundedness", 1.0):
                scores["groundedness"] = groundedness
            # Same for hedging fidelity (lexical-level groundedness).
            if hedging < scores.get("hedging_fidelity", 1.0):
                scores["hedging_fidelity"] = hedging

            # Re-evaluate weakest dimension across the merged scores.
            score_dims = {
                k: scores[k] for k in (
                    "groundedness", "completeness", "coherence",
                    "citation_accuracy", "hedging_fidelity",
                ) if k in scores and isinstance(scores[k], (int, float))
            }
            if score_dims:
                weakest_now = min(score_dims, key=score_dims.get)
                scores["weakest_dimension"] = weakest_now

                # Generate a targeted revision instruction if verification flags issues.
                claims = vdata.get("claims", [])
                n_extrap = sum(1 for c in claims if c.get("verdict") == "EXTRAPOLATED")
                n_misrep = sum(1 for c in claims if c.get("verdict") == "MISREPRESENTED")
                n_overst = sum(1 for c in claims if c.get("verdict") == "OVERSTATED")

                if weakest_now == "groundedness" and (n_extrap + n_misrep) > 0:
                    scores["revision_instruction"] = (
                        f"Fix {n_extrap + n_misrep} citation(s) flagged as extrapolated or "
                        "misrepresented. Ensure every [N] reference accurately reflects what "
                        "the cited paper states."
                    )
                elif weakest_now == "hedging_fidelity" and n_overst > 0:
                    overstated_examples = [
                        f"{c.get('citation', '[?]')}: {c.get('reason', '')[:120]}"
                        for c in claims if c.get("verdict") == "OVERSTATED"
                    ][:3]
                    scores["revision_instruction"] = (
                        f"Soften {n_overst} overstated claim(s) — restore the source's "
                        "epistemic strength (e.g. 'suggests' instead of 'proves', 'is "
                        "associated with' instead of 'causes'). Specific issues: "
                        + " | ".join(overstated_examples)
                    )

            yield {"type": "verification", "data": vdata}

            # Phase 2: Chain-of-Verification (decoupled fact check).
            # Gated to fire only when standard verification looks weak,
            # because CoVe issues 1 + N_questions extra LLM calls.
            should_run_cove = (
                use_cove
                and (
                    groundedness < cove_threshold
                    or hedging < cove_threshold
                    or scores.get("groundedness", 1.0) < cove_threshold
                    or scores.get("hedging_fidelity", 1.0) < cove_threshold
                )
            )
            if should_run_cove:
                log.stage("cove", iteration=iteration + 1)
                yield {"type": "progress", "stage": "cove",
                       "detail": "Running Chain-of-Verification (decoupled fact check)..."}
                # Phase 54.6.320 — eviction before CoVe's batched-N call.
                # Phase 55.V1 — already in score phase from verify
                # above; swap_to_phase("score") is a fast no-op when
                # the scorer is already up. Keeps the explicit hint
                # close to the call site for readability.
                try:
                    _release_gpu_models()
                    _swap_to_phase("score")
                except Exception:
                    pass
                try:
                    # Phase 15.3: streamed CoVe so the GUI sees question
                    # generation + each independent answer flow as token
                    # events instead of going dark for 1 + N silent
                    # llm_complete calls.
                    # Phase 55.S1 — CoVe is a judging task; route to
                    # scorer role for cross-family signal + prompt-cache
                    # reuse with the score/verify block above. Falls
                    # back to writer via rag.llm safety net when
                    # USE_LLAMACPP_SCORER=false.
                    cove = yield from _cove_verify_streaming(
                        content, results, model=scorer_model, role="scorer",
                    )
                    yield {"type": "cove_verification", "data": cove}

                    high_sev = [m for m in cove.get("mismatches", [])
                                if m.get("severity") == "high"]
                    med_sev = [m for m in cove.get("mismatches", [])
                               if m.get("severity") == "medium"]

                    cove_score_v = cove.get("cove_score", 1.0)
                    # If CoVe is more pessimistic than the standard verifier
                    # for groundedness, take CoVe's number — independent
                    # re-answering is harder to fool.
                    if cove_score_v < scores.get("groundedness", 1.0):
                        scores["groundedness"] = cove_score_v
                        scores["weakest_dimension"] = "groundedness"

                    # If CoVe found high-severity mismatches, generate a
                    # targeted revision instruction that overrides the
                    # standard one. We name the specific claims so the
                    # writer can address them directly.
                    if high_sev:
                        examples = []
                        for m in high_sev[:3]:
                            claim_short = (m.get("draft_claim") or "")[:140]
                            ind_short = (m.get("independent_answer") or "")[:140]
                            examples.append(
                                f"CLAIM: {claim_short!r} — INDEPENDENT CHECK: {ind_short!r}"
                            )
                        scores["revision_instruction"] = (
                            f"Chain-of-Verification flagged {len(high_sev)} claim(s) "
                            "as NOT supported by the sources when checked independently. "
                            "Either remove these claims, or replace them with what the "
                            "sources actually say. Specific issues:\n  - "
                            + "\n  - ".join(examples)
                        )
                    elif med_sev and scores.get("weakest_dimension") in (
                        "groundedness", "hedging_fidelity",
                    ):
                        examples = []
                        for m in med_sev[:3]:
                            claim_short = (m.get("draft_claim") or "")[:140]
                            notes_short = (m.get("notes") or "")[:140]
                            examples.append(
                                f"CLAIM: {claim_short!r} — SCOPE: {notes_short!r}"
                            )
                        scores["revision_instruction"] = (
                            f"Chain-of-Verification flagged {len(med_sev)} claim(s) "
                            "where the source has a NARROWER scope than the draft "
                            "implies. Restore the source's scope qualifiers (region, "
                            "period, conditions). Specific issues:\n  - "
                            + "\n  - ".join(examples)
                        )
                except Exception as cove_exc:
                    logger.warning("CoVe verification failed: %s", cove_exc)
        except Exception as exc:
            logger.warning("Verification failed: %s", exc)

        # Phase 17 — inject length as a 7th scoring dimension.
        # The scorer is instructed to return 1.0 for length; we overwrite
        # it here with a mechanical min(1, actual/target) so that the
        # revision loop reacts to short drafts. Anti-oscillation guard:
        # length only takes over as the weakest dimension when it's
        # genuinely low (< LENGTH_PRIORITY_THRESHOLD, i.e. under ~70% of
        # target) AND lower than all other dimensions. Above the
        # threshold we leave weakest_dimension alone so hedging /
        # groundedness revisions aren't stolen by a mild length miss.
        length_score = _compute_length_score(content, effective_target_words)
        scores["length"] = length_score

        other_dims = {
            k: scores[k] for k in (
                "groundedness", "completeness", "coherence",
                "citation_accuracy", "hedging_fidelity",
            ) if k in scores and isinstance(scores[k], (int, float))
        }
        min_other = min(other_dims.values()) if other_dims else 1.0
        if (
            effective_target_words
            and length_score < LENGTH_PRIORITY_THRESHOLD
            and length_score < min_other
        ):
            scores["weakest_dimension"] = "length"
            actual_words = len(content.split())
            n_more_paragraphs = max(
                2, (effective_target_words - actual_words) // 150
            )
            scores["revision_instruction"] = (
                f"The draft is too short: {actual_words} words vs a "
                f"target of ~{effective_target_words}. Expand with "
                f"approximately {n_more_paragraphs} additional substantive "
                f"paragraphs, pulling new claims and quantitative detail "
                f"from the provided source passages. Do NOT pad with "
                f"filler phrases or repeat existing content — every new "
                f"paragraph must introduce distinct evidence or argument. "
                f"Preserve existing citations and add new [N] references "
                f"for the new claims."
            )

        overall = scores.get("overall", 0)
        weakest = scores.get("weakest_dimension", "unknown")
        instruction = scores.get("revision_instruction", "Improve the draft.")
        missing = scores.get("missing_topics", [])
        # Persist a richer history entry — not just the raw score dict, but
        # also the verification flag counts and the iteration index. This is
        # what `book draft scores` later reads back from custom_metadata.
        # vdata may be undefined if verification raised; locals() check is the
        # right way to detect that without depending on prior assignment.
        vdata_local = locals().get("vdata") or {}
        cove_local = locals().get("cove") or {}
        claims_local = vdata_local.get("claims") or []
        history_entry = {
            "iteration": iteration + 1,
            "scores": dict(scores),
            "verification": {
                "groundedness_score": vdata_local.get("groundedness_score"),
                "hedging_fidelity_score": vdata_local.get("hedging_fidelity_score"),
                "n_supported": sum(
                    1 for c in claims_local if c.get("verdict") == "SUPPORTED"
                ),
                "n_extrapolated": sum(
                    1 for c in claims_local if c.get("verdict") == "EXTRAPOLATED"
                ),
                "n_overstated": sum(
                    1 for c in claims_local if c.get("verdict") == "OVERSTATED"
                ),
                "n_misrepresented": sum(
                    1 for c in claims_local if c.get("verdict") == "MISREPRESENTED"
                ),
            },
            "cove": {
                "ran": bool(cove_local.get("questions_asked", 0)),
                "score": cove_local.get("cove_score"),
                "questions_asked": cove_local.get("questions_asked", 0),
                "n_high_severity": sum(
                    1 for m in (cove_local.get("mismatches") or [])
                    if m.get("severity") == "high"
                ),
                "n_medium_severity": sum(
                    1 for m in (cove_local.get("mismatches") or [])
                    if m.get("severity") == "medium"
                ),
            },
            "revision_verdict": None,  # filled in below if a revision happens
            "post_revision_overall": None,
        }
        history.append(history_entry)

        # Phase 32.6 — Layer 0: persist the iteration row immediately so
        # a Stop / crash mid-revision still leaves the score data behind.
        # The delta is computed against the previous iteration's word count
        # (tracked in the local _telemetry_prev_word_count, initialized to
        # the placeholder content above). On the FIRST iteration the delta
        # is NULL — there's no prior iteration to compare to.
        _word_count_now = len(content.split())
        _word_count_delta = (
            None if iteration == 0
            else _word_count_now - _telemetry_prev_word_count
        )
        _telemetry_iterations_used = iteration + 1
        _telemetry_final_overall = overall
        # Phase 32.9 — Layer 4: capture the pre-revision content (what
        # the scorer scored). This becomes the "rejected" side of the
        # KEEP preference pair, OR the "chosen" side of a DISCARD pair.
        _persist_autowrite_iteration(
            autowrite_run_id, iteration + 1, history_entry,
            word_count=_word_count_now,
            word_count_delta=_word_count_delta,
            overall_pre=overall,
            pre_revision_content=content,
        )

        yield {"type": "scores", "scores": scores, "iteration": iteration + 1}
        log.event("scores", iteration=iteration + 1, overall=overall, weakest=weakest)

        # Convergence check
        if overall >= target_score:
            yield {"type": "converged", "iteration": iteration + 1,
                   "final_score": overall}
            log.event("converged", iteration=iteration + 1, final_score=overall)
            # Phase 32.6 — mark the run as converged for finalize.
            _telemetry_converged = True
            break

        # Fix 4: Auto-expand — check if missing topics have weak corpus coverage
        if auto_expand and missing:
            yield {"type": "progress", "stage": "expanding",
                   "detail": f"Checking corpus coverage for: {', '.join(missing[:3])}"}
            # Phase 55.V1 — swap to retrieve phase before any embedder
            # hits. Score-phase prep (above) just brought up the scorer;
            # we now need the embedder + reranker for these probe-queries.
            try:
                _swap_to_phase("retrieve")
            except Exception:
                pass
            weak_topics = []
            with get_session() as session:
                for topic_q in missing[:3]:
                    try:
                        extra_results, _ = _retrieve(
                            session, qdrant, topic_q, candidate_k=5, context_k=3,
                        )
                        if len(extra_results) < 3:
                            weak_topics.append(topic_q)
                    except Exception:
                        pass
            if weak_topics:
                yield {"type": "progress", "stage": "expanding",
                       "detail": f"Weak coverage: {', '.join(weak_topics)}. "
                                 f"Consider: sciknow db expand -q \"{weak_topics[0]}\""}

        # Phase 54.6.26 — Adaptive revision (from WriteHERE's dynamic
        # planning). When scoring reveals completeness gaps or
        # missing_topics, run a TARGETED re-retrieval for the weakest
        # area before revising. This gives the revision prompt fresh
        # evidence it didn't have in the initial write pass. The extra
        # results are APPENDED to the existing results, not replaced.
        if missing and weakest in ("completeness", "length"):
            # Phase 55.V1 — same retrieve-phase swap as auto-expand.
            # Idempotent if already in retrieve phase from auto-expand
            # above. Cheap when the embedder + reranker are already hot.
            try:
                _swap_to_phase("retrieve")
            except Exception:
                pass
            for gap_query in missing[:2]:
                try:
                    with get_session() as session:
                        extra_results, _ = _retrieve(
                            session, qdrant, gap_query,
                            candidate_k=10, context_k=3,
                        )
                        if extra_results:
                            results = results + extra_results
                            yield {"type": "progress", "stage": "adaptive_retrieval",
                                   "detail": f"Found {len(extra_results)} extra passages for: {gap_query[:50]}"}
                except Exception:
                    pass

        # Revise (Fix 3: pass writer's results to revision context)
        log.stage("revising", iteration=iteration + 1)
        yield {"type": "progress", "stage": "revising",
               "detail": f"Revising (targeting {weakest})..."}

        # Phase 55.V1 — swap from score → generate phase. The previous
        # block (verify + CoVe + maybe adaptive_retrieval) leaves the
        # scorer hot (or, if adaptive_retrieval fired, the embedder/
        # reranker hot). Bring the writer up and evict everything else
        # before the revision call so the writer gets the full 24 GB.
        try:
            _swap_to_phase("generate")
        except Exception:
            pass

        sys_r, usr_r = rag_prompts.revise(content, instruction, results)

        # Phase 19 — incremental save during the revising stream. We
        # save the in-flight revision tokens to drafts.content directly,
        # overwriting the last KEEP state in this row. Rationale:
        # the user's reported pain is "I lost work when I clicked Stop"
        # — they want to SEE the in-progress revision after refresh,
        # not the previous (older) KEEP state. The previous KEEP is
        # still recoverable from the snapshots table if needed.
        #
        # If this revision later turns out to be DISCARD, the next
        # iteration's logic restores `content` to the prior value via
        # _update_draft_content, so the discard path is unaffected.
        revising_metadata = {
            "score_history": history,
            "feature_versions": feature_versions,
            "final_overall": overall,
            "max_iter": max_iter,
            "target_score": target_score,
            "target_words": effective_target_words,
            "checkpoint": f"iteration_{iteration + 1}_revising",
        }

        def _save_revising(text: str) -> None:
            _update_draft_content(
                draft_id, text,
                custom_metadata=revising_metadata,
            )

        revised = yield from _stream_with_save(
            sys_r, usr_r, "revising",
            model=model, save_callback=_save_revising,
            token_observer=log.token,
        )

        # Re-score (Fix 3: same results) — Phase 15.3 streamed
        log.stage("rescoring", iteration=iteration + 1)
        yield {"type": "progress", "stage": "scoring", "detail": "Re-scoring revision..."}
        # Phase 54.6.320 — eviction before re-scoring too.
        # Phase 55.V1 — also swap to score phase (writer→scorer) for
        # v2 substrate; the writer just produced the revision.
        try:
            _release_gpu_models()
            _swap_to_phase("score")
        except Exception:
            pass
        # Phase 54.6.306 — same sticky-LLM swap problem as the initial
        # scoring call above; the writer was just used for "revising"
        # with keep_alive=-1, so release it before the scorer tries to
        # load its different model.  See the comment on the first
        # scoring block for the full rationale.
        try:
            if scorer_model and settings.book_write_model and scorer_model != settings.book_write_model:
                from sciknow.rag.llm import release_llm as _release_llm
                _release_llm([settings.book_write_model])
        except Exception as exc:
            logger.debug("pre-rescorer writer release failed: %s", exc)
        try:
            sys_rs, usr_rs = rag_prompts.score_draft(section_type, topic, revised, results)
            # Phase 55.S1 — same scorer-role routing as the initial score.
            rescore_raw = yield from _stream_phase(
                sys_rs, usr_rs, "rescoring",
                model=scorer_model, temperature=0.0, num_ctx=16384,
                token_observer=log.token, role="scorer",
            )
            new_scores = json.loads(_clean_json(rescore_raw), strict=False)
        except Exception:
            new_scores = {"overall": overall}
        try:
            if scorer_model and settings.book_write_model and scorer_model != settings.book_write_model:
                from sciknow.rag.llm import release_llm as _release_llm
                _release_llm([scorer_model])
        except Exception as exc:
            logger.debug("post-rescorer release failed: %s", exc)

        # Phase 17 — inject length into new_scores so the KEEP/DISCARD
        # comparison is fair. Without this, a length-driven revision
        # that genuinely expands the draft would not see its length
        # improvement reflected in the overall score and could be
        # discarded. We also blend length into overall when length was
        # the driving dimension, so the KEEP verdict tracks what we
        # asked the writer to fix.
        new_length_score = _compute_length_score(revised, effective_target_words)
        new_scores["length"] = new_length_score
        if weakest == "length" and effective_target_words:
            # When length was the revision target, fold the length
            # improvement into overall so the KEEP comparison rewards
            # a longer draft even if other dims stayed flat. We use a
            # weighted blend rather than min() because a long draft
            # that scored 0.82 on other dims but 1.0 on length should
            # beat a short draft at 0.82 with 0.5 length.
            base = new_scores.get("overall", overall)
            new_scores["overall"] = 0.7 * base + 0.3 * new_length_score

        new_overall = new_scores.get("overall", 0)

        if new_overall >= overall:
            yield {"type": "revision_verdict", "action": "KEEP",
                   "old_score": overall, "new_score": new_overall}
            log.event("revision_verdict", action="KEEP",
                      old=overall, new=new_overall, iteration=iteration + 1)
            # Phase 32.6 — capture the pre-revision overall BEFORE the
            # `overall = new_overall` reassignment so the persist call
            # below records the right value in the overall_pre column.
            _overall_pre = overall
            content = revised
            overall = new_overall
            # Phase 32.6 — track the kept content's word count for the
            # next iteration's word_count_delta computation.
            _telemetry_prev_word_count = len(content.split())
            _telemetry_final_overall = overall
            if history:
                history[-1]["revision_verdict"] = "KEEP"
                history[-1]["post_revision_overall"] = new_overall
                # Update the iteration row with the verdict + post-overall.
                # Phase 32.9 — also capture the revised content as
                # post_revision_content for Layer 4 DPO pair extraction.
                _persist_autowrite_iteration(
                    autowrite_run_id, iteration + 1, history[-1],
                    word_count=_telemetry_prev_word_count,
                    word_count_delta=None,  # captured at the pre-revision persist
                    overall_pre=_overall_pre,
                    post_revision_content=revised,
                )
            # Phase 15.1 — INCREMENTAL SAVE checkpoint #N: persist the
            # accepted revision so the user never loses more than the
            # in-flight iteration's tokens if they click Stop.
            checkpoint_metadata = {
                "score_history": history,
                "feature_versions": feature_versions,
                "final_overall": overall,
                "max_iter": max_iter,
                "target_score": target_score,
                "target_words": effective_target_words,
                "checkpoint": f"iteration_{iteration + 1}_keep",
            }
            # Phase 19 — version is bumped RELATIVE to the starting
            # draft_version (set by _next_draft_version), not from 1.
            # Otherwise a new autowrite that started above an existing
            # draft would roll back to a lower version on first KEEP
            # and lose the latest-version-per-section sort, putting
            # the OLDER draft back on top.
            _update_draft_content(
                draft_id, content,
                custom_metadata=checkpoint_metadata,
                version=draft_version + iteration + 1,
            )
            yield {"type": "checkpoint", "draft_id": draft_id,
                   "stage": f"iteration_{iteration + 1}_keep",
                   "word_count": len(content.split())}
        else:
            yield {"type": "revision_verdict", "action": "DISCARD",
                   "old_score": overall, "new_score": new_overall}
            log.event("revision_verdict", action="DISCARD",
                      old=overall, new=new_overall, iteration=iteration + 1)
            if history:
                history[-1]["revision_verdict"] = "DISCARD"
                history[-1]["post_revision_overall"] = new_overall
                # Phase 32.6 — record the DISCARD verdict in the
                # iteration row. Word count stays the same (rejected
                # revision was not adopted).
                # Phase 32.9 — DISCARD verdicts ALSO produce a Layer 4
                # preference pair (with chosen and rejected swapped):
                # the original `content` won, the `revised` lost. Capture
                # the revised text as post_revision_content so the
                # exporter can build the inverse pair.
                _persist_autowrite_iteration(
                    autowrite_run_id, iteration + 1, history[-1],
                    word_count=_telemetry_prev_word_count,
                    word_count_delta=None,
                    overall_pre=overall,
                    post_revision_content=revised,
                )
            # Update metadata only — content stays at previous KEEP state.
            discard_metadata = {
                "score_history": history,
                "feature_versions": feature_versions,
                "final_overall": overall,
                "max_iter": max_iter,
                "target_score": target_score,
                "target_words": effective_target_words,
                "checkpoint": f"iteration_{iteration + 1}_discard",
            }
            _update_draft_content(draft_id, content, custom_metadata=discard_metadata)

    # Step 3: Final polish — auto-summarize and finalize the existing draft.
    # No new INSERT here because the draft was created at checkpoint #0 above.
    log.stage("finalizing")
    yield {"type": "progress", "stage": "saving", "detail": "Finalizing draft..."}
    summary = _auto_summarize(content, section_type, ch_title, model=model)
    persisted_metadata = {
        "score_history": history,
        "feature_versions": feature_versions,
        "final_overall": overall,
        "max_iter": max_iter,
        "target_score": target_score,
        "target_words": effective_target_words,
        "final_word_count": len(content.split()),
        "checkpoint": "final",
    }
    _update_draft_content(
        draft_id, content,
        summary=summary,
        custom_metadata=persisted_metadata,
        # Phase 19 — same rationale as the per-iteration version bump:
        # bump relative to the starting draft_version so a finalized
        # draft never lands below an older draft for the same section.
        version=draft_version + len(history),
    )

    log.event("completed",
              draft_id=draft_id,
              word_count=len(content.split()),
              iterations=len(history),
              final_score=overall)

    # Phase 32.6 — Layer 0: finalize the autowrite_runs row. This also
    # back-fills was_cited on autowrite_retrievals based on [N] markers
    # in the final draft text. Failure-soft: any SQL hiccup is logged
    # and the autowrite still completes successfully for the user.
    _finalize_autowrite_run(
        autowrite_run_id,
        status="completed",
        final_draft_id=draft_id,
        final_overall=overall,
        iterations_used=_telemetry_iterations_used,
        converged=_telemetry_converged,
        # Phase 33 — persist the token count from the logger so the
        # dashboard can aggregate cumulative LLM usage.
        tokens_used=log._state.get("total_tokens", 0),
    )

    yield {
        "type": "completed",
        "draft_id": draft_id,
        "word_count": len(content.split()),
        "iterations": len(history),
        "history": history,
        "final_score": overall,
    }


def _prune_chapter_unnamed_snapshots(session, chapter_id: str, *, keep: int) -> int:
    """Phase 54.6.328 (Phase 4) — keep the N newest auto-trigger
    snapshots per chapter, drop older ones.

    Auto-trigger snapshots have names that start with one of the
    well-known machine prefixes (``pre-autowrite-``, ``pre-revise-``,
    etc. — anything matching ``pre-<word>-`` where the suffix carries
    a timestamp). User-named snapshots (anything else) are NEVER
    pruned by this helper — they're the explicit "I want this
    forever" anchors.

    Returns the number of rows deleted.
    """
    from sqlalchemy import text
    rows = session.execute(text("""
        SELECT id::text FROM draft_snapshots
        WHERE chapter_id::text = :cid
          AND scope = 'chapter'
          AND name ~ '^pre-[a-z0-9_-]+-[0-9]'
        ORDER BY created_at DESC
        OFFSET :keep
    """), {"cid": chapter_id, "keep": int(keep)}).fetchall()
    if not rows:
        return 0
    ids = [r[0] for r in rows]
    n = session.execute(text(
        "DELETE FROM draft_snapshots WHERE id::text = ANY(:ids)"
    ), {"ids": ids}).rowcount or 0
    return int(n)


def autowrite_chapter_all_sections_stream(
    book_id: str,
    chapter_id: str,
    *,
    model: str | None = None,
    max_iter: int = 3,
    target_score: float = 0.85,
    auto_expand: bool = False,
    use_plan: bool = True,
    use_step_back: bool = True,
    use_cove: bool = True,
    cove_threshold: float = 0.85,
    target_words: int | None = None,
    rebuild: bool = False,
    resume: bool = False,
    only_below_target: bool = False,
    include_visuals: bool = False,   # Phase 54.6.144
) -> Iterator[Event]:
    """Phase 20 — autowrite EVERY section of a chapter in sequence.

    The user's complaint that motivated this: clicking the toolbar
    Autowrite from a chapter (no section pre-selected) defaulted to
    section_type='introduction' and created exactly one orphan draft,
    even though the chapter had 5 user-defined sections waiting. The
    GUI now routes "no section selected + chapter selected" to this
    function, which iterates over the chapter's sections list and
    chains autowrite_section_stream for each.

    Existing-draft handling — three modes:

    - **default (skip)** — sections that already have a draft are
      skipped entirely. The existing draft is preserved unchanged.
    - **rebuild=True** — existing drafts are ignored; a fresh
      autowrite runs from scratch. New row gets a higher version
      than the existing one (Phase 19).
    - **resume=True (Phase 28)** — for sections that already have a
      draft AND the draft is in a finished state (per
      _is_resumable_draft), the autowrite loads it as the starting
      content and runs the score → revise loop on it. Sections
      whose latest draft is in a partial state (writing_in_progress
      etc.) are SKIPPED with a warning instead of being clobbered,
      since resuming a partial buffer would silently corrupt the
      user's work.

    rebuild and resume are mutually exclusive — if both are passed,
    rebuild wins and resume is silently ignored (matching the CLI
    semantic that --rebuild overwrites everything).

    Cancellation: clicking Stop in the GUI calls gen.close() on this
    generator, which propagates GeneratorExit through the active
    `yield from autowrite_section_stream(...)` call. The Phase 19
    streaming-save finally block in the inner generator flushes the
    in-flight tokens before unwinding, so worst-case loss is the
    same ≤5s window as a single-section autowrite.
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    # rebuild wins over resume if both somehow get passed.
    if rebuild and resume:
        resume = False

    # --only-below-target without --rebuild implies --resume — same
    # convention as the CLI: re-iterate on below-target drafts rather
    # than skipping them as the default mode would.
    if only_below_target and not rebuild and not resume:
        resume = True

    # Phase 54.6.328 (snapshot-versioning Phase 4) — auto-snapshot the
    # chapter's current draft state BEFORE letting autowrite touch it.
    # No-ops cleanly when the chapter is empty (snapshot returns None).
    # Snapshot name encodes the trigger so the Timeline can group by
    # source. Best-effort: a snapshot failure must never block the
    # autowrite from running.
    import datetime as _dt
    try:
        from sciknow.web.routes.snapshots import (
            _snapshot_chapter_drafts as _snap_ch,
        )
        from sciknow.core.snapshot_diff import compute_bundle_brief
        with get_session() as _sess:
            _bundle = _snap_ch(_sess, chapter_id)
            if _bundle.get("drafts"):
                _stamp = _dt.datetime.now(_dt.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                _snap_name = f"pre-autowrite-{_stamp}"
                _prev = _sess.execute(text("""
                    SELECT content FROM draft_snapshots
                    WHERE chapter_id::text = :cid AND scope = 'chapter'
                    ORDER BY created_at DESC LIMIT 1
                """), {"cid": chapter_id}).fetchone()
                _prev_bundle = None
                if _prev and _prev[0]:
                    try:
                        import json as _json_mod
                        _prev_bundle = _json_mod.loads(_prev[0])
                    except Exception:
                        _prev_bundle = None
                _meta = compute_bundle_brief(_bundle, _prev_bundle)
                import json as _json_mod
                _wc = sum(d.get("word_count") or 0 for d in _bundle["drafts"])
                _sess.execute(text("""
                    INSERT INTO draft_snapshots
                        (chapter_id, scope, name, content, word_count, meta)
                    VALUES
                        (CAST(:cid AS uuid), 'chapter', :name, :content, :wc,
                         CAST(:meta AS jsonb))
                """), {"cid": chapter_id, "name": _snap_name,
                       "content": _json_mod.dumps(_bundle),
                       "wc": _wc,
                       "meta": _json_mod.dumps(_meta)})
                _sess.commit()
                # Prune unnamed older snapshots (keep last 14 per chapter).
                _prune_chapter_unnamed_snapshots(_sess, chapter_id, keep=14)
                _sess.commit()
    except Exception:  # noqa: BLE001
        # Snapshot is the safety net for the user, but it cannot
        # become a blocker. Failure to snapshot is logged via the
        # existing autowrite event stream's later steps if relevant.
        pass

    # Resolve the chapter's sections list (slug + title + plan).
    with get_session() as session:
        sections = _get_chapter_sections_normalized(session, chapter_id)
        if not sections:
            sections = [
                {"slug": slug, "title": _titleify_slug(slug), "plan": ""}
                for slug in _DEFAULT_BOOK_SECTION_SLUGS
            ]

        # Find which sections already have a draft.
        # Phase 28 — also fetch the latest draft id + custom_metadata
        # + word_count per section so resume mode can pick the latest
        # draft to use as the starting content AND check it's not in
        # a partial state.
        existing_rows = session.execute(text("""
            SELECT DISTINCT ON (section_type)
                   section_type, id::text, custom_metadata, word_count
            FROM drafts
            WHERE book_id::text = :bid AND chapter_id::text = :cid
            ORDER BY section_type, version DESC, created_at DESC
        """), {"bid": book_id, "cid": chapter_id}).fetchall()
        existing_by_slug: dict[str, dict] = {
            (r[0] or "").strip().lower(): {
                "draft_id": r[1],
                "custom_metadata": r[2],
                "word_count": r[3] or 0,
            }
            for r in existing_rows
        }

    n_total = len(sections)
    n_skipped = 0
    n_completed = 0
    n_failed = 0
    final_drafts: list[dict] = []

    yield {
        "type": "chapter_autowrite_start",
        "chapter_id": chapter_id,
        "n_sections": n_total,
        "sections": [{"slug": s["slug"], "title": s["title"]} for s in sections],
        "rebuild": rebuild,
        "resume": resume,
        "only_below_target": only_below_target,
    }

    for i, sec in enumerate(sections, start=1):
        slug = sec["slug"]
        title = sec["title"]
        existing = existing_by_slug.get(slug)
        already_exists = existing is not None

        # --only-below-target filter — runs BEFORE the rebuild/resume/skip
        # mode logic so it overrides a "rebuild all" or "resume all" run
        # for sections that have already crossed the target. Sections
        # without a draft, or whose final_overall is None (partial /
        # non-autowrite), are not skipped here.
        if only_below_target and already_exists:
            final_overall = _extract_draft_final_overall(
                existing["custom_metadata"]
            )
            if final_overall is not None and final_overall >= target_score:
                n_skipped += 1
                yield {
                    "type": "section_start", "index": i, "total": n_total,
                    "slug": slug, "title": title, "skipped": True,
                    "reason": (
                        f"already at target "
                        f"({final_overall:.2f} >= {target_score:.2f})"
                    ),
                }
                yield {
                    "type": "section_done", "index": i,
                    "slug": slug, "skipped": True,
                    "final_score": final_overall,
                }
                continue

        # Phase 28 — three-way handling of existing drafts.
        resume_draft_id: str | None = None
        if already_exists:
            if rebuild:
                pass  # fall through to a fresh autowrite
            elif resume:
                # Check resume eligibility. Refuse partial states.
                ok, reason = _is_resumable_draft(
                    existing["custom_metadata"], existing["word_count"],
                )
                if not ok:
                    n_skipped += 1
                    yield {
                        "type": "section_start", "index": i, "total": n_total,
                        "slug": slug, "title": title, "skipped": True,
                        "reason": (
                            f"resume refused: {reason} "
                            f"(rerun with rebuild to overwrite from scratch)"
                        ),
                    }
                    yield {
                        "type": "section_done", "index": i,
                        "slug": slug, "skipped": True,
                    }
                    continue
                resume_draft_id = existing["draft_id"]
            else:
                # Default: skip existing drafts.
                n_skipped += 1
                yield {
                    "type": "section_start", "index": i, "total": n_total,
                    "slug": slug, "title": title, "skipped": True,
                    "reason": "draft already exists (use rebuild or resume)",
                }
                yield {
                    "type": "section_done", "index": i,
                    "slug": slug, "skipped": True,
                }
                continue

        yield {
            "type": "section_start", "index": i, "total": n_total,
            "slug": slug, "title": title, "skipped": False,
            "resume": resume_draft_id is not None,
        }

        # Track per-section state so the all-sections summary can
        # report which ones converged vs which errored.
        section_completed = False
        section_draft_id = None
        section_final_score = None

        try:
            inner = autowrite_section_stream(
                book_id=book_id,
                chapter_id=chapter_id,
                section_type=slug,
                model=model,
                max_iter=max_iter,
                target_score=target_score,
                auto_expand=auto_expand,
                use_plan=use_plan,
                use_step_back=use_step_back,
                use_cove=use_cove,
                cove_threshold=cove_threshold,
                target_words=target_words,
                resume_from_draft_id=resume_draft_id,
                include_visuals=include_visuals,
            )
            for event in inner:
                # Forward all child events; tag with section index so
                # the GUI knows which section the event belongs to.
                event = dict(event)
                event["section_index"] = i
                event["section_slug"] = slug
                event["section_total"] = n_total
                if event.get("type") == "completed":
                    section_completed = True
                    section_draft_id = event.get("draft_id")
                    section_final_score = event.get("final_score")
                yield event
        except GeneratorExit:
            # Propagate cancellation. The inner generator's finally
            # block already flushed any in-flight buffer; we just
            # unwind cleanly.
            raise
        except Exception as exc:
            n_failed += 1
            logger.exception(
                "autowrite_chapter section %r failed: %s", slug, exc
            )
            yield {
                "type": "section_error", "index": i, "slug": slug,
                "message": str(exc),
            }
            yield {
                "type": "section_done", "index": i, "slug": slug,
                "error": str(exc),
            }
            continue

        if section_completed:
            n_completed += 1
            final_drafts.append({
                "slug": slug, "title": title,
                "draft_id": section_draft_id,
                "final_score": section_final_score,
            })
            yield {
                "type": "section_done", "index": i, "slug": slug,
                "draft_id": section_draft_id,
                "final_score": section_final_score,
            }
        else:
            # Inner generator returned without a 'completed' event —
            # treat as a failed section but don't bail out of the
            # whole chapter.
            n_failed += 1
            yield {
                "type": "section_done", "index": i, "slug": slug,
                "error": "no completed event",
            }

    yield {
        "type": "all_sections_complete",
        "n_total": n_total,
        "n_completed": n_completed,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        "drafts": final_drafts,
    }


def autowrite_book_all_chapters_stream(
    book_id: str,
    *,
    model: str | None = None,
    max_iter: int = 3,
    target_score: float = 0.85,
    target_words: int | None = None,
    rebuild: bool = False,
    resume: bool = False,
    only_below_target: bool = False,
    include_visuals: bool = False,
) -> Iterator[Event]:
    """Phase 54.6.x — autowrite EVERY section of EVERY chapter in the book.

    Iterates the book's chapters in their stored order and chains
    ``autowrite_chapter_all_sections_stream`` for each one. The same
    rebuild / resume / skip semantics apply at the per-section level
    inside each chapter; this wrapper just walks the chapter list.

    Cancellation propagates through GeneratorExit on the inner generator
    just like the chapter-level wrapper, so the in-flight section's
    streaming-save flush still runs before the whole-book run unwinds.
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    if rebuild and resume:
        resume = False

    if only_below_target and not rebuild and not resume:
        resume = True

    with get_session() as session:
        rows = session.execute(text("""
            SELECT id::text, number, title
            FROM book_chapters
            WHERE book_id::text = :bid
            ORDER BY number ASC
        """), {"bid": book_id}).fetchall()

    chapters = [{"id": r[0], "number": r[1], "title": r[2] or ""} for r in rows]
    n_chapters = len(chapters)

    yield {
        "type": "book_autowrite_start",
        "book_id": book_id,
        "n_chapters": n_chapters,
        "chapters": chapters,
        "rebuild": rebuild,
        "resume": resume,
        "only_below_target": only_below_target,
    }

    if n_chapters == 0:
        yield {
            "type": "all_chapters_complete",
            "n_chapters": 0,
            "n_chapters_completed": 0,
            "n_sections_completed": 0,
            "n_sections_skipped": 0,
            "n_sections_failed": 0,
            "message": "no chapters in this book — add chapters first",
        }
        return

    n_chapters_completed = 0
    total_sections_completed = 0
    total_sections_skipped = 0
    total_sections_failed = 0

    for ci, ch in enumerate(chapters, start=1):
        yield {
            "type": "chapter_start",
            "chapter_index": ci, "chapter_total": n_chapters,
            "chapter_id": ch["id"],
            "chapter_number": ch["number"],
            "chapter_title": ch["title"],
        }
        try:
            inner = autowrite_chapter_all_sections_stream(
                book_id=book_id,
                chapter_id=ch["id"],
                model=model,
                max_iter=max_iter,
                target_score=target_score,
                target_words=target_words,
                rebuild=rebuild,
                resume=resume,
                only_below_target=only_below_target,
                include_visuals=include_visuals,
            )
            for event in inner:
                event = dict(event)
                event["chapter_index"] = ci
                event["chapter_total"] = n_chapters
                event["chapter_id"] = ch["id"]
                if event.get("type") == "all_sections_complete":
                    total_sections_completed += int(event.get("n_completed", 0) or 0)
                    total_sections_skipped += int(event.get("n_skipped", 0) or 0)
                    total_sections_failed += int(event.get("n_failed", 0) or 0)
                yield event
        except GeneratorExit:
            raise
        except Exception as exc:
            logger.exception(
                "autowrite_book chapter %r failed: %s", ch["id"], exc,
            )
            yield {
                "type": "chapter_error",
                "chapter_index": ci, "chapter_id": ch["id"],
                "message": str(exc),
            }
            yield {
                "type": "chapter_done",
                "chapter_index": ci, "chapter_id": ch["id"],
                "error": str(exc),
            }
            continue

        n_chapters_completed += 1
        yield {
            "type": "chapter_done",
            "chapter_index": ci, "chapter_id": ch["id"],
        }

    yield {
        "type": "all_chapters_complete",
        "n_chapters": n_chapters,
        "n_chapters_completed": n_chapters_completed,
        "n_sections_completed": total_sections_completed,
        "n_sections_skipped": total_sections_skipped,
        "n_sections_failed": total_sections_failed,
    }


# ════════════════════════════════════════════════════════════════════
# Phase 46 — Two-stage citation insertion (A)
# ════════════════════════════════════════════════════════════════════
#
# AI-Scientist's `citation_first_prompt` / `citation_second_prompt`
# ported into sciknow. Pass 1 sees only the draft and emits
# {location, claim, query} records. Pass 2 sees each location + the
# top-K hybrid-search candidates and picks (or rejects). The event
# stream mirrors the other generators so the CLI and the web reader
# can subscribe uniformly:
#
#   progress, citation_scan_start, citation_needs, citation_candidates,
#   citation_selected | citation_skipped, citation_inserted,
#   completed, error


