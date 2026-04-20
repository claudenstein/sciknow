"""
Service layer for book operations.

Generator-based functions that yield typed event dicts, consumable by both the
CLI (Rich console) and the web layer (SSE endpoints).  Every function is a plain
Python generator — no asyncio dependency — so the web layer can run them inside
``asyncio.to_thread()`` and drain events through a queue.

Event types
-----------
- ``token``      — streaming LLM text fragment
- ``progress``   — stage/status update (human-readable)
- ``scores``     — structured quality scores from the reviewer
- ``completed``  — operation finished, includes draft_id / word_count / sources
- ``error``      — something went wrong
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("sciknow.core.book_ops")

# Type alias for the event dicts yielded by every generator.
Event = dict


# ── Shared helpers ────────────────────────────────────────────────────────────

# Phase 17 — default chapter-length target in words. Chosen because popular
# science / trade-science chapters typically run 5k–10k words; 6k is the
# centre of that range and produces drafts long enough to feel like book
# chapters rather than extended abstracts. Per-book and per-call overrides
# live in books.custom_metadata.target_chapter_words and the --target-words
# CLI / web flags.
DEFAULT_TARGET_CHAPTER_WORDS = 6000

# Phase 17 — minimum length score below which "length" can become the
# weakest dimension and drive a targeted "expand" revision. 0.7 means the
# draft is under ~70% of target words. Above 0.7 we consider length good
# enough and let other dimensions (groundedness, hedging, etc.) drive
# revision. This is an anti-oscillation guard: without it, length would
# constantly win over a draft that's at 0.65 on hedging_fidelity and
# 0.9 on length, flipping revision targets between iterations.
LENGTH_PRIORITY_THRESHOLD = 0.7


def _get_book_length_target(session, book_id: str) -> int:
    """Read books.custom_metadata.target_chapter_words, fall back to default.

    Phase 17. Returns an int number of words. Safe to call on any book —
    a missing column, missing key, or non-int value all fall through to
    DEFAULT_TARGET_CHAPTER_WORDS.
    """
    from sqlalchemy import text
    try:
        row = session.execute(text("""
            SELECT custom_metadata FROM books WHERE id::text = :bid LIMIT 1
        """), {"bid": book_id}).fetchone()
    except Exception:
        return DEFAULT_TARGET_CHAPTER_WORDS
    if not row:
        return DEFAULT_TARGET_CHAPTER_WORDS
    meta = row[0] or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            return DEFAULT_TARGET_CHAPTER_WORDS
    val = meta.get("target_chapter_words") if isinstance(meta, dict) else None
    try:
        n = int(val)
        if n > 0:
            return n
    except (TypeError, ValueError):
        pass
    return DEFAULT_TARGET_CHAPTER_WORDS


def _section_target_words(chapter_target: int, num_sections: int) -> int:
    """Split a chapter-level target across its sections.

    Phase 17. A chapter with 4 sections and a 6000-word target asks each
    section for ~1500 words. We floor at 400 because a section shorter
    than that rarely reads as a book-grade section; and ceiling at the
    chapter target itself (1-section chapters just inherit the full
    chapter budget).
    """
    if num_sections <= 0:
        return max(400, chapter_target)
    per = chapter_target // num_sections
    return max(400, min(chapter_target, per))


def _compute_length_score(content: str, target_words: int | None) -> float:
    """Return a length score in [0, 1] = min(1, actual / target).

    Phase 17. Over-length drafts are NOT penalised — the model is allowed
    to overshoot when it genuinely has more to say. A missing/zero target
    returns 1.0 (length is not evaluated).
    """
    if not target_words or target_words <= 0:
        return 1.0
    actual = len((content or "").split())
    return min(1.0, actual / float(target_words))


def _get_chapter_num_sections(session, chapter_id: str) -> int:
    """Count the sections configured on a chapter (from book_chapters.sections).

    Phase 17. Used to divide the chapter-level length target across its
    sections. Falls back to 1 (treat the whole target as one section's
    budget) if the sections array is missing or empty.
    """
    return len(_get_chapter_sections_normalized(session, chapter_id)) or 1


# Phase 18 — chapter sections as first-class entities.
#
# Storage shape (book_chapters.sections JSONB):
#
#   New: [{"slug": "solar_cycle", "title": "The 11-Year Cycle",
#          "plan": "Cover sunspot counts, butterfly diagram, ..."},
#         {...}]
#
#   Legacy: ["solar_cycle", "geomagnetic_storms", ...]
#
# All read paths go through _normalize_chapter_sections so legacy data
# keeps working without a migration. New writes from the GUI always
# produce the dict shape, so chapters auto-upgrade lazily on the
# first edit.

def _slugify_section_name(name: str) -> str:
    """Lowercase + spaces-to-underscores so display titles like
    'The 11-Year Solar Cycle' become drafts.section_type-friendly slugs
    like 'the_11-year_solar_cycle'."""
    return (name or "").strip().lower().replace(" ", "_")


def _titleify_slug(slug: str) -> str:
    """Best-effort display title for a legacy slug. 'key_evidence' →
    'Key Evidence'."""
    return (slug or "").replace("_", " ").strip().title()


def _normalize_chapter_sections(raw) -> list[dict]:
    """Return a list of {slug, title, plan} dicts, regardless of input shape.

    Phase 18. Accepts the new dict shape, the legacy string-list shape,
    JSON-encoded versions of either, and None/empty (returns []).
    Items missing fields get sensible defaults: slug auto-derived from
    title (or vice versa), plan defaults to empty string.

    Example legacy input:
        ["overview", "key_evidence"]
    Returns:
        [{"slug": "overview", "title": "Overview", "plan": ""},
         {"slug": "key_evidence", "title": "Key Evidence", "plan": ""}]

    Example new input:
        [{"slug": "solar_cycle", "title": "The 11-Year Cycle", "plan": "Cover sunspot counts..."}]
    Returns the same with any missing fields filled in.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, str):
            slug = _slugify_section_name(item)
            if not slug:
                continue
            # Phase 29 — target_words defaults to None (means "use the
            # chapter target divided by num_sections"). Stored as int
            # when set per-section via the GUI dropdown.
            out.append({
                "slug": slug,
                "title": _titleify_slug(slug),
                "plan": "",
                "target_words": None,
            })
        elif isinstance(item, dict):
            slug = _slugify_section_name(item.get("slug") or item.get("title") or "")
            if not slug:
                continue
            title = (item.get("title") or _titleify_slug(slug)).strip()
            plan = (item.get("plan") or "").strip()
            # Phase 29 — preserve target_words. Coerce to int when
            # present, treat zero/None/non-int as "no override".
            tw_raw = item.get("target_words")
            try:
                tw = int(tw_raw) if tw_raw not in (None, "", 0) else None
                if tw is not None and tw <= 0:
                    tw = None
            except (TypeError, ValueError):
                tw = None
            # Phase 37 — per-section model override. Non-empty string
            # means "use this Ollama model for write/autowrite/review/
            # revise on this section"; None/empty falls through to the
            # caller-provided or global default.
            model_raw = item.get("model")
            model_val = (model_raw.strip() if isinstance(model_raw, str) else None) or None
            out.append({
                "slug": slug, "title": title, "plan": plan,
                "target_words": tw,
                "model": model_val,
            })
    return out


def _get_chapter_sections_normalized(session, chapter_id: str) -> list[dict]:
    """Read book_chapters.sections and normalize to [{slug, title, plan}, ...].

    Phase 18. Returns an empty list if the chapter doesn't exist or has
    no sections set. Callers that need a non-empty fallback (e.g. the
    GUI's empty-state CTA) should use _DEFAULT_BOOK_SECTION_SLUGS.
    """
    from sqlalchemy import text
    try:
        row = session.execute(text(
            "SELECT sections FROM book_chapters WHERE id::text = :cid LIMIT 1"
        ), {"cid": chapter_id}).fetchone()
    except Exception:
        return []
    if not row:
        return []
    return _normalize_chapter_sections(row[0])


# Phase 18 — fallback section slugs for chapters that have no explicit
# section list. Mirrors web/app.py's _DEFAULT_BOOK_SECTIONS so the GUI
# and the writer agree on the empty-state shape.
_DEFAULT_BOOK_SECTION_SLUGS = [
    "overview", "key_evidence", "current_understanding",
    "open_questions", "summary",
]


def _get_section_plan(session, chapter_id: str, section_slug: str) -> str:
    """Return the per-section plan text for a (chapter, section) pair, or "".

    Phase 18. Looks up the section in the chapter's sections JSONB and
    returns its plan field. Used by write_section_stream and
    autowrite_section_stream to inject the plan into the writer prompt.
    """
    if not section_slug:
        return ""
    sections = _get_chapter_sections_normalized(session, chapter_id)
    target = _slugify_section_name(section_slug)
    for s in sections:
        if s["slug"] == target:
            return s.get("plan") or ""
    return ""


def _get_section_target_words(
    session, chapter_id: str, section_slug: str,
) -> int | None:
    """Phase 29 — return the per-section target_words override, or None
    if no override is set on this section.

    The autowrite/write target resolution is now a 3-level priority:
        1. caller arg (--target-words / target_words form field)
        2. per-section meta override (this function)
        3. derived from chapter target / num_sections (default)

    None at this level means "fall through to level 3". The GUI's
    section dropdown writes this field via the existing
    PUT /api/chapters/{id}/sections endpoint (which calls
    _normalize_chapter_sections, which preserves target_words).
    """
    if not section_slug:
        return None
    sections = _get_chapter_sections_normalized(session, chapter_id)
    target = _slugify_section_name(section_slug)
    for s in sections:
        if s["slug"] == target:
            tw = s.get("target_words")
            if tw and isinstance(tw, int) and tw > 0:
                return tw
            return None
    return None


def _get_section_model(
    session, chapter_id: str, section_slug: str,
) -> str | None:
    """Phase 37 — return the per-section Ollama model override, or None
    if this section has none set.

    Resolution precedence at every LLM call site is:
        1. Explicit caller model (CLI --model, API `model` form field)
        2. Per-section model (this function)
        3. settings.llm_model default (set inside rag.llm.stream)

    Paired with the compute counter (Phase 35) and the Tools panel
    (Phase 36): you see per-section spend, then dial expensive models
    down to just the sections that need them. All four streams
    (write / autowrite / review / revise) consult this resolver.
    """
    if not section_slug:
        return None
    sections = _get_chapter_sections_normalized(session, chapter_id)
    target = _slugify_section_name(section_slug)
    for s in sections:
        if s["slug"] == target:
            m = s.get("model")
            if isinstance(m, str) and m.strip():
                return m.strip()
            return None
    return None


def adopt_orphan_section(
    book_id: str,
    chapter_id: str,
    section_slug: str,
    *,
    title: str | None = None,
    plan: str | None = None,
) -> dict:
    """Phase 25 — append an orphan section_type slug to a chapter's
    sections list, so a draft whose section_type didn't match any
    template slug becomes "drafted" instead of "orphan" in the GUI.

    The motivating scenario: the user defined 5 sections on a chapter
    AFTER autowriting an introduction draft. The introduction draft's
    section_type doesn't match any of the new slugs, so the GUI shows
    it with a red "orphan" dot. They want to keep the draft AND have
    it appear as a regular drafted section. This function does that
    by appending {slug, title, plan} to book_chapters.sections.

    Idempotent: if the slug is already in the sections list, returns
    the existing entry without duplication. The title defaults to a
    titleified version of the slug; the plan defaults to "".

    Returns a dict:
        {
          "ok": bool,
          "added": bool,           # True if a new entry was appended
          "section": {slug, title, plan},  # the resulting entry
          "sections": [...]        # the full updated sections list
        }

    Raises ValueError if the chapter doesn't exist or the slug is empty.
    Used by both the CLI (`sciknow book section adopt`) and the web
    endpoint (POST /api/chapters/{id}/sections/adopt).
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    if not section_slug or not section_slug.strip():
        raise ValueError("section_slug is required")

    # Normalize the slug the same way the rest of the code does so
    # "Introduction" and "introduction" both map to the same entry.
    target = _slugify_section_name(section_slug)
    if not target:
        raise ValueError(f"section_slug {section_slug!r} normalizes to empty")

    with get_session() as session:
        # Verify the chapter exists and belongs to this book.
        row = session.execute(text("""
            SELECT bc.sections, bc.title
            FROM book_chapters bc
            WHERE bc.id::text = :cid
              AND bc.book_id::text = :bid
            LIMIT 1
        """), {"cid": chapter_id, "bid": book_id}).fetchone()
        if not row:
            raise ValueError(
                f"chapter {chapter_id!r} not found in book {book_id!r}"
            )

        sections = _normalize_chapter_sections(row[0])

        # Idempotent: if the slug already exists, return without writing.
        for s in sections:
            if s["slug"] == target:
                return {
                    "ok": True,
                    "added": False,
                    "section": s,
                    "sections": sections,
                }

        # Append a new entry. Title falls back to a titleified slug
        # (e.g. "introduction" → "Introduction"), plan defaults to ""
        # so the writer doesn't get a stale plan injected.
        new_entry = {
            "slug": target,
            "title": (title or _titleify_slug(target)).strip(),
            "plan": (plan or "").strip(),
        }
        sections.append(new_entry)

        session.execute(text("""
            UPDATE book_chapters
            SET sections = CAST(:secs AS jsonb)
            WHERE id::text = :cid
        """), {"cid": chapter_id, "secs": json.dumps(sections)})
        session.commit()

    return {
        "ok": True,
        "added": True,
        "section": new_entry,
        "sections": sections,
    }


# ── Phase 54.6.65 — data-weighted section-count resizing ─────────────────────
#
# The outline LLM can pick any section count it likes within the prompt's
# 3–8 range, but (a) LLMs tend to converge on a single "safe" count across
# all chapters and (b) a fixed count doesn't reflect how much evidence is
# actually in the corpus for a given chapter topic. After the outline is
# generated, run one hybrid retrieval per chapter on the chapter's
# topic_query, count distinct papers in the top-100, and trim the chapter's
# section list to a target count derived from that density. We ONLY trim
# (never grow) because adding coherent section names would require another
# LLM pass and the grow-up case is rare — the prompt already asks for 3–8,
# and a chapter where the LLM picked 3 when evidence supports 6 is usually
# fine as-is (the writer can still find enough material per section).

_DENSITY_SECTION_TARGETS: list[tuple[int, int]] = [
    # (max distinct papers for this bucket, target section count)
    (3,  2),
    (8,  3),
    (20, 4),
    (40, 5),
    (70, 6),
    (10**9, 7),
]


def _target_sections_for_density(n_papers: int) -> int:
    for cap, target in _DENSITY_SECTION_TARGETS:
        if n_papers <= cap:
            return target
    return 7  # safety


def _grow_sections_llm(chapter: dict, n_add: int, model: str | None = None) -> list[str]:
    """Ask the fast LLM for `n_add` additional section names that
    complement the chapter's existing sections.

    Returns a list of new section name strings. Empty list on any
    failure (safe — caller just keeps the original list).
    """
    from sciknow.rag.llm import complete as _complete
    import json as _json
    import re as _re

    existing = chapter.get("sections") or []
    system = (
        "You are a scientific book editor extending a chapter outline. "
        "The existing sections are given; propose ADDITIONAL section names "
        "that fit the chapter's scope without overlapping the existing ones. "
        "Respond ONLY with a JSON array of strings."
    )
    user = (
        f"Chapter title: {chapter.get('title','')}\n"
        f"Chapter description: {chapter.get('description','')}\n"
        f"Topic query: {chapter.get('topic_query','')}\n"
        f"Existing sections: {_json.dumps(existing)}\n\n"
        f"Propose {n_add} ADDITIONAL section names that complement the "
        f"existing ones (no overlap, no duplication). Return a JSON array "
        f'of strings, e.g. ["New Section 1", "New Section 2"].'
    )
    try:
        raw = _complete(system, user, model=model, temperature=0.4,
                        num_ctx=8192, keep_alive=-1)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        # Be lenient: the LLM might wrap in {"sections": [...]}.
        parsed = _json.loads(raw, strict=False)
        if isinstance(parsed, dict):
            for key in ("sections", "additional_sections", "new_sections"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
        if not isinstance(parsed, list):
            return []
        out: list[str] = []
        existing_lower = {str(s).strip().lower() for s in existing}
        for item in parsed[:n_add]:
            name = str(item).strip() if item is not None else ""
            if name and name.lower() not in existing_lower:
                out.append(name)
        return out
    except Exception:
        return []


def resize_sections_by_density(chapters: list[dict], *,
                                grow: bool = True,
                                model: str | None = None) -> list[dict]:
    """Align each chapter's `sections` count to the evidence density
    in the corpus for that chapter's topic_query.

    - Over-specified chapters (len(sections) > target) are trimmed.
    - Under-specified chapters (len(sections) < target, ≥2 short) grow
      via a fast-LLM call that proposes complementary section names.
    - Chapters already at target are left alone.

    Mutates the chapter dicts in place and returns them. Safe to call
    with retrieval/LLM offline — any failure for a given chapter
    leaves that chapter's section list untouched.
    """
    from sciknow.retrieval import hybrid_search as _hs
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    try:
        qdrant = get_client()
    except Exception:
        return chapters  # retrieval stack not available — leave as-is

    with get_session() as session:
        for ch in chapters:
            topic = (ch.get("topic_query") or ch.get("title") or "").strip()
            secs = list(ch.get("sections") or [])
            if not topic or not secs:
                continue
            try:
                candidates = _hs.search(topic, qdrant, session, candidate_k=100)
            except Exception:
                continue
            n_papers = len({c.document_id for c in candidates})
            target = _target_sections_for_density(n_papers)
            original = len(secs)
            action = "kept"
            if original > target:
                ch["sections"] = secs[:target]
                action = "trimmed"
            elif grow and original < target - 1:
                # Grow only when ≥2 short; a single section gap isn't
                # worth an LLM call and is well within LLM noise.
                n_add = target - original
                extra = _grow_sections_llm(ch, n_add, model=model)
                if extra:
                    ch["sections"] = secs + extra
                    action = f"grown (+{len(extra)})"
            ch["_density_info"] = {
                "n_papers": n_papers,
                "target": target,
                "original_count": original,
                "final_count": len(ch.get("sections") or []),
                "action": action,
            }
    return chapters


def _get_book(session, title_or_id: str):
    from sqlalchemy import text
    return session.execute(text("""
        SELECT id::text, title, description, status, created_at, plan
        FROM books WHERE title ILIKE :q OR id::text LIKE :q
        LIMIT 1
    """), {"q": f"%{title_or_id}%"}).fetchone()


def _get_chapter(session, book_id: str, chapter_ref: str):
    from sqlalchemy import text
    if chapter_ref.isdigit():
        row = session.execute(text("""
            SELECT id::text, number, title, description, topic_query, topic_cluster
            FROM book_chapters WHERE book_id = :bid AND number = :num
        """), {"bid": book_id, "num": int(chapter_ref)}).fetchone()
        if row:
            return row
    return session.execute(text("""
        SELECT id::text, number, title, description, topic_query, topic_cluster
        FROM book_chapters WHERE book_id = :bid AND title ILIKE :q
        LIMIT 1
    """), {"bid": book_id, "q": f"%{chapter_ref}%"}).fetchone()


def _get_prior_summaries(session, book_id: str, before_chapter_number: int) -> list[dict]:
    from sqlalchemy import text
    rows = session.execute(text("""
        SELECT bc.number, d.section_type, d.summary
        FROM drafts d
        JOIN book_chapters bc ON bc.id = d.chapter_id
        WHERE d.book_id = :bid AND bc.number < :ch_num AND d.summary IS NOT NULL
        ORDER BY bc.number, d.section_type
    """), {"bid": book_id, "ch_num": before_chapter_number}).fetchall()
    return [{"chapter_number": r[0], "section_type": r[1], "summary": r[2]} for r in rows]


def _auto_summarize(content: str, section_type: str, chapter_title: str, model: str | None = None) -> str:
    from sciknow.rag import prompts
    from sciknow.rag.llm import complete
    system, user = prompts.draft_summary(section_type, chapter_title, content)
    try:
        # Phase 54.6.30 — 16384 to match the main write pass and avoid
        # Ollama reloading the model between write and summarize
        # (summarize runs right after a section is written in
        # write_section_stream and autowrite, same model in scope).
        return complete(system, user, model=model, temperature=0.1, num_ctx=16384, keep_alive=-1).strip()
    except Exception as exc:
        logger.warning("Auto-summarize failed for %s/%s: %s", chapter_title, section_type, exc)
        return ""


# ── Phase 32.6 — Compound learning Layer 0: autowrite telemetry ─────────
#
# Four helpers that let autowrite_section_stream persist its full
# trajectory into the autowrite_runs / autowrite_iterations /
# autowrite_retrievals tables (added in migration 0011). Designed to
# fail soft: any persistence error is logged and swallowed so it never
# kills a running autowrite job. Read more in docs/RESEARCH.md §21.


def _create_autowrite_run(
    *,
    book_id: str | None,
    chapter_id: str | None,
    section_slug: str,
    model: str | None,
    target_words: int | None,
    max_iter: int | None,
    target_score: float | None,
    feature_versions: dict | None,
) -> str | None:
    """Phase 32.6 — open a new autowrite_runs row, return its id.

    Returns None on any failure (the autowrite generator continues
    without telemetry rather than crashing on a SQL hiccup).
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    try:
        with get_session() as session:
            row = session.execute(text("""
                INSERT INTO autowrite_runs (
                    book_id, chapter_id, section_slug, status, model,
                    target_words, max_iter, target_score, feature_versions
                ) VALUES (
                    CAST(:book_id AS uuid),
                    CAST(:chapter_id AS uuid),
                    :section_slug,
                    'running',
                    :model,
                    :target_words,
                    :max_iter,
                    :target_score,
                    CAST(:feature_versions AS jsonb)
                )
                RETURNING id::text
            """), {
                "book_id": book_id,
                "chapter_id": chapter_id,
                "section_slug": section_slug,
                "model": model,
                "target_words": target_words,
                "max_iter": max_iter,
                "target_score": target_score,
                "feature_versions": json.dumps(feature_versions or {}),
            }).fetchone()
            session.commit()
            return row[0] if row else None
    except Exception as exc:
        logger.warning("autowrite telemetry: failed to create run row: %s", exc)
        return None


def _persist_autowrite_retrievals(run_id: str | None, results: list) -> None:
    """Phase 32.6 — persist all retrieved chunks for a run.

    `results` is the list of SearchResult objects from
    context_builder.build(). Each entry's `rank` becomes the 1-indexed
    `[N]` source position the writer references; `chunk_id` is actually
    the qdrant_point_id (the existing convention).
    """
    if not run_id or not results:
        return
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    try:
        with get_session() as session:
            for r in results:
                session.execute(text("""
                    INSERT INTO autowrite_retrievals (
                        run_id, source_position, chunk_qdrant_id,
                        document_id, rrf_score
                    ) VALUES (
                        CAST(:run_id AS uuid),
                        :source_position,
                        CAST(:chunk_qdrant_id AS uuid),
                        CAST(:document_id AS uuid),
                        :rrf_score
                    )
                """), {
                    "run_id": run_id,
                    "source_position": getattr(r, "rank", 0) or 0,
                    "chunk_qdrant_id": getattr(r, "chunk_id", None),
                    "document_id": getattr(r, "document_id", None),
                    "rrf_score": float(getattr(r, "score", 0) or 0),
                })
            session.commit()
    except Exception as exc:
        logger.warning("autowrite telemetry: failed to persist retrievals: %s", exc)


def _persist_autowrite_iteration(
    run_id: str | None,
    iteration: int,
    history_entry: dict,
    *,
    word_count: int,
    word_count_delta: int | None,
    overall_pre: float,
    pre_revision_content: str | None = None,
    post_revision_content: str | None = None,
) -> None:
    """Phase 32.6 — persist one iteration's pre-revision state.

    Called right after `history.append(history_entry)` in the autowrite
    loop. Uses ON CONFLICT DO UPDATE so a later call from the post-revision
    update path can fill in `action`/`overall_post` without a separate
    UPDATE statement.

    Phase 32.9 — also accepts `pre_revision_content` and
    `post_revision_content` (Layer 4). Both are NULL on the first call
    (the pre-revision persist) and only `post_revision_content` is set
    on the second call (after the revision stream completes). The
    upsert uses COALESCE so an UPDATE with NULL doesn't clobber a
    value that was set on the prior call.
    """
    if not run_id:
        return
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    scores = history_entry.get("scores") or {}
    try:
        with get_session() as session:
            session.execute(text("""
                INSERT INTO autowrite_iterations (
                    run_id, iteration, scores, verification, cove,
                    action, word_count, word_count_delta,
                    weakest_dimension, revision_instruction,
                    overall_pre, overall_post,
                    pre_revision_content, post_revision_content
                ) VALUES (
                    CAST(:run_id AS uuid), :iteration,
                    CAST(:scores AS jsonb),
                    CAST(:verification AS jsonb),
                    CAST(:cove AS jsonb),
                    :action, :word_count, :word_count_delta,
                    :weakest_dimension, :revision_instruction,
                    :overall_pre, :overall_post,
                    :pre_revision_content, :post_revision_content
                )
                ON CONFLICT (run_id, iteration) DO UPDATE SET
                    scores = EXCLUDED.scores,
                    verification = EXCLUDED.verification,
                    cove = EXCLUDED.cove,
                    action = EXCLUDED.action,
                    word_count = EXCLUDED.word_count,
                    word_count_delta = EXCLUDED.word_count_delta,
                    weakest_dimension = EXCLUDED.weakest_dimension,
                    revision_instruction = EXCLUDED.revision_instruction,
                    overall_pre = EXCLUDED.overall_pre,
                    overall_post = EXCLUDED.overall_post,
                    -- Phase 32.9 — COALESCE so a NULL on the post-revision
                    -- update never clobbers the pre-revision content set
                    -- by the first persist call (and vice versa).
                    pre_revision_content = COALESCE(
                        EXCLUDED.pre_revision_content,
                        autowrite_iterations.pre_revision_content
                    ),
                    post_revision_content = COALESCE(
                        EXCLUDED.post_revision_content,
                        autowrite_iterations.post_revision_content
                    )
            """), {
                "run_id": run_id,
                "iteration": iteration,
                "scores": json.dumps(scores),
                "verification": json.dumps(history_entry.get("verification") or {}),
                "cove": json.dumps(history_entry.get("cove") or {}),
                "action": history_entry.get("revision_verdict"),
                "word_count": word_count,
                "word_count_delta": word_count_delta,
                "weakest_dimension": scores.get("weakest_dimension"),
                "revision_instruction": scores.get("revision_instruction"),
                "overall_pre": float(overall_pre) if overall_pre is not None else None,
                "overall_post": (
                    float(history_entry.get("post_revision_overall"))
                    if history_entry.get("post_revision_overall") is not None
                    else None
                ),
                "pre_revision_content": pre_revision_content,
                "post_revision_content": post_revision_content,
            })
            session.commit()
    except Exception as exc:
        logger.warning(
            "autowrite telemetry: failed to persist iteration %s: %s",
            iteration, exc,
        )


def _finalize_autowrite_run(
    run_id: str | None,
    *,
    status: str,
    final_draft_id: str | None,
    final_overall: float | None,
    iterations_used: int,
    converged: bool,
    error_message: str | None = None,
    tokens_used: int | None = None,
) -> None:
    """Phase 32.6 — close out a run row and back-fill `was_cited` flags.

    After updating the run row, parses the final draft's content for
    `[N]` markers and flips `was_cited=true` on the matching retrieval
    rows. This is the link from "what we retrieved" to "what we
    actually used" — the data Layer 2 will read.
    """
    if not run_id:
        return
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    try:
        with get_session() as session:
            session.execute(text("""
                UPDATE autowrite_runs SET
                    status = :status,
                    finished_at = now(),
                    final_draft_id = CAST(:final_draft_id AS uuid),
                    final_overall = :final_overall,
                    iterations_used = :iterations_used,
                    converged = :converged,
                    error_message = :error_message,
                    tokens_used = :tokens_used
                WHERE id = CAST(:run_id AS uuid)
            """), {
                "run_id": run_id,
                "status": status,
                "final_draft_id": final_draft_id,
                "final_overall": (
                    float(final_overall) if final_overall is not None else None
                ),
                "iterations_used": iterations_used,
                "converged": converged,
                "error_message": (error_message or "")[:1000] or None,
                "tokens_used": int(tokens_used) if tokens_used else None,
            })

            # Back-fill was_cited on the retrieval rows. The final draft's
            # text is the source of truth: any [N] marker that resolves to
            # a source position becomes was_cited=true. Iteration-by-
            # iteration cited tracking is intentionally NOT done — what
            # matters for Layer 2 is whether the chunk made it into the
            # final accepted draft, not which intermediate iterations
            # touched it.
            if final_draft_id:
                row = session.execute(text("""
                    SELECT content FROM drafts WHERE id::text = :did LIMIT 1
                """), {"did": final_draft_id}).fetchone()
                content = row[0] if row else None
                if content:
                    cited = sorted({
                        int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", content)
                    })
                    if cited:
                        # Build a parameterised IN list — psycopg2 supports
                        # tuple expansion for IN clauses with bound params.
                        placeholders = ",".join(
                            f":p{i}" for i, _ in enumerate(cited)
                        )
                        params: dict = {f"p{i}": p for i, p in enumerate(cited)}
                        params["run_id"] = run_id
                        session.execute(text(f"""
                            UPDATE autowrite_retrievals SET was_cited = true
                            WHERE run_id = CAST(:run_id AS uuid)
                              AND source_position IN ({placeholders})
                        """), params)
            session.commit()
    except Exception as exc:
        logger.warning("autowrite telemetry: failed to finalize run %s: %s",
                       run_id, exc)
        return

    # Phase 32.7 — Layer 1: distill lessons from this run's trajectory.
    # Only fires for completed runs (status='completed') with a real
    # final draft — error/cancelled runs have nothing useful to learn
    # from. Fail-soft: any error inside _distill_lessons_from_run logs
    # and returns 0; the autowrite still finishes cleanly for the user.
    if status == "completed" and final_draft_id:
        try:
            n = _distill_lessons_from_run(run_id)
            if n > 0:
                logger.info(
                    "autowrite layer 1: distilled %d lesson(s) from run %s",
                    n, run_id,
                )
        except Exception as exc:
            logger.warning("layer 1 distillation failed: %s", exc)


# ── Phase 32.7 — Compound learning Layer 1: episodic memory (lessons) ───
#
# Producer: _distill_lessons_from_run is called inline at the tail of
# _finalize_autowrite_run. It reads the per-iteration trajectory from
# autowrite_iterations (Layer 0), prompts the FAST model (per the MAR
# critique — different model than the scorer to avoid confirmation bias),
# parses 1-3 concrete lessons, embeds each via bge-m3, and persists to
# autowrite_lessons.
#
# Consumer: _get_relevant_lessons is called from _autowrite_section_body
# right before write_section_v2. It embeds the section query, fetches
# all lessons for the (book, section_slug) scope, computes cosine
# similarity in Python (the lesson table stays small enough that
# all-pairs in Python beats setting up pgvector), ranks by the
# Generative Agents formula `importance × recency_decay × similarity`,
# and returns the top-K lesson texts.
#
# All four helpers are fail-soft: any error logs and returns an empty
# result. Layer 1 is purely additive — a Layer 1 hiccup never blocks
# autowrite from completing.

# Default cap on how many lessons get injected into the writer prompt.
# Per the ERL paper (arXiv:2603.24639), unbounded lesson buffers scale
# poorly — context bloat kills both speed and quality. Top-K is the
# right pattern. Five is generous; in practice 3 hits the sweet spot
# for most sections.
_MAX_LESSONS_INJECTED = 5

# Recency decay half-life in days. A lesson at age=30 days is half as
# salient as a fresh one. Mirrors the Generative Agents 2023 setting.
_LESSON_RECENCY_HALF_LIFE_DAYS = 30.0


def _embed_text_for_lessons(text: str) -> list[float] | None:
    """Phase 32.7 — embed a single text string with bge-m3 dense.

    Reuses the same _embed_query helper that hybrid_search.py uses for
    query embedding so the lesson embeddings live in the same vector
    space as the chunk embeddings (no model swap).
    """
    if not text or not text.strip():
        return None
    try:
        from sciknow.retrieval.hybrid_search import _embed_query
        dense, _sparse = _embed_query(text.strip())
        return dense
    except Exception as exc:
        logger.warning("lesson embedding failed: %s", exc)
        return None


_VALID_LESSON_KINDS: tuple[str, ...] = (
    "episode", "knowledge", "idea", "decision", "rejected_idea", "paper",
)

_VALID_LESSON_SCOPES: tuple[str, ...] = ("book", "global")


def _normalize_kind(kind: str | None) -> str:
    """Accept any string, return a valid kind or fall back to ``episode``.

    The LLM's output is free-form so we defensively coerce — any kind
    we don't recognise becomes ``episode`` (the legacy default) rather
    than crashing the persist call.
    """
    k = (kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    return k if k in _VALID_LESSON_KINDS else "episode"


def _persist_lesson(
    *,
    book_id: str | None,
    chapter_id: str | None,
    section_slug: str,
    lesson_text: str,
    source_run_id: str | None,
    score_delta: float | None,
    embedding: list[float] | None,
    importance: float,
    dimension: str | None,
    kind: str | None = None,
    scope: str = "book",
) -> None:
    """Phase 32.7 — INSERT one lesson row. Fail-soft.

    Phase 47.1 — accepts ``kind`` and ``scope`` kwargs. ``kind`` is
    coerced to a known value (unknowns become ``episode``). ``scope``
    is validated (must be ``book`` or ``global``) — a global scope
    implies ``book_id=None`` to satisfy the CHECK constraint from
    migration 0018.
    """
    if not lesson_text or not lesson_text.strip():
        return
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    k = _normalize_kind(kind)
    scope = scope if scope in _VALID_LESSON_SCOPES else "book"
    # Global scope must have book_id=NULL (CK_ autowrite_lessons_scope_book).
    if scope == "global":
        book_id = None
        chapter_id = None

    try:
        with get_session() as session:
            session.execute(text("""
                INSERT INTO autowrite_lessons (
                    book_id, chapter_id, section_slug,
                    lesson_text, source_run_id, score_delta,
                    embedding, importance, dimension, kind, scope
                ) VALUES (
                    CAST(:book_id AS uuid),
                    CAST(:chapter_id AS uuid),
                    :section_slug,
                    :lesson_text,
                    CAST(:source_run_id AS uuid),
                    :score_delta,
                    CAST(:embedding AS real[]),
                    :importance,
                    :dimension,
                    :kind,
                    :scope
                )
            """), {
                "book_id": book_id,
                "chapter_id": chapter_id,
                "section_slug": section_slug,
                "lesson_text": lesson_text.strip()[:1000],
                "source_run_id": source_run_id,
                "score_delta": float(score_delta) if score_delta is not None else None,
                # PG accepts a Python list as a real[] when bound via
                # the array literal cast.
                "embedding": embedding,
                "importance": float(importance),
                "dimension": dimension,
                "kind": k,
                "scope": scope,
            })
            session.commit()
    except Exception as exc:
        logger.warning("autowrite telemetry: failed to persist lesson: %s", exc)


def _distill_lessons_from_run(run_id: str | None) -> int:
    """Phase 32.7 — read a completed run's iteration history, prompt the
    FAST model to extract 1-3 lessons, embed them, and persist to
    autowrite_lessons. Returns the number of lessons persisted.

    Called inline at the tail of `_finalize_autowrite_run` for runs with
    status='completed'. Uses settings.llm_fast_model so the writer model
    stays warm in VRAM (no swap penalty) AND it's a different model
    than the scorer (mitigates the MAR confirmation-bias issue).
    """
    if not run_id:
        return 0
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    from sciknow.config import settings
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import complete as llm_complete

    try:
        # Read the run + iterations from the DB. We re-fetch instead of
        # passing them through as parameters because _distill is called
        # from _finalize, and the iteration data the finalizer has is
        # not in the right shape (it's the in-memory `history` list,
        # which has slightly different field names than the DB columns).
        with get_session() as session:
            run_row = session.execute(text("""
                SELECT book_id::text, chapter_id::text, section_slug,
                       final_overall, iterations_used, converged
                FROM autowrite_runs WHERE id::text = :id
            """), {"id": run_id}).fetchone()
            if not run_row:
                return 0
            book_id, chapter_id, section_slug, final_overall, iters_used, converged = run_row

            iter_rows = session.execute(text("""
                SELECT iteration, scores, action, weakest_dimension,
                       revision_instruction, word_count, word_count_delta,
                       overall_pre, overall_post
                FROM autowrite_iterations
                WHERE run_id::text = :id
                ORDER BY iteration
            """), {"id": run_id}).fetchall()

        if not iter_rows:
            # No iterations to learn from — skip silently.
            return 0

        # Compute the score delta: final - first iteration's overall_pre.
        first_overall = iter_rows[0][7]  # overall_pre
        score_delta = (
            (float(final_overall) - float(first_overall))
            if (final_overall is not None and first_overall is not None)
            else 0.0
        )

        # Build the iterations list in the shape distill_lessons() expects.
        iterations = [
            {
                "iteration": r[0],
                "scores": r[1] or {},
                "action": r[2],
                "weakest_dimension": r[3],
                "revision_instruction": r[4],
                "word_count": r[5],
                "word_count_delta": r[6],
            }
            for r in iter_rows
        ]

        sys_p, usr_p = rag_prompts.distill_lessons(
            section_slug=section_slug,
            final_overall=float(final_overall or 0.0),
            score_delta=score_delta,
            iterations_used=int(iters_used or 0),
            converged=bool(converged),
            iterations=iterations,
        )

        # Use the FAST model — different from the writer/scorer (MAR
        # critique) AND avoids a model swap since the fast model is
        # already loaded for metadata extraction.
        raw = llm_complete(
            sys_p, usr_p,
            model=settings.llm_fast_model,
            temperature=0.2, num_ctx=4096, keep_alive=-1,
        )
        # Strip thinking blocks the same way other parsers do.
        raw = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()
        try:
            parsed = json.loads(_clean_json(raw), strict=False)
        except Exception as exc:
            logger.warning(
                "lesson distillation: failed to parse JSON from LLM: %s | raw: %s",
                exc, raw[:200],
            )
            return 0

        lessons = parsed.get("lessons") or []
        if not isinstance(lessons, list):
            return 0

        # Importance starts at 1.0 + a small bonus for high score delta —
        # a run that genuinely improved during iteration produced a
        # stronger learning signal than one that converged immediately.
        importance_bonus = max(0.0, min(0.5, float(score_delta) * 2.0))
        importance = 1.0 + importance_bonus

        n_persisted = 0
        for ls in lessons[:3]:  # hard cap at 3 per run
            if not isinstance(ls, dict):
                continue
            text_val = (ls.get("text") or "").strip()
            dim  = (ls.get("dimension") or "general").strip().lower()
            # Phase 47.1 — kind is LLM-provided; unknowns coerce to 'episode'
            kind = _normalize_kind(ls.get("kind"))
            if not text_val:
                continue
            embedding = _embed_text_for_lessons(text_val)
            _persist_lesson(
                book_id=book_id,
                chapter_id=chapter_id,
                section_slug=section_slug,
                lesson_text=text_val,
                source_run_id=run_id,
                score_delta=score_delta,
                embedding=embedding,
                importance=importance,
                dimension=dim,
                kind=kind,
                scope="book",
            )
            n_persisted += 1
        return n_persisted
    except Exception as exc:
        logger.warning("lesson distillation: top-level failure: %s", exc)
        return 0


def _get_relevant_lessons(
    book_id: str | None,
    section_slug: str,
    query_text: str,
    *,
    top_k: int = _MAX_LESSONS_INJECTED,
    kinds: tuple[str, ...] | list[str] | None = None,
    include_global: bool = True,
    return_dicts: bool = False,
) -> list:
    """Phase 32.7 — fetch the top-K relevant lessons for an upcoming
    autowrite run.

    Scope: lessons from the SAME (book, section_slug) AND lessons from
    the SAME section_slug across other books. The former are tied to
    the user's specific book; the latter generalize across books on
    similar section types. Cross-book lessons are slightly downweighted
    by giving them a fixed importance multiplier of 0.7.

    Phase 47.1 — optional ``kinds`` filter (e.g. ``kinds=("idea", "knowledge")``
    to get only positive lessons, or ``kinds=("rejected_idea",)`` for
    the gap-finder gate). ``include_global=True`` (default) unions
    scope=global lessons too — the cross-book pool populated by
    Phase 47.4 ``promote_to_global``. Pass ``return_dicts=True`` to get
    ``[{"text", "kind", "dimension", "score", ...}]`` instead of raw
    strings — Phase 47.3 writer prompt uses this to group by kind.

    Ranking: `importance × recency_decay × cosine_similarity`. Recency
    decay is `2^(-age_days / half_life)` so a 30-day-old lesson is at
    half its stored importance.

    Returns a list of lesson texts (or dicts). Empty list on cold-start
    (no lessons exist yet) or any failure.
    """
    if not section_slug:
        return []
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    # Normalise the kinds filter into a set of valid values.
    kinds_set: set[str] | None = None
    if kinds:
        kinds_set = {k for k in (_normalize_kind(k) for k in kinds) if k}
        # If every provided kind normalises away, treat as no filter
        if not kinds_set:
            kinds_set = None

    try:
        # Embed the query text first. If embedding fails (e.g. embedder
        # OOM), fall back to non-similarity ranking — just importance ×
        # recency, which still surfaces the most useful lessons.
        query_emb = _embed_text_for_lessons(query_text)

        # Scope filter: default includes the active-book lessons AND any
        # globally-promoted ones. ``scope`` is indexed via the 0018
        # migration so this stays cheap at 10k+ rows.
        scope_filter = (
            "(scope = 'book' OR scope = 'global')"
            if include_global else
            "scope = 'book'"
        )

        with get_session() as session:
            rows = session.execute(text(f"""
                SELECT id::text, lesson_text, embedding, importance,
                       book_id::text, dimension, kind, scope,
                       EXTRACT(EPOCH FROM (now() - created_at)) / 86400.0 AS age_days
                FROM autowrite_lessons
                WHERE section_slug = :slug
                  AND lesson_text IS NOT NULL
                  AND {scope_filter}
                ORDER BY created_at DESC
                LIMIT 200
            """), {"slug": section_slug}).fetchall()

        if not rows:
            return []

        import math
        scored: list[tuple[float, dict]] = []
        for row in rows:
            (lesson_id, lesson_text, embedding, importance,
             l_book_id, dim, kind, scope, age_days) = row
            if not lesson_text:
                continue
            # Phase 47.1 — kind filter
            if kinds_set is not None and (kind or "episode") not in kinds_set:
                continue
            # Recency decay: exp(-age * ln(2) / half_life) = 2^(-age/half_life)
            recency = 2.0 ** (-(float(age_days or 0)) / _LESSON_RECENCY_HALF_LIFE_DAYS)
            imp = float(importance or 1.0)
            # Cross-book book-scoped lessons get a downweight — they're
            # generalized but less specific to this user's book. Global
            # promoted lessons are slightly less aggressive (0.85) since
            # the promote_to_global gate (Phase 47.4) already filtered
            # for multi-book presence.
            if scope == "global":
                imp *= 0.85
            elif book_id and l_book_id and l_book_id != book_id:
                imp *= 0.7
            # Cosine similarity if both embeddings are present
            if query_emb and embedding:
                try:
                    a = query_emb
                    b = list(embedding)
                    if len(a) == len(b):
                        dot = sum(x * y for x, y in zip(a, b))
                        na = math.sqrt(sum(x * x for x in a))
                        nb = math.sqrt(sum(y * y for y in b))
                        sim = dot / (na * nb) if na > 0 and nb > 0 else 0.0
                    else:
                        sim = 0.0
                except Exception:
                    sim = 0.0
            else:
                # No embedding ⇒ neutral similarity. Lesson can still
                # win on importance × recency alone.
                sim = 0.5
            score = imp * recency * (sim + 0.1)  # +0.1 floor so high-importance still wins on weak similarity
            scored.append((score, {
                "text":      lesson_text,
                "kind":      kind or "episode",
                "dimension": dim or "general",
                "scope":     scope or "book",
                "score":     round(float(score), 4),
                "same_book": (l_book_id == book_id) if book_id and l_book_id else False,
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [d for _, d in scored[:top_k]]
        if return_dicts:
            return top
        return [d["text"] for d in top]
    except Exception as exc:
        logger.warning("lesson retrieval failed: %s", exc)
        return []


# ════════════════════════════════════════════════════════════════════
# Phase 47.4 — Cross-book lesson promotion (DeepScientist-style)
# ════════════════════════════════════════════════════════════════════
#
# DeepScientist's Findings Memory (arXiv:2509.26603 §3.2) maintains two
# scopes — per-quest and global — with an explicit ``promote_to_global``
# action. sciknow's autowrite_lessons already has the Phase 47.1 scope
# column; this section is the promotion service.
#
# Promotion gate (all three must hold):
#   1. importance >= _PROMOTE_MIN_IMPORTANCE (default 0.8)
#   2. score_delta > 0  — the source run actually improved during
#      iteration (a high-importance lesson from a run that converged
#      immediately is less trustworthy)
#   3. the lesson appears in >= _PROMOTE_MIN_BOOKS distinct books via
#      embedding similarity >= _PROMOTE_COSINE (default 0.85)
#
# The third gate is the important one — it distinguishes a "general"
# writing lesson from a book-idiosyncratic tic. If three independent
# climate / materials / linguistics books all surface the same
# "don't overstate causality on correlations" lesson, that's global
# knowledge worth promoting.
#
# Promoted lessons are NEW rows with scope='global' and book_id=NULL
# (enforced by the migration-0018 CHECK constraint). The book-scoped
# originals are kept — promotion is additive, not destructive.


_PROMOTE_MIN_IMPORTANCE = 0.8
_PROMOTE_MIN_BOOKS      = 3
_PROMOTE_COSINE         = 0.85
_PROMOTE_MAX_BATCH      = 50    # safety cap per run


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity between two 1024-d bge-m3 vectors."""
    if not a or not b:
        return 0.0
    import math
    try:
        aa = list(a); bb = list(b)
        if len(aa) != len(bb):
            return 0.0
        dot = sum(x * y for x, y in zip(aa, bb))
        na = math.sqrt(sum(x * x for x in aa))
        nb = math.sqrt(sum(y * y for y in bb))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0
    except Exception:
        return 0.0


def promote_lessons_to_global(
    *,
    dry_run: bool = False,
    min_importance: float = _PROMOTE_MIN_IMPORTANCE,
    min_books: int       = _PROMOTE_MIN_BOOKS,
    cosine_threshold: float = _PROMOTE_COSINE,
    limit: int           = _PROMOTE_MAX_BATCH,
) -> dict:
    """Phase 47.4 — promote book-scoped lessons to the global pool.

    Iterates book-scoped lessons that pass the importance + score_delta
    gates, buckets them by embedding similarity, and promotes buckets
    that span >= ``min_books`` distinct books. Returns a summary dict
    so the CLI can report what happened.

    Idempotent: if a lesson was previously promoted (a near-duplicate
    already exists at scope='global'), we skip re-promotion.
    """
    from sqlalchemy import text as _text
    from sciknow.storage.db import get_session

    summary: dict = {
        "candidates": 0, "buckets": 0, "promoted": 0,
        "skipped_already_global": 0, "skipped_too_few_books": 0,
        "dry_run": dry_run,
        "details": [],
    }

    try:
        with get_session() as session:
            # Candidate pool: book-scoped, non-null embedding, non-null
            # book_id, passes importance + score_delta gates. Pull only
            # what we need to avoid round-tripping lesson_text for rows
            # we'll reject.
            rows = session.execute(_text("""
                SELECT id::text, book_id::text, section_slug,
                       lesson_text, embedding, importance, dimension, kind,
                       score_delta, created_at
                FROM autowrite_lessons
                WHERE scope = 'book'
                  AND embedding IS NOT NULL
                  AND book_id IS NOT NULL
                  AND importance >= :min_imp
                  AND (score_delta IS NULL OR score_delta > 0)
                ORDER BY importance DESC, created_at DESC
            """), {"min_imp": float(min_importance)}).fetchall()

            # Pre-load existing global embeddings so we can avoid
            # promoting near-duplicates.
            global_rows = session.execute(_text("""
                SELECT embedding FROM autowrite_lessons
                WHERE scope = 'global' AND embedding IS NOT NULL
            """)).fetchall()
            global_embeddings = [list(r[0]) for r in global_rows if r[0]]
    except Exception as exc:
        logger.warning("promote: candidate fetch failed: %s", exc)
        summary["error"] = str(exc)
        return summary

    summary["candidates"] = len(rows)
    if not rows:
        return summary

    # Bucket by embedding similarity (union-find style). A bucket is
    # "represented" by the first-seen candidate's embedding; later
    # candidates join if cosine >= threshold.
    buckets: list[dict] = []
    for r in rows:
        rid, book_id, slug, txt, emb, imp, dim, kind, sdelta, created = r
        emb_list = list(emb) if emb else None
        placed = False
        for b in buckets:
            if _cosine(emb_list, b["rep_emb"]) >= cosine_threshold:
                if book_id not in b["books"]:
                    b["books"].add(book_id)
                b["members"].append({
                    "id": rid, "book_id": book_id, "section_slug": slug,
                    "text": txt, "importance": float(imp or 1.0),
                    "kind": kind, "dimension": dim,
                    "score_delta": float(sdelta) if sdelta is not None else None,
                    "embedding": emb_list,
                })
                placed = True
                break
        if not placed:
            buckets.append({
                "rep_emb":  emb_list,
                "rep_text": txt,
                "rep_kind": kind,
                "rep_dim":  dim,
                "rep_section_slug": slug,
                "books":    {book_id},
                "members":  [{
                    "id": rid, "book_id": book_id, "section_slug": slug,
                    "text": txt, "importance": float(imp or 1.0),
                    "kind": kind, "dimension": dim,
                    "score_delta": float(sdelta) if sdelta is not None else None,
                    "embedding": emb_list,
                }],
            })

    summary["buckets"] = len(buckets)
    logger.info(
        "promote_to_global: %d candidates → %d buckets (cosine>=%.2f)",
        len(rows), len(buckets), cosine_threshold,
    )

    # Promote buckets that span enough distinct books
    to_promote: list[dict] = []
    for b in buckets:
        if len(b["books"]) < min_books:
            summary["skipped_too_few_books"] += 1
            continue
        # Skip if a near-duplicate already exists in global
        is_dupe = False
        for gemb in global_embeddings:
            if _cosine(b["rep_emb"], gemb) >= cosine_threshold:
                is_dupe = True
                break
        if is_dupe:
            summary["skipped_already_global"] += 1
            continue
        # Choose a representative text — the highest-importance
        # member, then longest text as a tie-breaker.
        rep = max(
            b["members"],
            key=lambda m: (m["importance"], len(m["text"] or "")),
        )
        to_promote.append({
            "bucket_size": len(b["members"]),
            "n_books":     len(b["books"]),
            "text":        rep["text"],
            "kind":        rep["kind"] or "episode",
            "dimension":   rep["dimension"] or "general",
            "importance":  rep["importance"],
            "embedding":   rep["embedding"],
            "section_slug": rep["section_slug"],
        })
        if len(to_promote) >= limit:
            break

    if dry_run:
        summary["details"] = [
            {
                "text": p["text"][:140],
                "bucket_size": p["bucket_size"],
                "n_books": p["n_books"],
                "kind": p["kind"],
                "dimension": p["dimension"],
            }
            for p in to_promote
        ]
        return summary

    # Actually write them
    n_ok = 0
    for p in to_promote:
        try:
            _persist_lesson(
                book_id=None, chapter_id=None,
                section_slug=p["section_slug"],
                lesson_text=p["text"],
                source_run_id=None,
                score_delta=None,
                embedding=p["embedding"],
                importance=p["importance"],
                dimension=p["dimension"],
                kind=p["kind"],
                scope="global",
            )
            n_ok += 1
        except Exception as exc:
            logger.warning("promote: persist failed: %s", exc)
    summary["promoted"] = n_ok
    summary["details"] = [
        {
            "text": p["text"][:140],
            "bucket_size": p["bucket_size"],
            "n_books": p["n_books"],
            "kind": p["kind"],
            "dimension": p["dimension"],
        }
        for p in to_promote[:n_ok]
    ]
    return summary


# ── Phase 32.9 — Compound learning Layer 4: DPO preference dataset ──────
#
# Each KEEP verdict in the autowrite loop is a preference pair: the
# revised draft beat the pre-revision draft. Each DISCARD is the inverse:
# the pre-revision draft beat the revision attempt. We capture BOTH
# (Phase 32.6 captured the verdicts; Phase 32.9 added the actual content
# columns) and export them as standard `{prompt, chosen, rejected}` JSONL
# for future DPO fine-tuning when the DGX Spark arrives (Layer 6).
#
# Filter rules (configurable via the CLI flags):
#   - Drop pairs where BOTH overall_pre and overall_post are below the
#     `min_score` floor (default 0.7) — low signal on both sides.
#   - Drop pairs where the score gap is below `min_delta` (default 0.02)
#     — too noisy to learn from.
#   - With --require-approval, only keep pairs from runs where the
#     final draft is marked approved=true in drafts.custom_metadata.
#     This is the human-in-the-loop bias mitigation from RESEARCH.md §21.


def _export_preference_pairs(
    *,
    book_id: str | None = None,
    output_path: "Path | None" = None,
    min_score: float = 0.7,
    min_delta: float = 0.02,
    require_approval: bool = False,
    include_discard: bool = True,
) -> tuple[int, "Path"]:
    """Phase 32.9 — Layer 4: walk autowrite_iterations and export
    preference pairs as JSONL.

    Returns (n_pairs_written, output_path).

    Pair shape (one per JSONL line):
        {
          "prompt": str,                 # chapter title + section + topic
          "chosen": str,                 # the higher-scored draft text
          "rejected": str,               # the lower-scored draft text
          "score_chosen": float,
          "score_rejected": float,
          "score_delta": float,
          "verdict": "KEEP" | "DISCARD",
          "section_slug": str,
          "iteration": int,
          "run_id": str,
          "feature_versions": dict,      # which Phase X features were on
          "model": str,
        }
    """
    from pathlib import Path
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    from sciknow.config import settings

    if output_path is None:
        if book_id:
            output_path = settings.data_dir / "preferences" / f"book_{book_id[:8]}.jsonl"
        else:
            output_path = settings.data_dir / "preferences" / "all_books.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pull the joinable rows: every iteration with both content columns
    # populated belongs to a completed run.
    where_clauses = [
        "i.pre_revision_content IS NOT NULL",
        "i.post_revision_content IS NOT NULL",
        "i.action IS NOT NULL",
        "r.status = 'completed'",
    ]
    params: dict = {
        "min_score": float(min_score),
        "min_delta": float(min_delta),
    }
    if book_id:
        where_clauses.append("r.book_id::text = :book_id")
        params["book_id"] = book_id

    with get_session() as session:
        rows = session.execute(text(f"""
            SELECT
                i.run_id::text,
                i.iteration,
                i.action,
                i.pre_revision_content,
                i.post_revision_content,
                i.overall_pre,
                i.overall_post,
                i.weakest_dimension,
                r.section_slug,
                r.model,
                r.feature_versions,
                r.book_id::text,
                r.chapter_id::text,
                r.final_draft_id::text,
                COALESCE(d.topic, '') AS topic,
                COALESCE(bc.title, '') AS chapter_title,
                COALESCE(bc.number, 0) AS chapter_num
            FROM autowrite_iterations i
            JOIN autowrite_runs r ON r.id = i.run_id
            LEFT JOIN drafts d ON d.id = r.final_draft_id
            LEFT JOIN book_chapters bc ON bc.id = r.chapter_id
            WHERE {' AND '.join(where_clauses)}
            ORDER BY r.started_at, i.iteration
        """), params).fetchall()

    n_written = 0
    n_skipped_low_score = 0
    n_skipped_small_delta = 0
    n_skipped_unapproved = 0
    n_skipped_discard = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            (run_id, iteration, action, pre_text, post_text,
             overall_pre, overall_post, weakest, section_slug, model,
             feature_versions, _book, _chapter, final_draft_id,
             topic, chapter_title, chapter_num) = row

            pre = float(overall_pre or 0.0)
            post = float(overall_post or 0.0)
            delta = abs(post - pre)

            # Filter: both sides below the floor
            if max(pre, post) < min_score:
                n_skipped_low_score += 1
                continue
            # Filter: delta too small to learn from
            if delta < min_delta:
                n_skipped_small_delta += 1
                continue
            # Filter: DISCARD verdicts (only if explicitly excluded)
            if not include_discard and action == "DISCARD":
                n_skipped_discard += 1
                continue

            # Build chosen/rejected based on the verdict
            if action == "KEEP":
                chosen, rejected = post_text, pre_text
                score_chosen, score_rejected = post, pre
            elif action == "DISCARD":
                chosen, rejected = pre_text, post_text
                score_chosen, score_rejected = pre, post
            else:
                continue

            # Optional approval gate (human-in-the-loop bias mitigation).
            # Require the final draft to have custom_metadata.preference_approved = true.
            if require_approval:
                if not final_draft_id:
                    n_skipped_unapproved += 1
                    continue
                with get_session() as s:
                    meta = s.execute(text(
                        "SELECT custom_metadata FROM drafts WHERE id::text = :id"
                    ), {"id": final_draft_id}).scalar()
                if not isinstance(meta, dict) or not meta.get("preference_approved"):
                    n_skipped_unapproved += 1
                    continue

            prompt_text = (
                f"Chapter {chapter_num}: {chapter_title}\n"
                f"Section: {section_slug}\n"
                f"Topic: {topic}".strip()
            )
            record = {
                "prompt": prompt_text,
                "chosen": chosen,
                "rejected": rejected,
                "score_chosen": round(score_chosen, 4),
                "score_rejected": round(score_rejected, 4),
                "score_delta": round(score_chosen - score_rejected, 4),
                "verdict": action,
                "section_slug": section_slug,
                "iteration": iteration,
                "run_id": run_id,
                "feature_versions": feature_versions or {},
                "model": model,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    logger.info(
        "exported %d preference pairs to %s "
        "(skipped: %d low-score, %d small-delta, %d unapproved, %d discard)",
        n_written, output_path,
        n_skipped_low_score, n_skipped_small_delta,
        n_skipped_unapproved, n_skipped_discard,
    )
    return n_written, output_path


def _save_draft(session, *, title, book_id, chapter_id, section_type, topic,
                content, sources, model, summary=None, parent_draft_id=None,
                review_feedback=None, version=1, custom_metadata=None):
    from sqlalchemy import text
    # Note: use CAST(:x AS jsonb) instead of :x::jsonb because SQLAlchemy's
    # parameter parser confuses `::jsonb` with the bound-param name and
    # passes it through unparameterized. CAST(...) is unambiguous.
    row = session.execute(text("""
        INSERT INTO drafts (title, book_id, chapter_id, section_type, topic, content,
                            word_count, sources, model_used, version, summary,
                            parent_draft_id, review_feedback, custom_metadata)
        VALUES (:title, :book_id, :chapter_id, :section, :topic, :content,
                :wc, CAST(:sources AS jsonb), :model, :version, :summary,
                :parent_id, :review_feedback, CAST(:metadata AS jsonb))
        RETURNING id::text
    """), {
        "title": title, "book_id": book_id, "chapter_id": chapter_id,
        "section": section_type, "topic": topic, "content": content,
        "wc": len(content.split()),
        "sources": json.dumps(sources or []),
        "model": model, "version": version, "summary": summary,
        "parent_id": parent_draft_id, "review_feedback": review_feedback,
        "metadata": json.dumps(custom_metadata or {}),
    })
    session.commit()
    result = row.fetchone()
    if not result:
        raise RuntimeError("Failed to save draft — INSERT returned no rows")
    return result[0]


def _release_gpu_models():
    """Free bge-m3 embedder + reranker from VRAM so Ollama can use the full GPU."""
    from sciknow.ingestion.embedder import release_model
    from sciknow.retrieval.hybrid_search import release_embed_model
    from sciknow.retrieval.reranker import release_reranker
    release_embed_model()
    release_model()
    release_reranker()


def _stream_phase(system: str, user: str, phase: str, *, model=None,
                  token_observer=None, **kw):
    """Phase 15.3 — generator that streams an LLM call as token events.

    Wraps llm.stream() so each token becomes
    {"type": "token", "text": tok, "phase": phase}, and returns the full
    accumulated text via the generator's StopIteration.value (captured by
    `yield from`).

    Used by autowrite_section_stream to make scoring/verification/CoVe/
    tree-plan calls visible to the GUI's live token counter. Without this,
    those phases use the synchronous llm.complete() and emit zero token
    events even though the LLM is working hard — the user sees a stuck
    "0 tok / 0 tok/s" while the timer counts up.

    Phase 24 — accepts an optional ``token_observer`` callback (e.g.
    ``log.token``) so the autowrite progress logger can record real-
    time token throughput per phase. Cheap (one function call per
    token); silently swallows callback errors so a busted observer
    can't kill the stream.

    Phase 54.6.30 — defaults ``keep_alive=-1`` on the underlying
    llm_stream call so every autowrite phase that routes through
    this helper (planning, scoring, verifying, rescoring, CoVe) is
    explicitly sticky at the call site. Callers can override by
    passing ``keep_alive=...`` in ``**kw``.

    Usage in a generator (re-yields events to consumer):
        raw = yield from _stream_phase(sys, usr, "scoring",
                                       model=model, num_ctx=16384)
        scores = json.loads(_clean_json(raw))
    """
    from sciknow.rag.llm import stream as llm_stream
    kw.setdefault("keep_alive", -1)
    tokens: list[str] = []
    for tok in llm_stream(system, user, model=model, **kw):
        tokens.append(tok)
        if token_observer is not None:
            try:
                token_observer()
            except Exception:
                pass
        yield {"type": "token", "text": tok, "phase": phase}
    return "".join(tokens)


# Phase 19 — incremental save during streaming. Without this, clicking
# Stop during the writing/revising phase loses every token accumulated
# since the last checkpoint (which can be hundreds of words). The fix:
# every N tokens / T seconds, persist the in-flight buffer to the
# draft row. The try/finally guarantees that even GeneratorExit
# (raised when the consumer calls gen.close() on Stop) flushes the
# latest buffer before unwinding.

# Tunables: 150 tokens (~one paragraph at typical token granularity)
# OR 5 seconds, whichever fires first. Picked to keep DB writes light
# (≤12/min on slow streams, ~3/min on fast ones) while keeping the
# worst-case data loss under one paragraph.
_STREAM_SAVE_INTERVAL_TOKENS = 150
_STREAM_SAVE_INTERVAL_SECONDS = 5.0


def _stream_with_save(
    system: str,
    user: str,
    phase: str,
    *,
    model=None,
    save_callback,
    save_interval_tokens: int = _STREAM_SAVE_INTERVAL_TOKENS,
    save_interval_seconds: float = _STREAM_SAVE_INTERVAL_SECONDS,
    token_observer=None,
    **stream_kw,
):
    """Phase 19 — generator that streams an LLM call AND periodically
    flushes the accumulated buffer via ``save_callback``.

    The callback is invoked with a single positional arg: the full
    accumulated text so far. It MUST be a regular function (not a
    generator) — yields are forbidden inside the finally block of a
    closing generator. Exceptions raised by the callback are logged
    and swallowed so a single failed save doesn't kill the loop.

    On consumer Stop:
      1. The web layer calls gen.close() on the outer autowrite generator.
      2. GeneratorExit propagates through `yield from` into THIS generator.
      3. The finally block runs and calls save_callback one last time.
      4. The user sees the latest in-flight content on next refresh.

    Returns the full accumulated text via StopIteration.value, captured
    by `yield from` in the caller — same pattern as _stream_phase.

    Usage:
        def _save(text):
            _update_draft_content(draft_id, text, custom_metadata=meta)
        content = yield from _stream_with_save(
            sys, usr, "writing", model=model, save_callback=_save,
        )
    """
    from sciknow.rag.llm import stream as llm_stream
    tokens: list[str] = []
    last_save_t = time.monotonic()
    last_save_len = 0

    def _flush() -> None:
        if not tokens:
            return
        try:
            save_callback("".join(tokens))
        except Exception as exc:
            logger.warning("Streaming save (%s) failed: %s", phase, exc)

    # Phase 54.6.30 — sticky model by default for the writing loop.
    stream_kw.setdefault("keep_alive", -1)

    try:
        for tok in llm_stream(system, user, model=model, **stream_kw):
            tokens.append(tok)
            # Phase 24 — notify the autowrite logger so its heartbeat
            # can show real-time tokens-per-sec. Cheap (lock + 2 adds)
            # and only fires when an observer is wired in.
            if token_observer is not None:
                try:
                    token_observer()
                except Exception:
                    pass
            yield {"type": "token", "text": tok, "phase": phase}
            now = time.monotonic()
            if (
                len(tokens) - last_save_len >= save_interval_tokens
                or now - last_save_t >= save_interval_seconds
            ):
                _flush()
                last_save_t = now
                last_save_len = len(tokens)
    finally:
        # Always flush the latest buffer, even on GeneratorExit /
        # exceptions. This is the critical bit: it's what makes Stop
        # preserve work-in-progress instead of throwing it away.
        _flush()
    return "".join(tokens)


# Phase 28 — autowrite resume eligibility.
#
# A draft is "resumable" if it represents a complete, coherent body
# of text — NOT a partial token buffer left over from an interrupted
# writing or revising stream. The criterion is the
# custom_metadata.checkpoint string written by the autowrite generator
# at every transition (Phase 19 + 24):
#
#   "writing_in_progress"   ← partial: writer was streaming, got cut off
#   "placeholder"           ← partial: row was just inserted, no content yet
#   "iteration_N_revising"  ← partial: a revision was streaming, got cut off
#   "initial"               ← FINISHED: writer completed, no iterations yet
#   "iteration_N_keep"      ← FINISHED: KEEP verdict applied, between iters
#   "iteration_N_discard"   ← FINISHED: DISCARD verdict applied, between iters
#   "final"                 ← FINISHED: full convergence loop completed
#   "draft"                 ← FINISHED: manual write_section_stream done
#
# Drafts created BEFORE the checkpoint system was added (very old)
# have no checkpoint key at all — we conservatively allow those if
# they have a non-trivial word count, since they predate the partial
# state tracking.

_RESUMABLE_CHECKPOINTS = frozenset({
    "initial", "final", "draft",
})
_RESUMABLE_CHECKPOINT_PREFIXES = ("iteration_",)
_RESUMABLE_CHECKPOINT_SUFFIXES = ("_keep", "_discard")
_PARTIAL_CHECKPOINTS = frozenset({
    "writing_in_progress", "placeholder",
})
_PARTIAL_CHECKPOINT_SUFFIXES = ("_revising",)
# Drafts shorter than this are treated as "no real content yet" and
# refused for resume regardless of checkpoint state.
_MIN_RESUMABLE_WORDS = 100


def _is_resumable_draft(custom_metadata, word_count: int | None) -> tuple[bool, str]:
    """Phase 28 — return (is_resumable, reason). Used by autowrite resume
    mode to refuse picking up partial drafts.

    The reason string is empty when ok and contains a human-readable
    explanation otherwise (used by the CLI / GUI / log entries).
    """
    wc = int(word_count or 0)
    if wc < _MIN_RESUMABLE_WORDS:
        return False, (
            f"draft has only {wc} words (need >= {_MIN_RESUMABLE_WORDS}); "
            "too short to resume from — looks like an empty placeholder"
        )

    meta = custom_metadata or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    checkpoint = (meta.get("checkpoint") or "").strip()

    if not checkpoint:
        # Pre-checkpoint era (very old drafts). Allow with a warning
        # baked into the reason — the caller can surface it.
        return True, "draft predates the checkpoint system; resuming anyway"

    # Reject partial states explicitly.
    if checkpoint in _PARTIAL_CHECKPOINTS:
        return False, (
            f"draft is in partial state '{checkpoint}' "
            "(writing was interrupted; not safe to resume)"
        )
    if any(checkpoint.endswith(s) for s in _PARTIAL_CHECKPOINT_SUFFIXES):
        return False, (
            f"draft is in partial revising state '{checkpoint}' "
            "(a revision stream was interrupted; not safe to resume)"
        )

    # Accept known-finished states.
    if checkpoint in _RESUMABLE_CHECKPOINTS:
        return True, ""
    if any(checkpoint.startswith(p) for p in _RESUMABLE_CHECKPOINT_PREFIXES) \
       and any(checkpoint.endswith(s) for s in _RESUMABLE_CHECKPOINT_SUFFIXES):
        return True, ""

    # Unknown checkpoint string — be conservative and refuse with
    # a clear message rather than guessing.
    return False, (
        f"draft has unknown checkpoint state '{checkpoint}' — "
        "refusing to resume (run with --rebuild to start fresh)"
    )


def _next_draft_version(session, book_id: str, chapter_id: str, section_type: str) -> int:
    """Phase 19 — return the version number a new draft should claim so
    it'll be the latest visible to the GUI's max-version-per-section sort.

    Without this, a new autowrite run starts at version=1 and gets
    OUTRANKED in the sidebar by any previous completed draft (which
    might be at version=4 after a few iterations). The user clicks
    Stop, refreshes, and sees the OLD draft because the new one's
    version=1 < old's version=4. They report it as "we reverted".

    Returns max(existing version for this section) + 1, or 1 if none.
    """
    from sqlalchemy import text
    if not chapter_id:
        return 1
    row = session.execute(text("""
        SELECT COALESCE(MAX(version), 0)
        FROM drafts
        WHERE book_id::text = :bid
          AND chapter_id::text = :cid
          AND section_type = :st
    """), {"bid": book_id, "cid": chapter_id, "st": section_type}).fetchone()
    return int((row[0] if row else 0) or 0) + 1


# ── Phase 24 — autowrite progress log ───────────────────────────────────────
#
# Motivation: a user reported autowrite running 35 minutes with token
# count stuck at 0 and the GPU mostly idle. With only the SSE event
# stream as visibility, there's no way to tell whether the generator
# is stuck in retrieval, waiting on Ollama to load a model, or
# deadlocked between stages — the GUI just shows "writing..." with
# no progression.
#
# This logger writes a JSONL file under data/autowrite/ that captures
# every stage transition AND a periodic heartbeat (every 30s) even
# when no LLM tokens are flowing. Tail it with:
#
#     tail -f data/autowrite/latest.jsonl | jq
#
# The heartbeat fires from a side daemon thread, so it's independent
# of the main generator's yield cadence — even if the generator is
# blocked inside an LLM call for minutes, the heartbeat still records
# "stage=writing, stage_elapsed=120s, tokens=0" so you can spot the
# stall in real time.

# Heartbeat cadence in seconds. 30s is short enough to spot a stall
# within ~1 minute, long enough that the file doesn't bloat on a
# multi-hour run (~120 lines/hour).
_AUTOWRITE_LOG_HEARTBEAT_SECONDS = 30.0


def _slugify_for_filename(s: str) -> str:
    """Make a string safe for a filename. Lowercase, alphanum + dashes."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:40] or "section"


class _AutowriteLogger:
    """Thread-safe append-only JSONL logger for autowrite runs.

    Writes to ``{settings.data_dir}/autowrite/{run_id}.jsonl`` and
    maintains a ``latest.jsonl`` symlink in the same directory so the
    user can ``tail -f`` without knowing the exact run id.

    A daemon heartbeat thread writes a {"kind": "heartbeat", ...}
    entry every ``_AUTOWRITE_LOG_HEARTBEAT_SECONDS`` capturing the
    current stage, stage elapsed time, total tokens, and tokens-per-
    second — even when no LLM activity is happening (so a stall in
    retrieval or model load is visible).

    Use as a context manager:
        with _AutowriteLogger(book_id, chapter_id, section_type) as log:
            log.stage("retrieval")
            ...
            log.stage("writing")
            ...

    Or manually with try/finally:
        log = _AutowriteLogger(...)
        try:
            log.stage("setup")
            ...
        finally:
            log.close()
    """

    def __init__(
        self,
        book_id: str,
        chapter_id: str,
        section_type: str,
        *,
        heartbeat_seconds: float = _AUTOWRITE_LOG_HEARTBEAT_SECONDS,
        log_dir: Path | None = None,
    ) -> None:
        if log_dir is None:
            from sciknow.config import settings
            log_dir = settings.data_dir / "autowrite"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

        # Phase 33 — log rotation: keep the most recent 50 log files,
        # delete older ones. Phase 24 creates a file per run and they
        # accumulate forever; even at one autowrite per day this stays
        # under 2 MB total (a typical log is 10-30 KB). The sweep runs
        # at logger init so we never hold more than 50 logs on disk.
        self._rotate_old_logs(log_dir, keep=50)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _slugify_for_filename(section_type)
        self.run_id = f"{ts}_{slug}_{(chapter_id or 'noch')[:8]}"
        self.path = log_dir / f"{self.run_id}.jsonl"

        self.fh = open(self.path, "a", encoding="utf-8")

        # Thread-shared state. Lock protects state + writes.
        self._lock = threading.Lock()
        self._state = {
            "stage": "init",
            "stage_started_at": time.monotonic(),
            "stage_tokens": 0,
            "total_tokens": 0,
            "section": section_type,
            "iteration": None,
        }

        self._stop_event = threading.Event()
        self._heartbeat_seconds = heartbeat_seconds
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"autowrite-hb-{self.run_id}",
            daemon=True,
        )
        self._closed = False

        # Initial entry — written before the heartbeat starts so the
        # very first line of the log identifies the run.
        self._write({
            "kind": "start",
            "book_id": book_id,
            "chapter_id": chapter_id,
            "section_type": section_type,
            "run_id": self.run_id,
        })

        # Update the latest symlink. Best-effort; failures are logged
        # but not fatal because the per-run file is the source of truth.
        self._update_latest_symlink()

        # Start the heartbeat last so the start entry is always first.
        self._hb_thread.start()

    # ── Context manager protocol ─────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc:
            try:
                self.event(
                    "exception",
                    type=exc_type.__name__ if exc_type else "unknown",
                    message=str(exc)[:500],
                )
            except Exception:
                pass
        self.close()
        return False  # don't suppress exceptions

    # ── Public API ───────────────────────────────────────────────────────
    def stage(self, name: str, **extra) -> None:
        """Mark a stage transition. Writes a stage_end for the previous
        stage (with duration + tokens) and a stage_start for the new
        one. Resets the stage_tokens counter."""
        with self._lock:
            prev = self._state["stage"]
            prev_started = self._state["stage_started_at"]
            prev_tokens = self._state["stage_tokens"]
        if prev != "init":
            self._write({
                "kind": "stage_end",
                "stage": prev,
                "duration_s": round(time.monotonic() - prev_started, 1),
                "stage_tokens": prev_tokens,
            })
        with self._lock:
            self._state["stage"] = name
            self._state["stage_started_at"] = time.monotonic()
            self._state["stage_tokens"] = 0
            for k, v in extra.items():
                if k in ("section", "iteration"):
                    self._state[k] = v
        entry = {"kind": "stage_start", "stage": name}
        entry.update(extra)
        self._write(entry)

    def token(self) -> None:
        """Record one streamed token. Cheap (lock + two int adds).
        Called from _stream_with_save's per-token loop."""
        with self._lock:
            self._state["total_tokens"] += 1
            self._state["stage_tokens"] += 1

    def event(self, kind: str, **fields) -> None:
        """Write a one-off structured event (e.g. error, retry, verdict).
        Doesn't change the stage state machine."""
        entry = {"kind": kind}
        entry.update(fields)
        self._write(entry)

    def close(self) -> None:
        """Stop the heartbeat thread, close the file. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Final stage_end so the last stage's duration is recorded.
        with self._lock:
            prev = self._state["stage"]
            prev_started = self._state["stage_started_at"]
            prev_tokens = self._state["stage_tokens"]
            total = self._state["total_tokens"]
        if prev != "init":
            self._write({
                "kind": "stage_end",
                "stage": prev,
                "duration_s": round(time.monotonic() - prev_started, 1),
                "stage_tokens": prev_tokens,
            })
        self._write({"kind": "end", "total_tokens": total})

        self._stop_event.set()
        try:
            self._hb_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.fh.close()
        except Exception:
            pass

    @property
    def state(self) -> dict:
        """Snapshot of the current shared state (for tests/debugging)."""
        with self._lock:
            return dict(self._state)

    # ── Internals ────────────────────────────────────────────────────────
    def _write(self, entry: dict) -> None:
        """Append one JSONL line. Thread-safe via the lock so the
        heartbeat thread and the main generator don't interleave."""
        line = {"t": datetime.now().isoformat(timespec="seconds"), **entry}
        with self._lock:
            try:
                self.fh.write(json.dumps(line, default=str) + "\n")
                self.fh.flush()
            except Exception as exc:
                # File closed unexpectedly; log to module logger and
                # carry on — the heartbeat thread should not crash the
                # autowrite if its log file disappears.
                logger.warning("autowrite log write failed: %s", exc)

    @staticmethod
    def _rotate_old_logs(log_dir: Path, *, keep: int = 50) -> int:
        """Phase 33 — delete log files older than the most recent `keep`.

        Sorts by modification time (newest first) and unlinks the tail.
        Ignores non-files (e.g. the `latest.jsonl` symlink). Returns the
        number of files deleted.
        """
        try:
            files = sorted(
                (f for f in log_dir.iterdir()
                 if f.is_file() and f.suffix == ".jsonl" and not f.is_symlink()),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            stale = files[keep:]
            for f in stale:
                try:
                    f.unlink()
                except Exception:
                    pass
            return len(stale)
        except Exception:
            return 0

    def _update_latest_symlink(self) -> None:
        """Point data/autowrite/latest.jsonl at this run's file so the
        user can `tail -f data/autowrite/latest.jsonl` without
        knowing the run id. Best-effort; on systems without symlink
        support (e.g. Windows without dev mode) we silently skip."""
        latest = self.log_dir / "latest.jsonl"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(self.path.name)
        except Exception as exc:
            logger.debug("autowrite latest symlink update failed: %s", exc)

    def _heartbeat_loop(self) -> None:
        """Daemon thread that writes a heartbeat entry every
        ``heartbeat_seconds`` seconds until close() is called.

        Reads the shared state under the lock, so a snapshot of
        stage/tokens/elapsed is consistent within one entry."""
        while not self._stop_event.wait(self._heartbeat_seconds):
            with self._lock:
                state = dict(self._state)
            stage_elapsed = time.monotonic() - state["stage_started_at"]
            tps = state["stage_tokens"] / stage_elapsed if stage_elapsed > 0.1 else 0.0
            self._write({
                "kind": "heartbeat",
                "stage": state["stage"],
                "stage_elapsed_s": round(stage_elapsed, 1),
                "stage_tokens": state["stage_tokens"],
                "total_tokens": state["total_tokens"],
                "tokens_per_sec": round(tps, 2),
                "section": state["section"],
                "iteration": state["iteration"],
            })


def _update_draft_content(draft_id, content, *, summary=None, custom_metadata=None,
                          review_feedback=None, version=None):
    """Phase 15.1 — incremental update of an existing draft.

    Used by autowrite_section_stream and write_section_stream to persist
    intermediate state at each checkpoint, so a user clicking Stop never
    loses more than the in-flight iteration's tokens.

    Only fields you pass get updated; the rest are left untouched. The
    word_count is recomputed from the new content.
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    if not draft_id:
        return
    updates = ["content = :content", "word_count = :wc"]
    params: dict = {
        "did": draft_id, "content": content,
        "wc": len(content.split()),
    }
    if summary is not None:
        updates.append("summary = :summary")
        params["summary"] = summary
    if custom_metadata is not None:
        updates.append("custom_metadata = CAST(:metadata AS jsonb)")
        params["metadata"] = json.dumps(custom_metadata)
    if review_feedback is not None:
        updates.append("review_feedback = :rf")
        params["rf"] = review_feedback
    if version is not None:
        updates.append("version = :version")
        params["version"] = version

    with get_session() as session:
        session.execute(
            text(f"UPDATE drafts SET {', '.join(updates)} WHERE id::text = :did"),
            params,
        )
        session.commit()


def _retrieve(session, qdrant, query: str, candidate_k: int = 50,
              context_k: int = 12, topic_cluster: str | None = None,
              year_from: int | None = None, year_to: int | None = None,
              use_query_expansion: bool = False):
    """Run hybrid search + rerank + context build. Returns (results, sources).

    After retrieval completes, GPU models (bge-m3 + reranker) are released
    so the LLM has full VRAM available.
    """
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.rag.prompts import format_sources

    candidates = hybrid_search.search(
        query=query, qdrant_client=qdrant, session=session,
        candidate_k=candidate_k, year_from=year_from, year_to=year_to,
        topic_cluster=topic_cluster, use_query_expansion=use_query_expansion,
    )
    if not candidates:
        _release_gpu_models()
        return [], []
    candidates = reranker.rerank(query, candidates, top_k=context_k)
    _release_gpu_models()
    results = context_builder.build(candidates, session)
    sources = format_sources(results).splitlines()
    return results, sources


def _retrieve_visuals(
    query: str,
    *,
    cited_doc_ids: list[str] | tuple[str, ...] = (),
    section_type: str | None = None,
    candidate_k: int = 15,
    top_k: int = 5,
) -> list:
    """Phase 54.6.141 — writer-side visuals retrieval helper.

    Thin wrapper around ``rank_visuals`` that fits the ``_retrieve()``
    pattern (takes the query + cited docs, returns ranked results).
    Kept as a separate helper so the writer-integration decision (when
    to call this, how much weight to give it in the prompt, whether to
    validate [Fig. N] markers in verify pass) can be made per call site
    without modifying the existing ``_retrieve`` signature.

    Returns an empty list on any internal failure — the caller is
    expected to treat visuals as *optional* augmentation, not as a
    hard dependency of the writer stage.

    ``section_type`` should be one of the canonical chunker section
    types ("introduction", "methods", "results", "discussion", ...)
    so the Phase 54.6.139 section-type prior signal activates.
    """
    if not (query or "").strip():
        return []
    try:
        from sciknow.retrieval.visuals_ranker import rank_visuals
        return rank_visuals(
            query,
            cited_doc_ids=list(cited_doc_ids or []),
            section_type=section_type,
            candidate_k=candidate_k,
            top_k=top_k,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("_retrieve_visuals failed (writer continues without visuals): %s", exc)
        return []


def _generate_step_back_query(query: str, model: str | None = None) -> str | None:
    """Ask the LLM for an abstract reformulation of `query`.

    Phase 15.4 — uses the MAIN model (settings.llm_model), not the fast
    one. Original Phase 1 design used llm_fast_model on the assumption
    that "the abstraction step doesn't need the main model's reasoning
    horsepower and we want to keep latency low." That assumption was
    wrong on a single-GPU setup: switching from the main model to the
    fast model forces Ollama to evict the main model and load the
    fast one (~60s cold load for a 30B model on a 24GB GPU). Then
    after step-back returns, switching back to the main model evicts
    the fast model and reloads the main one (~60s again). Two model
    swaps cost ~2 minutes — far more than the milliseconds the fast
    model saves on a 1-line query.

    Using the main model means zero swaps: the main model is already
    loaded for the writer, the 1-line step-back query takes <1 second
    once warm, and the writer reuses the same hot model immediately
    after.

    The caller can still pass an explicit `model=` to override.
    """
    from sciknow.config import settings
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import complete as llm_complete
    sys_p, usr_p = rag_prompts.step_back(query)
    try:
        raw = llm_complete(
            sys_p, usr_p,
            model=model or settings.llm_model,  # main model, not fast
            # Phase 54.6.30 — standardise on 16384 across the autowrite
            # call chain so this utility (step-back query) doesn't
            # trigger an Ollama model reload when the main write pass
            # arrives expecting 16384. Extra KV cache cost is trivial
            # compared to model weights.
            temperature=0.1, num_ctx=16384, keep_alive=-1,
        )
        # Strip thinking blocks and any leading/trailing punctuation/quotes.
        raw = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()
        # Take only the first non-empty line and strip wrapping quotes.
        for line in raw.splitlines():
            line = line.strip().strip('"\'').strip()
            if line:
                return line
        return None
    except Exception as exc:
        logger.warning("Step-back query generation failed: %s", exc)
        return None


def _retrieve_with_step_back(
    session, qdrant, query: str,
    candidate_k: int = 50, context_k: int = 12,
    topic_cluster: str | None = None,
    year_from: int | None = None, year_to: int | None = None,
    use_query_expansion: bool = False,
    use_step_back: bool = True,
    model: str | None = None,
):
    """Retrieve with optional step-back query augmentation.

    Strategy: run the concrete query through hybrid search to get
    `candidate_k` candidates. If step-back is enabled, generate an
    abstract reformulation and retrieve `candidate_k // 2` more
    candidates for it. Union the candidate pools (deduping by chunk_id),
    then rerank the union against the ORIGINAL query and take top
    `context_k`. Reranking against the original query keeps the final
    relevance scoped to what the writer actually needs to draft, while
    the step-back pool brings in mechanism / background chunks the
    concrete query would have missed.

    Source: Zheng et al. "Take a Step Back: Evoking Reasoning via
    Abstraction in LLMs", ICLR 2024 (arXiv:2310.06117).
    """
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.rag.prompts import format_sources

    candidates = hybrid_search.search(
        query=query, qdrant_client=qdrant, session=session,
        candidate_k=candidate_k, year_from=year_from, year_to=year_to,
        topic_cluster=topic_cluster, use_query_expansion=use_query_expansion,
    )

    if use_step_back:
        sb_query = _generate_step_back_query(query, model=model)
        if sb_query and sb_query.lower() != query.lower():
            logger.info("Step-back query: %r → %r", query, sb_query)
            sb_candidates = hybrid_search.search(
                query=sb_query, qdrant_client=qdrant, session=session,
                candidate_k=max(candidate_k // 2, 10),
                year_from=year_from, year_to=year_to,
                topic_cluster=topic_cluster,
                use_query_expansion=False,  # already abstract; don't expand again
            )
            # Union by chunk_id, preserving the concrete-query candidates first.
            seen = {c.chunk_id for c in candidates}
            for c in sb_candidates:
                if c.chunk_id not in seen:
                    candidates.append(c)
                    seen.add(c.chunk_id)

    if not candidates:
        _release_gpu_models()
        return [], []

    # Rerank against the ORIGINAL (concrete) query so the final ordering
    # reflects what the writer actually needs.
    candidates = reranker.rerank(query, candidates, top_k=context_k)
    _release_gpu_models()
    results = context_builder.build(candidates, session)
    sources = format_sources(results).splitlines()
    return results, sources


def _clean_json(raw: str) -> str:
    """Strip markdown fences and extract the JSON object."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', '', cleaned)
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first >= 0 and last > first:
        cleaned = cleaned[first:last + 1]
    return cleaned


# ── write_section_stream ─────────────────────────────────────────────────────

def write_section_stream(
    book_id: str,
    chapter_id: str,
    section_type: str = "introduction",
    *,
    context_k: int = 12,
    candidate_k: int = 50,
    year_from: int | None = None,
    year_to: int | None = None,
    model: str | None = None,
    expand: bool = False,
    show_plan: bool = False,
    verify: bool = False,
    save: bool = True,
    use_step_back: bool = True,
    target_words: int | None = None,
) -> Iterator[Event]:
    """
    Draft a chapter section.  Yields events as the operation progresses.
    """
    from sciknow.rag import prompts
    from sciknow.rag.llm import stream as llm_stream, complete as llm_complete
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    yield {"type": "progress", "stage": "setup", "detail": "Loading book data..."}

    with get_session() as session:
        from sqlalchemy import text
        book = session.execute(text("""
            SELECT id::text, title, description, status, created_at, plan
            FROM books WHERE id::text = :bid
        """), {"bid": book_id}).fetchone()
        if not book:
            yield {"type": "error", "message": f"Book not found: {book_id}"}
            return

        ch = session.execute(text("""
            SELECT id::text, number, title, description, topic_query, topic_cluster
            FROM book_chapters WHERE id::text = :cid
        """), {"cid": chapter_id}).fetchone()
        if not ch:
            yield {"type": "error", "message": f"Chapter not found: {chapter_id}"}
            return

        b_plan = book[5]
        ch_id, ch_num, ch_title, ch_desc, topic_query, topic_cluster = ch
        search_query = f"{section_type} {topic_query or ch_title}"
        topic = topic_query or ch_title

        prior_summaries = _get_prior_summaries(session, book_id, ch_num)

        # Phase 17 / 29 — resolve the effective per-section length target.
        # 3-level priority: caller arg > per-section meta override
        # > chapter target / num_sections.
        if target_words is not None:
            effective_target_words = target_words
        else:
            section_override = _get_section_target_words(session, ch_id, section_type)
            if section_override is not None:
                effective_target_words = section_override
            else:
                chapter_target = _get_book_length_target(session, book_id)
                num_sections = _get_chapter_num_sections(session, ch_id)
                effective_target_words = _section_target_words(chapter_target, num_sections)

        # Phase 18 — pull the per-section plan if the user has set one
        # via the chapter modal's Sections tab. Empty string is fine —
        # write_section_v2 / tree_plan ignore an empty plan.
        section_plan = _get_section_plan(session, ch_id, section_type)

        # Phase 37 — per-section model override.  Precedence:
        # caller arg > per-section override > global default
        # (resolved inside rag.llm.stream).
        if model is None:
            section_model_override = _get_section_model(session, ch_id, section_type)
            if section_model_override:
                model = section_model_override

    yield {"type": "progress", "stage": "retrieval", "detail": "Searching literature..."}

    qdrant = get_client()
    with get_session() as session:
        results, sources = _retrieve_with_step_back(
            session, qdrant, search_query,
            candidate_k=candidate_k, context_k=context_k,
            topic_cluster=topic_cluster, year_from=year_from,
            year_to=year_to, use_query_expansion=expand,
            use_step_back=use_step_back, model=model,
        )

    if not results:
        yield {"type": "error", "message": "No relevant passages found."}
        return

    # Optional hierarchical tree plan (TreeWriter pattern)
    if show_plan:
        yield {"type": "progress", "stage": "planning",
               "detail": "Creating hierarchical paragraph plan..."}
        sys_p, usr_p = prompts.tree_plan(
            section_type, topic, results,
            book_plan=b_plan, prior_summaries=prior_summaries,
            target_words=effective_target_words,
            section_plan=section_plan,
        )
        plan_raw = ""
        for tok in llm_stream(sys_p, usr_p, model=model, keep_alive=-1):
            plan_raw += tok
        # Strip thinking blocks and try to parse as JSON
        plan_clean = re.sub(r'<think>.*?</think>\s*', '', plan_raw, flags=re.DOTALL).strip()
        try:
            plan_data = json.loads(_clean_json(plan_clean), strict=False)
            paragraph_plan = plan_data.get("paragraphs") or []
            yield {"type": "tree_plan", "data": plan_data, "raw": plan_clean}
        except Exception:
            paragraph_plan = None
            yield {"type": "plan", "content": plan_clean}
    else:
        paragraph_plan = None

    # Draft — Phase 14.6: emit model info before the first token so the
    # user can confirm the writer is using the flagship model.
    from sciknow.config import settings as _settings
    resolved_model = model or _settings.llm_model
    yield {"type": "model_info", "writer_model": resolved_model,
           "fast_model": _settings.llm_fast_model,
           "writer_role": "writing this section",
           "fast_role": "step-back retrieval (utility)"}
    yield {"type": "progress", "stage": "writing",
           "detail": f"Drafting {section_type} for Ch.{ch_num}: {ch_title} · model: {resolved_model}"}

    # Phase 21.e — query visuals for this section (same pattern as autowrite)
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
                ORDER BY v.kind, v.created_at LIMIT 8
            """), {"pat": f"%{(topic or '')[:50]}%"}).fetchall()
            section_visuals = [
                {"kind": r[0], "content": r[1], "caption": r[2],
                 "figure_num": r[3], "surrounding_text": r[4]}
                for r in vis_rows
            ]
    except Exception:
        pass

    system, user = prompts.write_section_v2(
        section_type, topic, results,
        book_plan=b_plan, prior_summaries=prior_summaries,
        paragraph_plan=paragraph_plan,
        target_words=effective_target_words,
        section_plan=section_plan,
        visuals=section_visuals,
    )

    yield {"type": "length_target", "target_words": effective_target_words}

    # Phase 19 — INSERT placeholder draft BEFORE writing so periodic
    # saves have a row to UPDATE. Without this, clicking Stop mid-write
    # loses every token. New draft gets the next version above all
    # existing drafts for this section so it always wins the GUI's
    # latest-version sort (so refreshes during writing show the new
    # in-flight content, not an older draft).
    draft_id = None
    if save:
        draft_title = f"Ch.{ch_num} {ch_title} — {section_type.capitalize()}"
        with get_session() as session:
            draft_version = _next_draft_version(
                session, book_id, ch_id, section_type
            )
            draft_id = _save_draft(
                session, title=draft_title, book_id=book_id, chapter_id=ch_id,
                section_type=section_type, topic=topic, content="",
                sources=sources, model=model, summary=None,
                version=draft_version,
                custom_metadata={"checkpoint": "writing_in_progress"},
            )
        yield {"type": "checkpoint", "draft_id": draft_id, "stage": "placeholder",
               "word_count": 0}

    # Phase 19 — incremental save during the writing stream. The
    # callback closure captures draft_id; if save is False there's no
    # callback and no checkpointing (single-shot in-memory write,
    # used by callers that don't want a DB row).
    if save and draft_id:
        def _save_writing(text: str) -> None:
            _update_draft_content(
                draft_id, text,
                custom_metadata={"checkpoint": "writing_in_progress"},
            )
        content = yield from _stream_with_save(
            system, user, "writing",
            model=model, save_callback=_save_writing,
        )
        # Final flush + checkpoint marker after the stream completes
        # cleanly (Stop is handled by _stream_with_save's finally).
        _update_draft_content(
            draft_id, content,
            custom_metadata={"checkpoint": "draft"},
        )
        yield {"type": "checkpoint", "draft_id": draft_id, "stage": "draft",
               "word_count": len(content.split())}
    else:
        content_tokens: list[str] = []
        for tok in llm_stream(system, user, model=model, keep_alive=-1):
            content_tokens.append(tok)
            yield {"type": "token", "text": tok}
        content = "".join(content_tokens)

    # Optional claim verification
    verify_feedback = None
    if verify:
        yield {"type": "progress", "stage": "verifying", "detail": "Verifying claims..."}
        sys_v, usr_v = prompts.verify_claims(content, results)
        try:
            raw = llm_complete(sys_v, usr_v, model=model, temperature=0.0, num_ctx=16384, keep_alive=-1)
            vdata = json.loads(_clean_json(raw), strict=False)
            verify_feedback = raw
            yield {"type": "verification", "data": vdata}
        except Exception as exc:
            yield {"type": "progress", "stage": "verifying",
                   "detail": f"Verification failed: {exc}"}

    # Generate summary + finalize the draft (phases 15.1 — UPDATE existing
    # row instead of inserting a new one; the draft was already saved above).
    if save and draft_id:
        yield {"type": "progress", "stage": "saving", "detail": "Generating summary..."}
        summary = _auto_summarize(content, section_type, ch_title, model=model)
        _update_draft_content(
            draft_id, content,
            summary=summary,
            review_feedback=verify_feedback,
        )

    yield {
        "type": "completed",
        "draft_id": draft_id,
        "word_count": len(content.split()),
        "sources": sources,
        "content": content,
    }


# ── review_draft_stream ──────────────────────────────────────────────────────

def review_draft_stream(
    draft_id: str,
    *,
    model: str | None = None,
    save: bool = True,
) -> Iterator[Event]:
    """Run a critic pass over a saved draft."""
    from sqlalchemy import text
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    with get_session() as session:
        row = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.topic, d.content,
                   d.book_id::text, d.chapter_id::text
            FROM drafts d WHERE d.id::text LIKE :q
            LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()

    if not row:
        yield {"type": "error", "message": f"Draft not found: {draft_id}"}
        return

    d_id, d_title, d_section, d_topic, d_content, d_book_id, d_chapter_id = row

    # Phase 37 — per-section model override (review path).
    if model is None and d_chapter_id and d_section:
        with get_session() as session:
            section_model_override = _get_section_model(session, d_chapter_id, d_section)
        if section_model_override:
            model = section_model_override

    # Phase 54.6.55 — global review-role override (BOOK_REVIEW_MODEL env),
    # applied only if no per-call and no per-section override was set.
    # The 2026-04-17-full bench showed gemma3:27b-it-qat beats the
    # unified qwen LLM_MODEL on book_review (judge 100% vs 71.4%,
    # dimensions 5/5 vs 3/5), but losing everywhere else — so we keep
    # qwen as the global default and let this setting re-route reviews.
    if model is None:
        from sciknow.config import settings as _settings
        if _settings.book_review_model:
            model = _settings.book_review_model

    yield {"type": "progress", "stage": "retrieval",
           "detail": f"Retrieving context for review of '{d_title}'..."}

    qdrant = get_client()
    search_query = f"{d_section or ''} {d_topic or d_title}"
    with get_session() as session:
        results, _ = _retrieve(session, qdrant, search_query, context_k=12)

    yield {"type": "progress", "stage": "reviewing",
           "detail": f"Reviewing: {d_title}..."}

    sys_r, usr_r = rag_prompts.review(d_section, d_topic or d_title, d_content, results)
    output_tokens: list[str] = []
    for tok in llm_stream(sys_r, usr_r, model=model, keep_alive=-1):
        output_tokens.append(tok)
        yield {"type": "token", "text": tok}

    feedback = "".join(output_tokens).strip()

    if save:
        with get_session() as session:
            session.execute(text(
                "UPDATE drafts SET review_feedback = :fb WHERE id::text = :did"
            ), {"fb": feedback, "did": d_id})
            session.commit()

    yield {"type": "completed", "draft_id": d_id, "feedback": feedback}


# ── revise_draft_stream ──────────────────────────────────────────────────────

def revise_draft_stream(
    draft_id: str,
    *,
    instruction: str = "",
    context_k: int = 8,
    model: str | None = None,
) -> Iterator[Event]:
    """Revise a draft, creating a new version."""
    from sqlalchemy import text
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    with get_session() as session:
        row = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.topic, d.content,
                   d.book_id::text, d.chapter_id::text, d.version,
                   d.review_feedback, d.sources
            FROM drafts d WHERE d.id::text LIKE :q
            LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()

    if not row:
        yield {"type": "error", "message": f"Draft not found: {draft_id}"}
        return

    d_id, d_title, d_section, d_topic, d_content, d_book_id, d_chapter_id, \
        d_version, d_review_feedback, d_sources = row

    # Phase 37 — per-section model override (revise path).
    if model is None and d_chapter_id and d_section:
        with get_session() as session:
            section_model_override = _get_section_model(session, d_chapter_id, d_section)
        if section_model_override:
            model = section_model_override

    rev_instruction = instruction.strip()
    if not rev_instruction:
        if d_review_feedback:
            rev_instruction = f"Apply the following review feedback:\n\n{d_review_feedback}"
        else:
            yield {"type": "error", "message": "No instruction provided and no saved review feedback."}
            return

    # Retrieve additional passages
    results = []
    if context_k > 0:
        yield {"type": "progress", "stage": "retrieval", "detail": "Retrieving context..."}
        qdrant = get_client()
        search_query = f"{d_section or ''} {d_topic or d_title}"
        with get_session() as session:
            results, _ = _retrieve(session, qdrant, search_query, context_k=context_k)

    yield {"type": "progress", "stage": "revising",
           "detail": f"Revising {d_title} (v{d_version} -> v{d_version + 1})..."}

    sys_r, usr_r = rag_prompts.revise(d_content, rev_instruction, results or None)
    output_tokens: list[str] = []
    for tok in llm_stream(sys_r, usr_r, model=model, keep_alive=-1):
        output_tokens.append(tok)
        yield {"type": "token", "text": tok}

    revised_content = "".join(output_tokens).strip()

    yield {"type": "progress", "stage": "saving", "detail": "Generating summary and saving..."}
    summary = _auto_summarize(revised_content, d_section or "text", d_title, model=model)

    existing_sources = json.loads(d_sources) if isinstance(d_sources, str) else (d_sources or [])

    with get_session() as session:
        new_id = _save_draft(
            session, title=d_title, book_id=d_book_id, chapter_id=d_chapter_id,
            section_type=d_section, topic=d_topic, content=revised_content,
            sources=existing_sources, model=model, summary=summary,
            parent_draft_id=d_id, version=d_version + 1,
        )

    yield {
        "type": "completed",
        "draft_id": new_id,
        "parent_draft_id": d_id,
        "version": d_version + 1,
        "word_count": len(revised_content.split()),
        "content": revised_content,
    }


# ── run_gaps_stream ──────────────────────────────────────────────────────────

def run_gaps_stream(
    book_id: str,
    *,
    model: str | None = None,
    save: bool = True,
    method_preamble: str = "",
) -> Iterator[Event]:
    """Run gap analysis on a book.

    Phase 54.6.14 — ``method_preamble`` (optional) is prepended to the
    user prompt so callers can steer the LLM's framing via a named
    brainstorming method (Reverse Brainstorming, Five Whys, Scope
    Boundaries, etc.). Pass the pre-rendered string; see
    ``sciknow.core.methods.method_preamble``.
    """
    from sqlalchemy import text
    from sciknow.rag import prompts
    from sciknow.rag.llm import stream as llm_stream, complete as llm_complete
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = session.execute(text("""
            SELECT id::text, title FROM books WHERE id::text = :bid
        """), {"bid": book_id}).fetchone()
        if not book:
            yield {"type": "error", "message": f"Book not found: {book_id}"}
            return

        chapters = session.execute(text("""
            SELECT number, title, description FROM book_chapters
            WHERE book_id = :bid ORDER BY number
        """), {"bid": book_id}).fetchall()

        papers = session.execute(text(
            "SELECT title, year FROM paper_metadata ORDER BY year DESC NULLS LAST LIMIT 150"
        )).fetchall()

        drafts_rows = session.execute(text("""
            SELECT d.title, d.section_type, bc.number as chapter_number
            FROM drafts d LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.book_id = :bid ORDER BY bc.number, d.created_at
        """), {"bid": book_id}).fetchall()

    if not chapters:
        yield {"type": "error", "message": "No chapters defined. Run `book outline` first."}
        return

    ch_list = [{"number": c[0], "title": c[1], "description": c[2]} for c in chapters]
    p_list = [{"title": p[0], "year": p[1]} for p in papers if p[0]]
    d_list = [{"title": d[0], "section_type": d[1], "chapter_number": d[2]} for d in drafts_rows]

    # Phase 47.2 — rejected-idea gate. Query lessons with kind='rejected_idea'
    # scoped to this book OR globally-promoted (Phase 47.4), across any
    # section of this book. The gap generator is book-level, so we
    # concatenate hits from every section_slug on this book's chapters.
    rejected_ideas: list[str] = []
    try:
        with get_session() as session:
            rej_rows = session.execute(text("""
                SELECT DISTINCT lesson_text
                FROM autowrite_lessons
                WHERE kind = 'rejected_idea'
                  AND lesson_text IS NOT NULL
                  AND (
                    (scope = 'book' AND book_id = CAST(:bid AS uuid))
                    OR scope = 'global'
                  )
                ORDER BY lesson_text
                LIMIT 12
            """), {"bid": book_id}).fetchall()
        rejected_ideas = [r[0] for r in rej_rows if r[0]]
    except Exception as exc:
        logger.warning("gap gate: rejected-idea lookup failed: %s", exc)
    if rejected_ideas:
        yield {
            "type": "progress", "stage": "rejected_ideas_loaded",
            "detail": (
                f"Phase 47.2: blocking {len(rejected_ideas)} "
                "previously-rejected idea(s) from re-surfacing."
            ),
            "n_rejected_ideas": len(rejected_ideas),
        }

    # Pass 1: human-readable narrative (streamed)
    yield {"type": "progress", "stage": "analyzing", "detail": "Running gap analysis..."}
    system, user = prompts.gaps(
        book_title=book[1], chapters=ch_list, papers=p_list, drafts=d_list,
        rejected_ideas=rejected_ideas or None,
    )
    if method_preamble:
        user = method_preamble + user
    for tok in llm_stream(system, user, model=model, num_ctx=16384, keep_alive=-1):
        yield {"type": "token", "text": tok}

    # Pass 2: structured JSON extraction
    if save:
        yield {"type": "progress", "stage": "extracting", "detail": "Extracting structured gaps..."}
        sys_j, usr_j = prompts.gaps_json(
            book_title=book[1], chapters=ch_list, papers=p_list, drafts=d_list,
            rejected_ideas=rejected_ideas or None,
        )
        try:
            raw = llm_complete(sys_j, usr_j, model=model, temperature=0.0, num_ctx=16384, keep_alive=-1)
            gap_data = json.loads(_clean_json(raw), strict=False)
            gap_list = gap_data.get("gaps", [])
        except Exception as exc:
            yield {"type": "progress", "stage": "extracting",
                   "detail": f"Structured extraction failed: {exc}"}
            gap_list = []

        if gap_list:
            with get_session() as session:
                ch_rows = session.execute(text("""
                    SELECT number, id::text FROM book_chapters WHERE book_id = :bid
                """), {"bid": book_id}).fetchall()
                ch_id_map = {r[0]: r[1] for r in ch_rows}

                session.execute(text("DELETE FROM book_gaps WHERE book_id = :bid"),
                                {"bid": book_id})
                for g in gap_list:
                    ch_num = g.get("chapter_number")
                    session.execute(text("""
                        INSERT INTO book_gaps (book_id, gap_type, description, chapter_id, status)
                        VALUES (:bid, :gtype, :desc, :ch_id, 'open')
                    """), {
                        "bid": book_id,
                        "gtype": g.get("type", "topic"),
                        "desc": g.get("description", ""),
                        "ch_id": ch_id_map.get(ch_num) if ch_num else None,
                    })
                session.commit()

            yield {"type": "completed", "gaps_saved": len(gap_list), "gaps": gap_list}
        else:
            yield {"type": "completed", "gaps_saved": 0, "gaps": []}
    else:
        yield {"type": "completed", "gaps_saved": 0}


# ── run_argue_stream ─────────────────────────────────────────────────────────

def run_argue_stream(
    claim: str,
    *,
    book_id: str | None = None,
    context_k: int = 15,
    candidate_k: int = 60,
    year_from: int | None = None,
    year_to: int | None = None,
    model: str | None = None,
    save: bool = False,
) -> Iterator[Event]:
    """Map evidence for/against a claim."""
    from sciknow.rag import prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    yield {"type": "progress", "stage": "retrieval", "detail": "Searching literature..."}

    qdrant = get_client()
    with get_session() as session:
        results, sources = _retrieve(
            session, qdrant, claim,
            candidate_k=candidate_k, context_k=context_k,
            year_from=year_from, year_to=year_to,
        )

    if not results:
        yield {"type": "error", "message": "No relevant passages found."}
        return

    yield {"type": "progress", "stage": "arguing",
           "detail": f"Mapping argument: {claim[:60]}..."}

    system, user = prompts.argue(claim, results)
    output_tokens: list[str] = []
    for tok in llm_stream(system, user, model=model, num_ctx=16384, keep_alive=-1):
        output_tokens.append(tok)
        yield {"type": "token", "text": tok}

    content = "".join(output_tokens)

    if save and book_id:
        from sqlalchemy import text
        draft_title = f"Argument Map: {claim[:60]}"
        with get_session() as session:
            session.execute(text("""
                INSERT INTO drafts (title, book_id, section_type, topic, content,
                                    word_count, sources, model_used)
                VALUES (:title, :book_id, 'argument_map', :topic, :content,
                        :wc, CAST(:sources AS jsonb), :model)
            """), {
                "title": draft_title, "book_id": book_id, "topic": claim,
                "content": content, "wc": len(content.split()),
                "sources": json.dumps(sources), "model": model,
            })
            session.commit()

    yield {"type": "completed", "content": content, "sources": sources}


# ── autowrite_section_stream ─────────────────────────────────────────────────

DEFAULT_SECTIONS = ["overview", "key_evidence", "current_understanding", "open_questions", "summary"]


def _score_draft_inner(draft_content, section_type, topic, results, model=None):
    """Score a draft against provided results. Returns scores_dict.

    Fix 3: the caller passes the writer's own results so the scorer
    evaluates against the same evidence the writer used.

    Phase 54.6.59 — apply the AUTOWRITE_SCORER_MODEL fallback here too
    so ad-hoc callers (length-controlled eval, refinement_gate tests)
    get the same scorer routing as the live autowrite loop when model
    is not explicitly passed.
    """
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import complete as llm_complete
    from sciknow.config import settings as _s

    if model is None and _s.autowrite_scorer_model:
        model = _s.autowrite_scorer_model

    sys_s, usr_s = rag_prompts.score_draft(section_type, topic, draft_content, results)
    raw = llm_complete(sys_s, usr_s, model=model, temperature=0.0, num_ctx=16384, keep_alive=-1)

    return json.loads(_clean_json(raw), strict=False)


def _verify_draft_inner(draft_content, results, model=None):
    """Run claim verification and return parsed verification data.

    Fix 1: integrated into the autowrite scoring loop so every iteration
    checks citation groundedness, not just when the user passes --verify.
    """
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import complete as llm_complete

    sys_v, usr_v = rag_prompts.verify_claims(draft_content, results)
    raw = llm_complete(sys_v, usr_v, model=model, temperature=0.0, num_ctx=16384, keep_alive=-1)

    return json.loads(_clean_json(raw), strict=False)


def _cove_verify(draft_content, results, model=None, max_questions: int = 6):
    """Chain-of-Verification: decoupled fact-checking.

    1. Ask the LLM to generate falsifiable verification questions about the
       draft (sees ONLY the draft, not the sources).
    2. For each question, ask the LLM to answer it from the source passages
       alone (sees ONLY the sources, NOT the draft) — no anchoring.
    3. Compare independent answers to the draft's claims and return a
       structured report of mismatches.

    The whole point is the *decoupling* — the answerer for each question
    cannot be biased by the draft's framing because it never sees the draft.
    This catches the failure mode where the standard one-shot verifier
    rubberstamps a claim because it's reading the draft and the evidence
    in the same context window.

    Returns a dict:
        {
          "questions_asked": int,
          "mismatches": [
            {"question": ..., "draft_claim": ..., "draft_citation": ...,
             "independent_answer": ..., "verdict": ..., "severity": ...},
            ...
          ],
          "cove_score": float (1.0 - mismatches / questions_asked),
        }

    Source: Dhuliawala et al., "Chain-of-Verification Reduces Hallucination
    in Large Language Models", Findings of ACL 2024 (arXiv:2309.11495).
    """
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import complete as llm_complete

    # Step 1: Generate verification questions (sees only the draft).
    try:
        sys_q, usr_q = rag_prompts.cove_questions(draft_content)
        raw_q = llm_complete(sys_q, usr_q, model=model, temperature=0.1, num_ctx=16384, keep_alive=-1)
        q_clean = re.sub(r'<think>.*?</think>\s*', '', raw_q, flags=re.DOTALL).strip()
        q_data = json.loads(_clean_json(q_clean), strict=False)
        questions = (q_data.get("questions") or [])[:max_questions]
    except Exception as exc:
        logger.warning("CoVe question generation failed: %s", exc)
        return {"questions_asked": 0, "mismatches": [], "cove_score": 1.0}

    if not questions:
        return {"questions_asked": 0, "mismatches": [], "cove_score": 1.0}

    # Step 2: For each question, answer it independently from the sources.
    # The answerer never sees the draft — that's the whole point.
    mismatches: list[dict] = []
    for q in questions:
        question_text = (q.get("question") or "").strip()
        draft_claim = (q.get("draft_claim") or "").strip()
        draft_citation = (q.get("citation") or "").strip()
        if not question_text:
            continue

        try:
            sys_a, usr_a = rag_prompts.cove_answer(question_text, results)
            raw_a = llm_complete(
                sys_a, usr_a, model=model, temperature=0.0, num_ctx=16384,
                keep_alive=-1,
            )
            a_clean = re.sub(r'<think>.*?</think>\s*', '', raw_a, flags=re.DOTALL).strip()
            a_data = json.loads(_clean_json(a_clean), strict=False)
        except Exception as exc:
            logger.warning("CoVe answer failed for question %r: %s", question_text, exc)
            continue

        verdict = (a_data.get("verdict") or "").upper()
        independent_answer = (a_data.get("answer") or "").strip()
        notes = (a_data.get("notes") or "").strip()

        # A mismatch is any verdict that isn't a clean CONFIRMED.
        # NOT_IN_SOURCES means the draft made a claim the sources don't support.
        # DIFFERENT_SCOPE means the draft generalised past the source's stated scope.
        # PARTIAL means the source supports only part of what the draft asserts.
        if verdict and verdict != "CONFIRMED":
            severity = {
                "NOT_IN_SOURCES": "high",
                "DIFFERENT_SCOPE": "medium",
                "PARTIAL": "low",
            }.get(verdict, "medium")
            mismatches.append({
                "question": question_text,
                "draft_claim": draft_claim,
                "draft_citation": draft_citation,
                "independent_answer": independent_answer,
                "verdict": verdict,
                "severity": severity,
                "notes": notes,
            })

    cove_score = 1.0 - (len(mismatches) / max(len(questions), 1))
    return {
        "questions_asked": len(questions),
        "mismatches": mismatches,
        "cove_score": round(cove_score, 3),
    }


def _cove_verify_streaming(draft_content, results, model=None, max_questions: int = 6):
    """Phase 15.3 — generator variant of _cove_verify that yields token
    events for both the question generation pass and each independent
    answer pass. Returns the final mismatch report dict via
    StopIteration.value (captured by `yield from` in autowrite).

    Without this, CoVe runs 1 + N silent llm_complete calls and the GUI's
    token counter sits at 0 for the entire CoVe phase (often the longest
    silent stretch in autowrite).
    """
    from sciknow.rag import prompts as rag_prompts

    # Step 1 — generate verification questions, streamed
    try:
        sys_q, usr_q = rag_prompts.cove_questions(draft_content)
        raw_q = yield from _stream_phase(
            sys_q, usr_q, "cove_questions",
            model=model, temperature=0.1, num_ctx=16384,
        )
        q_clean = re.sub(r'<think>.*?</think>\s*', '', raw_q, flags=re.DOTALL).strip()
        q_data = json.loads(_clean_json(q_clean), strict=False)
        questions = (q_data.get("questions") or [])[:max_questions]
    except Exception as exc:
        logger.warning("CoVe question generation failed: %s", exc)
        return {"questions_asked": 0, "mismatches": [], "cove_score": 1.0}

    if not questions:
        return {"questions_asked": 0, "mismatches": [], "cove_score": 1.0}

    # Step 2 — independent answers, also streamed.
    mismatches: list[dict] = []
    for q_idx, q in enumerate(questions, 1):
        question_text = (q.get("question") or "").strip()
        draft_claim = (q.get("draft_claim") or "").strip()
        draft_citation = (q.get("citation") or "").strip()
        if not question_text:
            continue

        try:
            sys_a, usr_a = rag_prompts.cove_answer(question_text, results)
            raw_a = yield from _stream_phase(
                sys_a, usr_a, f"cove_answer_{q_idx}",
                model=model, temperature=0.0, num_ctx=16384,
            )
            a_clean = re.sub(r'<think>.*?</think>\s*', '', raw_a, flags=re.DOTALL).strip()
            a_data = json.loads(_clean_json(a_clean), strict=False)
        except Exception as exc:
            logger.warning("CoVe answer failed for question %r: %s", question_text, exc)
            continue

        verdict = (a_data.get("verdict") or "").upper()
        independent_answer = (a_data.get("answer") or "").strip()
        notes = (a_data.get("notes") or "").strip()

        if verdict and verdict != "CONFIRMED":
            severity = {
                "NOT_IN_SOURCES": "high",
                "DIFFERENT_SCOPE": "medium",
                "PARTIAL": "low",
            }.get(verdict, "medium")
            mismatches.append({
                "question": question_text,
                "draft_claim": draft_claim,
                "draft_citation": draft_citation,
                "independent_answer": independent_answer,
                "verdict": verdict,
                "severity": severity,
                "notes": notes,
            })

    cove_score = 1.0 - (len(mismatches) / max(len(questions), 1))
    return {
        "questions_asked": len(questions),
        "mismatches": mismatches,
        "cove_score": round(cove_score, 3),
    }


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
        )
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
    from sciknow.config import settings as _s
    _llm_warm_up(model=model or _s.llm_model, num_ctx=16384, num_batch=1024)

    # Phase 54.6.59 — scorer-role resolution. Explicit `--model` beats
    # everything (the user asked for one model end-to-end). Otherwise,
    # route scoring + rescoring through AUTOWRITE_SCORER_MODEL if set,
    # else fall through to the writer's llm_model (scorer_model=None
    # → llm.stream/complete default). NOTE: verify + CoVe stay on the
    # writer model — the gemopus4 scorer win is specific to the score
    # task; we did not bench it for verify/CoVe and other scorer tasks
    # on gemopus4 hang or produce 0 words.
    if model is not None:
        scorer_model = model
    elif _s.autowrite_scorer_model:
        scorer_model = _s.autowrite_scorer_model
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
            SELECT id::text, number, title, description, topic_query, topic_cluster
            FROM book_chapters WHERE id::text = :cid
        """), {"cid": chapter_id}).fetchone()

    if not book or not ch:
        log.event("error", message="book or chapter not found")
        yield {"type": "error", "message": "Book or chapter not found."}
        return

    b_title, b_plan = book[1], book[2]
    ch_id, ch_num, ch_title, ch_desc, topic_query, topic_cluster = ch
    topic = topic_query or ch_title

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
        if not ok:
            msg = f"cannot resume from draft {row[0][:8]}: {reason}"
            log.event("resume_refused", reason=reason, draft_id=row[0])
            yield {"type": "error", "message": msg}
            return

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
                chapter_target = _get_book_length_target(session, book_id)
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
    from sciknow.config import settings
    resolved_model = model or settings.llm_model

    yield {"type": "model_info", "writer_model": resolved_model,
           "fast_model": settings.llm_fast_model,
           "writer_role": "writing/scoring/verification/CoVe (flagship)",
           "fast_role": "step-back retrieval (utility)"}
    yield {"type": "length_target", "target_words": effective_target_words}
    yield {"type": "progress", "stage": "setup",
           "detail": (
               f"Autowrite Ch.{ch_num}: {ch_title} — {section_type.capitalize()} · "
               f"model: {resolved_model} · target: ~{effective_target_words} words"
           )}

    # Step 1: Initial draft
    log.stage("retrieval")
    with get_session() as session:
        prior_summaries = _get_prior_summaries(session, book_id, ch_num)
        results, sources = _retrieve_with_step_back(
            session, qdrant, f"{section_type} {topic}",
            topic_cluster=topic_cluster, model=model,
            use_step_back=use_step_back,
        )

    if not results:
        log.event("error", message="no relevant passages found")
        yield {"type": "error", "message": "No relevant passages found."}
        return

    log.event("retrieval_done", n_results=len(results), n_sources=len(sources))

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
        try:
            sys_p, usr_p = rag_prompts.tree_plan(
                section_type, topic, results,
                book_plan=b_plan, prior_summaries=prior_summaries,
                target_words=effective_target_words,
                section_plan=section_plan,
            )
            # Phase 15.3 — stream so the token counter sees activity
            # Phase 24 — token_observer feeds the heartbeat's t/s stat
            plan_raw = yield from _stream_phase(
                sys_p, usr_p, "planning",
                model=model, temperature=0.2, num_ctx=16384,
                token_observer=log.token,
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
    relevant_lessons = _get_relevant_lessons(
        book_id, section_type, _lessons_query,
        kinds=("knowledge", "idea", "decision", "paper", "episode"),
        return_dicts=True,
    )
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

    if resume_content is None:
        system, user = rag_prompts.write_section_v2(
            section_type, topic, results,
            book_plan=b_plan, prior_summaries=prior_summaries,
            paragraph_plan=paragraph_plan,
            target_words=effective_target_words,
            section_plan=section_plan,
            lessons=relevant_lessons,
            style_fingerprint_block=style_fingerprint_block,
            visuals=section_visuals,
        )
        yield {"type": "progress", "stage": "writing",
               "detail": f"Generating initial draft (~{effective_target_words} words, "
                         f"{len(section_visuals)} visuals)..."}
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
        try:
            sys_s, usr_s = rag_prompts.score_draft(section_type, topic, content, results)
            score_raw = yield from _stream_phase(
                sys_s, usr_s, "scoring",
                model=scorer_model, temperature=0.0, num_ctx=16384,
                token_observer=log.token,
            )
            scores = json.loads(_clean_json(score_raw), strict=False)
        except Exception as exc:
            logger.warning("Scoring failed: %s", exc)
            log.event("scoring_failed", message=str(exc)[:200])
            scores = {"overall": 0.5, "weakest_dimension": "unknown",
                      "revision_instruction": "Improve overall quality."}

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

        # Fix 1: Run claim verification as part of scoring
        log.stage("verifying", iteration=iteration + 1)
        yield {"type": "progress", "stage": "verifying",
               "detail": "Verifying citations..."}
        try:
            sys_v, usr_v = rag_prompts.verify_claims(content, results)
            verify_raw = yield from _stream_phase(
                sys_v, usr_v, "verifying",
                model=model, temperature=0.0, num_ctx=16384,
                token_observer=log.token,
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
                try:
                    # Phase 15.3: streamed CoVe so the GUI sees question
                    # generation + each independent answer flow as token
                    # events instead of going dark for 1 + N silent
                    # llm_complete calls.
                    cove = yield from _cove_verify_streaming(content, results, model=model)
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
        try:
            sys_rs, usr_rs = rag_prompts.score_draft(section_type, topic, revised, results)
            rescore_raw = yield from _stream_phase(
                sys_rs, usr_rs, "rescoring",
                model=scorer_model, temperature=0.0, num_ctx=16384,
                token_observer=log.token,
            )
            new_scores = json.loads(_clean_json(rescore_raw), strict=False)
        except Exception:
            new_scores = {"overall": overall}

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
    }

    for i, sec in enumerate(sections, start=1):
        slug = sec["slug"]
        title = sec["title"]
        existing = existing_by_slug.get(slug)
        already_exists = existing is not None

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


def insert_citations_stream(
    draft_id: str,
    *,
    model: str | None = None,
    candidate_k: int = 8,
    max_needs: int | None = None,
    dry_run: bool = False,
    save: bool = True,
) -> Iterator[Event]:
    """Run the two-stage citation insertion pass over a saved draft.

    Phase 46.A — auditable citations. Each yielded event is typed; the
    CLI/web layers render them. The heavy work is LLM pass-1 (identify
    needs), hybrid retrieval per need, LLM pass-2 (choose), and a
    single deterministic insertion pass that rewrites the draft text.

    Parameters:
        draft_id:     prefix of the drafts.id (matches `LIKE draft_id%`)
        model:        LLM override; falls back to section-model override,
                      then settings.llm_model
        candidate_k:  top-K candidates per claim to show pass-2
        max_needs:    cap the number of locations processed (saves LLM
                      calls when a draft has many candidate locations)
        dry_run:      don't rewrite the draft; yield what would change
        save:         persist the rewritten content as a new draft version
    """
    import json as _json
    import re as _re

    from sqlalchemy import text
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import complete
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    with get_session() as session:
        row = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.topic, d.content,
                   d.book_id::text, d.chapter_id::text, d.version,
                   d.sources, d.word_count
            FROM drafts d WHERE d.id::text LIKE :q
            LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()
    if not row:
        yield {"type": "error", "message": f"Draft not found: {draft_id}"}
        return

    d_id, d_title, d_section, d_topic, d_content, d_book_id, d_chapter_id, \
        d_version, d_sources, d_wc = row
    d_sources = d_sources or []

    # Per-section model override (Phase 37) so a custom model picked for
    # this section also drives its citation editing.
    if model is None and d_chapter_id and d_section:
        with get_session() as session:
            override = _get_section_model(session, d_chapter_id, d_section)
        if override:
            model = override

    yield {"type": "progress", "stage": "citation_scan_start",
           "detail": f"Scanning draft '{d_title}' for citation opportunities…"}

    sys_n, usr_n = rag_prompts.citation_needs(
        section_type=d_section or "unknown",
        section_title=d_topic or d_title or "",
        draft_content=d_content or "",
    )
    try:
        raw_needs = complete(sys_n, usr_n, model=model, temperature=0.1)
    except Exception as exc:
        yield {"type": "error", "message": f"citation_needs LLM call failed: {exc}"}
        return

    needs = _parse_citation_needs_json(raw_needs)
    if not needs:
        yield {"type": "completed", "draft_id": d_id,
               "n_needs": 0, "n_inserted": 0,
               "message": "no citation opportunities identified"}
        return
    if max_needs is not None and max_needs > 0:
        needs = needs[:max_needs]

    yield {"type": "citation_needs", "count": len(needs),
           "needs": needs}

    qdrant = get_client()

    # existing_sources tracks the draft's current [N] → source_id mapping
    # so newly-inserted citations don't clobber existing marker numbers.
    # d_sources comes back as a JSON list from the DB.
    existing_sources: list[dict] = list(d_sources) if isinstance(d_sources, list) else []
    next_marker = max((s.get("n", 0) for s in existing_sources), default=0) + 1

    new_citations: list[dict] = []   # {location, marker, source}
    skipped: list[dict] = []         # {location, reason}

    for idx, need in enumerate(needs):
        location = (need.get("location") or "").strip()
        claim    = (need.get("claim")    or "").strip()
        query    = (need.get("query")    or "").strip() or claim
        if not location or not query:
            skipped.append({"index": idx, "reason": "missing location or query"})
            continue

        yield {"type": "progress", "stage": "citation_retrieve",
               "detail": f"({idx+1}/{len(needs)}) retrieving for '{claim[:60]}'…"}

        with get_session() as session:
            results, _ = _retrieve(session, qdrant, query, context_k=candidate_k)

        if not results:
            skipped.append({"index": idx, "location": location,
                            "reason": "no retrieval candidates"})
            yield {"type": "citation_skipped", "index": idx,
                   "location": location, "reason": "no retrieval candidates"}
            continue

        # Shape the candidate list for the pass-2 prompt.
        cand_dicts = [
            {
                "title":   r.get("title") or r.get("paper_title") or "",
                "year":    r.get("year") or "",
                "section": r.get("section_type") or "",
                "preview": r.get("content") or r.get("content_preview") or "",
                "doc_id":  r.get("document_id") or r.get("doc_id") or "",
                "chunk_id": r.get("chunk_id") or r.get("id") or "",
            }
            for r in results
        ]

        yield {"type": "citation_candidates", "index": idx,
               "location": location, "candidates": [
                   {"i": i, "title": c["title"], "year": c["year"]}
                   for i, c in enumerate(cand_dicts)
               ]}

        sys_c, usr_c = rag_prompts.citation_choose(
            claim=claim, location=location, candidates=cand_dicts,
        )
        try:
            raw_choice = complete(sys_c, usr_c, model=model, temperature=0.1)
        except Exception as exc:
            skipped.append({"index": idx, "reason": f"choose LLM failed: {exc}"})
            yield {"type": "citation_skipped", "index": idx,
                   "reason": f"choose LLM failed: {exc}"}
            continue

        choice = _parse_citation_choice_json(raw_choice)
        verdict = (choice.get("verdict") or "").upper()
        if verdict != "CITE":
            reason = choice.get("reason") or "no good candidate"
            skipped.append({"index": idx, "location": location, "reason": reason})
            yield {"type": "citation_skipped", "index": idx,
                   "location": location, "reason": reason}
            continue

        # For each chosen candidate, attach a marker.
        picked = choice.get("chosen") or []
        if not isinstance(picked, list) or not picked:
            skipped.append({"index": idx, "reason": "empty chosen list"})
            continue

        cite_markers_for_location: list[int] = []
        for pick in picked:
            try:
                ci = int(pick.get("candidate_index"))
            except (TypeError, ValueError):
                continue
            if ci < 0 or ci >= len(cand_dicts):
                continue
            cand = cand_dicts[ci]
            marker = next_marker
            next_marker += 1
            source_entry = {
                "n": marker,
                "title":   cand["title"],
                "year":    cand["year"],
                "doc_id":  cand["doc_id"],
                "chunk_id": cand["chunk_id"],
                "confidence": float(pick.get("confidence") or 0.0),
                "why":     pick.get("why") or "",
                "inserted_by": "phase46_citation_loop",
            }
            existing_sources.append(source_entry)
            new_citations.append({"location": location, "marker": marker,
                                  "source": source_entry})
            cite_markers_for_location.append(marker)

            yield {"type": "citation_selected", "index": idx,
                   "location": location, "marker": marker,
                   "source": {"title": cand["title"], "year": cand["year"]},
                   "confidence": source_entry["confidence"]}

    # ── Deterministic insertion pass ──────────────────────────────────
    #
    # Replace each `location` verbatim with `location [N]` (or
    # `location [N][M]` if two markers). We use a single Python-level
    # str.replace per location to avoid regex misfires on LaTeX/mathjax
    # content. Only the first occurrence is replaced, which matches the
    # pass-1 contract (location must be unique in the draft).
    new_content = d_content or ""
    n_inserted = 0
    for cite in new_citations:
        loc = cite["location"]
        marker = cite["marker"]
        if loc not in new_content:
            # LLM paraphrased slightly or whitespace differs; skip.
            skipped.append({"location": loc,
                            "reason": "exact location no longer in draft"})
            continue
        # Append the marker to the end of the matched span rather than a
        # literal substring append, so consecutive selections for the
        # same location get grouped: "claim. [7][8]"
        replacement = f"{loc} [{marker}]"
        new_content = new_content.replace(loc, replacement, 1)
        n_inserted += 1

    yield {"type": "citation_inserted", "count": n_inserted,
           "skipped": len(skipped)}

    if dry_run or not save:
        yield {"type": "completed", "draft_id": d_id,
               "n_needs": len(needs), "n_inserted": n_inserted,
               "n_skipped": len(skipped),
               "new_content_preview": new_content[:400],
               "citations": new_citations,
               "saved": False}
        return

    # Persist as a new draft version. Matches the pattern used by
    # revise_draft_stream: increment version, keep the same title,
    # parent to the old draft_id.
    with get_session() as session:
        new_id = session.execute(text("""
            INSERT INTO drafts
              (title, book_id, chapter_id, section_type, topic, content,
               word_count, sources, model_used, version, parent_draft_id,
               summary, status, custom_metadata)
            SELECT title, book_id, chapter_id, section_type, topic,
                   :content,
                   ARRAY_LENGTH(string_to_array(:content, ' '), 1),
                   CAST(:sources AS jsonb),
                   COALESCE(:model, model_used),
                   version + 1,
                   id,
                   summary, status,
                   COALESCE(custom_metadata, '{}'::jsonb) ||
                     jsonb_build_object(
                       'phase46_citations_added', :n_inserted,
                       'phase46_citations_skipped', :n_skipped)
            FROM drafts WHERE id::text = :src
            RETURNING id::text
        """), {
            "content": new_content,
            "sources": _json.dumps(existing_sources),
            "model":   model,
            "n_inserted": n_inserted,
            "n_skipped": len(skipped),
            "src": d_id,
        }).scalar()
        session.commit()

    yield {"type": "completed",
           "draft_id": new_id or d_id,
           "parent_draft_id": d_id,
           "n_needs": len(needs), "n_inserted": n_inserted,
           "n_skipped": len(skipped),
           "citations": new_citations,
           "saved": True}


# ── JSON parsers for the citation loop ────────────────────────────────

def _strip_code_fence(raw: str) -> str:
    """Remove ```json ... ``` fences if the LLM added them."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()


def _parse_citation_needs_json(raw: str) -> list[dict]:
    """Tolerant parser for CITATION_NEEDS output."""
    import json as _json
    s = _strip_code_fence(raw)
    try:
        data = _json.loads(s)
    except Exception:
        return []
    needs = data.get("needs") if isinstance(data, dict) else None
    if not isinstance(needs, list):
        return []
    out: list[dict] = []
    for n in needs:
        if not isinstance(n, dict):
            continue
        out.append({
            "location": str(n.get("location") or "").strip(),
            "claim":    str(n.get("claim")    or "").strip(),
            "query":    str(n.get("query")    or "").strip(),
            "reason":   str(n.get("reason")   or ""),
            "existing_citations": n.get("existing_citations") or [],
        })
    return out


def _parse_citation_choice_json(raw: str) -> dict:
    """Tolerant parser for CITATION_CHOOSE output."""
    import json as _json
    s = _strip_code_fence(raw)
    try:
        data = _json.loads(s)
    except Exception:
        return {"verdict": "NONE", "reason": "unparseable LLM output"}
    if not isinstance(data, dict):
        return {"verdict": "NONE", "reason": "non-object LLM output"}
    return data


# ════════════════════════════════════════════════════════════════════
# Phase 46.C — Ensemble NeurIPS-rubric review + meta-reviewer
# ════════════════════════════════════════════════════════════════════
#
# AI-Scientist `perform_review.perform_review` pattern, adapted:
# run N independent reviewers over one draft section (each with
# temperature 0.75 and its own stance — neutral / pessimistic /
# optimistic) then fuse with a meta-reviewer. Persist the full panel
# + meta to drafts.custom_metadata.ensemble_review so downstream tools
# can read the history.
#
# Why ensemble: a single-pass review (the pre-Phase-46 `review_draft_
# stream`) is high-variance; different runs disagree on scores and
# decisions. Taking the median across N independent reviewers reduces
# the variance predictably (≈ 1/√N) and the "disagreement" field on
# the meta-review acts as an early warning that a draft is borderline
# (some reviewers accept, some reject — likely genuinely ambiguous).


_DEFAULT_REVIEW_STANCES = ["neutral", "pessimistic", "optimistic"]


def _parse_review_json(raw: str) -> dict:
    """Tolerant parser for a single NeurIPS-rubric review record."""
    import json as _json
    s = _strip_code_fence(raw)
    try:
        data = _json.loads(s)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    if n % 2:
        return float(xs[n // 2])
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def _compute_meta_fallback(reviews: list[dict]) -> dict:
    """If the meta-reviewer LLM fails or returns junk, compute a
    mechanical fallback locally so the user never gets nothing.

    Mirrors the instructions in REVIEW_META_SYSTEM (median scores,
    union of free-text lists, disagreement = stdev/range).
    """
    import statistics as _st

    def _nums(field: str) -> list[float]:
        out = []
        for r in reviews:
            v = r.get(field)
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    overall_vals = _nums("overall")
    disagreement = 0.0
    if len(overall_vals) >= 2:
        span = max(overall_vals) - min(overall_vals)
        if span > 0:
            disagreement = round(min(1.0, _st.stdev(overall_vals) / span), 2)

    med_overall = _median(overall_vals) or 0.0
    if med_overall >= 8:
        decision = "accept"
    elif med_overall >= 7:
        decision = "weak_accept"
    elif med_overall >= 5:
        decision = "borderline"
    elif med_overall >= 4:
        decision = "weak_reject"
    else:
        decision = "reject"

    # Dedup + agreement-rank free-text lists
    def _union_ranked(field: str) -> list[str]:
        from collections import Counter
        c: Counter = Counter()
        for r in reviews:
            items = r.get(field) or []
            if not isinstance(items, list):
                continue
            seen_in_review: set = set()
            for it in items:
                if not isinstance(it, str):
                    continue
                norm = it.strip()
                if not norm or norm in seen_in_review:
                    continue
                seen_in_review.add(norm)
                c[norm] += 1
        # Sort by (count desc, first-occurrence) and cap at 10
        return [item for item, _n in c.most_common(10)]

    confidences = _nums("confidence")
    any_ethics = any(bool(r.get("ethical_concerns")) for r in reviews)

    return {
        "summary":       (reviews[0].get("summary", "") if reviews else "")[:1200],
        "strengths":     _union_ranked("strengths"),
        "weaknesses":    _union_ranked("weaknesses"),
        "questions":     _union_ranked("questions"),
        "limitations":   _union_ranked("limitations"),
        "ethical_concerns": any_ethics,
        "soundness":     _median(_nums("soundness")),
        "presentation":  _median(_nums("presentation")),
        "contribution":  _median(_nums("contribution")),
        "overall":       med_overall,
        "confidence":    max(confidences) if confidences else None,
        "decision":      decision,
        "rationale":     f"Mechanical fallback: median across {len(reviews)} reviewers.",
        "disagreement":  disagreement,
        "source":        "fallback_no_meta_llm",
    }


def ensemble_review_stream(
    draft_id: str,
    *,
    n_reviewers: int = 3,
    temperature: float = 0.75,
    model: str | None = None,
    context_k: int = 12,
    save: bool = True,
    stances: list[str] | None = None,
) -> Iterator[Event]:
    """Phase 46.C — run an ensemble of independent reviewers + a meta-reviewer.

    Parameters:
        draft_id:    prefix of drafts.id
        n_reviewers: how many independent reviewers to run (default 3)
        temperature: per-reviewer temperature (0.75 = NeurIPS convention)
        model:       LLM override; falls back to section-model override,
                     then settings.llm_model
        context_k:   passages to retrieve for each reviewer to see
        save:        if True, persist panel + meta into
                     drafts.custom_metadata.ensemble_review
        stances:     list of stances to cycle through. Defaults to
                     ["neutral", "pessimistic", "optimistic"] —
                     AI-Scientist's positivity-bias mitigation.

    Events yielded:
        progress, reviewer_done, meta_review_start, completed, error
    """
    from sqlalchemy import text
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import complete
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    # Load the draft
    with get_session() as session:
        row = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.topic, d.content,
                   d.chapter_id::text
            FROM drafts d WHERE d.id::text LIKE :q
            LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()
    if not row:
        yield {"type": "error", "message": f"Draft not found: {draft_id}"}
        return
    d_id, d_title, d_section, d_topic, d_content, d_chapter_id = row

    # Per-section model override (Phase 37)
    if model is None and d_chapter_id and d_section:
        with get_session() as session:
            override = _get_section_model(session, d_chapter_id, d_section)
        if override:
            model = override

    # Stance rotation
    stance_pool = list(stances) if stances else _DEFAULT_REVIEW_STANCES
    if not stance_pool:
        stance_pool = ["neutral"]
    assigned_stances: list[str] = []
    for i in range(max(1, n_reviewers)):
        assigned_stances.append(stance_pool[i % len(stance_pool)])

    yield {"type": "progress", "stage": "retrieve",
           "detail": f"Retrieving context for ensemble review of '{d_title}'…"}

    qdrant = get_client()
    search_query = f"{d_section or ''} {d_topic or d_title}"
    with get_session() as session:
        results, _ = _retrieve(session, qdrant, search_query, context_k=context_k)

    reviews: list[dict] = []
    for i, stance in enumerate(assigned_stances, 1):
        yield {"type": "progress", "stage": "reviewing",
               "detail": f"Reviewer {i}/{len(assigned_stances)} ({stance})…"}
        sys_r, usr_r = rag_prompts.review_neurips(
            d_section, d_topic or d_title, d_content, results,
            stance=stance,
        )
        try:
            raw = complete(sys_r, usr_r, model=model, temperature=temperature)
        except Exception as exc:
            yield {"type": "reviewer_done", "index": i, "stance": stance,
                   "status": "error", "message": str(exc)}
            continue

        parsed = _parse_review_json(raw)
        if not parsed:
            yield {"type": "reviewer_done", "index": i, "stance": stance,
                   "status": "parse_error"}
            continue
        parsed["_stance"] = stance
        parsed["_reviewer_index"] = i
        reviews.append(parsed)
        yield {
            "type": "reviewer_done",
            "index": i, "stance": stance,
            "status": "ok",
            "overall":    parsed.get("overall"),
            "decision":   parsed.get("decision"),
            "soundness":  parsed.get("soundness"),
            "presentation": parsed.get("presentation"),
            "contribution": parsed.get("contribution"),
            "confidence": parsed.get("confidence"),
        }

    if not reviews:
        yield {"type": "error",
               "message": "all reviewers failed — cannot compute meta-review"}
        return

    yield {"type": "meta_review_start",
           "n_reviewers": len(reviews),
           "overall_scores": [r.get("overall") for r in reviews]}

    # Meta-reviewer pass. If it fails or returns garbage, fall back to
    # the deterministic aggregation so the user always gets a result.
    sys_m, usr_m = rag_prompts.review_meta(
        d_section, d_topic or d_title, reviews,
    )
    meta: dict = {}
    try:
        raw_meta = complete(sys_m, usr_m, model=model, temperature=0.2)
        meta = _parse_review_json(raw_meta)
    except Exception as exc:
        logger.warning("meta-reviewer LLM failed: %s", exc)

    if not meta or meta.get("overall") is None:
        meta = _compute_meta_fallback(reviews)
    else:
        # Defensive: make sure critical fields exist even if the LLM
        # omitted them; fall back to the mechanical values per-field.
        fb = _compute_meta_fallback(reviews)
        for k in ("soundness", "presentation", "contribution",
                  "confidence", "disagreement", "decision"):
            if meta.get(k) in (None, ""):
                meta[k] = fb[k]
        meta.setdefault("source", "llm_meta_reviewer")

    if save:
        import json as _json
        with get_session() as session:
            session.execute(text("""
                UPDATE drafts
                SET custom_metadata =
                  COALESCE(custom_metadata, '{}'::jsonb)
                  || jsonb_build_object(
                       'ensemble_review',
                       CAST(:payload AS jsonb))
                WHERE id::text = :did
            """), {
                "payload": _json.dumps({
                    "n_reviewers": len(reviews),
                    "stances":     assigned_stances,
                    "reviews":     reviews,
                    "meta":        meta,
                    "model":       model,
                    "temperature": temperature,
                    "context_k":   context_k,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }),
                "did": d_id,
            })
            session.commit()

    yield {
        "type": "completed",
        "draft_id": d_id,
        "n_reviewers": len(reviews),
        "meta": meta,
        "reviews": reviews,
        "saved": save,
    }


# ── Phase 54.6.14 — BMAD-inspired Critic Skills ─────────────────────────
# Two critic passes that are orthogonal to the existing graded review:
# adversarial finds ≥10 issues, edge-case hunter exhaustively enumerates
# unhandled paths. Both adapted from bmad-code-org/BMAD-METHOD (MIT).


def adversarial_review_stream(
    draft_id: str,
    *,
    model: str | None = None,
) -> Iterator[Event]:
    """Cynical critic pass over a saved draft. Yields tokens + a final
    ``completed`` event with the full findings markdown. Not persisted
    to ``review_feedback`` so it coexists with the normal graded
    review rather than overwriting it.
    """
    from sqlalchemy import text
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    with get_session() as session:
        row = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.topic, d.content,
                   d.chapter_id::text
            FROM drafts d WHERE d.id::text LIKE :q
            LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()
    if not row:
        yield {"type": "error", "message": f"Draft not found: {draft_id}"}
        return
    d_id, d_title, d_section, d_topic, d_content, d_chapter_id = row

    if model is None and d_chapter_id and d_section:
        with get_session() as session:
            override = _get_section_model(session, d_chapter_id, d_section)
        if override:
            model = override

    yield {"type": "progress", "stage": "retrieval",
           "detail": "Gathering evidence to contrast with…"}
    qdrant = get_client()
    search_query = f"{d_section or ''} {d_topic or d_title}"
    with get_session() as session:
        results, _ = _retrieve(session, qdrant, search_query, context_k=8)

    yield {"type": "progress", "stage": "adversarial_review",
           "detail": f"Adversarial review of '{d_title}'…"}
    sys_p, usr_p = rag_prompts.adversarial_review(
        d_section, d_topic or d_title, d_content or "", results,
    )
    out: list[str] = []
    # Thinking-model headroom — same rationale as extract-kg / consensus.
    for tok in llm_stream(sys_p, usr_p, model=model, num_ctx=24576, keep_alive=-1):
        out.append(tok)
        yield {"type": "token", "text": tok}

    findings = "".join(out).strip()
    # Strip any <think> blocks so the returned findings are clean.
    import re as _re
    findings = _re.sub(r"<think>.*?</think>\s*", "", findings,
                       flags=_re.DOTALL).strip()

    # Heuristic: count the findings (numbered list items). If the
    # model returned <10, flag it in the event so the UI can warn.
    n_findings = len(_re.findall(r"^\s*(\d+)[\.)]\s", findings, _re.MULTILINE))

    yield {
        "type": "completed",
        "draft_id": d_id,
        "findings_markdown": findings,
        "n_findings": n_findings,
    }


def edge_case_hunter_stream(
    draft_id: str,
    *,
    model: str | None = None,
) -> Iterator[Event]:
    """Exhaustive path enumeration over a draft's claims. Returns
    structured findings via Ollama's format=json_schema so the UI can
    render a proper table (location / trigger / consequence / severity).
    """
    import json as _json
    from sqlalchemy import text
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import complete as llm_complete
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    with get_session() as session:
        row = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.topic, d.content,
                   d.chapter_id::text
            FROM drafts d WHERE d.id::text LIKE :q
            LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()
    if not row:
        yield {"type": "error", "message": f"Draft not found: {draft_id}"}
        return
    d_id, d_title, d_section, d_topic, d_content, d_chapter_id = row

    if model is None and d_chapter_id and d_section:
        with get_session() as session:
            override = _get_section_model(session, d_chapter_id, d_section)
        if override:
            model = override

    yield {"type": "progress", "stage": "retrieval",
           "detail": "Pulling context for edge-case analysis…"}
    qdrant = get_client()
    search_query = f"{d_section or ''} {d_topic or d_title}"
    with get_session() as session:
        results, _ = _retrieve(session, qdrant, search_query, context_k=6)

    yield {"type": "progress", "stage": "edge_cases",
           "detail": f"Walking branches for '{d_title}'…"}
    sys_p, usr_p = rag_prompts.edge_case_hunter(
        d_section, d_topic or d_title, d_content or "", results,
    )
    schema = {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "location":    {"type": "string"},
                        "trigger":     {"type": "string"},
                        "consequence": {"type": "string"},
                        "severity":    {"type": "string",
                                        "enum": ["high", "medium", "low"]},
                    },
                    "required": ["location", "trigger", "consequence"],
                },
            },
        },
        "required": ["findings"],
    }
    try:
        # Phase 54.6.32 — num_predict=4096 cap on structured output
        # to prevent runaway generation (see wiki_ops entity extraction).
        raw = llm_complete(
            sys_p, usr_p, model=model,
            temperature=0.0, num_ctx=24576, num_predict=4096, keep_alive=-1,
            format=schema,
        )
    except Exception as exc:
        yield {"type": "error", "message": f"LLM call failed: {exc}"}
        return

    import re as _re
    cleaned = _re.sub(r"<think>.*?</think>\s*", "", raw or "",
                      flags=_re.DOTALL).strip()
    if not cleaned:
        yield {"type": "error",
               "message": "LLM returned empty output (likely context overflow)."}
        return
    try:
        data = _json.loads(cleaned)
    except Exception as exc:
        yield {"type": "error", "message": f"JSON parse failed: {exc}"}
        return

    findings = data.get("findings") or []
    # Stable ordering — high severity first, then medium, then low.
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: sev_rank.get(
        (f.get("severity") or "low").lower(), 3))

    # Stream findings one at a time so the UI can render incrementally.
    for i, f in enumerate(findings):
        yield {"type": "finding", "index": i, "data": f}

    yield {
        "type": "completed",
        "draft_id": d_id,
        "n_findings": len(findings),
        "findings": findings,
    }
