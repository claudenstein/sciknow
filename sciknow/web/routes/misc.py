"""``sciknow.web.routes.misc`` — misc endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. Behaviour
unchanged. Cross-module deps resolved via the standard lazy
`from sciknow.web import app as _app` shim.
"""
from __future__ import annotations

import asyncio
import json
import threading

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response, StreamingResponse, PlainTextResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

router = APIRouter()


@router.get("/api/section/{draft_id}")
async def api_section(draft_id: str):
    """Return section data as JSON for SPA navigation."""
    from sciknow.web import app as _app
    book, chapters, drafts, gaps, comments = _app._get_book_data()

    # Build draft map and find the target
    draft_map = {}
    chapter_drafts = {}
    for d in drafts:
        draft_map[d[0]] = d
        key = d[9] or "__none__"
        if key not in chapter_drafts:
            chapter_drafts[key] = []
        existing = [x for x in chapter_drafts[key] if x[2] == d[2]]
        # Phase 54.6.309 — first-seen wins; the SELECT already orders
        # active-first then MAX(version), see _app._get_book_data.
        if not existing:
            chapter_drafts[key].append(d)

    # Phase 27 — chapter_id → sections JSONB lookup so the display
    # title can be derived from the meta even if drafts.title is stale.
    chapter_sections_by_id = {ch[0]: ch[6] for ch in chapters}

    # Phase 54.6.309 — synthetic Bibliography pseudo-chapter.
    if draft_id == _app.BIBLIOGRAPHY_PSEUDO_ID:
        try:
            with get_session() as _session:
                _bib = _app.BookBibliography.from_book(_session, _app._book_id)
        except Exception as exc:
            _app.logger.warning("bibliography rebuild failed: %s", exc)
            _bib = _app.BookBibliography()
        _md = render_bibliography_markdown(_bib)
        _bib_ch_num = (chapters[-1][1] if chapters else 0) + 1 if chapters else 1
        return {
            "id": _app.BIBLIOGRAPHY_PSEUDO_ID,
            "title": _app.BIBLIOGRAPHY_TITLE,
            "display_title": _app.BIBLIOGRAPHY_TITLE,
            "section_type": "bibliography",
            "content_html": _app._md_to_html(_md),
            "content_raw": _md,
            "word_count": sum(len((s or "").split()) for s in _bib.global_sources),
            "version": 1,
            "review_feedback": "",
            "review_html": "<em>The bibliography is auto-generated and not reviewed.</em>",
            "sources_html": _app._render_sources(_bib.global_sources),
            "sources": list(_bib.global_sources),
            "comments_html": "",
            "chapter_id": _app.BIBLIOGRAPHY_PSEUDO_ID,
            "chapter_num": _bib_ch_num,
            "chapter_title": _app.BIBLIOGRAPHY_TITLE,
            "status": "drafted",
            "target_words": None,
            "is_bibliography": True,
        }

    active = draft_map.get(draft_id)
    if not active:
        raise HTTPException(404, "Draft not found")

    # Group comments for this draft
    active_comments = [c for c in comments if c[1] == draft_id]
    sources = json.loads(active[5]) if isinstance(active[5], str) else (active[5] or [])

    # Phase 54.6.309 — apply the global bibliography renumbering so the
    # fetched content + right-panel list match what the reader would
    # show on a full page reload.
    try:
        with get_session() as _session:
            _bib = _app.BookBibliography.from_book(_session, _app._book_id)
        _remapped_content = _bib.remap_content(draft_id, active[3] or "")
        _cited_globals = _bib.cited_sources_for_draft(draft_id)
        if _cited_globals:
            sources_for_panel = _cited_globals
        else:
            sources_for_panel = sources
    except Exception as _exc:
        _app.logger.warning("global renumber failed (draft %s): %s", draft_id, _exc)
        _remapped_content = active[3] or ""
        sources_for_panel = sources

    # Phase 22 — fetch the section's word target so the GUI can render
    # a progress bar in the subtitle. The target lives on the draft's
    # custom_metadata (set by autowrite/Phase 17). If the draft was
    # made by a single-shot write_section_stream that pre-dates Phase 17,
    # we fall back to deriving it from the book's chapter_target /
    # num_sections.
    target_words = None
    try:
        from sciknow.core.book_ops import (
            _get_book_length_target, _section_target_words,
            _get_chapter_num_sections,
        )
        with get_session() as _session:
            # 1) Try the draft's own custom_metadata.target_words
            row = _session.execute(text("""
                SELECT custom_metadata FROM drafts WHERE id::text = :did LIMIT 1
            """), {"did": active[0]}).fetchone()
            meta = (row[0] if row else None) or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            tw = meta.get("target_words") if isinstance(meta, dict) else None
            if tw and tw > 0:
                target_words = int(tw)
            elif active[9]:
                # 2) Derive from book + chapter section count
                chapter_target = _get_book_length_target(_session, _app._book_id)
                num_sections = _get_chapter_num_sections(_session, str(active[9]))
                target_words = _section_target_words(chapter_target, num_sections)
    except Exception as exc:
        _app.logger.warning("target_words lookup failed: %s", exc)

    # Phase 27 — display_title is derived from the chapter sections meta
    # at every read so renaming a section in the chapter modal updates
    # the center frame on the next navigation. Falls back to the stored
    # drafts.title for orphans (section_type doesn't match any current
    # slug) and any other lookup failure.
    sections_raw = chapter_sections_by_id.get(active[9])
    display_title = _app._draft_display_title(
        draft_title=active[1],
        section_type=active[2],
        chapter_num=active[12],
        chapter_title=active[13],
        chapter_sections_raw=sections_raw,
    )

    return {
        "id": active[0],
        "title": active[1],
        "display_title": display_title,
        "section_type": active[2],
        "content_html": _app._md_to_html(_remapped_content),
        "content_raw": _remapped_content,
        "word_count": active[4] or 0,
        "version": active[6] or 1,
        "review_feedback": active[8] or "",
        "review_html": _app._md_to_html(active[8]) if active[8] else "<em>No review yet.</em>",
        "sources_html": _app._render_sources(sources_for_panel),
        "sources": sources_for_panel,
        "comments_html": _app._render_comments(active_comments),
        "chapter_id": active[9],
        "chapter_num": active[12],
        "chapter_title": active[13],
        "status": _app._get_draft_status(draft_id),
        "target_words": target_words,
    }


@router.get("/api/dashboard")
async def api_dashboard():
    """Return dashboard data: completion heatmap, stats, gaps.

    Phase 30 — heatmap restructured to use POSITIONAL columns (1, 2,
    3 ...) instead of a union of all section_type slugs. The previous
    layout showed every distinct slug from every chapter as its own
    column, which left orphan/legacy slugs (overview/key_evidence/etc)
    visible even after the user defined custom sections, and made the
    table sparse + confusing as different chapters use different
    section names.

    New layout:
      - Columns are 1..N where N = max(num_sections_in_chapter)
      - Each row shows the chapter's actual sections in their
        defined order
      - Empty cells past the chapter's section count are marked
        ``status="absent"`` so the GUI can render them as blank
      - Each cell carries the actual section title for hover tooltips
    """
    from sciknow.web import app as _app
    book, chapters, drafts, gaps, comments = _app._get_book_data()

    # Build draft lookup: chapter_id -> {section_type: {version, words, id, has_review}}
    ch_section_drafts: dict[str, dict] = {}
    total_words = 0
    for d in drafts:
        draft_id, title, sec_type, content, wc, sources, version, summary, \
            review_fb, ch_id, parent_id, created, ch_num, ch_title = d
        total_words += wc or 0
        if not ch_id:
            continue
        if ch_id not in ch_section_drafts:
            ch_section_drafts[ch_id] = {}
        # Phase 54.6.309 — _app._get_book_data sorts active-first then highest-
        # version within each (chapter, section) group, so the first row
        # seen is the one to display. No more MAX(version) compare.
        existing = ch_section_drafts[ch_id].get(sec_type)
        if not existing:
            ch_section_drafts[ch_id][sec_type] = {
                "id": draft_id, "version": version or 1, "words": wc or 0,
                "has_review": bool(review_fb),
            }

    # Phase 30 — compute the per-chapter sections lists FIRST so we
    # can find max(N) for the column count.
    chapter_sections: dict[str, list[dict]] = {}
    for ch in chapters:
        chapter_sections[ch[0]] = _app._chapter_sections_dicts(ch)
    max_sections = max(
        (len(s) for s in chapter_sections.values()), default=1
    )

    heatmap = []
    for ch in chapters:
        ch_id, ch_num, ch_title, ch_desc, tq, tc, ch_sections_template = ch
        sections_meta = chapter_sections.get(ch_id, [])
        row = {
            "num": ch_num, "title": ch_title, "id": ch_id, "cells": [],
            "sections_template": [s["slug"] for s in sections_meta],
            "description": ch_desc or "",
            "topic_query": tq or "",
        }
        secs = ch_section_drafts.get(ch_id, {})
        # First N cells: this chapter's actual sections in order
        for sec_meta in sections_meta:
            slug = sec_meta["slug"]
            sec_title = sec_meta.get("title") or slug
            info = secs.get(slug)
            if info:
                status = "reviewed" if info["has_review"] else "drafted"
                row["cells"].append({
                    "type": slug, "title": sec_title, "status": status,
                    "draft_id": info["id"], "version": info["version"],
                    "words": info["words"],
                })
            else:
                row["cells"].append({
                    "type": slug, "title": sec_title, "status": "empty",
                })
        # Remaining cells: this chapter has fewer sections than max(N).
        # Render as 'absent' so the GUI can blank them out.
        while len(row["cells"]) < max_sections:
            row["cells"].append({
                "type": None, "title": None, "status": "absent",
            })
        heatmap.append(row)

    open_gaps = [{"id": g[0], "type": g[1], "description": g[2], "status": g[3],
                  "chapter_num": g[4]} for g in gaps if g[3] == "open"]

    # Phase 33 — cumulative autowrite stats from the Layer 0 telemetry
    # tables. Aggregate token usage and time across all completed runs
    # for this book. Fail-soft: if the query errors, return zeros.
    autowrite_stats = {"total_tokens": 0, "total_seconds": 0, "total_runs": 0}
    try:
        with get_session() as session:
            aw = session.execute(text("""
                SELECT
                    COUNT(*),
                    COALESCE(SUM(tokens_used), 0),
                    COALESCE(SUM(EXTRACT(EPOCH FROM (finished_at - started_at))), 0)
                FROM autowrite_runs
                WHERE book_id::text = :bid
                  AND status = 'completed'
                  AND finished_at IS NOT NULL
            """), {"bid": _app._book_id}).fetchone()
            if aw:
                autowrite_stats = {
                    "total_runs": int(aw[0] or 0),
                    "total_tokens": int(aw[1] or 0),
                    "total_seconds": int(aw[2] or 0),
                }
    except Exception as exc:
        _app.logger.warning("dashboard autowrite stats failed: %s", exc)

    # Phase 35 — Total Compute ledger aggregated from llm_usage_log.
    # Covers every LLM-backed op (write/review/revise/argue/gaps/
    # autowrite/plan/...) so the dashboard can show cumulative GPU
    # compute per book, plus a per-operation breakdown. Autowrite is a
    # strict subset — the Autowrite Effort panel and the `autowrite`
    # row of by_operation reconcile.
    total_compute = {
        "total_tokens": 0, "total_seconds": 0.0, "total_jobs": 0,
        "by_operation": [],
    }
    try:
        with get_session() as session:
            totals = session.execute(text("""
                SELECT
                    COUNT(*),
                    COALESCE(SUM(tokens), 0),
                    COALESCE(SUM(duration_seconds), 0)
                FROM llm_usage_log
                WHERE book_id::text = :bid
            """), {"bid": _app._book_id}).fetchone()
            by_op = session.execute(text("""
                SELECT operation,
                       COUNT(*),
                       COALESCE(SUM(tokens), 0),
                       COALESCE(SUM(duration_seconds), 0)
                FROM llm_usage_log
                WHERE book_id::text = :bid
                GROUP BY operation
                ORDER BY SUM(tokens) DESC
            """), {"bid": _app._book_id}).fetchall()
            if totals:
                total_compute = {
                    "total_jobs": int(totals[0] or 0),
                    "total_tokens": int(totals[1] or 0),
                    "total_seconds": float(totals[2] or 0.0),
                    "by_operation": [
                        {
                            "operation": r[0],
                            "jobs": int(r[1] or 0),
                            "tokens": int(r[2] or 0),
                            "seconds": float(r[3] or 0.0),
                        }
                        for r in by_op
                    ],
                }
    except Exception as exc:
        _app.logger.warning("dashboard total_compute stats failed: %s", exc)

    return {
        "heatmap": heatmap,
        # Phase 30 — column headers are positional integers, not slugs
        "n_columns": max_sections,
        "stats": {
            "total_words": total_words,
            "chapters": len(chapters),
            "drafts": len(drafts),
            "gaps_open": len(open_gaps),
            "comments": len(comments),
        },
        "autowrite_stats": autowrite_stats,
        "total_compute": total_compute,
        "gaps": open_gaps,
    }


@router.get("/api/versions/{draft_id}")
async def api_versions(draft_id: str):
    """Return the version chain for a draft (all versions of the same section).

    Phase 54.6.309 — each version now carries its ``final_overall`` score
    (autowrite) and an ``is_active`` flag so the Versions panel can show
    scores next to each version and mark which one the reader is
    currently displaying. The active rule is the same one used by the
    main reader collapse: explicit ``custom_metadata.is_active = true``
    wins; otherwise the highest version is active.
    """
    with get_session() as session:
        # Find the draft to get its chapter_id and section_type
        draft = session.execute(text("""
            SELECT d.chapter_id::text, d.section_type, d.book_id::text
            FROM drafts d WHERE d.id::text LIKE :q LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()
        if not draft:
            raise HTTPException(404, "Draft not found")

        ch_id, sec_type, book_id = draft

        # Get all drafts for this chapter + section_type (all versions).
        # Pull final_overall + is_active out of custom_metadata so the
        # panel can render scores and the active marker in one payload.
        rows = session.execute(text("""
            SELECT d.id::text, d.version, d.word_count, d.created_at,
                   d.parent_draft_id::text,
                   d.review_feedback IS NOT NULL AS has_review,
                   (d.custom_metadata->>'final_overall')::float AS final_overall,
                   (d.custom_metadata->>'is_active')::boolean AS is_active,
                   d.model_used,
                   d.custom_metadata->>'version_name' AS version_name,
                   d.custom_metadata->>'version_description' AS version_description,
                   d.updated_at
            FROM drafts d
            WHERE d.chapter_id::text = :cid AND d.section_type = :st AND d.book_id::text = :bid
            ORDER BY d.version ASC
        """), {"cid": ch_id, "st": sec_type, "bid": book_id}).fetchall()

    versions = [
        {"id": r[0], "version": r[1], "word_count": r[2] or 0,
         "created_at": r[3].isoformat() if r[3] else "",
         "parent_id": r[4],
         "has_review": bool(r[5]),
         "final_overall": float(r[6]) if r[6] is not None else None,
         "is_active": bool(r[7]) if r[7] is not None else False,
         "model_used": r[8] or "",
         "version_name": r[9] or "",
         "version_description": r[10] or "",
         "updated_at": r[11].isoformat() if r[11] else ""}
        for r in rows
    ]
    # If no version carries an explicit is_active flag, mark the
    # highest-version one active so the UI has a sensible default.
    if versions and not any(v["is_active"] for v in versions):
        versions_sorted = sorted(versions, key=lambda v: v["version"] or 0, reverse=True)
        versions_sorted[0]["is_active"] = True

    return {"versions": versions}


@router.get("/api/diff/{old_id}/{new_id}")
async def api_diff(old_id: str, new_id: str):
    """Return a word-level diff between two drafts as HTML."""
    import difflib

    with get_session() as session:
        old = session.execute(text(
            "SELECT content FROM drafts WHERE id::text LIKE :q LIMIT 1"
        ), {"q": f"{old_id}%"}).fetchone()
        new = session.execute(text(
            "SELECT content FROM drafts WHERE id::text LIKE :q LIMIT 1"
        ), {"q": f"{new_id}%"}).fetchone()

    if not old or not new:
        raise HTTPException(404, "Draft not found")

    old_words = (old[0] or "").split()
    new_words = (new[0] or "").split()

    sm = difflib.SequenceMatcher(None, old_words, new_words)
    html_parts = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            html_parts.append(" ".join(old_words[i1:i2]))
        elif op == "delete":
            html_parts.append(f'<del class="diff-del">{" ".join(old_words[i1:i2])}</del>')
        elif op == "insert":
            html_parts.append(f'<ins class="diff-ins">{" ".join(new_words[j1:j2])}</ins>')
        elif op == "replace":
            html_parts.append(f'<del class="diff-del">{" ".join(old_words[i1:i2])}</del>')
            html_parts.append(f'<ins class="diff-ins">{" ".join(new_words[j1:j2])}</ins>')

    return {"diff_html": " ".join(html_parts)}


@router.get("/api/kg")
async def api_kg(
    subject: str = "",
    predicate: str = "",
    object: str = "",
    document_id: str = "",
    any_side: str = "",
    limit: int = 200,
    offset: int = 0,
):
    """Phase 30 — return knowledge_graph triples filtered by any of:
    subject (substring, case-insensitive), predicate (exact),
    object (substring, case-insensitive), document_id (exact UUID).

    Phase 48 — `any_side` matches rows where the substring appears in
    EITHER subject or object. Used by the graph's right-click
    "Expand around this node" action to fetch a 1-hop ego network in
    one query without changing the subject/object filter boxes.

    Returns at most `limit` rows (capped at 1000), with pagination via
    offset. Each row has the source paper title joined in for the GUI
    so the user can see which document a triple was extracted from.
    """
    limit = max(1, min(int(limit or 200), 1000))
    offset = max(0, int(offset or 0))
    # Phase 41 — always-bind pattern. Every optional filter is bound
    # to its real value or NULL so the SQL can stay fully static.
    # Removes the WHERE-clause f-string that the Phase 22 audit flagged.
    subj_q = subject.strip()
    pred_q = predicate.strip()
    obj_q = object.strip()
    doc_q = document_id.strip()
    any_q = any_side.strip()
    params: dict = {
        "limit": limit,
        "offset": offset,
        "subject_q": f"%{subj_q}%" if subj_q else None,
        "predicate_q": pred_q or None,
        "object_q": f"%{obj_q}%" if obj_q else None,
        "doc_q": doc_q or None,
        "any_q": f"%{any_q}%" if any_q else None,
    }

    with get_session() as session:
        # Total count for pagination
        total = session.execute(text("""
            SELECT COUNT(*) FROM knowledge_graph kg
            WHERE (:subject_q   IS NULL OR kg.subject ILIKE :subject_q)
              AND (:predicate_q IS NULL OR kg.predicate = :predicate_q)
              AND (:object_q    IS NULL OR kg.object ILIKE :object_q)
              AND (:doc_q       IS NULL OR kg.source_doc_id::text = :doc_q)
              AND (:any_q       IS NULL OR kg.subject ILIKE :any_q
                                        OR kg.object  ILIKE :any_q)
        """), params).scalar()

        rows = session.execute(text("""
            SELECT kg.subject, kg.predicate, kg.object,
                   kg.source_doc_id::text, kg.confidence,
                   pm.title, kg.source_sentence
            FROM knowledge_graph kg
            LEFT JOIN paper_metadata pm ON pm.document_id = kg.source_doc_id
            WHERE (:subject_q   IS NULL OR kg.subject ILIKE :subject_q)
              AND (:predicate_q IS NULL OR kg.predicate = :predicate_q)
              AND (:object_q    IS NULL OR kg.object ILIKE :object_q)
              AND (:doc_q       IS NULL OR kg.source_doc_id::text = :doc_q)
              AND (:any_q       IS NULL OR kg.subject ILIKE :any_q
                                        OR kg.object  ILIKE :any_q)
            ORDER BY kg.confidence DESC, kg.subject
            LIMIT :limit OFFSET :offset
        """), params).fetchall()

        # Distinct predicates so the GUI can populate a filter dropdown
        predicates = [r[0] for r in session.execute(text("""
            SELECT DISTINCT predicate FROM knowledge_graph
            ORDER BY predicate LIMIT 200
        """)).fetchall()]

    return {
        "total": int(total or 0),
        "offset": offset,
        "limit": limit,
        "predicates": predicates,
        "triples": [
            {
                "subject": r[0], "predicate": r[1], "object": r[2],
                "source_doc_id": r[3], "confidence": float(r[4] or 1.0),
                "source_title": r[5],
                # Phase 48d — may be None for pre-migration-0019 triples
                "source_sentence": r[6],
            }
            for r in rows
        ],
    }


@router.get("/api/chapter-reader/{chapter_id}")
async def chapter_reader(chapter_id: str, only_section: str = ""):
    """Return all sections of a chapter concatenated for continuous reading.

    Phase 18 — section order respects book_chapters.sections JSONB,
    not a hardcoded paper-style map. Sources are stitched into a
    global renumbered list so citation click-to-source works.

    Phase 31 — accepts an optional ``only_section`` query parameter.
    When set, the response contains only that one section but in
    the same continuous-scroll layout (same h2 styling, same
    sources panel, just one section). Used by the Read button when
    the user has a section selected — they expect Read to filter to
    that section, not always dump the whole chapter.
    """
    from sciknow.web import app as _app
    from sciknow.core.book_ops import _normalize_chapter_sections

    with get_session() as session:
        ch = session.execute(text("""
            SELECT bc.number, bc.title, bc.sections FROM book_chapters bc
            WHERE bc.id::text = :cid
        """), {"cid": chapter_id}).fetchone()
        if not ch:
            raise HTTPException(404, "Chapter not found")

        # Phase 31 — fetch only the matching section_type when filtered
        if only_section:
            drafts = session.execute(text("""
                SELECT d.id::text, d.section_type, d.content, d.word_count,
                       d.version, d.status, d.sources
                FROM drafts d
                WHERE d.chapter_id::text = :cid
                  AND LOWER(d.section_type) = LOWER(:sec)
                ORDER BY d.version DESC
            """), {"cid": chapter_id, "sec": only_section.strip()}).fetchall()
        else:
            drafts = session.execute(text("""
                SELECT d.id::text, d.section_type, d.content, d.word_count,
                       d.version, d.status, d.sources
                FROM drafts d
                WHERE d.chapter_id::text = :cid
                ORDER BY d.section_type, d.version DESC
            """), {"cid": chapter_id}).fetchall()

    # Keep only latest version per section_type
    seen: dict[str, tuple] = {}
    for d in drafts:
        st = d[1] or "text"
        if st not in seen or (d[4] or 1) > (seen[st][4] or 1):
            seen[st] = d

    # Phase 18 — order by chapter's sections list (user's chosen order),
    # not hardcoded paper-style. Sections not in the meta list go last
    # in alphabetical order — visible but tagged as "orphaned" so the
    # user can spot a renamed section that left a stale draft behind.
    sections_meta = _normalize_chapter_sections(ch[2])
    title_by_slug = {s["slug"]: s["title"] for s in sections_meta}
    order_by_slug = {s["slug"]: i for i, s in enumerate(sections_meta)}

    def _sort_key(d):
        slug = (d[1] or "").strip().lower()
        return (order_by_slug.get(slug, 999), slug)

    section_drafts = sorted(seen.values(), key=_sort_key)

    # Phase 18 — global source renumbering. Each draft's `sources` is a
    # 1-indexed list. We build a global list, map each draft's local
    # number to the global number, then rewrite the draft's [N] tags
    # accordingly. Dedup is by the source string itself (the APA-rendered
    # text); two drafts citing the same paper share one global source.
    global_sources: list[str] = []
    source_to_global: dict[str, int] = {}

    def _register_source(src: str) -> int:
        if src in source_to_global:
            return source_to_global[src]
        global_sources.append(src)
        n = len(global_sources)
        source_to_global[src] = n
        return n

    combined_html = ""
    total_words = 0
    for d in section_drafts:
        slug = (d[1] or "").strip().lower()
        title = title_by_slug.get(slug) or _titleify_slug_for_display(slug)
        is_orphan = slug not in title_by_slug and sections_meta
        # Per-draft local→global citation map
        srcs = d[6]
        if isinstance(srcs, str):
            try:
                srcs = json.loads(srcs)
            except Exception:
                srcs = []
        local_to_global: dict[int, int] = {}
        for local_idx, src_text in enumerate(srcs or [], start=1):
            if not src_text:
                continue
            local_to_global[local_idx] = _register_source(src_text)

        content = d[2] or ""
        # Rewrite [N] → [global_N]. Citations whose local N has no
        # source (orphan citation, e.g. writer hallucinated a number)
        # get rewritten to a clearly broken marker so the user notices.
        def _renumber(match):
            local = int(match.group(1))
            global_n = local_to_global.get(local)
            if global_n is None:
                return f"[?]"
            return f"[{global_n}]"
        content = re.sub(r'\[(\d+)\]', _renumber, content)

        orphan_tag = " (orphaned section)" if is_orphan else ""
        combined_html += (
            f'<h2 class="reader-section-title" id="reader-section-{slug}">'
            f'{title}{orphan_tag}</h2>'
        )
        combined_html += _app._md_to_html(content)
        total_words += d[3] or 0

    sources_html = _app._render_sources(global_sources)

    return {
        "chapter_num": ch[0],
        "chapter_title": ch[1],
        "html": combined_html,
        "total_words": total_words,
        "section_count": len(section_drafts),
        # Phase 18 — global sources panel for the chapter view, plus a
        # short outline so the user can jump to any section.
        "sources_html": sources_html,
        "outline": [
            {"slug": (d[1] or "").strip().lower(),
             "title": title_by_slug.get((d[1] or "").strip().lower())
                      or _titleify_slug_for_display(d[1] or ""),
             "words": d[3] or 0}
            for d in section_drafts
        ],
    }


@router.get("/api/corkboard")
async def corkboard_data():
    """Return data for the corkboard view: cards for each chapter/section.

    Phase 18 — uses each chapter's actual sections list (the user's
    chosen names + order) instead of the previous hardcoded paper-style
    [introduction, methods, results, discussion, conclusion]. Chapters
    with custom sections now show ALL their sections; chapters without
    a sections list show the default science-book set.
    """
    from sciknow.web import app as _app
    book, chapters, drafts, gaps, comments = _app._get_book_data()

    # Build latest draft per chapter+section
    ch_sections: dict[str, dict] = {}
    for d in drafts:
        draft_id, title, sec_type, content, wc, sources, version, summary, \
            review_fb, ch_id, parent_id, created, ch_num, ch_title = d
        if not ch_id:
            continue
        if ch_id not in ch_sections:
            ch_sections[ch_id] = {}
        # Phase 54.6.309 — see comment in dashboard builder above.
        existing = ch_sections[ch_id].get(sec_type)
        if not existing:
            ch_sections[ch_id][sec_type] = {
                "draft_id": draft_id, "version": version or 1,
                "words": wc or 0, "summary": (summary or "")[:200],
                "has_review": bool(review_fb),
                "status": "drafted",  # default; real status from DB below
            }

    # Fetch statuses
    with get_session() as session:
        status_rows = session.execute(text("""
            SELECT id::text, COALESCE(status, 'drafted') FROM drafts WHERE book_id = :bid
        """), {"bid": _app._book_id}).fetchall()
    status_map = {r[0]: r[1] for r in status_rows}

    cards = []
    for ch in chapters:
        ch_id, ch_num, ch_title, ch_desc, tq, tc, ch_sections_template = ch
        # Phase 18 — chapter's own sections list (with rich {slug,title,plan})
        section_template = _app._chapter_sections_dicts(ch)
        secs = ch_sections.get(ch_id, {})
        for tmpl in section_template:
            slug = tmpl["slug"]
            display_title = tmpl["title"]
            info = secs.get(slug)
            if info:
                info["status"] = status_map.get(info["draft_id"], "drafted")
                cards.append({
                    "chapter_id": ch_id, "chapter_num": ch_num,
                    "chapter_title": ch_title,
                    "section_type": slug,
                    "section_title": display_title,
                    "draft_id": info["draft_id"],
                    "version": info["version"], "words": info["words"],
                    "summary": info["summary"], "has_review": info["has_review"],
                    "status": info["status"],
                })
            else:
                cards.append({
                    "chapter_id": ch_id, "chapter_num": ch_num,
                    "chapter_title": ch_title,
                    "section_type": slug,
                    "section_title": display_title,
                    "draft_id": None,
                    "version": 0, "words": 0, "summary": "",
                    "has_review": False, "status": "to_do",
                })

    return {"cards": cards}


@router.get("/api/methods")
async def api_methods(kind: str = "elicitation"):
    from sciknow.core.methods import list_methods
    try:
        return JSONResponse({"methods": list_methods(kind), "kind": kind})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/api/setup/status")
async def api_setup_status():
    """Phase 46.F — aggregate "where am I in the pipeline?" snapshot.

    Returns per-stage booleans + counts so the wizard can render a
    progress trail: which steps are done, which need running. Cheap —
    one round-trip to PG + one to Qdrant, no embeddings or LLM.
    """
    out: dict = {}
    with get_session() as session:
        out["n_documents"] = session.execute(text(
            "SELECT COUNT(*) FROM documents"
        )).scalar() or 0
        out["n_complete"] = session.execute(text(
            "SELECT COUNT(*) FROM documents WHERE ingestion_status='complete'"
        )).scalar() or 0
        out["n_chunks"] = session.execute(text(
            "SELECT COUNT(*) FROM chunks"
        )).scalar() or 0
        out["n_with_topic"] = session.execute(text(
            "SELECT COUNT(*) FROM paper_metadata "
            "WHERE topic_cluster IS NOT NULL AND topic_cluster != ''"
        )).scalar() or 0
        try:
            out["n_wiki_pages"] = session.execute(text(
                "SELECT COUNT(*) FROM wiki_pages"
            )).scalar() or 0
        except Exception:
            out["n_wiki_pages"] = 0
        try:
            out["n_books"] = session.execute(text(
                "SELECT COUNT(*) FROM books"
            )).scalar() or 0
        except Exception:
            out["n_books"] = 0
    # RAPTOR presence — cheap Qdrant count with an indexed filter
    raptor_levels: dict[str, int] = {}
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        from sciknow.storage.qdrant import PAPERS_COLLECTION, get_client
        qdrant = get_client()
        for lvl in (1, 2, 3):
            try:
                info = qdrant.count(
                    collection_name=PAPERS_COLLECTION,
                    count_filter=Filter(must=[
                        FieldCondition(key="node_level", match=MatchValue(value=lvl))
                    ]), exact=False,
                )
                n = info.count if hasattr(info, "count") else int(info)
                if n > 0:
                    raptor_levels[f"L{lvl}"] = n
            except Exception:
                pass
    except Exception:
        pass
    out["raptor_levels"] = raptor_levels

    # Active project info (Phase 43)
    try:
        from sciknow.core.project import get_active_project
        p = get_active_project()
        out["project"] = {
            "slug": p.slug,
            "is_default": p.is_default,
            "data_dir": str(p.data_dir),
        }
    except Exception:
        out["project"] = {"slug": "unknown"}
    return JSONResponse(out)


@router.get("/api/bench/section-lengths")
async def api_bench_section_lengths():
    """Phase 54.6.159 — surface the 54.6.157 section-length IQR bench
    data in the web UI. Runs the same bench function so the numbers
    stay in sync (no duplicate SQL); returns the per-section rows as
    JSON with the alignment tag parsed out of the bench note for easy
    rendering.
    """
    from sciknow.web import app as _app
    from sciknow.testing.bench import b_corpus_section_length_distribution
    try:
        metrics = list(b_corpus_section_length_distribution())
    except Exception as exc:  # noqa: BLE001
        _app.logger.warning("section-length bench failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)

    rows = []
    # The bench emits "iqr_<section_type>" metrics with a dotted note.
    # Parse the note into structured fields so the UI doesn't have to
    # re-implement the text parsing.
    for m in metrics:
        st = m.name.removeprefix("iqr_")
        note = m.note or ""
        # Note shape: "n=640 · median=630w · ref IQR 400-760 → aligned"
        parts = [p.strip() for p in note.split("·")]
        data: dict = {"section_type": st, "iqr": str(m.value), "unit": m.unit}
        for p in parts:
            if p.startswith("n="):
                try:
                    data["n"] = int(p[2:])
                except ValueError:
                    pass
            elif p.startswith("median="):
                try:
                    data["median"] = int(p[len("median="):].rstrip("w"))
                except ValueError:
                    pass
            elif p.startswith("ref IQR"):
                # "ref IQR 400-760 → aligned"
                try:
                    ref_part, tag = p.split("→")
                    data["ref_iqr"] = ref_part.removeprefix("ref IQR").strip()
                    data["alignment"] = tag.strip()
                except ValueError:
                    pass
        rows.append(data)
    return JSONResponse({"sections": rows})


@router.post("/api/cli-stream")
async def api_cli_stream(request: Request):
    """Phase 54.6.97 — generic CLI dispatcher for draft-level actions.

    Body: ``{argv: ["book", "verify-draft", "<id>"]}``. Only commands on
    the allowlist below are accepted — keeps this from becoming a
    remote-shell. Streams stdout as SSE log events via
    ``_app._spawn_cli_streaming`` just like the backup endpoints.
    """
    from sciknow.web import app as _app
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    argv = body.get("argv") or []
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise HTTPException(400, "argv must be a non-empty list of strings")
    ALLOWED: set[tuple[str, str]] = {
        ("book", "verify-draft"),
        ("book", "align-citations"),
        ("book", "ensemble-review"),
        # Phase 54.6.111 — corpus-growth operations. Each is idempotent
        # and stateful (updates PG + Qdrant, logs to data/*.log); none
        # writes to the filesystem anywhere the user hasn't already
        # sanctioned by running sciknow.
        ("db", "enrich"),
        ("db", "expand"),
        ("db", "refresh-retractions"),
        ("db", "classify-papers"),
        ("db", "parse-tables"),
        ("db", "expand-oeuvre"),
        ("db", "expand-inbound"),
        ("db", "reconcile-preprints"),
        # Phase 54.6.156 — book-wide auto-plan. Reuses the existing
        # sciknow book plan-sections CLI (54.6.154) via the cli-stream
        # SSE channel so the long-running book-scope action (48 sections
        # × ~5-10s = 4-8 min) streams progress without needing a bespoke
        # job pipeline.
        ("book", "plan-sections"),
        # Phase 54.6.162 — pre-export L3 VLM claim-depiction verify
        # (Phase 54.6.145). Exposes the CLI-only finalize-draft in the
        # Verify dropdown so users don't drop to the CLI before export.
        ("book", "finalize-draft"),
    }
    if len(argv) < 2 or (argv[0], argv[1]) not in ALLOWED:
        raise HTTPException(403, f"command not on allowlist: {argv[:2]}")
    job_id, _ = _app._create_job("cli_stream_" + argv[1].replace("-", "_"))
    loop = asyncio.get_event_loop()
    _app._spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


@router.post("/api/server/shutdown")
async def api_server_shutdown(request: Request):
    """Phase 54.6.2 — cleanly stop the running ``sciknow book serve``.

    We can't hot-swap the DB/Qdrant singletons against the new
    ``.active-project``, so switching projects mid-session requires a
    restart. Rather than asking the user to switch to a terminal and
    Ctrl-C, this endpoint fires SIGTERM at the server's own PID — the
    uvicorn event loop handles the signal, shuts down gracefully, and
    the terminal returns to ``$``. The user then re-runs
    ``sciknow book serve <book>`` to pick up the new project.

    The frontend confirms twice before calling this (it IS a destructive
    UX — any unsaved job is killed).
    """
    import os as _os
    import signal as _signal
    import threading as _threading
    import time as _time
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    # Delay 200ms so the HTTP response can flush before the process dies.
    def _fire():
        _time.sleep(0.2)
        _os.kill(_os.getpid(), _signal.SIGTERM)
    _threading.Thread(target=_fire, daemon=True).start()
    return JSONResponse({
        "ok": True,
        "message": "Server shutting down. Re-run `sciknow book serve <book>`"
                   " in your terminal to pick up the new active project.",
    })
