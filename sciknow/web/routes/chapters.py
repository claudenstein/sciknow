"""``sciknow.web.routes.chapters`` — chapter CRUD + planning endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. 9 handlers
covering chapter list / create / update / sections-edit / delete /
reorder / plan-sections / resolved-targets.

Heavy cross-module deps (resolved via lazy `_app` shim per handler):
  `_app._get_book_data`, `_app._chapter_sections_dicts`, `_app._normalize_section`,
  `_app._book_id`, `_app.BookBibliography`, `_app.BIBLIOGRAPHY_PSEUDO_ID`,
  `_app.BIBLIOGRAPHY_TITLE`, `logger`.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

router = APIRouter()


@router.get("/api/chapters")
async def api_chapters():
    """Return chapter list with their sections for sidebar building."""
    from sciknow.web import app as _app
    book, chapters, drafts, gaps, comments = _app._get_book_data()

    chapter_drafts = {}
    for d in drafts:
        key = d[9] or "__none__"
        if key not in chapter_drafts:
            chapter_drafts[key] = []
        existing = [x for x in chapter_drafts[key] if x[2] == d[2]]
        if not existing:
            chapter_drafts[key].append(d)

    result = []
    for ch in chapters:
        ch_id, ch_num, ch_title, ch_desc, tq, tc, ch_sections_template = ch
        ch_ds = chapter_drafts.get(ch_id, [])
        template_dicts = _app._chapter_sections_dicts(ch)
        template = [s["slug"] for s in template_dicts]
        section_order = {s: i for i, s in enumerate(template)}
        sections = []
        for d in sorted(ch_ds, key=lambda x: section_order.get(_app._normalize_section(x[2] or ""), 99)):
            sections.append({
                "id": d[0], "type": d[2] or "text",
                "version": d[6] or 1, "words": d[4] or 0,
            })
        result.append({
            "id": ch_id, "num": ch_num, "title": ch_title,
            "description": ch_desc, "topic_query": tq,
            "sections": sections,
            "sections_template": template,
            "sections_meta": template_dicts,
        })

    if result:
        try:
            with get_session() as _bib_session:
                _bib = _app.BookBibliography.from_book(_bib_session, _app._book_id)
        except Exception as exc:
            _app.logger.warning("bibliography fetch failed: %s", exc)
            _bib = _app.BookBibliography()
        _bib_ch_num = int(result[-1]["num"] or len(result)) + 1
        result.append({
            "id": _app.BIBLIOGRAPHY_PSEUDO_ID,
            "num": _bib_ch_num,
            "title": _app.BIBLIOGRAPHY_TITLE,
            "description": "All publications cited across the book, numbered once.",
            "topic_query": "",
            "sections": [{
                "id": _app.BIBLIOGRAPHY_PSEUDO_ID,
                "type": "bibliography",
                "version": 1,
                "words": sum(len((s or "").split()) for s in _bib.global_sources),
            }],
            "sections_template": ["bibliography"],
            "sections_meta": [{
                "slug": "bibliography",
                "title": _app.BIBLIOGRAPHY_TITLE,
                "plan": "Auto-generated — do not edit by hand.",
            }],
            "is_bibliography": True,
        })

    return {"chapters": result, "gaps_count": len([g for g in gaps if g[3] == "open"])}


@router.post("/api/chapters")
async def create_chapter(
    title: str = Form(...),
    description: str = Form(""),
    topic_query: str = Form(""),
    number: int = Form(None),
):
    """Add a chapter to the book."""
    from sciknow.web import app as _app
    with get_session() as session:
        if number is None:
            max_n = session.execute(text(
                "SELECT COALESCE(MAX(number), 0) FROM book_chapters WHERE book_id = :bid"
            ), {"bid": _app._book_id}).scalar()
            number = max_n + 1

        session.execute(text("""
            INSERT INTO book_chapters (book_id, number, title, description, topic_query)
            VALUES (:bid, :num, :title, :desc, :tq)
        """), {"bid": _app._book_id, "num": number, "title": title,
               "desc": description or None, "tq": topic_query or None})
        session.commit()

    return JSONResponse({"ok": True, "number": number})


@router.put("/api/chapters/{chapter_id}")
async def update_chapter(
    chapter_id: str,
    title: str = Form(None),
    description: str = Form(None),
    topic_query: str = Form(None),
):
    """Update a chapter's title, description, or topic_query."""
    updates = []
    params: dict = {"cid": chapter_id}
    if title is not None:
        updates.append("title = :title")
        params["title"] = title
    if description is not None:
        updates.append("description = :desc")
        params["desc"] = description
    if topic_query is not None:
        updates.append("topic_query = :tq")
        params["tq"] = topic_query

    if not updates:
        return JSONResponse({"ok": True})

    with get_session() as session:
        session.execute(text(
            f"UPDATE book_chapters SET {', '.join(updates)} WHERE id::text = :cid"
        ), params)
        session.commit()

    return JSONResponse({"ok": True})


@router.post("/api/chapters/{chapter_id}/sections/adopt")
async def adopt_orphan_section_endpoint(chapter_id: str, request: Request):
    """Phase 25 — adopt an orphan draft's section_type into the chapter's sections list."""
    from sciknow.core.book_ops import adopt_orphan_section as _adopt
    from sciknow.web import app as _app

    body = await request.json()
    slug = (body.get("slug") or "").strip()
    title = body.get("title")
    plan = body.get("plan")
    if not slug:
        raise HTTPException(400, "slug is required")
    try:
        result = _adopt(_app._book_id, chapter_id, slug, title=title, plan=plan)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return JSONResponse(result)


@router.put("/api/chapters/{chapter_id}/sections")
async def update_chapter_sections(chapter_id: str, request: Request):
    """Replace a chapter's sections list. Phase 18."""
    from sciknow.core.book_ops import _normalize_chapter_sections

    body = await request.json()
    raw_sections = body.get("sections", [])
    normalized = _normalize_chapter_sections(raw_sections)

    seen: set[str] = set()
    deduped: list[dict] = []
    for s in normalized:
        if s["slug"] in seen:
            continue
        seen.add(s["slug"])
        deduped.append(s)

    with get_session() as session:
        session.execute(text("""
            UPDATE book_chapters SET sections = CAST(:secs AS jsonb)
            WHERE id::text = :cid
        """), {"cid": chapter_id, "secs": json.dumps(deduped)})
        session.commit()

    return JSONResponse({"ok": True, "sections": deduped})


@router.delete("/api/chapters/{chapter_id}")
async def delete_chapter(chapter_id: str):
    """Delete a chapter (drafts are preserved but unlinked)."""
    from sciknow.web import app as _app
    if chapter_id == _app.BIBLIOGRAPHY_PSEUDO_ID:
        raise HTTPException(
            400,
            "The Bibliography is auto-generated from cited sources and "
            "cannot be deleted. It disappears automatically if no draft "
            "has any citations."
        )
    with get_session() as session:
        session.execute(text(
            "UPDATE drafts SET chapter_id = NULL WHERE chapter_id::text = :cid"
        ), {"cid": chapter_id})
        session.execute(text(
            "DELETE FROM book_chapters WHERE id::text = :cid"
        ), {"cid": chapter_id})
        session.commit()

    return JSONResponse({"ok": True})


@router.post("/api/chapters/reorder")
async def reorder_chapters(request: Request):
    """Reorder chapters. Body: {"chapter_ids": ["id1", "id2", ...]}"""
    body = await request.json()
    chapter_ids = body.get("chapter_ids", [])

    with get_session() as session:
        for i, cid in enumerate(chapter_ids, 1):
            session.execute(text(
                "UPDATE book_chapters SET number = :num WHERE id::text = :cid"
            ), {"num": i, "cid": cid})
        session.commit()

    return JSONResponse({"ok": True})


@router.post("/api/chapters/{chapter_id}/plan-sections")
async def api_chapter_plan_sections(
    chapter_id: str,
    force: bool = Form(False),
    model: str = Form(None),
):
    """Phase 54.6.155 — web wrapper over book_ops.generate_section_plan."""
    from sciknow.core.book_ops import (
        generate_section_plan,
        _get_chapter_sections_normalized,
    )
    from sciknow.web import app as _app
    with get_session() as session:
        row = session.execute(text(
            "SELECT book_id::text FROM book_chapters WHERE id::text = :cid"
        ), {"cid": chapter_id}).fetchone()
        if not row:
            return JSONResponse({"error": f"no chapter {chapter_id!r}"}, status_code=404)
        book_id = row[0]
        sections = _get_chapter_sections_normalized(session, chapter_id)

    results = []
    for s in sections:
        slug = s.get("slug", "")
        try:
            r = generate_section_plan(
                book_id, chapter_id, slug,
                model=model or None,
                force=force,
            )
            results.append({
                "slug": slug,
                "title": s.get("title", ""),
                "wrote": r["wrote"],
                "n_concepts": r["n_concepts"],
                "skipped_reason": r["skipped_reason"],
                "first_bullet": (r["new_plan"].splitlines() or [""])[0][:120],
            })
        except Exception as exc:
            _app.logger.warning("plan-sections failed for %s: %s", slug, exc)
            results.append({
                "slug": slug,
                "title": s.get("title", ""),
                "wrote": False,
                "n_concepts": 0,
                "skipped_reason": f"error: {str(exc)[:120]}",
                "first_bullet": "",
            })
    n_planned = sum(1 for r in results if r["wrote"])
    n_skipped = sum(1 for r in results if not r["wrote"] and r.get("skipped_reason"))
    return JSONResponse({
        "chapter_id": chapter_id,
        "results": results,
        "n_total": len(results),
        "n_planned": n_planned,
        "n_skipped": n_skipped,
    })


@router.get("/api/chapters/{chapter_id}/resolved-targets")
async def api_chapter_resolved_targets(chapter_id: str):
    """Phase 54.6.149 — per-section target + which fallback level fired."""
    from sciknow.core.book_ops import (
        _get_book_length_target, _section_target_words,
        _get_section_target_words, _get_section_concept_density_target,
        _get_chapter_sections_normalized, _count_plan_concepts,
        _get_section_plan, DEFAULT_TARGET_CHAPTER_WORDS,
    )
    from sciknow.core.project_type import get_project_type
    from sciknow.web import app as _app

    with get_session() as session:
        book_row = session.execute(text("""
            SELECT book_type, COALESCE(custom_metadata, '{}'::jsonb),
                   CAST(:cid AS uuid)
            FROM books WHERE id::text = :bid LIMIT 1
        """), {"bid": _app._book_id, "cid": chapter_id}).fetchone()
        book_type = (book_row[0] if book_row else None) or "scientific_book"
        book_meta = (book_row[1] if book_row else {}) or {}
        if isinstance(book_meta, str):
            try:
                book_meta = json.loads(book_meta)
            except Exception:
                book_meta = {}
        ch_row = session.execute(text(
            "SELECT target_words FROM book_chapters WHERE id::text = :cid"
        ), {"cid": chapter_id}).fetchone()
        chapter_override = (
            int(ch_row[0]) if ch_row and ch_row[0] and int(ch_row[0]) > 0 else None
        )
        if chapter_override:
            chapter_target, chapter_level = chapter_override, "explicit_chapter_override"
        elif isinstance(book_meta.get("target_chapter_words"), (int, float)) and book_meta["target_chapter_words"] > 0:
            chapter_target, chapter_level = int(book_meta["target_chapter_words"]), "book_default"
        else:
            try:
                chapter_target = get_project_type(book_type).default_target_chapter_words
                chapter_level = "type_default"
            except Exception:
                chapter_target, chapter_level = DEFAULT_TARGET_CHAPTER_WORDS, "hardcoded_fallback"

        sections = _get_chapter_sections_normalized(session, chapter_id)
        n = max(1, len(sections))
        chapter_split = _section_target_words(chapter_target, n)

        out_sections = []
        pt = None
        try:
            pt = get_project_type(book_type)
        except Exception:
            pt = None
        wpc_mid = None
        if pt is not None:
            wlo, whi = pt.words_per_concept_range
            wpc_mid = (wlo + whi) // 2

        for s in sections:
            slug = s.get("slug", "")
            title = s.get("title", "")
            override = _get_section_target_words(session, chapter_id, slug)
            if override is not None:
                out_sections.append({
                    "slug": slug, "title": title,
                    "target": override, "level": "explicit_section_override",
                    "concepts": None, "wpc_midpoint": wpc_mid,
                    "explanation": f"Per-section override ({override:,} words).",
                })
                continue
            plan_text = _get_section_plan(session, chapter_id, slug)
            n_concepts = _count_plan_concepts(plan_text)
            if n_concepts > 0 and wpc_mid:
                concept_target = _get_section_concept_density_target(
                    session, chapter_id, slug, _app._book_id,
                )
                if concept_target:
                    out_sections.append({
                        "slug": slug, "title": title,
                        "target": concept_target, "level": "concept_density",
                        "concepts": n_concepts, "wpc_midpoint": wpc_mid,
                        "explanation": (
                            f"Bottom-up: {n_concepts} concept(s) × {wpc_mid} "
                            f"words/concept = {concept_target:,} words. Cap 4 "
                            f"per Cowan 2001 (RESEARCH.md §24)."
                        ),
                    })
                    continue
            out_sections.append({
                "slug": slug, "title": title,
                "target": chapter_split, "level": "chapter_split",
                "concepts": None, "wpc_midpoint": wpc_mid,
                "explanation": (
                    f"Top-down: chapter target {chapter_target:,} ÷ "
                    f"{n} sections = {chapter_split:,}. Add a section plan "
                    f"to switch to bottom-up concept-density sizing."
                ),
            })

    return JSONResponse({
        "chapter_target": chapter_target,
        "chapter_level": chapter_level,
        "sections": out_sections,
    })
