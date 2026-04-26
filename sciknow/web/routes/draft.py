"""``sciknow.web.routes.draft`` — draft endpoints.

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


@router.post("/api/draft/{draft_id}/activate")
async def api_activate_draft(draft_id: str):
    """Phase 54.6.309 — mark one version as the active draft for its
    (chapter_id, section_type) group. Clears ``is_active`` on siblings
    in the same group so there's at most one active version per section.

    Body: empty. Returns the new active draft id.
    """
    with get_session() as session:
        row = session.execute(text("""
            SELECT d.chapter_id::text, d.section_type, d.book_id::text
            FROM drafts d WHERE d.id::text = :did
        """), {"did": draft_id}).fetchone()
        if not row:
            raise HTTPException(404, "Draft not found")
        ch_id, sec_type, book_id = row

        # Clear is_active on all siblings in the same (chapter, section)
        # group, then set it on the target. The || merge preserves other
        # keys (score_history, target_words, …).
        session.execute(text("""
            UPDATE drafts
               SET custom_metadata = custom_metadata - 'is_active'
             WHERE chapter_id::text = :cid
               AND section_type = :st
               AND book_id::text = :bid
               AND id::text <> :did
        """), {"cid": ch_id, "st": sec_type, "bid": book_id, "did": draft_id})
        session.execute(text("""
            UPDATE drafts
               SET custom_metadata = custom_metadata || '{"is_active": true}'::jsonb
             WHERE id::text = :did
        """), {"did": draft_id})
        session.commit()

    return JSONResponse({"ok": True, "active_draft_id": draft_id})


@router.post("/api/draft/{draft_id}/save-as-version")
async def api_draft_save_as_version(
    draft_id: str,
    content: str = Form(...),
    version_name: str = Form(""),
):
    """Phase 54.6.312 — save the editor buffer as a NEW version row
    instead of overwriting the current draft.

    Creates a fresh ``drafts`` row keyed to the same
    ``(book_id, chapter_id, section_type)`` as the source draft, with
    ``version = max(version) + 1`` in that group, ``parent_draft_id``
    pointing back at the caller's id, and an optional ``version_name``
    persisted into ``custom_metadata.version_name`` so the Versions
    panel can surface the user-supplied label (e.g. "before Jane's
    review", "post-review polish").

    The new version is marked active so the reader immediately shows
    what the user just saved. The legacy ``/edit/{id}`` path (in-place
    overwrite) is still there for the autosave loop — this endpoint is
    the "Save as new version…" button.
    """
    with get_session() as session:
        src = session.execute(text("""
            SELECT book_id::text, chapter_id::text, section_type, title,
                   sources, topic, model_used, summary
            FROM drafts WHERE id::text = :did
        """), {"did": draft_id}).fetchone()
        if not src:
            raise HTTPException(404, "Draft not found")
        book_id, chapter_id, sec, title, sources, topic, model_used, summary = src

        mv = session.execute(text("""
            SELECT COALESCE(MAX(version), 0) FROM drafts
            WHERE book_id::text = :bid AND chapter_id::text = :cid
              AND section_type = :st
        """), {"bid": book_id, "cid": chapter_id, "st": sec}).fetchone()
        next_ver = int((mv and mv[0]) or 0) + 1

        meta = {}
        if version_name:
            meta["version_name"] = version_name[:120]
        meta["is_active"] = True

        wc = len((content or "").split())
        row = session.execute(text("""
            INSERT INTO drafts
              (book_id, chapter_id, section_type, topic, title, content, word_count,
               sources, model_used, version, summary, parent_draft_id,
               custom_metadata, status)
            VALUES
              (CAST(:bid AS uuid), CAST(:cid AS uuid), :st, :topic, :ttl,
               :content, :wc,
               COALESCE(CAST(:sources AS jsonb), '[]'::jsonb), :mu, :ver, :sum,
               CAST(:parent AS uuid), CAST(:meta AS jsonb), 'drafted')
            RETURNING id::text
        """), {
            "bid": book_id, "cid": chapter_id, "st": sec, "topic": topic,
            "ttl": title, "content": content, "wc": wc,
            "sources": json.dumps(sources if isinstance(sources, list) else []),
            "mu": model_used, "ver": next_ver, "sum": summary,
            "parent": draft_id, "meta": json.dumps(meta),
        }).fetchone()
        new_id = row[0]

        # Clear is_active on every other version in the same group.
        session.execute(text("""
            UPDATE drafts
               SET custom_metadata = custom_metadata - 'is_active'
             WHERE book_id::text = :bid AND chapter_id::text = :cid
               AND section_type = :st
               AND id::text <> :nid
        """), {"bid": book_id, "cid": chapter_id, "st": sec, "nid": new_id})
        session.commit()

    return JSONResponse({"ok": True, "new_draft_id": new_id,
                          "version": next_ver,
                          "version_name": version_name or None})


@router.post("/api/draft/{draft_id}/rename-version")
async def api_draft_rename_version(draft_id: str, name: str = Form("")):
    """Rename an existing version (edits ``custom_metadata.version_name``).
    Pass an empty string to clear the label."""
    with get_session() as session:
        if (name or "").strip():
            session.execute(text("""
                UPDATE drafts
                   SET custom_metadata = COALESCE(custom_metadata, '{}'::jsonb)
                                         || jsonb_build_object('version_name', CAST(:n AS text))
                 WHERE id::text = :did
            """), {"did": draft_id, "n": name[:120]})
        else:
            session.execute(text("""
                UPDATE drafts
                   SET custom_metadata = custom_metadata - 'version_name'
                 WHERE id::text = :did
            """), {"did": draft_id})
        session.commit()
    return JSONResponse({"ok": True})


@router.post("/api/draft/{draft_id}/version-description")
async def api_draft_version_description(
    draft_id: str, description: str = Form("")
):
    """Phase 54.6.314 — edit a version's short description. Stored in
    ``custom_metadata.version_description``. Pass an empty string to
    clear it. Capped at 500 chars (the field is meant for one-line
    notes like "pre-review polish" or "Jane's edits, round 2"; longer
    notes belong in the per-draft Comments surface)."""
    with get_session() as session:
        trimmed = (description or "").strip()[:500]
        if trimmed:
            session.execute(text("""
                UPDATE drafts
                   SET custom_metadata = COALESCE(custom_metadata, '{}'::jsonb)
                                         || jsonb_build_object('version_description',
                                                                 CAST(:d AS text))
                 WHERE id::text = :did
            """), {"did": draft_id, "d": trimmed})
        else:
            session.execute(text("""
                UPDATE drafts
                   SET custom_metadata = custom_metadata - 'version_description'
                 WHERE id::text = :did
            """), {"did": draft_id})
        session.commit()
    return JSONResponse({"ok": True})


@router.put("/api/draft/{draft_id}/status")
async def update_draft_status(draft_id: str, status: str = Form(...)):
    """Update a draft's status (to_do, drafted, reviewed, revised, final)."""
    valid = {"to_do", "drafted", "reviewed", "revised", "final"}
    if status not in valid:
        return JSONResponse({"error": f"Invalid status. Use: {', '.join(sorted(valid))}"}, status_code=400)
    with get_session() as session:
        session.execute(text(
            "UPDATE drafts SET status = :st WHERE id::text LIKE :q"
        ), {"st": status, "q": f"{draft_id}%"})
        session.commit()
    return JSONResponse({"ok": True})


@router.put("/api/draft/{draft_id}/metadata")
async def update_draft_metadata(request: Request, draft_id: str):
    """Merge custom metadata keys into a draft. Body: {"key": "value", ...}"""
    body = await request.json()
    with get_session() as session:
        session.execute(text("""
            UPDATE drafts SET custom_metadata = custom_metadata || CAST(:meta AS jsonb)
            WHERE id::text LIKE :q
        """), {"meta": json.dumps(body), "q": f"{draft_id}%"})
        session.commit()
    return JSONResponse({"ok": True})


@router.put("/api/draft/{draft_id}/chapter")
async def move_draft_to_chapter(draft_id: str, chapter_id: str = Form(...)):
    """Phase 33 — move a draft to a different chapter. Updates
    drafts.chapter_id. Used by cross-chapter section drag-and-drop."""
    with get_session() as session:
        session.execute(text(
            "UPDATE drafts SET chapter_id = CAST(:cid AS uuid) "
            "WHERE id::text = :did"
        ), {"did": draft_id, "cid": chapter_id})
        session.commit()
    return JSONResponse({"ok": True})


@router.delete("/api/draft/{draft_id}")
async def delete_draft(draft_id: str):
    """Phase 22 — permanently delete a single draft. Used by the GUI's
    inline X button on orphan drafts so users can clean up leftovers
    from before Phase 18 (when section_type was hardcoded). Comments
    and snapshots referencing the draft are dropped via ON DELETE
    CASCADE if configured, or left orphaned in the DB otherwise.
    """
    if not draft_id:
        raise HTTPException(400, "draft_id required")
    with get_session() as session:
        # Match by full id (canonical) OR prefix (matches the GUI's
        # short id convention used elsewhere). Returns the count so
        # the GUI knows whether anything actually happened.
        row = session.execute(text("""
            DELETE FROM drafts
            WHERE id::text = :did OR id::text LIKE :prefix
            RETURNING id::text
        """), {"did": draft_id, "prefix": f"{draft_id}%"}).fetchall()
        session.commit()
    return JSONResponse({"ok": True, "deleted": [r[0] for r in row]})


@router.get("/api/draft/{draft_id}/scores")
async def api_draft_scores(draft_id: str):
    """Return the persisted score history for an autowrite draft.

    Reads drafts.custom_metadata. Empty history is a valid response —
    drafts created by `book write` (not autowrite) won't have one, and the
    GUI shows an empty state for those.
    """
    with get_session() as session:
        row = session.execute(text("""
            SELECT id::text, title, version, word_count, model_used,
                   custom_metadata, created_at, section_type
            FROM drafts WHERE id::text LIKE :q
            ORDER BY created_at DESC LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()
    if not row:
        raise HTTPException(404, "Draft not found")
    meta = row[5] or {}
    return JSONResponse({
        "id": row[0],
        "title": row[1],
        "version": row[2],
        "word_count": row[3],
        "model_used": row[4],
        "section_type": row[7],
        "score_history": meta.get("score_history") or [],
        "feature_versions": meta.get("feature_versions") or {},
        "final_overall": meta.get("final_overall"),
        "max_iter": meta.get("max_iter"),
        "target_score": meta.get("target_score"),
    })
