"""``sciknow.web.routes.comments`` — comments endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. Behaviour
unchanged. Cross-module deps resolved via the standard lazy
`from sciknow.web import app as _app` shim.
"""
from __future__ import annotations

import asyncio
import json
import threading

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

router = APIRouter()


@router.post("/comment")
async def add_comment(
    draft_id: str = Form(...),
    paragraph_index: int = Form(None),
    selected_text: str = Form(None),
    comment: str = Form(...),
):
    with get_session() as session:
        session.execute(text("""
            INSERT INTO draft_comments (draft_id, paragraph_index, selected_text, comment)
            VALUES (CAST(:did AS uuid), :para, :sel, :comment)
        """), {"did": draft_id, "para": paragraph_index, "sel": selected_text, "comment": comment})
        session.commit()
    return RedirectResponse(f"/section/{draft_id}", status_code=303)


@router.post("/comment/{comment_id}/resolve")
async def resolve_comment(comment_id: str):
    with get_session() as session:
        session.execute(text(
            "UPDATE draft_comments SET status = 'resolved' WHERE id::text = :cid"
        ), {"cid": comment_id})
        session.commit()
    return JSONResponse({"ok": True})


@router.post("/edit/{draft_id}")
async def edit_draft(draft_id: str, content: str = Form(...)):
    with get_session() as session:
        session.execute(text("""
            UPDATE drafts SET content = :content, word_count = :wc WHERE id::text = :did
        """), {"did": draft_id, "content": content, "wc": len(content.split())})
        session.commit()
    return RedirectResponse(f"/section/{draft_id}", status_code=303)


@router.get("/search", response_class=HTMLResponse)
async def search_book(q: str = ""):
    from sciknow.web import app as _app
    if not q:
        return RedirectResponse("/")
    book, chapters, drafts, gaps, comments = _app._get_book_data()
    matched = [d for d in drafts if q.lower() in (d[3] or "").lower()]
    return HTMLResponse(_app._render_book(book, chapters, drafts, gaps, comments, search_q=q, search_results=matched))
