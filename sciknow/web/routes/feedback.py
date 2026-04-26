"""``sciknow.web.routes.feedback`` — feedback endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. Behaviour
unchanged; the only edit is `@app.X(...)` → `@router.X(...)`.

Self-contained: only sqlalchemy + storage.db + json (the
sciknow.core.expand_feedback module already lazy-imports inside
each handler, so no app.py reference needed).
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

router = APIRouter()


@router.post("/api/feedback/thumbs")
async def api_feedback_thumbs(
    op: str = Form("ask"),
    score: int = Form(...),
    query: str = Form(""),
    preview: str = Form(""),
    comment: str = Form(""),
    draft_id: str = Form(""),
    chunk_ids: str = Form(""),
):
    """Record one thumbs-up / thumbs-down row.

    Originally called from a 👍/👎 button next to any generated answer
    in the reader (the UI was later removed). Fields are permissive —
    only `op` and `score` are required structurally; everything else is
    optional metadata the eventual LambdaMART trainer will project.

    Returns {id, created_at} so the client can render a confirmation
    toast or offer an undo via a follow-up PATCH."""
    if score not in (-1, 0, 1):
        return JSONResponse(
            {"error": "score must be -1, 0, or +1"}, status_code=400
        )
    chunks = [c.strip() for c in (chunk_ids or "").split(",") if c.strip()]
    did: str | None = None
    if draft_id:
        with get_session() as session:
            row = session.execute(text(
                "SELECT id::text FROM drafts WHERE id::text LIKE :q LIMIT 1"
            ), {"q": f"{draft_id.strip()}%"}).fetchone()
        if not row:
            return JSONResponse(
                {"error": f"no draft matches {draft_id!r}"}, status_code=404
            )
        did = row[0]
    with get_session() as session:
        row = session.execute(text("""
            INSERT INTO feedback (op, query, response_preview, score, comment,
                                  draft_id, chunk_ids, extras)
            VALUES (:op, :q, :preview, :score, :comment,
                    CAST(:did AS uuid), CAST(:chunks AS jsonb), '{}'::jsonb)
            RETURNING id::text, created_at
        """), {
            "op": op[:40] or "ask",
            "q": (query or "")[:4000] or None,
            "preview": (preview or "")[:1000] or None,
            "score": score,
            "comment": (comment or "")[:2000] or None,
            "did": did,
            "chunks": json.dumps(chunks),
        })
        session.commit()
        fb_id, created_at = row.fetchone()
    return {"id": fb_id, "created_at": created_at.isoformat()}


@router.get("/api/feedback/stats")
async def api_feedback_stats():
    """Counts by op × score for the sidebar badge."""
    with get_session() as session:
        rows = session.execute(text("""
            SELECT op, score, COUNT(*) FROM feedback GROUP BY op, score
        """)).fetchall()
    out: dict = {"total": 0, "by_op": {}}
    for op, score, n in rows:
        out["total"] += int(n)
        bucket = out["by_op"].setdefault(op, {"pos": 0, "zero": 0, "neg": 0})
        if score > 0:
            bucket["pos"] += int(n)
        elif score < 0:
            bucket["neg"] += int(n)
        else:
            bucket["zero"] += int(n)
    return out


@router.get("/api/feedback")
async def api_feedback_get():
    """Phase 54.6.115 (Tier 2 #3) — active project's expand feedback."""
    from sciknow.core.project import get_active_project
    from sciknow.core import expand_feedback as _fb
    project = get_active_project()
    fb = _fb.load(project.root)
    return JSONResponse({
        "slug": project.slug,
        "path": str(_fb.path_for(project.root)),
        "positive": [e.to_dict() for e in fb.positive],
        "negative": [e.to_dict() for e in fb.negative],
    })


@router.post("/api/feedback")
async def api_feedback_post(request: Request):
    """Mutate the active project's feedback.

    Body:
      {"action":"add","kind":"positive"|"negative","doi":"...","arxiv_id":"...","title":"...","topic":"..."}
      {"action":"remove","key":"...","kind":"positive"|"negative"|null}
    """
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    action = body.get("action")
    from sciknow.core.project import get_active_project
    from sciknow.core import expand_feedback as _fb
    project = get_active_project()
    if action == "add":
        kind = body.get("kind")
        if kind in ("pos", "+"): kind = "positive"
        if kind in ("neg", "-"): kind = "negative"
        if kind not in ("positive", "negative"):
            raise HTTPException(400, "kind must be 'positive' or 'negative'")
        doi = body.get("doi") or ""
        arxiv_id = body.get("arxiv_id") or ""
        title = body.get("title") or ""
        topic = body.get("topic") or ""
        if not (doi or arxiv_id or title):
            raise HTTPException(400, "need at least one of doi / arxiv_id / title")
        fb, added = _fb.add_entry(
            project.root, kind=kind, doi=doi, arxiv_id=arxiv_id,
            title=title, topic=topic,
        )
    elif action == "remove":
        key = (body.get("key") or "").strip()
        if not key:
            raise HTTPException(400, "key required")
        kind = body.get("kind")
        if kind in ("pos", "+"): kind = "positive"
        if kind in ("neg", "-"): kind = "negative"
        if kind not in (None, "positive", "negative"):
            raise HTTPException(400, "kind must be 'positive' or 'negative' if given")
        fb, _removed = _fb.remove_entry(project.root, key=key, kind=kind)
    else:
        raise HTTPException(400, "action must be 'add' or 'remove'")
    return JSONResponse({
        "slug": project.slug,
        "positive": [e.to_dict() for e in fb.positive],
        "negative": [e.to_dict() for e in fb.negative],
    })
