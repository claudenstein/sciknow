"""
Book web reader — FastAPI app serving a live, interactive view of a sciknow book.

Launched via `sciknow book serve "Book Title"`. Features:
  - Sidebar chapter/section navigation (SPA — no full page reloads)
  - Live content from PostgreSQL (refreshes on fetch)
  - Action toolbar: Write / Review / Revise / Argue from the browser
  - SSE streaming of LLM output (tokens appear live)
  - Inline comments/annotations per paragraph
  - Edit-in-place with markdown editing
  - Search within the book
  - Dark/light theme toggle
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

logger = logging.getLogger("sciknow.web")

app = FastAPI(title="SciKnow Book Reader")


# Global state — set by the CLI before launching uvicorn
_book_id: str = ""
_book_title: str = ""


def set_book(book_id: str, book_title: str) -> None:
    global _book_id, _book_title
    _book_id = book_id
    _book_title = book_title


# ── Job management ───────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}     # job_id -> {queue, status, type, cancel}
_job_lock = threading.Lock()


def _create_job(job_type: str) -> tuple[str, asyncio.Queue]:
    job_id = uuid4().hex[:12]
    queue: asyncio.Queue = asyncio.Queue()
    with _job_lock:
        _jobs[job_id] = {
            "queue": queue,
            "status": "running",
            "type": job_type,
            "cancel": threading.Event(),
        }
    return job_id, queue


def _finish_job(job_id: str):
    with _job_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"


# ── Data helpers ─────────────────────────────────────────────────────────────

def _get_book_data():
    with get_session() as session:
        book = session.execute(text("""
            SELECT id::text, title, description, plan, status
            FROM books WHERE id::text = :bid
        """), {"bid": _book_id}).fetchone()

        chapters = session.execute(text("""
            SELECT bc.id::text, bc.number, bc.title, bc.description,
                   bc.topic_query, bc.topic_cluster
            FROM book_chapters bc
            WHERE bc.book_id = :bid ORDER BY bc.number
        """), {"bid": _book_id}).fetchall()

        drafts = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.content,
                   d.word_count, d.sources, d.version, d.summary,
                   d.review_feedback, d.chapter_id::text,
                   d.parent_draft_id::text, d.created_at,
                   bc.number AS ch_num, bc.title AS ch_title
            FROM drafts d
            LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.book_id = :bid
            ORDER BY bc.number, d.section_type, d.version DESC
        """), {"bid": _book_id}).fetchall()

        gaps = session.execute(text("""
            SELECT bg.id::text, bg.gap_type, bg.description, bg.status,
                   bc.number AS ch_num
            FROM book_gaps bg
            LEFT JOIN book_chapters bc ON bc.id = bg.chapter_id
            WHERE bg.book_id = :bid
            ORDER BY bc.number NULLS LAST, bg.gap_type
        """), {"bid": _book_id}).fetchall()

        comments = session.execute(text("""
            SELECT dc.id::text, dc.draft_id::text, dc.paragraph_index,
                   dc.selected_text, dc.comment, dc.status, dc.created_at
            FROM draft_comments dc
            JOIN drafts d ON d.id = dc.draft_id
            WHERE d.book_id = :bid
            ORDER BY dc.created_at
        """), {"bid": _book_id}).fetchall()

    return book, chapters, drafts, gaps, comments


def _md_to_html(text_content: str) -> str:
    """Simple markdown -> HTML conversion for draft content."""
    if not text_content:
        return ""
    html = text_content
    # Headers
    html = re.sub(r'^### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    # Bold and italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    # Citation references [N] -> styled spans
    html = re.sub(r'\[(\d+)\]', r'<span class="citation" data-ref="\1">[\1]</span>', html)
    # Paragraphs
    paragraphs = html.split('\n\n')
    html = ''.join(
        f'<p data-para="{i}">{p.strip()}</p>' if not p.strip().startswith('<h')
        else p.strip()
        for i, p in enumerate(paragraphs) if p.strip()
    )
    return html


# ── Page routes ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    book, chapters, drafts, gaps, comments = _get_book_data()
    if not book:
        return HTMLResponse("<h1>Book not found</h1>", status_code=404)
    return HTMLResponse(_render_book(book, chapters, drafts, gaps, comments))


@app.get("/section/{draft_id}", response_class=HTMLResponse)
async def section(draft_id: str):
    book, chapters, drafts, gaps, comments = _get_book_data()
    return HTMLResponse(_render_book(book, chapters, drafts, gaps, comments, focus_draft=draft_id))


@app.post("/comment")
async def add_comment(
    draft_id: str = Form(...),
    paragraph_index: int = Form(None),
    selected_text: str = Form(None),
    comment: str = Form(...),
):
    with get_session() as session:
        session.execute(text("""
            INSERT INTO draft_comments (draft_id, paragraph_index, selected_text, comment)
            VALUES (:did::uuid, :para, :sel, :comment)
        """), {"did": draft_id, "para": paragraph_index, "sel": selected_text, "comment": comment})
        session.commit()
    return RedirectResponse(f"/section/{draft_id}", status_code=303)


@app.post("/comment/{comment_id}/resolve")
async def resolve_comment(comment_id: str):
    with get_session() as session:
        session.execute(text(
            "UPDATE draft_comments SET status = 'resolved' WHERE id::text = :cid"
        ), {"cid": comment_id})
        session.commit()
    return JSONResponse({"ok": True})


@app.post("/edit/{draft_id}")
async def edit_draft(draft_id: str, content: str = Form(...)):
    with get_session() as session:
        session.execute(text("""
            UPDATE drafts SET content = :content, word_count = :wc WHERE id::text = :did
        """), {"did": draft_id, "content": content, "wc": len(content.split())})
        session.commit()
    return RedirectResponse(f"/section/{draft_id}", status_code=303)


@app.get("/search", response_class=HTMLResponse)
async def search_book(q: str = ""):
    if not q:
        return RedirectResponse("/")
    book, chapters, drafts, gaps, comments = _get_book_data()
    matched = [d for d in drafts if q.lower() in (d[3] or "").lower()]
    return HTMLResponse(_render_book(book, chapters, drafts, gaps, comments, search_q=q, search_results=matched))


# ── JSON API ─────────────────────────────────────────────────────────────────

@app.get("/api/book")
async def api_book():
    book, chapters, drafts, gaps, comments = _get_book_data()
    return {
        "title": book[1] if book else "",
        "chapters": len(chapters),
        "drafts": len(drafts),
        "gaps": len(gaps),
        "comments": len(comments),
    }


@app.get("/api/section/{draft_id}")
async def api_section(draft_id: str):
    """Return section data as JSON for SPA navigation."""
    book, chapters, drafts, gaps, comments = _get_book_data()

    # Build draft map and find the target
    draft_map = {}
    chapter_drafts = {}
    for d in drafts:
        draft_map[d[0]] = d
        key = d[9] or "__none__"
        if key not in chapter_drafts:
            chapter_drafts[key] = []
        existing = [x for x in chapter_drafts[key] if x[2] == d[2]]
        if not existing or (d[6] or 1) > (existing[0][6] or 1):
            chapter_drafts[key] = [x for x in chapter_drafts[key] if x[2] != d[2]]
            chapter_drafts[key].append(d)

    active = draft_map.get(draft_id)
    if not active:
        raise HTTPException(404, "Draft not found")

    # Group comments for this draft
    active_comments = [c for c in comments if c[1] == draft_id]
    sources = json.loads(active[5]) if isinstance(active[5], str) else (active[5] or [])

    return {
        "id": active[0],
        "title": active[1],
        "section_type": active[2],
        "content_html": _md_to_html(active[3] or ""),
        "content_raw": active[3] or "",
        "word_count": active[4] or 0,
        "version": active[6] or 1,
        "review_feedback": active[8] or "",
        "review_html": _md_to_html(active[8]) if active[8] else "<em>No review yet.</em>",
        "sources_html": _render_sources(sources),
        "sources": sources,
        "comments_html": _render_comments(active_comments),
        "chapter_id": active[9],
        "chapter_num": active[12],
        "chapter_title": active[13],
    }


@app.get("/api/chapters")
async def api_chapters():
    """Return chapter list with their sections for sidebar building."""
    book, chapters, drafts, gaps, comments = _get_book_data()

    chapter_drafts = {}
    for d in drafts:
        key = d[9] or "__none__"
        if key not in chapter_drafts:
            chapter_drafts[key] = []
        existing = [x for x in chapter_drafts[key] if x[2] == d[2]]
        if not existing or (d[6] or 1) > (existing[0][6] or 1):
            chapter_drafts[key] = [x for x in chapter_drafts[key] if x[2] != d[2]]
            chapter_drafts[key].append(d)

    result = []
    for ch in chapters:
        ch_id, ch_num, ch_title, ch_desc, tq, tc = ch
        ch_ds = chapter_drafts.get(ch_id, [])
        section_order = {"introduction": 0, "methods": 1, "results": 2, "discussion": 3, "conclusion": 4}
        sections = []
        for d in sorted(ch_ds, key=lambda x: section_order.get(x[2] or "", 9)):
            sections.append({
                "id": d[0], "type": d[2] or "text",
                "version": d[6] or 1, "words": d[4] or 0,
            })
        result.append({
            "id": ch_id, "num": ch_num, "title": ch_title,
            "description": ch_desc, "topic_query": tq,
            "sections": sections,
        })

    return {"chapters": result, "gaps_count": len([g for g in gaps if g[3] == "open"])}


@app.get("/api/dashboard")
async def api_dashboard():
    """Return dashboard data: completion heatmap, stats, gaps."""
    book, chapters, drafts, gaps, comments = _get_book_data()

    section_types = ["introduction", "methods", "results", "discussion", "conclusion"]

    # Build draft lookup: chapter_id -> {section_type: {version, words, id, has_review}}
    ch_sections: dict[str, dict] = {}
    total_words = 0
    for d in drafts:
        draft_id, title, sec_type, content, wc, sources, version, summary, \
            review_fb, ch_id, parent_id, created, ch_num, ch_title = d
        total_words += wc or 0
        if not ch_id:
            continue
        if ch_id not in ch_sections:
            ch_sections[ch_id] = {}
        # Keep only the latest version per section_type
        existing = ch_sections[ch_id].get(sec_type)
        if not existing or (version or 1) > existing["version"]:
            ch_sections[ch_id][sec_type] = {
                "id": draft_id, "version": version or 1, "words": wc or 0,
                "has_review": bool(review_fb),
            }

    heatmap = []
    for ch in chapters:
        ch_id, ch_num, ch_title, ch_desc, tq, tc = ch
        row = {"num": ch_num, "title": ch_title, "id": ch_id, "cells": []}
        secs = ch_sections.get(ch_id, {})
        for st in section_types:
            info = secs.get(st)
            if info:
                status = "reviewed" if info["has_review"] else "drafted"
                row["cells"].append({"type": st, "status": status, "draft_id": info["id"],
                                     "version": info["version"], "words": info["words"]})
            else:
                row["cells"].append({"type": st, "status": "empty"})
        heatmap.append(row)

    open_gaps = [{"id": g[0], "type": g[1], "description": g[2], "status": g[3],
                  "chapter_num": g[4]} for g in gaps if g[3] == "open"]

    return {
        "heatmap": heatmap,
        "section_types": section_types,
        "stats": {
            "total_words": total_words,
            "chapters": len(chapters),
            "drafts": len(drafts),
            "gaps_open": len(open_gaps),
            "comments": len(comments),
        },
        "gaps": open_gaps,
    }


@app.get("/api/versions/{draft_id}")
async def api_versions(draft_id: str):
    """Return the version chain for a draft (all versions of the same section)."""
    with get_session() as session:
        # Find the draft to get its chapter_id and section_type
        draft = session.execute(text("""
            SELECT d.chapter_id::text, d.section_type, d.book_id::text
            FROM drafts d WHERE d.id::text LIKE :q LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()
        if not draft:
            raise HTTPException(404, "Draft not found")

        ch_id, sec_type, book_id = draft

        # Get all drafts for this chapter + section_type (all versions)
        rows = session.execute(text("""
            SELECT d.id::text, d.version, d.word_count, d.created_at,
                   d.parent_draft_id::text, d.review_feedback IS NOT NULL as has_review
            FROM drafts d
            WHERE d.chapter_id::text = :cid AND d.section_type = :st AND d.book_id::text = :bid
            ORDER BY d.version ASC
        """), {"cid": ch_id, "st": sec_type, "bid": book_id}).fetchall()

    return {"versions": [
        {"id": r[0], "version": r[1], "word_count": r[2] or 0,
         "created_at": str(r[3]) if r[3] else "", "parent_id": r[4],
         "has_review": bool(r[5])}
        for r in rows
    ]}


@app.get("/api/diff/{old_id}/{new_id}")
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


# ── Chapter management ───────────────────────────────────────────────────────

@app.post("/api/chapters")
async def create_chapter(
    title: str = Form(...),
    description: str = Form(""),
    topic_query: str = Form(""),
    number: int = Form(None),
):
    """Add a chapter to the book."""
    with get_session() as session:
        if number is None:
            max_n = session.execute(text(
                "SELECT COALESCE(MAX(number), 0) FROM book_chapters WHERE book_id = :bid"
            ), {"bid": _book_id}).scalar()
            number = max_n + 1

        session.execute(text("""
            INSERT INTO book_chapters (book_id, number, title, description, topic_query)
            VALUES (:bid, :num, :title, :desc, :tq)
        """), {"bid": _book_id, "num": number, "title": title,
               "desc": description or None, "tq": topic_query or None})
        session.commit()

    return JSONResponse({"ok": True, "number": number})


@app.put("/api/chapters/{chapter_id}")
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


@app.delete("/api/chapters/{chapter_id}")
async def delete_chapter(chapter_id: str):
    """Delete a chapter (drafts are preserved but unlinked)."""
    with get_session() as session:
        session.execute(text(
            "UPDATE drafts SET chapter_id = NULL WHERE chapter_id::text = :cid"
        ), {"cid": chapter_id})
        session.execute(text(
            "DELETE FROM book_chapters WHERE id::text = :cid"
        ), {"cid": chapter_id})
        session.commit()

    return JSONResponse({"ok": True})


@app.post("/api/chapters/reorder")
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


# ── SSE / Streaming endpoints ───────────────────────────────────────────────

def _run_generator_in_thread(job_id: str, generator_fn, loop):
    """Run a blocking generator in a thread, pushing events to the job queue."""
    queue = _jobs[job_id]["queue"]
    cancel = _jobs[job_id]["cancel"]
    try:
        for event in generator_fn():
            if cancel.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "cancelled"})
                break
            loop.call_soon_threadsafe(queue.put_nowait, event)
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(exc)})
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel
        _finish_job(job_id)


@app.post("/api/write")
async def api_write(
    chapter_id: str = Form(...),
    section_type: str = Form("introduction"),
    model: str = Form(None),
):
    """Start a write operation, returns job_id for SSE streaming."""
    from sciknow.core.book_ops import write_section_stream

    job_id, queue = _create_job("write")
    loop = asyncio.get_event_loop()

    def gen():
        return write_section_stream(
            book_id=_book_id, chapter_id=chapter_id,
            section_type=section_type, model=model or None,
        )

    thread = threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@app.post("/api/review/{draft_id}")
async def api_review(draft_id: str, model: str = Form(None)):
    from sciknow.core.book_ops import review_draft_stream

    job_id, queue = _create_job("review")
    loop = asyncio.get_event_loop()

    def gen():
        return review_draft_stream(draft_id, model=model or None)

    thread = threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@app.post("/api/revise/{draft_id}")
async def api_revise(
    draft_id: str,
    instruction: str = Form(""),
    model: str = Form(None),
):
    from sciknow.core.book_ops import revise_draft_stream

    job_id, queue = _create_job("revise")
    loop = asyncio.get_event_loop()

    def gen():
        return revise_draft_stream(
            draft_id, instruction=instruction, model=model or None)

    thread = threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@app.post("/api/gaps")
async def api_gaps(model: str = Form(None)):
    from sciknow.core.book_ops import run_gaps_stream

    job_id, queue = _create_job("gaps")
    loop = asyncio.get_event_loop()

    def gen():
        return run_gaps_stream(book_id=_book_id, model=model or None)

    thread = threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@app.post("/api/argue")
async def api_argue(
    claim: str = Form(...),
    model: str = Form(None),
):
    from sciknow.core.book_ops import run_argue_stream

    job_id, queue = _create_job("argue")
    loop = asyncio.get_event_loop()

    def gen():
        return run_argue_stream(claim, book_id=_book_id, model=model or None, save=True)

    thread = threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@app.post("/api/verify/{draft_id}")
async def api_verify(draft_id: str, model: str = Form(None)):
    """Run claim verification on a draft via SSE."""
    from sciknow.core.book_ops import review_draft_stream

    # We reuse the review infrastructure but with a verify-specific generator
    job_id, queue = _create_job("verify")
    loop = asyncio.get_event_loop()

    def gen():
        """Verify claims — uses the verify_claims prompt from book_ops."""
        from sciknow.rag import prompts as rag_prompts
        from sciknow.rag.llm import complete as llm_complete
        from sciknow.storage.db import get_session
        from sciknow.storage.qdrant import get_client
        from sciknow.core.book_ops import _retrieve, _clean_json
        import json as _json

        with get_session() as session:
            from sqlalchemy import text as sql_text
            row = session.execute(sql_text("""
                SELECT d.id::text, d.title, d.section_type, d.topic, d.content
                FROM drafts d WHERE d.id::text LIKE :q LIMIT 1
            """), {"q": f"{draft_id}%"}).fetchone()

        if not row:
            yield {"type": "error", "message": f"Draft not found: {draft_id}"}
            return

        d_id, d_title, d_section, d_topic, d_content = row

        yield {"type": "progress", "stage": "retrieval", "detail": "Retrieving source passages..."}

        qdrant = get_client()
        search_query = f"{d_section or ''} {d_topic or d_title}"
        with get_session() as session:
            results, _ = _retrieve(session, qdrant, search_query, context_k=12)

        yield {"type": "progress", "stage": "verifying", "detail": "Verifying claims..."}

        sys_v, usr_v = rag_prompts.verify_claims(d_content, results)
        try:
            raw = llm_complete(sys_v, usr_v, model=model or None, temperature=0.0, num_ctx=16384)
            vdata = _json.loads(_clean_json(raw), strict=False)
            yield {"type": "verification", "data": vdata}
        except Exception as exc:
            yield {"type": "error", "message": f"Verification failed: {exc}"}

        yield {"type": "completed", "draft_id": d_id}

    thread = threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@app.post("/api/autowrite")
async def api_autowrite(
    chapter_id: str = Form(None),
    section_type: str = Form("introduction"),
    max_iter: int = Form(3),
    target_score: float = Form(0.85),
    full: bool = Form(False),
    model: str = Form(None),
):
    from sciknow.core.book_ops import autowrite_section_stream

    if full:
        # For full book autowrite, we'd need to chain multiple generators.
        # For now, require chapter_id for single-section autowrite from web.
        if not chapter_id:
            return JSONResponse({"error": "chapter_id required (full-book autowrite not yet supported from web)"}, status_code=400)

    job_id, queue = _create_job("autowrite")
    loop = asyncio.get_event_loop()

    def gen():
        return autowrite_section_stream(
            book_id=_book_id, chapter_id=chapter_id,
            section_type=section_type, model=model or None,
            max_iter=max_iter, target_score=target_score,
        )

    thread = threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@app.get("/api/stream/{job_id}")
async def stream_job(job_id: str):
    """SSE endpoint — streams events from a running job."""
    with _job_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    queue = job["queue"]

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=300)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'timeout'})}\n\n"
                break
            if event is None:
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    with _job_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job["cancel"].set()
    return JSONResponse({"ok": True})


@app.get("/api/jobs")
async def list_jobs():
    with _job_lock:
        return [
            {"id": jid, "type": j["type"], "status": j["status"]}
            for jid, j in _jobs.items()
        ]


# ── HTML rendering helpers ───────────────────────────────────────────────────

def _render_book(book, chapters, drafts, gaps, comments,
                 focus_draft=None, search_q="", search_results=None):
    """Render the full book reader as a self-contained HTML page."""

    chapter_drafts = {}
    draft_map = {}
    for d in drafts:
        draft_id, title, sec_type, content, wc, sources, version, summary, \
            review_fb, ch_id, parent_id, created, ch_num, ch_title = d
        draft_map[draft_id] = d
        key = ch_id or "__none__"
        if key not in chapter_drafts:
            chapter_drafts[key] = []
        existing = [x for x in chapter_drafts[key] if x[2] == sec_type]
        if not existing or (version or 1) > (existing[0][6] or 1):
            chapter_drafts[key] = [x for x in chapter_drafts[key] if x[2] != sec_type]
            chapter_drafts[key].append(d)

    draft_comments = {}
    for c in comments:
        cid, did, para, sel, comm, status, created = c
        if did not in draft_comments:
            draft_comments[did] = []
        draft_comments[did].append(c)

    sidebar_items = []
    for ch in chapters:
        ch_id, ch_num, ch_title, ch_desc, tq, tc = ch
        ch_ds = chapter_drafts.get(ch_id, [])
        sections = []
        for d in sorted(ch_ds, key=lambda x: {"introduction": 0, "methods": 1, "results": 2, "discussion": 3, "conclusion": 4}.get(x[2] or "", 9)):
            sections.append({
                "id": d[0], "type": d[2] or "text", "version": d[6] or 1,
                "words": d[4] or 0,
            })
        sidebar_items.append({
            "num": ch_num, "title": ch_title, "id": ch_id,
            "description": ch_desc or "", "topic_query": tq or "",
            "sections": sections,
        })

    active_draft = None
    if focus_draft:
        active_draft = draft_map.get(focus_draft)
    elif drafts:
        active_draft = drafts[0]

    active_html = ""
    active_comments = []
    active_sources = []
    active_review = ""
    active_id = ""
    active_title = ""
    active_raw = ""
    if active_draft:
        active_id = active_draft[0]
        active_title = active_draft[1]
        active_html = _md_to_html(active_draft[3] or "")
        active_raw = active_draft[3] or ""
        active_sources = json.loads(active_draft[5]) if isinstance(active_draft[5], str) else (active_draft[5] or [])
        active_review = active_draft[8] or ""
        active_comments = draft_comments.get(active_id, [])

    open_gaps = [g for g in gaps if g[3] == "open"]

    # Build chapters_json for the JS-side SPA
    chapters_json = json.dumps([{
        "id": si["id"], "num": si["num"], "title": si["title"],
        "description": si["description"], "topic_query": si["topic_query"],
        "sections": si["sections"],
    } for si in sidebar_items])

    return TEMPLATE.format(
        book_title=book[1] if book else "Untitled",
        book_id=_book_id,
        book_plan=(book[3] or "No plan set.") if book else "",
        sidebar_html=_render_sidebar(sidebar_items, active_id),
        content_html=active_html,
        content_raw=_escape_for_js(active_raw),
        active_id=active_id,
        active_title=active_title,
        active_version=active_draft[6] if active_draft else 1,
        active_words=active_draft[4] if active_draft else 0,
        active_chapter_id=active_draft[9] if active_draft else "",
        active_section_type=active_draft[2] if active_draft else "",
        sources_html=_render_sources(active_sources),
        review_html=_md_to_html(active_review) if active_review else "<em>No review yet.</em>",
        comments_html=_render_comments(active_comments),
        gaps_count=len(open_gaps),
        search_q=search_q,
        search_results_html=_render_search(search_results) if search_results else "",
        chapters_json=chapters_json,
    )


def _escape_for_js(s: str) -> str:
    """Escape a string for embedding in a JS template literal."""
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def _render_sidebar(items, active_id):
    html = ""
    for ch in items:
        html += f'<div class="ch-group" data-ch-id="{ch["id"]}">'
        html += (f'<div class="ch-title">Ch.{ch["num"]}: {ch["title"]}'
                 f'<span class="ch-actions">'
                 f'<button onclick="deleteChapter(\'{ch["id"]}\')" title="Delete chapter">\u2717</button>'
                 f'</span></div>')
        for sec in ch["sections"]:
            active = "active" if sec["id"] == active_id else ""
            html += (
                f'<a class="sec-link {active}" href="/section/{sec["id"]}" '
                f'data-draft-id="{sec["id"]}" onclick="return navTo(this)">'
                f'{sec["type"].capitalize()} '
                f'<span class="meta">v{sec["version"]} \u00b7 {sec["words"]}w</span></a>'
            )
        if not ch["sections"]:
            html += '<div class="sec-link empty">No drafts yet</div>'
        html += '</div>'
    return html


def _render_sources(sources):
    if not sources:
        return "<em>No sources.</em>"
    html = "<ol>"
    for i, s in enumerate(sources):
        if s:
            html += f'<li id="source-{i+1}">{s}</li>'
    html += "</ol>"
    return html


def _render_comments(comments):
    if not comments:
        return ""
    html = ""
    for c in comments:
        cid, did, para, sel, comm, status, created = c
        cls = "resolved" if status == "resolved" else "open"
        sel_html = f'<div class="sel-text">"{sel[:100]}"</div>' if sel else ""
        para_html = f'<span class="para-ref">P{para}</span> ' if para is not None else ""
        resolve_btn = (
            f'<button class="resolve-btn" onclick="resolveComment(\'{cid}\')">Resolve</button>'
            if status == "open" else '<span class="resolved-tag">Resolved</span>'
        )
        html += f'<div class="comment {cls}">{para_html}{sel_html}<div class="comm-text">{comm}</div>{resolve_btn}</div>'
    return html


def _render_search(results):
    if not results:
        return "<p>No results.</p>"
    html = ""
    for d in results[:20]:
        html += f'<a href="/section/{d[0]}" class="search-result" data-draft-id="{d[0]}" onclick="return navTo(this)"><strong>{d[1]}</strong> ({d[4] or 0} words)</a>'
    return html


# ── HTML Template ────────────────────────────────────────────────────────────

TEMPLATE = """\
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{book_title} — SciKnow Reader</title>
<style>
:root {{ --bg: #fff; --fg: #1a1a1a; --sidebar-bg: #f5f5f5; --border: #e0e0e0;
         --accent: #2563eb; --accent-light: #dbeafe; --success: #16a34a;
         --warning: #d97706; --danger: #dc2626; --code-bg: #f8f8f8;
         --toolbar-bg: #f0f4ff; }}
[data-theme="dark"] {{ --bg: #1a1a2e; --fg: #e0e0e0; --sidebar-bg: #16213e;
         --border: #333; --accent: #60a5fa; --accent-light: #1e3a5f;
         --code-bg: #0f3460; --toolbar-bg: #1e2a4a; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Georgia', serif; color: var(--fg); background: var(--bg);
        display: flex; height: 100vh; }}
/* Sidebar */
.sidebar {{ width: 280px; background: var(--sidebar-bg); border-right: 1px solid var(--border);
            overflow-y: auto; flex-shrink: 0; padding: 16px 0; }}
.sidebar h2 {{ padding: 8px 16px; font-size: 15px; color: var(--accent); }}
.ch-group {{ margin-bottom: 8px; }}
.ch-title {{ padding: 6px 16px; font-weight: bold; font-size: 13px; color: var(--fg); opacity: 0.7; }}
.sec-link {{ display: block; padding: 4px 16px 4px 32px; text-decoration: none;
             color: var(--fg); font-size: 13px; border-left: 3px solid transparent; cursor: pointer; }}
.sec-link:hover {{ background: var(--accent-light); }}
.sec-link.active {{ border-left-color: var(--accent); background: var(--accent-light); font-weight: bold; }}
.sec-link .meta {{ font-size: 11px; opacity: 0.5; }}
.sec-link.empty {{ color: var(--fg); opacity: 0.3; font-style: italic; }}
/* Main */
.main {{ flex: 1; overflow-y: auto; padding: 32px 48px; max-width: 900px; }}
.main h1 {{ font-size: 26px; margin-bottom: 4px; }}
.main .subtitle {{ font-size: 13px; color: var(--fg); opacity: 0.5; margin-bottom: 8px; }}
.main p {{ line-height: 1.8; margin-bottom: 12px; text-align: justify; }}
.main h2,.main h3,.main h4 {{ margin: 24px 0 12px; }}
.citation {{ color: var(--accent); cursor: pointer; font-weight: bold; }}
/* Action toolbar */
.toolbar {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 20px; padding: 10px 14px;
            background: var(--toolbar-bg); border-radius: 8px; border: 1px solid var(--border); }}
.toolbar button {{ font-size: 12px; padding: 5px 14px; border: 1px solid var(--border);
                   border-radius: 6px; cursor: pointer; background: var(--bg); color: var(--fg);
                   font-family: -apple-system, sans-serif; transition: all .15s; }}
.toolbar button:hover {{ background: var(--accent); color: white; border-color: var(--accent); }}
.toolbar button.active {{ background: var(--accent); color: white; }}
.toolbar .sep {{ width: 1px; background: var(--border); margin: 0 4px; }}
/* Streaming panel */
.stream-panel {{ display: none; margin-bottom: 20px; border: 1px solid var(--border);
                  border-radius: 8px; overflow: hidden; }}
.stream-header {{ padding: 8px 14px; background: var(--toolbar-bg); font-size: 12px;
                   font-family: -apple-system, sans-serif; display: flex;
                   justify-content: space-between; align-items: center; }}
.stream-header .status {{ opacity: 0.6; }}
.stream-body {{ padding: 16px; max-height: 500px; overflow-y: auto; font-size: 14px;
                 line-height: 1.7; white-space: pre-wrap; }}
.stream-scores {{ padding: 8px 14px; background: var(--toolbar-bg); font-size: 12px;
                   font-family: -apple-system, sans-serif; display: none; }}
.score-bar {{ display: inline-block; margin-right: 12px; }}
.score-bar .label {{ opacity: 0.6; }}
.score-bar .value {{ font-weight: bold; }}
.score-bar .value.good {{ color: var(--success); }}
.score-bar .value.mid {{ color: var(--warning); }}
.score-bar .value.low {{ color: var(--danger); }}
.stop-btn {{ font-size: 11px; padding: 3px 10px; background: var(--danger); color: white;
             border: none; border-radius: 4px; cursor: pointer; }}
/* Right panel */
.panel {{ width: 320px; border-left: 1px solid var(--border); overflow-y: auto;
          padding: 16px; font-size: 13px; background: var(--sidebar-bg); }}
.panel h3 {{ font-size: 14px; margin: 16px 0 8px; color: var(--accent); }}
.panel ol {{ padding-left: 20px; }} .panel li {{ margin-bottom: 4px; font-size: 12px; }}
/* Comments */
.comment {{ padding: 8px; margin: 4px 0; border-left: 3px solid var(--accent);
            background: var(--accent-light); border-radius: 4px; }}
.comment.resolved {{ opacity: 0.5; border-left-color: var(--success); }}
.sel-text {{ font-style: italic; font-size: 12px; opacity: 0.6; margin-bottom: 4px; }}
.para-ref {{ font-size: 11px; background: var(--accent); color: white; padding: 1px 6px;
             border-radius: 8px; }}
.comm-text {{ margin: 4px 0; }}
.resolve-btn {{ font-size: 11px; background: var(--success); color: white; border: none;
                padding: 2px 8px; border-radius: 4px; cursor: pointer; }}
.resolved-tag {{ font-size: 11px; color: var(--success); }}
/* Comment form */
.comment-form {{ margin-top: 12px; }}
.comment-form textarea {{ width: 100%; padding: 6px; font-size: 12px; border: 1px solid var(--border);
                          border-radius: 4px; resize: vertical; min-height: 60px; background: var(--bg);
                          color: var(--fg); }}
.comment-form button {{ margin-top: 4px; padding: 4px 12px; background: var(--accent); color: white;
                        border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }}
/* Search */
.search-bar {{ padding: 8px 16px; }}
.search-bar input {{ width: 100%; padding: 6px 10px; border: 1px solid var(--border);
                     border-radius: 4px; font-size: 13px; background: var(--bg); color: var(--fg); }}
.search-result {{ display: block; padding: 6px 16px; text-decoration: none; color: var(--fg);
                  border-bottom: 1px solid var(--border); }}
.search-result:hover {{ background: var(--accent-light); }}
/* Theme toggle */
.theme-toggle {{ position: fixed; bottom: 16px; right: 16px; background: var(--accent);
                 color: white; border: none; padding: 8px 12px; border-radius: 20px;
                 cursor: pointer; font-size: 12px; z-index: 100; }}
/* Edit */
.edit-btn {{ background: var(--accent); color: white; border: none; padding: 4px 12px;
             border-radius: 4px; cursor: pointer; font-size: 12px; margin-left: 8px; }}
.edit-area {{ width: 100%; min-height: 400px; padding: 12px; font-family: 'Courier New', monospace;
              font-size: 14px; border: 1px solid var(--border); border-radius: 4px;
              background: var(--bg); color: var(--fg); }}
/* Job indicator */
.job-indicator {{ display: none; padding: 4px 16px; font-size: 11px; color: var(--accent);
                   font-family: -apple-system, sans-serif; animation: pulse 1.5s infinite; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}
/* Dashboard */
.dashboard {{ font-family: -apple-system, sans-serif; }}
.dashboard h2 {{ font-size: 22px; margin-bottom: 16px; }}
.dash-stats {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
.stat-card {{ padding: 14px 20px; border-radius: 8px; border: 1px solid var(--border);
              background: var(--toolbar-bg); min-width: 120px; }}
.stat-card .num {{ font-size: 28px; font-weight: bold; color: var(--accent); }}
.stat-card .lbl {{ font-size: 12px; opacity: 0.6; margin-top: 2px; }}
.heatmap {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; font-size: 13px; }}
.heatmap th {{ padding: 6px 10px; text-align: center; font-weight: 600; opacity: 0.6;
               border-bottom: 2px solid var(--border); }}
.heatmap td {{ padding: 6px 10px; text-align: center; border: 1px solid var(--border); }}
.heatmap .ch-label {{ text-align: left; font-weight: 600; white-space: nowrap; }}
.hm-cell {{ border-radius: 4px; padding: 4px 8px; cursor: pointer; display: inline-block;
            min-width: 44px; font-size: 11px; }}
.hm-cell.reviewed {{ background: var(--success); color: white; }}
.hm-cell.drafted {{ background: var(--warning); color: white; }}
.hm-cell.empty {{ background: var(--border); opacity: 0.5; }}
.hm-cell:hover {{ opacity: 0.85; }}
/* Gaps in dashboard */
.gap-list {{ margin-bottom: 20px; }}
.gap-item {{ display: flex; align-items: center; gap: 8px; padding: 8px 12px;
             border-left: 3px solid var(--warning); background: var(--toolbar-bg);
             border-radius: 4px; margin-bottom: 6px; font-size: 13px; }}
.gap-item .gap-type {{ font-size: 11px; font-weight: bold; text-transform: uppercase;
                       padding: 2px 6px; border-radius: 4px; background: var(--warning);
                       color: white; flex-shrink: 0; }}
.gap-item .gap-desc {{ flex: 1; }}
.gap-item button {{ font-size: 11px; padding: 3px 10px; border: 1px solid var(--border);
                    border-radius: 4px; cursor: pointer; background: var(--bg); color: var(--fg);
                    white-space: nowrap; }}
.gap-item button:hover {{ background: var(--accent); color: white; }}
/* Version history */
.version-panel {{ display: none; margin-bottom: 20px; border: 1px solid var(--border);
                   border-radius: 8px; overflow: hidden; }}
.version-header {{ padding: 8px 14px; background: var(--toolbar-bg); font-size: 12px;
                    font-family: -apple-system, sans-serif; display: flex;
                    justify-content: space-between; align-items: center; }}
.version-timeline {{ display: flex; gap: 6px; padding: 10px 14px; flex-wrap: wrap;
                      font-family: -apple-system, sans-serif; }}
.version-badge {{ padding: 4px 12px; border-radius: 6px; font-size: 12px; cursor: pointer;
                  border: 1px solid var(--border); background: var(--bg); }}
.version-badge:hover {{ background: var(--accent-light); }}
.version-badge.selected {{ background: var(--accent); color: white; border-color: var(--accent); }}
.diff-view {{ padding: 16px; max-height: 500px; overflow-y: auto; font-size: 14px; line-height: 1.7; }}
.diff-del {{ background: #fdd; color: #900; text-decoration: line-through; padding: 1px 2px; }}
.diff-ins {{ background: #dfd; color: #060; padding: 1px 2px; }}
[data-theme="dark"] .diff-del {{ background: #4a1c1c; color: #fbb; }}
[data-theme="dark"] .diff-ins {{ background: #1c4a1c; color: #bfb; }}
/* Chapter management */
.ch-add-form {{ padding: 8px 16px; margin-top: 8px; }}
.ch-add-form input {{ width: 100%; padding: 5px 8px; border: 1px solid var(--border);
                      border-radius: 4px; font-size: 12px; background: var(--bg); color: var(--fg);
                      margin-bottom: 4px; }}
.ch-add-form button {{ padding: 3px 10px; font-size: 11px; background: var(--accent); color: white;
                       border: none; border-radius: 4px; cursor: pointer; }}
.ch-title {{ cursor: default; }}
.ch-title .ch-actions {{ display: none; float: right; }}
.ch-group:hover .ch-actions {{ display: inline; }}
.ch-actions button {{ font-size: 10px; padding: 1px 6px; border: none; background: none;
                      color: var(--fg); cursor: pointer; opacity: 0.4; }}
.ch-actions button:hover {{ opacity: 1; }}
/* Enhanced editor */
.editor-split {{ display: flex; gap: 12px; }}
.editor-split .editor-src {{ flex: 1; }}
.editor-split .editor-preview {{ flex: 1; border: 1px solid var(--border); border-radius: 4px;
                                  padding: 12px; overflow-y: auto; max-height: 600px;
                                  font-size: 14px; line-height: 1.7; }}
.editor-toolbar {{ display: flex; gap: 4px; margin-bottom: 6px; }}
.editor-toolbar button {{ font-size: 11px; padding: 3px 8px; border: 1px solid var(--border);
                          border-radius: 4px; cursor: pointer; background: var(--bg);
                          color: var(--fg); font-family: monospace; }}
.editor-toolbar button:hover {{ background: var(--accent-light); }}
.editor-toolbar .autosave {{ font-size: 10px; opacity: 0.5; margin-left: auto; line-height: 24px; }}
/* Citation popover */
.citation {{ position: relative; }}
.citation-popover {{ display: none; position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%);
                     background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
                     padding: 10px 14px; font-size: 12px; width: 320px; z-index: 50;
                     box-shadow: 0 4px 12px rgba(0,0,0,0.15); font-weight: normal;
                     line-height: 1.5; text-align: left; pointer-events: none; }}
.citation:hover .citation-popover {{ display: block; }}
.citation-popover .cp-title {{ font-weight: 600; margin-bottom: 4px; color: var(--accent); }}
.citation-popover .cp-authors {{ opacity: 0.7; font-size: 11px; }}
.citation-popover .cp-meta {{ font-size: 11px; opacity: 0.6; margin-top: 4px; }}
/* Verification indicators */
.citation.verified-supported {{ background: #d1fae5; border-radius: 3px; }}
.citation.verified-extrapolated {{ background: #fef3c7; border-radius: 3px; }}
.citation.verified-misrepresented {{ background: #fee2e2; border-radius: 3px; }}
[data-theme="dark"] .citation.verified-supported {{ background: #064e3b; }}
[data-theme="dark"] .citation.verified-extrapolated {{ background: #78350f; }}
[data-theme="dark"] .citation.verified-misrepresented {{ background: #7f1d1d; }}
/* Autowrite chart */
.aw-dashboard {{ margin-bottom: 16px; }}
.aw-chart {{ border: 1px solid var(--border); border-radius: 6px; padding: 8px;
             background: var(--toolbar-bg); margin-bottom: 8px; }}
.aw-chart svg {{ width: 100%; height: 120px; }}
.aw-chart .chart-line {{ fill: none; stroke: var(--accent); stroke-width: 2; }}
.aw-chart .chart-target {{ stroke: var(--success); stroke-width: 1; stroke-dasharray: 4; }}
.aw-chart .chart-dot {{ fill: var(--accent); }}
.aw-log {{ font-size: 12px; font-family: -apple-system, sans-serif; max-height: 200px;
           overflow-y: auto; padding: 8px 12px; background: var(--toolbar-bg);
           border-radius: 6px; border: 1px solid var(--border); }}
.aw-log .log-keep {{ color: var(--success); }}
.aw-log .log-discard {{ color: var(--danger); }}
/* Argument map */
.argue-map {{ border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
              margin-bottom: 16px; }}
.argue-map svg {{ width: 100%; min-height: 300px; background: var(--toolbar-bg); }}
.argue-map .node {{ cursor: pointer; }}
.argue-map .node text {{ font-size: 11px; fill: var(--fg); }}
.argue-map .link-supports {{ stroke: var(--success); stroke-width: 1.5; }}
.argue-map .link-contradicts {{ stroke: var(--danger); stroke-width: 1.5; }}
.argue-map .link-neutral {{ stroke: var(--border); stroke-width: 1; }}
</style>
</head>
<body>

<!-- Sidebar -->
<nav class="sidebar">
  <h2>{book_title}</h2>
  <div class="search-bar">
    <form action="/search" method="get">
      <input type="text" name="q" placeholder="Search..." value="{search_q}">
    </form>
  </div>
  {search_results_html}
  <div id="sidebar-sections">
    {sidebar_html}
  </div>
  <div class="ch-add-form" id="ch-add-form" style="display:none;">
    <input type="text" id="ch-add-title" placeholder="New chapter title...">
    <button onclick="addChapter()">Add Chapter</button>
    <button onclick="document.getElementById('ch-add-form').style.display='none'" style="background:var(--danger);">Cancel</button>
  </div>
  <div style="padding: 4px 16px;">
    <button onclick="document.getElementById('ch-add-form').style.display='block'"
            style="font-size:11px;padding:2px 8px;border:1px solid var(--border);border-radius:4px;cursor:pointer;background:var(--bg);color:var(--fg);">+ Add Chapter</button>
  </div>
  <div class="job-indicator" id="job-indicator">Working...</div>
  <div style="padding: 8px 16px; font-size: 12px; opacity: 0.5; margin-top: 16px; cursor:pointer;"
       onclick="showDashboard()">
    <span id="gaps-count">{gaps_count}</span> open gaps — click for dashboard
  </div>
</nav>

<!-- Main content -->
<main class="main" id="content">
  <h1 id="draft-title">{active_title}</h1>
  <div class="subtitle" id="draft-subtitle">
    Version <span id="draft-version">{active_version}</span> ·
    <span id="draft-words">{active_words}</span> words
    <button class="edit-btn" onclick="toggleEdit()">Edit</button>
  </div>

  <!-- Action Toolbar -->
  <div class="toolbar" id="toolbar">
    <button onclick="doWrite()" title="Draft this section from scratch">Write</button>
    <button onclick="doReview()" title="Run a critic pass on this section">Review</button>
    <button onclick="doRevise()" title="Revise based on review feedback">Revise</button>
    <div class="sep"></div>
    <button onclick="doAutowrite()" title="Autonomous write-review-revise loop">Autowrite</button>
    <div class="sep"></div>
    <button onclick="promptArgue()" title="Map evidence for/against a claim">Argue</button>
    <button onclick="doGaps()" title="Analyse gaps in the book">Gaps</button>
    <div class="sep"></div>
    <button onclick="doVerify()" title="Verify citations against sources">Verify</button>
    <div class="sep"></div>
    <button onclick="showVersions()" title="View version history and diffs">History</button>
    <button onclick="showDashboard()" title="Book dashboard with completion heatmap">Dashboard</button>
  </div>

  <!-- Version history panel -->
  <div class="version-panel" id="version-panel">
    <div class="version-header">
      <span>Version History</span>
      <button class="stop-btn" onclick="document.getElementById('version-panel').style.display='none'"
              style="background:var(--border);color:var(--fg);">Close</button>
    </div>
    <div class="version-timeline" id="version-timeline"></div>
    <div class="diff-view" id="diff-view"></div>
  </div>

  <!-- Dashboard (hidden by default, shown via JS) -->
  <div id="dashboard-view" style="display:none;"></div>

  <!-- Streaming output panel -->
  <div class="stream-panel" id="stream-panel">
    <div class="stream-header">
      <span class="status" id="stream-status">Starting...</span>
      <button class="stop-btn" id="stream-stop" onclick="stopJob()">Stop</button>
    </div>
    <div class="stream-scores" id="stream-scores"></div>
    <div class="stream-body" id="stream-body"></div>
  </div>

  <div id="read-view">{content_html}</div>

  <div id="edit-view" style="display:none;">
    <div class="editor-toolbar">
      <button onclick="edInsert('**','**')" title="Bold"><b>B</b></button>
      <button onclick="edInsert('*','*')" title="Italic"><i>I</i></button>
      <button onclick="edInsert('## ','')" title="Heading">H2</button>
      <button onclick="edInsert('### ','')" title="Subheading">H3</button>
      <button onclick="edInsertCite()" title="Citation">[N]</button>
      <span class="autosave" id="autosave-status"></span>
    </div>
    <div class="editor-split">
      <div class="editor-src">
        <textarea class="edit-area" id="edit-area" oninput="edPreview()"></textarea>
      </div>
      <div class="editor-preview" id="edit-preview"></div>
    </div>
    <div style="margin-top:8px;">
      <button class="edit-btn" onclick="edSave()">Save</button>
      <button class="edit-btn" style="background:var(--danger);" onclick="toggleEdit()">Cancel</button>
    </div>
  </div>

  <!-- Argument map container -->
  <div id="argue-map-view" style="display:none;"></div>
</main>

<!-- Right panel -->
<aside class="panel">
  <h3>Sources</h3>
  <div id="panel-sources">{sources_html}</div>

  <h3>Review Feedback</h3>
  <div id="panel-review" style="font-size:12px;">{review_html}</div>

  <h3>Comments</h3>
  <div id="panel-comments">{comments_html}</div>
  <form class="comment-form" action="/comment" method="post" id="comment-form">
    <input type="hidden" name="draft_id" value="{active_id}" id="comment-draft-id">
    <textarea name="comment" placeholder="Add a comment..."></textarea>
    <button type="submit">Add Comment</button>
  </form>
</aside>

<button class="theme-toggle" onclick="toggleTheme()">Toggle Theme</button>

<script>
// ── State ─────────────────────────────────────────────────────────────────
let currentDraftId = '{active_id}';
let currentChapterId = '{active_chapter_id}';
let currentSectionType = '{active_section_type}';
let currentJobId = null;
let currentEventSource = null;
const chaptersData = {chapters_json};

// ── Theme ─────────────────────────────────────────────────────────────────
function toggleTheme() {{
  const html = document.documentElement;
  html.dataset.theme = html.dataset.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('theme', html.dataset.theme);
}}
if (localStorage.getItem('theme')) {{
  document.documentElement.dataset.theme = localStorage.getItem('theme');
}}

// ── SPA Navigation ────────────────────────────────────────────────────────
function navTo(el) {{
  const draftId = el.dataset.draftId;
  if (!draftId) return true;  // fallback to normal navigation
  loadSection(draftId);
  return false;  // prevent default <a> navigation
}}

async function loadSection(draftId) {{
  try {{
    const res = await fetch('/api/section/' + draftId);
    if (!res.ok) return;
    const data = await res.json();

    currentDraftId = data.id;
    currentChapterId = data.chapter_id || '';
    currentSectionType = data.section_type || '';

    // Update main content
    document.getElementById('draft-title').textContent = data.title;
    document.getElementById('draft-version').textContent = data.version;
    document.getElementById('draft-words').textContent = data.word_count;
    document.getElementById('read-view').innerHTML = data.content_html;
    document.getElementById('edit-view').style.display = 'none';
    document.getElementById('read-view').style.display = 'block';
    document.getElementById('edit-view').action = '/edit/' + data.id;

    // Update right panel
    document.getElementById('panel-sources').innerHTML = data.sources_html;
    document.getElementById('panel-review').innerHTML = data.review_html;
    document.getElementById('panel-comments').innerHTML = data.comments_html;
    document.getElementById('comment-draft-id').value = data.id;

    // Update sidebar active state
    document.querySelectorAll('.sec-link').forEach(l => l.classList.remove('active'));
    const link = document.querySelector('[data-draft-id="' + draftId + '"]');
    if (link) link.classList.add('active');

    // Update URL without reload
    history.pushState({{draftId: draftId}}, '', '/section/' + draftId);

    // Hide stream panel
    document.getElementById('stream-panel').style.display = 'none';
  }} catch(e) {{
    console.error('Navigation failed:', e);
  }}
}}

// Handle browser back/forward
window.addEventListener('popstate', function(e) {{
  if (e.state && e.state.draftId) loadSection(e.state.draftId);
}});

// ── Edit toggle ───────────────────────────────────────────────────────────
function toggleEdit() {{
  const rv = document.getElementById('read-view');
  const ev = document.getElementById('edit-view');
  const ta = document.getElementById('edit-area');
  if (ev.style.display === 'none') {{
    ta.value = `{content_raw}`;
    rv.style.display = 'none';
    ev.style.display = 'block';
  }} else {{
    rv.style.display = 'block';
    ev.style.display = 'none';
  }}
}}

// ── Comments ──────────────────────────────────────────────────────────────
function resolveComment(cid) {{
  fetch('/comment/' + cid + '/resolve', {{method: 'POST'}})
    .then(() => loadSection(currentDraftId));
}}

// ── Streaming helpers ─────────────────────────────────────────────────────
function showStreamPanel(label) {{
  const panel = document.getElementById('stream-panel');
  const body = document.getElementById('stream-body');
  const status = document.getElementById('stream-status');
  const scores = document.getElementById('stream-scores');
  panel.style.display = 'block';
  body.innerHTML = '';
  scores.style.display = 'none';
  scores.innerHTML = '';
  status.textContent = label;
  document.getElementById('job-indicator').style.display = 'block';
  body.scrollTop = 0;
}}

function hideStreamPanel() {{
  document.getElementById('job-indicator').style.display = 'none';
}}

function startStream(jobId) {{
  currentJobId = jobId;
  if (currentEventSource) currentEventSource.close();

  const source = new EventSource('/api/stream/' + jobId);
  currentEventSource = source;
  const body = document.getElementById('stream-body');
  const status = document.getElementById('stream-status');
  const scoresEl = document.getElementById('stream-scores');

  source.onmessage = function(e) {{
    const evt = JSON.parse(e.data);

    if (evt.type === 'token') {{
      body.innerHTML += evt.text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
      body.scrollTop = body.scrollHeight;
    }}
    else if (evt.type === 'progress') {{
      status.textContent = evt.detail || evt.stage;
    }}
    else if (evt.type === 'scores') {{
      scoresEl.style.display = 'block';
      const s = evt.scores;
      const dims = ['groundedness', 'completeness', 'coherence', 'citation_accuracy', 'overall'];
      scoresEl.innerHTML = 'Iteration ' + evt.iteration + ': ' + dims.map(d => {{
        const v = (s[d] || 0).toFixed(2);
        const cls = v >= 0.85 ? 'good' : v >= 0.7 ? 'mid' : 'low';
        return '<span class="score-bar"><span class="label">' + d.slice(0,5) + '</span> ' +
               '<span class="value ' + cls + '">' + v + '</span></span>';
      }}).join('');
    }}
    else if (evt.type === 'revision_verdict') {{
      const icon = evt.action === 'KEEP' ? '\\u2713' : '\\u2717';
      const color = evt.action === 'KEEP' ? 'var(--success)' : 'var(--danger)';
      body.innerHTML += '<div style="color:' + color + ';font-weight:bold;margin:8px 0;">' +
        icon + ' ' + evt.action + ': ' + evt.old_score.toFixed(2) + ' \\u2192 ' + evt.new_score.toFixed(2) +
        '</div>';
    }}
    else if (evt.type === 'converged') {{
      status.textContent = 'Converged at iteration ' + evt.iteration +
        ' (score: ' + evt.final_score.toFixed(2) + ')';
    }}
    else if (evt.type === 'iteration_start') {{
      body.innerHTML += '<div style="opacity:0.5;margin:12px 0;border-top:1px solid var(--border);padding-top:8px;">' +
        'Iteration ' + evt.iteration + '/' + evt.max + '</div>';
    }}
    else if (evt.type === 'completed') {{
      status.textContent = 'Done';
      hideStreamPanel();
      source.close();
      currentEventSource = null;
      currentJobId = null;
      // Refresh sidebar and current section
      refreshAfterJob(evt.draft_id);
    }}
    else if (evt.type === 'error') {{
      status.textContent = 'Error: ' + evt.message;
      body.innerHTML += '<div style="color:var(--danger);margin:8px 0;">' + evt.message + '</div>';
      hideStreamPanel();
      source.close();
      currentEventSource = null;
      currentJobId = null;
    }}
    else if (evt.type === 'done') {{
      hideStreamPanel();
      source.close();
      currentEventSource = null;
      currentJobId = null;
    }}
  }};

  source.onerror = function() {{
    status.textContent = 'Connection lost';
    hideStreamPanel();
    source.close();
    currentEventSource = null;
    currentJobId = null;
  }};
}}

function stopJob() {{
  if (currentJobId) {{
    fetch('/api/jobs/' + currentJobId, {{method: 'DELETE'}});
    document.getElementById('stream-status').textContent = 'Stopping...';
  }}
}}

async function refreshAfterJob(newDraftId) {{
  // Reload sidebar data
  try {{
    const res = await fetch('/api/chapters');
    const data = await res.json();
    rebuildSidebar(data.chapters, newDraftId || currentDraftId);
    document.getElementById('gaps-count').textContent = data.gaps_count;
  }} catch(e) {{}}
  // Navigate to the new draft if one was created
  if (newDraftId) loadSection(newDraftId);
}}

function rebuildSidebar(chapters, activeId) {{
  const container = document.getElementById('sidebar-sections');
  let html = '';
  chapters.forEach(ch => {{
    html += '<div class="ch-group" data-ch-id="' + ch.id + '">';
    html += '<div class="ch-title">Ch.' + ch.num + ': ' + ch.title +
      '<span class="ch-actions"><button onclick="deleteChapter(\\\'' + ch.id + '\\\')" title="Delete chapter">\\u2717</button></span></div>';
    if (ch.sections.length === 0) {{
      html += '<div class="sec-link empty">No drafts yet</div>';
    }} else {{
      ch.sections.forEach(sec => {{
        const active = sec.id === activeId ? 'active' : '';
        html += '<a class="sec-link ' + active + '" href="/section/' + sec.id +
          '" data-draft-id="' + sec.id + '" onclick="return navTo(this)">' +
          sec.type.charAt(0).toUpperCase() + sec.type.slice(1) +
          ' <span class="meta">v' + sec.version + ' \\u00b7 ' + sec.words + 'w</span></a>';
      }});
    }}
    html += '</div>';
  }});
  container.innerHTML = html;
}}

// ── Action handlers ───────────────────────────────────────────────────────
async function doWrite() {{
  if (!currentChapterId) {{ alert('No chapter selected.'); return; }}
  const section = currentSectionType || 'introduction';
  showStreamPanel('Writing ' + section + '...');

  const fd = new FormData();
  fd.append('chapter_id', currentChapterId);
  fd.append('section_type', section);
  const res = await fetch('/api/write', {{method: 'POST', body: fd}});
  const data = await res.json();
  startStream(data.job_id);
}}

async function doReview() {{
  if (!currentDraftId) {{ alert('No draft selected.'); return; }}
  showStreamPanel('Reviewing...');

  const fd = new FormData();
  const res = await fetch('/api/review/' + currentDraftId, {{method: 'POST', body: fd}});
  const data = await res.json();
  startStream(data.job_id);
}}

async function doRevise() {{
  if (!currentDraftId) {{ alert('No draft selected.'); return; }}
  const instruction = prompt('Revision instruction (leave empty to use review feedback):');
  if (instruction === null) return;  // cancelled
  showStreamPanel('Revising...');

  const fd = new FormData();
  fd.append('instruction', instruction);
  const res = await fetch('/api/revise/' + currentDraftId, {{method: 'POST', body: fd}});
  const data = await res.json();
  startStream(data.job_id);
}}

async function doAutowrite() {{
  if (!currentChapterId) {{ alert('No chapter selected.'); return; }}
  const section = currentSectionType || 'introduction';
  const maxIter = prompt('Max iterations (default 3):', '3');
  if (maxIter === null) return;
  showStreamPanel('Autowriting ' + section + '...');

  const fd = new FormData();
  fd.append('chapter_id', currentChapterId);
  fd.append('section_type', section);
  fd.append('max_iter', maxIter || '3');
  fd.append('target_score', '0.85');
  const res = await fetch('/api/autowrite', {{method: 'POST', body: fd}});
  const data = await res.json();
  startStream(data.job_id);
}}

async function promptArgue() {{
  const claim = prompt('Enter a claim to map evidence for/against:');
  if (!claim) return;
  showStreamPanel('Mapping argument...');

  const fd = new FormData();
  fd.append('claim', claim);
  const res = await fetch('/api/argue', {{method: 'POST', body: fd}});
  const data = await res.json();
  startStream(data.job_id);
}}

async function doGaps() {{
  showStreamPanel('Analysing gaps...');
  const fd = new FormData();
  const res = await fetch('/api/gaps', {{method: 'POST', body: fd}});
  const data = await res.json();
  startStream(data.job_id);
}}

// ── Dashboard ─────────────────────────────────────────────────────────────
async function showDashboard() {{
  const res = await fetch('/api/dashboard');
  const data = await res.json();
  const s = data.stats;

  let html = '<div class="dashboard">';
  html += '<h2>Book Dashboard</h2>';

  // Stats cards
  html += '<div class="dash-stats">';
  html += '<div class="stat-card"><div class="num">' + s.total_words.toLocaleString() + '</div><div class="lbl">Words</div></div>';
  html += '<div class="stat-card"><div class="num">' + s.chapters + '</div><div class="lbl">Chapters</div></div>';
  html += '<div class="stat-card"><div class="num">' + s.drafts + '</div><div class="lbl">Drafts</div></div>';
  html += '<div class="stat-card"><div class="num">' + s.gaps_open + '</div><div class="lbl">Open Gaps</div></div>';
  html += '<div class="stat-card"><div class="num">' + s.comments + '</div><div class="lbl">Comments</div></div>';
  html += '</div>';

  // Heatmap
  html += '<h3 style="margin-bottom:8px;">Completion Heatmap</h3>';
  html += '<table class="heatmap"><thead><tr><th></th>';
  data.section_types.forEach(st => {{
    html += '<th>' + st.charAt(0).toUpperCase() + st.slice(1,5) + '</th>';
  }});
  html += '</tr></thead><tbody>';
  data.heatmap.forEach(row => {{
    html += '<tr><td class="ch-label">Ch.' + row.num + ' ' + row.title.substring(0,25) + '</td>';
    row.cells.forEach(cell => {{
      if (cell.status === 'empty') {{
        html += '<td><span class="hm-cell empty" onclick="writeForCell(\'' + row.id + '\',\'' + cell.type + '\')" title="Click to write">—</span></td>';
      }} else {{
        const label = 'v' + cell.version + ' ' + cell.words + 'w';
        html += '<td><span class="hm-cell ' + cell.status + '" onclick="loadSection(\'' + cell.draft_id + '\')" title="' + label + '">' + label + '</span></td>';
      }}
    }});
    html += '</tr>';
  }});
  html += '</tbody></table>';

  // Gaps
  if (data.gaps.length > 0) {{
    html += '<h3 style="margin-bottom:8px;">Open Gaps</h3><div class="gap-list">';
    data.gaps.forEach(g => {{
      let btn = '';
      if (g.type === 'draft' && g.chapter_num) {{
        btn = '<button onclick="writeForGap(' + g.chapter_num + ')">Write</button>';
      }} else if (g.type === 'evidence') {{
        btn = '<button onclick="alert(\'Run: sciknow db expand -q \\x22' + g.description.substring(0,30).replace(/'/g, '') + '\\x22\')">Expand</button>';
      }}
      html += '<div class="gap-item">';
      html += '<span class="gap-type">' + g.type + '</span>';
      html += '<span class="gap-desc">' + (g.chapter_num ? 'Ch.' + g.chapter_num + ': ' : '') + g.description.substring(0, 120) + '</span>';
      html += btn + '</div>';
    }});
    html += '</div>';
  }}

  html += '</div>';

  // Show dashboard, hide section view
  document.getElementById('dashboard-view').innerHTML = html;
  document.getElementById('dashboard-view').style.display = 'block';
  document.getElementById('read-view').style.display = 'none';
  document.getElementById('edit-view').style.display = 'none';
  document.getElementById('draft-title').textContent = 'Dashboard';
  document.getElementById('draft-subtitle').style.display = 'none';
  document.getElementById('toolbar').style.display = 'none';
  document.getElementById('stream-panel').style.display = 'none';
  document.getElementById('version-panel').style.display = 'none';

  // Clear sidebar active
  document.querySelectorAll('.sec-link').forEach(l => l.classList.remove('active'));
  history.pushState({{dashboard: true}}, '', '/');
}}

function writeForCell(chapterId, sectionType) {{
  currentChapterId = chapterId;
  currentSectionType = sectionType;
  // Hide dashboard, show section view
  document.getElementById('dashboard-view').style.display = 'none';
  document.getElementById('read-view').style.display = 'block';
  document.getElementById('read-view').innerHTML = '<p style="opacity:0.5;">Starting write...</p>';
  document.getElementById('draft-subtitle').style.display = 'block';
  document.getElementById('toolbar').style.display = 'flex';
  doWrite();
}}

function writeForGap(chapterNum) {{
  // Find the chapter ID from chaptersData
  const ch = chaptersData.find(c => c.num === chapterNum);
  if (ch) writeForCell(ch.id, 'introduction');
}}

// Override loadSection to restore section view from dashboard
const _origLoadSection = loadSection;
loadSection = async function(draftId) {{
  document.getElementById('dashboard-view').style.display = 'none';
  document.getElementById('read-view').style.display = 'block';
  document.getElementById('draft-subtitle').style.display = 'block';
  document.getElementById('toolbar').style.display = 'flex';
  await _origLoadSection(draftId);
}};

// ── Version History ───────────────────────────────────────────────────────
let versionData = [];
let selectedVersions = [];

async function showVersions() {{
  if (!currentDraftId) {{ alert('No draft selected.'); return; }}
  const res = await fetch('/api/versions/' + currentDraftId);
  const data = await res.json();
  versionData = data.versions;

  if (versionData.length < 1) {{ alert('No version history.'); return; }}

  const panel = document.getElementById('version-panel');
  const timeline = document.getElementById('version-timeline');
  const diffView = document.getElementById('diff-view');
  panel.style.display = 'block';
  diffView.innerHTML = '<p style="opacity:0.5;">Select two versions to compare.</p>';
  selectedVersions = [];

  let html = '';
  versionData.forEach(v => {{
    const review = v.has_review ? ' \\u2713' : '';
    html += '<span class="version-badge" data-vid="' + v.id + '" onclick="selectVersion(\'' + v.id + '\')">' +
      'v' + v.version + ' (' + v.word_count + 'w)' + review + '</span>';
  }});
  timeline.innerHTML = html;
}}

async function selectVersion(vid) {{
  // Toggle selection (max 2)
  const idx = selectedVersions.indexOf(vid);
  if (idx >= 0) {{
    selectedVersions.splice(idx, 1);
  }} else {{
    if (selectedVersions.length >= 2) selectedVersions.shift();
    selectedVersions.push(vid);
  }}

  // Update badges
  document.querySelectorAll('.version-badge').forEach(b => {{
    b.classList.toggle('selected', selectedVersions.includes(b.dataset.vid));
  }});

  if (selectedVersions.length === 2) {{
    // Show diff
    const diffView = document.getElementById('diff-view');
    diffView.innerHTML = '<p style="opacity:0.5;">Loading diff...</p>';
    const res = await fetch('/api/diff/' + selectedVersions[0] + '/' + selectedVersions[1]);
    const data = await res.json();
    diffView.innerHTML = data.diff_html;
  }} else if (selectedVersions.length === 1) {{
    // Navigate to that version
    loadSection(selectedVersions[0]);
  }}
}}

// ── Chapter Management ────────────────────────────────────────────────────
async function addChapter() {{
  const title = document.getElementById('ch-add-title').value.trim();
  if (!title) return;

  const fd = new FormData();
  fd.append('title', title);
  await fetch('/api/chapters', {{method: 'POST', body: fd}});
  document.getElementById('ch-add-title').value = '';
  document.getElementById('ch-add-form').style.display = 'none';
  await refreshAfterJob(null);
}}

async function deleteChapter(chapterId) {{
  if (!confirm('Delete this chapter? Drafts will be preserved but unlinked.')) return;
  await fetch('/api/chapters/' + chapterId, {{method: 'DELETE'}});
  await refreshAfterJob(null);
}}

// ── Enhanced Editor (Phase 3a) ────────────────────────────────────────
let _autosaveTimer = null;
let _currentRaw = '';

function toggleEdit() {{
  const rv = document.getElementById('read-view');
  const ev = document.getElementById('edit-view');
  const ta = document.getElementById('edit-area');
  if (ev.style.display === 'none') {{
    // Load current raw content via API
    fetch('/api/section/' + currentDraftId).then(r => r.json()).then(data => {{
      _currentRaw = data.content_raw;
      ta.value = _currentRaw;
      edPreview();
    }});
    rv.style.display = 'none';
    ev.style.display = 'block';
    // Start autosave
    _autosaveTimer = setInterval(edAutosave, 5000);
  }} else {{
    rv.style.display = 'block';
    ev.style.display = 'none';
    if (_autosaveTimer) {{ clearInterval(_autosaveTimer); _autosaveTimer = null; }}
  }}
}}

function edPreview() {{
  const ta = document.getElementById('edit-area');
  const preview = document.getElementById('edit-preview');
  let md = ta.value;
  // Simple markdown → HTML for preview
  md = md.replace(/^### (.+)$/gm, '<h4>$1</h4>');
  md = md.replace(/^## (.+)$/gm, '<h3>$1</h3>');
  md = md.replace(/^# (.+)$/gm, '<h2>$1</h2>');
  md = md.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  md = md.replace(/\*(.+?)\*/g, '<em>$1</em>');
  md = md.replace(/\[(\d+)\]/g, '<span class="citation">[$1]</span>');
  const paras = md.split('\\n\\n');
  preview.innerHTML = paras.map(p => {{
    p = p.trim();
    if (!p) return '';
    if (p.startsWith('<h')) return p;
    return '<p>' + p + '</p>';
  }}).join('');
}}

function edInsert(before, after) {{
  const ta = document.getElementById('edit-area');
  const start = ta.selectionStart;
  const end = ta.selectionEnd;
  const sel = ta.value.substring(start, end) || 'text';
  ta.value = ta.value.substring(0, start) + before + sel + after + ta.value.substring(end);
  ta.selectionStart = start + before.length;
  ta.selectionEnd = start + before.length + sel.length;
  ta.focus();
  edPreview();
}}

function edInsertCite() {{
  const n = prompt('Citation number:');
  if (n) edInsert('[' + n + ']', '');
}}

async function edSave() {{
  const ta = document.getElementById('edit-area');
  const fd = new FormData();
  fd.append('content', ta.value);
  await fetch('/edit/' + currentDraftId, {{method: 'POST', body: fd}});
  document.getElementById('autosave-status').textContent = 'Saved';
  if (_autosaveTimer) {{ clearInterval(_autosaveTimer); _autosaveTimer = null; }}
  toggleEdit();
  loadSection(currentDraftId);
}}

async function edAutosave() {{
  const ta = document.getElementById('edit-area');
  if (ta.value === _currentRaw) return;
  _currentRaw = ta.value;
  const fd = new FormData();
  fd.append('content', ta.value);
  await fetch('/edit/' + currentDraftId, {{method: 'POST', body: fd}});
  document.getElementById('autosave-status').textContent = 'Auto-saved';
  setTimeout(() => {{ document.getElementById('autosave-status').textContent = ''; }}, 2000);
}}

// ── Claim Verification (Phase 5b) ────────────────────────────────────
async function doVerify() {{
  if (!currentDraftId) {{ alert('No draft selected.'); return; }}
  showStreamPanel('Verifying citations...');

  const fd = new FormData();
  const res = await fetch('/api/verify/' + currentDraftId, {{method: 'POST', body: fd}});
  const data = await res.json();

  // Override stream handler to process verification data
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();

  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;
  const body = document.getElementById('stream-body');
  const status = document.getElementById('stream-status');

  source.onmessage = function(e) {{
    const evt = JSON.parse(e.data);
    if (evt.type === 'progress') {{
      status.textContent = evt.detail || evt.stage;
    }}
    else if (evt.type === 'verification') {{
      const vd = evt.data;
      status.textContent = 'Groundedness: ' + (vd.groundedness_score || '?');

      // Show results in stream body
      let html = '<div style="font-family:-apple-system,sans-serif;">';
      html += '<div style="font-size:18px;font-weight:bold;margin-bottom:12px;">Groundedness Score: ' +
        '<span style="color:' + (vd.groundedness_score >= 0.8 ? 'var(--success)' : vd.groundedness_score >= 0.6 ? 'var(--warning)' : 'var(--danger)') + '">' +
        (vd.groundedness_score || '?') + '</span></div>';

      if (vd.claims) {{
        vd.claims.forEach(c => {{
          const color = c.verdict === 'SUPPORTED' ? 'var(--success)' : c.verdict === 'EXTRAPOLATED' ? 'var(--warning)' : 'var(--danger)';
          html += '<div style="margin:6px 0;padding:6px 10px;border-left:3px solid ' + color + ';background:var(--toolbar-bg);border-radius:4px;">';
          html += '<span style="font-weight:bold;color:' + color + ';">' + c.verdict + '</span> ';
          html += '<span style="font-size:12px;">' + c.citation + '</span><br>';
          html += '<span style="font-size:12px;opacity:0.7;">' + (c.text || '').substring(0, 120) + '</span>';
          if (c.reason) html += '<br><span style="font-size:11px;opacity:0.5;">' + c.reason + '</span>';
          html += '</div>';
        }});
      }}

      if (vd.unsupported_claims && vd.unsupported_claims.length > 0) {{
        html += '<div style="margin-top:12px;font-weight:bold;color:var(--danger);">Unsupported Claims:</div>';
        vd.unsupported_claims.forEach(u => {{
          html += '<div style="font-size:12px;color:var(--danger);margin:4px 0;">- ' + u.substring(0, 120) + '</div>';
        }});
      }}
      html += '</div>';
      body.innerHTML = html;

      // Apply color indicators to citations in the read view
      applyVerificationColors(vd);
    }}
    else if (evt.type === 'completed') {{
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }}
    else if (evt.type === 'error') {{
      status.textContent = 'Error: ' + evt.message;
      body.innerHTML = '<div style="color:var(--danger);">' + evt.message + '</div>';
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }}
    else if (evt.type === 'done') {{
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }}
  }};
}}

function applyVerificationColors(vd) {{
  if (!vd.claims) return;
  const classMap = {{ 'SUPPORTED': 'verified-supported', 'EXTRAPOLATED': 'verified-extrapolated', 'MISREPRESENTED': 'verified-misrepresented' }};
  vd.claims.forEach(c => {{
    const ref = c.citation ? c.citation.replace(/[\[\]]/g, '') : '';
    if (!ref) return;
    document.querySelectorAll('.citation[data-ref="' + ref + '"]').forEach(el => {{
      el.className = 'citation ' + (classMap[c.verdict] || '');
    }});
  }});
}}

// ── Citation Popovers (Phase 5a) ─────────────────────────────────────
function buildPopovers() {{
  // Extract source data from the sources panel
  const sourceItems = document.querySelectorAll('#panel-sources li');
  const sourceData = {{}};
  sourceItems.forEach((li, i) => {{
    sourceData[i + 1] = li.textContent;
  }});

  document.querySelectorAll('.citation').forEach(el => {{
    const ref = el.dataset.ref;
    if (!ref || el.querySelector('.citation-popover')) return;
    const src = sourceData[parseInt(ref)];
    if (!src) return;

    const popover = document.createElement('div');
    popover.className = 'citation-popover';

    // Parse the APA-style citation: [N] Author (year). Title. Journal. doi:...
    const parts = src.replace(/^\[\d+\]\s*/, '').split('. ');
    const titlePart = parts.length > 1 ? parts[1] || '' : parts[0] || '';
    const authorYear = parts[0] || '';
    const rest = parts.slice(2).join('. ');

    popover.innerHTML = '<div class="cp-title">' + titlePart + '</div>' +
      '<div class="cp-authors">' + authorYear + '</div>' +
      (rest ? '<div class="cp-meta">' + rest + '</div>' : '');

    el.appendChild(popover);
    el.style.cursor = 'pointer';
    el.onclick = function() {{
      const target = document.getElementById('source-' + ref);
      if (target) target.scrollIntoView({{behavior: 'smooth', block: 'center'}});
    }};
  }});
}}

// Build popovers after page load and after SPA navigation
document.addEventListener('DOMContentLoaded', buildPopovers);
const _origLoadSection2 = loadSection;
loadSection = async function(draftId) {{
  await _origLoadSection2(draftId);
  setTimeout(buildPopovers, 100);
}};

// ── Autowrite Dashboard (Phase 4) ────────────────────────────────────
// Enhanced autowrite that shows convergence chart
let awScores = [];
let awTargetScore = 0.85;

async function doAutowrite() {{
  if (!currentChapterId) {{ alert('No chapter selected.'); return; }}
  const section = currentSectionType || 'introduction';
  const maxIter = prompt('Max iterations (default 3):', '3');
  if (maxIter === null) return;
  const targetStr = prompt('Target score (default 0.85):', '0.85');
  if (targetStr === null) return;
  awTargetScore = parseFloat(targetStr) || 0.85;
  awScores = [];

  showStreamPanel('Autowriting ' + section + '...');

  // Add chart + log to the stream panel
  const body = document.getElementById('stream-body');
  body.innerHTML = '<div class="aw-dashboard">' +
    '<div class="aw-chart" id="aw-chart"><svg viewBox="0 0 400 120"></svg></div>' +
    '<div class="aw-log" id="aw-log"></div>' +
    '</div>' +
    '<div id="aw-content" style="margin-top:12px;white-space:pre-wrap;"></div>';

  const fd = new FormData();
  fd.append('chapter_id', currentChapterId);
  fd.append('section_type', section);
  fd.append('max_iter', maxIter || '3');
  fd.append('target_score', String(awTargetScore));
  const res = await fetch('/api/autowrite', {{method: 'POST', body: fd}});
  const data = await res.json();

  // Custom stream handler for autowrite
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();
  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;
  const status = document.getElementById('stream-status');
  const scoresEl = document.getElementById('stream-scores');
  const awContent = document.getElementById('aw-content');
  const awLog = document.getElementById('aw-log');

  source.onmessage = function(e) {{
    const evt = JSON.parse(e.data);
    if (evt.type === 'token') {{
      awContent.innerHTML += evt.text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
      awContent.scrollTop = awContent.scrollHeight;
    }}
    else if (evt.type === 'progress') {{
      status.textContent = evt.detail || evt.stage;
    }}
    else if (evt.type === 'scores') {{
      awScores.push(evt.scores);
      drawConvergenceChart();
      // Show score bars
      scoresEl.style.display = 'block';
      const s = evt.scores;
      const dims = ['groundedness', 'completeness', 'coherence', 'citation_accuracy', 'overall'];
      scoresEl.innerHTML = 'Iteration ' + evt.iteration + ': ' + dims.map(d => {{
        const v = (s[d] || 0).toFixed(2);
        const cls = v >= 0.85 ? 'good' : v >= 0.7 ? 'mid' : 'low';
        return '<span class="score-bar"><span class="label">' + d.slice(0,5) + '</span> ' +
          '<span class="value ' + cls + '">' + v + '</span></span>';
      }}).join('');
    }}
    else if (evt.type === 'iteration_start') {{
      awContent.innerHTML = '';
      awLog.innerHTML += '<div style="opacity:0.5;border-top:1px solid var(--border);padding-top:4px;margin-top:4px;">Iteration ' + evt.iteration + '/' + evt.max + '</div>';
    }}
    else if (evt.type === 'revision_verdict') {{
      const cls = evt.action === 'KEEP' ? 'log-keep' : 'log-discard';
      const icon = evt.action === 'KEEP' ? '\\u2713' : '\\u2717';
      awLog.innerHTML += '<div class="' + cls + '">' + icon + ' ' + evt.action +
        ': ' + evt.old_score.toFixed(2) + ' \\u2192 ' + evt.new_score.toFixed(2) + '</div>';
      awContent.innerHTML = '';
    }}
    else if (evt.type === 'converged') {{
      status.textContent = 'Converged at iteration ' + evt.iteration + ' (score: ' + evt.final_score.toFixed(2) + ')';
      awLog.innerHTML += '<div class="log-keep" style="font-weight:bold;">\\u2713 CONVERGED (score: ' + evt.final_score.toFixed(2) + ')</div>';
    }}
    else if (evt.type === 'completed') {{
      status.textContent = 'Done';
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
      refreshAfterJob(evt.draft_id);
    }}
    else if (evt.type === 'error') {{
      status.textContent = 'Error: ' + evt.message;
      awLog.innerHTML += '<div style="color:var(--danger);">' + evt.message + '</div>';
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }}
    else if (evt.type === 'done') {{
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }}
  }};
}}

function drawConvergenceChart() {{
  const svg = document.querySelector('#aw-chart svg');
  if (!svg || awScores.length === 0) return;

  const w = 400, h = 120, pad = 20;
  const n = awScores.length;
  const maxN = Math.max(n, 3);

  let html = '';
  // Target line
  const ty = h - pad - (awTargetScore * (h - 2 * pad));
  html += '<line x1="' + pad + '" y1="' + ty + '" x2="' + (w - pad) + '" y2="' + ty + '" class="chart-target"/>';
  html += '<text x="' + (w - pad + 4) + '" y="' + (ty + 3) + '" font-size="9" fill="var(--success)">' + awTargetScore + '</text>';

  // Score line
  const points = awScores.map((s, i) => {{
    const x = pad + (i / (maxN - 1)) * (w - 2 * pad);
    const y = h - pad - ((s.overall || 0) * (h - 2 * pad));
    return x + ',' + y;
  }});
  html += '<polyline points="' + points.join(' ') + '" class="chart-line"/>';

  // Dots
  awScores.forEach((s, i) => {{
    const x = pad + (i / (maxN - 1)) * (w - 2 * pad);
    const y = h - pad - ((s.overall || 0) * (h - 2 * pad));
    html += '<circle cx="' + x + '" cy="' + y + '" r="4" class="chart-dot"/>';
    html += '<text x="' + x + '" y="' + (y - 8) + '" font-size="10" text-anchor="middle" fill="var(--fg)">' + (s.overall || 0).toFixed(2) + '</text>';
  }});

  // Axes labels
  html += '<text x="' + pad + '" y="' + (h - 2) + '" font-size="9" fill="var(--fg)" opacity="0.5">1</text>';
  if (n > 1) html += '<text x="' + (w - pad) + '" y="' + (h - 2) + '" font-size="9" fill="var(--fg)" opacity="0.5" text-anchor="end">' + n + '</text>';

  svg.innerHTML = html;
}}

// ── Argument Map Visualization (Phase 5c) ─────────────────────────────
async function promptArgue() {{
  const claim = prompt('Enter a claim to map evidence for/against:');
  if (!claim) return;
  showStreamPanel('Mapping argument...');

  const fd = new FormData();
  fd.append('claim', claim);
  const res = await fetch('/api/argue', {{method: 'POST', body: fd}});
  const data = await res.json();

  // Custom handler that parses the argue output and builds a map
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();
  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;
  const body = document.getElementById('stream-body');
  const status = document.getElementById('stream-status');
  let fullText = '';

  source.onmessage = function(e) {{
    const evt = JSON.parse(e.data);
    if (evt.type === 'token') {{
      fullText += evt.text;
      body.innerHTML = fullText.replace(/</g, '&lt;').replace(/>/g, '&gt;');
      body.scrollTop = body.scrollHeight;
    }}
    else if (evt.type === 'progress') {{
      status.textContent = evt.detail || evt.stage;
    }}
    else if (evt.type === 'completed') {{
      status.textContent = 'Done';
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
      // Build visual argument map from the text
      buildArgueMap(claim, fullText);
      if (evt.draft_id) refreshAfterJob(evt.draft_id);
    }}
    else if (evt.type === 'error') {{
      status.textContent = 'Error: ' + evt.message;
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }}
    else if (evt.type === 'done') {{
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }}
  }};
}}

function buildArgueMap(claim, text) {{
  // Parse the structured argue output to extract supporting/contradicting/neutral
  const sections = {{supporting: [], contradicting: [], neutral: []}};
  let currentSection = null;

  text.split('\\n').forEach(line => {{
    const lower = line.toLowerCase();
    if (lower.includes('evidence supporting') || lower.includes('supports')) currentSection = 'supporting';
    else if (lower.includes('counterargument') || lower.includes('contradict')) currentSection = 'contradicting';
    else if (lower.includes('methodological') || lower.includes('neutral')) currentSection = 'neutral';
    else if (lower.includes('assessment') || lower.includes('overall')) currentSection = null;

    // Extract citations from the line
    const cites = line.match(/\[\d+\]/g);
    if (currentSection && cites) {{
      cites.forEach(c => {{
        if (!sections[currentSection].includes(c)) sections[currentSection].push(c);
      }});
    }}
  }});

  const mapDiv = document.getElementById('argue-map-view');
  if (sections.supporting.length === 0 && sections.contradicting.length === 0 && sections.neutral.length === 0) {{
    mapDiv.style.display = 'none';
    return;
  }}

  const w = 700, h = 350;
  const cx = w / 2, cy = h / 2;

  let svg = '<svg viewBox="0 0 ' + w + ' ' + h + '" xmlns="http://www.w3.org/2000/svg">';

  // Central claim
  svg += '<rect x="' + (cx - 120) + '" y="' + (cy - 20) + '" width="240" height="40" rx="8" fill="var(--accent)" opacity="0.9"/>';
  svg += '<text x="' + cx + '" y="' + (cy + 5) + '" text-anchor="middle" fill="white" font-size="12" font-weight="bold">' +
    claim.substring(0, 40) + (claim.length > 40 ? '...' : '') + '</text>';

  // Supporting (left)
  sections.supporting.forEach((c, i) => {{
    const ny = 30 + i * 45;
    const nx = 80;
    svg += '<line x1="' + nx + '" y1="' + ny + '" x2="' + (cx - 120) + '" y2="' + cy + '" class="link-supports"/>';
    svg += '<rect x="' + (nx - 40) + '" y="' + (ny - 14) + '" width="80" height="28" rx="6" fill="var(--success)" opacity="0.2" stroke="var(--success)"/>';
    svg += '<text x="' + nx + '" y="' + (ny + 4) + '" text-anchor="middle" font-size="12" fill="var(--fg)">' + c + '</text>';
  }});

  // Contradicting (right)
  sections.contradicting.forEach((c, i) => {{
    const ny = 30 + i * 45;
    const nx = w - 80;
    svg += '<line x1="' + nx + '" y1="' + ny + '" x2="' + (cx + 120) + '" y2="' + cy + '" class="link-contradicts"/>';
    svg += '<rect x="' + (nx - 40) + '" y="' + (ny - 14) + '" width="80" height="28" rx="6" fill="var(--danger)" opacity="0.2" stroke="var(--danger)"/>';
    svg += '<text x="' + nx + '" y="' + (ny + 4) + '" text-anchor="middle" font-size="12" fill="var(--fg)">' + c + '</text>';
  }});

  // Neutral (bottom)
  sections.neutral.forEach((c, i) => {{
    const nx = cx - 100 + i * 70;
    const ny = h - 40;
    svg += '<line x1="' + nx + '" y1="' + ny + '" x2="' + cx + '" y2="' + (cy + 20) + '" class="link-neutral"/>';
    svg += '<rect x="' + (nx - 30) + '" y="' + (ny - 14) + '" width="60" height="28" rx="6" fill="var(--border)" opacity="0.3" stroke="var(--border)"/>';
    svg += '<text x="' + nx + '" y="' + (ny + 4) + '" text-anchor="middle" font-size="12" fill="var(--fg)">' + c + '</text>';
  }});

  // Legend
  svg += '<text x="10" y="' + (h - 5) + '" font-size="10" fill="var(--success)">\\u25cf Supports</text>';
  svg += '<text x="100" y="' + (h - 5) + '" font-size="10" fill="var(--danger)">\\u25cf Contradicts</text>';
  svg += '<text x="210" y="' + (h - 5) + '" font-size="10" fill="var(--fg)" opacity="0.5">\\u25cf Neutral</text>';

  svg += '</svg>';

  mapDiv.innerHTML = '<div class="argue-map">' + svg + '</div>';
  mapDiv.style.display = 'block';
}}
</script>
</body>
</html>
"""
