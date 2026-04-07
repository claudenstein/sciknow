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
        html += f'<div class="ch-title">Ch.{ch["num"]}: {ch["title"]}</div>'
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
    return "<ol>" + "".join(f"<li>{s}</li>" for s in sources if s) + "</ol>"


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
  <div class="job-indicator" id="job-indicator">Working...</div>
  <div style="padding: 8px 16px; font-size: 12px; opacity: 0.5; margin-top: 16px;">
    <span id="gaps-count">{gaps_count}</span> open gaps
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
  </div>

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

  <form id="edit-view" action="/edit/{active_id}" method="post" style="display:none;">
    <textarea class="edit-area" name="content" id="edit-area"></textarea>
    <br>
    <button type="submit" class="edit-btn" style="margin-top:8px;">Save</button>
    <button type="button" class="edit-btn" style="background:var(--danger);" onclick="toggleEdit()">Cancel</button>
  </form>
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
    html += '<div class="ch-title">Ch.' + ch.num + ': ' + ch.title + '</div>';
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
</script>
</body>
</html>
"""
