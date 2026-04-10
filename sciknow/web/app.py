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

# Default chapter sections for a *book* (not a paper). Used when a chapter
# has no custom sections defined (book_chapters.sections is empty). Mirrors
# core/book_ops.py:DEFAULT_SECTIONS so the GUI and CLI agree on the fallback.
_DEFAULT_BOOK_SECTIONS = [
    "overview", "key_evidence", "current_understanding",
    "open_questions", "summary",
]


def _normalize_section(name: str) -> str:
    """Lowercase + spaces-to-underscores so per-chapter custom section names
    in book_chapters.sections (which can be 'Historical Context', 'Key Evidence')
    match the storage convention used in drafts.section_type."""
    return (name or "").strip().lower().replace(" ", "_")


def _chapter_sections(ch_row) -> list[str]:
    """Return the section template for a chapter row.

    The chapter SQL row carries `bc.sections` as a JSONB list (possibly empty).
    Falls back to _DEFAULT_BOOK_SECTIONS when empty so the GUI never shows
    paper-style placeholders for a science book.
    """
    raw = ch_row[6] if len(ch_row) > 6 else None
    if isinstance(raw, list) and raw:
        return [_normalize_section(s) for s in raw if s]
    return list(_DEFAULT_BOOK_SECTIONS)


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
                   bc.topic_query, bc.topic_cluster, bc.sections
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


def _get_draft_status(draft_id: str) -> str:
    """Get the status of a draft, defaulting to 'drafted'."""
    with get_session() as session:
        row = session.execute(text(
            "SELECT COALESCE(status, 'drafted') FROM drafts WHERE id::text LIKE :q LIMIT 1"
        ), {"q": f"{draft_id}%"}).fetchone()
    return row[0] if row else "drafted"


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
    # book columns: id, title, description, plan, status
    return {
        "id": book[0] if book else "",
        "title": book[1] if book else "",
        "description": (book[2] or "") if book else "",
        "plan": (book[3] or "") if book else "",
        "status": (book[4] or "draft") if book else "draft",
        "chapters": len(chapters),
        "drafts": len(drafts),
        "gaps": len(gaps),
        "comments": len(comments),
    }


@app.put("/api/book")
async def api_book_update(
    title: str = Form(None),
    description: str = Form(None),
    plan: str = Form(None),
):
    """Update the book's title, description (short blurb), or plan
    (the 200-500 word thesis/scope document used by the writer prompt).
    All three are optional — only the fields you pass get updated."""
    updates = []
    params: dict = {"bid": _book_id}
    if title is not None:
        updates.append("title = :title")
        params["title"] = title
    if description is not None:
        updates.append("description = :desc")
        params["desc"] = description
    if plan is not None:
        updates.append("plan = :plan")
        params["plan"] = plan
    if not updates:
        return JSONResponse({"ok": True})

    with get_session() as session:
        session.execute(text(
            f"UPDATE books SET {', '.join(updates)} WHERE id::text = :bid"
        ), params)
        session.commit()
    return JSONResponse({"ok": True})


@app.post("/api/book/plan/generate")
async def api_book_plan_generate(model: str = Form(None)):
    """Generate (or regenerate) the book plan via the LLM, streaming
    tokens to the browser via SSE. Mirrors the `sciknow book plan --edit`
    CLI flow but persists to drafts.custom_metadata-style streaming."""
    job_id, queue = _create_job("book_plan_generate")
    loop = asyncio.get_event_loop()

    def gen():
        from sciknow.rag import prompts as rag_prompts
        from sciknow.rag.llm import stream as llm_stream

        with get_session() as session:
            book = session.execute(text("""
                SELECT id::text, title, description, plan
                FROM books WHERE id::text = :bid
            """), {"bid": _book_id}).fetchone()
            if not book:
                yield {"type": "error", "message": "Book not found."}
                return
            chapters = session.execute(text("""
                SELECT number, title, description FROM book_chapters
                WHERE book_id = :bid ORDER BY number
            """), {"bid": _book_id}).fetchall()
            papers = session.execute(text("""
                SELECT pm.title, pm.year FROM paper_metadata pm
                JOIN documents d ON d.id = pm.document_id
                WHERE d.ingestion_status = 'complete' AND pm.title IS NOT NULL
                ORDER BY pm.year DESC NULLS LAST LIMIT 200
            """)).fetchall()

        ch_list = [{"number": r[0], "title": r[1], "description": r[2] or ""}
                   for r in chapters]
        paper_list = [{"title": r[0], "year": r[1]} for r in papers]

        yield {"type": "progress", "stage": "generating",
               "detail": f"Drafting plan from {len(ch_list)} chapters and {len(paper_list)} papers..."}

        sys_p, usr_p = rag_prompts.book_plan(book[1], book[2], ch_list, paper_list)
        tokens: list[str] = []
        for tok in llm_stream(sys_p, usr_p, model=model or None):
            tokens.append(tok)
            yield {"type": "token", "text": tok}

        new_plan = "".join(tokens).strip()
        # Persist
        with get_session() as session:
            session.execute(text(
                "UPDATE books SET plan = :plan WHERE id::text = :bid"
            ), {"plan": new_plan, "bid": _book_id})
            session.commit()
        yield {"type": "completed", "plan": new_plan, "chars": len(new_plan)}

    threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


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
        "status": _get_draft_status(draft_id),
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
        ch_id, ch_num, ch_title, ch_desc, tq, tc, ch_sections_template = ch
        ch_ds = chapter_drafts.get(ch_id, [])
        # Order drafts by their chapter's own section template if defined,
        # otherwise by the book defaults. Falls through to alphabetical for
        # any unknown section types.
        template = _chapter_sections(ch)
        section_order = {s: i for i, s in enumerate(template)}
        sections = []
        for d in sorted(ch_ds, key=lambda x: section_order.get(_normalize_section(x[2] or ""), 99)):
            sections.append({
                "id": d[0], "type": d[2] or "text",
                "version": d[6] or 1, "words": d[4] or 0,
            })
        result.append({
            "id": ch_id, "num": ch_num, "title": ch_title,
            "description": ch_desc, "topic_query": tq,
            "sections": sections,
            # Phase 14.4 — per-chapter section template (the menu of section
            # types this chapter wants, drawn from book_chapters.sections
            # or the book-style defaults).
            "sections_template": template,
        })

    return {"chapters": result, "gaps_count": len([g for g in gaps if g[3] == "open"])}


@app.get("/api/dashboard")
async def api_dashboard():
    """Return dashboard data: completion heatmap, stats, gaps."""
    book, chapters, drafts, gaps, comments = _get_book_data()

    # Phase 14.4 — section_types is now a UNION computed from real data
    # rather than a hardcoded paper-style list. The columns of the
    # heatmap reflect:
    #   - actual section types of any saved drafts (so existing work
    #     doesn't disappear if a draft uses an unusual section type)
    #   - per-chapter custom sections from book_chapters.sections
    #     (populated by `book outline` for new books)
    #   - the book-style defaults (overview/key_evidence/.../summary)
    #     so a fresh book always has columns to show.
    section_types_set: set[str] = set(_DEFAULT_BOOK_SECTIONS)
    for ch in chapters:
        for s in _chapter_sections(ch):
            section_types_set.add(s)

    # Build draft lookup: chapter_id -> {section_type: {version, words, id, has_review}}
    ch_section_drafts: dict[str, dict] = {}
    total_words = 0
    for d in drafts:
        draft_id, title, sec_type, content, wc, sources, version, summary, \
            review_fb, ch_id, parent_id, created, ch_num, ch_title = d
        total_words += wc or 0
        section_types_set.add(_normalize_section(sec_type or ""))
        if not ch_id:
            continue
        if ch_id not in ch_section_drafts:
            ch_section_drafts[ch_id] = {}
        # Keep only the latest version per section_type
        existing = ch_section_drafts[ch_id].get(sec_type)
        if not existing or (version or 1) > existing["version"]:
            ch_section_drafts[ch_id][sec_type] = {
                "id": draft_id, "version": version or 1, "words": wc or 0,
                "has_review": bool(review_fb),
            }

    section_types_set.discard("")

    # Stable ordering: book defaults first (they read in narrative order),
    # then any extras alphabetically.
    section_types: list[str] = [s for s in _DEFAULT_BOOK_SECTIONS if s in section_types_set]
    extras = sorted(section_types_set - set(_DEFAULT_BOOK_SECTIONS))
    section_types.extend(extras)

    heatmap = []
    for ch in chapters:
        ch_id, ch_num, ch_title, ch_desc, tq, tc, ch_sections_template = ch
        row = {
            "num": ch_num, "title": ch_title, "id": ch_id, "cells": [],
            # Phase 14.4 — include per-chapter section template so the GUI
            # can highlight the columns this chapter actually wants.
            "sections_template": _chapter_sections(ch),
            "description": ch_desc or "",
            "topic_query": tq or "",
        }
        secs = ch_section_drafts.get(ch_id, {})
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


# ── Snapshots ────────────────────────────────────────────────────────────────

@app.post("/api/snapshot/{draft_id}")
async def create_snapshot(draft_id: str, name: str = Form("")):
    """Save a named snapshot of a draft's current content."""
    with get_session() as session:
        draft = session.execute(text(
            "SELECT content, word_count FROM drafts WHERE id::text LIKE :q LIMIT 1"
        ), {"q": f"{draft_id}%"}).fetchone()
        if not draft:
            raise HTTPException(404, "Draft not found")

        snap_name = name or f"Snapshot {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}"
        session.execute(text("""
            INSERT INTO draft_snapshots (draft_id, name, content, word_count)
            VALUES (:did::uuid, :name, :content, :wc)
        """), {"did": draft_id, "name": snap_name,
               "content": draft[0], "wc": draft[1]})
        session.commit()

    return JSONResponse({"ok": True, "name": snap_name})


@app.get("/api/snapshots/{draft_id}")
async def list_snapshots(draft_id: str):
    """List all snapshots for a draft."""
    with get_session() as session:
        rows = session.execute(text("""
            SELECT id::text, name, word_count, created_at
            FROM draft_snapshots WHERE draft_id::text LIKE :q
            ORDER BY created_at DESC
        """), {"q": f"{draft_id}%"}).fetchall()

    return {"snapshots": [
        {"id": r[0], "name": r[1], "word_count": r[2] or 0,
         "created_at": str(r[3]) if r[3] else ""}
        for r in rows
    ]}


@app.get("/api/snapshot-content/{snapshot_id}")
async def get_snapshot_content(snapshot_id: str):
    """Get the content of a specific snapshot."""
    with get_session() as session:
        row = session.execute(text(
            "SELECT content FROM draft_snapshots WHERE id::text = :sid"
        ), {"sid": snapshot_id}).fetchone()
    if not row:
        raise HTTPException(404, "Snapshot not found")
    return {"content": row[0]}


# ── Draft status + metadata ──────────────────────────────────────────────────

@app.put("/api/draft/{draft_id}/status")
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


@app.put("/api/draft/{draft_id}/metadata")
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


# ── Chapter reader (continuous scroll) ───────────────────────────────────────

@app.get("/api/chapter-reader/{chapter_id}")
async def chapter_reader(chapter_id: str):
    """Return all sections of a chapter concatenated for continuous reading."""
    section_order = {"introduction": 0, "methods": 1, "results": 2, "discussion": 3, "conclusion": 4}

    with get_session() as session:
        ch = session.execute(text("""
            SELECT bc.number, bc.title FROM book_chapters bc
            WHERE bc.id::text = :cid
        """), {"cid": chapter_id}).fetchone()
        if not ch:
            raise HTTPException(404, "Chapter not found")

        # Get latest version per section_type
        drafts = session.execute(text("""
            SELECT d.id::text, d.section_type, d.content, d.word_count,
                   d.version, d.status
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

    sections = sorted(seen.values(), key=lambda x: section_order.get(x[1] or "", 9))

    combined_html = ""
    total_words = 0
    for d in sections:
        sec_type = (d[1] or "text").capitalize()
        combined_html += f'<h2 class="reader-section-title">{sec_type}</h2>'
        combined_html += _md_to_html(d[2] or "")
        total_words += d[3] or 0

    return {
        "chapter_num": ch[0],
        "chapter_title": ch[1],
        "html": combined_html,
        "total_words": total_words,
        "section_count": len(sections),
    }


# ── Corkboard data ──────────────────────────────────────────────────────────

@app.get("/api/corkboard")
async def corkboard_data():
    """Return data for the corkboard view: cards for each chapter/section."""
    book, chapters, drafts, gaps, comments = _get_book_data()

    section_order = {"introduction": 0, "methods": 1, "results": 2, "discussion": 3, "conclusion": 4}

    # Build latest draft per chapter+section
    ch_sections: dict[str, dict] = {}
    for d in drafts:
        draft_id, title, sec_type, content, wc, sources, version, summary, \
            review_fb, ch_id, parent_id, created, ch_num, ch_title = d
        if not ch_id:
            continue
        if ch_id not in ch_sections:
            ch_sections[ch_id] = {}
        existing = ch_sections[ch_id].get(sec_type)
        if not existing or (version or 1) > existing["version"]:
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
        """), {"bid": _book_id}).fetchall()
    status_map = {r[0]: r[1] for r in status_rows}

    cards = []
    for ch in chapters:
        ch_id, ch_num, ch_title, ch_desc, tq, tc, ch_sections_template = ch
        secs = ch_sections.get(ch_id, {})
        for sec_type in ["introduction", "methods", "results", "discussion", "conclusion"]:
            info = secs.get(sec_type)
            if info:
                info["status"] = status_map.get(info["draft_id"], "drafted")
                cards.append({
                    "chapter_id": ch_id, "chapter_num": ch_num,
                    "chapter_title": ch_title,
                    "section_type": sec_type, "draft_id": info["draft_id"],
                    "version": info["version"], "words": info["words"],
                    "summary": info["summary"], "has_review": info["has_review"],
                    "status": info["status"],
                })
            else:
                cards.append({
                    "chapter_id": ch_id, "chapter_num": ch_num,
                    "chapter_title": ch_title,
                    "section_type": sec_type, "draft_id": None,
                    "version": 0, "words": 0, "summary": "",
                    "has_review": False, "status": "to_do",
                })

    return {"cards": cards}


# ── SSE / Streaming endpoints ───────────────────────────────────────────────

def _run_generator_in_thread(job_id: str, generator_fn, loop):
    """Run a blocking generator in a thread, pushing events to the job queue.

    Phase 15.1 — when the job is cancelled, we explicitly close the
    generator (`gen.close()`) so any try/finally blocks inside it can
    flush partial state. The generator functions in book_ops use
    incremental save at every checkpoint, so the worst case is the
    in-flight iteration's tokens being lost — never the whole draft.
    """
    queue = _jobs[job_id]["queue"]
    cancel = _jobs[job_id]["cancel"]
    gen = generator_fn()
    try:
        for event in gen:
            if cancel.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "cancelled"})
                break
            loop.call_soon_threadsafe(queue.put_nowait, event)
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(exc)})
    finally:
        try:
            gen.close()  # raises GeneratorExit at the current yield point
        except Exception:
            pass
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


# ── Phase 14: Web v2 endpoints ──────────────────────────────────────────────
#
# These add CLI parity to the GUI: score history viewer (Phase 13), wiki
# query, corpus RAG ask, catalog browser, and a stats panel for the
# dashboard.


@app.get("/api/draft/{draft_id}/scores")
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


@app.get("/api/wiki/pages")
async def api_wiki_pages(page_type: str = None, page: int = 1, per_page: int = 50):
    """Phase 15 — paginated wiki page list with optional type filter.

    Wraps wiki_ops.list_pages and adds pagination so the GUI can browse
    a corpus with thousands of compiled wiki pages without loading
    everything into the browser at once.
    """
    from sciknow.core.wiki_ops import list_pages
    try:
        all_pages = list_pages(page_type=page_type or None)
    except Exception as exc:
        # If wiki_pages table doesn't exist yet, return empty list rather
        # than 500 — the GUI shows an empty state.
        return JSONResponse({"page": 1, "per_page": per_page, "total": 0,
                             "n_pages": 0, "pages": [], "available_types": [],
                             "error": str(exc)})

    page = max(page, 1)
    per_page = min(max(per_page, 1), 200)
    total = len(all_pages)
    start = (page - 1) * per_page
    end = start + per_page

    # Build the list of available page_types so the filter dropdown is
    # populated dynamically (paper_summary / concept / synthesis).
    available_types = sorted({p["page_type"] for p in all_pages if p.get("page_type")})

    return JSONResponse({
        "page": page, "per_page": per_page, "total": total,
        "n_pages": (total + per_page - 1) // per_page,
        "pages": all_pages[start:end],
        "available_types": available_types,
    })


@app.get("/api/wiki/page/{slug}")
async def api_wiki_page(slug: str):
    """Phase 15 — return one wiki page's full content + metadata."""
    from sciknow.core.wiki_ops import show_page

    page = show_page(slug)
    if not page:
        raise HTTPException(404, f"Wiki page not found: {slug}")

    # Pull metadata from the DB so the GUI can show word count, sources,
    # last updated, page type alongside the markdown.
    try:
        with get_session() as session:
            row = session.execute(text("""
                SELECT title, page_type, word_count,
                       array_length(source_doc_ids, 1) AS n_sources,
                       updated_at
                FROM wiki_pages WHERE slug = :slug
            """), {"slug": slug}).fetchone()
        if row:
            page.update({
                "title": row[0], "page_type": row[1], "word_count": row[2] or 0,
                "n_sources": row[3] or 0, "updated_at": str(row[4]),
            })
    except Exception:
        pass

    page["content_html"] = _md_to_html(page.get("content", ""))
    return JSONResponse(page)


@app.post("/api/wiki/query")
async def api_wiki_query(question: str = Form(...), model: str = Form(None)):
    """Stream a wiki query — wraps wiki_ops.query_wiki as an SSE job."""
    from sciknow.core.wiki_ops import query_wiki

    job_id, queue = _create_job("wiki_query")
    loop = asyncio.get_event_loop()

    def gen():
        return query_wiki(question, model=model or None)

    threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


@app.post("/api/ask")
async def api_ask(
    question: str = Form(...),
    model: str = Form(None),
    context_k: int = Form(8),
    year_from: int = Form(None),
    year_to: int = Form(None),
):
    """Stream a corpus-wide RAG question — full hybrid search + LLM stream.

    Implemented inline as an event generator so the GUI gets the same SSE
    contract as the other operations: progress events, token events, sources,
    completed.
    """
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.storage.qdrant import get_client

    job_id, queue = _create_job("ask")
    loop = asyncio.get_event_loop()

    def gen():
        qdrant = get_client()
        yield {"type": "progress", "stage": "retrieving",
               "detail": "Hybrid search across the corpus..."}
        with get_session() as session:
            candidates = hybrid_search.search(
                query=question, qdrant_client=qdrant, session=session,
                candidate_k=50,
                year_from=year_from if year_from else None,
                year_to=year_to if year_to else None,
            )
            if not candidates:
                yield {"type": "error", "message": "No relevant passages found."}
                return
            candidates = reranker.rerank(question, candidates, top_k=context_k)
            results = context_builder.build(candidates, session)

        from sciknow.rag.prompts import format_sources
        sources = format_sources(results).splitlines()
        yield {"type": "sources", "sources": sources, "n": len(sources)}

        system, user = rag_prompts.qa(question, results)
        yield {"type": "progress", "stage": "generating",
               "detail": f"Generating answer from {len(results)} passages..."}
        for tok in llm_stream(system, user, model=model or None):
            yield {"type": "token", "text": tok}
        yield {"type": "completed"}

    threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


@app.get("/api/catalog")
async def api_catalog(
    page: int = 1,
    per_page: int = 25,
    year_from: int = None,
    year_to: int = None,
    author: str = None,
    journal: str = None,
    topic_cluster: str = None,
):
    """Paginated paper list with optional filters. Mirrors `sciknow catalog list`."""
    page = max(page, 1)
    per_page = min(max(per_page, 1), 100)
    offset = (page - 1) * per_page

    where = []
    params: dict = {"limit": per_page, "offset": offset}
    if year_from is not None:
        where.append("pm.year >= :year_from")
        params["year_from"] = year_from
    if year_to is not None:
        where.append("pm.year <= :year_to")
        params["year_to"] = year_to
    if author:
        where.append("EXISTS (SELECT 1 FROM jsonb_array_elements(pm.authors) a WHERE a->>'name' ILIKE :author)")
        params["author"] = f"%{author}%"
    if journal:
        where.append("pm.journal ILIKE :journal")
        params["journal"] = f"%{journal}%"
    if topic_cluster:
        where.append("pm.topic_cluster = :topic_cluster")
        params["topic_cluster"] = topic_cluster

    where_clause = ("WHERE " + " AND ".join(where)) if where else ""

    with get_session() as session:
        total = session.execute(text(f"""
            SELECT COUNT(*) FROM paper_metadata pm {where_clause}
        """), {k: v for k, v in params.items() if k not in ("limit", "offset")}).scalar() or 0

        rows = session.execute(text(f"""
            SELECT pm.document_id::text, pm.title, pm.year, pm.authors,
                   pm.journal, pm.doi, pm.abstract, pm.topic_cluster,
                   pm.metadata_source
            FROM paper_metadata pm
            {where_clause}
            ORDER BY pm.year DESC NULLS LAST, pm.title
            LIMIT :limit OFFSET :offset
        """), params).fetchall()

    papers = [
        {
            "document_id": r[0],
            "title": r[1] or "(untitled)",
            "year": r[2],
            "authors": r[3] or [],
            "journal": r[4],
            "doi": r[5],
            "abstract": (r[6] or "")[:600],
            "topic_cluster": r[7],
            "metadata_source": r[8],
        }
        for r in rows
    ]
    return JSONResponse({
        "page": page,
        "per_page": per_page,
        "total": total,
        "n_pages": (total + per_page - 1) // per_page,
        "papers": papers,
    })


@app.get("/api/stats")
async def api_stats():
    """Aggregate stats for the enhanced dashboard panel.

    Mirrors a subset of `sciknow db stats` + `catalog raptor stats` +
    `catalog topics`. Cheap to compute — runs four counts plus two
    GROUP BYs against PostgreSQL, no Qdrant scrolls except for the
    RAPTOR level breakdown which uses an indexed payload filter so it's
    O(N_summary_nodes) not O(N_chunks).
    """
    out: dict = {}
    with get_session() as session:
        out["n_documents"] = session.execute(text(
            "SELECT COUNT(*) FROM documents"
        )).scalar() or 0
        out["n_completed"] = session.execute(text(
            "SELECT COUNT(*) FROM documents WHERE ingestion_status = 'complete'"
        )).scalar() or 0
        out["n_chunks"] = session.execute(text(
            "SELECT COUNT(*) FROM chunks"
        )).scalar() or 0
        out["n_citations"] = session.execute(text(
            "SELECT COUNT(*) FROM citations"
        )).scalar() or 0

        # Ingest source breakdown (seed vs expand)
        rows = session.execute(text("""
            SELECT ingest_source, COUNT(*) FROM documents
            GROUP BY ingest_source ORDER BY COUNT(*) DESC
        """)).fetchall()
        out["ingest_sources"] = [{"source": r[0] or "unknown", "n": r[1]} for r in rows]

        # Topic clusters
        rows = session.execute(text("""
            SELECT topic_cluster, COUNT(*) FROM paper_metadata
            WHERE topic_cluster IS NOT NULL AND topic_cluster != ''
            GROUP BY topic_cluster ORDER BY COUNT(*) DESC LIMIT 20
        """)).fetchall()
        out["topic_clusters"] = [{"name": r[0], "n": r[1]} for r in rows]

        # Wiki page count if the table exists
        try:
            out["n_wiki_pages"] = session.execute(text(
                "SELECT COUNT(*) FROM wiki_pages"
            )).scalar() or 0
        except Exception:
            out["n_wiki_pages"] = 0

    # RAPTOR level counts via Qdrant filter (indexed → fast).
    raptor_levels: dict[str, int] = {}
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        from sciknow.storage.qdrant import PAPERS_COLLECTION, get_client
        qdrant = get_client()
        for lvl in (0, 1, 2, 3, 4):
            try:
                info = qdrant.count(
                    collection_name=PAPERS_COLLECTION,
                    count_filter=Filter(must=[
                        FieldCondition(key="node_level", match=MatchValue(value=lvl))
                    ]),
                    exact=False,
                )
                n = info.count if hasattr(info, "count") else int(info)
                if n > 0:
                    raptor_levels[f"L{lvl}"] = n
            except Exception:
                pass
    except Exception:
        pass
    out["raptor_levels"] = raptor_levels

    return JSONResponse(out)


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
        ch_id, ch_num, ch_title, ch_desc, tq, tc, ch_sections_template = ch
        ch_ds = chapter_drafts.get(ch_id, [])
        template = _chapter_sections(ch)
        section_order_map = {s: i for i, s in enumerate(template)}
        sections = []
        for d in sorted(ch_ds, key=lambda x: section_order_map.get(_normalize_section(x[2] or ""), 99)):
            sections.append({
                "id": d[0], "type": d[2] or "text", "version": d[6] or 1,
                "words": d[4] or 0,
            })
        sidebar_items.append({
            "num": ch_num, "title": ch_title, "id": ch_id,
            "description": ch_desc or "", "topic_query": tq or "",
            "sections": sections,
            # Phase 14.4 — per-chapter section template for the GUI's
            # empty-state picker and the dashboard heatmap.
            "sections_template": template,
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
    if active_draft:
        active_id = active_draft[0]
        active_title = active_draft[1]
        active_html = _md_to_html(active_draft[3] or "")
        active_sources = json.loads(active_draft[5]) if isinstance(active_draft[5], str) else (active_draft[5] or [])
        active_review = active_draft[8] or ""
        active_comments = draft_comments.get(active_id, [])
    else:
        # Phase 14.2 — Empty-state landing for fresh books with no drafts.
        # Without this, users see a blank page and clicking toolbar buttons
        # silently fails because currentDraftId / currentChapterId are empty.
        if chapters:
            active_title = book[1] if book else "Untitled book"
            active_html = (
                '<div class="empty-state">'
                '<h3>Welcome to your book</h3>'
                f'<p>This book has <strong>{len(chapters)} chapters</strong> outlined and '
                '<strong>0 drafts</strong>. To start writing, click any chapter title in the sidebar '
                '(it will highlight) and then use the toolbar above. The fastest path to a first draft '
                'is to click the <strong>&#9889; Autowrite</strong> button after selecting a chapter '
                '&mdash; it will run the full write &rarr; review &rarr; revise convergence loop.</p>'
                '<p>You can also explore the corpus without writing anything: '
                '<strong>&#128270; Ask Corpus</strong>, <strong>&#128218; Wiki Query</strong>, '
                'and <strong>&#128194; Browse Papers</strong> all work without an active draft.</p>'
                '<p style="font-size:12px;color:var(--fg-faint);">Tip: each chapter in the sidebar now '
                'has a <span style="color:var(--accent);">&#9998; Start writing</span> shortcut '
                'that selects the chapter and immediately drafts an overview.</p>'
                '</div>'
            )
        else:
            active_title = book[1] if book else "Untitled book"
            active_html = (
                '<div class="empty-state">'
                '<h3>This book has no chapters yet</h3>'
                '<p>Run <code>uv run sciknow book outline "your topic" "Book Title"</code> '
                'in your terminal to generate a chapter outline from the literature, '
                'or use <code>sciknow book chapter add</code> to add chapters one at a time. '
                'Then refresh this page.</p>'
                '</div>'
            )

    open_gaps = [g for g in gaps if g[3] == "open"]

    # Build chapters_json for the JS-side SPA
    chapters_json = json.dumps([{
        "id": si["id"], "num": si["num"], "title": si["title"],
        "description": si["description"], "topic_query": si["topic_query"],
        "sections": si["sections"],
        # Phase 14.4 — per-chapter book-style section template
        "sections_template": si["sections_template"],
    } for si in sidebar_items])

    return TEMPLATE.format(
        book_title=book[1] if book else "Untitled",
        book_id=_book_id,
        book_plan=(book[3] or "No plan set.") if book else "",
        sidebar_html=_render_sidebar(sidebar_items, active_id),
        content_html=active_html,
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


def _render_sidebar(items, active_id):
    html = ""
    for ch in items:
        html += f'<div class="ch-group" data-ch-id="{ch["id"]}">'
        # Phase 14.2 — chapter title is now clickable to SELECT the chapter
        # (sets currentChapterId so toolbar buttons like Write work even
        # when there are no drafts yet).
        html += (f'<div class="ch-title clickable" onclick="selectChapter(this.parentElement)">'
                 f'Ch.{ch["num"]}: {ch["title"]}'
                 f'<span class="ch-actions">'
                 f'<button onclick="event.stopPropagation();deleteChapter(this.closest(&quot;.ch-group&quot;).dataset.chId)" title="Delete chapter">\u2717</button>'
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
            # Phase 14.2 — explicit "Start writing" CTA so empty chapters
            # have a one-click path to create a first draft.
            html += (
                f'<div class="sec-link sec-empty-cta" '
                f'onclick="startWritingChapter(&quot;{ch["id"]}&quot;)">'
                f'\u270e Start writing</div>'
            )
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
/* ── Phase 14 — Web reader v2 design system ───────────────────────────────
   Modern indigo accent, refined neutrals, hairline borders, polished
   dark mode. UI uses Inter / system sans; draft body keeps Georgia for
   reading. */
:root {{
  /* Surfaces */
  --bg: #ffffff;
  --bg-elevated: #ffffff;
  --sidebar-bg: #fafafa;
  --toolbar-bg: #f7f8fa;
  --code-bg: #f4f4f6;
  --modal-overlay: rgba(15, 23, 42, 0.45);
  /* Text */
  --fg: #0f172a;
  --fg-muted: #64748b;
  --fg-faint: #94a3b8;
  /* Borders & accents */
  --border: #e5e7eb;
  --border-strong: #d1d5db;
  --accent: #4f46e5;          /* indigo-600 */
  --accent-hover: #4338ca;    /* indigo-700 */
  --accent-light: #eef2ff;    /* indigo-50 */
  --accent-fg: #ffffff;
  /* Semantic */
  --success: #059669;         /* emerald-600 */
  --success-light: #d1fae5;
  --warning: #d97706;         /* amber-600 */
  --warning-light: #fef3c7;
  --danger: #e11d48;          /* rose-600 */
  --danger-light: #ffe4e6;
  --info: #0284c7;            /* sky-600 */
  /* Spacing scale */
  --sp-1: 4px;
  --sp-2: 8px;
  --sp-3: 12px;
  --sp-4: 16px;
  --sp-5: 24px;
  --sp-6: 32px;
  /* Radius */
  --r-sm: 4px;
  --r-md: 6px;
  --r-lg: 10px;
  --r-xl: 14px;
  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(15,23,42,0.04);
  --shadow-md: 0 4px 12px rgba(15,23,42,0.08);
  --shadow-lg: 0 12px 32px rgba(15,23,42,0.14);
  /* Type */
  --font-sans: 'Inter', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-serif: 'Georgia', 'Times New Roman', serif;
  --font-mono: ui-monospace, 'SF Mono', Menlo, Monaco, Consolas, monospace;
}}
[data-theme="dark"] {{
  --bg: #0b1120;              /* slate-950ish */
  --bg-elevated: #111827;
  --sidebar-bg: #0f172a;      /* slate-900 */
  --toolbar-bg: #131c2d;
  --code-bg: #1e293b;
  --modal-overlay: rgba(0, 0, 0, 0.65);
  --fg: #e2e8f0;              /* slate-200 */
  --fg-muted: #94a3b8;
  --fg-faint: #64748b;
  --border: #1f2937;          /* gray-800 */
  --border-strong: #374151;
  --accent: #818cf8;          /* indigo-400 */
  --accent-hover: #a5b4fc;
  --accent-light: #1e1b4b;    /* indigo-950 */
  --success: #10b981;
  --success-light: #064e3b;
  --warning: #f59e0b;
  --warning-light: #78350f;
  --danger: #f43f5e;
  --danger-light: #4c0519;
  --info: #38bdf8;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.30);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.40);
  --shadow-lg: 0 16px 40px rgba(0,0,0,0.55);
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
html {{ -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }}
body {{ font-family: var(--font-sans); color: var(--fg); background: var(--bg);
        display: flex; height: 100vh; font-size: 14px; }}
button, input, textarea, select {{ font-family: inherit; color: inherit; }}
/* Sidebar */
.sidebar {{ width: 280px; background: var(--sidebar-bg); border-right: 1px solid var(--border);
            overflow-y: auto; flex-shrink: 0; padding: var(--sp-4) 0;
            display: flex; flex-direction: column; }}
.sidebar h2 {{ padding: var(--sp-2) var(--sp-4) var(--sp-3); font-size: 15px;
              font-weight: 600; color: var(--fg); letter-spacing: -0.01em; }}
.ch-group {{ margin-bottom: var(--sp-2); }}
.ch-title {{ padding: var(--sp-2) var(--sp-4) var(--sp-1); font-weight: 600; font-size: 11px;
             text-transform: uppercase; letter-spacing: 0.06em; color: var(--fg-muted); }}
.sec-link {{ display: block; padding: 6px var(--sp-4) 6px 28px; text-decoration: none;
             color: var(--fg); font-size: 13px; border-left: 2px solid transparent; cursor: pointer;
             transition: background .12s ease; }}
.sec-link:hover {{ background: var(--toolbar-bg); }}
.sec-link.active {{ border-left-color: var(--accent); background: var(--accent-light);
                   color: var(--accent); font-weight: 600; }}
.sec-link .meta {{ font-size: 11px; color: var(--fg-faint); margin-left: 6px; }}
.sec-link.empty {{ color: var(--fg-faint); font-style: italic; }}
/* Main */
.main {{ flex: 1; overflow-y: auto; padding: var(--sp-6) 48px; max-width: 980px; }}
.main h1 {{ font-size: 28px; font-weight: 700; letter-spacing: -0.02em;
            margin-bottom: var(--sp-1); color: var(--fg); }}
.main .subtitle {{ font-size: 13px; color: var(--fg-muted); margin-bottom: var(--sp-3);
                   display: flex; align-items: center; gap: var(--sp-2); }}
.main p {{ font-family: var(--font-serif); font-size: 16px; line-height: 1.78;
           margin-bottom: var(--sp-3); text-align: justify; color: var(--fg); }}
.main h2,.main h3,.main h4 {{ margin: var(--sp-5) 0 var(--sp-3); font-weight: 700;
                              letter-spacing: -0.01em; color: var(--fg); }}
.main h2 {{ font-size: 22px; }} .main h3 {{ font-size: 18px; }} .main h4 {{ font-size: 15px; }}
.citation {{ color: var(--accent); cursor: pointer; font-weight: 600;
             text-decoration: none; padding: 0 1px; }}
.citation:hover {{ background: var(--accent-light); border-radius: 2px; }}
/* Action toolbar — grouped sections, modern pills */
.toolbar {{ display: flex; gap: var(--sp-1); flex-wrap: wrap; margin-bottom: var(--sp-5);
            padding: var(--sp-2); background: var(--toolbar-bg);
            border-radius: var(--r-lg); border: 1px solid var(--border);
            box-shadow: var(--shadow-sm); align-items: center; }}
.toolbar .tg {{ display: flex; gap: 2px; padding: 0 4px; }}
.toolbar button {{ font-size: 12px; font-weight: 500; padding: 6px 12px;
                   border: 1px solid transparent; border-radius: var(--r-md);
                   cursor: pointer; background: transparent; color: var(--fg);
                   transition: all .12s ease; display: inline-flex; align-items: center;
                   gap: 6px; line-height: 1; }}
.toolbar button:hover {{ background: var(--bg-elevated); border-color: var(--border);
                         box-shadow: var(--shadow-sm); }}
.toolbar button:active {{ transform: translateY(1px); }}
.toolbar button.primary {{ background: var(--accent); color: var(--accent-fg);
                           border-color: var(--accent); }}
.toolbar button.primary:hover {{ background: var(--accent-hover); border-color: var(--accent-hover); }}
.toolbar button.active {{ background: var(--accent); color: var(--accent-fg);
                          border-color: var(--accent); }}
.toolbar .sep {{ width: 1px; align-self: stretch; background: var(--border); margin: 4px 4px; }}
.tg-label {{ font-size: 10px; font-weight: 600; color: var(--fg-faint);
             text-transform: uppercase; letter-spacing: 0.06em; padding: 0 6px;
             align-self: center; }}
/* Modal infrastructure */
.modal-overlay {{ display: none; position: fixed; inset: 0; background: var(--modal-overlay);
                  backdrop-filter: blur(4px); -webkit-backdrop-filter: blur(4px);
                  z-index: 1000; align-items: flex-start; justify-content: center;
                  padding-top: 64px; }}
.modal-overlay.open {{ display: flex; animation: fadeIn .15s ease; }}
.modal {{ background: var(--bg-elevated); border: 1px solid var(--border);
          border-radius: var(--r-xl); box-shadow: var(--shadow-lg); width: 90%;
          max-width: 720px; max-height: 80vh; display: flex; flex-direction: column;
          overflow: hidden; animation: slideUp .18s ease; }}
.modal.wide {{ max-width: 920px; }}
.modal-header {{ padding: var(--sp-4) var(--sp-5); border-bottom: 1px solid var(--border);
                display: flex; align-items: center; justify-content: space-between; }}
.modal-header h3 {{ font-size: 16px; font-weight: 600; color: var(--fg); }}
.modal-close {{ background: transparent; border: none; font-size: 22px; color: var(--fg-muted);
                cursor: pointer; line-height: 1; padding: 4px 8px; border-radius: var(--r-sm); }}
.modal-close:hover {{ background: var(--toolbar-bg); color: var(--fg); }}
.modal-body {{ padding: var(--sp-5); overflow-y: auto; flex: 1; }}
.modal-footer {{ padding: var(--sp-3) var(--sp-5); border-top: 1px solid var(--border);
                background: var(--toolbar-bg); display: flex; gap: var(--sp-2);
                justify-content: flex-end; }}
@keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes slideUp {{ from {{ opacity: 0; transform: translateY(8px); }}
                      to {{ opacity: 1; transform: translateY(0); }} }}
/* Form fields */
.field {{ margin-bottom: var(--sp-3); }}
.field label {{ display: block; font-size: 12px; font-weight: 500;
                color: var(--fg-muted); margin-bottom: var(--sp-1); }}
.field input, .field textarea, .field select {{ width: 100%; padding: 8px var(--sp-3);
                font-size: 14px; border: 1px solid var(--border); border-radius: var(--r-md);
                background: var(--bg); color: var(--fg); transition: border-color .12s; }}
.field input:focus, .field textarea:focus {{ outline: none; border-color: var(--accent);
                box-shadow: 0 0 0 3px var(--accent-light); }}
.field textarea {{ resize: vertical; min-height: 80px; font-family: var(--font-sans); }}
.btn-primary {{ background: var(--accent); color: var(--accent-fg); border: 1px solid var(--accent);
                padding: 8px 16px; font-size: 13px; font-weight: 500; border-radius: var(--r-md);
                cursor: pointer; transition: all .12s; }}
.btn-primary:hover {{ background: var(--accent-hover); border-color: var(--accent-hover); }}
.btn-secondary {{ background: var(--bg); color: var(--fg); border: 1px solid var(--border);
                  padding: 8px 16px; font-size: 13px; font-weight: 500; border-radius: var(--r-md);
                  cursor: pointer; transition: all .12s; }}
.btn-secondary:hover {{ background: var(--toolbar-bg); border-color: var(--border-strong); }}
/* Modal-specific content classes */
.modal-stream {{ font-family: var(--font-serif); font-size: 15px; line-height: 1.7;
                 padding: var(--sp-3); background: var(--toolbar-bg); border-radius: var(--r-md);
                 white-space: pre-wrap; min-height: 100px; max-height: 320px; overflow-y: auto; }}
.modal-sources {{ font-size: 12px; color: var(--fg-muted); margin-top: var(--sp-3);
                  padding: var(--sp-3); border-top: 1px solid var(--border); }}
.modal-sources .src-item {{ padding: 4px 0; }}
.catalog-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.catalog-table th {{ text-align: left; padding: 8px 12px; font-size: 11px; font-weight: 600;
                     color: var(--fg-muted); text-transform: uppercase; letter-spacing: 0.04em;
                     border-bottom: 1px solid var(--border); position: sticky; top: 0;
                     background: var(--bg-elevated); }}
.catalog-table td {{ padding: 10px 12px; border-bottom: 1px solid var(--border);
                     vertical-align: top; }}
.catalog-table tr:hover {{ background: var(--toolbar-bg); cursor: pointer; }}
.catalog-table .ct-title {{ font-weight: 600; color: var(--fg); }}
.catalog-table .ct-meta {{ font-size: 11px; color: var(--fg-muted); margin-top: 2px; }}
.catalog-pager {{ display: flex; align-items: center; gap: var(--sp-2);
                 justify-content: center; padding: var(--sp-3); font-size: 12px;
                 color: var(--fg-muted); }}
.catalog-pager button {{ padding: 4px 12px; font-size: 12px; background: var(--bg);
                         border: 1px solid var(--border); border-radius: var(--r-sm); cursor: pointer;
                         color: var(--fg); }}
.catalog-pager button:hover:not(:disabled) {{ background: var(--toolbar-bg); }}
.catalog-pager button:disabled {{ opacity: 0.3; cursor: not-allowed; }}
/* Score history viewer */
.scores-panel {{ display: none; margin-bottom: var(--sp-5); border: 1px solid var(--border);
                border-radius: var(--r-lg); overflow: hidden; background: var(--bg-elevated); }}
.scores-panel.open {{ display: block; }}
.scores-header {{ padding: var(--sp-3) var(--sp-4); background: var(--toolbar-bg);
                  border-bottom: 1px solid var(--border); display: flex;
                  align-items: center; justify-content: space-between; }}
.scores-header h4 {{ font-size: 13px; font-weight: 600; color: var(--fg); }}
.scores-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.scores-table th {{ padding: 8px 10px; font-size: 10px; text-transform: uppercase;
                    letter-spacing: 0.04em; color: var(--fg-muted); font-weight: 600;
                    text-align: right; border-bottom: 1px solid var(--border); }}
.scores-table th:first-child {{ text-align: left; }}
.scores-table td {{ padding: 8px 10px; text-align: right; font-variant-numeric: tabular-nums;
                    border-bottom: 1px solid var(--border); }}
.scores-table td:first-child {{ text-align: left; font-weight: 600; }}
.scores-table .score-good {{ color: var(--success); }}
.scores-table .score-mid {{ color: var(--warning); }}
.scores-table .score-low {{ color: var(--danger); }}
.scores-spark {{ padding: var(--sp-3) var(--sp-4); }}
.scores-spark svg {{ width: 100%; height: 60px; }}
.scores-empty {{ padding: var(--sp-5); text-align: center; color: var(--fg-muted);
                font-size: 13px; font-style: italic; }}
/* Stat cards (dashboard) */
.stat-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
              gap: var(--sp-3); margin-bottom: var(--sp-5); }}
.stat-tile {{ padding: var(--sp-4); background: var(--bg-elevated); border: 1px solid var(--border);
              border-radius: var(--r-lg); transition: all .15s; }}
.stat-tile:hover {{ box-shadow: var(--shadow-md); transform: translateY(-1px); }}
.stat-tile .num {{ font-size: 26px; font-weight: 700; color: var(--accent);
                  letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }}
.stat-tile .lbl {{ font-size: 11px; color: var(--fg-muted); margin-top: 2px;
                  text-transform: uppercase; letter-spacing: 0.04em; font-weight: 500; }}
.stat-tile .sub {{ font-size: 11px; color: var(--fg-faint); margin-top: 4px; }}
.raptor-bar {{ display: flex; gap: 4px; margin-top: var(--sp-2); }}
.raptor-bar .raptor-lvl {{ flex: 1; padding: 4px 6px; background: var(--toolbar-bg);
                          border-radius: var(--r-sm); font-size: 10px; text-align: center;
                          color: var(--fg-muted); }}
.raptor-bar .raptor-lvl strong {{ display: block; font-size: 13px; color: var(--accent); }}
/* Phase 14.2 — empty-state UX for chapters without drafts */
.ch-title.clickable {{ cursor: pointer; transition: color .12s; }}
.ch-title.clickable:hover {{ color: var(--accent); }}
.ch-group.selected .ch-title {{ color: var(--accent); }}
.ch-group.selected {{ background: var(--accent-light); border-radius: var(--r-sm); }}
[data-theme="dark"] .ch-group.selected {{ background: rgba(129, 140, 248, 0.08); }}
.sec-link.sec-empty-cta {{ color: var(--accent); font-style: normal; cursor: pointer;
                           font-weight: 500; opacity: 0.85; }}
.sec-link.sec-empty-cta:hover {{ background: var(--accent-light); opacity: 1; }}
.empty-state {{ padding: var(--sp-6) var(--sp-5); border: 1px dashed var(--border-strong);
                border-radius: var(--r-xl); background: var(--toolbar-bg); margin-top: var(--sp-3); }}
.empty-state h3 {{ font-size: 18px; font-weight: 600; margin-bottom: var(--sp-3);
                  color: var(--fg); }}
.empty-state p {{ font-family: var(--font-sans); font-size: 14px; line-height: 1.6;
                 color: var(--fg-muted); margin-bottom: var(--sp-3); text-align: left; }}
.empty-section-picker {{ display: flex; flex-wrap: wrap; gap: var(--sp-2); margin-top: var(--sp-4); }}
.section-chip {{ font-size: 12px; padding: 6px 14px; border: 1px solid var(--border);
                background: var(--bg); color: var(--fg); border-radius: 999px;
                cursor: pointer; transition: all .12s; }}
.section-chip:hover {{ border-color: var(--accent); color: var(--accent); }}
.section-chip.active {{ background: var(--accent); color: var(--accent-fg);
                        border-color: var(--accent); }}
.empty-hint {{ position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
              background: var(--bg-elevated); color: var(--fg); border: 1px solid var(--border);
              box-shadow: var(--shadow-lg); padding: var(--sp-3) var(--sp-4);
              border-radius: var(--r-lg); z-index: 2000; max-width: 600px;
              font-size: 13px; display: flex; align-items: center; gap: var(--sp-3);
              animation: slideUp .18s ease; }}
/* Phase 14.3 — chapter scope card inside the empty state */
.ch-scope {{ background: var(--bg); border: 1px solid var(--border);
            border-radius: var(--r-md); padding: var(--sp-3) var(--sp-4);
            margin: var(--sp-3) 0 var(--sp-4); }}
.ch-scope-row {{ display: flex; gap: var(--sp-3); padding: var(--sp-2) 0;
                font-size: 13px; align-items: flex-start; }}
.ch-scope-row + .ch-scope-row {{ border-top: 1px solid var(--border); }}
.ch-scope-label {{ font-weight: 600; color: var(--fg-muted); width: 100px;
                  text-transform: uppercase; font-size: 10px;
                  letter-spacing: 0.06em; padding-top: 2px; flex-shrink: 0; }}
.ch-scope-val {{ flex: 1; color: var(--fg); line-height: 1.5; }}
.ch-scope-val code {{ font-family: var(--font-mono); font-size: 12px;
                     padding: 1px 6px; background: var(--toolbar-bg);
                     border-radius: 3px; }}
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
.theme-toggle {{ position: fixed; bottom: 16px; right: 16px; background: var(--sidebar-bg);
                 color: var(--fg); border: 1px solid var(--border); padding: 8px 14px;
                 border-radius: 20px; cursor: pointer; font-size: 16px; z-index: 100;
                 display: flex; align-items: center; gap: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                 transition: all .2s; }}
.theme-toggle:hover {{ background: var(--accent); color: white; border-color: var(--accent); }}
.theme-toggle .label {{ font-size: 11px; }}
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
.heatmap .ch-label.clickable {{ cursor: pointer; transition: color .12s; }}
.heatmap .ch-label.clickable:hover {{ color: var(--accent); }}
.heatmap .ch-label-num {{ color: var(--fg-muted); font-weight: 500; margin-right: 4px; }}
.heatmap .ch-label-edit {{ opacity: 0; margin-left: 4px; transition: opacity .12s; color: var(--accent); }}
.heatmap .ch-label.clickable:hover .ch-label-edit {{ opacity: 1; }}
.heatmap th {{ font-size: 11px; text-transform: capitalize; }}
.hm-cell.off-template {{ background: transparent; color: var(--fg-faint); opacity: 0.4;
                         min-width: auto; padding: 4px 6px; cursor: default; }}
.hm-cell.empty.off-template {{ opacity: 0.2; }}
/* Phase 14.5 — heatmap header with inline link to the book plan */
.heatmap-header {{ display: flex; align-items: center; justify-content: space-between;
                   margin-bottom: 4px; gap: var(--sp-3); }}
.heatmap-header h3 {{ margin: 0; }}
.btn-link {{ background: transparent; border: 1px solid var(--border);
            color: var(--accent); padding: 4px 12px; font-size: 12px;
            font-weight: 500; border-radius: var(--r-md); cursor: pointer;
            transition: all .12s; display: inline-flex; align-items: center; gap: 4px; }}
.btn-link:hover {{ background: var(--accent-light); border-color: var(--accent); }}
/* Phase 15 — modal tabs (Wiki Query / Browse) */
.tabs {{ display: flex; gap: 0; padding: 0 var(--sp-5); border-bottom: 1px solid var(--border);
         background: var(--toolbar-bg); }}
.tab {{ background: transparent; border: none; padding: 10px var(--sp-4);
       font-size: 13px; font-weight: 500; color: var(--fg-muted); cursor: pointer;
       border-bottom: 2px solid transparent; margin-bottom: -1px;
       transition: all .12s; display: flex; align-items: center; gap: 6px; }}
.tab:hover {{ color: var(--fg); }}
.tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
.tab-pane {{ animation: fadeIn .15s ease; }}
/* Phase 15 — wiki page detail rendering */
.wiki-page-list {{ list-style: none; padding: 0; }}
.wiki-page-row {{ display: flex; align-items: center; gap: var(--sp-3);
                 padding: 10px var(--sp-3); border-bottom: 1px solid var(--border);
                 cursor: pointer; transition: background .12s; }}
.wiki-page-row:hover {{ background: var(--toolbar-bg); }}
.wiki-page-row .wp-title {{ flex: 1; font-weight: 500; color: var(--fg); }}
.wiki-page-row .wp-meta {{ font-size: 11px; color: var(--fg-muted);
                           white-space: nowrap; }}
.wiki-page-row .wp-type {{ font-size: 10px; padding: 2px 8px;
                          border-radius: 999px; background: var(--accent-light);
                          color: var(--accent); text-transform: uppercase;
                          letter-spacing: 0.04em; font-weight: 600; }}
.wiki-page-content {{ font-family: var(--font-serif); font-size: 15px;
                     line-height: 1.7; color: var(--fg); padding: var(--sp-3);
                     max-height: 60vh; overflow-y: auto;
                     background: var(--toolbar-bg); border-radius: var(--r-md);
                     border: 1px solid var(--border); }}
.wiki-page-content h1, .wiki-page-content h2, .wiki-page-content h3, .wiki-page-content h4 {{
    margin: var(--sp-4) 0 var(--sp-2); font-family: var(--font-sans);
    font-weight: 600; color: var(--fg); }}
.wiki-page-content h1 {{ font-size: 22px; }}
.wiki-page-content h2 {{ font-size: 18px; }}
.wiki-page-content h3 {{ font-size: 15px; }}
.wiki-page-content p {{ margin-bottom: var(--sp-3); }}
.wiki-page-content code {{ font-family: var(--font-mono); font-size: 13px;
                           padding: 1px 6px; background: var(--bg);
                           border-radius: 3px; }}
/* Phase 15 — live streaming stats footer (tok/s, elapsed, model) */
.stream-stats {{ display: flex; align-items: center; gap: var(--sp-3);
                font-family: var(--font-mono); font-size: 11px;
                color: var(--fg-muted); padding: var(--sp-2) var(--sp-3);
                background: var(--toolbar-bg); border-radius: var(--r-md);
                border: 1px solid var(--border); margin-top: var(--sp-2);
                min-height: 28px; }}
.stream-stats.idle {{ opacity: 0.4; }}
.stream-stats .ss-dot {{ width: 8px; height: 8px; border-radius: 50%;
                        background: var(--fg-faint); flex-shrink: 0; }}
.stream-stats.streaming .ss-dot {{ background: var(--success);
                                   animation: pulse 1.2s infinite; }}
.stream-stats.done .ss-dot {{ background: var(--success); }}
.stream-stats.error .ss-dot {{ background: var(--danger); }}
.stream-stats .ss-stat {{ display: flex; align-items: center; gap: 4px; }}
.stream-stats .ss-stat strong {{ color: var(--fg); font-weight: 600; }}
.stream-stats .ss-sep {{ color: var(--border-strong); }}
.stream-stats .ss-phase {{ padding: 1px 8px; border-radius: 999px;
                          background: var(--accent-light); color: var(--accent);
                          font-weight: 600; font-size: 10px;
                          text-transform: uppercase; letter-spacing: 0.04em; }}
.stream-cursor {{ display: inline-block; width: 6px; height: 14px;
                 background: var(--accent); margin-left: 2px;
                 animation: blink 1.1s steps(2, jump-none) infinite;
                 vertical-align: text-bottom; }}
@keyframes blink {{ 0%, 49% {{ opacity: 1; }} 50%, 100% {{ opacity: 0.15; }} }}
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
.editor-toolbar .autosave {{ font-size: 11px; margin-left: auto; line-height: 24px;
                              display: flex; align-items: center; gap: 6px;
                              padding: 2px 10px; border-radius: 999px;
                              background: var(--toolbar-bg); border: 1px solid var(--border); }}
.editor-toolbar .autosave .dot {{ width: 8px; height: 8px; border-radius: 50%;
                                   background: var(--success); display: inline-block; }}
.editor-toolbar .autosave.saving .dot {{ background: var(--warning); animation: pulse 1.2s infinite; }}
.editor-toolbar .autosave.unsaved .dot {{ background: var(--warning); }}
.editor-toolbar .autosave.error .dot {{ background: var(--danger); }}
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
.citation.verified-overstated {{ background: #ffedd5; border-radius: 3px; }}
.citation.verified-misrepresented {{ background: #fee2e2; border-radius: 3px; }}
[data-theme="dark"] .citation.verified-supported {{ background: #064e3b; }}
[data-theme="dark"] .citation.verified-extrapolated {{ background: #78350f; }}
[data-theme="dark"] .citation.verified-overstated {{ background: #7c2d12; }}
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
/* Corkboard */
.corkboard {{ display: flex; flex-wrap: wrap; gap: 12px; padding: 8px 0; }}
.cork-card {{ width: 180px; min-height: 140px; border: 1px solid var(--border); border-radius: 8px;
              padding: 10px 12px; background: var(--bg); cursor: pointer; transition: all .15s;
              display: flex; flex-direction: column; font-family: -apple-system, sans-serif; }}
.cork-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.1); transform: translateY(-2px); }}
.cork-card .cc-head {{ font-size: 11px; font-weight: 600; margin-bottom: 6px; display: flex;
                       justify-content: space-between; align-items: center; }}
.cork-card .cc-head .cc-ch {{ opacity: 0.5; }}
.cork-card .cc-type {{ font-size: 13px; font-weight: bold; margin-bottom: 4px; }}
.cork-card .cc-summary {{ font-size: 11px; opacity: 0.6; flex: 1; overflow: hidden;
                          display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; }}
.cork-card .cc-footer {{ font-size: 10px; opacity: 0.4; margin-top: 6px; }}
.cork-card .cc-status {{ font-size: 9px; padding: 1px 6px; border-radius: 8px; color: white;
                         text-transform: uppercase; font-weight: 600; }}
.cc-status.to_do {{ background: var(--border); color: var(--fg); }}
.cc-status.drafted {{ background: var(--warning); }}
.cc-status.reviewed {{ background: var(--accent); }}
.cc-status.revised {{ background: #8b5cf6; }}
.cc-status.final {{ background: var(--success); }}
/* Chapter reader */
.reader-view {{ max-width: 800px; }}
.reader-section-title {{ font-size: 20px; margin: 32px 0 16px; padding-bottom: 8px;
                          border-bottom: 2px solid var(--accent); color: var(--accent); }}
/* Snapshot list */
.snap-list {{ font-size: 12px; font-family: -apple-system, sans-serif; }}
.snap-item {{ display: flex; justify-content: space-between; align-items: center;
              padding: 4px 0; border-bottom: 1px solid var(--border); }}
.snap-item button {{ font-size: 10px; padding: 2px 8px; border: 1px solid var(--border);
                     border-radius: 4px; cursor: pointer; background: var(--bg); color: var(--fg); }}
.snap-item button:hover {{ background: var(--accent-light); }}
/* Status selector */
.status-select {{ font-size: 11px; padding: 2px 6px; border: 1px solid var(--border);
                   border-radius: 4px; background: var(--bg); color: var(--fg); cursor: pointer;
                   margin-left: 8px; }}
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
    <select class="status-select" id="status-select" onchange="updateStatus(this.value)">
      <option value="to_do">To Do</option>
      <option value="drafted" selected>Drafted</option>
      <option value="reviewed">Reviewed</option>
      <option value="revised">Revised</option>
      <option value="final">Final</option>
    </select>
  </div>

  <!-- Action Toolbar — Phase 14 v2 grouped layout -->
  <div class="toolbar" id="toolbar">
    <div class="tg">
      <button class="primary" onclick="doAutowrite()" title="Autonomous write → review → revise loop">&#9889; Autowrite</button>
      <button onclick="doWrite()" title="Draft this section from scratch">Write</button>
      <button onclick="doReview()" title="Run a critic pass on this section">Review</button>
      <button onclick="doRevise()" title="Revise based on review feedback">Revise</button>
    </div>
    <div class="sep"></div>
    <div class="tg">
      <button onclick="doVerify()" title="Verify citations against sources (Phases 7+11)">&#10003; Verify</button>
      <button onclick="showScoresPanel()" title="Phase 13 — convergence trajectory for autowrite drafts">&#9783; Scores</button>
      <button onclick="promptArgue()" title="Map evidence for/against a claim">Argue</button>
      <button onclick="doGaps()" title="Analyse gaps in the book">Gaps</button>
    </div>
    <div class="sep"></div>
    <div class="tg">
      <button onclick="openPlanModal()" title="View / edit / regenerate the book plan (the leitmotiv)">&#128221; Plan</button>
      <button onclick="openAskModal()" title="Full corpus RAG question (sciknow ask question)">&#128270; Ask Corpus</button>
      <button onclick="openWikiModal()" title="Query the compiled knowledge wiki (sciknow wiki query)">&#128218; Wiki Query</button>
      <button onclick="openCatalogModal()" title="Browse the paper catalog (sciknow catalog list)">&#128194; Browse Papers</button>
    </div>
    <div class="sep"></div>
    <div class="tg">
      <button onclick="showVersions()" title="View version history and diffs">History</button>
      <button onclick="takeSnapshot()" title="Save a snapshot of current content">Snapshot</button>
      <button onclick="showCorkboard()" title="Visual card-based view">Corkboard</button>
      <button onclick="showChapterReader()" title="Read entire chapter as continuous scroll">Read</button>
      <button onclick="showDashboard()" title="Book dashboard with stats + heatmap">Dashboard</button>
    </div>
  </div>

  <!-- Phase 13 — Score history panel (collapsible, lazy-loaded) -->
  <div class="scores-panel" id="scores-panel">
    <div class="scores-header">
      <h4>Convergence trajectory</h4>
      <button class="modal-close" onclick="document.getElementById('scores-panel').classList.remove('open')">&times;</button>
    </div>
    <div id="scores-panel-body"></div>
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
    <div id="main-stream-stats" class="stream-stats" style="margin: 0 14px 12px;"></div>
  </div>

  <div id="read-view">{content_html}</div>

  <div id="edit-view" style="display:none;">
    <div class="editor-toolbar">
      <button onclick="edInsert('**','**')" title="Bold"><b>B</b></button>
      <button onclick="edInsert('*','*')" title="Italic"><i>I</i></button>
      <button onclick="edInsert('## ','')" title="Heading">H2</button>
      <button onclick="edInsert('### ','')" title="Subheading">H3</button>
      <button onclick="edInsertCite()" title="Citation">[N]</button>
      <span class="autosave" id="autosave-status" title="Autosaves every 5 seconds while editing">
        <span class="dot"></span><span id="autosave-text">Autosave on</span>
      </span>
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

<!-- ── Phase 14 modals ─────────────────────────────────────────────────── -->

<!-- Wiki Modal — Phase 15: Query + Browse tabs -->
<div class="modal-overlay" id="wiki-modal" onclick="if(event.target===this)closeModal('wiki-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3>&#128218; Compiled Knowledge Wiki</h3>
      <button class="modal-close" onclick="closeModal('wiki-modal')">&times;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="wiki-query" onclick="switchWikiTab('wiki-query')">&#128270; Query</button>
      <button class="tab" data-tab="wiki-browse" onclick="switchWikiTab('wiki-browse')">&#128194; Browse pages</button>
    </div>
    <div class="modal-body">
      <!-- Query tab -->
      <div class="tab-pane active" id="wiki-query-pane">
        <div class="field">
          <label>Question</label>
          <input type="text" id="wiki-query-input" placeholder="What does the wiki say about ..."
                 onkeydown="if(event.key==='Enter')doWikiQuery()">
        </div>
        <div class="field">
          <button class="btn-primary" onclick="doWikiQuery()">Search Wiki</button>
        </div>
        <div id="wiki-status" style="font-size:12px;color:var(--fg-muted);margin-bottom:8px;"></div>
        <div class="modal-stream" id="wiki-stream"></div>
        <div id="wiki-stream-stats" class="stream-stats"></div>
        <div class="modal-sources" id="wiki-sources" style="display:none;"></div>
      </div>
      <!-- Browse tab -->
      <div class="tab-pane" id="wiki-browse-pane" style="display:none;">
        <div class="field" style="display:flex;gap:8px;align-items:flex-end;">
          <div style="flex:1;">
            <label>Filter by type</label>
            <select id="wiki-type-filter" onchange="loadWikiPages(1)">
              <option value="">All types</option>
            </select>
          </div>
          <button class="btn-secondary" onclick="loadWikiPages(1)">Refresh</button>
        </div>
        <div id="wiki-browse-list" style="margin-top:12px;"></div>
        <!-- Detail view (hidden until a page is opened) -->
        <div id="wiki-page-detail" style="display:none;">
          <button class="btn-secondary" style="margin-bottom:12px;" onclick="closeWikiPageDetail()">&larr; Back to list</button>
          <div id="wiki-page-meta" style="font-size:11px;color:var(--fg-muted);margin-bottom:12px;"></div>
          <div id="wiki-page-content" class="wiki-page-content"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Corpus Ask Modal -->
<div class="modal-overlay" id="ask-modal" onclick="if(event.target===this)closeModal('ask-modal')">
  <div class="modal">
    <div class="modal-header">
      <h3>&#128270; Ask the Corpus (RAG)</h3>
      <button class="modal-close" onclick="closeModal('ask-modal')">&times;</button>
    </div>
    <div class="modal-body">
      <div class="field">
        <label>Question</label>
        <input type="text" id="ask-input" placeholder="What are the main mechanisms of ..."
               onkeydown="if(event.key==='Enter')doAsk()">
      </div>
      <div class="field" style="display:flex;gap:8px;">
        <div style="flex:1;">
          <label>Year from</label>
          <input type="number" id="ask-year-from" placeholder="(optional)">
        </div>
        <div style="flex:1;">
          <label>Year to</label>
          <input type="number" id="ask-year-to" placeholder="(optional)">
        </div>
      </div>
      <div class="field">
        <button class="btn-primary" onclick="doAsk()">Ask</button>
        <span style="font-size:11px;color:var(--fg-muted);margin-left:8px;">Hybrid retrieval + bge-reranker + LLM</span>
      </div>
      <div id="ask-status" style="font-size:12px;color:var(--fg-muted);margin-bottom:8px;"></div>
      <div class="modal-stream" id="ask-stream"></div>
      <div id="ask-stream-stats" class="stream-stats"></div>
      <div class="modal-sources" id="ask-sources" style="display:none;"></div>
    </div>
  </div>
</div>

<!-- Phase 14.3 — Book Plan Modal -->
<div class="modal-overlay" id="plan-modal" onclick="if(event.target===this)closeModal('plan-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3>&#128221; Book Plan &mdash; the leitmotiv</h3>
      <button class="modal-close" onclick="closeModal('plan-modal')">&times;</button>
    </div>
    <div class="modal-body">
      <p style="font-size:12px;color:var(--fg-muted);margin-bottom:12px;">
        The book plan is a 200&ndash;500 word document defining the central thesis,
        scope, intended audience, and key terms. It is injected into every
        <code>book write</code> / <code>autowrite</code> call so all chapters stay
        aligned with the same argument. Edit it manually below, or click
        <strong>Regenerate with LLM</strong> to draft a new one from your chapters
        and paper corpus.
      </p>
      <div class="field">
        <label>Book title</label>
        <input type="text" id="plan-title-input">
      </div>
      <div class="field">
        <label>Short description (one or two sentences)</label>
        <textarea id="plan-desc-input" style="min-height:50px;"></textarea>
      </div>
      <div class="field">
        <label>Plan / leitmotiv (the full thesis &amp; scope document)</label>
        <textarea id="plan-text-input" style="min-height:280px;font-family:var(--font-serif);font-size:14px;line-height:1.6;"></textarea>
      </div>
      <div id="plan-status" style="font-size:12px;color:var(--fg-muted);margin-bottom:8px;"></div>
      <div id="plan-stream-stats" class="stream-stats"></div>
    </div>
    <div class="modal-footer">
      <button class="btn-secondary" onclick="closeModal('plan-modal')">Close</button>
      <button class="btn-secondary" onclick="regeneratePlan()">&#9889; Regenerate with LLM</button>
      <button class="btn-primary" onclick="savePlan()">Save</button>
    </div>
  </div>
</div>

<!-- Phase 14.3 — Chapter Info Modal (description + topic_query) -->
<div class="modal-overlay" id="chapter-modal" onclick="if(event.target===this)closeModal('chapter-modal')">
  <div class="modal">
    <div class="modal-header">
      <h3>&#9881; Chapter scope</h3>
      <button class="modal-close" onclick="closeModal('chapter-modal')">&times;</button>
    </div>
    <div class="modal-body">
      <p style="font-size:12px;color:var(--fg-muted);margin-bottom:12px;">
        The chapter description sets the per-chapter scope: what the chapter
        covers and what stays out. The topic query is a 3&ndash;6 word search
        phrase used to retrieve the most relevant papers from the corpus when
        you click Write or Autowrite for this chapter.
      </p>
      <div class="field">
        <label>Chapter title</label>
        <input type="text" id="ch-title-input">
      </div>
      <div class="field">
        <label>Description (per-chapter scope)</label>
        <textarea id="ch-desc-input" style="min-height:120px;"></textarea>
      </div>
      <div class="field">
        <label>Topic query (retrieval phrase)</label>
        <input type="text" id="ch-tq-input" placeholder="e.g. solar irradiance satellite measurements">
      </div>
      <div id="chapter-modal-status" style="font-size:12px;color:var(--fg-muted);margin-bottom:8px;"></div>
    </div>
    <div class="modal-footer">
      <button class="btn-secondary" onclick="closeModal('chapter-modal')">Close</button>
      <button class="btn-primary" onclick="saveChapterInfo()">Save</button>
    </div>
  </div>
</div>

<!-- Catalog Browser Modal -->
<div class="modal-overlay" id="catalog-modal" onclick="if(event.target===this)closeModal('catalog-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3>&#128194; Browse Papers</h3>
      <button class="modal-close" onclick="closeModal('catalog-modal')">&times;</button>
    </div>
    <div class="modal-body">
      <div class="field" style="display:flex;gap:8px;align-items:flex-end;">
        <div style="flex:2;">
          <label>Author</label>
          <input type="text" id="cat-author" placeholder="(any)">
        </div>
        <div style="flex:2;">
          <label>Journal</label>
          <input type="text" id="cat-journal" placeholder="(any)">
        </div>
        <div style="flex:1;">
          <label>Year from</label>
          <input type="number" id="cat-year-from">
        </div>
        <div style="flex:1;">
          <label>Year to</label>
          <input type="number" id="cat-year-to">
        </div>
        <button class="btn-primary" onclick="loadCatalog(1)">Filter</button>
      </div>
      <div id="catalog-results" style="margin-top:12px;"></div>
    </div>
  </div>
</div>

<button class="theme-toggle" onclick="toggleTheme()" id="theme-btn">
  <span id="theme-icon">&#9788;</span>
  <span class="label" id="theme-label">Light</span>
</button>

<script>
// Phase 15.4 — page-load tag visible in DevTools Console.
// If you don't see this line in the console, your browser is running
// stale JS — hard-refresh with Ctrl+Shift+R (or Cmd+Shift+R on macOS).
console.log('[sciknow] web reader loaded · build phase-15.4');

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
  updateThemeButton();
}}
function updateThemeButton() {{
  const isDark = document.documentElement.dataset.theme === 'dark';
  document.getElementById('theme-icon').innerHTML = isDark ? '&#9790;' : '&#9788;';
  document.getElementById('theme-label').textContent = isDark ? 'Dark' : 'Light';
}}
if (localStorage.getItem('theme')) {{
  document.documentElement.dataset.theme = localStorage.getItem('theme');
}}
updateThemeButton();

// ── SPA Navigation ────────────────────────────────────────────────────────
function navTo(el) {{
  const draftId = el.dataset.draftId;
  if (!draftId) return true;  // fallback to normal navigation
  loadSection(draftId);
  return false;  // prevent default <a> navigation
}}

// Phase 14.2 — empty-state UX: select a chapter without needing a draft.
function selectChapter(chGroupEl) {{
  if (!chGroupEl) return;
  const chId = chGroupEl.dataset.chId;
  if (!chId) return;
  currentChapterId = chId;
  currentDraftId = '';
  currentSectionType = '';
  // Visual highlight
  document.querySelectorAll('.ch-group').forEach(g => g.classList.remove('selected'));
  chGroupEl.classList.add('selected');
  document.querySelectorAll('.sec-link.active').forEach(l => l.classList.remove('active'));
  // Look up the chapter title for the breadcrumb
  const chTitleEl = chGroupEl.querySelector('.ch-title');
  const chLabel = chTitleEl ? chTitleEl.textContent.replace(/\\s*\u2717\\s*$/, '').trim() : 'Chapter';
  // Show the empty state in the main area
  showChapterEmptyState(chLabel, chId);
}}

function showChapterEmptyState(chLabel, chId) {{
  document.getElementById('draft-title').textContent = chLabel;
  const subtitle = document.getElementById('draft-subtitle');
  subtitle.innerHTML = '<span style="color:var(--fg-muted);">No drafts yet &mdash; pick a section type and click Write, or use Autowrite to draft all sections.</span>';
  subtitle.style.display = 'block';
  // Show toolbar
  document.getElementById('toolbar').style.display = 'flex';
  // Hide other panels
  document.getElementById('dashboard-view').style.display = 'none';
  document.getElementById('edit-view').style.display = 'none';
  document.getElementById('version-panel').style.display = 'none';
  document.getElementById('stream-panel').style.display = 'none';
  document.getElementById('scores-panel').classList.remove('open');

  // Phase 14.3 — look up the chapter's saved scope (description + topic_query)
  const ch = chaptersData.find(c => c.id === chId);
  const desc = (ch && ch.description) ? ch.description : '';
  const tq = (ch && ch.topic_query) ? ch.topic_query : '';

  // Phase 14.4 — use the chapter's actual book-style section template,
  // not a hardcoded paper-style list. Falls back to book defaults if the
  // chapter has no template (which should never happen post-14.4 since
  // the backend always returns at least _DEFAULT_BOOK_SECTIONS).
  const sections = (ch && Array.isArray(ch.sections_template) && ch.sections_template.length)
    ? ch.sections_template
    : ['overview', 'key_evidence', 'current_understanding', 'open_questions', 'summary'];
  let html = '<div class="empty-state">';
  html += '<h3>Start writing this chapter</h3>';

  // Phase 14.3 — show the chapter scope right here in the empty state
  html += '<div class="ch-scope">';
  html += '<div class="ch-scope-row"><span class="ch-scope-label">Scope</span><div class="ch-scope-val">' +
          (desc ? desc.replace(/</g, '&lt;') : '<em style="color:var(--fg-faint);">No description set. Click Edit chapter scope to add one.</em>') +
          '</div></div>';
  html += '<div class="ch-scope-row"><span class="ch-scope-label">Topic query</span><div class="ch-scope-val">' +
          (tq ? '<code>' + tq.replace(/</g, '&lt;') + '</code>' : '<em style="color:var(--fg-faint);">Not set &mdash; the chapter title will be used as the retrieval query.</em>') +
          '</div></div>';
  html += '<button class="btn-secondary" style="margin-top:8px;font-size:12px;" onclick="openChapterModal(&#39;' + chId + '&#39;)">&#9881; Edit chapter scope</button>';
  html += '</div>';

  html += '<p>Once the scope feels right, choose a section type below and click <strong>Write</strong> in the toolbar, or click <strong>Autowrite</strong> to draft a section autonomously with the convergence loop.</p>';
  html += '<div class="empty-section-picker">';
  sections.forEach(s => {{
    const active = s === currentSectionType ? ' active' : '';
    html += '<button class="section-chip' + active + '" onclick="setSectionType(&#39;' + s + '&#39;, this)">' + s.replace(/_/g, ' ') + '</button>';
  }});
  html += '</div>';
  html += '<p style="margin-top:16px;font-size:12px;color:var(--fg-muted);">Tip: you can also explore the corpus without writing anything &mdash; use <strong>Ask Corpus</strong>, <strong>Wiki Query</strong>, or <strong>Browse Papers</strong>. The book&#39;s overall <strong>&#128221; Plan</strong> is also editable from the toolbar.</p>';
  html += '</div>';
  document.getElementById('read-view').innerHTML = html;
  document.getElementById('read-view').style.display = 'block';
}}

function setSectionType(t, btn) {{
  currentSectionType = t;
  document.querySelectorAll('.section-chip').forEach(c => c.classList.remove('active'));
  if (btn) btn.classList.add('active');
}}

function startWritingChapter(chId) {{
  // Find the chapter group and select it, then prompt to write the first section.
  const grp = document.querySelector('[data-ch-id="' + chId + '"]');
  if (grp) selectChapter(grp);
  if (!currentSectionType) currentSectionType = 'overview';
  // Auto-trigger write
  doWrite();
}}

// Phase 14.2 — show an inline guidance toast instead of a JS alert.
// Replaces the silent-failing alert() calls in doWrite/doReview/etc.
function showEmptyHint(html) {{
  let hint = document.getElementById('empty-hint');
  if (!hint) {{
    hint = document.createElement('div');
    hint.id = 'empty-hint';
    hint.className = 'empty-hint';
    document.body.appendChild(hint);
  }}
  hint.innerHTML = html + '<button onclick="document.getElementById(&#39;empty-hint&#39;).remove()" style="margin-left:12px;background:transparent;border:1px solid var(--border);color:var(--fg);padding:2px 8px;border-radius:4px;cursor:pointer;font-size:11px;">Dismiss</button>';
  // Auto-dismiss after 6s
  if (hint._timer) clearTimeout(hint._timer);
  hint._timer = setTimeout(() => {{
    if (hint && hint.parentElement) hint.remove();
  }}, 6000);
}}

async function loadSection(draftId) {{
  try {{
    const res = await fetch('/api/section/' + draftId);
    if (!res.ok) return;
    const data = await res.json();

    currentDraftId = data.id;
    currentChapterId = data.chapter_id || '';
    currentSectionType = data.section_type || '';

    // Restore section view (if coming from dashboard)
    document.getElementById('dashboard-view').style.display = 'none';
    document.getElementById('read-view').style.display = 'block';
    document.getElementById('draft-subtitle').style.display = 'block';
    document.getElementById('toolbar').style.display = 'flex';
    document.getElementById('argue-map-view').style.display = 'none';

    // Update main content
    document.getElementById('draft-title').textContent = data.title;
    document.getElementById('draft-version').textContent = data.version;
    document.getElementById('draft-words').textContent = data.word_count;
    document.getElementById('read-view').innerHTML = data.content_html;
    document.getElementById('edit-view').style.display = 'none';

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

    // Hide panels
    document.getElementById('stream-panel').style.display = 'none';
    document.getElementById('version-panel').style.display = 'none';

    // Build citation popovers + update status selector
    setTimeout(buildPopovers, 100);
    if (data.status) document.getElementById('status-select').value = data.status;
  }} catch(e) {{
    console.error('Navigation failed:', e);
  }}
}}

// Handle browser back/forward
window.addEventListener('popstate', function(e) {{
  if (e.state && e.state.draftId) loadSection(e.state.draftId);
}});

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

  // Phase 15 — live stats footer for the main stream panel
  const stats = createStreamStats('main-stream-stats', 'qwen3.5:27b');
  stats.start();

  source.onmessage = function(e) {{
    const evt = JSON.parse(e.data);

    if (evt.type === 'token') {{
      setStreamCursor(body, false);
      body.innerHTML += evt.text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
      setStreamCursor(body, true);
      body.scrollTop = body.scrollHeight;
      stats.update(evt.text);
    }}
    else if (evt.type === 'progress') {{
      status.textContent = evt.detail || evt.stage;
    }}
    else if (evt.type === 'scores') {{
      scoresEl.style.display = 'block';
      const s = evt.scores;
      const dims = ['groundedness', 'completeness', 'coherence', 'citation_accuracy', 'hedging_fidelity', 'overall'];
      scoresEl.innerHTML = 'Iteration ' + evt.iteration + ': ' + dims.map(d => {{
        const v = (s[d] || 0).toFixed(2);
        const cls = v >= 0.85 ? 'good' : v >= 0.7 ? 'mid' : 'low';
        return '<span class="score-bar"><span class="label">' + d.slice(0,6) + '</span> ' +
               '<span class="value ' + cls + '">' + v + '</span></span>';
      }}).join('');
    }}
    else if (evt.type === 'cove_verification') {{
      // Phase 11 — Chain-of-Verification (decoupled fact check). Only fires
      // when standard groundedness or hedging_fidelity is below threshold.
      const cd = evt.data || {{}};
      const score = (cd.cove_score != null) ? cd.cove_score.toFixed(2) : '?';
      const mismatches = cd.mismatches || [];
      const high = mismatches.filter(m => m.severity === 'high');
      const med = mismatches.filter(m => m.severity === 'medium');
      let html = '<div style="margin:10px 0;padding:8px;border-left:3px solid var(--warning);background:var(--toolbar-bg);">';
      html += '<div style="font-weight:bold;">Chain-of-Verification: ' + score + '</div>';
      if (high.length || med.length) {{
        html += '<div style="font-size:12px;opacity:0.8;">' +
                '<span style="color:var(--danger);">' + high.length + ' NOT_IN_SOURCES</span> · ' +
                '<span style="color:var(--warning);">' + med.length + ' DIFFERENT_SCOPE</span></div>';
      }}
      html += '</div>';
      body.innerHTML += html;
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
    else if (evt.type === 'model_info') {{
      // Phase 15.5 — model name now shown only in the stats footer (which
      // already pulls it via setModel below). The previous in-body line
      // duplicated the same info redundantly.
      stats.setModel(evt.writer_model || 'qwen3.5:27b');
    }}
    else if (evt.type === 'checkpoint') {{
      // Phase 15.1 — incremental save reached. Briefly note in the body.
      body.innerHTML += '<div style="font-size:11px;color:var(--success);padding:4px 0;">' +
        '\\u2693 checkpoint saved · ' + (evt.stage || '') + ' · ' +
        (evt.word_count || 0) + ' words</div>';
    }}
    else if (evt.type === 'completed') {{
      status.textContent = 'Done';
      stats.done('done');
      setStreamCursor(body, false);
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
      stats.done('error');
      setStreamCursor(body, false);
      hideStreamPanel();
      source.close();
      currentEventSource = null;
      currentJobId = null;
    }}
    else if (evt.type === 'done') {{
      stats.done('done');
      setStreamCursor(body, false);
      hideStreamPanel();
      source.close();
      currentEventSource = null;
      currentJobId = null;
    }}
  }};

  source.onerror = function() {{
    status.textContent = 'Connection lost';
    stats.done('error');
    setStreamCursor(body, false);
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
  if (!currentChapterId) {{
    showEmptyHint('Select a chapter from the sidebar first &mdash; click any chapter title in the left panel, then come back and click Write.');
    return;
  }}
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
  if (!currentDraftId) {{ showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft."); return; }}
  showStreamPanel('Reviewing...');

  const fd = new FormData();
  const res = await fetch('/api/review/' + currentDraftId, {{method: 'POST', body: fd}});
  const data = await res.json();
  startStream(data.job_id);
}}

async function doRevise() {{
  if (!currentDraftId) {{ showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft."); return; }}
  const instruction = prompt('Revision instruction (leave empty to use review feedback):');
  if (instruction === null) return;  // cancelled
  showStreamPanel('Revising...');

  const fd = new FormData();
  fd.append('instruction', instruction);
  const res = await fetch('/api/revise/' + currentDraftId, {{method: 'POST', body: fd}});
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
  const [res, statsRes] = await Promise.all([
    fetch('/api/dashboard'),
    fetch('/api/stats').catch(() => null),
  ]);
  const data = await res.json();
  const corpusStats = statsRes ? await statsRes.json().catch(() => null) : null;
  const s = data.stats;

  let html = '<div class="dashboard">';
  html += '<h2>Book Dashboard</h2>';

  // Book stats — modernized stat-tile cards
  html += '<div class="stat-grid">';
  html += '<div class="stat-tile"><div class="num">' + s.total_words.toLocaleString() + '</div><div class="lbl">Words</div></div>';
  html += '<div class="stat-tile"><div class="num">' + s.chapters + '</div><div class="lbl">Chapters</div></div>';
  html += '<div class="stat-tile"><div class="num">' + s.drafts + '</div><div class="lbl">Drafts</div></div>';
  html += '<div class="stat-tile"><div class="num">' + s.gaps_open + '</div><div class="lbl">Open Gaps</div></div>';
  html += '<div class="stat-tile"><div class="num">' + s.comments + '</div><div class="lbl">Comments</div></div>';
  html += '</div>';

  // Phase 14 — Corpus stats panel (mirrors `db stats` + RAPTOR + topics)
  if (corpusStats) {{
    html += '<h3 style="margin:24px 0 12px;font-size:14px;font-weight:600;color:var(--fg-muted);text-transform:uppercase;letter-spacing:0.04em;">Corpus</h3>';
    html += '<div class="stat-grid">';
    html += '<div class="stat-tile"><div class="num">' + (corpusStats.n_documents || 0).toLocaleString() + '</div><div class="lbl">Documents</div>';
    if (corpusStats.n_completed != null) {{
      html += '<div class="sub">' + corpusStats.n_completed.toLocaleString() + ' complete</div>';
    }}
    html += '</div>';
    html += '<div class="stat-tile"><div class="num">' + (corpusStats.n_chunks || 0).toLocaleString() + '</div><div class="lbl">Chunks</div></div>';
    html += '<div class="stat-tile"><div class="num">' + (corpusStats.n_citations || 0).toLocaleString() + '</div><div class="lbl">Citations</div></div>';
    if (corpusStats.n_wiki_pages) {{
      html += '<div class="stat-tile"><div class="num">' + corpusStats.n_wiki_pages.toLocaleString() + '</div><div class="lbl">Wiki Pages</div></div>';
    }}
    if (corpusStats.topic_clusters && corpusStats.topic_clusters.length) {{
      html += '<div class="stat-tile"><div class="num">' + corpusStats.topic_clusters.length + '</div><div class="lbl">Topic Clusters</div>';
      html += '<div class="sub">' + corpusStats.topic_clusters[0].name + ' (' + corpusStats.topic_clusters[0].n + ')</div></div>';
    }}
    if (corpusStats.raptor_levels && Object.keys(corpusStats.raptor_levels).length) {{
      const totalNodes = Object.values(corpusStats.raptor_levels).reduce((a, b) => a + b, 0);
      html += '<div class="stat-tile"><div class="num">' + totalNodes.toLocaleString() + '</div><div class="lbl">RAPTOR Nodes</div>';
      html += '<div class="raptor-bar">';
      Object.entries(corpusStats.raptor_levels).forEach(([lvl, n]) => {{
        html += '<div class="raptor-lvl"><strong>' + n.toLocaleString() + '</strong>' + lvl + '</div>';
      }});
      html += '</div></div>';
    }} else {{
      html += '<div class="stat-tile" style="opacity:0.6;"><div class="num">—</div><div class="lbl">RAPTOR</div><div class="sub">Not built</div></div>';
    }}
    html += '</div>';
  }}

  // Heatmap — Phase 14.4 — book-style section columns from real data
  // Phase 14.5 — heatmap header now includes a Plan link so the leitmotiv
  // is one click away from the dashboard.
  html += '<div class="heatmap-header">';
  html += '<h3>Completion Heatmap</h3>';
  html += '<button class="btn-link" onclick="openPlanModal()" title="View, edit, or regenerate the book plan (the leitmotiv)">&#128221; Book Plan</button>';
  html += '</div>';
  html += '<p style="font-size:11px;color:var(--fg-muted);margin-bottom:6px;">Click a chapter title to edit its scope. Click an empty cell to write that section. Click a filled cell to open the draft. The <strong>&#128221; Book Plan</strong> link above opens the leitmotiv editor.</p>';
  html += '<table class="heatmap"><thead><tr><th></th>';
  data.section_types.forEach(st => {{
    // Show the full section name (replacing underscores with spaces) but
    // keep the column compact via CSS rotation if needed.
    const display = st.replace(/_/g, ' ');
    html += '<th title="' + display + '">' + display + '</th>';
  }});
  html += '</tr></thead><tbody>';
  data.heatmap.forEach(row => {{
    // Phase 14.4 — chapter title is now clickable to open the chapter
    // scope modal (where you can edit title / description / topic_query).
    html += '<tr><td class="ch-label clickable" onclick="openChapterModal(&#39;' + row.id + '&#39;)" title="Click to edit chapter title and scope">';
    html += '<span class="ch-label-num">Ch.' + row.num + '</span> ' + row.title.replace(/</g, '&lt;').substring(0, 36);
    html += ' <span class="ch-label-edit">&#9881;</span></td>';
    // Per-chapter section template — highlight cells whose section type
    // is in this chapter's template, dim cells that are out-of-template.
    const tmpl = (row.sections_template && row.sections_template.length)
      ? new Set(row.sections_template) : null;
    row.cells.forEach(cell => {{
      const inTemplate = !tmpl || tmpl.has(cell.type);
      const cls = inTemplate ? '' : ' off-template';
      if (cell.status === 'empty') {{
        if (inTemplate) {{
          html += '<td><span class="hm-cell empty' + cls + '" onclick="writeForCell(&#39;' + row.id + '&#39;,&#39;' + cell.type + '&#39;)" title="Click to write ' + cell.type.replace(/_/g, ' ') + '">+</span></td>';
        }} else {{
          html += '<td><span class="hm-cell off-template" title="Not in this chapter\\u2019s section template">·</span></td>';
        }}
      }} else {{
        const label = 'v' + cell.version + ' ' + cell.words + 'w';
        html += '<td><span class="hm-cell ' + cell.status + cls + '" onclick="loadSection(&#39;' + cell.draft_id + '&#39;)" title="' + label + '">' + label + '</span></td>';
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
        const cmdHint = 'Run: sciknow db expand -q ' + g.description.substring(0,30).replace(/[&<>"\\']/g, '');
        btn = '<button data-cmd="' + cmdHint.replace(/&/g, '&amp;').replace(/"/g, '&quot;') + '" onclick="alert(this.dataset.cmd)">Expand</button>';
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


// ── Version History ───────────────────────────────────────────────────────
let versionData = [];
let selectedVersions = [];

async function showVersions() {{
  if (!currentDraftId) {{ showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft."); return; }}
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
    html += '<span class="version-badge" data-vid="' + v.id + '" onclick="selectVersion(&#39;' + v.id + '&#39;)">' +
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
  // Phase 14.4 — flag unsaved changes immediately so the user sees the
  // amber dot the moment they type, not after the next 5s autosave tick.
  if (ta.value !== _currentRaw) setAutosaveState('unsaved', 'Unsaved changes');
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

// Phase 14.4 — autosave UX: a small status pill that's always visible
// while in edit mode. Colours: green=saved, amber-pulse=saving,
// amber-static=unsaved, red=error.
function setAutosaveState(state, label) {{
  const el = document.getElementById('autosave-status');
  if (!el) return;
  el.classList.remove('saving', 'unsaved', 'error');
  if (state) el.classList.add(state);
  const text = document.getElementById('autosave-text');
  if (text) text.textContent = label;
}}

async function edSave() {{
  const ta = document.getElementById('edit-area');
  setAutosaveState('saving', 'Saving...');
  try {{
    const fd = new FormData();
    fd.append('content', ta.value);
    await fetch('/edit/' + currentDraftId, {{method: 'POST', body: fd}});
    setAutosaveState('', 'Saved');
  }} catch (e) {{
    setAutosaveState('error', 'Save failed');
    return;
  }}
  if (_autosaveTimer) {{ clearInterval(_autosaveTimer); _autosaveTimer = null; }}
  toggleEdit();
  loadSection(currentDraftId);
}}

async function edAutosave() {{
  const ta = document.getElementById('edit-area');
  if (ta.value === _currentRaw) return;
  setAutosaveState('saving', 'Saving...');
  try {{
    _currentRaw = ta.value;
    const fd = new FormData();
    fd.append('content', ta.value);
    await fetch('/edit/' + currentDraftId, {{method: 'POST', body: fd}});
    const t = new Date().toLocaleTimeString([], {{hour: '2-digit', minute: '2-digit'}});
    setAutosaveState('', 'Saved at ' + t);
  }} catch (e) {{
    setAutosaveState('error', 'Save failed');
  }}
}}

// ── Claim Verification (Phase 5b) ────────────────────────────────────
async function doVerify() {{
  if (!currentDraftId) {{ showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft."); return; }}
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
      const grStr = (vd.groundedness_score != null) ? vd.groundedness_score.toFixed(2) : '?';
      const hfStr = (vd.hedging_fidelity_score != null) ? vd.hedging_fidelity_score.toFixed(2) : '?';
      status.textContent = 'Groundedness: ' + grStr + '  ·  Hedging fidelity: ' + hfStr;

      // Show results in stream body
      let html = '<div style="font-family:-apple-system,sans-serif;">';
      html += '<div style="font-size:18px;font-weight:bold;margin-bottom:12px;">Groundedness: ' +
        '<span style="color:' + (vd.groundedness_score >= 0.8 ? 'var(--success)' : vd.groundedness_score >= 0.6 ? 'var(--warning)' : 'var(--danger)') + '">' +
        grStr + '</span>' +
        '   <span style="font-size:14px;font-weight:normal;opacity:0.7;">Hedging fidelity: ' +
        '<span style="color:' + (vd.hedging_fidelity_score >= 0.8 ? 'var(--success)' : vd.hedging_fidelity_score >= 0.6 ? 'var(--warning)' : 'var(--danger)') + '">' +
        hfStr + '</span></span></div>';

      if (vd.claims) {{
        vd.claims.forEach(c => {{
          // Phase 11 — OVERSTATED is the lexical-modality counterpart to
          // EXTRAPOLATED. Render in orange to distinguish from yellow
          // (extrapolated) and red (misrepresented).
          let color;
          if (c.verdict === 'SUPPORTED') color = 'var(--success)';
          else if (c.verdict === 'EXTRAPOLATED') color = 'var(--warning)';
          else if (c.verdict === 'OVERSTATED') color = '#f97316';  // orange-500
          else color = 'var(--danger)';
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
  const classMap = {{ 'SUPPORTED': 'verified-supported', 'EXTRAPOLATED': 'verified-extrapolated', 'OVERSTATED': 'verified-overstated', 'MISREPRESENTED': 'verified-misrepresented' }};
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

// Build popovers on page load
document.addEventListener('DOMContentLoaded', buildPopovers);

// ── Autowrite Dashboard (Phase 4) ────────────────────────────────────
// Enhanced autowrite that shows convergence chart
let awScores = [];
let awTargetScore = 0.85;

async function doAutowrite() {{
  if (!currentChapterId) {{ showEmptyHint("No chapter selected &mdash; click any chapter title in the sidebar to select it, then try again."); return; }}
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
    '<div id="aw-content" style="margin-top:12px;white-space:pre-wrap;line-height:1.6;font-family:var(--font-serif);font-size:15px;"></div>';

  // Phase 15 — live stats footer wired to the main stream-stats element
  const stats = createStreamStats('main-stream-stats', 'qwen3.5:27b');
  stats.start();

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
      // Phase 15.3 — route tokens by phase. Writing/revising tokens go
      // to the visible draft area; scoring/verify/CoVe/planning JSON
      // tokens only feed the stats counter (they'd be ugly to show in
      // the draft pane). Tokens without a phase are treated as draft
      // tokens for backward compatibility.
      const phase = evt.phase || 'writing';
      stats.update(evt.text);
      stats.setPhase(phase);
      if (phase === 'writing' || phase === 'revising') {{
        setStreamCursor(awContent, false);
        awContent.innerHTML += evt.text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        setStreamCursor(awContent, true);
        awContent.scrollTop = awContent.scrollHeight;
      }}
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
      const dims = ['groundedness', 'completeness', 'coherence', 'citation_accuracy', 'hedging_fidelity', 'overall'];
      scoresEl.innerHTML = 'Iteration ' + evt.iteration + ': ' + dims.map(d => {{
        const v = (s[d] || 0).toFixed(2);
        const cls = v >= 0.85 ? 'good' : v >= 0.7 ? 'mid' : 'low';
        return '<span class="score-bar"><span class="label">' + d.slice(0,6) + '</span> ' +
          '<span class="value ' + cls + '">' + v + '</span></span>';
      }}).join('');
    }}
    else if (evt.type === 'cove_verification') {{
      // Phase 11 — Chain-of-Verification mismatches in the autowrite stream.
      const cd = evt.data || {{}};
      const score = (cd.cove_score != null) ? cd.cove_score.toFixed(2) : '?';
      const mismatches = cd.mismatches || [];
      const high = mismatches.filter(m => m.severity === 'high');
      const med = mismatches.filter(m => m.severity === 'medium');
      let icon = high.length ? '\\u2717' : med.length ? '\\u26a0' : '\\u2713';
      let cls = high.length ? 'log-discard' : 'log-keep';
      awLog.innerHTML += '<div class="' + cls + '">' + icon + ' CoVe ' + score +
        ' · ' + high.length + 'H/' + med.length + 'M mismatches</div>';
    }}
    else if (evt.type === 'model_info') {{
      // Phase 15.5 — model name shown only in the stats footer; the
      // previous awLog line duplicated information that's already in
      // the live stats pill above the dashboard.
      stats.setModel(evt.writer_model || 'qwen3.5:27b');
    }}
    else if (evt.type === 'checkpoint') {{
      // Phase 15.1 — incremental save checkpoint reached. Show a brief
      // green note in the log so the user knows their work is persisted
      // and Stop won't lose anything past this point.
      awLog.innerHTML += '<div class="log-keep" style="font-size:11px;">' +
        '\\u2693 checkpoint saved · ' + (evt.stage || '') + ' · ' +
        (evt.word_count || 0) + ' words</div>';
    }}
    else if (evt.type === 'verification') {{
      // Standard verifier in the autowrite stream — emit a brief log line.
      const vd = evt.data || {{}};
      const grStr = (vd.groundedness_score != null) ? vd.groundedness_score.toFixed(2) : '?';
      const hfStr = (vd.hedging_fidelity_score != null) ? vd.hedging_fidelity_score.toFixed(2) : '?';
      awLog.innerHTML += '<div style="opacity:0.7;font-size:11px;">verify: gr=' + grStr + ' · hf=' + hfStr + '</div>';
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
      stats.done('done');
      setStreamCursor(awContent, false);
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
      refreshAfterJob(evt.draft_id);
    }}
    else if (evt.type === 'error') {{
      status.textContent = 'Error: ' + evt.message;
      awLog.innerHTML += '<div style="color:var(--danger);">' + evt.message + '</div>';
      stats.done('error');
      setStreamCursor(awContent, false);
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }}
    else if (evt.type === 'done') {{
      stats.done('done');
      setStreamCursor(awContent, false);
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

// ── Phase 14: Modal infrastructure ────────────────────────────────────
function openModal(id) {{
  document.getElementById(id).classList.add('open');
}}
function closeModal(id) {{
  document.getElementById(id).classList.remove('open');
  // Stop any in-flight job that the modal launched
  if (currentEventSource && (id === 'wiki-modal' || id === 'ask-modal')) {{
    try {{ currentEventSource.close(); }} catch (e) {{}}
    currentEventSource = null;
    if (currentJobId) {{
      fetch('/api/jobs/' + currentJobId, {{method: 'DELETE'}}).catch(() => {{}});
      currentJobId = null;
    }}
  }}
}}
// Escape closes any open modal
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') {{
    document.querySelectorAll('.modal-overlay.open').forEach(m => {{
      closeModal(m.id);
    }});
  }}
}});

// ── Phase 14: Score history viewer (Phase 13 GUI integration) ─────────
async function showScoresPanel() {{
  if (!currentDraftId) {{ showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft."); return; }}
  const panel = document.getElementById('scores-panel');
  const body = document.getElementById('scores-panel-body');
  panel.classList.add('open');
  body.innerHTML = '<div class="scores-empty">Loading...</div>';

  try {{
    const res = await fetch('/api/draft/' + currentDraftId + '/scores');
    if (!res.ok) {{
      body.innerHTML = '<div class="scores-empty">Draft not found.</div>';
      return;
    }}
    const data = await res.json();
    const history = data.score_history || [];

    if (history.length === 0) {{
      body.innerHTML = '<div class="scores-empty">' +
        'No score history persisted on this draft.<br>' +
        '<span style="font-size:11px;">Only autowrite drafts record convergence trajectories — drafts made with `book write` only have a final state.</span>' +
        '</div>';
      return;
    }}

    // Build the iteration table.
    const dims = ['groundedness', 'completeness', 'coherence', 'citation_accuracy', 'hedging_fidelity', 'overall'];
    let html = '<table class="scores-table"><thead><tr><th>Iter</th>';
    dims.forEach(d => {{ html += '<th>' + d.slice(0, 6) + '</th>'; }});
    html += '<th>Weakest</th><th>Verdict</th></tr></thead><tbody>';

    history.forEach(h => {{
      const s = h.scores || {{}};
      html += '<tr><td>' + h.iteration + '</td>';
      dims.forEach(d => {{
        const v = s[d];
        if (v == null) {{ html += '<td>—</td>'; }}
        else {{
          const cls = v >= 0.85 ? 'score-good' : v >= 0.7 ? 'score-mid' : 'score-low';
          html += '<td class="' + cls + '">' + Number(v).toFixed(2) + '</td>';
        }}
      }});
      html += '<td>' + (s.weakest_dimension || '—') + '</td>';
      const v = h.revision_verdict || '—';
      const verdictColor = v === 'KEEP' ? 'var(--success)' : v === 'DISCARD' ? 'var(--danger)' : 'var(--fg-faint)';
      html += '<td style="color:' + verdictColor + '">' + v + '</td>';
      html += '</tr>';
    }});
    html += '</tbody></table>';

    // CoVe + verification summary line
    const cove_runs = history.filter(h => h.cove && h.cove.ran);
    const total_overstated = history.reduce((acc, h) => acc + ((h.verification && h.verification.n_overstated) || 0), 0);
    const total_extrapolated = history.reduce((acc, h) => acc + ((h.verification && h.verification.n_extrapolated) || 0), 0);
    if (cove_runs.length || total_overstated || total_extrapolated) {{
      html += '<div style="padding:8px 16px;font-size:11px;color:var(--fg-muted);background:var(--toolbar-bg);border-top:1px solid var(--border);">';
      html += '<strong>Verification:</strong> ' + total_overstated + ' OVERSTATED · ' + total_extrapolated + ' EXTRAPOLATED across ' + history.length + ' iterations';
      if (cove_runs.length) {{
        html += '  ·  CoVe ran in ' + cove_runs.length + '/' + history.length + ' iterations';
      }}
      html += '</div>';
    }}

    // Sparkline for overall score trajectory
    const overalls = history.map(h => (h.scores && h.scores.overall) || 0);
    if (overalls.length >= 2) {{
      const w = 380, hgt = 50, pad = 4;
      const minV = Math.min(...overalls, 0.5);
      const maxV = Math.max(...overalls, 1.0);
      const range = maxV - minV || 1;
      const points = overalls.map((v, i) => {{
        const x = pad + (i / (overalls.length - 1)) * (w - 2 * pad);
        const y = hgt - pad - ((v - minV) / range) * (hgt - 2 * pad);
        return x + ',' + y;
      }}).join(' ');
      html += '<div class="scores-spark">';
      html += '<div style="font-size:11px;color:var(--fg-muted);margin-bottom:4px;">Overall score trajectory</div>';
      html += '<svg viewBox="0 0 ' + w + ' ' + hgt + '" preserveAspectRatio="none">';
      html += '<polyline points="' + points + '" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>';
      overalls.forEach((v, i) => {{
        const x = pad + (i / (overalls.length - 1)) * (w - 2 * pad);
        const y = hgt - pad - ((v - minV) / range) * (hgt - 2 * pad);
        html += '<circle cx="' + x + '" cy="' + y + '" r="3" fill="var(--accent)"/>';
      }});
      html += '</svg></div>';
    }}

    // Final overall + features active
    if (data.final_overall != null) {{
      html += '<div style="padding:8px 16px;font-size:11px;color:var(--fg-muted);">';
      html += 'Final overall: <strong>' + Number(data.final_overall).toFixed(3) + '</strong>';
      if (data.target_score) html += '  ·  target: ' + data.target_score;
      if (data.max_iter) html += '  ·  max_iter: ' + data.max_iter;
      html += '</div>';
    }}

    body.innerHTML = html;
  }} catch (e) {{
    body.innerHTML = '<div class="scores-empty" style="color:var(--danger);">Error loading score history: ' + e.message + '</div>';
  }}
}}

// ── Phase 15: live streaming stats helper ─────────────────────────────
//
// Returns an object with `start()`, `update(text)`, and `done(state)`
// methods that maintain a small footer showing rolling tok/s, total
// tokens, elapsed time, time-to-first-token, and the model name.
//
// Pure client-side — uses performance.now() timestamps and counts
// whitespace-delimited tokens (a rough proxy for actual model tokens
// but accurate enough for live feedback).
// Phase 15.4 — build tag the user can check from the browser console:
//   typeof STREAM_STATS_BUILD === "string" && STREAM_STATS_BUILD
// If undefined, the page is running stale JS — hard-refresh (Ctrl+Shift+R).
const STREAM_STATS_BUILD = "phase-15.4";

function createStreamStats(containerId, modelName) {{
  const el = document.getElementById(containerId);
  if (!el) return {{ start: ()=>{{}}, update: ()=>{{}}, done: ()=>{{}}, setModel: ()=>{{}}, setPhase: ()=>{{}} }};
  let started = 0, firstTok = 0, lastTok = 0, count = 0;
  let recentTokens = []; // sliding window for rolling tok/s
  let timer = null;
  let currentModel = modelName || '?';
  let currentPhase = '';
  // Phase 15.4 — debug logging: prints once when the FIRST token is
  // received so the user can confirm in DevTools Console that token
  // events are actually flowing through stats.update().
  let _firstUpdateLogged = false;

  function fmtTime(ms) {{
    const s = ms / 1000;
    if (s < 60) return s.toFixed(1) + 's';
    const m = Math.floor(s / 60);
    return m + 'm ' + Math.floor(s % 60).toString().padStart(2, '0') + 's';
  }}

  function render(state) {{
    const elapsed = started ? performance.now() - started : 0;
    const ttft = firstTok ? firstTok - started : 0;
    // Rolling tok/s over the last 3 seconds
    const cutoff = performance.now() - 3000;
    const recent = recentTokens.filter(t => t > cutoff);
    const tps = recent.length / 3;
    const avgTps = (firstTok && elapsed > 0) ? count / ((elapsed - ttft) / 1000) : 0;

    el.className = 'stream-stats ' + (state || 'streaming');
    el.innerHTML =
      '<span class="ss-dot"></span>' +
      '<span class="ss-stat"><strong>' + currentModel + '</strong></span>' +
      (currentPhase ? '<span class="ss-sep">·</span><span class="ss-stat ss-phase">' + currentPhase + '</span>' : '') +
      '<span class="ss-sep">·</span>' +
      '<span class="ss-stat"><strong>' + count + '</strong>&nbsp;tok</span>' +
      '<span class="ss-sep">·</span>' +
      '<span class="ss-stat"><strong>' + tps.toFixed(1) + '</strong>&nbsp;tok/s</span>' +
      (avgTps > 0 ? '<span class="ss-sep">·</span><span class="ss-stat" title="Average since first token">avg <strong>' + avgTps.toFixed(1) + '</strong></span>' : '') +
      '<span class="ss-sep">·</span>' +
      '<span class="ss-stat">' + fmtTime(elapsed) + '</span>' +
      (ttft > 0 ? '<span class="ss-sep">·</span><span class="ss-stat" title="Time to first token">ttft ' + fmtTime(ttft) + '</span>' : '');
  }}

  return {{
    start() {{
      started = performance.now();
      firstTok = 0; count = 0; recentTokens = [];
      el.style.display = 'flex';
      render('streaming');
      if (timer) clearInterval(timer);
      timer = setInterval(() => render('streaming'), 200);
    }},
    update(text) {{
      if (!started) this.start();
      if (!firstTok) {{
        firstTok = performance.now();
        if (!_firstUpdateLogged) {{
          _firstUpdateLogged = true;
          console.log('[stream-stats] first token received',
            {{ text: (text || '').slice(0, 40), build: STREAM_STATS_BUILD }});
        }}
      }}
      // Phase 15.4 — defensive: text might be non-string if a producer
      // emits malformed events. Coerce so the regex below doesn't throw.
      const safeText = (typeof text === 'string') ? text : String(text || '');
      // Approximate tokens via whitespace + punctuation splits.
      const n = (safeText.match(/\\S+/g) || []).length;
      count += n;
      const now = performance.now();
      for (let i = 0; i < n; i++) recentTokens.push(now);
    }},
    done(state) {{
      lastTok = performance.now();
      if (timer) {{ clearInterval(timer); timer = null; }}
      render(state || 'done');
    }},
    setModel(m) {{
      if (m) currentModel = m;
    }},
    setPhase(p) {{
      currentPhase = p || '';
    }},
  }};
}}

// Helper that adds a blinking cursor to an element while streaming.
function setStreamCursor(el, on) {{
  if (!el) return;
  let cursor = el.querySelector('.stream-cursor');
  if (on && !cursor) {{
    cursor = document.createElement('span');
    cursor.className = 'stream-cursor';
    el.appendChild(cursor);
  }} else if (!on && cursor) {{
    cursor.remove();
  }}
}}

// ── Phase 14: Wiki Query modal ────────────────────────────────────────
let wikiCurrentTab = 'wiki-query';
function switchWikiTab(name) {{
  wikiCurrentTab = name;
  document.querySelectorAll('#wiki-modal .tab').forEach(t => {{
    t.classList.toggle('active', t.dataset.tab === name);
  }});
  document.getElementById('wiki-query-pane').style.display = (name === 'wiki-query') ? 'block' : 'none';
  document.getElementById('wiki-browse-pane').style.display = (name === 'wiki-browse') ? 'block' : 'none';
  if (name === 'wiki-browse') loadWikiPages(1);
}}

let wikiBrowsePage = 1;
async function loadWikiPages(page) {{
  wikiBrowsePage = page || 1;
  const filter = document.getElementById('wiki-type-filter');
  const params = new URLSearchParams({{page: wikiBrowsePage, per_page: 50}});
  if (filter && filter.value) params.set('page_type', filter.value);
  const list = document.getElementById('wiki-browse-list');
  const detail = document.getElementById('wiki-page-detail');
  detail.style.display = 'none';
  list.innerHTML = '<div style="padding:24px;text-align:center;color:var(--fg-muted);">Loading...</div>';

  try {{
    const res = await fetch('/api/wiki/pages?' + params.toString());
    const data = await res.json();

    // Populate the type filter dropdown if it's still empty
    if (filter && filter.options.length <= 1 && data.available_types) {{
      data.available_types.forEach(t => {{
        const opt = document.createElement('option');
        opt.value = t;
        opt.textContent = t.replace(/_/g, ' ');
        filter.appendChild(opt);
      }});
    }}

    if (!data.pages || data.pages.length === 0) {{
      list.innerHTML = '<div style="padding:24px;text-align:center;color:var(--fg-muted);">' +
        'No wiki pages found.<br><span style="font-size:11px;">Run <code>uv run sciknow wiki compile</code> to build wiki pages from your corpus.</span></div>';
      return;
    }}

    let html = '<div class="wiki-page-list">';
    data.pages.forEach(p => {{
      const slug = (p.slug || '').replace(/'/g, '&#39;');
      html += '<div class="wiki-page-row" onclick="openWikiPage(&#39;' + slug + '&#39;)">';
      html += '<div class="wp-title">' + (p.title || p.slug || '').replace(/</g, '&lt;') + '</div>';
      html += '<div class="wp-meta">' + (p.word_count || 0).toLocaleString() + ' words · ' + (p.n_sources || 0) + ' src</div>';
      html += '<div class="wp-type">' + (p.page_type || '').replace(/_/g, ' ') + '</div>';
      html += '</div>';
    }});
    html += '</div>';

    if (data.n_pages > 1) {{
      html += '<div class="catalog-pager">';
      html += '<button onclick="loadWikiPages(' + (wikiBrowsePage - 1) + ')" ' + (wikiBrowsePage <= 1 ? 'disabled' : '') + '>‹ Prev</button>';
      html += '<span>Page ' + data.page + ' of ' + data.n_pages + '  ·  ' + data.total + ' pages</span>';
      html += '<button onclick="loadWikiPages(' + (wikiBrowsePage + 1) + ')" ' + (wikiBrowsePage >= data.n_pages ? 'disabled' : '') + '>Next ›</button>';
      html += '</div>';
    }}

    list.innerHTML = html;
  }} catch (e) {{
    list.innerHTML = '<div style="padding:24px;text-align:center;color:var(--danger);">Error: ' + e.message + '</div>';
  }}
}}

async function openWikiPage(slug) {{
  const list = document.getElementById('wiki-browse-list');
  const detail = document.getElementById('wiki-page-detail');
  const meta = document.getElementById('wiki-page-meta');
  const content = document.getElementById('wiki-page-content');
  list.style.display = 'none';
  detail.style.display = 'block';
  content.innerHTML = '<div style="padding:24px;text-align:center;color:var(--fg-muted);">Loading...</div>';

  try {{
    const res = await fetch('/api/wiki/page/' + encodeURIComponent(slug));
    if (!res.ok) {{
      content.innerHTML = '<div style="color:var(--danger);">Page not found.</div>';
      return;
    }}
    const data = await res.json();
    const metaParts = [];
    if (data.page_type) metaParts.push('<strong>' + data.page_type.replace(/_/g, ' ') + '</strong>');
    if (data.word_count) metaParts.push(data.word_count.toLocaleString() + ' words');
    if (data.n_sources) metaParts.push(data.n_sources + ' source(s)');
    if (data.updated_at) metaParts.push('updated ' + data.updated_at.substring(0, 10));
    meta.innerHTML = metaParts.join(' · ');
    content.innerHTML = data.content_html || '<em>(empty page)</em>';
  }} catch (e) {{
    content.innerHTML = '<div style="color:var(--danger);">Error: ' + e.message + '</div>';
  }}
}}

function closeWikiPageDetail() {{
  document.getElementById('wiki-browse-list').style.display = 'block';
  document.getElementById('wiki-page-detail').style.display = 'none';
}}

function openWikiModal() {{
  openModal('wiki-modal');
  setTimeout(() => document.getElementById('wiki-query-input').focus(), 100);
}}

async function doWikiQuery() {{
  const q = document.getElementById('wiki-query-input').value.trim();
  if (!q) return;
  const status = document.getElementById('wiki-status');
  const stream = document.getElementById('wiki-stream');
  const sources = document.getElementById('wiki-sources');

  status.textContent = 'Querying wiki...';
  stream.textContent = '';
  sources.style.display = 'none';
  sources.innerHTML = '';

  // Phase 15 — live tok/s + elapsed stats footer
  const stats = createStreamStats('wiki-stream-stats', 'wiki LLM');
  stats.start();
  setStreamCursor(stream, true);

  const fd = new FormData();
  fd.append('question', q);
  const res = await fetch('/api/wiki/query', {{method: 'POST', body: fd}});
  const data = await res.json();
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();

  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;

  source.onmessage = function(e) {{
    const evt = JSON.parse(e.data);
    if (evt.type === 'token') {{
      setStreamCursor(stream, false);
      stream.textContent += evt.text;
      setStreamCursor(stream, true);
      stream.scrollTop = stream.scrollHeight;
      stats.update(evt.text);
    }} else if (evt.type === 'model_info') {{
      stats.setModel(evt.writer_model || evt.fast_model || 'wiki LLM');
    }} else if (evt.type === 'progress') {{
      status.textContent = evt.detail || evt.stage;
    }} else if (evt.type === 'completed') {{
      status.textContent = 'Done';
      stats.done('done');
      setStreamCursor(stream, false);
      if (evt.sources && evt.sources.length) {{
        let html = '<div style="font-weight:600;color:var(--fg);margin-bottom:6px;">Sources</div>';
        evt.sources.forEach(s => {{ html += '<div class="src-item">' + s + '</div>'; }});
        sources.innerHTML = html;
        sources.style.display = 'block';
      }}
      source.close(); currentEventSource = null; currentJobId = null;
    }} else if (evt.type === 'error') {{
      status.textContent = 'Error: ' + evt.message;
      stats.done('error');
      setStreamCursor(stream, false);
      source.close(); currentEventSource = null; currentJobId = null;
    }} else if (evt.type === 'done') {{
      stats.done('done');
      setStreamCursor(stream, false);
      source.close(); currentEventSource = null; currentJobId = null;
    }}
  }};
}}

// ── Phase 14: Corpus Ask modal (RAG question) ─────────────────────────
function openAskModal() {{
  openModal('ask-modal');
  setTimeout(() => document.getElementById('ask-input').focus(), 100);
}}

async function doAsk() {{
  const q = document.getElementById('ask-input').value.trim();
  if (!q) return;
  const status = document.getElementById('ask-status');
  const stream = document.getElementById('ask-stream');
  const sources = document.getElementById('ask-sources');

  status.textContent = 'Retrieving and generating...';
  stream.textContent = '';
  sources.style.display = 'none';
  sources.innerHTML = '';

  const stats = createStreamStats('ask-stream-stats', 'qwen3.5:27b');
  stats.start();
  setStreamCursor(stream, true);

  const fd = new FormData();
  fd.append('question', q);
  const yf = document.getElementById('ask-year-from').value;
  const yt = document.getElementById('ask-year-to').value;
  if (yf) fd.append('year_from', yf);
  if (yt) fd.append('year_to', yt);

  const res = await fetch('/api/ask', {{method: 'POST', body: fd}});
  const data = await res.json();
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();

  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;
  let collectedSources = null;

  source.onmessage = function(e) {{
    const evt = JSON.parse(e.data);
    if (evt.type === 'token') {{
      setStreamCursor(stream, false);
      stream.textContent += evt.text;
      setStreamCursor(stream, true);
      stream.scrollTop = stream.scrollHeight;
      stats.update(evt.text);
    }} else if (evt.type === 'model_info') {{
      stats.setModel(evt.writer_model);
    }} else if (evt.type === 'progress') {{
      status.textContent = evt.detail || evt.stage;
    }} else if (evt.type === 'sources') {{
      collectedSources = evt.sources;
      status.textContent = 'Generating from ' + (evt.n || evt.sources.length) + ' passages...';
    }} else if (evt.type === 'completed') {{
      status.textContent = 'Done';
      stats.done('done');
      setStreamCursor(stream, false);
      if (collectedSources && collectedSources.length) {{
        let html = '<div style="font-weight:600;color:var(--fg);margin-bottom:6px;">Sources (' + collectedSources.length + ')</div>';
        collectedSources.forEach(s => {{ html += '<div class="src-item">' + s + '</div>'; }});
        sources.innerHTML = html;
        sources.style.display = 'block';
      }}
      source.close(); currentEventSource = null; currentJobId = null;
    }} else if (evt.type === 'error') {{
      status.textContent = 'Error: ' + evt.message;
      stats.done('error');
      setStreamCursor(stream, false);
      source.close(); currentEventSource = null; currentJobId = null;
    }} else if (evt.type === 'done') {{
      stats.done('done');
      setStreamCursor(stream, false);
      source.close(); currentEventSource = null; currentJobId = null;
    }}
  }};
}}

// ── Phase 14: Catalog Browser modal ───────────────────────────────────
let catalogPage = 1;

function openCatalogModal() {{
  openModal('catalog-modal');
  loadCatalog(1);
}}

async function loadCatalog(page) {{
  catalogPage = page || 1;
  const params = new URLSearchParams({{page: catalogPage, per_page: 25}});
  const author = document.getElementById('cat-author').value.trim();
  const journal = document.getElementById('cat-journal').value.trim();
  const yf = document.getElementById('cat-year-from').value;
  const yt = document.getElementById('cat-year-to').value;
  if (author) params.set('author', author);
  if (journal) params.set('journal', journal);
  if (yf) params.set('year_from', yf);
  if (yt) params.set('year_to', yt);

  const results = document.getElementById('catalog-results');
  results.innerHTML = '<div style="padding:24px;text-align:center;color:var(--fg-muted);">Loading...</div>';

  try {{
    const res = await fetch('/api/catalog?' + params.toString());
    const data = await res.json();
    if (!data.papers || data.papers.length === 0) {{
      results.innerHTML = '<div style="padding:24px;text-align:center;color:var(--fg-muted);">No papers match.</div>';
      return;
    }}

    let html = '<table class="catalog-table"><thead><tr><th>Title</th><th>Year</th><th>Journal</th><th>Authors</th></tr></thead><tbody>';
    data.papers.forEach(p => {{
      const authorStr = (p.authors || []).slice(0, 2).map(a => (a.name || '').split(/\\s+/).slice(-1)[0]).filter(Boolean).join(', ') + (p.authors && p.authors.length > 2 ? ' et al.' : '');
      html += '<tr onclick="askAboutPaper(' + JSON.stringify(p.title || '').replace(/"/g, '&quot;') + ')">';
      html += '<td><div class="ct-title">' + (p.title || '').replace(/</g, '&lt;') + '</div>';
      if (p.abstract) html += '<div class="ct-meta">' + p.abstract.substring(0, 160).replace(/</g, '&lt;') + '...</div>';
      html += '</td>';
      html += '<td>' + (p.year || '—') + '</td>';
      html += '<td>' + (p.journal || '—').substring(0, 30) + '</td>';
      html += '<td>' + authorStr + '</td>';
      html += '</tr>';
    }});
    html += '</tbody></table>';

    html += '<div class="catalog-pager">';
    html += '<button onclick="loadCatalog(' + (catalogPage - 1) + ')" ' + (catalogPage <= 1 ? 'disabled' : '') + '>‹ Prev</button>';
    html += '<span>Page ' + data.page + ' of ' + data.n_pages + '  ·  ' + data.total + ' papers</span>';
    html += '<button onclick="loadCatalog(' + (catalogPage + 1) + ')" ' + (catalogPage >= data.n_pages ? 'disabled' : '') + '>Next ›</button>';
    html += '</div>';

    results.innerHTML = html;
  }} catch (e) {{
    results.innerHTML = '<div style="padding:24px;text-align:center;color:var(--danger);">Error: ' + e.message + '</div>';
  }}
}}

function askAboutPaper(title) {{
  closeModal('catalog-modal');
  openAskModal();
  document.getElementById('ask-input').value = 'In the paper "' + title + '", ';
  document.getElementById('ask-input').focus();
}}

// ── Phase 14.3: Book Plan modal (the leitmotiv) ──────────────────────
async function openPlanModal() {{
  openModal('plan-modal');
  document.getElementById('plan-status').textContent = 'Loading...';
  document.getElementById('plan-text-input').value = '';
  try {{
    const res = await fetch('/api/book');
    const data = await res.json();
    document.getElementById('plan-title-input').value = data.title || '';
    document.getElementById('plan-desc-input').value = data.description || '';
    document.getElementById('plan-text-input').value = data.plan || '';
    if (!data.plan) {{
      document.getElementById('plan-status').innerHTML =
        '<span style="color:var(--warning);">No plan set yet.</span> Click <strong>Regenerate with LLM</strong> to draft one from your chapters and corpus, or write one manually below.';
    }} else {{
      document.getElementById('plan-status').textContent =
        data.plan.split(/\\s+/).filter(Boolean).length + ' words';
    }}
  }} catch (e) {{
    document.getElementById('plan-status').textContent = 'Error loading book: ' + e.message;
  }}
}}

async function savePlan() {{
  const title = document.getElementById('plan-title-input').value.trim();
  const desc = document.getElementById('plan-desc-input').value.trim();
  const plan = document.getElementById('plan-text-input').value.trim();
  document.getElementById('plan-status').textContent = 'Saving...';
  const fd = new FormData();
  fd.append('title', title);
  fd.append('description', desc);
  fd.append('plan', plan);
  try {{
    const res = await fetch('/api/book', {{method: 'PUT', body: fd}});
    if (!res.ok) throw new Error('save failed');
    document.getElementById('plan-status').innerHTML =
      '<span style="color:var(--success);">Saved.</span> ' +
      plan.split(/\\s+/).filter(Boolean).length + ' words. The new plan will be injected into all future writes.';
    // Refresh the page header to show the new title if it changed
    if (title) document.querySelector('.sidebar h2').textContent = title;
  }} catch (e) {{
    document.getElementById('plan-status').textContent = 'Save failed: ' + e.message;
  }}
}}

async function regeneratePlan() {{
  const status = document.getElementById('plan-status');
  const ta = document.getElementById('plan-text-input');
  if (ta.value.trim() && !confirm('This will replace the current plan with an LLM-generated one. Continue?')) return;
  status.textContent = 'Starting generation...';
  ta.value = '';

  const stats = createStreamStats('plan-stream-stats', 'qwen3.5:27b');
  stats.start();

  const fd = new FormData();
  const res = await fetch('/api/book/plan/generate', {{method: 'POST', body: fd}});
  const data = await res.json();
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();
  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;

  source.onmessage = function(e) {{
    const evt = JSON.parse(e.data);
    if (evt.type === 'token') {{
      ta.value += evt.text;
      ta.scrollTop = ta.scrollHeight;
      stats.update(evt.text);
    }} else if (evt.type === 'model_info') {{
      stats.setModel(evt.writer_model);
    }} else if (evt.type === 'progress') {{
      status.textContent = evt.detail || evt.stage;
    }} else if (evt.type === 'completed') {{
      status.innerHTML = '<span style="color:var(--success);">Generated and saved.</span> ' +
        (evt.chars || ta.value.length) + ' chars.';
      stats.done('done');
      source.close(); currentEventSource = null; currentJobId = null;
    }} else if (evt.type === 'error') {{
      status.textContent = 'Error: ' + evt.message;
      stats.done('error');
      source.close(); currentEventSource = null; currentJobId = null;
    }} else if (evt.type === 'done') {{
      stats.done('done');
      source.close(); currentEventSource = null; currentJobId = null;
    }}
  }};
}}

// ── Phase 14.3: Chapter scope modal (description + topic_query) ─────
function openChapterModal(chId) {{
  if (!chId) chId = currentChapterId;
  if (!chId) {{
    showEmptyHint('Select a chapter from the sidebar first.');
    return;
  }}
  // Look up the chapter from chaptersData (already in JS state)
  const ch = chaptersData.find(c => c.id === chId);
  if (!ch) {{
    showEmptyHint('Chapter not found.');
    return;
  }}
  document.getElementById('ch-title-input').value = ch.title || '';
  document.getElementById('ch-desc-input').value = ch.description || '';
  document.getElementById('ch-tq-input').value = ch.topic_query || '';
  document.getElementById('chapter-modal-status').textContent = 'Editing Ch.' + ch.num + ': ' + (ch.title || '');
  document.getElementById('chapter-modal').dataset.chId = chId;
  openModal('chapter-modal');
}}

async function saveChapterInfo() {{
  const chId = document.getElementById('chapter-modal').dataset.chId;
  if (!chId) return;
  const title = document.getElementById('ch-title-input').value.trim();
  const desc = document.getElementById('ch-desc-input').value;
  const tq = document.getElementById('ch-tq-input').value.trim();
  const status = document.getElementById('chapter-modal-status');
  status.textContent = 'Saving...';

  const fd = new FormData();
  if (title) fd.append('title', title);
  fd.append('description', desc);
  fd.append('topic_query', tq);

  try {{
    const res = await fetch('/api/chapters/' + chId, {{method: 'PUT', body: fd}});
    if (!res.ok) throw new Error('save failed');
    // Update the in-memory chapter cache
    const ch = chaptersData.find(c => c.id === chId);
    if (ch) {{
      if (title) ch.title = title;
      ch.description = desc;
      ch.topic_query = tq;
    }}
    status.innerHTML = '<span style="color:var(--success);">Saved.</span>';
    // Refresh the sidebar so renamed chapters show new title
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters, currentDraftId);
    setTimeout(() => closeModal('chapter-modal'), 800);
  }} catch (e) {{
    status.textContent = 'Save failed: ' + e.message;
  }}
}}

// ── Corkboard View ────────────────────────────────────────────────────
async function showCorkboard() {{
  const res = await fetch('/api/corkboard');
  const data = await res.json();

  let html = '<div style="font-family:-apple-system,sans-serif;">';
  html += '<h2>Corkboard</h2>';
  html += '<p style="font-size:12px;opacity:0.5;margin-bottom:12px;">Click a card to navigate. Color = status.</p>';
  html += '<div class="corkboard">';

  data.cards.forEach(c => {{
    const statusCls = c.status || 'to_do';
    const onclick = c.draft_id
      ? 'loadSection(\\\'' + c.draft_id + '\\\')'
      : 'writeForCell(\\\'' + c.chapter_id + '\\\',\\\'' + c.section_type + '\\\')';
    html += '<div class="cork-card" onclick="' + onclick + '">';
    html += '<div class="cc-head"><span class="cc-ch">Ch.' + c.chapter_num + '</span>';
    html += '<span class="cc-status ' + statusCls + '">' + statusCls.replace('_', ' ') + '</span></div>';
    html += '<div class="cc-type">' + c.section_type.charAt(0).toUpperCase() + c.section_type.slice(1) + '</div>';
    if (c.summary) {{
      html += '<div class="cc-summary">' + c.summary + '</div>';
    }} else {{
      html += '<div class="cc-summary" style="opacity:0.3;">' + (c.draft_id ? 'No summary' : 'Not started') + '</div>';
    }}
    html += '<div class="cc-footer">';
    if (c.draft_id) html += 'v' + c.version + ' \\u00b7 ' + c.words + 'w';
    html += '</div></div>';
  }});

  html += '</div></div>';

  document.getElementById('dashboard-view').innerHTML = html;
  document.getElementById('dashboard-view').style.display = 'block';
  document.getElementById('read-view').style.display = 'none';
  document.getElementById('edit-view').style.display = 'none';
  document.getElementById('draft-title').textContent = 'Corkboard';
  document.getElementById('draft-subtitle').style.display = 'none';
  document.getElementById('toolbar').style.display = 'none';
  document.getElementById('stream-panel').style.display = 'none';
  document.getElementById('version-panel').style.display = 'none';
  document.querySelectorAll('.sec-link').forEach(l => l.classList.remove('active'));
  history.pushState({{corkboard: true}}, '', '/');
}}

// ── Chapter Reader (continuous scroll) ────────────────────────────────
async function showChapterReader() {{
  if (!currentChapterId) {{ showEmptyHint("No chapter selected &mdash; click any chapter title in the sidebar to select it, then try again."); return; }}

  const res = await fetch('/api/chapter-reader/' + currentChapterId);
  if (!res.ok) {{ alert('Chapter not found.'); return; }}
  const data = await res.json();

  let html = '<div class="reader-view">';
  html += '<h1>Chapter ' + data.chapter_num + ': ' + data.chapter_title + '</h1>';
  html += '<p style="font-size:13px;opacity:0.5;margin-bottom:24px;">' +
    data.total_words + ' words \\u00b7 ' + data.section_count + ' sections</p>';
  html += data.html;
  html += '</div>';

  document.getElementById('dashboard-view').innerHTML = html;
  document.getElementById('dashboard-view').style.display = 'block';
  document.getElementById('read-view').style.display = 'none';
  document.getElementById('edit-view').style.display = 'none';
  document.getElementById('draft-title').textContent = 'Ch.' + data.chapter_num + ': ' + data.chapter_title;
  document.getElementById('draft-subtitle').style.display = 'none';
  document.getElementById('toolbar').style.display = 'none';
  history.pushState({{reader: true}}, '', '/');

  // Build popovers for citations in reader view
  setTimeout(buildPopovers, 100);
}}

// ── Snapshots ─────────────────────────────────────────────────────────
async function takeSnapshot() {{
  if (!currentDraftId) {{ showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft."); return; }}
  const name = prompt('Snapshot name (leave empty for timestamp):');
  if (name === null) return;

  const fd = new FormData();
  fd.append('name', name);
  const res = await fetch('/api/snapshot/' + currentDraftId, {{method: 'POST', body: fd}});
  const data = await res.json();
  if (data.ok) {{
    alert('Snapshot saved: ' + data.name);
  }}
}}

async function showSnapshots() {{
  if (!currentDraftId) return;
  const res = await fetch('/api/snapshots/' + currentDraftId);
  const data = await res.json();
  if (!data.snapshots || data.snapshots.length === 0) return;

  // Show in the version panel
  const panel = document.getElementById('version-panel');
  const timeline = document.getElementById('version-timeline');
  const diffView = document.getElementById('diff-view');
  panel.style.display = 'block';

  let html = '<div class="snap-list">';
  html += '<div style="font-weight:600;margin-bottom:6px;">Snapshots</div>';
  data.snapshots.forEach(s => {{
    html += '<div class="snap-item">';
    html += '<span>' + s.name + ' (' + s.word_count + 'w)</span>';
    html += '<div>';
    html += '<button onclick="diffSnapshot(\\\'' + s.id + '\\\')">Diff</button> ';
    html += '<button onclick="restoreSnapshot(\\\'' + s.id + '\\\')">Restore</button>';
    html += '</div></div>';
  }});
  html += '</div>';
  timeline.innerHTML = html;
  diffView.innerHTML = '<p style="opacity:0.5;">Click "Diff" to compare a snapshot with current content.</p>';
}}

async function diffSnapshot(snapId) {{
  // Get snapshot content and current content, do client-side word diff
  const snapRes = await fetch('/api/snapshot-content/' + snapId);
  const snapData = await snapRes.json();
  const secRes = await fetch('/api/section/' + currentDraftId);
  const secData = await secRes.json();

  // Simple word diff
  const oldWords = snapData.content.split(/\\s+/);
  const newWords = secData.content_raw.split(/\\s+/);

  // Use a basic LCS-based diff
  let html = '';
  let i = 0, j = 0;
  // Simplified: just show both for now, use the server diff endpoint
  const diffRes = await fetch('/api/diff/' + 'snapshot' + '/' + currentDraftId);
  // Fallback: show snapshot content with note
  html = '<div style="margin-bottom:8px;font-weight:bold;">Snapshot content:</div>';
  html += '<div style="opacity:0.7;white-space:pre-wrap;">' + snapData.content.substring(0, 5000).replace(/</g, '&lt;') + '</div>';
  document.getElementById('diff-view').innerHTML = html;
}}

async function restoreSnapshot(snapId) {{
  if (!confirm('Restore this snapshot? Current content will be overwritten.')) return;
  const snapRes = await fetch('/api/snapshot-content/' + snapId);
  const snapData = await snapRes.json();

  const fd = new FormData();
  fd.append('content', snapData.content);
  await fetch('/edit/' + currentDraftId, {{method: 'POST', body: fd}});
  loadSection(currentDraftId);
}}

// ── Status selector ───────────────────────────────────────────────────
async function updateStatus(status) {{
  if (!currentDraftId) return;
  const fd = new FormData();
  fd.append('status', status);
  await fetch('/api/draft/' + currentDraftId + '/status', {{method: 'PUT', body: fd}});
}}


</script>
</body>
</html>
"""
