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
import time
from html import escape as _esc
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


def _chapter_sections_dicts(ch_row) -> list[dict]:
    """Return the section template for a chapter row, normalized to
    [{slug, title, plan}, ...].

    Phase 18. Delegates to core.book_ops._normalize_chapter_sections so
    the web layer and the backend agree on the legacy/new shape rules.
    Falls back to a small default set when the chapter has no sections
    set, so the GUI never shows an empty list for a fresh chapter.
    """
    from sciknow.core.book_ops import _normalize_chapter_sections, _titleify_slug
    raw = ch_row[6] if len(ch_row) > 6 else None
    out = _normalize_chapter_sections(raw)
    if out:
        return out
    # Empty → fallback so the sidebar still has something to show.
    return [
        {"slug": s, "title": _titleify_slug(s), "plan": ""}
        for s in _DEFAULT_BOOK_SECTIONS
    ]


def _chapter_sections(ch_row) -> list[str]:
    """Return the section template for a chapter row as a flat list of slugs.

    Kept for backward compatibility with sidebar/heatmap code that only
    needs the slug ordering. New code should prefer _chapter_sections_dicts.
    """
    return [s["slug"] for s in _chapter_sections_dicts(ch_row)]


def set_book(book_id: str, book_title: str) -> None:
    global _book_id, _book_title
    _book_id = book_id
    _book_title = book_title


# ── Job management ───────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}     # job_id -> {queue, status, type, cancel, finished_at}
_job_lock = threading.Lock()

# Phase 22 — _jobs used to grow unbounded: every job ever created stayed
# in memory with its queue object. For a writing session of an hour or
# two that's a few hundred KB; over a long-running deployment it leaks.
# We now sweep finished jobs older than this window on every _create_job
# call. 5 minutes is long enough that the SSE consumer has fully drained
# the queue, but short enough that the dict stays bounded.
_JOB_GC_AGE_SECONDS = 300


def _gc_old_jobs() -> int:
    """Drop finished jobs older than _JOB_GC_AGE_SECONDS. Returns the
    number evicted. Caller MUST hold _job_lock."""
    now = time.monotonic()
    stale = [
        jid for jid, j in _jobs.items()
        if j.get("status") == "done"
        and (now - j.get("finished_at", now)) > _JOB_GC_AGE_SECONDS
    ]
    for jid in stale:
        del _jobs[jid]
    return len(stale)


def _create_job(job_type: str) -> tuple[str, asyncio.Queue]:
    job_id = uuid4().hex[:12]
    queue: asyncio.Queue = asyncio.Queue()
    with _job_lock:
        # Sweep stale finished jobs before inserting the new one. Cheap
        # lock-protected dict scan; runs at most a few times per minute.
        _gc_old_jobs()
        _jobs[job_id] = {
            "queue": queue,
            "status": "running",
            "type": job_type,
            "cancel": threading.Event(),
            "finished_at": None,
        }
    return job_id, queue


def _finish_job(job_id: str):
    with _job_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["finished_at"] = time.monotonic()


# ── Data helpers ─────────────────────────────────────────────────────────────

def _get_book_data():
    with get_session() as session:
        book = session.execute(text("""
            SELECT id::text, title, description, plan, status, custom_metadata
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


def _draft_display_title(
    draft_title: str | None,
    section_type: str | None,
    chapter_num: int | None,
    chapter_title: str | None,
    chapter_sections_raw,
) -> str:
    """Phase 27 — derive the display title for a draft from the
    chapter sections meta, falling back to drafts.title if no match.

    Why: drafts.title is set ONCE at draft creation by _save_draft as
    f"Ch.{n} {ch_title} - {section_type.capitalize()}". When the user
    later renames the section in the chapter modal (Phase 18), the
    chapter sections JSONB updates but the stored draft.title doesn't
    — so the sidebar shows the new title (it reads sections_meta) but
    the center h1 keeps showing the stale slug-based snapshot. The
    fix is to always derive the display title from the meta on read.

    Format: "Ch.{n} {chapter_title} - {section_meta_title}". Falls
    back to draft_title for orphan drafts (where section_type doesn't
    match any current slug in the chapter sections meta) and for any
    other lookup failure.
    """
    from sciknow.core.book_ops import _normalize_chapter_sections
    fallback = draft_title or ""
    if not section_type or not chapter_sections_raw:
        return fallback
    try:
        meta = _normalize_chapter_sections(chapter_sections_raw)
    except Exception:
        return fallback
    target = (section_type or "").strip().lower()
    matched = next((s for s in meta if s["slug"] == target), None)
    if not matched:
        return fallback
    section_title = matched.get("title") or matched["slug"]
    if chapter_num is not None and chapter_title:
        return f"Ch.{int(chapter_num)} {chapter_title} \u2014 {section_title}"
    if chapter_title:
        return f"{chapter_title} \u2014 {section_title}"
    return section_title


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
    # book columns: id, title, description, plan, status, custom_metadata
    meta = (book[5] if book and len(book) > 5 else None) or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    # Phase 17 — expose the length target so the GUI Plan modal can
    # render it (and show the default when unset).
    target_chapter_words = meta.get("target_chapter_words") if isinstance(meta, dict) else None
    return {
        "id": book[0] if book else "",
        "title": book[1] if book else "",
        "description": (book[2] or "") if book else "",
        "plan": (book[3] or "") if book else "",
        "status": (book[4] or "draft") if book else "draft",
        "target_chapter_words": target_chapter_words,  # may be None → client shows default
        "default_target_chapter_words": 6000,
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
    target_chapter_words: int = Form(None),
):
    """Update the book's title, description (short blurb), plan
    (the 200-500 word thesis/scope document used by the writer prompt),
    or length target. All fields are optional — only the ones you pass
    get updated.

    Phase 17 — target_chapter_words lives in books.custom_metadata as
    a JSONB key so we can add more book-level settings without a
    schema change each time. Passing a zero or negative value clears
    the setting (reverts to the default 6000).
    """
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
    if target_chapter_words is not None:
        # Merge into JSONB so we preserve any other keys. We use the
        # `||` concat operator + jsonb_build_object so the JSON shape
        # is built server-side. For a clear/delete, we use `- key`.
        # Note: use CAST(... AS int) instead of `:tcw::int` because
        # SQLAlchemy's parameter parser confuses `::int` with a bound
        # parameter name. Same gotcha as _save_draft() in book_ops.py.
        if target_chapter_words > 0:
            updates.append(
                "custom_metadata = "
                "COALESCE(custom_metadata, CAST('{}' AS jsonb)) || "
                "jsonb_build_object('target_chapter_words', CAST(:tcw AS int))"
            )
            params["tcw"] = int(target_chapter_words)
        else:
            updates.append(
                "custom_metadata = "
                "COALESCE(custom_metadata, CAST('{}' AS jsonb)) - 'target_chapter_words'"
            )
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

    # Phase 27 — chapter_id → sections JSONB lookup so the display
    # title can be derived from the meta even if drafts.title is stale.
    chapter_sections_by_id = {ch[0]: ch[6] for ch in chapters}

    active = draft_map.get(draft_id)
    if not active:
        raise HTTPException(404, "Draft not found")

    # Group comments for this draft
    active_comments = [c for c in comments if c[1] == draft_id]
    sources = json.loads(active[5]) if isinstance(active[5], str) else (active[5] or [])

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
                chapter_target = _get_book_length_target(_session, _book_id)
                num_sections = _get_chapter_num_sections(_session, str(active[9]))
                target_words = _section_target_words(chapter_target, num_sections)
    except Exception as exc:
        logger.warning("target_words lookup failed: %s", exc)

    # Phase 27 — display_title is derived from the chapter sections meta
    # at every read so renaming a section in the chapter modal updates
    # the center frame on the next navigation. Falls back to the stored
    # drafts.title for orphans (section_type doesn't match any current
    # slug) and any other lookup failure.
    sections_raw = chapter_sections_by_id.get(active[9])
    display_title = _draft_display_title(
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
        "target_words": target_words,
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
        template_dicts = _chapter_sections_dicts(ch)
        template = [s["slug"] for s in template_dicts]
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
            # Phase 18 — rich {slug, title, plan} dicts for the chapter
            # modal's Sections tab. The legacy `sections_template` (flat
            # slug list) is kept for the sidebar / heatmap code that
            # only needs slugs, so existing renderers don't break.
            "sections_meta": template_dicts,
        })

    return {"chapters": result, "gaps_count": len([g for g in gaps if g[3] == "open"])}


@app.get("/api/dashboard")
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
    book, chapters, drafts, gaps, comments = _get_book_data()

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
        # Keep only the latest version per section_type
        existing = ch_section_drafts[ch_id].get(sec_type)
        if not existing or (version or 1) > existing["version"]:
            ch_section_drafts[ch_id][sec_type] = {
                "id": draft_id, "version": version or 1, "words": wc or 0,
                "has_review": bool(review_fb),
            }

    # Phase 30 — compute the per-chapter sections lists FIRST so we
    # can find max(N) for the column count.
    chapter_sections: dict[str, list[dict]] = {}
    for ch in chapters:
        chapter_sections[ch[0]] = _chapter_sections_dicts(ch)
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


@app.post("/api/chapters/{chapter_id}/sections/adopt")
async def adopt_orphan_section_endpoint(chapter_id: str, request: Request):
    """Phase 25 — adopt an orphan draft's section_type into the chapter's
    sections list. Body: ``{"slug": "...", "title": "...", "plan": "..."}``
    (title and plan are optional; defaults are titleified slug + empty
    plan).

    Idempotent: if the slug already exists in the chapter's sections,
    returns the existing entry without duplication. The orphan draft
    keeps its content unchanged — only the chapter's sections JSONB is
    modified, which causes the GUI to re-classify the draft from
    "orphan" to "drafted" on the next refresh.
    """
    from sciknow.core.book_ops import adopt_orphan_section as _adopt

    body = await request.json()
    slug = (body.get("slug") or "").strip()
    title = body.get("title")
    plan = body.get("plan")
    if not slug:
        raise HTTPException(400, "slug is required")
    try:
        result = _adopt(_book_id, chapter_id, slug, title=title, plan=plan)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return JSONResponse(result)


@app.put("/api/chapters/{chapter_id}/sections")
async def update_chapter_sections(chapter_id: str, request: Request):
    """Replace a chapter's sections list.

    Phase 18. Body: ``{"sections": [{"slug": "...", "title": "...", "plan": "..."}, ...]}``

    The slug is the unique key (used as drafts.section_type). Duplicate
    slugs are dropped (first wins). Sections with empty slug+title are
    dropped. The list order matters: it determines the order shown in
    the chapter reader and the sidebar.

    DOES NOT delete existing drafts whose section_type no longer
    appears in the new list. They become orphans visible only in raw
    SQL — that's a deliberate safety choice so renaming a section
    can't silently destroy work.
    """
    from sciknow.core.book_ops import _normalize_chapter_sections

    body = await request.json()
    raw_sections = body.get("sections", [])
    normalized = _normalize_chapter_sections(raw_sections)

    # Drop duplicates by slug, keeping the first occurrence — the
    # client can theoretically send dupes, and we don't want them
    # silently colliding with each other.
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


# ── Phase 30: Export endpoints (text / markdown / printable HTML) ───────────
#
# Pandoc isn't installed and adding weasyprint would pull a lot of
# system deps, so the "PDF" path is HTML with print-friendly CSS —
# the user opens it in their browser and uses File → Print → Save
# as PDF. For most single-user book-writing workflows that's the
# right tradeoff: zero new dependencies, near-perfect typography,
# editable in the browser before saving.
#
# Endpoints:
#   GET /api/export/draft/{draft_id}.{ext}
#   GET /api/export/chapter/{chapter_id}.{ext}
#   GET /api/export/book.{ext}
#
# ext ∈ {txt, md, html}. txt is markdown stripped of formatting; md
# is the raw stored content (which IS markdown); html is the same
# rendered with print CSS.

_EXPORT_PRINT_CSS = """
<style>
  @page {
    size: A4;
    margin: 25mm 22mm;
    @bottom-right { content: counter(page); }
  }
  body {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #111;
    max-width: 720px;
    margin: 0 auto;
    padding: 24px;
  }
  h1 { font-size: 24pt; margin: 32pt 0 12pt; page-break-before: always; }
  h1:first-of-type { page-break-before: avoid; }
  h2 { font-size: 16pt; margin: 24pt 0 8pt; }
  h3 { font-size: 13pt; margin: 18pt 0 6pt; }
  p { margin: 0 0 8pt; text-align: justify; }
  .meta { color: #666; font-size: 9pt; margin-bottom: 24pt;
          font-family: -apple-system, sans-serif; }
  .citation {
    color: #2563eb; font-weight: 600; vertical-align: super;
    font-size: 0.78em; text-decoration: none;
  }
  .sources { margin-top: 32pt; padding-top: 16pt;
             border-top: 1px solid #ccc; }
  .sources h2 { font-size: 13pt; }
  .sources ol { padding-left: 20pt; font-size: 9pt; line-height: 1.5; }
  .sources li { margin-bottom: 6pt; color: #444; }
  @media print {
    body { padding: 0; }
    .no-print { display: none !important; }
  }
</style>
"""


def _strip_md(text_in: str) -> str:
    """Best-effort markdown → plain text. Removes heading markers,
    bold/italic markers, and converts citations to plain [N]. Keeps
    line breaks and paragraphs intact."""
    if not text_in:
        return ""
    out = text_in
    out = re.sub(r"^#{1,6}\s*", "", out, flags=re.MULTILINE)
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", out)
    out = re.sub(r"\*(.+?)\*", r"\1", out)
    return out


def _draft_to_md(draft_row, *, include_sources: bool = True) -> str:
    """Render a single draft as markdown. draft_row is the 14-tuple
    from _get_book_data's drafts query."""
    title = draft_row[1] or "Untitled"
    content = draft_row[3] or ""
    sources = draft_row[5]
    if isinstance(sources, str):
        try:
            sources = json.loads(sources)
        except Exception:
            sources = []
    sources = sources or []
    out = f"# {title}\n\n{content.strip()}\n"
    if include_sources and sources:
        out += "\n\n## Sources\n\n"
        for i, s in enumerate(sources, start=1):
            if s:
                out += f"{i}. {s}\n"
    return out


def _draft_to_html_body(draft_row) -> str:
    """Render a single draft as the inner HTML body for the export
    template. Uses _md_to_html for markdown + citations and
    _render_sources for the bibliography panel."""
    title = _esc(draft_row[1] or "Untitled")
    content_html = _md_to_html(draft_row[3] or "")
    sources = draft_row[5]
    if isinstance(sources, str):
        try:
            sources = json.loads(sources)
        except Exception:
            sources = []
    sources_html = _render_sources(sources or [])
    return (
        f"<h1>{title}</h1>\n"
        f"<div class='meta'>{int(draft_row[4] or 0)} words &middot; "
        f"version {int(draft_row[6] or 1)}</div>\n"
        f"{content_html}\n"
        f"<div class='sources'><h2>Sources</h2>{sources_html}</div>"
    )


def _wrap_html_export(title: str, body_html: str) -> str:
    """Wrap export body content in a complete HTML document with
    print-friendly CSS. The user opens this in a browser and uses
    File → Print → Save as PDF."""
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{_esc(title)}</title>{_EXPORT_PRINT_CSS}</head>"
        f"<body>{body_html}</body></html>"
    )


def _ordered_chapter_drafts(drafts, chapter_id: str) -> list:
    """Helper: filter drafts to one chapter and order them by the
    chapter's sections list, returning latest version per section_type."""
    from sciknow.core.book_ops import _normalize_chapter_sections
    with get_session() as session:
        row = session.execute(text(
            "SELECT sections FROM book_chapters WHERE id::text = :cid"
        ), {"cid": chapter_id}).fetchone()
    sections_raw = row[0] if row else None
    sections_meta = _normalize_chapter_sections(sections_raw)
    order = {s["slug"]: i for i, s in enumerate(sections_meta)}
    in_chapter = [d for d in drafts if d[9] == chapter_id]
    # Latest version per section_type
    latest: dict[str, tuple] = {}
    for d in in_chapter:
        st = (d[2] or "").strip().lower()
        if st not in latest or (d[6] or 1) > (latest[st][6] or 1):
            latest[st] = d
    return sorted(latest.values(), key=lambda d: order.get(
        (d[2] or "").strip().lower(), 999
    ))


def _slugify_for_filename(s: str) -> str:
    """Make a string safe for a filename (lowercase alphanum + dashes).
    Mirrors core.book_ops._slugify_for_filename so the web layer doesn't
    have to import a private helper."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:60] or "export"


def _html_to_pdf_response(html: str, filename: str):
    """Phase 31 — render an HTML export to PDF using weasyprint.
    Returns a Response with the right Content-Type and a friendly
    error message if weasyprint isn't installed.

    weasyprint pulls Cairo + Pango via system libs; on Ubuntu the
    deps are usually already there for GTK apps. The dependency is
    declared in pyproject.toml so a fresh `uv sync` will install it,
    but we still wrap the import in a try/except so a missing system
    lib downgrades to a 503 with a clear message instead of crashing
    the whole web server.
    """
    from fastapi.responses import Response
    try:
        from weasyprint import HTML
    except ImportError as exc:
        raise HTTPException(
            503,
            f"PDF export requires weasyprint ({exc}). "
            "Install with: uv add weasyprint",
        )
    try:
        pdf_bytes = HTML(string=html).write_pdf()
    except Exception as exc:
        logger.exception("PDF render failed")
        raise HTTPException(500, f"PDF render failed: {exc}")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


_VALID_EXPORT_EXTS = ("txt", "md", "html", "pdf")


@app.get("/api/export/draft/{draft_id}.{ext}")
async def export_draft(draft_id: str, ext: str):
    """Phase 30/31 — export a single draft as txt/md/html/pdf."""
    if ext not in _VALID_EXPORT_EXTS:
        raise HTTPException(400, f"ext must be one of {_VALID_EXPORT_EXTS}")
    book, chapters, drafts, gaps, comments = _get_book_data()
    draft = next((d for d in drafts if d[0] == draft_id or d[0].startswith(draft_id)), None)
    if not draft:
        raise HTTPException(404, "Draft not found")
    md = _draft_to_md(draft)
    if ext == "md":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(md, media_type="text/markdown; charset=utf-8")
    if ext == "txt":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(_strip_md(md), media_type="text/plain; charset=utf-8")
    # html or pdf
    body = _draft_to_html_body(draft)
    html = _wrap_html_export(draft[1] or "Untitled", body)
    if ext == "pdf":
        slug = _slugify_for_filename(draft[1] or "draft") or "draft"
        return _html_to_pdf_response(html, f"{slug}.pdf")
    return HTMLResponse(html)


@app.get("/api/export/chapter/{chapter_id}.{ext}")
async def export_chapter(chapter_id: str, ext: str):
    """Phase 30/31 — export every drafted section in a chapter, ordered
    by the chapter's sections meta. Skips empty/orphan sections."""
    if ext not in _VALID_EXPORT_EXTS:
        raise HTTPException(400, f"ext must be one of {_VALID_EXPORT_EXTS}")
    book, chapters, drafts, gaps, comments = _get_book_data()
    ch = next((c for c in chapters if c[0] == chapter_id), None)
    if not ch:
        raise HTTPException(404, "Chapter not found")
    ch_num, ch_title = ch[1], ch[2]
    section_drafts = _ordered_chapter_drafts(drafts, chapter_id)
    if ext == "md":
        from fastapi.responses import PlainTextResponse
        parts = [f"# Ch.{ch_num} {ch_title}\n"]
        for d in section_drafts:
            parts.append(_draft_to_md(d))
        return PlainTextResponse("\n\n".join(parts), media_type="text/markdown; charset=utf-8")
    if ext == "txt":
        from fastapi.responses import PlainTextResponse
        parts = [f"Ch.{ch_num} {ch_title}\n{'=' * 40}\n"]
        for d in section_drafts:
            parts.append(_strip_md(_draft_to_md(d)))
        return PlainTextResponse("\n\n".join(parts), media_type="text/plain; charset=utf-8")
    # html or pdf
    body = (
        f"<h1>Ch.{ch_num} {_esc(ch_title or '')}</h1>"
        f"<div class='meta'>{len(section_drafts)} sections</div>"
    )
    for d in section_drafts:
        body += _draft_to_html_body(d)
    html = _wrap_html_export(f"Ch.{ch_num} {ch_title}", body)
    if ext == "pdf":
        slug = _slugify_for_filename(ch_title or "chapter") or "chapter"
        return _html_to_pdf_response(html, f"ch{ch_num}_{slug}.pdf")
    return HTMLResponse(html)


@app.get("/api/export/book.{ext}")
async def export_book(ext: str):
    """Phase 30/31 — export the whole book as one file. Iterates
    chapters in order, then sections in each chapter's defined order."""
    if ext not in _VALID_EXPORT_EXTS:
        raise HTTPException(400, f"ext must be one of {_VALID_EXPORT_EXTS}")
    book, chapters, drafts, gaps, comments = _get_book_data()
    book_title = book[1] if book else "Untitled book"
    if ext == "md":
        from fastapi.responses import PlainTextResponse
        parts = [f"# {book_title}\n"]
        for ch in chapters:
            parts.append(f"\n## Ch.{ch[1]} {ch[2]}\n")
            for d in _ordered_chapter_drafts(drafts, ch[0]):
                parts.append(_draft_to_md(d, include_sources=False))
        # Bibliography at the end of the book, not per-section
        all_sources = []
        seen = set()
        for d in drafts:
            srcs = d[5]
            if isinstance(srcs, str):
                try:
                    srcs = json.loads(srcs)
                except Exception:
                    srcs = []
            for s in srcs or []:
                if s and s not in seen:
                    seen.add(s)
                    all_sources.append(s)
        if all_sources:
            parts.append("\n\n## Bibliography\n\n")
            for i, s in enumerate(all_sources, start=1):
                parts.append(f"{i}. {s}\n")
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("\n".join(parts), media_type="text/markdown; charset=utf-8")
    if ext == "txt":
        from fastapi.responses import PlainTextResponse
        parts = [f"{book_title}\n{'=' * len(book_title)}\n"]
        for ch in chapters:
            parts.append(f"\nCh.{ch[1]} {ch[2]}\n{'-' * 40}\n")
            for d in _ordered_chapter_drafts(drafts, ch[0]):
                parts.append(_strip_md(_draft_to_md(d, include_sources=False)))
        return PlainTextResponse("\n".join(parts), media_type="text/plain; charset=utf-8")
    # html or pdf
    body = f"<h1>{_esc(book_title)}</h1><div class='meta'>{len(chapters)} chapters</div>"
    for ch in chapters:
        body += f"<h1>Ch.{ch[1]} {_esc(ch[2] or '')}</h1>"
        for d in _ordered_chapter_drafts(drafts, ch[0]):
            body += _draft_to_html_body(d)
    html = _wrap_html_export(book_title, body)
    if ext == "pdf":
        slug = _slugify_for_filename(book_title) or "book"
        return _html_to_pdf_response(html, f"{slug}.pdf")
    return HTMLResponse(html)


# ── Phase 30: Knowledge Graph browse endpoint ────────────────────────────────
#
# The KG schema (sciknow.storage.models.KnowledgeGraphTriple, table
# `knowledge_graph`) was added by an earlier phase but had zero web
# exposure — only the wiki compile pipeline writes to it. This
# endpoint exposes a simple filtered list view so the user can
# sanity-check the corpus's extracted (subject, predicate, object)
# triples without dropping into psql.

@app.get("/api/kg")
async def api_kg(
    subject: str = "",
    predicate: str = "",
    object: str = "",
    document_id: str = "",
    limit: int = 200,
    offset: int = 0,
):
    """Phase 30 — return knowledge_graph triples filtered by any of:
    subject (substring, case-insensitive), predicate (exact),
    object (substring, case-insensitive), document_id (exact UUID).

    Returns at most `limit` rows (capped at 1000), with pagination via
    offset. Each row has the source paper title joined in for the GUI
    so the user can see which document a triple was extracted from.
    """
    limit = max(1, min(int(limit or 200), 1000))
    offset = max(0, int(offset or 0))
    where = []
    params: dict = {"limit": limit, "offset": offset}
    if subject.strip():
        where.append("kg.subject ILIKE :subject_q")
        params["subject_q"] = f"%{subject.strip()}%"
    if predicate.strip():
        where.append("kg.predicate = :predicate_q")
        params["predicate_q"] = predicate.strip()
    if object.strip():
        where.append("kg.object ILIKE :object_q")
        params["object_q"] = f"%{object.strip()}%"
    if document_id.strip():
        where.append("kg.source_doc_id::text = :doc_q")
        params["doc_q"] = document_id.strip()
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    with get_session() as session:
        # Total count for pagination
        total = session.execute(text(
            f"SELECT COUNT(*) FROM knowledge_graph kg {where_sql}"
        ), params).scalar()

        rows = session.execute(text(f"""
            SELECT kg.subject, kg.predicate, kg.object,
                   kg.source_doc_id::text, kg.confidence,
                   pm.title
            FROM knowledge_graph kg
            LEFT JOIN paper_metadata pm ON pm.document_id = kg.source_doc_id
            {where_sql}
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
            }
            for r in rows
        ],
    }


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


@app.delete("/api/draft/{draft_id}")
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


# ── Chapter reader (continuous scroll) ───────────────────────────────────────

@app.get("/api/chapter-reader/{chapter_id}")
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
        combined_html += _md_to_html(content)
        total_words += d[3] or 0

    sources_html = _render_sources(global_sources)

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


def _titleify_slug_for_display(slug: str) -> str:
    """Local web fallback for unknown slugs (orphaned drafts whose
    section was renamed/deleted). Mirrors core.book_ops._titleify_slug
    but kept here so this module doesn't have to import lazily."""
    return (slug or "").replace("_", " ").strip().title()


# ── Corkboard data ──────────────────────────────────────────────────────────

@app.get("/api/corkboard")
async def corkboard_data():
    """Return data for the corkboard view: cards for each chapter/section.

    Phase 18 — uses each chapter's actual sections list (the user's
    chosen names + order) instead of the previous hardcoded paper-style
    [introduction, methods, results, discussion, conclusion]. Chapters
    with custom sections now show ALL their sections; chapters without
    a sections list show the default science-book set.
    """
    book, chapters, drafts, gaps, comments = _get_book_data()

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
        # Phase 18 — chapter's own sections list (with rich {slug,title,plan})
        section_template = _chapter_sections_dicts(ch)
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
    target_words: int = Form(None),
):
    """Start a write operation, returns job_id for SSE streaming.

    Phase 17 — target_words is optional; when None, book_ops resolves
    it from the book's custom_metadata.target_chapter_words (or the
    default) divided by the chapter's section count.
    """
    from sciknow.core.book_ops import write_section_stream

    job_id, queue = _create_job("write")
    loop = asyncio.get_event_loop()

    def gen():
        return write_section_stream(
            book_id=_book_id, chapter_id=chapter_id,
            section_type=section_type, model=model or None,
            target_words=target_words if target_words and target_words > 0 else None,
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


@app.post("/api/autowrite-chapter")
async def api_autowrite_chapter(
    chapter_id: str = Form(...),
    max_iter: int = Form(3),
    target_score: float = Form(0.85),
    model: str = Form(None),
    target_words: int = Form(None),
    rebuild: bool = Form(False),
    resume: bool = Form(False),
):
    """Phase 20 — autowrite EVERY section of a chapter in sequence.

    The toolbar Autowrite button routes here when the user has a
    chapter selected but no specific section, instead of defaulting
    to a single 'introduction' draft (which doesn't match any of the
    chapter's user-defined sections and creates an orphan).

    The backend generator handles the section iteration, draft skip
    logic, and per-section progress events. This endpoint just kicks
    it off as a job and returns the job_id for SSE streaming.
    """
    from sciknow.core.book_ops import autowrite_chapter_all_sections_stream

    job_id, queue = _create_job("autowrite_chapter")
    loop = asyncio.get_event_loop()

    def gen():
        return autowrite_chapter_all_sections_stream(
            book_id=_book_id, chapter_id=chapter_id,
            model=model or None,
            max_iter=max_iter, target_score=target_score,
            target_words=target_words if target_words and target_words > 0 else None,
            rebuild=rebuild,
            resume=resume,
        )

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
    target_words: int = Form(None),
):
    """Phase 17 — target_words is optional; when None, the effective
    per-section target is resolved from the book's custom_metadata
    (target_chapter_words / num_sections_in_chapter). When set, it
    overrides the book-level value for this run only."""
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
            target_words=target_words if target_words and target_words > 0 else None,
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
        template_dicts = _chapter_sections_dicts(ch)
        template = [s["slug"] for s in template_dicts]
        title_by_slug = {s["slug"]: s["title"] for s in template_dicts}
        plan_by_slug = {s["slug"]: s["plan"] for s in template_dicts}

        # Build a slug → draft index for quick lookup. Multiple drafts
        # per slug are reduced to the highest-version one above.
        draft_by_slug: dict[str, tuple] = {}
        for d in ch_ds:
            slug = _normalize_section(d[2] or "")
            existing = draft_by_slug.get(slug)
            if not existing or (d[6] or 1) > (existing[6] or 1):
                draft_by_slug[slug] = d

        # Phase 21 — sections list now reflects the FULL chapter
        # template, not just sections that already have drafts. Empty
        # template slots get a placeholder entry with id=None and a
        # status="empty" so the sidebar renderer can show a "Write"
        # CTA inline. This makes the chapter's planned structure
        # visible at a glance and removes the "where do I write the
        # next section?" friction.
        sections = []
        seen_slugs: set[str] = set()
        for tmpl in template_dicts:
            slug = tmpl["slug"]
            seen_slugs.add(slug)
            d = draft_by_slug.get(slug)
            if d:
                sections.append({
                    "id": d[0], "type": d[2] or "text",
                    "title": tmpl["title"],
                    "plan": tmpl["plan"],
                    "version": d[6] or 1, "words": d[4] or 0,
                    "status": "drafted",
                })
            else:
                sections.append({
                    "id": None, "type": slug,
                    "title": tmpl["title"],
                    "plan": tmpl["plan"],
                    "version": 0, "words": 0,
                    "status": "empty",
                })
        # Append orphan drafts (drafts whose section_type doesn't match
        # any current template slug) at the end so the user can find
        # and clean them up. They show as "(orphaned)" in the sidebar.
        for slug, d in draft_by_slug.items():
            if slug in seen_slugs:
                continue
            sections.append({
                "id": d[0], "type": d[2] or "text",
                "title": _titleify_slug_for_display(slug),
                "plan": "",
                "version": d[6] or 1, "words": d[4] or 0,
                "status": "orphan",
            })

        sidebar_items.append({
            "num": ch_num, "title": ch_title, "id": ch_id,
            "description": ch_desc or "", "topic_query": tq or "",
            "sections": sections,
            # Phase 14.4 — per-chapter section template for the GUI's
            # empty-state picker and the dashboard heatmap.
            "sections_template": template,
            # Phase 18 — rich {slug, title, plan} dicts for the chapter
            # modal's Sections tab.
            "sections_meta": template_dicts,
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
        # Phase 27 — derive the display title from the chapter sections
        # meta on read so a renamed section shows the new title in the
        # initial page-load h1, not the stale slug-based draft.title.
        sections_raw_for_active = next(
            (ch[6] for ch in chapters if ch[0] == active_draft[9]), None
        )
        active_title = _draft_display_title(
            draft_title=active_draft[1],
            section_type=active_draft[2],
            chapter_num=active_draft[12],
            chapter_title=active_draft[13],
            chapter_sections_raw=sections_raw_for_active,
        )
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
        # Phase 18 — rich section meta for the Sections tab editor
        "sections_meta": si["sections_meta"],
    } for si in sidebar_items])

    # Phase 22 — escape every user-controlled string that lands inside an
    # HTML attribute or text node in the template. The *_html fields are
    # already-rendered HTML produced by helpers that escape internally,
    # so they pass through unchanged. Numeric fields are coerced to int
    # so they can't smuggle markup either.
    return TEMPLATE.format(
        book_title=_esc(book[1] if book else "Untitled"),
        book_id=_esc(str(_book_id)),
        book_plan=_esc((book[3] or "No plan set.") if book else ""),
        sidebar_html=_render_sidebar(sidebar_items, active_id),
        content_html=active_html,
        active_id=_esc(str(active_id)),
        active_title=_esc(active_title),
        active_version=int(active_draft[6]) if active_draft and active_draft[6] is not None else 1,
        active_words=int(active_draft[4]) if active_draft and active_draft[4] is not None else 0,
        active_chapter_id=_esc(str(active_draft[9]) if active_draft else ""),
        active_section_type=_esc(str(active_draft[2]) if active_draft else ""),
        sources_html=_render_sources(active_sources),
        review_html=_md_to_html(active_review) if active_review else "<em>No review yet.</em>",
        comments_html=_render_comments(active_comments),
        gaps_count=int(len(open_gaps)),
        search_q=_esc(search_q or ""),
        search_results_html=_render_search(search_results) if search_results else "",
        chapters_json=chapters_json,
    )


def _render_sidebar(items, active_id):
    """Phase 22 — every user-controlled string (chapter title, section
    title, plan tooltip, draft id) is escaped before injection.

    Also adds the Phase 22 chapter completion progress bar: each
    chapter group shows "x/N drafted" with a small CSS bar derived
    from the section status counts.
    """
    out = ""
    for ch in items:
        ch_id = _esc(str(ch["id"]))
        ch_num = int(ch["num"]) if ch["num"] is not None else 0
        ch_title = _esc(ch["title"] or "")
        out += f'<div class="ch-group" data-ch-id="{ch_id}">'
        # Phase 14.2 — chapter title is clickable to SELECT the chapter.
        # Phase 23 — chevron at the start toggles collapse/expand of
        # the chapter's sections (event.stopPropagation so it doesn't
        # also fire selectChapter). Persistence + restore on page load
        # is handled by JS via localStorage.
        out += (
            f'<div class="ch-title clickable" onclick="selectChapter(this.parentElement)">'
            f'<button class="ch-toggle" '
            f'onclick="event.stopPropagation();toggleChapter(this.closest(&quot;.ch-group&quot;))" '
            f'title="Collapse or expand sections">\u25be</button>'
            f'Ch.{ch_num}: {ch_title}'
            f'<span class="ch-actions">'
            f'<button onclick="event.stopPropagation();deleteChapter(this.closest(&quot;.ch-group&quot;).dataset.chId)" title="Delete chapter">\u2717</button>'
            f'</span></div>'
        )

        # Phase 22 — chapter completion progress bar. Counts only
        # template slots (drafted + empty), excluding orphans which
        # represent dead drafts that shouldn't pad the denominator.
        n_drafted = sum(1 for s in ch["sections"] if s.get("status") == "drafted")
        n_template = sum(1 for s in ch["sections"] if s.get("status") in ("drafted", "empty"))
        if n_template > 0:
            pct = int(round(100 * n_drafted / n_template))
            out += (
                f'<div class="ch-progress" title="{n_drafted} of {n_template} sections drafted">'
                f'<span class="ch-progress-bar"><span class="ch-progress-fill" style="width:{pct}%"></span></span>'
                f'<span class="ch-progress-label">{n_drafted}/{n_template}</span>'
                f'</div>'
            )

        # Phase 21 — render sections from the chapter template (slot list),
        # not just from existing drafts. Each entry has a status:
        #   - drafted: clickable link to the section view
        #   - empty:   shows the section title + an inline write CTA
        #              that calls writeForCell(chapter_id, slug)
        #   - orphan:  draft whose section_type no longer matches any
        #              template slug — visible so the user can clean it up
        for sec in ch["sections"]:
            active = "active" if sec["id"] and sec["id"] == active_id else ""
            display = _esc(sec.get("title") or sec.get("type", "").capitalize())
            plan_text = sec.get("plan") or ""
            plan_attr = (
                _esc(plan_text.replace("\n", " ")[:200]) if plan_text else ""
            )
            status = sec.get("status", "drafted")
            sec_type = _esc(sec.get("type", "") or "")
            sec_id = _esc(str(sec["id"]) if sec["id"] else "")
            sec_v = int(sec.get("version") or 0)
            sec_w = int(sec.get("words") or 0)

            if status == "empty":
                # Phase 26 — draggable for reordering empty slots before drafting.
                # Phase 29 — clicking an empty section now PREVIEWS it
                # (selects + shows title/plan/target in the read-view)
                # instead of immediately triggering doWrite(). The
                # writing only fires after a deliberate click on the
                # "Start writing" button inside the preview.
                out += (
                    f'<div class="sec-link sec-empty" '
                    f'draggable="true" '
                    f'data-section-slug="{sec_type}" '
                    f'title="{plan_attr}" '
                    f'onclick="previewEmptySection(&quot;{ch_id}&quot;,&quot;{sec_type}&quot;)">'
                    f'<span class="sec-status-dot empty"></span>'
                    f'{display}'
                    f'<span class="meta">empty \u00b7 \u270e</span></div>'
                )
            elif status == "orphan":
                # Phase 22 — inline X button on orphan drafts so the
                # user can clean up leftovers from before Phase 18.
                # Phase 25 — also show a "+" button that adopts the
                # slug into the chapter's sections list, re-classifying
                # the draft from "orphan" to "drafted".
                out += (
                    f'<a class="sec-link sec-orphan" href="/section/{sec_id}" '
                    f'data-draft-id="{sec_id}" onclick="return navTo(this)" '
                    f'title="Orphan draft: section_type={sec_type!r} doesn&#39;t match any current template slug. Click to inspect, + to adopt into sections, \u2717 to delete.">'
                    f'<span class="sec-status-dot orphan"></span>'
                    f'{display} '
                    f'<span class="meta">orphan \u00b7 v{sec_v} \u00b7 {sec_w}w</span>'
                    f'<button class="sec-orphan-adopt" '
                    f'onclick="event.preventDefault();event.stopPropagation();adoptOrphanSection(&quot;{ch_id}&quot;,&quot;{sec_type}&quot;)" '
                    f'title="Add this section_type to the chapter\u2019s sections list (idempotent)">+</button>'
                    f'<button class="sec-orphan-delete" '
                    f'onclick="event.preventDefault();event.stopPropagation();deleteOrphanDraft(&quot;{sec_id}&quot;)" '
                    f'title="Delete this orphan draft permanently">\u2717</button>'
                    f'</a>'
                )
            else:
                # Phase 26 — drafted rows are draggable for reordering.
                # The click handler (navTo) still fires on plain clicks;
                # the browser distinguishes click from drag based on
                # whether the cursor moved during mousedown.
                out += (
                    f'<a class="sec-link {active}" href="/section/{sec_id}" '
                    f'draggable="true" '
                    f'data-draft-id="{sec_id}" '
                    f'data-section-slug="{sec_type}" '
                    f'title="{plan_attr}" '
                    f'onclick="return navTo(this)">'
                    f'<span class="sec-status-dot drafted"></span>'
                    f'{display} '
                    f'<span class="meta">v{sec_v} \u00b7 {sec_w}w</span></a>'
                )
        if not ch["sections"]:
            out += (
                f'<div class="sec-link sec-empty-cta" '
                f'onclick="startWritingChapter(&quot;{ch_id}&quot;)">'
                f'\u270e Start writing</div>'
            )
        out += '</div>'
    return out


def _render_sources(sources):
    """Render the right-panel sources list. Phase 22 — escapes each
    source string. Without this, an APA citation containing characters
    like '<' or '&' (rare but possible from a paper title) would either
    silently corrupt the page or open an XSS hole."""
    if not sources:
        return "<em>No sources.</em>"
    out = "<ol>"
    for i, s in enumerate(sources):
        if s:
            out += f'<li id="source-{i+1}">{_esc(s)}</li>'
    out += "</ol>"
    return out


def _render_comments(comments):
    """Render the inline comment list. Phase 22 — every user-controlled
    field (selected text, comment body, comment id used in onclick) is
    escaped before injection. The status string is whitelist-only so
    it doesn't need escaping."""
    if not comments:
        return ""
    out = ""
    for c in comments:
        cid, did, para, sel, comm, status, created = c
        cls = "resolved" if status == "resolved" else "open"
        sel_html = (
            f'<div class="sel-text">&quot;{_esc(sel[:100])}&quot;</div>'
            if sel else ""
        )
        para_html = (
            f'<span class="para-ref">P{int(para)}</span> '
            if para is not None else ""
        )
        resolve_btn = (
            f'<button class="resolve-btn" onclick="resolveComment(&quot;{_esc(str(cid))}&quot;)">Resolve</button>'
            if status == "open" else '<span class="resolved-tag">Resolved</span>'
        )
        comm_html = _esc(comm or "").replace("\n", "<br>")
        out += (
            f'<div class="comment {cls}">{para_html}{sel_html}'
            f'<div class="comm-text">{comm_html}</div>{resolve_btn}</div>'
        )
    return out


def _render_search(results):
    """Phase 22 — escape draft id (used in href + data attributes) and
    title (used in anchor text). The id is a UUID so escaping is
    paranoid but cheap; the title can be anything the user typed."""
    if not results:
        return "<p>No results.</p>"
    out = ""
    for d in results[:20]:
        did = _esc(str(d[0]))
        title = _esc(d[1] or "")
        words = int(d[4] or 0)
        out += (
            f'<a href="/section/{did}" class="search-result" '
            f'data-draft-id="{did}" onclick="return navTo(this)">'
            f'<strong>{title}</strong> ({words} words)</a>'
        )
    return out


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
             transition: background .12s ease; position: relative; }}
.sec-link:hover {{ background: var(--toolbar-bg); }}
.sec-link.active {{ border-left-color: var(--accent); background: var(--accent-light);
                   color: var(--accent); font-weight: 600; }}
.sec-link .meta {{ font-size: 11px; color: var(--fg-faint); margin-left: 6px; }}
.sec-link.empty {{ color: var(--fg-faint); font-style: italic; }}
/* Phase 21 — section status dots in the sidebar. A small coloured circle
   to the LEFT of the section title indicates whether that template slot
   has a draft yet. Empty = grey, drafted = amber, orphan = red. */
.sec-status-dot {{ display: inline-block; width: 7px; height: 7px;
                   border-radius: 50%; margin-right: 6px;
                   vertical-align: middle; flex-shrink: 0; }}
.sec-status-dot.drafted {{ background: var(--success); }}
.sec-status-dot.empty   {{ background: var(--fg-faint); opacity: 0.4; }}
.sec-status-dot.orphan  {{ background: var(--danger); }}
/* Phase 21 — empty template slot in the sidebar (clickable to write).
   Visually distinct from drafted sections so you can see chapter
   structure at a glance: greyed text + write icon in the meta column. */
.sec-link.sec-empty {{ color: var(--fg-muted); }}
.sec-link.sec-empty:hover {{ background: var(--accent-light); color: var(--accent); }}
.sec-link.sec-empty .meta {{ color: var(--fg-faint); font-style: italic; }}
/* Phase 21 — orphan draft (section_type doesn't match any current
   template slug). Visible but de-emphasized so the user can spot
   stale drafts left behind by section renames. */
.sec-link.sec-orphan {{ color: var(--fg-muted); opacity: 0.7;
                        display: flex; align-items: center; }}
.sec-link.sec-orphan:hover {{ opacity: 1; }}
.sec-link.sec-orphan .meta {{ color: var(--danger); flex: 1; }}
/* Phase 22 — inline delete button on orphan drafts. Sits to the right
   of the meta column and only becomes prominent on hover.
   Phase 25 — alongside it, "+" adoption button (left of the X). */
.sec-orphan-delete, .sec-orphan-adopt {{
  background: transparent; border: 1px solid transparent;
  color: var(--fg-faint); cursor: pointer; padding: 0 6px;
  font-size: 13px; border-radius: 3px;
  margin-left: 4px; line-height: 1.4;
}}
.sec-link.sec-orphan:hover .sec-orphan-delete {{ color: var(--danger);
                                                  border-color: var(--danger); }}
.sec-link.sec-orphan:hover .sec-orphan-adopt {{ color: var(--success);
                                                  border-color: var(--success); }}
.sec-orphan-delete:hover {{ background: var(--danger); color: white; }}
.sec-orphan-adopt:hover {{ background: var(--success); color: white; }}
/* Phase 32.4 — inline delete button on regular (drafted/empty)
   sections. Lives at the right edge of the sec-link, hidden until
   hover so it doesn't clutter the sidebar at rest. Same visual
   language as the orphan delete button. */
.sec-delete-btn {{
  background: transparent; border: 1px solid transparent;
  color: var(--fg-faint); cursor: pointer; padding: 0 6px;
  font-size: 12px; border-radius: 3px;
  margin-left: 4px; line-height: 1.4;
  display: none;
}}
.sec-link:hover .sec-delete-btn {{ display: inline-block;
                                   color: var(--danger);
                                   border-color: var(--danger); }}
.sec-delete-btn:hover {{ background: var(--danger); color: white !important; }}
/* Phase 32.4 — "+ Add section" CTA at the bottom of each chapter's
   section list, distinct from the empty-chapter "Start writing" CTA. */
.sec-link.sec-add-cta {{ color: var(--fg-muted); font-style: normal;
                         cursor: pointer; padding: 6px var(--sp-4) 6px 28px;
                         font-size: 12px; }}
.sec-link.sec-add-cta:hover {{ background: var(--accent-light);
                               color: var(--accent); }}
/* Phase 26 — drag-and-drop section reordering. Sections become
   draggable=true; CSS gives the dragged row a dimmed look and the
   target row a coloured top/bottom border to show where the drop
   will land. Within-chapter only; the JS handler enforces that. */
.sec-link[draggable="true"] {{ cursor: grab; }}
.sec-link[draggable="true"]:active {{ cursor: grabbing; }}
.sec-link.dragging {{ opacity: 0.4; cursor: grabbing; }}
.sec-link.drag-over-top {{ box-shadow: inset 0 2px 0 0 var(--accent); }}
.sec-link.drag-over-bottom {{ box-shadow: inset 0 -2px 0 0 var(--accent); }}
/* Phase 23 / Phase 25 — collapse/expand chapter sections. Each chapter
   has a chevron button at the start of its title; clicking toggles a
   .collapsed class on the .ch-group, which hides the section list AND
   the progress bar. State persists in localStorage.
   Phase 25: bumped from 10px/fg-muted (invisible) to 13px/fg with
   accent on hover so the chevron is actually findable in the sidebar. */
.ch-toggle {{ background: transparent; border: none; color: var(--fg);
             font-size: 13px; cursor: pointer; padding: 2px 4px;
             margin-right: 2px; line-height: 1; opacity: 0.7;
             transition: transform 0.15s ease, opacity 0.12s ease, color 0.12s ease;
             display: inline-block; width: 18px; height: 18px;
             vertical-align: middle; border-radius: 3px; }}
.ch-toggle:hover {{ color: var(--accent); opacity: 1;
                   background: var(--accent-light); }}
.ch-group.collapsed .ch-toggle {{ transform: rotate(-90deg); }}
.ch-group.collapsed .sec-link,
.ch-group.collapsed .ch-progress {{ display: none; }}
/* Phase 23 — sidebar header collapse/expand-all button. */
.sidebar-controls {{ padding: 6px var(--sp-4); border-bottom: 1px solid var(--border);
                    display: flex; justify-content: flex-end; }}
.sidebar-toggle-all {{ background: transparent; border: 1px solid var(--border);
                      color: var(--fg-muted); padding: 3px 10px;
                      border-radius: var(--r-sm); cursor: pointer;
                      font-size: 11px; font-family: var(--font-sans);
                      display: inline-flex; align-items: center; gap: 4px; }}
.sidebar-toggle-all:hover {{ color: var(--accent); border-color: var(--accent); }}
/* Phase 22 — chapter completion progress bar. Lives between the chapter
   title and its first section in the sidebar. Greyscale so it doesn't
   compete with the active section's accent color. */
.ch-progress {{ display: flex; align-items: center; gap: 8px;
               padding: 2px var(--sp-4) 6px 28px; }}
.ch-progress-bar {{ flex: 1; height: 4px; background: var(--border);
                   border-radius: 2px; overflow: hidden; }}
.ch-progress-fill {{ display: block; height: 100%; background: var(--success);
                    border-radius: 2px; transition: width .3s ease; }}
.ch-progress-label {{ font-size: 10px; color: var(--fg-faint);
                     font-family: var(--font-mono); flex-shrink: 0;
                     min-width: 32px; text-align: right; }}
/* Phase 22 — section word-target progress shown in the draft subtitle.
   Visible only when a target is set on the active draft. */
.word-target {{ display: inline-flex; align-items: center; gap: 6px;
               margin-left: var(--sp-3); font-family: var(--font-mono);
               font-size: 11px; color: var(--fg-muted); }}
.word-target-bar {{ display: inline-block; width: 80px; height: 4px;
                   background: var(--border); border-radius: 2px;
                   overflow: hidden; vertical-align: middle; }}
.word-target-fill {{ display: block; height: 100%; background: var(--accent);
                    border-radius: 2px; transition: width .3s ease; }}
.word-target-fill.over {{ background: var(--success); }}
.word-target-fill.under {{ background: var(--warning); }}
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
/* Phase 15.6 — live writing preview rendered into the main read-view */
.live-writing-banner {{ display: flex; align-items: center; gap: var(--sp-2);
                       padding: var(--sp-2) var(--sp-3); margin-bottom: var(--sp-4);
                       background: var(--accent-light); color: var(--accent);
                       border: 1px solid var(--accent); border-radius: var(--r-md);
                       font-size: 12px; font-family: var(--font-sans);
                       font-weight: 500; }}
.live-writing-body {{ font-family: var(--font-serif); font-size: 16px;
                     line-height: 1.78; color: var(--fg); }}
.live-writing-body p {{ margin-bottom: var(--sp-3); text-align: justify; }}
.live-writing-body p .citation {{ color: var(--accent); font-weight: 600; }}
/* Phase 18 — Sections tab in chapter modal */
.sec-row {{ display: flex; gap: var(--sp-2); align-items: flex-start;
             padding: var(--sp-3); border: 1px solid var(--border);
             border-radius: var(--r-md); background: var(--bg-elevated);
             margin-bottom: var(--sp-2); }}
.sec-row .sec-handle {{ display: flex; flex-direction: column; gap: 2px;
                        flex-shrink: 0; padding-top: 4px; }}
.sec-row .sec-handle button {{ background: transparent; border: 1px solid var(--border);
                               color: var(--fg-muted); cursor: pointer;
                               width: 22px; height: 18px; font-size: 10px;
                               border-radius: 3px; padding: 0;
                               display: flex; align-items: center; justify-content: center; }}
.sec-row .sec-handle button:hover {{ color: var(--accent); border-color: var(--accent); }}
.sec-row .sec-fields {{ flex: 1; display: flex; flex-direction: column; gap: 6px;
                       min-width: 0; }}
.sec-row .sec-fields input,
.sec-row .sec-fields textarea {{ width: 100%; padding: 6px 10px; font-size: 13px;
                                 background: var(--bg); border: 1px solid var(--border);
                                 border-radius: var(--r-sm); color: var(--fg);
                                 font-family: var(--font-sans); }}
.sec-row .sec-fields textarea {{ min-height: 60px; resize: vertical;
                                 font-family: var(--font-sans); }}
.sec-row .sec-slug {{ font-family: var(--font-mono); font-size: 11px;
                     color: var(--fg-muted); padding: 0 0 0 4px; }}
/* Phase 29 — per-section size dropdown row */
.sec-row .sec-size-row {{ display: flex; align-items: center; gap: 6px;
                         font-size: 12px; padding: 2px 0 0 0; }}
.sec-row .sec-size-row label {{ color: var(--fg-muted); }}
.sec-row .sec-size-row select {{ padding: 3px 6px; font-size: 12px;
                                background: var(--bg); border: 1px solid var(--border);
                                border-radius: var(--r-sm); color: var(--fg);
                                font-family: var(--font-sans); }}
.sec-row .sec-size-row input.sec-size-custom {{ width: 80px; padding: 3px 6px;
                                font-size: 12px; background: var(--bg);
                                border: 1px solid var(--border);
                                border-radius: var(--r-sm); color: var(--fg);
                                font-family: var(--font-sans); }}
/* Phase 32.1 — visible per-section target badge so the user can
   always see what word budget THIS section will be written to. */
.sec-row .sec-target-badge {{ display: inline-flex; align-items: center;
                              gap: 4px; padding: 2px 8px; margin-left: 6px;
                              font-size: 11px; font-weight: 600;
                              color: var(--fg); background: var(--toolbar-bg);
                              border: 1px solid var(--border);
                              border-radius: 10px; }}
.sec-row .sec-target-badge.override {{ color: var(--accent);
                                       border-color: var(--accent); }}
.sec-row .sec-target-badge .badge-tag {{ font-size: 10px;
                                         font-weight: 400;
                                         color: var(--fg-muted);
                                         text-transform: uppercase;
                                         letter-spacing: 0.5px; }}
.sec-row .sec-target-badge .badge-tag.muted {{ opacity: 0.7; }}
.sec-row .sec-delete {{ background: transparent; border: 1px solid var(--border);
                       color: var(--fg-muted); cursor: pointer; padding: 4px 8px;
                       border-radius: var(--r-sm); font-size: 11px; flex-shrink: 0;
                       align-self: flex-start; }}
.sec-row .sec-delete:hover {{ color: var(--danger); border-color: var(--danger); }}
.hm-cell {{ border-radius: 4px; padding: 4px 8px; cursor: pointer; display: inline-block;
            min-width: 44px; font-size: 11px; }}
.hm-cell.reviewed {{ background: var(--success); color: white; }}
.hm-cell.drafted {{ background: var(--warning); color: white; }}
.hm-cell.empty {{ background: var(--border); opacity: 0.5; }}
/* Phase 30 — chapter has fewer sections than max(N); blank cell. */
.hm-cell.absent {{ background: transparent; color: var(--fg-faint);
                  opacity: 0.25; cursor: default; }}
.hm-cell:hover {{ opacity: 0.85; }}
.hm-cell.absent:hover {{ opacity: 0.25; }}
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
/* Phase 20 — orphan/broken citation: source ref doesn't exist in the
   sources panel. Visually obvious so the user knows it's a data issue
   not a click bug. Click is suppressed (cursor not-allowed). */
.citation.citation-broken {{ color: var(--danger); text-decoration: line-through;
                              background: var(--danger-light); cursor: not-allowed; }}
.citation.citation-broken::after {{ content: " \26A0"; font-size: 10px;
                                     vertical-align: super; }}
.citation.verified-supported {{ background: #d1fae5; border-radius: 3px; }}
.citation.verified-extrapolated {{ background: #fef3c7; border-radius: 3px; }}
.citation.verified-overstated {{ background: #ffedd5; border-radius: 3px; }}
.citation.verified-misrepresented {{ background: #fee2e2; border-radius: 3px; }}
[data-theme="dark"] .citation.verified-supported {{ background: #064e3b; }}
[data-theme="dark"] .citation.verified-extrapolated {{ background: #78350f; }}
[data-theme="dark"] .citation.verified-overstated {{ background: #7c2d12; }}
[data-theme="dark"] .citation.verified-misrepresented {{ background: #7f1d1d; }}
/* Phase 30 — Export modal grid + KG table */
.export-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr;
               gap: 16px; margin-top: 8px; }}
.export-grid > div {{ padding: 14px; border: 1px solid var(--border);
                     border-radius: 6px; background: var(--toolbar-bg); }}
.export-grid h4 {{ margin: 0 0 6px; font-size: 13px; color: var(--fg); }}
.export-grid .export-target {{ font-size: 11px; color: var(--fg-muted);
                              margin: 0 0 10px; min-height: 14px;
                              overflow: hidden; text-overflow: ellipsis;
                              white-space: nowrap; }}
.export-grid .export-btns {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.export-grid .export-btns a {{ padding: 4px 10px; font-size: 11px;
                              text-decoration: none; border-radius: 4px;
                              background: var(--bg); color: var(--accent);
                              border: 1px solid var(--border); }}
.export-grid .export-btns a:hover {{ background: var(--accent);
                                     color: white; border-color: var(--accent); }}
.export-grid .export-btns a.disabled {{ opacity: 0.4; cursor: not-allowed;
                                        pointer-events: none; }}
.kg-table {{ width: 100%; border-collapse: collapse; font-size: 12px;
            font-family: var(--font-sans); }}
.kg-table th, .kg-table td {{ padding: 6px 8px; text-align: left;
                             border-bottom: 1px solid var(--border); }}
.kg-table th {{ background: var(--toolbar-bg); font-weight: 600;
              color: var(--fg-muted); position: sticky; top: 0; }}
.kg-table .kg-pred {{ color: var(--accent); font-family: var(--font-mono);
                    font-size: 11px; }}
.kg-table .kg-source {{ color: var(--fg-muted); font-size: 10px;
                      max-width: 220px; overflow: hidden;
                      text-overflow: ellipsis; white-space: nowrap; }}
/* Phase 31 — KG force-directed graph (SVG nodes + edges) */
#kg-graph-canvas {{ width: 100%; height: 520px; cursor: grab; }}
#kg-graph-canvas svg {{ width: 100%; height: 100%; display: block; }}
#kg-graph-canvas .kg-node circle {{ fill: var(--accent); stroke: var(--bg-elevated);
                                    stroke-width: 2; cursor: pointer;
                                    transition: r 0.15s ease, fill 0.15s ease; }}
#kg-graph-canvas .kg-node:hover circle {{ fill: var(--accent-hover); r: 9; }}
#kg-graph-canvas .kg-node text {{ font-size: 10px; fill: var(--fg);
                                  pointer-events: none;
                                  font-family: var(--font-sans); }}
#kg-graph-canvas .kg-edge {{ stroke: var(--fg-muted); stroke-width: 1;
                            opacity: 0.5; }}
#kg-graph-canvas .kg-edge.highlighted {{ stroke: var(--accent);
                                         stroke-width: 2; opacity: 1; }}
#kg-graph-canvas .kg-edge-label {{ font-size: 8px; fill: var(--fg-muted);
                                   pointer-events: none;
                                   font-family: var(--font-mono); }}
#kg-graph-canvas .kg-node.selected circle {{ fill: var(--success);
                                             stroke: var(--success-light);
                                             stroke-width: 3; r: 10; }}
/* Phase 30 — persistent global task bar (top of viewport, full width).
   Visible whenever a job is running, regardless of SPA navigation.
   Designed to be unobtrusive: ~40px tall, mono font for the numerics,
   accent colour only for the activity dot.
   Phase 32.3 — must be `position: fixed`, NOT `position: sticky`. The
   <body> is a horizontal flex container (sidebar + main), and a
   sticky child of a horizontal flex container becomes a flex column
   sibling — which is exactly what the user reported as "appears as
   a left column instead of a top bar". `position: fixed` takes the
   bar out of the flex flow and floats it across the top; the
   `.task-bar-open` class on <body> adds matching padding-top so the
   sidebar/main aren't covered by the bar. */
.task-bar {{ position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
            display: flex; align-items: center; gap: 8px;
            padding: 8px 16px; min-height: 40px;
            background: var(--bg-elevated);
            border-bottom: 1px solid var(--border-strong);
            font-family: var(--font-mono); font-size: 12px;
            color: var(--fg); box-shadow: 0 2px 8px rgba(0,0,0,0.04); }}
body.task-bar-open {{ padding-top: 40px; }}
.task-bar .tb-dot {{ width: 10px; height: 10px; border-radius: 50%;
                    background: var(--success); flex-shrink: 0;
                    animation: pulse 1.2s infinite; }}
.task-bar .tb-dot.error {{ background: var(--danger); animation: none; }}
.task-bar .tb-dot.done  {{ background: var(--success); animation: none; }}
.task-bar .tb-dot.idle  {{ background: var(--fg-faint); animation: none; }}
.task-bar .tb-task {{ font-family: var(--font-sans); color: var(--fg);
                     font-weight: 600; max-width: 360px;
                     overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.task-bar .tb-stat {{ display: inline-flex; align-items: center; gap: 4px; }}
.task-bar .tb-stat strong {{ color: var(--accent); font-weight: 700; }}
.task-bar .tb-sep {{ color: var(--border-strong); }}
.task-bar .tb-spacer {{ flex: 1; }}
.task-bar .tb-stop, .task-bar .tb-dismiss {{ background: var(--danger);
                    border: none; color: white; cursor: pointer;
                    padding: 4px 12px; border-radius: var(--r-sm);
                    font-size: 11px; font-family: var(--font-sans);
                    font-weight: 600; }}
.task-bar .tb-stop:hover, .task-bar .tb-dismiss:hover {{ background: #b91c1c; }}
.task-bar .tb-dismiss {{ background: var(--fg-muted); }}
.task-bar .tb-dismiss:hover {{ background: var(--fg); }}
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

<!-- Phase 30 — persistent global task bar.
     Hidden by default; shown when startGlobalJob() runs. Lives at
     the very top of the body so it survives all SPA navigation
     (loadSection, showDashboard, showCorkboard, etc) and never gets
     overwritten by innerHTML rebuilds of the main content area. -->
<div id="task-bar" class="task-bar" style="display:none;">
  <span class="tb-dot" id="tb-dot"></span>
  <span class="tb-task" id="tb-task">…</span>
  <span class="tb-sep">·</span>
  <span class="tb-stat"><span id="tb-model">?</span></span>
  <span class="tb-sep">·</span>
  <span class="tb-stat"><strong id="tb-tokens">0</strong>&nbsp;tok</span>
  <span class="tb-sep">·</span>
  <span class="tb-stat"><strong id="tb-tps">0.0</strong>&nbsp;tok/s</span>
  <span class="tb-sep">·</span>
  <span class="tb-stat" id="tb-elapsed">0s</span>
  <span class="tb-stat tb-eta" id="tb-eta" style="display:none;">
    <span class="tb-sep">·</span>ETA <strong id="tb-eta-val">?</strong>
  </span>
  <span class="tb-spacer"></span>
  <button class="tb-stop" id="tb-stop" onclick="stopGlobalJob()" title="Stop the running task">&#9632; Stop</button>
  <button class="tb-dismiss" id="tb-dismiss" onclick="dismissTaskBar()" title="Dismiss" style="display:none;">&times;</button>
</div>

<!-- Sidebar -->
<nav class="sidebar">
  <h2>{book_title}</h2>
  <div class="search-bar">
    <form action="/search" method="get">
      <input type="text" name="q" placeholder="Search..." value="{search_q}">
    </form>
  </div>
  {search_results_html}
  <!-- Phase 23 — collapse / expand all chapters in the sidebar.
       The icon flips from \u25be (down arrow, expanded) to \u25b8
       (right arrow, collapsed) to mirror the per-chapter chevrons. -->
  <div class="sidebar-controls">
    <button class="sidebar-toggle-all" onclick="toggleAllChapters()"
            title="Collapse or expand all chapter sections">
      <span id="toggle-all-icon">\u25bd</span>
      <span id="toggle-all-label">Collapse all</span>
    </button>
  </div>
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
    Version <span id="draft-version">{active_version}</span> &middot;
    <span id="draft-words">{active_words}</span> words
    <!-- Phase 22 — word target progress (shown only when a target is set) -->
    <span class="word-target" id="word-target" style="display:none;">
      <span id="word-target-text"></span>
      <span class="word-target-bar"><span class="word-target-fill" id="word-target-fill"></span></span>
    </span>
    <button class="edit-btn" onclick="toggleEdit()">Edit</button>
    <select class="status-select" id="status-select" onchange="updateStatus(this.value)">
      <option value="to_do">To Do</option>
      <option value="drafted" selected>Drafted</option>
      <option value="reviewed">Reviewed</option>
      <option value="revised">Revised</option>
      <option value="final">Final</option>
    </select>
  </div>

  <!-- Action Toolbar — Phase 14 v2 grouped layout
       Phase 31 — split AI actions from manual editing so the user
       can find the inline editor without hunting for the small
       edit-btn in the subtitle. -->
  <div class="toolbar" id="toolbar">
    <div class="tg">
      <button class="primary" onclick="toggleEdit()" title="Manually edit the draft content (in-browser markdown editor with autosave)">&#9998; Edit</button>
      <button onclick="doAutowrite()" title="Autonomous AI write → review → revise loop">&#9889; AI Autowrite</button>
      <button onclick="doWrite()" title="AI drafts this section from scratch (single pass)">AI Write</button>
      <button onclick="doReview()" title="AI critic pass on this section">AI Review</button>
      <button onclick="doRevise()" title="AI revises based on review feedback">AI Revise</button>
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
      <button onclick="openKgModal()" title="Browse the knowledge graph (extracted entity-relationship triples)">&#128279; KG</button>
      <button onclick="openCatalogModal()" title="Browse the paper catalog (sciknow catalog list)">&#128194; Browse Papers</button>
    </div>
    <div class="sep"></div>
    <div class="tg">
      <button onclick="showVersions()" title="View version history and diffs">History</button>
      <button onclick="takeSnapshot()" title="Save a snapshot of current content">Snapshot</button>
      <button onclick="openExportModal()" title="Export this section, chapter, or the whole book to text or printable HTML/PDF">&#128229; Export</button>
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

<!-- Phase 14.3 — Book Plan Modal
     Phase 21 — context-aware: tabs for Book / Chapter / Section -->
<div class="modal-overlay" id="plan-modal" onclick="if(event.target===this)closeModal('plan-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3>&#128221; Plans</h3>
      <button class="modal-close" onclick="closeModal('plan-modal')">&times;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="plan-book" onclick="switchPlanTab('plan-book')">Book</button>
      <button class="tab" data-tab="plan-chapter" onclick="switchPlanTab('plan-chapter')" id="plan-tab-chapter">Chapter sections</button>
      <button class="tab" data-tab="plan-section" onclick="switchPlanTab('plan-section')" id="plan-tab-section">Section</button>
    </div>
    <div class="modal-body">
      <!-- Book tab — the leitmotiv (existing) -->
      <div class="tab-pane active" id="plan-book-pane">
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
        <div class="field">
          <label>Target chapter length &mdash; autowrite &amp; write aim for this many words per chapter</label>
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
            <button type="button" class="btn-secondary" onclick="setLengthPreset(3000)">Short &middot; 3000</button>
            <button type="button" class="btn-secondary" onclick="setLengthPreset(6000)">Standard &middot; 6000</button>
            <button type="button" class="btn-secondary" onclick="setLengthPreset(10000)">Long &middot; 10000</button>
            <input type="number" id="plan-target-words-input" min="0" step="500" placeholder="custom"
                   style="width:120px;padding:6px 8px;font-size:13px;"
                   title="Words per chapter. Leave empty for the default (6000). Zero clears the setting.">
            <span id="plan-length-status" style="font-size:12px;color:var(--fg-muted);"></span>
          </div>
          <p style="font-size:11px;color:var(--fg-muted);margin-top:6px;">
            Each section gets a proportional share: a 4-section chapter at 6000
            words asks the writer for ~1500 words per section. In autowrite,
            length becomes a 7th scoring dimension &mdash; drafts under ~70% of
            target trigger a targeted expansion revision.
          </p>
        </div>
      </div>
      <!-- Chapter tab — Phase 21: shows the active chapter's title +
           description + the full ordered list of section plans
           (read+edit). Save updates each section's plan via PUT
           /api/chapters/{{id}}/sections. -->
      <div class="tab-pane" id="plan-chapter-pane" style="display:none;">
        <p style="font-size:12px;color:var(--fg-muted);margin-bottom:12px;">
          The chapter view shows the book leitmotiv at the top (read-only summary)
          plus the full plan for every section in the chapter, in order. Edit any
          section&rsquo;s plan inline and click Save to persist.
        </p>
        <div id="plan-chapter-header" style="margin-bottom:14px;font-size:13px;color:var(--fg-muted);"></div>
        <div id="plan-chapter-sections"></div>
      </div>
      <!-- Section tab — Phase 21: focused single-section plan editor.
           Opens here when the user clicks Plan from a section view. -->
      <div class="tab-pane" id="plan-section-pane" style="display:none;">
        <p style="font-size:12px;color:var(--fg-muted);margin-bottom:12px;">
          The section plan tells the writer what THIS specific section must cover,
          narrower than the chapter description. Injected into the writer prompt as
          a &ldquo;Section plan&rdquo; block above the book plan and prior chapter
          summaries.
        </p>
        <div id="plan-section-context" style="font-size:12px;color:var(--fg-muted);margin-bottom:8px;"></div>
        <div class="field">
          <label>Section title</label>
          <input type="text" id="plan-section-title">
        </div>
        <div class="field">
          <label>Section plan (a few sentences)</label>
          <textarea id="plan-section-text" style="min-height:200px;"></textarea>
        </div>
        <!-- Phase 32.2 — per-section length dropdown. Mirrors the
             chapter modal Sections tab so the user can pick a
             length while editing the focused section plan. -->
        <div class="field">
          <label>Target length for this section</label>
          <div class="sec-size-row" style="padding-top:4px;">
            <select id="plan-section-target-select" onchange="updatePlanSectionTargetWords(this.value)">
              <option value="">Auto (chapter target / num sections)</option>
              <option value="400">Very short (~400w)</option>
              <option value="800">Short (~800w)</option>
              <option value="1500">Medium (~1500w)</option>
              <option value="3000">Long (~3000w)</option>
              <option value="6000">Extra long (~6000w)</option>
              <option value="custom">Custom&hellip;</option>
            </select>
            <input type="number" id="plan-section-target-custom" class="sec-size-custom" placeholder="words"
                   min="100" step="100" style="display:none;"
                   oninput="updatePlanSectionTargetWordsCustom(this.value)">
            <span id="plan-section-target-badge" class="sec-target-badge"></span>
          </div>
        </div>
      </div>
      <div id="plan-status" style="font-size:12px;color:var(--fg-muted);margin-bottom:8px;"></div>
      <div id="plan-stream-stats" class="stream-stats"></div>
    </div>
    <div class="modal-footer">
      <button class="btn-secondary" onclick="closeModal('plan-modal')">Close</button>
      <button class="btn-secondary" onclick="regeneratePlan()" id="plan-regen-btn">&#9889; Regenerate with LLM</button>
      <button class="btn-primary" onclick="savePlan()">Save</button>
    </div>
  </div>
</div>

<!-- Phase 14.3 — Chapter Info Modal (description + topic_query)
     Phase 18 — Tabs: Scope (existing) + Sections (new) -->
<div class="modal-overlay" id="chapter-modal" onclick="if(event.target===this)closeModal('chapter-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3>&#9881; Chapter</h3>
      <button class="modal-close" onclick="closeModal('chapter-modal')">&times;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="ch-scope" onclick="switchChapterTab('ch-scope')">Scope</button>
      <button class="tab" data-tab="ch-sections" onclick="switchChapterTab('ch-sections')">Sections</button>
    </div>
    <div class="modal-body">
      <!-- Scope tab -->
      <div class="tab-pane active" id="ch-scope-pane">
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
      </div>
      <!-- Sections tab -->
      <div class="tab-pane" id="ch-sections-pane">
        <p style="font-size:12px;color:var(--fg-muted);margin-bottom:12px;">
          A chapter is broken into named sections. Each section becomes its own
          draft when you click <strong>Write</strong> or <strong>Autowrite</strong>,
          and gets a proportional share of the chapter&rsquo;s word target. The
          plan tells the writer what THIS section must cover &mdash; narrower
          than the chapter description. Reorder with the &uarr;/&darr; buttons.
          Renaming a section <strong>does not</strong> rename existing drafts;
          they keep their old slug until rewritten.
        </p>
        <div id="ch-sections-list"></div>
        <button class="btn-secondary" onclick="addSection()" style="margin-top:8px;">
          &#43; Add section
        </button>
      </div>
      <div id="chapter-modal-status" style="font-size:12px;color:var(--fg-muted);margin:8px 0;"></div>
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

<!-- Phase 30/31 — Knowledge Graph Modal with Graph + Table tabs -->
<div class="modal-overlay" id="kg-modal" onclick="if(event.target===this)closeModal('kg-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3>&#128279; Knowledge Graph</h3>
      <button class="modal-close" onclick="closeModal('kg-modal')">&times;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="kg-graph" onclick="switchKgTab('kg-graph')">Graph</button>
      <button class="tab" data-tab="kg-table" onclick="switchKgTab('kg-table')">Table</button>
    </div>
    <div class="modal-body">
      <p style="font-size:11px;color:var(--fg-muted);margin-bottom:8px;">
        Entity-relationship triples extracted from the corpus during wiki compile.
        Filter by subject substring, predicate (exact match), object substring,
        or document id. <strong>Graph</strong> shows up to 100 triples as nodes
        and edges; <strong>Table</strong> is searchable and shows up to 200.
      </p>
      <div class="field" style="display:flex;gap:8px;align-items:flex-end;">
        <div style="flex:2;">
          <label>Subject contains</label>
          <input type="text" id="kg-subject" placeholder="(any)" onkeydown="if(event.key==='Enter')loadKg(0)">
        </div>
        <div style="flex:1;">
          <label>Predicate</label>
          <select id="kg-predicate" onchange="loadKg(0)">
            <option value="">(any)</option>
          </select>
        </div>
        <div style="flex:2;">
          <label>Object contains</label>
          <input type="text" id="kg-object" placeholder="(any)" onkeydown="if(event.key==='Enter')loadKg(0)">
        </div>
        <button class="btn-primary" onclick="loadKg(0)">Filter</button>
      </div>
      <div id="kg-status" style="font-size:11px;color:var(--fg-muted);margin:8px 0;"></div>
      <!-- Graph tab pane (default) -->
      <div id="kg-graph-pane" style="display:block;">
        <div id="kg-graph-canvas" style="border:1px solid var(--border);border-radius:6px;background:var(--toolbar-bg);"></div>
        <p style="font-size:10px;color:var(--fg-muted);margin-top:6px;">
          &middot; Drag nodes to reposition. Click any node to filter the table to triples involving that entity. Showing the top 100 highest-confidence triples in the current filter.
        </p>
      </div>
      <!-- Table tab pane -->
      <div id="kg-table-pane" style="display:none;max-height:60vh;overflow-y:auto;">
        <div id="kg-results"></div>
      </div>
    </div>
  </div>
</div>

<!-- Phase 30 — Export Modal -->
<div class="modal-overlay" id="export-modal" onclick="if(event.target===this)closeModal('export-modal')">
  <div class="modal">
    <div class="modal-header">
      <h3>&#128229; Export</h3>
      <button class="modal-close" onclick="closeModal('export-modal')">&times;</button>
    </div>
    <div class="modal-body">
      <p style="font-size:12px;color:var(--fg-muted);margin-bottom:14px;">
        <strong>PDF</strong> is rendered server-side via weasyprint &mdash; ready for
        printing or sharing. <strong>HTML</strong> is the same content with print-friendly
        CSS, useful as a fallback or for editing in another tool.
        <strong>Markdown</strong> preserves all formatting and citations.
        <strong>Text</strong> is plain prose with the markdown stripped.
      </p>
      <div class="export-grid">
        <div>
          <h4>This section</h4>
          <p class="export-target" id="export-section-name">(no draft selected)</p>
          <div class="export-btns" id="export-section-btns"></div>
        </div>
        <div>
          <h4>This chapter</h4>
          <p class="export-target" id="export-chapter-name">(no chapter selected)</p>
          <div class="export-btns" id="export-chapter-btns"></div>
        </div>
        <div>
          <h4>Whole book</h4>
          <p class="export-target" id="export-book-name">&nbsp;</p>
          <div class="export-btns" id="export-book-btns"></div>
        </div>
      </div>
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
// Phase 32.4 — needs to be `let` (was `const`) so deleteSection /
// addSectionToChapter can refresh the in-memory cache after a PUT.
let chaptersData = {chapters_json};

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
    // Phase 15.6 — clear any in-progress live preview when navigating
    // to a different section. The saved draft we're about to load will
    // replace the read-view content anyway.
    clearLiveWrite();
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
    // Phase 27 — prefer the display_title computed from the chapter
    // sections meta so a renamed section shows the new title in the
    // center h1 instead of the stale slug-based drafts.title snapshot.
    document.getElementById('draft-title').textContent = data.display_title || data.title;
    document.getElementById('draft-version').textContent = data.version;
    document.getElementById('draft-words').textContent = data.word_count;
    document.getElementById('read-view').innerHTML = data.content_html;
    document.getElementById('edit-view').style.display = 'none';

    // Phase 22 — word target progress bar in the subtitle. Hidden when
    // the section has no target set.
    updateWordTargetBar(data.word_count, data.target_words);

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
      // Phase 15.6 — also stream into the main read-view live preview
      // for `book write` (not just autowrite). The token has no `phase`
      // tag here because plain `book write` doesn't have multi-phase
      // streaming, so all tokens are writer tokens by definition.
      appendLiveWrite(evt.text);
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
  // Phase 30 — also notify the global task bar so the user gets
  // immediate visual feedback even when the inner stop button is
  // pressed instead of the global one.
  stopGlobalJob();
}}

// ── Phase 30: persistent global task bar ──────────────────────────────
//
// One global SSE source per job, NOT closed by loadSection or any
// SPA navigation. The task bar at the top of the viewport reflects
// the live state. doAutowrite/doWrite/etc all call startGlobalJob()
// after kicking off their HTTP request; the bar handles its own
// rendering loop and stop button.

let _globalJob = null;
let _globalJobSource = null;
let _globalJobTimer = null;

function _formatElapsed(ms) {{
  const s = Math.floor(ms / 1000);
  if (s < 60) return s + 's';
  const m = Math.floor(s / 60);
  if (m < 60) return m + 'm ' + (s % 60).toString().padStart(2, '0') + 's';
  // Phase 30 — beyond 60 minutes show hours + minutes (per user request)
  const h = Math.floor(m / 60);
  return h + 'h ' + (m % 60).toString().padStart(2, '0') + 'm';
}}

function _renderTaskBar() {{
  const j = _globalJob;
  if (!j) return;
  const bar = document.getElementById('task-bar');
  if (!bar) return;
  bar.style.display = 'flex';
  // Phase 32.3 — body class adds padding-top so the fixed-position
  // task bar doesn't cover the sidebar/main top edge.
  document.body.classList.add('task-bar-open');
  document.getElementById('tb-task').textContent = j.taskDesc || j.type || 'Working';
  document.getElementById('tb-task').title = j.taskDesc || j.type || '';
  document.getElementById('tb-model').textContent = j.modelName || 'qwen3.5:27b';
  document.getElementById('tb-tokens').textContent = j.tokens.toLocaleString();
  // Rolling t/s over the last 3 seconds
  const now = performance.now();
  const cutoff = now - 3000;
  while (j.recentTokens.length > 0 && j.recentTokens[0] < cutoff) {{
    j.recentTokens.shift();
  }}
  const tps = j.recentTokens.length / 3;
  document.getElementById('tb-tps').textContent = tps.toFixed(1);
  const elapsed = j.startedAt ? (now - j.startedAt) : 0;
  document.getElementById('tb-elapsed').textContent = _formatElapsed(elapsed);
  // ETA — only when target_words is known and tokens are flowing
  const etaWrap = document.getElementById('tb-eta');
  if (j.targetWords && tps > 0.1 && j.tokens > 0) {{
    const remaining = Math.max(0, j.targetWords - j.tokens);
    const etaMs = (remaining / tps) * 1000;
    document.getElementById('tb-eta-val').textContent = _formatElapsed(etaMs);
    etaWrap.style.display = 'inline-flex';
  }} else {{
    etaWrap.style.display = 'none';
  }}
  // Dot state
  const dot = document.getElementById('tb-dot');
  dot.className = 'tb-dot ' + (j.state || 'streaming');
}}

function startGlobalJob(jobId, opts) {{
  if (!jobId) return;
  // Clean up any previous job
  if (_globalJobSource) {{
    try {{ _globalJobSource.close(); }} catch (e) {{}}
    _globalJobSource = null;
  }}
  if (_globalJobTimer) {{
    clearInterval(_globalJobTimer);
    _globalJobTimer = null;
  }}

  _globalJob = {{
    id: jobId,
    type: (opts && opts.type) || 'job',
    taskDesc: (opts && opts.taskDesc) || 'Running…',
    modelName: (opts && opts.modelName) || 'qwen3.5:27b',
    targetWords: (opts && opts.targetWords) || null,
    tokens: 0,
    recentTokens: [],
    startedAt: performance.now(),
    state: 'streaming',
    sectionType: (opts && opts.sectionType) || null,
    chapterId: (opts && opts.chapterId) || null,
  }};

  // Show buttons in their starting state
  document.getElementById('tb-stop').style.display = '';
  document.getElementById('tb-dismiss').style.display = 'none';

  _renderTaskBar();
  // Re-render every 500ms so elapsed/t/s/eta tick in real time even
  // when no events are flowing (e.g. during a long retrieval phase)
  _globalJobTimer = setInterval(_renderTaskBar, 500);

  // Open SSE source. The handler updates _globalJob state but DOES
  // NOT directly touch the read-view DOM — we leave the per-section
  // live preview to the existing handlers (which can hook into the
  // global stream by listening for the corresponding section).
  const source = new EventSource('/api/stream/' + jobId);
  _globalJobSource = source;
  source.onmessage = function(e) {{
    let evt;
    try {{ evt = JSON.parse(e.data); }} catch (err) {{ return; }}
    const j = _globalJob;
    if (!j || j.id !== jobId) return;

    if (evt.type === 'token') {{
      // Approximate token count by whitespace splits (Phase 15.4
      // pattern, but the count IS the count we display now).
      const text = (typeof evt.text === 'string') ? evt.text : '';
      const n = (text.match(/\\S+/g) || []).length || 1;
      j.tokens += n;
      const now = performance.now();
      for (let i = 0; i < n; i++) j.recentTokens.push(now);
    }} else if (evt.type === 'progress') {{
      if (evt.detail) j.taskDesc = evt.detail;
    }} else if (evt.type === 'model_info') {{
      if (evt.writer_model) j.modelName = evt.writer_model;
    }} else if (evt.type === 'length_target') {{
      if (evt.target_words) j.targetWords = evt.target_words;
    }} else if (evt.type === 'completed' || evt.type === 'all_sections_complete') {{
      j.state = 'done';
      j.taskDesc = 'Done';
      _renderTaskBar();
      _finishGlobalJob('done', 4000);
    }} else if (evt.type === 'error') {{
      j.state = 'error';
      j.taskDesc = 'Error: ' + (evt.message || 'unknown').slice(0, 80);
      _renderTaskBar();
      _finishGlobalJob('error', 0);  // wait for explicit dismiss
    }} else if (evt.type === 'cancelled') {{
      j.state = 'done';
      j.taskDesc = 'Stopped';
      _renderTaskBar();
      _finishGlobalJob('done', 2000);
    }} else if (evt.type === 'done') {{
      // Sentinel from the SSE side — close the source
      _finishGlobalJob('done', 1000);
    }}
  }};
  source.onerror = function() {{
    if (_globalJob) {{
      _globalJob.state = 'error';
      _globalJob.taskDesc = 'Connection lost';
      _renderTaskBar();
      _finishGlobalJob('error', 0);
    }}
  }};
}}

function _finishGlobalJob(state, autoDismissMs) {{
  if (!_globalJob) return;
  if (_globalJobSource) {{
    try {{ _globalJobSource.close(); }} catch (e) {{}}
    _globalJobSource = null;
  }}
  if (_globalJobTimer) {{
    clearInterval(_globalJobTimer);
    _globalJobTimer = null;
  }}
  _globalJob.state = state;
  _renderTaskBar();
  // Show the dismiss button instead of the stop button
  document.getElementById('tb-stop').style.display = 'none';
  document.getElementById('tb-dismiss').style.display = '';
  // Auto-dismiss after the grace period (0 = wait for user)
  if (autoDismissMs > 0) {{
    setTimeout(() => {{
      if (_globalJob && _globalJob.state !== 'streaming') dismissTaskBar();
    }}, autoDismissMs);
  }}
}}

function stopGlobalJob() {{
  if (!_globalJob) return;
  // Optimistic UI: change state immediately so the user sees feedback
  _globalJob.state = 'idle';
  _globalJob.taskDesc = 'Stopping…';
  _renderTaskBar();
  fetch('/api/jobs/' + _globalJob.id, {{method: 'DELETE'}}).catch(() => {{}});
  // The cancelled / done event from the server will trigger the
  // actual cleanup in onmessage. As a safety net, force-dismiss
  // after 5s in case the server never emits.
  const jobIdAtClick = _globalJob.id;
  setTimeout(() => {{
    if (_globalJob && _globalJob.id === jobIdAtClick && _globalJob.state !== 'streaming') {{
      _finishGlobalJob('done', 1500);
    }}
  }}, 5000);
}}

function dismissTaskBar() {{
  const bar = document.getElementById('task-bar');
  if (bar) bar.style.display = 'none';
  // Phase 32.3 — drop the body padding so the layout returns to
  // full height when the bar isn't visible.
  document.body.classList.remove('task-bar-open');
  _globalJob = null;
  if (_globalJobSource) {{
    try {{ _globalJobSource.close(); }} catch (e) {{}}
    _globalJobSource = null;
  }}
  if (_globalJobTimer) {{
    clearInterval(_globalJobTimer);
    _globalJobTimer = null;
  }}
}}

// ── Phase 30/31: Knowledge Graph browse modal (Graph + Table tabs) ────
async function openKgModal() {{
  openModal('kg-modal');
  switchKgTab('kg-graph');
  await loadKg(0);
}}

function switchKgTab(name) {{
  document.querySelectorAll('#kg-modal .tab').forEach(t => {{
    t.classList.toggle('active', t.dataset.tab === name);
  }});
  document.getElementById('kg-graph-pane').style.display = (name === 'kg-graph') ? 'block' : 'none';
  document.getElementById('kg-table-pane').style.display = (name === 'kg-table') ? 'block' : 'none';
}}

let _kgPredicatesLoaded = false;
let _kgTriples = [];

async function loadKg(offset) {{
  const subject = document.getElementById('kg-subject').value.trim();
  const predicate = document.getElementById('kg-predicate').value.trim();
  const obj = document.getElementById('kg-object').value.trim();
  const params = new URLSearchParams({{
    subject: subject, predicate: predicate, object: obj,
    limit: 200, offset: offset || 0,
  }});
  document.getElementById('kg-status').textContent = 'Loading…';
  try {{
    const res = await fetch('/api/kg?' + params.toString());
    const data = await res.json();
    if (!_kgPredicatesLoaded && data.predicates && data.predicates.length) {{
      const sel = document.getElementById('kg-predicate');
      data.predicates.forEach(p => {{
        const opt = document.createElement('option');
        opt.value = p; opt.textContent = p;
        sel.appendChild(opt);
      }});
      _kgPredicatesLoaded = true;
    }}
    _kgTriples = data.triples || [];
    document.getElementById('kg-status').textContent =
      data.total + ' triple' + (data.total === 1 ? '' : 's') + ' total · ' +
      'showing ' + _kgTriples.length + ' (top 100 in graph view)';

    // Render BOTH the table and the graph from the same data so
    // switching tabs is instant.
    _renderKgTable(_kgTriples);
    _renderKgGraph(_kgTriples.slice(0, 100));
  }} catch (e) {{
    document.getElementById('kg-status').textContent = 'Error: ' + e.message;
  }}
}}

function _renderKgTable(triples) {{
  if (triples.length === 0) {{
    document.getElementById('kg-results').innerHTML =
      '<div style="padding:24px;text-align:center;color:var(--fg-muted);font-size:12px;">No triples match your filter.</div>';
    return;
  }}
  let html = '<table class="kg-table"><thead><tr>';
  html += '<th>Subject</th><th>Predicate</th><th>Object</th><th>Source</th></tr></thead><tbody>';
  triples.forEach(t => {{
    html += '<tr>';
    html += '<td>' + escapeHtml(t.subject) + '</td>';
    html += '<td class="kg-pred">' + escapeHtml(t.predicate) + '</td>';
    html += '<td>' + escapeHtml(t.object) + '</td>';
    html += '<td class="kg-source" title="' + escapeHtml(t.source_title || '') + '">' +
            escapeHtml((t.source_title || '').substring(0, 60)) + '</td>';
    html += '</tr>';
  }});
  html += '</tbody></table>';
  document.getElementById('kg-results').innerHTML = html;
}}

// Phase 31 — render the KG as an actual force-directed SVG graph.
// Pure SVG + minimal JS (no D3 dep). Each unique entity becomes a
// node; each triple becomes a labeled edge. Layout is a tiny
// Fruchterman-Reingold-style spring simulation that runs for ~150
// iterations on load.
function _renderKgGraph(triples) {{
  const canvas = document.getElementById('kg-graph-canvas');
  if (!canvas) return;
  if (!triples || triples.length === 0) {{
    canvas.innerHTML = '<div style="padding:80px 24px;text-align:center;color:var(--fg-muted);font-size:12px;">No triples match your filter.</div>';
    return;
  }}
  const width = canvas.clientWidth || 800;
  const height = 520;

  // Build node + edge sets
  const nodeIndex = new Map();
  const nodes = [];
  function ensureNode(label) {{
    if (!nodeIndex.has(label)) {{
      nodeIndex.set(label, nodes.length);
      nodes.push({{
        id: nodes.length, label: label,
        x: Math.random() * width, y: Math.random() * height,
        vx: 0, vy: 0,
      }});
    }}
    return nodeIndex.get(label);
  }}
  const edges = triples.map(t => ({{
    source: ensureNode((t.subject || '').substring(0, 60)),
    target: ensureNode((t.object || '').substring(0, 60)),
    predicate: t.predicate,
  }}));

  // Truncate node label for display
  function nodeLabel(label) {{
    return label.length > 24 ? label.substring(0, 24) + '\\u2026' : label;
  }}

  // Tiny spring simulation: F-R-style with cooling
  const k = Math.sqrt((width * height) / Math.max(nodes.length, 1)) * 0.4;
  let temperature = width / 10;
  for (let iter = 0; iter < 150; iter++) {{
    // Repulsion
    for (let i = 0; i < nodes.length; i++) {{
      nodes[i].vx = 0; nodes[i].vy = 0;
      for (let j = 0; j < nodes.length; j++) {{
        if (i === j) continue;
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d2 = dx * dx + dy * dy + 0.01;
        const f = (k * k) / d2;
        nodes[i].vx += dx * f;
        nodes[i].vy += dy * f;
      }}
    }}
    // Attraction along edges
    edges.forEach(e => {{
      const a = nodes[e.source], b = nodes[e.target];
      const dx = a.x - b.x, dy = a.y - b.y;
      const d = Math.sqrt(dx * dx + dy * dy) + 0.01;
      const f = (d * d) / k;
      const fx = (dx / d) * f;
      const fy = (dy / d) * f;
      a.vx -= fx; a.vy -= fy;
      b.vx += fx; b.vy += fy;
    }});
    // Apply velocity capped by temperature
    nodes.forEach(n => {{
      const v = Math.sqrt(n.vx * n.vx + n.vy * n.vy);
      const lim = Math.min(v, temperature);
      n.x += (n.vx / (v || 1)) * lim;
      n.y += (n.vy / (v || 1)) * lim;
      // Stay inside the canvas
      n.x = Math.max(20, Math.min(width - 20, n.x));
      n.y = Math.max(20, Math.min(height - 20, n.y));
    }});
    temperature *= 0.97;
  }}

  // Render the SVG
  let svg = '<svg viewBox="0 0 ' + width + ' ' + height + '">';
  // Edges first so nodes render on top
  edges.forEach((e, i) => {{
    const a = nodes[e.source], b = nodes[e.target];
    svg += '<line class="kg-edge" x1="' + a.x.toFixed(1) + '" y1="' + a.y.toFixed(1) +
           '" x2="' + b.x.toFixed(1) + '" y2="' + b.y.toFixed(1) +
           '" data-edge-i="' + i + '"></line>';
    // Edge label at midpoint
    const mx = ((a.x + b.x) / 2).toFixed(1);
    const my = ((a.y + b.y) / 2).toFixed(1);
    svg += '<text class="kg-edge-label" x="' + mx + '" y="' + my + '" text-anchor="middle">' +
           escapeHtml(e.predicate) + '</text>';
  }});
  // Nodes
  nodes.forEach(n => {{
    svg += '<g class="kg-node" data-node-id="' + n.id + '" data-label="' + escapeHtml(n.label) + '" ' +
           'transform="translate(' + n.x.toFixed(1) + ',' + n.y.toFixed(1) + ')">';
    svg += '<circle r="6"></circle>';
    svg += '<text x="9" y="3">' + escapeHtml(nodeLabel(n.label)) + '</text>';
    svg += '</g>';
  }});
  svg += '</svg>';
  canvas.innerHTML = svg;

  // Click handler — clicking a node fills the subject filter and
  // re-runs loadKg, effectively zooming into that entity.
  canvas.querySelectorAll('.kg-node').forEach(g => {{
    g.addEventListener('click', () => {{
      const label = g.getAttribute('data-label');
      document.getElementById('kg-subject').value = label;
      switchKgTab('kg-table');
      loadKg(0);
    }});
  }});
}}

// ── Phase 30: Export modal ────────────────────────────────────────────
function openExportModal() {{
  openModal('export-modal');

  // Phase 31 — separate HTML and PDF buttons. PDF uses weasyprint
  // server-side; HTML is the printable browser-rendered version
  // (still useful as a fallback if weasyprint can't run).
  const exts = [
    {{ ext: 'pdf',  label: 'PDF' }},
    {{ ext: 'html', label: 'HTML' }},
    {{ ext: 'md',   label: 'Markdown' }},
    {{ ext: 'txt',  label: 'Text' }},
  ];
  function _btnHtml(base, enabled) {{
    return exts.map(e => enabled
      ? '<a href="' + base + '.' + e.ext + '" target="_blank">' + e.label + '</a>'
      : '<a class="disabled">' + e.label + '</a>'
    ).join('');
  }}

  // This section
  const sectionName = currentDraftId
    ? (document.getElementById('draft-title').textContent || 'current section')
    : null;
  document.getElementById('export-section-name').textContent =
    sectionName || '(no draft selected)';
  document.getElementById('export-section-btns').innerHTML =
    _btnHtml('/api/export/draft/' + currentDraftId, !!sectionName);

  // This chapter
  const ch = chaptersData.find(c => c.id === currentChapterId);
  const chName = ch ? ('Ch.' + ch.num + ': ' + ch.title) : null;
  document.getElementById('export-chapter-name').textContent =
    chName || '(no chapter selected)';
  document.getElementById('export-chapter-btns').innerHTML =
    _btnHtml('/api/export/chapter/' + currentChapterId, !!chName);

  // Whole book — always enabled
  document.getElementById('export-book-name').textContent =
    document.querySelector('.sidebar h2').textContent || 'Book';
  document.getElementById('export-book-btns').innerHTML =
    _btnHtml('/api/export/book', true);
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
    const safeTitle = escapeHtml(ch.title || '');
    html += '<div class="ch-group" data-ch-id="' + ch.id + '">';
    // Phase 23 — chevron toggle at the start of the chapter title.
    html += '<div class="ch-title clickable" onclick="selectChapter(this.parentElement)">' +
      '<button class="ch-toggle" ' +
      'onclick="event.stopPropagation();toggleChapter(this.closest(\\'.ch-group\\'))" ' +
      'title="Collapse or expand sections">\\u25be</button>' +
      'Ch.' + ch.num + ': ' + safeTitle +
      '<span class="ch-actions"><button onclick="event.stopPropagation();deleteChapter(\\\'' + ch.id + '\\\')" title="Delete chapter">\\u2717</button></span></div>';

    // Phase 21 — render the FULL section template, not just sections
    // with drafts. Empty slots become "Write" CTAs; orphan drafts
    // (whose section_type no longer matches a template slug) appear
    // at the end with a danger marker. This keeps the sidebar in
    // sync with whatever the chapter modal Section editor saves.
    const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];
    const titleBySlug = {{}};
    const planBySlug = {{}};
    meta.forEach(s => {{
      titleBySlug[s.slug] = s.title;
      planBySlug[s.slug] = s.plan || '';
    }});

    // Group existing drafts by slug
    const draftBySlug = {{}};
    ch.sections.forEach(sec => {{
      const slug = (sec.type || '').toLowerCase();
      if (!draftBySlug[slug] || (sec.version > (draftBySlug[slug].version || 1))) {{
        draftBySlug[slug] = sec;
      }}
    }});

    // Phase 22 — chapter completion progress bar (drafted / template).
    // Computed BEFORE rendering sections so it lives between the
    // chapter title and the section list, mirroring _render_sidebar.
    if (meta.length > 0) {{
      let nDrafted = 0;
      meta.forEach(t => {{ if (draftBySlug[t.slug]) nDrafted += 1; }});
      const nTotal = meta.length;
      const pct = Math.round(100 * nDrafted / nTotal);
      html += '<div class="ch-progress" title="' + nDrafted + ' of ' + nTotal + ' sections drafted">' +
        '<span class="ch-progress-bar"><span class="ch-progress-fill" style="width:' + pct + '%"></span></span>' +
        '<span class="ch-progress-label">' + nDrafted + '/' + nTotal + '</span>' +
        '</div>';
    }}

    // 1) Render template slots in declared order (drafted or empty).
    const seenSlugs = new Set();
    if (meta.length > 0) {{
      meta.forEach(tmpl => {{
        seenSlugs.add(tmpl.slug);
        const draft = draftBySlug[tmpl.slug];
        const planAttr = escapeHtml((tmpl.plan || '').replace(/\\n/g, ' ').slice(0, 200));
        const safeTmplTitle = escapeHtml(tmpl.title || '');
        // Phase 32.4 — inline delete button. The handler removes the
        // slug from sections_meta; if a draft exists, it becomes an
        // orphan in the sidebar (recoverable via the existing + adopt
        // button — fully reversible).
        const delBtn = '<button class="sec-delete-btn" ' +
          'onclick="event.preventDefault();event.stopPropagation();' +
          'deleteSection(\\'' + ch.id + '\\',\\'' + tmpl.slug + '\\')" ' +
          'title="Remove this section from the chapter (draft becomes an orphan)">\\u2717</button>';
        if (draft) {{
          const active = draft.id === activeId ? 'active' : '';
          // Phase 26 — draggable for reordering
          html += '<a class="sec-link ' + active + '" href="/section/' + draft.id +
            '" draggable="true" data-draft-id="' + draft.id + '" ' +
            'data-section-slug="' + tmpl.slug + '" title="' + planAttr + '" ' +
            'onclick="return navTo(this)">' +
            '<span class="sec-status-dot drafted"></span>' +
            safeTmplTitle +
            ' <span class="meta">v' + draft.version + ' \\u00b7 ' + draft.words + 'w</span>' +
            delBtn + '</a>';
        }} else {{
          // Phase 29 — preview-on-click instead of immediate doWrite()
          html += '<div class="sec-link sec-empty" draggable="true" ' +
            'data-section-slug="' + tmpl.slug + '" ' +
            'title="' + planAttr + '" ' +
            'onclick="previewEmptySection(\\'' + ch.id + '\\',\\'' + tmpl.slug + '\\')">' +
            '<span class="sec-status-dot empty"></span>' +
            safeTmplTitle +
            ' <span class="meta">empty \\u00b7 \\u270e</span>' +
            delBtn + '</div>';
        }}
      }});
    }}

    // 2) Render orphan drafts (drafts whose slug isn't in the template).
    Object.keys(draftBySlug).forEach(slug => {{
      if (seenSlugs.has(slug)) return;
      const draft = draftBySlug[slug];
      const display = escapeHtml(draft.title || (slug.charAt(0).toUpperCase() + slug.slice(1)));
      // Phase 22 — inline X button to delete the orphan
      // Phase 25 — also "+" button to adopt the slug into sections
      html += '<a class="sec-link sec-orphan" href="/section/' + draft.id +
        '" data-draft-id="' + draft.id + '" onclick="return navTo(this)" ' +
        'title="Orphan draft. Click to inspect, + to adopt into sections, X to delete.">' +
        '<span class="sec-status-dot orphan"></span>' +
        display +
        ' <span class="meta">orphan \\u00b7 v' + draft.version + ' \\u00b7 ' + draft.words + 'w</span>' +
        '<button class="sec-orphan-adopt" ' +
        'onclick="event.preventDefault();event.stopPropagation();adoptOrphanSection(\\'' + ch.id + '\\',\\'' + slug + '\\')" ' +
        'title="Add this section_type to the chapter sections list">+</button>' +
        '<button class="sec-orphan-delete" ' +
        'onclick="event.preventDefault();event.stopPropagation();deleteOrphanDraft(\\'' + draft.id + '\\')" ' +
        'title="Delete this orphan draft permanently">\\u2717</button>' +
        '</a>';
    }});

    if (meta.length === 0 && Object.keys(draftBySlug).length === 0) {{
      html += '<div class="sec-link sec-empty-cta" onclick="startWritingChapter(\\'' + ch.id + '\\')">\\u270e Start writing</div>';
    }}
    // Phase 32.4 — "+ Add section" CTA at the bottom of every
    // chapter's section list. Click → prompt for a title → POST
    // a new section dict via PUT /api/chapters/{{id}}/sections.
    html += '<div class="sec-link sec-add-cta" ' +
      'onclick="addSectionToChapter(\\'' + ch.id + '\\')" ' +
      'title="Add a new section to this chapter">+ Add section</div>';
    html += '</div>';
  }});
  container.innerHTML = html;
  // Phase 23 — re-apply collapsed state after rebuilding the DOM.
  restoreCollapsedChapters();
}}

// ── Action handlers ───────────────────────────────────────────────────────
async function doWrite() {{
  if (!currentChapterId) {{
    showEmptyHint('Select a chapter from the sidebar first &mdash; click any chapter title in the left panel, then come back and click Write.');
    return;
  }}
  const section = currentSectionType || 'introduction';
  showStreamPanel('Writing ' + section + '...');
  // Phase 15.6 — clear the read-view and prepare it for live writing
  startLiveWrite();

  const fd = new FormData();
  fd.append('chapter_id', currentChapterId);
  fd.append('section_type', section);
  const res = await fetch('/api/write', {{method: 'POST', body: fd}});
  const data = await res.json();
  startStream(data.job_id);
  // Phase 30 — persistent task bar
  startGlobalJob(data.job_id, {{
    type: 'write',
    taskDesc: 'Writing ' + section,
    modelName: 'qwen3.5:27b',
    sectionType: section,
    chapterId: currentChapterId,
  }});
}}

async function doReview() {{
  if (!currentDraftId) {{ showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft."); return; }}
  showStreamPanel('Reviewing...');

  const fd = new FormData();
  const res = await fetch('/api/review/' + currentDraftId, {{method: 'POST', body: fd}});
  const data = await res.json();
  // Phase 30 — persistent task bar
  startGlobalJob(data.job_id, {{
    type: 'review',
    taskDesc: 'Reviewing draft',
    modelName: 'qwen3.5:27b',
  }});
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
  html += '<p style="font-size:11px;color:var(--fg-muted);margin-bottom:6px;">Click a chapter title to edit its scope. Click an empty cell to preview that section. Click a filled cell to open the draft. Hover any cell to see the section title.</p>';
  // Phase 30 — columns are POSITIONAL (1, 2, 3, ...) up to max(N) across
  // all chapters. Each chapter shows its actual sections in order;
  // chapters with fewer sections get blank "absent" cells in the extra
  // slots so the table is rectangular.
  const nCols = data.n_columns || 1;
  html += '<table class="heatmap"><thead><tr><th></th>';
  for (let i = 1; i <= nCols; i++) {{
    html += '<th title="Section position ' + i + '">' + i + '</th>';
  }}
  html += '</tr></thead><tbody>';
  data.heatmap.forEach(row => {{
    html += '<tr><td class="ch-label clickable" onclick="openChapterModal(&#39;' + row.id + '&#39;)" title="Click to edit chapter title and scope">';
    html += '<span class="ch-label-num">Ch.' + row.num + '</span> ' + escapeHtml((row.title || '').substring(0, 36));
    html += ' <span class="ch-label-edit">&#9881;</span></td>';
    row.cells.forEach((cell, idx) => {{
      const posLabel = (idx + 1) + '. ' + (cell.title || '');
      if (cell.status === 'absent') {{
        // This chapter has fewer sections than max(N) — render blank
        html += '<td><span class="hm-cell absent" title="(no section ' + (idx + 1) + ' in this chapter)">·</span></td>';
      }} else if (cell.status === 'empty') {{
        // Empty template slot — Phase 29 click-to-preview
        html += '<td><span class="hm-cell empty" ' +
          'onclick="previewEmptySection(&#39;' + row.id + '&#39;,&#39;' + cell.type + '&#39;)" ' +
          'title="' + escapeHtml(posLabel) + ' (empty — click to preview)">+</span></td>';
      }} else {{
        const label = 'v' + cell.version + ' ' + cell.words + 'w';
        html += '<td><span class="hm-cell ' + cell.status + '" ' +
          'onclick="loadSection(&#39;' + cell.draft_id + '&#39;)" ' +
          'title="' + escapeHtml(posLabel) + ' &mdash; ' + label + '">' + label + '</span></td>';
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

// Phase 29 — clicking an empty section row in the sidebar now SHOWS
// a preview placeholder in the read-view instead of immediately
// triggering doWrite(). The user can:
//   - read the section title + plan + target words
//   - click "Start writing" to fire doWrite (single section)
//   - click "Autowrite" to fire doAutowrite (with iterations)
//   - click another section in the sidebar to navigate away
// All without an LLM call happening accidentally on a single click.
function previewEmptySection(chapterId, sectionType) {{
  currentChapterId = chapterId;
  currentSectionType = sectionType;

  // Look up the section meta from the in-memory chapters cache
  const ch = chaptersData.find(c => c.id === chapterId);
  if (!ch) {{
    showEmptyHint('Chapter not found in cache.');
    return;
  }}
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];
  const sec = meta.find(s => s.slug === sectionType);
  const sectionTitle = sec ? sec.title : (sectionType.charAt(0).toUpperCase() + sectionType.slice(1));
  const sectionPlan = sec ? (sec.plan || '') : '';
  const sectionTarget = sec && sec.target_words && sec.target_words > 0
    ? sec.target_words
    : (window._chapterWordTarget && Math.floor(window._chapterWordTarget / Math.max(1, meta.length)));

  // Switch to read-view (hide dashboard if it's showing)
  document.getElementById('dashboard-view').style.display = 'none';
  document.getElementById('read-view').style.display = 'block';
  document.getElementById('draft-subtitle').style.display = 'block';
  document.getElementById('toolbar').style.display = 'flex';
  document.getElementById('edit-view').style.display = 'none';

  // Update the title bar to reflect the section
  document.getElementById('draft-title').textContent =
    'Ch.' + ch.num + ': ' + ch.title + ' \\u2014 ' + sectionTitle;
  document.getElementById('draft-version').textContent = '0';
  document.getElementById('draft-words').textContent = '0';
  updateWordTargetBar(0, sectionTarget);

  // Clear any active state on other section rows + highlight the
  // empty row we just clicked.
  document.querySelectorAll('.sec-link').forEach(l => l.classList.remove('active'));
  const link = document.querySelector(
    '.sec-link.sec-empty[data-section-slug="' + sectionType + '"]'
  );
  if (link) link.classList.add('active');

  // Render the preview content into the read-view
  const planHtml = sectionPlan
    ? '<div style="margin:16px 0;padding:12px 16px;background:var(--toolbar-bg);' +
      'border-left:3px solid var(--accent);border-radius:4px;">' +
      '<div style="font-size:11px;color:var(--fg-muted);text-transform:uppercase;' +
      'letter-spacing:0.04em;margin-bottom:6px;">Section plan</div>' +
      '<div style="font-family:var(--font-serif);font-size:14px;line-height:1.6;' +
      'white-space:pre-wrap;">' + escapeHtml(sectionPlan) + '</div></div>'
    : '<div style="margin:16px 0;font-size:13px;color:var(--fg-muted);font-style:italic;">' +
      'No section plan set yet. Open the chapter modal (\\u2699 icon) and add one in the Sections tab.</div>';

  const html =
    '<div class="empty-section-preview">' +
    '<div style="font-size:12px;color:var(--fg-muted);margin-bottom:8px;">' +
    '<span class="sec-status-dot empty"></span> empty section &middot; ' +
    'target ~' + (sectionTarget || 'auto') + ' words' +
    '</div>' +
    planHtml +
    '<div style="margin-top:24px;display:flex;gap:8px;flex-wrap:wrap;">' +
    '<button class="btn-primary" onclick="doWrite()">\\u270e Start writing</button>' +
    '<button class="btn-secondary" onclick="doAutowrite()">\\u26a1 Autowrite (with iterations)</button>' +
    '</div>' +
    '<p style="margin-top:24px;font-size:12px;color:var(--fg-muted);">' +
    'This section has no draft yet. Click <strong>Start writing</strong> for a single ' +
    'pass, or <strong>Autowrite</strong> to run the score &rarr; verify &rarr; revise loop.' +
    ' You can also reorder this slot in the sidebar by dragging it.' +
    '</p>' +
    '</div>';
  document.getElementById('read-view').innerHTML = html;

  // Update URL so refresh keeps the preview state
  history.pushState({{previewSection: sectionType, chapterId: chapterId}},
    '', '/');
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

// Phase 32.4 — delete a single section from a chapter's sections_meta.
// The associated draft (if any) becomes an orphan in the sidebar — it
// is NOT hard-deleted, so the user can adopt it back via the existing
// + button or hard-delete it via the X. This mirrors the chapter
// delete UX (drafts preserved, just unlinked).
async function deleteSection(chapterId, slug) {{
  if (!chapterId || !slug) return;
  const ch = chaptersData.find(c => c.id === chapterId);
  if (!ch) return;
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];
  const sec = meta.find(s => s.slug === slug);
  const title = (sec && sec.title) ? sec.title : slug;
  // Detect whether a draft exists for this slug — affects the warning text.
  const drafted = (ch.sections || []).some(d => (d.type || '').toLowerCase() === slug.toLowerCase());
  const msg = drafted
    ? 'Remove section "' + title + '" from this chapter?\\n\\nThe existing draft will become an orphan (still listed in the sidebar with a + to re-add and X to permanently delete).'
    : 'Remove section "' + title + '" from this chapter?';
  if (!confirm(msg)) return;
  const updated = meta.filter(s => s.slug !== slug).map(s => ({{
    slug: s.slug,
    title: s.title || s.slug,
    plan: s.plan || '',
    target_words: (s.target_words && s.target_words > 0) ? s.target_words : null,
  }}));
  try {{
    const res = await fetch('/api/chapters/' + chapterId + '/sections', {{
      method: 'PUT',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{sections: updated}}),
    }});
    if (!res.ok) throw new Error('save failed (' + res.status + ')');
    // If the user was viewing the draft we just orphaned, fall back
    // to the dashboard so the center frame doesn't show stale content.
    if (currentDraftId && drafted) {{
      const activeIsThis = (ch.sections || []).some(d =>
        d.id === currentDraftId && (d.type || '').toLowerCase() === slug.toLowerCase()
      );
      if (activeIsThis) {{
        currentDraftId = '';
        showDashboard();
      }}
    }}
    // Refresh the sidebar so the slot disappears and any orphaned
    // draft surfaces in the orphan list.
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
    // Update local cache so subsequent operations see the new list
    if (Array.isArray(sd.chapters)) chaptersData = sd.chapters;
  }} catch (e) {{
    alert('Failed to remove section: ' + e.message);
  }}
}}

// Phase 32.4 — inline add-section flow from the sidebar. Prompts for
// a title, derives a slug client-side (matches core.book_ops's
// _slugify_section_name), appends a new section dict to the chapter's
// sections_meta, and PUTs the new list. The chapter modal Sections
// tab still works for richer edits (plan, target_words, reorder).
async function addSectionToChapter(chapterId) {{
  if (!chapterId) return;
  const ch = chaptersData.find(c => c.id === chapterId);
  if (!ch) return;
  const title = prompt('New section title (e.g. "The 11-Year Solar Cycle"):');
  if (!title || !title.trim()) return;
  const cleanTitle = title.trim();
  const slug = cleanTitle.toLowerCase().replace(/\\s+/g, '_').replace(/[^a-z0-9_]/g, '');
  if (!slug) {{
    alert('Could not derive a slug from that title. Try plain letters.');
    return;
  }}
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];
  if (meta.some(s => s.slug === slug)) {{
    alert('A section with slug "' + slug + '" already exists in this chapter.');
    return;
  }}
  const updated = meta.map(s => ({{
    slug: s.slug,
    title: s.title || s.slug,
    plan: s.plan || '',
    target_words: (s.target_words && s.target_words > 0) ? s.target_words : null,
  }}));
  updated.push({{slug: slug, title: cleanTitle, plan: '', target_words: null}});
  try {{
    const res = await fetch('/api/chapters/' + chapterId + '/sections', {{
      method: 'PUT',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{sections: updated}}),
    }});
    if (!res.ok) throw new Error('save failed (' + res.status + ')');
    // Refresh sidebar so the new slot appears
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
    if (Array.isArray(sd.chapters)) chaptersData = sd.chapters;
  }} catch (e) {{
    alert('Failed to add section: ' + e.message);
  }}
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

  // Phase 20 — count broken citations so we can warn the user once.
  // A broken citation is one whose ref number > number of sources;
  // it's typically the result of a pre-Phase-18 draft (where the
  // writer's prompt and the saved sources had inconsistent numbering)
  // or the writer hallucinating a citation number.
  let brokenCount = 0;
  let brokenRefs = new Set();

  document.querySelectorAll('.citation').forEach(el => {{
    const ref = el.dataset.ref;
    if (!ref) return;
    // Always (re)attach click handler — buildPopovers may run multiple
    // times on the same DOM (after section nav, after autowrite finish),
    // and the source map could have changed in between.
    const src = sourceData[parseInt(ref)];

    if (!src) {{
      // Phase 20 — broken citation: visually mark and short-circuit.
      // Don't attach a click handler that scrolls to a non-existent
      // anchor — instead, on click show a tooltip explaining the
      // citation has no matching source.
      el.classList.add('citation-broken');
      el.title = 'Citation [' + ref + '] has no matching source. ' +
                 'This usually means the draft predates the Phase 18 ' +
                 'fix — re-run autowrite to regenerate it.';
      el.onclick = function(e) {{
        e.preventDefault();
        e.stopPropagation();
      }};
      brokenCount += 1;
      brokenRefs.add(ref);
      return;
    }}

    // Healthy citation: build popover + scroll-to-source click handler.
    if (!el.querySelector('.citation-popover')) {{
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
    }}
    el.classList.remove('citation-broken');
    el.style.cursor = 'pointer';
    el.onclick = function() {{
      const target = document.getElementById('source-' + ref);
      if (target) target.scrollIntoView({{behavior: 'smooth', block: 'center'}});
    }};
  }});

  // Phase 20 — surface broken citations once per page so the user knows
  // the dead links aren't a UI bug but a data problem they can fix by
  // re-running autowrite on the affected draft.
  if (brokenCount > 0) {{
    console.warn('[sciknow] ' + brokenCount + ' broken citation(s) ' +
                 'in this draft (orphan refs: ' + Array.from(brokenRefs).sort().join(', ') +
                 '). The draft cites source numbers that aren\\'t in the sources panel — ' +
                 'usually a pre-Phase-18 draft. Re-run autowrite to regenerate.');
  }}
}}

// Build popovers on page load
document.addEventListener('DOMContentLoaded', buildPopovers);

// ── Autowrite Dashboard (Phase 4) ────────────────────────────────────
// Enhanced autowrite that shows convergence chart
let awScores = [];
let awTargetScore = 0.85;

async function doAutowrite() {{
  if (!currentChapterId) {{ showEmptyHint("No chapter selected &mdash; click any chapter title in the sidebar to select it, then try again."); return; }}

  // Phase 20 — when no specific section is selected (the user has a
  // chapter highlighted in the sidebar but hasn't picked a section),
  // run autowrite for ALL of the chapter's defined sections instead
  // of defaulting to a single 'introduction' draft (which doesn't
  // match any user-defined section and creates an orphan).
  const isAllSections = !currentSectionType;
  const section = currentSectionType || '__all__';
  const maxIter = prompt('Max iterations per section (default 3):', '3');
  if (maxIter === null) return;
  const targetStr = prompt('Target score (default 0.85):', '0.85');
  if (targetStr === null) return;
  awTargetScore = parseFloat(targetStr) || 0.85;
  awScores = [];

  // Phase 28 — three-way handling for sections that already have a draft.
  // Default = skip; alternatives = rebuild (overwrite from scratch) or
  // resume (load existing content + run more iterations).
  let modeRebuild = false;
  let modeResume = false;
  if (isAllSections) {{
    const ch = chaptersData.find(c => c.id === currentChapterId);
    const nSecs = (ch && ch.sections_meta && ch.sections_meta.length) || 5;
    // Count sections that already have a draft so the prompt is meaningful
    let nDrafted = 0;
    if (ch && Array.isArray(ch.sections)) {{
      const draftedSlugs = new Set(ch.sections.filter(s => s.id).map(s => (s.type || '').toLowerCase()));
      nDrafted = (ch.sections_meta || []).filter(m => draftedSlugs.has(m.slug)).length;
    }}
    let modeStr;
    if (nDrafted === 0) {{
      // No existing drafts — just confirm the long run.
      if (!confirm('Autowrite all ' + nSecs + ' sections of this chapter? This can take ' + (nSecs * 5) + '-' + (nSecs * 10) + ' minutes.')) return;
    }} else {{
      modeStr = prompt(
        nDrafted + ' of ' + nSecs + ' sections already have a draft.\\n\\n' +
        'Pick how to handle them:\\n' +
        '  s = skip   (default — only fill the missing sections)\\n' +
        '  r = rebuild (overwrite drafted sections from scratch)\\n' +
        '  i = iterate (resume from existing content, run more iterations)\\n\\n' +
        'Type s, r, or i:',
        's'
      );
      if (modeStr === null) return;
      modeStr = (modeStr || 's').trim().toLowerCase();
      if (modeStr === 'r' || modeStr === 'rebuild') modeRebuild = true;
      else if (modeStr === 'i' || modeStr === 'iterate' || modeStr === 'resume') modeResume = true;
      // else: default skip behaviour
    }}
  }}

  showStreamPanel(isAllSections
    ? 'Autowriting all sections...'
    : 'Autowriting ' + section + '...');
  // Phase 15.6 — clear the read-view and prepare it for live writing
  startLiveWrite();

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
  if (!isAllSections) fd.append('section_type', section);
  fd.append('max_iter', maxIter || '3');
  fd.append('target_score', String(awTargetScore));
  // Phase 28 — pass the user's chosen mode (only for chapter-wide runs;
  // single-section autowrite always rewrites the targeted section)
  if (isAllSections && modeRebuild) fd.append('rebuild', 'true');
  if (isAllSections && modeResume) fd.append('resume', 'true');
  const endpoint = isAllSections ? '/api/autowrite-chapter' : '/api/autowrite';
  const res = await fetch(endpoint, {{method: 'POST', body: fd}});
  const data = await res.json();

  // Phase 30 — start the persistent task bar so the live state
  // survives navigation. The bar OWNS its own SSE source; the
  // existing per-section source.onmessage handler below still
  // runs for live preview / dashboard chart, but it doesn't
  // own the lifecycle anymore.
  startGlobalJob(data.job_id, {{
    type: 'autowrite',
    taskDesc: isAllSections
      ? 'Autowriting all sections of Ch.' + (chaptersData.find(c => c.id === currentChapterId) || {{}}).num
      : 'Autowriting ' + section,
    modelName: 'qwen3.5:27b',
    sectionType: section,
    chapterId: currentChapterId,
  }});

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
      // to the visible draft area + main read-view live preview;
      // scoring/verify/CoVe/planning JSON tokens only feed the stats
      // counter (they'd be ugly to show in the draft pane).
      const phase = evt.phase || 'writing';
      stats.update(evt.text);
      stats.setPhase(phase);
      if (phase === 'writing' || phase === 'revising') {{
        // Existing autowrite-dashboard token area (now redundant with the
        // read-view live preview, but kept for users who want to focus on
        // the dashboard).
        setStreamCursor(awContent, false);
        awContent.innerHTML += evt.text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        setStreamCursor(awContent, true);
        awContent.scrollTop = awContent.scrollHeight;
        // Phase 15.6 — also stream into the main read-view as a live
        // markdown preview, so the user sees the writing happen in the
        // same place they'd read the final draft.
        appendLiveWrite(evt.text);
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
      // Phase 15.6 — clear the live preview so each new iteration shows
      // its own writing fresh in the read-view.
      clearLiveWrite();
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
      // Phase 20 — for multi-section runs, this fires once per section.
      // We refresh after each section so the user sees the new draft
      // appear in the sidebar, but DON'T close the stream — there are
      // more sections to write. The all_sections_complete event below
      // closes the stream when truly done.
      if (isAllSections) {{
        // Refresh sidebar so the new section's draft becomes clickable.
        // Don't navigate away — the user is watching the live preview.
        fetch('/api/chapters').then(r => r.json()).then(d => {{
          rebuildSidebar(d.chapters || d, currentDraftId);
        }}).catch(() => {{}});
      }} else {{
        status.textContent = 'Done';
        stats.done('done');
        setStreamCursor(awContent, false);
        hideStreamPanel();
        source.close(); currentEventSource = null; currentJobId = null;
        refreshAfterJob(evt.draft_id);
      }}
    }}
    // Phase 20 — multi-section autowrite envelope events
    else if (evt.type === 'chapter_autowrite_start') {{
      awLog.innerHTML += '<div style="font-weight:bold;border-top:1px solid var(--border);padding-top:6px;margin-top:6px;">' +
        '\\u270e Chapter autowrite: ' + evt.n_sections + ' sections' +
        (evt.rebuild ? ' (rebuild mode)' : '') + '</div>';
    }}
    else if (evt.type === 'section_start') {{
      const skip = evt.skipped ? ' [SKIPPED — already drafted]' : '';
      awLog.innerHTML += '<div style="font-weight:600;color:var(--accent);border-top:1px solid var(--border);padding-top:6px;margin-top:6px;">' +
        '\\u25b6 Section ' + evt.index + '/' + evt.total + ': ' + evt.title + skip + '</div>';
      status.textContent = 'Section ' + evt.index + '/' + evt.total + ': ' + evt.title +
        (evt.skipped ? ' (skipping)' : '');
      if (!evt.skipped) {{
        // New section starting — clear the previous section's preview.
        clearLiveWrite();
        if (awContent) awContent.innerHTML = '';
      }}
    }}
    else if (evt.type === 'section_done') {{
      const score = (evt.final_score != null) ? ' (' + evt.final_score.toFixed(2) + ')' : '';
      const cls = evt.error ? 'log-discard' : 'log-keep';
      const icon = evt.error ? '\\u2717' : evt.skipped ? '\\u2014' : '\\u2713';
      const note = evt.error ? ' error: ' + evt.error : evt.skipped ? ' skipped' : ' done' + score;
      awLog.innerHTML += '<div class="' + cls + '">' + icon + ' Section ' + evt.index + note + '</div>';
    }}
    else if (evt.type === 'section_error') {{
      awLog.innerHTML += '<div class="log-discard">\\u2717 Section ' + evt.index + ' failed: ' + evt.message + '</div>';
    }}
    else if (evt.type === 'all_sections_complete') {{
      status.textContent = 'Chapter done: ' + evt.n_completed + '/' + evt.n_total +
        ' written, ' + evt.n_skipped + ' skipped' +
        (evt.n_failed > 0 ? ', ' + evt.n_failed + ' failed' : '');
      awLog.innerHTML += '<div class="log-keep" style="font-weight:bold;border-top:1px solid var(--border);padding-top:6px;margin-top:6px;">' +
        '\\u2713 Chapter complete: ' + evt.n_completed + ' written, ' +
        evt.n_skipped + ' skipped, ' + evt.n_failed + ' failed</div>';
      stats.done('done');
      setStreamCursor(awContent, false);
      // Don't auto-hide; let the user read the summary. Close the SSE
      // source so we don't leak the connection.
      source.close(); currentEventSource = null; currentJobId = null;
      // Refresh sidebar so all new section drafts are visible
      fetch('/api/chapters').then(r => r.json()).then(d => {{
        rebuildSidebar(d.chapters || d, currentDraftId);
      }}).catch(() => {{}});
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

// ── Phase 15.6: live writing preview in the main read-view ──────────
//
// As writing/revising tokens stream in, accumulate them and re-render
// the read-view as live HTML — same Georgia serif body the final draft
// uses, with paragraph breaks and [N] citation styling. On completion,
// refreshAfterJob() reloads the saved draft from the API and the live
// preview gets replaced with the proper rendered version.

let _liveWriteText = '';
let _liveWriteActive = false;

function _escapeHtml(s) {{
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}}

function _renderLiveMarkdown(text) {{
  // Cheap, fast HTML rendering of streaming markdown:
  //   - Escape HTML first
  //   - Style [N] citations like the final view
  //   - Convert \\n\\n into paragraph boundaries
  //   - Convert single \\n inside a paragraph into <br>
  // This won't handle every markdown feature (no headings, no bold, no
  // lists) but it's enough to make the streaming text look like prose
  // instead of console output.
  const escaped = _escapeHtml(text);
  const withCitations = escaped.replace(/\\[(\\d+)\\]/g,
    '<span class="citation" data-ref="$1">[$1]</span>');
  const paragraphs = withCitations.split(/\\n\\n+/);
  return paragraphs.map(p => '<p data-live="1">' + p.replace(/\\n/g, '<br>') + '</p>').join('');
}}

function clearLiveWrite() {{
  _liveWriteText = '';
  _liveWriteActive = false;
}}

function startLiveWrite() {{
  _liveWriteText = '';
  _liveWriteActive = true;
  const rv = document.getElementById('read-view');
  if (rv) {{
    rv.innerHTML =
      '<div class="live-writing-banner">&#9998; Writing live &mdash; this preview will be replaced with the saved draft when generation completes.</div>' +
      '<div class="live-writing-body" id="live-writing-body"></div>';
  }}
}}

function appendLiveWrite(text) {{
  if (!_liveWriteActive) startLiveWrite();
  _liveWriteText += (text || '');
  const body = document.getElementById('live-writing-body');
  if (body) {{
    body.innerHTML = _renderLiveMarkdown(_liveWriteText);
    // Add a blinking cursor to the very last paragraph
    const lastP = body.querySelector('p:last-child');
    if (lastP) setStreamCursor(lastP, true);
    // Auto-scroll the read-view to follow new tokens
    const rv = document.getElementById('read-view');
    if (rv) rv.scrollTop = rv.scrollHeight;
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

// ── Phase 14.3 + Phase 21: Plans modal (book / chapter / section) ────
//
// Phase 21 — context-aware. The Plan toolbar button auto-detects what's
// selected and routes to the right tab:
//   - section selected → Section tab focused on that section
//   - chapter selected → Chapter tab showing the chapter's section plans
//   - nothing selected → Book tab (the leitmotiv)
//
// All three tabs are always present in the DOM; openPlanModal hides
// the chapter/section tabs when there's no relevant context, and
// switchPlanTab is the manual override.

let _planContext = {{ mode: 'book', chapterId: null, sectionSlug: null }};

async function openPlanModal(context) {{
  // Default context: derive from current selection state.
  if (!context) {{
    if (currentChapterId && currentSectionType) {{
      context = {{ mode: 'section', chapterId: currentChapterId, sectionSlug: currentSectionType }};
    }} else if (currentChapterId) {{
      context = {{ mode: 'chapter', chapterId: currentChapterId, sectionSlug: null }};
    }} else {{
      context = {{ mode: 'book', chapterId: null, sectionSlug: null }};
    }}
  }}
  _planContext = context;

  // Phase 32.2 — reset per-chapter editing state every time the modal
  // opens so slug collisions between chapters (e.g. "introduction"
  // appearing in every chapter) don't leak overrides from one chapter
  // into another.
  _editingChapterPlans = {{}};
  _editingChapterTargetWords = {{}};
  _editingChapterCustomMode = {{}};

  openModal('plan-modal');
  document.getElementById('plan-status').textContent = 'Loading...';

  // Show/hide context-specific tabs based on what's selected
  const chapterTab = document.getElementById('plan-tab-chapter');
  const sectionTab = document.getElementById('plan-tab-section');
  if (chapterTab) chapterTab.style.display = context.chapterId ? '' : 'none';
  if (sectionTab) sectionTab.style.display = (context.chapterId && context.sectionSlug) ? '' : 'none';

  // Always populate the Book tab fields (cheap fetch + the user might
  // switch tabs back to it).
  try {{
    const res = await fetch('/api/book');
    const data = await res.json();
    document.getElementById('plan-title-input').value = data.title || '';
    document.getElementById('plan-desc-input').value = data.description || '';
    document.getElementById('plan-text-input').value = data.plan || '';
    const tcw = data.target_chapter_words;
    const dflt = data.default_target_chapter_words || 6000;
    window._chapterWordTarget = tcw || dflt;
    const tcwInput = document.getElementById('plan-target-words-input');
    const lstatus = document.getElementById('plan-length-status');
    if (tcwInput) tcwInput.value = tcw ? String(tcw) : '';
    if (lstatus) {{
      lstatus.textContent = tcw
        ? ('current: ' + tcw + ' words/chapter')
        : ('using default: ' + dflt + ' words/chapter');
    }}
    if (!data.plan) {{
      document.getElementById('plan-status').innerHTML =
        '<span style="color:var(--warning);">No book plan set yet.</span> Click <strong>Regenerate with LLM</strong> to draft one.';
    }} else {{
      document.getElementById('plan-status').textContent =
        data.plan.split(/\\s+/).filter(Boolean).length + ' words in book plan';
    }}
  }} catch (e) {{
    document.getElementById('plan-status').textContent = 'Error loading book: ' + e.message;
  }}

  // Populate the Chapter / Section tabs from the in-memory chaptersData
  // (no extra fetch needed). Falls back to refetching /api/chapters
  // if the local cache is missing the active chapter.
  if (context.chapterId) {{
    const ch = chaptersData.find(c => c.id === context.chapterId);
    if (ch) populatePlanChapterTab(ch);
  }}

  // Open the right tab based on context
  if (context.mode === 'section') {{
    switchPlanTab('plan-section');
    populatePlanSectionTab(context.chapterId, context.sectionSlug);
  }} else if (context.mode === 'chapter') {{
    switchPlanTab('plan-chapter');
  }} else {{
    switchPlanTab('plan-book');
  }}
}}

function switchPlanTab(name) {{
  document.querySelectorAll('#plan-modal .tab').forEach(t => {{
    t.classList.toggle('active', t.dataset.tab === name);
  }});
  document.getElementById('plan-book-pane').style.display = (name === 'plan-book') ? 'block' : 'none';
  document.getElementById('plan-chapter-pane').style.display = (name === 'plan-chapter') ? 'block' : 'none';
  document.getElementById('plan-section-pane').style.display = (name === 'plan-section') ? 'block' : 'none';
  // The Regenerate button only makes sense for the book leitmotiv.
  const regenBtn = document.getElementById('plan-regen-btn');
  if (regenBtn) regenBtn.style.display = (name === 'plan-book') ? '' : 'none';
}}

// Phase 21 — render the chapter sections view inside the Plan modal.
// Each section gets a title input + plan textarea, just like the
// chapter modal's Sections tab — but read-only of the title (you
// rename via the chapter modal), editable for the plan.
// Phase 32.2 — every section now also gets a per-section length
// dropdown so the user can pick "Long" / "Custom" / etc directly
// inside the Plan modal without having to switch over to the
// chapter modal Sections tab. Save round-trips through the same
// PUT /api/chapters/{{id}}/sections endpoint, which already accepts
// target_words as part of each section dict.
function populatePlanChapterTab(ch) {{
  const header = document.getElementById('plan-chapter-header');
  if (header) {{
    header.innerHTML = '<strong>Ch.' + ch.num + ': ' + escapeHtml(ch.title) + '</strong>' +
      (ch.description ? '<br><span style="font-size:12px;">' + escapeHtml(ch.description.substring(0, 200)) + (ch.description.length > 200 ? '\\u2026' : '') + '</span>' : '');
  }}
  const list = document.getElementById('plan-chapter-sections');
  if (!list) return;
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];
  if (meta.length === 0) {{
    list.innerHTML = '<div style="font-size:12px;color:var(--fg-muted);padding:12px;text-align:center;border:1px dashed var(--border);border-radius:4px;">This chapter has no sections defined. Open the chapter modal (\\u2699 icon) and use the Sections tab to add some.</div>';
    return;
  }}

  // Compute the per-section auto budget so the dropdown's "Auto"
  // option can show what the writer would aim for if no override
  // is set. Mirrors the logic in renderSectionEditor.
  const chapterTarget = (window._chapterWordTarget && window._chapterWordTarget > 0)
    ? window._chapterWordTarget
    : 6000;
  const nSec = Math.max(1, meta.length);
  const perSection = Math.max(400, Math.min(chapterTarget, Math.floor(chapterTarget / nSec)));

  // Seed the editing state with current target_words from meta so
  // the dropdown reflects whatever was previously saved.
  meta.forEach(s => {{
    if (!(s.slug in _editingChapterTargetWords)) {{
      _editingChapterTargetWords[s.slug] = (s.target_words && s.target_words > 0)
        ? s.target_words : null;
    }}
  }});

  let html = '<div style="font-size:11px;color:var(--fg-muted);margin-bottom:8px;padding:6px 10px;background:var(--toolbar-bg);border-radius:4px;">';
  html += '<strong>' + meta.length + '</strong> section' + (meta.length === 1 ? '' : 's') +
          ' &middot; chapter target: <strong>' + chapterTarget + '</strong> words &middot; ' +
          'auto per section: <strong>~' + perSection + '</strong> words';
  html += '</div>';

  meta.forEach((s, i) => {{
    const tw = _editingChapterTargetWords[s.slug];
    const isAuto = !tw || tw <= 0;
    const presets = [400, 800, 1500, 3000, 6000];
    const isCustomMode = !!_editingChapterCustomMode[s.slug];
    const presetMatch = !isAuto && presets.includes(tw);
    const isCustom = isCustomMode || (!isAuto && !presetMatch);
    let optsHtml = '<option value="">Auto (~' + perSection + 'w)</option>';
    presets.forEach(p => {{
      const sel = (!isCustom && tw === p) ? ' selected' : '';
      const labelMap = {{400: 'Very short', 800: 'Short', 1500: 'Medium', 3000: 'Long', 6000: 'Extra long'}};
      optsHtml += '<option value="' + p + '"' + sel + '>' + labelMap[p] + ' (~' + p + 'w)</option>';
    }});
    optsHtml += '<option value="custom"' + (isCustom ? ' selected' : '') + '>Custom\\u2026</option>';
    const customStyle = isCustom ? '' : 'display:none;';
    const customVal = (isCustom && tw) ? String(tw) : '';
    const effectiveTw = (tw && tw > 0) ? tw : perSection;
    const badgeClass = (tw && tw > 0) ? 'sec-target-badge override' : 'sec-target-badge';
    const badgeTag = (tw && tw > 0)
      ? '<span class="badge-tag">override</span>'
      : '<span class="badge-tag muted">auto</span>';

    html += '<div class="sec-row" data-slug="' + s.slug + '">';
    html += '  <div class="sec-fields">';
    html += '    <div style="font-weight:600;font-size:13px;color:var(--fg);margin-bottom:4px;">' +
            (i + 1) + '. ' + escapeHtml(s.title || s.slug) + '</div>';
    html += '    <textarea data-plan-slug="' + s.slug + '" placeholder="Section plan — what THIS section must cover" ' +
            'oninput="updatePlanChapterSection(\\'' + s.slug + '\\', this.value)">' +
            escapeHtml(s.plan || '') + '</textarea>';
    html += '    <div class="sec-size-row">';
    html += '      <label>Target:</label>';
    html += '      <select onchange="updatePlanChapterTargetWords(\\'' + s.slug + '\\', this.value)">' + optsHtml + '</select>';
    html += '      <input type="number" class="sec-size-custom" placeholder="words" min="100" step="100" ';
    html += '             value="' + customVal + '" style="' + customStyle + '" ';
    html += '             oninput="updatePlanChapterTargetWordsCustom(\\'' + s.slug + '\\', this.value)">';
    html += '      <span class="' + badgeClass + '">~' + effectiveTw + ' words ' + badgeTag + '</span>';
    html += '    </div>';
    html += '    <div class="sec-slug">slug: <code>' + s.slug + '</code></div>';
    html += '  </div>';
    html += '</div>';
  }});
  list.innerHTML = html;
}}

// Track plan edits for the chapter tab so save can collect them.
let _editingChapterPlans = {{}};
// Phase 32.2 — parallel maps for per-section target_words overrides
// edited in the Plan modal's Chapter sections tab. Keyed by slug.
// Reset whenever a different chapter is loaded into the tab.
let _editingChapterTargetWords = {{}};
let _editingChapterCustomMode = {{}};
function updatePlanChapterSection(slug, value) {{
  _editingChapterPlans[slug] = value;
}}
function updatePlanChapterTargetWords(slug, value) {{
  if (value === "" || value === "auto") {{
    _editingChapterTargetWords[slug] = null;
    _editingChapterCustomMode[slug] = false;
  }} else if (value === "custom") {{
    _editingChapterCustomMode[slug] = true;
    if (!_editingChapterTargetWords[slug]) _editingChapterTargetWords[slug] = 1500;
  }} else {{
    const n = parseInt(value, 10);
    _editingChapterTargetWords[slug] = isNaN(n) ? null : n;
    _editingChapterCustomMode[slug] = false;
  }}
  // Re-render the tab so the badge/custom-input visibility updates.
  const ch = chaptersData.find(c => c.id === _planContext.chapterId);
  if (ch) populatePlanChapterTab(ch);
}}
function updatePlanChapterTargetWordsCustom(slug, value) {{
  const n = parseInt(value, 10);
  _editingChapterTargetWords[slug] = (isNaN(n) || n <= 0) ? null : n;
  _editingChapterCustomMode[slug] = true;
  // Don't re-render — the user is actively typing in the custom input.
}}

// Phase 21 — populate the single-section editor for the Section tab.
// Phase 32.2 — also populates the per-section length dropdown.
let _editingPlanSectionTargetWords = null;  // null = auto, number = override
function populatePlanSectionTab(chapterId, sectionSlug) {{
  const ch = chaptersData.find(c => c.id === chapterId);
  if (!ch) return;
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];
  const sec = meta.find(s => s.slug === sectionSlug);
  const ctx = document.getElementById('plan-section-context');
  if (ctx) {{
    ctx.innerHTML = '<strong>Ch.' + ch.num + ': ' + escapeHtml(ch.title) + '</strong> &middot; ' +
      'section <code>' + sectionSlug + '</code>';
  }}
  document.getElementById('plan-section-title').value = sec ? sec.title : sectionSlug;
  document.getElementById('plan-section-text').value = sec ? (sec.plan || '') : '';

  // Phase 32.2 — seed the target dropdown from the section meta.
  const tw = (sec && sec.target_words && sec.target_words > 0) ? sec.target_words : null;
  _editingPlanSectionTargetWords = tw;
  const select = document.getElementById('plan-section-target-select');
  const customInput = document.getElementById('plan-section-target-custom');
  const presets = [400, 800, 1500, 3000, 6000];
  if (select) {{
    if (!tw) {{
      select.value = '';
    }} else if (presets.includes(tw)) {{
      select.value = String(tw);
    }} else {{
      select.value = 'custom';
    }}
  }}
  if (customInput) {{
    if (tw && !presets.includes(tw)) {{
      customInput.style.display = '';
      customInput.value = String(tw);
    }} else {{
      customInput.style.display = 'none';
      customInput.value = '';
    }}
  }}
  _refreshPlanSectionTargetBadge(meta.length);

  // If the section is missing from the meta (orphan), warn the user.
  if (!sec) {{
    const status = document.getElementById('plan-status');
    if (status) {{
      status.innerHTML = '<span style="color:var(--warning);">This section\\'s slug \\'' +
        sectionSlug + '\\' isn\\'t in the chapter\\'s sections list. Saving will add it.</span>';
    }}
  }}
}}

// Phase 32.2 — refresh the target badge text + style based on the
// current editing state. Called from both the dropdown and custom-
// input change handlers.
function _refreshPlanSectionTargetBadge(numSections) {{
  const badge = document.getElementById('plan-section-target-badge');
  if (!badge) return;
  const chapterTarget = (window._chapterWordTarget && window._chapterWordTarget > 0)
    ? window._chapterWordTarget : 6000;
  const n = Math.max(1, numSections || 1);
  const perSection = Math.max(400, Math.min(chapterTarget, Math.floor(chapterTarget / n)));
  const tw = _editingPlanSectionTargetWords;
  const effective = (tw && tw > 0) ? tw : perSection;
  const tag = (tw && tw > 0)
    ? '<span class="badge-tag">override</span>'
    : '<span class="badge-tag muted">auto</span>';
  badge.innerHTML = '~' + effective + ' words ' + tag;
  badge.className = (tw && tw > 0) ? 'sec-target-badge override' : 'sec-target-badge';
}}

function updatePlanSectionTargetWords(value) {{
  const customInput = document.getElementById('plan-section-target-custom');
  if (value === '' || value === 'auto') {{
    _editingPlanSectionTargetWords = null;
    if (customInput) {{ customInput.style.display = 'none'; customInput.value = ''; }}
  }} else if (value === 'custom') {{
    if (!_editingPlanSectionTargetWords) _editingPlanSectionTargetWords = 1500;
    if (customInput) {{
      customInput.style.display = '';
      customInput.value = String(_editingPlanSectionTargetWords);
      customInput.focus();
      customInput.select();
    }}
  }} else {{
    const n = parseInt(value, 10);
    _editingPlanSectionTargetWords = isNaN(n) ? null : n;
    if (customInput) {{ customInput.style.display = 'none'; customInput.value = ''; }}
  }}
  // Re-fetch num sections from the active chapter for the badge.
  const ch = chaptersData.find(c => c.id === _planContext.chapterId);
  const n = (ch && Array.isArray(ch.sections_meta)) ? ch.sections_meta.length : 1;
  _refreshPlanSectionTargetBadge(n);
}}

function updatePlanSectionTargetWordsCustom(value) {{
  const n = parseInt(value, 10);
  _editingPlanSectionTargetWords = (isNaN(n) || n <= 0) ? null : n;
  const ch = chaptersData.find(c => c.id === _planContext.chapterId);
  const num = (ch && Array.isArray(ch.sections_meta)) ? ch.sections_meta.length : 1;
  _refreshPlanSectionTargetBadge(num);
}}

// Phase 17 — length preset buttons in the Plan modal. Setting a preset
// just fills the input; the user still has to click Save to persist.
function setLengthPreset(words) {{
  const input = document.getElementById('plan-target-words-input');
  if (input) input.value = String(words);
  const lstatus = document.getElementById('plan-length-status');
  if (lstatus) lstatus.textContent = 'preset: ' + words + ' — click Save to apply';
}}

async function savePlan() {{
  // Phase 21 — savePlan dispatches based on which tab is active so a
  // single Save button works for all three contexts. Each branch saves
  // the relevant subset and refreshes the in-memory chaptersData cache.
  const activeTabBtn = document.querySelector('#plan-modal .tab.active');
  const tab = activeTabBtn ? activeTabBtn.dataset.tab : 'plan-book';

  if (tab === 'plan-book') {{
    return savePlanBook();
  }} else if (tab === 'plan-chapter') {{
    return savePlanChapterSections();
  }} else if (tab === 'plan-section') {{
    return savePlanSection();
  }}
}}

async function savePlanBook() {{
  const title = document.getElementById('plan-title-input').value.trim();
  const desc = document.getElementById('plan-desc-input').value.trim();
  const plan = document.getElementById('plan-text-input').value.trim();
  const tcwRaw = document.getElementById('plan-target-words-input').value.trim();
  document.getElementById('plan-status').textContent = 'Saving...';
  const fd = new FormData();
  fd.append('title', title);
  fd.append('description', desc);
  fd.append('plan', plan);
  if (tcwRaw !== '') {{
    const n = parseInt(tcwRaw, 10);
    if (!isNaN(n)) fd.append('target_chapter_words', String(n));
  }}
  try {{
    const res = await fetch('/api/book', {{method: 'PUT', body: fd}});
    if (!res.ok) throw new Error('save failed');
    document.getElementById('plan-status').innerHTML =
      '<span style="color:var(--success);">Saved.</span> ' +
      plan.split(/\\s+/).filter(Boolean).length + ' words. The new plan will be injected into all future writes.';
    if (title) document.querySelector('.sidebar h2').textContent = title;
    if (tcwRaw !== '') {{
      const n = parseInt(tcwRaw, 10);
      const lstatus = document.getElementById('plan-length-status');
      if (lstatus) {{
        if (n > 0) lstatus.textContent = 'current: ' + n + ' words/chapter';
        else lstatus.textContent = 'cleared — using default';
      }}
      window._chapterWordTarget = n > 0 ? n : 6000;
    }}
  }} catch (e) {{
    document.getElementById('plan-status').textContent = 'Save failed: ' + e.message;
  }}
}}

// Phase 21 — save edits to all section plans for the active chapter.
// Sends the full sections list (with merged plan edits) via PUT
// /api/chapters/{{id}}/sections, which is a full-replace endpoint —
// existing titles + slugs are preserved, only plans get updated.
async function savePlanChapterSections() {{
  const chId = _planContext.chapterId;
  if (!chId) return;
  const ch = chaptersData.find(c => c.id === chId);
  if (!ch) return;
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];

  // Apply pending plan + target_words edits.
  // Phase 32.2 — target_words: prefer the editing-state map (set by
  // the dropdown), fall back to whatever was on the section meta
  // when the tab was opened. null/0 means "use the chapter auto".
  const updated = meta.map(s => {{
    const tw = (s.slug in _editingChapterTargetWords)
      ? _editingChapterTargetWords[s.slug]
      : (s.target_words || null);
    return {{
      slug: s.slug,
      title: s.title || s.slug,
      plan: (s.slug in _editingChapterPlans) ? _editingChapterPlans[s.slug] : (s.plan || ''),
      target_words: (tw && tw > 0) ? tw : null,
    }};
  }});

  document.getElementById('plan-status').textContent = 'Saving section plans...';
  try {{
    const res = await fetch('/api/chapters/' + chId + '/sections', {{
      method: 'PUT',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{sections: updated}}),
    }});
    if (!res.ok) throw new Error('save failed');
    const data = await res.json();
    if (data && Array.isArray(data.sections)) {{
      ch.sections_meta = data.sections;
      ch.sections_template = data.sections.map(s => s.slug);
    }}
    _editingChapterPlans = {{}};
    _editingChapterTargetWords = {{}};
    _editingChapterCustomMode = {{}};
    document.getElementById('plan-status').innerHTML =
      '<span style="color:var(--success);">Saved ' + updated.length + ' section plans.</span>';
    // Refresh sidebar so plan tooltips update
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
  }} catch (e) {{
    document.getElementById('plan-status').textContent = 'Save failed: ' + e.message;
  }}
}}

// Phase 21 — save the single-section plan editor (Section tab).
// Round-trips through the same /sections endpoint by patching the
// chapter's full sections list with this one section's new plan.
async function savePlanSection() {{
  const chId = _planContext.chapterId;
  const slug = _planContext.sectionSlug;
  if (!chId || !slug) return;
  const ch = chaptersData.find(c => c.id === chId);
  if (!ch) return;
  const newPlan = document.getElementById('plan-section-text').value;
  const newTitle = document.getElementById('plan-section-title').value.trim();
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta.slice() : [];

  // Phase 32.2 — also persist the per-section target_words override
  // edited via the dropdown.
  const newTw = (_editingPlanSectionTargetWords && _editingPlanSectionTargetWords > 0)
    ? _editingPlanSectionTargetWords : null;

  let found = false;
  const updated = meta.map(s => {{
    if (s.slug === slug) {{
      found = true;
      return {{
        slug: s.slug,
        title: newTitle || s.title || s.slug,
        plan: newPlan,
        target_words: newTw,
      }};
    }}
    // Phase 32.2 — preserve target_words on every other section so
    // saving from the Section tab doesn't accidentally wipe overrides
    // set elsewhere in the chapter.
    return {{
      slug: s.slug,
      title: s.title || s.slug,
      plan: s.plan || '',
      target_words: (s.target_words && s.target_words > 0) ? s.target_words : null,
    }};
  }});
  // Orphan section (slug not in meta): append it.
  if (!found) {{
    updated.push({{slug: slug, title: newTitle || slug, plan: newPlan, target_words: newTw}});
  }}

  document.getElementById('plan-status').textContent = 'Saving section plan...';
  try {{
    const res = await fetch('/api/chapters/' + chId + '/sections', {{
      method: 'PUT',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{sections: updated}}),
    }});
    if (!res.ok) throw new Error('save failed');
    const data = await res.json();
    if (data && Array.isArray(data.sections)) {{
      ch.sections_meta = data.sections;
      ch.sections_template = data.sections.map(s => s.slug);
    }}
    document.getElementById('plan-status').innerHTML =
      '<span style="color:var(--success);">Section plan saved.</span> Will be injected into the next write/autowrite of this section.';
    // Refresh sidebar so the tooltip updates
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
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
// Phase 18 — chapter modal carries an in-memory copy of the chapter's
// sections list while the modal is open. Saved on Save, discarded on
// Close. Each item is {{slug, title, plan}}; new rows have an empty slug
// and the server slugifies the title on save.
let _editingSections = [];

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

  // Phase 21 — fetch the book's chapter word target so the section
  // editor can show "≈ N words per section". Cached on window so the
  // re-renders that happen on every keystroke don't refetch.
  if (!window._chapterWordTarget) {{
    fetch('/api/book').then(r => r.json()).then(d => {{
      window._chapterWordTarget = d.target_chapter_words || d.default_target_chapter_words || 6000;
      renderSectionEditor();
    }}).catch(() => {{
      window._chapterWordTarget = 6000;
    }});
  }}

  // Phase 18 — copy the sections meta into the editor's working state.
  // sections_meta is the rich [{{slug, title, plan, target_words}}, ...]
  // shape; falls back to deriving from sections_template (slugs only)
  // for legacy chapters that haven't been opened yet under the new schema.
  // Phase 32.1 — also copy target_words so a previously-saved per-section
  // override is restored when the modal reopens (was being silently
  // dropped, which is why the size dropdown always reset to "Auto").
  if (Array.isArray(ch.sections_meta) && ch.sections_meta.length > 0) {{
    _editingSections = ch.sections_meta.map(s => ({{
      slug: s.slug || '',
      title: s.title || '',
      plan: s.plan || '',
      target_words: (s.target_words && s.target_words > 0) ? s.target_words : null,
    }}));
  }} else if (Array.isArray(ch.sections_template)) {{
    _editingSections = ch.sections_template.map(slug => ({{
      slug: slug, title: titleifyClient(slug), plan: ''
    }}));
  }} else {{
    _editingSections = [];
  }}
  renderSectionEditor();

  switchChapterTab('ch-scope');
  openModal('chapter-modal');
}}

// Tiny client-side titleifier mirroring _titleify_slug. Used for legacy
// chapters that only have a flat slug list — the editor synthesizes a
// best-effort display title so the user can immediately see + edit.
function titleifyClient(slug) {{
  return (slug || '').replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
}}

function switchChapterTab(name) {{
  document.querySelectorAll('#chapter-modal .tab').forEach(t => {{
    t.classList.toggle('active', t.dataset.tab === name);
  }});
  document.getElementById('ch-scope-pane').style.display = (name === 'ch-scope') ? 'block' : 'none';
  document.getElementById('ch-sections-pane').style.display = (name === 'ch-sections') ? 'block' : 'none';
}}

// Phase 18 + Phase 21 — render the working sections list into the editor.
// Re-rendered on every change so reorder/add/delete reflect immediately.
// Phase 21 adds: live slug preview while typing the title, per-section
// word-count budget (chapter target / num sections), and a header showing
// the chapter-level target.
function renderSectionEditor() {{
  const list = document.getElementById('ch-sections-list');
  if (!list) return;
  // Compute the per-section word budget from the book's chapter target.
  // This is identical to core.book_ops._section_target_words: floor 400,
  // ceiling = chapter_target. We read the target from the cached book
  // settings populated by openPlanModal().
  const chapterTarget = (window._chapterWordTarget && window._chapterWordTarget > 0)
    ? window._chapterWordTarget
    : 6000;
  const n = Math.max(1, _editingSections.length || 1);
  const perSection = Math.max(400, Math.min(chapterTarget, Math.floor(chapterTarget / n)));

  if (_editingSections.length === 0) {{
    list.innerHTML = '<div style="font-size:12px;color:var(--fg-muted);padding:12px;text-align:center;border:1px dashed var(--border);border-radius:4px;">No sections yet. Click <strong>Add section</strong> to start.</div>';
    return;
  }}

  // Header showing the budget split
  let html = '<div style="font-size:11px;color:var(--fg-muted);margin-bottom:8px;padding:6px 10px;background:var(--toolbar-bg);border-radius:4px;">';
  html += '<strong>' + _editingSections.length + '</strong> section' + (_editingSections.length === 1 ? '' : 's') +
          ' &middot; chapter target: <strong>' + chapterTarget + '</strong> words &middot; ' +
          'per section: <strong>~' + perSection + '</strong> words';
  html += '</div>';

  _editingSections.forEach((s, i) => {{
    // Live slug preview: derive from current slug or fall back to a
    // slugified title (matches core.book_ops._slugify_section_name).
    const liveSlug = (s.slug || s.title || '').trim().toLowerCase().replace(/\\s+/g, '_');
    const slugDisplay = liveSlug || '<em>(slug auto-generated)</em>';
    // Phase 29 — per-section target_words override. The dropdown
    // offers presets + Custom (which reveals a number input). When
    // "Auto" is selected, target_words is null and the autowrite
    // resolution falls through to chapter target / num_sections.
    const tw = s.target_words;
    const isAuto = !tw || tw <= 0;
    const presets = [400, 800, 1500, 3000, 6000];
    // Phase 31 — bug fix: previously isCustom was derived from
    // `!presets.includes(tw)`, but the "Custom" branch initialized
    // tw=1500 (in presets) so on re-render isCustom became false and
    // the input stayed hidden. Fix: track an explicit _customMode flag
    // on the section dict, set when the user picks Custom, cleared
    // when they pick a preset or Auto.
    const presetMatch = !isAuto && presets.includes(tw);
    const isCustom = s._customMode || (!isAuto && !presetMatch);
    let optsHtml = '<option value="">Auto (~' + perSection + 'w)</option>';
    presets.forEach(p => {{
      const sel = (!isCustom && tw === p) ? ' selected' : '';
      const labelMap = {{400: 'Very short', 800: 'Short', 1500: 'Medium', 3000: 'Long', 6000: 'Extra long'}};
      optsHtml += '<option value="' + p + '"' + sel + '>' + labelMap[p] + ' (~' + p + 'w)</option>';
    }});
    optsHtml += '<option value="custom"' + (isCustom ? ' selected' : '') + '>Custom\u2026</option>';
    const customStyle = isCustom ? '' : 'display:none;';
    const customVal = (isCustom && tw) ? String(tw) : '';
    html += '<div class="sec-row" data-idx="' + i + '">';
    html += '  <div class="sec-handle">';
    html += '    <button onclick="moveSection(' + i + ', -1)" title="Move up"' + (i === 0 ? ' disabled style="opacity:0.3;cursor:default;"' : '') + '>&uarr;</button>';
    html += '    <button onclick="moveSection(' + i + ', 1)" title="Move down"' + (i === _editingSections.length - 1 ? ' disabled style="opacity:0.3;cursor:default;"' : '') + '>&darr;</button>';
    html += '  </div>';
    html += '  <div class="sec-fields">';
    html += '    <input type="text" placeholder="Section title (e.g. The 11-Year Solar Cycle)" ';
    html += '           value="' + escapeHtml(s.title) + '" oninput="updateSectionTitle(' + i + ', this.value)">';
    html += '    <textarea placeholder="Section plan — what THIS section must cover (a few sentences)" ';
    html += '              oninput="updateSection(' + i + ', \\'plan\\', this.value)">' + escapeHtml(s.plan) + '</textarea>';
    // Phase 29 — size dropdown row, just below the plan textarea.
    // Phase 32.1 — show the effective target words inline next to the
    // dropdown so the user always sees what budget THIS section will
    // be written to (rather than burying it in the muted slug line).
    const effectiveTw = (tw && tw > 0) ? tw : perSection;
    const targetBadgeClass = (tw && tw > 0) ? 'sec-target-badge override' : 'sec-target-badge';
    const targetBadgeTitle = (tw && tw > 0)
      ? 'Per-section override (set explicitly)'
      : 'Auto: chapter target / number of sections';
    html += '    <div class="sec-size-row">';
    html += '      <label>Target:</label>';
    html += '      <select onchange="updateSectionTargetWords(' + i + ', this.value)">' + optsHtml + '</select>';
    html += '      <input type="number" class="sec-size-custom" placeholder="words" min="100" step="100" ';
    html += '             value="' + customVal + '" style="' + customStyle + '" ';
    html += '             oninput="updateSectionTargetWordsCustom(' + i + ', this.value)">';
    html += '      <span class="' + targetBadgeClass + '" title="' + targetBadgeTitle + '">';
    html += '~' + effectiveTw + ' words';
    html += (tw && tw > 0 ? ' <span class="badge-tag">override</span>' : ' <span class="badge-tag muted">auto</span>');
    html += '</span>';
    html += '    </div>';
    html += '    <div class="sec-slug">slug: <code>' + slugDisplay + '</code></div>';
    html += '  </div>';
    html += '  <button class="sec-delete" onclick="removeSection(' + i + ')" title="Delete this section">&times;</button>';
    html += '</div>';
  }});
  list.innerHTML = html;
}}

// Phase 29/31 — handle the size dropdown selection. "" → Auto
// (clear override + clear custom mode), a numeric preset → set
// (and clear custom mode), "custom" → enter custom mode (reveal
// the number input).
function updateSectionTargetWords(idx, value) {{
  if (idx < 0 || idx >= _editingSections.length) return;
  const sec = _editingSections[idx];
  if (value === "" || value === "auto") {{
    sec.target_words = null;
    sec._customMode = false;
  }} else if (value === "custom") {{
    // Phase 31 — set the explicit _customMode flag so the next
    // re-render keeps the input visible regardless of whether
    // target_words happens to coincide with a preset value.
    sec._customMode = true;
    // Default to a starting value if there isn't one yet
    if (!sec.target_words) sec.target_words = 1500;
  }} else {{
    const n = parseInt(value, 10);
    sec.target_words = isNaN(n) ? null : n;
    sec._customMode = false;
  }}
  renderSectionEditor();
  // After re-render, focus the custom input if we just entered
  // custom mode so the user can type immediately.
  if (value === "custom") {{
    setTimeout(() => {{
      const rows = document.querySelectorAll('#ch-sections-list .sec-row');
      if (rows[idx]) {{
        const input = rows[idx].querySelector('.sec-size-custom');
        if (input) {{
          input.focus();
          input.select();
        }}
      }}
    }}, 0);
  }}
}}

function updateSectionTargetWordsCustom(idx, value) {{
  if (idx < 0 || idx >= _editingSections.length) return;
  const sec = _editingSections[idx];
  const n = parseInt(value, 10);
  if (isNaN(n) || n <= 0) {{
    sec.target_words = null;
  }} else {{
    sec.target_words = n;
  }}
  // Stay in custom mode while the user types — only the dropdown
  // change handler clears it.
  sec._customMode = true;
  // Don't re-render here — the user is actively typing in the input.
}}

// Phase 21 — title-input handler that ALSO updates the slug live so the
// slug preview shows immediately instead of waiting for save. We only
// auto-derive slug from title when slug is empty (untouched), so users
// who manually entered a slug aren't surprised when it overwrites.
function updateSectionTitle(idx, value) {{
  if (idx < 0 || idx >= _editingSections.length) return;
  const sec = _editingSections[idx];
  sec.title = value;
  // Live slug derivation only when slug is empty or matches the
  // previously-derived slug (so manual slug edits stick).
  if (!sec.slug) {{
    sec.slug = (value || '').trim().toLowerCase().replace(/\\s+/g, '_');
  }}
  // Re-render to refresh the live slug preview at the bottom of the row.
  // We preserve focus by re-finding the input and restoring the cursor
  // position after the innerHTML rebuild.
  const activeIdx = document.activeElement && document.activeElement.closest('.sec-row');
  const cursorPos = document.activeElement && document.activeElement.selectionStart;
  renderSectionEditor();
  if (activeIdx) {{
    const newRows = document.querySelectorAll('#ch-sections-list .sec-row');
    if (newRows[idx]) {{
      const newInput = newRows[idx].querySelector('input');
      if (newInput) {{
        newInput.focus();
        if (typeof cursorPos === 'number') {{
          try {{ newInput.setSelectionRange(cursorPos, cursorPos); }} catch (e) {{}}
        }}
      }}
    }}
  }}
}}

function escapeHtml(s) {{
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}}

// Phase 23 — collapse/expand chapter sections. Per-chapter chevron
// toggles a .collapsed class; state persists in localStorage so
// refreshes don't reset the user's view. The collapse-all button at
// the top of the sidebar flips every chapter at once.

const _COLLAPSED_KEY = 'sciknow.collapsedChapters';

function _getCollapsedChapterIds() {{
  try {{
    return JSON.parse(localStorage.getItem(_COLLAPSED_KEY) || '[]');
  }} catch (e) {{
    return [];
  }}
}}

function _setCollapsedChapterIds(ids) {{
  try {{
    localStorage.setItem(_COLLAPSED_KEY, JSON.stringify(ids));
  }} catch (e) {{ /* localStorage full or disabled — ignore */ }}
}}

function toggleChapter(group) {{
  if (!group) return;
  const chId = group.dataset.chId;
  group.classList.toggle('collapsed');
  const collapsed = _getCollapsedChapterIds();
  const idx = collapsed.indexOf(chId);
  if (group.classList.contains('collapsed')) {{
    if (idx === -1) collapsed.push(chId);
  }} else {{
    if (idx !== -1) collapsed.splice(idx, 1);
  }}
  _setCollapsedChapterIds(collapsed);
  _refreshToggleAllButton();
}}

function toggleAllChapters() {{
  const groups = document.querySelectorAll('#sidebar-sections .ch-group');
  // If ANY chapter is currently expanded, collapse all. Otherwise expand all.
  let anyExpanded = false;
  groups.forEach(g => {{ if (!g.classList.contains('collapsed')) anyExpanded = true; }});
  const newCollapsed = [];
  groups.forEach(g => {{
    if (anyExpanded) {{
      g.classList.add('collapsed');
      newCollapsed.push(g.dataset.chId);
    }} else {{
      g.classList.remove('collapsed');
    }}
  }});
  _setCollapsedChapterIds(newCollapsed);
  _refreshToggleAllButton();
}}

function _refreshToggleAllButton() {{
  const groups = document.querySelectorAll('#sidebar-sections .ch-group');
  if (groups.length === 0) return;
  let anyExpanded = false;
  groups.forEach(g => {{ if (!g.classList.contains('collapsed')) anyExpanded = true; }});
  const icon = document.getElementById('toggle-all-icon');
  const label = document.getElementById('toggle-all-label');
  if (icon) icon.textContent = anyExpanded ? '\\u25bd' : '\\u25b7';
  if (label) label.textContent = anyExpanded ? 'Collapse all' : 'Expand all';
}}

// Restore the persisted collapsed state on page load + after every
// rebuildSidebar. Idempotent — safe to call multiple times.
function restoreCollapsedChapters() {{
  const collapsed = _getCollapsedChapterIds();
  collapsed.forEach(chId => {{
    const group = document.querySelector(
      '#sidebar-sections .ch-group[data-ch-id="' + chId + '"]'
    );
    if (group) group.classList.add('collapsed');
  }});
  _refreshToggleAllButton();
}}

// Run on initial page load (after the static sidebar HTML is parsed).
document.addEventListener('DOMContentLoaded', restoreCollapsedChapters);

// Phase 22 — word target progress bar in the subtitle. Shows
// "actual/target" plus a coloured bar (warning under 70%, accent
// 70-100%, success over). Hidden when no target is set.
function updateWordTargetBar(actual, target) {{
  const wrap = document.getElementById('word-target');
  const fill = document.getElementById('word-target-fill');
  const txt  = document.getElementById('word-target-text');
  if (!wrap || !fill || !txt) return;
  if (!target || target <= 0) {{
    wrap.style.display = 'none';
    return;
  }}
  wrap.style.display = 'inline-flex';
  const pct = Math.min(150, Math.round((actual / target) * 100));
  fill.style.width = Math.min(100, pct) + '%';
  fill.classList.remove('over', 'under');
  if (pct >= 100) fill.classList.add('over');
  else if (pct < 70) fill.classList.add('under');
  txt.textContent = actual + ' / ' + target + 'w';
}}

// Phase 22 — delete an orphan draft from the sidebar. Confirms first
// because deletion is permanent.
async function deleteOrphanDraft(draftId) {{
  if (!draftId) return;
  if (!confirm('Permanently delete this orphan draft? This cannot be undone.')) return;
  try {{
    const res = await fetch('/api/draft/' + draftId, {{method: 'DELETE'}});
    if (!res.ok) throw new Error('delete failed (' + res.status + ')');
    // If the deleted draft was the active one, fall back to the
    // dashboard so we don't show stale content.
    if (currentDraftId === draftId) {{
      currentDraftId = '';
      showDashboard();
    }}
    // Refresh sidebar so the orphan disappears
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
  }} catch (e) {{
    alert('Delete failed: ' + e.message);
  }}
}}

// ── Phase 26: drag-and-drop section reordering ────────────────────────
//
// Sections in the sidebar are HTML5-draggable. The user clicks and
// holds a section row, drags it above or below another section in
// the SAME chapter, and drops it. The handler computes the new
// order, sends the full sections list to PUT /api/chapters/{{id}}/sections
// (the existing full-replace endpoint, also used by the chapter
// modal's Sections tab), and refreshes the sidebar.
//
// Cross-chapter drags are not supported — they would need to also
// update drafts.chapter_id, which is a more dangerous operation
// that deserves its own confirmation flow. The handler silently
// rejects drops onto a different .ch-group.
//
// Click vs drag: the browser distinguishes a plain click (no
// movement) from a drag (movement during mousedown). The existing
// onclick handlers for navigation (navTo, writeForCell) keep
// firing on plain clicks even though the row is draggable.

let _draggedSection = null;

function _findDraggableSection(target) {{
  return target && target.closest && target.closest('.sec-link[draggable="true"]');
}}

function handleSectionDragStart(e) {{
  const link = _findDraggableSection(e.target);
  if (!link) return;
  const group = link.closest('.ch-group');
  if (!group) return;
  _draggedSection = {{
    chapterId: group.dataset.chId,
    slug: link.dataset.sectionSlug,
  }};
  link.classList.add('dragging');
  if (e.dataTransfer) {{
    e.dataTransfer.effectAllowed = 'move';
    // Required by Firefox to actually fire dragstart properly.
    e.dataTransfer.setData('text/plain', _draggedSection.slug);
  }}
}}

function handleSectionDragOver(e) {{
  if (!_draggedSection) return;
  const link = _findDraggableSection(e.target);
  if (!link) return;
  // Only allow drops within the same chapter.
  const group = link.closest('.ch-group');
  if (!group || group.dataset.chId !== _draggedSection.chapterId) return;
  // Don't show a drop indicator on the dragged row itself.
  if (link.dataset.sectionSlug === _draggedSection.slug) return;
  e.preventDefault();
  if (e.dataTransfer) e.dataTransfer.dropEffect = 'move';
  // Compute drop position based on cursor Y vs row midpoint.
  const rect = link.getBoundingClientRect();
  const isAbove = e.clientY < (rect.top + rect.height / 2);
  // Clear previous indicator on any other row, set new one.
  document.querySelectorAll('.sec-link.drag-over-top, .sec-link.drag-over-bottom')
    .forEach(el => el.classList.remove('drag-over-top', 'drag-over-bottom'));
  link.classList.add(isAbove ? 'drag-over-top' : 'drag-over-bottom');
}}

function handleSectionDrop(e) {{
  if (!_draggedSection) {{ _cleanupDrag(); return; }}
  const link = _findDraggableSection(e.target);
  if (!link) {{ _cleanupDrag(); return; }}
  const group = link.closest('.ch-group');
  if (!group || group.dataset.chId !== _draggedSection.chapterId) {{
    _cleanupDrag();
    return;
  }}
  e.preventDefault();
  const targetSlug = link.dataset.sectionSlug;
  if (targetSlug === _draggedSection.slug) {{
    _cleanupDrag();
    return;
  }}
  const rect = link.getBoundingClientRect();
  const position = e.clientY < (rect.top + rect.height / 2) ? 'before' : 'after';
  reorderSections(_draggedSection.chapterId, _draggedSection.slug, targetSlug, position);
  _cleanupDrag();
}}

function handleSectionDragEnd() {{
  _cleanupDrag();
}}

function _cleanupDrag() {{
  _draggedSection = null;
  document.querySelectorAll('.sec-link.dragging')
    .forEach(el => el.classList.remove('dragging'));
  document.querySelectorAll('.sec-link.drag-over-top, .sec-link.drag-over-bottom')
    .forEach(el => el.classList.remove('drag-over-top', 'drag-over-bottom'));
}}

// Reorder a section by sending the full updated sections list to the
// existing PUT /api/chapters/{{id}}/sections endpoint. Idempotent and
// safe — the endpoint is full-replace, not patch.
async function reorderSections(chapterId, draggedSlug, targetSlug, position) {{
  const ch = chaptersData.find(c => c.id === chapterId);
  if (!ch || !Array.isArray(ch.sections_meta)) return;
  const sections = ch.sections_meta.slice();
  const draggedIdx = sections.findIndex(s => s.slug === draggedSlug);
  if (draggedIdx === -1) return;
  const dragged = sections.splice(draggedIdx, 1)[0];
  // After removing the dragged row, the target index may have shifted.
  const targetIdx = sections.findIndex(s => s.slug === targetSlug);
  if (targetIdx === -1) {{
    sections.push(dragged);
  }} else {{
    const insertAt = position === 'before' ? targetIdx : targetIdx + 1;
    sections.splice(insertAt, 0, dragged);
  }}
  try {{
    const res = await fetch('/api/chapters/' + chapterId + '/sections', {{
      method: 'PUT',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{sections: sections}}),
    }});
    if (!res.ok) throw new Error('reorder failed (' + res.status + ')');
    const data = await res.json();
    // Patch local cache so the next render uses the new order
    // immediately (the /api/chapters refetch below is just to be sure).
    if (data && Array.isArray(data.sections)) {{
      ch.sections_meta = data.sections;
      ch.sections_template = data.sections.map(s => s.slug);
    }}
    // Full sidebar refresh — preserves collapsed state via Phase 23
    // restoreCollapsedChapters() at the end of rebuildSidebar.
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
  }} catch (e) {{
    alert('Reorder failed: ' + e.message);
  }}
}}

// Wire drag-and-drop handlers via event delegation on the sidebar
// container. Idempotent — re-running attaches a NEW listener but the
// old ones are still there. We only call this once on DOMContentLoaded
// (the container element is stable; only its children get replaced
// by rebuildSidebar, but event delegation handles that automatically).
let _sectionDndWired = false;
function setupSectionDragDrop() {{
  if (_sectionDndWired) return;
  const container = document.getElementById('sidebar-sections');
  if (!container) return;
  container.addEventListener('dragstart', handleSectionDragStart);
  container.addEventListener('dragover', handleSectionDragOver);
  container.addEventListener('drop', handleSectionDrop);
  container.addEventListener('dragend', handleSectionDragEnd);
  // Also clear the drop indicator if the user drags outside the container
  container.addEventListener('dragleave', e => {{
    // Only clear when leaving the whole sidebar, not jumping between rows
    if (!container.contains(e.relatedTarget)) {{
      document.querySelectorAll('.sec-link.drag-over-top, .sec-link.drag-over-bottom')
        .forEach(el => el.classList.remove('drag-over-top', 'drag-over-bottom'));
    }}
  }});
  _sectionDndWired = true;
}}
document.addEventListener('DOMContentLoaded', setupSectionDragDrop);

// Phase 25 — adopt an orphan draft's slug into the chapter's sections
// list. Idempotent: if the slug already exists, the server returns
// added=false and we just refresh the sidebar (which is a no-op
// visually). On success the orphan re-classifies as drafted.
async function adoptOrphanSection(chapterId, slug) {{
  if (!chapterId || !slug) return;
  try {{
    const res = await fetch('/api/chapters/' + chapterId + '/sections/adopt', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{slug: slug}}),
    }});
    if (!res.ok) {{
      const text = await res.text();
      throw new Error('adopt failed: ' + text);
    }}
    const data = await res.json();
    // Update the in-memory chapter cache so the next sidebar render
    // reflects the new section without a full /api/chapters refetch.
    const ch = chaptersData.find(c => c.id === chapterId);
    if (ch && data.sections) {{
      ch.sections_meta = data.sections;
      ch.sections_template = data.sections.map(s => s.slug);
    }}
    // Refresh sidebar so the row re-renders as a drafted section.
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
  }} catch (e) {{
    alert('Adopt failed: ' + e.message);
  }}
}}

function addSection() {{
  _editingSections.push({{slug: '', title: '', plan: ''}});
  renderSectionEditor();
  // Focus the new row's title input.
  setTimeout(() => {{
    const rows = document.querySelectorAll('#ch-sections-list .sec-row');
    if (rows.length > 0) {{
      const last = rows[rows.length - 1];
      const input = last.querySelector('input');
      if (input) input.focus();
    }}
  }}, 50);
}}

function removeSection(idx) {{
  if (idx < 0 || idx >= _editingSections.length) return;
  const s = _editingSections[idx];
  // Confirm if the section has content (non-empty title).
  if (s.title || s.plan) {{
    if (!confirm('Delete section "' + (s.title || s.slug) + '"? This will not delete any existing drafts.')) return;
  }}
  _editingSections.splice(idx, 1);
  renderSectionEditor();
}}

function moveSection(idx, delta) {{
  const j = idx + delta;
  if (j < 0 || j >= _editingSections.length) return;
  const tmp = _editingSections[idx];
  _editingSections[idx] = _editingSections[j];
  _editingSections[j] = tmp;
  renderSectionEditor();
}}

function updateSection(idx, field, value) {{
  if (idx < 0 || idx >= _editingSections.length) return;
  _editingSections[idx][field] = value;
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
    // 1) Save scope (existing endpoint)
    const res = await fetch('/api/chapters/' + chId, {{method: 'PUT', body: fd}});
    if (!res.ok) throw new Error('save failed');

    // 2) Save sections (Phase 18) — only sections with a non-empty
    // title are persisted. The server slugifies for us.
    // Phase 29 — also include target_words per section.
    const sectionsToSave = _editingSections
      .filter(s => (s.title || '').trim() || (s.slug || '').trim())
      .map(s => ({{
        slug: (s.slug || '').trim() || (s.title || '').trim(),
        title: (s.title || '').trim() || (s.slug || ''),
        plan: (s.plan || '').trim(),
        target_words: (s.target_words && s.target_words > 0) ? s.target_words : null,
      }}));
    const secRes = await fetch('/api/chapters/' + chId + '/sections', {{
      method: 'PUT',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{sections: sectionsToSave}})
    }});
    if (!secRes.ok) throw new Error('sections save failed');
    const secData = await secRes.json();

    // Update the in-memory chapter cache
    const ch = chaptersData.find(c => c.id === chId);
    if (ch) {{
      if (title) ch.title = title;
      ch.description = desc;
      ch.topic_query = tq;
      if (secData && Array.isArray(secData.sections)) {{
        ch.sections_meta = secData.sections;
        ch.sections_template = secData.sections.map(s => s.slug);
      }}
    }}
    status.innerHTML = '<span style="color:var(--success);">Saved.</span>';
    // Refresh the sidebar so renamed chapters + new sections show up
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
    html += '<div class="cc-type">' + (c.section_title || (c.section_type.charAt(0).toUpperCase() + c.section_type.slice(1))) + '</div>';
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

  // Phase 31 — context-aware Read button. If the user has a section
  // selected, fetch only that section in the reader layout (same h2
  // styling, sources panel, citation popovers — just one section).
  // If only the chapter is selected, show the whole chapter.
  const url = currentSectionType
    ? ('/api/chapter-reader/' + currentChapterId + '?only_section=' + encodeURIComponent(currentSectionType))
    : ('/api/chapter-reader/' + currentChapterId);
  const res = await fetch(url);
  if (!res.ok) {{ alert('Chapter not found.'); return; }}
  const data = await res.json();

  const isSectionOnly = !!currentSectionType;
  let html = '<div class="reader-view">';
  html += '<h1>Chapter ' + data.chapter_num + ': ' + data.chapter_title + '</h1>';
  html += '<p style="font-size:13px;opacity:0.5;margin-bottom:8px;">' +
    data.total_words + ' words \\u00b7 ' + data.section_count + ' section' +
    (data.section_count === 1 ? '' : 's') +
    (isSectionOnly ? ' &middot; <strong>showing only ' + escapeHtml(currentSectionType) + '</strong>' : '') +
    '</p>';
  // Phase 18 — outline / table-of-contents at the top of the chapter
  // view so the user can see the section structure at a glance and
  // jump straight to any section.
  if (data.outline && data.outline.length > 0) {{
    html += '<div style="margin-bottom:24px;padding:12px 16px;background:var(--toolbar-bg);border-radius:6px;font-size:13px;">';
    html += '<div style="font-weight:600;margin-bottom:6px;color:var(--fg-muted);font-size:11px;text-transform:uppercase;letter-spacing:0.04em;">Sections</div>';
    data.outline.forEach((o, i) => {{
      html += '<div style="margin:4px 0;"><a href="#reader-section-' + o.slug + '" style="color:var(--accent);text-decoration:none;">' + (i + 1) + '. ' + escapeHtml(o.title) + '</a> <span style="color:var(--fg-muted);font-size:11px;">' + o.words + 'w</span></div>';
    }});
    html += '</div>';
  }}
  html += data.html;
  html += '</div>';

  document.getElementById('dashboard-view').innerHTML = html;
  document.getElementById('dashboard-view').style.display = 'block';
  document.getElementById('read-view').style.display = 'none';
  document.getElementById('edit-view').style.display = 'none';
  document.getElementById('draft-title').textContent = 'Ch.' + data.chapter_num + ': ' + data.chapter_title;
  document.getElementById('draft-subtitle').style.display = 'none';
  document.getElementById('toolbar').style.display = 'none';

  // Phase 18 — populate the right-hand sources panel with the chapter's
  // global (renumbered) source list so [N] click-to-source works in
  // the chapter reader view. Without this, panel-sources still has
  // whatever section was last loaded — and the global citation
  // numbers would point at the wrong papers.
  if (data.sources_html) {{
    document.getElementById('panel-sources').innerHTML = data.sources_html;
  }}

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
