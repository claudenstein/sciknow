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
from collections import deque
from datetime import datetime, timezone
from html import escape as _esc
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

logger = logging.getLogger("sciknow.web")

app = FastAPI(title="SciKnow Book Reader")

# Phase 33 — build tag: a short version string visible in the browser
# tab title and in the DevTools console so the user can instantly tell
# whether their browser has stale JS (the "hard-refresh to see the new
# chevron" problem from Phase 25). Computed once at import time from
# the git short hash if possible, else from the current UTC timestamp.
def _compute_build_tag() -> str:
    import subprocess
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")

_BUILD_TAG = _compute_build_tag()


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
            # Phase 32.5 — server-side counters so the task bar can poll
            # GET /api/jobs/{id}/stats instead of opening a SECOND SSE
            # source on /api/stream/{id}, which would compete with the
            # per-section preview consumer for the same asyncio.Queue
            # (Queue.get() removes items, so two consumers split the
            # event stream and neither sees a coherent view). Polling a
            # server-side counter is the only architecturally correct
            # way to deliver the same stats to two independent UIs.
            "started_at": time.monotonic(),
            # Phase 35 — wallclock start so we can persist a real
            # timestamp to llm_usage_log on completion. `started_at`
            # above is a monotonic stopwatch reading that can't be
            # written to a TIMESTAMPTZ column.
            "started_wall": datetime.now(timezone.utc),
            "tokens": 0,
            # 200-token rolling window is enough for ~30s of fast LLMs
            "token_timestamps": deque(maxlen=200),
            "model_name": None,
            "task_desc": job_type,
            "target_words": None,
            "stream_state": "streaming",  # streaming | done | error
            "error_message": None,
            # Phase 50.A — reasoning-steps trace. Collected automatically
            # by _observe_event_for_stats from the same event stream the
            # task bar already watches; persisted to the new draft's
            # custom_metadata.reasoning_trace in _run_generator_in_thread's
            # finally block. Capped to avoid log-level bloat.
            "reasoning_trace": [],
            "reasoning_draft_id": None,
        }
    return job_id, queue


def _observe_event_for_stats(job_id: str, event: dict) -> None:
    """Phase 32.5 — bump the per-job counters as events flow through.

    Called from `_run_generator_in_thread` BEFORE the event is put on
    the consumer queue. The poll endpoint reads the resulting
    counters; the consumer queue is left untouched (so the per-section
    SSE preview keeps working unchanged).

    Tracks token count, rolling timestamps for tokens-per-second,
    model name, task description, target words, and lifecycle state.
    Lock-protected dict mutation — fast enough to not bottleneck the
    streaming generator.
    """
    et = event.get("type")
    if not et:
        return
    with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        if et == "token":
            text_val = event.get("text") or ""
            # Approximate token count by whitespace splits — same
            # heuristic the previous client-side code used. Falls back
            # to 1 for short non-whitespace bursts.
            n = len([w for w in text_val.split() if w]) or 1
            job["tokens"] += n
            now = time.monotonic()
            ts = job["token_timestamps"]
            for _ in range(n):
                ts.append(now)
        elif et == "model_info":
            mn = event.get("writer_model") or event.get("model")
            if mn:
                job["model_name"] = mn
        elif et == "progress":
            d = event.get("detail") or event.get("stage")
            if d:
                job["task_desc"] = str(d)[:200]
        elif et == "length_target":
            tw = event.get("target_words")
            if tw:
                job["target_words"] = int(tw)
        elif et in ("completed", "all_sections_complete"):
            job["stream_state"] = "done"
        elif et == "cancelled":
            job["stream_state"] = "done"
            job["task_desc"] = "Stopped"
        elif et == "error":
            job["stream_state"] = "error"
            job["error_message"] = str(event.get("message") or "unknown")[:200]

        # Phase 50.A — reasoning-steps trace. Record a compact entry
        # for every event that isn't pure token streaming (skip 'token'
        # — too noisy; skip unknown-type). Each entry stays ≤300 bytes
        # so an autowrite run with ~30 step transitions accumulates a
        # trace well under 10 KB.
        if et not in ("token",):
            trace = job.get("reasoning_trace")
            if trace is not None and len(trace) < 60:
                entry: dict = {
                    "t": round(time.monotonic() - job["started_at"], 3),
                    "type": et,
                }
                for key in ("phase", "stage", "detail", "progress",
                            "score", "iteration", "step", "model",
                            "writer_model", "target_words", "words"):
                    val = event.get(key)
                    if val is None:
                        continue
                    if isinstance(val, (int, float, bool)):
                        entry[key] = val
                    elif isinstance(val, str):
                        entry[key] = val[:140]
                trace.append(entry)
            # Capture the first draft_id we see so _run_generator_in_thread
            # can persist the trace at the end even if later events swap it
            did = event.get("draft_id") or event.get("new_draft_id")
            if did and not job.get("reasoning_draft_id"):
                job["reasoning_draft_id"] = str(did)


def _finish_job(job_id: str):
    with _job_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["finished_at"] = time.monotonic()


def _persist_reasoning_trace(job_id: str) -> None:
    """Phase 50.A — append the per-job reasoning trace to the draft's
    custom_metadata once the streaming generator finishes. Stored
    under the `reasoning_trace` key in drafts.custom_metadata (JSONB)
    so retrieval can read it via the existing draft row without a
    schema migration. Best-effort: any exception is swallowed — the
    trace is a debug nicety and must never affect the draft save."""
    with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        trace = job.get("reasoning_trace") or []
        draft_id = job.get("reasoning_draft_id")
        job_type = job.get("type") or "unknown"
    if not trace or not draft_id:
        return
    try:
        from sqlalchemy import text as _sql_text
        from sciknow.storage.db import get_session
        with get_session() as session:
            # Merge into the existing custom_metadata JSON object in a
            # single statement so we don't clobber other keys (e.g.
            # Phase 17 autowrite telemetry already writes there).
            session.execute(_sql_text("""
                UPDATE drafts
                   SET custom_metadata = COALESCE(custom_metadata, '{}'::jsonb)
                       || jsonb_build_object(
                            'reasoning_trace',
                            CAST(:trace AS jsonb),
                            'reasoning_op',
                            CAST(:op AS text)
                          )
                 WHERE id::text = :did
            """), {
                "trace": json.dumps(trace),
                "op": job_type,
                "did": draft_id,
            })
            session.commit()
    except Exception as exc:
        logger.debug("reasoning-trace persist skipped for %s: %s", draft_id, exc)


def _persist_llm_usage(job_id: str) -> None:
    """Phase 35 — append one row to llm_usage_log when a job finishes.

    Called from `_run_generator_in_thread`'s finally block. Reads the
    per-job counters maintained by `_observe_event_for_stats` and writes
    a single compact row. Skip if tokens == 0 so zero-token noops (e.g.
    immediate error before the first token, or cancel-before-start)
    don't pollute the ledger.

    All fields are best-effort — if the insert fails we swallow the
    exception, the counter is an accounting nicety and must never take
    down a streaming endpoint.
    """
    with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        tokens = int(job.get("tokens") or 0)
        if tokens <= 0:
            return
        op_name = job.get("type") or "unknown"
        model_name = job.get("model_name")
        started_wall = job.get("started_wall")
        stream_state = job.get("stream_state") or "done"
        # stream_state is (streaming|done|error); fold "Stopped" (from
        # the cancel branch in _observe_event_for_stats) into cancelled.
        if stream_state == "error":
            status_val = "error"
        elif job.get("task_desc") == "Stopped":
            status_val = "cancelled"
        else:
            status_val = "completed"
        started_mono = job.get("started_at")
        now_mono = time.monotonic()
        duration = (now_mono - started_mono) if started_mono else None
    try:
        finished_wall = datetime.now(timezone.utc)
        with get_session() as session:
            session.execute(
                text(
                    """
                    INSERT INTO llm_usage_log
                        (book_id, chapter_id, operation, model_name,
                         tokens, duration_seconds, status,
                         started_at, finished_at)
                    VALUES
                        (CAST(:bid AS uuid), NULL, :op, :model,
                         :tokens, :dur, :status,
                         :started, :finished)
                    """
                ),
                {
                    "bid": _book_id or None,
                    "op": op_name,
                    "model": model_name,
                    "tokens": tokens,
                    "dur": duration,
                    "status": status_val,
                    "started": started_wall,
                    "finished": finished_wall,
                },
            )
            session.commit()
    except Exception as exc:
        logger.warning("llm_usage_log insert failed for job %s: %s", job_id, exc)


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


def _slugify_heading(text: str, seen: dict) -> str:
    """Turn a heading string into a URL-safe id. Counter-suffixes repeats
    so two headings with the same text still get unique ids. Empty /
    punctuation-only falls back to 'section'."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:60]
    if not s:
        s = "section"
    n = seen.get(s, 0)
    seen[s] = n + 1
    return s if n == 0 else f"{s}-{n}"


def _md_to_html(text_content: str) -> str:
    """Markdown -> HTML conversion for draft + wiki content.

    Phase 54 additions (Wiki UX research doc #2 + #5):
      - Every heading gets a slug-safe `id=` so TOC + deep-link
        scrolling work (``#wiki/<slug>#heading-id`` handled client-
        side).
      - ``[[slug]]`` and ``[[slug|alt text]]`` render as real
        hyperlinks into the wiki SPA route. Previously the syntax was
        parsed by the linter but never made it past the renderer.

    All edits stay backwards-compatible: book-draft content that
    happens to use either syntax is free to. Wiki-links to slugs that
    don't exist are still emitted as links — the router surfaces a
    "no such page" state on follow, which is the MediaWiki convention.
    """
    if not text_content:
        return ""
    html = text_content

    # Wiki-links go BEFORE emphasis so ``**[[foo]]**`` still becomes
    # a bold wiki-link rather than a bold-then-link mismatch.
    # [[slug|alt text]]
    html = re.sub(
        r"\[\[([^\]\|]+)\|([^\]]+)\]\]",
        r'<a class="wiki-link" href="#wiki/\1">\2</a>',
        html,
    )
    # [[slug]]  (use the slug itself as display text)
    html = re.sub(
        r"\[\[([^\]]+)\]\]",
        r'<a class="wiki-link" href="#wiki/\1">\1</a>',
        html,
    )

    # Headers with IDs for auto-TOC + deep linking.
    _heading_seen: dict[str, int] = {}

    def _h_sub(tag: str):
        def _inner(match):
            text_val = match.group(1)
            hid = _slugify_heading(text_val, _heading_seen)
            return f'<{tag} id="{hid}">{text_val}</{tag}>'
        return _inner

    html = re.sub(r"^### (.+)$", _h_sub("h4"), html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$",  _h_sub("h3"), html, flags=re.MULTILINE)
    html = re.sub(r"^# (.+)$",   _h_sub("h2"), html, flags=re.MULTILINE)

    # Bold and italic
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
    # Citation references [N] -> styled spans
    html = re.sub(r"\[(\d+)\]", r'<span class="citation" data-ref="\1">[\1]</span>', html)
    # Paragraphs
    paragraphs = html.split("\n\n")
    html = "".join(
        f'<p data-para="{i}">{p.strip()}</p>' if not p.strip().startswith("<h")
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
    # Phase 39 — expose the style fingerprint so the Book Settings
    # modal can render metrics + a "last refreshed" stamp without a
    # second round-trip.
    style_fingerprint = meta.get("style_fingerprint") if isinstance(meta, dict) else None
    return {
        "id": book[0] if book else "",
        "title": book[1] if book else "",
        "description": (book[2] or "") if book else "",
        "plan": (book[3] or "") if book else "",
        "status": (book[4] or "draft") if book else "draft",
        "target_chapter_words": target_chapter_words,  # may be None → client shows default
        "default_target_chapter_words": 6000,
        "style_fingerprint": style_fingerprint,
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


@app.post("/api/book/style-fingerprint/refresh")
async def api_book_style_fingerprint_refresh():
    """Phase 39 — recompute the book's style fingerprint on demand.

    Mirrors `sciknow book style refresh` in the web layer so the Book
    Settings modal can trigger a rebuild without dropping to the CLI.
    Runs synchronously (no SSE) because the work is pure SQL + regex
    over the book's drafts — sub-second for books with <500 drafts.
    """
    from sciknow.core.style_fingerprint import compute_style_fingerprint
    try:
        fp = compute_style_fingerprint(_book_id)
    except Exception as exc:
        logger.warning("style fingerprint refresh failed: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, "fingerprint": fp})


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
            """), {"bid": _book_id}).fetchone()
            if aw:
                autowrite_stats = {
                    "total_runs": int(aw[0] or 0),
                    "total_tokens": int(aw[1] or 0),
                    "total_seconds": int(aw[2] or 0),
                }
    except Exception as exc:
        logger.warning("dashboard autowrite stats failed: %s", exc)

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
            """), {"bid": _book_id}).fetchone()
            by_op = session.execute(text("""
                SELECT operation,
                       COUNT(*),
                       COALESCE(SUM(tokens), 0),
                       COALESCE(SUM(duration_seconds), 0)
                FROM llm_usage_log
                WHERE book_id::text = :bid
                GROUP BY operation
                ORDER BY SUM(tokens) DESC
            """), {"bid": _book_id}).fetchall()
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
        logger.warning("dashboard total_compute stats failed: %s", exc)

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


# ── Phase 38: chapter + book snapshots (bundle of all drafts) ────────────────
# Design:
#   - A scope='chapter' snapshot stores {"chapter_id", "drafts": [...]}
#     in `content` as JSON. Each draft entry carries section_type,
#     title, version, word_count, and the full content text at snapshot
#     time.
#   - A scope='book' snapshot stores {"book_id", "chapters": [{chapter
#     bundle}, ...]} — the natural union.
#   - Restore is NON-DESTRUCTIVE for chapter/book snapshots: it inserts
#     new draft rows with fresh version numbers instead of overwriting
#     existing content. Safer than the per-draft overwrite path because
#     chapter restores touch many sections at once.
#
# Snapshot the chapter BEFORE firing autowrite-all. Restore if the
# autowrite-all run produces worse output than what you had.


def _snapshot_chapter_drafts(session, chapter_id: str) -> dict:
    """Build the JSON bundle that will be stored in `content`.

    Captures the LATEST version per (chapter_id, section_type) —
    i.e. what the user would see in the GUI right now. Orphan drafts
    (chapter_id set but section_type is None or already replaced) are
    not special-cased: we just take the newest row per section_type.
    """
    rows = session.execute(text("""
        SELECT DISTINCT ON (d.section_type)
            d.id::text, d.section_type, d.title, d.version, d.word_count,
            d.content, d.sources
        FROM drafts d
        WHERE d.chapter_id::text = :cid
        ORDER BY d.section_type, d.version DESC, d.created_at DESC
    """), {"cid": chapter_id}).fetchall()
    return {
        "chapter_id": chapter_id,
        "drafts": [
            {
                "id": r[0],
                "section_type": r[1],
                "title": r[2] or "",
                "version": r[3] or 1,
                "word_count": r[4] or 0,
                "content": r[5] or "",
                "sources": r[6] if isinstance(r[6], list) else [],
            }
            for r in rows
        ],
    }


@app.post("/api/snapshot/chapter/{chapter_id}")
async def create_chapter_snapshot(chapter_id: str, name: str = Form("")):
    """Snapshot every draft in a chapter as one bundle.

    Phase 38 — the safety net for autowrite-all on a chapter. Takes a
    label, stores the chapter's current draft state as a JSON bundle
    in a single `draft_snapshots` row with scope='chapter'.
    """
    import datetime as _dt
    with get_session() as session:
        ch = session.execute(text(
            "SELECT id::text, title FROM book_chapters WHERE id::text = :cid"
        ), {"cid": chapter_id}).fetchone()
        if not ch:
            raise HTTPException(404, "Chapter not found")
        bundle = _snapshot_chapter_drafts(session, chapter_id)
        if not bundle["drafts"]:
            raise HTTPException(400, "Chapter has no drafts to snapshot")
        snap_name = (name or "").strip() or (
            f"{ch[1]} — {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        payload = json.dumps(bundle)
        total_words = sum(d.get("word_count") or 0 for d in bundle["drafts"])
        session.execute(text("""
            INSERT INTO draft_snapshots
                (chapter_id, scope, name, content, word_count)
            VALUES
                (CAST(:cid AS uuid), 'chapter', :name, :content, :wc)
        """), {"cid": chapter_id, "name": snap_name,
               "content": payload, "wc": total_words})
        session.commit()
    return JSONResponse({
        "ok": True, "name": snap_name,
        "drafts_included": len(bundle["drafts"]),
        "total_words": total_words,
    })


@app.post("/api/snapshot/book/{book_id}")
async def create_book_snapshot(book_id: str, name: str = Form("")):
    """Snapshot every draft across every chapter in a book.

    Phase 38 — the union-of-chapters safety net. Useful before running
    autowrite at the whole-book level or before a risky refactor.
    """
    import datetime as _dt
    with get_session() as session:
        book = session.execute(text(
            "SELECT id::text, title FROM books WHERE id::text = :bid"
        ), {"bid": book_id}).fetchone()
        if not book:
            raise HTTPException(404, "Book not found")
        chapters = session.execute(text(
            "SELECT id::text, number, title FROM book_chapters "
            "WHERE book_id::text = :bid ORDER BY number"
        ), {"bid": book_id}).fetchall()

        chapter_bundles = []
        grand_total = 0
        for ch in chapters:
            bundle = _snapshot_chapter_drafts(session, ch[0])
            if not bundle["drafts"]:
                continue
            bundle["chapter_number"] = ch[1]
            bundle["chapter_title"] = ch[2] or ""
            chapter_bundles.append(bundle)
            grand_total += sum(d.get("word_count") or 0 for d in bundle["drafts"])

        if not chapter_bundles:
            raise HTTPException(400, "Book has no drafts to snapshot")

        snap_name = (name or "").strip() or (
            f"{book[1]} — {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        payload = json.dumps({
            "book_id": book_id, "chapters": chapter_bundles,
        })
        session.execute(text("""
            INSERT INTO draft_snapshots
                (book_id, scope, name, content, word_count)
            VALUES
                (CAST(:bid AS uuid), 'book', :name, :content, :wc)
        """), {"bid": book_id, "name": snap_name,
               "content": payload, "wc": grand_total})
        session.commit()
    return JSONResponse({
        "ok": True, "name": snap_name,
        "chapters_included": len(chapter_bundles),
        "total_words": grand_total,
    })


@app.get("/api/snapshots/chapter/{chapter_id}")
async def list_chapter_snapshots(chapter_id: str):
    """List chapter-scope snapshots for a chapter."""
    with get_session() as session:
        rows = session.execute(text("""
            SELECT id::text, name, word_count, created_at, scope
            FROM draft_snapshots
            WHERE chapter_id::text = :cid AND scope = 'chapter'
            ORDER BY created_at DESC
        """), {"cid": chapter_id}).fetchall()
    return {"snapshots": [
        {"id": r[0], "name": r[1], "word_count": r[2] or 0,
         "created_at": str(r[3]) if r[3] else "", "scope": r[4]}
        for r in rows
    ]}


@app.get("/api/snapshots/book/{book_id}")
async def list_book_snapshots(book_id: str):
    """List book-scope snapshots for a book."""
    with get_session() as session:
        rows = session.execute(text("""
            SELECT id::text, name, word_count, created_at, scope
            FROM draft_snapshots
            WHERE book_id::text = :bid AND scope = 'book'
            ORDER BY created_at DESC
        """), {"bid": book_id}).fetchall()
    return {"snapshots": [
        {"id": r[0], "name": r[1], "word_count": r[2] or 0,
         "created_at": str(r[3]) if r[3] else "", "scope": r[4]}
        for r in rows
    ]}


def _restore_chapter_bundle(session, bundle: dict) -> int:
    """Insert a NEW draft version per section in the bundle.

    Non-destructive: the existing drafts stay put with their current
    versions; the restored bundle shows up as the newest version of
    each section (so the GUI's "latest version" resolver picks it).
    Returns the number of drafts created.
    """
    from uuid import uuid4 as _uuid4

    chapter_id = bundle.get("chapter_id")
    drafts = bundle.get("drafts") or []
    created = 0
    for d in drafts:
        section_type = d.get("section_type")
        if not section_type:
            continue
        # Determine the next version number for this (chapter_id,
        # section_type) pair so the restored row outranks the current
        # latest on the GUI's version sort.
        row = session.execute(text(
            "SELECT COALESCE(MAX(version), 0) FROM drafts "
            "WHERE chapter_id::text = :cid AND section_type = :st"
        ), {"cid": chapter_id, "st": section_type}).fetchone()
        next_ver = int((row[0] if row else 0) or 0) + 1
        # book_id: read from an existing draft in the chapter (they
        # all share one).
        bk_row = session.execute(text(
            "SELECT book_id::text FROM drafts "
            "WHERE chapter_id::text = :cid LIMIT 1"
        ), {"cid": chapter_id}).fetchone()
        book_id = bk_row[0] if bk_row else None
        sources_json = json.dumps(d.get("sources") or [])
        restore_title = d.get("title") or f"Restored {section_type}"
        session.execute(text("""
            INSERT INTO drafts
                (id, title, book_id, chapter_id, section_type, topic,
                 content, word_count, sources, version, model_used,
                 custom_metadata)
            VALUES
                (:id, :title, CAST(:bid AS uuid), CAST(:cid AS uuid),
                 :st, :topic, :content, :wc, CAST(:sources AS jsonb),
                 :ver, :model, CAST(:meta AS jsonb))
        """), {
            "id": str(_uuid4()),
            "title": restore_title,
            "bid": book_id,
            "cid": chapter_id,
            "st": section_type,
            "topic": None,
            "content": d.get("content") or "",
            "wc": int(d.get("word_count") or 0),
            "sources": sources_json,
            "ver": next_ver,
            "model": "snapshot-restore",
            "meta": json.dumps({
                "checkpoint": "restored_from_snapshot",
                "restored_from_draft_id": d.get("id"),
                "restored_from_version": d.get("version"),
            }),
        })
        created += 1
    return created


@app.post("/api/snapshot/restore-bundle/{snapshot_id}")
async def restore_snapshot_bundle(snapshot_id: str):
    """Non-destructively restore a chapter/book bundle snapshot.

    Phase 38. For each section in the bundle, inserts a new `drafts`
    row at `version = max_current_version + 1`, so the restored
    content becomes the new latest. Existing rows are untouched,
    giving the user an undo path if the restore itself was wrong.
    """
    with get_session() as session:
        row = session.execute(text(
            "SELECT id::text, scope, content, chapter_id::text, "
            "       book_id::text, name "
            "FROM draft_snapshots WHERE id::text = :sid LIMIT 1"
        ), {"sid": snapshot_id}).fetchone()
        if not row:
            raise HTTPException(404, "Snapshot not found")
        _, scope, content, ch_id, bk_id, snap_name = row
        if scope not in ("chapter", "book"):
            raise HTTPException(
                400,
                f"Snapshot scope {scope!r} is not a bundle — use the "
                f"per-draft restore endpoint instead."
            )
        try:
            payload = json.loads(content)
        except Exception as exc:
            raise HTTPException(500, f"Malformed snapshot bundle: {exc}")

        total_drafts = 0
        chapters_touched = 0
        if scope == "chapter":
            total_drafts = _restore_chapter_bundle(session, payload)
            chapters_touched = 1
        else:  # book
            for ch_bundle in payload.get("chapters") or []:
                total_drafts += _restore_chapter_bundle(session, ch_bundle)
                chapters_touched += 1
        session.commit()

    return JSONResponse({
        "ok": True, "name": snap_name, "scope": scope,
        "chapters_restored": chapters_touched,
        "drafts_created": total_drafts,
    })


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


@app.put("/api/draft/{draft_id}/chapter")
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
                cancel_evt = {"type": "cancelled"}
                _observe_event_for_stats(job_id, cancel_evt)
                loop.call_soon_threadsafe(queue.put_nowait, cancel_evt)
                break
            # Phase 32.5 — observe BEFORE enqueueing so the polled
            # stats stay ahead of (or at least in lockstep with) the
            # SSE consumer reading from the queue.
            _observe_event_for_stats(job_id, event)
            loop.call_soon_threadsafe(queue.put_nowait, event)
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        err_evt = {"type": "error", "message": str(exc)}
        _observe_event_for_stats(job_id, err_evt)
        loop.call_soon_threadsafe(queue.put_nowait, err_evt)
    finally:
        try:
            gen.close()  # raises GeneratorExit at the current yield point
        except Exception:
            pass
        # Phase 32.5 — mark stream done so a polling task bar that
        # never observed an explicit completed/error event still
        # transitions out of the streaming state on the next poll.
        with _job_lock:
            if job_id in _jobs and _jobs[job_id].get("stream_state") == "streaming":
                _jobs[job_id]["stream_state"] = "done"
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel
        # Phase 35 — book-level GPU compute ledger: persist the final
        # token/duration counters to llm_usage_log before marking the
        # job done. Runs for EVERY job type (write/review/revise/
        # argue/gaps/autowrite/plan/...) so the dashboard's Total
        # Compute panel reflects all ops, not just autowrite.
        _persist_llm_usage(job_id)
        # Phase 50.A — also flush the reasoning-steps trace onto the
        # resulting draft row (no-op if no draft_id was observed).
        _persist_reasoning_trace(job_id)
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


@app.post("/api/insert-citations/{draft_id}")
async def api_insert_citations(
    draft_id: str,
    model: str = Form(None),
    candidate_k: int = Form(8),
    max_needs: int = Form(0),
    dry_run: bool = Form(False),
):
    """Phase 46.A — auditable [N]-citation insertion over a saved draft.

    Two-pass LLM flow wrapped by ``book_ops.insert_citations_stream``: pass
    1 finds locations where a citation is needed, pass 2 retrieves top-K
    candidates via hybrid search and picks (or rejects) per claim. Events
    are streamed over the usual ``/api/stream/{job_id}`` SSE channel.
    """
    from sciknow.core.book_ops import insert_citations_stream

    job_id, queue = _create_job("insert_citations")
    loop = asyncio.get_event_loop()

    def gen():
        return insert_citations_stream(
            draft_id,
            model=(model or None),
            candidate_k=max(1, int(candidate_k)),
            max_needs=(int(max_needs) if max_needs and max_needs > 0 else None),
            dry_run=bool(dry_run),
            save=True,
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


@app.get("/api/wiki/page/{slug}/annotation")
async def api_wiki_annotation_get(slug: str):
    """Phase 54.5 — fetch the user's "My take" annotation for a
    page. Returns an empty body if none exists (no 404 — the reader
    treats absence as an empty textarea)."""
    with get_session() as session:
        row = session.execute(text("""
            SELECT body, updated_at FROM wiki_annotations WHERE slug = :s
        """), {"s": slug}).fetchone()
    if not row:
        return JSONResponse({"slug": slug, "body": "", "updated_at": None})
    return JSONResponse({
        "slug": slug, "body": row[0] or "", "updated_at": str(row[1]),
    })


@app.put("/api/wiki/page/{slug}/annotation")
async def api_wiki_annotation_put(slug: str, body: str = Form("")):
    """Phase 54.5 — upsert the annotation. Empty body deletes the row
    so the page goes back to "no annotation" cleanly."""
    body = (body or "").strip()
    with get_session() as session:
        if not body:
            session.execute(text(
                "DELETE FROM wiki_annotations WHERE slug = :s"
            ), {"s": slug})
            session.commit()
            return JSONResponse({"slug": slug, "deleted": True})
        session.execute(text("""
            INSERT INTO wiki_annotations (slug, body, updated_at)
            VALUES (:s, :b, now())
            ON CONFLICT (slug) DO UPDATE
                SET body = EXCLUDED.body, updated_at = now()
        """), {"s": slug, "b": body[:20000]})
        session.commit()
        row = session.execute(text(
            "SELECT updated_at FROM wiki_annotations WHERE slug = :s"
        ), {"s": slug}).fetchone()
    return JSONResponse({
        "slug": slug, "deleted": False,
        "updated_at": str(row[0]) if row else None,
    })


@app.post("/api/wiki/page/{slug}/ask")
async def api_wiki_page_ask(
    slug: str,
    question: str = Form(...),
    model: str = Form(None),
    context_k: int = Form(6),
    broaden: bool = Form(False),
):
    """Phase 54.3 — "Ask this page" inline RAG.

    Runs hybrid search and an LLM-streamed answer scoped to the wiki
    page's ``source_doc_ids``. That scope is the right default for
    questions a reader forms while mid-read (e.g. "What's the effect
    size in the Results section?") — nothing cross-paper slips in.
    Pass ``broaden=true`` to fall back to the full corpus if the
    scoped retrieval returned too few hits.
    """
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.storage.qdrant import get_client

    # Resolve the page's source_doc_ids so we know what to scope to.
    with get_session() as session:
        row = session.execute(text("""
            SELECT array_remove(source_doc_ids, NULL)::text[]
            FROM wiki_pages WHERE slug = :s
        """), {"s": slug}).fetchone()
    source_doc_ids: list[str] = list(row[0]) if row and row[0] else []
    source_set = {d.lower() for d in source_doc_ids}

    job_id, queue = _create_job("wiki_page_ask")
    loop = asyncio.get_event_loop()

    def gen():
        qdrant = get_client()
        yield {"type": "progress", "stage": "retrieving",
               "detail": (
                   f"Searching this page's {len(source_doc_ids)} source paper(s)..."
                   if source_doc_ids and not broaden
                   else "Searching the whole corpus..."
               )}

        # Scoped retrieval: over-fetch, then keep only chunks whose
        # document_id is in the page's source set. If the page has no
        # source_doc_ids OR `broaden` is explicitly set, skip the
        # filter and run a normal corpus-wide search.
        over_k = 200 if (source_set and not broaden) else 50
        with get_session() as session:
            candidates = hybrid_search.search(
                query=question, qdrant_client=qdrant, session=session,
                candidate_k=over_k,
            )
            if source_set and not broaden:
                candidates = [
                    c for c in candidates
                    if (c.document_id or "").lower() in source_set
                ]
            if not candidates:
                if source_set and not broaden:
                    yield {
                        "type": "error",
                        "message": (
                            "No matching passages in this page's source papers. "
                            "Try the 'broaden to full corpus' toggle."
                        ),
                    }
                else:
                    yield {"type": "error",
                           "message": "No relevant passages found."}
                return
            candidates = reranker.rerank(
                question, candidates, top_k=max(1, min(int(context_k or 6), 12)),
            )
            results = context_builder.build(candidates, session)

        from sciknow.rag.prompts import format_sources
        sources_lines = format_sources(results).splitlines()
        yield {"type": "sources", "sources": sources_lines,
               "n": len(sources_lines),
               "scope": "this-page" if (source_set and not broaden) else "corpus"}

        system, user = rag_prompts.qa(question, results)
        yield {"type": "progress", "stage": "generating",
               "detail": f"Generating answer from {len(results)} passage(s)..."}
        for tok in llm_stream(system, user, model=model or None):
            yield {"type": "token", "text": tok}
        yield {"type": "completed"}

    threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id, "scope_size": len(source_doc_ids)})


@app.get("/api/wiki/page/{slug}/backlinks")
async def api_wiki_backlinks(slug: str):
    """Phase 54.2 — pages that link to this one via ``[[slug]]``.

    Backed by an in-process cache built lazily from a walk of every
    markdown page on disk. Rebuild is O(N_pages × page_size) and
    runs at most once every 10 minutes without an explicit
    invalidate call."""
    from sciknow.core.wiki_ops import get_backlinks_for
    try:
        return JSONResponse(get_backlinks_for(slug))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/wiki/page/{slug}/related")
async def api_wiki_related(slug: str, limit: int = 5):
    """Phase 54.2 — top-N pages nearest in the WIKI_COLLECTION
    embedding space, excluding the source. Uses the same bge-m3
    vectors that power wiki search."""
    from sciknow.core.wiki_ops import get_related_pages
    limit = max(1, min(int(limit or 5), 20))
    try:
        return JSONResponse(get_related_pages(slug, limit=limit))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/wiki/titles")
async def api_wiki_titles():
    """Phase 54 — compact title/slug index for the Ctrl-K command palette.

    Returns every wiki page's (slug, title, page_type) in one shot so
    the client can do fuzzy-filter locally without round-tripping. The
    payload is bounded by the wiki size (typically ≤ a few hundred
    pages) so sending it whole is fine."""
    with get_session() as session:
        rows = session.execute(text("""
            SELECT slug, title, page_type
            FROM wiki_pages
            ORDER BY title
        """)).fetchall()
    return JSONResponse([
        {"slug": r[0], "title": r[1] or r[0], "page_type": r[2]}
        for r in rows
    ])


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
                       updated_at, needs_rewrite
                FROM wiki_pages WHERE slug = :slug
            """), {"slug": slug}).fetchone()
        if row:
            page.update({
                "title": row[0], "page_type": row[1], "word_count": row[2] or 0,
                "n_sources": row[3] or 0, "updated_at": str(row[4]),
                # Phase 54.1 — surface the staleness flag so the reader
                # can render a banner on pages that are older than their
                # source and would benefit from `wiki compile --rewrite-stale`.
                "needs_rewrite": (str(row[5]).lower() == "true") if row[5] else False,
            })
    except Exception:
        pass

    # Phase 54.4 — for concept pages, include the knowledge_graph
    # triples whose subject or object matches this concept (by title
    # or slug, lowercased). Turns stub concept pages into useful
    # "Facts from the corpus" views without waiting for
    # `wiki compile --rewrite-stale` to fill out the prose.
    if page.get("page_type") == "concept":
        try:
            title_lower = (page.get("title") or slug).lower()
            slug_spaced = slug.replace("-", " ").lower()
            with get_session() as session:
                tri_rows = session.execute(text("""
                    SELECT kg.subject, kg.predicate, kg.object,
                           kg.source_doc_id::text, kg.confidence,
                           pm.title, kg.source_sentence
                    FROM knowledge_graph kg
                    LEFT JOIN paper_metadata pm ON pm.document_id = kg.source_doc_id
                    WHERE LOWER(kg.subject) IN (:t, :s)
                       OR LOWER(kg.object)  IN (:t, :s)
                    ORDER BY kg.confidence DESC, kg.subject
                    LIMIT 50
                """), {"t": title_lower, "s": slug_spaced}).fetchall()
            page["related_triples"] = [
                {
                    "subject": r[0], "predicate": r[1], "object": r[2],
                    "source_doc_id": r[3],
                    "confidence": float(r[4] or 1.0),
                    "source_title": r[5],
                    "source_sentence": r[6],
                }
                for r in tri_rows
            ]
        except Exception:
            page["related_triples"] = []

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


@app.post("/api/wiki/lint")
async def api_wiki_lint(deep: bool = Form(False), model: str = Form(None)):
    """Phase 54.6.2 — stream `sciknow wiki lint` over SSE.

    Yields lint_issue events per problem found, plus a final completed
    event with the aggregate count. deep=True adds LLM-based
    cross-paper contradiction detection (slow).
    """
    from sciknow.core.wiki_ops import lint_wiki

    job_id, queue = _create_job("wiki_lint")
    loop = asyncio.get_event_loop()

    def gen():
        return lint_wiki(deep=bool(deep), model=(model or None))

    threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


@app.post("/api/wiki/consensus")
async def api_wiki_consensus(topic: str = Form(...), model: str = Form(None)):
    """Phase 54.6.2 — stream `sciknow wiki consensus` over SSE.

    Emits a single consensus event with the structured claims dict, then
    a completed event. The generator also persists a synthesis wiki page.
    """
    if not topic.strip():
        raise HTTPException(status_code=400, detail="topic required")
    from sciknow.core.wiki_ops import consensus_map

    job_id, queue = _create_job("wiki_consensus")
    loop = asyncio.get_event_loop()

    def gen():
        return consensus_map(topic.strip(), model=(model or None))

    threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


# ── Phase 50.B — user feedback capture (LambdaMART feedstock) ────────────

@app.post("/api/feedback")
async def api_feedback(
    op: str = Form("ask"),
    score: int = Form(...),
    query: str = Form(""),
    preview: str = Form(""),
    comment: str = Form(""),
    draft_id: str = Form(""),
    chunk_ids: str = Form(""),
):
    """Record one thumbs-up / thumbs-down row.

    Called from a 👍/👎 button next to any generated answer in the
    reader. Fields are permissive — only `op` and `score` are required
    structurally; everything else is optional metadata the eventual
    LambdaMART trainer will project.

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


@app.get("/api/feedback/stats")
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
    """Paginated paper list with optional filters. Mirrors `sciknow catalog list`.

    Phase 41 — query is fully static. Every optional filter is
    always bound (as the real value or NULL) and gated by a
    ``(:param IS NULL OR …)`` short-circuit. This removes the
    f-string interpolation of pre-built WHERE fragments that the
    Phase 22 audit flagged: no code path builds SQL from Python
    strings anymore, so a future maintainer can't accidentally
    concatenate user input into the query shape.
    """
    page = max(page, 1)
    per_page = min(max(per_page, 1), 100)
    offset = (page - 1) * per_page

    params: dict = {
        "limit": per_page,
        "offset": offset,
        "year_from": year_from,
        "year_to": year_to,
        # Both ILIKE filters are wrapped with %% here so the SQL can
        # stay static. NULL means "no filter" courtesy of the gating
        # IS NULL check on each clause.
        "author": f"%{author}%" if author else None,
        "journal": f"%{journal}%" if journal else None,
        "topic_cluster": topic_cluster or None,
    }

    with get_session() as session:
        total = session.execute(text("""
            SELECT COUNT(*) FROM paper_metadata pm
            WHERE (:year_from IS NULL OR pm.year >= :year_from)
              AND (:year_to   IS NULL OR pm.year <= :year_to)
              AND (:author    IS NULL OR EXISTS (
                   SELECT 1 FROM jsonb_array_elements(pm.authors) a
                   WHERE a->>'name' ILIKE :author))
              AND (:journal        IS NULL OR pm.journal ILIKE :journal)
              AND (:topic_cluster  IS NULL OR pm.topic_cluster = :topic_cluster)
        """), params).scalar() or 0

        rows = session.execute(text("""
            SELECT pm.document_id::text, pm.title, pm.year, pm.authors,
                   pm.journal, pm.doi, pm.abstract, pm.topic_cluster,
                   pm.metadata_source
            FROM paper_metadata pm
            WHERE (:year_from IS NULL OR pm.year >= :year_from)
              AND (:year_to   IS NULL OR pm.year <= :year_to)
              AND (:author    IS NULL OR EXISTS (
                   SELECT 1 FROM jsonb_array_elements(pm.authors) a
                   WHERE a->>'name' ILIKE :author))
              AND (:journal        IS NULL OR pm.journal ILIKE :journal)
              AND (:topic_cluster  IS NULL OR pm.topic_cluster = :topic_cluster)
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


# ── Phase 36: Tools panel endpoints ──────────────────────────────────────────
# Brings four CLI-only capabilities into the web GUI:
#   search query / search similar  → JSON
#   ask synthesize                 → SSE (mirrors /api/ask which is the
#                                        `ask question` equivalent)
#   catalog topics                 → JSON (topic cluster breakdown)
#   db enrich / db expand          → SSE, subprocess-backed so the CLI
#                                    stays the source of truth for these
#                                    long-running ops with complex flag
#                                    surfaces (workers, resolvers, …).


@app.post("/api/search/query")
async def api_search_query(
    q: str = Form(...),
    top_k: int = Form(10),
    candidate_k: int = Form(50),
    no_rerank: bool = Form(False),
    year_from: int = Form(None),
    year_to: int = Form(None),
    section: str = Form(None),
    topic: str = Form(None),
    expand: bool = Form(False),
):
    """Hybrid corpus search (sciknow search query). JSON response.

    Quick-enough to return a JSON response rather than SSE: the search
    fan-out is one Qdrant dense + one sparse + one PG FTS query, fused,
    then a cross-encoder rerank over ~50 candidates. Typical <2s on the
    reference hardware.
    """
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.storage.qdrant import get_client

    qdrant = get_client()
    with get_session() as session:
        candidates = hybrid_search.search(
            query=q, qdrant_client=qdrant, session=session,
            candidate_k=candidate_k,
            year_from=year_from, year_to=year_to,
            section=section, topic_cluster=topic,
            use_query_expansion=expand,
        )
        if not candidates:
            return JSONResponse({"results": [], "n": 0})
        if not no_rerank:
            candidates = reranker.rerank(q, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]
        results = context_builder.build(candidates, session)

    import re as _re
    out = []
    for r in results:
        preview = _re.sub(r"<[^>]+>", "", r.content or "")
        preview = _re.sub(r"\s+", " ", preview).strip()[:300]
        out.append({
            "rank": r.rank,
            "title": r.title or "(untitled)",
            "year": r.year,
            "authors": r.authors or [],
            "journal": r.journal,
            "doi": r.doi,
            "section_type": r.section_type,
            "section_title": r.section_title,
            "score": r.score,
            "preview": preview,
            "document_id": str(r.document_id) if r.document_id else None,
        })
    return JSONResponse({"results": out, "n": len(out)})


@app.post("/api/search/similar")
async def api_search_similar(
    identifier: str = Form(...),
    top_k: int = Form(10),
):
    """Nearest-neighbour paper search in the abstracts collection.

    Mirrors `sciknow search similar`. Accepts a DOI, arXiv ID, title
    fragment, or document UUID; resolves it to a document_id, pulls its
    abstract embedding from Qdrant, and returns the top_k nearest
    neighbours (excluding the query paper itself).
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from sciknow.storage.qdrant import ABSTRACTS_COLLECTION, get_client

    qdrant = get_client()
    ident = (identifier or "").strip()
    if not ident:
        raise HTTPException(400, "identifier required")

    with get_session() as session:
        row = session.execute(text(
            "SELECT d.id::text, pm.title, pm.doi, pm.year "
            "FROM paper_metadata pm JOIN documents d ON d.id = pm.document_id "
            "WHERE LOWER(pm.doi) = LOWER(:q) OR LOWER(pm.arxiv_id) = LOWER(:q) "
            "LIMIT 1"
        ), {"q": ident}).first()
        if not row:
            row = session.execute(text(
                "SELECT d.id::text, pm.title, pm.doi, pm.year "
                "FROM paper_metadata pm JOIN documents d ON d.id = pm.document_id "
                "WHERE pm.title ILIKE :pattern "
                "ORDER BY pm.year DESC NULLS LAST LIMIT 1"
            ), {"pattern": f"%{ident}%"}).first()
        if not row:
            try:
                row = session.execute(text(
                    "SELECT d.id::text, pm.title, pm.doi, pm.year "
                    "FROM paper_metadata pm JOIN documents d ON d.id = pm.document_id "
                    "WHERE d.id::text = :q LIMIT 1"
                ), {"q": ident}).first()
            except Exception:
                row = None

    if not row:
        return JSONResponse({"error": "Paper not found", "query": ident}, status_code=404)

    doc_id, title, doi, year = row
    abstract_points = qdrant.scroll(
        collection_name=ABSTRACTS_COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=doc_id))
        ]),
        with_vectors=["dense"],
        limit=1,
    )[0]
    if not abstract_points:
        return JSONResponse({
            "error": "No abstract embedding for this paper (was it ingested?)",
            "query": ident, "document_id": doc_id, "title": title,
        }, status_code=404)

    query_vec = abstract_points[0].vector
    if isinstance(query_vec, dict):
        query_vec = query_vec.get("dense")

    hits = qdrant.query_points(
        collection_name=ABSTRACTS_COLLECTION,
        query=query_vec, using="dense",
        limit=top_k + 1, with_payload=True,
    )

    results = []
    for point in hits.points:
        payload = point.payload or {}
        if payload.get("document_id") == doc_id:
            continue  # exclude the query paper
        results.append({
            "title": payload.get("title") or (payload.get("content_preview") or "")[:80],
            "year": payload.get("year"),
            "authors": payload.get("authors") or [],
            "document_id": payload.get("document_id"),
            "score": float(point.score) if point.score is not None else None,
        })
        if len(results) >= top_k:
            break

    return JSONResponse({
        "query": {"title": title, "year": year, "doi": doi, "document_id": doc_id},
        "results": results, "n": len(results),
    })


@app.post("/api/ask/synthesize")
async def api_ask_synthesize(
    topic: str = Form(...),
    model: str = Form(None),
    context_k: int = Form(12),
    year_from: int = Form(None),
    year_to: int = Form(None),
    domain: str = Form(None),
    topic_filter: str = Form(None),
):
    """Multi-paper synthesis on a topic (sciknow ask synthesize). SSE.

    Distinct from /api/ask which runs `ask question` (Q&A). Synthesize
    biases the prompt toward consensus/method/open-question framing.
    """
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.storage.qdrant import get_client

    job_id, _queue = _create_job("synthesize")
    loop = asyncio.get_event_loop()

    def gen():
        qdrant = get_client()
        yield {"type": "progress", "stage": "retrieving",
               "detail": f"Retrieving passages for: {topic}..."}
        with get_session() as session:
            candidates = hybrid_search.search(
                query=topic, qdrant_client=qdrant, session=session,
                candidate_k=50,
                year_from=year_from if year_from else None,
                year_to=year_to if year_to else None,
                domain=domain or None,
                topic_cluster=topic_filter or None,
            )
            if not candidates:
                yield {"type": "error", "message": "No relevant passages found."}
                return
            candidates = reranker.rerank(topic, candidates, top_k=context_k)
            results = context_builder.build(candidates, session)

        from sciknow.rag.prompts import format_sources
        sources = format_sources(results).splitlines()
        yield {"type": "sources", "sources": sources, "n": len(sources)}

        system, user = rag_prompts.synthesis(topic, results)
        yield {"type": "progress", "stage": "generating",
               "detail": f"Synthesising from {len(results)} passages..."}
        for tok in llm_stream(system, user, model=model or None):
            yield {"type": "token", "text": tok}
        yield {"type": "completed"}

    threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


# ── Phase 46.E — Authors + domains + expand-author (web surface) ─────────────
#
# Makes the "grow the corpus" capability reachable from the browser:
#   GET /api/catalog/authors       — ranked + searchable authors
#   GET /api/catalog/domains       — paper_metadata.domains unnested + ranked
#   POST /api/corpus/expand-author — run `sciknow db expand-author` as an
#                                    SSE-streamed subprocess (same pattern as
#                                    the existing /api/corpus/expand)
#
# Ranking rationale: the user selecting an author for expansion wants to
# see their most-cited / most-authored-in-this-corpus candidates first,
# so the default sort key is (citation_count DESC, paper_count DESC).
# An optional ?q=fragment filters by substring on the author name — PG's
# ILIKE on an already-GROUP-BYed name list is fast enough at our scale
# (~5k distinct authors in the global-cooling project); if the author
# count ever exceeds ~50k we'd promote this to a materialized view.


@app.get("/api/catalog/authors")
async def api_catalog_authors(
    q: str = "",
    limit: int = 50,
    min_papers: int = 1,
):
    """Phase 46.E — ranked + searchable author index.

    Returns authors from ``paper_metadata.authors`` unnested and grouped
    by name, ranked by ``(citation_count DESC, paper_count DESC)`` so
    the most-cited / most-prolific names surface first. Used by the
    web UI's Expand-by-Author picker.

    Params:
      q          — substring match (case-insensitive) on the author name;
                   empty string returns the top ``limit`` overall.
      limit      — cap on rows returned (default 50).
      min_papers — require at least this many papers in the corpus
                   (default 1 — every author).
    """
    limit = max(1, min(500, int(limit or 50)))
    min_papers = max(1, int(min_papers or 1))
    q_like = f"%{q.strip()}%" if q and q.strip() else None
    with get_session() as session:
        # CTE approach — CROSS JOIN LATERAL is the correct PG dance for
        # jsonb_array_elements inside a join that also hits citations.
        # Using DISTINCT on (document_id) within the explosion to
        # deduplicate the many-to-one from authors → document.
        base_sql = """
            WITH exploded AS (
                SELECT author->>'name' AS name,
                       COALESCE(author->>'orcid', '') AS orcid,
                       pm.document_id
                FROM paper_metadata pm
                CROSS JOIN LATERAL jsonb_array_elements(pm.authors) AS author
                WHERE pm.authors IS NOT NULL
                  AND author->>'name' IS NOT NULL
                  AND trim(author->>'name') != ''
            )
            SELECT e.name,
                   COUNT(DISTINCT e.document_id) AS n_papers,
                   COUNT(c.id)                   AS n_cites,
                   (ARRAY_AGG(DISTINCT NULLIF(e.orcid, '')))[1] AS first_orcid
            FROM exploded e
            LEFT JOIN citations c ON c.cited_document_id = e.document_id
            {where}
            GROUP BY e.name
            HAVING COUNT(DISTINCT e.document_id) >= :min_papers
            ORDER BY n_cites DESC, n_papers DESC, e.name
            LIMIT :limit
        """
        params = {"min_papers": min_papers, "limit": limit}
        if q_like:
            where = "WHERE e.name ILIKE :q"
            params["q"] = q_like
        else:
            where = ""
        rows = session.execute(
            text(base_sql.format(where=where)), params,
        ).fetchall()

    return JSONResponse({
        "authors": [
            {"name": r[0], "n_papers": int(r[1] or 0),
             "n_citations": int(r[2] or 0), "orcid": r[3] or None}
            for r in rows
        ],
        "query": q,
        "limit": limit,
    })


@app.get("/api/catalog/domains")
async def api_catalog_domains(limit: int = 60):
    """Phase 46.E — ranked domain / tag index (paper_metadata.domains unnested).

    ``domains`` is a text[] column populated by the metadata cascade —
    Crossref's subject list or the arXiv primary-category field. Empty
    arrays are common (our corpus ingested mostly via embedded_pdf /
    Crossref for works without subject tags), but this endpoint still
    ranks what's present so the UI can expose tag-filtering.
    """
    limit = max(1, min(500, int(limit or 60)))
    with get_session() as session:
        rows = session.execute(text("""
            SELECT tag, COUNT(DISTINCT pm.document_id) AS n
            FROM paper_metadata pm
            CROSS JOIN LATERAL unnest(pm.domains) AS tag
            WHERE pm.domains IS NOT NULL
              AND array_length(pm.domains, 1) > 0
              AND trim(tag) != ''
            GROUP BY tag
            ORDER BY n DESC, tag
            LIMIT :lim
        """), {"lim": limit}).fetchall()
    return JSONResponse({
        "domains": [{"name": r[0], "n": int(r[1])} for r in rows],
    })


@app.get("/api/catalog/topics")
async def api_catalog_topics(name: str = None):
    """Topic cluster breakdown (sciknow catalog topics).

    With no args: returns every non-null cluster name + paper count.
    With ?name=...: returns the cluster's paper list (title/year/doi).
    """
    with get_session() as session:
        if name:
            rows = session.execute(text("""
                SELECT pm.document_id::text, pm.title, pm.year, pm.doi,
                       pm.authors, pm.journal
                FROM paper_metadata pm
                WHERE pm.topic_cluster = :n
                ORDER BY pm.year DESC NULLS LAST, pm.title
                LIMIT 500
            """), {"n": name}).fetchall()
            papers = [
                {"document_id": r[0], "title": r[1], "year": r[2],
                 "doi": r[3], "authors": r[4] or [], "journal": r[5]}
                for r in rows
            ]
            return JSONResponse({"name": name, "papers": papers, "n": len(papers)})

        rows = session.execute(text("""
            SELECT topic_cluster, COUNT(*)
            FROM paper_metadata
            WHERE topic_cluster IS NOT NULL AND topic_cluster != ''
            GROUP BY topic_cluster
            ORDER BY COUNT(*) DESC
        """)).fetchall()
    return JSONResponse({
        "topics": [{"name": r[0], "n": int(r[1])} for r in rows],
    })


# ── Corpus actions — subprocess-backed, stream stdout as SSE ─────────────────

import os  # noqa: E402 — kept local-ish with the block that uses it
import shlex  # noqa: E402
import subprocess  # noqa: E402


def _sciknow_cli_bin() -> str:
    """Absolute path to the CLI entry point in the active venv.

    Falls back to plain `sciknow` on PATH if the venv binary isn't found
    (e.g. when running the test suite from a different Python).
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / ".venv" / "bin" / "sciknow"
    return str(candidate) if candidate.exists() else "sciknow"


def _spawn_cli_streaming(job_id: str, argv: list[str], loop, on_finish=None):
    """Run `sciknow <argv...>` as a subprocess and forward its output as
    SSE events. Each stdout line becomes `{type: "log", text: ...}`.

    The subprocess is terminated on job cancel. stderr is merged into
    stdout so Rich progress bars and error messages all surface in the
    same log pane.

    ``on_finish`` (optional) runs once the process has exited, regardless
    of success/failure/cancel. Useful for cleaning up tempfiles tied to
    the job (see ``/api/corpus/expand-author/download-selected``).
    """
    import shutil  # lazy — only used when cancelling

    def gen():
        cmd = [_sciknow_cli_bin(), *argv]
        # Force Rich to plain-text and disable progress bars' fancy
        # rendering so the web log pane doesn't fill up with ANSI noise.
        env = os.environ.copy()
        env.setdefault("NO_COLOR", "1")
        env.setdefault("TERM", "dumb")
        yield {"type": "progress", "stage": "starting",
               "detail": "$ " + " ".join(shlex.quote(c) for c in cmd)}
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
        except FileNotFoundError as exc:
            yield {"type": "error", "message": f"CLI not found: {exc}"}
            return

        cancel = _jobs[job_id]["cancel"]
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if cancel.is_set():
                    break
                line = line.rstrip("\n")
                if line:
                    yield {"type": "log", "text": line}
            proc.wait(timeout=5)
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
            return
        finally:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        rc = proc.returncode or 0
        if rc == 0:
            yield {"type": "completed", "returncode": 0}
        else:
            yield {"type": "error",
                   "message": f"CLI exited with code {rc}"}
        if on_finish is not None:
            try:
                on_finish()
            except Exception as exc:
                logger.warning("_spawn_cli_streaming on_finish failed: %s", exc)
        _ = shutil  # silence lint for the lazy import

    threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()


@app.post("/api/corpus/enrich")
async def api_corpus_enrich(
    dry_run: bool = Form(False),
    threshold: float = Form(0.85),
    limit: int = Form(0),
    delay: float = Form(0.2),
):
    """Invoke `sciknow db enrich` from the web UI — SSE log stream."""
    job_id, _queue = _create_job("corpus_enrich")
    loop = asyncio.get_event_loop()
    argv = ["db", "enrich",
            "--threshold", str(threshold),
            "--limit", str(limit),
            "--delay", str(delay)]
    if dry_run:
        argv.append("--dry-run")
    _spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


# ── Phase 46.F — End-to-end web flow (ingest / indices / book create) ────────
#
# Wires every CLI step from "empty project" to "book export" to a web
# endpoint so the Setup Wizard modal can walk a new user through the
# entire pipeline from the browser. All long-running ops use the same
# SSE-streamed-subprocess pattern as /api/corpus/{enrich,expand}.
#
# Endpoints added here:
#   POST /api/corpus/ingest-directory   stream `sciknow ingest directory <path>`
#   POST /api/corpus/upload             multipart upload → stage → ingest
#   POST /api/catalog/cluster           stream `sciknow catalog cluster`
#   POST /api/catalog/raptor/build      stream `sciknow catalog raptor build`
#   POST /api/wiki/compile              stream `sciknow wiki compile`
#   POST /api/book/create               create a book + optional bootstrap
#                                       (inline, no subprocess — fast)
#   GET  /api/setup/status              aggregate "where am I in the pipeline?"


@app.post("/api/corpus/ingest-directory")
async def api_corpus_ingest_directory(
    path: str = Form(...),
    recursive: bool = Form(True),
    force: bool = Form(False),
    workers: int = Form(0),
):
    """SSE-streamed wrapper around ``sciknow ingest directory <path>``.

    Phase 46.F — path is a server-side directory. The wizard UI
    usually pairs this with an ``ingest/upload`` step that stages
    uploaded files into ``{data_dir}/inbox/`` first.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {p}")
    job_id, _ = _create_job("corpus_ingest_directory")
    loop = asyncio.get_event_loop()
    argv: list[str] = ["ingest", "directory", str(p)]
    if not recursive:
        argv.append("--no-recursive")
    if force:
        argv.append("--force")
    if workers and int(workers) > 0:
        argv += ["--workers", str(int(workers))]
    _spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id, "path": str(p)})


@app.post("/api/corpus/upload")
async def api_corpus_upload(request: Request):
    """Accept a multipart PDF upload + (optionally) queue an ingest job.

    Files are saved to ``{data_dir}/inbox/uploads_<ts>/`` inside the
    active project so multi-project isolation holds. If ``start_ingest``
    is truthy, an ingest job is spawned automatically and its job_id
    is returned; otherwise the client can trigger
    ``POST /api/corpus/ingest-directory`` itself with the returned
    ``staging_dir``.
    """
    from datetime import datetime, timezone
    from sciknow.config import settings

    form = await request.form()
    files = form.getlist("files")
    start_ingest = form.get("start_ingest", "false").lower() in {"1", "true", "yes"}
    force        = form.get("force",        "false").lower() in {"1", "true", "yes"}
    recursive    = form.get("recursive",    "true").lower()  in {"1", "true", "yes"}
    if not files:
        raise HTTPException(status_code=400, detail="no files in upload")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = Path(settings.data_dir) / "inbox" / f"uploads_{ts}"
    staging.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for f in files:
        if not hasattr(f, "filename") or not f.filename:
            continue
        # Strip path components — we only accept a flat filename.
        name = Path(f.filename).name
        if not name.lower().endswith(".pdf"):
            # Skip non-PDFs silently; ingestion would fail anyway.
            continue
        dest = staging / name
        # Avoid collisions within this batch.
        i = 0
        while dest.exists():
            i += 1
            dest = staging / f"{dest.stem}_{i}{dest.suffix}"
        contents = await f.read()
        dest.write_bytes(contents)
        saved.append(dest.name)

    if not saved:
        # Clean up empty staging dir
        try: staging.rmdir()
        except Exception: pass
        raise HTTPException(status_code=400,
                             detail="no .pdf files in upload (only PDFs accepted)")

    payload: dict = {
        "staging_dir": str(staging),
        "n_files": len(saved),
        "files": saved,
    }
    if start_ingest:
        job_id, _ = _create_job("corpus_ingest_upload")
        loop = asyncio.get_event_loop()
        argv = ["ingest", "directory", str(staging)]
        if not recursive: argv.append("--no-recursive")
        if force:         argv.append("--force")
        _spawn_cli_streaming(job_id, argv, loop)
        payload["job_id"] = job_id
    return JSONResponse(payload)


@app.post("/api/catalog/cluster")
async def api_catalog_cluster(
    min_cluster_size: int = Form(0),
    rebuild: bool = Form(False),
    dry_run: bool = Form(False),
):
    """SSE-streamed wrapper around ``sciknow catalog cluster``."""
    job_id, _ = _create_job("catalog_cluster")
    loop = asyncio.get_event_loop()
    argv = ["catalog", "cluster"]
    if min_cluster_size and int(min_cluster_size) > 0:
        argv += ["--min-cluster-size", str(int(min_cluster_size))]
    if rebuild: argv.append("--rebuild")
    if dry_run: argv.append("--dry-run")
    _spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


@app.post("/api/catalog/raptor/build")
async def api_catalog_raptor_build():
    """SSE-streamed wrapper around ``sciknow catalog raptor build``.

    RAPTOR is a one-off batch op (typically 5-30 min depending on
    corpus size). No options exposed — build policy uses the CLI
    defaults.
    """
    job_id, _ = _create_job("catalog_raptor_build")
    loop = asyncio.get_event_loop()
    _spawn_cli_streaming(job_id, ["catalog", "raptor", "build"], loop)
    return JSONResponse({"job_id": job_id})


@app.post("/api/wiki/compile")
async def api_wiki_compile(
    rebuild: bool = Form(False),
    rewrite_stale: bool = Form(False),
    doc_id: str = Form(""),
):
    """SSE-streamed wrapper around ``sciknow wiki compile``."""
    job_id, _ = _create_job("wiki_compile")
    loop = asyncio.get_event_loop()
    argv = ["wiki", "compile"]
    if rebuild:        argv.append("--rebuild")
    if rewrite_stale:  argv.append("--rewrite-stale")
    if doc_id.strip(): argv += ["--doc-id", doc_id.strip()]
    _spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


@app.post("/api/book/create")
async def api_book_create(request: Request):
    """Phase 46.F — web-side book creation with type selection.

    Runs inline (no subprocess) because book creation is fast (~50 ms)
    and doesn't need streaming progress. Mirrors ``sciknow book create
    --type=<slug>`` including the flat-type bootstrap that auto-creates
    chapter 1 with canonical sections for ``scientific_paper`` and
    other ``is_flat=True`` types.
    """
    import json as _json
    from sciknow.core.project_type import (
        default_sections_as_dicts, get_project_type, validate_type_slug,
    )

    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else dict(await request.form())
    title = (body.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    btype = (body.get("type") or "scientific_book").strip()
    try:
        validate_type_slug(btype)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    description = (body.get("description") or "").strip() or None
    try:
        tcw = int(body.get("target_chapter_words") or 0)
    except (TypeError, ValueError):
        tcw = 0
    bootstrap = (str(body.get("bootstrap", "true")).lower() in {"1", "true", "yes"})

    pt = get_project_type(btype)
    custom_meta: dict = {}
    effective_target = tcw if tcw > 0 else pt.default_target_chapter_words
    custom_meta["target_chapter_words"] = effective_target

    with get_session() as session:
        existing = session.execute(text(
            "SELECT id::text FROM books WHERE title = :t"
        ), {"t": title}).fetchone()
        if existing:
            raise HTTPException(status_code=409,
                                 detail=f"book already exists: {title}")
        row = session.execute(text("""
            INSERT INTO books (title, description, book_type, custom_metadata)
            VALUES (:t, :d, :bt, CAST(:m AS jsonb))
            RETURNING id::text
        """), {
            "t": title, "d": description, "bt": pt.slug,
            "m": _json.dumps(custom_meta),
        })
        book_id = row.fetchone()[0]

        chapter_id: str | None = None
        if bootstrap and pt.is_flat:
            sections_json = _json.dumps(default_sections_as_dicts(pt))
            cid_row = session.execute(text("""
                INSERT INTO book_chapters
                  (book_id, number, title, description, sections)
                VALUES
                  (CAST(:book_id AS uuid), 1, :ch_title, :ch_desc,
                   CAST(:sections AS jsonb))
                RETURNING id::text
            """), {
                "book_id": book_id,
                "ch_title": title,
                "ch_desc": description or f"{pt.display_name} — {title}",
                "sections": sections_json,
            })
            chapter_id = cid_row.fetchone()[0]
        session.commit()

    return JSONResponse({
        "ok": True,
        "book_id": book_id,
        "title": title,
        "book_type": pt.slug,
        "display_name": pt.display_name,
        "is_flat": pt.is_flat,
        "chapter_id_bootstrapped": chapter_id,
        "default_sections": [s.key for s in pt.default_sections],
    })


@app.get("/api/setup/status")
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


@app.post("/api/corpus/expand-author/preview")
async def api_corpus_expand_author_preview(
    name: str = Form(""),
    orcid: str = Form(""),
    year_from: int = Form(0),
    year_to: int = Form(0),
    limit: int = Form(0),
    strict_author: bool = Form(True),
    all_matches: bool = Form(False),
    relevance_query: str = Form(""),
):
    """Phase 54.6.1 — preview candidates without downloading.

    Runs search + corpus-dedup + relevance scoring, returns JSON. The UI
    renders a checkboxed list so the user can cherry-pick which DOIs to
    download via ``/api/corpus/expand-author/download-selected`` — the
    existing ``POST /api/corpus/expand-author`` still exists for the
    "auto-download by relevance threshold" override path.

    May take 10-30s due to external API calls. Blocking — not SSE.
    """
    if not name.strip() and not orcid.strip():
        raise HTTPException(
            status_code=400,
            detail="provide either author name or ORCID",
        )
    from sciknow.core.expand_ops import find_author_candidates
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: find_author_candidates(
                name=name,
                orcid=(orcid.strip() or None),
                year_from=(year_from or None),
                year_to=(year_to or None),
                limit=limit,
                all_matches=all_matches,
                strict_author=strict_author,
                relevance_query=relevance_query,
                score_relevance=True,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("expand-author preview failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return JSONResponse(result)


@app.post("/api/corpus/expand-author")
async def api_corpus_expand_author(
    name: str = Form(...),
    orcid: str = Form(""),
    year_from: int = Form(0),
    year_to: int = Form(0),
    limit: int = Form(0),
    strict_author: bool = Form(True),
    all_matches: bool = Form(False),
    relevance: bool = Form(True),
    relevance_threshold: float = Form(0.0),
    relevance_query: str = Form(""),
    workers: int = Form(0),
    ingest: bool = Form(True),
    dry_run: bool = Form(False),
):
    """Phase 46.E — invoke ``sciknow db expand-author`` from the web UI.

    Runs as a background subprocess; stdout streams as SSE ``log`` events.
    """
    if not name.strip():
        raise HTTPException(status_code=400, detail="author name required")
    job_id, _queue = _create_job("corpus_expand_author")
    loop = asyncio.get_event_loop()
    argv: list[str] = ["db", "expand-author", name.strip()]
    if orcid.strip():
        argv += ["--orcid", orcid.strip()]
    if year_from:
        argv += ["--from", str(year_from)]
    if year_to:
        argv += ["--to", str(year_to)]
    if limit and int(limit) > 0:
        argv += ["--limit", str(int(limit))]
    argv += ["--workers", str(int(workers or 0))]
    argv += ["--relevance-threshold", str(float(relevance_threshold or 0.0))]
    if strict_author:   argv.append("--strict-author")
    else:               argv.append("--no-strict-author")
    if all_matches:     argv.append("--all-matches")
    argv.append("--relevance" if relevance else "--no-relevance")
    if relevance_query.strip():
        argv += ["--relevance-query", relevance_query.strip()]
    argv.append("--ingest" if ingest else "--no-ingest")
    if dry_run:
        argv.append("--dry-run")
    _spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


@app.post("/api/corpus/expand-author/download-selected")
async def api_corpus_expand_author_download_selected(request: Request):
    """Phase 54.6.1 — download + ingest the user-chosen subset from the
    Expand-by-Author preview modal.

    Body: JSON ``{"candidates": [{"doi": "...", "title": "...",
    "year": 2020}, ...], "workers": int, "ingest": bool}``.

    Spawns ``sciknow db download-dois --dois-file <tmp.json>`` and streams
    stdout as SSE. Tmp file is cleaned up when the job finishes.
    """
    import json as _json
    import tempfile

    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    raw_cands = body.get("candidates") or []
    if not isinstance(raw_cands, list) or not raw_cands:
        raise HTTPException(status_code=400, detail="candidates list required")

    # Sanitize to {doi, title, year}, drop entries missing DOI.
    clean: list[dict] = []
    for c in raw_cands:
        if not isinstance(c, dict):
            continue
        doi = (c.get("doi") or "").strip()
        if not doi:
            continue
        clean.append({
            "doi": doi,
            "title": (c.get("title") or "")[:500],
            "year": c.get("year") if isinstance(c.get("year"), int) else None,
        })
    if not clean:
        raise HTTPException(status_code=400, detail="no valid DOIs in candidates")

    workers = int(body.get("workers") or 0)
    ingest = bool(body.get("ingest", True))

    # Persist the DOI list to a tempfile the CLI will read. Use the
    # project's data dir so it's namespaced per-project and survives
    # subprocess launch. We delete it after the job wraps in the finish
    # hook; until then the CLI has read access.
    tmp_dir = Path(tempfile.gettempdir()) / "sciknow-selected"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    import uuid
    tmp_path = tmp_dir / f"dois-{uuid.uuid4().hex[:12]}.json"
    tmp_path.write_text(_json.dumps(clean))

    job_id, _queue = _create_job("corpus_download_selected")
    loop = asyncio.get_event_loop()
    argv = [
        "db", "download-dois",
        "--dois-file", str(tmp_path),
        "--workers", str(workers),
    ]
    argv.append("--ingest" if ingest else "--no-ingest")

    def _cleanup_tmp():
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    _spawn_cli_streaming(job_id, argv, loop, on_finish=_cleanup_tmp)
    return JSONResponse({
        "job_id": job_id,
        "n_selected": len(clean),
    })


@app.post("/api/corpus/expand")
async def api_corpus_expand(
    limit: int = Form(0),
    dry_run: bool = Form(False),
    resolve: bool = Form(False),
    ingest: bool = Form(True),
    relevance: bool = Form(True),
    relevance_threshold: float = Form(0.0),
    relevance_query: str = Form(""),
    workers: int = Form(0),
):
    """Invoke `sciknow db expand` from the web UI — SSE log stream.

    The heavy flags (download_dir, delay) are left at CLI defaults to
    keep the web UX simple; power users can still invoke the CLI
    directly for unusual configurations.
    """
    job_id, _queue = _create_job("corpus_expand")
    loop = asyncio.get_event_loop()
    argv = ["db", "expand", "--limit", str(limit),
            "--relevance-threshold", str(relevance_threshold),
            "--workers", str(workers)]
    if dry_run:
        argv.append("--dry-run")
    argv.append("--resolve" if resolve else "--no-resolve")
    argv.append("--ingest" if ingest else "--no-ingest")
    argv.append("--relevance" if relevance else "--no-relevance")
    if relevance_query:
        argv += ["--relevance-query", relevance_query]
    _spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


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


@app.get("/api/jobs/{job_id}/stats")
async def get_job_stats(job_id: str):
    """Phase 32.5 — server-side counter snapshot for the persistent
    task bar.

    The task bar polls this every ~500ms instead of opening a second
    SSE source on /api/stream/{job_id}. The previous SSE-based design
    competed with the per-section preview consumer for the SAME
    asyncio.Queue, and Queue.get() removes items, so the two
    consumers split the event stream and the task bar saw a tiny
    fraction of the tokens (or zero if the per-section consumer was
    faster). Polling a server-side counter eliminates that race.

    Returns 410 (Gone) for jobs that finished and were swept by the
    GC, so the client can transition to "done" cleanly without
    re-fetching the job list.
    """
    with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            # Treat as already-finished — the GC swept it after the
            # 5-minute window, OR the job was never created.
            raise HTTPException(410, "Job not found (likely already finished)")
        now = time.monotonic()
        ts = job.get("token_timestamps") or deque()
        cutoff = now - 3.0
        recent = sum(1 for t in ts if t >= cutoff)
        tps = recent / 3.0 if recent else 0.0
        elapsed_s = now - job.get("started_at", now)
        return JSONResponse({
            "id": job_id,
            "stream_state": job.get("stream_state", "streaming"),
            "tokens": int(job.get("tokens", 0)),
            "tps": round(tps, 2),
            "elapsed_s": round(elapsed_s, 1),
            "model_name": job.get("model_name"),
            "task_desc": job.get("task_desc") or job.get("type"),
            "target_words": job.get("target_words"),
            "error_message": job.get("error_message"),
        })


@app.get("/api/jobs")
async def list_jobs():
    with _job_lock:
        return [
            {"id": jid, "type": j["type"], "status": j["status"]}
            for jid, j in _jobs.items()
        ]


# ── Project management (Phase 43h GUI) ───────────────────────────────────
#
# Mirrors the `sciknow project …` CLI surface. The web reader was
# launched against a specific book in a specific project, so switching
# projects from the running server would leave the UI pointing at a
# book that no longer exists in the active project's DB. We therefore:
#   - show the active project + all siblings read-only (list / show)
#   - let the user init new projects, destroy others, or `use` a different
#     slug — but `use` just writes .active-project and returns a notice
#     that the user must restart the server to actually switch corpora.
# This keeps the CLI semantics intact and avoids the footgun of hot-
# swapping the DB connection under the running app.


@app.get("/api/projects")
async def api_projects_list():
    """List all projects with their active marker + health."""
    import os as _os
    from sciknow.cli.project import _pg_database_exists
    from sciknow.core.project import (
        get_active_project, list_projects, read_active_slug_from_file,
    )
    active_slug = (
        _os.environ.get("SCIKNOW_PROJECT")
        or read_active_slug_from_file()
        or get_active_project().slug
    )
    out = []
    for p in list_projects():
        try:
            db_ok = _pg_database_exists(p.pg_database)
        except Exception:
            db_ok = False
        out.append({
            "slug": p.slug,
            "active": p.slug == active_slug,
            "pg_database": p.pg_database,
            "data_dir": str(p.data_dir),
            "papers_collection": p.papers_collection,
            "abstracts_collection": p.abstracts_collection,
            "wiki_collection": p.wiki_collection,
            "status": "ok" if (p.exists() and db_ok) else "incomplete",
            "is_default": p.is_default,
        })
    # Running-process view — what the web reader itself is currently
    # using (may differ from .active-project if the server was started
    # with --project / SCIKNOW_PROJECT and that has since changed).
    running = get_active_project()
    return JSONResponse({
        "projects": out,
        "active_slug": active_slug,
        "running_slug": running.slug,
    })


@app.get("/api/projects/{slug}")
async def api_projects_show(slug: str):
    """Show details for a single project (counts + collection names)."""
    from sciknow.cli.project import _pg_database_exists
    from sciknow.core.project import Project, get_active_project, validate_slug
    try:
        validate_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    project = Project(slug=slug, repo_root=get_active_project().repo_root)
    db_ok = _pg_database_exists(project.pg_database)
    payload = {
        "slug": project.slug,
        "root": str(project.root),
        "data_dir": str(project.data_dir),
        "data_dir_exists": project.data_dir.exists(),
        "pg_database": project.pg_database,
        "pg_database_exists": db_ok,
        "qdrant_prefix": project.qdrant_prefix,
        "papers_collection": project.papers_collection,
        "abstracts_collection": project.abstracts_collection,
        "wiki_collection": project.wiki_collection,
        "env_overlay_path": str(project.env_overlay_path),
        "env_overlay_exists": project.env_overlay_path.exists(),
        "is_default": project.is_default,
    }
    if db_ok:
        try:
            from sciknow.storage.db import get_session as _gs
            with _gs(db_name=project.pg_database) as session:
                payload["n_documents"] = session.execute(text("SELECT COUNT(*) FROM documents")).scalar() or 0
                payload["n_chunks"]    = session.execute(text("SELECT COUNT(*) FROM chunks")).scalar() or 0
                payload["n_books"]     = session.execute(text("SELECT COUNT(*) FROM books")).scalar() or 0
                payload["n_drafts"]    = session.execute(text("SELECT COUNT(*) FROM drafts")).scalar() or 0
        except Exception as exc:
            payload["counts_error"] = str(exc)
    return JSONResponse(payload)


@app.post("/api/projects/use")
async def api_projects_use(request: Request):
    """Set the active project (writes ``.active-project``).

    Does NOT hot-swap the running server's DB connection. Returns a
    ``restart_required`` flag so the frontend can tell the user to
    restart ``sciknow book serve`` in a book that exists in the target
    project.
    """
    from sciknow.core.project import (
        Project, get_active_project, list_projects, validate_slug,
        write_active_slug,
    )
    body = await request.json()
    slug = (body or {}).get("slug", "").strip()
    try:
        validate_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if slug != "default":
        candidate = Project(slug=slug, repo_root=get_active_project().repo_root)
        if not candidate.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Project {slug!r} not found. Available: "
                       + ", ".join(p.slug for p in list_projects()),
            )
    write_active_slug(slug)
    running = get_active_project().slug
    return JSONResponse({
        "ok": True,
        "active_slug": slug,
        "running_slug": running,
        "restart_required": slug != running,
        "message": (
            f"Active project set to {slug!r}. Restart `sciknow book serve` "
            f"in a book that belongs to {slug!r} to work on that corpus."
            if slug != running else
            f"Active project is already {slug!r}."
        ),
    })


@app.post("/api/projects/init")
async def api_projects_init(request: Request):
    """Create a new empty project. Runs synchronously (fast: ~3–5 s).

    Does NOT accept ``--from-existing`` — that's a one-shot migration
    that should be done from the CLI where the user has a shell for
    monitoring. The GUI only creates fresh empty projects.
    """
    from sciknow.cli.project import (
        _create_pg_database, _ensure_init_dirs, _init_qdrant_collections_for_project,
        _pg_database_exists, _run_alembic_upgrade,
    )
    from sciknow.core.project import (
        Project, get_active_project, validate_slug,
    )
    body = await request.json()
    slug = (body or {}).get("slug", "").strip()
    try:
        validate_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if slug == "default":
        raise HTTPException(status_code=400, detail="'default' is reserved for the legacy layout.")
    project = Project(slug=slug, repo_root=get_active_project().repo_root)
    if project.root.exists():
        raise HTTPException(status_code=409, detail=f"Project directory already exists: {project.root}")
    if _pg_database_exists(project.pg_database):
        raise HTTPException(status_code=409, detail=f"PG database already exists: {project.pg_database}")

    try:
        _ensure_init_dirs(project)
        _create_pg_database(project.pg_database)
        _run_alembic_upgrade(project.pg_database)
        _init_qdrant_collections_for_project(project)
    except Exception as exc:
        logger.exception("project init failed")
        raise HTTPException(status_code=500, detail=f"init failed: {exc}")
    return JSONResponse({
        "ok": True,
        "slug": slug,
        "pg_database": project.pg_database,
        "papers_collection": project.papers_collection,
    })


@app.post("/api/projects/destroy")
async def api_projects_destroy(request: Request):
    """Drop a project's DB + collections + data dir. Requires ``confirm``
    matching the slug to prevent accidental clicks."""
    import shutil as _shutil
    from sciknow.cli.project import (
        _delete_qdrant_collections_for_project, _drop_pg_database,
    )
    from sciknow.core.project import (
        Project, _active_project_file, get_active_project,
        read_active_slug_from_file, validate_slug,
    )
    body = await request.json()
    slug    = (body or {}).get("slug", "").strip()
    confirm = (body or {}).get("confirm", "").strip()
    try:
        validate_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if slug == "default":
        raise HTTPException(status_code=400, detail="Refusing to destroy the legacy 'default' project.")
    if confirm != slug:
        raise HTTPException(status_code=400, detail="Confirmation slug does not match.")
    if slug == get_active_project().slug:
        raise HTTPException(
            status_code=409,
            detail="Cannot destroy the project the web reader is currently running against. "
                   "Restart `sciknow book serve` in a different project first.",
        )

    project = Project(slug=slug, repo_root=get_active_project().repo_root)
    errors: list[str] = []
    try:
        _drop_pg_database(project.pg_database)
    except Exception as exc:
        errors.append(f"pg drop: {exc}")
    try:
        n_dropped = _delete_qdrant_collections_for_project(project)
    except Exception as exc:
        n_dropped = 0
        errors.append(f"qdrant: {exc}")
    try:
        if project.root.exists():
            _shutil.rmtree(project.root)
    except Exception as exc:
        errors.append(f"data dir: {exc}")
    # If destroying the on-disk active project, clear .active-project
    active = read_active_slug_from_file()
    if active == slug:
        f = _active_project_file()
        if f.exists():
            f.unlink()
    return JSONResponse({"ok": not errors, "errors": errors, "qdrant_dropped": n_dropped})


@app.post("/api/server/shutdown")
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
        _BUILD_TAG=_BUILD_TAG,
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
                # Phase 42 — data-action dispatch replaces the
                # f-string onclick handler with interpolation.
                out += (
                    f'<div class="sec-link sec-empty" '
                    f'draggable="true" '
                    f'data-section-slug="{sec_type}" '
                    f'title="{plan_attr}" '
                    f'data-action="preview-empty-section" '
                    f'data-chapter-id="{ch_id}" '
                    f'data-sec-type="{sec_type}">'
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
                # Phase 42 — orphan +/✗ buttons switched to data-action.
                # The dispatcher handler calls preventDefault+stopPropagation
                # internally so the anchor's navigation doesn't fire.
                out += (
                    f'<a class="sec-link sec-orphan" href="/section/{sec_id}" '
                    f'data-draft-id="{sec_id}" onclick="return navTo(this)" '
                    f'title="Orphan draft: section_type={sec_type!r} doesn&#39;t match any current template slug. Click to inspect, + to adopt into sections, \u2717 to delete.">'
                    f'<span class="sec-status-dot orphan"></span>'
                    f'{display} '
                    f'<span class="meta">orphan \u00b7 v{sec_v} \u00b7 {sec_w}w</span>'
                    f'<button class="sec-orphan-adopt" '
                    f'data-action="adopt-orphan-section" '
                    f'data-chapter-id="{ch_id}" '
                    f'data-sec-type="{sec_type}" '
                    f'title="Add this section_type to the chapter\u2019s sections list (idempotent)">+</button>'
                    f'<button class="sec-orphan-delete" '
                    f'data-action="delete-orphan-draft" '
                    f'data-draft-id="{sec_id}" '
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
                f'data-action="start-writing-chapter" '
                f'data-chapter-id="{ch_id}">'
                f'\u270e Start writing</div>'
            )
        out += '</div>'
    return out


def _render_sources(sources):
    """Render the right-panel sources list. Phase 22 — escapes each
    source string. Phase 48 — the source strings already carry their
    own `[1]`, `[2]` prefix (emitted by the writer prompt), so we
    suppress the <ol> auto-numbering to avoid the redundant
    "1. [1] Smith et al…" rendering. The <li> tag itself stays so that
    buildPopovers can still query `#panel-sources li` in order."""
    if not sources:
        return "<em>No sources.</em>"
    out = '<ol style="list-style:none;padding-left:0;margin-left:0;">'
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
        # Phase 42 — data-action dispatch (see ACTIONS registry in <script>).
        resolve_btn = (
            f'<button class="resolve-btn" data-action="resolve-comment" '
            f'data-comment-id="{_esc(str(cid))}">Resolve</button>'
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
<title>{book_title} — SciKnow [{_BUILD_TAG}]</title>
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
        display: flex; flex-direction: column; height: 100vh; font-size: 14px; }}
/* Phase 54.6 — top bar spans the full viewport width above the sidebar+main.
   Holds app-level navigation (Plan, Settings, Ask, Wiki, KG, Papers, Tools,
   Setup, Dashboard, Projects) so the per-chapter toolbar stays focused on
   writing actions. */
.topbar {{ display: flex; align-items: center; justify-content: flex-end;
           gap: 2px; padding: 6px 16px; min-height: 44px;
           background: var(--toolbar-bg); border-bottom: 1px solid var(--border);
           flex-shrink: 0; flex-wrap: wrap; }}
.topbar .topbar-brand {{ font-size: 13px; font-weight: 700; letter-spacing: -0.01em;
                         color: var(--fg-muted); margin-right: auto; padding-left: 4px; }}
.topbar .nav-btn {{ font-size: 12px; font-weight: 500; padding: 6px 12px;
                    border: 1px solid transparent; border-radius: var(--r-md);
                    cursor: pointer; background: transparent; color: var(--fg);
                    transition: all .12s ease; display: inline-flex;
                    align-items: center; gap: 6px; line-height: 1; }}
.topbar .nav-btn:hover {{ background: var(--bg-elevated); border-color: var(--border);
                          box-shadow: var(--shadow-sm); }}
.topbar .nav-btn:active {{ transform: translateY(1px); }}
.app-body {{ display: flex; flex: 1; overflow: hidden; min-height: 0; }}
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
/* Phase 33 — chapter drag-and-drop visual indicators. Same pattern
   as Phase 26 section drag-drop but on .ch-title elements. */
.ch-title[draggable="true"] {{ cursor: grab; }}
.ch-title[draggable="true"]:active {{ cursor: grabbing; }}
.ch-group.dragging {{ opacity: 0.4; }}
.ch-title.ch-drag-over-top {{ box-shadow: inset 0 2px 0 0 var(--accent); }}
.ch-title.ch-drag-over-bottom {{ box-shadow: inset 0 -2px 0 0 var(--accent); }}
/* Phase 26 — section drag-and-drop */
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
/* Phase 46.F — Setup Wizard trail pills */
.sw-step {{ padding: 4px 10px; border-radius: 12px; background: var(--toolbar-bg);
           color: var(--fg-muted); border: 1px solid var(--border); }}
.sw-step.active {{ background: var(--accent-light); color: var(--accent);
                  border-color: var(--accent); font-weight: 600; }}
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
/* Phase 33 — autowrite mode picker buttons */
.aw-mode-btn {{ font-size: 12px; padding: 6px 12px; }}
.aw-mode-btn.active {{ background: var(--accent); color: var(--accent-fg);
                       border-color: var(--accent); }}
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
/* Phase 54 — wiki reading surface. Two-column layout: sticky TOC on
   the left, 72ch serif column on the right. Wider + more legible
   than the pre-54 600px cramped modal body. */
.wiki-detail-toolbar {{ display: flex; gap: var(--sp-2);
                        align-items: center; flex-wrap: wrap; }}
.wiki-kbd-hint {{ margin-left: auto; font-size: 11px;
                  color: var(--fg-muted); font-family: var(--font-sans); }}
.wiki-kbd-hint kbd {{ font-family: var(--font-mono); font-size: 10px;
                      padding: 1px 5px; border: 1px solid var(--border);
                      border-bottom-width: 2px; border-radius: 3px;
                      background: var(--bg-elevated); }}
.wiki-detail-layout {{ display: grid; grid-template-columns: 180px 1fr;
                       gap: var(--sp-4); align-items: start; }}
@media (max-width: 900px) {{
  .wiki-detail-layout {{ grid-template-columns: 1fr; }}
  .wiki-toc {{ display: none; }}
}}
.wiki-toc {{ position: sticky; top: 0; max-height: 70vh;
             overflow-y: auto; font-size: 12px;
             padding-right: var(--sp-2);
             border-right: 1px solid var(--border); }}
.wiki-toc-heading {{ text-transform: uppercase; font-size: 10px;
                     letter-spacing: 0.1em; color: var(--fg-muted);
                     margin-bottom: var(--sp-2); font-weight: 600; }}
.wiki-toc-list {{ list-style: none; padding: 0; margin: 0; }}
.wiki-toc-list li {{ padding: 2px 0; line-height: 1.4; }}
.wiki-toc-list li.wiki-toc-h3 {{ padding-left: var(--sp-2); }}
.wiki-toc-list li.wiki-toc-h4 {{ padding-left: var(--sp-4); }}
.wiki-toc-list a {{ color: var(--fg-muted); text-decoration: none;
                     cursor: pointer; }}
.wiki-toc-list a:hover,
.wiki-toc-list a.active {{ color: var(--accent); }}
.wiki-page-content {{ font-family: var(--font-serif); font-size: 16px;
                     line-height: 1.65; color: var(--fg);
                     padding: var(--sp-3) var(--sp-4);
                     max-width: 72ch;
                     max-height: 72vh; overflow-y: auto;
                     background: var(--toolbar-bg); border-radius: var(--r-md);
                     border: 1px solid var(--border); }}
.wiki-page-content h1, .wiki-page-content h2, .wiki-page-content h3, .wiki-page-content h4 {{
    margin: var(--sp-4) 0 var(--sp-2); font-family: var(--font-sans);
    font-weight: 600; color: var(--fg); scroll-margin-top: 16px; }}
.wiki-page-content h1 {{ font-size: 24px; }}
.wiki-page-content h2 {{ font-size: 20px; }}
.wiki-page-content h3 {{ font-size: 17px; }}
.wiki-page-content h4 {{ font-size: 15px; color: var(--fg-muted); }}
.wiki-page-content p {{ margin-bottom: var(--sp-3); }}
.wiki-page-content p:first-of-type {{ /* lead paragraph accent */
    font-size: 1.1em; line-height: 1.55;
    padding-left: var(--sp-3);
    border-left: 3px solid var(--accent);
    color: var(--fg); }}
.wiki-page-content code {{ font-family: var(--font-mono); font-size: 13px;
                           padding: 1px 6px; background: var(--bg);
                           border-radius: 3px; }}
/* Phase 54 — [[wiki-slug]] links rendered by _md_to_html */
.wiki-page-content .wiki-link {{ color: var(--accent);
    border-bottom: 1px solid rgba(79,158,255,0.35);
    text-decoration: none;
    transition: border-bottom-color .1s ease; }}
.wiki-page-content .wiki-link:hover {{
    border-bottom-color: var(--accent); }}
/* Phase 54 — Ctrl-K command palette */
.wiki-palette-modal {{ padding: 0 !important; width: 600px !important;
                       max-width: 92vw !important; }}
#wiki-palette {{ align-items: flex-start; padding-top: 15vh; }}
#wiki-palette-input {{ width: 100%; border: 0; border-bottom: 1px solid var(--border);
                        padding: 14px 18px; font-size: 16px;
                        background: var(--bg-elevated); color: var(--fg);
                        outline: none; box-sizing: border-box; }}
#wiki-palette-input:focus {{ border-bottom-color: var(--accent); }}
#wiki-palette-results {{ list-style: none; margin: 0; padding: 4px 0;
                          max-height: 50vh; overflow-y: auto; }}
.wiki-palette-item {{ display: flex; justify-content: space-between;
                       align-items: center; gap: var(--sp-3);
                       padding: 8px 18px; cursor: pointer; }}
.wiki-palette-item:hover, .wiki-palette-item.active {{
    background: var(--accent); color: #fff; }}
.wiki-palette-item:hover .wp-type,
.wiki-palette-item.active .wp-type {{ color: rgba(255,255,255,0.8); }}
.wp-title {{ flex: 1; overflow: hidden; text-overflow: ellipsis;
             white-space: nowrap; font-size: 14px; }}
.wp-type {{ font-size: 10px; font-family: var(--font-mono);
            text-transform: uppercase; color: var(--fg-muted);
            padding: 1px 6px; border: 1px solid var(--border);
            border-radius: 3px; flex: 0 0 auto; }}
.wiki-palette-empty {{ padding: 16px 18px; color: var(--fg-muted);
                        text-align: center; font-size: 13px; }}
.wiki-palette-foot {{ padding: 8px 18px; border-top: 1px solid var(--border);
                       color: var(--fg-muted); font-size: 10px;
                       text-align: right; background: var(--toolbar-bg); }}
.wiki-palette-foot kbd {{ font-family: var(--font-mono);
                           padding: 0 4px;
                           border: 1px solid var(--border);
                           border-bottom-width: 2px;
                           border-radius: 2px;
                           background: var(--bg-elevated); }}
/* Phase 54.1 — staleness banner on wiki pages whose source drifted
   since last compile. The yellow-warning idiom matches the standard
   scientific-paper staleness flag. */
.wiki-stale-banner {{ background: rgba(255, 210, 100, 0.15);
                       border-left: 3px solid #e6b800;
                       color: var(--fg);
                       padding: 10px 14px;
                       margin-bottom: var(--sp-3);
                       border-radius: 0 var(--r-md) var(--r-md) 0;
                       font-size: 12px; line-height: 1.5; }}
.wiki-stale-banner code {{ font-family: var(--font-mono);
                            font-size: 11px; padding: 1px 5px;
                            background: rgba(0,0,0,0.15);
                            border-radius: 3px; }}
/* Phase 54.1 — keyboard cheatsheet overlay (? shortcut). */
.kb-help {{ position: fixed; inset: 0;
            background: rgba(0,0,0,0.55); z-index: 10500;
            display: none; align-items: center; justify-content: center; }}
.kb-help .kb-box {{ background: var(--bg-elevated);
                    border: 1px solid var(--border);
                    border-radius: var(--r-md);
                    padding: 24px 28px; width: 520px;
                    max-width: 94vw;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3); }}
.kb-help h4 {{ margin: 0 0 var(--sp-3); font-size: 14px; }}
.kb-help dl {{ margin: 0; display: grid;
               grid-template-columns: auto 1fr; gap: 8px 16px;
               font-size: 12px; line-height: 1.5; }}
.kb-help dt {{ font-family: var(--font-mono);
               color: var(--fg-muted);
               white-space: nowrap; }}
.kb-help dt kbd {{ font-family: var(--font-mono); font-size: 11px;
                    padding: 1px 6px;
                    border: 1px solid var(--border);
                    border-bottom-width: 2px;
                    border-radius: 3px;
                    background: var(--bg); color: var(--fg); }}
.kb-help .kb-hint {{ margin-top: var(--sp-3); font-size: 11px;
                     color: var(--fg-muted); text-align: right; }}
/* Phase 54.2 — Related + Referenced-by sections below wiki content. */
.wiki-extras {{ margin-top: var(--sp-4); padding: var(--sp-3);
                background: var(--toolbar-bg);
                border: 1px solid var(--border);
                border-radius: var(--r-md); }}
.wiki-extras-h {{ margin: 0 0 var(--sp-2); font-size: 12px;
                   text-transform: uppercase; letter-spacing: 0.08em;
                   color: var(--fg-muted); font-weight: 600; }}
.wiki-compact-list {{ list-style: none; padding: 0; margin: 0;
                       display: grid; gap: 6px; }}
.wiki-compact-list li {{ display: flex; justify-content: space-between;
                          gap: var(--sp-2); align-items: center; }}
.wiki-compact-list a {{ flex: 1; color: var(--accent);
                         text-decoration: none; font-size: 13px;
                         line-height: 1.35;
                         overflow: hidden; text-overflow: ellipsis;
                         white-space: nowrap; }}
.wiki-compact-list a:hover {{ text-decoration: underline; }}
.wiki-compact-list .wp-type {{ font-size: 9px;
                                font-family: var(--font-mono);
                                color: var(--fg-muted);
                                padding: 1px 5px;
                                border: 1px solid var(--border);
                                border-radius: 2px;
                                flex: 0 0 auto; }}
.wiki-compact-list .wp-alt {{ font-size: 11px;
                               color: var(--fg-muted);
                               font-style: italic; }}
/* Phase 54.3 — Ask this page inline chat. Follows the same SSE
   contract the book reader's ask modal uses; rendered here as an
   inline section so it's one scroll away from the page content. */
.wiki-ask-extras {{ background: var(--bg-elevated); }}
.wiki-ask-form {{ display: flex; gap: var(--sp-2); align-items: center;
                   flex-wrap: wrap; }}
.wiki-ask-form input[type="text"] {{ flex: 1; min-width: 220px;
                                      padding: 8px 12px;
                                      border: 1px solid var(--border);
                                      border-radius: var(--r-md);
                                      background: var(--bg);
                                      color: var(--fg); font-size: 14px; }}
.wiki-ask-form input[type="text"]:focus {{ outline: none;
                                            border-color: var(--accent); }}
.wiki-ask-broaden {{ font-size: 11px; color: var(--fg-muted);
                      display: inline-flex; gap: 4px; align-items: center; }}
.wiki-ask-status {{ margin-top: var(--sp-2); font-size: 11px;
                     color: var(--fg-muted); font-family: var(--font-mono); }}
.wiki-ask-stream {{ margin-top: var(--sp-2); font-size: 14px;
                     line-height: 1.55; white-space: pre-wrap;
                     font-family: var(--font-serif); }}
.wiki-ask-sources {{ margin-top: var(--sp-3); padding: var(--sp-2) var(--sp-3);
                      background: var(--toolbar-bg);
                      border: 1px solid var(--border);
                      border-radius: var(--r-md);
                      font-size: 11px; color: var(--fg-muted); }}
.wiki-ask-sources ol {{ padding-left: var(--sp-3); margin: 4px 0; }}
/* Phase 54.4 — Facts from the corpus (KG triples scoped to a concept). */
.wiki-extras-head {{ display: flex; justify-content: space-between;
                     align-items: center; margin-bottom: var(--sp-2); }}
.wiki-extras-head .wiki-extras-h {{ margin: 0; }}
.wiki-facts-kglink {{ font-size: 11px; color: var(--accent);
                      text-decoration: none; }}
.wiki-facts-kglink:hover {{ text-decoration: underline; }}
.wiki-facts-list {{ list-style: none; padding: 0; margin: 0;
                    display: flex; flex-direction: column; gap: 6px;
                    max-height: 360px; overflow-y: auto; }}
.wiki-facts-list li {{ font-size: 13px; line-height: 1.45;
                       padding: 6px 10px;
                       background: var(--bg);
                       border-left: 3px solid var(--accent);
                       border-radius: 0 var(--r-md) var(--r-md) 0; }}
.wiki-facts-list .wf-subject, .wiki-facts-list .wf-object {{
    font-weight: 600; color: var(--fg); }}
.wiki-facts-list .wf-pred {{ color: var(--fg-muted);
                              font-family: var(--font-mono);
                              font-size: 11px;
                              padding: 0 4px; }}
.wiki-facts-list .wf-src {{ display: block; font-size: 11px;
                             color: var(--fg-muted);
                             font-style: italic;
                             margin-top: 2px; }}
/* Colour the border by predicate family when the client has detected one */
.wiki-facts-list li.wf-fam-causal       {{ border-left-color: #D55E00; }}
.wiki-facts-list li.wf-fam-measurement  {{ border-left-color: #0072B2; }}
.wiki-facts-list li.wf-fam-taxonomic    {{ border-left-color: #009E73; }}
.wiki-facts-list li.wf-fam-compositional{{ border-left-color: #8ed6a5; }}
.wiki-facts-list li.wf-fam-citational   {{ border-left-color: #999999; }}
.wiki-facts-list li.wf-fam-other        {{ border-left-color: #CC79A7; }}
/* Phase 54.5 — "My take" annotation. */
.wiki-annotation-extras {{ background: var(--bg-elevated); }}
#wiki-annotation-body {{ width: 100%; box-sizing: border-box;
                          padding: 10px 12px;
                          border: 1px solid var(--border);
                          border-radius: var(--r-md);
                          background: var(--bg); color: var(--fg);
                          font-family: var(--font-sans); font-size: 13px;
                          line-height: 1.5; resize: vertical;
                          min-height: 80px; }}
#wiki-annotation-body:focus {{ outline: none; border-color: var(--accent); }}
.wiki-annotation-actions {{ display: flex; gap: var(--sp-2);
                             align-items: center; margin-top: var(--sp-2);
                             flex-wrap: wrap; }}
/* Phase 54.5 — Active row in the wiki browse list for j/k nav. */
.wiki-page-list li.active-row,
.wiki-page-list tr.active-row {{ outline: 2px solid var(--accent);
                                  outline-offset: -2px;
                                  background: rgba(79,158,255,0.08); }}
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
/* Phase 48 — 3D KG canvas (orbit camera, drag nodes, wheel zoom). CSS is
   intentionally minimal so the inline presentation attributes (radial
   gradients, per-node opacity/scale) aren't shadowed by class rules.
   The `background` color is set inline from the active theme. */
#kg-graph-canvas {{ width: 100%; height: 520px; border-radius: 6px;
                   overflow: hidden; user-select: none; touch-action: none; }}
#kg-graph-canvas svg {{ width: 100%; height: 100%; display: block;
                       cursor: grab; }}
#kg-graph-canvas svg.kg-grabbing {{ cursor: grabbing; }}
#kg-graph-canvas .kg-node {{ cursor: grab; }}
#kg-graph-canvas .kg-node:hover circle {{ filter: brightness(1.25); }}
/* KG theme chip row (Graph tab) */
.kg-controls {{ display: flex; gap: 6px; align-items: center;
                margin: 0 0 8px 0; flex-wrap: wrap; }}
.kg-controls-label {{ font-size: 11px; color: var(--fg-muted);
                      margin-right: 4px; text-transform: uppercase;
                      letter-spacing: 0.06em; }}
.kg-theme-chip {{ width: 22px; height: 22px; padding: 0;
                  border: 2px solid transparent; border-radius: 50%;
                  cursor: pointer; box-shadow: 0 1px 3px rgba(0,0,0,0.25);
                  transition: border-color 0.15s, transform 0.1s; }}
.kg-theme-chip:hover {{ transform: scale(1.12); }}
.kg-theme-chip.active {{ border-color: var(--accent);
                         box-shadow: 0 0 0 2px var(--bg-elevated),
                                     0 0 0 4px var(--accent); }}
.kg-invert-btn {{ margin-left: 8px; font-size: 11px;
                  padding: 3px 10px; border: 1px solid var(--border);
                  background: var(--bg-elevated); color: var(--fg);
                  border-radius: 4px; cursor: pointer;
                  white-space: nowrap; }}
.kg-invert-btn:hover {{ background: var(--accent); color: #fff;
                        border-color: var(--accent); }}
/* Secondary toolbar row: search + color mode + sliders + action buttons */
.kg-controls-2 {{ margin-top: 2px; margin-bottom: 10px; }}
.kg-chip-group {{ display: flex; gap: 4px; align-items: center;
                  padding: 2px 4px; border-radius: 4px;
                  background: var(--bg-elevated); }}
.kg-search-input {{ font-size: 12px; padding: 3px 8px;
                    border: 1px solid var(--border); border-radius: 4px;
                    background: var(--bg-elevated); color: var(--fg);
                    min-width: 140px; flex: 0 0 160px; }}
.kg-search-input:focus {{ outline: none; border-color: var(--accent);
                          box-shadow: 0 0 0 2px rgba(79,158,255,0.15); }}
.kg-select {{ font-size: 11px; padding: 2px 4px;
              border: 1px solid var(--border); border-radius: 3px;
              background: var(--bg); color: var(--fg); }}
.kg-slider {{ width: 90px; accent-color: var(--accent); }}
.kg-slider-short {{ width: 70px; }}
/* Floating right-click context menu */
#kg-context-menu {{ display: none; position: fixed;
                    background: var(--bg-elevated);
                    border: 1px solid var(--border); border-radius: 6px;
                    box-shadow: 0 10px 24px rgba(0,0,0,0.3);
                    min-width: 200px; padding: 4px 0; z-index: 10001;
                    font-size: 12px; }}
#kg-context-menu .kg-menu-item {{ display: flex; align-items: center;
                                  gap: 8px; width: 100%; text-align: left;
                                  border: 0; background: transparent;
                                  color: var(--fg); cursor: pointer;
                                  padding: 6px 12px; font-size: 12px;
                                  font-family: var(--font-sans); }}
#kg-context-menu .kg-menu-item:hover {{ background: var(--accent);
                                        color: #fff; }}
#kg-context-menu .kg-menu-hint {{ margin-left: auto;
                                  color: var(--fg-muted);
                                  font-size: 10px;
                                  font-family: var(--font-mono); }}
#kg-context-menu .kg-menu-item:hover .kg-menu-hint {{ color: #fff;
                                                      opacity: 0.8; }}
#kg-context-menu .kg-menu-sep {{ height: 1px; margin: 4px 8px;
                                 background: var(--border); }}
/* Edge paths should receive hover + right-click events */
#kg-graph-canvas .kg-edge {{ pointer-events: stroke; cursor: pointer; }}
/* Custom color pickers: compact inline "swatch" buttons. The native
   <input type="color"> renders as a small rounded rectangle showing
   the picked color — paired with a short text tag (BG/Aa/Ed/No) so
   users don't need a tooltip to know which channel each one controls. */
.kg-color-box {{ display: inline-flex; gap: 3px; align-items: center;
                 padding: 1px 4px 1px 5px; border-radius: 4px;
                 background: var(--bg-elevated);
                 border: 1px solid var(--border); cursor: pointer;
                 font-size: 10px; line-height: 1;
                 color: var(--fg-muted); }}
.kg-color-box:hover {{ border-color: var(--accent); }}
.kg-color-tag {{ font-family: var(--font-mono);
                 text-transform: uppercase;
                 letter-spacing: 0.05em; }}
.kg-color-box input[type="color"] {{
  width: 22px; height: 18px; padding: 0; border: 0;
  background: transparent; cursor: pointer;
}}
.kg-color-box input[type="color"]::-webkit-color-swatch-wrapper {{ padding: 0; }}
.kg-color-box input[type="color"]::-webkit-color-swatch {{
  border: 1px solid var(--border); border-radius: 3px;
}}
.kg-color-box input[type="color"]::-moz-color-swatch {{
  border: 1px solid var(--border); border-radius: 3px;
}}
.kg-sep {{ width: 1px; height: 20px; background: var(--border);
           margin: 0 6px; display: inline-block; }}
.kg-invert-btn-sm {{ padding: 2px 8px; font-size: 13px;
                     line-height: 1; margin-left: 0; }}
/* Fullscreen for the KG graph pane: pane fills the viewport, canvas
   fills the available vertical space after the toolbar + status line. */
#kg-graph-pane:fullscreen {{
  background: var(--bg);
  padding: 16px 20px;
  overflow: auto;
  width: 100vw;
  height: 100vh;
  box-sizing: border-box;
}}
#kg-graph-pane:-webkit-full-screen {{
  background: var(--bg);
  padding: 16px 20px;
  overflow: auto;
  width: 100vw;
  height: 100vh;
  box-sizing: border-box;
}}
#kg-graph-pane:fullscreen #kg-graph-canvas {{
  height: calc(100vh - 160px);
  min-height: 400px;
}}
#kg-graph-pane:-webkit-full-screen #kg-graph-canvas {{
  height: calc(100vh - 160px);
  min-height: 400px;
}}
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

<!-- Phase 54.6 — full-width top bar with app-level navigation (right-aligned icon + text buttons).
     The per-chapter writing toolbar stays inside <main> so it scrolls with the draft. -->
<header class="topbar" id="topbar">
  <button class="nav-btn" onclick="openPlanModal()" title="View / edit / regenerate the book plan (the leitmotiv)">&#128221; Plan</button>
  <button class="nav-btn" onclick="openBookSettings()" title="Consolidated per-book settings: title, description, plan, length target, style fingerprint">&#9881; Settings</button>
  <button class="nav-btn" onclick="showDashboard()" title="Book dashboard with stats + heatmap">&#128200; Dashboard</button>
  <button class="nav-btn" onclick="showCorkboard()" title="Visual card-based view of the book">&#128204; Corkboard</button>
  <button class="nav-btn" onclick="showVersions()" title="View version history and diffs">&#128344; History</button>
  <button class="nav-btn" onclick="takeSnapshot()" title="Save a snapshot of current draft content">&#128248; Snapshot</button>
  <button class="nav-btn" onclick="openExportModal()" title="Export this section, chapter, or the whole book to text or printable HTML/PDF">&#128229; Export</button>
  <button class="nav-btn" onclick="openAskModal()" title="Full corpus RAG question (sciknow ask question)">&#128270; Ask Corpus</button>
  <button class="nav-btn" onclick="openWikiModal()" title="Query the compiled knowledge wiki (sciknow wiki query)">&#128218; Wiki Query</button>
  <button class="nav-btn" onclick="openKgModal()" title="Browse the knowledge graph (extracted entity-relationship triples)">&#128279; KG</button>
  <button class="nav-btn" onclick="openCatalogModal()" title="Browse the paper catalog (sciknow catalog list)">&#128194; Browse Papers</button>
  <button class="nav-btn" onclick="openToolsModal()" title="CLI tools in the GUI: search, synthesize, topics, corpus enrich/expand">&#128736; Tools</button>
  <button class="nav-btn" onclick="openSetupWizard()" title="End-to-end setup: create project → upload PDFs → ingest → build indices → create book. Phase 46.F.">&#128295; Setup</button>
  <button class="nav-btn" onclick="openProjectsModal()" title="Manage sciknow projects (list / switch / create / destroy). See `sciknow project --help`."><span id="proj-btn-label">&#128193; Projects</span></button>
</header>

<div class="app-body">

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
      <button onclick="doInsertCitations()" title="Two-pass LLM inserts [N] citation markers where needed; mirrors `sciknow book insert-citations`. Saves a new version.">&#128209; Insert Citations</button>
      <button onclick="showScoresPanel()" title="Phase 13 — convergence trajectory for autowrite drafts">&#9783; Scores</button>
      <button onclick="promptArgue()" title="Map evidence for/against a claim">Argue</button>
      <button onclick="doGaps()" title="Analyse gaps in the book">Gaps</button>
    </div>
    <div class="sep"></div>
    <div class="tg">
      <button onclick="openBundleSnapshots()" title="Snapshot / restore whole chapter or whole book — safety net for autowrite-all">&#128230; Bundles</button>
      <button onclick="showChapterReader()" title="Read entire chapter as continuous scroll">Read</button>
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

</div> <!-- /.app-body -->

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
      <button class="tab" data-tab="wiki-lint" onclick="switchWikiTab('wiki-lint')">&#9888;&#65039; Lint</button>
      <button class="tab" data-tab="wiki-consensus" onclick="switchWikiTab('wiki-consensus')">&#9878;&#65039; Consensus</button>
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
          <div class="wiki-detail-toolbar">
            <button class="btn-secondary" onclick="closeWikiPageDetail()">&larr; Back to list</button>
            <button class="btn-secondary" onclick="copyWikiPermalink()" title="Copy permalink to this page">&#128279; Copy link</button>
            <kbd class="wiki-kbd-hint">Press <kbd>Ctrl</kbd>+<kbd>K</kbd> to jump to any page</kbd>
          </div>
          <div id="wiki-page-meta" style="font-size:11px;color:var(--fg-muted);margin:8px 0 12px 0;"></div>
          <div id="wiki-stale-banner" class="wiki-stale-banner" style="display:none;">
            &#9888;&#65039; This page is flagged as stale — run
            <code>sciknow wiki compile --rewrite-stale</code>
            to refresh it from the current sources.
          </div>
          <div class="wiki-detail-layout">
            <aside id="wiki-toc" class="wiki-toc"></aside>
            <div>
              <div id="wiki-page-content" class="wiki-page-content"></div>
              <!-- Phase 54.4 — "Facts from the corpus" on concept pages -->
              <section id="wiki-facts-block" class="wiki-extras" style="display:none;">
                <div class="wiki-extras-head">
                  <h3 class="wiki-extras-h">Facts from the corpus</h3>
                  <a id="wiki-facts-kg-link" class="wiki-facts-kglink"
                     href="#" title="Open this concept in the 3D knowledge graph">
                    &#127760; open in graph
                  </a>
                </div>
                <ul id="wiki-facts-list" class="wiki-facts-list"></ul>
              </section>
              <!-- Phase 54.5 — "My take" user annotation -->
              <section id="wiki-annotation-block" class="wiki-extras wiki-annotation-extras">
                <div class="wiki-extras-head">
                  <h3 class="wiki-extras-h">My take</h3>
                  <span id="wiki-annotation-ts" class="wiki-facts-kglink" style="color:var(--fg-muted);"></span>
                </div>
                <textarea id="wiki-annotation-body"
                          placeholder="Your own notes on this page — disagreements, follow-up questions, how it connects to other work. Saved locally in your project database."
                          rows="4"></textarea>
                <div class="wiki-annotation-actions">
                  <button class="btn-primary" onclick="saveWikiAnnotation()" id="wiki-annotation-save">Save note</button>
                  <button class="btn-secondary" onclick="deleteWikiAnnotation()" id="wiki-annotation-delete">Clear</button>
                  <span id="wiki-annotation-status" class="wiki-ask-status"></span>
                </div>
              </section>
              <!-- Phase 54.3 — Ask this page inline RAG -->
              <section id="wiki-ask-block" class="wiki-extras wiki-ask-extras">
                <h3 class="wiki-extras-h">Ask a question about this page</h3>
                <form class="wiki-ask-form" onsubmit="event.preventDefault(); askWikiPage();">
                  <input type="text" id="wiki-ask-input"
                         placeholder="e.g. What effect size is reported?"
                         autocomplete="off"/>
                  <label class="wiki-ask-broaden" title="Search the whole corpus instead of just this page's sources">
                    <input type="checkbox" id="wiki-ask-broaden"/> broaden
                  </label>
                  <button type="submit" class="btn-primary" id="wiki-ask-submit">Ask</button>
                </form>
                <div id="wiki-ask-status" class="wiki-ask-status"></div>
                <div id="wiki-ask-stream" class="wiki-ask-stream"></div>
                <div id="wiki-ask-sources" class="wiki-ask-sources" style="display:none;"></div>
              </section>
              <section id="wiki-related-block" class="wiki-extras" style="display:none;">
                <h3 class="wiki-extras-h">Related pages</h3>
                <ol id="wiki-related-list" class="wiki-compact-list"></ol>
              </section>
              <section id="wiki-backlinks-block" class="wiki-extras" style="display:none;">
                <h3 class="wiki-extras-h">Referenced by</h3>
                <ol id="wiki-backlinks-list" class="wiki-compact-list"></ol>
              </section>
            </div>
          </div>
        </div>
      </div>
      <!-- Phase 54.6.2 — Lint tab: surfaces `sciknow wiki lint` in the GUI. -->
      <div class="tab-pane" id="wiki-lint-pane" style="display:none;">
        <div style="font-size:12px;color:var(--fg-muted);margin-bottom:8px;">
          Check wiki health: broken links, stale pages, orphaned concepts,
          missing summaries, and optionally contradictions across paper
          summaries (deep mode uses the LLM — slower).
        </div>
        <div class="field" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
          <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;">
            <input type="checkbox" id="wiki-lint-deep"> deep (LLM contradiction detection)
          </label>
          <button class="btn-primary" id="wiki-lint-run" onclick="doWikiLint()">Run Lint</button>
          <button class="btn-secondary" id="wiki-lint-stop" onclick="stopWikiLint()" style="display:none;">Stop</button>
        </div>
        <div id="wiki-lint-status" style="margin-top:8px;font-size:12px;color:var(--fg-muted);"></div>
        <div id="wiki-lint-summary" style="margin-top:8px;"></div>
        <div id="wiki-lint-issues" style="margin-top:10px;max-height:400px;overflow:auto;"></div>
      </div>
      <!-- Phase 54.6.2 — Consensus tab: surfaces `sciknow wiki consensus` in the GUI. -->
      <div class="tab-pane" id="wiki-consensus-pane" style="display:none;">
        <div style="font-size:12px;color:var(--fg-muted);margin-bottom:8px;">
          Map the agreement landscape for a topic. Uses the knowledge graph
          plus paper summaries to classify claims as strong / moderate /
          weak / contested, identify supporting vs contradicting papers,
          and flag the most-debated sub-topics. The result is saved as a
          wiki synthesis page under <code>/synthesis/</code>.
        </div>
        <div class="field">
          <label>Topic</label>
          <input type="text" id="wiki-consensus-topic"
                 placeholder="e.g. cosmic ray cloud nucleation"
                 onkeydown="if(event.key==='Enter')doWikiConsensus()">
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          <button class="btn-primary" id="wiki-consensus-run" onclick="doWikiConsensus()">Map Consensus</button>
          <button class="btn-secondary" id="wiki-consensus-stop" onclick="stopWikiConsensus()" style="display:none;">Stop</button>
        </div>
        <div id="wiki-consensus-status" style="margin-top:8px;font-size:12px;color:var(--fg-muted);"></div>
        <div id="wiki-consensus-summary" style="margin-top:8px;font-size:13px;"></div>
        <div id="wiki-consensus-claims" style="margin-top:10px;max-height:400px;overflow:auto;"></div>
        <div id="wiki-consensus-debated" style="margin-top:10px;"></div>
      </div>
    </div>
  </div>
</div>

<!-- Phase 54.1 — KaTeX math rendering (loaded from the jsDelivr CDN
     with Subresource Integrity so first load is verified; falls
     back silently if the network is offline — math just renders as
     its raw `$...$` source). ~90KB JS + ~60KB CSS gzipped. -->
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
      integrity="sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+"
      crossorigin="anonymous"/>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"
        integrity="sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFB"
        crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
        integrity="sha384-43gviWU0YVjaDtb/GhzOouOXtZMP/7XUzwPTstBeZFe/+rCMvRwr4yROQP43s0Xk"
        crossorigin="anonymous"></script>

<!-- Phase 54.1 — keyboard shortcuts cheatsheet (? to toggle) -->
<div id="kb-help" class="kb-help" onclick="if(event.target===this)closeKbHelp()">
  <div class="kb-box">
    <h4>Keyboard shortcuts</h4>
    <dl>
      <dt><kbd>Ctrl</kbd>+<kbd>K</kbd></dt><dd>Quick-jump to any wiki page</dd>
      <dt><kbd>/</kbd></dt><dd>Focus the palette search box</dd>
      <dt><kbd>g</kbd> then <kbd>w</kbd></dt><dd>Go to Wiki (Browse)</dd>
      <dt><kbd>g</kbd> then <kbd>h</kbd></dt><dd>Go home (close all modals)</dd>
      <dt><kbd>Esc</kbd></dt><dd>Close modal / palette / help</dd>
      <dt><kbd>?</kbd></dt><dd>Toggle this help</dd>
    </dl>
    <div class="kb-hint">press <kbd>Esc</kbd> or click outside to close</div>
  </div>
</div>

<!-- Phase 54 — Ctrl-K wiki command palette (quick-jump by fuzzy title) -->
<div class="modal-overlay" id="wiki-palette" onclick="if(event.target===this)closeWikiPalette()" style="display:none;">
  <div class="modal wiki-palette-modal">
    <input type="text" id="wiki-palette-input"
           placeholder="Jump to wiki page… (type to filter)"
           autocomplete="off"
           oninput="_renderWikiPalette()"
           onkeydown="_wikiPaletteKey(event)"/>
    <ol id="wiki-palette-results"></ol>
    <div class="wiki-palette-foot">
      <kbd>&uarr;</kbd><kbd>&darr;</kbd> navigate
      &nbsp;&nbsp;<kbd>&crarr;</kbd> open
      &nbsp;&nbsp;<kbd>Esc</kbd> close
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

<!-- Phase 33 — Autowrite Configuration Modal (replaces the old triple-prompt UX) -->
<div class="modal-overlay" id="autowrite-config-modal" onclick="if(event.target===this)closeModal('autowrite-config-modal')">
  <div class="modal" style="max-width:480px;">
    <div class="modal-header">
      <h3>&#9889; Autowrite</h3>
      <button class="modal-close" onclick="closeModal('autowrite-config-modal')">&times;</button>
    </div>
    <div class="modal-body">
      <p id="aw-config-scope" style="font-size:13px;color:var(--fg);margin-bottom:16px;font-weight:600;"></p>
      <div class="field">
        <label>Max iterations per section</label>
        <input type="number" id="aw-config-max-iter" value="3" min="1" max="10" style="width:80px;">
        <span style="font-size:11px;color:var(--fg-muted);margin-left:8px;">Each iteration: score &rarr; verify &rarr; revise</span>
      </div>
      <div class="field">
        <label>Target score (0.0 &ndash; 1.0)</label>
        <input type="number" id="aw-config-target-score" value="0.85" min="0" max="1" step="0.05" style="width:80px;">
        <span style="font-size:11px;color:var(--fg-muted);margin-left:8px;">Stop iterating when overall &ge; this</span>
      </div>
      <div id="aw-config-mode-section" style="display:none;margin-top:16px;">
        <label>Existing drafts</label>
        <p id="aw-config-mode-info" style="font-size:12px;color:var(--fg-muted);margin:4px 0 10px;"></p>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <button class="btn-secondary aw-mode-btn active" data-mode="skip" onclick="selectAwMode('skip')" title="Only fill sections that don't have a draft yet">Skip (fill missing)</button>
          <button class="btn-secondary aw-mode-btn" data-mode="rebuild" onclick="selectAwMode('rebuild')" title="Overwrite all sections from scratch">Rebuild</button>
          <button class="btn-secondary aw-mode-btn" data-mode="resume" onclick="selectAwMode('resume')" title="Load existing content + run more iterations">Resume</button>
        </div>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn-secondary" onclick="closeModal('autowrite-config-modal')">Cancel</button>
      <button class="btn-primary" onclick="confirmAutowrite()">&#9889; Start</button>
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

<!-- Phase 39 — Consolidated Book Settings modal -->
<div class="modal-overlay" id="book-settings-modal" onclick="if(event.target===this)closeModal('book-settings-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3>&#9881; Book Settings</h3>
      <button class="modal-close" onclick="closeModal('book-settings-modal')">&times;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="bs-basics" onclick="switchBookSettingsTab('bs-basics')">Basics</button>
      <button class="tab" data-tab="bs-leitmotiv" onclick="switchBookSettingsTab('bs-leitmotiv')">Leitmotiv</button>
      <button class="tab" data-tab="bs-style" onclick="switchBookSettingsTab('bs-style')">Style</button>
    </div>
    <div class="modal-body">

      <!-- Basics tab -->
      <div id="bs-basics-pane">
        <p style="font-size:11px;color:var(--fg-muted);margin-bottom:12px;">
          Persistent per-book settings. Writes through
          <code>PUT /api/book</code>; empty values leave the field untouched.
        </p>
        <div class="field">
          <label>Title</label>
          <input type="text" id="bs-title" placeholder="(required)">
        </div>
        <div class="field">
          <label>Description</label>
          <input type="text" id="bs-description" placeholder="One-line blurb shown in catalog / stats">
        </div>
        <div class="field" style="display:flex;gap:8px;align-items:flex-end;">
          <div style="flex:1;">
            <label>Target words per chapter</label>
            <input type="number" id="bs-target-chapter-words" min="0" step="500"
                   placeholder="6000 (default)">
          </div>
          <div style="flex:2;font-size:11px;color:var(--fg-muted);padding-bottom:8px;">
            Per-section target = chapter target ÷ number of sections.
            Set <strong>0</strong> to clear the override and use the default.
            Override per section in the Chapter modal's Sections tab.
          </div>
        </div>
        <div id="bs-basics-meta" style="margin-top:10px;font-size:11px;color:var(--fg-muted);"></div>
        <div style="display:flex;gap:8px;align-items:center;margin-top:14px;">
          <button class="btn-primary" onclick="saveBookSettings('basics')">Save Basics</button>
          <span id="bs-basics-status" style="font-size:12px;color:var(--fg-muted);"></span>
        </div>
      </div>

      <!-- Leitmotiv tab (the book plan) -->
      <div id="bs-leitmotiv-pane" style="display:none;">
        <p style="font-size:11px;color:var(--fg-muted);margin-bottom:8px;">
          The book's thesis / scope document (200&ndash;500 words). Injected into
          every writer prompt so chapter sections stay aligned with the overall
          argument. Use the <strong>&#128221; Plan</strong> quick-editor for
          regeneration; this tab is for direct editing.
        </p>
        <div class="field">
          <label>Plan / leitmotiv</label>
          <textarea id="bs-plan" rows="16" style="font-family:var(--font-sans,inherit);font-size:13px;line-height:1.55;"></textarea>
        </div>
        <div style="display:flex;gap:8px;align-items:center;margin-top:8px;">
          <button class="btn-primary" onclick="saveBookSettings('leitmotiv')">Save Plan</button>
          <span id="bs-leitmotiv-status" style="font-size:12px;color:var(--fg-muted);"></span>
        </div>
      </div>

      <!-- Style tab -->
      <div id="bs-style-pane" style="display:none;">
        <p style="font-size:11px;color:var(--fg-muted);margin-bottom:10px;">
          Style fingerprint extracted from drafts marked
          <em>final</em> / <em>reviewed</em> / <em>revised</em> (Phase 32.10 / Layer 5).
          Injected into the autowrite writer prompt so future sections match your
          already-approved style. Refresh after you've accepted or edited drafts.
        </p>
        <div id="bs-style-fingerprint"
             style="padding:14px;border:1px solid var(--border);border-radius:6px;background:var(--bg-alt,#f8f8f8);min-height:120px;"></div>
        <div style="display:flex;gap:8px;align-items:center;margin-top:10px;">
          <button class="btn-primary" onclick="refreshStyleFingerprint()">Recompute Fingerprint</button>
          <span id="bs-style-status" style="font-size:12px;color:var(--fg-muted);"></span>
        </div>
      </div>

    </div>
  </div>
</div>

<!-- Phase 43h — Project management modal -->
<div class="modal-overlay" id="projects-modal" onclick="if(event.target===this)closeModal('projects-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3>&#128193; Projects</h3>
      <button class="modal-close" onclick="closeModal('projects-modal')">&times;</button>
    </div>
    <div class="modal-body">
      <p style="font-size:11px;color:var(--fg-muted);margin-bottom:12px;">
        Each project has its own PostgreSQL DB, Qdrant collections, and <code>data/</code> directory.
        Mirrors <code>sciknow project …</code>. The web reader is currently running against
        <strong id="proj-running"></strong>; switching the active project only takes effect after
        restarting <code>sciknow book serve</code>.
      </p>
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px;">
        <button class="btn-primary" onclick="refreshProjectsList()">Refresh</button>
        <span id="proj-msg" style="font-size:12px;color:var(--fg-muted);flex:1;"></span>
      </div>
      <div id="projects-list-wrap" style="margin-bottom:16px;">
        <div id="projects-list" style="font-size:13px;">Loading…</div>
      </div>
      <div style="border-top:1px solid var(--border);padding-top:12px;margin-top:8px;">
        <h4 style="font-size:13px;margin-bottom:8px;">Create new project</h4>
        <p style="font-size:11px;color:var(--fg-muted);margin-bottom:8px;">
          Creates an empty project (DB + collections + dir + migrations). For the one-shot
          migration of the legacy install, use <code>sciknow project init &lt;slug&gt; --from-existing</code>
          from the CLI.
        </p>
        <div style="display:flex;gap:8px;align-items:center;">
          <input type="text" id="proj-new-slug" placeholder="new-project-slug (lowercase, hyphens)"
                 style="flex:1;padding:6px 10px;border:1px solid var(--border);border-radius:var(--r-sm);background:var(--bg);color:var(--fg);">
          <button class="btn-primary" onclick="createProject()">Create</button>
        </div>
      </div>
      <div id="proj-detail" style="margin-top:14px;"></div>
    </div>
  </div>
</div>

<!-- Phase 46.F — End-to-end Setup Wizard: project → corpus → indices → expand → book -->
<div class="modal-overlay" id="setup-wizard-modal" onclick="if(event.target===this)closeModal('setup-wizard-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3>&#128736; Setup Wizard</h3>
      <button class="modal-close" onclick="closeModal('setup-wizard-modal')">&times;</button>
    </div>
    <!-- Step progress trail -->
    <div id="sw-trail" style="display:flex;gap:8px;padding:8px 18px;border-bottom:1px solid var(--border);font-size:12px;">
      <span data-sw-step="project" class="sw-step active">1 · Project</span>
      <span data-sw-step="corpus"  class="sw-step">2 · Corpus</span>
      <span data-sw-step="indices" class="sw-step">3 · Indices</span>
      <span data-sw-step="expand"  class="sw-step">4 · Expand</span>
      <span data-sw-step="book"    class="sw-step">5 · Book</span>
    </div>

    <div class="modal-body" style="padding-top:0;">

      <!-- STEP 1 — Project -->
      <div id="sw-step-project" class="sw-step-pane" style="padding:14px 18px;">
        <p style="font-size:12px;color:var(--fg-muted);margin-bottom:10px;">
          A <strong>project</strong> isolates one corpus (its own DB + Qdrant
          collections + data dir). Pick an existing one or create a new one.
          The web reader is currently serving the project shown as
          <strong>active</strong>; creating a new project here writes
          <code>.active-project</code> but does NOT hot-swap this running
          server — finish the wizard, then restart <code>sciknow book
          serve</code> against the new project.
        </p>
        <div style="display:flex;gap:10px;align-items:flex-start;">
          <div style="flex:1;">
            <h4 style="font-size:13px;margin:0 0 6px;">Existing projects</h4>
            <div id="sw-project-list" style="font-size:12px;max-height:180px;overflow:auto;border:1px solid var(--border);border-radius:6px;">
              Loading…
            </div>
          </div>
          <div style="flex:1;border-left:1px solid var(--border);padding-left:12px;">
            <h4 style="font-size:13px;margin:0 0 6px;">Create new</h4>
            <div class="field">
              <label>Slug</label>
              <input type="text" id="sw-new-slug" placeholder="global-cooling">
            </div>
            <button class="btn-primary" onclick="swCreateProject()">Create empty project</button>
            <p style="font-size:11px;color:var(--fg-muted);margin-top:6px;">
              Slug is lowercase alphanumerics + hyphens. Runs migrations
              + creates Qdrant collections (~3 s).
            </p>
          </div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:12px;">
          <span id="sw-project-status" style="font-size:12px;color:var(--fg-muted);"></span>
          <button class="btn-primary" onclick="swGoto('corpus')">Next: Corpus &rarr;</button>
        </div>
      </div>

      <!-- STEP 2 — Corpus -->
      <div id="sw-step-corpus" class="sw-step-pane" style="display:none;padding:14px 18px;">
        <p style="font-size:12px;color:var(--fg-muted);margin-bottom:10px;">
          Feed the project PDFs. Two paths:
          <strong>upload</strong> files from your browser, or point to a
          <strong>directory on this server</strong>. Either way, the
          ingestion pipeline runs (PDF → metadata → sections → chunks →
          embeddings). Large corpora take hours.
        </p>
        <div id="sw-corpus-status" style="font-size:12px;padding:8px;background:var(--toolbar-bg);border:1px solid var(--border);border-radius:4px;margin-bottom:10px;">
          Loading corpus status…
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
          <div style="border:1px solid var(--border);border-radius:6px;padding:10px;">
            <div style="font-weight:600;font-size:13px;margin-bottom:4px;">&#128190; Upload PDFs</div>
            <p style="font-size:11px;color:var(--fg-muted);margin-bottom:8px;">
              Files are staged under
              <code>{{data_dir}}/inbox/uploads_&lt;ts&gt;/</code> and then
              ingested.
            </p>
            <input type="file" id="sw-upload-files" accept="application/pdf,.pdf" multiple
                   style="display:block;margin-bottom:8px;">
            <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;margin-bottom:4px;">
              <input type="checkbox" id="sw-upload-start-ingest" checked>
              start ingesting immediately
            </label>
            <button class="btn-primary" onclick="swUploadPDFs()">Upload</button>
          </div>
          <div style="border:1px solid var(--border);border-radius:6px;padding:10px;">
            <div style="font-weight:600;font-size:13px;margin-bottom:4px;">&#128193; Server directory</div>
            <p style="font-size:11px;color:var(--fg-muted);margin-bottom:8px;">
              Path is resolved on the server. Useful when a corpus is
              already on disk (or over a network mount).
            </p>
            <div class="field">
              <input type="text" id="sw-ingest-path" placeholder="/path/to/pdfs">
            </div>
            <div class="field" style="display:flex;gap:10px;flex-wrap:wrap;">
              <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;">
                <input type="checkbox" id="sw-ingest-recursive" checked> recursive
              </label>
              <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;">
                <input type="checkbox" id="sw-ingest-force"> force re-ingest
              </label>
            </div>
            <button class="btn-primary" onclick="swIngestDirectory()">Ingest directory</button>
          </div>
        </div>
        <pre id="sw-ingest-log" style="margin-top:10px;max-height:260px;overflow:auto;background:var(--bg-alt,#f8f8f8);border:1px solid var(--border);border-radius:6px;padding:10px;font-size:11px;font-family:ui-monospace,monospace;white-space:pre-wrap;"></pre>
        <div style="display:flex;gap:8px;justify-content:space-between;margin-top:8px;">
          <button onclick="swGoto('project')">&larr; Back</button>
          <div>
            <button onclick="swRefreshStatus()">Refresh status</button>
            <button class="btn-primary" onclick="swGoto('indices')">Next: Indices &rarr;</button>
          </div>
        </div>
      </div>

      <!-- STEP 3 — Indices -->
      <div id="sw-step-indices" class="sw-step-pane" style="display:none;padding:14px 18px;">
        <p style="font-size:12px;color:var(--fg-muted);margin-bottom:10px;">
          After ingestion, build the three optional indices. Each improves
          downstream quality — you can skip any of them. Run in any order.
        </p>
        <div id="sw-indices-status" style="font-size:12px;padding:8px;background:var(--toolbar-bg);border:1px solid var(--border);border-radius:4px;margin-bottom:10px;">
          Loading index status…
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">
          <div style="border:1px solid var(--border);border-radius:6px;padding:10px;">
            <div style="font-weight:600;font-size:13px;margin-bottom:4px;">&#127918; Topic Clusters</div>
            <p style="font-size:11px;color:var(--fg-muted);margin-bottom:8px;">
              BERTopic over abstracts. Fast (seconds). Enables
              <code>--topic</code> filtering in retrieval + the Topics
              browser.
            </p>
            <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;margin-bottom:4px;">
              <input type="checkbox" id="sw-cluster-rebuild"> rebuild from scratch
            </label>
            <button class="btn-primary" onclick="swRunIndex('cluster')">Cluster</button>
          </div>
          <div style="border:1px solid var(--border);border-radius:6px;padding:10px;">
            <div style="font-weight:600;font-size:13px;margin-bottom:4px;">&#127794; RAPTOR tree</div>
            <p style="font-size:11px;color:var(--fg-muted);margin-bottom:8px;">
              Hierarchical summaries (UMAP + GMM). Slow (5–30 min).
              Enables broad-synthesis retrieval.
            </p>
            <button class="btn-primary" onclick="swRunIndex('raptor')">Build RAPTOR</button>
          </div>
          <div style="border:1px solid var(--border);border-radius:6px;padding:10px;">
            <div style="font-weight:600;font-size:13px;margin-bottom:4px;">&#128218; Wiki compile</div>
            <p style="font-size:11px;color:var(--fg-muted);margin-bottom:8px;">
              Compile per-paper wiki pages + KG triples. Slow
              (LLM-bound, ~1 min per paper).
            </p>
            <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;margin-bottom:4px;">
              <input type="checkbox" id="sw-wiki-rebuild"> rebuild
            </label>
            <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;margin-bottom:4px;">
              <input type="checkbox" id="sw-wiki-stale" checked> rewrite stale
            </label>
            <button class="btn-primary" onclick="swRunIndex('wiki')">Compile wiki</button>
          </div>
        </div>
        <pre id="sw-indices-log" style="margin-top:10px;max-height:260px;overflow:auto;background:var(--bg-alt,#f8f8f8);border:1px solid var(--border);border-radius:6px;padding:10px;font-size:11px;font-family:ui-monospace,monospace;white-space:pre-wrap;"></pre>
        <div style="display:flex;gap:8px;justify-content:space-between;margin-top:8px;">
          <button onclick="swGoto('corpus')">&larr; Back</button>
          <div>
            <button onclick="swRefreshStatus()">Refresh status</button>
            <button class="btn-primary" onclick="swGoto('expand')">Next: Expand &rarr;</button>
          </div>
        </div>
      </div>

      <!-- STEP 4 — Expand -->
      <div id="sw-step-expand" class="sw-step-pane" style="display:none;padding:14px 18px;">
        <p style="font-size:12px;color:var(--fg-muted);margin-bottom:10px;">
          Optional: grow the corpus by following citations or pulling
          everything an author has published. Uses the full
          Expand tab — see the <strong>Tools</strong> toolbar button for
          the detailed surface.
        </p>
        <div style="display:flex;gap:10px;">
          <button class="btn-primary" onclick="closeModal('setup-wizard-modal');openToolsModal();switchToolsTab('tl-corpus');">
            &#128736; Open Tools &rarr; Corpus tab
          </button>
        </div>
        <div style="display:flex;gap:8px;justify-content:space-between;margin-top:20px;">
          <button onclick="swGoto('indices')">&larr; Back</button>
          <button class="btn-primary" onclick="swGoto('book')">Next: Book &rarr;</button>
        </div>
      </div>

      <!-- STEP 5 — Book -->
      <div id="sw-step-book" class="sw-step-pane" style="display:none;padding:14px 18px;">
        <p style="font-size:12px;color:var(--fg-muted);margin-bottom:10px;">
          Create the writing project. Pick a type — <strong>scientific
          book</strong> (hierarchical, chapters → sections) or
          <strong>scientific paper</strong> (flat IMRaD, one chapter with
          canonical sections). The type drives section defaults, prompt
          conditioning, length targets.
        </p>
        <div class="field" style="display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap;">
          <div style="flex:3;min-width:220px;">
            <label>Title</label>
            <input type="text" id="sw-book-title" placeholder="e.g. Global Cooling: The Coming Solar Minimum">
          </div>
          <div style="flex:2;min-width:180px;">
            <label>Type</label>
            <select id="sw-book-type">
              <option value="scientific_book" selected>Scientific Book (chapters)</option>
              <option value="scientific_paper">Scientific Paper (IMRaD, flat)</option>
            </select>
          </div>
          <div style="flex:1;min-width:120px;">
            <label>Target words/chap</label>
            <input type="number" id="sw-book-target" placeholder="(type default)">
          </div>
        </div>
        <div class="field">
          <label>Description (optional)</label>
          <input type="text" id="sw-book-desc" placeholder="One-line blurb.">
        </div>
        <button class="btn-primary" onclick="swCreateBook()">Create project</button>
        <div id="sw-book-status" style="margin-top:8px;font-size:12px;color:var(--fg-muted);"></div>
        <div style="display:flex;gap:8px;justify-content:space-between;margin-top:14px;">
          <button onclick="swGoto('expand')">&larr; Back</button>
          <button onclick="closeModal('setup-wizard-modal')">Done</button>
        </div>
      </div>

    </div>
  </div>
</div>

<!-- Phase 38 — Scoped snapshot bundles (chapter + book) -->
<div class="modal-overlay" id="bundle-modal" onclick="if(event.target===this)closeModal('bundle-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3>&#128230; Snapshot Bundles</h3>
      <button class="modal-close" onclick="closeModal('bundle-modal')">&times;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="sb-chapter" onclick="switchBundleTab('sb-chapter')">Chapter</button>
      <button class="tab" data-tab="sb-book" onclick="switchBundleTab('sb-book')">Book</button>
    </div>
    <div class="modal-body">

      <!-- Chapter scope -->
      <div id="sb-chapter-pane">
        <p style="font-size:11px;color:var(--fg-muted);margin-bottom:10px;">
          Snapshot every section in the current chapter as one bundle.
          Restore is <strong>non-destructive</strong> &mdash; each section
          gets a NEW draft version, so existing drafts stay put as an undo path.
          Best used before firing <code>autowrite</code> on a whole chapter.
        </p>
        <div class="field" style="display:flex;gap:8px;align-items:flex-end;">
          <div style="flex:2;">
            <label>Snapshot label (optional)</label>
            <input type="text" id="sb-chapter-name" placeholder="(auto: chapter title + timestamp)">
          </div>
          <button class="btn-primary" onclick="doBundleSnapshot('chapter')">Snapshot Current Chapter</button>
        </div>
        <div id="sb-chapter-status" style="font-size:12px;color:var(--fg-muted);margin:8px 0;"></div>
        <div id="sb-chapter-list" style="margin-top:10px;"></div>
      </div>

      <!-- Book scope -->
      <div id="sb-book-pane" style="display:none;">
        <p style="font-size:11px;color:var(--fg-muted);margin-bottom:10px;">
          Snapshot every draft across every chapter in this book.
          Restore walks each chapter bundle and creates new draft versions per section.
          Slow on big books &mdash; prefer a chapter snapshot when you only need scope for one chapter.
        </p>
        <div class="field" style="display:flex;gap:8px;align-items:flex-end;">
          <div style="flex:2;">
            <label>Snapshot label (optional)</label>
            <input type="text" id="sb-book-name" placeholder="(auto: book title + timestamp)">
          </div>
          <button class="btn-primary" onclick="doBundleSnapshot('book')">Snapshot Whole Book</button>
        </div>
        <div id="sb-book-status" style="font-size:12px;color:var(--fg-muted);margin:8px 0;"></div>
        <div id="sb-book-list" style="margin-top:10px;"></div>
      </div>

    </div>
  </div>
</div>

<!-- Phase 36 — Tools Modal: CLI-parity panel (search / synthesize / topics / corpus) -->
<div class="modal-overlay" id="tools-modal" onclick="if(event.target===this)closeModal('tools-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3>&#128736; Tools</h3>
      <button class="modal-close" onclick="closeModal('tools-modal')">&times;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="tl-search" onclick="switchToolsTab('tl-search')">Search</button>
      <button class="tab" data-tab="tl-synth" onclick="switchToolsTab('tl-synth')">Synthesize</button>
      <button class="tab" data-tab="tl-topics" onclick="switchToolsTab('tl-topics')">Topics</button>
      <button class="tab" data-tab="tl-corpus" onclick="switchToolsTab('tl-corpus')">Corpus</button>
    </div>
    <div class="modal-body">

      <!-- Search tab (sciknow search query + search similar) -->
      <div id="tl-search-pane">
        <p style="font-size:11px;color:var(--fg-muted);margin-bottom:8px;">
          Hybrid retrieval (dense + sparse + FTS) with cross-encoder rerank.
          Mirrors <code>sciknow search query</code> and <code>sciknow search similar</code>.
        </p>
        <div class="field" style="display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap;">
          <div style="flex:3;min-width:220px;">
            <label>Query &nbsp;<span style="color:var(--fg-muted);font-weight:400;">or DOI / title fragment for similar-paper search</span></label>
            <input type="text" id="tl-search-q" placeholder="e.g. sea surface temperature reconstruction"
                   onkeydown="if(event.key==='Enter')doToolSearch('query')">
          </div>
          <div style="flex:1;min-width:100px;"><label>Top-K</label>
            <input type="number" id="tl-search-topk" value="10" min="1" max="50"></div>
          <div style="flex:1;min-width:90px;"><label>Year from</label>
            <input type="number" id="tl-search-yfrom"></div>
          <div style="flex:1;min-width:90px;"><label>Year to</label>
            <input type="number" id="tl-search-yto"></div>
        </div>
        <div class="field" style="display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap;">
          <div style="flex:2;min-width:160px;"><label>Section</label>
            <select id="tl-search-section">
              <option value="">(any)</option>
              <option value="abstract">abstract</option>
              <option value="introduction">introduction</option>
              <option value="methods">methods</option>
              <option value="results">results</option>
              <option value="discussion">discussion</option>
              <option value="conclusion">conclusion</option>
            </select></div>
          <div style="flex:2;min-width:160px;"><label>Topic cluster</label>
            <input type="text" id="tl-search-topic" placeholder="(any)"></div>
          <div style="flex:1;min-width:100px;display:flex;align-items:center;">
            <label style="display:flex;align-items:center;gap:6px;font-weight:400;">
              <input type="checkbox" id="tl-search-expand"> LLM expand
            </label></div>
          <button class="btn-primary" onclick="doToolSearch('query')">Search</button>
          <button onclick="doToolSearch('similar')" title="Find papers with a similar abstract to the one you typed (DOI or title fragment)">Similar</button>
        </div>
        <div id="tl-search-status" style="font-size:12px;color:var(--fg-muted);margin:4px 0;"></div>
        <div id="tl-search-results" style="margin-top:8px;"></div>
      </div>

      <!-- Synthesize tab (sciknow ask synthesize) -->
      <div id="tl-synth-pane" style="display:none;">
        <p style="font-size:11px;color:var(--fg-muted);margin-bottom:8px;">
          Multi-paper synthesis on a topic &mdash; biases the prompt toward
          consensus, methods and open questions. Mirrors <code>sciknow ask synthesize</code>.
          (For single Q&amp;A use <strong>&#128270; Ask Corpus</strong> in the toolbar.)
        </p>
        <div class="field">
          <label>Topic</label>
          <input type="text" id="tl-synth-topic" placeholder="e.g. solar activity and climate variability"
                 onkeydown="if(event.key==='Enter')doToolSynthesize()">
        </div>
        <div class="field" style="display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap;">
          <div style="flex:1;"><label>Context-K</label>
            <input type="number" id="tl-synth-k" value="12" min="4" max="30"></div>
          <div style="flex:1;"><label>Year from</label><input type="number" id="tl-synth-yfrom"></div>
          <div style="flex:1;"><label>Year to</label><input type="number" id="tl-synth-yto"></div>
          <div style="flex:2;"><label>Topic cluster filter</label>
            <input type="text" id="tl-synth-topicfilter" placeholder="(any)"></div>
          <button class="btn-primary" onclick="doToolSynthesize()">Synthesize</button>
        </div>
        <div id="tl-synth-status" style="font-size:12px;color:var(--fg-muted);margin-bottom:8px;"></div>
        <div class="modal-stream" id="tl-synth-stream"></div>
        <div id="tl-synth-stats" class="stream-stats"></div>
        <div class="modal-sources" id="tl-synth-sources" style="display:none;"></div>
      </div>

      <!-- Topics tab (sciknow catalog topics + Phase 46.E domain tags) -->
      <div id="tl-topics-pane" style="display:none;">
        <div style="display:flex;gap:12px;align-items:flex-start;">
          <div style="flex:2;">
            <p style="font-size:11px;color:var(--fg-muted);margin-bottom:8px;">
              <strong>Topic clusters</strong> (from <code>sciknow catalog cluster</code>).
              Click a cluster to see its papers. Ranked by paper count.
            </p>
            <div id="tl-topics-list" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;"></div>
          </div>
          <div style="flex:1;border-left:1px solid var(--border);padding-left:12px;min-width:180px;">
            <p style="font-size:11px;color:var(--fg-muted);margin-bottom:8px;">
              <strong>Domain tags</strong> (from <code>paper_metadata.domains</code>).
              Empty if no tags are populated for this corpus.
            </p>
            <div id="tl-domains-list" style="display:flex;flex-wrap:wrap;gap:4px;"></div>
          </div>
        </div>
        <div id="tl-topics-papers"></div>
      </div>

      <!-- Corpus tab (Phase 46.E — full expand surface: enrich / expand-citations / expand-author) -->
      <div id="tl-corpus-pane" style="display:none;">
        <p style="font-size:11px;color:var(--fg-muted);margin-bottom:12px;">
          Grow and enrich the paper corpus from the browser. Pick one of
          three modes. All are long-running; the log streams below. Cancel
          with the red button.
        </p>
        <div class="tabs" style="margin-bottom:10px;">
          <button class="tab active" data-ctab="corp-enrich" onclick="switchCorpusTab('corp-enrich')">&#128270; Enrich</button>
          <button class="tab" data-ctab="corp-cites" onclick="switchCorpusTab('corp-cites')">&#127760; Expand (citations)</button>
          <button class="tab" data-ctab="corp-author" onclick="switchCorpusTab('corp-author')">&#128100; Expand by author</button>
        </div>

        <!-- Enrich (metadata) panel -->
        <div id="corp-enrich-pane" style="border:1px solid var(--border);border-radius:6px;padding:10px;">
          <div style="font-size:12px;color:var(--fg-muted);margin-bottom:8px;">
            Fill missing DOIs via Crossref / OpenAlex / arXiv title search.
            Mirrors <code>sciknow db enrich</code>.
          </div>
          <div class="field" style="display:flex;gap:6px;align-items:flex-end;flex-wrap:wrap;">
            <div style="flex:1;min-width:70px;"><label>Limit</label>
              <input type="number" id="tl-enr-limit" value="0" min="0" title="0 = all"></div>
            <div style="flex:1;min-width:80px;"><label>Threshold</label>
              <input type="number" id="tl-enr-thresh" value="0.85" min="0" max="1" step="0.01"></div>
            <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;">
              <input type="checkbox" id="tl-enr-dry"> dry-run
            </label>
          </div>
          <button class="btn-primary" style="margin-top:8px;" onclick="doToolCorpus('enrich')">Run Enrich</button>
        </div>

        <!-- Expand by citations -->
        <div id="corp-cites-pane" style="display:none;border:1px solid var(--border);border-radius:6px;padding:10px;">
          <div style="font-size:12px;color:var(--fg-muted);margin-bottom:8px;">
            Follow references &rarr; download OA PDFs &rarr; ingest.
            Mirrors <code>sciknow db expand</code>.
          </div>
          <div class="field" style="display:flex;gap:6px;align-items:flex-end;flex-wrap:wrap;">
            <div style="flex:1;min-width:70px;"><label>Limit</label>
              <input type="number" id="tl-exp-limit" value="0" min="0"></div>
            <div style="flex:1;min-width:80px;"><label>Workers</label>
              <input type="number" id="tl-exp-workers" value="0" min="0" title="0 = .env default"></div>
            <div style="flex:1;min-width:80px;"><label>Relev. thr</label>
              <input type="number" id="tl-exp-relthr" value="0.0" min="0" max="1" step="0.05"></div>
          </div>
          <div class="field" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:4px;">
            <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;">
              <input type="checkbox" id="tl-exp-dry"> dry-run
            </label>
            <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;">
              <input type="checkbox" id="tl-exp-resolve"> resolve titles
            </label>
            <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;">
              <input type="checkbox" id="tl-exp-ingest" checked> ingest
            </label>
            <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;">
              <input type="checkbox" id="tl-exp-relevance" checked> relevance filter
            </label>
          </div>
          <div class="field" style="margin-top:4px;display:flex;gap:8px;align-items:flex-end;">
            <div style="flex:3;">
              <label>Relevance anchor query (optional)</label>
              <input type="text" id="tl-exp-relq" placeholder="(corpus centroid if blank)">
            </div>
            <div style="flex:2;">
              <label>Anchor from topic</label>
              <select id="tl-exp-relq-topic" onchange="if(this.value){{document.getElementById('tl-exp-relq').value=this.value;}}">
                <option value="">(pick a topic…)</option>
              </select>
            </div>
          </div>
          <button class="btn-primary" style="margin-top:8px;" onclick="doToolCorpus('expand')">Run Expand</button>
        </div>

        <!-- Phase 46.E — Expand by author panel -->
        <div id="corp-author-pane" style="display:none;border:1px solid var(--border);border-radius:6px;padding:10px;">
          <div style="font-size:12px;color:var(--fg-muted);margin-bottom:10px;">
            Find every paper by an author across OpenAlex + Crossref, then
            download the open-access ones and ingest. Mirrors
            <code>sciknow db expand-author</code>. The picker below ranks
            authors by <strong>citation count</strong> (within this
            corpus) then paper count, so the most-authoritative names
            surface first.
          </div>
          <div class="field" style="display:flex;gap:8px;align-items:flex-end;">
            <div style="flex:3;">
              <label>Search author</label>
              <input type="text" id="tl-eauth-q"
                     placeholder="Type a name — e.g. Solanki, Lockwood…"
                     oninput="onExpandAuthorSearchInput(event)"
                     onkeydown="if(event.key==='Enter'){{event.preventDefault();onExpandAuthorSearchInput(event);}}">
            </div>
            <div style="flex:2;">
              <label>ORCID (optional)</label>
              <input type="text" id="tl-eauth-orcid" placeholder="0000-0000-0000-0000">
            </div>
          </div>
          <div id="tl-eauth-results"
               style="max-height:260px;overflow:auto;border:1px solid var(--border);
                      border-radius:4px;margin-top:6px;font-size:12px;display:none;"></div>
          <div id="tl-eauth-selected"
               style="margin-top:6px;font-size:12px;color:var(--fg-muted);">
            No author selected yet — search above and click a row.
          </div>
          <div class="field" style="display:flex;gap:6px;align-items:flex-end;flex-wrap:wrap;margin-top:6px;">
            <div style="flex:1;min-width:80px;"><label>Year from</label>
              <input type="number" id="tl-eauth-yfrom" placeholder="(any)"></div>
            <div style="flex:1;min-width:80px;"><label>Year to</label>
              <input type="number" id="tl-eauth-yto" placeholder="(any)"></div>
            <div style="flex:1;min-width:70px;"><label>Limit</label>
              <input type="number" id="tl-eauth-limit" value="0" min="0" title="0 = all"></div>
            <div style="flex:1;min-width:80px;"><label>Workers</label>
              <input type="number" id="tl-eauth-workers" value="0" min="0"></div>
            <div style="flex:1;min-width:80px;"><label>Relev. thr</label>
              <input type="number" id="tl-eauth-relthr" value="0.0" min="0" max="1" step="0.05"></div>
          </div>
          <div class="field" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:4px;">
            <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;">
              <input type="checkbox" id="tl-eauth-strict" checked> strict author match
            </label>
            <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;">
              <input type="checkbox" id="tl-eauth-all"> keep all matches (skip disamb.)
            </label>
            <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;">
              <input type="checkbox" id="tl-eauth-relevance" checked> relevance filter
            </label>
            <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;">
              <input type="checkbox" id="tl-eauth-ingest" checked> ingest
            </label>
            <label style="display:flex;align-items:center;gap:4px;font-weight:400;font-size:12px;">
              <input type="checkbox" id="tl-eauth-dry"> dry-run
            </label>
          </div>
          <div class="field" style="margin-top:4px;">
            <label>Relevance anchor query (optional)</label>
            <input type="text" id="tl-eauth-relq"
                   placeholder="(corpus centroid if blank)">
          </div>
          <div style="display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap;">
            <button class="btn-primary" onclick="openExpandAuthorPreview()">
              &#128269; Preview candidates
            </button>
            <button class="btn-secondary" onclick="doToolCorpus('expand-author')"
                    title="Skip preview — run the full pipeline with the relevance-filter threshold auto-downloading everything above it. Equivalent to `sciknow db expand-author --relevance`.">
              &#9889; Auto-download (override)
            </button>
            <span style="font-size:11px;color:var(--fg-muted);">
              Preview lets you cherry-pick; Auto uses the relevance threshold.
            </span>
          </div>
        </div>

        <div style="display:flex;gap:8px;align-items:center;margin-top:12px;">
          <div id="tl-corpus-status" style="flex:1;font-size:12px;color:var(--fg-muted);"></div>
          <button id="tl-corpus-cancel" onclick="cancelToolCorpus()"
                  style="display:none;background:var(--danger,#c53030);color:white;border:0;padding:4px 10px;border-radius:4px;cursor:pointer;">
            Cancel
          </button>
        </div>
        <pre id="tl-corpus-log" style="margin-top:8px;max-height:360px;overflow:auto;background:var(--bg-alt,#f8f8f8);border:1px solid var(--border);border-radius:6px;padding:10px;font-size:11px;font-family:ui-monospace,monospace;white-space:pre-wrap;"></pre>
      </div>

    </div>
  </div>
</div>

<!-- Phase 54.6.1 — Expand-by-Author preview modal.
     Shows all candidates found by search (post corpus-dedup), with
     relevance scores annotated. User cherry-picks via checkboxes and
     clicks "Download selected" which hits
     /api/corpus/expand-author/download-selected. -->
<div class="modal-overlay" id="expand-author-preview-modal"
     onclick="if(event.target===this)closeModal('expand-author-preview-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3>&#128269; Expand-by-Author &mdash; Preview Candidates</h3>
      <button class="modal-close" onclick="closeModal('expand-author-preview-modal')">&times;</button>
    </div>
    <div class="modal-body">
      <div id="eap-loading" style="display:none;padding:20px;text-align:center;color:var(--fg-muted);">
        <div style="font-size:14px;">Searching OpenAlex + Crossref&hellip;</div>
        <div style="font-size:11px;margin-top:4px;">(typically 10-30s depending on author's paper count)</div>
      </div>
      <div id="eap-error" style="display:none;padding:10px;background:var(--danger-bg,#fee);color:var(--danger,#c53030);border:1px solid var(--danger,#c53030);border-radius:4px;margin-bottom:10px;"></div>
      <div id="eap-content" style="display:none;">
        <div id="eap-info" style="font-size:12px;color:var(--fg-muted);margin-bottom:10px;padding:8px;background:var(--bg-alt,#f8f8f8);border-radius:4px;"></div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px;">
          <button class="btn-secondary" onclick="eapSelectAll(true)">Select all</button>
          <button class="btn-secondary" onclick="eapSelectAll(false)">Select none</button>
          <label style="font-size:12px;display:flex;align-items:center;gap:4px;">
            Select where score &ge;
            <input type="number" id="eap-threshold" value="0.55" min="0" max="1" step="0.05"
                   style="width:70px;font-size:12px;padding:2px 4px;">
            <button class="btn-secondary" onclick="eapSelectByThreshold()"
                    style="font-size:11px;padding:2px 8px;">Apply</button>
          </label>
          <label style="font-size:12px;display:flex;align-items:center;gap:4px;">
            Sort:
            <select id="eap-sort" onchange="eapRender()"
                    style="font-size:12px;padding:2px 4px;">
              <option value="score">by relevance score</option>
              <option value="year">by year (newest)</option>
              <option value="title">by title (A-Z)</option>
            </select>
          </label>
          <span id="eap-selected-count" style="margin-left:auto;font-size:12px;color:var(--fg-muted);"></span>
        </div>
        <div id="eap-table-wrap"
             style="max-height:360px;overflow:auto;border:1px solid var(--border);border-radius:4px;">
          <table id="eap-table" style="width:100%;border-collapse:collapse;font-size:12px;">
            <thead style="background:var(--bg-alt,#f8f8f8);position:sticky;top:0;z-index:1;">
              <tr>
                <th style="width:32px;padding:6px 8px;text-align:left;">
                  <input type="checkbox" id="eap-header-cb"
                         onchange="eapSelectAll(this.checked)">
                </th>
                <th style="padding:6px 8px;text-align:left;">Title</th>
                <th style="padding:6px 8px;text-align:left;white-space:nowrap;">Authors</th>
                <th style="padding:6px 8px;text-align:left;white-space:nowrap;">Year</th>
                <th style="padding:6px 8px;text-align:left;white-space:nowrap;">Score</th>
              </tr>
            </thead>
            <tbody id="eap-tbody"></tbody>
          </table>
        </div>
        <div style="display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap;">
          <label style="font-size:12px;display:flex;align-items:center;gap:4px;">
            Workers:
            <input type="number" id="eap-workers" value="0" min="0"
                   style="width:60px;font-size:12px;padding:2px 4px;"
                   title="0 = INGEST_WORKERS from .env">
          </label>
          <label style="font-size:12px;display:flex;align-items:center;gap:4px;">
            <input type="checkbox" id="eap-ingest" checked> ingest after download
          </label>
          <button class="btn-primary" id="eap-download-btn"
                  onclick="eapDownloadSelected()" style="margin-left:auto;">
            &#128229; Download selected
          </button>
        </div>
        <div id="eap-status" style="margin-top:8px;font-size:12px;color:var(--fg-muted);"></div>
        <pre id="eap-log" style="margin-top:6px;max-height:260px;overflow:auto;background:var(--bg-alt,#f8f8f8);border:1px solid var(--border);border-radius:4px;padding:8px;font-size:11px;font-family:ui-monospace,monospace;white-space:pre-wrap;display:none;"></pre>
      </div>
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
        <div class="kg-controls">
          <span class="kg-controls-label">Theme</span>
          <button class="kg-theme-chip" data-theme="deep-space"></button>
          <button class="kg-theme-chip" data-theme="paper"></button>
          <button class="kg-theme-chip" data-theme="blueprint"></button>
          <button class="kg-theme-chip" data-theme="solarized"></button>
          <button class="kg-theme-chip" data-theme="solarized-light"></button>
          <button class="kg-theme-chip" data-theme="terminal"></button>
          <button class="kg-theme-chip" data-theme="neon"></button>
          <button class="kg-invert-btn" onclick="invertKgTheme()"
                  title="Swap to the paired light/dark preset">
            &#8646; Invert
          </button>
          <span class="kg-sep"></span>
          <span class="kg-controls-label">Custom</span>
          <label class="kg-color-box" title="Background color (any RGB)">
            <span class="kg-color-tag">BG</span>
            <input type="color" id="kg-color-bg"
                   oninput="kgSetCustomColor('bg', this.value)"
                   value="#060b18"/>
          </label>
          <label class="kg-color-box" title="Letter / label color — stroke auto-contrasts">
            <span class="kg-color-tag">Aa</span>
            <input type="color" id="kg-color-label"
                   oninput="kgSetCustomColor('label', this.value)"
                   value="#e6f0ff"/>
          </label>
          <label class="kg-color-box" title="Edge (connection line) color">
            <span class="kg-color-tag">Ed</span>
            <input type="color" id="kg-color-edge"
                   oninput="kgSetCustomColor('edge', this.value)"
                   value="#7fb6ff"/>
          </label>
          <label class="kg-color-box" title="Node accent color (sphere gradient mid stop)">
            <span class="kg-color-tag">No</span>
            <input type="color" id="kg-color-node"
                   oninput="kgSetCustomColor('node', this.value)"
                   value="#8fc3ff"/>
          </label>
          <button class="kg-invert-btn kg-invert-btn-sm"
                  onclick="kgClearCustomColors()"
                  title="Clear custom colors — revert to the active preset">
            &#8634;
          </button>
          <span class="kg-sep"></span>
          <button id="kg-fullscreen-btn" class="kg-invert-btn"
                  onclick="kgToggleFullscreen()"
                  title="Fill the screen with just the graph pane (Esc to exit)">
            &#9974; Fullscreen
          </button>
          <button class="kg-invert-btn" onclick="kgCopyShareLink()"
                  title="Copy a URL that opens this view — theme, filters, camera, pinned nodes — on any other machine.">
            &#128279; Share
          </button>
        </div>
        <div class="kg-controls kg-controls-2">
          <input type="search" id="kg-search" class="kg-search-input"
                 placeholder="Search nodes…"
                 oninput="kgSearch(this.value)"
                 title="Filter nodes by label (substring). Pulse-matches stay bright, everything else dims."/>
          <label class="kg-chip-group" title="Color scheme for nodes + edges">
            <span class="kg-controls-label">Color</span>
            <select class="kg-select" onchange="kgSetColorBy(this.value)">
              <option value="cluster">Cluster</option>
              <option value="predicate">Predicate family</option>
              <option value="theme">Plain</option>
            </select>
          </label>
          <label class="kg-chip-group" title="Label size">
            <span class="kg-controls-label">Labels</span>
            <input type="range" min="0.5" max="2.0" step="0.05" value="1.0"
                   class="kg-slider" oninput="kgSetLabelScale(this.value)"/>
          </label>
          <label class="kg-chip-group" title="Hide nodes whose degree exceeds this (tames hubs)">
            <span class="kg-controls-label">Max deg</span>
            <input type="range" min="1" max="99" step="1" value="99"
                   class="kg-slider kg-slider-short"
                   oninput="kgSetDegFilter(this.value)"/>
            <span id="kg-degfilter-label" class="kg-controls-label">&infin;</span>
          </label>
          <button class="kg-invert-btn" onclick="kgToggleFreeze()"
                  title="Freeze / resume the force simulation (Space)">
            &#9208; Freeze
          </button>
          <button class="kg-invert-btn" onclick="kgResetView()"
                  title="Re-center camera, unhide all nodes, clear search">
            &#8634; Reset
          </button>
          <button class="kg-invert-btn" onclick="kgDownloadPng()"
                  title="Download the current view as a high-DPI PNG">
            &#128247; PNG
          </button>
        </div>
        <div id="kg-graph-canvas" style="border:1px solid var(--border);border-radius:6px;"></div>
        <p style="font-size:10px;color:var(--fg-muted);margin-top:6px;">
          &middot; Drag the background to orbit &middot; drag a node to reposition &middot; scroll to zoom &middot; click a node to center the camera on it &middot; right-click for a context menu (pin, hide, expand, copy, show paper) &middot; Space freezes / resumes physics.
        </p>
      </div>
      <!-- Phase 48 context menu for node / edge / background right-click -->
      <div id="kg-context-menu" role="menu"></div>
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
// Phase 33 — page-load tag visible in DevTools Console. The build tag
// is the git short hash (or a UTC timestamp if git isn't available).
// If you see an old hash in the console after a deploy, your browser
// is running stale JS — hard-refresh with Ctrl+Shift+R (macOS: Cmd+Shift+R).
console.log('[sciknow] web reader loaded · build {_BUILD_TAG}');

// ── State ─────────────────────────────────────────────────────────────────
let currentDraftId = '{active_id}';
let currentChapterId = '{active_chapter_id}';
let currentSectionType = '{active_section_type}';
let currentJobId = null;
let currentEventSource = null;
// Phase 32.4 — needs to be `let` (was `const`) so deleteSection /
// addSectionToChapter can refresh the in-memory cache after a PUT.
let chaptersData = {chapters_json};

// ── Phase 42: data-action click dispatcher ───────────────────────────────
// Replaces ~20 inline onclick handlers that interpolated variables via
// Python f-strings with HTML-entity escaping (&quot;). That pattern was
// flagged by the Phase 22 audit: mitigated for XSS by _esc(), but
// fragile — the next maintainer could easily forget escaping. It also
// stops us from adopting a strict script-src CSP.
//
// New pattern: every such button carries `data-action="kebab-name"` plus
// one or two `data-*` attrs for its arguments. The single listener below
// looks up the handler in ACTIONS and invokes it with the element (which
// owns its dataset) plus the raw event. Browsers escape data-* values
// automatically, so the injection vector goes away by construction.
//
// Static handlers like `onclick="openPlanModal()"` are left alone — no
// interpolation = no fragility. A future CSP pass can convert them too.
const ACTIONS = {{
  // Cluster 1 — Python-rendered sidebar + comments. Element carries
  // data-chapter-id / data-sec-type / data-draft-id / data-comment-id
  // so the handler doesn't need any interpolation.
  'preview-empty-section': (el) =>
    previewEmptySection(el.dataset.chapterId, el.dataset.secType),
  'start-writing-chapter': (el) =>
    startWritingChapter(el.dataset.chapterId),
  'adopt-orphan-section': (el, e) => {{
    // Adopt/delete buttons sit inside an anchor that navigates to the
    // draft; stop the click from bubbling so the page doesn't change.
    e.preventDefault(); e.stopPropagation();
    adoptOrphanSection(el.dataset.chapterId, el.dataset.secType);
  }},
  'delete-orphan-draft': (el, e) => {{
    e.preventDefault(); e.stopPropagation();
    deleteOrphanDraft(el.dataset.draftId);
  }},
  'resolve-comment': (el) => resolveComment(el.dataset.commentId),

  // Cluster 2 — dashboard heatmap + gaps. The chapter-modal / load-
  // section opens happen directly from the cells. `preview-empty-
  // section` is already registered above and works for both the
  // sidebar and the heatmap callers.
  'open-chapter-modal': (el) => openChapterModal(el.dataset.chapterId),
  'load-section': (el) => loadSection(el.dataset.draftId),
  'write-for-gap': (el) => writeForGap(parseInt(el.dataset.chapterNum, 10)),

  // Cluster 3 — sidebar + section editor. setSectionType wants the
  // clicked element as its second arg so it can toggle `.active` on
  // the right chip; the delegator already passes `el`.
  'set-section-type': (el) => setSectionType(el.dataset.secType, el),
  'add-section-to-chapter': (el) => addSectionToChapter(el.dataset.chapterId),
  'move-section': (el) => moveSection(
    parseInt(el.dataset.sectionIndex, 10),
    parseInt(el.dataset.delta, 10),
  ),
  'remove-section': (el) => removeSection(parseInt(el.dataset.sectionIndex, 10)),

  // Cluster 4 — wiki browser + catalog pagination.
  'open-wiki-page': (el) => openWikiPage(el.dataset.slug),
  'load-wiki-pages': (el) => loadWikiPages(parseInt(el.dataset.page, 10)),
  'ask-about-paper': (el) => askAboutPaper(el.dataset.paperTitle),
  'load-catalog': (el) => loadCatalog(parseInt(el.dataset.page, 10)),

  // Cluster 5 — version history + snapshot bundle/draft restore.
  'select-version': (el) => selectVersion(el.dataset.versionId),
  'restore-bundle': (el) => restoreBundle(el.dataset.snapshotId, el.dataset.scope),
  'diff-snapshot': (el) => diffSnapshot(el.dataset.snapshotId),
  'restore-snapshot': (el) => restoreSnapshot(el.dataset.snapshotId),
}};

document.addEventListener('click', function(e) {{
  const el = e.target.closest('[data-action]');
  if (!el) return;
  const fn = ACTIONS[el.dataset.action];
  if (typeof fn !== 'function') return;
  // Event handlers may return false (legacy form), throw, or not —
  // mirror the inline-onclick semantics so call sites don't change.
  try {{
    const ret = fn(el, e);
    if (ret === false) {{ e.preventDefault(); e.stopPropagation(); }}
  }} catch (exc) {{
    console.error('[sciknow] data-action "' + el.dataset.action + '" failed:', exc);
  }}
}});

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
  // Phase 42 — data-action dispatch (see ACTIONS registry).
  html += '<button class="btn-secondary" style="margin-top:8px;font-size:12px;" data-action="open-chapter-modal" data-chapter-id="' + chId + '">&#9881; Edit chapter scope</button>';
  html += '</div>';

  html += '<p>Once the scope feels right, choose a section type below and click <strong>Write</strong> in the toolbar, or click <strong>Autowrite</strong> to draft a section autonomously with the convergence loop.</p>';
  html += '<div class="empty-section-picker">';
  sections.forEach(s => {{
    const active = s === currentSectionType ? ' active' : '';
    html += '<button class="section-chip' + active + '" data-action="set-section-type" data-sec-type="' + s + '">' + s.replace(/_/g, ' ') + '</button>';
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

// ── Phase 30 / 32.5: persistent global task bar ───────────────────────
//
// Phase 30 first design used a SECOND EventSource on /api/stream/{{id}}
// alongside the per-section preview consumer. Both ended up calling
// queue.get() on the same server-side asyncio.Queue, which REMOVES
// items, so the two consumers split the event stream and the task
// bar saw a tiny (often zero) subset of tokens. Patching the task
// bar's parser/regex/math couldn't fix it — the events weren't
// arriving in the first place.
//
// Phase 32.5 fix: stop using SSE for the task bar. The server now
// tracks token count, rolling tps, elapsed, model name, and stream
// state per job in `_jobs[id]` (server side, plain Python). The task
// bar polls GET /api/jobs/{{id}}/stats every 500ms and reads a
// fixed-shape snapshot. Single source of truth, no race, no parsing.

let _globalJob = null;
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
  document.getElementById('tb-tokens').textContent = (j.tokens || 0).toLocaleString();
  document.getElementById('tb-tps').textContent = (j.tps || 0).toFixed(1);
  document.getElementById('tb-elapsed').textContent = _formatElapsed((j.elapsedS || 0) * 1000);
  // ETA — only when target_words is known and tokens are flowing
  const etaWrap = document.getElementById('tb-eta');
  if (j.targetWords && j.tps > 0.1 && j.tokens > 0) {{
    const remaining = Math.max(0, j.targetWords - j.tokens);
    const etaMs = (remaining / j.tps) * 1000;
    document.getElementById('tb-eta-val').textContent = _formatElapsed(etaMs);
    etaWrap.style.display = 'inline-flex';
  }} else {{
    etaWrap.style.display = 'none';
  }}
  // Dot state
  const dot = document.getElementById('tb-dot');
  dot.className = 'tb-dot ' + (j.state || 'streaming');
}}

// Phase 32.5 — poll the server-side stats endpoint. Replaces the
// previous (broken) SSE consumer. Called every 500ms while a job
// is active.
async function _pollGlobalJobStats(jobId) {{
  if (!_globalJob || _globalJob.id !== jobId) return;
  try {{
    const res = await fetch('/api/jobs/' + jobId + '/stats');
    if (res.status === 410 || res.status === 404) {{
      // Job already finished and was swept by GC, or never existed.
      // Treat as a clean finish so the bar dismisses on its own.
      const j = _globalJob;
      if (j && j.state === 'streaming') {{
        j.state = 'done';
        j.taskDesc = 'Done';
        _renderTaskBar();
      }}
      _finishGlobalJob('done', 2000);
      return;
    }}
    if (!res.ok) return;  // network blip — keep polling
    const stats = await res.json();
    const j = _globalJob;
    if (!j || j.id !== jobId) return;
    // Mirror server snapshot directly into the local state — server
    // is the source of truth, no client-side accounting.
    j.tokens = stats.tokens || 0;
    j.tps = stats.tps || 0;
    j.elapsedS = stats.elapsed_s || 0;
    if (stats.model_name) j.modelName = stats.model_name;
    if (stats.task_desc) j.taskDesc = stats.task_desc;
    if (stats.target_words) j.targetWords = stats.target_words;
    // Lifecycle: if the server says we're done/error, transition.
    if (stats.stream_state === 'done') {{
      j.state = 'done';
      // Don't overwrite a server-supplied "Stopped" / final message
      if (j.taskDesc === stats.task_desc || j.taskDesc === 'Running…') {{
        j.taskDesc = 'Done';
      }}
      _renderTaskBar();
      _finishGlobalJob('done', 4000);
      return;
    }} else if (stats.stream_state === 'error') {{
      j.state = 'error';
      j.taskDesc = 'Error: ' + ((stats.error_message || 'unknown').slice(0, 80));
      _renderTaskBar();
      _finishGlobalJob('error', 0);  // wait for explicit dismiss
      return;
    }}
    _renderTaskBar();
  }} catch (e) {{
    // Network blip — keep polling. Don't surface; the next tick will retry.
  }}
}}

function startGlobalJob(jobId, opts) {{
  if (!jobId) return;
  // Clean up any previous job's poll timer
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
    tps: 0,
    elapsedS: 0,
    state: 'streaming',
    sectionType: (opts && opts.sectionType) || null,
    chapterId: (opts && opts.chapterId) || null,
  }};

  // Show buttons in their starting state
  document.getElementById('tb-stop').style.display = '';
  document.getElementById('tb-dismiss').style.display = 'none';

  _renderTaskBar();

  // Phase 32.5 — kick off the poll loop. 500ms is fast enough that
  // the t/s number feels live but slow enough to be negligible HTTP
  // load (about 2 small JSON requests per second).
  _pollGlobalJobStats(jobId);  // immediate first tick
  _globalJobTimer = setInterval(() => _pollGlobalJobStats(jobId), 500);
}}

function _finishGlobalJob(state, autoDismissMs) {{
  if (!_globalJob) return;
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
  // Phase 32.5 — the next _pollGlobalJobStats tick will see
  // stream_state === 'done' (set by _observe_event_for_stats when
  // the cancelled event flows through _run_generator_in_thread) and
  // call _finishGlobalJob. Safety net: force-dismiss after 5s in
  // case the generator never reaches its next yield.
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
  // Phase 32.5 — only the poll timer needs cleanup now; the broken
  // SSE source approach was removed.
  if (_globalJobTimer) {{
    clearInterval(_globalJobTimer);
    _globalJobTimer = null;
  }}
}}

// ── Phase 30/31: Knowledge Graph browse modal (Graph + Table tabs) ────
async function openKgModal() {{
  // Restore the user's saved palette / overrides BEFORE the first
  // render so the canvas opens in the last theme they used instead
  // of briefly flashing the default. (Chip swatches + color pickers
  // get initialised inside _initKgThemeChips on the first render.)
  _kgLoadPrefs();
  // Phase 48d — if the page was loaded with a #kg=… share URL, pre-
  // fill the filter inputs from its payload so the initial loadKg
  // fetches exactly the same slice. Theme + overrides were already
  // applied by _kgMaybeParseHashOnLoad; camera + pins get applied
  // inside the first render frame via _kgApplyPendingShare.
  const pending = window._kgPendingShare;
  if (pending && pending.f) {{
    const subj = document.getElementById('kg-subject');
    const pred = document.getElementById('kg-predicate');
    const obj  = document.getElementById('kg-object');
    if (subj) subj.value = pending.f.s || '';
    if (pred) {{
      // Predicate options are populated on first loadKg; adding a
      // stub option ensures the current value survives the reload.
      const v = pending.f.p || '';
      if (v && !Array.from(pred.options).some(o => o.value === v)) {{
        const opt = document.createElement('option');
        opt.value = v; opt.textContent = v;
        pred.appendChild(opt);
      }}
      pred.value = v;
    }}
    if (obj)  obj.value  = pending.f.o || '';
  }}
  openModal('kg-modal');
  switchKgTab('kg-graph');
  await loadKg(0);
}}
// Phase 48d — auto-open the modal when the URL has a #kg=… share hash
// so a shared link lands the recipient directly in the graph view.
window.addEventListener('DOMContentLoaded', () => {{
  if (window._kgPendingShare) {{
    try {{ openKgModal(); }} catch (e) {{}}
  }}
}});

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

// Phase 48 — KG color presets. Each preset defines the full palette:
// the canvas gradient background, the node sphere shading (inner →
// mid → outer gradient stops), the highlight (fixed/pinned) gradient,
// the edge line color, and the label fill + outline. `inverse` names
// the paired light/dark preset for the Invert button.
const KG_THEMES = {{
  'deep-space': {{
    name: 'Deep Space',
    canvasBg: '#060b18',
    bgInner: '#1e2a44', bgOuter: '#060b18',
    nodeInner: '#ffffff', nodeMid: '#8fc3ff', nodeOuter: '#0e2a54',
    hiMid: '#ffd98a', hiOuter: '#6a3e00',
    nodeStroke: '#0a1a33',
    edge: '#7fb6ff',
    label: '#e6f0ff', labelStroke: '#040914',
    inverse: 'paper',
  }},
  'paper': {{
    name: 'Paper',
    canvasBg: '#f3f5f9',
    bgInner: '#ffffff', bgOuter: '#d8ddea',
    nodeInner: '#ffffff', nodeMid: '#6c8ec8', nodeOuter: '#1f3a6b',
    hiMid: '#d48f2d', hiOuter: '#7a3f00',
    nodeStroke: '#1f3a6b',
    edge: '#5a7bb0',
    label: '#0a1a33', labelStroke: '#ffffff',
    inverse: 'deep-space',
  }},
  'terminal': {{
    name: 'Terminal',
    canvasBg: '#020402',
    bgInner: '#0a1a0a', bgOuter: '#000000',
    nodeInner: '#f0ffe8', nodeMid: '#55e86b', nodeOuter: '#083015',
    hiMid: '#ffef40', hiOuter: '#5a4a00',
    nodeStroke: '#062008',
    edge: '#3ccc6a',
    label: '#bfffc6', labelStroke: '#000000',
    inverse: 'paper',
  }},
  'blueprint': {{
    name: 'Blueprint',
    canvasBg: '#061530',
    bgInner: '#123466', bgOuter: '#040d22',
    nodeInner: '#ffffff', nodeMid: '#6bf2ff', nodeOuter: '#033a55',
    hiMid: '#ffb347', hiOuter: '#5c3500',
    nodeStroke: '#02101f',
    edge: '#7bd5ff',
    label: '#eafaff', labelStroke: '#02101f',
    inverse: 'paper',
  }},
  'solarized': {{
    name: 'Solarized',
    canvasBg: '#002b36',
    bgInner: '#08414e', bgOuter: '#001820',
    nodeInner: '#fdf6e3', nodeMid: '#b58900', nodeOuter: '#3a2a00',
    hiMid: '#cb4b16', hiOuter: '#4a1e00',
    nodeStroke: '#001820',
    edge: '#268bd2',
    label: '#eee8d5', labelStroke: '#002b36',
    inverse: 'solarized-light',
  }},
  'solarized-light': {{
    name: 'Solarized Light',
    canvasBg: '#fdf6e3',
    bgInner: '#ffffff', bgOuter: '#eee8d5',
    nodeInner: '#fdf6e3', nodeMid: '#b58900', nodeOuter: '#3a2a00',
    hiMid: '#cb4b16', hiOuter: '#4a1e00',
    nodeStroke: '#3a2a00',
    edge: '#268bd2',
    label: '#073642', labelStroke: '#fdf6e3',
    inverse: 'solarized',
  }},
  'neon': {{
    name: 'Neon',
    canvasBg: '#000000',
    bgInner: '#1a0030', bgOuter: '#000000',
    nodeInner: '#ffffff', nodeMid: '#ff3db7', nodeOuter: '#3a0028',
    hiMid: '#2affd5', hiOuter: '#003d3a',
    nodeStroke: '#0a0014',
    edge: '#c66bff',
    label: '#ffd6ff', labelStroke: '#0a0014',
    inverse: 'paper',
  }},
}};
let _kgActiveTheme = 'deep-space';
// Per-user color overrides layered on top of the active preset. Empty
// object = pure preset. Any key in here wins over the same key in
// KG_THEMES[_kgActiveTheme] when the sim reads its colors. Persisted
// to localStorage alongside _kgActiveTheme so the last palette the
// user picked (preset + any custom tweaks) is restored next session.
let _kgCustomOverrides = {{}};

// Choose a readable text color (black or white) given a background hex,
// using the standard luminance formula. Used so that picking a label
// color auto-sets the label stroke to maintain contrast against
// whatever background the node happens to be over.
function _kgContrast(hex) {{
  const m = (hex || '').replace('#', '').trim();
  if (m.length !== 6 && m.length !== 3) return '#000000';
  const full = m.length === 3
    ? m.split('').map(c => c + c).join('')
    : m;
  const r = parseInt(full.slice(0, 2), 16);
  const g = parseInt(full.slice(2, 4), 16);
  const b = parseInt(full.slice(4, 6), 16);
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return lum > 0.55 ? '#0a0a0a' : '#ffffff';
}}

// Darken a hex color by a 0..1 factor (0.7 = 70% of original). Used to
// derive the outer shading stop of a sphere-shaded node when the user
// picks the node's main color.
function _kgDarken(hex, factor) {{
  const m = (hex || '').replace('#', '').trim();
  if (m.length !== 6) return hex;
  const r = Math.max(0, Math.min(255, Math.round(parseInt(m.slice(0, 2), 16) * factor)));
  const g = Math.max(0, Math.min(255, Math.round(parseInt(m.slice(2, 4), 16) * factor)));
  const b = Math.max(0, Math.min(255, Math.round(parseInt(m.slice(4, 6), 16) * factor)));
  const toHex = (v) => v.toString(16).padStart(2, '0');
  return '#' + toHex(r) + toHex(g) + toHex(b);
}}

// Merged palette: active preset with overrides applied on top. Callers
// that need colors should go through this, never read KG_THEMES
// directly, so custom tweaks are respected.
function _kgEffectiveTheme() {{
  const base = KG_THEMES[_kgActiveTheme] || KG_THEMES['deep-space'];
  return Object.assign({{}}, base, _kgCustomOverrides || {{}});
}}

// Persist theme + overrides across sessions. The key is versioned
// (`_v1`) so we can evolve the shape later without loading stale data.
function _kgSavePrefs() {{
  try {{
    localStorage.setItem('kg_prefs_v1', JSON.stringify({{
      theme: _kgActiveTheme,
      overrides: _kgCustomOverrides,
    }}));
  }} catch (e) {{ /* localStorage can throw in private mode */ }}
}}
function _kgLoadPrefs() {{
  try {{
    const s = localStorage.getItem('kg_prefs_v1');
    if (!s) return;
    const p = JSON.parse(s);
    if (p && p.theme && KG_THEMES[p.theme]) _kgActiveTheme = p.theme;
    if (p && p.overrides && typeof p.overrides === 'object') {{
      _kgCustomOverrides = p.overrides;
    }}
  }} catch (e) {{ /* ignore corrupted prefs */ }}
}}

// Push the effective theme into the running simulation without
// restarting it. Called after any preset / override change.
function _kgRefreshLiveTheme() {{
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.refreshTheme) c._kgSim.refreshTheme();
  // Keep the color-picker inputs in sync with the effective theme so
  // reopening the modal doesn't show stale values against a live bg.
  const t = _kgEffectiveTheme();
  const bgPicker = document.getElementById('kg-color-bg');
  const lbPicker = document.getElementById('kg-color-label');
  const edPicker = document.getElementById('kg-color-edge');
  const ndPicker = document.getElementById('kg-color-node');
  if (bgPicker) bgPicker.value = t.canvasBg || t.bgOuter || '#060b18';
  if (lbPicker) lbPicker.value = t.label || '#e6f0ff';
  if (edPicker) edPicker.value = t.edge || '#7fb6ff';
  if (ndPicker) ndPicker.value = t.nodeMid || '#8fc3ff';
}}

// Handler for each color-picker input (BG / Label / Edge / Node). For
// BG we set all three bg stops to the same color (solid fill); for
// Label we auto-derive the stroke as the contrast color so labels
// remain readable. Overrides are saved to localStorage immediately.
function kgSetCustomColor(kind, value) {{
  if (!value || typeof value !== 'string') return;
  if (kind === 'bg') {{
    _kgCustomOverrides.canvasBg = value;
    _kgCustomOverrides.bgInner = value;
    _kgCustomOverrides.bgOuter = value;
  }} else if (kind === 'label') {{
    _kgCustomOverrides.label = value;
    _kgCustomOverrides.labelStroke = _kgContrast(value);
  }} else if (kind === 'edge') {{
    _kgCustomOverrides.edge = value;
  }} else if (kind === 'node') {{
    _kgCustomOverrides.nodeMid = value;
    _kgCustomOverrides.nodeOuter = _kgDarken(value, 0.35);
    _kgCustomOverrides.nodeStroke = _kgDarken(value, 0.2);
  }} else {{
    return;
  }}
  _kgSavePrefs();
  _kgRefreshLiveTheme();
}}

// Wipe all per-user overrides, snapping the live theme back to whatever
// preset is currently active.
function kgClearCustomColors() {{
  _kgCustomOverrides = {{}};
  _kgSavePrefs();
  _kgRefreshLiveTheme();
}}

// Toggle fullscreen for the KG graph pane. Using the pane (not the
// whole modal) means the toolbar + canvas fill the screen while the
// rest of the modal chrome is hidden by the browser's fullscreen UI.
function kgToggleFullscreen() {{
  const target = document.getElementById('kg-graph-pane');
  if (!target) return;
  const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
  if (!isFs) {{
    const req = target.requestFullscreen || target.webkitRequestFullscreen;
    if (req) req.call(target).catch(() => {{}});
  }} else {{
    const ex = document.exitFullscreen || document.webkitExitFullscreen;
    if (ex) ex.call(document).catch(() => {{}});
  }}
}}
// Keep the fullscreen button label accurate across user-initiated
// enter/exit (e.g. ESC or browser chrome).
document.addEventListener('fullscreenchange', () => {{
  const btn = document.getElementById('kg-fullscreen-btn');
  if (!btn) return;
  btn.innerHTML = document.fullscreenElement
    ? '\\u2922 Exit fullscreen'
    : '\\u26F6 Fullscreen';
}});

// Apply a theme's gradient stops to the SVG <defs>. Called on init and
// whenever the user picks a new preset — the render() loop then paints
// next frame with the new palette without restarting the simulation.
function _applyKgDefs(svg, theme) {{
  const defs = svg.querySelector('defs');
  if (!defs) return;
  defs.innerHTML =
    '<radialGradient id="kg-nodeg" cx="30%" cy="30%" r="75%">' +
      '<stop offset="0%" stop-color="' + theme.nodeInner + '"/>' +
      '<stop offset="35%" stop-color="' + theme.nodeMid + '"/>' +
      '<stop offset="100%" stop-color="' + theme.nodeOuter + '"/>' +
    '</radialGradient>' +
    '<radialGradient id="kg-nodeh" cx="30%" cy="30%" r="75%">' +
      '<stop offset="0%" stop-color="' + theme.nodeInner + '"/>' +
      '<stop offset="30%" stop-color="' + theme.hiMid + '"/>' +
      '<stop offset="100%" stop-color="' + theme.hiOuter + '"/>' +
    '</radialGradient>' +
    '<radialGradient id="kg-bg" cx="50%" cy="50%" r="80%">' +
      '<stop offset="0%" stop-color="' + theme.bgInner + '"/>' +
      '<stop offset="100%" stop-color="' + theme.bgOuter + '"/>' +
    '</radialGradient>';
}}

// Switch the active KG theme. Clicking a preset clears any custom
// overrides (the chip is meant as a reset-to-known-good). Safe to
// call before the graph is built — it just updates state that the
// next _renderKgGraph will read.
function setKgTheme(name) {{
  if (!KG_THEMES[name]) return;
  _kgActiveTheme = name;
  _kgCustomOverrides = {{}};
  _kgSavePrefs();
  document.querySelectorAll('.kg-theme-chip').forEach(c => {{
    c.classList.toggle('active', c.getAttribute('data-theme') === name);
  }});
  _kgRefreshLiveTheme();
}}

// One-click swap to the paired light/dark preset of the current theme.
function invertKgTheme() {{
  const cur = KG_THEMES[_kgActiveTheme];
  if (cur && cur.inverse) setKgTheme(cur.inverse);
}}

// Paint each theme chip with a tiny radial preview of its palette
// (one-time; chips live inside the modal markup). Also loads the
// persisted prefs from the last session so the initial render uses
// the user's saved theme + overrides instead of the defaults.
function _initKgThemeChips() {{
  if (window._kgChipsReady) return;
  _kgLoadPrefs();
  document.querySelectorAll('.kg-theme-chip').forEach(chip => {{
    const name = chip.getAttribute('data-theme');
    const t = KG_THEMES[name];
    if (!t) return;
    chip.style.background =
      'radial-gradient(circle at 35% 35%, ' +
      t.nodeMid + ' 0%, ' + t.nodeOuter + ' 55%, ' +
      t.bgOuter + ' 100%)';
    chip.title = t.name;
    chip.classList.toggle('active', name === _kgActiveTheme);
    chip.addEventListener('click', () => setKgTheme(name));
  }});
  // Seed the color pickers from the effective theme.
  const t = _kgEffectiveTheme();
  const pairs = [
    ['kg-color-bg', t.canvasBg],
    ['kg-color-label', t.label],
    ['kg-color-edge', t.edge],
    ['kg-color-node', t.nodeMid],
  ];
  pairs.forEach(([id, val]) => {{
    const el = document.getElementById(id);
    if (el && val) el.value = val;
  }});
  window._kgChipsReady = true;
}}

// Phase 48b — Okabe-Ito-derived palette for Louvain community coloring.
// Colorblind-safe, eight distinct hues; paired (mid, outer) stops give
// each cluster a sphere-shaded gradient without a full per-theme
// cluster palette. The 9th entry is a neutral gray for overflow.
const KG_CLUSTER_PALETTE = [
  {{ mid: '#56B4E9', outer: '#0c3850' }},  // sky blue
  {{ mid: '#E69F00', outer: '#50300c' }},  // orange
  {{ mid: '#009E73', outer: '#0a3d2c' }},  // bluish green
  {{ mid: '#F0E442', outer: '#504a0c' }},  // yellow
  {{ mid: '#0072B2', outer: '#001f3a' }},  // blue
  {{ mid: '#D55E00', outer: '#3d1a00' }},  // vermillion
  {{ mid: '#CC79A7', outer: '#3e1f32' }},  // reddish purple
  {{ mid: '#8ed6a5', outer: '#234a31' }},  // mint
  {{ mid: '#999999', outer: '#333333' }},  // overflow gray
];

// Predicate family → semantic category → color. Follows VOWL/WebVOWL
// convention of grouping predicates by what they *do* rather than
// coloring every predicate separately (unreadable above ~8 hues).
const KG_PREDICATE_FAMILIES = {{
  causal:      {{ color: '#D55E00', glyph: '\\u2192',
                 keys: ['cause', 'increase', 'decrease', 'induce', 'affect',
                        'drive', 'reduce', 'enhance', 'inhibit', 'trigger',
                        'lead to', 'result in', 'contribute', 'produce'] }},
  measurement: {{ color: '#0072B2', glyph: '\\u2248',
                 keys: ['measure', 'observe', 'detect', 'record', 'sample',
                        'estimate', 'quantify', 'report', 'find', 'show'] }},
  taxonomic:   {{ color: '#009E73', glyph: '\\u2282',
                 keys: ['is a', 'is-a', 'type', 'subtype', 'part of',
                        'part-of', 'contains', 'includes', 'belongs',
                        'kind of', 'category'] }},
  compositional:{{ color: '#8ed6a5', glyph: '\\u25c6',
                 keys: ['uses', 'use', 'composed', 'consists', 'built',
                        'based on', 'based-on', 'relies', 'applies'] }},
  citational:  {{ color: '#999999', glyph: '\\u00a7',
                 keys: ['cite', 'reference', 'evidence', 'support',
                        'contradict', 'agree', 'disagree', 'extend'] }},
  other:       {{ color: '#CC79A7', glyph: '\\u2022', keys: [] }},
}};

function _kgPredicateFamily(pred) {{
  const p = (pred || '').toLowerCase();
  for (const fam of ['causal', 'measurement', 'taxonomic', 'compositional', 'citational']) {{
    for (const key of KG_PREDICATE_FAMILIES[fam].keys) {{
      if (p.indexOf(key) !== -1) return fam;
    }}
  }}
  return 'other';
}}

// Louvain community detection (one-level local-moving pass). Input:
// number of nodes + edges with source/target/count. Output: array of
// community index per node, densely reindexed from 0. Fast enough for
// n ≤ 500 (≈ ms on a modern laptop); the one-level variant skips the
// aggregation step but still produces clusters good enough to drive
// node coloring + gravity wells for our scale.
function _kgLouvain(numNodes, edges) {{
  if (numNodes === 0) return [];
  // Build weighted undirected adjacency
  const adj = [];
  for (let i = 0; i < numNodes; i++) adj.push(new Map());
  let m2 = 0;
  edges.forEach(e => {{
    if (e.source === e.target) return;
    const w = Math.max(1, e.count || 1);
    adj[e.source].set(e.target, (adj[e.source].get(e.target) || 0) + w);
    adj[e.target].set(e.source, (adj[e.target].get(e.source) || 0) + w);
    m2 += 2 * w;
  }});
  if (m2 === 0) return new Array(numNodes).fill(0);
  const degree = new Array(numNodes);
  for (let i = 0; i < numNodes; i++) {{
    let d = 0;
    for (const w of adj[i].values()) d += w;
    degree[i] = d;
  }}
  const community = new Array(numNodes);
  for (let i = 0; i < numNodes; i++) community[i] = i;
  const commSum = {{}};
  for (let i = 0; i < numNodes; i++) commSum[i] = degree[i];

  let changed = true, iter = 0;
  while (changed && iter < 12) {{
    changed = false; iter++;
    for (let i = 0; i < numNodes; i++) {{
      const cur = community[i];
      // Sum of weights from i to each neighboring community
      const wToComm = new Map();
      for (const [j, w] of adj[i]) {{
        const c = community[j];
        wToComm.set(c, (wToComm.get(c) || 0) + w);
      }}
      // Temporarily remove i from its community for the gain calc
      commSum[cur] -= degree[i];
      let bestC = cur, bestGain = 0;
      const kI = degree[i];
      for (const [c, kIin] of wToComm) {{
        if (c === cur) continue;
        const sigmaTot = commSum[c] || 0;
        const gain = kIin - (sigmaTot * kI) / m2;
        if (gain > bestGain) {{ bestGain = gain; bestC = c; }}
      }}
      // Stay in cur if no better community
      commSum[bestC] = (commSum[bestC] || 0) + degree[i];
      if (bestC !== cur) {{ community[i] = bestC; changed = true; }}
    }}
  }}
  // Relabel to dense 0..k
  const remap = new Map();
  let k = 0;
  const result = new Array(numNodes);
  for (let i = 0; i < numNodes; i++) {{
    const c = community[i];
    if (!remap.has(c)) remap.set(c, k++);
    result[i] = remap.get(c);
  }}
  return result;
}}

// Inject per-cluster gradient defs into the SVG so node `fill` can
// reference `url(#kg-cluster-N)`. Called once after Louvain runs and
// again on theme changes (so the gradient inner/stroke stays coherent).
function _applyKgClusterDefs(svg, theme, numClusters) {{
  const defs = svg.querySelector('defs');
  if (!defs) return;
  let extra = '';
  for (let i = 0; i < numClusters; i++) {{
    const p = KG_CLUSTER_PALETTE[i % KG_CLUSTER_PALETTE.length];
    extra +=
      '<radialGradient id="kg-cluster-' + i + '" cx="30%" cy="30%" r="75%">' +
        '<stop offset="0%" stop-color="' + theme.nodeInner + '"/>' +
        '<stop offset="35%" stop-color="' + p.mid + '"/>' +
        '<stop offset="100%" stop-color="' + p.outer + '"/>' +
      '</radialGradient>';
  }}
  // Append to the existing defs (which _applyKgDefs already populated)
  defs.insertAdjacentHTML('beforeend', extra);
}}

// Cubic ease-in-out for the center-on-node camera tween.
function _kgEase(t) {{
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}}

// Render/hide the floating context menu. The <div> lives in the modal
// HTML so any menu action has access to the current simulation closure
// via the global canvas._kgSim handle.
function _kgShowMenu(x, y, items) {{
  const menu = document.getElementById('kg-context-menu');
  if (!menu) return;
  menu.innerHTML = '';
  items.forEach(item => {{
    if (item === '-') {{
      const sep = document.createElement('div');
      sep.className = 'kg-menu-sep';
      menu.appendChild(sep);
      return;
    }}
    const btn = document.createElement('button');
    btn.className = 'kg-menu-item';
    btn.textContent = item.label;
    if (item.hint) {{
      const hint = document.createElement('span');
      hint.className = 'kg-menu-hint';
      hint.textContent = item.hint;
      btn.appendChild(hint);
    }}
    btn.addEventListener('click', () => {{
      _kgHideMenu();
      try {{ item.onClick(); }} catch (e) {{ console.error(e); }}
    }});
    menu.appendChild(btn);
  }});
  // Position; clamp to viewport
  menu.style.display = 'block';
  const mw = menu.offsetWidth, mh = menu.offsetHeight;
  const vw = window.innerWidth, vh = window.innerHeight;
  menu.style.left = Math.min(x, vw - mw - 8) + 'px';
  menu.style.top  = Math.min(y, vh - mh - 8) + 'px';
}}
function _kgHideMenu() {{
  const menu = document.getElementById('kg-context-menu');
  if (menu) menu.style.display = 'none';
}}
// Dismiss the menu on any outside click or Escape
document.addEventListener('click', (e) => {{
  const menu = document.getElementById('kg-context-menu');
  if (menu && menu.style.display === 'block' && !menu.contains(e.target)) {{
    _kgHideMenu();
  }}
}});
document.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape') _kgHideMenu();
}});

// Toolbar click/input handlers. Defined at module scope so the HTML can
// reference them via onclick; each forwards to the live simulation via
// canvas._kgSim.
function kgToggleFreeze() {{
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.togglePause) c._kgSim.togglePause();
}}
function kgResetView() {{
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.resetView) c._kgSim.resetView();
}}
function kgDownloadPng() {{
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.downloadPng) c._kgSim.downloadPng();
}}
function kgSetColorBy(mode) {{
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.setColorBy) c._kgSim.setColorBy(mode);
}}
function kgSetLabelScale(v) {{
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.setLabelScale) c._kgSim.setLabelScale(parseFloat(v));
}}
function kgSetDegFilter(v) {{
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.setDegFilter) c._kgSim.setDegFilter(parseInt(v, 10));
  const lbl = document.getElementById('kg-degfilter-label');
  if (lbl) lbl.textContent = (parseInt(v, 10) >= 99 ? '∞' : v);
}}
function kgSearch(q) {{
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.search) c._kgSim.search(q);
}}

// Phase 48d — cached layout per filter. Stores the final node
// positions for a given filter combination in localStorage so
// re-opening the same view warm-starts from the prior layout instead
// of re-running the full settle from random seeds every time.

function _kgLayoutKey() {{
  const s = (document.getElementById('kg-subject') || {{}}).value || '';
  const p = (document.getElementById('kg-predicate') || {{}}).value || '';
  const o = (document.getElementById('kg-object') || {{}}).value || '';
  // Compact base64 without +/= so the key stays URL-safe and short.
  const raw = JSON.stringify({{ s: s, p: p, o: o }});
  try {{
    return 'kg_layout_' + btoa(raw).replace(/[+/=]/g, '').slice(0, 40);
  }} catch (e) {{
    return 'kg_layout_default';
  }}
}}
function _kgLoadLayout(key) {{
  try {{
    const s = localStorage.getItem(key);
    if (!s) return null;
    const d = JSON.parse(s);
    return (d && d.nodes) || null;
  }} catch (e) {{ return null; }}
}}
function _kgSaveLayout(key, nodes) {{
  const positions = nodes.map(n => ({{
    l: n.label,
    p: [Math.round(n.x), Math.round(n.y), Math.round(n.z)],
  }}));
  const payload = JSON.stringify({{ nodes: positions, ts: Date.now() }});
  try {{
    localStorage.setItem(key, payload);
  }} catch (e) {{
    // Storage full → evict aggressively then retry once.
    _kgEvictOldLayouts(5);
    try {{ localStorage.setItem(key, payload); }} catch (e2) {{ /* give up */ }}
  }}
  _kgEvictOldLayouts(30);
}}
function _kgEvictOldLayouts(maxCount) {{
  const keys = [];
  for (let i = 0; i < localStorage.length; i++) {{
    const k = localStorage.key(i);
    if (!k || k.indexOf('kg_layout_') !== 0) continue;
    try {{
      const ts = (JSON.parse(localStorage.getItem(k)) || {{}}).ts || 0;
      keys.push([k, ts]);
    }} catch (e) {{ /* skip corrupt */ }}
  }}
  keys.sort((a, b) => a[1] - b[1]);
  while (keys.length > maxCount) {{
    const [k] = keys.shift();
    try {{ localStorage.removeItem(k); }} catch (e) {{}}
  }}
}}

// ── Shareable URL ─────────────────────────────────────────────────
// Encodes theme + overrides + filter fields + camera + pinned node
// labels in the URL hash as compact base64 JSON. Anyone who opens
// the URL gets the KG modal auto-opened on exactly the same view.

function kgCopyShareLink() {{
  const c = document.getElementById('kg-graph-canvas');
  if (!c || !c._kgSim || !c._kgSim.getShareState) return;
  const st = c._kgSim.getShareState();
  st.t = _kgActiveTheme;
  st.o = _kgCustomOverrides;
  st.f = {{
    s: (document.getElementById('kg-subject') || {{}}).value || '',
    p: (document.getElementById('kg-predicate') || {{}}).value || '',
    o: (document.getElementById('kg-object') || {{}}).value || '',
  }};
  let enc;
  try {{
    enc = btoa(unescape(encodeURIComponent(JSON.stringify(st))));
  }} catch (e) {{ return; }}
  const url = window.location.origin + window.location.pathname + '#kg=' + enc;
  try {{
    navigator.clipboard.writeText(url).then(() => {{
      const status = document.getElementById('kg-status');
      if (status) status.textContent = 'Share link copied to clipboard.';
    }}, () => window.prompt('Copy this link:', url));
  }} catch (e) {{ window.prompt('Copy this link:', url); }}
}}
// Parse #kg=… on load; if present, stash the state and have openKgModal
// apply it after the first render completes.
(function _kgMaybeParseHashOnLoad() {{
  const m = (window.location.hash || '').match(/^#kg=(.+)$/);
  if (!m) return;
  try {{
    const st = JSON.parse(decodeURIComponent(escape(atob(m[1]))));
    window._kgPendingShare = st;
    if (st.t && typeof KG_THEMES !== 'undefined' && KG_THEMES[st.t]) {{
      _kgActiveTheme = st.t;
    }}
    if (st.o) _kgCustomOverrides = st.o;
  }} catch (e) {{ /* ignore bad hash */ }}
}})();

// Apply a pending share state (filter + cam + pinned) after the sim
// has rendered at least once. Called from the first render() frame.
function _kgApplyPendingShare() {{
  const st = window._kgPendingShare;
  if (!st) return;
  window._kgPendingShare = null;
  // Filter fields were already set before the load via openKgModal's
  // hook; re-apply camera + pins now that the sim exists.
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.applyShareState) {{
    c._kgSim.applyShareState(st);
  }}
}}

// Phase 48 — interactive 3D knowledge graph. Every unique entity is a
// node in a real 3D world; triples are weighted edges (same-pair
// triples merged into one visual edge with count + family info). A
// continuous rAF loop runs a ForceAtlas2-derived physics model
// (log-weighted attraction with dissuade-hubs, repulsion scaled by
// degree, per-cluster gravity wells) + a 3D orbit camera. Interactions
// include drag-to-orbit, drag-nodes, wheel-zoom, hover-dim-1-hop,
// spacebar-freeze, right-click context menu, search, center-on-click,
// PNG export. Palette + coloring mode are swappable live without
// restarting the simulation. No extra deps — still pure SVG.
function _renderKgGraph(triples) {{
  const canvas = document.getElementById('kg-graph-canvas');
  if (!canvas) return;
  _initKgThemeChips();
  if (canvas._kgSim) {{ try {{ canvas._kgSim.stop(); }} catch (e) {{}} }}
  canvas.innerHTML = '';
  if (!triples || triples.length === 0) {{
    canvas.innerHTML = '<div style="padding:80px 24px;text-align:center;color:var(--fg-muted);font-size:12px;">No triples match your filter.</div>';
    return;
  }}

  const W = canvas.clientWidth || 800;
  const H = 520;

  // ── Build nodes + aggregated edges ─────────────────────────────
  // Multiple triples between the same (subject, object) pair collapse
  // into one logical edge with a `count` and the list of source
  // triples (so right-click → "source paper" still works for each
  // underlying claim). Direction is preserved — a→b and b→a stay
  // separate edges and get opposite curvature offsets.
  const nodeIndex = new Map();
  const nodes = [];
  function ensureNode(label) {{
    if (!nodeIndex.has(label)) {{
      nodeIndex.set(label, nodes.length);
      nodes.push({{
        id: nodes.length, label: label,
        x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0,
        fixed: false, hidden: false, degree: 0, cluster: 0,
      }});
    }}
    return nodeIndex.get(label);
  }}
  const edgeMap = new Map();  // "s→t" → edge
  triples.forEach(t => {{
    const sLab = (t.subject || '').substring(0, 60);
    const oLab = (t.object  || '').substring(0, 60);
    if (!sLab || !oLab || sLab === oLab) return;
    const s = ensureNode(sLab), o = ensureNode(oLab);
    const key = s + '\\u2192' + o;
    let e = edgeMap.get(key);
    if (!e) {{
      e = {{ source: s, target: o, count: 0,
             triples: [], families: new Set(), predicates: new Set() }};
      edgeMap.set(key, e);
    }}
    e.count++;
    e.predicates.add(t.predicate || '');
    e.families.add(_kgPredicateFamily(t.predicate || ''));
    e.triples.push({{
      predicate: t.predicate || '',
      doc_id: t.source_doc_id,
      doc_title: t.source_title,
      confidence: t.confidence,
      // Phase 48d — per-triple verbatim sentence from the source
      // paper; may be null for rows ingested before migration 0019.
      source_sentence: t.source_sentence || '',
    }});
  }});
  const edges = Array.from(edgeMap.values());
  edges.forEach(e => {{ nodes[e.source].degree++; nodes[e.target].degree++; }});
  function nodeLabel(label) {{
    return label.length > 24 ? label.substring(0, 24) + '\\u2026' : label;
  }}

  // ── Community detection (Louvain, undirected, count-weighted) ──
  const communities = _kgLouvain(nodes.length, edges);
  const numClusters = Math.max(1, (communities.length
    ? Math.max.apply(null, communities) + 1 : 1));
  nodes.forEach((n, i) => {{ n.cluster = communities[i] || 0; }});

  // Place cluster centroids on a sphere (Fibonacci lattice), scaled by
  // √numClusters so the per-well radius stays roughly constant.
  const clusterCenters = [];
  const wellR = 180 + 30 * Math.sqrt(numClusters);
  for (let c = 0; c < numClusters; c++) {{
    const frac = (c + 0.5) / numClusters;
    const phi = Math.acos(1 - 2 * frac);
    const theta = Math.PI * (1 + Math.sqrt(5)) * c;
    clusterCenters.push({{
      x: wellR * Math.sin(phi) * Math.cos(theta),
      y: wellR * Math.sin(phi) * Math.sin(theta),
      z: wellR * Math.cos(phi),
    }});
  }}
  // Seed node positions inside their cluster so layout converges fast
  nodes.forEach(n => {{
    const c = clusterCenters[n.cluster % clusterCenters.length];
    n.x = c.x + (Math.random() - 0.5) * 80;
    n.y = c.y + (Math.random() - 0.5) * 80;
    n.z = c.z + (Math.random() - 0.5) * 80;
  }});
  // Phase 48d — warm-start from the cached layout for this filter, if
  // we have one. Only positions saved under the *current* filter hash
  // apply; nodes that didn't exist in the cached layout keep their
  // freshly-seeded cluster coords (the layout still settles, just
  // faster and with less drift between views).
  const _kgLayoutKeyCurrent = _kgLayoutKey();
  const _kgLayoutCached = _kgLoadLayout(_kgLayoutKeyCurrent);
  if (_kgLayoutCached && Array.isArray(_kgLayoutCached)) {{
    const byLabel = new Map();
    _kgLayoutCached.forEach(p => byLabel.set(p.l, p.p));
    nodes.forEach(n => {{
      const pos = byLabel.get(n.label);
      if (pos && pos.length === 3) {{
        n.x = pos[0]; n.y = pos[1]; n.z = pos[2];
      }}
    }});
  }}

  // ── Curve-bundle offsets (parallel + bidirectional edges) ──────
  // For each unordered pair {{u,v}}, count how many edges connect them
  // and assign each an offset index so they fan out around the line.
  // A→B and B→A flip sign of the offset so they sit on opposite sides.
  const bundleCount = new Map();
  edges.forEach(e => {{
    const lo = Math.min(e.source, e.target);
    const hi = Math.max(e.source, e.target);
    const pair = lo + '|' + hi;
    const n = bundleCount.get(pair) || 0;
    e._pair = pair;
    e._bundleIdx = n;
    e._dirSign = (e.source < e.target) ? 1 : -1;
    bundleCount.set(pair, n + 1);
  }});
  edges.forEach(e => {{
    const total = bundleCount.get(e._pair);
    const mid = (total - 1) / 2;
    // Bundle offset lives in screen-space px; ±15 per lane works well
    e._offset = (e._bundleIdx - mid) * 15 * e._dirSign;
  }});

  // ── Camera + view state ────────────────────────────────────────
  const cam = {{ rotX: -0.22, rotY: 0.55, dist: 850, fov: 680 }};
  const camDefault = {{ rotX: -0.22, rotY: 0.55, dist: 850 }};
  let theme = _kgEffectiveTheme();
  let colorBy = 'cluster';  // 'cluster' | 'predicate' | 'theme'
  let labelScale = 1.0;
  let degFilter = 999;       // max degree; nodes above this are hidden
  let hoverId = -1;          // node id under the mouse (-1 if none)
  let hoverNeighbors = null; // Set<id> of 1-hop neighbors when hovering
  let hoverEdgeSet = null;   // Set<edge-index> of edges incident to hover
  let searchMatches = null;  // Set<id> of nodes matching the live search
  let running = true, paused = false, raf = null;

  // ── SVG scaffold ───────────────────────────────────────────────
  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('viewBox', (-W/2) + ' ' + (-H/2) + ' ' + W + ' ' + H);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.innerHTML =
    '<defs></defs>' +
    '<rect class="kg-bg" x="' + (-W/2) + '" y="' + (-H/2) + '" width="' + W +
      '" height="' + H + '" fill="url(#kg-bg)" pointer-events="all"/>' +
    '<g class="kg-edges"></g><g class="kg-nodes"></g>';
  canvas.appendChild(svg);
  _applyKgDefs(svg, theme);
  _applyKgClusterDefs(svg, theme, numClusters);
  canvas.style.background = theme.canvasBg;
  const edgeLayer = svg.querySelector('.kg-edges');
  const nodeLayer = svg.querySelector('.kg-nodes');

  // ── Projection ─────────────────────────────────────────────────
  function project(n) {{
    const cy = Math.cos(cam.rotY), sy = Math.sin(cam.rotY);
    const cx = Math.cos(cam.rotX), sxA = Math.sin(cam.rotX);
    const xr = n.x * cy + n.z * sy;
    const zr = -n.x * sy + n.z * cy;
    const yr = n.y;
    const yc = yr * cx - zr * sxA;
    const zc = yr * sxA + zr * cx;
    const zcam = zc + cam.dist;
    if (zcam <= 1) return {{ sx: 0, sy: 0, zcam: 1, scale: 0.0001 }};
    const scale = cam.fov / zcam;
    return {{ sx: xr * scale, sy: yc * scale, zcam: zcam, scale: scale }};
  }}
  function worldDelta(dx, dy, zcam) {{
    const scale = cam.fov / Math.max(zcam, 1);
    const dxc = dx / scale, dyc = dy / scale;
    const cx = Math.cos(cam.rotX), sxA = Math.sin(cam.rotX);
    const cy = Math.cos(cam.rotY), sy = Math.sin(cam.rotY);
    const xr = dxc, yr = dyc * cx, zr = -dyc * sxA;
    return {{ x: xr * cy - zr * sy, y: yr, z: xr * sy + zr * cy }};
  }}

  // ── ForceAtlas2-derived physics ────────────────────────────────
  // Repulsion scales with (deg+1)(deg+1) so hubs push each other away
  // strongly (the FA2 "dissuade-hubs" trick). Attraction uses log(1+d)
  // — the linLog mode of Noack/FA2 — which gives dramatically better
  // hub separation than d² on real citation graphs. Edge weight is
  // log(1+count) so a single high-count merged edge can't collapse
  // the layout. A weak attractor at each Louvain centroid keeps
  // communities visually together.
  const KR = 120;             // repulsion strength
  const KA = 0.08;            // attraction strength
  const KW = 0.0025;          // cluster-well strength
  const KC = 0.0004;          // origin-centering strength
  const DAMP = 0.78;
  function step() {{
    if (paused) return;
    for (let i = 0; i < nodes.length; i++) {{
      nodes[i].ax = 0; nodes[i].ay = 0; nodes[i].az = 0;
    }}
    // Repulsion (O(n²); fine ≤ 500 nodes)
    for (let i = 0; i < nodes.length; i++) {{
      if (nodes[i].hidden) continue;
      for (let j = i + 1; j < nodes.length; j++) {{
        if (nodes[j].hidden) continue;
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        const d2 = dx*dx + dy*dy + dz*dz + 1;
        const d = Math.sqrt(d2);
        const k = KR * (a.degree + 1) * (b.degree + 1);
        const f = k / d2;
        const ux = dx/d, uy = dy/d, uz = dz/d;
        a.ax += ux*f; a.ay += uy*f; a.az += uz*f;
        b.ax -= ux*f; b.ay -= uy*f; b.az -= uz*f;
      }}
    }}
    // Attraction (linLog, log-count-weighted, /degree for hub dissuade)
    edges.forEach(e => {{
      const a = nodes[e.source], b = nodes[e.target];
      if (a.hidden || b.hidden) return;
      const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
      const d = Math.sqrt(dx*dx + dy*dy + dz*dz) + 0.01;
      const w = Math.log(1 + e.count);
      const f = KA * w * Math.log(1 + d);
      const ux = dx/d, uy = dy/d, uz = dz/d;
      const fa = f / Math.max(a.degree + 1, 1);
      const fb = f / Math.max(b.degree + 1, 1);
      a.ax += ux * fa; a.ay += uy * fa; a.az += uz * fa;
      b.ax -= ux * fb; b.ay -= uy * fb; b.az -= uz * fb;
    }});
    // Cluster gravity wells + weak origin pull
    nodes.forEach(n => {{
      if (n.hidden) return;
      const c = clusterCenters[n.cluster % clusterCenters.length];
      n.ax += (c.x - n.x) * KW;
      n.ay += (c.y - n.y) * KW;
      n.az += (c.z - n.z) * KW;
      n.ax -= n.x * KC;
      n.ay -= n.y * KC;
      n.az -= n.z * KC;
      if (n.fixed) {{ n.vx = 0; n.vy = 0; n.vz = 0; return; }}
      n.vx = (n.vx + n.ax) * DAMP;
      n.vy = (n.vy + n.ay) * DAMP;
      n.vz = (n.vz + n.az) * DAMP;
      n.x += n.vx; n.y += n.vy; n.z += n.vz;
    }});
  }}

  // ── Render ─────────────────────────────────────────────────────
  function edgeColor(e) {{
    if (colorBy === 'predicate') {{
      // If the edge has multiple families, pick the first by family
      // priority order so the color is stable across frames
      const prio = ['causal', 'measurement', 'taxonomic', 'compositional', 'citational', 'other'];
      for (const fam of prio) {{
        if (e.families.has(fam)) return KG_PREDICATE_FAMILIES[fam].color;
      }}
      return KG_PREDICATE_FAMILIES.other.color;
    }}
    return theme.edge;
  }}
  function nodeFill(n) {{
    if (n.fixed) return 'url(#kg-nodeh)';
    if (colorBy === 'cluster' && numClusters > 1) {{
      return 'url(#kg-cluster-' + (n.cluster % KG_CLUSTER_PALETTE.length) + ')';
    }}
    return 'url(#kg-nodeg)';
  }}
  function nodeVisible(n) {{
    if (n.hidden) return false;
    if (n.degree > degFilter) return false;
    return true;
  }}

  function render() {{
    const proj = nodes.map(n => ({{ n: n, p: project(n) }}));
    const order = proj.slice().sort((a, b) => b.p.zcam - a.p.zcam);

    // Dim rules: while hovering a node, non-neighbors fade; same for
    // search matches. The effects compose — a match that isn't a
    // hover-neighbor is both search-boosted and hover-dimmed.
    function nodeDim(id) {{
      let k = 1.0;
      if (hoverNeighbors && !hoverNeighbors.has(id)) k *= 0.18;
      if (searchMatches && !searchMatches.has(id)) k *= 0.35;
      return k;
    }}
    function edgeDim(ei, srcId, tgtId) {{
      let k = 1.0;
      if (hoverEdgeSet && !hoverEdgeSet.has(ei)) k *= 0.15;
      if (searchMatches && !searchMatches.has(srcId) && !searchMatches.has(tgtId)) k *= 0.35;
      return k;
    }}

    // Edges as curved quadratic Béziers
    let eHtml = '';
    edges.forEach((e, ei) => {{
      const a = nodes[e.source], b = nodes[e.target];
      if (!nodeVisible(a) || !nodeVisible(b)) return;
      const pa = proj[e.source].p, pb = proj[e.target].p;
      const dxs = pb.sx - pa.sx, dys = pb.sy - pa.sy;
      const len = Math.sqrt(dxs*dxs + dys*dys) + 0.01;
      const nx = -dys / len, ny = dxs / len;
      const mx = (pa.sx + pb.sx) / 2 + nx * e._offset;
      const my = (pa.sy + pb.sy) / 2 + ny * e._offset;
      const avg = (pa.zcam + pb.zcam) / 2;
      const baseOp = Math.max(0.12, Math.min(0.78, 900 / avg));
      const op = baseOp * edgeDim(ei, e.source, e.target);
      const w = Math.max(0.5, Math.min(3.2,
                    0.9 * Math.min(pa.scale, pb.scale) *
                    (1 + Math.log(1 + e.count))));
      // Phase 48d — native SVG <title> gives a 1-line browser tooltip
      // on edge hover showing the source sentence (if any). Zero
      // runtime cost; no custom popover to maintain.
      const tip = (e.triples.find(t => t.source_sentence) || {{}}).source_sentence || '';
      const tipAttr = tip ? '<title>' + escapeHtml(tip) + '</title>' : '';
      eHtml += '<path class="kg-edge" data-ei="' + ei + '" ' +
               'd="M ' + pa.sx.toFixed(1) + ' ' + pa.sy.toFixed(1) +
               ' Q ' + mx.toFixed(1) + ' ' + my.toFixed(1) +
               ' ' + pb.sx.toFixed(1) + ' ' + pb.sy.toFixed(1) + '" ' +
               'stroke="' + edgeColor(e) + '" stroke-width="' + w.toFixed(2) +
               '" fill="none" opacity="' + op.toFixed(2) + '">' +
               tipAttr + '</path>';
      // Count badge for merged edges (3+ triples)
      if (e.count >= 3 && pa.scale > 0.5) {{
        const bx = mx, by = my;
        eHtml += '<circle cx="' + bx.toFixed(1) + '" cy="' + by.toFixed(1) +
                 '" r="6" fill="' + theme.canvasBg +
                 '" stroke="' + edgeColor(e) + '" stroke-width="1" opacity="' +
                 (op * 1.3).toFixed(2) + '" pointer-events="none"/>';
        eHtml += '<text x="' + bx.toFixed(1) + '" y="' + (by + 2.5).toFixed(1) +
                 '" text-anchor="middle" font-size="8" fill="' + theme.label +
                 '" pointer-events="none" style="font-family:var(--font-mono);">' +
                 e.count + '</text>';
      }}
    }});
    edgeLayer.innerHTML = eHtml;

    // Nodes
    let nHtml = '';
    order.forEach(item => {{
      const n = item.n, p = item.p;
      if (!nodeVisible(n)) return;
      const rBase = 4 + Math.sqrt(Math.min(n.degree, 25)) * 2;
      const r = Math.max(2, rBase * p.scale);
      const baseOp = Math.max(0.35, Math.min(1.0, 1200 / p.zcam));
      const op = baseOp * nodeDim(n.id);
      const fill = nodeFill(n);
      const isHover = (n.id === hoverId);
      const rEff = r * (isHover ? 1.25 : 1.0);
      nHtml += '<g class="kg-node" data-id="' + n.id + '" opacity="' + op.toFixed(2) + '">';
      nHtml += '<circle cx="' + p.sx.toFixed(1) + '" cy="' + p.sy.toFixed(1) +
               '" r="' + (rEff * 2.2).toFixed(2) + '" fill="' + fill +
               '" opacity="0.18" pointer-events="none"/>';
      nHtml += '<circle cx="' + p.sx.toFixed(1) + '" cy="' + p.sy.toFixed(1) +
               '" r="' + rEff.toFixed(2) + '" fill="' + fill +
               '" stroke="' + (isHover ? theme.label : theme.nodeStroke) +
               '" stroke-width="' + (isHover ? '1.6' : '0.7') + '"/>';
      if (p.scale > 0.45) {{
        const fs = Math.max(8, 10.5 * p.scale) * labelScale;
        nHtml += '<text x="' + (p.sx + rEff + 3).toFixed(1) + '" y="' +
                 (p.sy + 3).toFixed(1) + '" font-size="' + fs.toFixed(1) +
                 '" fill="' + theme.label + '" pointer-events="none" ' +
                 'style="font-family:var(--font-sans);paint-order:stroke;' +
                 'stroke:' + theme.labelStroke + ';stroke-width:2.5px;">' +
                 escapeHtml(nodeLabel(n.label)) + '</text>';
      }}
      nHtml += '</g>';
    }});
    nodeLayer.innerHTML = nHtml;
  }}

  // Settle
  for (let i = 0; i < 120; i++) step();

  // Phase 48d — persist the post-settle layout for this filter so the
  // next open of the same filter warm-starts from it. Done once, a
  // few seconds after the main rAF loop starts (gives the live
  // physics a little more time to refine the cold cluster seeds
  // without blocking the first paint). Idempotent: re-saves on every
  // render would be wasteful; one shot is enough.
  let _kgLayoutSaved = false;
  setTimeout(() => {{
    if (_kgLayoutSaved || !nodes.length) return;
    _kgSaveLayout(_kgLayoutKeyCurrent, nodes);
    _kgLayoutSaved = true;
  }}, 2500);

  function loop() {{
    if (!running) return;
    if (!paused) step();
    render();
    // Apply any pending shareable-URL state once the first frame is up
    if (window._kgPendingShare) {{
      try {{ _kgApplyPendingShare(); }} catch (e) {{}}
    }}
    raf = requestAnimationFrame(loop);
  }}
  loop();

  // ── Interaction: drag/orbit/wheel + hover dim + right-click menu ─
  let drag = null;
  // Convert a pointer event to local coordinates in the SVG's viewBox
  // space. Returns both the raw screen-space delta (`sx/sy`, used for
  // orbit sensitivity tuning) and viewBox-space coordinates (`x/y`,
  // used for node drag) so dragging a node stays under the cursor
  // even when the SVG is scaled (e.g. in fullscreen, where 1 screen
  // pixel ≠ 1 viewBox unit).
  function localPoint(evt) {{
    const rect = svg.getBoundingClientRect();
    const sx = evt.clientX - rect.left - rect.width / 2;
    const sy = evt.clientY - rect.top - rect.height / 2;
    const scaleX = W / Math.max(rect.width, 1);
    const scaleY = H / Math.max(rect.height, 1);
    return {{ x: sx * scaleX, y: sy * scaleY, sx: sx, sy: sy }};
  }}
  function neighborsOf(id) {{
    const ns = new Set([id]);
    const es = new Set();
    edges.forEach((e, ei) => {{
      if (e.source === id) {{ ns.add(e.target); es.add(ei); }}
      else if (e.target === id) {{ ns.add(e.source); es.add(ei); }}
    }});
    return {{ ns: ns, es: es }};
  }}

  svg.addEventListener('mousedown', (evt) => {{
    if (evt.button !== 0) return;  // left-click only for drag
    _kgHideMenu();
    const pt = localPoint(evt);
    const el = evt.target.closest && evt.target.closest('.kg-node');
    if (el) {{
      const id = parseInt(el.getAttribute('data-id'), 10);
      const n = nodes[id];
      const p = project(n);
      n.fixed = true;
      // startX/Y are viewBox-space; startSX/SY are screen-px — both
      // kept so orbit and node drag each use the right scale.
      drag = {{ mode: 'node', id: id,
                startX: pt.x, startY: pt.y,
                startSX: pt.sx, startSY: pt.sy,
                startWX: n.x, startWY: n.y, startWZ: n.z,
                startZcam: p.zcam, moved: false }};
    }} else {{
      drag = {{ mode: 'orbit', startX: pt.x, startY: pt.y,
                startSX: pt.sx, startSY: pt.sy,
                startRotX: cam.rotX, startRotY: cam.rotY }};
    }}
    svg.classList.add('kg-grabbing');
    evt.preventDefault();
  }});
  function onMove(evt) {{
    if (!drag) return;
    const pt = localPoint(evt);
    if (drag.mode === 'orbit') {{
      // Orbit uses screen-px delta so sensitivity stays consistent
      // regardless of how the SVG is scaled on screen.
      const dsx = pt.sx - drag.startSX, dsy = pt.sy - drag.startSY;
      cam.rotY = drag.startRotY + dsx * 0.006;
      cam.rotX = Math.max(-1.45, Math.min(1.45, drag.startRotX + dsy * 0.006));
    }} else {{
      // Node drag uses viewBox-space delta so the node stays exactly
      // under the cursor even when the SVG is scaled (fullscreen).
      const dx = pt.x - drag.startX, dy = pt.y - drag.startY;
      const dsx = pt.sx - drag.startSX, dsy = pt.sy - drag.startSY;
      if (Math.abs(dsx) + Math.abs(dsy) > 2) drag.moved = true;
      const d = worldDelta(dx, dy, drag.startZcam);
      const n = nodes[drag.id];
      n.x = drag.startWX + d.x;
      n.y = drag.startWY + d.y;
      n.z = drag.startWZ + d.z;
      n.vx = 0; n.vy = 0; n.vz = 0;
    }}
  }}
  function onUp() {{
    if (!drag) return;
    const d = drag; drag = null;
    svg.classList.remove('kg-grabbing');
    if (d.mode === 'node') {{
      const n = nodes[d.id];
      // If shift was held on mousedown we'd keep it pinned; but SVG
      // doesn't report modifier state at mouseup reliably, so we use
      // right-click → Pin for persistent pinning. Default mouseup
      // releases the node back to physics.
      n.fixed = false;
      if (!d.moved) {{
        // Tap-center: tween the camera so this node is framed at origin
        tweenCenterOn(n);
      }}
    }}
  }}
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);

  // Hover: dim everything except node + 1-hop neighbors. Delegation
  // via mouseover/mouseout on svg + target.closest('.kg-node') —
  // works even though the DOM is rebuilt every frame, because the
  // listener is on the stable <svg>, not per-node.
  svg.addEventListener('mouseover', (evt) => {{
    const el = evt.target.closest && evt.target.closest('.kg-node');
    if (!el) return;
    const id = parseInt(el.getAttribute('data-id'), 10);
    if (isNaN(id)) return;
    hoverId = id;
    const {{ ns, es }} = neighborsOf(id);
    hoverNeighbors = ns;
    hoverEdgeSet = es;
  }});
  svg.addEventListener('mouseout', (evt) => {{
    const rel = evt.relatedTarget;
    if (!rel || !svg.contains(rel) ||
        !(rel.closest && rel.closest('.kg-node'))) {{
      hoverId = -1; hoverNeighbors = null; hoverEdgeSet = null;
    }}
  }});

  // Right-click context menu. Actions differ for node / edge /
  // background (each branch ends with a `_kgShowMenu` call).
  svg.addEventListener('contextmenu', (evt) => {{
    evt.preventDefault();
    const nodeEl = evt.target.closest && evt.target.closest('.kg-node');
    const edgeEl = evt.target.closest && evt.target.closest('.kg-edge');
    if (nodeEl) {{
      const id = parseInt(nodeEl.getAttribute('data-id'), 10);
      const n = nodes[id];
      _kgShowMenu(evt.clientX, evt.clientY, [
        {{ label: 'Expand 1 hop', onClick: () => kgEgoExpand(n.label, 1) }},
        {{ label: 'Expand 2 hops', onClick: () => kgEgoExpand(n.label, 2) }},
        {{ label: (n.fixed ? 'Unpin' : 'Pin node'),
           onClick: () => {{ n.fixed = !n.fixed; }} }},
        {{ label: 'Center view', onClick: () => tweenCenterOn(n) }},
        {{ label: 'Hide node', onClick: () => {{ n.hidden = true; }} }},
        '-',
        {{ label: 'Show in table',
           onClick: () => {{
             document.getElementById('kg-subject').value = n.label;
             switchKgTab('kg-table');
             loadKg(0);
           }} }},
        {{ label: 'Copy label',
           onClick: () => {{ try {{ navigator.clipboard.writeText(n.label); }} catch(e) {{}} }} }},
      ]);
      return;
    }}
    if (edgeEl) {{
      const ei = parseInt(edgeEl.getAttribute('data-ei'), 10);
      const e = edges[ei];
      const firstT = e.triples[0] || {{}};
      const title = (firstT.doc_title || '(unknown source)').substring(0, 70);
      const preds = Array.from(e.predicates).slice(0, 3).join(', ');
      // Pick the first non-empty source sentence across merged triples
      const sent = (e.triples.find(t => t.source_sentence)
                    || {{}}).source_sentence || '';
      const sentDisplay = sent
        ? ('\\u201C' + (sent.length > 80 ? sent.substring(0, 80) + '\\u2026' : sent) + '\\u201D')
        : '(no source sentence — re-compile wiki to backfill)';
      _kgShowMenu(evt.clientX, evt.clientY, [
        {{ label: 'Source: ' + title, onClick: () => {{}} }},
        {{ label: 'Predicates: ' + preds, onClick: () => {{}} }},
        {{ label: sentDisplay, onClick: () => {{}} }},
        '-',
        {{ label: sent ? 'Copy sentence' : 'Copy sentence (none)',
           onClick: () => {{
             if (!sent) return;
             try {{ navigator.clipboard.writeText(sent); }} catch (ex) {{}}
           }} }},
        {{ label: 'Copy triple',
           onClick: () => {{
             const txt = nodes[e.source].label + '  [' + preds + ']  ' + nodes[e.target].label;
             try {{ navigator.clipboard.writeText(txt); }} catch (ex) {{}}
           }} }},
        {{ label: 'Show this paper in table',
           onClick: () => {{
             if (firstT.doc_id) {{
               document.getElementById('kg-subject').value = '';
               document.getElementById('kg-object').value = '';
               kgSearchByDoc(firstT.doc_id);
             }}
           }} }},
        {{ label: 'Filter by first predicate',
           onClick: () => {{
             const p = (firstT.predicate || '').trim();
             if (!p) return;
             const sel = document.getElementById('kg-predicate');
             const has = Array.from(sel.options).some(o => o.value === p);
             if (!has) {{
               const opt = document.createElement('option');
               opt.value = p; opt.textContent = p; sel.appendChild(opt);
             }}
             sel.value = p; loadKg(0);
           }} }},
      ]);
      return;
    }}
    // Background menu
    _kgShowMenu(evt.clientX, evt.clientY, [
      {{ label: (paused ? 'Resume physics' : 'Freeze physics'),
         hint: 'Space', onClick: () => {{ paused = !paused; }} }},
      {{ label: 'Reset view', onClick: () => {{ resetView(); }} }},
      {{ label: 'Unhide all nodes',
         onClick: () => {{ nodes.forEach(n => {{ n.hidden = false; }}); }} }},
      '-',
      {{ label: 'Download PNG', onClick: () => downloadPng() }},
    ]);
  }});

  svg.addEventListener('wheel', (evt) => {{
    evt.preventDefault();
    const factor = Math.exp(evt.deltaY * 0.0015);
    cam.dist = Math.max(250, Math.min(3000, cam.dist * factor));
  }}, {{ passive: false }});

  // Keyboard: Space toggles freeze; only when KG modal is open and
  // focus isn't in an input. Prevents the page from scrolling on Space.
  function onKey(evt) {{
    const modal = document.getElementById('kg-modal');
    if (!modal || modal.style.display === 'none') return;
    const tag = (evt.target && evt.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (evt.code === 'Space') {{
      paused = !paused; evt.preventDefault();
    }}
  }}
  window.addEventListener('keydown', onKey);

  // ── Camera tween: smoothly frame a node at origin ──────────────
  function tweenCenterOn(n) {{
    const startRotX = cam.rotX, startRotY = cam.rotY, startDist = cam.dist;
    // Pick rotation so the node's world vector points toward -Z (into
    // screen), then back off to a comfortable distance. Clamp pitch so
    // we don't flip through the poles.
    const len = Math.sqrt(n.x*n.x + n.y*n.y + n.z*n.z) + 0.01;
    const targetRotY = Math.atan2(n.x, Math.max(n.z, 0.01));
    const targetRotX = Math.max(-1.4, Math.min(1.4,
                         -Math.atan2(n.y, Math.sqrt(n.x*n.x + n.z*n.z) + 0.01)));
    const targetDist = Math.max(400, Math.min(1200, len * 1.8 + 300));
    const t0 = performance.now();
    function frame(t) {{
      const k = Math.min((t - t0) / 450, 1);
      const e = _kgEase(k);
      cam.rotX = startRotX + (targetRotX - startRotX) * e;
      cam.rotY = startRotY + (targetRotY - startRotY) * e;
      cam.dist = startDist + (targetDist - startDist) * e;
      if (k < 1) requestAnimationFrame(frame);
    }}
    requestAnimationFrame(frame);
  }}

  // ── Reset view ─────────────────────────────────────────────────
  function resetView() {{
    cam.rotX = camDefault.rotX;
    cam.rotY = camDefault.rotY;
    cam.dist = camDefault.dist;
    nodes.forEach(n => {{ n.hidden = false; n.fixed = false; }});
    searchMatches = null;
    const s = document.getElementById('kg-search');
    if (s) s.value = '';
  }}

  // ── Search: live-highlight nodes whose label contains the query ─
  function doSearch(q) {{
    q = (q || '').trim().toLowerCase();
    if (!q) {{ searchMatches = null; return; }}
    const m = new Set();
    nodes.forEach(n => {{
      if (n.label.toLowerCase().indexOf(q) !== -1) m.add(n.id);
    }});
    searchMatches = m;
    // Auto-center on the first match
    for (const id of m) {{ tweenCenterOn(nodes[id]); break; }}
  }}

  // ── PNG export: serialize SVG → <img> → canvas → blob → download ─
  function downloadPng() {{
    const xml = new XMLSerializer().serializeToString(svg);
    const svgBlob = new Blob(
      ['<?xml version="1.0" encoding="UTF-8"?>' + xml],
      {{ type: 'image/svg+xml;charset=utf-8' }});
    const url = URL.createObjectURL(svgBlob);
    const img = new Image();
    const outW = svg.clientWidth || W;
    const outH = svg.clientHeight || H;
    img.onload = () => {{
      const cvs = document.createElement('canvas');
      cvs.width = outW * 2; cvs.height = outH * 2;  // 2× for retina
      const ctx = cvs.getContext('2d');
      ctx.fillStyle = theme.canvasBg;
      ctx.fillRect(0, 0, cvs.width, cvs.height);
      ctx.drawImage(img, 0, 0, cvs.width, cvs.height);
      URL.revokeObjectURL(url);
      cvs.toBlob(b => {{
        const a = document.createElement('a');
        a.href = URL.createObjectURL(b);
        a.download = 'knowledge-graph.png';
        document.body.appendChild(a); a.click(); a.remove();
      }}, 'image/png');
    }};
    img.onerror = () => {{ URL.revokeObjectURL(url); }};
    img.src = url;
  }}

  // ── Expose sim controls ────────────────────────────────────────
  canvas._kgSim = {{
    stop: () => {{
      running = false;
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('keydown', onKey);
    }},
    setTheme: (name) => {{
      if (name && KG_THEMES[name]) _kgActiveTheme = name;
      theme = _kgEffectiveTheme();
      _applyKgDefs(svg, theme);
      _applyKgClusterDefs(svg, theme, numClusters);
      canvas.style.background = theme.canvasBg;
    }},
    refreshTheme: () => {{
      // Called after custom-color overrides change OR after a preset
      // swap. Reads the effective (preset + overrides) theme and
      // repushes the gradient stops. No simulation restart; the
      // render loop picks up the new `theme` closure next frame.
      theme = _kgEffectiveTheme();
      _applyKgDefs(svg, theme);
      _applyKgClusterDefs(svg, theme, numClusters);
      canvas.style.background = theme.canvasBg;
    }},
    togglePause: () => {{ paused = !paused; }},
    setColorBy: (mode) => {{ if (mode) colorBy = mode; }},
    setLabelScale: (v) => {{ labelScale = Math.max(0.4, Math.min(2.2, v)); }},
    setDegFilter: (v) => {{ degFilter = (v >= 99) ? 999 : v; }},
    search: doSearch,
    resetView: resetView,
    centerOn: tweenCenterOn,
    downloadPng: downloadPng,
    // Phase 48d — return the subset of state that defines the visible
    // view (camera pose + which nodes are pinned). The share-URL
    // builder merges this with the theme + filter fields into the
    // compact base64 blob. We serialize pin by node label (not id)
    // because ids are per-filter-load and unstable.
    getShareState: () => ({{
      c: {{ rx: cam.rotX, ry: cam.rotY, d: cam.dist }},
      p: nodes.filter(n => n.fixed).map(n => n.label),
    }}),
    applyShareState: (st) => {{
      if (!st) return;
      if (st.c) {{
        if (typeof st.c.rx === 'number') cam.rotX = st.c.rx;
        if (typeof st.c.ry === 'number') cam.rotY = st.c.ry;
        if (typeof st.c.d  === 'number') cam.dist = Math.max(250, Math.min(3000, st.c.d));
      }}
      if (Array.isArray(st.p)) {{
        const pinSet = new Set(st.p);
        nodes.forEach(n => {{ if (pinSet.has(n.label)) n.fixed = true; }});
      }}
    }},
  }};
}}

// ── Module-scope KG helpers used by the context menu + toolbar ────

// Replace the current graph view with the ego network around a label.
// depth=1 → just the node's direct 1-hop (single /api/kg call using
// `any_side`). depth=2 → 1-hop + the 1-hop of each of the top-10
// most-frequent neighbors (parallel fetches, deduped, confidence-
// ranked, capped at 200 triples). Two hops is where scientific KGs
// get interesting — depth 1 is often just a claim, depth 2 is the
// surrounding context.
async function kgEgoExpand(label, depth) {{
  depth = depth || 1;
  const status = document.getElementById('kg-status');
  try {{
    if (status) status.textContent = 'Expanding around "' + label + '" (depth ' + depth + ')…';
    const r1 = await fetch('/api/kg?' + new URLSearchParams({{ any_side: label, limit: 200 }}));
    let all = ((await r1.json()).triples) || [];
    if (depth >= 2 && all.length) {{
      // Count neighbor frequency to pick who to expand next
      const freq = new Map();
      all.forEach(t => {{
        const s = (t.subject || '').substring(0, 60);
        const o = (t.object  || '').substring(0, 60);
        if (s.toLowerCase() !== label.toLowerCase()) {{
          freq.set(s, (freq.get(s) || 0) + 1);
        }}
        if (o.toLowerCase() !== label.toLowerCase()) {{
          freq.set(o, (freq.get(o) || 0) + 1);
        }}
      }});
      const topN = Array.from(freq.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(entry => entry[0]);
      // Parallel fetch, 50 triples per neighbor (keeps response time sane)
      const batches = await Promise.all(topN.map(n =>
        fetch('/api/kg?' + new URLSearchParams({{ any_side: n, limit: 50 }}))
          .then(r => r.json()).then(d => d.triples || [])
          .catch(() => [])
      ));
      const seen = new Set();
      all.forEach(t => seen.add((t.subject || '') + '|' + (t.predicate || '') + '|' + (t.object || '')));
      batches.forEach(batch => {{
        batch.forEach(t => {{
          const k = (t.subject || '') + '|' + (t.predicate || '') + '|' + (t.object || '');
          if (!seen.has(k)) {{ all.push(t); seen.add(k); }}
        }});
      }});
      // Confidence-sort, cap at 200 so the graph stays navigable
      all.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
      all = all.slice(0, 200);
    }}
    _kgTriples = all;
    if (status) status.textContent =
      'Expanded around "' + label + '" (depth ' + depth + ') · ' +
      all.length + ' triple(s)';
    _renderKgTable(all);
    _renderKgGraph(all.slice(0, 100));
  }} catch (e) {{
    if (status) status.textContent = 'Error: ' + e.message;
  }}
}}

// Fetch all triples for a given document and re-render. Used by the
// edge context menu's "Show this paper" action.
async function kgSearchByDoc(docId) {{
  const params = new URLSearchParams({{ document_id: docId, limit: 200 }});
  try {{
    const res = await fetch('/api/kg?' + params.toString());
    const data = await res.json();
    _kgTriples = data.triples || [];
    document.getElementById('kg-status').textContent =
      'Showing triples from selected paper · ' + _kgTriples.length;
    _renderKgTable(_kgTriples);
    _renderKgGraph(_kgTriples.slice(0, 100));
    switchKgTab('kg-graph');
  }} catch (e) {{
    document.getElementById('kg-status').textContent = 'Error: ' + e.message;
  }}
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
    // Phase 33 — chapter title is draggable for reordering (mirrors
    // section drag-drop from Phase 26 but operates on ch-group).
    html += '<div class="ch-title clickable" draggable="true" ' +
      'data-ch-id="' + ch.id + '" ' +
      'ondragstart="chDragStart(event,\\\'' + ch.id + '\\\')" ' +
      'ondragover="chDragOver(event)" ' +
      'ondrop="chDrop(event,\\\'' + ch.id + '\\\')" ' +
      'ondragend="chDragEnd(event)" ' +
      'onclick="selectChapter(this.parentElement)">' +
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
          // Phase 42 — data-action dispatch (preview-empty-section).
          html += '<div class="sec-link sec-empty" draggable="true" ' +
            'data-section-slug="' + tmpl.slug + '" ' +
            'title="' + planAttr + '" ' +
            'data-action="preview-empty-section" ' +
            'data-chapter-id="' + ch.id + '" data-sec-type="' + tmpl.slug + '">' +
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
      // Phase 42 — data-action dispatch for orphan +/✗ buttons.
      html += '<a class="sec-link sec-orphan" href="/section/' + draft.id +
        '" data-draft-id="' + draft.id + '" onclick="return navTo(this)" ' +
        'title="Orphan draft. Click to inspect, + to adopt into sections, X to delete.">' +
        '<span class="sec-status-dot orphan"></span>' +
        display +
        ' <span class="meta">orphan \\u00b7 v' + draft.version + ' \\u00b7 ' + draft.words + 'w</span>' +
        '<button class="sec-orphan-adopt" ' +
        'data-action="adopt-orphan-section" ' +
        'data-chapter-id="' + ch.id + '" data-sec-type="' + slug + '" ' +
        'title="Add this section_type to the chapter sections list">+</button>' +
        '<button class="sec-orphan-delete" ' +
        'data-action="delete-orphan-draft" data-draft-id="' + draft.id + '" ' +
        'title="Delete this orphan draft permanently">\\u2717</button>' +
        '</a>';
    }});

    if (meta.length === 0 && Object.keys(draftBySlug).length === 0) {{
      html += '<div class="sec-link sec-empty-cta" data-action="start-writing-chapter" data-chapter-id="' + ch.id + '">\\u270e Start writing</div>';
    }}
    // Phase 32.4 — "+ Add section" CTA at the bottom of every
    // chapter's section list. Click → prompt for a title → POST
    // a new section dict via PUT /api/chapters/{{id}}/sections.
    html += '<div class="sec-link sec-add-cta" ' +
      'data-action="add-section-to-chapter" data-chapter-id="' + ch.id + '" ' +
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

  // Phase 35 — Total Compute panel. Aggregates every LLM-backed job
  // (write/review/revise/argue/gaps/autowrite/plan/...) from the
  // llm_usage_log ledger so the user can see total GPU compute spent
  // on the book. Per-operation breakdown appears as chips below the
  // three headline tiles. Autowrite is a strict subset — its detail
  // panel follows directly below.
  const tc = data.total_compute || {{}};
  const fmtTokens = (n) => (n || 0) >= 1000
    ? ((n / 1000).toFixed(1) + 'K')
    : String(n || 0);
  const fmtSecs = (secs) => {{
    secs = Math.round(secs || 0);
    if (secs < 60) return secs + 's';
    if (secs < 3600) return Math.floor(secs / 60) + 'm ' + (secs % 60) + 's';
    return Math.floor(secs / 3600) + 'h ' + Math.floor((secs % 3600) / 60) + 'm';
  }};
  if ((tc.total_jobs || 0) > 0) {{
    html += '<h3 style="margin:24px 0 12px;font-size:14px;font-weight:600;color:var(--fg-muted);text-transform:uppercase;letter-spacing:0.04em;">Total Compute</h3>';
    html += '<div class="stat-grid">';
    html += '<div class="stat-tile"><div class="num">' + fmtTokens(tc.total_tokens) + '</div><div class="lbl">Total Tokens</div></div>';
    html += '<div class="stat-tile"><div class="num">' + fmtSecs(tc.total_seconds) + '</div><div class="lbl">Total Time</div></div>';
    html += '<div class="stat-tile"><div class="num">' + (tc.total_jobs || 0) + '</div><div class="lbl">LLM Jobs</div></div>';
    html += '</div>';
    // Per-operation breakdown as compact chips (only ops that actually ran)
    if (Array.isArray(tc.by_operation) && tc.by_operation.length > 0) {{
      html += '<div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;">';
      tc.by_operation.forEach(o => {{
        html += '<span style="display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:12px;background:var(--bg-alt,#f3f4f6);font-size:12px;color:var(--fg-muted);">';
        html += '<strong style="color:var(--fg);">' + (o.operation || '?') + '</strong>';
        html += '<span>' + fmtTokens(o.tokens) + ' tok</span>';
        html += '<span>' + fmtSecs(o.seconds) + '</span>';
        html += '<span>×' + (o.jobs || 0) + '</span>';
        html += '</span>';
      }});
      html += '</div>';
    }}
  }}

  // Phase 33 — Autowrite effort stats from the Layer 0 telemetry tables.
  // Shows cumulative token usage + time spent across all completed runs.
  const aw = data.autowrite_stats || {{}};
  if (aw.total_runs > 0) {{
    html += '<h3 style="margin:24px 0 12px;font-size:14px;font-weight:600;color:var(--fg-muted);text-transform:uppercase;letter-spacing:0.04em;">Autowrite Effort</h3>';
    html += '<div class="stat-grid">';
    html += '<div class="stat-tile"><div class="num">' + (aw.total_runs || 0) + '</div><div class="lbl">Runs</div></div>';
    const tokStr = (aw.total_tokens || 0) >= 1000
      ? ((aw.total_tokens / 1000).toFixed(1) + 'K')
      : String(aw.total_tokens || 0);
    html += '<div class="stat-tile"><div class="num">' + tokStr + '</div><div class="lbl">Tokens Used</div></div>';
    // Format total_seconds as hours + minutes
    const secs = aw.total_seconds || 0;
    let timeStr;
    if (secs < 60) timeStr = secs + 's';
    else if (secs < 3600) timeStr = Math.floor(secs / 60) + 'm ' + (secs % 60) + 's';
    else timeStr = Math.floor(secs / 3600) + 'h ' + Math.floor((secs % 3600) / 60) + 'm';
    html += '<div class="stat-tile"><div class="num">' + timeStr + '</div><div class="lbl">Time Spent</div></div>';
    html += '</div>';
  }}

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
    // Phase 42 — data-action dispatch for the heatmap's clickable cells.
    html += '<tr><td class="ch-label clickable" data-action="open-chapter-modal" data-chapter-id="' + row.id + '" title="Click to edit chapter title and scope">';
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
          'data-action="preview-empty-section" ' +
          'data-chapter-id="' + row.id + '" data-sec-type="' + cell.type + '" ' +
          'title="' + escapeHtml(posLabel) + ' (empty — click to preview)">+</span></td>';
      }} else {{
        const label = 'v' + cell.version + ' ' + cell.words + 'w';
        html += '<td><span class="hm-cell ' + cell.status + '" ' +
          'data-action="load-section" data-draft-id="' + cell.draft_id + '" ' +
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
        // Phase 42 — data-action dispatch; chapter_num is a numeric
        // attr, parsed back via parseInt in the handler.
        btn = '<button data-action="write-for-gap" data-chapter-num="' + g.chapter_num + '">Write</button>';
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
    html += '<span class="version-badge" data-vid="' + v.id + '" data-action="select-version" data-version-id="' + v.id + '">' +
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

// ── Phase 46.A — Auto-insert [N] citations (two-pass LLM) ───────────────
async function doInsertCitations() {{
  if (!currentDraftId) {{
    showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft.");
    return;
  }}
  if (!confirm(
    "Auto-insert [N] citation markers?\\n\\n"
    + "This scans the draft for claims that need citations, hybrid-searches "
    + "your corpus for top-8 candidates per claim, and saves a new version "
    + "with the accepted citations applied.\\n\\n"
    + "Takes ~1-3 minutes depending on draft length. Mirrors "
    + "`sciknow book insert-citations`."
  )) return;
  showStreamPanel('Inserting citations...');
  const fd = new FormData();
  const res = await fetch('/api/insert-citations/' + currentDraftId, {{
    method: 'POST', body: fd
  }});
  const data = await res.json();
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();
  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;
  const body = document.getElementById('stream-body');
  const status = document.getElementById('stream-status');
  body.innerHTML =
    '<div id="ic-summary" style="font-size:16px;margin-bottom:12px;"></div>'
    + '<div id="ic-log" style="font-size:12px;font-family:ui-monospace,monospace;'
    + 'max-height:320px;overflow:auto;padding:8px;background:var(--toolbar-bg);'
    + 'border-radius:4px;"></div>';
  const summary = document.getElementById('ic-summary');
  const log = document.getElementById('ic-log');
  function _append(html) {{ log.innerHTML += html; log.scrollTop = log.scrollHeight; }}
  source.onmessage = function(e) {{
    const evt = JSON.parse(e.data);
    if (evt.type === 'progress') {{
      status.textContent = evt.detail || evt.stage;
      _append('<div style="color:var(--fg-muted);">' + (evt.detail || evt.stage) + '</div>');
    }} else if (evt.type === 'citation_needs') {{
      summary.innerHTML = '<strong>' + evt.count + '</strong> location(s) flagged for citation';
      status.textContent = 'Retrieving candidates...';
    }} else if (evt.type === 'citation_candidates') {{
      _append('<div>#' + (evt.index + 1) + ': ' + evt.n_candidates + ' candidate(s)</div>');
    }} else if (evt.type === 'citation_selected') {{
      const conf = (evt.confidence != null) ? evt.confidence.toFixed(2) : '?';
      _append('<div style="color:var(--success);">&nbsp;&nbsp;&#10003; #'
        + (evt.index + 1) + ' picked (conf ' + conf + ')</div>');
    }} else if (evt.type === 'citation_skipped') {{
      _append('<div style="color:var(--fg-muted);">&nbsp;&nbsp;&mdash; #'
        + (evt.index + 1) + ' skipped (' + (evt.reason || 'no match') + ')</div>');
    }} else if (evt.type === 'citation_inserted') {{
      // Total count emitted right before completed
    }} else if (evt.type === 'completed') {{
      const msg = evt.message
        || ('Inserted ' + (evt.n_inserted || 0) + ' / ' + (evt.n_needs || 0) + ' citations');
      status.textContent = msg;
      summary.innerHTML = '<span style="color:var(--success);font-weight:bold;">&#10003; '
        + msg + '</span>';
      source.close(); currentEventSource = null; currentJobId = null;
      if ((evt.n_inserted || 0) > 0 && currentDraftId) {{
        setTimeout(() => loadSection(currentDraftId), 800);
      }}
      setTimeout(() => hideStreamPanel(), 3500);
    }} else if (evt.type === 'error') {{
      status.textContent = 'Error: ' + evt.message;
      summary.innerHTML = '<span style="color:var(--danger);">&#10007; ' + evt.message + '</span>';
      source.close(); currentEventSource = null; currentJobId = null;
    }} else if (evt.type === 'done') {{
      source.close(); currentEventSource = null; currentJobId = null;
    }}
  }};
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

// Phase 33 — autowrite mode picker via a proper modal (replaces the
// triple-prompt() UX from Phase 28). doAutowrite opens the modal;
// confirmAutowrite reads the values and fires the request.
let _awSelectedMode = 'skip';

function selectAwMode(mode) {{
  _awSelectedMode = mode;
  document.querySelectorAll('.aw-mode-btn').forEach(b => {{
    b.classList.toggle('active', b.dataset.mode === mode);
  }});
}}

function doAutowrite() {{
  if (!currentChapterId) {{ showEmptyHint("No chapter selected &mdash; click any chapter title in the sidebar to select it, then try again."); return; }}

  const isAllSections = !currentSectionType;
  const section = currentSectionType || '__all__';
  const ch = chaptersData.find(c => c.id === currentChapterId);
  const chTitle = ch ? ('Ch.' + ch.num + ': ' + ch.title) : 'this chapter';

  // Configure the modal scope label
  const scopeEl = document.getElementById('aw-config-scope');
  if (scopeEl) {{
    scopeEl.textContent = isAllSections
      ? 'Autowrite ALL sections of ' + chTitle
      : 'Autowrite ' + section + ' in ' + chTitle;
  }}

  // Reset inputs to defaults
  document.getElementById('aw-config-max-iter').value = '3';
  document.getElementById('aw-config-target-score').value = '0.85';
  _awSelectedMode = 'skip';
  document.querySelectorAll('.aw-mode-btn').forEach(b => {{
    b.classList.toggle('active', b.dataset.mode === 'skip');
  }});

  // Show the mode section only when running all sections AND some already
  // have drafts — otherwise mode choice is irrelevant.
  const modeSection = document.getElementById('aw-config-mode-section');
  const modeInfo = document.getElementById('aw-config-mode-info');
  if (isAllSections && ch) {{
    const nSecs = (ch.sections_meta && ch.sections_meta.length) || 0;
    let nDrafted = 0;
    if (Array.isArray(ch.sections)) {{
      const draftedSlugs = new Set(ch.sections.filter(s => s.id).map(s => (s.type || '').toLowerCase()));
      nDrafted = (ch.sections_meta || []).filter(m => draftedSlugs.has(m.slug)).length;
    }}
    if (nDrafted > 0) {{
      modeSection.style.display = 'block';
      modeInfo.textContent = nDrafted + ' of ' + nSecs + ' sections already have a draft.';
    }} else {{
      modeSection.style.display = 'none';
    }}
  }} else {{
    modeSection.style.display = 'none';
  }}

  openModal('autowrite-config-modal');
}}

async function confirmAutowrite() {{
  closeModal('autowrite-config-modal');

  const isAllSections = !currentSectionType;
  const section = currentSectionType || '__all__';
  const maxIter = document.getElementById('aw-config-max-iter').value || '3';
  const targetStr = document.getElementById('aw-config-target-score').value || '0.85';
  awTargetScore = parseFloat(targetStr) || 0.85;
  awScores = [];

  const modeRebuild = _awSelectedMode === 'rebuild';
  const modeResume = _awSelectedMode === 'resume';

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
  fd.append('max_iter', maxIter);
  fd.append('target_score', String(awTargetScore));
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
// ── Phase 33: keyboard shortcuts ──────────────────────────────────────
//
// Global keydown handler for the web reader. All shortcuts are designed
// to be non-destructive and avoid conflicts with browser defaults:
//   Esc           — close any open modal (already existed since Phase 14)
//   Ctrl+S        — force save in editor (suppresses the browser's "Save as")
//   Ctrl+K        — focus the sidebar search bar
//   Ctrl+E        — toggle the inline editor
//   ← / →         — navigate to previous / next section in the sidebar
//                   (only when focus is NOT in an input/textarea/select)
//   D             — show dashboard (same exclusion)
//   P             — open plan modal (same exclusion)
//
// The "not in an input" guard prevents letter shortcuts from swallowing
// keystrokes while the user types in a form field, textarea, search bar,
// or the editor. Ctrl+* shortcuts work everywhere because the user
// explicitly holds Ctrl.

document.addEventListener('keydown', function(e) {{
  const tag = (e.target.tagName || '').toUpperCase();
  const inInput = (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
                   || e.target.isContentEditable);

  // ── Esc — close modals (Phase 14, kept) ──────────────────────────
  if (e.key === 'Escape') {{
    document.querySelectorAll('.modal-overlay.open').forEach(m => {{
      closeModal(m.id);
    }});
    return;
  }}

  // ── Ctrl+S — force save in editor ────────────────────────────────
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {{
    e.preventDefault();  // suppress browser "Save page as..."
    const ev = document.getElementById('edit-view');
    if (ev && ev.style.display !== 'none') {{
      edAutosave();
    }}
    return;
  }}

  // ── Ctrl+K — focus search bar ────────────────────────────────────
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {{
    e.preventDefault();
    const searchInput = document.querySelector('.search-bar input');
    if (searchInput) {{
      searchInput.focus();
      searchInput.select();
    }}
    return;
  }}

  // ── Ctrl+E — toggle editor ───────────────────────────────────────
  if ((e.ctrlKey || e.metaKey) && e.key === 'e') {{
    e.preventDefault();
    if (currentDraftId) toggleEdit();
    return;
  }}

  // ── Letter shortcuts (only when NOT typing in a field) ───────────
  if (inInput) return;

  // ← / → — previous / next section
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {{
    const links = Array.from(document.querySelectorAll('#sidebar-sections .sec-link[href]'));
    if (links.length === 0) return;
    const idx = links.findIndex(a => a.classList.contains('active'));
    let next;
    if (e.key === 'ArrowLeft') {{
      next = idx > 0 ? links[idx - 1] : links[links.length - 1];
    }} else {{
      next = idx < links.length - 1 ? links[idx + 1] : links[0];
    }}
    if (next) {{
      next.click();
      next.scrollIntoView({{block: 'nearest'}});
    }}
    return;
  }}

  // D — dashboard
  if (e.key === 'd' || e.key === 'D') {{
    showDashboard();
    return;
  }}

  // P — plan modal
  if (e.key === 'p' || e.key === 'P') {{
    openPlanModal();
    return;
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
  ['wiki-query', 'wiki-browse', 'wiki-lint', 'wiki-consensus'].forEach(tab => {{
    const pane = document.getElementById(tab + '-pane');
    if (pane) pane.style.display = (name === tab) ? 'block' : 'none';
  }});
  if (name === 'wiki-browse') loadWikiPages(1);
}}

// ── Phase 54.6.2 — Wiki Lint + Consensus (surface CLI in GUI) ───────────
let _wikiLintJob = null;
let _wikiLintSource = null;

async function doWikiLint() {{
  const deep = document.getElementById('wiki-lint-deep').checked;
  const runBtn = document.getElementById('wiki-lint-run');
  const stopBtn = document.getElementById('wiki-lint-stop');
  const status = document.getElementById('wiki-lint-status');
  const summary = document.getElementById('wiki-lint-summary');
  const issuesEl = document.getElementById('wiki-lint-issues');
  runBtn.disabled = true;
  stopBtn.style.display = 'inline-block';
  status.textContent = 'Running structural checks…';
  summary.innerHTML = '';
  issuesEl.innerHTML = '';

  const fd = new FormData();
  fd.append('deep', deep);
  let res;
  try {{
    res = await fetch('/api/wiki/lint', {{method: 'POST', body: fd}});
  }} catch (exc) {{
    status.textContent = 'Request failed: ' + exc.message;
    runBtn.disabled = false; stopBtn.style.display = 'none';
    return;
  }}
  const data = await res.json();
  _wikiLintJob = data.job_id;
  const source = new EventSource('/api/stream/' + _wikiLintJob);
  _wikiLintSource = source;
  const bySeverity = {{ high: [], medium: [], low: [] }};

  source.onmessage = function(e) {{
    const evt = JSON.parse(e.data);
    if (evt.type === 'progress') {{
      status.textContent = evt.detail || evt.stage;
    }} else if (evt.type === 'lint_issue') {{
      const sev = (evt.severity || 'low').toLowerCase();
      (bySeverity[sev] || bySeverity.low).push(evt);
      _renderWikiLintIssues(bySeverity);
    }} else if (evt.type === 'completed') {{
      const n = evt.issues_count || 0;
      status.textContent = n === 0 ? 'All checks passed.' : (n + ' issue(s) found.');
      summary.innerHTML = (n === 0)
        ? '<div style="color:var(--success);font-weight:bold;">&#10003; Wiki is clean.</div>'
        : '<div>'
          + '<span style="color:var(--danger);font-weight:bold;">' + (bySeverity.high.length) + '</span> high · '
          + '<span style="color:var(--warning);font-weight:bold;">' + (bySeverity.medium.length) + '</span> medium · '
          + '<span style="color:var(--fg-muted);font-weight:bold;">' + (bySeverity.low.length) + '</span> low'
          + '</div>';
      source.close(); _wikiLintSource = null; _wikiLintJob = null;
      runBtn.disabled = false; stopBtn.style.display = 'none';
    }} else if (evt.type === 'error') {{
      status.textContent = 'Error: ' + evt.message;
      source.close(); _wikiLintSource = null; _wikiLintJob = null;
      runBtn.disabled = false; stopBtn.style.display = 'none';
    }} else if (evt.type === 'done') {{
      source.close(); _wikiLintSource = null; _wikiLintJob = null;
      runBtn.disabled = false; stopBtn.style.display = 'none';
    }}
  }};
}}

function _renderWikiLintIssues(bySeverity) {{
  const el = document.getElementById('wiki-lint-issues');
  const order = ['high', 'medium', 'low'];
  const colors = {{high:'var(--danger)', medium:'var(--warning)', low:'var(--fg-muted)'}};
  const labels = {{high:'HIGH', medium:'MEDIUM', low:'LOW'}};
  let html = '';
  for (const sev of order) {{
    const issues = bySeverity[sev] || [];
    if (!issues.length) continue;
    html += '<div style="margin-top:8px;"><div style="font-weight:bold;color:'
      + colors[sev] + ';font-size:11px;letter-spacing:0.05em;">' + labels[sev]
      + ' (' + issues.length + ')</div>';
    for (const i of issues) {{
      const kind = i.type_ || i.kind || 'issue';
      html += '<div style="padding:6px 8px;margin-top:4px;border-left:3px solid '
        + colors[sev] + ';background:var(--toolbar-bg);border-radius:4px;font-size:12px;">'
        + '<code style="font-size:10px;color:var(--fg-muted);">' + _escHtml(kind) + '</code> '
        + _escHtml(i.detail || i.message || JSON.stringify(i)) + '</div>';
    }}
    html += '</div>';
  }}
  el.innerHTML = html;
}}

async function stopWikiLint() {{
  if (_wikiLintJob) {{
    await fetch('/api/jobs/' + _wikiLintJob, {{method: 'DELETE'}});
  }}
}}

let _wikiConsensusJob = null;
let _wikiConsensusSource = null;

async function doWikiConsensus() {{
  const topic = document.getElementById('wiki-consensus-topic').value.trim();
  if (!topic) {{ alert('Enter a topic first.'); return; }}
  const runBtn = document.getElementById('wiki-consensus-run');
  const stopBtn = document.getElementById('wiki-consensus-stop');
  const status = document.getElementById('wiki-consensus-status');
  const summaryEl = document.getElementById('wiki-consensus-summary');
  const claimsEl = document.getElementById('wiki-consensus-claims');
  const debatedEl = document.getElementById('wiki-consensus-debated');
  runBtn.disabled = true;
  stopBtn.style.display = 'inline-block';
  status.textContent = 'Gathering evidence…';
  summaryEl.innerHTML = '';
  claimsEl.innerHTML = '';
  debatedEl.innerHTML = '';

  const fd = new FormData();
  fd.append('topic', topic);
  const res = await fetch('/api/wiki/consensus', {{method: 'POST', body: fd}});
  const data = await res.json();
  _wikiConsensusJob = data.job_id;
  const source = new EventSource('/api/stream/' + _wikiConsensusJob);
  _wikiConsensusSource = source;

  source.onmessage = function(e) {{
    const evt = JSON.parse(e.data);
    if (evt.type === 'progress') {{
      status.textContent = evt.detail || evt.stage;
    }} else if (evt.type === 'consensus') {{
      _renderConsensus(evt.data || {{}}, summaryEl, claimsEl, debatedEl);
    }} else if (evt.type === 'completed') {{
      const slug = evt.slug || '';
      const claims = evt.claims || 0;
      status.innerHTML = 'Saved <strong>' + claims + '</strong> claim(s)'
        + (slug ? ' as <a href="#" onclick="event.preventDefault();switchWikiTab(\\'wiki-browse\\');openWikiPage(\\'' + _escHtml(slug) + '\\');">' + _escHtml(slug) + '</a>.' : '.');
      source.close(); _wikiConsensusSource = null; _wikiConsensusJob = null;
      runBtn.disabled = false; stopBtn.style.display = 'none';
    }} else if (evt.type === 'error') {{
      status.textContent = 'Error: ' + evt.message;
      source.close(); _wikiConsensusSource = null; _wikiConsensusJob = null;
      runBtn.disabled = false; stopBtn.style.display = 'none';
    }} else if (evt.type === 'done') {{
      source.close(); _wikiConsensusSource = null; _wikiConsensusJob = null;
      runBtn.disabled = false; stopBtn.style.display = 'none';
    }}
  }};
}}

function _renderConsensus(data, summaryEl, claimsEl, debatedEl) {{
  if (data.summary) {{
    summaryEl.innerHTML = '<div style="padding:8px;background:var(--toolbar-bg);border-radius:4px;">'
      + _escHtml(data.summary) + '</div>';
  }}
  const colorOf = {{
    strong: 'var(--success)', moderate: 'var(--accent)',
    weak: 'var(--warning)', contested: 'var(--danger)',
  }};
  const claims = data.claims || [];
  if (claims.length) {{
    let html = '<div style="font-size:11px;font-weight:bold;color:var(--fg-muted);margin:8px 0;">CLAIMS</div>';
    for (const c of claims) {{
      const level = (c.consensus_level || 'unknown').toLowerCase();
      const color = colorOf[level] || 'var(--fg-muted)';
      const trend = c.trend ? (' · trend: ' + c.trend) : '';
      const sup = c.supporting_papers || [];
      const con = c.contradicting_papers || [];
      html += '<div style="padding:8px 10px;margin-top:6px;border-left:3px solid '
        + color + ';background:var(--toolbar-bg);border-radius:4px;font-size:12px;">'
        + '<span style="font-weight:bold;color:' + color + ';text-transform:uppercase;font-size:10px;letter-spacing:0.05em;">'
        + _escHtml(level) + '</span>'
        + '<span style="color:var(--fg-muted);font-size:10px;">' + _escHtml(trend) + '</span>'
        + '<div style="margin-top:4px;">' + _escHtml(c.claim || '') + '</div>';
      if (sup.length) {{
        html += '<div style="margin-top:4px;color:var(--success);font-size:11px;">Supports ('
          + sup.length + '): ' + sup.slice(0, 4).map(_escHtml).join(', ')
          + (sup.length > 4 ? ' +' + (sup.length - 4) : '') + '</div>';
      }}
      if (con.length) {{
        html += '<div style="margin-top:2px;color:var(--danger);font-size:11px;">Contradicts ('
          + con.length + '): ' + con.slice(0, 4).map(_escHtml).join(', ')
          + (con.length > 4 ? ' +' + (con.length - 4) : '') + '</div>';
      }}
      html += '</div>';
    }}
    claimsEl.innerHTML = html;
  }}
  const debated = data.most_debated || [];
  if (debated.length) {{
    debatedEl.innerHTML = '<div style="font-size:11px;font-weight:bold;color:var(--fg-muted);margin-bottom:4px;">MOST DEBATED</div>'
      + '<ul style="margin:0 0 0 20px;padding:0;font-size:12px;">'
      + debated.map(d => '<li>' + _escHtml(d) + '</li>').join('') + '</ul>';
  }}
}}

async function stopWikiConsensus() {{
  if (_wikiConsensusJob) {{
    await fetch('/api/jobs/' + _wikiConsensusJob, {{method: 'DELETE'}});
  }}
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
      html += '<div class="wiki-page-row" data-action="open-wiki-page" data-slug="' + slug + '">';
      html += '<div class="wp-title">' + (p.title || p.slug || '').replace(/</g, '&lt;') + '</div>';
      html += '<div class="wp-meta">' + (p.word_count || 0).toLocaleString() + ' words · ' + (p.n_sources || 0) + ' src</div>';
      html += '<div class="wp-type">' + (p.page_type || '').replace(/_/g, ' ') + '</div>';
      html += '</div>';
    }});
    html += '</div>';

    if (data.n_pages > 1) {{
      html += '<div class="catalog-pager">';
      html += '<button data-action="load-wiki-pages" data-page="' + (wikiBrowsePage - 1) + '" ' + (wikiBrowsePage <= 1 ? 'disabled' : '') + '>‹ Prev</button>';
      html += '<span>Page ' + data.page + ' of ' + data.n_pages + '  ·  ' + data.total + ' pages</span>';
      html += '<button data-action="load-wiki-pages" data-page="' + (wikiBrowsePage + 1) + '" ' + (wikiBrowsePage >= data.n_pages ? 'disabled' : '') + '>Next ›</button>';
      html += '</div>';
    }}

    list.innerHTML = html;
  }} catch (e) {{
    list.innerHTML = '<div style="padding:24px;text-align:center;color:var(--danger);">Error: ' + e.message + '</div>';
  }}
}}

// Phase 54 — track the currently-viewed wiki slug so the TOC + Copy
// Permalink know what to point at, and so the hashchange handler
// can short-circuit no-op re-renders.
let _currentWikiSlug = null;

async function openWikiPage(slug) {{
  const list = document.getElementById('wiki-browse-list');
  const detail = document.getElementById('wiki-page-detail');
  const meta = document.getElementById('wiki-page-meta');
  const content = document.getElementById('wiki-page-content');
  const toc = document.getElementById('wiki-toc');
  list.style.display = 'none';
  detail.style.display = 'block';
  content.innerHTML = '<div style="padding:24px;text-align:center;color:var(--fg-muted);">Loading...</div>';
  if (toc) toc.innerHTML = '';
  // Phase 54.3 — reset the inline "Ask this page" state when the
  // active page changes so a stale answer doesn't hang around on
  // the new page.
  const _askInput = document.getElementById('wiki-ask-input');
  const _askStatus = document.getElementById('wiki-ask-status');
  const _askStream = document.getElementById('wiki-ask-stream');
  const _askSources = document.getElementById('wiki-ask-sources');
  if (_askInput) _askInput.value = '';
  if (_askStatus) _askStatus.textContent = '';
  if (_askStream) _askStream.textContent = '';
  if (_askSources) {{ _askSources.style.display = 'none'; _askSources.innerHTML = ''; }}
  if (_wikiAskSource) {{ try {{ _wikiAskSource.close(); }} catch (e) {{}} _wikiAskSource = null; }}
  // Phase 54.5 — reset annotation textarea + status between pages so
  // a stale note from the previous page isn't mistakenly kept.
  const _annBody = document.getElementById('wiki-annotation-body');
  const _annStatus = document.getElementById('wiki-annotation-status');
  const _annTs = document.getElementById('wiki-annotation-ts');
  if (_annBody) _annBody.value = '';
  if (_annStatus) _annStatus.textContent = '';
  if (_annTs) _annTs.textContent = '';
  _currentWikiSlug = slug;

  // Phase 54 — reflect the open page in the URL hash so back/forward
  // work and permalinks are shareable. Use pushState only when the
  // hash isn't already right, to avoid loops with the hashchange
  // listener below.
  const target = '#wiki/' + encodeURIComponent(slug);
  if (window.location.hash !== target) {{
    history.pushState(null, '', target);
  }}

  try {{
    const res = await fetch('/api/wiki/page/' + encodeURIComponent(slug));
    if (!res.ok) {{
      content.innerHTML = '<div style="color:var(--danger);padding:24px;">Wiki page <code>' + slug + '</code> not found.</div>';
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

    // Phase 54.1 — render math via KaTeX auto-render if loaded.
    // Falls back silently to the raw `$...$` when KaTeX isn't
    // available (e.g. offline on first-ever page load).
    if (window.renderMathInElement) {{
      try {{
        window.renderMathInElement(content, {{
          delimiters: [
            {{ left: '$$', right: '$$', display: true }},
            {{ left: '$',  right: '$',  display: false }},
            {{ left: '\\\\(', right: '\\\\)', display: false }},
            {{ left: '\\\\[', right: '\\\\]', display: true }},
          ],
          throwOnError: false,
        }});
      }} catch (e) {{ /* render failure is non-fatal */ }}
    }}

    // Phase 54.1 — staleness banner (surfaces wiki_pages.needs_rewrite).
    const banner = document.getElementById('wiki-stale-banner');
    if (banner) banner.style.display = data.needs_rewrite ? 'block' : 'none';

    // Phase 54 — build the TOC from the rendered headings.
    _buildWikiTOC();

    // Phase 54.2 — Related pages + Referenced-by (backlinks) panels.
    // Fire both in parallel; render each section only when it returns
    // non-empty. No blocking — the main content paints first.
    _loadWikiRelated(slug);
    _loadWikiBacklinks(slug);

    // Phase 54.4 — Facts from the corpus (concept pages only).
    _renderWikiFacts(data);

    // Phase 54.5 — load the user's "My take" annotation for this page.
    _loadWikiAnnotation(slug);
    // Phase 54 — honour `?h=<heading-id>` in the hash if present.
    const m = (window.location.hash || '').match(/\\?h=([^&]+)$/);
    if (m) {{
      const el = document.getElementById(decodeURIComponent(m[1]));
      if (el) el.scrollIntoView({{ behavior: 'instant', block: 'start' }});
    }} else {{
      content.scrollTop = 0;
    }}
  }} catch (e) {{
    content.innerHTML = '<div style="color:var(--danger);">Error: ' + e.message + '</div>';
  }}
}}

// Phase 54 — post-render TOC builder. Scans h2/h3/h4 inside
// #wiki-page-content, emits a sticky sidebar nav with click-to-
// scroll handlers. Zero-cost for pages without headings — the
// sidebar just renders empty.
function _buildWikiTOC() {{
  const host = document.getElementById('wiki-toc');
  const content = document.getElementById('wiki-page-content');
  if (!host || !content) return;
  const heads = content.querySelectorAll('h2, h3, h4');
  if (!heads.length) {{ host.innerHTML = ''; return; }}
  let html = '<div class="wiki-toc-heading">On this page</div><ol class="wiki-toc-list">';
  heads.forEach(h => {{
    if (!h.id) return;
    const cls = 'wiki-toc-' + h.tagName.toLowerCase();
    html += '<li class="' + cls + '"><a data-heading="' + h.id + '">' +
            escapeHtml(h.textContent) + '</a></li>';
  }});
  html += '</ol>';
  host.innerHTML = html;
}}
// Delegated click → smooth-scroll the target heading into view.
document.addEventListener('click', (evt) => {{
  const a = evt.target.closest && evt.target.closest('#wiki-toc [data-heading]');
  if (!a) return;
  evt.preventDefault();
  const el = document.getElementById(a.dataset.heading);
  if (el) {{
    el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    // Also update the hash to make the heading shareable.
    if (_currentWikiSlug) {{
      const target = '#wiki/' + encodeURIComponent(_currentWikiSlug) +
                     '?h=' + encodeURIComponent(a.dataset.heading);
      history.replaceState(null, '', target);
    }}
  }}
}});

// Phase 54.3 — "Ask this page" inline RAG. Hits the new
// /api/wiki/page/<slug>/ask endpoint, streams tokens into the
// page's inline chat box via the same SSE contract the book-
// reader uses.
let _wikiAskSource = null;
async function askWikiPage() {{
  const q = (document.getElementById('wiki-ask-input').value || '').trim();
  if (!q || !_currentWikiSlug) return;
  const broaden = document.getElementById('wiki-ask-broaden').checked;
  const status = document.getElementById('wiki-ask-status');
  const streamEl = document.getElementById('wiki-ask-stream');
  const sourcesEl = document.getElementById('wiki-ask-sources');
  const submit = document.getElementById('wiki-ask-submit');

  status.textContent = 'Retrieving and generating…';
  streamEl.textContent = '';
  sourcesEl.style.display = 'none';
  sourcesEl.innerHTML = '';
  submit.disabled = true;

  const fd = new FormData();
  fd.append('question', q);
  if (broaden) fd.append('broaden', 'true');

  let data;
  try {{
    const res = await fetch('/api/wiki/page/' + encodeURIComponent(_currentWikiSlug) + '/ask',
                            {{ method: 'POST', body: fd }});
    data = await res.json();
  }} catch (e) {{
    status.textContent = 'Error: ' + e.message;
    submit.disabled = false;
    return;
  }}

  if (_wikiAskSource) {{ try {{ _wikiAskSource.close(); }} catch (e) {{}} _wikiAskSource = null; }}
  const source = new EventSource('/api/stream/' + data.job_id);
  _wikiAskSource = source;
  let collected = null;
  let scope = null;

  source.onmessage = function(e) {{
    const evt = JSON.parse(e.data);
    if (evt.type === 'token') {{
      streamEl.textContent += evt.text;
    }} else if (evt.type === 'progress') {{
      status.textContent = evt.detail || evt.stage;
    }} else if (evt.type === 'sources') {{
      collected = evt.sources;
      scope = evt.scope || 'corpus';
      status.textContent = 'Generating from ' + (evt.n || collected.length) +
                           ' passage(s) · scope=' + scope;
    }} else if (evt.type === 'completed') {{
      status.textContent = 'Done' + (scope ? ' · scope=' + scope : '');
      submit.disabled = false;
      if (collected && collected.length) {{
        let html = '<div style="font-weight:600;color:var(--fg);margin-bottom:6px;">Sources (' +
                   collected.length + ')</div><ol>';
        collected.forEach(s => {{ html += '<li>' + escapeHtml(s) + '</li>'; }});
        html += '</ol>';
        sourcesEl.innerHTML = html;
        sourcesEl.style.display = 'block';
      }}
      source.close(); _wikiAskSource = null;
    }} else if (evt.type === 'error') {{
      status.textContent = 'Error: ' + evt.message;
      submit.disabled = false;
      source.close(); _wikiAskSource = null;
    }}
  }};
  source.onerror = function() {{
    status.textContent = 'Stream disconnected';
    submit.disabled = false;
    try {{ source.close(); }} catch (e) {{}}
    _wikiAskSource = null;
  }};
}}

// Phase 54.5 — "My take" annotation: load / save / delete.
async function _loadWikiAnnotation(slug) {{
  const body = document.getElementById('wiki-annotation-body');
  const ts = document.getElementById('wiki-annotation-ts');
  const status = document.getElementById('wiki-annotation-status');
  if (!body) return;
  body.value = '';
  if (ts) ts.textContent = '';
  if (status) status.textContent = '';
  try {{
    const res = await fetch('/api/wiki/page/' + encodeURIComponent(slug) +
                            '/annotation');
    if (!res.ok) return;
    const d = await res.json();
    body.value = d.body || '';
    if (d.updated_at && ts) {{
      ts.textContent = 'last saved ' + d.updated_at.substring(0, 16).replace('T', ' ');
    }}
  }} catch (e) {{ /* silent — empty textarea is the fallback */ }}
}}

async function saveWikiAnnotation() {{
  if (!_currentWikiSlug) return;
  const body = document.getElementById('wiki-annotation-body');
  const status = document.getElementById('wiki-annotation-status');
  const ts = document.getElementById('wiki-annotation-ts');
  if (!body) return;
  const fd = new FormData();
  fd.append('body', body.value || '');
  status.textContent = 'saving…';
  try {{
    const res = await fetch(
      '/api/wiki/page/' + encodeURIComponent(_currentWikiSlug) + '/annotation',
      {{ method: 'PUT', body: fd }},
    );
    const d = await res.json();
    if (d.deleted) {{
      status.textContent = 'cleared';
      if (ts) ts.textContent = '';
    }} else {{
      status.textContent = 'saved';
      if (ts && d.updated_at) {{
        ts.textContent = 'last saved ' + d.updated_at.substring(0, 16).replace('T', ' ');
      }}
    }}
    setTimeout(() => {{ if (status.textContent === 'saved' || status.textContent === 'cleared') status.textContent = ''; }}, 2000);
  }} catch (e) {{
    status.textContent = 'save failed: ' + e.message;
  }}
}}

async function deleteWikiAnnotation() {{
  const body = document.getElementById('wiki-annotation-body');
  if (body) body.value = '';
  await saveWikiAnnotation();
}}

// Phase 54.5 — j/k navigation through the wiki browse list.
// Only active when the browse-list pane is visible, nobody's typing
// in a form field, and no modifier keys are held.
let _wikiListIdx = -1;
function _wikiListItems() {{
  return document.querySelectorAll(
    '#wiki-browse-list [data-slug], #wiki-browse-list tr[data-slug], #wiki-browse-list li[data-slug]'
  );
}}
function _setWikiListActive(idx) {{
  const items = _wikiListItems();
  if (!items.length) return;
  _wikiListIdx = Math.max(0, Math.min(idx, items.length - 1));
  items.forEach((n, i) => n.classList.toggle('active-row', i === _wikiListIdx));
  items[_wikiListIdx].scrollIntoView({{ block: 'nearest' }});
}}
document.addEventListener('keydown', (evt) => {{
  // Only fire when the browse list is on-screen and no other
  // input is focused.
  const listVisible = (() => {{
    const el = document.getElementById('wiki-browse-list');
    if (!el) return false;
    if (el.style.display === 'none') return false;
    const detail = document.getElementById('wiki-page-detail');
    return !(detail && detail.style.display !== 'none');
  }})();
  if (!listVisible) return;
  const tag = (evt.target && evt.target.tagName) || '';
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if (evt.metaKey || evt.ctrlKey || evt.altKey) return;
  if (evt.key === 'j') {{
    evt.preventDefault();
    _setWikiListActive(_wikiListIdx < 0 ? 0 : _wikiListIdx + 1);
  }} else if (evt.key === 'k') {{
    evt.preventDefault();
    _setWikiListActive(_wikiListIdx < 0 ? 0 : _wikiListIdx - 1);
  }} else if (evt.key === 'Enter') {{
    const items = _wikiListItems();
    if (_wikiListIdx >= 0 && items[_wikiListIdx]) {{
      const slug = items[_wikiListIdx].dataset.slug;
      if (slug) {{
        evt.preventDefault();
        window.location.hash = '#wiki/' + encodeURIComponent(slug);
      }}
    }}
  }}
}});

// Phase 54.4 — Render the "Facts from the corpus" block using the
// triples the API attaches to concept pages (server-side join
// against the knowledge_graph table). Hidden when the page isn't a
// concept or no triples matched.
function _renderWikiFacts(data) {{
  const block = document.getElementById('wiki-facts-block');
  const list = document.getElementById('wiki-facts-list');
  const link = document.getElementById('wiki-facts-kg-link');
  if (!block || !list) return;
  const triples = (data && data.related_triples) || [];
  if (data.page_type !== 'concept' || !triples.length) {{
    block.style.display = 'none';
    return;
  }}
  // "Open in graph" — use the KG modal's existing share-URL convention
  // so the concept opens pinned at the centre of the 3D orbit view.
  if (link && typeof KG_THEMES !== 'undefined') {{
    try {{
      const shareState = {{
        f: {{ s: '', p: '', o: '' }},
        c: {{ rx: -0.22, ry: 0.55, d: 850 }},
        p: [ (data.title || _currentWikiSlug).toLowerCase() ],
      }};
      const hash = '#kg=' + btoa(unescape(encodeURIComponent(
        JSON.stringify(shareState))));
      link.href = hash;
      // Also pre-fill the subject filter when the user lands in the KG
      link.addEventListener('click', (evt) => {{
        evt.preventDefault();
        const subj = document.getElementById('kg-subject');
        if (subj) subj.value = data.title || _currentWikiSlug;
        window.location.hash = hash;
      }}, {{ once: true }});
    }} catch (e) {{ /* leave link as-is on encode failure */ }}
  }}

  // Classify each triple by predicate family so the left-border
  // matches the KG colour scheme. Reuses KG_PREDICATE_FAMILIES
  // from the KG modal script if it's loaded.
  function _family(predicate) {{
    try {{
      if (typeof _kgPredicateFamily === 'function') return _kgPredicateFamily(predicate);
    }} catch (e) {{}}
    return 'other';
  }}

  list.innerHTML = triples.slice(0, 40).map(t => {{
    const fam = _family(t.predicate || '');
    const sent = t.source_sentence || '';
    const docTitle = t.source_title || '';
    const tipParts = [];
    if (docTitle) tipParts.push(docTitle);
    if (sent) tipParts.push('“' + sent + '”');
    const tipAttr = tipParts.length
      ? ' title="' + escapeHtml(tipParts.join(' — ')) + '"'
      : '';
    let srcHtml = '';
    if (sent) {{
      const short = sent.length > 140 ? sent.substring(0, 140) + '…' : sent;
      srcHtml = '<span class="wf-src">' + escapeHtml(short) + '</span>';
    }}
    return (
      '<li class="wf-fam-' + fam + '"' + tipAttr + '>' +
      '<span class="wf-subject">' + escapeHtml(t.subject || '') + '</span>' +
      '<span class="wf-pred"> ' + escapeHtml(t.predicate || '') + ' </span>' +
      '<span class="wf-object">' + escapeHtml(t.object || '') + '</span>' +
      srcHtml +
      '</li>'
    );
  }}).join('');
  block.style.display = 'block';
}}

// Phase 54.2 — Related / backlinks loaders.
async function _loadWikiRelated(slug) {{
  const block = document.getElementById('wiki-related-block');
  const list = document.getElementById('wiki-related-list');
  if (!block || !list) return;
  block.style.display = 'none';
  try {{
    const res = await fetch('/api/wiki/page/' + encodeURIComponent(slug) +
                            '/related?limit=5');
    if (!res.ok) return;
    const items = await res.json();
    if (!Array.isArray(items) || items.length === 0) return;
    list.innerHTML = items.map(t =>
      '<li>' +
      '<a href="#wiki/' + encodeURIComponent(t.slug) + '">' +
         escapeHtml(t.title) + '</a>' +
      '<span class="wp-type">' + (t.page_type || '').replace(/_/g, ' ') + '</span>' +
      '</li>'
    ).join('');
    block.style.display = 'block';
  }} catch (e) {{ /* silent — related pages is best-effort */ }}
}}

async function _loadWikiBacklinks(slug) {{
  const block = document.getElementById('wiki-backlinks-block');
  const list = document.getElementById('wiki-backlinks-list');
  if (!block || !list) return;
  block.style.display = 'none';
  try {{
    const res = await fetch('/api/wiki/page/' + encodeURIComponent(slug) +
                            '/backlinks');
    if (!res.ok) return;
    const items = await res.json();
    if (!Array.isArray(items) || items.length === 0) return;
    list.innerHTML = items.map(t => {{
      const alt = t.alt && t.alt !== t.from_slug
        ? ' <span class="wp-alt">&ldquo;' + escapeHtml(t.alt) + '&rdquo;</span>'
        : '';
      return '<li>' +
        '<a href="#wiki/' + encodeURIComponent(t.from_slug) + '">' +
           escapeHtml(t.from_title || t.from_slug) + '</a>' +
        alt +
        '</li>';
    }}).join('');
    block.style.display = 'block';
  }} catch (e) {{ /* silent */ }}
}}

function closeWikiPageDetail() {{
  document.getElementById('wiki-browse-list').style.display = 'block';
  document.getElementById('wiki-page-detail').style.display = 'none';
  _currentWikiSlug = null;
  if ((window.location.hash || '').startsWith('#wiki/')) {{
    history.pushState(null, '', '#wiki');
  }}
}}

function copyWikiPermalink() {{
  if (!_currentWikiSlug) return;
  const url = window.location.origin + window.location.pathname +
              '#wiki/' + encodeURIComponent(_currentWikiSlug);
  navigator.clipboard.writeText(url).then(
    () => {{
      const btn = event.target.closest('button');
      if (btn) {{
        const prev = btn.innerHTML;
        btn.innerHTML = '&check; Copied';
        setTimeout(() => {{ btn.innerHTML = prev; }}, 1500);
      }}
    }},
    () => prompt('Copy this link:', url),
  );
}}

function openWikiModal() {{
  openModal('wiki-modal');
  setTimeout(() => document.getElementById('wiki-query-input').focus(), 100);
}}

// Phase 54 — hash router for the wiki SPA surface.
// Supports:
//   #wiki             → open modal on Browse list
//   #wiki/<slug>      → open modal on page detail
//   #wiki/<slug>?h=X  → open page detail + scroll to heading id X
function _wikiRouteFromHash() {{
  const h = window.location.hash || '';
  if (!h.startsWith('#wiki')) return false;
  const rest = h.substring(5); // strip "#wiki"
  const modal = document.getElementById('wiki-modal');
  if (modal && modal.style.display !== 'flex' && modal.style.display !== 'block') {{
    openModal('wiki-modal');
  }}
  switchWikiTab('wiki-browse');
  if (!rest || rest === '' || rest === '/') {{
    closeWikiPageDetail();
    loadWikiPages(1);
    return true;
  }}
  // rest is "/slug" optionally followed by "?h=heading"
  const m = rest.match(/^\\/([^?]+)(?:\\?h=(.+))?$/);
  if (!m) return true;
  const slug = decodeURIComponent(m[1]);
  if (slug !== _currentWikiSlug) {{
    openWikiPage(slug);
  }}
  return true;
}}
window.addEventListener('hashchange', _wikiRouteFromHash);
window.addEventListener('DOMContentLoaded', () => {{
  if ((window.location.hash || '').startsWith('#wiki')) {{
    _wikiRouteFromHash();
  }}
}});

// ── Phase 54 — Ctrl-K / Cmd-K wiki command palette ─────────────────
// Fuzzy-filter over wiki page titles + slugs, keyboard-navigable.
// Titles are fetched once per session and cached in-memory; the
// wiki size is bounded (a few hundred pages typical) so shipping the
// whole list is fine.

let _wikiTitlesCache = null;
let _wikiPaletteIdx = 0;

async function _loadWikiTitles() {{
  if (_wikiTitlesCache) return _wikiTitlesCache;
  try {{
    const res = await fetch('/api/wiki/titles');
    if (res.ok) _wikiTitlesCache = await res.json();
    else _wikiTitlesCache = [];
  }} catch (e) {{ _wikiTitlesCache = []; }}
  return _wikiTitlesCache;
}}

// Tiny fuzzy scorer — substring hit dominates; char-in-order fallback
// for typos / abbreviations. Good enough for a few hundred items.
function _wikiFuzzyScore(needle, hay) {{
  if (!needle) return 1;
  needle = needle.toLowerCase();
  hay = (hay || '').toLowerCase();
  const idx = hay.indexOf(needle);
  if (idx !== -1) return 1000 - idx;  // earlier hits rank higher
  let hi = 0, hits = 0;
  for (const ch of needle) {{
    const i = hay.indexOf(ch, hi);
    if (i === -1) return 0;
    hits += 1;
    hi = i + 1;
  }}
  return hits;
}}

async function _renderWikiPalette() {{
  const q = document.getElementById('wiki-palette-input').value.trim();
  const titles = await _loadWikiTitles();
  let items;
  if (!q) {{
    items = titles.slice(0, 10);
  }} else {{
    items = titles
      .map(t => ({{
        ...t,
        _s: _wikiFuzzyScore(q, t.title) + _wikiFuzzyScore(q, t.slug) * 0.5,
      }}))
      .filter(t => t._s > 0)
      .sort((a, b) => b._s - a._s)
      .slice(0, 10);
  }}
  _wikiPaletteIdx = 0;
  const host = document.getElementById('wiki-palette-results');
  if (!items.length) {{
    host.innerHTML = '<li class="wiki-palette-empty">No pages match</li>';
    return;
  }}
  host.innerHTML = items.map((t, i) => {{
    const cls = (i === 0) ? 'wiki-palette-item active' : 'wiki-palette-item';
    return '<li class="' + cls + '" data-slug="' + t.slug + '">' +
           '<span class="wp-title">' + escapeHtml(t.title) + '</span>' +
           '<span class="wp-type">' + (t.page_type || '').replace(/_/g, ' ') + '</span>' +
           '</li>';
  }}).join('');
}}

function _wikiPaletteKey(evt) {{
  const host = document.getElementById('wiki-palette-results');
  const items = host.querySelectorAll('.wiki-palette-item');
  if (evt.key === 'Escape') {{
    evt.preventDefault(); closeWikiPalette(); return;
  }}
  if (!items.length) return;
  if (evt.key === 'ArrowDown' || evt.key === 'ArrowUp') {{
    evt.preventDefault();
    items[_wikiPaletteIdx].classList.remove('active');
    _wikiPaletteIdx = (evt.key === 'ArrowDown')
      ? (_wikiPaletteIdx + 1) % items.length
      : (_wikiPaletteIdx - 1 + items.length) % items.length;
    items[_wikiPaletteIdx].classList.add('active');
    items[_wikiPaletteIdx].scrollIntoView({{ block: 'nearest' }});
    return;
  }}
  if (evt.key === 'Enter') {{
    evt.preventDefault();
    const slug = items[_wikiPaletteIdx].dataset.slug;
    closeWikiPalette();
    window.location.hash = '#wiki/' + encodeURIComponent(slug);
    return;
  }}
}}

function openWikiPalette() {{
  const modal = document.getElementById('wiki-palette');
  if (!modal) return;
  modal.style.display = 'flex';
  const input = document.getElementById('wiki-palette-input');
  input.value = '';
  _renderWikiPalette();
  setTimeout(() => input.focus(), 30);
}}

function closeWikiPalette() {{
  const modal = document.getElementById('wiki-palette');
  if (modal) modal.style.display = 'none';
}}

// Delegated click on a palette row → navigate.
document.addEventListener('click', (evt) => {{
  const item = evt.target.closest && evt.target.closest('.wiki-palette-item');
  if (!item) return;
  const slug = item.dataset.slug;
  if (!slug) return;
  closeWikiPalette();
  window.location.hash = '#wiki/' + encodeURIComponent(slug);
}});

// Global Ctrl-K / Cmd-K — open the palette. Skip if the user is
// typing in a textarea / input (except the palette itself, whose
// input handler takes arrow keys + Escape via onkeydown).
document.addEventListener('keydown', (evt) => {{
  if ((evt.metaKey || evt.ctrlKey) && (evt.key === 'k' || evt.key === 'K')) {{
    // Allow command palette from anywhere, including other inputs.
    evt.preventDefault();
    openWikiPalette();
  }}
}});

// ── Phase 54.1 — keyboard shortcut router (?, /, g-chord) ─────────────
// A small state machine for the "g then h / g then w" two-key chord,
// plus single-key shortcuts that only fire outside form fields so we
// don't swallow user typing.

let _kbChord = null;            // 'g' while waiting for the second key
let _kbChordTimer = null;

function _inFormField(el) {{
  if (!el) return false;
  const tag = (el.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
  return !!el.isContentEditable;
}}

function openKbHelp() {{
  const el = document.getElementById('kb-help');
  if (el) el.style.display = 'flex';
}}
function closeKbHelp() {{
  const el = document.getElementById('kb-help');
  if (el) el.style.display = 'none';
}}

document.addEventListener('keydown', (evt) => {{
  // Chord continuation always fires, even inside form fields,
  // because we only enter a chord state outside form fields below.
  if (_kbChord === 'g') {{
    _kbChord = null;
    if (_kbChordTimer) {{ clearTimeout(_kbChordTimer); _kbChordTimer = null; }}
    if (evt.key === 'w' || evt.key === 'W') {{
      evt.preventDefault();
      window.location.hash = '#wiki';
      return;
    }}
    if (evt.key === 'h' || evt.key === 'H') {{
      evt.preventDefault();
      // Close every open modal overlay.
      document.querySelectorAll('.modal-overlay').forEach(m => {{
        m.style.display = 'none';
      }});
      history.pushState(null, '', window.location.pathname);
      return;
    }}
    // Unknown second key — drop the chord and fall through.
  }}
  if (_inFormField(evt.target)) return;
  // Don't intercept when modifier keys are held — Ctrl-K etc. has
  // its own handler above.
  if (evt.metaKey || evt.ctrlKey || evt.altKey) return;

  if (evt.key === '?') {{
    evt.preventDefault();
    const el = document.getElementById('kb-help');
    if (el && el.style.display === 'flex') closeKbHelp(); else openKbHelp();
    return;
  }}
  if (evt.key === 'Escape') {{
    closeKbHelp();
    // Let the rest of the app's Escape handlers (modal, menu) run too.
    return;
  }}
  if (evt.key === '/') {{
    evt.preventDefault();
    openWikiPalette();
    return;
  }}
  if (evt.key === 'g' || evt.key === 'G') {{
    _kbChord = 'g';
    // Abandon the chord if the user doesn't follow up within 1.2 s.
    if (_kbChordTimer) clearTimeout(_kbChordTimer);
    _kbChordTimer = setTimeout(() => {{ _kbChord = null; }}, 1200);
    return;
  }}
}});

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

// ── Phase 36: Tools modal (CLI-parity panel) ──────────────────────────
// Four tabs, four flows:
//   Search     → POST /api/search/(query|similar)  (JSON)
//   Synthesize → POST /api/ask/synthesize + SSE    (streaming)
//   Topics     → GET  /api/catalog/topics[?name=]  (JSON)
//   Corpus     → POST /api/corpus/(enrich|expand) + SSE (subprocess)
let _toolsCorpusJob = null;
let _toolsSynthJob = null;

function openToolsModal() {{
  openModal('tools-modal');
  switchToolsTab('tl-search');
  setTimeout(() => document.getElementById('tl-search-q').focus(), 100);
}}

function switchToolsTab(name) {{
  // Only flip the TOP-level Tools tabs (not any inner Corpus-subtabs,
  // which carry data-ctab instead of data-tab so they don't collide).
  document.querySelectorAll('#tools-modal > .tabs > .tab').forEach(t => {{
    t.classList.toggle('active', t.dataset.tab === name);
  }});
  ['tl-search', 'tl-synth', 'tl-topics', 'tl-corpus'].forEach(n => {{
    const pane = document.getElementById(n + '-pane');
    if (pane) pane.style.display = (n === name) ? 'block' : 'none';
  }});
  if (name === 'tl-topics') loadToolTopics();
  if (name === 'tl-corpus') {{
    // Default to Enrich sub-tab on first open
    switchCorpusTab('corp-enrich');
    loadCorpusTopicList();
  }}
}}


// Phase 46.E — inner tabs for the Corpus pane (Enrich / Expand-citations /
// Expand-by-Author). Uses data-ctab to avoid colliding with the outer
// Tools tabs' data-tab.
function switchCorpusTab(name) {{
  document.querySelectorAll('#tl-corpus-pane .tab').forEach(t => {{
    t.classList.toggle('active', t.dataset.ctab === name);
  }});
  ['corp-enrich', 'corp-cites', 'corp-author'].forEach(n => {{
    const pane = document.getElementById(n + '-pane');
    if (pane) pane.style.display = (n === name) ? 'block' : 'none';
  }});
}}


// Phase 46.E — author search-as-you-type for the Expand-by-Author panel.
// Debounced 200ms. Hits /api/catalog/authors?q=…&limit=15 and renders a
// clickable list; clicking a row sets window._selectedExpandAuthorName
// and populates the input + orcid + selected-line.
//
// Phase 54.6.1 — clicking a row was dead because the old implementation
// inline-interpolated JSON.stringify(a) into an onclick attribute wrapped
// in double quotes. The inner quotes terminated the attribute and
// corrupted everything after. Fixed by caching the author list and
// dispatching via event delegation (data-idx → closure lookup).
let _authorSearchTimer = null;
let _lastAuthorResults = [];
window._selectedExpandAuthorName = null;

function _escHtml(s) {{
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}}

function onExpandAuthorSearchInput(_event) {{
  if (_authorSearchTimer) clearTimeout(_authorSearchTimer);
  _authorSearchTimer = setTimeout(runExpandAuthorSearch, 200);
}}

async function runExpandAuthorSearch() {{
  const q = document.getElementById('tl-eauth-q').value.trim();
  const box = document.getElementById('tl-eauth-results');
  try {{
    const url = '/api/catalog/authors?limit=15'
              + (q.length >= 2 ? '&q=' + encodeURIComponent(q) : '');
    const res = await fetch(url);
    const data = await res.json();
    const authors = data.authors || [];
    _lastAuthorResults = authors;
    if (!authors.length) {{
      box.style.display = 'block';
      box.innerHTML = '<div style="padding:10px;color:var(--fg-muted);">No authors match.</div>';
      return;
    }}
    const rows = authors.map((a, i) => {{
      const safe = _escHtml(a.name);
      const orcidBit = a.orcid
        ? ` <span style="color:var(--fg-muted);">orcid ${{_escHtml(a.orcid)}}</span>`
        : '';
      return `<div class="eauth-row" data-idx="${{i}}"
        style="padding:6px 10px;cursor:pointer;border-bottom:1px solid var(--border);"
        onmouseenter="this.style.background='var(--accent-light,#eef2ff)';"
        onmouseleave="this.style.background='';">
        <strong>${{safe}}</strong>${{orcidBit}}
        <div style="color:var(--fg-muted);font-size:11px;">
          ${{a.n_papers}} paper${{a.n_papers === 1 ? '' : 's'}} ·
          ${{a.n_citations}} citation${{a.n_citations === 1 ? '' : 's'}} in corpus
        </div></div>`;
    }}).join('');
    box.innerHTML = rows;
    box.style.display = 'block';
    // Event delegation — robust to any characters in author.name.
    box.querySelectorAll('.eauth-row').forEach(row => {{
      row.addEventListener('click', () => {{
        const idx = parseInt(row.dataset.idx, 10);
        const author = _lastAuthorResults[idx];
        if (author) selectExpandAuthor(author);
      }});
    }});
  }} catch (exc) {{
    box.style.display = 'block';
    box.innerHTML = `<div style="padding:10px;color:var(--danger,#c53030);">Search failed: ${{exc}}</div>`;
  }}
}}

function selectExpandAuthor(author) {{
  window._selectedExpandAuthorName = author.name;
  document.getElementById('tl-eauth-q').value = author.name;
  if (author.orcid) document.getElementById('tl-eauth-orcid').value = author.orcid;
  document.getElementById('tl-eauth-results').style.display = 'none';
  document.getElementById('tl-eauth-selected').innerHTML =
    '<span style="color:var(--success,#059669);">Selected:</span> <strong>'
    + author.name + '</strong>'
    + (author.orcid ? ' <code>' + author.orcid + '</code>' : '')
    + ' — ' + author.n_papers + ' paper(s), ' + author.n_citations + ' citation(s) in this corpus';
}}


// ── Phase 54.6.1 — Expand-by-Author preview modal ──────────────────────
// openExpandAuthorPreview() pulls the params from the panel, POSTs to
// /api/corpus/expand-author/preview, and renders a checkboxed table of
// candidates. User cherry-picks → eapDownloadSelected() ships the DOIs
// to /api/corpus/expand-author/download-selected (SSE-streamed CLI).
let _eapCandidates = [];   // all candidates from the last preview
let _eapSelected = new Set(); // doi set — source of truth for selection

async function openExpandAuthorPreview() {{
  const name = document.getElementById('tl-eauth-q').value.trim();
  const orcid = document.getElementById('tl-eauth-orcid').value.trim();
  if (!name && !orcid) {{
    alert('Type an author name (or ORCID) first.');
    return;
  }}
  // Reset modal state
  _eapCandidates = [];
  _eapSelected = new Set();
  document.getElementById('eap-loading').style.display = 'block';
  document.getElementById('eap-error').style.display = 'none';
  document.getElementById('eap-content').style.display = 'none';
  document.getElementById('eap-log').style.display = 'none';
  document.getElementById('eap-log').textContent = '';
  document.getElementById('eap-status').textContent = '';
  openModal('expand-author-preview-modal');

  // Gather params from the panel
  const fd = new FormData();
  fd.append('name', name);
  if (orcid) fd.append('orcid', orcid);
  const yFrom = parseInt(document.getElementById('tl-eauth-yfrom').value || '0', 10);
  const yTo = parseInt(document.getElementById('tl-eauth-yto').value || '0', 10);
  if (yFrom) fd.append('year_from', yFrom);
  if (yTo) fd.append('year_to', yTo);
  const limit = parseInt(document.getElementById('tl-eauth-limit').value || '0', 10);
  if (limit > 0) fd.append('limit', limit);
  fd.append('strict_author', document.getElementById('tl-eauth-strict').checked);
  fd.append('all_matches', document.getElementById('tl-eauth-all').checked);
  const relq = document.getElementById('tl-eauth-relq').value.trim();
  if (relq) fd.append('relevance_query', relq);

  try {{
    const res = await fetch('/api/corpus/expand-author/preview', {{
      method: 'POST',
      body: fd,
    }});
    if (!res.ok) {{
      const detail = await res.text();
      throw new Error('HTTP ' + res.status + ': ' + detail);
    }}
    const data = await res.json();
    _eapCandidates = data.candidates || [];
    const info = data.info || {{}};
    // Pre-select everything above the default threshold; otherwise all.
    const threshold = info.relevance_threshold || 0.0;
    document.getElementById('eap-threshold').value = threshold.toFixed(2);
    _eapCandidates.forEach(c => {{
      if (c.doi && (c.relevance_score == null || c.relevance_score >= threshold)) {{
        _eapSelected.add(c.doi);
      }}
    }});
    // Render info line
    const pickedAuthors = (info.picked_authors || [])
      .map(a => a.display_name || a.name || '').filter(Boolean).slice(0, 3).join(', ');
    const pickedSuffix = pickedAuthors
      ? ` · matched author(s): ${{_escHtml(pickedAuthors)}}` : '';
    document.getElementById('eap-info').innerHTML =
      `Found <strong>${{info.merged || 0}}</strong> paper(s) `
      + `(<span title="from OpenAlex canonical author search">${{info.openalex || 0}} OA</span> + `
      + `<span title="from Crossref surname search — may include false positives for common names">${{info.crossref_extra || 0}} CR</span>), `
      + `dropped <strong>${{info.dedup_dropped || 0}}</strong> already in corpus`
      + (info.relevance_query_used
          ? ` · relevance anchor: <code>${{_escHtml(info.relevance_query_used)}}</code>`
          : ` · no relevance scoring`)
      + pickedSuffix + '.';
    document.getElementById('eap-loading').style.display = 'none';
    if (!_eapCandidates.length) {{
      document.getElementById('eap-error').style.display = 'block';
      document.getElementById('eap-error').textContent =
        'No candidates returned. Everything may already be in your corpus, or the search found nothing.';
      return;
    }}
    document.getElementById('eap-content').style.display = 'block';
    eapRender();
  }} catch (exc) {{
    document.getElementById('eap-loading').style.display = 'none';
    document.getElementById('eap-error').style.display = 'block';
    document.getElementById('eap-error').textContent = 'Preview failed: ' + exc.message;
  }}
}}

function eapRender() {{
  const tbody = document.getElementById('eap-tbody');
  const sort = document.getElementById('eap-sort').value;
  const sorted = _eapCandidates.slice();
  if (sort === 'year') {{
    sorted.sort((a, b) => (b.year || 0) - (a.year || 0));
  }} else if (sort === 'title') {{
    sorted.sort((a, b) => (a.title || '').localeCompare(b.title || ''));
  }} else {{
    sorted.sort((a, b) => (b.relevance_score || -1) - (a.relevance_score || -1));
  }}
  const rows = sorted.map(c => {{
    const checked = _eapSelected.has(c.doi) ? 'checked' : '';
    const scoreText = (c.relevance_score == null)
      ? '<span style="color:var(--fg-muted);">—</span>'
      : c.relevance_score.toFixed(3);
    const authors = (c.authors || []).slice(0, 3).join(', ')
      + ((c.authors || []).length > 3 ? ` +${{c.authors.length - 3}}` : '');
    const doi = c.doi
      ? `<a href="https://doi.org/${{_escHtml(c.doi)}}" target="_blank" rel="noopener" style="color:var(--accent);text-decoration:none;font-family:ui-monospace,monospace;font-size:10px;" onclick="event.stopPropagation();">${{_escHtml(c.doi)}}</a>`
      : '<span style="color:var(--fg-muted);">(no DOI)</span>';
    return `<tr style="border-top:1px solid var(--border);cursor:pointer;" data-doi="${{_escHtml(c.doi || '')}}">
      <td style="padding:6px 8px;"><input type="checkbox" class="eap-row-cb" ${{checked}}
           data-doi="${{_escHtml(c.doi || '')}}" onclick="event.stopPropagation();"></td>
      <td style="padding:6px 8px;">
        <div style="font-weight:500;">${{_escHtml(c.title || '(untitled)')}}</div>
        <div style="font-size:10px;margin-top:2px;">${{doi}}</div>
      </td>
      <td style="padding:6px 8px;color:var(--fg-muted);">${{_escHtml(authors)}}</td>
      <td style="padding:6px 8px;color:var(--fg-muted);">${{c.year || '—'}}</td>
      <td style="padding:6px 8px;font-family:ui-monospace,monospace;">${{scoreText}}</td>
    </tr>`;
  }}).join('');
  tbody.innerHTML = rows;
  // Row click toggles the row's checkbox (but not on link click — event.stopPropagation above).
  tbody.querySelectorAll('tr').forEach(row => {{
    row.addEventListener('click', () => {{
      const cb = row.querySelector('.eap-row-cb');
      if (!cb || !cb.dataset.doi) return;
      cb.checked = !cb.checked;
      if (cb.checked) _eapSelected.add(cb.dataset.doi);
      else _eapSelected.delete(cb.dataset.doi);
      eapUpdateCount();
    }});
  }});
  tbody.querySelectorAll('.eap-row-cb').forEach(cb => {{
    cb.addEventListener('change', () => {{
      if (!cb.dataset.doi) return;
      if (cb.checked) _eapSelected.add(cb.dataset.doi);
      else _eapSelected.delete(cb.dataset.doi);
      eapUpdateCount();
    }});
  }});
  eapUpdateCount();
}}

function eapUpdateCount() {{
  const tot = _eapCandidates.length;
  const sel = _eapSelected.size;
  document.getElementById('eap-selected-count').textContent =
    `${{sel}} of ${{tot}} selected`;
  const hdr = document.getElementById('eap-header-cb');
  if (hdr) {{
    hdr.checked = sel > 0 && sel === tot;
    hdr.indeterminate = sel > 0 && sel < tot;
  }}
}}

function eapSelectAll(on) {{
  _eapSelected = new Set();
  if (on) {{
    _eapCandidates.forEach(c => {{ if (c.doi) _eapSelected.add(c.doi); }});
  }}
  eapRender();
}}

function eapSelectByThreshold() {{
  const thr = parseFloat(document.getElementById('eap-threshold').value || '0');
  _eapSelected = new Set();
  _eapCandidates.forEach(c => {{
    if (!c.doi) return;
    if (c.relevance_score == null || c.relevance_score >= thr) {{
      _eapSelected.add(c.doi);
    }}
  }});
  eapRender();
}}

async function eapDownloadSelected() {{
  if (!_eapSelected.size) {{
    alert('Pick at least one paper (or use "Select all").');
    return;
  }}
  const chosen = _eapCandidates.filter(c => c.doi && _eapSelected.has(c.doi));
  const payload = {{
    candidates: chosen.map(c => ({{
      doi: c.doi, title: c.title || '', year: c.year || null,
    }})),
    workers: parseInt(document.getElementById('eap-workers').value || '0', 10),
    ingest: document.getElementById('eap-ingest').checked,
  }};
  const btn = document.getElementById('eap-download-btn');
  btn.disabled = true;
  btn.textContent = 'Starting…';
  document.getElementById('eap-status').textContent = '';
  document.getElementById('eap-log').style.display = 'block';
  document.getElementById('eap-log').textContent = '';
  try {{
    const res = await fetch('/api/corpus/expand-author/download-selected', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(payload),
    }});
    if (!res.ok) {{
      const detail = await res.text();
      throw new Error('HTTP ' + res.status + ': ' + detail);
    }}
    const data = await res.json();
    const jobId = data.job_id;
    document.getElementById('eap-status').textContent =
      `Job started (${{data.n_selected}} DOI(s)). Streaming log below…`;
    btn.textContent = 'Running…';
    startGlobalJob(jobId, {{
      type: 'download-selected',
      taskDesc: `download-selected (${{data.n_selected}} papers)`,
    }});
    const es = new EventSource('/api/stream/' + jobId);
    const logEl = document.getElementById('eap-log');
    es.onmessage = function(ev) {{
      let evt;
      try {{ evt = JSON.parse(ev.data); }} catch (_) {{ return; }}
      if (evt.type === 'log') {{
        logEl.textContent += evt.text + '\\n';
        logEl.scrollTop = logEl.scrollHeight;
      }} else if (evt.type === 'progress') {{
        document.getElementById('eap-status').textContent = evt.detail || evt.stage || '';
      }} else if (evt.type === 'completed') {{
        es.close();
        document.getElementById('eap-status').textContent = 'Completed.';
        btn.innerHTML = '&#10003; Done';
        btn.disabled = false;
      }} else if (evt.type === 'error') {{
        es.close();
        document.getElementById('eap-status').textContent =
          'Failed: ' + (evt.message || 'see log.');
        btn.innerHTML = '&#128229; Download selected';
        btn.disabled = false;
      }} else if (evt.type === 'done') {{
        es.close();
      }}
    }};
    es.onerror = function() {{
      es.close();
      btn.disabled = false;
      btn.innerHTML = '&#128229; Download selected';
    }};
  }} catch (exc) {{
    document.getElementById('eap-status').textContent = 'Failed: ' + exc.message;
    btn.innerHTML = '&#128229; Download selected';
    btn.disabled = false;
  }}
}}


// Populate the "Anchor from topic" dropdown on the Expand-citations panel
// with the current corpus's ranked topic clusters.
async function loadCorpusTopicList() {{
  const sel = document.getElementById('tl-exp-relq-topic');
  if (!sel || sel.dataset.loaded) return;
  try {{
    const res = await fetch('/api/catalog/topics');
    const data = await res.json();
    const topics = (data.topics || []).slice(0, 40);
    topics.forEach(t => {{
      const opt = document.createElement('option');
      opt.value = t.name;
      opt.textContent = t.name + '  (' + t.n + ' papers)';
      sel.appendChild(opt);
    }});
    sel.dataset.loaded = '1';
  }} catch (_) {{}}
}}

// ── Search tab ────────────────────────────────────────────────────────
async function doToolSearch(mode) {{
  const q = document.getElementById('tl-search-q').value.trim();
  if (!q) return;
  const status = document.getElementById('tl-search-status');
  const results = document.getElementById('tl-search-results');
  status.textContent = (mode === 'similar' ? 'Finding similar papers...' : 'Searching...');
  results.innerHTML = '';

  const fd = new FormData();
  if (mode === 'similar') {{
    fd.append('identifier', q);
    fd.append('top_k', document.getElementById('tl-search-topk').value || '10');
  }} else {{
    fd.append('q', q);
    fd.append('top_k', document.getElementById('tl-search-topk').value || '10');
    const yf = document.getElementById('tl-search-yfrom').value;
    const yt = document.getElementById('tl-search-yto').value;
    const sec = document.getElementById('tl-search-section').value;
    const tc = document.getElementById('tl-search-topic').value.trim();
    const ex = document.getElementById('tl-search-expand').checked;
    if (yf) fd.append('year_from', yf);
    if (yt) fd.append('year_to', yt);
    if (sec) fd.append('section', sec);
    if (tc) fd.append('topic', tc);
    if (ex) fd.append('expand', 'true');
  }}

  try {{
    const url = (mode === 'similar') ? '/api/search/similar' : '/api/search/query';
    const res = await fetch(url, {{method: 'POST', body: fd}});
    const data = await res.json();
    if (data.error) {{
      status.textContent = data.error;
      return;
    }}
    const hits = data.results || [];
    if (hits.length === 0) {{
      status.textContent = 'No results.';
      return;
    }}
    status.textContent = (mode === 'similar'
      ? 'Similar to: ' + ((data.query || {{}}).title || q)
      : hits.length + ' result' + (hits.length === 1 ? '' : 's'));
    let html = '<ol class="tl-search-list" style="padding-left:20px;">';
    hits.forEach(h => {{
      const authors = (h.authors || []).slice(0, 3).map(a => (a.name || '').split(/\\s+/).slice(-1)[0]).filter(Boolean).join(', ');
      const year = h.year ? ' (' + h.year + ')' : '';
      const sec = h.section_type ? '<span style="color:var(--accent);font-size:11px;">[' + h.section_type + ']</span> ' : '';
      const score = (typeof h.score === 'number') ? ' <span style="color:var(--fg-muted);font-size:11px;">score=' + h.score.toFixed(3) + '</span>' : '';
      html += '<li style="margin-bottom:10px;">';
      html += sec + '<strong>' + (h.title || '(untitled)').replace(/</g, '&lt;') + '</strong>' + year + score;
      if (authors) html += '<div style="color:var(--fg-muted);font-size:11px;">' + authors + '</div>';
      if (h.doi) html += '<div style="font-size:11px;"><a href="https://doi.org/' + h.doi + '" target="_blank" rel="noopener">doi:' + h.doi + '</a></div>';
      if (h.preview) html += '<div style="color:var(--fg-muted);font-size:12px;margin-top:2px;">' + h.preview.replace(/</g, '&lt;') + '</div>';
      html += '</li>';
    }});
    html += '</ol>';
    results.innerHTML = html;
  }} catch (exc) {{
    status.textContent = 'Error: ' + exc;
  }}
}}

// ── Synthesize tab (SSE, mirrors doAsk) ──────────────────────────────
async function doToolSynthesize() {{
  const topic = document.getElementById('tl-synth-topic').value.trim();
  if (!topic) return;
  const status = document.getElementById('tl-synth-status');
  const stream = document.getElementById('tl-synth-stream');
  const sources = document.getElementById('tl-synth-sources');
  status.textContent = 'Retrieving and synthesising...';
  stream.textContent = '';
  sources.style.display = 'none';
  sources.innerHTML = '';

  const stats = createStreamStats('tl-synth-stats', 'qwen3.5:27b');
  stats.start();
  setStreamCursor(stream, true);

  const fd = new FormData();
  fd.append('topic', topic);
  fd.append('context_k', document.getElementById('tl-synth-k').value || '12');
  const yf = document.getElementById('tl-synth-yfrom').value;
  const yt = document.getElementById('tl-synth-yto').value;
  const tf = document.getElementById('tl-synth-topicfilter').value.trim();
  if (yf) fd.append('year_from', yf);
  if (yt) fd.append('year_to', yt);
  if (tf) fd.append('topic_filter', tf);

  const res = await fetch('/api/ask/synthesize', {{method: 'POST', body: fd}});
  const data = await res.json();
  _toolsSynthJob = data.job_id;
  const source = new EventSource('/api/stream/' + data.job_id);
  let collected = null;

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
      collected = evt.sources;
      status.textContent = 'Synthesising from ' + (evt.n || evt.sources.length) + ' passages...';
    }} else if (evt.type === 'completed') {{
      status.textContent = 'Done';
      stats.done('done');
      setStreamCursor(stream, false);
      if (collected && collected.length) {{
        let html = '<div style="font-weight:600;margin-bottom:6px;">Sources (' + collected.length + ')</div>';
        collected.forEach(s => {{ html += '<div class="src-item">' + s + '</div>'; }});
        sources.innerHTML = html;
        sources.style.display = 'block';
      }}
      source.close(); _toolsSynthJob = null;
    }} else if (evt.type === 'error') {{
      status.textContent = 'Error: ' + evt.message;
      stats.done('error');
      setStreamCursor(stream, false);
      source.close(); _toolsSynthJob = null;
    }} else if (evt.type === 'done') {{
      stats.done('done');
      setStreamCursor(stream, false);
      source.close(); _toolsSynthJob = null;
    }}
  }};
}}

// ── Topics tab ───────────────────────────────────────────────────────
async function loadToolTopics() {{
  const list = document.getElementById('tl-topics-list');
  const papers = document.getElementById('tl-topics-papers');
  const dlist = document.getElementById('tl-domains-list');
  list.innerHTML = '<span style="color:var(--fg-muted);font-size:12px;">Loading…</span>';
  papers.innerHTML = '';
  if (dlist) dlist.innerHTML = '<span style="color:var(--fg-muted);font-size:11px;">Loading…</span>';
  try {{
    // Load topics + domains in parallel
    const [topicsRes, domainsRes] = await Promise.all([
      fetch('/api/catalog/topics'),
      fetch('/api/catalog/domains?limit=80'),
    ]);
    const topics = (await topicsRes.json()).topics || [];
    const domains = (await domainsRes.json()).domains || [];

    if (topics.length === 0) {{
      list.innerHTML = '<span style="color:var(--fg-muted);font-size:12px;">No topic clusters assigned yet. Run <code>sciknow catalog cluster</code> to build them.</span>';
    }} else {{
      list.innerHTML = '';
      topics.forEach(t => {{
        const btn = document.createElement('button');
        btn.textContent = t.name + ' (' + t.n + ')';
        btn.title = t.name + ' — ' + t.n + ' papers';
        btn.style.cssText = 'background:var(--bg-alt,#f3f4f6);border:1px solid var(--border);border-radius:12px;padding:4px 10px;font-size:12px;cursor:pointer;';
        btn.onclick = () => loadToolTopicPapers(t.name);
        list.appendChild(btn);
      }});
    }}

    if (dlist) {{
      if (!domains.length) {{
        dlist.innerHTML = '<span style="color:var(--fg-muted);font-size:11px;">No domain tags on this corpus.</span>';
      }} else {{
        dlist.innerHTML = '';
        domains.forEach(d => {{
          const el = document.createElement('span');
          el.textContent = d.name + ' (' + d.n + ')';
          el.title = d.name + ' — ' + d.n + ' papers';
          el.style.cssText = 'background:var(--accent-light,#eef2ff);border:1px solid var(--border);border-radius:10px;padding:2px 8px;font-size:11px;';
          dlist.appendChild(el);
        }});
      }}
    }}
  }} catch (exc) {{
    list.innerHTML = '<span style="color:var(--danger,#c53030);font-size:12px;">Error: ' + exc + '</span>';
  }}
}}

async function loadToolTopicPapers(name) {{
  const papers = document.getElementById('tl-topics-papers');
  papers.innerHTML = '<div style="padding:12px;color:var(--fg-muted);">Loading papers in "' + name + '"…</div>';
  try {{
    const res = await fetch('/api/catalog/topics?name=' + encodeURIComponent(name));
    const data = await res.json();
    const list = data.papers || [];
    if (list.length === 0) {{
      papers.innerHTML = '<div style="padding:12px;color:var(--fg-muted);">No papers in this cluster.</div>';
      return;
    }}
    let html = '<h4 style="margin:4px 0 8px;">' + name.replace(/</g, '&lt;') + ' &mdash; ' + list.length + ' papers</h4>';
    html += '<ol style="padding-left:20px;">';
    list.forEach(p => {{
      const year = p.year ? ' (' + p.year + ')' : '';
      const authors = (p.authors || []).slice(0, 3).map(a => (a.name || '').split(/\\s+/).slice(-1)[0]).filter(Boolean).join(', ');
      html += '<li style="margin-bottom:6px;"><strong>' + (p.title || '(untitled)').replace(/</g, '&lt;') + '</strong>' + year;
      if (authors) html += '<div style="color:var(--fg-muted);font-size:11px;">' + authors + '</div>';
      if (p.doi) html += '<div style="font-size:11px;"><a href="https://doi.org/' + p.doi + '" target="_blank" rel="noopener">doi:' + p.doi + '</a></div>';
      html += '</li>';
    }});
    html += '</ol>';
    papers.innerHTML = html;
  }} catch (exc) {{
    papers.innerHTML = '<div style="color:var(--danger,#c53030);padding:12px;">Error: ' + exc + '</div>';
  }}
}}

// ── Corpus tab (enrich / expand, subprocess SSE) ─────────────────────
async function doToolCorpus(action) {{
  const status = document.getElementById('tl-corpus-status');
  const logEl = document.getElementById('tl-corpus-log');
  const cancelBtn = document.getElementById('tl-corpus-cancel');
  logEl.textContent = '';
  status.textContent = 'Starting ' + action + '...';
  cancelBtn.style.display = 'inline-block';

  const fd = new FormData();
  if (action === 'enrich') {{
    fd.append('limit', document.getElementById('tl-enr-limit').value || '0');
    fd.append('threshold', document.getElementById('tl-enr-thresh').value || '0.85');
    fd.append('dry_run', document.getElementById('tl-enr-dry').checked ? 'true' : 'false');
  }} else if (action === 'expand-author') {{
    // Phase 46.E — expand-by-author
    const nm = (window._selectedExpandAuthorName
                 || document.getElementById('tl-eauth-q').value || '').trim();
    if (!nm) {{
      status.textContent = 'Pick an author from the list (or type a name and press Enter).';
      cancelBtn.style.display = 'none';
      return;
    }}
    fd.append('name', nm);
    const orcid = document.getElementById('tl-eauth-orcid').value.trim();
    if (orcid) fd.append('orcid', orcid);
    fd.append('year_from', document.getElementById('tl-eauth-yfrom').value || '0');
    fd.append('year_to',   document.getElementById('tl-eauth-yto').value   || '0');
    fd.append('limit',     document.getElementById('tl-eauth-limit').value || '0');
    fd.append('workers',   document.getElementById('tl-eauth-workers').value || '0');
    fd.append('relevance_threshold', document.getElementById('tl-eauth-relthr').value || '0.0');
    fd.append('strict_author', document.getElementById('tl-eauth-strict').checked ? 'true' : 'false');
    fd.append('all_matches',   document.getElementById('tl-eauth-all').checked ? 'true' : 'false');
    fd.append('relevance',     document.getElementById('tl-eauth-relevance').checked ? 'true' : 'false');
    fd.append('ingest',        document.getElementById('tl-eauth-ingest').checked ? 'true' : 'false');
    fd.append('dry_run',       document.getElementById('tl-eauth-dry').checked ? 'true' : 'false');
    const rq = document.getElementById('tl-eauth-relq').value.trim();
    if (rq) fd.append('relevance_query', rq);
  }} else {{
    fd.append('limit', document.getElementById('tl-exp-limit').value || '0');
    fd.append('workers', document.getElementById('tl-exp-workers').value || '0');
    fd.append('relevance_threshold', document.getElementById('tl-exp-relthr').value || '0.0');
    fd.append('dry_run', document.getElementById('tl-exp-dry').checked ? 'true' : 'false');
    fd.append('resolve', document.getElementById('tl-exp-resolve').checked ? 'true' : 'false');
    fd.append('ingest', document.getElementById('tl-exp-ingest').checked ? 'true' : 'false');
    fd.append('relevance', document.getElementById('tl-exp-relevance').checked ? 'true' : 'false');
    const rq = document.getElementById('tl-exp-relq').value.trim();
    if (rq) fd.append('relevance_query', rq);
  }}

  try {{
    const res = await fetch('/api/corpus/' + action, {{method: 'POST', body: fd}});
    const data = await res.json();
    if (!res.ok) {{
      status.textContent = 'Failed to start: ' + (data.detail || res.status);
      cancelBtn.style.display = 'none';
      return;
    }}
    _toolsCorpusJob = data.job_id;
  }} catch (exc) {{
    status.textContent = 'Failed to start: ' + exc;
    cancelBtn.style.display = 'none';
    return;
  }}

  const source = new EventSource('/api/stream/' + _toolsCorpusJob);
  source.onmessage = function(e) {{
    const evt = JSON.parse(e.data);
    if (evt.type === 'log') {{
      logEl.textContent += evt.text + '\\n';
      logEl.scrollTop = logEl.scrollHeight;
    }} else if (evt.type === 'progress') {{
      status.textContent = evt.detail || evt.stage;
      if (evt.detail && evt.detail.startsWith('$ ')) {{
        logEl.textContent += evt.detail + '\\n';
      }}
    }} else if (evt.type === 'completed') {{
      status.textContent = action + ' finished.';
      cancelBtn.style.display = 'none';
      source.close(); _toolsCorpusJob = null;
    }} else if (evt.type === 'error') {{
      status.textContent = 'Error: ' + evt.message;
      cancelBtn.style.display = 'none';
      source.close(); _toolsCorpusJob = null;
    }} else if (evt.type === 'done') {{
      cancelBtn.style.display = 'none';
      source.close(); _toolsCorpusJob = null;
    }}
  }};
}}

async function cancelToolCorpus() {{
  if (!_toolsCorpusJob) return;
  try {{
    await fetch('/api/jobs/' + _toolsCorpusJob, {{method: 'DELETE'}});
  }} catch (exc) {{}}
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
      // Phase 42 — data-paper-title carries the raw title; askAboutPaper
      // reads it from the dataset.  Browsers escape data-* values, so no
      // more JSON.stringify + quote juggling.
      html += '<tr data-action="ask-about-paper" data-paper-title="' + escapeHtml(p.title || '') + '">';
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
    html += '<button data-action="load-catalog" data-page="' + (catalogPage - 1) + '" ' + (catalogPage <= 1 ? 'disabled' : '') + '>‹ Prev</button>';
    html += '<span>Page ' + data.page + ' of ' + data.n_pages + '  ·  ' + data.total + ' papers</span>';
    html += '<button data-action="load-catalog" data-page="' + (catalogPage + 1) + '" ' + (catalogPage >= data.n_pages ? 'disabled' : '') + '>Next ›</button>';
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
      // Phase 37 — per-section model override. Empty/null = use
      // caller model / global default.
      model: (s.model && typeof s.model === 'string') ? s.model : '',
    }}));
  }} else if (Array.isArray(ch.sections_template)) {{
    _editingSections = ch.sections_template.map(slug => ({{
      slug: slug, title: titleifyClient(slug), plan: '', model: ''
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
    // Phase 42 — data-action dispatch. delta is a signed int in dataset.
    html += '    <button data-action="move-section" data-section-index="' + i + '" data-delta="-1" title="Move up"' + (i === 0 ? ' disabled style="opacity:0.3;cursor:default;"' : '') + '>&uarr;</button>';
    html += '    <button data-action="move-section" data-section-index="' + i + '" data-delta="1" title="Move down"' + (i === _editingSections.length - 1 ? ' disabled style="opacity:0.3;cursor:default;"' : '') + '>&darr;</button>';
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
    // Phase 37 — per-section model override. Free-text input with a
    // shared datalist of common Ollama tags. Empty = use the caller's
    // model (CLI --model / API form) or fall through to settings.
    // llm_model. Pairs with the Phase 35 compute counter: dial
    // expensive models up only on sections that need them.
    const modelVal = (s.model || '').trim();
    const modelBadgeClass = modelVal ? 'sec-target-badge override' : 'sec-target-badge';
    const modelBadgeTitle = modelVal
      ? 'Per-section model override (this section only)'
      : 'Uses the caller-provided model or settings.llm_model default';
    html += '    <div class="sec-size-row">';
    html += '      <label>Model:</label>';
    html += '      <input type="text" list="sec-model-suggestions" class="sec-size-custom" ';
    html += '             style="width:160px;" placeholder="(default)" ';
    html += '             value="' + escapeHtml(modelVal) + '" ';
    html += '             oninput="updateSectionModel(' + i + ', this.value)">';
    html += '      <span class="' + modelBadgeClass + '" title="' + modelBadgeTitle + '">';
    html += (modelVal ? escapeHtml(modelVal) + ' <span class="badge-tag">override</span>'
                      : '— <span class="badge-tag muted">default</span>');
    html += '</span>';
    html += '    </div>';
    html += '    <div class="sec-slug">slug: <code>' + slugDisplay + '</code></div>';
    html += '  </div>';
    html += '  <button class="sec-delete" data-action="remove-section" data-section-index="' + i + '" title="Delete this section">&times;</button>';
    html += '</div>';
  }});
  // Phase 37 — one shared datalist for the per-section model inputs.
  // The list is hints, not a whitelist — Ollama accepts any tag.
  html += '<datalist id="sec-model-suggestions">';
  ['qwen3:32b', 'qwen3:14b', 'qwen3:8b', 'qwen2.5:32b', 'qwen2.5:14b',
   'qwen2.5:7b', 'llama3.1:70b', 'llama3.1:8b', 'mistral-nemo:12b',
   'gemma2:27b', 'gemma2:9b', 'phi3.5:3.8b']
    .forEach(m => {{ html += '<option value="' + m + '">'; }});
  html += '</datalist>';
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

// ── Phase 33: chapter drag-and-drop reordering ──────────────────────────
//
// The POST /api/chapters/reorder endpoint has existed since Phase 14
// but was never wired to a GUI affordance. Phase 33 adds the drag
// handlers, mirroring Phase 26's section drag-drop. The user drags
// a chapter title bar above or below another chapter title bar; on
// drop, the full chapter_ids order is POSTed and the sidebar rebuilt.
//
// Unlike section drag-drop, there's no within-chapter constraint —
// chapters can be reordered freely across the whole book.

let _chDragId = null;
function chDragStart(e, chId) {{
  _chDragId = chId;
  e.dataTransfer.effectAllowed = 'move';
  e.dataTransfer.setData('text/plain', chId);
  const group = e.target.closest('.ch-group');
  if (group) setTimeout(() => group.classList.add('dragging'), 0);
}}
function chDragOver(e) {{
  if (!_chDragId) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'move';
  const title = e.target.closest('.ch-title');
  // Visual indicator: top/bottom border based on cursor position.
  document.querySelectorAll('.ch-title').forEach(t => {{
    t.classList.remove('ch-drag-over-top', 'ch-drag-over-bottom');
  }});
  if (title) {{
    const rect = title.getBoundingClientRect();
    const mid = rect.top + rect.height / 2;
    if (e.clientY < mid) {{
      title.classList.add('ch-drag-over-top');
    }} else {{
      title.classList.add('ch-drag-over-bottom');
    }}
  }}
}}
function chDragEnd(e) {{
  _chDragId = null;
  document.querySelectorAll('.ch-group').forEach(g => g.classList.remove('dragging'));
  document.querySelectorAll('.ch-title').forEach(t => {{
    t.classList.remove('ch-drag-over-top', 'ch-drag-over-bottom');
  }});
}}
async function chDrop(e, targetChId) {{
  e.preventDefault();
  if (!_chDragId || _chDragId === targetChId) {{
    chDragEnd(e);
    return;
  }}
  // Compute the new order by removing the dragged chapter from
  // its current position and inserting it before or after the target.
  const ids = chaptersData.map(c => c.id);
  const fromIdx = ids.indexOf(_chDragId);
  if (fromIdx < 0) {{ chDragEnd(e); return; }}
  ids.splice(fromIdx, 1);
  const toIdx = ids.indexOf(targetChId);
  if (toIdx < 0) {{ chDragEnd(e); return; }}
  // Insert above or below based on cursor position in the title bar.
  const title = e.target.closest('.ch-title');
  let insertIdx = toIdx;
  if (title) {{
    const rect = title.getBoundingClientRect();
    insertIdx = e.clientY > rect.top + rect.height / 2 ? toIdx + 1 : toIdx;
  }}
  ids.splice(insertIdx, 0, _chDragId);
  chDragEnd(e);
  // POST the new order
  try {{
    const res = await fetch('/api/chapters/reorder', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{chapter_ids: ids}}),
    }});
    if (!res.ok) throw new Error('reorder failed (' + res.status + ')');
    // Refresh the sidebar with the new order.
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    if (Array.isArray(sd.chapters)) chaptersData = sd.chapters;
    rebuildSidebar(sd.chapters || sd, currentDraftId);
  }} catch (err) {{
    alert('Chapter reorder failed: ' + err.message);
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
  const group = link.closest('.ch-group');
  if (!group) return;
  // Don't show a drop indicator on the dragged row itself.
  if (link.dataset.sectionSlug === _draggedSection.slug
      && group.dataset.chId === _draggedSection.chapterId) return;
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
  if (!group) {{ _cleanupDrag(); return; }}
  e.preventDefault();

  const targetChId = group.dataset.chId;
  const targetSlug = link.dataset.sectionSlug;
  const sourceChId = _draggedSection.chapterId;
  const sourceSlug = _draggedSection.slug;

  if (targetSlug === sourceSlug && targetChId === sourceChId) {{
    _cleanupDrag();
    return;
  }}

  const rect = link.getBoundingClientRect();
  const position = e.clientY < (rect.top + rect.height / 2) ? 'before' : 'after';

  if (targetChId === sourceChId) {{
    // Within-chapter reorder (Phase 26, unchanged)
    reorderSections(sourceChId, sourceSlug, targetSlug, position);
  }} else {{
    // Phase 33 — cross-chapter move. Requires a confirm because it
    // updates drafts.chapter_id, which changes where the draft lives
    // in the book's chapter structure.
    const srcCh = chaptersData.find(c => c.id === sourceChId);
    const tgtCh = chaptersData.find(c => c.id === targetChId);
    const srcName = srcCh ? 'Ch.' + srcCh.num : 'source chapter';
    const tgtName = tgtCh ? 'Ch.' + tgtCh.num : 'target chapter';
    if (confirm('Move section "' + sourceSlug + '" from ' + srcName + ' to ' + tgtName + '?')) {{
      moveSectionCrossChapter(sourceChId, sourceSlug, targetChId, targetSlug, position);
    }}
  }}
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

// Phase 33 — cross-chapter section move. Four API calls:
// 1. Find the draft for this slug in the source chapter
// 2. PUT /api/draft/{{id}}/chapter — move the draft to the target chapter
// 3. Remove the slug from source chapter's sections
// 4. Add the slug to target chapter's sections at the right position
// Then refresh the sidebar.
async function moveSectionCrossChapter(srcChId, slug, tgtChId, targetSlug, position) {{
  const srcCh = chaptersData.find(c => c.id === srcChId);
  const tgtCh = chaptersData.find(c => c.id === tgtChId);
  if (!srcCh || !tgtCh) return;

  // 1) Find the draft id for this section in the source chapter.
  // Draft could be in the sections list returned from /api/chapters.
  const draft = (srcCh.sections || []).find(s =>
    (s.type || '').toLowerCase() === slug.toLowerCase() && s.id
  );
  const draftId = draft ? draft.id : null;

  try {{
    // 2) Move the draft if it exists
    if (draftId) {{
      const fd = new FormData();
      fd.append('chapter_id', tgtChId);
      const r = await fetch('/api/draft/' + draftId + '/chapter', {{method: 'PUT', body: fd}});
      if (!r.ok) throw new Error('draft move failed (' + r.status + ')');
    }}

    // 3) Remove slug from source chapter's sections
    const srcMeta = (srcCh.sections_meta || []).filter(s => s.slug !== slug);
    const srcSections = srcMeta.map(s => ({{
      slug: s.slug, title: s.title || s.slug, plan: s.plan || '',
      target_words: (s.target_words && s.target_words > 0) ? s.target_words : null,
    }}));
    await fetch('/api/chapters/' + srcChId + '/sections', {{
      method: 'PUT',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{sections: srcSections}}),
    }});

    // 4) Add slug to target chapter's sections at the right position
    const movedEntry = (srcCh.sections_meta || []).find(s => s.slug === slug) || {{slug: slug, title: slug, plan: ''}};
    const tgtMeta = (tgtCh.sections_meta || []).map(s => ({{
      slug: s.slug, title: s.title || s.slug, plan: s.plan || '',
      target_words: (s.target_words && s.target_words > 0) ? s.target_words : null,
    }}));
    const insertIdx = tgtMeta.findIndex(s => s.slug === targetSlug);
    const newEntry = {{
      slug: movedEntry.slug, title: movedEntry.title || movedEntry.slug,
      plan: movedEntry.plan || '',
      target_words: (movedEntry.target_words && movedEntry.target_words > 0)
        ? movedEntry.target_words : null,
    }};
    if (insertIdx >= 0) {{
      tgtMeta.splice(position === 'before' ? insertIdx : insertIdx + 1, 0, newEntry);
    }} else {{
      tgtMeta.push(newEntry);
    }}
    await fetch('/api/chapters/' + tgtChId + '/sections', {{
      method: 'PUT',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{sections: tgtMeta}}),
    }});

    // 5) Refresh sidebar
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    if (Array.isArray(sd.chapters)) chaptersData = sd.chapters;
    rebuildSidebar(sd.chapters || sd, currentDraftId);
  }} catch (e) {{
    alert('Cross-chapter move failed: ' + e.message);
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

// Phase 37 — per-section model override input handler.  Blank string
// clears the override so the section falls back to the caller-provided
// model / global default. Trim, store; do NOT re-render (user is
// typing; re-render would eat focus).
function updateSectionModel(idx, value) {{
  if (idx < 0 || idx >= _editingSections.length) return;
  _editingSections[idx].model = (value || '').trim();
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
        // Phase 37 — per-section model override. Empty string is
        // persisted as null by _normalize_chapter_sections so the
        // section falls through to the caller/global default.
        model: (s.model || '').trim() || null,
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

// ── Phase 39: consolidated Book Settings modal ───────────────────────
// Brings title/description/plan/target_chapter_words/style_fingerprint
// into one editor. All fields round-trip through existing endpoints:
// GET /api/book for reads, PUT /api/book for writes, and a new
// POST /api/book/style-fingerprint/refresh for the style tab.
async function openBookSettings() {{
  openModal('book-settings-modal');
  switchBookSettingsTab('bs-basics');
  await loadBookSettings();
}}

function switchBookSettingsTab(name) {{
  document.querySelectorAll('#book-settings-modal .tab').forEach(t => {{
    t.classList.toggle('active', t.dataset.tab === name);
  }});
  ['bs-basics', 'bs-leitmotiv', 'bs-style'].forEach(n => {{
    const pane = document.getElementById(n + '-pane');
    if (pane) pane.style.display = (n === name) ? 'block' : 'none';
  }});
}}

async function loadBookSettings() {{
  try {{
    const res = await fetch('/api/book');
    const data = await res.json();
    document.getElementById('bs-title').value = data.title || '';
    document.getElementById('bs-description').value = data.description || '';
    document.getElementById('bs-target-chapter-words').value = (data.target_chapter_words != null) ? String(data.target_chapter_words) : '';
    document.getElementById('bs-plan').value = data.plan || '';
    // Basics meta — chapter / draft / gaps counts come through the
    // same endpoint, so surface them as a read-only summary.
    const meta = document.getElementById('bs-basics-meta');
    const defaultTcw = data.default_target_chapter_words || 6000;
    const effectiveTcw = data.target_chapter_words || defaultTcw;
    meta.innerHTML = '<strong>' + (data.chapters || 0) + '</strong> chapter' + ((data.chapters || 0) === 1 ? '' : 's')
                   + ' · <strong>' + (data.drafts || 0) + '</strong> draft' + ((data.drafts || 0) === 1 ? '' : 's')
                   + ' · Status: <strong>' + (data.status || 'draft') + '</strong>'
                   + ' · Effective target: <strong>~' + effectiveTcw + '</strong> words/chapter'
                   + (data.target_chapter_words ? '' : ' <em>(default)</em>');
    renderStyleFingerprint(data.style_fingerprint);
  }} catch (exc) {{
    document.getElementById('bs-basics-status').textContent = 'Load failed: ' + exc;
  }}
}}

function renderStyleFingerprint(fp) {{
  const el = document.getElementById('bs-style-fingerprint');
  if (!fp || !fp.n_drafts_sampled) {{
    el.innerHTML = '<div style="color:var(--fg-muted);font-size:13px;">No fingerprint yet &mdash; mark some drafts as <em>final</em> / <em>reviewed</em> / <em>revised</em> and click <strong>Recompute</strong>.</div>';
    return;
  }}
  const rows = [
    ['Drafts sampled', fp.n_drafts_sampled],
    ['Average words per draft', fp.avg_words_per_draft],
    ['Median sentence length', (fp.median_sentence_length || 0) + ' words'],
    ['Median paragraph length', (fp.median_paragraph_words || 0) + ' words'],
    ['Citations per 100 words', (fp.citations_per_100_words != null) ? fp.citations_per_100_words.toFixed(2) : '—'],
    ['Hedging rate', (fp.hedging_rate != null) ? (fp.hedging_rate * 100).toFixed(1) + '%' : '—'],
  ];
  let html = '<div style="display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;font-size:13px;">';
  rows.forEach(([k, v]) => {{
    html += '<div style="color:var(--fg-muted);">' + k + '</div>';
    html += '<div><strong>' + (v != null ? v : '—') + '</strong></div>';
  }});
  html += '</div>';
  const trans = fp.top_transitions || [];
  if (trans.length) {{
    html += '<div style="margin-top:10px;font-size:12px;color:var(--fg-muted);">Top sentence-initial transitions: ';
    html += trans.slice(0, 8).map(t => '<span style="display:inline-block;padding:2px 8px;margin:2px;border:1px solid var(--border);border-radius:10px;background:var(--bg);font-family:ui-monospace,monospace;font-size:11px;">' + (t[0] || t).replace(/</g, '&lt;') + '</span>').join('');
    html += '</div>';
  }}
  if (fp.computed_at) {{
    html += '<div style="margin-top:10px;font-size:11px;color:var(--fg-muted);">Computed at: ' + fp.computed_at + '</div>';
  }}
  el.innerHTML = html;
}}

async function saveBookSettings(tab) {{
  const statusId = 'bs-' + tab + '-status';
  const status = document.getElementById(statusId);
  status.textContent = 'Saving…';
  const fd = new FormData();
  if (tab === 'basics') {{
    fd.append('title', document.getElementById('bs-title').value);
    fd.append('description', document.getElementById('bs-description').value);
    const tcw = document.getElementById('bs-target-chapter-words').value;
    // Blank leaves unchanged; 0 clears back to default; positive sets.
    if (tcw !== '') fd.append('target_chapter_words', tcw);
  }} else if (tab === 'leitmotiv') {{
    fd.append('plan', document.getElementById('bs-plan').value);
  }}
  try {{
    const res = await fetch('/api/book', {{method: 'PUT', body: fd}});
    const data = await res.json();
    if (!res.ok || !data.ok) {{
      status.textContent = 'Save failed: ' + (data.detail || 'unknown');
      return;
    }}
    status.innerHTML = '<span style="color:var(--success);">Saved.</span>';
    // Rehydrate meta line — chapter counts may shift if the target changed
    if (tab === 'basics') await loadBookSettings();
  }} catch (exc) {{
    status.textContent = 'Save failed: ' + exc;
  }}
}}

async function refreshStyleFingerprint() {{
  const status = document.getElementById('bs-style-status');
  status.textContent = 'Computing from approved drafts…';
  try {{
    const res = await fetch('/api/book/style-fingerprint/refresh', {{method: 'POST'}});
    const data = await res.json();
    if (!res.ok || !data.ok) {{
      status.textContent = 'Refresh failed: ' + (data.error || data.detail || 'unknown');
      return;
    }}
    renderStyleFingerprint(data.fingerprint);
    const sampled = (data.fingerprint && data.fingerprint.n_drafts_sampled) || 0;
    if (sampled === 0) {{
      status.textContent = 'No approved drafts yet — mark some as final/reviewed/revised first.';
    }} else {{
      status.innerHTML = '<span style="color:var(--success);">Updated from ' + sampled + ' draft' + (sampled === 1 ? '' : 's') + '.</span>';
    }}
  }} catch (exc) {{
    status.textContent = 'Refresh failed: ' + exc;
  }}
}}

// ── Phase 38: scoped snapshot bundles (chapter + book) ───────────────
// Safety net for autowrite-all. Chapter bundle stores every section's
// current content; restore creates NEW draft versions per section
// (non-destructive) so the old versions stay around as an undo path.
function openBundleSnapshots() {{
  openModal('bundle-modal');
  switchBundleTab('sb-chapter');
}}

function switchBundleTab(name) {{
  document.querySelectorAll('#bundle-modal .tab').forEach(t => {{
    t.classList.toggle('active', t.dataset.tab === name);
  }});
  document.getElementById('sb-chapter-pane').style.display = (name === 'sb-chapter') ? 'block' : 'none';
  document.getElementById('sb-book-pane').style.display = (name === 'sb-book') ? 'block' : 'none';
  if (name === 'sb-chapter') loadBundleList('chapter');
  else loadBundleList('book');
}}

async function doBundleSnapshot(scope) {{
  const nameEl = document.getElementById('sb-' + scope + '-name');
  const status = document.getElementById('sb-' + scope + '-status');
  let url;
  if (scope === 'chapter') {{
    if (!currentChapterId) {{
      status.textContent = 'No current chapter — select a section first.';
      return;
    }}
    url = '/api/snapshot/chapter/' + currentChapterId;
  }} else {{
    url = '/api/snapshot/book/{book_id}';
  }}
  status.textContent = 'Saving…';
  const fd = new FormData();
  fd.append('name', nameEl.value || '');
  try {{
    const res = await fetch(url, {{method: 'POST', body: fd}});
    const data = await res.json();
    if (!res.ok || !data.ok) {{
      status.textContent = 'Error: ' + (data.detail || data.error || 'failed');
      return;
    }}
    const extra = scope === 'chapter'
      ? ` (${{data.drafts_included}} section${{data.drafts_included === 1 ? '' : 's'}}, ${{data.total_words}} words)`
      : ` (${{data.chapters_included}} chapter${{data.chapters_included === 1 ? '' : 's'}}, ${{data.total_words}} words)`;
    status.innerHTML = '<span style="color:var(--success);">Saved &quot;' + (data.name || '').replace(/</g, '&lt;') + '&quot;</span>' + extra;
    nameEl.value = '';
    loadBundleList(scope);
  }} catch (exc) {{
    status.textContent = 'Error: ' + exc;
  }}
}}

async function loadBundleList(scope) {{
  const list = document.getElementById('sb-' + scope + '-list');
  const target = (scope === 'chapter') ? currentChapterId : '{book_id}';
  if (!target) {{
    list.innerHTML = '<div style="color:var(--fg-muted);font-size:12px;">Open any section first so a chapter is active.</div>';
    return;
  }}
  list.innerHTML = '<div style="color:var(--fg-muted);font-size:12px;">Loading…</div>';
  try {{
    const res = await fetch('/api/snapshots/' + scope + '/' + target);
    const data = await res.json();
    const snaps = data.snapshots || [];
    if (snaps.length === 0) {{
      list.innerHTML = '<div style="color:var(--fg-muted);font-size:12px;">No ' + scope + ' snapshots yet.</div>';
      return;
    }}
    let html = '<table style="width:100%;border-collapse:collapse;font-size:13px;">';
    html += '<thead><tr style="color:var(--fg-muted);text-align:left;border-bottom:1px solid var(--border);">';
    html += '<th style="padding:6px 4px;">Label</th><th style="padding:6px 4px;">Words</th><th style="padding:6px 4px;">Saved</th><th></th></tr></thead><tbody>';
    snaps.forEach(s => {{
      const created = (s.created_at || '').split('.')[0].replace('T', ' ');
      html += '<tr style="border-bottom:1px solid var(--border);">';
      html += '<td style="padding:6px 4px;">' + (s.name || '').replace(/</g, '&lt;') + '</td>';
      html += '<td style="padding:6px 4px;color:var(--fg-muted);">' + (s.word_count || 0).toLocaleString() + '</td>';
      html += '<td style="padding:6px 4px;color:var(--fg-muted);font-size:11px;">' + created + '</td>';
      html += '<td style="padding:6px 4px;text-align:right;">';
      html += '<button data-action="restore-bundle" data-snapshot-id="' + s.id + '" data-scope="' + scope + '" style="font-size:12px;padding:3px 10px;">Restore</button>';
      html += '</td></tr>';
    }});
    html += '</tbody></table>';
    list.innerHTML = html;
  }} catch (exc) {{
    list.innerHTML = '<div style="color:var(--danger,#c53030);font-size:12px;">Error: ' + exc + '</div>';
  }}
}}

async function restoreBundle(snapId, scope) {{
  const confirmMsg = scope === 'chapter'
    ? 'Restore this chapter snapshot? Each section will get a NEW draft version. Existing drafts stay untouched.'
    : 'Restore this BOOK snapshot? Every chapter will get new draft versions for every section. Existing drafts stay untouched.';
  if (!confirm(confirmMsg)) return;
  const status = document.getElementById('sb-' + scope + '-status');
  status.textContent = 'Restoring…';
  try {{
    const res = await fetch('/api/snapshot/restore-bundle/' + snapId, {{method: 'POST'}});
    const data = await res.json();
    if (!res.ok || !data.ok) {{
      status.textContent = 'Error: ' + (data.detail || data.error || 'failed');
      return;
    }}
    status.innerHTML = '<span style="color:var(--success);">Restored ' + data.drafts_created + ' draft' + (data.drafts_created === 1 ? '' : 's') + ' across ' + data.chapters_restored + ' chapter' + (data.chapters_restored === 1 ? '' : 's') + '. Reload to see them.</span>';
  }} catch (exc) {{
    status.textContent = 'Error: ' + exc;
  }}
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
    html += '<button data-action="diff-snapshot" data-snapshot-id="' + s.id + '">Diff</button> ';
    html += '<button data-action="restore-snapshot" data-snapshot-id="' + s.id + '">Restore</button>';
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


// ── Phase 46.F — Setup Wizard (end-to-end from empty to book) ─────────
//
// Walks a new user through: project choice → corpus ingest → index
// builds → expand → book creation. Each step reads live state via
// /api/setup/status so the trail shows real progress, not just UI
// position. Subprocess-backed steps (ingest, cluster, raptor, wiki)
// stream to a per-step <pre> log.

let _swCurrentStep = 'project';
let _swCurrentJob  = null;

function openSetupWizard() {{
  openModal('setup-wizard-modal');
  swGoto('project');
  swRefreshStatus();
  swLoadProjectsForWizard();
}}

function swGoto(step) {{
  _swCurrentStep = step;
  document.querySelectorAll('#setup-wizard-modal .sw-step-pane').forEach(p => {{
    p.style.display = 'none';
  }});
  const pane = document.getElementById('sw-step-' + step);
  if (pane) pane.style.display = 'block';
  document.querySelectorAll('#sw-trail .sw-step').forEach(s => {{
    s.classList.toggle('active', s.dataset.swStep === step);
  }});
  // Auto-refresh status on entering steps 2 + 3 so counts are fresh
  if (step === 'corpus' || step === 'indices') swRefreshStatus();
}}

async function swRefreshStatus() {{
  try {{
    const res = await fetch('/api/setup/status');
    const d = await res.json();
    const proj = d.project || {{slug: 'unknown'}};
    // Corpus step
    const cstat = document.getElementById('sw-corpus-status');
    if (cstat) {{
      cstat.innerHTML =
        '<strong>Active project:</strong> <code>' + proj.slug + '</code>'
        + (proj.is_default ? ' <em>(legacy default)</em>' : '')
        + '<br><strong>Documents:</strong> ' + (d.n_documents || 0).toLocaleString()
        + ' &middot; <strong>Complete:</strong> ' + (d.n_complete || 0).toLocaleString()
        + ' &middot; <strong>Chunks:</strong> ' + (d.n_chunks || 0).toLocaleString();
    }}
    // Indices step
    const istat = document.getElementById('sw-indices-status');
    if (istat) {{
      const rapt = d.raptor_levels || {{}};
      const raptStr = Object.keys(rapt).length
        ? Object.keys(rapt).sort().map(k => k + '=' + rapt[k]).join(', ')
        : '(not built)';
      istat.innerHTML =
        '<strong>Topic clusters:</strong> ' + (d.n_with_topic || 0)
        + ' papers tagged &middot; <strong>RAPTOR:</strong> ' + raptStr
        + ' &middot; <strong>Wiki pages:</strong> ' + (d.n_wiki_pages || 0);
    }}
  }} catch (_) {{}}
}}

async function swLoadProjectsForWizard() {{
  const list = document.getElementById('sw-project-list');
  list.innerHTML = 'Loading…';
  try {{
    const res = await fetch('/api/projects');
    const d = await res.json();
    const active = d.active_slug;
    const running = d.running_slug;
    if (!d.projects || !d.projects.length) {{
      list.innerHTML = '<div style="padding:10px;color:var(--fg-muted);">No projects yet. Create one on the right.</div>';
      return;
    }}
    list.innerHTML = d.projects.map(p => {{
      const mark = p.slug === active ? '●' : '○';
      const running_mark = p.slug === running
        ? ' <span style="color:var(--accent);font-size:10px;">(running here)</span>'
        : '';
      const useBtn = p.slug === active ? '' :
        `<button onclick="swUseProject('${{p.slug}}')">Use</button>`;
      return `<div style="padding:6px 10px;border-bottom:1px solid var(--border);
                           display:flex;align-items:center;gap:8px;">
        <span style="color:var(--accent);">${{mark}}</span>
        <strong style="flex:1;">${{p.slug}}</strong>${{running_mark}}
        ${{useBtn}}</div>`;
    }}).join('');
  }} catch (exc) {{
    list.innerHTML = '<div style="padding:10px;color:var(--danger);">Failed: ' + exc + '</div>';
  }}
}}

async function swUseProject(slug) {{
  try {{
    const res = await fetch('/api/projects/use', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{slug: slug}}),
    }});
    const d = await res.json();
    document.getElementById('sw-project-status').textContent =
      d.message || ('Active project: ' + slug);
    swLoadProjectsForWizard();
    swRefreshStatus();
  }} catch (exc) {{
    document.getElementById('sw-project-status').textContent = 'Failed: ' + exc;
  }}
}}

async function swCreateProject() {{
  const slug = (document.getElementById('sw-new-slug').value || '').trim();
  if (!slug) return;
  if (!/^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/.test(slug)) {{
    document.getElementById('sw-project-status').textContent =
      'Slug must be lowercase alphanumerics + hyphens.';
    return;
  }}
  document.getElementById('sw-project-status').textContent =
    'Creating ' + slug + '…';
  try {{
    const res = await fetch('/api/projects/init', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{slug: slug}}),
    }});
    const d = await res.json();
    if (!res.ok) {{
      document.getElementById('sw-project-status').textContent =
        'Failed: ' + (d.detail || res.status);
      return;
    }}
    document.getElementById('sw-project-status').textContent =
      '✓ Created ' + slug + '. Use it below to activate.';
    swLoadProjectsForWizard();
  }} catch (exc) {{
    document.getElementById('sw-project-status').textContent = 'Failed: ' + exc;
  }}
}}

// ── Corpus step ─────────────────────────────────────────────────────
async function swUploadPDFs() {{
  const input = document.getElementById('sw-upload-files');
  if (!input.files || !input.files.length) return;
  const fd = new FormData();
  for (const f of input.files) fd.append('files', f);
  fd.append('start_ingest',
    document.getElementById('sw-upload-start-ingest').checked ? 'true' : 'false');
  const log = document.getElementById('sw-ingest-log');
  log.textContent = 'Uploading ' + input.files.length + ' file(s)…\\n';
  try {{
    const res = await fetch('/api/corpus/upload', {{method: 'POST', body: fd}});
    const d = await res.json();
    if (!res.ok) {{
      log.textContent += 'Upload failed: ' + (d.detail || res.status) + '\\n';
      return;
    }}
    log.textContent += '✓ Staged ' + d.n_files + ' file(s) to ' + d.staging_dir + '\\n';
    if (d.job_id) swAttachLogStream(d.job_id, 'sw-ingest-log');
  }} catch (exc) {{
    log.textContent += 'Upload failed: ' + exc + '\\n';
  }}
}}

async function swIngestDirectory() {{
  const path = (document.getElementById('sw-ingest-path').value || '').trim();
  if (!path) return;
  const fd = new FormData();
  fd.append('path', path);
  fd.append('recursive',
    document.getElementById('sw-ingest-recursive').checked ? 'true' : 'false');
  fd.append('force',
    document.getElementById('sw-ingest-force').checked ? 'true' : 'false');
  const log = document.getElementById('sw-ingest-log');
  log.textContent = 'Starting ingest of ' + path + '…\\n';
  try {{
    const res = await fetch('/api/corpus/ingest-directory',
      {{method: 'POST', body: fd}});
    const d = await res.json();
    if (!res.ok) {{
      log.textContent += 'Failed: ' + (d.detail || res.status) + '\\n';
      return;
    }}
    swAttachLogStream(d.job_id, 'sw-ingest-log');
  }} catch (exc) {{
    log.textContent += 'Failed: ' + exc + '\\n';
  }}
}}

// ── Indices step ────────────────────────────────────────────────────
async function swRunIndex(kind) {{
  const fd = new FormData();
  let url = '';
  if (kind === 'cluster') {{
    url = '/api/catalog/cluster';
    fd.append('rebuild',
      document.getElementById('sw-cluster-rebuild').checked ? 'true' : 'false');
  }} else if (kind === 'raptor') {{
    url = '/api/catalog/raptor/build';
  }} else if (kind === 'wiki') {{
    url = '/api/wiki/compile';
    fd.append('rebuild',
      document.getElementById('sw-wiki-rebuild').checked ? 'true' : 'false');
    fd.append('rewrite_stale',
      document.getElementById('sw-wiki-stale').checked ? 'true' : 'false');
  }}
  const log = document.getElementById('sw-indices-log');
  log.textContent = 'Starting ' + kind + '…\\n';
  try {{
    const res = await fetch(url, {{method: 'POST', body: fd}});
    const d = await res.json();
    if (!res.ok) {{
      log.textContent += 'Failed: ' + (d.detail || res.status) + '\\n';
      return;
    }}
    swAttachLogStream(d.job_id, 'sw-indices-log');
  }} catch (exc) {{
    log.textContent += 'Failed: ' + exc + '\\n';
  }}
}}

// ── Book step ───────────────────────────────────────────────────────
async function swCreateBook() {{
  const title = (document.getElementById('sw-book-title').value || '').trim();
  if (!title) {{
    document.getElementById('sw-book-status').textContent = 'Title is required.';
    return;
  }}
  const type = document.getElementById('sw-book-type').value;
  const desc = document.getElementById('sw-book-desc').value.trim();
  const target = document.getElementById('sw-book-target').value;
  const payload = {{
    title: title, type: type, description: desc, bootstrap: true,
  }};
  if (target) payload.target_chapter_words = parseInt(target, 10);
  const stat = document.getElementById('sw-book-status');
  stat.textContent = 'Creating…';
  try {{
    const res = await fetch('/api/book/create', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(payload),
    }});
    const d = await res.json();
    if (!res.ok) {{
      stat.innerHTML = '<span style="color:var(--danger);">Failed: '
        + (d.detail || res.status) + '</span>';
      return;
    }}
    const flatNote = d.is_flat
      ? ' &middot; Auto-created chapter 1 with sections: '
        + (d.default_sections || []).join(', ')
      : '';
    stat.innerHTML = '<span style="color:var(--success);">✓ Created '
      + d.display_name + ' "' + d.title + '"</span>'
      + '<br><code>' + d.book_id.slice(0, 8) + '</code>' + flatNote
      + '<br>Next: restart <code>sciknow book serve "' + d.title
      + '"</code> to open this book in the reader.';
  }} catch (exc) {{
    stat.innerHTML = '<span style="color:var(--danger);">Failed: ' + exc + '</span>';
  }}
}}

// Shared SSE log attacher — renders `log` events line-by-line, and
// updates status counts when the job ends.
function swAttachLogStream(jobId, logElId) {{
  _swCurrentJob = jobId;
  const logEl = document.getElementById(logElId);
  const source = new EventSource('/api/stream/' + jobId);
  source.onmessage = function(e) {{
    const evt = JSON.parse(e.data);
    if (evt.type === 'log') {{
      logEl.textContent += evt.text + '\\n';
      logEl.scrollTop = logEl.scrollHeight;
    }} else if (evt.type === 'progress') {{
      if (evt.detail && evt.detail.startsWith('$ ')) {{
        logEl.textContent += evt.detail + '\\n';
      }}
    }} else if (evt.type === 'error') {{
      logEl.textContent += 'ERROR: ' + evt.message + '\\n';
      source.close(); _swCurrentJob = null;
      swRefreshStatus();
    }} else if (evt.type === 'completed' || evt.type === 'done') {{
      logEl.textContent += '— done —\\n';
      source.close(); _swCurrentJob = null;
      swRefreshStatus();
    }}
  }};
}}


// ── Phase 43h — Project management modal ──────────────────────────────
// Mirrors `sciknow project` from the CLI. Switching the active project
// only writes .active-project; the running web reader keeps serving its
// original book until the user restarts `sciknow book serve`.

function openProjectsModal() {{
  openModal('projects-modal');
  refreshProjectsList();
}}

function _projMsg(text, kind) {{
  const el = document.getElementById('proj-msg');
  if (!el) return;
  el.textContent = text || '';
  el.style.color = kind === 'error' ? 'var(--danger)'
                 : kind === 'ok'    ? 'var(--success)'
                 : 'var(--fg-muted)';
}}

async function refreshProjectsList() {{
  _projMsg('Loading…');
  const wrap = document.getElementById('projects-list');
  try {{
    const resp = await fetch('/api/projects');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    document.getElementById('proj-running').textContent = data.running_slug || '(unknown)';
    if (!data.projects || data.projects.length === 0) {{
      wrap.innerHTML = '<em style="color:var(--fg-muted);">No projects yet. Create one below.</em>';
      _projMsg('');
      return;
    }}
    const rows = data.projects.map(p => {{
      const activeMark = p.active ? '<span style="color:var(--accent);font-weight:600;">●</span>' : '<span style="color:var(--fg-faint);">○</span>';
      const statusBadge = p.status === 'ok'
        ? '<span style="color:var(--success);">ok</span>'
        : '<span style="color:var(--warning);">incomplete</span>';
      const isRunning = p.slug === data.running_slug;
      const useBtn    = p.active ? ''
        : `<button onclick="useProject('${{p.slug}}')" title="Set .active-project to ${{p.slug}}">Use</button>`;
      const destroyBtn = (p.is_default || isRunning) ? ''
        : `<button onclick="destroyProject('${{p.slug}}')" style="color:var(--danger);" title="Drop DB + collections + data dir">Destroy</button>`;
      const showBtn = `<button onclick="showProjectDetail('${{p.slug}}')">Details</button>`;
      return `<tr>
        <td style="text-align:center;width:30px;">${{activeMark}}</td>
        <td style="font-weight:600;">${{p.slug}}${{isRunning ? ' <span style="font-size:10px;color:var(--accent);">(running)</span>' : ''}}</td>
        <td style="color:var(--fg-muted);font-family:var(--font-mono);font-size:11px;">${{p.pg_database}}</td>
        <td style="color:var(--fg-muted);font-family:var(--font-mono);font-size:11px;">${{p.papers_collection}}</td>
        <td>${{statusBadge}}</td>
        <td style="white-space:nowrap;">${{showBtn}} ${{useBtn}} ${{destroyBtn}}</td>
      </tr>`;
    }}).join('');
    wrap.innerHTML = `<table style="width:100%;border-collapse:collapse;font-size:12px;">
      <thead><tr style="text-align:left;border-bottom:1px solid var(--border);color:var(--fg-muted);">
        <th></th><th>Slug</th><th>PG DB</th><th>Papers coll.</th><th>Status</th><th></th>
      </tr></thead>
      <tbody>${{rows}}</tbody></table>`;
    _projMsg(data.projects.length + ' project' + (data.projects.length === 1 ? '' : 's') + '.', 'ok');
  }} catch (exc) {{
    wrap.innerHTML = '';
    _projMsg('Failed to list projects: ' + exc, 'error');
  }}
}}

async function showProjectDetail(slug) {{
  const dest = document.getElementById('proj-detail');
  dest.innerHTML = 'Loading details for <code>' + slug + '</code>…';
  try {{
    const resp = await fetch('/api/projects/' + encodeURIComponent(slug));
    if (!resp.ok) {{
      const msg = await resp.text();
      throw new Error('HTTP ' + resp.status + ': ' + msg);
    }}
    const d = await resp.json();
    const counts = (d.n_documents !== undefined)
      ? `<ul style="margin:6px 0 0 18px;font-size:12px;">
           <li>Documents: <strong>${{(d.n_documents||0).toLocaleString()}}</strong></li>
           <li>Chunks: <strong>${{(d.n_chunks||0).toLocaleString()}}</strong></li>
           <li>Books: <strong>${{d.n_books||0}}</strong></li>
           <li>Drafts: <strong>${{d.n_drafts||0}}</strong></li>
         </ul>`
      : (d.counts_error ? `<div style="color:var(--warning);font-size:11px;">Counts unavailable: ${{d.counts_error}}</div>` : '');
    dest.innerHTML = `<div style="border:1px solid var(--border);border-radius:var(--r-md);padding:10px;background:var(--toolbar-bg);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <strong>${{d.slug}}${{d.is_default ? ' <span style="font-size:11px;color:var(--fg-muted);">(legacy default)</span>' : ''}}</strong>
        <button onclick="document.getElementById('proj-detail').innerHTML=''">&times;</button>
      </div>
      <dl style="display:grid;grid-template-columns:140px 1fr;gap:4px 12px;font-size:12px;margin:0;">
        <dt style="color:var(--fg-muted);">Root</dt><dd style="font-family:var(--font-mono);font-size:11px;">${{d.root}}</dd>
        <dt style="color:var(--fg-muted);">Data dir</dt><dd style="font-family:var(--font-mono);font-size:11px;">${{d.data_dir}}${{d.data_dir_exists ? '' : ' <span style="color:var(--warning);">(missing)</span>'}}</dd>
        <dt style="color:var(--fg-muted);">PG database</dt><dd style="font-family:var(--font-mono);font-size:11px;">${{d.pg_database}}${{d.pg_database_exists ? '' : ' <span style="color:var(--warning);">(missing)</span>'}}</dd>
        <dt style="color:var(--fg-muted);">Qdrant prefix</dt><dd style="font-family:var(--font-mono);font-size:11px;">${{d.qdrant_prefix || '(none)'}}</dd>
        <dt style="color:var(--fg-muted);">Collections</dt><dd style="font-family:var(--font-mono);font-size:11px;">${{d.papers_collection}}, ${{d.abstracts_collection}}, ${{d.wiki_collection}}</dd>
        <dt style="color:var(--fg-muted);">Env overlay</dt><dd style="font-family:var(--font-mono);font-size:11px;">${{d.env_overlay_path}}${{d.env_overlay_exists ? '' : ' <span style="color:var(--fg-faint);">(not present)</span>'}}</dd>
      </dl>
      ${{counts}}
    </div>`;
  }} catch (exc) {{
    dest.innerHTML = '<div style="color:var(--danger);font-size:12px;">Failed: ' + exc + '</div>';
  }}
}}

async function useProject(slug) {{
  _projMsg('Switching active project to ' + slug + '…');
  try {{
    const resp = await fetch('/api/projects/use', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{slug: slug}}),
    }});
    const data = await resp.json();
    if (!resp.ok) {{
      _projMsg('Use failed: ' + (data.detail || resp.status), 'error');
      return;
    }}
    _projMsg(data.message || ('Active project: ' + slug), 'ok');
    if (data.restart_required) {{
      renderProjectSwitchBanner(slug, data.running_slug);
    }}
    refreshProjectsList();
  }} catch (exc) {{
    _projMsg('Use failed: ' + exc, 'error');
  }}
}}

function renderProjectSwitchBanner(newSlug, runningSlug) {{
  // Phase 54.6.2 — after a successful /api/projects/use, replace the
  // old "Ctrl-C your terminal" alert with an inline banner that offers
  // a one-click graceful shutdown. The user still needs to rerun
  // `sciknow book serve` themselves (no supervisor to auto-restart)
  // but at least they don't have to leave the browser.
  const dest = document.getElementById('proj-detail');
  if (!dest) return;
  const safeNew = _escHtml(newSlug);
  const safeRun = _escHtml(runningSlug || '?');
  const cmd = 'sciknow --project ' + safeNew + ' book serve "<book title>"';
  dest.innerHTML =
    '<div style="margin-top:14px;padding:14px;border:2px solid var(--accent);'
    + 'border-radius:8px;background:var(--bg-elevated);">'
    + '<div style="font-weight:bold;margin-bottom:6px;">&#9888;&#65039; Restart required</div>'
    + '<div style="font-size:12px;color:var(--fg-muted);margin-bottom:10px;">'
    + 'The <code>.active-project</code> file now points at <strong>' + safeNew
    + '</strong>, but this server is still bound to <strong>' + safeRun
    + '</strong>. DB / Qdrant clients can&rsquo;t hot-swap, so you need to '
    + 'restart the server to work on the new project.'
    + '</div>'
    + '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">'
    + '<button class="btn-primary" onclick="shutdownServer()" '
    + 'title="Graceful shutdown — your terminal will return to $, ready for the re-run command below">'
    + '&#9211; Stop this server</button>'
    + '<code style="flex:1;padding:4px 8px;background:var(--toolbar-bg);border-radius:4px;font-size:12px;cursor:pointer;"'
    + ' onclick="navigator.clipboard.writeText(this.textContent);_projMsg(&quot;Command copied.&quot;,&quot;ok&quot;);" '
    + 'title="Click to copy">' + _escHtml(cmd) + '</code>'
    + '</div>'
    + '<div style="margin-top:8px;font-size:11px;color:var(--fg-muted);">'
    + 'After stopping, paste the command into your terminal and edit '
    + '<code>"&lt;book title&gt;"</code> to a book that exists in <strong>'
    + safeNew + '</strong>.</div>'
    + '</div>';
}}

async function shutdownServer() {{
  if (!confirm(
    'Stop the running sciknow server?\\n\\n'
    + 'This will return your terminal to the shell prompt. Any in-flight LLM job '
    + 'will be killed. You will need to rerun `sciknow book serve ...` manually '
    + 'to open the reader again.'
  )) return;
  _projMsg('Shutting down…');
  try {{
    await fetch('/api/server/shutdown', {{method: 'POST'}});
    document.body.innerHTML =
      '<div style="padding:40px;max-width:620px;margin:60px auto;font-family:-apple-system,sans-serif;'
      + 'border:1px solid #ddd;border-radius:8px;background:#f8f8f8;">'
      + '<h2 style="margin:0 0 12px;">Server stopped</h2>'
      + '<p>Your terminal is back at <code>$</code>. To pick up the new active project, run:</p>'
      + '<pre style="padding:10px;background:#fff;border:1px solid #ddd;border-radius:4px;">'
      + 'sciknow book serve "&lt;book title&gt;"</pre>'
      + '<p style="font-size:12px;color:#666;margin:12px 0 0;">Then reload this browser tab.</p>'
      + '</div>';
  }} catch (exc) {{
    _projMsg('Shutdown request failed: ' + exc, 'error');
  }}
}}

async function createProject() {{
  const slug = (document.getElementById('proj-new-slug').value || '').trim();
  if (!slug) {{
    _projMsg('Enter a slug first.', 'error');
    return;
  }}
  if (!/^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/.test(slug)) {{
    _projMsg('Slug must be lowercase alphanumerics + hyphens (e.g. "global-cooling").', 'error');
    return;
  }}
  if (!confirm('Create empty project "' + slug + '"? This runs migrations + initialises Qdrant collections (takes a few seconds).')) return;
  _projMsg('Creating ' + slug + '…');
  try {{
    const resp = await fetch('/api/projects/init', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{slug: slug}}),
    }});
    const data = await resp.json();
    if (!resp.ok) {{
      _projMsg('Create failed: ' + (data.detail || resp.status), 'error');
      return;
    }}
    _projMsg('Created ' + slug + ' (DB: ' + data.pg_database + ').', 'ok');
    document.getElementById('proj-new-slug').value = '';
    refreshProjectsList();
  }} catch (exc) {{
    _projMsg('Create failed: ' + exc, 'error');
  }}
}}

async function destroyProject(slug) {{
  const confirmSlug = prompt(
    'DESTROY project "' + slug + '"?\\n\\n'
    + 'This drops the PostgreSQL database, the Qdrant collections, and the data directory. '
    + 'Run `sciknow project archive ' + slug + '` from the CLI first if you might want it back.\\n\\n'
    + 'Type the slug to confirm:');
  if (confirmSlug === null) return;
  if (confirmSlug !== slug) {{
    _projMsg('Slug did not match — destroy cancelled.', 'error');
    return;
  }}
  _projMsg('Destroying ' + slug + '…');
  try {{
    const resp = await fetch('/api/projects/destroy', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{slug: slug, confirm: slug}}),
    }});
    const data = await resp.json();
    if (!resp.ok) {{
      _projMsg('Destroy failed: ' + (data.detail || resp.status), 'error');
      return;
    }}
    const errs = (data.errors && data.errors.length) ? (' (errors: ' + data.errors.join('; ') + ')') : '';
    _projMsg('Destroyed ' + slug + errs, errs ? 'error' : 'ok');
    refreshProjectsList();
  }} catch (exc) {{
    _projMsg('Destroy failed: ' + exc, 'error');
  }}
}}

</script>
</body>
</html>
"""
