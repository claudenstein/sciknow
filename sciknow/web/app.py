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
from sciknow.core.bibliography import (
    BookBibliography,
    BIBLIOGRAPHY_PSEUDO_ID,
    BIBLIOGRAPHY_TITLE,
    render_bibliography_markdown,
)

logger = logging.getLogger("sciknow.web")

app = FastAPI(title="SciKnow Book Reader")

# Phase 54.6.48 — serve vendored frontend libraries (KaTeX + ECharts)
# from `/static/` instead of the public jsdelivr CDN. Eliminates the
# external network dependency (equations / charts work offline), removes
# the third-party IP/User-Agent ping on every page load, and lets us
# enforce Subresource-Integrity-equivalent trust by committing the bytes.
# Files live under sciknow/web/static/vendor/ (see vendor/README for
# origin + versions).
from fastapi.staticfiles import StaticFiles as _StaticFiles
from pathlib import Path as _Path
_STATIC_DIR = _Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", _StaticFiles(directory=str(_STATIC_DIR)), name="static")

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
            # Phase 54.6.317 — bumped to 20000 so multi-window MAs
            # (10s / 1m / 5m / 30m) have enough history for the slow
            # tail. ~160 KB worst case per job, negligible.
            "token_timestamps": deque(maxlen=20000),
            "model_name": None,
            "task_desc": job_type,
            "target_words": None,
            "stream_state": "streaming",  # streaming | done | error
            "error_message": None,
            # Phase 54.6.246 — placeholder so the cross-process pulse
            # still knows about this job before the first `token` /
            # `progress` event fires. Filled in by `_observe_event_for_stats`
            # on the first model_info event.
            # Phase 50.A — reasoning-steps trace. Collected automatically
            # by _observe_event_for_stats from the same event stream the
            # task bar already watches; persisted to the new draft's
            # custom_metadata.reasoning_trace in _run_generator_in_thread's
            # finally block. Capped to avoid log-level bloat.
            "reasoning_trace": [],
            "reasoning_draft_id": None,
            # Phase 54.6.318 — per-call Ollama stats accumulator.
            # Each completed LLM call appends:
            #   {"eval_count": int, "eval_duration_ns": int,
            #    "prompt_eval_count": int, "prompt_eval_duration_ns": int,
            #    "load_duration_ns": int, "total_duration_ns": int,
            #    "model": str}
            # so the dashboard can compute decode tok/s separately
            # from wall-clock tok/s — under heavy prefill workloads
            # (long contexts, multi-question CoV) wall-clock is
            # dominated by prefill periods that emit no tokens.
            "llm_calls": [],
        }
        # Phase 54.6.246 — announce the new job to the cross-process
        # pulse so `sciknow db monitor` sees it before the first event.
        _write_web_jobs_pulse()
    return job_id, queue


def _job_tps(job: dict, *, window_s: float = 3.0) -> float:
    """Phase 54.6.247 — rolling tokens-per-second over the last
    ``window_s`` seconds, derived from the per-job
    ``token_timestamps`` deque maintained by
    ``_observe_event_for_stats``. Shared by the pulse writer and
    the ``/api/jobs/{id}/stats`` endpoint so both UIs see the same
    number (was duplicated inline pre-247)."""
    ts = job.get("token_timestamps") or deque()
    if not ts:
        return 0.0
    now = time.monotonic()
    cutoff = now - window_s
    recent = sum(1 for t in ts if t >= cutoff)
    return (recent / window_s) if recent else 0.0


def _job_decode_stats(job: dict) -> dict:
    """Phase 54.6.318 — aggregate Ollama-reported decode + prefill
    rates across every completed LLM call this job has made so far.

    Distinguishes three rates the user actually cares about:

    - **decode_tps**: tokens emitted / time spent decoding.
      Reflects raw model throughput. Should match the published
      benchmark (~30–40 t/s for Qwen3.6-27B Q4_K_M on a 3090).
    - **prefill_tps**: prompt tokens / time spent prefilling.
      Typically 5–15× higher than decode (200–500 t/s). Usually
      irrelevant for total wall-clock unless contexts are huge.
    - **wall_tps** (= the existing rolling tps): tokens emitted /
      wall-clock time. Includes all silent prefill / model-load
      / inter-call gaps. THIS is what looks like "1 t/s" in the
      task bar when the model is fine but the workload is
      prefill-dominated.

    Returns:
      {"decode_tps": float, "prefill_tps": float, "calls": int,
       "prefill_share": float (0..1, fraction of LLM time spent
       prefilling vs decoding), "load_seconds": float}
    """
    calls = job.get("llm_calls") or []
    if not calls:
        return {"decode_tps": 0.0, "prefill_tps": 0.0, "calls": 0,
                "prefill_share": 0.0, "load_seconds": 0.0}
    eval_count = sum(c.get("eval_count", 0) for c in calls)
    eval_dur = sum(c.get("eval_duration_ns", 0) for c in calls) / 1e9
    pre_count = sum(c.get("prompt_eval_count", 0) for c in calls)
    pre_dur = sum(c.get("prompt_eval_duration_ns", 0) for c in calls) / 1e9
    load_dur = sum(c.get("load_duration_ns", 0) for c in calls) / 1e9
    decode_tps = (eval_count / eval_dur) if eval_dur > 0 else 0.0
    prefill_tps = (pre_count / pre_dur) if pre_dur > 0 else 0.0
    total_active = eval_dur + pre_dur
    prefill_share = (pre_dur / total_active) if total_active > 0 else 0.0
    return {
        "decode_tps": round(decode_tps, 2),
        "prefill_tps": round(prefill_tps, 2),
        "calls": len(calls),
        "prefill_share": round(prefill_share, 3),
        "load_seconds": round(load_dur, 2),
    }


def _job_tps_windows(job: dict) -> dict:
    """Phase 54.6.317 — rolling tokens-per-second over four standard
    windows: 10 s (instantaneous-ish), 1 min, 5 min, 30 min. Each
    window is normalised by its own duration so values are directly
    comparable.

    The CLI dashboard picks which windows to show based on how long
    the job has been running:

    - elapsed <    1 min: only ``w10s`` is meaningful; longer windows
                          would just be the same number with lower
                          resolution (a window can't "average" data it
                          doesn't have yet).
    - elapsed >=   1 min: add ``w1m``.
    - elapsed >=   5 min: add ``w5m``.
    - elapsed >=  30 min: add ``w30m``.

    Returns a dict ``{"w10s": float, "w1m": float, "w5m": float, "w30m": float}``.
    Zeros for windows with no data are kept in the response so the
    consumer always sees the same shape.
    """
    ts = job.get("token_timestamps") or deque()
    if not ts:
        return {"w10s": 0.0, "w1m": 0.0, "w5m": 0.0, "w30m": 0.0}
    now = time.monotonic()
    out: dict[str, float] = {}
    for tag, secs in (("w10s", 10.0), ("w1m", 60.0), ("w5m", 300.0), ("w30m", 1800.0)):
        cutoff = now - secs
        n = sum(1 for t in ts if t >= cutoff)
        out[tag] = round(n / secs, 2) if n else 0.0
    return out


def _write_web_jobs_pulse() -> None:
    """Phase 54.6.246 — cross-process active-job pulse.

    Serialises the currently-active slice of ``_jobs`` into
    ``<data_dir>/monitor/web_jobs.json`` so a separate ``sciknow db
    monitor`` process (typically an SSH session) can see what the
    web server is working on. Best-effort — any failure is swallowed
    rather than blocking the streaming event loop.

    Throttled implicitly by the callers: written from
    ``_observe_event_for_stats`` only on state-transition events
    (progress / model_info / length_target / completed / error /
    cancelled), NOT per-token — which keeps file writes at tens per
    section rather than thousands. The monitor's own pulse staleness
    threshold (120s) means a job that genuinely stalls still stops
    showing as active after a reasonable window.

    Caller must hold ``_job_lock`` so the jobs dict doesn't mutate
    under the iteration.

    Phase 54.6.247 — pulse payload now carries ``tps`` (3-second
    rolling tokens/sec) per job so downstream CLI can show the
    generation rate. Zero when no tokens have streamed yet.
    """
    try:
        from sciknow.core.pulse import write_pulse
        from sciknow.core.project import get_active_project
        now = time.monotonic()
        active: list[dict] = []
        for jid, j in _jobs.items():
            if j.get("status") not in ("running", "starting"):
                continue
            if j.get("stream_state") not in ("streaming", "starting", None):
                # "done"/"error" — already terminal; skip
                continue
            started = j.get("started_at") or now
            active.append({
                "id": jid[:8],
                "type": j.get("task_desc") or j.get("type") or "?",
                "model": j.get("model_name"),
                "tokens": j.get("tokens", 0),
                "tps": round(_job_tps(j), 2),
                # Phase 54.6.317 — multi-window MAs so the dashboard
                # can show 10s / 1m / 5m / 30m bands keyed off elapsed.
                "tps_windows": _job_tps_windows(j),
                # Phase 54.6.318 — Ollama-reported decode + prefill
                # rates. Resolves "the wall-clock tok/s reads 1.0 but
                # the model is rated 30 t/s" into "decode 31 t/s,
                # prefill 250 t/s, prefill_share 0.92" — i.e. the
                # workload is 92% prefill so the wall-clock rate is
                # decoupled from raw model speed.
                "decode_stats": _job_decode_stats(j),
                "target_words": j.get("target_words"),
                "elapsed_s": max(0.0, now - started),
                "stream_state": j.get("stream_state"),
            })
        data_dir = get_active_project().data_dir
        write_pulse(data_dir, "web_jobs", {"active": active})
    except Exception:
        pass


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

    Phase 54.6.246 — also writes a cross-process active-jobs pulse
    on non-token events so `sciknow db monitor` over SSH sees which
    jobs are running without polling the web endpoint.
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

        # Phase 54.6.246 — refresh the cross-process jobs pulse on
        # state transitions (every non-token event). Skipping `token`
        # keeps writes at ~tens/section rather than thousands; the
        # task bar's in-process poll already covers per-token updates.
        # Caller holds _job_lock, required by _write_web_jobs_pulse.
        if et != "token":
            _write_web_jobs_pulse()

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
            SELECT id::text, title, description, plan, status, custom_metadata,
                   book_type
            FROM books WHERE id::text = :bid
        """), {"bid": _book_id}).fetchone()

        chapters = session.execute(text("""
            SELECT bc.id::text, bc.number, bc.title, bc.description,
                   bc.topic_query, bc.topic_cluster, bc.sections
            FROM book_chapters bc
            WHERE bc.book_id = :bid ORDER BY bc.number
        """), {"bid": _book_id}).fetchall()

        # Phase 54.6.309 — keep the tuple shape (14 columns) but push an
        # ``is_active`` sort key into ORDER BY so a user-pinned version
        # beats MAX(version). The downstream collapse in _render_book
        # already picks the first row per (chapter, section), so this
        # ordering is enough — no consumer needs to read the flag.
        drafts = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.content,
                   d.word_count, d.sources, d.version, d.summary,
                   d.review_feedback, d.chapter_id::text,
                   d.parent_draft_id::text, d.created_at,
                   bc.number AS ch_num, bc.title AS ch_title
            FROM drafts d
            LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.book_id = :bid
            ORDER BY bc.number,
                     d.section_type,
                     -- Phase 54.6.321 — prefer NON-EMPTY content. A
                     -- crashed autowrite can leave the highest-version
                     -- draft with empty content; the previous order
                     -- picked it as active and the GUI displayed a
                     -- blank section even though earlier versions had
                     -- 10K+ chars. Sort empty drafts to the bottom of
                     -- their (chapter, section) group so the
                     -- "first-seen wins" collapse downstream picks the
                     -- newest *populated* version instead.
                     CASE WHEN COALESCE((d.custom_metadata->>'is_active')::boolean, FALSE)
                          THEN 0 ELSE 1 END,
                     CASE WHEN d.content IS NULL OR LENGTH(d.content) < 50
                          THEN 1 ELSE 0 END,
                     d.version DESC
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
    # Phase 54.6.87 — markdown images ![alt](url). Constrained to our
    # own /api/visuals/image/ paths OR absolute http(s) URLs so a user
    # can't smuggle ``javascript:`` or ``data:`` URIs through the
    # renderer. alt text is escaped via the existing citation path.
    def _img_sub(m: "re.Match") -> str:
        alt = (m.group(1) or "").replace('"', "&quot;").replace("<", "&lt;")
        src = (m.group(2) or "").strip()
        if not (src.startswith("/api/visuals/image/")
                or src.startswith("http://")
                or src.startswith("https://")
                or src.startswith("/static/")):
            # Unknown scheme → drop the image; leave the alt as plain text.
            return alt
        return (
            f'<img class="inline-figure" src="{src}" alt="{alt}" '
            f'loading="lazy" style="max-width:100%;height:auto;'
            f'border:1px solid var(--border,#ddd);border-radius:6px;'
            f'margin:10px 0;" title="{alt}">'
        )
    html = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", _img_sub, html)
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
    # Phase 54.6.50 — disable browser caching of the reader page. The
    # HTML is ~680 KB and embeds all JS inline; when the backend restarts
    # with a new commit, browsers that cached the previous response keep
    # serving the old JS and the user sees stale behaviour (e.g. the
    # multi-select disambiguation banner from Phase 54.6.49 silently
    # missing because the browser has the pre-49 script cached). This
    # header trades a re-fetch of ~680 KB on every navigation for
    # always-fresh code, which is worth it for a local-only dev tool.
    resp = HTMLResponse(_render_book(book, chapters, drafts, gaps, comments))
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


@app.get("/section/{draft_id}", response_class=HTMLResponse)
async def section(draft_id: str):
    book, chapters, drafts, gaps, comments = _get_book_data()
    return HTMLResponse(_render_book(book, chapters, drafts, gaps, comments, focus_draft=draft_id))


# ── Phase 54.6.178 — routed views ────────────────────────────────────────────
# Each modal that serves as a standalone "place" (Plan, Settings, Wiki, …)
# gets its own URL. The page still renders the reader layout — shell +
# sidebar + main — but the matching modal auto-opens on load, so the URL
# is shareable and the browser back/forward buttons traverse modal state.
# This is deeplink-style routing, not a layout swap: the modal floats over
# the reader as before, but its visibility is now URL-driven.

_ROUTE_MODALS = {
    "/plan":        "plan-modal",
    "/settings":    "book-settings-modal",
    "/wiki":        "wiki-modal",
    "/bundles":     "bundles-modal",
    "/tools":       "tools-modal",
    "/projects":    "projects-modal",
    "/catalog":     "catalog-modal",
    "/export":      "export-modal",
    "/corpus":      "corpus-modal",
    "/visualize":   "viz-modal",
    "/kg":          "kg-modal",
    "/ask":         "ask-modal",
    "/setup":       "setup-modal",
    "/backups":     "backups-modal",
    "/visuals":     "visuals-modal",
    "/help":        "ai-help-modal",
}


def _routed_view(modal_id: str) -> HTMLResponse:
    """Render the book page with ``modal_id`` auto-opened on load."""
    book, chapters, drafts, gaps, comments = _get_book_data()
    if not book:
        return HTMLResponse("<h1>Book not found</h1>", status_code=404)
    html = _render_book(book, chapters, drafts, gaps, comments,
                        auto_open_modal=modal_id)
    resp = HTMLResponse(html)
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


@app.get("/plan",      response_class=HTMLResponse)
async def route_plan():      return _routed_view("plan-modal")

@app.get("/settings",  response_class=HTMLResponse)
async def route_settings():  return _routed_view("book-settings-modal")

@app.get("/wiki",      response_class=HTMLResponse)
async def route_wiki():      return _routed_view("wiki-modal")

@app.get("/bundles",   response_class=HTMLResponse)
async def route_bundles():   return _routed_view("bundles-modal")

@app.get("/tools",     response_class=HTMLResponse)
async def route_tools():     return _routed_view("tools-modal")

@app.get("/projects",  response_class=HTMLResponse)
async def route_projects():  return _routed_view("projects-modal")

@app.get("/catalog",   response_class=HTMLResponse)
async def route_catalog():   return _routed_view("catalog-modal")

@app.get("/export",    response_class=HTMLResponse)
async def route_export():    return _routed_view("export-modal")

@app.get("/corpus",    response_class=HTMLResponse)
async def route_corpus():    return _routed_view("corpus-modal")

@app.get("/visualize", response_class=HTMLResponse)
async def route_visualize(): return _routed_view("viz-modal")

@app.get("/kg",        response_class=HTMLResponse)
async def route_kg():        return _routed_view("kg-modal")

@app.get("/ask",       response_class=HTMLResponse)
async def route_ask():       return _routed_view("ask-modal")

@app.get("/setup",     response_class=HTMLResponse)
async def route_setup():     return _routed_view("setup-modal")

@app.get("/backups",   response_class=HTMLResponse)
async def route_backups():   return _routed_view("backups-modal")

@app.get("/visuals",   response_class=HTMLResponse)
async def route_visuals():   return _routed_view("visuals-modal")

@app.get("/help",      response_class=HTMLResponse)
async def route_help():      return _routed_view("ai-help-modal")


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
            VALUES (CAST(:did AS uuid), :para, :sel, :comment)
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
    # Phase 54.6.148 — expose book_type so Book Settings can restore
    # the dropdown selection, and derive the project-type default for
    # the effective-target display so users see where the fallback
    # currently lands.
    from sciknow.core.project_type import get_project_type
    book_type = (book[6] if book and len(book) > 6 else None) or "scientific_book"
    try:
        pt = get_project_type(book_type)
        default_tcw = pt.default_target_chapter_words
    except Exception:
        default_tcw = 6000
    return {
        "id": book[0] if book else "",
        "title": book[1] if book else "",
        "description": (book[2] or "") if book else "",
        "plan": (book[3] or "") if book else "",
        "status": (book[4] or "draft") if book else "draft",
        "target_chapter_words": target_chapter_words,  # may be None → client shows default
        "default_target_chapter_words": default_tcw,
        "book_type": book_type,
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
    book_type: str = Form(None),  # Phase 54.6.148
):
    """Update the book's title, description (short blurb), plan
    (the 200-500 word thesis/scope document used by the writer prompt),
    length target, or project type. All fields are optional — only the
    ones you pass get updated.

    Phase 17 — target_chapter_words lives in books.custom_metadata as
    a JSONB key so we can add more book-level settings without a
    schema change each time. Passing a zero or negative value clears
    the setting (reverts to the project-type default via 54.6.143's
    fallback chain).

    Phase 54.6.148 — book_type can be changed post-creation. Validated
    against the ProjectType registry (rejects unknown slugs). Changing
    type only affects future autowrite runs via the resolver's Level 3
    fallback; explicit per-chapter / per-section targets are unchanged.
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
    if book_type is not None:
        # Validate against the registry so a typo doesn't silently
        # downgrade to the default fallback in get_project_type.
        from sciknow.core.project_type import validate_type_slug
        try:
            validate_type_slug(book_type)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        updates.append("book_type = :btype")
        params["btype"] = book_type
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


@app.post("/api/book/outline/generate")
async def api_book_outline_generate(
    model: str = Form(None),
    method: str = Form(""),
):
    """Phase 54.6.8 — generate + save chapter outline for the active book.

    Mirrors ``sciknow book outline``: prompts the LLM with the book title
    + paper corpus, parses the JSON chapters list, inserts any chapter
    whose number isn't already in ``book_chapters``. Streams tokens so
    the user sees progress; existing chapters are preserved (the flow
    is additive, never destructive).

    Phase 54.6.14 — optional ``method`` name from the elicitation
    catalogue steers the LLM's approach (e.g. "Tree of Thoughts",
    "First Principles"). Prepended as a one-paragraph preamble to
    the user prompt.
    """
    job_id, queue = _create_job("book_outline_generate")
    loop = asyncio.get_event_loop()

    def gen():
        import json as _json
        from sciknow.rag import prompts as rag_prompts
        from sciknow.rag.llm import stream as llm_stream
        from sciknow.core.project_type import get_project_type
        from sciknow.core.methods import get_method, method_preamble

        with get_session() as session:
            book = session.execute(text("""
                SELECT id::text, title, description, project_type
                FROM books WHERE id::text = :bid
            """), {"bid": _book_id}).fetchone()
            if not book:
                yield {"type": "error", "message": "Book not found."}
                return
            # Flat project types have a fixed one-chapter shape. Refuse
            # to generate an outline for them — mirrors the CLI.
            pt = get_project_type(book[3] if len(book) > 3 else None)
            if pt.is_flat:
                yield {"type": "error", "message":
                       f"Project type {pt.slug!r} is flat — no outline to generate."}
                return
            papers = session.execute(text("""
                SELECT pm.title, pm.year FROM paper_metadata pm
                JOIN documents d ON d.id = pm.document_id
                WHERE d.ingestion_status = 'complete' AND pm.title IS NOT NULL
                ORDER BY pm.year DESC NULLS LAST LIMIT 200
            """)).fetchall()

        paper_list = [{"title": r[0], "year": r[1]} for r in papers if r[0]]
        yield {"type": "progress", "stage": "generating",
               "detail": f"Drafting outline from {len(paper_list)} papers…"}

        sys_p, usr_p = rag_prompts.outline(book_title=book[1], papers=paper_list)
        # Inject method preamble if the user picked one.
        if method and method.strip():
            m = get_method("elicitation", method)
            if m:
                usr_p = method_preamble(m) + usr_p
        # Phase 54.6.297 — resolve the outline-specific model.
        # Explicit request param > BOOK_OUTLINE_MODEL env > LLM_MODEL
        # (the llm_stream default).
        from sciknow.config import settings as _settings
        effective_model = (
            model
            or getattr(_settings, "book_outline_model", None)
            or None
        )
        tokens: list[str] = []
        for tok in llm_stream(sys_p, usr_p, model=effective_model):
            tokens.append(tok)
            yield {"type": "token", "text": tok}

        raw = "".join(tokens).strip()
        # Same JSON-fence strip as the CLI.
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            data = _json.loads(raw, strict=False)
            chapters = data.get("chapters", [])
        except Exception as exc:
            yield {"type": "error",
                   "message": f"LLM returned invalid JSON: {exc}"}
            return

        if not chapters:
            yield {"type": "error",
                   "message": "No chapters in LLM response."}
            return

        # Phase 54.6.65 — per-chapter density-based section trim. One
        # hybrid retrieval per chapter on topic_query → count distinct
        # papers → bucket to a target section count → trim if over.
        try:
            from sciknow.core.book_ops import resize_sections_by_density as _resize
            yield {"type": "progress", "stage": "resizing",
                   "detail": "Resizing sections by corpus evidence density…"}
            # Phase 54.6.297 — pass the outline-specific model into
            # _grow_sections_llm so the resize step agrees with the
            # generation step on model choice.
            chapters = _resize(chapters, model=effective_model)
        except Exception as exc:
            logger.warning("density resize failed: %s", exc)

        # Additive insert — skip numbers that already exist so the
        # user can re-run without destroying manual edits.
        inserted = 0
        skipped = 0
        with get_session() as session:
            for ch in chapters:
                num = ch.get("number")
                if not isinstance(num, int):
                    continue
                existing = session.execute(text("""
                    SELECT id FROM book_chapters
                    WHERE book_id::text = :bid AND number = :num
                """), {"bid": _book_id, "num": num}).fetchone()
                if existing:
                    skipped += 1
                    continue
                sections_json = _json.dumps(ch.get("sections", []) or [])
                session.execute(text("""
                    INSERT INTO book_chapters
                        (book_id, number, title, description, topic_query, sections)
                    VALUES (CAST(:bid AS uuid), :num, :title,
                            :desc, :tq, CAST(:secs AS jsonb))
                """), {
                    "bid": _book_id, "num": num,
                    "title": ch.get("title") or f"Chapter {num}",
                    "desc": ch.get("description"),
                    "tq": ch.get("topic_query"),
                    "secs": sections_json,
                })
                inserted += 1
            session.commit()

        yield {"type": "completed", "n_chapters": len(chapters),
               "n_inserted": inserted, "n_skipped": skipped}

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
        # Phase 54.6.309 — first-seen wins; the SELECT already orders
        # active-first then MAX(version), see _get_book_data.
        if not existing:
            chapter_drafts[key].append(d)

    # Phase 27 — chapter_id → sections JSONB lookup so the display
    # title can be derived from the meta even if drafts.title is stale.
    chapter_sections_by_id = {ch[0]: ch[6] for ch in chapters}

    # Phase 54.6.309 — synthetic Bibliography pseudo-chapter.
    if draft_id == BIBLIOGRAPHY_PSEUDO_ID:
        try:
            with get_session() as _session:
                _bib = BookBibliography.from_book(_session, _book_id)
        except Exception as exc:
            logger.warning("bibliography rebuild failed: %s", exc)
            _bib = BookBibliography()
        _md = render_bibliography_markdown(_bib)
        _bib_ch_num = (chapters[-1][1] if chapters else 0) + 1 if chapters else 1
        return {
            "id": BIBLIOGRAPHY_PSEUDO_ID,
            "title": BIBLIOGRAPHY_TITLE,
            "display_title": BIBLIOGRAPHY_TITLE,
            "section_type": "bibliography",
            "content_html": _md_to_html(_md),
            "content_raw": _md,
            "word_count": sum(len((s or "").split()) for s in _bib.global_sources),
            "version": 1,
            "review_feedback": "",
            "review_html": "<em>The bibliography is auto-generated and not reviewed.</em>",
            "sources_html": _render_sources(_bib.global_sources),
            "sources": list(_bib.global_sources),
            "comments_html": "",
            "chapter_id": BIBLIOGRAPHY_PSEUDO_ID,
            "chapter_num": _bib_ch_num,
            "chapter_title": BIBLIOGRAPHY_TITLE,
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
            _bib = BookBibliography.from_book(_session, _book_id)
        _remapped_content = _bib.remap_content(draft_id, active[3] or "")
        _cited_globals = _bib.cited_sources_for_draft(draft_id)
        if _cited_globals:
            sources_for_panel = _cited_globals
        else:
            sources_for_panel = sources
    except Exception as _exc:
        logger.warning("global renumber failed (draft %s): %s", draft_id, _exc)
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
        "content_html": _md_to_html(_remapped_content),
        "content_raw": _remapped_content,
        "word_count": active[4] or 0,
        "version": active[6] or 1,
        "review_feedback": active[8] or "",
        "review_html": _md_to_html(active[8]) if active[8] else "<em>No review yet.</em>",
        "sources_html": _render_sources(sources_for_panel),
        "sources": sources_for_panel,
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
        # Phase 54.6.309 — first-seen wins (see _get_book_data).
        if not existing:
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

    # Phase 54.6.309 — synthetic Bibliography pseudo-chapter appended at
    # the end. Survives sidebar rebuilds after writes/revises. The JS
    # renderer uses `is_bibliography` to skip the delete button + the
    # "Ch.N:" prefix.
    if result:
        try:
            with get_session() as _bib_session:
                _bib = BookBibliography.from_book(_bib_session, _book_id)
        except Exception as exc:
            logger.warning("bibliography fetch failed: %s", exc)
            _bib = BookBibliography()
        _bib_ch_num = int(result[-1]["num"] or len(result)) + 1
        result.append({
            "id": BIBLIOGRAPHY_PSEUDO_ID,
            "num": _bib_ch_num,
            "title": BIBLIOGRAPHY_TITLE,
            "description": "All publications cited across the book, numbered once.",
            "topic_query": "",
            "sections": [{
                "id": BIBLIOGRAPHY_PSEUDO_ID,
                "type": "bibliography",
                "version": 1,
                "words": sum(len((s or "").split()) for s in _bib.global_sources),
            }],
            "sections_template": ["bibliography"],
            "sections_meta": [{
                "slug": "bibliography",
                "title": BIBLIOGRAPHY_TITLE,
                "plan": "Auto-generated — do not edit by hand.",
            }],
            "is_bibliography": True,
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
        # Phase 54.6.309 — _get_book_data sorts active-first then highest-
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


@app.post("/api/draft/{draft_id}/activate")
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


@app.post("/api/draft/{draft_id}/save-as-version")
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


@app.post("/api/draft/{draft_id}/rename-version")
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


@app.post("/api/draft/{draft_id}/version-description")
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


# ── Bibliography endpoints (Phase 54.6.312) ────────────────────────────
# The global bibliography is rebuilt on every book-reader load via
# `BookBibliography.from_book`, so the raw draft content stays in
# *local* numbering and remap_content projects it to global at render
# time. These endpoints expose three user-facing operations the UI was
# missing:
#
#   GET  /api/bibliography/audit  — per-draft sanity check (broken refs,
#                                   orphan sources, duplicates).
#   POST /api/bibliography/sort   — flatten local→global numbers INTO
#                                   the stored draft content so the
#                                   numbers the reader sees match the
#                                   markdown the editor shows. Reorders
#                                   draft.sources to global order too.
#   GET  /api/bibliography/citation/{book_id}/{n}
#                                — fetch full publication metadata for
#                                  a global citation number (title,
#                                  authors, year, journal, doi, abstract,
#                                  open-access URL) for click-to-preview.


@app.get("/api/bibliography/audit")
async def api_bibliography_audit():
    """Per-draft sanity check for the book-wide bibliography.

    For each draft, reports:
      - broken_refs: citation numbers used in the body that have NO
                     matching entry in the draft's sources array.
      - orphan_sources: sources listed on the draft but never cited
                        in its body text.
      - duplicate_keys: the same publication appearing twice at
                        different local numbers (after strip ``[N]``).

    The user hits "Sanity check" once after a bibliography churn to see
    if the rewrite left any draft with references that point at a
    non-existent paper, which is almost always what "the citations look
    changed" means in practice.
    """
    book, _chapters, _drafts, _gaps, _comments = _get_book_data()
    if not book:
        return JSONResponse({"rows": [], "note": "No active book."})

    import re as _re
    rows: list[dict] = []
    totals = {"drafts_checked": 0, "broken": 0, "orphans": 0, "dupes": 0}

    with get_session() as session:
        draft_rows = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.content, d.sources,
                   d.version, d.chapter_id::text, d.custom_metadata,
                   COALESCE(bc.number, 999999) AS ch_num,
                   bc.title AS ch_title
            FROM drafts d
            LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.book_id = :bid
            ORDER BY ch_num, d.section_type, d.version DESC
        """), {"bid": str(book[0])}).fetchall()

    # Collapse to the active draft per (chapter, section) pair.
    def _meta_dict(raw):
        if isinstance(raw, dict): return raw
        if isinstance(raw, str) and raw:
            try: return json.loads(raw) or {}
            except Exception: return {}
        return {}

    # Phase 54.6.321 — same three-tier pick as core.bibliography:
    # is_active wins, then prefer non-empty content, then highest version.
    best: dict[tuple, tuple] = {}
    for r in draft_rows:
        did, title, sec, content, sources, ver, ch_id, meta, ch_num, ch_title = r
        key = (ch_id, sec)
        is_active = bool(_meta_dict(meta).get("is_active"))
        has_content = bool((content or "").strip())
        cur = best.get(key)
        if cur is None:
            best[key] = (r, is_active, has_content); continue
        cur_row, cur_active, cur_content = cur
        if is_active and not cur_active:
            best[key] = (r, True, has_content); continue
        if cur_active and not is_active:
            continue
        if has_content and not cur_content:
            best[key] = (r, is_active, True); continue
        if cur_content and not has_content:
            continue
        if (ver or 1) > (cur_row[5] or 1):
            best[key] = (r, is_active, has_content)

    for r, _is_a, _has_c in best.values():
        did, title, sec, content, sources, ver, ch_id, meta, ch_num, ch_title = r
        src_list = sources if isinstance(sources, list) else []
        src_by_num: dict[int, str] = {}
        for s in src_list:
            m = _re.match(r"^\s*\[(\d+)\]\s*(.*)$", s or "")
            if not m: continue
            src_by_num[int(m.group(1))] = m.group(2).strip()
        cited_nums = set()
        for m in _re.finditer(r"\[(\d+)\]", content or ""):
            cited_nums.add(int(m.group(1)))
        broken = sorted([n for n in cited_nums if n not in src_by_num])
        orphans = sorted([n for n in src_by_num.keys() if n not in cited_nums])
        # Duplicate detection: same normalized source body under two nums.
        seen_bodies: dict[str, list[int]] = {}
        for n, body in src_by_num.items():
            key2 = _re.sub(r"\s+", " ", body).lower().strip()[:300]
            seen_bodies.setdefault(key2, []).append(n)
        dupes = [nums for nums in seen_bodies.values() if len(nums) > 1]
        if broken or orphans or dupes:
            rows.append({
                "draft_id": did,
                "title": title,
                "section_type": sec,
                "chapter_num": int(ch_num) if ch_num and ch_num != 999999 else None,
                "chapter_title": ch_title,
                "broken_refs": broken,
                "orphan_sources": orphans,
                "duplicate_groups": dupes,
            })
            totals["broken"] += len(broken)
            totals["orphans"] += len(orphans)
            totals["dupes"] += len(dupes)
        totals["drafts_checked"] += 1

    return JSONResponse({"rows": rows, "totals": totals})


@app.post("/api/bibliography/sort")
async def api_bibliography_sort():
    """Flatten the global bibliography numbering INTO the stored draft
    content so the editor, markdown, and reader all agree on the same
    ``[N]`` values.

    Without this, the reader shows the global numbering (via
    ``BookBibliography.remap_content``) but the raw markdown in the
    editor still has the original local numbers. Running "Sort" once
    rewrites each draft's content + sources list so ``[N]`` everywhere
    means the same paper. Idempotent — re-running after new sections
    are drafted re-numbers them into the book's reading order.
    """
    book, _chapters, _drafts, _gaps, _comments = _get_book_data()
    if not book:
        return JSONResponse({"ok": False, "note": "No active book."})

    import re as _re
    from sciknow.core.bibliography import BookBibliography, _renumber_source_line

    updated = 0
    with get_session() as session:
        bib = BookBibliography.from_book(session, book[0])
        for did, lmap in bib.draft_local_to_global.items():
            if not lmap: continue
            row = session.execute(text(
                "SELECT content, sources FROM drafts WHERE id::text = :did"
            ), {"did": did}).fetchone()
            if not row: continue
            content, sources_raw = row
            # Rewrite body [N] → [G] via a placeholder two-pass.
            def _to_ph(match):
                n = int(match.group(1))
                g = lmap.get(n)
                return f"[__CITE_{g}__]" if g is not None else match.group(0)
            new_content = _re.sub(r"\[(\d+)\]", _to_ph, content or "")
            new_content = _re.sub(r"\[__CITE_(\d+)__\]", r"[\1]", new_content)
            # Rewrite the sources list to carry the new global numbers,
            # sorted ascending so the draft's local sources tab reads
            # 1,2,3… of the paper's actual global order.
            src_in = sources_raw if isinstance(sources_raw, list) else []
            new_sources: list[tuple[int, str]] = []
            for s in src_in:
                m = _re.match(r"^\s*\[(\d+)\]\s*", s or "")
                if not m: continue
                n = int(m.group(1))
                g = lmap.get(n)
                if g is None: continue
                new_sources.append((g, _renumber_source_line(s, g)))
            new_sources.sort(key=lambda t: t[0])
            if new_content != (content or "") or [s for _, s in new_sources] != (src_in or []):
                session.execute(text("""
                    UPDATE drafts SET content = :c, sources = CAST(:s AS jsonb)
                     WHERE id::text = :did
                """), {"c": new_content,
                       "s": json.dumps([s for _, s in new_sources]),
                       "did": did})
                updated += 1
        session.commit()

    return JSONResponse({"ok": True, "drafts_updated": updated,
                          "total_global_sources": len(bib.global_sources)})


@app.get("/api/bibliography/citation/{global_num}")
async def api_bibliography_citation(global_num: int):
    """Return full metadata for a global citation number — title,
    authors, year, journal, doi, abstract, best-open-URL — so the
    reader can render a click-to-preview popover richer than the hover
    tooltip in ``buildPopovers``.
    """
    book, _chapters, _drafts, _gaps, _comments = _get_book_data()
    if not book:
        raise HTTPException(404, "No active book.")
    from sciknow.core.bibliography import BookBibliography, _source_key
    import re as _re
    with get_session() as session:
        bib = BookBibliography.from_book(session, book[0])
        idx = int(global_num) - 1
        if idx < 0 or idx >= len(bib.global_sources):
            raise HTTPException(404, "Citation out of range.")
        source_line = bib.global_sources[idx]
        # Attempt to map the source line back to a paper_metadata row.
        key = _source_key(source_line)
        # Best-effort: extract the title inside ... (year). Title. journal
        m = _re.search(r"\(\d{4}(?:-\d{2}-\d{2})?\)\.\s+([^.]+)\.", source_line)
        title_guess = m.group(1).strip() if m else None
        meta: dict | None = None
        if title_guess:
            mrow = session.execute(text("""
                SELECT document_id::text, title, authors, year, journal, doi, abstract,
                       open_access_url, url
                FROM paper_metadata
                WHERE lower(title) = lower(:t)
                LIMIT 1
            """), {"t": title_guess}).fetchone()
            if mrow:
                did, title, authors, year, journal, doi, abstract, oa_url, url = mrow
                meta = {
                    "document_id": did,
                    "title": title,
                    "authors": authors or [],
                    "year": year,
                    "journal": journal,
                    "doi": doi,
                    "abstract": (abstract or "")[:1200],
                    "open_access_url": oa_url or url or (f"https://doi.org/{doi}" if doi else None),
                }
    return JSONResponse({
        "global_num": int(global_num),
        "source_line": source_line,
        "key": key,
        "metadata": meta,
    })


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
    # Phase 54.6.309 — refuse to "delete" the synthetic Bibliography
    # chapter. It's derived from draft.sources at render time, so there
    # is nothing to delete — suppress with a 400 rather than no-op so
    # the UI surfaces the cause.
    if chapter_id == BIBLIOGRAPHY_PSEUDO_ID:
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
            VALUES (CAST(:did AS uuid), :name, :content, :wc)
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


@app.get("/api/ledger/book/{book_id}")
async def api_ledger_book(book_id: str):
    """Phase 54.6.76 (#15) — GPU-time + token ledger for a book.

    Returns the book-level rollup plus a per-chapter breakdown
    (`{header: {...}, chapters: [...]}`). Empty `chapters` list means
    nothing was autowrite-generated in this book yet.
    """
    from sciknow.core import gpu_ledger
    with get_session() as session:
        header = gpu_ledger.ledger_for_book(session, book_id)
        if header is None:
            raise HTTPException(404, "Book not found")
        per_ch = gpu_ledger.ledger_per_chapter(session, book_id)
    return {
        "header": gpu_ledger.ledger_as_dict(header),
        "chapters": [gpu_ledger.ledger_as_dict(r) for r in per_ch],
    }


@app.get("/api/ledger/chapter/{chapter_id}")
async def api_ledger_chapter(chapter_id: str):
    """Phase 54.6.76 (#15) — ledger for one chapter with per-section rows."""
    from sciknow.core import gpu_ledger
    with get_session() as session:
        header = gpu_ledger.ledger_for_chapter(session, chapter_id)
        if header is None:
            raise HTTPException(404, "Chapter not found")
        per_sec = gpu_ledger.ledger_per_section(session, chapter_id)
    return {
        "header": gpu_ledger.ledger_as_dict(header),
        "sections": [gpu_ledger.ledger_as_dict(r) for r in per_sec],
    }


@app.get("/api/ledger/draft/{draft_id}")
async def api_ledger_draft(draft_id: str):
    """Phase 54.6.76 (#15) — ledger for one draft."""
    from sciknow.core import gpu_ledger
    with get_session() as session:
        row = gpu_ledger.ledger_for_draft(session, draft_id)
    if row is None:
        raise HTTPException(404, "Draft not found")
    return gpu_ledger.ledger_as_dict(row)


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

    Phase 54.6.21 — lookup the job under ``_job_lock`` to guard against
    ``_gc_old_jobs()`` evicting the entry between our scheduling and
    our first read. Without this, a job that lives longer than
    ``_JOB_GC_AGE_SECONDS`` after another finishes could KeyError on
    the first ``_jobs[job_id]`` access.
    """
    with _job_lock:
        job = _jobs.get(job_id)
        if job is None:
            logger.warning(
                "_run_generator_in_thread: job %s evicted before start", job_id
            )
            return
        queue = job["queue"]
        cancel = job["cancel"]
    gen = generator_fn()
    try:
        # Phase 54.6.318 — drain any LLM-call stats that buffered up
        # before the generator yielded its first event (the warm-up
        # call in book_ops fires before any token event reaches us).
        from sciknow.rag.llm import drain_call_stats as _drain_llm_calls
        _drained = _drain_llm_calls()
        if _drained:
            with _job_lock:
                if job_id in _jobs:
                    _jobs[job_id]["llm_calls"].extend(_drained)
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
            # Phase 54.6.318 — drain any LLM-call stats accumulated
            # since the last event. Each completed Ollama call within
            # this event appends a stats dict to the thread-local
            # buffer; folding them onto the job here keeps the
            # decode-tps display fresh in step with token events.
            _drained = _drain_llm_calls()
            if _drained:
                with _job_lock:
                    if job_id in _jobs:
                        # Cap the list at 200 so very long sessions
                        # don't grow the job dict without bound.
                        existing = _jobs[job_id].get("llm_calls") or []
                        existing.extend(_drained)
                        if len(existing) > 200:
                            existing = existing[-200:]
                        _jobs[job_id]["llm_calls"] = existing
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
async def api_gaps(model: str = Form(None), method: str = Form("")):
    """Phase 54.6.14 — optional ``method`` name from the brainstorming
    catalogue (e.g. "Reverse Brainstorming", "Five Whys", "Scope
    Boundaries") steers the LLM's gap-finding approach."""
    from sciknow.core.book_ops import run_gaps_stream
    from sciknow.core.methods import get_method, method_preamble as _mp

    job_id, queue = _create_job("gaps")
    loop = asyncio.get_event_loop()

    preamble = ""
    if method and method.strip():
        m = get_method("brainstorming", method)
        if m:
            preamble = _mp(m)

    def gen():
        return run_gaps_stream(
            book_id=_book_id, model=model or None,
            method_preamble=preamble,
        )

    thread = threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@app.post("/api/book/auto-expand/preview")
async def api_book_auto_expand_preview(
    per_gap_limit: int = Form(100),
):
    """Phase 54.6.5 — preview corpus-expansion candidates derived from
    the current book's open gaps.

    Composition: book_gaps (open, type in {topic, evidence}) → per-gap
    topic search → merged + corpus-centroid-scored candidate list. Each
    candidate carries a ``gap_ids`` list so the UI can display how many
    open gaps the paper would close.
    """
    from sciknow.core.expand_ops import find_candidates_for_book_gaps
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: find_candidates_for_book_gaps(
                _book_id, per_gap_limit=int(per_gap_limit),
                score_relevance=True,
            ),
        )
    except Exception as exc:
        logger.exception("book/auto-expand preview failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return JSONResponse(result)


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
    include_visuals: bool = Form(False),   # Phase 54.6.144
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
            include_visuals=include_visuals,
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
    include_visuals: bool = Form(False),   # Phase 54.6.144
):
    """Phase 17 — target_words is optional; when None, the effective
    per-section target is resolved from the book's custom_metadata
    (target_chapter_words / num_sections_in_chapter). When set, it
    overrides the book-level value for this run only.

    Phase 54.6.144 — ``include_visuals`` turns on the 54.6.142 autowrite
    visuals integration: the 5-signal ranker surfaces figures to the
    writer, the gated instruction ships in the system prompt, and
    ``visual_citation`` joins the scorer dimensions. Default off so the
    existing workflow is untouched."""
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
            include_visuals=include_visuals,
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


# ── Phase 54.6.14 — method catalogues ──────────────────────────────────
# Pickers in Plan / Outline / Gaps surface these so the user can steer
# the LLM's approach (Tree of Thoughts, Five Whys, Pre-mortem, etc.)
# without rewriting prompts. Data adapted from BMAD-METHOD (MIT).

@app.get("/api/methods")
async def api_methods(kind: str = "elicitation"):
    from sciknow.core.methods import list_methods
    try:
        return JSONResponse({"methods": list_methods(kind), "kind": kind})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Phase 54.6.14 — Critic Skills (adversarial review + edge-case hunter) ──

@app.post("/api/adversarial-review/{draft_id}")
async def api_adversarial_review(draft_id: str, model: str = Form(None)):
    """BMAD-inspired cynical critic pass — streams findings over SSE."""
    from sciknow.core.book_ops import adversarial_review_stream
    job_id, _queue = _create_job("adversarial_review")
    loop = asyncio.get_event_loop()

    def gen():
        return adversarial_review_stream(draft_id, model=(model or None))

    threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


@app.post("/api/edge-cases/{draft_id}")
async def api_edge_cases(draft_id: str, model: str = Form(None)):
    """Exhaustive edge-case hunter — structured JSON findings over SSE."""
    from sciknow.core.book_ops import edge_case_hunter_stream
    job_id, _queue = _create_job("edge_case_hunter")
    loop = asyncio.get_event_loop()

    def gen():
        return edge_case_hunter_stream(draft_id, model=(model or None))

    threading.Thread(
        target=_run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
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


@app.post("/api/wiki/extract-kg")
async def api_wiki_extract_kg(force: bool = Form(False)):
    """Phase 54.6.8 — backfill knowledge_graph triples for already-
    compiled wiki pages. Spawns ``sciknow wiki extract-kg`` as a
    subprocess so the stdout streams as SSE. Safe to invoke from the
    Wiki Lint tab while other jobs run.
    """
    job_id, _queue = _create_job("wiki_extract_kg")
    loop = asyncio.get_event_loop()
    argv = ["wiki", "extract-kg"]
    if force:
        argv.append("--force")
    _spawn_cli_streaming(job_id, argv, loop)
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

# Phase 54.6.135 — route was `/api/feedback` which collided with the
# Phase-54.6.115 expand-candidates ±mark endpoint (line 6072). FastAPI
# dispatched to whichever was registered first (this one), so the active
# ±mark JS calls always hit the form-based thumbs handler here and got
# 422 "score: Field required". The thumbs UI was removed long ago but
# the endpoint stayed; moved to a disambiguated path so the ±mark route
# can own `/api/feedback` again. The rename has no frontend caller
# (there never was one at the time of the fix).
@app.post("/api/feedback/thumbs")
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
# ── Phase 21.b — Visuals API ────────────────────────────────────────────────

@app.get("/api/visuals")
async def api_visuals_list(
    kind: str = None, doc_id: str = None,
    query: str = None, limit: int = 50, offset: int = 0,
    order_by: str = "default",
):
    """Phase 21.b — list/search visual elements (tables, equations, figures).

    Filters: ?kind=table|equation|figure|chart|code, ?doc_id=..., ?query=caption search.
    Pagination: ``limit`` (default 50, capped at 500) and ``offset`` (54.6.100).
    Ordering (54.6.101) via ``order_by``:

      - ``default`` / ``recent`` — newest paper first (year DESC, created_at DESC)
      - ``importance`` — composite score: year + caption length + has-figure-num
          + paper_type weight (peer_reviewed > preprint > … > unknown). Good
          default for "show me the high-signal visuals first".
      - ``paper`` — group by paper title, figures ordered within
      - ``figure_num`` — natural figure numbering ascending (1, 2, 3…)
      - ``caption_richness`` — longer AI caption = better-documented = first
      - ``random`` — random sample (uses postgres random(), non-stable)

    Response shape unchanged for back-compat — total count lives at /api/visuals/stats.
    """
    from sqlalchemy import text as _vtext
    conditions = ["1=1"]
    params: dict = {}
    if kind:
        conditions.append("v.kind = :kind")
        params["kind"] = kind
    if doc_id:
        conditions.append("v.document_id::text = :doc_id")
        params["doc_id"] = doc_id
    if query:
        conditions.append("(v.caption ILIKE :q OR v.surrounding_text ILIKE :q)")
        params["q"] = f"%{query[:100]}%"
    where = " AND ".join(conditions)
    safe_limit = max(1, min(int(limit or 50), 500))
    safe_offset = max(0, int(offset or 0))

    # 54.6.101 — ordering heuristics. Each maps to a SQL ORDER BY.
    # ``importance`` is a deterministic composite so two calls with the
    # same params return the same rows in the same order; ``random``
    # is intentionally non-stable.
    _order_clauses = {
        "default": "pm.year DESC NULLS LAST, v.created_at DESC",
        "recent": "pm.year DESC NULLS LAST, v.created_at DESC",
        "paper": "pm.title ASC NULLS LAST, v.figure_num ASC NULLS LAST, v.block_idx ASC",
        "figure_num": "v.figure_num ASC NULLS LAST, pm.title ASC NULLS LAST",
        "caption_richness": "LENGTH(COALESCE(v.ai_caption, v.caption, '')) DESC",
        "random": "random()",
        "importance": (
            # Composite score in [0, ~1.5]:
            #   0.30 * year_bonus     — newer = higher (2026 → 1.0, 2000 → 0)
            #   0.25 * caption_rich   — log-ish saturation at ~400 chars
            #   0.20 * has_fig_num    — 1 if figure_num is set
            #   0.15 * paper_type_w   — peer_reviewed 1.0 … opinion 0.4
            #   0.10 * has_ai_caption — 1 if the VLM captioner has run
            "("
            " 0.30 * GREATEST(0, LEAST(1, (COALESCE(pm.year, 2000) - 2000) / 26.0))"
            " + 0.25 * LEAST(1, LENGTH(COALESCE(v.ai_caption, v.caption, ''))::float / 400.0)"
            " + 0.20 * (CASE WHEN v.figure_num IS NOT NULL AND v.figure_num <> '' THEN 1 ELSE 0 END)"
            " + 0.15 * (CASE COALESCE(pm.paper_type, 'unknown')"
            "              WHEN 'peer_reviewed' THEN 1.0"
            "              WHEN 'preprint'      THEN 0.9"
            "              WHEN 'thesis'        THEN 0.85"
            "              WHEN 'book_chapter'  THEN 0.85"
            "              WHEN 'editorial'     THEN 0.7"
            "              WHEN 'policy'        THEN 0.7"
            "              WHEN 'opinion'       THEN 0.4"
            "              ELSE 0.6 END)"
            " + 0.10 * (CASE WHEN v.ai_caption IS NOT NULL AND v.ai_caption <> '' THEN 1 ELSE 0 END)"
            ") DESC, pm.title ASC NULLS LAST"
        ),
    }
    order_sql = _order_clauses.get((order_by or "default").lower(), _order_clauses["default"])

    try:
        with get_session() as session:
            rows = session.execute(_vtext(f"""
                SELECT v.id::text, v.document_id::text, v.kind, v.content,
                       v.caption, v.asset_path, v.figure_num, v.block_idx,
                       pm.title AS paper_title, pm.year,
                       v.ai_caption, v.ai_caption_model,
                       v.table_title, v.table_headers, v.table_summary,
                       v.table_n_rows, v.table_n_cols
                FROM visuals v
                JOIN paper_metadata pm ON pm.document_id = v.document_id
                WHERE {where}
                ORDER BY {order_sql}
                LIMIT :lim OFFSET :off
            """), {**params, "lim": safe_limit, "off": safe_offset}).fetchall()
        return JSONResponse([
            {"id": r[0], "document_id": r[1], "kind": r[2],
             "content": (r[3] or "")[:2000], "caption": r[4],
             "asset_path": r[5], "figure_num": r[6], "block_idx": r[7],
             "paper_title": r[8], "year": r[9],
             "ai_caption": r[10], "ai_caption_model": r[11],
             "table_title": r[12], "table_headers": r[13],
             "table_summary": r[14],
             "table_n_rows": r[15], "table_n_cols": r[16]}
            for r in rows
        ])
    except Exception as exc:
        return JSONResponse({"error": str(exc), "visuals": []})


@app.get("/api/visuals/stats")
async def api_visuals_stats():
    """Phase 21.b — visual element counts by kind."""
    try:
        with get_session() as session:
            rows = session.execute(text(
                "SELECT kind, COUNT(*) FROM visuals GROUP BY kind ORDER BY kind"
            )).fetchall()
        return JSONResponse({
            "stats": {r[0]: r[1] for r in rows},
            "total": sum(r[1] for r in rows),
        })
    except Exception:
        return JSONResponse({"stats": {}, "total": 0})


@app.get("/api/visuals/search")
async def api_visuals_search(q: str, kind: str = "", k: int = 10):
    """Phase 54.6.82 (#11 follow-up) — semantic search over embedded
    AI captions + equation paraphrases via the visuals Qdrant collection.

    Results come back ranked by RRF-fused dense + sparse scores on
    the caption text, then joined to the full ``visuals`` row so the
    caller gets paper_title, asset_path, etc. for display. Empty
    result list when the visuals collection hasn't been populated
    yet (`sciknow db embed-visuals`).
    """
    from sciknow.retrieval.visuals_search import search_visuals
    from sciknow.storage.qdrant import get_client as _get_qdrant
    from sqlalchemy import text as _vtext

    qdrant = _get_qdrant()
    hits = search_visuals(
        q, qdrant, candidate_k=max(5, min(k, 50)),
        kind=kind.strip() or None,
    )
    if not hits:
        return JSONResponse({"hits": [], "total": 0})

    vids = [h.visual_id for h in hits if h.visual_id]
    if not vids:
        return JSONResponse({"hits": [], "total": 0})
    with get_session() as session:
        placeholders = ", ".join(f":v{i}" for i, _ in enumerate(vids))
        params = {f"v{i}": v for i, v in enumerate(vids)}
        rows = session.execute(_vtext(f"""
            SELECT v.id::text, v.kind, v.asset_path, v.figure_num,
                   v.caption, v.ai_caption, v.content,
                   pm.title AS paper_title, pm.year
            FROM visuals v
            JOIN paper_metadata pm ON pm.document_id = v.document_id
            WHERE v.id::text IN ({placeholders})
        """), params).fetchall()
    by_id = {r[0]: r for r in rows}
    out = []
    for h in hits:
        r = by_id.get(h.visual_id)
        if not r:
            continue
        out.append({
            "id": r[0], "kind": r[1], "asset_path": r[2],
            "figure_num": r[3], "caption": r[4] or "",
            "ai_caption": r[5] or "", "content": (r[6] or "")[:1500],
            "paper_title": r[7] or "", "year": r[8],
            "rrf_score": round(h.rrf_score, 4),
        })
    return JSONResponse({"hits": out, "total": len(out)})


def _visuals_rank_for_draft(draft_id: str, limit: int) -> dict:
    """Shared ranker path for GET(compute=1) / POST. Runs ``rank_visuals``
    against the draft's prose and returns a dict ready to persist in
    ``drafts.custom_metadata['visual_suggestions']`` or hand back to the
    client. The caller is responsible for persisting on write paths."""
    limit = max(1, min(int(limit), 30))
    if draft_id == BIBLIOGRAPHY_PSEUDO_ID:
        return {
            "draft_id": draft_id,
            "hits": [],
            "note": "The bibliography has no prose to match visuals against.",
        }

    from sciknow.retrieval.visuals_ranker import rank_visuals

    with get_session() as session:
        row = session.execute(text("""
            SELECT d.content, d.section_type, d.sources
            FROM drafts d
            WHERE d.id::text = :did
        """), {"did": draft_id}).fetchone()
    if not row:
        raise HTTPException(404, "Draft not found")
    content, section_type, sources_raw = row
    content = (content or "").strip()
    if not content:
        return {
            "draft_id": draft_id,
            "hits": [],
            "note": "This draft has no prose yet; write something first.",
        }

    cited_doc_ids: list[str] = []
    try:
        if sources_raw:
            raw = sources_raw if isinstance(sources_raw, list) else (
                json.loads(sources_raw) if isinstance(sources_raw, str) else []
            )
            import re as _re
            titles = []
            for s in raw or []:
                m = _re.search(r"\(\d{4}(?:-\d{2}-\d{2})?\)\.\s+([^.]+)\.", s or "")
                if m:
                    titles.append(m.group(1).strip())
            if titles:
                with get_session() as session:
                    placeholders = ", ".join(f":t{i}" for i in range(len(titles)))
                    params = {f"t{i}": t for i, t in enumerate(titles)}
                    rows = session.execute(text(f"""
                        SELECT document_id::text
                        FROM paper_metadata
                        WHERE title IN ({placeholders})
                    """), params).fetchall()
                    cited_doc_ids = [r[0] for r in rows if r[0]]
    except Exception as exc:
        logger.debug("cited-doc resolution failed: %s", exc)

    sentence = content[:2500]
    try:
        ranked = rank_visuals(
            sentence,
            cited_doc_ids=cited_doc_ids,
            section_type=section_type,
            candidate_k=max(limit * 3, 15),
            top_k=limit,
        )
    except Exception as exc:
        logger.warning("rank_visuals failed: %s", exc)
        return {
            "draft_id": draft_id, "hits": [],
            "note": f"Ranker error: {exc}",
        }

    hits = []
    for rv in ranked:
        cap = (rv.ai_caption or "").strip() or "(no caption)"
        img_url = None
        if rv.kind in ("figure", "chart", "table"):
            img_url = f"/api/visuals/image/{rv.visual_id}"
        hits.append({
            "visual_id": rv.visual_id,
            "document_id": rv.document_id,
            "kind": rv.kind,
            "figure_num": rv.figure_num,
            "caption": cap[:400],
            "paper_title": rv.paper_title,
            "image_url": img_url,
            "composite_score": rv.composite_score,
            "same_paper": rv.same_paper,
        })

    return {"draft_id": draft_id, "hits": hits}


def _visuals_get_cached(draft_id: str) -> dict | None:
    """Return the persisted visual-suggestion payload from
    drafts.custom_metadata['visual_suggestions'], or None if none. The
    blob includes both the ranked hits and the content-hash signature
    the ranking was computed against so the UI can flag staleness."""
    if draft_id == BIBLIOGRAPHY_PSEUDO_ID:
        return None
    with get_session() as session:
        row = session.execute(text("""
            SELECT d.custom_metadata, d.content
            FROM drafts d WHERE d.id::text = :did
        """), {"did": draft_id}).fetchone()
    if not row:
        return None
    meta, content = row
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    blob = (meta or {}).get("visual_suggestions") if isinstance(meta, dict) else None
    if not isinstance(blob, dict):
        return None
    import hashlib as _h
    cur_hash = _h.md5((content or "").encode("utf-8", errors="ignore")).hexdigest()
    blob = dict(blob)
    blob["stale"] = bool(blob.get("content_hash") and blob["content_hash"] != cur_hash)
    return blob


@app.get("/api/visuals/suggestions")
async def api_visuals_suggestions(
    draft_id: str,
    limit: int = 10,
    compute: int = 0,
):
    """Visual suggestions for the right-panel "Visuals" tab.

    Behavior (Phase 54.6.312): the GET returns the cached payload by
    default — **does not** run the ranker unless ``compute=1`` is
    explicitly passed. This matches the UX of a manual "Rank" button:
    the ranker is ~1–2s and the user shouldn't pay that cost on every
    draft navigation. POST to this same endpoint to force a rank + save.
    """
    cached = _visuals_get_cached(draft_id)
    if cached is not None and not int(compute or 0):
        return JSONResponse({
            "draft_id": draft_id,
            "hits": cached.get("hits") or [],
            "cached": True,
            "stale": bool(cached.get("stale")),
            "ranked_at": cached.get("ranked_at"),
            "note": cached.get("note"),
        })

    if not int(compute or 0):
        return JSONResponse({
            "draft_id": draft_id, "hits": [], "cached": False,
            "note": "No ranking saved for this draft. Click Rank to compute.",
        })

    payload = _visuals_rank_for_draft(draft_id, limit=limit)
    payload["cached"] = False
    return JSONResponse(payload)


@app.post("/api/visuals/suggestions")
async def api_visuals_suggestions_rank(
    draft_id: str,
    limit: int = 10,
):
    """Compute the visual ranking for a draft and persist it into
    ``drafts.custom_metadata['visual_suggestions']`` so subsequent opens
    of the Visuals panel serve the saved result instead of re-ranking."""
    payload = _visuals_rank_for_draft(draft_id, limit=limit)

    if draft_id != BIBLIOGRAPHY_PSEUDO_ID and payload.get("hits") is not None:
        from datetime import datetime as _dt
        import hashlib as _h
        with get_session() as session:
            row = session.execute(text(
                "SELECT content FROM drafts WHERE id::text = :did"
            ), {"did": draft_id}).fetchone()
            cur_hash = _h.md5(((row and row[0]) or "").encode(
                "utf-8", errors="ignore")).hexdigest() if row else ""
            blob = {
                "hits": payload.get("hits") or [],
                "ranked_at": _dt.utcnow().isoformat() + "Z",
                "content_hash": cur_hash,
                "note": payload.get("note"),
            }
            session.execute(text("""
                UPDATE drafts
                   SET custom_metadata = COALESCE(custom_metadata, '{}'::jsonb)
                                         || jsonb_build_object(
                                              'visual_suggestions',
                                              CAST(:blob AS jsonb))
                 WHERE id::text = :did
            """), {"did": draft_id, "blob": json.dumps(blob)})
            session.commit()
        payload["ranked_at"] = blob["ranked_at"]

    payload["cached"] = False
    return JSONResponse(payload)


@app.delete("/api/visuals/suggestions")
async def api_visuals_suggestions_clear(draft_id: str):
    """Drop the persisted visual-suggestions cache for a draft so the
    next Rank-button press recomputes from scratch. Useful for a
    "Clear saved ranking" control."""
    if draft_id == BIBLIOGRAPHY_PSEUDO_ID:
        return JSONResponse({"ok": True, "cleared": False})
    with get_session() as session:
        session.execute(text("""
            UPDATE drafts
               SET custom_metadata = custom_metadata - 'visual_suggestions'
             WHERE id::text = :did
        """), {"did": draft_id})
        session.commit()
    return JSONResponse({"ok": True, "cleared": True})


@app.get("/debug/equations")
async def debug_equations(limit: int = 60):
    """Phase 54.6.107 — self-contained equation-render diagnostic.

    Loads N random equations from the corpus and renders each with
    KaTeX directly in the browser. Emits a summary counter (rendered
    ok / KaTeX-error / empty) so we can diagnose why the user sees
    half the equations failing without needing headless automation.
    """
    from sqlalchemy import text as _dt
    from starlette.responses import HTMLResponse
    try:
        with get_session() as session:
            rows = session.execute(_dt("""
                SELECT id::text, content
                FROM visuals
                WHERE kind='equation' AND content IS NOT NULL AND content <> ''
                ORDER BY random()
                LIMIT :lim
            """), {"lim": max(1, min(limit, 300))}).fetchall()
    except Exception as exc:
        return HTMLResponse(f"<pre>DB error: {exc}</pre>", status_code=500)

    import json as _j
    eqs = [{"id": r[0], "content": r[1]} for r in rows]
    body = f"""<!doctype html>
<html><head><title>sciknow equation diagnostic</title>
<link rel="stylesheet" href="/static/vendor/katex/katex.min.css"/>
<script defer src="/static/vendor/katex/katex.min.js"></script>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #f8f8f8; }}
h1 {{ font-size: 18px; margin-bottom: 8px; }}
#stats {{ position: sticky; top: 0; background: #fff; padding: 10px 14px;
         border: 1px solid #ccc; border-radius: 6px; margin-bottom: 14px;
         font-size: 13px; z-index: 10; }}
.eq-card {{ background: #fff; border: 1px solid #ddd; border-radius: 6px;
            padding: 12px 14px; margin-bottom: 10px; }}
.eq-card.fail {{ border-color: #e57373; background: #fff5f5; }}
.eq-head {{ font: 11px/1 monospace; color: #888; margin-bottom: 6px; }}
.eq-render {{ font-size: 17px; min-height: 30px; overflow-x: auto; }}
.eq-latex {{ font: 11px/1.4 monospace; color: #555; margin-top: 6px;
             white-space: pre-wrap; word-break: break-word;
             padding: 4px 6px; background: #fafafa; border-radius: 3px; }}
.eq-err {{ font: 11px monospace; color: #c00; margin-top: 4px; }}
</style>
</head><body>
<h1>Equation rendering diagnostic — {len(eqs)} random equations</h1>
<div id="stats">Loading KaTeX…</div>
<div id="out"></div>
<script>
const EQS = {_j.dumps(eqs)};
function strip(src) {{
  let b = (src || '').trim();
  b = b.replace(/^\\s*\\$\\$\\s*/, '').replace(/\\s*\\$\\$\\s*$/, '');
  b = b.replace(/^\\s*\\$\\s*/, '').replace(/\\s*\\$\\s*$/, '');
  b = b.replace(/^\\s*\\\\\\[\\s*/, '').replace(/\\s*\\\\\\]\\s*$/, '');
  b = b.replace(/^\\s*\\\\\\(\\s*/, '').replace(/\\s*\\\\\\)\\s*$/, '');
  return b.trim();
}}
function esc(s) {{
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}
function run() {{
  if (typeof window.katex === 'undefined') {{ setTimeout(run, 200); return; }}
  const out = document.getElementById('out');
  const stats = document.getElementById('stats');
  let ok = 0, fail = 0, failMsgs = new Map();
  let html = '';
  for (const eq of EQS) {{
    const body = strip(eq.content);
    let rendered = '', errMsg = '', failed = false;
    try {{
      rendered = window.katex.renderToString(body, {{
        displayMode: true, throwOnError: true, strict: false, output: 'html'
      }});
      ok++;
    }} catch (e) {{
      failed = true;
      fail++;
      errMsg = (e.message || String(e)).split('\\n')[0];
      const key = errMsg.replace(/pos \\d+/g, 'pos N').slice(0, 70);
      failMsgs.set(key, (failMsgs.get(key) || 0) + 1);
      try {{
        rendered = window.katex.renderToString(body, {{
          displayMode: true, throwOnError: false, strict: 'ignore', output: 'html'
        }});
      }} catch (_) {{
        rendered = '<em>KaTeX refused even in lenient mode</em>';
      }}
    }}
    html += '<div class="eq-card' + (failed ? ' fail' : '') + '">'
      + '<div class="eq-head">' + esc(eq.id.slice(0, 8)) + (failed ? ' — FAIL' : ' — ok') + '</div>'
      + '<div class="eq-render">' + rendered + '</div>'
      + (failed ? '<div class="eq-err">' + esc(errMsg) + '</div>' : '')
      + '<div class="eq-latex">' + esc(body) + '</div>'
      + '</div>';
  }}
  out.innerHTML = html;
  const sorted = [...failMsgs.entries()].sort((a, b) => b[1] - a[1]);
  let sHtml = '<strong>' + ok + '/' + (ok + fail) + ' rendered (' + Math.round(100 * ok / (ok + fail)) + '%).</strong>';
  if (fail > 0) {{
    sHtml += ' <strong>' + fail + ' failed.</strong> Error buckets:<ul style="margin:4px 0 0 18px;">';
    for (const [msg, count] of sorted) {{
      sHtml += '<li>' + count + '× ' + esc(msg) + '</li>';
    }}
    sHtml += '</ul>';
  }}
  stats.innerHTML = sHtml;
}}
run();
</script>
</body></html>
"""
    return HTMLResponse(body)


@app.get("/api/visuals/image/{visual_id}")
async def api_visuals_image(visual_id: str):
    """Phase 54.6.61 — stream a figure's JPG back to the browser.

    MinerU writes figure images to a per-doc subtree under
    ``data/mineru_output/<doc_id>/<doc_slug>/auto/images/<sha>.jpg``,
    with `asset_path` in the DB being the path relative to the inner
    `auto/` dir (e.g. ``images/<sha>.jpg``). We resolve against the
    active project's data_dir rather than mounting the directory as
    static, so:

      * multi-project setups route correctly (the mount would pin
        whichever project.data_dir was resolved at import time)
      * there's no path-traversal surface: the client passes a UUID,
        never a path. We look up the row, then constrain the filesystem
        read to the known mineru_output subtree.
    """
    from pathlib import Path as _P
    from starlette.responses import FileResponse as _FileResponse
    from sqlalchemy import text as _vtext
    from sciknow.config import settings as _settings

    try:
        with get_session() as session:
            row = session.execute(_vtext(
                "SELECT document_id::text, asset_path, kind "
                "FROM visuals WHERE id::text = :vid LIMIT 1"
            ), {"vid": visual_id}).fetchone()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB lookup failed: {exc}")
    if not row:
        raise HTTPException(status_code=404, detail="visual not found")
    doc_id, asset_path, kind = row
    if not asset_path:
        raise HTTPException(status_code=404, detail="visual has no asset_path")
    if kind not in ("figure", "chart"):
        # Phase 54.6.62 — `chart` also ships as a JPG (MinerU classifies
        # plots/bar charts/etc. as chart blocks, same img_path shape as
        # image blocks). Equations/tables/code stay text-only — reject
        # early so a confused caller gets a clear error.
        raise HTTPException(status_code=400,
                            detail=f"visual kind={kind!r} has no image asset")

    doc_dir = _P(_settings.data_dir) / "mineru_output" / str(doc_id)
    if not doc_dir.is_dir():
        raise HTTPException(status_code=404,
                            detail=f"mineru_output dir missing for {doc_id}")
    # The per-doc subfolder is the doc's MinerU slug (arbitrary title string).
    # Usually exactly one; we iterate in case a re-ingest left more than one.
    #
    # MinerU backends diverge on the inner folder name:
    #   * pipeline mode  → `<slug>/auto/images/<sha>.jpg`
    #   * VLM-Pro (2.5)  → `<slug>/vlm/images/<sha>.jpg`
    #   * older outputs  → `<slug>/images/<sha>.jpg` (no infix)
    # Try them in that order; break on first hit.
    candidates = []
    for sub in doc_dir.iterdir():
        if not sub.is_dir():
            continue
        for infix in ("auto", "vlm", None):
            probe = (sub / infix / asset_path) if infix else (sub / asset_path)
            probe = probe.resolve()
            if probe.is_file() and doc_dir.resolve() in probe.parents:
                candidates.append(probe)
                break
        if candidates:
            break
    if not candidates:
        raise HTTPException(status_code=404,
                            detail=f"image file missing: {asset_path}")

    # Pick by extension; MinerU writes .jpg but be permissive.
    target = candidates[0]
    ext = target.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }.get(ext, "application/octet-stream")
    return _FileResponse(
        str(target), media_type=media_type,
        # Browser can cache — the asset SHA is immutable for a given
        # extracted figure, so a week of cache is safe.
        headers={"Cache-Control": "public, max-age=604800, immutable"},
    )


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
        # Phase 54.6.57 — force unbuffered stdout. Python block-buffers
        # when stdout is a pipe (our case), which can hold back 4–8 KB
        # of log lines before a visible flush — makes the web log pane
        # look frozen during multi-second sub-steps of enrich / expand
        # / cleanup. `PYTHONUNBUFFERED=1` makes stdout line-flushed so
        # each `console.print` arrives as an SSE event immediately.
        env.setdefault("PYTHONUNBUFFERED", "1")
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

        # Phase 54.6.21 — same _jobs race guard as
        # _run_generator_in_thread. Cheap lock-protected get; if the
        # job was GC'd between spawn and start, kill the subprocess
        # and bail rather than KeyError.
        with _job_lock:
            job = _jobs.get(job_id)
        if job is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                pass
            yield {"type": "error",
                   "message": f"job {job_id} evicted before subprocess start"}
            return
        cancel = job["cancel"]
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if cancel.is_set():
                    break
                line = line.rstrip("\n")
                if line:
                    yield {"type": "log", "text": line}
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
            return
        finally:
            # Phase 54.6.21 — proc.wait moved INTO finally so a mid-
            # stream exception still reaps the subprocess instead of
            # leaving it as a zombie. Two paths:
            #   - Process still alive (cancel break or exception):
            #     terminate then wait.
            #   - Process already exited (clean stdout EOF): just
            #     wait to collect the return code.
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            else:
                try:
                    proc.wait(timeout=5)
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
    author_ids: str = Form(""),
):
    """Phase 54.6.1 — preview candidates without downloading.

    Runs search + corpus-dedup + relevance scoring, returns JSON. The UI
    renders a checkboxed list so the user can cherry-pick which DOIs to
    download via ``/api/corpus/expand-author/download-selected`` — the
    existing ``POST /api/corpus/expand-author`` still exists for the
    "auto-download by relevance threshold" override path.

    Phase 54.6.49 — ``author_ids`` (comma-separated OpenAlex short
    IDs like "A5079882695,A5033301096") bypasses name resolution and
    targets exactly those canonical authors. Used by the multi-select
    disambiguation banner: when OpenAlex returns the same person under
    multiple name variants, the user ticks the ones that are actually
    them and re-queries with all of them merged.

    May take 10-30s due to external API calls. Blocking — not SSE.
    """
    # Parse author_ids — accept comma/pipe/space separated
    id_list = [x.strip() for x in re.split(r"[,|\s]+", author_ids) if x.strip()] if author_ids else None

    if not name.strip() and not orcid.strip() and not id_list:
        raise HTTPException(
            status_code=400,
            detail="provide either author name, ORCID, or selected author IDs",
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
                author_ids=id_list,
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


@app.post("/api/corpus/agentic/preview")
async def api_corpus_agentic_preview(
    question: str = Form(""),
    budget: int = Form(10),
    threshold: int = Form(3),
    model: str = Form(""),
):
    """Phase 54.6.132 — Agentic preview, single round. Decomposes the
    question, measures coverage, identifies gap sub-topics, gathers
    candidates per gap (OpenAlex topic search via
    ``find_topic_candidates``) and returns the merged shortlist
    annotated with each candidate's source sub-topic for cherry-pick
    in the candidates modal. The user picks + downloads via the
    existing ``/api/corpus/expand-author/download-selected`` route;
    re-calling this endpoint advances to the next round (coverage is
    re-measured against the now-updated corpus)."""
    from sciknow.ingestion.agentic_expand import run_preview_round

    q = (question or "").strip()
    if not q:
        return JSONResponse({"error": "question is required"}, status_code=400)
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_preview_round(
                q, budget_per_gap=budget,
                doc_threshold=threshold,
                model=(model.strip() or None),
            ),
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)[:500]}, status_code=500)
    return JSONResponse(result)


@app.get("/api/corpus/authors/top")
async def api_corpus_authors_top(limit: int = 40):
    """Phase 54.6.309 — list the top corpus authors by paper count, so
    the expand-author-refs picker doesn't have to spawn the CLI just to
    build its dropdown. Surname collisions are aggregated client-side.
    """
    lim = max(5, min(int(limit), 200))
    with get_session() as session:
        rows = session.execute(text("""
            SELECT auth->>'name' AS author_name, COUNT(*) AS n_papers
              FROM paper_metadata,
                   jsonb_array_elements(COALESCE(authors, '[]'::jsonb)) AS auth
             WHERE auth ? 'name'
             GROUP BY author_name
             ORDER BY n_papers DESC
             LIMIT :lim
        """), {"lim": lim}).fetchall()
    return JSONResponse({"authors": [
        {"name": r[0], "n_papers": int(r[1] or 0)} for r in rows
    ]})


@app.post("/api/corpus/expand-author-refs/preview")
async def api_corpus_expand_author_refs_preview(
    author: str = Form(...),
    min_mentions: int = Form(1),
    limit: int = Form(0),
    include_in_corpus: bool = Form(False),
    relevance_query: str = Form(""),
):
    """Phase 54.6.309 — preview the references cited by ``author``'s corpus
    works. Same output shape as the existing expand-author preview so
    the GUI's cherry-pick candidates modal renders it without changes.

    The response sorts by the in-corpus mention count descending; ties
    broken by cited_year descending. Each candidate carries an
    ``author_mentions`` count and ``self_cite_count`` so the UI can
    surface "cited 5× (2 self)" badges.
    """
    import re as _re
    from sciknow.ingestion.references import normalise_title_for_dedup

    author = (author or "").strip()
    if not author:
        raise HTTPException(400, "author required")

    def _surname(name: str) -> str:
        parts = [p for p in _re.split(r"[,;\s]+", (name or "").strip()) if p]
        if not parts:
            return ""
        last = parts[-1] if "," not in (name or "") else parts[0]
        return _re.sub(r"[^A-Za-zÀ-ſ]", "", last).lower()

    surname = _surname(author)
    if not surname:
        raise HTTPException(400, "could not derive a surname")

    with get_session() as session:
        paper_rows = session.execute(text("""
            SELECT pm.document_id::text, pm.title, pm.year, pm.doi
              FROM paper_metadata pm,
                   jsonb_array_elements(COALESCE(pm.authors, '[]'::jsonb)) AS auth
             WHERE auth ? 'name'
               AND LOWER(regexp_replace(
                     split_part(auth->>'name', ' ',
                       cardinality(regexp_split_to_array(auth->>'name', '\s+'))),
                     '[^A-Za-zÀ-ſ]', '', 'g'
                   )) = :sn
        """), {"sn": surname}).fetchall()

    if not paper_rows:
        return JSONResponse({
            "author": author, "surname": surname,
            "n_author_papers": 0, "candidates": [],
            "note": f"No corpus papers found for surname '{surname}'.",
        })

    paper_ids = [r[0] for r in paper_rows]
    with get_session() as session:
        cite_rows = session.execute(text("""
            SELECT cited_doi, cited_title, cited_authors, cited_year,
                   is_self_cite
              FROM citations
             WHERE citing_document_id = ANY(CAST(:ids AS uuid[]))
        """), {"ids": paper_ids}).fetchall()

    if not cite_rows:
        return JSONResponse({
            "author": author, "surname": surname,
            "n_author_papers": len(paper_rows), "candidates": [],
            "note": "No citations for those papers. Run `sciknow db link-citations` first.",
        })

    agg: dict[str, dict] = {}
    for r in cite_rows:
        cited_doi, cited_title, cited_authors, cited_year, is_self = r
        key = (cited_doi or "").lower().strip()
        if not key:
            key = "title:" + normalise_title_for_dedup(cited_title or "")
        if not key or key == "title:":
            continue
        a = agg.setdefault(key, {
            "doi": cited_doi or None, "title": cited_title or None,
            "authors": list(cited_authors or []), "year": cited_year,
            "mentions": 0, "self_cites": 0,
        })
        a["mentions"] += 1
        if is_self:
            a["self_cites"] += 1
        if not a["title"] and cited_title:
            a["title"] = cited_title
        if not a["year"] and cited_year:
            a["year"] = cited_year

    shortlist = [v for v in agg.values() if v["mentions"] >= min_mentions]
    shortlist.sort(key=lambda v: (-v["mentions"], -(v["year"] or 0)))

    dropped_in_corpus = 0
    if not include_in_corpus:
        with get_session() as session:
            ex = session.execute(text(
                "SELECT LOWER(doi) FROM paper_metadata WHERE doi IS NOT NULL"
            )).fetchall()
            existing = {r[0] for r in ex}
        before = len(shortlist)
        shortlist = [
            v for v in shortlist
            if not (v["doi"] and v["doi"].lower() in existing)
        ]
        dropped_in_corpus = before - len(shortlist)

    if limit and limit > 0:
        shortlist = shortlist[:limit]

    # Phase 54.6.315 — resolve title/authors/year via Crossref for
    # candidates whose `citations` row stored only a DOI. This was
    # the missing field the user was seeing as "(untitled)" rows in
    # the cherry-pick modal: reference extraction for some citing
    # papers (notably V.V. Zharkova's) captured DOIs but never
    # resolved the downstream metadata, and the preview had no
    # fallback. Parallel HTTP with a shared rate limiter keeps this
    # well inside Crossref's 50 RPS polite pool.
    needs_resolve = [v for v in shortlist
                     if v.get("doi") and (not v.get("title")
                                           or len((v.get("title") or "").strip()) < 5)]
    if needs_resolve:
        import concurrent.futures as _cf
        import html as _html
        import httpx as _httpx
        from sciknow.config import settings as _settings

        def _fetch(doi_in: str) -> dict | None:
            # Phase 54.6.315b — three real-world DOI gotchas:
            # 1. AMS-style DOIs stored HTML-encoded in the DB
            #    (`…079&lt;0061:apgtwa&gt;2.0.co;2`) — unescape first.
            # 2. Some older Springer/Nature DOIs (10.1007/s005850050897)
            #    resolve via 301 redirect, not direct 200 — opt into
            #    follow_redirects so httpx chases the redirect.
            # 3. JATS XML tags in Crossref titles & abstracts; strip
            #    them out before returning.
            import time as _time
            doi = _html.unescape((doi_in or "").strip())
            if not doi:
                return None
            url = f"https://api.crossref.org/works/{doi}"
            headers = {"User-Agent": f"sciknow/0.1 (mailto:{_settings.crossref_email})"}
            # Retry once on transient failure. Crossref's polite pool
            # occasionally 5xx's or socket-drops under parallel load;
            # a single 1 s backoff retry rescues those rows.
            for attempt in range(2):
                try:
                    with _httpx.Client(timeout=20, follow_redirects=True) as client:
                        r = client.get(url, headers=headers)
                    if r.status_code != 200:
                        if attempt == 0 and r.status_code in (429, 500, 502, 503, 504):
                            _time.sleep(1.0)
                            continue
                        return None
                    d = (r.json() or {}).get("message") or {}
                    title_list = d.get("title") or []
                    raw_title = title_list[0] if title_list else ""
                    clean_title = re.sub(r"<[^>]+>", "", raw_title).strip()
                    authors = d.get("author") or []
                    issued = (d.get("issued") or {}).get("date-parts") or []
                    year = int(issued[0][0]) if issued and issued[0] else None
                    return {
                        "title": clean_title,
                        "authors": [f"{(a.get('given') or '').strip()} {(a.get('family') or '').strip()}".strip()
                                    for a in authors],
                        "year": year,
                    }
                except Exception:
                    if attempt == 0:
                        _time.sleep(1.0)
                        continue
                    return None
            return None

        # Larger timeout window — one slow request should never starve
        # the whole preview. 8 workers × 20 s per request × 63 rows is
        # still ~16 s wall-clock worst case with good parallelism.
        with _cf.ThreadPoolExecutor(max_workers=8) as pool:
            fut_map = {pool.submit(_fetch, v["doi"]): v for v in needs_resolve}
            try:
                for fut in _cf.as_completed(fut_map, timeout=180):
                    v = fut_map[fut]
                    try:
                        res = fut.result()
                    except Exception:
                        res = None
                    if not res:
                        continue
                    if not v.get("title") and res.get("title"):
                        v["title"] = res["title"]
                    if not v.get("year") and res.get("year"):
                        v["year"] = res["year"]
                    if (not v.get("authors") or v.get("authors") == []) and res.get("authors"):
                        v["authors"] = res["authors"]
            except _cf.TimeoutError:
                logger.warning(
                    "expand-author-refs resolve timed out after 180s; "
                    "%d/%d candidates may still be (untitled)",
                    sum(1 for f in fut_map if not f.done()),
                    len(fut_map),
                )

    # Relevance score. Phase 54.6.315 — auto-compute against the
    # corpus centroid even when relevance_query is empty, so the
    # cherry-pick modal always shows a meaningful Score column (the
    # sister expand-author preview already does this via score_relevance=True).
    relevance_scores: list[float] = []
    if shortlist:
        try:
            from sciknow.retrieval.relevance import (
                compute_corpus_centroid, embed_query, score_candidates,
            )
            anchor = (
                embed_query(relevance_query.strip())
                if relevance_query.strip()
                else compute_corpus_centroid()
            )
            titles = [v["title"] or "" for v in shortlist]
            relevance_scores = list(score_candidates(titles, anchor))
        except Exception as exc:
            logger.warning("expand-author-refs relevance scoring failed: %s", exc)
            relevance_scores = []

    # Shape matches expand-author preview so the existing download-
    # selected endpoint and cherry-pick modal can consume it unchanged.
    candidates_out = []
    for i, v in enumerate(shortlist):
        rscore = relevance_scores[i] if i < len(relevance_scores) else None
        candidates_out.append({
            "doi": v["doi"],
            "title": v["title"] or "",
            "year": v["year"],
            "authors": [a if isinstance(a, str) else str(a) for a in (v["authors"] or [])],
            "author_mentions": v["mentions"],
            "self_cite_count": v["self_cites"],
            "relevance_score": rscore,
        })

    return JSONResponse({
        "author": author,
        "surname": surname,
        "n_author_papers": len(paper_rows),
        "n_unique_references": len(shortlist),
        "dropped_in_corpus": dropped_in_corpus,
        "candidates": candidates_out,
    })


@app.post("/api/corpus/expand-oeuvre/preview")
async def api_corpus_expand_oeuvre_preview(
    min_corpus_papers: int = Form(3),
    per_author_limit: int = Form(10),
    max_authors: int = Form(10),
    relevance_query: str = Form(""),
    strict_author: bool = Form(True),
):
    """Phase 54.6.131 — Oeuvre preview. Enumerates qualifying authors
    + per-author candidates without downloading; returns the merged
    shortlist annotated with the source author for cherry-pick in the
    candidates modal. Reuses ``find_oeuvre_candidates`` so the GUI
    plan matches the CLI plan exactly."""
    from sciknow.core.expand_ops import find_oeuvre_candidates

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: find_oeuvre_candidates(
                min_corpus_papers=min_corpus_papers,
                per_author_limit=per_author_limit,
                max_authors=max_authors,
                relevance_query=relevance_query.strip(),
                strict_author=strict_author,
                score_relevance=True,
            ),
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)[:500]}, status_code=500)
    return JSONResponse(result)


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

    # Sanitize to {doi, title, year, alternate_dois, alternate_arxiv_ids},
    # drop entries missing DOI. Phase 54.6.51 — alternates are used by
    # the downloader as fallback sources when the primary DOI's OA
    # discovery returns nothing (preprint-vs-journal duplicates).
    clean: list[dict] = []
    for c in raw_cands:
        if not isinstance(c, dict):
            continue
        doi = (c.get("doi") or "").strip()
        if not doi:
            continue
        alt_dois = [d for d in (c.get("alternate_dois") or []) if isinstance(d, str) and d.strip()]
        alt_arxiv = [a for a in (c.get("alternate_arxiv_ids") or []) if isinstance(a, str) and a.strip()]
        clean.append({
            "doi": doi,
            "title": (c.get("title") or "")[:500],
            "year": c.get("year") if isinstance(c.get("year"), int) else None,
            "alternate_dois": alt_dois[:10],
            "alternate_arxiv_ids": alt_arxiv[:10],
        })
    if not clean:
        raise HTTPException(status_code=400, detail="no valid DOIs in candidates")

    workers = int(body.get("workers") or 0)
    ingest = bool(body.get("ingest", True))
    # Phase 54.6.52 — retry_failed bypasses .no_oa_cache + .ingest_failed
    # for the current batch. Used by the GUI's "Retry previously-failed"
    # checkbox when the user wants to re-probe cached DOIs (e.g. after
    # new OA sources like HAL/Zenodo land in Phase 54.6.51).
    retry_failed = bool(body.get("retry_failed", False))

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
    if retry_failed:
        argv.append("--retry-failed")

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


@app.post("/api/corpus/expand-cites/preview")
async def api_corpus_expand_cites_preview(
    per_seed_cap: int = Form(50),
    total_limit: int = Form(500),
    relevance_query: str = Form(""),
):
    """Phase 54.6.4 — preview inbound-citation candidates (blocking JSON)."""
    from sciknow.core.expand_ops import find_inbound_citation_candidates
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: find_inbound_citation_candidates(
                per_seed_cap=int(per_seed_cap), total_limit=int(total_limit),
                relevance_query=relevance_query, score_relevance=True,
            ),
        )
    except Exception as exc:
        logger.exception("expand-cites preview failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return JSONResponse(result)


@app.post("/api/corpus/expand-topic/preview")
async def api_corpus_expand_topic_preview(
    query: str = Form(""),
    limit: int = Form(500),
    relevance_query: str = Form(""),
):
    """Phase 54.6.4 — preview topic-search candidates."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="query required")
    from sciknow.core.expand_ops import find_topic_candidates
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: find_topic_candidates(
                query, limit=int(limit),
                relevance_query=relevance_query, score_relevance=True,
            ),
        )
    except Exception as exc:
        logger.exception("expand-topic preview failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return JSONResponse(result)


@app.post("/api/corpus/expand-coauthors/preview")
async def api_corpus_expand_coauthors_preview(
    depth: int = Form(1),
    per_author_cap: int = Form(10),
    total_limit: int = Form(500),
    relevance_query: str = Form(""),
):
    """Phase 54.6.4 — preview coauthor-snowball candidates."""
    from sciknow.core.expand_ops import find_coauthor_candidates
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: find_coauthor_candidates(
                depth=int(depth), per_author_cap=int(per_author_cap),
                total_limit=int(total_limit),
                relevance_query=relevance_query, score_relevance=True,
            ),
        )
    except Exception as exc:
        logger.exception("expand-coauthors preview failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return JSONResponse(result)


@app.get("/api/pending-downloads")
async def api_pending_downloads_list(
    status: str = "pending", source: str = "", limit: int = 500,
):
    """Phase 54.6.7 — list rows from the pending_downloads table."""
    from sciknow.core.pending_ops import list_pending
    rows = list_pending(
        status=(status or None),
        source_method=(source.strip() or None),
        limit=int(limit),
    )
    return JSONResponse({"rows": rows, "count": len(rows)})


@app.post("/api/pending-downloads/update")
async def api_pending_downloads_update(request: Request):
    """Update one row's status (manual_acquired / abandoned / pending).

    Body: ``{"doi": "...", "status": "...", "notes": "..."}``
    """
    from sciknow.core.pending_ops import update_status
    body = await request.json()
    doi = (body.get("doi") or "").strip()
    st = (body.get("status") or "").strip()
    notes = body.get("notes")
    if not doi or st not in ("pending", "manual_acquired", "abandoned"):
        raise HTTPException(status_code=400, detail="doi + valid status required")
    ok = update_status(doi, status=st, notes=notes)
    return JSONResponse({"ok": ok, "updated": ok})


@app.post("/api/pending-downloads/remove")
async def api_pending_downloads_remove(request: Request):
    """Delete a pending row outright. Body: ``{"doi": "..."}``."""
    from sciknow.core.pending_ops import remove as _remove
    body = await request.json()
    doi = (body.get("doi") or "").strip()
    if not doi:
        raise HTTPException(status_code=400, detail="doi required")
    return JSONResponse({"ok": _remove(doi)})


@app.post("/api/pending-downloads/retry")
async def api_pending_downloads_retry(request: Request):
    """Retry a set of pending DOIs by spawning `db download-dois
    --retry-failed --dois-file <tmp.json>` and streaming SSE.

    Body: ``{"dois": [list], "workers": int, "ingest": bool}``. If ``dois``
    is empty, retries ALL currently-pending rows.
    """
    import json as _json
    import tempfile
    import uuid
    from sciknow.core.pending_ops import list_pending

    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    requested = body.get("dois") or []
    if not requested:
        rows = list_pending(status="pending", limit=10000)
        requested = [{"doi": r["doi"], "title": r["title"],
                      "year": r["year"]} for r in rows]
    else:
        # Sanitize: expect list of strings OR list of {doi,title,year}.
        clean = []
        for item in requested:
            if isinstance(item, str):
                clean.append({"doi": item.strip()})
            elif isinstance(item, dict) and (item.get("doi") or "").strip():
                clean.append({
                    "doi": item["doi"].strip(),
                    "title": (item.get("title") or "")[:500],
                    "year": item.get("year") if isinstance(item.get("year"), int) else None,
                })
        requested = clean
    if not requested:
        raise HTTPException(status_code=400, detail="no DOIs to retry")

    workers = int(body.get("workers") or 0)
    ingest = bool(body.get("ingest", True))

    tmp_dir = Path(tempfile.gettempdir()) / "sciknow-pending-retry"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"retry-{uuid.uuid4().hex[:12]}.json"
    tmp_path.write_text(_json.dumps(requested))

    job_id, _queue = _create_job("pending_retry")
    loop = asyncio.get_event_loop()
    argv = [
        "db", "download-dois",
        "--dois-file", str(tmp_path),
        "--workers", str(workers),
        "--retry-failed",
    ]
    argv.append("--ingest" if ingest else "--no-ingest")

    def _cleanup_tmp():
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    _spawn_cli_streaming(job_id, argv, loop, on_finish=_cleanup_tmp)
    return JSONResponse({"job_id": job_id, "n_retried": len(requested)})


@app.post("/api/corpus/cleanup-downloads")
async def api_corpus_cleanup_downloads(
    dry_run: bool = Form(False),
    delete_dupes: bool = Form(True),
    cross_project: bool = Form(True),
    clean_failed: bool = Form(True),
    include_inbox: bool = Form(True),
):
    """Phase 54.6.4 + 54.6.19 + 54.6.273 — trigger `sciknow db cleanup-downloads`.

    Streams the subprocess log over SSE. Defaults: dry_run=False,
    delete_dupes=True, cross_project=True, clean_failed=True,
    include_inbox=True — so a single click removes all downloads +
    inbox files already ingested anywhere (including other projects),
    nukes the failed-ingest archive + associated documents rows, and
    removes empty inbox subfolders. The GUI exposes one button;
    advanced users can flip the flags via the CLI.
    """
    job_id, _queue = _create_job("corpus_cleanup_downloads")
    loop = asyncio.get_event_loop()
    argv = ["db", "cleanup-downloads"]
    if dry_run:
        argv.append("--dry-run")
    if delete_dupes:
        argv.append("--delete-dupes")
    argv.append("--cross-project" if cross_project else "--no-cross-project")
    argv.append("--clean-failed" if clean_failed else "--no-clean-failed")
    argv.append("--include-inbox" if include_inbox else "--no-include-inbox")
    _spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


@app.post("/api/corpus/expand/preview")
async def api_corpus_expand_preview(
    limit: int = Form(0),
    strategy: str = Form("rrf"),
    budget: int = Form(50),
    relevance: bool = Form(True),
    relevance_threshold: float = Form(0.0),
    relevance_query: str = Form(""),
    resolve: bool = Form(False),
):
    """Phase 54.6.3 — Preview the citation-expansion shortlist without
    downloading anything.

    Spawns ``sciknow db expand --dry-run --shortlist-tsv <tmp>`` as a
    subprocess so SSE streams its progress; the tempfile path is
    stored on the job record so the follow-up GET
    ``/api/corpus/expand/preview/{job_id}/candidates`` can parse it
    once the job has completed.

    Why subprocess + TSV rather than an in-process helper like
    expand-author: the citation expansion pipeline is ~250 lines of
    intertwined reference extraction, multi-source RRF ranking, and
    console output — far riskier to duplicate than ``find_author_-
    candidates``. Using the shortlist TSV keeps us bit-for-bit
    identical to the CLI path.
    """
    import tempfile
    import uuid

    tmp_dir = Path(tempfile.gettempdir()) / "sciknow-expand-preview"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = tmp_dir / f"shortlist-{uuid.uuid4().hex[:12]}.tsv"

    job_id, _queue = _create_job("corpus_expand_preview")
    # Stash the tempfile path so the candidates GET can find it.
    _jobs[job_id]["preview_tsv"] = str(tsv_path)

    loop = asyncio.get_event_loop()
    argv = [
        "db", "expand",
        "--dry-run",
        "--shortlist-tsv", str(tsv_path),
        "--limit", str(int(limit)),
        "--strategy", (strategy or "rrf"),
        "--budget", str(int(budget) if budget else 50),
        "--relevance-threshold", str(float(relevance_threshold or 0.0)),
    ]
    argv.append("--relevance" if relevance else "--no-relevance")
    argv.append("--resolve" if resolve else "--no-resolve")
    if relevance_query.strip():
        argv += ["--relevance-query", relevance_query.strip()]

    _spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id, "tsv_key": job_id})


@app.get("/api/corpus/expand/preview/{job_id}/candidates")
async def api_corpus_expand_preview_candidates(job_id: str):
    """Parse the shortlist TSV produced by a preview job and return
    JSON candidates (same shape as expand-author preview so the same
    UI can render both).
    """
    import csv
    with _job_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    tsv_path = Path(job.get("preview_tsv", ""))
    if not tsv_path.exists():
        raise HTTPException(404, "Preview TSV not found (job may still be running)")

    candidates: list[dict] = []
    kept = 0
    dropped = 0
    drop_reasons: dict[str, int] = {}
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            decision = (row.get("decision") or "").upper()
            doi = (row.get("doi") or "").strip() or None
            if decision == "KEEP":
                kept += 1
            else:
                dropped += 1
                reason = row.get("drop_reason") or "unspecified"
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            # Parse optional numerics.
            def _f(k):
                v = row.get(k)
                if not v or v in ("None", "nan"):
                    return None
                try:
                    return float(v)
                except Exception:
                    return None
            def _i(k):
                v = row.get(k)
                if not v or v in ("None", "nan"):
                    return None
                try:
                    return int(float(v))
                except Exception:
                    return None
            year = _i("year")
            candidates.append({
                "doi": doi,
                "arxiv_id": (row.get("arxiv_id") or "").strip() or None,
                "title": (row.get("title") or "").strip(),
                "authors": [],
                "year": year,
                "relevance_score": _f("bge_m3_cosine"),
                "rrf_score": _f("rrf_score"),
                "decision": decision or None,
                "drop_reason": (row.get("drop_reason") or "").strip() or None,
                "signals": {
                    "co_citation":       _f("co_citation"),
                    "bib_coupling":      _f("bib_coupling"),
                    "pagerank":          _f("pagerank"),
                    "influential_cites": _i("influential_cites"),
                    "cited_by":          _i("cited_by"),
                    "velocity":          _f("velocity"),
                    "concept_overlap":   _f("concept_overlap"),
                    "venue":             (row.get("venue") or "").strip() or None,
                },
            })
    # Clean up after parse — one-shot read, no reason to leave on disk.
    try:
        tsv_path.unlink(missing_ok=True)
    except Exception:
        pass
    return JSONResponse({
        "candidates": candidates,
        "info": {
            "total": kept + dropped,
            "kept": kept,
            "dropped": dropped,
            "drop_reasons": drop_reasons,
        },
    })


@app.post("/api/corpus/expand")
async def api_corpus_expand(
    limit: int = Form(0),
    # Phase 54.6.113 — RRF pool size per round. Default mirrors the CLI's
    # default (50). Separate from `limit` which is the total download cap.
    budget: int = Form(50),
    dry_run: bool = Form(False),
    resolve: bool = Form(False),
    ingest: bool = Form(True),
    relevance: bool = Form(True),
    relevance_threshold: float = Form(0.0),
    relevance_query: str = Form(""),
    workers: int = Form(0),
):
    """Invoke `sciknow db expand` from the web UI — SSE log stream.

    ``budget`` is the RRF pool size per round (default 50); ``limit`` is
    the hard cap on total downloads. See ``docs/EXPAND_RESEARCH.md`` §6a.
    Heavy flags (download_dir, delay) are left at CLI defaults to keep
    the web UX simple.
    """
    job_id, _queue = _create_job("corpus_expand")
    loop = asyncio.get_event_loop()
    argv = ["db", "expand",
            "--limit", str(limit),
            "--budget", str(max(5, min(int(budget), 200))),
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


# ── Phase 54.6.11 — /api/viz/* endpoints for the Visualize modal ───────
# Six lightweight JSON endpoints backing the six tabs. Heavy work
# (UMAP fit) is done in the helper and cached on disk per-project.

@app.get("/api/viz/topic-map")
async def api_viz_topic_map(refresh: bool = False):
    """UMAP 2D projection of every paper's abstract embedding."""
    from sciknow.core.viz_ops import topic_map
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: topic_map(refresh=bool(refresh)),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(result)


@app.get("/api/viz/raptor-tree")
async def api_viz_raptor_tree():
    """Hierarchical RAPTOR summary tree for the sunburst view."""
    from sciknow.core.viz_ops import raptor_tree
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, raptor_tree)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(result)


@app.post("/api/viz/consensus-landscape")
async def api_viz_consensus_landscape(topic: str = Form(""), model: str = Form(None)):
    """Claims scatter for a topic — wraps wiki consensus_map."""
    if not topic.strip():
        raise HTTPException(status_code=400, detail="topic required")
    from sciknow.core.viz_ops import consensus_landscape
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: consensus_landscape(topic, model=(model or None)),
        )
    except Exception as exc:
        logger.exception("viz consensus-landscape failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return JSONResponse(result)


@app.get("/api/viz/timeline")
async def api_viz_timeline():
    """Year × cluster stacked area for the timeline river."""
    from sciknow.core.viz_ops import timeline
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, timeline)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(result)


@app.get("/api/viz/ego-radial")
async def api_viz_ego_radial(document_id: str, k: int = 20):
    """Top-K similar papers radially around a document."""
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id required")
    from sciknow.core.viz_ops import ego_radial
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: ego_radial(document_id, k=int(k)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(result)


@app.get("/api/viz/gap-radar")
async def api_viz_gap_radar():
    """Per-chapter coverage radar for the active book."""
    from sciknow.core.viz_ops import gap_radar
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: gap_radar(_book_id),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(result)


@app.get("/api/settings/models")
async def api_settings_models():
    """54.6.106 — effective model assignments, surfaced in the Book
    Settings → Models tab. Read-only snapshot of the Settings object.

    Phase 54.6.244 — exposes ``book_write_model`` (added alongside the
    qwen3.6:27b-dense writer-role split in 54.6.243). The Models tab
    renders it with an explicit "inherits LLM_MODEL" indicator when
    unset, so the user can tell "writer explicitly pinned to the
    global" apart from "writer role defaults to the global".
    """
    from sciknow.config import settings as _s
    return JSONResponse({
        "llm_model": _s.llm_model,
        "llm_fast_model": _s.llm_fast_model,
        "book_write_model": getattr(_s, "book_write_model", None),
        "book_outline_model": getattr(_s, "book_outline_model", None),
        "book_review_model": _s.book_review_model,
        "autowrite_scorer_model": _s.autowrite_scorer_model,
        "visuals_caption_model": _s.visuals_caption_model,
        "mineru_vlm_model": getattr(_s, "mineru_vlm_model", None),
        "embedding_model": _s.embedding_model,
        "reranker_model": _s.reranker_model,
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


@app.get("/api/monitor/alerts-md")
async def api_monitor_alerts_md():
    """Phase 54.6.268 — return current alerts as a Markdown block.

    Shares ``core.monitor.alerts_as_markdown`` with the CLI
    ``sciknow db monitor --alerts-md`` so both UIs produce the same
    paste-ready format. Returns ``text/plain`` so copy-to-clipboard
    in the browser sees unescaped Markdown.
    """
    from fastapi.responses import PlainTextResponse
    from sciknow.core.monitor import (
        collect_monitor_snapshot, alerts_as_markdown,
    )
    snap = collect_monitor_snapshot()
    return PlainTextResponse(alerts_as_markdown(snap))


@app.get("/api/monitor")
async def api_monitor(days: int = 14):
    """Phase 54.6.230 — unified monitor snapshot for the web reader.

    One endpoint, one dict — same shape as ``sciknow db monitor
    --json`` because both call ``core.monitor.collect_monitor_
    snapshot``. The web "System Monitor" modal polls this every
    5s. Read-only; safe during active ingestion.

    Shape documented in ``sciknow/core/monitor.py``.

    Phase 54.6.238 — the web-process in-memory ``_jobs`` dict and
    the refresh pulse file are both added to the snapshot here so
    the modal can render live LLM-job + refresh progress without
    opening extra endpoints.
    """
    import time as _time
    from sciknow.core.monitor import collect_monitor_snapshot
    from sciknow.core.project import get_active_project
    from sciknow.core.pulse import read_pulse
    snap = collect_monitor_snapshot(
        throughput_days=max(1, int(days)),
        llm_usage_days=max(1, int(days)),
    )
    # Active web jobs — same process, direct read of _jobs dict.
    active: list[dict] = []
    now = _time.monotonic()
    with _job_lock:
        for jid, j in _jobs.items():
            if j.get("status") not in ("running", "starting"):
                continue
            started = j.get("started_at") or now
            active.append({
                "id": jid[:8],
                "type": j.get("task_desc") or j.get("job_type") or "?",
                "model": j.get("model_name") or None,
                "tokens": j.get("tokens", 0),
                # Phase 54.6.247 — TPS, shared helper with pulse writer
                "tps": round(_job_tps(j), 2),
                "elapsed_s": max(0, now - started),
                "target_words": j.get("target_words"),
                "stream_state": j.get("stream_state"),
            })
    snap["active_jobs"] = active
    # Refresh pulse — cross-process signal from `sciknow refresh`.
    try:
        active_project = get_active_project()
        snap["refresh_pulse"] = read_pulse(
            active_project.data_dir, "refresh",
        )
    except Exception:
        snap["refresh_pulse"] = None
    return JSONResponse(snap)


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
        # Phase 54.6.247 — shared helper with _write_web_jobs_pulse
        # so both UIs see the same number.
        tps = _job_tps(job)
        elapsed_s = time.monotonic() - job.get("started_at", time.monotonic())
        return JSONResponse({
            "id": job_id,
            "stream_state": job.get("stream_state", "streaming"),
            "tokens": int(job.get("tokens", 0)),
            "tps": round(tps, 2),
            # Phase 54.6.318 — surface the Ollama decode/prefill split
            # to the task-bar consumer too. Browsers can render
            # "decode 31 t/s · 92% prefill" alongside the 1.0 t/s
            # wall-clock so users instantly see what's actually slow.
            "decode_stats": _job_decode_stats(job),
            "tps_windows": _job_tps_windows(job),
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


@app.post("/api/admin/release-vram")
async def api_admin_release_vram():
    """Phase 54.6.320 — runtime VRAM-eviction switch.

    Call this while an autowrite / verify job is running and decode
    tok/s has tanked because bge-m3 + reranker + ColBERT got reloaded
    by an iteration's retrieve step and are now squeezing Ollama's
    writer model into a partial GPU load (the classic "decode 4 t/s
    instead of 30 t/s" regression).

    Frees the retrieval models held by THIS process (sciknow web
    server) and reports the VRAM delta. Ollama doesn't auto-rebalance
    a partial-load on its own — we also issue a model-unload via
    Ollama's API so the next LLM call re-pages it with full GPU.
    """
    import shutil
    import subprocess
    from sciknow.core.book_ops import _release_gpu_models

    def _vram_used() -> tuple[int, int, int]:
        """(used_mib, free_mib, total_mib) on GPU 0; (-1,-1,-1) if nvidia-smi missing."""
        if not shutil.which("nvidia-smi"):
            return (-1, -1, -1)
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
                 "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True, timeout=5,
            )
            parts = [int(x.strip()) for x in r.stdout.strip().split(",")]
            return (parts[0], parts[1], parts[2])
        except Exception:
            return (-1, -1, -1)

    before = _vram_used()
    _release_gpu_models()

    # Force Ollama to drop the model so the next call re-pages it
    # under the new (now-larger) free-VRAM budget. keep_alive=0 unloads.
    unloaded: list[str] = []
    try:
        import ollama as _ollama
        client = _ollama.Client(host=settings.ollama_host, timeout=10)
        # /api/ps returns the loaded models; unload each by issuing a
        # zero-token chat with keep_alive=0.
        try:
            ps = client.ps()
            loaded = [m.get("name") or m.get("model") for m in ps.get("models", [])]
        except Exception:
            loaded = []
        for name in loaded:
            if not name:
                continue
            try:
                client.generate(model=name, prompt="", keep_alive=0, stream=False)
                unloaded.append(name)
            except Exception:
                pass
    except Exception as exc:
        logger.warning("ollama unload during release-vram failed: %s", exc)

    after = _vram_used()
    return JSONResponse({
        "ok": True,
        "vram_before_mib": {"used": before[0], "free": before[1], "total": before[2]},
        "vram_after_mib": {"used": after[0], "free": after[1], "total": after[2]},
        "ollama_unloaded": unloaded,
        "note": (
            "Retrieval models freed and Ollama models unloaded. The "
            "next LLM call (the running job's next phase) will re-page "
            "the writer model with the full GPU budget. Decode tok/s "
            "should jump from the partial-load value back to native."
        ),
    })


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


@app.post("/api/chapters/{chapter_id}/plan-sections")
async def api_chapter_plan_sections(
    chapter_id: str,
    force: bool = Form(False),
    model: str = Form(None),
):
    """Phase 54.6.155 — web wrapper over book_ops.generate_section_plan.

    Walks every section of the chapter, generates a concept-list plan
    via LLM_FAST_MODEL (or the model override), skips sections with an
    existing plan unless force=True. Returns per-section results so
    the UI can re-render and show what changed.

    Non-streaming — each section takes ~5-10s on a cheap fast model,
    and total chapter latency is usually under 30s; a plain JSON
    response is simpler than wiring SSE for this. If chapter size
    grows beyond 8 sections per typical book, revisit.
    """
    from sciknow.core.book_ops import (
        generate_section_plan,
        _get_chapter_sections_normalized,
    )
    with get_session() as session:
        # Confirm chapter exists + pull sections list (read-only — the
        # generator re-reads inside its own session)
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
        except Exception as exc:  # noqa: BLE001
            logger.warning("plan-sections failed for %s: %s", slug, exc)
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


@app.get("/api/chapters/{chapter_id}/resolved-targets")
async def api_chapter_resolved_targets(chapter_id: str):
    """Phase 54.6.149 — per-section target + which fallback level fired.

    For each section in the chapter, calls the same resolver chain that
    autowrite uses (core.book_ops) and returns the resulting target
    along with a human-readable explanation of which level won. This
    makes the concept-density + per-chapter + per-section + project-
    type cascade visible in the Chapter modal so users don't need to
    start autowrite to find out.

    Returns:
      {
        "chapter_target": int,           # Level-3-style chapter target
        "chapter_level": "explicit_override" | "book_default" | "type_default",
        "sections": [
          {"slug": str, "title": str,
           "target": int,
           "level": "explicit_override" | "concept_density" | "chapter_split",
           "concepts": int | null,
           "wpc_midpoint": int | null,
           "explanation": str},
          ...
        ]
      }
    """
    from sciknow.core.book_ops import (
        _get_book_length_target, _section_target_words,
        _get_section_target_words, _get_section_concept_density_target,
        _get_chapter_sections_normalized, _count_plan_concepts,
        _get_section_plan, DEFAULT_TARGET_CHAPTER_WORDS,
    )
    from sciknow.core.project_type import get_project_type

    with get_session() as session:
        # Chapter target + which of its four levels fired
        book_row = session.execute(text("""
            SELECT book_type, COALESCE(custom_metadata, '{}'::jsonb),
                   CAST(:cid AS uuid)
            FROM books WHERE id::text = :bid LIMIT 1
        """), {"bid": _book_id, "cid": chapter_id}).fetchone()
        book_type = (book_row[0] if book_row else None) or "scientific_book"
        book_meta = (book_row[1] if book_row else {}) or {}
        if isinstance(book_meta, str):
            try:
                book_meta = json.loads(book_meta)
            except Exception:
                book_meta = {}
        # Chapter-override check
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

        # Sections
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
                    session, chapter_id, slug, _book_id,
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


@app.get("/api/book/length-report")
async def api_book_length_report():
    """Phase 54.6.162 — web wrapper over core.length_report.walk_book_lengths.

    Surfaces the 54.6.153 CLI report in the GUI so users see the whole
    book's projected length without leaving the browser. Pure wrapper —
    no arithmetic duplication; the walker calls the real resolver
    helpers.
    """
    from sciknow.core.length_report import walk_book_lengths
    try:
        report = walk_book_lengths(_book_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("length-report failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse(report.to_dict())


@app.get("/api/bench/section-lengths")
async def api_bench_section_lengths():
    """Phase 54.6.159 — surface the 54.6.157 section-length IQR bench
    data in the web UI. Runs the same bench function so the numbers
    stay in sync (no duplicate SQL); returns the per-section rows as
    JSON with the alignment tag parsed out of the bench note for easy
    rendering.
    """
    from sciknow.testing.bench import b_corpus_section_length_distribution
    try:
        metrics = list(b_corpus_section_length_distribution())
    except Exception as exc:  # noqa: BLE001
        logger.warning("section-length bench failed: %s", exc)
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


@app.get("/api/book-types")
async def api_book_types():
    """Phase 54.6.147 — list all registered project types with their
    research-grounded length ranges, for the setup-wizard dropdown +
    info panel to render.

    Each entry is self-contained (no joins needed): display_name,
    description, chapter target, concept count range, words-per-
    concept range, and a derived section-at-midpoint range so the UI
    can show users what each type implies before they pick one.
    """
    from sciknow.core.project_type import list_project_types
    out = []
    for pt in list_project_types():
        clo, chi = pt.concepts_per_section_range
        wlo, whi = pt.words_per_concept_range
        wmid = (wlo + whi) // 2
        out.append({
            "slug": pt.slug,
            "display_name": pt.display_name,
            "description": pt.description,
            "is_flat": pt.is_flat,
            "default_chapter_count": pt.default_chapter_count,
            "default_target_chapter_words": pt.default_target_chapter_words,
            "concepts_per_section_range": [clo, chi],
            "words_per_concept_range":   [wlo, whi],
            "section_at_midpoint_range": [clo * wmid, chi * wmid],
            "default_sections": [
                {"key": s.key, "title": s.title, "target_words": s.target_words}
                for s in pt.default_sections
            ],
        })
    return JSONResponse({"types": out})


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


@app.get("/api/reconciliations")
async def api_reconciliations():
    """Phase 54.6.125 (Tier 3 #3) — list current preprint↔journal
    reconciliations for the active project."""
    from sciknow.core.preprint_reconcile import list_reconciliations
    return JSONResponse({"pairs": list_reconciliations()})


@app.post("/api/reconciliations/undo")
async def api_reconciliations_undo(request: Request):
    """Body: {doc_id: 'uuid prefix'}. Clears canonical_document_id."""
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    doc_id = (body.get("doc_id") or "").strip()
    if not doc_id:
        raise HTTPException(400, "doc_id required")
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    from sciknow.core.preprint_reconcile import undo_reconciliation
    with get_session() as session:
        rows = session.execute(text("""
            SELECT id::text FROM documents
            WHERE id::text LIKE :p || '%' AND canonical_document_id IS NOT NULL
            LIMIT 2
        """), {"p": doc_id}).fetchall()
    if len(rows) == 0:
        raise HTTPException(404, "no non-canonical document matches")
    if len(rows) > 1:
        raise HTTPException(409, "ambiguous prefix — give more of the UUID")
    ok = undo_reconciliation(rows[0][0])
    return JSONResponse({"ok": bool(ok), "doc_id": rows[0][0]})


@app.get("/api/provenance")
async def api_provenance(key: str):
    """Phase 54.6.117 (Tier 4 #1) — provenance record by DOI / arxiv / doc-id prefix."""
    from sciknow.core.provenance import lookup as _lookup
    doc_id, rec = _lookup(key)
    return JSONResponse({
        "document_id": doc_id,
        "provenance": rec,
    })


@app.get("/api/feedback")
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


@app.post("/api/feedback")
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


@app.get("/api/projects/{slug}/venues")
async def api_project_venues_get(slug: str):
    """Phase 54.6.112 — return the project's venue_config.json as JSON."""
    from sciknow.core.project import Project, get_active_project
    from sciknow.core import venue_config as _vc
    if slug == "active":
        project = get_active_project()
    else:
        project = Project(slug=slug, repo_root=get_active_project().repo_root)
    cfg = _vc.load(project.root)
    return JSONResponse({
        "slug": project.slug,
        "path": str(_vc.path_for(project.root)),
        "blocklist": cfg.blocklist,
        "allowlist": cfg.allowlist,
    })


@app.post("/api/projects/{slug}/venues")
async def api_project_venues_post(slug: str, request: Request):
    """Mutate the venue block/allow lists. Body: ``{action, kind, pattern}``.

    action ∈ {add, remove}. kind ∈ {block, allow}.
    """
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    action = body.get("action")
    kind = body.get("kind")
    pattern = (body.get("pattern") or "").strip()
    if action not in ("add", "remove") or kind not in ("block", "allow") or not pattern:
        raise HTTPException(400, "Body must be {action:'add'|'remove', kind:'block'|'allow', pattern:'...'}")

    from sciknow.core.project import Project, get_active_project
    from sciknow.core import venue_config as _vc
    if slug == "active":
        project = get_active_project()
    else:
        project = Project(slug=slug, repo_root=get_active_project().repo_root)

    if action == "add":
        _, changed = _vc.add_pattern(project.root, pattern, kind=kind)
    else:
        _, changed = _vc.remove_pattern(project.root, pattern, kind=kind)
    cfg = _vc.load(project.root)
    return JSONResponse({
        "slug": project.slug,
        "changed": changed,
        "blocklist": cfg.blocklist,
        "allowlist": cfg.allowlist,
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


# ── Phase 54.6.24 — backup endpoints ───────────────────────────────────────

@app.get("/api/backups")
async def api_backups_list():
    """Return backup history + schedule status from the JSON state file."""
    from sciknow.cli.backup import _read_state, _backup_root
    state = _read_state()
    return JSONResponse({
        "backups": state.get("backups", []),
        "schedule": state.get("schedule"),
        "backup_dir": str(_backup_root()),
    })


@app.post("/api/backups/run")
async def api_backups_run():
    """Trigger a full backup via _spawn_cli_streaming."""
    job_id, _ = _create_job("backup_run")
    loop = asyncio.get_event_loop()
    _spawn_cli_streaming(job_id, ["backup", "run", "--all-projects"], loop)
    return JSONResponse({"job_id": job_id})


@app.get("/api/backups/download/{dirname}/{filename}")
async def api_backups_download(dirname: str, filename: str):
    """Serve a backup file. Validates the path stays within archives/backups/."""
    from sciknow.cli.backup import _backup_root
    from starlette.responses import FileResponse

    safe_dir = dirname.replace("..", "").replace("/", "")
    safe_file = filename.replace("..", "").replace("/", "")
    path = _backup_root() / safe_dir / safe_file
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Backup file not found")
    abs_root = _backup_root().resolve()
    if not path.resolve().is_relative_to(abs_root):
        raise HTTPException(403, "Path traversal rejected")
    return FileResponse(path, filename=safe_file,
                        media_type="application/octet-stream")


@app.post("/api/backups/restore")
async def api_backups_restore(request: Request):
    """Trigger a restore from a backup set via _spawn_cli_streaming.
    Body: {timestamp: "latest"|"20260415T030000Z", force: true|false}."""
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    ts = body.get("timestamp", "latest")
    force = body.get("force", False)
    job_id, _ = _create_job("backup_restore")
    loop = asyncio.get_event_loop()
    argv = ["backup", "restore", ts]
    if force:
        argv.append("--force")
    _spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


@app.post("/api/backups/schedule")
async def api_backups_schedule(request: Request):
    """Install or remove the cron. 54.6.94 adds frequency + minute + weekday.

    Body on enable: {action:"enable", frequency:"hourly"|"daily"|"weekly",
                     hour:0-23, minute:0-59, weekday:0-6}
    Body on disable: {action:"disable"}.
    """
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    action = body.get("action", "enable")
    if action == "enable":
        job_id, _ = _create_job("backup_schedule")
        loop = asyncio.get_event_loop()
        argv = ["backup", "schedule"]
        freq = (body.get("frequency") or "daily").strip().lower()
        argv += ["--frequency", freq]
        hour = int(body.get("hour", 3))
        minute = int(body.get("minute", 0))
        weekday = int(body.get("weekday", 0))
        argv += ["--hour", str(max(0, min(23, hour)))]
        argv += ["--minute", str(max(0, min(59, minute)))]
        argv += ["--weekday", str(max(0, min(6, weekday)))]
        _spawn_cli_streaming(job_id, argv, loop)
        return JSONResponse({"job_id": job_id})
    else:
        job_id, _ = _create_job("backup_unschedule")
        loop = asyncio.get_event_loop()
        _spawn_cli_streaming(job_id, ["backup", "unschedule"], loop)
        return JSONResponse({"job_id": job_id})


@app.post("/api/cli-stream")
async def api_cli_stream(request: Request):
    """Phase 54.6.97 — generic CLI dispatcher for draft-level actions.

    Body: ``{argv: ["book", "verify-draft", "<id>"]}``. Only commands on
    the allowlist below are accepted — keeps this from becoming a
    remote-shell. Streams stdout as SSE log events via
    ``_spawn_cli_streaming`` just like the backup endpoints.
    """
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
    job_id, _ = _create_job("cli_stream_" + argv[1].replace("-", "_"))
    loop = asyncio.get_event_loop()
    _spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


@app.post("/api/backups/delete")
async def api_backups_delete(request: Request):
    """54.6.94 — delete one backup set. Body: {timestamp: "<ts>"|"latest"}."""
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    ts = (body.get("timestamp") or "").strip()
    if not ts:
        raise HTTPException(400, "timestamp required")
    job_id, _ = _create_job("backup_delete")
    loop = asyncio.get_event_loop()
    _spawn_cli_streaming(job_id, ["backup", "delete", ts, "--yes"], loop)
    return JSONResponse({"job_id": job_id})


@app.post("/api/backups/purge")
async def api_backups_purge(request: Request):
    """54.6.94 — bulk delete. Body: {all:true} OR {older_than_days:N}."""
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    argv = ["backup", "purge", "--yes"]
    if body.get("all"):
        argv.append("--all")
    elif body.get("older_than_days"):
        try:
            n = int(body["older_than_days"])
        except (TypeError, ValueError):
            raise HTTPException(400, "older_than_days must be an integer")
        if n <= 0:
            raise HTTPException(400, "older_than_days must be positive")
        argv += ["--older-than-days", str(n)]
    else:
        raise HTTPException(400, "pass either {all:true} or {older_than_days:N}")
    job_id, _ = _create_job("backup_purge")
    loop = asyncio.get_event_loop()
    _spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


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
                 focus_draft=None, search_q="", search_results=None,
                 auto_open_modal=None):
    """Render the full book reader as a self-contained HTML page.

    Phase 54.6.178 — ``auto_open_modal`` accepts a modal DOM id
    (e.g. ``book-settings-modal``); when set, a <script> at the end
    of <body> opens that modal on DOMContentLoaded. Used by routed
    views (/settings, /plan, /wiki, …) so a URL deep-links to the
    matching modal without a layout refactor.
    """

    chapter_drafts = {}
    draft_map = {}
    for d in drafts:
        draft_id, title, sec_type, content, wc, sources, version, summary, \
            review_fb, ch_id, parent_id, created, ch_num, ch_title = d
        draft_map[draft_id] = d
        key = ch_id or "__none__"
        if key not in chapter_drafts:
            chapter_drafts[key] = []
        # Phase 54.6.309 — the SELECT already sorts is_active first then
        # MAX(version), so the first row per section is the one to
        # display. Skip any later ones instead of comparing versions.
        existing = [x for x in chapter_drafts[key] if x[2] == sec_type]
        if not existing:
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
        # per slug are reduced to the first-seen (active / highest-version)
        # by the Phase 54.6.309 ORDER BY in _get_book_data.
        draft_by_slug: dict[str, tuple] = {}
        for d in ch_ds:
            slug = _normalize_section(d[2] or "")
            existing = draft_by_slug.get(slug)
            if not existing:
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

    # Phase 54.6.309 — compute the book-wide global bibliography so the
    # active draft renders with global [N] numbers and the right panel
    # shows a deduped, cross-chapter source list. Also powers the
    # synthetic Bibliography chapter appended below.
    try:
        with get_session() as _bib_session:
            _bib = BookBibliography.from_book(_bib_session, _book_id)
    except Exception as _bib_exc:
        logger.warning("global bibliography build failed: %s", _bib_exc)
        _bib = BookBibliography()

    if chapters:
        _bib_ch_num = int(chapters[-1][1] or len(chapters)) + 1
        _bib_word_count = sum(len((s or "").split()) for s in _bib.global_sources)
        sidebar_items.append({
            "num": _bib_ch_num,
            "title": BIBLIOGRAPHY_TITLE,
            "id": BIBLIOGRAPHY_PSEUDO_ID,
            "description": "All publications cited across the book, numbered once.",
            "topic_query": "",
            "sections": [{
                "id": BIBLIOGRAPHY_PSEUDO_ID,
                "type": "bibliography",
                "title": BIBLIOGRAPHY_TITLE,
                "plan": "Auto-generated from every draft's citations.",
                "version": 1,
                "words": _bib_word_count,
                "status": "drafted",
            }],
            "sections_template": ["bibliography"],
            "sections_meta": [{
                "slug": "bibliography",
                "title": BIBLIOGRAPHY_TITLE,
                "plan": "Auto-generated — do not edit by hand.",
            }],
            "is_bibliography": True,
        })

    active_draft = None
    if focus_draft == BIBLIOGRAPHY_PSEUDO_ID:
        active_draft = None  # handled in the bibliography branch below
    elif focus_draft:
        active_draft = draft_map.get(focus_draft)
    elif drafts:
        active_draft = drafts[0]

    active_html = ""
    active_comments = []
    active_sources = []
    active_review = ""
    active_id = ""
    active_title = ""
    if focus_draft == BIBLIOGRAPHY_PSEUDO_ID:
        # Synthetic Bibliography pseudo-chapter view.
        active_id = BIBLIOGRAPHY_PSEUDO_ID
        active_title = BIBLIOGRAPHY_TITLE
        active_html = _md_to_html(render_bibliography_markdown(_bib))
        active_sources = list(_bib.global_sources)
    elif active_draft:
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
        # Phase 54.6.309 — remap [N] markers in the draft body to global
        # bibliography numbers before rendering.
        _raw_content = active_draft[3] or ""
        _remapped = _bib.remap_content(active_id, _raw_content)
        active_html = _md_to_html(_remapped)
        # Right panel shows only sources cited in THIS draft but with
        # their global [N] prefixes so the anchors match the body.
        _cited = _bib.cited_sources_for_draft(active_id)
        if _cited:
            active_sources = _cited
        else:
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
                '<svg class="empty-state__icon"><use href="#i-feather"/></svg>'
                '<h3>Welcome to your book</h3>'
                f'<p>This book has <strong>{len(chapters)} chapters</strong> outlined and '
                '<strong>0 drafts</strong>. To start writing, click any chapter title in the sidebar '
                '(it will highlight) and then use the toolbar above. The fastest path to a first draft '
                'is <strong>AI Autowrite</strong> &mdash; it runs the full write &rarr; review &rarr; '
                'revise convergence loop on the selected chapter.</p>'
                '<p>You can also explore the corpus without writing anything: '
                '<strong>Ask Corpus</strong>, <strong>Wiki Query</strong>, and '
                '<strong>Browse Papers</strong> all work without an active draft. Press '
                '<kbd>&#8984;K</kbd> / <kbd>Ctrl+K</kbd> to jump to any of them.</p>'
                '<p class="empty-state__tip">Each chapter in the sidebar has a '
                '<span class="u-accent">Start writing</span> shortcut that selects the '
                'chapter and immediately drafts an overview.</p>'
                '</div>'
            )
        else:
            active_title = book[1] if book else "Untitled book"
            active_html = (
                '<div class="empty-state">'
                '<svg class="empty-state__icon"><use href="#i-book-open"/></svg>'
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
    # Phase 54.6.178 — routed views. If the caller specified a modal
    # id (via ``auto_open_modal``), emit a <script> that opens it on
    # page load. The allowlist guards against injected path segments
    # landing in a DOM id lookup.
    auto_open_script = ""
    _MODAL_ALLOWLIST = {
        "plan-modal", "book-settings-modal", "wiki-modal",
        "bundles-modal", "tools-modal", "projects-modal",
        "catalog-modal", "export-modal", "pending-modal",
        "corpus-modal", "viz-modal", "kg-modal", "ask-modal",
        "setup-modal", "backups-modal", "visuals-modal",
        "reconciliations-modal", "ai-help-modal",
    }
    if auto_open_modal in _MODAL_ALLOWLIST:
        auto_open_script = (
            '<script>\n'
            'document.addEventListener("DOMContentLoaded", function () {\n'
            '  var m = document.getElementById("' + auto_open_modal + '");\n'
            '  if (m && typeof openModal === "function") {\n'
            '    setTimeout(function () {\n'
            '      window._routeNavigating = true;\n'
            '      try { openModal("' + auto_open_modal + '"); }\n'
            '      finally { window._routeNavigating = false; }\n'
            '    }, 30);\n'
            '  }\n'
            '});\n'
            '</script>'
        )

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
        auto_open_script=auto_open_script,
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
        # Phase 54.6.309 — bibliography pseudo-chapter gets its own CSS
        # hook, no "Ch.N:" prefix, and no delete button (auto-generated).
        is_bib = bool(ch.get("is_bibliography"))
        extra_cls = " ch-group--bibliography" if is_bib else ""
        title_prefix = "" if is_bib else f"Ch.{ch_num}: "
        delete_btn = (
            "" if is_bib else
            '<button onclick="event.stopPropagation();deleteChapter('
            'this.closest(&quot;.ch-group&quot;).dataset.chId)" '
            'title="Delete chapter">✗</button>'
        )
        out += f'<div class="ch-group{extra_cls}" data-ch-id="{ch_id}" data-ch-num="{ch_num}">'
        # Phase 14.2 — chapter title is clickable to SELECT the chapter.
        # Phase 23 — chevron at the start toggles collapse/expand of
        # the chapter's sections (event.stopPropagation so it doesn't
        # also fire selectChapter). Persistence + restore on page load
        # is handled by JS via localStorage.
        out += (
            f'<div class="ch-title clickable" data-ch-num="{ch_num}" '
            f'onclick="selectChapter(this.parentElement)">'
            f'<button class="ch-toggle" '
            f'onclick="event.stopPropagation();toggleChapter(this.closest(&quot;.ch-group&quot;))" '
            f'title="Collapse or expand sections">\u25be</button>'
            f'<span class="ch-title-text">{title_prefix}{ch_title}</span>'
            f'<span class="ch-actions">{delete_btn}</span></div>'
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400;1,6..72,500&family=JetBrains+Mono:wght@400;500;600&display=swap">
<link rel="stylesheet" href="/static/css/sciknow.css">
</head>
<body>

<!-- Phase 54.6.168 — Monoline icon sprite. Lucide-inspired paths;
     stroke attributes set per-symbol so each <use> inherits currentColor
     + stroke sizing through the symbol itself. Hidden via
     position:absolute + size:0 so it never enters layout. -->
<svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true"
     style="position:absolute;width:0;height:0;overflow:hidden">
  <symbol id="i-edit" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 20h9"/>
    <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4 12.5-12.5z"/>
  </symbol>
  <symbol id="i-zap" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
  </symbol>
  <symbol id="i-feather" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M20.24 12.24a6 6 0 0 0-8.49-8.49L5 10.5V19h8.5z"/>
    <line x1="16" y1="8" x2="2" y2="22"/>
    <line x1="17.5" y1="15" x2="9" y2="15"/>
  </symbol>
  <symbol id="i-message-square" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
  </symbol>
  <symbol id="i-refresh-cw" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="23 4 23 10 17 10"/>
    <polyline points="1 20 1 14 7 14"/>
    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
  </symbol>
  <symbol id="i-shield-check" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>
    <path d="M9 12l2 2 4-4"/>
  </symbol>
  <symbol id="i-brain" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18z"/>
    <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18z"/>
  </symbol>
  <symbol id="i-package" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M16.5 9.4L7.55 4.24"/>
    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
    <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
    <line x1="12" y1="22.08" x2="12" y2="12"/>
  </symbol>
  <symbol id="i-help-circle" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
    <line x1="12" y1="17" x2="12.01" y2="17"/>
  </symbol>
  <symbol id="i-chevron-down" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="6 9 12 15 18 9"/>
  </symbol>
  <symbol id="i-chevron-left" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="15 18 9 12 15 6"/>
  </symbol>
  <symbol id="i-chevron-right" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="9 18 15 12 9 6"/>
  </symbol>
  <symbol id="i-stop-square" viewBox="0 0 24 24" fill="currentColor" stroke="none">
    <rect x="6" y="6" width="12" height="12" rx="1.5"/>
  </symbol>
  <symbol id="i-check" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </symbol>
  <symbol id="i-x" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"/>
    <line x1="6" y1="6" x2="18" y2="18"/>
  </symbol>
  <symbol id="i-book-open" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
    <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
  </symbol>
  <symbol id="i-search" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="11" cy="11" r="7"/>
    <line x1="21" y1="21" x2="16.65" y2="16.65"/>
  </symbol>
  <symbol id="i-bar-chart" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <line x1="12" y1="20" x2="12" y2="10"/>
    <line x1="18" y1="20" x2="18" y2="4"/>
    <line x1="6" y1="20" x2="6" y2="16"/>
  </symbol>
  <symbol id="i-sprout" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M7 20h10"/>
    <path d="M10 20c5.5-2.5.8-6.4 3-10"/>
    <path d="M9.5 9.4c1.1.8 1.8 2.2 2.3 3.7-2 .4-3.5.4-4.8-.3-1.2-.6-2.3-1.9-3-4.2 2.8-.5 4.4 0 5.5.8z"/>
    <path d="M14.1 6a7 7 0 0 0-1.1 4c1.9-.1 3.3-.6 4.3-1.4 1-1 1.6-2.3 1.7-4.6-2.7.1-4 1-4.9 2z"/>
  </symbol>
  <symbol id="i-sliders" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <line x1="4" y1="21" x2="4" y2="14"/>
    <line x1="4" y1="10" x2="4" y2="3"/>
    <line x1="12" y1="21" x2="12" y2="12"/>
    <line x1="12" y1="8" x2="12" y2="3"/>
    <line x1="20" y1="21" x2="20" y2="16"/>
    <line x1="20" y1="12" x2="20" y2="3"/>
    <line x1="1" y1="14" x2="7" y2="14"/>
    <line x1="9" y1="8" x2="15" y2="8"/>
    <line x1="17" y1="16" x2="23" y2="16"/>
  </symbol>
  <symbol id="i-file-text" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
    <line x1="16" y1="13" x2="8" y2="13"/>
    <line x1="16" y1="17" x2="8" y2="17"/>
  </symbol>
  <!-- Phase 54.6.176 — second sprite batch for the dropdown-menu
       emoji sweep. Each action gets a distinct icon so the
       semantic signal the emoji carried is preserved. -->
  <symbol id="i-save" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
    <polyline points="17 21 17 13 7 13 7 21"/>
    <polyline points="7 3 7 8 15 8"/>
  </symbol>
  <symbol id="i-link" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
    <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
  </symbol>
  <symbol id="i-file-plus" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
    <line x1="12" y1="18" x2="12" y2="12"/>
    <line x1="9" y1="15" x2="15" y2="15"/>
  </symbol>
  <symbol id="i-trending-up" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
    <polyline points="17 6 23 6 23 12"/>
  </symbol>
  <symbol id="i-scale" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M16 16.5a4 4 0 0 0 8 0c0-3.5-4-7-4-7s-4 3.5-4 7z"/>
    <path d="M0 16.5a4 4 0 0 0 8 0c0-3.5-4-7-4-7s-4 3.5-4 7z"/>
    <path d="M7 21h10"/>
    <path d="M12 3v18"/>
    <path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>
  </symbol>
  <symbol id="i-target" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <circle cx="12" cy="12" r="6"/>
    <circle cx="12" cy="12" r="2"/>
  </symbol>
  <symbol id="i-users" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
    <circle cx="9" cy="7" r="4"/>
    <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
    <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
  </symbol>
  <symbol id="i-alert-octagon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86"/>
    <line x1="12" y1="8" x2="12" y2="12"/>
    <line x1="12" y1="16" x2="12.01" y2="16"/>
  </symbol>
  <symbol id="i-flask" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/>
    <path d="M8.5 2h7"/>
    <path d="M7 16h10"/>
  </symbol>
  <symbol id="i-download" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="7 10 12 15 17 10"/>
    <line x1="12" y1="15" x2="12" y2="3"/>
  </symbol>
  <symbol id="i-history" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 3v5h5"/>
    <path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/>
    <polyline points="12 7 12 12 16 14"/>
  </symbol>
  <symbol id="i-folder" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
  </symbol>
  <symbol id="i-layers" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <polygon points="12 2 2 7 12 12 22 7 12 2"/>
    <polyline points="2 17 12 22 22 17"/>
    <polyline points="2 12 12 17 22 12"/>
  </symbol>
  <symbol id="i-wrench" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>
  </symbol>
  <symbol id="i-globe" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <line x1="2" y1="12" x2="22" y2="12"/>
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
  </symbol>
  <symbol id="i-image" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
    <circle cx="8.5" cy="8.5" r="1.5"/>
    <polyline points="21 15 16 10 5 21"/>
  </symbol>
  <symbol id="i-clipboard" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>
    <rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>
  </symbol>
  <symbol id="i-trash" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="3 6 5 6 21 6"/>
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
  </symbol>
  <symbol id="i-tag" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/>
    <line x1="7" y1="7" x2="7.01" y2="7"/>
  </symbol>
  <symbol id="i-layout-grid" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <rect x="3" y="3" width="7" height="7" rx="1"/>
    <rect x="14" y="3" width="7" height="7" rx="1"/>
    <rect x="14" y="14" width="7" height="7" rx="1"/>
    <rect x="3" y="14" width="7" height="7" rx="1"/>
  </symbol>
  <symbol id="i-camera" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
    <circle cx="12" cy="13" r="4"/>
  </symbol>
  <symbol id="i-archive" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <rect x="3" y="3" width="18" height="4" rx="1"/>
    <path d="M5 7v13a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7"/>
    <line x1="10" y1="12" x2="14" y2="12"/>
  </symbol>
  <symbol id="i-user" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
    <circle cx="12" cy="7" r="4"/>
  </symbol>
  <symbol id="i-wand" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M15 4V2M15 16v-2M8 9h2M20 9h2M17.8 11.8L19 13M15 9h0M17.8 6.2L19 5M3 21l9-9M12.2 6.2L11 5"/>
  </symbol>
  <symbol id="i-inbox" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/>
    <path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>
  </symbol>
  <symbol id="i-menu" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <line x1="3"  y1="6"  x2="21" y2="6"/>
    <line x1="3"  y1="12" x2="21" y2="12"/>
    <line x1="3"  y1="18" x2="21" y2="18"/>
  </symbol>
  <symbol id="i-home" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 9.5 12 2l9 7.5V20a2 2 0 0 1-2 2h-4v-7h-6v7H5a2 2 0 0 1-2-2z"/>
  </symbol>
  <symbol id="i-sidebar" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2"/>
    <line x1="9" y1="3" x2="9" y2="21"/>
  </symbol>
  <symbol id="i-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="4"/>
    <line x1="12" y1="2"  x2="12" y2="4"/>
    <line x1="12" y1="20" x2="12" y2="22"/>
    <line x1="4.93" y1="4.93"  x2="6.34"  y2="6.34"/>
    <line x1="17.66" y1="17.66" x2="19.07" y2="19.07"/>
    <line x1="2"  y1="12" x2="4"  y2="12"/>
    <line x1="20" y1="12" x2="22" y2="12"/>
    <line x1="4.93" y1="19.07" x2="6.34"  y2="17.66"/>
    <line x1="17.66" y1="6.34"  x2="19.07" y2="4.93"/>
  </symbol>
  <symbol id="i-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
  </symbol>
</svg>

<!-- Phase 30 — persistent global task bar.
     Hidden by default; shown when startGlobalJob() runs. Lives at
     the very top of the body so it survives all SPA navigation
     (loadSection, showDashboard, showCorkboard, etc) and never gets
     overwritten by innerHTML rebuilds of the main content area. -->
<div id="task-bar" class="task-bar u-hidden">
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
  <span class="tb-stat tb-eta u-hidden" id="tb-eta">
    <span class="tb-sep">·</span>ETA <strong id="tb-eta-val">?</strong>
  </span>
  <span class="tb-spacer"></span>
  <button class="tb-stop" id="tb-stop" onclick="stopGlobalJob()" title="Stop the running task"><svg class="icon icon--sm"><use href="#i-stop-square"/></svg> Stop</button>
  <button class="tb-dismiss u-hidden" id="tb-dismiss" onclick="dismissTaskBar()" title="Dismiss">&times;</button>
</div>

<!-- Phase 54.6.186 — consolidated topbar. Left: book-level nav
     (Plan / Dashboard + 4 dropdowns). Right: per-draft actions (was
     the old in-main `.toolbar`). One bar, one menu surface, pinned
     at the top so writing actions stay in reach while the draft
     scrolls. Manage items (Projects / Backups / Tools / Setup
     Wizard) are ⌘K-only — rare enough not to merit topbar real
     estate. -->
<header class="topbar" id="topbar">
  <div class="topbar__left">
    <!-- Phase 54.6.193 — home anchor at the leftmost position. Clicks
         navigate to `/`, which reloads into the default reader view
         regardless of current URL or overlay state (Dashboard /
         Corkboard / Chapter Reader). Escape key does the same
         (see keydown handler). -->
    <a class="nav-btn topbar-home" href="/"
       title="Return to the reader home. Press Escape anywhere to do the same."
       aria-label="Return to reader home">
      <svg class="icon"><use href="#i-home"/></svg>
    </a>
    <button class="nav-btn" onclick="showDashboard()" title="Book dashboard with stats + heatmap"><svg class="icon"><use href="#i-bar-chart"/></svg> Dashboard</button>
    <div class="nav-dropdown" id="book-dropdown">
      <button class="nav-btn" onclick="toggleNavDropdown('book-dropdown', event)"
              title="Book-level surfaces: plan / leitmotiv, visual browse, history, snapshots, export, settings">
        <svg class="icon"><use href="#i-book-open"/></svg> Book <svg class="icon icon--sm"><use href="#i-chevron-down"/></svg>
      </button>
      <div class="nav-dropdown-menu" role="menu">
        <!-- Phase 54.6.193 — Plan moves out of the topbar direct row
             and into the Book menu as the first item: it's book-level
             rather than per-session and belongs alongside Corkboard /
             History / Snapshot / Export / Settings. -->
        <button role="menuitem" onclick="openPlanModal()" title="View / edit / regenerate the book plan (the leitmotiv)."><svg class="icon"><use href="#i-file-text"/></svg> Plan</button>
        <button role="menuitem" onclick="showCorkboard()" title="Book-wide corkboard view: every chapter's sections as cards you can drag + reorder."><svg class="icon"><use href="#i-layout-grid"/></svg> Corkboard</button>
        <button role="menuitem" onclick="showVersions()" title="Per-draft version history. Every save keeps the prior version so you can diff or revert."><svg class="icon"><use href="#i-history"/></svg> History</button>
        <button role="menuitem" onclick="openBibliographyTools()" title="Phase 54.6.312 — Sanity-check and renumber the global bibliography. Use after adding new chapters/sections when the citation numbers in stored markdown look inconsistent."><svg class="icon"><use href="#i-book-open"/></svg> Bibliography tools</button>
        <button role="menuitem" onclick="takeSnapshot()" title="Snapshot the whole book's draft state — safety net before a destructive operation like autowrite-all."><svg class="icon"><use href="#i-camera"/></svg> Snapshot</button>
        <button role="menuitem" onclick="openExportModal()" title="Export the book to Markdown, HTML, PDF (WeasyPrint), EPUB (pandoc), LaTeX, DOCX, or BibTeX."><svg class="icon"><use href="#i-download"/></svg> Export</button>
        <button role="menuitem" onclick="openBookSettings()" title="Per-book settings: title, description, plan (leitmotiv), target chapter length, style fingerprint, per-role model assignments."><svg class="icon"><use href="#i-sliders"/></svg> Settings</button>
      </div>
    </div>
    <div class="nav-dropdown" id="explore-dropdown">
      <button class="nav-btn" onclick="toggleNavDropdown('explore-dropdown', event)"
              title="Query the corpus: RAG, compiled wiki, paper catalog">
        <svg class="icon"><use href="#i-search"/></svg> Explore <svg class="icon icon--sm"><use href="#i-chevron-down"/></svg>
      </button>
      <div class="nav-dropdown-menu" role="menu">
        <button role="menuitem" onclick="openAskModal()" title="Natural-language question against the corpus with grounded citations. Mirrors `sciknow ask question`."><svg class="icon"><use href="#i-search"/></svg> Ask Corpus</button>
        <button role="menuitem" onclick="openWikiModal()" title="Query the pre-compiled wiki summaries (one per paper). Faster than Ask Corpus and returns the summary prose + source paper list."><svg class="icon"><use href="#i-book-open"/></svg> Wiki Query</button>
        <button role="menuitem" onclick="openCatalogModal()" title="Browse every paper in the corpus with filters for year, section type, topic cluster, paper type."><svg class="icon"><use href="#i-folder"/></svg> Browse Papers</button>
        <button role="menuitem" onclick="openVisualsModal()" title="Browse every extracted table, equation, figure, chart, and code block. Gallery + list modes with pagination and importance ranking."><svg class="icon"><use href="#i-image"/></svg> Visuals (Tables/Figs/Eqs)</button>
      </div>
    </div>
    <div class="nav-dropdown" id="corpus-dropdown">
      <button class="nav-btn" onclick="toggleNavDropdown('corpus-dropdown', event)"
              title="Grow and enrich the corpus: enrich metadata + five expand vectors + cleanup + pending">
        <svg class="icon"><use href="#i-sprout"/></svg> Corpus <svg class="icon icon--sm"><use href="#i-chevron-down"/></svg>
      </button>
      <div class="nav-dropdown-menu" role="menu">
        <button role="menuitem" onclick="openCorpusModal('corp-enrich')" title="Fill missing DOIs via Crossref/OpenAlex/arXiv title search + persist OpenAlex extras (concepts/funders/grants/ROR). Mirrors `sciknow db enrich`."><svg class="icon"><use href="#i-search"/></svg> Enrich metadata</button>
        <button role="menuitem" onclick="openCorpusModal('corp-cites')" title="Outbound reference crawl — follow citations IN your papers to discover new work. RRF + MMR diversity + citation-context signals. Mirrors `sciknow db expand`."><svg class="icon"><use href="#i-globe"/></svg> Expand (citations)</button>
        <button role="menuitem" onclick="openCorpusModal('corp-author')" title="Fetch every paper by a named author via OpenAlex. Use when you want an author's full bibliography regardless of current citations. Mirrors `sciknow db expand-author`."><svg class="icon"><use href="#i-user"/></svg> Expand by author</button>
        <button role="menuitem" onclick="openExpandAuthorRefs()" title="Phase 54.6.309 — pick an existing author from the corpus. Aggregates every paper they cited across all their works (including self-cites), ranks by citation frequency, and lands in the cherry-pick modal before download. Mirrors `sciknow db expand-author-refs`."><svg class="icon"><use href="#i-user"/></svg> Expand by author's references</button>
        <button role="menuitem" onclick="openCorpusModal('corp-inbound')" title="Forward-in-time mirror of Expand: find papers that CITE your corpus. Mirrors `sciknow db expand-inbound`."><svg class="icon"><use href="#i-inbox"/></svg> Inbound cites</button>
        <button role="menuitem" onclick="openCorpusModal('corp-topic')" title="OpenAlex free-text topic search ranked by citation count. Good for kickstarting a new project."><svg class="icon"><use href="#i-tag"/></svg> Topic search</button>
        <button role="menuitem" onclick="openCorpusModal('corp-coauth')" title="Find people who coauthored with your corpus's authors. Useful for invisible-college expansion."><svg class="icon"><use href="#i-users"/></svg> Coauthors</button>
        <div class="u-border-b" style="height:1px;margin:2px 0;"></div>
        <button role="menuitem" onclick="openCorpusModal('corp-enrich');doToolCorpus('cleanup')" title="Remove already-ingested duplicates from the downloads/ directory, delete inbox/ PDFs already 'complete' in the DB (plus empty inbox subfolders), AND permanently delete the failed-ingest archive. Frees disk; the main pipeline archive stays intact."><svg class="icon"><use href="#i-trash"/></svg> Cleanup downloads + inbox + failed</button>
        <button role="menuitem" onclick="openPendingDownloadsModal()" title="Papers you selected for download but couldn't be auto-retrieved (no open-access PDF). Retry, mark manually acquired, or export for ILL."><svg class="icon"><use href="#i-clipboard"/></svg> Pending downloads</button>
      </div>
    </div>
    <div class="nav-dropdown" id="viz-dropdown">
      <button class="nav-btn" onclick="toggleNavDropdown('viz-dropdown', event)"
              title="Seven visualizations of the corpus"><svg class="icon"><use href="#i-layers"/></svg> Visualize <svg class="icon icon--sm"><use href="#i-chevron-down"/></svg></button>
      <div class="nav-dropdown-menu" id="viz-dropdown-menu" role="menu">
        <button role="menuitem" onclick="openKgModal()" title="Knowledge graph of entities + relationships extracted from the corpus (Phase 54.6.50+). Zoomable force layout + table view."><svg class="icon"><use href="#i-link"/></svg> Knowledge Graph</button>
        <button role="menuitem" onclick="openVizModal('viz-topic')" title="UMAP 2D projection of every paper's abstract embedding. Colour-coded by topic cluster. Hover to identify."><svg class="icon"><use href="#i-globe"/></svg> Topic map (UMAP)</button>
        <button role="menuitem" onclick="openVizModal('viz-sunburst')" title="Hierarchical RAPTOR summary tree as a sunburst: root is the whole corpus, each ring is a coarser summary level. Click a wedge to drill in."><svg class="icon"><use href="#i-layers"/></svg> RAPTOR sunburst</button>
        <button role="menuitem" onclick="openVizModal('viz-consensus')" title="Consensus landscape: for a topic, show claim-level support/contradict structure across papers."><svg class="icon"><use href="#i-scale"/></svg> Consensus landscape</button>
        <button role="menuitem" onclick="openVizModal('viz-timeline')" title="Timeline river: paper counts by year, colour-banded by topic cluster. Reveals temporal trends."><svg class="icon"><use href="#i-trending-up"/></svg> Timeline river</button>
        <button role="menuitem" onclick="openVizModal('viz-ego')" title="Ego radial: pick a paper, see its citation neighbourhood as concentric rings (depth-1/2 citers + cited-by)."><svg class="icon"><use href="#i-target"/></svg> Ego radial</button>
        <button role="menuitem" onclick="openVizModal('viz-radar')" title="Gap radar: per-topic coverage vs what the corpus plan says it should cover. Big gaps visible at a glance."><svg class="icon"><use href="#i-bar-chart"/></svg> Gap radar</button>
      </div>
    </div>
  </div>
  <!-- Per-draft actions (lifted from the former in-main `.toolbar`). -->
  <div class="topbar__right toolbar" id="toolbar">
    <button class="primary" onclick="toggleEdit()" title="Manually edit the draft content (in-browser markdown editor with autosave)"><svg class="icon"><use href="#i-edit"/></svg> Edit</button>
    <div class="sep"></div>
    <!-- Phase 54.6.188 — the AI verbs (Autowrite / Write / Review /
         Revise) collapse into one dropdown. Saves horizontal space
         in the toolbar and matches the Verify / Critique / Extras
         grouping pattern. Each item stays one click away via the
         dropdown; ⌘K reaches them directly by name. -->
    <div class="nav-dropdown nav-dropdown-left tb-dropdown" id="ai-tb-dropdown">
      <button onclick="toggleNavDropdown('ai-tb-dropdown', event)"
              title="AI writing actions — autowrite loop, one-shot draft, review, revise">
        <svg class="icon"><use href="#i-zap"/></svg> AI <svg class="icon icon--sm"><use href="#i-chevron-down"/></svg>
      </button>
      <div class="nav-dropdown-menu" role="menu">
        <button role="menuitem" onclick="doAutowrite()" title="Autonomous AI write → review → revise loop"><svg class="icon"><use href="#i-zap"/></svg> Autowrite</button>
        <button role="menuitem" onclick="doWrite()" title="AI drafts this section from scratch (single pass)"><svg class="icon"><use href="#i-feather"/></svg> Write</button>
        <button role="menuitem" onclick="doReview()" title="AI critic pass on this section"><svg class="icon"><use href="#i-message-square"/></svg> Review</button>
        <button role="menuitem" onclick="doRevise()" title="AI revises based on review feedback"><svg class="icon"><use href="#i-refresh-cw"/></svg> Revise</button>
      </div>
    </div>
    <div class="sep"></div>
    <div class="nav-dropdown nav-dropdown-left tb-dropdown" id="verify-tb-dropdown">
      <button onclick="toggleNavDropdown('verify-tb-dropdown', event)"
              title="Verify citations, insert [N] markers, view autowrite score history">
        <svg class="icon"><use href="#i-shield-check"/></svg> Verify <svg class="icon icon--sm"><use href="#i-chevron-down"/></svg>
      </button>
      <div class="nav-dropdown-menu" role="menu">
        <button role="menuitem" onclick="doVerify()" title="Verify citations against sources (Phases 7+11)"><svg class="icon"><use href="#i-check"/></svg> Verify</button>
        <button role="menuitem" onclick="doVerifyDraft()" title="Atomize each sentence and NLI-check every sub-claim for mixed-truth failures (54.6.83)"><svg class="icon"><use href="#i-flask"/></svg> Verify Draft (claim-atomization)</button>
        <button role="menuitem" onclick="doFinalizeDraft()" title="Phase 54.6.145/162 — Level-3 VLM claim-depiction verify on every [Fig. N] marker. Runs the vision-language model on (claim, image) pairs and flags figures whose images don't clearly depict the cited claim. Deferred from the per-iteration autowrite loop (too expensive) — run once before export. Exit code 0 if clean, 1 if any flagged. ~3-10s per marker on CPU; 8-figure chapter ≈ 1-2 min."><svg class="icon"><use href="#i-save"/></svg> Finalize Draft (L3 VLM verify)</button>
        <button role="menuitem" onclick="doAlignCitations()" title="Remap [N] markers to the chunk that actually entails each sentence (54.6.71, conservative)"><svg class="icon"><use href="#i-link"/></svg> Align Citations</button>
        <button role="menuitem" onclick="doInsertCitations()" title="Two-pass LLM inserts [N] citation markers where needed; mirrors `sciknow book insert-citations`. Saves a new version."><svg class="icon"><use href="#i-file-plus"/></svg> Insert Citations</button>
        <button role="menuitem" onclick="showScoresPanel()" title="Phase 13 — convergence trajectory for autowrite drafts"><svg class="icon"><use href="#i-trending-up"/></svg> Scores</button>
      </div>
    </div>
    <div class="nav-dropdown nav-dropdown-left tb-dropdown" id="critique-tb-dropdown">
      <button onclick="toggleNavDropdown('critique-tb-dropdown', event)"
              title="Evidence mapping + BMAD-inspired critic skills">
        <svg class="icon"><use href="#i-brain"/></svg> Critique <svg class="icon icon--sm"><use href="#i-chevron-down"/></svg>
      </button>
      <div class="nav-dropdown-menu" role="menu">
        <button role="menuitem" onclick="promptArgue()" title="Map evidence for/against a claim"><svg class="icon"><use href="#i-scale"/></svg> Argue (map claim)</button>
        <button role="menuitem" onclick="doGaps()" title="Analyse gaps in the book"><svg class="icon"><use href="#i-search"/></svg> Gaps</button>
        <button role="menuitem" onclick="doAdversarialReview()" title="Cynical critic pass — finds ≥10 concrete issues, never graded. BMAD-inspired. Doesn't overwrite review_feedback."><svg class="icon"><use href="#i-alert-octagon"/></svg> Adversarial review</button>
        <button role="menuitem" onclick="doEdgeCases()" title="Exhaustive edge-case hunter — walks every scope boundary, counter-case, causal alternative, and quantitative limit. Structured findings."><svg class="icon"><use href="#i-target"/></svg> Edge cases</button>
        <button role="menuitem" onclick="doEnsembleReview()" title="N independent NeurIPS-rubric reviewers (default 3, T=0.75, rotating stance) + meta-fusion. Higher variance reduction but N× cost."><svg class="icon"><use href="#i-users"/></svg> Ensemble Review</button>
      </div>
    </div>
    <div class="nav-dropdown nav-dropdown-left tb-dropdown" id="extras-tb-dropdown">
      <button onclick="toggleNavDropdown('extras-tb-dropdown', event)"
              title="Chapter-scoped snapshots + continuous read">
        <svg class="icon"><use href="#i-package"/></svg> Extras <svg class="icon icon--sm"><use href="#i-chevron-down"/></svg>
      </button>
      <div class="nav-dropdown-menu" role="menu">
        <button role="menuitem" onclick="openBundleSnapshots()" title="Snapshot / restore whole chapter or whole book — safety net for autowrite-all"><svg class="icon"><use href="#i-package"/></svg> Bundles (chapter / book)</button>
        <button role="menuitem" onclick="showChapterReader()" title="Read entire chapter as continuous scroll"><svg class="icon"><use href="#i-book-open"/></svg> Chapter reader</button>
      </div>
    </div>
    <button onclick="openAIActionsHelp()" title="What does each AI action do? Quick reference." aria-label="AI actions help"><svg class="icon"><use href="#i-help-circle"/></svg></button>
    <button class="cmdk-trigger" onclick="openCmdK()"
            title="Commands · open the palette (⌘K / Ctrl+K). Jump to any action, setting, or panel by name."
            aria-label="Open command palette">
      <kbd>⌘K</kbd>
    </button>
  </div>
</header>

<div class="app-body">

<!-- Sidebar -->
<nav class="sidebar">
  <h2>{book_title}</h2>
  <div class="search-bar">
    <form action="/search" method="get">
      <input type="text" name="q" placeholder="Search..." value="{search_q}"
             title="Hybrid retrieval across the corpus (dense + sparse + FTS + reranker). Press Enter to search. Same pipeline as `sciknow search query`."/>
    </form>
  </div>
  {search_results_html}
  <!-- Phase 23 — collapse / expand all chapters in the sidebar.
       The icon flips from \u25be (down arrow, expanded) to \u25b8
       (right arrow, collapsed) to mirror the per-chapter chevrons. -->
  <div class="sidebar-controls">
    <button class="col-hide-btn" onclick="toggleColumn('sidebar')"
            title="Hide the chapters column. A peek button at the left edge brings it back. Auto-hide preference lives in Book Settings → View.">
      <svg class="icon icon--sm"><use href="#i-chevron-left"/></svg> Hide
    </button>
    <!-- Phase 54.6.194 — rail toggle. Cycles between full (280 px
         with titles) and rail (~64 px with chapter numbers +
         status dots). Independent of Hide. -->
    <button class="sidebar-rail-btn" onclick="toggleSidebarRail()"
            title="Toggle rail mode (compact chapter navigator). Shows chapter numbers and section status dots only; hover any chapter to see its full title."
            aria-label="Toggle sidebar rail mode">
      <svg class="icon icon--sm"><use href="#i-sidebar"/></svg>
    </button>
    <button class="sidebar-toggle-all" onclick="toggleAllChapters()"
            title="Collapse or expand all chapter sections">
      <span id="toggle-all-icon">\u25bd</span>
      <span id="toggle-all-label">Collapse all</span>
    </button>
  </div>
  <div id="sidebar-sections">
    {sidebar_html}
  </div>
  <div class="ch-add-form u-hidden" id="ch-add-form">
    <input type="text" id="ch-add-title" placeholder="New chapter title..."
           title="Title for the new chapter. A slug is auto-generated from the title."/>
    <button onclick="addChapter()"
            title="Create a new empty chapter at the end of the book. Sections can be added via the Plan modal's Chapters tab.">Add Chapter</button>
    <button class="u-bg-danger" onclick="document.getElementById('ch-add-form').style.display='none'"
            title="Close without creating a chapter.">Cancel</button>
  </div>
  <div style="padding: 4px 16px;">
    <button class="u-tiny u-pill u-border u-r-sm u-click u-bg" onclick="document.getElementById('ch-add-form').style.display='block'"
            title="Add a new chapter at the end of the book. Opens an inline title input.">+ Add Chapter</button>
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
    <span class="word-target u-hidden" id="word-target">
      <span id="word-target-text"></span>
      <span class="word-target-bar"><span class="word-target-fill" id="word-target-fill"></span></span>
    </span>
    <button class="edit-btn" onclick="toggleEdit()"
            title="Toggle the in-browser markdown editor with autosave, KaTeX math, and inline figure thumbnails.">Edit</button>
    <select class="status-select" id="status-select" onchange="updateStatus(this.value)"
            title="Workflow status for this draft. Drafted → Reviewed → Revised → Final. `final` / `reviewed` / `revised` drafts feed the style-fingerprint extractor.">
      <option value="to_do">To Do</option>
      <option value="drafted" selected>Drafted</option>
      <option value="reviewed">Reviewed</option>
      <option value="revised">Revised</option>
      <option value="final">Final</option>
    </select>
  </div>


  <!-- Phase 13 — Score history panel (collapsible, lazy-loaded) -->
  <div class="scores-panel" id="scores-panel">
    <div class="scores-header">
      <h4>Convergence trajectory</h4>
      <button class="modal-close" onclick="document.getElementById('scores-panel').classList.remove('open')" title="Close the scores panel.">&times;</button>
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
  <div class="u-hidden" id="dashboard-view"></div>

  <!-- Streaming output panel -->
  <div class="stream-panel" id="stream-panel">
    <div class="stream-header">
      <span class="status" id="stream-status">Starting...</span>
      <button class="stop-btn" id="stream-stop" onclick="stopJob()"
              title="Cancel the running job (write/review/revise/autowrite/ask/…). Partial output is kept.">Stop</button>
    </div>
    <div class="stream-scores" id="stream-scores"></div>
    <div class="stream-body" id="stream-body"></div>
    <div id="main-stream-stats" class="stream-stats" style="margin: 0 14px 12px;"></div>
  </div>

  <div id="read-view">{content_html}</div>

  <div class="u-hidden" id="edit-view">
    <div class="editor-toolbar" role="toolbar" aria-label="Editor toolbar">
      <button onclick="edInsert('**','**')" title="Bold (Ctrl+B)"><b>B</b></button>
      <button onclick="edInsert('*','*')" title="Italic (Ctrl+I)"><i>I</i></button>
      <button onclick="edInsert('~~','~~')" title="Strikethrough (~~text~~)"><s>S</s></button>
      <button onclick="edInsert('`','`')" title="Inline code (`code`)"><code>&lt;/&gt;</code></button>
      <span class="toolbar-sep" aria-hidden="true"></span>
      <button onclick="edInsert('## ','')" title="Heading 2 (## text)">H2</button>
      <button onclick="edInsert('### ','')" title="Heading 3 (### text)">H3</button>
      <button onclick="edInsert('#### ','')" title="Heading 4 (#### text)">H4</button>
      <span class="toolbar-sep" aria-hidden="true"></span>
      <button onclick="edInsertLine('- ')" title="Bulleted list item">&#8226; List</button>
      <button onclick="edInsertLine('1. ')" title="Numbered list item">1. List</button>
      <button onclick="edInsertLine('> ')" title="Block quote">&ldquo; Quote</button>
      <button onclick="edInsertLine('---\\n')" title="Horizontal rule">&mdash;</button>
      <span class="toolbar-sep" aria-hidden="true"></span>
      <button onclick="edInsertLink()" title="Insert a link ([text](url))">&#128279; Link</button>
      <button onclick="edInsertCite()" title="Inline citation marker [N]">[N]</button>
      <button onclick="edInsert('$','$')" title="Inline math ($E=mc^2$)">&#8747;</button>
      <button onclick="edInsertLine('$$\\n', '\\n$$\\n')" title="Display math block ($$…$$)">&#8721;</button>
      <span class="toolbar-sep" aria-hidden="true"></span>
      <button onclick="edUndo()" title="Undo (Ctrl+Z) — uses the browser's native undo on the textarea.">&#8630;</button>
      <button onclick="edRedo()" title="Redo (Ctrl+Shift+Z)">&#8631;</button>
      <span class="toolbar-sep" aria-hidden="true"></span>
      <button onclick="edSaveAsVersion()"
              title="Save the current buffer as a NEW version with an optional name (e.g. 'pre-review'). Lets you keep the current version around as an undo path.">
        &#128190;&#43; Save as new version…
      </button>
      <button onclick="showVersions()"
              title="Open the versions panel to review, diff, pin, or rename every saved version of this section.">
        &#128197; Versions
      </button>
      <span class="autosave" id="autosave-status" title="Autosaves every 5 seconds while editing">
        <span class="dot"></span><span id="autosave-text">Autosave on</span>
      </span>
    </div>
    <div class="editor-split">
      <div class="editor-src">
        <textarea class="edit-area" id="edit-area" oninput="edPreview()"
                  title="Markdown source. Autosaves every 5s. Supports $$…$$ KaTeX and ![alt](/api/visuals/image/<id>) for inline figures."></textarea>
      </div>
      <div class="editor-preview" id="edit-preview"
           title="Live-rendered preview of the markdown source."></div>
    </div>
    <div class="u-mt-2">
      <button class="edit-btn" onclick="edSave()"
              title="Save immediately. The autosave also fires every 5s.">Save</button>
      <button class="edit-btn u-bg-danger" onclick="toggleEdit()"
              title="Close the editor. Unsaved changes are lost — the autosave is your safety net.">Cancel</button>
    </div>
  </div>

  <!-- Argument map container -->
  <div class="u-hidden" id="argue-map-view"></div>
</main>

<!-- Right panel -->
<aside class="panel">
  <!-- Phase 54.6.167 — Context rail. One section visible at a time,
       driven by a segmented control. Replaces the three always-on
       <h3> stacks with focused single-tab view per draft. -->
  <div class="panel-head">
    <div class="panel-seg" role="tablist" aria-label="Context panel section">
      <button class="panel-seg-btn is-active" role="tab" aria-selected="true"
              data-ctx="sources" onclick="switchContextTab('sources')"
              title="Retrieved chunks used to ground this draft.">Sources</button>
      <button class="panel-seg-btn" role="tab" aria-selected="false"
              data-ctx="review" onclick="switchContextTab('review')"
              title="Critic pass output for the current draft version.">Review</button>
      <button class="panel-seg-btn" role="tab" aria-selected="false"
              data-ctx="comments" onclick="switchContextTab('comments')"
              title="Per-draft comments. Supports resolve + Markdown.">Comments</button>
      <button class="panel-seg-btn" role="tab" aria-selected="false"
              data-ctx="visuals" onclick="switchContextTab('visuals')"
              title="Figures, tables and charts from the corpus, ranked by relevance to this draft. Click an image to insert it into the text.">Visuals</button>
    </div>
    <button class="col-hide-btn col-hide-btn--icon" onclick="toggleColumn('panel')"
            title="Hide this column. A peek button at the right edge brings it back. Auto-hide preference lives in Book Settings → View."
            aria-label="Hide column">
      <svg class="icon icon--sm"><use href="#i-chevron-right"/></svg>
    </button>
  </div>

  <div class="panel-pane is-active" id="panel-pane-sources" role="tabpanel">
    <div id="panel-sources">{sources_html}</div>
  </div>

  <div class="panel-pane" id="panel-pane-review" role="tabpanel">
    <div id="panel-review">{review_html}</div>
  </div>

  <div class="panel-pane" id="panel-pane-comments" role="tabpanel">
    <div id="panel-comments">{comments_html}</div>
    <form class="comment-form" action="/comment" method="post" id="comment-form">
      <input type="hidden" name="draft_id" value="{active_id}" id="comment-draft-id">
      <textarea name="comment" placeholder="Add a comment…"
                title="Per-draft comments persist in the database. Use Markdown; @-mentions are NOT wired."></textarea>
      <button type="submit"
              title="Save the comment to this draft. Resolved comments get struck through via the Resolve button on each saved comment.">Add Comment</button>
    </form>
  </div>

  <!-- Phase 54.6.309 — Visuals suggestions pane. Ranked list of
       figures/tables/charts from the corpus whose content matches the
       current draft. Filled lazily when the tab is first opened (or
       on explicit Refresh) so the ranker cost isn't paid on every
       navigation. -->
  <div class="panel-pane" id="panel-pane-visuals" role="tabpanel">
    <div class="panel-visuals-head">
      <button class="btn-primary btn-sm" id="visuals-rank-btn" onclick="rankVisualSuggestions()"
              title="Compute visual rankings for this draft. Runs the 5-signal cross-encoder over up to 15 corpus visuals (~1–2s). Result is saved to the draft so subsequent opens are instant; use Re-rank to recompute after the draft changes.">Rank</button>
      <button class="btn-sm u-hidden" id="visuals-clear-btn" onclick="clearVisualSuggestions()"
              title="Delete the saved ranking for this draft so the next Rank press recomputes from scratch.">Clear</button>
      <span class="panel-visuals-view-toggle"
            title="Switch between gallery (big thumbnails in a grid) and list (one row per visual with details) layouts.">
        <button class="btn-sm is-active" id="vis-pane-view-gallery" onclick="setVisualsPaneView('gallery')">Gallery</button>
        <button class="btn-sm" id="vis-pane-view-list" onclick="setVisualsPaneView('list')">List</button>
      </span>
      <span class="panel-visuals-hint" id="panel-visuals-hint">Click a thumbnail to enlarge, Insert to append to the draft.</span>
    </div>
    <div id="panel-visuals"><em>Open this tab, then click Rank to compute visual suggestions.</em></div>
  </div>
</aside>

</div> <!-- /.app-body -->

<!-- Phase 54.6.164 — peek buttons. Visible only when the matching
     column is hidden (body class `sidebar-hidden` / `panel-hidden`). -->
<button class="col-peek-btn col-peek-sidebar" onclick="toggleColumn('sidebar')"
        title="Show the chapters column" aria-label="Show chapters column">
  <svg class="icon"><use href="#i-chevron-right"/></svg>
</button>
<button class="col-peek-btn col-peek-panel" onclick="toggleColumn('panel')"
        title="Show the sources/comments column" aria-label="Show sources/comments column">
  <svg class="icon"><use href="#i-chevron-left"/></svg>
</button>

<!-- Phase 54.6.170 — Command palette. Cmd/Ctrl+K opens; lists every
     toolbar + nav + settings action with fuzzy match. Always in the
     DOM so the shortcut works on every page (Read, Dashboard,
     Chapter Reader, …). -->
<div class="cmdk u-hidden" id="cmdk" role="dialog" aria-label="Command palette">
  <div class="cmdk-scrim" onclick="closeCmdK()"></div>
  <div class="cmdk-box">
    <input class="cmdk-input" id="cmdk-input" type="text"
           placeholder="Jump to an action, a setting, or a panel…"
           autocomplete="off" spellcheck="false" aria-label="Search commands"/>
    <div class="cmdk-list" id="cmdk-list" role="listbox"></div>
    <div class="cmdk-hint">
      <span><kbd>↑</kbd><kbd>↓</kbd> navigate</span>
      <span><kbd>↵</kbd> run</span>
      <span><kbd>Esc</kbd> close</span>
    </div>
  </div>
</div>

<!-- ── Phase 14 modals ─────────────────────────────────────────────────── -->

<!-- Wiki Modal — Phase 15: Query + Browse tabs -->
<div class="modal-overlay" id="wiki-modal" onclick="if(event.target===this)closeModal('wiki-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-book-open"/></svg> Compiled Knowledge Wiki</h3>
      <button class="modal-close" onclick="closeModal('wiki-modal')" title="Close the Compiled Knowledge Wiki modal.">&times;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="wiki-query" onclick="switchWikiTab('wiki-query')"
              title="Ask a natural-language question against the compiled wiki summaries. Fastest mode — no retrieval over raw chunks.">&#128270; Query</button>
      <button class="tab" data-tab="wiki-summaries" onclick="switchWikiTab('wiki-summaries')"
              title="Browse every per-paper LLM-written summary from `sciknow wiki compile`.">&#128196; Summaries</button>
      <button class="tab" data-tab="wiki-visuals" onclick="switchWikiTab('wiki-visuals')"
              title="Figures, tables, equations and code snippets extracted from papers. Rendered inline with captions.">&#128444;&#65039; Visuals</button>
      <button class="tab" data-tab="wiki-browse" onclick="switchWikiTab('wiki-browse')"
              title="Browse the full wiki page tree: paper pages, concept pages and synthesis pages.">&#128194; Browse pages</button>
      <button class="tab" data-tab="wiki-lint" onclick="switchWikiTab('wiki-lint')"
              title="Health check for the wiki: broken links, stale pages, orphan concepts, missing KG triples, contradictions.">&#9888;&#65039; Lint</button>
      <button class="tab" data-tab="wiki-consensus" onclick="switchWikiTab('wiki-consensus')"
              title="Build a consensus map for a topic: strong / moderate / weak / contested claims across the corpus.">&#9878;&#65039; Consensus</button>
    </div>
    <div class="modal-body">
      <!-- Query tab -->
      <div class="tab-pane active" id="wiki-query-pane">
        <div class="field">
          <label>Question</label>
          <input type="text" id="wiki-query-input" placeholder="What does the wiki say about ..."
                 onkeydown="if(event.key==='Enter')doWikiQuery()"
                 title="Natural-language question against the pre-compiled wiki summaries (one per paper). Press Enter to search."/>
        </div>
        <div class="field">
          <button class="btn-primary" onclick="doWikiQuery()"
                  title="Query the wiki summaries. Faster than Ask Corpus; returns a synthesized answer + list of source summaries.">Search Wiki</button>
        </div>
        <div class="u-note-sm" id="wiki-status"></div>
        <div class="modal-stream" id="wiki-stream"></div>
        <div id="wiki-stream-stats" class="stream-stats"></div>
        <div class="modal-sources u-hidden" id="wiki-sources"></div>
      </div>
      <!-- Phase 54.6.61 — Summaries tab: dedicated paper-summary browser. -->
      <div class="tab-pane u-hidden" id="wiki-summaries-pane">
        <div class="u-flex-raw u-gap-10 u-ai-center u-wrap u-mb-m">
          <input type="text" id="wiki-sum-search" placeholder="Filter by title or author…"
                 style="flex:1;min-width:200px;padding:6px 10px;"
                 oninput="renderWikiSummaries()"
                 title="Substring match against paper title, authors, or slug. Filters live as you type."/>
          <label class="u-hint-sm">Sort:</label>
          <select class="u-pill-md" id="wiki-sum-sort" onchange="renderWikiSummaries()"
                  title="How to order the summary list.">
            <option value="year_desc">Year (newest first)</option>
            <option value="year_asc">Year (oldest first)</option>
            <option value="updated_desc">Recently compiled</option>
            <option value="title_asc">Title A-Z</option>
            <option value="words_desc">Word count</option>
          </select>
          <span class="u-hint" id="wiki-sum-count"></span>
        </div>
        <div class="u-modal-scroll" id="wiki-summaries-list"></div>
      </div>
      <!-- Phase 54.6.61 — Visuals tab: figures / equations / tables / code,
           with actual image rendering for figures via /api/visuals/image. -->
      <div class="tab-pane u-hidden" id="wiki-visuals-pane">
        <div class="u-row-wrap-mb">
          <select class="u-pill-md" id="wiki-vis-kind" onchange="loadWikiVisuals()"
                  title="Which visual kind to show. Figures/charts render as thumbnails; equations as KaTeX; tables as HTML; code in a pre block.">
            <option value="figure">Figures (thumbnails)</option>
            <option value="chart">Charts (thumbnails)</option>
            <option value="table">Tables</option>
            <option value="equation">Equations</option>
            <option value="code">Code</option>
          </select>
          <input type="text" id="wiki-vis-search" placeholder="Search captions…"
                 style="flex:1;min-width:160px;padding:4px 8px;"
                 onkeydown="if(event.key==='Enter')loadWikiVisuals()"
                 title="Substring match against caption + surrounding text. Press Enter to search."/>
          <input class="u-w-70 u-p-4-6" type="number" id="wiki-vis-limit" value="60" min="10" max="500" step="10"
                 title="How many visuals to load per click. Raise to 200+ for a full-corpus review; 60 is a fast browse size.">
          <button class="btn-secondary" onclick="loadWikiVisuals()"
                  title="Fetch visuals matching the current filters.">&#128269; Load</button>
          <span class="u-hint" id="wiki-vis-stats"></span>
        </div>
        <div class="u-modal-scroll" id="wiki-visuals-list"></div>
      </div>
      <!-- Browse tab -->
      <div class="tab-pane u-hidden" id="wiki-browse-pane">
        <div class="field u-row-end">
          <div class="u-flex-1">
            <label>Filter by type</label>
            <select id="wiki-type-filter" onchange="loadWikiPages(1)"
                    title="Filter pages by kind: Paper (one per ingested paper), Concept (entity glossary), Synthesis (consensus maps).">
              <option value="">All types</option>
            </select>
          </div>
          <button class="btn-secondary" onclick="loadWikiPages(1)"
                  title="Re-fetch the wiki page list from the server.">Refresh</button>
        </div>
        <div class="u-mt-3" id="wiki-browse-list"></div>
        <!-- Detail view (hidden until a page is opened) -->
        <div class="u-hidden" id="wiki-page-detail">
          <div class="wiki-detail-toolbar">
            <button class="btn-secondary" onclick="closeWikiPageDetail()"
                    title="Return to the wiki page list.">&larr; Back to list</button>
            <button class="btn-secondary" onclick="copyWikiPermalink()" title="Copy permalink to this page">&#128279; Copy link</button>
            <kbd class="wiki-kbd-hint">Press <kbd>Ctrl</kbd>+<kbd>K</kbd> to jump to any page</kbd>
          </div>
          <div id="wiki-page-meta" style="font-size:11px;color:var(--fg-muted);margin:8px 0 12px 0;"></div>
          <div id="wiki-stale-banner" class="wiki-stale-banner u-hidden">
            &#9888;&#65039; This page is flagged as stale — run
            <code>sciknow wiki compile --rewrite-stale</code>
            to refresh it from the current sources.
          </div>
          <div class="wiki-detail-layout">
            <aside id="wiki-toc" class="wiki-toc"></aside>
            <div>
              <div id="wiki-page-content" class="wiki-page-content"></div>
              <!-- Phase 54.4 — "Facts from the corpus" on concept pages -->
              <section id="wiki-facts-block" class="wiki-extras u-hidden">
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
                  <span id="wiki-annotation-ts" class="wiki-facts-kglink u-muted"></span>
                </div>
                <textarea id="wiki-annotation-body"
                          placeholder="Your own notes on this page — disagreements, follow-up questions, how it connects to other work. Saved locally in your project database."
                          rows="4"></textarea>
                <div class="wiki-annotation-actions">
                  <button class="btn-primary" onclick="saveWikiAnnotation()" id="wiki-annotation-save"
                          title="Save your personal note to the project database. Only visible in this project.">Save note</button>
                  <button class="btn-secondary" onclick="deleteWikiAnnotation()" id="wiki-annotation-delete"
                          title="Delete this page's annotation. Does NOT affect the wiki content itself.">Clear</button>
                  <span id="wiki-annotation-status" class="wiki-ask-status"></span>
                </div>
              </section>
              <!-- Phase 54.3 — Ask this page inline RAG -->
              <section id="wiki-ask-block" class="wiki-extras wiki-ask-extras">
                <h3 class="wiki-extras-h">Ask a question about this page</h3>
                <form class="wiki-ask-form" onsubmit="event.preventDefault(); askWikiPage();">
                  <input type="text" id="wiki-ask-input"
                         placeholder="e.g. What effect size is reported?"
                         autocomplete="off"
                         title="Inline RAG: ask a question scoped to this page's source chunks (press Enter to submit)."/>
                  <label class="wiki-ask-broaden" title="Search the whole corpus instead of just this page's sources">
                    <input type="checkbox" id="wiki-ask-broaden"
                           title="Expand retrieval from this page's sources to the full corpus."/> broaden
                  </label>
                  <button type="submit" class="btn-primary" id="wiki-ask-submit"
                          title="Run the scoped RAG query and stream the answer inline below.">Ask</button>
                </form>
                <div id="wiki-ask-status" class="wiki-ask-status"></div>
                <div id="wiki-ask-stream" class="wiki-ask-stream"></div>
                <div id="wiki-ask-sources" class="wiki-ask-sources u-hidden"></div>
              </section>
              <section id="wiki-related-block" class="wiki-extras u-hidden">
                <h3 class="wiki-extras-h">Related pages</h3>
                <ol id="wiki-related-list" class="wiki-compact-list"></ol>
              </section>
              <section id="wiki-backlinks-block" class="wiki-extras u-hidden">
                <h3 class="wiki-extras-h">Referenced by</h3>
                <ol id="wiki-backlinks-list" class="wiki-compact-list"></ol>
              </section>
            </div>
          </div>
        </div>
      </div>
      <!-- Phase 54.6.2 — Lint tab: surfaces `sciknow wiki lint` in the GUI. -->
      <div class="tab-pane u-hidden" id="wiki-lint-pane">
        <div class="u-note-sm">
          Check wiki health: broken links, stale pages, orphaned concepts,
          missing summaries, and optionally contradictions across paper
          summaries (deep mode uses the LLM — slower).
        </div>
        <div class="field u-row-wrap">
          <label class="u-label-row"
                 title="Also run LLM-based contradiction detection across paper summaries. Significantly slower (one LLM call per concept).">
            <input type="checkbox" id="wiki-lint-deep"
                   title="Also run LLM-based contradiction detection. Significantly slower (one LLM call per concept)."> deep (LLM contradiction detection)
          </label>
          <button class="btn-primary" id="wiki-lint-run" onclick="doWikiLint()"
                  title="Scan the wiki for broken links, stale pages, orphan concepts and missing KG triples.">Run Lint</button>
          <button class="btn-secondary u-hidden" id="wiki-lint-stop" onclick="stopWikiLint()"
                  title="Cancel the running lint job.">Stop</button>
        </div>
        <div class="u-caption" id="wiki-lint-status"></div>
        <div class="u-mt-2" id="wiki-lint-summary"></div>
        <div id="wiki-lint-issues" style="margin-top:10px;max-height:calc(70vh - 160px);overflow:auto;"></div>
        <!-- Phase 54.6.8 — KG backfill utility. If the Knowledge Graph
             modal is empty for your corpus, this is the fix: older wiki
             compiles didn't run the combined entity+KG extraction, so
             paper pages can exist with zero triples. This walks those
             orphans and runs ONLY the extraction step (no re-summarize). -->
        <div style="margin-top:20px;padding:10px;border:1px solid var(--border);border-radius:6px;background:var(--bg-alt,#f8f8f8);">
          <div class="u-note-sm">
            <strong>Backfill KG triples.</strong> Papers with a wiki page but no
            <code>knowledge_graph</code> rows — usually the result of an older
            <code>wiki compile</code> that predates the combined entity+KG
            extraction step. Runs one LLM call per orphan paper (no
            re-summarizing). Mirrors <code>sciknow wiki extract-kg</code>.
          </div>
          <div class="u-row-wrap">
            <label class="u-label-row"
                   title="Re-run KG extraction on every paper, not just those with zero triples. Expensive — use only after changing the extraction prompt or model.">
              <input type="checkbox" id="wiki-extractkg-force"
                     title="Re-run KG extraction on every paper, not just those with zero triples."> force re-extract every paper
            </label>
            <button class="btn-primary" id="wiki-extractkg-run"
                    onclick="doWikiExtractKg()"
                    title="One LLM call per orphan paper (or all papers if force is checked) to populate knowledge_graph rows. No re-summarization.">Extract / Backfill KG</button>
          </div>
          <div class="u-caption" id="wiki-extractkg-status"></div>
          <pre id="wiki-extractkg-log" style="display:none;margin-top:6px;max-height:280px;overflow:auto;background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:8px;font-size:11px;font-family:ui-monospace,monospace;white-space:pre-wrap;"></pre>
        </div>
      </div>
      <!-- Phase 54.6.2 — Consensus tab: surfaces `sciknow wiki consensus` in the GUI. -->
      <div class="tab-pane u-hidden" id="wiki-consensus-pane">
        <div class="u-note-sm">
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
                 onkeydown="if(event.key==='Enter')doWikiConsensus()"
                 title="Free-text topic for the consensus map. Press Enter to run. The LLM will classify claims as strong / moderate / weak / contested and cite supporting vs contradicting papers.">
        </div>
        <div class="u-row-gap-sm">
          <button class="btn-primary" id="wiki-consensus-run" onclick="doWikiConsensus()"
                  title="Build the consensus map and save it as a synthesis page under /synthesis/.">Map Consensus</button>
          <button class="btn-secondary u-hidden" id="wiki-consensus-stop" onclick="stopWikiConsensus()"
                  title="Cancel the running consensus job.">Stop</button>
        </div>
        <div class="u-caption" id="wiki-consensus-status"></div>
        <div class="u-mt-2 u-md" id="wiki-consensus-summary"></div>
        <div id="wiki-consensus-claims" style="margin-top:10px;max-height:calc(70vh - 180px);overflow:auto;"></div>
        <div class="u-mt-10" id="wiki-consensus-debated"></div>
      </div>
    </div>
  </div>
</div>

<!-- Phase 54.1 + 54.6.48 — KaTeX math rendering. Vendored locally
     under sciknow/web/static/vendor/katex/ (see vendor/README.md for
     origin + versions). Loading from /static/ instead of jsDelivr
     eliminates the third-party CDN ping + makes equations work
     offline. ~90KB JS + ~60KB CSS gzipped + ~800KB of woff2 fonts. -->
<link rel="stylesheet" href="/static/vendor/katex/katex.min.css"/>
<script defer src="/static/vendor/katex/katex.min.js"></script>
<script defer src="/static/vendor/katex/contrib/auto-render.min.js"></script>
<!-- Phase 54.6.12 + 54.6.48 — ECharts 5 for the Visualize modal.
     Vendored locally. One library covers all six tabs (scatter,
     sunburst, stacked area, radar,
     polar) with proper pan / zoom / tooltips built in. ~1 MB. -->
<script src="/static/vendor/echarts/echarts.min.js"></script>

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
<div class="modal-overlay u-hidden" id="wiki-palette" onclick="if(event.target===this)closeWikiPalette()">
  <div class="modal wiki-palette-modal">
    <input type="text" id="wiki-palette-input"
           placeholder="Jump to wiki page… (type to filter)"
           autocomplete="off"
           oninput="_renderWikiPalette()"
           onkeydown="_wikiPaletteKey(event)"
           title="Fuzzy-search every wiki page by title. Arrow keys + Enter to open, Esc to close."/>
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
      <h3><svg class="icon icon--lg"><use href="#i-search"/></svg> Ask the Corpus (RAG)</h3>
      <button class="modal-close" onclick="closeModal('ask-modal')" title="Close the Ask the Corpus modal.">&times;</button>
    </div>
    <div class="modal-body">
      <div class="field">
        <label>Question</label>
        <input type="text" id="ask-input" placeholder="What are the main mechanisms of ..."
               onkeydown="if(event.key==='Enter')doAsk()"
               title="Natural-language question. Hybrid retrieval (dense + sparse + FTS) → RRF fusion → reranker → LLM answer. Press Enter to submit.">
      </div>
      <div class="field u-flex-raw u-gap-2">
        <div class="u-flex-1">
          <label>Year from</label>
          <input type="number" id="ask-year-from" placeholder="(optional)"
                 title="Restrict retrieval to papers published in this year or later.">
        </div>
        <div class="u-flex-1">
          <label>Year to</label>
          <input type="number" id="ask-year-to" placeholder="(optional)"
                 title="Restrict retrieval to papers published in this year or earlier.">
        </div>
      </div>
      <div class="field">
        <button class="btn-primary" onclick="doAsk()"
                title="Run the RAG pipeline and stream the answer + source citations inline below.">Ask</button>
        <span class="u-note-ml">Hybrid retrieval + bge-reranker + LLM</span>
      </div>
      <div class="u-note-sm" id="ask-status"></div>
      <div class="modal-stream" id="ask-stream"></div>
      <div id="ask-stream-stats" class="stream-stats"></div>
      <div class="modal-sources u-hidden" id="ask-sources"></div>
    </div>
  </div>
</div>

<!-- Phase 33 — Autowrite Configuration Modal (replaces the old triple-prompt UX) -->
<div class="modal-overlay" id="autowrite-config-modal" onclick="if(event.target===this)closeModal('autowrite-config-modal')">
  <div class="modal" style="max-width:480px;">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-zap"/></svg> Autowrite</h3>
      <button class="modal-close" onclick="closeModal('autowrite-config-modal')" title="Close the Autowrite Config modal.">&times;</button>
    </div>
    <div class="modal-body">
      <p id="aw-config-scope" style="font-size:13px;color:var(--fg);margin-bottom:16px;font-weight:600;"></p>
      <div class="field">
        <label>Max iterations per section</label>
        <input class="u-w-80" type="number" id="aw-config-max-iter" value="3" min="1" max="10"
               title="Upper bound on revise cycles per section. Each iteration: write → score → verify → (CoVe) → revise → rescore. Default 3 is usually enough; raise to 5 for tough sections."/>
        <span class="u-note-ml">Each iteration: score &rarr; verify &rarr; revise</span>
      </div>
      <div class="field">
        <label>Target score (0.0 &ndash; 1.0)</label>
        <input class="u-w-80" type="number" id="aw-config-target-score" value="0.85" min="0" max="1" step="0.05"
               title="Early-stop threshold on the scorer's `overall` dimension. Autowrite stops as soon as it hits this, before exhausting Max iterations. 0.85 is conservative; 0.8 is faster, 0.9 requires very clean drafts."/>
        <span class="u-note-ml">Stop iterating when overall &ge; this</span>
      </div>
      <div class="u-mt-4" id="aw-config-mode-section">
        <label>Existing drafts</label>
        <p id="aw-config-mode-info" style="font-size:12px;color:var(--fg-muted);margin:4px 0 10px;"></p>
        <div class="u-flex-raw u-gap-2 u-wrap">
          <button class="btn-secondary aw-mode-btn active" data-mode="skip" onclick="selectAwMode('skip')" title="Only fill sections that don't have a draft yet">Skip (fill missing)</button>
          <button class="btn-secondary aw-mode-btn" data-mode="rebuild" onclick="selectAwMode('rebuild')" title="Overwrite all sections from scratch">Rebuild</button>
          <button class="btn-secondary aw-mode-btn" data-mode="resume" onclick="selectAwMode('resume')" title="Load existing content + run more iterations">Resume</button>
        </div>
      </div>
      <!-- Phase 54.6.144 — visuals-in-writer opt-in -->
      <div class="field" style="margin-top:18px;padding-top:12px;border-top:1px dashed var(--border);">
        <label class="u-row-click"
               title="Phase 54.6.142 visuals-in-writer. When on, the 5-signal ranker surfaces figures/tables to the writer, and the prompt includes the 'cite [Fig. N] only when directly depicted' gated instruction. Adds a visual_citation scoring dimension (hallucinated markers = hard 0.0, missed opportunities = 0.5). Level-1 + Level-2 verify run per iteration; Level-3 VLM claim-depiction is deferred to the finalize-draft pass. Default off so existing runs are untouched.">
          <input type="checkbox" id="aw-config-include-visuals"
                 title="Include visuals (figures / tables / equations) in the writer's retrieval pool. See docs/RESEARCH.md §7.X."/>
          <span>&#128206; Include visuals in the writer</span>
        </label>
        <p style="font-size:11px;color:var(--fg-muted);margin:4px 0 0 28px;line-height:1.4;">
          Writer gets a shortlist of ranked figures/tables per section and may cite them as
          <code>[Fig.&nbsp;N]</code> when the claim is directly depicted. Verify pass catches hallucinated
          markers each iteration. ~+0.7s per section on CPU.
        </p>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn-secondary" onclick="closeModal('autowrite-config-modal')"
              title="Close without starting autowrite.">Cancel</button>
      <button class="btn-primary" onclick="confirmAutowrite()"
              title="Start autowrite with the current settings. Streams tokens live; close the modal at any time — the job keeps running server-side and the persistent task bar reflects progress.">&#9889; Start</button>
    </div>
  </div>
</div>

<!-- Phase 14.3 — Book Plan Modal
     Phase 21 — context-aware: tabs for Book / Chapter / Section -->
<div class="modal-overlay" id="plan-modal" onclick="if(event.target===this)closeModal('plan-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-file-text"/></svg> Plans</h3>
      <button class="modal-close" onclick="closeModal('plan-modal')" title="Close the Plans modal.">&times;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="plan-book" onclick="switchPlanTab('plan-book')"
              title="Book-level plan / leitmotiv — 200–500 words defining thesis, scope, audience, key terms. Injected into every writer prompt.">Book</button>
      <button class="tab" data-tab="plan-outline" onclick="switchPlanTab('plan-outline')" id="plan-tab-outline"
              title="LLM-generated high-level chapter outline. Edit here to reshape the book before drafting.">Outline</button>
      <button class="tab" data-tab="plan-chapters" onclick="switchPlanTab('plan-chapters')" id="plan-tab-chapters"
              title="List + reorder chapters. Each chapter has its own scope description and target word count.">Chapters</button>
      <button class="tab" data-tab="plan-chapter" onclick="switchPlanTab('plan-chapter')" id="plan-tab-chapter"
              title="Edit sections within the active chapter. Each section becomes its own draft when you Write / Autowrite.">Sections</button>
    </div>
    <div class="modal-body">
      <!-- Book tab — the leitmotiv (existing) -->
      <div class="tab-pane active" id="plan-book-pane">
        <p class="u-note-lg">
          The book plan is a 200&ndash;500 word document defining the central thesis,
          scope, intended audience, and key terms. It is injected into every
          <code>book write</code> / <code>autowrite</code> call so all chapters stay
          aligned with the same argument. Edit it manually below, or click
          <strong>Regenerate with LLM</strong> to draft a new one from your chapters
          and paper corpus.
        </p>
        <div class="field">
          <label>Book title</label>
          <input type="text" id="plan-title-input"
                 title="The book's display title. Shown in the sidebar, browser tab, and every export."/>
        </div>
        <div class="field">
          <label>Short description (one or two sentences)</label>
          <textarea id="plan-desc-input" style="min-height:50px;"
                    title="One-line blurb shown in the catalog + stats. Not injected into writer prompts."></textarea>
        </div>
        <div class="field">
          <label>Plan / leitmotiv (the full thesis &amp; scope document)</label>
          <textarea id="plan-text-input" style="min-height:280px;font-family:var(--font-serif);font-size:14px;line-height:1.6;"
                    title="200–500 word thesis document. Injected into every `write` / `autowrite` prompt so all chapters stay aligned with the same argument. Most important field for writer consistency."></textarea>
        </div>
        <div class="field">
          <label>Target chapter length &mdash; autowrite &amp; write aim for this many words per chapter</label>
          <div class="u-row-wrap">
            <button type="button" class="btn-secondary" onclick="setLengthPreset(3000)"
                    title="3000 words per chapter — blog-post length. About 750 words per section in a 4-section chapter.">Short &middot; 3000</button>
            <button type="button" class="btn-secondary" onclick="setLengthPreset(6000)"
                    title="6000 words per chapter (sciknow default). About 1500 words per section in a 4-section chapter.">Standard &middot; 6000</button>
            <button type="button" class="btn-secondary" onclick="setLengthPreset(10000)"
                    title="10 000 words per chapter — substantial chapter-of-record length. About 2500 words per section in a 4-section chapter.">Long &middot; 10000</button>
            <input type="number" id="plan-target-words-input" min="0" step="500" placeholder="custom"
                   style="width:120px;padding:6px 8px;font-size:13px;"
                   title="Words per chapter. Leave empty for the default (6000). Zero clears the setting.">
            <span class="u-hint-sm" id="plan-length-status"></span>
          </div>
          <p class="u-note-mt-6">
            Each section gets a proportional share: a 4-section chapter at 6000
            words asks the writer for ~1500 words per section. In autowrite,
            length becomes a 7th scoring dimension &mdash; drafts under ~70% of
            target trigger a targeted expansion revision.
          </p>
        </div>
      </div>
      <!-- Phase 54.6.96 — Outline tab: promoted from a footer button
           into its own pane so the user has a clear home for chapter
           STRUCTURE generation (distinct from Book's leitmotiv and
           from per-draft Review). Mirrors `sciknow book outline`. -->
      <div class="tab-pane u-hidden" id="plan-outline-pane">
        <p class="u-note-lg">
          <strong>Outline</strong> proposes a chapter structure for the book from your paper
          corpus. The LLM generates 3 candidate outlines (temperature-diversified),
          scores each for breadth + section-count variance, picks the winner, and
          then density-resizes each chapter&rsquo;s section list by counting corpus
          evidence per topic. Insert is <strong>additive</strong>: existing chapters
          and drafts are never touched &mdash; new chapters are appended with fresh
          numbers. Run again to re-roll when the first pass isn&rsquo;t right.
        </p>
        <p class="u-tiny u-muted u-bg-alt-raw u-p-8-10 u-r-md u-mb-14">
          <strong>Outline vs Review:</strong>
          <em>Outline</em> plans the book&rsquo;s chapter/section structure from the corpus (no drafts needed).
          <em>Review</em> (in the draft toolbar) critiques the prose in a single existing draft
          across 5 dimensions (groundedness, completeness, accuracy, coherence, redundancy).
          Different inputs, different outputs, different stages of the workflow.
        </p>
        <div class="field">
          <label>Elicitation method (optional)
            <span class="u-hint">&mdash; steers the LLM through a named cognitive technique.</span>
          </label>
          <select class="u-w-full u-pill-lg" id="plan-outline-method-select"
                  title="Optional elicitation method prepended to the outline prompt as a preamble. Steers the LLM through a named cognitive technique (e.g. Tree of Thoughts, First Principles, Peer Review Simulation). See docs/BOOK_ACTIONS.md for the full 24-method catalogue.">
            <option value="">(default generic prompt)</option>
          </select>
        </div>
        <div class="field">
          <label>Model override (optional)
            <span class="u-hint">&mdash; leave empty to use <code>LLM_MODEL</code>.</span>
          </label>
          <input class="u-w-full u-pill-lg u-mono u-small" type="text" id="plan-outline-model-input" placeholder="(leave empty for default)"
                 title="Optional Ollama model tag to override the default (LLM_MODEL). Useful for experimenting with a different planner without changing .env. Leave empty to use the configured default."/>
        </div>
        <div class="u-flex-raw u-gap-2 u-ai-center u-mb-3 u-wrap">
          <button class="btn-primary" id="plan-outline-run-btn" onclick="runOutlineFromTab()"
                  title="Generate 3 candidate chapter outlines at rising temperatures, score by breadth + section-count variance, pick the winner, density-resize sections, and add to book_chapters. Additive: never touches existing chapters. Mirrors `sciknow book outline`.">&#128214; Generate outline</button>
          <button class="btn-secondary u-hidden" id="plan-outline-cancel-btn" onclick="cancelOutline()"
                  title="Cancel the running outline generation. Any chapters already committed stay; the in-flight LLM call is aborted.">Cancel</button>
          <span class="u-hint-sm" id="plan-outline-status"></span>
        </div>
        <div id="plan-outline-stream" style="display:none;max-height:200px;overflow:auto;font-family:var(--font-mono);font-size:11px;line-height:1.4;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:8px;white-space:pre-wrap;margin-bottom:12px;"></div>
        <div class="u-small" id="plan-outline-result"></div>
      </div>
      <!-- Phase 54.6.66 — Chapters tab (book-wide chapter manager):
           list all chapters with editable title / description /
           topic_query, ↑/↓ reorder, delete, and an Add-chapter row.
           Uses existing /api/chapters, /api/chapters/{{id}},
           /api/chapters/reorder, DELETE /api/chapters/{{id}}. -->
      <div class="tab-pane u-hidden" id="plan-chapters-pane">
        <p class="u-note-lg">
          Book-wide chapter plan: every chapter&rsquo;s title, description, and
          retrieval query. Reorder with &uarr; / &darr;. Delete unlinks the
          chapter&rsquo;s drafts but does not delete them. Section-level editing
          moves to the <strong>Sections</strong> tab.
        </p>
        <div id="plan-chapters-list" style="display:flex;flex-direction:column;gap:10px;"></div>
        <div class="u-mt-3">
          <button class="btn-secondary" onclick="addPlanChapter()"
                  title="Append a new empty chapter to the book at the next chapter number.">+ Add chapter</button>
        </div>
      </div>
      <!-- Sections tab (was "Chapter sections"). Now includes a
           chapter picker at the top so the user can switch between
           chapters without closing the modal. Each section row stays
           inline-editable (plan + target_words). -->
      <div class="tab-pane u-hidden" id="plan-chapter-pane">
        <p class="u-note-lg">
          Per-chapter section plans: pick a chapter, then edit each
          section&rsquo;s plan and target word count. To rename / reorder /
          add / delete sections themselves, use the chapter modal&rsquo;s
          Sections tab.
        </p>
        <div class="u-flex-raw u-gap-2 u-ai-center u-mb-3 u-wrap">
          <label class="u-hint-sm">Chapter:</label>
          <select id="plan-sections-chapter-picker"
                  onchange="onPlanSectionsChapterChange(this.value)"
                  style="flex:1;min-width:200px;padding:4px 8px;"
                  title="Which chapter's sections to edit. Changing the picker saves the current chapter's edits before loading the next.">
            <option value="">(choose a chapter…)</option>
          </select>
        </div>
        <!-- Phase 54.6.163 — duplicate the 54.6.155 Chapter-modal
             auto-plan button here in the Plans modal, because this
             is where users naturally land when they want to plan
             sections (the modal is literally titled "Plans" and has
             a per-section plan editor). Previously the button only
             lived in the Chapter modal's Sections tab, which is a
             different modal with a different tab structure. -->
        <div class="u-flex-raw u-gap-2 u-ai-center u-mb-3 u-wrap u-p-2 u-bg-tb u-border u-r-md">
          <button class="btn-secondary" onclick="planModalAutoPlanSections()"
                  title="Phase 54.6.154/163 — LLM-generate a 3-4 bullet concept plan per empty section in the currently-picked chapter. Bullet counts drive the Phase-54.6.146 concept-density resolver: target_words = N bullets × wpc_midpoint. Skips sections that already have a plan unless 'Force overwrite' is ticked. Cost: ~5-10s per empty section (LLM_FAST_MODEL). See docs/RESEARCH.md §24 and docs/CONCEPT_DENSITY.md.">
            &#129504; Auto-plan sections
          </button>
          <label class="u-chip"
                 title="Overwrite existing plans instead of skipping them.">
            <input type="checkbox" id="plan-auto-plan-force"
                   title="Overwrite existing plans instead of skipping.">
            force overwrite
          </label>
          <span class="u-hint" id="plan-auto-plan-status"></span>
        </div>
        <div class="u-mb-14 u-md u-muted" id="plan-chapter-header"></div>
        <div id="plan-chapter-sections"></div>
      </div>
      <div class="u-note-sm" id="plan-status"></div>
      <div id="plan-stream-stats" class="stream-stats"></div>
    </div>
    <div class="modal-footer u-wrap u-gap-6">
      <button class="btn-secondary" onclick="closeModal('plan-modal')"
              title="Dismiss the Plan modal. Unsaved edits to any tab are discarded.">Close</button>
      <button class="btn-secondary u-ml-auto" onclick="regeneratePlan()" id="plan-regen-btn"
              title="LLM-regenerate the Book plan (leitmotiv) from current chapters + paper corpus. Visible only on the Book tab.">&#9889; Regenerate with LLM</button>
      <button class="btn-primary" onclick="savePlan()"
              title="Persist changes on the currently-visible tab (Book / Chapters / Sections) to the database. Outline-tab changes are committed inline when you click Generate outline.">Save</button>
    </div>
  </div>
</div>

<!-- Phase 14.3 — Chapter Info Modal (description + topic_query)
     Phase 18 — Tabs: Scope (existing) + Sections (new) -->
<div class="modal-overlay" id="chapter-modal" onclick="if(event.target===this)closeModal('chapter-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-sliders"/></svg> Chapter</h3>
      <button class="modal-close" onclick="closeModal('chapter-modal')" title="Close the Chapter modal.">&times;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="ch-scope" onclick="switchChapterTab('ch-scope')"
              title="Edit the chapter title, scope description, and retrieval topic query.">Scope</button>
      <button class="tab" data-tab="ch-sections" onclick="switchChapterTab('ch-sections')"
              title="Manage the sections that make up this chapter. Each section gets its own draft when you Write / Autowrite.">Sections</button>
    </div>
    <div class="modal-body">
      <!-- Scope tab -->
      <div class="tab-pane active" id="ch-scope-pane">
        <p class="u-note-lg">
          The chapter description sets the per-chapter scope: what the chapter
          covers and what stays out. The topic query is a 3&ndash;6 word search
          phrase used to retrieve the most relevant papers from the corpus when
          you click Write or Autowrite for this chapter.
        </p>
        <div class="field">
          <label>Chapter title</label>
          <input type="text" id="ch-title-input"
                 title="Human-readable chapter title. Used in the sidebar, TOC, and exported book.">
        </div>
        <div class="field">
          <label>Description (per-chapter scope)</label>
          <textarea id="ch-desc-input" style="min-height:120px;"
                    title="1–3 sentence scope for this chapter. Tells the writer what belongs IN the chapter and (equally important) what stays out."></textarea>
        </div>
        <div class="field">
          <label>Topic query (retrieval phrase)</label>
          <input type="text" id="ch-tq-input" placeholder="e.g. solar irradiance satellite measurements"
                 title="3–6 word search phrase. Used by hybrid retrieval to pull the most relevant chunks when writing this chapter.">
        </div>
      </div>
      <!-- Sections tab -->
      <div class="tab-pane" id="ch-sections-pane">
        <p class="u-note-lg">
          A chapter is broken into named sections. Each section becomes its own
          draft when you click <strong>Write</strong> or <strong>Autowrite</strong>,
          and gets a proportional share of the chapter&rsquo;s word target. The
          plan tells the writer what THIS section must cover &mdash; narrower
          than the chapter description. Reorder with the &uarr;/&darr; buttons.
          Renaming a section <strong>does not</strong> rename existing drafts;
          they keep their old slug until rewritten.
        </p>
        <div id="ch-sections-list"></div>
        <div class="u-row-mt">
          <button class="btn-secondary" onclick="addSection()"
                  title="Append a new empty section to the end of the chapter. You can rename and reorder after adding.">
            &#43; Add section
          </button>
          <!-- Phase 54.6.155 — wrap `sciknow book plan-sections --chapter N`
               so users can trigger LLM-generated concept plans from the
               modal. Default: skip sections that already have a plan
               (matches the CLI's default behaviour). Force overwrites. -->
          <button class="btn-secondary" onclick="autoPlanChapterSections()"
                  title="Phase 54.6.154/155 — LLM-generate a 3-4 bullet concept plan per empty section, using this chapter's scope + section titles. Bullet counts drive the Phase-54.6.146 concept-density resolver: target_words = N bullets × wpc_midpoint. Skips sections that already have a plan unless 'Force overwrite' is ticked. Cost: ~5-10s per empty section (LLM_FAST_MODEL, cheap structured task). See docs/RESEARCH.md §24.">
            &#129504; Auto-plan sections
          </button>
          <label class="u-chip"
                 title="Overwrite existing plans instead of skipping them. Use when you want to regenerate from scratch — e.g. after changing the chapter description.">
            <input type="checkbox" id="ch-auto-plan-force"
                   title="Overwrite existing plans instead of skipping.">
            force overwrite
          </label>
          <span id="ch-auto-plan-status" style="font-size:12px;color:var(--fg-muted);margin-left:8px;"></span>
        </div>
      </div>
      <div class="u-note-vertical" id="chapter-modal-status"></div>
    </div>
    <div class="modal-footer">
      <button class="btn-secondary" onclick="closeModal('chapter-modal')"
              title="Dismiss without saving. Unsaved edits to title/description/topic are discarded.">Close</button>
      <button class="btn-primary" onclick="saveChapterInfo()"
              title="Persist chapter + section changes to the database.">Save</button>
    </div>
  </div>
</div>

<!-- Catalog Browser Modal -->
<div class="modal-overlay" id="catalog-modal" onclick="if(event.target===this)closeModal('catalog-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-folder"/></svg> Browse Papers</h3>
      <button class="modal-close" onclick="closeModal('catalog-modal')" title="Close the Browse Papers modal.">&times;</button>
    </div>
    <div class="modal-body">
      <div class="field u-row-end">
        <div class="u-flex-2">
          <label>Author</label>
          <input type="text" id="cat-author" placeholder="(any)"
                 title="Substring match against the authors field. Partial names are fine (e.g. 'Lockwood' will match 'M. Lockwood').">
        </div>
        <div class="u-flex-2">
          <label>Journal</label>
          <input type="text" id="cat-journal" placeholder="(any)"
                 title="Substring match against the journal/venue name. Useful to e.g. filter 'Nature' papers.">
        </div>
        <div class="u-flex-1">
          <label>Year from</label>
          <input type="number" id="cat-year-from"
                 title="Only show papers from this year onward.">
        </div>
        <div class="u-flex-1">
          <label>Year to</label>
          <input type="number" id="cat-year-to"
                 title="Only show papers up to and including this year.">
        </div>
        <button class="btn-primary" onclick="loadCatalog(1)"
                title="Apply the filter and reload the paper list from page 1.">Filter</button>
      </div>
      <div class="u-mt-3" id="catalog-results"></div>
    </div>
  </div>
</div>

<!-- Phase 39 — Consolidated Book Settings modal -->
<div class="modal-overlay" id="book-settings-modal" onclick="if(event.target===this)closeModal('book-settings-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-sliders"/></svg> Book Settings</h3>
      <button class="modal-close" onclick="closeModal('book-settings-modal')" title="Close the Book Settings modal.">&times;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="bs-basics" onclick="switchBookSettingsTab('bs-basics')"
              title="Title, description, book type, target length.">Basics</button>
      <button class="tab" data-tab="bs-leitmotiv" onclick="switchBookSettingsTab('bs-leitmotiv')"
              title="The guiding argument of the book — a 1–3 sentence thesis threaded through every chapter's scoring.">Leitmotiv</button>
      <button class="tab" data-tab="bs-style" onclick="switchBookSettingsTab('bs-style')"
              title="Tone, voice, reading level and persona guide used by the writer.">Style</button>
      <button class="tab" data-tab="bs-models" onclick="switchBookSettingsTab('bs-models')"
              title="Per-role model assignments (writer, scorer, verifier, reviewer, extractor). Overrides .env defaults.">Models</button>
      <button class="tab" data-tab="bs-view" onclick="switchBookSettingsTab('bs-view')"
              title="Local UI preferences — auto-hide the chapters / comments columns on page load. Stored per browser (localStorage), not per book.">View</button>
    </div>
    <div class="modal-body">

      <!-- Basics tab -->
      <div id="bs-basics-pane">
        <p class="u-note-mb-3">
          Persistent per-book settings. Writes through
          <code>PUT /api/book</code>; empty values leave the field untouched.
        </p>
        <div class="field">
          <label>Title</label>
          <input type="text" id="bs-title" placeholder="(required)"
                 title="The book's display title. Sidebar + browser tab + export filename are derived from this."/>
        </div>
        <div class="field">
          <label>Description</label>
          <input type="text" id="bs-description" placeholder="One-line blurb shown in catalog / stats"
                 title="Free-text blurb for the catalog + stats. NOT injected into writer prompts — use the Leitmotiv tab for thesis-level guidance."/>
        </div>
        <div class="field u-row-end">
          <div class="u-flex-1">
            <label>Project type</label>
            <select id="bs-book-type" onchange="bsUpdateTypeInfo()"
                    title="Phase 54.6.148 — change the book type. Drives autowrite defaults via the fallback chain (per-section > per-chapter > book custom_metadata > project-type default > hardcoded 6000). Changing the type doesn't touch explicit overrides you've already set; it only shifts the Level-3 fallback."/>
          </div>
          <div class="u-flex-1">
            <label>Target words per chapter</label>
            <input type="number" id="bs-target-chapter-words" min="0" step="500"
                   placeholder="(type default)"
                   title="Autowrite and write use this as the per-chapter word target. Leave blank to inherit the project-type default (shown in the info panel below). Set 0 to clear. Per-section share = chapter target ÷ section count — unless concept-density sizing (54.6.146) fires, which overrides when a section has a bullet plan."/>
          </div>
        </div>
        <!-- Phase 54.6.148 — same concept-density info panel as the wizard. -->
        <div id="bs-book-type-info"
             style="margin-top:6px;padding:10px 12px;background:var(--toolbar-bg);border:1px solid var(--border);border-radius:6px;font-size:12px;line-height:1.5;color:var(--fg);">
          <em class="u-muted">Loading type info…</em>
        </div>
        <p class="u-note-top">
          Sections with a bullet plan auto-size bottom-up (concept count × wpc midpoint).
          Sections without a plan fall back to chapter target ÷ section count.
          Override per section in the Chapter modal's Sections tab.
          See <code>docs/RESEARCH.md §24</code> for the research behind these ranges.
        </p>
        <!-- Phase 54.6.156 — book-wide auto-plan wrapper for the 54.6.154 CLI. -->
        <div class="u-divider">
          <div class="u-row-wrap">
            <button class="btn-secondary" onclick="autoPlanEntireBook()"
                    title="Phase 54.6.156 — iterate every chapter × every empty section and ask LLM_FAST_MODEL for a 3-4 bullet concept plan per section. Uses the same generator as the Chapter modal's per-chapter button (54.6.155) but scoped to the whole book. Cost: ~5-10s per empty section (typical book ≈ 4-8 min). Streams progress into a log panel. Skips sections that already have a plan unless 'Force overwrite' is ticked. Activates the Phase-54.6.146 concept-density resolver across the book in one click.">
              &#129504; Auto-plan entire book
            </button>
            <label class="u-chip"
                   title="Overwrite existing plans instead of skipping them.">
              <input type="checkbox" id="bs-plan-book-force"
                     title="Overwrite existing plans instead of skipping.">
              force overwrite
            </label>
          </div>
          <p style="font-size:11px;color:var(--fg-muted);margin:6px 0 0 4px;line-height:1.4;">
            One-click adoption for the concept-density resolver: empty sections become
            bottom-up-sized (bullet count × wpc midpoint) instead of top-down (chapter ÷ sections).
            Run <code>sciknow book length-report</code> before/after to see the shift.
          </p>
          <div class="u-mt-6 u-tiny u-muted" id="bs-plan-book-status"></div>
          <pre id="bs-plan-book-log"
               style="display:none;margin-top:6px;max-height:220px;overflow:auto;padding:8px;background:var(--toolbar-bg);border:1px solid var(--border);border-radius:4px;font-size:10px;font-family:ui-monospace,monospace;line-height:1.3;white-space:pre-wrap;"></pre>
        </div>
        <!-- Phase 54.6.162 — projected length-report panel. GUI wrapper for
             `sciknow book length-report` (54.6.153). Lets users see the
             whole book's per-chapter + per-section projected target +
             resolver level without leaving the GUI. -->
        <div class="u-divider">
          <div class="u-row-wrap">
            <strong class="u-small">&#128196; Projected length report</strong>
            <button class="btn-secondary u-pill-xs" onclick="loadBookLengthReportPanel()"
                    title="Phase 54.6.153/162 — walks every chapter × every section through the resolver chain (per-section override → concept-density → chapter-split) and shows the target + level + explanation per section, plus chapter and book totals. No resolver arithmetic duplication — delegates to the real helpers.">
              refresh
            </button>
          </div>
          <div class="u-mt-6 u-tiny" id="bs-length-report-panel">
            <em class="u-muted">Click refresh to compute the whole-book projection…</em>
          </div>
        </div>
        <!-- Phase 54.6.159 — corpus-grounded section-length panel.
             Surfaces the 54.6.157 bench data (per-section IQRs with
             §24 alignment tags) inline so users don't need to run
             the CLI to see where their corpus sits vs reference. -->
        <div class="u-divider">
          <div class="u-row-wrap">
            <strong class="u-small">&#128202; Corpus section-length distribution</strong>
            <button class="btn-secondary u-pill-xs" onclick="loadSectionLengthPanel()"
                    title="Phase 54.6.157/159 — walks paper_sections.word_count and shows per-section IQR alongside the RESEARCH.md §24 PubMed reference (N=61,517). Use to check whether your corpus is paper-shaped, monograph-shaped, or mixed — informs whether the project-type default wpc is right for your data.">
              refresh
            </button>
          </div>
          <div class="u-mt-6 u-tiny" id="bs-section-length-panel">
            <em class="u-muted">Click refresh to load section-length IQRs…</em>
          </div>
        </div>
        <div class="u-mt-10 u-tiny u-muted" id="bs-basics-meta"></div>
        <div class="u-flex-raw u-gap-2 u-ai-center u-mt-14">
          <button class="btn-primary" onclick="saveBookSettings('basics')"
                  title="Save title / description / target words. Empty fields are left unchanged.">Save Basics</button>
          <span class="u-hint-sm" id="bs-basics-status"></span>
        </div>
      </div>

      <!-- Leitmotiv tab (the book plan) -->
      <div class="u-hidden" id="bs-leitmotiv-pane">
        <p class="u-note">
          The book's thesis / scope document (200&ndash;500 words). Injected into
          every writer prompt so chapter sections stay aligned with the overall
          argument. Use the <strong>&#128221; Plan</strong> quick-editor for
          regeneration; this tab is for direct editing.
        </p>
        <div class="field">
          <label>Plan / leitmotiv</label>
          <textarea id="bs-plan" rows="16" style="font-family:var(--font-sans,inherit);font-size:13px;line-height:1.55;"
                    title="The book's thesis / scope document (200–500 words). Injected into every writer prompt. Same field edited by the ⚡ Plan quick-editor + regenerate button; this tab is for direct editing."></textarea>
        </div>
        <div class="u-flex-raw u-gap-2 u-ai-center u-mt-2">
          <button class="btn-primary" onclick="saveBookSettings('leitmotiv')"
                  title="Save the leitmotiv / plan text to the database. Future `write` / `autowrite` calls use the new value immediately.">Save Plan</button>
          <span class="u-hint-sm" id="bs-leitmotiv-status"></span>
        </div>
      </div>

      <!-- 54.6.106 — Models tab: read-only overview of the effective
           model assignments for this project. Pulls from /api/settings/models
           so the user sees which LLM is being used for what, and where it's
           configured (.env, config default, or per-role override). -->
      <div class="u-hidden" id="bs-models-pane">
        <p class="u-note-mb-3">
          Current model assignments. Edit these in the project's <code>.env</code> file
          (see <a href="https://github.com/claudenstein/sciknow/blob/main/docs/BOOK_ACTIONS.md" target="_blank">BOOK_ACTIONS.md</a>
          for per-role guidance) and restart <code>sciknow book serve</code> to apply.
          Per-section model overrides are also available (Chapter modal → Sections tab, per row).
        </p>
        <table class="u-table-full-sm">
          <thead>
            <tr class="u-border-b">
              <th class="u-cell-left">Role</th>
              <th class="u-cell-left">Model</th>
              <th class="u-cell-left">Used by</th>
            </tr>
          </thead>
          <tbody id="bs-models-table">
            <tr><td class="u-p-2 u-muted" colspan="3">Loading…</td></tr>
          </tbody>
        </table>
        <p class="u-tiny u-muted u-mt-3">
          Picks validated by the quality bench (<code>sciknow bench --layer quality</code>). See
          <a href="https://github.com/claudenstein/sciknow/blob/main/docs/PHASE_LOG.md" target="_blank">PHASE_LOG 54.6.92</a>
          for the v3 verdict that nailed down the current defaults.
        </p>
      </div>

      <!-- Style tab -->
      <div class="u-hidden" id="bs-style-pane">
        <p class="u-note-mb-m">
          Style fingerprint extracted from drafts marked
          <em>final</em> / <em>reviewed</em> / <em>revised</em> (Phase 32.10 / Layer 5).
          Injected into the autowrite writer prompt so future sections match your
          already-approved style. Refresh after you've accepted or edited drafts.
        </p>
        <div id="bs-style-fingerprint"
             style="padding:14px;border:1px solid var(--border);border-radius:6px;background:var(--bg-alt,#f8f8f8);min-height:120px;"></div>
        <div class="u-flex-raw u-gap-2 u-ai-center u-mt-10">
          <button class="btn-primary" onclick="refreshStyleFingerprint()"
                  title="Re-scan drafts marked `final` / `reviewed` / `revised` and rebuild the style fingerprint (median sentence length, citations per 100 words, hedging rate, top transitions). Future autowrite runs pick up the new fingerprint immediately.">Recompute Fingerprint</button>
          <span class="u-hint-sm" id="bs-style-status"></span>
        </div>
      </div>

      <!-- Phase 54.6.164 — View tab: local UI preferences. Not persisted
           to /api/book; stored in localStorage so they follow the
           browser, not the book. -->
      <div class="u-hidden" id="bs-view-pane">
        <p class="u-note-mb-3">
          Per-browser UI preferences. Stored in <code>localStorage</code>;
          not synced to the book or other devices.
        </p>
        <div class="field">
          <label class="u-row-click">
            <input type="checkbox" id="bs-autohide-sidebar" onchange="bsSaveViewPrefs()"
                   title="When on, the chapters column starts hidden on every page load. You can still toggle it during the session with the « Hide / » peek buttons."/>
            <span>Auto-hide chapters column on page load</span>
          </label>
          <p class="u-indent-24">
            Keeps the reader pane wider by default. Click the <code>»</code>
            peek button on the left edge to bring it back for the session.
          </p>
        </div>
        <div class="field">
          <label class="u-row-click">
            <input type="checkbox" id="bs-autohide-panel" onchange="bsSaveViewPrefs()"
                   title="When on, the sources / review / comments column starts hidden on every page load. You can still toggle it during the session with the Hide » / « peek buttons."/>
            <span>Auto-hide sources/comments column on page load</span>
          </label>
          <p class="u-indent-24">
            Hides Sources / Review / Comments at load. Click the <code>«</code>
            peek button on the right edge to bring it back for the session.
          </p>
        </div>
        <p style="font-size:11px;color:var(--fg-muted);margin-top:16px;padding-top:10px;border-top:1px dashed var(--border);">
          With auto-hide off, the current hidden/shown state is remembered
          across reloads. With auto-hide on, the column always starts hidden.
        </p>
      </div>

    </div>
  </div>
</div>

<!-- Phase 54.6.125 — Reconciliations viewer -->
<div class="modal-overlay" id="reconciliations-modal" onclick="if(event.target===this)closeModal('reconciliations-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-clipboard"/></svg> Preprint ↔ journal reconciliations</h3>
      <button class="modal-close" onclick="closeModal('reconciliations-modal')" title="Close the Preprint Reconciliations modal.">&times;</button>
    </div>
    <div class="modal-body u-md u-lh-1-5">
      <p class="u-note-mb-m">
        Phase 54.6.125. Each row is a pair where two corpus documents resolved to the same OpenAlex work_id (usually a preprint + its journal publication).
        The <em>canonical</em> row stays visible to retrieval; the <em>non-canonical</em> row is hidden (not deleted).
        Click <strong>Undo</strong> to restore a non-canonical row to retrieval.
      </p>
      <div class="u-small" id="recon-list">Loading…</div>
    </div>
  </div>
</div>

<!-- Phase 54.6.97 — AI Actions Help modal. Central reference for every
     LLM button in the draft toolbar, with what it does, when to use it,
     what it requires, what it produces, and the CLI equivalent. -->
<div class="modal-overlay" id="ai-help-modal" onclick="if(event.target===this)closeModal('ai-help-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-help-circle"/></svg> AI Actions &mdash; what each button does</h3>
      <button class="modal-close" onclick="closeModal('ai-help-modal')" title="Close the AI Actions Help modal.">&times;</button>
    </div>
    <div class="modal-body u-md u-lh-155">
      <p class="u-muted">
        sciknow&rsquo;s book workflow is a pipeline: <strong>plan</strong> the shape of the book,
        then <strong>write</strong> drafts, then <strong>critique</strong> + <strong>fix</strong>.
        Each button below belongs to one of those stages. Every action maps 1:1 to a
        <code>sciknow book ...</code> CLI command, so what you see in the GUI is exactly what
        the terminal does.
      </p>

      <h4 class="u-section-h">&#128221; Planning &mdash; shape the book before writing</h4>
      <table class="u-table-full">
        <tr class="u-border-b"><td class="u-td-wide"><strong>Outline</strong><br><em class="u-hint">Plan modal &rarr; Outline tab</em></td>
          <td class="u-td">
            Proposes a chapter structure from your paper corpus. Generates 3 candidate outlines
            at rising temperatures, scores each for breadth + section-count variance, picks the
            winner, then density-resizes each chapter&rsquo;s section list by counting corpus
            evidence. <strong>Additive</strong>: never touches existing chapters or drafts.
            <em>Use when:</em> bootstrapping a new book or expanding the plan.
            <em>Produces:</em> new <code>book_chapters</code> rows. <em>CLI:</em> <code>book outline</code>.
          </td></tr>
        <tr class="u-border-b"><td class="u-td-thin"><strong>Leitmotiv (Book plan)</strong><br><em class="u-hint">Plan modal &rarr; Book tab</em></td>
          <td class="u-td">
            The 200&ndash;500 word thesis document that gets injected into every <code>write</code> /
            <code>autowrite</code> call so all chapters stay aligned. Edit it by hand, or click
            <em>Regenerate with LLM</em>.
            <em>Use when:</em> the book&rsquo;s central argument drifts or needs sharpening.
            <em>CLI:</em> <code>book plan</code>.
          </td></tr>
      </table>

      <h4 class="u-section-h">&#9998; Writing &mdash; produce draft prose</h4>
      <table class="u-table-full">
        <tr class="u-border-b"><td class="u-td-wide"><strong>AI Write</strong></td>
          <td class="u-td">
            Single-pass draft of the current section. Retrieves sources, writes the prose, stops.
            Fast, but doesn&rsquo;t self-improve. <em>Use when:</em> you want a fresh baseline you&rsquo;ll
            hand-edit. <em>CLI:</em> <code>book write</code>.
          </td></tr>
        <tr class="u-border-b"><td class="u-td-thin"><strong>AI Autowrite</strong></td>
          <td class="u-td">
            Full convergence loop: <em>write &rarr; score &rarr; verify claims &rarr; revise &rarr; rescore</em>
            for up to 3 iterations or until overall score hits 0.85. Uses the writer model for
            prose and <code>AUTOWRITE_SCORER_MODEL</code> (gemopus4) for discrimination.
            <em>Use when:</em> you want a polished draft, not a starter.
            <em>CLI:</em> <code>book autowrite</code>.
          </td></tr>
        <tr class="u-border-b"><td class="u-td-thin"><strong>Edit</strong></td>
          <td class="u-td">
            Manual in-browser markdown editor with autosave, KaTeX math, and inline figure
            thumbnails. <em>Use when:</em> humans need to be humans.
          </td></tr>
      </table>

      <h4 class="u-section-h">&#128269; Critique &mdash; find what&rsquo;s wrong with a draft</h4>
      <table class="u-table-full">
        <tr class="u-border-b"><td class="u-td-wide"><strong>AI Review</strong></td>
          <td class="u-td">
            Single-pass critic across 5 dimensions (groundedness, completeness, accuracy,
            coherence, redundancy). Produces structured feedback with quotes + actionable
            suggestions. Writes to <code>drafts.review_feedback</code>.
            <em>Use when:</em> you have a draft and want to know where to fix it.
            <em>Different from Outline:</em> Outline builds structure; Review critiques prose.
            Uses <code>BOOK_REVIEW_MODEL</code> (gemma3:27b). <em>CLI:</em> <code>book review</code>.
          </td></tr>
        <tr class="u-border-b"><td class="u-td-thin"><strong>Adversarial review</strong></td>
          <td class="u-td">
            Harsher critic &mdash; forces <strong>&ge; 10</strong> concrete issues, never graded.
            Doesn&rsquo;t overwrite the normal <code>review_feedback</code>.
            <em>Use when:</em> the standard review feels too gentle.
            <em>CLI:</em> <code>book adversarial-review</code>.
          </td></tr>
        <tr class="u-border-b"><td class="u-td-thin"><strong>Edge cases</strong></td>
          <td class="u-td">
            Exhaustive boundary-condition hunter: walks every scope boundary, counter-case,
            causal alternative, and quantitative limit. Structured JSON output.
            <em>Use when:</em> you need the draft to survive hostile readers.
            <em>CLI:</em> <code>book edge-cases</code>.
          </td></tr>
        <tr class="u-border-b"><td class="u-td-thin"><strong>Ensemble Review</strong></td>
          <td class="u-td">
            N independent NeurIPS-rubric reviewers (default 3, T=0.75, rotating neutral/pessimistic/
            optimistic stance) + a meta-reviewer that medians scores and unions findings.
            Costs ~N&times; the single-review. <em>Use when:</em> a single reviewer felt flaky.
            <em>CLI:</em> <code>book ensemble-review</code>.
          </td></tr>
        <tr class="u-border-b"><td class="u-td-thin"><strong>Argue (map claim)</strong></td>
          <td class="u-td">
            For any claim you type, builds a SUPPORTS / CONTRADICTS / NEUTRAL evidence map
            from your corpus. <em>Use when:</em> you want to see how solid a single
            assertion is. <em>CLI:</em> <code>book argue</code>.
          </td></tr>
        <tr class="u-border-b"><td class="u-td-thin"><strong>Gaps</strong></td>
          <td class="u-td">
            Identifies topics the book&rsquo;s plan names but the drafts don&rsquo;t cover yet.
            <em>Use when:</em> a chapter feels thin and you&rsquo;re not sure why.
            <em>CLI:</em> <code>book gaps</code>.
          </td></tr>
      </table>

      <h4 class="u-section-h">&#10003; Verification &mdash; check citations actually support claims</h4>
      <table class="u-table-full">
        <tr class="u-border-b"><td class="u-td-wide"><strong>Verify</strong></td>
          <td class="u-td">
            Sentence-level LLM verifier: classifies each <code>[N]</code> claim as SUPPORTED /
            EXTRAPOLATED / MISREPRESENTED / OVERSTATED and returns a groundedness + hedging
            fidelity score. <em>CLI:</em> <code>book verify-citations</code>.
          </td></tr>
        <tr class="u-border-b"><td class="u-td-thin"><strong>Verify Draft</strong><br><em class="u-hint">claim-atomization</em></td>
          <td class="u-td">
            Splits each sentence into atomic sub-claims (regex first, LLM fallback for compound
            sentences), NLI-scores each sub-claim against source chunks, and surfaces
            <strong>mixed-truth</strong> sentences &mdash; where part of the sentence is supported
            and part isn&rsquo;t. Catches what the sentence-level verifier misses.
            Read-only. <em>CLI:</em> <code>book verify-draft</code>.
          </td></tr>
        <tr class="u-border-b"><td class="u-td-thin"><strong>Align Citations</strong></td>
          <td class="u-td">
            Post-pass that (conservatively) remaps <code>[N]</code> markers to the chunk that
            actually entails the sentence, when the currently-cited chunk has
            entailment &lt; 0.5 AND the top chunk beats it by &ge; 0.15.
            <em>Use when:</em> verify reports mismatched citations.
            <em>CLI:</em> <code>book align-citations</code>.
          </td></tr>
        <tr class="u-border-b"><td class="u-td-thin"><strong>Insert Citations</strong></td>
          <td class="u-td">
            Two-pass LLM adds <code>[N]</code> markers where the prose asserts something
            source-worthy but has no citation. Saves a new draft version.
            <em>Use when:</em> hand-written prose is missing its citation layer.
            <em>CLI:</em> <code>book insert-citations</code>.
          </td></tr>
      </table>

      <h4 class="u-section-h">&#9888;&#65039; Fixing &mdash; apply feedback to the draft</h4>
      <table class="u-table-full">
        <tr class="u-border-b"><td class="u-td-wide"><strong>AI Revise</strong></td>
          <td class="u-td">
            Reads the latest <code>review_feedback</code> saved on the draft and rewrites the
            prose to address it. <em>Requires:</em> a Review (or Autowrite, which reviews
            internally) has been run first. <em>Use when:</em> you&rsquo;re happy with the
            critique and want the draft to act on it. <em>CLI:</em> <code>book revise</code>.
          </td></tr>
      </table>

      <h4 class="u-section-h">&#128202; Diagnostics &mdash; inspect the autowrite trajectory</h4>
      <table class="u-table-full">
        <tr class="u-border-b"><td class="u-td-wide"><strong>Scores</strong></td>
          <td class="u-td">
            Shows the 5-dimension score trajectory across autowrite iterations &mdash; which
            scores rose, which plateaued, which triggered revisions. Read-only.
          </td></tr>
        <tr class="u-border-b"><td class="u-td-thin"><strong>Bundles</strong></td>
          <td class="u-td">
            Chapter- or book-wide snapshots. Safety net for <em>autowrite-all-sections</em>
            runs &mdash; snapshot before, restore if you hate the result.
            <em>CLI:</em> <code>book snapshot</code> / <code>snapshot-restore</code>.
          </td></tr>
        <tr class="u-border-b"><td class="u-td-thin"><strong>Chapter reader</strong></td>
          <td class="u-td">
            Read-only continuous-scroll view of the whole chapter so you can feel the flow
            without the editor chrome.
          </td></tr>
      </table>

      <h4 class="u-section-h">&#128737;&#65039; Typical workflow</h4>
      <ol style="margin:0 0 8px 20px;">
        <li><strong>Outline</strong> (Plan modal) &rarr; book has chapters</li>
        <li><strong>AI Autowrite</strong> per section &rarr; first drafts exist, already reviewed + revised once</li>
        <li><strong>Verify Draft</strong> + <strong>Align Citations</strong> &rarr; confirm citations hold up</li>
        <li><strong>AI Review</strong> (or Adversarial / Ensemble) &rarr; structured critique</li>
        <li><strong>AI Revise</strong> &rarr; apply feedback</li>
        <li>Repeat 4&ndash;5 until happy, then <strong>Bundle</strong> snapshot the chapter</li>
      </ol>
    </div>
    <div class="modal-footer">
      <button class="btn-primary" onclick="closeModal('ai-help-modal')"
              title="Close this help card.">Got it</button>
    </div>
  </div>
</div>

<!-- Phase 43h — Project management modal -->
<!-- Phase 54.6.312 — thumbnail lightbox. Outside the .modal-overlay
     stack because the panel-visuals thumbnails trigger it while the
     rest of the reader stays interactive behind it. -->
<div class="visuals-lightbox" id="visuals-lightbox" onclick="if(event.target===this)closeVisualLightbox()">
  <button class="visuals-lightbox-close" onclick="closeVisualLightbox()" aria-label="Close">&times;</button>
  <img id="visuals-lightbox-img" alt="" />
  <div class="visuals-lightbox-cap" id="visuals-lightbox-cap"></div>
</div>

<!-- Phase 54.6.312 — Citation preview popup. Shown when the user clicks
     an inline [N] marker in any draft. Populates from
     /api/bibliography/citation/<N>. Close button, click-outside, Esc
     all dismiss. -->
<div class="visuals-lightbox" id="citation-preview" onclick="if(event.target===this)closeCitationPreview()">
  <button class="visuals-lightbox-close" onclick="closeCitationPreview()" aria-label="Close">&times;</button>
  <div class="citation-preview-card" id="citation-preview-card">
    <em>Loading…</em>
  </div>
</div>

<!-- Phase 54.6.312 — Bibliography tools modal: audit + renumber. -->
<div class="modal-overlay" id="bib-tools-modal" onclick="if(event.target===this)closeModal('bib-tools-modal')">
  <div class="modal">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-book-open"/></svg> Bibliography Tools</h3>
      <button class="modal-close" onclick="closeModal('bib-tools-modal')" title="Close the Bibliography Tools modal.">&times;</button>
    </div>
    <div class="modal-body">
      <div class="u-note-sm u-mb-m">
        Two helpers for when the global bibliography numbering drifts
        from what's stored in draft markdown (typically after adding a
        new chapter or section).
      </div>
      <div class="u-flex-raw u-gap-2 u-wrap u-mb-m">
        <button class="btn-secondary" onclick="runBibliographyAudit()"
                title="Scan every draft for citations pointing at a missing source, sources never cited in the body, and duplicate entries. Read-only, ~1s.">&#128269; Sanity check</button>
        <button class="btn-primary" onclick="runBibliographySort()"
                title="Flatten local→global citation numbers INTO the stored draft content. Renumbers every draft's [N] markers and sources list so the raw markdown matches the global bibliography. Idempotent — run again after new chapters/sections.">&#128260; Sort &amp; renumber bibliography</button>
      </div>
      <div id="bib-tools-output" class="u-note-sm" style="min-height: 120px; font-family: var(--font-mono); white-space: pre-wrap;"></div>
    </div>
  </div>
</div>
<!-- Phase 21.c — Visuals browser modal -->
<div class="modal-overlay" id="visuals-modal" onclick="if(event.target===this)closeModal('visuals-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-image"/></svg> Visual Elements</h3>
      <button class="modal-close" onclick="closeModal('visuals-modal')" title="Close the Visuals modal.">&times;</button>
    </div>
    <div class="modal-body u-md">
      <!-- Phase 54.6.99 — Visual Elements browser redesigned:
           * Default view is a gallery of image thumbnails (figures + charts),
             which is the 10,881 items (of 17,322 total) that actually have
             rendered JPG/PNGs on disk.
           * Mode toggle: Gallery (images only) vs List (all kinds incl.
             equations/tables/code).
           * Kind filter includes `chart` (was missing in 54.6.87 and
             earlier — users picking "Figures" missed 7,626 charts).
           * Figures + charts now render with real <img> thumbnails and
             a click-to-enlarge link, matching wiki-visuals-list. -->
      <div class="u-flex-raw u-gap-2 u-mb-m u-wrap u-ai-center">
        <label class="u-tiny u-flex-raw u-ai-center u-gap-1">
          Mode:
          <select class="u-pill-md" id="vis-mode" onchange="loadVisuals()"
                  title="Gallery = CSS grid of thumbnails (figures + charts). List = all kinds in a row layout (equations / tables / code render inline, images get real thumbnails).">
            <option value="gallery" selected>Gallery (figures + charts)</option>
            <option value="list">List (all kinds)</option>
          </select>
        </label>
        <select class="u-pill-md" id="vis-kind-filter" onchange="loadVisuals()"
                title="Narrow to a single visual kind. In Gallery mode, non-image kinds widen the card grid to 340px columns.">
          <option value="">All types</option>
          <option value="figure">Figures</option>
          <option value="chart">Charts</option>
          <option value="equation">Equations</option>
          <option value="table">Tables</option>
          <option value="code">Code</option>
        </select>
        <label class="u-tiny u-flex-raw u-ai-center u-gap-1" title="How to order the visuals">
          Order:
          <select class="u-p-4-6" id="vis-order" onchange="loadVisuals()"
                  title="Importance = deterministic composite score (year + caption richness + has-figure-num + paper-type weight). Others are single-signal sorts; Random picks a non-stable sample for variety.">
            <option value="importance" selected>Importance (ranked)</option>
            <option value="recent">Recent papers first</option>
            <option value="paper">By paper (title)</option>
            <option value="figure_num">Figure number</option>
            <option value="caption_richness">Richest captions</option>
            <option value="random">Random</option>
          </select>
        </label>
        <input type="text" id="vis-search" placeholder="Search captions..." style="flex:1;min-width:150px;padding:4px 8px;" onkeyup="if(event.key==='Enter')loadVisuals()"
               title="Substring match against caption + surrounding text. Press Enter to search."/>
        <button class="btn-secondary" onclick="loadVisuals()"
                title="Apply the current search + filter + order and reload.">&#128269; Search</button>
        <span class="u-hint" id="vis-stats"></span>
      </div>
      <div class="u-modal-scroll" id="vis-results">
        <em>Loading...</em>
      </div>
    </div>
  </div>
</div>

<!-- Phase 54.6.24 — Backups modal -->
<div class="modal-overlay" id="backups-modal" onclick="if(event.target===this)closeModal('backups-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-archive"/></svg> Backups</h3>
      <button class="modal-close" onclick="closeModal('backups-modal')" title="Close the Backups modal.">&times;</button>
    </div>
    <div class="modal-body u-md">
      <div class="u-mb-3 u-p-10 u-bg-alt-raw u-r-md" id="backup-status">
        Loading backup status...
      </div>
      <div id="backup-location" style="margin-bottom:10px;font-size:11px;color:var(--fg-muted);word-break:break-all;"></div>

      <details class="u-mb-3 u-border u-r-md u-p-8-10">
        <summary class="u-click u-semibold">&#128339; Schedule auto-backup</summary>
        <div class="u-flex-raw u-gap-2 u-wrap u-ai-center u-mt-10">
          <label>Frequency:
            <select class="u-pad-xs" id="backup-sched-freq"
                    title="How often to snapshot the project(s). Hourly for active work; Daily for the default; Weekly if you mostly do bulk runs.">
              <option value="hourly">Hourly</option>
              <option value="daily" selected>Daily</option>
              <option value="weekly">Weekly</option>
            </select>
          </label>
          <label class="u-hidden" id="backup-sched-weekday-label">Day:
            <select class="u-pad-xs" id="backup-sched-weekday"
                    title="Day of the week to run the backup. Only applies when Frequency = Weekly.">
              <option value="0">Sun</option><option value="1">Mon</option>
              <option value="2">Tue</option><option value="3">Wed</option>
              <option value="4">Thu</option><option value="5">Fri</option>
              <option value="6">Sat</option>
            </select>
          </label>
          <label id="backup-sched-hour-label">Hour:
            <input class="u-w-60 u-pad-xs" type="number" id="backup-sched-hour" min="0" max="23" value="3"
                   title="Hour of the day (0–23) when the backup fires. Defaults to 03:00 — usually off-hours."/>
          </label>
          <label>Minute:
            <input class="u-w-60 u-pad-xs" type="number" id="backup-sched-minute" min="0" max="59" value="0"
                   title="Minute past the hour. Hourly mode uses this alone (e.g. `:15` = every hour at quarter past)."/>
          </label>
          <button class="btn-primary" onclick="enableBackupSchedule()"
                  title="Install the crontab entry. Overwrites any existing sciknow-auto-backup line.">Save schedule</button>
          <button class="btn-secondary u-hidden" id="backup-unschedule-btn" onclick="disableBackupSchedule()"
                  title="Remove the crontab entry. Existing backups are NOT deleted.">Disable</button>
        </div>
      </details>

      <details class="u-mb-3 u-border u-r-md u-p-8-10">
        <summary class="u-click u-semibold">&#128465; Autodelete old backups</summary>
        <div class="u-flex-raw u-gap-2 u-wrap u-ai-center u-mt-10">
          <label title="DESTRUCTIVE: wipes every backup in archives/backups/. Confirms first. No recovery.">
            <input type="checkbox" id="backup-purge-all-check"
                   title="Check to wipe EVERY backup. Overrides the days field."> Delete ALL backups</label>
          <span class="u-muted">— or —</span>
          <label>Older than
            <input class="u-w-70 u-pad-xs" type="number" id="backup-purge-days" min="1" value="30"
                   title="Delete backups strictly older than this many days. 30 is a safe starting point."/>
            days
          </label>
          <button class="btn-secondary" onclick="purgeBackups()" style="color:var(--danger);border-color:var(--danger);"
                  title="Apply the selected purge rule. Confirms first and cannot be undone — archives/backups/* files are deleted from disk.">Purge now</button>
        </div>
        <p class="u-note-top">
          Auto-age retention on every run is controlled by <code>BACKUP_RETAIN_DAYS</code> in <code>.env</code>
          (0 = disabled). Count-based retention is <code>BACKUP_RETAIN_COUNT</code> (default 7).
        </p>
      </details>

      <div class="u-flex-raw u-gap-2 u-mb-3 u-wrap">
        <button class="btn-primary" onclick="runBackupNow()"
                title="Take a fresh snapshot of all projects + optional system bundle. Streams progress into the log below.">&#128190; Run Backup Now</button>
        <button class="btn-secondary" onclick="restoreBackup()"
                title="Restore the most recent backup. DESTRUCTIVE — overwrites current projects with the backup version. Confirms first; Qdrant vectors will need rebuilding after.">&#128260; Restore Latest</button>
      </div>
      <div id="backup-log" style="display:none;max-height:120px;overflow-y:auto;font-family:var(--font-mono);font-size:11px;background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:8px;margin-bottom:12px;white-space:pre-wrap;"></div>
      <h4 style="margin:0 0 8px;">Backup History</h4>
      <div id="backup-list" style="max-height:calc(80vh - 380px);min-height:220px;overflow-y:auto;">
        <em>Loading...</em>
      </div>
    </div>
  </div>
</div>

<div class="modal-overlay" id="projects-modal" onclick="if(event.target===this)closeModal('projects-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-folder"/></svg> Projects</h3>
      <button class="modal-close" onclick="closeModal('projects-modal')" title="Close the Projects modal.">&times;</button>
    </div>
    <div class="modal-body">
      <p class="u-note-mb-3">
        Each project has its own PostgreSQL DB, Qdrant collections, and <code>data/</code> directory.
        Mirrors <code>sciknow project …</code>. The web reader is currently running against
        <strong id="proj-running"></strong>; switching the active project only takes effect after
        restarting <code>sciknow book serve</code>.
      </p>
      <div class="u-flex-raw u-gap-2 u-ai-center u-mb-m">
        <button class="btn-primary" onclick="refreshProjectsList()"
                title="Reload the project list from the server.">Refresh</button>
        <span class="u-small u-muted u-flex-1" id="proj-msg"></span>
      </div>
      <div id="projects-list-wrap" style="margin-bottom:16px;">
        <div id="projects-list" class="u-small">
          <div class="skeleton-card">
            <div class="skeleton skeleton-line"></div>
            <div class="skeleton skeleton-line skeleton-line--short"></div>
          </div>
          <div class="skeleton-card">
            <div class="skeleton skeleton-line"></div>
            <div class="skeleton skeleton-line skeleton-line--short"></div>
          </div>
          <div class="skeleton-card">
            <div class="skeleton skeleton-line"></div>
            <div class="skeleton skeleton-line skeleton-line--short"></div>
          </div>
        </div>
      </div>
      <div class="u-border-t u-pt-3 u-mt-2">
        <h4 class="u-md u-mb-2">Create new project</h4>
        <p class="u-note">
          Creates an empty project (DB + collections + dir + migrations). For the one-shot
          migration of the legacy install, use <code>sciknow project init &lt;slug&gt; --from-existing</code>
          from the CLI.
        </p>
        <div class="u-row-gap-sm">
          <input type="text" id="proj-new-slug" placeholder="new-project-slug (lowercase, hyphens)"
                 style="flex:1;padding:6px 10px;border:1px solid var(--border);border-radius:var(--r-sm);background:var(--bg);color:var(--fg);"
                 title="Slug for the new project. Must be lowercase alphanumeric + hyphens; becomes the PostgreSQL database name `sciknow_<slug>` and the Qdrant collection prefix."/>
          <button class="btn-primary" onclick="createProject()"
                  title="Create the project: PG database, Qdrant collections (papers / abstracts / wiki / visuals), data directory, .env.overlay template, Alembic migrations. Empty corpus — ingest papers after.">Create</button>
        </div>
      </div>
      <div class="u-mt-14" id="proj-detail"></div>

      <!-- Phase 54.6.112 (Tier 1 #5) — Venue block/allow lists for
           db expand. JSON lives at <project>/venue_config.json. -->
      <div class="u-border-t u-pt-3 u-mt-14">
        <h4 class="u-md u-mb-6">&#128683; Venues — block / allow lists</h4>
        <p class="u-note">
          Per-project substring patterns matched against candidate papers' publisher / host-organization / source names during
          <code>db expand</code>. Blocklist extends the built-in predatory pattern set; allowlist wins over both so you can
          rescue legitimate venues from a false-positive. Prefix with <code>^</code> or suffix with <code>$</code> for regex.
          Mirrors <code>sciknow project venue-block / venue-allow / venue-remove</code>.
        </p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
          <div class="u-card">
            <div class="u-row-between-mb">
              <strong class="u-small u-danger">Blocklist</strong>
              <span class="u-label-xs" id="proj-ven-block-count"></span>
            </div>
            <div class="u-flex-raw u-gap-1 u-mb-6">
              <input class="u-flex-1 u-p-4-6 u-tiny" type="text" id="proj-ven-block-in" placeholder="e.g. scirp" onkeydown="if(event.key==='Enter')addVenuePattern('block')"
                     title="Substring (or regex with ^/$) to block. Matched against candidate publisher / host-organization / source names.">
              <button class="btn-secondary u-pill-xs" onclick="addVenuePattern('block')"
                      title="Add this pattern to the project's blocklist.">+</button>
            </div>
            <ul id="proj-ven-block-list" style="list-style:none;margin:0;padding:0;max-height:140px;overflow:auto;font-size:11px;"></ul>
          </div>
          <div class="u-card">
            <div class="u-row-between-mb">
              <strong class="u-small u-success">Allowlist</strong>
              <span class="u-label-xs" id="proj-ven-allow-count"></span>
            </div>
            <div class="u-flex-raw u-gap-1 u-mb-6">
              <input class="u-flex-1 u-p-4-6 u-tiny" type="text" id="proj-ven-allow-in" placeholder="e.g. frontiers in climate" onkeydown="if(event.key==='Enter')addVenuePattern('allow')"
                     title="Substring (or regex) to ALWAYS allow — wins over both the built-in predatory pattern set and the blocklist.">
              <button class="btn-secondary u-pill-xs" onclick="addVenuePattern('allow')"
                      title="Add this pattern to the project's allowlist.">+</button>
            </div>
            <ul id="proj-ven-allow-list" style="list-style:none;margin:0;padding:0;max-height:140px;overflow:auto;font-size:11px;"></ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Phase 46.F — End-to-end Setup Wizard: project → corpus → indices → expand → book -->
<div class="modal-overlay" id="setup-wizard-modal" onclick="if(event.target===this)closeModal('setup-wizard-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-wand"/></svg> Setup Wizard</h3>
      <button class="modal-close" onclick="closeModal('setup-wizard-modal')" title="Close the Setup Wizard modal.">&times;</button>
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
        <p class="u-note-md">
          A <strong>project</strong> isolates one corpus (its own DB + Qdrant
          collections + data dir). Pick an existing one or create a new one.
          The web reader is currently serving the project shown as
          <strong>active</strong>; creating a new project here writes
          <code>.active-project</code> but does NOT hot-swap this running
          server — finish the wizard, then restart <code>sciknow book
          serve</code> against the new project.
        </p>
        <div class="u-flex-raw u-gap-10 u-ai-start">
          <div class="u-flex-1">
            <h4 class="u-md u-mh-6">Existing projects</h4>
            <div id="sw-project-list" style="font-size:12px;max-height:180px;overflow:auto;border:1px solid var(--border);border-radius:6px;">
              Loading…
            </div>
          </div>
          <div style="flex:1;border-left:1px solid var(--border);padding-left:12px;">
            <h4 class="u-md u-mh-6">Create new</h4>
            <div class="field">
              <label>Slug</label>
              <input type="text" id="sw-new-slug" placeholder="global-cooling"
                     title="Lowercase letters / digits / hyphens only. Used as the PostgreSQL DB name suffix (sciknow_<slug>), Qdrant collection prefix and data/ sub-directory.">
            </div>
            <button class="btn-primary" onclick="swCreateProject()"
                    title="Runs migrations + creates Qdrant collections for the new slug (~3s). Then you still need to restart `sciknow book serve` to use it.">Create empty project</button>
            <p class="u-note-mt-6">
              Slug is lowercase alphanumerics + hyphens. Runs migrations
              + creates Qdrant collections (~3 s).
            </p>
          </div>
        </div>
        <div class="u-flex-raw u-jc-between u-ai-center u-mt-3">
          <span class="u-hint-sm" id="sw-project-status"></span>
          <button class="btn-primary" onclick="swGoto('corpus')"
                  title="Advance to Step 2 — feed PDFs into the active project.">Next: Corpus &rarr;</button>
        </div>
      </div>

      <!-- STEP 2 — Corpus -->
      <div id="sw-step-corpus" class="sw-step-pane u-hidden-pad">
        <p class="u-note-md">
          Feed the project PDFs. Two paths:
          <strong>upload</strong> files from your browser, or point to a
          <strong>directory on this server</strong>. Either way, the
          ingestion pipeline runs (PDF → metadata → sections → chunks →
          embeddings). Large corpora take hours.
        </p>
        <div class="u-small u-p-2 u-bg-tb u-border u-r-sm u-mb-m" id="sw-corpus-status">
          Loading corpus status…
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
          <div class="u-card">
            <div class="u-label-strong">&#128190; Upload PDFs</div>
            <p class="u-note">
              Files are staged under
              <code>{{data_dir}}/inbox/uploads_&lt;ts&gt;/</code> and then
              ingested.
            </p>
            <input class="u-block u-mb-2" type="file" id="sw-upload-files" accept="application/pdf,.pdf" multiple
                   title="Pick one or many PDFs from your browser. They're uploaded to the server's inbox.">
            <label class="u-label-row-mb"
                   title="Kick off the convert → metadata → chunk → embed pipeline right after upload. Uncheck to just stage the files.">
              <input type="checkbox" id="sw-upload-start-ingest" checked
                     title="Kick off the convert → metadata → chunk → embed pipeline right after upload.">
              start ingesting immediately
            </label>
            <button class="btn-primary" onclick="swUploadPDFs()"
                    title="Send selected PDFs to the server's uploads_<ts>/ inbox. If 'start ingesting' is checked the pipeline runs next.">Upload</button>
          </div>
          <div class="u-card">
            <div class="u-label-strong">&#128193; Server directory</div>
            <p class="u-note">
              Path is resolved on the server. Useful when a corpus is
              already on disk (or over a network mount).
            </p>
            <div class="field">
              <input type="text" id="sw-ingest-path" placeholder="/path/to/pdfs"
                     title="Absolute server-side path to a directory of PDFs. Must be readable by the sciknow process (no remote paths, no ~).">
            </div>
            <div class="field u-flex-raw u-gap-10 u-wrap">
              <label class="u-label-row"
                     title="Walk sub-directories. Uncheck to only ingest PDFs directly in the given folder.">
                <input type="checkbox" id="sw-ingest-recursive" checked
                       title="Walk sub-directories. Uncheck to only ingest PDFs directly in the given folder."> recursive
              </label>
              <label class="u-label-row"
                     title="Re-ingest PDFs even if their SHA-256 is already in the DB. Use only when you deliberately want to re-run the pipeline.">
                <input type="checkbox" id="sw-ingest-force"
                       title="Re-ingest PDFs even if their SHA-256 hash is already present in the DB."> force re-ingest
              </label>
            </div>
            <button class="btn-primary" onclick="swIngestDirectory()"
                    title="Stream the ingestion log from the server. Idempotent — re-runs skip completed papers unless force is on.">Ingest directory</button>
          </div>
        </div>
        <pre id="sw-ingest-log" style="margin-top:10px;max-height:260px;overflow:auto;background:var(--bg-alt,#f8f8f8);border:1px solid var(--border);border-radius:6px;padding:10px;font-size:11px;font-family:ui-monospace,monospace;white-space:pre-wrap;"></pre>
        <div class="u-flex-raw u-gap-2 u-jc-between u-mt-2">
          <button onclick="swGoto('project')"
                  title="Return to Step 1 (pick / create project).">&larr; Back</button>
          <div>
            <button onclick="swRefreshStatus()"
                    title="Re-read corpus stats (paper count, status breakdown) from the server.">Refresh status</button>
            <button class="btn-primary" onclick="swGoto('indices')"
                    title="Advance to Step 3 — build optional retrieval/synthesis indices.">Next: Indices &rarr;</button>
          </div>
        </div>
      </div>

      <!-- STEP 3 — Indices -->
      <div id="sw-step-indices" class="sw-step-pane u-hidden-pad">
        <p class="u-note-md">
          After ingestion, build the three optional indices. Each improves
          downstream quality — you can skip any of them. Run in any order.
        </p>
        <div class="u-small u-p-2 u-bg-tb u-border u-r-sm u-mb-m" id="sw-indices-status">
          Loading index status…
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">
          <div class="u-card">
            <div class="u-label-strong">&#127918; Topic Clusters</div>
            <p class="u-note">
              BERTopic over abstracts. Fast (seconds). Enables
              <code>--topic</code> filtering in retrieval + the Topics
              browser.
            </p>
            <label class="u-label-row-mb"
                   title="Drop the existing clusters and rebuild from all abstracts. Default is incremental (only new papers).">
              <input type="checkbox" id="sw-cluster-rebuild"
                     title="Drop the existing clusters and rebuild from all abstracts. Default is incremental."> rebuild from scratch
            </label>
            <button class="btn-primary" onclick="swRunIndex('cluster')"
                    title="Run BERTopic over paper abstracts. Needed before `--topic` filtering or the Topics browser works.">Cluster</button>
          </div>
          <div class="u-card">
            <div class="u-label-strong">&#127794; RAPTOR tree</div>
            <p class="u-note">
              Hierarchical summaries (UMAP + GMM). Slow (5–30 min).
              Enables broad-synthesis retrieval.
            </p>
            <button class="btn-primary" onclick="swRunIndex('raptor')"
                    title="Cluster chunks with UMAP+GMM, summarize each cluster and recurse. Enables broad-synthesis retrieval on long queries.">Build RAPTOR</button>
          </div>
          <div class="u-card">
            <div class="u-label-strong">&#128218; Wiki compile</div>
            <p class="u-note">
              Compile per-paper wiki pages + KG triples. Slow
              (LLM-bound, ~1 min per paper).
            </p>
            <label class="u-label-row-mb"
                   title="Re-compile every paper's wiki page, even if it's already present.">
              <input type="checkbox" id="sw-wiki-rebuild"
                     title="Re-compile every paper's wiki page, even if it's already present."> rebuild
            </label>
            <label class="u-label-row-mb"
                   title="Re-compile only pages flagged as stale (sources have changed since last compile).">
              <input type="checkbox" id="sw-wiki-stale" checked
                     title="Re-compile only pages flagged as stale (sources changed since last compile)."> rewrite stale
            </label>
            <button class="btn-primary" onclick="swRunIndex('wiki')"
                    title="Run `sciknow wiki compile`: one LLM call per paper → page, visuals, KG triples.">Compile wiki</button>
          </div>
        </div>
        <pre id="sw-indices-log" style="margin-top:10px;max-height:260px;overflow:auto;background:var(--bg-alt,#f8f8f8);border:1px solid var(--border);border-radius:6px;padding:10px;font-size:11px;font-family:ui-monospace,monospace;white-space:pre-wrap;"></pre>
        <div class="u-flex-raw u-gap-2 u-jc-between u-mt-2">
          <button onclick="swGoto('corpus')"
                  title="Return to Step 2 (corpus ingestion).">&larr; Back</button>
          <div>
            <button onclick="swRefreshStatus()"
                    title="Re-read index status from the server.">Refresh status</button>
            <button class="btn-primary" onclick="swGoto('expand')"
                    title="Advance to Step 4 — optional corpus expansion.">Next: Expand &rarr;</button>
          </div>
        </div>
      </div>

      <!-- STEP 4 — Expand -->
      <div id="sw-step-expand" class="sw-step-pane u-hidden-pad">
        <p class="u-note-md">
          Optional: grow the corpus by following citations or pulling
          everything an author has published. Uses the full
          Expand tab — see the <strong>Tools</strong> toolbar button for
          the detailed surface.
        </p>
        <div class="u-flex-raw u-gap-10">
          <button class="btn-primary" onclick="closeModal('setup-wizard-modal');openCorpusModal();"
                  title="Close the wizard and jump straight to the full Tools → Corpus panel (enrich / expand / author / inbound / topic / coauthor / agentic).">
            &#128736; Open Tools &rarr; Corpus tab
          </button>
        </div>
        <div style="display:flex;gap:8px;justify-content:space-between;margin-top:20px;">
          <button onclick="swGoto('indices')"
                  title="Return to Step 3 (indices).">&larr; Back</button>
          <button class="btn-primary" onclick="swGoto('book')"
                  title="Advance to Step 5 — create the writing project.">Next: Book &rarr;</button>
        </div>
      </div>

      <!-- STEP 5 — Book -->
      <div id="sw-step-book" class="sw-step-pane u-hidden-pad">
        <p class="u-note-md">
          Create the writing project. The <strong>type</strong> drives section
          defaults, prompt conditioning, and length targets. Ranges below
          each type come from the Phase 54.6.146 concept-density literature
          review (see <code>docs/RESEARCH.md §24</code>) — Cowan 2001's
          3&ndash;4 novel-chunk capacity × genre-appropriate words per concept.
        </p>
        <div class="field u-row-wrap-end">
          <div class="u-col-220">
            <label>Title</label>
            <input type="text" id="sw-book-title" placeholder="e.g. Global Cooling: The Coming Solar Minimum"
                   title="Full book title. Used in sidebar, browser tab, and PDF/epub exports.">
          </div>
          <div style="flex:2;min-width:180px;">
            <label>Type</label>
            <select id="sw-book-type" onchange="swUpdateTypeInfo()"
                    title="Drives section defaults, prompt conditioning, default chapter length, and the concept-density resolver's wpc midpoint. See the info panel below for each type's length bands.">
              <option value="scientific_book">Scientific Book (chapters)</option>
            </select>
          </div>
          <div class="u-col-120">
            <label>Target words/chap</label>
            <input type="number" id="sw-book-target" placeholder="(type default)"
                   title="Target word count per chapter. Leave blank to use the type default shown in the info panel below. Concept-density resolver will still override this per-section when a section plan exists.">
          </div>
        </div>
        <!-- Phase 54.6.147 — live info panel showing the selected type's
             length ranges so the user sees what they're committing to. -->
        <div id="sw-book-type-info"
             style="margin-top:6px;padding:10px 12px;background:var(--toolbar-bg);border:1px solid var(--border);border-radius:6px;font-size:12px;line-height:1.5;color:var(--fg);">
          <em class="u-muted">Loading type info…</em>
        </div>
        <div class="field u-mt-3">
          <label>Description (optional)</label>
          <input type="text" id="sw-book-desc" placeholder="One-line blurb."
                 title="Short description shown in the catalog.">
        </div>
        <button class="btn-primary" onclick="swCreateBook()"
                title="Create the book + LLM-generate an initial chapter outline. You can reorder / edit chapters afterwards.">Create project</button>
        <div class="u-caption" id="sw-book-status"></div>
        <div class="u-flex-raw u-gap-2 u-jc-between u-mt-14">
          <button onclick="swGoto('expand')"
                  title="Return to Step 4 (expand).">&larr; Back</button>
          <button onclick="closeModal('setup-wizard-modal')"
                  title="Close the wizard. You can re-open from the top navigation at any time.">Done</button>
        </div>
      </div>

    </div>
  </div>
</div>

<!-- Phase 38 — Scoped snapshot bundles (chapter + book) -->
<div class="modal-overlay" id="bundle-modal" onclick="if(event.target===this)closeModal('bundle-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-package"/></svg> Snapshot Bundles</h3>
      <button class="modal-close" onclick="closeModal('bundle-modal')" title="Close the Snapshot Bundles modal.">&times;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="sb-chapter" onclick="switchBundleTab('sb-chapter')"
              title="Snapshot / restore every draft in the currently active chapter.">Chapter</button>
      <button class="tab" data-tab="sb-book" onclick="switchBundleTab('sb-book')"
              title="Snapshot / restore every draft across every chapter. Slow on big books.">Book</button>
    </div>
    <div class="modal-body">

      <!-- Chapter scope -->
      <div id="sb-chapter-pane">
        <p class="u-note-mb-m">
          Snapshot every section in the current chapter as one bundle.
          Restore is <strong>non-destructive</strong> &mdash; each section
          gets a NEW draft version, so existing drafts stay put as an undo path.
          Best used before firing <code>autowrite</code> on a whole chapter.
        </p>
        <div class="field u-row-end">
          <div class="u-flex-2">
            <label>Snapshot label (optional)</label>
            <input type="text" id="sb-chapter-name" placeholder="(auto: chapter title + timestamp)"
                   title="Optional human label for this snapshot. Leave blank to auto-generate from chapter title + timestamp.">
          </div>
          <button class="btn-primary" onclick="doBundleSnapshot('chapter')"
                  title="Create a bundle containing every draft in this chapter. Restore is non-destructive (new draft versions).">Snapshot Current Chapter</button>
        </div>
        <div class="u-note-vertical" id="sb-chapter-status"></div>
        <div class="u-mt-10" id="sb-chapter-list"></div>
      </div>

      <!-- Book scope -->
      <div class="u-hidden" id="sb-book-pane">
        <p class="u-note-mb-m">
          Snapshot every draft across every chapter in this book.
          Restore walks each chapter bundle and creates new draft versions per section.
          Slow on big books &mdash; prefer a chapter snapshot when you only need scope for one chapter.
        </p>
        <div class="field u-row-end">
          <div class="u-flex-2">
            <label>Snapshot label (optional)</label>
            <input type="text" id="sb-book-name" placeholder="(auto: book title + timestamp)"
                   title="Optional human label for this snapshot. Leave blank to auto-generate.">
          </div>
          <button class="btn-primary" onclick="doBundleSnapshot('book')"
                  title="Create a bundle containing every draft across every chapter. Recommended before a full-book autowrite or bulk revision.">Snapshot Whole Book</button>
        </div>
        <div class="u-note-vertical" id="sb-book-status"></div>
        <div class="u-mt-10" id="sb-book-list"></div>
      </div>

    </div>
  </div>
</div>

<!-- Phase 36 — Tools Modal: CLI-parity panel (search / synthesize / topics / corpus) -->
<!-- Phase 54.6.230 — System monitor modal (unified CLI parity) -->
<div class="modal-overlay" id="monitor-modal" onclick="if(event.target===this)closeModal('monitor-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-sliders"/></svg> System Monitor</h3>
      <div style="flex:1;"></div>
      <span id="monitor-last-updated" style="color:var(--fg-muted);font-size:0.85em;margin-right:12px;"></span>
      <label style="color:var(--fg-muted);font-size:0.85em;margin-right:12px;"
             title="Seconds between auto-refresh polls. Set to 0 to stop polling; click Refresh to force a manual update. Automatically speeds up to 2s while any active job is running.">
        Poll <input type="number" id="monitor-poll-seconds" value="5" min="0" max="600"
                    style="width:60px;margin-left:4px;" onchange="restartMonitorPoll()">s
        <span id="monitor-poll-badge" style="display:none;margin-left:4px;font-weight:bold;"></span>
      </label>
      <button class="btn btn--sm" onclick="refreshMonitor()"
              title="Force an immediate refresh of every panel.">Refresh</button>
      <button class="btn btn--sm" onclick="downloadMonitorSnapshot()"
              title="Phase 54.6.255 — download the current /api/monitor JSON as a timestamped file. Useful for debugging / sharing state.">⬇ Snapshot</button>
      <button class="modal-close" onclick="closeModal('monitor-modal'); stopMonitorPoll();"
              title="Close the monitor. Stops the poll.">&times;</button>
    </div>
    <div class="modal-body">
      <!-- Phase 54.6.272 — keyboard-help overlay. Hidden by default,
           toggled by `?` or the `?` button below the filter input.
           Lists every hidden shortcut so the growing modal surface
           stays discoverable. --><div id="monitor-help-overlay"
           style="display:none;position:absolute;top:60px;right:20px;z-index:100;
           background:var(--bg,#fff);border:2px solid #36a;border-radius:6px;
           padding:0.75em 1em;max-width:520px;box-shadow:0 4px 16px rgba(0,0,0,0.15);">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5em;">
          <strong>Monitor keyboard shortcuts</strong>
          <button class="btn btn--sm" onclick="toggleMonitorHelp(false)" title="Close">✕</button>
        </div>
        <table style="font-size:0.9em;">
          <tr><td><kbd>/</kbd></td><td>Focus the filter input</td></tr>
          <tr><td><kbd>Esc</kbd> (in filter)</td><td>Clear filter</td></tr>
          <tr><td><kbd>?</kbd></td><td>Toggle this help overlay</td></tr>
        </table>
        <strong style="display:block;margin-top:0.75em;">Features</strong>
        <ul style="font-size:0.85em;margin:0.25em 0 0 1em;padding:0;">
          <li>Filter hides non-matching <em>data</em> rows (headers stay)</li>
          <li>Jump-to chips under the filter — click to scroll, URL hash updates (shareable <code>/#mon-...</code>)</li>
          <li>Alert banner — <strong>📋 Copy as MD</strong> exports all alerts as Markdown</li>
          <li>Each alert — <strong>📋</strong> per-alert copy of the suggested fix command</li>
          <li><strong>⬇ Snapshot</strong> button — download the current /api/monitor JSON</li>
          <li><strong>NEW</strong> badges flag alert codes not seen on this browser before</li>
          <li>Poll auto-speeds to 2 s while any job is running (badge appears in Poll label)</li>
          <li>Notifications fire on hidden tabs when a new error alert appears (permission prompt on first open)</li>
          <li>Log-tail panel at the bottom is collapsed by default — click to expand</li>
        </ul>
      </div>

      <!-- Phase 54.6.254 — live filter input. Case-insensitive
           substring match across all data rows. Hides non-matches
           in place (no re-fetch) so scroll + poll state is
           preserved. Reset with the ✕ or blank the input. -->
      <div style="display:flex;align-items:center;gap:0.75em;margin-bottom:0.75em;">
        <input type="text" id="monitor-filter"
               placeholder="Filter rows (press / to focus, Esc to clear)…"
               oninput="applyMonitorFilter()"
               onkeydown="if(event.key==='Escape'){{event.preventDefault();clearMonitorFilter();}}"
               style="flex:1;min-width:180px;padding:0.3em 0.5em;"
               title="Live filter across every row of every table in this modal. Blank = show all. Press / to focus from anywhere while the modal is open." />
        <button class="btn btn--sm" onclick="clearMonitorFilter()" title="Clear filter">✕</button>
        <span id="monitor-filter-count" class="u-muted" style="font-size:0.85em;"></span>
        <button class="btn btn--sm" onclick="toggleMonitorHelp()"
                title="Show keyboard shortcuts + feature index (or press ?)">?</button>
      </div>
      <!-- Phase 54.6.256 — jump-to navigation strip. Rebuilt after
           every render from the h4 headings inside monitor-content. -->
      <div id="monitor-nav-strip"
           style="display:flex;flex-wrap:wrap;gap:0.3em;margin-bottom:0.75em;"></div>
      <div id="monitor-content" style="font-size:0.92em;">
        <p class="u-note" style="text-align:center;padding:2em;">
          Loading system state…
        </p>
      </div>
      <p class="u-note" style="margin-top:1em;color:var(--fg-muted);">
        Mirrors <code>sciknow db monitor</code>. Same data source
        (<code>core.monitor.collect_monitor_snapshot</code>) feeds both
        CLI and GUI; the endpoint is <code>GET /api/monitor</code> if
        you want to pipe into another tool. Read-only — safe to leave
        open during active ingestion.
      </p>
    </div>
  </div>
</div>

<div class="modal-overlay" id="tools-modal" onclick="if(event.target===this)closeModal('tools-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-wrench"/></svg> Tools</h3>
      <button class="modal-close" onclick="closeModal('tools-modal')" title="Close the Tools modal.">&times;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="tl-search" onclick="switchToolsTab('tl-search')"
              title="Hybrid retrieval over the corpus. Chunk-level results with rerank — the CLI's `sciknow search query/similar`.">Search</button>
      <button class="tab" data-tab="tl-synth" onclick="switchToolsTab('tl-synth')"
              title="Multi-chunk synthesized answer with citations — the CLI's `sciknow ask synthesize`.">Synthesize</button>
      <button class="tab" data-tab="tl-topics" onclick="switchToolsTab('tl-topics')"
              title="Browse the BERTopic clusters and jump to papers in each topic.">Topics</button>
      <button class="tab" data-tab="tl-corpus" onclick="switchToolsTab('tl-corpus')"
              title="Grow / enrich / reconcile the corpus: enrich DOIs, follow citations, pull by author, inbound cites, topic search, coauthor snowball, agentic expansion, author oeuvre.">Corpus</button>
    </div>
    <div class="modal-body">

      <!-- Search tab (sciknow search query + search similar) -->
      <div id="tl-search-pane">
        <p class="u-note">
          Hybrid retrieval (dense + sparse + FTS) with cross-encoder rerank.
          Mirrors <code>sciknow search query</code> and <code>sciknow search similar</code>.
        </p>
        <div class="field u-row-wrap-end">
          <div class="u-col-220">
            <label>Query &nbsp;<span style="color:var(--fg-muted);font-weight:400;">or DOI / title fragment for similar-paper search</span></label>
            <input type="text" id="tl-search-q" placeholder="e.g. sea surface temperature reconstruction"
                   onkeydown="if(event.key==='Enter')doToolSearch('query')"
                   title="Free-text query OR a DOI/title fragment to find similar papers. Press Enter to run.">
          </div>
          <div class="u-col-wide"><label>Top-K</label>
            <input type="number" id="tl-search-topk" value="10" min="1" max="50"
                   title="How many reranked results to return."></div>
          <div class="u-col-90"><label>Year from</label>
            <input type="number" id="tl-search-yfrom"
                   title="Restrict retrieval to papers published in this year or later."></div>
          <div class="u-col-90"><label>Year to</label>
            <input type="number" id="tl-search-yto"
                   title="Restrict retrieval to papers published in this year or earlier."></div>
        </div>
        <div class="field u-row-wrap-end">
          <div class="u-col-160"><label>Section</label>
            <select id="tl-search-section"
                    title="Limit hits to chunks from a specific paper section.">
              <option value="">(any)</option>
              <option value="abstract">abstract</option>
              <option value="introduction">introduction</option>
              <option value="methods">methods</option>
              <option value="results">results</option>
              <option value="discussion">discussion</option>
              <option value="conclusion">conclusion</option>
            </select></div>
          <div class="u-col-160"><label>Topic cluster</label>
            <input type="text" id="tl-search-topic" placeholder="(any)"
                   title="Filter by BERTopic cluster label (substring match). Requires `catalog cluster` to have been run."></div>
          <div style="flex:1;min-width:100px;display:flex;align-items:center;">
            <label style="display:flex;align-items:center;gap:6px;font-weight:400;"
                   title="Ask an LLM to rewrite / expand your query (HyDE-style) before retrieval. Slight latency cost, sometimes big recall win for terse queries.">
              <input type="checkbox" id="tl-search-expand"
                     title="Ask an LLM to rewrite / expand your query (HyDE-style) before retrieval. Small latency cost, sometimes big recall win for terse queries."> LLM expand
            </label></div>
          <button class="btn-primary" onclick="doToolSearch('query')"
                  title="Run the hybrid search. Streams results inline.">Search</button>
          <button onclick="doToolSearch('similar')" title="Find papers with a similar abstract to the one you typed (DOI or title fragment)">Similar</button>
        </div>
        <div class="u-small u-muted u-my-1" id="tl-search-status"></div>
        <div class="u-mt-2" id="tl-search-results"></div>
      </div>

      <!-- Synthesize tab (sciknow ask synthesize) -->
      <div class="u-hidden" id="tl-synth-pane">
        <p class="u-note">
          Multi-paper synthesis on a topic &mdash; biases the prompt toward
          consensus, methods and open questions. Mirrors <code>sciknow ask synthesize</code>.
          (For single Q&amp;A use <strong>&#128270; Ask Corpus</strong> in the toolbar.)
        </p>
        <div class="field">
          <label>Topic</label>
          <input type="text" id="tl-synth-topic" placeholder="e.g. solar activity and climate variability"
                 onkeydown="if(event.key==='Enter')doToolSynthesize()"
                 title="Topic to synthesize across. Press Enter to run.">
        </div>
        <div class="field u-row-wrap-end">
          <div class="u-flex-1"><label>Context-K</label>
            <input type="number" id="tl-synth-k" value="12" min="4" max="30"
                   title="How many reranked chunks to feed the synthesizer. Higher = broader context but longer prompt."></div>
          <div class="u-flex-1"><label>Year from</label><input type="number" id="tl-synth-yfrom"
                   title="Lower year bound for retrieval."></div>
          <div class="u-flex-1"><label>Year to</label><input type="number" id="tl-synth-yto"
                   title="Upper year bound for retrieval."></div>
          <div class="u-flex-2"><label>Topic cluster filter</label>
            <input type="text" id="tl-synth-topicfilter" placeholder="(any)"
                   title="Restrict the synthesis to papers whose BERTopic cluster label matches (substring)."></div>
          <button class="btn-primary" onclick="doToolSynthesize()"
                  title="Retrieve + rerank → LLM synthesizes a consensus-biased answer with citations.">Synthesize</button>
        </div>
        <div class="u-note-sm" id="tl-synth-status"></div>
        <div class="modal-stream" id="tl-synth-stream"></div>
        <div id="tl-synth-stats" class="stream-stats"></div>
        <div class="modal-sources u-hidden" id="tl-synth-sources"></div>
      </div>

      <!-- Topics tab (sciknow catalog topics + Phase 46.E domain tags) -->
      <div class="u-hidden" id="tl-topics-pane">
        <div class="u-flex-raw u-gap-3 u-ai-start">
          <div class="u-flex-2">
            <p class="u-note">
              <strong>Topic clusters</strong> (from <code>sciknow catalog cluster</code>).
              Click a cluster to see its papers. Ranked by paper count.
            </p>
            <div class="u-flex-raw u-wrap u-gap-6 u-mb-3" id="tl-topics-list"></div>
          </div>
          <div style="flex:1;border-left:1px solid var(--border);padding-left:12px;min-width:180px;">
            <p class="u-note">
              <strong>Domain tags</strong> (from <code>paper_metadata.domains</code>).
              Empty if no tags are populated for this corpus.
            </p>
            <div class="u-flex-raw u-wrap u-gap-1" id="tl-domains-list"></div>
          </div>
        </div>
        <div id="tl-topics-papers"></div>
      </div>


    </div>
  </div>
</div>

<!-- Phase 54.6.18 — standalone Corpus modal. Moved out of the
     Tools modal so the top-bar "Corpus ▾" dropdown can surface it
     directly. The pane's internal IDs and sub-tab logic are
     untouched — openCorpusModal() just unhides the pane + calls
     switchCorpusTab to pick the right sub-surface. -->
<div class="modal-overlay" id="corpus-modal"
     onclick="if(event.target===this)closeModal('corpus-modal')">
  <div class="modal wide xwide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-sprout"/></svg> Corpus &mdash; Enrich &amp; Expand</h3>
      <button class="modal-close" onclick="closeModal('corpus-modal')" title="Close the Corpus modal.">&times;</button>
    </div>
    <div class="modal-body">
      <!-- Corpus tab (Phase 46.E + 54.6.4 — expand/enrich/cleanup) -->
      <div class="u-hidden" id="tl-corpus-pane">
        <p class="u-note-mb-3">
          Grow and enrich the paper corpus from the browser. Pick one of
          the modes below. All are long-running; the log streams at the
          bottom. Cancel with the red button.
        </p>
        <!-- Phase 54.6.4 + 54.6.7 — utility buttons at the top of the
             Corpus tab: cleanup (dedup) + pending-downloads (retry /
             manual acquisition for papers without a legal OA PDF). -->
        <div class="u-flex-raw u-gap-2 u-ai-center u-mb-m u-p-2 u-bg-alt u-r-md u-small u-wrap">
          <button class="btn-secondary" onclick="doToolCorpus('cleanup')"
                  title="Remove PDFs from downloads/ and data/inbox/ that are already ingested in this or any other project's DB (inbox also gets empty subfolders removed), AND permanently nuke failed-ingest PDFs (data/failed/ + downloads/failed_ingest/) plus their documents rows. Frees disk; pipeline archive is preserved.">
            &#129529; Cleanup downloads + inbox + failed
          </button>
          <button class="btn-secondary" onclick="openPendingDownloadsModal()"
                  title="Papers you selected but couldn't be auto-downloaded (no legal OA PDF). Retry, mark manually acquired, or export for ILL.">
            &#128203; Pending downloads
          </button>
          <button class="btn-secondary" onclick="runCorpusCliAction(['db','refresh-retractions'], 'Retraction sweep…')"
                  title="Phase 54.6.111 — query Crossref's update-type:retraction index for every paper with a DOI, flag retracted/corrected. Skips papers checked within 30 days.">
            &#128203; Retraction sweep
          </button>
          <button class="btn-secondary" onclick="runCorpusCliAction(['db','reconcile-preprints','--dry-run'], 'Detecting preprint+journal pairs…')"
                  title="Phase 54.6.125 — detect preprint↔journal duplicates via OpenAlex work_id grouping. Shows the plan without applying.">
            &#128202; Detect duplicates
          </button>
          <button class="btn-secondary" onclick="if(confirm('Apply reconciliations? Non-canonical rows will be hidden from retrieval (reversible via Reconciliations list).')) runCorpusCliAction(['db','reconcile-preprints'], 'Reconciling preprint+journal pairs…')"
                  title="Phase 54.6.125 — apply preprint↔journal reconciliation. Non-canonical rows get canonical_document_id set; hidden from retrieval, fully reversible via `db unreconcile` or the Reconciliations tab.">
            &#128279; Reconcile preprints
          </button>
          <button class="btn-secondary" onclick="openReconciliationsModal()"
                  title="List all active reconciliations; reverse individual mappings.">
            &#128203; Reconciliations
          </button>
          <span class="u-muted">
            Cleanup removes already-ingested dupes <em>and</em> the failed-ingest archive. Pending lists papers still waiting on an OA PDF. Retraction sweep flags newly-retracted work.
          </span>
        </div>
        <div class="tabs u-mb-m">
          <button class="tab active" data-ctab="corp-enrich" onclick="switchCorpusTab('corp-enrich')"
                  title="Fill missing DOIs via Crossref/OpenAlex/arXiv title-search and persist OpenAlex concepts/funders/grants/ROR. No new papers downloaded.">&#128270; Enrich</button>
          <button class="tab" data-ctab="corp-cites" onclick="switchCorpusTab('corp-cites')"
                  title="Follow references FROM each corpus paper → download OA PDFs → ingest. The classic expand command.">&#127760; Expand (citations)</button>
          <button class="tab" data-ctab="corp-agentic" onclick="switchCorpusTab('corp-agentic')" title="Phase 54.6.114 — LLM decomposes a research question into sub-topics, measures corpus coverage, auto-expands gaps.">&#129504; Agentic (question-driven)</button>
          <button class="tab" data-ctab="corp-author" onclick="switchCorpusTab('corp-author')"
                  title="Pull every paper by a specific author (ORCID-preferred). Strict-author matching to defend against name collisions.">&#128100; Expand by author</button>
          <button class="tab" data-ctab="corp-author-refs" onclick="switchCorpusTab('corp-author-refs')"
                  title="Phase 54.6.312 — Aggregate every paper cited by one author's corpus works (including self-cites), rank by citation frequency, and cherry-pick before download. Mirrors `sciknow db expand-author-refs`.">&#128218; Expand by author's refs</button>
          <button class="tab" data-ctab="corp-inbound" onclick="switchCorpusTab('corp-inbound')"
                  title="Find papers that CITE the corpus — forward-in-time mirror of db expand. Uses OpenAlex cites: filter.">&#128258; Inbound cites</button>
          <button class="tab" data-ctab="corp-topic" onclick="switchCorpusTab('corp-topic')"
                  title="Free-text OpenAlex search. Push-based expansion — solves bootstrap / sideways-expansion problem.">&#128269; Topic search</button>
          <button class="tab" data-ctab="corp-coauth" onclick="switchCorpusTab('corp-coauth')"
                  title="Snowball via coauthors of corpus papers. Captures the 'invisible college' — same-lab researchers who don't always cite each other.">&#128101; Coauthors</button>
        </div>

        <!-- Enrich (metadata) panel -->
        <div class="u-card" id="corp-enrich-pane">
          <div class="u-note-sm">
            Fill missing DOIs via Crossref / OpenAlex / arXiv title search.
            Mirrors <code>sciknow db enrich</code>.
          </div>
          <div class="field u-row-wrap-end-tight">
            <div class="u-col-70"><label>Limit</label>
              <input type="number" id="tl-enr-limit" value="0" min="0"
                     title="Max papers to process this run. 0 = all papers lacking a DOI."/></div>
            <div class="u-col-num"><label>Threshold</label>
              <input type="number" id="tl-enr-thresh" value="0.85" min="0" max="1" step="0.01"
                     title="Minimum title-similarity score (0–1) to accept a Crossref match. 0.85 is conservative; 0.78 is the 54.6.x dual-signal default when author+year also agree."/></div>
            <label class="u-label-row">
              <input type="checkbox" id="tl-enr-dry"
                     title="Show what would be updated without writing to the database."> dry-run
            </label>
          </div>
          <button class="btn-primary u-mt-2" onclick="doToolCorpus('enrich')"
                  title="Run `db enrich`: Crossref/OpenAlex/arXiv title search to fill missing DOIs + persist OpenAlex concepts/funders/grants/ROR. Streams logs into the console below.">Run Enrich</button>
        </div>

        <!-- Expand by citations -->
        <div class="u-hidden-card" id="corp-cites-pane">
          <div class="u-note-sm">
            Follow references &rarr; download OA PDFs &rarr; ingest.
            Mirrors <code>sciknow db expand</code>.
          </div>
          <div class="field u-row-wrap-end-tight">
            <div class="u-col-70"><label title="Hard cap on total papers downloaded this run. 0 = no cap.">Limit</label>
              <input type="number" id="tl-exp-limit" value="0" min="0"
                     title="Hard cap on total papers downloaded this run. 0 = no cap."></div>
            <div class="u-col-70"><label title="Phase 54.6.113 — RRF pool size per round. The ranker fuses signals, applies MMR diversity, then takes the top-N for the download phase. Default 50. Smaller = tighter top picks only.">Budget</label>
              <input type="number" id="tl-exp-budget" value="50" min="5" max="200"
                     title="Phase 54.6.113 — RRF pool size per round. Smaller = tighter top picks only."></div>
            <div class="u-col-num"><label>Workers</label>
              <input type="number" id="tl-exp-workers" value="0" min="0"
                     title="Parallel ingestion worker subprocesses. 0 = use INGEST_WORKERS from .env (default 1). Each worker loads its own MinerU (~7GB VRAM) + bge-m3 — raise only when the LLM is off-GPU."/></div>
            <div class="u-col-num"><label>Relev. thr</label>
              <input type="number" id="tl-exp-relthr" value="0.0" min="0" max="1" step="0.05"
                     title="Cosine similarity floor against the corpus centroid. 0 = use EXPAND_RELEVANCE_THRESHOLD from .env (default 0.55). Raise to require tighter topical match."/></div>
          </div>
          <div class="field u-flex-raw u-gap-10 u-ai-center u-wrap u-mt-1">
            <label class="u-label-row">
              <input type="checkbox" id="tl-exp-dry"
                     title="Run the ranker + preview shortlist without downloading any PDFs. Writes the shortlist TSV to data/downloads/expand_shortlist.tsv."> dry-run
            </label>
            <label class="u-label-row">
              <input type="checkbox" id="tl-exp-resolve"
                     title="Also resolve title-only references (no DOI in source) via Crossref title-search. Slow (~0.3s per title)."> resolve titles
            </label>
            <label class="u-label-row"
                   title="Run the full ingest pipeline (convert → metadata → chunk → embed) on downloaded PDFs. Uncheck to only download into data/downloads without ingesting.">
              <input type="checkbox" id="tl-exp-ingest" checked
                     title="Run the full convert → metadata → chunk → embed pipeline on downloaded PDFs. Uncheck for download-only."> ingest
            </label>
            <label class="u-label-row"
                   title="Gate downloads by centroid/anchor cosine threshold. Uncheck to download every ranked candidate regardless of topical similarity.">
              <input type="checkbox" id="tl-exp-relevance" checked
                     title="Gate downloads by centroid/anchor cosine threshold. Uncheck to download every ranked candidate."> relevance filter
            </label>
          </div>
          <div class="field u-mt-1 u-flex-raw u-gap-2 u-ai-end">
            <div class="u-flex-3">
              <label>Relevance anchor query (optional)</label>
              <input type="text" id="tl-exp-relq" placeholder="(corpus centroid if blank)"
                     title="Free-text anchor for relevance scoring. Leave blank to use the corpus centroid (avg of all paper embeddings). A sharp query tightens results; a broad corpus benefits from a focused anchor.">
            </div>
            <div class="u-flex-2">
              <label>Anchor from topic</label>
              <select id="tl-exp-relq-topic" onchange="if(this.value){{document.getElementById('tl-exp-relq').value=this.value;}}"
                      title="Shortcut: pick one of your catalog topic clusters and its label will be copied into the anchor textbox on the left.">
                <option value="">(pick a topic…)</option>
              </select>
            </div>
          </div>
          <div class="u-row-mt">
            <button class="btn-primary" onclick="openExpandCitesPreview()"
                    title="Build the ranked shortlist and open the preview modal. You can inspect, filter and individually check candidates before downloading.">
              &#128269; Preview candidates
            </button>
            <button class="btn-secondary" onclick="doToolCorpus('expand')"
                    title="Skip preview — run the full pipeline with auto-download using the relevance threshold.">
              &#9889; Auto-download (override)
            </button>
            <span class="u-hint">
              Preview shows the ranked shortlist (RRF signals) and lets you check papers one-by-one.
            </span>
          </div>
        </div>

        <!-- Phase 54.6.114 (Tier 2 #2) — Agentic question-driven expansion -->
        <div class="u-hidden-card" id="corp-agentic-pane">
          <div class="u-note-md">
            <strong>Agentic mode.</strong> Give a research question; the LLM decomposes it
            into 3-6 sub-topics, measures corpus coverage for each (via hybrid search),
            and runs targeted expansion on the gaps. Stops when every sub-topic has
            ≥ <em>threshold</em> papers OR when <em>max rounds</em> is reached. Uses
            the EXISTING Phase 49 RRF ranker (with the new citation-context signal
            from Tier 2 #1) for each sub-topic.
          </div>
          <div class="field">
            <label>Research question</label>
            <textarea class="u-w-full u-p-2 u-md u-lh-145" id="tl-ag-question" rows="3"
                      placeholder="e.g. What is the current best estimate of equilibrium climate sensitivity, and how confident are we?"></textarea>
          </div>
          <div class="field u-flex-raw u-gap-2 u-ai-end u-wrap u-mt-1">
            <div class="u-col-num"><label>Max rounds</label>
              <input type="number" id="tl-ag-rounds" value="3" min="1" max="8"
                     title="Max agentic rounds. Each round re-measures coverage and expands remaining gaps."></div>
            <div class="u-col-num"><label>Budget / gap</label>
              <input type="number" id="tl-ag-budget" value="10" min="1" max="60"
                     title="Max papers to download per gap sub-topic per round."></div>
            <div class="u-col-num"><label>Cover threshold</label>
              <input type="number" id="tl-ag-threshold" value="3" min="1" max="20"
                     title="Corpus papers required to call a sub-topic 'covered'."></div>
            <label class="u-label-row">
              <input type="checkbox" id="tl-ag-dry"
                     title="Compute the plan + coverage measurement without downloading. Useful to confirm sub-topic decomposition before burning API calls."> dry-run
            </label>
            <label class="u-label-row"
                   title="Phase 54.6.124 — resume from a prior checkpoint. The same question re-uses its state via a slug+hash path at <project>/data/expand/agentic/">
              <input type="checkbox" id="tl-ag-resume"
                     title="Phase 54.6.124 — resume from a prior checkpoint. Same question → same state path at <project>/data/expand/agentic/"> resume
            </label>
          </div>
          <div class="u-flex-raw u-gap-2 u-ai-center u-mt-10 u-wrap">
            <button class="btn-primary" onclick="openAgenticPreview()"
                    title="Phase 54.6.133 — INTERACTIVE per-round preview using the SAME Phase-49 RRF ranker auto-mode runs. Decomposes the question, measures coverage, runs the ranker per gap sub-topic, and shows you exactly the candidates auto would download — you cherry-pick before any disk write. ~30-90s per gap (typically 3-7 min total for 5 gaps). After ingestion, click again to advance to round 2 (coverage gets re-measured). Recommended for personal-bibliography work.">
              &#128269; Preview round candidates
            </button>
            <button class="btn-secondary" onclick="runAgenticExpand()"
                    title="AUTO mode: LLM decomposes the question → measures coverage per sub-topic → runs expand on gaps → re-plans. Loops automatically until every sub-topic is covered or max rounds reached. Worst case: rounds × gaps × budget papers downloaded WITHOUT user approval (default 3 × 5 × 10 = 150).">&#129504; Start agentic expansion (auto)</button>
            <span class="u-hint">
              Auto streams progress into the log panel. Preview opens an interactive shortlist per round.
            </span>
          </div>

          <!-- Phase 54.6.116 (Tier 2 #4) — author oeuvre completion -->
          <div class="u-divider">
            <h5 class="u-mh-6 u-small">&#128100; Author oeuvre completion</h5>
            <p style="font-size:11px;color:var(--fg-muted);margin:0 0 8px;line-height:1.4;">
              Scan the corpus, find authors with ≥ N papers already present, run
              <code>expand-author</code> for each (ORCID-preferred, strict-author). Uses the same relevance + retraction + MMR filters as any other expansion.
            </p>
            <div class="u-row-wrap-end">
              <div><label class="u-tiny" title="Minimum number of papers an author must already have in this corpus to qualify for oeuvre expansion. Lower = more authors, noisier; higher = focus on the highly-represented names.">Min corpus papers
                <input class="u-input-60" type="number" id="tl-oeu-min" value="3" min="2" max="10"
                       title="Minimum corpus papers for an author to qualify. Default 3."></label></div>
              <div><label class="u-tiny" title="Max new papers fetched per qualifying author, passed as --limit to expand-author.">Per-author limit
                <input class="u-input-60" type="number" id="tl-oeu-limit" value="10" min="1" max="50"
                       title="Max new papers per qualifying author. Default 10."></label></div>
              <div><label class="u-tiny" title="Cap on how many authors get processed this run. Prevents an unbounded sweep over the whole corpus.">Max authors
                <input class="u-input-60" type="number" id="tl-oeu-max" value="10" min="1" max="30"
                       title="Cap on authors processed this run. Default 10, ordered by corpus citation count."></label></div>
              <label class="u-kv-xs"
                     title="Compute the author list + per-author plan without downloading. Useful to confirm which names will be processed.">
                <input type="checkbox" id="tl-oeu-dry"
                       title="Compute the author list + per-author plan without downloading."> dry-run
              </label>
              <button class="btn-primary" onclick="openExpandOeuvrePreview()"
                      title="Phase 54.6.131 — Pre-fetch every qualifying author's candidates without downloading. Surfaces the merged shortlist (annotated with the source author per row) in the candidates modal so you can cherry-pick before any disk write.">
                &#128269; Preview oeuvre candidates
              </button>
              <button class="btn-secondary" onclick="runOeuvreExpand()"
                      title="Auto-mode: scan corpus → find authors with ≥ Min papers → loop expand-author over them with the configured limits. ORCID-preferred, strict-author match. NO preview — downloads and ingests every candidate above the relevance threshold.">Run oeuvre expansion</button>
            </div>
          </div>
        </div>

        <!-- Phase 46.E — Expand by author panel -->
        <div class="u-hidden-card" id="corp-author-pane">
          <div class="u-note-md">
            Find every paper by an author across OpenAlex + Crossref, then
            download the open-access ones and ingest. Mirrors
            <code>sciknow db expand-author</code>. The picker below ranks
            authors by <strong>citation count</strong> (within this
            corpus) then paper count, so the most-authoritative names
            surface first.
          </div>
          <div class="field u-row-end">
            <div class="u-flex-3">
              <label>Search author</label>
              <input type="text" id="tl-eauth-q"
                     placeholder="Type a name — e.g. Solanki, Lockwood…"
                     oninput="onExpandAuthorSearchInput(event)"
                     onkeydown="if(event.key==='Enter'){{event.preventDefault();onExpandAuthorSearchInput(event);}}"
                     title="Search OpenAlex authors by name. Results are ranked by citations within THIS corpus, then by overall paper count. Click a row to select.">
            </div>
            <div class="u-flex-2">
              <label>ORCID (optional)</label>
              <input type="text" id="tl-eauth-orcid" placeholder="0000-0000-0000-0000"
                     title="Pin to a specific ORCID iD — use when two authors share a name. Overrides name-search disambiguation.">
            </div>
          </div>
          <div id="tl-eauth-results"
               style="max-height:260px;overflow:auto;border:1px solid var(--border);
                      border-radius:4px;margin-top:6px;font-size:12px;display:none;"></div>
          <div class="u-mt-6 u-small u-muted" id="tl-eauth-selected">
            No author selected yet — search above and click a row.
          </div>
          <div class="field u-flex-raw u-gap-6 u-ai-end u-wrap u-mt-6">
            <div class="u-col-num"><label>Year from</label>
              <input type="number" id="tl-eauth-yfrom" placeholder="(any)"
                     title="Only fetch papers published in this year or later. Leave blank for no lower bound."></div>
            <div class="u-col-num"><label>Year to</label>
              <input type="number" id="tl-eauth-yto" placeholder="(any)"
                     title="Only fetch papers published in this year or earlier. Leave blank for no upper bound."></div>
            <div class="u-col-70"><label>Limit</label>
              <input type="number" id="tl-eauth-limit" value="0" min="0" title="Cap on papers downloaded for this author. 0 = no cap (fetch all)."></div>
            <div class="u-col-num"><label>Workers</label>
              <input type="number" id="tl-eauth-workers" value="0" min="0"
                     title="Parallel download workers. 0 = use EXPAND_WORKERS from .env (default 8)."></div>
            <div class="u-col-num"><label>Relev. thr</label>
              <input type="number" id="tl-eauth-relthr" value="0.0" min="0" max="1" step="0.05"
                     title="Cosine-similarity floor against the corpus centroid or anchor. 0 = use EXPAND_RELEVANCE_THRESHOLD from .env (default 0.55)."></div>
          </div>
          <div class="field u-flex-raw u-gap-10 u-ai-center u-wrap u-mt-1">
            <label class="u-label-row"
                   title="Filter fetched papers to keep only those where this author is actually on the authorship list (defends against OpenAlex name-collision hits).">
              <input type="checkbox" id="tl-eauth-strict" checked
                     title="Drop fetched papers where the queried author is NOT on the authorship list. Defends against OpenAlex name collisions."> strict author match
            </label>
            <label class="u-label-row"
                   title="Skip the interactive disambiguation banner and keep every hit that matches the queried name. Use for truly unambiguous names.">
              <input type="checkbox" id="tl-eauth-all"
                     title="Skip the multi-candidate disambiguation banner and keep every hit matching the queried name. Use only for unambiguous names."> keep all matches (skip disamb.)
            </label>
            <label class="u-label-row"
                   title="Gate downloads by centroid/anchor cosine threshold. Uncheck to download every paper by this author regardless of topical similarity.">
              <input type="checkbox" id="tl-eauth-relevance" checked
                     title="Gate downloads by centroid/anchor cosine threshold. Uncheck to download every paper by this author."> relevance filter
            </label>
            <label class="u-label-row"
                   title="Run the full convert → metadata → chunk → embed pipeline on downloaded PDFs. Uncheck to download-only into data/downloads/.">
              <input type="checkbox" id="tl-eauth-ingest" checked
                     title="Run the full ingest pipeline on downloaded PDFs. Uncheck for download-only."> ingest
            </label>
            <label class="u-label-row"
                   title="Compute the plan (candidate list + relevance scores) without downloading PDFs.">
              <input type="checkbox" id="tl-eauth-dry"
                     title="Compute the plan (candidate list + relevance scores) without downloading PDFs."> dry-run
            </label>
          </div>
          <div class="field u-mt-1">
            <label>Relevance anchor query (optional)</label>
            <input type="text" id="tl-eauth-relq"
                   placeholder="(corpus centroid if blank)"
                   title="Free-text anchor for relevance scoring. Leave blank to use the corpus centroid. A focused anchor helps keep the author's off-topic papers out.">
          </div>
          <div class="u-row-mt">
            <button class="btn-primary" onclick="openExpandAuthorPreview()"
                    title="Build the author's candidate shortlist and open the cherry-pick preview. Nothing downloads until you tick rows.">
              &#128269; Preview candidates
            </button>
            <button class="btn-secondary" onclick="doToolCorpus('expand-author')"
                    title="Skip preview — run the full pipeline with the relevance-filter threshold auto-downloading everything above it. Equivalent to `sciknow db expand-author --relevance`.">
              &#9889; Auto-download (override)
            </button>
            <span class="u-hint">
              Preview lets you cherry-pick; Auto uses the relevance threshold.
            </span>
          </div>
        </div>

        <!-- Phase 54.6.312 — Expand-by-author's references.
             Phase 54.6.314 — live-autocomplete author picker copied
             from corp-author-pane (different IDs: tl-earef-* so they
             coexist). Same /api/catalog/authors backend feeds it.
             Clicking a row fills the selection state; "Preview
             references" calls the existing openExpandAuthorRefs()
             flow with the pre-selected author name. -->
        <div class="u-hidden-card" id="corp-author-refs-pane">
          <div class="u-note-md">
            Aggregate every paper <strong>cited by</strong> one author's
            corpus works (including self-cites), rank by citation
            frequency, cherry-pick the ones to fetch, then download +
            ingest via the shared expand-author pipeline. Mirrors
            <code>sciknow db expand-author-refs</code>.
          </div>
          <div class="field u-row-end">
            <div class="u-flex-3">
              <label>Search author</label>
              <input type="text" id="tl-earef-q"
                     placeholder="Type a name — e.g. Solanki, Lockwood…"
                     oninput="onExpandAuthorRefsSearchInput(event)"
                     onkeydown="if(event.key==='Enter'){{event.preventDefault();onExpandAuthorRefsSearchInput(event);}}"
                     title="Search corpus authors by name (same backend as 'Expand by author'). Click a row to select.">
            </div>
            <div class="u-flex-2">
              <label>Min mentions</label>
              <input type="number" id="tl-earef-min" value="1" min="1"
                     title="Only include references cited at least this many times across the author's corpus papers. Raise to trim the long tail of single-cite references.">
            </div>
          </div>
          <div id="tl-earef-results"
               style="max-height:260px;overflow:auto;border:1px solid var(--border);
                      border-radius:4px;margin-top:6px;font-size:12px;display:none;"></div>
          <div class="u-mt-6 u-small u-muted" id="tl-earef-selected">
            No author selected yet — search above and click a row.
          </div>
          <div class="u-row-mt u-flex-raw u-gap-2 u-ai-center u-wrap">
            <button class="btn-primary" onclick="runExpandAuthorRefsPreview()"
                    title="Aggregate every reference cited by this author's corpus works, rank by citation frequency, and open the cherry-pick preview. No disk writes until you tick rows.">
              &#128269; Preview references
            </button>
            <span class="u-hint">
              Preview opens the shared candidates modal for multi-select before any download.
            </span>
          </div>
        </div>

        <!-- Phase 54.6.4 — Inbound cites panel -->
        <div class="u-hidden-card" id="corp-inbound-pane">
          <div class="u-note-md">
            Find papers that <strong>cite</strong> papers already in your
            corpus — the forward-in-time mirror of <code>db expand</code>.
            Calls OpenAlex's <code>/works?filter=cites:W…</code> per seed.
            Mirrors <code>sciknow db expand-cites</code>.
          </div>
          <div class="field u-row-wrap-end-tight">
            <div class="u-col-120"><label>Per-seed cap</label>
              <input type="number" id="tl-inb-seed" value="30" min="1" title="Max citing-papers fetched per corpus seed. Keeps viral-cited seeds from dominating the shortlist."></div>
            <div class="u-col-wide"><label>Total limit</label>
              <input type="number" id="tl-inb-total" value="300" min="10"
                     title="Hard cap on the combined candidate pool across all seeds."></div>
          </div>
          <div class="field u-mt-1">
            <label>Relevance anchor query (optional)</label>
            <input type="text" id="tl-inb-relq" placeholder="(corpus centroid if blank)"
                   title="Free-text anchor for relevance scoring. Leave blank to use the corpus centroid (avg of all paper embeddings).">
          </div>
          <!-- Phase 54.6.123 (Tier 3 #2) — full-pipeline inbound crawl -->
          <div class="field u-flex-raw u-gap-2 u-ai-end u-wrap u-mt-1">
            <div class="u-col-num"><label>Limit</label>
              <input type="number" id="tl-inb-limit" value="20" min="1" max="200"
                     title="Max papers to download + ingest this run."></div>
            <div class="u-col-num"><label>Relev. thr</label>
              <input type="number" id="tl-inb-relthr" value="0.55" min="0" max="1" step="0.05"
                     title="Drop candidates below this bge-m3 cosine score."></div>
            <label class="u-kv-xs"
                   title="Compute the candidate shortlist without downloading. Useful to check per-seed fan-out sizes.">
              <input type="checkbox" id="tl-inb-dry"
                     title="Compute the candidate shortlist without downloading. Useful to check per-seed fan-out sizes."> dry-run
            </label>
            <label class="u-kv-xs"
                   title="Ignore the no_oa / ingest_failed caches.">
              <input type="checkbox" id="tl-inb-retry"
                     title="Ignore the .no_oa_cache + .ingest_failed sidecar files for this batch."> retry failed
            </label>
          </div>
          <div class="u-row-mt">
            <button class="btn-primary" onclick="openExpandInboundPreview()"
                    title="Build the cites-me shortlist and open the preview modal. You can cherry-pick rows before downloading.">&#128269; Preview candidates</button>
            <button class="btn-primary" onclick="runInboundExpand()"
                    title="Phase 54.6.123 — runs the full cites-me pipeline: crawl → relevance filter → download → ingest.">
              &#127793; Expand now
            </button>
            <span class="u-hint">
              Preview shows the shortlist without downloading. Expand now runs the full pipeline.
            </span>
          </div>
        </div>

        <!-- Phase 54.6.4 — Topic search panel -->
        <div class="u-hidden-card" id="corp-topic-pane">
          <div class="u-note-md">
            Free-text OpenAlex search. Push-based expansion — solves the
            bootstrap / sideways-expansion problem <code>expand</code> can't
            address. Mirrors <code>sciknow db expand-topic "QUERY"</code>.
          </div>
          <div class="field">
            <label>Topic query</label>
            <input type="text" id="tl-top-q" placeholder="e.g. thermospheric cooling"
                   onkeydown="if(event.key==='Enter')openExpandTopicPreview()"
                   title="Free-text search over OpenAlex title/abstract/keywords. Press Enter to run the preview.">
          </div>
          <div class="field u-row-wrap-end-tight">
            <div class="u-col-wide"><label>Limit</label>
              <input type="number" id="tl-top-limit" value="300" min="10"
                     title="How many OpenAlex hits to pull before relevance filtering. Results come back sorted by citation count."></div>
            <div style="flex:3;min-width:200px;"><label>Relevance anchor (defaults to the query itself)</label>
              <input type="text" id="tl-top-relq" placeholder="(query if blank)"
                     title="Override the anchor used for relevance scoring. Leave blank to score against the topic query."></div>
          </div>
          <div class="u-row-mt">
            <button class="btn-primary" onclick="openExpandTopicPreview()"
                    title="Run the topic search, score candidates by relevance, and open the cherry-pick preview.">&#128269; Preview candidates</button>
            <span class="u-hint">
              Results sorted by citation count, then filtered by relevance.
            </span>
          </div>
        </div>

        <!-- Phase 54.6.4 — Coauthor snowball panel -->
        <div class="u-hidden-card" id="corp-coauth-pane">
          <div class="u-note-md">
            Fetch papers by every OpenAlex author on any paper in the
            corpus (depth=1). Captures the <em>invisible college</em> —
            researchers in the same lab who don't always cite each
            other directly. Mirrors <code>sciknow db expand-coauthors</code>.
          </div>
          <div class="field u-row-wrap-end-tight">
            <div class="u-col-wide"><label>Depth</label>
              <select id="tl-coa-depth"
                      title="Depth of the coauthor graph walk. 1 = coauthors of corpus papers. 2 = coauthors-of-coauthors (much noisier — use only with a tight relevance threshold).">
                <option value="1" selected>1 (recommended)</option>
                <option value="2">2 (noisy — use with strict threshold)</option>
              </select></div>
            <div class="u-col-120"><label>Per-author cap</label>
              <input type="number" id="tl-coa-per" value="8" min="1" title="Max papers per coauthor. Keeps prolific coauthors from swamping the shortlist."></div>
            <div class="u-col-wide"><label>Total limit</label>
              <input type="number" id="tl-coa-total" value="300" min="10"
                     title="Hard cap on the combined candidate pool across all coauthors."></div>
          </div>
          <div class="field u-mt-1">
            <label>Relevance anchor query (optional)</label>
            <input type="text" id="tl-coa-relq" placeholder="(corpus centroid if blank)"
                   title="Free-text anchor for relevance scoring. Leave blank to use the corpus centroid. For coauthor snowballing a focused anchor is strongly recommended.">
          </div>
          <div class="u-row-mt">
            <button class="btn-primary" onclick="openExpandCoauthPreview()"
                    title="Build the coauthor shortlist and open the cherry-pick preview. No downloads until you approve rows.">&#128269; Preview candidates</button>
            <span class="u-hint">
              Best with a tight relevance threshold (0.6+) — this method has the noisiest recall.
            </span>
          </div>
        </div>

        <div class="u-flex-raw u-gap-2 u-ai-center u-mt-3">
          <div class="u-flex-1 u-small u-muted" id="tl-corpus-status"
               title="Current status of the running command (enrich/expand/agentic/…). Updated as events stream in."></div>
          <button id="tl-corpus-cancel" onclick="cancelToolCorpus()"
                  style="display:none;background:var(--danger,#c53030);color:white;border:0;padding:4px 10px;border-radius:4px;cursor:pointer;"
                  title="Stop the currently running corpus command. In-flight downloads and ingestion jobs are interrupted at the next checkpoint.">
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
<div class="modal-overlay" id="candidates-preview-modal"
     onclick="if(event.target===this)closeModal('candidates-preview-modal')">
  <div class="modal wide xwide">
    <div class="modal-header">
      <h3 id="eap-title">&#128269; Preview Candidates</h3>
      <button class="modal-close" onclick="closeModal('candidates-preview-modal')" title="Close the Candidates Preview modal.">&times;</button>
    </div>
    <div class="modal-body">
      <div class="u-p-20 u-text-center u-muted" id="eap-loading">
        <div class="u-lg" id="eap-loading-msg">Searching&hellip;</div>
        <div class="u-tiny u-mt-1 u-muted" id="eap-loading-sub"></div>
        <pre id="eap-loading-log" style="display:none;margin-top:10px;max-height:240px;overflow:auto;background:var(--bg-alt,#f8f8f8);border:1px solid var(--border);border-radius:4px;padding:8px;font-size:11px;font-family:ui-monospace,monospace;white-space:pre-wrap;text-align:left;"></pre>
      </div>
      <div id="eap-error" style="display:none;padding:10px;background:var(--danger-bg,#fee);color:var(--danger,#c53030);border:1px solid var(--danger,#c53030);border-radius:4px;margin-bottom:10px;"></div>
      <div class="u-hidden" id="eap-content">
        <div class="u-small u-muted u-mb-m u-p-2 u-bg-alt u-r-sm" id="eap-info"></div>
        <div class="u-row-wrap-mb">
          <button class="btn-secondary" onclick="eapSelectAll(true)"
                  title="Check every row in the current filtered view.">Select all</button>
          <button class="btn-secondary" onclick="eapSelectAll(false)"
                  title="Uncheck every row.">Select none</button>
          <label class="u-kv-row">
            Select where score &ge;
            <input class="u-w-70 u-small u-p-2-4" type="number" id="eap-threshold" value="0.55" min="0" max="1" step="0.05"
                   title="bge-m3 cosine similarity threshold (0–1). 0.55 is a reasonable floor; raise for a tighter shortlist, lower to see more."/>
            <button class="btn-secondary u-pill-xs" onclick="eapSelectByThreshold()"
                    title="Check every row whose relevance score is at or above the threshold.">Apply</button>
          </label>
          <label class="u-kv-row">
            Sort:
            <select class="u-small u-p-2-4" id="eap-sort" onchange="eapRender()"
                    title="Order the candidate rows by relevance, year, or title.">
              <option value="score">by relevance score</option>
              <option value="year">by year (newest)</option>
              <option value="title">by title (A-Z)</option>
            </select>
          </label>
          <!-- Phase 54.6.52 — hide previously-cached failures so the
               user doesn't waste selections on DOIs the pipeline will
               silently skip. Count filled in by eapRender. -->
          <label class="u-kv-row"
                 title="Rows previously marked as no-OA or ingest-failed are hidden by default. Uncheck to see them.">
            <input type="checkbox" id="eap-hide-cached" checked onchange="eapRender()"
                   title="Hide rows previously marked as no-OA or ingest-failed. Uncheck to see them.">
            Hide cached <span class="u-muted" id="eap-cached-count"></span>
          </label>
          <span class="u-ml-auto u-small u-muted" id="eap-selected-count"></span>
        </div>
        <div id="eap-table-wrap"
             style="max-height:360px;overflow:auto;border:1px solid var(--border);border-radius:4px;">
          <table class="u-table-full-sm" id="eap-table">
            <thead class="u-bg-alt u-sticky-top">
              <tr>
                <th class="u-w-32-th">
                  <input type="checkbox" id="eap-header-cb"
                         onchange="eapSelectAll(this.checked)"
                         title="Select or deselect every row in the current filtered view.">
                </th>
                <th class="u-th-wrap">Title</th>
                <th class="u-th">Authors</th>
                <th class="u-th">Year</th>
                <th class="u-th">Score</th>
                <th class="u-th" title="Phase 54.6.115 — ± marks feed future expand rounds via anchor-vector bias">Feedback</th>
              </tr>
            </thead>
            <tbody id="eap-tbody"></tbody>
          </table>
        </div>
        <div class="u-flex-raw u-gap-2 u-ai-center u-mt-3 u-wrap">
          <label class="u-kv-row">
            Workers:
            <input class="u-w-60 u-small u-p-2-4" type="number" id="eap-workers" value="0" min="0"
                   title="0 = INGEST_WORKERS from .env">
          </label>
          <label class="u-kv-row"
                 title="Run the full ingest pipeline (convert → metadata → chunk → embed) on the selected PDFs right after they download. Uncheck for download-only.">
            <input type="checkbox" id="eap-ingest" checked
                   title="Run the full ingest pipeline on downloaded PDFs. Uncheck for download-only."> ingest after download
          </label>
          <!-- Phase 54.6.52 — retry bypasses .no_oa_cache + .ingest_failed
               for this batch. Off by default (honor the cache) so the
               normal path doesn't re-probe dead-ends, but one-click
               recovery after a new source lands or the user cleans up
               a broken PDF. -->
          <label class="u-kv-row"
                 title="Ignore the .no_oa_cache + .ingest_failed sidecar files for this batch. Use after adding a new OA source (HAL/Zenodo in 54.6.51) or fixing a broken PDF converter.">
            <input type="checkbox" id="eap-retry-failed"
                   title="Ignore the .no_oa_cache + .ingest_failed sidecars for this batch. Use after adding a new OA source or fixing a broken PDF converter."> retry previously-failed
          </label>
          <button class="btn-primary u-ml-auto" id="eap-download-btn"
                  onclick="eapDownloadSelected()"
                  title="Download the checked rows (OA cascade: Unpaywall / S2 / EuropePMC / arXiv / HAL / Zenodo). Rows with no OA fallback go to the Pending Downloads queue.">
            &#128229; Download selected
          </button>
        </div>
        <div class="u-caption" id="eap-status"></div>
        <pre id="eap-log" style="margin-top:6px;max-height:260px;overflow:auto;background:var(--bg-alt,#f8f8f8);border:1px solid var(--border);border-radius:4px;padding:8px;font-size:11px;font-family:ui-monospace,monospace;white-space:pre-wrap;display:none;"></pre>
      </div>
    </div>
  </div>
</div>

<!-- Phase 54.6.7 — Pending downloads modal.
     Papers the user selected via any expand flow that couldn't be
     auto-downloaded (no OA PDF). Retry / mark-done / abandon / export. -->
<div class="modal-overlay" id="pending-downloads-modal"
     onclick="if(event.target===this)closeModal('pending-downloads-modal')">
  <div class="modal wide xwide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-clipboard"/></svg> Pending downloads</h3>
      <button class="modal-close" onclick="closeModal('pending-downloads-modal')" title="Close the Pending Downloads modal.">&times;</button>
    </div>
    <div class="modal-body">
      <div class="u-note-md">
        Papers you selected that had no legal open-access PDF. Retry runs
        the 6-source OA cascade again (sometimes Unpaywall / Semantic
        Scholar / Europe PMC surface a link that wasn't there before).
        Mark-done records that you got the paper manually (ILL, author
        email, library). Export lets you hand off a CSV for manual work.
      </div>
      <div class="u-row-wrap-mb">
        <label class="u-small">
          Status:
          <select class="u-small u-p-2-4 u-ml-1" id="pdl-status" onchange="refreshPendingDownloads()"
                  title="Filter rows by pipeline status. 'pending' = still awaiting acquisition. 'manual_acquired' = you ticked it off after getting the PDF yourself. 'abandoned' = gave up.">
            <option value="pending" selected>pending</option>
            <option value="manual_acquired">manual_acquired</option>
            <option value="abandoned">abandoned</option>
            <option value="all">all</option>
          </select>
        </label>
        <label class="u-small">
          Source:
          <select class="u-small u-p-2-4 u-ml-1" id="pdl-source" onchange="refreshPendingDownloads()"
                  title="Filter rows by which expand command queued them.">
            <option value="">(any)</option>
            <option value="expand">expand</option>
            <option value="expand-author">expand-author</option>
            <option value="expand-cites">expand-cites</option>
            <option value="expand-topic">expand-topic</option>
            <option value="expand-coauthors">expand-coauthors</option>
            <option value="auto-expand">auto-expand</option>
            <option value="download-dois">download-dois</option>
          </select>
        </label>
        <button class="btn-secondary" onclick="refreshPendingDownloads()"
                title="Re-fetch the pending-downloads list from the server.">Refresh</button>
        <span class="u-hint" id="pdl-count"></span>
        <span class="u-ml-auto"></span>
        <button class="btn-secondary" onclick="pendingSelectAll(true)"
                title="Check every visible row.">Select all</button>
        <button class="btn-secondary" onclick="pendingSelectAll(false)"
                title="Uncheck every row.">Select none</button>
        <button class="btn-primary" onclick="pendingRetrySelected()"
                title="Retry the selected DOIs against the OA cascade, bypassing .no_oa_cache.">
          &#8635; Retry selected
        </button>
        <button class="btn-secondary" onclick="pendingExportCsv()"
                title="Download a CSV of rows currently shown (status / source filtered).">
          &#128229; Export CSV
        </button>
      </div>
      <div style="max-height:60vh;overflow:auto;border:1px solid var(--border);border-radius:4px;">
        <table class="u-table-full-sm" id="pdl-table">
          <thead class="u-bg-alt u-sticky-top">
            <tr>
              <th class="u-w-32-th">
                <input type="checkbox" id="pdl-header-cb" onchange="pendingSelectAll(this.checked)"
                       title="Select or deselect every visible row.">
              </th>
              <th class="u-th-wrap">Title / DOI</th>
              <th class="u-th">Authors</th>
              <th class="u-th">Year</th>
              <th class="u-th">Source</th>
              <th class="u-th">Tries</th>
              <th class="u-th">Last reason</th>
              <th class="u-th">Actions</th>
            </tr>
          </thead>
          <tbody id="pdl-tbody">
            <tr><td colspan="8" class="u-td"><div class="skeleton skeleton-line"></div></td></tr>
            <tr><td colspan="8" class="u-td"><div class="skeleton skeleton-line"></div></td></tr>
            <tr><td colspan="8" class="u-td"><div class="skeleton skeleton-line"></div></td></tr>
          </tbody>
        </table>
      </div>
      <div class="u-caption" id="pdl-retry-status"></div>
      <pre id="pdl-retry-log" style="display:none;margin-top:6px;max-height:240px;overflow:auto;background:var(--bg-alt,#f8f8f8);border:1px solid var(--border);border-radius:4px;padding:8px;font-size:11px;font-family:ui-monospace,monospace;white-space:pre-wrap;"></pre>
    </div>
  </div>
</div>

<!-- Phase 30/31 — Knowledge Graph Modal with Graph + Table tabs -->
<div class="modal-overlay" id="kg-modal" onclick="if(event.target===this)closeModal('kg-modal')">
  <div class="modal wide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-link"/></svg> Knowledge Graph</h3>
      <button class="modal-close" onclick="closeModal('kg-modal')" title="Close the Knowledge Graph modal.">&times;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="kg-graph" onclick="switchKgTab('kg-graph')"
              title="3D force-directed graph of (subject, predicate, object) triples.">Graph</button>
      <button class="tab" data-tab="kg-table" onclick="switchKgTab('kg-table')"
              title="Searchable table of triples. Faster when you want to scan rows instead of panning a graph.">Table</button>
    </div>
    <div class="modal-body">
      <p class="u-note">
        Entity-relationship triples extracted from the corpus during wiki compile.
        Filter by subject substring, predicate (exact match), object substring,
        or document id. <strong>Graph</strong> shows up to 100 triples as nodes
        and edges; <strong>Table</strong> is searchable and shows up to 200.
      </p>
      <div class="field u-row-end">
        <div class="u-flex-2">
          <label>Subject contains</label>
          <input type="text" id="kg-subject" placeholder="(any)" onkeydown="if(event.key==='Enter')loadKg(0)"
                 title="Substring match against the triple's subject. Press Enter to filter.">
        </div>
        <div class="u-flex-1">
          <label>Predicate</label>
          <select id="kg-predicate" onchange="loadKg(0)"
                  title="Exact predicate match. The dropdown is populated from triples currently in the database.">
            <option value="">(any)</option>
          </select>
        </div>
        <div class="u-flex-2">
          <label>Object contains</label>
          <input type="text" id="kg-object" placeholder="(any)" onkeydown="if(event.key==='Enter')loadKg(0)"
                 title="Substring match against the triple's object. Press Enter to filter.">
        </div>
        <button class="btn-primary" onclick="loadKg(0)"
                title="Apply the three filters and reload the graph + table.">Filter</button>
      </div>
      <div class="u-tiny u-muted u-my-2" id="kg-status"></div>
      <!-- Graph tab pane (default) -->
      <div class="u-block" id="kg-graph-pane">
        <div class="kg-controls">
          <span class="kg-controls-label">Theme</span>
          <button class="kg-theme-chip" data-theme="deep-space"
                  title="Deep Space — dark navy background, cool cyan edges."></button>
          <button class="kg-theme-chip" data-theme="paper"
                  title="Paper — light cream background, sepia edges. Print-friendly."></button>
          <button class="kg-theme-chip" data-theme="blueprint"
                  title="Blueprint — dark blue background, white edges. Engineering drawing vibe."></button>
          <button class="kg-theme-chip" data-theme="solarized"
                  title="Solarized (dark) — Ethan Schoonover palette."></button>
          <button class="kg-theme-chip" data-theme="solarized-light"
                  title="Solarized (light) — same palette on a cream background."></button>
          <button class="kg-theme-chip" data-theme="terminal"
                  title="Terminal — black background, green phosphor edges."></button>
          <button class="kg-theme-chip" data-theme="neon"
                  title="Neon — black background, magenta/cyan gradient nodes."></button>
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
            <select class="kg-select" onchange="kgSetColorBy(this.value)"
                    title="Color-by: Cluster colors nodes by connected component; Predicate family colors edges by verb class; Plain uses only the active theme's single accent.">
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
          <!-- Phase 54.6.10 — label typography picker. The "solid"
               variants drop the paint-order halo (stroke=none) so the
               label is one colour instead of two — noticeably more
               readable on bright themes where the fill + halo combo
               hurts contrast. -->
          <label class="kg-chip-group" title="Label typography + solid/halo treatment">
            <span class="kg-controls-label">Font</span>
            <select class="kg-select" id="kg-font-select" onchange="kgSetFont(this.value)"
                    title="Font family + halo treatment. 'Solid' variants drop the paint-order halo (stroke=none) — more readable on bright themes.">
              <option value="sans-halo">Sans &middot; halo</option>
              <option value="sans-solid">Sans &middot; solid</option>
              <option value="serif-solid">Serif &middot; solid</option>
              <option value="serif-halo">Serif &middot; halo</option>
              <option value="mono-solid">Mono &middot; solid</option>
              <option value="mono-halo">Mono &middot; halo</option>
              <option value="condensed-solid">Condensed &middot; solid</option>
              <option value="display-solid">Display &middot; solid</option>
            </select>
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
        <div class="u-border u-r-md" id="kg-graph-canvas"></div>
        <p class="u-xxs u-muted u-mt-6">
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

<!-- Phase 54.6.11 — Visualize modal. Six tabs share one SVG canvas
     element per tab; each render function is independent and pulls
     from its own /api/viz/* endpoint. -->
<div class="modal-overlay" id="viz-modal"
     onclick="if(event.target===this)closeModal('viz-modal')">
  <div class="modal wide xwide">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-layers"/></svg> Visualize</h3>
      <button class="modal-close" onclick="closeModal('viz-modal')" title="Close the Visualize modal.">&times;</button>
    </div>
    <!-- Phase 54.6.15 — shared theming bar above the tabs. Theme chips,
         font, label-size, fullscreen and download PNG apply to ALL six
         tabs. Swap is instant (no data refetch) because every loadX
         caches its base option and _vizReapplyTheme() re-decorates. -->
    <div class="viz-controls" id="viz-controls">
      <span class="viz-controls-label">Theme</span>
      <button class="viz-theme-chip" data-theme="paper" title="Paper"></button>
      <button class="viz-theme-chip" data-theme="deep-space" title="Deep Space"></button>
      <button class="viz-theme-chip" data-theme="blueprint" title="Blueprint"></button>
      <button class="viz-theme-chip" data-theme="solarized" title="Solarized"></button>
      <button class="viz-theme-chip" data-theme="solarized-light" title="Solarized Light"></button>
      <button class="viz-theme-chip" data-theme="terminal" title="Terminal"></button>
      <button class="viz-theme-chip" data-theme="neon" title="Neon"></button>
      <button class="viz-ctrl-btn" onclick="vizInvertTheme()" title="Swap to the paired light/dark preset">&#8646;</button>
      <span class="viz-sep"></span>
      <span class="viz-controls-label">Font</span>
      <select id="viz-font-select" class="viz-select" onchange="vizSetFont(this.value)"
              title="Label typography across every tab">
        <option value="sans-solid">Sans</option>
        <option value="serif-solid">Serif</option>
        <option value="mono-solid">Mono</option>
        <option value="condensed-solid">Condensed</option>
        <option value="display-solid">Display</option>
      </select>
      <span class="viz-sep"></span>
      <span class="viz-controls-label">Labels</span>
      <input type="range" min="0.6" max="2.0" step="0.05" value="1.0"
             id="viz-labelscale" class="viz-slider"
             oninput="vizSetLabelScale(this.value)"
             title="Global label size multiplier"/>
      <span class="viz-sep"></span>
      <span class="viz-controls-label">Custom</span>
      <label class="viz-color-box" title="Background color">
        <span class="viz-color-tag">BG</span>
        <input type="color" id="viz-color-bg"
               oninput="vizSetCustomColor('bg', this.value)"
               value="#f3f5f9"/>
      </label>
      <label class="viz-color-box" title="Label / axis / legend text color">
        <span class="viz-color-tag">Aa</span>
        <input type="color" id="viz-color-label"
               oninput="vizSetCustomColor('label', this.value)"
               value="#0a1a33"/>
      </label>
      <button class="viz-ctrl-btn" onclick="vizClearCustomColors()"
              title="Clear custom colors — revert to the active preset">&#8634;</button>
      <span class="viz-sep"></span>
      <button class="viz-ctrl-btn" onclick="vizToggleFullscreen()" title="Fill the screen with this modal (Esc to exit)">&#9974;</button>
      <button class="viz-ctrl-btn" onclick="vizDownloadPng()" title="Download the current tab as PNG">&#128190;</button>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="viz-topic" onclick="switchVizTab('viz-topic')"
              title="UMAP 2D projection of abstract embeddings, coloured by BERTopic cluster.">&#127760; Topic map</button>
      <button class="tab" data-tab="viz-sunburst" onclick="switchVizTab('viz-sunburst')"
              title="Sunburst of the RAPTOR cluster hierarchy. Inner rings = higher-level summaries.">&#127773; RAPTOR</button>
      <button class="tab" data-tab="viz-consensus" onclick="switchVizTab('viz-consensus')"
              title="Consensus landscape — strength of support for each claim family in the corpus.">&#9878;&#65039; Consensus</button>
      <button class="tab" data-tab="viz-timeline" onclick="switchVizTab('viz-timeline')"
              title="Stacked timeline of papers per year per topic cluster.">&#128200; Timeline</button>
      <button class="tab" data-tab="viz-ego" onclick="switchVizTab('viz-ego')"
              title="Radial ego graph centered on one paper, showing citation neighbours and their topic clusters.">&#128269; Ego radial</button>
      <button class="tab" data-tab="viz-radar" onclick="switchVizTab('viz-radar')"
              title="Coverage gap radar — which topic axes have thin corpus coverage vs strong ones.">&#128504;&#65039; Gap radar</button>
    </div>
    <div class="modal-body" style="padding:var(--sp-3);">
      <div class="u-small u-muted u-mb-6" id="viz-status"></div>
      <!-- Topic map -->
      <div id="viz-topic-pane" class="viz-pane">
        <div class="u-note-xs">
          UMAP 2D projection of every paper's abstract embedding,
          coloured by BERTopic cluster. Click a point to see title /
          authors / year; click "Refresh" after re-clustering.
        </div>
        <div class="u-row-mb-6">
          <button class="btn-secondary" onclick="loadTopicMap(false)"
                  title="Load the topic map from cache (fast).">Load</button>
          <button class="btn-secondary" onclick="loadTopicMap(true)"
                  title="Re-run UMAP from scratch. Slower — only needed after re-clustering or ingesting new papers.">Refresh UMAP</button>
          <span class="u-tiny" id="viz-topic-legend"></span>
        </div>
        <div id="viz-topic-chart" class="viz-chart"></div>
      </div>
      <!-- RAPTOR sunburst -->
      <div id="viz-sunburst-pane" class="viz-pane u-hidden">
        <div class="u-note-xs">
          Sunburst of the RAPTOR cluster hierarchy. Inner ring = highest-
          level summaries, outer rings = children. Click a slice to zoom in,
          click the centre to zoom back out.
        </div>
        <button class="btn-secondary u-mb-6" onclick="loadSunburst()"
                title="Build the RAPTOR sunburst from the server. Requires `catalog raptor build` to have been run.">Load</button>
        <div id="viz-sunburst-chart" class="viz-chart"></div>
      </div>
      <!-- Consensus landscape -->
      <div id="viz-consensus-pane" class="viz-pane u-hidden">
        <div class="u-note-xs">
          Every claim in the corpus for this topic, plotted on
          (# supporting) × (# contradicting) axes. Colour =
          consensus_level (strong / moderate / weak / contested).
          Runs <code>wiki consensus</code> synchronously — 30s-2min.
        </div>
        <div class="u-row-mb-6">
          <input class="u-flex-1 u-pill-md u-small" type="text" id="viz-consensus-topic"
                 placeholder="Topic (e.g. cosmic ray cloud nucleation)"
                 onkeydown="if(event.key==='Enter')loadConsensusLandscape()"
                 title="Free-text topic label. Press Enter to run `wiki consensus`.">
          <button class="btn-primary" onclick="loadConsensusLandscape()"
                  title="Compute the consensus landscape for this topic (LLM-backed, 30s–2min).">Map Consensus</button>
        </div>
        <div id="viz-consensus-chart" class="viz-chart"></div>
      </div>
      <!-- Timeline river -->
      <div id="viz-timeline-pane" class="viz-pane u-hidden">
        <div class="u-note-xs">
          Stacked-area of papers per year, coloured by BERTopic cluster.
          The "history of the field" view. Drag the mini-map below the
          axis to zoom into a specific era.
        </div>
        <button class="btn-secondary u-mb-6" onclick="loadTimeline()"
                title="Build the stacked timeline chart (cluster × year).">Load</button>
        <div id="viz-timeline-chart" class="viz-chart"></div>
      </div>
      <!-- Ego radial -->
      <div id="viz-ego-pane" class="viz-pane u-hidden">
        <div class="u-note-xs">
          Pick a paper, see its top-K nearest papers arranged on a
          polar plot. Radius = cosine distance (closer → centre);
          angle spreads neighbours evenly. Drag to rotate.
        </div>
        <div class="u-row-mb-6">
          <input class="u-flex-1 u-pill-md u-small" type="text" id="viz-ego-docid"
                 placeholder="Document UUID or the first ~8 chars"
                 title="UUID (or uniquely matching prefix) of the center paper. Find these in Browse Papers or the Catalog.">
          <input class="u-w-70 u-pill-md u-small" type="number" id="viz-ego-k" value="20" min="4" max="60"
                 title="How many nearest-neighbour papers to plot.">
          <button class="btn-primary" onclick="loadEgoRadial()"
                  title="Render the polar-plot ego graph for this paper.">Show</button>
        </div>
        <div id="viz-ego-chart" class="viz-chart"></div>
      </div>
      <!-- Gap radar -->
      <div id="viz-radar-pane" class="viz-pane u-hidden">
        <div class="u-note-xs">
          Per-chapter section coverage for this book. Each polygon is
          one chapter over six canonical axes (intro / methods /
          results / discussion / conclusion / related_work); values
          are penalised by any open gap naming the section.
        </div>
        <button class="btn-secondary u-mb-6" onclick="loadGapRadar()"
                title="Compute the per-chapter section-coverage radar. Requires `book gaps` to have been run.">Load</button>
        <div id="viz-radar-chart" class="viz-chart"></div>
      </div>
    </div>
  </div>
</div>

<!-- Phase 30 — Export Modal -->
<div class="modal-overlay" id="export-modal" onclick="if(event.target===this)closeModal('export-modal')">
  <div class="modal">
    <div class="modal-header">
      <h3><svg class="icon icon--lg"><use href="#i-download"/></svg> Export</h3>
      <button class="modal-close" onclick="closeModal('export-modal')" title="Close the Export modal.">&times;</button>
    </div>
    <div class="modal-body">
      <p class="u-small u-muted u-mb-14">
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

<button class="theme-toggle" onclick="toggleTheme()" id="theme-btn"
        title="Toggle between light and dark themes. Preference is saved to localStorage.">
  <svg class="icon" id="theme-icon-svg"><use id="theme-icon-use" href="#i-sun"/></svg>
  <span class="label" id="theme-label">Light</span>
</button>

<script>
// Bootstrap state — read once by the external sciknow.js module.
// Phase E (v2 roadmap) — JS extracted to /static/js/sciknow.js so we
// stop shipping 16 kLOC of inline JavaScript on every page render.
window.SCIKNOW_BOOTSTRAP = {{
  buildTag: '{_BUILD_TAG}',
  activeId: '{active_id}',
  activeChapterId: '{active_chapter_id}',
  activeSectionType: '{active_section_type}',
  bookId: '{book_id}',
  chaptersData: {chapters_json}
}};
</script>
<script src="/static/js/sciknow.js?v={_BUILD_TAG}"></script>
{auto_open_script}
</body>
</html>
"""
