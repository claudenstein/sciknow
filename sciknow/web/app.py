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

# v2 Phase E (route split) — resource-scoped routers under web/routes/.
# Routers are imported here (not at module top) to avoid circular
# imports between handlers and helpers defined later in this module.
from sciknow.web.routes import projects as _projects_routes  # noqa: E402
app.include_router(_projects_routes.router)


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

TEMPLATE = (Path(__file__).resolve().parent / "templates" / "book_reader.html").read_text(encoding="utf-8")
