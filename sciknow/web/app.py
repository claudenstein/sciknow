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
from uuid import uuid4

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import text

from sciknow.storage.db import get_session
from sciknow.core.bibliography import (
    BookBibliography,
    BIBLIOGRAPHY_PSEUDO_ID,
    BIBLIOGRAPHY_TITLE,
    render_bibliography_markdown,
)

logger = logging.getLogger("sciknow.web")

from sciknow import __version__ as _SCIKNOW_VERSION
app = FastAPI(
    title="SciKnow Book Reader",
    version=_SCIKNOW_VERSION,
    description=(
        "v2 substrate: writer + embedder + reranker on llama-server "
        "(:8090/:8091/:8092). Bring up with `sciknow infer up --role <r>` "
        "and verify with `sciknow library doctor`."
    ),
)

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
from sciknow.web.routes import feedback as _feedback_routes  # noqa: E402
app.include_router(_feedback_routes.router)
from sciknow.web.routes import bibliography as _bibliography_routes  # noqa: E402
app.include_router(_bibliography_routes.router)
from sciknow.web.routes import viz as _viz_routes  # noqa: E402
app.include_router(_viz_routes.router)
from sciknow.web.routes import jobs as _jobs_routes  # noqa: E402
app.include_router(_jobs_routes.router)
from sciknow.web.routes import backups as _backups_routes  # noqa: E402
app.include_router(_backups_routes.router)
from sciknow.web.routes import ledger as _ledger_routes  # noqa: E402
app.include_router(_ledger_routes.router)
from sciknow.web.routes import pending as _pending_routes  # noqa: E402
app.include_router(_pending_routes.router)
from sciknow.web.routes import reconciliations as _reconciliations_routes  # noqa: E402
app.include_router(_reconciliations_routes.router)
from sciknow.web.routes import system as _system_routes  # noqa: E402
app.include_router(_system_routes.router)
from sciknow.web.routes import tools as _tools_routes  # noqa: E402
app.include_router(_tools_routes.router)
from sciknow.web.routes import autowrite as _autowrite_routes  # noqa: E402
app.include_router(_autowrite_routes.router)
from sciknow.web.routes import export as _export_routes  # noqa: E402
app.include_router(_export_routes.router)
from sciknow.web.routes import catalog as _catalog_routes  # noqa: E402
app.include_router(_catalog_routes.router)
from sciknow.web.routes import snapshots as _snapshots_routes  # noqa: E402
app.include_router(_snapshots_routes.router)
from sciknow.web.routes import wiki as _wiki_routes  # noqa: E402
app.include_router(_wiki_routes.router)
from sciknow.web.routes import chapters as _chapters_routes  # noqa: E402
app.include_router(_chapters_routes.router)
from sciknow.web.routes import book as _book_routes  # noqa: E402
app.include_router(_book_routes.router)
from sciknow.web.routes import draft as _draft_routes  # noqa: E402
app.include_router(_draft_routes.router)
from sciknow.web.routes import visuals as _visuals_routes  # noqa: E402
app.include_router(_visuals_routes.router)
from sciknow.web.routes import corpus as _corpus_routes  # noqa: E402
app.include_router(_corpus_routes.router)
from sciknow.web.routes import draft_actions as _draft_actions_routes  # noqa: E402
app.include_router(_draft_actions_routes.router)
from sciknow.web.routes import pages as _pages_routes  # noqa: E402
app.include_router(_pages_routes.router)
from sciknow.web.routes import comments as _comments_routes  # noqa: E402
app.include_router(_comments_routes.router)
from sciknow.web.routes import misc as _misc_routes  # noqa: E402
app.include_router(_misc_routes.router)


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
    # Phase 54.6.x — guard against an unset/empty `_book_id`. When the
    # web app is started by `sciknow book serve <slug>`, `set_book` is
    # called and binds `_book_id` to a real UUID. In a TestClient
    # (which doesn't go through the CLI) `_book_id` is the empty
    # string, and Postgres rejects it with a UUID DataError. Returning
    # an empty 5-tuple here lets every downstream handler surface the
    # missing-book state cleanly (404 / empty list) instead of 500.
    if not _book_id:
        return (None, [], [], [], [])
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


# ── Chapter reader (continuous scroll) ───────────────────────────────────────

def _titleify_slug_for_display(slug: str) -> str:
    """Local web fallback for unknown slugs (orphaned drafts whose
    section was renamed/deleted). Mirrors core.book_ops._titleify_slug
    but kept here so this module doesn't have to import lazily."""
    return (slug or "").replace("_", " ").strip().title()


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


# ── Phase 50.B — user feedback capture (LambdaMART feedstock) ────────────

# Phase 54.6.135 — route was `/api/feedback` which collided with the
# Phase-54.6.115 expand-candidates ±mark endpoint (line 6072). FastAPI
# dispatched to whichever was registered first (this one), so the active
# ±mark JS calls always hit the form-based thumbs handler here and got
# 422 "score: Field required". The thumbs UI was removed long ago but
# the endpoint stayed; moved to a disambiguated path so the ±mark route
# can own `/api/feedback` again. The rename has no frontend caller
# (there never was one at the time of the fix).
# ── Corpus actions — subprocess-backed, stream stdout as SSE ─────────────────

import os  # noqa: E402 — kept local-ish with the block that uses it
# ── Phase 21.b — Visuals API ────────────────────────────────────────────────

_EQUATIONS_DIAGNOSTIC_TEMPLATE = (
    Path(__file__).resolve().parent / "templates" / "equations_diagnostic.html"
).read_text(encoding="utf-8")


@app.get("/debug/equations")
async def debug_equations(limit: int = 60):
    """Phase 54.6.107 — self-contained equation-render diagnostic.

    Loads N random equations from the corpus and renders each with
    KaTeX directly in the browser. Emits a summary counter (rendered
    ok / KaTeX-error / empty) so we can diagnose why the user sees
    half the equations failing without needing headless automation.
    """
    import json as _j
    try:
        with get_session() as session:
            rows = session.execute(text("""
                SELECT id::text, content
                FROM visuals
                WHERE kind='equation' AND content IS NOT NULL AND content <> ''
                ORDER BY random()
                LIMIT :lim
            """), {"lim": max(1, min(limit, 300))}).fetchall()
    except Exception as exc:
        return HTMLResponse(f"<pre>DB error: {exc}</pre>", status_code=500)

    eqs = [{"id": r[0], "content": r[1]} for r in rows]
    # Plain str.replace (not .format) so the CSS + JS braces in the
    # template don't have to be escaped — only two placeholders.
    body = (
        _EQUATIONS_DIAGNOSTIC_TEMPLATE
            .replace("__N_EQS__", str(len(eqs)))
            .replace("__EQS_JSON__", _j.dumps(eqs))
    )
    return HTMLResponse(body)


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

            # Phase 32.4 — inline ✗ delete button mirrors what
            # rebuildSidebar() in sciknow.js renders. Without this, the
            # section ✗ button only existed AFTER a JS rebuild fired
            # (e.g. after deleting a chapter / adopting an orphan); on
            # initial page load users had no way to remove a section
            # from a chapter's sections_meta. Inline onclick stops the
            # click from bubbling to the wrapping anchor's
            # onclick="return navTo(this)" — same pattern as the
            # sec-orphan-delete button above.
            sec_del_btn = (
                f'<button class="sec-delete-btn" '
                f'onclick="event.preventDefault();event.stopPropagation();'
                f'deleteSection(\'{ch_id}\',\'{sec_type}\')" '
                f'title="Remove this section from the chapter '
                f'(draft becomes an orphan)">\u2717</button>'
            )

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
                    f'<span class="meta">empty \u00b7 \u270e</span>'
                    f'{sec_del_btn}</div>'
                )
            elif status == "orphan":
                # Phase 22 — inline X button on orphan drafts so the
                # user can clean up leftovers from before Phase 18.
                # Phase 25 — also show a "+" button that adopts the
                # slug into the chapter's sections list, re-classifying
                # the draft from "orphan" to "drafted".
                # Inline onclick mirrors sec-delete-btn so the click
                # stops propagating BEFORE the wrapping anchor's
                # onclick="return navTo(this)" fires — otherwise
                # loadSection runs first, currentDraftId flips to the
                # orphan, and the subsequent delete falls through to
                # showDashboard() so the user feels like nothing happened.
                out += (
                    f'<a class="sec-link sec-orphan" href="/section/{sec_id}" '
                    f'data-draft-id="{sec_id}" onclick="return navTo(this)" '
                    f'title="Orphan draft: section_type={sec_type!r} doesn&#39;t match any current template slug. Click to inspect, + to adopt into sections, \u2717 to delete.">'
                    f'<span class="sec-status-dot orphan"></span>'
                    f'{display} '
                    f'<span class="meta">orphan \u00b7 v{sec_v} \u00b7 {sec_w}w</span>'
                    f'<button class="sec-orphan-adopt" '
                    f'onclick="event.preventDefault();event.stopPropagation();'
                    f'adoptOrphanSection(\'{ch_id}\',\'{sec_type}\')" '
                    f'title="Add this section_type to the chapter\u2019s sections list (idempotent)">+</button>'
                    f'<button class="sec-orphan-delete" '
                    f'onclick="event.preventDefault();event.stopPropagation();'
                    f'deleteOrphanDraft(\'{sec_id}\')" '
                    f'title="Delete this orphan draft permanently">\u2717</button>'
                    f'</a>'
                )
            else:
                # Phase 26 — drafted rows are draggable for reordering.
                # The click handler (navTo) still fires on plain clicks;
                # the browser distinguishes click from drag based on
                # whether the cursor moved during mousedown.
                # Phase 32.4 — sec_del_btn appended inside the anchor
                # so users can remove the section from sections_meta
                # without first triggering a JS rebuild via some other
                # action.
                out += (
                    f'<a class="sec-link {active}" href="/section/{sec_id}" '
                    f'draggable="true" '
                    f'data-draft-id="{sec_id}" '
                    f'data-section-slug="{sec_type}" '
                    f'title="{plan_attr}" '
                    f'onclick="return navTo(this)">'
                    f'<span class="sec-status-dot drafted"></span>'
                    f'{display} '
                    f'<span class="meta">v{sec_v} \u00b7 {sec_w}w</span>'
                    f'{sec_del_btn}</a>'
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


# ── v2 Phase E (route split) — re-exports ─────────────────────────────────
#
# When a route handler moves into `web/routes/<resource>.py`, L1 tests
# that grab it via `getattr(web_app, "<handler>")` or
# `inspect.getsource(web_app.<handler>)` would otherwise fail. Re-bind
# the moved names back onto the app module so the contract surface
# stays where the tests expect — `getattr` resolves, `inspect.getsource`
# returns the source from the new file (correct: same code, just
# different __module__).
from sciknow.web.routes.projects import (  # noqa: E402, F401
    api_projects_list, api_projects_show, api_projects_use,
    api_projects_init, api_project_venues_get, api_project_venues_post,
    api_projects_destroy,
)
from sciknow.web.routes.feedback import (  # noqa: E402, F401
    api_feedback_thumbs, api_feedback_stats, api_feedback_get,
    api_feedback_post,
)
from sciknow.web.routes.bibliography import (  # noqa: E402, F401
    api_bibliography_audit, api_bibliography_sort,
    api_bibliography_citation,
)
from sciknow.web.routes.viz import (  # noqa: E402, F401
    api_viz_topic_map, api_viz_raptor_tree,
    api_viz_consensus_landscape, api_viz_timeline,
    api_viz_ego_radial, api_viz_gap_radar,
)
from sciknow.web.routes.jobs import (  # noqa: E402, F401
    stream_job, cancel_job, get_job_stats, list_jobs,
)
from sciknow.web.routes.backups import (  # noqa: E402, F401
    api_backups_list, api_backups_run, api_backups_download,
    api_backups_restore, api_backups_schedule,
    api_backups_delete, api_backups_purge,
)
from sciknow.web.routes.ledger import (  # noqa: E402, F401
    api_ledger_book, api_ledger_chapter, api_ledger_draft,
)
from sciknow.web.routes.pending import (  # noqa: E402, F401
    api_pending_downloads_list, api_pending_downloads_update,
    api_pending_downloads_remove, api_pending_downloads_retry,
)
from sciknow.web.routes.reconciliations import (  # noqa: E402, F401
    api_reconciliations, api_reconciliations_undo, api_provenance,
)
from sciknow.web.routes.system import (  # noqa: E402, F401
    api_settings_models, api_stats, api_monitor,
    api_monitor_alerts_md, api_admin_release_vram,
)
from sciknow.web.routes.tools import (  # noqa: E402, F401
    api_ask, api_search_query, api_search_similar, api_ask_synthesize,
)
from sciknow.web.routes.autowrite import (  # noqa: E402, F401
    api_autowrite_chapter, api_autowrite,
)
from sciknow.web.routes.export import (  # noqa: E402, F401
    export_draft, export_chapter, export_book,
)
from sciknow.web.routes.catalog import (  # noqa: E402, F401
    api_catalog, api_catalog_authors, api_catalog_domains,
    api_catalog_topics, api_catalog_cluster, api_catalog_raptor_build,
)
from sciknow.web.routes.snapshots import (  # noqa: E402, F401
    create_snapshot, list_snapshots, get_snapshot_content,
    create_chapter_snapshot, create_book_snapshot,
    list_chapter_snapshots, list_book_snapshots,
    restore_snapshot_bundle,
    _snapshot_chapter_drafts, _restore_chapter_bundle,
)
from sciknow.web.routes.wiki import (  # noqa: E402, F401
    api_wiki_pages, api_wiki_annotation_get, api_wiki_annotation_put,
    api_wiki_page_ask, api_wiki_backlinks, api_wiki_related,
    api_wiki_titles, api_wiki_page, api_wiki_query,
    api_wiki_extract_kg, api_wiki_lint, api_wiki_consensus,
    api_wiki_compile,
)
from sciknow.web.routes.chapters import (  # noqa: E402, F401
    api_chapters, create_chapter, update_chapter,
    adopt_orphan_section_endpoint, update_chapter_sections,
    delete_chapter, reorder_chapters,
    api_chapter_plan_sections, api_chapter_resolved_targets,
)
from sciknow.web.routes.book import (  # noqa: E402, F401
    api_book, api_book_update,
    api_book_style_fingerprint_refresh, api_book_plan_generate,
    api_book_outline_generate, api_book_auto_expand_preview,
    api_book_create, api_book_length_report, api_book_types,
)
from sciknow.web.routes.draft import (  # noqa: E402, F401
    api_activate_draft, api_draft_save_as_version,
    api_draft_rename_version, api_draft_version_description,
    update_draft_status, update_draft_metadata,
    move_draft_to_chapter, delete_draft, api_draft_scores,
)
from sciknow.web.routes.visuals import (  # noqa: E402, F401
    api_visuals_list, api_visuals_stats, api_visuals_search,
    api_visuals_suggestions, api_visuals_suggestions_rank,
    api_visuals_suggestions_clear, api_visuals_image,
)
from sciknow.web.routes.corpus import (  # noqa: E402, F401
    api_corpus_enrich, api_corpus_ingest_directory,
    api_corpus_upload, api_corpus_expand_author_preview,
    api_corpus_expand_author, api_corpus_agentic_preview,
    api_corpus_authors_top,
    api_corpus_expand_author_refs_preview,
    api_corpus_expand_oeuvre_preview,
    api_corpus_expand_author_download_selected,
    api_corpus_expand_cites_preview,
    api_corpus_expand_topic_preview,
    api_corpus_expand_coauthors_preview,
    api_corpus_cleanup_downloads, api_corpus_expand_preview,
    api_corpus_expand_preview_candidates, api_corpus_expand,
)
from sciknow.web.routes.draft_actions import (  # noqa: E402, F401
    api_write, api_review, api_revise, api_gaps, api_argue,
    api_verify, api_insert_citations, api_adversarial_review,
    api_edge_cases,
)
from sciknow.web.routes.pages import (  # noqa: E402, F401
    section, route_plan, route_settings, route_wiki,
    route_bundles, route_tools, route_projects, route_catalog,
    route_export, route_corpus, route_visualize, route_kg,
    route_ask, route_setup, route_backups, route_visuals,
    route_help,
)
from sciknow.web.routes.comments import (  # noqa: E402, F401
    add_comment, resolve_comment, edit_draft, search_book,
)
from sciknow.web.routes.misc import (  # noqa: E402, F401
    api_section, api_dashboard, api_versions, api_diff, api_kg,
    chapter_reader, corkboard_data, api_methods,
    api_setup_status, api_bench_section_lengths, api_cli_stream,
    api_server_shutdown,
)
