"""``sciknow.web.routes.jobs`` — job lifecycle + SSE streaming endpoints.

v2 Phase E (route split) — extracted from `web/app.py`.

The 4 handlers (SSE stream / cancel / stats / list) consume the
process-local `_jobs` dict + `_job_lock` lock + the throughput
helpers (`_job_tps`, `_job_decode_stats`, `_job_tps_windows`) all
defined in `web.app`. Resolved lazily inside each handler via the
standard `from sciknow.web import app as _app` shim — by call-time
app.py is fully loaded, so the bindings exist on the module.
"""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter()


@router.get("/api/stream/{job_id}")
async def stream_job(job_id: str):
    """SSE endpoint — streams events from a running job."""
    from sciknow.web import app as _app
    with _app._job_lock:
        job = _app._jobs.get(job_id)
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


@router.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    from sciknow.web import app as _app
    with _app._job_lock:
        job = _app._jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job["cancel"].set()
    return JSONResponse({"ok": True})


@router.get("/api/jobs/{job_id}/stats")
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
    from sciknow.web import app as _app
    with _app._job_lock:
        job = _app._jobs.get(job_id)
        if not job:
            # Treat as already-finished — the GC swept it after the
            # 5-minute window, OR the job was never created.
            raise HTTPException(410, "Job not found (likely already finished)")
        # Phase 54.6.247 — shared helper with _write_web_jobs_pulse
        # so both UIs see the same number.
        tps = _app._job_tps(job)
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
            "decode_stats": _app._job_decode_stats(job),
            "tps_windows": _app._job_tps_windows(job),
            "elapsed_s": round(elapsed_s, 1),
            "model_name": job.get("model_name"),
            "task_desc": job.get("task_desc") or job.get("type"),
            "target_words": job.get("target_words"),
            "error_message": job.get("error_message"),
        })


@router.get("/api/jobs")
async def list_jobs():
    from sciknow.web import app as _app
    with _app._job_lock:
        return [
            {"id": jid, "type": j["type"], "status": j["status"]}
            for jid, j in _app._jobs.items()
        ]
