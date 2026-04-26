"""``sciknow.web.routes.backups`` — backup lifecycle endpoints.

v2 Phase E (route split) — extracted from `web/app.py`.

7 handlers (list / run / download / restore / schedule / delete /
purge) all spawn a streaming CLI subprocess via the
`_spawn_cli_streaming` helper that lives in `web.app` (it threads
its stdout into the per-job event queue). `_create_job` (also in
app.py) issues the job id. Both resolved lazily via the standard
`from sciknow.web import app as _app` shim.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse

router = APIRouter()


@router.get("/api/backups")
async def api_backups_list():
    """Return backup history + schedule status from the JSON state file."""
    from sciknow.cli.backup import _read_state, _backup_root
    state = _read_state()
    return JSONResponse({
        "backups": state.get("backups", []),
        "schedule": state.get("schedule"),
        "backup_dir": str(_backup_root()),
    })


@router.post("/api/backups/run")
async def api_backups_run():
    """Trigger a full backup via _spawn_cli_streaming."""
    from sciknow.web import app as _app
    job_id, _ = _app._create_job("backup_run")
    loop = asyncio.get_event_loop()
    _app._spawn_cli_streaming(job_id, ["backup", "run", "--all-projects"], loop)
    return JSONResponse({"job_id": job_id})


@router.get("/api/backups/download/{dirname}/{filename}")
async def api_backups_download(dirname: str, filename: str):
    """Serve a backup file. Validates the path stays within archives/backups/."""
    from sciknow.cli.backup import _backup_root

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


@router.post("/api/backups/restore")
async def api_backups_restore(request: Request):
    """Trigger a restore from a backup set via _spawn_cli_streaming.
    Body: {timestamp: "latest"|"20260415T030000Z", force: true|false}."""
    from sciknow.web import app as _app
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    ts = body.get("timestamp", "latest")
    force = body.get("force", False)
    job_id, _ = _app._create_job("backup_restore")
    loop = asyncio.get_event_loop()
    argv = ["backup", "restore", ts]
    if force:
        argv.append("--force")
    _app._spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


@router.post("/api/backups/schedule")
async def api_backups_schedule(request: Request):
    """Install or remove the cron. 54.6.94 adds frequency + minute + weekday.

    Body on enable: {action:"enable", frequency:"hourly"|"daily"|"weekly",
                     hour:0-23, minute:0-59, weekday:0-6}
    Body on disable: {action:"disable"}.
    """
    from sciknow.web import app as _app
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    action = body.get("action", "enable")
    if action == "enable":
        job_id, _ = _app._create_job("backup_schedule")
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
        _app._spawn_cli_streaming(job_id, argv, loop)
        return JSONResponse({"job_id": job_id})
    else:
        job_id, _ = _app._create_job("backup_unschedule")
        loop = asyncio.get_event_loop()
        _app._spawn_cli_streaming(job_id, ["backup", "unschedule"], loop)
        return JSONResponse({"job_id": job_id})


@router.post("/api/backups/delete")
async def api_backups_delete(request: Request):
    """54.6.94 — delete one backup set. Body: {timestamp: "<ts>"|"latest"}."""
    from sciknow.web import app as _app
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    ts = (body.get("timestamp") or "").strip()
    if not ts:
        raise HTTPException(400, "timestamp required")
    job_id, _ = _app._create_job("backup_delete")
    loop = asyncio.get_event_loop()
    _app._spawn_cli_streaming(job_id, ["backup", "delete", ts, "--yes"], loop)
    return JSONResponse({"job_id": job_id})


@router.post("/api/backups/purge")
async def api_backups_purge(request: Request):
    """54.6.94 — bulk delete. Body: {all:true} OR {older_than_days:N}."""
    from sciknow.web import app as _app
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
    job_id, _ = _app._create_job("backup_purge")
    loop = asyncio.get_event_loop()
    _app._spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})
