"""``sciknow.web.routes.pending`` — pending-downloads endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. The 4
handlers (list / update / remove / retry) wrap
`sciknow.core.pending_ops`. The retry handler also writes a
temp `dois.json` and spawns the `db download-dois` CLI via
`_app._spawn_cli_streaming` (resolved lazily via the `_app` shim).
"""
from __future__ import annotations

import asyncio
import json as _json
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/api/pending-downloads")
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


@router.post("/api/pending-downloads/update")
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


@router.post("/api/pending-downloads/remove")
async def api_pending_downloads_remove(request: Request):
    """Delete a pending row outright. Body: ``{"doi": "..."}``."""
    from sciknow.core.pending_ops import remove as _remove
    body = await request.json()
    doi = (body.get("doi") or "").strip()
    if not doi:
        raise HTTPException(status_code=400, detail="doi required")
    return JSONResponse({"ok": _remove(doi)})


@router.post("/api/pending-downloads/retry")
async def api_pending_downloads_retry(request: Request):
    """Retry a set of pending DOIs by spawning `db download-dois
    --retry-failed --dois-file <tmp.json>` and streaming SSE.

    Body: ``{"dois": [list], "workers": int, "ingest": bool}``. If ``dois``
    is empty, retries ALL currently-pending rows.
    """
    from sciknow.core.pending_ops import list_pending
    from sciknow.web import app as _app

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

    job_id, _queue = _app._create_job("pending_retry")
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

    _app._spawn_cli_streaming(job_id, argv, loop, on_finish=_cleanup_tmp)
    return JSONResponse({"job_id": job_id, "n_retried": len(requested)})
