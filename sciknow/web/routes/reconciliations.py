"""``sciknow.web.routes.reconciliations`` — preprint↔journal + provenance lookup endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. The 3
handlers (list reconciliations / undo / provenance lookup) all read
from `sciknow.core.preprint_reconcile` + `sciknow.core.provenance`.
No app.py cross-deps.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

router = APIRouter()


@router.get("/api/reconciliations")
async def api_reconciliations():
    """Phase 54.6.125 (Tier 3 #3) — list current preprint↔journal
    reconciliations for the active project."""
    from sciknow.core.preprint_reconcile import list_reconciliations
    return JSONResponse({"pairs": list_reconciliations()})


@router.post("/api/reconciliations/undo")
async def api_reconciliations_undo(request: Request):
    """Body: {doc_id: 'uuid prefix'}. Clears canonical_document_id."""
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    doc_id = (body.get("doc_id") or "").strip()
    if not doc_id:
        raise HTTPException(400, "doc_id required")
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


@router.get("/api/provenance")
async def api_provenance(key: str):
    """Phase 54.6.117 (Tier 4 #1) — provenance record by DOI / arxiv / doc-id prefix."""
    from sciknow.core.provenance import lookup as _lookup
    doc_id, rec = _lookup(key)
    return JSONResponse({
        "document_id": doc_id,
        "provenance": rec,
    })
