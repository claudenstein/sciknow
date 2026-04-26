"""``sciknow.web.routes.viz`` — visualisation endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. Each handler
runs the corresponding `sciknow.core.viz_ops` builder in a thread-
pool executor (these are heavy: UMAP projections, RAPTOR tree walks,
year × cluster aggregations) and returns the JSON-shaped result.

Cross-module dep: `_app._book_id` (the active-book id stamped on the
running web process). Resolved lazily inside `api_viz_gap_radar`
so the route file imports cleanly while app.py is mid-load.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger("sciknow.web.routes.viz")
router = APIRouter()


@router.get("/api/viz/topic-map")
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


@router.get("/api/viz/raptor-tree")
async def api_viz_raptor_tree():
    """Hierarchical RAPTOR summary tree for the sunburst view."""
    from sciknow.core.viz_ops import raptor_tree
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, raptor_tree)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(result)


@router.post("/api/viz/consensus-landscape")
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


@router.get("/api/viz/timeline")
async def api_viz_timeline():
    """Year × cluster stacked area for the timeline river."""
    from sciknow.core.viz_ops import timeline
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, timeline)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(result)


@router.get("/api/viz/ego-radial")
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


@router.get("/api/viz/gap-radar")
async def api_viz_gap_radar():
    """Per-chapter coverage radar for the active book."""
    from sciknow.core.viz_ops import gap_radar
    from sciknow.web import app as _app
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: gap_radar(_app._book_id),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(result)
