"""``sciknow.web.routes.visuals`` — visuals endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. Behaviour
unchanged. Cross-module deps resolved via the standard lazy
`from sciknow.web import app as _app` shim.
"""
from __future__ import annotations

import asyncio
import json
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response
from sqlalchemy import text

from sciknow.storage.db import get_session

router = APIRouter()


@router.get("/api/visuals")
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


@router.get("/api/visuals/stats")
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


@router.get("/api/visuals/search")
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


@router.get("/api/visuals/suggestions")
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
    from sciknow.web import app as _app
    cached = _app._visuals_get_cached(draft_id)
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

    payload = _app._visuals_rank_for_draft(draft_id, limit=limit)
    payload["cached"] = False
    return JSONResponse(payload)


@router.post("/api/visuals/suggestions")
async def api_visuals_suggestions_rank(
    draft_id: str,
    limit: int = 10,
):
    """Compute the visual ranking for a draft and persist it into
    ``drafts.custom_metadata['visual_suggestions']`` so subsequent opens
    of the Visuals panel serve the saved result instead of re-ranking."""
    from sciknow.web import app as _app
    payload = _app._visuals_rank_for_draft(draft_id, limit=limit)

    if draft_id != _app.BIBLIOGRAPHY_PSEUDO_ID and payload.get("hits") is not None:
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


@router.delete("/api/visuals/suggestions")
async def api_visuals_suggestions_clear(draft_id: str):
    """Drop the persisted visual-suggestions cache for a draft so the
    next Rank-button press recomputes from scratch. Useful for a
    "Clear saved ranking" control."""
    from sciknow.web import app as _app
    if draft_id == _app.BIBLIOGRAPHY_PSEUDO_ID:
        return JSONResponse({"ok": True, "cleared": False})
    with get_session() as session:
        session.execute(text("""
            UPDATE drafts
               SET custom_metadata = custom_metadata - 'visual_suggestions'
             WHERE id::text = :did
        """), {"did": draft_id})
        session.commit()
    return JSONResponse({"ok": True, "cleared": True})


@router.get("/api/visuals/image/{visual_id}")
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
