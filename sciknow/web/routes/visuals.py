"""``sciknow.web.routes.visuals`` — visuals endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. Behaviour
unchanged. Cross-module deps resolved via the standard lazy
`from sciknow.web import app as _app` shim.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response
from sqlalchemy import text

from sciknow.core.bibliography import BIBLIOGRAPHY_PSEUDO_ID
from sciknow.storage.db import get_session

logger = logging.getLogger("sciknow.web.routes.visuals")
router = APIRouter()


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


@router.post("/api/visuals/suggestions")
async def api_visuals_suggestions_rank(
    draft_id: str,
    limit: int = 10,
):
    """Compute the visual ranking for a draft and persist it into
    ``drafts.custom_metadata['visual_suggestions']`` so subsequent opens
    of the Visuals panel serve the saved result instead of re-ranking."""
    from sciknow.web import app as _app
    payload = _visuals_rank_for_draft(draft_id, limit=limit)

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


@router.post("/api/visuals/suggestions/batch")
async def api_visuals_suggestions_batch(
    limit: int = 10,
    overwrite: bool = False,
):
    """Phase 55.V8 — bulk visuals ranker for the active book.

    Walks every active draft of every section in the active book and
    runs `_visuals_rank_for_draft`, persisting the result to
    `drafts.custom_metadata['visual_suggestions']`. Streams progress
    events through the standard SSE job queue so the GUI can render
    a live progress log. Skips drafts whose persisted ranking still
    matches the current content (use `overwrite=True` to force).
    """
    from sciknow.web import app as _app
    job_id, queue = _app._create_job("rank_visuals_batch")
    loop = asyncio.get_event_loop()

    def gen():
        return _rank_visuals_book_stream(
            book_id=_app._book_id, limit=limit, overwrite=overwrite,
        )

    thread = threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


def _rank_visuals_book_stream(
    *, book_id: str | None, limit: int = 10, overwrite: bool = False,
):
    """Generator yielding `progress` / `rank_done` / `rank_skipped` /
    `rank_failed` / `completed` events for every section in the book.
    Same persistence path as the per-draft endpoint."""
    import hashlib as _h
    from datetime import datetime as _dt

    if not book_id:
        yield {"type": "error", "message": "no active book"}
        return

    with get_session() as session:
        ch_rows = session.execute(text("""
            SELECT id::text, number, title FROM book_chapters
            WHERE book_id::text = :bid
            ORDER BY number
        """), {"bid": book_id}).fetchall()
        ch_ids = [c[0] for c in ch_rows]
        ch_map = {c[0]: (c[1], c[2]) for c in ch_rows}
        if not ch_ids:
            yield {"type": "error", "message": "no chapters"}
            return
        placeholders = ", ".join(f":c{i}" for i in range(len(ch_ids)))
        params = {f"c{i}": cid for i, cid in enumerate(ch_ids)}
        drafts = session.execute(text(f"""
            SELECT DISTINCT ON (chapter_id, section_type)
                   id::text, chapter_id::text, section_type, content,
                   custom_metadata
            FROM drafts
            WHERE chapter_id::text IN ({placeholders})
            ORDER BY chapter_id, section_type,
                     (custom_metadata->>'is_active')::boolean DESC NULLS LAST,
                     version DESC
        """), params).fetchall()

    n_total = len(drafts)
    yield {"type": "progress",
           "detail": f"Ranking visuals for {n_total} drafts across {len(ch_rows)} chapters…"}

    n_ranked = 0
    n_skipped = 0
    n_failed = 0
    for d in drafts:
        did, cid, sec, content, meta = d
        ch_num, ch_title = ch_map.get(cid, ("?", ""))
        label = f"Ch.{ch_num} {sec}"
        if not (content or "").strip():
            n_skipped += 1
            yield {"type": "rank_skipped", "section": label,
                   "reason": "empty draft"}
            continue
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not overwrite and isinstance(meta, dict):
            blob = meta.get("visual_suggestions")
            if isinstance(blob, dict):
                cur_hash = _h.md5((content or "").encode(
                    "utf-8", errors="ignore")).hexdigest()
                if blob.get("content_hash") == cur_hash:
                    n_skipped += 1
                    yield {"type": "rank_skipped", "section": label,
                           "reason": "cached", "n_hits": len(blob.get("hits") or [])}
                    continue
        try:
            payload = _visuals_rank_for_draft(did, limit=limit)
            hits = payload.get("hits") or []
            cur_hash = _h.md5((content or "").encode(
                "utf-8", errors="ignore")).hexdigest()
            blob = {
                "hits": hits,
                "ranked_at": _dt.utcnow().isoformat() + "Z",
                "content_hash": cur_hash,
                "note": payload.get("note"),
            }
            with get_session() as session:
                session.execute(text("""
                    UPDATE drafts
                       SET custom_metadata = COALESCE(custom_metadata, '{}'::jsonb)
                                             || jsonb_build_object(
                                                  'visual_suggestions',
                                                  CAST(:blob AS jsonb))
                     WHERE id::text = :did
                """), {"did": did, "blob": json.dumps(blob)})
                session.commit()
            n_ranked += 1
            avg = (sum((h.get("composite_score") or 0) for h in hits)
                   / max(len(hits), 1)) if hits else 0
            yield {"type": "rank_done", "section": label,
                   "n_hits": len(hits), "avg_score": avg}
        except Exception as exc:
            n_failed += 1
            yield {"type": "rank_failed", "section": label,
                   "error": f"{type(exc).__name__}: {exc}"}

    yield {"type": "completed",
           "n_ranked": n_ranked, "n_skipped": n_skipped, "n_failed": n_failed}


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
