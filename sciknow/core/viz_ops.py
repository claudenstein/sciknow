"""Phase 54.6.11 — data helpers for the six Visualize tabs.

Every helper returns a JSON-ready dict / list so the web layer can
hand it straight to ``JSONResponse``. Expensive steps (UMAP fit on
the abstracts collection) are one-shot and cached on disk per-project
so the modal opens in <1s after the first run.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from sqlalchemy import text as sql_text

from sciknow.config import settings
from sciknow.storage.db import get_session
from sciknow.storage.qdrant import abstracts_collection, get_client

logger = logging.getLogger(__name__)


# ── 1. Topic map (UMAP 2D of paper abstracts) ───────────────────────────

def _topic_map_cache_path() -> Path:
    return settings.data_dir / "viz" / "topic_map.json"


def topic_map(*, refresh: bool = False, n_neighbors: int = 15,
              min_dist: float = 0.1) -> dict[str, Any]:
    """UMAP projection of every paper's abstract embedding.

    Returns ``{"points": [{doc_id, title, year, cluster, x, y}, ...],
    "clusters": [{id, name, color}], "cached_at": iso}``. First call
    runs UMAP (~5-60s depending on corpus size) and caches to
    ``<data_dir>/viz/topic_map.json``; subsequent calls return the
    cache unless ``refresh=True``.
    """
    cache = _topic_map_cache_path()
    if not refresh and cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass  # fall through to rebuild

    # Pull vectors + doc_ids from the abstracts collection.
    client = get_client()
    coll = abstracts_collection()
    points: list[dict] = []
    offset = None
    try:
        while True:
            batch, offset = client.scroll(
                collection_name=coll, limit=200, offset=offset,
                with_vectors=True, with_payload=True,
            )
            for pt in batch:
                vec = None
                if isinstance(pt.vector, dict):
                    vec = pt.vector.get("dense") or next(iter(pt.vector.values()), None)
                else:
                    vec = pt.vector
                if vec is None:
                    continue
                payload = pt.payload or {}
                points.append({
                    "document_id": payload.get("document_id"),
                    "vector": list(vec),
                    "title": (payload.get("title") or "")[:200],
                    "year": payload.get("year"),
                })
            if offset is None or not batch:
                break
    except Exception as exc:
        logger.exception("topic_map: abstracts scroll failed")
        raise RuntimeError(f"abstracts scroll failed: {exc}") from exc

    if not points:
        return {"points": [], "clusters": [],
                "message": "no abstract vectors — run `sciknow catalog cluster` first"}

    # Enrich with cluster + metadata from Postgres in one round trip.
    doc_ids = [p["document_id"] for p in points if p["document_id"]]
    meta_by_id: dict[str, dict] = {}
    if doc_ids:
        with get_session() as session:
            rows = session.execute(sql_text("""
                SELECT pm.document_id::text, pm.title, pm.year, pm.topic_cluster,
                       pm.authors
                FROM paper_metadata pm
                WHERE pm.document_id::text = ANY(:ids)
            """), {"ids": doc_ids}).fetchall()
        for r in rows:
            meta_by_id[r[0]] = {
                "title": r[1] or "",
                "year": r[2],
                "cluster": r[3],
                "authors": r[4] or [],
            }

    # Fit UMAP. umap-learn is already a dependency (BERTopic uses it).
    import numpy as np
    try:
        import umap  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "umap-learn not installed but declared as dep — "
            f"{type(exc).__name__}: {exc}"
        )
    X = np.array([p["vector"] for p in points], dtype=np.float32)
    try:
        reducer = umap.UMAP(
            n_neighbors=min(n_neighbors, max(2, len(points) - 1)),
            min_dist=min_dist, n_components=2, metric="cosine",
            random_state=42,
        )
        XY = reducer.fit_transform(X)
    except Exception as exc:
        raise RuntimeError(f"UMAP failed: {exc}") from exc

    # Rescale to a [-1, 1] box so the canvas mapping is trivial.
    xs, ys = XY[:, 0], XY[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    def _norm(v, lo, hi):
        if hi - lo < 1e-9:
            return 0.0
        return 2.0 * (v - lo) / (hi - lo) - 1.0

    out_points: list[dict] = []
    cluster_ids: set[str] = set()
    for p, (xx, yy) in zip(points, XY):
        m = meta_by_id.get(p["document_id"], {})
        authors = m.get("authors") or []
        first_author = ""
        if authors:
            a0 = authors[0]
            first_author = a0.get("name", "") if isinstance(a0, dict) else str(a0)
        cid = m.get("cluster")
        if cid is not None:
            cluster_ids.add(str(cid))
        out_points.append({
            "document_id": p["document_id"],
            "title": (m.get("title") or p["title"])[:200],
            "year": m.get("year") or p.get("year"),
            "first_author": first_author,
            "cluster": cid,
            "x": float(_norm(float(xx), x_min, x_max)),
            "y": float(_norm(float(yy), y_min, y_max)),
        })

    # Cluster labels + a stable deterministic color per cluster.
    cluster_records: list[dict] = []
    if cluster_ids:
        with get_session() as session:
            try:
                rows = session.execute(sql_text("""
                    SELECT topic_cluster, COUNT(*)
                    FROM paper_metadata
                    WHERE topic_cluster IS NOT NULL
                    GROUP BY topic_cluster
                    ORDER BY topic_cluster
                """)).fetchall()
                for i, (cid, n) in enumerate(rows):
                    cluster_records.append({
                        "id": str(cid),
                        "name": f"Cluster {cid}",
                        "count": n,
                        "color": _golden_hue(i),
                    })
            except Exception:
                pass

    result = {
        "points": out_points,
        "clusters": cluster_records,
        "n_papers": len(out_points),
    }
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result))
    except Exception as exc:
        logger.warning("topic_map cache write failed: %s", exc)
    return result


def _golden_hue(i: int) -> str:
    """Deterministic HSL color picker using the golden-ratio trick."""
    phi = 0.61803398875
    h = (i * phi) % 1.0
    return f"hsl({int(h*360)}, 62%, 55%)"


# ── 2. RAPTOR tree ──────────────────────────────────────────────────────

def raptor_tree() -> dict[str, Any]:
    """Return the RAPTOR hierarchy as a nested dict suitable for a
    sunburst: ``{name, value, children: [...]}``. If RAPTOR hasn't
    been built, returns an empty tree with a hint.
    """
    client = get_client()
    coll = abstracts_collection().replace("abstracts", "papers")  # main papers coll
    try:
        info = client.get_collection(coll)
    except Exception as exc:
        return {"name": "root", "children": [], "error": str(exc)}

    # Scroll every point with node_level >= 1 (summaries).
    from qdrant_client.http import models as qm
    try:
        filt = qm.Filter(must=[qm.FieldCondition(
            key="node_level", range=qm.Range(gte=1.0)
        )])
    except Exception:
        filt = None

    nodes: list[dict] = []
    offset = None
    try:
        while True:
            batch, offset = client.scroll(
                collection_name=coll, limit=500, offset=offset,
                scroll_filter=filt, with_payload=True, with_vectors=False,
            )
            for pt in batch:
                p = pt.payload or {}
                nodes.append({
                    "id": str(pt.id),
                    "level": int(p.get("node_level") or 0),
                    "summary": (p.get("summary_text") or "")[:200],
                    "child_ids": list(p.get("child_chunk_ids") or []),
                    "n_docs": len(p.get("document_ids") or []),
                    "year_range": (p.get("year_min"), p.get("year_max")),
                })
            if offset is None or not batch:
                break
    except Exception as exc:
        return {"name": "root", "children": [], "error": str(exc)}

    if not nodes:
        return {"name": "root", "children": [],
                "message": "RAPTOR tree not built — run `sciknow catalog raptor build`"}

    # Build a child-of index. Any node whose id appears in another
    # node's child_ids is a child. Roots = nodes not referenced as
    # child anywhere.
    by_id = {n["id"]: n for n in nodes}
    child_ids: set[str] = set()
    for n in nodes:
        for cid in n["child_ids"]:
            child_ids.add(str(cid))
    roots = [n for n in nodes if n["id"] not in child_ids]
    # Fall back to "highest-level nodes" as roots if the child-id
    # graph is sparse (common if leaves live in a different collection).
    if not roots:
        max_lv = max(n["level"] for n in nodes)
        roots = [n for n in nodes if n["level"] == max_lv]

    def _build(n: dict) -> dict:
        kids = [by_id[str(cid)] for cid in n["child_ids"] if str(cid) in by_id]
        return {
            "name": (n["summary"][:80] or f"L{n['level']}:{n['id'][:6]}"),
            "level": n["level"],
            "n_docs": n["n_docs"],
            "year_min": n["year_range"][0],
            "year_max": n["year_range"][1],
            "value": max(n["n_docs"], 1),
            "children": [_build(k) for k in kids] if kids else [],
        }

    tree = {"name": "corpus", "children": [_build(r) for r in roots]}
    tree["total_nodes"] = len(nodes)
    return tree


# ── 3. Consensus landscape (reuses wiki consensus_map) ──────────────────

def consensus_landscape(topic: str, *, model: str | None = None) -> dict[str, Any]:
    """Run wiki consensus_map synchronously and return its claims list
    reshaped for the landscape scatter ({x=#supporting, y=#contradicting,
    consensus_level, trend, label}).
    """
    if not topic.strip():
        raise ValueError("topic required")
    from sciknow.core.wiki_ops import consensus_map
    data = None
    slug = None
    for evt in consensus_map(topic.strip(), model=model):
        if evt.get("type") == "consensus":
            data = evt.get("data") or {}
        elif evt.get("type") == "completed":
            slug = evt.get("slug")
        elif evt.get("type") == "error":
            raise RuntimeError(evt.get("message", "consensus_map error"))
    if not data:
        return {"claims": [], "summary": "", "slug": slug}

    claims = []
    for c in (data.get("claims") or []):
        sup = c.get("supporting_papers") or []
        con = c.get("contradicting_papers") or []
        claims.append({
            "claim": (c.get("claim") or "")[:240],
            "consensus_level": (c.get("consensus_level") or "unknown").lower(),
            "trend": c.get("trend"),
            "x": len(sup),
            "y": len(con),
            "supporting": sup[:8],
            "contradicting": con[:8],
        })
    return {
        "claims": claims,
        "summary": data.get("summary", ""),
        "debated": data.get("most_debated") or [],
        "slug": slug,
        "topic": topic,
    }


# ── 4. Timeline river (year × cluster stacked area) ─────────────────────

def timeline() -> dict[str, Any]:
    """Year × bucket counts for the stacked-area timeline.

    Bucketing rule:

    * If ANY paper has ``topic_cluster`` set, split by cluster. Each
      cluster is given a name pulled from the first paper in the
      cluster's keyword set (or ``Cluster {id}`` as a fallback).
    * Otherwise fall back to a decade split (1960s / 1970s / ...).
      Still meaningful "history of the field" without having to run
      `catalog cluster` first — the user sees a real stacked area
      instead of a single monochrome stream.

    Returns ``{"years": [...], "series": [{cluster, values[], total,
    color}, ...], "mode": "cluster" | "decade", "message": str?}``.
    """
    with get_session() as session:
        n_clustered = session.execute(sql_text(
            "SELECT COUNT(*) FROM paper_metadata WHERE topic_cluster IS NOT NULL"
        )).scalar() or 0

        if n_clustered > 0:
            rows = session.execute(sql_text("""
                SELECT pm.year,
                       COALESCE(pm.topic_cluster::text, 'noise') AS bucket,
                       COUNT(*) AS n
                FROM paper_metadata pm
                JOIN documents d ON d.id = pm.document_id
                WHERE d.ingestion_status = 'complete'
                  AND pm.year IS NOT NULL
                GROUP BY pm.year, bucket
                ORDER BY pm.year
            """)).fetchall()
            mode = "cluster"
            msg = None
        else:
            # Decade fallback — still year-granular on the x-axis but
            # the series colour reflects the decade, giving a proper
            # "history" feel without requiring clustering to be built.
            rows = session.execute(sql_text("""
                SELECT pm.year,
                       CONCAT(10 * (pm.year / 10), 's') AS bucket,
                       COUNT(*) AS n
                FROM paper_metadata pm
                JOIN documents d ON d.id = pm.document_id
                WHERE d.ingestion_status = 'complete'
                  AND pm.year IS NOT NULL
                GROUP BY pm.year, bucket
                ORDER BY pm.year
            """)).fetchall()
            mode = "decade"
            msg = ("No BERTopic clusters yet — run `sciknow catalog cluster` "
                   "for a proper cluster-coloured view. Falling back to "
                   "decade bins for now.")

    years_set: set[int] = set()
    by_bucket: dict[str, dict[int, int]] = {}
    for year, bucket, n in rows:
        years_set.add(int(year))
        key = str(bucket)
        by_bucket.setdefault(key, {})[int(year)] = int(n)

    if not years_set:
        return {"years": [], "series": [], "mode": mode, "message": msg}

    years = sorted(years_set)
    # Build series; pick colours that differ meaningfully. For decade
    # mode we sort buckets chronologically so the colour sweep feels
    # natural (earlier decades → cooler, later → warmer).
    bucket_keys = sorted(by_bucket.keys())
    series = []
    for i, key in enumerate(bucket_keys):
        per_year = by_bucket[key]
        series.append({
            "cluster": key,
            "values": [per_year.get(y, 0) for y in years],
            "total": sum(per_year.values()),
            "color": _golden_hue(i),
        })
    # Sort by total desc so the biggest buckets draw on the bottom.
    series.sort(key=lambda s: s["total"], reverse=True)
    return {"years": years, "series": series, "mode": mode, "message": msg}


# ── 5. Ego radial (top-K similar papers for a given DOI) ────────────────

def ego_radial(doc_id: str, *, k: int = 20) -> dict[str, Any]:
    """Top-K most similar papers to the given document via cosine
    distance on the abstracts collection. Centre = target paper.
    """
    client = get_client()
    coll = abstracts_collection()

    # Find the target point by document_id payload.
    from qdrant_client.http import models as qm
    try:
        filt = qm.Filter(must=[qm.FieldCondition(
            key="document_id", match=qm.MatchValue(value=doc_id)
        )])
        target_batch, _ = client.scroll(
            collection_name=coll, limit=1, scroll_filter=filt,
            with_vectors=True, with_payload=True,
        )
    except Exception as exc:
        raise RuntimeError(f"scroll failed: {exc}") from exc
    if not target_batch:
        raise ValueError(f"no abstract vector for document_id={doc_id}")
    target = target_batch[0]
    tgt_vec = target.vector.get("dense") if isinstance(target.vector, dict) else target.vector

    # Top-K neighbours (excluding self via score threshold).
    # qdrant-client ≥1.9 deprecated .search() in favour of
    # .query_points(); use the same pattern as hybrid_search.py.
    try:
        response = client.query_points(
            collection_name=coll,
            query=list(tgt_vec), using="dense",
            limit=k + 1, with_payload=True,
        )
        hits = response.points
    except Exception as exc:
        raise RuntimeError(f"similarity search failed: {exc}") from exc

    # Enrich with Postgres metadata for each hit.
    doc_ids = [
        h.payload.get("document_id")
        for h in hits
        if h.payload.get("document_id") and h.payload.get("document_id") != doc_id
    ]
    meta_by_id: dict[str, dict] = {}
    all_ids = [doc_id] + doc_ids
    if all_ids:
        with get_session() as session:
            rows = session.execute(sql_text("""
                SELECT pm.document_id::text, pm.title, pm.year, pm.authors
                FROM paper_metadata pm
                WHERE pm.document_id::text = ANY(:ids)
            """), {"ids": all_ids}).fetchall()
        for r in rows:
            authors = r[3] or []
            fa = ""
            if authors:
                a0 = authors[0]
                fa = a0.get("name", "") if isinstance(a0, dict) else str(a0)
            meta_by_id[r[0]] = {"title": r[1] or "",
                                "year": r[2], "first_author": fa}

    centre = meta_by_id.get(doc_id, {})
    neighbours: list[dict] = []
    for h in hits:
        did = h.payload.get("document_id")
        if not did or did == doc_id:
            continue
        m = meta_by_id.get(did, {})
        neighbours.append({
            "document_id": did,
            "title": m.get("title", "")[:160],
            "year": m.get("year"),
            "first_author": m.get("first_author", ""),
            "score": float(h.score),
        })
        if len(neighbours) >= k:
            break

    # Ring layout — polar coords, neighbour i at angle 2πi/n, radius
    # derived from (1 - score) so closer papers sit nearer the centre.
    for i, n in enumerate(neighbours):
        theta = 2.0 * math.pi * i / max(1, len(neighbours))
        r = 0.25 + 0.65 * (1.0 - n["score"])  # empirically good
        n["x"] = math.cos(theta) * r
        n["y"] = math.sin(theta) * r

    return {
        "centre": {
            "document_id": doc_id,
            "title": centre.get("title", "")[:160],
            "year": centre.get("year"),
            "first_author": centre.get("first_author", ""),
        },
        "neighbours": neighbours,
    }


# ── 6. Gap radar (per-chapter section coverage) ─────────────────────────

_RADAR_AXES = [
    "introduction", "methods", "results", "discussion",
    "conclusion", "related_work",
]


def gap_radar(book_id: str) -> dict[str, Any]:
    """Per-chapter coverage radar. Each chapter gets a vector over the
    six canonical sections; value = 1 if a draft exists for that
    section, minus 0.4 per unresolved gap targeting that chapter on
    the same section. Floors at 0.
    """
    if not book_id:
        raise ValueError("book_id required")

    with get_session() as session:
        chapters = session.execute(sql_text("""
            SELECT id::text, number, title
            FROM book_chapters WHERE book_id::text = :bid
            ORDER BY number
        """), {"bid": book_id}).fetchall()
        drafts = session.execute(sql_text("""
            SELECT d.chapter_id::text, LOWER(d.section_type) AS stype
            FROM drafts d
            JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE bc.book_id::text = :bid AND d.content IS NOT NULL
              AND length(trim(d.content)) > 200
        """), {"bid": book_id}).fetchall()
        gaps = session.execute(sql_text("""
            SELECT chapter_id::text, LOWER(description) AS desc
            FROM book_gaps
            WHERE book_id::text = :bid AND status = 'open'
              AND chapter_id IS NOT NULL
        """), {"bid": book_id}).fetchall()

    by_ch: dict[str, dict[str, float]] = {}
    for ch_id, number, title in chapters:
        by_ch[ch_id] = {
            "_number": number, "_title": title or f"Ch.{number}",
            **{ax: 0.0 for ax in _RADAR_AXES},
        }

    for ch_id, stype in drafts:
        if ch_id not in by_ch:
            continue
        if stype in _RADAR_AXES:
            by_ch[ch_id][stype] = max(by_ch[ch_id][stype], 1.0)

    for ch_id, desc in gaps:
        if ch_id not in by_ch:
            continue
        for ax in _RADAR_AXES:
            if ax in (desc or ""):
                by_ch[ch_id][ax] = max(0.0, by_ch[ch_id][ax] - 0.4)

    chapters_out: list[dict] = []
    for ch_id, rec in by_ch.items():
        chapters_out.append({
            "chapter_id": ch_id,
            "number": rec["_number"],
            "title": rec["_title"][:80],
            "values": [rec[ax] for ax in _RADAR_AXES],
        })
    chapters_out.sort(key=lambda c: c["number"] or 0)
    return {"axes": _RADAR_AXES, "chapters": chapters_out}
