"""Phase 54.6.82 (#11 follow-up) — search the visuals Qdrant collection.

Companion to ``hybrid_search`` for text chunks. Queries the per-project
visuals Qdrant collection (populated by ``sciknow db embed-visuals``)
using the same bge-m3 dense + sparse signals fused via RRF that
``hybrid_search`` uses for papers.

Returned hits point at rows in the PostgreSQL ``visuals`` table so
downstream callers can hydrate the full record (content / asset_path /
paper_title / etc.) via a single JOIN.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class VisualHit:
    """A single visuals-collection search result."""
    visual_id: str
    document_id: str
    kind: str
    figure_num: str
    ai_caption: str
    original_caption: str
    dense_score: float
    sparse_score: float
    rrf_score: float


def search_visuals(
    query: str,
    qdrant_client,
    candidate_k: int = 20,
    kind: str | None = None,
) -> list[VisualHit]:
    """Hybrid dense + sparse search over the visuals collection, fused
    via the same RRF used for papers.

    ``kind`` filter (server-side payload filter) restricts to a single
    kind — e.g. ``kind='equation'`` when the caller only wants
    paraphrased equations.

    When the visuals collection is empty (nothing has been embedded
    yet) the function returns an empty list rather than raising.
    """
    from qdrant_client.models import (
        FieldCondition, Filter, MatchValue,
    )
    from sciknow.retrieval.hybrid_search import _embed_query, _rrf_merge
    from sciknow.storage.qdrant import visuals_collection

    coll = visuals_collection()
    if not (query or "").strip():
        return []

    dense_vec, sparse_vec = _embed_query(query)
    qdrant_filter = None
    if kind:
        qdrant_filter = Filter(must=[
            FieldCondition(key="kind", match=MatchValue(value=kind))
        ])

    # Use the modern query_points API — client.search() was deprecated
    # in qdrant-client 1.9+. Same pattern as hybrid_search._qdrant_dense
    # / _qdrant_sparse.
    try:
        dense_resp = qdrant_client.query_points(
            collection_name=coll,
            query=dense_vec,
            using="dense",
            limit=candidate_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        dense = dense_resp.points
        sparse_resp = qdrant_client.query_points(
            collection_name=coll,
            query=sparse_vec,
            using="sparse",
            limit=candidate_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        sparse = sparse_resp.points
    except Exception as exc:
        logger.warning("visuals search failed: %s", exc)
        return []

    dense_ids = [str(p.id) for p in dense]
    sparse_ids = [str(p.id) for p in sparse]
    if not dense_ids and not sparse_ids:
        return []

    # RRF fuse. Dense gets slightly higher weight than sparse — same
    # default as hybrid_search for paper chunks.
    fused = _rrf_merge([dense_ids, sparse_ids], weights=[1.0, 1.0])
    fused = fused[:candidate_k]

    # Build lookup from point ID → (score, payload).
    by_id: dict[str, dict] = {}
    for p in dense:
        by_id.setdefault(str(p.id), {"dense": float(p.score), "sparse": 0.0,
                                      "payload": p.payload or {}})
    for p in sparse:
        rec = by_id.setdefault(str(p.id), {"dense": 0.0, "sparse": 0.0,
                                            "payload": p.payload or {}})
        rec["sparse"] = float(p.score)
        if not rec.get("payload"):
            rec["payload"] = p.payload or {}

    hits: list[VisualHit] = []
    for pid, rrf in fused:
        info = by_id.get(pid)
        if not info:
            continue
        payload = info["payload"] or {}
        hits.append(VisualHit(
            visual_id=payload.get("visual_id", ""),
            document_id=payload.get("document_id", ""),
            kind=payload.get("kind", ""),
            figure_num=payload.get("figure_num", ""),
            ai_caption=payload.get("ai_caption", ""),
            original_caption=payload.get("original_caption", ""),
            dense_score=info.get("dense", 0.0),
            sparse_score=info.get("sparse", 0.0),
            rrf_score=rrf,
        ))
    return hits
