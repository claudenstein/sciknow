"""Phase 54.6.228 — roadmap 3.4.3 Phase 2: ColBERT late-interaction rerank.

Two-stage abstract retrieval:

  1. **Prefetch** — dense-vector top-K candidates (K=50 by default).
     Cheap ANN over the abstracts collection's ``dense`` field.
  2. **Rerank** — ColBERT MAX_SIM late-interaction over the
     prefetched candidates, using the ``colbert`` multi-vector
     field populated in Phase 1 (54.6.227).

Runs entirely inside Qdrant via the ``query_points`` API with a
``prefetch`` clause — no round-trip to Python for the scoring loop.

**Fallback contract.** If the collection was created without the
``colbert`` slot (the user hasn't opted in yet), or if a runtime
query hits the slot-missing error, the function returns ``None`` so
the caller can drop back to the plain dense-only path. Callers
MUST handle ``None``; we explicitly do NOT fall back silently
because silent fallback would mask a misconfigured opt-in.

See ``docs/roadmap/ROADMAP_INGESTION.md`` §3.4.3.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from qdrant_client.models import Prefetch

from sciknow.config import settings
from sciknow.storage.qdrant import (
    ABSTRACTS_COLLECTION, get_client as _get_qdrant,
)

logger = logging.getLogger(__name__)


@dataclass
class ColbertRerankHit:
    """One rerank result. Lightweight wrapper so callers don't need
    to import Qdrant types."""
    document_id: str
    score: float              # MAX_SIM late-interaction score
    payload: dict[str, Any]


# Cached once per process — collection schema doesn't change under
# a running server, and one get_collection call per query is wasted
# RTT.
_SLOT_CHECKED: bool = False
_SLOT_PRESENT: bool = False


def _abstracts_has_colbert_slot() -> bool:
    """True when the abstracts collection has a `colbert` multi-vector
    field declared. Cached per-process."""
    global _SLOT_CHECKED, _SLOT_PRESENT
    if _SLOT_CHECKED:
        return _SLOT_PRESENT
    _SLOT_CHECKED = True
    try:
        client = _get_qdrant()
        info = client.get_collection(ABSTRACTS_COLLECTION)
        vectors = info.config.params.vectors or {}
        _SLOT_PRESENT = "colbert" in vectors
    except Exception as exc:
        logger.debug(
            "colbert slot check failed on abstracts: %s", exc
        )
        _SLOT_PRESENT = False
    return _SLOT_PRESENT


def _encode_query(text: str) -> tuple[list[float], list[list[float]]] | None:
    """Encode the query text with bge-m3, returning (dense, colbert).
    Returns None on any error so the caller can drop back to dense-only."""
    try:
        from sciknow.ingestion.embedder import _get_model
        model = _get_model()
        out = model.encode(
            [text], batch_size=1, max_length=512,
            return_dense=True, return_sparse=False,
            return_colbert_vecs=True,
        )
    except Exception as exc:
        logger.debug("bge-m3 query encode failed: %s", exc)
        return None

    dense_vec = out.get("dense_vecs")
    colbert_vecs = out.get("colbert_vecs")
    if dense_vec is None or colbert_vecs is None or not colbert_vecs:
        return None
    dense = dense_vec[0].tolist()
    colbert = [v.tolist() for v in colbert_vecs[0]]
    if not colbert:
        return None
    return dense, colbert


def search_abstracts_with_colbert(
    query: str,
    *,
    top_k: int = 10,
    prefetch_k: int = 50,
) -> list[ColbertRerankHit] | None:
    """Run dense-prefetch + ColBERT MAX_SIM rerank on the abstracts
    collection.

    Args:
        query: Natural-language query text.
        top_k: Final result count after rerank.
        prefetch_k: Candidate pool size from the dense prefetch.
                    50 is the roadmap's "cheap pilot" default — enough
                    recall to let MAX_SIM find the real winners
                    without paying the rerank cost on the whole
                    collection.

    Returns:
        List of ``ColbertRerankHit`` ordered by MAX_SIM score
        descending, OR ``None`` when the colbert path is unavailable
        (setting off, collection pre-flip, encoding error, Qdrant
        error). Callers must graceful-degrade on None.
    """
    if not settings.enable_colbert_abstracts:
        return None
    if not _abstracts_has_colbert_slot():
        return None
    if not query or not query.strip():
        return None

    encoded = _encode_query(query)
    if encoded is None:
        return None
    dense, colbert = encoded

    try:
        client = _get_qdrant()
        result = client.query_points(
            collection_name=ABSTRACTS_COLLECTION,
            prefetch=[
                Prefetch(
                    query=dense,
                    using="dense",
                    limit=prefetch_k,
                ),
            ],
            query=colbert,
            using="colbert",
            limit=top_k,
            with_payload=True,
        )
    except Exception as exc:
        logger.debug("colbert rerank query failed: %s", exc)
        return None

    hits: list[ColbertRerankHit] = []
    for p in result.points or []:
        payload = dict(p.payload or {})
        doc_id = str(payload.get("document_id") or "")
        hits.append(ColbertRerankHit(
            document_id=doc_id,
            score=float(p.score),
            payload=payload,
        ))
    return hits
