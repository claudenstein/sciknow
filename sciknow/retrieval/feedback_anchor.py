"""Phase 54.6.115 (Tier 2 #3) — bias the expand ranker's anchor
vector with user feedback.

Caller supplies the base anchor (usually ``compute_corpus_centroid()``
or a free-text query embedding). This module fetches bge-m3 embeddings
for the positive / negative papers the user has marked in
``<project>/expand_feedback.json`` and returns a modified anchor:

    biased = base + alpha_pos * mean(pos_embs) - alpha_neg * mean(neg_embs)

Defaults chosen to nudge, not dominate: ``alpha_pos=0.25, alpha_neg=0.25``
so even large |pos|/|neg| sets can only shift the centroid by up to
half the magnitude of ``base``. Fetches are cached per-process so
repeated expand rounds in the same session don't re-embed titles.

Embedding source preference, in order:
1. bge-m3 abstract from Qdrant (``abstracts`` collection). Exact
   representation — same one used at ingest time.
2. On-the-fly bge-m3 encoding of ``title`` + (if present) abstract
   from ``paper_metadata`` rows.
3. Skip — entries whose embedding can't be resolved are dropped
   silently.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


_CACHE: dict[str, np.ndarray] = {}  # key -> dense vec


def _resolve_embedding(entry) -> np.ndarray | None:
    """Fetch a bge-m3 dense vector for one FeedbackEntry.

    Tries Qdrant first (exact match on DOI / arxiv_id via payload
    filter on the abstracts collection), then falls back to encoding
    the title+abstract on-the-fly.
    """
    cache_key = entry.key
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    # 1. Qdrant lookup via DOI / arxiv_id payload filter
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        from sciknow.storage.qdrant import ABSTRACTS_COLLECTION, get_client
        client = get_client()
        conds = []
        if entry.doi:
            conds.append(FieldCondition(key="doi", match=MatchValue(value=entry.doi)))
        if entry.arxiv_id:
            conds.append(FieldCondition(key="arxiv_id", match=MatchValue(value=entry.arxiv_id)))
        if conds:
            flt = Filter(should=conds)
            try:
                points, _ = client.scroll(
                    collection_name=ABSTRACTS_COLLECTION,
                    scroll_filter=flt,
                    limit=1,
                    with_payload=False,
                    with_vectors=["dense"],
                )
                for p in points:
                    v = p.vector.get("dense") if isinstance(p.vector, dict) else p.vector
                    if v is not None:
                        vec = np.asarray(v, dtype=np.float32)
                        _CACHE[cache_key] = vec
                        return vec
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("feedback anchor: Qdrant lookup failed for %s: %s",
                     entry.key, exc)

    # 2. Encode title (+ abstract if we have it) on the fly
    text = entry.title or entry.doi or entry.arxiv_id
    if not text:
        return None
    try:
        from sciknow.ingestion.embedder import _get_model
        model = _get_model()
        out = model.encode(
            [text], batch_size=1, max_length=1024,
            return_dense=True, return_sparse=False,
            return_colbert_vecs=False,
        )
        vec = np.asarray(out["dense_vecs"][0], dtype=np.float32)
        _CACHE[cache_key] = vec
        return vec
    except Exception as exc:  # noqa: BLE001
        logger.debug("feedback anchor: on-the-fly embed failed for %s: %s",
                     entry.key, exc)
        return None


def bias_anchor(
    base_anchor: np.ndarray | None,
    positives: Iterable,
    negatives: Iterable,
    *,
    alpha_pos: float = 0.25,
    alpha_neg: float = 0.25,
) -> tuple[np.ndarray | None, dict]:
    """Return ``(biased_anchor, stats)`` where stats has the counts of
    resolved pos/neg embeddings for logging. When ``base_anchor`` is
    None (no corpus yet), the returned anchor is
    ``alpha_pos * mean(pos) - alpha_neg * mean(neg)`` (still usable).
    """
    if base_anchor is None and not (positives or negatives):
        return None, {"pos": 0, "neg": 0, "note": "no_base_no_feedback"}

    pos_vecs = [v for v in (_resolve_embedding(e) for e in (positives or [])) if v is not None]
    neg_vecs = [v for v in (_resolve_embedding(e) for e in (negatives or [])) if v is not None]

    biased = base_anchor.copy() if base_anchor is not None else None
    if pos_vecs:
        p_mean = np.mean(np.stack(pos_vecs, axis=0), axis=0)
        if biased is None:
            biased = alpha_pos * p_mean
        else:
            biased = biased + alpha_pos * p_mean
    if neg_vecs:
        n_mean = np.mean(np.stack(neg_vecs, axis=0), axis=0)
        if biased is None:
            biased = -alpha_neg * n_mean
        else:
            biased = biased - alpha_neg * n_mean

    return biased, {
        "pos": len(pos_vecs),
        "neg": len(neg_vecs),
        "alpha_pos": alpha_pos,
        "alpha_neg": alpha_neg,
    }


def clear_cache() -> None:
    """Drop the per-process embedding cache. Useful for tests."""
    _CACHE.clear()
