"""
Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Loaded lazily on first use and kept in memory.
"""
from __future__ import annotations

from sciknow.retrieval.hybrid_search import SearchCandidate

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker
        from sciknow.config import settings
        _reranker = FlagReranker(settings.reranker_model, use_fp16=True)
    return _reranker


def release_reranker() -> None:
    """Drop the cached reranker model and free VRAM."""
    global _reranker
    if _reranker is None:
        return
    try:
        del _reranker
    finally:
        _reranker = None
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def rerank(
    query: str,
    candidates: list[SearchCandidate],
    top_k: int = 10,
) -> list[SearchCandidate]:
    """
    Score each (query, chunk) pair with the cross-encoder and return top_k results
    sorted by descending reranker score.

    The reranker score is stored back into candidate.rrf_score so callers can
    display it without knowing which scoring stage produced it.
    """
    if not candidates:
        return []

    reranker = _get_reranker()

    pairs = [[query, c.content_preview] for c in candidates]
    scores: list[float] = reranker.compute_score(pairs, normalize=True)

    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    result = []
    for score, candidate in scored[:top_k]:
        candidate.rrf_score = float(score)
        result.append(candidate)
    return result
