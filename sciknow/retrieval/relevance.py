"""
Relevance filtering for collection expansion.

Given a list of candidate references and either a free-text query or the
current corpus as an anchor, compute cosine similarity against the bge-m3
dense embedding and return per-candidate scores. Used by `sciknow corpus expand`
to drop off-topic references before paying the download + ingest cost.

Two anchor modes:
  - "query"    — embed a user-provided query string (most targeted)
  - "centroid" — compute the mean of all abstract embeddings in the Qdrant
                 `abstracts` collection (represents the whole corpus in one
                 vector; best when your library is thematically coherent)

Both use the SAME model that the ingestion pipeline uses (bge-m3 FP16), so
candidate embeddings are directly comparable to the existing collection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sciknow.storage.qdrant import ABSTRACTS_COLLECTION, get_client as get_qdrant


@dataclass
class RelevanceResult:
    """Per-candidate relevance score and the reference it belongs to."""
    ref_index: int         # position in the input list
    score: float           # cosine similarity in [-1, 1], typically [0, 1]


def compute_corpus_centroid() -> np.ndarray | None:
    """
    Return the mean of all dense abstract embeddings in the Qdrant
    `abstracts` collection, or None if the collection is empty.

    Scrolls through the collection in batches (abstracts is small — typically
    a few hundred to a few thousand vectors) and averages the dense vectors.
    The centroid vector is NOT L2-normalised here; callers normalise both
    the centroid and candidate embeddings before taking the dot product.
    """
    client = get_qdrant()

    try:
        info = client.get_collection(ABSTRACTS_COLLECTION)
    except Exception:
        return None

    total = info.points_count or 0
    if total == 0:
        return None

    vecs: list[np.ndarray] = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=ABSTRACTS_COLLECTION,
            limit=256,
            offset=next_offset,
            with_payload=False,
            with_vectors=["dense"],
        )
        for p in points:
            v = p.vector
            if isinstance(v, dict):
                v = v.get("dense")
            if v is None:
                continue
            vecs.append(np.asarray(v, dtype=np.float32))
        if next_offset is None:
            break

    if not vecs:
        return None

    mean = np.mean(np.stack(vecs, axis=0), axis=0)
    return mean


def _l2_normalise(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm


def score_candidates(
    titles: list[str],
    anchor_vector: np.ndarray,
    batch_size: int = 64,
) -> list[float]:
    """
    Embed candidate `titles` with bge-m3 (dense only) and return cosine
    similarity to `anchor_vector` for each.

    `anchor_vector` is expected to be a bge-m3 dense vector of dimension
    `settings.embedding_dim`. It will be L2-normalised internally; caller
    may pass it already-normalised with no difference.
    """
    if not titles:
        return []

    from sciknow.ingestion.embedder import _get_model

    model = _get_model()
    anchor = _l2_normalise(anchor_vector.astype(np.float32))

    scores: list[float] = []
    for start in range(0, len(titles), batch_size):
        batch = titles[start : start + batch_size]
        out = model.encode(
            batch,
            batch_size=len(batch),
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        # Row-wise L2 normalise, then dot with the anchor.
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        dense_n = dense / norms
        batch_scores = dense_n @ anchor
        scores.extend(float(s) for s in batch_scores.tolist())

    return scores


def embed_query(query: str) -> np.ndarray:
    """Return the bge-m3 dense embedding of a single free-text query."""
    from sciknow.ingestion.embedder import _get_model

    model = _get_model()
    out = model.encode(
        [query],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return np.asarray(out["dense_vecs"][0], dtype=np.float32)


def score_histogram(scores: list[float], bins: int = 10) -> list[tuple[float, float, int]]:
    """
    Return `(low, high, count)` tuples describing a histogram of `scores`
    across `bins` equal-width buckets from min to max. Useful for showing
    the user the relevance-score distribution in dry-run mode.
    """
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return [(lo, hi, len(scores))]
    width = (hi - lo) / bins
    out: list[tuple[float, float, int]] = []
    for i in range(bins):
        b_lo = lo + i * width
        b_hi = b_lo + width
        if i == bins - 1:
            count = sum(1 for s in scores if b_lo <= s <= b_hi)
        else:
            count = sum(1 for s in scores if b_lo <= s < b_hi)
        out.append((b_lo, b_hi, count))
    return out
