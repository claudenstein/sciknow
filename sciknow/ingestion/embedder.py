"""
Embed chunks with BAAI/bge-m3 (dense + sparse) and upsert into Qdrant.
The model is loaded once and reused across calls.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from sciknow.config import settings
from sciknow.ingestion.chunker import Chunk
from sciknow.storage.qdrant import ABSTRACTS_COLLECTION, PAPERS_COLLECTION

_model = None


def _get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        from sciknow.retrieval.device import load_with_cpu_fallback

        # Phase 15.2 — CPU fallback when the GPU is mostly full of an LLM.
        # Falls back transparently if CUDA OOMs during load. The cache in
        # device.py remembers the choice for the rest of the session.
        _model = load_with_cpu_fallback(
            BGEM3FlagModel, settings.embedding_model, use_fp16=True,
        )
    return _model


def release_model() -> None:
    """
    Drop the cached bge-m3 model from this process and clear the CUDA
    allocator cache. Used before spawning ingestion worker subprocesses in
    `db expand`, so the main process doesn't hold a redundant copy of the
    embedder while each worker loads its own. Safe to call when no model
    has been loaded yet.
    """
    global _model
    if _model is None:
        return
    try:
        del _model
    finally:
        _model = None
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _to_sparse(lexical_weights: dict) -> SparseVector:
    indices = []
    values = []
    for k, v in lexical_weights.items():
        indices.append(int(k))
        values.append(float(v))
    return SparseVector(indices=indices, values=values)


def embed_chunks(
    chunks: list[Chunk],
    document_id: UUID,
    payload_base: dict,
    qdrant_client: QdrantClient,
) -> list[UUID]:
    """
    Embed a list of chunks and upsert them into the 'papers' collection.
    Returns list of Qdrant point UUIDs (same order as input chunks).
    """
    if not chunks:
        return []

    model = _get_model()
    texts = [c.content for c in chunks]
    point_ids: list[UUID] = []

    batch_size = settings.embedding_batch_size
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        batch_chunks = chunks[batch_start : batch_start + batch_size]

        output = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs = output["dense_vecs"]
        sparse_vecs = output["lexical_weights"]

        points = []
        for i, chunk in enumerate(batch_chunks):
            point_id = uuid4()
            point_ids.append(point_id)

            payload = {
                **payload_base,
                "chunk_id": str(point_id),
                "document_id": str(document_id),
                "section_type": chunk.section_type,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "content_preview": chunk.raw_content[:200],
                "embedding_model": settings.embedding_model,
            }

            points.append(PointStruct(
                id=str(point_id),
                vector={
                    "dense": dense_vecs[i].tolist(),
                    "sparse": _to_sparse(sparse_vecs[i]),
                },
                payload=payload,
            ))

        qdrant_client.upsert(collection_name=PAPERS_COLLECTION, points=points)

    return point_ids


def embed_summary_text(
    summary_text: str,
    payload: dict,
    qdrant_client: QdrantClient,
) -> UUID | None:
    """
    Embed an arbitrary text into the `papers` collection with the given payload.

    Used by sciknow.ingestion.raptor to add hierarchical summary nodes to the
    same collection as the leaf chunks. The caller is responsible for setting
    `node_level`, `summary_text`, `child_chunk_ids`, etc. in `payload`.

    Returns the new Qdrant point UUID, or None if the text was empty.
    """
    if not summary_text or not summary_text.strip():
        return None

    model = _get_model()
    output = model.encode(
        [summary_text],
        batch_size=1,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    point_id = uuid4()
    full_payload = {
        **payload,
        "chunk_id": str(point_id),
        "embedding_model": settings.embedding_model,
    }
    qdrant_client.upsert(
        collection_name=PAPERS_COLLECTION,
        points=[
            PointStruct(
                id=str(point_id),
                vector={
                    "dense": output["dense_vecs"][0].tolist(),
                    "sparse": _to_sparse(output["lexical_weights"][0]),
                },
                payload=full_payload,
            )
        ],
    )
    return point_id


def embed_to_visuals_collection(
    text_to_embed: str,
    payload: dict,
    qdrant_client: QdrantClient,
) -> UUID | None:
    """Phase 54.6.82 (#11 follow-up) — embed arbitrary text into the
    active project's *visuals* Qdrant collection.

    Used to index equation paraphrases (Phase 54.6.78) and figure/chart
    AI captions (Phase 54.6.72) for retrieval. The caller is expected
    to set ``kind``, ``document_id``, ``visual_id`` etc. in ``payload``.

    Returns the new Qdrant point UUID, or None for empty input.
    """
    if not text_to_embed or not text_to_embed.strip():
        return None

    from sciknow.storage.qdrant import visuals_collection
    model = _get_model()
    output = model.encode(
        [text_to_embed],
        batch_size=1,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    point_id = uuid4()
    full_payload = {
        **payload,
        "visual_point_id": str(point_id),
        "embedding_model": settings.embedding_model,
    }
    qdrant_client.upsert(
        collection_name=visuals_collection(),
        points=[
            PointStruct(
                id=str(point_id),
                vector={
                    "dense": output["dense_vecs"][0].tolist(),
                    "sparse": _to_sparse(output["lexical_weights"][0]),
                },
                payload=full_payload,
            )
        ],
    )
    return point_id


def embed_paper_abstract(paper_metadata_id: str) -> UUID | None:
    """Phase 54.6.111 (Tier 1 #4) — re-embed a paper's abstract into
    Qdrant when it was just updated (e.g. by enrich filling in a
    previously-NULL abstract).

    Looks up ``paper_metadata`` by id, joins to ``documents`` for the
    document_id, and writes a new point into the `abstracts` Qdrant
    collection. Idempotent: the old point stays (no ledger to find it)
    but the newer one will outrank it at query time via recency.
    """
    from sqlalchemy import text as _text
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client as _get_q

    with get_session() as session:
        row = session.execute(_text("""
            SELECT pm.abstract, pm.title, pm.year, pm.document_id::text
            FROM paper_metadata pm
            WHERE pm.id::text = :pid
        """), {"pid": paper_metadata_id}).fetchone()
    if not row or not (row[0] or "").strip():
        return None
    abstract_text, title, year, doc_id = row
    payload = {
        "title": title or "",
        "year": year,
        "paper_metadata_id": paper_metadata_id,
        "embedded_via": "enrich-reembed",
    }
    return embed_abstract(
        abstract_text=abstract_text,
        document_id=doc_id,
        payload_base=payload,
        qdrant_client=_get_q(),
    )


def embed_abstract(
    abstract_text: str,
    document_id: UUID,
    payload_base: dict,
    qdrant_client: QdrantClient,
) -> UUID | None:
    """Embed the paper abstract into the 'abstracts' collection.

    Phase 54.6.227 (roadmap 3.4.3 Phase 1) — when
    ``settings.enable_colbert_abstracts`` is True, also requests
    ColBERT token vectors from bge-m3 and writes them to the
    ``colbert`` multi-vector field on the point. The field has to
    be declared on the collection at creation time (see
    ``qdrant.init_collections``) — embedding into a collection that
    was created pre-flip with this setting True is a no-op on the
    colbert slot and only the dense + sparse halves populate.
    """
    if not abstract_text.strip():
        return None

    from sciknow.config import settings as _s

    model = _get_model()
    want_colbert = bool(_s.enable_colbert_abstracts)
    output = model.encode(
        [abstract_text],
        batch_size=1,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=want_colbert,
    )

    vec_payload: dict = {
        "dense": output["dense_vecs"][0].tolist(),
        "sparse": _to_sparse(output["lexical_weights"][0]),
    }
    if want_colbert:
        # bge-m3 returns colbert vecs as a list of np.ndarrays, one
        # per token. Qdrant multi-vector expects list[list[float]].
        colbert_vecs = output.get("colbert_vecs") or []
        if colbert_vecs and len(colbert_vecs) > 0:
            first = colbert_vecs[0]
            if _abstracts_collection_has_colbert(qdrant_client):
                vec_payload["colbert"] = [v.tolist() for v in first]
            else:
                _warn_colbert_slot_missing_once()

    point_id = uuid4()
    qdrant_client.upsert(
        collection_name=ABSTRACTS_COLLECTION,
        points=[
            PointStruct(
                id=str(point_id),
                vector=vec_payload,
                payload={
                    **payload_base,
                    "document_id": str(document_id),
                    "content_preview": abstract_text[:500],
                },
            )
        ],
    )
    return point_id


# Phase 54.6.227 (roadmap 3.4.3 Phase 1) — guard against the
# config-drift case where `enable_colbert_abstracts=True` but the
# collection was created pre-flip (dense+sparse only). Without this
# the upsert would crash per-embed with "unknown vector name
# 'colbert'". The check caches its result per-process so we pay one
# get_collection() call at startup, not per embed. Users who flip
# the setting on an existing install see one warning and then clean
# dense+sparse embeds until they reset the abstracts collection.
_COLBERT_SLOT_CACHE: dict[str, bool] = {}
_COLBERT_WARNED: bool = False


def _abstracts_collection_has_colbert(qdrant_client: QdrantClient) -> bool:
    """True when the abstracts collection carries a `colbert` named
    vector. Cached per-process, refreshed on `db init` since that
    recreates the collection."""
    key = ABSTRACTS_COLLECTION
    if key in _COLBERT_SLOT_CACHE:
        return _COLBERT_SLOT_CACHE[key]
    try:
        info = qdrant_client.get_collection(key)
        vectors = info.config.params.vectors or {}
        has = "colbert" in vectors
    except Exception:
        has = False
    _COLBERT_SLOT_CACHE[key] = has
    return has


def _warn_colbert_slot_missing_once() -> None:
    global _COLBERT_WARNED
    if _COLBERT_WARNED:
        return
    _COLBERT_WARNED = True
    import warnings
    warnings.warn(
        "ENABLE_COLBERT_ABSTRACTS=True but the abstracts Qdrant "
        "collection was created without a `colbert` multi-vector "
        "slot. New embeds will be dense+sparse only. To enable "
        "colbert: `sciknow db reset` + `sciknow db init` (destructive "
        "— recreates every collection) or delete just the abstracts "
        "collection manually and re-run `db init`.",
        UserWarning,
        stacklevel=2,
    )
