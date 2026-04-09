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

        _model = BGEM3FlagModel(settings.embedding_model, use_fp16=True)
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


def embed_abstract(
    abstract_text: str,
    document_id: UUID,
    payload_base: dict,
    qdrant_client: QdrantClient,
) -> UUID | None:
    """Embed the paper abstract into the 'abstracts' collection."""
    if not abstract_text.strip():
        return None

    model = _get_model()
    output = model.encode(
        [abstract_text],
        batch_size=1,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    point_id = uuid4()
    qdrant_client.upsert(
        collection_name=ABSTRACTS_COLLECTION,
        points=[
            PointStruct(
                id=str(point_id),
                vector={
                    "dense": output["dense_vecs"][0].tolist(),
                    "sparse": _to_sparse(output["lexical_weights"][0]),
                },
                payload={
                    **payload_base,
                    "document_id": str(document_id),
                    "content_preview": abstract_text[:500],
                },
            )
        ],
    )
    return point_id
