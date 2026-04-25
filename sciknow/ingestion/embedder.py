"""
Embed chunks with BAAI/bge-m3 (dense + sparse) and upsert into Qdrant.
The model is loaded once and reused across calls.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from sciknow.config import settings
from sciknow.ingestion.chunker import Chunk
from sciknow.storage.qdrant import ABSTRACTS_COLLECTION, PAPERS_COLLECTION

logger = logging.getLogger(__name__)

_model = None


class _LlamaCppBgeM3Adapter:
    """v2 Phase B — BGEM3FlagModel.encode() shim backed by llama-server.

    bge-m3 emits dense + sparse + colbert vectors from a single forward
    pass in the upstream FlagEmbedding implementation. llama-server's
    ``/v1/embeddings`` endpoint exposes only the dense channel.

    The adapter satisfies the call-shape contract by returning empty
    sparse maps (``{}``) and an empty colbert array when the caller
    asks for either. Downstream code in ``embed_chunks()`` then upserts
    a SparseVector with no indices, which Qdrant accepts cleanly. The
    practical effect on retrieval: hybrid_search loses the sparse
    lexical signal — RRF still fuses dense + FTS so recall holds, but
    the sparse channel's contribution to MRR (~+0.02 in v1) is
    deferred until a sparse role lands.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedder_model_name

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 8192,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
        **_extra,
    ) -> dict:
        from sciknow.infer import client as _infer_client
        # Stream large batches through the embedder server; llama-server
        # caps internal --ubatch at 512 tokens, so we let httpx's HTTP/1.1
        # do the cross-batch work and just shard at our batch_size.
        if not isinstance(texts, list):
            texts = list(texts)
        out_dense: list[list[float]] = []
        bs = max(1, int(batch_size or len(texts) or 1))
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            out_dense.extend(_infer_client.embed(chunk, model=self.model_name))

        # Mimic FlagEmbedding's numpy return for `dense_vecs`.
        try:
            import numpy as _np
            dense_arr = _np.asarray(out_dense, dtype=_np.float32)
        except Exception:
            dense_arr = out_dense  # type: ignore[assignment]

        result: dict = {}
        if return_dense:
            result["dense_vecs"] = dense_arr
        if return_sparse:
            result["lexical_weights"] = [{} for _ in texts]
        if return_colbert_vecs:
            result["colbert_vecs"] = []
        return result


def _get_model():
    global _model
    if _model is None:
        if getattr(settings, "use_llamacpp_embedder", True):
            _model = _LlamaCppBgeM3Adapter()
        else:
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

    Phase 54.6.305 — move to CPU first so the GPU-side tensors are
    truly deallocated instead of lingering in PyTorch's caching
    allocator (which keeps the memory visible to Ollama, forcing the
    writer model to partial-load).  BGEM3FlagModel wraps a transformer
    in its ``model`` attribute; the wrapper itself has no ``.to``.
    """
    global _model
    if _model is None:
        return
    try:
        inner = getattr(_model, "model", None)
        if inner is not None:
            to_cpu = getattr(inner, "to", None)
            if callable(to_cpu):
                to_cpu("cpu")
        else:
            to_cpu = getattr(_model, "to", None)
            if callable(to_cpu):
                to_cpu("cpu")
    except Exception:
        pass
    try:
        del _model
    finally:
        _model = None
    try:
        import gc, torch
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def release_dense_embedder() -> None:
    """Phase 54.6.290 — drop the Qwen3-Embedding-4B cache from the
    ingest process.  Called from the VRAM preflight releaser so a
    pre-convert preflight can reclaim ~8 GB without waiting for the
    process to exit."""
    global _dense_model_cache
    if _dense_model_cache is None:
        return
    try:
        del _dense_model_cache
    finally:
        _dense_model_cache = None
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def release_embedders() -> int:
    """Phase 54.6.290 — release both bge-m3 and the Qwen3 dense
    sidecar model at once.  Returns approximate MB freed for
    logging (authoritative number comes from nvidia-smi after)."""
    freed_approx = 0
    if _model is not None:
        release_model()
        freed_approx += 2300  # bge-m3 FP16 footprint
    if _dense_model_cache is not None:
        release_dense_embedder()
        freed_approx += 8000  # Qwen3-Embedding-4B BF16 footprint
    return freed_approx


# Register with the VRAM budget module so preflight() can fire this
# during a cascade (e.g. a reranker-load preflight will drop embedders
# ahead of the 8 GB reranker weights).  priority=50 (default) → last
# resort; embedders are the thing the embed stage *needs*.
try:
    from sciknow.core.vram_budget import register_releaser as _register
    _register("embedders", release_embedders, priority=50)
except ImportError:  # pragma: no cover
    pass


def _to_sparse(lexical_weights: dict) -> SparseVector:
    indices = []
    values = []
    for k, v in lexical_weights.items():
        indices.append(int(k))
        values.append(float(v))
    return SparseVector(indices=indices, values=values)


_dense_model_cache = None


def _get_dense_embedder():
    """Phase 54.6.279 — return the configured dense embedder
    (sentence-transformers SentenceTransformer, BF16) when the
    dual-embedder split is active, else None. Lazy-loaded and
    cached for the lifetime of the ingestion process.
    """
    tag = getattr(settings, "dense_embedder_model", None)
    if not tag or tag == settings.embedding_model:
        return None
    global _dense_model_cache
    if _dense_model_cache is None:
        import torch as _torch
        from sentence_transformers import SentenceTransformer
        _dense_model_cache = SentenceTransformer(
            tag, device="cuda", trust_remote_code=True,
            model_kwargs={"torch_dtype": _torch.bfloat16},
        )
        # Cap seq length so outlier long chunks don't OOM activations.
        # Matches the A/B harness policy. Chunks are typically
        # 500-2500 tokens so this truncates only the top few %.
        _dense_model_cache.max_seq_length = 2048
    return _dense_model_cache


def _sidecar_collection_name() -> str:
    """Mirror of retrieval/hybrid_search.py::_dense_collection_name.
    Resolves the sidecar name for the active project + configured
    dense embedder. Called only when dual-embedder split is active."""
    from sciknow.core.project import get_active_project
    explicit = getattr(settings, "dense_sidecar_collection", None)
    if explicit:
        return explicit
    tag = settings.dense_embedder_model
    prefix = get_active_project().qdrant_prefix
    slug = tag.replace("/", "_").replace(":", "_").lower()
    return f"{prefix}_ab_{slug}_papers"


def backfill_sidecar_payload(
    qdrant_client: QdrantClient, *, batch: int = 256,
) -> dict:
    """Phase 54.6.279 — copy payload from the prod papers collection
    onto matching sidecar points. One-shot migration for users who
    ran the A/B harness before the dual-embedder ship (the harness
    omitted payload on sidecar points to save disk; production
    retrieval needs payload for filter pushdown).

    Idempotent. Returns ``{matched, missing, skipped}`` counts.
    """
    if not getattr(settings, "dense_embedder_model", None):
        return {"matched": 0, "missing": 0, "skipped": 0,
                "note": "dual-embedder not active; nothing to backfill"}

    sidecar = _sidecar_collection_name()
    if not qdrant_client.collection_exists(sidecar):
        return {"matched": 0, "missing": 0, "skipped": 0,
                "note": f"sidecar {sidecar} does not exist"}

    matched = missing = skipped = 0
    missing_points: list[tuple[str, dict, str]] = []  # (id, payload, content_proxy)
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=PAPERS_COLLECTION,
            limit=batch, offset=offset, with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        ids_in_batch = [p.id for p in points]
        existing = qdrant_client.retrieve(
            collection_name=sidecar, ids=ids_in_batch,
            with_payload=True, with_vectors=False,
        )
        existing_map = {str(e.id): e for e in existing}
        for p in points:
            sid = str(p.id)
            sp = existing_map.get(sid)
            if not sp:
                missing += 1
                missing_points.append((sid, p.payload or {}, ""))
                continue
            if sp.payload:  # Already has payload, skip
                skipped += 1
                continue
            qdrant_client.set_payload(
                collection_name=sidecar, payload=p.payload or {},
                points=[sid],
            )
            matched += 1
        if offset is None:
            break

    # Phase 54.6.279 — also embed any chunks that exist in prod but
    # not the sidecar (new ingests added after the A/B harness ran).
    # Fetch their text content from PG (sidecar can't source it) and
    # dual-embed into the sidecar so retrieval is complete.
    embedded_missing = 0
    if missing_points:
        from sqlalchemy import text as _sql
        from sciknow.storage.db import get_session
        dense_model = _get_dense_embedder()
        if dense_model is not None:
            dense_model.max_seq_length = 2048
            missing_ids = [mid for mid, _, _ in missing_points]
            # Fetch content from chunks table by qdrant_point_id.
            with get_session() as session:
                rows = session.execute(_sql(
                    "SELECT qdrant_point_id::text AS qp, content "
                    "FROM chunks WHERE qdrant_point_id::text = ANY(:ids)"
                ), {"ids": missing_ids}).fetchall()
            text_by_id = {r.qp: r.content for r in rows}
            bsize = 8
            points_to_add: list[PointStruct] = []
            for i in range(0, len(missing_points), bsize):
                batch_chunk = missing_points[i:i + bsize]
                texts = [text_by_id.get(mid, "") for mid, _, _ in batch_chunk]
                # Skip any point whose text isn't available in the DB
                valid = [(mid, pld, txt) for (mid, pld, _), txt in zip(batch_chunk, texts) if txt]
                if not valid:
                    continue
                vecs = dense_model.encode(
                    [t for _, _, t in valid], batch_size=len(valid),
                    convert_to_numpy=True, normalize_embeddings=True,
                    show_progress_bar=False,
                )
                for (mid, pld, _), v in zip(valid, vecs):
                    points_to_add.append(PointStruct(
                        id=mid, vector={"dense": v.tolist()}, payload=pld,
                    ))
                embedded_missing += len(valid)
            if points_to_add:
                qdrant_client.upsert(
                    collection_name=sidecar, points=points_to_add,
                )

    return {
        "matched": matched, "missing": missing, "skipped": skipped,
        "embedded_missing": embedded_missing,
    }


def _ensure_sidecar_exists(qdrant_client: QdrantClient) -> str:
    """Create the sidecar collection on-demand if missing. Dim is
    read from settings.dense_embedder_dim. Returns the collection
    name. Idempotent — no-op when the collection already exists.

    Phase 54.6.296 — also ensures the same payload indexes that
    ``storage/qdrant.py::init_collections`` creates on the prod
    papers collection.  Without these, filter pushdown on the
    dense leg (``document_id``, ``year``, ``section_type``,
    ``domains``, ``journal``, ``node_level``) degrades to a full
    scan — the problem that motivated this phase (sidecars created
    pre-54.6.296 had zero payload indexes).  Calls are idempotent
    per the Qdrant semantics (``create_payload_index`` on an
    existing index is a no-op).
    """
    from qdrant_client.models import (
        Distance, VectorParams, PayloadSchemaType,
        HnswConfigDiff, ScalarQuantization, ScalarQuantizationConfig,
        ScalarType,
    )
    coll = _sidecar_collection_name()
    if not qdrant_client.collection_exists(coll):
        dim = int(getattr(settings, "dense_embedder_dim", 2560))
        # Phase 54.6.299 — apply the same HNSW + quantization tuning
        # as the prod papers collection (see init_collections in
        # storage/qdrant.py).  Pre-54.6.299 sidecars took Qdrant
        # defaults (m=16, ef_construct=100, no quantization) which
        # hurt dense-leg recall vs the tuned prod collection.  Since
        # the sidecar carries the same number of points as prod, it
        # should share the same tuning.
        hnsw_cfg = HnswConfigDiff(
            m=settings.qdrant_hnsw_m,
            ef_construct=settings.qdrant_hnsw_ef_construct,
        )
        quant_cfg = (
            ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8, always_ram=True,
                ),
            )
            if settings.qdrant_scalar_quantization else None
        )
        qdrant_client.create_collection(
            collection_name=coll,
            vectors_config={
                "dense": VectorParams(
                    size=dim, distance=Distance.COSINE,
                    hnsw_config=hnsw_cfg,
                ),
            },
            quantization_config=quant_cfg,
        )
    # Ensure payload indexes — must mirror init_collections() for the
    # prod papers collection so retrieval filters hit indexes on both
    # sides.  Idempotent: create_payload_index raises on duplicate,
    # which we swallow.
    _expected_indexes = [
        ("document_id", PayloadSchemaType.KEYWORD),
        ("section_type", PayloadSchemaType.KEYWORD),
        ("year", PayloadSchemaType.INTEGER),
        ("domains", PayloadSchemaType.KEYWORD),
        ("journal", PayloadSchemaType.KEYWORD),
        ("node_level", PayloadSchemaType.INTEGER),
    ]
    for field, schema in _expected_indexes:
        try:
            qdrant_client.create_payload_index(coll, field, schema)
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" in msg or "already created" in msg:
                continue
            logger.debug(
                "sidecar payload index %s failed: %s", field, exc,
            )
    return coll


def embed_chunks(
    chunks: list[Chunk],
    document_id: UUID,
    payload_base: dict,
    qdrant_client: QdrantClient,
) -> list[UUID]:
    """
    Embed a list of chunks and upsert them into the 'papers' collection.
    Returns list of Qdrant point UUIDs (same order as input chunks).

    Phase 54.6.279 — when ``settings.dense_embedder_model`` is set
    to a model other than bge-m3, ingestion also encodes every chunk
    with the dense embedder (Qwen3-Embedding-4B by default) and
    upserts to the configured sidecar collection. The prod papers
    collection still carries bge-m3's sparse vectors; its dense
    vectors are left populated but ignored at query time when the
    split is active.
    """
    if not chunks:
        return []

    model = _get_model()
    dense_model = _get_dense_embedder()  # None when split inactive
    sidecar_coll = (
        _ensure_sidecar_exists(qdrant_client) if dense_model else None
    )
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

        # Phase 54.6.279 — parallel dense-embed pass when the split
        # is active. Documents are encoded WITHOUT the query-side
        # instruction prefix (HF model card: queries get "Instruct:
        # … Query: …", documents get raw text).
        sidecar_dense_vecs = None
        if dense_model is not None:
            sidecar_batch = min(len(batch_texts), 8)  # smaller batch for Qwen3-4B
            sidecar_dense_vecs = dense_model.encode(
                batch_texts, batch_size=sidecar_batch,
                convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )

        points = []
        sidecar_points = []
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

            if sidecar_dense_vecs is not None:
                # Full payload copied to sidecar so hybrid_search's
                # dense leg honours filters (year/domain/section_type
                # etc.) identically on the sidecar collection. Storage
                # overhead is small — payload is shared text/numbers,
                # ~200 bytes per point × 31k points ≈ 6 MB for the
                # current corpus, negligible next to the 2560-dim
                # vectors themselves (~40 MB in float32).
                sidecar_points.append(PointStruct(
                    id=str(point_id),
                    vector={"dense": sidecar_dense_vecs[i].tolist()},
                    payload=payload,
                ))

        qdrant_client.upsert(collection_name=PAPERS_COLLECTION, points=points)
        if sidecar_points:
            qdrant_client.upsert(
                collection_name=sidecar_coll, points=sidecar_points,
            )

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
