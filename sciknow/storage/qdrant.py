"""Qdrant connection + collection initialization.

Phase 43c — project-aware. Collection names are now derived from the
active project (``<slug>_papers`` etc.) instead of being hardcoded
strings. Existing code that imports ``PAPERS_COLLECTION`` /
``ABSTRACTS_COLLECTION`` / ``WIKI_COLLECTION`` keeps working thanks to
a module-level ``__getattr__`` fallback (PEP 562) that resolves the old
names to the active project's collection at lookup time.

- Legacy / ``default`` project: unprefixed collection names
  (``papers``, ``abstracts``, ``wiki``). Keeps pre-Phase-43 deployments
  working without migration.
- Real project: collection names are ``<sql_safe_slug>_papers`` etc.
- Cross-project access: pass the collection name as a string. Most
  call sites use the local constants, which means they automatically
  target the active project. For cross-project operations (e.g. the
  one-shot migration in Phase 43f) pass explicit names.

The Qdrant *client* stays global — one Qdrant instance serves every
project; only the collection names differ. No reason to spin up per-
project HTTP sessions.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    QuantizationSearchParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseVectorParams,
    VectorParams,
)

from sciknow.config import settings

# ── Project-aware collection names ──────────────────────────────────────


def papers_collection() -> str:
    """Return the active project's ``papers`` collection name."""
    from sciknow.core.project import get_active_project
    return get_active_project().papers_collection


def abstracts_collection() -> str:
    """Return the active project's ``abstracts`` collection name."""
    from sciknow.core.project import get_active_project
    return get_active_project().abstracts_collection


def wiki_collection() -> str:
    """Return the active project's ``wiki`` collection name."""
    from sciknow.core.project import get_active_project
    return get_active_project().wiki_collection


def visuals_collection() -> str:
    """Return the active project's ``visuals`` collection name (Phase 21.b)."""
    from sciknow.core.project import get_active_project
    p = get_active_project()
    return f"{p.qdrant_prefix}visuals"


# ── Legacy-constant compatibility shim (PEP 562) ───────────────────────
#
# Pre-Phase-43 call sites import the names ``PAPERS_COLLECTION``,
# ``ABSTRACTS_COLLECTION``, and ``WIKI_COLLECTION`` directly. Rather than
# touching ~60 call sites mechanically, we resolve these module-level
# attributes on demand via ``__getattr__``. Each access returns the
# active project's collection name, so the constants Just Work when the
# active project changes between processes.
#
# Within a single process the active project is stable, so the value
# captured by ``from ... import PAPERS_COLLECTION`` at import time is
# the right one for the entire run. When Phase 43e's
# ``sciknow project use`` is invoked, a fresh CLI process picks up the
# new project automatically.

_LEGACY_ALIASES = {
    "PAPERS_COLLECTION": papers_collection,
    "ABSTRACTS_COLLECTION": abstracts_collection,
    "WIKI_COLLECTION": wiki_collection,
}


def __getattr__(name: str):  # PEP 562 — module-level fallback
    if name in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[name]()
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


# ── Client singleton ────────────────────────────────────────────────────
#
# QdrantClient holds an httpx session internally; reusing it avoids
# per-call connection setup across the many retrieval and ingestion
# call sites.

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _client


# ── Collection provisioning ─────────────────────────────────────────────


def init_collections(client: QdrantClient | None = None) -> None:
    """Create the active project's collections if they don't exist.

    Idempotent — collections that already exist are skipped. Reads the
    active project's collection names at call time so ``sciknow db init``
    against a freshly-created project produces the right prefixes.
    """
    if client is None:
        client = get_client()

    # Resolve once at call time so we use consistent names throughout
    # this invocation (guards against the unlikely case of the active
    # project changing mid-call).
    papers_coll = papers_collection()
    abstracts_coll = abstracts_collection()
    wiki_coll = wiki_collection()

    existing = {c.name for c in client.get_collections().collections}

    # Phase 54.6.21 — validate that pre-existing collections still
    # match settings.embedding_dim. Without this check, changing
    # EMBEDDING_DIM in .env between runs silently produces vectors
    # sized for the new model in a collection still configured for the
    # old one — Qdrant returns 422 buried in an upsert log line, or
    # (worse) silent corruption depending on version. Fail fast and
    # tell the user how to recover.
    for coll in (papers_coll, abstracts_coll, wiki_coll, visuals_collection()):
        if coll in existing:
            try:
                info = client.get_collection(coll)
                cfg = info.config.params.vectors
                actual = (
                    cfg["dense"].size if isinstance(cfg, dict) and "dense" in cfg
                    else (cfg.size if hasattr(cfg, "size") else None)
                )
            except Exception:
                actual = None  # introspection failed — don't block init
            if actual is not None and actual != settings.embedding_dim:
                raise ValueError(
                    f"Qdrant collection {coll!r} has dense vector size "
                    f"{actual}, but settings.embedding_dim={settings.embedding_dim}. "
                    f"Either revert EMBEDDING_DIM in .env or drop the "
                    f"collection (`sciknow db reset`) and re-ingest."
                )

    if papers_coll not in existing:
        # Scalar int8 quantization trades a tiny recall hit for ~75% memory
        # savings on dense vectors; `always_ram=True` keeps the quantized copy
        # hot while original vectors stay on disk (see settings.qdrant_*).
        quant_config = (
            ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )
            if settings.qdrant_scalar_quantization
            else None
        )
        client.create_collection(
            collection_name=papers_coll,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=True,
                    hnsw_config=HnswConfigDiff(
                        m=settings.qdrant_hnsw_m,
                        ef_construct=settings.qdrant_hnsw_ef_construct,
                    ),
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=50000,
            ),
            quantization_config=quant_config,
        )
        # Payload indexes for fast filtering
        for field, schema in [
            ("document_id", PayloadSchemaType.KEYWORD),
            ("section_type", PayloadSchemaType.KEYWORD),
            ("year", PayloadSchemaType.INTEGER),
            ("domains", PayloadSchemaType.KEYWORD),
            ("journal", PayloadSchemaType.KEYWORD),
            # node_level is the RAPTOR tree level: 0 = leaf chunk,
            # 1+ = cluster summary at that hierarchical level. Indexed
            # so retrieval can opt in/out of summary nodes via filter.
            ("node_level", PayloadSchemaType.INTEGER),
        ]:
            client.create_payload_index(papers_coll, field, schema)

    if abstracts_coll not in existing:
        client.create_collection(
            collection_name=abstracts_coll,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=False,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        client.create_payload_index(
            abstracts_coll, "document_id", PayloadSchemaType.KEYWORD
        )


    if wiki_coll not in existing:
        client.create_collection(
            collection_name=wiki_coll,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=False,  # in-memory — wiki pages are few
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        for field, schema in [
            ("page_type", PayloadSchemaType.KEYWORD),
            ("slug", PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(wiki_coll, field, schema)

    # Phase 21.b — visuals collection for caption-based retrieval
    visuals_coll = visuals_collection()
    if visuals_coll not in existing:
        client.create_collection(
            collection_name=visuals_coll,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=False,  # in-memory — visuals are few per project
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        for field, schema in [
            ("document_id", PayloadSchemaType.KEYWORD),
            ("kind", PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(visuals_coll, field, schema)


def ensure_node_level_index(client: QdrantClient | None = None) -> bool:
    """
    Ensure the `node_level` payload index exists on the active project's
    papers collection.

    init_collections() only creates the index when the collection itself is
    being created, so existing installations from before the RAPTOR work
    will not have it. This helper is idempotent and safe to call from
    `sciknow catalog raptor build` to make sure the index is in place
    before any RAPTOR upserts.

    Returns True if the index exists (either was already present or was
    created), False if creation failed for an unexpected reason.
    """
    if client is None:
        client = get_client()
    try:
        client.create_payload_index(
            papers_collection(),
            "node_level",
            PayloadSchemaType.INTEGER,
        )
        return True
    except Exception as exc:
        # Qdrant raises if the index already exists; that's fine.
        msg = str(exc).lower()
        if "already exists" in msg or "already created" in msg:
            return True
        # Anything else is a real failure.
        import logging
        logging.getLogger(__name__).warning(
            "ensure_node_level_index failed: %s", exc,
        )
        return False


def check_connection(client: QdrantClient | None = None) -> bool:
    try:
        if client is None:
            client = get_client()
        client.get_collections()
        return True
    except Exception:
        return False
