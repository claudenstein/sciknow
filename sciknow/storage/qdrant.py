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

PAPERS_COLLECTION = "papers"
ABSTRACTS_COLLECTION = "abstracts"
WIKI_COLLECTION = "wiki"


# Module-level singleton client — QdrantClient holds an httpx session internally;
# reusing it avoids per-call connection setup across the many retrieval and
# ingestion call sites.
_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _client


def init_collections(client: QdrantClient | None = None) -> None:
    if client is None:
        client = get_client()

    existing = {c.name for c in client.get_collections().collections}

    if PAPERS_COLLECTION not in existing:
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
            collection_name=PAPERS_COLLECTION,
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
        ]:
            client.create_payload_index(PAPERS_COLLECTION, field, schema)

    if ABSTRACTS_COLLECTION not in existing:
        client.create_collection(
            collection_name=ABSTRACTS_COLLECTION,
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
            ABSTRACTS_COLLECTION, "document_id", PayloadSchemaType.KEYWORD
        )


    if WIKI_COLLECTION not in existing:
        client.create_collection(
            collection_name=WIKI_COLLECTION,
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
            client.create_payload_index(WIKI_COLLECTION, field, schema)


def check_connection(client: QdrantClient | None = None) -> bool:
    try:
        if client is None:
            client = get_client()
        client.get_collections()
        return True
    except Exception:
        return False
