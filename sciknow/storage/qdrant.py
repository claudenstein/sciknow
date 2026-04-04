from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    SparseVectorParams,
    VectorParams,
)

from sciknow.config import settings

PAPERS_COLLECTION = "papers"
ABSTRACTS_COLLECTION = "abstracts"


def get_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def init_collections(client: QdrantClient | None = None) -> None:
    if client is None:
        client = get_client()

    existing = {c.name for c in client.get_collections().collections}

    if PAPERS_COLLECTION not in existing:
        client.create_collection(
            collection_name=PAPERS_COLLECTION,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=True,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=50000,
            ),
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


def check_connection(client: QdrantClient | None = None) -> bool:
    try:
        if client is None:
            client = get_client()
        client.get_collections()
        return True
    except Exception:
        return False
