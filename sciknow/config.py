from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base data directory (relative to cwd or absolute)
    data_dir: Path = Path("data")

    # PostgreSQL
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "sciknow"
    pg_password: str = "sciknow"
    pg_database: str = "sciknow"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Ollama
    ollama_host: str = "http://localhost:11434"

    # Models
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    llm_model: str = "qwen2.5:32b-instruct-q4_K_M"
    llm_fast_model: str = "mistral:7b-instruct-q4_K_M"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Crossref polite pool
    crossref_email: str = "user@example.com"

    # PDF converter backend. "mineru" (default) uses OpenDataLab MinerU 2.5
    # pipeline — best quality on scientific papers per OmniDocBench. "marker"
    # uses datalab-to/marker (legacy). "auto" tries MinerU first and falls
    # back to Marker on failure.
    pdf_converter_backend: str = "auto"

    # Ingestion
    # Chunks per bge-m3 batch. Default 32 is safe on a 24GB GPU when the LLM is
    # also resident; raise to 64-128 for embedder-only runs, lower to 8-16 if
    # running alongside a large LLM (e.g. 32B q4) on the same GPU.
    embedding_batch_size: int = 32

    # Marker batch_multiplier (parallel_factor). Higher = more pages per Marker
    # forward pass; costs more VRAM. ~5GB peak per unit on Marker docs. Safe
    # default 2 on a 24GB GPU.
    marker_batch_multiplier: int = 2

    # SQLAlchemy connection pool. Raise when running parallel ingestion workers
    # or bulk ops (db enrich, db expand) that open many sessions concurrently.
    pg_pool_size: int = 20
    pg_max_overflow: int = 20

    # Qdrant HNSW + quantization knobs. Changes apply only on collection
    # (re)creation — existing collections keep their original params.
    qdrant_hnsw_m: int = 32
    qdrant_hnsw_ef_construct: int = 256
    qdrant_hnsw_ef: int = 128
    qdrant_scalar_quantization: bool = True

    # Bulk-op concurrency knobs
    enrich_workers: int = 8            # concurrent Crossref/OpenAlex lookups
    expand_download_workers: int = 6   # concurrent OA PDF lookups/downloads
    llm_parallel_workers: int = 4      # concurrent LLM calls (match OLLAMA_NUM_PARALLEL)

    # Ingestion worker processes for `sciknow ingest directory`. Each worker
    # loads its own Marker (~5GB VRAM peak) + bge-m3 (~2.2GB). On a 24GB GPU
    # with an LLM actively resident (e.g. Ollama-held 32B q4 ~18GB), keep this
    # at 1. Raise to 2 only when the LLM is off-GPU (no Ollama model loaded,
    # or LLM moved to a remote host like DGX Spark).
    ingest_workers: int = 1

    # `sciknow db expand` relevance filter. Cosine similarity threshold under
    # which candidate references are dropped before download. 0.55 is a sane
    # default for bge-m3 — ~80% of on-topic refs score above it on a focused
    # library, while off-topic references (statistical methods cited from a
    # climate paper, etc.) typically score 0.3-0.45.
    expand_relevance_threshold: float = 0.55

    @computed_field
    @property
    def pg_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @computed_field
    @property
    def inbox_dir(self) -> Path:
        return self.data_dir / "inbox"

    @computed_field
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @computed_field
    @property
    def failed_dir(self) -> Path:
        return self.data_dir / "failed"

    @computed_field
    @property
    def mineru_output_dir(self) -> Path:
        return self.data_dir / "mineru_output"


settings = Settings()
