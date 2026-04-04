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

    # Ingestion
    embedding_batch_size: int = 8

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
