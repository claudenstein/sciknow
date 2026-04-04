"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-01
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm"')

    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("file_hash", sa.Text, nullable=False, unique=True),
        sa.Column("original_path", sa.Text, nullable=False),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger),
        sa.Column("ingestion_status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("ingestion_error", sa.Text),
        sa.Column("mineru_output_path", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_documents_status", "documents", ["ingestion_status"])
    op.create_index("idx_documents_hash", "documents", ["file_hash"])

    op.create_table(
        "paper_metadata",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("title", sa.Text),
        sa.Column("abstract", sa.Text),
        sa.Column("year", sa.SmallInteger),
        sa.Column("doi", sa.Text),
        sa.Column("arxiv_id", sa.Text),
        sa.Column("journal", sa.Text),
        sa.Column("volume", sa.Text),
        sa.Column("issue", sa.Text),
        sa.Column("pages", sa.Text),
        sa.Column("publisher", sa.Text),
        sa.Column("authors", JSONB, nullable=False, server_default="[]"),
        sa.Column("keywords", ARRAY(sa.Text)),
        sa.Column("domains", ARRAY(sa.Text)),
        sa.Column("metadata_source", sa.Text, nullable=False, server_default="unknown"),
        sa.Column("crossref_raw", JSONB),
        sa.Column("arxiv_raw", JSONB),
        sa.Column("search_vector", TSVECTOR),
        sa.Column("extra", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_metadata_doi", "paper_metadata", ["doi"], postgresql_where=sa.text("doi IS NOT NULL"))
    op.create_index("idx_metadata_arxiv", "paper_metadata", ["arxiv_id"], postgresql_where=sa.text("arxiv_id IS NOT NULL"))
    op.create_index("idx_metadata_year", "paper_metadata", ["year"], postgresql_where=sa.text("year IS NOT NULL"))
    op.create_index("idx_metadata_domains", "paper_metadata", ["domains"], postgresql_using="gin")
    op.create_index("idx_metadata_keywords", "paper_metadata", ["keywords"], postgresql_using="gin")
    op.create_index("idx_metadata_fts", "paper_metadata", ["search_vector"], postgresql_using="gin")

    # Trigger to maintain search_vector
    op.execute("""
        CREATE FUNCTION update_search_vector() RETURNS TRIGGER AS $$
        BEGIN
            NEW.search_vector :=
                setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
                setweight(to_tsvector('english', COALESCE(NEW.abstract, '')), 'B') ||
                setweight(to_tsvector('english', COALESCE(array_to_string(NEW.keywords, ' '), '')), 'C') ||
                setweight(to_tsvector('english', COALESCE(NEW.journal, '')), 'D');
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trig_update_search_vector
            BEFORE INSERT OR UPDATE ON paper_metadata
            FOR EACH ROW EXECUTE FUNCTION update_search_vector();
    """)

    op.create_table(
        "paper_sections",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_type", sa.Text, nullable=False),
        sa.Column("section_title", sa.Text),
        sa.Column("section_index", sa.SmallInteger, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("word_count", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_sections_document", "paper_sections", ["document_id"])
    op.create_index("idx_sections_type", "paper_sections", ["section_type"])

    op.create_table(
        "chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_id", UUID(as_uuid=True), sa.ForeignKey("paper_sections.id", ondelete="SET NULL")),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("section_type", sa.Text),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("content_tokens", sa.Integer),
        sa.Column("qdrant_point_id", UUID(as_uuid=True)),
        sa.Column("embedded_at", sa.DateTime(timezone=True)),
        sa.Column("embedding_model", sa.Text),
        sa.Column("char_start", sa.Integer),
        sa.Column("char_end", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_chunks_document", "chunks", ["document_id"])
    op.create_index("idx_chunks_qdrant_id", "chunks", ["qdrant_point_id"], postgresql_where=sa.text("qdrant_point_id IS NOT NULL"))
    op.create_index("idx_chunks_section_type", "chunks", ["section_type"])

    op.create_table(
        "citations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("citing_document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("cited_doi", sa.Text),
        sa.Column("cited_title", sa.Text),
        sa.Column("cited_authors", JSONB),
        sa.Column("cited_year", sa.SmallInteger),
        sa.Column("cited_journal", sa.Text),
        sa.Column("cited_document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="SET NULL")),
        sa.Column("raw_reference_text", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_citations_citing", "citations", ["citing_document_id"])
    op.create_index("idx_citations_cited_doi", "citations", ["cited_doi"], postgresql_where=sa.text("cited_doi IS NOT NULL"))

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="SET NULL")),
        sa.Column("stage", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("details", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_jobs_document", "ingestion_jobs", ["document_id"])
    op.create_index("idx_jobs_created", "ingestion_jobs", ["created_at"])


def downgrade() -> None:
    op.drop_table("ingestion_jobs")
    op.drop_table("citations")
    op.drop_table("chunks")
    op.drop_table("paper_sections")
    op.execute("DROP TRIGGER IF EXISTS trig_update_search_vector ON paper_metadata")
    op.execute("DROP FUNCTION IF EXISTS update_search_vector()")
    op.drop_table("paper_metadata")
    op.drop_table("documents")
