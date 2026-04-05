from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    file_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    original_path: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    ingestion_status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    ingestion_error: Mapped[str | None] = mapped_column(Text)
    mineru_output_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    paper_metadata: Mapped["PaperMetadata | None"] = relationship(back_populates="document")
    sections: Mapped[list["PaperSection"]] = relationship(back_populates="document")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")
    citations: Mapped[list["Citation"]] = relationship(
        back_populates="citing_document", foreign_keys="Citation.citing_document_id"
    )

    __table_args__ = (
        Index("idx_documents_status", "ingestion_status"),
        Index("idx_documents_hash", "file_hash"),
    )


class PaperMetadata(Base):
    __tablename__ = "paper_metadata"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    # Core bibliographic fields
    title: Mapped[str | None] = mapped_column(Text)
    abstract: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(SmallInteger)
    doi: Mapped[str | None] = mapped_column(Text)
    arxiv_id: Mapped[str | None] = mapped_column(Text)
    journal: Mapped[str | None] = mapped_column(Text)
    volume: Mapped[str | None] = mapped_column(Text)
    issue: Mapped[str | None] = mapped_column(Text)
    pages: Mapped[str | None] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(Text)

    # Authors: [{"name": "...", "orcid": "...", "affiliation": "..."}]
    authors: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Tags
    keywords: Mapped[list | None] = mapped_column(ARRAY(Text))
    domains: Mapped[list | None] = mapped_column(ARRAY(Text))
    topic_cluster: Mapped[str | None] = mapped_column(Text)

    # Source tracking
    metadata_source: Mapped[str] = mapped_column(Text, nullable=False, default="unknown")

    # Raw API responses
    crossref_raw: Mapped[dict | None] = mapped_column(JSONB)
    arxiv_raw: Mapped[dict | None] = mapped_column(JSONB)

    # Full-text search vector (maintained by DB trigger)
    search_vector: Mapped[str | None] = mapped_column(TSVECTOR)

    # Overflow
    extra: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    document: Mapped["Document"] = relationship(back_populates="paper_metadata")

    __table_args__ = (
        Index("idx_metadata_doi", "doi", postgresql_where="doi IS NOT NULL"),
        Index("idx_metadata_arxiv", "arxiv_id", postgresql_where="arxiv_id IS NOT NULL"),
        Index("idx_metadata_year", "year", postgresql_where="year IS NOT NULL"),
        Index("idx_metadata_domains", "domains", postgresql_using="gin"),
        Index("idx_metadata_keywords", "keywords", postgresql_using="gin"),
        Index("idx_metadata_fts", "search_vector", postgresql_using="gin"),
    )


class PaperSection(Base):
    __tablename__ = "paper_sections"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    section_type: Mapped[str] = mapped_column(Text, nullable=False)
    section_title: Mapped[str | None] = mapped_column(Text)
    section_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped["Document"] = relationship(back_populates="sections")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="section")

    __table_args__ = (
        Index("idx_sections_document", "document_id"),
        Index("idx_sections_type", "section_type"),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    section_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("paper_sections.id", ondelete="SET NULL")
    )

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section_type: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_tokens: Mapped[int | None] = mapped_column(Integer)

    # Qdrant reference
    qdrant_point_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    embedding_model: Mapped[str | None] = mapped_column(Text)

    # Character offsets in original markdown
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")
    section: Mapped["PaperSection | None"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("idx_chunks_document", "document_id"),
        Index(
            "idx_chunks_qdrant_id",
            "qdrant_point_id",
            postgresql_where="qdrant_point_id IS NOT NULL",
        ),
        Index("idx_chunks_section_type", "section_type"),
    )


class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    citing_document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    cited_doi: Mapped[str | None] = mapped_column(Text)
    cited_title: Mapped[str | None] = mapped_column(Text)
    cited_authors: Mapped[list | None] = mapped_column(JSONB)
    cited_year: Mapped[int | None] = mapped_column(SmallInteger)
    cited_journal: Mapped[str | None] = mapped_column(Text)
    cited_document_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL")
    )
    raw_reference_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    citing_document: Mapped["Document"] = relationship(
        back_populates="citations", foreign_keys=[citing_document_id]
    )

    __table_args__ = (
        Index("idx_citations_citing", "citing_document_id"),
        Index(
            "idx_citations_cited_doi",
            "cited_doi",
            postgresql_where="cited_doi IS NOT NULL",
        ),
    )


class Book(Base):
    __tablename__ = "books"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="draft")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    chapters: Mapped[list["BookChapter"]] = relationship(
        back_populates="book", order_by="BookChapter.number", cascade="all, delete-orphan"
    )
    drafts: Mapped[list["Draft"]] = relationship(back_populates="book")


class BookChapter(Base):
    __tablename__ = "book_chapters"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    book_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("books.id", ondelete="CASCADE"), nullable=False
    )
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    topic_query: Mapped[str | None] = mapped_column(Text)
    topic_cluster: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    book: Mapped["Book"] = relationship(back_populates="chapters")
    drafts: Mapped[list["Draft"]] = relationship(back_populates="chapter")

    __table_args__ = (
        Index("idx_chapters_book", "book_id"),
    )


class Draft(Base):
    __tablename__ = "drafts"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    book_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("books.id", ondelete="SET NULL"), nullable=True
    )
    chapter_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("book_chapters.id", ondelete="SET NULL"), nullable=True
    )
    section_type: Mapped[str | None] = mapped_column(Text)
    topic: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int | None] = mapped_column(Integer)
    sources: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    model_used: Mapped[str | None] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    book: Mapped["Book | None"] = relationship(back_populates="drafts")
    chapter: Mapped["BookChapter | None"] = relationship(back_populates="drafts")

    __table_args__ = (
        Index("idx_drafts_book", "book_id", postgresql_where="book_id IS NOT NULL"),
        Index("idx_drafts_chapter", "chapter_id", postgresql_where="chapter_id IS NOT NULL"),
    )


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL")
    )
    stage: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    details: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_jobs_document", "document_id"),
        Index("idx_jobs_created", "created_at"),
    )
