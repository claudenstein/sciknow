from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, REAL, TSVECTOR, UUID as PG_UUID
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
    # How this document entered the collection. 'seed' = manually ingested,
    # 'expand' = auto-discovered via `sciknow db expand`. Used for provenance
    # and to support future `db prune --source expand` operations.
    ingest_source: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="seed"
    )
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
    plan: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="draft")
    # Phase 17 — per-book settings that aren't worth a dedicated column.
    # Currently used for target_chapter_words (length target), but
    # intentionally kept open so we can drop new book-level knobs in
    # here without a migration each time. Shape:
    #   {"target_chapter_words": 6000, ...}
    custom_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
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
    sections: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")
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
    summary: Mapped[str | None] = mapped_column(Text)
    parent_draft_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("drafts.id", ondelete="SET NULL"), nullable=True
    )
    review_feedback: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="drafted")
    custom_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
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


class DraftSnapshot(Base):
    __tablename__ = "draft_snapshots"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    draft_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("drafts.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_snapshots_draft", "draft_id"),
    )


class DraftComment(Base):
    __tablename__ = "draft_comments"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    draft_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("drafts.id", ondelete="CASCADE"), nullable=False
    )
    paragraph_index: Mapped[int | None] = mapped_column(Integer)
    selected_text: Mapped[str | None] = mapped_column(Text)
    comment: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class BookGap(Base):
    __tablename__ = "book_gaps"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    book_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("books.id", ondelete="CASCADE"), nullable=False
    )
    gap_type: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    chapter_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("book_chapters.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="open")
    resolved_draft_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("drafts.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
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


class WikiPage(Base):
    __tablename__ = "wiki_pages"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    slug: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    page_type: Mapped[str] = mapped_column(Text, nullable=False)  # paper_summary | concept | synthesis
    source_doc_ids: Mapped[list | None] = mapped_column(ARRAY(PG_UUID(as_uuid=True)))
    word_count: Mapped[int | None] = mapped_column(Integer)
    needs_rewrite: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="false"
    )  # "true" / "false" as text (migration 0006)
    qdrant_point_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_wiki_slug", "slug", unique=True),
        Index("idx_wiki_type", "page_type"),
    )


# ── Phase 32.6 — Compound learning Layer 0: autowrite telemetry ─────────
#
# These three tables capture per-run / per-iteration / per-retrieval data
# that the autowrite loop produces but currently throws away. They are
# the data foundation for the Layer 1+ learning system documented in
# docs/RESEARCH.md §21 ("Compound learning from iteration history") and
# tracked in docs/ROADMAP.md.
#
# Layer 0 (this migration): just persist the data. The information is
# already in scope inside autowrite_section_stream — it's just not being
# captured in queryable form. Subsequent layers (lessons, useful_count
# retrieval boost, DPO preference dataset, heuristic distillation, style
# fingerprint) all read from these tables.
#
# Why three tables and not one wide JSONB column on `drafts`:
#   - JSONB on drafts hides the per-iteration structure from SQL (no
#     "show me all iterations where weakest_dimension was 'length'")
#   - cross-run aggregation ("which chunks were cited in >5 final
#     drafts across this book") would have to walk every drafts row
#   - JSONB doesn't compose with foreign keys for cascade deletes


class AutowriteRun(Base):
    """Phase 32.6 — one row per autowrite invocation."""
    __tablename__ = "autowrite_runs"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    book_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("books.id", ondelete="CASCADE"), nullable=True
    )
    chapter_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("book_chapters.id", ondelete="SET NULL"), nullable=True
    )
    section_slug: Mapped[str] = mapped_column(Text, nullable=False)
    final_draft_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("drafts.id", ondelete="SET NULL"), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # running | completed | error | cancelled
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="running")
    # Configuration captured at run start
    model: Mapped[str | None] = mapped_column(Text)
    target_words: Mapped[int | None] = mapped_column(Integer)
    max_iter: Mapped[int | None] = mapped_column(Integer)
    target_score: Mapped[float | None] = mapped_column(Float)
    feature_versions: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    # Final outcome (null until status=completed/error)
    final_overall: Mapped[float | None] = mapped_column(Float)
    iterations_used: Mapped[int | None] = mapped_column(Integer)
    converged: Mapped[bool | None] = mapped_column(Boolean)
    error_message: Mapped[str | None] = mapped_column(Text)
    # Phase 33 — cumulative token count from the _AutowriteLogger's
    # total_tokens counter. Set at finalization time so the dashboard
    # can aggregate LLM token usage across all autowrite runs.
    tokens_used: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        Index("idx_autowrite_runs_book", "book_id"),
        Index("idx_autowrite_runs_section", "book_id", "chapter_id", "section_slug"),
    )


class AutowriteIteration(Base):
    """Phase 32.6 — per-iteration record. Mirrors the existing
    drafts.custom_metadata.score_history shape but in queryable columns."""
    __tablename__ = "autowrite_iterations"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("autowrite_runs.id", ondelete="CASCADE"), nullable=False
    )
    iteration: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-indexed
    scores: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    verification: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    cove: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    # KEEP | DISCARD | NULL (unset until the rescore comparison happens)
    action: Mapped[str | None] = mapped_column(Text)
    word_count: Mapped[int | None] = mapped_column(Integer)
    word_count_delta: Mapped[int | None] = mapped_column(Integer)
    weakest_dimension: Mapped[str | None] = mapped_column(Text)
    revision_instruction: Mapped[str | None] = mapped_column(Text)
    overall_pre: Mapped[float | None] = mapped_column(Float)
    overall_post: Mapped[float | None] = mapped_column(Float)
    # Phase 32.9 — Layer 4: pre/post revision content for DPO pair
    # extraction. pre_revision_content is what the scorer scored at the
    # top of the iteration; post_revision_content is what the writer
    # produced after the revision (regardless of KEEP/DISCARD).
    # `sciknow book preferences export` walks these fields to build the
    # preference dataset.
    pre_revision_content: Mapped[str | None] = mapped_column(Text)
    post_revision_content: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("run_id", "iteration", name="uq_autowrite_iterations_run_iter"),
        Index("idx_autowrite_iterations_run", "run_id"),
    )


class AutowriteRetrieval(Base):
    """Phase 32.6 — one row per retrieved chunk per run.

    `chunk_qdrant_id` matches `chunks.qdrant_point_id` (NOT chunks.id —
    that's a different UUID). The retrieval pipeline keys everything by
    qdrant_point_id, so we use the same convention here for join consistency.

    `was_cited` is set in `_finalize_autowrite_run()` after the final draft
    is parsed for `[N]` markers. This boolean is the raw signal that powers
    Layer 2 (useful_count retrieval boost): a chunk that's been cited in
    many final drafts is likely to be useful for similar future sections.
    """
    __tablename__ = "autowrite_retrievals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("autowrite_runs.id", ondelete="CASCADE"), nullable=False
    )
    source_position: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-indexed [N] marker
    chunk_qdrant_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    document_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    rrf_score: Mapped[float | None] = mapped_column(Float)
    was_cited: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")

    __table_args__ = (
        Index("idx_autowrite_retrievals_run", "run_id"),
        Index(
            "idx_autowrite_retrievals_chunk_cited",
            "chunk_qdrant_id",
            postgresql_where="was_cited = true",
        ),
        Index("idx_autowrite_retrievals_doc", "document_id"),
    )


class AutowriteLesson(Base):
    """Phase 32.7 — Layer 1: episodic memory store.

    A 1-3-sentence lesson distilled from a single autowrite run, embedded
    via bge-m3 for similarity retrieval. Producer: `_distill_lessons_from_run`
    is called inline at the tail of `_finalize_autowrite_run` (Layer 0).
    Consumer: `_get_relevant_lessons` is called before `write_section_v2`
    in `_autowrite_section_body` and injects top-K lessons into the writer
    system prompt as a "Lessons from prior runs" block.

    The 1024-dim embedding is stored in PG as REAL[] rather than indexed
    in Qdrant because lesson tables stay small (~hundreds of rows per
    book at steady state) and all-pairs cosine similarity in Python is
    fast enough. If lesson counts ever exceed ~10k rows we'd switch to
    pgvector — but that's a migration, not a schema change.

    Ranking formula on read (Generative Agents 2023): the consumer applies
    `importance × recency_decay × cosine_similarity`, NOT a simple cosine.
    Recency decay is `exp(-age_days / 30)` so lessons stay relevant for
    about a month after distillation.
    """
    __tablename__ = "autowrite_lessons"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    book_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("books.id", ondelete="CASCADE"), nullable=True
    )
    chapter_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("book_chapters.id", ondelete="SET NULL"), nullable=True
    )
    section_slug: Mapped[str] = mapped_column(Text, nullable=False)
    lesson_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_run_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("autowrite_runs.id", ondelete="CASCADE"), nullable=True
    )
    score_delta: Mapped[float | None] = mapped_column(Float)
    # 1024-dim bge-m3 dense vector. May be NULL if embedding failed at
    # distillation time — the lesson stays in the table but is excluded
    # from similarity retrieval.
    embedding: Mapped[list[float] | None] = mapped_column(ARRAY(REAL))
    importance: Mapped[float] = mapped_column(Float, nullable=False, server_default="1.0")
    # groundedness | completeness | coherence | citation_accuracy |
    # hedging_fidelity | length | general
    dimension: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_autowrite_lessons_section", "book_id", "chapter_id", "section_slug"),
        Index("idx_autowrite_lessons_book", "book_id"),
        Index(
            "idx_autowrite_lessons_dimension",
            "dimension",
            postgresql_where="dimension IS NOT NULL",
        ),
    )


class KnowledgeGraphTriple(Base):
    __tablename__ = "knowledge_graph"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    predicate: Mapped[str] = mapped_column(Text, nullable=False)
    object: Mapped[str] = mapped_column(Text, nullable=False)
    source_doc_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True
    )
    confidence: Mapped[float] = mapped_column(nullable=False, server_default="1.0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_kg_subject", "subject"),
        Index("idx_kg_object", "object"),
        Index("idx_kg_predicate", "predicate"),
        Index("idx_kg_source", "source_doc_id"),
    )
