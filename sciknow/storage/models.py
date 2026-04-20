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
    # Phase 54.6.80 (#10) — paper_type classification for retrieval filtering.
    # One of: peer_reviewed | preprint | thesis | editorial | opinion |
    # policy | book_chapter | unknown. NULL means not yet classified.
    paper_type: Mapped[str | None] = mapped_column(Text)
    paper_type_confidence: Mapped[float | None] = mapped_column(Float)
    paper_type_model: Mapped[str | None] = mapped_column(Text)

    # Raw API responses
    crossref_raw: Mapped[dict | None] = mapped_column(JSONB)
    arxiv_raw: Mapped[dict | None] = mapped_column(JSONB)

    # Phase 54.6.111 (Tier 1 #1) — persisted OpenAlex enrichment.
    # Hydrated from a single /works/{id} call at enrich time; see
    # docs/EXPAND_ENRICH_RESEARCH_2.md §1.1.
    oa_concepts: Mapped[list | None] = mapped_column(JSONB)           # [{display_name, level, score}]
    oa_funders: Mapped[list | None] = mapped_column(JSONB)            # [{name, id}]
    oa_grants: Mapped[list | None] = mapped_column(JSONB)             # [{funder, award_id}]
    oa_institutions_ror: Mapped[list | None] = mapped_column(JSONB)   # [ror_id]
    oa_counts_by_year: Mapped[list | None] = mapped_column(JSONB)     # [{year, cited_by_count}]
    oa_biblio: Mapped[dict | None] = mapped_column(JSONB)             # {volume, issue, first_page, last_page}
    oa_cited_by_count: Mapped[int | None] = mapped_column(Integer)
    oa_enriched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Phase 54.6.111 (Tier 1 #2) — retraction sweep bookkeeping.
    retraction_status: Mapped[str | None] = mapped_column(Text)       # none | retracted | corrected | withdrawn
    retraction_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

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
    # Phase 52 — stamp of the chunker version that parsed these
    # sections. Same motivation as Chunk.chunker_version.
    chunker_version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
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
    # Phase 52 — integer stamp of the chunker version that produced
    # this row. Compared against ingestion.chunker.CHUNKER_VERSION
    # to detect staleness when the chunker implementation changes.
    chunker_version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )

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
    # Phase 45 — project type: scientific_book (default, legacy) |
    # scientific_paper | (future: scifi_novel, literature_review, ...)
    # Drives section defaults, prompt conditioning, length targets,
    # export templates. Lives in books despite the name because books
    # is the project root record. See sciknow.core.project_type.
    book_type: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="scientific_book",
    )
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
    """Named snapshots of draft/chapter/book state.

    Phase 38 — scope extended beyond a single draft. Three shapes:

    - `scope='draft'` (original): `draft_id` set, `content` is raw text,
      restore overwrites the current draft content. Existing endpoints
      `/api/snapshot/{draft_id}` keep working unchanged.

    - `scope='chapter'`: `chapter_id` set, `draft_id` NULL, `content`
      is a JSON bundle `{"chapter_id": ..., "drafts": [{id, section_type,
      title, content, word_count, version}, ...]}`. Restore creates NEW
      draft VERSIONS for every section in the bundle — non-destructive.

    - `scope='book'`: `book_id` set, `draft_id` and `chapter_id` NULL,
      `content` is a JSON bundle containing every chapter bundle.
      Restore behaves like a chapter restore, per-chapter.

    Chapter/book snapshots are the safety net for the destructive
    `autowrite-all` op: one click snapshots a chapter's current state
    before letting autowrite loose on it.
    """
    __tablename__ = "draft_snapshots"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    # Nullable since Phase 38: chapter/book scope snapshots have no
    # single draft to point at.
    draft_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("drafts.id", ondelete="CASCADE"), nullable=True
    )
    chapter_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("book_chapters.id", ondelete="CASCADE"),
        nullable=True,
    )
    book_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("books.id", ondelete="CASCADE"),
        nullable=True,
    )
    # 'draft' | 'chapter' | 'book' — Phase 38 default preserves legacy
    # rows as draft-scoped without a backfill.
    scope: Mapped[str] = mapped_column(Text, nullable=False, server_default="draft")
    name: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_snapshots_draft", "draft_id"),
        # Phase 38 — targeted indexes for the new list endpoints.
        Index("idx_snapshots_chapter", "chapter_id",
              postgresql_where="chapter_id IS NOT NULL"),
        Index("idx_snapshots_book", "book_id",
              postgresql_where="book_id IS NOT NULL"),
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


# ── Phase 21.a — Visuals as first-class evidence ─────────────────────────

class Visual(Base):
    """A figure, table, equation, or code block extracted from a paper.

    Populated by ``sciknow db extract-visuals`` which walks each paper's
    ``content_list.json`` (MinerU output). Later phases (21.b–21.f) build
    retrieval, UI, and write-loop integration on top of this table.
    """
    __tablename__ = "visuals"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)  # table | equation | figure | chart | code
    content: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    caption: Mapped[str | None] = mapped_column(Text)
    asset_path: Mapped[str | None] = mapped_column(Text)
    block_idx: Mapped[int | None] = mapped_column(Integer)
    figure_num: Mapped[str | None] = mapped_column(Text)
    surrounding_text: Mapped[str | None] = mapped_column(Text)
    qdrant_point_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    # Phase 54.6.72 — vision-LLM caption for figures/charts (NULL for
    # text-only kinds like equation/table/code). Filled by
    # `sciknow db caption-visuals`. ai_caption_model records which
    # VLM produced it so re-runs with a better model can target the
    # stale rows.
    ai_caption: Mapped[str | None] = mapped_column(Text)
    ai_caption_model: Mapped[str | None] = mapped_column(Text)
    ai_captioned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_visuals_document", "document_id"),
        Index("idx_visuals_kind", "kind"),
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


class LLMUsageLog(Base):
    """Phase 35 — generic LLM usage ledger for GPU-compute accounting.

    One row per completed job originating from the web UI (write, review,
    revise, argue, gaps, autowrite, plan, etc.). Populated from
    `_run_generator_in_thread`'s finally block using the per-job counters
    maintained by `_observe_event_for_stats` (Phase 32.5).

    The autowrite-specific `autowrite_runs.tokens_used` column (Phase 33)
    is kept because it's wired into the autowrite telemetry subsystem;
    this table is the superset used for the book dashboard's Total
    Compute panel. Autowrite appears in both places by design — the
    per-operation breakdown on the dashboard reconciles the totals.

    A row is only inserted when tokens > 0 (skip zero-token no-ops / early
    errors so the table stays signal-dense).
    """
    __tablename__ = "llm_usage_log"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    book_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("books.id", ondelete="CASCADE"), nullable=True
    )
    chapter_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("book_chapters.id", ondelete="SET NULL"), nullable=True
    )
    # Job type — matches _create_job's job_type argument in web/app.py
    # (write | review | revise | argue | gaps | autowrite | plan | ...).
    operation: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str | None] = mapped_column(Text)
    tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    # completed | error | cancelled
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="completed")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_llm_usage_book", "book_id"),
        Index("idx_llm_usage_book_op", "book_id", "operation"),
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
    # Phase 48d — verbatim sentence from the source paper that
    # evidences the triple. Nullable: the extraction LLM doesn't
    # always pin a triple to one sentence, and pre-0019 rows have
    # NULL. Backfill via `wiki compile --rebuild`.
    source_sentence: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_kg_subject", "subject"),
        Index("idx_kg_object", "object"),
        Index("idx_kg_predicate", "predicate"),
        Index("idx_kg_source", "source_doc_id"),
    )


class Feedback(Base):
    """Phase 50.B — user feedback capture.

    Every answer the system produces (ask / write / review / autowrite
    section / KG edge click / whatever) can be rated. Fields are
    deliberately loose so new op types don't need schema changes —
    just insert a row with op='my_new_op' and the UI knows to show
    the thumbs-up/down next to it.

    Longer-term consumer: the parked LambdaMART learn-to-rank upgrade
    (see docs/EXPAND_RESEARCH.md). (query, chunk_ids, score) triples
    out of this table feed directly into LightGBM's pairwise training
    once ≥500 labeled positives accumulate."""

    __tablename__ = "feedback"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    # Which op produced the rated answer — e.g. 'ask', 'write',
    # 'autowrite', 'review', 'kg_edge'. Not an enum so new ops can
    # land without a migration.
    op: Mapped[str] = mapped_column(Text, nullable=False)
    query: Mapped[str | None] = mapped_column(Text)
    response_preview: Mapped[str | None] = mapped_column(Text)
    # -1 negative, 0 neutral, +1 positive. SmallInt keeps the column
    # semantically boolean-ish while leaving room for granular ratings.
    score: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)
    draft_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("drafts.id", ondelete="SET NULL"),
        nullable=True,
    )
    # chunk_ids is an array of the Qdrant point ids used to generate
    # the rated answer — this is the part LambdaMART consumes.
    chunk_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # Free-form op-specific payload (e.g. section_type for write, kg
    # triple id for kg_edge, model name). Keeps the core schema stable.
    extras: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_feedback_op", "op"),
        Index("idx_feedback_score", "score"),
        Index("idx_feedback_created_at", "created_at"),
    )


class PendingDownload(Base):
    """Phase 54.6.7 — papers the user selected for ingest but no OA PDF
    was found. The curator can:

    * come back later and retry (re-running the 6-source OA cascade
      because Unpaywall / Copernicus / Semantic Scholar sometimes
      surface a link that wasn't available weeks ago)
    * export to CSV and acquire manually (ILL, author email, library)
    * abandon with a note explaining why

    Upserted by DOI — a DOI already on file bumps ``attempt_count``
    and updates ``last_attempt_at`` / ``last_failure_reason`` rather
    than creating a duplicate row.

    NOT used for ingest failures (those land in ``data/failed/`` via
    the existing pipeline). This table is specifically for the
    "~50% of expand selections have no legal OA PDF" failure mode.
    """

    __tablename__ = "pending_downloads"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    # Canonical identifier — the download pipeline keys on this.
    # Unique so upserts are trivial.
    doi: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # Optional alternate identifier (arXiv preprints that lack a
    # journal DOI can still be recovered via arxiv_id).
    arxiv_id: Mapped[str | None] = mapped_column(Text)
    # Bibliographic metadata captured at selection time — we store
    # this on the row so the user can make informed decisions
    # (manual acquisition, email the author, etc.) without having
    # to re-fetch metadata later.
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    authors: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    year: Mapped[int | None] = mapped_column(Integer)
    # Which expand flow surfaced this candidate — ``expand`` /
    # ``expand-author`` / ``expand-cites`` / ``expand-topic`` /
    # ``expand-coauthors`` / ``auto-expand`` / ``download-dois``.
    source_method: Mapped[str | None] = mapped_column(Text)
    # The query / seed / author name the flow was given. Useful when
    # the same paper surfaces from multiple searches.
    source_query: Mapped[str | None] = mapped_column(Text)
    relevance_score: Mapped[float | None] = mapped_column(Float)
    # Retry bookkeeping.
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    last_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    # Human-readable reason — ``no_oa`` / ``http_500`` / ``timeout`` /
    # etc. Not an enum: the 6-source cascade adds new failure modes
    # faster than we want to migrate a CHECK constraint.
    last_failure_reason: Mapped[str | None] = mapped_column(Text)
    # ``pending`` = still want it, queued for retry
    # ``manual_acquired`` = user got it another way (ILL, sci-hub, etc.)
    # ``abandoned`` = decided it's not worth chasing
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(), onupdate=func.now(),
    )

    __table_args__ = (
        Index("idx_pending_downloads_status", "status"),
        Index("idx_pending_downloads_source_method", "source_method"),
        Index("idx_pending_downloads_created_at", "created_at"),
    )
