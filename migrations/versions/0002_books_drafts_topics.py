"""Books, drafts, topic clustering

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-04
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Topic cluster column on paper_metadata ─────────────────────────────────
    op.add_column("paper_metadata", sa.Column("topic_cluster", sa.Text, nullable=True))
    op.create_index(
        "idx_metadata_topic_cluster",
        "paper_metadata",
        ["topic_cluster"],
        postgresql_where=sa.text("topic_cluster IS NOT NULL"),
    )

    # ── Books ──────────────────────────────────────────────────────────────────
    op.create_table(
        "books",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("title", sa.Text, nullable=False, unique=True),
        sa.Column("description", sa.Text),
        sa.Column("status", sa.Text, nullable=False, server_default="draft"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
    )

    # ── Book chapters ──────────────────────────────────────────────────────────
    op.create_table(
        "book_chapters",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("book_id", UUID(as_uuid=True),
                  sa.ForeignKey("books.id", ondelete="CASCADE"), nullable=False),
        sa.Column("number", sa.SmallInteger, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("topic_query", sa.Text),      # search query for this chapter's papers
        sa.Column("topic_cluster", sa.Text),    # optional cluster filter
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
    )
    op.create_index("idx_chapters_book", "book_chapters", ["book_id"])
    op.create_unique_constraint("uq_chapter_number", "book_chapters", ["book_id", "number"])

    # ── Drafts ─────────────────────────────────────────────────────────────────
    op.create_table(
        "drafts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("book_id", UUID(as_uuid=True),
                  sa.ForeignKey("books.id", ondelete="SET NULL"), nullable=True),
        sa.Column("chapter_id", UUID(as_uuid=True),
                  sa.ForeignKey("book_chapters.id", ondelete="SET NULL"), nullable=True),
        sa.Column("section_type", sa.Text),
        sa.Column("topic", sa.Text),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("word_count", sa.Integer),
        sa.Column("sources", JSONB, nullable=False, server_default="[]"),
        sa.Column("model_used", sa.Text),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
    )
    op.create_index("idx_drafts_book", "drafts", ["book_id"],
                    postgresql_where=sa.text("book_id IS NOT NULL"))
    op.create_index("idx_drafts_chapter", "drafts", ["chapter_id"],
                    postgresql_where=sa.text("chapter_id IS NOT NULL"))


def downgrade() -> None:
    op.drop_table("drafts")
    op.drop_constraint("uq_chapter_number", "book_chapters")
    op.drop_table("book_chapters")
    op.drop_table("books")
    op.drop_index("idx_metadata_topic_cluster", table_name="paper_metadata")
    op.drop_column("paper_metadata", "topic_cluster")
