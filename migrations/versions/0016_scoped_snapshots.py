"""Phase 38 — scoped snapshots (chapter + book bundles).

Revision ID: 0016
Revises: 0015
Create Date: 2026-04-12

Extends `draft_snapshots` so a single row can represent a chapter-wide
or book-wide snapshot, not just a single draft. See
sciknow/storage/models.py:DraftSnapshot for the invariants and the
content-column contract for each scope.

- Adds nullable `chapter_id` FK to `book_chapters.id` (CASCADE)
- Adds nullable `book_id` FK to `books.id` (CASCADE)
- Adds `scope` TEXT (default 'draft') — keeps every existing row valid
- Makes `draft_id` nullable — chapter/book rows have no single draft
- Adds two partial indexes for the new list endpoints
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "0016"
down_revision: Union[str, None] = "0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "draft_snapshots",
        sa.Column(
            "chapter_id", PG_UUID(as_uuid=True),
            sa.ForeignKey("book_chapters.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.add_column(
        "draft_snapshots",
        sa.Column(
            "book_id", PG_UUID(as_uuid=True),
            sa.ForeignKey("books.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.add_column(
        "draft_snapshots",
        sa.Column("scope", sa.Text, nullable=False, server_default="draft"),
    )
    op.alter_column("draft_snapshots", "draft_id", nullable=True)
    op.create_index(
        "idx_snapshots_chapter", "draft_snapshots", ["chapter_id"],
        postgresql_where=sa.text("chapter_id IS NOT NULL"),
    )
    op.create_index(
        "idx_snapshots_book", "draft_snapshots", ["book_id"],
        postgresql_where=sa.text("book_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_snapshots_book", table_name="draft_snapshots")
    op.drop_index("idx_snapshots_chapter", table_name="draft_snapshots")
    op.alter_column("draft_snapshots", "draft_id", nullable=False)
    op.drop_column("draft_snapshots", "scope")
    op.drop_column("draft_snapshots", "book_id")
    op.drop_column("draft_snapshots", "chapter_id")
