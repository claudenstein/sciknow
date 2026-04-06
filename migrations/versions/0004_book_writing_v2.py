"""Book writing v2: plan, summaries, gaps, parent_draft

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-05
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Book plan (thesis statement) ──────────────────────────────────────────
    op.add_column("books", sa.Column("plan", sa.Text(), nullable=True))

    # ── Draft summaries + revision chain ──────────────────────────────────────
    op.add_column("drafts", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column("drafts", sa.Column(
        "parent_draft_id",
        UUID(as_uuid=True),
        sa.ForeignKey("drafts.id", ondelete="SET NULL"),
        nullable=True,
    ))
    op.add_column("drafts", sa.Column("review_feedback", sa.Text(), nullable=True))

    # ── Book gaps tracking ────────────────────────────────────────────────────
    op.create_table(
        "book_gaps",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("book_id", UUID(as_uuid=True),
                  sa.ForeignKey("books.id", ondelete="CASCADE"), nullable=False),
        sa.Column("gap_type", sa.Text(), nullable=False),  # topic / evidence / argument / draft
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("chapter_id", UUID(as_uuid=True),
                  sa.ForeignKey("book_chapters.id", ondelete="SET NULL"), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="open"),
        sa.Column("resolved_draft_id", UUID(as_uuid=True),
                  sa.ForeignKey("drafts.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
    )
    op.create_index("idx_gaps_book", "book_gaps", ["book_id"])


def downgrade() -> None:
    op.drop_table("book_gaps")
    op.drop_column("drafts", "review_feedback")
    op.drop_column("drafts", "parent_draft_id")
    op.drop_column("drafts", "summary")
    op.drop_column("books", "plan")
