"""Scrivener-inspired features: draft status, custom metadata, snapshots

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-07
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add status + custom_metadata to drafts
    op.add_column("drafts", sa.Column("status", sa.Text(), server_default="drafted"))
    op.add_column("drafts", sa.Column("custom_metadata", JSONB, server_default="{}"))

    # Snapshots table
    op.create_table(
        "draft_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("draft_id", UUID(as_uuid=True),
                  sa.ForeignKey("drafts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("word_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("idx_snapshots_draft", "draft_snapshots", ["draft_id"])


def downgrade() -> None:
    op.drop_table("draft_snapshots")
    op.drop_column("drafts", "custom_metadata")
    op.drop_column("drafts", "status")
