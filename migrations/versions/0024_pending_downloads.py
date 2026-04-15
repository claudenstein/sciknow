"""Phase 54.6.7 — pending_downloads table.

Revision ID: 0024
Revises: 0023
Create Date: 2026-04-14

Papers the user selected via any expand flow that couldn't be
auto-downloaded (the 6-source OA cascade returned no PDF). Keyed on
DOI with UNIQUE constraint so hooks can upsert safely. See
``sciknow/storage/models.py:PendingDownload`` for the model's docstring
and column-by-column rationale.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0024"
down_revision: Union[str, None] = "0023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "pending_downloads",
        sa.Column("id", UUID(as_uuid=True),
                  server_default=sa.text("gen_random_uuid()"),
                  primary_key=True),
        sa.Column("doi", sa.Text(), nullable=False, unique=True),
        sa.Column("arxiv_id", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False, server_default=""),
        sa.Column("authors", JSONB(), nullable=False,
                  server_default=sa.text("'[]'::jsonb")),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("source_method", sa.Text(), nullable=True),
        sa.Column("source_query", sa.Text(), nullable=True),
        sa.Column("relevance_score", sa.Float(), nullable=True),
        sa.Column("attempt_count", sa.Integer(),
                  nullable=False, server_default="1"),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_failure_reason", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(),
                  nullable=False, server_default="pending"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("idx_pending_downloads_status",
                    "pending_downloads", ["status"])
    op.create_index("idx_pending_downloads_source_method",
                    "pending_downloads", ["source_method"])
    op.create_index("idx_pending_downloads_created_at",
                    "pending_downloads", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_pending_downloads_created_at",
                  table_name="pending_downloads")
    op.drop_index("idx_pending_downloads_source_method",
                  table_name="pending_downloads")
    op.drop_index("idx_pending_downloads_status",
                  table_name="pending_downloads")
    op.drop_table("pending_downloads")
