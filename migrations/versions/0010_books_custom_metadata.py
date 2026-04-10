"""Add custom_metadata JSONB to books (length targets, etc.)

Revision ID: 0010
Revises: 0009
Create Date: 2026-04-09

Phase 17 — adds a free-form JSONB column on books so we can persist
per-book length targets (target_chapter_words) and any future
book-level settings without a schema change each time.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "books",
        sa.Column("custom_metadata", JSONB, nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("books", "custom_metadata")
