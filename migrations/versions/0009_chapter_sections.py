"""Per-chapter section list (replaces hardcoded paper-style sections)

Revision ID: 0009
Revises: 0008
Create Date: 2026-04-07
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("book_chapters", sa.Column("sections", JSONB, server_default="[]"))


def downgrade() -> None:
    op.drop_column("book_chapters", "sections")
