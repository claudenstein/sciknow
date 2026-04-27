"""Phase 54.6.x — per-chapter flexible-length opt-in.

Revision ID: 0042
Revises: 0041
Create Date: 2026-04-27

Adds ``book_chapters.flexible_length`` (boolean, default FALSE). When
TRUE, the autowrite length resolver is permitted to grow the
chapter's effective target up to 2× the configured target_words IF
the corpus retrieval pool for that chapter is rich enough to support
deeper coverage. The growth is one-directional — a flexible chapter
never shrinks below its target, only ever above it.

Why a column and not custom_metadata: book_chapters has no
custom_metadata JSONB today, and a single nullable bool is cheaper
than introducing one for one flag.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0042"
down_revision: Union[str, None] = "0041"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("book_chapters") as batch:
        batch.add_column(
            sa.Column(
                "flexible_length",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("FALSE"),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("book_chapters") as batch:
        batch.drop_column("flexible_length")
