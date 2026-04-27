"""Phase 54.6.328 — diff-brief column on draft_snapshots.

Adds a ``meta JSONB`` column to ``draft_snapshots`` so each snapshot
row can carry the precomputed prose / structural diff brief produced
by ``sciknow.core.snapshot_diff``. The column defaults to an empty
JSON object so existing rows render with a placeholder until they're
recomputed (or replaced by a fresh snapshot).

Revision ID: 0041
Revises:     0040
Create Date: 2026-04-26
"""
from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0041"
down_revision: Union[str, None] = "0040"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.add_column(
        "draft_snapshots",
        sa.Column(
            "meta",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("draft_snapshots", "meta")
