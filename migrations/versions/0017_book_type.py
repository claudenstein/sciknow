"""Add book_type column to books (Phase 45 — project types).

Revision ID: 0017
Revises: 0016
Create Date: 2026-04-13

Phase 45 — introduces a typed project model. The ``books`` table
(despite the name) is the root record for any long-form writing
project sciknow supports. Types shipped in Phase 45:

- ``scientific_book``  (default, preserves pre-Phase-45 behaviour)
- ``scientific_paper`` (IMRaD-style single-document project)

The type is persisted so every downstream feature — section defaults,
autowrite prompts, length targets, export templates, review rubrics —
can branch on it. Future types (``scifi_novel`` etc.) drop in without
a schema change.

Backfill policy: existing rows become ``scientific_book`` since that's
what they were authored under. A new NOT NULL column with
``server_default`` gives us the backfill for free.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: Union[str, None] = "0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "books",
        sa.Column(
            "book_type", sa.Text(), nullable=False,
            server_default="scientific_book",
        ),
    )
    # A CHECK would force a migration every time we add a type.
    # Validation lives in sciknow.core.project_type.PROJECT_TYPES.


def downgrade() -> None:
    op.drop_column("books", "book_type")
