"""Phase 54.6.143 — per-chapter length target.

Revision ID: 0034
Revises: 0033
Create Date: 2026-04-20

Adds ``book_chapters.target_words`` (nullable integer). When set, it
overrides the book-level ``target_chapter_words`` for this specific
chapter — useful when a dense methods chapter needs different sizing
from a narrative introduction in the same book.

Resolution order in ``_get_book_length_target`` (Phase 54.6.143):

    1. per-chapter ``book_chapters.target_words``   (this migration)
    2. book-level ``custom_metadata.target_chapter_words``
    3. project_type ``default_target_chapter_words`` (Phase 54.6.143
       also wires `textbook` and `review_article` types)
    4. hardcoded ``DEFAULT_TARGET_CHAPTER_WORDS = 6000`` last resort

NULL on existing rows preserves the legacy behaviour (each chapter
uses the book-level target). Setting a value only affects autowrite
on that specific chapter.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0034"
down_revision: Union[str, None] = "0033"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("book_chapters") as batch:
        batch.add_column(
            sa.Column("target_words", sa.Integer(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("book_chapters") as batch:
        batch.drop_column("target_words")
