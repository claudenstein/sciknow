"""Phase 35 — generic LLM usage ledger.

Revision ID: 0015
Revises: 0014
Create Date: 2026-04-12

Adds `llm_usage_log` — a thin, per-job row written from
`_run_generator_in_thread`'s finally block so the book dashboard can
show cumulative GPU compute (tokens, wall time) across every LLM-backed
operation, not just autowrite. See sciknow/storage/models.py:LLMUsageLog
for the rationale and invariants.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "0015"
down_revision: Union[str, None] = "0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_usage_log",
        sa.Column(
            "id", PG_UUID(as_uuid=True), primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "book_id", PG_UUID(as_uuid=True),
            sa.ForeignKey("books.id", ondelete="CASCADE"), nullable=True,
        ),
        sa.Column(
            "chapter_id", PG_UUID(as_uuid=True),
            sa.ForeignKey("book_chapters.id", ondelete="SET NULL"), nullable=True,
        ),
        sa.Column("operation", sa.Text, nullable=False),
        sa.Column("model_name", sa.Text, nullable=True),
        sa.Column("tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("duration_seconds", sa.Float, nullable=True),
        sa.Column("status", sa.Text, nullable=False, server_default="completed"),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_llm_usage_book", "llm_usage_log", ["book_id"])
    op.create_index("idx_llm_usage_book_op", "llm_usage_log", ["book_id", "operation"])


def downgrade() -> None:
    op.drop_index("idx_llm_usage_book_op", table_name="llm_usage_log")
    op.drop_index("idx_llm_usage_book", table_name="llm_usage_log")
    op.drop_table("llm_usage_log")
