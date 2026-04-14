"""Phase 50.B — user feedback capture (LambdaMART feedstock).

Revision ID: 0020
Revises: 0019
Create Date: 2026-04-14

Stores a thumbs-up / thumbs-down + optional comment against any
answer the system produces (a book-draft section, a `sciknow ask`
response, a `sciknow book review` report, etc.). The fields are
deliberately loose so any op can log into it without a per-op
schema change. The expected longer-term consumer is the parked
LambdaMART learn-to-rank upgrade described in
``docs/EXPAND_RESEARCH.md`` — (query, chunk_ids, score) triples are
exactly its training format.

``score`` is a signed small int rather than float so it's easy to
reason about: -1 negative, 0 neutral, +1 positive. More granular
ratings can go in ``extras`` without reshaping the column.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0020"
down_revision: Union[str, None] = "0019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("op", sa.Text(), nullable=False),
        sa.Column("query", sa.Text(), nullable=True),
        sa.Column("response_preview", sa.Text(), nullable=True),
        sa.Column("score", sa.SmallInteger(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("draft_id", UUID(as_uuid=True),
                  sa.ForeignKey("drafts.id", ondelete="SET NULL"), nullable=True),
        sa.Column("chunk_ids", JSONB(), nullable=False, server_default="[]"),
        sa.Column("extras", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("idx_feedback_op", "feedback", ["op"])
    op.create_index("idx_feedback_score", "feedback", ["score"])
    op.create_index("idx_feedback_created_at", "feedback", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_feedback_created_at", "feedback")
    op.drop_index("idx_feedback_score", "feedback")
    op.drop_index("idx_feedback_op", "feedback")
    op.drop_table("feedback")
