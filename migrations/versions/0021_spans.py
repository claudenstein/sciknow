"""Phase 50.C — Langfuse-pattern span tracing (local SQLite-style).

Revision ID: 0021
Revises: 0020
Create Date: 2026-04-14

Single `spans` table for structured per-operation timing + payload
capture. Deliberately NOT a vendor-specific trace format (OTel / Jaeger
/ Langfuse) so we can keep the footprint local — the full value of
Langfuse / OTel at single-user scale is ~10% of the surface (span
start/end, parent link, metadata), which fits in ~500 LOC plus a
PG table.

Shape: a trace is a DAG of spans tied by `trace_id`; parent/child is
tracked via `parent_span_id`. Metadata is JSONB so we can store any
per-span payload (model name, tokens, retrieval query, …). Created
ordered by `started_at DESC` so the `sciknow spans tail` command's
default query is a simple indexed scan.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0021"
down_revision: Union[str, None] = "0020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "spans",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("trace_id", UUID(as_uuid=True), nullable=False),
        sa.Column("parent_span_id", UUID(as_uuid=True), nullable=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="ok"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("metadata_json", JSONB(), nullable=False, server_default="{}"),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index("idx_spans_trace", "spans", ["trace_id"])
    op.create_index("idx_spans_started_at", "spans", ["started_at"])
    op.create_index("idx_spans_name", "spans", ["name"])


def downgrade() -> None:
    op.drop_index("idx_spans_name", "spans")
    op.drop_index("idx_spans_started_at", "spans")
    op.drop_index("idx_spans_trace", "spans")
    op.drop_table("spans")
