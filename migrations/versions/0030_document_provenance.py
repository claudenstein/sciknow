"""Phase 54.6.117 (Tier 4 #1) — per-document provenance JSONB.

Revision ID: 0030
Revises: 0029
Create Date: 2026-04-19

Adds ``documents.provenance`` JSONB so every paper carries a structured
record of *why* it entered the corpus: which source (ingest / expand /
oeuvre / agentic), which round, which signals won, which seeds pointed
at it. Complements the existing ``ingest_source`` scalar (kept for the
partial-index fast path) with the rich context the research doc §4.2
called out.

Nullable — existing rows stay as-is. The ranker hooks in 54.6.117
write forward; a future ``db backfill-provenance`` command can
approximate historical records from ``data/downloads/expand.log``.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg


revision: str = "0030"
down_revision: Union[str, None] = "0029"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("documents") as batch:
        batch.add_column(sa.Column("provenance", pg.JSONB(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("documents") as batch:
        batch.drop_column("provenance")
