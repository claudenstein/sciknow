"""Phase 54.6.106 (#2) — structured-table parsing columns on visuals.

Revision ID: 0028
Revises: 0027
Create Date: 2026-04-19

Adds parsed-table metadata so the 1,501 MinerU HTML tables become
queryable beyond substring matches:

* ``table_title`` — the table's caption / title as inferred by the
  parser LLM (may duplicate ``caption`` when MinerU already captured it,
  but is set unconditionally by the parser so downstream consumers have
  a canonical field).
* ``table_headers`` — JSONB array of column header strings.
* ``table_summary`` — one-paragraph semantic summary (e.g.
  "Climate-proxy reconstruction sites with latitude, sample
  resolution, season bias, and source references for 12 sites"). Used
  for retrieval + the Visuals modal card subtitle.
* ``table_n_rows`` / ``table_n_cols`` — basic shape so the UI can
  label large tables differently from small ones.
* ``table_parsed_at`` — timestamp of the last parse; the CLI skips
  rows that already have it set unless ``--force``.

All nullable — parsing is opt-in via ``sciknow db parse-tables``
and rows without a parse simply render as MinerU HTML (the existing
behaviour).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg


revision: str = "0028"
down_revision: Union[str, None] = "0027"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("visuals") as batch:
        batch.add_column(sa.Column("table_title", sa.Text(), nullable=True))
        batch.add_column(sa.Column("table_headers", pg.JSONB(), nullable=True))
        batch.add_column(sa.Column("table_summary", sa.Text(), nullable=True))
        batch.add_column(sa.Column("table_n_rows", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("table_n_cols", sa.Integer(), nullable=True))
        batch.add_column(sa.Column(
            "table_parsed_at", sa.DateTime(timezone=True), nullable=True
        ))


def downgrade() -> None:
    with op.batch_alter_table("visuals") as batch:
        batch.drop_column("table_parsed_at")
        batch.drop_column("table_n_cols")
        batch.drop_column("table_n_rows")
        batch.drop_column("table_summary")
        batch.drop_column("table_headers")
        batch.drop_column("table_title")
