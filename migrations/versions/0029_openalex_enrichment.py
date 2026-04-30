"""Phase 54.6.111 (Tier 1 #1+#2) — OpenAlex-enrichment + retraction columns.

Revision ID: 0029
Revises: 0028
Create Date: 2026-04-19

Adds six JSONB columns + two scalar columns to ``paper_metadata`` so we
stop discarding OpenAlex fields we already fetch at enrich time, and
one new timestamp/status pair for periodic retraction sweeps
(``sciknow db refresh-retractions``).

Rationale: see ``docs/research/EXPAND_ENRICH_RESEARCH_2.md`` §1.1-1.2.

All nullable — existing rows stay as-is until the next enrich run
populates them. The enricher is responsible for keeping them in sync
with whatever OpenAlex currently returns; stale rows are OK.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg


revision: str = "0029"
down_revision: Union[str, None] = "0028"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("paper_metadata") as batch:
        # OpenAlex extras
        batch.add_column(sa.Column("oa_concepts", pg.JSONB(), nullable=True))
        batch.add_column(sa.Column("oa_funders", pg.JSONB(), nullable=True))
        batch.add_column(sa.Column("oa_grants", pg.JSONB(), nullable=True))
        batch.add_column(sa.Column("oa_institutions_ror", pg.JSONB(), nullable=True))
        batch.add_column(sa.Column("oa_counts_by_year", pg.JSONB(), nullable=True))
        batch.add_column(sa.Column("oa_biblio", pg.JSONB(), nullable=True))
        batch.add_column(sa.Column("oa_cited_by_count", sa.Integer(), nullable=True))
        batch.add_column(sa.Column(
            "oa_enriched_at", sa.DateTime(timezone=True), nullable=True
        ))
        # Retraction tracking
        batch.add_column(sa.Column(
            "retraction_status", sa.Text(), nullable=True,
            comment="none | retracted | corrected | withdrawn. NULL = unchecked."
        ))
        batch.add_column(sa.Column(
            "retraction_checked_at", sa.DateTime(timezone=True), nullable=True
        ))

    # Partial indexes for the common filters
    op.create_index(
        "idx_paper_metadata_retracted",
        "paper_metadata", ["retraction_status"],
        postgresql_where=sa.text("retraction_status IS NOT NULL AND retraction_status <> 'none'"),
    )


def downgrade() -> None:
    op.drop_index("idx_paper_metadata_retracted", table_name="paper_metadata")
    with op.batch_alter_table("paper_metadata") as batch:
        batch.drop_column("retraction_checked_at")
        batch.drop_column("retraction_status")
        batch.drop_column("oa_enriched_at")
        batch.drop_column("oa_cited_by_count")
        batch.drop_column("oa_biblio")
        batch.drop_column("oa_counts_by_year")
        batch.drop_column("oa_institutions_ror")
        batch.drop_column("oa_grants")
        batch.drop_column("oa_funders")
        batch.drop_column("oa_concepts")
