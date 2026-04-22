"""Phase 54.6.221 — roadmap 3.2.4: per-paper institutions table.

Revision ID: 0038
Revises: 0037
Create Date: 2026-04-22

OpenAlex returns ``authorships[].institutions[]`` with ROR IDs,
display names, country codes, and institution types. Phase 54.6.111
persists only the unique ROR IDs onto ``paper_metadata.oa_institutions_ror``
— that's enough for "does this paper involve NOAA?" but not for
"show me climate papers from NOAA over the last 10 years" (which
needs display_name for filtering + display).

This migration adds a first-class ``paper_institutions`` table so
institution-level queries become cheap index scans instead of
JSONB extraction across every row. Schema:

  * ``paper_institutions(id, document_id, ror_id, display_name,
                         country_code, institution_type,
                         author_position, created_at)``
  * Index on (document_id) for per-paper joins.
  * Index on (lower(display_name)) for case-insensitive
    "papers from NOAA" queries.
  * Index on (ror_id) for ROR-based joins.

Populated at enrich time via the Phase 54.6.221 update to
``apply_openalex_enrichment``; backfill via
``sciknow db backfill-institutions`` which re-queries OpenAlex by
DOI for any paper with a DOI but no institutions row.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0038"
down_revision: Union[str, None] = "0037"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "paper_institutions",
        sa.Column("id", sa.BigInteger(), primary_key=True,
                  autoincrement=True),
        sa.Column(
            "document_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ror_id", sa.Text(), nullable=True),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("country_code", sa.Text(), nullable=True),
        sa.Column("institution_type", sa.Text(), nullable=True),
        sa.Column("author_position", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            nullable=False, server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "idx_paper_institutions_document",
        "paper_institutions", ["document_id"],
    )
    op.create_index(
        "idx_paper_institutions_ror",
        "paper_institutions", ["ror_id"],
    )
    op.create_index(
        "idx_paper_institutions_display_name_ci",
        "paper_institutions", [sa.text("lower(display_name)")],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_paper_institutions_display_name_ci",
        table_name="paper_institutions",
    )
    op.drop_index(
        "idx_paper_institutions_ror", table_name="paper_institutions",
    )
    op.drop_index(
        "idx_paper_institutions_document",
        table_name="paper_institutions",
    )
    op.drop_table("paper_institutions")
