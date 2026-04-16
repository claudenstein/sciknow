"""Phase 21.a — visuals table for first-class figure/table/equation tracking.

Revision ID: 0025
Revises: 0024
Create Date: 2026-04-16

Structured catalog of every visual element (figure, table, equation)
extracted from ingested papers. Populated by ``sciknow db extract-visuals``
which walks each paper's ``content_list.json`` (MinerU output). Later
phases (21.b–21.f) build retrieval, UI, and write-loop integration on
top of this table.

See docs/RESEARCH.md §21 for the full design.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0025"
down_revision: Union[str, None] = "0024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "visuals",
        sa.Column("id", UUID(as_uuid=True),
                  server_default=sa.text("gen_random_uuid()"),
                  primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"),
                  nullable=False),
        # kind: 'table' | 'equation' | 'figure' | 'code'
        sa.Column("kind", sa.Text(), nullable=False),
        # For tables: HTML table_body; for equations: LaTeX string;
        # for figures: image caption; for code: code body
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        # Human-readable caption (from MinerU's caption field or
        # surrounding_paragraph for figures without explicit captions)
        sa.Column("caption", sa.Text(), nullable=True),
        # Path to the image asset (figures only), relative to data dir
        sa.Column("asset_path", sa.Text(), nullable=True),
        # Position in the source PDF (block index in content_list.json)
        sa.Column("block_idx", sa.Integer(), nullable=True),
        # Extracted figure/table number (e.g. "Figure 3", "Table 1")
        sa.Column("figure_num", sa.Text(), nullable=True),
        # Text from the paragraph immediately before/after the visual
        sa.Column("surrounding_text", sa.Text(), nullable=True),
        # Qdrant point ID (populated by Phase 21.b caption embedding)
        sa.Column("qdrant_point_id", UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_visuals_document", "visuals", ["document_id"])
    op.create_index("idx_visuals_kind", "visuals", ["kind"])


def downgrade() -> None:
    op.drop_table("visuals")
