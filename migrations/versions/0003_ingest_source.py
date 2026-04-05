"""Ingest source (provenance) column on documents

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-05
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Default existing rows to 'seed' (manually-ingested). Future rows will
    # be set explicitly by the pipeline (seed | expand | restore | ...).
    op.add_column(
        "documents",
        sa.Column(
            "ingest_source",
            sa.Text(),
            nullable=False,
            server_default="seed",
        ),
    )
    op.create_index(
        "idx_documents_ingest_source",
        "documents",
        ["ingest_source"],
    )


def downgrade() -> None:
    op.drop_index("idx_documents_ingest_source", table_name="documents")
    op.drop_column("documents", "ingest_source")
