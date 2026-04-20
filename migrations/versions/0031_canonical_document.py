"""Phase 54.6.125 (Tier 3 #3) — preprint ↔ journal reconciliation FK.

Revision ID: 0031
Revises: 0030
Create Date: 2026-04-20

Adds ``documents.canonical_document_id`` (nullable, FK to
documents.id). Set on the "non-canonical" row in a preprint+journal
pair; retrieval filters rows where this is NOT NULL, so they become
invisible without being deleted.

Fully reversible: ``sciknow db unreconcile <doc_id>`` clears the FK.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0031"
down_revision: Union[str, None] = "0030"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("documents") as batch:
        batch.add_column(
            sa.Column(
                "canonical_document_id",
                sa.dialects.postgresql.UUID(as_uuid=True),
                sa.ForeignKey("documents.id", ondelete="SET NULL"),
                nullable=True,
            )
        )
    op.create_index(
        "idx_documents_non_canonical",
        "documents",
        ["canonical_document_id"],
        postgresql_where=sa.text("canonical_document_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_documents_non_canonical", table_name="documents")
    with op.batch_alter_table("documents") as batch:
        batch.drop_column("canonical_document_id")
