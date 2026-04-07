"""Knowledge graph — entity-relationship triples

Revision ID: 0008
Revises: 0007
Create Date: 2026-04-07
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "knowledge_graph",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("predicate", sa.Text(), nullable=False),
        sa.Column("object", sa.Text(), nullable=False),
        sa.Column("source_doc_id", UUID(as_uuid=True),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=True),
        sa.Column("confidence", sa.Float(), server_default="1.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("idx_kg_subject", "knowledge_graph", ["subject"])
    op.create_index("idx_kg_object", "knowledge_graph", ["object"])
    op.create_index("idx_kg_predicate", "knowledge_graph", ["predicate"])
    op.create_index("idx_kg_source", "knowledge_graph", ["source_doc_id"])


def downgrade() -> None:
    op.drop_table("knowledge_graph")
