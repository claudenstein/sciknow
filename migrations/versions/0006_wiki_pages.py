"""Wiki pages — persistent compiled knowledge layer

Revision ID: 0006
Revises: 0005
Create Date: 2026-04-07
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, UUID

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "wiki_pages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("slug", sa.Text(), nullable=False, unique=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("page_type", sa.Text(), nullable=False),
        sa.Column("source_doc_ids", ARRAY(UUID(as_uuid=True)), nullable=True),
        sa.Column("word_count", sa.Integer(), nullable=True),
        sa.Column("needs_rewrite", sa.Text(), nullable=False, server_default="false"),
        sa.Column("qdrant_point_id", UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("idx_wiki_slug", "wiki_pages", ["slug"], unique=True)
    op.create_index("idx_wiki_type", "wiki_pages", ["page_type"])


def downgrade() -> None:
    op.drop_table("wiki_pages")
