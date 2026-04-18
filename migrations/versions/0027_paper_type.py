"""Phase 54.6.80 (#10) — paper_type column on paper_metadata.

Revision ID: 0027
Revises: 0026
Create Date: 2026-04-18

Classifies each paper into {peer_reviewed, preprint, thesis, editorial,
opinion, policy, book_chapter, unknown}. Populated by the new
``sciknow db classify-papers`` CLI. Enables retrieval filtering (only
peer-reviewed for ask) and default downweight for opinion/policy on
factual queries.

Stored as free text (not an enum) so future categories can be added
without a migration. An index on the column keeps filter queries
fast even on a 10k-paper corpus.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0027"
down_revision: Union[str, None] = "0026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("paper_metadata") as batch:
        batch.add_column(sa.Column("paper_type", sa.Text(), nullable=True))
        batch.add_column(sa.Column("paper_type_confidence", sa.Float(), nullable=True))
        batch.add_column(sa.Column(
            "paper_type_model", sa.Text(), nullable=True,
        ))
    op.create_index(
        "idx_paper_metadata_paper_type",
        "paper_metadata",
        ["paper_type"],
        postgresql_where=sa.text("paper_type IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_paper_metadata_paper_type",
                  table_name="paper_metadata")
    with op.batch_alter_table("paper_metadata") as batch:
        batch.drop_column("paper_type_model")
        batch.drop_column("paper_type_confidence")
        batch.drop_column("paper_type")
