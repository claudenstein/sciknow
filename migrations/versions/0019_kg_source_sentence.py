"""Phase 48d — source sentence provenance on knowledge_graph triples.

Revision ID: 0019
Revises: 0018
Create Date: 2026-04-14

Adds a nullable ``source_sentence`` column so each triple can carry the
verbatim sentence from its source paper that evidences the claim. The
column is populated by ``wiki compile`` going forward; pre-0019 triples
have NULL and surface as "(no source sentence)" in the KG modal's edge
right-click menu. To backfill the column for a previously-compiled
corpus, run ``sciknow wiki compile --rebuild``.

Kept nullable (not NOT NULL) on purpose — the extraction LLM may
legitimately fail to pin a triple to a single sentence, and the KG is
still useful without it.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0019"
down_revision: Union[str, None] = "0018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "knowledge_graph",
        sa.Column("source_sentence", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("knowledge_graph", "source_sentence")
