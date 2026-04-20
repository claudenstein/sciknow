"""Phase 54.6.138 — visuals.mention_paragraphs column.

Revision ID: 0033
Revises: 0032
Create Date: 2026-04-20

Adds ``visuals.mention_paragraphs jsonb`` to store the paragraphs in
the source paper's body text that reference this specific visual
(e.g. "… as shown in Fig. 3, the trend persists …"). Populated by
``sciknow db link-visual-mentions`` which scans the existing
``content_list.json`` — no re-ingestion needed.

Rationale (from docs/RESEARCH.md §7.X, signal 3):

SciCap+ (Yang et al., 2023) established that the **mention-paragraph**
— the author's own rhetorical framing of why the figure was cited at
that point — is a stronger retrieval signal for matching a figure to
target prose than either the caption or the image itself.
``Visual.surrounding_text`` already stores the IMMEDIATELY preceding
text block (Phase 21.a); this adds the scattered body references that
typically live in the Results / Discussion sections, nowhere near the
figure's own block.

JSONB shape: a list of objects, each
``{"block_idx": int, "text": str, "context_before": str | null}`` where
``context_before`` is the trimmed sentence preceding the match in the
same block, for the writer agent's "which section of the paper is
this mention in" prior.

Empty list (``[]``) means the link-pass ran and found no body-text
mentions of this visual's figure_num. NULL means the link-pass hasn't
run for this visual yet.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0033"
down_revision: Union[str, None] = "0032"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("visuals") as batch:
        batch.add_column(
            sa.Column(
                "mention_paragraphs",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=True,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("visuals") as batch:
        batch.drop_column("mention_paragraphs")
