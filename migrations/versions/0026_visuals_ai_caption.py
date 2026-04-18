"""Phase 54.6.72 (#1) — ai_caption column on visuals for VLM descriptions.

Revision ID: 0026
Revises: 0025
Create Date: 2026-04-18

Stores the vision-LLM-generated description for figures and charts,
populated by ``sciknow db caption-visuals``. Enables semantic retrieval
over the 9,988 MinerU-extracted images that were previously silent:
the GUI's Visuals tab shows real descriptions and the wiki Summaries
can include figures by caption match.

``ai_caption`` is nullable because (a) not every visual is image-kind
(tables/equations/code stay NULL), and (b) the captioning pass is
opt-in — missing ``ai_caption`` just means "not yet captioned",
letting the backfill run incrementally.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0026"
down_revision: Union[str, None] = "0025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("visuals") as batch:
        batch.add_column(sa.Column("ai_caption", sa.Text(), nullable=True))
        batch.add_column(sa.Column("ai_caption_model", sa.Text(), nullable=True))
        batch.add_column(sa.Column(
            "ai_captioned_at", sa.DateTime(timezone=True), nullable=True
        ))


def downgrade() -> None:
    with op.batch_alter_table("visuals") as batch:
        batch.drop_column("ai_captioned_at")
        batch.drop_column("ai_caption_model")
        batch.drop_column("ai_caption")
