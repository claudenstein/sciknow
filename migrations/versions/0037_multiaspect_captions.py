"""Phase 54.6.214 — multi-aspect captions (closes roadmap 3.5.2).

Revision ID: 0037
Revises: 0036
Create Date: 2026-04-22

Phase 5 of roadmap 3.1.6 (MinerU 2.5-Pro migration). Schema-only
drop that adds two nullable TEXT columns to ``visuals``:

  * ``literal_caption`` — what the image literally shows (axes,
    labels, data shape). Populated at extract-visuals time from
    MinerU 2.5-Pro's structured per-figure output, when available.
    Stays NULL for pipeline-era rows and for visual kinds MinerU-
    Pro doesn't analyse (tables, equations, code).
  * ``query_caption`` — a short, keyword-dense paraphrase of
    ``ai_caption`` optimised for retrieval. Population path
    deferred to a follow-on task after Phase 4 re-ingest
    completes — populating all 9k+ figures now would burn VLM
    time ahead of a re-ingest that will produce the source
    ai_caption anyway.

Both are nullable so this migration is safe to apply before Phase 4
runs — the columns just stay empty until the data exists. Downstream
consumers (embed-visuals, the writer's visual-grounding path) must
graceful-degrade to ``ai_caption`` when either field is NULL.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0037"
down_revision: Union[str, None] = "0036"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("visuals") as batch:
        batch.add_column(
            sa.Column("literal_caption", sa.Text(), nullable=True)
        )
        batch.add_column(
            sa.Column("query_caption", sa.Text(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("visuals") as batch:
        batch.drop_column("query_caption")
        batch.drop_column("literal_caption")
