"""Phase 54.6.223 — roadmap 3.6.2: self-citation flagging.

Revision ID: 0039
Revises: 0038
Create Date: 2026-04-22

Flags each citation as self-referential when one or more authors of
the citing paper also appear in the cited paper's author list.
Enables consensus auditing ("is this claim supported by independent
work or only by the same group citing itself?") and feeds the
writer's groundedness / overstated-claim passes with the signal.

Schema:
  * ``citations.is_self_cite`` (boolean, nullable) — TRUE when the
    author sets overlap, FALSE when they don't, NULL when the
    classifier couldn't decide (author list missing on either side).
  * ``citations.self_cite_authors`` (JSONB, nullable) — list of
    normalised surname keys that appear on both sides, for audit.

Populated by ``sciknow db flag-self-citations`` which walks citations
with both citing and cited author lists and runs the overlap check
in pure Python (no LLM, no API).

Schema-only migration. The rows stay NULL until the CLI runs;
downstream consumers must graceful-degrade on NULL.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0039"
down_revision: Union[str, None] = "0038"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("citations") as batch:
        batch.add_column(
            sa.Column("is_self_cite", sa.Boolean(), nullable=True)
        )
        batch.add_column(
            sa.Column(
                "self_cite_authors",
                sa.dialects.postgresql.JSONB(),
                nullable=True,
            )
        )
    op.create_index(
        "idx_citations_self_cite",
        "citations", ["is_self_cite"],
        postgresql_where=sa.text("is_self_cite = true"),
    )


def downgrade() -> None:
    op.drop_index(
        "idx_citations_self_cite", table_name="citations",
    )
    with op.batch_alter_table("citations") as batch:
        batch.drop_column("self_cite_authors")
        batch.drop_column("is_self_cite")
