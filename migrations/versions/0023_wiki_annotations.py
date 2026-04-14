"""Phase 54.5 — per-wiki-page "My take" annotation.

Revision ID: 0023
Revises: 0022
Create Date: 2026-04-14

One row per (active-project) wiki slug: the user's freeform note /
highlight / disagreement attached to that page, rendered below the
main content in the web reader. The model doesn't write here; this
is pure user authorship.

Keyed on `slug` (primary key) rather than the `wiki_pages.id` UUID
because the wiki can be recompiled and slugs survive — an
annotation should track the concept, not a specific compile's
primary key row. If a wiki page is renamed, the annotation orphans
silently (intentional; the `wiki lint` command can surface orphans
later).

Single text field + updated_at. Intentionally simple — this is a
user-authored margin-note, not a structured document.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0023"
down_revision: Union[str, None] = "0022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "wiki_annotations",
        sa.Column("slug", sa.Text(), primary_key=True),
        sa.Column("body", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )


def downgrade() -> None:
    op.drop_table("wiki_annotations")
