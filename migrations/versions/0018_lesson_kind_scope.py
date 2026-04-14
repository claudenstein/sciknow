"""Phase 47.1 — kind + scope on autowrite_lessons (DeepScientist taxonomy).

Revision ID: 0018
Revises: 0017
Create Date: 2026-04-13

Phase 47 audit of DeepScientist's Findings Memory surfaced two axes on
memory cards that sciknow's existing ``autowrite_lessons`` table
didn't carry. ``dimension`` (already present) records which scorer
axis the lesson is about (groundedness, coherence, citation_accuracy,
…). The two new axes are orthogonal:

- ``kind`` — what *kind of thing* the lesson refers to. Default
  ``episode`` (a single writing session's trajectory). Other values:

  * ``paper``         — a corpus paper that kept showing up as a
                        retrieval source; lesson is about its
                        usefulness / placement / trustworthiness.
  * ``idea``          — a topic or framing that worked well; the
                        writer prompt should consider it again.
  * ``decision``      — a pivot / rollback / revise verdict made during
                        a run (why we KEEP'd or DISCARD'd).
  * ``knowledge``     — a domain fact the writer should internalize
                        across future runs ("BP 1950 is the radiocarbon
                        convention; don't conflate with calendar dates").
  * ``rejected_idea`` — a topic tried and scored poorly; the gap-finder
                        should NOT re-propose it (Phase 47.2 gate).
  * ``episode``       — default / legacy bucket for undifferentiated
                        per-run lessons (the pre-Phase-47 shape).

- ``scope`` — where the lesson applies. Default ``book`` (the run's
  book_id). The other value is ``global`` (no book_id; cross-project
  knowledge promoted from ≥ 3 books via Phase 47.4 promote-to-global).

Both columns are NOT NULL with explicit server_default so all legacy
rows are backfilled cleanly and no producer/consumer change is
required immediately to keep working.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0018"
down_revision: Union[str, None] = "0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "autowrite_lessons",
        sa.Column("kind", sa.Text(), nullable=False,
                  server_default="episode"),
    )
    op.add_column(
        "autowrite_lessons",
        sa.Column("scope", sa.Text(), nullable=False,
                  server_default="book"),
    )
    # An index on (kind, section_slug) makes the Phase 47.2
    # rejected-idea gate a single-index lookup.
    op.create_index(
        "idx_autowrite_lessons_kind_section",
        "autowrite_lessons",
        ["kind", "section_slug"],
    )
    # An index on (scope, section_slug) keeps the consumer's global-
    # scope query under a millisecond even at 10k+ global rows.
    op.create_index(
        "idx_autowrite_lessons_scope_section",
        "autowrite_lessons",
        ["scope", "section_slug"],
    )
    # A global lesson is scope='global' AND book_id IS NULL. Validate at
    # the DB level with a CHECK so a faulty producer can't accidentally
    # write scope='global' with a non-null book_id.
    op.create_check_constraint(
        "ck_autowrite_lessons_scope_book",
        "autowrite_lessons",
        "scope IN ('book', 'global') AND "
        "(scope = 'book' OR book_id IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_autowrite_lessons_scope_book",
        "autowrite_lessons", type_="check",
    )
    op.drop_index("idx_autowrite_lessons_scope_section",
                  table_name="autowrite_lessons")
    op.drop_index("idx_autowrite_lessons_kind_section",
                  table_name="autowrite_lessons")
    op.drop_column("autowrite_lessons", "scope")
    op.drop_column("autowrite_lessons", "kind")
