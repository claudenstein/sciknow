"""Phase 32.9 — Compound learning Layer 4: capture pre/post-revision content
on autowrite_iterations.

Revision ID: 0013
Revises: 0012
Create Date: 2026-04-11

Layer 4 of the compound-learning roadmap (docs/RESEARCH.md §21) needs
the actual revised text from each autowrite iteration so it can build
DPO preference pairs. Phase 32.6 (Layer 0) captured per-iteration scores
and verdicts but not the content — drafts.content only holds the LATEST
state, and the in-flight revisions get overwritten by Phase 19's
incremental save callback.

This migration adds two TEXT columns:

  pre_revision_content   — the draft content as it ENTERED the iteration
                           (i.e. what the scorer scored at the top of
                           the loop)
  post_revision_content  — the draft content AFTER the revision stream
                           completed (regardless of KEEP/DISCARD verdict)

Pair extraction rule (used by `sciknow book preferences export` —
shipping in the same phase):

  For action='KEEP':
    chosen   = post_revision_content
    rejected = pre_revision_content

  For action='DISCARD':
    chosen   = pre_revision_content
    rejected = post_revision_content

So every revision attempt — accepted or rejected — produces one
preference pair. That's roughly 2× the data the original roadmap
estimated, since the roadmap only counted KEEPs.

Storage cost: typical autowrite run is 3-5 iterations × ~10KB content
× 2 columns = 60-100KB per run. Over 1000 runs that's <100MB. Both
columns are nullable so existing rows (from runs that completed
before this migration) just have NULLs and the export job skips them.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "autowrite_iterations",
        sa.Column("pre_revision_content", sa.Text, nullable=True),
    )
    op.add_column(
        "autowrite_iterations",
        sa.Column("post_revision_content", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("autowrite_iterations", "post_revision_content")
    op.drop_column("autowrite_iterations", "pre_revision_content")
