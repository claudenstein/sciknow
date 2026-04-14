"""Phase 52 — chunker_version on chunks + paper_sections.

Revision ID: 0022
Revises: 0021
Create Date: 2026-04-14

Adds an integer stamp so we can tell when the chunker implementation
has changed and the stored chunks are stale relative to the code. The
motivation is explicit in CLAUDE.md: `_SECTION_PATTERNS`,
`_SKIP_SECTIONS`, and `_PARAMS` are a "contract" and changing them
silently invalidates previously-stored chunks — but today the only
way to recover is `db reset`, which is a sledgehammer.

With this column we can surface staleness per-document and re-chunk
on demand without touching successful ingests of unrelated papers.
Default = 0 so every existing row looks stale vs the first versioned
release; incrementing the constant in `sciknow/ingestion/chunker.py`
is the signal to the pipeline that stored chunks need re-chunking.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0022"
down_revision: Union[str, None] = "0021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add to chunks (the primary carrier) + paper_sections (so the
    # chunker can skip re-parsing when sections are still current).
    op.add_column(
        "chunks",
        sa.Column("chunker_version", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "paper_sections",
        sa.Column("chunker_version", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("idx_chunks_chunker_version", "chunks", ["chunker_version"])


def downgrade() -> None:
    op.drop_index("idx_chunks_chunker_version", "chunks")
    op.drop_column("paper_sections", "chunker_version")
    op.drop_column("chunks", "chunker_version")
