"""Phase 32.7 — Compound learning Layer 1: episodic memory store (lessons).

Revision ID: 0012
Revises: 0011
Create Date: 2026-04-11

Adds the autowrite_lessons table — the producer/consumer surface for the
Reflexion-style episodic memory described in docs/research/RESEARCH.md §21 Layer 1.

Producer side: after each successful autowrite run, _distill_lessons_from_run
reads the per-iteration trajectory from autowrite_iterations (added in
Layer 0), prompts a SEPARATE LLM head (the fast model, not the scorer —
per the MAR critique) to extract 1-3 concrete lessons, and embeds each
lesson via bge-m3 for similarity retrieval.

Consumer side: before each new autowrite run, _get_relevant_lessons embeds
the section topic + plan, fetches all lessons for this book scoped to the
same section_slug (or, if none exist there, the broader book), computes
cosine similarity in Python, ranks by `importance × recency × similarity`
(the Generative Agents formula), and returns top-K (default 3-5).

Schema notes:
- `embedding` is REAL[] (1024 floats for bge-m3 dense vectors). Stored
  in PG rather than Qdrant because lesson tables stay small (~hundreds
  of rows per book at the steady state) so all-pairs similarity in
  Python is fast and avoids the Qdrant collection bookkeeping. If
  lesson counts ever grow beyond ~10k rows we'd switch to pgvector.
- `dimension` is the scorer dimension this lesson is about, when
  attributable: groundedness | completeness | coherence |
  citation_accuracy | hedging_fidelity | length | general.
- `score_delta` is `final_overall - first_iteration_overall` from the
  source run — a high-delta lesson came from a run that genuinely
  improved during iteration, which is a stronger signal than a lesson
  from a run that converged immediately or never improved.
- `importance` starts at 1.0 and is decayed externally on read by
  the recency formula (no scheduled job — the read path applies the
  decay so a lesson that's never read just sits at its stored value).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, REAL, UUID

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "autowrite_lessons",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("book_id", UUID(as_uuid=True),
                  sa.ForeignKey("books.id", ondelete="CASCADE"),
                  nullable=True),
        sa.Column("chapter_id", UUID(as_uuid=True),
                  sa.ForeignKey("book_chapters.id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("section_slug", sa.Text, nullable=False),
        sa.Column("lesson_text", sa.Text, nullable=False),
        sa.Column("source_run_id", UUID(as_uuid=True),
                  sa.ForeignKey("autowrite_runs.id", ondelete="CASCADE"),
                  nullable=True),
        sa.Column("score_delta", sa.Float, nullable=True),
        # 1024-dim bge-m3 dense vector. Stored as REAL[] (PG native array).
        sa.Column("embedding", ARRAY(REAL), nullable=True),
        sa.Column("importance", sa.Float, nullable=False, server_default="1.0"),
        # groundedness | completeness | coherence | citation_accuracy
        # | hedging_fidelity | length | general
        sa.Column("dimension", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "idx_autowrite_lessons_section",
        "autowrite_lessons",
        ["book_id", "chapter_id", "section_slug"],
    )
    op.create_index(
        "idx_autowrite_lessons_book",
        "autowrite_lessons",
        ["book_id"],
    )
    op.create_index(
        "idx_autowrite_lessons_dimension",
        "autowrite_lessons",
        ["dimension"],
        postgresql_where=sa.text("dimension IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_autowrite_lessons_dimension", table_name="autowrite_lessons")
    op.drop_index("idx_autowrite_lessons_book", table_name="autowrite_lessons")
    op.drop_index("idx_autowrite_lessons_section", table_name="autowrite_lessons")
    op.drop_table("autowrite_lessons")
