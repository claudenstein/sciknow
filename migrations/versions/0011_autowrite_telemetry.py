"""Phase 32.6 — Compound learning Layer 0: autowrite telemetry tables.

Revision ID: 0011
Revises: 0010
Create Date: 2026-04-11

Adds three tables to capture per-run / per-iteration / per-retrieval data
that the autowrite loop currently throws away. This is the data foundation
for Layer 1 (episodic memory / lessons), Layer 2 (useful_count retrieval
boost), Layer 4 (DPO preference dataset), and the eventual style fingerprint
+ heuristic distillation passes.

Schema:

  autowrite_runs        — one row per autowrite invocation. Tracks the
                          full configuration, model, lifecycle, and final
                          outcome (final_overall, iterations_used,
                          converged, status).

  autowrite_iterations  — one row per (run, iteration). Mirrors the
                          existing drafts.custom_metadata.score_history
                          shape (already tracked since Phase 13) but
                          gives it queryability across runs and a
                          stable schema for downstream learning passes.

  autowrite_retrievals  — one row per (run, retrieved chunk). Records
                          the rank, rrf_score, and a was_cited boolean
                          set after the final draft is parsed for [N]
                          markers. This is the data that powers the
                          "useful chunks" learning loop in Layer 2.

Why now: the data was already implicitly there (custom_metadata,
SearchResult.chunk_id), it just wasn't being captured in queryable form.
This migration is non-destructive — existing autowrite runs won't have
historical rows but will start writing them on the next invocation.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── autowrite_runs ──────────────────────────────────────────────────
    op.create_table(
        "autowrite_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("book_id", UUID(as_uuid=True),
                  sa.ForeignKey("books.id", ondelete="CASCADE"),
                  nullable=True),
        sa.Column("chapter_id", UUID(as_uuid=True),
                  sa.ForeignKey("book_chapters.id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("section_slug", sa.Text, nullable=False),
        sa.Column("final_draft_id", UUID(as_uuid=True),
                  sa.ForeignKey("drafts.id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        # Lifecycle: running | completed | error | cancelled
        sa.Column("status", sa.Text, nullable=False, server_default="running"),
        # Run configuration
        sa.Column("model", sa.Text, nullable=True),
        sa.Column("target_words", sa.Integer, nullable=True),
        sa.Column("max_iter", sa.Integer, nullable=True),
        sa.Column("target_score", sa.Float, nullable=True),
        sa.Column("feature_versions", JSONB, nullable=False,
                  server_default="{}"),
        # Final outcome
        sa.Column("final_overall", sa.Float, nullable=True),
        sa.Column("iterations_used", sa.Integer, nullable=True),
        sa.Column("converged", sa.Boolean, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
    )
    op.create_index("idx_autowrite_runs_book", "autowrite_runs", ["book_id"])
    op.create_index("idx_autowrite_runs_section", "autowrite_runs",
                    ["book_id", "chapter_id", "section_slug"])
    op.create_index("idx_autowrite_runs_started", "autowrite_runs",
                    [sa.text("started_at DESC")])

    # ── autowrite_iterations ────────────────────────────────────────────
    op.create_table(
        "autowrite_iterations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("run_id", UUID(as_uuid=True),
                  sa.ForeignKey("autowrite_runs.id", ondelete="CASCADE"),
                  nullable=False),
        # 1-indexed (matches the existing score_history convention)
        sa.Column("iteration", sa.Integer, nullable=False),
        sa.Column("scores", JSONB, nullable=False, server_default="{}"),
        sa.Column("verification", JSONB, nullable=False, server_default="{}"),
        sa.Column("cove", JSONB, nullable=False, server_default="{}"),
        # KEEP | DISCARD | NULL (set after the rescore comparison)
        sa.Column("action", sa.Text, nullable=True),
        sa.Column("word_count", sa.Integer, nullable=True),
        sa.Column("word_count_delta", sa.Integer, nullable=True),
        sa.Column("weakest_dimension", sa.Text, nullable=True),
        sa.Column("revision_instruction", sa.Text, nullable=True),
        sa.Column("overall_pre", sa.Float, nullable=True),
        sa.Column("overall_post", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("run_id", "iteration",
                            name="uq_autowrite_iterations_run_iter"),
    )
    op.create_index("idx_autowrite_iterations_run",
                    "autowrite_iterations", ["run_id"])

    # ── autowrite_retrievals ────────────────────────────────────────────
    op.create_table(
        "autowrite_retrievals",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("run_id", UUID(as_uuid=True),
                  sa.ForeignKey("autowrite_runs.id", ondelete="CASCADE"),
                  nullable=False),
        # 1-indexed; matches the [N] marker the writer uses in the draft.
        sa.Column("source_position", sa.Integer, nullable=False),
        # Qdrant point id (the field SearchResult.chunk_id refers to —
        # not the postgres chunks.id, which is a different UUID).
        sa.Column("chunk_qdrant_id", UUID(as_uuid=True), nullable=True),
        sa.Column("document_id", UUID(as_uuid=True), nullable=True),
        sa.Column("rrf_score", sa.Float, nullable=True),
        # Set after the final draft is parsed for [N] markers in
        # _finalize_autowrite_run. This is the signal that powers the
        # "useful chunk" learning loop in Layer 2.
        sa.Column("was_cited", sa.Boolean, nullable=False,
                  server_default=sa.text("false")),
    )
    op.create_index("idx_autowrite_retrievals_run",
                    "autowrite_retrievals", ["run_id"])
    op.create_index(
        "idx_autowrite_retrievals_chunk_cited",
        "autowrite_retrievals", ["chunk_qdrant_id"],
        postgresql_where=sa.text("was_cited = true"),
    )
    op.create_index("idx_autowrite_retrievals_doc",
                    "autowrite_retrievals", ["document_id"])


def downgrade() -> None:
    op.drop_index("idx_autowrite_retrievals_doc", table_name="autowrite_retrievals")
    op.drop_index("idx_autowrite_retrievals_chunk_cited", table_name="autowrite_retrievals")
    op.drop_index("idx_autowrite_retrievals_run", table_name="autowrite_retrievals")
    op.drop_table("autowrite_retrievals")
    op.drop_index("idx_autowrite_iterations_run", table_name="autowrite_iterations")
    op.drop_table("autowrite_iterations")
    op.drop_index("idx_autowrite_runs_started", table_name="autowrite_runs")
    op.drop_index("idx_autowrite_runs_section", table_name="autowrite_runs")
    op.drop_index("idx_autowrite_runs_book", table_name="autowrite_runs")
    op.drop_table("autowrite_runs")
