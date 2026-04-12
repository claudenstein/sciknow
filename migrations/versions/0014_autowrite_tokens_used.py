"""Phase 33 — add tokens_used to autowrite_runs.

Revision ID: 0014
Revises: 0013
Create Date: 2026-04-12

The _AutowriteLogger already tracks total_tokens in its _state dict
throughout the run. This column persists that count to autowrite_runs
at finalization time so we can aggregate cumulative token usage across
all autowrite runs for the dashboard.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: Union[str, None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "autowrite_runs",
        sa.Column("tokens_used", sa.Integer, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("autowrite_runs", "tokens_used")
