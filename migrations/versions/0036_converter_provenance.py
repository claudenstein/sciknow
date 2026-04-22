"""Phase 54.6.211 — converter provenance stamp.

Revision ID: 0036
Revises: 0035
Create Date: 2026-04-22

Fulfils Phase 1 of roadmap item 3.1.6 (full MinerU 2.5-Pro migration
via vLLM). Adds two nullable TEXT columns to ``documents``:

  * ``converter_backend`` — the specific backend variant that
    produced this document's MinerU/Marker output. Example values:
    ``mineru-pipeline``, ``mineru-vlm-pro-vllm``,
    ``mineru-vlm-pro-transformers``, ``mineru-vlm-auto``,
    ``marker-json``, ``marker-md``.
  * ``converter_version`` — free-form producer version string, e.g.
    ``mineru 3.0.9`` or ``marker-pdf 1.6.2``. Captured from the
    packages' ``__version__`` attribute at convert time.

Both columns are nullable because pre-54.6.211 rows have no stamp
(they were produced before this migration ran). The ingest
pipeline populates them on every subsequent convert regardless of
backend, so new rows + any re-ingested rows are fully attributed.

Downstream consumers:
  * ``sciknow db failures`` (§3.11.6) attributes failure classes
    to a backend variant.
  * Post-migration audits can distinguish pipeline-era chunks from
    VLM-Pro-era chunks even though the content_list schema is
    shared.
  * Future retrieval-era filters (if we ever compare quality
    across the boundary).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0036"
down_revision: Union[str, None] = "0035"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("documents") as batch:
        batch.add_column(
            sa.Column("converter_backend", sa.Text(), nullable=True)
        )
        batch.add_column(
            sa.Column("converter_version", sa.Text(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("documents") as batch:
        batch.drop_column("converter_version")
        batch.drop_column("converter_backend")
