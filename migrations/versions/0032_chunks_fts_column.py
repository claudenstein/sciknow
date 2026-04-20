"""Phase 54.6.136 — chunk-level FTS column.

Revision ID: 0032
Revises: 0031
Create Date: 2026-04-20

Adds ``chunks.search_vector tsvector`` as a GENERATED ALWAYS column
over ``chunks.content`` (english dictionary). GIN-indexed for fast
``@@`` queries.

Rationale:

The pre-existing FTS signal in hybrid_search queried
``paper_metadata.search_vector``, which indexes title + abstract +
keywords + journal. That is a *paper-level* lexical signal, while
dense and sparse signals operate on *chunk-level* content. The
bench's signal-overlap probe expected sparse ∩ FTS to be meaningfully
non-zero (both lexical) and consistently measured ~0.0 on both
targeted and generic probes — chunk IDs from paper-level FTS almost
never overlapped with chunks picked by dense/sparse because FTS
returned all chunks of metadata-matching papers rather than specific
matching chunks.

This migration makes FTS a true chunk-level complement to sparse.
Exact-term queries (chemical formulas, specific numbers, rare author
names, uncommon terminology) that don't appear in titles/abstracts
but DO appear in body text become findable via FTS for the first
time. Paper-level relevance is still captured by dense embeddings on
chunks + the separate abstracts collection.

GENERATED … STORED means:
  * No trigger needed, no application-level maintenance
  * Automatically populated for all existing rows on ALTER TABLE
  * Automatically maintained on INSERT/UPDATE of ``content``
  * Costs ~50-100 MB of storage on a 33k-chunk corpus (tsvector +
    GIN index); acceptable.

``paper_metadata.search_vector`` is kept as-is — it still powers
the catalog / search_by_metadata code paths. Only ``_postgres_fts``
in hybrid_search switches to the chunk column.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0032"
down_revision: Union[str, None] = "0031"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ALTER TABLE ... ADD COLUMN ... GENERATED ALWAYS AS (...) STORED
    # populates existing rows as part of the DDL; on 33k chunks this
    # is a few seconds of to_tsvector() work.
    op.execute("""
        ALTER TABLE chunks
        ADD COLUMN search_vector tsvector
        GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED
    """)
    op.create_index(
        "idx_chunks_search_vector",
        "chunks",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("idx_chunks_search_vector", table_name="chunks")
    with op.batch_alter_table("chunks") as batch:
        batch.drop_column("search_vector")
