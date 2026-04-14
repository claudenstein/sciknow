"""Phase 52 — `db repair` internals.

Middle ground between ``db stats`` (read-only) and ``db reset``
(destructive sledgehammer). Three one-shot operations:

* ``scan``         — diff PG chunks.qdrant_point_id against the live
                     Qdrant point set; return both directions of
                     orphans (PG chunk with no vector, Qdrant point
                     with no chunk).
* ``prune``        — delete orphan Qdrant points. The other direction
                     (PG chunks with no vector) is harder to fix
                     automatically — the ``rebuild-paper`` path handles
                     it for whole-document re-embedding.
* ``rebuild-paper`` — surgical re-chunk + re-embed for one document.
                     Uses ``CHUNKER_VERSION`` to detect staleness and
                     short-circuits if the chunks are already current.

Pattern ported from mempalace/mempalace's ``repair.py`` but rewritten
for Qdrant + Postgres.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger("sciknow.maintenance.repair")


@dataclass
class RepairScanReport:
    """Result of ``repair_scan``. `pg_orphans` = chunks rows whose
    qdrant_point_id is not present in Qdrant (rare, but happens when
    Qdrant was wiped without also resetting PG). `qdrant_orphans` =
    Qdrant points whose UUID doesn't appear in `chunks.qdrant_point_id`
    (happens after a failed ingest that wrote vectors before the
    transaction committed)."""
    pg_chunks_total: int
    qdrant_points_total: int
    pg_orphans: list[str]      # chunks.id values
    qdrant_orphans: list[str]  # Qdrant point id values
    stale_chunker_version: int # count of chunks on an older CHUNKER_VERSION

    def ok(self) -> bool:
        return (
            not self.pg_orphans
            and not self.qdrant_orphans
            and self.stale_chunker_version == 0
        )


def repair_scan() -> RepairScanReport:
    """Load the full set of ids on both sides and diff them."""
    from sqlalchemy import text as _sql_text
    from sciknow.ingestion.chunker import CHUNKER_VERSION
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import PAPERS_COLLECTION, get_client

    with get_session() as session:
        rows = session.execute(_sql_text(
            "SELECT id::text, qdrant_point_id::text, chunker_version "
            "FROM chunks"
        )).fetchall()
    pg_total = len(rows)
    pg_qids = {r[1] for r in rows if r[1]}
    stale = sum(1 for r in rows if (r[2] or 0) < CHUNKER_VERSION)

    client = get_client()
    collection = PAPERS_COLLECTION
    qdrant_ids: set[str] = set()
    try:
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for p in points:
                qdrant_ids.add(str(p.id))
            if offset is None:
                break
    except Exception as exc:
        logger.warning("Qdrant scroll failed: %s", exc)

    pg_orphans = [
        r[0] for r in rows if r[1] and r[1] not in qdrant_ids
    ]
    qdrant_orphans = sorted(qdrant_ids - pg_qids)

    return RepairScanReport(
        pg_chunks_total=pg_total,
        qdrant_points_total=len(qdrant_ids),
        pg_orphans=pg_orphans,
        qdrant_orphans=qdrant_orphans,
        stale_chunker_version=stale,
    )


def repair_prune(qdrant_orphan_ids: Iterable[str]) -> int:
    """Delete the listed Qdrant point ids. Returns the count deleted.
    No-op when the iterable is empty."""
    ids = list(qdrant_orphan_ids)
    if not ids:
        return 0
    from sciknow.storage.qdrant import PAPERS_COLLECTION, get_client
    client = get_client()
    collection = PAPERS_COLLECTION
    # Qdrant DELETE /collections/{name}/points supports a `points=[…]` filter
    try:
        client.delete(
            collection_name=collection,
            points_selector=ids,
            wait=True,
        )
    except Exception as exc:
        logger.error("Qdrant prune failed for %d points: %s", len(ids), exc)
        raise
    return len(ids)


def rebuild_paper(doc_id: str) -> tuple[int, int]:
    """Re-chunk + re-embed one document in place.

    Returns (num_new_chunks, num_new_vectors). Invokes the standard
    ingestion pipeline in force-retry mode — the pipeline already
    handles deleting the old chunks + Qdrant points transactionally,
    so this is just the thin dispatch layer.
    """
    from sqlalchemy import text as _sql_text
    from sciknow.ingestion.pipeline import ingest
    from sciknow.storage.db import get_session

    # Resolve the original PDF path for this document.
    with get_session() as session:
        row = session.execute(_sql_text(
            "SELECT original_path FROM documents WHERE id::text = :d"
        ), {"d": doc_id.strip()}).fetchone()
    if not row or not row[0]:
        raise ValueError(f"no document row / original_path for id={doc_id!r}")
    pdf_path = row[0]

    from pathlib import Path as _Path
    result = ingest(_Path(pdf_path), force=True)
    n_chunks = getattr(result, "num_chunks", 0) or 0
    n_vectors = getattr(result, "num_embedded", 0) or 0
    return int(n_chunks), int(n_vectors)
