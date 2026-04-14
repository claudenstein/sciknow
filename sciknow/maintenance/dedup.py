"""Phase 52 — chunk-level near-duplicate dedup.

Mempalace ports: `dedup.py`'s greedy keep-longest algorithm, adapted
for Qdrant + Postgres.

Use case: when `db expand` pulls a preprint v1 and v2 of the same
paper, SHA-256 at the ingest gate only catches byte-identical files,
so v1 and v2 both land as separate documents with largely overlapping
chunks. Qdrant ends up with paraphrased duplicate paragraphs and the
reranker wastes slots on near-identical hits.

Default scope: within-document only (catches MinerU retry artefacts
that occasionally produce the same chunk twice). Set
`cross_document=True` to also scan across the whole corpus — more
expensive (O(n²) Qdrant point-pair queries) but catches the
preprint-vs-journal case. The cross-document branch still groups by
SHA-similar title first to bound the comparison set; naïve
all-pairs would be unusable above ~5k chunks.

Threshold: cosine similarity above which two chunks are considered
duplicates. Default 0.92 — empirically tight enough that genuine
paraphrases don't collapse together while near-identical copies do.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger("sciknow.maintenance.dedup")


@dataclass
class DedupReport:
    """Result of a dedup run."""
    groups_seen: int
    chunks_scanned: int
    duplicates_found: int
    chunks_deleted: int
    dry_run: bool


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _greedy_keep_longest(
    items: list[tuple[str, str, np.ndarray, int]],
    *,
    threshold: float,
) -> list[str]:
    """Given [(chunk_id, doc_id, vector, content_len), ...] sorted by
    content length DESC, return chunk_ids to DELETE (those whose
    cosine to any already-kept chunk exceeds threshold).

    Greedy: for each incoming chunk, check cosine against every kept
    chunk. If above threshold, mark as dup. Otherwise keep.
    """
    kept: list[tuple[str, np.ndarray]] = []
    to_delete: list[str] = []
    for chunk_id, _doc_id, vec, _n_len in items:
        is_dup = False
        for _, kv in kept:
            if _cosine(vec, kv) >= threshold:
                is_dup = True
                break
        if is_dup:
            to_delete.append(chunk_id)
        else:
            kept.append((chunk_id, vec))
    return to_delete


def dedup_corpus(
    *,
    threshold: float = 0.92,
    cross_document: bool = False,
    dry_run: bool = True,
    limit_docs: int = 0,
) -> DedupReport:
    """Scan chunks, group them, identify duplicates, optionally delete.

    `cross_document=False` (default) groups chunks by their
    document_id and runs dedup within each document only. Safe and
    cheap. `cross_document=True` groups by (section_type, first 60
    chars of content) as a coarse pre-filter then dedups within each
    bucket — catches preprint-vs-journal near-duplicates without
    blowing up to all-pairs.

    Returns a DedupReport. When dry_run is True no deletes happen —
    the counts still reflect what would have been deleted.
    """
    from sqlalchemy import text as _sql_text
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import PAPERS_COLLECTION, get_client

    # Pull chunk_id + document_id + content_len + qdrant_point_id
    with get_session() as session:
        rows = session.execute(_sql_text("""
            SELECT id::text, document_id::text,
                   COALESCE(content_tokens, length(content)) AS n,
                   qdrant_point_id::text,
                   section_type,
                   SUBSTRING(content FROM 1 FOR 60) AS head
            FROM chunks
            WHERE qdrant_point_id IS NOT NULL
            ORDER BY document_id, n DESC
        """)).fetchall()

    if limit_docs:
        unique_docs = []
        seen = set()
        for r in rows:
            if r[1] not in seen:
                seen.add(r[1])
                unique_docs.append(r[1])
                if len(unique_docs) >= limit_docs:
                    break
        rows = [r for r in rows if r[1] in set(unique_docs)]

    groups: dict[tuple, list] = defaultdict(list)
    for chunk_id, doc_id, n, qid, section_type, head in rows:
        if cross_document:
            key = ((section_type or "")[:20], (head or "")[:40].strip().lower())
        else:
            key = (doc_id,)
        groups[key].append((chunk_id, doc_id, qid, n))

    # Multi-group batch Qdrant scroll. Collect all qdrant_ids to fetch
    # vectors for.
    needed_qids = [item[2] for grp in groups.values() for item in grp if item[2]]
    client = get_client()
    collection = PAPERS_COLLECTION
    vec_map: dict[str, np.ndarray] = {}
    if needed_qids:
        try:
            # Qdrant retrieve lets us fetch specific points by id
            for start in range(0, len(needed_qids), 256):
                batch = needed_qids[start:start + 256]
                points = client.retrieve(
                    collection_name=collection,
                    ids=batch,
                    with_vectors=True,
                )
                for p in points:
                    v = p.vector
                    if isinstance(v, dict):
                        # Named vectors — take the dense one
                        v = v.get("dense") or next(iter(v.values()))
                    if v is not None:
                        vec_map[str(p.id)] = np.asarray(v, dtype=np.float32)
        except Exception as exc:
            logger.warning("Qdrant vector fetch failed: %s", exc)

    duplicates_found = 0
    to_delete: list[tuple[str, str]] = []  # (chunk_id, qdrant_point_id)
    for _key, items in groups.items():
        # Build (chunk_id, doc_id, vector, n_len) and sort by n DESC
        scored = []
        for chunk_id, doc_id, qid, n in items:
            if qid and qid in vec_map:
                scored.append((chunk_id, doc_id, vec_map[qid], int(n or 0)))
        if len(scored) < 2:
            continue
        scored.sort(key=lambda t: -t[3])
        dup_ids = _greedy_keep_longest(scored, threshold=threshold)
        for did in dup_ids:
            # Find qid for this chunk
            qid = next((q for cid, _d, q, _n in items if cid == did), None)
            if qid:
                to_delete.append((did, qid))
                duplicates_found += 1

    deleted = 0
    if not dry_run and to_delete:
        from sqlalchemy import text as _sql_text2
        with get_session() as session:
            # Delete PG chunks in one IN () batch
            chunk_ids = [cid for cid, _q in to_delete]
            for start in range(0, len(chunk_ids), 500):
                batch = chunk_ids[start:start + 500]
                session.execute(_sql_text2(
                    "DELETE FROM chunks WHERE id::text = ANY(:ids)"
                ), {"ids": batch})
            session.commit()
            deleted += len(chunk_ids)
        # Delete Qdrant points (same ids)
        try:
            client.delete(
                collection_name=collection,
                points_selector=[q for _c, q in to_delete],
                wait=True,
            )
        except Exception as exc:
            logger.error("Qdrant point delete failed: %s", exc)

    return DedupReport(
        groups_seen=len(groups),
        chunks_scanned=len(rows),
        duplicates_found=duplicates_found,
        chunks_deleted=deleted,
        dry_run=dry_run,
    )
