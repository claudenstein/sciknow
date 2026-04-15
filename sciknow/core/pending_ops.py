"""Phase 54.6.7 — CRUD helpers for the pending_downloads table.

Every expand flow (db expand / expand-author / expand-cites / expand-
topic / expand-coauthors / book auto-expand / db download-dois) writes
here whenever ``find_and_download`` returns no OA PDF. The web reader
and CLI then let the user retry, mark-done, abandon, or export to CSV
for manual acquisition.

Keep the public surface small — other modules should use these
functions rather than reaching into the SQL directly.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import text as sql_text

from sciknow.storage.db import get_session

logger = logging.getLogger(__name__)


def record_failure(
    *, doi: str, title: str = "", authors: list[str] | None = None,
    year: int | None = None, arxiv_id: str | None = None,
    source_method: str | None = None, source_query: str | None = None,
    relevance_score: float | None = None,
    reason: str = "no_oa",
) -> None:
    """Upsert one pending-download row by DOI.

    * New DOI → INSERT with attempt_count=1, status='pending'.
    * Existing DOI → UPDATE: bump attempt_count, refresh
      last_attempt_at / last_failure_reason, keep status unless
      user has explicitly moved it to manual_acquired / abandoned
      (those are sticky — we don't re-open them from a silent retry).

    ``doi`` is required. The other fields are stored best-effort so
    the user has enough metadata to act on the row later.
    """
    if not doi or not doi.strip():
        return
    now = datetime.now(timezone.utc)
    try:
        with get_session() as session:
            session.execute(sql_text("""
                INSERT INTO pending_downloads
                    (doi, arxiv_id, title, authors, year,
                     source_method, source_query, relevance_score,
                     attempt_count, last_attempt_at, last_failure_reason,
                     status)
                VALUES
                    (:doi, :arxiv_id, :title,
                     CAST(:authors AS jsonb), :year,
                     :source_method, :source_query, :relevance_score,
                     1, :now, :reason, 'pending')
                ON CONFLICT (doi) DO UPDATE SET
                    -- bump retry counter + update last-attempt fields
                    attempt_count       = pending_downloads.attempt_count + 1,
                    last_attempt_at     = :now,
                    last_failure_reason = :reason,
                    -- Refresh metadata if the incoming row has richer
                    -- content (the first call might have only had a
                    -- DOI; a later one might have title + authors).
                    title               = COALESCE(
                        NULLIF(EXCLUDED.title, ''), pending_downloads.title
                    ),
                    authors             = CASE
                        WHEN jsonb_array_length(EXCLUDED.authors) > 0
                        THEN EXCLUDED.authors
                        ELSE pending_downloads.authors
                    END,
                    year                = COALESCE(EXCLUDED.year, pending_downloads.year),
                    arxiv_id            = COALESCE(EXCLUDED.arxiv_id, pending_downloads.arxiv_id),
                    source_method       = COALESCE(EXCLUDED.source_method, pending_downloads.source_method),
                    source_query        = COALESCE(EXCLUDED.source_query, pending_downloads.source_query),
                    relevance_score     = COALESCE(EXCLUDED.relevance_score, pending_downloads.relevance_score),
                    updated_at          = :now
                    -- IMPORTANT: do NOT touch status here — sticky
                    -- user states (manual_acquired / abandoned)
                    -- must not be auto-reopened on a silent retry.
            """), {
                "doi": doi.strip(),
                "arxiv_id": arxiv_id,
                "title": title or "",
                "authors": _json_dumps(authors or []),
                "year": year,
                "source_method": source_method,
                "source_query": source_query,
                "relevance_score": relevance_score,
                "now": now,
                "reason": reason,
            })
            session.commit()
    except Exception as exc:
        # Don't let a pending-table write break the expand flow —
        # log and move on. The main pipeline's own failure reporting
        # still surfaces the no-OA count to the user.
        logger.warning("record_failure pending_downloads upsert failed: %s", exc)


def record_failures_bulk(rows: Iterable[dict[str, Any]]) -> int:
    """Bulk upsert — used when a batch finishes (e.g. the post-download
    summary in the expand CLI). Each dict mirrors the kwargs of
    ``record_failure``; ``doi`` is required. Returns the number of
    rows processed.
    """
    n = 0
    for r in rows:
        record_failure(**r)
        n += 1
    return n


def list_pending(
    *, status: str | None = "pending",
    source_method: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """List rows from the pending_downloads table."""
    q = "SELECT id::text, doi, arxiv_id, title, authors, year, " \
        "source_method, source_query, relevance_score, " \
        "attempt_count, last_attempt_at, last_failure_reason, " \
        "status, notes, created_at, updated_at " \
        "FROM pending_downloads WHERE 1=1"
    params: dict[str, Any] = {}
    if status and status != "all":
        q += " AND status = :status"
        params["status"] = status
    if source_method:
        q += " AND source_method = :sm"
        params["sm"] = source_method
    q += " ORDER BY created_at DESC"
    if limit and limit > 0:
        q += " LIMIT :lim"
        params["lim"] = int(limit)
    with get_session() as session:
        rows = session.execute(sql_text(q), params).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "id": r[0], "doi": r[1], "arxiv_id": r[2],
            "title": r[3], "authors": r[4] or [], "year": r[5],
            "source_method": r[6], "source_query": r[7],
            "relevance_score": r[8],
            "attempt_count": r[9],
            "last_attempt_at": r[10].isoformat() if r[10] else None,
            "last_failure_reason": r[11],
            "status": r[12], "notes": r[13],
            "created_at": r[14].isoformat() if r[14] else None,
            "updated_at": r[15].isoformat() if r[15] else None,
        })
    return out


def update_status(doi: str, *, status: str, notes: str | None = None) -> bool:
    """Mark a pending row as manual_acquired / abandoned / reopened
    (back to 'pending'). Returns True if a row was actually updated.
    """
    if status not in ("pending", "manual_acquired", "abandoned"):
        raise ValueError(f"invalid status {status!r}")
    with get_session() as session:
        result = session.execute(sql_text("""
            UPDATE pending_downloads
               SET status = :s,
                   notes  = COALESCE(:notes, notes),
                   updated_at = now()
             WHERE doi = :doi
        """), {"s": status, "notes": notes, "doi": doi.strip()})
        session.commit()
    return (result.rowcount or 0) > 0


def remove(doi: str) -> bool:
    """Delete a pending row outright. Returns True if it existed."""
    with get_session() as session:
        result = session.execute(sql_text(
            "DELETE FROM pending_downloads WHERE doi = :doi"
        ), {"doi": doi.strip()})
        session.commit()
    return (result.rowcount or 0) > 0


def mark_acquired_by_hash_if_ingested(doi: str) -> bool:
    """On successful ingest (pipeline archive), check if the paper's
    DOI is in pending_downloads — if yes, mark it manual_acquired
    so the user sees the row resolved next time they open the panel.
    Not called from the main pipeline today, but exposed here for
    future integration in ``_archive_pdf``.
    """
    return update_status(doi, status="manual_acquired",
                         notes="auto-resolved via successful ingest")


# ── Helpers ─────────────────────────────────────────────────────────

def _json_dumps(value: Any) -> str:
    import json as _json
    return _json.dumps(value, ensure_ascii=False)
