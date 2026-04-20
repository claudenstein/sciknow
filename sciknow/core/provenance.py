"""Phase 54.6.117 (Tier 4 #1) — write + query per-document provenance.

Single-write helper (``record(...)``) called by the expand pipeline at
the point each downloaded paper becomes a ``documents`` row; query
helpers (``get_by_doc_id`` / ``get_by_doi`` / ``get_by_paper_id``) for
the CLI + web "why is this paper here?" tooltip.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def record(
    *,
    doc_id: str | None = None,
    doi: str | None = None,
    source: str,
    round_n: int | None = None,
    relevance_query: str = "",
    question: str = "",
    subtopic: str = "",
    seed_paper_ids: list[str] | None = None,
    signals: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """Write a provenance record to the matching ``documents`` row.

    Caller must supply either ``doc_id`` (documents.id UUID) OR ``doi``
    (case-insensitive). Returns True when one row was updated. Merges
    with any existing provenance dict so re-entries (e.g. a paper
    rediscovered via a new source) keep the earlier context under
    ``provenance.history[]``.
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    if not (doc_id or doi):
        return False

    body: dict[str, Any] = {
        "source": source,
        "selected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if round_n is not None:
        body["round"] = round_n
    if relevance_query:
        body["relevance_query"] = relevance_query
    if question:
        body["question"] = question
    if subtopic:
        body["subtopic"] = subtopic
    if seed_paper_ids:
        body["seed_paper_ids"] = list(seed_paper_ids)
    if signals:
        body["signals"] = {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in signals.items() if v is not None
        }
    if extra:
        body["extra"] = extra

    with get_session() as session:
        if doc_id:
            row = session.execute(text(
                "SELECT provenance FROM documents WHERE id::text = :x LIMIT 1"
            ), {"x": str(doc_id)}).fetchone()
            where_sql = "id::text = :x"
            where_param = str(doc_id)
        else:
            row = session.execute(text("""
                SELECT d.provenance FROM documents d
                JOIN paper_metadata pm ON pm.document_id = d.id
                WHERE LOWER(pm.doi) = LOWER(:x) LIMIT 1
            """), {"x": doi}).fetchone()
            where_sql = """id = (SELECT document_id FROM paper_metadata
                                  WHERE LOWER(doi) = LOWER(:x) LIMIT 1)"""
            where_param = doi
        if row is None:
            return False

        existing = row[0] or {}
        # If a previous record exists and it's meaningfully different,
        # keep it under history[]. Shallow equality check on source +
        # round + subtopic avoids history spam on retries of the same
        # round.
        if existing:
            sig_prev = (existing.get("source"), existing.get("round"),
                        existing.get("subtopic"))
            sig_new = (body.get("source"), body.get("round"),
                       body.get("subtopic"))
            if sig_prev != sig_new:
                history = list(existing.get("history") or [])
                # Drop the nested `history` from the archived copy.
                archived = {k: v for k, v in existing.items() if k != "history"}
                history.append(archived)
                body["history"] = history[-10:]   # cap so we don't grow unbounded
            else:
                body["history"] = existing.get("history") or []

        session.execute(
            text(f"UPDATE documents SET provenance = CAST(:p AS jsonb) "
                 f"WHERE {where_sql}"),
            {"p": json.dumps(body), "x": where_param},
        )
        session.commit()
    return True


def get_by_doc_id(doc_id: str) -> dict | None:
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        row = session.execute(text(
            "SELECT provenance FROM documents WHERE id::text = :x"
        ), {"x": str(doc_id)}).fetchone()
    return (row[0] if row else None) or None


def get_by_doi(doi: str) -> dict | None:
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        row = session.execute(text("""
            SELECT d.provenance FROM documents d
            JOIN paper_metadata pm ON pm.document_id = d.id
            WHERE LOWER(pm.doi) = LOWER(:x) LIMIT 1
        """), {"x": doi}).fetchone()
    return (row[0] if row else None) or None


def lookup(key: str) -> tuple[str | None, dict | None]:
    """Look up provenance by DOI, arxiv_id, or a document.id prefix.
    Returns ``(doc_id, provenance)`` or ``(None, None)``."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        # Try as an arxiv_id / DOI first (text match on paper_metadata)
        row = session.execute(text("""
            SELECT d.id::text, d.provenance
            FROM documents d
            JOIN paper_metadata pm ON pm.document_id = d.id
            WHERE LOWER(pm.doi) = LOWER(:k)
               OR LOWER(pm.arxiv_id) = LOWER(:k)
            LIMIT 1
        """), {"k": key.strip()}).fetchone()
        if row:
            return row[0], row[1]
        # Fall through: documents.id prefix match
        row = session.execute(text("""
            SELECT id::text, provenance
            FROM documents WHERE id::text LIKE :k
            LIMIT 1
        """), {"k": key.strip() + "%"}).fetchone()
        if row:
            return row[0], row[1]
    return None, None
