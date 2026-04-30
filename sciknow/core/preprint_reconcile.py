"""Phase 54.6.125 (Tier 3 #3) — preprint ↔ journal reconciliation.

Detects when two ``documents`` rows represent the same paper (most
commonly an arXiv preprint + its later journal publication) and marks
the non-canonical row with ``canonical_document_id`` so retrieval
hides it without deleting it.

See ``docs/research/EXPAND_ENRICH_RESEARCH_2.md`` §1.3.

Design (locked 54.6.125 — `user has no drafts` simplified):
- **Canonical**: journal > preprint (peer-reviewed, stable DOI). Tie
  break: whichever has more ingested chunks (content-richness proxy).
- **Non-destructive**: nothing deleted. ``documents.canonical_document_id``
  FK on the non-canonical row; retrieval filters where this is NULL.
- **Reversible**: ``sciknow db unreconcile <doc_id>`` clears the FK.
- **Detection**: group corpus DOIs by OpenAlex ``work_id``. Any group
  with ≥ 2 rows is a reconciliation candidate. OpenAlex returns the
  same ``work_id`` for preprint + journal DOIs of the same paper via
  its ``ids.doi`` cross-reference.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)


# ── Detection ────────────────────────────────────────────────────────


@dataclass
class DocInfo:
    """A corpus document's reconciliation-relevant state."""
    doc_id: str
    doi: str
    arxiv_id: str | None
    title: str
    year: int | None
    journal: str | None
    n_chunks: int
    canonical_document_id: str | None = None

    @property
    def is_canonical(self) -> bool:
        return self.canonical_document_id is None

    @property
    def is_preprint(self) -> bool:
        d = (self.doi or "").lower()
        # arXiv DOIs: 10.48550/arxiv.*
        if d.startswith("10.48550/arxiv."):
            return True
        if self.arxiv_id:
            return True
        # No journal assigned is a weak preprint signal (but not
        # decisive; some theses / technical reports have no journal
        # and aren't preprints either).
        return False


@dataclass
class ReconciliationPair:
    """One canonical + one non-canonical document mapped as the same paper."""
    canonical: DocInfo
    non_canonical: DocInfo
    reason: str    # "journal-beats-preprint" | "more-chunks" | "..."
    openalex_work_id: str = ""


def _fetch_all_corpus_docs() -> list[DocInfo]:
    """Snapshot every document + its paper_metadata row + chunk count."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        rows = session.execute(text("""
            SELECT d.id::text, LOWER(COALESCE(pm.doi, '')) AS doi,
                   pm.arxiv_id, pm.title, pm.year, pm.journal,
                   (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS n_chunks,
                   d.canonical_document_id::text
            FROM documents d
            JOIN paper_metadata pm ON pm.document_id = d.id
            WHERE d.ingestion_status = 'complete'
        """)).fetchall()
    out = []
    for r in rows:
        doc_id, doi, arxiv, title, year, journal, n, canonical = r
        if not doi:
            continue
        out.append(DocInfo(
            doc_id=doc_id, doi=doi, arxiv_id=arxiv,
            title=title or "", year=year, journal=journal,
            n_chunks=int(n or 0),
            canonical_document_id=canonical,
        ))
    return out


def _group_by_openalex_work(
    docs: list[DocInfo],
    *,
    on_progress=None,
) -> dict[str, list[DocInfo]]:
    """For each DOI batch, hit OpenAlex `/works?filter=doi:a|b|...` and
    group docs by the returned OpenAlex work_id. Returns
    ``{work_id: [DocInfo, …]}`` with at most one list per work.
    """
    from sciknow.ingestion.expand_apis import fetch_openalex_work
    # We can also use expand_ops._paginate_works for batched DOI lookup,
    # but that path pulls the full abstract/referenced_works payload.
    # For reconciliation we just need the work_id, so fetch_openalex_work
    # (which caches per DOI) is cheaper.
    out: dict[str, list[DocInfo]] = {}
    for i, d in enumerate(docs):
        if on_progress:
            on_progress(i, len(docs), d.doi)
        try:
            w = fetch_openalex_work(d.doi)
        except Exception as exc:  # noqa: BLE001
            logger.debug("openalex fetch failed for %s: %s", d.doi, exc)
            continue
        if not w:
            continue
        wid = (w.get("id") or "").rsplit("/", 1)[-1]
        if not wid:
            continue
        out.setdefault(wid, []).append(d)
    return out


def _pick_canonical(a: DocInfo, b: DocInfo) -> tuple[DocInfo, DocInfo, str]:
    """Return (canonical, non_canonical, reason).

    Rules (first match wins):
      1. One side is preprint and the other isn't → journal wins.
      2. One side has strictly more chunks → more-chunks wins.
      3. Newer year wins (later publication is usually the one of record).
      4. Deterministic tie-break on doc_id ascending.
    """
    if a.is_preprint and not b.is_preprint:
        return b, a, "journal-beats-preprint"
    if b.is_preprint and not a.is_preprint:
        return a, b, "journal-beats-preprint"
    if a.n_chunks > b.n_chunks:
        return a, b, "more-chunks"
    if b.n_chunks > a.n_chunks:
        return b, a, "more-chunks"
    ay, by = a.year or 0, b.year or 0
    if ay > by:
        return a, b, "newer-year"
    if by > ay:
        return b, a, "newer-year"
    # Deterministic fallback so re-runs give the same pair.
    if a.doc_id < b.doc_id:
        return a, b, "tie-break-doc-id"
    return b, a, "tie-break-doc-id"


def detect_pairs(*, on_progress=None) -> list[ReconciliationPair]:
    """Scan the corpus and return all preprint+journal reconciliation
    candidates that are NOT already reconciled."""
    docs = _fetch_all_corpus_docs()
    unresolved = [d for d in docs if d.canonical_document_id is None]
    groups = _group_by_openalex_work(unresolved, on_progress=on_progress)

    pairs: list[ReconciliationPair] = []
    for wid, members in groups.items():
        if len(members) < 2:
            continue
        # Resolve n-way groups by pairwise-reduction against the first
        # canonical pick. This keeps the algorithm simple and correct:
        # we always end with one canonical row per work_id.
        canonical = members[0]
        for other in members[1:]:
            canonical, nc, reason = _pick_canonical(canonical, other)
            pairs.append(ReconciliationPair(
                canonical=canonical, non_canonical=nc,
                reason=reason, openalex_work_id=wid,
            ))
    return pairs


# ── Apply ────────────────────────────────────────────────────────────


def apply_reconciliation(pair: ReconciliationPair) -> bool:
    """Set ``documents.canonical_document_id`` on the non-canonical
    row; also copy the preprint DOI onto the canonical's
    ``paper_metadata.preprint_doi`` if the non-canonical had one."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        session.execute(text("""
            UPDATE documents SET canonical_document_id = CAST(:canon AS uuid)
            WHERE id::text = :nc
        """), {"canon": pair.canonical.doc_id, "nc": pair.non_canonical.doc_id})
        # Preprint DOI goes onto the canonical's paper_metadata row so
        # `db provenance` / retrieval output show both identifiers.
        # SQLAlchemy interprets `{…}` in a textual SQL as a bind param
        # substitution, which collides with jsonb_set's text-array
        # path syntax `'{key}'`. Work around with a CAST.
        nc_doi = pair.non_canonical.doi
        if nc_doi and pair.non_canonical.is_preprint:
            session.execute(text("""
                UPDATE paper_metadata
                SET extra = jsonb_set(
                    COALESCE(extra, '{}'::jsonb),
                    CAST(:path AS text[]),
                    to_jsonb(CAST(:d AS text)), true
                )
                WHERE document_id = (
                    SELECT id FROM documents WHERE id::text = :canon
                )
            """), {"path": ["preprint_doi"], "d": nc_doi, "canon": pair.canonical.doc_id})
        session.commit()
    return True


def undo_reconciliation(doc_id: str) -> bool:
    """Clear ``canonical_document_id`` on a document. Returns True if
    a row was actually updated."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        res = session.execute(text("""
            UPDATE documents SET canonical_document_id = NULL
            WHERE id::text = :x AND canonical_document_id IS NOT NULL
        """), {"x": doc_id})
        session.commit()
        return (res.rowcount or 0) > 0


def list_reconciliations() -> list[dict]:
    """Return all currently-active canonical→non-canonical mappings,
    with the titles + DOIs of both sides for display."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        rows = session.execute(text("""
            SELECT nc.id::text AS nc_id,
                   nc_pm.title AS nc_title, nc_pm.doi AS nc_doi,
                   nc_pm.year AS nc_year,
                   canon.id::text AS canon_id,
                   canon_pm.title AS canon_title, canon_pm.doi AS canon_doi,
                   canon_pm.year AS canon_year, canon_pm.journal AS canon_journal
            FROM documents nc
            JOIN documents canon ON canon.id = nc.canonical_document_id
            JOIN paper_metadata nc_pm ON nc_pm.document_id = nc.id
            JOIN paper_metadata canon_pm ON canon_pm.document_id = canon.id
            ORDER BY nc_pm.year DESC NULLS LAST, canon_pm.title
        """)).fetchall()
    return [
        {
            "non_canonical_id": r[0], "non_canonical_title": r[1],
            "non_canonical_doi": r[2], "non_canonical_year": r[3],
            "canonical_id": r[4], "canonical_title": r[5],
            "canonical_doi": r[6], "canonical_year": r[7],
            "canonical_journal": r[8],
        }
        for r in rows
    ]
