"""Phase 54.6.111 (Tier 1 #1) — persist OpenAlex enrichment.

Single-call enricher that hydrates the ``paper_metadata.oa_*`` columns
from a single ``/works/{id}`` fetch. Zero additional API load over what
``db expand`` / ``db enrich`` already does — we were fetching and
discarding these fields.

See ``docs/EXPAND_ENRICH_RESEARCH_2.md`` §1.1.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def extract_openalex_enrichment(work: dict | None) -> dict[str, Any]:
    """Pull the useful extras from an OpenAlex work response.

    Returns a dict of column updates (all JSONB / Integer / Timestamp);
    safe to splat into an UPDATE statement. Returns an empty dict when
    the work is None / empty.
    """
    if not work or not isinstance(work, dict):
        return {}

    out: dict[str, Any] = {}

    # Concepts — keep the most useful fields (display_name, level, score)
    # so we can do Jaccard + keyword browsing without the full payload.
    concepts = work.get("concepts") or []
    if concepts:
        out["oa_concepts"] = [
            {
                "display_name": c.get("display_name"),
                "level": c.get("level"),
                "score": c.get("score"),
                "wikidata": c.get("wikidata"),
            }
            for c in concepts if c.get("display_name")
        ][:30]  # 30 is plenty; OpenAlex usually returns 5-15

    # Funders (from grants) — {name, id}
    grants = work.get("grants") or []
    if grants:
        funders = {}
        grant_list = []
        for g in grants:
            fid = g.get("funder")
            fname = g.get("funder_display_name")
            if fid and fname:
                funders[fid] = fname
            if g.get("award_id"):
                grant_list.append({
                    "funder": fid,
                    "funder_name": fname,
                    "award_id": g.get("award_id"),
                })
        if funders:
            out["oa_funders"] = [{"id": fid, "name": fname}
                                 for fid, fname in funders.items()]
        if grant_list:
            out["oa_grants"] = grant_list

    # Institution ROR IDs from authorships — unique, ordered by first
    # appearance so the list stays deterministic.
    authorships = work.get("authorships") or []
    seen_ror: list[str] = []
    seen_set: set[str] = set()
    for a in authorships:
        for inst in (a.get("institutions") or []):
            ror = inst.get("ror")
            if ror and ror not in seen_set:
                seen_set.add(ror)
                seen_ror.append(ror)
    if seen_ror:
        out["oa_institutions_ror"] = seen_ror[:25]

    # Phase 54.6.221 (roadmap 3.2.4) — rich institution records for
    # the paper_institutions table. One row per (author_position,
    # institution). Preserves display_name, country_code, and
    # institution_type that oa_institutions_ror drops. Caller writes
    # these into paper_institutions separately from the oa_* columns
    # on paper_metadata.
    institutions_out: list[dict] = []
    for a in authorships:
        pos = a.get("author_position")
        # OpenAlex uses string positions ("first" / "middle" / "last")
        # in some responses and integer indices in others. Normalise
        # to int-or-None; unknown positions round-trip as NULL.
        pos_int: int | None = None
        if isinstance(pos, int):
            pos_int = pos
        elif isinstance(pos, str):
            mapping = {"first": 1, "middle": 2, "last": 3}
            pos_int = mapping.get(pos.lower())
        for inst in (a.get("institutions") or []):
            name = (inst.get("display_name") or "").strip()
            if not name:
                continue
            institutions_out.append({
                "ror_id": inst.get("ror"),
                "display_name": name[:300],
                "country_code": (inst.get("country_code") or None),
                "institution_type": (inst.get("type") or None),
                "author_position": pos_int,
            })
    # Cap at 50 rows per paper — a realistic upper bound for authorships.
    if institutions_out:
        out["_institutions"] = institutions_out[:50]

    # Citation counts
    cbc = work.get("cited_by_count")
    if isinstance(cbc, int):
        out["oa_cited_by_count"] = cbc
    cby = work.get("counts_by_year") or []
    if cby:
        out["oa_counts_by_year"] = [
            {"year": c.get("year"), "cited_by_count": c.get("cited_by_count")}
            for c in cby if c.get("year") is not None
        ][:15]  # 15 years is enough

    # Biblio — volume/issue/first_page/last_page
    b = work.get("biblio") or {}
    cleaned = {k: b.get(k) for k in ("volume", "issue", "first_page", "last_page")
               if b.get(k) not in (None, "")}
    if cleaned:
        out["oa_biblio"] = cleaned

    out["oa_enriched_at"] = datetime.now(timezone.utc)
    return out


def apply_openalex_enrichment(session, paper_id: str, work: dict | None) -> bool:
    """Hydrate a paper's oa_* columns from an OpenAlex work dict.

    ``paper_id`` is the ``paper_metadata.id`` UUID as a string. Returns
    True if any columns were updated, False if the work was empty.
    Caller is responsible for commit.
    """
    from sqlalchemy import text

    updates = extract_openalex_enrichment(work)
    if not updates:
        return False

    # Phase 54.6.221 — pull out the rich institutions list before the
    # generic UPDATE loop, because institutions go into a different
    # table (paper_institutions), not a paper_metadata column. The
    # underscore-prefixed key guarantees it can't collide with a real
    # column name.
    institutions_records: list[dict] = updates.pop("_institutions", []) or []

    import json
    # JSONB columns take dicts/lists; psycopg serializes them. Timestamp
    # is a datetime object. Integer is already an int.
    set_parts = []
    params: dict[str, Any] = {"pid": paper_id}
    for k, v in updates.items():
        if isinstance(v, (dict, list)):
            set_parts.append(f"{k} = CAST(:{k} AS jsonb)")
            params[k] = json.dumps(v)
        else:
            set_parts.append(f"{k} = :{k}")
            params[k] = v
    if set_parts:
        sql = (
            "UPDATE paper_metadata SET "
            + ", ".join(set_parts)
            + " WHERE id::text = :pid"
        )
        session.execute(text(sql), params)

    # Phase 54.6.221 — sync paper_institutions from the rich records.
    # Replace-all semantics: delete prior rows for this document, then
    # insert the new set. Idempotent across re-runs; simpler than
    # diffing. Looks up document_id from paper_metadata.id so callers
    # don't need to pass it separately.
    if institutions_records:
        doc_row = session.execute(text(
            "SELECT document_id::text FROM paper_metadata "
            "WHERE id::text = :pid"
        ), {"pid": paper_id}).fetchone()
        if doc_row and doc_row[0]:
            doc_id = doc_row[0]
            session.execute(text(
                "DELETE FROM paper_institutions "
                "WHERE document_id::text = :did"
            ), {"did": doc_id})
            for rec in institutions_records:
                session.execute(text("""
                    INSERT INTO paper_institutions
                        (document_id, ror_id, display_name,
                         country_code, institution_type,
                         author_position)
                    VALUES
                        (CAST(:did AS uuid), :ror_id, :display_name,
                         :country_code, :institution_type,
                         :author_position)
                """), {"did": doc_id, **rec})
    return True
