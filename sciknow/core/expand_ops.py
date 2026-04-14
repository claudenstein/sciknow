"""Phase 54.6.1 — preview-and-select helper for `sciknow db expand-author`.

Splits the pre-download phase of the CLI flow into a library call so the web
UI can:

  1. Preview candidate papers WITHOUT committing to a download.
  2. Let the user pick individually via checkboxes.
  3. Ship the selected DOIs off to the normal download + ingest pipeline.

The CLI (`sciknow/cli/db.py:expand_author`) still owns its own flow and is
unchanged — this module is used only by the web-facing preview endpoint.
Keep the two in sync when the search / dedup / relevance rules evolve.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text as sql_text

from sciknow.config import settings
from sciknow.ingestion.author_search import search_author
from sciknow.storage.db import get_session

logger = logging.getLogger(__name__)


def _ref_to_dict(ref, score: float | None = None) -> dict[str, Any]:
    return {
        "doi": ref.doi,
        "arxiv_id": ref.arxiv_id,
        "title": ref.title or "",
        "authors": list(ref.authors or []),
        "year": ref.year,
        "relevance_score": (None if score is None else float(score)),
    }


def find_author_candidates(
    name: str,
    *,
    orcid: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int = 0,
    all_matches: bool = False,
    strict_author: bool = True,
    relevance_query: str = "",
    score_relevance: bool = True,
) -> dict[str, Any]:
    """Run search + corpus-dedup + (optional) relevance scoring.

    Returns a JSON-serializable dict:

        {
          "candidates": [
              {"doi": "...", "title": "...", "authors": [...],
               "year": 2020, "relevance_score": 0.63 | null, ...},
              ...
          ],
          "info": {
              "openalex": int, "crossref_extra": int, "merged": int,
              "dedup_dropped": int, "dropped_no_surname": int,
              "relevance_threshold": float | null,
              "relevance_query_used": "centroid" | "<query>" | null,
              "picked_authors": [...], "candidate_authors": [...]
          }
        }

    Does NOT filter by relevance threshold — scores are annotated so the UI
    can sort / filter / checkbox-select. The existing "Run Expand-by-Author
    (auto)" path still uses the CLI's hard filter; this preview path shows
    everything and lets the user choose.
    """
    if not name.strip() and not orcid:
        raise ValueError("must provide either name or orcid")

    refs, info = search_author(
        name.strip(),
        orcid=(orcid or None),
        year_from=(year_from or None),
        year_to=(year_to or None),
        limit=(limit if limit > 0 else None),
        all_matches=all_matches,
        strict_author=strict_author,
    )

    # ── dedup against existing corpus by DOI ────────────────────────────
    with get_session() as session:
        existing = session.execute(sql_text(
            "SELECT LOWER(doi) FROM paper_metadata WHERE doi IS NOT NULL"
        )).fetchall()
    existing_dois = {r[0] for r in existing}
    before_dedup = len(refs)
    refs = [r for r in refs if r.doi and r.doi.lower() not in existing_dois]
    dedup_dropped = before_dedup - len(refs)

    # ── optional relevance scoring (annotate, don't filter) ─────────────
    scores: list[float | None] = [None] * len(refs)
    relevance_query_used: str | None = None
    threshold = getattr(settings, "expand_relevance_threshold", 0.55)
    if score_relevance and refs:
        try:
            from sciknow.retrieval.relevance import (
                compute_corpus_centroid, embed_query, score_candidates,
            )
            anchor_vec = (
                embed_query(relevance_query) if relevance_query.strip()
                else compute_corpus_centroid()
            )
            titles = [r.title or "" for r in refs]
            raw_scores = score_candidates(titles, anchor_vec)
            scores = [float(s) for s in raw_scores]
            relevance_query_used = (
                relevance_query.strip() if relevance_query.strip() else "centroid"
            )
        except Exception as exc:
            logger.warning("Relevance scoring failed in preview: %s", exc)

    # sort by score desc where available, else by year desc
    paired = list(zip(refs, scores))
    paired.sort(
        key=lambda p: (
            p[1] if p[1] is not None else -1.0,
            p[0].year or 0,
        ),
        reverse=True,
    )

    candidates = [_ref_to_dict(r, s) for r, s in paired]

    out_info: dict[str, Any] = {
        "openalex": info["openalex"],
        "crossref_extra": info["crossref_extra"],
        "merged": info["merged"],
        "dedup_dropped": dedup_dropped,
        "dropped_no_surname": info.get("dropped_no_surname", 0),
        "relevance_threshold": (
            float(threshold) if relevance_query_used else None
        ),
        "relevance_query_used": relevance_query_used,
        "picked_authors": info.get("picked", []),
        "candidate_authors": info.get("candidates", []),
    }
    return {"candidates": candidates, "info": out_info}
