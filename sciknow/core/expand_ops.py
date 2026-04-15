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


# ── Phase 54.6.4 — three new expansion methods ──────────────────────────
#
# Each returns the same {"candidates": [...], "info": {...}} shape as
# ``find_author_candidates`` so the web preview modal can render them
# uniformly. Candidates carry DOI, title, authors, year, relevance_score.
#
# All three hit OpenAlex's /works endpoint with different filters. We
# reuse the HTTP settings from ``ingestion.author_search`` (User-Agent,
# polite-pool mailto) rather than inventing a new client.

import requests as _requests
import time as _time


_OPENALEX_PER_PAGE = 200
_HTTP_TIMEOUT = 30


def _openalex_headers_params():
    headers = {
        "User-Agent": f"sciknow ({settings.crossref_email or 'noreply@example.com'})"
    }
    params_base: dict[str, Any] = {"per-page": _OPENALEX_PER_PAGE}
    if settings.crossref_email:
        params_base["mailto"] = settings.crossref_email
    return headers, params_base


def _work_to_ref_dict(work: dict) -> dict[str, Any] | None:
    """Convert an OpenAlex /works record into our candidate dict shape.

    Drops works without a DOI (download pipeline can't route without one).
    """
    doi_url = (work.get("doi") or "").strip()
    if not doi_url:
        return None
    # OpenAlex returns DOIs as "https://doi.org/10.xxxx/..." — strip prefix.
    doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "")
    title = work.get("title") or work.get("display_name") or ""
    year = work.get("publication_year")
    authors = []
    for a in (work.get("authorships") or [])[:20]:
        nm = (a.get("author") or {}).get("display_name")
        if nm:
            authors.append(nm)
    return {
        "doi": doi.strip(),
        "arxiv_id": None,
        "title": title,
        "authors": authors,
        "year": year if isinstance(year, int) else None,
        "relevance_score": None,  # set later by the relevance-scoring pass
        "openalex_id": (work.get("id") or "").rsplit("/", 1)[-1],
        "cited_by_count": work.get("cited_by_count") or 0,
    }


def _paginate_works(filter_value: str, *, limit: int, delay: float = 0.2,
                    extra_params: dict[str, Any] | None = None,
                    sort: str | None = None) -> list[dict]:
    """Cursor-paginate /works?filter=... until we have `limit` results."""
    headers, params_base = _openalex_headers_params()
    params = dict(params_base)
    params["filter"] = filter_value
    if sort:
        params["sort"] = sort
    if extra_params:
        params.update(extra_params)
    params["cursor"] = "*"
    out: list[dict] = []
    while True:
        try:
            r = _requests.get(
                "https://api.openalex.org/works",
                params=params, headers=headers, timeout=_HTTP_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            logger.warning("OpenAlex /works paginate failed: %s", exc)
            break
        results = data.get("results") or []
        out.extend(results)
        nxt = (data.get("meta") or {}).get("next_cursor")
        if not nxt or not results or (limit > 0 and len(out) >= limit):
            break
        params["cursor"] = nxt
        _time.sleep(delay)
    if limit > 0:
        out = out[:limit]
    return out


def _score_and_dedup(raw: list[dict], *, relevance_query: str,
                     score_relevance: bool) -> tuple[list[dict], int, str | None]:
    """Shared tail for all three methods: dedup vs corpus by DOI, score
    against corpus centroid (or user query), sort. Returns
    (candidates, dedup_dropped, relevance_query_used).
    """
    # Dedup against existing corpus by DOI.
    with get_session() as session:
        existing = session.execute(sql_text(
            "SELECT LOWER(doi) FROM paper_metadata WHERE doi IS NOT NULL"
        )).fetchall()
    existing_dois = {r[0] for r in existing}
    before = len(raw)
    raw = [c for c in raw if c.get("doi") and c["doi"].lower() not in existing_dois]
    dedup_dropped = before - len(raw)

    relevance_query_used: str | None = None
    if score_relevance and raw:
        try:
            from sciknow.retrieval.relevance import (
                compute_corpus_centroid, embed_query, score_candidates,
            )
            anchor = (
                embed_query(relevance_query)
                if (relevance_query or "").strip()
                else compute_corpus_centroid()
            )
            titles = [c["title"] or "" for c in raw]
            scores = score_candidates(titles, anchor)
            for c, s in zip(raw, scores):
                c["relevance_score"] = float(s)
            relevance_query_used = (
                relevance_query.strip() if (relevance_query or "").strip()
                else "centroid"
            )
        except Exception as exc:
            logger.warning("relevance scoring failed: %s", exc)

    raw.sort(
        key=lambda c: (
            c.get("relevance_score") if c.get("relevance_score") is not None else -1.0,
            c.get("year") or 0,
        ),
        reverse=True,
    )
    return raw, dedup_dropped, relevance_query_used


def find_inbound_citation_candidates(
    *, per_seed_cap: int = 50, total_limit: int = 500,
    relevance_query: str = "", score_relevance: bool = True,
) -> dict[str, Any]:
    """Find papers that **cite** papers already in the corpus.

    For each of the corpus's OpenAlex-indexed works, query
    ``/works?filter=cites:W123``, pull up to ``per_seed_cap`` citers,
    union across seeds, dedup by DOI, relevance-score, sort.
    """
    # Step 1: resolve existing corpus DOIs to OpenAlex work IDs.
    with get_session() as session:
        existing_dois = [
            r[0] for r in session.execute(sql_text(
                "SELECT doi FROM paper_metadata WHERE doi IS NOT NULL "
                "ORDER BY created_at DESC LIMIT 500"
            )).fetchall() if r[0]
        ]
    if not existing_dois:
        return {"candidates": [], "info": {
            "seeds": 0, "raw": 0, "dedup_dropped": 0,
            "message": "corpus has no DOIs — nothing to expand from",
        }}

    headers, params_base = _openalex_headers_params()
    seed_work_ids: list[str] = []
    # Batch-resolve DOIs → work IDs via /works?filter=doi:...
    # OpenAlex caps filter|... at ~50 per request.
    batch_size = 50
    for i in range(0, len(existing_dois), batch_size):
        batch = existing_dois[i:i + batch_size]
        params = dict(params_base)
        params["filter"] = "doi:" + "|".join(batch)
        try:
            r = _requests.get(
                "https://api.openalex.org/works",
                params=params, headers=headers, timeout=_HTTP_TIMEOUT,
            )
            r.raise_for_status()
            results = (r.json() or {}).get("results") or []
            for w in results:
                wid = (w.get("id") or "").rsplit("/", 1)[-1]
                if wid:
                    seed_work_ids.append(wid)
        except Exception as exc:
            logger.warning("doi→work_id batch failed: %s", exc)
        _time.sleep(0.1)

    # Step 2: for each seed work, fetch up to per_seed_cap citers.
    raw_by_doi: dict[str, dict] = {}
    for i, wid in enumerate(seed_work_ids):
        if total_limit and len(raw_by_doi) >= total_limit:
            break
        works = _paginate_works(f"cites:{wid}", limit=per_seed_cap)
        for w in works:
            d = _work_to_ref_dict(w)
            if d and d["doi"] not in raw_by_doi:
                raw_by_doi[d["doi"]] = d
        _time.sleep(0.05)

    raw_list = list(raw_by_doi.values())
    cands, dropped, rq_used = _score_and_dedup(
        raw_list, relevance_query=relevance_query,
        score_relevance=score_relevance,
    )
    return {"candidates": cands, "info": {
        "seeds_resolved": len(seed_work_ids),
        "seeds_requested": len(existing_dois),
        "raw": len(raw_list),
        "dedup_dropped": dropped,
        "relevance_query_used": rq_used,
    }}


def find_topic_candidates(
    query: str, *, limit: int = 500,
    relevance_query: str = "", score_relevance: bool = True,
) -> dict[str, Any]:
    """Free-text OpenAlex /works search, ranked by citation count."""
    if not query.strip():
        raise ValueError("query required")
    headers, params_base = _openalex_headers_params()
    params = dict(params_base)
    params["search"] = query.strip()
    params["sort"] = "cited_by_count:desc"
    params["cursor"] = "*"
    raw: list[dict] = []
    while True:
        try:
            r = _requests.get(
                "https://api.openalex.org/works",
                params=params, headers=headers, timeout=_HTTP_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            logger.warning("topic /works search failed: %s", exc)
            break
        for w in data.get("results") or []:
            d = _work_to_ref_dict(w)
            if d:
                raw.append(d)
        nxt = (data.get("meta") or {}).get("next_cursor")
        if not nxt or (limit > 0 and len(raw) >= limit):
            break
        params["cursor"] = nxt
        _time.sleep(0.2)
    if limit > 0:
        raw = raw[:limit]

    # Default relevance anchor to the query itself for topic search —
    # the user's query is literally the anchor.
    rq = relevance_query if relevance_query.strip() else query
    cands, dropped, rq_used = _score_and_dedup(
        raw, relevance_query=rq, score_relevance=score_relevance,
    )
    return {"candidates": cands, "info": {
        "query": query, "raw": len(raw),
        "dedup_dropped": dropped,
        "relevance_query_used": rq_used,
    }}


def find_candidates_for_book_gaps(
    book_id: str, *, per_gap_limit: int = 100,
    only_types: tuple[str, ...] = ("topic", "evidence"),
    score_relevance: bool = True,
) -> dict[str, Any]:
    """Phase 54.6.5 — Auto-expand from book gaps.

    For every open ``BookGap`` on the book whose ``gap_type`` is in
    ``only_types``, use the gap's ``description`` as a topic query,
    fetch candidates via ``find_topic_candidates``, merge across all
    gaps (dedup by DOI while remembering which gap(s) each candidate
    addresses), dedup against the corpus, relevance-score against the
    corpus centroid, and return a unified candidate list.

    The key UX affordance: each candidate carries a ``gap_ids`` list
    of the gap UUIDs it was fetched for, so the UI can display
    "addresses N of your open gaps" and the user can prioritise papers
    that close multiple gaps at once.
    """
    if not book_id:
        raise ValueError("book_id required")

    with get_session() as session:
        rows = session.execute(sql_text("""
            SELECT id::text, gap_type, description
            FROM book_gaps
            WHERE book_id = :b AND status = 'open'
              AND gap_type = ANY(:types)
              AND description IS NOT NULL AND length(trim(description)) > 5
            ORDER BY created_at DESC
        """), {"b": book_id, "types": list(only_types)}).fetchall()

    if not rows:
        return {"candidates": [], "info": {
            "gaps_total": 0, "gaps_processed": 0,
            "message": "no open gaps of the requested type(s)",
        }}

    # Per-gap fetch. We DON'T score each sub-query against its own gap
    # (expensive and noisy); we score the merged list once against the
    # corpus centroid afterwards so the same threshold maps consistently
    # across all three expand flows.
    merged: dict[str, dict] = {}
    per_gap_info: list[dict] = []
    raw_total = 0
    for gap_id, gap_type, desc in rows:
        try:
            sub = find_topic_candidates(
                desc, limit=per_gap_limit,
                relevance_query="",  # pass-through
                score_relevance=False,  # we re-score merged set below
            )
        except Exception as exc:
            logger.warning("gap topic-search failed (%s): %s", gap_id, exc)
            continue
        n_sub = len(sub.get("candidates") or [])
        raw_total += n_sub
        per_gap_info.append({
            "gap_id": gap_id, "gap_type": gap_type,
            "description": desc[:140], "n_raw": n_sub,
        })
        for c in sub.get("candidates") or []:
            doi = c.get("doi")
            if not doi:
                continue
            if doi in merged:
                merged[doi]["gap_ids"].append(gap_id)
            else:
                mc = dict(c)
                mc["gap_ids"] = [gap_id]
                merged[doi] = mc

    raw_list = list(merged.values())
    cands, dropped, rq_used = _score_and_dedup(
        raw_list, relevance_query="", score_relevance=score_relevance,
    )
    # Secondary sort: more gaps addressed = more valuable.
    cands.sort(
        key=lambda c: (
            len(c.get("gap_ids") or []),
            c.get("relevance_score") if c.get("relevance_score") is not None else -1.0,
            c.get("year") or 0,
        ),
        reverse=True,
    )
    return {"candidates": cands, "info": {
        "gaps_total": len(rows),
        "gaps_processed": len(per_gap_info),
        "per_gap": per_gap_info,
        "raw": raw_total,
        "merged": len(raw_list),
        "dedup_dropped": dropped,
        "relevance_query_used": rq_used,
    }}


def find_coauthor_candidates(
    *, depth: int = 1, per_author_cap: int = 10, total_limit: int = 500,
    relevance_query: str = "", score_relevance: bool = True,
) -> dict[str, Any]:
    """Fetch papers by every OpenAlex author already in the corpus.

    depth=1: papers by authors already on corpus papers.
    depth=2: ALSO coauthors-of-coauthors — can explode, use carefully.
    """
    if depth not in (1, 2):
        raise ValueError("depth must be 1 or 2")
    # Step 1: enumerate corpus authors via OpenAlex author IDs stored
    # in crossref_raw / authors JSONB. Cheapest path: query OpenAlex
    # /works?filter=doi:... on the existing corpus DOIs and extract
    # authorships[].author.id.
    with get_session() as session:
        existing_dois = [
            r[0] for r in session.execute(sql_text(
                "SELECT doi FROM paper_metadata WHERE doi IS NOT NULL "
                "ORDER BY created_at DESC LIMIT 300"
            )).fetchall() if r[0]
        ]
    if not existing_dois:
        return {"candidates": [], "info": {
            "seed_authors": 0, "message": "corpus has no DOIs"
        }}

    headers, params_base = _openalex_headers_params()
    author_ids: set[str] = set()
    batch_size = 50
    for i in range(0, len(existing_dois), batch_size):
        batch = existing_dois[i:i + batch_size]
        params = dict(params_base)
        params["filter"] = "doi:" + "|".join(batch)
        try:
            r = _requests.get(
                "https://api.openalex.org/works",
                params=params, headers=headers, timeout=_HTTP_TIMEOUT,
            )
            r.raise_for_status()
            for w in (r.json() or {}).get("results") or []:
                for a in (w.get("authorships") or []):
                    aid = ((a.get("author") or {}).get("id") or "").rsplit("/", 1)[-1]
                    if aid:
                        author_ids.add(aid)
        except Exception as exc:
            logger.warning("doi→authors batch failed: %s", exc)
        _time.sleep(0.1)

    # Step 2: for each author, fetch their works (capped).
    raw_by_doi: dict[str, dict] = {}
    for aid in list(author_ids):
        if total_limit and len(raw_by_doi) >= total_limit:
            break
        works = _paginate_works(
            f"author.id:{aid}", limit=per_author_cap,
            sort="cited_by_count:desc",
        )
        for w in works:
            d = _work_to_ref_dict(w)
            if d and d["doi"] not in raw_by_doi:
                raw_by_doi[d["doi"]] = d
        _time.sleep(0.05)

    raw_list = list(raw_by_doi.values())
    cands, dropped, rq_used = _score_and_dedup(
        raw_list, relevance_query=relevance_query,
        score_relevance=score_relevance,
    )
    return {"candidates": cands, "info": {
        "seed_authors": len(author_ids),
        "raw": len(raw_list),
        "dedup_dropped": dropped,
        "relevance_query_used": rq_used,
    }}
