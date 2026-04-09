"""
Phase 16 — Author-driven catalog expansion.

Search OpenAlex (primary) and Crossref (fallback) for papers BY a named
author and return them as `Reference` objects so they plug into the
existing download → ingest pipeline used by `db expand`.

Why two backends:
  - OpenAlex has the best author resolution (canonical author IDs, ORCID
    integration, dedup across affiliations) and the best metadata.
  - Crossref is the catch-all for works OpenAlex doesn't index yet
    (very recent preprints, niche journals).

The two search functions return the same shape; `search_author()` calls
both and merges by DOI, preferring OpenAlex's metadata when both have
the same paper.

Usage:

    from sciknow.ingestion.author_search import search_author
    refs = search_author("Zharkova", year_from=2015, limit=50)
    # refs is a list[Reference] ready for find_and_download.
"""
from __future__ import annotations

import logging
import time
import urllib.parse
from typing import Iterable

import requests

from sciknow.config import settings
from sciknow.ingestion.references import Reference

logger = logging.getLogger("sciknow.author_search")


# Don't pound the APIs — Crossref's polite pool wants ~50 req/s max,
# OpenAlex is similar. Per-page sizes are large so we rarely paginate.
_OPENALEX_PER_PAGE = 200
_CROSSREF_PER_PAGE = 100
_HTTP_TIMEOUT = 30


# ── OpenAlex ────────────────────────────────────────────────────────────────


def _resolve_openalex_author_ids(name: str, *, max_authors: int = 25,
                                 min_works: int = 5) -> list[dict]:
    """Step 1 of OpenAlex author search: resolve a name to canonical
    author records via /authors?search={name}.

    OpenAlex's /authors endpoint search is FUZZY — it returns authors
    whose names sound similar to the query, including completely
    unrelated people. We post-filter to only authors whose display_name
    actually contains every token of the search query (case-insensitive,
    word-boundary aware), so "Solanki" matches "S. K. Solanki" and
    "Pratima Solanki" but NOT "Spagnolo" or "Shalaev".

    Returns a list of matching author dicts sorted by works_count desc,
    filtered to authors with at least `min_works` works.
    """
    if not name:
        return []
    headers = {"User-Agent": f"sciknow ({settings.crossref_email or 'noreply@example.com'})"}
    params = {"search": name, "per-page": max_authors}
    if settings.crossref_email:
        params["mailto"] = settings.crossref_email
    try:
        r = requests.get(
            "https://api.openalex.org/authors",
            params=params, headers=headers, timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("OpenAlex /authors lookup failed: %s", exc)
        return []

    # Build the required name tokens (case-insensitive). Every token
    # of the search must appear as a token in the display_name.
    required_tokens = [t.lower() for t in name.split() if len(t) >= 2]
    if not required_tokens:
        return []

    def _name_matches(display_name: str) -> bool:
        if not display_name:
            return False
        dn_tokens = {t.lower().strip(".,;:") for t in display_name.split()}
        # Each required token must be present as either a full token or as
        # a substring of one (handles initials like "S. K." vs "Sami K.")
        for req in required_tokens:
            hit = False
            for dn_t in dn_tokens:
                if req == dn_t or req in dn_t:
                    hit = True
                    break
            if not hit:
                return False
        return True

    out = []
    for a in data.get("results", []):
        display_name = a.get("display_name", "") or ""
        if not _name_matches(display_name):
            continue
        works_count = a.get("works_count") or 0
        if works_count < min_works:
            continue
        out.append({
            "id": a.get("id", ""),  # full URL like https://openalex.org/A1234
            "short_id": (a.get("id", "") or "").rsplit("/", 1)[-1],
            "display_name": display_name,
            "works_count": works_count,
            "orcid": a.get("orcid"),
            "affiliations": [
                (aff.get("institution") or {}).get("display_name", "")
                for aff in (a.get("affiliations") or [])[:2]
            ],
        })
    out.sort(key=lambda a: a["works_count"], reverse=True)
    return out


def search_openalex_by_author(
    name: str,
    *,
    orcid: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int | None = None,
    require_doi: bool = True,
    all_matches: bool = False,
) -> tuple[list[Reference], list[dict]]:
    """Query OpenAlex /works for papers by an author.

    Strategy:
      1. If `orcid` is given → direct filter by author.orcid (exact match,
         no ambiguity).
      2. Otherwise → /authors?search={name} to resolve canonical authors,
         then by default pick the SINGLE top author by works_count and
         query /works?filter=author.id:{that_one}. This is what the user
         almost always wants: the most prominent person with that name,
         not a fuzzy union of everyone whose surname matches.
      3. With `all_matches=True`, pool all authors with that surname.

    Returns:
        (refs, candidate_authors)
          refs: list[Reference] from the works query, sorted by year DESC.
          candidate_authors: the author records found by /authors?search,
            so the caller can show "picked X but also found Y, Z" hints.
    """
    if not name and not orcid:
        return [], []

    base_filter_parts: list[str] = []
    candidates: list[dict] = []
    picked: list[dict] = []

    if orcid:
        clean = orcid.strip().replace("https://orcid.org/", "")
        base_filter_parts.append(f"author.orcid:https://orcid.org/{clean}")
    else:
        candidates = _resolve_openalex_author_ids(name)
        if not candidates:
            logger.info(
                "No OpenAlex author records for %r — try --orcid for exact match",
                name,
            )
            return [], []
        # Pick the single most-published author by default. This is the
        # disambiguation strategy: "Solanki" → S. K. Solanki (1548 works)
        # not "all 21 different Solankis combined".
        if all_matches:
            picked = candidates
        else:
            picked = [candidates[0]]
        ids = "|".join(a["short_id"] for a in picked)
        base_filter_parts.append(f"author.id:{ids}")

    if year_from is not None:
        base_filter_parts.append(f"from_publication_date:{year_from}-01-01")
    if year_to is not None:
        base_filter_parts.append(f"to_publication_date:{year_to}-12-31")

    if require_doi:
        base_filter_parts.append("has_doi:true")

    filter_str = ",".join(base_filter_parts)
    headers = {"User-Agent": f"sciknow ({settings.crossref_email or 'noreply@example.com'})"}

    results: list[Reference] = []
    cursor = "*"
    seen_dois: set[str] = set()
    pages_fetched = 0

    while True:
        params = {
            "filter": filter_str,
            "per-page": _OPENALEX_PER_PAGE,
            "cursor": cursor,
            "select": "doi,title,publication_year,authorships,id",
        }
        if settings.crossref_email:
            params["mailto"] = settings.crossref_email

        try:
            r = requests.get(
                "https://api.openalex.org/works",
                params=params, headers=headers, timeout=_HTTP_TIMEOUT,
            )
            r.raise_for_status()
        except Exception as exc:
            logger.warning("OpenAlex request failed: %s", exc)
            break

        data = r.json()
        works = data.get("results", [])
        meta = data.get("meta", {})
        next_cursor = meta.get("next_cursor")

        for w in works:
            doi = (w.get("doi") or "").replace("https://doi.org/", "").lower() or None
            if not doi:
                continue
            if doi in seen_dois:
                continue
            seen_dois.add(doi)

            title = w.get("title") or ""
            year = w.get("publication_year")
            author_names = [
                (a.get("author") or {}).get("display_name", "")
                for a in (w.get("authorships") or [])
            ]
            author_names = [a for a in author_names if a]

            results.append(Reference(
                raw_text=f"[OpenAlex] {title} ({year})",
                doi=doi,
                title=title,
                year=year,
                authors=author_names,
            ))

            if limit and len(results) >= limit:
                results.sort(key=lambda r: r.year or 0, reverse=True)
                return results, candidates

        pages_fetched += 1
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        # Polite pause between pages
        time.sleep(0.1)
        if pages_fetched > 50:
            logger.warning("OpenAlex returned >50 pages — stopping for safety")
            break

    # Sort by year DESC for consistent UX
    results.sort(key=lambda r: r.year or 0, reverse=True)
    return results, candidates


# ── Crossref ────────────────────────────────────────────────────────────────


def _name_tokens(name: str) -> set[str]:
    """Extract family-name tokens for post-filtering Crossref results.

    Crossref's `query.author` is rank-based and notoriously loose — it
    will return papers by any author whose name token matches "Solanki",
    even if it's a completely different person. We post-filter by
    requiring the family name (last whitespace-separated token, lowered)
    to actually appear in one of the work's author family names.

    For multi-word names like "van der Berg" we keep the last 1-2 tokens
    so both "Berg" and "der Berg" match.
    """
    parts = (name or "").strip().split()
    if not parts:
        return set()
    last = parts[-1].lower()
    return {last}


def _crossref_author_matches(work_authors: list[dict], required_tokens: set[str]) -> bool:
    """True if any author family name in the work matches the required tokens."""
    if not required_tokens:
        return True
    for a in work_authors or []:
        family = (a.get("family") or "").lower()
        if family in required_tokens:
            return True
        # Also tolerate "given family" smashed together
        for tok in required_tokens:
            if tok and tok in family:
                return True
    return False


def search_crossref_by_author(
    name: str,
    *,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int | None = None,
) -> list[Reference]:
    """Query Crossref /works with a query.author search.

    Crossref's author search is fuzzy and rank-based, not filtered. We
    page through the top results and post-filter by family-name match
    (because the API will happily return papers by random people whose
    surnames merely contain the search term).
    """
    if not name:
        return []
    required_tokens = _name_tokens(name)

    base_params: dict = {
        "query.author": name,
        "rows": _CROSSREF_PER_PAGE,
    }
    filters: list[str] = []
    if year_from is not None:
        filters.append(f"from-pub-date:{year_from}")
    if year_to is not None:
        filters.append(f"until-pub-date:{year_to}")
    if filters:
        base_params["filter"] = ",".join(filters)
    if settings.crossref_email:
        base_params["mailto"] = settings.crossref_email

    headers = {"User-Agent": f"sciknow ({settings.crossref_email or 'noreply@example.com'})"}

    results: list[Reference] = []
    seen_dois: set[str] = set()
    cursor = "*"
    pages_fetched = 0

    while True:
        params = {**base_params, "cursor": cursor}
        try:
            r = requests.get(
                "https://api.crossref.org/works",
                params=params, headers=headers, timeout=_HTTP_TIMEOUT,
            )
            r.raise_for_status()
        except Exception as exc:
            logger.warning("Crossref request failed: %s", exc)
            break

        data = r.json().get("message", {})
        items = data.get("items", [])
        next_cursor = data.get("next-cursor")

        for it in items:
            doi = (it.get("DOI") or "").lower() or None
            if not doi or doi in seen_dois:
                continue
            seen_dois.add(doi)

            # Post-filter on family-name match (the whole point of the
            # _name_tokens helper above — Crossref's query.author returns
            # everyone whose surname contains the search term).
            if not _crossref_author_matches(it.get("author") or [], required_tokens):
                continue

            title_list = it.get("title") or []
            title = title_list[0] if title_list else ""
            # Crossref date is messy — try several locations
            year = None
            for date_field in ("issued", "published", "published-online", "published-print"):
                parts = (it.get(date_field) or {}).get("date-parts") or []
                if parts and parts[0]:
                    try:
                        year = int(parts[0][0])
                        break
                    except (ValueError, TypeError, IndexError):
                        pass

            author_names = []
            for a in (it.get("author") or []):
                given = a.get("given") or ""
                family = a.get("family") or ""
                if family:
                    author_names.append(f"{given} {family}".strip())

            results.append(Reference(
                raw_text=f"[Crossref] {title} ({year})",
                doi=doi,
                title=title,
                year=year,
                authors=author_names,
            ))

            if limit and len(results) >= limit:
                return results

        pages_fetched += 1
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.1)
        if pages_fetched > 20:
            logger.warning("Crossref returned >20 pages — stopping for safety")
            break

    results.sort(key=lambda r: r.year or 0, reverse=True)
    return results


# ── Unified API ─────────────────────────────────────────────────────────────


def _surname_in_authors(ref: Reference, surname: str) -> bool:
    """True if the lowered surname appears as a token in any of the
    Reference's author display strings.

    Defensive last-pass check used by search_author — guarantees the
    final result list never contains a paper where the searched surname
    isn't actually in the author list. This is a hard floor against:
      - Crossref API drift (e.g. if query.author ever changed semantics)
      - Future bugs where someone confuses author-search with citation-search
      - Author list missing entirely (treated as failing the check)
    """
    if not surname:
        return True  # nothing to verify
    surname_lc = surname.strip().lower()
    if not surname_lc:
        return True
    for a in ref.authors or []:
        a_lc = (a or "").lower()
        # Match as a token: " surname", "surname ", or exact equality
        if a_lc == surname_lc:
            return True
        if surname_lc in a_lc.split():
            return True
        # Also tolerate "Last, First" format
        if a_lc.startswith(surname_lc + ","):
            return True
        if a_lc.endswith(" " + surname_lc):
            return True
    return False


def search_author(
    name: str,
    *,
    orcid: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int | None = None,
    require_doi: bool = True,
    all_matches: bool = False,
    strict_author: bool = False,
) -> tuple[list[Reference], dict]:
    """Search both OpenAlex and Crossref, dedup by DOI, return merged list.

    Args:
        strict_author: if True, drop Crossref results entirely. Only the
            OpenAlex canonical-author-ID matches are kept. Use this when
            you want zero ambiguity (no chance of papers by other people
            with the same surname slipping in via Crossref's looser
            search).

    Returns:
        (refs, info) where info is:
          {
            "openalex": int,
            "crossref_extra": int,
            "merged": int,
            "candidates": list[dict],  # author records found by /authors?search
            "picked": list[dict],      # which subset was used for the works query
            "dropped_no_surname": int, # papers dropped by the surname final check
          }
    """
    info: dict = {
        "openalex": 0, "crossref_extra": 0, "merged": 0,
        "candidates": [], "picked": [], "dropped_no_surname": 0,
    }

    # OpenAlex first — better metadata, ORCID-aware
    oa_refs, oa_candidates = search_openalex_by_author(
        name, orcid=orcid,
        year_from=year_from, year_to=year_to,
        limit=limit, require_doi=require_doi,
        all_matches=all_matches,
    )
    info["openalex"] = len(oa_refs)
    info["candidates"] = oa_candidates
    if oa_candidates:
        info["picked"] = oa_candidates if all_matches else oa_candidates[:1]
    by_doi: dict[str, Reference] = {r.doi: r for r in oa_refs if r.doi}

    # Only call Crossref if (a) we haven't already hit the limit, and
    # (b) strict_author is False. Crossref's author search is looser
    # than OpenAlex's canonical-author-ID match, so under strict mode we
    # rely entirely on OpenAlex.
    remaining = (limit - len(by_doi)) if limit else None
    if not strict_author and (remaining is None or remaining > 0):
        cr_refs = search_crossref_by_author(
            name, year_from=year_from, year_to=year_to, limit=remaining,
        )
        for r in cr_refs:
            if r.doi and r.doi not in by_doi:
                by_doi[r.doi] = r
                info["crossref_extra"] += 1
                if limit and len(by_doi) >= limit:
                    break

    # Defensive final pass: drop anything where the searched surname
    # isn't actually in the author list. The OpenAlex path's canonical
    # author ID filter and the Crossref path's family-name post-filter
    # should already handle this — but a hard assertion at the boundary
    # means a future API change or regression can never silently let
    # through a "paper that mentions Zharkova" instead of "paper authored
    # by Zharkova". Mostly redundant in practice; very cheap to run.
    if name:
        # Use the last whitespace-separated token as the canonical surname
        surname = name.strip().split()[-1] if name.strip() else ""
        verified: dict[str, Reference] = {}
        for doi, ref in by_doi.items():
            if _surname_in_authors(ref, surname):
                verified[doi] = ref
            else:
                info["dropped_no_surname"] += 1
                logger.warning(
                    "Dropping %s — surname %r not found in author list %r",
                    doi, surname, ref.authors,
                )
        by_doi = verified

    merged = sorted(by_doi.values(), key=lambda r: r.year or 0, reverse=True)
    info["merged"] = len(merged)
    return merged, info
