"""Phase 49 — API clients used by the RRF-based expand ranker.

Thin wrappers over OpenAlex + Semantic Scholar Graph API:
- OpenAlex `/works/{id}` — full work metadata (refs, cited-by count,
  concepts, authors, counts_by_year, type, venue).
- OpenAlex `/works?filter=cites:{id}` — forward-citation set for a
  seed; powers the co-citation strength metric.
- Semantic Scholar `/graph/v1/paper/{id}/citations` — per-edge
  `isInfluential` + `intents`, backing the Valenzuela/meaningful-
  citations signal.

All calls are polite-pool: OpenAlex reads `settings.crossref_email`
as the `mailto` parameter (shared with the rest of ingestion);
Semantic Scholar runs at 1 RPS without a key. Network errors never
raise — they return None so the ranker can degrade one signal at a
time without failing the whole expand run.

Caching is in-process (simple dict) so that repeated lookups for the
same DOI during one expand invocation don't hit the wire twice. No
disk persistence — the batch size is bounded per run anyway, and
stale on-disk caches would mask upstream updates (retractions,
citation accrual) we actually want.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from sciknow.config import settings

logger = logging.getLogger("sciknow.expand_apis")

# ── OpenAlex ────────────────────────────────────────────────────────

_OPENALEX_BASE = "https://api.openalex.org"
_OA_WORK_CACHE: dict[str, dict | None] = {}
_OA_CITED_BY_CACHE: dict[str, list[dict]] = {}
_OA_LOCK = threading.Lock()


def _oa_url_for_work(doi_or_id: str) -> str:
    """Resolve a DOI or OpenAlex work-id to the canonical /works URL.
    OpenAlex accepts both `/works/W12345` and `/works/https://doi.org/10.xxxx/...`
    but rejects bare DOIs; the prefix is required."""
    ident = doi_or_id.strip()
    if not ident:
        return ""
    if ident.startswith("W") and ident[1:].isdigit():
        return f"{_OPENALEX_BASE}/works/{ident}"
    if ident.startswith("http"):
        return f"{_OPENALEX_BASE}/works/{ident}"
    return f"{_OPENALEX_BASE}/works/https://doi.org/{ident}"


def fetch_openalex_work(
    doi_or_id: str,
    *,
    timeout: float = 15.0,
    fields: str | None = None,
) -> dict | None:
    """Fetch a single OpenAlex work. Returns None on any error.

    `fields` (comma-separated, e.g. `"id,referenced_works,concepts"`)
    narrows the response to cut bandwidth — OpenAlex respects the
    `select` query param. The ranker uses the full set by default."""
    ident = doi_or_id.strip()
    if not ident:
        return None
    cache_key = ident.lower()
    with _OA_LOCK:
        if cache_key in _OA_WORK_CACHE:
            return _OA_WORK_CACHE[cache_key]
    url = _oa_url_for_work(ident)
    params: dict[str, str] = {}
    if settings.crossref_email:
        params["mailto"] = settings.crossref_email
    if fields:
        params["select"] = fields
    try:
        r = httpx.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
        else:
            data = None
    except Exception as exc:
        logger.debug("OpenAlex work fetch failed for %s: %s", ident, exc)
        data = None
    with _OA_LOCK:
        _OA_WORK_CACHE[cache_key] = data
    return data


def fetch_openalex_cited_by(
    seed_openalex_id: str,
    *,
    per_page: int = 200,
    max_pages: int = 2,
    timeout: float = 20.0,
) -> list[dict]:
    """Fetch the *forward-citation set* for a seed: papers that cite
    the seed. Used to compute co-citation strength — for every
    candidate C, count how many papers in ∪ (cited_by(seeds)) also
    cite C.

    Returns each paper's {id, referenced_works}; those are the only
    fields the co-citation math needs. Capped at `per_page × max_pages`
    (default 400) since OpenAlex paginates and we don't need every
    last citer for the signal to work."""
    if not seed_openalex_id:
        return []
    with _OA_LOCK:
        cached = _OA_CITED_BY_CACHE.get(seed_openalex_id)
    if cached is not None:
        return cached
    out: list[dict] = []
    for page in range(1, max_pages + 1):
        params: dict[str, Any] = {
            "filter": f"cites:{seed_openalex_id}",
            "per_page": per_page,
            "page": page,
            "select": "id,referenced_works",
        }
        if settings.crossref_email:
            params["mailto"] = settings.crossref_email
        try:
            r = httpx.get(f"{_OPENALEX_BASE}/works", params=params, timeout=timeout)
            if r.status_code != 200:
                break
            results = (r.json() or {}).get("results") or []
            if not results:
                break
            out.extend(results)
            if len(results) < per_page:
                break
        except Exception as exc:
            logger.debug("OpenAlex cited-by fetch failed (%s, page=%s): %s",
                         seed_openalex_id, page, exc)
            break
    with _OA_LOCK:
        _OA_CITED_BY_CACHE[seed_openalex_id] = out
    return out


# ── Semantic Scholar ────────────────────────────────────────────────

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_RATE_LIMIT_S = 1.05  # unauthenticated is 1 RPS; add 5% safety margin
_S2_LAST_CALL_TS: float = 0.0
_S2_CACHE: dict[str, dict | None] = {}
_S2_LOCK = threading.Lock()


def _s2_throttle() -> None:
    """Serialise S2 calls to respect the 1 RPS unauthenticated cap."""
    global _S2_LAST_CALL_TS
    with _S2_LOCK:
        now = time.monotonic()
        wait = _S2_LAST_CALL_TS + _S2_RATE_LIMIT_S - now
        if wait > 0:
            time.sleep(wait)
        _S2_LAST_CALL_TS = time.monotonic()


def fetch_s2_citations(
    doi: str,
    *,
    timeout: float = 15.0,
    limit: int = 1000,
) -> list[dict] | None:
    """Fetch S2 citations for a DOI with the per-edge influential
    flag + intent labels (Valenzuela et al. 2015's "meaningful
    citations" classifier exposed as a free field).

    Returns the `data` list from S2's response, where each entry has
    at least `{isInfluential: bool, intents: [str], contexts: [str],
    citingPaper: {paperId, externalIds, title}}`. None on error."""
    doi = (doi or "").strip()
    if not doi:
        return None
    cache_key = doi.lower()
    with _S2_LOCK:
        if cache_key in _S2_CACHE:
            return _S2_CACHE[cache_key].get("data") if _S2_CACHE[cache_key] else None
    _s2_throttle()
    url = f"{_S2_BASE}/paper/DOI:{doi}/citations"
    params = {
        "fields": "isInfluential,intents,contexts,citingPaper.externalIds",
        "limit": str(limit),
    }
    try:
        r = httpx.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            body = r.json()
        else:
            body = None
    except Exception as exc:
        logger.debug("S2 citations fetch failed for %s: %s", doi, exc)
        body = None
    with _S2_LOCK:
        _S2_CACHE[cache_key] = body
    return (body or {}).get("data") if body else None


# ── Corpus-scoped helpers (pure; no network) ────────────────────────

def count_influential_from_corpus(
    s2_citations: list[dict] | None,
    corpus_doi_set: set[str],
) -> int:
    """Given the S2 citation list for a candidate and the set of DOIs
    of papers already in our corpus, count how many of those corpus
    papers cite the candidate *with isInfluential=True OR an intent
    of method/result*. This is the Valenzuela-style signal the ranker
    feeds into RRF."""
    if not s2_citations or not corpus_doi_set:
        return 0
    corpus_doi_set = {d.lower() for d in corpus_doi_set if d}
    count = 0
    for edge in s2_citations:
        citing = edge.get("citingPaper") or {}
        ext = citing.get("externalIds") or {}
        cdoi = (ext.get("DOI") or "").lower()
        if cdoi and cdoi in corpus_doi_set:
            if edge.get("isInfluential") or any(
                i in ("method", "result") for i in (edge.get("intents") or [])
            ):
                count += 1
    return count


def clear_caches() -> None:
    """Drop in-process caches. Used only in tests — a fresh expand
    run gets a fresh cache per invocation anyway (process re-start)."""
    with _OA_LOCK:
        _OA_WORK_CACHE.clear()
        _OA_CITED_BY_CACHE.clear()
    with _S2_LOCK:
        _S2_CACHE.clear()
