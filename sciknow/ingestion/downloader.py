"""
Open-access PDF discovery and download.

Sources probed for every candidate DOI, in rough priority order:

  1. **Copernicus pattern** — deterministic URL construction for EGU
     journal DOIs (10.5194/*). Zero HTTP lookup needed, instant hit for
     any climate/earth-science paper in one of those journals.
  2. **arXiv** — direct PDF for any arXiv ID. Zero lookup; just
     constructs the canonical URL and attempts download.
  3. **Unpaywall** — largest general-purpose OA PDF database. One
     HTTP call; returns a curated "best OA location".
  4. **OpenAlex** — catches preprints + institutional repos that
     Unpaywall misses. Polite-pool friendly (up to 100k req/day with
     an email).
  5. **Europe PMC** — free full text for biomedical / environmental
     health papers that cross-index from PubMed Central.
  6. **Semantic Scholar** — public API with its own independent OA
     index.
  7. **HAL** — French open archive (hal.science). Strong coverage of
     European physics + astronomy + some climate.
  8. **Zenodo** — CERN-hosted DOI-resolving OA repository; catches
     dataset-attached papers + some preprints.

Phase 54.6.51 — two big changes to the discovery phase that used to
dominate wall time on no-OA DOIs:

  1. All URL-lookup calls now run in parallel via a bounded
     ThreadPoolExecutor. Pre-fix, for a DOI with no open access we
     burned ~6 × per-API latency (≈2-5 s) serially before giving up;
     now we burn ~1 × per-API latency since the slowest dominates.

  2. A single module-level ``httpx.Client`` is reused across all
     lookups. Pre-fix, each helper opened a fresh Client for every
     call, paying TLS handshake + DNS resolution on every request.
     The pooled client reuses keep-alive connections to unpaywall.org,
     openalex.org, etc., cutting per-request overhead to near zero.

Downloads themselves still run in priority order after discovery
(cheapest source first) — no bandwidth is wasted fetching the same
PDF from multiple origins.
"""
from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx


logger = logging.getLogger(__name__)


_HEADERS = {
    "User-Agent": "sciknow/0.1 (research use; https://github.com/claudenstein/sciknow)",
    "Accept": "application/pdf,*/*;q=0.9",
}

# Separate timeouts for metadata lookups vs PDF downloads. Metadata calls
# are pure JSON — if they take > 8 s something is wrong and we want to
# move on. PDF downloads can legitimately take 30-60 s for a large paper.
_LOOKUP_TIMEOUT = 8
_DOWNLOAD_TIMEOUT = 60
_MAX_PDF_MB = 50


# ────────────────────────────────────────────────────────────────────────
# Shared HTTP client (Phase 54.6.51)
# ────────────────────────────────────────────────────────────────────────


_CLIENT_LOCK = threading.Lock()
_CLIENT: httpx.Client | None = None


def _get_client() -> httpx.Client:
    """Lazily construct a process-wide httpx.Client with generous
    connection pooling + keep-alive. All lookup helpers use this client
    instead of opening their own — the TLS handshake + DNS resolution
    per request otherwise dominated the API-check wall time.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = httpx.Client(
                timeout=_LOOKUP_TIMEOUT,
                follow_redirects=True,
                headers=_HEADERS,
                limits=httpx.Limits(
                    max_keepalive_connections=32,
                    max_connections=64,
                    keepalive_expiry=30.0,
                ),
            )
    return _CLIENT


def close_shared_client() -> None:
    """Tear down the shared client. Call at end of CLI runs if you want
    to release sockets promptly; otherwise Python will clean up at exit."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            try:
                _CLIENT.close()
            except Exception:
                pass
            _CLIENT = None


# ────────────────────────────────────────────────────────────────────────
# Per-source URL lookups (each returns a candidate PDF URL or None)
# ────────────────────────────────────────────────────────────────────────


def find_oa_pdf_url(doi: str, email: str) -> str | None:
    """Unpaywall — largest general-purpose OA PDF database."""
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        resp = _get_client().get(url)
        if resp.status_code != 200:
            return None
        data = resp.json()
        loc = data.get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url") or None
    except Exception as exc:
        logger.debug("unpaywall lookup failed for %s: %s", doi, exc)
        return None


def find_arxiv_pdf_url(arxiv_id: str) -> str:
    """Return the canonical arXiv PDF URL for a given arXiv ID."""
    base = arxiv_id.split("v")[0]
    return f"https://arxiv.org/pdf/{base}.pdf"


def find_arxiv_id_by_doi(doi: str) -> str | None:
    """Phase 54.6.51 — resolve DOI → arXiv ID via arXiv's query API.

    Many journal-DOI papers also have an arXiv preprint under a different
    DOI (10.48550/arXiv.X); previously we only used arXiv when the caller
    already knew the arxiv_id. Now we'll query arXiv.org for any DOI that
    starts with something common and get the arxiv_id if there is one.
    """
    try:
        resp = _get_client().get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f"doi:{doi}", "max_results": 1},
        )
        if resp.status_code != 200:
            return None
        # Response is Atom XML; just regex out the ID — keeps us
        # dependency-free vs pulling in feedparser.
        m = re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", resp.text)
        if m:
            return m.group(1).strip()
        return None
    except Exception:
        return None


def find_semantic_scholar_pdf_url(doi: str) -> str | None:
    """Semantic Scholar — independent OA index."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {"fields": "openAccessPdf"}
    try:
        resp = _get_client().get(url, params=params)
        if resp.status_code != 200:
            return None
        data = resp.json()
        oa = data.get("openAccessPdf") or {}
        return oa.get("url") or None
    except Exception as exc:
        logger.debug("semantic scholar lookup failed for %s: %s", doi, exc)
        return None


def find_openalex_pdf_url(doi: str, email: str) -> str | None:
    """OpenAlex — preprints + institutional repos."""
    url = f"https://api.openalex.org/works/doi:{doi}"
    params = {"mailto": email}
    try:
        resp = _get_client().get(url, params=params)
        if resp.status_code != 200:
            return None
        data = resp.json()
        for key in ("best_oa_location", "primary_location"):
            loc = data.get(key) or {}
            pdf = loc.get("pdf_url")
            if pdf:
                return pdf
        for loc in data.get("oa_locations") or data.get("locations") or []:
            if not isinstance(loc, dict):
                continue
            pdf = loc.get("pdf_url")
            if pdf:
                return pdf
        return None
    except Exception as exc:
        logger.debug("openalex lookup failed for %s: %s", doi, exc)
        return None


# Copernicus / EGU journal DOI pattern:
#   10.5194/{journal}-{volume}-{start}-{year}
# PDF URL pattern:
#   https://{journal}.copernicus.org/articles/{volume}/{start}/{year}/{journal}-{volume}-{start}-{year}.pdf
_COPERNICUS_DOI_RE = re.compile(
    r"^10\.5194/([a-z]+)-(\d+)-(\d+)-(\d{4})$",
    re.IGNORECASE,
)


def find_copernicus_pdf_url(doi: str) -> str | None:
    """Deterministic URL for EGU / Copernicus journals — no API call."""
    m = _COPERNICUS_DOI_RE.match(doi.strip())
    if not m:
        return None
    journal, vol, start, year = (
        m.group(1).lower(), m.group(2), m.group(3), m.group(4),
    )
    return (
        f"https://{journal}.copernicus.org/articles/{vol}/{start}/{year}/"
        f"{journal}-{vol}-{start}-{year}.pdf"
    )


def find_europepmc_pdf_url(doi: str) -> str | None:
    """Europe PMC — free full text for biomedical / env-health papers."""
    search_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": f"DOI:{doi}", "format": "json", "resultType": "lite"}
    try:
        resp = _get_client().get(search_url, params=params)
        if resp.status_code != 200:
            return None
        data = resp.json()
        results = (data.get("resultList") or {}).get("result") or []
        if not results:
            return None
        hit = results[0]
        url_list = (hit.get("fullTextUrlList") or {}).get("fullTextUrl") or []
        for u in url_list:
            if (
                isinstance(u, dict)
                and u.get("documentStyle") == "pdf"
                and u.get("availability", "").lower() in ("open access", "free", "subscription")
            ):
                return u.get("url")
        pmcid = hit.get("pmcid")
        if pmcid:
            return f"https://europepmc.org/articles/{pmcid}?pdf=render"
        return None
    except Exception as exc:
        logger.debug("europepmc lookup failed for %s: %s", doi, exc)
        return None


def find_hal_pdf_url(doi: str) -> str | None:
    """Phase 54.6.51 — HAL (hal.science) French open archive.

    Strong coverage of European physics + astronomy + some climate
    content (Obs. de Paris, LATMOS, IPSL). Search API is free and
    doesn't require a key.
    """
    try:
        resp = _get_client().get(
            "https://api.archives-ouvertes.fr/search/",
            params={
                "q": f"doiId_s:\"{doi}\"",
                "fl": "files_s,fileMain_s",
                "wt": "json",
                "rows": 1,
            },
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        docs = (data.get("response") or {}).get("docs") or []
        if not docs:
            return None
        hit = docs[0]
        # Prefer fileMain_s (HAL's primary PDF link)
        main = hit.get("fileMain_s")
        if main:
            return main
        files = hit.get("files_s") or []
        for f in files:
            if isinstance(f, str) and f.lower().endswith(".pdf"):
                return f
        return None
    except Exception as exc:
        logger.debug("hal lookup failed for %s: %s", doi, exc)
        return None


def find_zenodo_pdf_url(doi: str) -> str | None:
    """Phase 54.6.51 — Zenodo (CERN-hosted OA repo).

    Useful for dataset-attached papers + some preprints that don't
    live anywhere else. Zenodo DOIs start with 10.5281/zenodo but
    papers posted to Zenodo with OTHER DOIs still register there.
    We search Zenodo's records API by DOI.
    """
    try:
        resp = _get_client().get(
            "https://zenodo.org/api/records",
            params={"q": f"doi:\"{doi}\"", "size": 1},
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        hits = ((data.get("hits") or {}).get("hits")) or []
        if not hits:
            return None
        files = (hits[0].get("files") or [])
        for f in files:
            if not isinstance(f, dict):
                continue
            key = f.get("key") or ""
            if key.lower().endswith(".pdf"):
                links = f.get("links") or {}
                return links.get("self") or links.get("download")
        return None
    except Exception as exc:
        logger.debug("zenodo lookup failed for %s: %s", doi, exc)
        return None


# ────────────────────────────────────────────────────────────────────────
# Download helpers
# ────────────────────────────────────────────────────────────────────────


def download_pdf(url: str, dest_path: Path) -> bool:
    """Download `url` to `dest_path`. Returns True on success, False on any
    error (network, non-PDF, too large)."""
    try:
        # Use the shared client for connection reuse, but with a longer
        # per-request timeout since actual PDFs can be multi-MB.
        resp = _get_client().get(url, timeout=_DOWNLOAD_TIMEOUT)
        if resp.status_code != 200:
            return False
        content = resp.content
        if len(content) > _MAX_PDF_MB * 1024 * 1024:
            return False
        if not content.startswith(b"%PDF"):
            return False
        dest_path.write_bytes(content)
        return True
    except Exception as exc:
        logger.debug("pdf download failed for %s: %s", url, exc)
        return False


# ────────────────────────────────────────────────────────────────────────
# Parallel multi-source discovery + download
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _LookupSpec:
    """Name of a source + how to resolve a URL for it."""
    source: str
    priority: int               # smaller = tried first at download time
    fn: object                  # callable returning str | None
    args: tuple                 # positional args to fn


def _gather_candidate_urls(
    doi: str | None,
    arxiv_id: str | None,
    email: str,
    resolve_arxiv_from_doi: bool = True,
) -> list[tuple[str, str]]:
    """Run every source-lookup in parallel, collecting (source, url)
    pairs for every source that produces a candidate URL.

    Phase 54.6.51 — two short-circuits for the common cases:

      - If the DOI matches the Copernicus/EGU pattern, skip the HTTP
        lookups entirely. Copernicus's URL is deterministic and their
        download is reliable, so there's no point paying 8 s of parallel
        API waits to learn what we already know.
      - If a ready-made arXiv ID is provided, its PDF URL is also
        template-generated. Skip the HTTP fan-out and just return it.

    Returns a list ordered by source priority (lower = higher priority).
    Duplicate URLs across sources are kept — that's intentional because
    a 403 from one CDN still means the paper might be reachable via
    another mirror.
    """
    # Fast-path: deterministic URLs need no API calls.
    if doi:
        co_url = find_copernicus_pdf_url(doi)
        if co_url:
            return [("copernicus", co_url)]
    if arxiv_id:
        return [("arxiv", find_arxiv_pdf_url(arxiv_id))]

    specs: list[_LookupSpec] = []
    if doi:
        specs.extend([
            _LookupSpec("unpaywall", 2, find_oa_pdf_url, (doi, email)),
            _LookupSpec("openalex", 3, find_openalex_pdf_url, (doi, email)),
            _LookupSpec("europepmc", 4, find_europepmc_pdf_url, (doi,)),
            _LookupSpec("semantic_scholar", 5, find_semantic_scholar_pdf_url, (doi,)),
            _LookupSpec("hal", 6, find_hal_pdf_url, (doi,)),
            _LookupSpec("zenodo", 7, find_zenodo_pdf_url, (doi,)),
        ])
    if doi and resolve_arxiv_from_doi and not arxiv_id:
        # Resolve an arXiv preprint via the DOI as a fallback source.
        # This runs in parallel with the others and only costs one
        # extra API call for papers where it's plausible.
        def _arxiv_from_doi():
            resolved = find_arxiv_id_by_doi(doi)
            return find_arxiv_pdf_url(resolved) if resolved else None
        specs.append(_LookupSpec("arxiv_via_doi", 1, _arxiv_from_doi, ()))

    if not specs:
        return []

    urls: list[tuple[int, str, str]] = []  # (priority, source, url)
    with ThreadPoolExecutor(max_workers=min(len(specs), 8)) as pool:
        futures = {pool.submit(s.fn, *s.args): s for s in specs}
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                url = fut.result()
            except Exception:
                continue
            if url:
                urls.append((spec.priority, spec.source, url))
    urls.sort(key=lambda t: t[0])
    return [(s, u) for _, s, u in urls]


def find_and_download(
    doi: str | None,
    arxiv_id: str | None,
    dest_path: Path,
    email: str,
    alternate_dois: list[str] | None = None,
    alternate_arxiv_ids: list[str] | None = None,
) -> tuple[bool, str]:
    """Discover + download an OA PDF for the given (doi, arxiv_id) pair.

    Phase 54.6.51 — URL discovery across all sources runs in parallel,
    then downloads are attempted in priority order (cheapest source
    first) so bandwidth isn't wasted on redundant copies.

    When primary identifiers don't produce a working URL, fall through
    to any `alternate_dois` / `alternate_arxiv_ids` (from title-dedup
    merged candidates — the same paper under a preprint vs published
    DOI, for example). Each alternate gets its own full parallel
    discovery pass.

    Returns (True, "<source>") on first successful download,
    (False, "") if every source on every identifier failed.
    """
    # Collect every identifier we should try, primary first.
    id_pairs: list[tuple[str | None, str | None]] = [(doi, arxiv_id)]
    for d in alternate_dois or []:
        if d and d.lower() != (doi or "").lower():
            id_pairs.append((d, None))
    for a in alternate_arxiv_ids or []:
        if a and a != arxiv_id:
            id_pairs.append((None, a))

    for d, a in id_pairs:
        candidates = _gather_candidate_urls(d, a, email)
        for source, url in candidates:
            if download_pdf(url, dest_path):
                return True, source
    return False, ""
