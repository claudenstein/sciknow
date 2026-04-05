"""
Open-access PDF discovery and download.

Priority order for finding a downloadable PDF:
  1. Unpaywall      — largest general-purpose OA PDF database by DOI
  2. arXiv          — direct PDF for papers with an arXiv ID
  3. OpenAlex       — catches preprints + institutional repos Unpaywall misses
  4. Copernicus     — deterministic URL pattern for EGU journals (ACP, CP, TC,
                      ESSD, BG, ...). Huge for climate/earth science libraries.
  5. Europe PMC     — free full-text for biomedical / environmental health papers
  6. Semantic Scholar — final fallback via their public API

Each source is a cheap HTTP lookup that either returns a candidate PDF URL or
None, followed by a download + PDF magic-byte validation. Order matters: the
cheapest/most reliable sources run first.
"""
from __future__ import annotations

import re
from pathlib import Path

import httpx


_HEADERS = {
    "User-Agent": "sciknow/0.1 (research use; https://github.com/claudenstein/sciknow)",
    "Accept": "application/pdf,*/*;q=0.9",
}

_TIMEOUT = 60          # seconds per download
_MAX_PDF_MB = 50       # skip files larger than this


def find_oa_pdf_url(doi: str, email: str) -> str | None:
    """
    Query Unpaywall for the best open-access PDF URL for a given DOI.
    Returns the URL string, or None if no OA version is found.
    """
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, headers={"User-Agent": _HEADERS["User-Agent"]})
        if resp.status_code != 200:
            return None
        data = resp.json()
        loc = data.get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url") or None
    except Exception:
        return None


def find_arxiv_pdf_url(arxiv_id: str) -> str:
    """Return the canonical arXiv PDF URL for a given arXiv ID."""
    # Strip version suffix for the canonical URL
    base = arxiv_id.split("v")[0]
    return f"https://arxiv.org/pdf/{base}.pdf"


def find_semantic_scholar_pdf_url(doi: str) -> str | None:
    """
    Query the Semantic Scholar API for an open-access PDF URL.
    Used as a final fallback when other sources fail.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {"fields": "openAccessPdf"}
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params=params,
                              headers={"User-Agent": _HEADERS["User-Agent"]})
        if resp.status_code != 200:
            return None
        data = resp.json()
        oa = data.get("openAccessPdf") or {}
        return oa.get("url") or None
    except Exception:
        return None


def find_openalex_pdf_url(doi: str, email: str) -> str | None:
    """
    Query OpenAlex for a PDF URL. Covers preprints and institutional repos
    that Unpaywall sometimes misses, and is polite-pool friendly (up to
    100k requests/day with an email).
    """
    url = f"https://api.openalex.org/works/doi:{doi}"
    params = {"mailto": email}
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params=params,
                              headers={"User-Agent": _HEADERS["User-Agent"]})
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Preferred field; falls through the OpenAlex OA location hierarchy.
        for key in ("best_oa_location", "primary_location"):
            loc = data.get(key) or {}
            pdf = loc.get("pdf_url")
            if pdf:
                return pdf
        # `oa_locations` is a list of all known OA copies (oldest OpenAlex
        # schema still returns this in some responses).
        for loc in data.get("oa_locations") or data.get("locations") or []:
            if not isinstance(loc, dict):
                continue
            pdf = loc.get("pdf_url")
            if pdf:
                return pdf
        return None
    except Exception:
        return None


# Copernicus / EGU journal DOI pattern:
#   10.5194/{journal}-{volume}-{start}-{year}
# PDF URL pattern:
#   https://{journal}.copernicus.org/articles/{volume}/{start}/{year}/{journal}-{volume}-{start}-{year}.pdf
#
# Matches the core EGU journals relevant to a climate/earth-science library:
# acp (Atmos. Chem. Phys.), cp (Climate of the Past), tc (The Cryosphere),
# essd (Earth System Science Data), bg (Biogeosciences), hess (Hydrol. Earth
# Syst. Sci.), gmd (Geosci. Model Dev.), esd (Earth System Dynamics), nhess,
# os (Ocean Science), se (Solid Earth), wcd (Weather and Climate Dynamics).
_COPERNICUS_DOI_RE = re.compile(
    r"^10\.5194/([a-z]+)-(\d+)-(\d+)-(\d{4})$",
    re.IGNORECASE,
)


def find_copernicus_pdf_url(doi: str) -> str | None:
    """
    Deterministic PDF URL for Copernicus (EGU) journals. No API call.

    Copernicus DOIs follow a rigid pattern that maps directly to a PDF on
    their article portal, so a regex + f-string is sufficient — no HTTP
    lookup required, which makes this the fastest source in the chain for
    any paper with a matching DOI.
    """
    m = _COPERNICUS_DOI_RE.match(doi.strip())
    if not m:
        return None
    journal, vol, start, year = m.group(1).lower(), m.group(2), m.group(3), m.group(4)
    return (
        f"https://{journal}.copernicus.org/articles/{vol}/{start}/{year}/"
        f"{journal}-{vol}-{start}-{year}.pdf"
    )


def find_europepmc_pdf_url(doi: str) -> str | None:
    """
    Europe PMC full-text lookup. Free full text for biomedical papers and
    anything cross-indexed from PubMed Central — useful overlap with
    atmospheric-health / environmental-health / climate-health literature.
    """
    search_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": f"DOI:{doi}",
        "format": "json",
        "resultType": "lite",
    }
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(search_url, params=params,
                              headers={"User-Agent": _HEADERS["User-Agent"]})
        if resp.status_code != 200:
            return None
        data = resp.json()
        results = (data.get("resultList") or {}).get("result") or []
        if not results:
            return None
        hit = results[0]
        # Prefer the explicit full-text PDF URL from fullTextUrlList if given.
        url_list = (hit.get("fullTextUrlList") or {}).get("fullTextUrl") or []
        for u in url_list:
            if (
                isinstance(u, dict)
                and u.get("documentStyle") == "pdf"
                and u.get("availability", "").lower() in ("open access", "free", "subscription")
            ):
                return u.get("url")
        # Otherwise fall back to the PMC canonical PDF if we have a PMCID.
        pmcid = hit.get("pmcid")
        if pmcid:
            return f"https://europepmc.org/articles/{pmcid}?pdf=render"
        return None
    except Exception:
        return None


def download_pdf(url: str, dest_path: Path) -> bool:
    """
    Download a PDF from `url` and save it to `dest_path`.
    Returns True on success, False on any error (network, not-a-PDF, too large).
    """
    try:
        with httpx.Client(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers=_HEADERS,
        ) as client:
            resp = client.get(url)

        if resp.status_code != 200:
            return False

        content = resp.content
        if len(content) > _MAX_PDF_MB * 1024 * 1024:
            return False
        if not content.startswith(b"%PDF"):
            return False

        dest_path.write_bytes(content)
        return True
    except Exception:
        return False


def find_and_download(
    doi: str | None,
    arxiv_id: str | None,
    dest_path: Path,
    email: str,
) -> tuple[bool, str]:
    """
    Try all available sources in priority order. Returns (success, source_name).

    Order:
      1. Copernicus pattern  — zero-cost URL construction for 10.5194/* DOIs
      2. arXiv               — direct PDF for any arXiv ID
      3. Unpaywall           — largest general OA database
      4. OpenAlex            — preprints + institutional repos
      5. Europe PMC          — biomedical / environmental health
      6. Semantic Scholar    — final fallback
    """
    # 1. Copernicus (deterministic, no API call, high hit rate for climate)
    if doi:
        pdf_url = find_copernicus_pdf_url(doi)
        if pdf_url and download_pdf(pdf_url, dest_path):
            return True, "copernicus"

    # 2. arXiv
    if arxiv_id:
        pdf_url = find_arxiv_pdf_url(arxiv_id)
        if download_pdf(pdf_url, dest_path):
            return True, "arxiv"

    # 3. Unpaywall (requires DOI)
    if doi:
        pdf_url = find_oa_pdf_url(doi, email)
        if pdf_url and download_pdf(pdf_url, dest_path):
            return True, "unpaywall"

    # 4. OpenAlex (catches preprints + institutional repos)
    if doi:
        pdf_url = find_openalex_pdf_url(doi, email)
        if pdf_url and download_pdf(pdf_url, dest_path):
            return True, "openalex"

    # 5. Europe PMC (biomedical overlap)
    if doi:
        pdf_url = find_europepmc_pdf_url(doi)
        if pdf_url and download_pdf(pdf_url, dest_path):
            return True, "europepmc"

    # 6. Semantic Scholar (final fallback)
    if doi:
        pdf_url = find_semantic_scholar_pdf_url(doi)
        if pdf_url and download_pdf(pdf_url, dest_path):
            return True, "semantic_scholar"

    return False, ""
