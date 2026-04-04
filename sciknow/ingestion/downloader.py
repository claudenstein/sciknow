"""
Open-access PDF discovery and download.

Priority order for finding a downloadable PDF:
  1. Unpaywall  — queries the largest OA PDF database by DOI
  2. arXiv      — direct PDF for papers with an arXiv ID
  3. Semantic Scholar — open access PDFs via their public API

All downloads are validated as real PDFs (%PDF magic bytes) before being saved.
"""
from __future__ import annotations

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
    Used as a fallback when Unpaywall returns nothing.
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
    """
    # 1. Unpaywall (requires DOI)
    if doi:
        pdf_url = find_oa_pdf_url(doi, email)
        if pdf_url and download_pdf(pdf_url, dest_path):
            return True, "unpaywall"

    # 2. arXiv
    if arxiv_id:
        pdf_url = find_arxiv_pdf_url(arxiv_id)
        if download_pdf(pdf_url, dest_path):
            return True, "arxiv"

    # 3. Semantic Scholar (requires DOI)
    if doi:
        pdf_url = find_semantic_scholar_pdf_url(doi)
        if pdf_url and download_pdf(pdf_url, dest_path):
            return True, "semantic_scholar"

    return False, ""
