"""
Reference extraction from scientific papers.

Two sources:
  1. Markdown bibliography section  — regex-based parsing of the reference list
     produced by Marker when converting a PDF.
  2. Crossref raw data              — the `reference` array stored in paper_metadata
     when a paper was fetched via the Crossref API.

Each reference is returned as a Reference dataclass with whatever fields could
be extracted (DOI is the most valuable; title is used as a fallback for lookup).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Reference:
    raw_text: str                   # original text of the reference entry
    doi: str | None = None
    arxiv_id: str | None = None
    title: str | None = None
    year: int | None = None
    authors: list[str] = field(default_factory=list)


# ── Markdown reference section detection ──────────────────────────────────────

_REF_HEADING_RE = re.compile(
    r'^#{1,4}\s*(references?|bibliography|works\s+cited|literature\s+cited|citations?)\s*$',
    re.IGNORECASE | re.MULTILINE,
)

# Split a block of text into individual reference entries.
# Handles: [1] ..., 1. ..., and blank-line-separated entries.
_NUMBERED_SPLIT_RE = re.compile(
    r'(?:^|\n)(?=\[?\d{1,3}[\]\.]\s)',
)


def extract_references_from_markdown(text: str) -> list[Reference]:
    """
    Find the references / bibliography section in the markdown and parse
    individual entries. Returns a (possibly empty) list of Reference objects.
    """
    m = _REF_HEADING_RE.search(text)
    if not m:
        return []

    ref_section = text[m.end():]

    # Trim at the next heading that is NOT another references-like section
    next_h = re.search(r'\n#{1,4}\s+\S', ref_section)
    if next_h:
        ref_section = ref_section[: next_h.start()]

    # Try to split on numbered markers first
    parts = _NUMBERED_SPLIT_RE.split(ref_section)
    if len(parts) < 3:
        # Fall back to blank-line-separated entries
        parts = re.split(r'\n\s*\n', ref_section)

    refs: list[Reference] = []
    for entry in parts:
        entry = entry.strip()
        if len(entry) < 25:          # too short to be a real reference
            continue
        r = _parse_entry(entry)
        if r is not None:
            refs.append(r)

    return refs


def _parse_entry(text: str) -> Reference | None:
    """Parse a single reference string into a Reference object."""
    from sciknow.utils.doi import extract_doi, extract_arxiv_id, normalize_doi

    # Strip leading number / bracket: "[12]" or "12."
    cleaned = re.sub(r'^\[?\d{1,3}[\]\.]\s*', '', text).strip()

    doi = extract_doi(cleaned)
    if doi:
        doi = normalize_doi(doi)
    arxiv_id = extract_arxiv_id(cleaned)

    # Year: (2023) or , 2023,
    year: int | None = None
    ym = re.search(r'\((\d{4})\)', cleaned)
    if not ym:
        ym = re.search(r'\b(19\d{2}|20\d{2})\b', cleaned)
    if ym:
        y = int(ym.group(1))
        if 1800 <= y <= 2035:
            year = y

    # Title: try to extract from APA-style "Authors (year). **Title**."
    # or "Authors (year). Title. Journal"
    title: str | None = None
    # Markdown bold: **Title**
    tbold = re.search(r'\*\*(.+?)\*\*', cleaned)
    if tbold:
        title = tbold.group(1).strip()
    else:
        # APA: everything between "(year). " and the next ". "
        tapa = re.search(r'\(\d{4}\)\.\s+(.+?)\.', cleaned)
        if tapa:
            candidate = tapa.group(1).strip()
            # Discard if it looks like a journal name (all caps, very short)
            if len(candidate) > 15 and not candidate.isupper():
                title = candidate

    # Need at least a DOI, arXiv ID, or a parseable title to be useful
    if not doi and not arxiv_id and not title:
        return None

    return Reference(
        raw_text=text[:400],
        doi=doi,
        arxiv_id=arxiv_id,
        title=title,
        year=year,
    )


# ── Crossref stored reference list ────────────────────────────────────────────

def extract_references_from_crossref(crossref_raw: dict | None) -> list[Reference]:
    """
    Parse the `reference` array from a stored Crossref API response.
    Crossref references are already partially structured; many include a DOI.
    """
    if not crossref_raw:
        return []

    from sciknow.utils.doi import extract_doi, normalize_doi

    refs: list[Reference] = []
    for r in crossref_raw.get("reference", []):
        doi: str | None = r.get("DOI")
        if doi:
            doi = normalize_doi(doi)
        else:
            # Sometimes the DOI is buried in the unstructured text
            unstructured = r.get("unstructured", "")
            if unstructured:
                raw_doi = extract_doi(unstructured)
                if raw_doi:
                    doi = normalize_doi(raw_doi)

        title = r.get("article-title") or r.get("volume-title")
        year_str = r.get("year", "")
        year: int | None = None
        if year_str and str(year_str).isdigit():
            y = int(year_str)
            if 1800 <= y <= 2035:
                year = y

        unstructured = r.get("unstructured", "")
        raw_text = unstructured or f"{title or ''} ({year or ''})"

        if not doi and not title:
            continue

        refs.append(Reference(
            raw_text=raw_text[:400],
            doi=doi,
            title=title,
            year=year,
        ))

    return refs
