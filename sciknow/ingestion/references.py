"""
Reference extraction from scientific papers.

Three sources:
  1. MinerU content_list.json       — walks the structured block list produced
     by the MinerU 2.5 pipeline backend; finds the References/Bibliography
     heading and harvests DOIs/arXiv IDs from subsequent text blocks. This is
     the primary source for new (MinerU-ingested) papers.
  2. Markdown bibliography section  — legacy regex parser for the reference
     list produced by Marker when converting a PDF. Still used as a fallback
     for any documents ingested before the MinerU switch.
  3. Crossref raw data              — the `reference` array stored in
     paper_metadata when a paper was fetched via the Crossref API. Complements
     the other two by catching references the publisher deposited to Crossref
     but which may not have been rendered legibly in the source PDF.

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


# ── MinerU content_list.json reference section detection ─────────────────────

# Heading text that indicates the start of the references section. Matched
# case-insensitively with normalisation (strip punctuation + numeric prefixes).
_REF_HEADING_WORDS: frozenset[str] = frozenset({
    "references", "reference", "bibliography", "works cited",
    "literature cited", "citations", "citation",
})


def _normalise_heading(s: str) -> str:
    import re as _re
    s = _re.sub(r"^[\d.]+\s*", "", (s or "").lower().strip())
    s = _re.sub(r"[^\w\s]", "", s).strip()
    return s


def _is_reference_heading(text: str) -> bool:
    norm = _normalise_heading(text)
    if not norm:
        return False
    for w in _REF_HEADING_WORDS:
        if norm == w or norm.startswith(w + " "):
            return True
    return False


def extract_references_from_mineru_content_list(
    content_list: list[dict],
) -> list[Reference]:
    """
    Walk a MinerU 2.5 content_list.json and harvest reference entries.

    Algorithm:
      1. Scan for the first text item with `text_level` >= 1 whose text looks
         like a References/Bibliography heading.
      2. From that point forward, treat every text item as a potential
         reference entry until we hit another level-1/level-2 heading that
         is clearly NOT part of the references section (Appendix, Supplementary,
         Author Contributions, Funding, etc.) or the end of the document.
      3. For each candidate entry, run the standard DOI/arXiv/title parser.

    MinerU sometimes splits a single reference across two or three text items
    (particularly when a reference contains a URL that wraps). We heuristically
    merge consecutive short items that don't start with a typical entry marker
    (`[1]`, `1.`, or a capitalised surname) into the previous entry.
    """
    if not content_list:
        return []

    # Phase 1: find the references heading.
    ref_start_idx: int | None = None
    for i, item in enumerate(content_list):
        if item.get("type") != "text":
            continue
        level = item.get("text_level") or 0
        text = (item.get("text") or "").strip()
        if not text:
            continue
        # Accept either a properly-levelled heading, or a short plain-text
        # block that matches a known references word (common for older PDFs
        # where MinerU's layout model didn't emit a text_level).
        is_short = len(text.split()) <= 4
        if (level >= 1 or is_short) and _is_reference_heading(text):
            ref_start_idx = i + 1
            break

    if ref_start_idx is None:
        return []

    # Phase 2: collect candidate entries until the next top-level heading that
    # is clearly the end of the references section.
    STOP_WORDS = {
        "appendix", "appendices", "supplementary", "supporting information",
        "author contributions", "acknowledgments", "acknowledgements",
        "funding", "data availability", "conflict of interest",
        "competing interests", "about the author",
    }

    raw_entries: list[str] = []
    current: list[str] = []

    def _flush_current() -> None:
        if current:
            joined = " ".join(current).strip()
            if joined:
                raw_entries.append(joined)
            current.clear()

    for item in content_list[ref_start_idx:]:
        if item.get("type") != "text":
            # Tables/equations/images inside a reference section are noise;
            # they don't break the current entry — skip them.
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue

        level = item.get("text_level") or 0
        if level >= 1:
            norm = _normalise_heading(text)
            if any(norm == w or norm.startswith(w + " ") for w in STOP_WORDS):
                _flush_current()
                break
            if _is_reference_heading(text):
                # Another "references" heading (shouldn't happen but be safe)
                continue
            # A new top-level heading that isn't in the stop list — assume
            # the ref section ended here anyway.
            _flush_current()
            break

        # Body text — either a new entry or a continuation of the previous.
        if _looks_like_entry_start(text):
            _flush_current()
            current.append(text)
        else:
            if current:
                current.append(text)
            else:
                # Orphan continuation before any entry started — treat as
                # first entry.
                current.append(text)

    _flush_current()

    # Phase 3: parse each raw entry string.
    refs: list[Reference] = []
    for entry in raw_entries:
        if len(entry) < 25:
            continue
        parsed = _parse_entry(entry)
        if parsed is not None:
            refs.append(parsed)

    return refs


def _looks_like_entry_start(text: str) -> bool:
    """
    Heuristic for whether a text block starts a new reference entry (vs
    continuing the previous one).

    Positive signals:
      - Starts with a numbered marker:  `[12]`, `12.`, `(12)`, `12 `
      - Starts with a Capitalised surname followed by , or initial:
        `Smith, J.`, `Smith J.`, `Smith J 2023`
      - Starts with a year in parens after a name block (harder to detect;
        we fall through to the default "assume new entry" for longish lines)
    """
    import re as _re
    if not text:
        return False
    # Numbered markers: [12], 12., 12)
    if _re.match(r"^\[?\d{1,4}[\]\.\)]\s", text):
        return True
    # Capitalised surname + comma/initial: Smith, J. or Smith J
    if _re.match(r"^[A-Z][a-z]+[,\s][A-Z]", text):
        return True
    # URL fragment / "doi:" / "http" — continuation markers
    if text[:5].lower() in ("http:", "https", "doi:"):
        return False
    # A short fragment without a capital letter start is likely a continuation.
    if len(text) < 30 and not text[0].isupper():
        return False
    # Otherwise, assume it's a new entry when long enough to stand alone.
    return len(text) >= 40


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


# ── OpenAlex referenced_works lookup ─────────────────────────────────────────

def fetch_openalex_references(
    doi: str,
    email: str,
    timeout: float = 10.0,
) -> list[Reference]:
    """
    Query OpenAlex for a paper by DOI and return its referenced_works resolved
    to Reference objects.

    OpenAlex's `referenced_works` array contains OpenAlex work IDs (W-numbers),
    not DOIs directly. We resolve those IDs to DOIs by batching them through
    the `/works?filter=openalex:W1|W2|...` endpoint (up to 50 IDs per request
    per the OpenAlex API limits).

    Returns an empty list on any error — this is a best-effort enrichment
    source, not a hard requirement.
    """
    import httpx

    try:
        with httpx.Client(timeout=timeout) as client:
            # Step 1: fetch the citing paper to get its referenced_works list
            resp = client.get(
                f"https://api.openalex.org/works/doi:{doi}",
                params={"mailto": email, "select": "referenced_works"},
                headers={"User-Agent": "sciknow/0.1"},
            )
            if resp.status_code != 200:
                return []
            ref_ids = resp.json().get("referenced_works") or []
            if not ref_ids:
                return []

            # Normalize: full URLs → bare W-numbers
            w_ids: list[str] = []
            for rid in ref_ids:
                if not isinstance(rid, str):
                    continue
                w = rid.rsplit("/", 1)[-1]
                if w.startswith("W"):
                    w_ids.append(w)

            # Step 2: batch-resolve W-numbers to DOIs + titles + years
            refs: list[Reference] = []
            BATCH = 50
            for start in range(0, len(w_ids), BATCH):
                batch = w_ids[start : start + BATCH]
                filt = "openalex:" + "|".join(batch)
                resp = client.get(
                    "https://api.openalex.org/works",
                    params={
                        "filter": filt,
                        "per_page": BATCH,
                        "select": "id,doi,title,publication_year",
                        "mailto": email,
                    },
                    headers={"User-Agent": "sciknow/0.1"},
                )
                if resp.status_code != 200:
                    continue
                for w in resp.json().get("results") or []:
                    doi_full = w.get("doi") or ""
                    if doi_full.startswith("https://doi.org/"):
                        ref_doi = doi_full[len("https://doi.org/"):]
                    else:
                        ref_doi = doi_full or None
                    title = w.get("title") or None
                    year = w.get("publication_year") or None
                    if not ref_doi and not title:
                        continue
                    refs.append(Reference(
                        raw_text=f"{title or ''} ({year or ''})"[:400],
                        doi=ref_doi,
                        title=title,
                        year=year,
                    ))
            return refs
    except Exception:
        return []


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
