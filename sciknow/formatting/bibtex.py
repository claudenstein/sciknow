"""Bibliography → BibTeX entries with stable cite keys.

Strategy:

1. ``BookBibliography.from_book`` gives us global APA-formatted source
   lines for every paper cited anywhere in the book.
2. For each line we extract the DOI (regex on ``https://doi.org/…`` or
   ``doi:…``).
3. We look up ``paper_metadata`` by DOI to get structured fields
   (authors, title, year, journal, volume, pages, …).
4. We mint a stable, deterministic, unique citekey: ``LastnameYear`` with
   a 4-char hash suffix on collision so re-export of the same book
   always produces the same .tex.
5. We emit a ``BibEntry`` (``BibTeX text + citekey``) per source.

If a DOI lookup fails, we still emit a usable entry by parsing the APA
string itself — the renderer will get a slightly less rich entry but
the citation will resolve.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from sciknow.formatting.ir import BibEntry

log = logging.getLogger(__name__)


_DOI_RE = re.compile(r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)([^\s,]+)", re.IGNORECASE)
_YEAR_RE = re.compile(r"\((\d{4})\)")


def _extract_doi(apa_line: str) -> Optional[str]:
    m = _DOI_RE.search(apa_line or "")
    if not m:
        return None
    doi = m.group(1).rstrip(".,;)")
    return doi or None


def _extract_year(apa_line: str) -> Optional[str]:
    m = _YEAR_RE.search(apa_line or "")
    return m.group(1) if m else None


def _first_author_lastname(apa_line: str) -> str:
    """``[3] Smith, J., et al. (2022). ...`` → ``Smith``."""
    s = re.sub(r"^\s*\[\d+\]\s*", "", apa_line or "")
    # Stop at the first comma or open-paren or period
    head = re.split(r"[,(.]", s, maxsplit=1)[0].strip()
    parts = head.split()
    if not parts:
        return "Anon"
    last = parts[-1]
    last = re.sub(r"[^A-Za-z]", "", last)
    return last or "Anon"


def _short_hash(s: str, n: int = 4) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _safe_bibtex_value(v) -> str:
    """Escape a value for BibTeX braces. Null → empty string."""
    if v is None:
        return ""
    s = str(v)
    # No need to escape backslash inside braces; only need to balance braces.
    s = s.replace("{", r"\{").replace("}", r"\}")
    return s


def _bibtex_entry_from_metadata(citekey: str, row) -> str:
    """Format a paper_metadata row as BibTeX. ``row`` columns:
    (doi, title, year, authors, journal, volume, issue, pages, publisher).
    """
    doi, title, year, authors, journal, volume, issue, pages, publisher = row
    author_str = ""
    if authors and isinstance(authors, list):
        names = []
        for a in authors[:50]:
            if isinstance(a, dict):
                names.append(a.get("name") or a.get("display_name") or "")
            else:
                names.append(str(a))
        author_str = " and ".join(n for n in names if n)
    fields = []
    if title:        fields.append(f"  title     = {{{_safe_bibtex_value(title)}}}")
    if author_str:   fields.append(f"  author    = {{{_safe_bibtex_value(author_str)}}}")
    if year:         fields.append(f"  year      = {{{year}}}")
    if journal:      fields.append(f"  journal   = {{{_safe_bibtex_value(journal)}}}")
    if volume:       fields.append(f"  volume    = {{{_safe_bibtex_value(volume)}}}")
    if issue:        fields.append(f"  number    = {{{_safe_bibtex_value(issue)}}}")
    if pages:        fields.append(f"  pages     = {{{_safe_bibtex_value(pages)}}}")
    if publisher:    fields.append(f"  publisher = {{{_safe_bibtex_value(publisher)}}}")
    if doi:          fields.append(f"  doi       = {{{_safe_bibtex_value(doi)}}}")
    body = ",\n".join(fields)
    return f"@article{{{citekey},\n{body}\n}}"


def _bibtex_entry_from_apa(citekey: str, apa_line: str) -> str:
    """Last-resort BibTeX from a parsed APA line (when DOI lookup fails).

    We use ``@misc`` because we lack confidence in journal/volume/pages
    extraction, but we still surface authors/year/title/note.
    """
    s = re.sub(r"^\s*\[\d+\]\s*", "", apa_line or "").strip()
    year = _extract_year(s) or ""
    doi = _extract_doi(s) or ""
    # Authors: everything up to "(YEAR)."
    authors = ""
    title = s
    m = re.search(r"^(.*?)\(\d{4}\)\.\s*", s)
    if m:
        authors = m.group(1).rstrip(", ").strip()
        title = s[m.end():].split(".")[0].strip()
    fields = [f"  title = {{{_safe_bibtex_value(title)}}}"]
    if authors:
        fields.append(f"  author = {{{_safe_bibtex_value(authors)}}}")
    if year:
        fields.append(f"  year = {{{year}}}")
    if doi:
        fields.append(f"  doi = {{{_safe_bibtex_value(doi)}}}")
    fields.append(f"  note = {{{_safe_bibtex_value(s[:300])}}}")
    return f"@misc{{{citekey},\n" + ",\n".join(fields) + "\n}"


def build_bibliography(
    session,
    global_sources: list[str],
) -> tuple[list[BibEntry], dict[str, str]]:
    """Build BibTeX entries for the book bibliography.

    Returns ``(entries, citekeys_by_source_key)``:

    - ``entries`` are in the same order as ``global_sources`` (i.e.
      first-cited order in the book), each with a unique ``citekey``.
    - ``citekeys_by_source_key`` maps the dedup key (the APA line with
      the leading ``[N] `` stripped) → citekey, so callers can build
      the ``[N] → citekey`` map for inline citation rewriting.
    """
    from sqlalchemy import text

    # 1. Collect DOIs from APA lines
    parsed: list[tuple[str, str | None, str | None]] = []   # (key, doi, year)
    for line in global_sources:
        key = re.sub(r"^\s*\[\d+\]\s*", "", line).strip()
        if not key:
            continue
        parsed.append((key, _extract_doi(line), _extract_year(line)))

    # 2. Bulk-load metadata for all known DOIs
    dois = sorted({p[1] for p in parsed if p[1]})
    rows_by_doi: dict[str, tuple] = {}
    if dois and session is not None:
        try:
            rs = session.execute(text("""
                SELECT doi, title, year, authors, journal, volume, issue, pages, publisher
                FROM paper_metadata
                WHERE doi = ANY(:dois)
            """), {"dois": dois}).fetchall()
            for r in rs:
                if r[0]:
                    rows_by_doi[r[0]] = tuple(r)
        except Exception as e:
            log.warning("bibliography metadata lookup failed: %s", e)

    # 3. Mint cite keys and build entries
    entries: list[BibEntry] = []
    citekeys_by_source_key: dict[str, str] = {}
    used_keys: set[str] = set()

    for orig_line, (key, doi, year) in zip(global_sources, parsed):
        last = _first_author_lastname(orig_line)
        yr = year or "nd"
        base = f"{last}{yr}"
        # Stable disambiguator from the dedup key (so reorderings or
        # re-runs produce the same citekey).
        suffix = _short_hash(key)
        candidate = f"{base}_{suffix}"
        # In the unlikely event of collision (different keys hashing to
        # the same suffix and producing same lastname+year), bump.
        bump = 1
        while candidate in used_keys:
            bump += 1
            candidate = f"{base}_{suffix}{bump}"
        used_keys.add(candidate)
        citekey = candidate

        row = rows_by_doi.get(doi) if doi else None
        if row:
            bibtex = _bibtex_entry_from_metadata(citekey, row)
        else:
            bibtex = _bibtex_entry_from_apa(citekey, orig_line)

        entries.append(BibEntry(citekey=citekey, bibtex=bibtex, apa=orig_line, doi=doi))
        citekeys_by_source_key[key] = citekey

    return entries, citekeys_by_source_key


def render_bibtex_file(entries: list[BibEntry]) -> str:
    """Concatenate all entries into one .bib file body."""
    return "\n\n".join(e.bibtex for e in entries) + "\n"
