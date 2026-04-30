"""Additional metadata sources for `db enrich` — Phase 54.6.313.

Implements the top-ROI strategies documented in docs/research/ENRICH_RESEARCH.md:

  - extract_xmp_doi(pdf_path)           — PDF XMP packet parser. Reads the
                                          Adobe PRISM/DC namespace fields
                                          publishers embed at export time
                                          (prism:doi, dc:identifier, prism:url).
                                          Near-zero false positives.
  - extract_fulltext_doi(pdf_path)      — regex DOI scan across the first
                                          3 pages of extracted text; validates
                                          candidates via Crossref /works/{doi}.
  - search_semantic_scholar_match(title,
                                   author, year)
                                          — uses /graph/v1/paper/search/match
                                          (purpose-built, single-best-hit).
  - search_datacite_by_title(title,
                             author, year)
                                          — DataCite REST query. Useful for
                                          climate / earth-sci datasets and
                                          Zenodo/PANGAEA preprints that aren't
                                          indexed by Crossref.
  - search_openlibrary_by_title(title,
                                author)  — books only: returns ISBN-10/13 +
                                          LC subjects. No key, polite-UA.
  - validate_doi_resolves(doi)          — HEAD the DOI via doi.org resolver
                                          and verify the redirect target
                                          returns HTTP 200. Kills OCR-mangled
                                          candidates before Crossref gets hit.

The cascade order recommended by the research memo is wired into
`sciknow.ingestion.metadata.extract` as a superset of the existing
4-layer pipeline.

All helpers swallow network/parse errors and return ``None`` instead of
raising — the metadata cascade is best-effort and must never block ingest.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import httpx
import pymupdf  # fitz

from sciknow.config import settings
from sciknow.utils.doi import normalize_doi

logger = logging.getLogger("sciknow.enrich_sources")


# ── 1. PDF XMP packet extraction ────────────────────────────────────────

# Publisher-embedded DOI lives in PRISM or DC namespaces. Elsevier,
# Wiley, Springer, IOP, Taylor & Francis all stamp this field at
# copy-edit time, so the signal is extremely clean when present.
# We parse defensively: real-world XMP has whitespace, attribute vs
# element variants, and occasional mangled CDATA. A regex is more
# robust than lxml here because we only need one field.
_XMP_DOI_PATTERNS = [
    # <prism:doi>10.xxxx/yyyy</prism:doi>
    re.compile(r"<prism:doi[^>]*>([^<]+)</prism:doi>", re.IGNORECASE),
    # <dc:identifier ...>doi:10.xxxx/yyyy</dc:identifier>
    re.compile(r"<dc:identifier[^>]*>[^<]*?(10\.\d{4,9}/[^\s<]+)[^<]*</dc:identifier>", re.IGNORECASE),
    # <prism:url>https://doi.org/10.xxxx/yyyy</prism:url>
    re.compile(r"<prism:url[^>]*>[^<]*?(10\.\d{4,9}/[^\s<]+)[^<]*</prism:url>", re.IGNORECASE),
    # Fallback: any DOI-shaped string anywhere in the XMP packet
    # (captures publishers that use non-standard tags).
    re.compile(r"\b(10\.\d{4,9}/[-._;()/:\w]+)", re.IGNORECASE),
]

# Strip trailing punctuation publishers sometimes concatenate.
_DOI_TRAIL_RE = re.compile(r"[\s\.,;\)<>\"']+$")


def extract_xmp_doi(pdf_path: Path | str) -> str | None:
    """Return the DOI embedded in a PDF's XMP metadata packet, if any.

    Only the DOI is returned — the other XMP fields are rarely used
    downstream and would bloat the return type for marginal benefit.
    On PyMuPDF ≥ 1.23 ``doc.xref_xml_metadata()`` returns the xref of
    the XMP stream; ``doc.xref_stream(xref)`` then gives the raw bytes.

    Returns:
        Normalised DOI string (no ``doi:`` prefix, no URL wrapper) or None.
    """
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        logger.debug("pymupdf.open failed on %s: %s", pdf_path, exc)
        return None
    try:
        xref = 0
        try:
            xref = doc.xref_xml_metadata()
        except Exception:
            xref = 0
        if not xref:
            return None
        raw = doc.xref_stream(xref)
        if not raw:
            return None
        try:
            packet = raw.decode("utf-8", errors="ignore")
        except Exception:
            return None
    finally:
        doc.close()

    for pat in _XMP_DOI_PATTERNS:
        m = pat.search(packet)
        if not m:
            continue
        candidate = m.group(1).strip()
        # dc:identifier and prism:url can return "doi:10.xxx/yyy" or URL
        candidate = _DOI_TRAIL_RE.sub("", candidate)
        try:
            candidate = normalize_doi(candidate)
        except Exception:
            continue
        # Sanity: must look like a DOI.
        if re.match(r"^10\.\d{4,9}/", candidate):
            return candidate
    return None


# ── 1.5. Filename-encoded DOI ───────────────────────────────────────────

# The sciknow downloader persists PDFs as ``<doi_slashes_to_underscores>.pdf``
# for papers fetched by DOI, so many on-disk files encode the
# canonical DOI directly in the filename — faster + more reliable than
# reading the XMP or first-3-page body when present. We still validate
# the candidate against Crossref before accepting, to kill any
# renamed-by-hand / OCR-mangled filenames.
_FILENAME_DOI_RE = re.compile(
    r"(10\.\d{4,5})[_\-](.+?)(?:\.pdf)?$",
    re.IGNORECASE,
)


def extract_filename_doi(
    pdf_path: Path | str,
    *,
    validate: bool = True,
) -> str | None:
    """Recover a DOI from the PDF's filename when the sciknow downloader
    persisted it as ``10.NNNN_suffix.pdf``.

    Returns:
        Normalised DOI (validated via Crossref when ``validate=True``)
        or None.
    """
    try:
        from pathlib import Path as _P
        fn = _P(str(pdf_path)).name
    except Exception:
        return None
    m = _FILENAME_DOI_RE.search(fn)
    if not m:
        return None
    # Reassemble: "10.1017" + "/" + "s1743921311015195"
    cand = f"{m.group(1)}/{m.group(2).rstrip('.pdf')}".lower()
    # Normalise common downloader substitutions:
    #   "_" in the suffix often replaces "/" in multi-part DOIs (10.1017/S174392…)
    # but that's rare — most are single-slash. Leave as-is and validate.
    try:
        cand = normalize_doi(cand)
    except Exception:
        return None
    if not re.match(r"^10\.\d{4,9}/", cand):
        return None
    if not validate:
        return cand
    if validate_doi_resolves(cand):
        return cand
    return None


# ── 2. First-3-page regex DOI scrape ────────────────────────────────────

_FULLTEXT_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:\w]+)", re.IGNORECASE)


def extract_fulltext_doi(
    pdf_path: Path | str,
    *,
    max_pages: int = 3,
    validate: bool = True,
) -> str | None:
    """Scan the first ``max_pages`` of a PDF for a DOI-shaped string.

    Publishers routinely print the DOI in the footer of page 1 — but
    old scans, OCR'd PDFs, and open-submission preprints often have
    this signal in the body text only, never in the Info dict or XMP.
    Scraping the rendered text catches those.

    Args:
        pdf_path: path to the PDF.
        max_pages: how many leading pages to scan. 3 is the sweet spot —
                   catches title-page + letter-first-page + two-column
                   layouts where the DOI is on page 2 of the first
                   spread. Higher values drift into references where a
                   DOI-shaped string is a cited paper, not this paper.
        validate: if True, call `validate_doi_resolves` before accepting
                  a candidate. Defeats OCR-mangled substrings like
                  `10.10116/j.icarus.…` (typo of 10.1016).

    Returns:
        Normalised DOI or None.
    """
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        logger.debug("pymupdf.open failed on %s: %s", pdf_path, exc)
        return None
    try:
        text = ""
        for i in range(min(max_pages, doc.page_count)):
            try:
                text += doc.load_page(i).get_text("text") + "\n"
            except Exception:
                continue
    finally:
        doc.close()

    if not text:
        return None

    # Collect up to 5 candidates so we can try more than the first
    # match — the first regex hit is sometimes a citation to a prior
    # paper (e.g. "as shown by [3] doi:10.XXXX/ZZZZ").
    seen: set[str] = set()
    candidates: list[str] = []
    for m in _FULLTEXT_DOI_RE.finditer(text):
        raw = _DOI_TRAIL_RE.sub("", m.group(1).strip())
        try:
            cand = normalize_doi(raw)
        except Exception:
            continue
        if cand in seen:
            continue
        seen.add(cand)
        candidates.append(cand)
        if len(candidates) >= 5:
            break

    for cand in candidates:
        if not validate:
            return cand
        if validate_doi_resolves(cand):
            return cand
    return None


# ── 3. DOI resolver validation ──────────────────────────────────────────

def validate_doi_resolves(doi: str, *, timeout: float = 8.0) -> bool:
    """Return True iff ``doi`` resolves to a registered DOI.

    We hit Crossref's cheap endpoint first — it returns JSON only for
    registered DOIs and 404s otherwise. Crossref's /works is ~50 RPS
    polite-pool, so this is the right layer for a validation call.
    """
    if not doi or not doi.startswith("10."):
        return False
    url = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": f"sciknow/0.1 (mailto:{settings.crossref_email})"}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=headers)
        return resp.status_code == 200
    except Exception as exc:
        logger.debug("validate_doi_resolves(%r) network error: %s", doi, exc)
        return False


# ── 4. Semantic Scholar /match ──────────────────────────────────────────

# Process-wide token bucket for Semantic Scholar. Without a key the
# unauth pool gets aggressively 429'd in parallel workers; this keeps
# the inter-call gap ≥ 1s so we share politely across threads.
import threading as _threading
_S2_LOCK = _threading.Lock()
_S2_LAST_CALL = [0.0]   # epoch seconds, list for mutability


def _s2_wait_for_token(min_gap_seconds: float = 1.05) -> None:
    """Block until ``min_gap_seconds`` has elapsed since the last S2 call."""
    import time
    with _S2_LOCK:
        delta = time.time() - _S2_LAST_CALL[0]
        if delta < min_gap_seconds:
            time.sleep(min_gap_seconds - delta)
        _S2_LAST_CALL[0] = time.time()


def search_semantic_scholar_match(
    title: str,
    *,
    first_author: str | None = None,
    year: int | None = None,
    timeout: float = 15.0,
    max_retries: int = 4,
) -> dict | None:
    """Call Semantic Scholar's /graph/v1/paper/search/match endpoint.

    /match is different from /search — it's designed for "given a
    noisy title, find the single best match in our 200M paper
    graph". It returns one record (or 404 on no-match), which is what
    the enrich path wants. Coverage is broader than Crossref for
    preprints and non-traditional venues.

    Without an API key the endpoint is rate-limited against a global
    unauth pool that 429s in bulk — we retry with exponential backoff
    up to ``max_retries`` times (0.5s → 1s → 2s) before giving up.
    This turns a rate-limit spike from "every paper misses" into "a
    small polite pause".

    Returns:
        Dict shaped like {title, year, authors, externalIds:{DOI,…}}
        or None on miss / network error / rate-limit after retries.
    """
    import time
    if not title or len(title.strip()) < 10:
        return None
    headers = {
        "User-Agent": f"sciknow/0.1 (mailto:{settings.crossref_email})",
    }
    key = getattr(settings, "semantic_scholar_api_key", None)
    if key:
        headers["x-api-key"] = key
    params = {
        "query": title[:300].strip(),
        "fields": "title,year,authors,externalIds,journal,abstract,venue",
    }
    body = None
    for attempt in range(max_retries):
        # Shared rate limiter — keeps us well under the ~1 RPS unauth
        # ceiling even with parallel workers in the enrich pipeline.
        # With an API key, 1 RPS is still polite (the 1/sec key tier
        # is the public default).
        _s2_wait_for_token(1.1)
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search/match",
                    params=params,
                    headers=headers,
                )
        except Exception as exc:
            logger.debug("Semantic Scholar /match network error: %s", exc)
            return None
        if resp.status_code == 429:
            # Honour Retry-After if present, else exponential backoff
            # starting at 2s (since /match is stricter than /search).
            wait = float(resp.headers.get("Retry-After", 0)) or (2.0 * (2 ** attempt))
            wait = min(wait, 16.0)
            logger.debug("S2 /match 429 (attempt %d); backing off %.1fs", attempt + 1, wait)
            time.sleep(wait)
            continue
        if resp.status_code == 404:
            # /match explicitly 404s on no-match.
            return None
        if resp.status_code != 200:
            return None
        try:
            body = resp.json() or {}
        except Exception:
            return None
        break
    if body is None:
        return None

    # /match returns {"data": [<one record>]} on hit; empty data on miss.
    data = body.get("data") or []
    if not data:
        return None
    hit = data[0] or {}

    # Title-similarity gate. S2 /match is happy to return overlapping
    # papers on generic titles ("Climate of the Past, Present and
    # Future" → gets mapped to a gravity-wave parametrization paper
    # because the word overlap is "past present future"). Require a
    # firm Dice coefficient ≥ 0.6 on the normalised titles.
    hit_title = (hit.get("title") or "").strip()
    if hit_title and _loose_title_match(title, hit_title) < 0.6:
        return None

    # Optional corroboration: if caller provided year / author,
    # reject hits that clearly disagree (±1 year tolerance; any
    # surname overlap).
    if year and hit.get("year"):
        try:
            if abs(int(hit["year"]) - int(year)) > 1:
                return None
        except (TypeError, ValueError):
            pass
    if first_author:
        surnames_ours = _surnames_set(first_author)
        surnames_hit = set()
        for a in (hit.get("authors") or []):
            surnames_hit |= _surnames_set(a.get("name", ""))
        if surnames_ours and surnames_hit and not (surnames_ours & surnames_hit):
            # Only reject when BOTH sides have surnames and they
            # don't intersect at all — missing author fields should
            # not reject an otherwise-matching hit.
            return None
    return hit


# ── 5. DataCite ─────────────────────────────────────────────────────────

def search_datacite_by_title(
    title: str,
    *,
    first_author: str | None = None,
    year: int | None = None,
    timeout: float = 15.0,
) -> dict | None:
    """Search DataCite REST for a title match.

    DataCite hosts ~38M dataset / preprint / software DOIs — the
    climate corpus overlaps their PANGAEA (prefix 10.1594), NASA
    (10.5067), and Zenodo (10.5281) collections. Free, no auth.

    Returns:
        Dict with {doi, title, year, authors, journal} or None.
    """
    if not title or len(title.strip()) < 10:
        return None

    # DataCite's Elasticsearch-flavour query DSL: the exact-phrase
    # `"<...>"` form is too strict (misses punctuation variants), and
    # field-less search drags in every DOI whose abstract mentions
    # any word. The middle ground — `titles.title:(word word word)` —
    # scores documents that contain ≥ 1 of the tokens in the title
    # field, which is what we want as a *candidate* pool before the
    # `_loose_title_match` gate below filters to the real hits.
    norm = re.sub(r"[^A-Za-z0-9 ]+", " ", title)
    toks = [t for t in norm.split() if len(t) >= 3][:10]
    if not toks:
        return None
    safe_query = "titles.title:(" + " ".join(toks) + ")"
    params = {
        "query": safe_query,
        "page[size]": 10,
    }
    headers = {"User-Agent": f"sciknow/0.1 (mailto:{settings.crossref_email})"}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                "https://api.datacite.org/dois", params=params, headers=headers
            )
        if resp.status_code != 200:
            return None
        body = resp.json() or {}
    except Exception as exc:
        logger.debug("DataCite network error: %s", exc)
        return None

    for record in (body.get("data") or []):
        attrs = record.get("attributes") or {}
        doi = attrs.get("doi") or record.get("id") or ""
        if not doi:
            continue
        hit_title = ""
        titles = attrs.get("titles") or []
        if titles:
            hit_title = (titles[0] or {}).get("title") or ""
        if not hit_title:
            continue
        # Gate on title similarity ≥ 0.85 via a cheap token_set check.
        if _loose_title_match(title, hit_title) < 0.85:
            continue
        hit_year = attrs.get("publicationYear")
        if year and hit_year:
            try:
                if abs(int(hit_year) - int(year)) > 1:
                    continue
            except (TypeError, ValueError):
                pass
        if first_author:
            surnames_ours = _surnames_set(first_author)
            creators = attrs.get("creators") or []
            surnames_hit: set[str] = set()
            for c in creators:
                nm = (c.get("name") or "").strip()
                if nm:
                    surnames_hit |= _surnames_set(nm)
            if surnames_ours and surnames_hit and not (surnames_ours & surnames_hit):
                continue
        # Strong candidate.
        return {
            "doi": normalize_doi(doi),
            "title": hit_title,
            "year": int(hit_year) if hit_year else None,
            "authors": [{"name": (c.get("name") or "").strip()}
                        for c in (attrs.get("creators") or [])],
            "journal": (attrs.get("publisher") or None),
            "raw": record,
        }
    return None


# ── 6. Europe PMC (life-sci / climate-health) ──────────────────────────

def search_europepmc_by_title(
    title: str,
    *,
    first_author: str | None = None,
    year: int | None = None,
    timeout: float = 15.0,
) -> dict | None:
    """Search Europe PMC REST for title matches.

    Europe PMC indexes 33M life-science records (PubMed + Agricola +
    patents + full-text OA). Strong coverage for anything in the
    climate-health overlap, the agricultural-science periphery, and
    the WHO / FAO grey-literature space. Free, no auth.

    Returns:
        Dict with {doi, pmid, pmcid, title, year, authors, journal}
        or None on miss.
    """
    if not title or len(title.strip()) < 10:
        return None

    # Europe PMC's Lucene-ish query DSL: TITLE:"<phrase>" for title
    # field search. The phrase form is tolerant because the field is
    # already tokenised; no need for our own OR-expansion.
    safe_title = title.replace('"', " ").strip()
    params = {
        "query": f'TITLE:"{safe_title[:200]}"',
        "format": "json",
        "pageSize": 10,
        "resultType": "core",  # full record, incl. DOI/PMID/authors
    }
    headers = {"User-Agent": f"sciknow/0.1 (mailto:{settings.crossref_email})"}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params=params, headers=headers,
            )
        if resp.status_code != 200:
            return None
        body = resp.json() or {}
    except Exception as exc:
        logger.debug("Europe PMC network error: %s", exc)
        return None

    hits = (body.get("resultList") or {}).get("result") or []
    for h in hits:
        h_title = (h.get("title") or "").strip().rstrip(".")
        if not h_title:
            continue
        if _loose_title_match(title, h_title) < 0.85:
            continue
        h_year = None
        try:
            h_year = int(str(h.get("pubYear") or "")[:4])
        except Exception:
            pass
        if year and h_year and abs(h_year - int(year)) > 1:
            continue
        if first_author:
            surnames_ours = _surnames_set(first_author)
            surnames_hit: set[str] = set()
            # authorString is "Surname F1, Surname F2, …"
            for seg in (h.get("authorString") or "").split(","):
                surnames_hit |= _surnames_set(seg.strip())
            if surnames_ours and surnames_hit and not (surnames_ours & surnames_hit):
                continue
        doi = (h.get("doi") or "").strip()
        try:
            doi = normalize_doi(doi) if doi else None
        except Exception:
            doi = None
        return {
            "doi": doi,
            "pmid": h.get("pmid"),
            "pmcid": h.get("pmcid"),
            "title": h_title,
            "year": h_year,
            "authors": [
                {"name": seg.strip()}
                for seg in (h.get("authorString") or "").split(",")
                if seg.strip()
            ],
            "journal": h.get("journalTitle"),
            "source_id": h.get("source"),   # e.g. MED, PMC, AGR
        }
    return None


# ── 7. arXiv title-search (preprints) ──────────────────────────────────

def search_arxiv_by_title(
    title: str,
    *,
    first_author: str | None = None,
    year: int | None = None,
    timeout: float = 15.0,
) -> dict | None:
    """Search arXiv by title via export.arxiv.org/api/query.

    The existing arXiv metadata layer (``metadata._layer_arxiv``) only
    runs when an arXiv ID is already known. But many preprints in the
    corpus arrive as journal-version PDFs whose text never mentions
    their arXiv counterpart. arXiv's Atom-XML API exposes
    ``search_query=ti:"..." AND au:"..."`` which catches these.

    Returns:
        Dict with {arxiv_id, doi, title, year, authors} or None.
    """
    import xml.etree.ElementTree as ET
    if not title or len(title.strip()) < 10:
        return None
    # arXiv API expects quoted phrase search with AND between clauses.
    safe_title = re.sub(r'"', "", title[:200])
    query_parts = [f'ti:"{safe_title}"']
    if first_author:
        surname = _surnames_set(first_author)
        if surname:
            query_parts.append(f'au:{next(iter(surname))}')
    search_query = " AND ".join(query_parts)
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": 5,
    }
    headers = {"User-Agent": f"sciknow/0.1 (mailto:{settings.crossref_email})"}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                "http://export.arxiv.org/api/query",
                params=params, headers=headers,
            )
        if resp.status_code != 200:
            return None
        # Parse the Atom XML. arXiv returns 200 + empty <feed/> on no-hit.
        ns = {
            "a": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(resp.text)
    except Exception as exc:
        logger.debug("arXiv title-search error: %s", exc)
        return None

    for entry in root.findall("a:entry", ns):
        e_title = (entry.find("a:title", ns) or ET.Element("x")).text or ""
        e_title = re.sub(r"\s+", " ", e_title).strip()
        if not e_title:
            continue
        if _loose_title_match(title, e_title) < 0.85:
            continue
        # Pull year from published.
        published = (entry.find("a:published", ns) or ET.Element("x")).text or ""
        try:
            e_year = int(published[:4]) if published else None
        except Exception:
            e_year = None
        if year and e_year and abs(e_year - int(year)) > 1:
            continue
        # The arxiv_id is the last path component of the id URL.
        id_url = (entry.find("a:id", ns) or ET.Element("x")).text or ""
        arxiv_id = id_url.rsplit("/", 1)[-1] if id_url else None
        # Optional DOI element (arxiv:doi).
        doi_el = entry.find("arxiv:doi", ns)
        doi = (doi_el.text.strip() if doi_el is not None and doi_el.text else None)
        if doi:
            try:
                doi = normalize_doi(doi)
            except Exception:
                doi = None
        authors = []
        for a in entry.findall("a:author", ns):
            n_el = a.find("a:name", ns)
            if n_el is not None and n_el.text:
                authors.append({"name": n_el.text.strip()})
        return {
            "arxiv_id": arxiv_id,
            "doi": doi,
            "title": e_title,
            "year": e_year,
            "authors": authors,
        }
    return None


# ── 7. OpenLibrary (books) ──────────────────────────────────────────────

def search_openlibrary_by_title(
    title: str,
    *,
    first_author: str | None = None,
    timeout: float = 15.0,
) -> dict | None:
    """Search OpenLibrary for a book by title + author.

    Returns the first record with an ISBN-13 (or ISBN-10) and the
    best-guess LC subject classifications. No API key; the identifying
    User-Agent gets us the 3-RPS polite pool.

    Returns:
        Dict with {isbn_13, isbn_10, lccn, subjects, title, authors,
                    publish_year, openlibrary_key} or None.
    """
    if not title or len(title.strip()) < 4:
        return None

    params = {
        "title": title[:200],
        "fields": "key,title,author_name,first_publish_year,isbn,lccn,subject,publisher",
        "limit": 5,
    }
    if first_author:
        params["author"] = first_author[:120]

    headers = {
        "User-Agent": (
            f"sciknow/0.1 (mailto:{settings.crossref_email})"
        ),
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                "https://openlibrary.org/search.json",
                params=params, headers=headers,
            )
        if resp.status_code != 200:
            return None
        body = resp.json() or {}
    except Exception as exc:
        logger.debug("OpenLibrary network error: %s", exc)
        return None

    for doc in body.get("docs") or []:
        hit_title = (doc.get("title") or "").strip()
        if not hit_title:
            continue
        if _loose_title_match(title, hit_title) < 0.82:
            continue
        isbns = doc.get("isbn") or []
        isbn_13 = next((x for x in isbns if len(x) == 13), None)
        isbn_10 = next((x for x in isbns if len(x) == 10), None)
        if not (isbn_13 or isbn_10):
            continue
        return {
            "isbn_13": isbn_13,
            "isbn_10": isbn_10,
            "lccn": (doc.get("lccn") or [None])[0],
            "subjects": (doc.get("subject") or [])[:15],
            "title": hit_title,
            "authors": [{"name": n} for n in (doc.get("author_name") or [])],
            "publish_year": doc.get("first_publish_year"),
            "openlibrary_key": doc.get("key"),
            "publisher": (doc.get("publisher") or [None])[0],
        }
    return None


# ── 8.5. LLM-assisted title recovery (last-resort) ──────────────────────

def llm_recover_title(
    pdf_path: Path | str,
    *,
    max_chars: int = 3500,
) -> str | None:
    """Ollama-based title recovery when geometric layout fails.

    Extracts the first ``max_chars`` of the PDF as text and asks the
    fast LLM to return the paper's canonical title. Does NOT touch
    abstract/authors/year — those are recovered by the normal
    metadata pipeline once we have a title and hit Crossref. Keeping
    the prompt narrow (title-only) minimises hallucination + tokens.

    Returns:
        Cleaned title string (or None on failure / hallucinated garbage).
    """
    try:
        doc = pymupdf.open(str(pdf_path))
        text = ""
        for i in range(min(3, doc.page_count)):
            try:
                text += doc.load_page(i).get_text("text") + "\n"
            except Exception:
                pass
        doc.close()
        if not text.strip():
            return None
        text = text[:max_chars]
    except Exception as exc:
        logger.debug("llm_recover_title: PDF open failed: %s", exc)
        return None

    # V2_FINAL Stage 2: route via rag.llm.complete (dispatches to
    # llama-server when USE_LLAMACPP_WRITER=True, Ollama otherwise).
    prompt = (
        "You will be given the first few pages of a scientific paper. "
        "Return ONLY the paper's title as a single line of text — no "
        "JSON, no labels, no quotes, no author names, no year, no "
        "journal. If you cannot identify a clear title (e.g. the text "
        "is a blog post, a government report, a letter, or otherwise "
        "untitled), return exactly the string NO_TITLE.\n\n"
        "Text:\n" + text
    )
    try:
        from sciknow.rag.llm import complete as _llm_complete
        raw = (_llm_complete(
            "You are a careful scientific bibliographer.",
            prompt,
            model=settings.llm_fast_model,
            temperature=0,
            num_predict=120,
            keep_alive=-1,
        ) or "").strip()
    except Exception as exc:
        logger.debug("llm_recover_title: LLM call failed: %s", exc)
        return None

    if not raw or raw == "NO_TITLE":
        return None
    # Strip common wrappers the model sometimes emits despite the prompt.
    raw = raw.strip('"\'*_#').strip()
    # One line only — in case the model wrote "Title: X" or similar.
    raw = raw.split("\n", 1)[0].strip()
    raw = re.sub(r"^(?i)title[:\s]+", "", raw).strip()
    if len(raw) < 10 or len(raw) > 300:
        return None
    # Reject obviously-hallucinated outputs.
    lowered = raw.lower()
    for bad in ("no title", "unknown", "not applicable", "n/a"):
        if bad in lowered:
            return None
    return raw


# ── 8. Title-recovery from PDF (fallback for garbage-titled rows) ───────

def recover_title_from_pdf(
    pdf_path: Path | str,
    *,
    max_pages: int = 2,
) -> str | None:
    """Best-effort recovery of a paper title from the first 1–2 pages.

    Papers with garbage db titles like ``iau1200511a`` or ``Mishev-2.dvi``
    often have the REAL title printed as the largest text block near the
    top of page 1. We use PyMuPDF's dict layout to find the span with
    the largest font size in the top third of page 1, then sanity-check.

    Returns:
        A candidate title string suitable for title-search fuzzy match,
        or None on no signal.
    """
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception:
        return None
    try:
        if doc.page_count == 0:
            return None
        best_text = ""
        best_size = 0.0
        for pg in range(min(max_pages, doc.page_count)):
            try:
                page = doc.load_page(pg)
                d = page.get_text("dict")
            except Exception:
                continue
            ph = (page.rect.height or 1000)
            for block in d.get("blocks", []):
                for line in block.get("lines", []):
                    # Take the first 40 % of the page (title band).
                    y = line.get("bbox", [0, 0, 0, 0])[1]
                    if y > ph * 0.4:
                        continue
                    line_text = " ".join(
                        span.get("text", "") for span in line.get("spans", [])
                    ).strip()
                    if len(line_text) < 10:
                        continue
                    # Pick the max font size in this line.
                    sizes = [
                        float(span.get("size") or 0)
                        for span in line.get("spans", [])
                    ]
                    ms = max(sizes) if sizes else 0.0
                    if ms > best_size and ms >= 11.0:
                        best_size = ms
                        best_text = line_text
            # Early out if we already have a strong candidate on page 1.
            if best_text and pg == 0 and best_size >= 13:
                break
    finally:
        doc.close()

    if not best_text:
        return None
    # Strip obvious junk prefixes.
    t = re.sub(r"^\s*\*+\s*", "", best_text).strip()
    # Reject if looks like a section/boilerplate.
    lowered = t.lower()
    for bad in (
        "abstract", "introduction", "acknowledgements", "references",
        "figure", "table", "keywords",
    ):
        if lowered.startswith(bad):
            return None
    if len(t) < 12:
        return None
    return t[:300]


# ── Helpers ─────────────────────────────────────────────────────────────

def _surnames_set(name: str) -> set[str]:
    """Return a set of lowercased surname tokens for a person name.

    Handles "Smith, John", "John Smith", "John van der Smith", and
    "S. J. Smith" by taking the last whitespace-separated token as
    the surname. Multi-word surnames ("van der Smith") are imperfect
    but the intersection check only needs one surname to match.
    """
    if not name:
        return set()
    s = name.strip()
    if "," in s:
        s = s.split(",", 1)[0]
    parts = [p.strip() for p in s.split() if p.strip()]
    if not parts:
        return set()
    surname = parts[-1].lower()
    # Strip trailing punctuation (e.g. "Smith.")
    surname = re.sub(r"[^a-z0-9]+$", "", surname)
    # Ignore very short tokens (initials).
    if len(surname) < 2:
        return set()
    return {surname}


def _loose_title_match(a: str, b: str) -> float:
    """Token-set Jaccard-ish similarity on normalised titles.

    Cheap approximation of rapidfuzz.fuzz.token_set_ratio — good
    enough for a gating check on DataCite / OpenLibrary candidates.
    Returns 0.0–1.0.
    """
    if not a or not b:
        return 0.0
    ta = set(re.sub(r"[^a-z0-9 ]", " ", a.lower()).split())
    tb = set(re.sub(r"[^a-z0-9 ]", " ", b.lower()).split())
    # Drop very common stop-words so "The" prefix swaps don't drag score.
    stop = {"a", "an", "the", "of", "and", "for", "in", "on", "to", "with"}
    ta -= stop
    tb -= stop
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    # Dice-ish: 2|A∩B| / (|A| + |B|)
    return 2.0 * len(inter) / (len(ta) + len(tb))
