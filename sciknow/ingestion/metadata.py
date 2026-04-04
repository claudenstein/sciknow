"""
4-layer metadata extraction:
  1. PyMuPDF  — embedded PDF fields
  2. Crossref — authoritative by DOI
  3. arXiv    — for preprints by arXiv ID
  4. LLM      — Ollama fallback from markdown text
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pymupdf  # fitz

from sciknow.config import settings
from sciknow.utils.doi import extract_arxiv_id, extract_doi, normalize_doi


@dataclass
class PaperMeta:
    title: str | None = None
    abstract: str | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    publisher: str | None = None
    authors: list[dict] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    source: str = "unknown"
    crossref_raw: dict | None = None
    arxiv_raw: dict | None = None


def extract(pdf_path: Path, markdown_text: str) -> PaperMeta:
    meta = PaperMeta()

    # Layer 1: embedded PDF metadata
    _layer_pymupdf(pdf_path, meta)

    # Scan full text for identifiers if not found in metadata
    scan_text = markdown_text[:8000]
    if not meta.doi:
        doi = extract_doi(scan_text)
        if doi:
            meta.doi = normalize_doi(doi)
    if not meta.arxiv_id:
        meta.arxiv_id = extract_arxiv_id(scan_text)

    # Layer 2: Crossref (by DOI)
    if meta.doi:
        _layer_crossref(meta)

    # Layer 3: arXiv
    if meta.arxiv_id and not meta.title:
        _layer_arxiv(meta)

    # Layer 4: LLM fallback — also runs if PyMuPDF returned a garbage title
    if not meta.title or _is_garbage_title(meta.title):
        meta.title = None
        # Try fast heading extraction from markdown before hitting the LLM
        meta.title = _extract_title_from_markdown(markdown_text)
        if not meta.title:
            _layer_llm(markdown_text[:3000], meta)

    # Last resort: use a cleaned-up filename so title is never null
    if not meta.title:
        meta.title = _title_from_filename(pdf_path)

    return meta


# Patterns that indicate a PDF title field contains a filename or template string
# rather than the actual paper title.
_GARBAGE_TITLE_RE = re.compile(
    r'(?i)'
    r'(^microsoft\s+word\s*[-–])'                       # "Microsoft Word - filename.docx"
    r'|(\.docx?|\.pdf|\.pptx?|\.xlsx?|\.indd)\s*$'      # ends with a file extension
    r'|(^untitled[-\s]*\d*$)'                           # "Untitled", "Untitled-1"
    r'|(^document\d*$)'                                 # "Document", "Document1"
    r'|(^presentation\d*$)'
    r'|(^doi\s*:)'                                      # "doi:10.1016/..."
    r'|(^10\.\d{4,}/)'                                  # bare DOI starting with "10."
    r'|(^[a-z]{1,6}\d+$)'                               # short codes: "Avery6", "Avery8"
    r'|(^(preprint|draft|confidential|final|version)\s*[\d.]*$)'  # status markers
    r'|(.*_.*_.*)'                                      # 2+ underscores → report/filename code
    r'|(version\s+\d+\.\d+\.indd)'                      # InDesign versioned filename
    r'|(intechopen.*publisher)'                          # "We are IntechOpen, the world's leading publisher..."
    r'|(world.s leading publisher)'
    r'|(^pii\s*:)'                                      # "PII: S1364-..."
    r'|(^s\d{4}-\d{4})'                                 # bare PII like "S1364-6826..."
    r'|(print\s+kdp)'                                   # Kindle Direct Publishing artefact
)

# Sentence openers that indicate the "title" is actually body text
_SENTENCE_OPENER_RE = re.compile(
    r'(?i)^(several of|based on|in this|we show|we present|this paper|the paper'
    r'|in the present|the purpose|the aim|the goal|it is|there (is|are)'
    r'|here we|our results|results show)',
)


def _is_garbage_title(title: str) -> bool:
    """Return True if the title looks like a filename, editor artefact, or body text."""
    t = title.strip()
    if len(t) < 6:
        return True
    if _is_mostly_non_ascii(t):
        return True
    if _GARBAGE_TITLE_RE.search(t):
        return True
    # Titles over 200 chars are almost certainly body text, not titles
    if len(t) > 200:
        return True
    # Sentence openers indicate LLM returned body text instead of a title
    if _SENTENCE_OPENER_RE.match(t):
        return True
    return False


def _is_mostly_non_ascii(s: str, threshold: float = 0.3) -> bool:
    """Return True if more than `threshold` fraction of chars are non-ASCII."""
    if not s:
        return False
    non_ascii = sum(1 for c in s if ord(c) > 127)
    return (non_ascii / len(s)) > threshold


# Patterns that indicate an author field contains a system/tool string rather
# than a real person's name.
_GARBAGE_AUTHOR_RE = re.compile(
    r'(?i)'
    r'(^administrator$)'
    r'|(^admin$)'
    r'|(^user$)'
    r'|(^owner$)'
    r'|(publishing)'          # "IOP Publishing", "Elsevier Publishing"
    r'|(^unknown$)'
    r'|(\d{4}\s+\w+\s+\d)'   # date strings like "2014 Oct 9"
    r'|(intechopen)'          # publisher boilerplate
    r'|(working\s+group)'     # "IPCC Working Group"
    r'|(TSU\b)'               # "WGII TSU" (Technical Support Unit)
)

# Chars that appear in garbled/mis-encoded strings but not real names
_GARBLE_CHARS_RE = re.compile(r'[<>=@\\|~^]')


def _is_garbage_author(name: str) -> bool:
    """Return True if the author name is clearly not a real person."""
    n = name.strip()
    if len(n) < 2:
        return True
    if _is_mostly_non_ascii(n, threshold=0.4):
        return True
    # Garbled encoding produces strings rich in punctuation/symbol chars
    garble_count = len(_GARBLE_CHARS_RE.findall(n))
    if garble_count >= 2 or (garble_count >= 1 and len(n) < 15):
        return True
    # Email-style username: all-lowercase, contains dot, no spaces (e.g. "james.marusek")
    if "." in n and " " not in n and n == n.lower() and n.replace(".", "").isalpha():
        return True
    return bool(_GARBAGE_AUTHOR_RE.search(n))


# ── Title helpers ──────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r'^#{1,3}\s+(.+)', re.MULTILINE)
# Lines that look like section headings rather than paper titles
_SECTION_WORDS_RE = re.compile(
    r'(?i)^(abstract|introduction|background|contents?|table of|references?'
    r'|acknowledgem|appendix|chapter\s+\d|section\s+\d)',
)


def _strip_markdown(text: str) -> str:
    """Remove common inline markdown and HTML formatting from a string."""
    text = re.sub(r'<[^>]+>', '', text)                    # HTML tags (<sup>, <i>, etc.)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)  # bold/italic
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)    # underscore bold/italic
    text = re.sub(r'`([^`]+)`', r'\1', text)               # inline code
    text = re.sub(r'^\[+', '', text)                       # leading [ from link/citation refs
    text = re.sub(r'\]+$', '', text)                       # trailing ]
    return text.strip()


def _extract_title_from_markdown(text: str) -> str | None:
    """
    Try to extract the paper title from the first substantive heading in the
    converted markdown. Skips headings that look like section names.
    """
    for m in _HEADING_RE.finditer(text[:5000]):
        candidate = _strip_markdown(m.group(1))
        # Skip very short or clearly section-level headings
        if len(candidate) < 10:
            continue
        if _SECTION_WORDS_RE.match(candidate):
            continue
        if _is_garbage_title(candidate):
            continue
        return candidate
    return None


def _title_from_filename(pdf_path: Path) -> str:
    """Derive a display title from the PDF filename as an absolute last resort."""
    stem = pdf_path.stem
    # Remove common prefixes like "0-", leading numbers, etc.
    stem = re.sub(r'^[\d\s\-_]+', '', stem)
    # Replace underscores/hyphens with spaces
    stem = re.sub(r'[-_]+', ' ', stem).strip()
    return stem if len(stem) >= 3 else pdf_path.stem


# ---------------------------------------------------------------------------
# Layer 1: PyMuPDF
# ---------------------------------------------------------------------------

def _layer_pymupdf(pdf_path: Path, meta: PaperMeta) -> None:
    try:
        doc = pymupdf.open(str(pdf_path))
        info = doc.metadata or {}
        doc.close()

        if info.get("title"):
            meta.title = _strip_markdown(info["title"]) or None
        if info.get("author"):
            for name in re.split(r'[;,]', info["author"]):
                name = name.strip()
                if name and not _is_garbage_author(name):
                    meta.authors.append({"name": name})
        if info.get("subject"):
            meta.keywords = [k.strip() for k in info["subject"].split(",") if k.strip()]
        if info.get("creationDate"):
            year = _parse_pdf_year(info["creationDate"])
            if year:
                meta.year = year

        if meta.title:
            meta.source = "embedded_pdf"
    except Exception:
        pass


def _parse_pdf_year(date_str: str) -> int | None:
    match = re.search(r'(\d{4})', date_str)
    if match:
        y = int(match.group(1))
        if 1900 <= y <= 2100:
            return y
    return None


# ---------------------------------------------------------------------------
# Layer 2: Crossref
# ---------------------------------------------------------------------------

def _layer_crossref(meta: PaperMeta) -> None:
    url = f"https://api.crossref.org/works/{meta.doi}"
    headers = {
        "User-Agent": f"sciknow/0.1 (mailto:{settings.crossref_email})"
    }
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, headers=headers)
        if resp.status_code != 200:
            return

        data = resp.json().get("message", {})
        meta.crossref_raw = data

        if data.get("title"):
            meta.title = _strip_markdown(data["title"][0])
        if data.get("abstract"):
            # Crossref abstracts often have JATS XML tags
            meta.abstract = re.sub(r'<[^>]+>', '', data["abstract"]).strip()
        if data.get("published"):
            parts = data["published"].get("date-parts", [[]])[0]
            if parts:
                meta.year = int(parts[0])
        if data.get("container-title"):
            import html as _html
            meta.journal = _html.unescape(data["container-title"][0])
        if data.get("volume"):
            meta.volume = str(data["volume"])
        if data.get("issue"):
            meta.issue = str(data["issue"])
        if data.get("page"):
            meta.pages = data["page"]
        if data.get("publisher"):
            import html as _html
            meta.publisher = _html.unescape(data["publisher"])
        if data.get("author"):
            meta.authors = [
                {
                    "name": f"{a.get('given', '')} {a.get('family', '')}".strip(),
                    "orcid": a.get("ORCID", "").split("/")[-1] or None,
                    "affiliation": a.get("affiliation", [{}])[0].get("name") if a.get("affiliation") else None,
                }
                for a in data["author"]
            ]
        if data.get("subject"):
            meta.keywords = data["subject"]

        meta.source = "crossref"
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Crossref title search (for papers without a DOI)
# ---------------------------------------------------------------------------

def search_crossref_by_title(
    title: str,
    first_author: str | None = None,
    threshold: float = 0.85,
) -> "PaperMeta | None":
    """
    Search Crossref by title (and optionally first author) to find a DOI.
    Returns a fully populated PaperMeta if a confident match is found,
    or None if no match exceeds the similarity threshold.
    """
    import difflib
    import html as _html

    params: dict = {
        "query.title": title,
        "rows": 5,
        "select": "DOI,title",
    }
    if first_author:
        params["query.author"] = first_author

    url = "https://api.crossref.org/works"
    headers = {"User-Agent": f"sciknow/0.1 (mailto:{settings.crossref_email})"}

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            return None

        items = resp.json().get("message", {}).get("items", [])
        if not items:
            return None

        # Normalise for comparison: lowercase, strip non-alphanumeric
        def _norm(s: str) -> str:
            return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()

        norm_query = _norm(_html.unescape(title))

        best_doi: str | None = None
        best_score = 0.0
        for item in items:
            item_titles = item.get("title", [])
            if not item_titles:
                continue
            item_doi = item.get("DOI", "")
            if not item_doi:
                continue
            norm_item = _norm(_html.unescape(item_titles[0]))
            score = difflib.SequenceMatcher(None, norm_query, norm_item).ratio()
            if score > best_score:
                best_score = score
                best_doi = item_doi

        if best_score < threshold or not best_doi:
            return None

        # Fetch full metadata via the standard Crossref layer
        meta = PaperMeta(doi=normalize_doi(best_doi))
        _layer_crossref(meta)
        return meta if meta.title else None

    except Exception:
        return None


def search_openalex_by_title(
    title: str,
    first_author: str | None = None,
    threshold: float = 0.85,
) -> "PaperMeta | None":
    """
    Search OpenAlex by title as a fallback to Crossref.
    OpenAlex covers many works Crossref misses (e.g. some preprints, book chapters).
    Returns a PaperMeta with at least a DOI set, or None on no confident match.
    """
    import difflib
    import html as _html

    params: dict = {
        "search": title,
        "per-page": 5,
        "select": "doi,title,authorships,publication_year,primary_location,abstract_inverted_index",
        "mailto": settings.crossref_email,
    }
    if first_author:
        # OpenAlex author filter is approximate; just add to the text search
        params["search"] = f"{title} {first_author}"

    url = "https://api.openalex.org/works"
    headers = {"User-Agent": f"sciknow/0.1 (mailto:{settings.crossref_email})"}

    def _norm(s: str) -> str:
        return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            return None

        items = resp.json().get("results", [])
        if not items:
            return None

        norm_query = _norm(_html.unescape(title))

        best: dict | None = None
        best_score = 0.0
        for item in items:
            item_title = item.get("title") or ""
            item_doi = item.get("doi") or ""
            if not item_title or not item_doi:
                continue
            # OpenAlex DOIs come as full URLs: "https://doi.org/10.xxx/yyy"
            norm_item = _norm(_html.unescape(item_title))
            score = difflib.SequenceMatcher(None, norm_query, norm_item).ratio()
            if score > best_score:
                best_score = score
                best = item

        if best_score < threshold or best is None:
            return None

        raw_doi = best.get("doi", "")
        # Strip leading URL if present
        if "doi.org/" in raw_doi:
            raw_doi = raw_doi.split("doi.org/")[-1]
        doi = normalize_doi(raw_doi)
        if not doi:
            return None

        # Fetch full metadata via Crossref (authoritative) if we got a DOI
        meta = PaperMeta(doi=doi)
        _layer_crossref(meta)

        # If Crossref didn't fill it in, build from OpenAlex data directly
        if not meta.title:
            import html as _html2
            meta.title = _html2.unescape(best.get("title") or "")
            yr = best.get("publication_year")
            if yr:
                meta.year = int(yr)
            for a in best.get("authorships", []):
                name = (a.get("author") or {}).get("display_name", "")
                if name:
                    meta.authors.append({"name": name})
            loc = best.get("primary_location") or {}
            src = (loc.get("source") or {}).get("display_name", "")
            if src:
                meta.journal = src
            meta.source = "openalex"

        return meta if meta.title else None

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Layer 3: arXiv
# ---------------------------------------------------------------------------

def _layer_arxiv(meta: PaperMeta) -> None:
    try:
        import arxiv

        search = arxiv.Search(id_list=[meta.arxiv_id], max_results=1)
        results = list(search.results())
        if not results:
            return

        result = results[0]
        meta.arxiv_raw = {
            "entry_id": result.entry_id,
            "title": result.title,
            "summary": result.summary,
            "published": result.published.isoformat() if result.published else None,
            "categories": result.categories,
        }

        if not meta.title:
            meta.title = result.title
        if not meta.abstract:
            meta.abstract = result.summary
        if not meta.year and result.published:
            meta.year = result.published.year
        if not meta.authors:
            meta.authors = [{"name": str(a)} for a in result.authors]
        if not meta.keywords:
            meta.keywords = result.categories

        meta.source = "arxiv"
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Layer 4: LLM fallback (Ollama)
# ---------------------------------------------------------------------------

_LLM_PROMPT = """\
Extract bibliographic metadata from the beginning of this scientific paper.
Return ONLY valid JSON with these fields (use null for missing fields):
{
  "title": "...",
  "abstract": "...",
  "year": 2023,
  "authors": [{"name": "First Last"}, ...],
  "journal": "...",
  "doi": "..."
}

Paper text:
{text}
"""


def _layer_llm(text: str, meta: PaperMeta) -> None:
    try:
        import ollama

        client = ollama.Client(host=settings.ollama_host)
        response = client.chat(
            model=settings.llm_fast_model,
            messages=[{"role": "user", "content": _LLM_PROMPT.format(text=text)}],
            format="json",
            options={"temperature": 0},
        )

        raw = response.message.content
        data = json.loads(raw)

        if data.get("title"):
            meta.title = _strip_markdown(str(data["title"]))
        if data.get("abstract"):
            meta.abstract = data["abstract"]
        if data.get("year"):
            try:
                meta.year = int(data["year"])
            except (ValueError, TypeError):
                pass
        if data.get("authors"):
            meta.authors = data["authors"]
        if data.get("journal"):
            meta.journal = data["journal"]
        if data.get("doi") and not meta.doi:
            meta.doi = normalize_doi(str(data["doi"]))

        meta.source = "llm_extracted"
    except Exception:
        meta.source = "unknown"
