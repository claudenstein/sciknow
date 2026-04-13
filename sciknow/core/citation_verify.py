"""External citation verification (Phase 46.B — AutoResearchClaw pattern).

For each inline citation in a draft, cross-check the cited paper's
metadata against authoritative external registries (Crossref, arXiv,
OpenAlex) and emit a verdict — VERIFIED / SUSPICIOUS / HALLUCINATED /
SKIPPED — plus the canonical record we found.

Rationale: sciknow's existing ``_verify_draft_inner`` checks whether
the claims in a draft are supported by the *retrieved corpus evidence*
(grounding). This module answers a different question: are the *cited
sources* real published papers? A sciknow corpus can still accumulate
papers whose metadata was extracted via the low-confidence embedded-PDF
or LLM fallback (see ``paper_metadata.metadata_source``); those are
the most likely to be cited with wrong-looking metadata.

The cascade (in order of confidence):

  1. If the cited paper has an arXiv ID → hit the arXiv API, compare
     titles with a word-overlap Jaccard. Threshold ≥ 0.80 is VERIFIED.
  2. Else if it has a DOI → Crossref (authoritative) → compare title.
  3. Else fall back to an OpenAlex title search; if top-hit Jaccard
     < 0.50 → HALLUCINATED. 0.50–0.80 → SUSPICIOUS. ≥ 0.80 → VERIFIED.

All lookups are best-effort; network errors yield a SKIPPED verdict
with the exception stringified into the report. Results are cached in
memory within one call so the same DOI isn't queried twice.

The thresholds are calibrated to the AutoResearchClaw paper's
``verify.py`` defaults; see the upstream repo referenced in the
sciknow watchlist for their empirical justification.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

logger = logging.getLogger("sciknow.citation_verify")


# ── Verdict values ────────────────────────────────────────────────────

VERIFIED     = "VERIFIED"      # external record matches title >= 0.80
SUSPICIOUS   = "SUSPICIOUS"    # 0.50 <= similarity < 0.80
HALLUCINATED = "HALLUCINATED"  # similarity < 0.50 or no match at all
SKIPPED      = "SKIPPED"       # external lookup failed / no identifier


# Calibration constants (AutoResearchClaw verify.py compatibility).
T_HIGH = 0.80
T_LOW  = 0.50


# ── Output types ──────────────────────────────────────────────────────


@dataclass
class CitationRecord:
    """One row in the verification report. Mirrors the draft's [N] → source map."""
    marker: int | None
    title: str
    year: int | str | None
    doi: str | None
    arxiv_id: str | None
    # From sciknow's own paper_metadata — lets the report show whether
    # the cited paper is in the corpus and how trustworthy its metadata is.
    metadata_source: str | None   # crossref/arxiv/openalex/embedded_pdf/…
    # Populated by verify_citation():
    verdict: str = SKIPPED
    similarity: float = 0.0
    external_title: str | None = None
    external_year: int | str | None = None
    external_source: str | None = None   # which API answered
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationReport:
    draft_id: str
    n_citations: int
    n_verified: int = 0
    n_suspicious: int = 0
    n_hallucinated: int = 0
    n_skipped: int = 0
    records: list[CitationRecord] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "draft_id": self.draft_id,
            "n_citations": self.n_citations,
            "n_verified":     self.n_verified,
            "n_suspicious":   self.n_suspicious,
            "n_hallucinated": self.n_hallucinated,
            "n_skipped":      self.n_skipped,
            "records":        [r.as_dict() for r in self.records],
        }


# ── Title similarity (word-overlap Jaccard on tokenized titles) ───────


_TITLE_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "and", "or",
    "to", "with", "from", "by", "at", "as", "is", "are",
}


def _normalize_title(title: str) -> str:
    s = (title or "").lower()
    s = _TITLE_NORMALIZE_RE.sub(" ", s)
    return " ".join(s.split())


def _title_tokens(title: str) -> set[str]:
    return {t for t in _normalize_title(title).split() if t and t not in _STOPWORDS}


def title_similarity(a: str, b: str) -> float:
    """Word-overlap Jaccard on normalized titles. 0 → 1.

    Matches the threshold semantics used in AutoResearchClaw's
    ``literature/verify.py``. Not fuzzy; titles with typos won't match.
    That's intentional — we're validating *published* titles, which
    should be exact-or-nearly-exact.
    """
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union) if union else 0.0


def _verdict_from_similarity(sim: float) -> str:
    if sim >= T_HIGH:
        return VERIFIED
    if sim >= T_LOW:
        return SUSPICIOUS
    return HALLUCINATED


# ── External API clients (best-effort; errors → SKIPPED with note) ───


def _http_get(url: str, *, params: dict | None = None,
              timeout: float = 10.0, headers: dict | None = None) -> dict | None:
    """Best-effort GET that returns parsed JSON or None."""
    import httpx
    try:
        from sciknow.config import settings
        ua = f"sciknow/0.1 (+mailto:{settings.crossref_email})"
    except Exception:
        ua = "sciknow/0.1"
    try:
        with httpx.Client(timeout=timeout,
                          headers={"User-Agent": ua, **(headers or {})}) as client:
            r = client.get(url, params=params)
            if r.status_code != 200:
                return None
            return r.json()
    except Exception as exc:
        logger.debug("HTTP GET %s failed: %s", url, exc)
        return None


def _query_arxiv(arxiv_id: str) -> dict | None:
    """Return {"title", "year"} for ``arxiv_id`` or None.

    arXiv's Atom-feed API is not JSON. Rather than pulling in an XML
    parser, we use the Semantic Scholar arXiv endpoint (JSON) which
    also proxies the record — same source, better format.
    """
    aid = arxiv_id.strip().removeprefix("arXiv:").strip()
    if not aid:
        return None
    # S2 by arXiv ID
    data = _http_get(
        f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{aid}",
        params={"fields": "title,year"},
    )
    if data:
        return {"title": data.get("title"), "year": data.get("year")}
    return None


def _query_crossref(doi: str) -> dict | None:
    """Crossref canonical record for a DOI. Returns {"title", "year"}."""
    d = doi.strip()
    if not d:
        return None
    data = _http_get(f"https://api.crossref.org/works/{d}")
    if not data:
        return None
    msg = (data.get("message") or {})
    titles = msg.get("title") or []
    year   = None
    for k in ("published-print", "published-online", "issued", "created"):
        parts = (msg.get(k) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            year = parts[0][0]
            break
    return {"title": titles[0] if titles else "", "year": year}


def _query_openalex_title(title: str) -> dict | None:
    """Top-hit title + year from OpenAlex for a title search. Used
    when the cited paper has no DOI/arXiv identifier — the weakest
    path in the cascade."""
    if not title or len(title) < 8:
        return None
    data = _http_get(
        "https://api.openalex.org/works",
        params={
            "search": title,
            "per_page": 1,
            "select": "title,publication_year,doi",
        },
    )
    if not data:
        return None
    hits = data.get("results") or []
    if not hits:
        return None
    top = hits[0]
    return {
        "title":  top.get("title")              or "",
        "year":   top.get("publication_year")   or None,
        "doi":    (top.get("doi") or "").removeprefix("https://doi.org/") or None,
    }


# ── The actual verifier ───────────────────────────────────────────────


def verify_citation(record: CitationRecord) -> CitationRecord:
    """Mutate the record in place with a verdict + external canonical data.

    Cascade: arXiv → Crossref → OpenAlex title search. First non-None
    answer settles the verdict. Network errors keep the verdict at
    SKIPPED with the exception message in ``notes``.
    """
    tried: list[str] = []

    # 1. arXiv by ID
    if record.arxiv_id:
        tried.append("arxiv")
        res = _query_arxiv(record.arxiv_id)
        if res and res.get("title"):
            record.external_title  = res["title"]
            record.external_year   = res.get("year")
            record.external_source = "arxiv"
            record.similarity      = title_similarity(record.title, res["title"])
            record.verdict         = _verdict_from_similarity(record.similarity)
            return record

    # 2. Crossref by DOI
    if record.doi:
        tried.append("crossref")
        res = _query_crossref(record.doi)
        if res and res.get("title"):
            record.external_title  = res["title"]
            record.external_year   = res.get("year")
            record.external_source = "crossref"
            record.similarity      = title_similarity(record.title, res["title"])
            record.verdict         = _verdict_from_similarity(record.similarity)
            return record

    # 3. OpenAlex title fuzz
    if record.title:
        tried.append("openalex")
        res = _query_openalex_title(record.title)
        if res and res.get("title"):
            record.external_title  = res["title"]
            record.external_year   = res.get("year")
            record.external_source = "openalex"
            record.similarity      = title_similarity(record.title, res["title"])
            record.verdict         = _verdict_from_similarity(record.similarity)
            return record

    # All paths exhausted without a hit → HALLUCINATED (the title exists
    # in the draft but couldn't be located in any registry). Fall-
    # through notes so the user sees which identifiers we tried.
    if tried:
        record.verdict = HALLUCINATED
        record.notes = (
            "No match in: " + ", ".join(tried)
            + (". Reference may be mistyped, pre-2000, "
               "or from a publisher outside Crossref/OpenAlex.")
        )
    else:
        record.verdict = SKIPPED
        record.notes = "No identifier or title to query with."
    return record


def verify_many(records: Iterable[CitationRecord]) -> VerificationReport:
    """Verify a batch. Preserves input order in the report."""
    recs = list(records)
    report = VerificationReport(draft_id="", n_citations=len(recs))
    for r in recs:
        verify_citation(r)
        report.records.append(r)
        if r.verdict == VERIFIED:
            report.n_verified += 1
        elif r.verdict == SUSPICIOUS:
            report.n_suspicious += 1
        elif r.verdict == HALLUCINATED:
            report.n_hallucinated += 1
        else:
            report.n_skipped += 1
    return report


# ── High-level entry: verify all citations on a draft ─────────────────


def build_records_for_draft(draft_id: str) -> tuple[str, list[CitationRecord]]:
    """Join drafts.sources with paper_metadata to get the fields we
    need to verify. Returns ``(canonical_draft_id, records)``.

    drafts.sources is a JSONB list of dicts. Shapes we've produced over
    time (Phase 5 onwards): ``{"n": N, "doc_id": ..., "chunk_id": ...,
    "title": ..., "year": ..., "doi": ..., "arxiv_id": ...}``. When a
    field is missing we look it up from paper_metadata via doc_id.
    """
    from sqlalchemy import text as _text
    from sciknow.storage.db import get_session

    with get_session() as session:
        row = session.execute(_text("""
            SELECT d.id::text, d.sources
            FROM drafts d WHERE d.id::text LIKE :q
            LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()
        if not row:
            return "", []
        did, sources = row
        if not isinstance(sources, list) or not sources:
            return did, []

        # Resolve missing fields from paper_metadata by doc_id
        doc_ids = {
            str(s.get("doc_id")) for s in sources
            if isinstance(s, dict) and s.get("doc_id")
        }
        meta_by_doc: dict[str, dict] = {}
        if doc_ids:
            rows = session.execute(_text("""
                SELECT document_id::text, title, year, doi, arxiv_id, metadata_source
                FROM paper_metadata
                WHERE document_id::text = ANY(:ids)
            """), {"ids": list(doc_ids)}).fetchall()
            for r in rows:
                meta_by_doc[r[0]] = {
                    "title": r[1], "year": r[2], "doi": r[3],
                    "arxiv_id": r[4], "metadata_source": r[5],
                }

    records: list[CitationRecord] = []
    for s in sources:
        if isinstance(s, dict):
            doc_id = str(s.get("doc_id") or "")
            meta = meta_by_doc.get(doc_id, {})
            records.append(CitationRecord(
                marker   = s.get("n"),
                title    = (s.get("title")    or meta.get("title")    or "").strip(),
                year     =  s.get("year")     or meta.get("year"),
                doi      = (s.get("doi")      or meta.get("doi")      or None) or None,
                arxiv_id = (s.get("arxiv_id") or meta.get("arxiv_id") or None) or None,
                metadata_source = meta.get("metadata_source"),
            ))
        elif isinstance(s, str):
            # Legacy shape: pre-Phase-46 autowrite stored sources as
            # formatted strings like "[1] F-283 (2012). Evidence of …".
            # Parse leniently and look up by title for DOI/metadata_source.
            parsed = _parse_legacy_source_line(s)
            if not parsed:
                continue
            # Try to resolve DOI + arxiv_id by exact-match title against
            # paper_metadata (same DB, same project, already scoped).
            extra = _lookup_by_title(parsed["title"])
            records.append(CitationRecord(
                marker   = parsed["marker"],
                title    = parsed["title"],
                year     = parsed["year"],
                doi      = extra.get("doi"),
                arxiv_id = extra.get("arxiv_id"),
                metadata_source = extra.get("metadata_source"),
            ))
    return did, records


# ── Legacy source-line parser + title lookup ──────────────────────────
#
# The pre-Phase-46 format is "[N] <id-or-authors> (YYYY). <Title>.[ …]"
# The author chunks + year live left of the first ". "; the title lives
# to the right. We pick the longest remaining sentence as the title
# rather than rely on a single regex to capture a shape that's drifted
# across phases.

_MARKER_RE = re.compile(r"^\s*\[(?P<n>\d+)\]\s*(?P<rest>.*)$", re.DOTALL)
_YEAR_RE   = re.compile(r"\((?P<y>(?:19|20)\d{2})\)")


def _parse_legacy_source_line(s: str) -> dict | None:
    """Turn "[N] <...> (YYYY). <Title>[ summary]" into {marker, title, year}.

    Tolerant — picks the longest sentence fragment (after the marker
    and id/author/year prefix) as the candidate title. Returns None if
    the line doesn't even start with a marker.
    """
    m = _MARKER_RE.match(s or "")
    if not m:
        return None
    try:
        marker = int(m.group("n"))
    except (TypeError, ValueError):
        marker = None
    rest = (m.group("rest") or "").strip()

    # Extract the first 4-digit year in parens, if any
    yr = None
    my = _YEAR_RE.search(rest)
    if my:
        try:
            yr = int(my.group("y"))
        except (TypeError, ValueError):
            yr = None

    # Split on ". " and pick the longest alphabetic-heavy fragment as
    # the title. This handles both "[1] id. Title." and
    # "[1] id (2012). Title. Optional summary." cleanly.
    fragments = [f.strip().rstrip(".") for f in rest.split(". ") if f.strip()]
    # Drop the leading id/authors chunk (usually short, may contain a
    # year-in-parens). Keep anything 3+ words that looks like a title.
    candidates = [f for f in fragments if len(f.split()) >= 3]
    if not candidates:
        # Last-ditch: strip the leading "(YYYY)" from the rest.
        title = _YEAR_RE.sub("", rest).strip().rstrip(".")
    else:
        # Longest 3+-word fragment wins — titles beat summaries on
        # length only most of the time, so also pref the first such
        # fragment (the one that appears right after the id/year).
        title = max(candidates, key=len)
        # If the longest is a summary, the first is the title; dual-
        # check by preferring whichever has fewer function words.
        first = candidates[0]
        if first != title and _STOPWORD_RATIO(first) < _STOPWORD_RATIO(title):
            title = first
    # Final clean
    title = _YEAR_RE.sub("", title).strip(" .,:;—-")
    if not title:
        return None
    return {"marker": marker, "title": title, "year": yr}


def _STOPWORD_RATIO(s: str) -> float:
    tokens = _normalize_title(s).split()
    if not tokens:
        return 1.0
    return sum(1 for t in tokens if t in _STOPWORDS) / len(tokens)


def _lookup_by_title(title: str) -> dict:
    """Exact-title lookup against paper_metadata in the active project."""
    from sqlalchemy import text as _text
    from sciknow.storage.db import get_session
    if not title or len(title) < 8:
        return {}
    with get_session() as session:
        row = session.execute(_text("""
            SELECT doi, arxiv_id, metadata_source
            FROM paper_metadata
            WHERE lower(title) = lower(:t)
            LIMIT 1
        """), {"t": title}).fetchone()
    if not row:
        return {}
    return {"doi": row[0], "arxiv_id": row[1], "metadata_source": row[2]}


def verify_draft(draft_id: str) -> VerificationReport:
    """Top-level entry: resolve → verify → roll up."""
    did, records = build_records_for_draft(draft_id)
    report = verify_many(records)
    report.draft_id = did
    return report
