"""Research bibliography for sciknow itself.

Walks ``docs/`` recursively, extracts citations to external papers
(arXiv, ACL Anthology, DOI links), and downloads the open-access PDFs
into ``<repo>/bibliography/`` so the literature underpinning sciknow's
design is browsable as a directory. Future-facing: when we eventually
write a whitepaper about how sciknow works, the source set is already
curated.

Sources handled:
  * arXiv URLs ``arxiv.org/(abs|pdf|html)/<id>`` — direct PDF fetch.
  * ACL Anthology URLs ``aclanthology.org/<paper-id>`` — direct PDF
    fetch (Anthology serves PDFs from the same path with ``.pdf``).
  * DOI URLs ``doi.org/<doi>`` — best-effort: tries Unpaywall for an
    OA copy. Logged as "no OA found" if nothing's available.

The folder is gitignored — derived artifact, regeneratable from the
docs at any time.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Citation extraction
# ──────────────────────────────────────────────────────────────────────


_ARXIV_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/(?P<id>\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
_ACL_RE = re.compile(
    r"aclanthology\.org/(?P<id>\d{4}\.[A-Za-z0-9\-]+\.\d+)",
    re.IGNORECASE,
)
_DOI_RE = re.compile(
    r"doi\.org/(?P<doi>10\.\d{4,9}/[^\s)\]\}\"]+?)(?=[)\]\}\.\s\"]|$)",
    re.IGNORECASE,
)


@dataclass
class Citation:
    """One external paper referenced in the docs.

    ``key`` is the canonical filename stem we'll save the PDF as
    (e.g. ``arxiv__2401.18059``, ``acl__2023.emnlp-main.741``,
    ``doi__10.1162-tacl_a_00601``).
    ``urls`` is the list of distinct PDF URLs we'll try in order.
    ``cited_in`` is the set of doc paths (relative to repo root) that
    referenced this paper, for the index file.
    """
    key: str
    kind: str          # arxiv | acl | doi
    identifier: str    # the raw id / doi
    urls: list[str]
    cited_in: set[str] = field(default_factory=set)


def _doi_filename_safe(doi: str) -> str:
    """Make a DOI safe for use as a filename component.

    DOIs contain ``/`` (which is a path separator) and frequently
    ``.`` (fine in filenames). Replace ``/`` with ``-`` so the entire
    DOI sits in one filename component.
    """
    return doi.replace("/", "-")


def extract_citations_from_text(text: str) -> list[Citation]:
    """Return Citation objects for every recognised reference in
    ``text``. Idempotent / order-preserving: duplicates are merged."""
    out: dict[str, Citation] = {}

    for m in _ARXIV_RE.finditer(text):
        aid = m.group("id")
        key = f"arxiv__{aid}"
        url = f"https://arxiv.org/pdf/{aid}.pdf"
        out.setdefault(key, Citation(key=key, kind="arxiv",
                                      identifier=aid, urls=[url]))

    for m in _ACL_RE.finditer(text):
        pid = m.group("id")
        key = f"acl__{pid}"
        url = f"https://aclanthology.org/{pid}.pdf"
        out.setdefault(key, Citation(key=key, kind="acl",
                                      identifier=pid, urls=[url]))

    for m in _DOI_RE.finditer(text):
        doi = m.group("doi").rstrip(".,;)")
        key = f"doi__{_doi_filename_safe(doi)}"
        out.setdefault(key, Citation(key=key, kind="doi",
                                      identifier=doi, urls=[]))

    return list(out.values())


def discover_citations(docs_root: Path) -> list[Citation]:
    """Walk ``docs_root`` recursively, extract every citation, merge
    duplicates across files, and record where each was cited."""
    merged: dict[str, Citation] = {}
    for md in docs_root.rglob("*.md"):
        try:
            content = md.read_text(encoding="utf-8")
        except OSError:
            continue
        rel = md.relative_to(docs_root.parent).as_posix()
        for c in extract_citations_from_text(content):
            existing = merged.get(c.key)
            if existing is None:
                merged[c.key] = c
                c.cited_in.add(rel)
            else:
                existing.cited_in.add(rel)
    # Stable order: by kind, then key.
    return sorted(merged.values(), key=lambda c: (c.kind, c.key))


# ──────────────────────────────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────────────────────────────


@dataclass
class DownloadResult:
    bibliography_dir: Path
    n_total: int = 0
    n_already_present: int = 0
    n_downloaded: int = 0
    n_failed: int = 0
    n_skipped_no_oa: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.n_total} citations · "
            f"already present: {self.n_already_present} · "
            f"downloaded: {self.n_downloaded} · "
            f"no OA found: {self.n_skipped_no_oa} · "
            f"failed: {self.n_failed}"
        )


def _unpaywall_oa_url(doi: str, *, email: str, client) -> str | None:
    """Probe Unpaywall for an OA URL. Returns None if nothing found.

    Unpaywall is rate-limited to 100k requests/day with email contact;
    we pass the configured ``CROSSREF_EMAIL`` to be a polite client.
    """
    try:
        r = client.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": email}, timeout=20.0,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        loc = data.get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url")
    except Exception as exc:
        logger.debug("Unpaywall probe failed for %s: %s", doi, exc)
        return None


def _download_pdf(url: str, dst: Path, client) -> bool:
    """Streamed download. ``dst`` is written atomically — temp file
    first, rename on success — so a partial download never gets
    confused for a complete one."""
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with client.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
            if r.status_code != 200:
                logger.warning("HTTP %s on %s", r.status_code, url)
                return False
            with tmp.open("wb") as fp:
                for chunk in r.iter_bytes():
                    fp.write(chunk)
        if tmp.stat().st_size < 1024:
            # Suspiciously small — almost certainly an HTML error page.
            tmp.unlink(missing_ok=True)
            return False
        tmp.rename(dst)
        return True
    except Exception as exc:
        logger.warning("download %s failed: %s", url, exc)
        tmp.unlink(missing_ok=True)
        return False


def download_research_bibliography(
    docs_root: Path,
    *,
    bibliography_dir: Path,
    email: str | None = None,
    rate_limit_seconds: float = 0.5,
) -> DownloadResult:
    """Discover citations under ``docs_root``, download each to
    ``bibliography_dir``. Polite-rate by default.

    Args:
      docs_root: usually ``<repo>/docs``.
      bibliography_dir: usually ``<repo>/bibliography``.
      email: contact email for Unpaywall. Required for DOI fallback;
        without it DOI citations are skipped.
      rate_limit_seconds: minimum delay between download requests.
    """
    import httpx

    bibliography_dir.mkdir(parents=True, exist_ok=True)
    citations = discover_citations(docs_root)
    result = DownloadResult(bibliography_dir=bibliography_dir,
                            n_total=len(citations))

    if not citations:
        return result

    headers = {
        "User-Agent": "sciknow-research-bibliography/1.0 (https://github.com/claudenstein/sciknow)",
    }
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for i, c in enumerate(citations):
            target = bibliography_dir / f"{c.key}.pdf"
            if target.is_file() and target.stat().st_size >= 1024:
                result.n_already_present += 1
                continue

            urls = list(c.urls)
            if c.kind == "doi" and not urls:
                if not email:
                    result.n_skipped_no_oa += 1
                    continue
                oa = _unpaywall_oa_url(c.identifier, email=email, client=client)
                if oa:
                    urls.append(oa)
                else:
                    result.n_skipped_no_oa += 1
                    continue

            ok = False
            for url in urls:
                if _download_pdf(url, target, client):
                    ok = True
                    break
                time.sleep(rate_limit_seconds)
            if ok:
                result.n_downloaded += 1
            else:
                result.n_failed += 1
                result.failures.append((c.key, urls[0] if urls else "(no url)"))
            time.sleep(rate_limit_seconds)

    return result


# ──────────────────────────────────────────────────────────────────────
# Index file
# ──────────────────────────────────────────────────────────────────────


def write_index(citations: Iterable[Citation], bibliography_dir: Path) -> Path:
    """Write ``bibliography/INDEX.md`` listing every paper with the
    docs that cited it. Markdown so it renders cleanly in any reader.
    """
    citations = list(citations)
    by_kind: dict[str, list[Citation]] = {}
    for c in citations:
        by_kind.setdefault(c.kind, []).append(c)

    out: list[str] = [
        "# sciknow research bibliography — index",
        "",
        f"{len(citations)} unique papers cited across the docs.",
        "Generated automatically by `sciknow.core.research_bibliography`. ",
        "Don't edit by hand — re-run the build to refresh.",
        "",
    ]
    pretty = {"arxiv": "arXiv", "acl": "ACL Anthology", "doi": "DOI / journal"}
    for kind in ("arxiv", "acl", "doi"):
        items = by_kind.get(kind, [])
        if not items:
            continue
        out.append(f"## {pretty[kind]} ({len(items)})")
        out.append("")
        for c in sorted(items, key=lambda x: x.identifier):
            cited = ", ".join(sorted(c.cited_in))
            out.append(f"- `{c.key}.pdf` — `{c.identifier}` (cited in: {cited})")
        out.append("")

    path = bibliography_dir / "INDEX.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path
