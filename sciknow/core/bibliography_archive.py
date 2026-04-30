"""Phase 55.V19k — bibliography PDF archive.

Match strategy is two-stage to maximise the recovery rate:
  1. DOI: looks up ``paper_metadata.doi`` joined to ``documents``.
  2. Title fallback: when DOI is absent or didn't resolve, matches on
     a normalised title (lowercase, alphanumeric only) extracted from
     the APA source line. Covers entries whose APA string doesn't carry
     a DOI URL but whose paper IS in the corpus under a different
     identifier.


For each citable paper in a book's deduped bibliography, ensure a
human-friendly PDF is available under ``<project>/bibliography/`` so a
reader (or a future whitepaper draft) can browse the actual sources
that contributed to the book.

We don't duplicate the PDFs on disk; we **symlink** to whatever
``documents.original_path`` points at. The symlink target name is the
BibTeX citekey (e.g. ``Mrner2015_762a.pdf``) so it cross-references
1:1 with the exported ``refs.bib`` and the body's ``\\cite{}`` markers.

Excluded from git via ``.gitignore`` — the archive is a derived
artifact.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import text

from sciknow.core.bibliography import BookBibliography
from sciknow.core.project import get_active_project
from sciknow.formatting.bibtex import build_bibliography
from sciknow.storage.db import get_session

logger = logging.getLogger(__name__)


@dataclass
class ArchiveResult:
    """Outcome of one ``archive_book_bibliography`` run."""
    bibliography_dir: Path
    n_entries: int
    n_already_present: int = 0
    n_linked: int = 0
    n_no_local_pdf: int = 0
    n_doi_unmatched: int = 0
    missing_titles: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"bibliography: {self.n_entries} entries · "
            f"linked: {self.n_linked} · "
            f"already present: {self.n_already_present} · "
            f"no local PDF: {self.n_no_local_pdf} · "
            f"DOI unmatched: {self.n_doi_unmatched}"
        )


def _bib_dir_for_active_project() -> Path:
    """Return ``<project_root>/bibliography/``, creating it if needed."""
    project = get_active_project()
    target = project.root / "bibliography"
    target.mkdir(parents=True, exist_ok=True)
    return target


_NORM_TITLE_RE = re.compile(r"[^a-z0-9]")


def _norm_title(s: str) -> str:
    """Lowercase + strip everything except alphanumerics. Same scheme
    as ``rag/prompts.py:_norm_title`` so cross-component title
    comparisons stay consistent."""
    return _NORM_TITLE_RE.sub("", (s or "").lower())


def _extract_title_from_apa(apa_line: str) -> str:
    """Best-effort title extraction from an APA source string.

    APA shape: ``[N] Author (year). Title. Journal. doi: ...`` — title
    is the segment between ``(year). `` and the next sentence
    boundary. Falls back to the first segment after stripping the
    leading marker if the year-paren parse fails.
    """
    if not apa_line:
        return ""
    body = re.sub(r"^\s*\[\d+\]\s*", "", apa_line)
    m = re.search(r"\(\d{4}\)\.\s*([^.]+)\.", body)
    if m:
        return m.group(1).strip()
    # Fall back to the first sentence-shaped chunk.
    parts = body.split(".")
    return parts[0].strip() if parts else ""


def _safe_link(src: Path, dst: Path) -> bool:
    """Create or refresh a symlink ``dst → src``. Returns True on success.

    If ``dst`` already exists and points at a different source, refresh
    it (cheap; we want the link to track ``original_path`` if the file
    was relocated by an ingestion rerun).
    """
    try:
        src_real = src.resolve()
    except (OSError, RuntimeError):
        return False
    if not src_real.is_file():
        return False
    if dst.is_symlink() or dst.exists():
        try:
            current = dst.resolve()
            if current == src_real:
                return True
            dst.unlink()
        except OSError:
            return False
    try:
        dst.symlink_to(src_real)
        return True
    except OSError as exc:
        logger.warning("symlink %s -> %s failed: %s", dst, src_real, exc)
        return False


def archive_book_bibliography(book_id: str) -> ArchiveResult:
    """Populate ``<project>/bibliography/`` with one PDF symlink per
    cited paper in the book.

    Idempotent: re-running only links new entries; existing links are
    preserved. Removed-from-bib entries leave their orphaned symlinks
    in place — call ``prune_bibliography_archive`` separately if you
    want to scrub them.
    """
    bib_dir = _bib_dir_for_active_project()

    with get_session() as session:
        bib = BookBibliography.from_book(session, book_id)
        entries, _ = build_bibliography(session, bib.global_sources)

    result = ArchiveResult(
        bibliography_dir=bib_dir,
        n_entries=len(entries),
    )
    if not entries:
        return result

    with get_session() as session:
        for entry in entries:
            citekey = entry.citekey or "unknown"
            target = bib_dir / f"{citekey}.pdf"
            if target.is_symlink() or target.exists():
                # Re-validate the link in case the symlink target is dead.
                try:
                    if target.resolve().is_file():
                        result.n_already_present += 1
                        continue
                except (OSError, RuntimeError):
                    target.unlink(missing_ok=True)

            pdf_path: Path | None = None
            if entry.doi:
                row = session.execute(text(
                    "SELECT d.original_path FROM documents d "
                    "JOIN paper_metadata pm ON pm.document_id = d.id "
                    "WHERE pm.doi = :doi LIMIT 1"
                ), {"doi": entry.doi}).fetchone()
                if row and row[0]:
                    pdf_path = Path(row[0])
                else:
                    result.n_doi_unmatched += 1

            # Title fallback — APA strings sometimes drop the DOI URL
            # entirely (legacy retrievals, @misc fallbacks) but the
            # paper is in the corpus under a real document. Match by
            # normalised title; high-precision because article titles
            # are nearly unique post-normalisation.
            if pdf_path is None or not pdf_path.is_file():
                tnorm = _norm_title(_extract_title_from_apa(entry.apa))
                if tnorm and len(tnorm) >= 12:   # short titles too ambiguous
                    row = session.execute(text(
                        "SELECT d.original_path FROM documents d "
                        "JOIN paper_metadata pm ON pm.document_id = d.id "
                        "WHERE regexp_replace(lower(pm.title), '[^a-z0-9]', '', 'g') = :t "
                        "LIMIT 1"
                    ), {"t": tnorm}).fetchone()
                    if row and row[0]:
                        pdf_path = Path(row[0])

            if pdf_path is None or not pdf_path.is_file():
                result.n_no_local_pdf += 1
                preview = (entry.apa or "")[:120].replace("\n", " ")
                result.missing_titles.append(preview)
                continue

            if _safe_link(pdf_path, target):
                result.n_linked += 1
            else:
                result.n_no_local_pdf += 1

    return result


def prune_bibliography_archive(book_id: str) -> int:
    """Remove symlinks in ``<project>/bibliography/`` whose citekey is
    no longer in the book's current bibliography. Returns the number
    of removed links.

    Useful after edits / re-ingestion changed which papers are cited.
    Real PDFs (non-symlink files) are NOT touched.
    """
    bib_dir = _bib_dir_for_active_project()
    if not bib_dir.is_dir():
        return 0

    with get_session() as session:
        bib = BookBibliography.from_book(session, book_id)
        entries, _ = build_bibliography(session, bib.global_sources)
    keep = {f"{e.citekey}.pdf" for e in entries if e.citekey}

    removed = 0
    for child in bib_dir.iterdir():
        if not child.is_symlink():
            continue
        if child.suffix != ".pdf":
            continue
        if child.name not in keep:
            try:
                child.unlink()
                removed += 1
            except OSError:
                pass
    return removed
