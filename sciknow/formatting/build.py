"""End-to-end build orchestrator.

``build_book_pdf(book_id, opts)`` is the only entry point. It:

1. Loads the book metadata, chapters, drafts from Postgres.
2. Resolves the global bibliography (``BookBibliography.from_book``) and
   builds BibTeX entries with stable cite keys.
3. Walks each draft's markdown through the citation remapper, parses
   to IR, and assembles the ``Document``.
4. Renders LaTeX via ``latex_renderer.render_document``.
5. Writes a temporary working directory containing ``main.tex``,
   ``refs.bib``, and any referenced figures.
6. Compiles via ``compile_tex``.
7. Returns ``(pdf_bytes, log, tex_source)``.

A second entry point ``build_book_tex_bundle`` packages the same
intermediate files as a zip so the user can take the .tex offline.
"""
from __future__ import annotations

import io
import json
import logging
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from sqlalchemy import text

from sciknow.config import settings
from sciknow.core.bibliography import BookBibliography
from sciknow.core.project_type import get_project_type
from sciknow.formatting.bibtex import build_bibliography, render_bibtex_file
from sciknow.formatting.compile import LatexCompileError, compile_tex
from sciknow.formatting.ir import (
    Chapter,
    Document,
    FrontMatter,
    Section,
)
from sciknow.formatting.latex_renderer import render_document
from sciknow.formatting.markdown_to_ir import (
    cite_map_from_global_sources,
    markdown_to_blocks,
)
from sciknow.formatting.options import ExportOptions
from sciknow.storage.db import get_session

log = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────


def _load_book_rows(book_id: str):
    with get_session() as sess:
        book = sess.execute(text("""
            SELECT id::text, title, description, plan, custom_metadata, book_type
            FROM books WHERE id::text = :bid
        """), {"bid": str(book_id)}).fetchone()
        if not book:
            raise ValueError(f"book not found: {book_id}")

        chapters = sess.execute(text("""
            SELECT id::text, number, title, description, sections
            FROM book_chapters WHERE book_id = :bid ORDER BY number
        """), {"bid": str(book_id)}).fetchall()

        # Same active-version logic as the reader: prefer is_active=true,
        # then non-empty content, then highest version.
        drafts = sess.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.content,
                   d.word_count, d.sources, d.version, d.chapter_id::text,
                   d.custom_metadata
            FROM drafts d
            WHERE d.book_id = :bid
            ORDER BY d.chapter_id, d.section_type,
                     CASE WHEN COALESCE((d.custom_metadata->>'is_active')::boolean, FALSE)
                          THEN 0 ELSE 1 END,
                     CASE WHEN d.content IS NULL OR LENGTH(d.content) < 50
                          THEN 1 ELSE 0 END,
                     d.version DESC
        """), {"bid": str(book_id)}).fetchall()

    return book, chapters, drafts


def _collapse_drafts(drafts) -> dict[tuple[str, str], tuple]:
    """Pick one draft per (chapter_id, section_type) using the same
    ordering the SELECT just produced (first-seen wins)."""
    out: dict[tuple[str, str], tuple] = {}
    for d in drafts:
        key = (d[7], d[2] or "")
        if key not in out:
            out[key] = d
    return out


def _ordered_chapter_drafts(
    chapter_id: str, sections_meta, drafts_by_key: dict[tuple[str, str], tuple]
) -> list[tuple]:
    """Return drafts for a chapter in the order declared in chapter.sections."""
    if not sections_meta:
        return [d for k, d in drafts_by_key.items() if k[0] == chapter_id]
    out = []
    for sec in sections_meta:
        slug = sec.get("slug", "") if isinstance(sec, dict) else ""
        d = drafts_by_key.get((chapter_id, slug))
        if d:
            out.append(d)
    return out


# ── IR assembly ───────────────────────────────────────────────────────


def _strip_md(text_content: str) -> str:
    """Best-effort markdown → plain text for the abstract override path."""
    if not text_content:
        return ""
    s = text_content
    s = re.sub(r"```.*?```", "", s, flags=re.DOTALL)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"#+\s+", "", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    s = re.sub(r"\[(\d+)\]", "", s)
    return s.strip()


def _build_document(
    book_id: str,
    opts: ExportOptions,
) -> Document:
    book_row, chapter_rows, draft_rows = _load_book_rows(book_id)

    book_id_str, title, description, plan, custom_metadata, book_type = book_row
    pt = get_project_type(book_type)

    # 1. Bibliography (uses its own session; safe to call here)
    with get_session() as sess:
        bib = BookBibliography.from_book(sess, book_id_str)

    # 2. BibTeX entries + cite map
    with get_session() as sess:
        entries, citekeys_by_key = build_bibliography(sess, bib.global_sources)
    cite_map_global = cite_map_from_global_sources(bib.global_sources, citekeys_by_key)

    # 3. Walk drafts, collapse to active versions, parse markdown
    drafts_by_key = _collapse_drafts(draft_rows)

    figure_assets: dict[str, Path] = {}
    image_root = Path(settings.data_dir)

    chapters: list[Chapter] = []
    for ch_row in chapter_rows:
        ch_id, ch_num, ch_title, ch_desc, sections_meta = ch_row
        if isinstance(sections_meta, str):
            try:
                sections_meta = json.loads(sections_meta)
            except Exception:
                sections_meta = None
        ch_drafts = _ordered_chapter_drafts(ch_id, sections_meta, drafts_by_key)

        sections: list[Section] = []
        for d in ch_drafts:
            d_id, d_title, d_section, d_content, d_words, _, _, _, _ = d
            if (d_section or "") == "argument_map":
                continue
            # Remap [N] markers to global numbers
            remapped = bib.remap_content(d_id, d_content or "")
            blocks = markdown_to_blocks(
                remapped,
                cite_map_global,
                figure_assets=figure_assets,
                image_root=image_root,
            )
            # Section title: section_type from project_type if available,
            # falling back to the draft's own title.
            display_title = d_title or ""
            if not display_title and d_section:
                from sciknow.core.project_type import target_words_for  # noqa: F401
                for s in pt.default_sections:
                    if s.key == d_section:
                        display_title = s.title
                        break
                if not display_title:
                    display_title = d_section.replace("_", " ").title()
            sections.append(Section(
                title=display_title or "Section",
                section_type=d_section,
                blocks=blocks,
                word_count=d_words or 0,
            ))

        chapters.append(Chapter(
            number=ch_num or 0,
            title=ch_title or f"Chapter {ch_num}",
            description=ch_desc,
            sections=sections,
        ))

    # 4. Front matter
    md = custom_metadata if isinstance(custom_metadata, dict) else {}
    if isinstance(custom_metadata, str) and custom_metadata:
        try:
            md = json.loads(custom_metadata)
        except Exception:
            md = {}
    authors = []
    if opts.author_override:
        authors = [opts.author_override]
    elif md.get("authors"):
        authors = md["authors"] if isinstance(md["authors"], list) else [str(md["authors"])]
    else:
        authors = ["sciknow"]
    affiliations = [opts.affiliation] if opts.affiliation else (
        md.get("affiliations") if isinstance(md.get("affiliations"), list) else []
    )

    abstract_text: Optional[str] = opts.abstract_override
    if not abstract_text and pt.is_flat:
        # Pull abstract section's text (or auto-generated abstract from
        # the book) for paper-style docs.
        if chapters:
            for s in chapters[0].sections:
                if (s.section_type or "").lower() == "abstract":
                    # Concatenate paragraph text from blocks
                    from sciknow.formatting.latex_renderer import inlines_to_latex  # noqa
                    parts = []
                    for b in s.blocks:
                        if hasattr(b, "inlines"):
                            parts.append(" ".join(
                                getattr(i, "text", "") for i in b.inlines if hasattr(i, "text")
                            ))
                    abstract_text = "\n\n".join(p for p in parts if p.strip())
                    break

    front = FrontMatter(
        title=title or "Untitled",
        subtitle=md.get("subtitle"),
        authors=authors,
        affiliations=affiliations or [],
        date=md.get("date"),
        abstract=abstract_text,
        dedication=opts.dedication,
        acknowledgements=opts.acknowledgements,
        keywords=md.get("keywords") if isinstance(md.get("keywords"), list) else [],
    )

    # Suppress the bibliography if no draft actually cites anything.
    # natbib/biblatex both emit broken .bbl files when given a .bib +
    # \bibliography{} but zero \cite commands — the resulting empty
    # thebibliography environment causes IEEEtran and similar bst
    # styles to fail at compile time.
    has_citations = _document_has_citations(chapters)
    if not has_citations:
        entries = []

    return Document(
        front_matter=front,
        project_type=pt.slug,
        is_flat=pt.is_flat,
        chapters=chapters,
        bibliography=entries,
        figure_assets=figure_assets,
    )


def _document_has_citations(chapters: list[Chapter]) -> bool:
    """True iff any rendered block contains an inline Citation node."""
    from sciknow.formatting.ir import Citation, ListBlock, Paragraph, QuoteBlock

    def _scan(blocks):
        for b in blocks:
            if isinstance(b, Paragraph):
                for inl in b.inlines:
                    if isinstance(inl, Citation):
                        return True
            elif isinstance(b, ListBlock):
                for item in b.items:
                    if _scan(item):
                        return True
            elif isinstance(b, QuoteBlock):
                if _scan(b.blocks):
                    return True
        return False

    for ch in chapters:
        for s in ch.sections:
            if _scan(s.blocks):
                return True
    return False


# ── Public entry points ───────────────────────────────────────────────


def build_book_pdf(
    book_id: str,
    opts: Optional[ExportOptions] = None,
) -> tuple[bytes, str, str]:
    """Build the book PDF end-to-end.

    Returns ``(pdf_bytes, log_text, tex_source)``. Raises
    ``LatexCompileError`` on compile failure with the log attached.
    """
    opts = opts or ExportOptions()
    doc = _build_document(book_id, opts)
    tex_source, spec = render_document(doc, opts)

    with tempfile.TemporaryDirectory(prefix="sciknow_latex_") as tmp:
        workdir = Path(tmp)
        # Drop figure assets
        for fname, src in doc.figure_assets.items():
            try:
                shutil.copy2(src, workdir / fname)
            except Exception as e:
                log.warning("failed to copy figure %s: %s", src, e)
        # Drop .bib file
        if doc.bibliography:
            (workdir / "refs.bib").write_text(
                render_bibtex_file(doc.bibliography), encoding="utf-8"
            )
        # Drop vendored class files (templates/<slug>/*.cls etc.)
        from sciknow.formatting.latex_renderer import TEMPLATES_DIR
        template_dir = TEMPLATES_DIR / spec.slug
        for ext in (".cls", ".sty", ".def", ".clo"):
            for path in template_dir.glob(f"*{ext}"):
                shutil.copy2(path, workdir / path.name)
        # Drop common shared style files
        shared_dir = TEMPLATES_DIR / "_shared"
        for ext in (".sty", ".cls", ".tex"):
            for path in shared_dir.glob(f"*{ext}"):
                shutil.copy2(path, workdir / path.name)
        # Also drop any .tex inputs that live alongside the template
        # (preamble.tex etc.) — main.tex.j2 \input's them by name.
        for ext in (".tex",):
            for path in template_dir.glob(f"*{ext}"):
                if path.name.endswith(".j2"):
                    continue
                shutil.copy2(path, workdir / path.name)

        pdf_bytes, log_text = compile_tex(
            tex_source, workdir, engine=spec.engine,
            bib_backend=spec.bib_backend,
        )
    return pdf_bytes, log_text, tex_source


def build_wiki_pdf(
    page_slug: Optional[str] = None,
    opts: Optional[ExportOptions] = None,
) -> tuple[bytes, str, str]:
    """Build a PDF of the wiki.

    If ``page_slug`` is set, render just that page as an article-class
    PDF. Otherwise compile every wiki page (concepts + papers +
    synthesis) into a single kaobook with one chapter per page-type
    bucket.

    Returns ``(pdf_bytes, log_text, tex_source)``.
    """
    opts = opts or ExportOptions()
    doc = _build_wiki_document(page_slug, opts)
    tex_source, spec = render_document(doc, opts)

    with tempfile.TemporaryDirectory(prefix="sciknow_wiki_latex_") as tmp:
        workdir = Path(tmp)
        for fname, src in doc.figure_assets.items():
            try:
                shutil.copy2(src, workdir / fname)
            except Exception as e:
                log.warning("failed to copy figure %s: %s", src, e)
        if doc.bibliography:
            (workdir / "refs.bib").write_text(
                render_bibtex_file(doc.bibliography), encoding="utf-8"
            )
        from sciknow.formatting.latex_renderer import TEMPLATES_DIR
        template_dir = TEMPLATES_DIR / spec.slug
        for ext in (".cls", ".sty", ".def", ".clo"):
            for path in template_dir.glob(f"*{ext}"):
                shutil.copy2(path, workdir / path.name)
        for ext in (".tex",):
            for path in template_dir.glob(f"*{ext}"):
                if path.name.endswith(".j2"):
                    continue
                shutil.copy2(path, workdir / path.name)
        shared_dir = TEMPLATES_DIR / "_shared"
        for ext in (".sty", ".cls", ".tex"):
            for path in shared_dir.glob(f"*{ext}"):
                shutil.copy2(path, workdir / path.name)

        pdf_bytes, log_text = compile_tex(
            tex_source, workdir, engine=spec.engine, bib_backend=spec.bib_backend,
        )
    return pdf_bytes, log_text, tex_source


def _build_wiki_document(
    page_slug: Optional[str],
    opts: ExportOptions,
) -> "Document":
    """Assemble an IR Document from the on-disk wiki tree.

    Wiki pages live under ``<data_dir>/wiki/{concepts,papers,synthesis}/<slug>.md``
    with a small DB index in the ``wiki_pages`` table.
    """
    from sciknow.formatting.markdown_to_ir import markdown_to_blocks
    wiki_root = Path(settings.data_dir) / "wiki"

    # Pull metadata from DB so we have human titles + page_type.
    with get_session() as sess:
        rows = sess.execute(text("""
            SELECT slug, title, page_type, source_doc_ids
            FROM wiki_pages ORDER BY page_type, title
        """)).fetchall()

    by_slug: dict[str, tuple] = {r[0]: r for r in rows}

    # Wiki is its own kind of project — we tag it as scientific_book
    # so it gets a book-style template by default. Users can override
    # with --template article for a single-page paper layout.
    project_type = "scientific_book"

    figure_assets: dict[str, Path] = {}
    image_root = wiki_root

    def _read_page_md(slug: str, page_type: str) -> str:
        # The on-disk filename has a hash prefix for uniqueness; we
        # match by slug-stem suffix.
        bucket = wiki_root / {
            "concept": "concepts", "paper_summary": "papers", "synthesis": "synthesis",
        }.get(page_type, "concepts")
        if not bucket.exists():
            return ""
        # Try exact match first, then any file ending in <slug>.md
        candidates = sorted(bucket.glob(f"*{slug}.md"))
        if not candidates:
            return ""
        try:
            return candidates[0].read_text(encoding="utf-8")
        except Exception:
            return ""

    if page_slug:
        row = by_slug.get(page_slug)
        if not row:
            raise ValueError(f"wiki page not found: {page_slug}")
        slug, title, page_type, _ = row
        md = _read_page_md(slug, page_type)
        blocks = markdown_to_blocks(md, {}, figure_assets=figure_assets, image_root=image_root)
        chapters = [Chapter(
            number=1, title=title or slug,
            sections=[Section(title=title or slug, blocks=blocks)],
        )]
        # Single page → flat layout (renders well in article + kaobook).
        is_flat = True
        front = FrontMatter(
            title=title or slug,
            authors=["sciknow wiki"],
            date=None,
        )
    else:
        # Bucket pages by type into chapters.
        bucket_titles = {
            "concept": "Concepts",
            "paper_summary": "Paper summaries",
            "synthesis": "Syntheses",
        }
        chapters_by_type: dict[str, list[Section]] = {k: [] for k in bucket_titles}
        for slug, title, page_type, _ in rows:
            md = _read_page_md(slug, page_type)
            if not md.strip():
                continue
            blocks = markdown_to_blocks(md, {}, figure_assets=figure_assets, image_root=image_root)
            sec = Section(title=title or slug, section_type=page_type, blocks=blocks)
            chapters_by_type.setdefault(page_type, []).append(sec)

        chapters = []
        n = 1
        for ptype, ch_title in bucket_titles.items():
            secs = chapters_by_type.get(ptype) or []
            if not secs:
                continue
            chapters.append(Chapter(number=n, title=ch_title, sections=secs))
            n += 1
        is_flat = False
        front = FrontMatter(
            title="Wiki",
            authors=["sciknow"],
            date=None,
        )

    # Wikis don't carry [N] citations — source_doc_ids is the
    # provenance, but we don't currently emit a bibliography here.
    # F5+ could resolve source_doc_ids to BibTeX entries; left as a
    # follow-up because the wiki UI surfaces sources via wiki-links
    # rather than [N] markers.
    return Document(
        front_matter=front,
        project_type=project_type,
        is_flat=is_flat,
        chapters=chapters,
        bibliography=[],
        figure_assets=figure_assets,
    )


def build_book_tex_bundle(
    book_id: str,
    opts: Optional[ExportOptions] = None,
) -> tuple[bytes, str]:
    """Build a .zip containing main.tex, refs.bib, all figures, and any
    vendored .cls/.sty files needed to compile offline.

    Returns ``(zip_bytes, tex_source)``.
    """
    opts = opts or ExportOptions()
    doc = _build_document(book_id, opts)
    tex_source, spec = render_document(doc, opts)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.tex", tex_source)
        if doc.bibliography:
            zf.writestr("refs.bib", render_bibtex_file(doc.bibliography))
        for fname, src in doc.figure_assets.items():
            try:
                zf.write(src, fname)
            except Exception as e:
                log.warning("failed to bundle figure %s: %s", src, e)
        from sciknow.formatting.latex_renderer import TEMPLATES_DIR
        template_dir = TEMPLATES_DIR / spec.slug
        for ext in (".cls", ".sty", ".def", ".clo"):
            for path in template_dir.glob(f"*{ext}"):
                zf.write(path, path.name)
        shared_dir = TEMPLATES_DIR / "_shared"
        for ext in (".sty", ".cls", ".tex"):
            for path in shared_dir.glob(f"*{ext}"):
                zf.write(path, path.name)
        for ext in (".tex",):
            for path in template_dir.glob(f"*{ext}"):
                if path.name.endswith(".j2"):
                    continue
                zf.write(path, path.name)
        # Throw in a tiny README so the user knows how to compile
        zf.writestr(
            "README.txt",
            f"Compile with:\n  latexmk -{spec.engine} main.tex\n\n"
            f"Generated by sciknow formatting module.\n"
            f"Template: {spec.slug} ({spec.display_name})\n"
            f"Document class: {spec.document_class}\n",
        )
    return buf.getvalue(), tex_source
