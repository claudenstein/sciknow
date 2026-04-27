"""IR → LaTeX rendering via Jinja2.

Two layers:

1. **Block/inline → LaTeX strings.** ``inlines_to_latex(...)`` and
   ``blocks_to_latex(...)`` convert the IR primitives to escaped LaTeX
   text. These are pure functions; they don't read templates.

2. **Document → ``main.tex`` string.** ``render_document(doc, opts)``
   loads the right template package from ``templates/<slug>/`` and
   passes the rendered chapter/section bodies as Jinja variables.

LaTeX-safe Jinja delimiters (the standard pattern used by every
real-world Latex+Jinja project):

    block:    ((* ... *))
    variable: ((( ... )))
    comment:  ((= ... =))

This avoids collisions with ``{`` / ``%`` / ``#`` in LaTeX source.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import jinja2

from sciknow.formatting.ir import (
    BibEntry,
    Block,
    Chapter,
    CodeBlock,
    Citation,
    CodeInline,
    Document,
    Emph,
    EquationBlock,
    FigureBlock,
    Heading,
    Inline,
    Link,
    ListBlock,
    MathInline,
    Paragraph,
    QuoteBlock,
    RuleBlock,
    Section,
    Strong,
    TableBlock,
    TextRun,
)
from sciknow.formatting.options import (
    BIB_STYLES,
    ExportOptions,
    FONT_FAMILIES,
    TemplateSpec,
    find_template,
)

log = logging.getLogger(__name__)


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


# ── LaTeX escaping ────────────────────────────────────────────────────


_TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{":  r"\{",
    "}":  r"\}",
    "$":  r"\$",
    "&":  r"\&",
    "%":  r"\%",
    "#":  r"\#",
    "_":  r"\_",
    "^":  r"\^{}",
    "~":  r"\~{}",
}


def tex_escape(s: str) -> str:
    """Escape a plain string for LaTeX text mode.

    Order matters: backslash first so we don't double-escape the
    backslashes we just inserted.
    """
    if s is None:
        return ""
    out = []
    for ch in s:
        out.append(_TEX_ESCAPES.get(ch, ch))
    return "".join(out)


# ── Inline rendering ──────────────────────────────────────────────────


def _inline_to_latex(inline: Inline) -> str:
    if isinstance(inline, TextRun):
        return tex_escape(inline.text)
    if isinstance(inline, Emph):
        return r"\emph{" + inlines_to_latex(inline.inlines) + "}"
    if isinstance(inline, Strong):
        return r"\textbf{" + inlines_to_latex(inline.inlines) + "}"
    if isinstance(inline, CodeInline):
        # \texttt with a verbatim-ish escape; for code blocks we use listings,
        # but inline fragments use \texttt so they stay in flow.
        return r"\texttt{" + tex_escape(inline.code) + "}"
    if isinstance(inline, Link):
        url = inline.href
        # Escape only what \url cannot tolerate: %, #, and backslash.
        # \url is robust to most punctuation.
        text = inlines_to_latex(inline.inlines).strip()
        if not text or text == url:
            return r"\url{" + url.replace("%", r"\%").replace("#", r"\#") + "}"
        return r"\href{" + url.replace("%", r"\%").replace("#", r"\#") + "}{" + text + "}"
    if isinstance(inline, Citation):
        if not inline.citekeys:
            return ""
        return r"\cite{" + ",".join(inline.citekeys) + "}"
    if isinstance(inline, MathInline):
        return f"${inline.tex}$"
    return ""


def inlines_to_latex(inlines: Iterable[Inline]) -> str:
    return "".join(_inline_to_latex(i) for i in inlines)


# ── Block rendering ───────────────────────────────────────────────────


def _block_to_latex(block: Block, depth_offset: int) -> str:
    if isinstance(block, Paragraph):
        body = inlines_to_latex(block.inlines).strip()
        return body + "\n" if body else ""

    if isinstance(block, Heading):
        # Map relative depth → LaTeX section command. depth_offset is
        # added by the caller (e.g. inside a chapter, depth 1 → \subsection).
        d = max(1, block.depth + depth_offset)
        cmd = {
            1: r"\section",
            2: r"\subsection",
            3: r"\subsubsection",
            4: r"\paragraph",
        }.get(d, r"\subparagraph")
        return f"{cmd}{{{inlines_to_latex(block.inlines).strip()}}}\n"

    if isinstance(block, ListBlock):
        env = "enumerate" if block.ordered else "itemize"
        lines = [rf"\begin{{{env}}}"]
        for item in block.items:
            lines.append(r"  \item " + blocks_to_latex(item, depth_offset).strip())
        lines.append(rf"\end{{{env}}}")
        return "\n".join(lines) + "\n"

    if isinstance(block, QuoteBlock):
        body = blocks_to_latex(block.blocks, depth_offset).strip()
        return f"\\begin{{quote}}\n{body}\n\\end{{quote}}\n"

    if isinstance(block, CodeBlock):
        # listings package; renderer's preamble defines a 'sciknowcode'
        # style. Language is passed if recognised.
        lang = block.language or ""
        opts = f"[language={lang}]" if lang and re.match(r"^[A-Za-z0-9+\-_]+$", lang) else ""
        return f"\\begin{{lstlisting}}{opts}\n{block.code}\n\\end{{lstlisting}}\n"

    if isinstance(block, EquationBlock):
        env = "align*" if block.aligned else "equation*"
        if block.label:
            env = "align" if block.aligned else "equation"
            return f"\\begin{{{env}}}\n{block.tex}\n\\label{{{block.label}}}\n\\end{{{env}}}\n"
        return f"\\begin{{{env}}}\n{block.tex}\n\\end{{{env}}}\n"

    if isinstance(block, FigureBlock):
        cap = tex_escape(block.caption) if block.caption else ""
        label = f"\\label{{{block.label}}}\n" if block.label else ""
        # Use the filename relative to compile dir; build.py copies the
        # actual file there.
        fname = block.image_path.name
        return (
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            f"\\includegraphics[width={block.width}]{{{fname}}}\n"
            + (f"\\caption{{{cap}}}\n" if cap else "")
            + label
            + "\\end{figure}\n"
        )

    if isinstance(block, TableBlock):
        cap = tex_escape(block.caption) if block.caption else ""
        label = f"\\label{{{block.label}}}\n" if block.label else ""
        if block.column_spec == "__verbatim__":
            # Pandoc-converted: latex_body is a complete table environment
            return block.latex_body + ("\n" if not block.latex_body.endswith("\n") else "")
        spec = block.column_spec or "l"
        return (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\begin{tabular}{" + spec + "}\n"
            "\\toprule\n"
            f"{block.latex_body}\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            + (f"\\caption{{{cap}}}\n" if cap else "")
            + label
            + "\\end{table}\n"
        )

    if isinstance(block, RuleBlock):
        return "\\noindent\\rule{\\textwidth}{0.4pt}\n"

    return ""


def blocks_to_latex(blocks: Iterable[Block], depth_offset: int = 0) -> str:
    """Render a list of blocks. ``depth_offset`` shifts ``Heading`` depth
    so that an in-section heading rendered inside a ``\\section{}`` becomes
    ``\\subsection`` rather than ``\\section``.
    """
    return "\n".join(_block_to_latex(b, depth_offset) for b in blocks)


# ── Section/chapter rendering ─────────────────────────────────────────


def _section_to_latex(section: Section, *, depth_offset: int = 0) -> str:
    """Render a section. ``depth_offset=0`` puts headings at \\section level
    (papers); =1 puts them at \\subsection level (chapters in a book).
    """
    cmd = r"\section" if depth_offset == 0 else r"\subsection"
    body = blocks_to_latex(section.blocks, depth_offset=depth_offset)
    title = tex_escape(section.title)
    return f"{cmd}{{{title}}}\n\n{body}\n"


def _chapter_to_latex(chapter: Chapter) -> str:
    parts = [f"\\chapter{{{tex_escape(chapter.title)}}}\n"]
    if chapter.description:
        parts.append(f"\\begingroup\\itshape\n{tex_escape(chapter.description)}\n\\endgroup\n\n")
    for sec in chapter.sections:
        parts.append(_section_to_latex(sec, depth_offset=1))
    return "\n".join(parts)


# ── Top-level render ──────────────────────────────────────────────────


def _make_jinja_env(template_dir: Path) -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader([
            str(template_dir),
            str(TEMPLATES_DIR / "_shared"),
        ]),
        block_start_string="((*",
        block_end_string="*))",
        variable_start_string="(((",
        variable_end_string=")))",
        comment_start_string="((#",
        comment_end_string="#))",
        line_statement_prefix="%%",
        line_comment_prefix="%#",
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
        keep_trailing_newline=True,
    )


def _abstract_section(doc: Document) -> Section | None:
    """Find the abstract section for paper-style docs (is_flat)."""
    if not doc.is_flat or not doc.chapters:
        return None
    for s in doc.chapters[0].sections:
        if (s.section_type or "").lower() == "abstract":
            return s
    return None


def _non_abstract_sections(doc: Document) -> list[Section]:
    if not doc.is_flat or not doc.chapters:
        return []
    return [s for s in doc.chapters[0].sections if (s.section_type or "").lower() != "abstract"]


def render_document(doc: Document, opts: ExportOptions) -> tuple[str, TemplateSpec]:
    """Render the IR document to a complete ``main.tex`` string.

    Returns ``(tex_source, template_spec)`` so the build orchestrator
    knows which engine and class files to ensure are available.
    """
    spec = find_template(doc.project_type, opts.template_slug)
    template_dir = TEMPLATES_DIR / spec.slug
    if not template_dir.is_dir():
        raise FileNotFoundError(
            f"template package not found: {template_dir}. "
            f"Available: {[p.name for p in TEMPLATES_DIR.iterdir() if p.is_dir()]}"
        )

    env = _make_jinja_env(template_dir)
    env.filters["tex_escape"] = tex_escape
    env.filters["inlines_to_latex"] = inlines_to_latex
    env.filters["blocks_to_latex"] = blocks_to_latex
    env.filters["section_to_latex"] = _section_to_latex
    env.filters["chapter_to_latex"] = _chapter_to_latex

    template = env.get_template("main.tex.j2")

    font_pkg = FONT_FAMILIES.get(opts.font_family, FONT_FAMILIES["lmodern"])[1]
    bib_style = BIB_STYLES.get(opts.bib_style, BIB_STYLES["numeric"])[1]

    abstract_section = _abstract_section(doc)
    body_sections = _non_abstract_sections(doc)

    ctx = {
        "doc": doc,
        "front": doc.front_matter,
        "chapters": doc.chapters,
        "is_flat": doc.is_flat,
        "abstract_section": abstract_section,
        "body_sections": body_sections,
        "bib_entries": doc.bibliography,
        "options": opts,
        "font_package": font_pkg,
        "bib_style": bib_style,
        "spec": spec,
    }
    return template.render(**ctx), spec
