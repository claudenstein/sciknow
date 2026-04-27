"""Internal Representation for the formatting pipeline.

A ``Document`` is the parsed, fully-resolved book/paper ready for
template rendering. It is plain dataclasses (no Pydantic) — this is
internal IR, validated by construction, not user input.

Hierarchy::

    Document
    ├── front_matter (title, authors, abstract, …)
    ├── chapters: list[Chapter]
    │   └── sections: list[Section]
    │       └── blocks: list[Block]   (Paragraph, Heading, ListBlock,
    │                                  TableBlock, EquationBlock,
    │                                  FigureBlock, CodeBlock, RuleBlock,
    │                                  QuoteBlock)
    └── bibliography: list[BibEntry]   (BibTeX-shaped, dedup'd globally)

Inline elements live inside ``Paragraph.inlines`` and represent text
runs, emphasis, citations, links, inline math.

The IR is *layout-agnostic*: the renderer decides whether to emit
``\\chapter`` vs ``\\section`` vs ``\\begin{abstract}`` based on the
project type and the section_type marker on each section.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Inlines ───────────────────────────────────────────────────────────


@dataclass
class TextRun:
    """Plain unformatted text. The renderer escapes LaTeX specials."""
    text: str
    kind: str = "text"


@dataclass
class Emph:
    inlines: list["Inline"]
    kind: str = "emph"


@dataclass
class Strong:
    inlines: list["Inline"]
    kind: str = "strong"


@dataclass
class CodeInline:
    code: str
    kind: str = "code"


@dataclass
class Link:
    href: str
    inlines: list["Inline"]
    kind: str = "link"


@dataclass
class Citation:
    """Inline citation. ``citekeys`` resolve to BibTeX entries.

    Multiple keys are emitted as ``\\cite{a,b,c}``. The original ``[N]``
    markers from the markdown are remapped to global numbers, then to
    citekeys, before this node is constructed.
    """
    citekeys: list[str]
    kind: str = "citation"


@dataclass
class MathInline:
    """Inline math, raw LaTeX (already a math expression, no $ wrap)."""
    tex: str
    kind: str = "math_inline"


Inline = TextRun | Emph | Strong | CodeInline | Link | Citation | MathInline


# ── Block-level ───────────────────────────────────────────────────────


@dataclass
class Paragraph:
    inlines: list[Inline]
    kind: str = "paragraph"


@dataclass
class Heading:
    """In-section heading.

    ``depth`` is *relative* to the enclosing section (1 = first heading
    inside a section → ``\\subsection``, 2 → ``\\subsubsection``, etc.).
    Top-level chapter/section headings are emitted by the renderer
    directly from the ``Chapter``/``Section`` records, not as ``Heading``
    blocks — Heading is only for in-body subdivisions.
    """
    inlines: list[Inline]
    depth: int = 1
    kind: str = "heading"


@dataclass
class ListBlock:
    ordered: bool
    items: list[list["Block"]]
    kind: str = "list"


@dataclass
class QuoteBlock:
    blocks: list["Block"]
    kind: str = "quote"


@dataclass
class CodeBlock:
    code: str
    language: Optional[str] = None
    kind: str = "code_block"


@dataclass
class EquationBlock:
    """Display equation. ``tex`` is a raw expression *without* the
    surrounding ``\\begin{equation}…\\end{equation}`` (the renderer adds it).
    Use ``aligned`` to indicate the body is multi-line and should go in
    ``align*`` instead of ``equation*``.
    """
    tex: str
    aligned: bool = False
    label: Optional[str] = None
    kind: str = "equation"


@dataclass
class FigureBlock:
    """Image figure. ``image_path`` must be absolute (renderer copies
    it into the compile dir under a stable filename).

    ``width`` is a LaTeX length expression (e.g. ``0.8\\textwidth``).
    """
    image_path: Path
    caption: str = ""
    label: Optional[str] = None
    width: str = "0.8\\textwidth"
    kind: str = "figure"


@dataclass
class TableBlock:
    """Table block.

    ``latex_body`` is the pre-converted LaTeX inside a ``tabular`` (i.e.
    only the rows/cells, no ``\\begin{table}`` wrapper — renderer adds
    that). HTML tables from MinerU are converted via pandoc fragment
    shellout in ``markdown_to_ir.py``.
    """
    latex_body: str
    column_spec: str = ""   # e.g. "lcr"; if empty, renderer infers from latex_body
    caption: str = ""
    label: Optional[str] = None
    kind: str = "table"


@dataclass
class RuleBlock:
    kind: str = "rule"


Block = (
    Paragraph
    | Heading
    | ListBlock
    | QuoteBlock
    | CodeBlock
    | EquationBlock
    | FigureBlock
    | TableBlock
    | RuleBlock
)


# ── Document structure ────────────────────────────────────────────────


@dataclass
class Section:
    """One drafted section within a chapter (or paper)."""
    title: str
    section_type: Optional[str] = None   # canonical key (e.g. "abstract")
    blocks: list[Block] = field(default_factory=list)
    word_count: int = 0


@dataclass
class Chapter:
    number: int
    title: str
    description: Optional[str] = None
    sections: list[Section] = field(default_factory=list)


@dataclass
class BibEntry:
    """A BibTeX-shaped entry. ``bibtex`` is the full ``@article{…}``
    string. ``citekey`` is the key inside the entry (used by ``\\cite{}``).
    """
    citekey: str
    bibtex: str
    apa: str = ""        # for fallback rendering when biblatex is disabled
    doi: Optional[str] = None


@dataclass
class FrontMatter:
    title: str
    subtitle: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    date: Optional[str] = None
    abstract: Optional[str] = None
    dedication: Optional[str] = None
    acknowledgements: Optional[str] = None
    keywords: list[str] = field(default_factory=list)


@dataclass
class Document:
    """The fully-resolved book/paper, ready for template rendering."""
    front_matter: FrontMatter
    project_type: str                    # slug from sciknow.core.project_type
    is_flat: bool                        # one-chapter paper layout?
    chapters: list[Chapter] = field(default_factory=list)
    bibliography: list[BibEntry] = field(default_factory=list)
    # Path-relative copies of figures the renderer should drop into the
    # compile dir (filename → absolute source path). Filled by the
    # markdown→IR pass when it visits FigureBlock nodes.
    figure_assets: dict[str, Path] = field(default_factory=dict)
