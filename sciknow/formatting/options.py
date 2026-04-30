"""Export options, template registry, font registry.

The single source of truth for *what users can pick*. The Export tab
in the web GUI and the ``sciknow book export --format pdf-pro`` CLI
both read from here so the surfaces stay in sync.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


# ── Template registry ─────────────────────────────────────────────────
#
# Each project_type slug maps to one or more templates. The first entry
# is the default. ``available`` is checked at runtime against the
# installed TeX Live or vendored class files; if the class is missing
# we drop the template from the picker and warn once.

@dataclass(frozen=True)
class TemplateSpec:
    slug: str                                # machine id used on disk
    display_name: str                        # human label
    description: str
    document_class: str                      # \documentclass{...}
    requires_tex_class: str                  # kpsewhich target; "" = vendored
    engine: Literal["lualatex", "pdflatex", "xelatex"] = "lualatex"
    bib_backend: Literal["biber", "bibtex"] = "biber"
    # Which citation/bibliography toolchain the template wires up. Most
    # modern templates (kaobook, classicthesis, memoir, scrbook,
    # IEEEtran via biblatex-ieee, springer-nature) use biblatex+biber.
    # elsarticle is hardwired to natbib internally and conflicts with
    # biblatex when both load, so it stays on natbib+bibtex.
    bib_system: Literal["biblatex", "natbib"] = "biblatex"


# Templates per project type. Order = picker order. Index 0 = default.
TEMPLATES: dict[str, list[TemplateSpec]] = {
    "scientific_paper": [
        TemplateSpec(
            slug="elsarticle",
            display_name="Elsevier (elsarticle)",
            description="Single/double column journal article — Elsevier style.",
            document_class="elsarticle",
            requires_tex_class="elsarticle.cls",
            bib_backend="bibtex",
            bib_system="natbib",
        ),
        TemplateSpec(
            slug="ieeetran",
            display_name="IEEE (IEEEtran)",
            description="Two-column IEEE journal/conference layout.",
            document_class="IEEEtran",
            requires_tex_class="IEEEtran.cls",
            bib_backend="bibtex",
            bib_system="natbib",
        ),
        TemplateSpec(
            slug="revtex",
            display_name="REVTeX 4.2 (APS/AIP)",
            description="Physics-style: structured author lists, equation tagging.",
            document_class="revtex4-2",
            requires_tex_class="revtex4-2.cls",
            bib_backend="bibtex",
            bib_system="natbib",
        ),
        TemplateSpec(
            slug="article",
            display_name="Generic article",
            description="Plain article class — broadest compatibility.",
            document_class="article",
            requires_tex_class="article.cls",
        ),
    ],
    "scientific_book": [
        TemplateSpec(
            slug="kaobook",
            display_name="kaobook (modern monograph)",
            description="Wide margins, side-margin notes, modern typography.",
            document_class="kaobook",
            requires_tex_class="",   # vendored under templates/kaobook/
        ),
        TemplateSpec(
            slug="memoir",
            display_name="memoir",
            description="Highly customisable book class; balanced defaults.",
            document_class="memoir",
            requires_tex_class="memoir.cls",
        ),
        TemplateSpec(
            slug="scrbook",
            display_name="KOMA-Script (scrbook)",
            description="European-style book class, clean defaults.",
            document_class="scrbook",
            requires_tex_class="scrbook.cls",
        ),
    ],
    "popular_science": [
        TemplateSpec(
            slug="memoir-narrative",
            display_name="Memoir (narrative)",
            description="Veelo chapter style, drop caps, accent colour — "
                        "Legrand-Orange-style narrative book look.",
            document_class="memoir",
            requires_tex_class="memoir.cls",
        ),
        TemplateSpec(
            slug="tufte-book",
            display_name="Tufte-style book",
            description="Side-margin notes, Bringhurst-inspired typography.",
            document_class="tufte-book",
            requires_tex_class="tufte-book.cls",
            bib_backend="bibtex",
            bib_system="natbib",
        ),
        TemplateSpec(
            slug="kaobook",
            display_name="kaobook",
            description="Modern monograph with margin notes.",
            document_class="kaobook",
            requires_tex_class="",
        ),
    ],
    "instructional_textbook": [
        TemplateSpec(
            slug="kaobook",
            display_name="kaobook (textbook)",
            description="Margin notes great for definitions and side "
                        "examples; example/exercise boxes via mdframed.",
            document_class="kaobook",
            requires_tex_class="",
        ),
        TemplateSpec(
            slug="memoir-textbook",
            display_name="Memoir (textbook)",
            description="Numbered exercises, worked-example boxes, "
                        "checkpoint summaries.",
            document_class="memoir",
            requires_tex_class="memoir.cls",
        ),
    ],
    "academic_monograph": [
        TemplateSpec(
            slug="classicthesis",
            display_name="ClassicThesis (André Miede)",
            description="Bringhurst typography, dense academic prose.",
            document_class="scrbook",   # classicthesis is loaded as a style
            requires_tex_class="classicthesis.sty",
        ),
        TemplateSpec(
            slug="kaobook",
            display_name="kaobook",
            description="Modern monograph with margin notes.",
            document_class="kaobook",
            requires_tex_class="",
        ),
        TemplateSpec(
            slug="memoir",
            display_name="memoir",
            description="Customisable; good fallback if classicthesis is missing.",
            document_class="memoir",
            requires_tex_class="memoir.cls",
        ),
    ],
    "review_article": [
        TemplateSpec(
            slug="elsarticle-review",
            display_name="Elsevier (review mode)",
            description="Single-column elsarticle in review layout.",
            document_class="elsarticle",
            requires_tex_class="elsarticle.cls",
            bib_backend="bibtex",
            bib_system="natbib",
        ),
        TemplateSpec(
            slug="article",
            display_name="Generic article",
            description="Plain single-column review.",
            document_class="article",
            requires_tex_class="article.cls",
        ),
    ],
}


def list_templates_for_type(project_type: str) -> list[TemplateSpec]:
    """Return templates available for a project type, default first.

    Falls back to ``scientific_book`` templates if the slug is unknown.
    """
    return TEMPLATES.get(project_type) or TEMPLATES["scientific_book"]


def default_template_for(project_type: str) -> TemplateSpec:
    return list_templates_for_type(project_type)[0]


def find_template(project_type: str, template_slug: Optional[str]) -> TemplateSpec:
    """Resolve a template slug, fallback to default for the project type.

    Slug resolution is two-stage: first within the project type's own
    template list (so picker order is preserved), then a global search
    across every type (so users can render e.g. a book as a paper if
    they want). Defaults to the first template of the project type.
    """
    candidates = list_templates_for_type(project_type)
    if template_slug:
        for t in candidates:
            if t.slug == template_slug:
                return t
        for type_candidates in TEMPLATES.values():
            for t in type_candidates:
                if t.slug == template_slug:
                    return t
    return candidates[0]


# ── Font registry ─────────────────────────────────────────────────────
#
# All entries must be available in TeX Live's `texlive-fontsextra`
# (which is part of texlive-full). Renderer maps these to the right
# package incantation.

FONT_FAMILIES = {
    "lmodern":     ("Latin Modern (default)",   "\\usepackage{lmodern}"),
    "libertinus":  ("Libertinus",               "\\usepackage{libertinus}"),
    "termes":      ("TeX Gyre Termes (Times)",  "\\usepackage{tgtermes}\\usepackage{tgheros}"),
    "ebgaramond":  ("EB Garamond",              "\\usepackage{ebgaramond}\\usepackage[scaled=0.95]{cabin}"),
    "palatino":    ("TeX Gyre Pagella (Palatino)", "\\usepackage{tgpagella}"),
    "sourceserif": ("Source Serif Pro",         "\\usepackage{sourceserifpro}\\usepackage{sourcesanspro}"),
}

BIB_STYLES = {
    "numeric":     ("Numeric (default)",          "numeric-comp"),
    "authoryear":  ("Author–year",                "authoryear-comp"),
    "ieee":        ("IEEE",                       "ieee"),
    "apa":         ("APA",                        "apa"),
    "chicago":     ("Chicago author-date",        "chicago-authordate"),
}


# ── Export options struct ─────────────────────────────────────────────


@dataclass
class ExportOptions:
    """User-controllable knobs for one export run.

    All have sensible defaults, so callers can do ``ExportOptions()`` and
    rely on the project type to pick everything for them.
    """
    template_slug: Optional[str] = None       # None → default for project type
    font_family: str = "lmodern"
    font_size_pt: int = 11                    # 10/11/12
    paper: Literal["a4paper", "letterpaper"] = "a4paper"
    two_column: bool = False                  # papers only

    # Bibliography
    bib_style: str = "numeric"
    sort_bib: Literal["citeorder", "alpha"] = "citeorder"
    hyperlink_dois: bool = True
    # Where to place the bibliography in book-style templates (kaobook,
    # classicthesis, scrbook, memoir*). Paper-style templates ignore
    # this and always print at end-of-document.
    #   "book"    — single bibliography at end of book (default)
    #   "chapter" — per-chapter bibliography via biblatex refsection
    bibliography_placement: Literal["book", "chapter"] = "book"

    # Front matter / structure
    cover_page: bool = True
    table_of_contents: bool = True
    list_of_figures: bool = False
    list_of_tables: bool = False
    abstract_override: Optional[str] = None
    author_override: Optional[str] = None
    affiliation: Optional[str] = None
    dedication: Optional[str] = None
    acknowledgements: Optional[str] = None
    toc_depth: int = 2                        # 1=chapter, 2=section, 3=subsection

    # Output
    include_bibliography_chapter: bool = True

    def merged_with(self, **overrides) -> "ExportOptions":
        return ExportOptions(**{**self.__dict__, **overrides})
