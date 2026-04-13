"""Project type registry (Phase 45).

sciknow's ``books`` table is the root record for any long-form writing
project. A project *type* decides the default shape of that project —
what sections, what lengths, what prompt conditioning the writer gets,
what export template the book export path should use.

This module is the single source of truth for per-type behaviour. It
is deliberately a pure-data registry rather than a class hierarchy:
every value is a dict/string, so new types can be added without
subclassing anything, and every caller that reads type-specific data
does so by looking up the slug, failing over to ``scientific_book``
(the legacy default) if the type is missing.

Shipped types (Phase 45):

- ``scientific_book`` — hierarchical: book → chapters → sections. The
  pre-Phase-45 behaviour; all existing corpora stay here.
- ``scientific_paper`` — flat IMRaD: one project = one paper, with
  canonical sections (Abstract, Introduction, Methods, Results,
  Discussion, Conclusion). A paper is modelled as a single chapter
  with ``is_flat=True`` semantics on the UI side.

Planned types (next tick of the plan in docs/STRATEGY.md):

- ``scifi_novel``       — chapters + arcs, no citations, narrative prose.
- ``literature_review`` — multi-chapter like a book but with CARS-heavy
                          structure and different scoring weights.
- ``grant_proposal``    — short-form with fixed sections (Aims,
                          Significance, Approach, Rationale, Timeline).

Everything below is read-only at runtime; edit-and-redeploy semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


# A structural "section" template. For book types, sections live inside
# chapters and are replicated per-chapter. For paper-like types the
# list IS the whole project.
@dataclass(frozen=True)
class SectionTemplate:
    key: str                       # canonical section_type, matches chunker vocabulary
    title: str                     # display title in the UI
    target_words: int              # default per-section target
    required: bool = False         # if True, a `book gaps` call flags its absence


# Per-type defaults. Everything downstream reads from here.
@dataclass(frozen=True)
class ProjectType:
    slug: str                              # machine id — matches books.book_type
    display_name: str                      # human label
    description: str                       # one-line pitch
    is_flat: bool                          # True = single-document (no chapter hierarchy)
    default_sections: tuple[SectionTemplate, ...]
    default_chapter_count: int             # used by `book outline` when we don't have one yet
    default_target_chapter_words: int
    # Keys the writer prompt may condition on (see rag/prompts.py).
    # Kept small and declarative so prompts can branch without import
    # cycles with this module.
    writer_hints: tuple[str, ...] = ()
    # Extra type-specific behaviour flags readable by any caller.
    #   allow_citations:        insert [N] markers in the draft
    #   require_abstract:       enforce a <= 300-word abstract section
    #   cli_export_defaults:    format hints for `book export`
    flags: dict[str, Any] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────

SCIENTIFIC_BOOK = ProjectType(
    slug="scientific_book",
    display_name="Scientific Book",
    description="Hierarchical book with chapters. Chapters contain "
                "sections; each section is grounded in retrieval.",
    is_flat=False,
    # Default section template applied to each chapter. Individual
    # chapters may override via ``book_chapters.sections`` (Phase 11).
    default_sections=(
        SectionTemplate("overview",              "Overview",               1200),
        SectionTemplate("key_evidence",          "Key Evidence",           1600),
        SectionTemplate("current_understanding", "Current Understanding",  1600),
        SectionTemplate("open_questions",        "Open Questions",         1200),
        SectionTemplate("summary",               "Summary",                 800),
    ),
    default_chapter_count=8,
    default_target_chapter_words=6400,     # 1200+1600+1600+1200+800 ≈ 6400
    writer_hints=(
        "long_form",
        "chapter_level_continuity",
        "hedged_scientific_voice",
    ),
    flags={
        "allow_citations": True,
        "require_abstract": False,
        "cli_export_defaults": {"format": "markdown"},
    },
)


SCIENTIFIC_PAPER = ProjectType(
    slug="scientific_paper",
    display_name="Scientific Paper",
    description="Single-document IMRaD paper (one chapter, canonical "
                "sections). Tighter length targets + abstract enforcement.",
    is_flat=True,
    # Canonical IMRaD section set. A paper lives in a single chapter
    # in the DB so existing chapter/draft plumbing works unchanged.
    default_sections=(
        SectionTemplate("abstract",     "Abstract",      200, required=True),
        SectionTemplate("introduction", "Introduction",  800),
        SectionTemplate("methods",      "Methods",       900),
        SectionTemplate("results",      "Results",       900),
        SectionTemplate("discussion",   "Discussion",    900),
        SectionTemplate("conclusion",   "Conclusion",    300),
    ),
    default_chapter_count=1,               # one "chapter" = the whole paper
    default_target_chapter_words=4000,     # ~4k words is the typical paper body
    writer_hints=(
        "concise_scientific_voice",
        "imrad_structure",
        "inline_citations_required",
        "abstract_under_300_words",
    ),
    flags={
        "allow_citations": True,
        "require_abstract": True,
        "cli_export_defaults": {"format": "latex"},
    },
)


# Central registry. Extend here + add a default_sections tuple to ship
# a new type. No schema change required.
PROJECT_TYPES: dict[str, ProjectType] = {
    SCIENTIFIC_BOOK.slug:  SCIENTIFIC_BOOK,
    SCIENTIFIC_PAPER.slug: SCIENTIFIC_PAPER,
}

# The slug we store for pre-Phase-45 rows (server_default on the column).
DEFAULT_TYPE_SLUG = SCIENTIFIC_BOOK.slug


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────


def get_project_type(slug: str | None) -> ProjectType:
    """Return the ProjectType for ``slug``, fallback to default on miss.

    Use this everywhere — never call ``PROJECT_TYPES[...]`` directly,
    because that would crash on any row whose ``book_type`` is NULL or
    an unknown slug (e.g. from a future schema downgrade).
    """
    if slug and slug in PROJECT_TYPES:
        return PROJECT_TYPES[slug]
    return PROJECT_TYPES[DEFAULT_TYPE_SLUG]


def list_project_types() -> list[ProjectType]:
    """All registered types, in insertion order. Used by the CLI picker."""
    return list(PROJECT_TYPES.values())


def validate_type_slug(slug: str) -> None:
    """Raise ValueError if ``slug`` is not a registered type."""
    if slug not in PROJECT_TYPES:
        raise ValueError(
            f"unknown project type {slug!r} — available: "
            + ", ".join(PROJECT_TYPES) + "."
        )


def default_sections_as_dicts(pt: ProjectType) -> list[dict]:
    """Shape expected by BookChapter.sections (JSONB list of dicts).

    Mirrors the legacy ``DEFAULT_SECTIONS`` shape from
    ``sciknow.core.book_ops`` so we can feed it straight into a chapter
    row without a translation layer.
    """
    return [
        {"key": s.key, "title": s.title,
         "target_words": s.target_words, "required": s.required}
        for s in pt.default_sections
    ]


def section_keys(pt: ProjectType) -> list[str]:
    """Just the canonical section_type keys, for quick iteration."""
    return [s.key for s in pt.default_sections]


def target_words_for(pt: ProjectType, section_key: str) -> int | None:
    """Per-section target word count, or None if not in the template."""
    for s in pt.default_sections:
        if s.key == section_key:
            return s.target_words
    return None
