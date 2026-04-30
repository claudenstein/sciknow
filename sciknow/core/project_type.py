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

Planned types (next tick of the plan in docs/roadmap/STRATEGY.md):

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
    # Phase 54.6.146 — concept-density sizing inputs.
    #
    # These two ranges feed the new Level-0 resolver path:
    # when a section has a plan (bullet list of concepts), its target
    # word count is computed as num_concepts × words_per_concept
    # (midpoint of the range by default). Chapter/book length then
    # emerges from the sum of section lengths rather than being
    # top-down allocated.
    #
    # Numbers grounded in docs/research/RESEARCH.md §24 (Cowan 2001 for
    # concept counts, derived from section-length distributions for
    # wpc). The (low, high) ranges let the Level-0 resolver pick a
    # midpoint by default but carry enough information for a future
    # retrieval-density-aware widener to push toward the high end
    # when the evidence justifies it (RESEARCH.md §24 guideline 4 —
    # "honest novel engineering").
    concepts_per_section_range: tuple[int, int] = (3, 4)       # Cowan 2001 consensus
    words_per_concept_range:   tuple[int, int] = (300, 625)    # derived


# ────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────

SCIENTIFIC_BOOK = ProjectType(
    slug="scientific_book",
    display_name="Scientific Book",
    description="Hierarchical book with chapters. Trade-science voice: "
                "dense but readable, each chapter building a narrative arc.",
    is_flat=False,
    # Phase 54.6.146 — section defaults updated to concept-density norms
    # (3-4 novel concepts × 500-800 wpc). Sum to ~8000 = trade-science
    # band midpoint (Kahneman, Pinker: 8k-15k/chapter per RESEARCH.md §24).
    default_sections=(
        SectionTemplate("overview",              "Overview",               1400),
        SectionTemplate("key_evidence",          "Key Evidence",           2200),
        SectionTemplate("current_understanding", "Current Understanding",  2200),
        SectionTemplate("open_questions",        "Open Questions",         1400),
        SectionTemplate("summary",               "Summary",                 800),
    ),
    default_chapter_count=8,
    default_target_chapter_words=8000,
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
    # Concept-density: 3-4 novel ideas per section, 500-800 words each
    # (midpoint 650 × 3.5 concepts ≈ 2300 — matches section defaults)
    concepts_per_section_range=(3, 4),
    words_per_concept_range=(500, 800),
)


SCIENTIFIC_PAPER = ProjectType(
    slug="scientific_paper",
    display_name="Scientific Paper",
    description="Single-document IMRaD paper (one chapter, canonical "
                "sections). Tighter length targets + abstract enforcement.",
    is_flat=True,
    # Canonical IMRaD section set. A paper lives in a single chapter
    # in the DB so existing chapter/draft plumbing works unchanged.
    # Section lengths match PubMed IQRs (RESEARCH.md §24): intro 400-760,
    # results 610-1660, discussion 820-1480.
    default_sections=(
        SectionTemplate("abstract",     "Abstract",      200, required=True),
        SectionTemplate("introduction", "Introduction",  550),
        SectionTemplate("methods",      "Methods",       700),
        SectionTemplate("results",      "Results",      1000),
        SectionTemplate("discussion",   "Discussion",   1100),
        SectionTemplate("conclusion",   "Conclusion",    450),
    ),
    default_chapter_count=1,               # one "chapter" = the whole paper
    default_target_chapter_words=4000,     # sum of section defaults, matches PubMed body norms
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
    # Research papers are dense, expert-aimed prose. Brown 2008 idea-
    # density norms + PubMed IQRs → 200-400 words/concept, 2-4 concepts
    # per section (methods can sprawl to 5 in the expert case).
    concepts_per_section_range=(2, 4),
    words_per_concept_range=(200, 400),
)


# Phase 54.6.146 — popular-science project type.
# Trade-narrative voice (Sagan, Gould, Carson). Chapters are shorter
# than trade science (Kahneman, Pinker sit in `scientific_book`) and
# sections prioritise narrative arc over exhaustive coverage. Concept
# density is LOW per RESEARCH.md §24 — popular prose unpacks each
# idea slowly, weaving story and metaphor.
POPULAR_SCIENCE = ProjectType(
    slug="popular_science",
    display_name="Popular Science",
    description="Narrative science for a lay audience. Shorter chapters, "
                "fewer concepts per section, story-driven pacing.",
    is_flat=False,
    default_sections=(
        SectionTemplate("opening_hook",          "Opening Hook",          1200),
        SectionTemplate("core_idea",             "Core Idea",             2000),
        SectionTemplate("evidence_and_stakes",   "Evidence & Stakes",     2000),
        SectionTemplate("closing_reflection",    "Closing Reflection",    1300),
    ),
    default_chapter_count=12,
    # 1200+2000+2000+1300 = 6500 (centre of 5k-8k popular-science band)
    default_target_chapter_words=6500,
    writer_hints=(
        "narrative_voice",
        "low_concept_density",
        "story_driven",
    ),
    flags={
        "allow_citations": True,
        "require_abstract": False,
        "cli_export_defaults": {"format": "markdown"},
    },
    # Narrative prose unpacks each idea generously. 400-700 wpc is wider
    # than any academic band because popular prose earns its length
    # from story scaffolding around each concept, not more concepts.
    concepts_per_section_range=(3, 4),
    words_per_concept_range=(400, 700),
)


# Phase 54.6.146 — instructional-textbook project type.
# Intro-level pedagogy: each section introduces a discrete concept,
# works an example, summarises. Chapters are short (3k-6k) per
# Automateed 2025 / instructional-design norms. Not to be confused
# with ``academic_monograph`` (Bishop PRML, Goodfellow DL — 8k-15k).
INSTRUCTIONAL_TEXTBOOK = ProjectType(
    slug="instructional_textbook",
    display_name="Instructional Textbook",
    description="Intro-level pedagogical textbook. Short chapters, "
                "worked examples, concept-per-section pacing.",
    is_flat=False,
    default_sections=(
        SectionTemplate("concept_introduction", "Concept Introduction", 1000),
        SectionTemplate("worked_example",       "Worked Example",       1500),
        SectionTemplate("deeper_discussion",    "Deeper Discussion",    1500),
        SectionTemplate("check_understanding",  "Check Understanding",   500),
    ),
    default_chapter_count=15,
    # 1000+1500+1500+500 = 4500 (midpoint of 3k-6k intro-textbook band)
    default_target_chapter_words=4500,
    writer_hints=(
        "pedagogical_voice",
        "worked_examples",
        "concept_per_section",
    ),
    flags={
        "allow_citations": True,
        "require_abstract": False,
        "cli_export_defaults": {"format": "markdown"},
    },
    # Pedagogical texts repeat + unpack; slightly wider wpc than
    # monograph because the same concept is introduced, applied, then
    # re-explained in a check-understanding lens.
    concepts_per_section_range=(3, 4),
    words_per_concept_range=(400, 700),
)


# Phase 54.6.143/146 — academic-monograph project type.
# Formerly called "textbook" (54.6.143) — corrected per RESEARCH.md §24:
# Bishop PRML, Goodfellow DL, and similar dense-pedagogy references are
# academic monographs, not intro textbooks. Chapters run 8k-15k.
ACADEMIC_MONOGRAPH = ProjectType(
    slug="academic_monograph",
    display_name="Academic Monograph",
    description="Dense research-grade pedagogical book. Long chapters, "
                "formal derivations, comprehensive per-topic coverage. "
                "Fits Bishop PRML, Goodfellow DL, research monographs.",
    is_flat=False,
    default_sections=(
        SectionTemplate("introduction",         "Introduction",         1800),
        SectionTemplate("background",           "Background",           3500),
        SectionTemplate("main_development",     "Main Development",     4500),
        SectionTemplate("analysis",             "Analysis",             3500),
        SectionTemplate("discussion_open_qs",   "Discussion & Open Questions", 1700),
    ),
    default_chapter_count=10,
    # 1800+3500+4500+3500+1700 = 15000 (midpoint of 8k-15k monograph band)
    default_target_chapter_words=15000,
    writer_hints=(
        "long_form",
        "pedagogical_voice",
        "formal_derivation",
        "concept_density",
    ),
    flags={
        "allow_citations": True,
        "require_abstract": False,
        "cli_export_defaults": {"format": "markdown"},
    },
    # Monograph prose is the densest pedagogical form: 4-5 concepts
    # per section, 600-1000 wpc to support formal derivation + deep
    # analysis. Reader assumed to have working-memory templates.
    concepts_per_section_range=(4, 5),
    words_per_concept_range=(600, 1000),
)


# Phase 54.6.146 — review-article length project type (values
# updated from 54.6.143). Short-form literature review / single-topic
# survey. Total ~4k-8k across 5-10 thematic sections.
REVIEW_ARTICLE = ProjectType(
    slug="review_article",
    display_name="Review Article",
    description="Short-form literature review. Single document, "
                "5–10 thematic sections, tight word budgets.",
    is_flat=True,
    default_sections=(
        SectionTemplate("abstract",           "Abstract",         250, required=True),
        SectionTemplate("introduction",       "Introduction",     700),
        SectionTemplate("historical_context", "Historical Context", 1000),
        SectionTemplate("current_state",      "Current State",    1400),
        SectionTemplate("open_questions",     "Open Questions",   1000),
        SectionTemplate("conclusion",         "Conclusion",        650),
    ),
    default_chapter_count=1,
    # 250+700+1000+1400+1000+650 = 5000 (Nature Review main-text norm)
    default_target_chapter_words=5000,
    writer_hints=(
        "concise_scientific_voice",
        "synthesis_over_exposition",
        "inline_citations_required",
    ),
    flags={
        "allow_citations": True,
        "require_abstract": True,
        "cli_export_defaults": {"format": "markdown"},
    },
    # Review prose is curated: more concepts per section than a paper
    # (synthesis) but less than a monograph (no derivations); tight wpc.
    concepts_per_section_range=(3, 5),
    words_per_concept_range=(250, 450),
)


# Central registry. Extend here + add a default_sections tuple to ship
# a new type. No schema change required.
PROJECT_TYPES: dict[str, ProjectType] = {
    SCIENTIFIC_BOOK.slug:        SCIENTIFIC_BOOK,
    SCIENTIFIC_PAPER.slug:       SCIENTIFIC_PAPER,
    POPULAR_SCIENCE.slug:        POPULAR_SCIENCE,
    INSTRUCTIONAL_TEXTBOOK.slug: INSTRUCTIONAL_TEXTBOOK,
    ACADEMIC_MONOGRAPH.slug:     ACADEMIC_MONOGRAPH,
    REVIEW_ARTICLE.slug:         REVIEW_ARTICLE,
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
