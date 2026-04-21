"""Phase 54.6.153 — whole-book length report.

Walks every chapter × every section and runs the same resolver chain
autowrite uses (``book_ops._get_section_target_words``,
``_get_section_concept_density_target``, ``_get_book_length_target``,
etc.) so users can see their book's *projected* length before running
autowrite.

Reused by:
  * ``sciknow book length-report`` CLI
  * Potential future web endpoint / UI button

Returns structured data so the caller can render a Rich table, JSON,
or whatever. No I/O of its own.

Note: this deliberately does NOT fire the Phase 54.6.150 retrieval-
density widener. That adjustment depends on actual retrieval results
(``len(results)`` from ``hybrid_search``), which would require running
retrieval against every section just to preview targets — too
expensive for an at-a-glance report. The report uses the
concept-density midpoint (the pre-widener value), which is typically
within ±50% of the final widened target.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)


@dataclass
class SectionEntry:
    slug: str
    title: str
    target: int
    level: str                # explicit_section_override | concept_density | chapter_split
    concepts: int | None = None
    wpc_midpoint: int | None = None
    explanation: str = ""


@dataclass
class ChapterEntry:
    chapter_id: str
    number: int
    title: str
    chapter_target: int       # resolver Level-3 chapter target (before the split)
    chapter_level: str        # explicit_chapter_override | book_default | type_default | hardcoded_fallback
    sections: list[SectionEntry] = field(default_factory=list)

    @property
    def total_words(self) -> int:
        """Sum of resolved per-section targets."""
        return sum(s.target for s in self.sections)

    @property
    def n_sections(self) -> int:
        return len(self.sections)


@dataclass
class BookLengthReport:
    book_id: str
    title: str
    book_type: str
    chapters: list[ChapterEntry] = field(default_factory=list)

    @property
    def total_words(self) -> int:
        return sum(c.total_words for c in self.chapters)

    @property
    def n_chapters(self) -> int:
        return len(self.chapters)

    @property
    def n_sections(self) -> int:
        return sum(c.n_sections for c in self.chapters)

    def level_histogram(self) -> dict[str, int]:
        """How many sections fall into each fallback level? Useful for
        answering "what fraction of my book uses concept-density vs
        chapter-split?"."""
        out: dict[str, int] = {}
        for c in self.chapters:
            for s in c.sections:
                out[s.level] = out.get(s.level, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "book_id": self.book_id,
            "title": self.title,
            "book_type": self.book_type,
            "n_chapters": self.n_chapters,
            "n_sections": self.n_sections,
            "total_words": self.total_words,
            "level_histogram": self.level_histogram(),
            "chapters": [
                {
                    "chapter_id": c.chapter_id,
                    "number": c.number,
                    "title": c.title,
                    "chapter_target": c.chapter_target,
                    "chapter_level": c.chapter_level,
                    "total_words": c.total_words,
                    "sections": [
                        {
                            "slug": s.slug, "title": s.title,
                            "target": s.target, "level": s.level,
                            "concepts": s.concepts,
                            "wpc_midpoint": s.wpc_midpoint,
                            "explanation": s.explanation,
                        }
                        for s in c.sections
                    ],
                }
                for c in self.chapters
            ],
        }


def walk_book_lengths(book_id: str) -> BookLengthReport:
    """Walk every chapter × every section of the book and resolve the
    target for each. Uses the real resolver helpers so the report never
    drifts from autowrite's actual behaviour.
    """
    from sciknow.core.book_ops import (
        _get_book_length_target, _section_target_words,
        _get_section_target_words, _get_section_concept_density_target,
        _get_chapter_sections_normalized, _count_plan_concepts,
        _get_section_plan, DEFAULT_TARGET_CHAPTER_WORDS,
    )
    from sciknow.core.project_type import get_project_type
    from sciknow.storage.db import get_session
    import json as _json

    with get_session() as session:
        book_row = session.execute(sql_text("""
            SELECT id::text, title, book_type,
                   COALESCE(custom_metadata, '{}'::jsonb)
            FROM books WHERE id::text = :bid LIMIT 1
        """), {"bid": book_id}).fetchone()
        if not book_row:
            raise ValueError(f"no book with id {book_id!r}")
        book_id_out, title, book_type, book_meta = book_row
        if isinstance(book_meta, str):
            try:
                book_meta = _json.loads(book_meta)
            except Exception:
                book_meta = {}
        book_meta = book_meta or {}
        book_type = book_type or "scientific_book"

        pt = None
        wpc_mid = None
        try:
            pt = get_project_type(book_type)
            wlo, whi = pt.words_per_concept_range
            wpc_mid = (wlo + whi) // 2
        except Exception as exc:
            logger.debug("project_type lookup failed: %s", exc)

        ch_rows = session.execute(sql_text("""
            SELECT id::text, number, title, target_words
            FROM book_chapters
            WHERE book_id = CAST(:bid AS uuid)
            ORDER BY number
        """), {"bid": book_id}).fetchall()

        chapters: list[ChapterEntry] = []
        for ch_id, ch_num, ch_title, ch_tw in ch_rows:
            # Resolve chapter target (Level 1 → 2 → 3 → 4).
            if ch_tw and int(ch_tw) > 0:
                chapter_target = int(ch_tw)
                chapter_level = "explicit_chapter_override"
            elif isinstance(book_meta.get("target_chapter_words"), (int, float)) \
                    and book_meta["target_chapter_words"] > 0:
                chapter_target = int(book_meta["target_chapter_words"])
                chapter_level = "book_default"
            elif pt is not None:
                chapter_target = pt.default_target_chapter_words
                chapter_level = "type_default"
            else:
                chapter_target = DEFAULT_TARGET_CHAPTER_WORDS
                chapter_level = "hardcoded_fallback"

            sections = _get_chapter_sections_normalized(session, ch_id)
            n = max(1, len(sections))
            chapter_split = _section_target_words(chapter_target, n)

            sec_entries: list[SectionEntry] = []
            for s in sections:
                slug = s.get("slug", "")
                stitle = s.get("title", "")
                override = _get_section_target_words(session, ch_id, slug)
                if override is not None:
                    sec_entries.append(SectionEntry(
                        slug=slug, title=stitle,
                        target=override, level="explicit_section_override",
                        wpc_midpoint=wpc_mid,
                        explanation=f"per-section override {override:,}",
                    ))
                    continue
                plan_text = _get_section_plan(session, ch_id, slug)
                n_concepts = _count_plan_concepts(plan_text)
                if n_concepts > 0 and wpc_mid:
                    concept_target = _get_section_concept_density_target(
                        session, ch_id, slug, book_id,
                    )
                    if concept_target:
                        sec_entries.append(SectionEntry(
                            slug=slug, title=stitle,
                            target=concept_target, level="concept_density",
                            concepts=n_concepts, wpc_midpoint=wpc_mid,
                            explanation=(
                                f"{n_concepts} concept(s) × {wpc_mid} wpc "
                                f"(bottom-up from plan)"
                            ),
                        ))
                        continue
                sec_entries.append(SectionEntry(
                    slug=slug, title=stitle,
                    target=chapter_split, level="chapter_split",
                    wpc_midpoint=wpc_mid,
                    explanation=(
                        f"chapter target {chapter_target:,} ÷ {n} sections "
                        f"(top-down fallback)"
                    ),
                ))

            chapters.append(ChapterEntry(
                chapter_id=ch_id, number=int(ch_num),
                title=ch_title or "",
                chapter_target=chapter_target,
                chapter_level=chapter_level,
                sections=sec_entries,
            ))

    return BookLengthReport(
        book_id=book_id_out,
        title=title or "",
        book_type=book_type,
        chapters=chapters,
    )
