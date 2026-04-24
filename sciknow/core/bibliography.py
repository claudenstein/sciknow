"""Project-scoped bibliography management.

Phase 54.6.309 — globalises citation numbering across a book.

Each chapter/section draft stores its own local `[1], [2], …` source list
(emitted by ``format_sources`` at write time). For export and the web
reader we want ONE bibliography for the whole book with a single
per-publication number, ordered by first appearance in reading order
(chapter number, then section order within the chapter, then existing
local order). This module owns that transformation so the CLI export
path and the web reader don't drift.

Usage::

    bib = BookBibliography.from_book(session, book_id)
    globalised = bib.remap_content(draft_id, draft.content)
    shown = bib.cited_sources_for_draft(draft_id)   # right-panel list
    full  = bib.global_sources                       # bibliography chapter

Dedup key: the source string with its leading ``[N] `` stripped. This
matches what ``format_sources`` emits — same paper, same rendering, so
string equality is a reliable paper identity here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from sqlalchemy import text

# Slug used as the synthetic chapter/draft id for the bibliography
# pseudo-chapter. Deliberately not a valid UUID so it can't collide
# with a real draft id.
BIBLIOGRAPHY_PSEUDO_ID = "__bibliography__"
BIBLIOGRAPHY_TITLE = "Bibliography"

_LOCAL_NUM_RE = re.compile(r"^\s*\[(\d+)\]\s*")


def _source_key(s: str) -> str:
    """Strip the leading ``[N] `` so two drafts that cite the same paper
    at different local numbers still dedupe to one bibliography entry."""
    return _LOCAL_NUM_RE.sub("", s or "").strip()


def _renumber_source_line(s: str, new_num: int) -> str:
    """Rewrite the leading ``[N]`` of a source line to ``[new_num]``.

    Preserves everything after the ``[N] `` prefix. If the line has no
    leading ``[N]`` (malformed source), prepends one so bibliography
    rendering is consistent.
    """
    s = s or ""
    if _LOCAL_NUM_RE.match(s):
        return _LOCAL_NUM_RE.sub(f"[{new_num}] ", s, count=1)
    return f"[{new_num}] {s}"


@dataclass
class BookBibliography:
    """Book-wide bibliography with a single global [N] per publication.

    Attributes:
      global_sources: APA-formatted source lines, each beginning with
        its *global* ``[N]``. Ordered by first appearance when walking
        chapters in ``bc.number`` order, then the section order used
        by the web reader (same SELECT as ``_get_book_data``).
      draft_local_to_global: ``draft_id -> {local_num: global_num}``.
        Only covers drafts whose sources array is non-empty.
      draft_cited_global_nums: ``draft_id -> ordered list[int]`` of the
        global numbers actually referenced in that draft's content
        (for the right-panel "Sources" list). Order is first-cited.
      publication_order: ``global_num -> source_key`` so callers can
        index into `global_sources` without re-parsing.
    """

    global_sources: list[str] = field(default_factory=list)
    draft_local_to_global: dict[str, dict[int, int]] = field(default_factory=dict)
    draft_cited_global_nums: dict[str, list[int]] = field(default_factory=dict)
    # Internal: source_key → global_num (1-based)
    _global_num_by_key: dict[str, int] = field(default_factory=dict)

    # ── Construction ────────────────────────────────────────────────

    @classmethod
    def from_book(cls, session, book_id) -> "BookBibliography":
        """Build the global bibliography for one book.

        Uses the MAX(version) draft per ``(chapter_id, section_type)``
        — same rule as the web reader — so revisions that add/drop a
        source take effect in the book-wide numbering.

        Phase 54.6.311 — honors the active-version flag (``is_active``)
        when present in ``drafts.custom_metadata`` so manually pinning
        an older version via the Versions panel also shifts the global
        numbering to match what the reader sees.
        """
        rows = session.execute(text("""
            SELECT d.id::text, d.chapter_id::text, d.section_type,
                   d.content, d.sources, d.version, d.custom_metadata,
                   COALESCE(bc.number, 999999) AS ch_num
            FROM drafts d
            LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.book_id = :bid
            ORDER BY ch_num, d.section_type, d.version DESC
        """), {"bid": str(book_id)}).fetchall()

        # Collapse to one draft per (chapter_id, section_type) using
        # the same "active version" rule as the reader: if any version
        # has `custom_metadata.is_active = true`, prefer it; otherwise
        # pick the highest version. That keeps the global bibliography
        # numbering locked to what the user sees.
        def _meta_dict(raw) -> dict:
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str) and raw:
                try:
                    loaded = json.loads(raw)
                    return loaded if isinstance(loaded, dict) else {}
                except Exception:
                    return {}
            return {}

        best_for_key: dict[tuple, tuple] = {}
        for r in rows:
            did, ch_id, sec, content, sources, ver, meta, ch_num = r
            key = (ch_id, sec)
            is_active = bool(_meta_dict(meta).get("is_active"))
            cur = best_for_key.get(key)
            if cur is None:
                best_for_key[key] = (r, is_active)
                continue
            cur_row, cur_active = cur
            if is_active and not cur_active:
                best_for_key[key] = (r, True)
            elif is_active == cur_active and (ver or 1) > (cur_row[5] or 1):
                best_for_key[key] = (r, is_active)

        # Order kept drafts by (chapter number, section slug) so global
        # numbering mirrors reading order.
        kept = sorted(
            (v[0] for v in best_for_key.values()),
            key=lambda r: (r[7] if r[7] is not None else 999999, r[2] or ""),
        )

        bib = cls()

        for r in kept:
            did, ch_id, sec, content, sources, ver, meta, ch_num = r
            local_sources = cls._coerce_sources(sources)
            if not local_sources:
                continue
            local_map: dict[int, int] = {}
            for ls in local_sources:
                m = _LOCAL_NUM_RE.match(ls)
                if not m:
                    continue
                local_num = int(m.group(1))
                key = _source_key(ls)
                if not key:
                    continue
                if key not in bib._global_num_by_key:
                    bib._global_num_by_key[key] = len(bib.global_sources) + 1
                    bib.global_sources.append(
                        _renumber_source_line(ls, bib._global_num_by_key[key])
                    )
                local_map[local_num] = bib._global_num_by_key[key]
            bib.draft_local_to_global[did] = local_map

            # Figure out which global numbers this draft actually cites in its body.
            cited = _collect_cited_local_nums(content or "")
            cited_globals: list[int] = []
            seen_g: set[int] = set()
            for ln in cited:
                g = local_map.get(ln)
                if g is None:
                    continue
                if g in seen_g:
                    continue
                seen_g.add(g)
                cited_globals.append(g)
            bib.draft_cited_global_nums[did] = cited_globals

        return bib

    @staticmethod
    def _coerce_sources(raw) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return [s for s in raw if s]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception:
                return []
            return [s for s in parsed if s] if isinstance(parsed, list) else []
        return []

    # ── Query API ────────────────────────────────────────────────────

    def remap_content(self, draft_id: str, content: str) -> str:
        """Rewrite ``[N]`` markers in a draft's content to global numbers.

        Uses a two-pass substitution with ``__CITE_n__`` placeholders so
        renumbering ``[1]→[3]`` doesn't cascade into ``[3]→[something]``.
        Unknown locals (missing from the source list) are left as-is.
        """
        local_map = self.draft_local_to_global.get(draft_id)
        if not content or not local_map:
            return content or ""

        def _to_placeholder(match: re.Match) -> str:
            n = int(match.group(1))
            g = local_map.get(n)
            return f"[__CITE_{g}__]" if g is not None else match.group(0)

        out = re.sub(r"\[(\d+)\]", _to_placeholder, content)
        out = re.sub(r"\[__CITE_(\d+)__\]", r"[\1]", out)
        return out

    def cited_sources_for_draft(self, draft_id: str) -> list[str]:
        """Return the subset of the global bibliography actually cited by a
        draft, ordered by first mention. Each line keeps its global ``[N]``
        so the right-panel anchors line up with the rewritten content.
        """
        nums = self.draft_cited_global_nums.get(draft_id, [])
        if not nums:
            return []
        return [self.global_sources[n - 1] for n in nums if 1 <= n <= len(self.global_sources)]

    def synthetic_bibliography_chapter_num(self, chapter_count: int) -> int:
        """Sidebar number for the synthetic Bibliography chapter: one
        past the last real chapter. Callers pass the real chapter count."""
        return int(chapter_count) + 1


def _collect_cited_local_nums(content: str) -> list[int]:
    """Return local citation numbers in content in first-mention order.

    Only matches bare ``[N]`` markers (digits only), not ranges like
    ``[1, 2]`` (which aren't produced by our writer prompts anyway).
    Duplicates are collapsed.
    """
    out: list[int] = []
    seen: set[int] = set()
    for m in re.finditer(r"\[(\d+)\]", content or ""):
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def render_bibliography_markdown(bib: BookBibliography) -> str:
    """Render the bibliography pseudo-chapter as Markdown. One entry
    per line, pre-numbered so it renders cleanly through ``_md_to_html``.
    The ``[N]`` prefix inside each entry becomes the click target that
    ``buildPopovers`` already knows how to highlight.
    """
    if not bib.global_sources:
        return (
            "*No citations yet. Write a section that uses retrieval and its "
            "sources will appear here, numbered once for the whole book.*"
        )
    lines: list[str] = []
    for i, s in enumerate(bib.global_sources, 1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)
