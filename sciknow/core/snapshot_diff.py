"""Phase 54.6.328 — diff brief computation for snapshots.

Each snapshot row carries a structured ``meta`` dict computed at
create-time. CLI list + GUI Snapshots modal render briefs without
re-fetching content. Three shapes:

- **Prose** (scope='draft' or per-section inside a bundle):
  ``{words_added, words_removed, paragraphs_added,
     paragraphs_removed, citations_added, citations_removed}``
- **Bundle** (scope='chapter' or 'book'): aggregated prose stats +
  per-section briefs + optional structural diff.
- **Structural** (book scope only, Phase 6 follow-up): chapter
  add/remove/rename, section list deltas.

All shapes are JSON-serialisable, ASCII-only, computed in <50 ms
for typical inputs (5kw section / 10-chapter book).
"""
from __future__ import annotations

import re
from typing import Any

# A "citation marker" in sciknow's drafts is `[N]` where N is the
# 1-indexed source number. Same regex used by the bibliography
# remap path in core/bibliography.py.
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _word_set_diff(before: str, after: str) -> tuple[int, int]:
    """Return (added, removed) word counts using a Counter-based diff.

    NOT the same as a longest-common-subsequence diff — this is a
    multiset diff. For prose change-magnitude reporting it's what
    users want ("how many words were added vs removed in net").
    """
    from collections import Counter
    bw = Counter(before.split())
    aw = Counter(after.split())
    added = sum((aw - bw).values())
    removed = sum((bw - aw).values())
    return added, removed


def _paragraph_count(text: str) -> int:
    """Count non-empty paragraphs (separated by blank lines)."""
    if not text:
        return 0
    return sum(1 for p in re.split(r"\n\s*\n", text) if p.strip())


def _citation_set(text: str) -> set[int]:
    """Set of unique citation indices in the text."""
    return {int(m) for m in _CITATION_RE.findall(text or "")}


def compute_prose_diff(before: str, after: str) -> dict[str, int]:
    """Compute the prose diff brief for a (before, after) text pair.

    Both texts may be empty strings. Returns a stable shape so
    consumers can index into the dict without checking for keys.
    """
    before = before or ""
    after = after or ""
    words_added, words_removed = _word_set_diff(before, after)
    p_before = _paragraph_count(before)
    p_after = _paragraph_count(after)
    cites_before = _citation_set(before)
    cites_after = _citation_set(after)
    return {
        "words_added": int(words_added),
        "words_removed": int(words_removed),
        "paragraphs_added": max(0, p_after - p_before),
        "paragraphs_removed": max(0, p_before - p_after),
        "citations_added": len(cites_after - cites_before),
        "citations_removed": len(cites_before - cites_after),
    }


def compute_bundle_brief(
    bundle: dict[str, Any],
    prev_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute the bundle brief for a chapter or book scope snapshot.

    ``bundle`` matches the shape produced by ``_snapshot_chapter_drafts``
    in ``web/routes/snapshots.py``: ``{drafts: [{section_type, content,
    word_count, ...}, ...]}`` for scope=chapter, or ``{chapters: [{...,
    drafts: [...]}, ...]}`` for scope=book.

    ``prev_bundle`` is the immediately prior snapshot of the same
    container, or None for the first snapshot. When None, every
    section's "before" is the empty string and the brief reads as
    "everything was added".

    Returns ``{sections: {<slug>: <prose-brief>, ...}, totals: {...}}``
    plus, for book-scope, ``structural`` (Phase 6).
    """
    # Normalise: book bundles wrap chapter bundles; chapter bundles
    # carry drafts directly. Walk down to a flat list of sections.
    def _flatten(b: dict | None) -> list[dict]:
        if not b:
            return []
        if "chapters" in b:
            out: list[dict] = []
            for ch in b.get("chapters") or []:
                out.extend(ch.get("drafts") or [])
            return out
        return list(b.get("drafts") or [])

    cur_sections = _flatten(bundle)
    prev_sections = _flatten(prev_bundle)

    # Index by section_type for cross-bundle pairing. When the same
    # section appears twice (multiple chapter copies), pair by
    # (chapter_id, section_type) using whichever fields exist.
    def _key(d: dict) -> str:
        return f"{d.get('chapter_id', '')}:{d.get('section_type', '')}"

    prev_map = {_key(d): d for d in prev_sections}
    section_briefs: dict[str, dict] = {}
    totals = {
        "words_added": 0, "words_removed": 0,
        "paragraphs_added": 0, "paragraphs_removed": 0,
        "citations_added": 0, "citations_removed": 0,
        "sections_total": len(cur_sections),
        "sections_changed": 0,
    }
    for d in cur_sections:
        st = d.get("section_type") or "?"
        prev_d = prev_map.get(_key(d))
        before = (prev_d or {}).get("content") or ""
        after = d.get("content") or ""
        brief = compute_prose_diff(before, after)
        section_briefs[st] = brief
        if any((
            brief["words_added"], brief["words_removed"],
            brief["paragraphs_added"], brief["paragraphs_removed"],
        )):
            totals["sections_changed"] += 1
        for k in (
            "words_added", "words_removed",
            "paragraphs_added", "paragraphs_removed",
            "citations_added", "citations_removed",
        ):
            totals[k] += brief[k]

    out = {"sections": section_briefs, "totals": totals}

    # Phase 6 — when both bundles are book-scope (carry a `chapters`
    # list with chapter-shape entries), include the structural diff.
    # Each chapter dict needs at least number + title + sections (slug
    # list); _snapshot_book_drafts enriches the bundle accordingly.
    if (
        bundle and "chapters" in bundle
        and prev_bundle and "chapters" in prev_bundle
    ):
        out["structural"] = compute_outline_structural_diff(
            _outline_chapters(prev_bundle.get("chapters") or []),
            _outline_chapters(bundle.get("chapters") or []),
        )
    return out


def _outline_chapters(chapter_bundles: list[dict]) -> list[dict]:
    """Project a list of chapter snapshot bundles to the minimum
    structural shape ``[{number, title, sections}, ...]``.

    Bundles produced before Phase 6 didn't carry the section list;
    they fall through with empty sections, so structural diffs against
    them just show "~added" / "~removed" chapter changes.
    """
    out: list[dict] = []
    for ch in chapter_bundles or []:
        out.append({
            "number": ch.get("chapter_number"),
            "title": ch.get("chapter_title", ""),
            "sections": ch.get("sections_meta") or ch.get("sections") or [],
        })
    return out


def compute_outline_structural_diff(
    before_chapters: list[dict],
    after_chapters: list[dict],
) -> dict[str, Any]:
    """Phase 6 — structural diff between two outlines (book-scope only).

    Each chapter dict must carry at least ``number``, ``title``, and
    ``sections`` (list of slug strings). Returns a JSON-serialisable
    dict with chapter add/remove/rename + section-list deltas per
    chapter.

    Idempotent: passing the same list twice returns all-empty deltas.
    """
    def _norm_sections(ch: dict) -> list[str]:
        secs = ch.get("sections") or []
        out: list[str] = []
        for s in secs:
            if isinstance(s, str):
                out.append(s)
            elif isinstance(s, dict):
                out.append(s.get("slug") or s.get("type") or "")
        return [s for s in out if s]

    before_by_num = {int(c.get("number", 0) or 0): c for c in before_chapters}
    after_by_num = {int(c.get("number", 0) or 0): c for c in after_chapters}
    before_titles = {
        (c.get("title") or "").strip().lower(): int(c.get("number", 0) or 0)
        for c in before_chapters
    }

    added_chapters: list[dict] = []
    removed_chapters: list[dict] = []
    renamed_chapters: list[dict] = []
    section_changes: list[dict] = []

    # By number — primary key. Number drift handled in the rename pass.
    for num, ch in after_by_num.items():
        if num not in before_by_num:
            tlow = (ch.get("title") or "").strip().lower()
            # Possibly renumbered — if the title matches a removed-by-
            # number chapter, count as a renumber rather than add.
            if tlow in before_titles:
                renamed_chapters.append({
                    "from_number": before_titles[tlow],
                    "to_number": num,
                    "title": ch.get("title", ""),
                    "kind": "renumbered",
                })
            else:
                added_chapters.append({
                    "number": num,
                    "title": ch.get("title", ""),
                })
    for num, ch in before_by_num.items():
        if num not in after_by_num:
            tlow = (ch.get("title") or "").strip().lower()
            after_titles = {
                (c.get("title") or "").strip().lower()
                for c in after_chapters
            }
            if tlow not in after_titles:
                removed_chapters.append({
                    "number": num,
                    "title": ch.get("title", ""),
                })

    # Title rename within the same chapter number.
    for num, ach in after_by_num.items():
        bch = before_by_num.get(num)
        if not bch:
            continue
        a_title = (ach.get("title") or "").strip()
        b_title = (bch.get("title") or "").strip()
        if a_title != b_title:
            renamed_chapters.append({
                "number": num,
                "from_title": b_title,
                "to_title": a_title,
                "kind": "retitled",
            })

    # Section-list delta per chapter. Compares slug sets; renames
    # within a slug aren't visible at this layer.
    for num, ach in after_by_num.items():
        bch = before_by_num.get(num)
        if not bch:
            continue
        a_secs = _norm_sections(ach)
        b_secs = _norm_sections(bch)
        a_set, b_set = set(a_secs), set(b_secs)
        added = sorted(a_set - b_set)
        removed = sorted(b_set - a_set)
        if added or removed:
            section_changes.append({
                "chapter_number": num,
                "chapter_title": ach.get("title", ""),
                "added": added,
                "removed": removed,
            })

    return {
        "added_chapters": added_chapters,
        "removed_chapters": removed_chapters,
        "renamed_chapters": renamed_chapters,
        "section_changes": section_changes,
    }


def render_brief_one_line(meta: dict[str, Any]) -> str:
    """Format a snapshot row's ``meta`` dict as a compact one-line
    brief suitable for the CLI list + GUI table cell.

    Empty / missing meta renders as ``"—"`` so old rows that predate
    the diff-brief column (default '{}') don't blank-line the table.
    """
    if not meta:
        return "—"
    # Section/draft scope — top-level prose keys.
    if "words_added" in meta or "words_removed" in meta:
        return _format_prose_segment(meta)
    # Bundle scope — totals dict.
    totals = meta.get("totals") or {}
    if not totals:
        return "—"
    parts: list[str] = [_format_prose_segment(totals)]
    sec_total = totals.get("sections_total")
    sec_changed = totals.get("sections_changed")
    if sec_total is not None and sec_changed is not None:
        parts.append(f"{sec_changed}/{sec_total}§")
    structural = meta.get("structural") or {}
    s_bits: list[str] = []
    n_added = len(structural.get("added_chapters") or [])
    n_removed = len(structural.get("removed_chapters") or [])
    n_renamed = len(structural.get("renamed_chapters") or [])
    if n_added:
        s_bits.append(f"+{n_added}ch")
    if n_removed:
        s_bits.append(f"-{n_removed}ch")
    if n_renamed:
        s_bits.append(f"~{n_renamed}ch")
    if s_bits:
        parts.append(" ".join(s_bits))
    return " · ".join(parts)


def _format_prose_segment(d: dict) -> str:
    wa = int(d.get("words_added", 0) or 0)
    wr = int(d.get("words_removed", 0) or 0)
    pa = int(d.get("paragraphs_added", 0) or 0)
    pr = int(d.get("paragraphs_removed", 0) or 0)
    ca = int(d.get("citations_added", 0) or 0)
    cr = int(d.get("citations_removed", 0) or 0)
    bits: list[str] = []
    if wa or wr:
        bits.append(f"+{wa:,}/-{wr:,}w")
    if pa or pr:
        bits.append(f"+{pa}/-{pr}¶")
    if ca or cr:
        bits.append(f"+{ca}/-{cr}cite")
    return " · ".join(bits) if bits else "—"
