"""Phase 54.6.328 (snapshot-versioning Phase 3) — version history walker.

Two helpers feed the `book history` and `book diff` CLI verbs:

- ``list_section_history(chapter_id, section_slug)`` — interleaves the
  ``drafts.version`` chain with any ``draft_snapshots`` rows scoped to
  one of those drafts. Result ordered by created_at DESC, each entry
  carries the diff brief computed lazily against the immediately
  prior entry in chronological order.

- ``resolve_version_ref(ref)`` — accepts polymorphic identifiers and
  returns ``(kind, id, content, label)``:

    full UUID                   → 'draft' if in drafts, else 'snapshot'
    UUID prefix (≥6 chars)      → unique-resolve in drafts then snapshots
    "<chapter>:<section>:vN"    → drafts row at that version
    "<chapter>:<section>:latest"→ active draft

  Used by ``book diff`` so any two refs (draft, snapshot, current
  state) can be compared.

Both helpers are pure functions over a SQLAlchemy session — they don't
own a session lifetime. Callers compose with ``get_session()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text


@dataclass
class HistoryEntry:
    """One row in a section's interleaved version+snapshot timeline."""
    kind: str            # 'draft' | 'snapshot'
    id: str              # UUID
    label: str           # "v8" / snapshot name / etc.
    created_at: str      # ISO timestamp
    word_count: int
    content: str
    meta: dict           # diff brief vs prior entry (lazily filled)
    is_active: bool      # True only for the currently-active draft
    version: int | None  # drafts row's version, or None for snapshots
    extra: dict          # model_used, score, parent_draft_id, snapshot scope


def list_section_history(
    session,
    *,
    chapter_id: str | None = None,
    section_slug: str | None = None,
    draft_id: str | None = None,
    include_briefs: bool = True,
) -> list[HistoryEntry]:
    """Walk every drafts.version + draft_snapshots row for one section.

    Either pass (chapter_id, section_slug) or a single draft_id; in the
    latter case we infer the chapter+section by looking the draft up.

    Returns entries newest-first. Each entry's ``meta`` is the prose
    diff brief of THIS entry's content vs the immediately-older
    entry's content. The oldest entry's brief uses an empty-string
    baseline ("everything was added").
    """
    from sciknow.core.snapshot_diff import compute_prose_diff

    if not draft_id and not (chapter_id and section_slug):
        raise ValueError(
            "list_section_history needs either draft_id or "
            "(chapter_id, section_slug)"
        )

    if draft_id and not (chapter_id and section_slug):
        row = session.execute(text("""
            SELECT chapter_id::text, section_type FROM drafts
            WHERE id::text LIKE :q LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()
        if not row:
            return []
        chapter_id, section_slug = row[0], row[1]

    # Pull every version row for this (chapter, section_type).
    drafts = session.execute(text("""
        SELECT id::text, version, content, word_count, created_at,
               COALESCE((custom_metadata->>'is_active')::boolean, FALSE),
               model_used, parent_draft_id::text,
               custom_metadata->>'final_overall'
        FROM drafts
        WHERE chapter_id::text = :cid AND section_type = :slug
        ORDER BY version DESC, created_at DESC
    """), {"cid": chapter_id, "slug": section_slug}).fetchall()

    snaps: list[Any] = []
    if drafts:
        ids = [d[0] for d in drafts]
        snaps = session.execute(text("""
            SELECT id::text, name, content, word_count, created_at,
                   scope, meta
            FROM draft_snapshots
            WHERE draft_id::text = ANY(:ids)
            ORDER BY created_at DESC
        """), {"ids": ids}).fetchall()

    out: list[HistoryEntry] = []
    for d in drafts:
        out.append(HistoryEntry(
            kind="draft",
            id=d[0],
            label=f"v{d[1] or 0}",
            created_at=str(d[4] or ""),
            word_count=int(d[3] or 0),
            content=d[2] or "",
            meta={},
            is_active=bool(d[5]),
            version=int(d[1] or 0),
            extra={
                "model_used": d[6],
                "parent_draft_id": d[7],
                "final_overall": (
                    float(d[8]) if d[8] is not None else None
                ),
            },
        ))
    for s in snaps:
        out.append(HistoryEntry(
            kind="snapshot",
            id=s[0],
            label=(s[1] or "snapshot")[:60],
            created_at=str(s[4] or ""),
            word_count=int(s[3] or 0),
            content=s[2] or "",
            meta=(s[6] if isinstance(s[6], dict) else {}) or {},
            is_active=False,
            version=None,
            extra={"scope": s[5]},
        ))

    out.sort(key=lambda e: e.created_at, reverse=True)

    if include_briefs:
        # The oldest entry's prior is empty-string; each newer entry's
        # prior is the next-older entry. Walk in created_at-ascending
        # order so each compute_prose_diff has the right baseline.
        chrono = list(reversed(out))
        prev_content = ""
        for entry in chrono:
            entry.meta = compute_prose_diff(prev_content, entry.content)
            prev_content = entry.content

    return out


def resolve_version_ref(session, ref: str) -> dict | None:
    """Resolve a polymorphic version ref to ``{kind, id, content,
    label, word_count}`` or None.

    Accepts:
      - Full UUID or prefix (≥6 chars) — looks in drafts first,
        then draft_snapshots.
      - "<chapter-num>:<section-slug>:vN" → drafts row
      - "<chapter-num>:<section-slug>:latest" → active draft
      - "<chapter-num>:<section-slug>" (no version) → active draft
    """
    ref = (ref or "").strip()
    if not ref:
        return None

    # Format A: chapter:section[:version] — at least one colon.
    if ":" in ref:
        parts = ref.split(":")
        if len(parts) == 2:
            ch_part, slug = parts
            ver_part = "latest"
        elif len(parts) == 3:
            ch_part, slug, ver_part = parts
        else:
            return None
        try:
            ch_num = int(ch_part)
        except ValueError:
            return None
        ch = session.execute(text("""
            SELECT id::text FROM book_chapters WHERE number = :n
            ORDER BY created_at DESC LIMIT 1
        """), {"n": ch_num}).fetchone()
        if not ch:
            return None
        chapter_id = ch[0]
        if ver_part in ("latest", "active", ""):
            row = session.execute(text("""
                SELECT id::text, content, word_count, version
                FROM drafts
                WHERE chapter_id::text = :cid AND section_type = :slug
                ORDER BY
                    COALESCE((custom_metadata->>'is_active')::boolean, FALSE) DESC,
                    CASE WHEN content IS NULL OR LENGTH(content) < 50
                         THEN 1 ELSE 0 END,
                    version DESC
                LIMIT 1
            """), {"cid": chapter_id, "slug": slug}).fetchone()
        else:
            try:
                ver_n = int(ver_part.lstrip("v"))
            except ValueError:
                return None
            row = session.execute(text("""
                SELECT id::text, content, word_count, version
                FROM drafts
                WHERE chapter_id::text = :cid AND section_type = :slug
                  AND version = :v
                LIMIT 1
            """), {"cid": chapter_id, "slug": slug, "v": ver_n}).fetchone()
        if not row:
            return None
        return {
            "kind": "draft",
            "id": row[0],
            "content": row[1] or "",
            "word_count": int(row[2] or 0),
            "label": f"ch{ch_num}:{slug}:v{row[3] or 0}",
        }

    # Format B: UUID or prefix (≥6 chars).
    if len(ref) < 6:
        return None
    drow = session.execute(text("""
        SELECT id::text, content, word_count, version, section_type
        FROM drafts WHERE id::text LIKE :q LIMIT 2
    """), {"q": f"{ref}%"}).fetchall()
    if len(drow) == 1:
        d = drow[0]
        return {
            "kind": "draft",
            "id": d[0],
            "content": d[1] or "",
            "word_count": int(d[2] or 0),
            "label": f"draft {d[0][:8]} v{d[3] or 0} ({d[4]})",
        }
    if len(drow) > 1:
        return None  # ambiguous

    srow = session.execute(text("""
        SELECT id::text, content, word_count, name, scope
        FROM draft_snapshots WHERE id::text LIKE :q LIMIT 2
    """), {"q": f"{ref}%"}).fetchall()
    if len(srow) == 1:
        s = srow[0]
        return {
            "kind": "snapshot",
            "id": s[0],
            "content": s[1] or "",
            "word_count": int(s[2] or 0),
            "label": f"snapshot {s[0][:8]} {s[3]} (scope={s[4]})",
        }
    return None
