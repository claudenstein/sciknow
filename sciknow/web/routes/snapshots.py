"""``sciknow.web.routes.snapshots`` — draft / chapter / book snapshot endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. 8 handlers
covering the snapshot system:

  POST /api/snapshot/{draft_id}                save named per-draft snapshot
  GET  /api/snapshots/{draft_id}               list per-draft snapshots
  GET  /api/snapshot-content/{snapshot_id}     fetch one snapshot's body
  POST /api/snapshot/chapter/{chapter_id}      bundle every draft in a chapter
  POST /api/snapshot/book/{book_id}            bundle every chapter in a book
  GET  /api/snapshots/chapter/{chapter_id}     list chapter-scope snapshots
  GET  /api/snapshots/book/{book_id}           list book-scope snapshots
  POST /api/snapshot/restore-bundle/{snap_id}  non-destructive bundle restore

Self-contained — no app.py cross-deps. The two helpers
(`_snapshot_chapter_drafts`, `_restore_chapter_bundle`) move with
the routes since they aren't referenced anywhere else.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import text

from sciknow.core.snapshot_diff import (
    compute_bundle_brief,
    compute_prose_diff,
)
from sciknow.storage.db import get_session

router = APIRouter()


@router.post("/api/snapshot/{draft_id}")
async def create_snapshot(draft_id: str, name: str = Form("")):
    """Save a named snapshot of a draft's current content."""
    with get_session() as session:
        draft = session.execute(text(
            "SELECT content, word_count FROM drafts WHERE id::text LIKE :q LIMIT 1"
        ), {"q": f"{draft_id}%"}).fetchone()
        if not draft:
            raise HTTPException(404, "Draft not found")

        # Phase 54.6.328 — compute the diff brief vs the prior snapshot
        # of the same draft (or empty-string baseline for first snapshot).
        prev = session.execute(text("""
            SELECT content FROM draft_snapshots
            WHERE draft_id::text LIKE :q
            ORDER BY created_at DESC LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()
        prev_content = prev[0] if prev else ""
        meta = compute_prose_diff(prev_content, draft[0] or "")

        snap_name = name or f"Snapshot {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}"
        session.execute(text("""
            INSERT INTO draft_snapshots (draft_id, name, content, word_count, meta)
            VALUES (CAST(:did AS uuid), :name, :content, :wc, CAST(:meta AS jsonb))
        """), {"did": draft_id, "name": snap_name,
               "content": draft[0], "wc": draft[1],
               "meta": json.dumps(meta)})
        session.commit()

    return JSONResponse({"ok": True, "name": snap_name, "meta": meta})


@router.get("/api/snapshots/{draft_id}")
async def list_snapshots(draft_id: str):
    """List all snapshots for a draft."""
    with get_session() as session:
        rows = session.execute(text("""
            SELECT id::text, name, word_count, created_at, meta
            FROM draft_snapshots WHERE draft_id::text LIKE :q
            ORDER BY created_at DESC
        """), {"q": f"{draft_id}%"}).fetchall()

    return {"snapshots": [
        {"id": r[0], "name": r[1], "word_count": r[2] or 0,
         "created_at": str(r[3]) if r[3] else "",
         "meta": (r[4] if isinstance(r[4], dict) else {}) or {}}
        for r in rows
    ]}


@router.get("/api/snapshot-content/{snapshot_id}")
async def get_snapshot_content(snapshot_id: str):
    """Get the content of a specific snapshot."""
    with get_session() as session:
        row = session.execute(text(
            "SELECT content FROM draft_snapshots WHERE id::text = :sid"
        ), {"sid": snapshot_id}).fetchone()
    if not row:
        raise HTTPException(404, "Snapshot not found")
    return {"content": row[0]}


# ── Phase 38: chapter + book snapshots (bundle of all drafts) ────────────────


def _snapshot_chapter_drafts(session, chapter_id: str) -> dict:
    """Build the JSON bundle that will be stored in `content`.

    Captures the LATEST version per (chapter_id, section_type) —
    i.e. what the user would see in the GUI right now. Orphan drafts
    (chapter_id set but section_type is None or already replaced) are
    not special-cased: we just take the newest row per section_type.
    """
    rows = session.execute(text("""
        SELECT DISTINCT ON (d.section_type)
            d.id::text, d.section_type, d.title, d.version, d.word_count,
            d.content, d.sources
        FROM drafts d
        WHERE d.chapter_id::text = :cid
        ORDER BY d.section_type, d.version DESC, d.created_at DESC
    """), {"cid": chapter_id}).fetchall()
    return {
        "chapter_id": chapter_id,
        "drafts": [
            {
                "id": r[0],
                "section_type": r[1],
                "title": r[2] or "",
                "version": r[3] or 1,
                "word_count": r[4] or 0,
                "content": r[5] or "",
                "sources": r[6] if isinstance(r[6], list) else [],
            }
            for r in rows
        ],
    }


def _prev_bundle_content(session, *, scope: str, container_id: str) -> dict | None:
    """Fetch the most-recent prior bundle for diff-brief computation.

    scope='chapter' / 'book'. Returns the parsed JSON (the same shape
    that gets stored under ``content``) or None if no prior snapshot.
    """
    if scope == "chapter":
        row = session.execute(text("""
            SELECT content FROM draft_snapshots
            WHERE chapter_id::text = :cid AND scope = 'chapter'
            ORDER BY created_at DESC LIMIT 1
        """), {"cid": container_id}).fetchone()
    else:
        row = session.execute(text("""
            SELECT content FROM draft_snapshots
            WHERE book_id::text = :bid AND scope = 'book'
            ORDER BY created_at DESC LIMIT 1
        """), {"bid": container_id}).fetchone()
    if not row or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def _snapshot_book_drafts(session, book_id: str, *, name: str = "") -> str | None:
    """Snapshot every draft in a book as a single ``scope='book'`` row.

    Mirrors the inline logic in ``cli/book.py::snapshot`` (whole-book
    branch). Used by ``book outline --overwrite`` so the destructive
    re-outline path can capture state with one call instead of
    duplicating ~25 lines of bundle-building.

    Returns the inserted ``draft_snapshots.id`` (UUID as string). Returns
    None if the book has no drafts to snapshot — caller should treat
    that as "nothing to preserve" and continue.

    Phase 54.6.328 — also computes the diff brief vs the prior book
    snapshot and stores it in the new ``meta`` column.
    """
    import datetime as _dt
    chapters = session.execute(text(
        "SELECT id::text, number, title, sections FROM book_chapters "
        "WHERE book_id::text = :bid ORDER BY number"
    ), {"bid": book_id}).fetchall()
    chapter_bundles: list[dict] = []
    grand = 0
    for ch in chapters:
        b = _snapshot_chapter_drafts(session, ch[0])
        if not b["drafts"]:
            continue
        b["chapter_number"] = ch[1]
        b["chapter_title"] = ch[2] or ""
        # Phase 54.6.328 (snapshot-versioning Phase 6) — carry the
        # chapter's sections list (slug-only projection) so the
        # bundle brief's structural diff can detect chapter / section
        # add/remove/rename across two book snapshots.
        raw_sec = ch[3] if isinstance(ch[3], list) else []
        b["sections_meta"] = [
            (s.get("slug") if isinstance(s, dict) else str(s))
            for s in raw_sec
        ]
        chapter_bundles.append(b)
        grand += sum(d.get("word_count") or 0 for d in b["drafts"])
    if not chapter_bundles:
        return None
    snap_name = (name or "").strip() or (
        f"book {book_id[:8]} — {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    bundle = {"book_id": book_id, "chapters": chapter_bundles}
    payload = json.dumps(bundle)
    prev_bundle = _prev_bundle_content(session, scope="book", container_id=book_id)
    meta = compute_bundle_brief(bundle, prev_bundle)
    row = session.execute(text("""
        INSERT INTO draft_snapshots
            (book_id, scope, name, content, word_count, meta)
        VALUES
            (CAST(:bid AS uuid), 'book', :name, :content, :wc,
             CAST(:meta AS jsonb))
        RETURNING id::text
    """), {"bid": book_id, "name": snap_name,
           "content": payload, "wc": grand,
           "meta": json.dumps(meta)}).fetchone()
    return row[0] if row else None


@router.post("/api/snapshot/chapter/{chapter_id}")
async def create_chapter_snapshot(chapter_id: str, name: str = Form("")):
    """Snapshot every draft in a chapter as one bundle.

    Phase 38 — the safety net for autowrite-all on a chapter. Takes a
    label, stores the chapter's current draft state as a JSON bundle
    in a single `draft_snapshots` row with scope='chapter'.

    Phase 54.6.328 — also computes the diff brief vs the prior
    chapter snapshot and stores it in ``meta``.
    """
    import datetime as _dt
    with get_session() as session:
        ch = session.execute(text(
            "SELECT id::text, title FROM book_chapters WHERE id::text = :cid"
        ), {"cid": chapter_id}).fetchone()
        if not ch:
            raise HTTPException(404, "Chapter not found")
        bundle = _snapshot_chapter_drafts(session, chapter_id)
        if not bundle["drafts"]:
            raise HTTPException(400, "Chapter has no drafts to snapshot")
        snap_name = (name or "").strip() or (
            f"{ch[1]} — {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        payload = json.dumps(bundle)
        total_words = sum(d.get("word_count") or 0 for d in bundle["drafts"])
        prev_bundle = _prev_bundle_content(
            session, scope="chapter", container_id=chapter_id,
        )
        meta = compute_bundle_brief(bundle, prev_bundle)
        session.execute(text("""
            INSERT INTO draft_snapshots
                (chapter_id, scope, name, content, word_count, meta)
            VALUES
                (CAST(:cid AS uuid), 'chapter', :name, :content, :wc,
                 CAST(:meta AS jsonb))
        """), {"cid": chapter_id, "name": snap_name,
               "content": payload, "wc": total_words,
               "meta": json.dumps(meta)})
        session.commit()
    return JSONResponse({
        "ok": True, "name": snap_name,
        "drafts_included": len(bundle["drafts"]),
        "total_words": total_words,
        "meta": meta,
    })


@router.post("/api/snapshot/book/{book_id}")
async def create_book_snapshot(book_id: str, name: str = Form("")):
    """Snapshot every draft across every chapter in a book.

    Phase 38 — the union-of-chapters safety net. Useful before running
    autowrite at the whole-book level or before a risky refactor.
    """
    import datetime as _dt
    with get_session() as session:
        book = session.execute(text(
            "SELECT id::text, title FROM books WHERE id::text = :bid"
        ), {"bid": book_id}).fetchone()
        if not book:
            raise HTTPException(404, "Book not found")
        chapters = session.execute(text(
            "SELECT id::text, number, title, sections FROM book_chapters "
            "WHERE book_id::text = :bid ORDER BY number"
        ), {"bid": book_id}).fetchall()

        chapter_bundles = []
        grand_total = 0
        for ch in chapters:
            bundle = _snapshot_chapter_drafts(session, ch[0])
            if not bundle["drafts"]:
                continue
            bundle["chapter_number"] = ch[1]
            bundle["chapter_title"] = ch[2] or ""
            raw_sec = ch[3] if isinstance(ch[3], list) else []
            bundle["sections_meta"] = [
                (s.get("slug") if isinstance(s, dict) else str(s))
                for s in raw_sec
            ]
            chapter_bundles.append(bundle)
            grand_total += sum(d.get("word_count") or 0 for d in bundle["drafts"])

        if not chapter_bundles:
            raise HTTPException(400, "Book has no drafts to snapshot")

        snap_name = (name or "").strip() or (
            f"{book[1]} — {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        bundle = {"book_id": book_id, "chapters": chapter_bundles}
        payload = json.dumps(bundle)
        prev_bundle = _prev_bundle_content(
            session, scope="book", container_id=book_id,
        )
        meta = compute_bundle_brief(bundle, prev_bundle)
        session.execute(text("""
            INSERT INTO draft_snapshots
                (book_id, scope, name, content, word_count, meta)
            VALUES
                (CAST(:bid AS uuid), 'book', :name, :content, :wc,
                 CAST(:meta AS jsonb))
        """), {"bid": book_id, "name": snap_name,
               "content": payload, "wc": grand_total,
               "meta": json.dumps(meta)})
        session.commit()
    return JSONResponse({
        "ok": True, "name": snap_name,
        "chapters_included": len(chapter_bundles),
        "total_words": grand_total,
        "meta": meta,
    })


@router.get("/api/snapshots/chapter/{chapter_id}")
async def list_chapter_snapshots(chapter_id: str):
    """List chapter-scope snapshots for a chapter."""
    with get_session() as session:
        rows = session.execute(text("""
            SELECT id::text, name, word_count, created_at, scope, meta
            FROM draft_snapshots
            WHERE chapter_id::text = :cid AND scope = 'chapter'
            ORDER BY created_at DESC
        """), {"cid": chapter_id}).fetchall()
    return {"snapshots": [
        {"id": r[0], "name": r[1], "word_count": r[2] or 0,
         "created_at": str(r[3]) if r[3] else "", "scope": r[4],
         "meta": (r[5] if isinstance(r[5], dict) else {}) or {}}
        for r in rows
    ]}


@router.get("/api/snapshots/book/{book_id}")
async def list_book_snapshots(book_id: str):
    """List book-scope snapshots for a book."""
    with get_session() as session:
        rows = session.execute(text("""
            SELECT id::text, name, word_count, created_at, scope, meta
            FROM draft_snapshots
            WHERE book_id::text = :bid AND scope = 'book'
            ORDER BY created_at DESC
        """), {"bid": book_id}).fetchall()
    return {"snapshots": [
        {"id": r[0], "name": r[1], "word_count": r[2] or 0,
         "created_at": str(r[3]) if r[3] else "", "scope": r[4],
         "meta": (r[5] if isinstance(r[5], dict) else {}) or {}}
        for r in rows
    ]}


def _restore_chapter_bundle(session, bundle: dict) -> int:
    """Insert a NEW draft version per section in the bundle.

    Non-destructive: the existing drafts stay put with their current
    versions; the restored bundle shows up as the newest version of
    each section (so the GUI's "latest version" resolver picks it).
    Returns the number of drafts created.
    """
    from uuid import uuid4 as _uuid4

    chapter_id = bundle.get("chapter_id")
    drafts = bundle.get("drafts") or []
    created = 0
    for d in drafts:
        section_type = d.get("section_type")
        if not section_type:
            continue
        row = session.execute(text(
            "SELECT COALESCE(MAX(version), 0) FROM drafts "
            "WHERE chapter_id::text = :cid AND section_type = :st"
        ), {"cid": chapter_id, "st": section_type}).fetchone()
        next_ver = int((row[0] if row else 0) or 0) + 1
        bk_row = session.execute(text(
            "SELECT book_id::text FROM drafts "
            "WHERE chapter_id::text = :cid LIMIT 1"
        ), {"cid": chapter_id}).fetchone()
        book_id = bk_row[0] if bk_row else None
        sources_json = json.dumps(d.get("sources") or [])
        restore_title = d.get("title") or f"Restored {section_type}"
        session.execute(text("""
            INSERT INTO drafts
                (id, title, book_id, chapter_id, section_type, topic,
                 content, word_count, sources, version, model_used,
                 custom_metadata)
            VALUES
                (:id, :title, CAST(:bid AS uuid), CAST(:cid AS uuid),
                 :st, :topic, :content, :wc, CAST(:sources AS jsonb),
                 :ver, :model, CAST(:meta AS jsonb))
        """), {
            "id": str(_uuid4()),
            "title": restore_title,
            "bid": book_id,
            "cid": chapter_id,
            "st": section_type,
            "topic": None,
            "content": d.get("content") or "",
            "wc": int(d.get("word_count") or 0),
            "sources": sources_json,
            "ver": next_ver,
            "model": "snapshot-restore",
            "meta": json.dumps({
                "checkpoint": "restored_from_snapshot",
                "restored_from_draft_id": d.get("id"),
                "restored_from_version": d.get("version"),
            }),
        })
        created += 1
    return created


@router.post("/api/snapshot/restore-bundle/{snapshot_id}")
async def restore_snapshot_bundle(snapshot_id: str):
    """Non-destructively restore a chapter/book bundle snapshot.

    Phase 38. For each section in the bundle, inserts a new `drafts`
    row at `version = max_current_version + 1`, so the restored
    content becomes the new latest. Existing rows are untouched,
    giving the user an undo path if the restore itself was wrong.
    """
    with get_session() as session:
        row = session.execute(text(
            "SELECT id::text, scope, content, chapter_id::text, "
            "       book_id::text, name "
            "FROM draft_snapshots WHERE id::text = :sid LIMIT 1"
        ), {"sid": snapshot_id}).fetchone()
        if not row:
            raise HTTPException(404, "Snapshot not found")
        _, scope, content, ch_id, bk_id, snap_name = row
        if scope not in ("chapter", "book"):
            raise HTTPException(
                400,
                f"Snapshot scope {scope!r} is not a bundle — use the "
                f"per-draft restore endpoint instead."
            )
        try:
            payload = json.loads(content)
        except Exception as exc:
            raise HTTPException(500, f"Malformed snapshot bundle: {exc}")

        total_drafts = 0
        chapters_touched = 0
        if scope == "chapter":
            total_drafts = _restore_chapter_bundle(session, payload)
            chapters_touched = 1
        else:  # book
            for ch_bundle in payload.get("chapters") or []:
                total_drafts += _restore_chapter_bundle(session, ch_bundle)
                chapters_touched += 1
        session.commit()

    return JSONResponse({
        "ok": True, "name": snap_name, "scope": scope,
        "chapters_restored": chapters_touched,
        "drafts_created": total_drafts,
    })


# ── Phase 54.6.328 (snapshot-versioning Phase 5) — Timeline endpoints ─────


@router.get("/api/timeline/section/{draft_id}")
async def timeline_section(draft_id: str):
    """Section-scope timeline: drafts.version chain + draft-scope
    snapshots, interleaved by created_at, briefs per row.
    """
    from sciknow.core.version_history import list_section_history
    with get_session() as session:
        entries = list_section_history(session, draft_id=draft_id)
    return {
        "scope": "section",
        "entries": [
            {
                "kind": e.kind, "id": e.id, "label": e.label,
                "created_at": e.created_at, "word_count": e.word_count,
                "version": e.version, "is_active": e.is_active,
                "meta": e.meta or {}, "extra": e.extra or {},
            }
            for e in entries
        ],
    }


@router.get("/api/timeline/chapter/{chapter_id}")
async def timeline_chapter(chapter_id: str):
    """Chapter-scope timeline: latest active draft per section
    + all chapter-scope snapshots in this chapter.
    """
    with get_session() as session:
        sec_rows = session.execute(text("""
            SELECT DISTINCT ON (section_type)
                id::text, section_type, version, word_count, created_at,
                COALESCE((custom_metadata->>'is_active')::boolean, FALSE) AS active
            FROM drafts
            WHERE chapter_id::text = :cid
            ORDER BY section_type,
                     COALESCE((custom_metadata->>'is_active')::boolean, FALSE) DESC,
                     CASE WHEN content IS NULL OR LENGTH(content) < 50
                          THEN 1 ELSE 0 END,
                     version DESC
        """), {"cid": chapter_id}).fetchall()
        snap_rows = session.execute(text("""
            SELECT id::text, name, word_count, created_at, meta
            FROM draft_snapshots
            WHERE chapter_id::text = :cid AND scope = 'chapter'
            ORDER BY created_at DESC
        """), {"cid": chapter_id}).fetchall()
    return {
        "scope": "chapter",
        "sections": [
            {
                "id": r[0], "section_type": r[1],
                "version": int(r[2] or 0), "word_count": int(r[3] or 0),
                "created_at": str(r[4] or ""), "is_active": bool(r[5]),
            }
            for r in sec_rows
        ],
        "snapshots": [
            {
                "id": r[0], "name": r[1], "word_count": int(r[2] or 0),
                "created_at": str(r[3] or ""),
                "meta": (r[4] if isinstance(r[4], dict) else {}) or {},
            }
            for r in snap_rows
        ],
    }


@router.get("/api/timeline/book/{book_id}")
async def timeline_book(book_id: str):
    """Book-scope timeline: all chapter + book snapshots in this book."""
    with get_session() as session:
        snap_rows = session.execute(text("""
            SELECT s.id::text, s.name, s.word_count, s.created_at,
                   s.scope, s.meta, bc.number, bc.title
            FROM draft_snapshots s
            LEFT JOIN book_chapters bc ON bc.id = s.chapter_id
            WHERE s.book_id::text = :bid
               OR s.chapter_id IN (
                   SELECT id FROM book_chapters WHERE book_id::text = :bid
               )
            ORDER BY s.created_at DESC
        """), {"bid": book_id}).fetchall()
    return {
        "scope": "book",
        "snapshots": [
            {
                "id": r[0], "name": r[1], "word_count": int(r[2] or 0),
                "created_at": str(r[3] or ""), "scope": r[4],
                "meta": (r[5] if isinstance(r[5], dict) else {}) or {},
                "chapter_number": r[6],
                "chapter_title": r[7] or "",
            }
            for r in snap_rows
        ],
    }


# ── Phase 55.V19f: delete snapshot ───────────────────────────────────────────


@router.delete("/api/snapshot/{snapshot_id}")
async def delete_snapshot(snapshot_id: str):
    """Delete a snapshot by id. Idempotent: 404 if it doesn't exist."""
    with get_session() as session:
        row = session.execute(
            text("DELETE FROM draft_snapshots WHERE id::text = :sid RETURNING id"),
            {"sid": snapshot_id},
        ).fetchone()
        session.commit()
    if not row:
        raise HTTPException(404, "Snapshot not found")
    return {"deleted": snapshot_id}


# ── Phase 55.V19f: auto-snapshot configuration (per book, persisted on
# the book.custom_metadata JSON) ─────────────────────────────────────────────


@router.get("/api/autosnapshot/{book_id}")
async def get_autosnapshot_config(book_id: str):
    """Read the auto-snapshot config for one book.

    Stored under ``books.custom_metadata->'auto_snapshot'`` so it lives
    with the rest of the book's editor preferences and survives across
    sessions without a new column.
    """
    with get_session() as session:
        row = session.execute(
            text("SELECT custom_metadata FROM books WHERE id::text = :bid"),
            {"bid": book_id},
        ).fetchone()
    if not row:
        raise HTTPException(404, "Book not found")
    md = row[0] if isinstance(row[0], dict) else {}
    cfg = md.get("auto_snapshot") if isinstance(md, dict) else None
    if not isinstance(cfg, dict):
        cfg = {
            "enabled": False,
            "after_autowrite": False,
            "after_save": False,
            "interval_minutes": 0,
            "scope": "section",
        }
    return cfg


@router.put("/api/autosnapshot/{book_id}")
async def set_autosnapshot_config(
    book_id: str,
    enabled: bool = Form(False),
    after_autowrite: bool = Form(False),
    after_save: bool = Form(False),
    interval_minutes: int = Form(0),
    scope: str = Form("section"),
):
    """Persist the auto-snapshot config. Validates scope ∈ {section, chapter, book}."""
    if scope not in ("section", "chapter", "book"):
        raise HTTPException(400, "scope must be section|chapter|book")
    cfg = {
        "enabled": bool(enabled),
        "after_autowrite": bool(after_autowrite),
        "after_save": bool(after_save),
        "interval_minutes": max(0, int(interval_minutes)),
        "scope": scope,
    }
    with get_session() as session:
        existing = session.execute(
            text("SELECT custom_metadata FROM books WHERE id::text = :bid"),
            {"bid": book_id},
        ).fetchone()
        if not existing:
            raise HTTPException(404, "Book not found")
        md = existing[0] if isinstance(existing[0], dict) else {}
        if not isinstance(md, dict):
            md = {}
        md["auto_snapshot"] = cfg
        session.execute(
            text("UPDATE books SET custom_metadata = CAST(:m AS jsonb) "
                 "WHERE id::text = :bid"),
            {"m": json.dumps(md), "bid": book_id},
        )
        session.commit()
    return cfg
