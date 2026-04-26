"""``sciknow.web.routes.ledger`` — GPU-time + token ledger endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. Each
handler reads from the per-book / per-chapter / per-draft
aggregations in `sciknow.core.gpu_ledger` against a fresh PG
session. No app.py cross-deps.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from sciknow.storage.db import get_session

router = APIRouter()


@router.get("/api/ledger/book/{book_id}")
async def api_ledger_book(book_id: str):
    """Phase 54.6.76 (#15) — GPU-time + token ledger for a book.

    Returns the book-level rollup plus a per-chapter breakdown
    (`{header: {...}, chapters: [...]}`). Empty `chapters` list means
    nothing was autowrite-generated in this book yet.
    """
    from sciknow.core import gpu_ledger
    with get_session() as session:
        header = gpu_ledger.ledger_for_book(session, book_id)
        if header is None:
            raise HTTPException(404, "Book not found")
        per_ch = gpu_ledger.ledger_per_chapter(session, book_id)
    return {
        "header": gpu_ledger.ledger_as_dict(header),
        "chapters": [gpu_ledger.ledger_as_dict(r) for r in per_ch],
    }


@router.get("/api/ledger/chapter/{chapter_id}")
async def api_ledger_chapter(chapter_id: str):
    """Phase 54.6.76 (#15) — ledger for one chapter with per-section rows."""
    from sciknow.core import gpu_ledger
    with get_session() as session:
        header = gpu_ledger.ledger_for_chapter(session, chapter_id)
        if header is None:
            raise HTTPException(404, "Chapter not found")
        per_sec = gpu_ledger.ledger_per_section(session, chapter_id)
    return {
        "header": gpu_ledger.ledger_as_dict(header),
        "sections": [gpu_ledger.ledger_as_dict(r) for r in per_sec],
    }


@router.get("/api/ledger/draft/{draft_id}")
async def api_ledger_draft(draft_id: str):
    """Phase 54.6.76 (#15) — ledger for one draft."""
    from sciknow.core import gpu_ledger
    with get_session() as session:
        row = gpu_ledger.ledger_for_draft(session, draft_id)
    if row is None:
        raise HTTPException(404, "Draft not found")
    return gpu_ledger.ledger_as_dict(row)
