"""``sciknow.web.routes.export`` — export endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. Each
handler renders a draft / chapter / book in one of txt / md / html
/ pdf and returns the right Response subtype.

Heavy cross-module dep set — these routes lean on a lot of the
shared helpers in app.py:
  `_app._VALID_EXPORT_EXTS`, `_app._get_book_data`, `_app._draft_to_md`,
  `_app._strip_md`, `_app._draft_to_html_body`, `_app._wrap_html_export`,
  `_app._slugify_for_filename`, `_app._html_to_pdf_response`, `_app._esc`,
  `_app._ordered_chapter_drafts`.

Resolved via the standard lazy `_app` shim — by call-time the
parent module is fully loaded so the bindings exist on it.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse

router = APIRouter()


@router.get("/api/export/draft/{draft_id}.{ext}")
async def export_draft(draft_id: str, ext: str):
    """Phase 30/31 — export a single draft as txt/md/html/pdf."""
    from sciknow.web import app as _app
    if ext not in _app._VALID_EXPORT_EXTS:
        raise HTTPException(400, f"ext must be one of {_app._VALID_EXPORT_EXTS}")
    book, chapters, drafts, gaps, comments = _app._get_book_data()
    draft = next((d for d in drafts if d[0] == draft_id or d[0].startswith(draft_id)), None)
    if not draft:
        raise HTTPException(404, "Draft not found")
    md = _app._draft_to_md(draft)
    if ext == "md":
        return PlainTextResponse(md, media_type="text/markdown; charset=utf-8")
    if ext == "txt":
        return PlainTextResponse(_app._strip_md(md), media_type="text/plain; charset=utf-8")
    body = _app._draft_to_html_body(draft)
    html = _app._wrap_html_export(draft[1] or "Untitled", body)
    if ext == "pdf":
        slug = _app._slugify_for_filename(draft[1] or "draft") or "draft"
        return _app._html_to_pdf_response(html, f"{slug}.pdf")
    return HTMLResponse(html)


@router.get("/api/export/chapter/{chapter_id}.{ext}")
async def export_chapter(chapter_id: str, ext: str):
    """Phase 30/31 — export every drafted section in a chapter, ordered
    by the chapter's sections meta. Skips empty/orphan sections."""
    from sciknow.web import app as _app
    if ext not in _app._VALID_EXPORT_EXTS:
        raise HTTPException(400, f"ext must be one of {_app._VALID_EXPORT_EXTS}")
    book, chapters, drafts, gaps, comments = _app._get_book_data()
    ch = next((c for c in chapters if c[0] == chapter_id), None)
    if not ch:
        raise HTTPException(404, "Chapter not found")
    ch_num, ch_title = ch[1], ch[2]
    section_drafts = _app._ordered_chapter_drafts(drafts, chapter_id)
    if ext == "md":
        parts = [f"# Ch.{ch_num} {ch_title}\n"]
        for d in section_drafts:
            parts.append(_app._draft_to_md(d))
        return PlainTextResponse("\n\n".join(parts), media_type="text/markdown; charset=utf-8")
    if ext == "txt":
        parts = [f"Ch.{ch_num} {ch_title}\n{'=' * 40}\n"]
        for d in section_drafts:
            parts.append(_app._strip_md(_app._draft_to_md(d)))
        return PlainTextResponse("\n\n".join(parts), media_type="text/plain; charset=utf-8")
    body = (
        f"<h1>Ch.{ch_num} {_app._esc(ch_title or '')}</h1>"
        f"<div class='meta'>{len(section_drafts)} sections</div>"
    )
    for d in section_drafts:
        body += _app._draft_to_html_body(d)
    html = _app._wrap_html_export(f"Ch.{ch_num} {ch_title}", body)
    if ext == "pdf":
        slug = _app._slugify_for_filename(ch_title or "chapter") or "chapter"
        return _app._html_to_pdf_response(html, f"ch{ch_num}_{slug}.pdf")
    return HTMLResponse(html)


@router.get("/api/export/book.{ext}")
async def export_book(ext: str):
    """Phase 30/31 — export the whole book as one file. Iterates
    chapters in order, then sections in each chapter's defined order."""
    from sciknow.web import app as _app
    if ext not in _app._VALID_EXPORT_EXTS:
        raise HTTPException(400, f"ext must be one of {_app._VALID_EXPORT_EXTS}")
    book, chapters, drafts, gaps, comments = _app._get_book_data()
    book_title = book[1] if book else "Untitled book"
    if ext == "md":
        parts = [f"# {book_title}\n"]
        for ch in chapters:
            parts.append(f"\n## Ch.{ch[1]} {ch[2]}\n")
            for d in _app._ordered_chapter_drafts(drafts, ch[0]):
                parts.append(_app._draft_to_md(d, include_sources=False))
        all_sources = []
        seen = set()
        for d in drafts:
            srcs = d[5]
            if isinstance(srcs, str):
                try:
                    srcs = json.loads(srcs)
                except Exception:
                    srcs = []
            for s in srcs or []:
                if s and s not in seen:
                    seen.add(s)
                    all_sources.append(s)
        if all_sources:
            parts.append("\n\n## Bibliography\n\n")
            for i, s in enumerate(all_sources, start=1):
                parts.append(f"{i}. {s}\n")
        return PlainTextResponse("\n".join(parts), media_type="text/markdown; charset=utf-8")
    if ext == "txt":
        parts = [f"{book_title}\n{'=' * len(book_title)}\n"]
        for ch in chapters:
            parts.append(f"\nCh.{ch[1]} {ch[2]}\n{'-' * 40}\n")
            for d in _app._ordered_chapter_drafts(drafts, ch[0]):
                parts.append(_app._strip_md(_app._draft_to_md(d, include_sources=False)))
        return PlainTextResponse("\n".join(parts), media_type="text/plain; charset=utf-8")
    body = f"<h1>{_app._esc(book_title)}</h1><div class='meta'>{len(chapters)} chapters</div>"
    for ch in chapters:
        body += f"<h1>Ch.{ch[1]} {_app._esc(ch[2] or '')}</h1>"
        for d in _app._ordered_chapter_drafts(drafts, ch[0]):
            body += _app._draft_to_html_body(d)
    html = _app._wrap_html_export(book_title, body)
    if ext == "pdf":
        slug = _app._slugify_for_filename(book_title) or "book"
        return _app._html_to_pdf_response(html, f"{slug}.pdf")
    return HTMLResponse(html)
