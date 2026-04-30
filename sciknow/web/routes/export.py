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

Phase 55 (LaTeX export) — adds a parallel /api/export/pro/* family
of endpoints that drives the new ``sciknow.formatting`` pipeline
(IR → Jinja2 → latexmk). Uses a tiny in-process job pool because
compiles can take 30+ seconds on big books and we don't want the
HTTP request to hang.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

router = APIRouter()
log = logging.getLogger(__name__)


# ── /api/export/pro/* — LaTeX → PDF jobs ───────────────────────────


@dataclass
class _ProExportJob:
    job_id: str
    status: str = "pending"           # pending / running / done / error
    started_at: float = 0.0
    finished_at: float = 0.0
    progress: str = ""                 # human-readable line for the UI
    error: str = ""
    log_text: str = ""
    output_bytes: bytes = b""
    output_filename: str = ""
    output_mimetype: str = ""
    saved_to: str = ""           # absolute path on the server, if save_to_path was set
    save_error: str = ""         # populated when save_to_path failed
    options_snapshot: dict = field(default_factory=dict)


_pro_export_jobs: dict[str, _ProExportJob] = {}
_pro_export_lock = threading.Lock()
_PRO_EXPORT_TTL_SECONDS = 60 * 30


def _gc_pro_export_jobs() -> None:
    """Drop jobs older than the TTL so the dict doesn't grow unbounded."""
    now = time.monotonic()
    with _pro_export_lock:
        stale = [
            jid for jid, j in _pro_export_jobs.items()
            if j.finished_at and (now - j.finished_at) > _PRO_EXPORT_TTL_SECONDS
        ]
        for jid in stale:
            _pro_export_jobs.pop(jid, None)


class ProExportRequest(BaseModel):
    """User-controllable knobs for /api/export/pro/build.

    Mirrors ``sciknow.formatting.options.ExportOptions`` with friendly
    JSON-shaped types so the form in the GUI can serialise directly.
    """
    fmt: str = Field(default="pdf", description="'pdf' or 'tex-bundle'")
    template_slug: Optional[str] = None
    font_family: str = "lmodern"
    font_size_pt: int = 11
    paper: str = "a4paper"
    two_column: bool = False
    bib_style: str = "numeric"
    bibliography_placement: str = "book"   # "book" | "chapter"
    # Phase 55.V19g — optional server-side save path. When set, the
    # built file is also written to this folder (in addition to being
    # returned via the /download endpoint). Path is expanded through
    # os.path.expanduser so "~" works.
    save_to_path: Optional[str] = None
    cover_page: bool = True
    table_of_contents: bool = True
    list_of_figures: bool = False
    list_of_tables: bool = False
    abstract_override: Optional[str] = None
    author_override: Optional[str] = None
    affiliation: Optional[str] = None
    dedication: Optional[str] = None
    acknowledgements: Optional[str] = None
    toc_depth: int = 2
    include_bibliography_chapter: bool = True


def _build_export_options(req: ProExportRequest):
    from sciknow.formatting.options import ExportOptions
    return ExportOptions(
        template_slug=req.template_slug,
        font_family=req.font_family,
        font_size_pt=req.font_size_pt,
        paper=req.paper if req.paper in ("a4paper", "letterpaper") else "a4paper",
        two_column=req.two_column,
        bib_style=req.bib_style,
        bibliography_placement=(req.bibliography_placement
                                if req.bibliography_placement in ("book", "chapter")
                                else "book"),
        cover_page=req.cover_page,
        table_of_contents=req.table_of_contents,
        list_of_figures=req.list_of_figures,
        list_of_tables=req.list_of_tables,
        abstract_override=req.abstract_override or None,
        author_override=req.author_override or None,
        affiliation=req.affiliation or None,
        dedication=req.dedication or None,
        acknowledgements=req.acknowledgements or None,
        toc_depth=req.toc_depth,
        include_bibliography_chapter=req.include_bibliography_chapter,
    )


def _run_pro_export(job_id: str, book_id: str, req: ProExportRequest) -> None:
    """Worker thread for /api/export/pro/build."""
    job = _pro_export_jobs[job_id]
    job.status = "running"
    job.started_at = time.monotonic()
    job.progress = "loading book data"
    try:
        from sciknow.formatting import build_book_pdf, build_book_tex_bundle
        from sciknow.formatting.compile import LatexCompileError
        opts = _build_export_options(req)
        if req.fmt == "tex-bundle":
            job.progress = "rendering LaTeX bundle"
            data, _tex = build_book_tex_bundle(book_id, opts)
            job.output_bytes = data
            job.output_mimetype = "application/zip"
            job.output_filename = f"{book_id[:8]}-tex-bundle.zip"
        else:
            job.progress = "compiling with latexmk"
            pdf, log_text, _tex = build_book_pdf(book_id, opts)
            job.output_bytes = pdf
            job.output_mimetype = "application/pdf"
            job.output_filename = f"{book_id[:8]}.pdf"
            job.log_text = log_text
        # Phase 55.V19g — also write to server-side save path if requested.
        if req.save_to_path:
            try:
                import os, pathlib
                target_dir = pathlib.Path(os.path.expanduser(req.save_to_path)).resolve()
                target_dir.mkdir(parents=True, exist_ok=True)
                target_file = target_dir / job.output_filename
                target_file.write_bytes(job.output_bytes)
                job.saved_to = str(target_file)
            except Exception as save_exc:
                log.warning("save_to_path failed for %s: %s",
                            req.save_to_path, save_exc)
                job.save_error = f"{type(save_exc).__name__}: {save_exc}"
        job.status = "done"
        job.progress = "done"
    except LatexCompileError as e:
        job.status = "error"
        job.error = str(e)
        job.log_text = getattr(e, "log_text", "") or ""
    except Exception as e:
        log.exception("pro export failed for book %s", book_id)
        job.status = "error"
        job.error = f"{type(e).__name__}: {e}"
        job.log_text = traceback.format_exc()
    finally:
        job.finished_at = time.monotonic()


@router.post("/api/export/pro/build")
async def export_pro_build(req: ProExportRequest):
    """Kick off a LaTeX-compile job for the active book.

    Returns immediately with ``{"job_id": "..."}``. Poll
    ``/api/export/pro/job/{id}`` for status and download with
    ``/api/export/pro/job/{id}/download`` once status is ``done``.
    """
    from sciknow.web import app as _app
    if not getattr(_app, "_book_id", "") :
        raise HTTPException(400, "no active book; open one first")

    _gc_pro_export_jobs()

    job_id = uuid.uuid4().hex[:16]
    snap = req.model_dump()
    job = _ProExportJob(job_id=job_id, options_snapshot=snap)
    with _pro_export_lock:
        _pro_export_jobs[job_id] = job

    t = threading.Thread(
        target=_run_pro_export,
        args=(job_id, _app._book_id, req),
        daemon=True,
        name=f"pro-export-{job_id}",
    )
    t.start()
    return {"job_id": job_id, "status": "pending"}


@router.get("/api/export/pro/job/{job_id}")
async def export_pro_job_status(job_id: str):
    job = _pro_export_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
        "log_excerpt": (job.log_text[-2000:] if job.log_text else ""),
        "filename": job.output_filename,
        "saved_to": job.saved_to,
        "save_error": job.save_error,
        "size": len(job.output_bytes),
        "elapsed_seconds": (
            (job.finished_at or time.monotonic()) - job.started_at
            if job.started_at else 0.0
        ),
    }


@router.get("/api/export/pro/job/{job_id}/download")
async def export_pro_job_download(job_id: str):
    job = _pro_export_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status != "done":
        raise HTTPException(409, f"job not ready (status={job.status})")
    return Response(
        content=job.output_bytes,
        media_type=job.output_mimetype,
        headers={
            "Content-Disposition": f'attachment; filename="{job.output_filename}"',
        },
    )


@router.get("/api/export/pro/templates")
async def export_pro_templates():
    """List available templates per project type so the GUI dropdown
    stays in sync with the registry."""
    from sciknow.core.project_type import PROJECT_TYPES
    from sciknow.formatting.options import (
        BIB_STYLES, FONT_FAMILIES, list_templates_for_type,
    )
    out = {
        "by_project_type": {
            slug: [
                {"slug": t.slug, "name": t.display_name,
                 "description": t.description}
                for t in list_templates_for_type(slug)
            ]
            for slug in PROJECT_TYPES
        },
        "fonts": [
            {"slug": k, "name": v[0]} for k, v in FONT_FAMILIES.items()
        ],
        "bib_styles": [
            {"slug": k, "name": v[0]} for k, v in BIB_STYLES.items()
        ],
    }
    return out


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
