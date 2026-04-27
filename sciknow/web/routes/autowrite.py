"""``sciknow.web.routes.autowrite`` — autowrite job-launcher endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. The 2 thin
job-launcher handlers wrap the autowrite engine generators
(`autowrite_section_stream`, `autowrite_chapter_all_sections_stream`)
in a background thread and return a job id for SSE streaming.

Cross-module deps (via standard lazy `_app` shim):
  - `_app._create_job` and `_app._run_generator_in_thread` (background runner)
  - `_app._book_id` (active-book id stamped on the running process)
"""
from __future__ import annotations

import asyncio
import threading

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/api/autowrite-book")
async def api_autowrite_book(
    max_iter: int = Form(3),
    target_score: float = Form(0.85),
    model: str = Form(None),
    target_words: int = Form(None),
    rebuild: bool = Form(False),
    resume: bool = Form(False),
    only_below_target: bool = Form(False),
    include_visuals: bool = Form(False),
):
    """Phase 54.6.x — autowrite EVERY section of EVERY chapter in the
    book in a single job.

    Wraps ``autowrite_book_all_chapters_stream`` (which itself iterates
    over ``autowrite_chapter_all_sections_stream``). The chapter-level
    wrapper already handles the pre-autowrite snapshot + the
    rebuild / resume semantics; this endpoint just kicks off one job
    that fans out across the whole book.
    """
    from sciknow.core.book_ops import autowrite_book_all_chapters_stream
    from sciknow.web import app as _app

    job_id, queue = _app._create_job("autowrite_book")
    loop = asyncio.get_event_loop()

    def gen():
        return autowrite_book_all_chapters_stream(
            book_id=_app._book_id,
            model=model or None,
            max_iter=max_iter, target_score=target_score,
            target_words=target_words if target_words and target_words > 0 else None,
            rebuild=rebuild,
            resume=resume,
            only_below_target=only_below_target,
            include_visuals=include_visuals,
        )

    thread = threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@router.post("/api/autowrite-chapter")
async def api_autowrite_chapter(
    chapter_id: str = Form(...),
    max_iter: int = Form(3),
    target_score: float = Form(0.85),
    model: str = Form(None),
    target_words: int = Form(None),
    rebuild: bool = Form(False),
    resume: bool = Form(False),
    only_below_target: bool = Form(False),
    include_visuals: bool = Form(False),
):
    """Phase 20 — autowrite EVERY section of a chapter in sequence.

    The toolbar Autowrite button routes here when the user has a
    chapter selected but no specific section, instead of defaulting
    to a single 'introduction' draft (which doesn't match any of the
    chapter's user-defined sections and creates an orphan).

    The backend generator handles the section iteration, draft skip
    logic, and per-section progress events. This endpoint just kicks
    it off as a job and returns the job_id for SSE streaming.
    """
    from sciknow.core.book_ops import autowrite_chapter_all_sections_stream
    from sciknow.web import app as _app

    job_id, queue = _app._create_job("autowrite_chapter")
    loop = asyncio.get_event_loop()

    def gen():
        return autowrite_chapter_all_sections_stream(
            book_id=_app._book_id, chapter_id=chapter_id,
            model=model or None,
            max_iter=max_iter, target_score=target_score,
            target_words=target_words if target_words and target_words > 0 else None,
            rebuild=rebuild,
            resume=resume,
            only_below_target=only_below_target,
            include_visuals=include_visuals,
        )

    thread = threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@router.post("/api/autowrite")
async def api_autowrite(
    chapter_id: str = Form(None),
    section_type: str = Form("introduction"),
    max_iter: int = Form(3),
    target_score: float = Form(0.85),
    full: bool = Form(False),
    model: str = Form(None),
    target_words: int = Form(None),
    include_visuals: bool = Form(False),
):
    """Phase 17 — target_words is optional; when None, the effective
    per-section target is resolved from the book's custom_metadata
    (target_chapter_words / num_sections_in_chapter). When set, it
    overrides the book-level value for this run only.

    Phase 54.6.144 — ``include_visuals`` turns on the 54.6.142 autowrite
    visuals integration: the 5-signal ranker surfaces figures to the
    writer, the gated instruction ships in the system prompt, and
    ``visual_citation`` joins the scorer dimensions. Default off so the
    existing workflow is untouched."""
    from sciknow.core.book_ops import autowrite_section_stream
    from sciknow.web import app as _app

    if full:
        if not chapter_id:
            return JSONResponse(
                {"error": "chapter_id required (full-book autowrite not yet supported from web)"},
                status_code=400,
            )

    job_id, queue = _app._create_job("autowrite")
    loop = asyncio.get_event_loop()

    def gen():
        return autowrite_section_stream(
            book_id=_app._book_id, chapter_id=chapter_id,
            section_type=section_type, model=model or None,
            max_iter=max_iter, target_score=target_score,
            target_words=target_words if target_words and target_words > 0 else None,
            include_visuals=include_visuals,
        )

    thread = threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})
