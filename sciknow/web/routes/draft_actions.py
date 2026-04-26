"""``sciknow.web.routes.draft_actions`` — draft_actions endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. Behaviour
unchanged. Cross-module deps resolved via the standard lazy
`from sciknow.web import app as _app` shim.
"""
from __future__ import annotations

import asyncio
import json
import threading

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response, StreamingResponse, PlainTextResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

router = APIRouter()


@router.post("/api/write")
async def api_write(
    chapter_id: str = Form(...),
    section_type: str = Form("introduction"),
    model: str = Form(None),
    target_words: int = Form(None),
):
    """Start a write operation, returns job_id for SSE streaming.

    Phase 17 — target_words is optional; when None, book_ops resolves
    it from the book's custom_metadata.target_chapter_words (or the
    default) divided by the chapter's section count.
    """
    from sciknow.web import app as _app
    from sciknow.core.book_ops import write_section_stream

    job_id, queue = _app._create_job("write")
    loop = asyncio.get_event_loop()

    def gen():
        return write_section_stream(
            book_id=_app._book_id, chapter_id=chapter_id,
            section_type=section_type, model=model or None,
            target_words=target_words if target_words and target_words > 0 else None,
        )

    thread = threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@router.post("/api/review/{draft_id}")
async def api_review(draft_id: str, model: str = Form(None)):
    from sciknow.web import app as _app
    from sciknow.core.book_ops import review_draft_stream

    job_id, queue = _app._create_job("review")
    loop = asyncio.get_event_loop()

    def gen():
        return review_draft_stream(draft_id, model=model or None)

    thread = threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@router.post("/api/revise/{draft_id}")
async def api_revise(
    draft_id: str,
    instruction: str = Form(""),
    model: str = Form(None),
):
    from sciknow.web import app as _app
    from sciknow.core.book_ops import revise_draft_stream

    job_id, queue = _app._create_job("revise")
    loop = asyncio.get_event_loop()

    def gen():
        return revise_draft_stream(
            draft_id, instruction=instruction, model=model or None)

    thread = threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@router.post("/api/gaps")
async def api_gaps(model: str = Form(None), method: str = Form("")):
    """Phase 54.6.14 — optional ``method`` name from the brainstorming
    catalogue (e.g. "Reverse Brainstorming", "Five Whys", "Scope
    Boundaries") steers the LLM's gap-finding approach."""
    from sciknow.web import app as _app
    from sciknow.core.book_ops import run_gaps_stream
    from sciknow.core.methods import get_method, method_preamble as _mp

    job_id, queue = _app._create_job("gaps")
    loop = asyncio.get_event_loop()

    preamble = ""
    if method and method.strip():
        m = get_method("brainstorming", method)
        if m:
            preamble = _mp(m)

    def gen():
        return run_gaps_stream(
            book_id=_app._book_id, model=model or None,
            method_preamble=preamble,
        )

    thread = threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@router.post("/api/argue")
async def api_argue(
    claim: str = Form(...),
    model: str = Form(None),
):
    from sciknow.web import app as _app
    from sciknow.core.book_ops import run_argue_stream

    job_id, queue = _app._create_job("argue")
    loop = asyncio.get_event_loop()

    def gen():
        return run_argue_stream(claim, book_id=_app._book_id, model=model or None, save=True)

    thread = threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@router.post("/api/verify/{draft_id}")
async def api_verify(draft_id: str, model: str = Form(None)):
    """Run claim verification on a draft via SSE."""
    from sciknow.web import app as _app
    from sciknow.core.book_ops import review_draft_stream

    # We reuse the review infrastructure but with a verify-specific generator
    job_id, queue = _app._create_job("verify")
    loop = asyncio.get_event_loop()

    def gen():
        """Verify claims — uses the verify_claims prompt from book_ops."""
        from sciknow.rag import prompts as rag_prompts
        from sciknow.rag.llm import complete as llm_complete
        from sciknow.storage.db import get_session
        from sciknow.storage.qdrant import get_client
        from sciknow.core.book_ops import _retrieve, _clean_json
        import json as _json

        with get_session() as session:
            from sqlalchemy import text as sql_text
            row = session.execute(sql_text("""
                SELECT d.id::text, d.title, d.section_type, d.topic, d.content
                FROM drafts d WHERE d.id::text LIKE :q LIMIT 1
            """), {"q": f"{draft_id}%"}).fetchone()

        if not row:
            yield {"type": "error", "message": f"Draft not found: {draft_id}"}
            return

        d_id, d_title, d_section, d_topic, d_content = row

        yield {"type": "progress", "stage": "retrieval", "detail": "Retrieving source passages..."}

        qdrant = get_client()
        search_query = f"{d_section or ''} {d_topic or d_title}"
        with get_session() as session:
            results, _ = _retrieve(session, qdrant, search_query, context_k=12)

        yield {"type": "progress", "stage": "verifying", "detail": "Verifying claims..."}

        sys_v, usr_v = rag_prompts.verify_claims(d_content, results)
        try:
            raw = llm_complete(sys_v, usr_v, model=model or None, temperature=0.0, num_ctx=16384)
            vdata = _json.loads(_clean_json(raw), strict=False)
            yield {"type": "verification", "data": vdata}
        except Exception as exc:
            yield {"type": "error", "message": f"Verification failed: {exc}"}

        yield {"type": "completed", "draft_id": d_id}

    thread = threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@router.post("/api/insert-citations/{draft_id}")
async def api_insert_citations(
    draft_id: str,
    model: str = Form(None),
    candidate_k: int = Form(8),
    max_needs: int = Form(0),
    dry_run: bool = Form(False),
):
    """Phase 46.A — auditable [N]-citation insertion over a saved draft.

    Two-pass LLM flow wrapped by ``book_ops.insert_citations_stream``: pass
    1 finds locations where a citation is needed, pass 2 retrieves top-K
    candidates via hybrid search and picks (or rejects) per claim. Events
    are streamed over the usual ``/api/stream/{job_id}`` SSE channel.
    """
    from sciknow.web import app as _app
    from sciknow.core.book_ops import insert_citations_stream

    job_id, queue = _app._create_job("insert_citations")
    loop = asyncio.get_event_loop()

    def gen():
        return insert_citations_stream(
            draft_id,
            model=(model or None),
            candidate_k=max(1, int(candidate_k)),
            max_needs=(int(max_needs) if max_needs and max_needs > 0 else None),
            dry_run=bool(dry_run),
            save=True,
        )

    thread = threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@router.post("/api/adversarial-review/{draft_id}")
async def api_adversarial_review(draft_id: str, model: str = Form(None)):
    """BMAD-inspired cynical critic pass — streams findings over SSE."""
    from sciknow.web import app as _app
    from sciknow.core.book_ops import adversarial_review_stream
    job_id, _queue = _app._create_job("adversarial_review")
    loop = asyncio.get_event_loop()

    def gen():
        return adversarial_review_stream(draft_id, model=(model or None))

    threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


@router.post("/api/edge-cases/{draft_id}")
async def api_edge_cases(draft_id: str, model: str = Form(None)):
    """Exhaustive edge-case hunter — structured JSON findings over SSE."""
    from sciknow.web import app as _app
    from sciknow.core.book_ops import edge_case_hunter_stream
    job_id, _queue = _app._create_job("edge_case_hunter")
    loop = asyncio.get_event_loop()

    def gen():
        return edge_case_hunter_stream(draft_id, model=(model or None))

    threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})
