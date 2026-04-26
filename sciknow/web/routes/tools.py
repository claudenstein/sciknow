"""``sciknow.web.routes.tools`` — Tools panel endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. Groups the
4 Q&A / search handlers wired to the Tools panel:

  POST /api/ask              — corpus-wide RAG (SSE stream)
  POST /api/ask/synthesize   — multi-paper synthesis (SSE stream)
  POST /api/search/query     — hybrid search (JSON)
  POST /api/search/similar   — nearest-neighbour by abstract (JSON)

The two SSE handlers spawn a background generator via
`_app._run_generator_in_thread` from app.py and tag it with `_app._create_job`;
both resolved via the standard lazy `_app` shim.
"""
from __future__ import annotations

import asyncio
import threading

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

router = APIRouter()


@router.post("/api/ask")
async def api_ask(
    question: str = Form(...),
    model: str = Form(None),
    context_k: int = Form(8),
    year_from: int = Form(None),
    year_to: int = Form(None),
):
    """Stream a corpus-wide RAG question — full hybrid search + LLM stream."""
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.storage.qdrant import get_client
    from sciknow.web import app as _app

    job_id, queue = _app._create_job("ask")
    loop = asyncio.get_event_loop()

    def gen():
        qdrant = get_client()
        yield {"type": "progress", "stage": "retrieving",
               "detail": "Hybrid search across the corpus..."}
        with get_session() as session:
            candidates = hybrid_search.search(
                query=question, qdrant_client=qdrant, session=session,
                candidate_k=50,
                year_from=year_from if year_from else None,
                year_to=year_to if year_to else None,
            )
            if not candidates:
                yield {"type": "error", "message": "No relevant passages found."}
                return
            candidates = reranker.rerank(question, candidates, top_k=context_k)
            results = context_builder.build(candidates, session)

        from sciknow.rag.prompts import format_sources
        sources = format_sources(results).splitlines()
        yield {"type": "sources", "sources": sources, "n": len(sources)}

        system, user = rag_prompts.qa(question, results)
        yield {"type": "progress", "stage": "generating",
               "detail": f"Generating answer from {len(results)} passages..."}
        for tok in llm_stream(system, user, model=model or None):
            yield {"type": "token", "text": tok}
        yield {"type": "completed"}

    threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


@router.post("/api/search/query")
async def api_search_query(
    q: str = Form(...),
    top_k: int = Form(10),
    candidate_k: int = Form(50),
    no_rerank: bool = Form(False),
    year_from: int = Form(None),
    year_to: int = Form(None),
    section: str = Form(None),
    topic: str = Form(None),
    expand: bool = Form(False),
):
    """Hybrid corpus search (sciknow search query). JSON response."""
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.storage.qdrant import get_client

    qdrant = get_client()
    with get_session() as session:
        candidates = hybrid_search.search(
            query=q, qdrant_client=qdrant, session=session,
            candidate_k=candidate_k,
            year_from=year_from, year_to=year_to,
            section=section, topic_cluster=topic,
            use_query_expansion=expand,
        )
        if not candidates:
            return JSONResponse({"results": [], "n": 0})
        if not no_rerank:
            candidates = reranker.rerank(q, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]
        results = context_builder.build(candidates, session)

    import re as _re
    out = []
    for r in results:
        preview = _re.sub(r"<[^>]+>", "", r.content or "")
        preview = _re.sub(r"\s+", " ", preview).strip()[:300]
        out.append({
            "rank": r.rank,
            "title": r.title or "(untitled)",
            "year": r.year,
            "authors": r.authors or [],
            "journal": r.journal,
            "doi": r.doi,
            "section_type": r.section_type,
            "section_title": r.section_title,
            "score": r.score,
            "preview": preview,
            "document_id": str(r.document_id) if r.document_id else None,
        })
    return JSONResponse({"results": out, "n": len(out)})


@router.post("/api/search/similar")
async def api_search_similar(
    identifier: str = Form(...),
    top_k: int = Form(10),
):
    """Nearest-neighbour paper search in the abstracts collection."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    from sciknow.storage.qdrant import ABSTRACTS_COLLECTION, get_client

    qdrant = get_client()
    ident = (identifier or "").strip()
    if not ident:
        raise HTTPException(400, "identifier required")

    with get_session() as session:
        row = session.execute(text(
            "SELECT d.id::text, pm.title, pm.doi, pm.year "
            "FROM paper_metadata pm JOIN documents d ON d.id = pm.document_id "
            "WHERE LOWER(pm.doi) = LOWER(:q) OR LOWER(pm.arxiv_id) = LOWER(:q) "
            "LIMIT 1"
        ), {"q": ident}).first()
        if not row:
            row = session.execute(text(
                "SELECT d.id::text, pm.title, pm.doi, pm.year "
                "FROM paper_metadata pm JOIN documents d ON d.id = pm.document_id "
                "WHERE pm.title ILIKE :pattern "
                "ORDER BY pm.year DESC NULLS LAST LIMIT 1"
            ), {"pattern": f"%{ident}%"}).first()
        if not row:
            try:
                row = session.execute(text(
                    "SELECT d.id::text, pm.title, pm.doi, pm.year "
                    "FROM paper_metadata pm JOIN documents d ON d.id = pm.document_id "
                    "WHERE d.id::text = :q LIMIT 1"
                ), {"q": ident}).first()
            except Exception:
                row = None

    if not row:
        return JSONResponse({"error": "Paper not found", "query": ident}, status_code=404)

    doc_id, title, doi, year = row
    abstract_points = qdrant.scroll(
        collection_name=ABSTRACTS_COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=doc_id))
        ]),
        with_vectors=["dense"],
        limit=1,
    )[0]
    if not abstract_points:
        return JSONResponse({
            "error": "No abstract embedding for this paper (was it ingested?)",
            "query": ident, "document_id": doc_id, "title": title,
        }, status_code=404)

    query_vec = abstract_points[0].vector
    if isinstance(query_vec, dict):
        query_vec = query_vec.get("dense")

    hits = qdrant.query_points(
        collection_name=ABSTRACTS_COLLECTION,
        query=query_vec, using="dense",
        limit=top_k + 1, with_payload=True,
    )

    results = []
    for point in hits.points:
        payload = point.payload or {}
        if payload.get("document_id") == doc_id:
            continue
        results.append({
            "title": payload.get("title") or (payload.get("content_preview") or "")[:80],
            "year": payload.get("year"),
            "authors": payload.get("authors") or [],
            "document_id": payload.get("document_id"),
            "score": float(point.score) if point.score is not None else None,
        })
        if len(results) >= top_k:
            break

    return JSONResponse({
        "query": {"title": title, "year": year, "doi": doi, "document_id": doc_id},
        "results": results, "n": len(results),
    })


@router.post("/api/ask/synthesize")
async def api_ask_synthesize(
    topic: str = Form(...),
    model: str = Form(None),
    context_k: int = Form(12),
    year_from: int = Form(None),
    year_to: int = Form(None),
    domain: str = Form(None),
    topic_filter: str = Form(None),
):
    """Multi-paper synthesis on a topic (sciknow ask synthesize). SSE."""
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.storage.qdrant import get_client
    from sciknow.web import app as _app

    job_id, _queue = _app._create_job("synthesize")
    loop = asyncio.get_event_loop()

    def gen():
        qdrant = get_client()
        yield {"type": "progress", "stage": "retrieving",
               "detail": f"Retrieving passages for: {topic}..."}
        with get_session() as session:
            candidates = hybrid_search.search(
                query=topic, qdrant_client=qdrant, session=session,
                candidate_k=50,
                year_from=year_from if year_from else None,
                year_to=year_to if year_to else None,
                domain=domain or None,
                topic_cluster=topic_filter or None,
            )
            if not candidates:
                yield {"type": "error", "message": "No relevant passages found."}
                return
            candidates = reranker.rerank(topic, candidates, top_k=context_k)
            results = context_builder.build(candidates, session)

        from sciknow.rag.prompts import format_sources
        sources = format_sources(results).splitlines()
        yield {"type": "sources", "sources": sources, "n": len(sources)}

        system, user = rag_prompts.synthesis(topic, results)
        yield {"type": "progress", "stage": "generating",
               "detail": f"Synthesising from {len(results)} passages..."}
        for tok in llm_stream(system, user, model=model or None):
            yield {"type": "token", "text": tok}
        yield {"type": "completed"}

    threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})
