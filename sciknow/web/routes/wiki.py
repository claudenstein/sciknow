"""``sciknow.web.routes.wiki`` — wiki page + KG endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. 13 handlers
covering wiki page browse, ask, annotation, query, lint, consensus,
extract-kg, and compile.

Cross-module deps (via lazy `_app` shim):
  - `_create_job`, `_run_generator_in_thread`, `_spawn_cli_streaming`
    for the SSE-streamed wrappers.
  - `_md_to_html` for the wiki-page HTML render.
"""
from __future__ import annotations

import asyncio
import threading

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

router = APIRouter()


@router.get("/api/wiki/pages")
async def api_wiki_pages(page_type: str = None, page: int = 1, per_page: int = 50):
    """Phase 15 — paginated wiki page list with optional type filter."""
    from sciknow.core.wiki_ops import list_pages
    try:
        all_pages = list_pages(page_type=page_type or None)
    except Exception as exc:
        return JSONResponse({"page": 1, "per_page": per_page, "total": 0,
                             "n_pages": 0, "pages": [], "available_types": [],
                             "error": str(exc)})

    page = max(page, 1)
    per_page = min(max(per_page, 1), 200)
    total = len(all_pages)
    start = (page - 1) * per_page
    end = start + per_page

    available_types = sorted({p["page_type"] for p in all_pages if p.get("page_type")})

    return JSONResponse({
        "page": page, "per_page": per_page, "total": total,
        "n_pages": (total + per_page - 1) // per_page,
        "pages": all_pages[start:end],
        "available_types": available_types,
    })


@router.get("/api/wiki/page/{slug}/annotation")
async def api_wiki_annotation_get(slug: str):
    """Phase 54.5 — fetch the user's "My take" annotation for a page."""
    with get_session() as session:
        row = session.execute(text("""
            SELECT body, updated_at FROM wiki_annotations WHERE slug = :s
        """), {"s": slug}).fetchone()
    if not row:
        return JSONResponse({"slug": slug, "body": "", "updated_at": None})
    return JSONResponse({
        "slug": slug, "body": row[0] or "", "updated_at": str(row[1]),
    })


@router.put("/api/wiki/page/{slug}/annotation")
async def api_wiki_annotation_put(slug: str, body: str = Form("")):
    """Phase 54.5 — upsert the annotation. Empty body deletes."""
    body = (body or "").strip()
    with get_session() as session:
        if not body:
            session.execute(text(
                "DELETE FROM wiki_annotations WHERE slug = :s"
            ), {"s": slug})
            session.commit()
            return JSONResponse({"slug": slug, "deleted": True})
        session.execute(text("""
            INSERT INTO wiki_annotations (slug, body, updated_at)
            VALUES (:s, :b, now())
            ON CONFLICT (slug) DO UPDATE
                SET body = EXCLUDED.body, updated_at = now()
        """), {"s": slug, "b": body[:20000]})
        session.commit()
        row = session.execute(text(
            "SELECT updated_at FROM wiki_annotations WHERE slug = :s"
        ), {"s": slug}).fetchone()
    return JSONResponse({
        "slug": slug, "deleted": False,
        "updated_at": str(row[0]) if row else None,
    })


@router.post("/api/wiki/page/{slug}/ask")
async def api_wiki_page_ask(
    slug: str,
    question: str = Form(...),
    model: str = Form(None),
    context_k: int = Form(6),
    broaden: bool = Form(False),
):
    """Phase 54.3 — "Ask this page" inline RAG."""
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.storage.qdrant import get_client
    from sciknow.web import app as _app

    with get_session() as session:
        row = session.execute(text("""
            SELECT array_remove(source_doc_ids, NULL)::text[]
            FROM wiki_pages WHERE slug = :s
        """), {"s": slug}).fetchone()
    source_doc_ids: list[str] = list(row[0]) if row and row[0] else []
    source_set = {d.lower() for d in source_doc_ids}

    job_id, queue = _app._create_job("wiki_page_ask")
    loop = asyncio.get_event_loop()

    def gen():
        qdrant = get_client()
        yield {"type": "progress", "stage": "retrieving",
               "detail": (
                   f"Searching this page's {len(source_doc_ids)} source paper(s)..."
                   if source_doc_ids and not broaden
                   else "Searching the whole corpus..."
               )}

        over_k = 200 if (source_set and not broaden) else 50
        with get_session() as session:
            candidates = hybrid_search.search(
                query=question, qdrant_client=qdrant, session=session,
                candidate_k=over_k,
            )
            if source_set and not broaden:
                candidates = [
                    c for c in candidates
                    if (c.document_id or "").lower() in source_set
                ]
            if not candidates:
                if source_set and not broaden:
                    yield {
                        "type": "error",
                        "message": (
                            "No matching passages in this page's source papers. "
                            "Try the 'broaden to full corpus' toggle."
                        ),
                    }
                else:
                    yield {"type": "error",
                           "message": "No relevant passages found."}
                return
            candidates = reranker.rerank(
                question, candidates, top_k=max(1, min(int(context_k or 6), 12)),
            )
            results = context_builder.build(candidates, session)

        from sciknow.rag.prompts import format_sources
        sources_lines = format_sources(results).splitlines()
        yield {"type": "sources", "sources": sources_lines,
               "n": len(sources_lines),
               "scope": "this-page" if (source_set and not broaden) else "corpus"}

        system, user = rag_prompts.qa(question, results)
        yield {"type": "progress", "stage": "generating",
               "detail": f"Generating answer from {len(results)} passage(s)..."}
        for tok in llm_stream(system, user, model=model or None):
            yield {"type": "token", "text": tok}
        yield {"type": "completed"}

    threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id, "scope_size": len(source_doc_ids)})


@router.get("/api/wiki/page/{slug}/backlinks")
async def api_wiki_backlinks(slug: str):
    """Phase 54.2 — pages that link to this one via ``[[slug]]``."""
    from sciknow.core.wiki_ops import get_backlinks_for
    try:
        return JSONResponse(get_backlinks_for(slug))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/wiki/page/{slug}/related")
async def api_wiki_related(slug: str, limit: int = 5):
    """Phase 54.2 — top-N pages nearest in the WIKI_COLLECTION embedding space."""
    from sciknow.core.wiki_ops import get_related_pages
    limit = max(1, min(int(limit or 5), 20))
    try:
        return JSONResponse(get_related_pages(slug, limit=limit))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/wiki/titles")
async def api_wiki_titles():
    """Phase 54 — compact title/slug index for the Ctrl-K command palette."""
    with get_session() as session:
        rows = session.execute(text("""
            SELECT slug, title, page_type
            FROM wiki_pages
            ORDER BY title
        """)).fetchall()
    return JSONResponse([
        {"slug": r[0], "title": r[1] or r[0], "page_type": r[2]}
        for r in rows
    ])


@router.get("/api/wiki/page/{slug}")
async def api_wiki_page(slug: str):
    """Phase 15 — return one wiki page's full content + metadata."""
    from sciknow.core.wiki_ops import show_page
    from sciknow.web import app as _app

    page = show_page(slug)
    if not page:
        raise HTTPException(404, f"Wiki page not found: {slug}")

    try:
        with get_session() as session:
            row = session.execute(text("""
                SELECT title, page_type, word_count,
                       array_length(source_doc_ids, 1) AS n_sources,
                       updated_at, needs_rewrite
                FROM wiki_pages WHERE slug = :slug
            """), {"slug": slug}).fetchone()
        if row:
            page.update({
                "title": row[0], "page_type": row[1], "word_count": row[2] or 0,
                "n_sources": row[3] or 0, "updated_at": str(row[4]),
                "needs_rewrite": (str(row[5]).lower() == "true") if row[5] else False,
            })
    except Exception:
        pass

    if page.get("page_type") == "concept":
        try:
            title_lower = (page.get("title") or slug).lower()
            slug_spaced = slug.replace("-", " ").lower()
            with get_session() as session:
                tri_rows = session.execute(text("""
                    SELECT kg.subject, kg.predicate, kg.object,
                           kg.source_doc_id::text, kg.confidence,
                           pm.title, kg.source_sentence
                    FROM knowledge_graph kg
                    LEFT JOIN paper_metadata pm ON pm.document_id = kg.source_doc_id
                    WHERE LOWER(kg.subject) IN (:t, :s)
                       OR LOWER(kg.object)  IN (:t, :s)
                    ORDER BY kg.confidence DESC, kg.subject
                    LIMIT 50
                """), {"t": title_lower, "s": slug_spaced}).fetchall()
            page["related_triples"] = [
                {
                    "subject": r[0], "predicate": r[1], "object": r[2],
                    "source_doc_id": r[3],
                    "confidence": float(r[4] or 1.0),
                    "source_title": r[5],
                    "source_sentence": r[6],
                }
                for r in tri_rows
            ]
        except Exception:
            page["related_triples"] = []

    page["content_html"] = _app._md_to_html(page.get("content", ""))
    return JSONResponse(page)


@router.post("/api/wiki/query")
async def api_wiki_query(question: str = Form(...), model: str = Form(None)):
    """Stream a wiki query — wraps wiki_ops.query_wiki as an SSE job."""
    from sciknow.core.wiki_ops import query_wiki
    from sciknow.web import app as _app

    job_id, queue = _app._create_job("wiki_query")
    loop = asyncio.get_event_loop()

    def gen():
        return query_wiki(question, model=model or None)

    threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


@router.post("/api/wiki/extract-kg")
async def api_wiki_extract_kg(force: bool = Form(False)):
    """Phase 54.6.8 — backfill knowledge_graph triples for already-compiled pages."""
    from sciknow.web import app as _app
    job_id, _queue = _app._create_job("wiki_extract_kg")
    loop = asyncio.get_event_loop()
    argv = ["wiki", "extract-kg"]
    if force:
        argv.append("--force")
    _app._spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


@router.post("/api/wiki/lint")
async def api_wiki_lint(deep: bool = Form(False), model: str = Form(None)):
    """Phase 54.6.2 — stream `sciknow wiki lint` over SSE."""
    from sciknow.core.wiki_ops import lint_wiki
    from sciknow.web import app as _app

    job_id, queue = _app._create_job("wiki_lint")
    loop = asyncio.get_event_loop()

    def gen():
        return lint_wiki(deep=bool(deep), model=(model or None))

    threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


@router.post("/api/wiki/consensus")
async def api_wiki_consensus(topic: str = Form(...), model: str = Form(None)):
    """Phase 54.6.2 — stream `sciknow wiki consensus` over SSE."""
    if not topic.strip():
        raise HTTPException(status_code=400, detail="topic required")
    from sciknow.core.wiki_ops import consensus_map
    from sciknow.web import app as _app

    job_id, queue = _app._create_job("wiki_consensus")
    loop = asyncio.get_event_loop()

    def gen():
        return consensus_map(topic.strip(), model=(model or None))

    threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


@router.post("/api/wiki/compile")
async def api_wiki_compile(
    rebuild: bool = Form(False),
    rewrite_stale: bool = Form(False),
    doc_id: str = Form(""),
):
    """SSE-streamed wrapper around ``sciknow wiki compile``."""
    from sciknow.web import app as _app
    job_id, _ = _app._create_job("wiki_compile")
    loop = asyncio.get_event_loop()
    argv = ["wiki", "compile"]
    if rebuild:        argv.append("--rebuild")
    if rewrite_stale:  argv.append("--rewrite-stale")
    if doc_id.strip(): argv += ["--doc-id", doc_id.strip()]
    _app._spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})
