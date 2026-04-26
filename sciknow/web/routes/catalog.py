"""``sciknow.web.routes.catalog`` — catalog browse + cluster endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. 6 handlers
covering paper-list / authors / domains / topics browse plus the
two SSE-streamed CLI wrappers (cluster + raptor build).

Cross-module deps (via lazy `_app` shim):
  - `_app._create_job`, `_app._spawn_cli_streaming` for the SSE wrappers.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

router = APIRouter()


@router.get("/api/catalog")
async def api_catalog(
    page: int = 1,
    per_page: int = 25,
    year_from: int = None,
    year_to: int = None,
    author: str = None,
    journal: str = None,
    topic_cluster: str = None,
):
    """Paginated paper list with optional filters. Mirrors `sciknow catalog list`.

    Phase 41 — query is fully static. Every optional filter is
    always bound (as the real value or NULL) and gated by a
    ``(:param IS NULL OR …)`` short-circuit. This removes the
    f-string interpolation of pre-built WHERE fragments that the
    Phase 22 audit flagged: no code path builds SQL from Python
    strings anymore, so a future maintainer can't accidentally
    concatenate user input into the query shape.
    """
    page = max(page, 1)
    per_page = min(max(per_page, 1), 100)
    offset = (page - 1) * per_page

    params: dict = {
        "limit": per_page,
        "offset": offset,
        "year_from": year_from,
        "year_to": year_to,
        "author": f"%{author}%" if author else None,
        "journal": f"%{journal}%" if journal else None,
        "topic_cluster": topic_cluster or None,
    }

    with get_session() as session:
        total = session.execute(text("""
            SELECT COUNT(*) FROM paper_metadata pm
            WHERE (:year_from IS NULL OR pm.year >= :year_from)
              AND (:year_to   IS NULL OR pm.year <= :year_to)
              AND (:author    IS NULL OR EXISTS (
                   SELECT 1 FROM jsonb_array_elements(pm.authors) a
                   WHERE a->>'name' ILIKE :author))
              AND (:journal        IS NULL OR pm.journal ILIKE :journal)
              AND (:topic_cluster  IS NULL OR pm.topic_cluster = :topic_cluster)
        """), params).scalar() or 0

        rows = session.execute(text("""
            SELECT pm.document_id::text, pm.title, pm.year, pm.authors,
                   pm.journal, pm.doi, pm.abstract, pm.topic_cluster,
                   pm.metadata_source
            FROM paper_metadata pm
            WHERE (:year_from IS NULL OR pm.year >= :year_from)
              AND (:year_to   IS NULL OR pm.year <= :year_to)
              AND (:author    IS NULL OR EXISTS (
                   SELECT 1 FROM jsonb_array_elements(pm.authors) a
                   WHERE a->>'name' ILIKE :author))
              AND (:journal        IS NULL OR pm.journal ILIKE :journal)
              AND (:topic_cluster  IS NULL OR pm.topic_cluster = :topic_cluster)
            ORDER BY pm.year DESC NULLS LAST, pm.title
            LIMIT :limit OFFSET :offset
        """), params).fetchall()

    papers = [
        {
            "document_id": r[0],
            "title": r[1] or "(untitled)",
            "year": r[2],
            "authors": r[3] or [],
            "journal": r[4],
            "doi": r[5],
            "abstract": (r[6] or "")[:600],
            "topic_cluster": r[7],
            "metadata_source": r[8],
        }
        for r in rows
    ]
    return JSONResponse({
        "page": page,
        "per_page": per_page,
        "total": total,
        "n_pages": (total + per_page - 1) // per_page,
        "papers": papers,
    })


@router.get("/api/catalog/authors")
async def api_catalog_authors(
    q: str = "",
    limit: int = 50,
    min_papers: int = 1,
):
    """Phase 46.E — ranked + searchable author index.

    Returns authors from ``paper_metadata.authors`` unnested and grouped
    by name, ranked by ``(citation_count DESC, paper_count DESC)`` so
    the most-cited / most-prolific names surface first.
    """
    limit = max(1, min(500, int(limit or 50)))
    min_papers = max(1, int(min_papers or 1))
    q_like = f"%{q.strip()}%" if q and q.strip() else None
    with get_session() as session:
        base_sql = """
            WITH exploded AS (
                SELECT author->>'name' AS name,
                       COALESCE(author->>'orcid', '') AS orcid,
                       pm.document_id
                FROM paper_metadata pm
                CROSS JOIN LATERAL jsonb_array_elements(pm.authors) AS author
                WHERE pm.authors IS NOT NULL
                  AND author->>'name' IS NOT NULL
                  AND trim(author->>'name') != ''
            )
            SELECT e.name,
                   COUNT(DISTINCT e.document_id) AS n_papers,
                   COUNT(c.id)                   AS n_cites,
                   (ARRAY_AGG(DISTINCT NULLIF(e.orcid, '')))[1] AS first_orcid
            FROM exploded e
            LEFT JOIN citations c ON c.cited_document_id = e.document_id
            {where}
            GROUP BY e.name
            HAVING COUNT(DISTINCT e.document_id) >= :min_papers
            ORDER BY n_cites DESC, n_papers DESC, e.name
            LIMIT :limit
        """
        params = {"min_papers": min_papers, "limit": limit}
        if q_like:
            where = "WHERE e.name ILIKE :q"
            params["q"] = q_like
        else:
            where = ""
        rows = session.execute(
            text(base_sql.format(where=where)), params,
        ).fetchall()

    return JSONResponse({
        "authors": [
            {"name": r[0], "n_papers": int(r[1] or 0),
             "n_citations": int(r[2] or 0), "orcid": r[3] or None}
            for r in rows
        ],
        "query": q,
        "limit": limit,
    })


@router.get("/api/catalog/domains")
async def api_catalog_domains(limit: int = 60):
    """Phase 46.E — ranked domain / tag index (paper_metadata.domains unnested)."""
    limit = max(1, min(500, int(limit or 60)))
    with get_session() as session:
        rows = session.execute(text("""
            SELECT tag, COUNT(DISTINCT pm.document_id) AS n
            FROM paper_metadata pm
            CROSS JOIN LATERAL unnest(pm.domains) AS tag
            WHERE pm.domains IS NOT NULL
              AND array_length(pm.domains, 1) > 0
              AND trim(tag) != ''
            GROUP BY tag
            ORDER BY n DESC, tag
            LIMIT :lim
        """), {"lim": limit}).fetchall()
    return JSONResponse({
        "domains": [{"name": r[0], "n": int(r[1])} for r in rows],
    })


@router.get("/api/catalog/topics")
async def api_catalog_topics(name: str = None):
    """Topic cluster breakdown (sciknow catalog topics).

    With no args: returns every non-null cluster name + paper count.
    With ?name=...: returns the cluster's paper list (title/year/doi).
    """
    with get_session() as session:
        if name:
            rows = session.execute(text("""
                SELECT pm.document_id::text, pm.title, pm.year, pm.doi,
                       pm.authors, pm.journal
                FROM paper_metadata pm
                WHERE pm.topic_cluster = :n
                ORDER BY pm.year DESC NULLS LAST, pm.title
                LIMIT 500
            """), {"n": name}).fetchall()
            papers = [
                {"document_id": r[0], "title": r[1], "year": r[2],
                 "doi": r[3], "authors": r[4] or [], "journal": r[5]}
                for r in rows
            ]
            return JSONResponse({"name": name, "papers": papers, "n": len(papers)})

        rows = session.execute(text("""
            SELECT topic_cluster, COUNT(*)
            FROM paper_metadata
            WHERE topic_cluster IS NOT NULL AND topic_cluster != ''
            GROUP BY topic_cluster
            ORDER BY COUNT(*) DESC
        """)).fetchall()
    return JSONResponse({
        "topics": [{"name": r[0], "n": int(r[1])} for r in rows],
    })


@router.post("/api/catalog/cluster")
async def api_catalog_cluster(
    min_cluster_size: int = Form(0),
    rebuild: bool = Form(False),
    dry_run: bool = Form(False),
):
    """SSE-streamed wrapper around ``sciknow catalog cluster``."""
    from sciknow.web import app as _app
    job_id, _ = _app._create_job("catalog_cluster")
    loop = asyncio.get_event_loop()
    argv = ["catalog", "cluster"]
    if min_cluster_size and int(min_cluster_size) > 0:
        argv += ["--min-cluster-size", str(int(min_cluster_size))]
    if rebuild: argv.append("--rebuild")
    if dry_run: argv.append("--dry-run")
    _app._spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


@router.post("/api/catalog/raptor/build")
async def api_catalog_raptor_build():
    """SSE-streamed wrapper around ``sciknow catalog raptor build``.

    RAPTOR is a one-off batch op (typically 5-30 min depending on
    corpus size). No options exposed — build policy uses the CLI
    defaults.
    """
    from sciknow.web import app as _app
    job_id, _ = _app._create_job("catalog_raptor_build")
    loop = asyncio.get_event_loop()
    _app._spawn_cli_streaming(job_id, ["catalog", "raptor", "build"], loop)
    return JSONResponse({"job_id": job_id})
