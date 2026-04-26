"""``sciknow.web.routes.corpus`` — corpus endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. Behaviour
unchanged. Cross-module deps resolved via the standard lazy
`from sciknow.web import app as _app` shim.
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

router = APIRouter()


@router.post("/api/corpus/enrich")
async def api_corpus_enrich(
    dry_run: bool = Form(False),
    threshold: float = Form(0.85),
    limit: int = Form(0),
    delay: float = Form(0.2),
):
    """Invoke `sciknow db enrich` from the web UI — SSE log stream."""
    from sciknow.web import app as _app
    job_id, _queue = _app._create_job("corpus_enrich")
    loop = asyncio.get_event_loop()
    argv = ["db", "enrich",
            "--threshold", str(threshold),
            "--limit", str(limit),
            "--delay", str(delay)]
    if dry_run:
        argv.append("--dry-run")
    _app._spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


@router.post("/api/corpus/ingest-directory")
async def api_corpus_ingest_directory(
    path: str = Form(...),
    recursive: bool = Form(True),
    force: bool = Form(False),
    workers: int = Form(0),
):
    """SSE-streamed wrapper around ``sciknow ingest directory <path>``.

    Phase 46.F — path is a server-side directory. The wizard UI
    usually pairs this with an ``ingest/upload`` step that stages
    uploaded files into ``{data_dir}/inbox/`` first.
    """
    from sciknow.web import app as _app
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {p}")
    job_id, _ = _app._create_job("corpus_ingest_directory")
    loop = asyncio.get_event_loop()
    argv: list[str] = ["ingest", "directory", str(p)]
    if not recursive:
        argv.append("--no-recursive")
    if force:
        argv.append("--force")
    if workers and int(workers) > 0:
        argv += ["--workers", str(int(workers))]
    _app._spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id, "path": str(p)})


@router.post("/api/corpus/upload")
async def api_corpus_upload(request: Request):
    """Accept a multipart PDF upload + (optionally) queue an ingest job.

    Files are saved to ``{data_dir}/inbox/uploads_<ts>/`` inside the
    active project so multi-project isolation holds. If ``start_ingest``
    is truthy, an ingest job is spawned automatically and its job_id
    is returned; otherwise the client can trigger
    ``POST /api/corpus/ingest-directory`` itself with the returned
    ``staging_dir``.
    """
    from sciknow.web import app as _app
    from datetime import datetime, timezone
    from sciknow.config import settings

    form = await request.form()
    files = form.getlist("files")
    start_ingest = form.get("start_ingest", "false").lower() in {"1", "true", "yes"}
    force        = form.get("force",        "false").lower() in {"1", "true", "yes"}
    recursive    = form.get("recursive",    "true").lower()  in {"1", "true", "yes"}
    if not files:
        raise HTTPException(status_code=400, detail="no files in upload")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = Path(settings.data_dir) / "inbox" / f"uploads_{ts}"
    staging.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for f in files:
        if not hasattr(f, "filename") or not f.filename:
            continue
        # Strip path components — we only accept a flat filename.
        name = Path(f.filename).name
        if not name.lower().endswith(".pdf"):
            # Skip non-PDFs silently; ingestion would fail anyway.
            continue
        dest = staging / name
        # Avoid collisions within this batch.
        i = 0
        while dest.exists():
            i += 1
            dest = staging / f"{dest.stem}_{i}{dest.suffix}"
        contents = await f.read()
        dest.write_bytes(contents)
        saved.append(dest.name)

    if not saved:
        # Clean up empty staging dir
        try: staging.rmdir()
        except Exception: pass
        raise HTTPException(status_code=400,
                             detail="no .pdf files in upload (only PDFs accepted)")

    payload: dict = {
        "staging_dir": str(staging),
        "n_files": len(saved),
        "files": saved,
    }
    if start_ingest:
        job_id, _ = _app._create_job("corpus_ingest_upload")
        loop = asyncio.get_event_loop()
        argv = ["ingest", "directory", str(staging)]
        if not recursive: argv.append("--no-recursive")
        if force:         argv.append("--force")
        _app._spawn_cli_streaming(job_id, argv, loop)
        payload["job_id"] = job_id
    return JSONResponse(payload)


@router.post("/api/corpus/expand-author/preview")
async def api_corpus_expand_author_preview(
    name: str = Form(""),
    orcid: str = Form(""),
    year_from: int = Form(0),
    year_to: int = Form(0),
    limit: int = Form(0),
    strict_author: bool = Form(True),
    all_matches: bool = Form(False),
    relevance_query: str = Form(""),
    author_ids: str = Form(""),
):
    """Phase 54.6.1 — preview candidates without downloading.

    Runs search + corpus-dedup + relevance scoring, returns JSON. The UI
    renders a checkboxed list so the user can cherry-pick which DOIs to
    download via ``/api/corpus/expand-author/download-selected`` — the
    existing ``POST /api/corpus/expand-author`` still exists for the
    "auto-download by relevance threshold" override path.

    Phase 54.6.49 — ``author_ids`` (comma-separated OpenAlex short
    IDs like "A5079882695,A5033301096") bypasses name resolution and
    targets exactly those canonical authors. Used by the multi-select
    disambiguation banner: when OpenAlex returns the same person under
    multiple name variants, the user ticks the ones that are actually
    them and re-queries with all of them merged.

    May take 10-30s due to external API calls. Blocking — not SSE.
    """
    from sciknow.web import app as _app
    # Parse author_ids — accept comma/pipe/space separated
    id_list = [x.strip() for x in re.split(r"[,|\s]+", author_ids) if x.strip()] if author_ids else None

    if not name.strip() and not orcid.strip() and not id_list:
        raise HTTPException(
            status_code=400,
            detail="provide either author name, ORCID, or selected author IDs",
        )
    from sciknow.core.expand_ops import find_author_candidates
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: find_author_candidates(
                name=name,
                orcid=(orcid.strip() or None),
                year_from=(year_from or None),
                year_to=(year_to or None),
                limit=limit,
                all_matches=all_matches,
                strict_author=strict_author,
                relevance_query=relevance_query,
                score_relevance=True,
                author_ids=id_list,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _app.logger.exception("expand-author preview failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return JSONResponse(result)


@router.post("/api/corpus/expand-author")
async def api_corpus_expand_author(
    name: str = Form(...),
    orcid: str = Form(""),
    year_from: int = Form(0),
    year_to: int = Form(0),
    limit: int = Form(0),
    strict_author: bool = Form(True),
    all_matches: bool = Form(False),
    relevance: bool = Form(True),
    relevance_threshold: float = Form(0.0),
    relevance_query: str = Form(""),
    workers: int = Form(0),
    ingest: bool = Form(True),
    dry_run: bool = Form(False),
):
    """Phase 46.E — invoke ``sciknow db expand-author`` from the web UI.

    Runs as a background subprocess; stdout streams as SSE ``log`` events.
    """
    from sciknow.web import app as _app
    if not name.strip():
        raise HTTPException(status_code=400, detail="author name required")
    job_id, _queue = _app._create_job("corpus_expand_author")
    loop = asyncio.get_event_loop()
    argv: list[str] = ["db", "expand-author", name.strip()]
    if orcid.strip():
        argv += ["--orcid", orcid.strip()]
    if year_from:
        argv += ["--from", str(year_from)]
    if year_to:
        argv += ["--to", str(year_to)]
    if limit and int(limit) > 0:
        argv += ["--limit", str(int(limit))]
    argv += ["--workers", str(int(workers or 0))]
    argv += ["--relevance-threshold", str(float(relevance_threshold or 0.0))]
    if strict_author:   argv.append("--strict-author")
    else:               argv.append("--no-strict-author")
    if all_matches:     argv.append("--all-matches")
    argv.append("--relevance" if relevance else "--no-relevance")
    if relevance_query.strip():
        argv += ["--relevance-query", relevance_query.strip()]
    argv.append("--ingest" if ingest else "--no-ingest")
    if dry_run:
        argv.append("--dry-run")
    _app._spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


@router.post("/api/corpus/agentic/preview")
async def api_corpus_agentic_preview(
    question: str = Form(""),
    budget: int = Form(10),
    threshold: int = Form(3),
    model: str = Form(""),
):
    """Phase 54.6.132 — Agentic preview, single round. Decomposes the
    question, measures coverage, identifies gap sub-topics, gathers
    candidates per gap (OpenAlex topic search via
    ``find_topic_candidates``) and returns the merged shortlist
    annotated with each candidate's source sub-topic for cherry-pick
    in the candidates modal. The user picks + downloads via the
    existing ``/api/corpus/expand-author/download-selected`` route;
    re-calling this endpoint advances to the next round (coverage is
    re-measured against the now-updated corpus)."""
    from sciknow.ingestion.agentic_expand import run_preview_round

    q = (question or "").strip()
    if not q:
        return JSONResponse({"error": "question is required"}, status_code=400)
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_preview_round(
                q, budget_per_gap=budget,
                doc_threshold=threshold,
                model=(model.strip() or None),
            ),
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)[:500]}, status_code=500)
    return JSONResponse(result)


@router.get("/api/corpus/authors/top")
async def api_corpus_authors_top(limit: int = 40):
    """Phase 54.6.309 — list the top corpus authors by paper count, so
    the expand-author-refs picker doesn't have to spawn the CLI just to
    build its dropdown. Surname collisions are aggregated client-side.
    """
    lim = max(5, min(int(limit), 200))
    with get_session() as session:
        rows = session.execute(text("""
            SELECT auth->>'name' AS author_name, COUNT(*) AS n_papers
              FROM paper_metadata,
                   jsonb_array_elements(COALESCE(authors, '[]'::jsonb)) AS auth
             WHERE auth ? 'name'
             GROUP BY author_name
             ORDER BY n_papers DESC
             LIMIT :lim
        """), {"lim": lim}).fetchall()
    return JSONResponse({"authors": [
        {"name": r[0], "n_papers": int(r[1] or 0)} for r in rows
    ]})


@router.post("/api/corpus/expand-author-refs/preview")
async def api_corpus_expand_author_refs_preview(
    author: str = Form(...),
    min_mentions: int = Form(1),
    limit: int = Form(0),
    include_in_corpus: bool = Form(False),
    relevance_query: str = Form(""),
):
    """Phase 54.6.309 — preview the references cited by ``author``'s corpus
    works. Same output shape as the existing expand-author preview so
    the GUI's cherry-pick candidates modal renders it without changes.

    The response sorts by the in-corpus mention count descending; ties
    broken by cited_year descending. Each candidate carries an
    ``author_mentions`` count and ``self_cite_count`` so the UI can
    surface "cited 5× (2 self)" badges.
    """
    from sciknow.web import app as _app
    import re as _re
    from sciknow.ingestion.references import normalise_title_for_dedup

    author = (author or "").strip()
    if not author:
        raise HTTPException(400, "author required")

    def _surname(name: str) -> str:
        parts = [p for p in _re.split(r"[,;\s]+", (name or "").strip()) if p]
        if not parts:
            return ""
        last = parts[-1] if "," not in (name or "") else parts[0]
        return _re.sub(r"[^A-Za-zÀ-ſ]", "", last).lower()

    surname = _surname(author)
    if not surname:
        raise HTTPException(400, "could not derive a surname")

    with get_session() as session:
        paper_rows = session.execute(text("""
            SELECT pm.document_id::text, pm.title, pm.year, pm.doi
              FROM paper_metadata pm,
                   jsonb_array_elements(COALESCE(pm.authors, '[]'::jsonb)) AS auth
             WHERE auth ? 'name'
               AND LOWER(regexp_replace(
                     split_part(auth->>'name', ' ',
                       cardinality(regexp_split_to_array(auth->>'name', '\s+'))),
                     '[^A-Za-zÀ-ſ]', '', 'g'
                   )) = :sn
        """), {"sn": surname}).fetchall()

    if not paper_rows:
        return JSONResponse({
            "author": author, "surname": surname,
            "n_author_papers": 0, "candidates": [],
            "note": f"No corpus papers found for surname '{surname}'.",
        })

    paper_ids = [r[0] for r in paper_rows]
    with get_session() as session:
        cite_rows = session.execute(text("""
            SELECT cited_doi, cited_title, cited_authors, cited_year,
                   is_self_cite
              FROM citations
             WHERE citing_document_id = ANY(CAST(:ids AS uuid[]))
        """), {"ids": paper_ids}).fetchall()

    if not cite_rows:
        return JSONResponse({
            "author": author, "surname": surname,
            "n_author_papers": len(paper_rows), "candidates": [],
            "note": "No citations for those papers. Run `sciknow db link-citations` first.",
        })

    agg: dict[str, dict] = {}
    for r in cite_rows:
        cited_doi, cited_title, cited_authors, cited_year, is_self = r
        key = (cited_doi or "").lower().strip()
        if not key:
            key = "title:" + normalise_title_for_dedup(cited_title or "")
        if not key or key == "title:":
            continue
        a = agg.setdefault(key, {
            "doi": cited_doi or None, "title": cited_title or None,
            "authors": list(cited_authors or []), "year": cited_year,
            "mentions": 0, "self_cites": 0,
        })
        a["mentions"] += 1
        if is_self:
            a["self_cites"] += 1
        if not a["title"] and cited_title:
            a["title"] = cited_title
        if not a["year"] and cited_year:
            a["year"] = cited_year

    shortlist = [v for v in agg.values() if v["mentions"] >= min_mentions]
    shortlist.sort(key=lambda v: (-v["mentions"], -(v["year"] or 0)))

    dropped_in_corpus = 0
    if not include_in_corpus:
        with get_session() as session:
            ex = session.execute(text(
                "SELECT LOWER(doi) FROM paper_metadata WHERE doi IS NOT NULL"
            )).fetchall()
            existing = {r[0] for r in ex}
        before = len(shortlist)
        shortlist = [
            v for v in shortlist
            if not (v["doi"] and v["doi"].lower() in existing)
        ]
        dropped_in_corpus = before - len(shortlist)

    if limit and limit > 0:
        shortlist = shortlist[:limit]

    # Phase 54.6.315 — resolve title/authors/year via Crossref for
    # candidates whose `citations` row stored only a DOI. This was
    # the missing field the user was seeing as "(untitled)" rows in
    # the cherry-pick modal: reference extraction for some citing
    # papers (notably V.V. Zharkova's) captured DOIs but never
    # resolved the downstream metadata, and the preview had no
    # fallback. Parallel HTTP with a shared rate limiter keeps this
    # well inside Crossref's 50 RPS polite pool.
    needs_resolve = [v for v in shortlist
                     if v.get("doi") and (not v.get("title")
                                           or len((v.get("title") or "").strip()) < 5)]
    if needs_resolve:
        import concurrent.futures as _cf
        import html as _html
        import httpx as _httpx
        from sciknow.config import settings as _settings

        def _fetch(doi_in: str) -> dict | None:
            # Phase 54.6.315b — three real-world DOI gotchas:
            # 1. AMS-style DOIs stored HTML-encoded in the DB
            #    (`…079&lt;0061:apgtwa&gt;2.0.co;2`) — unescape first.
            # 2. Some older Springer/Nature DOIs (10.1007/s005850050897)
            #    resolve via 301 redirect, not direct 200 — opt into
            #    follow_redirects so httpx chases the redirect.
            # 3. JATS XML tags in Crossref titles & abstracts; strip
            #    them out before returning.
            import time as _time
            doi = _html.unescape((doi_in or "").strip())
            if not doi:
                return None
            url = f"https://api.crossref.org/works/{doi}"
            headers = {"User-Agent": f"sciknow/0.1 (mailto:{_settings.crossref_email})"}
            # Retry once on transient failure. Crossref's polite pool
            # occasionally 5xx's or socket-drops under parallel load;
            # a single 1 s backoff retry rescues those rows.
            for attempt in range(2):
                try:
                    with _httpx.Client(timeout=20, follow_redirects=True) as client:
                        r = client.get(url, headers=headers)
                    if r.status_code != 200:
                        if attempt == 0 and r.status_code in (429, 500, 502, 503, 504):
                            _time.sleep(1.0)
                            continue
                        return None
                    d = (r.json() or {}).get("message") or {}
                    title_list = d.get("title") or []
                    raw_title = title_list[0] if title_list else ""
                    clean_title = re.sub(r"<[^>]+>", "", raw_title).strip()
                    authors = d.get("author") or []
                    issued = (d.get("issued") or {}).get("date-parts") or []
                    year = int(issued[0][0]) if issued and issued[0] else None
                    return {
                        "title": clean_title,
                        "authors": [f"{(a.get('given') or '').strip()} {(a.get('family') or '').strip()}".strip()
                                    for a in authors],
                        "year": year,
                    }
                except Exception:
                    if attempt == 0:
                        _time.sleep(1.0)
                        continue
                    return None
            return None

        # Larger timeout window — one slow request should never starve
        # the whole preview. 8 workers × 20 s per request × 63 rows is
        # still ~16 s wall-clock worst case with good parallelism.
        with _cf.ThreadPoolExecutor(max_workers=8) as pool:
            fut_map = {pool.submit(_fetch, v["doi"]): v for v in needs_resolve}
            try:
                for fut in _cf.as_completed(fut_map, timeout=180):
                    v = fut_map[fut]
                    try:
                        res = fut.result()
                    except Exception:
                        res = None
                    if not res:
                        continue
                    if not v.get("title") and res.get("title"):
                        v["title"] = res["title"]
                    if not v.get("year") and res.get("year"):
                        v["year"] = res["year"]
                    if (not v.get("authors") or v.get("authors") == []) and res.get("authors"):
                        v["authors"] = res["authors"]
            except _cf.TimeoutError:
                _app.logger.warning(
                    "expand-author-refs resolve timed out after 180s; "
                    "%d/%d candidates may still be (untitled)",
                    sum(1 for f in fut_map if not f.done()),
                    len(fut_map),
                )

    # Relevance score. Phase 54.6.315 — auto-compute against the
    # corpus centroid even when relevance_query is empty, so the
    # cherry-pick modal always shows a meaningful Score column (the
    # sister expand-author preview already does this via score_relevance=True).
    relevance_scores: list[float] = []
    if shortlist:
        try:
            from sciknow.retrieval.relevance import (
                compute_corpus_centroid, embed_query, score_candidates,
            )
            anchor = (
                embed_query(relevance_query.strip())
                if relevance_query.strip()
                else compute_corpus_centroid()
            )
            titles = [v["title"] or "" for v in shortlist]
            relevance_scores = list(score_candidates(titles, anchor))
        except Exception as exc:
            _app.logger.warning("expand-author-refs relevance scoring failed: %s", exc)
            relevance_scores = []

    # Shape matches expand-author preview so the existing download-
    # selected endpoint and cherry-pick modal can consume it unchanged.
    candidates_out = []
    for i, v in enumerate(shortlist):
        rscore = relevance_scores[i] if i < len(relevance_scores) else None
        candidates_out.append({
            "doi": v["doi"],
            "title": v["title"] or "",
            "year": v["year"],
            "authors": [a if isinstance(a, str) else str(a) for a in (v["authors"] or [])],
            "author_mentions": v["mentions"],
            "self_cite_count": v["self_cites"],
            "relevance_score": rscore,
        })

    return JSONResponse({
        "author": author,
        "surname": surname,
        "n_author_papers": len(paper_rows),
        "n_unique_references": len(shortlist),
        "dropped_in_corpus": dropped_in_corpus,
        "candidates": candidates_out,
    })


@router.post("/api/corpus/expand-oeuvre/preview")
async def api_corpus_expand_oeuvre_preview(
    min_corpus_papers: int = Form(3),
    per_author_limit: int = Form(10),
    max_authors: int = Form(10),
    relevance_query: str = Form(""),
    strict_author: bool = Form(True),
):
    """Phase 54.6.131 — Oeuvre preview. Enumerates qualifying authors
    + per-author candidates without downloading; returns the merged
    shortlist annotated with the source author for cherry-pick in the
    candidates modal. Reuses ``find_oeuvre_candidates`` so the GUI
    plan matches the CLI plan exactly."""
    from sciknow.core.expand_ops import find_oeuvre_candidates

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: find_oeuvre_candidates(
                min_corpus_papers=min_corpus_papers,
                per_author_limit=per_author_limit,
                max_authors=max_authors,
                relevance_query=relevance_query.strip(),
                strict_author=strict_author,
                score_relevance=True,
            ),
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)[:500]}, status_code=500)
    return JSONResponse(result)


@router.post("/api/corpus/expand-author/download-selected")
async def api_corpus_expand_author_download_selected(request: Request):
    """Phase 54.6.1 — download + ingest the user-chosen subset from the
    Expand-by-Author preview modal.

    Body: JSON ``{"candidates": [{"doi": "...", "title": "...",
    "year": 2020}, ...], "workers": int, "ingest": bool}``.

    Spawns ``sciknow db download-dois --dois-file <tmp.json>`` and streams
    stdout as SSE. Tmp file is cleaned up when the job finishes.
    """
    from sciknow.web import app as _app
    import json as _json
    import tempfile

    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    raw_cands = body.get("candidates") or []
    if not isinstance(raw_cands, list) or not raw_cands:
        raise HTTPException(status_code=400, detail="candidates list required")

    # Sanitize to {doi, title, year, alternate_dois, alternate_arxiv_ids},
    # drop entries missing DOI. Phase 54.6.51 — alternates are used by
    # the downloader as fallback sources when the primary DOI's OA
    # discovery returns nothing (preprint-vs-journal duplicates).
    clean: list[dict] = []
    for c in raw_cands:
        if not isinstance(c, dict):
            continue
        doi = (c.get("doi") or "").strip()
        if not doi:
            continue
        alt_dois = [d for d in (c.get("alternate_dois") or []) if isinstance(d, str) and d.strip()]
        alt_arxiv = [a for a in (c.get("alternate_arxiv_ids") or []) if isinstance(a, str) and a.strip()]
        clean.append({
            "doi": doi,
            "title": (c.get("title") or "")[:500],
            "year": c.get("year") if isinstance(c.get("year"), int) else None,
            "alternate_dois": alt_dois[:10],
            "alternate_arxiv_ids": alt_arxiv[:10],
        })
    if not clean:
        raise HTTPException(status_code=400, detail="no valid DOIs in candidates")

    workers = int(body.get("workers") or 0)
    ingest = bool(body.get("ingest", True))
    # Phase 54.6.52 — retry_failed bypasses .no_oa_cache + .ingest_failed
    # for the current batch. Used by the GUI's "Retry previously-failed"
    # checkbox when the user wants to re-probe cached DOIs (e.g. after
    # new OA sources like HAL/Zenodo land in Phase 54.6.51).
    retry_failed = bool(body.get("retry_failed", False))

    # Persist the DOI list to a tempfile the CLI will read. Use the
    # project's data dir so it's namespaced per-project and survives
    # subprocess launch. We delete it after the job wraps in the finish
    # hook; until then the CLI has read access.
    tmp_dir = Path(tempfile.gettempdir()) / "sciknow-selected"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    import uuid
    tmp_path = tmp_dir / f"dois-{uuid.uuid4().hex[:12]}.json"
    tmp_path.write_text(_json.dumps(clean))

    job_id, _queue = _app._create_job("corpus_download_selected")
    loop = asyncio.get_event_loop()
    argv = [
        "db", "download-dois",
        "--dois-file", str(tmp_path),
        "--workers", str(workers),
    ]
    argv.append("--ingest" if ingest else "--no-ingest")
    if retry_failed:
        argv.append("--retry-failed")

    def _cleanup_tmp():
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    _app._spawn_cli_streaming(job_id, argv, loop, on_finish=_cleanup_tmp)
    return JSONResponse({
        "job_id": job_id,
        "n_selected": len(clean),
    })


@router.post("/api/corpus/expand-cites/preview")
async def api_corpus_expand_cites_preview(
    per_seed_cap: int = Form(50),
    total_limit: int = Form(500),
    relevance_query: str = Form(""),
):
    """Phase 54.6.4 — preview inbound-citation candidates (blocking JSON)."""
    from sciknow.web import app as _app
    from sciknow.core.expand_ops import find_inbound_citation_candidates
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: find_inbound_citation_candidates(
                per_seed_cap=int(per_seed_cap), total_limit=int(total_limit),
                relevance_query=relevance_query, score_relevance=True,
            ),
        )
    except Exception as exc:
        _app.logger.exception("expand-cites preview failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return JSONResponse(result)


@router.post("/api/corpus/expand-topic/preview")
async def api_corpus_expand_topic_preview(
    query: str = Form(""),
    limit: int = Form(500),
    relevance_query: str = Form(""),
):
    """Phase 54.6.4 — preview topic-search candidates."""
    from sciknow.web import app as _app
    if not query.strip():
        raise HTTPException(status_code=400, detail="query required")
    from sciknow.core.expand_ops import find_topic_candidates
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: find_topic_candidates(
                query, limit=int(limit),
                relevance_query=relevance_query, score_relevance=True,
            ),
        )
    except Exception as exc:
        _app.logger.exception("expand-topic preview failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return JSONResponse(result)


@router.post("/api/corpus/expand-coauthors/preview")
async def api_corpus_expand_coauthors_preview(
    depth: int = Form(1),
    per_author_cap: int = Form(10),
    total_limit: int = Form(500),
    relevance_query: str = Form(""),
):
    """Phase 54.6.4 — preview coauthor-snowball candidates."""
    from sciknow.web import app as _app
    from sciknow.core.expand_ops import find_coauthor_candidates
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: find_coauthor_candidates(
                depth=int(depth), per_author_cap=int(per_author_cap),
                total_limit=int(total_limit),
                relevance_query=relevance_query, score_relevance=True,
            ),
        )
    except Exception as exc:
        _app.logger.exception("expand-coauthors preview failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return JSONResponse(result)


@router.post("/api/corpus/cleanup-downloads")
async def api_corpus_cleanup_downloads(
    dry_run: bool = Form(False),
    delete_dupes: bool = Form(True),
    cross_project: bool = Form(True),
    clean_failed: bool = Form(True),
    include_inbox: bool = Form(True),
):
    """Phase 54.6.4 + 54.6.19 + 54.6.273 — trigger `sciknow db cleanup-downloads`.

    Streams the subprocess log over SSE. Defaults: dry_run=False,
    delete_dupes=True, cross_project=True, clean_failed=True,
    include_inbox=True — so a single click removes all downloads +
    inbox files already ingested anywhere (including other projects),
    nukes the failed-ingest archive + associated documents rows, and
    removes empty inbox subfolders. The GUI exposes one button;
    advanced users can flip the flags via the CLI.
    """
    from sciknow.web import app as _app
    job_id, _queue = _app._create_job("corpus_cleanup_downloads")
    loop = asyncio.get_event_loop()
    argv = ["db", "cleanup-downloads"]
    if dry_run:
        argv.append("--dry-run")
    if delete_dupes:
        argv.append("--delete-dupes")
    argv.append("--cross-project" if cross_project else "--no-cross-project")
    argv.append("--clean-failed" if clean_failed else "--no-clean-failed")
    argv.append("--include-inbox" if include_inbox else "--no-include-inbox")
    _app._spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})


@router.post("/api/corpus/expand/preview")
async def api_corpus_expand_preview(
    limit: int = Form(0),
    strategy: str = Form("rrf"),
    budget: int = Form(50),
    relevance: bool = Form(True),
    relevance_threshold: float = Form(0.0),
    relevance_query: str = Form(""),
    resolve: bool = Form(False),
):
    """Phase 54.6.3 — Preview the citation-expansion shortlist without
    downloading anything.

    Spawns ``sciknow db expand --dry-run --shortlist-tsv <tmp>`` as a
    subprocess so SSE streams its progress; the tempfile path is
    stored on the job record so the follow-up GET
    ``/api/corpus/expand/preview/{job_id}/candidates`` can parse it
    once the job has completed.

    Why subprocess + TSV rather than an in-process helper like
    expand-author: the citation expansion pipeline is ~250 lines of
    intertwined reference extraction, multi-source RRF ranking, and
    console output — far riskier to duplicate than ``find_author_-
    candidates``. Using the shortlist TSV keeps us bit-for-bit
    identical to the CLI path.
    """
    from sciknow.web import app as _app
    import tempfile
    import uuid

    tmp_dir = Path(tempfile.gettempdir()) / "sciknow-expand-preview"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = tmp_dir / f"shortlist-{uuid.uuid4().hex[:12]}.tsv"

    job_id, _queue = _app._create_job("corpus_expand_preview")
    # Stash the tempfile path so the candidates GET can find it.
    _app._jobs[job_id]["preview_tsv"] = str(tsv_path)

    loop = asyncio.get_event_loop()
    argv = [
        "db", "expand",
        "--dry-run",
        "--shortlist-tsv", str(tsv_path),
        "--limit", str(int(limit)),
        "--strategy", (strategy or "rrf"),
        "--budget", str(int(budget) if budget else 50),
        "--relevance-threshold", str(float(relevance_threshold or 0.0)),
    ]
    argv.append("--relevance" if relevance else "--no-relevance")
    argv.append("--resolve" if resolve else "--no-resolve")
    if relevance_query.strip():
        argv += ["--relevance-query", relevance_query.strip()]

    _app._spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id, "tsv_key": job_id})


@router.get("/api/corpus/expand/preview/{job_id}/candidates")
async def api_corpus_expand_preview_candidates(job_id: str):
    """Parse the shortlist TSV produced by a preview job and return
    JSON candidates (same shape as expand-author preview so the same
    UI can render both).
    """
    from sciknow.web import app as _app
    import csv
    with _app._job_lock:
        job = _app._jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    tsv_path = Path(job.get("preview_tsv", ""))
    if not tsv_path.exists():
        raise HTTPException(404, "Preview TSV not found (job may still be running)")

    candidates: list[dict] = []
    kept = 0
    dropped = 0
    drop_reasons: dict[str, int] = {}
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            decision = (row.get("decision") or "").upper()
            doi = (row.get("doi") or "").strip() or None
            if decision == "KEEP":
                kept += 1
            else:
                dropped += 1
                reason = row.get("drop_reason") or "unspecified"
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            # Parse optional numerics.
            def _f(k):
                v = row.get(k)
                if not v or v in ("None", "nan"):
                    return None
                try:
                    return float(v)
                except Exception:
                    return None
            def _i(k):
                v = row.get(k)
                if not v or v in ("None", "nan"):
                    return None
                try:
                    return int(float(v))
                except Exception:
                    return None
            year = _i("year")
            candidates.append({
                "doi": doi,
                "arxiv_id": (row.get("arxiv_id") or "").strip() or None,
                "title": (row.get("title") or "").strip(),
                "authors": [],
                "year": year,
                "relevance_score": _f("bge_m3_cosine"),
                "rrf_score": _f("rrf_score"),
                "decision": decision or None,
                "drop_reason": (row.get("drop_reason") or "").strip() or None,
                "signals": {
                    "co_citation":       _f("co_citation"),
                    "bib_coupling":      _f("bib_coupling"),
                    "pagerank":          _f("pagerank"),
                    "influential_cites": _i("influential_cites"),
                    "cited_by":          _i("cited_by"),
                    "velocity":          _f("velocity"),
                    "concept_overlap":   _f("concept_overlap"),
                    "venue":             (row.get("venue") or "").strip() or None,
                },
            })
    # Clean up after parse — one-shot read, no reason to leave on disk.
    try:
        tsv_path.unlink(missing_ok=True)
    except Exception:
        pass
    return JSONResponse({
        "candidates": candidates,
        "info": {
            "total": kept + dropped,
            "kept": kept,
            "dropped": dropped,
            "drop_reasons": drop_reasons,
        },
    })


@router.post("/api/corpus/expand")
async def api_corpus_expand(
    limit: int = Form(0),
    # Phase 54.6.113 — RRF pool size per round. Default mirrors the CLI's
    # default (50). Separate from `limit` which is the total download cap.
    budget: int = Form(50),
    dry_run: bool = Form(False),
    resolve: bool = Form(False),
    ingest: bool = Form(True),
    relevance: bool = Form(True),
    relevance_threshold: float = Form(0.0),
    relevance_query: str = Form(""),
    workers: int = Form(0),
):
    """Invoke `sciknow db expand` from the web UI — SSE log stream.

    ``budget`` is the RRF pool size per round (default 50); ``limit`` is
    the hard cap on total downloads. See ``docs/EXPAND_RESEARCH.md`` §6a.
    Heavy flags (download_dir, delay) are left at CLI defaults to keep
    the web UX simple.
    """
    from sciknow.web import app as _app
    job_id, _queue = _app._create_job("corpus_expand")
    loop = asyncio.get_event_loop()
    argv = ["db", "expand",
            "--limit", str(limit),
            "--budget", str(max(5, min(int(budget), 200))),
            "--relevance-threshold", str(relevance_threshold),
            "--workers", str(workers)]
    if dry_run:
        argv.append("--dry-run")
    argv.append("--resolve" if resolve else "--no-resolve")
    argv.append("--ingest" if ingest else "--no-ingest")
    argv.append("--relevance" if relevance else "--no-relevance")
    if relevance_query:
        argv += ["--relevance-query", relevance_query]
    _app._spawn_cli_streaming(job_id, argv, loop)
    return JSONResponse({"job_id": job_id})
