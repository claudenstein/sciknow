"""``sciknow.web.routes.book`` — book-level CRUD + planning endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. 9 handlers
covering book metadata get/put, plan/outline generation,
auto-expand preview, create, length-report, and book-types.

Cross-module deps via the standard lazy `_app` shim (prepended into
each handler that touches `_app._get_book_data` or `_app._book_id`).
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

logger = logging.getLogger("sciknow.web.routes.book")
router = APIRouter()


@router.get("/api/book")
async def api_book():
    from sciknow.web import app as _app
    book, chapters, drafts, gaps, comments = _app._get_book_data()
    # book columns: id, title, description, plan, status, custom_metadata
    meta = (book[5] if book and len(book) > 5 else None) or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    # Phase 17 — expose the length target so the GUI Plan modal can
    # render it (and show the default when unset).
    target_chapter_words = meta.get("target_chapter_words") if isinstance(meta, dict) else None
    # Phase 39 — expose the style fingerprint so the Book Settings
    # modal can render metrics + a "last refreshed" stamp without a
    # second round-trip.
    style_fingerprint = meta.get("style_fingerprint") if isinstance(meta, dict) else None
    # Phase 54.6.148 — expose book_type so Book Settings can restore
    # the dropdown selection, and derive the project-type default for
    # the effective-target display so users see where the fallback
    # currently lands.
    from sciknow.core.project_type import get_project_type
    book_type = (book[6] if book and len(book) > 6 else None) or "scientific_book"
    try:
        pt = get_project_type(book_type)
        default_tcw = pt.default_target_chapter_words
    except Exception:
        default_tcw = 6000
    return {
        "id": book[0] if book else "",
        "title": book[1] if book else "",
        "description": (book[2] or "") if book else "",
        "plan": (book[3] or "") if book else "",
        "status": (book[4] or "draft") if book else "draft",
        "target_chapter_words": target_chapter_words,  # may be None → client shows default
        "default_target_chapter_words": default_tcw,
        "book_type": book_type,
        "style_fingerprint": style_fingerprint,
        "chapters": len(chapters),
        "drafts": len(drafts),
        "gaps": len(gaps),
        "comments": len(comments),
    }


@router.put("/api/book")
async def api_book_update(
    title: str = Form(None),
    description: str = Form(None),
    plan: str = Form(None),
    target_chapter_words: int = Form(None),
    book_type: str = Form(None),  # Phase 54.6.148
):
    """Update the book's title, description (short blurb), plan
    (the 200-500 word thesis/scope document used by the writer prompt),
    length target, or project type. All fields are optional — only the
    ones you pass get updated.

    Phase 17 — target_chapter_words lives in books.custom_metadata as
    a JSONB key so we can add more book-level settings without a
    schema change each time. Passing a zero or negative value clears
    the setting (reverts to the project-type default via 54.6.143's
    fallback chain).

    Phase 54.6.148 — book_type can be changed post-creation. Validated
    against the ProjectType registry (rejects unknown slugs). Changing
    type only affects future autowrite runs via the resolver's Level 3
    fallback; explicit per-chapter / per-section targets are unchanged.
    """
    from sciknow.web import app as _app
    updates = []
    params: dict = {"bid": _app._book_id}
    if title is not None:
        updates.append("title = :title")
        params["title"] = title
    if description is not None:
        updates.append("description = :desc")
        params["desc"] = description
    if plan is not None:
        updates.append("plan = :plan")
        params["plan"] = plan
    if book_type is not None:
        # Validate against the registry so a typo doesn't silently
        # downgrade to the default fallback in get_project_type.
        from sciknow.core.project_type import validate_type_slug
        try:
            validate_type_slug(book_type)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        updates.append("book_type = :btype")
        params["btype"] = book_type
    if target_chapter_words is not None:
        # Merge into JSONB so we preserve any other keys. We use the
        # `||` concat operator + jsonb_build_object so the JSON shape
        # is built server-side. For a clear/delete, we use `- key`.
        # Note: use CAST(... AS int) instead of `:tcw::int` because
        # SQLAlchemy's parameter parser confuses `::int` with a bound
        # parameter name. Same gotcha as _save_draft() in book_ops.py.
        if target_chapter_words > 0:
            updates.append(
                "custom_metadata = "
                "COALESCE(custom_metadata, CAST('{}' AS jsonb)) || "
                "jsonb_build_object('target_chapter_words', CAST(:tcw AS int))"
            )
            params["tcw"] = int(target_chapter_words)
        else:
            updates.append(
                "custom_metadata = "
                "COALESCE(custom_metadata, CAST('{}' AS jsonb)) - 'target_chapter_words'"
            )
    if not updates:
        return JSONResponse({"ok": True})

    with get_session() as session:
        session.execute(text(
            f"UPDATE books SET {', '.join(updates)} WHERE id::text = :bid"
        ), params)
        session.commit()
    return JSONResponse({"ok": True})


@router.post("/api/book/style-fingerprint/refresh")
async def api_book_style_fingerprint_refresh():
    """Phase 39 — recompute the book's style fingerprint on demand.

    Mirrors `sciknow book style refresh` in the web layer so the Book
    Settings modal can trigger a rebuild without dropping to the CLI.
    Runs synchronously (no SSE) because the work is pure SQL + regex
    over the book's drafts — sub-second for books with <500 drafts.
    """
    from sciknow.web import app as _app
    from sciknow.core.style_fingerprint import compute_style_fingerprint
    try:
        fp = compute_style_fingerprint(_app._book_id)
    except Exception as exc:
        logger.warning("style fingerprint refresh failed: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, "fingerprint": fp})


@router.post("/api/book/plan/generate")
async def api_book_plan_generate(model: str = Form(None)):
    """Generate (or regenerate) the book plan via the LLM, streaming
    tokens to the browser via SSE. Mirrors the `sciknow book plan --edit`
    CLI flow but persists to drafts.custom_metadata-style streaming."""
    from sciknow.web import app as _app
    job_id, queue = _app._create_job("book_plan_generate")
    loop = asyncio.get_event_loop()

    def gen():
        from sciknow.rag import prompts as rag_prompts
        from sciknow.rag.llm import stream as llm_stream

        with get_session() as session:
            book = session.execute(text("""
                SELECT id::text, title, description, plan
                FROM books WHERE id::text = :bid
            """), {"bid": _app._book_id}).fetchone()
            if not book:
                yield {"type": "error", "message": "Book not found."}
                return
            chapters = session.execute(text("""
                SELECT number, title, description FROM book_chapters
                WHERE book_id = :bid ORDER BY number
            """), {"bid": _app._book_id}).fetchall()
            papers = session.execute(text("""
                SELECT pm.title, pm.year FROM paper_metadata pm
                JOIN documents d ON d.id = pm.document_id
                WHERE d.ingestion_status = 'complete' AND pm.title IS NOT NULL
                ORDER BY pm.year DESC NULLS LAST LIMIT 200
            """)).fetchall()

        ch_list = [{"number": r[0], "title": r[1], "description": r[2] or ""}
                   for r in chapters]
        paper_list = [{"title": r[0], "year": r[1]} for r in papers]

        yield {"type": "progress", "stage": "generating",
               "detail": f"Drafting plan from {len(ch_list)} chapters and {len(paper_list)} papers..."}

        sys_p, usr_p = rag_prompts.book_plan(book[1], book[2], ch_list, paper_list)
        tokens: list[str] = []
        for tok in llm_stream(sys_p, usr_p, model=model or None):
            tokens.append(tok)
            yield {"type": "token", "text": tok}

        new_plan = "".join(tokens).strip()
        # Persist
        with get_session() as session:
            session.execute(text(
                "UPDATE books SET plan = :plan WHERE id::text = :bid"
            ), {"plan": new_plan, "bid": _app._book_id})
            session.commit()
        yield {"type": "completed", "plan": new_plan, "chars": len(new_plan)}

    threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


@router.post("/api/book/outline/generate")
async def api_book_outline_generate(
    model: str = Form(None),
    method: str = Form(""),
):
    """Phase 54.6.8 — generate + save chapter outline for the active book.

    Mirrors ``sciknow book outline``: prompts the LLM with the book title
    + paper corpus, parses the JSON chapters list, inserts any chapter
    whose number isn't already in ``book_chapters``. Streams tokens so
    the user sees progress; existing chapters are preserved (the flow
    is additive, never destructive).

    Phase 54.6.14 — optional ``method`` name from the elicitation
    catalogue steers the LLM's approach (e.g. "Tree of Thoughts",
    "First Principles"). Prepended as a one-paragraph preamble to
    the user prompt.
    """
    from sciknow.web import app as _app
    job_id, queue = _app._create_job("book_outline_generate")
    loop = asyncio.get_event_loop()

    def gen():
        import json as _json
        from sciknow.rag import prompts as rag_prompts
        from sciknow.rag.llm import stream as llm_stream
        from sciknow.core.project_type import get_project_type
        from sciknow.core.methods import get_method, method_preamble

        with get_session() as session:
            book = session.execute(text("""
                SELECT id::text, title, description, project_type, plan
                FROM books WHERE id::text = :bid
            """), {"bid": _app._book_id}).fetchone()
            if not book:
                yield {"type": "error", "message": "Book not found."}
                return
            # Flat project types have a fixed one-chapter shape. Refuse
            # to generate an outline for them — mirrors the CLI.
            pt = get_project_type(book[3] if len(book) > 3 else None)
            if pt.is_flat:
                yield {"type": "error", "message":
                       f"Project type {pt.slug!r} is flat — no outline to generate."}
                return
            papers = session.execute(text("""
                SELECT pm.title, pm.year FROM paper_metadata pm
                JOIN documents d ON d.id = pm.document_id
                WHERE d.ingestion_status = 'complete' AND pm.title IS NOT NULL
                ORDER BY pm.year DESC NULLS LAST LIMIT 200
            """)).fetchall()

            # Phase 54.6.x — gather topic-cluster catalogue + per-cluster
            # representative abstracts so the LLM sees what the corpus
            # actually contains, not just paper titles. We pull every
            # cluster (paper count >= 2) and the 3 most-recent papers
            # in each one; per-paper abstracts are truncated to 280
            # chars so 19 clusters × 3 papers × 280 ≈ 16 KB, well
            # within a 16K context after the rest of the prompt.
            cluster_rows = session.execute(text("""
                SELECT pm.topic_cluster, COUNT(*) AS n
                FROM paper_metadata pm
                JOIN documents d ON d.id = pm.document_id
                WHERE d.ingestion_status = 'complete'
                  AND pm.topic_cluster IS NOT NULL
                  AND pm.topic_cluster != ''
                GROUP BY pm.topic_cluster
                HAVING COUNT(*) >= 2
                ORDER BY COUNT(*) DESC
            """)).fetchall()
            cluster_catalogue: list[dict] = []
            for cname, n in cluster_rows:
                rep_rows = session.execute(text("""
                    SELECT pm.title, pm.year, pm.abstract
                    FROM paper_metadata pm
                    JOIN documents d ON d.id = pm.document_id
                    WHERE d.ingestion_status = 'complete'
                      AND pm.topic_cluster = :tc
                      AND pm.title IS NOT NULL
                    ORDER BY pm.year DESC NULLS LAST,
                             LENGTH(COALESCE(pm.abstract, '')) DESC
                    LIMIT 3
                """), {"tc": cname}).fetchall()
                cluster_catalogue.append({
                    "name": cname,
                    "count": int(n or 0),
                    "papers": [
                        {
                            "title": r[0],
                            "year": r[1],
                            "abstract": (
                                (r[2] or "").strip().replace("\n", " ")[:280]
                            ),
                        }
                        for r in rep_rows
                    ],
                })

        paper_list = [{"title": r[0], "year": r[1]} for r in papers if r[0]]
        plan_text = book[4] if len(book) > 4 else None
        n_clusters = len(cluster_catalogue)
        yield {"type": "progress", "stage": "generating",
               "detail": (
                   f"Drafting outline from {len(paper_list)} papers"
                   + (f" + {n_clusters} topic clusters" if n_clusters else "")
                   + (" + book plan" if plan_text and plan_text.strip() else "")
                   + "…"
               )}

        # Phase 54.6.x — book.plan (the leitmotiv) AND topic-cluster
        # catalogue (with abstracts) are now explicit inputs. Without
        # them, the prompt was effectively blind: just paper titles
        # and the book title, with no signal about what the corpus
        # actually says or how it groups topically.
        sys_p, usr_p = rag_prompts.outline(
            book_title=book[1],
            papers=paper_list,
            plan=plan_text,
            clusters=cluster_catalogue,
        )
        # Inject method preamble if the user picked one.
        if method and method.strip():
            m = get_method("elicitation", method)
            if m:
                usr_p = method_preamble(m) + usr_p
        # Phase 54.6.297 — resolve the outline-specific model.
        # Explicit request param > BOOK_OUTLINE_MODEL env > LLM_MODEL
        # (the llm_stream default).
        from sciknow.config import settings as _settings
        effective_model = (
            model
            or getattr(_settings, "book_outline_model", None)
            or None
        )
        tokens: list[str] = []
        for tok in llm_stream(sys_p, usr_p, model=effective_model):
            tokens.append(tok)
            yield {"type": "token", "text": tok}

        raw = "".join(tokens).strip()
        # Same JSON-fence strip as the CLI.
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            data = _json.loads(raw, strict=False)
            chapters = data.get("chapters", [])
        except Exception as exc:
            yield {"type": "error",
                   "message": f"LLM returned invalid JSON: {exc}"}
            return

        if not chapters:
            yield {"type": "error",
                   "message": "No chapters in LLM response."}
            return

        # Phase 54.6.65 — per-chapter density-based section trim. One
        # hybrid retrieval per chapter on topic_query → count distinct
        # papers → bucket to a target section count → trim if over.
        try:
            from sciknow.core.book_ops import resize_sections_by_density as _resize
            yield {"type": "progress", "stage": "resizing",
                   "detail": "Resizing sections by corpus evidence density…"}
            chapters = _resize(chapters, model=effective_model)
        except Exception as exc:
            logger.warning("density resize failed: %s", exc)

        # Phase 54.6.x — DEEP outline post-pass. For every chapter ×
        # section, run hybrid retrieval and call SECTION_PLAN with the
        # leitmotiv + retrieved evidence + earlier chapters to produce
        # bullet-list concept plans. After this, sections are saved as
        # full {slug, title, plan, target_words} dicts and are
        # immediately ready for autowrite without a manual auto-plan
        # pass. Failures per section degrade gracefully — the section
        # is still inserted, just without a plan.
        try:
            from sciknow.core.book_ops import deep_plan_outline_chapters as _deep
            yield {"type": "progress", "stage": "deep_planning",
                   "detail": (
                       "Deep section planning (per-section retrieval + "
                       "leitmotiv-grounded concept lists)…"
                   )}
            for evt in _deep(
                chapters,
                book_title=book[1],
                book_type=(book[3] if len(book) > 3 else None) or "scientific_book",
                book_plan=plan_text,
                model=effective_model,
            ):
                # Forward every event to the SSE stream so the GUI shows
                # per-section progress.
                yield evt
        except Exception as exc:  # noqa: BLE001
            logger.warning("deep section planning failed: %s", exc)

        # Additive insert — skip numbers that already exist so the
        # user can re-run without destroying manual edits.
        inserted = 0
        skipped = 0
        with get_session() as session:
            for ch in chapters:
                num = ch.get("number")
                if not isinstance(num, int):
                    continue
                existing = session.execute(text("""
                    SELECT id FROM book_chapters
                    WHERE book_id::text = :bid AND number = :num
                """), {"bid": _app._book_id, "num": num}).fetchone()
                if existing:
                    skipped += 1
                    continue
                sections_json = _json.dumps(ch.get("sections", []) or [])
                session.execute(text("""
                    INSERT INTO book_chapters
                        (book_id, number, title, description, topic_query, sections)
                    VALUES (CAST(:bid AS uuid), :num, :title,
                            :desc, :tq, CAST(:secs AS jsonb))
                """), {
                    "bid": _app._book_id, "num": num,
                    "title": ch.get("title") or f"Chapter {num}",
                    "desc": ch.get("description"),
                    "tq": ch.get("topic_query"),
                    "secs": sections_json,
                })
                inserted += 1
            session.commit()

        yield {"type": "completed", "n_chapters": len(chapters),
               "n_inserted": inserted, "n_skipped": skipped}

    threading.Thread(
        target=_app._run_generator_in_thread, args=(job_id, gen, loop), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


@router.post("/api/book/insert-introduction")
async def api_book_insert_introduction():
    """Phase 54.6.x — prepend an Introduction chapter at position 1.

    Standard front-matter for a scientific divulgation book: every
    existing chapter is renumbered ``+1`` and a new chapter ``number=1``
    titled "Introduction" is inserted with framing-style sections
    (motivation / scope / key terms / roadmap). Idempotent guard: if
    chapter 1 already looks like an introduction (title matches
    /introduction/i), the call is a no-op.

    Sections list mirrors the project-type Introduction template:
    "Motivation & Stakes", "Scope of the Argument", "Key Terms",
    "Roadmap of the Book". Existing drafts are NOT touched — they
    move with their chapters via the +1 number bump.
    """
    import json as _json
    from sciknow.web import app as _app

    intro_sections = [
        "Motivation & Stakes",
        "Scope of the Argument",
        "Key Terms",
        "Roadmap of the Book",
    ]

    with get_session() as session:
        chapters = session.execute(text("""
            SELECT number, title FROM book_chapters
            WHERE book_id::text = :bid ORDER BY number ASC
        """), {"bid": _app._book_id}).fetchall()

        if chapters and (chapters[0][1] or "").strip().lower().startswith(
            "introduction"
        ):
            return JSONResponse({
                "ok": True,
                "renumbered": 0,
                "noop": True,
                "message": "Chapter 1 already looks like an introduction.",
            })

        # Two-step renumber to dodge the (book_id, number) UNIQUE check:
        # bump every existing chapter by N+1 (out of the way) first,
        # then bring them back to (n+1) so chapter 1 is free.
        n_existing = len(chapters)
        if n_existing:
            session.execute(text("""
                UPDATE book_chapters
                   SET number = number + :bump
                 WHERE book_id::text = :bid
            """), {"bid": _app._book_id, "bump": n_existing + 100})
            session.execute(text("""
                UPDATE book_chapters
                   SET number = number - :unwind
                 WHERE book_id::text = :bid
            """), {"bid": _app._book_id, "unwind": n_existing + 100 - 1})

        session.execute(text("""
            INSERT INTO book_chapters
                (book_id, number, title, description, topic_query, sections)
            VALUES (CAST(:bid AS uuid), 1, :title, :desc, :tq,
                    CAST(:secs AS jsonb))
        """), {
            "bid": _app._book_id,
            "title": "Introduction",
            "desc": (
                "Frames the book's thesis, motivates the questions the "
                "rest of the book answers, and orients the reader. No "
                "substantive evidence here — that lives in later chapters."
            ),
            "tq": "introduction overview scope motivation",
            "secs": _json.dumps(intro_sections),
        })
        session.commit()

    return JSONResponse({
        "ok": True,
        "renumbered": n_existing,
        "noop": False,
        "message": (
            f"Introduction inserted as Ch.1. {n_existing} existing "
            f"chapter(s) renumbered to 2..{n_existing + 1}."
        ),
    })


@router.post("/api/book/auto-expand/preview")
async def api_book_auto_expand_preview(
    per_gap_limit: int = Form(100),
):
    """Phase 54.6.5 — preview corpus-expansion candidates derived from
    the current book's open gaps.

    Composition: book_gaps (open, type in {topic, evidence}) → per-gap
    topic search → merged + corpus-centroid-scored candidate list. Each
    candidate carries a ``gap_ids`` list so the UI can display how many
    open gaps the paper would close.
    """
    from sciknow.web import app as _app
    from sciknow.core.expand_ops import find_candidates_for_book_gaps
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: find_candidates_for_book_gaps(
                _app._book_id, per_gap_limit=int(per_gap_limit),
                score_relevance=True,
            ),
        )
    except Exception as exc:
        logger.exception("book/auto-expand preview failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return JSONResponse(result)


@router.post("/api/book/create")
async def api_book_create(request: Request):
    """Phase 46.F — web-side book creation with type selection.

    Runs inline (no subprocess) because book creation is fast (~50 ms)
    and doesn't need streaming progress. Mirrors ``sciknow book create
    --type=<slug>`` including the flat-type bootstrap that auto-creates
    chapter 1 with canonical sections for ``scientific_paper`` and
    other ``is_flat=True`` types.
    """
    import json as _json
    from sciknow.core.project_type import (
        default_sections_as_dicts, get_project_type, validate_type_slug,
    )

    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else dict(await request.form())
    title = (body.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    btype = (body.get("type") or "scientific_book").strip()
    try:
        validate_type_slug(btype)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    description = (body.get("description") or "").strip() or None
    try:
        tcw = int(body.get("target_chapter_words") or 0)
    except (TypeError, ValueError):
        tcw = 0
    bootstrap = (str(body.get("bootstrap", "true")).lower() in {"1", "true", "yes"})

    pt = get_project_type(btype)
    custom_meta: dict = {}
    effective_target = tcw if tcw > 0 else pt.default_target_chapter_words
    custom_meta["target_chapter_words"] = effective_target

    with get_session() as session:
        existing = session.execute(text(
            "SELECT id::text FROM books WHERE title = :t"
        ), {"t": title}).fetchone()
        if existing:
            raise HTTPException(status_code=409,
                                 detail=f"book already exists: {title}")
        row = session.execute(text("""
            INSERT INTO books (title, description, book_type, custom_metadata)
            VALUES (:t, :d, :bt, CAST(:m AS jsonb))
            RETURNING id::text
        """), {
            "t": title, "d": description, "bt": pt.slug,
            "m": _json.dumps(custom_meta),
        })
        book_id = row.fetchone()[0]

        chapter_id: str | None = None
        if bootstrap and pt.is_flat:
            sections_json = _json.dumps(default_sections_as_dicts(pt))
            cid_row = session.execute(text("""
                INSERT INTO book_chapters
                  (book_id, number, title, description, sections)
                VALUES
                  (CAST(:book_id AS uuid), 1, :ch_title, :ch_desc,
                   CAST(:sections AS jsonb))
                RETURNING id::text
            """), {
                "book_id": book_id,
                "ch_title": title,
                "ch_desc": description or f"{pt.display_name} — {title}",
                "sections": sections_json,
            })
            chapter_id = cid_row.fetchone()[0]
        session.commit()

    return JSONResponse({
        "ok": True,
        "book_id": book_id,
        "title": title,
        "book_type": pt.slug,
        "display_name": pt.display_name,
        "is_flat": pt.is_flat,
        "chapter_id_bootstrapped": chapter_id,
        "default_sections": [s.key for s in pt.default_sections],
    })


@router.get("/api/book/length-report")
async def api_book_length_report():
    """Phase 54.6.162 — web wrapper over core.length_report.walk_book_lengths.

    Surfaces the 54.6.153 CLI report in the GUI so users see the whole
    book's projected length without leaving the browser. Pure wrapper —
    no arithmetic duplication; the walker calls the real resolver
    helpers.
    """
    from sciknow.web import app as _app
    from sciknow.core.length_report import walk_book_lengths
    try:
        report = walk_book_lengths(_app._book_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("length-report failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse(report.to_dict())


@router.get("/api/book-types")
async def api_book_types():
    """Phase 54.6.147 — list all registered project types with their
    research-grounded length ranges, for the setup-wizard dropdown +
    info panel to render.

    Each entry is self-contained (no joins needed): display_name,
    description, chapter target, concept count range, words-per-
    concept range, and a derived section-at-midpoint range so the UI
    can show users what each type implies before they pick one.
    """
    from sciknow.core.project_type import list_project_types
    out = []
    for pt in list_project_types():
        clo, chi = pt.concepts_per_section_range
        wlo, whi = pt.words_per_concept_range
        wmid = (wlo + whi) // 2
        out.append({
            "slug": pt.slug,
            "display_name": pt.display_name,
            "description": pt.description,
            "is_flat": pt.is_flat,
            "default_chapter_count": pt.default_chapter_count,
            "default_target_chapter_words": pt.default_target_chapter_words,
            "concepts_per_section_range": [clo, chi],
            "words_per_concept_range":   [wlo, whi],
            "section_at_midpoint_range": [clo * wmid, chi * wmid],
            "default_sections": [
                {"key": s.key, "title": s.title, "target_words": s.target_words}
                for s in pt.default_sections
            ],
        })
    return JSONResponse({"types": out})
