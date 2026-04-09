"""
Service layer for book operations.

Generator-based functions that yield typed event dicts, consumable by both the
CLI (Rich console) and the web layer (SSE endpoints).  Every function is a plain
Python generator — no asyncio dependency — so the web layer can run them inside
``asyncio.to_thread()`` and drain events through a queue.

Event types
-----------
- ``token``      — streaming LLM text fragment
- ``progress``   — stage/status update (human-readable)
- ``scores``     — structured quality scores from the reviewer
- ``completed``  — operation finished, includes draft_id / word_count / sources
- ``error``      — something went wrong
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator

logger = logging.getLogger("sciknow.core.book_ops")

# Type alias for the event dicts yielded by every generator.
Event = dict


# ── Shared helpers ────────────────────────────────────────────────────────────

def _get_book(session, title_or_id: str):
    from sqlalchemy import text
    return session.execute(text("""
        SELECT id::text, title, description, status, created_at, plan
        FROM books WHERE title ILIKE :q OR id::text LIKE :q
        LIMIT 1
    """), {"q": f"%{title_or_id}%"}).fetchone()


def _get_chapter(session, book_id: str, chapter_ref: str):
    from sqlalchemy import text
    if chapter_ref.isdigit():
        row = session.execute(text("""
            SELECT id::text, number, title, description, topic_query, topic_cluster
            FROM book_chapters WHERE book_id = :bid AND number = :num
        """), {"bid": book_id, "num": int(chapter_ref)}).fetchone()
        if row:
            return row
    return session.execute(text("""
        SELECT id::text, number, title, description, topic_query, topic_cluster
        FROM book_chapters WHERE book_id = :bid AND title ILIKE :q
        LIMIT 1
    """), {"bid": book_id, "q": f"%{chapter_ref}%"}).fetchone()


def _get_prior_summaries(session, book_id: str, before_chapter_number: int) -> list[dict]:
    from sqlalchemy import text
    rows = session.execute(text("""
        SELECT bc.number, d.section_type, d.summary
        FROM drafts d
        JOIN book_chapters bc ON bc.id = d.chapter_id
        WHERE d.book_id = :bid AND bc.number < :ch_num AND d.summary IS NOT NULL
        ORDER BY bc.number, d.section_type
    """), {"bid": book_id, "ch_num": before_chapter_number}).fetchall()
    return [{"chapter_number": r[0], "section_type": r[1], "summary": r[2]} for r in rows]


def _auto_summarize(content: str, section_type: str, chapter_title: str, model: str | None = None) -> str:
    from sciknow.rag import prompts
    from sciknow.rag.llm import complete
    system, user = prompts.draft_summary(section_type, chapter_title, content)
    try:
        return complete(system, user, model=model, temperature=0.1, num_ctx=4096).strip()
    except Exception as exc:
        logger.warning("Auto-summarize failed for %s/%s: %s", chapter_title, section_type, exc)
        return ""


def _save_draft(session, *, title, book_id, chapter_id, section_type, topic,
                content, sources, model, summary=None, parent_draft_id=None,
                review_feedback=None, version=1, custom_metadata=None):
    from sqlalchemy import text
    # Note: use CAST(:x AS jsonb) instead of :x::jsonb because SQLAlchemy's
    # parameter parser confuses `::jsonb` with the bound-param name and
    # passes it through unparameterized. CAST(...) is unambiguous.
    row = session.execute(text("""
        INSERT INTO drafts (title, book_id, chapter_id, section_type, topic, content,
                            word_count, sources, model_used, version, summary,
                            parent_draft_id, review_feedback, custom_metadata)
        VALUES (:title, :book_id, :chapter_id, :section, :topic, :content,
                :wc, CAST(:sources AS jsonb), :model, :version, :summary,
                :parent_id, :review_feedback, CAST(:metadata AS jsonb))
        RETURNING id::text
    """), {
        "title": title, "book_id": book_id, "chapter_id": chapter_id,
        "section": section_type, "topic": topic, "content": content,
        "wc": len(content.split()),
        "sources": json.dumps(sources or []),
        "model": model, "version": version, "summary": summary,
        "parent_id": parent_draft_id, "review_feedback": review_feedback,
        "metadata": json.dumps(custom_metadata or {}),
    })
    session.commit()
    result = row.fetchone()
    if not result:
        raise RuntimeError("Failed to save draft — INSERT returned no rows")
    return result[0]


def _release_gpu_models():
    """Free bge-m3 embedder + reranker from VRAM so Ollama can use the full GPU."""
    from sciknow.ingestion.embedder import release_model
    from sciknow.retrieval.hybrid_search import release_embed_model
    from sciknow.retrieval.reranker import release_reranker
    release_embed_model()
    release_model()
    release_reranker()


def _stream_phase(system: str, user: str, phase: str, *, model=None, **kw):
    """Phase 15.3 — generator that streams an LLM call as token events.

    Wraps llm.stream() so each token becomes
    {"type": "token", "text": tok, "phase": phase}, and returns the full
    accumulated text via the generator's StopIteration.value (captured by
    `yield from`).

    Used by autowrite_section_stream to make scoring/verification/CoVe/
    tree-plan calls visible to the GUI's live token counter. Without this,
    those phases use the synchronous llm.complete() and emit zero token
    events even though the LLM is working hard — the user sees a stuck
    "0 tok / 0 tok/s" while the timer counts up.

    Usage in a generator (re-yields events to consumer):
        raw = yield from _stream_phase(sys, usr, "scoring",
                                       model=model, num_ctx=16384)
        scores = json.loads(_clean_json(raw))
    """
    from sciknow.rag.llm import stream as llm_stream
    tokens: list[str] = []
    for tok in llm_stream(system, user, model=model, **kw):
        tokens.append(tok)
        yield {"type": "token", "text": tok, "phase": phase}
    return "".join(tokens)


def _update_draft_content(draft_id, content, *, summary=None, custom_metadata=None,
                          review_feedback=None, version=None):
    """Phase 15.1 — incremental update of an existing draft.

    Used by autowrite_section_stream and write_section_stream to persist
    intermediate state at each checkpoint, so a user clicking Stop never
    loses more than the in-flight iteration's tokens.

    Only fields you pass get updated; the rest are left untouched. The
    word_count is recomputed from the new content.
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    if not draft_id:
        return
    updates = ["content = :content", "word_count = :wc"]
    params: dict = {
        "did": draft_id, "content": content,
        "wc": len(content.split()),
    }
    if summary is not None:
        updates.append("summary = :summary")
        params["summary"] = summary
    if custom_metadata is not None:
        updates.append("custom_metadata = CAST(:metadata AS jsonb)")
        params["metadata"] = json.dumps(custom_metadata)
    if review_feedback is not None:
        updates.append("review_feedback = :rf")
        params["rf"] = review_feedback
    if version is not None:
        updates.append("version = :version")
        params["version"] = version

    with get_session() as session:
        session.execute(
            text(f"UPDATE drafts SET {', '.join(updates)} WHERE id::text = :did"),
            params,
        )
        session.commit()


def _retrieve(session, qdrant, query: str, candidate_k: int = 50,
              context_k: int = 12, topic_cluster: str | None = None,
              year_from: int | None = None, year_to: int | None = None,
              use_query_expansion: bool = False):
    """Run hybrid search + rerank + context build. Returns (results, sources).

    After retrieval completes, GPU models (bge-m3 + reranker) are released
    so the LLM has full VRAM available.
    """
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.rag.prompts import format_sources

    candidates = hybrid_search.search(
        query=query, qdrant_client=qdrant, session=session,
        candidate_k=candidate_k, year_from=year_from, year_to=year_to,
        topic_cluster=topic_cluster, use_query_expansion=use_query_expansion,
    )
    if not candidates:
        _release_gpu_models()
        return [], []
    candidates = reranker.rerank(query, candidates, top_k=context_k)
    _release_gpu_models()
    results = context_builder.build(candidates, session)
    sources = format_sources(results).splitlines()
    return results, sources


def _generate_step_back_query(query: str, model: str | None = None) -> str | None:
    """Ask the fast LLM for an abstract reformulation of `query`.

    Returns the step-back query, or None on failure. Uses the fast model
    because the abstraction step doesn't need the main model's reasoning
    horsepower and we want to keep latency low.
    """
    from sciknow.config import settings
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import complete as llm_complete
    sys_p, usr_p = rag_prompts.step_back(query)
    try:
        raw = llm_complete(
            sys_p, usr_p,
            model=model or settings.llm_fast_model,
            temperature=0.1, num_ctx=2048,
        )
        # Strip thinking blocks and any leading/trailing punctuation/quotes.
        raw = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()
        # Take only the first non-empty line and strip wrapping quotes.
        for line in raw.splitlines():
            line = line.strip().strip('"\'').strip()
            if line:
                return line
        return None
    except Exception as exc:
        logger.warning("Step-back query generation failed: %s", exc)
        return None


def _retrieve_with_step_back(
    session, qdrant, query: str,
    candidate_k: int = 50, context_k: int = 12,
    topic_cluster: str | None = None,
    year_from: int | None = None, year_to: int | None = None,
    use_query_expansion: bool = False,
    use_step_back: bool = True,
    model: str | None = None,
):
    """Retrieve with optional step-back query augmentation.

    Strategy: run the concrete query through hybrid search to get
    `candidate_k` candidates. If step-back is enabled, generate an
    abstract reformulation and retrieve `candidate_k // 2` more
    candidates for it. Union the candidate pools (deduping by chunk_id),
    then rerank the union against the ORIGINAL query and take top
    `context_k`. Reranking against the original query keeps the final
    relevance scoped to what the writer actually needs to draft, while
    the step-back pool brings in mechanism / background chunks the
    concrete query would have missed.

    Source: Zheng et al. "Take a Step Back: Evoking Reasoning via
    Abstraction in LLMs", ICLR 2024 (arXiv:2310.06117).
    """
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.rag.prompts import format_sources

    candidates = hybrid_search.search(
        query=query, qdrant_client=qdrant, session=session,
        candidate_k=candidate_k, year_from=year_from, year_to=year_to,
        topic_cluster=topic_cluster, use_query_expansion=use_query_expansion,
    )

    if use_step_back:
        sb_query = _generate_step_back_query(query, model=model)
        if sb_query and sb_query.lower() != query.lower():
            logger.info("Step-back query: %r → %r", query, sb_query)
            sb_candidates = hybrid_search.search(
                query=sb_query, qdrant_client=qdrant, session=session,
                candidate_k=max(candidate_k // 2, 10),
                year_from=year_from, year_to=year_to,
                topic_cluster=topic_cluster,
                use_query_expansion=False,  # already abstract; don't expand again
            )
            # Union by chunk_id, preserving the concrete-query candidates first.
            seen = {c.chunk_id for c in candidates}
            for c in sb_candidates:
                if c.chunk_id not in seen:
                    candidates.append(c)
                    seen.add(c.chunk_id)

    if not candidates:
        _release_gpu_models()
        return [], []

    # Rerank against the ORIGINAL (concrete) query so the final ordering
    # reflects what the writer actually needs.
    candidates = reranker.rerank(query, candidates, top_k=context_k)
    _release_gpu_models()
    results = context_builder.build(candidates, session)
    sources = format_sources(results).splitlines()
    return results, sources


def _clean_json(raw: str) -> str:
    """Strip markdown fences and extract the JSON object."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', '', cleaned)
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first >= 0 and last > first:
        cleaned = cleaned[first:last + 1]
    return cleaned


# ── write_section_stream ─────────────────────────────────────────────────────

def write_section_stream(
    book_id: str,
    chapter_id: str,
    section_type: str = "introduction",
    *,
    context_k: int = 12,
    candidate_k: int = 50,
    year_from: int | None = None,
    year_to: int | None = None,
    model: str | None = None,
    expand: bool = False,
    show_plan: bool = False,
    verify: bool = False,
    save: bool = True,
    use_step_back: bool = True,
) -> Iterator[Event]:
    """
    Draft a chapter section.  Yields events as the operation progresses.
    """
    from sciknow.rag import prompts
    from sciknow.rag.llm import stream as llm_stream, complete as llm_complete
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    yield {"type": "progress", "stage": "setup", "detail": "Loading book data..."}

    with get_session() as session:
        from sqlalchemy import text
        book = session.execute(text("""
            SELECT id::text, title, description, status, created_at, plan
            FROM books WHERE id::text = :bid
        """), {"bid": book_id}).fetchone()
        if not book:
            yield {"type": "error", "message": f"Book not found: {book_id}"}
            return

        ch = session.execute(text("""
            SELECT id::text, number, title, description, topic_query, topic_cluster
            FROM book_chapters WHERE id::text = :cid
        """), {"cid": chapter_id}).fetchone()
        if not ch:
            yield {"type": "error", "message": f"Chapter not found: {chapter_id}"}
            return

        b_plan = book[5]
        ch_id, ch_num, ch_title, ch_desc, topic_query, topic_cluster = ch
        search_query = f"{section_type} {topic_query or ch_title}"
        topic = topic_query or ch_title

        prior_summaries = _get_prior_summaries(session, book_id, ch_num)

    yield {"type": "progress", "stage": "retrieval", "detail": "Searching literature..."}

    qdrant = get_client()
    with get_session() as session:
        results, sources = _retrieve_with_step_back(
            session, qdrant, search_query,
            candidate_k=candidate_k, context_k=context_k,
            topic_cluster=topic_cluster, year_from=year_from,
            year_to=year_to, use_query_expansion=expand,
            use_step_back=use_step_back, model=model,
        )

    if not results:
        yield {"type": "error", "message": "No relevant passages found."}
        return

    # Optional hierarchical tree plan (TreeWriter pattern)
    if show_plan:
        yield {"type": "progress", "stage": "planning",
               "detail": "Creating hierarchical paragraph plan..."}
        sys_p, usr_p = prompts.tree_plan(
            section_type, topic, results,
            book_plan=b_plan, prior_summaries=prior_summaries,
        )
        plan_raw = ""
        for tok in llm_stream(sys_p, usr_p, model=model):
            plan_raw += tok
        # Strip thinking blocks and try to parse as JSON
        plan_clean = re.sub(r'<think>.*?</think>\s*', '', plan_raw, flags=re.DOTALL).strip()
        try:
            plan_data = json.loads(_clean_json(plan_clean), strict=False)
            paragraph_plan = plan_data.get("paragraphs") or []
            yield {"type": "tree_plan", "data": plan_data, "raw": plan_clean}
        except Exception:
            paragraph_plan = None
            yield {"type": "plan", "content": plan_clean}
    else:
        paragraph_plan = None

    # Draft — Phase 14.6: emit model info before the first token so the
    # user can confirm the writer is using the flagship model.
    from sciknow.config import settings as _settings
    resolved_model = model or _settings.llm_model
    yield {"type": "model_info", "writer_model": resolved_model,
           "fast_model": _settings.llm_fast_model,
           "writer_role": "writing this section",
           "fast_role": "step-back retrieval (utility)"}
    yield {"type": "progress", "stage": "writing",
           "detail": f"Drafting {section_type} for Ch.{ch_num}: {ch_title} · model: {resolved_model}"}

    system, user = prompts.write_section_v2(
        section_type, topic, results,
        book_plan=b_plan, prior_summaries=prior_summaries,
        paragraph_plan=paragraph_plan,
    )

    content_tokens: list[str] = []
    for tok in llm_stream(system, user, model=model):
        content_tokens.append(tok)
        yield {"type": "token", "text": tok}

    content = "".join(content_tokens)

    # Phase 15.1 — INCREMENTAL SAVE: persist the draft IMMEDIATELY after
    # token collection, before verification. If the user clicks Stop
    # during the verify step (which can take 30+ seconds for a 30B model),
    # the draft is already in the DB. The verification feedback gets
    # appended via _update_draft_content below.
    draft_id = None
    if save:
        draft_title = f"Ch.{ch_num} {ch_title} — {section_type.capitalize()}"
        with get_session() as session:
            draft_id = _save_draft(
                session, title=draft_title, book_id=book_id, chapter_id=ch_id,
                section_type=section_type, topic=topic, content=content,
                sources=sources, model=model, summary=None,
            )
        yield {"type": "checkpoint", "draft_id": draft_id, "stage": "draft",
               "word_count": len(content.split())}

    # Optional claim verification
    verify_feedback = None
    if verify:
        yield {"type": "progress", "stage": "verifying", "detail": "Verifying claims..."}
        sys_v, usr_v = prompts.verify_claims(content, results)
        try:
            raw = llm_complete(sys_v, usr_v, model=model, temperature=0.0, num_ctx=16384)
            vdata = json.loads(_clean_json(raw), strict=False)
            verify_feedback = raw
            yield {"type": "verification", "data": vdata}
        except Exception as exc:
            yield {"type": "progress", "stage": "verifying",
                   "detail": f"Verification failed: {exc}"}

    # Generate summary + finalize the draft (phases 15.1 — UPDATE existing
    # row instead of inserting a new one; the draft was already saved above).
    if save and draft_id:
        yield {"type": "progress", "stage": "saving", "detail": "Generating summary..."}
        summary = _auto_summarize(content, section_type, ch_title, model=model)
        _update_draft_content(
            draft_id, content,
            summary=summary,
            review_feedback=verify_feedback,
        )

    yield {
        "type": "completed",
        "draft_id": draft_id,
        "word_count": len(content.split()),
        "sources": sources,
        "content": content,
    }


# ── review_draft_stream ──────────────────────────────────────────────────────

def review_draft_stream(
    draft_id: str,
    *,
    model: str | None = None,
    save: bool = True,
) -> Iterator[Event]:
    """Run a critic pass over a saved draft."""
    from sqlalchemy import text
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    with get_session() as session:
        row = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.topic, d.content,
                   d.book_id::text, d.chapter_id::text
            FROM drafts d WHERE d.id::text LIKE :q
            LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()

    if not row:
        yield {"type": "error", "message": f"Draft not found: {draft_id}"}
        return

    d_id, d_title, d_section, d_topic, d_content, d_book_id, d_chapter_id = row

    yield {"type": "progress", "stage": "retrieval",
           "detail": f"Retrieving context for review of '{d_title}'..."}

    qdrant = get_client()
    search_query = f"{d_section or ''} {d_topic or d_title}"
    with get_session() as session:
        results, _ = _retrieve(session, qdrant, search_query, context_k=12)

    yield {"type": "progress", "stage": "reviewing",
           "detail": f"Reviewing: {d_title}..."}

    sys_r, usr_r = rag_prompts.review(d_section, d_topic or d_title, d_content, results)
    output_tokens: list[str] = []
    for tok in llm_stream(sys_r, usr_r, model=model):
        output_tokens.append(tok)
        yield {"type": "token", "text": tok}

    feedback = "".join(output_tokens).strip()

    if save:
        with get_session() as session:
            session.execute(text(
                "UPDATE drafts SET review_feedback = :fb WHERE id::text = :did"
            ), {"fb": feedback, "did": d_id})
            session.commit()

    yield {"type": "completed", "draft_id": d_id, "feedback": feedback}


# ── revise_draft_stream ──────────────────────────────────────────────────────

def revise_draft_stream(
    draft_id: str,
    *,
    instruction: str = "",
    context_k: int = 8,
    model: str | None = None,
) -> Iterator[Event]:
    """Revise a draft, creating a new version."""
    from sqlalchemy import text
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    with get_session() as session:
        row = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.topic, d.content,
                   d.book_id::text, d.chapter_id::text, d.version,
                   d.review_feedback, d.sources
            FROM drafts d WHERE d.id::text LIKE :q
            LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()

    if not row:
        yield {"type": "error", "message": f"Draft not found: {draft_id}"}
        return

    d_id, d_title, d_section, d_topic, d_content, d_book_id, d_chapter_id, \
        d_version, d_review_feedback, d_sources = row

    rev_instruction = instruction.strip()
    if not rev_instruction:
        if d_review_feedback:
            rev_instruction = f"Apply the following review feedback:\n\n{d_review_feedback}"
        else:
            yield {"type": "error", "message": "No instruction provided and no saved review feedback."}
            return

    # Retrieve additional passages
    results = []
    if context_k > 0:
        yield {"type": "progress", "stage": "retrieval", "detail": "Retrieving context..."}
        qdrant = get_client()
        search_query = f"{d_section or ''} {d_topic or d_title}"
        with get_session() as session:
            results, _ = _retrieve(session, qdrant, search_query, context_k=context_k)

    yield {"type": "progress", "stage": "revising",
           "detail": f"Revising {d_title} (v{d_version} -> v{d_version + 1})..."}

    sys_r, usr_r = rag_prompts.revise(d_content, rev_instruction, results or None)
    output_tokens: list[str] = []
    for tok in llm_stream(sys_r, usr_r, model=model):
        output_tokens.append(tok)
        yield {"type": "token", "text": tok}

    revised_content = "".join(output_tokens).strip()

    yield {"type": "progress", "stage": "saving", "detail": "Generating summary and saving..."}
    summary = _auto_summarize(revised_content, d_section or "text", d_title, model=model)

    existing_sources = json.loads(d_sources) if isinstance(d_sources, str) else (d_sources or [])

    with get_session() as session:
        new_id = _save_draft(
            session, title=d_title, book_id=d_book_id, chapter_id=d_chapter_id,
            section_type=d_section, topic=d_topic, content=revised_content,
            sources=existing_sources, model=model, summary=summary,
            parent_draft_id=d_id, version=d_version + 1,
        )

    yield {
        "type": "completed",
        "draft_id": new_id,
        "parent_draft_id": d_id,
        "version": d_version + 1,
        "word_count": len(revised_content.split()),
        "content": revised_content,
    }


# ── run_gaps_stream ──────────────────────────────────────────────────────────

def run_gaps_stream(
    book_id: str,
    *,
    model: str | None = None,
    save: bool = True,
) -> Iterator[Event]:
    """Run gap analysis on a book."""
    from sqlalchemy import text
    from sciknow.rag import prompts
    from sciknow.rag.llm import stream as llm_stream, complete as llm_complete
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = session.execute(text("""
            SELECT id::text, title FROM books WHERE id::text = :bid
        """), {"bid": book_id}).fetchone()
        if not book:
            yield {"type": "error", "message": f"Book not found: {book_id}"}
            return

        chapters = session.execute(text("""
            SELECT number, title, description FROM book_chapters
            WHERE book_id = :bid ORDER BY number
        """), {"bid": book_id}).fetchall()

        papers = session.execute(text(
            "SELECT title, year FROM paper_metadata ORDER BY year DESC NULLS LAST LIMIT 150"
        )).fetchall()

        drafts_rows = session.execute(text("""
            SELECT d.title, d.section_type, bc.number as chapter_number
            FROM drafts d LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.book_id = :bid ORDER BY bc.number, d.created_at
        """), {"bid": book_id}).fetchall()

    if not chapters:
        yield {"type": "error", "message": "No chapters defined. Run `book outline` first."}
        return

    ch_list = [{"number": c[0], "title": c[1], "description": c[2]} for c in chapters]
    p_list = [{"title": p[0], "year": p[1]} for p in papers if p[0]]
    d_list = [{"title": d[0], "section_type": d[1], "chapter_number": d[2]} for d in drafts_rows]

    # Pass 1: human-readable narrative (streamed)
    yield {"type": "progress", "stage": "analyzing", "detail": "Running gap analysis..."}
    system, user = prompts.gaps(
        book_title=book[1], chapters=ch_list, papers=p_list, drafts=d_list,
    )
    for tok in llm_stream(system, user, model=model, num_ctx=16384):
        yield {"type": "token", "text": tok}

    # Pass 2: structured JSON extraction
    if save:
        yield {"type": "progress", "stage": "extracting", "detail": "Extracting structured gaps..."}
        sys_j, usr_j = prompts.gaps_json(
            book_title=book[1], chapters=ch_list, papers=p_list, drafts=d_list,
        )
        try:
            raw = llm_complete(sys_j, usr_j, model=model, temperature=0.0, num_ctx=16384)
            gap_data = json.loads(_clean_json(raw), strict=False)
            gap_list = gap_data.get("gaps", [])
        except Exception as exc:
            yield {"type": "progress", "stage": "extracting",
                   "detail": f"Structured extraction failed: {exc}"}
            gap_list = []

        if gap_list:
            with get_session() as session:
                ch_rows = session.execute(text("""
                    SELECT number, id::text FROM book_chapters WHERE book_id = :bid
                """), {"bid": book_id}).fetchall()
                ch_id_map = {r[0]: r[1] for r in ch_rows}

                session.execute(text("DELETE FROM book_gaps WHERE book_id = :bid"),
                                {"bid": book_id})
                for g in gap_list:
                    ch_num = g.get("chapter_number")
                    session.execute(text("""
                        INSERT INTO book_gaps (book_id, gap_type, description, chapter_id, status)
                        VALUES (:bid, :gtype, :desc, :ch_id, 'open')
                    """), {
                        "bid": book_id,
                        "gtype": g.get("type", "topic"),
                        "desc": g.get("description", ""),
                        "ch_id": ch_id_map.get(ch_num) if ch_num else None,
                    })
                session.commit()

            yield {"type": "completed", "gaps_saved": len(gap_list), "gaps": gap_list}
        else:
            yield {"type": "completed", "gaps_saved": 0, "gaps": []}
    else:
        yield {"type": "completed", "gaps_saved": 0}


# ── run_argue_stream ─────────────────────────────────────────────────────────

def run_argue_stream(
    claim: str,
    *,
    book_id: str | None = None,
    context_k: int = 15,
    candidate_k: int = 60,
    year_from: int | None = None,
    year_to: int | None = None,
    model: str | None = None,
    save: bool = False,
) -> Iterator[Event]:
    """Map evidence for/against a claim."""
    from sciknow.rag import prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    yield {"type": "progress", "stage": "retrieval", "detail": "Searching literature..."}

    qdrant = get_client()
    with get_session() as session:
        results, sources = _retrieve(
            session, qdrant, claim,
            candidate_k=candidate_k, context_k=context_k,
            year_from=year_from, year_to=year_to,
        )

    if not results:
        yield {"type": "error", "message": "No relevant passages found."}
        return

    yield {"type": "progress", "stage": "arguing",
           "detail": f"Mapping argument: {claim[:60]}..."}

    system, user = prompts.argue(claim, results)
    output_tokens: list[str] = []
    for tok in llm_stream(system, user, model=model, num_ctx=16384):
        output_tokens.append(tok)
        yield {"type": "token", "text": tok}

    content = "".join(output_tokens)

    if save and book_id:
        from sqlalchemy import text
        draft_title = f"Argument Map: {claim[:60]}"
        with get_session() as session:
            session.execute(text("""
                INSERT INTO drafts (title, book_id, section_type, topic, content,
                                    word_count, sources, model_used)
                VALUES (:title, :book_id, 'argument_map', :topic, :content,
                        :wc, CAST(:sources AS jsonb), :model)
            """), {
                "title": draft_title, "book_id": book_id, "topic": claim,
                "content": content, "wc": len(content.split()),
                "sources": json.dumps(sources), "model": model,
            })
            session.commit()

    yield {"type": "completed", "content": content, "sources": sources}


# ── autowrite_section_stream ─────────────────────────────────────────────────

DEFAULT_SECTIONS = ["overview", "key_evidence", "current_understanding", "open_questions", "summary"]


def _score_draft_inner(draft_content, section_type, topic, results, model=None):
    """Score a draft against provided results. Returns scores_dict.

    Fix 3: the caller passes the writer's own results so the scorer
    evaluates against the same evidence the writer used.
    """
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import complete as llm_complete

    sys_s, usr_s = rag_prompts.score_draft(section_type, topic, draft_content, results)
    raw = llm_complete(sys_s, usr_s, model=model, temperature=0.0, num_ctx=16384)

    return json.loads(_clean_json(raw), strict=False)


def _verify_draft_inner(draft_content, results, model=None):
    """Run claim verification and return parsed verification data.

    Fix 1: integrated into the autowrite scoring loop so every iteration
    checks citation groundedness, not just when the user passes --verify.
    """
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import complete as llm_complete

    sys_v, usr_v = rag_prompts.verify_claims(draft_content, results)
    raw = llm_complete(sys_v, usr_v, model=model, temperature=0.0, num_ctx=16384)

    return json.loads(_clean_json(raw), strict=False)


def _cove_verify(draft_content, results, model=None, max_questions: int = 6):
    """Chain-of-Verification: decoupled fact-checking.

    1. Ask the LLM to generate falsifiable verification questions about the
       draft (sees ONLY the draft, not the sources).
    2. For each question, ask the LLM to answer it from the source passages
       alone (sees ONLY the sources, NOT the draft) — no anchoring.
    3. Compare independent answers to the draft's claims and return a
       structured report of mismatches.

    The whole point is the *decoupling* — the answerer for each question
    cannot be biased by the draft's framing because it never sees the draft.
    This catches the failure mode where the standard one-shot verifier
    rubberstamps a claim because it's reading the draft and the evidence
    in the same context window.

    Returns a dict:
        {
          "questions_asked": int,
          "mismatches": [
            {"question": ..., "draft_claim": ..., "draft_citation": ...,
             "independent_answer": ..., "verdict": ..., "severity": ...},
            ...
          ],
          "cove_score": float (1.0 - mismatches / questions_asked),
        }

    Source: Dhuliawala et al., "Chain-of-Verification Reduces Hallucination
    in Large Language Models", Findings of ACL 2024 (arXiv:2309.11495).
    """
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import complete as llm_complete

    # Step 1: Generate verification questions (sees only the draft).
    try:
        sys_q, usr_q = rag_prompts.cove_questions(draft_content)
        raw_q = llm_complete(sys_q, usr_q, model=model, temperature=0.1, num_ctx=16384)
        q_clean = re.sub(r'<think>.*?</think>\s*', '', raw_q, flags=re.DOTALL).strip()
        q_data = json.loads(_clean_json(q_clean), strict=False)
        questions = (q_data.get("questions") or [])[:max_questions]
    except Exception as exc:
        logger.warning("CoVe question generation failed: %s", exc)
        return {"questions_asked": 0, "mismatches": [], "cove_score": 1.0}

    if not questions:
        return {"questions_asked": 0, "mismatches": [], "cove_score": 1.0}

    # Step 2: For each question, answer it independently from the sources.
    # The answerer never sees the draft — that's the whole point.
    mismatches: list[dict] = []
    for q in questions:
        question_text = (q.get("question") or "").strip()
        draft_claim = (q.get("draft_claim") or "").strip()
        draft_citation = (q.get("citation") or "").strip()
        if not question_text:
            continue

        try:
            sys_a, usr_a = rag_prompts.cove_answer(question_text, results)
            raw_a = llm_complete(
                sys_a, usr_a, model=model, temperature=0.0, num_ctx=16384,
            )
            a_clean = re.sub(r'<think>.*?</think>\s*', '', raw_a, flags=re.DOTALL).strip()
            a_data = json.loads(_clean_json(a_clean), strict=False)
        except Exception as exc:
            logger.warning("CoVe answer failed for question %r: %s", question_text, exc)
            continue

        verdict = (a_data.get("verdict") or "").upper()
        independent_answer = (a_data.get("answer") or "").strip()
        notes = (a_data.get("notes") or "").strip()

        # A mismatch is any verdict that isn't a clean CONFIRMED.
        # NOT_IN_SOURCES means the draft made a claim the sources don't support.
        # DIFFERENT_SCOPE means the draft generalised past the source's stated scope.
        # PARTIAL means the source supports only part of what the draft asserts.
        if verdict and verdict != "CONFIRMED":
            severity = {
                "NOT_IN_SOURCES": "high",
                "DIFFERENT_SCOPE": "medium",
                "PARTIAL": "low",
            }.get(verdict, "medium")
            mismatches.append({
                "question": question_text,
                "draft_claim": draft_claim,
                "draft_citation": draft_citation,
                "independent_answer": independent_answer,
                "verdict": verdict,
                "severity": severity,
                "notes": notes,
            })

    cove_score = 1.0 - (len(mismatches) / max(len(questions), 1))
    return {
        "questions_asked": len(questions),
        "mismatches": mismatches,
        "cove_score": round(cove_score, 3),
    }


def _cove_verify_streaming(draft_content, results, model=None, max_questions: int = 6):
    """Phase 15.3 — generator variant of _cove_verify that yields token
    events for both the question generation pass and each independent
    answer pass. Returns the final mismatch report dict via
    StopIteration.value (captured by `yield from` in autowrite).

    Without this, CoVe runs 1 + N silent llm_complete calls and the GUI's
    token counter sits at 0 for the entire CoVe phase (often the longest
    silent stretch in autowrite).
    """
    from sciknow.rag import prompts as rag_prompts

    # Step 1 — generate verification questions, streamed
    try:
        sys_q, usr_q = rag_prompts.cove_questions(draft_content)
        raw_q = yield from _stream_phase(
            sys_q, usr_q, "cove_questions",
            model=model, temperature=0.1, num_ctx=16384,
        )
        q_clean = re.sub(r'<think>.*?</think>\s*', '', raw_q, flags=re.DOTALL).strip()
        q_data = json.loads(_clean_json(q_clean), strict=False)
        questions = (q_data.get("questions") or [])[:max_questions]
    except Exception as exc:
        logger.warning("CoVe question generation failed: %s", exc)
        return {"questions_asked": 0, "mismatches": [], "cove_score": 1.0}

    if not questions:
        return {"questions_asked": 0, "mismatches": [], "cove_score": 1.0}

    # Step 2 — independent answers, also streamed.
    mismatches: list[dict] = []
    for q_idx, q in enumerate(questions, 1):
        question_text = (q.get("question") or "").strip()
        draft_claim = (q.get("draft_claim") or "").strip()
        draft_citation = (q.get("citation") or "").strip()
        if not question_text:
            continue

        try:
            sys_a, usr_a = rag_prompts.cove_answer(question_text, results)
            raw_a = yield from _stream_phase(
                sys_a, usr_a, f"cove_answer_{q_idx}",
                model=model, temperature=0.0, num_ctx=16384,
            )
            a_clean = re.sub(r'<think>.*?</think>\s*', '', raw_a, flags=re.DOTALL).strip()
            a_data = json.loads(_clean_json(a_clean), strict=False)
        except Exception as exc:
            logger.warning("CoVe answer failed for question %r: %s", question_text, exc)
            continue

        verdict = (a_data.get("verdict") or "").upper()
        independent_answer = (a_data.get("answer") or "").strip()
        notes = (a_data.get("notes") or "").strip()

        if verdict and verdict != "CONFIRMED":
            severity = {
                "NOT_IN_SOURCES": "high",
                "DIFFERENT_SCOPE": "medium",
                "PARTIAL": "low",
            }.get(verdict, "medium")
            mismatches.append({
                "question": question_text,
                "draft_claim": draft_claim,
                "draft_citation": draft_citation,
                "independent_answer": independent_answer,
                "verdict": verdict,
                "severity": severity,
                "notes": notes,
            })

    cove_score = 1.0 - (len(mismatches) / max(len(questions), 1))
    return {
        "questions_asked": len(questions),
        "mismatches": mismatches,
        "cove_score": round(cove_score, 3),
    }


def autowrite_section_stream(
    book_id: str,
    chapter_id: str,
    section_type: str,
    *,
    model: str | None = None,
    max_iter: int = 3,
    target_score: float = 0.85,
    auto_expand: bool = False,
    use_plan: bool = True,
    use_step_back: bool = True,
    use_cove: bool = True,
    cove_threshold: float = 0.85,
) -> Iterator[Event]:
    """
    Full convergence loop for one section: write -> score -> revise -> re-score.
    Yields rich events for live dashboard rendering.
    """
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream, complete as llm_complete
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    qdrant = get_client()

    # Load book/chapter data
    with get_session() as session:
        from sqlalchemy import text
        book = session.execute(text("""
            SELECT id::text, title, plan FROM books WHERE id::text = :bid
        """), {"bid": book_id}).fetchone()
        ch = session.execute(text("""
            SELECT id::text, number, title, description, topic_query, topic_cluster
            FROM book_chapters WHERE id::text = :cid
        """), {"cid": chapter_id}).fetchone()

    if not book or not ch:
        yield {"type": "error", "message": "Book or chapter not found."}
        return

    b_title, b_plan = book[1], book[2]
    ch_id, ch_num, ch_title, ch_desc, topic_query, topic_cluster = ch
    topic = topic_query or ch_title

    # Phase 14.6 — Surface which model is going to do the writing so the
    # user never has to ask. resolved_model is what `llm.complete()` will
    # default to when called with model=None below; this mirrors the
    # `chosen_model = model or settings.llm_model` line in rag/llm.py.
    from sciknow.config import settings
    resolved_model = model or settings.llm_model

    yield {"type": "model_info", "writer_model": resolved_model,
           "fast_model": settings.llm_fast_model,
           "writer_role": "writing/scoring/verification/CoVe (flagship)",
           "fast_role": "step-back retrieval (utility)"}
    yield {"type": "progress", "stage": "setup",
           "detail": f"Autowrite Ch.{ch_num}: {ch_title} — {section_type.capitalize()} · model: {resolved_model}"}

    # Step 1: Initial draft
    with get_session() as session:
        prior_summaries = _get_prior_summaries(session, book_id, ch_num)
        results, sources = _retrieve_with_step_back(
            session, qdrant, f"{section_type} {topic}",
            topic_cluster=topic_cluster, model=model,
            use_step_back=use_step_back,
        )

    if not results:
        yield {"type": "error", "message": "No relevant passages found."}
        return

    # Optional tree plan with PDTB-lite discourse relations.
    paragraph_plan = None
    if use_plan:
        yield {"type": "progress", "stage": "planning",
               "detail": "Building paragraph plan with discourse relations..."}
        try:
            sys_p, usr_p = rag_prompts.tree_plan(
                section_type, topic, results,
                book_plan=b_plan, prior_summaries=prior_summaries,
            )
            # Phase 15.3 — stream so the token counter sees activity
            plan_raw = yield from _stream_phase(
                sys_p, usr_p, "planning",
                model=model, temperature=0.2, num_ctx=16384,
            )
            plan_clean = re.sub(r'<think>.*?</think>\s*', '', plan_raw,
                                flags=re.DOTALL).strip()
            plan_data = json.loads(_clean_json(plan_clean), strict=False)
            paragraph_plan = plan_data.get("paragraphs") or []
            yield {"type": "tree_plan", "data": plan_data}
        except Exception as exc:
            logger.warning("Tree-plan failed in autowrite, continuing without: %s", exc)
            paragraph_plan = None

    system, user = rag_prompts.write_section_v2(
        section_type, topic, results,
        book_plan=b_plan, prior_summaries=prior_summaries,
        paragraph_plan=paragraph_plan,
    )

    yield {"type": "progress", "stage": "writing", "detail": "Generating initial draft..."}
    tokens: list[str] = []
    for tok in llm_stream(system, user, model=model):
        tokens.append(tok)
        yield {"type": "token", "text": tok, "phase": "writing"}

    content = "".join(tokens)

    # Phase 15.1 — INCREMENTAL SAVE checkpoint #0: persist the initial draft
    # immediately, before any scoring/verification/revision. If the user
    # clicks Stop now (or the connection drops), at least the initial draft
    # is in the DB. The draft_id created here is reused by all subsequent
    # checkpoint updates so we always have one row per autowrite run.
    draft_title = f"Ch.{ch_num} {ch_title} — {section_type.capitalize()} (autowrite)"
    feature_versions = {
        "use_plan": use_plan,
        "use_step_back": use_step_back,
        "use_cove": use_cove,
        "cove_threshold": cove_threshold,
        "phase7_hedging_fidelity": True,
        "phase8_entity_bridge": True,
        "phase9_pdtb_discourse_relations": True,
        "phase10_step_back_retrieval": True,
        "phase11_chain_of_verification": True,
        "phase12_raptor_retrieval": True,
    }
    history: list[dict] = []
    initial_metadata = {
        "score_history": history,
        "feature_versions": feature_versions,
        "final_overall": None,
        "max_iter": max_iter,
        "target_score": target_score,
        "checkpoint": "initial",
    }
    with get_session() as session:
        draft_id = _save_draft(
            session, title=draft_title, book_id=book_id, chapter_id=ch_id,
            section_type=section_type, topic=topic, content=content,
            sources=sources, model=model, summary=None,
            version=1,
            custom_metadata=initial_metadata,
        )
    yield {"type": "checkpoint", "draft_id": draft_id, "stage": "initial",
           "word_count": len(content.split())}

    for iteration in range(max_iter):
        yield {"type": "iteration_start", "iteration": iteration + 1, "max": max_iter}

        # Score (Fix 3: use writer's results, not re-retrieved)
        # Phase 15.3: streamed via _stream_phase so the GUI's token counter
        # stays alive during this 30+ second phase. The output is JSON; the
        # user sees it flowing instead of staring at "0 tok / 0 tok/s".
        yield {"type": "progress", "stage": "scoring",
               "detail": f"Scoring iteration {iteration + 1}..."}
        try:
            sys_s, usr_s = rag_prompts.score_draft(section_type, topic, content, results)
            score_raw = yield from _stream_phase(
                sys_s, usr_s, "scoring",
                model=model, temperature=0.0, num_ctx=16384,
            )
            scores = json.loads(_clean_json(score_raw), strict=False)
        except Exception as exc:
            logger.warning("Scoring failed: %s", exc)
            scores = {"overall": 0.5, "weakest_dimension": "unknown",
                      "revision_instruction": "Improve overall quality."}

        # Fix 1: Run claim verification as part of scoring
        yield {"type": "progress", "stage": "verifying",
               "detail": "Verifying citations..."}
        try:
            sys_v, usr_v = rag_prompts.verify_claims(content, results)
            verify_raw = yield from _stream_phase(
                sys_v, usr_v, "verifying",
                model=model, temperature=0.0, num_ctx=16384,
            )
            vdata = json.loads(_clean_json(verify_raw), strict=False)
            groundedness = vdata.get("groundedness_score", 1.0)
            hedging = vdata.get("hedging_fidelity_score", 1.0)

            # Merge groundedness into scores — if verification finds issues,
            # it can override the scorer's groundedness and force a revision.
            if groundedness < scores.get("groundedness", 1.0):
                scores["groundedness"] = groundedness
            # Same for hedging fidelity (lexical-level groundedness).
            if hedging < scores.get("hedging_fidelity", 1.0):
                scores["hedging_fidelity"] = hedging

            # Re-evaluate weakest dimension across the merged scores.
            score_dims = {
                k: scores[k] for k in (
                    "groundedness", "completeness", "coherence",
                    "citation_accuracy", "hedging_fidelity",
                ) if k in scores and isinstance(scores[k], (int, float))
            }
            if score_dims:
                weakest_now = min(score_dims, key=score_dims.get)
                scores["weakest_dimension"] = weakest_now

                # Generate a targeted revision instruction if verification flags issues.
                claims = vdata.get("claims", [])
                n_extrap = sum(1 for c in claims if c.get("verdict") == "EXTRAPOLATED")
                n_misrep = sum(1 for c in claims if c.get("verdict") == "MISREPRESENTED")
                n_overst = sum(1 for c in claims if c.get("verdict") == "OVERSTATED")

                if weakest_now == "groundedness" and (n_extrap + n_misrep) > 0:
                    scores["revision_instruction"] = (
                        f"Fix {n_extrap + n_misrep} citation(s) flagged as extrapolated or "
                        "misrepresented. Ensure every [N] reference accurately reflects what "
                        "the cited paper states."
                    )
                elif weakest_now == "hedging_fidelity" and n_overst > 0:
                    overstated_examples = [
                        f"{c.get('citation', '[?]')}: {c.get('reason', '')[:120]}"
                        for c in claims if c.get("verdict") == "OVERSTATED"
                    ][:3]
                    scores["revision_instruction"] = (
                        f"Soften {n_overst} overstated claim(s) — restore the source's "
                        "epistemic strength (e.g. 'suggests' instead of 'proves', 'is "
                        "associated with' instead of 'causes'). Specific issues: "
                        + " | ".join(overstated_examples)
                    )

            yield {"type": "verification", "data": vdata}

            # Phase 2: Chain-of-Verification (decoupled fact check).
            # Gated to fire only when standard verification looks weak,
            # because CoVe issues 1 + N_questions extra LLM calls.
            should_run_cove = (
                use_cove
                and (
                    groundedness < cove_threshold
                    or hedging < cove_threshold
                    or scores.get("groundedness", 1.0) < cove_threshold
                    or scores.get("hedging_fidelity", 1.0) < cove_threshold
                )
            )
            if should_run_cove:
                yield {"type": "progress", "stage": "cove",
                       "detail": "Running Chain-of-Verification (decoupled fact check)..."}
                try:
                    # Phase 15.3: streamed CoVe so the GUI sees question
                    # generation + each independent answer flow as token
                    # events instead of going dark for 1 + N silent
                    # llm_complete calls.
                    cove = yield from _cove_verify_streaming(content, results, model=model)
                    yield {"type": "cove_verification", "data": cove}

                    high_sev = [m for m in cove.get("mismatches", [])
                                if m.get("severity") == "high"]
                    med_sev = [m for m in cove.get("mismatches", [])
                               if m.get("severity") == "medium"]

                    cove_score_v = cove.get("cove_score", 1.0)
                    # If CoVe is more pessimistic than the standard verifier
                    # for groundedness, take CoVe's number — independent
                    # re-answering is harder to fool.
                    if cove_score_v < scores.get("groundedness", 1.0):
                        scores["groundedness"] = cove_score_v
                        scores["weakest_dimension"] = "groundedness"

                    # If CoVe found high-severity mismatches, generate a
                    # targeted revision instruction that overrides the
                    # standard one. We name the specific claims so the
                    # writer can address them directly.
                    if high_sev:
                        examples = []
                        for m in high_sev[:3]:
                            claim_short = (m.get("draft_claim") or "")[:140]
                            ind_short = (m.get("independent_answer") or "")[:140]
                            examples.append(
                                f"CLAIM: {claim_short!r} — INDEPENDENT CHECK: {ind_short!r}"
                            )
                        scores["revision_instruction"] = (
                            f"Chain-of-Verification flagged {len(high_sev)} claim(s) "
                            "as NOT supported by the sources when checked independently. "
                            "Either remove these claims, or replace them with what the "
                            "sources actually say. Specific issues:\n  - "
                            + "\n  - ".join(examples)
                        )
                    elif med_sev and scores.get("weakest_dimension") in (
                        "groundedness", "hedging_fidelity",
                    ):
                        examples = []
                        for m in med_sev[:3]:
                            claim_short = (m.get("draft_claim") or "")[:140]
                            notes_short = (m.get("notes") or "")[:140]
                            examples.append(
                                f"CLAIM: {claim_short!r} — SCOPE: {notes_short!r}"
                            )
                        scores["revision_instruction"] = (
                            f"Chain-of-Verification flagged {len(med_sev)} claim(s) "
                            "where the source has a NARROWER scope than the draft "
                            "implies. Restore the source's scope qualifiers (region, "
                            "period, conditions). Specific issues:\n  - "
                            + "\n  - ".join(examples)
                        )
                except Exception as cove_exc:
                    logger.warning("CoVe verification failed: %s", cove_exc)
        except Exception as exc:
            logger.warning("Verification failed: %s", exc)

        overall = scores.get("overall", 0)
        weakest = scores.get("weakest_dimension", "unknown")
        instruction = scores.get("revision_instruction", "Improve the draft.")
        missing = scores.get("missing_topics", [])
        # Persist a richer history entry — not just the raw score dict, but
        # also the verification flag counts and the iteration index. This is
        # what `book draft scores` later reads back from custom_metadata.
        # vdata may be undefined if verification raised; locals() check is the
        # right way to detect that without depending on prior assignment.
        vdata_local = locals().get("vdata") or {}
        cove_local = locals().get("cove") or {}
        claims_local = vdata_local.get("claims") or []
        history_entry = {
            "iteration": iteration + 1,
            "scores": dict(scores),
            "verification": {
                "groundedness_score": vdata_local.get("groundedness_score"),
                "hedging_fidelity_score": vdata_local.get("hedging_fidelity_score"),
                "n_supported": sum(
                    1 for c in claims_local if c.get("verdict") == "SUPPORTED"
                ),
                "n_extrapolated": sum(
                    1 for c in claims_local if c.get("verdict") == "EXTRAPOLATED"
                ),
                "n_overstated": sum(
                    1 for c in claims_local if c.get("verdict") == "OVERSTATED"
                ),
                "n_misrepresented": sum(
                    1 for c in claims_local if c.get("verdict") == "MISREPRESENTED"
                ),
            },
            "cove": {
                "ran": bool(cove_local.get("questions_asked", 0)),
                "score": cove_local.get("cove_score"),
                "questions_asked": cove_local.get("questions_asked", 0),
                "n_high_severity": sum(
                    1 for m in (cove_local.get("mismatches") or [])
                    if m.get("severity") == "high"
                ),
                "n_medium_severity": sum(
                    1 for m in (cove_local.get("mismatches") or [])
                    if m.get("severity") == "medium"
                ),
            },
            "revision_verdict": None,  # filled in below if a revision happens
            "post_revision_overall": None,
        }
        history.append(history_entry)

        yield {"type": "scores", "scores": scores, "iteration": iteration + 1}

        # Convergence check
        if overall >= target_score:
            yield {"type": "converged", "iteration": iteration + 1,
                   "final_score": overall}
            break

        # Fix 4: Auto-expand — check if missing topics have weak corpus coverage
        if auto_expand and missing:
            yield {"type": "progress", "stage": "expanding",
                   "detail": f"Checking corpus coverage for: {', '.join(missing[:3])}"}
            weak_topics = []
            with get_session() as session:
                for topic_q in missing[:3]:
                    try:
                        extra_results, _ = _retrieve(
                            session, qdrant, topic_q, candidate_k=5, context_k=3,
                        )
                        if len(extra_results) < 3:
                            weak_topics.append(topic_q)
                    except Exception:
                        pass
            if weak_topics:
                yield {"type": "progress", "stage": "expanding",
                       "detail": f"Weak coverage: {', '.join(weak_topics)}. "
                                 f"Consider: sciknow db expand -q \"{weak_topics[0]}\""}

        # Revise (Fix 3: pass writer's results to revision context)
        yield {"type": "progress", "stage": "revising",
               "detail": f"Revising (targeting {weakest})..."}

        sys_r, usr_r = rag_prompts.revise(content, instruction, results)
        rev_tokens: list[str] = []
        for tok in llm_stream(sys_r, usr_r, model=model):
            rev_tokens.append(tok)
            yield {"type": "token", "text": tok, "phase": "revising"}

        revised = "".join(rev_tokens)

        # Re-score (Fix 3: same results) — Phase 15.3 streamed
        yield {"type": "progress", "stage": "scoring", "detail": "Re-scoring revision..."}
        try:
            sys_rs, usr_rs = rag_prompts.score_draft(section_type, topic, revised, results)
            rescore_raw = yield from _stream_phase(
                sys_rs, usr_rs, "rescoring",
                model=model, temperature=0.0, num_ctx=16384,
            )
            new_scores = json.loads(_clean_json(rescore_raw), strict=False)
        except Exception:
            new_scores = {"overall": overall}

        new_overall = new_scores.get("overall", 0)

        if new_overall >= overall:
            yield {"type": "revision_verdict", "action": "KEEP",
                   "old_score": overall, "new_score": new_overall}
            content = revised
            overall = new_overall
            if history:
                history[-1]["revision_verdict"] = "KEEP"
                history[-1]["post_revision_overall"] = new_overall
            # Phase 15.1 — INCREMENTAL SAVE checkpoint #N: persist the
            # accepted revision so the user never loses more than the
            # in-flight iteration's tokens if they click Stop.
            checkpoint_metadata = {
                "score_history": history,
                "feature_versions": feature_versions,
                "final_overall": overall,
                "max_iter": max_iter,
                "target_score": target_score,
                "checkpoint": f"iteration_{iteration + 1}_keep",
            }
            _update_draft_content(
                draft_id, content,
                custom_metadata=checkpoint_metadata,
                version=iteration + 2,
            )
            yield {"type": "checkpoint", "draft_id": draft_id,
                   "stage": f"iteration_{iteration + 1}_keep",
                   "word_count": len(content.split())}
        else:
            yield {"type": "revision_verdict", "action": "DISCARD",
                   "old_score": overall, "new_score": new_overall}
            if history:
                history[-1]["revision_verdict"] = "DISCARD"
                history[-1]["post_revision_overall"] = new_overall
            # Update metadata only — content stays at previous KEEP state.
            discard_metadata = {
                "score_history": history,
                "feature_versions": feature_versions,
                "final_overall": overall,
                "max_iter": max_iter,
                "target_score": target_score,
                "checkpoint": f"iteration_{iteration + 1}_discard",
            }
            _update_draft_content(draft_id, content, custom_metadata=discard_metadata)

    # Step 3: Final polish — auto-summarize and finalize the existing draft.
    # No new INSERT here because the draft was created at checkpoint #0 above.
    yield {"type": "progress", "stage": "saving", "detail": "Finalizing draft..."}
    summary = _auto_summarize(content, section_type, ch_title, model=model)
    persisted_metadata = {
        "score_history": history,
        "feature_versions": feature_versions,
        "final_overall": overall,
        "max_iter": max_iter,
        "target_score": target_score,
        "checkpoint": "final",
    }
    _update_draft_content(
        draft_id, content,
        summary=summary,
        custom_metadata=persisted_metadata,
        version=len(history) + 1,
    )

    yield {
        "type": "completed",
        "draft_id": draft_id,
        "word_count": len(content.split()),
        "iterations": len(history),
        "history": history,
        "final_score": overall,
    }
