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
                review_feedback=None, version=1):
    from sqlalchemy import text
    row = session.execute(text("""
        INSERT INTO drafts (title, book_id, chapter_id, section_type, topic, content,
                            word_count, sources, model_used, version, summary,
                            parent_draft_id, review_feedback)
        VALUES (:title, :book_id, :chapter_id, :section, :topic, :content,
                :wc, :sources::jsonb, :model, :version, :summary,
                :parent_id, :review_feedback)
        RETURNING id::text
    """), {
        "title": title, "book_id": book_id, "chapter_id": chapter_id,
        "section": section_type, "topic": topic, "content": content,
        "wc": len(content.split()),
        "sources": json.dumps(sources or []),
        "model": model, "version": version, "summary": summary,
        "parent_id": parent_draft_id, "review_feedback": review_feedback,
    })
    session.commit()
    result = row.fetchone()
    if not result:
        raise RuntimeError("Failed to save draft — INSERT returned no rows")
    return result[0]


def _retrieve(session, qdrant, query: str, candidate_k: int = 50,
              context_k: int = 12, topic_cluster: str | None = None,
              year_from: int | None = None, year_to: int | None = None,
              use_query_expansion: bool = False):
    """Run hybrid search + rerank + context build. Returns (results, sources)."""
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.rag.prompts import format_sources

    candidates = hybrid_search.search(
        query=query, qdrant_client=qdrant, session=session,
        candidate_k=candidate_k, year_from=year_from, year_to=year_to,
        topic_cluster=topic_cluster, use_query_expansion=use_query_expansion,
    )
    if not candidates:
        return [], []
    candidates = reranker.rerank(query, candidates, top_k=context_k)
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
        results, sources = _retrieve(
            session, qdrant, search_query,
            candidate_k=candidate_k, context_k=context_k,
            topic_cluster=topic_cluster, year_from=year_from,
            year_to=year_to, use_query_expansion=expand,
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
        import re as _re
        plan_clean = _re.sub(r'<think>.*?</think>\s*', '', plan_raw, flags=_re.DOTALL).strip()
        try:
            plan_data = json.loads(_clean_json(plan_clean), strict=False)
            yield {"type": "tree_plan", "data": plan_data, "raw": plan_clean}
        except Exception:
            yield {"type": "plan", "content": plan_clean}

    # Draft
    yield {"type": "progress", "stage": "writing",
           "detail": f"Drafting {section_type} for Ch.{ch_num}: {ch_title}..."}

    system, user = prompts.write_section_v2(
        section_type, topic, results,
        book_plan=b_plan, prior_summaries=prior_summaries,
    )

    content_tokens: list[str] = []
    for tok in llm_stream(system, user, model=model):
        content_tokens.append(tok)
        yield {"type": "token", "text": tok}

    content = "".join(content_tokens)

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

    # Save
    draft_id = None
    if save:
        yield {"type": "progress", "stage": "saving", "detail": "Generating summary and saving..."}
        summary = _auto_summarize(content, section_type, ch_title, model=model)
        draft_title = f"Ch.{ch_num} {ch_title} — {section_type.capitalize()}"
        with get_session() as session:
            draft_id = _save_draft(
                session, title=draft_title, book_id=book_id, chapter_id=ch_id,
                section_type=section_type, topic=topic, content=content,
                sources=sources, model=model, summary=summary,
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
                        :wc, :sources::jsonb, :model)
            """), {
                "title": draft_title, "book_id": book_id, "topic": claim,
                "content": content, "wc": len(content.split()),
                "sources": json.dumps(sources), "model": model,
            })
            session.commit()

    yield {"type": "completed", "content": content, "sources": sources}


# ── autowrite_section_stream ─────────────────────────────────────────────────

DEFAULT_SECTIONS = ["introduction", "methods", "results", "discussion", "conclusion"]


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


def autowrite_section_stream(
    book_id: str,
    chapter_id: str,
    section_type: str,
    *,
    model: str | None = None,
    max_iter: int = 3,
    target_score: float = 0.85,
    auto_expand: bool = False,
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

    yield {"type": "progress", "stage": "setup",
           "detail": f"Autowrite Ch.{ch_num}: {ch_title} — {section_type.capitalize()}"}

    # Step 1: Initial draft
    with get_session() as session:
        prior_summaries = _get_prior_summaries(session, book_id, ch_num)
        results, sources = _retrieve(
            session, qdrant, f"{section_type} {topic}",
            topic_cluster=topic_cluster,
        )

    if not results:
        yield {"type": "error", "message": "No relevant passages found."}
        return

    system, user = rag_prompts.write_section_v2(
        section_type, topic, results,
        book_plan=b_plan, prior_summaries=prior_summaries,
    )

    yield {"type": "progress", "stage": "writing", "detail": "Generating initial draft..."}
    tokens: list[str] = []
    for tok in llm_stream(system, user, model=model):
        tokens.append(tok)
        yield {"type": "token", "text": tok}

    content = "".join(tokens)

    # Step 2: Score -> verify -> revise -> re-score loop
    # Fix 3: `results` from the write step are passed to scorer/verifier
    #         so they evaluate against the same evidence the writer used.
    history: list[dict] = []

    for iteration in range(max_iter):
        yield {"type": "iteration_start", "iteration": iteration + 1, "max": max_iter}

        # Score (Fix 3: use writer's results, not re-retrieved)
        yield {"type": "progress", "stage": "scoring",
               "detail": f"Scoring iteration {iteration + 1}..."}
        try:
            scores = _score_draft_inner(
                content, section_type, topic, results, model=model,
            )
        except Exception as exc:
            logger.warning("Scoring failed: %s", exc)
            scores = {"overall": 0.5, "weakest_dimension": "unknown",
                      "revision_instruction": "Improve overall quality."}

        # Fix 1: Run claim verification as part of scoring
        yield {"type": "progress", "stage": "verifying",
               "detail": "Verifying citations..."}
        try:
            vdata = _verify_draft_inner(content, results, model=model)
            groundedness = vdata.get("groundedness_score", 1.0)
            # Merge groundedness into scores — if verification finds issues,
            # it can override the scorer's groundedness and force a revision
            if groundedness < scores.get("groundedness", 1.0):
                scores["groundedness"] = groundedness
            # If groundedness is the weakest dimension, target it
            if groundedness < scores.get(scores.get("weakest_dimension", ""), 1.0):
                scores["weakest_dimension"] = "groundedness"
                n_bad = len([c for c in vdata.get("claims", [])
                             if c.get("verdict") in ("EXTRAPOLATED", "MISREPRESENTED")])
                if n_bad:
                    scores["revision_instruction"] = (
                        f"Fix {n_bad} citation(s) flagged as extrapolated or misrepresented. "
                        "Ensure every [N] reference accurately reflects what the cited paper states."
                    )
            yield {"type": "verification", "data": vdata}
        except Exception as exc:
            logger.warning("Verification failed: %s", exc)

        overall = scores.get("overall", 0)
        weakest = scores.get("weakest_dimension", "unknown")
        instruction = scores.get("revision_instruction", "Improve the draft.")
        missing = scores.get("missing_topics", [])
        history.append(scores)

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
            yield {"type": "token", "text": tok}

        revised = "".join(rev_tokens)

        # Re-score (Fix 3: same results)
        yield {"type": "progress", "stage": "scoring", "detail": "Re-scoring revision..."}
        try:
            new_scores = _score_draft_inner(
                revised, section_type, topic, results, model=model,
            )
        except Exception:
            new_scores = {"overall": overall}

        new_overall = new_scores.get("overall", 0)

        if new_overall >= overall:
            yield {"type": "revision_verdict", "action": "KEEP",
                   "old_score": overall, "new_score": new_overall}
            content = revised
            overall = new_overall
        else:
            yield {"type": "revision_verdict", "action": "DISCARD",
                   "old_score": overall, "new_score": new_overall}

    # Step 3: Save
    yield {"type": "progress", "stage": "saving", "detail": "Saving final draft..."}
    summary = _auto_summarize(content, section_type, ch_title, model=model)

    draft_title = f"Ch.{ch_num} {ch_title} — {section_type.capitalize()} (autowrite)"
    with get_session() as session:
        draft_id = _save_draft(
            session, title=draft_title, book_id=book_id, chapter_id=ch_id,
            section_type=section_type, topic=topic, content=content,
            sources=sources, model=model, summary=summary,
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
