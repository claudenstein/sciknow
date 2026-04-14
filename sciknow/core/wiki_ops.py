"""
Wiki knowledge layer — Karpathy-style compiled wiki over the paper corpus.

Generator-based functions that yield typed event dicts, consumable by both
the CLI (Rich console) and the web layer (SSE endpoints).

Instead of RAG on raw chunks every time, the wiki pre-synthesizes papers
into interconnected pages (paper summaries, concept pages, synthesis pages).
Pages are stored as markdown in data/wiki/, indexed in PostgreSQL, and
embedded in a Qdrant wiki collection for search.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sciknow.config import settings

logger = logging.getLogger("sciknow.wiki")

Event = dict


# ── Helpers ──────────────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Strip Qwen 3.x <think>...</think> blocks from LLM output."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    s = text.lower().strip()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s_]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s[:80]


def _ensure_wiki_dirs():
    """Create the wiki directory structure if it doesn't exist."""
    for subdir in ["papers", "concepts", "synthesis"]:
        (settings.wiki_dir / subdir).mkdir(parents=True, exist_ok=True)


def _load_existing_slugs(session) -> list[str]:
    """Return all existing wiki page slugs from the DB."""
    from sqlalchemy import text
    rows = session.execute(text("SELECT slug FROM wiki_pages ORDER BY slug")).fetchall()
    return [r[0] for r in rows]


def _save_page(session, *, slug: str, title: str, page_type: str,
               content: str, source_doc_ids: list[str],
               subdir: str) -> str:
    """Write markdown file + upsert DB row. Returns the slug."""
    from sqlalchemy import text

    _ensure_wiki_dirs()

    # Write markdown file
    filepath = settings.wiki_dir / subdir / f"{slug}.md"
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    header = (
        f"---\n"
        f"title: {title}\n"
        f"type: {page_type}\n"
        f"sources: {len(source_doc_ids)} papers\n"
        f"last_updated: {now_str}\n"
        f"---\n\n"
    )
    filepath.write_text(header + content, encoding="utf-8")

    # Upsert DB row
    existing = session.execute(text(
        "SELECT id::text FROM wiki_pages WHERE slug = :slug"
    ), {"slug": slug}).fetchone()

    wc = len(content.split())
    src_ids = "{" + ",".join(source_doc_ids) + "}" if source_doc_ids else None

    if existing:
        session.execute(text("""
            UPDATE wiki_pages SET title = :title, word_count = :wc,
                   source_doc_ids = CAST(:src AS uuid[]), needs_rewrite = 'false',
                   updated_at = now()
            WHERE slug = :slug
        """), {"title": title, "wc": wc, "src": src_ids, "slug": slug})
    else:
        session.execute(text("""
            INSERT INTO wiki_pages (slug, title, page_type, source_doc_ids, word_count)
            VALUES (:slug, :title, :ptype, CAST(:src AS uuid[]), :wc)
        """), {"slug": slug, "title": title, "ptype": page_type,
               "src": src_ids, "wc": wc})
    session.commit()

    return slug


def _embed_wiki_page(slug: str, content: str, page_type: str,
                     qdrant=None) -> str | None:
    """Embed a wiki page into the wiki Qdrant collection. Returns point_id."""
    from sciknow.storage.qdrant import WIKI_COLLECTION, get_client

    try:
        from FlagEmbedding import BGEM3FlagModel
        from sciknow.ingestion.embedder import _get_model

        if qdrant is None:
            qdrant = get_client()

        model = _get_model()
        output = model.encode(
            [content[:8000]],  # truncate for embedding
            return_dense=True,
            return_sparse=True,
        )

        point_id = str(uuid4())
        from qdrant_client.models import PointStruct, SparseVector

        sparse_data = output["lexical_weights"][0]
        sparse_indices = [int(k) for k in sparse_data.keys()]
        sparse_values = list(sparse_data.values())

        point = PointStruct(
            id=point_id,
            vector={
                "dense": output["dense_vecs"][0].tolist(),
                "sparse": SparseVector(indices=sparse_indices, values=sparse_values),
            },
            payload={
                "slug": slug,
                "page_type": page_type,
                "content_preview": content[:300],
            },
        )
        qdrant.upsert(collection_name=WIKI_COLLECTION, points=[point])

        # Update DB with qdrant_point_id
        from sciknow.storage.db import get_session
        from sqlalchemy import text
        with get_session() as session:
            session.execute(text(
                "UPDATE wiki_pages SET qdrant_point_id = CAST(:pid AS uuid) WHERE slug = :slug"
            ), {"pid": point_id, "slug": slug})
            session.commit()

        return point_id
    except Exception as exc:
        logger.warning("Failed to embed wiki page %s: %s", slug, exc)
        return None


def _clean_json(raw: str) -> str:
    """Strip markdown fences and extract JSON."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', '', cleaned)
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first >= 0 and last > first:
        cleaned = cleaned[first:last + 1]
    return cleaned


def _update_index():
    """Rebuild the index.md manifest from the wiki_pages table."""
    from sciknow.storage.db import get_session
    from sqlalchemy import text

    _ensure_wiki_dirs()

    with get_session() as session:
        rows = session.execute(text("""
            SELECT slug, title, page_type, word_count, updated_at
            FROM wiki_pages ORDER BY page_type, title
        """)).fetchall()

    lines = ["# Wiki Index\n", f"*{len(rows)} pages · Last updated {datetime.now(timezone.utc).strftime('%Y-%m-%d')}*\n"]

    current_type = None
    for slug, title, ptype, wc, updated in rows:
        if ptype != current_type:
            current_type = ptype
            lines.append(f"\n## {ptype.replace('_', ' ').title()}\n")
        lines.append(f"- [[{slug}]] — {title} ({wc or 0}w)")

    (settings.wiki_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _append_log(entry: str):
    """Append a timestamped entry to log.md."""
    _ensure_wiki_dirs()
    log_path = settings.wiki_dir / "log.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n## [{now}] {entry}\n")


# ── compile_paper_summary ────────────────────────────────────────────────────

def _wiki_model(model: str | None = None) -> str:
    """Default model for wiki operations — uses llm_fast_model for speed.

    Wiki pages are an exploration/browsing layer, not the final writing output.
    Book writing always uses the full model via the separate book_ops pipeline.
    """
    if model:
        return model
    return settings.llm_fast_model or settings.llm_model


def compile_paper_summary(
    doc_id: str,
    *,
    model: str | None = None,
    force: bool = False,
) -> Iterator[Event]:
    """Generate a wiki summary page for one paper."""
    from sqlalchemy import text
    from sciknow.rag import wiki_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.storage.db import get_session

    model = _wiki_model(model)

    with get_session() as session:
        # Load paper metadata
        meta = session.execute(text("""
            SELECT pm.title, pm.abstract, pm.year, pm.journal, pm.doi,
                   pm.authors, pm.keywords, pm.domains, pm.document_id::text
            FROM paper_metadata pm WHERE pm.document_id::text = :did
        """), {"did": doc_id}).fetchone()

        if not meta:
            yield {"type": "error", "message": f"Paper not found: {doc_id}"}
            return

        title, abstract, year, journal, doi, authors, keywords, domains, _ = meta

        slug = _slugify(f"{doc_id[:8]}-{title or 'untitled'}")

        # Check if already compiled (skip unless force)
        if not force:
            existing = session.execute(text(
                "SELECT id FROM wiki_pages WHERE slug = :slug"
            ), {"slug": slug}).fetchone()
            if existing:
                yield {"type": "progress", "stage": "skip",
                       "detail": f"Already compiled: {slug}"}
                yield {"type": "completed", "slug": slug, "skipped": True}
                return

        # Load section contents
        sections = session.execute(text("""
            SELECT section_type, content FROM paper_sections
            WHERE document_id::text = :did ORDER BY section_index
        """), {"did": doc_id}).fetchall()
        section_text = "\n\n".join(
            f"[{s[0]}]\n{s[1][:3000]}" for s in sections
        )[:12000]

        existing_slugs = _load_existing_slugs(session)

    # Format metadata
    author_str = ", ".join(
        a.get("name", "") if isinstance(a, dict) else str(a)
        for a in (authors or [])[:5]
    )
    kw_str = ", ".join(keywords or [])
    dom_str = ", ".join(domains or [])

    yield {"type": "progress", "stage": "writing",
           "detail": f"Compiling summary: {title or doc_id[:8]}..."}

    system, user = wiki_prompts.wiki_paper_summary(
        title=title, authors=author_str, year=str(year or "n.d."),
        journal=journal or "", doi=doi or "",
        keywords=kw_str, domains=dom_str,
        abstract=abstract or "", sections=section_text,
        existing_slugs=existing_slugs,
    )

    tokens: list[str] = []
    for tok in llm_stream(system, user, model=model, keep_alive=-1):
        tokens.append(tok)
        yield {"type": "token", "text": tok}

    content = _strip_thinking("".join(tokens))

    # Save
    with get_session() as session:
        _save_page(
            session, slug=slug, title=title or "Untitled",
            page_type="paper_summary", content=content,
            source_doc_ids=[doc_id], subdir="papers",
        )

    # Embed (may fail if GPU VRAM is full — non-fatal)
    point_id = _embed_wiki_page(slug, content, "paper_summary")
    if not point_id:
        yield {"type": "progress", "stage": "warning",
               "detail": "Embedding skipped (GPU VRAM full)"}

    # Combined entity + KG extraction in a single LLM call (speedup: 2 calls → 1)
    yield {"type": "progress", "stage": "extracting",
           "detail": "Extracting entities + knowledge graph..."}
    entities, kg_count = _extract_entities_and_kg(
        doc_id, slug, title, author_str, str(year or "n.d."),
        kw_str, dom_str, abstract or "", section_text,
        existing_slugs, model=model,
    )

    _append_log(f"ingest | {title} → [[{slug}]] ({kg_count} triples, {len(entities)} entities)")

    yield {"type": "completed", "slug": slug, "word_count": len(content.split()),
           "kg_triples": kg_count, "entities": entities}


def _extract_entities_and_kg(
    doc_id, slug, title, authors, year, keywords, domains,
    abstract, sections, existing_slugs, model=None,
) -> tuple[list[str], int]:
    """Combined entity + KG triple extraction in one LLM call."""
    from sciknow.rag import wiki_prompts
    from sciknow.rag.llm import complete as llm_complete
    from sciknow.storage.db import get_session
    from sqlalchemy import text

    sys_e, usr_e = wiki_prompts.wiki_extract_entities(
        title=title, authors=authors, year=year,
        keywords=keywords, domains=domains,
        abstract=abstract, existing_slugs=existing_slugs,
        slug=slug, sections=sections,
    )

    # Structured output schema — Ollama guarantees valid JSON matching this
    extraction_schema = {
        "type": "object",
        "properties": {
            "concepts": {"type": "array", "items": {"type": "string"}},
            "methods": {"type": "array", "items": {"type": "string"}},
            "datasets": {"type": "array", "items": {"type": "string"}},
            "triples": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    # Phase 48d — verbatim source sentence. Not required
                    # (the LLM may legitimately fail to pin one); empty
                    # string is acceptable and flows through as NULL.
                    "source_sentence": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
            }},
        },
        "required": ["concepts", "methods", "datasets", "triples"],
    }

    try:
        raw = llm_complete(sys_e, usr_e, model=model,
                           temperature=0.0, num_ctx=8192,
                           keep_alive=-1, format=extraction_schema)
        # With structured output, no need for _strip_thinking or _clean_json
        data = json.loads(raw, strict=False)
    except Exception as exc:
        logger.warning("Entity+KG extraction failed for %s: %s", slug, exc)
        return [], 0

    # Save entities (concepts list)
    all_entities = (
        data.get("concepts", []) +
        data.get("methods", []) +
        data.get("datasets", [])
    )

    # Save KG triples
    triples = data.get("triples", [])
    kg_count = 0
    if triples:
        with get_session() as session:
            session.execute(text(
                "DELETE FROM knowledge_graph WHERE source_doc_id::text = :did"
            ), {"did": doc_id})
            for t in triples:
                subj = (t.get("subject") or "").strip().lower()[:200]
                pred = (t.get("predicate") or "").strip().lower()[:100]
                obj = (t.get("object") or "").strip().lower()[:200]
                # Phase 48d — empty / whitespace-only source sentence
                # flows through as NULL so the UI can render "(no
                # source sentence)" consistently instead of "".
                sent_raw = (t.get("source_sentence") or "").strip()[:500]
                sent = sent_raw if sent_raw else None
                if subj and pred and obj:
                    session.execute(text("""
                        INSERT INTO knowledge_graph
                            (subject, predicate, object, source_doc_id, source_sentence)
                        VALUES (:subj, :pred, :obj, CAST(:did AS uuid), :sent)
                    """), {"subj": subj, "pred": pred, "obj": obj,
                            "did": doc_id, "sent": sent})
                    kg_count += 1
            session.commit()

    # Create concept page stubs (no LLM call — just filesystem + DB)
    for entity_slug_raw in all_entities:
        eslug = _slugify(entity_slug_raw)
        if not eslug:
            continue
        concept_path = settings.wiki_dir / "concepts" / f"{eslug}.md"
        if not concept_path.exists():
            _ensure_wiki_dirs()
            stub = f"# {entity_slug_raw.replace('-', ' ').title()}\n\n*Mentioned in: {title} ({year})*\n"
            with get_session() as session:
                _save_page(
                    session, slug=eslug,
                    title=entity_slug_raw.replace("-", " ").title(),
                    page_type="concept", content=stub,
                    source_doc_ids=[doc_id], subdir="concepts",
                )

    return all_entities, kg_count


# ── update_concepts_for_paper ────────────────────────────────────────────────

def update_concepts_for_paper(
    doc_id: str,
    *,
    model: str | None = None,
) -> Iterator[Event]:
    """Post-ingest hook: extract entities from a paper and update concept pages."""
    from sqlalchemy import text
    from sciknow.rag import wiki_prompts
    from sciknow.rag.llm import complete as llm_complete
    from sciknow.storage.db import get_session

    # Use the same model as the caller (avoids Ollama model-swap OOM).
    # Previously used llm_fast_model, but on a single GPU the swap from
    # 27B → 7B → 27B causes 500 errors ~20% of the time.
    fast_model = model

    with get_session() as session:
        meta = session.execute(text("""
            SELECT pm.title, pm.abstract, pm.year, pm.authors, pm.keywords, pm.domains
            FROM paper_metadata pm WHERE pm.document_id::text = :did
        """), {"did": doc_id}).fetchone()

        if not meta:
            yield {"type": "error", "message": f"Paper not found: {doc_id}"}
            return

        title, abstract, year, authors, keywords, domains = meta
        existing_slugs = _load_existing_slugs(session)

    author_str = ", ".join(
        a.get("name", "") if isinstance(a, dict) else str(a)
        for a in (authors or [])[:5]
    )
    kw_str = ", ".join(keywords or [])
    dom_str = ", ".join(domains or [])

    # Extract entities
    yield {"type": "progress", "stage": "extracting",
           "detail": f"Extracting entities from {title or doc_id[:8]}..."}

    sys_e, usr_e = wiki_prompts.wiki_extract_entities(
        title=title, authors=author_str, year=str(year or "n.d."),
        keywords=kw_str, domains=dom_str,
        abstract=abstract or "", existing_slugs=existing_slugs,
    )

    try:
        raw = _strip_thinking(llm_complete(sys_e, usr_e, model=fast_model, temperature=0.0, num_ctx=8192))
        entities = json.loads(_clean_json(raw), strict=False)
    except Exception as exc:
        yield {"type": "error", "message": f"Entity extraction failed: {exc}"}
        return

    all_entities = (
        entities.get("concepts", []) +
        entities.get("methods", []) +
        entities.get("datasets", [])
    )

    if not all_entities:
        yield {"type": "completed", "concepts_updated": 0}
        return

    yield {"type": "progress", "stage": "updating",
           "detail": f"Updating {len(all_entities)} concept pages..."}

    # Get a relevant passage from the paper for context
    with get_session() as session:
        abstract_row = session.execute(text("""
            SELECT abstract FROM paper_metadata
            WHERE document_id::text = :did
        """), {"did": doc_id}).fetchone()
    passage = (abstract_row[0] if abstract_row and abstract_row[0] else "")[:2000]

    updated = 0
    for entity_slug in all_entities:
        slug = _slugify(entity_slug)
        if not slug:
            continue

        # Check if concept page exists
        concept_path = settings.wiki_dir / "concepts" / f"{slug}.md"

        if concept_path.exists():
            # Append update using fast model
            sys_a, usr_a = wiki_prompts.wiki_concept_append(
                concept_name=entity_slug,
                paper_title=title or "Unknown",
                paper_year=str(year or "n.d."),
                passage=passage,
            )
            try:
                addition = _strip_thinking(llm_complete(sys_a, usr_a, model=fast_model,
                                        temperature=0.1, num_ctx=4096))
                # Append to existing file
                existing_content = concept_path.read_text(encoding="utf-8")
                updated_content = existing_content.rstrip() + "\n\n" + addition.strip() + "\n"
                concept_path.write_text(updated_content, encoding="utf-8")

                # Update DB
                with get_session() as session:
                    session.execute(text("""
                        UPDATE wiki_pages SET
                            word_count = :wc,
                            source_doc_ids = array_append(source_doc_ids, CAST(:did AS uuid)),
                            updated_at = now()
                        WHERE slug = :slug
                    """), {"wc": len(updated_content.split()), "did": doc_id, "slug": slug})
                    session.commit()

                updated += 1
            except Exception as exc:
                logger.warning("Failed to update concept %s: %s", slug, exc)
        else:
            # Create stub page
            stub = (
                f"# {entity_slug.replace('-', ' ').title()}\n\n"
                f"**{title}** ({year or 'n.d.'}) — {passage[:200]}...\n"
            )
            _ensure_wiki_dirs()
            with get_session() as session:
                _save_page(
                    session, slug=slug,
                    title=entity_slug.replace("-", " ").title(),
                    page_type="concept", content=stub,
                    source_doc_ids=[doc_id], subdir="concepts",
                )
            updated += 1

    _update_index()
    _append_log(f"concepts | Updated {updated} concepts from: {title}")

    yield {"type": "completed", "concepts_updated": updated}


# ── compile_all ──────────────────────────────────────────────────────────────

def compile_all(
    *,
    model: str | None = None,
    force: bool = False,
    rewrite_stale: bool = False,
) -> Iterator[Event]:
    """Build the full wiki from all ingested papers."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        rows = session.execute(text("""
            SELECT d.id::text, pm.title
            FROM documents d
            JOIN paper_metadata pm ON pm.document_id = d.id
            WHERE d.ingestion_status = 'complete' AND pm.title IS NOT NULL
            ORDER BY CASE WHEN d.ingest_source = 'seed' THEN 0 ELSE 1 END,
                     pm.year DESC NULLS LAST
        """)).fetchall()

    total = len(rows)
    yield {"type": "compile_start", "total": total}

    compiled = 0
    skipped = 0
    failed = 0
    total_tokens = 0
    for i, (doc_id, title) in enumerate(rows, 1):
        short_title = (title or doc_id[:8])[:60]
        paper_t0 = time.monotonic()
        paper_tokens = 0

        yield {"type": "paper_start", "index": i, "total": total,
               "title": short_title, "doc_id": doc_id}

        # Compile paper summary — pass through token events for live display
        paper_ok = False
        paper_skipped = False
        for event in compile_paper_summary(doc_id, model=model, force=force):
            if event.get("type") == "token":
                paper_tokens += 1
                total_tokens += 1
                yield {"type": "token", "text": event["text"],
                       "paper_tokens": paper_tokens, "total_tokens": total_tokens}
            elif event.get("type") == "error":
                failed += 1
                yield {"type": "paper_done", "index": i, "total": total,
                       "title": short_title, "status": "error",
                       "detail": event.get("message", ""),
                       "tokens": paper_tokens, "elapsed": time.monotonic() - paper_t0,
                       "compiled": compiled, "skipped": skipped, "failed": failed,
                       "total_tokens": total_tokens}
                break
            elif event.get("type") == "completed":
                if event.get("skipped"):
                    skipped += 1
                    paper_skipped = True
                else:
                    compiled += 1
                    paper_ok = True

        if paper_skipped:
            yield {"type": "paper_done", "index": i, "total": total,
                   "title": short_title, "status": "skipped",
                   "compiled": compiled, "skipped": skipped, "failed": failed,
                   "tokens": 0, "elapsed": 0, "total_tokens": total_tokens}
            continue

        # Concept stubs + KG triples are now created inside compile_paper_summary
        # (merged into a single LLM call). Full concept page content is deferred
        # to keep the bulk compile fast. Run `wiki compile --rewrite-stale` later
        # to flesh out concept pages with full LLM-written content.

        paper_elapsed = time.monotonic() - paper_t0
        yield {"type": "paper_done", "index": i, "total": total,
               "title": short_title,
               "status": "compiled" if paper_ok else "error",
               "compiled": compiled, "skipped": skipped, "failed": failed,
               "tokens": paper_tokens, "elapsed": paper_elapsed,
               "total_tokens": total_tokens}

    _update_index()
    _append_log(f"compile-all | {compiled} new, {skipped} skipped, {failed} failed from {total} papers")

    yield {"type": "completed", "compiled": compiled, "skipped": skipped,
           "failed": failed, "total": total}


# ── compile_synthesis ────────────────────────────────────────────────────────

def compile_synthesis(
    topic: str,
    *,
    model: str | None = None,
) -> Iterator[Event]:
    """Generate a synthesis page on a topic from existing wiki pages."""
    model = _wiki_model(model)
    from sciknow.rag import wiki_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.storage.db import get_session
    from sqlalchemy import text

    slug = _slugify(f"synthesis-{topic}")

    yield {"type": "progress", "stage": "gathering",
           "detail": f"Gathering wiki pages for: {topic}..."}

    # Find relevant paper summaries and concept pages
    with get_session() as session:
        existing_slugs = _load_existing_slugs(session)

    # Read paper summary files
    paper_summaries = []
    papers_dir = settings.wiki_dir / "papers"
    if papers_dir.exists():
        for f in sorted(papers_dir.glob("*.md"))[:30]:
            content = f.read_text(encoding="utf-8")
            if topic.lower() in content.lower():
                paper_summaries.append(content[:2000])

    # Read concept pages
    concept_pages = []
    concepts_dir = settings.wiki_dir / "concepts"
    if concepts_dir.exists():
        for f in sorted(concepts_dir.glob("*.md"))[:20]:
            content = f.read_text(encoding="utf-8")
            if topic.lower() in content.lower():
                concept_pages.append(content[:1500])

    yield {"type": "progress", "stage": "writing",
           "detail": f"Synthesizing {len(paper_summaries)} papers, {len(concept_pages)} concepts..."}

    system, user = wiki_prompts.wiki_synthesis(
        topic=topic,
        paper_summaries="\n\n---\n\n".join(paper_summaries),
        concept_pages="\n\n---\n\n".join(concept_pages),
        existing_slugs=existing_slugs,
    )

    tokens: list[str] = []
    for tok in llm_stream(system, user, model=model, num_ctx=16384):
        tokens.append(tok)
        yield {"type": "token", "text": tok}

    content = _strip_thinking("".join(tokens))

    with get_session() as session:
        _save_page(
            session, slug=slug, title=f"{topic}: State of Research",
            page_type="synthesis", content=content,
            source_doc_ids=[], subdir="synthesis",
        )

    _embed_wiki_page(slug, content, "synthesis")
    _update_index()
    _append_log(f"synthesis | {topic} → [[{slug}]]")

    yield {"type": "completed", "slug": slug, "word_count": len(content.split())}


# ── query_wiki ───────────────────────────────────────────────────────────────

def query_wiki(
    question: str,
    *,
    context_k: int = 8,
    model: str | None = None,
) -> Iterator[Event]:
    """Search the wiki collection and answer from compiled pages."""
    model = _wiki_model(model)
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.storage.qdrant import WIKI_COLLECTION, get_client

    yield {"type": "progress", "stage": "searching", "detail": "Searching wiki..."}

    qdrant = get_client()

    try:
        from sciknow.ingestion.embedder import _get_model
        emb_model = _get_model()
        output = emb_model.encode([question], return_dense=True, return_sparse=True)

        from qdrant_client.models import SparseVector
        sparse_data = output["lexical_weights"][0]

        # Dense search
        dense_hits = qdrant.query_points(
            collection_name=WIKI_COLLECTION,
            query=output["dense_vecs"][0].tolist(),
            using="dense",
            limit=context_k,
        ).points

        # Sparse search
        sparse_hits = qdrant.query_points(
            collection_name=WIKI_COLLECTION,
            query=SparseVector(
                indices=[int(k) for k in sparse_data.keys()],
                values=list(sparse_data.values()),
            ),
            using="sparse",
            limit=context_k,
        ).points

    except Exception as exc:
        yield {"type": "error", "message": f"Wiki search failed: {exc}"}
        return

    # Merge and deduplicate hits
    seen = set()
    all_hits = []
    for hit in dense_hits + sparse_hits:
        slug = hit.payload.get("slug", "")
        if slug not in seen:
            seen.add(slug)
            all_hits.append(hit)

    if not all_hits:
        yield {"type": "error", "message": "No relevant wiki pages found."}
        return

    # Read the actual wiki page content
    yield {"type": "progress", "stage": "reading",
           "detail": f"Reading {len(all_hits)} wiki pages..."}

    context_parts = []
    for i, hit in enumerate(all_hits[:context_k], 1):
        slug = hit.payload.get("slug", "")
        ptype = hit.payload.get("page_type", "")

        # Find the markdown file
        for subdir in ["papers", "concepts", "synthesis"]:
            path = settings.wiki_dir / subdir / f"{slug}.md"
            if path.exists():
                content = path.read_text(encoding="utf-8")
                # Strip YAML front matter
                if content.startswith("---"):
                    content = content.split("---", 2)[-1].strip()
                context_parts.append(f"[{i}] ({ptype}) {content[:3000]}")
                break

    context = "\n\n---\n\n".join(context_parts)

    # Answer from wiki context
    system = (
        "You are a scientific research assistant answering from a compiled knowledge wiki. "
        "The wiki pages below are pre-synthesized summaries of scientific papers and concepts. "
        "Answer using ONLY the provided wiki pages. Cite pages as [N]. "
        "If the wiki doesn't cover the question, say so clearly."
    )
    user = f"Wiki pages:\n\n{context}\n\n---\n\nQuestion: {question}"

    yield {"type": "progress", "stage": "answering", "detail": "Generating answer..."}

    for tok in llm_stream(system, user, model=model, num_ctx=16384):
        yield {"type": "token", "text": tok}

    # List sources
    sources = [f"[{i+1}] {hit.payload.get('slug', '')}" for i, hit in enumerate(all_hits[:context_k])]
    yield {"type": "completed", "sources": sources}


# ── consensus_map ────────────────────────────────────────────────────────────

def consensus_map(
    topic: str,
    *,
    model: str | None = None,
) -> Iterator[Event]:
    """
    Map the consensus landscape for a topic using the knowledge graph
    and wiki paper summaries. Returns structured agreement/disagreement data.
    """
    model = _wiki_model(model)
    from sciknow.rag import wiki_prompts
    from sciknow.rag.llm import complete as llm_complete
    from sciknow.storage.db import get_session
    from sqlalchemy import text

    yield {"type": "progress", "stage": "gathering",
           "detail": f"Gathering evidence for: {topic}..."}

    pattern = f"%{topic.lower()}%"

    # Get KG triples
    triples_text = ""
    with get_session() as session:
        try:
            kg_rows = session.execute(text("""
                SELECT subject, predicate, object
                FROM knowledge_graph
                WHERE LOWER(subject) LIKE :pat OR LOWER(object) LIKE :pat
                ORDER BY predicate
                LIMIT 50
            """), {"pat": pattern}).fetchall()
            triples_text = "\n".join(
                f"({r[0]}) --{r[1]}--> ({r[2]})" for r in kg_rows
            )
        except Exception:
            pass  # KG table might not exist yet

    # Get paper summaries mentioning the topic
    summaries_text = ""
    papers_dir = settings.wiki_dir / "papers"
    if papers_dir.exists():
        for f in sorted(papers_dir.glob("*.md"))[:50]:
            content = f.read_text(encoding="utf-8")
            if topic.lower() in content.lower():
                # Take the first 500 chars
                summaries_text += f"\n---\n{content[:500]}"
                if len(summaries_text) > 12000:
                    break

    if not triples_text and not summaries_text:
        yield {"type": "error",
               "message": f"No data found for '{topic}'. Run wiki compile first."}
        return

    yield {"type": "progress", "stage": "analyzing",
           "detail": f"Analyzing consensus ({len(kg_rows) if triples_text else 0} triples, "
                     f"{summaries_text.count('---')} papers)..."}

    sys_c, usr_c = wiki_prompts.wiki_consensus(topic, triples_text, summaries_text)

    try:
        raw = _strip_thinking(llm_complete(sys_c, usr_c, model=model,
                                            temperature=0.0, num_ctx=16384))
        data = json.loads(_clean_json(raw), strict=False)
        yield {"type": "consensus", "data": data}
    except Exception as exc:
        yield {"type": "error", "message": f"Consensus analysis failed: {exc}"}
        return

    # Save as a wiki synthesis page
    slug = _slugify(f"consensus-{topic}")
    summary = data.get("summary", "")
    claims = data.get("claims", [])
    content = f"# Consensus Map: {topic}\n\n{summary}\n\n"
    for c in claims:
        level = c.get("consensus_level", "unknown")
        content += f"## {c.get('claim', '')}\n"
        content += f"**Consensus:** {level} | **Trend:** {c.get('trend', 'unknown')}\n\n"
        if c.get("supporting_papers"):
            content += f"**Supporting:** {', '.join(c['supporting_papers'])}\n\n"
        if c.get("contradicting_papers"):
            content += f"**Contradicting:** {', '.join(c['contradicting_papers'])}\n\n"
    content += f"\n## Most Debated\n" + "\n".join(
        f"- {t}" for t in data.get("most_debated", [])
    )

    _ensure_wiki_dirs()
    with get_session() as session:
        _save_page(
            session, slug=slug, title=f"Consensus: {topic}",
            page_type="synthesis", content=content,
            source_doc_ids=[], subdir="synthesis",
        )

    _append_log(f"consensus | {topic} → [[{slug}]]")

    yield {"type": "completed", "slug": slug, "claims": len(claims),
           "most_debated": data.get("most_debated", [])}


# ── lint_wiki ────────────────────────────────────────────────────────────────

def lint_wiki(
    *,
    deep: bool = False,
    model: str | None = None,
) -> Iterator[Event]:
    """Run wiki health checks."""
    model = _wiki_model(model)
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    yield {"type": "progress", "stage": "lint", "detail": "Running structural checks..."}

    issues: list[dict] = []

    with get_session() as session:
        # 1. Missing summaries: papers with no wiki page
        missing = session.execute(text("""
            SELECT d.id::text, pm.title
            FROM documents d
            JOIN paper_metadata pm ON pm.document_id = d.id
            LEFT JOIN wiki_pages wp ON d.id::text = ANY(
                SELECT unnest(source_doc_ids)::text FROM wiki_pages WHERE page_type = 'paper_summary'
            )
            WHERE d.ingestion_status = 'complete' AND pm.title IS NOT NULL
                  AND wp.id IS NULL
            LIMIT 50
        """)).fetchall()

        for doc_id, title in missing:
            issues.append({"type": "missing_summary", "severity": "medium",
                           "detail": f"No wiki page for: {title or doc_id[:8]}"})

        # 2. Stale pages
        stale = session.execute(text("""
            SELECT wp.slug, wp.title
            FROM wiki_pages wp WHERE wp.needs_rewrite = 'true'
        """)).fetchall()

        for slug, title in stale:
            issues.append({"type": "stale", "severity": "low",
                           "detail": f"Needs rewrite: [[{slug}]] ({title})"})

    # 3. Broken wiki links
    if settings.wiki_dir.exists():
        all_slugs = set()
        all_links = []
        for subdir in ["papers", "concepts", "synthesis"]:
            d = settings.wiki_dir / subdir
            if d.exists():
                for f in d.glob("*.md"):
                    all_slugs.add(f.stem)
                    content = f.read_text(encoding="utf-8")
                    links = re.findall(r'\[\[([^\]]+)\]\]', content)
                    for link in links:
                        all_links.append((f.stem, link))

        for source, target in all_links:
            if target not in all_slugs:
                issues.append({"type": "broken_link", "severity": "low",
                               "detail": f"[[{source}]] links to [[{target}]] (not found)"})

        # 4. Orphaned concept pages (no inbound links)
        linked_targets = {target for _, target in all_links}
        concept_dir = settings.wiki_dir / "concepts"
        if concept_dir.exists():
            for f in concept_dir.glob("*.md"):
                if f.stem not in linked_targets:
                    issues.append({"type": "orphaned", "severity": "low",
                                   "detail": f"Orphaned concept: [[{f.stem}]]"})

    # Yield issues
    for issue in issues:
        yield {"type": "lint_issue", **issue}

    yield {"type": "progress", "stage": "lint",
           "detail": f"Found {len(issues)} issues"}

    # Deep lint: contradiction detection (LLM-based)
    if deep and settings.wiki_dir.exists():
        yield {"type": "progress", "stage": "deep_lint",
               "detail": "Running contradiction detection..."}

        from sciknow.rag import wiki_prompts
        from sciknow.rag.llm import complete as llm_complete

        concept_dir = settings.wiki_dir / "concepts"
        if concept_dir.exists():
            for f in sorted(concept_dir.glob("*.md"))[:20]:
                content = f.read_text(encoding="utf-8")
                if len(content.split()) < 50:
                    continue

                yield {"type": "progress", "stage": "deep_lint",
                       "detail": f"Checking: {f.stem}..."}

                sys_l, usr_l = wiki_prompts.wiki_lint_contradictions(
                    concept_name=f.stem.replace("-", " "),
                    claims=content[:8000],
                )
                try:
                    raw = _strip_thinking(llm_complete(sys_l, usr_l, model=model,
                                       temperature=0.0, num_ctx=8192))
                    data = json.loads(_clean_json(raw), strict=False)
                    for c in data.get("contradictions", []):
                        issues.append({
                            "type": "contradiction", "severity": c.get("severity", "medium"),
                            "detail": f"[[{f.stem}]]: {c.get('explanation', '')}",
                        })
                        yield {"type": "lint_issue", "type_": "contradiction",
                               "severity": c.get("severity", "medium"),
                               "detail": f"[[{f.stem}]]: {c.get('explanation', '')}"}
                except Exception:
                    pass

    _append_log(f"lint | {len(issues)} issues found" + (" (deep)" if deep else ""))

    yield {"type": "completed", "issues_count": len(issues), "issues": issues}


# ── list_pages / show_page ───────────────────────────────────────────────────

def list_pages(*, page_type: str | None = None) -> list[dict]:
    """Return wiki page metadata from DB."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    where = "WHERE page_type = :ptype" if page_type else ""
    params = {"ptype": page_type} if page_type else {}

    with get_session() as session:
        rows = session.execute(text(f"""
            SELECT slug, title, page_type, word_count,
                   array_length(source_doc_ids, 1) as n_sources,
                   created_at, updated_at
            FROM wiki_pages {where}
            ORDER BY page_type, title
        """), params).fetchall()

    return [
        {"slug": r[0], "title": r[1], "page_type": r[2], "word_count": r[3] or 0,
         "n_sources": r[4] or 0, "created_at": str(r[5]), "updated_at": str(r[6])}
        for r in rows
    ]


def show_page(slug: str) -> dict | None:
    """Read and return a wiki page's content."""
    for subdir in ["papers", "concepts", "synthesis"]:
        path = settings.wiki_dir / subdir / f"{slug}.md"
        if path.exists():
            return {
                "slug": slug,
                "path": str(path),
                "content": path.read_text(encoding="utf-8"),
            }
    return None
