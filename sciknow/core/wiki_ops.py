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


def _entity_name(item) -> str:
    """Phase 54.6.40 — normalize an entity entry to a plain string.

    qwen2.5:32b-instruct frequently returns concepts/methods/datasets
    as ``[{"name": "...", "description": "..."}]`` objects even when
    the prompt asks for a flat list of slug strings. Accept both
    shapes so _slugify doesn't crash with 'dict object has no
    attribute lower'. Also defensively normalizes triple
    subject/predicate/object values in case a future model decides
    to nest those too.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("name", "slug", "id", "value", "term"):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                return v
        return ""
    return str(item or "")


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


def _wiki_num_ctx(model: str) -> int:
    """Phase 54.6.28 — single shared num_ctx for every LLM call in
    compile_paper_summary so Ollama doesn't reload the model between
    perspectives / summary / polish / entity-extraction calls.

    Ollama treats each unique (model, num_ctx) pair as a separate
    model instance. Changing num_ctx between calls evicts and
    reloads — wiping the benefit of keep_alive=-1. A fixed ctx for
    the whole compile pass means ONE model load per worker.

    Thinking models (qwen3:30b-a3b, deepseek-r1, o1-like) emit long
    chain-of-thought inside the response budget, so entity extraction
    with schema-constrained JSON needs headroom: 24576 is the safe
    floor established in Phase 54.6.8.

    Dense non-thinking models (mistral, qwen2, qwen3.5, llama3)
    don't blow up their own output; 8192 is plenty and loads ~2-3×
    faster than 24576.
    """
    m = (model or "").lower()
    thinking_markers = ("a3b", "r1", "o1", ":think", "thinking", "qwq")
    if any(marker in m for marker in thinking_markers):
        return 24576
    return 8192


def compile_paper_summary(
    doc_id: str,
    *,
    model: str | None = None,
    force: bool = False,
    with_entities: bool = False,
    _existing_slugs_cache: list[str] | None = None,
) -> Iterator[Event]:
    """Generate a wiki summary page for one paper.

    Phase 54.6.35 — entity + KG extraction is now OPT-IN via
    ``with_entities=True`` (was default-on pre-54.6.35). Rationale:
    the `format=json_schema` extraction path has model-dependent
    failure modes (runaway JSON generation on dense content, thinking-
    token loops, etc.) that routinely dominate compile wall time and
    produce (0 triples, 0 entities) on the majority of papers. The
    cleaner architecture is:

      * ``wiki compile``      — build paper summaries only (one job)
      * ``wiki extract-kg``   — backfill KG triples separately
                                (retry-safe, runs on demand, can
                                pick its own model + strategy)

    This matches the separation already embodied by ``wiki extract-kg``:
    that command exists precisely to backfill entity extraction for
    compiled papers. Making it the canonical path (rather than also
    running inline during compile) eliminates an entire class of
    reliability issues from the compile critical path.

    ``_existing_slugs_cache`` (Phase 54.6.25): when compiling in bulk
    via ``compile_all()``, the caller loads the slug list ONCE and
    passes it through. Pre-fix, every paper triggered a fresh
    ``SELECT slug FROM wiki_pages`` — 600 identical queries on a
    600-paper corpus.  Callers that compile a single paper (CLI
    ``--doc-id``, web API) pass ``None`` and the per-paper lookup
    runs as before.
    """
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

        # Check if already compiled (skip unless force). Phase 54.6.23 —
        # even when the summary exists, we still check whether entities
        # + KG triples have been extracted for this doc. Pre-fix, a
        # paper whose summary landed but whose entity pass never ran
        # (or was rolled back by a DB wipe after the fact) would be
        # permanently stuck — the skip path returned before
        # _extract_entities_and_kg could backfill concept pages.
        # Now: skip iff summary exists AND KG triples exist. Otherwise
        # fall through to entity extraction (without re-doing the
        # expensive summary LLM call).
        skip_summary = False
        need_entities = True
        # Phase 54.6.34 — --skip-entities short-circuits the expensive
        # structured-output call. When True, we still check if the
        # summary exists (for skip detection) but never run entity
        # extraction, so a skipped paper with summary already in place
        # is a total no-op.
        if not with_entities:
            need_entities = False
        if not force:
            existing = session.execute(text(
                "SELECT id FROM wiki_pages WHERE slug = :slug"
            ), {"slug": slug}).fetchone()
            if existing:
                skip_summary = True
                if with_entities:
                    kg_count = session.execute(text(
                        "SELECT COUNT(*) FROM knowledge_graph "
                        "WHERE source_doc_id::text = :did"
                    ), {"did": doc_id}).scalar() or 0
                    need_entities = (kg_count == 0)
                if not need_entities:
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

        existing_slugs = (
            _existing_slugs_cache
            if _existing_slugs_cache is not None
            else _load_existing_slugs(session)
        )

    # Format metadata
    author_str = ", ".join(
        a.get("name", "") if isinstance(a, dict) else str(a)
        for a in (authors or [])[:5]
    )
    kw_str = ", ".join(keywords or [])
    dom_str = ", ".join(domains or [])

    # Phase 54.6.23 — only call the expensive summary LLM when the
    # summary is actually missing. If we're here because entities are
    # missing but the summary is present, skip straight to entity
    # extraction with the existing on-disk content.
    if skip_summary:
        yield {"type": "progress", "stage": "entity_only",
               "detail": f"Summary exists, backfilling entities: {slug}"}
        # Read the already-saved content from disk (or fall back to
        # section_text if the file is missing — rare, but don't crash).
        summary_path = settings.wiki_dir / "papers" / f"{slug}.md"
        content = summary_path.read_text(encoding="utf-8") if summary_path.exists() else section_text
    else:
        # Phase 54.6.26 — multi-perspective pre-research (from STORM).
        # Phase 54.6.28 — use ONE num_ctx across all 4 LLM calls so
        # Ollama keeps the model resident instead of reloading between
        # perspectives / summary / polish / entity-extraction (each
        # unique num_ctx is a separate Ollama model instance).
        # Phase 54.6.34b — with --no-entities, we NEVER make the
        # structured-output call that needed 24576 ctx for thinking-
        # model headroom. The remaining calls (perspectives + summary +
        # polish) fit comfortably in 8192. Using a smaller ctx here
        # shrinks KV cache ~3× which lets us run more parallel workers
        # AND reduces per-token latency.
        shared_ctx = _wiki_num_ctx(model) if with_entities else 8192

        perspectives_text = ""
        try:
            from sciknow.rag.llm import complete as llm_complete
            p_sys, p_usr = wiki_prompts.wiki_perspectives(
                title=title, authors=author_str,
                year=str(year or "n.d."),
                abstract=abstract or "",
                sections=section_text[:3000],
            )
            raw_perspectives = _strip_thinking(
                llm_complete(
                    p_sys, p_usr, model=model,
                    temperature=0.3, num_ctx=shared_ctx,
                    # Phase 54.6.42 — 400-tok cap retained as hygiene.
                    # `/no_think` removed: the new LLM_FAST_MODEL
                    # (qwen3:30b-a3b-instruct-2507) is non-thinking so
                    # the tag is a no-op. _strip_thinking() also kept
                    # as belt-and-braces in case the model is swapped
                    # back to a thinking variant by .env override.
                    num_predict=400,
                    keep_alive=-1) or ""
            ).strip()
            if raw_perspectives:
                perspectives_text = (
                    "\n\nExpert perspectives to address in your summary:\n"
                    + raw_perspectives
                )
                yield {"type": "progress", "stage": "perspectives",
                       "detail": f"Generated expert perspectives"}
        except Exception:
            pass  # best-effort; if it fails, write without perspectives

        yield {"type": "progress", "stage": "writing",
               "detail": f"Compiling summary: {title or doc_id[:8]}..."}

        system, user = wiki_prompts.wiki_paper_summary(
            title=title, authors=author_str, year=str(year or "n.d."),
            journal=journal or "", doi=doi or "",
            keywords=kw_str, domains=dom_str,
            abstract=abstract or "",
            sections=section_text + perspectives_text,
            existing_slugs=existing_slugs,
        )

        tokens: list[str] = []
        # Phase 54.6.42 — 1500-tok cap retained as hygiene (typical wiki
        # summary is 300-600 words ≈ 400-800 tokens; 1500 is generous).
        # `/no_think` removed: new LLM_FAST_MODEL is non-thinking.
        for tok in llm_stream(system, user, model=model,
                              num_ctx=shared_ctx, num_predict=1500,
                              keep_alive=-1):
            tokens.append(tok)
            yield {"type": "token", "text": tok}

        content = _strip_thinking("".join(tokens))

        # Phase 54.6.26 — polishing pass (from STORM Article Polishing).
        # Quick cleanup: remove duplication, smooth transitions, fix
        # formatting. Uses the same model with keep_alive=-1 so no
        # model-swap overhead. Best-effort — if it fails, save the
        # unpolished version (still perfectly usable).
        try:
            pol_sys, pol_usr = wiki_prompts.wiki_polish(content)
            polished = _strip_thinking(
                llm_complete(
                    pol_sys, pol_usr, model=model,
                    temperature=0.1, num_ctx=shared_ctx,
                    # Phase 54.6.42 — /no_think removed (new fast model
                    # is non-thinking). 1500-tok cap retained as hygiene;
                    # polished output shouldn't exceed the original
                    # summary length.
                    num_predict=1500,
                    keep_alive=-1) or ""
            ).strip()
            if polished and len(polished) > 100:
                content = polished
                yield {"type": "progress", "stage": "polished",
                       "detail": "Wiki page polished"}
        except Exception:
            pass  # save unpolished — still good

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
    # Phase 54.6.28 — pass shared_ctx so entity extraction doesn't
    # reload the model with a different num_ctx.
    # Phase 54.6.35 — entity extraction is opt-in. By default we don't
    # run it here; users run `sciknow wiki extract-kg` separately when
    # they want the KG populated.
    if not with_entities:
        _append_log(
            f"ingest | {title} → [[{slug}]] "
            f"(summary only — run `wiki extract-kg` for entities)"
        )
        yield {"type": "completed", "slug": slug,
               "word_count": len(content.split()),
               "kg_triples": 0, "entities": []}
        return

    yield {"type": "progress", "stage": "extracting",
           "detail": "Extracting entities + knowledge graph..."}
    # shared_ctx is defined in the not-skipped branch above; in the
    # skip_summary branch we still need it for entity-only backfill.
    _ctx_for_entities = locals().get("shared_ctx") or _wiki_num_ctx(model)
    entities, kg_count = _extract_entities_and_kg(
        doc_id, slug, title, author_str, str(year or "n.d."),
        kw_str, dom_str, abstract or "", section_text,
        existing_slugs, model=model, num_ctx=_ctx_for_entities,
    )

    _append_log(f"ingest | {title} → [[{slug}]] ({kg_count} triples, {len(entities)} entities)")

    yield {"type": "completed", "slug": slug, "word_count": len(content.split()),
           "kg_triples": kg_count, "entities": entities}


def _find_json_block(raw: str) -> str | None:
    """Phase 54.6.36 — robust JSON extraction from free-form LLM output.

    Tries these strategies in order:
      1. Markdown code fence: ```json ... ``` or ``` ... ```
      2. Outermost balanced braces: from first { to its matching }
      3. Return None if neither yields parseable JSON

    Handles common LLM noise: preamble text, trailing commentary,
    <think>...</think> blocks (stripped by caller), nested objects.
    """
    import re as _re
    if not raw:
        return None

    # Try markdown code fences (greedy, prefer ```json over bare ```)
    fence_match = _re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, _re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate, strict=False)
            return candidate
        except Exception:
            pass  # fall through to brace matching

    # Fall back: find first { and its matching closing }
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        c = raw[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : i + 1]
                try:
                    json.loads(candidate, strict=False)
                    return candidate
                except Exception:
                    return None

    # Phase 54.6.42 — salvage for truncated JSON (num_predict cutoff).
    # Scenario: model emits valid JSON but was cut mid-array before
    # the closing brace. Walk back from the end of the raw string,
    # drop any partial trailing element, close open arrays and the
    # outer object. Returns something `json.loads` can parse so the
    # caller keeps 80% of the payload instead of 0%.
    tail = raw[start:]
    # Trim trailing incomplete string literal, if any: find the last
    # unescaped unmatched quote and cut everything after it.
    if in_string:
        # Inside an unterminated string — cut back to last newline or
        # last comma outside the string context we can safely close.
        last_safe = max(tail.rfind(',\n'), tail.rfind('\n'))
        if last_safe > 0:
            tail = tail[:last_safe]
    # Trim trailing comma if present
    tail = tail.rstrip().rstrip(",")
    # Close any open brackets (array depth + object depth).
    # Count in a fresh pass: we want a minimal closing sequence.
    open_sq = 0
    open_cu = 0
    in_s = False
    esc = False
    for c in tail:
        if esc:
            esc = False; continue
        if c == "\\":
            esc = True; continue
        if c == '"':
            in_s = not in_s; continue
        if in_s:
            continue
        if c == "[":
            open_sq += 1
        elif c == "]":
            open_sq -= 1
        elif c == "{":
            open_cu += 1
        elif c == "}":
            open_cu -= 1
    repaired = tail + ("]" * max(open_sq, 0)) + ("}" * max(open_cu, 0))
    try:
        json.loads(repaired, strict=False)
        return repaired
    except Exception:
        return None


def _extract_entities_and_kg(
    doc_id, slug, title, authors, year, keywords, domains,
    abstract, sections, existing_slugs, model=None,
    num_ctx: int | None = None,
) -> tuple[list[str], int]:
    """Combined entity + KG triple extraction.

    Phase 54.6.36 — reworked to NOT use ``format=json_schema`` after
    observing 100% runaway rate across mistral:7b, qwen3.5:27b, and
    qwen3:30b-a3b on dense scientific content. The schema constraint
    overrides the model's natural stop-token behavior, causing the
    model to generate thinking tokens or garbage JSON indefinitely
    until ``num_predict`` cuts it off. By the time we cap at 2048
    tokens, the output is usually all ``<think>`` with no JSON.

    New approach:
      * No ``format=`` parameter — model uses free-form output
      * Prompt asks for JSON in a ```json code fence``` for parsing
      * ``_find_json_block()`` extracts the JSON robustly, tolerating
        preamble/trailing text and nested braces in string values
      * Much smaller ctx (8192) and num_predict (1500) since the model
        doesn't need headroom for schema-induced thinking loops

    ``num_ctx`` (Phase 54.6.28) still accepted for backward compat,
    but defaults to 8192 now (was 24576).
    """
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

    try:
        # Phase 54.6.42 — default upgraded to qwen3:30b-a3b-instruct-2507
        # after the 2026-04-17 sweep. This is the official non-thinking
        # variant of qwen3:30b-a3b — same MoE architecture, same VRAM,
        # but no thinking mode.
        #
        # Empirical results on paper 4092d6ad (math-heavy Nature-CO2):
        #   qwen3:30b-a3b-instruct-2507  →  13s, 12 triples, 50% verbatim  ✓
        #   qwen2.5:32b-instruct         →  108s, 10 triples, 40% verbatim (old default)
        #   gemma3:27b-it-qat            →   38s, 12 triples, 33% verbatim  (#2)
        #   qwen3:30b-a3b (thinking)     →   runaway: 0 content, 2048 tok spent thinking
        #   qwen3.5:27b  (thinking)      →   runaway (same failure mode)
        #   command-r:35b                →   356s, ran past 2048 num_predict mid-JSON
        # Callers can still override via the ``model`` parameter.
        extract_model = model or "qwen3:30b-a3b-instruct-2507-q4_K_M"
        effective_ctx = num_ctx if num_ctx is not None else 8192
        raw = llm_complete(
            sys_e, usr_e, model=extract_model,
            temperature=0.0, num_ctx=effective_ctx,
            # Phase 54.6.42 — raised 1500 → 3000 after observing
            # qwen3:30b-a3b-instruct-2507 overshoot 2048 (tokens=2049,
            # done_reason=length) on the 4092d6ad retry: the model
            # emitted 12+ concepts and was still going when the cap
            # bit. Content quality is excellent (real CO2/solar
            # concepts, no "electron beam" hallucinations like qwen2.5
            # produced) so the right move is to give it headroom, not
            # fight the verbosity. 3000 tokens ≈ 11 KB of JSON — still
            # cheap at 14s elapsed.
            num_predict=3000,
            keep_alive=-1,
            # Note: NO format= argument. Schema-constrained JSON
            # mode triggered runaways on this corpus.
        )
        # Strip thinking, then find the JSON block.
        raw_cleaned = _strip_thinking(raw or "").strip()
        if not raw_cleaned:
            logger.warning(
                "Entity+KG extraction returned empty output for %s "
                "(model generated only <think> tokens; try --model "
                "<non-thinking>)", slug,
            )
            return [], 0
        json_text = _find_json_block(raw_cleaned)
        if not json_text:
            logger.warning(
                "Entity+KG extraction for %s: no JSON block found in "
                "model output (first 300 chars: %r)",
                slug, raw_cleaned[:300],
            )
            return [], 0
        data = json.loads(json_text, strict=False)
    except Exception as exc:
        logger.warning("Entity+KG extraction failed for %s: %s", slug, exc)
        return [], 0

    # Save entities (concepts list). Phase 54.6.40 — normalize entries
    # through _entity_name: qwen2.5:32b often returns {"name": ...,
    # "description": ...} objects despite the prompt asking for flat
    # slug strings.
    all_entities = [
        name for name in (
            _entity_name(x) for x in (
                data.get("concepts", []) +
                data.get("methods", []) +
                data.get("datasets", [])
            )
        ) if name
    ]

    # Save KG triples
    triples = data.get("triples", [])
    kg_count = 0
    if triples:
        # Phase 54.6.209 — roadmap 3.7.1: collapse surface-variant
        # entity names onto a canonical form before insert so
        # "CO2" / "CO₂" / "carbon dioxide" / "Carbon Dioxide" all
        # become one KG node. Predicate left alone (different
        # vocabulary-stabilisation concern — see roadmap 3.7.2).
        from sciknow.core.kg_canonicalize import canonicalize as _canon_entity

        with get_session() as session:
            session.execute(text(
                "DELETE FROM knowledge_graph WHERE source_doc_id::text = :did"
            ), {"did": doc_id})
            for t in triples:
                if not isinstance(t, dict):
                    continue
                subj = _canon_entity(_entity_name(t.get("subject")))[:200]
                pred = _entity_name(t.get("predicate")).strip().lower()[:100]
                obj = _canon_entity(_entity_name(t.get("object")))[:200]
                # Phase 48d — empty / whitespace-only source sentence
                # flows through as NULL so the UI can render "(no
                # source sentence)" consistently instead of "".
                sent_val = t.get("source_sentence")
                sent_raw = (sent_val if isinstance(sent_val, str) else "").strip()[:500]
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

                # Update DB. Phase 54.6.23 — only append the doc_id to
                # source_doc_ids if it isn't already there. Pre-fix, a
                # paper that updated the same concept twice (from two
                # runs, or after a wiki_pages re-insert) would
                # accumulate duplicate UUIDs in the array, which in
                # turn inflated the "sources: N papers" count and
                # broke the single-source branch of list_pages.
                with get_session() as session:
                    session.execute(text("""
                        UPDATE wiki_pages SET
                            word_count = :wc,
                            source_doc_ids = CASE
                                WHEN CAST(:did AS uuid) = ANY(source_doc_ids)
                                THEN source_doc_ids
                                ELSE array_append(source_doc_ids, CAST(:did AS uuid))
                            END,
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
    with_entities: bool = False,
) -> Iterator[Event]:
    """Build the full wiki from all ingested papers.

    ``with_entities`` (Phase 54.6.35) routes through to each
    ``compile_paper_summary`` call. Off by default — entity
    extraction is its own step, run via ``sciknow wiki extract-kg``.
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    # Phase 54.6.31 — warm up the LLM before entering the hot loop so the
    # first paper doesn't pay cold-start latency (~3-10s on a 7B model).
    # Uses the same ctx + batch settings the workload will use so
    # Ollama doesn't create a second instance on the first real call.
    # Phase 54.6.34b — when entity extraction is off, the workload
    # uses ctx=8192; warm up at 8192 to match, otherwise warmup
    # wastes a load on a 24576 instance that immediately gets
    # evicted for the 8192 instance.
    from sciknow.rag.llm import warm_up as _llm_warm_up
    _resolved_model = _wiki_model(model)
    _warmup_ctx = _wiki_num_ctx(_resolved_model) if with_entities else 8192
    _llm_warm_up(
        model=_resolved_model,
        num_ctx=_warmup_ctx,
        num_batch=1024,
    )

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
    # Phase 55.1.1 — opt-in parallelism. Defaults to 1 because the
    # speedup is hardware-dependent on MoE models and single-GPU
    # setups; raise `WIKI_COMPILE_WORKERS` and set a matching
    # `OLLAMA_NUM_PARALLEL` only after benchmarking on your box.
    workers = max(1, int(settings.wiki_compile_workers or 1))
    effective_workers = min(workers, max(1, total))
    compile_wall_t0 = time.monotonic()

    # Phase 54.6.25 — load the slug list ONCE and pass through to
    # every paper. Pre-fix, each of the 600 papers fired an
    # identical `SELECT slug FROM wiki_pages ORDER BY slug` —
    # ~600 wasted queries. The list is read-only during compile
    # (new slugs added by _save_page won't be visible to the LLM
    # prompt until the next compile run anyway; pre-fix they weren't
    # either, since the per-paper load ran BEFORE the new page was
    # saved). Thread-safe: only passed by reference, never mutated.
    with get_session() as session:
        existing_slugs_cache = _load_existing_slugs(session)

    yield {"type": "compile_start", "total": total,
           "workers": effective_workers}

    compiled = 0
    skipped = 0
    failed = 0
    total_tokens = 0

    def _run_paper(index_title_doc: tuple[int, str, str]) -> dict:
        """Worker: compile one paper, collect events into a list,
        return a summary dict. Runs inside a thread so Ollama can
        serve multiple wiki-compile calls concurrently when
        OLLAMA_NUM_PARALLEL ≥ workers."""
        i, doc_id, title = index_title_doc
        short_title = (title or doc_id[:8])[:60]
        paper_t0 = time.monotonic()
        collected: list[dict] = []
        paper_tokens = 0
        status = "compiled"
        err: str | None = None
        try:
            for event in compile_paper_summary(
                doc_id, model=model, force=force,
                with_entities=with_entities,
                _existing_slugs_cache=existing_slugs_cache,
            ):
                collected.append(event)
                if event.get("type") == "token":
                    paper_tokens += 1
                elif event.get("type") == "error":
                    status = "error"
                    err = event.get("message", "")
                    break
                elif event.get("type") == "completed":
                    if event.get("skipped"):
                        status = "skipped"
        except Exception as exc:
            status = "error"
            err = f"{type(exc).__name__}: {exc}"
        return {
            "index": i, "doc_id": doc_id, "title": short_title,
            "events": collected, "paper_tokens": paper_tokens,
            "status": status, "err": err,
            "elapsed": time.monotonic() - paper_t0,
        }

    work = list(enumerate(rows, 1))  # [(i, (doc_id, title)), ...]
    flat = [(i, d, t) for i, (d, t) in work]

    if effective_workers == 1:
        # Sequential path — same per-token streaming as before.
        for i, doc_id, title in flat:
            short_title = (title or doc_id[:8])[:60]
            paper_t0 = time.monotonic()
            paper_tokens = 0
            yield {"type": "paper_start", "index": i, "total": total,
                   "title": short_title, "doc_id": doc_id}
            paper_ok = False
            paper_skipped = False
            for event in compile_paper_summary(
                doc_id, model=model, force=force,
                with_entities=with_entities,
                _existing_slugs_cache=existing_slugs_cache,
            ):
                if event.get("type") == "token":
                    paper_tokens += 1
                    total_tokens += 1
                    yield {"type": "token", "text": event["text"],
                           "paper_tokens": paper_tokens,
                           "total_tokens": total_tokens}
                elif event.get("type") == "error":
                    failed += 1
                    yield {"type": "paper_done", "index": i, "total": total,
                           "title": short_title, "status": "error",
                           "detail": event.get("message", ""),
                           "tokens": paper_tokens,
                           "elapsed": time.monotonic() - paper_t0,
                           "compiled": compiled, "skipped": skipped,
                           "failed": failed, "total_tokens": total_tokens}
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
                       "compiled": compiled, "skipped": skipped,
                       "failed": failed, "tokens": 0, "elapsed": 0,
                       "total_tokens": total_tokens}
                continue
            yield {"type": "paper_done", "index": i, "total": total,
                   "title": short_title,
                   "status": "compiled" if paper_ok else "error",
                   "compiled": compiled, "skipped": skipped, "failed": failed,
                   "tokens": paper_tokens,
                   "elapsed": time.monotonic() - paper_t0,
                   "total_tokens": total_tokens}
    else:
        # Parallel path — N papers in flight at once. Ollama must be
        # started with OLLAMA_NUM_PARALLEL >= workers or the requests
        # will head-of-line-block and lose the throughput win.
        # Per-paper events are buffered and replayed atomically on
        # paper completion (we lose per-token streaming, but that's
        # a fair trade for the 1.5-2.5× overall speedup).
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_run_paper, t): t for t in flat}
            for fut in as_completed(futures):
                res = fut.result()
                i = res["index"]
                short_title = res["title"]
                yield {"type": "paper_start", "index": i, "total": total,
                       "title": short_title, "doc_id": res["doc_id"]}
                # Replay tokens and other events in the order the
                # worker saw them, so the CLI renderer still sees a
                # coherent per-paper story (even if two papers
                # interleave at the paper boundary).
                for event in res["events"]:
                    if event.get("type") == "token":
                        total_tokens += 1
                        yield {"type": "token", "text": event["text"],
                               "paper_tokens": 1,  # running counter unknown here
                               "total_tokens": total_tokens}
                if res["status"] == "error":
                    failed += 1
                    yield {"type": "paper_done", "index": i, "total": total,
                           "title": short_title, "status": "error",
                           "detail": res.get("err") or "",
                           "tokens": res["paper_tokens"],
                           "elapsed": res["elapsed"],
                           "compiled": compiled, "skipped": skipped,
                           "failed": failed, "total_tokens": total_tokens}
                elif res["status"] == "skipped":
                    skipped += 1
                    yield {"type": "paper_done", "index": i, "total": total,
                           "title": short_title, "status": "skipped",
                           "compiled": compiled, "skipped": skipped,
                           "failed": failed, "tokens": 0, "elapsed": 0,
                           "total_tokens": total_tokens}
                else:
                    compiled += 1
                    yield {"type": "paper_done", "index": i, "total": total,
                           "title": short_title, "status": "compiled",
                           "compiled": compiled, "skipped": skipped,
                           "failed": failed, "tokens": res["paper_tokens"],
                           "elapsed": res["elapsed"],
                           "total_tokens": total_tokens}

    _update_index()
    # Phase 54.2 — drop the backlinks cache so the next reader request
    # rebuilds against the fresh set of pages on disk.
    invalidate_backlinks()
    # Phase 55.1.1 — wall-clock timing so the user can compare
    # WIKI_COMPILE_WORKERS settings empirically. The per-paper avg
    # is computed over `compiled + failed` (skipped papers didn't do
    # real work and would bias the average downward).
    compile_wall_elapsed = time.monotonic() - compile_wall_t0
    did_real_work = max(1, compiled + failed)
    avg_sec_per_paper = compile_wall_elapsed / did_real_work
    _append_log(
        f"compile-all | {compiled} new, {skipped} skipped, {failed} failed "
        f"from {total} papers in {compile_wall_elapsed:.1f}s "
        f"(workers={effective_workers}, {avg_sec_per_paper:.1f}s/paper)"
    )

    yield {"type": "completed", "compiled": compiled, "skipped": skipped,
           "failed": failed, "total": total,
           "elapsed_seconds": round(compile_wall_elapsed, 1),
           "workers": effective_workers,
           "avg_sec_per_paper": round(avg_sec_per_paper, 1)}


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
    kg_rows: list = []
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
        except Exception as exc:
            # KG table might not exist yet (older install) — but a real
            # failure (permissions, malformed query, dead connection)
            # used to silently degrade to "no triples found" with no
            # log breadcrumb. Phase 54.6.21: log and continue.
            logger.warning("consensus_map: KG query failed (%s)", exc)

    # Get paper summaries mentioning the topic
    summaries_text = ""
    papers_dir = settings.wiki_dir / "papers"
    if papers_dir.exists():
        for f in sorted(papers_dir.glob("*.md"))[:50]:
            content = f.read_text(encoding="utf-8")
            if topic.lower() in content.lower():
                # Take the first 500 chars. Phase 54.6.21 — check
                # length BEFORE appending so we cap at the budget
                # instead of overshooting by 500 chars per iter.
                snippet = f"\n---\n{content[:500]}"
                if len(summaries_text) + len(snippet) > 12000:
                    break
                summaries_text += snippet

    if not triples_text and not summaries_text:
        yield {"type": "error",
               "message": f"No data found for '{topic}'. Run wiki compile first."}
        return

    yield {"type": "progress", "stage": "analyzing",
           "detail": f"Analyzing consensus ({len(kg_rows) if triples_text else 0} triples, "
                     f"{summaries_text.count('---')} papers)..."}

    sys_c, usr_c = wiki_prompts.wiki_consensus(topic, triples_text, summaries_text)

    try:
        # Phase 54.6.42 — /no_think removed; the new LLM_FAST_MODEL
        # (qwen3:30b-a3b-instruct-2507) is non-thinking, and the 24576
        # ctx headroom now just serves the ~12k evidence-text volume.
        # _strip_thinking() retained in case someone overrides --model
        # back to a thinking variant.
        raw = llm_complete(sys_c, usr_c, model=model,
                           temperature=0.0, num_ctx=24576)
        cleaned = _strip_thinking(raw or "").strip()
        if not cleaned:
            yield {"type": "error",
                   "message": "LLM returned empty output (likely context "
                              "overflow even at ctx=24576). Try a smaller "
                              "topic or --model gemma3:27b-it-qat."}
            return
        data = json.loads(_clean_json(cleaned), strict=False)
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

    # Yield issues. Phase 54.6.43 — emit ``type_`` for the inner
    # issue kind so the outer event key ``type`` stays as "lint_issue".
    # Pre-fix, ``{"type": "lint_issue", **issue}`` collided with
    # ``issue["type"] = "orphaned"`` (later wins in dict spread), so
    # every yielded event looked like ``type="orphaned"`` and the
    # CLI's ``if t == "lint_issue"`` branch never fired — user saw
    # only the trailing count, never the individual issues.
    for issue in issues:
        yield {
            "type": "lint_issue",
            "type_":    issue.get("type", "unknown"),
            "severity": issue.get("severity", "low"),
            "detail":   issue.get("detail", ""),
        }

    # Summary breakdown by (severity, kind). 3666 "issues" with no
    # decomposition is useless; showing "2500 orphaned / 1000 broken_link
    # / 50 stale" lets the user triage.
    from collections import Counter
    by_kind = Counter(i.get("type", "unknown") for i in issues)
    by_sev  = Counter(i.get("severity", "low") for i in issues)
    yield {
        "type":          "lint_summary",
        "total":         len(issues),
        "by_kind":       dict(by_kind),
        "by_severity":   dict(by_sev),
    }

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
    """Return wiki page metadata from DB.

    Phase 54.6.9 — enriched with ``year`` and ``authors`` from the
    linked paper when there's exactly one source_doc_id (the common
    paper_summary case). Synthesis / concept pages can aggregate
    multiple documents so we don't try to pick a single year/authors
    string for them — those fields come back as None / [].
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    where = "WHERE wp.page_type = :ptype" if page_type else ""
    params = {"ptype": page_type} if page_type else {}

    # Phase 54.6.22 — replaced LEFT JOIN LATERAL with a single bulk
    # SELECT into a dict, then merge in Python. The LATERAL pattern
    # forced PostgreSQL to re-execute the paper_metadata subquery once
    # per wiki_pages row (~961 times on a real wiki). For single-source
    # pages we only need year + authors; the new shape does one scan of
    # paper_metadata filtered to the source_doc_ids set we actually need.
    with get_session() as session:
        rows = session.execute(text(f"""
            SELECT wp.slug, wp.title, wp.page_type, wp.word_count,
                   array_length(wp.source_doc_ids, 1) AS n_sources,
                   wp.created_at, wp.updated_at,
                   wp.source_doc_ids
            FROM wiki_pages wp
            {where}
            ORDER BY wp.page_type, wp.title
        """), params).fetchall()

        # Collect the single-source doc_ids that we actually need
        # paper_metadata for (paper_summary pages typically).
        single_source_doc_ids: set = set()
        for r in rows:
            src = r[7]
            if src and len(src) == 1:
                single_source_doc_ids.add(src[0])

        meta_by_doc: dict = {}
        if single_source_doc_ids:
            placeholders = ", ".join(
                f":d{i}" for i, _ in enumerate(single_source_doc_ids)
            )
            mparams = {
                f"d{i}": str(d) for i, d in enumerate(single_source_doc_ids)
            }
            mrows = session.execute(text(f"""
                SELECT document_id::text, year, authors
                FROM paper_metadata
                WHERE document_id::text IN ({placeholders})
            """), mparams).fetchall()
            meta_by_doc = {m[0]: (m[1], m[2]) for m in mrows}

    out: list[dict] = []
    for r in rows:
        src = r[7]
        year, raw_authors = (None, [])
        if src and len(src) == 1:
            year, raw_authors = meta_by_doc.get(str(src[0]), (None, []))
        # `authors` in paper_metadata is a JSONB array of dicts /
        # strings; normalize to a comma-separated display string for
        # the list view, keeping the raw list in `authors_raw` for
        # callers that want to format themselves.
        raw_authors = raw_authors or []
        names: list[str] = []
        for a in raw_authors[:6]:
            if isinstance(a, dict):
                nm = a.get("name") or a.get("full_name") or ""
                if nm:
                    names.append(str(nm))
            elif isinstance(a, str):
                names.append(a)
        out.append({
            "slug": r[0], "title": r[1], "page_type": r[2],
            "word_count": r[3] or 0,
            "n_sources": r[4] or 0,
            "created_at": str(r[5]), "updated_at": str(r[6]),
            "year": year,
            "authors": names,
            "authors_display": ", ".join(names)
                + (f" +{len(raw_authors) - 6}" if len(raw_authors) > 6 else ""),
        })
    return out


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


# ── Phase 54.2 — wiki backlinks + related pages ──────────────────────
# Backlinks live on disk (every page is a .md under data/wiki/).
# Scanning all files costs O(N_pages × avg_page_size) per build, so
# cache the result at module level and invalidate only when the
# caller tells us to (`wiki compile` will call invalidate_backlinks()
# after writing; the cache is also time-bounded as a safety net).

import time as _time
import threading as _threading
import re as _re

_BACKLINKS_CACHE: dict[str, list[dict]] | None = None
_BACKLINKS_CACHE_TS: float = 0.0
_BACKLINKS_CACHE_TTL = 10 * 60  # 10 minutes — upper bound even without
                                # an explicit invalidate_backlinks() call
_BACKLINKS_LOCK = _threading.Lock()


def _scan_backlinks_index(base_dir=None) -> dict[str, list[dict]]:
    """Walk every .md page under ``base_dir`` (or ``settings.wiki_dir``
    if not given) and build a reverse index:
    ``{target_slug: [{from_slug, from_title, alt}]}``. Links of both
    shapes are collected — ``[[slug]]`` (no alt text) and
    ``[[slug|alt text]]``. A self-link (`from == target`) is skipped
    silently. ``base_dir`` is overridable so L1 tests can point at a
    tmp tree without mutating Pydantic Settings (which are frozen)."""
    from sqlalchemy import text as _sql_text
    from sciknow.storage.db import get_session

    base = base_dir if base_dir is not None else settings.wiki_dir
    out: dict[str, list[dict]] = {}
    if not base.exists():
        return out

    # Pull slug → title once so we can annotate backlinks without a
    # per-row DB round trip later.
    slug_to_title: dict[str, str] = {}
    try:
        with get_session() as session:
            rows = session.execute(_sql_text(
                "SELECT slug, title FROM wiki_pages"
            )).fetchall()
        for s, t in rows:
            slug_to_title[s] = t or s
    except Exception:
        pass

    # Compile the regexes once; they run against every file.
    alt_re = _re.compile(r"\[\[([^\]\|]+)\|([^\]]+)\]\]")
    bare_re = _re.compile(r"\[\[([^\]\|]+)\]\]")

    for subdir in ("papers", "concepts", "synthesis"):
        d = base / subdir
        if not d.exists():
            continue
        for p in d.glob("*.md"):
            from_slug = p.stem
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            seen: set[tuple[str, str]] = set()
            # [[slug|alt]] first so bare-re doesn't double-count the
            # same span when the two patterns overlap.
            for m in alt_re.finditer(txt):
                target, alt = m.group(1), m.group(2)
                if target == from_slug:
                    continue
                key = (target, alt)
                if key in seen:
                    continue
                seen.add(key)
                out.setdefault(target, []).append({
                    "from_slug": from_slug,
                    "from_title": slug_to_title.get(from_slug, from_slug),
                    "alt": alt,
                })
            # Strip [[slug|alt]] before searching for bare [[slug]] so
            # we don't double-count.
            txt_no_alt = alt_re.sub("", txt)
            for m in bare_re.finditer(txt_no_alt):
                target = m.group(1)
                if target == from_slug:
                    continue
                key = (target, "")
                if key in seen:
                    continue
                seen.add(key)
                out.setdefault(target, []).append({
                    "from_slug": from_slug,
                    "from_title": slug_to_title.get(from_slug, from_slug),
                    "alt": "",
                })
    return out


def invalidate_backlinks() -> None:
    """Drop the cached backlinks index. `wiki compile` calls this
    after writing new / rewritten pages so readers pick up the fresh
    graph without a server restart."""
    global _BACKLINKS_CACHE, _BACKLINKS_CACHE_TS
    with _BACKLINKS_LOCK:
        _BACKLINKS_CACHE = None
        _BACKLINKS_CACHE_TS = 0.0


def get_backlinks_for(slug: str) -> list[dict]:
    """Return the list of pages that link to `slug`. Rebuilds the
    cache lazily if cold / expired."""
    global _BACKLINKS_CACHE, _BACKLINKS_CACHE_TS
    now = _time.monotonic()
    with _BACKLINKS_LOCK:
        stale = (
            _BACKLINKS_CACHE is None
            or (now - _BACKLINKS_CACHE_TS) > _BACKLINKS_CACHE_TTL
        )
    if stale:
        fresh = _scan_backlinks_index()
        with _BACKLINKS_LOCK:
            _BACKLINKS_CACHE = fresh
            _BACKLINKS_CACHE_TS = now
    return list((_BACKLINKS_CACHE or {}).get(slug, []))


def get_related_pages(slug: str, *, limit: int = 5) -> list[dict]:
    """Top-N pages whose WIKI_COLLECTION embedding is nearest to
    `slug`. Returns ``[{slug, title, page_type, score}]`` sorted by
    cosine descending, excluding the source page itself. Empty list
    when the source has no Qdrant embedding (e.g. stub concept page
    that hasn't been compiled yet)."""
    from sqlalchemy import text as _sql_text
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import WIKI_COLLECTION, get_client

    with get_session() as session:
        row = session.execute(_sql_text(
            "SELECT qdrant_point_id::text FROM wiki_pages WHERE slug = :s"
        ), {"s": slug}).fetchone()
    if not row or not row[0]:
        return []
    point_id = row[0]

    try:
        client = get_client()
        anchor = client.retrieve(
            collection_name=WIKI_COLLECTION,
            ids=[point_id],
            with_vectors=True,
        )
    except Exception:
        return []
    if not anchor:
        return []
    v = anchor[0].vector
    if isinstance(v, dict):
        # Named-vector schema: take the dense vector.
        v = v.get("dense") or next(iter(v.values()))

    try:
        hits = client.search(
            collection_name=WIKI_COLLECTION,
            query_vector=v,
            limit=limit + 1,
            with_payload=False,
        )
    except Exception:
        return []

    # Drop the source page itself if it showed up first (it usually
    # does since the query vector is its own embedding).
    ids = [str(h.id) for h in hits if str(h.id) != point_id][:limit]
    score_by_id = {str(h.id): h.score for h in hits}
    if not ids:
        return []
    with get_session() as session:
        rows = session.execute(_sql_text("""
            SELECT slug, title, page_type, qdrant_point_id::text
            FROM wiki_pages
            WHERE qdrant_point_id::text = ANY(:ids)
        """), {"ids": ids}).fetchall()
    lookup = {r[3]: (r[0], r[1], r[2]) for r in rows}
    out = []
    for pid in ids:
        meta = lookup.get(pid)
        if meta:
            out.append({
                "slug": meta[0], "title": meta[1] or meta[0],
                "page_type": meta[2],
                "score": float(score_by_id.get(pid, 0.0)),
            })
    return out
