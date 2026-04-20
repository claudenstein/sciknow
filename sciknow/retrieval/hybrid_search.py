"""
Hybrid search: dense (Qdrant) + sparse (Qdrant) + full-text (PostgreSQL),
fused with Reciprocal Rank Fusion (RRF).

Returns up to `candidate_k` results (default 50) ready to be passed to the reranker.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, SparseVector
from sqlalchemy.orm import Session

logger = logging.getLogger("sciknow.retrieval.hybrid_search")

from sciknow.storage.qdrant import PAPERS_COLLECTION


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class SearchCandidate:
    """A single retrieval candidate before reranking."""
    chunk_id: str          # Qdrant point UUID / PostgreSQL qdrant_point_id
    document_id: str
    section_type: str | None
    section_title: str | None
    content_preview: str   # short snippet for display / reranker
    rrf_score: float
    # metadata (populated from PostgreSQL after RRF fusion)
    title: str | None = None
    year: int | None = None
    authors: list[dict] = field(default_factory=list)
    journal: str | None = None
    doi: str | None = None
    # Citation count: how many other papers in the corpus cite this paper.
    # Populated by _apply_citation_boost after hydration.
    citation_count: int = 0
    # Phase 32.8 — Layer 2: useful_count is how many times this chunk was
    # cited in a FINISHED autowrite draft (was_cited=true in
    # autowrite_retrievals). Populated by _apply_useful_boost.
    useful_count: int = 0
    # Phase 54.6.70 (#9) — co-citation count: how many edges in the
    # citation graph connect this candidate's document to the retrieval's
    # anchor set (top-N candidates). Populated by _apply_cocite_boost.
    cocite_count: int = 0
    # Phase 54.6.81 (#10 part 2) — paper_type classification tag for
    # retrieval weighting + GUI filtering. Defaults to empty string;
    # populated by _apply_paper_type_weight when enabled.
    paper_type: str = ""


# ── RRF ───────────────────────────────────────────────────────────────────────

_RRF_K = 60  # standard constant


def _rrf_merge(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists into a single scored ordering using RRF.

    Each list contains chunk_ids in descending relevance order.
    Returns [(chunk_id, score), ...] sorted by descending score.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[str, float] = {}
    for ranked, w in zip(ranked_lists, weights):
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + w / (_RRF_K + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Query embedding ────────────────────────────────────────────────────────────

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from FlagEmbedding import BGEM3FlagModel
        from sciknow.config import settings
        from sciknow.retrieval.device import load_with_cpu_fallback
        # Phase 15.2 — falls back to CPU when the GPU is mostly full of LLM.
        _embed_model = load_with_cpu_fallback(
            BGEM3FlagModel, settings.embedding_model, use_fp16=True,
        )
    return _embed_model


def release_embed_model() -> None:
    """Drop the cached query embedding model and free VRAM."""
    global _embed_model
    if _embed_model is None:
        return
    try:
        del _embed_model
    finally:
        _embed_model = None
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _embed_query(query: str) -> tuple[list[float], SparseVector]:
    model = _get_embed_model()
    output = model.encode(
        [query],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = output["dense_vecs"][0].tolist()
    lw = output["lexical_weights"][0]
    sparse = SparseVector(
        indices=[int(k) for k in lw.keys()],
        values=[float(v) for v in lw.values()],
    )
    return dense, sparse


# ── Filter helpers ─────────────────────────────────────────────────────────────

def _build_qdrant_filter(
    year_from: int | None,
    year_to: int | None,
    domain: str | None,
    section: str | None,
    topic_cluster: str | None = None,
    has_table: bool | None = None,
    has_equation: bool | None = None,
) -> Filter | None:
    conditions = []
    if year_from is not None or year_to is not None:
        conditions.append(
            FieldCondition(
                key="year",
                range=Range(
                    gte=year_from,
                    lte=year_to,
                ),
            )
        )
    if domain:
        conditions.append(FieldCondition(key="domains", match=MatchValue(value=domain)))
    if section:
        conditions.append(FieldCondition(key="section_type", match=MatchValue(value=section)))
    if topic_cluster:
        conditions.append(FieldCondition(key="topic_cluster", match=MatchValue(value=topic_cluster)))
    if has_table:
        conditions.append(FieldCondition(key="has_table", match=MatchValue(value=True)))
    if has_equation:
        conditions.append(FieldCondition(key="has_equation", match=MatchValue(value=True)))

    if not conditions:
        return None
    return Filter(must=conditions)


# ── Individual search legs ─────────────────────────────────────────────────────

def _search_params() -> "SearchParams | None":
    """Query-time HNSW ef + quantization rescoring, driven by settings."""
    from sciknow.config import settings
    from qdrant_client.models import QuantizationSearchParams, SearchParams
    return SearchParams(
        hnsw_ef=settings.qdrant_hnsw_ef,
        quantization=QuantizationSearchParams(rescore=True)
        if settings.qdrant_scalar_quantization
        else None,
    )


def _qdrant_dense(
    client: QdrantClient,
    dense_vec: list[float],
    top_k: int,
    qdrant_filter: Filter | None,
) -> list[str]:
    response = client.query_points(
        collection_name=PAPERS_COLLECTION,
        query=dense_vec,
        using="dense",
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
        search_params=_search_params(),
    )
    return [str(p.id) for p in response.points]


def _qdrant_sparse(
    client: QdrantClient,
    sparse_vec: SparseVector,
    top_k: int,
    qdrant_filter: Filter | None,
) -> list[str]:
    response = client.query_points(
        collection_name=PAPERS_COLLECTION,
        query=sparse_vec,
        using="sparse",
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )
    return [str(p.id) for p in response.points]


def _postgres_fts(
    session: Session,
    query: str,
    top_k: int,
    year_from: int | None,
    year_to: int | None,
    domain: str | None,
    section: str | None,
    topic_cluster: str | None = None,
) -> list[str]:
    """
    Full-text search over ``chunks.search_vector`` (chunk body content,
    English dictionary). Returns chunk ``qdrant_point_id`` values in
    ts_rank_cd order.

    Phase 54.6.136 — switched from paper-level FTS (title+abstract+
    keywords on ``paper_metadata.search_vector``) to chunk-level FTS.
    Paper-level FTS was structurally disjoint from dense/sparse (which
    operate on chunk content), which the bench's signal-overlap probe
    caught as sparse ∩ FTS ≈ 0.0. The chunk-level index is a true
    lexical complement to sparse: it surfaces chunks whose body text
    contains the query terms exactly, catching rare formulas / numbers
    / uncommon terminology that don't appear in titles or abstracts.
    Paper-level relevance is still captured by dense embeddings +
    the separate abstracts collection.
    """
    from sqlalchemy import text

    extra_conditions = []
    params: dict = {"query": query, "top_k": top_k}

    if year_from is not None:
        extra_conditions.append("pm.year >= :year_from")
        params["year_from"] = year_from
    if year_to is not None:
        extra_conditions.append("pm.year <= :year_to")
        params["year_to"] = year_to
    if domain:
        extra_conditions.append(":domain = ANY(pm.domains)")
        params["domain"] = domain
    if section:
        extra_conditions.append("c.section_type = :section")
        params["section"] = section
    if topic_cluster:
        extra_conditions.append("pm.topic_cluster = :topic_cluster")
        params["topic_cluster"] = topic_cluster

    # The filter-aware joins to paper_metadata / documents are kept
    # so payload filters (year, domain, topic_cluster, canonical) work
    # the same as before — only the FTS predicate moved tables.
    needs_pm = any("pm." in c for c in extra_conditions)
    pm_join = "JOIN paper_metadata pm ON pm.document_id = c.document_id" if needs_pm else ""
    extra_where = ("AND " + " AND ".join(extra_conditions)) if extra_conditions else ""

    sql = text(f"""
        SELECT c.qdrant_point_id::text
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        {pm_join}
        WHERE c.search_vector @@ websearch_to_tsquery('english', :query)
          AND c.qdrant_point_id IS NOT NULL
          AND d.canonical_document_id IS NULL
          {extra_where}
        ORDER BY ts_rank_cd(c.search_vector, websearch_to_tsquery('english', :query)) DESC
        LIMIT :top_k
    """)

    rows = session.execute(sql, params).fetchall()
    return [row[0] for row in rows if row[0]]


# ── Metadata hydration ─────────────────────────────────────────────────────────

def _hydrate(
    client: QdrantClient,
    session: Session,
    ranked: list[tuple[str, float]],
) -> list[SearchCandidate]:
    """
    Fetch payload from Qdrant + paper metadata from PostgreSQL for the merged list.

    Handles two kinds of points:

    - Leaf chunks (`node_level == 0` or unset): join to paper_metadata for
      title / year / authors / journal / doi.
    - RAPTOR summary nodes (`node_level >= 1`): populate the candidate's
      title / year / section_type directly from the Qdrant payload, and put
      the full `summary_text` into `content_preview` so context_builder.build
      uses it as the chunk content (these nodes have no row in the chunks
      table, so the build() join falls through to content_preview).
    """
    if not ranked:
        return []

    ids = [r[0] for r in ranked]
    score_map = {r[0]: r[1] for r in ranked}

    # Phase 54.6.22 — fetch ONLY the payload fields we actually consume
    # in the loop below. The papers collection's payload also carries
    # domains / authors_short / journal / year etc. that are either
    # populated from PG via meta_rows (leaf path) or overridden by
    # year_max/year_min (RAPTOR path), so pulling them on every search
    # is wasted bytes and JSON parse work — typically 50-100 chunks
    # per query, multiplied across every search. Qdrant's REST API
    # honours the include-list and skips serialising the rest.
    _PAYLOAD_FIELDS = [
        "node_level", "document_id", "section_type", "section_title",
        "content_preview",
        # RAPTOR-only:
        "summary_text", "n_documents", "document_ids",
        "year_max", "year_min", "title",
    ]
    points = client.retrieve(
        collection_name=PAPERS_COLLECTION,
        ids=ids,
        with_payload=_PAYLOAD_FIELDS,
        with_vectors=False,
    )
    payload_map = {str(p.id): p.payload for p in points}

    # Collect unique document_ids — but ONLY from leaf nodes. RAPTOR
    # summary nodes don't have a single source document.
    doc_ids = set()
    for p in points:
        if not p.payload:
            continue
        if int(p.payload.get("node_level") or 0) >= 1:
            continue  # skip summary nodes
        did = p.payload.get("document_id")
        if did:
            doc_ids.add(did)

    # Fetch metadata from PostgreSQL
    from sqlalchemy import text
    meta_rows = {}
    # Phase 54.6.125 (Tier 3 #3) — hide non-canonical documents from
    # retrieval. The INNER JOIN to ``documents`` with
    # canonical_document_id IS NULL drops preprint rows that have been
    # reconciled with their journal counterpart; their chunks are
    # still in Qdrant but never surface in search results.
    non_canonical_doc_ids: set[str] = set()
    if doc_ids:
        placeholders = ", ".join(f":d{i}" for i, _ in enumerate(doc_ids))
        params = {f"d{i}": did for i, did in enumerate(doc_ids)}
        rows = session.execute(
            text(f"""
                SELECT pm.document_id::text, pm.title, pm.year,
                       pm.authors, pm.journal, pm.doi,
                       d.canonical_document_id::text
                FROM paper_metadata pm
                JOIN documents d ON d.id = pm.document_id
                WHERE pm.document_id::text IN ({placeholders})
            """),
            params,
        ).fetchall()
        for row in rows:
            if row[6]:  # canonical_document_id != NULL → non-canonical
                non_canonical_doc_ids.add(row[0])
                continue
            meta_rows[row[0]] = {
                "title": row[1],
                "year": row[2],
                "authors": row[3] or [],
                "journal": row[4],
                "doi": row[5],
            }

    candidates = []
    for chunk_id in ids:
        payload = payload_map.get(chunk_id) or {}
        node_level = int(payload.get("node_level") or 0)

        if node_level >= 1:
            # RAPTOR summary node — read everything from payload, no PG join.
            summary_text = payload.get("summary_text") or payload.get("content_preview", "")
            n_docs = payload.get("n_documents") or len(payload.get("document_ids") or [])
            year_max = payload.get("year_max")
            year_min = payload.get("year_min")
            # Display year: prefer the max year so the citation reflects "as of".
            year_disp = year_max or year_min or None
            # Display title: f"RAPTOR L{n} ({n_docs} papers): {topic_hint}".
            # We don't always have a topic hint, but the section_title from the
            # builder will be set to the cluster's central concept where possible.
            title = payload.get("title") or (
                f"RAPTOR L{node_level} ({n_docs} papers)"
                if n_docs else f"RAPTOR L{node_level} synthesis"
            )
            candidates.append(SearchCandidate(
                chunk_id=chunk_id,
                document_id=payload.get("document_id") or "",
                section_type=payload.get("section_type") or f"raptor_l{node_level}",
                section_title=payload.get("section_title") or "",
                content_preview=summary_text,  # full summary text goes here
                rrf_score=score_map[chunk_id],
                title=title,
                year=year_disp,
                authors=[],
                journal=None,
                doi=None,
            ))
            continue

        # Leaf chunk: standard PG metadata join.
        doc_id = payload.get("document_id", "")
        # Phase 54.6.125 — drop chunks whose document was reconciled
        # with a canonical counterpart (non-canonical). Their chunks
        # stay in Qdrant but never surface in search results.
        if doc_id in non_canonical_doc_ids:
            continue
        meta = meta_rows.get(doc_id, {})
        candidates.append(SearchCandidate(
            chunk_id=chunk_id,
            document_id=doc_id,
            section_type=payload.get("section_type"),
            section_title=payload.get("section_title"),
            content_preview=payload.get("content_preview", ""),
            rrf_score=score_map[chunk_id],
            title=meta.get("title"),
            year=meta.get("year"),
            authors=meta.get("authors", []),
            journal=meta.get("journal"),
            doi=meta.get("doi"),
        ))
    return candidates


# ── Citation boost ─────────────────────────────────────────────────────────────

def _apply_citation_boost(
    candidates: list[SearchCandidate],
    session: Session,
    boost_factor: float = 0.0,
) -> list[SearchCandidate]:
    """
    Apply a log-dampened multiplicative citation boost to RRF scores.

    For each candidate, look up how many other corpus papers cite the same
    document. The boost formula:

        boosted_score = rrf_score × (1 + boost_factor × log2(1 + citation_count))

    At the default boost_factor=0.1:
        0 citations → ×1.00  (no change)
        5 citations → ×1.26
       10 citations → ×1.35
       50 citations → ×1.57

    This is a gentle nudge, not a takeover — the three retrieval signals
    (dense + sparse + FTS) still dominate. The reranker runs AFTER this
    and can still override the boost for any genuinely irrelevant chunk.

    If `boost_factor` is 0.0, the function is a no-op (just populates
    citation_count without modifying scores).
    """
    if not candidates:
        return candidates

    # RAPTOR summary nodes have empty document_id — skip them entirely.
    doc_ids = {c.document_id for c in candidates if c.document_id}
    if not doc_ids:
        return candidates

    from sqlalchemy import text

    placeholders = ", ".join(f":d{i}" for i, _ in enumerate(doc_ids))
    params = {f"d{i}": did for i, did in enumerate(doc_ids)}
    rows = session.execute(
        text(f"""
            SELECT cited_document_id::text, COUNT(*) AS cnt
            FROM citations
            WHERE cited_document_id IS NOT NULL
              AND cited_document_id::text IN ({placeholders})
            GROUP BY cited_document_id
        """),
        params,
    ).fetchall()
    cite_counts = {r[0]: r[1] for r in rows}

    import math

    for c in candidates:
        count = cite_counts.get(c.document_id, 0)
        c.citation_count = count
        if boost_factor > 0 and count > 0:
            c.rrf_score *= 1.0 + boost_factor * math.log2(1 + count)

    if boost_factor > 0:
        candidates.sort(key=lambda c: c.rrf_score, reverse=True)

    return candidates


# ── Co-citation / bib-coupling boost (Phase 54.6.70 — #9) ────────────────────

def _apply_cocite_boost(
    candidates: list[SearchCandidate],
    session: Session,
    boost_factor: float = 0.0,
    anchor_size: int = 20,
) -> list[SearchCandidate]:
    """Co-citation / bibliographic-coupling boost WITHIN the retrieved set.

    Distinct from ``_apply_citation_boost`` (which is *global* popularity:
    how many papers anywhere in the corpus cite this one). This boost is
    *topic-local*: for each candidate, count citation edges to the TOP-K
    anchor set from the same query's retrieval. A candidate that cites
    or is cited by several other top hits is in the citation neighborhood
    of this query — topical relevance signal independent of text
    embedding similarity.

    References: Kessler 1963 (bibliographic coupling), Small 1973
    (co-citation). Both classical IR techniques; not GraphRAG-global,
    which was rejected in RESEARCH.md §526 — that was entity-graph
    community summaries, a different axis.

    Formula:
      boosted_score = rrf_score × (1 + boost_factor × log2(1 + n_edges))

    where n_edges counts edges in EITHER direction between the
    candidate's document_id and the top-``anchor_size`` candidates'
    document_ids (excluding self). When the corpus citation graph is
    sparse (sciknow currently has ~4.5% in-corpus resolution per
    RESEARCH.md), most candidates get 0 edges and the boost is a no-op.

    ``boost_factor=0.0`` is a clean no-op (doesn't even run the SQL).
    """
    if boost_factor <= 0 or not candidates:
        return candidates

    # Anchor set: the top-N candidates by current score.
    anchors = [c for c in candidates[:anchor_size] if c.document_id]
    if len(anchors) < 2:
        # Need at least 2 anchors for edges to mean anything.
        return candidates
    anchor_ids = {a.document_id for a in anchors}

    # Collect every candidate doc (anchors + tail) — we want edges for
    # the tail too, so a tail candidate that cites two anchors can still
    # get boosted and float up.
    all_doc_ids = {c.document_id for c in candidates if c.document_id}
    if not all_doc_ids:
        return candidates

    from sqlalchemy import text

    # One SQL query: fetch every citation edge where EITHER end is an
    # anchor AND the other end is any candidate (anchors or tail). We
    # index-filter on cited_document_id IS NOT NULL because unresolved
    # cites (external papers) have no signal here.
    anchor_ph = ", ".join(f":a{i}" for i, _ in enumerate(anchor_ids))
    all_ph = ", ".join(f":c{i}" for i, _ in enumerate(all_doc_ids))
    params: dict = {}
    for i, d in enumerate(anchor_ids):
        params[f"a{i}"] = d
    for i, d in enumerate(all_doc_ids):
        params[f"c{i}"] = d
    rows = session.execute(
        text(f"""
            SELECT citing_document_id::text, cited_document_id::text
            FROM citations
            WHERE cited_document_id IS NOT NULL
              AND (
                (citing_document_id::text IN ({anchor_ph})
                 AND cited_document_id::text IN ({all_ph}))
                OR
                (cited_document_id::text IN ({anchor_ph})
                 AND citing_document_id::text IN ({all_ph}))
              )
        """),
        params,
    ).fetchall()

    # Tally edges per candidate (exclude self-loops).
    edge_counts: dict[str, int] = {}
    for citing, cited in rows:
        if citing == cited:
            continue
        # Which direction connects a candidate to an anchor?
        if citing in anchor_ids and cited in all_doc_ids and cited != citing:
            edge_counts[cited] = edge_counts.get(cited, 0) + 1
        if cited in anchor_ids and citing in all_doc_ids and citing != cited:
            edge_counts[citing] = edge_counts.get(citing, 0) + 1

    if not edge_counts:
        # Sparse corpus → nothing to boost; skip the sort.
        return candidates

    import math

    for c in candidates:
        n = edge_counts.get(c.document_id, 0)
        # Exclude self-citations by NOT boosting anchors for edges TO
        # themselves — already handled above (citing == cited skipped).
        if n > 0:
            c.rrf_score *= 1.0 + boost_factor * math.log2(1 + n)
            # Stash the count on the candidate so downstream consumers
            # (GUI, bench diag) can surface it.
            c.cocite_count = n

    candidates.sort(key=lambda c: c.rrf_score, reverse=True)
    return candidates


# ── Paper-type weighting (Phase 54.6.81 — #10 part 2) ────────────────────────

# Default per-type multiplicative weights. Tuned for factual corpus-
# retrieval use: peer-reviewed / preprint / thesis / book_chapter
# count as "legit research" (1.0); editorial + policy + unknown are
# neutral-ish (0.7 / 0.7 / 0.8); opinion is down-weighted hardest
# (0.4) because on factual queries an opinion piece is mostly a
# liability even when it mentions the right terms.
_DEFAULT_PAPER_TYPE_WEIGHTS: dict[str, float] = {
    "peer_reviewed": 1.0,
    "preprint":      1.0,
    "thesis":        1.0,
    "book_chapter":  1.0,
    "editorial":     0.7,
    "policy":        0.7,
    "unknown":       0.8,
    "opinion":       0.4,
}


def _apply_paper_type_weight(
    candidates: list[SearchCandidate],
    session: Session,
    enabled: bool = False,
) -> list[SearchCandidate]:
    """Multiply each candidate's rrf_score by a paper-type-specific
    weight. When ``enabled=False`` this is a pure no-op (doesn't even
    run the SQL), so the classifier backfill from Phase 54.6.80 can
    complete before any ranking effect kicks in.

    Rows whose document has NULL ``paper_type`` (never classified)
    fall back to the 'unknown' weight (0.8).
    """
    if not enabled or not candidates:
        return candidates

    # Allow .env override: PAPER_TYPE_WEIGHTS='{"opinion": 0.2, ...}'
    from sciknow.config import settings
    override = getattr(settings, "paper_type_weights", None)
    weights = dict(_DEFAULT_PAPER_TYPE_WEIGHTS)
    if isinstance(override, dict) and override:
        weights.update({k: float(v) for k, v in override.items()})

    doc_ids = {c.document_id for c in candidates if c.document_id}
    if not doc_ids:
        return candidates

    from sqlalchemy import text
    placeholders = ", ".join(f":d{i}" for i, _ in enumerate(doc_ids))
    params = {f"d{i}": d for i, d in enumerate(doc_ids)}
    rows = session.execute(text(f"""
        SELECT pm.document_id::text, COALESCE(pm.paper_type, 'unknown')
        FROM paper_metadata pm
        WHERE pm.document_id::text IN ({placeholders})
    """), params).fetchall()
    type_by_doc = {r[0]: r[1] for r in rows}

    changed = False
    for c in candidates:
        t = type_by_doc.get(c.document_id, "unknown")
        w = weights.get(t, 0.8)
        c.paper_type = t
        if w != 1.0:
            c.rrf_score *= w
            changed = True
    if changed:
        candidates.sort(key=lambda c: c.rrf_score, reverse=True)
    return candidates


# ── Useful-count boost (Phase 32.8 — Compound learning Layer 2) ────────────────

def _apply_useful_boost(
    candidates: list[SearchCandidate],
    session: Session,
    boost_factor: float = 0.0,
) -> list[SearchCandidate]:
    """
    Phase 32.8 — apply a log-dampened boost based on how often each
    candidate chunk has been cited in a finished autowrite draft.

    The data source is autowrite_retrievals.was_cited (set by
    _finalize_autowrite_run when the final draft text is parsed for
    [N] markers — see Phase 32.6 / Layer 0). For each candidate's
    chunk_qdrant_id, we count how many distinct runs cited it in
    their final draft, then apply:

        boosted_score = rrf_score × (1 + boost_factor × log2(1 + useful_count))

    At the default boost_factor=0.15:
        0 useful   → ×1.00  (no change — cold start, no learning yet)
        1 useful   → ×1.15
        3 useful   → ×1.30
        7 useful   → ×1.45
       15 useful   → ×1.60

    The signal is intentionally stronger than citation_boost (0.15
    vs 0.1) because useful_count is a more direct relevance indicator:
    citation_count tracks passive popularity ("other authors cite this
    paper"), while useful_count tracks active utility ("this chunk
    actually made it into a finished draft on similar work"). It's
    transfer learning across the user's library — chunks that the
    autowrite loop has already proven useful for one section start
    ranking higher for similar future sections.

    If `boost_factor` is 0.0, the function is a no-op (just populates
    useful_count without modifying scores). The reranker still runs
    after this and can override the boost for any genuinely irrelevant
    chunk.
    """
    if not candidates:
        return candidates

    chunk_ids = {c.chunk_id for c in candidates if c.chunk_id}
    if not chunk_ids:
        return candidates

    from sqlalchemy import text

    placeholders = ", ".join(f":c{i}" for i, _ in enumerate(chunk_ids))
    params = {f"c{i}": cid for i, cid in enumerate(chunk_ids)}
    try:
        rows = session.execute(
            text(f"""
                SELECT chunk_qdrant_id::text, COUNT(DISTINCT run_id) AS cnt
                FROM autowrite_retrievals
                WHERE was_cited = true
                  AND chunk_qdrant_id::text IN ({placeholders})
                GROUP BY chunk_qdrant_id
            """),
            params,
        ).fetchall()
    except Exception as exc:
        # Layer 2 is purely additive — a SQL hiccup must never break
        # the search path. Log and skip the boost; the citation_boost
        # already-applied scores stand.
        import logging
        logging.getLogger(__name__).warning(
            "useful_count boost lookup failed: %s", exc,
        )
        return candidates
    use_counts = {r[0]: r[1] for r in rows}

    import math

    for c in candidates:
        count = use_counts.get(c.chunk_id, 0)
        c.useful_count = count
        if boost_factor > 0 and count > 0:
            c.rrf_score *= 1.0 + boost_factor * math.log2(1 + count)

    if boost_factor > 0:
        candidates.sort(key=lambda c: c.rrf_score, reverse=True)

    return candidates


# ── Query expansion ────────────────────────────────────────────────────────────

def expand_query(query: str) -> str:
    """
    Use the fast LLM to expand a search query with synonyms and related terms.

    Returns the expanded query string, or the original query on any failure
    (Ollama unreachable, timeout, bad output). The expansion is designed to
    improve recall for terse queries: e.g. "solar forcing" → "solar forcing
    total solar irradiance TSI sunspot cycle solar variability climate".

    Uses LLM_FAST_MODEL (qwen3:30b-a3b by default) for speed (~1 s). Never blocks
    the search pipeline — falls through silently on error.
    """
    try:
        from sciknow.rag.llm import complete

        system = (
            "You are a scientific search query expander. Given a short search "
            "query about scientific papers, output an expanded version that "
            "includes synonyms, acronyms, related technical terms, and alternate "
            "phrasings that researchers might use. Keep the original query terms "
            "and ADD related terms. Output ONLY the expanded query, no "
            "explanation, no bullet points, no formatting. Keep it under 60 words."
        )
        from sciknow.config import settings
        expanded = complete(
            system, query,
            model=settings.llm_fast_model,
            temperature=0.1,
            num_ctx=512,
        ).strip()

        # Sanity: if LLM returned something weird or empty, use original
        if not expanded or len(expanded) < len(query) or len(expanded) > 500:
            return query
        return expanded
    except Exception:
        return query


# ── Public API ─────────────────────────────────────────────────────────────────

def search(
    query: str,
    qdrant_client: QdrantClient,
    session: Session,
    candidate_k: int = 50,
    year_from: int | None = None,
    year_to: int | None = None,
    domain: str | None = None,
    section: str | None = None,
    topic_cluster: str | None = None,
    weights: tuple[float, float, float] = (1.0, 1.0, 0.5),
    use_query_expansion: bool = False,
    has_table: bool | None = None,
    has_equation: bool | None = None,
) -> list[SearchCandidate]:
    """
    Run hybrid search and return up to `candidate_k` results sorted by RRF score,
    with optional query expansion and citation-count boost.

    weights = (dense_weight, sparse_weight, fts_weight)
    """
    from sciknow.config import settings

    # Phase 52 — sanitise the incoming query before embedding. Defends
    # against the "MCP client leaks a 2000-char system prompt in
    # front of the real question" failure mode where bge-m3's output
    # ends up dominated by boilerplate and recall silently collapses.
    # Passthrough path is zero-cost for short clean queries.
    from sciknow.retrieval.query_sanitizer import sanitize_query as _sanitize
    _san = _sanitize(query)
    if _san.method != "passthrough":
        logger.info(
            "query sanitised: method=%s original_len=%d clean_len=%d",
            _san.method, _san.original_len, _san.clean_len,
        )
    effective_query = _san.clean_query or query
    if use_query_expansion:
        effective_query = expand_query(effective_query)

    dense_vec, sparse_vec = _embed_query(effective_query)
    qdrant_filter = _build_qdrant_filter(
        year_from, year_to, domain, section, topic_cluster,
        has_table=has_table, has_equation=has_equation,
    )

    dense_ids  = _qdrant_dense(qdrant_client, dense_vec, candidate_k, qdrant_filter)
    sparse_ids = _qdrant_sparse(qdrant_client, sparse_vec, candidate_k, qdrant_filter)
    fts_ids    = _postgres_fts(session, query, candidate_k, year_from, year_to, domain, section, topic_cluster)

    merged = _rrf_merge(
        [dense_ids, sparse_ids, fts_ids],
        weights=list(weights),
    )[:candidate_k]

    candidates = _hydrate(qdrant_client, session, merged)

    boost = getattr(settings, "citation_boost_factor", 0.1)
    candidates = _apply_citation_boost(candidates, session, boost_factor=boost)

    # Phase 32.8 — Layer 2: useful-chunk boost. Applied AFTER the
    # citation boost so the two effects compose multiplicatively.
    # Both are gentle log-dampened nudges; together they raise the
    # ranking of chunks that are both well-cited in the literature
    # AND have proven useful in past autowrite drafts.
    useful_boost = getattr(settings, "useful_count_boost_factor", 0.15)
    candidates = _apply_useful_boost(candidates, session, boost_factor=useful_boost)

    # Phase 54.6.81 (#10 part 2) — paper-type weighting. Multiplies
    # rrf_score by the per-type weight (default 1.0 for research kinds,
    # 0.7 for editorial/policy, 0.4 for opinion, 0.8 for unknown). Off
    # by default (enable_weight=False) because it requires the
    # classifier backfill from #10 part 1 to have populated paper_type.
    candidates = _apply_paper_type_weight(
        candidates, session,
        enabled=bool(getattr(settings, "paper_type_weighting", False)),
    )

    # Phase 54.6.70 (#9) — co-citation / bib-coupling boost inside the
    # retrieved set. Topic-local citation-graph signal. Default off
    # (boost_factor=0.1 is gentle but still noticeable); set
    # COCITE_BOOST_FACTOR=0.0 in .env to disable if it regresses MRR
    # on a corpus with dense in-corpus citations.
    cocite_boost = getattr(settings, "cocite_boost_factor", 0.1)
    return _apply_cocite_boost(candidates, session, boost_factor=cocite_boost)
