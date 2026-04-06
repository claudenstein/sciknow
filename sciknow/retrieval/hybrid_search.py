"""
Hybrid search: dense (Qdrant) + sparse (Qdrant) + full-text (PostgreSQL),
fused with Reciprocal Rank Fusion (RRF).

Returns up to `candidate_k` results (default 50) ready to be passed to the reranker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, SparseVector
from sqlalchemy.orm import Session

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
        _embed_model = BGEM3FlagModel(settings.embedding_model, use_fp16=True)
    return _embed_model


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
    Full-text search over paper_metadata.search_vector, joined to chunks.
    Returns chunk qdrant_point_ids in relevance order.
    """
    from sqlalchemy import text

    # Build WHERE clause additions
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

    extra_where = ("AND " + " AND ".join(extra_conditions)) if extra_conditions else ""

    sql = text(f"""
        SELECT c.qdrant_point_id::text
        FROM paper_metadata pm
        JOIN chunks c ON c.document_id = pm.document_id
        WHERE pm.search_vector @@ websearch_to_tsquery('english', :query)
          AND c.qdrant_point_id IS NOT NULL
          {extra_where}
        ORDER BY ts_rank_cd(pm.search_vector, websearch_to_tsquery('english', :query)) DESC
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
    """
    if not ranked:
        return []

    ids = [r[0] for r in ranked]
    score_map = {r[0]: r[1] for r in ranked}

    # Qdrant payload for content_preview, section info
    points = client.retrieve(
        collection_name=PAPERS_COLLECTION,
        ids=ids,
        with_payload=True,
        with_vectors=False,
    )
    payload_map = {str(p.id): p.payload for p in points}

    # Collect unique document_ids
    doc_ids = {p.payload.get("document_id") for p in points if p.payload}
    doc_ids.discard(None)

    # Fetch metadata from PostgreSQL
    from sqlalchemy import text
    meta_rows = {}
    if doc_ids:
        placeholders = ", ".join(f":d{i}" for i, _ in enumerate(doc_ids))
        params = {f"d{i}": did for i, did in enumerate(doc_ids)}
        rows = session.execute(
            text(f"""
                SELECT pm.document_id::text, pm.title, pm.year,
                       pm.authors, pm.journal, pm.doi
                FROM paper_metadata pm
                WHERE pm.document_id::text IN ({placeholders})
            """),
            params,
        ).fetchall()
        for row in rows:
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
        doc_id = payload.get("document_id", "")
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

    doc_ids = {c.document_id for c in candidates}
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


# ── Query expansion ────────────────────────────────────────────────────────────

def expand_query(query: str) -> str:
    """
    Use the fast LLM to expand a search query with synonyms and related terms.

    Returns the expanded query string, or the original query on any failure
    (Ollama unreachable, timeout, bad output). The expansion is designed to
    improve recall for terse queries: e.g. "solar forcing" → "solar forcing
    total solar irradiance TSI sunspot cycle solar variability climate".

    Uses LLM_FAST_MODEL (Mistral 7B by default) for speed (~1 s). Never blocks
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
) -> list[SearchCandidate]:
    """
    Run hybrid search and return up to `candidate_k` results sorted by RRF score,
    with optional query expansion and citation-count boost.

    weights = (dense_weight, sparse_weight, fts_weight)
    """
    from sciknow.config import settings

    effective_query = query
    if use_query_expansion:
        effective_query = expand_query(query)

    dense_vec, sparse_vec = _embed_query(effective_query)
    qdrant_filter = _build_qdrant_filter(year_from, year_to, domain, section, topic_cluster)

    dense_ids  = _qdrant_dense(qdrant_client, dense_vec, candidate_k, qdrant_filter)
    sparse_ids = _qdrant_sparse(qdrant_client, sparse_vec, candidate_k, qdrant_filter)
    fts_ids    = _postgres_fts(session, query, candidate_k, year_from, year_to, domain, section, topic_cluster)

    merged = _rrf_merge(
        [dense_ids, sparse_ids, fts_ids],
        weights=list(weights),
    )[:candidate_k]

    candidates = _hydrate(qdrant_client, session, merged)

    boost = getattr(settings, "citation_boost_factor", 0.1)
    return _apply_citation_boost(candidates, session, boost_factor=boost)
