"""
Build full-text context from ranked search candidates.

Fetches the complete chunk content from PostgreSQL (the Qdrant payload only
stores a short preview), and formats results for display or RAG context injection.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from sciknow.retrieval.hybrid_search import SearchCandidate


@dataclass
class SearchResult:
    """Fully hydrated search result with complete chunk text."""
    rank: int
    score: float
    chunk_id: str
    document_id: str
    section_type: str | None
    section_title: str | None
    content: str           # full chunk text (from PostgreSQL)
    title: str | None
    year: int | None
    authors: list[dict]
    journal: str | None
    doi: str | None
    citation_count: int = 0
    # Phase 32.8 — Layer 2: how many times this chunk was cited in a
    # finished autowrite draft (data source: autowrite_retrievals.was_cited).
    # Powers the useful_count retrieval boost — see hybrid_search._apply_useful_boost.
    useful_count: int = 0

    @property
    def citation(self) -> str:
        """APA-style citation string: Last, F., et al. (year). Title. Journal. doi:..."""
        from sciknow.rag.prompts import format_authors_apa, _apa_citation
        return _apa_citation(self, self.rank)

    def format_for_rag(self) -> str:
        """Format chunk for insertion into an LLM context window."""
        header = f"[{self.section_type or 'text'}] {self.title or 'Unknown'}"
        if self.year:
            header += f" ({self.year})"
        return f"{header}\n\n{self.content}"


def build(
    candidates: list[SearchCandidate],
    session: Session,
    *,
    min_content_chars: int = 200,
) -> list[SearchResult]:
    """
    Hydrate candidates with full chunk content from PostgreSQL and return
    SearchResult objects ranked by their input order (already sorted by score).

    Phase 55.V19 — `min_content_chars` filters out hydrated chunks whose
    body is too short to serve as a citation source. Corpus audit
    (2026-04-28) showed 1047 of 33,136 chunks (~3%) are <200 chars —
    almost all are TOC entries, page numbers, dates, or section
    headers without body text. They rank highly via sparse keyword
    matching but cannot ground a claim. Set to 0 to disable.
    """
    if not candidates:
        return []

    from sqlalchemy import text

    # Fetch full content for all chunk qdrant_point_ids at once
    placeholders = ", ".join(f":id{i}" for i, _ in enumerate(candidates))
    params = {f"id{i}": c.chunk_id for i, c in enumerate(candidates)}

    rows = session.execute(
        text(f"""
            SELECT qdrant_point_id::text, content
            FROM chunks
            WHERE qdrant_point_id::text IN ({placeholders})
        """),
        params,
    ).fetchall()

    content_map = {row[0]: row[1] for row in rows}

    results = []
    for candidate in candidates:
        content = content_map.get(candidate.chunk_id, candidate.content_preview)
        if min_content_chars > 0 and content and len(content) < min_content_chars:
            # Skip header-only / TOC / page-number chunks — they retrieve
            # well via sparse keyword density but have no body to cite.
            continue
        results.append(SearchResult(
            rank=len(results) + 1,
            score=candidate.rrf_score,
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            section_type=candidate.section_type,
            section_title=candidate.section_title,
            content=content,
            title=candidate.title,
            year=candidate.year,
            authors=candidate.authors,
            journal=candidate.journal,
            doi=candidate.doi,
            citation_count=candidate.citation_count,
            useful_count=getattr(candidate, "useful_count", 0),
        ))
    return results
