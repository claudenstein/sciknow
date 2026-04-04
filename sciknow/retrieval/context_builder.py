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
) -> list[SearchResult]:
    """
    Hydrate candidates with full chunk content from PostgreSQL and return
    SearchResult objects ranked by their input order (already sorted by score).
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
    for rank, candidate in enumerate(candidates, start=1):
        results.append(SearchResult(
            rank=rank,
            score=candidate.rrf_score,
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            section_type=candidate.section_type,
            section_title=candidate.section_title,
            content=content_map.get(candidate.chunk_id, candidate.content_preview),
            title=candidate.title,
            year=candidate.year,
            authors=candidate.authors,
            journal=candidate.journal,
            doi=candidate.doi,
        ))
    return results
