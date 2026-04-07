"""
Multimodal retrieval support — tables, equations, and figure captions
as first-class searchable objects.

Extracts these from existing paper_sections content (already parsed by
the chunker) and makes them independently retrievable via Qdrant
payload filters: section_type = "table" | "equation" | "figure".

This module works on already-ingested data — no re-ingestion needed.
Call ``index_multimodal()`` to scan existing chunks and tag those
containing tables/equations with additional payload metadata.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("sciknow.multimodal")


def extract_tables_from_content(content: str) -> list[str]:
    """Extract pipe-delimited tables from chunk content."""
    tables = []
    lines = content.split("\n")
    table_lines: list[str] = []
    in_table = False

    for line in lines:
        if "|" in line and line.strip().startswith("|"):
            in_table = True
            table_lines.append(line)
        else:
            if in_table and len(table_lines) >= 2:
                tables.append("\n".join(table_lines))
            table_lines = []
            in_table = False

    if in_table and len(table_lines) >= 2:
        tables.append("\n".join(table_lines))

    return tables


def extract_equations_from_content(content: str) -> list[str]:
    """Extract LaTeX equations from chunk content."""
    equations = []
    # Display equations: $$...$$
    equations.extend(re.findall(r'\$\$(.+?)\$\$', content, re.DOTALL))
    # Inline equations: $...$  (only substantial ones, >10 chars)
    for m in re.findall(r'\$([^$]+)\$', content):
        if len(m.strip()) > 10:
            equations.append(m)
    return equations


def tag_multimodal_chunks(
    *,
    batch_size: int = 200,
) -> dict[str, int]:
    """
    Scan all chunks in PostgreSQL and tag Qdrant payloads for chunks
    containing tables or equations with 'has_table' and 'has_equation'
    boolean payload fields.

    Returns {"tables_tagged": N, "equations_tagged": N}.
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import PAPERS_COLLECTION, get_client

    qdrant = get_client()
    tables_tagged = 0
    equations_tagged = 0

    with get_session() as session:
        rows = session.execute(text("""
            SELECT c.qdrant_point_id::text, c.content
            FROM chunks c
            WHERE c.qdrant_point_id IS NOT NULL
        """)).fetchall()

    logger.info("Scanning %d chunks for tables/equations...", len(rows))

    table_ids = []
    equation_ids = []

    for point_id, content in rows:
        if not content:
            continue
        has_table = bool(extract_tables_from_content(content))
        has_equation = bool(extract_equations_from_content(content))

        if has_table:
            table_ids.append(point_id)
        if has_equation:
            equation_ids.append(point_id)

    # Batch update Qdrant payloads
    if table_ids:
        for i in range(0, len(table_ids), batch_size):
            batch = table_ids[i:i + batch_size]
            qdrant.set_payload(
                collection_name=PAPERS_COLLECTION,
                payload={"has_table": True},
                points=batch,
            )
        tables_tagged = len(table_ids)
        logger.info("Tagged %d chunks with has_table=True", tables_tagged)

    if equation_ids:
        for i in range(0, len(equation_ids), batch_size):
            batch = equation_ids[i:i + batch_size]
            qdrant.set_payload(
                collection_name=PAPERS_COLLECTION,
                payload={"has_equation": True},
                points=batch,
            )
        equations_tagged = len(equation_ids)
        logger.info("Tagged %d chunks with has_equation=True", equations_tagged)

    return {"tables_tagged": tables_tagged, "equations_tagged": equations_tagged}
