"""
Full ingestion pipeline: PDF → markdown → metadata → sections → chunks → embeddings.

Each stage updates documents.ingestion_status in PostgreSQL so progress is
visible and failures are recoverable.
"""
import hashlib
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from sciknow.config import settings
from sciknow.ingestion import chunker, embedder, metadata, pdf_converter
from sciknow.storage.db import get_session
from sciknow.storage.models import (
    Chunk,
    Citation,
    Document,
    IngestionJob,
    PaperMetadata,
    PaperSection,
)
from sciknow.storage.qdrant import get_client as get_qdrant


class PipelineError(Exception):
    pass


class AlreadyIngested(Exception):
    def __init__(self, document_id: UUID):
        self.document_id = document_id
        super().__init__(str(document_id))


def _delete_qdrant_vectors(qdrant, document_id: str) -> None:
    """Remove all Qdrant points for a document before re-ingesting it."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from sciknow.storage.qdrant import PAPERS_COLLECTION

    try:
        qdrant.delete(
            collection_name=PAPERS_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )
    except Exception:
        pass  # collection may not exist yet on first run


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _log_job(
    session: Session,
    document_id: UUID | None,
    stage: str,
    status: str,
    duration_ms: int = 0,
    details: dict | None = None,
) -> None:
    session.add(IngestionJob(
        document_id=document_id,
        stage=stage,
        status=status,
        duration_ms=duration_ms,
        details=details or {},
    ))
    session.flush()


def _set_status(session: Session, doc: Document, status: str) -> None:
    doc.ingestion_status = status
    session.flush()


def ingest(
    pdf_path: Path,
    force: bool = False,
    ingest_source: str = "seed",
) -> UUID:
    """
    Ingest a single PDF. Returns the document UUID.

    force=True  — re-ingest even if already complete. Deletes existing
                  sections, chunks, and Qdrant vectors before reprocessing.
    force=False — raises AlreadyIngested if the PDF has been seen before
                  and completed successfully.
    ingest_source — provenance tag recorded in documents.ingest_source.
                  'seed'   = manually ingested via CLI (default)
                  'expand' = auto-discovered via `sciknow db expand`
                  Applied only on first insert; existing rows keep their
                  original source (so a failed seed paper re-tried from
                  expand stays tagged 'seed').
    """
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise PipelineError(f"File not found: {pdf_path}")

    file_hash = _sha256(pdf_path)
    qdrant = get_qdrant()

    with get_session() as session:
        existing = session.query(Document).filter_by(file_hash=file_hash).first()
        if existing:
            if existing.ingestion_status == "complete" and not force:
                raise AlreadyIngested(existing.id)
            # Force re-ingest or resume a failed/partial ingest
            doc = existing
            if force:
                # Delete stale Qdrant vectors for this document
                _delete_qdrant_vectors(qdrant, str(existing.id))
            doc.ingestion_status = "pending"
            doc.ingestion_error = None
        else:
            doc = Document(
                file_hash=file_hash,
                original_path=str(pdf_path),
                filename=pdf_path.name,
                file_size_bytes=pdf_path.stat().st_size,
                ingestion_status="pending",
                ingest_source=ingest_source,
            )
            session.add(doc)
            session.flush()

        doc_id = doc.id

        try:
            # ------------------------------------------------------------------
            # Stage 1: PDF → JSON (Marker structured output)
            # Falls back to markdown automatically if JSON conversion fails.
            # ------------------------------------------------------------------
            _set_status(session, doc, "converting")
            t0 = time.monotonic()

            output_dir = settings.mineru_output_dir / str(doc_id)
            result = pdf_converter.convert(pdf_path, output_dir)
            doc.mineru_output_path = str(output_dir)

            output_file = result.content_list_path or result.json_path or result.md_path
            _log_job(session, doc_id, "convert", "completed",
                     int((time.monotonic() - t0) * 1000),
                     {
                         "output": str(output_file) if output_file else None,
                         "backend": result.backend,
                         "mode": "json" if result.is_json else "markdown",
                         "chars": len(result.text),
                     })

            # ------------------------------------------------------------------
            # Stage 2: Metadata extraction  (uses plain text from the result)
            # ------------------------------------------------------------------
            _set_status(session, doc, "metadata_extraction")
            t0 = time.monotonic()

            meta = metadata.extract(pdf_path, result.text)

            # Delete old metadata row if resuming
            session.query(PaperMetadata).filter_by(document_id=doc_id).delete()
            pm = PaperMetadata(
                document_id=doc_id,
                title=meta.title,
                abstract=meta.abstract,
                year=meta.year,
                doi=meta.doi,
                arxiv_id=meta.arxiv_id,
                journal=meta.journal,
                volume=meta.volume,
                issue=meta.issue,
                pages=meta.pages,
                publisher=meta.publisher,
                authors=meta.authors,
                keywords=meta.keywords,
                domains=meta.domains,
                metadata_source=meta.source,
                crossref_raw=meta.crossref_raw,
                arxiv_raw=meta.arxiv_raw,
            )
            session.add(pm)
            session.flush()

            _log_job(session, doc_id, "metadata", "completed",
                     int((time.monotonic() - t0) * 1000),
                     {"source": meta.source, "title": meta.title})

            # ------------------------------------------------------------------
            # Stage 3: Section parsing + chunking
            # JSON mode uses block-type-aware section detection.
            # Markdown fallback uses regex heading detection.
            # ------------------------------------------------------------------
            _set_status(session, doc, "chunking")
            t0 = time.monotonic()

            # Clean up previous sections/chunks if resuming
            session.query(PaperSection).filter_by(document_id=doc_id).delete()
            session.query(Chunk).filter_by(document_id=doc_id).delete()

            if result.backend == "mineru":
                sections = chunker.parse_sections_from_mineru(result.content_list)
            elif result.backend == "marker_json":
                sections = chunker.parse_sections_from_json(result.json_data)
            else:
                sections = chunker.parse_sections(result.text)
            db_sections: dict[int, UUID] = {}

            for sec in sections:
                db_sec = PaperSection(
                    document_id=doc_id,
                    section_type=sec.section_type,
                    section_title=sec.section_title,
                    section_index=sec.section_index,
                    content=sec.content,
                    word_count=sec.word_count,
                )
                session.add(db_sec)
                session.flush()
                db_sections[sec.section_index] = db_sec.id

            raw_chunks = chunker.chunk_document(sections, meta.title or "", meta.year)

            db_chunks: list[Chunk] = []
            for rc in raw_chunks:
                section_id = db_sections.get(rc.section_index)
                db_chunk = Chunk(
                    document_id=doc_id,
                    section_id=section_id,
                    chunk_index=rc.chunk_index,
                    section_type=rc.section_type,
                    content=rc.content,
                    content_tokens=rc.content_tokens,
                    char_start=rc.char_start,
                    char_end=rc.char_end,
                )
                session.add(db_chunk)
                db_chunks.append(db_chunk)
            session.flush()

            _log_job(session, doc_id, "chunking", "completed",
                     int((time.monotonic() - t0) * 1000),
                     {"sections": len(sections), "chunks": len(db_chunks)})

            # ------------------------------------------------------------------
            # Stage 4: Embedding + Qdrant upsert
            # ------------------------------------------------------------------
            _set_status(session, doc, "embedding")
            t0 = time.monotonic()

            payload_base = {
                "title": meta.title or "",
                "authors_short": _authors_short(meta.authors),
                "year": meta.year,
                "journal": meta.journal or "",
                "doi": meta.doi or "",
                "domains": meta.domains or [],
            }

            chunker_chunks = raw_chunks  # list[chunker.Chunk]
            point_ids = embedder.embed_chunks(
                chunker_chunks, doc_id, payload_base, qdrant
            )

            now = datetime.now(timezone.utc)
            for db_chunk, point_id in zip(db_chunks, point_ids):
                db_chunk.qdrant_point_id = point_id
                db_chunk.embedded_at = now
                db_chunk.embedding_model = settings.embedding_model

            if meta.abstract:
                embedder.embed_abstract(meta.abstract, doc_id, payload_base, qdrant)

            _log_job(session, doc_id, "embedding", "completed",
                     int((time.monotonic() - t0) * 1000),
                     {"points": len(point_ids)})

            # ------------------------------------------------------------------
            # Done
            # ------------------------------------------------------------------
            _set_status(session, doc, "complete")

            # Move PDF to processed/
            dest = settings.processed_dir / pdf_path.name
            settings.processed_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(pdf_path), str(dest))

            return doc_id

        except Exception as exc:
            doc.ingestion_status = "failed"
            doc.ingestion_error = str(exc)[:2000]
            _log_job(session, doc_id, "pipeline", "failed", details={"error": str(exc)})

            # Move PDF to failed/
            dest = settings.failed_dir / pdf_path.name
            settings.failed_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(pdf_path), str(dest))

            raise PipelineError(str(exc)) from exc


def _authors_short(authors: list[dict]) -> str:
    if not authors:
        return ""
    names = [a.get("name", "") for a in authors[:3]]
    result = ", ".join(n for n in names if n)
    if len(authors) > 3:
        result += " et al."
    return result
