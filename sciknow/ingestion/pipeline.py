"""
Full ingestion pipeline: PDF → markdown → metadata → sections → chunks → embeddings.

Each stage updates documents.ingestion_status in PostgreSQL so progress is
visible and failures are recoverable.
"""
import hashlib
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger("sciknow.pipeline")

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


def _sanitize(text: str | None) -> str | None:
    """Strip NUL bytes that PostgreSQL rejects in text columns.

    Some PDFs (especially scanned/OCR'd) produce text with embedded \\x00
    from MinerU or Marker. PostgreSQL raises ``ValueError: A string literal
    cannot contain NUL (0x00) characters`` on insert.
    """
    if text is None:
        return None
    return text.replace("\x00", "")


def _log_job(
    session: Session,
    document_id: UUID | None,
    stage: str,
    status: str,
    duration_ms: int = 0,
    details: dict | None = None,
) -> None:
    try:
        session.add(IngestionJob(
            document_id=document_id,
            stage=stage,
            status=status,
            duration_ms=duration_ms,
            details=details or {},
        ))
        session.flush()
    except Exception:
        # If the session is rolled back (e.g. from a prior NUL-byte error),
        # silently skip the job log rather than raising a secondary exception
        # that masks the real root cause.
        try:
            session.rollback()
        except Exception:
            pass


def _set_status(session: Session, doc: Document, status: str) -> None:
    logger.debug(f"  stage → {status}  doc={doc.id}")
    doc.ingestion_status = status
    session.flush()


def _archive_pdf(pdf_path: Path, dest_dir: Path) -> None:
    """
    Record a PDF in `dest_dir` (processed/ or failed/) without duplicating it
    on disk.

    - If the PDF lives under `data/downloads/` (expand-created, we own it):
      **move** it to dest_dir. This stops downloads/ from growing forever.
    - If the PDF is external (user's original, e.g. ~/Papers/...):
      **symlink** into dest_dir. Zero disk cost; the symlink serves as an
      index so `sciknow ingest directory data/processed/` works after a
      db reset without needing the originals to be copied.
    - Falls back to copy if symlinks fail (cross-filesystem edge case).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / pdf_path.name

    # Avoid overwriting an existing file/symlink with the same name.
    if dest.exists() or dest.is_symlink():
        return

    downloads_dir = (settings.data_dir / "downloads").resolve()
    is_our_file = False
    try:
        is_our_file = pdf_path.resolve().is_relative_to(downloads_dir)
    except (ValueError, OSError):
        pass

    if is_our_file:
        shutil.move(str(pdf_path), str(dest))
    else:
        shutil.copy2(str(pdf_path), str(dest))


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

    logger.info(f"INGEST START  {pdf_path.name}  source={ingest_source}")
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

            # Sanitize the converted text before feeding it to metadata extraction
            # and DB insertion — MinerU/Marker can produce NUL bytes from OCR
            # on corrupted scanned pages, which PostgreSQL rejects.
            result.text = _sanitize(result.text) or ""

            meta = metadata.extract(pdf_path, result.text)

            # Delete old metadata row if resuming
            session.query(PaperMetadata).filter_by(document_id=doc_id).delete()
            pm = PaperMetadata(
                document_id=doc_id,
                title=_sanitize(meta.title),
                abstract=_sanitize(meta.abstract),
                year=meta.year,
                doi=_sanitize(meta.doi),
                arxiv_id=_sanitize(meta.arxiv_id),
                journal=_sanitize(meta.journal),
                volume=_sanitize(meta.volume),
                issue=_sanitize(meta.issue),
                pages=_sanitize(meta.pages),
                publisher=_sanitize(meta.publisher),
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
            # Stage 2b: Citation extraction + cross-linking
            #
            # Best-effort: a citation extraction failure must NEVER kill the
            # paper's ingestion. The whole block runs in a try/except so that
            # a malformed reference, constraint violation, or unexpected data
            # degrades gracefully to "paper ingested without citations" rather
            # than "paper failed entirely."
            # ------------------------------------------------------------------
            try:
                session.query(Citation).filter_by(citing_document_id=doc_id).delete()

                from sciknow.ingestion.references import (
                    extract_references_from_crossref,
                    extract_references_from_mineru_content_list,
                    extract_references_from_markdown,
                )

                raw_refs = []
                if meta.crossref_raw:
                    raw_refs.extend(extract_references_from_crossref(meta.crossref_raw))
                if result.backend == "mineru" and result.content_list:
                    raw_refs.extend(
                        extract_references_from_mineru_content_list(result.content_list)
                    )
                elif result.backend == "marker_md" and result.text:
                    raw_refs.extend(extract_references_from_markdown(result.text))

                seen_ref_keys: set[str] = set()
                for ref in raw_refs:
                    key = (ref.doi or "").lower() or (ref.title or "")[:60].lower()
                    if not key or key in seen_ref_keys:
                        continue
                    seen_ref_keys.add(key)

                    cited_doc_id = None
                    if ref.doi:
                        from sqlalchemy import text as _text
                        row = session.execute(
                            _text("SELECT d.id FROM documents d JOIN paper_metadata pm "
                                  "ON pm.document_id = d.id WHERE LOWER(pm.doi) = :doi "
                                  "AND d.ingestion_status = 'complete' LIMIT 1"),
                            {"doi": ref.doi.lower()},
                        ).first()
                        if row:
                            cited_doc_id = row[0]

                    session.add(Citation(
                        citing_document_id=doc_id,
                        cited_doi=ref.doi,
                        cited_title=(ref.title or "")[:500],
                        cited_year=ref.year,
                        cited_document_id=cited_doc_id,
                        raw_reference_text=(ref.raw_text or "")[:400],
                    ))

                if meta.doi:
                    from sqlalchemy import text as _text
                    session.execute(
                        _text("UPDATE citations SET cited_document_id = :doc_id "
                              "WHERE LOWER(cited_doi) = :doi AND cited_document_id IS NULL"),
                        {"doc_id": doc_id, "doi": meta.doi.lower()},
                    )

                session.flush()
            except Exception as cit_exc:
                # Roll back the citation-specific changes but keep the session
                # alive for subsequent stages (chunking, embedding).
                try:
                    session.rollback()
                    # Re-set the document status which was lost in the rollback
                    doc = session.query(Document).filter_by(id=doc_id).first()
                    if doc:
                        doc.ingestion_status = "metadata_extraction"
                    pm = session.query(PaperMetadata).filter_by(document_id=doc_id).first()
                    if not pm:
                        # Re-insert metadata that was lost in the rollback
                        pm = PaperMetadata(
                            document_id=doc_id,
                            title=meta.title, abstract=meta.abstract, year=meta.year,
                            doi=meta.doi, arxiv_id=meta.arxiv_id, journal=meta.journal,
                            volume=meta.volume, issue=meta.issue, pages=meta.pages,
                            publisher=meta.publisher, authors=meta.authors,
                            keywords=meta.keywords, domains=meta.domains,
                            metadata_source=meta.source,
                            crossref_raw=meta.crossref_raw, arxiv_raw=meta.arxiv_raw,
                        )
                        session.add(pm)
                    session.flush()
                except Exception:
                    pass  # if recovery itself fails, let the outer handler catch it
                _log_job(session, doc_id, "citations", "failed", details={
                    "error": str(cit_exc)[:300],
                })

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
                    section_title=_sanitize(sec.section_title) or "",
                    section_index=sec.section_index,
                    content=_sanitize(sec.content) or "",
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
                    content=_sanitize(rc.content) or "",
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
            # Phase 44.1 — evict any Ollama model that the metadata
            # cascade left resident. The bench harness showed bge-m3
            # drops 50× (104 → 2.1 chunks/s) when an LLM squats on VRAM
            # and the retrieval device fallback kicks it to CPU. The
            # metadata extraction's LLM step is the main culprit here
            # (fast model, held for keep_alive). Unloading is cheap
            # (~50 ms) and idempotent; if no model is loaded, nothing
            # happens.
            try:
                from sciknow.rag.llm import release_llm
                released = release_llm()
                if released:
                    logger.info("freed VRAM before embed: %s", released)
            except Exception as exc:
                logger.debug("release_llm pre-embed failed: %s", exc)

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
            _archive_pdf(pdf_path, settings.processed_dir)

            logger.info(f"INGEST OK    {pdf_path.name}  doc={doc_id}")

            # Wiki post-ingest hook: update concept pages if wiki is initialized.
            # Wrapped in try/except so wiki failures never block ingestion.
            try:
                if settings.wiki_dir.exists():
                    from sciknow.core.wiki_ops import update_concepts_for_paper
                    for event in update_concepts_for_paper(str(doc_id)):
                        if event.get("type") == "error":
                            logger.warning("Wiki post-ingest: %s", event.get("message"))
            except Exception as wiki_exc:
                logger.warning("Wiki post-ingest hook failed (non-fatal): %s", wiki_exc)

            return doc_id

        except Exception as exc:
            logger.error(
                f"INGEST FAIL  {pdf_path.name}  doc={doc_id}  "
                f"error={type(exc).__name__}: {exc}",
                exc_info=True,
            )
            doc.ingestion_status = "failed"
            doc.ingestion_error = _sanitize(str(exc)[:2000])
            _log_job(session, doc_id, "pipeline", "failed", details={
                "error": _sanitize(str(exc)[:500]),
            })

            _archive_pdf(pdf_path, settings.failed_dir)

            raise PipelineError(str(exc)) from exc


def _authors_short(authors: list[dict]) -> str:
    if not authors:
        return ""
    names = [a.get("name", "") for a in authors[:3]]
    result = ", ".join(n for n in names if n)
    if len(authors) > 3:
        result += " et al."
    return result
