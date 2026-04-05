# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The `sciknow` CLI lives in the uv-managed venv at `.venv/bin/sciknow`. It is **not** on PATH by default. Run commands one of these ways (from the repo root):

```bash
uv run sciknow <subcommand> ...        # preferred — no activation needed
source .venv/bin/activate && sciknow ... # for interactive sessions
.venv/bin/sciknow <subcommand> ...     # direct
```

Python deps are managed with `uv` against `pyproject.toml` / `uv.lock`. Use `uv sync` to install/update, `uv add <pkg>` to add a dependency. Do **not** use pip directly.

Configuration is loaded from `.env` via Pydantic Settings in `sciknow/config.py`. All paths, ports, model names, batch sizes, and Ollama host come from there — never hardcode them. `CROSSREF_EMAIL` must be set for metadata extraction to work politely.

Runtime services (all native, no Docker): **PostgreSQL 16** (`sciknow`/`sciknow`@localhost:5432), **Qdrant** (6333, systemd user unit), **Ollama** (11434). A remote GPU host is configured only by changing `OLLAMA_HOST` in `.env`.

## Common commands

```bash
# Database / vector store lifecycle
uv run sciknow db init                 # alembic upgrade head + init Qdrant collections
uv run sciknow db reset                # wipe PG + Qdrant + data/processed + data/downloads + mineru_output (destructive)
uv run sciknow db stats                # counts + per-stage status breakdown (use to check resume state)

# Ingestion
uv run sciknow ingest file paper.pdf
uv run sciknow ingest directory ./papers/              # recursive by default
uv run sciknow ingest file paper.pdf --force           # force re-ingest of an already-complete paper

# Growing the library
uv run sciknow db expand                               # follow citations → download OA PDFs → ingest
uv run sciknow db enrich                               # find DOIs for papers missing one (Crossref/OpenAlex/arXiv)

# Migrations (Alembic — config in alembic.ini, versions under migrations/versions/)
uv run alembic revision -m "message"
uv run alembic upgrade head
uv run alembic downgrade -1
```

There is no test suite, no linter, and no formatter configured in this repo. Do not invent `pytest`, `ruff`, or `black` commands — they are not wired up.

Every CLI invocation is logged to `data/sciknow.log`; `sciknow db expand` additionally appends to `data/downloads/expand.log`.

## Architecture

### Ingestion pipeline (one stage feeds the next; state machine lives in Postgres)

The pipeline is orchestrated by `sciknow/ingestion/pipeline.py` and advances each `document` through these stages (visible in `db stats`):

```
pending → converting → metadata_extraction → chunking → embedding → complete
                                                                  → failed
```

1. **Marker** (`ingestion/pdf_converter.py`) — `marker-pdf` converts PDF to a structured **JSON block tree** (`SectionHeader`, `Text`, `Table`, `Equation`, ...). Falls back to Markdown only if JSON rendering fails. Output at `data/mineru_output/{doc_id}/`. Models cached in `~/.cache/datalab/`; Marker models are loaded **once per ingestion run** and kept in memory — batching matters for throughput.
2. **Metadata** (`ingestion/metadata.py`) — 4-layer cascade, each only runs if the previous didn't fully populate fields: PyMuPDF (embedded XMP) → Crossref by DOI → arXiv by ID → Ollama `LLM_FAST_MODEL` on the first ~3000 chars. The source is stored in `paper_metadata.metadata_source` (`crossref` / `arxiv` / `embedded_pdf` / `llm_extracted` / `unknown`), which downstream features (e.g. `db enrich`) filter on.
3. **Chunker** (`ingestion/chunker.py`) — in JSON mode, walks the block tree; h1/h2 headers open sections, h3/h4 become inline bold. Sections are classified into canonical types via `_SECTION_PATTERNS` (`abstract`, `introduction`, `methods`, `results`, `discussion`, `conclusion`, `related_work`, `appendix`, plus non-embedded `references`/`acknowledgments` in `_SKIP_SECTIONS`). Per-type chunking parameters live in `_PARAMS`. Each chunk is prefixed with a context header `[section] Title (year)` so embeddings carry paper identity in isolation.
4. **Embedder** (`ingestion/embedder.py`) — `BAAI/bge-m3` produces dense (1024-dim) **and** sparse lexical vectors simultaneously; both land in Qdrant in a single upsert. `EMBEDDING_BATCH_SIZE` is the VRAM knob (default 8 is conservative for a 3090 with LLM co-resident; 32–64 when only the embedder is loaded).

**Idempotency and resume:** `documents` is keyed on SHA-256 hash of the file bytes, so re-running `sciknow ingest directory …` skips completed papers and retries failed/partial ones from scratch. `data/failed/` is a **copy**; the pipeline re-reads from the original path stored in `documents.original_path`. `--force` is only for deliberately re-ingesting completed papers.

### Retrieval (Phase 2+)

`retrieval/hybrid_search.py` fans out three signals against a single query embedding from the same bge-m3 model used at ingest time:
- Dense vector search (Qdrant)
- Sparse lexical search (Qdrant, same bge-m3 output)
- PostgreSQL full-text search (`tsvector` column on `paper_metadata`, maintained by a trigger defined in the initial migration)

Results are fused with **Reciprocal Rank Fusion** into a top-50 candidate pool, then reranked by `retrieval/reranker.py` (`BAAI/bge-reranker-v2-m3` cross-encoder) into the final top-k. Filters (`--year-from`, `--section`, `--domain`, `--topic`) are pushed down as Qdrant payload filters — that's why there are payload indexes on `document_id`, `section_type`, `year`, `domains`, `journal` in the `papers` collection.

Two Qdrant collections exist: `papers` (all chunks, on-disk) and `abstracts` (paper-level, in-memory) for paper-scoped searches.

### Storage boundaries

- **PostgreSQL** (`storage/models.py`, sync SQLAlchemy in `storage/db.py`) owns: `documents`, `paper_metadata`, `paper_sections`, `chunks`, `citations`, `ingestion_jobs`, plus books/chapters/drafts (Phases 5–6) and topic clusters (Phase 7). `chunks.qdrant_point_id` is the join key to vectors.
- **Qdrant** (`storage/qdrant.py`) owns only vectors and payload. Never treat Qdrant payload as source of truth for bibliographic data — always join back to Postgres.

### CLI layout

`sciknow/cli/main.py` is the Typer root that composes subapps: `db`, `ingest`, `search`, `ask`, `catalog`, `book`, `draft`. Each subapp module owns a coherent domain — when adding a command, add it to the matching module, don't create a new top-level.

### Phases beyond ingestion/retrieval

- **RAG** (`rag/llm.py`, `rag/prompts.py`) — Ollama streaming wrapper + prompt templates for `sciknow ask {question,synthesize,write}`.
- **Books/drafts** — `book` creates projects with LLM-generated chapter outlines; `book write` drafts chapter sections grounded in retrieval and persists to the drafts table; `book argue` builds SUPPORTS/CONTRADICTS/NEUTRAL argument maps; `book gaps` identifies missing topics.
- **Topic clustering** (`catalog cluster`) — LLM assigns papers to 6–14 named clusters, then retrieval can filter on `--topic`.

## Conventions specific to this repo

- **Never use `sciknow db reset` as a "fix" for a broken ingestion.** It wipes Postgres, Qdrant, `data/processed/`, `data/downloads/`, and `mineru_output/`. Resume is automatic — just re-run the same `ingest` command and check `db stats` first.
- When adding a new canonical section type, edit `_SECTION_PATTERNS`, `_SKIP_SECTIONS` (if non-embedded), and `_PARAMS` in `sciknow/ingestion/chunker.py` — all three are the contract.
- Changing `EMBEDDING_MODEL` / `EMBEDDING_DIM` requires `sciknow db init` to create fresh Qdrant collections with the new dimension; existing chunks will have a mismatched `embedding_model` and need re-embedding.
- All date-bearing fields on `documents` / `paper_metadata` track ingestion lineage — do not backfill them from the filesystem mtime.
