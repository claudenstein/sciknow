# Architecture

[&larr; Back to README](../README.md)

---

## System Overview

```
PDFs (scanned or text)
        │
        ▼
 [1] MinerU 2.5 (pipeline)      PDF → content_list.json (typed blocks)
     ↳ Marker fallback          ↳ used if MinerU fails (rare)
        │
        ▼
 [2] Metadata extraction       PyMuPDF → Crossref API → arXiv API → LLM fallback
        │
        ▼
 [3] Section-aware chunker     Detects Abstract/Methods/Results/etc., chunks per type
        │
        ▼
 [4] BAAI/bge-m3 embedder      Dense (1024-dim) + Sparse vectors per chunk
        │
        ├──→  PostgreSQL        Documents, metadata, sections, chunks, citations
        └──→  Qdrant            Vector index (papers + abstracts collections)

Query
        │
        ├── [5a] Dense vector search     (Qdrant, bge-m3)
        ├── [5b] Sparse vector search    (Qdrant, bge-m3 lexical weights)
        └── [5c] Full-text search        (PostgreSQL tsvector)
                │
                ▼
           RRF fusion (top 50 candidates)
                │
                ▼
        [6] bge-reranker-v2-m3          Cross-encoder reranking (top 10)
                │
                ▼
           Ranked results with citations
```

### Knowledge Wiki Layer (Karpathy Pattern)

```
Papers (raw chunks) → wiki compile → data/wiki/*.md + Qdrant wiki collection
                            ↑ grows with each ingest

wiki query → search wiki collection → LLM answers from pre-synthesized knowledge
book write → searches wiki first → higher-quality, cross-referenced context
```

### Book Writing Pipeline

```
plan → outline → (per chapter) sentence plan → write → review → revise → verify → export
         │                                        │        │        │
         └── LLM proposes chapters from corpus    │        │        └── applies feedback (v N+1)
                                                  │        └── 5-dim scoring + claim verification
                                                  └── RAG-grounded + cross-chapter coherence
```

---

## Services (all native, no Docker)

| Service | Default port | Purpose |
|---|---|---|
| PostgreSQL 16 | 5432 | Relational storage, full-text search |
| Qdrant | 6333 | Vector store (dense + sparse) |
| llama-server (writer)   | 8090  | Writer LLM via OpenAI-compat API (Qwen3.6-27B) |
| llama-server (embedder) | 8091  | bge-m3 dense+sparse via `/v1/embeddings` |
| llama-server (reranker) | 8092  | bge-reranker-v2-m3 cross-encoder via `/v1/rerank` |
| Ollama (optional, v1 rollback or visuals VLM) | 11434 | Only used in v2 for `corpus caption-visuals` (qwen2.5vl); writer/embedder/reranker on v1 fallback paths |

---

## Project Structure

```
sciknow/
├── pyproject.toml
├── alembic.ini
├── .env                        # local config (gitignore this)
├── .env.example                # template
├── scripts/
│   └── setup.sh                # one-shot machine setup
├── migrations/
│   ├── env.py
│   └── versions/
│       └── 0001_initial.py     # full schema + triggers
├── data/
│   ├── inbox/                  # drop PDFs here
│   ├── processed/              # PDFs moved here after successful ingest
│   ├── failed/                 # PDFs moved here on failure
│   ├── wiki/                   # compiled wiki markdown pages
│   │   ├── papers/             # one summary per paper
│   │   ├── concepts/           # concept/method/dataset pages
│   │   └── synthesis/          # high-level overviews
│   └── mineru_output/          # raw converter output per paper
│       └── {uuid}/
│           └── {stem}/
│               └── auto/       # MinerU pipeline backend default subdir
│                   ├── {stem}_content_list.json  # MinerU typed block list (primary)
│                   └── {stem}_middle.json        # MinerU intermediate rep (debug)
└── sciknow/
    ├── config.py               # Pydantic Settings — all config flows through here
    ├── cli/
    │   ├── main.py             # root Typer app
    │   ├── db.py               # sciknow db ...
    │   ├── ingest.py           # sciknow ingest ...
    │   ├── search.py           # sciknow search ...
    │   ├── ask.py              # sciknow ask ...
    │   ├── catalog.py          # sciknow catalog ...
    │   ├── wiki.py             # sciknow wiki ...
    │   ├── book.py             # sciknow book ...
    │   └── draft.py            # sciknow draft ...
    ├── core/
    │   ├── book_ops.py         # generator-based service layer for book operations
    │   └── wiki_ops.py         # wiki compile, query, lint, consensus
    ├── ingestion/
    │   ├── pdf_converter.py    # MinerU + Marker fallback dispatcher
    │   ├── metadata.py         # 4-layer metadata extraction
    │   ├── chunker.py          # section detection + chunking
    │   ├── embedder.py         # bge-m3 → Qdrant upsert
    │   ├── pipeline.py         # orchestrates all stages + PostgreSQL state tracking
    │   └── topic_cluster.py    # BERTopic clustering
    ├── retrieval/
    │   ├── hybrid_search.py    # dense + sparse + FTS → RRF fusion
    │   ├── reranker.py         # bge-reranker-v2-m3 cross-encoder
    │   ├── context_builder.py  # fetch full chunk text, format for display / RAG
    │   ├── self_rag.py         # retrieval evaluation + grounding checks
    │   └── multimodal.py       # table/equation tagging
    ├── rag/
    │   ├── llm.py              # Dispatch facade: routes to llama-server (v2) or Ollama (v1 rollback)
    │   ├── prompts.py          # prompt templates (Q&A, synthesis, write, finetune)
    │   └── wiki_prompts.py     # wiki compilation prompt templates
    ├── storage/
    │   ├── models.py           # SQLAlchemy ORM (all tables)
    │   ├── db.py               # sync engine + session factory
    │   └── qdrant.py           # Qdrant client + collection init
    ├── web/
    │   └── app.py              # FastAPI SPA — web reader + authoring platform
    └── utils/
        ├── doi.py              # DOI / arXiv ID regex extraction
        └── text.py             # text normalization helpers
```

---

## Database Schema

### PostgreSQL Tables

| Table | Description |
|---|---|
| `documents` | One row per PDF. Tracks ingestion status, file hash (SHA-256 for dedup), `ingest_source` (`seed` / `expand`). |
| `paper_metadata` | Bibliographic metadata: title, abstract, authors, DOI, year, journal, keywords, domains, topic_cluster. Full-text search via `tsvector` trigger. |
| `paper_sections` | Sections extracted from the document, classified by type. |
| `chunks` | Individual retrieval units. Links to a section and a Qdrant point UUID. |
| `citations` | Reference list entries (from Crossref, MinerU, OpenAlex). `cited_document_id` cross-links to corpus papers. |
| `ingestion_jobs` | Audit log of every pipeline stage with timing, status, and backend used. |
| `books` | Book projects with title, description, plan (thesis), and status. |
| `book_chapters` | Chapters within a book (number, title, topic_query, topic_cluster, sections JSONB). |
| `drafts` | Written sections with content, sources, summary, version, parent_draft_id, review_feedback, status, custom_metadata. |
| `book_gaps` | Persistent gap tracking (type, description, chapter, status, resolved_draft_id). |
| `wiki_pages` | Wiki page metadata (slug, title, page_type, source_doc_ids, qdrant_point_id). |
| `knowledge_graph_triples` | Entity-relationship triples (subject, predicate, object, source_doc_id). |
| `draft_snapshots` | Named snapshots of draft content for version tracking. |

### Qdrant Collections

| Collection | Vectors | On disk | Purpose |
|---|---|---|---|
| `papers` | dense (1024) + sparse | Yes | All chunk embeddings |
| `abstracts` | dense (1024) + sparse | No | Abstract-level paper search |
| `wiki` | dense (1024) | No | Wiki page embeddings |

Payload indexes in `papers` (enable fast pre-filter): `document_id`, `section_type`, `year`, `domains`, `journal`.

---

## AI Models

### GPU-resident models (loaded once per process, stay in VRAM)

| Model | VRAM | Used by | Purpose |
|---|---|---|---|
| **MinerU 2.5 pipeline** (DocLayout-YOLO, MFD, MFR, table OCR, text OCR) | ~7 GB peak | PDF conversion (primary) | PDF → structured `content_list.json` |
| **BAAI/bge-m3** (FP16, 1024-dim dense + sparse) | ~2.2 GB | Embedding, search, relevance scoring | Chunk & query embeddings, expand relevance filter |
| **BAAI/bge-reranker-v2-m3** (FP16 cross-encoder) | ~0.5 GB | Reranking | Re-scores top-50 candidates to top-k |
| **Marker** (Surya OCR + layout, fallback only) | ~5 GB peak | PDF conversion (fallback) | Only loaded if MinerU fails |

### LLM: `LLM_FAST_MODEL` (default: `qwen3:8b`)

Small, fast model (~1 s/call) for lightweight structured extraction:

| Task | Where |
|---|---|
| Metadata fallback (layer 4) | Extracts title/authors/year from first ~3000 chars when Crossref/arXiv fail |
| Query expansion (`--expand`) | Adds synonyms/acronyms to search queries before embedding |
| Draft summary generation | Auto-generates 100-200 word summaries for cross-chapter coherence |
| Wiki compilation | Paper summaries + entity/KG extraction during `wiki compile` |

### LLM: `LLM_MODEL` (default: `qwen3.5:27b`)

Primary model for all generation, analysis, and reasoning (~5-30 s/call):

| Task | Where |
|---|---|
| RAG Q&A | `sciknow ask question` |
| Multi-paper synthesis | `sciknow ask synthesize` |
| Section drafting | `sciknow ask write`, `sciknow book write` |
| Book outline generation | `sciknow book outline` |
| Book plan (thesis) | `sciknow book plan` |
| Sentence planning | `sciknow book write --plan` |
| Review (critic agent) | `sciknow book review` |
| Revision | `sciknow book revise` |
| Claim verification | `sciknow book write --verify` |
| Argument mapping | `sciknow book argue` |
| Gap analysis | `sciknow book gaps` |
| Topic cluster naming | `sciknow catalog cluster` (single LLM call) |

### Model lifecycle by pipeline phase

```
INGESTION:    MinerU 2.5-Pro VLM (GPU) → writer LLM (metadata cascade L4) → bge-m3 embedder
RETRIEVAL:    bge-m3 embedder → [writer for --expand relevance filter] → bge-reranker
WIKI:         writer LLM (summaries, entities, KG) → bge-m3 (wiki page embedding)
RAG/WRITING:  writer LLM (all generation) — single canonical writer per spec §2.1
EXPANSION:    bge-m3 embedder (relevance filter) → no model (reference extraction)
```

In v2, all LLM/embed/rerank calls dispatch through `sciknow.infer.client` to llama-server (writer :8090, embedder :8091, reranker :8092). `sciknow.rag.llm` is a thin facade so v1 callers keep working when `USE_LLAMACPP_WRITER=False` flips back to the Ollama path. Any command with `--model` can override the default for that invocation (only meaningful on the v1 fallback path; on v2 the loaded GGUF is the writer).

### Recommended Model Configuration

For the best balance of quality and speed on an RTX 3090 (24 GB):

| Model | Architecture | Quant | Speed on 3090 | Role |
|---|---|---|---|---|
| `Qwen3.6-27B-UD-Q4_K_XL.gguf` (Unsloth Dynamic 2.0) | 27B dense, 262K ctx native | UD-Q4_K_XL (~17.6 GB) | 30-40 tok/s decode | Writer (book write/autowrite/review/argue, wiki compile, extract-kg, RAG synthesis, metadata L4) |
| `bge-m3-Q8_0.gguf` (gpustack mirror) | xlm-roberta-large-class, dense + sparse | Q8 (~0.6 GB) | <20 ms / chunk | Embedder (ingestion + query-side) |
| `bge-reranker-v2-m3-Q8_0.gguf` (gpustack mirror) | xlm-roberta-large cross-encoder | Q8 (~0.6 GB) | <50 ms / pair | Reranker (top-50 RRF → top-k) |

---

## Storage Boundaries

- **PostgreSQL** (`storage/models.py`, sync SQLAlchemy in `storage/db.py`) owns all relational data. `chunks.qdrant_point_id` is the join key to vectors.
- **Qdrant** (`storage/qdrant.py`) owns only vectors and payload. Never treat Qdrant payload as source of truth for bibliographic data — always join back to Postgres.
- **Filesystem** (`data/wiki/*.md`) owns wiki page content. PostgreSQL tracks metadata + Qdrant pointer. Markdown files are human-readable and git-friendly.

---

## Service Layer Pattern

All book and wiki operations use generator-based functions that yield typed event dicts (`token`, `progress`, `scores`, `verification`, `completed`, `error`). These are consumed by both the CLI (Rich console rendering) and the web layer (SSE endpoints via `asyncio.Queue`). When adding a new operation, implement it as a generator in `core/`, then wire it into both CLI and web.

---

## Python Dependencies

Managed by `uv` via `pyproject.toml`. Key packages:

| Package | Purpose |
|---|---|
| `mineru[vllm]` | PDF→JSON conversion via MinerU 2.5-Pro VLM (primary). `mineru[core]` (pipeline mode) and `marker-pdf` are layered fallbacks. |
| `marker-pdf` | PDF→JSON/Markdown conversion (Marker, last-resort fallback) |
| `httpx` | OpenAI-compat HTTP client to llama-server (writer/embedder/reranker) + Crossref + arXiv |
| `qdrant-client` | Qdrant vector store client |
| `sqlalchemy` + `psycopg2-binary` | PostgreSQL ORM + driver |
| `alembic` | Database migrations |
| `pymupdf` | Embedded PDF metadata extraction |
| `arxiv` | arXiv API client |
| `tiktoken` | Token counting for chunking |
| `typer` + `rich` | CLI framework + terminal UI |
| `pydantic` + `pydantic-settings` | Data validation and config management |
| `fastapi` + `uvicorn` | Web reader / authoring platform |
| `bertopic` + `umap-learn` + `hdbscan` | Topic clustering |

**Rollback-only extras** (not pulled by default, opt-in via `uv add` when `USE_LLAMACPP_*=False`):

| Package | Activated by |
|---|---|
| `ollama` | `USE_LLAMACPP_WRITER=False` (v1 writer path); also used by `corpus caption-visuals` for the qwen2.5vl visuals model in v2 by design |
| `FlagEmbedding` | `USE_LLAMACPP_EMBEDDER=False` (v1 in-process bge-m3) |
| `sentence-transformers` | `USE_LLAMACPP_RERANKER=False` (v1 in-process cross-encoder) |
