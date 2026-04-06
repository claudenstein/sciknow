# SciKnow

A local-first, large-scale scientific knowledge system. Ingests scientific papers (PDFs — both scanned and text-based), converts them to structured markdown, extracts metadata, chunks them section-aware, embeds them with a dense+sparse model, and stores everything in a PostgreSQL + Qdrant stack for fast hybrid retrieval.

All AI inference runs locally. No cloud APIs required to operate the system.

---

## Features

- **Ingestion pipeline** — PDF → structured JSON (MinerU 2.5) → metadata extraction → section-aware chunking → dense+sparse embedding → PostgreSQL + Qdrant
- **Hybrid search** — dense vector + sparse lexical + full-text search, fused with Reciprocal Rank Fusion and reranked with a cross-encoder. Citation-count boosted scoring. Optional LLM query expansion.
- **RAG question answering** — grounded Q&A over your papers with inline citations and source attribution
- **Writing assistant** — multi-paper synthesis, section drafting, sentence planning, claim verification, IPCC uncertainty language
- **Book projects** — structured book → chapter hierarchy with LLM-generated outlines, book plans (thesis + scope), cross-chapter coherence, iterative write → review → revise workflow
- **Argument mapping** — evidence classification (SUPPORTS / CONTRADICTS / NEUTRAL) for any claim
- **Gap analysis** — identifies missing topics, weak chapters, and unwritten sections; persists gaps for tracking
- **Topic clustering** — LLM assigns papers to named thematic clusters for filtered search and writing
- **Citation graph** — extracts + cross-links references during ingestion; citation-count boosted retrieval
- **Collection expansion** — follows citations to discover + download open-access papers with semantic relevance filtering (6 OA sources: Copernicus, arXiv, Unpaywall, OpenAlex, Europe PMC, Semantic Scholar)
- **Multi-format export** — Markdown, BibTeX, LaTeX, and DOCX via Pandoc
- **Similar papers** — find papers with similar abstracts via embedding similarity
- **Backup & restore** — portable archives for migrating between machines

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3090 (24 GB VRAM) | RTX 3090 + remote GPU server |
| RAM | 32 GB | 64 GB |
| Storage | 500 GB SSD | 2 TB NVMe |
| OS | Ubuntu 22.04+ | Ubuntu 22.04+ |
| CUDA | 12.x | 12.x |

**VRAM budget on 3090 (24 GB):**
- bge-m3 embedding model: ~2.2 GB
- qwen2.5:32b-q4_K_M (LLM): ~18–19 GB
- bge-reranker-v2-m3: ~0.5 GB
- Total: fits comfortably within 24 GB

**Remote GPU server:** Set `OLLAMA_HOST=http://your-gpu-server:11434` in `.env`. Zero code changes needed. At that point you can upgrade to `qwen2.5:72b` or `llama3.1:405b-q4`.

---

## Architecture

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

### Services (all native, no Docker)

| Service | Default port | Purpose |
|---|---|---|
| PostgreSQL 16 | 5432 | Relational storage, full-text search |
| Qdrant | 6333 | Vector store (dense + sparse) |
| Ollama | 11434 | Local LLM inference |

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
    │   ├── book.py             # sciknow book ...
    │   └── draft.py            # sciknow draft ...
    ├── ingestion/
    │   ├── pdf_converter.py    # MinerU + Marker fallback dispatcher
    │   ├── metadata.py         # 4-layer metadata extraction
    │   ├── chunker.py          # section detection + chunking
    │   ├── embedder.py         # bge-m3 → Qdrant upsert
    │   └── pipeline.py         # orchestrates all stages + PostgreSQL state tracking
    ├── retrieval/
    │   ├── hybrid_search.py    # dense + sparse + FTS → RRF fusion
    │   ├── reranker.py         # bge-reranker-v2-m3 cross-encoder
    │   └── context_builder.py  # fetch full chunk text, format for display / RAG
    ├── rag/
    │   ├── llm.py              # Ollama streaming/completion wrapper
    │   └── prompts.py          # prompt templates (Q&A, synthesis, write, finetune)
    ├── storage/
    │   ├── models.py           # SQLAlchemy ORM (all tables)
    │   ├── db.py               # sync engine + session factory
    │   └── qdrant.py           # Qdrant client + collection init
    └── utils/
        ├── doi.py              # DOI / arXiv ID regex extraction
        └── text.py             # text normalization helpers
```

---

## Installation

### 1. Run the setup script

The setup script installs and configures all dependencies on Ubuntu/Debian with CUDA 12.x:

```bash
cd /path/to/sciknow
bash scripts/setup.sh
```

What it does:
- Installs PostgreSQL 16 (via apt), creates `sciknow` user and database
- Downloads Qdrant binary to `~/.local/qdrant/` and registers it as a systemd user service
- Installs Ollama and pulls the fast metadata model (`mistral:7b-instruct-q4_K_M`)
- Installs `uv` and runs `uv sync` to set up the Python environment
- Installs MinerU 2.5 (`mineru[core]`) as the primary PDF backend and Marker (`marker-pdf`) as fallback; both download their models lazily on first use (MinerU → `~/.cache/modelscope`, Marker → `~/.cache/datalab`)
- Copies `.env.example` → `.env`

### 2. Pull the main LLM

After setup, pull the primary model when you have the full VRAM budget available:

```bash
ollama pull qwen2.5:32b-instruct-q4_K_M
```

### 3. Configure

Edit `.env` — the only required change is your email for the Crossref polite pool:

```bash
nano .env
# Set: CROSSREF_EMAIL=you@youremail.com
```

### 4. Initialize the schema

```bash
sciknow db init
```

This runs Alembic migrations (creates all tables, indexes, and the full-text search trigger) and initializes the Qdrant collections.

### 5. Verify

```bash
sciknow db stats
```

Should show zeros with green checkmarks for PostgreSQL and Qdrant.

---

## Manual Installation (without setup.sh)

If you prefer to install components manually:

**PostgreSQL:**
```bash
sudo apt install postgresql-16 postgresql-client-16
sudo systemctl enable --now postgresql
sudo -u postgres psql -c "CREATE USER sciknow WITH PASSWORD 'sciknow';"
sudo -u postgres psql -c "CREATE DATABASE sciknow OWNER sciknow;"
```

**Qdrant:**
```bash
# Download latest binary
wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant*.tar.gz
./qdrant   # runs on port 6333 by default
```

**Ollama:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:7b-instruct-q4_K_M
ollama pull qwen2.5:32b-instruct-q4_K_M
```

**Ollama performance tuning (recommended):**

Two environment variables that significantly improve LLM inference speed at no quality cost:

```bash
# Add to ~/.bashrc to make permanent
export OLLAMA_FLASH_ATTENTION=1     # 25-47% speed boost, cuts attention VRAM ~8GB → ~1.5GB at 16K context
export OLLAMA_KV_CACHE_TYPE=q8_0    # Halves KV cache VRAM with <5% speed hit and negligible quality loss
```

Then start Ollama:
```bash
ollama serve &
```

| Setting | What it does | Impact on RTX 3090 |
|---|---|---|
| `OLLAMA_FLASH_ATTENTION=1` | Uses FlashAttention-2 for the attention layer | **+47% tok/s** at 16K context; critical for long-context calls (clustering, book writing) |
| `OLLAMA_KV_CACHE_TYPE=q8_0` | Quantizes the KV cache from FP16 to INT8 | Frees ~4 GB VRAM at 32K context; faster long-context generation |

Real measured improvement on qwen2.5:32b-instruct-q4_K_M with an RTX 3090:

| Configuration | Speed |
|---|---|
| Default (no FA, FP16 KV) | ~5 tok/s |
| **FA + q8_0 KV** | **~7.7 tok/s (+54%)** |

These settings apply to all sciknow LLM operations: `ask`, `book write`, `book review`, `catalog cluster`, `db export --generate-qa`, etc.

**Python environment:**
```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

**PDF converters:**
```bash
# Primary: MinerU 2.5 (OpenDataLab, OmniDocBench SOTA for scientific papers)
uv add "mineru[core]"

# Fallback: Marker (kept for robustness)
uv pip install marker-pdf
```

MinerU downloads its pipeline models (layout, MFD, MFR, table OCR, text OCR) on first use to `~/.cache/modelscope` (~2 GB). Marker downloads Surya OCR + layout models to `~/.cache/datalab` on first use. Both use CUDA automatically when available. You can select the backend with `PDF_CONVERTER_BACKEND` in `.env`:

- `auto` (default) — MinerU → Marker JSON → Marker markdown fallback chain
- `mineru` — MinerU only, fails hard on error
- `marker` — legacy Marker-only path (Marker JSON → markdown)

---

## Configuration Reference

All settings are read from `.env` (or environment variables). Managed by Pydantic Settings in `sciknow/config.py`.

| Variable | Default | Description |
|---|---|---|
| `PG_HOST` | `localhost` | PostgreSQL host |
| `PG_PORT` | `5432` | PostgreSQL port |
| `PG_USER` | `sciknow` | PostgreSQL user |
| `PG_PASSWORD` | `sciknow` | PostgreSQL password |
| `PG_DATABASE` | `sciknow` | PostgreSQL database name |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model (HuggingFace ID) |
| `EMBEDDING_DIM` | `1024` | Embedding vector dimension |
| `LLM_MODEL` | `qwen2.5:32b-instruct-q4_K_M` | Main LLM (Ollama model name) |
| `LLM_FAST_MODEL` | `mistral:7b-instruct-q4_K_M` | Fast LLM for metadata extraction fallback |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker (search + ask + book write) |
| `CROSSREF_EMAIL` | `user@example.com` | **Set this.** Used in Crossref/OpenAlex polite pool User-Agent |
| `PDF_CONVERTER_BACKEND` | `auto` | `auto` / `mineru` / `marker` — see above |
| `EMBEDDING_BATCH_SIZE` | `32` | Chunks per bge-m3 batch (16 if LLM co-resident, 64 for embedder-only runs) |
| `MARKER_BATCH_MULTIPLIER` | `2` | Marker/Surya internal batch size multiplier (Marker fallback path) |
| `PG_POOL_SIZE` / `PG_MAX_OVERFLOW` | `20` / `20` | SQLAlchemy connection pool for parallel workers |
| `QDRANT_HNSW_M` | `32` | HNSW graph connectivity (applied on collection creation) |
| `QDRANT_HNSW_EF_CONSTRUCT` | `256` | HNSW build-time exploration (applied on collection creation) |
| `QDRANT_HNSW_EF` | `128` | HNSW query-time exploration (applied per query) |
| `QDRANT_SCALAR_QUANTIZATION` | `true` | int8 dense-vector quantization (~75% memory savings, negligible recall hit) |
| `INGEST_WORKERS` | `1` | Parallel worker subprocesses for `ingest directory` (raise to 2 only when LLM is off-GPU) |
| `ENRICH_WORKERS` | `8` | Concurrent Crossref/OpenAlex lookups in `db enrich` |
| `EXPAND_DOWNLOAD_WORKERS` | `6` | Concurrent OA PDF lookups in `db expand` |
| `LLM_PARALLEL_WORKERS` | `4` | Concurrent LLM calls in bulk commands; **must be ≤ `OLLAMA_NUM_PARALLEL` on the Ollama server** (server default is 1) |
| `EXPAND_RELEVANCE_THRESHOLD` | `0.55` | Cosine similarity cut-off for the `db expand` relevance filter |

**Remote GPU server:** Change only `OLLAMA_HOST=http://your-gpu-server:11434`. Everything else stays the same.

**Server-side Ollama parallelism:** the default `OLLAMA_NUM_PARALLEL=1` serialises every LLM request regardless of client-side concurrency. To unlock `LLM_PARALLEL_WORKERS`, set it on the Ollama host:
```bash
systemctl --user edit ollama
# add:  Environment="OLLAMA_NUM_PARALLEL=4"
systemctl --user restart ollama
```

---

## CLI Reference

### `sciknow db`

```bash
sciknow db init              # Run migrations + init Qdrant collections
sciknow db reset             # Wipe everything and re-initialise (use before a full re-ingest)
sciknow db stats             # Show paper/chunk counts and status breakdown
sciknow db refresh-metadata  # Re-run metadata extraction for papers with poor-quality metadata
sciknow db enrich            # Find DOIs for papers that don't have one (Crossref + OpenAlex search by title)
sciknow db expand            # Discover and download cited papers to grow the collection
sciknow db backup            # Back up the full collection to a portable archive
sciknow db restore           # Restore a backup on a new machine
```

### `sciknow ingest`

```bash
# Ingest a single PDF
sciknow ingest file paper.pdf

# Ingest all PDFs in a directory (recursive by default)
sciknow ingest directory ./papers/

# Non-recursive
sciknow ingest directory ./papers/ --no-recursive

# Parallel ingestion — spawns N worker subprocesses, each with its own
# Marker/MinerU + bge-m3 models. On a 24 GB 3090 with an LLM resident,
# keep --workers 1. Raise to 2 only when the LLM is off-GPU.
sciknow ingest directory ./papers/ --workers 2

# Re-ingest a paper that was already processed
sciknow ingest file paper.pdf --force
```

### `sciknow search`

```bash
# Basic query — hybrid search + reranking, returns top 10
sciknow search query "sea surface temperature reconstruction methods"

# Show full chunk text for each result
sciknow search query "DNA methylation cancer" --show-content

# Filter by year range and section type
sciknow search query "aerosol radiative forcing" --year-from 2010 --year-to 2023 --section methods

# Filter by domain
sciknow search query "stellar evolution" --domain astrophysics

# Filter by topic cluster (set topic clusters with `sciknow catalog cluster` first)
sciknow search query "solar forcing" --topic "Solar Irradiance"

# Skip reranking (faster, uses RRF scores directly)
sciknow search query "climate sensitivity" --no-rerank --top-k 20

# Show relevance scores
sciknow search query "protein folding" --show-scores

# Control candidate pool size (more candidates = more thorough, slower reranking)
sciknow search query "galaxy formation" --candidates 100 --top-k 15
```

Valid `--section` values: `abstract`, `introduction`, `methods`, `results`, `discussion`, `conclusion`, `related_work`, `appendix`

### `sciknow ask`

```bash
# Answer a question using RAG (retrieves top 8 chunks, reranks, streams answer)
sciknow ask question "What are the main mechanisms of aerosol radiative forcing?"

# Use a specific model (default: LLM_MODEL from .env)
sciknow ask question "Explain stellar nucleosynthesis" --model mistral:7b-instruct-q4_K_M

# Filter by year and section
sciknow ask question "How is SST reconstructed from proxies?" \
    --year-from 2000 --section methods

# More context chunks (default: 8)
sciknow ask question "Compare tree-ring proxy methods" --context-k 15

# Hide sources
sciknow ask question "What drives the Atlantic Meridional Overturning Circulation?" --no-sources

# Synthesise findings across papers on a topic
sciknow ask synthesize "solar activity and climate variability"

# Synthesis with domain filter and more context
sciknow ask synthesize "galaxy formation and feedback mechanisms" \
    --domain astrophysics --context-k 15

# Draft a paper section grounded in your library
sciknow ask write "aerosol-cloud interactions" --section introduction
sciknow ask write "stellar population synthesis" --section methods --domain astrophysics

# Save the draft to the database (optionally link to a book chapter)
sciknow ask write "solar forcing mechanisms" --section introduction --save
sciknow ask write "ocean heat content trends" --section results --save \
    --book "Global Cooling" --chapter 3

# Filter retrieval to a topic cluster
sciknow ask question "How does solar activity affect climate?" --topic "Solar Irradiance"
sciknow ask synthesize "ocean-atmosphere heat exchange" --topic "Ocean Heat Transport"
```

### `sciknow catalog`

```bash
# Overview: papers by year (bar chart), top journals, metadata source quality
sciknow catalog stats

# Paginated table of all papers with optional filters
sciknow catalog list
sciknow catalog list --author Zharkova --sort year
sciknow catalog list --from 2015 --to 2023 --journal "Nature"
sciknow catalog list --title "solar cycle" --limit 20 --page 2

# Full record for one paper — by DOI, arXiv ID, or title fragment
sciknow catalog show 10.1093/mnras/stad1001
sciknow catalog show "solar magnetic field eigenvectors"

# Export the catalog to CSV or JSON
sciknow catalog export --output catalog.csv
sciknow catalog export --format json --output catalog.json
sciknow catalog export --author Zharkova --output zharkova.csv

# Assign topic clusters to all papers (LLM groups papers into 6–14 named clusters)
sciknow catalog cluster
sciknow catalog cluster --dry-run           # Preview clusters without saving
sciknow catalog cluster --batch 100         # Smaller batches for large collections

# Resume after partial failure (only processes papers without a cluster yet)
sciknow catalog cluster --resume

# List all clusters with paper counts
sciknow catalog topics
```

### `sciknow book`

The book system organises writing projects with an iterative, coherent pipeline:

```
plan → outline → (per chapter) sentence plan → write → review → revise → verify → export
```

Each step is grounded in retrieved papers and maintains cross-chapter coherence via a persistent book plan and auto-generated chapter summaries.

```bash
# ── Project setup ─────────────────────────────────────────────────────────

# Create a new book project
sciknow book create "Global Cooling"
sciknow book create "Solar Cycle Mechanisms" --description "Overview of solar variability"

sciknow book list                       # List all books
sciknow book show "Global Cooling"      # Chapters, drafts, gaps, progress

# Add chapters manually or auto-generate with LLM
sciknow book chapter add "Global Cooling" "The Maunder Minimum"
sciknow book outline "Global Cooling"   # LLM proposes 6–12 chapters from your library

# ── Book plan (thesis + scope) ────────────────────────────────────────────

# Generate the book plan — defines central argument, scope, audience, key terms.
# Injected into every `book write` call for cross-chapter consistency.
sciknow book plan "Global Cooling"
sciknow book plan "Global Cooling" --edit   # Regenerate

# ── Writing ───────────────────────────────────────────────────────────────

# Draft a chapter section (with cross-chapter coherence: injects book plan +
# summaries of prior chapters so the LLM maintains consistency)
sciknow book write "Global Cooling" 2 --section methods

# Show a sentence plan before drafting (paragraph-by-paragraph skeleton)
sciknow book write "Global Cooling" 3 --section results --plan

# Run claim verification after drafting (checks each [N] citation)
sciknow book write "Global Cooling" 1 --section introduction --verify

# Add IPCC-style calibrated uncertainty language (very likely, high confidence, etc.)
sciknow book write "Global Cooling" 4 --section discussion --ipcc

# Combine all flags for maximum quality
sciknow book write "Global Cooling" 5 --section conclusion --plan --verify --ipcc --expand

# ── Review + revise loop ──────────────────────────────────────────────────

# Run a critic pass over a saved draft (groundedness, completeness, accuracy,
# coherence, redundancy). Feedback saved to the draft.
sciknow book review 3f2a1b4c

# Revise based on instructions (creates version N+1, preserves original)
sciknow book revise 3f2a1b4c -i "expand the section on solar cycles with more evidence"
sciknow book revise 3f2a1b4c -i "add counterarguments from the skeptic literature"
sciknow book revise 3f2a1b4c         # applies saved review feedback automatically

# ── Evidence analysis ─────────────────────────────────────────────────────

# Map evidence for/against a claim (SUPPORTS / CONTRADICTS / NEUTRAL)
sciknow book argue "solar activity is the primary driver of 20th century warming"
sciknow book argue "cosmic rays modulate cloud cover" --save

# ── Gap analysis ──────────────────────────────────────────────────────────

# Identify + persist gaps (topic, evidence, argument, draft gaps → saved to DB)
sciknow book gaps "Global Cooling"
sciknow book gaps "Global Cooling" --no-save   # informational only

# ── Autowrite (autonomous convergence loop) ───────────────────────────────

# Autowrite one section: write → score → revise → re-score → keep/discard → repeat
sciknow book autowrite "Global Cooling" 1 --section introduction

# More iterations and higher quality target
sciknow book autowrite "Global Cooling" 3 --section methods --max-iter 5 --target-score 0.90

# All sections of a chapter
sciknow book autowrite "Global Cooling" 3 --section all --ipcc

# FULL BOOK: all chapters × all sections (the ultimate autonomous pipeline)
sciknow book autowrite "Global Cooling" --full --max-iter 3 --target-score 0.85

# With auto-expand: fetches new papers when reviewer identifies evidence gaps
sciknow book autowrite "Global Cooling" --full --auto-expand --ipcc

# ── Web reader ────────────────────────────────────────────────────────────

# Launch a local web reader in your browser — sidebar navigation, inline
# editing, comments, citation links, dark/light theme. Live from the database.
sciknow book serve "Global Cooling"
sciknow book serve "Global Cooling" --port 9000

# ── Export ────────────────────────────────────────────────────────────────

# Markdown (default)
sciknow book export "Global Cooling"
sciknow book export "Global Cooling" -o manuscript.md

# Self-contained HTML (open in any browser, shareable without sciknow)
sciknow book export "Global Cooling" --format html -o book.html

# BibTeX bibliography from all cited papers
sciknow book export "Global Cooling" --format bibtex -o refs.bib

# LaTeX via Pandoc (requires pandoc installed)
sciknow book export "Global Cooling" --format latex -o book.tex

# DOCX via Pandoc
sciknow book export "Global Cooling" --format docx -o book.docx
```

**Writing workflow in practice:**
1. `book plan` — set the thesis. All subsequent chapters will reference it.
2. `book outline` — LLM proposes chapters from your corpus.
3. For each chapter: `book write <book> <ch> --section X --plan --verify`
4. Review: `book review <draft_id>`
5. Revise: `book revise <draft_id>` (uses saved review feedback) or `book revise <draft_id> -i "your instruction"`
6. Repeat steps 3-5 until satisfied.
7. `book gaps` — identify what's still missing.
8. `book export --format latex` — compile for publication.

### `sciknow draft`

```bash
# List all saved drafts
sciknow draft list
sciknow draft list --book "Global Cooling"
sciknow draft list --page 2

# Print full draft content (use first 8 chars of ID from draft list)
sciknow draft show 3f2a1b4c

# Delete a draft
sciknow draft delete 3f2a1b4c
sciknow draft delete 3f2a1b4c --yes   # Skip confirmation

# Export a draft to Markdown
sciknow draft export 3f2a1b4c
sciknow draft export 3f2a1b4c --output chapter2_intro.md
```

### `sciknow db export`

```bash
# Export all chunks with metadata as JSONL (raw knowledge base dump)
sciknow db export --output dataset.jsonl

# Export only chunks with >= 100 tokens
sciknow db export --output dataset.jsonl --min-tokens 100

# Generate synthetic Q&A pairs using Ollama (~5-10 s per chunk, slow)
sciknow db export --output qa_dataset.jsonl --generate-qa

# Limit export size (useful for testing)
sciknow db export --output sample.jsonl --generate-qa --limit 50
```

Output format:
```json
{"title": "...", "year": 2021, "section": "methods", "doi": "...", "content": "..."}
```
With `--generate-qa`:
```json
{"title": "...", "year": 2021, "section": "methods", "doi": "...", "content": "...",
 "question": "...", "answer": "..."}
```

The ingestion pipeline stages (visible in `db stats` during a run):

```
pending → converting → metadata_extraction → chunking → embedding → complete
                                                                   → failed
```

Failed PDFs are copied to `data/failed/`. Successfully processed PDFs are copied to `data/processed/`. Original files are never deleted.

---

## Ingestion Pipeline Details

### Stage 1 — PDF Conversion (MinerU → Marker fallback)

`sciknow/ingestion/pdf_converter.py` dispatches based on `PDF_CONVERTER_BACKEND`:

1. **MinerU 2.5** (primary) — OpenDataLab's pipeline backend. Runs a cascade of specialised models: DocLayout-YOLO for layout, MFD (Math Formula Detection) + MFR (Math Formula Recognition) for LaTeX extraction, table OCR + structure reconstruction (HTML tables), text OCR, seal detection. Scores 86.2 on [OmniDocBench v1.5](https://arxiv.org/abs/2412.07626) — current SOTA among open-source pipeline tools on scientific papers. Runs on any GPU with ≥8 GB VRAM (Volta or newer). Models cached to `~/.cache/modelscope` on first use (~2 GB).

2. **Marker JSON** (fallback) — `marker-pdf`'s `JSONRenderer` produces a structured block tree (`SectionHeader`, `Text`, `Table`, `Equation`, `ListItem`, ...). Used automatically when MinerU fails, or when `PDF_CONVERTER_BACKEND=marker`.

3. **Marker markdown** (last resort) — if Marker's JSON path also fails, the markdown renderer runs as a final fallback.

**MinerU output format:** `content_list.json` — a flat list of typed blocks:
- `text` with `text_level` (0 = body, 1 = title, 2 = section heading, 3+ = subheading)
- `table` with HTML `table_body` + caption arrays
- `equation` with `text` containing LaTeX and `text_format: "latex"`
- `image` / `chart` / `seal` with paths + captions
- `code` with `code_body` and `sub_type` (code vs algorithm)
- `list` with `list_items` array
- Auxiliary blocks (`header`, `footer`, `page_number`, `page_footnote`, `aside_text`) — dropped by the chunker

Output lands in `data/mineru_output/{doc_id}/{stem}/auto/` (or `pipeline/` depending on MinerU version). Models for both backends load once per Python process and stay resident in VRAM — batching ingestion with `--workers N` amortises the load across many PDFs.

Why MinerU as primary: Marker has a known severe performance regression on RTX 3090 ([datalab-to/marker#919](https://github.com/datalab-to/marker/issues/919) — ~0.03 pages/s with 18–19 GB of VRAM sitting idle). MinerU 2.5 on the same GPU runs at ~0.4 pages/s single-stream. See `OPTIMIZATION.md` for the full rationale and benchmarks.

### Stage 2 — Metadata Extraction (4 layers)

Each layer is tried in order; later layers only run if the previous didn't fully populate the metadata:

1. **PyMuPDF** — reads embedded XMP/Info fields from the PDF. Fast but often incomplete.
2. **Crossref API** — authoritative bibliographic data by DOI. DOI is extracted from the paper text via regex. Uses the Crossref polite pool (rate limit: 50 req/s with email in User-Agent).
3. **arXiv API** — for preprints. arXiv IDs are extracted from URLs and filenames.
4. **Ollama LLM fallback** — sends the first ~3000 characters of the document text to `LLM_FAST_MODEL` with a structured JSON extraction prompt. Used when no DOI or arXiv ID is found.

Fields extracted: title, abstract, year, DOI, arXiv ID, journal, volume, issue, pages, publisher, authors (with ORCID and affiliation where available), keywords.

### Stage 3 — Section-Aware Chunking

The chunker has three parallel parsers, one per PDF backend, all producing the same `Section` list downstream:

**MinerU mode (primary):** `parse_sections_from_mineru(content_list)` walks the flat typed-block list. Text items with `text_level == 1 or 2` open a new top-level section; `text_level >= 3` become inline bold subheadings; `text_level == 0` (body) is accumulated. Tables are rendered to pipe-delimited plain text (with caption prepended). Equations contribute their LaTeX string. Code blocks contribute `code_body`. Lists are joined with newlines. Images/charts/seals and page-level blocks (headers, footers, page numbers) are dropped. Short body-text blocks matching a known section keyword (`Abstract`, `Introduction`, ...) are promoted to implicit section headers — handles older PDFs where MinerU's layout model didn't emit a `text_level`.

**Marker JSON mode (fallback):** `parse_sections_from_json(json_data)` walks Marker's nested block tree. `SectionHeader` blocks at heading level h1/h2 open new sections; h3/h4 are inlined as bold subheadings. `Text`, `Table`, `Equation`, `ListItem` blocks are accumulated. `PageHeader`, `PageFooter`, `Figure`, `Picture` are discarded. Same implicit-heading heuristic as the MinerU path.

**Markdown mode (last resort):** `parse_sections(markdown)` detects sections from markdown headings (`#` through `####`) using regex. Only used if both MinerU and Marker JSON fail.

Both modes classify sections into canonical types:

| Canonical type | Detected headings |
|---|---|
| `abstract` | Abstract |
| `introduction` | Introduction, Background, Motivation, Overview |
| `methods` | Methods, Materials, Experimental Setup, Data Collection, Simulation, ... |
| `results` | Results, Findings, Evaluation, Measurements, ... |
| `discussion` | Discussion, Analysis, Interpretation, Implications |
| `conclusion` | Conclusion, Summary, Outlook, Future Work, ... |
| `related_work` | Related Work, Prior Work, Literature Review, ... |
| `references` | References, Bibliography *(not embedded — stored in citations table)* |
| `acknowledgments` | Acknowledgments, Funding *(not embedded)* |
| `appendix` | Appendix, Supplementary |

Chunking parameters per section type:

| Section | Target tokens | Overlap | Keep whole if under |
|---|---|---|---|
| abstract | 512 | 0 | 512 |
| introduction | 512 | 64 | 768 |
| methods | 512 | 128 | 768 |
| results | 512 | 128 | 768 |
| discussion | 512 | 64 | 768 |
| conclusion | 512 | 0 | 1024 |

Each chunk is prefixed with a context header:
```
[methods] Paper Title Here (2023)

<actual chunk content...>
```
This ensures the embedding captures paper identity even when a chunk is retrieved in isolation.

### Stage 4 — Embedding (bge-m3)

`BAAI/bge-m3` produces two vector types per chunk simultaneously:
- **Dense** (1024-dim cosine): semantic similarity search
- **Sparse** (learned lexical weights): keyword/term matching

Both are stored in Qdrant. Abstracts are also embedded separately in the `abstracts` collection for paper-level search.

---

## AI Models

Sciknow uses four types of AI models, each serving a different role:

### GPU-resident models (loaded once per process, stay in VRAM)

| Model | VRAM | Used by | Purpose |
|---|---|---|---|
| **MinerU 2.5 pipeline** (DocLayout-YOLO, MFD, MFR, table OCR, text OCR) | ~7 GB peak | PDF conversion (primary) | PDF → structured `content_list.json` |
| **BAAI/bge-m3** (FP16, 1024-dim dense + sparse) | ~2.2 GB | Embedding, search, relevance scoring | Chunk & query embeddings, expand relevance filter |
| **BAAI/bge-reranker-v2-m3** (FP16 cross-encoder) | ~0.5 GB | Reranking | Re-scores top-50 candidates to top-k |
| **Marker** (Surya OCR + layout, fallback only) | ~5 GB peak | PDF conversion (fallback) | Only loaded if MinerU fails |

### LLM: `LLM_FAST_MODEL` (default: `mistral:7b-instruct-q4_K_M`)

Small, fast model (~1 s/call) for lightweight structured extraction. Used where speed matters more than reasoning depth:

| Task | Where |
|---|---|
| Metadata fallback (layer 4) | Extracts title/authors/year from first ~3000 chars when Crossref/arXiv fail |
| Query expansion (`--expand`) | Adds synonyms/acronyms to search queries before embedding |
| Draft summary generation | Auto-generates 100-200 word summaries for cross-chapter coherence |

### LLM: `LLM_MODEL` (default: `qwen2.5:32b-instruct-q4_K_M`)

Primary model for all generation, analysis, and reasoning (~5-30 s/call). Used wherever output quality matters:

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
| IPCC uncertainty language | `sciknow book write --ipcc` |
| Argument mapping | `sciknow book argue` |
| Gap analysis | `sciknow book gaps` |
| Topic clustering | `sciknow catalog cluster` |
| QA pair generation | `sciknow db export --generate-qa` |

### Model lifecycle by pipeline phase

```
INGESTION:    MinerU (GPU) → LLM_FAST_MODEL (metadata fallback) → bge-m3 (GPU)
RETRIEVAL:    bge-m3 (GPU) → [LLM_FAST_MODEL for --expand] → bge-reranker (GPU)
RAG/WRITING:  LLM_MODEL (all generation) + LLM_FAST_MODEL (draft summaries)
EXPANSION:    bge-m3 (GPU, relevance filter) → no model (reference extraction)
```

All LLM calls go through Ollama at `OLLAMA_HOST`. Any command with `--model` can override the default for that invocation. When DGX Spark arrives, the plan is to split: fast model stays on the 3090, heavy model moves to Spark (see `OPTIMIZATION.md`).

---

## Database Schema

### Tables

| Table | Description |
|---|---|
| `documents` | One row per PDF. Tracks ingestion status, file hash (SHA-256 — used for deduplication), and `ingest_source` (`seed` / `expand`). |
| `paper_metadata` | Bibliographic metadata: title, abstract, authors, DOI, year, journal, keywords, domains, topic_cluster. Full-text search via `tsvector` trigger. |
| `paper_sections` | Sections extracted from the document, classified by type. |
| `chunks` | Individual retrieval units. Links to a section and a Qdrant point UUID. |
| `citations` | Reference list entries (from Crossref, MinerU content_list, OpenAlex). `cited_document_id` cross-links to corpus papers. |
| `ingestion_jobs` | Audit log of every pipeline stage with timing, status, and backend used. |
| `books` | Book projects with title, description, plan (thesis), and status. |
| `book_chapters` | Chapters within a book (number, title, topic_query, topic_cluster). |
| `drafts` | Written sections with content, sources, summary, version, parent_draft_id, review_feedback. |
| `book_gaps` | Persistent gap tracking (type, description, chapter, status, resolved_draft_id). |

### Qdrant Collections

| Collection | Vectors | On disk | Purpose |
|---|---|---|---|
| `papers` | dense (1024) + sparse | Yes | All chunk embeddings |
| `abstracts` | dense (1024) + sparse | No | Abstract-level paper search |

Payload indexes in `papers` (enable fast pre-filter): `document_id`, `section_type`, `year`, `domains`, `journal`.

---

## Python Dependencies

Managed by `uv` via `pyproject.toml`. Key packages:

| Package | Purpose |
|---|---|
| `mineru[core]` | PDF→JSON conversion (MinerU 2.5 pipeline backend, primary) |
| `marker-pdf` | PDF→JSON/Markdown conversion (Marker, fallback) |
| `FlagEmbedding` | BAAI/bge-m3 embedder and bge-reranker-v2-m3 cross-encoder |
| `qdrant-client` | Qdrant vector store client |
| `sqlalchemy` + `psycopg2-binary` | PostgreSQL ORM + driver |
| `alembic` | Database migrations |
| `httpx` | Crossref + other async HTTP calls |
| `pymupdf` | Embedded PDF metadata extraction |
| `arxiv` | arXiv API client |
| `ollama` | Ollama Python client |
| `tiktoken` | Token counting for chunking |
| `typer` + `rich` | CLI framework + terminal UI |
| `pydantic` + `pydantic-settings` | Data validation and config management |

---

## Development Notes

### Adding a new section type

Edit `_SECTION_PATTERNS` in `sciknow/ingestion/chunker.py` — add a new `(canonical_type, [prefixes])` tuple. If it should not be embedded, add the type to `_SKIP_SECTIONS`. Add chunking parameters to `_PARAMS` if different from the default.

### Changing the embedding model

1. Update `EMBEDDING_MODEL` and `EMBEDDING_DIM` in `.env`
2. Re-initialize Qdrant collections: `sciknow db init` (will create new collections if missing; existing data is preserved)
3. Re-embed all papers: existing chunks will have a mismatched `embedding_model` field and need re-embedding via `db reset` + re-ingest

### Metadata quality

Papers are tagged with `metadata_source` in `paper_metadata`:
- `crossref` — highest quality, authoritative
- `arxiv` — good for preprints
- `embedded_pdf` — variable quality
- `llm_extracted` — fallback, may have errors
- `unknown` — LLM fallback also failed

Papers with `metadata_source = 'llm_extracted'` or `'unknown'` are candidates for manual review or re-ingestion after adding the DOI to the filename.

### Resuming failed ingestion

A failed PDF can be re-ingested — the pipeline detects the existing record by SHA-256 hash, resets its status, and retries from the beginning. The `data/failed/` directory is a copy; the pipeline re-reads from the original path stored in `documents.original_path`.

### Scaling embedding batch size

`EMBEDDING_BATCH_SIZE=32` is the default — safe on a 24 GB 3090 with a 32B q4 LLM co-resident. With only the embedder on the GPU (no LLM), raise to 64 for faster bulk ingestion. Drop to 16 when running MinerU `--workers 2` simultaneously (each worker loads its own bge-m3 + MinerU models).

See `OPTIMIZATION.md` for the full tuning guide (3090 + incoming DGX Spark).

---

## Retrieval

Hybrid search combining three signal types, fused with Reciprocal Rank Fusion (RRF), reranked with a cross-encoder, and boosted by citation count.

### How it works

**Step 1: Query embedding**

The query is embedded with the same `BAAI/bge-m3` model used at ingest time, producing both a dense vector (1024-dim) and a sparse lexical weight vector.

**Step 2: Three parallel search legs**

| Leg | Backend | Strengths |
|---|---|---|
| Dense vector | Qdrant | Semantic similarity — finds conceptually related text even if phrasing differs |
| Sparse vector | Qdrant | Keyword matching — precise on technical terms, acronyms, species names |
| Full-text search | PostgreSQL `tsvector` | Classical BM25-style relevance, stemming, phrase proximity |

Each leg returns up to 50 candidates (configurable with `--candidates`).

**Step 3: RRF fusion**

Results from all three legs are merged using Reciprocal Rank Fusion:

```
score(chunk) = Σ weight_i / (60 + rank_i)
```

Default weights: dense=1.0, sparse=1.0, FTS=0.5. The top 50 fused candidates proceed to reranking.

**Step 4: Citation-count boost**

Papers cited by more corpus papers receive a log-dampened score boost: `score *= (1 + 0.1 × log₂(1 + citation_count))`. Controlled by `CITATION_BOOST_FACTOR` (0 to disable). The boost is gentle — retrieval signals still dominate.

**Step 5: Cross-encoder reranking**

`BAAI/bge-reranker-v2-m3` scores each `(query, chunk_text)` pair directly. This is slower but much more accurate than embedding similarity alone. Returns top `--top-k` results (default 10).

**Step 6: Metadata hydration**

Full chunk content and bibliographic metadata (title, authors, year, DOI, journal, citation count) are fetched from PostgreSQL and attached to each result.

**Optional: LLM query expansion** (`--expand` / `-e`)

Before step 1, sends the query to `LLM_FAST_MODEL` to add synonyms, acronyms, and related terms. The expanded query feeds the dense + sparse legs; PostgreSQL FTS keeps the original for precision. Falls through silently if Ollama is unavailable.

### Filters

All filters are applied before the vector search (as Qdrant pre-filters and SQL WHERE clauses), so they do not reduce recall within the matching set:

| Filter | Flag | Example |
|---|---|---|
| Year range | `--year-from`, `--year-to` | `--year-from 2015 --year-to 2023` |
| Domain tag | `--domain` | `--domain climatology` |
| Section type | `--section` | `--section methods` |
| Topic cluster | `--topic` | `--topic "Solar Irradiance"` |

## Question Answering (RAG)

`sciknow ask question` retrieves the most relevant passages from your library and asks the configured LLM to answer grounded strictly in those sources.

### How it works

1. **Retrieval** — same hybrid search pipeline (dense + sparse + FTS → RRF → citation boost → reranker). Produces the top `--context-k` chunks (default 8).
2. **Context assembly** — chunks are numbered `[1]`, `[2]`, … and formatted with paper title, year, and section type as a header.
3. **LLM completion** — the context + question is sent to Ollama via the `sciknow/rag/llm.py` wrapper. The response is **streamed token by token** to the terminal.
4. **Sources** — after the answer, the source list (title, year, authors, DOI) is printed.

### LLM requirements

The default model is `LLM_MODEL` from `.env` (default: `qwen2.5:32b-instruct-q4_K_M`). Pull it once:
```bash
ollama pull qwen2.5:32b-instruct-q4_K_M
```
Until then, use `--model mistral:7b-instruct-q4_K_M` (already downloaded by setup.sh) for testing. The 7B model works but produces shallower answers.

### Context window sizing

| Model | Context (`--num-ctx`) | Max chunks |
|---|---|---|
| qwen2.5:32b | 16 384 tokens (default) | ~30 chunks |
| mistral:7b | 8 192 tokens (default) | ~15 chunks |

Each chunk is ~512 tokens + headers. The default `--context-k 8` uses ~4 500 tokens, well within both budgets.

---

## Writing Assistant

Extends the RAG pipeline with longer-form generation tasks. All commands stream output and print sources.

### `ask synthesize`

Retrieves the most relevant passages for a topic (default `--context-k 12`) and asks the LLM to write a structured synthesis covering: key findings, methodological approaches, consensus, and open questions.

### `ask write`

Drafts a specific paper section (`introduction`, `methods`, `results`, `discussion`, `conclusion`) on a given topic. The search query is biased toward the target section type to retrieve the most relevant content.

### `ask write --save`

Add `--save` to any `ask write` invocation to persist the draft to the database. Optionally associate it with a book and chapter:

```bash
sciknow ask write "solar activity proxies" --section introduction --save
sciknow ask write "ocean heat content trends" --section results --save \
    --book "Global Cooling" --chapter 3
```

Saved drafts appear in `sciknow draft list` and can be exported to Markdown with `sciknow draft export`.

### `db export`

Exports the knowledge base as a JSONL fine-tuning dataset:
- **Default (no flags)**: exports each chunk with its metadata. Use for embedding model fine-tuning or retrieval augmentation training.
- **`--generate-qa`**: calls Ollama on each chunk to generate a specific question whose answer is contained in that passage. Produces `(question, context, answer)` triples for instruction fine-tuning. Rate: ~5–10 s/chunk; for 500 chunks expect ~1 hour.

Fields always present: `title`, `year`, `section`, `doi`, `content`
Fields with `--generate-qa`: adds `question`, `answer`

---

## Book Writing System

A structured writing pipeline for long-form academic projects (books, reports, review articles). Built on the RAG pipeline, with iterative refinement and cross-chapter coherence.

### Data model

```
Book  (title, description, plan, status)
  ├── BookChapter  (number, title, description, topic_query, topic_cluster)
  │     └── Draft  (section_type, content, sources, summary, version,
  │                  parent_draft_id, review_feedback, model_used)
  └── BookGap  (gap_type, description, chapter_id, status, resolved_draft_id)
```

Key fields added in writing system v2:
- `books.plan` — thesis statement + scope (200-500 words), injected into every write call
- `drafts.summary` — auto-generated 100-200 word summary for cross-chapter context
- `drafts.parent_draft_id` — links revisions to their parent (version chain: v1 → v2 → v3)
- `drafts.review_feedback` — the reviewer agent's structured assessment
- `book_gaps` — persistent gap tracking with type, priority, and resolution status

### Cross-chapter coherence

The biggest quality improvement over simple per-section generation. When writing Chapter N, the prompt automatically includes:
1. **Book plan** — the thesis statement and scope document (generated once with `book plan`)
2. **Prior chapter summaries** — auto-generated 100-200 word summaries of every draft from chapters 1 through N-1

This prevents contradictions, repeated explanations, and inconsistent terminology across chapters.

### The write → review → revise loop

Every draft goes through an iterative refinement cycle:

1. **Write** (`book write`) — RAG-grounded draft with cross-chapter context
2. **Review** (`book review`) — LLM critic assesses groundedness, completeness, accuracy, coherence, redundancy. Saves structured feedback.
3. **Revise** (`book revise`) — applies review feedback (or a custom instruction) to produce version N+1. The original is preserved.
4. Repeat 2-3 until satisfied.

### Additional quality passes (flags on `book write`)

| Flag | What it does |
|---|---|
| `--plan` | Shows a paragraph-by-paragraph sentence plan before drafting |
| `--verify` | Post-generation claim verification — checks each [N] citation against its source passage, reports a groundedness score |
| `--ipcc` | Adds IPCC-style calibrated uncertainty language (very likely, high confidence, etc.) based on evidence assessment |
| `--expand` | LLM query expansion before retrieval |

### Topic clustering

Before writing, `sciknow catalog cluster` groups papers into 6-14 named thematic clusters. Clusters act as pre-filters in search and writing:

```bash
sciknow search query "solar cycle" --topic "Solar Irradiance"
sciknow book write "Global Cooling" 1 --section introduction    # uses chapter's topic_cluster
```

### Argument mapping

`sciknow book argue` classifies retrieved evidence as SUPPORTS / CONTRADICTS / NEUTRAL for any claim and writes a structured argument map. Optionally saved as a draft.

### Gap analysis

`sciknow book gaps` identifies four types of gaps (topic, evidence, argument, draft) and persists them to the `book_gaps` table for tracking across sessions. View saved gaps in `book show`.

### Web reader (`book serve`)

`sciknow book serve "Global Cooling"` launches a local web application at `http://127.0.0.1:8765` with:

- **Sidebar navigation** — chapters and sections with word counts and version numbers
- **Live rendering** — content served from PostgreSQL, refreshes on each page load (sees autowrite progress in real-time)
- **Inline editing** — click "Edit" to modify draft text directly in the browser, saves back to DB
- **Comments/annotations** — add comments per section, resolve them when addressed. Stored in the `draft_comments` table.
- **Citation links** — `[N]` references are highlighted and the source list is shown in the right panel
- **Review feedback** — the right panel shows the latest review from `book review`
- **Search** — search within all book content
- **Dark/light theme** — toggle with the button in the bottom-right corner
- **No external dependencies** — pure HTML/CSS/JS, no npm, no React, no build step

The web reader is the recommended way to read and interact with the book while writing. The autowrite loop can run in one terminal while you browse the evolving book in the browser.

### Export formats

| Format | Command | Notes |
|---|---|---|
| Markdown | `book export "..." -o book.md` | Default. Inline [N] citations + bibliography |
| HTML | `book export "..." --format html -o book.html` | Self-contained reader with sidebar + theme. Shareable. |
| BibTeX | `book export "..." --format bibtex -o refs.bib` | From paper_metadata (DOI, authors, journal, etc.) |
| LaTeX | `book export "..." --format latex -o book.tex` | Via Pandoc + `--citeproc` (requires `pandoc` installed) |
| DOCX | `book export "..." --format docx -o book.docx` | Via Pandoc (requires `pandoc` installed) |

### Workflow: writing a book from scratch

#### Step 1 — Cluster your papers into topics

Gives the LLM a map of what's in your library before proposing chapters:

```bash
sciknow catalog cluster
sciknow catalog topics              # see what clusters were created
```

#### Step 2 — Create the book and generate its structure

```bash
sciknow book create "Global Cooling" \
    --description "Evidence for solar-driven climate variability and the case for an approaching cooling period"

sciknow book outline "Global Cooling"     # LLM proposes 6-12 chapters from your corpus
sciknow book show "Global Cooling"        # review the proposed chapters
```

Adjust manually if needed:

```bash
sciknow book chapter add "Global Cooling" "Policy Implications" --number 12
```

#### Step 3 — Generate the book plan

The thesis statement that anchors every chapter — the single most important step for coherence:

```bash
sciknow book plan "Global Cooling"
```

Read it carefully. Regenerate with `--edit` if the scope or thesis doesn't match your intent.

#### Step 4 — Check gaps before writing

```bash
sciknow book gaps "Global Cooling"
```

This tells you which chapters have weak paper support and suggests search terms to fill them. If gaps are severe, expand the corpus in that specific area:

```bash
sciknow db expand --limit 50 -q "solar grand minimum Little Ice Age cooling"
```

#### Step 5 — Write chapter by chapter

Start from chapter 1 and work forward — each chapter gets the summaries of all prior chapters as context:

```bash
# Chapter 1: introduction with sentence plan + verification
sciknow book write "Global Cooling" 1 --section introduction --plan --verify

# Chapter 1: methods
sciknow book write "Global Cooling" 1 --section methods

# Chapter 2 (now aware of chapter 1's content via cross-chapter summaries)
sciknow book write "Global Cooling" 2 --section introduction --plan
sciknow book write "Global Cooling" 2 --section results --ipcc
```

#### Step 6 — Review and revise

After each section:

```bash
# See all drafts for the book
sciknow draft list --book "Global Cooling"

# Run the critic (assesses groundedness, completeness, accuracy, coherence)
sciknow book review <draft_id>

# Apply the review feedback (creates v2, preserves v1)
sciknow book revise <draft_id>

# Or give a specific instruction
sciknow book revise <draft_id> -i "expand the discussion of Maunder Minimum evidence with more proxy data"
```

#### Step 7 — Argument mapping for contested claims

For any claim where the evidence is mixed:

```bash
sciknow book argue "solar activity is the primary driver of 20th century warming" --save
sciknow book argue "cosmic rays significantly modulate low cloud cover" --save
```

These map SUPPORTS / CONTRADICTS / NEUTRAL evidence. Use the insights to write more nuanced discussion sections.

#### Step 8 — Re-check gaps, iterate

```bash
sciknow book gaps "Global Cooling"          # what's still missing?
sciknow book show "Global Cooling"          # overview of all chapters + drafts + gaps
```

#### Step 9 — Export

```bash
# Markdown
sciknow book export "Global Cooling" -o manuscript.md

# LaTeX for journal submission (requires pandoc)
sciknow book export "Global Cooling" --format latex -o manuscript.tex
sciknow book export "Global Cooling" --format bibtex -o refs.bib

# DOCX for collaborators
sciknow book export "Global Cooling" --format docx -o manuscript.docx
```

### Autowrite: autonomous convergence (inspired by Karpathy's autoresearch)

The `book autowrite` command implements an autonomous **write → score → revise → re-score** loop that iteratively converges on a high-quality draft. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), which runs propose → train → evaluate → keep/discard loops for ML experiments.

**How it works:**

1. Generates an initial draft (with book plan + cross-chapter summaries for coherence)
2. Scores the draft on 5 dimensions (0.0–1.0): groundedness, completeness, coherence, citation accuracy, overall
3. If overall ≥ target score → **converged**, stop
4. Identifies the **weakest dimension** and generates a targeted revision instruction
5. Revises the draft targeting that specific weakness
6. Re-scores the revision: if improved → **keep**, if regressed → **discard**
7. Repeats until converged or max iterations exhausted

**Convergence example:**
```
v1: groun=0.65  compl=0.60  coher=0.80  citat=0.70  overall=0.69
    Weakest: completeness → "Add discussion of proxy calibration methods"
v2: groun=0.72  compl=0.78  coher=0.82  citat=0.75  overall=0.77  ✓ KEEP
    Weakest: groundedness → "Cite primary sources for claims in paragraph 3"
v3: groun=0.85  compl=0.80  coher=0.85  citat=0.82  overall=0.83  ✓ KEEP
v4: groun=0.88  compl=0.83  coher=0.87  citat=0.85  overall=0.86  ✓ CONVERGED
```

**Modes:**
- Single section: `book autowrite "Book" 3 --section methods`
- All sections of one chapter: `book autowrite "Book" 3 --section all`
- Full book (all chapters × all sections): `book autowrite "Book" --full`

**Flags:**
- `--max-iter N` — max iterations per section (default 3)
- `--target-score 0.85` — quality threshold to stop (default 0.85)
- `--auto-expand` — when the reviewer identifies missing evidence, checks if the corpus has coverage and flags topics for expansion
- `--ipcc` — applies IPCC uncertainty language to the final converged version

### Tips for effective book writing with sciknow

- **Write sequentially** (ch1 → ch2 → ch3). Each chapter's prompt includes summaries of all prior chapters — writing out of order loses this coherence benefit.
- **Use `--plan` on the first draft** of each section. It shows the paragraph skeleton before the full draft streams, letting you spot structural issues early.
- **Use `--verify` on important sections.** Catches hallucinated citations before they propagate.
- **Use `--ipcc` on results/discussion sections.** Inserts calibrated uncertainty language ("very likely", "high confidence") appropriate for climate science publications.
- **Argue before you write discussion sections.** Run `book argue` on your key claims first, then use those structured evidence maps to write a more nuanced discussion.
- **Expand targetedly when gaps appear.** If `book gaps` says chapter 7 has weak support, run `db expand -q "your topic" --limit 30` to grow the corpus in that area, then re-write.
- **Review every section before moving on.** The `book review → book revise` loop catches groundedness issues, missing topics, and redundancy early rather than in the final manuscript.

## Backup & Restore (`db backup` / `db restore`)

Move or clone the full sciknow collection to another machine.

### What gets backed up

| Component | Default | Flag to change |
|---|---|---|
| PostgreSQL database (papers, chunks, metadata) | ✅ always | — |
| Qdrant vector embeddings | ✅ always | `--skip-vectors` on restore |
| Original ingested PDFs (`data/processed/`) | ✅ on | `--no-pdfs` |
| Auto-downloaded PDFs (`data/downloads/`) | ✅ on | `--no-downloads` |
| Marker markdown output (`data/mineru_output/`) | ❌ off | `--marker` |
| `.env` config file | ✅ always | — |

The Marker output is off by default because it can be regenerated from the PDFs
during re-ingestion. Excluding it makes the backup significantly smaller.

### Backup

```bash
# Full backup (PDFs + vectors + DB) — recommended
sciknow db backup --output sciknow_2026-04-04.tar.gz

# Metadata + vectors only (much smaller, no raw PDFs)
sciknow db backup --no-pdfs --no-downloads --output sciknow_metadata.tar.gz

# Include Marker output (avoids re-running OCR on restore)
sciknow db backup --marker --output sciknow_full.tar.gz
```

### Restore on a new machine

```bash
# 1. Install sciknow and start services (PostgreSQL + Qdrant + Ollama)
git clone https://github.com/you/sciknow
cd sciknow
./scripts/setup.sh

# 2. Restore the backup
sciknow db restore sciknow_2026-04-04.tar.gz

# 3. If the DB already exists (e.g. a previous partial install), use --force
sciknow db restore sciknow_2026-04-04.tar.gz --force

# 4. Verify
sciknow db stats
sciknow catalog stats
```

> **After restore:** edit `.env` if the new machine uses different database
> credentials, hostnames, or model names. The restored `.env` reflects the
> original machine's settings.

## Reference Expansion (`db expand`)

Automatically grows the collection by following citations in existing papers. Expand v2 (2026-04) adds semantic relevance filtering, four new OA sources, provenance tracking, and batched in-process ingestion.

### How it works

1. **Reference extraction** — for each paper in the collection, references are pulled from four sources and unioned:
   - **Crossref reference list** — structured `reference` array from the Crossref API (the bulk of refs for papers with DOIs).
   - **MinerU `content_list.json`** — walks the typed-block list looking for the References heading, then harvests DOIs/arXiv IDs from subsequent text blocks until the next top-level heading (Appendix, Supplementary, etc.). Primary source for MinerU-converted papers.
   - **Marker markdown bibliography** — legacy regex parser for papers ingested before the MinerU switch.
   - **OpenAlex `referenced_works`** — per-paper API call for papers where local sources yielded < 10 refs. Batch-resolves OpenAlex work IDs to DOIs (50 per request). Catches preprints and post-2020 publications that Crossref deposits often miss.

2. **Deduplication** — references already present in the collection (by DOI or arXiv ID) are skipped. Within-batch dedup also collapses near-duplicates by normalised title prefix.

3. **Semantic relevance filter** *(`--relevance`, on by default)* — candidate reference titles are embedded with bge-m3 and scored against either the **corpus centroid** (mean of all abstract embeddings — default) or a **user-provided topic query** (`-q "solar forcing"` / `--relevance-query ...`). References scoring below `EXPAND_RELEVANCE_THRESHOLD` (default 0.55 cosine similarity) are dropped. A score histogram with the cut point is printed so you can sanity-check the threshold before committing. Degrades gracefully on GPU OOM (common when ingestion is active) — prints a warning and continues without the filter. Disable explicitly with `--no-relevance`.

4. **Title resolution** *(opt-in, `--resolve`)* — for references with only a title (no DOI), Crossref title search is used to find a DOI (~0.3 s each). Off by default because the base pool of DOI-bearing references is usually large enough.

5. **Open-access PDF discovery** — for each surviving candidate, six sources are queried in priority order:
   - **Copernicus** — zero-cost URL construction for `10.5194/*` DOIs covering ACP, CP, TC, ESSD, BG, HESS, GMD, ESD, NHESS, OS, SE, WCD. Typically 15–25% of a climate/earth-science corpus lands here with no network lookup at all.
   - **arXiv** — direct PDF for any arXiv ID.
   - **Unpaywall** (`api.unpaywall.org`) — largest general OA database.
   - **OpenAlex** `best_oa_location.pdf_url` — catches preprints and institutional repos Unpaywall misses.
   - **Europe PMC** — free full-text for biomedical papers (`fullTextUrlList` + PMC article fallback).
   - **Semantic Scholar** — final fallback via their public API.

6. **Parallel download + batch ingest** — downloads run in a thread pool (`EXPAND_DOWNLOAD_WORKERS`, default 6). After **all** downloads complete, the main process calls the ingestion pipeline directly on each new PDF in-process, so MinerU + bge-m3 models load once and stay resident across the whole batch. This eliminates the ~15–20s of per-file Python/MinerU startup that the old subprocess-per-file approach paid. Sciknow's SHA-256 hash dedup makes the old `.ingest_done` cache redundant (still read for backward-compat, no longer written).

7. **Provenance tagging** — every paper added by expand is tagged `ingest_source='expand'` in the `documents` table. `sciknow db stats` shows an "Ingest source" breakdown alongside the status breakdown, so you can see at a glance how much of your library is seed material vs grown.

```bash
# Preview what would be added, with the relevance filter on (default)
sciknow db expand --dry-run --limit 50

# Run it for real with a bounded first expansion
sciknow db expand --limit 100

# Topic-targeted expansion using a free-text anchor instead of the centroid
sciknow db expand --limit 50 -q "solar irradiance climate forcing"

# Tune the relevance threshold (higher = more selective)
sciknow db expand --relevance-threshold 0.65 --limit 100

# Disable the filter entirely (not recommended — expand will drift off-topic)
sciknow db expand --no-relevance --limit 100

# Download-only (skip ingest, useful if you want to curate the downloads folder first)
sciknow db expand --no-ingest --limit 50

# Also resolve title-only references to DOIs (slow, ~0.3 s each)
sciknow db expand --resolve --limit 50

# Parallel post-download ingestion (N worker subprocesses, each loading
# its own MinerU + bge-m3). Same VRAM rules as `sciknow ingest directory`:
# ~9-10 GB per worker, so --workers 2 only when the LLM is off-GPU.
sciknow db expand --limit 200 --workers 2
```

The ingest phase of `db expand` (just-downloaded PDFs) reuses the same worker-subprocess fan-out as `sciknow ingest directory`. The relevance filter's bge-m3 is automatically released before workers spawn, so you don't pay for two copies even with `--relevance` on. When `--workers` is omitted, the command falls back to `INGEST_WORKERS` from `.env` (default 1).

### Expected open-access hit rate

Not every referenced paper is freely available. Typical coverage with the expand v2 six-source chain (Copernicus + arXiv + Unpaywall + OpenAlex + Europe PMC + Semantic Scholar):

| Paper age | OA hit rate (Unpaywall only) | OA hit rate (v2 chain) |
|---|---|---|
| 2020–present | ~50–65% | ~65–80% |
| 2010–2019 | ~35–50% | ~45–60% |
| Pre-2010 | ~15–30% | ~20–35% |

Climate / earth-science libraries see larger lifts because Copernicus journals (ACP, CP, TC, ESSD, ...) hit with zero API calls via URL pattern matching. Biomedical / environmental-health overlap benefits from Europe PMC.

Running `db expand` is non-destructive and idempotent — already-downloaded files are skipped on re-runs, and papers already in the collection are never re-ingested (sciknow's SHA-256 hash dedup handles this automatically, independent of the on-disk `.no_oa_cache` file).

## Metadata Enrichment (`db enrich`)

After ingestion, many papers will be missing a DOI because the standard extraction
pipeline only finds DOIs that are **already embedded in the PDF**. `db enrich`
closes this gap by doing an active lookup for each paper that lacks a DOI.

### How it works

1. **Crossref title search** — queries `api.crossref.org/works?query.title=...` with
   the paper's title and (if available) first author. Returns up to 5 candidates.
2. **Fuzzy title matching** — normalises both titles (lowercase, strip punctuation)
   and computes a similarity score with `difflib.SequenceMatcher`. Only accepts
   matches above the `--threshold` (default 0.85).
3. **OpenAlex fallback** — if Crossref returns no confident match, queries
   `api.openalex.org/works` as a second source. OpenAlex covers some works
   Crossref misses (preprints on institutional repos, book chapters, etc.).
4. **arXiv fallback** — for papers that have an arXiv ID but no full metadata.
5. **Full Crossref hydration** — once a DOI is confirmed, fetches the complete
   record (abstract, authors, journal, volume, pages, keywords) from Crossref.

```bash
sciknow db enrich                     # Full run on all papers without a DOI
sciknow db enrich --dry-run           # Preview matches without writing to DB
sciknow db enrich --threshold 0.80    # Looser matching (more matches, small false-positive risk)
sciknow db enrich --limit 50          # Process only the first 50 papers
sciknow db enrich --delay 0.5         # Slower, more polite to the APIs
```

### Why 100% DOI coverage is not achievable

DOIs only exist for content formally deposited in the Crossref/DataCite
registries. A typical research library will include material that has no DOI by
design:

| Content type | Why no DOI | Example |
|---|---|---|
| Books | Have ISBNs, not DOIs | *Climate of the Past, Present and Future* |
| IPCC / UN reports | Grey literature, some chapters have DOIs | IPCC AR6 WG2 |
| Preprints on personal sites | Never registered with Crossref | Self-hosted PDFs |
| Blog posts / podcasts | Not academic publications | Smithsonian articles |
| Self-published pamphlets | No publisher registration | Climate commentary PDFs |
| Conference abstracts | Not always indexed | Some AGU/EGU contributions |

A well-curated library of peer-reviewed journal articles can reach ~95% DOI
coverage. A collection that also includes books, reports, and grey literature
will typically plateau at 55–70%.

## Citation Graph

References are extracted from every ingested paper (from Crossref, MinerU content_list, Marker markdown, and OpenAlex referenced_works) and stored in the `citations` table. When a cited paper is also in the corpus, the row is cross-linked via `cited_document_id`.

**Automatic cross-linking:** during ingestion, for each new paper's references, the pipeline checks if the cited DOI matches an existing corpus paper (forward link). It also backlinks: existing citation rows pointing to this paper's DOI get updated. The graph is consistent regardless of ingestion order.

**Batch re-linking:** `sciknow db link-citations` scans all unlinked citations and sets pointers wherever the cited paper is now in the corpus. Useful after bulk ingestion or expand runs. Shows a "Most-Cited Papers" table.

**Citation-boosted retrieval:** papers cited by more corpus papers rank slightly higher in search results. The boost is log-dampened and configurable via `CITATION_BOOST_FACTOR` (default 0.1, set to 0 to disable).

## Similar Papers

Find papers with similar abstracts using the Qdrant `abstracts` collection:

```bash
sciknow search similar "Water Vapor Feedback"          # by title fragment
sciknow search similar 10.1038/nature12345             # by DOI
sciknow search similar 2301.12345                      # by arXiv ID
sciknow search similar "Water Vapor" --show-scores -k 20  # with similarity scores
```

Uses the same bge-m3 dense embeddings that power the main search pipeline. Returns a ranked table of nearest neighbours from the paper library.

## Provenance Tracking

Every document has an `ingest_source` field:
- `seed` — manually ingested via CLI (`sciknow ingest file`, `sciknow ingest directory`)
- `expand` — auto-discovered via `sciknow db expand`

`sciknow db stats` shows an "Ingest source" breakdown alongside the status breakdown, so you can see at a glance how much of your library is seed material vs auto-grown.

## Pre-flight Checks

All long-running commands (`ingest`, `expand`, `enrich`, `export`, `catalog cluster`) verify that PostgreSQL and/or Qdrant are reachable **before** doing any work. If a service is down, you get an actionable error message immediately instead of waiting for MinerU to run for minutes and then crashing at the database write.

```
✗ Qdrant is unreachable.
  Check that the service is running:
    systemctl --user status qdrant
    systemctl --user start qdrant
  And that .env has the correct QDRANT_HOST/QDRANT_PORT.
```
