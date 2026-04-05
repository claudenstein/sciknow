# SciKnow

A local-first, large-scale scientific knowledge system. Ingests scientific papers (PDFs — both scanned and text-based), converts them to structured markdown, extracts metadata, chunks them section-aware, embeds them with a dense+sparse model, and stores everything in a PostgreSQL + Qdrant stack for fast hybrid retrieval.

All AI inference runs locally. No cloud APIs required to operate the system.

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 0 | Infrastructure (PostgreSQL, Qdrant, migrations) | **Done** |
| 1 | Ingestion pipeline (PDF → markdown → metadata → chunks → embeddings) | **Done** |
| 2 | Hybrid search (dense + sparse + FTS, RRF fusion, reranker) | **Done** |
| 3 | RAG — grounded question answering over papers | **Done** |
| 4 | Writing assistant — multi-paper synthesis, section drafting | **Done** |
| 5 | Draft persistence — save, browse, export writing drafts | **Done** |
| 6 | Book structure — books → chapters hierarchy with LLM-generated outline | **Done** |
| 7 | Topic clustering — LLM assigns named clusters to papers for filtered search | **Done** |
| 8 | Argument mapping — evidence classification (SUPPORTS/CONTRADICTS/NEUTRAL) | **Done** |
| 9 | Gap analysis — LLM identifies missing topics, weak chapters, unwritten sections | **Done** |
| 10 | Multi-project support — separate books, shared paper library | **Done** |

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
 [1] Marker (marker-pdf)        PDF → structured Markdown + images
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
│   └── mineru_output/          # raw Marker output per paper
│       └── {uuid}/
│           ├── {name}.json     # Marker block-structured JSON (primary)
│           ├── {name}.md       # markdown fallback (only if JSON fails)
│           └── images/         # extracted images (markdown fallback only)
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
    │   ├── pdf_converter.py    # Marker (marker-pdf) wrapper
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
- Installs Marker (`marker-pdf`); models are auto-downloaded from HuggingFace on first use
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

**Python environment:**
```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

**Marker:**
```bash
uv pip install marker-pdf
```

Models (Surya OCR, layout) are downloaded automatically from HuggingFace on first use and cached in `~/.cache/huggingface/`. GPU is used automatically if CUDA is available.

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
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker (used in Phase 2+) |
| `CROSSREF_EMAIL` | `user@example.com` | **Set this.** Used in Crossref User-Agent header |
| `EMBEDDING_BATCH_SIZE` | `8` | Chunks per embedding batch (tune for your VRAM) |

**Remote GPU server:** Change only `OLLAMA_HOST=http://your-gpu-server:11434`. Everything else stays the same.

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

# List all clusters with paper counts
sciknow catalog topics
```

### `sciknow book`

The book system organises writing projects. Each book has chapters; each chapter can have multiple draft sections.

```bash
# Create a new book project
sciknow book create "Global Cooling"
sciknow book create "Solar Cycle Mechanisms" --description "Overview of solar variability and climate links"

# List all books
sciknow book list

# Show a book's chapters and draft progress
sciknow book show "Global Cooling"

# Add a chapter manually
sciknow book chapter add "Global Cooling" "The Maunder Minimum"
sciknow book chapter add "Global Cooling" "Ocean Heat Uptake and Thermal Lag" --number 5

# Generate a full chapter outline with the LLM (proposes 6–12 chapters based on your paper library)
sciknow book outline "Global Cooling"

# Draft a chapter section (streams output + saves to drafts table)
sciknow book write "Global Cooling" 2                       # Chapter 2, default section = introduction
sciknow book write "Global Cooling" 3 --section methods
sciknow book write "Global Cooling" 1 --context-k 15       # More context passages

# Map evidence for/against a claim (argument analysis)
sciknow book argue "Global cooling is primarily driven by solar variability"
sciknow book argue "Aerosol forcing exceeds solar forcing since 1980" --save

# Identify gaps in the book: missing topics, weak chapters, unwritten sections
sciknow book gaps "Global Cooling"

# Compile all chapter drafts into a single Markdown document
sciknow book export "Global Cooling"
sciknow book export "Global Cooling" --output global_cooling_manuscript.md
```

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

### Stage 1 — PDF Conversion (Marker)

Marker (`marker-pdf`) detects whether a PDF is text-based or scanned and applies the appropriate pipeline:
- **Text PDF:** layout analysis, reading order detection, multi-column handling, LaTeX math
- **Scanned PDF:** Surya OCR + layout analysis, GPU-accelerated

**Output format — JSON (primary):** Marker's `JSONRenderer` produces a structured block tree where every element is typed: `SectionHeader` (with explicit heading level h1–h4), `Text`, `Table`, `Equation`, `ListItem`, `Caption`, `PageHeader`, `PageFooter`, `Figure`, etc. This is used by the section-aware chunker for exact structural detection instead of regex heuristics.

**Fallback — Markdown:** if JSON conversion fails for any document, Marker falls back to `MarkdownRenderer` and saves a `.md` file. The rest of the pipeline continues normally.

Output is saved to `data/mineru_output/{doc_id}/{stem}.json` (or `.md` on fallback). Models are downloaded automatically from HuggingFace on first use and cached in `~/.cache/datalab/`. Marker models are loaded once and cached in memory for the duration of an ingestion run.

### Stage 2 — Metadata Extraction (4 layers)

Each layer is tried in order; later layers only run if the previous didn't fully populate the metadata:

1. **PyMuPDF** — reads embedded XMP/Info fields from the PDF. Fast but often incomplete.
2. **Crossref API** — authoritative bibliographic data by DOI. DOI is extracted from the paper text via regex. Uses the Crossref polite pool (rate limit: 50 req/s with email in User-Agent).
3. **arXiv API** — for preprints. arXiv IDs are extracted from URLs and filenames.
4. **Ollama LLM fallback** — sends the first ~3000 characters of the document text to `LLM_FAST_MODEL` with a structured JSON extraction prompt. Used when no DOI or arXiv ID is found.

Fields extracted: title, abstract, year, DOI, arXiv ID, journal, volume, issue, pages, publisher, authors (with ORCID and affiliation where available), keywords.

### Stage 3 — Section-Aware Chunking

**JSON mode (primary):** the chunker reads the Marker block tree directly. `SectionHeader` blocks at heading level h1/h2 open new sections; h3/h4 are inlined as bold subheadings within the parent section to avoid fragmentation. `Text`, `Table`, `Equation`, and `ListItem` blocks are accumulated into the current section. `PageHeader`, `PageFooter`, `Figure`, and `Picture` blocks are discarded. Short `Text` blocks that contain only a known section keyword (`Abstract`, `Introduction`, etc.) are also recognised as implicit section headers — this handles older PDFs where heading formatting was lost during scanning.

**Markdown mode (fallback):** sections are detected from markdown headings (`#` through `####`) using regex.

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

## Database Schema

### Tables

| Table | Description |
|---|---|
| `documents` | One row per PDF. Tracks ingestion status and file hash (SHA-256, used for deduplication). |
| `paper_metadata` | Bibliographic metadata: title, abstract, authors, DOI, year, journal, keywords, domains. Maintains a `tsvector` column for full-text search via an automatic trigger. |
| `paper_sections` | Sections extracted from the document (JSON or markdown), classified by type. |
| `chunks` | Individual retrieval units. Each chunk links to a section and a Qdrant point UUID. |
| `citations` | Reference list entries (from Crossref). Cross-linked if a cited paper is also in the DB. |
| `ingestion_jobs` | Audit log of every pipeline stage with timing and status. |

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
| `marker-pdf` | PDF→Markdown conversion (Marker) |
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
3. Re-embed all papers: existing chunks will have a mismatched `embedding_model` field — use `sciknow db reindex` (Phase 2)

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

`EMBEDDING_BATCH_SIZE=8` is conservative for a 3090. With only the embedding model loaded (no LLM), you can increase to 32–64 for faster bulk ingestion. Set back to 8 when running LLM + embedder simultaneously.

---

## Phase 2 — Retrieval

Hybrid search combining three signal types, fused with Reciprocal Rank Fusion (RRF) and reranked with a cross-encoder.

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

**Step 4: Cross-encoder reranking**

`BAAI/bge-reranker-v2-m3` scores each `(query, chunk_text)` pair directly. This is slower but much more accurate than embedding similarity alone. Returns top `--top-k` results (default 10).

**Step 5: Metadata hydration**

Full chunk content and bibliographic metadata (title, authors, year, DOI, journal) are fetched from PostgreSQL and attached to each result.

### Filters

All filters are applied before the vector search (as Qdrant pre-filters and SQL WHERE clauses), so they do not reduce recall within the matching set:

| Filter | Flag | Example |
|---|---|---|
| Year range | `--year-from`, `--year-to` | `--year-from 2015 --year-to 2023` |
| Domain tag | `--domain` | `--domain climatology` |
| Section type | `--section` | `--section methods` |

## Phase 3 — RAG

`sciknow ask question` retrieves the most relevant passages from your library and asks the configured LLM to answer grounded strictly in those sources.

### How it works

1. **Retrieval** — same hybrid search pipeline as Phase 2 (dense + sparse + FTS → RRF → reranker). Produces the top `--context-k` chunks (default 8).
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

## Phase 4 — Writing Assistant

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

## Phase 5–10 — Book Writing System

Phases 5–10 add a structured book-writing workflow on top of the RAG pipeline. The book system is designed for long-form academic writing projects (books, reports, review articles) where many papers contribute to a coherent narrative across multiple chapters.

### Data model

```
Book  (title, description, status)
  └── BookChapter  (number, title, description, topic_query, topic_cluster)
        └── Draft  (section_type, content, sources, word_count, model_used)
```

All three tables live in PostgreSQL alongside the paper library. Drafts store the raw LLM output, the section type (introduction / methods / etc.), and the APA-formatted source list used to generate them.

### Topic clustering (Phase 7)

Before writing, run `sciknow catalog cluster` to group your papers into named thematic clusters. The LLM reads all paper titles and proposes 6–14 descriptive cluster names, then assigns every paper to exactly one cluster.

Once clusters are assigned, they act as a first-level filter in search and writing:

```bash
sciknow search query "solar cycle length" --topic "Solar Irradiance"
sciknow ask question "What drives the 11-year solar cycle?" --topic "Solar Activity"
sciknow book write "Global Cooling" 1 --topic "Solar Irradiance"
```

Cluster assignments are stored in `paper_metadata.topic_cluster` and indexed in Qdrant for fast pre-filtering.

### Workflow: writing a book from scratch

```bash
# 1. Cluster papers into topics
sciknow catalog cluster

# 2. Create the book
sciknow book create "Global Cooling"

# 3. Generate a chapter outline (LLM proposes structure from your paper titles)
sciknow book outline "Global Cooling"

# 4. Review and adjust chapters if needed
sciknow book show "Global Cooling"
sciknow book chapter add "Global Cooling" "Policy Implications" --number 11

# 5. Check for gaps before writing
sciknow book gaps "Global Cooling"

# 6. Write chapter by chapter
sciknow book write "Global Cooling" 1
sciknow book write "Global Cooling" 2 --section introduction
sciknow book write "Global Cooling" 2 --section methods

# 7. Review and export drafts
sciknow draft list --book "Global Cooling"
sciknow draft show <id>
sciknow book export "Global Cooling" --output manuscript.md
```

### Argument mapping (Phase 8)

Use `sciknow book argue` to map the evidence landscape for any claim. The LLM classifies retrieved passages as SUPPORTS, CONTRADICTS, or NEUTRAL and writes a structured argument map:

```bash
sciknow book argue "Global cooling is primarily driven by reduced solar output"
sciknow book argue "Aerosol forcing explains the post-1980 cooling hiatus" --save
```

Saved argument maps are stored as drafts with `section_type = "argument_map"`.

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

Automatically grows the collection by following citations in existing papers.

### How it works

1. **Reference extraction** — for each paper in the collection, references are
   pulled from two sources:
   - *Crossref reference list*: the structured `reference` array returned by
     the Crossref API when the paper has a DOI. These often include DOIs for
     each cited work directly.
   - *Markdown bibliography section*: the references / bibliography heading in
     the Marker-converted markdown, parsed with regex to extract DOIs, arXiv
     IDs, and titles.

2. **Deduplication** — references already present in the collection (by DOI or
   arXiv ID) are skipped.

3. **Title resolution** *(opt-in, `--resolve`)* — for references that have
   only a title (no DOI), Crossref title search is used to find a DOI (~0.3 s
   each). Off by default because the base pool of DOI-bearing references is
   usually large enough.

4. **Open-access PDF discovery** — for each new reference with a DOI or arXiv
   ID, the following sources are queried in order:
   - **Unpaywall** (`api.unpaywall.org`) — largest database of legal OA PDFs
   - **arXiv** — direct PDF for papers with an arXiv ID
   - **Semantic Scholar** — additional OA source

5. **Download + ingest** — valid PDFs are saved to `--download-dir` and
   immediately passed through the full ingestion pipeline (conversion →
   chunking → embedding).

```bash
sciknow db expand                              # Download and ingest all discoverable OA papers
sciknow db expand --dry-run                    # Preview what would be downloaded
sciknow db expand --limit 50                   # Cap at 50 new papers
sciknow db expand --no-ingest                  # Download PDFs but don't ingest yet
sciknow db expand --resolve                    # Also resolve title-only references (slow)
sciknow db expand --download-dir ~/papers/new  # Custom download directory
```

### Expected open-access hit rate

Not every referenced paper is freely available. Typical Unpaywall coverage:

| Paper age | OA hit rate |
|---|---|
| 2020–present | ~50–65% |
| 2010–2019 | ~35–50% |
| Pre-2010 | ~15–30% |

Running `db expand` is non-destructive and idempotent — already-downloaded
files are skipped on re-runs, and papers already in the collection are never
re-ingested.

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
