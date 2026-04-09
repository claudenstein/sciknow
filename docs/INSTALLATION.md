# Installation

[&larr; Back to README](../README.md)

---

## Quick Start (setup.sh)

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

### Pull the main LLM

After setup, pull the primary model when you have the full VRAM budget available:

```bash
ollama pull qwen3.5:27b
```

### Configure

Edit `.env` — the only required change is your email for the Crossref polite pool:

```bash
nano .env
# Set: CROSSREF_EMAIL=you@youremail.com
```

### Initialize the schema

```bash
sciknow db init
```

This runs Alembic migrations (creates all tables, indexes, and the full-text search trigger) and initializes the Qdrant collections.

### Verify

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
ollama pull qwen3.5:27b
ollama pull qwen3:8b
```

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

## Ollama Performance Tuning

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
| `LLM_MODEL` | `qwen3.5:27b` | Main LLM (Ollama model name) |
| `LLM_FAST_MODEL` | `qwen3:8b` | Fast LLM for metadata extraction, wiki compile |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `CROSSREF_EMAIL` | `user@example.com` | **Set this.** Used in Crossref/OpenAlex polite pool User-Agent |
| `PDF_CONVERTER_BACKEND` | `auto` | `auto` / `mineru` / `marker` |
| `EMBEDDING_BATCH_SIZE` | `32` | Chunks per bge-m3 batch (16 if LLM co-resident, 64 for embedder-only runs) |
| `MARKER_BATCH_MULTIPLIER` | `2` | Marker/Surya internal batch size multiplier |
| `PG_POOL_SIZE` / `PG_MAX_OVERFLOW` | `20` / `20` | SQLAlchemy connection pool |
| `QDRANT_HNSW_M` | `32` | HNSW graph connectivity (collection creation) |
| `QDRANT_HNSW_EF_CONSTRUCT` | `256` | HNSW build-time exploration |
| `QDRANT_HNSW_EF` | `128` | HNSW query-time exploration |
| `QDRANT_SCALAR_QUANTIZATION` | `true` | int8 dense-vector quantization (~75% memory savings) |
| `INGEST_WORKERS` | `1` | Parallel worker subprocesses for `ingest directory` |
| `ENRICH_WORKERS` | `8` | Concurrent Crossref/OpenAlex lookups |
| `EXPAND_DOWNLOAD_WORKERS` | `6` | Concurrent OA PDF lookups |
| `LLM_PARALLEL_WORKERS` | `4` | Concurrent LLM calls; must be &le; `OLLAMA_NUM_PARALLEL` |
| `EXPAND_RELEVANCE_THRESHOLD` | `0.55` | Cosine similarity cut-off for expand relevance filter |

**Remote GPU server:** Change only `OLLAMA_HOST=http://your-gpu-server:11434`. Everything else stays the same.

**Server-side Ollama parallelism:** the default `OLLAMA_NUM_PARALLEL=1` serialises every LLM request regardless of client-side concurrency. To unlock `LLM_PARALLEL_WORKERS`, set it on the Ollama host:
```bash
systemctl --user edit ollama
# add:  Environment="OLLAMA_NUM_PARALLEL=4"
systemctl --user restart ollama
```

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
- qwen3.5:27b-q4_K_M (LLM, weights only): ~14 GB
- KV cache for the LLM at 16K context: ~7-9 GB (with `OLLAMA_KV_CACHE_TYPE=q8_0`)
- bge-reranker-v2-m3: ~0.5 GB
- **Realistic total with the LLM warm: ~22-23 GB** — leaves only ~1 GB headroom

**The 16K context wrinkle.** With `num_ctx=16384` (sciknow's default for `score`/`verify`/`cove`/`raptor` calls), the KV cache alone is 7-9 GB even with `q8_0` cache. Combined with model weights, qwen3.5:27b at Q4_K_M uses **~23 GB of VRAM all by itself** while it's loaded by Ollama. There is **no room left** for sciknow's bge-m3 + bge-reranker (~3 GB combined). That's why you may have seen intermittent CUDA OOM errors during retrieval.

**Phase 15.2 fix (automatic).** sciknow now auto-detects this situation: at retrieval time, `sciknow/retrieval/device.py` queries `torch.cuda.mem_get_info()` and falls back to **CPU** for bge-m3 + reranker when less than 4 GB of VRAM is free. The cost is small (~2 seconds extra per retrieval — bge-m3 query encoding on CPU is ~1 s and reranking 50 candidates is ~5 s) and the LLM stays warm in VRAM throughout the autowrite session, which is far better than the alternative of evicting and reloading the model on every retrieval (~30-60 s per reload).

You can override the auto-detection with an env var:

```bash
SCIKNOW_RETRIEVAL_DEVICE=cpu       # always CPU (most conservative)
SCIKNOW_RETRIEVAL_DEVICE=cuda      # always GPU (will OOM if LLM is loaded)
SCIKNOW_RETRIEVAL_DEVICE=auto      # default — probes free VRAM
```

If you still hit OOM after the 15.2 fix, your options in order of effort:

1. **Reduce `num_ctx`** in sciknow's flagship calls (e.g. score/verify from 16384 → 8192). Cuts the KV cache in half. Cost: less context for scoring/verification, which can hurt quality on long sections.
2. **Use a smaller LLM**: `qwen3:14b` at Q4 = ~9 GB, leaves ~14 GB for everything else.
3. **Lower the keep-alive**: `OLLAMA_KEEP_ALIVE=30s` in your environment. The LLM gets evicted between retrieval phases. Cost: ~30-60 s reload before each LLM call.
4. **Move the LLM to a remote GPU**: `OLLAMA_HOST=http://your-gpu-server:11434` (see below).

**Remote GPU server:** Set `OLLAMA_HOST=http://your-gpu-server:11434` in `.env`. Zero code changes needed. With the LLM offloaded to a remote box, the local 24 GB GPU is free for bge-m3 + reranker at full speed.
