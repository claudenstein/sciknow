# sciknow v2 — Specification

**Status:** draft (2026-04-24). Authored as the design document for a from-scratch reimplementation that retires the `last-ollama-build` tag (commit `cf91386`) and replaces Ollama with `llama-server` (llama.cpp HTTP server) as the single inference backend.

## 0. Purpose & framing

sciknow today (`v1.x`, last release `54.6.323`) is a working but heavily-patched local-first research/book-writing system. After ~310 patch phases on top of the original architecture, accumulated scar tissue is now the dominant cost: dual-embedder mode, three rerank backends, two retrieval-device fallback paths, releaser registries, GPU ledgers, three pulse formats, 40 migrations, a 31 kLOC `web/app.py` template, a 11.5 kLOC `cli/db.py`, ~20 deprecated alert codes, and several no-op fallbacks that exist only to satisfy phased L1 regression tests.

`v2` keeps everything that earned its keep — the ingestion stage machine, the hybrid-search recipe, the autowrite event protocol, the book-bibliography globaliser, the multi-project layout — and rebuilds them on a single inference substrate (`llama-server`) with an opinionated module layout, a single canonical embedder, and a dedicated rerank server. MinerU 2.5-Pro VLM is the **only** acknowledged non-llama.cpp dependency.

**Non-goals**:
- Re-implementing MinerU. Pin `mineru[vllm]` and treat it as an external service.
- Remote training / fine-tuning. v2 is inference-only.
- Multi-user / multi-tenant. v2 stays single-user, local-first.
- Cloud SaaS. v2 stays self-hosted.

## 1. Design principles

1. **One inference backend.** `llama-server` (llama.cpp HTTP) for *every* generative call, *every* embedding, *every* rerank. No `ollama.chat`, no `FlagEmbedding`, no `sentence-transformers`. Speculative decoding via Lucebox DFlash is opt-in.
2. **One canonical embedder.** Drop the dual-embedder (bge-m3 + Qwen3-Embedding-4B) mode. Pick one (default: `bge-m3` via `llama-server --embedding`) and reflect that in a single Qdrant collection per project.
3. **No two ways to do the same thing.** If a code path has two implementations gated by a flag, the spec picks one. Examples retired in v2: ColBERT prefetch (kept), classic dense+sparse path (retired); per-question CoV (retired), batched CoV (kept); `cli/db.py` expand subcommands (regrouped into `sciknow corpus`).
4. **Stable event protocol.** The 16-event autowrite/book SSE protocol moves to an explicit Pydantic schema in `sciknow/core/events.py` and is the contract between core and every consumer (CLI Rich, web SSE, MCP, tests).
5. **Server-rendered HTML, externalised assets.** The single 31 kLOC f-string template is split into Jinja2 partials + static CSS/JS. No build step (no npm), but assets are real files on disk.
6. **Process boundaries are real.** `llama-server` runs as a managed subprocess (via `sciknow infer up`). sciknow speaks to it over HTTP. No in-process model loads. This kills the entire VRAM-ledger / releaser / preflight class of bugs because there is one process owning VRAM and sciknow never touches it.
7. **Single source of truth.** Postgres for relational data, Qdrant for vectors, filesystem for blobs. No Qdrant-payload-as-source-of-truth, no JSON-blob-in-DB-as-source-of-truth.
8. **Test the contract, not the implementation.** v2 tests the public event protocol, the SSE wire format, the CLI exit codes, and the SQL/Qdrant schemas — not source-grep-for-symbol-name regressions like `l1_phase32_5_*`.

## 2. Module inventory (target)

```
sciknow/
├── infer/                    # NEW — llama-server lifecycle + clients
│   ├── server.py             # Subprocess manager: up/down/swap/health
│   ├── client.py             # /v1/chat /v1/embeddings /v1/rerank wrappers
│   ├── slots.py              # Slot accounting, prompt-cache hints
│   └── speculative.py        # Lucebox DFlash opt-in (draft+target swap)
├── ingestion/
│   ├── pipeline.py           # Stage machine — UNCHANGED in shape
│   ├── pdf/
│   │   ├── mineru_vlm.py     # Primary backend
│   │   ├── mineru_pipeline.py
│   │   └── marker.py         # Final fallback
│   ├── metadata.py           # 4-layer cascade (XMP→Crossref→arXiv→LLM)
│   ├── enrich.py             # Sources for db enrich (was enrich_sources.py)
│   ├── chunker.py
│   └── embedder.py           # Calls infer/client, NOT FlagEmbedding
├── retrieval/
│   ├── hybrid.py             # Dense + sparse + FTS, RRF
│   ├── rerank.py             # llama-server /v1/rerank only
│   ├── colbert.py            # Optional prefetch (kept; default off)
│   └── visuals.py            # CLIP-via-llama-server for figure search
├── rag/
│   ├── prompts.py            # Templates only — no I/O
│   └── llm.py                # Thin wrapper over infer/client (streaming)
├── core/
│   ├── events.py             # NEW — Pydantic schema for SSE event types
│   ├── book_ops.py           # Generators yielding events
│   ├── autowrite.py          # SPLIT OUT of book_ops — write/score/verify/cove/revise/rescore
│   ├── bibliography.py       # Global [N] renumbering
│   ├── citation_verify.py
│   ├── claim_atomize.py
│   ├── kg/                   # was kg_*.py + visuals_*.py
│   ├── wiki_ops.py
│   └── project.py            # Multi-project resolution
├── storage/
│   ├── models.py             # SQLAlchemy
│   ├── db.py
│   └── qdrant.py
├── observability/
│   ├── pulse.py              # ONE pulse format
│   ├── tracer.py             # Span tracing
│   └── alerts.py             # Alert codes (ACTIVE only — no deprecated)
├── web/
│   ├── app.py                # FastAPI mount points only (~1–2 kLOC)
│   ├── routes/               # One module per resource
│   ├── templates/            # Jinja2 partials
│   └── static/               # CSS, JS, fonts
├── cli/
│   ├── main.py               # Typer root
│   ├── corpus.py             # was: parts of cli/db.py (ingest/expand/enrich)
│   ├── library.py            # was: rest of cli/db.py (stats/reset/migrate)
│   ├── book.py
│   ├── search.py
│   ├── ask.py
│   ├── infer.py              # NEW — sciknow infer up/down/status/swap
│   └── project.py
└── testing/
    ├── protocol.py           # Layer harness
    ├── helpers.py            # get_test_client, etc.
    └── contracts/            # Schema tests (events, SSE wire, SQL)
```

## 3. llama.cpp / llama-server integration

### 3.1 One server, many endpoints

A single `llama-server` instance exposes (as of llama.cpp `b4xxx`):
- `POST /v1/chat/completions` — OpenAI-compatible streaming chat (used by `rag/llm.py`)
- `POST /v1/completions` — completion (used by autowrite "continuation" paths if any)
- `POST /v1/embeddings` — embedding (with a `--embedding` server or a separate one)
- `POST /v1/rerank` — cross-encoder rerank (llama.cpp added this; uses `bge-reranker-v2-m3-gguf`)
- `GET /health` — readiness
- `GET /props` — model id, ctx, slot count
- `POST /props` — runtime knobs (ROPE etc.)
- `GET /slots` — per-slot prompt-cache state (drives sciknow's pulse)

### 3.2 Multi-server topology

Three logical roles, three server slots:

| Role     | Default model                          | Port  | Memory       |
|----------|----------------------------------------|-------|--------------|
| writer   | Qwen3.5-27B-Instruct Q4_K_M            | 8090  | ~16 GB VRAM  |
| embedder | bge-m3 (gguf)                          | 8091  | ~2 GB VRAM   |
| reranker | bge-reranker-v2-m3 (gguf)              | 8092  | ~1 GB VRAM   |

- Default profile fits all three on a 24 GB card (RTX 3090) without eviction.
- A "low-vram" profile co-locates embedder+reranker on CPU; writer keeps the GPU.
- A "spec-dec" profile attaches a small draft model (Qwen3-1.7B) to the writer for speculative decoding (Lucebox DFlash; see `docs/research/speculative_decoding.md`).

`sciknow infer up [--profile default|low-vram|spec-dec]` brings the whole stack up with a `tmuxp`-style session manager (or systemd user services if `--systemd`). State lives in `data/infer/` (PIDs, logs, slot snapshots).

### 3.3 What this kills

| v1 module / mechanism             | v2 status   | Reason                                                        |
|-----------------------------------|-------------|---------------------------------------------------------------|
| `core/gpu_ledger.py`              | RETIRED     | Single VRAM owner — no in-process loads to track              |
| `core/vram_budget.py`             | RETIRED     | Same                                                          |
| `_release_gpu_models()` registry  | RETIRED     | Same                                                          |
| `retrieval/device.py` headroom    | RETIRED     | Embedder/reranker are server-side; no local CUDA detection    |
| `FlagEmbedding` dependency        | DROPPED     | Replaced by `/v1/embeddings`                                  |
| `sentence-transformers` (rerank)  | DROPPED     | Replaced by `/v1/rerank`                                      |
| `ollama` Python client            | DROPPED     | Replaced by `httpx` against llama-server's OpenAI API         |
| Dual-embedder dim mismatch errors | DROPPED     | One canonical embedder                                        |
| Autowrite VRAM-eviction phase     | DROPPED     | Server keeps writer warm forever                              |
| `OLLAMA_KEEP_ALIVE` tuning        | DROPPED     | llama-server holds models indefinitely                        |

### 3.4 Speculative decoding

Opt-in. When `--profile spec-dec` is used:
- Writer = Qwen3.5-27B Q4_K_M
- Draft = Qwen3-1.7B Q4_K_M (compatible tokenizer)
- llama-server flags: `--draft-model <draft.gguf> --draft-max 16 --draft-min 2`

Expected speedup on this workload (autowrite section drafting, ~2k–4k token outputs): 1.6×–2.1× on a 3090 per the speculative-decoding memo. The default profile is **non-speculative**; spec-dec is a per-session opt-in until empirical results land in `docs/benchmarks/BENCHMARKS.md`.

### 3.5 Prompt cache

llama-server keeps per-slot KV-cache automatically. sciknow exploits two patterns:

- **Stable system prompts.** Every prompt template starts with a long, frozen system block (prompt schema, taxonomy lists). This block is identical across calls, so KV-cache hits are >95%.
- **Batched CoV answers.** v1 phase 54.6.319 already batched CoV answers into a single call to amortise prefill. v2 keeps this and extends to scoring (one call per draft, all rubric dimensions in one structured response).

## 4. Storage schema (target)

### 4.1 Postgres (24 tables, was 27)

Tables retired in v2:
- `paper_institutions` (rolled into `paper_metadata` JSON)
- `autowrite_lesson` (replaced by a single `lessons` table indexed by section_type + topic_hash)
- `pending_download` (state moves to filesystem queue under `data/downloads/queue/`)

Tables added:
- `infer_event` — append-only log of llama-server slot events for debugging (rotates after 7 days).

Schema-level decisions:
- All `custom_metadata JSONB` columns get a CHECK constraint enforcing `jsonb_typeof = 'object'` at the DB level (catches the v1 string-vs-object bug at write time).
- `is_active` on `drafts` becomes a real `BOOLEAN` column (was `custom_metadata.is_active`) with a partial-unique index ensuring at most one active version per `(chapter_id, section_type)`. Eliminates the v1 three-tier active-version pick.
- `documents.converter_backend` and `documents.converter_version` are required (not nullable) — every backend write must stamp them.

### 4.2 Qdrant

| Collection      | Vectors                                   | Shape                  |
|-----------------|-------------------------------------------|------------------------|
| `papers`        | bge-m3 dense (1024) + bge-m3 sparse       | on-disk, HNSW M=16     |
| `abstracts`     | bge-m3 dense (1024)                       | in-memory              |
| `wiki`          | bge-m3 dense                              | on-disk                |
| `visuals`       | CLIP ViT-L/14 (768)                       | on-disk                |

The v1 sidecar collection (`papers_qwen3e4b`) and dual-vector configs are removed at v2 init time. No A/B columns in Postgres referencing it.

### 4.3 Filesystem layout

```
projects/<slug>/
├── .active                 # marker
├── data/
│   ├── downloads/          # PDFs in flight + queue/
│   ├── processed/          # MinerU/Marker outputs per doc_id
│   ├── failed/             # copies for forensics
│   ├── snapshots/          # backup tarballs
│   ├── infer/              # llama-server PIDs, logs, slot snapshots
│   └── sciknow.log         # rotated daily
└── pyproject.local.toml    # per-project overrides (rare)
```

## 5. Public surfaces

### 5.1 CLI

Top-level subapps (in `cli/main.py`):

| Subapp     | Verbs                                                              |
|------------|--------------------------------------------------------------------|
| `corpus`   | `ingest`, `expand`, `enrich`, `cluster`                            |
| `library`  | `init`, `reset`, `stats`, `migrate`, `validate`, `snapshot`        |
| `book`     | `new`, `outline`, `write`, `autowrite`, `argue`, `gaps`, `serve`   |
| `search`   | (single command)                                                   |
| `ask`      | `question`, `synthesize`, `write`                                  |
| `infer`    | `up`, `down`, `status`, `swap`, `logs`                             |
| `project`  | `init`, `use`, `list`, `show`, `archive`, `unarchive`, `destroy`   |
| `feedback` | `add`, `list`, `export`                                            |

Retired top-level subapp: `db` (renamed `library` with surgically removed expand/enrich/ingest verbs that move to `corpus`).

`sciknow web` does NOT exist (per memory note). Web reader stays at `sciknow book serve [--port 8000]`.

### 5.2 Web

- FastAPI app in `web/app.py` (mount points only, target ≤2 kLOC).
- Routes split by resource under `web/routes/`: `books.py`, `drafts.py`, `papers.py`, `wiki.py`, `kg.py`, `jobs.py`, `admin.py`.
- All HTML in Jinja2 partials under `web/templates/` (one file per page or panel; never inline f-strings >100 lines).
- All CSS in `web/static/css/sciknow.css` (one file). No `<style>` blocks in templates.
- All JS in `web/static/js/<module>.js` per concern (autowrite-stream.js, version-manager.js, popovers.js, etc.). No `<script>` blocks except a single bootstrap line per page.
- SSE endpoints unchanged in shape: `GET /api/stream/{job_id}` emits the v2 event schema.
- `GET /api/jobs/{id}/stats` is the canonical pulse endpoint (the v1 32.5-era fix is enshrined as the only way to drive the task bar).

### 5.3 SSE event schema (v2)

`sciknow/core/events.py` defines a tagged-union:

```python
class TokenEvent(BaseModel):
    type: Literal["token"]; job_id: str; text: str; phase: str

class ProgressEvent(BaseModel):
    type: Literal["progress"]; job_id: str; phase: str
    pct: float; tokens: int; eta_seconds: float | None

class ScoresEvent(BaseModel):     # rubric scores from scorer
    type: Literal["scores"]; job_id: str; iteration: int
    grounded: float; coherent: float; complete: float; concise: float
    rationale: str

class VerificationEvent(BaseModel):
    type: Literal["verification"]; job_id: str; iteration: int
    grounded: float; cove_decisions: list[CoveDecision]

class CompletedEvent(BaseModel):
    type: Literal["completed"]; job_id: str; draft_id: str | None

class ErrorEvent(BaseModel):
    type: Literal["error"]; job_id: str; code: str; message: str

# … 10 more — see code
SciknowEvent = Annotated[Union[TokenEvent, ProgressEvent, ...], Field(discriminator="type")]
```

This is the contract. Tests assert that `book_ops` generators only yield instances matching the union, and that the SSE wire format round-trips through `model_dump_json()`.

### 5.4 MCP server

`sciknow/mcp_server.py` exposes 8 tools (was 14): `search_papers`, `read_paper`, `list_books`, `read_chapter`, `read_draft`, `start_autowrite`, `cancel_job`, `job_status`. Drop the v1 paths that wrapped admin commands (`reset_db`, `release_vram`, etc.) — those are CLI-only.

## 6. Testing strategy

### 6.1 Layers (preserved from v1)

- **L1** — pure-Python, no services. Runs in <10s. Schema/contract tests.
- **L2** — Postgres + Qdrant up. Smoke ingest, smoke retrieval. ~30s.
- **L3** — full stack (llama-server up). One write+score+verify cycle on a 4-paper toy corpus. ~3 minutes.
- **SMOKE** — the existing wrapper; CI default.

### 6.2 What v2 drops

The v1 protocol has 61 L1 tests; ~20 are source-grep regressions for specific phase fixes (`l1_phase32_5_task_bar_polls_stats_no_sse_competition`, `l1_phase54_6_323_retrieval_device_revert_to_free_headroom`, etc.). These are kept *only* if the underlying behaviour is still load-bearing in v2. The audit happens during Phase F of the roadmap.

### 6.3 What v2 adds

- **Contract tests** (`testing/contracts/events.py`): assert every `core/book_ops` generator yields union-conforming events; assert SSE `/api/stream/{id}` produces parseable lines.
- **Schema tests**: assert SQLAlchemy models match Alembic head; assert Qdrant collection configs match what `library init` would create.
- **Wire tests**: assert llama-server `/v1/chat` request/response shape against a recorded fixture (so a llama.cpp upgrade that breaks shape fails L1, not L3).

## 7. Configuration

`.env` shrinks. Removed keys:
- `OLLAMA_HOST`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_NUM_GPU`, `OLLAMA_NUM_THREAD`
- `EMBEDDING_BATCH_SIZE` (server-side concern now)
- `SCIKNOW_RETRIEVAL_DEVICE` (no in-process loads)
- `MINERU_VLM_BACKEND` swapped to a profile name in `mineru_profile`
- All A/B sidecar embedder keys

Added:
- `INFER_WRITER_URL`, `INFER_EMBEDDER_URL`, `INFER_RERANKER_URL` (default `http://127.0.0.1:809{0,1,2}`)
- `INFER_PROFILE` (default | low-vram | spec-dec)
- `WRITER_MODEL_GGUF` (path or HF id; default `Qwen3.5-27B-Instruct-Q4_K_M.gguf`)

`sciknow/config.py` keeps Pydantic Settings; the model just shrinks.

## 8. Migration semantics

v2 is a fresh install path, not an in-place upgrade. The `last-ollama-build` tag (commit `cf91386`) is the v1 frozen state. v2 installs *alongside* v1 by default:

- v1 stays on `main`, `last-ollama-build` tag, branch `v1-maintenance` (cherry-pick critical fixes only).
- v2 develops on branch `v2-llamacpp` until it cuts a release, then merges to `main`.
- An `sciknow project import-v1 <slug>` command (Phase F) reads a v1 project's Postgres + Qdrant + filesystem and writes it into a v2 project, doing the dual-embedder→single-embedder reduction at import time (re-embeds with the canonical model).

Alembic migration history starts fresh in v2 (`migrations/v2/versions/`), squashing v1's 40 phased migrations into a single canonical schema migration plus per-feature additions.

## 9. Open questions

These are NOT decided in the spec; they go to the roadmap as research items:

1. **Single-server vs three-server topology.** Three is the default in §3.2; a single-server with model-swap-on-demand is simpler but loses prompt-cache between roles. Decide after benchmarking.
2. **Visuals embedding model.** v1 uses a CLIP-shaped image embedding with payload-side text fallback. llama-server doesn't natively serve CLIP; v2 may need a CLIP sidecar (one allowed exception, like MinerU) or migrate to Qwen2.5-VL via llama.cpp's mmproj pathway when stable.
3. **DFlash availability.** The Lucebox DFlash kernel claims 129–207 t/s on a 3090 but is not yet upstream in llama.cpp `master`. Track upstream; if it stalls past 2026-Q3, fall back to vanilla speculative decoding.
4. **Web reader rebuild order.** Externalising the 31 kLOC f-string template is a multi-PR project; whether to do it before or after the inference swap is a sequencing call (roadmap leans "after" — get the engine right first).

## 10. Rejected alternatives

- **vLLM** as the writer backend. Faster per-token but: (a) requires a beefier GPU for non-trivial concurrency, (b) doesn't natively expose `/v1/rerank`, (c) installation footprint dwarfs llama.cpp. Rejected per `docs/research/inference_servers.md`.
- **Keeping Ollama and adding llama-server alongside.** Two backends, two prompt caches, two model registries. v1 already lives this — v2 explicitly rejects it.
- **Dropping the multi-project layout.** It's earned its keep on disk-space terms (one PG cluster, N projects) and on the user's actual workflow. Keep.
- **Replacing Postgres with SQLite.** SQLite can't do the trigram + tsvector + JSONB combo cleanly, and the v1 ingestion pipeline depends on Postgres advisory locks for the stage machine. Keep Postgres.
- **Building a JS bundler.** Server-rendered + plain JS modules is enough. No npm.

---

See `docs/roadmap/SCIKNOW_V2_ROADMAP.md` for the phased migration plan.
