# MIGRATION.md — sciknow v1 → v2

This document maps every v1 CLI verb / settings key / module path to
its v2 equivalent. The changes land incrementally on the `v2-llamacpp`
branch and merge to `main` as part of Phase G (cutover).

## TL;DR

```bash
# v1                                         v2 equivalent
sciknow db init                              sciknow library init
sciknow db reset                             sciknow library reset
sciknow db stats                             sciknow library stats
sciknow db backup                            sciknow library backup
sciknow db doctor                            sciknow library doctor
sciknow db monitor                           sciknow library monitor
sciknow db expand                            sciknow corpus expand
sciknow db enrich                            sciknow corpus enrich
sciknow db refresh-metadata                  sciknow corpus refresh-metadata
sciknow db repair                            sciknow corpus repair
sciknow db dedup                             sciknow corpus dedup
sciknow db extract-visuals                   sciknow corpus extract-visuals
sciknow db caption-visuals                   sciknow corpus caption-visuals
# ... every other db verb has the same library/corpus shape
```

The v1 `sciknow db <verb>` namespace remains mounted in v2.0 with a
one-shot deprecation warning per process. It will be removed in v2.1.
Update muscle memory now.

## Subapp restructure (Phase F)

| Old subapp        | New v2 home          | Notes                                      |
|-------------------|----------------------|--------------------------------------------|
| `sciknow db`      | split into two       | renamed; deprecated 1 release, removed v2.1 |
| `sciknow db init` | `sciknow library init` | infra lifecycle                          |
| `sciknow db reset` | `sciknow library reset` |                                         |
| `sciknow db stats` | `sciknow library stats` |                                         |
| `sciknow db backup` | `sciknow library backup` |                                       |
| `sciknow db restore` | `sciknow library restore` |                                     |
| `sciknow db failures` | `sciknow library failures` |                                    |
| `sciknow db doctor` | `sciknow library doctor` |                                       |
| `sciknow db monitor` | `sciknow library monitor` |                                     |
| `sciknow db dashboard` | `sciknow library dashboard` |                                 |
| `sciknow db drift` | `sciknow library drift` |                                          |
| `sciknow db provenance` | `sciknow library provenance` |                              |
| `sciknow db audit-sidecar` | `sciknow library audit-sidecar` |                       |
| **(new)** | `sciknow library migrate` | shorthand for `uv run alembic upgrade head` |
| **(new)** | `sciknow library validate` | shorthand for `alembic check`              |
| **(new)** | `sciknow library snapshot` | stable name for the backup tarball         |
| `sciknow db expand` | `sciknow corpus expand` | corpus growth                          |
| `sciknow db enrich` | `sciknow corpus enrich` |                                        |
| `sciknow db refresh-metadata` | `sciknow corpus refresh-metadata` |                  |
| `sciknow db refresh-retractions` | `sciknow corpus refresh-retractions` |            |
| `sciknow db cleanup-downloads` | `sciknow corpus cleanup-downloads` |               |
| `sciknow db reconcile-preprints` | `sciknow corpus reconcile-preprints` |           |
| `sciknow db reconciliations` | `sciknow corpus reconciliations` |                   |
| `sciknow db unreconcile` | `sciknow corpus unreconcile` |                           |
| `sciknow db repair` | `sciknow corpus repair` |                                        |
| `sciknow db dedup` | `sciknow corpus dedup` |                                          |
| `sciknow db reclassify-sections` | `sciknow corpus reclassify-sections` |           |
| `sciknow db link-citations` | `sciknow corpus link-citations` |                     |
| `sciknow db classify-papers` | `sciknow corpus classify-papers` |                   |
| `sciknow db flag-self-citations` | `sciknow corpus flag-self-citations` |           |
| `sciknow db sync-dense-sidecar` | `sciknow corpus sync-dense-sidecar` |              |
| `sciknow db expand-author` | `sciknow corpus expand-author` |                       |
| `sciknow db expand-author-refs` | `sciknow corpus expand-author-refs` |            |
| `sciknow db expand-cites` | `sciknow corpus expand-cites` |                         |
| `sciknow db expand-topic` | `sciknow corpus expand-topic` |                         |
| `sciknow db expand-coauthors` | `sciknow corpus expand-coauthors` |                 |
| `sciknow db expand-inbound` | `sciknow corpus expand-inbound` |                     |
| `sciknow db expand-oeuvre` | `sciknow corpus expand-oeuvre` |                       |
| `sciknow db extract-visuals` | `sciknow corpus extract-visuals` |                   |
| `sciknow db link-visual-mentions` | `sciknow corpus link-visual-mentions` |          |
| `sciknow db caption-visuals` | `sciknow corpus caption-visuals` |                   |
| `sciknow db embed-visuals` | `sciknow corpus embed-visuals` |                       |
| `sciknow db paraphrase-equations` | `sciknow corpus paraphrase-equations` |          |
| `sciknow db parse-tables` | `sciknow corpus parse-tables` |                         |
| `sciknow db download-dois` | `sciknow corpus download-dois` |                       |
| `sciknow ingest`  | `sciknow corpus ingest` (also still at top level for one release) | |
| `sciknow catalog cluster …` | `sciknow corpus cluster …`                            |
| **(new)** | `sciknow infer up\|down\|status\|swap\|logs` | manage llama-server roles |

`sciknow book serve` (web reader entry) is **unchanged** — it was
never `sciknow web` and still isn't.

## Inference backend (Phases A + B)

The biggest behavioural change in v2 is the inference substrate.

| v1 (Ollama-resident)                              | v2 (llama-server)                                      |
|---------------------------------------------------|--------------------------------------------------------|
| `OLLAMA_HOST=http://localhost:11434`              | `INFER_WRITER_URL=http://127.0.0.1:8090`               |
| `OLLAMA_KEEP_ALIVE=…` env                         | retired — llama-server holds models for process life   |
| `OLLAMA_NUM_GPU` / `OLLAMA_NUM_THREAD`            | retired — pass via `LLAMA_SERVER_*` flags / profiles    |
| `EMBEDDING_BATCH_SIZE` env                         | server-side concern; keep for ingest tuning            |
| `LLM_MODEL=qwen3.6:27b-dense` (Ollama tag)        | `WRITER_MODEL_GGUF=/path/to/Qwen3.6-27B-Q4_K_M.gguf`   |
| `EMBEDDING_MODEL=BAAI/bge-m3` (FlagEmbedding)     | served by `INFER_EMBEDDER_URL`; keep tag for labels    |
| `RERANKER_MODEL=BAAI/bge-reranker-v2-m3` (ST)     | served by `INFER_RERANKER_URL`; keep tag for labels    |
| `DENSE_EMBEDDER_MODEL=Qwen/Qwen3-Embedding-4B`    | retired — single canonical embedder per spec §2.1      |
| `DENSE_EMBEDDER_DIM`                              | retired                                                |
| `DENSE_SIDECAR_COLLECTION`                        | retired                                                |
| `SCIKNOW_RETRIEVAL_DEVICE` / device.py auto-pin   | retired — no in-process loads on the v2 default path   |

Three new toggles let v2 fall back to the v1 path if a regression
forces a rollback within a single commit:

| Toggle                          | Default | When True                                              |
|---------------------------------|---------|--------------------------------------------------------|
| `USE_LLAMACPP_WRITER`           | True    | `rag/llm.py` dispatches to `infer.client.chat_stream`  |
| `USE_LLAMACPP_EMBEDDER`         | True    | bge-m3 served via `/v1/embeddings`; no FlagEmbedding   |
| `USE_LLAMACPP_RERANKER`         | True    | reranker via `/v1/rerank`; no FlagReranker / ST adapter |

Set any to `False` to fall back to the v1 in-process path for that
role (`uv add ollama` / `FlagEmbedding` first if you removed them
during a Phase B cleanup).

Bring up the substrate with:

```bash
sciknow infer up --role writer    # :8090 — Qwen3.6-27B Q4_K_M
sciknow infer up --role embedder  # :8091 — bge-m3 GGUF
sciknow infer up --role reranker  # :8092 — bge-reranker-v2-m3 GGUF
sciknow infer status              # health + PID + model
```

`sciknow infer up --role all --profile <default|low-vram|spec-dec>`
once profiles are wired into the CLI in a follow-up.

## Storage schema (Phase C)

| v1 path                                         | v2 path                                       |
|-------------------------------------------------|-----------------------------------------------|
| `drafts.custom_metadata->>'is_active' = 'true'` | `drafts.is_active BOOLEAN` (nullable=False)  |
| three-tier active-version picker (active flag → highest-with-content → first) | `is_active = TRUE` partial-unique on `(chapter_id, section_type)`, with the v1 fallback rule applied at backfill time |

Migration `0040_drafts_is_active.py` handles the data transition;
`alembic downgrade -1` restores the JSON marker before dropping the
column so v1 readers can be re-pointed at the same DB.

## Event protocol (Phase C)

`sciknow.core.events` defines a Pydantic discriminated union spanning
all 42 event tags emitted by `book_ops` + `wiki_ops`. The SSE wire
format remains identical (`{"type": "<tag>", ...}`); the union is
permissive (`extra="allow"`) so call sites can migrate from raw dicts
to typed instances incrementally.

Adding a new event tag requires:

1. Define a `<Name>Event(_BaseEvent)` class in `core/events.py`.
2. Add it to `SciknowEvent` Union and `KNOWN_EVENT_TYPES` set.
3. Add it to `__all__`.
4. The L1 contract test `l1_v2_events_schema_covers_known_yields`
   asserts both ends stay in sync.

## Module layout

The v2 spec §2 lays out the target module layout. As of the current
branch:

| Module                  | v1 status | v2 status                                  |
|-------------------------|-----------|--------------------------------------------|
| `sciknow/infer/`        | n/a       | NEW — server, client, (slots/speculative deferred) |
| `sciknow/cli/infer.py`  | n/a       | NEW — up/down/status/swap/logs             |
| `sciknow/cli/library.py`| n/a       | NEW — see Phase F table                    |
| `sciknow/cli/corpus.py` | n/a       | NEW                                        |
| `sciknow/core/events.py`| n/a       | NEW — Pydantic union + KNOWN_EVENT_TYPES   |
| `sciknow/cli/db.py`     | 11.5 kLOC | retained as deprecation shim source; verb bodies still live here for one release |
| `sciknow/core/book_ops.py` | 6.7 kLOC | unsplit; v2 split into book_ops + autowrite is a follow-up commit |
| `sciknow/web/app.py`    | 31.6 kLOC | unsplit; Phase E externalises CSS / JS / templates incrementally |
| `sciknow/retrieval/device.py` | 156 LOC | no-op when llamacpp toggle is on; deletion deferred until v2.1 |
| `sciknow/core/gpu_ledger.py` / `vram_budget.py` | 218 + 317 LOC | no-op cascades on the v2 path; deletion deferred until v2.1 |

## Status by phase

- **Phase A** (infer substrate): ✅ shipped (`22db97d`)
- **Phase B** (embedder + reranker on llama-server): ✅ shipped + exit criteria met (`2058d1e`, `b6b91ad`, `e9878c6` — FlagEmbedding/sentence-transformers/ollama dropped from direct deps)
- **Phase C** (autowrite simplification): 🟡 events.py + is_active migration shipped (`8437db2`); 6.7 kLOC book_ops split is the remaining sub-task
- **Phase D** (retrieval cleanup): ✅ functionally shipped (`b6b91ad`) — single canonical embedder enforced, sidecar collection bypassed at query time + dropped by `library upgrade-v1`. Legacy code paths still present behind toggles for rollback.
- **Phase E** (web rebuild): 🟡 main CSS block extracted to `static/css/sciknow.css` (`8710b7d`); 16,394-line inline `<script>` extracted to `static/js/sciknow.js` with a 7-key `window.SCIKNOW_BOOTSTRAP` config emitted inline by the template. `web/app.py` shrunk 29 kLOC → 12.7 kLOC (-56%). Jinja2 template extraction + `web/routes/` split deferred to v2.1.
- **Phase F** (CLI reorg): ✅ shipped — `library` + `corpus` subapps live with deprecation shim (`027d09b`); MIGRATION.md (this file) covers every verb; the L1 + alert audit is deferred to v2.1.
- **Phase G** (v1 import + cutover): 🟡 in-place migrator `library upgrade-v1` shipped (`e9878c6`); cross-project `project import-v1 <slug> --as <slug>` deferred — single-tenant installs don't need it.

L1: 266/266 green across the v2-llamacpp branch.
