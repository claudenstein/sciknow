# AGENTS.md

Orientation file for coding agents. `CLAUDE.md` at the repo root is the
authoritative deep-dive — read it first. This file only lists the
highest-signal facts and gotchas an agent would otherwise get wrong.

## Running the CLI

- The `sciknow` entry point lives in `.venv/bin/sciknow` and is **not**
  on PATH. Always prefix with `uv run`:
  `uv run sciknow <subcommand> …`
- Python deps are managed with `uv` against `pyproject.toml` + `uv.lock`.
  Use `uv sync` / `uv add <pkg>`. **Do not `pip install` into the venv.**
- The web reader is launched with `uv run sciknow book serve`, **not**
  `sciknow web`. Default bind is `127.0.0.1:8000`.
- Every CLI invocation logs to `projects/<slug>/data/sciknow.log`
  (or `data/sciknow.log` for the legacy `default` project).

## Testing

There is no pytest, ruff, or black wired up. The test runner is a
layered smoke harness:

```bash
uv run sciknow test                 # L1 only (~8s, no service deps) — run before every push
uv run sciknow test --layer L2      # needs Postgres + Qdrant
uv run sciknow test --layer L3      # adds Ollama
uv run sciknow test --layer SMOKE   # single-example wiki/autowrite LLM pipeline checks
uv run sciknow test --layer all
```

Add new L1 tests to `sciknow/testing/protocol.py` and import shared
helpers (`get_test_client`, `a_book_id`, `rendered_template_static`, …)
from `sciknow/testing/helpers.py` — don't re-derive boilerplate.
`sciknow bench` is a separate harness for perf/quality numbers; see
`docs/TESTING.md` and `docs/BENCHMARKS.md`.

## Runtime services

All native, no Docker. PostgreSQL 16 (`sciknow`/`sciknow`@localhost:5432),
Qdrant (user systemd unit, port 6333), Ollama (11434). Config loaded via
Pydantic Settings from `.env` (`sciknow/config.py`). Set `CROSSREF_EMAIL`
or metadata extraction breaks.

## Multi-project (Phase 43+)

Each project has its own PG database (`sciknow_<slug>`), Qdrant
collections (`<slug>_papers`, `<slug>_abstracts`), and `projects/<slug>/data/`.
Active project resolution: `--project <slug>` / `-P` (root flag on every
command) → `SCIKNOW_PROJECT` env → `.active-project` file → legacy
`default`.

**Trap (Phase 54.6.20):** a stale `PG_DATABASE=sciknow` left over in
`.env` will silently split writes — DB rows go to the `.env` value while
disk writes follow the active project. The Settings model logs a warning
when it overrides `.env`; if you see that warning, drop the key from
`.env`. Full lifecycle commands: `sciknow project init|list|use|show|destroy|archive|unarchive`.

## Destructive ops — confirm first

- `sciknow db reset` wipes PG + Qdrant + `data/processed` + `data/downloads` + `mineru_output`.
  **Never use it as a fix for a broken ingestion.** Resume is automatic
  — re-run the same `ingest` command and check `db stats`. `db reset`
  is only for deliberate rebuilds (e.g. switching embedding dims).
- `sciknow project destroy` drops a whole project's DB + collections +
  data dir. Confirm scope before running.

## Ingestion pipeline invariants

The state machine in `sciknow/ingestion/pipeline.py`:

```
pending → converting → metadata_extraction → chunking → embedding → complete | failed
```

- `documents` is keyed on SHA-256 of file bytes — re-running
  `ingest directory` is idempotent and resumes failed/partial papers.
  Use `--force` only for deliberate re-ingest.
- Three chunker entry points must stay in sync:
  `parse_sections_from_mineru`, `parse_sections_from_json` (Marker),
  `parse_sections` (markdown fallback). When you add a canonical
  section type, edit **all three** of `_SECTION_PATTERNS`,
  `_SKIP_SECTIONS`, `_PARAMS` in `sciknow/ingestion/chunker.py`.
- Changing `EMBEDDING_MODEL` / `EMBEDDING_DIM` requires `db init` to
  create fresh Qdrant collections (dim is set at collection creation).

## Migrations

Alembic, config in `alembic.ini`, versions under `migrations/versions/`
(currently 0001–0034). Standard flow:

```bash
uv run alembic revision -m "message"
uv run alembic upgrade head
uv run alembic downgrade -1
```

`sciknow db init` runs `alembic upgrade head` plus Qdrant collection
creation.

## CLI layout

`sciknow/cli/main.py` composes Typer subapps (`db`, `ingest`, `search`,
`ask`, `catalog`, `book`, `draft`, `wiki`, `project`, `watch`,
`feedback`, `spans`, `backup`) plus top-level commands (`test`, `bench`,
`mcp-serve`, `refresh`, `bench-*`). Add new commands to the matching
subapp module — do not create a new top-level unless it truly doesn't
fit any existing domain.

## Service-layer convention (book ops)

`sciknow/core/book_ops.py` contains generator-based functions for every
book operation (`write`, `review`, `revise`, `autowrite`, `argue`,
`gaps`, …). Each yields typed event dicts (`token`, `progress`,
`scores`, `verification`, `completed`, `error`). Both the CLI (Rich
console) and the web (SSE via `asyncio.Queue` in `sciknow/web/app.py`)
consume the same generators. New book operations go here first, then
get wired into CLI and web.

## Web reader landmine (Phase 32.5)

LLM jobs in `sciknow/web/app.py` push events onto a per-job
`asyncio.Queue`. **Never add a second `EventSource` to
`/api/stream/{id}` for a non-preview UI** — `Queue.get()` removes items,
so two consumers split the event stream and the token counter breaks.
For stats (token count, wall time) the persistent task bar polls
`GET /api/jobs/{id}/stats` every 500 ms. Server-side counters live on
each `_jobs[id]` entry and are updated by `_observe_event_for_stats()`
**before** events are enqueued. Regression gated by the L1 test
`l1_phase32_5_task_bar_polls_stats_no_sse_competition`.

## Known dependency trap

`opencv-python` and `opencv-python-headless` both install to the same
`cv2/` directory and clobber each other's files. `pyproject.toml` has a
`[tool.uv] override-dependencies` entry that forces headless to be
unresolved. If `uv sync` ever leaves cv2 broken
(`module 'cv2' has no attribute 'imread'` during MinerU):

```bash
uv pip uninstall -y opencv-python opencv-python-headless
uv pip install --force-reinstall --no-deps opencv-python
```

## Shell quoting gotcha

`uv run sciknow ingest directory <path>` with unquoted shell
metacharacters (`&`, spaces, `$`, `(`, `)`, `;`, `*`) will silently
process a truncated path and report zero docs. Always quote paths
passed to ingest commands.

## Do NOT

- Invent pytest/ruff/black commands — they are not wired up.
- Hardcode model names, ports, paths, or Ollama host — read from
  `sciknow/config.py` settings (loaded from `.env`).
- Treat Qdrant payload as source of truth for bibliographic data —
  always join back to Postgres (`chunks.qdrant_point_id` is the join key).
- Backfill date fields on `documents` / `paper_metadata` from
  filesystem mtime — they track ingestion lineage.

## Further reading

- `CLAUDE.md` — full architecture and conventions (read first).
- `README.md` — user-facing feature tour + quick start.
- `docs/TESTING.md` — testing protocol, adding checks.
- `docs/WORKFLOW.md` — zero-to-book command sequence.
- `docs/PROJECTS.md` — multi-project design.
- `docs/BENCHMARKS.md` — `sciknow bench` layers.
- `OPTIMIZATION.md` — GPU/model split plans (RTX 3090 + incoming DGX Spark).
