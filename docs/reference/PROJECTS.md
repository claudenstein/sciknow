# Multi-Project Support — Design & Roadmap

Status: **Shipped (Phase 43a–h).** The design below is the plan that landed.
Phase 43a–d delivered the plumbing; 43e added `sciknow project …`
subcommands; 43f added the one-shot legacy-layout migration; 43g
added the root `--project` flag; 43h added the GUI project picker.

## Command surface (cheat sheet)

```bash
sciknow project init <slug>                     # fresh empty project
sciknow project init <slug> --from-existing     # adopt legacy install
sciknow project init <slug> --dry-run           # preview without changes
sciknow project list                            # all projects + active marker
sciknow project show [slug]                     # details (defaults to active)
sciknow project use <slug>                      # set .active-project
sciknow project destroy <slug> [--yes]          # drop DB + collections + dir
sciknow project archive <slug> [-o path]        # bundle to .skproj.tar, drop live
sciknow project archive <slug> --keep-live      # snapshot only, keep live state
sciknow project unarchive <archive-file>        # restore from archive
sciknow --project <slug> <any subcommand>       # one-shot override
```

The web reader exposes list / show / use / init / destroy through the
Projects modal (&#128193; button in the action toolbar).

---

## Goal

Convert sciknow from single-tenant to multi-project. Each project has
its own corpus (papers, metadata, chunks, embeddings), its own book(s),
and its own autowrite telemetry. Projects are fully isolated —
documents in one project are invisible to another — so a user can
maintain (say) a "Global Cooling" climate book and a "Materials Science"
book without cross-contamination.

Current state: the first project will be `global-cooling` (migrated
from the existing `sciknow` database / `data/` directory). New projects
get created with `sciknow project init <name>`.

---

## The sharing policy

After inventorying every global assumption in the code (see §"What's
single-tenant today" below), the right split is:

### Shared across projects (global / host-level)

Things that don't belong to a project:

- **Python environment** (`.venv/`) — one sciknow install serves all projects.
- **Infrastructure endpoints** — PostgreSQL host/port, Qdrant host/port,
  Ollama host. Live in the root `.env`.
- **Model weights** — `~/.cache/modelscope/` (MinerU), `~/.cache/datalab/`
  (Marker), `~/.cache/huggingface/` (bge-m3, bge-reranker-v2-m3). System-
  cache, not project state.
- **Crossref email** — same user identity for the polite pool regardless
  of project.
- **Default model names** (`LLM_MODEL`, `LLM_FAST_MODEL`, `EMBEDDING_MODEL`).
  The per-section override (Phase 37) can already tune this per book;
  making it per-project would add a layer with no user-visible payoff.

### Per-project (isolated)

Each project owns:

- **A PostgreSQL database**: `sciknow_<project_slug>`.
  All of `documents`, `paper_metadata`, `chunks`, `citations`, `books`,
  `book_chapters`, `drafts`, `autowrite_*`, `wiki_pages`,
  `knowledge_graph`, `llm_usage_log`, `draft_snapshots` live in this DB.
- **Qdrant collections**: `<slug>_papers`, `<slug>_abstracts`,
  `<slug>_wiki`.
- **A data directory**: `projects/<slug>/data/` with subdirs
  `downloads/`, `processed/`, `failed/`, `mineru_output/`, `wiki/`,
  `autowrite/`, plus `sciknow.log`.
- **An optional per-project `.env.overlay`** for project-specific overrides
  (different `LLM_MODEL` for a different domain, etc.).

### Deliberately not shared (by design)

A user might ask *"why can't two projects share a corpus?"* Answer:
because corpus = bias (`docs/roadmap/STRATEGY.md` §1). Two books drawing from
the same paper pool produce two books reflecting the same selection
bias. If you really want a shared corpus you can export from one
project and import to another, but the default is isolation.

---

## What's single-tenant today

From the architecture audit (condensed; full file:line citations live
in the planning notes):

| Where | What's hardcoded | Consequences |
|---|---|---|
| `config.py:22` | `pg_database = "sciknow"` | Every session connects to the same DB |
| `config.py:15` | `data_dir = Path("data")` | All downloads/ mineru_output/ autowrite/ live in one tree |
| `storage/db.py:10-15` | Module-level `engine` + `SessionLocal` | `get_session()` is global; one engine per process |
| `storage/qdrant.py:17-19` | `PAPERS_COLLECTION = "papers"` (and two more) | 60+ call sites reference the constants directly |
| `storage/qdrant.py:25-32` | Module-level `_client` singleton | `get_client()` has no project parameter |
| `migrations/env.py:10` | `settings.pg_url` → Alembic target | Migrations always run against one DB |
| `web/app.py:60-62` | Module-level `_book_id`, `_book_title` | `book serve` assumes one book per uvicorn |
| `cli/db.py` | `Path("data/downloads")` hardcoded (4 sites) | Not parameterized through `settings` |
| `cli/main.py:20-45` | No project concept | Every subcommand implicitly uses the global singleton |

---

## Project resolution — how the CLI decides which project is active

Three mechanisms, in precedence order:

1. **`--project <slug>`** command-line flag. Overrides everything.
   Useful for scripts and explicit single-command usage.
2. **`SCIKNOW_PROJECT=<slug>`** environment variable. Useful for
   per-terminal sessions.
3. **`.active-project`** file at repo root containing the slug.
   Stateful — set by `sciknow project use <slug>`. This is the
   default path for interactive work.

If none are set and no projects exist: the CLI prints a friendly
pointer to `sciknow project init <slug>` and exits non-zero.

If none are set but projects exist: the CLI picks the alphabetically
first project and warns — *"no active project set; using X. Run
`sciknow project use <slug>` to pick a different one."*

---

## New CLI: `sciknow project`

```
sciknow project init <slug> [--from-existing]   # create a new project
sciknow project list                            # show all projects + status
sciknow project show [slug]                     # details: DB, collections, counts, data dir
sciknow project use <slug>                      # set .active-project
sciknow project destroy <slug>                  # drop DB + collections + data dir (guarded)
```

`--from-existing` on `init` is the **one-shot migration path** — creates
the new project slot, moves the current `data/*` into
`projects/<slug>/data/*`, renames the PG database `sciknow` →
`sciknow_<slug>`, and renames Qdrant collections `papers` →
`<slug>_papers` etc. Idempotent and reversible (we snapshot the Qdrant
manifest + take a PG dump before renaming).

---

## Roadmap

Phases are ordered to keep the system runnable at every step. Don't
skip ahead.

### Phase 43a — Config refactor (non-breaking)

**Goal:** settings become project-aware but the default behaviour is
unchanged if no project is set.

- Introduce `sciknow.core.project` module: `Project` dataclass
  (slug, pg_database, qdrant_prefix, data_dir, root path).
- `get_active_project() -> Project` — reads `--project` flag → env var
  → `.active-project` file → default.
- `config.settings` gains:
  - `project: Project` (the currently-active one)
  - `pg_database`, `data_dir`, `qdrant_papers_collection`, etc. become
    `@property` delegates that return the active project's values.
- Root `.env` stays; per-project `.env.overlay` read and layered on top.
- Fallback: if nothing's configured yet, treat as if a legacy "default"
  project is active and use today's names (`sciknow`, `data/`, `papers`).
  Nothing breaks on first run of the new code.

### Phase 43b — PostgreSQL project-awareness

**Goal:** every session connects to the active project's DB.

- `storage/db.py`: replace module-level `engine` with a lazy
  `get_engine() -> Engine` keyed on `settings.pg_url`. Cached, so
  long-running processes reuse the connection pool.
- `get_session()` reads from the cache.
- Alembic `env.py` reads from `get_active_project().pg_url` at runtime.
- `alembic revision --autogenerate` keeps working; `alembic upgrade head`
  targets the active project's DB.
- L2 test that a fresh project can run `db init` end-to-end.

### Phase 43c — Qdrant project-awareness

**Goal:** collection names are derived, not hardcoded.

- `storage/qdrant.py`: replace the three module constants with
  `papers_collection()`, `abstracts_collection()`, `wiki_collection()`
  functions that read `settings.project.qdrant_prefix`.
- `get_client()` stays global (one Qdrant instance serves all projects;
  only collection names differ).
- Mechanical refactor at all 60+ call sites: replace
  `PAPERS_COLLECTION` with `papers_collection()`. `storage/qdrant.py`
  keeps the old constants as deprecation aliases during the transition.
- `init_collections()` creates the active project's collections.

### Phase 43d — Filesystem paths

**Goal:** all of `data/*` becomes `projects/<slug>/data/*`, parameterised.

- `config.settings.data_dir` returns `<repo>/projects/<slug>/data`.
- Hardcoded `Path("data/downloads")` sites in `cli/db.py` and
  `cli/book.py` route through `settings.data_dir`.
- `setup_logging()` gets its log dir from `settings.data_dir`.

### Phase 43e — `sciknow project` subcommand

**Goal:** the user-facing entry point.

- New Typer subapp at `sciknow/cli/project.py`:
  `init`, `list`, `show`, `use`, `destroy`.
- `init <slug>` creates the empty shell (DB + collections + data dir),
  runs migrations, initializes an empty `projects/<slug>/.env.overlay`.
- `init <slug> --from-existing` is the migration path (below).

### Phase 43f — One-shot migration of the current deployment

**Goal:** move the existing `sciknow` DB + `data/` tree into a project
slot without data loss.

The `sciknow project init global-cooling --from-existing` command:

1. **Guard:** refuse if `projects/global-cooling/` already exists.
2. **PG dump:** `pg_dump sciknow > projects/global-cooling/data/_migration.dump`.
3. **Create new DB:** `CREATE DATABASE sciknow_global_cooling WITH TEMPLATE sciknow`.
   (Template creates a structural + data clone in one go — faster and
   atomic vs dump+restore.)
4. **Rename Qdrant collections:** Qdrant doesn't support in-place
   rename, so we snapshot + recreate:
   `papers` → snapshot → restore as `global_cooling_papers` → delete
   original. Same for `abstracts`, `wiki`.
5. **Move filesystem:** `mv data/* projects/global-cooling/data/`.
6. **Set `.active-project`:** `echo global-cooling > .active-project`.
7. **Verify:** run a read-only smoke test (`db stats`) against the new
   project. If it matches the pre-migration stats, the migration is
   green.
8. **Offer cleanup:** prompt to drop the old `sciknow` DB once verified.
   Default is to keep it as a recovery path.

Each step is idempotent so re-running after a failure resumes where it
left off. A `--dry-run` flag prints the plan without executing.

### Phase 43g — Web reader

**Goal:** `book serve` uses the active project.

- `web/app.py`'s global `_book_id` keeps its meaning (one book per
  uvicorn), but the book is looked up in the active project's DB.
- If the user runs `sciknow book serve` without a book title, the web
  reader shows a book picker listing the active project's books.
- Port stays 8000 by default. Running two projects simultaneously =
  two terminals with different ports (`PORT=8001 sciknow book serve ...`).

### Phase 43h — Tests + docs

- L1: project resolution (`get_active_project()` respects precedence).
- L2: project isolation (create two projects, ingest different papers,
  assert no cross-visibility).
- Update `CLAUDE.md`, `README.md`, `docs/INSTALLATION.md`,
  `docs/reference/OPERATIONS.md`.
- This doc stays as the design record.

---

## What migration does NOT do

- **Does not merge data across projects** — if you accidentally end up
  with two projects and want to consolidate, that's a future `sciknow
  project merge` command. Not in scope.
- **Does not share a Qdrant Cache** — the bge-m3 model serves all
  projects from the same process, but each project's vectors are
  physically separate collections. No tricks, no aliasing.
- **Does not shard `wiki_pages` or `knowledge_graph`** — they become
  per-project along with the rest of the DB. A paper cited in both
  projects produces two KG entries (one per project's `documents`
  table). Duplicated, but cheap and clean.

---

## Open questions the user should answer before we implement

1. **Slug validation rules.** Should project slugs be restricted to
   `[a-z0-9-]`? (Recommended — they become DB and collection names,
   which have stricter charsets than arbitrary user strings.)
2. **Default project name when `init` is called without `--from-existing`.**
   I'd suggest refusing to auto-create a default so users have to name
   their project up front. Explicitness beats magic.
3. **What happens to a web reader that's already running when the user
   switches projects?** Option A: it keeps serving the old project's
   book until restarted. Option B: SIGHUP-style reload. I'd go with A
   — web reader is one-shot, meant for a writing session.
4. **Do we want per-project `LLM_MODEL` overrides?** Current Phase 37
   already gives per-section overrides within a project; per-project
   defaults via `.env.overlay` would be additive, not substitutional.
   Decision: yes, support it via `.env.overlay`, but don't require it.
5. **Archival path.** Once a project is finished, should `project
   destroy` be the only way to remove it, or do we want `project
   archive <slug>` that tars up the data dir + pg_dumps the DB +
   deletes the live ones? I'd defer — archive can be a Phase 44.

---

## Estimated scope

- Phase 43a-d (config + DB + Qdrant + paths): ~1 day of careful work,
  half of it finding every hardcoded site and making sure the tests
  still run.
- Phase 43e-f (project subcommand + migration): ~half a day.
- Phase 43g (web reader): ~2 hours.
- Phase 43h (tests + docs): ~half a day.

Total: ~2-3 solid days. This is not a session-end refactor — it
touches every file in `storage/` and `cli/`, and the migration has to
be rock-solid because the alternative is data loss.

**Recommendation:** do it in one focused session (or a couple of them)
with the user actively testing after each phase, not as a background
project.
