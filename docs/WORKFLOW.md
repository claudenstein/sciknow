# Zero-to-book workflow

End-to-end command sequence for starting fresh: wipe all projects,
create a new one, ingest a folder of PDFs, populate the knowledge
graph + wiki, and launch the web reader.

## Prerequisites

- Native PostgreSQL 16, Qdrant, and Ollama running (see
  `docs/INSTALLATION.md`). The `sciknow` CLI assumes all three are
  reachable at the addresses configured in `.env`.
- `CROSSREF_EMAIL` set in `.env` — metadata extraction uses it as
  the polite User-Agent identifier.
- `uv sync` done in the repo root so `uv run sciknow …` works.

## Command sequence

Run top-to-bottom from the repo root. Replace `<PAPERS_FOLDER>` with
the absolute (or relative) path to your folder of PDFs. **If the path
contains spaces or shell metacharacters — `&`, `$`, `(`, `)`, `;`, `*`
— wrap it in double quotes** (see the gotcha below).

```bash
# Inspect what's there, then wipe every existing project (destructive).
# Removes: PG database, Qdrant collections, projects/<slug>/data/.
uv run sciknow project list
for slug in $(uv run sciknow project list --json 2>/dev/null \
              | jq -r '.[].slug' || echo global-cooling); do
  uv run sciknow project destroy -y "$slug"
done

# Create the new project and activate it.
uv run sciknow project init global-cooling
uv run sciknow project use global-cooling

# Initialise schema + Qdrant collections (idempotent; safe to re-run).
uv run sciknow db init

# Ingest the folder. Recursive by default; MinerU converts each PDF,
# then metadata → chunking → embedding. Resumable: if you Ctrl-C and
# re-run, it picks up where it stopped.
uv run sciknow ingest directory "<PAPERS_FOLDER>"

# Progress + per-stage status (run any time you want to peek).
uv run sciknow db stats

# Fill in missing DOIs + metadata via Crossref + OpenAlex + arXiv.
uv run sciknow db enrich

# Compile the knowledge wiki — THIS is what populates the KG with
# entity-relationship triples (plus summaries + concept pages).
uv run sciknow wiki compile

# (Optional) follow citations outward to grow the corpus.
uv run sciknow db expand

# Create a book and launch the web reader — the KG lives there.
uv run sciknow book create "Global Cooling"
uv run sciknow book serve
```

## Gotcha: shell metacharacters in folder paths

Symptom: you run `ingest directory` against a folder you can see in the
file browser, it returns almost immediately, and `db stats` reports
zero documents.

Cause: shell metacharacters in an **unquoted** path. The most common
offender is `&`. For example, passing

```bash
# BROKEN — the shell splits on & and backgrounds the first command
uv run sciknow ingest directory ../../Texts/Papers/Climate & Energy/
```

runs as **two** separate commands:

1. `uv run sciknow ingest directory ../../Texts/Papers/Climate` (backgrounded
   — the directory doesn't exist, so the ingest finds zero files).
2. `Energy/` (tried as its own command; fails silently).

Fix: wrap the path in double quotes, or escape the metacharacter with
`\`:

```bash
uv run sciknow ingest directory "../../Texts/Papers/Climate & Energy/"
# or
uv run sciknow ingest directory ../../Texts/Papers/Climate\ \&\ Energy/
```

Same rule for folder names containing `$`, `(`, `)`, `;`, `*`, `?`,
spaces, or backticks.

## What each step actually does

| Step | Touches | Why it's separate |
| --- | --- | --- |
| `project destroy` | PG database, Qdrant collections, `projects/<slug>/data/` | Isolates each project; destroy leaves nothing behind. |
| `project init` + `project use` | Creates a new PG db (`sciknow_<slug>`), empty data dir, sets `.active-project` | Per-project isolation — see `docs/PROJECTS.md`. |
| `db init` | Runs Alembic migrations, creates both Qdrant collections (`<slug>_papers`, `<slug>_abstracts`) | Idempotent; you always want to run this on a new project before ingesting. |
| `ingest directory` | PDFs → MinerU → metadata → chunker → bge-m3 dense + sparse → Qdrant + PG | Resume-safe — the `documents` table is keyed on file SHA-256. |
| `db enrich` | Hits Crossref / OpenAlex / arXiv for any paper without a DOI | Cheap on subsequent runs (only queries papers that still lack metadata). |
| `wiki compile` | LLM-generates page summaries, concept pages, and the knowledge-graph triples visible in the browser KG modal | This is where the KG gets populated — if you skip it, the Graph tab is empty. |
| `db expand` | Follows citations, downloads open-access PDFs, re-enters the pipeline at stage 1 | Optional; grows the library past your seed folder. |
| `book create` + `book serve` | Book with LLM-generated outline; web reader on `http://localhost:8000` | `book serve` is the command that opens the web UI (not `sciknow web`). |

## Checking the ingest log for failures

Every CLI invocation appends to `projects/<slug>/data/sciknow.log`.
That log only records *invocations* — actual stage-level errors surface
on stdout/stderr at the time the command runs (and in `db stats`' per-
stage status column). If `db stats` shows non-zero "failed" rows,
re-run the same ingest command — it retries failures from scratch —
and watch the console output.
