# SciKnow

A local-first scientific knowledge system that ingests papers, builds a compiled knowledge wiki, and writes grounded scientific books — all running on your own hardware. No cloud APIs.

**Ingest PDFs** (scanned or text) **&rarr; search & synthesize** across your library **&rarr; write entire books** with iterative AI review, all from the browser or CLI.

---

## Table of Contents

- [Features](#features)
- [Workflow](#workflow-from-pdfs-to-book)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Documentation](#documentation)
- [Hardware Requirements](#hardware-requirements)
- [Credits](#credits--acknowledgements)

---

## Features

**Ingestion & Library Management**
- **PDF ingestion** — MinerU 2.5 (SOTA for scientific papers) with Marker fallback. Handles scanned and text PDFs, tables, equations, figures
- **Metadata extraction** — 4-layer cascade: embedded PDF → Crossref → arXiv → LLM
- **Citation graph** — extracts references, cross-links corpus papers, boosts highly-cited papers in search
- **Five corpus-expansion vectors**, all sharing a **preview-and-select** flow in the browser (checkbox shortlist, per-row relevance score, "Download selected"):
  - `db expand` — **outbound** citations (follow references in existing papers)
  - `db expand-author` — every paper by a **named author** across OpenAlex + Crossref
  - `db expand-cites` — **inbound** citations (papers that cite yours — forward-in-time mirror)
  - `db expand-topic` — **free-text topic** search (solves the bootstrap + sideways-expansion problem)
  - `db expand-coauthors` — **invisible college** (papers by coauthors of your corpus authors)
  - `book auto-expand` — **gap-driven** auto-expansion: every open `book gaps` entry becomes its own topic search, candidates merged + ranked so papers that close multiple gaps rise to the top
- **Cross-project dedup** — `db cleanup-downloads --cross-project` (default ON) checks every sciknow project's DB by SHA-256, so a PDF downloaded into project B that's already ingested in project A is recognised and cleaned
- **Pending downloads panel** — ~50% of expand selections typically have no legal OA PDF; those rows auto-persist to `pending_downloads` with full metadata (title, authors, year, source-method) so you can retry (the 6-source cascade bypasses `.no_oa_cache`), mark manually-acquired, abandon with a note, or export to CSV for ILL. Surfaced as the "📋 Pending downloads" entry in the top-bar **🌱 Corpus ▾** dropdown and `sciknow db pending list|retry|mark-done|abandon|export` from the CLI
- **Topic clustering** — BERTopic (UMAP + HDBSCAN + c-TF-IDF) assigns papers to named thematic clusters in seconds

**Search & Retrieval**
- **Hybrid search** — dense + sparse + full-text, fused with RRF, reranked with a cross-encoder
- **Self-correcting RAG** — evaluates retrieval quality, reformulates if poor, checks answer grounding
- **Multimodal awareness** — table/equation tagging for filtered retrieval

**Knowledge Wiki** (Karpathy LLM-wiki pattern)
- **Compiled knowledge layer** — papers synthesized into interconnected wiki pages (summaries, concept pages, synthesis overviews)
- **Knowledge graph** — entity-relationship triples extracted during compilation (GraphRAG-style), visualized as an interactive 3D orbit graph in the browser (drag background to rotate, drag nodes, wheel to zoom)
- **Consensus mapping** — tracks agreement/disagreement across the corpus over time
- **Contradiction detection** — LLM-based lint finds disagreements between papers

**Book Writing Platform**
- **Structured projects** — book → chapter hierarchy with LLM-generated outlines and per-chapter custom sections
- **Iterative refinement** — write → review → revise loop with 5-dimension scoring and claim verification
- **Auditable citation insertion** — `book insert-citations` (also wired to the "Insert Citations" toolbar button) runs a two-pass LLM flow: pass 1 identifies locations needing a citation, pass 2 retrieves top-K candidates per claim and picks (or rejects) with confidence; deterministic rewrite inserts `[N]` markers and saves as a new version
- **Critic Skills** (BMAD-inspired, adapted from [bmad-code-org/BMAD-METHOD](https://github.com/bmad-code-org/BMAD-METHOD) under MIT) — two new per-chapter toolbar buttons orthogonal to the graded review. **🧛 Adversarial** runs a cynical critic that must find ≥10 concrete issues per draft (unsupported claims, weasel words, missing counter-evidence, internal contradictions, loaded framing). **🤿 Edge cases** runs exhaustive path enumeration — scope boundaries, counter-cases, causal alternatives, quantitative limits, missing controls — and returns a severity-ranked findings table. CLIs: `book adversarial-review <draft>` / `book edge-cases <draft>`
- **Method catalogues** — 24 elicitation + 24 brainstorming methods (also adapted from BMAD, MIT): "Tree of Thoughts", "Pre-mortem", "Peer Review Simulation", "Strong Inference", "Reverse Brainstorming", "Five Whys", "Scope Boundaries", etc. Surface as a dropdown on the Plan modal (steers outline generation) and an interactive prompt on the Gaps button (steers gap analysis)
- **Autowrite** — autonomous convergence loop: generates, scores, verifies, revises until quality target is met
- **TreeWriter planning** — hierarchical paragraph-level plans before drafting
- **Web reader** — browser-based authoring with a full-width top bar that's organised around five dropdowns (Plan + Dashboard stay direct for high-frequency access): **📖 Book ▾** (Corkboard / History / Snapshot / Export / Settings) · **🔍 Explore ▾** (Ask Corpus / Wiki Query / Browse Papers) · **🌱 Corpus ▾** (Enrich / Expand citations / Expand by author / Inbound cites / Topic search / Coauthors / Cleanup / Pending downloads) · **📊 Visualize ▾** (Knowledge Graph + the six ECharts tabs) · **🛠 Manage ▾** (Tools / Setup / Projects). The per-chapter toolbar keeps the five-button **Write loop** flat (Edit / Autowrite / Write / Review / Revise — hit constantly) and collapses the rest into three dropdowns: **🔎 Verify ▾** (Verify / Insert Citations / Scores) · **🧠 Critique ▾** (Argue / Gaps / Adversarial / Edge cases) · **📦 Extras ▾** (Bundles / Chapter reader). Live LLM streaming, corkboard view, chapter reader, argument maps, citation popovers, snapshots, version diffs
- **Knowledge wiki in the browser** — the Wiki modal has four tabs: Query (RAG-streamed answer over compiled pages), Browse (paginated page index with detail view, KaTeX math, backlinks, related pages, per-page inline Ask + personal "My take" notes), **Lint** (broken links, stale pages, orphaned concepts, optional LLM contradiction detection, **Extract / Backfill KG** button for wikis compiled before the combined entity+KG extraction step), **Consensus** (strong / moderate / weak / contested claim classification for a topic with supporting vs contradicting papers)
- **Regenerate chapter outline from the browser** — the Plans modal now has a "📖 Generate outline" button that runs `sciknow book outline` against your paper library, streams the LLM response, parses the proposed chapter list, and adds any new chapters without touching existing drafts
- **Six-way corpus visualization** — **📊 Visualize ▾** dropdown in the top bar with six direct links, all backed by ECharts (zoom / pan / tooltips / legend toggles built in). **Topic map** (UMAP 2D of abstract embeddings, coloured by BERTopic cluster, cached per project), **RAPTOR sunburst** (drill-in hierarchical cluster tree), **Consensus landscape** (claims scattered on supporting × contradicting axes, coloured by consensus_level — runs `wiki consensus` synchronously), **Timeline river** (stacked-area of papers-per-cluster over years with brush-zoom), **Ego radial** (top-K nearest papers around one document on the abstract-embedding cosine, drawn on a polar plot), and **Gap radar** (per-chapter section-coverage polygon derived from `book_gaps`). All six share a **theming bar**: 7 palette chips (Paper, Deep Space, Blueprint, Solarized, Solarized Light, Terminal, Neon — same presets as the Knowledge Graph), invert-to-paired-theme, BG + label custom colour pickers, typography dropdown (Sans / Serif / Mono / Condensed / Display), label-size slider, fullscreen, and download-PNG. Theme + font + scale persist in localStorage — one choice applies across every tab, theme swaps re-render without re-fetching the data
- **Compute dashboard** — book-level GPU compute ledger: cumulative tokens, wall time, and per-operation breakdown (write/review/revise/argue/gaps/autowrite) across every LLM call
- **Tools panel** — CLI-parity in the browser: hybrid corpus search, similarity search, multi-paper synthesis, topic-cluster browser. Corpus-growing tools were split out into their own top-bar **🌱 Corpus ▾** dropdown (six preview-and-select expansion flows — Enrich / Expand citations / Expand by author / Inbound cites / Topic search / Coauthors — plus a one-click "🧹 Cleanup downloads + failed" that reclaims disk by deleting already-ingested duplicates **and** nuking the failed-ingest archive + matching `documents` rows, and a Pending-downloads manager)
- **Dashboard gap integration** — the Open Gaps panel has a top-level "🔍 Auto-expand from these gaps" button (runs `book auto-expand` and opens the preview modal with all gaps merged), plus a per-gap "Expand" button that prefills the Topic-search subtab with that specific gap's description
- **Per-section model override** — dial expensive models up on the sections that need them (methods, results) and down on the cheap ones (overviews, conclusions); set per section in the Chapter Sections tab
- **Scoped snapshot bundles** — whole-chapter and whole-book snapshots as a single click before firing autowrite-all; restore is non-destructive (creates new draft versions, existing drafts stay as undo path)
- **Book Settings panel** — one tabbed modal consolidates title, description, leitmotiv, target word count, and the style fingerprint (with on-demand refresh) so per-book config isn't scattered across four surfaces
- **Projects modal with graceful restart** — switch active project from the browser; when a restart is required (DB / Qdrant singletons can't hot-swap), a one-click "⏻ Stop this server" button cleanly SIGTERMs the process so the terminal returns to `$` ready for the next `sciknow book serve` invocation
- **Multi-format export** — Markdown, HTML, PDF (WeasyPrint), EPUB (pandoc), BibTeX, LaTeX, DOCX with global citation dedup, available from both the CLI and the web reader

**Infrastructure**
- **All local** — PostgreSQL + Qdrant + Ollama, no cloud APIs, no Docker
- **Backup & restore** — portable archives for migrating between machines
- **Multi-project** — host multiple isolated knowledge bases (own DB + Qdrant collections + data dir per project) with `sciknow project init/list/use/destroy/archive`

---

## Workflow: From PDFs to Book

```
 1  INGEST        PDFs → chunks in PostgreSQL + vectors in Qdrant
 2  ENRICH        Fill missing DOIs and metadata from Crossref/OpenAlex
 3  EXPAND        Follow citations → download related open-access papers
 4  CLUSTER       Group papers into thematic topics (BERTopic, seconds)
 5  WIKI          Build the compiled knowledge wiki (summaries + concepts + KG)
 6  EXPLORE       Ask questions, search, synthesize — understand your corpus
 7  BOOK          Create → outline → plan → write → review → export
```

> Steps 2-5 are optional but each improves downstream quality. You can ask questions right after step 1.

```bash
# Quick start — the essential commands
sciknow ingest directory ./papers/
sciknow db stats
sciknow catalog cluster
sciknow wiki compile
sciknow ask question "What is total solar irradiance?"
sciknow book create "My Book"
sciknow book outline "My Book"
sciknow book plan "My Book"
sciknow book serve "My Book"              # open browser, write from there
sciknow book autowrite "My Book" --full   # or let it write autonomously
sciknow book export "My Book" --format latex -o manuscript.tex
```

### Full rebuild sequence (from zero)

Every step is idempotent / resumable. Later blocks depend on earlier ones.

```bash
# ═══ 1. PROJECT + SCHEMA ══════════════════════════════════════════
uv run sciknow project init my-project
uv run sciknow project use my-project
uv run sciknow db init                             # alembic upgrade + Qdrant collections

# ═══ 2. INGEST + METADATA ═════════════════════════════════════════
uv run sciknow ingest directory ./papers/          # PDFs → Postgres + Qdrant
uv run sciknow db enrich                           # fill missing DOIs via Crossref / OpenAlex / arXiv
uv run sciknow db link-citations                   # cross-link cited_document_id for in-corpus papers
uv run sciknow db stats                            # sanity-check: all papers should be 'complete'

# ═══ 3. INDEXING LAYERS ════════════════════════════════════════════
uv run sciknow catalog cluster                     # BERTopic → paper_metadata.topic_cluster
                                                   #   (feeds Topic map viz, search --topic filter)
uv run sciknow catalog raptor build                # hierarchical summary tree in Qdrant
                                                   #   (feeds RAPTOR sunburst viz, enriches retrieval)
uv run sciknow db tag-multimodal                   # tag chunks with tables / equations for filtering

# ═══ 4. WIKI (SLOWEST STEP — hours) ═══════════════════════════════
uv run sciknow wiki compile                        # paper summaries (fast, reliable)
uv run sciknow wiki extract-kg                     # KG triples + concept entities (separate step)
uv run sciknow wiki lint                           # broken links, stale pages, orphaned concepts
uv run sciknow wiki lint --deep                    # + LLM contradiction detection (~30s per concept pair)
# Phase 54.6.35: wiki compile and wiki extract-kg are now two separate
# jobs. Inline entity extraction during compile is opt-in via
# `--with-entities` — off by default because `format=json_schema` on
# dense content is unreliable enough (runaway JSON, thinking loops)
# that we'd rather fail one job cleanly than block the whole compile.

# ═══ 5. BOOK ══════════════════════════════════════════════════════
uv run sciknow book create "My Book"
uv run sciknow book outline "My Book"              # LLM proposes chapter structure
uv run sciknow book plan "My Book"                 # thesis + scope document
uv run sciknow book serve "My Book"                # http://localhost:8765 — all GUI surfaces populated

# ═══ 6. OPTIONAL ══════════════════════════════════════════════════
uv run sciknow watch seed                          # pre-populate upstream-repo watchlist
uv run sciknow backup schedule                     # daily auto-backup to archives/backups/
```

### Adding new papers later

Drop new PDFs into `projects/<slug>/data/inbox/` and run the master refresh
command — it re-runs every step in the right order, skipping what's already
done:

```bash
uv run sciknow refresh                        # full pipeline (including wiki)
uv run sciknow refresh --no-wiki              # skip the hours-long wiki compile
uv run sciknow refresh --dry-run              # preview what would run
uv run sciknow refresh --budget-time=6h       # soft wall-clock cap (54.6.206)
uv run sciknow refresh --since=last-run       # incremental: skip the full-corpus scan (54.6.210)
uv run sciknow refresh --since=7d             # or name an explicit window (24h / 7d / 2026-04-22)
```

Every step is idempotent, so `refresh` is safe to run any time. Use
`--no-<step>` to skip individual steps (`--no-ingest`, `--no-cluster`,
`--no-raptor`, `--no-wiki`, etc.). `--since=last-run` reads the
timestamp of the last clean completion from
`projects/<slug>/data/.last_refresh` and restricts expensive LLM-heavy
steps (currently `wiki compile`) to papers ingested after that point —
a 5,000-paper corpus drops from minutes of prep to seconds when only
a dozen papers are new.

### Live monitor

`sciknow db monitor` is a btop-inspired single-screen dashboard for
the whole system: corpus counts with a done/total progress bar,
GPU VRAM + utilization bars with tri-colour heat, currently-loaded
Ollama models, Qdrant collection shapes (with ◆ColBERT / ●dense /
◇sparse markers), converter-backend distribution, pipeline stage
p95 timing bars normalised against the slowest stage, and a recent
activity feed. **Phase 54.6.243** adds a consolidated **alerts
banner** (stuck ingest, embedding drift, stale bench, GPU overheat,
RAM pressure, inbox waiting — worst-first, red/yellow/cyan), plus
retrieval-quality signals (abstract coverage %, chunk char p50/p95,
KG triples/doc), wiki materialization ratio, inbox drop-zone count,
and a cross-project inventory.

**Phases 54.6.244–270** extend the monitor with a 0-100 composite
health score, service-up probes (PG / Qdrant / Ollama with latency),
active-job cross-process visibility (CLI sees web jobs via a
`web_jobs.json` pulse file), stop-light TPS per job, stall watchdog,
backup/bench freshness chips, config-drift surfacing, a log tail,
11 more alert classes with actionable fix commands, and — on the web
side — filter/search, URL-hash deep links, snapshot download, NEW
badges on unseen alerts, a health-trend sparkline, Markdown export,
and browser notifications when new error alerts fire on a hidden tab.

**Phase 54.6.280** surfaces the corpus's **citation graph** in both
surfaces: internal-ref coverage (what fraction of outgoing refs land
on papers we also have), extraction coverage (how many papers had
their references parsed at all), and the orphan count (complete docs
with zero incoming citations from the rest of the corpus). The web
modal additionally lists the top-5 most-cited papers. Useful for
deciding when to run `sciknow db expand` (low coverage) vs when the
MinerU fallback is dropping reference sections (low extraction).

**Phase 54.6.281** adds an **inbox age histogram** — the inbox scan
now walks recursively (matching `ingest directory data/inbox/` and
`cleanup-downloads --include-inbox`) and buckets waiting PDFs into
fresh (<24h), week (1-7d), month (7-30d), and stale (>30d). Before
54.6.281 the top-level-only scan was under-counting inbox load; now
the operator sees which drops are recent vs forgotten. Colour-coded
inline in the CLI corpus panel and in the web monitor's rates/ETA
banner.

**Phase 54.6.282** adds a **section-type coverage** panel — chunks
are grouped by canonical section type (abstract / introduction /
methods / results / discussion / conclusion / related_work /
appendix / unknown) and rendered as a one-line summary in the CLI
corpus panel and as a stacked bar with legend in the web modal. A
high `unknown` percentage flags either a chunker regression or a
converter class that's losing heading structure — check the
`converter_backend` mix when the number climbs above 70 %.

**Phase 54.6.283** adds an **LLM role-usage heatmap** to the web
modal: rows=operations (autowrite_writer, wiki_compile, extract_kg,
…), columns=days over the monitor window, cell colour = call count
on a log ramp. Built from `llm_usage_log` so it only lights up for
operations that log there (web-UI runs today). Makes "did autowrite
run today?" and "when did wiki compile stall?" answerable at a
glance without SQL.

```bash
uv run sciknow db monitor              # one shot, full layout
uv run sciknow db monitor --watch 5    # btop-style in-place refresh
uv run sciknow db monitor --json       # JSON for scripting
uv run sciknow db monitor --compact    # minimal 1-page view (54.6.270)
uv run sciknow db monitor --filter foo # case-insensitive row filter  (54.6.254)
uv run sciknow db monitor --log-tail 20       # append last N log lines (54.6.260)
uv run sciknow db monitor --alerts-md         # alerts as Markdown block (54.6.268)

uv run sciknow db doctor               # go/no-go readiness (54.6.253)
uv run sciknow db doctor --json        # scriptable (exit 0/1/2)
```

`sciknow db doctor` is a focused wrapper that prints the traffic-
light verdict + health score + hardware summary + grouped alerts,
then exits with a shell-friendly code tied to the worst severity.
Pipeline-friendly: `sciknow db doctor && sciknow ingest directory …`
is safer than eyeballing the monitor before a long run.

Watch mode uses Rich's `Live` + alternate-screen buffer, so it
redraws the same character cells every tick (no scrolling history)
and cleanly restores the terminal on Ctrl+C. Panels are colour-
coded: green=healthy, yellow>50% load, red>85% — same visual
grammar as `btop` / `htop`.

In the web reader, the same data is exposed at `GET /api/monitor`
(schema identical to `--json`) and rendered inside a "System Monitor"
modal that polls every 5s — open via ⌘K → `monitor`. CLI and GUI
share the aggregator (`sciknow.core.monitor.collect_monitor_snapshot`),
so they can never drift out of sync.

**Web-side QoL** (Phases 54.6.254–269): filter input at the top of
the modal (press `/` to focus from anywhere, Esc to clear); ⬇
Snapshot button downloads the current `/api/monitor` JSON; a jump-
to nav strip lists every panel as a scroll-chip that also updates
the URL hash so links like `/#mon-active-jobs` deep-link into a
panel; alert banner shows a 📋 Copy-as-MD button plus per-alert
📋 copy buttons that pull suggested fix commands to the clipboard;
"NEW" badges flag alert codes not seen before (per-browser via
localStorage); poll cadence auto-drops to 2s while any job is
running and back when idle; and new error-level alerts raised
while the tab is hidden fire a browser notification so long
ingestion runs can babysit themselves.

Once the web reader is up, the top bar gives you:

| Action | What to click |
|---|---|
| Browse the compiled wiki (with Year + Authors cols) | 📚 Wiki Query → Browse tab |
| Lint the wiki + backfill KG | 📚 Wiki Query → Lint tab → "Extract / Backfill KG" |
| Consensus map for a topic | 📚 Wiki Query → Consensus tab |
| See the KG in 3D | 🔗 KG (+ the Font dropdown for label typography) |
| Grow the corpus with preview-and-select | **🌱 Corpus ▾** → any of the 6 expansion subtabs |
| Retry downloads that had no OA PDF | **🌱 Corpus ▾** → 📋 Pending downloads |
| Clean up duplicate downloads + nuke failed-ingest archive | **🌱 Corpus ▾** → 🧹 Cleanup downloads + failed |
| Ask the corpus a question | 🔍 Ask Corpus |
| Regenerate a chapter outline | 📝 Plan → 📖 Generate outline |
| Auto-insert citations in a draft | toolbar (per-chapter row) → 📑 Insert Citations |
| Auto-expand corpus from book gaps | Dashboard → Open Gaps → 🔍 Auto-expand from these gaps |
| Visualize the corpus | **📊 Visualize ▾** → pick one of six views |
| Manage projects (switch active / create / destroy) | 📁 Projects |

### Multi-project workflow

Each project has its own PostgreSQL database, Qdrant collections, and `data/`
directory, so corpora never cross-contaminate. Project resolution precedence:
`--project <slug>` flag → `SCIKNOW_PROJECT` env var → `.active-project` file →
legacy `default` (the pre-Phase-43 single-tenant layout).

```bash
sciknow project init <slug>                     # create a fresh empty project
sciknow project init global-cooling --from-existing
                                                # adopt the legacy install into a slot
sciknow project list                            # all projects + active marker
sciknow project show [slug]                     # details (defaults to active)
sciknow project use <slug>                      # set .active-project
sciknow project destroy <slug> --yes            # drop DB + collections + data dir
sciknow project archive <slug>                  # bundle to portable archive, drop live state
sciknow project unarchive archives/<slug>-<ts>.skproj.tar
sciknow --project <slug> <any subcommand>       # one-shot override
```

See [`docs/PROJECTS.md`](docs/PROJECTS.md) for the full design.

### Recent milestones

sciknow is active; the [`docs/PHASE_LOG.md`](docs/PHASE_LOG.md) records
every Phase commit. Most recent batches:

- **54.6.165-201 (April 2026).** Full web UI redesign — "scholar's
  editor, not LLM dashboard." 37-commit arc replaces the Tailwind-
  starter look with a considered writing environment. **Real
  webfonts** (Inter Tight for chrome, Newsreader with optical sizes
  for the reader body, JetBrains Mono for code), **warm stone
  neutrals on ivory paper** with **pitch-black dark mode**,
  **deep teal accent** (no more indigo) and desaturated semantic
  colours. **Unified `.btn` system** collapses 12 legacy button
  classes into one foundation with variant + size modifiers.
  **41-icon SVG sprite** (Lucide-inspired monoline) replaces emoji
  in toolbar, topbar, peek/hide, Stop, modal titles, and dropdown
  menu items. **Right panel** becomes a segmented-control context
  rail (Sources / Review / Comments) instead of three always-on
  stacks. **⌘K command palette** with 33-command fuzzy
  subsequence match + keyboard navigation, accessible from a topbar
  trigger pill. **Reader canvas** runs at 720px / 17px Newsreader
  with a 3.6em drop cap, plus full content-type styling
  (lists / blockquotes / code / tables / figures / scrollbars).
  **Routed views** — 16 URLs (`/plan` `/settings` `/wiki`
  `/projects` …) deep-link to modals; browser back/forward walks
  the modal stack. **Consolidated single-line topbar** (54.6.186,
  54.6.190) — book-level nav (Plan / Dashboard / Book / Explore /
  Corpus / Visualize) on the left, per-draft actions (Edit / AI ▾ /
  Verify ▾ / Critique ▾ / Extras ▾ / help / ⌘K) on the right, one
  row, `flex-wrap: nowrap` so it never breaks onto two lines. The
  four AI verbs (Autowrite / Write / Review / Revise) share one
  **AI ▾** dropdown (54.6.188). Topbar gets a **home anchor** at
  the leftmost position + Escape-goes-home progressive-disclosure
  key handler (54.6.193); Plan folds into the Book menu. **Sidebar
  rail mode** (54.6.194) — VS-Code-style compact 64-px navigator
  showing chapter numbers + status dots only, toggleable via a
  dedicated button, persists in localStorage. **Loading skeletons**
  (54.6.195) — shimmer-gradient placeholders for async lists.
  **Full-page routed views** (54.6.200) — every modal opened via
  URL fills the viewport below the topbar instead of floating as
  a scrim; Escape or × pops back to the reader. **Dark-mode
  vis-table / vis-eq outline** (54.6.201) so VLM-rendered white
  table/equation cards have a visible edge on pitch black.
  **Inline-style purge** retired **984 of 1,218** `style="…"`
  attributes (**81%**) across 10 mechanical waves into 120+
  utility classes — including a decomposing migrator that splits
  multi-declaration combos and a paired JS-toggle rewrite for
  `display:none`. All L1 tests green through the arc.
- **54.6.164 (April 2026).** Collapsible reader columns — hide buttons on
  the left (chapters) and right (sources/review/comments) panes, with
  edge peek buttons to bring them back. New **Book Settings → View** tab
  exposes per-browser auto-hide preferences (stored in localStorage),
  so the reader pane can open full-width on every load.
- **54.6.134-162 (April 2026).** Two major tracks closed end-to-end:
  **(A) visuals-in-writer** (138-145) — MinerU-extracted figures/tables
  are now first-class citable evidence in autowrite via a 5-signal
  ranker (caption similarity + same-paper co-citation + mention-paragraph
  alignment + section-type prior), per-iteration L1+L2 verify for
  `[Fig. N]` markers, and a pre-export L3 VLM claim-depiction check
  (`book finalize-draft`). **(B) concept-density length sizing** (146-162)
  — sections with a bullet plan auto-size bottom-up (Cowan 2001 3-4
  novel chunks × per-type words-per-concept), with live readout as
  the user types, a resolver-explanation badge per section, a
  retrieval-density widener (RESEARCH.md §24 guideline 4), a soft
  Delgado-2018 digital ceiling, corpus-grounded §24 validation
  bench, Brown 2008 idea-density regression harness, and a bottom-up
  vs top-down autowrite A/B harness. Research brief lives in
  `docs/RESEARCH.md §24`. Also shipped: chunk-level FTS (**+8.6% R@1**),
  scheduled OpenAlex velocity watcher.
- **54.6.68-83 (April 2026).** Research-sweep implementation: retrieval
  MRR/Recall/NDCG bench, citation marker alignment, vision captioning
  pipeline + VLM sweep harness, chapter/book snapshot CLI, GPU-time
  ledger, MCP server, equation paraphrase + Qdrant index, plan-coverage
  scoring dimension, paper-type classifier + weighted retrieval,
  offline claim-atomization verifier.
- **54.6.54-67 (April 2026).** Model benchmarks (full sweep + quality),
  LLM_MODEL unified on ``qwen3:30b-a3b-instruct-2507``,
  BOOK_REVIEW_MODEL / AUTOWRITE_SCORER_MODEL per-role overrides,
  verbose expand / enrich / cleanup output, `refresh` sweeps inbox +
  downloads + failed, Plan modal restructured to Book / Chapters /
  Sections, Summaries + Visuals tabs in the Compiled Knowledge Wiki.
- **54.6.20-53 (April 2026).** Multi-project isolation, per-project
  ``.env.overlay``, wiki UX polish, citation + reference extraction,
  two-round audit fixes, model-sweep + writing-quality benchmarks.

### When to use what

| I want to... | Use this |
|---|---|
| **Ask a quick question** about my papers | `sciknow ask question "..."` |
| **Get a pre-synthesized answer** from compiled knowledge | `sciknow wiki query "..."` |
| **Synthesize findings** across multiple papers | `sciknow ask synthesize "topic"` |
| **Write a full book** with chapters and review | `sciknow book ...` |
| **Browse compiled knowledge** about a concept | `sciknow wiki show concept-slug` |
| **Find contradictions** in your corpus | `sciknow wiki lint --deep` |
| **Map agreement/disagreement** on a topic | `sciknow wiki consensus "topic"` |
| **Map evidence for/against a claim** | `sciknow book argue "claim"` |
| **Find gaps** in a book project | `sciknow book gaps "Book"` |
| **Auto-fill book gaps** with new papers from OpenAlex | `sciknow book auto-expand "Book"` |
| **Follow references** of my papers | `sciknow db expand` |
| **Find papers that cite mine** (forward-in-time) | `sciknow db expand-cites` |
| **Pull every paper by an author** | `sciknow db expand-author "Solanki"` |
| **Broad-search OpenAlex** by topic (bootstrap / new direction) | `sciknow db expand-topic "thermospheric cooling"` |
| **Coauthor snowball** (same-lab researchers) | `sciknow db expand-coauthors` |
| **Auto-insert citations** into a draft | `sciknow book insert-citations <draft-id>` |
| **Reclaim disk** from already-ingested downloads | `sciknow db cleanup-downloads` |
| **See papers stuck without a legal OA PDF** | `sciknow db pending list` |
| **Retry pending downloads** (new OA links may have appeared) | `sciknow db pending retry` |
| **Backfill the KG** (empty or stale `knowledge_graph`) | `sciknow wiki extract-kg` |
| **Draft a chapter outline** from the LLM | `sciknow book outline "Title"` (also available from the Plans modal) |
| **Auto-plan section concepts** (3-4 bullets per section) | `sciknow book plan-sections "Title"` (also the Chapter modal + Book Settings buttons) |
| **See whole-book projected length** at a glance | `sciknow book length-report "Title"` (also the Book Settings panel) |
| **Pre-export verify figure citations** ([Fig. N] → VLM) | `sciknow book finalize-draft <draft-id>` (also the Verify dropdown) |
| **Switch a book's project type** (or unfreeze default) | `sciknow book set-target "Title" --unset` then `book set-target "Title" --words N` |
| **Link body-text mentions to figures** | `sciknow db link-visual-mentions` |
| **Watch a topic for new hot papers** | `sciknow watch add-velocity "thermospheric cooling"` |
| **Measure visuals ranker quality** (P@1 / R@3) | `sciknow bench-visuals-ranker` |
| **Measure corpus idea density vs §24** | `sciknow bench-idea-density` *(needs spaCy)* |
| **A/B autowrite: bottom-up vs top-down** | `sciknow bench-autowrite-ab <chapter-id>` |

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/claudenstein/sciknow
cd sciknow
bash scripts/setup.sh

# 2. Pull the main LLM
ollama pull qwen3.5:27b

# 3. Configure (set your email for Crossref polite pool)
nano .env

# 4. Initialize
sciknow db init

# 5. Ingest your papers
sciknow ingest directory ./papers/
```

See [Installation Guide](docs/INSTALLATION.md) for manual installation, Ollama performance tuning, and full configuration reference.

---

## Architecture Overview

```
PDFs ──→ MinerU 2.5 ──→ Metadata ──→ Chunker ──→ bge-m3 ──→ PostgreSQL + Qdrant
                                                                      │
Query ──→ Dense + Sparse + FTS ──→ RRF fusion ──→ Reranker ──→ Ranked results
                                                                      │
                                                              LLM (Ollama) ──→ Answer
```

Three services, all native (no Docker): **PostgreSQL 16** (relational + full-text), **Qdrant** (vectors), **Ollama** (LLM inference).

See [Architecture](docs/ARCHITECTURE.md) for the full system diagram, database schema, AI model details, and project structure.

---

## Documentation

Core guides — how to use sciknow, how it works, how to extend it:

| Document | Contents |
|---|---|
| **[Installation](docs/INSTALLATION.md)** | Setup script, manual install, Ollama tuning, configuration reference |
| **[Workflow](docs/WORKFLOW.md)** | End-to-end walkthrough: PDFs → ingest → wiki → book |
| **[CLI Reference](docs/CLI.md)** | Complete command reference for every `sciknow` subcommand |
| **[Architecture](docs/ARCHITECTURE.md)** | System diagram, project structure, DB schema, service layer pattern |
| **[Ingestion Pipeline](docs/INGESTION.md)** | PDF conversion (MinerU/Marker), metadata, chunking, embedding |
| **[Retrieval & RAG](docs/RETRIEVAL.md)** | Hybrid search, RRF fusion, reranking, corrective RAG |
| **[Book Writing System](docs/BOOK.md)** | Book workflow, autowrite convergence, web reader, export |
| **[Book Actions Reference](docs/BOOK_ACTIONS.md)** | Every AI button (outline/review/autowrite/verify/align/…) + the 24 elicitation + 24 brainstorming methods |
| **[Multi-Project](docs/PROJECTS.md)** | Per-project isolation (DB + Qdrant + `data/`), lifecycle CLI |
| **[Operations](docs/OPERATIONS.md)** | Backup/restore, expand, enrich, citations, dev notes |
| **[Testing Protocol](docs/TESTING.md)** | 3-layer smoke harness (`sciknow test`); how to add checks |
| **[Benchmarks](docs/BENCHMARKS.md)** | `sciknow bench` — performance + quality measurement; baseline findings |
| **[Bench methodology](docs/BENCH_METHODOLOGY.md)** | Hard-won rules for fair LLM/VLM benchmarking; the 2026-04-17 failure that motivated them |
| **[Concept-density length sizing](docs/CONCEPT_DENSITY.md)** | Bottom-up section sizing via Cowan 2001 (3-4 novel chunks × per-genre wpc). How to use the auto-plan buttons, length-report, retrieval-density widener, and Brown 2008 regression harness. |
| **[Expand + enrich research (2026-04)](docs/EXPAND_ENRICH_RESEARCH_2.md)** | Gaps audit + 2024-2026 literature sweep; follow-on to `EXPAND_RESEARCH.md` for what's next in corpus growth |

Project planning and release history:

| Document | Contents |
|---|---|
| **[Roadmap](docs/ROADMAP.md)** | Open items by source (QA, research runners-up, hardware-gated, compound-learning layers) |
| **[Ingestion roadmap](docs/ROADMAP_INGESTION.md)** | 43 research proposals for improving `sciknow refresh` / ingest — by pipeline stage + cross-cutting, priority-ranked |
| **[Phase log](docs/PHASE_LOG.md)** | Release notes per phase commit, newest first |
| **[Strategy](docs/STRATEGY.md)** | Long-range direction; why certain trade-offs were chosen |
| **[Lessons](docs/LESSONS.md)** | Post-mortem + what-not-to-do notes accumulated over phases |
| **[Credits](docs/CREDITS.md)** | Open-source projects and research papers sciknow builds on |

Research notes (reading material, not authoritative for current behaviour):

| Document | Contents |
|---|---|
| **[Research & innovations](docs/RESEARCH.md)** | Techniques implemented with literature references (BERTopic, GraphRAG, Self-RAG, TreeWriter, RAPTOR, CoVe, etc.) |
| **[Comparative analysis](docs/COMPARISON.md)** | Audit vs karpathy/autoresearch, AI-Scientist, AutoResearchClaw, FARS |
| **[Expand research](docs/EXPAND_RESEARCH.md)** | Discovery-pipeline design notes |
| **[KG research](docs/KG_RESEARCH.md)** | Knowledge-graph extraction design |
| **[Wiki compile speed](docs/WIKI_COMPILE_SPEED.md)** | Wiki-compile latency investigation |
| **[Wiki UX research](docs/WIKI_UX_RESEARCH.md)** | Wiki browsing UX decisions |
| **[Agentic RAG course review](docs/AGENTIC_RAG_COURSE_REVIEW.md)** | Historical review — what sciknow borrowed |
| **[Autoreason review](docs/AUTOREASON_REVIEW.md)** | Same shape, autoreason |
| **[MemPalace review](docs/MEMPALACE_REVIEW.md)** | Same shape, MemPalace |
| **[DeepScientist ports](docs/DEEPSCIENTIST_PORTS.md)** | DeepScientist technique ports pending evaluation |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3090 (24 GB VRAM) | RTX 3090 + remote GPU server |
| RAM | 32 GB | 64 GB |
| Storage | 500 GB SSD | 2 TB NVMe |
| OS | Ubuntu 22.04+ | Ubuntu 22.04+ |

**VRAM budget on 3090 (24 GB):** bge-m3 (~2.2 GB) + qwen3.5:27b (~18 GB) + bge-reranker (~0.5 GB) = fits comfortably.

**Remote GPU:** Set `OLLAMA_HOST=http://your-gpu-server:11434` in `.env`. Zero code changes.

---

## Credits & Acknowledgements

sciknow builds on excellent open-source projects and research. Full details in [Credits](docs/CREDITS.md).

**Key projects:** [MinerU](https://github.com/opendatalab/MinerU), [Marker](https://github.com/VikParuchuri/marker), [Qdrant](https://github.com/qdrant/qdrant), [Ollama](https://github.com/ollama/ollama), [bge-m3](https://huggingface.co/BAAI/bge-m3), [BERTopic](https://github.com/MaartenGr/BERTopic), [FastAPI](https://github.com/fastapi/fastapi), [Typer](https://github.com/tiangolo/typer), [Rich](https://github.com/Textualize/rich), [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy), [uv](https://github.com/astral-sh/uv)

**Key research:** [RRF (SIGIR 2009)](https://dl.acm.org/doi/10.1145/1571941.1572114), [BGE M3-Embedding (2024)](https://arxiv.org/abs/2402.03216), [GraphRAG (Microsoft 2024)](https://arxiv.org/abs/2404.16130), [BERTopic (2022)](https://arxiv.org/abs/2203.05794), [Self-RAG (2023)](https://arxiv.org/abs/2310.11511), [TreeWriter (2025)](https://arxiv.org/abs/2601.12740), [OmniDocBench (2024)](https://arxiv.org/abs/2412.07626), [Karpathy LLM-wiki](https://x.com/karpathy/status/1756130027985752370)

---

*sciknow is an independent research tool. All referenced projects retain their respective licenses.*
