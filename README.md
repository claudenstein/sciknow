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

**Phase 54.6.284** adds a **retraction detail panel** to the web
modal — the operator can see titles / DOIs / years / statuses
behind the existing `retracted_papers` info alert. Per Phase
54.6.276 policy, retracted and corrected papers stay flagged but
are not auto-excluded (they may be in the corpus for good reason;
the operator decides per-case). DOI cells link to doi.org.

**Phase 54.6.286** adds a **VRAM headroom watchdog** — `vram_low`
alerts fire at <15 % free, `vram_critical` at <5 % free, with a
suggested-fix command that dumps the per-process GPU memory map.
Motivated by the 54.6.285 verification finding: the dual-embedder
+ MinerU-VLM + Ollama stack can push a 24 GB 3090 to <100 MB free
and OOM mid-ingest. The CLI GPU header and the web GPU table now
show a colour-coded "free N%" chip so the pressure is visible
before the alert fires.

**Phase 54.6.287** splits the section-coverage signal by converter
backend — answers "is VLM-Pro actually giving us better heading
detection than pipeline, or is the chunker's regex the bottleneck
regardless?". CLI adds a compact "by backend" row when more than
one backend has contributed chunks; the web modal renders a
per-backend table with a distribution bar per row so the operator
can eyeball the comparison.

**Phase 54.6.288** adds a **stage-timing regression detector** —
compares this week's p95 per pipeline stage against the preceding
7 days. CLI timing panel gets a coloured `Δ±NN%` chip next to p50;
web modal gets a `Δ vs 7d prior` column. `stage_slowdown` alert
fires at ≥+50 %, flagging real slowdowns (vs the ±30 % "show in
panel only" threshold). Catches silent regressions like "embedding
got 2× slower after we flipped `dense_embedder_model`".

**Phase 54.6.289** adds an **Ollama model-swap counter** — records
every transition in the set of loaded Ollama model names into an
in-process ring buffer (same lifetime as the GPU trend). The CLI
gets a compact `↯ swaps N.N/hr` chip next to the Ollama mini-list
and the web modal renders a "Model swap churn" panel with the last
10 add/remove events. `model_thrash` alert fires at ≥15 swaps/hr
over ≥10 min — each swap costs 5-10 s of Ollama cold-load, so
15/hr = ~3 min of pipeline cold-loads per hour, worth surfacing.

**Phase 54.6.313** implements the top-ROI DOI recovery strategies
from `docs/ENRICH_RESEARCH.md` as nine new layers in the `db enrich`
cascade, **plus** a latent bug in the existing Crossref/OpenAlex
title-search where garbage `first_author` strings (`Usuario` /
`Propietario` / `Benutzer` / `ASUS` / `hy` …) were being passed as
search filters and starving the result set. **Measured on the full
217-paper no-DOI subset of the global-cooling corpus**: the old
pipeline (pure Crossref + OpenAlex title search) matched 2 papers;
the new pipeline matched **63 papers** — a **31× improvement** in
recovery rate. Corpus-wide DOI coverage crossed **80.9 %** (653 / 807)
as a result. Breakdown of the 63 hits after all iterations:

- **18 via `crossref+recovered_title`** — `recover_title_from_pdf`
  extracts the largest-font line in the top 40% of page 1 and uses
  it as a search title for rows whose DB `title` was a garbage
  artefact like `iau1200511a`, `Mishev-2.dvi`, or `qjpaper.dvi`.
  Then the existing Crossref title-search resolves it.
- **13 via `crossref+xmp_pdf`** — parses the Adobe PRISM/DC
  namespace in the PDF's XMP packet (Elsevier/Wiley/Springer/IOP
  stamp `prism:doi` at copy-edit time). Near-zero false positives
  with a title-corroboration gate for inherited-XMP cases.
- **11 via `crossref+fulltext_regex`** — regex scan of the first 3
  pages for `10.xxxx/yyyy` patterns, each candidate validated via
  Crossref `/works/{doi}` to reject OCR-mangled strings.
- **6 via `crossref+filename_doi`** — the downloader persists DOI-
  fetched PDFs as `10.xxxx_suffix.pdf`; the filename is parsed and
  validated against Crossref before acceptance. Cheapest + highest-
  precision signal; runs first in the PDF-read layer order.
- **5 via `arxiv_id_in_title`** — fast path for rows whose DB title
  is literally an arXiv stamp (`arXiv:astro-ph/0207637v1  29 Jul
  2002`). The `_ARXIV_OLD` regex needed a `(?:v\d+)?` tweak to
  accept the version suffix — that fix caught 5 rows the cascade
  had been silently dropping. (3 of these subsequently got full
  journal-version DOIs via Crossref in a later pass; 2 carry
  arXiv-issued DataCite DOIs under the 10.48550/arxiv prefix.)
- **2 via bare `crossref`** — the baseline title-search path.
- **3 via `datacite`** — DOIs under 10.48550/arxiv / PANGAEA /
  Zenodo / NASA prefixes.

An optional `--llm-fallback` flag adds one more layer: when every
other source misses and the DB title is still garbage, ship the
PDF's first 3 pages to `settings.llm_fast_model` with a narrow
"return only the title" prompt and retry the title-search cascade.
Slow (~5–15 s per row); off by default because it added 0 matches
on this corpus (the residuals are genuinely not in any DOI index),
but useful on other corpora with mainstream papers having garbled
DB titles.

New layers also wired but low-volume on this corpus: Semantic
Scholar `/graph/v1/paper/search/match` (with process-wide token
bucket for 1-RPS unauth pool + exp. backoff on 429; supports
`SEMANTIC_SCHOLAR_API_KEY`), arXiv title+author search via the Atom
API, and Europe PMC title search for the climate-health /
agricultural-science overlap. The residual 167 no-matches are
foreign-language grey literature (Spanish volcanology), AIAA
conference proceedings, blog posts, and pre-DOI-era works — no DOI
index carries them.

L1 regression test `l1_phase54_6_313_enrich_sources_surface` pins
all new source adapters, the cascade wiring, the
`documents.original_path` JOIN, the source-tagging contract, the
S2 rate-limiter + backoff, and the XMP title-corroboration gate.

All new code is in `sciknow/ingestion/enrich_sources.py`; the
cascade wiring is in `sciknow/cli/db.py::enrich._lookup`. The
enrich row lookup now LEFT-JOINs `documents.original_path` so the
PDF-read layers can read the file. End-of-run prints a per-source
hit breakdown so the user can see which layer pays for itself on
their corpus.

**Phase 54.6.312** is a GUI polish drop addressing five user reports
against the 54.6.309 bibliography / visuals-panel work:

1. **Visual-suggestions panel is now opt-in, persisted, and viewable
   in gallery or list mode.** The right-panel Visuals tab no longer
   re-runs the 1–2 s ranker every time the tab is opened — it loads a
   saved ranking for the current draft if one exists, otherwise shows
   a hint to click the **Rank** button. Subsequent opens serve the
   cached payload instantly. A **Re-rank** button recomputes; **Clear**
   discards the saved blob. Thumbnails are click-to-enlarge (a new
   lightbox modal overlays the reader). A **Gallery ⇄ List** toggle
   flips the pane between a grid of larger thumbnails and the
   original row layout. Backend: `drafts.custom_metadata.visual_suggestions`
   JSONB blob with `{hits, ranked_at, content_hash}`; `GET /api/visuals/
   suggestions` reads the cache, `POST` runs + persists,
   `DELETE` clears. No schema migration.
2. **Bibliography tools: sanity check + sort + renumber + click-to-preview.**
   New **Book → Bibliography tools** modal with two actions.
   `GET /api/bibliography/audit` scans every draft for citations that
   reference a missing source ("broken refs"), sources never cited in
   the body ("orphans"), and duplicate-source-at-different-number
   groups — the typical artefacts of the post-54.6.309 global-renumber
   churn. `POST /api/bibliography/sort` flattens local→global
   numbering INTO the stored draft content (rewrites `[N]` markers AND
   `drafts.sources` so the raw markdown the editor shows matches the
   reader's numbering; idempotent). Clicking any `[N]` in the reader
   now opens a rich preview card (title, authors, year, journal,
   abstract, DOI, open-access URL) via
   `GET /api/bibliography/citation/<N>`; Shift/Cmd/Ctrl+click keeps
   the original scroll-to-source behaviour.
3. **Expand-by-author-references is now a first-class tab in the
   Corpus modal**, alongside Enrich / Expand-citations / Expand-by-
   author / Inbound / Topic / Coauthors. The global-menu entry from
   54.6.309 still works; the new tab gives the feature a stable home
   inside the Corpus surface so users don't have to know it exists in
   a separate menu.
4. **Editor toolbar + named version control.** The markdown editor
   toolbar grows: strikethrough, inline code, H4, bulleted/numbered
   lists, quote, horizontal rule, link insert, inline + display math,
   undo/redo, and two new buttons: **Save as new version…** (prompts
   for an optional label, creates a fresh `drafts` row at
   `version=max+1`, marks it active, stores the label in
   `custom_metadata.version_name`) and **Versions** (opens the History
   panel inline — same route as the Book-menu History item, but
   accessible without leaving edit mode). The Versions panel now
   renders the user-supplied name per row and supports inline rename
   via `POST /api/draft/{{id}}/rename-version`.
5. **Research memo**: `docs/ENRICH_RESEARCH.md` documents the six most
   promising additional DOI/ISBN sources (Semantic Scholar `/match`,
   Europe PMC, DataCite, OpenLibrary, LoC SRU, PDF XMP+footer-regex)
   plus the fuzzy-matching signals the current `db enrich` pipeline
   doesn't yet use. No code change — this is the planning doc for a
   later implementation phase.

**Phase 54.6.309** ships a four-feature book-UX drop:

1. **Global per-book bibliography + "Bibliography" pseudo-chapter.**
   Every draft's local `[1]...[N]` citations are now remapped on
   render to a single, per-book numbering — ordered by first
   appearance when walking chapters in reading order. The sidebar
   grows a synthetic Bibliography chapter at the end (no `Ch.N:`
   prefix, non-deletable) that lists every cited publication once.
   The right-panel Sources list for each draft shows only the
   entries cited in THAT draft, but numbered globally so the
   anchors line up with the body. Implementation lives in
   `sciknow/core/bibliography.py::BookBibliography.from_book` and
   honours the `custom_metadata.is_active` version flag (see #3).
2. **"Visuals" tab in the right banner.** New right-panel segmented-
   control entry runs `retrieval/visuals_ranker.rank_visuals`
   against the open draft's prose (sentence = first 2.5 KB) and
   lists the top-12 figures/tables/charts from the corpus as a
   thumbnail grid. Composite score + "cited" flag (same-paper
   bonus) render inline; an **Insert** button appends
   `![caption](/api/visuals/image/<id>)` + caption to the draft
   body via the edit-in-place textarea (or the edit endpoint when
   the editor is closed). Lazy-loaded on first tab activation.
3. **Draft version browser with scores + active toggle.** The
   History panel (toolbar → History) now renders a richer version
   list: word count, `final_overall` autowrite score, model used,
   review/active tags, and a **Make active** button per row. Clicks
   flip `drafts.custom_metadata.is_active` so the reader and the
   global bibliography both pick that version up; siblings in the
   same `(chapter_id, section_type)` group are cleared in one
   transaction. `POST /api/draft/{id}/activate` is the API.
4. **`db expand-author-refs` — expand corpus by an author's
   references.** New CLI + web flow that picks an existing corpus
   author, aggregates every paper they cited across all their
   corpus works (including self-cites), dedupes by DOI/title, ranks
   by citation frequency, and lands in the existing cherry-pick
   candidates modal before download. Web entry: Corpus menu →
   **Expand by author's references**. Backend: `POST /api/corpus/
   expand-author-refs/preview` + re-uses `/api/corpus/expand-author/
   download-selected` for the download phase.

**Phase 54.6.303** is a two-part fix: (1) the visuals image
endpoint (`/api/visuals/image/{id}`) now also probes the `vlm/`
subfolder when serving figures — MinerU 2.5 VLM-Pro writes
images under `<doc_slug>/vlm/images/...`, but the handler only
tried `<doc_slug>/auto/...` (pipeline-mode layout) and the bare
fallback, so every figure ingested via VLM-Pro 404'd as "image
unavailable" in the web UI's visuals panel; (2) ships
`scripts/ollama-override.conf` — a corrected drop-in for
`/etc/systemd/system/ollama.service.d/override.conf` that fixes
a stray-space typo (`OLLAMA_F LASH_ATTENTION=1` →
`OLLAMA_FLASH_ATTENTION=1`) which was silently disabling flash
attention, and switches `OLLAMA_KV_CACHE_TYPE` from `q8_0` to
`q4_0` to match the 4090 throughput benchmark posted by
`@Punch_Taylor` (43.1 tok/s on Qwen3.6-27B Q4_K_M with
`-fa on --cache-type-k q4_0 --cache-type-v q4_0`). Apply with
`sudo cp scripts/ollama-override.conf /etc/systemd/system/ollama.service.d/override.conf && sudo systemctl daemon-reload && sudo systemctl restart ollama`.
Note on the third complaint ("no charts, 'no visuals found'"):
MinerU 2.5 VLM-Pro doesn't emit a separate `chart` block type
— plots come through as `image` and land in the `figure` kind.
The chart filter showing 0 results is a property of the
backend, not a bug; switch to `PDF_CONVERTER_BACKEND=mineru`
(pipeline mode) and re-run `db extract-visuals --force` if a
distinct chart classification is needed.

**Phase 54.6.302** adds a **per-chapter book-writing velocity panel**.
CLI gets an 8-block progress sparkline inline with the "book" footer
(one block per chapter, height encodes completion %, colour green ≥80 %
/ yellow ≥30 % / dim otherwise) so a brand-new book renders as
`▁▁▁▁▁▁▁▁` and a mostly-finished one as `███▅▇█▇▂`. Web modal gets a
full per-chapter table (number / title / progress bar / words
drafted vs target / version count / last updated) inside the
existing "Active book" panel — identifies stalled chapters at a
glance.

**Phase 54.6.301** adds **retrieval query-latency instrumentation**
— every `hybrid_search.search()` call records into a 60-entry
session ring buffer (same shape as the GPU-trend / model-swap /
preflight buffers). Per-leg timing: embed / dense / sparse / fts /
fuse. Monitor exposes p50 / p95 / avg over the window + a
per-leg breakdown. CLI gets a `search latency p50/p95` row in the
qdrant panel; web modal gets a "Retrieval latency" panel with
per-leg stacked bar + last-10-events table. Completes the
retrieval observability trio (54.6.296 payload indexes +
54.6.299 HNSW tuning + 54.6.301 timing).

**Phase 54.6.299** adds **Qdrant HNSW / quantization drift check**
+ fixes a second dual-embedder sidecar bug found while building it.
The sidecar was being created with Qdrant's default HNSW config
(m=16, ef_construct=100, no quantization) while the prod papers
collection used tuned `m=32 / ef_construct=256 / scalar quantization`
from `.env`. `_ensure_sidecar_exists` now mirrors prod's tuning so
new sidecars match. Monitor adds a per-collection HNSW column to
the web qdrant table, a "hnsw tuning" row in the CLI qdrant panel,
and an info-level `hnsw_drift` alert when any papers-class
collection is on defaults. Small collections (abstracts / wiki /
visuals) are expected to use defaults and never trigger drift.
Existing sidecars keep their old settings until rebuilt.

**Phase 54.6.298** adds a **metadata-enrichment coverage** panel —
per-field missing counts (DOI / abstract / authors / year / title /
journal) across all complete documents. Surfaces a compact
`enrich  doi 73% · abst 30% · auth 86%` row in the CLI corpus
panel and a full Coverage table + stacked bars in the web modal.
`enrichment_gap` info-level alert fires when any actionable field
(excluding year) is missing on >50% of docs, with `sciknow db
enrich` as the suggested fix. Live corpus: 70.5% missing abstract
— big gap for ColBERT / abstracts collection quality.

**Phase 54.6.297** adds a **`BOOK_OUTLINE_MODEL`** per-role override
(mirroring the `BOOK_WRITE_MODEL` / `BOOK_REVIEW_MODEL` /
`AUTOWRITE_SCORER_MODEL` precedent). Both surfaces resolve the
model as `--model` arg > `BOOK_OUTLINE_MODEL` env > `LLM_MODEL`
(the default). The `_grow_sections_llm` call inside
`resize_sections_by_density` now threads the same override so all
three LLM calls in an outline run agree on model choice.

Plus `scripts/bench_outline_model.py` A/B harness: runs the full
3-candidate tournament against two models (default: the current
`qwen3:30b-a3b-instruct-2507-q4_K_M` vs `qwen3.6:27b-dense`),
scores on JSON-validity + chapter count + section variance +
wall-clock, writes JSONL to `data/bench/outline_ab-<ts>.jsonl`,
prints a rec if a non-default model wins the mean scorer.

Monitor's `model_assignments` panel (CLI + web Models tab) now
renders the new `book-out` row alongside `book-wr` / `book-rv` /
`aw-score`.

**Phase 54.6.296** adds **Qdrant payload-index health check** — and
fixes a real bug found while building it: the dual-embedder sidecar
was being created with **zero payload indexes**, so filter pushdown
on the dense leg (`document_id` / `year` / `section_type` / …) was
silently degrading to a full scan — ~100× slower on a 30k-point
collection. `_ensure_sidecar_exists` now creates all six expected
indexes idempotently. Monitor adds a `payload_index_missing` alert,
a "payload indexes" row in the CLI qdrant panel, and an Indexes
column in the web qdrant table that flags any missing index per
collection with a tooltip listing the specific field names.

**Phase 54.6.295** adds a **`--deep` audit** — goes beyond the 292
count-match audit to check:

* UUID identity per doc (prod and sidecar must carry the *same*
  point IDs, not just the same count — a broken write path could
  leave same-size sets with different UUIDs that retrieval would
  silently mix)
* sidecar payload completeness (`document_id` + `chunk_id` +
  `section_type` must be present; missing keys break filter
  pushdown)
* sidecar vector dim sanity (must match `dense_embedder_dim`)
* `chunks.embedding_model` stamp drift (detects staleness after a
  config change)
* untagged-prod classification (the 60 "extra" prod points must
  all be RAPTOR summary nodes — any "stale" bucket flags cleanup)

Full 807-doc deep audit runs in ~13 s; sample mode via
`--uuid-sample 50` for ~2 s. Usage:

```bash
sciknow db audit-sidecar --deep              # full
sciknow db audit-sidecar --deep --uuid-sample 50  # fast sample
```

**Phase 54.6.294** adds a **slow-ingest leaderboard** — top-5 docs
by total ingestion wall-clock with per-stage breakdown. Identifies
outlier PDFs that eat pipeline time. CLI footer block under the
stage-timing panel; web modal renders a "Slow ingest" table with
a stacked stage-duration bar per row + legend (convert / metadata
/ chunking / embedding). Reveals which stage dominates the outlier
cost — typically convert at 95%+ for long scans.

**Phase 54.6.293** wires the **sidecar audit into the monitor** —
`_sidecar_audit_cached` runs the 54.6.292 helper once per 5 min
(module-level TTL) so every snapshot carries the current healthy/
total ratio without paying the 0.6 s Qdrant scroll each time. CLI
gets a `sidecar N/N ✓ age Xs` row in the corpus panel; web modal
gets a "Sidecar integrity" banner with drift counts + DB/prod/
sidecar totals. `sidecar_drift` alert fires when any critical
bucket is non-zero, with `sciknow db audit-sidecar` as the
suggested-fix command.

**Phase 54.6.292** adds a **per-doc sidecar integrity audit** —
`sciknow db audit-sidecar` cross-checks every complete document's
chunk count against both Qdrant collections (prod + dual-embedder
sidecar) and categorises mismatches (sidecar_missing,
sidecar_partial, sidecar_orphan, prod_missing, prod_partial,
prod_orphan). Pipeline-friendly: exit code 0 when clean, 1 when
any critical bucket is non-zero. `--json` for scripting, `--fix-
orphans` to sweep stale sidecar points for already-ingested docs.
Runs in ~0.6 s on the 807-doc corpus (single scroll per collection).
Confirms the dual-embedder state is internally consistent — 807/807
healthy on this corpus post-54.6.285/290/291 work.

**Phase 54.6.291** adds **preflight event history** to the monitor —
every call to `vram_budget.preflight()` records a ring-buffer entry
with reason / need / before-and-after free / which releasers fired /
whether the budget was met. CLI shows a `⚡ preflight N/M tight · X.Y
GB freed` chip in the GPU/Ollama panel; the web modal gets a "VRAM
preflight" panel with the last 10 events. Makes the 54.6.290 subsystem
self-reporting — operators can confirm the cascade is actually firing
and reclaiming the VRAM it promised.

**Phase 54.6.290** adds a **VRAM preflight + releaser registry** —
the **proactive** counterpart to 54.6.286's reactive headroom alert.
Before loading any heavy model the pipeline calls
`vram_budget.preflight(need_mb=…)`, which fires registered releasers
in priority order (Ollama unload → MinerU-VLM subprocess kill →
embedder cache drop) until the requested budget is met. Eliminates
the operational caveat from the 54.6.285 verification where the
dual-embedder + MinerU-VLM stack OOM'd the 3090. Verified with a
full end-to-end re-ingest against the default VLM-Pro converter
starting from 3.5 GB free: ollama unloads, VLM-Pro (vLLM) starts
cleanly, convert completes, the vLLM subprocess is killed before
embed, dual-embedder encodes both collections, and the doc re-
appears in top-5 retrieval results. No OOM, backend stamp
`mineru-vlm-pro-vllm`, counts match across prod/sidecar/DB.

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
