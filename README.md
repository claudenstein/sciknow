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

### See everything working end-to-end

The GUI has a lot of surfaces; here's the shortest path from a fresh install
to every feature populated (every step is idempotent / resumable):

```bash
# 1. Get a corpus in
uv run sciknow ingest directory ./papers/          # PDFs → Postgres + Qdrant (parallel OK)

# 2. Derive metadata + structure
uv run sciknow db enrich                           # fill missing DOIs via Crossref / OpenAlex
uv run sciknow catalog cluster                     # BERTopic → paper_metadata.topic_cluster
                                                   #            (feeds the Topic map viz)
uv run sciknow catalog raptor build                # hierarchical summary tree
                                                   #            (feeds the RAPTOR sunburst viz)

# 3. Compile the wiki (paper summaries + KG extraction)
uv run sciknow wiki compile                        # summaries + concept pages
uv run sciknow wiki extract-kg                     # backfill knowledge_graph triples
                                                   #            if you're upgrading a wiki
                                                   #            built before Phase 54.6.8

# 4. Start the web reader (the browser UI uses every surface above)
uv run sciknow book serve "My Book"                # http://localhost:8765 by default
```

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

> **Phase 54.6.20:** when an active project is explicitly selected, its `pg_database` and `data_dir` win over `PG_DATABASE` / `DATA_DIR` left in `.env`. The Settings model logs a one-line override warning so a stale `.env` can't silently split state across two projects (DB writes following `.env`, disk writes following the active project — that's how a wiki-compile resume can *appear* to lose work).

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

| Document | Contents |
|---|---|
| **[Architecture](docs/ARCHITECTURE.md)** | System diagram, project structure, database schema, AI models, service layer pattern |
| **[Installation](docs/INSTALLATION.md)** | Setup script, manual install, Ollama tuning, configuration reference, hardware requirements |
| **[CLI Reference](docs/CLI.md)** | Complete command reference for all `sciknow` subcommands |
| **[Ingestion Pipeline](docs/INGESTION.md)** | PDF conversion (MinerU/Marker), metadata extraction, section-aware chunking, embedding |
| **[Retrieval & RAG](docs/RETRIEVAL.md)** | Hybrid search, RRF fusion, reranking, self-correcting RAG, writing assistant |
| **[Book Writing System](docs/BOOK.md)** | Book workflow, autowrite convergence, web reader, export formats, tips |
| **[Operations](docs/OPERATIONS.md)** | Backup/restore, reference expansion, metadata enrichment, citation graph, development notes |
| **[Testing Protocol](docs/TESTING.md)** | The 3-layer smoke harness (`sciknow test`), what each layer covers, how to add new checks |
| **[Benchmarks](docs/BENCHMARKS.md)** | Performance + quality measurement harness (`sciknow bench`): fast/live/llm/full layers, baseline findings, optimization notes |
| **[Research & Innovations](docs/RESEARCH.md)** | All implemented techniques with research basis: BERTopic, GraphRAG, Self-RAG, TreeWriter, Karpathy wiki, consensus mapping |
| **[Multi-Project](docs/PROJECTS.md)** | Per-project isolation (DB + Qdrant + `data/`), project lifecycle CLI + GUI |
| **[Comparative Analysis](docs/COMPARISON.md)** | Phase 45 audit vs karpathy/autoresearch, AI-Scientist, AutoResearchClaw, FARS — what to borrow, what's already better |
| **[Credits](docs/CREDITS.md)** | Open-source projects and research papers that sciknow builds on |

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
