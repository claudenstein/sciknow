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
uv run sciknow refresh                 # full pipeline (including wiki)
uv run sciknow refresh --no-wiki       # skip the hours-long wiki compile
uv run sciknow refresh --dry-run       # preview what would run
```

Every step is idempotent, so `refresh` is safe to run any time. Use
`--no-<step>` to skip individual steps (`--no-ingest`, `--no-cluster`,
`--no-raptor`, `--no-wiki`, etc.).

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

> **Phase 54.6.21:** the per-project `.env.overlay` file (created by `project init`, surfaced in the GUI/CLI) is now actually loaded into Settings. Pre-fix, per-project `LLM_MODEL` overrides etc. were silently no-ops. Plus a batch of audit-driven fixes: Qdrant collection-init fails fast on `EMBEDDING_DIM` mismatch instead of silently corrupting vectors; `_jobs` dict accesses in the SSE thread runners are now lock-protected; `_spawn_cli_streaming` reaps the subprocess in `finally` (no zombies on mid-stream exception); `cleanup-downloads --clean-failed` also nukes orphan paper_summary wiki pages whose source documents were just deleted; `write_active_slug` is atomic; the chunker emits a warning when it produces 0 sections (instead of silently storing a complete-but-empty doc); `consensus_map` logs KG query failures instead of swallowing them.

> **Phase 54.6.22:** `sciknow wiki repair` recovers `wiki_pages` rows whose disk file is missing (common after `project init --from-existing` runs AFTER a `db reset` has wiped the legacy `data/wiki/`). Concept stubs are regenerated cheaply (no LLM); paper_summary + synthesis rows can be `--prune`'d so the next `wiki compile` recreates them from scratch. Also: hybrid-search Qdrant fetches now use a payload include-list (skip 4-5 unused fields × 50 candidates per query) and `wiki list_pages` replaced its per-row LATERAL paper_metadata join with a single bulk SELECT + dict merge.

> **Phase 54.6.35:** `wiki compile` and `wiki extract-kg` are now separate concerns. Inline entity + KG extraction during compile is opt-in via `--with-entities`; off by default because `format=json_schema` calls have model-dependent failure modes (runaway JSON, thinking-token loops) that can routinely dominate compile wall time and produce empty extractions anyway. Run `sciknow wiki extract-kg` separately to populate the knowledge graph — it can retry safely, pick its own model, and be debugged independently when extraction misbehaves. Matches the Unix principle: one command does one thing well.

> **Phase 54.6.23:** five verified bugs from the second-round wiki + ingestion audit. (1) `wiki compile` now runs entity+KG extraction even on the skip path when no triples exist for a doc — pre-fix, a paper whose summary landed but whose entity pass was rolled back was permanently stuck. (2) Citation extraction during ingest now bulk-fetches the DOI→doc_id map in one query instead of one SELECT per reference (N+1 → 1). (3) Empty-sections PDFs now raise and get marked `failed` with a specific `ingestion_error` instead of silently landing as `complete` with zero chunks. (4) Metadata-extraction LLM fallback has an explicit 60s timeout (pre-fix, a hung Ollama would block ingest indefinitely). (5) Concept `source_doc_ids` array de-dupes on append so a doc that updates the same concept twice no longer accumulates duplicate UUIDs.

> **Phase 54.6.40:** `wiki extract-kg` no longer crashes when `qwen2.5:32b-instruct` returns concepts as `{"name": ..., "description": ...}` objects instead of flat slug strings. The `_entity_name` normalizer accepts str/dict/None and routes through it for every entity list + every triple subject/predicate/object. Prompt also rewritten with a WRONG-vs-RIGHT schema demonstration that flipped the model from verbose dicts back to flat slugs. Side-by-side evaluation across five locally installed models (qwen2.5:32b, qwen3:30b-a3b, qwen3.5:27b, qwen3.6:35b-a3b, gemma4:31b) reconfirmed that `qwen2.5:32b-instruct-q4_K_M` is the only model whose content doesn't collapse to empty — all three qwen3-family thinking variants exhaust the 2048 num_predict budget in `<think>` alone (content_len=0, even at a 6000-token budget). Also fixed a stale kwarg in the Phase 54.6.39 SMOKE test (`section_type=` → `section=`) and pinned that smoke to `qwen2.5:32b-instruct` so a thinking `LLM_MODEL` doesn't cause false failures with `num_predict=400`.

> **Phase 54.6.41 / 42 — model bench sweep + swap (2026-04-17).** Added `sciknow bench --layer sweep` — a per-model comparison harness that runs every candidate in `model_sweep.CANDIDATE_MODELS` against the three sciknow LLM roles (extract-kg JSON, wiki compile prose, book autowrite long-form) on a fixed pathological paper (4092d6ad, Nature-Controls-CO2, math-heavy). Emits apples-to-apples metrics: JSON shape (flat/dict/mixed), snake_case predicate %, source-sentence verbatim-in-source %, word count on-target, citation density per 100w, thinking-tag leakage, wall time. Results land as JSONL under `data/bench/<ts>.jsonl`. The 2026-04-17 sweep drove a three-way model swap: **extract-kg** qwen2.5:32b-instruct → `qwen3:30b-a3b-instruct-2507-q4_K_M` (13s vs 108s, 12 triples vs 10, 50% verbatim vs 40%, no thinking hacks); **LLM_FAST_MODEL** (wiki compile + consensus) qwen3:30b-a3b → `qwen3:30b-a3b-instruct-2507-q4_K_M` (official non-thinking twin, eliminates the Phase 54.6.38 `/no_think` appends and tight `num_predict=400` clips); **LLM_MODEL** (book autowrite) qwen3.5:27b → `gemma3:27b-it-qat` (current baseline was 0 words under num_predict=600 — thinking runaway; gemma delivers 4.17 cites/100w at 2× speed). Also cut `existing_slugs` from 300→100 in the extract-kg prompt (the 300-slug list was ballooning the user prompt to 27 KB, which triggered qwen3-instruct's verbose-output mode and blew past any reasonable num_predict cap) and added a truncation-salvage fallback in `_find_json_block` so a mid-JSON cutoff still yields a parseable partial instead of zero triples. Post-swap SMOKE: extract-kg rose from 3 triples in 60s → 11 triples in 11s (5× faster, 3× more content). All model picks are documented in `.env` and `sciknow/core/wiki_ops.py::_extract_entities_and_kg` with sweep metrics inline so the "why this model" answer lives next to the code.

> **Phase 54.6.55 — `BOOK_REVIEW_MODEL` role override wired.** Added a new optional `book_review_model` setting on `Settings`, picked up by `review_draft_stream` when no `--model` CLI flag and no Phase-37 per-section override is set. Defaults to `gemma3:27b-it-qat` in `.env` because it's the one task the unified qwen default lost cleanly (judge 100% vs 71.4%, dims 5/5 vs 3/5 from the 2026-04-17-full quality bench). Precedence is the usual stack: per-call `--model` > per-section override > `BOOK_REVIEW_MODEL` > `LLM_MODEL`. The web book reviewer inherits the default because the web handler calls `review_draft_stream(..., model=model or None)` — when the browser doesn't send an explicit model, the new setting kicks in. Unset `BOOK_REVIEW_MODEL` to route reviews through `LLM_MODEL` again.

> **Phase 54.6.54 — full sweep + quality + infra bench (2026-04-17), `LLM_MODEL` unified on qwen3:30b-a3b-instruct-2507.** Ran the complete three-layer bench against every one of the 13 locally-installed candidates (`bench --layer sweep` → 3 tasks × 13 models, `bench --layer quality` → 7 tasks × 13 models, `bench --layer full` → 21 infra/speed benches). Results in `projects/global-cooling/data/bench/20260417T214833Z.jsonl` (sweep), `20260418T000004Z.jsonl` (quality), `20260418T012247Z.jsonl` (full). **`qwen3:30b-a3b-instruct-2507-q4_K_M` dominated 9/10 tasks on both speed and quality** against the judge-rotated pairwise grid, so `LLM_MODEL` flipped `gemma3:27b-it-qat` → `qwen3:30b-a3b-instruct-2507-q4_K_M` (which was already `LLM_FAST_MODEL`). Both roles now share one model, which eliminates the 5–10s Ollama cold-swap that any pipeline alternating the two roles was paying on every transition (wiki compile → book write → wiki compile, autowrite scorer → writer, etc). Concretely on the four `LLM_MODEL`-bound tasks: **autowrite_writer** 6.9s vs 15s (2.2× faster), citation precision **60% vs 11%** (5×), citation recall 60% vs 14%, judge tie at 100%, faithfulness 0.265 vs 0.210; **ask_synthesize** 14.8s vs 28.9s, judge 100% vs 85.7%; **autowrite_scorer** good=0.87 / bad=0.60, ranks_correctly=1 (faster, higher absolute scores than gemma3's 0.78/0.38, though narrower discrimination gap 0.27 vs 0.40 — both still rank correctly); only loss is **book_review** where gemma3 wins judge 100% vs 71.4% and covers 5/5 dimensions vs 3/5 — the `.env` header documents this and suggests overriding per-call (`sciknow book review <slug> --model gemma3:27b-it-qat`) if review depth matters for the workflow. Notable sweep failures that informed the candidate set going forward: thinking variants `qwen3.5:27b`, `qwen3.6:35b-a3b-q4_K_M`, `supergemma4:26b-uncensored` all produced 0 words on every prose quality task (thinking runaway burns the entire budget in `<think>`); `gemopus4:26b-a4b-q4_K_M` hung on `wiki_consensus` for 2,656s and returned empty (likely context overflow even at ctx=24576); `supergemma4:31b-abliterated` judge win rate 0% on wiki_summary; `nemotron-cascade-2:30b` is the second-place non-thinking option (wiki_summary faithfulness 0.405, judge 88.9%) but is 2–3× slower than qwen across the board and fails the autowrite_scorer JSON schema. Infrastructure numbers from the `full` layer as of 24,519 paper chunks: hybrid retrieval p50 137.6 ms / p95 153.8 ms over 8 queries; rerank p50 1,283 ms (reranker picks a new #1 on 75% of queries); dense×sparse Jaccard 0.063 (healthy complementarity — if they overlapped too much the fusion would be wasted); bge-m3 encode 117.3 chunks/s at batch 16 (dense + sparse in one forward pass); reranker 84.2 pairs/s; `qwen3:30b-a3b-instruct-2507-q4_K_M` tokens/sec 9.9 warm / 8.7 cold on a 24 GB 3090 with an LLM co-resident. `qwen2.5:32b-instruct-q4_K_M` stayed in `CANDIDATE_MODELS` but landed as `not-installed` across all benches — no regression, just a historical reference row that'll re-populate if `ollama pull`ed back in.

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
