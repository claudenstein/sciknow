# CLI Reference

[&larr; Back to README](../README.md)

---

## `sciknow db`

```bash
sciknow db init              # Run migrations + init Qdrant collections
sciknow db reset             # Wipe everything and re-initialise (use before a full re-ingest)
sciknow db stats             # Show paper/chunk counts and status breakdown
sciknow db refresh-metadata  # Re-run metadata extraction for papers with poor-quality metadata
sciknow db enrich            # Find DOIs for papers that don't have one (Crossref + OpenAlex search by title)
sciknow db expand            # Discover and download cited papers to grow the collection
sciknow db backup            # Back up the full collection to a portable archive
sciknow db restore           # Restore a backup on a new machine
sciknow db tag-multimodal    # Tag chunks with tables/equations for filtered search
```

---

## `sciknow ingest`

```bash
# Ingest a single PDF
sciknow ingest file paper.pdf

# Ingest all PDFs in a directory (recursive by default)
sciknow ingest directory ./papers/

# Non-recursive
sciknow ingest directory ./papers/ --no-recursive

# Parallel ingestion — spawns N worker subprocesses, each with its own
# Marker/MinerU + bge-m3 models. On a 24 GB 3090 with an LLM resident,
# keep --workers 1. Raise to 2 only when the LLM is off-GPU.
sciknow ingest directory ./papers/ --workers 2

# Re-ingest a paper that was already processed
sciknow ingest file paper.pdf --force
```

---

## `sciknow search`

```bash
# Basic query — hybrid search + reranking, returns top 10
sciknow search query "sea surface temperature reconstruction methods"

# Show full chunk text for each result
sciknow search query "DNA methylation cancer" --show-content

# Filter by year range and section type
sciknow search query "aerosol radiative forcing" --year-from 2010 --year-to 2023 --section methods

# Filter by domain
sciknow search query "stellar evolution" --domain astrophysics

# Filter by topic cluster (set topic clusters with `sciknow catalog cluster` first)
sciknow search query "solar forcing" --topic "Solar Irradiance"

# Skip reranking (faster, uses RRF scores directly)
sciknow search query "climate sensitivity" --no-rerank --top-k 20

# Show relevance scores
sciknow search query "protein folding" --show-scores

# Control candidate pool size (more candidates = more thorough, slower reranking)
sciknow search query "galaxy formation" --candidates 100 --top-k 15
```

---

## `sciknow ask`

```bash
# Answer a question using RAG (retrieves top 8 chunks, reranks, streams answer)
sciknow ask question "What are the main mechanisms of aerosol radiative forcing?"

# Use a specific model (default: LLM_MODEL from .env)
sciknow ask question "Explain stellar nucleosynthesis" --model mistral:7b-instruct-q4_K_M

# Filter by year and section
sciknow ask question "How is SST reconstructed from proxies?" \
    --year-from 2000 --section methods

# Self-correcting RAG: evaluates retrieval quality, retries if poor, checks grounding
sciknow ask question "What drives the Atlantic Meridional Overturning Circulation?" --self-correct

# More context chunks (default: 8)
sciknow ask question "Compare tree-ring proxy methods" --context-k 15

# Hide sources
sciknow ask question "What drives the Atlantic Meridional Overturning Circulation?" --no-sources

# Synthesise findings across papers on a topic
sciknow ask synthesize "solar activity and climate variability"

# Synthesis with domain filter and more context
sciknow ask synthesize "galaxy formation and feedback mechanisms" \
    --domain astrophysics --context-k 15

# Draft a paper section grounded in your library
sciknow ask write "aerosol-cloud interactions" --section introduction
sciknow ask write "stellar population synthesis" --section methods --domain astrophysics

# Save the draft to the database (optionally link to a book chapter)
sciknow ask write "solar forcing mechanisms" --section introduction --save
sciknow ask write "ocean heat content trends" --section results --save \
    --book "Global Cooling" --chapter 3

# Filter retrieval to a topic cluster
sciknow ask question "How does solar activity affect climate?" --topic "Solar Irradiance"
sciknow ask synthesize "ocean-atmosphere heat exchange" --topic "Ocean Heat Transport"
```

---

## `sciknow catalog`

```bash
# Overview: papers by year (bar chart), top journals, metadata source quality
sciknow catalog stats

# Paginated table of all papers with optional filters
sciknow catalog list
sciknow catalog list --author Zharkova --sort year
sciknow catalog list --from 2015 --to 2023 --journal "Nature"
sciknow catalog list --title "solar cycle" --limit 20 --page 2

# Full record for one paper — by DOI, arXiv ID, or title fragment
sciknow catalog show 10.1093/mnras/stad1001
sciknow catalog show "solar magnetic field eigenvectors"

# Export the catalog to CSV or JSON
sciknow catalog export --output catalog.csv
sciknow catalog export --format json --output catalog.json
sciknow catalog export --author Zharkova --output zharkova.csv

# Assign topic clusters using BERTopic (embedding-based — fast, deterministic)
# Uses bge-m3 embeddings already in Qdrant: UMAP → HDBSCAN → c-TF-IDF → LLM naming
sciknow catalog cluster                     # fast embedding-based clustering (seconds)
sciknow catalog cluster --dry-run           # preview clusters without saving
sciknow catalog cluster --min-cluster-size 3  # allow smaller clusters
sciknow catalog cluster --rebuild           # re-cluster ALL papers from scratch

# Legacy LLM-batch clustering (slower, kept for backward compatibility)
sciknow catalog cluster-llm                 # old approach: LLM batches of 50 papers

# List all clusters with paper counts
sciknow catalog topics

# RAPTOR hierarchical retrieval tree (Sarthi et al., ICLR 2024)
# Adds a tree of LLM-summarised cluster nodes on top of the existing
# chunk index. Pure additive layer — NO re-ingest, NO wiki recompile.
# After build, the writer's retriever automatically returns a mix of
# fine chunks AND mid-level summaries, with the reranker deciding.
sciknow catalog raptor build              # first build (uses LLM_FAST_MODEL by default)
sciknow catalog raptor build --dry-run    # preview cluster sizes without writing
sciknow catalog raptor build --rebuild    # wipe all level >= 1 nodes and rebuild
sciknow catalog raptor build --max-levels 3 --min-cluster-size 4
sciknow catalog raptor build --model qwen3.5:27b   # main model for higher-quality summaries
sciknow catalog raptor stats              # show node counts per RAPTOR level
```

---

## `sciknow wiki`

Karpathy-style compiled knowledge wiki. Instead of RAG on raw chunks every time, the wiki pre-synthesizes papers into interconnected pages that grow incrementally.

```bash
# Build wiki (only new papers by default — safe to re-run anytime)
# Shows live progress: bar + tok/s + ETA + running totals
# Per paper: 1 LLM call for summary + 1 structured output call for entities+KG
# Uses LLM_FAST_MODEL by default (wiki is exploration, not final writing)
# Override with --model if needed. Book writing always uses the main model.
sciknow wiki compile                    # only papers without wiki pages yet
sciknow wiki compile --doc-id abc123    # compile one paper
sciknow wiki compile --rebuild          # recompile everything from scratch

# Query the compiled wiki (not raw chunks)
sciknow wiki query "what is total solar irradiance?"
sciknow wiki query "how do cosmic rays affect cloud formation?"

# Generate a synthesis overview page
sciknow wiki synthesize "solar forcing and climate"

# Browse wiki pages
sciknow wiki list                       # all pages
sciknow wiki list --type concept        # only concept pages
sciknow wiki show total-solar-irradiance

# Health checks
sciknow wiki lint                       # structural: broken links, orphans, stale
sciknow wiki lint --deep                # + LLM contradiction detection

# Knowledge graph — explore entity-relationship triples
sciknow wiki graph "solar forcing"              # direct connections
sciknow wiki graph "total solar irradiance" --depth 2   # 2-hop traversal

# Consensus mapping — agreement/disagreement across corpus
sciknow wiki consensus "solar forcing and climate"
sciknow wiki consensus "cosmic ray cloud nucleation"
```

Wiki pages are stored as human-readable markdown in `data/wiki/` (git-friendly), indexed in PostgreSQL, and embedded in Qdrant for search. When new papers are ingested, relevant concept pages are automatically updated.

---

## `sciknow book`

The book system organises writing projects with an iterative, coherent pipeline:

```
plan → outline → (per chapter) sentence plan → write → review → revise → verify → export
```

Each step is grounded in retrieved papers and maintains cross-chapter coherence via a persistent book plan and auto-generated chapter summaries.

```bash
# ── Project setup ─────────────────────────────────────────────────────────

sciknow book create "Global Cooling"
sciknow book create "Solar Cycle Mechanisms" --description "Overview of solar variability"
sciknow book list                       # List all books
sciknow book show "Global Cooling"      # Chapters, drafts, gaps, progress

# Add chapters manually or auto-generate with LLM
sciknow book chapter add "Global Cooling" "The Maunder Minimum"
sciknow book outline "Global Cooling"   # LLM proposes 6-12 chapters from your library

# ── Book plan (thesis + scope) ────────────────────────────────────────────

sciknow book plan "Global Cooling"
sciknow book plan "Global Cooling" --edit   # Regenerate

# ── Writing ───────────────────────────────────────────────────────────────

sciknow book write "Global Cooling" 2 --section methods
sciknow book write "Global Cooling" 3 --section results --plan       # sentence plan first
sciknow book write "Global Cooling" 1 --section introduction --verify # claim verification
sciknow book write "Global Cooling" 5 --section conclusion --plan --verify --expand  # all flags

# ── Review + revise loop ──────────────────────────────────────────────────

sciknow book review 3f2a1b4c
sciknow book revise 3f2a1b4c -i "expand the section on solar cycles with more evidence"
sciknow book revise 3f2a1b4c         # applies saved review feedback automatically

# ── Evidence analysis ─────────────────────────────────────────────────────

sciknow book argue "solar activity is the primary driver of 20th century warming"
sciknow book argue "cosmic rays modulate cloud cover" --save

# ── Gap analysis ──────────────────────────────────────────────────────────

sciknow book gaps "Global Cooling"
sciknow book gaps "Global Cooling" --no-save   # informational only

# ── Autowrite (autonomous convergence loop) ───────────────────────────────

sciknow book autowrite "Global Cooling" 1 --section introduction
sciknow book autowrite "Global Cooling" 3 --section methods --max-iter 5 --target-score 0.90
sciknow book autowrite "Global Cooling" 3 --section all            # all sections of a chapter
sciknow book autowrite "Global Cooling" --full --max-iter 3 --target-score 0.85  # full book
sciknow book autowrite "Global Cooling" --full --auto-expand       # fetch papers for gaps

# Phase 17 — explicit per-section word target (overrides chapter target)
sciknow book autowrite "Global Cooling" 3 --section all --target-words 2500

# Phase 28 — resume mode: load an existing FINISHED draft and continue
# iterating on it instead of skipping. Refuses to resume from partial drafts
# (writing_in_progress / iteration_*_revising / placeholder) — use --rebuild
# to overwrite those.
sciknow book autowrite "Global Cooling" 3 --section methods --resume

# Force-overwrite existing drafts (mutually exclusive with --resume; rebuild wins)
sciknow book autowrite "Global Cooling" 3 --section all --rebuild

# ── Drafts inspection (Phase 13 — Track A measurement) ────────────────────

sciknow book draft scores 3f2a1b4c                              # iteration-by-iteration score history
sciknow book draft compare 3f2a1b4c 9d8e7f6a                    # side-by-side from persisted scores
sciknow book draft compare 3f2a1b4c 9d8e7f6a --rescore          # re-run scorer + verifier (current rubric)

# ── DPO preference dataset export (Phase 32.9 — Layer 4) ──────────────────

# Export every preference pair (chosen, rejected) from autowrite history
# in standard JSONL ready for HuggingFace TRL or any DPO trainer.
sciknow book preferences export                                # all books → data/preferences/all_books.jsonl
sciknow book preferences export "Global Cooling"               # one book
sciknow book preferences export "Global Cooling" --stats       # show counts only, don't write
sciknow book preferences export "Global Cooling" -o /tmp/dataset.jsonl
sciknow book preferences export "Global Cooling" --min-score 0.75 --min-delta 0.05  # tighter filters
sciknow book preferences export "Global Cooling" --no-discard  # KEEP verdicts only (conservative)
sciknow book preferences export "Global Cooling" --require-approval  # human-in-the-loop gate

# ── Style fingerprint (Phase 32.10 — Layer 5) ─────────────────────────────

# Extract a per-book writing style fingerprint from approved drafts
# (status in final/reviewed/revised). Injected into the writer prompt
# as a style anchor on the next autowrite run. Pure Python — no LLM cost.
sciknow book style refresh "Global Cooling"   # recompute from current state
sciknow book style show "Global Cooling"      # display persisted fingerprint

# ── Autowrite variance bench (Phase 13) ──────────────────────────────────

sciknow book autowrite-bench "Global Cooling" 3 overview --runs 5
sciknow book autowrite-bench "Global Cooling" 3 overview --no-cove        # disable Chain-of-Verification
sciknow book autowrite-bench "Global Cooling" 3 overview --no-step-back --no-plan

# ── Chapter management ───────────────────────────────────────────────────

sciknow book chapter add "Global Cooling" "The Maunder Minimum"
# (Reorder, rename, delete, and per-section editing all live in the web reader,
#  not the CLI — open `sciknow book serve` and use the chapter modal.)

# ── Adopt orphan section (Phase 25) ───────────────────────────────────────

# Re-classify a draft whose section_type no longer matches any current
# template slug. Adds the slug to the chapter's sections list.
sciknow book adopt-section "Global Cooling" 3 historical_context

# ── Web reader ────────────────────────────────────────────────────────────

sciknow book serve "Global Cooling"
sciknow book serve "Global Cooling" --port 9000

# ── Export (CLI: md / html / bibtex / latex / docx) ──────────────────────

sciknow book export "Global Cooling" -o manuscript.md                    # Markdown
sciknow book export "Global Cooling" --format html -o book.html          # HTML
sciknow book export "Global Cooling" --format bibtex -o refs.bib         # BibTeX
sciknow book export "Global Cooling" --format latex -o book.tex          # LaTeX
sciknow book export "Global Cooling" --format docx -o book.docx          # DOCX
```

> **PDF export** (Phase 31) is **only available in the web reader**, not in `sciknow book export`. The web reader exports `txt / md / html / pdf` via WeasyPrint at `/api/export/{draft,chapter,book}/{id}.{ext}`. CLI PDF/EPUB via Pandoc is on the roadmap (`docs/ROADMAP.md`).

See [Book Writing System](BOOK.md) for the full workflow guide.

---

## `sciknow draft`

```bash
sciknow draft list
sciknow draft list --book "Global Cooling"
sciknow draft list --page 2
sciknow draft show 3f2a1b4c
sciknow draft delete 3f2a1b4c
sciknow draft delete 3f2a1b4c --yes
sciknow draft export 3f2a1b4c
sciknow draft export 3f2a1b4c --output chapter2_intro.md
```

---

## `sciknow db export`

```bash
# Export all chunks with metadata as JSONL
sciknow db export --output dataset.jsonl

# Only chunks with >= 100 tokens
sciknow db export --output dataset.jsonl --min-tokens 100

# Generate synthetic Q&A pairs using Ollama (~5-10 s per chunk, slow)
sciknow db export --output qa_dataset.jsonl --generate-qa

# Limit export size
sciknow db export --output sample.jsonl --generate-qa --limit 50
```

Output format:
```json
{"title": "...", "year": 2021, "section": "methods", "doi": "...", "content": "..."}
```
With `--generate-qa`:
```json
{"title": "...", "year": 2021, "section": "methods", "doi": "...", "content": "...",
 "question": "...", "answer": "..."}
```

---

## `sciknow project` (multi-project)

Each project has its own PostgreSQL database (`sciknow_<slug>`), its own
Qdrant collections (`<slug>_papers`, `<slug>_abstracts`, `<slug>_wiki`),
and its own `projects/<slug>/data/` directory. Resolution precedence:
`--project <slug>` flag &rarr; `SCIKNOW_PROJECT` env var &rarr;
`.active-project` file at repo root &rarr; legacy `default` (pre-Phase-43
single-tenant layout using `data/` + the `sciknow` DB unchanged).

```bash
# Lifecycle
sciknow project init <slug>                     # fresh empty project (DB + collections + dir + migrations)
sciknow project init <slug> --from-existing     # one-shot migration of the legacy install into a slot
sciknow project init <slug> --dry-run           # preview steps without executing

# Inspect
sciknow project list                            # all projects + active marker + health
sciknow project show [slug]                     # details (defaults to active project)

# Switch
sciknow project use <slug>                      # writes .active-project
sciknow --project <slug> <any subcommand>       # one-shot override (no .active-project change)
SCIKNOW_PROJECT=<slug> sciknow ...              # env var equivalent

# Destructive
sciknow project destroy <slug> --yes            # drop DB + collections + data dir

# Portable
sciknow project archive <slug>                  # bundle to archives/<slug>-<ts>.skproj.tar, drop live
sciknow project archive <slug> --keep-live      # snapshot only, keep live state
sciknow project archive <slug> -o /path/out.tar # custom output path
sciknow project unarchive <archive-file>        # restore (becomes active)
```

The web reader exposes list / show / use / init / destroy through a Projects
modal in the action toolbar. See [Multi-project design](PROJECTS.md) for the
full architecture.
