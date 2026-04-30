# sciknow — documentation index

The docs are organized into five subfolders by audience and purpose.

## Top level

- [INSTALLATION.md](INSTALLATION.md) — set up the environment and dependencies.
- [ARCHITECTURE.md](ARCHITECTURE.md) — high-level system overview (read first).

## reference/ — user-facing how-to & feature docs

How to use sciknow, what the commands do, what the GUI does. Stable surface; updated when features change.

- [CLI.md](reference/CLI.md) — every CLI subcommand, with examples.
- [BOOK.md](reference/BOOK.md) — book-writing flow end-to-end.
- [BOOK_ACTIONS.md](reference/BOOK_ACTIONS.md) — per-draft actions (Write, Review, Revise, Verify, Argue, Gaps, …).
- [PROJECTS.md](reference/PROJECTS.md) — multi-project setup (one DB and Qdrant collection set per book).
- [WORKFLOW.md](reference/WORKFLOW.md) — recommended user workflow.
- [OPERATIONS.md](reference/OPERATIONS.md) — lifecycle: init / reset / snapshot / restore.
- [INGESTION.md](reference/INGESTION.md) — how PDFs flow into the corpus.
- [RETRIEVAL.md](reference/RETRIEVAL.md) — hybrid retrieval, reranking, RAPTOR.
- [FORMATTING.md](reference/FORMATTING.md) — pdf-pro / tex-bundle export.
- [TESTING.md](reference/TESTING.md) — the L1/L2/L3 smoke harness.
- [CONCEPT_DENSITY.md](reference/CONCEPT_DENSITY.md) — concept-density-driven planning.
- [LESSONS.md](reference/LESSONS.md) — Phase-32 compound-learning lessons system.
- [CREDITS.md](reference/CREDITS.md) — third-party libraries and acknowledgements.

## roadmap/ — planning, history, design specs

Where things are going, where they've been. Read the V2 roadmap + phase log to catch up; PHASE_56_ROADMAP if you're working on the next big drop.

- [ROADMAP.md](roadmap/ROADMAP.md) — current open work, by source.
- [POST_V2_ROADMAP.md](roadmap/POST_V2_ROADMAP.md) — shipping order after v2.
- [SCIKNOW_V2_ROADMAP.md](roadmap/SCIKNOW_V2_ROADMAP.md) — the v2 plan (llama-server substrate).
- [SCIKNOW_V2_SPEC.md](roadmap/SCIKNOW_V2_SPEC.md) — companion spec to the v2 roadmap.
- [V2_FINAL.md](roadmap/V2_FINAL.md) — v2 cutover notes.
- [PHASE_56_ROADMAP.md](roadmap/PHASE_56_ROADMAP.md) — claim-atomic autowrite (next big drop).
- [PHASE_LOG.md](roadmap/PHASE_LOG.md) — release notes per phase, newest first.
- [ROADMAP_INGESTION.md](roadmap/ROADMAP_INGESTION.md) — ingestion-side roadmap.
- [BENCH_OPTIMIZATION_PLAN.md](roadmap/BENCH_OPTIMIZATION_PLAN.md) — benchmark-driven optimization plan.
- [STRATEGY.md](roadmap/STRATEGY.md) — meta strategy notes.

## research/ — research surveys, design notes, reviews

Standalone reading on specific problems. Surveys map literature to sciknow's choices. Reviews are critiques of external systems (LangChain courses, AutoReason, Memory Palace, DeepScientist).

- [BIBLIOGRAPHY_COVERAGE_RESEARCH.md](research/BIBLIOGRAPHY_COVERAGE_RESEARCH.md) — 849 → ~246 corpus utilization, six families of techniques.
- [SECTION_ENDING_RESEARCH.md](research/SECTION_ENDING_RESEARCH.md) — mid-sentence cutoffs, EOS bias, six families of fixes.
- [RESEARCH.md](research/RESEARCH.md) — master research notes (cumulative, oldest at bottom).
- [ENRICH_RESEARCH.md](research/ENRICH_RESEARCH.md) — DOI / metadata enrichment.
- [EXPAND_RESEARCH.md](research/EXPAND_RESEARCH.md) — `corpus expand` design.
- [EXPAND_ENRICH_RESEARCH_2.md](research/EXPAND_ENRICH_RESEARCH_2.md) — second iteration.
- [KG_RESEARCH.md](research/KG_RESEARCH.md) — knowledge graph extraction.
- [SNAPSHOT_VERSIONING_RESEARCH.md](research/SNAPSHOT_VERSIONING_RESEARCH.md) — draft-versioning + snapshot design.
- [WIKI_UX_RESEARCH.md](research/WIKI_UX_RESEARCH.md) — wiki browser UX.
- [AGENTIC_RAG_COURSE_REVIEW.md](research/AGENTIC_RAG_COURSE_REVIEW.md) — review of an external course.
- [AUTOREASON_REVIEW.md](research/AUTOREASON_REVIEW.md) — review of AutoReason.
- [MEMPALACE_REVIEW.md](research/MEMPALACE_REVIEW.md) — review of Memory Palace.
- [DEEPSCIENTIST_PORTS.md](research/DEEPSCIENTIST_PORTS.md) — DeepScientist ports.
- `huggingface__ml-intern.md`, `inference_servers.md`, `Luce-Org__lucebox-hub.md`, `speculative_decoding.md` — substrate research notes.

## benchmarks/ — measured numbers

Apples-to-apples comparisons of model behaviour across LLM roles.

- [BENCHMARKS.md](benchmarks/BENCHMARKS.md) — bench results by date.
- [BENCH_METHODOLOGY.md](benchmarks/BENCH_METHODOLOGY.md) — how benches are run.
- [COMPARISON.md](benchmarks/COMPARISON.md) — model-vs-model comparisons.
- [WIKI_COMPILE_SPEED.md](benchmarks/WIKI_COMPILE_SPEED.md) — wiki-compile throughput.

## migrations/ — migration guides

One-off step-by-step guides for major substrate changes.

- [MINERU_VLM_PRO_MIGRATION.md](migrations/MINERU_VLM_PRO_MIGRATION.md) — MinerU 2.0 pipeline → MinerU 2.5-Pro VLM.
