# Comparative Analysis — sciknow vs. Upstream Auto-Research Systems

[&larr; Back to README](../README.md)

---

Phase 45 audit. Four systems were studied in depth against sciknow's current surface. The watchlist (`sciknow watch`) is pre-seeded with all four so subsequent releases are surfaced automatically.

| System | Repo / URL | Maturity | Focus |
|---|---|---|---|
| **karpathy/autoresearch** | [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch) | Demo (6 files) | Single-GPU LLM pretraining agent harness |
| **SakanaAI/AI-Scientist (v1 + v2)** | [github.com/SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) | Research prototype | End-to-end ML paper pipeline (idea → experiment → write → review) |
| **aiming-lab/AutoResearchClaw** | [github.com/aiming-lab/AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) | Active framework (522 files, 2,699 tests) | 23-stage pipeline → conference-template paper |
| **analemma FARS / openfars** | [analemma.ai/blog/introducing-fars/](https://analemma.ai/blog/introducing-fars/) + [github.com/open-fars/openfars](https://github.com/open-fars/openfars) | Blog-post teaser (framework closed; outputs public) | 100-paper autonomous ML-research deployment |

## What each system has that sciknow doesn't

| Capability | AI-Scientist | AutoResearchClaw | FARS | karpathy |
|---|:-:|:-:|:-:|:-:|
| Runs its own experiments (edits + executes Python, generates figures) | ● | ● | ● | ● |
| Writes LaTeX, compiles PDF | ● | ● | ○ | ○ |
| External citation verification (arXiv/Crossref/OpenAlex) | ○ | ● | ○ | ○ |
| Two-stage "where + what" citation insertion loop | ● | ○ | ○ | ○ |
| NeurIPS-style ensemble self-review with numeric rubric | ● | ● | ○ | ○ |
| HITL checkpoint gates between pipeline stages | ○ | ● | ○ | ○ |
| Cross-run "lessons" store with time decay | ○ | ● (MetaClaw) | ○ | ○ |
| Multi-agent debate (hypothesis, review, analysis) | v2 only | ● | ○ | ○ |
| Tree-search over drafts / experiments | v2 (BFTS) | ○ | ○ | ○ |
| Fixed wall-clock budget per step | ○ | ○ | ○ | ● |
| Keep-or-reset binary verdict (simple) | ○ | ○ | ○ | ● |
| `program.md` style skill file | ○ | ○ | ○ | ● |
| Multi-source literature discovery fan-out | ○ | ● | ○ | ○ |
| Per-project filesystem reference cache | ○ | ● | ● | ○ |
| Conference-template output (NeurIPS/ICML/ICLR) | ● | ● | ○ | ○ |
| Typed plan.json contract between stages | ○ | ● | ● | ○ |
| Negative-results-friendly output format | ○ | ○ | ● | — |

## What sciknow has that none of the four has

| Capability | Why it matters |
|---|---|
| **Real PDF ingestion pipeline** (MinerU 2.5 + Marker fallback) | None of the four parses PDFs — they rely on API-returned abstracts or their own generated code's outputs. sciknow can ingest 100k-chunk private corpora; AI-Scientist cannot tell you what a PDF said. |
| **Dense + sparse + FTS hybrid retrieval with RRF + cross-encoder** | AutoResearchClaw's "knowledge base" is Markdown files in `kb_root/` with no embeddings. AI-Scientist has no retrieval at all — it queries Semantic Scholar per-paper on demand. sciknow's signal-complementarity score (Phase 44 bench: mean Jaccard 0.012) is empirically validated. |
| **Local-first, Ollama-native, no API required** | AI-Scientist README: *"we do not advise using any model significantly weaker than GPT-4"*. AutoResearchClaw markets Ollama support as a footnote; FARS is API-only. sciknow runs every path on a single 3090 with qwen3:30b-a3b / qwen3.5:27b and never leaves the machine. |
| **Multi-project corpus isolation** (Phase 43) | Per-project DB + Qdrant collections + data dir. None of the four has this — each treats "the project" as a single scratch directory. |
| **Claim verification against your own corpus, not the internet** | AutoResearchClaw verifies citations against external APIs (good); AI-Scientist doesn't verify claims at all. sciknow verifies claims against the retrieved evidence that was used to write them — orthogonal to their approach. |
| **Benchmark harness with empirical signal complementarity + rerank displacement measurement** (Phase 44) | None of the four ships a quantitative benchmarking surface. Phase 44 data gave sciknow numbers for §2/§10 of RESEARCH.md. |
| **Phase 32.6 autowrite telemetry tables** (runs, iterations, retrievals, lessons) | MetaClaw is the only equivalent and it's a Markdown log, not a relational store. |
| **Style fingerprint extraction + per-chapter/section model override (Phase 37)** | Nobody else can mix e.g. `qwen3:30b` for narrative sections and `qwen3.5:27b` for methods within a single writing project. |

## Capabilities ranked by "we should build this"

### High-leverage, low-risk, fits the stack

**A. Two-stage citation insertion loop (AI-Scientist)** — ★★★★★
Clean port. LLM identifies a location in the draft and emits a query → local hybrid retrieval → LLM picks from top-k → structured edit inserts `[N]` + updates the references table. Strictly better than the current "write, then hope citations are correct" pattern because placement becomes auditable. Fits as a new generator in `core/book_ops.py` emitting `citation_proposal` / `citation_selected` / `citation_inserted` events. Budget-bounded (`num_cite_rounds`). 2–3 days of work.

**B. External citation-verification report (AutoResearchClaw)** — ★★★★★
Cascade: arXiv ID → Crossref/DataCite DOI → OpenAlex + Semantic Scholar title search (Jaccard ≥ 0.80 / 0.50–0.80 / < 0.50). Emit a JSON report marking each inline citation VERIFIED / SUSPICIOUS / HALLUCINATED. Complementary to sciknow's existing claim-verification (which checks against the corpus). Lives naturally as `sciknow book verify-citations <draft>`. 1–2 days.

**C. Ensemble self-review with NeurIPS-style rubric (AI-Scientist)** — ★★★★☆
Replace `book review` with: N=3–5 parallel reviews @ T=0.75 using Ollama, a meta-reviewer pass, numeric averaging over rubric fields (Soundness, Presentation, Contribution, Overall, Confidence, Decision). Persist the JSON alongside drafts. Pairs naturally with existing corpus-grounded claim verification. Mitigation for known positivity bias: use AI-Scientist's `reviewer_system_prompt_neg` ("if unsure, reject") variant. 2 days.

**D. Cross-run "lessons" store with recency-weighted retrieval (AutoResearchClaw MetaClaw)** — ★★★★☆
Phase 32.6 already has `autowrite_lessons`. Wire up a post-run extractor (3–5 bullets distilled from the per-iteration trajectory of each completed run), store with an embedding, retrieve top-K by `importance × recency × similarity_to_section_plan` before the next autowrite in the same project. Pure prompt engineering on top of existing infra. 2 days.

**E. HITL checkpoint gates on `book autowrite` (AutoResearchClaw)** — ★★★☆☆
Introduce stage-boundary checkpoints between outline → draft → review → revise, with state written to `projects/<slug>/data/autowrite/<run_id>/`. `--resume-from outline/draft/review` replays from the last gate. Phase 43 already gives us the per-project state dir; this is plumbing only. 2–3 days.

**F. Fixed wall-clock budget (karpathy)** — ★★★☆☆
Add `--time-budget 300s` to `book autowrite`, `ask synthesize`, `wiki compile`. The loop halts at wall time, not iteration count. Makes runs comparable across model sizes and hardware (relevant for the incoming DGX Spark). 1 day.

**G. `program.md` style per-operation skill files (karpathy)** — ★★★☆☆
One `programs/<operation>.md` per long-form op (`autowrite.md`, `lit_sweep.md`, `argue.md`) — a committed, diffable natural-language policy that the generator reads on startup. Makes research procedures fork-able and audit-able. 2 days.

### Medium-leverage, fits conceptually

**H. Typed `task_plan.json` contract between generator stages (FARS + AutoResearchClaw)** — ★★★☆☆
Today `core/book_ops.py` passes prose between stages. A structured `{index, category, title, description, status, summary, dependencies}` list would make chapter-level long-horizon work resumable and inspectable. Composes with (E). 3 days.

**I. Per-project filesystem reference cache mirroring FARS's layout** — ★★☆☆☆
Export per-book working set to `projects/<slug>/references/<paper-slug>/{meta/bibtex.txt, meta/metadata.json, sections/*.md}`. Gives Ollama agents a greppable working set without hitting the DB. Also makes a project portable across sciknow installs. 2 days. Lower ranked because the gain is marginal — sciknow's Qdrant+Postgres retrieval is already faster than grepping filesystem.

**J. `drafts.tsv` + git-branch experiment ledger per chapter (karpathy)** — ★★☆☆☆
One commit per attempted draft on a dedicated `autowrite/<chapter-slug>` branch, plus a flat TSV (commit, scores, keep/discard, description). Gives us Karpathy's "wake up to 100 experiments" pattern. Independent of the scorer — binary keep/reset instead of multi-score thresholding is much simpler, and the bench finding that autowrite plateaus at round 1 suggests simple-but-many is probably the right shape. 3 days.

### Low-leverage or out of scope

**K. Run actual experiments (all three competitors)** — out of scope.
sciknow's premise is that *the user* is the researcher running the experiments. Building an experiment runner is a different product — and a hugely risky one (the AI-Scientist README warns explicitly about LLM-written code running `pip install` and spawning long processes).

**L. Agentic tree search over drafts (AI-Scientist v2 / AIDE)** — defer.
Worth revisiting once (D) + (J) are in. The "experiments as nodes in a tree" insight is powerful but requires a search/prune policy that we don't have data to design yet.

**M. Multi-agent debate for hypothesis/review (AutoResearchClaw)** — defer.
Ensemble review (C) captures most of the value at a fraction of the complexity. Revisit after (C) is in and we have data on where ensemble disagreement clusters.

**N. LaTeX compile loop (AI-Scientist, AutoResearchClaw)** — defer.
sciknow already has Phase 30 multi-format export (LaTeX included) via direct template rendering. The AI-Scientist-style `pdflatex → bibtex → chktex → fix-loop` machinery is overkill when the source is markdown that we convert once.

**O. Conference template output (AI-Scientist, AutoResearchClaw)** — partial overlap.
sciknow's Phase 30 export already produces publication-grade artifacts. If we build (B), the `verification_report.json` becomes the missing piece that makes a sciknow draft directly submittable. No dedicated work needed.

## Suggested next phase (Phase 46)

Based on the audit, the highest-value Phase 46 would bundle **A + B + C + D** — together they turn sciknow's draft output from "grounded prose" into "submission-grade artifact with audit trail":

```
Phase 46 —— "Auditable Scientific Writing"
  A. Two-stage citation insertion loop    (core/book_ops.py: new generator)
  B. External citation verification        (new: core/citation_verify.py + `book verify-citations`)
  C. Ensemble NeurIPS-rubric self-review   (core/book_ops.py: replace review_section)
  D. MetaClaw-style lessons retrieval      (core/book_ops.py + autowrite_lessons already in DB)
```

Estimated: 8–10 focused days. Every step is prompt + Python, no new DB migrations (autowrite_lessons landed in Phase 32.6 / migration 0012).

## What the watchlist will tell us next

`sciknow watch list --check` prints the current HEAD + stars + new-commits delta for every seeded repo:

- **karpathy/autoresearch** — 71k stars, last push 2026-03-26. Inactive relative to the others; mainly a pedagogical demo.
- **SakanaAI/AI-Scientist** — 13k stars, last push 2025-12-19. v2 is the active branch; track for BFTS improvements.
- **aiming-lab/AutoResearchClaw** — 11k stars, last push 2026-04-10. Most-active of the four; weekly deltas expected. This is the one most likely to ship new primitives worth porting.
- **open-fars/openfars** — 31 stars, last push 2026-02-26. FARS's framework is closed; the re-implementation is an OpenAI-only reference. Useful if we ever want to copy the `task_plan.json` schema.
- **WecoAI/aideml** — 1.2k stars, last push 2026-02-12. Source of AI-Scientist v2's tree-search. Worth watching if we pick up (L) later.

Periodic check cadence: weekly is probably enough; none of these are security-critical. `sciknow watch list --check` from a `cron` entry or a manual tick at the start of each planning session covers it.

## Appendix — Seed watchlist rationale

See `sciknow/core/watchlist.py:SEED_REPOS`. Each entry carries a one-line note explaining what idea we already stole or plan to steal. The point is that **sciknow's debt to these upstream projects is now captured in a machine-readable form** — not a blog post, not a CREDITS line, but a structured list that evolves with the field.
