# Roadmap

Living document of deferred work, ordered by source. Items here are
**known** open work — either flagged by audits, researched but not
shipped, hardware-gated, or polish noticed during recent phases.
Cross-referenced from the relevant phase commits where applicable.

This doc replaces ad-hoc "planned" sections that were scattered across
`docs/RESEARCH.md` and the auto-memory files. When something here
ships, move it to a Phase commit and delete the entry.

> **Companion:** [`docs/POST_V2_ROADMAP.md`](POST_V2_ROADMAP.md) is the
> shipping-order view (v2.0.0 → v2.1 → v2.2 → Spark → data-gated). It
> *references* this doc's open items rather than duplicating them, so
> when something ships here, the Post-v2 doc tracks the milestone.

---

## 1. Deferred QA findings (Phase 22 audit)

The Phase 22 QA agent flagged 15 issues in `sciknow/web/app.py`. The
high-impact ones (XSS, job leak, draft delete) shipped in Phase 22.
These three were deferred:

- [x] **~~Dead `/api/chapters/reorder` endpoint.~~** Wired in Phase 33 — chapter title bars are now draggable; the drop handler POSTs to this endpoint. No longer dead code.
- [x] **~~Fragile `WHERE`-clause f-string~~** in the catalog query — shipped in Phase 41. Both `/api/catalog` and `/api/kg` rewritten to the always-bind pattern: every optional filter is bound as the real value or NULL, each WHERE clause gated with `(:param IS NULL OR …)`. The SQL is now fully static (no f-string, no `.join()`, no dynamic assembly) which makes the injection-vector question decidable at lint time instead of review time.
- [x] **~~`onclick="..."` pattern fragility~~** — shipped in Phase 42 (surgical scope: the ~20 interpolated handlers). New pattern: every such button carries `data-action="kebab-name"` + `data-*` attrs; one document-level click listener looks the action up in an `ACTIONS` registry and invokes it with the element. Static handlers (no interpolation → no fragility) left alone for a future CSP pass. Removes the XSS vector-by-construction and makes handlers breakpointable / auditable in one place.

---

## 2. Research runners-up (2026-04 lit sweep)

From `docs/RESEARCH.md` §512. The 2026-04 literature sweep produced
five candidates that didn't make the first ship batch (Phases 7–12).
In priority order:

- [x] **~~CARS-adapted chapter moves.~~** Shipped in Phase 34. `rhetorical_move` field added to the tree_plan prompt schema alongside `discourse_relation`. The 5-move vocabulary (orient/tension/evidence/qualify/integrate) is rendered as `[orient]`, `[tension]`, etc. in the writer's paragraph plan block. The planner labels each paragraph with both a PDTB-lite discourse relation (how it connects to the previous paragraph) AND a CARS rhetorical move (what function it serves in the section's argument). No schema change — pure prompt addition. **Top linguistics runner-up from the 2026-04 lit sweep.**
- [x] **~~LongCite-style sentence citations.~~** Shipped in Phase 34. "Sentence-level citation grounding" rule added to WRITE_V2_SYSTEM: every sentence that makes a factual claim MUST end with at least one `[N]` citation — no more paragraph-end citation clustering. Scorer prompt updated to check groundedness at the sentence level ("A paragraph with 4 claim sentences and 1 citation at the end has groundedness ~0.25, not 1.0"). **Top CS runner-up from the 2026-04 lit sweep.**
- [x] **~~Toulmin scaffolds.~~** Shipped in Phase 34. When a paragraph is marked `[tension]` (CARS move), the writer prompt includes Toulmin scaffold guidance: CLAIM → DATA → WARRANT → QUALIFIER → REBUTTAL. Conditional — only fires for tension paragraphs; other moves are unaffected. **Linguistics runner-up #2.**
- [x] **~~MADAM-RAG (prompt-only core).~~** Shipped in Phase 34 as MADAM-RAG-lite. The tree planner can now tag paragraphs with a `contradiction` object: `{for: ["[1]"], against: ["[4]"], nature: "..."}`. The writer prompt renders this as explicit pro/con source guidance with a ⚡ CONTRADICTION indicator, telling the writer to present both sides fairly (and use Toulmin structure for tension paragraphs). This is the prompt-engineering core of Wang et al.'s MADAM-RAG (COLM 2025) without the multi-agent debate overhead — the full multi-agent version (2-3 extra LLM calls per contradiction paragraph) is deferred to the DGX Spark roadmap where the extra compute is affordable.
- [x] **~~Soft RAPTOR clustering.~~** Shipped in Phase 34. The GMM `proba` matrix is now used: chunks with membership probability ≥ `RAPTOR_SOFT_THRESHOLD` (default 0.15) contribute to secondary clusters in addition to their argmax primary. This means a chunk about "solar forcing AND ocean heat content" can appear in BOTH the solar-forcing cluster summary AND the ocean-heat cluster summary, improving recall for queries that approach the overlap from either angle. Controlled by `settings.raptor_soft_threshold`; set to 0 to disable. Only affects `catalog raptor build`; existing RAPTOR nodes are unaffected until the next rebuild.

**Do NOT relitigate** (rejected with documented reasons in `docs/RESEARCH.md` §526):
HyDE, Self-RAG/CRAG (fine-tuned), Dense X / Propositional Retrieval,
GraphRAG global, Late Chunking (Jina), full RST tree parsing, full
Centering Cb/Cf/Cp machinery, FActScore as an online method, ALCE
benchmark.

---

## 3. Hardware-gated (DGX Spark)

Activate when the NVIDIA DGX Spark (GB10, 128GB unified LPDDR5X)
arrives. Tracked separately because they all need >24GB unified
memory and would be wasted on the existing 3090.

- [ ] **Phase B — LLM host routing.** Split `ask question` (3090, fast) vs `book write` / `ask synthesize` (Spark, 70B+ model). ~50 lines in `sciknow/rag/llm.py`. Simplest of the three. **Start here when the box arrives.**
- [ ] **Phase C — vLLM backend on Spark** for batch LLM workloads (catalog cluster, `db export --generate-qa`, `book argue` / `book gaps`). Big throughput win on batch operations.
- [ ] **Phase E — QLoRA fine-tuning of `bge-reranker-v2-m3`** on the user's own climate corpus. Specialised reranker for the user's domain.
- [ ] **Phase 47.S1 — CycleReviewer swap-in as the autowrite scorer.** `WestlakeNLP/CycleReviewer-ML-Llama3.1-8B` (fits the 3090 alone but not alongside our 27B writer). On Spark, co-reside with the flagship writer and route scoring requests to it. Paper claims 26.89% MAE vs individual human reviewers on OpenReview. Direct replacement for the current LLM-prompt scorer — we inherit the gain without training. Tracked in `docs/COMPARISON.md` Appendix D. **Preconditions**: Spark + mixed-workload scheduling.
- [ ] **Phase 47.S2 — Iterative DPO on our own KEEP/DISCARD verdicts.** Phase 32.6 autowrite telemetry already captures `(overall_pre, overall_post, action)` per iteration. A post-Spark job exports `data/preferences/<book>.jsonl` in `{prompt, chosen, rejected}` shape, trains a LoRA on the writer with 2k+ validated pairs (3 epochs). See `docs/RESEARCH.md` §21 Layer 4. **Preconditions**: Spark + ≥ 2k validated pairs (we're accumulating).
- [ ] **Phase 47.S3 — fast-detect-gpt pre-publish gate.** Bao et al. method, MIT; `Qwen2.5-1.5B` as both scoring and reference. Today's 3090 has the VRAM if the writer is evicted; works better on Spark where the writer stays hot. Calibrate threshold against ~100 known-human climate papers. Gate as a warning badge, not a block.

**Do NOT** move embedder / reranker to Spark — 3090 has 3.4× more
bandwidth and they're bandwidth-bound at single-batch-size.

### 3a. CycleResearcher — pending evaluation when Spark arrives

Three concrete ports from CycleResearcher (arXiv:2411.00816, ICLR 2025).
Watchlist: `zhu-minjun/Researcher`. All three map onto Spark's unified
memory; none are feasible on the 3090 alongside the current 27B writer.
In effort-impact order:

1. **CycleReviewer as autowrite scorer** (2 days + Spark). See Phase 47.S1 above. Direct drop-in against the existing scorer interface; inherits the published 26.89% MAE reduction without training on our end.
2. **9-block NeurIPS rubric in the writer prompt** (2 days, no Spark needed — ✅ **already shipped** as `sciknow book ensemble-review` in Phase 46.C with N=3 stance-rotated reviewers + meta-reviewer).
3. **Iterative DPO on sciknow's own preferences** (2 weeks + Spark). See Phase 47.S2 above. Already tracked in `docs/RESEARCH.md` §21 Layer 4 — CycleResearcher's RL loop is the current-gen reference implementation of the pattern.
4. **fast-detect-gpt pre-publish gate** (3 days, GPU). See Phase 47.S3 above.

The common thread: without Spark we're stuck running scoring on the
same GPU as the writer, which means either (a) evicting the writer
(cache-cold on resume) or (b) CPU-fallback on the scorer (50× slower —
see `docs/BENCHMARKS.md` contention finding).

---

## 4. Polish from recent phases

Things noticed while shipping Phases 17–28 but deferred for scope.
Mostly small.

- [x] **~~Cross-chapter section drag-and-drop.~~** Shipped in Phase 33. Drag a section from one chapter onto a section in a different chapter; a confirm dialog shows "Move section X from Ch.3 to Ch.7?". On confirm: (1) `PUT /api/draft/{id}/chapter` moves the draft, (2) source chapter's sections list has the slug removed, (3) target chapter's sections list gets the slug inserted at the drop position. The existing within-chapter reorder path is unchanged.
- [x] **~~Chapter drag-and-drop reordering.~~** Shipped in Phase 33. Chapter title bars are HTML5-draggable; the drop handler POSTs the new `chapter_ids` order to the existing `/api/chapters/reorder` endpoint and rebuilds the sidebar. Same visual language as Phase 26 section drag-drop (accent-colored top/bottom border on hover).
- [x] **~~Proper modal for the autowrite mode picker.~~** Shipped in Phase 33. The triple-`prompt()` UX (max iterations, target score, s/r/i mode) replaced by a proper `autowrite-config-modal` with numeric inputs + three styled mode buttons (Skip / Rebuild / Resume). Mode section only shows when applicable (all-sections run with existing drafts).
- [x] **~~Log rotation for `data/autowrite/`.~~** Shipped in Phase 33. `_rotate_old_logs` runs at `_AutowriteLogger.__init__` — keeps the most recent 50 `.jsonl` files, deletes older ones. Static method, no external deps.
- [x] **~~Autowrite ETA in heartbeats.~~** Shipped in Phase 30 — the persistent task bar shows ETA when `target_words` is known and tokens are flowing (`remaining / tps`). The polling architecture from Phase 32.5 now keeps it in lockstep with the server-side counter.
- [x] **~~Keyboard shortcuts.~~** Shipped in Phase 33. Esc (close modals, existing), Ctrl+S (force save in editor), Ctrl+K (focus search bar), Ctrl+E (toggle editor), ← / → (prev/next section in sidebar, guarded from inputs/textareas), D (dashboard), P (plan modal). All letter shortcuts only fire when focus is NOT in an input/textarea/select/contentEditable.
- [x] **~~Build-tag version string.~~** Shipped in Phase 33. `_BUILD_TAG` computed at import time from `git rev-parse --short=7 HEAD` (falls back to UTC timestamp). Visible in the browser tab title (`Book — SciKnow [abc1234]`) and in the DevTools console (`[sciknow] web reader loaded · build abc1234`). No more stale-JS guessing.

---

## 4b. Compound learning — Layers 1-6 (Phase 32.6 follow-on)

Layer 0 (autowrite telemetry tables + persistence helpers) shipped in
Phase 32.6 — see `docs/RESEARCH.md` §21 for the full architecture and
the literature review. Layers 1-6 are the actual learning passes that
read from those tables. Each is independently shippable; later layers
depend on earlier layers having data.

- [x] **~~Layer 1 — Episodic memory store (lessons).~~** Shipped in Phase 32.7 (commit pending). `autowrite_lessons` table with 1024-dim bge-m3 embeddings; producer (`_distill_lessons_from_run`) is called inline at the tail of `_finalize_autowrite_run` and uses the FAST model (per the MAR critique — different from the writer/scorer); consumer (`_get_relevant_lessons`) is called from `_autowrite_section_body` before `write_section_v2` and injects top-5 lessons by `importance × recency_decay × cosine_similarity` (Generative Agents formula) into the writer system prompt as a *Lessons from prior runs* block. **Validation gate:** `l1_phase32_7_lessons_layer1` (static) + `l2_phase32_7_lessons_roundtrip` (end-to-end against PG, with a real similarity-ranking assertion). **Still TBD:** the win-condition bench (`book autowrite-bench --runs 5` before/after) — needs ~10-20 real autowrite runs first to populate the lessons table.
- [x] **~~Layer 2 — Useful chunk retrieval boost.~~** Shipped in Phase 32.8. `_apply_useful_boost` runs inline at the tail of `hybrid_search.search()` (no nightly batch job needed — the SQL aggregation is sub-millisecond against the partial index on `autowrite_retrievals(chunk_qdrant_id) WHERE was_cited = true`). Boost formula: `score *= 1 + 0.15 × log2(1 + useful_count)`, controlled by `settings.useful_count_boost_factor`. The `useful_count` field propagates through `SearchCandidate` → `SearchResult` so downstream consumers (writer prompt context, GUI tooltip) can see how many past drafts cited each chunk. Verified against live PG with `l2_phase32_8_useful_boost_roundtrip` (insert 10 fake citations, re-search, assert score formula matches and rank moves up).
- [ ] **Layer 3 — Heuristic distillation (ERL-style).** Once ~50 runs have accumulated, periodically cluster Layer 1 lessons by embedding similarity and prompt the LLM to extract a *heuristic* per cluster (a generalized strategic principle that applies across many sections). Store in a `heuristics` table; prepend unconditionally to the writer prompt. Layer 1's raw lessons stay retrieved per-section. **Effort:** ~2 weeks. **Dependency:** Layer 1 + ≥50 runs of accumulated history.
- [x] **~~Layer 4 — Iterative DPO preference dataset (data only).~~** Shipped in Phase 32.9. Migration 0013 added `pre_revision_content` and `post_revision_content` columns to `autowrite_iterations`; `_autowrite_section_body` captures both at every revision; `_export_preference_pairs` walks the table and writes standard `{prompt, chosen, rejected}` JSONL via the new CLI command `sciknow book preferences export`. **Both KEEP and DISCARD verdicts produce pairs** (the original roadmap only counted KEEPs — DISCARDs become inverse pairs where chosen=pre, rejected=post, doubling the data yield). Filters: `--min-score 0.7` (drop pairs with both sides low), `--min-delta 0.02` (drop noise), `--require-approval` (human-in-the-loop bias gate via `drafts.custom_metadata.preference_approved`). **Still TBD:** the web UI "approve this KEEP" button — for now, approval has to be set via direct DB update or a future settings page. Verified end-to-end with `l2_phase32_9_dpo_export_roundtrip` (KEEP + inverted DISCARD + low-score filter).
- [x] **~~Layer 5 — Style fingerprint extraction.~~** Shipped in Phase 32.10. New module `sciknow/core/style_fingerprint.py` computes per-book metrics (median sentence length, median paragraph length, citations per 100 words, hedging rate from a BioScope-derived cue list mirroring the Phase 7 prompt rule, top sentence-initial transitions, average words per draft). Source data: drafts with status in `{final, reviewed, revised}` — i.e. user-touched, not the autowrite default `drafted`. Persisted to `books.custom_metadata.style_fingerprint` (existing JSONB column — no migration needed). Consumer wired into `_autowrite_section_body` via `get_style_fingerprint(book_id)` → `format_fingerprint_for_prompt(fp)` → `write_section_v2(style_fingerprint_block=...)` which renders into the new `{style_fingerprint_section}` placeholder in `WRITE_V2_SYSTEM`. Two new CLI commands: `sciknow book style refresh "Book"` recomputes from current draft state, `sciknow book style show "Book"` displays the persisted fingerprint. Verified end-to-end with `l2_phase32_10_style_fingerprint_roundtrip`.
- [ ] **Layer 6 — Domain LoRA on the writer (DGX Spark required).** When the Spark arrives and Layer 4 has accumulated ~2k validated preference pairs, LoRA-tune the current SOTA writer model with DPO on the user's preferences. Per Wolfe 2024, 2k pairs + 3 epochs is enough for meaningful gains. Output: a `qwen-sciknow:32b` (or successor) model. Becomes the new default `LLM_MODEL`; original stays for ablation. **Effort:** ~1 week post-Spark-arrival. **Dependency:** Layer 4 + DGX Spark + ≥2k pairs.

**Validation harness:** all six layers use the same Track A measurement methodology as Phase 13 — `book autowrite-bench --runs 5` on a control chapter, before/after each layer ships. The mean shift must exceed the baseline std for the win to be real.

**Anti-patterns to avoid** (full list in `docs/RESEARCH.md` §21): training a reward model from scratch instead of DPO; storing raw iteration text in the lessons table; prepending all past lessons to every prompt (the ExpeL anti-pattern that ERL specifically calls out); optimizing the writer with the same scorer as the supervisor without a human gate; trying to fine-tune on the 3090; conflating "compound learning" with "infinite context window".

---

## 5. Feature gaps

Things obviously missing for a book-writing system but never explicitly
asked for. Listed in rough priority order.

- [x] **~~Export to PDF (web reader) + CLI PDF/EPUB~~** — web shipped in Phase 31 via WeasyPrint; CLI EPUB via pandoc shipped in Phase 40 (`sciknow book export --format epub`); CLI PDF also in `book export`. All export paths now cover md/html/pdf/epub/latex/docx/bibtex.
- [ ] **Per-book settings page.** Things like `target_chapter_words`, `mineru_vlm_model`, custom_metadata are editable but scattered across the Plan modal, Chapter modal, and `.env`. A single "Book settings" modal would consolidate. **Effort:** half-day.
- [ ] **Autowrite stall investigation.** Phase 24 added the diagnostic logger (`tail -f data/autowrite/latest.jsonl | jq`) but didn't fix any underlying cause. The next stall is a chance to find a concrete root cause. **Effort:** depends on the root cause once it's reproduced.
- [x] **~~Per-draft and per-chapter snapshots.~~** Shipped in Phase 54.6.75 as `sciknow book snapshot / snapshots / snapshot-restore` wrapping the existing web endpoints. Non-destructive (inserts new versions, never overwrites).
- [x] **~~Per-section model override.~~** Shipped — `_get_section_model` at `sciknow/core/book_ops.py:298` pulls a per-section `model` key from the section's metadata and applies it in both write and autowrite paths, falling back to `settings.llm_model` when unset.

---

## 6. 2026-04-18 improvement sweep (Phase 54.6.68+)

Second-generation improvements sourced from a fresh audit after Phases
54.6.40–54.6.67 shipped (wiki UX, plan-modal rework, section density,
chart extraction). Items here are **not in RESEARCH.md's rejected list**
and don't duplicate already-shipped phases. Ordered by expected impact
per hour of effort.

### 6a. Tier 1 — ship next, foundational

- [x] **~~#3 — Retrieval-quality bench (`b_retrieval_recall`).~~** Shipped in Phase 54.6.69. `sciknow/testing/retrieval_eval.py` provides `generate_probe_set(n)` and `b_retrieval_recall()` returning MRR@10, Recall@1/10, NDCG@10 against a synthetic probe set persisted under `projects/<slug>/data/bench/retrieval_queries.jsonl`. Baseline on global-cooling: MRR 0.514.
- [x] **~~#5 — Tokenizer-aware chunk budgets.~~** **False alarm from the audit.**
  The chunker is already tokenizer-aware (uses `tiktoken.cl100k_base` at
  `sciknow/ingestion/chunker.py:22-24`). cl100k_base ≈ XLM-R within ~30%,
  and with target=512 / cap=8192 there's ~16× headroom — no real-world
  overflow possible even on LaTeX-dense sections. Swapping to bge-m3's
  exact XLM-R tokenizer would be mechanically correct but gains nothing
  measurable; pay no diff on this one.
- [x] **~~#7 — Citation marker → chunk alignment post-pass.~~** Shipped in Phase 54.6.71 as `sciknow/core/citation_align.py` + CLI `book align-citations`. Conservative remap: only when the claimed chunk entails < 0.5 AND the top chunk beats by ≥ 0.15. Tunable via `--low-threshold` / `--win-margin`. Also wired into the web draft toolbar in 54.6.97.
- [x] **~~#9 — Citation-graph retrieval boost.~~** Shipped in Phase 54.6.70 as `_apply_cocite_boost` in `retrieval/hybrid_search.py`. Bib-coupling / co-citation boost applied inside the RRF-fusion stage; controlled by `settings.cocite_boost_factor` (default 0.1). `cocite_count` propagates through `SearchCandidate` for downstream UI.
  Not in rejected list (bib-coupling / co-citation are classical IR,
  not GraphRAG-global). **Effort:** half-day.

### 6b. Tier 2 — meaningful research upgrade

- [x] **~~#1 — Vision-LLM auto-captioning for figures + charts.~~** Shipped in Phase 54.6.72 as `sciknow/core/visuals_caption.py` + CLI `db caption-visuals`. Bulk captioning (54.6.89) wired into `sciknow refresh` step 9 with `qwen2.5vl:7b` as the default bulk VLM (60% of 32b's quality, ~35× faster — 9,831 figures captioned in ~5h). Migration 0026 added `ai_caption` / `ai_caption_model` / `ai_captioned_at` columns.
- [x] **~~#1b — VLM sweep harness.~~** Shipped in Phase 54.6.74 as `sciknow/testing/vlm_sweep.py` + `sciknow bench --layer vlm-sweep`. Fair-fight sampling per model (54.6.85 methodology). Judge win-rate verdict: qwen2.5vl:32b 93.3%, qwen2.5vl:7b 60%, minicpm-v:8b 35.6%, llama3.2-vision:11b 8.9%.

- [x] **~~#2 — Structured table parsing.~~** Shipped in Phase 54.6.106.
  Migration 0028 adds `table_title / table_headers / table_summary /
  table_n_rows / table_n_cols / table_parsed_at` to `visuals`. CLI
  `sciknow corpus parse-tables` runs the fast LLM on MinerU's HTML and
  stores the structured output; Visuals modal table cards render the
  parsed block (title + summary + column list + shape) above the raw
  HTML when available. Wired into `refresh` as step 11. **Follow-up
  completed** in 54.6.109: `sciknow corpus embed-visuals` now picks the
  best embedding text per kind (tables use `table_summary`, figures/
  charts use `ai_caption`, equations use the natural-language
  paraphrase) — every visual kind is now semantically searchable.
- [x] **~~#6 — Coverage-based autowrite termination.~~** Shipped in Phase
  54.6.79. `sciknow/core/plan_coverage.py` computes NLI coverage of
  atomic plan bullets against the draft; folded into the autowrite
  scores dict as a new `plan_coverage` dimension. The existing
  weakest-dimension logic picks up the gap; when coverage is lowest,
  the revision instruction is overridden to name missed bullets
  explicitly. Fails silently on empty plans / unavailable NLI.
- [x] **~~#8 — Claim-atomization for offline verify.~~** Shipped in
  Phase 54.6.83 as a standalone `sciknow book verify-draft` CLI, NOT
  wired into the quality bench (doing so would multiply NLI cost 2-3×
  for marginal signal). `sciknow/core/claim_atomize.py`: heuristic-
  first atomizer (regex splits on `;`, em-dashes, clear `, and` clause
  boundaries), LLM fallback for long multi-conjunction sentences
  (>30 words with ≥2 conjunctions the heuristic missed). `verify_draft`
  batch-scores each sub-claim's max-over-sources NLI entailment, flags
  per-sentence mixed_truth (≥1 supported AND ≥1 unsupported sub-claim —
  the failure mode single-sentence NLI averages away). Reads from
  `drafts.sources`; read-only.
- [ ] **#4 — bge-m3 LoRA fine-tune on synthetic question-chunk pairs.**
  Uses #3's synthetic queries as contrastive-loss training data.
  Distinct from ROADMAP §3 Phase E (which is the *reranker*). Keep the
  base checkpoint as fallback. **Preconditions:** #3 shipped. **Effort:** 1–2 days.

### 6c. Tier 3 — architecturally useful

- [x] **~~#10 — Paper-type classification + retrieval weighting.~~**
  Shipped in two parts. Part 1 (Phase 54.6.80): migration 0027 adds
  `paper_type` / `paper_type_confidence` / `paper_type_model` to
  `paper_metadata`. New `sciknow/core/paper_type.py` classifier (LLM
  one-pass on abstract + first 2000 chars + bibliographic metadata)
  covers 8 categories. CLI `sciknow corpus classify-papers`. Part 2
  (Phase 54.6.81): `_apply_paper_type_weight` in hybrid_search
  multiplies rrf_score by a per-type weight (opinion=0.4 →
  peer_reviewed=1.0); defaults OFF behind `PAPER_TYPE_WEIGHTING=true`
  until the backfill completes on a meaningful corpus fraction.
- [x] **~~#11 — Equation natural-language paraphrase embedding.~~**
  Shipped in Phase 54.6.78 (paraphrase generator) + 54.6.82 (Qdrant
  embed). `sciknow/core/equation_paraphrase.py` +
  `sciknow corpus paraphrase-equations` CLI write paraphrases to
  `visuals.ai_caption`; `sciknow corpus embed-visuals` (`retrieval/visuals_search.py`)
  embeds captions + paraphrases into a per-project `<slug>_visuals`
  Qdrant collection. Wired into `sciknow refresh` step 11.
- [ ] **#12 — Compound learning Layer 3** (already in §4b, still pending,
  blocked on ≥50 autowrite runs).
- [x] **~~#13 — Chapter / book snapshot + restore.~~** Shipped in
  Phase 54.6.75. Three new CLI commands (`sciknow book snapshot` /
  `snapshots` / `snapshot-restore`) wrapping the existing Phase-38
  web endpoints. Dry-run supported on restore. Non-destructive —
  inserts new draft versions, never overwrites.

### 6d. Tier 4 — UX / observability

- [ ] **#14 — Consolidated Book settings modal** (already in §5, half-day).
- [x] **~~#15 — GPU-time ledger per draft / chapter / book.~~** Shipped
  in Phase 54.6.76. New module `sciknow/core/gpu_ledger.py` rolls up
  `autowrite_runs.started_at / finished_at / tokens_used` at every
  scope. CLI `sciknow book ledger <book> [-c N | -d ID]` +
  `/api/ledger/{book,chapter,draft}/{id}` endpoints. Read-only; no
  new tables.
- [x] **~~#16 — MCP server interface for the corpus.~~** Shipped in
  Phase 54.6.77. `sciknow mcp-serve` speaks Model Context Protocol
  over stdio. Four tools exposed: `search_corpus`, `ask_corpus`,
  `list_chapters`, `get_paper_summary`. Configure agent with
  `{"command": "uv", "args": ["run", "sciknow", "mcp-serve"]}`.
  Added `mcp` dep.

### Ingestion / enrichment research proposals

Research-track proposals for `sciknow refresh` + `sciknow ingest` —
discovery, PDF conversion, metadata, chunking, embedding, visuals,
citations, KG, topics, RAPTOR, wiki compile, and cross-cutting
observability — live in a dedicated doc so this file stays focused
on "what's next to ship":

→ [`docs/ROADMAP_INGESTION.md`](./ROADMAP_INGESTION.md)

43 proposals grouped by pipeline stage + cross-cutting concerns, each
with expected win, cost, and priority verdict (**Ship** / Investigate
/ Defer / Gated). Section 6 names the five items to land first if
the user green-lights it. Update it the same way as this doc: move
shipped proposals into a PHASE_LOG entry, delete from ROADMAP_
INGESTION.md.

### Rejected-adjacent items (from the 2026-04-18 audit)

These look similar to Tier 1/2 items but were already addressed or
explicitly rejected; **do NOT relitigate**:

- HyDE (RESEARCH.md §526: step-back prompting dominates it on our stack).
- Self-RAG / CRAG as runtime with fine-tuned reflection tokens
  (RESEARCH.md §526: prompt-only emulations lose the gains).
- GraphRAG global community summaries (RESEARCH.md §526: RAPTOR gets
  80% of the benefit at 20% of the cost on our chunk index; bib-coupling
  in #9 is *not* GraphRAG global — different axis).
- FActScore as online method (RESEARCH.md §526: the proposal in #8 is
  specifically the offline evaluator slice, which the same doc keeps open).
- Dense X / Propositional Retrieval, Late Chunking — all in §526.

---

## How to use this doc

When picking the next phase:

1. Check if any item here matches the user's current ask — ship it as a numbered Phase commit
2. After shipping, **delete the entry** (don't leave done items as historical clutter — Git history is the historical record)
3. Only add new entries when something is genuinely deferred (researched + decided + not shipped yet)
4. Cross-reference active items into the relevant code with `# TODO(roadmap): <id>` comments only when they require an inline anchor (most don't)

This doc is intentionally short-lived. If it grows past a few hundred lines, that's a sign the team is accumulating work faster than shipping it.
