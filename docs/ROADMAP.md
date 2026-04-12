# Roadmap

Living document of deferred work, ordered by source. Items here are
**known** open work — either flagged by audits, researched but not
shipped, hardware-gated, or polish noticed during recent phases.
Cross-referenced from the relevant phase commits where applicable.

This doc replaces ad-hoc "planned" sections that were scattered across
`docs/RESEARCH.md` and the auto-memory files. When something here
ships, move it to a Phase commit and delete the entry.

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

**Do NOT** move embedder / reranker to Spark — 3090 has 3.4× more
bandwidth and they're bandwidth-bound at single-batch-size.

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

- [x] **~~Export to PDF (web reader)~~** — shipped in Phase 31 via WeasyPrint. The web reader's export buttons can produce PDF for an individual draft, a chapter, or the full book (`/api/export/{draft,chapter,book}/...pdf`). **Still missing:** PDF / EPUB export from the **CLI** `book export` command (only md/html/bibtex/latex/docx there). EPUB output also still planned (via pandoc, half a day).
- [ ] **Per-book settings page.** Things like `target_chapter_words`, `mineru_vlm_model`, custom_metadata are editable but scattered across the Plan modal, Chapter modal, and `.env`. A single "Book settings" modal would consolidate. **Effort:** half-day.
- [ ] **Autowrite stall investigation.** Phase 24 added the diagnostic logger (`tail -f data/autowrite/latest.jsonl | jq`) but didn't fix any underlying cause. The next stall is a chance to find a concrete root cause. **Effort:** depends on the root cause once it's reproduced.
- [ ] **Per-draft and per-chapter snapshots.** The snapshots table exists for individual drafts; exposing a "snapshot the whole chapter" or "snapshot the whole book" operation would let the user roll back a bad autowrite-all run safely.
- [ ] **Per-section model override.** Right now `settings.llm_model` is global. A user might want to use the flagship model for technical sections and the fast model for brief ones. Per-section meta `model: str | None` would do it. Pairs well with the per-section word target dropdown shipped in Phase 29.

---

## How to use this doc

When picking the next phase:

1. Check if any item here matches the user's current ask — ship it as a numbered Phase commit
2. After shipping, **delete the entry** (don't leave done items as historical clutter — Git history is the historical record)
3. Only add new entries when something is genuinely deferred (researched + decided + not shipped yet)
4. Cross-reference active items into the relevant code with `# TODO(roadmap): <id>` comments only when they require an inline anchor (most don't)

This doc is intentionally short-lived. If it grows past a few hundred lines, that's a sign the team is accumulating work faster than shipping it.
