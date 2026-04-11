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

- [ ] **Dead `/api/chapters/reorder` endpoint** at `web/app.py:704`. Registered but no JS calls it. Either remove it or wire chapter drag-and-drop in the sidebar (the section drag-and-drop in Phase 26 is a natural template). **Effort:** 1-2 hours for the GUI side.
- [ ] **Fragile `WHERE`-clause f-string** in the catalog query (`web/app.py` ~1575). Current conditions are hardcoded so it's not exploitable, but it's a code smell that would bite if someone ever adds a user-controlled filter. Replace with explicit named bindings. **Effort:** 30 minutes.
- [ ] **`onclick="..."` pattern fragility.** A lot of inline handlers with interpolated strings. Phase 22 escaped the IDs but the pattern itself is harder to reason about than `addEventListener`. Refactor to event delegation. **Effort:** half-day.

---

## 2. Research runners-up (2026-04 lit sweep)

From `docs/RESEARCH.md` §512. The 2026-04 literature sweep produced
five candidates that didn't make the first ship batch (Phases 7–12).
In priority order:

- [ ] **CARS-adapted chapter moves** (Swales 1990 + Yang & Allison 2003) — 5-move scaffold (Orient → Tension → Evidence → Qualify → Integrate). Prompt-only, ~20 lines in `rag/prompts.py`. **Top linguistics runner-up.**
- [ ] **LongCite-style sentence citations** (THUDM 2024) — sentence-level grounding with span match for ALCE-compatible `citation_f1`. Pairs naturally with the existing hedging_fidelity scoring + OVERSTATED verdict from Phase 11. **Top CS runner-up.**
- [ ] **Toulmin scaffolds** for paragraphs the planner labels `Tension` (claim / data / warrant / qualifier / rebuttal). Linguistics runner-up #2.
- [ ] **MADAM-RAG** (Wang et al. COLM 2025) for paragraphs the argument-mapper flags as contradiction-heavy. CS runner-up.
- [ ] **Soft RAPTOR clustering.** Phase 12's RAPTOR build computes the GMM `proba` matrix but only uses `argmax`. Soft assignment would let chunks contribute to multiple cluster summaries above a probability threshold. **Polish, not a major lift.**

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

- [ ] **Cross-chapter section drag-and-drop.** Phase 26 deliberately restricted to within-chapter because moving a draft between chapters needs to also update `drafts.chapter_id` — different risk profile. Worth doing with a confirm prompt. **Effort:** 1-2 hours.
- [ ] **Chapter drag-and-drop reordering.** The endpoint (`/api/chapters/reorder`) exists, just needs the JS — same template as Phase 26. **Effort:** 1 hour.
- [ ] **Proper modal for the autowrite mode picker** (Phase 28). Currently uses `prompt("s/r/i")` which is ugly. A small 3-button modal would be cleaner. **Effort:** 1 hour.
- [ ] **Log rotation for `data/autowrite/`.** Phase 24 creates a file per run; they accumulate forever. A simple "keep last 50" sweep on `_AutowriteLogger` startup would be enough. **Effort:** 30 minutes.
- [x] **~~Autowrite ETA in heartbeats.~~** Shipped in Phase 30 — the persistent task bar shows ETA when `target_words` is known and tokens are flowing (`remaining / tps`). The polling architecture from Phase 32.5 now keeps it in lockstep with the server-side counter.
- [ ] **Keyboard shortcuts** in the web reader — `Ctrl+S` to force save, `Ctrl+K` to focus search, arrow keys to nav between sections, `Esc` to close modals. **Effort:** half-day.
- [ ] **Build-tag version string** in the template. Phase 25 had the "hard-refresh to see the new chevron" issue. If every render included a version string in the `<title>` or as a DevTools console line, users could instantly tell whether their browser has stale JS. **Effort:** 30 minutes.

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
