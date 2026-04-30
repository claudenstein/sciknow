# Phase 56 — Corpus-Driven Book Extraction

**Branch**: `v2-llamacpp`
**Revert anchor**: tag `autowrite-stable-pre-phase56` at commit `87c35d2`
**Status**: roadmap, ready to implement
**Companions**: [`docs/research/BIBLIOGRAPHY_COVERAGE_RESEARCH.md`](../research/BIBLIOGRAPHY_COVERAGE_RESEARCH.md), [`docs/research/SECTION_ENDING_RESEARCH.md`](../research/SECTION_ENDING_RESEARCH.md)

---

## North star

Replace the current "user picks outline → writer fills in to a target word count" with **"corpus determines book structure → claims emerge from clusters → writer transcribes claims with grounded evidence, dropping the unsupported ones."**

Length is **not a goal** in Phase 56 — it's a derived quantity. Section length = (number of well-supported claims) × (sentences per claim). Chapter length = sum of sections. Book length = sum of chapters. We deliberately defer length-control mechanisms (`min_tokens`, paragraph quotas, multi-pass length expansion) to Phase 57; they're a separate optimization that fights against the natural shape Phase 56 produces.

What we optimize:
1. **Optimal coverage** — every well-supported claim from the corpus reaches the reader, every cited paper supports a specific claim.
2. **No overfitting** — claims with no evidence above the NLI entailment floor are dropped, not padded with hand-waving prose. Citations are gated.
3. **No underfitting** — uncited corpus subclusters that genuinely warrant attention become new sections via the coverage feedback loop.
4. **Clean endings** — no mid-word truncation, no orphan `[` citation openers. Defensive post-pass on every writer output.
5. **Citation precision ≥ 0.92** (mean NLI entailment of cited chunk vs. cited sentence). The current ~0.50 baseline is the headline metric this phase moves.

What we do NOT optimize in this phase:
- Total word count per section / chapter / book.
- Wall-clock speed of writing (Phase 56 is several × slower than v55 — by design).
- Prose stylistic continuity beyond entity-bridge stitching (Phase 57).

---

## Architecture overview

```
                    ┌────────────────────────────────────────────────┐
                    │  CORPUS  (chunks + RAPTOR tree + metadata)     │
                    └────────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌─────────────────────────────────────────────────┐
                  │ 56.A   Topic tree                                │
                  │   cluster.centroid, .summary, .papers,          │
                  │   .depth_of_coverage, .scope_relevance(book)    │
                  └─────────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌─────────────────────────────────────────────────┐
                  │ 56.B   Outline proposal                          │
                  │   in-scope clusters → chapter candidates         │
                  │   sub-clusters → section candidates              │
                  │   user reviews + edits                           │
                  └─────────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌─────────────────────────────────────────────────┐
                  │ 56.C   Atomic claim extraction                   │
                  │   per section: N atomic claims with hedge_strength│
                  │   + scope qualifier + rough chunk pins          │
                  └─────────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌─────────────────────────────────────────────────┐
                  │ 56.D   Per-claim retrieval engine                │
                  │   multi-query (DMQR) → RRF → MMR → NLI gate     │
                  │   output: claim ↦ {chunks above entailment floor}│
                  │   weak claims flagged for drop                  │
                  └─────────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌─────────────────────────────────────────────────┐
                  │ 56.E   Claim-driven writer                       │
                  │   per claim: 1-3 sentences from pinned chunks   │
                  │   <C:claim_id> placeholder, no LLM-chosen [N]   │
                  └─────────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌─────────────────────────────────────────────────┐
                  │ 56.F   Draft assembler + stitch pass             │
                  │   claims joined in plan order                   │
                  │   stitch model adds entity-bridge transitions   │
                  │   <C:…> resolved to deterministic [N]           │
                  └─────────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌─────────────────────────────────────────────────┐
                  │ 56.H   Section-ending safety net (always-on)     │
                  │   detect mid-word / orphan-`[` / mid-sentence    │
                  │   slice to last sentence; strip dangling cite   │
                  └─────────────────────────────────────────────────┘
                                          │
                                          ▼
                                  PUBLISHED DRAFT
                                          │
                                          ▼
                  ┌─────────────────────────────────────────────────┐
                  │ 56.G   Coverage feedback loop                    │
                  │   embed published content                       │
                  │   find uncited subclusters above entailment    │
                  │   surface as gap candidates → new section       │
                  └─────────────────────────────────────────────────┘
                                          │
                                          ▼
                                ITERATE OR PLATEAU
```

---

## Sub-phases

Each sub-phase ships as one PR (or short chain). Each has clear scope, dependencies, exit criteria, and rollback. Numbering matches the architecture diagram, not strictly the implementation order — see §Implementation order below.

---

### 56.A — Corpus topic tree

**Goal**: a queryable representation of the corpus's natural topical structure that the outliner can walk to propose chapter / section / subsection candidates.

**Scope**:
- Extend the existing RAPTOR L0/L1/L2 tree with metadata each cluster needs:
  - `centroid_embedding` (already there)
  - `summary_text` (already there)
  - `member_documents: list[doc_id]` (already there)
  - `depth_of_coverage: float` ← new — average chunks-per-document and average pairwise similarity within cluster
  - `scope_relevance: float | None` ← new — computed lazily per-book, against `book.description ⊕ book.plan` embedding
- New module `sciknow/core/topic_tree.py`:
  - `class TopicTree` with `walk_in_scope(book)` and `subclusters_of(node)` and `papers_in(node)`
  - Loads from existing Qdrant `papers` collection's RAPTOR summary nodes
- DB: no schema changes. Coverage / scope are computed on demand and cached on `book_chapters.custom_metadata`.

**Out of scope**:
- Re-running RAPTOR. Existing tree is reused as-is.
- Multi-corpus / cross-project clustering.

**Dependencies**: existing RAPTOR build (`docs/reference/INGESTION.md` covers the build step).

**Architecture**:
```
TopicTree.from_qdrant(qdrant, project)
  → reads papers collection, filters node_level >= 1
  → builds a parent_id → child_ids map
  → returns TopicTree wrapping the in-memory structure
```

**Tests**:
- L1: `topic_tree_cluster_metadata_schema` — every cluster has the expected fields.
- L2: `topic_tree_walk_finds_all_papers` — walking from root reaches every paper exactly once.
- L2: `topic_tree_scope_relevance_filters_off_topic` — fixture book scope; off-topic clusters score below threshold.

**Exit criteria**:
- A book's topic tree loads in < 200 ms for a 1000-paper corpus.
- `walk_in_scope` returns at most chapters_target × 3 candidate top-level clusters (we'll let the outliner narrow further).
- L1+L2 passes.

**Rollback**: feature toggle `USE_TOPIC_TREE_OUTLINE` (default off in early sub-phases).

---

### 56.B — Outline proposal from topic tree

**Goal**: from a book's scope statement and the in-scope topic tree, propose a candidate book outline (chapters → sections → subsections), each anchored to specific clusters.

**Scope**:
- New module `sciknow/core/outline_proposer.py`.
- Algorithm:
  1. Walk topic tree top-down. At each level, score candidate clusters against book scope.
  2. Top in-scope clusters at depth-1 → chapter candidates. Cap at `book.target_chapters` (default 12, configurable).
  3. Within each chapter candidate, depth-2 sub-clusters → section candidates. Cap at `book.target_sections_per_chapter` (default 6).
  4. Within each section candidate, depth-3 sub-sub-clusters → subsection candidates. Cap at 4.
  5. LLM proposes for each level: `title`, `description`, `plan_bullets` (3-5 bullets per section).
  6. Each level carries `cluster_anchor: cluster_id` so claim extraction can retrieve from the right pool.
- New CLI: `sciknow book propose-outline <book>` writes the proposal to a "draft outline" tab in the GUI for user review.
- GUI: new modal "Outline review" with a tree view — accept / edit / reject each node. Persist accepted nodes to `book_chapters.sections`.

**Out of scope**:
- Auto-accepting the proposed outline. User-in-the-loop is mandatory.
- Cross-chapter restructuring once the user has edited.

**Dependencies**: 56.A.

**Architecture**:
```
propose_outline(book_id) → OutlineProposal
  for each in-scope chapter cluster c:
    chapter_title, chapter_desc = LLM(c.summary, book.scope)
    for each section cluster s in c.children:
      section_title, section_plan = LLM(s.summary, chapter_desc)
      for each subsection cluster ss in s.children:
        sub_title, sub_plan = LLM(ss.summary, section_title)
```

**Tests**:
- L1: outline-proposer prompt smoke (small fixture corpus).
- L2: end-to-end on global-cooling — proposes a recognizable structure.
- L2: scope-respect — book scoped to "solar minima" doesn't propose a "marine biology" chapter even if the corpus contains some.

**Exit criteria**:
- Proposed outline for global-cooling shows ≥ 8 chapters, ≥ 4 sections / chapter, with cluster anchors traceable to actual RAPTOR nodes.
- User can accept the full proposal in < 60 s of clicking from the GUI.

**Rollback**: feature toggle. Existing outline path stays untouched.

---

### 56.C — Atomic claim extraction

**Goal**: for each leaf section (or subsection) in the accepted outline, extract N atomic claims that the section will assert.

**Scope**:
- New module `sciknow/core/claim_extractor.py` with `Claim(text, scope, hedge_strength, anchor_cluster, candidate_chunk_ids)`.
- For each section + its anchor cluster:
  - Pull cluster summary + a sample of cluster chunks (top-5 by centrality).
  - LLM emits N claims (target 8-15 per section, no hard cap — emerges from cluster content).
  - Each claim is one assertion: subject-predicate-(quantifier|qualifier)-(scope).
  - Hedge strength inferred from cue words in the supporting chunks ("suggests" / "indicates" / "shows" / "demonstrates" → mapped to strong/qualified/speculative).
- New prompt `CLAIM_EXTRACTION_SYSTEM` in `sciknow/rag/prompts.py`.
- DB: new column `book_chapters.sections_meta[i].claims: list[Claim]` (additive Alembic migration).

**Out of scope**:
- User-editable claim list (Phase 56.B's outline review covers section-level edits; claim-level review is deferred).
- Cross-section claim deduplication.

**Dependencies**: 56.B.

**Architecture**:
```
extract_claims(section, cluster) → list[Claim]
  user_msg = f"Cluster: {cluster.summary}\nChunks:\n{format_chunks(top_5)}\nSection: {section.title}\nPlan: {section.plan}"
  raw = LLM(CLAIM_EXTRACTION_SYSTEM, user_msg, format=claim_schema_json)
  return parse_claims(raw)
```

**Tests**:
- L1: claim parser accepts the schema; Hedge enum validates.
- L2: extracted claims for a global-cooling section have ≥ 6 claims, all with valid hedge_strength.
- L2: claim atomicity — every claim is one assertion (no compound "and" sentences with multiple independent clauses).

**Exit criteria**:
- Per-section claim count distribution centered on 8-15.
- Spot-check on 5 sections: claims read as concrete assertions, not headlines.

**Rollback**: behind 56.B feature flag.

---

### 56.D — Per-claim retrieval engine

**Goal**: for each claim, find the chunks that genuinely support it. Multi-query expansion + diversity reranking + entailment gating.

**Scope**:
- New module `sciknow/retrieval/claim_retrieve.py`.
- Pipeline per claim:
  1. **Query expansion** — DMQR-style (Yang et al. 2024). LLM generates 4-5 paraphrases that probe different facets of the claim. New prompt `CLAIM_PARAPHRASE_SYSTEM`.
  2. **HyDE** when claim is short (< 12 words) or abstract (no concrete entities). LLM hallucinates a plausible answer; embed and retrieve against that.
  3. **Hybrid search per query** — existing `hybrid_search.py`. top-15 each.
  4. **RRF fusion** across all queries. Standard k=60.
  5. **MMR diversification** (λ=0.65). Output: top-12 candidates.
  6. **NLI entailment scoring** of each candidate against claim text. Reuse `testing/quality.py`'s `_nli_entail_probs` helper.
  7. **Entailment gate** — keep chunks with `entail_prob ≥ 0.65`. If fewer than 2 survive, claim is **weak**.
- Output: `ClaimEvidence(claim_id, chunks, weak: bool, weak_reason: str)`.

**Out of scope**:
- FLARE-style mid-stream re-retrieval (deferred to Phase 57).
- Cross-claim chunk sharing optimization (we accept some retrieval cost overhead).

**Dependencies**: 56.C.

**Architecture**:
```
retrieve_for_claim(claim, project) → ClaimEvidence
  paraphrases = paraphrase_dmqr(claim, n=4)
  if claim.is_short or claim.is_abstract:
    paraphrases.append(hyde_generate(claim))
  pools = [hybrid_search(q, k=15) for q in paraphrases]
  fused = rrf_fuse(pools, k=60)
  diversified = mmr(fused, k=12, lambda_=0.65)
  scored = [(c, nli_entail(c.content, claim.text)) for c in diversified]
  kept = [c for c, p in scored if p >= 0.65]
  return ClaimEvidence(claim, kept, weak=(len(kept) < 2))
```

**Tests**:
- L1: paraphrase prompt smoke; RRF fusion correctness on synthetic data.
- L2: per-claim retrieval against global-cooling DB. Sample 10 claims; manually verify top-3 chunks are on-topic.
- L2: weak-claim detection — fixture claim with no supporting evidence flagged correctly.
- L3: full pipeline on one section — claims get evidence; weak ones are flagged.

**Exit criteria**:
- Mean number of retrieval slots per section: ≥ 30 (vs. today's ~10).
- Mean unique papers per section: ≥ 15 (vs. today's ~8).
- Spot check: top-3 chunks per claim are unambiguously relevant on a 10-claim sample.

**Rollback**: 56.D is internal to v56 path; not used until 56.E flips.

---

### 56.E — Claim-driven writer

**Goal**: replace the section-shot writer with per-claim micro-generations. Each claim → 1-3 sentences supported by its pinned chunks.

**Scope**:
- New module `sciknow/core/claim_writer.py`.
- For each claim with non-weak evidence:
  - Build a tight prompt: claim text + 2-3 chunks + hedging fidelity rules + "produce 1-3 sentences asserting this claim, ending the cited sentence with `<C:claim_id>`."
  - Call writer LLM with `temperature=0.2` (lower than current 0.3 — we want grounded, not creative).
  - Output is short (~150-300 tokens) so EOS-bias mid-sentence cutoffs are rare.
- For weak claims: emit a telemetry event `claim_dropped`; do not write prose for them. Coverage report surfaces them.
- Output per section: `list[ClaimSentences]` (one entry per non-weak claim).
- New prompt `CLAIM_WRITER_SYSTEM` in `prompts.py`.

**Out of scope**:
- Length tuning per claim (1-3 sentences is a soft ceiling, not enforced).
- Multi-claim sentences ("Both claim A and claim B follow from chunk X" is allowed but not optimized for).

**Dependencies**: 56.D.

**Architecture**:
```
write_claim(claim, evidence) → str
  prompt = format_claim_prompt(claim, evidence.chunks)
  return llm_stream(CLAIM_WRITER_SYSTEM, prompt,
                    temperature=0.2, num_predict=400)
```

**Tests**:
- L1: claim prompt formatter — `<C:claim_id>` placeholders survive round-trip.
- L2: write_claim on fixture claim produces 1-3 sentences with at least one `<C:claim_id>`.
- L3: full per-section writing on global-cooling section. Compare against v55 output for prose quality (subjective).

**Exit criteria**:
- 95% of non-weak claims produce 1-3 sentences ending with `<C:…>`.
- Mid-word truncation rate per claim: 0% (short outputs eliminate the failure mode).
- Per-section wall time: 3-6 minutes (vs. v55's 15 min, BUT for many more claims so total comparable).

**Rollback**: behind v56 engine flag.

---

### 56.F — Draft assembler + stitch pass

**Goal**: combine per-claim sentences into flowing prose. Resolve `<C:claim_id>` placeholders to deterministic `[N]` markers. Add entity-bridge transitions.

**Scope**:
- New module `sciknow/core/draft_assembler.py`:
  - `assemble_draft(claims_in_order, claim_evidence_map) → str`
  - Joins per-claim sentences in plan-order
  - Resolves `<C:claim_id>` → `[N]` where N is the deduped index of the claim's chunks in the section's source list
  - The deduped index is per-section: if claim A and claim B both cite chunk X, X gets one `[N]` shared across them
- New module `sciknow/core/stitch_writer.py`:
  - Takes assembled draft + section plan
  - LLM rewrites the connective tissue between claims (transitions, paragraph breaks, entity bridges) — but cannot edit text within claim spans (verified by diff)
  - Hard constraint: any modification to text between `<C:…>` placeholders rolls back the stitch (assemble_draft output is the floor)
- New prompt `STITCH_SYSTEM` in `prompts.py`.

**Out of scope**:
- Re-paragraphing across the whole section (stitch is local).
- Cross-section coherence (handled by chapter-level prior summaries in current writer; preserved here).

**Dependencies**: 56.E.

**Architecture**:
```
assembled = assemble_draft(claims, evidence_map)
stitched = stitch_writer(assembled, section_plan, prior_summaries)
verify_stitch(assembled, stitched)   # claim spans untouched
draft_md = render_citations(stitched)  # <C:c1> → [3]
```

**Tests**:
- L1: assemble_draft round-trips placeholders; deterministic citation numbering.
- L1: verify_stitch detects modifications inside claim spans.
- L2: end-to-end section assemble + stitch + render.
- L3: full chapter through 56.A → 56.F.

**Exit criteria**:
- Assembled drafts have **zero LLM-chosen `[N]` markers** — every cite traces to a `<C:claim_id>` placeholder.
- Stitch pass either succeeds (claim spans intact) or falls back to assembled-only.
- Section markdown is valid and renders cleanly in the GUI.

**Rollback**: behind v56 engine flag.

---

### 56.G — Coverage feedback loop

**Goal**: after each section / chapter / book completes, surface uncited corpus subclusters that warrant new sections.

**Scope**:
- New module `sciknow/core/coverage_telemetry.py`:
  - `scan_book_coverage(book_id, mode) → list[CoverageGap]`
  - Three modes: `post_outline` (broad), `post_chapter` (focused), `pre_finalize` (strict). Thresholds tighten in that order.
- For each in-scope cluster not represented by any current section's anchor:
  - Compute average entailment of cluster's papers' claims (synthetic) against existing book content
  - If above threshold AND not previously rejected → emit a gap candidate
- DB: extend `book_gaps` table with `gap_kind: missing_topic | uncovered_cluster | weak_evidence` (additive Alembic migration). New column `cluster_id: text NULL`.
- GUI: existing book-gaps panel gets a "Corpus coverage" section. Each gap has Add / Note / Reject actions (exact same UX as the existing claim-gap flow).
- Hook: called automatically after the user accepts an outline (post-outline scan), after each chapter completes (post-chapter scan), and on user-triggered "finalize check" (pre-finalize scan).

**Out of scope**:
- Auto-adding accepted gap candidates to outline. User reviews each.
- Cross-book coverage analysis.

**Dependencies**: 56.A (topic tree), 56.B (outline accepted), 56.F (drafts written).

**Architecture**:
```
scan_book_coverage(book_id, mode='post_outline') → list[CoverageGap]
  cluster_set = topic_tree.in_scope_clusters(book)
  cited_set = {section.anchor_cluster_id for section in book.sections}
  uncited = cluster_set - cited_set
  for cluster in uncited:
    if cluster_size < threshold[mode].cluster_min: continue
    if cluster.scope_relevance < threshold[mode].scope_min: continue
    if previously_rejected(book, cluster): continue
    yield CoverageGap(cluster=cluster, kind='uncovered_cluster')
```

**Plateau detection**:
- Count gap candidates surfaced after each chapter scan. Two consecutive scans with zero candidates → switch to "on-demand only" for the rest of the book. User can still trigger manually but the system stops auto-suggesting.

**Tests**:
- L1: threshold gate logic — mode-specific filters apply correctly.
- L1: previously-rejected gaps don't resurface.
- L2: post-outline scan on a book with deliberate gaps surfaces them.
- L2: plateau detection — after 2 zero scans, mode switches.

**Exit criteria**:
- Post-outline scan on global-cooling surfaces 5-15 candidate sections, all of which a domain expert would say are reasonable.
- Reject button works and persists.
- Plateau detection fires after the book settles.

**Rollback**: behind feature flag. Doesn't gate any existing flow.

---

### 56.H — Section-ending safety net (always-on)

**Goal**: post-stream cleanup that catches any mid-sentence / mid-word / orphan-`[` output. Cheap, always runs, regardless of which writer engine produced the draft.

**Scope**:
- New helper in `sciknow/core/draft_finalize.py`:
  - `_ends_cleanly(text) → bool` (last char ∈ `.!?")']`)
  - `_strip_dangling_citation_opener(text) → str` (regex `\[[\d,\s]*$` at end)
  - `_slice_to_last_sentence(text) → str` (lossy fallback, walks back to last `.!?` followed by whitespace/EOS, includes trailing close-quote)
  - `ensure_clean_ending(text) → str` (combines: strip dangling cite, then if not clean, slice; if slicing loses > 15% of body, keep original)
- Wire into:
  - `_autowrite_section_body` after `_stream_with_save` returns (current v55 path).
  - `claim_writer` and `stitch_writer` outputs (Phase 56 path, even though they rarely need it).
- Telemetry: `ending_repair_sliced` event with `removed_chars` count so we can measure how often the safety net fires.

**Out of scope** (deferred to Phase 57):
- LLM continuation pass (`finish_reason=length` retry pattern).
- `min_tokens` decode floor.
- Numbered paragraph quotas.
- Skeleton-of-Thought multi-pass.

**Dependencies**: none. Independent of every other sub-phase. Ships first.

**Architecture**:
```
ensure_clean_ending(content: str) -> str:
    content = _strip_dangling_citation_opener(content)
    if _ends_cleanly(content): return content
    sliced = _slice_to_last_sentence(content)
    if len(sliced) >= 0.85 * len(content): return sliced
    return content   # too lossy, keep mid-word as the lesser evil
```

**Tests**:
- L1: every fixture (mid-word, mid-sentence, orphan `[`, clean) produces the expected output.
- L1: 0.85 retention floor is honored.
- L2: run on every active draft of global-cooling; report the rate of repair.

**Exit criteria**:
- Zero mid-word endings in newly written drafts.
- Orphan `[` count: 0.
- Slice fallback fires < 25% of the time on Phase 56 micro-generations (short outputs are mostly clean).

**Rollback**: revert PR. Stack-unsafe path otherwise unchanged.

---

### 56.I — Engine switch + v55 fallback

**Goal**: route `book write-section`, `book autowrite-section`, `book autowrite-chapter`, `book autowrite-book` through v56 when the user opts in. Keep v55 available behind a flag.

**Scope**:
- Single ExportOptions / WriteOptions field: `engine: Literal["v55", "v56"] = "v55"`. Default stays v55 until 56.E + 56.F are stable.
- Each entry point branches on `engine`:
  - `v55` → existing single-shot writer (unchanged)
  - `v56` → new pipeline: claim-extract → per-claim retrieve → claim-writer → assemble → stitch → safety-net
- CLI: `--engine v55|v56` flag.
- GUI: a setting in the autowrite modal.
- Drafts produced by either engine carry `engine_version` in their custom_metadata for forensic tracing.
- Default flips to v56 after **all** of: 56.A-G shipped, L3 smoke green, side-by-side citation precision shows ≥ 0.30 lift over v55.

**Out of scope**:
- Sunset of v55. Stays available indefinitely as a fallback.
- Mixed-engine drafts (e.g., v55 wrote sections 1-3, v56 writes 4 onward — fine, just engine_version per draft).

**Dependencies**: 56.E + 56.F (the writer + assembler must work end-to-end).

**Tests**:
- L1: engine flag plumbed end-to-end (CLI → engine → metadata stamp).
- L2: writing one section under v56, one under v55, both succeed.
- L3: full chapter under v56 on global-cooling.

**Exit criteria**:
- Users can opt in to v56 per-call, per-book, or globally via setting.
- Output drafts indistinguishable from v55 at the storage layer (same `drafts` rows, just different content).

**Rollback**: flip default engine back to v55. v56 stays available.

---

### 56.J — Tests + smoke + comparison harness

**Goal**: every sub-phase tested in isolation, full pipeline tested end-to-end, v55-vs-v56 comparison harness for go/no-go on the engine flip.

**Scope**:
- L1 protocol additions for every new module (`topic_tree`, `outline_proposer`, `claim_extractor`, `claim_retrieve`, `claim_writer`, `draft_assembler`, `stitch_writer`, `coverage_telemetry`, `draft_finalize`).
- L2 tests against the global-cooling DB (already-ingested 849-paper corpus).
- L3 tests requiring full substrate up.
- New `scripts/compare_v55_v56.py`:
  - Picks a fixed section (Sumerian/Egyptian Minima), runs both engines, captures:
    - Citation precision (mean NLI entailment of cited chunk vs. cited sentence)
    - Distinct papers cited
    - Word count
    - Hedging fidelity scorer dimension
    - Wall time
  - Outputs JSONL row per run for trending.
- Regression budget: every Phase 55 L1/L2/L3 check still passes when running `--engine=v55`.

**Tests**: meta-tests — the test harness itself is L1-asserted to run every new test by name.

**Exit criteria**:
- All Phase 55 tests pass.
- New L1: ≥ 25 new checks across the 9 new modules.
- L2/L3 stable green on global-cooling.
- Comparison harness produces a JSONL row in < 25 min per (section × engine) on the 3090.
- Citation precision: v55 ~0.50 → v56 ≥ 0.92 on the same section_plan.

**Rollback**: drop the harness; tests are append-only.

---

## Implementation order

Linear dependencies dictate most of the order. I'll ship one sub-phase at a time, with each one ending green on its own L1+L2+L3 before moving on.

```
1.  56.H   Section-ending safety net          ← independent, immediate hygiene win, ships first
2.  56.A   Topic tree                          ← foundation
3.  56.B   Outline proposer                    ← needs A
4.  56.C   Claim extractor                     ← needs B
5.  56.D   Per-claim retrieval                 ← needs C
6.  56.E   Claim-driven writer                 ← needs D
7.  56.F   Assembler + stitch                  ← needs E
8.  56.I   Engine switch (default still v55)   ← needs E + F
9.  56.G   Coverage feedback loop              ← needs A + B + F
10. 56.J   Comparison harness + go/no-go flip  ← needs all of the above
```

After 10, default engine flips to v56, v55 stays as fallback.

---

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Claim extraction produces compound / non-atomic claims | medium | high | Strict prompt with WRONG/RIGHT examples; L2 atomicity test; fallback to single-claim regex split. |
| Per-claim retrieval is too slow (4-5 paraphrases × hybrid + rerank + NLI) | medium | medium | Cache paraphrases per claim; batch NLI; parallel hybrid search. Budget cap: ≤ 2 s per claim on 3090. |
| Stitch pass corrupts claim spans | low | medium | Hard constraint: post-stitch diff verification, fall back to assembled-only on violation. |
| Outline proposer produces too many / too few chapters | medium | low | Tunable caps (`target_chapters`, `target_sections_per_chapter`); user reviews and edits. |
| Coverage telemetry over-suggests gaps | medium | medium | Three cadences with tightening thresholds; rejection persists; plateau detection. |
| v56 wall-time is so much higher that users avoid it | medium | high | Cache RAPTOR walks, batch NLI, parallelize per-claim retrieval. Budget: total v56 section time ≤ 2× v55. If higher, profile and cut. |
| Hedging fidelity regresses (was bottom-ranked even in v55) | medium | medium | Hedging is now a structural property of the claim (`hedge_strength`), not a writer-prompt rule. Should improve. Verify in 56.E L3. |
| Existing book drafts can't be re-rendered with v56 (no claims persisted) | low | low | Existing drafts stay v55. Only new writes go v56. Optional later: backfill claim metadata. |
| Topic tree's `scope_relevance` scoring is noisy | medium | medium | Calibrate threshold on global-cooling against expert-judged inclusion list; expose threshold as a tunable. |

---

## Success metrics

Measured at the end of 56.J:

| Metric | v55 baseline | v56 target |
|---|---:|---:|
| Citation precision (NLI entailment of cited chunk vs sentence) | ~0.50 | ≥ 0.92 |
| Distinct papers cited per section | 8.5 | ≥ 15 |
| Total unique papers cited per book | 246 / 849 (29%) | ≥ 600 / 849 (70%) |
| Hedging fidelity scorer dimension | bottom-ranked, ~0.55 | ≥ 0.75, no longer bottom |
| Mid-word section endings | ~75% of sections | 0 |
| Orphan `[` openers | observed | 0 |
| Wall time per section | ~25 min | ≤ 50 min (2× v55) |
| Citation alignment remap rate | 17-62% | ≤ 5% (post-pass should be near-no-op) |

---

## What stays out of scope (explicitly)

- **Length-controlled writing.** No `min_tokens`, no numbered paragraph quotas, no length expansion passes. Length emerges from claims. Phase 57 (or later) addresses length if user feedback demands it.
- **Training-time fixes.** EOS Token Weighting, Hansel countdown, length-conditioned PO — all require fine-tuning. We use pre-trained Qwen3.x and operate at the inference/orchestration layer.
- **Cross-book / cross-corpus reasoning.** Single-book scope per run.
- **Real-time outline editing during writing.** Outline is locked once the user accepts; coverage feedback proposes additions but doesn't restructure mid-flight.
- **Smaller / cheaper models.** Writer / scorer / extractor model selection is not changed.
- **GraphRAG, ColBERT, or other retrieval upgrades.** RAPTOR + multi-query + MMR + entailment is the architecture; other paradigms are out of scope.
- **Sunsetting v55.** Stays available indefinitely.

---

## Definition of done

- All 10 sub-phases complete (committed + green tests).
- L1: 297/298 → 320+/n (new modules add ≥25 L1 checks).
- L2: passes on global-cooling DB.
- L3: full chapter under v56 produces a valid draft from claim plan.
- Comparison harness shows ≥ 0.30 citation-precision lift on a 5-section sample.
- Default engine flips to v56 in `cli/book.py` and the GUI autowrite modal.
- `docs/roadmap/PHASE_LOG.md` carries an entry per sub-phase.
- A new tag `autowrite-stable-phase56-v1` anchors the v56-default state.
