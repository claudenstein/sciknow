# sciknow book actions — reference

This doc is the single source of truth for what every book-level AI action does, when to use it, what it requires, what it produces, and what it costs. It also catalogues the **elicitation** and **brainstorming** method libraries that steer outline / plan / gap-finding runs through a named cognitive technique.

Every action described here has two entry points: the CLI (`sciknow book ...`) and the web reader (draft toolbar or Plan modal). They run the same code path — the web endpoints either call the same generator functions the CLI uses or shell to the CLI binary via an SSE-streamed subprocess.

---

## Workflow stages

The book workflow is a pipeline. The table groups actions by stage so you can see which one to reach for at each step.

| Stage | Actions | Question it answers |
|---|---|---|
| **Plan** | Outline · Book plan (leitmotiv) · Chapter sections | What chapters does this book need, in what order, with what subsections? |
| **Write** | AI Write · AI Autowrite · Edit | How do I fill this section with grounded prose? |
| **Critique** | AI Review · Adversarial review · Edge cases · Ensemble review · Argue · Gaps | What's wrong with this draft — or with the book as a whole? |
| **Verify** | Verify · Verify Draft · Align Citations · Insert Citations | Do the `[N]` markers point at chunks that actually entail the claims? |
| **Fix** | AI Revise | Apply the feedback to the prose without writing from scratch |
| **Inspect** | Scores · Bundles · Chapter reader · `book draft-scores` / `draft-compare` | How did autowrite converge — and can I snapshot before risking a change? |

A typical first pass: **Outline → Autowrite each section → Verify Draft → Align Citations → Review → Revise → Bundle**.

---

## Planning actions

### Outline — `sciknow book outline`

**Where**: Plan modal → Outline tab (54.6.96). CLI: `uv run sciknow book outline "<book title>"`.

**What it does.** Generates a chapter structure for the book. Reads the active project's paper corpus (titles + years), generates 3 candidate outlines at rising temperatures (0.5 / 0.65 / 0.80), scores each candidate on breadth + section-count variance, picks the winner. Then per-chapter density-resizes the section list: for each chapter, runs one hybrid retrieval on `topic_query`, counts distinct papers, and trims to a target section count so chapters with more evidence keep more sections (54.6.65).

**What it's NOT.** It does not write prose. It does not evaluate existing drafts. It does not touch chapters that already exist — inserts are **additive** by chapter number. Re-run as many times as you like to re-roll the proposal.

**Requires.** The book must exist. The corpus must have at least a handful of papers with titles.

**Produces.** New rows in `book_chapters` with `title`, `description`, `topic_query`, and `sections` (each a `{slug, title, plan}` dict).

**Skipped on flat project types.** When `project_type` is a flat type like `scientific_paper`, the book has a single chapter with canonical sections bootstrapped at `book create` time. Outline refuses to run because its output shape doesn't apply.

**Cost.** ~3 × 20-60 s = ~1-3 min on `LLM_MODEL` (qwen3:30b-a3b-instruct-2507), since it generates 3 candidates.

### Book plan (leitmotiv) — `sciknow book plan`

**Where**: Plan modal → Book tab. CLI: `uv run sciknow book plan "<book title>"`.

**What it does.** Writes (or rewrites) the 200-500 word leitmotiv — the thesis + scope document that gets injected into every `write` / `autowrite` call so all chapters argue consistently. This is a prose document, not a chapter list. The Regenerate-with-LLM button uses the existing chapter titles and corpus to draft a new leitmotiv from scratch; editing by hand is also fine.

**When to use.** Early in the book's life, after you've outlined chapters but before you write drafts. Re-run when the central argument has drifted or sharpened.

### Chapter sections — manual

**Where**: Plan modal → Sections tab.

**What it does.** Per-chapter editing of section plans and target word counts. Read the tab's in-panel help — this is not an LLM action, just a form. When Outline has generated sections you want to adjust before writing, this is where.

---

## Writing actions

### AI Write — `sciknow book write`

**Where**: Draft toolbar (per-section).

**What it does.** Single-pass retrieval → draft. Runs hybrid search (dense + sparse + FTS + RRF) on the section's topic_query, reranks the top-50 candidates, feeds the top-k as context to the writer model with the leitmotiv, and streams prose. Writes one draft version and stops. No self-evaluation.

**Cost.** 10-60 s depending on section length + model. Default writer is `LLM_MODEL` (qwen3:30b-a3b-instruct-2507).

### AI Autowrite — `sciknow book autowrite`

**Where**: Draft toolbar.

**What it does.** Full convergence loop: **write → score → verify → (CoVe) → revise → rescore**, repeating up to `max_iter` (default 3) or until `overall ≥ target_score` (default 0.85). Key mechanics:

- **Score pass**: runs on `AUTOWRITE_SCORER_MODEL` (default gemopus4, gap 0.60 — 3× more good/bad discrimination than the writer alone).
- **Verify pass**: runs `verify_claims` on the writer model, gets `groundedness_score` and `hedging_fidelity_score`. If verification finds issues, it overrides the scorer's groundedness and targets the revision instruction at the flagged citations.
- **Plan coverage dimension** (54.6.79): scores NLI entailment of each atomic plan bullet against the draft; if coverage is the weakest dimension AND < target, the revision instruction names the missed bullets.
- **CoVe** (optional, `use_cove=True` default): Chain-of-Verification pass on any claim below `cove_threshold` before committing the draft.

**Cost.** 1-4 min per section typically. 4 model swaps per iteration on a 24 GB GPU (see `docs/roadmap/PHASE_LOG.md` 54.6.X for the breakdown). Expensive but the highest-quality output sciknow can produce single-shot.

### Edit

**Where**: Draft toolbar → Edit button.

**What it does.** Opens the in-browser markdown editor with autosave, KaTeX math rendering, and inline figure thumbnails (`![caption](/api/visuals/image/<id>)`). Not an LLM action.

---

## Critique actions

### AI Review — `sciknow book review`

**Where**: Draft toolbar → AI Review button. CLI: `uv run sciknow book review <draft_id>`.

**What it does.** Single-pass critic on one specific draft across 5 dimensions:

1. **Groundedness** — does every claim trace to an actual source chunk?
2. **Completeness** — does the draft cover the section plan?
3. **Accuracy** — do any quantitative or qualitative statements misrepresent the sources?
4. **Coherence** — does the prose flow, or does it read as stitched-together fragments?
5. **Redundancy** — repeated claims, over-citing, wheel-spinning.

Produces structured feedback with specific quotes from the draft + actionable suggestions. Saved to `drafts.review_feedback`. Uses `BOOK_REVIEW_MODEL` (default `gemma3:27b-it-qat`) — validated by quality bench (100% judge-win, 5/5 dimensions covered at 71.4% win-rate vs the writer as reviewer).

**Review vs Outline.** Different inputs, different outputs, different stages:

| | **Review** | **Outline** |
|---|---|---|
| Input | One saved draft's prose | Book title + paper corpus |
| Requires drafts? | Yes — targets a single draft | No — runs on an empty book |
| Output | 5-dim critique with quotes + suggestions, saved to `review_feedback` | New chapter rows with sections |
| DB effect | Feedback only — draft prose unchanged | Additive chapter inserts |
| Stage | Editing | Planning |
| Run when | A draft exists and you want to know where to improve | You're building or expanding the book's shape |

**Why you might not see the Review button.** The draft toolbar (including Edit / Autowrite / Write / Review / Revise) is hidden on the Dashboard view — it only appears once you click a draft in the sidebar. If you open the Plan modal, Review is **not** there (Outline is, because Outline operates on the book, not a draft). Select any draft from the sidebar first.

### Adversarial review — `sciknow book adversarial-review`

**Where**: Draft toolbar → Critique menu → Adversarial review.

**What it does.** Harsher single-pass critic. Forced to return **≥ 10** concrete issues (unsupported claims, overgeneralisation, weasel words, missing counter-evidence, internal contradictions, loaded framing). Never graded. Doesn't overwrite the normal `review_feedback` — output goes into its own metadata field. BMAD-inspired.

**Use when** the normal Review feels too gentle, or when you want a second opinion with a very different stance.

### Edge cases — `sciknow book edge-cases`

**Where**: Draft toolbar → Critique menu → Edge cases.

**What it does.** Exhaustive boundary-condition hunter. Walks every scope boundary, counter-case, causal alternative, quantitative limit, and missing control in the draft. Returns a severity-ranked structured JSON finding list.

**Use when** you need the draft to survive hostile readers and want every edge exposed.

### Ensemble Review — `sciknow book ensemble-review`

**Where**: Draft toolbar → Critique menu → Ensemble Review.

**What it does.** N independent NeurIPS-rubric reviewers (default 3, temperature 0.75, rotating neutral/pessimistic/optimistic stance — positivity-bias mitigation from AI-Scientist v1 §4) + a meta-reviewer that medians the numeric scores and unions the free-text lists weighted by agreement. Rubric: soundness / presentation / contribution (1-4), overall (1-10), confidence (1-5), decision ∈ strong_reject…strong_accept.

**Cost scales with N.** At 3 reviewers that's roughly 3 × AI Review wall time. Highest variance reduction we have; use when a single Review felt flaky.

### Argue — `sciknow book argue`

**Where**: Draft toolbar → Critique menu → Argue.

**What it does.** For any claim you type, builds a SUPPORTS / CONTRADICTS / NEUTRAL evidence map from the corpus. Returns per-side paper lists with short rationales.

**Use when** you want to stress-test a specific assertion before committing to it in prose.

### Gaps — `sciknow book gaps`

**Where**: Draft toolbar → Critique menu → Gaps. Also a dashboard badge in the top bar.

**What it does.** Compares the book's plan (both leitmotiv and chapter/section outlines) to what the drafts actually cover. Topics named in the plan but absent from drafts surface as gaps. Optionally triggers `auto-expand` to search for papers that would fill them.

---

## Verification actions

The four verification actions attack citation quality from different angles. Run them in roughly this order: **Insert → Verify → Verify Draft → Align** (`Insert` only when the draft is missing `[N]` markers entirely).

### Verify — `sciknow book verify-citations`

**Where**: Draft toolbar → Verify menu → Verify.

**What it does.** Sentence-level LLM verifier. For each `[N]` citation, classifies the cited claim as:

- `SUPPORTED` — source entails the claim
- `EXTRAPOLATED` — goes further than source allows
- `MISREPRESENTED` — distorts what source says
- `OVERSTATED` — removes hedging source had

Returns a `groundedness_score` and a `hedging_fidelity_score`.

### Verify Draft — `sciknow book verify-draft`

**Where**: Draft toolbar → Verify menu → Verify Draft (54.6.83).

**What it does.** Where Verify checks whole sentences, Verify Draft **atomizes** each sentence into sub-claims (regex heuristic first, LLM fallback for compound sentences > ~30 words with multiple verbs), then NLI-scores each sub-claim against source chunks individually. Reports `mixed_truth` sentences — where one sub-claim is well-supported but another in the same sentence isn't. This is the failure mode sentence-level verification misses: "Global temperature rose 0.6°C since 1980 [1], driven primarily by CO₂ [1]" where [1] supports the first half but not the causal claim.

Read-only — no database writes. Uses the NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`, ~440 MB, shared with the quality bench).

### Align Citations — `sciknow book align-citations`

**Where**: Draft toolbar → Verify menu → Align Citations (54.6.71).

**What it does.** Conservative post-pass. For every `[N]` marker, scores the sentence's entailment against every retrieval source chunk. Remaps N when:

1. The currently-cited chunk's entailment is < 0.5 (claimed source is bad), AND
2. The top-entailment chunk beats the claimed chunk by ≥ 0.15 (there's a clearly better source)

Both thresholds tunable via `--low-threshold` and `--win-margin`. When both conditions hold, writes the remapped draft back. Dry-run with `--no-save`.

**Requires.** The draft's `sources` field must be populated (it is, if the draft came from `book write` or `book autowrite`).

### Insert Citations — `sciknow book insert-citations`

**Where**: Draft toolbar → Verify menu → Insert Citations.

**What it does.** Two-pass LLM that adds `[N]` markers where the prose makes a source-worthy claim but has no citation. Pass 1 identifies claim-sentences; pass 2 matches each to the best chunk from the draft's retrieval set. Saves a new draft version so the original is still there.

**Use when** you hand-wrote a draft (or imported one) that's missing its citation layer.

---

## Fixing action

### AI Revise — `sciknow book revise`

**Where**: Draft toolbar → AI Revise button.

**What it does.** Reads the latest `review_feedback` saved on the draft and rewrites the prose to address it. **Does NOT run Review itself** — it consumes Review's output. If you haven't run Review (or Autowrite, which reviews internally) you'll get a "no feedback to apply" error.

Writes a new draft version.

**Use when** you've read the review and agreed with its points. If you want Review + Revise atomically, use Autowrite, which iterates write→score→verify→revise→rescore internally.

---

## Diagnostics

### Scores panel — draft toolbar → Verify menu → Scores

Shows the 5-dimension score trajectory across autowrite iterations. Read-only panel — no LLM work.

### Bundles — draft toolbar → Extras menu → Bundles

Chapter- or book-wide snapshots. Safety net for `autowrite-all-sections` runs: snapshot before, restore if you hate the result. CLI: `book snapshot` / `book snapshot-restore`.

### Chapter reader — draft toolbar → Extras menu → Chapter reader

Read-only continuous-scroll view of the entire chapter without the editor chrome. Useful for reading the whole chapter's flow as a reader would.

### Draft scores + compare (CLI-only)

```bash
uv run sciknow book draft-scores <draft_id>        # show the full 5-dim breakdown
uv run sciknow book draft-compare <a_id> <b_id>    # side-by-side comparison
```

Not currently exposed in the GUI — use the Scores panel in the toolbar for the visual version.

---

## Elicitation methods

These 24 named cognitive techniques (adapted from BMAD-METHOD under MIT) can be selected from the **Outline tab's** method picker to prepend a short preamble that steers the LLM's approach. They don't rewrite the core prompt — they layer on top. Each method is a `{category, name, description}` dict in `sciknow/core/methods.py::ELICITATION_METHODS`.

**How to choose.** The description in each card says when to use it. As a rule of thumb: **Collaboration** methods when a topic has multiple stakeholders or perspectives in tension; **Advanced** for complex reasoning / systems thinking; **Competitive** when you need stress-testing; **Critical** for epistemological depth; **Technical** when trade-offs need to be explicit; **Scientific** specifically for peer-reviewable rigor.

### Collaboration

| Method | When to pick it |
|---|---|
| **Stakeholder Round Table** | Convene multiple personas so each viewpoint gets heard. Best for requirements / scoping where stakeholders have competing interests. |
| **Expert Panel Review** | Domain experts go deep — ideal when technical depth and peer-review quality matter more than breadth. |
| **Debate Club Showdown** | Two personas argue opposing positions while a moderator scores; great for controversial decisions where you want the middle ground surfaced. |
| **Time Traveler Council** | Past-you + future-you advise present-you; powerful when short-term pressure is warping long-term judgment. |
| **Mentor and Apprentice** | Senior explains to junior who asks naive questions; surfaces hidden assumptions through the teaching dialogue. |
| **Good Cop Bad Cop** | Supportive + critical alternate; returns a balanced strengths-to-build-on + weaknesses-to-address breakdown. |

### Advanced reasoning

| Method | When to pick it |
|---|---|
| **Tree of Thoughts** | Explore multiple reasoning paths simultaneously, then evaluate and select. Perfect for problems with multiple valid approaches. |
| **Graph of Thoughts** | Model reasoning as an interconnected network to reveal hidden relationships. Ideal for systems thinking and emergent patterns. |
| **Self-Consistency Validation** | Generate multiple independent approaches then compare; critical when verification matters more than speed. |
| **Meta-Prompting Analysis** | Step back to analyze the methodology itself, not just the content. Good for improving future runs. |
| **Reasoning via Planning** | Build a reasoning tree guided by world models and goal states. Best for strategic planning and sequential decisions. |

### Competitive / adversarial

| Method | When to pick it |
|---|---|
| **Red Team vs Blue Team** | Adversarial attack-defend to find vulnerabilities. Security-mindset critique. |
| **Pre-mortem** | Imagine the project failed a year from now; work backwards to identify likely causes. Surfaces risks optimism normally hides. |
| **Shark Tank Pitch** | Pitch to skeptical investors who poke holes; forces clarity on value proposition and viability. |

### Critical / philosophical

| Method | When to pick it |
|---|---|
| **Socratic Method** | Progressively deeper questions; each answer triggers a sharper follow-up. Best for forcing you to examine assumptions. |
| **First Principles** | Strip away assumptions and rebuild from fundamental truths. Essential for breakthrough innovation. |
| **Five Whys** | Drill through layers of causation to root causes. Root-cause analysis. |
| **Assumption Reversal** | Flip core assumptions and rebuild. Essential for paradigm shifts — the method Einstein (and Musk) cite. |

### Technical / architectural

| Method | When to pick it |
|---|---|
| **Architecture Decision Records** | Propose and debate structural choices with explicit trade-offs. Ensures decisions are well-reasoned + documented. |
| **Rubber Duck Evolved** | Explain the content to progressively more technical ducks until you find the hole. Forces clarity at multiple abstraction levels. |
| **Algorithm Olympics** | Multiple approaches compete on the same problem with benchmarks. Finds the optimal solution through direct comparison. |

### Scientific writing-specific

| Method | When to pick it |
|---|---|
| **Peer Review Simulation** | Imagine three referees (methodological / empirical / conceptual). What do they each flag? Where do they disagree? |
| **Strong Inference (Platt)** | Devise an alternative hypothesis, design a crucial experiment that would disprove each, identify which the data currently rules out. |
| **Inference to the Best Explanation** | List all plausible explanations, rank by prior × fit, identify which evidence would discriminate between them. |

---

## Brainstorming methods

The 24 methods in `BRAINSTORMING_METHODS` are used by the **Gaps** flow and the Visualize modal's consensus tab when you want a wider net — same pattern as elicitation but with a generative rather than evaluative bent. They are grouped in four categories: **collaborative**, **creative / lateral**, **deep / analytical**, and **scientific writing-specific** (`Missing Control`, `Scope Boundaries`, `Benchmark-Hunting`).

Read the cards directly in the UI — each has a short description of when it applies. The scientific-writing three are the most useful for sciknow specifically:

- **Missing Control** — for every claim, ask what the null / baseline / placebo case would look like and whether the author addressed it.
- **Scope Boundaries** — for every claim, list the spatial / temporal / population / regime / instrument conditions under which it would fail. Each boundary = a potential gap.
- **Benchmark-Hunting** — what would a gold-standard benchmark for this claim look like? Does the literature have one? If not, what would it take to build?

---

## Where to find commands in code

| Action | CLI entry (cli/book.py) | Generator (core/book_ops.py) | Web endpoint (web/app.py) |
|---|---|---|---|
| outline | `outline` | (inline in CLI; web re-implements) | `POST /api/book/outline/generate` |
| plan (leitmotiv) | `plan` | (CLI uses `llm.complete`) | `POST /api/book/plan/regenerate` |
| write | `write` | `write_section_stream` | SSE via stream endpoints |
| autowrite | `autowrite` | `autowrite_section_stream` | SSE via stream endpoints |
| review | `review` | `review_draft_stream` | SSE via stream endpoints |
| revise | `revise` | `revise_draft_stream` | SSE via stream endpoints |
| adversarial-review | `adversarial-review` | `adversarial_review_stream` | `POST /api/adversarial-review/{id}` |
| edge-cases | `edge-cases` | `edge_cases_stream` | `POST /api/edge-cases/{id}` |
| ensemble-review | `ensemble-review` | `ensemble_review_stream` | `POST /api/cli-stream` (allowlist) |
| verify-draft | `verify-draft` | `claim_atomize.verify_draft` | `POST /api/cli-stream` (allowlist) |
| align-citations | `align-citations` | `citation_align.*` | `POST /api/cli-stream` (allowlist) |
| verify-citations | `verify-citations` | `citation_verify.*` | SSE via stream endpoints |
| insert-citations | `insert-citations` | inline CLI path | SSE via stream endpoints |
| argue | `argue` | `argue_stream` | SSE via stream endpoints |
| gaps | `gaps` | `gaps_stream` | SSE via stream endpoints |
| snapshot / snapshot-restore | `snapshot` / `snapshot-restore` | `bundle_*` | Bundles modal |

See `docs/roadmap/PHASE_LOG.md` for the phases that introduced each feature — search e.g. "54.6.83" for Verify Draft or "54.6.71" for Align Citations.
