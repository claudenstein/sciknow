# `NousResearch/autoreason` — portability review

**Sources**:
- Paper: https://github.com/NousResearch/autoreason/blob/main/paper/autoreason.pdf
  (SHL0MS & "Hermes Agent", Nous Research, 2026, ~32 pages).
- Repo: https://github.com/NousResearch/autoreason (MIT-ish license).

**Review date**: 2026-04-14.

Bottom line: autoreason is a ~500-LOC **inference-time orchestration
pattern** wrapped in a genuinely rigorous 32-page paper. The repo is
a *research-log release*, not a library — no pyproject.toml, no CLI,
no tests. Porting means taking the architectural ideas and
re-implementing natively. Three ideas are worth adopting; the rest is
either already in sciknow, training-only (we're not training), or
explicit negative-result from the paper itself.

## What autoreason actually is

One-paragraph thesis: single-agent critique-and-revise loops
(Self-Refine, Reflexion) systematically **degrade** outputs because
of three structural pathologies:

1. **Prompt bias** — "find problems" forces hallucinated flaws even on good
   input.
2. **Scope creep** — every revision pass adds bloat.
3. **Lack of restraint** — models never output "no changes needed".

The fix is architectural, not training-time:

- Each iteration runs **three fresh isolated agents**: an unchanged
  **incumbent A**, an adversarial **revision B**, and a **synthesis AB**
  (merging A's conservatism with B's changes).
- A **blind Borda-count judge panel** picks the winner with **randomized
  presentation order** to kill positional bias.
- **Ties go to the incumbent A** ("do nothing" is a first-class option).
- **Convergence at `k=2`** consecutive incumbent wins. `k=1` fires
  prematurely 94% of the time (Table 23); `k=3` fails 24% of trajectories
  at 2× cost; `k=4` fails 47%.
- **Fresh stateless agents** eliminate session-level sycophancy gradient.

The paper's single biggest transferable insight is the **four success
conditions** for when iterative refinement works at all:
(1) external verification, (2) constrained scope, (3) structured
reasoning, (4) sufficient decision space. Autoreason explicitly
**fails** on Sonnet-4.6 unconstrained writing tasks (Section 5,
Table 4) — the public negative result is unusually honest.

### Landscape positioning

It is *not* Chain-of-Thought, *not* CoT distillation, *not* STaR / Quiet-STaR
(those are SFT recipes), *not* o1-style internal reasoning, *not*
RLHF. Closest relatives:

- **Self-Refine** (Madaan et al. 2023, NeurIPS, ref [4]) — the baseline
  it's beating; the pattern sciknow currently implements.
- **LLM Council** (Zhao et al. 2024, ref [9]) — multi-model panel for
  single-turn eval; close cousin.
- **Verdict** (Kalra & Tang 2025, ref [12]) — hierarchical judge
  verification; paper tested it and it *failed* on the Sonnet 4.6
  scaling issue (Table 7).
- **DSPy** (Khattab 2023, ref [10]) — declarative pipeline compilation;
  more ambitious framing.

## Recommended adoptions — priority stack

### #1 — Tournament autowrite with conservative tiebreak (~150–200 LOC, 4–6 h)

**Why it fits sciknow**. Current `autowrite_section_stream` in
`sciknow/core/book_ops.py:3192` can make drafts *worse* across
iterations. The revision is accepted as long as `score > target_score`,
not as long as it *beats the previous draft*. Autoreason Table 12
quantifies this failure mode for Haiku 3.5: baselines "degrade outputs
below the unrefined single pass" by 59–70% word-count loss. Sciknow
runs on local mid-tier models (qwen2.5:14b, llama3.1:8b) — precisely
the capability regime where the paper shows tournament structure is
most valuable (Table 13 shows +8.3 Borda margin for Haiku 3.5).

**Scope**:
- New `_autowrite_tournament_step(incumbent, results, log)` in
  `book_ops.py`. Three fresh `call_ollama()` calls (critic → author-B →
  synthesizer), then N parallel judge calls (default 3), aggregated via
  Borda with conservative tiebreak.
- Replace the existing "score; if below target, revise" block in
  `_autowrite_section_body` with a tournament step + the k=2
  convergence rule.
- New prompt templates in `sciknow/rag/prompts.py` for critic /
  author-B / synthesizer / judge.
- Config: `tournament_enabled: bool`, `num_judges: int = 3`,
  `convergence_k: int = 2`.

**Trade-offs**:
- **3–6× more Ollama calls per iteration** (critic + author-B +
  synthesizer + 3 judges = 6 vs current 1–2). On a local 3090 running
  qwen2.5:14b that's ~90s → ~9 min per section at default settings —
  significant. Mitigation: `--tournament` flag, default off, or gate
  to final-chapter passes only.
- Borda-over-local-LLM is noisier than Borda-over-Sonnet (judge
  disagreement scales inversely with judge capability). Table 14:
  Haiku-judge = 20.7 Borda, Sonnet-judge = 23.0 — weak judges still
  work but not at Sonnet levels.

**L1 test**:
`l1_phaseN_tournament_conservative_tiebreak_returns_incumbent_on_borda_tie`
— mock three candidates (A=6, B=6, AB=3), assert winner == A.

### #2 — CoT judge prompts + length-controlled eval utility (~80 LOC, 2 h)

**Why it fits sciknow**. Two orthogonal, nearly-free wins from the
paper's own ablations:

- **CoT judges** cut convergence 3× on Task 1 (Table 3) by asking the
  judge four explicit questions before the verdict: *"What does it get
  right? wrong? Are numbers defensible? Is detail appropriate or
  bloated?"* (Appendix A.5). Zero architectural change.
- **Length-controlled pairwise eval** (Table 17) truncates candidates
  to the median word count before judging. Proves quality gains aren't
  just verbosity. Sciknow's scorer today has no defense against a
  longer-is-better bias.

**Scope**:
- Edit the scoring prompt in `sciknow/rag/prompts.py` to include the
  four CoT questions before the verdict (~20 LOC).
- New `sciknow/testing/length_controlled_eval.py::compare_at_matched_length()`
  that truncates all candidates to the median word count and runs a
  pairwise judge panel (~60 LOC).

**Trade-offs**: CoT judging adds ~500 tokens per judge call (small).
Length-controlled eval is a `bench`-time utility, not the hot-path
autowrite loop — no production latency cost.

**L1 test**:
`l1_phaseN_scorer_prompt_contains_four_cot_questions` (grep) +
`l1_phaseN_length_controlled_eval_truncates_to_median_word_count`.

### #3 — Four-conditions refinement gate (~80 LOC, 3 h)

**Why it fits sciknow**. The paper's most transferable *intellectual*
contribution is the claim that iterative refinement is only useful
when the task has (1) external verification, (2) constrained scope,
(3) structured reasoning, (4) sufficient decision space. Sciknow's
autowrite runs the same expensive loop on every section regardless of
whether it has retrieval hits, target_words, etc. Running the loop on
sections that fail the conditions is wasted compute — and Table 5
shows it can actively *harm* quality.

**Scope**:
- New `sciknow/core/refinement_gate.py`:
  `should_run_refinement(section_type, target_words, num_retrieval_hits,
  has_explicit_outline) -> tuple[bool, str]` returning the decision
  plus a human-readable reason.
- Called at the top of `_autowrite_section_body`. Start with warning-
  only mode for one release ("refinement skipped because N retrieval
  hits < 3; use `--force-refinement` to override") before hard-skipping.
- Defaults:
  - `num_retrieval_hits < 3` → no verification signal, skip.
  - `target_words is None and section_type in {"discussion", "conclusion"}`
    → unbounded scope, skip.
  - `has_explicit_outline is False` → no structured reasoning, skip.

**Trade-offs**: conservative defaults might skip the loop where it
would have helped — hence the warning-first rollout.

**L1 test**:
`l1_phaseN_refinement_gate_skips_when_retrieval_hits_below_threshold`.

### #4 — Bootstrap CI + McNemar's test utility for bench/expand evals (~130 LOC, 1.5 h)

**Why it fits sciknow**. `sciknow/testing/bench.py` reports raw
counts; no uncertainty quantification. For `db expand` HITL runs and
Phase 50 user-feedback A/B cohorts, statistical significance is a real
question. Autoreason's `experiments/v2/compute_stats.py` (131 LOC,
scipy + stdlib, zero Anthropic-specific code) is directly portable.

**Scope**:
- Copy `compute_stats.py` into `sciknow/testing/stats.py`, rename the
  experiment-loading function to accept the `bench` results shape.
- Expose `sciknow test --compare --strategy-a X --strategy-b Y`.

**Trade-offs**: adds `scipy` as a hard dep (already transitive via
numpy stack — check `uv.lock`).

**L1 test**:
`l1_phaseN_bootstrap_ci_returns_lo_le_mean_le_hi_on_binary_data`.

## Already in sciknow (don't re-port)

- **Retrieval-backed claim verification** — `_verify_draft_inner` at
  `sciknow/core/book_ops.py:2999` already does what autoreason's
  Section-8 "ground-truth critic" does, but stronger (live retrieval,
  not a static context blob).
- **Stateless stage agents** — each stage is its own `call_ollama()`.
- **Rich per-iteration logging** — `data/autowrite/latest.jsonl` +
  per-draft checkpoints are richer than autoreason's flat pass_N_*.md
  files.
- **RRF / cross-encoder reranker** — not part of autoreason at all.

## Explicit skip list

- **7-judge panel for everything** — 7× judge token cost for marginal
  gain over 3. Use 3 in-loop, save 7 for `sciknow test --gold-eval`.
- **Margin-based convergence (≥ 2 Borda-point gap)** — only needed
  to rescue Sonnet-4.6 scaling failure. Sciknow never hits that regime.
- **Verdict-style hierarchical verification** — paper tested it and it
  *did not fix* convergence (Table 7). Negative result; skip.
- **Scope-aware judges / plateau detection** — same; both tested and
  failed as Sonnet-4.6 rescue attempts (Table 7).
- **PSRO-style meta-strategy search** — paper explicitly disclaims it.
- **Ground-truth critic context injection** — already have a stronger
  analog (retrieval-grounded verification).
- **Iterative code-test loop** — we don't execute user code.
- **Anything training-related** — autoreason has zero training code;
  neither should this port. Not SFT, not RL, not reward modeling.

## Links worth keeping

- Paper itself — best LLM-as-judge failure-mode taxonomy I've seen.
- **karpathy/autoresearch** (ref [1]) — spiritual ancestor with
  `val_bpb` as fitness; already in the watchlist.
- **arXiv:2310.01798** (Huang et al., *LLMs Cannot Self-Correct
  Reasoning Without External Feedback*, ref [5]) — foundational
  negative-result. Cite when defending sciknow's retrieval-grounded
  verification.
- **arXiv:2303.17651** (Madaan et al., *Self-Refine*, NeurIPS 2023,
  ref [4]) — the baseline autoreason beats; the pattern sciknow
  currently implements.
- **arXiv:2406.08598** (Zhao et al., *LLM Council*, ref [9]) — closest
  prior art to the judge panel.
- **arXiv:2502.18018** (Kalra & Tang, *Verdict*, ref [12]) — hierarchical
  judge verification. Paper tried and rejected it; save as "things we
  tested that didn't work".
- **arXiv:2310.03714** (Khattab et al., *DSPy*, ref [10]) — declarative
  pipeline compilation; study if we ever consider a programmatic
  prompt-optimization layer.
- **arXiv:2407.01085 / arXiv:2310.10076** (refs [18], [19], length-bias
  papers) — motivation for #2's length-controlled eval.
- **arXiv:2406.07791 / arXiv:2410.21819** (refs [21], [22], position-
  bias and self-preference in LLM-as-judge) — justifies #1's
  randomized presentation order.
- **arXiv:2306.05685** (Zheng et al., *MT-Bench*, ref [8]) — canonical
  LLM-as-judge methodology.

## Key files in the autoreason repo

- `experiments/v2/run_v2.py` — 514 LOC, canonical tournament loop.
  The file to crib from when implementing #1.
- `experiments/v2/compute_stats.py` — 131 LOC, bootstrap CIs + McNemar's.
  Direct port for #4.
- `paper/run_paper_autoreason.py:92-113` — ground-truth context
  injection for the paper-writing case study; compare with sciknow's
  stronger retrieval-backed verification.
- `paper/autoreason.pdf` Appendix A (pp. 27-28) — every role prompt is
  one paragraph. Direct source for the prompts in #1 and #2.

## Suggested shipping order

1. **#2** (CoT judges + length-controlled eval) — 2 h, nearly free,
   immediately improves the current scorer without touching the
   autowrite loop architecture.
2. **#3** (four-conditions gate) — 3 h, warning-only mode first.
   Saves compute on sections the loop can't help.
3. **#4** (bootstrap CI for bench) — 1.5 h, unlocks A/B comparisons
   with confidence intervals.
4. **#1** (tournament autowrite) — 4–6 h. Biggest architectural
   change; gate behind `--tournament` flag; default off on the first
   release, make default on after at least one book's worth of
   observation.

Total: ~11 hours of focused work for the whole stack, or 2 h for the
minimum-viable `#2` alone.
