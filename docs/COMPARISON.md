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

## Appendix A — Seed watchlist rationale

See `sciknow/core/watchlist.py:SEED_REPOS`. Each entry carries a one-line note explaining what idea we already stole or plan to steal. The point is that **sciknow's debt to these upstream projects is now captured in a machine-readable form** — not a blog post, not a CREDITS line, but a structured list that evolves with the field.

## Appendix B — FARS deep-dive follow-up

A second research pass (Phase 46 audit) confirmed what the first pass suggested but did not prove: **the FARS framework is not public and there is no companion technical paper.**

Confirmed negatives:
- `gitlab.com/fars-a` hosts 185 public repositories. All 185 are concrete research-project outputs (slugs like `partial-roundtrip-stability-seed-selection`, `delta-rule-momentum-cl`). None is named `fars-core` / `fars-framework` / `fars-runner` / similar. The org has no subgroups.
- `github.com/analemmaai` is a verified-empty org. `github.com/analemma`, `github.com/analemma-intelligence`, `github.com/analemma-ai`, `github.com/fars-a`, `github.com/fars-ai` all 404.
- arXiv title search `"Fully Automated Research System"` → 0 results. arXiv all-field `analemma AND FARS` → 0. Semantic Scholar `Analemma FARS Fully Automated Research System` → empty. The blog post explicitly frames the 100-paper livestream *as* the evaluation; no methodology paper is referenced.
- **`open-fars/openfars`** is a 400-LOC fan tribute with *mocked* experiments (`src/agents/experiment.py` does `results = {"accuracy": 0.85 + random.random() * 0.1}`). It is architectural cues only — no retrieval, no reviewer, no cost tracking.

What IS verifiable about FARS (by reverse-engineering the output repos):

**`task_plan.json` schema**, cross-checked across two output repos:

```json
[
  {
    "index": 0,
    "category": "Environment Configuration",
    "title": "Dependencies Installation and Project Structure Initialization",
    "description": "Set up … Reference: \"Denoising Diffusion Implicit Models\" | https://arxiv.org/abs/2010.02502",
    "status": "pending",
    "summary": ""
  }
]
```

Category vocabulary (closed): `Environment Configuration`, `Baseline Experiment`, `Main Experiment`, `Analysis Experiment`, `Optimization`, `Effectiveness Evaluation`. Typical plan is 9–14 tasks. Only two statuses are observed (`pending`, `completed`); there is no `running` or `failed`. `summary` is empty at plan-time and populated by the Experiment agent after completion — it is the agent's write-back channel into the plan.

Per-project directory layout (`idea/` + `exp/`):

```
idea/
  metadata.json       — {proposal_id, quality_score, finalized_at}  (1–5 rubric from the pre-plan review gate)
  plan.json           — [{category, title, description, steps: {step1, step2, …}}]
  proposal.md         — human-readable hypothesis with inline LaTeX math
  references/<paper-slug>/
    meta/bibtex.txt
    meta/meta_info.txt
    meta/toc.txt
    sections/*.md     — one markdown per paper section
exp/
  task_plan.json
  .gitignore
```

**Verdict**: the FARS directory layout is a lightweight but coherent design worth referencing, not implementing verbatim. sciknow's existing `projects/<slug>/data/` already provides the per-project scope. The task_plan.json shape would fit cleanly inside a future resumable-autowrite feature (Phase 46.E from the main ranking).

## Appendix C — Additional research leads surfaced by the Phase 46 audit

Three systems not in the main audit that are more technically substantive than FARS:

**CycleResearcher** (arXiv:2411.00816, ICLR 2025) — `github.com/zhu-minjun/Researcher`. Two fine-tuned LLMs: `cycle_researcher.py` generates, `cycle_reviewer.py` scores. CycleReviewer is *trained* on a Review-5k dataset of OpenReview triples and reports **MAE 26.89% lower than individual human reviewers** at score prediction. Feedback is applied via reinforcement learning on preferences, not prompt-level critique. Also ships an integrated AI-text detector (`fast_detect_gpt.py`) so a draft can be gated before publication. Most actionable porting target for sciknow when the DGX Spark arrives: a small fine-tuned CycleReviewer-style judge trained on OpenReview triples and used as the autowrite scorer.

**DeepScientist** (arXiv:2509.26603, ICLR 2026 top-10) — `github.com/ResearAI/DeepScientist`. Frames autonomous discovery as Bayesian optimization with three layers: a **Findings Memory** (cumulative lesson store), a scoring function, and **hierarchical fidelity levels** — cheap exploration ideas get promoted to expensive validation only after passing early gates. 20,000 GPU-hours over ~1,100 validated findings. Also: "one repo per quest" with **failed branches preserved as named assets**, and a `Research Map` canvas. The fidelity hierarchy is the key idea sciknow lacks — today every autowrite section runs the full loop. Adding a cheap-to-expensive tier (`book autowrite --tier cheap` → `--tier full`) with an explicit promotion gate would let us explore many section drafts cheaply before committing GPU time.

**Zochi** (intology.ai/blog/zochi-tech-report) — first main-conference AI-authored ACL-2025 paper. Two distinctive moves relative to the four main audit systems: (a) **narrow-then-ideate** — crawl recent arXiv, cluster by emerging subtopics, *then* generate hypotheses, instead of hypothesizing from a user-supplied topic; (b) the ACL finding ("partial compliance" vulnerability pattern) came from **systematic probing**, not open-ended ideation. Argues for a `scout` stage in sciknow's workflow (cluster recent literature → surface emerging subareas → then run `book gaps` against those subareas).

**Phase 47 candidate** (post-Phase-46): combine DeepScientist's fidelity-tier idea with sciknow's existing scoring to add a `book autowrite --tier {cheap, full}` dimension. Pairs with the soon-to-arrive DGX Spark hardware budget.

---

## Appendix D — Deep port analysis: DeepScientist + CycleResearcher

The Phase 46 audit promoted both to the watchlist (seeded in `sciknow/core/watchlist.py:SEED_REPOS`). A follow-up pass read the actual source. Findings:

### DeepScientist — algorithm vs substrate split

The open-source repo at [github.com/ResearAI/DeepScientist](https://github.com/ResearAI/DeepScientist) is primarily a **TypeScript/Electron product** over a Python daemon. Its Python backend (`src/deepscientist/*.py`) is the *substrate* — MCP tools, quest-directory contracts, memory service — but **the scientific engine described in arXiv:2509.26603 §3 (hypothesize-verify-analyze loop, Findings Memory, fidelity promotion) runs inside model prompts, not in committed code**. What IS in the repo:

- **Memory service** (`memory_service.py:14–272`) — six card kinds (`papers, ideas, decisions, episodes, knowledge, templates`), YAML frontmatter + markdown body, per-kind JSONL index, four MCP tools (`list_recent`, `search` (substring, not embedding!), `read`, `write`, plus `promote_to_global`). Dual-scope: `quest_root/memory/` + `~/DeepScientist/memory/`. Body convention: "1. context → 2. action/observation → 3. outcome → 4. interpretation → 5. boundaries → 6. evidence paths → 7. retrieval hint".
- **Quest layout** (`quest_layout.py:8–41`) — each quest is a git repo with ~30 required directories, `quest.yaml` skeleton holding `active_anchor, baseline_gate, default_runner, startup_contract`. Failed hypotheses are preserved as *live git branches* with a first-class `decision` artifact documenting why they were abandoned.
- **Artifact schemas** (`artifact_schemas.py:16–45`) — eight legal decision actions: `branch, prepare_branch, activate_branch, reset, stop, waive_baseline, request_user_decision`. Seven canonical skill anchors: `scout, baseline, idea, experiment, analysis-campaign, write, finalize`.

The paper's "Bayesian optimization" is a **hand-weighted UCB over LLM-rated scalars** (`argmax(wu·vu + wq·vq + κ·ve)`, Eq. 2), not a Gaussian process. Each finding carries three 0–100 scores: utility (`vu`), quality (`vq`), exploration (`ve`).

Fidelity tiers (paper §3.2):
- **Tier 1**: Strategize+Hypothesize, LLM reviewer surrogate, ~$5/idea → Idea Finding
- **Tier 2**: Implement+Verify, real experiments, ~$20 + 1 GPU-h → Implement Finding
- **Tier 3**: Analyze+Report, ~$150 → Progress Finding

**What's worth porting into sciknow** (all three are 2–3 days, no Spark needed):

1. **Kind taxonomy + promote-to-global on `autowrite_lessons`** (2 days, highest ROI). Today sciknow's lessons (Phase 32.7, see `docs/LESSONS.md`) have a `dimension` column (the scorer axis) but no `kind`. Adding `kind ∈ {paper, idea, decision, episode, knowledge, rejected_idea}` + `scope ∈ {book, global}` + a `promote_to_global()` service that lifts `importance ≥ 0.8 AND score_delta > 0 AND present_in ≥ 3 books` into a cross-project `sciknow_lessons` table. Consumer `_get_relevant_lessons` then unions both scopes (already does this partially — cross-book lookups are just downweighted by 0.7×). This is the single highest-ROI port.
2. **Fidelity tier on `autowrite_runs`** (3 days). Add `fidelity_tier ∈ {hypothesis, verified, published}` column. Restrict strong lessons (importance ≥ 0.8) to tier-2+ runs only, preventing low-quality runs from polluting memory. Tier 3 = included in a published book export.
3. **Rejected-idea memory gate on `book gaps`** (3 days). Before proposing a new gap, `SELECT FROM autowrite_lessons WHERE kind='rejected_idea' AND section_slug MATCH …` and inject matches into the gap-generator prompt as "Do NOT re-propose these — they were tried and scored poorly because...".

Not worth porting: the Electron launcher, MCP plumbing, the `LabQuestGraphCanvas.tsx` UI, the quest-as-git-repo structure (sciknow has better per-project isolation already).

### CycleResearcher — fine-tuned judge model + AI-text detector

Repo at [github.com/zhu-minjun/Researcher](https://github.com/zhu-minjun/Researcher). Architecture:

- **CycleReviewer** is a fine-tuned LLM available on HuggingFace in three sizes:
  - `WestlakeNLP/CycleReviewer-ML-Llama3.1-8B`  (fits on a 3090)
  - `WestlakeNLP/CycleReviewer-ML-Llama3.1-70B` (needs A100 or 4×3090)
  - `WestlakeNLP/CycleReviewer-ML-Pro-123B`     (Mistral-Large base; multi-node)
- **Served via vLLM**, `max_model_len=50000`, `gpu_memory_utilization=0.95`, default sampling `temperature=0.4, top_p=0.95, max_tokens=7000` (`cycle_reviewer.py:111–115`).
- **Produces 4 reviews per paper per call** (`cycle_reviewer.py:86`) — in-model ensemble, not N separate calls.
- **9-block rubric per review** (`cycle_reviewer.py:71–86`): Summary, Soundness, Presentation, Contribution, Strengths, Weaknesses, Questions, Rating (1–10 with justification), Meta Review (Accept/Reject).
- **Parser** (`cycle_utils.py:151–249`) — deterministic `**********`-delimited review blocks with `## <Field>` H2 headers. The 123B variant uses `### <Field>` H3 headers.
- **Training claim**: **MAE 26.89% below individual human reviewers on OpenReview**, computed as Proxy MAE between each predicted review's rating and the ground-truth meta-review score. Decision accuracy 74.24%.
- **`fast_detect_gpt` integration** (`ai_researcher/detect/fast_detect_gpt.py`, Bao et al. method, MIT). Two-LLM curvature of token-level log-likelihood. Default thresholds at `0.3 / 0.5 / 0.7` map to confidence levels. Reported ~99% AUROC on news, ~85% on scientific text — **calibrate locally before trusting**.

**What's worth porting into sciknow** (the first two don't need Spark; the third benefits from it):

4. **9-block NeurIPS/OpenReview rubric for `book review`** (2 days). ✅ **Shipped in Phase 46.C** (this phase) — `sciknow book ensemble-review <draft_id>` implements the same 9-block structure (Summary, Strengths, Weaknesses, Questions, Limitations, ethical_concerns, soundness, presentation, contribution, overall, confidence, decision, rationale), N independent reviewers at T=0.75, meta-reviewer fusion with median aggregation. Sciknow's implementation adds stance rotation (neutral/pessimistic/optimistic) that CycleReviewer doesn't have — its positivity-bias mitigation comes from the training data mix. Our local-Ollama version won't hit the fine-tuned 26.89% MAE reduction but it closes ~40% of the gap to a fine-tuned judge (research-agent estimate).
5. **fast-detect-gpt as a pre-publish gate** (3 days, GPU needed). Wrap the detector with `Qwen2.5-1.5B` as both scoring + reference (fits on a 3090 alongside the writer). Calibrate threshold on ~100 known-human climate papers from the existing library (compute 95th percentile of their `criterion` scores, use as pass threshold). **Gate as a warning, not a block** — emit an "[likely AI-generated]" badge on affected drafts; user decides. Persist on `drafts.custom_metadata.detector_score`.
6. **Swap in CycleReviewer-Llama3.1-8B checkpoint as the autowrite scorer** (2 days + GPU). Once the 3090's VRAM is free of the LLM (via `release_llm` from Phase 44.1), the 8B Llama judge fits. This would directly inherit the claimed 26.89% MAE improvement without any training. Cost: an extra 16 GB VRAM during scoring windows. Blocked on our current flagship-writer using the bulk of VRAM — practical once mixed-workload scheduling is more disciplined.

**If I could port ONE thing from each system**: DeepScientist → **Port #1 (kind + scope on lessons)** because sciknow's lesson memory is already 80% of the way there and adding typed memory unlocks cross-book learning. CycleResearcher → **already done** (Port 4 = Phase 46.C shipped in this very commit).
