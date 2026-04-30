# Benchmark methodology — hard-won rules

[&larr; Back to README](../README.md)

---

This document captures rules for benchmarking LLMs / VLMs / retrieval
inside sciknow **so we don't repeat the methodology failure of the
2026-04-17 model sweep**, where qwen3.5 / qwen3.6 scored 0 words on
every prose task and we drew the wrong conclusion about their
capability. They are genuinely strong models; our bench was broken.

The root cause, the rules that followed, and the checklist for any
future bench all live here. Read this BEFORE adding new candidates
or re-running a sweep.

---

## The incident in one paragraph

On 2026-04-17 we ran `sciknow bench --layer sweep` across 13
locally-installed LLMs. Qwen3.5:27b and Qwen3.6:35b-a3b-* emitted
**0 words of prose** on every generation task (`compile_summary`,
`write_section`, every quality-layer prose task) while correctly
identified "thinking runaways" were logged. We read this as
"thinking models are broken on our corpus" and published a report
that picked qwen3-instruct-2507 — the non-thinking variant — as the
winner and promoted it to `LLM_MODEL`. Qwen3-instruct-2507 scores
**MMLU-Pro 78 / GPQA 70 / SWE-bench 43** per the official HF card,
while qwen3.6-35b-a3b scores **MMLU-Pro 85 / GPQA 86 / SWE-bench 73**.
We picked the weaker model because the stronger models never got to
produce any output at all.

## Why the bench was broken (three compounding mistakes)

1. **`num_predict = 2048`**. Qwen's own documentation for every
   3.5+ family member states the floor for thinking mode is **16k
   tokens**, and recommends 32k. A thinking model's chain-of-thought
   commonly spans 4-16k tokens *before a single output token*. At
   2048, the CoT was truncated mid-thought and no answer emerged.
   We read the 0-word result as "bad model" instead of "insufficient
   budget".

2. **`temperature = 0`**. Qwen's HF cards explicitly say *"do not use
   temperature=0 — it causes repetition loops"*. We applied `temp=0`
   uniformly for JSON-extraction reproducibility, which was correct
   reasoning for gemma / qwen2.5 but wrong for the Qwen3.x family.

3. **No thinking-mode toggle**. Ollama exposes a native `think:
   true/false` flag in its chat API for hybrid (3.5/3.6) models.
   We never used it; couldn't A/B "thinking on vs thinking off" for
   the same model.

All three came from applying **one-size-fits-all inference
parameters** to a candidate list that included models with
materially different inference requirements.

## The rules

### R1 — Read the model's own docs before benching it

Before a new model family enters `CANDIDATE_MODELS`, pull up its
**official HF card or model-publisher blog** and record:

- Recommended `temperature`, `top_p`, `top_k`, `min_p`
- Recommended `num_predict` / max output tokens
- Whether it thinks by default
- Any format requirements (tool calling, JSON mode quirks)
- Known quirks on our target hardware (3090 VRAM, llama.cpp / Ollama)

These go in `sciknow/testing/model_sweep.py::profile_for()` as a
structured mapping. Do not invent numbers; cite the source in a
comment. If the card is silent on a parameter, pick a conservative
default but flag it in the profile as `tentative`.

### R2 — Never reuse budgets across architecturally different families

Thinking and non-thinking models belong in **different budget tiers**
on every task. Non-thinking models with `num_predict=2048` may be
fine; thinking models with the same value are broken. The bench's
`effective_budget(task, model)` wrapper in `model_sweep.py`
implements this: the task-level BUDGETS are bases, and
`profile_for(model)` scales / overrides per family.

If a candidate's output looks anomalous (0 words, repetition, empty
JSON), treat it as a **budget / sampling bug first, model bug second**
— invert the default debugging prior.

### R3 — Temperature=0 is not a universal safe default

It's only safe for architectures that were trained / RLHF'd against
it. Qwen3.x, Llama3.2, and several thinking models specifically break
under temp=0. Default should be the model's documented recommendation,
not zero.

### R4 — Multi-input panels, not single-input snapshots

Variance between test inputs is almost always larger than variance
between candidate models on a single input. The 2026-04-17 sweep ran
on **one paper** (4092d6ad). A paper-specific anomaly that broke
thinking models would have been indistinguishable from a
systematically-broken model. 54.6.85's fix: `CANDIDATE_PAPERS` is
three papers covering math-heavy, descriptive, and chart-heavy
archetypes; metrics average across them.

Rule: every sweep input list has ≥3 items spanning the inputs the
model will see in production. For the VLM sweep that's 15 figures
across figure/chart kinds; for retrieval it's 200 synthetic queries.

### R5 — When a candidate drops to zero, do not publish the report

If any model produces "empty" / 0-word / "all failed" output,
treat the sweep as **not yet valid**. Go back to R1-R3 and find the
misconfiguration before drawing conclusions. The 2026-04-17 report
should have been blocked here; instead we promoted it to a phase
commit and changed `LLM_MODEL` based on broken data. That's the
failure this rule prevents.

### R6 — Always compare your numbers to public benchmarks

Every candidate has public benchmark scores (MMLU, MMLU-Pro, GPQA,
Arena Hard, SWE-bench, ChartQA for VLMs). If your local sweep ranks
them wildly differently from every public benchmark, you are
probably measuring your own bug — not a corpus-specific effect.
A 40-point gap between local and public is a red flag; investigate
before publishing.

### R7 — Report the configuration, not just the verdict

Every sweep row in the output JSONL must include the `num_predict`,
`temperature`, `top_p`, `top_k`, and `thinks_by_default` that were
actually used. If those aren't visible, the result is
unreproducible and the whole file is waste paper. 54.6.85's
`_call_model_raw` persists all five in the result dict.

## The checklist for any future sweep

Before hitting `sciknow bench --layer sweep`:

- [ ] Every candidate in `CANDIDATE_MODELS` has an entry in
      `profile_for()` citing its HF card
- [ ] `effective_budget(task, model)` returns a budget consistent
      with that profile
- [ ] Test inputs (papers / figures / queries) are ≥3, representative
      of production
- [ ] `temperature=0` is used ONLY on models whose docs approve it
- [ ] The output JSONL will include the full per-call config (per R7)
- [ ] If a first pass shows any model at 0-output, pause and debug
      (per R5)

If you cannot tick all six, the bench will produce another 2026-04-17
event sooner or later. The rules exist because we paid for them.

## What changed in 54.6.85 (the fix)

See `docs/roadmap/PHASE_LOG.md` entry for Phase 54.6.85 — the full summary of
code changes (new `ModelProfile`, `profile_for()`, `effective_budget()`,
3-paper panel, `think` / `top_p` / `top_k` kwargs on `rag/llm`).
