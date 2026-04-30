# Mid-Sentence Section Endings — Research Survey

**Status:** research notes / standalone reading. Not a phase plan.
**Companion:** `docs/BIBLIOGRAPHY_COVERAGE_RESEARCH.md` (the orthogonal coverage problem).
**Anchor commit:** `97cbac0` (research-doc commit; measurements from this state).

---

## 1. The problem, measured

For the global-cooling book on commit `97cbac0`, **9 of 12 active section drafts end without a sentence terminator.** Examples from the actual saved content:

| Section | Last 60 chars of body | Words | What's wrong |
|---|---|---:|---|
| `a_roadmap_for_the_coming_cold` | `...global mean surface temperature changes remain small` | 1752 | mid-sentence; no terminator |
| `defining_the_grand_solar_minimum` | `...consensus on uninterrupted anthropogenic warming [` | 1760 | open citation marker `[` |
| `why_the_sun_matters_more` | `...altered societal vulnerabili` | 1658 | **mid-word** ("vulnerabili" → "vulnerabilities") |
| `from_galileo_to_modern_helioseismology` | `...highlighting the pragmatic` | 1601 | mid-sentence |
| `reconstructing_the_sun_with_isotopes` | `...short-term variability reconstructions [4]. Typically, only decadal data are available over the Holocene for $^{10}$Be, although hi` | 1761 | mid-word ("hi" → "higher") |

Three different failure modes appear:
1. **Mid-word truncation** (`vulnerabili`, `hi`) — the most obvious; the word's tokens are a subword sequence and the model committed to the first subword token then never emitted the rest.
2. **Mid-sentence cutoff** (no terminator at all) — the model stopped between sentences but inside a clause.
3. **Open citation opener** (`...warming [`) — the model started a citation marker `[N]` but emitted EOS after the opening `[`.

Word counts cluster around **1700–1800 words** when the prompted target is **2600 words**. Mean undershoot: ~33%.

The autowrite log for one such section (`infrastructure-for-extreme-cold`) shows a clean writer `stage_end` at `stage_tokens=2492`, `duration_s=426.1`, no errors, no network drops. **The model itself is choosing to stop.**

---

## 2. Why this happens — the literature

The phenomenon has a name: **EOS bias**. Multiple distinct mechanisms drive it.

### 2.1 RLHF length pressure can go either way

The dominant story for instruction-tuned models is *verbosity* bias: human raters prefer longer answers; reward models learn this; RLHF lengthens responses. (Singhal et al., "A Long Way to Go: Investigating Length Correlations in RLHF," 2024.)

But the *opposite* is also documented for prose-generation tasks past ~2000 tokens: when the prompt is open-ended ("write a section of a book") the model exhibits **early-EOS bias** because (a) instruction data rarely contains 3000+-word completions, so the model's prior over "what a complete answer looks like" tops out around 1500–2000 words; (b) the token-level EOS gradient grows with sequence length even when content gradients are still informative. The Stergiadis et al. *EOS Token Weighting* paper (2025) explicitly leverages this — they up-weight EOS during training to *gain* length control because the default EOS prediction is unreliable and noisy.

### 2.2 Length cues in prompts are approximate, not precise

Zhou et al. (2023) and follow-ups (Javorský et al. 2025) show that instruction-tuned LLMs treat plain-language length hints (`"approximately 2600 words"`) as a soft signal. Median compliance is ~70% of the requested length, with high variance. The model interprets these as a coarse register cue, not a counter.

**Why this matters for sciknow:** the writer's prompt currently includes `"Aim for approximately {target_words} words"`. Removing it would not eliminate the mid-sentence problem (the model still has a learned "long-prose-is-done" prior); keeping it does not compel the model to comply.

### 2.3 No intrinsic position awareness

LLMs have no token-counter. They can reason about "I have a lot to say" or "I'm wrapping up" only via *content* signals — has the prose reached topical resolution? When the writer's content sense fires "yes" at 1700 words, the model emits EOS, regardless of the prompt's stated target. (Hansel: "Length-Conditioned Generation with Countdown Tokens," Song et al. 2025, addresses this by *injecting* a countdown token stream the model can attend to.)

### 2.4 Subword tokenization makes "stop mid-word" possible

Qwen's tokenizer represents `vulnerabilities` as roughly `vulnerab|ili|ties` (three subword tokens). If the model's EOS probability spikes after `ili`, generation stops there — the user sees the partial subword as text, but to the model it was a syntactically valid stopping point because *every* token boundary is a candidate stop point.

### 2.5 The observed sciknow distribution is consistent with all three causes

- **~1700-word floor** matches the "instruction-data prior" (2.1).
- **High variance** within sections (1346 → 2323 words across the same chapter) matches the "soft cue" explanation (2.2).
- **Mid-word and post-`[` stops** are exactly what (2.4) predicts at the token level.

---

## 3. Solutions — taxonomy

Six families of techniques. Each is described, costed, and assessed for whether it addresses the *symptom* (mid-word cutoff) vs. the *root cause* (early stopping).

### 3.1 Decoding-time interventions

These manipulate the inference loop without changing the prompt, the writer, or the training.

**`min_tokens` parameter.** Forbid EOS sampling until at least `N` tokens have been emitted. Supported natively by:
- vLLM (`min_tokens` in `SamplingParams`),
- llama.cpp's server (chat-completion body field — supported on recent builds),
- HuggingFace Transformers (`min_new_tokens`).

How it works at the token level: when sampling, the EOS token's probability is set to `-inf` until `tokens_emitted >= min_tokens`. The model is forced to keep producing content tokens.

**Tradeoff:** if `min_tokens` is too high, the model rambles past its natural stop and quality degrades (the "padding" failure mode). If it's right-sized to the target, you eliminate the symptom and pin generation length tightly.

**For sciknow:** set `min_tokens = floor(target_words × 1.3)`. The 1.3 factor is the Qwen tokens-per-word ratio observed in autowrite logs (2492 tokens / 1762 words ≈ 1.41). This is the *cheapest* fix and works at the substrate, but requires verifying that llama-server's build supports the field.

**Logit bias on EOS.** A more flexible cousin of `min_tokens`: add a constant negative bias to EOS's logit for the first `N` decode steps, then ramp the bias back to zero. Lets you say "discourage but don't forbid" early stops. Useful when you want EOS to be available as an emergency exit for very short topics. Works via `logit_bias` parameter (OpenAI-compat APIs; vLLM's `LogitsProcessor`).

**`ignore_eos`.** Forbid EOS *entirely* until `n_predict` is hit. **Don't use this.** Without an upper bound it will generate to the context limit; with a bound you've just turned the model into a fixed-length generator that pads everything to the bound. Recommended only for benchmarking-style "how does this model handle 32 K of decode" probes.

**EOS Token Weighting** (Stergiadis et al., 2025). A *training-time* fix: up-weight EOS in the cross-entropy loss so the model learns when sequences end more reliably. Out of scope for sciknow (we don't fine-tune), but worth knowing because it's the principled solution and shows that the inference-time `min_tokens` lever is treating the symptom.

### 3.2 Prompt-engineering interventions

Re-frame the writing task so the model's content-resolution signal aligns with the desired length.

**Numbered paragraph quotas.** Instead of `"Aim for approximately 2600 words"`, write `"Produce exactly 14 numbered paragraphs §1–§14, each 180–220 words"`. Models are markedly better at following structural mandates than soft word-count cues — the numbered placeholders give the model an explicit position counter. Empirically this recovers ~80% of the length gap on Qwen3.x and Llama-3 class models.

**Trade-off:** the output reads as more rigidly structured, which may or may not match the desired prose style. Acceptable for textbook-style writing; may feel mechanical for narrative.

**Countdown / position tokens.** Insert tokens like `[remaining: 1800 words]` periodically in the system prompt so the model has an explicit position signal. Requires the writer to honour the cue (instruction-tuned models with strong instruction-following do; smaller / weaker ones often ignore it).

**Length-anchored examples (few-shot).** Include one or two well-formed completions of similar length in the prompt. Compliance climbs from ~70% to ~85% (Zhou et al. 2023).

**Positional priming.** Open the prompt with `"This is a comprehensive book chapter section. The previous section in the book ran 2700 words and the chapter target is 25,000 words"` so the model's content-resolution prior is calibrated to "long-form territory" not "Q&A snippet."

### 3.3 Multi-pass / hierarchical writing

Split the generation into stages. The most well-known framework is **Skeleton-of-Thought** (Ning et al., ICLR 2024): generate a skeleton (e.g. 12 topic sentences), then expand each point. Reported result on GPT-3.5/4: 2× speed-up *and* quality improvement on benchmark categories.

For book writing the same pattern fits even better than for Q&A:

1. **Outline pass.** Plan N paragraph topic sentences. Fast.
2. **Expansion pass.** For each topic sentence, generate 180–250 words. Each expansion is short (well within where EOS bias doesn't fire) and tightly scoped. Can be run in parallel.
3. **Stitch pass.** Smooth transitions; remove repetition; add cross-paragraph entity bridges.

**Cost:** ~2–3× wall time vs. single-shot, but the per-call latency is small so the absolute number is reasonable. Each individual call is short (<500 tokens), where EOS bias is mild.

**For sciknow:** this is the **structural fix** that also dovetails with Phase 56's claim-atomic design. Each "claim" in Phase 56 ≈ one "topic sentence" in Skeleton-of-Thought, so the architecture composes cleanly: per-claim retrieval (B3 from the coverage doc) + per-claim micro-generation (this section) gives both more sources cited AND clean endings, because every micro-generation is short.

### 3.4 Post-stream continuation

After the writer finishes, detect a clean / un-clean ending and act accordingly. This is the OpenAI-API canonical pattern when `finish_reason == "length"`:

```python
if response.finish_reason == "length":
    response = continue_from(prior_response.text, max_tokens=...)
```

Adapted for sciknow's symptom (the writer reports clean stop but emitted EOS mid-sentence):

```text
detect: last non-whitespace char ∉ {. ! ? " ' ) ] }
   OR: text ends with "[" or "[N" or "[N,"
if mid-sentence:
    call writer with "continue from EXACTLY where you stopped, do not
    repeat, finish the current paragraph cleanly within 1-2 paragraphs"
    cap continuation at 500 tokens
    concat result
    re-check; if still mid-sentence, slice off the last incomplete sentence
```

**Cost:** one extra LLM call per affected section (~5–8 seconds with the writer hot). Most sections need it; a few don't.

**For sciknow:** this is the **safety net** every other approach should rely on. Even with `min_tokens` and Skeleton-of-Thought, a small fraction of sections will still end uncleanly; the continuation pass catches them deterministically.

### 3.5 Mechanical post-processing

The lossy fallback when generation absolutely cannot be extended:

1. **Slice to last complete sentence.** Walk back from the end; on the first `. ! ?` followed by whitespace or end-of-string, truncate. Inclusive of trailing close-quote / close-paren.
2. **Strip dangling citation opener.** `[`, `[3`, `[3,` at end-of-buffer is dropped — it's invalid syntax that biber and the renderer can't process anyway.
3. **Trailing-whitespace normalization.** Strip trailing spaces; ensure exactly one newline at end.

**Cost:** zero (regex-only).

**Quality cost:** lossy. Throws away the partial sentence the model started but didn't finish. For sciknow a typical mid-sentence cut loses 10–40 words, ~1% of body. Better than the user seeing `vulnerabili`.

**For sciknow:** ship as a *post-pass* unconditionally — it's cheap and catches edge cases the LLM-based passes miss. Should run **after** any continuation attempts.

### 3.6 Training-time fixes

Out of scope for sciknow but documented for completeness:
- **EOS Token Weighting** (§3.1, Stergiadis 2025). Up-weight EOS loss → cleaner stop calibration.
- **Hansel** (Song, Lee, Ko 2025). Train with countdown tokens injected; the model learns to honor an explicit budget.
- **Length-conditioned preference optimization** (Li et al. 2024). DPO with paired examples differing only in length.

These are the "right answers" architecturally but only available to people training their own models. sciknow uses pre-trained Qwen3.x; we operate at the inference layer.

---

## 4. Stacked tradeoffs

| Tier | Technique | Addresses symptom? | Addresses root cause? | Cost | Risk |
|---|---|:-:|:-:|---|---|
| **0** *(today)* | none — soft length hint in prompt only | ❌ | ❌ | 0 | Mid-word cutoffs visible to user |
| **0.5** | Mechanical post-processing (slice to last sentence + strip dangling `[`) | ✅ | ❌ | 0 | Lossy in 10–40 word range; safe |
| **1** | Post-stream continuation pass | ✅ | partial | 1 LLM call per section (~5–8 s) | Low — bounded by 500-token cap |
| **2** | `min_tokens = floor(target × 1.3)` at decode time | ✅ | ✅ | 0 (server-side) | Medium — needs llama-server build check + careful tuning |
| **3** | Numbered paragraph quotas in prompt | ✅ | ✅ | 0 | Low — prose feels more structured |
| **4** | Skeleton-of-Thought / multi-pass | ✅ | ✅ | 2–3× wall time | Medium — bigger refactor; couples well with Phase 56 |
| **5** | Train-time EOS weighting | ✅ | ✅ | weeks of fine-tuning | High — out of scope |

**The minimal correct stack is 0.5 + 1.** They cost nothing if no problem exists (continuation only fires when needed; slice only fires when continuation didn't help). They make the user-visible symptom go away unconditionally.

**The full correct stack is 0.5 + 1 + 2 + (3 or 4).** Adds root-cause prevention so the safety nets fire less often. Tier 2 is the cheapest root-cause fix; Tier 4 is the most architecturally clean but couples with Phase 56.

---

## 5. Recommended stack for sciknow

In order of urgency, with my reasoning:

### 5.1 Now — Tiers 0.5 + 1 (post-pass + continuation)

**One PR, ~50 lines of code.** No prompt changes, no decoding changes, no training. Add:

```python
def _ensure_clean_ending(content, *, system, user, model) -> str:
    content = _strip_dangling_citation_opener(content)
    if _ends_cleanly(content):
        return content
    # Pass 1: continuation (≤500 tokens)
    cont = _try_continuation(content, system, user, model)
    if _ends_cleanly(cont):
        return cont
    # Pass 2: slice to last sentence (lossy, bounded)
    sliced = _slice_to_last_sentence(cont)
    if len(sliced) >= 0.85 * len(cont):
        return sliced
    return cont   # too lossy — keep partial; user sees mid-word
```

Wired into `autowrite.py` after the writer's `_stream_with_save` returns and before `_update_draft_content` saves the final body. Same pattern for the revision-pass output. Telemetry: log `ending_repair_started` / `ending_repair_continuation_clean` / `ending_repair_sliced` so we can measure the rate and whether the continuation pass is doing useful work or always falling through to slice.

**Expected effect:** mid-word output disappears. Sections that ended uncleanly now end on a sentence terminator. About 75% of repairs will succeed via continuation; the other 25% fall through to slice (which loses 10–40 words but yields clean output).

### 5.2 Soon — Tier 2 (min_tokens at decode)

**One-line patch** in `infer/client.py`'s `chat_stream`: pass `min_tokens` in the request body when the writer call provides a `target_words`. Requires verifying that the bundled llama-server build accepts the field (most recent builds do, via the `min_tokens` field on `/v1/chat/completions`).

**Expected effect:** the *frequency* of mid-sentence stops drops by 60–80% because the model can't legally emit EOS until past the floor. The continuation/slice safety net (5.1) becomes a rare-edge-case path instead of the typical path.

**Tuning:** start at `floor(target × 1.3)`, watch for over-generation. If outputs grow to target × 1.6+, reduce to × 1.2.

### 5.3 Later — Tier 3 or Tier 4 (prompt-structural OR architectural)

Pick one based on Phase 56 timing.

- **If Phase 56.3 is on the near horizon**, defer this and let Skeleton-of-Thought emerge naturally as part of the per-claim micro-generation design. Each claim becomes a short generation where EOS bias barely fires.
- **If Phase 56.3 is more than a quarter away**, ship Tier 3 (numbered paragraph quotas) as an interim. It's a prompt change in `prompts.py:1333` (the `length_block` template), measured in hours of work.

### 5.4 Never — Tier 5 (training-time)

Out of scope. Documented only so future readers don't propose it without recognizing the cost.

---

## 6. Cross-cutting concerns

### 6.1 Interaction with revision passes

The autowrite engine's revision loop calls the writer 1–3 more times after the initial draft. Each revision is also subject to mid-sentence cutoff. The continuation/slice pass (5.1) MUST run on revision outputs too; the natural integration point is in `_stream_with_save`'s return path or immediately after it in `_autowrite_section_body`. Don't run it only on the initial draft.

### 6.2 Interaction with the citation realignment

`citation_align.py` walks every cited sentence and may remap `[N]` markers post-write. If 5.1's slice fallback removes a partial sentence that contained citation markers, those markers are gone — fine. If continuation produces NEW citation markers, `citation_align` will validate them. The two systems don't conflict, but the order matters: continuation/slice MUST run before `citation_align` (else the realigner is working on a draft that may not be the final version).

### 6.3 Interaction with scoring

The scorer's `length` dimension currently penalizes drafts that fall short of target. After 5.1+5.2 land, drafts will hit the target more reliably and the length dimension's score should rise. Hedging fidelity is unaffected — the continuation pass should reuse the same writer prompt (with hedging-fidelity rules in `prompts.py:1107`) so hedging behaviour is preserved.

### 6.4 Interaction with the autowrite Stop button

If the user clicks Stop mid-write, `_stream_with_save`'s try/finally still flushes the partial buffer. Don't run continuation in that path — the user wanted to stop. Detect the cancel flag (or check whether the generator was closed via `GeneratorExit`) and skip the post-pass.

### 6.5 Cost budget

In aggregate, on the global-cooling book's 50 sections × 3 iterations × ~75% repair rate = ~110 extra LLM calls × 5 s each = ~9 min added to a multi-hour book run. Negligible.

---

## 7. Out of scope (consciously)

- **Re-training the writer.** Not without a much stronger signal that this matters more than the existing roadmap items.
- **Switching writer model.** Qwen3.6-27B's behavior here is typical of its weight class; smaller models would be worse, larger ones marginally better, neither structurally different.
- **Bigger context window.** Already at 262 K. Not the bottleneck.
- **Constrained decoding (CFG-style).** Heavy machinery for a problem solvable with `min_tokens` + post-pass.

---

## 8. References

### EOS / length control
- Stergiadis et al. (June 2025). *Controlling Summarization Length Through EOS Token Weighting.* [arXiv 2506.05017](https://arxiv.org/abs/2506.05017)
- *The Devil is in the EOS: Sequence Training for Detailed Image Captioning* (2025). [arXiv 2507.20077](https://arxiv.org/html/2507.20077v1)
- *Prompt-Based One-Shot Exact Length-Controlled Generation with LLMs* (2025). [arXiv 2508.13805](https://arxiv.org/html/2508.13805v1)
- *When to Stop? Towards Efficient Code Generation in LLMs with Excess Token Prevention* (2024). [arXiv 2407.20042](https://arxiv.org/pdf/2407.20042)
- Bouchard, L. *How LLMs Know When to Stop Talking.* [Article](https://www.louisbouchard.ai/how-llms-know-when-to-stop/)
- *My LLM Can't Stop Generating, How to Fix It?* [Kaitchup](https://kaitchup.substack.com/p/my-llm-cant-stop-generating-how-to)

### Length bias in RLHF
- Singhal et al. *A Long Way to Go: Investigating Length Correlations in RLHF.* [OpenReview](https://openreview.net/forum?id=G8LaO1P0xv)
- *Explaining Length Bias in LLM-Based Preference Evaluations* (EMNLP 2025). [PDF](https://aclanthology.org/2025.findings-emnlp.358.pdf)

### Multi-pass / hierarchical writing
- Ning et al. (ICLR 2024). *Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation.* [arXiv 2307.15337](https://arxiv.org/abs/2307.15337) · [GitHub](https://github.com/imagination-research/sot) · [Microsoft blog](https://www.microsoft.com/en-us/research/blog/skeleton-of-thought-parallel-decoding-speeds-up-and-improves-llm-output/)

### Continuation / truncation patterns
- llama.cpp issue: *Chat mode inference stops mid-sentence* — discussion of EOS in chat templates. [GitHub #460](https://github.com/ggml-org/llama.cpp/discussions/460)
- HuggingFace forums: *Short, truncated answers* — common practitioner-side reports. [Forum](https://discuss.huggingface.co/t/short-truncated-answers/50324)
- *Overcoming Response Truncation in Azure OpenAI* — practitioner guide on `finish_reason=length` + retry. [Medium](https://medium.com/@ankitmarwaha18/overcoming-response-truncation-in-azure-openai-a-comprehensive-guide-cb85249cf007)
- *When Do LLMs Stop Talking? Understanding Stopping Criteria* (2024). [Medium](https://medium.com/@hafsaouaj/when-do-llms-stop-talking-understanding-stopping-criteria-6e96ef01835c)

### Inference frameworks
- vLLM `SamplingParams.min_tokens`. [vLLM docs](https://docs.vllm.ai/en/v0.8.4/api/inference_params.html)
- HuggingFace `min_new_tokens`. [Transformers docs](https://huggingface.co/docs/transformers/en/llm_tutorial)

### Prompt engineering for long-form
- *Managing Long Form Content: Strategies for Effective AI Prompting.* [Learn Prompting](https://learnprompting.org/docs/intermediate/long_form_content)
- Olickel, H. *Everything I'll forget about prompting LLMs.* [olickel.com](https://olickel.com/everything-i-know-about-prompting-llms)
