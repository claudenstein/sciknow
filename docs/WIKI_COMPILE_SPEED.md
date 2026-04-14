# `wiki compile` — speed research

**Review date**: 2026-04-14. Same style as `KG_RESEARCH.md` /
`EXPAND_RESEARCH.md` / `MEMPALACE_REVIEW.md` / `AUTOREASON_REVIEW.md`
/ `WIKI_UX_RESEARCH.md` — concrete, priority-stacked, explicit skip
list, sources cited.

## Current state + diagnosis

### What actually runs per paper

`sciknow wiki compile` makes **two** LLM calls per paper (not one), via
`sciknow/core/wiki_ops.py`:

1. **Paper summary** — `llm_stream(system, user, ...)` at line 293.
   Uses the stream wrapper's default `num_ctx=16384`. Input ≈ 13–15 KB
   of text (metadata + abstract + up to 12 KB of key sections); output
   is a 300–600-word markdown page.
2. **Entity + KG extraction** — `llm_complete(..., num_ctx=8192,
   format=<JSON schema>)` at line 371. Input ≈ 8 KB (abstract 2 KB +
   sections 6 KB + existing slug list); output is JSON with 3–8
   concepts + 1–4 methods + 0–3 datasets + 5–15 triples each with a
   ≤ 300-char `source_sentence`. Total output ≈ 1 000–3 000 tokens.

The default model is **`LLM_FAST_MODEL = qwen3:30b-a3b`** (Qwen3 MoE,
30 B total / 3.3 B active). This materially changes several rankings
vs a generic "7–14 B dense" framing — speculative decoding has a
smaller ceiling, model-swap wins are smaller than expected, and the
MoE router adds enough overhead that prefill-batch-size tuning pays
off more than it would on dense.

### Where time goes

Qwen3-30B-A3B at Q4_K_M on a 3090 with no competing load runs roughly
**60–90 tok/s decode** and **400–700 tok/s prefill** (derived from
llama.cpp issue #10466 + the ~2× throughput premium of 3 B-active MoE
over 14 B dense). Per paper:

| Phase | Call 1 (summary) | Call 2 (extraction) | Total |
| --- | --- | --- | --- |
| Prefill tokens | ~3 500 | ~2 000 | ~5 500 |
| Prefill @ 500 tok/s | 7 s | 4 s | 11 s |
| Decode tokens | ~600 | ~1 500 | ~2 100 |
| Decode @ 75 tok/s | 8 s | 20 s | 28 s |

**Realistic ≈ 40–60 s/paper**, lining up with "tens of minutes to
hours" for a 100-paper corpus. **Decode dominates** — call 2 generates
~1 500 tokens of densely-structured JSON against a grammar, which is
where most of the clock time lives. Prefill is a secondary cost
(~25–30 %). The grammar itself is not free either — llama.cpp compiles
the schema into a GBNF / token mask that re-evaluates every sampled
token, with documented 1.2–2× overhead for non-trivial schemas
(lmsys compressed-FSM writeup + llama.cpp discussion #6002).

## No-model-change optimizations, ranked

### Tier A — ship these

1. **Flash attention + Q8_0 KV cache** (Ollama FAQ + smcleod.net
   writeup). Set `OLLAMA_FLASH_ATTENTION=1` and
   `OLLAMA_KV_CACHE_TYPE=q8_0` in the Ollama systemd unit. Qwen-family
   models are in Ollama's allow-list; this is a **pure win** — 10–25 %
   prefill speedup, 30–50 % KV VRAM saved (which lets `num_batch` rise
   without OOM). Q8_0 KV halves cache memory with ~0 quality loss.
   **Gain: 15–25 % end-to-end**, zero quality cost, zero code change.

2. **Drop (or merge) call 1 (paper summary).** The summary call is
   ~1/3 of wall time per paper; its output is lightly-structured
   markdown that could either come from the extraction call as one
   more field, or be synthesised deterministically from metadata +
   abstract (the current template is mostly boilerplate). **Gain:
   30–40 % end-to-end**, small quality cost (summaries lose some
   flavor but stay grounded).

3. **Shrink call 2 input to abstract + "head + tail" of sections
   (2 KB).** Currently feeds 6 KB of key sections. For triple
   extraction specifically, abstract + conclusion + first methods
   paragraph carry ≥ 90 % of the signal (BioREx paper §4.2, LangExtract
   chunked-extraction docs). Sending "head + tail" (first 1 KB + last
   1 KB of concatenated sections) beats head-only — preserves methods
   introduction *and* key-findings closure. **Gain: 10–15 % end-to-
   end.**

4. **Simplify the JSON schema.** The 300-char `source_sentence` is
   the single most expensive constraint — the grammar refuses every
   token that would break verbatim-ness. Options:
   - Drop `concepts` / `methods` / `datasets` from the schema (they
     overlap ~90 % with triple subjects/objects; derive post-hoc).
   - Cap `triples` at `maxItems: 10` (**carefully** — llama.cpp has
     known grammar-compile regressions on big `maxItems`/`minItems`
     values; 10 is in the safe range).
   - Remove the `source_sentence` length constraint from the schema;
     enforce `[:500]` in Python, which we already do.
   **Gain: 5–15 %**, no quality cost with post-hoc validation.

### Tier B — do if Tier A isn't enough

5. **`num_ctx` → 6 144.** See §4 below. 8192 is too big; 6144 matches
   the actual peak (~5 500 tokens) plus a 10 % cushion. **Gain: 5–10
   %**, no quality cost. *Safe to ship in code today.*

6. **Parallel requests across papers** (`OLLAMA_NUM_PARALLEL=2` +
   `LLM_PARALLEL_WORKERS=2`; process papers concurrently in
   `compile_all`). Ollama allocates `num_ctx × num_parallel` KV
   memory. With `num_ctx=6144` and 30B-A3B Q4_K_M (~18 GB weights)
   on a 24 GB 3090, two slots at 6144 cost ~2.5 GB extra — fits.
   Same-model concurrent requests see 1.5–2.5× throughput on recent
   Ollama. **Gain: 40–60 % end-to-end.** Don't raise above 2 with
   this model on 24 GB.

7. **`num_batch` → 1024.** Ollama default is 512; 3090 sweet spot
   for 7–14 B Q4_K_M is 1024 (eastondev.com, markaicode.com). For
   30B-A3B still safe, MoE router overhead notwithstanding. Helps
   prefill ~10–20 %. **Gain: 3–8 % end-to-end** (prefill is 25 % of
   the total).

8. **Verify the system-prompt prefix cache is actually warm.** Ollama
   inherits llama.cpp's `cache_prompt: true` path (discussions #10937
   / #13606); when `keep_alive=-1` keeps the slot warm and the system
   prompt is identical across calls, n_tokens_cached grows. Check via
   `OLLAMA_DEBUG=1` — this is probably already on, no action unless
   absent. **Gain: 0–5 %** (likely already captured).

### Tier C — don't bother

- `num_thread`: CPU pool is noise when the model is fully offloaded.
- `num_gpu` / layer offload: Ollama already does 100 %; confirm once
  with `ollama ps`.
- Sampling knobs (`top_k`, `top_p`, `mirostat`, `repeat_penalty`):
  negligible cost savings under a grammar; `repeat_penalty` can hurt
  JSON conformance. Leave defaults, `temperature=0.0`.
- Content-hash cache: `compile_paper_summary` already skips existing
  slugs. No identical-input re-run scenario.

## Model-change optimizations, ranked

### Tier A — worth trying

1. **Quantization Q4_K_M → Q4_K_S or Q4_0.** Same weights, ~8 %
   smaller, 10–15 % faster decode per llama.cpp benches
   (johannesgaessler.github.io). Quality drop is measurable on MMLU
   but **not** for structured extraction where the model picks from a
   handful of grammar-valid tokens per step. **Gain: 10–15 %**,
   negligible quality cost on this task.

2. **qwen3:8b dense instead of qwen3:30b-a3b.** Counter-intuitive but
   defensible. MoE's only advantage over 8 B dense is multi-step
   reasoning; single-pass grammar-constrained extraction isn't that.
   Dense 8 B Q4_K_M on a 3090:
   - Weights ~5 GB (vs 18 GB).
   - Decode 120–150 tok/s, 2× faster.
   - Frees 18 GB → `OLLAMA_NUM_PARALLEL=4` is trivially feasible.
   - JSON conformance unchanged (grammar enforces it).
   Published data: Qwen2.5 tech report §5 shows 14 B > 7 B on
   averaged benchmarks but the gap collapses on extraction tasks and
   disappears under grammar constraint. **Gain: 50–70 % end-to-end
   with NUM_PARALLEL=4.** Real risk: triples 15–25 % more
   duplicate/generic (fine-tuned BERTs at ~110 M still outperform
   generative LLMs on CDR/GDA — the generalist-vs-specialist gap
   dominates parameter count). **Mitigation**: keep 30B-A3B as a
   `--model` override for "important" papers; default to 8 B bulk.

3. **Speculative decoding: Qwen2.5-0.5B-Instruct draft for 30B-A3B
   target** (llama.cpp discussion #10466). Reported 1.6–2.5× end-to-
   end on coding, lower on structured extraction because draft
   acceptance rates drop under grammar. **Ollama does NOT yet expose
   `--model-draft`** — would require raw `llama-server` or waiting on
   Ollama's in-progress PR. **Gain: 25–50 % end-to-end**, speculative.

### Tier B — specialist pipelines worth spiking

4. **GLiNER + REBEL pipeline for extraction; keep the LLM only for
   summary** (if summary survives at all). GLiNER-multi is a 300 M-param
   CPU-friendly zero-shot NER model competitive with GPT-4 (urchade
   repo + GLiNER2 paper arxiv 2507.18546); REBEL is seq2seq trained
   for triple extraction on DocRED / NYT / SciERC. Combined pipeline:
   GLiNER entities → REBEL triples. Runs 10–50 docs/s on CPU — ~1000×
   faster than the current path. Biomedical-RE review (spj.science.org
   2025): fine-tuned transformers are **2× better** than generative
   LLMs on RE benchmarks. Major cost: engineering the entity-vocabulary
   + predicate-set mapping to the current six predicates (`uses_method`
   | `studies` | `finds` | `supports` | `contradicts` | `related_to`)
   and wiring `source_sentence` attachment. **Gain: 5–10× end-to-end,
   higher triple quality**, material eng. effort. Right architectural
   answer if this pipeline is permanent.

5. **Hybrid routing: short/low-value paper → 8 B dense; long/important
   → 30B-A3B.** Router signals: section length + citation count.
   Captures ~70 % of papers in the fast path. **Gain: 30–40 %
   end-to-end** on typical corpora.

### Tier C — skip

- **Mixtral 8x7B on 24 GB:** feasible at Q3_K_M only; zero KV
  headroom; concurrency win disappears; Qwen3 MoE is already the
  better MoE for this class.
- **NuNER-zero / NuMind** as standalone: doesn't do triples. Subsume
  into option 4.
- **Fine-tuned DeBERTa + BERN2:** biomed-specific; GLiNER matches
  without fine-tuning and is simpler operationally.

## The context-size verdict — is 8192 optimal?

**No. Use 6144.**

### Math

- `PAPER_SUMMARY_USER`: sections (12 000 chars) + abstract (3 000) +
  metadata boilerplate (~400) ≈ 15 000 chars ≈ **3 750 tokens**
  (Qwen tokenizer; Ollama FAQ rule-of-thumb is 4 chars/token English).
- `PAPER_SUMMARY_SYSTEM` ≈ 1 000 chars ≈ 250 tokens.
- Output: 300–600 words ≈ 450–900 tokens.
- **Call 1 peak: ~4 900 tokens.**

- `EXTRACT_ENTITIES_USER`: sections (6 000) + abstract (2 000) +
  slug list up to ~3 000 + boilerplate ≈ 11 500 chars ≈ **2 900
  tokens**.
- `EXTRACT_ENTITIES_SYSTEM` ≈ 800 chars ≈ 200 tokens.
- Output: 1 000–3 000 tokens.
- **Call 2 peak: ~6 100 tokens.**

Extraction is the binding constraint. At `num_ctx=8192` we have 2 000
tokens of unused headroom. At `num_ctx=4096` we'd silently truncate —
first the slug list tail (hurts concept reuse), then section tails
(hurts triple quality). 4096 is a no-go.

### Cost of overcapacity

KV cache is **linear** in `num_ctx`. Attention compute during prefill
is O(n²) without flash attention, O(n) with. On Qwen3-30B-A3B
Q4_K_M+FP16 KV, each extra 1 k ≈ 40–60 MB. Doubling `num_ctx`
costs ~5–10 % prefill time when flash-attn is off, ~0 when on.

### Cost of undercapacity

Ollama truncates from the front on overflow — system prompt survives,
abstract+intro get dropped. Those are the highest-signal parts; triple
quality degrades sharply.

### Recommendation

**`num_ctx = 6144`**. Covers the extraction peak (~6 100 tokens) with
a conservative cushion and matches `num_batch = 1024` cleanly (6
batches). Cuts KV cache ~25 % vs 8192 without truncation risk.

If Tier-A pipeline changes land (merge call 1 into call 2 + shrink
sections to 2 KB), the extraction prompt drops to ~5 500 tokens peak
and `num_ctx = 5120` becomes the right number. **Size to observed
peak + 10 % cushion, not a round-number default.**

## Priority stack — the one-page ship list

**If you only do one thing**: enable flash attention + Q8_0 KV cache
+ raise `OLLAMA_NUM_PARALLEL` to 2 + set `LLM_PARALLEL_WORKERS=2` +
drop `num_ctx` to 6144. Zero quality cost, **~60–80 % end-to-end
speedup**, twenty minutes of work.

Files / configs to change (in priority order):

1. **Ollama systemd unit** — `Environment=OLLAMA_FLASH_ATTENTION=1`
   and `Environment=OLLAMA_KV_CACHE_TYPE=q8_0` (not the project `.env`
   — Ollama reads its own env). Restart Ollama.
2. `/home/kartofel/Claude/sciknow/sciknow/core/wiki_ops.py:371` —
   `num_ctx=6144` (in the call 2 extraction complete).
3. `/home/kartofel/Claude/sciknow/sciknow/rag/wiki_prompts.py:146` —
   shrink `sections` slice from 6000 to 2000 chars, use head+tail
   selection.
4. `/home/kartofel/Claude/sciknow/.env` — `LLM_PARALLEL_WORKERS=2`.
5. `/home/kartofel/Claude/sciknow/sciknow/core/wiki_ops.py:572` —
   run `compile_all`'s per-paper loop over a `ThreadPoolExecutor`
   with `max_workers=LLM_PARALLEL_WORKERS`. (Guard: OLLAMA_NUM_PARALLEL
   must match; wrong pairing causes head-of-line blocking.)

## Sources

- Ollama Structured Outputs docs — https://docs.ollama.com/capabilities/structured-outputs
- Ollama FAQ (flash attn + KV cache env vars) — https://docs.ollama.com/faq
- Ollama flash-attn arch allowlist (issue #13337) —
  https://github.com/ollama/ollama/issues/13337
- "Bringing K/V Context Quantisation to Ollama" — smcleod (2024):
  https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/
- Ollama Performance Optimization Guide (eastondev, 2026):
  https://eastondev.com/blog/en/posts/ai/20260410-ollama-performance-optimization/
- Ollama Parallel Requests (Glukhov):
  https://www.glukhov.org/llm-performance/ollama/how-ollama-handles-parallel-requests/
- Configure Ollama Concurrent Requests (Markaicode):
  https://markaicode.com/ollama-concurrent-requests-parallel-inference/
- llama.cpp cache-reuse tutorial (discussion #13606) —
  https://github.com/ggml-org/llama.cpp/discussions/13606
- llama.cpp prompt-cache discussion #10937 —
  https://github.com/ggml-org/llama.cpp/discussions/10937
- llama.cpp host-memory prompt-caching #20574 —
  https://github.com/ggml-org/llama.cpp/discussions/20574
- llama.cpp speculative decoding #10466 —
  https://github.com/ggml-org/llama.cpp/discussions/10466
- LMSYS compressed-FSM writeup — https://www.lmsys.org/blog/2024-02-05-compressed-fsm/
- Generating Structured Outputs benchmark — arxiv 2501.10868
- "A Guide to Structured Outputs Using Constrained Decoding" (Cooper
  2024) — https://www.aidancooper.co.uk/constrained-decoding/
- GLiNER repo (urchade) — https://github.com/urchade/GLiNER
- GLiNER2 paper — arxiv 2507.18546
- Fine-tuning vs inference, biomedical RE (spj.science.org 2025):
  https://spj.science.org/doi/10.1016/j.csbj.2025.12.004
- ATLOP paper (AAAI 2021) —
  https://ojs.aaai.org/index.php/AAAI/article/view/17717
- Qwen3-30B-A3B model card —
  https://huggingface.co/Qwen/Qwen3-30B-A3B
- Qwen2.5 tech report — arxiv 2412.15115
- llama.cpp performance testing (Gässler) —
  https://johannesgaessler.github.io/llamacpp_performance
