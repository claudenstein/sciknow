# Substrate optimization benchmark plan (Phase 55.V10/V15)

## 2026-04-28 results — slates #1, #1b, #3 measured

3090 (24 GB) + Qwen3.6-27B-UD-Q4_K_XL (17.6 GB on disk), llama.cpp from
`LLAMA_SERVER_BINARY`. Baseline = expert (np 1, fa on, ngl 999).
Workload = `typical` (5K input + 1500 output, reps=3).

```
   ctx           f16             q8_0            q4_0
 16384    37.8 t/s · 18.4 GB    36.5 · 17.9    36.1 · 17.7
 24576    37.9 t/s · 18.9 GB    36.5 · 18.2    36.0 · 17.8
 65536    37.8 t/s · 21.4 GB    36.4 · 19.5    36.0 · 18.5
131072         OOM              35.5 · 21.8    36.0 · 19.8
262144         OOM                 OOM         36.0 · 22.4
```

(VRAM ±0.5 GB — bench occasionally samples after teardown starts.)

**Settled findings:**

- **"262K on 24 GB" claim verified for q4_0 only.** q8_0 needs ~10 GB
  KV at 262K → can't fit (proved on both Q4_K_M and Q4_K_XL). f16
  OOMs at 131K and beyond.
- **f16 is fastest where it fits** (37.8 t/s, ≤65K) — no dequant
  overhead.
- **q4_0 beats q8_0 at 131K** (36.0 vs 35.5 t/s) — bandwidth wins
  over dequant cost once cache is large.
- **Decode TPS is essentially flat across context inside the feasible
  region.** Only q8_0 shows a measurable drop (~2.5%, 65K→131K).
  The viral "flat 40 t/s curve" claim has the right shape, just at
  ~36-38 t/s on this hardware/build.
- **q8_0:q4_0 asymmetric K:V is structurally broken** on this
  llama.cpp build. Server boots, inference hangs past 300s at every
  ctx tested (16K, 262K). Likely a missing fused flash-attn kernel
  for asymmetric K/V quants. Tracked separately; do not promote.

**Production decision (with slate #2 quality probe parked):**

- ≤65K ctx: any cache type works; **f16 wins on speed**. Production
  default writer ctx=24K should stay on f16 KV.
- 65K–131K: **q4_0** wins (faster than q8_0 at 131K, more headroom).
- ≥131K: **q4_0 only.**

Production-relevant ctx for autowrite is ≤24K (5–18K input), so the
current f16 KV default is correct. Don't promote q4_0 just for
memory savings — it costs ~1.7 t/s decode at typical workloads.

**Methodology note (bug):** the bench harness's `--workload large`
estimate of "18K input tokens" actually emits ~35.6K tokens (the
chunk-token-per-block heuristic at `bench_substrate_sweep.py:142` is
off by ~2×). Slate #4 cells at ctx=24K and ctx=32K therefore fail
with HTTP 400 ("request exceeds ctx"). Cells ≥49K succeed. Fix
deferred — the surviving cells still produce useful prefill-scaling
data, just at a different size than advertised.

**Open issues (don't promote until resolved):**

- **Verifier MISREPRESENTED false-positives** — claim verifier flags
  100% of writer's `[N]` citations as MISREPRESENTED (51/51 on a
  993-word draft). Was silently no-op'ing pre-Phase-55.S1; now runs
  reliably and dominates score. Slate #2 (quality probe) is parked
  on this. See memory `project_verifier_misrepresented_bug.md`.
- **Asymmetric K:V hang** — see above.

---



Two big-model roles drive every long-running flow on this stack:

| Role | GGUF | Current config | Used for |
|---|---|---|---|
| writer | Qwen3.6-27B-UD-Q4_K_XL (~17.6 GB) | ctx 24576, parallel 1, flash-attn on, **fp16 KV** | autowrite write/plan/revise, wiki compile, ask/write/review/revise/argue |
| scorer | Gemma-4-31B-it-Q4_1 (~18 GB) | ctx 24576, parallel 1, flash-attn on, **fp16 KV** | autowrite score/verify/CoVe/rescore (cross-family judge) |

The bench harness in `scripts/bench_substrate_sweep.py` measures one
flag at a time against a fixed synthetic workload (or, optionally, a
real autowrite section for quality probes). The discipline is **one
knob per single-mode run** — sweeping two at once makes deltas
unattributable. The `--matrix-knob` mode is explicit and produces
a full grid for the few cases where you need a 2-D map (typically
`cache-type × ctx-size`).

## Slate #2 redesign — once the writer-fabrication / weak-retrieval bugs are fixed

The original slate #2 (autowrite-section quality probe across 4 cache
types) was the binding gate for "does KV quant hurt quality?". When
run today (2026-04-28) it produced unusable data: all cells scored
groundedness=0.0 not because of cache-type effects but because the
writer fabricates citations regardless of cache type. The probe
measured pipeline brokenness, not substrate quality.

The verifier diagnostic (`scripts/debug_verifier.py`) confirmed:
chunk [1] for `the_science_of_sunspots` was a 52-char header
("Solar Influences\n\nON") with no body text. The writer cited it
34 times for specific sunspot claims. This is independent of cache
type; the bench cannot measure substrate quality through this lens
until the upstream fabrication is gone.

### Pre-conditions for re-running slate #2

Before the autowrite-section probe is meaningful, three fixes must
land (each is its own work item — see follow-up task list / memory
`project_verifier_misrepresented_bug.md`):

1. **Corpus audit + cleanup**: filter or re-ingest empty / header-
   only chunks. Min content length at retrieval time would be a
   one-line guard.
2. **Autowrite retrieval query**: replace `f"{section_type} {topic}"`
   (slug + chapter topic_query) with the section's actual title /
   description from `book_chapters.sections[].title|description`.
3. **Writer-prompt tightening**: explicit instruction to refuse
   fabrication. "If a claim cannot be supported by a specific chunk's
   text, omit it entirely. Do NOT cite a chunk that doesn't address
   the claim."

Validation gate: after fixes, debug_verifier.py on a fresh draft
should show groundedness_score > 0.7 (currently 0.0). Then the
autowrite-section bench probe is worth running.

### Slate #2-bis — controlled synthetic quality probe (alternative)

Even after fabrication is fixed, the autowrite-section probe is
NOISY — it measures the full retrieval+writer+score+verify pipeline,
and noise dominates the cache-type signal. A cleaner alternative for
"does KV quant cost quality?" is a controlled synthetic probe:

```
For each cache type ∈ {f16, q8_0, q4_0} at ctx=65536:
  For each known passage P (10-20 hand-curated):
    1. Feed writer the passage P + ask: "summarize the key claim and
       its supporting evidence in 200 words. Cite specific sentences
       from the passage with [Px:y] notation."
    2. Score: LLM-as-judge with a strict rubric (faithfulness, claim
       coverage, no-fabrication, citation accuracy) against the
       known-correct ground truth.
  Aggregate: mean score per cache type.
```

This bypasses retrieval quality entirely (each passage IS the
retrieval result), bypasses revision (it's a one-shot summarize),
and isolates the cache-type signal. Slow per-cell (~30s × 20 passages
× 3 cache types ≈ 30 min) but high signal-to-noise.

Hand-curated passage set: pull 10-20 chunks from the corpus that have
substantial body content, span multiple topics, and have unambiguous
key claims. The user (climate research domain) should pick or
validate these — bench harness can drive once the set is defined.

### When to run

After all three pre-condition fixes ship AND the synthetic probe set
is curated. Likely 1-2 weeks of work. Priority is lower than the
fabrication fixes themselves — those affect production book quality,
the bench just observes.

---

## Phase 55.V15/V16: quality-first, with two competing baselines

The bench has to settle a real disagreement between two
configurations that both have community advocates:

### Baseline A — `q8_0` KV (community consensus for RTX 3090 + Qwen3.6)

Documented in @aminrj's Qwen3.6 RTX 3090 bench, the post-PR-#19493
spec-decode harness, and ~every other 3090 + Qwen3.6 benchmark from
Q1 2026. Decode at ~135 t/s on 16K context, ~80 t/s at 65K context
on the writer-class 27B GGUF. Halves KV-cache footprint vs fp16
(~50% memory savings) with measured perplexity delta of +0.0043 on
Qwen2.5-Coder-7B at 128K — well within noise.

```
-ngl 99 -c 65536 -np 1 -fa on \
   --cache-type-k q8_0 --cache-type-v q8_0
```

### Baseline B — `q4_0` KV (the viral "262K on 24 GB" claim)

The power-user post: model 16 GB + q4_0 KV @ 262K = 5 GB = 21 GB
total at 40 t/s flat from 4K to 262K, with q8 alleged to be 3×
slower. **Multiple independent benchmarks contradict the speed
claim** (DGX Spark + Nemotron-30B shows q4_0 actually 18-35% SLOWER
than fp16 at long context, not faster; q8_0 not in the table). The
memory claim is correct; the speed-and-quality claims are the
hypotheses to test.

```
-ngl 99 -c 262144 -np 1 -fa on \
   --cache-type-k q4_0 --cache-type-v q4_0
```

### Baseline C — `q8_0:q4_0` mixed (GGML-discussion-#5932 sweet spot)

A user in GGML #5932 reports keeping K-cache at q8_0 (matters for
the GQA attention pattern Qwen-family models use heavily) while
quantizing V-cache to q4_0. Halves total KV memory vs full q8_0;
recovers the long-context recall q4_0 alone hurts. Not yet
mainstream but worth measuring.

```
-ngl 99 -c 131072 -np 1 -fa on \
   --cache-type-k q8_0 --cache-type-v q4_0
```

### Quality-first decision rule

**Book writing took less than a day** with the current config — we
do not have a speed problem. Quality is the binding constraint.
The decision rule for promoting any of these to production:

1. **Quality gate first** (sweep #2, autowrite-section). The
   candidate must score within **1% of fp16** on
   `final_score`. q4_0 has known long-context degradation on
   Qwen-family GQA (GGML #5932) — the bench has to prove it
   doesn't bite us at our 18K-token prompts.
2. **Speed gate second** (sweep #1). Any candidate that passes
   quality must also stay within **20% of the fastest decode
   TPS** to be worth promoting.
3. If `q8_0` and `q4_0` both pass, prefer `q8_0` — it's the
   community consensus and has the better measured-quality story.
4. If `q8_0` passes and `q4_0` fails quality, promote `q8_0`.
5. If only `q8_0:q4_0` (mixed) clears both gates, promote it.
6. **If everything fails**, stay on the current production config
   (writer ctx 24576, fp16 KV) and revisit when the substrate gets
   another upgrade pass.

`--baseline expert` (the harness default) currently sets every
non-swept parameter to baseline B (q4_0). Pass `--baseline current`
to keep production defaults instead. Future versions of this doc
may flip the harness default to `q8_0` once #1 + #2 settle.

## Methodology

### Workload

Three workload shapes — pick by what question you're answering:

| name | input | output | what it measures | when to use |
|---|---|---|---|---|
| `typical` | ~5 000 | 1 500 | autowrite write/revise call shape | default; verifies throughput under the production hot path |
| `large` | ~18 000 | 1 500 | ctx-size upper bound | when sweeping ctx-size or cache-type at large contexts |
| `autowrite-section` | (real) | (real) | scorer-judged final_score, groundedness, plan_coverage | quality probe — slow (~10 min/run), use `--reps 1` |

The synthetic workloads are **deterministic** (fixed seed corpus, no
randomness in the prompt). Only flag values vary across rows so
deltas are attributable.

The `autowrite-section` workload runs a real
`autowrite_section_stream` over a fixed seed section (default: first
section of the active book; override with
`--autowrite-section-slug`). It's the only workload that captures
**quality** — the scorer (gemma) judges the writer's output and
emits an overall + groundedness score. Use it to answer "does
this knob cost autowrite quality, not just throughput?" — the
synthetic workloads can't tell you that.

### Measurements per run

Per (role, knob, value) we record:

1. **decode TPS** — `timings.predicted_per_second` from the server.
   Headline number; how fast tokens come out once decoding starts.
2. **prompt-eval TPS** — `timings.prompt_per_second`. Only matters
   for the prefill phase; bottleneck on long-input workloads.
3. **peak VRAM** — `nvidia-smi memory.used` sampled after the run
   finishes. Indicator of cache-allocation pressure.
4. **cold load** — wall-clock from `up()` to `/health=200`. Indicator
   of model-init cost; only matters for the swap-heavy autowrite flow.
5. **wall** — total request time. Sanity check on the per-token rates.
6. **failure mode** — OOM / context-overflow / crash / OK. Critical
   for any value above the working configuration.

Each (knob, value) is run **three times** and the mean is reported
with min/max preserved (so noise is visible). All other roles are
torn down before each run for a clean GPU.

### Output

JSONL in `data/bench/substrate_sweep/<timestamp>-<role>-<knob>.jsonl`,
one row per (knob, value) — keeps every individual rep too. Re-runs
append. A Rich summary table prints to stdout.

## Suggested sweep slate

Highest-value-first. Verifying the expert claim is the new #1
because if it holds, every other sweep gets cheaper headroom to
work with. Total budget across the whole slate: **~8–10 GPU
hours** assuming reps=3 + the typical/large mix; the
autowrite-section quality probe adds ~1–2 hours per cache-type.

### 1. KV-cache quantisation — speed sweep across all four candidates

```
uv run python scripts/bench_substrate_sweep.py \
    --role writer --knob cache-type \
    --values f16,q8_0,q4_0,q8_0:q4_0 \
    --baseline expert --workload typical
```

**This sweep is the speed gate.** Every non-swept param starts at
the expert config (ctx 262144, np 1, fa on, ngl 99) so the only
variable is the KV-cache quant.

Expected outcomes per claim, based on the V16 web-research sweep:

- **fp16 KV** → expected to OOM at 262K on 24 GB (the only point
  where the expert claim "fp16 doesn't fit" is verifiable). If
  fp16 OOMs, that's the upper-context cliff.
- **q8_0 KV** → expected ~135 t/s short-ctx based on aminrj /
  thc1006 reproductions. The viral "3× slower" claim is contradicted
  by every other 3090 + Qwen benchmark; this row is where we
  empirically settle whether the claim holds on OUR GPU.
- **q4_0 KV** → expected ~21 GB peak. **Watch decode TPS at 262K**
  — Nemotron-30B benchmarks show q4_0 is actually SLOWER than fp16
  at long context (the opposite of the viral claim), not faster.
- **q8_0:q4_0 mixed KV** → ~75% of q8_0's memory cost, ~25% larger
  than full q4_0; expected throughput close to q8_0 (key reads
  dominate decode).

### 2. Quality probe — quality is the binding constraint

```
uv run python scripts/bench_substrate_sweep.py \
    --role writer --knob cache-type \
    --values f16,q8_0,q4_0,q8_0:q4_0 \
    --workload autowrite-section --reps 1 \
    --autowrite-section-slug the_engine_of_the_sun
```

**Book writing took less than a day on the current config — we do
not have a speed problem.** This sweep is what decides whether any
candidate ships.

The autowrite-section workload runs one full
`autowrite_section_stream` per cache-type, captures the scorer's
overall + groundedness + plan_coverage. Slow (~10 min × 4 cache
types = ~40 min wall) but the only way to catch quality
regressions before deploying.

Expected outcomes (from the V16 research):

- **fp16 KV** → reference quality. The bar.
- **q8_0 KV** → within ~1% of fp16. GGML #5932 measured +0.0043
  perplexity at 128K on Qwen2.5-Coder; this is the safe choice.
- **q4_0 KV** → **GGML #5932 explicitly flagged Qwen2's 8x GQA as
  hurt-more-than-average by aggressive KV quant.** Qwen3.6 inherits
  the GQA pattern. Likely loses ≥2% groundedness on our 5-18K
  prompts; long-context degradation may push that higher.
- **q8_0:q4_0 mixed** → preserves K-cache (the side that matters
  for GQA's attention-pattern reads), halves V-cache. Expected
  within 0.5% of pure q8_0 quality at ~25% less memory.

**Ship rules** (in order; the first that fires wins):

1. If `q4_0` final_score drops > 0.01 vs fp16, **reject q4_0**
   immediately — quality is the binding constraint.
2. If `q8_0` final_score drops > 0.01 vs fp16, **reject q8_0** too
   and stay on production defaults.
3. If `q8_0:q4_0` mixed survives both gates AND beats `q8_0` on
   speed by ≥10%, promote it.
4. Otherwise promote `q8_0` (community consensus + clean quality).
5. If everything fails the quality gate, stay on the current
   production config (writer ctx 24576, fp16 KV).

### 3. cache-type × ctx-size matrix — locate the speed cliffs

```
uv run python scripts/bench_substrate_sweep.py \
    --role writer \
    --matrix-knob cache-type --matrix-values f16,q8_0,q4_0,q8_0:q4_0 \
    --knob ctx-size \
    --values 16384,24576,65536,131072,262144 \
    --baseline expert --workload typical
```

4 × 5 = 20-cell grid. Answers:
- "Where does each cache type OOM on the 3090?"
- "Is decode TPS flat across context (the headline q4_0 claim) or
  does it actually drop at higher ctx (Nemotron benchmark says
  yes)?"
- "If we want to bump production ctx, what's the safe ceiling per
  cache type?"
- "Does the asymmetric `q8_0:q4_0` mixed mode beat full q8_0 at
  long context?"

### 4. Writer ctx-size at q4_0 KV — pick the production default

After 1+2 confirm q4_0 is safe, sweep ctx_size on q4_0 to find the
production sweet spot. (If q4_0 doesn't pan out, sweep at fp16
instead — the values list shifts to 16K–32K because larger won't
fit.)

```
uv run python scripts/bench_substrate_sweep.py \
    --role writer --knob ctx-size \
    --values 24576,32768,49152,65536,98304,131072,196608,262144 \
    --baseline expert --workload large
```

Pair with `--workload large` (18K input) to stress the prefill
phase — `typical` (5K) won't expose prefill scaling.

### 5. Scorer ctx-size — same question for gemma

```
uv run python scripts/bench_substrate_sweep.py \
    --role scorer --knob ctx-size \
    --values 16384,32768,65536,131072,262144 \
    --baseline expert
```

Gemma 4 31B is ~18 GB on disk vs Qwen3.6's 17.6 GB; tighter VRAM
margin so the 262K row likely fails with "compute_pp buffer alloc
failed" even with q4_0 KV. That's the diagnostic.

### 6. Batch / ubatch (writer prefill speed)

```
uv run python scripts/bench_substrate_sweep.py \
    --role writer --knob batch-size \
    --values 1024,2048,4096,8192
```

(`--ubatch-size` is auto-set to `batch-size / 4` by the harness.)

Expected outcome:

- prompt-eval TPS climbs with larger batch on long-prompt workloads.
- Plateau around 4096–8192.
- Larger batch = more compute buffer = ~200 MB extra peak VRAM.

Run with `--workload large` to make the prefill phase load-bearing
(`typical` only has ~5K prefill tokens, the prefill is fast either way).

### 7. n-gpu-layers (CPU-offload trade)

Only useful if a sweep above hits OOM at the working ctx and we want
to know whether 1–2 layers on CPU buys back enough headroom to
recover. Default is `999` (all on GPU). Sweep:

```
uv run python scripts/bench_substrate_sweep.py \
    --role writer --knob n-gpu-layers \
    --values 999,60,55,50
```

Expected outcome: decode TPS drops sharply (~3–5× per layer offloaded
on a 3090). Only makes sense as an emergency lever — keep at 999 for
production.

### 8. flash-attn (sanity check, control)

```
uv run python scripts/bench_substrate_sweep.py \
    --role writer --knob flash-attn \
    --values on,off
```

Expected outcome: `on` is 10–25% faster decode at long context, fully
established; this is just a control to confirm the flag is taking
effect. Default `on`; this run is optional.

### 9. parallel (writer slot count)

```
uv run python scripts/bench_substrate_sweep.py \
    --role writer --knob parallel \
    --values 1,2
```

Expected outcome: parallel=1 wins for autowrite (single-stream by
design). parallel=2 splits the ctx in half per slot, halving effective
context. Run as a control to verify.

## Knobs explicitly NOT in the slate

- **Concurrent multi-role**: Phase 55.V1 already evicts peers; that's
  not tunable, it's a discipline. Disabling the eviction (set
  `VRAM_CO_RESIDENCE_OK=true` in `.env`) on the 3090 will OOM and is
  not a defensible config — don't bench it.
- **Speculative decoding** (`--model-draft`): **proven net-negative
  on Ampere + Qwen3.6 MoE.** Post-PR-#19493 19-config bench
  ([thc1006 reproduction](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090),
  [HackMD writeup](https://hackmd.io/ODXuOQNzSiyUITz7g9mtBw)) shows
  every spec-decode mode (ngram-cache, ngram-mod, classic
  vocab-matched draft) loses 3–12% mean decode despite 100% draft
  acceptance on some prompts. Cause: A3B MoE expert-saturation
  threshold T_thres ≈ 94 is well above any realistic draft K. Each
  drafted token pulls a fresh expert slice through the memory
  hierarchy and the verification pass pays for the union. **Defer
  to DGX Spark or skip entirely.**
- **Engine knobs** (temperature, cove_threshold, retrieval `context_k`):
  these affect autowrite *quality*, not substrate throughput. They
  belong in a separate quality bench, not this throughput sweep.

## Known external pitfalls (not bench problems, but check before running)

- **CUDA 13.2 produces gibberish** with Qwen3.6
  ([aminrj observation](https://aminrj.com/posts/llamacpp-qwen36-35b/)).
  We're on driver 13.1 + nvcc/runtime 12.0 — verified safe before
  starting this bench. If you upgrade the NVIDIA driver to 13.2 mid
  bench, results invalidate.
- **q4_0 KV long-context degradation is real, despite the viral
  claim of "flat 40 t/s curve."** DGX Spark + Nemotron-30B 128K
  ([NVIDIA dev forum benchmark](https://forums.developer.nvidia.com/t/kv-cache-quantization-benchmarks-on-dgx-spark-q4-0-vs-q8-0-vs-f16-llama-cpp-nemotron-30b-128k-context/365138))
  shows q4_0 decode TPS drops 18–35% vs fp16 as context grows past
  32K. Don't take the speed claim at face value; sweep and measure.
- **Qwen-family GQA is q4_0-sensitive.** GGML #5932 explicitly
  flags Qwen2's 8x GQA as "hurts more than less aggressive GQA"
  with aggressive KV quant. Qwen3.6 inherits this pattern; if our
  quality-gate sweep #2 fails for q4_0, this is why.
- **Newer llama.cpp requires explicit `--flash-attn on/off`**, not
  just `--flash-attn` as a flag. Our role configs already use the
  explicit form — sanity-check after any llama.cpp build update.

## TurboQuant 3-bit KV (future)

[TurboQuant](https://github.com/ggml-org/llama.cpp/discussions/20969)
is a Google Research scheme that hits ~3-bit KV with 0-class
perplexity loss (TQ3 PPL = 6.20 vs f16 baseline 6.19; TQ4 PPL =
6.76 vs 6.89). Would unlock 5–6× compression vs q8_0 with cleaner
quality than q4_0. **Not yet in mainline llama.cpp** as of
2026-04-28; Metal works in community forks, CUDA support is
experimental, Vulkan in progress. Worth a follow-up bench once it
lands in mainline, but out of scope for this iteration.

## How to interpret the table

The harness prints a Rich table at the end of each sweep. Read it as:

| value | decode TPS | prompt-eval TPS | peak VRAM | cold load | wall | status |
|---|---|---|---|---|---|---|
| (the swept value) | tokens/sec out | tokens/sec in | GPU mem after | spawn → ready | total request time | ok / failure |

Picking a winner:

1. **Discard rows with `failure`** — non-starter at any value.
2. **Find the rows tied at the highest decode TPS** (within ~1 t/s of
   the max — that's noise floor).
3. **Among those, prefer the highest VRAM headroom** (lowest peak
   VRAM) — buys you margin for KV growth on long iterations.
4. **Tie-break on prompt-eval TPS** if the workload involves long
   prefills (`large` workload).

The current production defaults (writer ctx 24576, paired f16 KV, batch
8192/2048, flash-attn on) were picked by spec, not measurement; the
sweeps here either confirm them or surface a better point.

## Re-running after a fix

If you change `infer/server.py:ROLE_DEFAULTS` in production based on
a sweep finding, ALSO re-run `--knob ctx-size --workload large` to
verify the new default doesn't OOM under the worst-case prompt the
fix was meant to handle.

## Wall-clock budget

| Workload | Per-cell cost | Notes |
|---|---|---|
| `typical` (5K input, reps=3) | ~3 min | cold-load + 3× decode + tear-down |
| `large` (18K input, reps=3) | ~5 min | longer prefill |
| `autowrite-section` (reps=1) | ~10 min | full retrieve→write→score→verify→cove→revise→rescore loop |

**Recommended slate budget** (run pieces independently as GPU time allows):

| Slate item | Cells | Budget |
|---|---|---|
| #1 cache-type verify (typical) | 3 | ~10 min |
| #2 cache-type quality probe | 3 (reps=1) | ~30 min |
| #3 cache-type × ctx-size matrix | 15 | ~45 min |
| #4 ctx-size at q4_0 (large) | 8 | ~40 min |
| #5 scorer ctx-size | 5 | ~15 min |
| #6 batch-size (large) | 4 | ~20 min |
| #7–9 control sweeps | ~6 | ~20 min |
| **Total** | ~44 cells | **~3 hours** |

The `autowrite-section` quality probe (slate #2) is the only piece
that takes >10 min/cell. The rest is fast — the full 44-cell slate
fits in one half-day of GPU time.

Add reps to taste — `--reps 5` doubles the synthetic-workload time
budget, no change to autowrite-section.
