# Substrate optimization benchmark plan (Phase 55.V10/V15)

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

## Phase 55.V15: start from the expert baseline

Community-recommended Qwen3.6-27B 24-GB-3090 config (matches what
power users post and the @Punch_Taylor 4090 bench config we cited
in the Phase 54.6.303 commit):

```
-ngl 99 -c 262144 -np 1 -fa on \
   --cache-type-k q4_0 --cache-type-v q4_0
```

Memory claim: model = 16 GB (Q4_K_M); KV at 262K with q4_0 K/V = 5
GB; total = 21 GB on a 24 GB card; headroom = ~3 GB. Throughput
claim: 40 t/s flat curve from 4K to 262K. The bench's job is to
**verify or falsify** these claims on this hardware (RTX 3090 24
GB) with our slightly larger Q4_K_XL writer GGUF.

`--baseline expert` (the harness default) sets every non-swept
parameter to the expert config so sweeps test against a known-good
starting point rather than fresh-from-defaults. `--baseline current`
keeps whatever's in `ROLE_DEFAULTS` for "would this delta regress
my production config?" runs.

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

### 1. KV-cache quantisation — VERIFY the expert claim (priority #1)

```
uv run python scripts/bench_substrate_sweep.py \
    --role writer --knob cache-type \
    --values f16,q8_0,q4_0 \
    --baseline expert --workload typical
```

**This sweep is the gate.** Every non-swept param starts at the
expert config (ctx 262144, np 1, fa on, ngl 99) so the only
variable is the KV-cache quant.

Expected outcomes per claim:

- **fp16 KV** → expected to OOM at 262K (claim: "does not fit"). If
  it does fit, the claim is partly wrong and we have more headroom
  than expected.
- **q8_0 KV** → expected ~23 GB peak. **Speed claim: 3× slower than
  fp16/q4_0.** If decode TPS lands at ~12–15 t/s vs q4_0's 40 t/s,
  the claim is verified. If it's only 5–10% slower (the typical
  llama.cpp benchmark), the viral post is wrong about the speed
  cliff.
- **q4_0 KV** → expected ~21 GB peak at 40 t/s flat. The claim's
  load-bearing case.

Ship-decision rule: if q4_0 wins decode TPS AND survives the
autowrite-section quality probe (sweep #2 below) within ~2% of fp16
groundedness, switch the writer + scorer defaults to q4_0. Otherwise
stay at the smaller-ctx fp16 config.

### 2. Quality probe — does q4_0 KV cost autowrite groundedness?

```
uv run python scripts/bench_substrate_sweep.py \
    --role writer --knob cache-type \
    --values f16,q8_0,q4_0 \
    --workload autowrite-section --reps 1 \
    --autowrite-section-slug the_engine_of_the_sun
```

The autowrite-section workload runs one full
`autowrite_section_stream` per cache-type, captures the scorer's
overall + groundedness + plan_coverage. Slow (~10 min × 3 cache
types = ~30 min wall) but it's the only way to detect a quality
regression before deploying.

Expected outcome:

- **fp16 KV** → reference quality.
- **q8_0 KV** → within ~1% of fp16 (published llama.cpp benchmarks).
- **q4_0 KV** → within ~2–3% on short contexts; degradation
  growing on long contexts (>64K). Our autowrite prompts are
  5–18K tokens, so the degradation should be small.

If `q4_0` final_score drops > 0.05 vs fp16, that's a measurable
quality regression and we should NOT promote q4_0 to default.

### 3. cache-type × ctx-size matrix — locate the speed cliffs

```
uv run python scripts/bench_substrate_sweep.py \
    --role writer \
    --matrix-knob cache-type --matrix-values f16,q8_0,q4_0 \
    --knob ctx-size \
    --values 16384,24576,65536,131072,262144 \
    --baseline expert --workload typical
```

3 × 5 = 15-cell grid. Answers:
- "Where does each cache type OOM on the 3090?"
- "Is decode TPS flat across context (the headline q4_0 claim) or
  does it actually drop at higher ctx?"
- "If we want to bump production ctx, what's the safe ceiling per
  cache type?"

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
- **Speculative decoding** (`--draft`): would need a smaller draft
  model also loaded; the 3090 doesn't have headroom. Defer to DGX
  Spark.
- **Engine knobs** (temperature, cove_threshold, retrieval `context_k`):
  these affect autowrite *quality*, not substrate throughput. They
  belong in a separate quality bench, not this throughput sweep.

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
