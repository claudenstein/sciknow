# Substrate optimization benchmark plan (Phase 55.V10)

Two big-model roles drive every long-running flow on this stack:

| Role | GGUF | Default config | Used for |
|---|---|---|---|
| writer | Qwen3.6-27B-UD-Q4_K_XL (~17.6 GB) | ctx 24576, parallel 1, flash-attn on | autowrite write/plan/revise, wiki compile, ask/write/review/revise/argue |
| scorer | Gemma-4-31B-it-Q4_1 (~18 GB) | ctx 24576, parallel 1, flash-attn on | autowrite score/verify/CoVe/rescore (cross-family judge) |

The bench harness in `scripts/bench_substrate_sweep.py` measures one
flag at a time against a fixed synthetic workload. The discipline is
**one knob per run** — sweeping two at once makes deltas
unattributable. Re-run with a different `--knob` to compare results.

## Methodology

### Workload

Two synthetic prompt shapes:

| name | input tokens | output tokens | rationale |
|---|---|---|---|
| `typical` | ~5 000 | 1 500 | mirrors the autowrite write/revise call (12 retrieved chunks + leitmotiv + prior summaries → ~5K prompt; 1500-word draft) |
| `large` | ~18 000 | 1 500 | pushes the ctx-size upper bound; relevant for the recently-fixed planning-prompt-overflow class of failures |

The workload is **deterministic** (fixed seed corpus, no randomness in
the prompt itself). Only flag values vary across rows so deltas are
attributable.

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

The order below is highest-value-first. Run pieces independently as
GPU time allows. Total budget across the whole slate: **~6–8 GPU
hours** assuming default reps=3 + workload=typical.

### 1. Writer ctx-size — answers the immediate question

```
uv run python scripts/bench_substrate_sweep.py \
    --role writer --knob ctx-size \
    --values 16384,20480,24576,28672,32768
```

Expected outcome:

- decode TPS roughly flat across all values (Qwen3.6 attention scales
  linearly until the model hits a bandwidth wall).
- prompt-eval TPS drops at higher ctx (more prefill compute).
- peak VRAM grows linearly: each +4K of context costs ~250 MB of KV.
- 32K should cross 21 GB peak — within the 24 GB 3090 ceiling but
  thin if the embedder/reranker linger (Phase 55.V1 evicts them
  during generate, so this is fine).

Workload to pair: start with `typical`. Add a second pass with
`--workload large` once you've confirmed the typical sweep is clean.
The `large` shape stresses the upper end and is where the V6 fix
(default ctx-size 16384 → 24576) was driven from.

### 2. Scorer ctx-size — same question for gemma

```
uv run python scripts/bench_substrate_sweep.py \
    --role scorer --knob ctx-size \
    --values 16384,20480,24576,28672
```

(Gemma 4 31B is ~18 GB on disk vs Qwen3.6's 17.6 GB; tighter VRAM
margin so 32K may not fit on a 3090.)

Expected outcome: similar shape to the writer, but the 32K row likely
fails with "compute_pp buffer alloc failed" — that's the diagnostic.

### 3. KV-cache quantisation (paired K+V)

```
uv run python scripts/bench_substrate_sweep.py \
    --role writer --knob cache-type \
    --values f16,q8_0,q4_0
```

Same paired-quant pattern referenced in `scripts/ollama-override.conf`
(the @Punch_Taylor 4090 4096-context bench used q4_0). Expected
outcome:

- f16 → baseline.
- q8_0 → ~50% KV-cache size, modest decode TPS hit (~5–10%).
- q4_0 → ~25% KV-cache size, larger decode TPS hit (~10–15%) and
  small quality risk on long contexts (worth checking against
  autowrite scores in a follow-up integration test).

Pair with `--workload large` to see the VRAM unlock — this is where
q4_0 lets you push ctx to 32K on a 3090 without OOM.

### 4. Batch / ubatch (writer prefill speed)

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

### 5. n-gpu-layers (CPU-offload trade)

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

### 6. flash-attn (sanity check, control)

```
uv run python scripts/bench_substrate_sweep.py \
    --role writer --knob flash-attn \
    --values on,off
```

Expected outcome: `on` is 10–25% faster decode at long context, fully
established; this is just a control to confirm the flag is taking
effect. Default `on`; this run is optional.

### 7. parallel (writer slot count)

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

Per (knob, value) run with `--reps 3 --workload typical`:

- ~10–15 s cold-load
- 3 × (~50 s wall) = ~150 s decoding
- 3 × ~3 s tear-down

Total ≈ 3 minutes per value. A 5-value sweep = ~15 minutes. The full
slate (sweeps 1–4 above, ~5–6 values each) = ~1.5–2 hours.

`--workload large` adds ~30–60 s per rep (longer prefill); same shape
for the slate.

Add reps to taste — `--reps 5` adds variance shrinkage at ~2× wall
budget.
