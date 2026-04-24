# Luce-Org/lucebox-hub — research memo

**URL**: <https://github.com/Luce-Org/lucebox-hub>
**Shared by user**: 2026-04-24
**Clone scratch**: `data/research/Luce-Org__lucebox-hub/`
**Stack**: Python + C++17 + CUDA 12+

## One-line summary

Open LLM inference rewritten by hand for one chip at a time. Two
sub-projects today:

- **`megakernel/`** — Qwen3.5-0.8B on RTX 3090 in a single CUDA
  dispatch: 413 tok/s (1.55× llama.cpp BF16, 3.8× PyTorch HF, 1.87
  tok/J at 220 W).
- **`dflash/`** — **the first GGUF port of DFlash speculative
  decoding**. Qwen3.5-27B at up to **207.6 tok/s** (peak demo),
  **129.5 tok/s mean on HumanEval** on a single **RTX 3090** with Q4_K_M
  weights. 3.43× over autoregressive; runs 128K context on 24 GB via
  TQ3_0 KV cache.

Both projects target **single-stream, batch-size-1 local inference on
consumer hardware** — i.e. exactly the sciknow operating profile.

## Why this repo matters to sciknow

The user shared this repo alongside task 2 (ik_llama.cpp + Qwen3-1.7B
draft model → 154 tok/s on a 4090). Lucebox's DFlash port is a
**stronger** result on **our hardware class** (RTX 3090) with the
**same model family** we already use (Qwen 3.5-27B / Q4_K_M). The
headline numbers from `dflash/RESULTS.md`:

| Task      | AR tok/s | DFlash tok/s | Speedup |
|-----------|---------:|-------------:|--------:|
| HumanEval | 37.78    | **129.52**   | 3.43×   |
| Math500   | 37.71    | **110.51**   | 2.93×   |
| GSM8K     | 37.65    |  96.15       | 2.55×   |

Baseline AR matches what sciknow sees on the same 3090 (~37-43 tok/s
on Qwen3.6-27B Q4_K_M via Ollama). A 3× uplift on real-world
generation (not toy benchmarks) is a genuine sciknow-wide speed-up
target.

## Architecture — `dflash/`

- ~2000 LOC of C++/CUDA on top of `ggml` (no libllama, no Python
  runtime at inference time).
- Pinned fork of llama.cpp at `Luce-Org/llama.cpp@luce-dflash` adds
  three tree-mode ggml ops: `ggml_ssm_conv_tree`,
  `ggml_gated_delta_net_tree`,
  `ggml_gated_delta_net_tree_persist`.
- Hardcoded for one model pair:
  - Target: `unsloth/Qwen3.5-27B-GGUF` (Q4_K_M, ~16 GB)
  - Draft:  `z-lab/Qwen3.5-27B-DFlash` (BF16, 3.46 GB)
- Two-stage speculative decoding: block-diffusion draft proposes
  multiple tokens per step; DDTree (budget 22) tree-structured verify
  recovers the last 30 % of speedup.
- 128K context on 24 GB via `DFLASH27B_KV_Q4=1` (Q4_0 K+V cache, 8×
  compression vs F16) + sliding `target_feat` ring (4096 slots).
- Source inventory (`dflash/src/`):
  `delta_net_chunked.cpp`, `dflash_graph.h`, `f16_convert.cu`,
  `gguf_target_loader.cpp`, `kv_cache.cpp`,
  `qwen35_target_graph.cpp`, `qwen3_dflash_graph.cpp`,
  `safetensors_draft.cpp`.

## Architecture — `megakernel/`

- All 24 layers of Qwen3.5-0.8B in a **single CUDA dispatch**.
- 37,800 tok/s prefill, 413 tok/s decode on RTX 3090 (vs 11,247 /
  267 for llama.cpp BF16, 7,578 / 108 for PyTorch HF).
- Power-efficiency sweep: 1.87 tok/J at 220 W / 1635 MHz (30 % less
  power, 95 % speed vs stock 420 W). Matches Apple M5 Max silicon at
  2× throughput.
- Single source file per concern: `kernel.cu`, `prefill.cu`,
  `model.py`, `bench.py`, `final_bench.py`, `setup.py`,
  `torch_bindings.cpp`.
- **NOT directly relevant** — Qwen 3.5-0.8B is too small for
  sciknow's writer / scorer / reviewer roles (we run 27B / 32B
  models). This is infrastructure research, not a drop-in.

## What sciknow could port / adopt

1. **DFlash Qwen3.5-27B on 3090 is a direct speedup target.** If we
   add a new LLM backend alongside Ollama — specifically the pinned
   `luce-dflash` llama.cpp fork + GGUF target + BF16 draft — every
   sciknow role that uses the 27B writer (book-write, autowrite
   scorer, revise, critique) would get a 2.5-3.4× uplift. At current
   baseline, a 1000-word chapter takes ~26 s to generate; DFlash
   brings that to ~8 s. Autowrite loops dominated by draft-write time
   (3-5 min per iteration) drop to ~1-2 min.

2. **KV-cache quantisation pattern (`DFLASH27B_KV_Q4=1`).** Lucebox
   demonstrates **Q4_0 KV cache is the unlock** for 128K context on
   24 GB — 8× compression vs F16 at only ~3 % tok/s cost on short
   contexts. Our Ollama setup already has
   `OLLAMA_KV_CACHE_TYPE=q4_0` since 54.6.303, which is the same
   idea; Lucebox's RESULTS.md confirms empirically that this trade is
   correct on 3090-class hardware.

3. **Budget-sweep methodology for hyperparameters.** `dflash/RESULTS.md`
   has a rigorous DDTree-budget sweep (15 → 40) showing the AL /
   tok-s / memory Pareto frontier, chosen empirically as budget = 22.
   Our autowrite loop has analogous knobs
   (`AUTOWRITE_MAX_ITERATIONS`, `AUTOWRITE_TARGET_SCORE`, visual
   ranker `candidate_k`) that we should sweep + document the same
   way in `docs/BENCHMARKS.md`.

4. **Power-efficiency DVFS sweep.** The `tok/J` metric at different
   power caps is a framing we don't currently use. For a local-first
   system like sciknow running overnight jobs (book-write, autowrite-
   all, db expand), a 30 %-less-power-for-5 %-less-speed sweet spot
   is extremely relevant for users on home power.

5. **Block-diffusion draft (DFlash) vs chain draft (Qwen3-1.7B).** The
   user's shared snippet from task 2 proposes chain spec-dec with
   Qwen3-1.7B at 85 % acceptance for 154 tok/s. Lucebox's DFlash
   block-diffusion draft achieves ~8 tokens / step acceptance length
   (AL 8.31 on HumanEval) and 129-207 tok/s on a 3090. DFlash is
   +15 % over chain spec-dec per the README. **Stronger technique;
   specifically validated on Qwen3-27B-family weights on our GPU.**

## What NOT to port

- Megakernel for 0.8B models — wrong model size for our use case.
- The custom llama.cpp fork — it's hardcoded for one model pair.
  Adopting means vendoring the fork, not a clean dependency.
- The BF16 draft (3.46 GB) would increase our VRAM pressure. Our
  current Ollama setup keeps writer + embedder + reranker co-resident
  on 24 GB with Q4_K_M; adding a BF16 draft plus the verify tree
  state means we'd need to evict the reranker during writer turns.

## VRAM budget sanity check on our 3090 (24 GB)

Current resident (approx):
- Qwen3.6-27B Q4_K_M (writer)          ~16 GB
- bge-m3 embedder                      ~1 GB
- bge-reranker-v2-m3                   ~1 GB
- KV cache (q4_0, ctx 32K)             ~1 GB
- Headroom / other                     ~5 GB

DFlash swap would replace the writer block with:
- Qwen3.5-27B Q4_K_M target            ~16 GB
- z-lab Qwen3.5-27B-DFlash draft BF16  ~3.5 GB
- DDTree budget-22 verify state        ~1.3 GB
- Q4_0 KV cache (8K ctx)               ~0.25 GB
- target_feat ring (4096)              ~0.2 GB
- **total DFlash-side:**               **~21 GB**

Would force evicting the reranker during writer turns — manageable
via the existing VRAM-ledger pattern (see `sciknow/core/vram_budget.py`
+ `sciknow/core/gpu_ledger.py` which already do hot-swap bookkeeping).
Acceptable cost for the 3× throughput.

## Relevance verdict

- [x] **high — implement / port** (for the `dflash/` sub-project).
- [ ] Megakernel is informational-only.

## Next actions

- [ ] `docs/research/speculative_decoding.md` (task 2 in this /loop
      session) synthesises ik_llama.cpp + DFlash into a single
      comparison memo so the user can pick between them.
- [ ] Dry-run: clone `Luce-Org/llama.cpp@luce-dflash`, build the
      tree-mode kernels, run `scripts/bench_llm.py` on the 3090 to
      confirm the 129 tok/s number reproduces before committing to
      the integration.
- [ ] If numbers reproduce, design a `LLM_BACKEND=ollama | dflash`
      toggle in `sciknow/config.py` with the VRAM-ledger rewiring
      spelled out.
- [ ] Watch for updates: `sciknow watch check Luce-Org/lucebox-hub`
      — DFlash is a new project (2026-04-16 commit date), likely to
      evolve rapidly.
