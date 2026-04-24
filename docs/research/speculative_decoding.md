# Speculative decoding for sciknow — research memo

**Generated**: 2026-04-24.
**Context**: User shared a tweet about `ik_llama.cpp + Qwen3-1.7B`
draft for `Qwen3.6-27B` delivering 26 → 154 tok/s on a 4090 (85 %
acceptance rate). Asked whether this can be applied to sciknow.
Separately, the user also shared `Luce-Org/lucebox-hub` which
happens to contain a purpose-built spec-dec implementation
(`dflash/`) for Qwen3.5-27B on **RTX 3090** hardware.

**Decision needed**: which spec-dec path to adopt, and what the port
surface looks like against our current Ollama-only backend.

## TL;DR recommendation

**Prototype DFlash (Lucebox) first.** Strongest numbers on our actual
hardware class (3090), same model family we already use, rigorous
reproducibility, and ships with benchmarks and DDTree budget sweeps.
Fallback to plain `llama.cpp --draft-max` + Qwen3-1.7B chain draft
if DFlash doesn't build cleanly or if the `luce-dflash` ggml kernels
don't support our exact quantisation. `ik_llama.cpp` is a third
option useful mainly if DFlash blocks — it's more general but lacks
the tree-verify uplift.

## The three options

| Option | Target | Draft | Peak tok/s (3090 Q4_K_M) | Complexity | Status |
|---|---|---|---:|---|---|
| **A. DFlash (Lucebox)** | Qwen3.5-27B Q4_K_M | z-lab Qwen3.5-27B-DFlash (BF16, 3.46 GB) | **207 peak / 129.5 HumanEval mean** | Vendored llama.cpp fork (3 new ggml ops) + 2000 LOC C++/CUDA | Purpose-built for 3090 |
| **B. ik_llama.cpp + chain draft** | Qwen3.6-27B Q4_K_M | Qwen3-1.7B Q4_K_M | ~154 (4090 reported; est. ~100-130 on 3090) | Swap backend, pass `-md draft.gguf --draft-max 12 --draft-min 3 --draft-p-min 0.6` | Upstream, maintained |
| **C. llama.cpp-server + chain draft** | Same as B | Same as B | ~100-120 est. on 3090 | Swap backend, standard `llama-server` flags | Upstream, widely used |
| D. Ollama (current)     | Qwen3.6-27B Q4_K_M | — | ~37-43 | Current | Ollama doesn't expose draft flags |

(Option D is the baseline; Ollama deliberately hides the draft-model
plumbing so this is why "nobody's seeing this" per the tweet.)

## Why DFlash wins on paper

From `docs/research/Luce-Org__lucebox-hub.md` + `dflash/RESULTS.md`:

- **Block-diffusion draft** (non-causal 5-layer denoising) predicts
  8-10 tokens per step vs ~3 for chain spec-dec, so the AR baseline
  38 tok/s climbs to 129 mean / 207 peak.
- **DDTree tree verify (budget 22)** adds +15 % over chain spec-dec
  by verifying multiple candidate continuations in a single forward
  pass.
- **Q4_0 KV cache + sliding target_feat ring** unlocks 128K context
  on a 24 GB GPU.
- The benchmark is on the exact GPU we run on (RTX 3090 24 GB,
  CUDA 12, driver 535), with Q4_K_M weights we already use.
- **Lossless** — speculative decoding is mathematically identical to
  the target model's AR output (modulo numerical noise).

## Why ik_llama.cpp is the safer fallback

The user's tweet cites ik_llama.cpp specifically, and it's a fork of
mainline llama.cpp with additional kernel optimisations. The specific
command:

```bash
llama-server \
  -m Qwen3.6-27B-Q4_K_M.gguf \
  -md Qwen3-1.7B-Q4_K_M.gguf \
  -ngl 99 -ngld 99 -c 8192 -cd 4096 \
  -fa on -ctk q8_0 -ctv q8_0 \
  --draft-max 12 --draft-min 3 --draft-p-min 0.6
```

Key flags:
- `-md` — draft model path.
- `-ngl 99 -ngld 99` — offload all layers of both target and draft.
- `-c 8192 -cd 4096` — target context 8K, draft context 4K.
- `-fa on` — flash attention (we already use this — 54.6.303 fixed
  the `OLLAMA_FLASH_ATTENTION=1` typo).
- `-ctk q8_0 -ctv q8_0` — q8 KV cache (the 4090 has VRAM headroom
  for q8; on our 3090 we'd want q4_0 per DFlash's sweep).
- `--draft-max 12 --draft-min 3 --draft-p-min 0.6` — the spec-dec
  knobs: propose up to 12 tokens, minimum 3, minimum draft
  probability 0.6 to accept.

Caveats on a 3090 vs a 4090:
- 4090 has ~50 % more memory bandwidth (1008 GB/s vs 936 GB/s) and
  23 % more TFLOPS. The 26 → 154 tok/s result is a 5.9× uplift; on a
  3090 we can expect roughly 80-85 % of that throughput, so ~100-130
  tok/s realistic. Still 2.5-3× over Ollama baseline.
- VRAM budget: 27B Q4_K_M (~16 GB) + 1.7B Q4_K_M (~1.2 GB) + q8 KV
  (~1 GB for 8K) + headroom = ~18 GB. Fits on our 24 GB with the
  embedder + reranker co-resident.

## VRAM budget comparison on our 3090 (24 GB)

| Component | Ollama now | ik_llama.cpp+draft | DFlash |
|---|---:|---:|---:|
| Target 27B Q4_K_M | 16 GB | 16 GB | 16 GB |
| Draft | — | 1.2 GB (1.7B Q4) | 3.5 GB (BF16 DFlash) |
| KV cache (8K) | 1 GB (q4_0) | 1 GB (q8_0) | 0.25 GB (q4_0) |
| Verify tree / target_feat | — | — | 1.5 GB |
| bge-m3 embedder | 1 GB | 1 GB | 1 GB |
| bge-reranker-v2-m3 | 1 GB | 1 GB | evicted during writer turns |
| Headroom | ~5 GB | ~3.8 GB | ~1.75 GB |
| **Total footprint** | **~19 GB** | **~20 GB** | **~22 GB** |

DFlash is **the tightest fit on 24 GB**. If we want to run concurrent
embedder + reranker during writer turns, ik_llama.cpp chain-draft is
easier to squeeze. The tradeoff is ~30-40 tok/s of headroom peak
throughput.

## Required sciknow changes for ANY spec-dec path

Both DFlash and ik_llama.cpp require the same sciknow-side plumbing:

1. **Second LLM backend.** `sciknow/config.py::Settings` needs a new
   `llm_backend: Literal["ollama", "llama_cpp_server", "dflash"]`
   setting. Default stays `ollama` so the switch is opt-in.
2. **HTTP adapter.** `sciknow/rag/llm.py` currently speaks Ollama's
   `/api/chat` stream format. Both alternatives expose the
   OpenAI-compatible `/v1/chat/completions` endpoint — adapter code
   is ~40 LOC.
3. **Model-role mapping.** Our per-role model table
   (`LLM_MODEL`, `LLM_FAST_MODEL`, `BOOK_WRITE_MODEL`,
   `AUTOWRITE_SCORER_MODEL`, `EXTRACT_KG_MODEL`,
   `VISUALS_CAPTION_MODEL`, `MINERU_VLM_MODEL`) stays unchanged —
   the backend layer just points a given role at the right model
   path on the right server.
4. **Preflight check** (`sciknow/core/pulse.py`): when
   `llm_backend=dflash`, verify the `luce-dflash` binary exists and
   responds to a `/health`-style probe. Fail the preflight with an
   actionable error if not.
5. **VRAM ledger** (`sciknow/core/vram_budget.py` +
   `sciknow/core/gpu_ledger.py`): teach the swap-in / swap-out
   bookkeeping that DFlash's draft + verify tree state is held
   together with the target. Evict the reranker at writer-turn start
   and reload it at writer-turn end.
6. **L1 regression tests**: pin (a) config flag exists and defaults
   to ollama, (b) preflight routing, (c) per-role mapping still
   resolves identically across backends.

## Recommended execution plan

**Phase A (evaluation, 1 day, no code in `main`):**

1. Clone `Luce-Org/llama.cpp@luce-dflash` into a scratch dir.
2. Build with CUDA 12 (we have 535 driver matching their environment).
3. Pull `unsloth/Qwen3.5-27B-GGUF` Q4_K_M (we may already have it;
   otherwise ~16 GB download) + `z-lab/Qwen3.5-27B-DFlash` BF16
   draft (~3.5 GB).
4. Run `python3 scripts/bench_llm.py` from lucebox `dflash/`. Target:
   reproduce 129 tok/s HumanEval mean ±10 %. Success = go. Failure
   (builds broken, ops don't compile, numbers off by > 20 %) = fall
   back to Phase B.

**Phase B (fallback, ik_llama.cpp chain-draft):**

1. Build ik_llama.cpp from source.
2. Run `llama-server` with the flags from the tweet, minus `-ctk
   q8_0 -ctv q8_0` (use `q4_0` for 3090). Target: ~100-130 tok/s.
3. Same HTTP adapter applies.

**Phase C (integration, regardless of A vs B):**

1. `LLM_BACKEND` config flag + adapter + preflight + VRAM-ledger
   updates as listed above.
2. L1 regression test `l1_phase54_6_XXX_llm_backend_surface`.
3. Default backend stays Ollama — the new backend is opt-in via
   `.env` or per-project overlay.
4. Document measured tok/s in `docs/BENCHMARKS.md`.

**Phase D (rollout):**

1. Run a single `book write` chapter end-to-end with the new
   backend; compare score vs Ollama on the same prompt.
2. Run `autowrite-all` on one chapter; verify scorer/verify loops
   still converge.
3. If both pass, swap `BOOK_WRITE_MODEL` default to the new backend
   for this project (via overlay, not `.env` global).

## Risks

- **DFlash draft-target pair is fixed.** If z-lab hasn't shipped a
  draft for our current `Qwen3.6-27B` (we use Qwen3.6, they benchmark
  Qwen3.5), we'd need to either (a) downgrade to Qwen3.5 for the
  writer, or (b) wait for a 3.6 draft. The memo notes the official
  draft is `z-lab/Qwen3.5-27B-DFlash`; a version bump to Qwen3.6
  may require retraining the draft.
- **Custom llama.cpp fork maintenance.** Using the `luce-dflash`
  fork means pinning to a specific upstream commit and accepting the
  maintenance burden of rebasing when ggml ops change.
- **Sampling-param compatibility.** Our current writer uses
  temperature sampling with `top_p` / `repeat_penalty`. Both DFlash
  and llama-server support these; need to verify spec-dec accept
  probability computation is consistent with our sampler.
- **Multi-model eviction latency.** Swapping DFlash target + draft +
  BF16 pages into VRAM takes ~5-10 s cold. For one-shot writer calls
  this is negligible; for the autowrite scorer/verifier loop that
  hot-swaps between roles every iteration, the 5-10 s cost could
  dominate if not amortised. Answer: same `keep_alive=-1` pattern we
  already use with Ollama.

## Appendix — ik_llama.cpp specifics

From the user's tweet:

> My 4090 went from 26 → 154 tok/s Qwen 3.6 27B. Same GPU. Same
> Q4_K_M. No FP8, no extra quant. The unlock: ik_llama.cpp +
> speculative decoding using Qwen3-1.7B as the draft model. 85 %
> acceptance rate.
>
> How it works: The 1.7B "drafts" 12 tokens ahead, the 27B verifies
> them in a single forward pass. If 85 % are right, you get ~10
> tokens out per verification.
>
> Quality is identical to running the 27B solo — spec-dec is
> mathematically lossless.
>
> Ollama doesn't expose draft flags. That's why nobody's seeing this.

This is a clean description of classic chain spec-dec. The math:

- Without spec-dec: N tokens × 1 forward/token = N forwards.
- With spec-dec at α=0.85 acceptance, draft K=12: each "step"
  proposes K tokens + verifies in 1 target forward. Expected accepted
  tokens per step = K·α·(1−α^K)/(1−α) ≈ 5-6 at K=12, α=0.85.
  Throughput uplift ≈ 5-6× baseline — consistent with 26 → 154
  (5.9×) on 4090.

On 3090 we expect ~80-85 % of the 4090 number due to bandwidth +
TFLOPS gap → ~125-130 tok/s achievable. Still excellent.

## References

- Lucebox `dflash/RESULTS.md` — full HumanEval / Math500 / GSM8K
  reproducibility + budget sweep.
- `docs/research/Luce-Org__lucebox-hub.md` — architecture notes.
- `docs/research/huggingface__ml-intern.md` — companion memo (same
  /loop session).
- DFlash paper — arXiv:2602.06036 (z-lab, 2026).
- DDTree tree verify — Ringel & Romano, arXiv:2604.12989, 2026.
- `Luce-Org/llama.cpp@luce-dflash` — pinned llama.cpp fork with
  tree-mode ggml ops.
- User's ik_llama.cpp tweet — 2026-04-24 (no URL captured).

## Relevance verdict

- [x] **high — implement / port**. Phase A evaluation is a clear
  one-day spike; results decide between DFlash, ik_llama.cpp, or
  "stay with Ollama for now."
