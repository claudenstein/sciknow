# Inference servers — ranked for sciknow on RTX 3090

**Generated**: 2026-04-25.
**Profile assumed throughout**: single user, batch = 1, 24 GB VRAM,
multi-model swap pressure (writer Qwen3.6-27B Q4_K_M ~16 GB +
sometimes-different scorer + MinerU 2.5 VLM ~4 GB + bge-m3 embedder
~1 GB + bge-reranker-v2-m3 ~1 GB), GGUF current.

This memo synthesises 2026-04 ground-truth on every serious self-
hosted inference server and ranks them against that profile.

## tl;dr ranking

1. **Ollama (current)** — keep. It's the only server that holds
   writer + scorer + bge-m3 + bge-reranker co-resident and hot-swaps
   cleanly within 24 GB. Cost: no exposed speculative decoding.
2. **llama-server (mainline)** — best low-risk uplift path. Same
   GGUFs, native `/v1/rerank` (kills your sentence-transformers
   sidecar), `--model-draft` for chain spec-dec. Trivial migration.
3. **LocalAI** — direct Ollama replacement with richer multi-model
   orchestration, reranker as first-class, opt-in spec-dec via YAML
   per model. Same GGUF files.
4. **Lucebox DFlash** — opt-in writer-only backend; 37 → 129-207
   tok/s on this hardware. Research fork, single hardcoded model
   pair, not a daemon.
5. **TabbyAPI / ExLlamaV3** — strongest single-stream perf for a
   generic API. Worth it if you commit to re-quantising every model
   to EXL3 and accept losing the GGUF ecosystem.
6. **Skip for this profile**: vLLM, SGLang, TGI, Aphrodite (batch>1
   designs), MLC-LLM (mobile target), llamafile (distribution
   format), ik_llama.cpp (CPU/MoE niche, no win on dense 27B CUDA),
   HF TGI (datacentre).

## What sciknow actually needs

| Requirement | Why |
|---|---|
| **Multi-model co-resident with eviction** | Writer + scorer + embedder + reranker + sometimes VLM. The 54.6.305 / 54.6.320 phases are scar tissue from this. |
| **GGUF Q4_K_M support** | All current weights are GGUF; re-quantising is days of work + storage. |
| **OpenAI-compat streaming chat** | `sciknow/rag/llm.py::stream` already speaks Ollama's API; an OpenAI shim is a drop-in. |
| **Prefix-cache reuse across calls** | CoV / autowrite call the same source-passages prefix many times. 54.6.319 batches CoV but score / verify / revise all share prefixes too. |
| **Optional**: speculative decoding | The user's tweet-driven 26→154 tok/s win lives here. |
| **Optional**: built-in reranker | We currently run bge-reranker-v2-m3 in a separate sentence-transformers process. |
| **Single-user batch=1 throughput**, NOT multi-tenant | Anything optimised for batch>1 (PagedAttention, continuous batching) is wasted. |

## Per-server analysis

### 1. Ollama (current baseline)

- Decode tok/s on 3090: ~37 t/s Qwen3.5-27B Q4_K_M, ~107 t/s Qwen3.6-35B-A3B MoE.
- Quantization: GGUF only (delegates to bundled llama.cpp).
- Multi-model: **best-in-class.** Models load on first request, evict on `OLLAMA_KEEP_ALIVE` timeout, multiple co-resident if VRAM allows.
- Speculative decoding: **not exposed**. No `model-draft` config option in 2026-04. Hard wall — confirmed by the user's own tweet.
- Embeddings/rerankers: `/api/embeddings` works; reranker is feature request [#3749](https://github.com/ollama/ollama/issues/3749), not first-class. We ship a separate sentence-transformers sidecar for bge-reranker-v2-m3.
- OpenAI API: yes (`/v1/chat/completions`) + native `/api/chat`.
- Prefix cache: inherited from llama.cpp; survives keep-alive.
- Maintenance: extremely active, conservative on advanced features.
- **Migration cost from Ollama**: zero (it is Ollama).
- **Verdict**: keep as the writer host until a measurable spec-dec win is in hand. The pain points (no spec-dec, sentence-transformers sidecar) don't outweigh the multi-model story today.

### 2. llama.cpp / llama-server (mainline, ggml-org)

- Decode tok/s: ~37–43 Qwen 27–32B Q4_K_M (matches Ollama since it wraps llama.cpp).
- Quantization: **GGUF native** — Q4_K_M / Q5_K_M / Q6_K / Q8_0, IQ-quants, TQ1/TQ2/TQ3 ternary. No EXL2/AWQ/GPTQ.
- Multi-model: server-side model swap landed in 2026; `llama-server` can route to a models dir and load on demand. **One resident at a time** with disk-cache reuse. No simultaneous multi-model.
- Speculative decoding: `--model-draft` (chain spec-dec). Up to 2-3× on coding-style outputs. **Not** tree-verify, **not** block-diffusion. PR #19493 (April 2026) added ngram-cache/ngram-mod variants — community benchmark on Qwen3.6-35B-A3B + 3090 shows **net-negative** for that MoE; works fine for dense 27B/32B.
- Embeddings/rerankers: **native.** `/v1/embeddings` + `/v1/rerank`. bge-m3 and bge-reranker-v2-m3 both supported (`--embedding --pooling rank`, three INI keys: `reranking=true`, `pooling=rank`, `embedding=true`).
- OpenAI API: yes, streaming.
- Prefix cache: yes — `--cache-reuse` and slot-level KV reuse.
- Maintenance: heavily active, the upstream of nearly every other server here.
- **Ollama migration cost**: **trivial.** Same GGUFs, same backend; existing `OLLAMA_KV_CACHE_TYPE=q4_0` translates to `--cache-type-k q4_0 --cache-type-v q4_0`. /v1 endpoint is OpenAI-compatible.
- **Verdict**: the cleanest uplift if we can live without Ollama's multi-model eviction. Loses the writer-and-scorer-co-resident pattern; gains spec-dec + native reranker. Net win only if we run *one* writer model and stop the scorer-swap pattern.

### 3. ik_llama.cpp (Iwan Kawrakow's fork)

- Decode tok/s: faster than mainline on **CPU** and **hybrid CPU+GPU**; on a 3090 (CUDA-only) the gap shrinks to mainline-parity or small uplift on Qwen 27B/32B.
- Quantization: GGUF + new SOTA quants (IQ4_KS, IQ4_KSS, IQ5_K, IQ6_K, "Trellis" TQ1/TQ2 with new CUDA kernels). All quants now have quantized matmul CUDA kernels (May 2025 refactor).
- Multi-model: same single-model-per-process model as upstream.
- Speculative decoding: inherits llama.cpp chain spec-dec. **No** tree-verify / DFlash kernels. Niche optimisations are CPU-side and MoE/MLA (DeepSeek FlashMLA + fused MoE).
- Embeddings/rerankers: yes (inherited).
- OpenAI API: yes (inherited).
- Prefix cache: yes (inherited).
- Maintenance: active solo project. New home announced as `codeberg.org/ikawrakow/illama` ("under construction") — watch for upstream/downstream split.
- **Ollama migration cost**: trivial; same GGUF, same flags.
- **Caveat**: the user's tweet citing 26 → 154 tok/s on a 4090 was a chain spec-dec config — reproducible via mainline llama.cpp too. The IK kernels are not the source of that number.
- **Verdict**: try only if benchmarks show > 5 % on your specific dense-27B CUDA workload. Skip otherwise.

### 4. LocalAI (mudler, Go)

- Decode tok/s: equals whatever backend it dispatches to (llama.cpp by default).
- Quantization: anything its backends support — GGUF, GPTQ, AWQ, EXL2 via subprocess.
- Multi-model: **YES — designed for this.** Per-model YAML, lazy load, hot-swap, multiple resident if VRAM permits. Closest-to-Ollama orchestration.
- Speculative decoding: configurable via `spec_type` / `spec_n_max` in model YAML; auto-disabled when multimodal mmproj is active.
- Embeddings/rerankers: **native, multiple backends** including llama.cpp's `/v1/rerank`.
- OpenAI API: yes (project's mission).
- Prefix cache: through llama.cpp.
- Maintenance: active; March 2026 added Agent management, React UI, MCP client, MLX-distributed.
- **Ollama migration cost**: easy. OpenAI-compat plus a YAML-per-model. Same GGUF files.
- **Verdict**: the most direct Ollama replacement if we want (a) richer multi-model orchestration, (b) reranker as first-class, (c) speculative decoding without leaving the GGUF ecosystem. The upgrade Ollama could have been.

### 5. Lucebox DFlash (Luce-Org/lucebox-hub)

See `docs/research/Luce-Org__lucebox-hub.md` for the full architectural memo. Short version:

- Decode tok/s: **129.5 mean (HumanEval) / 207.6 peak** on Qwen3.5-27B Q4_K_M on RTX 3090. **3.43× over autoregressive.**
- Quantization: **only** Q4_K_M target + BF16 draft, hardcoded model pair (target `unsloth/Qwen3.5-27B-GGUF`, draft `z-lab/Qwen3.5-27B-DFlash`).
- Multi-model: **single hardcoded pair.** Not a general server.
- Speculative decoding: block-diffusion draft + DDTree budget-22 tree verify — strongest spec-dec on this hardware class today.
- Embeddings/rerankers: out of scope.
- OpenAI API: not advertised; the demo is a CLI.
- Prefix cache: KV q4_0 (8× compression) for 128K context; sliding `target_feat` ring.
- Maintenance: brand-new (April 2026), 1-person research project; vendor-fork dependency on `Luce-Org/llama.cpp@luce-dflash`.
- **Ollama migration cost**: high — full backend swap; needs VRAM-ledger reshuffle (DFlash side ~21 GB, evicts the reranker during writer turns).
- **Verdict**: highest single-stream throughput demonstrated, but production risk is high (single fork, one model pair, no API server). Best treated as an opt-in writer backend behind a feature flag. Companion memo at `docs/research/speculative_decoding.md` already lays out the staged plan.

### 6. TabbyAPI / ExLlamaV3

- Decode tok/s: **historically the king** of single-stream consumer GPU. Qwen 32B at 3.5–4.5 bpw EXL2 / EXL3: typically 40–60 tok/s on 3090 dense, similar to llama.cpp Q4 with lower memory and slightly better PPL.
- Quantization: **EXL2** (any avg bpw 2–8), **EXL3** (new 2025-2026, QTIP-based, smaller files for same quality), GPTQ (legacy).
- Multi-model: TabbyAPI supports load/unload via API; **one resident at a time.** Does NOT keep multiple models warm.
- Speculative decoding: yes. ExLlamaV3's dynamic generator unifies AR + spec-dec; tree branch supported, EAGLE in roadmap, no DFlash.
- Embeddings/rerankers: TabbyAPI added embedding model support; cross-encoder rerankers not a primary target.
- OpenAI API: yes, streaming.
- Prefix cache: yes (paged + radix-style for long contexts in V3).
- Maintenance: active. ExLlamaV3 (`turboderp-org/exllamav3`) is the going-forward branch; V2 enters maintenance.
- **Ollama migration cost**: high one-time — must re-download every model as EXL2/EXL3 (no GGUF). API surface is OpenAI-compat so client code is fine.
- **Verdict**: strongest *generic-API* single-user perf. Right answer if we commit to one writer model and accept re-quantising. Loses GGUF interop with anything else (MinerU, Ollama dev tools, etc.).

### 7. vLLM (Berkeley, official)

- Decode tok/s: not its strong suit at batch=1 — PagedAttention is designed for batch>1. On a 3090 with Qwen 32B you'll see ~30-48 tok/s depending on quant (FP8 ~48 on QwQ-32B; Q4 lower).
- Quantization: AWQ, GPTQ, Marlin, FP8, bitsandbytes, GGUF (added piecemeal — not the recommended path).
- Multi-model: **one model per process.** Multi-model = multiple vLLM processes + a router. CUDA graphs make swap cost high.
- Speculative decoding: native EAGLE / EAGLE-3 / Medusa / n-gram / DFlash; tree-style verify via EAGLE-3. GGUF target + spec-dec is buggy — known issue.
- Embeddings/rerankers: yes via `--task embed` / `--task score`, but again one model per process.
- OpenAI API: yes.
- Prefix cache: automatic prefix caching (block-level hashing).
- Maintenance: extremely active, de-facto cloud serving engine.
- **Verdict**: **dead-end for sciknow's writer slot.** Useful only as a sidecar for batched embedding workloads (which we don't have).

### 8. SGLang (LMSYS)

- Decode tok/s: comparable to vLLM at single stream; **wins on long shared prefixes** (RadixAttention's token-level radix tree beats vLLM's block hashing).
- Quantization: AWQ, GPTQ, FP8, INT8, MXFP4, FP4 (Blackwell). **GGUF support is partial.**
- Multi-model: one per server; multi-LoRA on a base.
- Speculative decoding: EAGLE / EAGLE-3 / Medusa / draft.
- Embeddings/rerankers: yes — explicit `EmbeddingModel` / cross-encoder support.
- Prefix cache: **best-in-class** (RadixAttention is the headline).
- Maintenance: extremely active; Feb 2026 announced 25× perf on GB300 NVL72 — direction is datacentre.
- **Verdict**: irrelevant unless we're doing heavy *shared-prefix* multi-tenant agents. sciknow's per-paper retrieval prompts have low cross-request prefix overlap.

### 9. HuggingFace TGI

- Decode tok/s: tuned for throughput, not single-stream. ~30-45 tok/s class on Qwen 32B Q4 on 3090.
- Quantization: AWQ, GPTQ, EETQ, FP8 (Marlin), bitsandbytes; GGUF runs but is second-class.
- Multi-model: one model per process.
- Speculative decoding: Medusa heads + assistant model; no DFlash/tree.
- Embeddings/rerankers: TEI is a **separate** HF project. Two services if you want both.
- License: was HFOIL (non-OSI) at v1.0; **reverted to Apache 2.0** as currently published.
- **Verdict**: built for cloud H100/A100 fleets, not single-3090 workstations. Skip.

### 10. Aphrodite Engine (vLLM fork, PygmalionAI / Ruliad)

- Quantization: **broadest list anywhere** — AQLM, AutoRound, AWQ, BitNet, bitsandbytes, EETQ, GGUF, GPTQ, QuIP#, SqueezeLLM, Marlin, FP2-FP12, ModelOpt, TorchAO, VPTQ, compressed_tensors, MXFP4, EXL2.
- Otherwise inherits vLLM (batch>1 design, one model per process).
- **Verdict**: shines if we want a quant vLLM doesn't ship. Otherwise vLLM is upstream.

### 11. MLC-LLM (TVM)

- Target is **mobile/edge** (Snapdragon, WebGPU, iOS). Qwen 3.5 support filed only March 2026 (issue #3448).
- **Verdict**: dead-end for this profile. Skip.

### 12. llamafile (Mozilla.ai)

- Decode tok/s ≈ mainline llama.cpp (it bundles llama.cpp). v0.10.0 (March 2026) was a from-scratch core rebuild.
- **Verdict**: distribution format, not a server platform. Wrong tool for sciknow's daemon use.

## Migration cost vs. Ollama (sciknow-side)

| Server | API change | Weights re-prep | Reranker sidecar gone? | spec-dec? | Multi-model? |
|---|---|---|---:|---:|---:|
| Ollama (now) | — | — | no | no | yes |
| llama-server | OpenAI shim (~40 LOC adapter) | none | yes | yes | no |
| LocalAI | OpenAI shim | none | yes | yes | yes |
| Lucebox DFlash | full backend swap | re-pull pair | no | yes (best) | no |
| TabbyAPI / EXL3 | OpenAI shim | full re-quantise | partial | yes | no |
| vLLM | OpenAI shim | AWQ/GPTQ re-prep recommended | partial | partly | no |

## Recommended sequencing for sciknow

**Phase A** (this quarter, low-risk):
- Add `LLM_BACKEND` setting (`ollama` default, `llama_cpp_server` opt-in).
- Spike a `llama-server` config that hosts the writer + bge-reranker-v2-m3 in one process; benchmark vs current Ollama + sentence-transformers sidecar.
- If wins ≥ 10 % wall-clock on a real autowrite chapter, ship the toggle as opt-in.

**Phase B** (next quarter, conditional):
- Run the DFlash spike from `docs/research/speculative_decoding.md` (Phase A there). If 129 tok/s reproduces, add `LLM_BACKEND=dflash` as a writer-only opt-in. The VRAM-ledger work landed in 54.6.305 / 54.6.320 already covers the eviction needed.

**Phase C** (year-out, if multi-model orchestration becomes painful):
- Migrate to LocalAI as the primary daemon. Brings reranker + spec-dec + multi-model under one OpenAI-compat surface, keeps GGUF, replaces the Ollama wrapper.

**Stay clear of**: vLLM, SGLang, TGI, Aphrodite for the writer path on this hardware. They're built for the wrong workload (batch>1, datacentre).

## Sources

- llama.cpp speculative decoding: <https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md>
- llama-server embeddings + reranker: <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- llama-server Qwen3 reranker + embedding gist: <https://gist.github.com/VooDisss/42bce4eb5c76d3c325633886c5e348ee>
- ik_llama.cpp README: <https://github.com/ikawrakow/ik_llama.cpp/blob/main/README.md>
- ik_llama.cpp new home: <https://codeberg.org/ikawrakow/illama>
- Ollama #3749 — rerankers feature request: <https://github.com/ollama/ollama/issues/3749>
- Ollama multi-GPU notes 2026: <https://www.knightli.com/en/2026/04/19/ollama-multiple-gpu-notes/>
- TGI license history (HFOIL → Apache 2.0): <https://github.com/huggingface/text-generation-inference/issues/726>
- TGI repo: <https://github.com/huggingface/text-generation-inference>
- vLLM speculative decoding: <https://docs.vllm.ai/en/latest/features/spec_decode/>
- vLLM EAGLE draft models: <https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/>
- SGLang RadixAttention overview: <https://medium.com/byte-sized-ai/prefix-caching-sglang-vs-vllm-token-level-radix-tree-vs-block-level-hashing-b99ece9977a1>
- SGLang docs: <https://sgl-project.github.io/>
- TabbyAPI repo: <https://github.com/theroyallab/tabbyAPI>
- ExLlamaV3 repo: <https://github.com/turboderp-org/exllamav3>
- Serving ExLlamaV3 with TabbyAPI: <https://kaitchup.substack.com/p/serving-exllamav3-with-tabbyapi-accuracy>
- MLC-LLM repo: <https://github.com/mlc-ai/mlc-llm>
- MLC-LLM Qwen3.5 request (2026-03): <https://github.com/mlc-ai/mlc-llm/issues/3448>
- Aphrodite Engine quantization wiki: <https://github.com/aphrodite-engine/aphrodite-engine/wiki/8.-Quantization>
- LocalAI llama.cpp backend / spec-dec: <https://github.com/mudler/LocalAI/blob/master/.agents/llama-cpp-backend.md>
- LocalAI What's New: <https://localai.io/basics/news/index.html>
- llamafile 0.10.0 release coverage (March 2026): <https://www.helpnetsecurity.com/2026/03/20/llamafile-0-10-0-released/>
- llamafile repo: <https://github.com/mozilla-ai/llamafile>
- Qwen3.6-35B-A3B + RTX 3090 spec-dec benchmark: <https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090>
- HackMD: every spec-dec mode tested on 3090: <https://hackmd.io/ODXuOQNzSiyUITz7g9mtBw>
- Lucebox DFlash 207 tok/s on 3090 (HN): <https://news.ycombinator.com/item?id=47838788>
- Companion memo: `docs/research/Luce-Org__lucebox-hub.md`
- Companion memo: `docs/research/speculative_decoding.md`
