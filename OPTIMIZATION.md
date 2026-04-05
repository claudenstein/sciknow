# SciKnow Optimization Plan — 3090 now, DGX Spark soon

Research-backed optimization plan based on a full audit of the sciknow codebase and 2026-current data on NVIDIA DGX Spark, marker-pdf, bge-m3/FlagEmbedding, Qdrant, Ollama, and vLLM.

---

## 1. The DGX Spark reality check

DGX Spark is **not faster than an RTX 3090 for single-user LLM inference**. It is a different tool for a different job.

| Metric | RTX 3090 | DGX Spark (GB10) |
|---|---|---|
| Memory bandwidth | ~936 GB/s (GDDR6X) | **~273 GB/s** (LPDDR5X unified) |
| VRAM / unified memory | 24 GB | 128 GB (122 GB usable) |
| Compute | ~35 TFLOPS FP16 | ~1 PFLOP FP4 w/ sparsity |

Published single-user numbers on Spark (early 2026):

- **Llama 3.1 70B FP8: ~2.7 tok/s decode**
- **Qwen 2.5 72B: ~4.6 tok/s**
- **Llama 3.2 90B TTFT: 133 s**; DeepSeek R1 70B TTFT: 180 s
- **Qwen3.5-35B-A3B MoE (patched vLLM 0.17.0 MXFP4): ~70 tok/s**
- **gpt-oss-120b MoE: ~80 tok/s**
- **QLoRA fine-tuning Llama 3.3 70B: ~5079 tok/s**

**Consequence for sciknow:** do not just move `LLM_MODEL` to Spark. The current `qwen2.5:32b-q4_K_M` on a 3090 (~30 tok/s interactive) will be **slower** on Spark. Use Spark where it changes the shape of what you can do — larger models for long-form synthesis/writing, and massively-parallel batch LLM jobs (clustering, Q&A generation, metadata enrichment, argument mapping).

---

## 2. Recommended compute split

| Workload | Where | Why |
|---|---|---|
| `bge-m3` embedding (ingestion + queries) | **3090** | Bandwidth-bound; 3090 is ~3.4× Spark. Fits in VRAM. |
| `bge-reranker-v2-m3` | **3090** | Same reason. Low latency matters for interactive search. |
| Marker (PDF → JSON) | **3090** | Surya OCR + layout models are bandwidth sensitive. |
| Qdrant + PostgreSQL | **CPU host of 3090** | Colocate with embeddings for lowest query latency. |
| Interactive `ask question` (small/medium LLM) | **3090** | 32B q4 at 30 tok/s beats Spark's 2.7–4.6 on 70B. |
| Interactive `ask` with **larger** model (70B dense or MoE) | **Spark** | Won't fit on 3090 at all. |
| `book write`, `ask write`, `ask synthesize` (long-form) | **Spark**, larger model | Quality >> latency here. |
| `catalog cluster` (batch LLM) | **Spark**, parallel | Embarrassingly parallel. |
| `db export --generate-qa` (5–10 s/chunk × thousands) | **Spark**, concurrent | Biggest wall-clock cost; ideal for continuous batching. |
| Fine-tuning retriever / reranker on your corpus | **Spark** | 5079 tok/s QLoRA on 70B + 122 GB unified. |

**Serving on Spark (early 2026):**
- **llama.cpp** — best single-user throughput
- **vLLM 0.17.0 + MXFP4 patches** — best for concurrent/batch workloads; MoE models (Qwen3.5-35B-A3B, gpt-oss-120b) are the sweet spot
- **Avoid NVFP4 quants** for now — vLLM support still incomplete; prefer AWQ-4bit or MXFP4
- **Ollama** works, ~3–4 tok/s behind llama.cpp; worth keeping for API compatibility. **Set `OLLAMA_NUM_PARALLEL` (default is 1!)**

---

## 3. Concrete changes, ordered by impact-per-effort

### 🔴 Tier 1 — biggest wins, small code changes

**1. Raise `EMBEDDING_BATCH_SIZE` from 8 to 32–64** (`sciknow/config.py:42`, used at `embedder.py:55`)

Current default is extremely conservative. bge-m3 at FP16 uses ~5.9 GB at batch 128 on A800. On a 3090 with only the embedder loaded, **batch 64 is safe, 128 is tight but possible**. Expected: **4–8× faster embedding stage**. Set to 32 when the LLM is co-resident, 64 when embedding-only.

**2. Set `OLLAMA_NUM_PARALLEL=4` on your Ollama host**

Default is **1** — every LLM-heavy sciknow command is serialized at the Ollama layer. Set on the 3090 now and on Spark later. Expected: **3–4× throughput** on bulk LLM commands at the cost of 20–40% per-request latency.

**3. Parallelize sequential loops in `db.py` / `catalog.py`**

Currently tight sequential loops with `time.sleep`:
- `db.py:745–776` — `db enrich` Crossref/OpenAlex calls. Expected: **~10× faster**.
- `db.py:1031–1115` — `db expand` Unpaywall → arXiv → Semantic Scholar lookups.
- `db.py:1213–1230` — `db export --generate-qa`. With `OLLAMA_NUM_PARALLEL=4`, submit concurrent chunks → **4× now, 8–16× on Spark**.
- `catalog.py:542–571` — `catalog cluster` sequential batches.

Each is a ~20-line diff around an existing loop.

**4. Expose Marker batch knobs** (`pdf_converter.py:201`)

Currently `config={}`. Marker supports `parallel_factor`/`batch_multiplier`. Projected jump from 25 pages/s to 122 pages/s on H100 with batching. Start at 2, monitor VRAM.

### 🟠 Tier 2 — moderate effort, significant wins

**5. Multi-worker ingestion** (`cli/ingest.py:52–144`)

Current `_run_worker_loop` spawns a single subprocess. Add `--workers N` to run 2 concurrent workers on the 3090 (Marker ~5 GB peak + embedder ~2 GB leaves room for 2). Better long-term: decouple Marker and embedding into separate worker pools fed by a Postgres job queue.

**6. Qdrant tuning** (`storage/qdrant.py:28–54`)

Current settings are defaults (`m=16`, `ef_construct=200`, `on_disk=True`, no quantization). Recommended:
- **Scalar int8 quantization** with `always_ram=True` — 75% memory savings on dense vectors, negligible recall hit, quantized vectors in RAM while originals stay on disk.
- Raise `m` to 32 and `ef_construct` to 400 on next re-index.
- Make all of the above configurable.

**7. Pool DB connections + persist clients**

- `storage/db.py:10` — default `pool_size=5`. Raise to 20+.
- `storage/qdrant.py:17` — create one `QdrantClient` module-level singleton.

**8. Route LLM calls by workload** (post-Spark)

Add `OLLAMA_HOST_HEAVY` → Spark. In `rag/llm.py`, pick the host based on call-site:
- Interactive `ask question` → 3090 (small/medium model)
- `book write`, `ask write`, `ask synthesize`, `book argue`, `book gaps` → Spark (larger model)
- Metadata fallback, `catalog cluster`, `db export --generate-qa` → Spark (parallel throughput)

### 🟡 Tier 3 — when Spark lands

**9. Upgrade the "heavy" LLM.** Sweet spots on Spark in early 2026:
- Qwen3.5-35B-A3B MoE + MXFP4 on patched vLLM: ~70 tok/s
- gpt-oss-120b MoE: ~80 tok/s
- Qwen3.5-122B-A10B NVFP4: fits at 75 GB
- Dense 70B/72B: only for offline batch

MoE models are the right call — they exploit Spark's memory capacity while keeping active parameters low enough for the bandwidth.

**10. Use vLLM (patched) for batch LLM workloads on Spark.** Continuous batching handles 100 concurrent users on Qwen3.5-35B-A3B at ~4.3 tok/s/user with zero errors. Wrap `rag/llm.py` with a small OpenAI-compatible client, point bulk commands at it, keep Ollama on 3090 for the interactive path.

**11. Fine-tune a domain reranker** on your library using QLoRA. Spark's 5000 tok/s on 70B makes this realistic. Probably the single largest retrieval-quality improvement available at this scale.

### 🟢 Tier 4 — nice-to-haves

- Metadata extraction LLM fallback parallelism
- ONNX / TensorRT export of bge-m3 and bge-reranker
- Binary quantization on the abstracts collection (small, in-memory already)

---

## 4. Execution order

1. `EMBEDDING_BATCH_SIZE=32` in `.env` + `OLLAMA_NUM_PARALLEL=4` on the Ollama host. Zero code changes, immediate 3–8× on two hot paths.
2. Parallelize `db enrich`, `db expand`, `db export --generate-qa`, `catalog cluster` loops.
3. Expose Marker `batch_multiplier` + raise `--workers N` in `sciknow ingest directory`. Watch VRAM.
4. Qdrant scalar quantization (requires collection rebuild — or apply next reset).
5. When Spark arrives: vLLM + MXFP4 with Qwen3.5-35B-A3B or gpt-oss-120b, add host-routing in `rag/llm.py`, point bulk/LLM-heavy commands at Spark.

---

## Sources

- [LMSYS DGX Spark in-depth review](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- [NVIDIA DGX Spark hardware docs](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA forum — Llama 70B 3.3 FP8 at 3 tok/s](https://forums.developer.nvidia.com/t/trouble-with-llama-70b-3-3-instruct-fp8-model-at-3-tokens-per-second/360643)
- [NVIDIA forum — vLLM 0.17.0 MXFP4 on DGX Spark](https://forums.developer.nvidia.com/t/vllm-0-17-0-mxfp4-patches-for-dgx-spark-qwen3-5-35b-a3b-70-tok-s-gpt-oss-120b-80-tok-s-tp-2/362824)
- [NVIDIA forum — Qwen3.5-122B-A10B NVFP4 on Spark](https://forums.developer.nvidia.com/t/qwen3-5-122b-a10b-nvfp4-quantized-for-dgx-spark-234gb-75gb-runs-on-128gb/361819)
- [NVIDIA forum — FP4/NVFP4 state in vLLM](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069)
- [Tom's Hardware DGX Spark review](https://www.tomshardware.com/pc-components/gpus/nvidia-dgx-spark-review)
- [Ollama on DGX Spark performance](https://ollama.com/blog/nvidia-spark-performance)
- [llama.cpp DGX Spark performance discussion](https://github.com/ggml-org/llama.cpp/discussions/16578)
- [Kaitchup — DGX Spark fine-tuning benchmarks](https://kaitchup.substack.com/p/dgx-spark-use-it-for-fine-tuning)
- [Sparktastic — choosing an inference engine on DGX Spark](https://medium.com/sparktastic/choosing-an-inference-engine-on-dgx-spark-8a312dfcaac6)
- [marker-pdf (datalab-to/marker) GitHub](https://github.com/datalab-to/marker)
- [marker multi-GPU processing guide](https://deepwiki.com/VikParuchuri/marker/8.2-multi-gpu-processing)
- [BAAI/bge-m3 model card](https://huggingface.co/BAAI/bge-m3)
- [FlagEmbedding BGE-M3 README](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/BGE_M3/README.md)
- [bge-m3 ONNX implementation](https://github.com/yuniko-software/bge-m3-onnx)
- [Qdrant vector search resource optimization](https://qdrant.tech/articles/vector-search-resource-optimization/)
- [Qdrant hybrid search with Query API](https://qdrant.tech/articles/hybrid-search/)
- [Glukhov — How Ollama handles parallel requests](https://www.glukhov.org/llm-performance/ollama/how-ollama-handles-parallel-requests/)
- [Ollama FAQ](https://docs.ollama.com/faq)
