"""
Phase 15.2 — Device selection for retrieval-time models (bge-m3, reranker).

The problem this solves: on a 24 GB GPU running a 27B Q4 LLM with a 16K
context window, Ollama uses ~23 GB of VRAM for the model + KV cache,
leaving only a few hundred MiB free. When sciknow tries to load
bge-m3 (~2 GB) for query embedding, it OOMs. The user sees an
intermittent CUDA OOM that "sometimes works, sometimes doesn't" — it
works when Ollama has unloaded the model after OLLAMA_KEEP_ALIVE
expires, fails when the model is still warm.

The fix: load bge-m3 + reranker on CPU when there isn't enough free
VRAM. CPU is slow for batch chunk embedding (used at ingest time),
but acceptable for retrieval-time query encoding (~1-2s) and reranking
of ~50 candidates (~5-10s). The total cost is small relative to the
LLM call that follows, and it lets Ollama keep the flagship model
warm in VRAM throughout the autowrite session.

Phase 54.6.322 — auto-default flipped to CPU on cards ≤24 GB.

Pre-fix: auto-detector probed free VRAM at the *moment* bge-m3 loaded.
That moment is usually BEFORE Ollama loads the 27B writer (the writer
loads lazily on the first chat call). So the probe saw ~22 GB free,
chose CUDA, parked 2-3 GB in VRAM permanently — and when the writer
later loaded, it had only ~10 GB free and partial-CPU-offloaded with
decode collapsing from 30 t/s to ~4 t/s. Phase 54.6.305/320 added
explicit eviction calls, but that's a band-aid: the next retrieve
re-loads the embedder onto GPU and the cycle repeats.

The math (see chat history of this phase): per-retrieve GPU saving is
~250 ms (50 ms vs 300 ms on CPU AVX-512). That is irrecoverable next
to the *minimum* 8 sec PCIe cost of swapping a 16 GB writer. The
asymmetry says: small + frequent thing → CPU; big + dominant thing →
GPU. So on cards where the writer is meant to OWN the GPU (≤24 GB
total VRAM, single-LLM workloads), the embedder + reranker default to
CPU.

Override remains: SCIKNOW_RETRIEVAL_DEVICE=cuda forces GPU even on
24 GB cards (use this on multi-GPU or 32+ GB setups, or when running
without an LLM co-resident — e.g. ingest-only workflows).

Env: SCIKNOW_RETRIEVAL_DEVICE=cpu|cuda|auto (default auto).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("sciknow.retrieval.device")

# Cached choice for the session — once we decide on a device, stick with
# it so we don't re-query VRAM every time. Reset by tests via
# reset_device_cache().
_cached_device: str | None = None

# Phase 54.6.322 — cards with TOTAL VRAM at or below this threshold
# default the retrieval models to CPU because they're sized for one
# big LLM resident at a time. 24 GB = RTX 3090, 4090, A5000.
_AUTO_CPU_TOTAL_VRAM_GB = 24.5


def reset_device_cache() -> None:
    """Forget the cached choice — useful for tests and after model swaps."""
    global _cached_device
    _cached_device = None


def get_retrieval_device() -> str:
    """Return 'cuda' or 'cpu' for the retrieval-time models.

    Decision order (Phase 54.6.322):
      1. Honour the SCIKNOW_RETRIEVAL_DEVICE env var if set to cpu/cuda.
      2. Auto: if torch.cuda is unavailable → cpu.
      3. Auto: total VRAM ≤ 24 GB → cpu (the GPU is sized to host one
         big LLM; the embedder is sub-second on CPU AVX-512 anyway, so
         the 250 ms per-retrieve saving never pays for the partial-
         offload cost it imposes on the writer).
      4. Auto: total VRAM > 24 GB → cuda (multi-LLM setup with room for
         both).

    Cached for the session after the first call.
    """
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    override = (os.environ.get("SCIKNOW_RETRIEVAL_DEVICE") or "").strip().lower()
    if override in ("cpu", "cuda"):
        logger.info("SCIKNOW_RETRIEVAL_DEVICE=%s (forced)", override)
        _cached_device = override
        return _cached_device

    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("torch.cuda.is_available() == False → cpu")
            _cached_device = "cpu"
            return _cached_device

        # mem_get_info returns (free, total) in bytes.
        free_b, total_b = torch.cuda.mem_get_info()
        free_gb = free_b / (1024 ** 3)
        total_gb = total_b / (1024 ** 3)

        # Phase 54.6.322 — auto-default to CPU on ≤24 GB cards.
        # The embedder + reranker on host RAM total ~2 GB; per-query
        # CPU AVX-512 latency is 0.3 s, irrelevant against an
        # autowrite phase dominated by minutes of LLM decode.
        # Keeping them out of VRAM lets the writer model own the GPU
        # at full speed (no partial CPU offload).
        if total_gb <= _AUTO_CPU_TOTAL_VRAM_GB:
            logger.info(
                "GPU total %.1f GB ≤ %.1f GB threshold → retrieval on CPU "
                "(embedder + reranker stay in host RAM so the writer LLM "
                "owns full VRAM). Set SCIKNOW_RETRIEVAL_DEVICE=cuda to "
                "force GPU when running ingest without an LLM co-resident.",
                total_gb, _AUTO_CPU_TOTAL_VRAM_GB,
            )
            _cached_device = "cpu"
        else:
            logger.info(
                "GPU total %.1f GB > %.1f GB threshold and %.1f GB free "
                "→ cuda",
                total_gb, _AUTO_CPU_TOTAL_VRAM_GB, free_gb,
            )
            _cached_device = "cuda"
    except Exception as exc:
        logger.warning("VRAM probe failed (%s) — defaulting to cpu", exc)
        _cached_device = "cpu"

    return _cached_device


def load_with_cpu_fallback(loader, *args, **kwargs):
    """Try to call `loader(*args, **kwargs, devices=device)` and fall back
    to devices='cpu' on CUDA OOM. Returns the loaded model.

    This is the second line of defence: even if get_retrieval_device()
    decided GPU was OK, the load itself can OOM if the LLM grew between
    the probe and the load. We catch that and retry on CPU.
    """
    chosen = get_retrieval_device()
    try:
        return loader(*args, devices=chosen, **kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg or "cublas" in msg:
            global _cached_device
            logger.warning(
                "Retrieval model failed to load on %s (%s) — retrying on CPU "
                "and caching that choice for the rest of the session.",
                chosen, type(exc).__name__,
            )
            _cached_device = "cpu"
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            return loader(*args, devices="cpu", **kwargs)
        raise
