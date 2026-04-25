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

Phase 54.6.323 — REVERT 54.6.322's "auto-default-CPU" behavior.

54.6.322 estimated CPU rerank at ~5-10 s for top-50 and projected
total CPU-side overhead at ~3 min per autowrite. **Measured on the
user's actual 32-core / 3090 / Qwen3-Embedding-4B-dense pipeline**:
the retrieve+rerank gap was 10 min 25 s for ONE cycle — three times
the writer LLM phase it preceded. The estimate undercounted by ~10×
because the dual-embedder pipeline runs (a) Qwen3-Embedding-4B
(4B params dense), (b) bge-m3 sparse + ColBERT prefetch, and
(c) the cross-encoder rerank — every step compounds on CPU.

Reverting to the pre-322 behavior: auto picks CUDA when there's free
VRAM, CPU only when free VRAM falls below the headroom threshold.
Eviction-before-LLM (54.6.305 / 54.6.320) remains the correct
band-aid for partial-load. The real culprit on 24 GB cards is the
8 GB Qwen3-Embedding-4B dense embedder; that gets a separate
eviction in `_get_dense_embed_model` (already in place since 54.6.305).

Env override unchanged: SCIKNOW_RETRIEVAL_DEVICE=cpu|cuda|auto.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("sciknow.retrieval.device")

# Cached choice for the session — once we decide on a device, stick with
# it so we don't re-query VRAM every time. Reset by tests via
# reset_device_cache().
_cached_device: str | None = None

# Phase 54.6.323 — revert to original headroom logic. The 54.6.322
# attempt to default CPU on ≤24 GB cards was reverted after measured
# data showed the dual-embedder + ColBERT-rerank pipeline runs ~10×
# slower on CPU than the back-of-envelope estimate. Keep this
# constant exported for the L1 regression's transitional check.
_AUTO_CPU_TOTAL_VRAM_GB = 24.5

# How much free VRAM (in GB) we want to see before choosing GPU.
# bge-m3 itself is ~2 GB; the reranker is another ~1 GB. We need 4 GB
# of slack to fit both with breathing room for working tensors.
_VRAM_HEADROOM_GB = 4.0


def reset_device_cache() -> None:
    """Forget the cached choice — useful for tests and after model swaps."""
    global _cached_device
    _cached_device = None


def get_retrieval_device() -> str:
    """Return 'cuda' or 'cpu' for the retrieval-time models.

    Decision order (Phase 54.6.323 — reverted from 54.6.322):
      1. Honour the SCIKNOW_RETRIEVAL_DEVICE env var if set to cpu/cuda.
      2. Auto: if torch.cuda is unavailable → cpu.
      3. Auto: free VRAM < `_VRAM_HEADROOM_GB` (4 GB) → cpu (LLM is
         loaded and squeezing us — fall back rather than OOM).
      4. Otherwise → cuda.

    The partial-load risk is now handled by:
      - 54.6.305 / 54.6.320: explicit `_release_gpu_models()` before
        every LLM-heavy phase in autowrite.
      - 54.6.305 itself: `_get_dense_embed_model` (Qwen3-Embedding-4B,
        the 8 GB outlier) preflights and falls back to CPU OOM.

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

        free_b, total_b = torch.cuda.mem_get_info()
        free_gb = free_b / (1024 ** 3)
        total_gb = total_b / (1024 ** 3)

        if free_gb < _VRAM_HEADROOM_GB:
            logger.warning(
                "Only %.1f / %.1f GB free on GPU — falling back to CPU "
                "for bge-m3 + reranker (likely the LLM is loaded in VRAM). "
                "Set SCIKNOW_RETRIEVAL_DEVICE=cuda to force GPU anyway.",
                free_gb, total_gb,
            )
            _cached_device = "cpu"
        else:
            logger.info(
                "%.1f / %.1f GB free on GPU → cuda", free_gb, total_gb,
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
