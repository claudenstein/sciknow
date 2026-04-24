"""
Thin wrapper around the Ollama Python client.

Provides streaming and non-streaming chat, reusing a single client instance.
All calls are logged to the sciknow logger with model name, prompt sizes,
and duration for post-hoc debugging.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import ollama

from sciknow.config import settings

logger = logging.getLogger("sciknow.llm")

_client: ollama.Client | None = None


def _get_client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=settings.ollama_host)
    return _client


def release_llm(models: list[str] | None = None) -> list[str]:
    """Unload Ollama models from VRAM.

    Phase 44.1 — the benchmark harness surfaced a 50–70× slowdown on
    the bge-m3 embedder and bge-reranker-v2-m3 when an LLM was still
    resident in VRAM (both fall back to CPU). Call this helper before
    a mixed LLM → embedder/reranker flow so the GPU is free for the
    small-model workload.

    Ollama's way to unload is to issue a tiny request with
    ``keep_alive=0``; the server evicts the model as soon as the
    request returns. There is no explicit ``/api/unload`` endpoint.

    Args:
        models: specific model names to unload. If None, queries
                ``/api/ps`` and unloads every currently-loaded model.

    Returns the list of model names that were asked to unload.
    Errors are swallowed per-model — a stuck unload shouldn't
    block the caller's real workload.
    """
    client = _get_client()
    if models is None:
        try:
            ps = client.ps()
            models = [m.model if hasattr(m, "model") else m.name
                      for m in (ps.models or [])]
        except Exception as exc:
            logger.debug("release_llm: ps() failed: %s", exc)
            return []
    released: list[str] = []
    for name in models or []:
        try:
            # Empty prompt + keep_alive=0 tells Ollama to unload. Using
            # generate() (not chat) because generate is cheaper and still
            # triggers the eviction on every Ollama version we support.
            client.generate(model=name, prompt="", keep_alive=0)
            released.append(name)
            logger.info("released Ollama model=%s", name)
        except Exception as exc:
            logger.debug("release_llm(%s) failed: %s", name, exc)
    return released


def _release_llm_for_preflight() -> int:
    """Phase 54.6.290 — VRAM-budget releaser adapter.

    The preflight registry expects ``() -> int (mb_freed)``; release_llm
    returns a name list.  This adapter wraps it and returns a coarse
    MB estimate (7000 per unloaded model — qwen3:30b fits that
    envelope; smaller models over-report by a couple GB, which just
    makes the subsequent preflight check pessimistic → safe).
    """
    try:
        names = release_llm()
    except Exception as exc:
        logger.debug("release_llm for preflight failed: %s", exc)
        return 0
    return len(names) * 7000


try:
    from sciknow.core.vram_budget import register_releaser as _register
    # priority=10 → fires FIRST in the preflight cascade.  Ollama
    # unload is a sub-second HTTP call and its models reload on
    # demand, so dropping them has the lowest operational cost.
    _register("ollama_llm", _release_llm_for_preflight, priority=10)
except ImportError:  # pragma: no cover
    pass


def warm_up(
    model: str | None = None,
    num_ctx: int = 16384,
    num_batch: int = 1024,
) -> bool:
    """Phase 54.6.31 — pre-load the model into VRAM with a zero-token
    generate call so the first real request doesn't pay cold-start.

    Ollama's recommended production pattern: trigger a model load at
    service startup so users never wait for the ~3-10s cold load on
    their first request. Here we use it at the top of long hot-loop
    commands (``wiki compile``, ``book autowrite``) so the first paper
    / section doesn't show anomalously slow timing.

    Uses keep_alive=-1 so the model stays resident for the actual
    workload. The ``num_ctx`` and ``num_batch`` values should match
    what the workload will use — otherwise Ollama loads one instance
    now and another on the first real call.

    Returns True on success, False if the server is unreachable.
    Never raises — this is best-effort.
    """
    try:
        client = _get_client()
        chosen = model or settings.llm_model
        t0 = time.monotonic()
        client.generate(
            model=chosen, prompt="",
            keep_alive=-1,
            options={"num_ctx": num_ctx, "num_batch": num_batch, "num_predict": 1},
        )
        elapsed = time.monotonic() - t0
        logger.info(f"LLM warmed model={chosen} ctx={num_ctx} batch={num_batch} in {elapsed:.1f}s")
        return True
    except Exception as exc:
        logger.warning(f"LLM warmup failed for {model}: {exc}")
        return False


# Phase 54.6.318 — per-call stats side channel.
#
# Ollama chunks the streamed response with the FINAL chunk carrying
# decode/prefill timing fields (eval_count, eval_duration,
# prompt_eval_count, prompt_eval_duration, load_duration in ns).
# Callers (the web job runner, autowrite, CoV, etc.) want to see
# decode tok/s separately from wall-clock tok/s — under heavy
# prefill workloads (long contexts, multi-question CoV) the wall-
# clock rate is dominated by prefill periods that emit no tokens,
# so the existing 3-second rolling tps reads ~1 t/s while decode
# is actually running at the model's native speed.
#
# We capture the final-chunk stats per call and stash them on a
# thread-local list. The streamer thread (web/app.py
# _run_generator_in_thread) drains the list after each yielded
# event and folds the numbers into the per-job pulse so the CLI +
# task-bar can show both rates.

import threading as _threading

_call_stats_local = _threading.local()


def _record_call_stats(stats: dict) -> None:
    """Append per-call Ollama stats to the current thread's buffer."""
    if not hasattr(_call_stats_local, "calls"):
        _call_stats_local.calls = []
    _call_stats_local.calls.append(stats)


def drain_call_stats() -> list[dict]:
    """Pop and return all stats accumulated on this thread since the
    last drain. Called by the web job runner after each event yield."""
    calls = getattr(_call_stats_local, "calls", [])
    _call_stats_local.calls = []
    return calls


def stream(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.2,
    num_ctx: int = 16384,
    num_batch: int = 1024,
    num_predict: int | None = None,
    keep_alive: str | int | None = -1,
    format: dict | str | None = None,
    think: bool | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> Iterator[str]:
    """
    Stream LLM response tokens. Yields string fragments as they arrive.

    keep_alive: Ollama keep_alive parameter ("-1" = infinite, "60m", etc.).
                Prevents model unload between calls, enabling prompt caching.
                **Phase 54.6.30: default changed from None → -1.** Pre-fix
                the default meant Ollama's 5-minute TTL applied, which
                evicted the model between multi-phase pipeline steps
                (autowrite loop: write → score → verify → revise → …)
                even though they were processing the same model. Sticky
                by default; callers that want to unload pass
                ``keep_alive=0`` or ``"30s"``.

    num_batch: Prompt-evaluation batch size. **Phase 54.6.31: default
                raised from Ollama's native 512 → 1024** based on the
                community benchmarks (Ollama perf docs, Medium guides):
                1024 gives ~60% higher prompt-eval throughput than 512
                on any modern GPU at a cost of <1 GB extra VRAM for
                7B-class models. 2048 can give another boost but
                occasionally OOMs on 10 GB cards, so 1024 is the safe
                default that captures most of the win. Must be ≥32
                (llama.cpp docs: below that the cuBLAS kernels are
                bypassed entirely, which is much slower). Increase to
                2048 on the 3090 / DGX if VRAM is comfortable.

    format:     JSON schema dict for structured output (Ollama v0.5+).
                Guarantees valid JSON matching the schema.
    """
    client = _get_client()
    chosen_model = model or settings.llm_model
    t0 = time.monotonic()
    logger.info(
        f"LLM stream  model={chosen_model}  system={len(system)}c  "
        f"user={len(user)}c  temp={temperature}  ctx={num_ctx}"
        f"{'  format=json_schema' if format else ''}"
    )
    token_count = 0
    try:
        options = {"temperature": temperature, "num_ctx": num_ctx,
                   "num_batch": num_batch}
        # Phase 54.6.32 — num_predict cap (if set) prevents runaway
        # generation. Critical for format=json_schema calls where the
        # schema has unbounded arrays.
        if num_predict is not None:
            options["num_predict"] = num_predict
        # Phase 54.6.85 — Qwen recommends top_p 0.8 non-thinking /
        # 0.95 thinking and top_k 20 for their entire 3.x family.
        # Passing them as kwargs lets callers opt in per model.
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        kwargs: dict = {
            "model": chosen_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": True,
            "options": options,
        }
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive
        if format is not None:
            kwargs["format"] = format
        # Phase 54.6.85 — Ollama native think flag for hybrid Qwen3.5/3.6
        # models. ``think=False`` injects /no_think so the model skips
        # CoT; ``think=True`` enables it on models that support soft-
        # switching. None leaves Ollama's default behavior (thinks-if-
        # the-model-thinks).
        if think is not None:
            kwargs["think"] = think
        response = client.chat(**kwargs)
        # Phase 54.6.318 — capture the final chunk's stats. Ollama
        # streams chunks; the last one carries done=True plus the
        # native decode/prefill timing fields. We record them all so
        # the dashboard can compute decode tok/s (prefill-free) and
        # surface model-load + prefill latency separately.
        last_stats: dict = {}
        for chunk in response:
            token_count += 1
            yield chunk.message.content
            # Inspect attribute on the chunk for `done` semantics —
            # python ollama returns ChatResponse with `done` + timing
            # fields populated only on the terminal chunk.
            try:
                if getattr(chunk, "done", False):
                    last_stats = {
                        "model": chosen_model,
                        "eval_count": int(getattr(chunk, "eval_count", 0) or 0),
                        "eval_duration_ns": int(getattr(chunk, "eval_duration", 0) or 0),
                        "prompt_eval_count": int(getattr(chunk, "prompt_eval_count", 0) or 0),
                        "prompt_eval_duration_ns": int(getattr(chunk, "prompt_eval_duration", 0) or 0),
                        "load_duration_ns": int(getattr(chunk, "load_duration", 0) or 0),
                        "total_duration_ns": int(getattr(chunk, "total_duration", 0) or 0),
                    }
            except Exception:
                pass
        if last_stats and last_stats.get("eval_count"):
            _record_call_stats(last_stats)
            # Compute decoded + prefill tok/s for the log line.
            ed = (last_stats.get("eval_duration_ns") or 0) / 1e9
            ec = last_stats.get("eval_count") or 0
            pd = (last_stats.get("prompt_eval_duration_ns") or 0) / 1e9
            pc = last_stats.get("prompt_eval_count") or 0
            decode_tps = (ec / ed) if ed > 0 else 0.0
            prefill_tps = (pc / pd) if pd > 0 else 0.0
            logger.info(
                "LLM perf    model=%s  decode=%dtok/%.2fs (%.1f t/s)  "
                "prefill=%dtok/%.2fs (%.1f t/s)  load=%.2fs",
                chosen_model, ec, ed, decode_tps,
                pc, pd, prefill_tps,
                (last_stats.get("load_duration_ns") or 0) / 1e9,
            )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error(
            f"LLM FAILED  model={chosen_model}  after={elapsed:.1f}s  "
            f"error={type(exc).__name__}: {exc}",
            exc_info=True,
        )
        msg = str(exc)
        if "not found" in msg or "404" in msg:
            raise RuntimeError(
                f"Model '{chosen_model}' not found in Ollama.\n"
                f"Pull it with:  ollama pull {chosen_model}\n"
                f"Or use the fast model:  sciknow ask question \"...\" --model {settings.llm_fast_model}"
            ) from exc
        raise
    finally:
        elapsed = time.monotonic() - t0
        logger.info(
            f"LLM done    model={chosen_model}  tokens≈{token_count}  "
            f"elapsed={elapsed:.1f}s"
        )


def complete(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.1,
    num_ctx: int = 16384,
    num_batch: int = 1024,
    num_predict: int | None = None,
    keep_alive: str | int | None = -1,
    format: dict | str | None = None,
    think: bool | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> str:
    """
    Non-streaming completion. Returns the full response string.

    Phase 54.6.30:
      - ``num_ctx`` default raised from 8192 → 16384 to match stream().
        Pre-fix, a script calling stream() then complete() with default
        ctx triggered an Ollama model reload because each unique
        (model, num_ctx) is a separate Ollama instance.
      - ``keep_alive`` default None → -1 (sticky). Pre-fix, calls without
        an explicit keep_alive fell back to Ollama's 5-minute TTL,
        which evicted the model between pipeline phases that take
        longer than 5 minutes (e.g. autowrite score → verify → revise).
        Callers that want to unload can still pass ``keep_alive=0``.

    Phase 54.6.31 — ``num_batch`` default 1024 (was Ollama's native
    512). See ``stream()`` docstring for rationale.
    """
    return "".join(stream(system, user, model=model, temperature=temperature,
                          num_ctx=num_ctx, num_batch=num_batch,
                          num_predict=num_predict,
                          keep_alive=keep_alive, format=format,
                          think=think, top_p=top_p, top_k=top_k))


def complete_with_status(
    system: str,
    user: str,
    label: str = "Generating",
    model: str | None = None,
    temperature: float = 0.1,
    num_ctx: int = 16384,
    num_batch: int = 1024,
    keep_alive: str | int | None = -1,
) -> str:
    """
    Like complete(), but shows a live token counter on the console while
    the LLM generates. Useful for blocking calls that would otherwise show
    no output for 30-120 seconds.

    Display: `  Generating... 847 tokens (4.2 tok/s, 32s)`

    Returns the full response string.

    Phase 54.6.30 — now accepts and forwards ``keep_alive``. Pre-fix,
    the utility wrapper silently dropped keep_alive so multi-pass
    workflows that went through this function lost model persistence.
    """
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text

    console = Console(stderr=True)
    tokens: list[str] = []
    t0 = time.monotonic()

    with Live(console=console, refresh_per_second=2, transient=True) as live:
        for tok in stream(system, user, model=model, temperature=temperature,
                          num_ctx=num_ctx, num_batch=num_batch,
                          keep_alive=keep_alive):
            tokens.append(tok)
            elapsed = time.monotonic() - t0
            tps = len(tokens) / elapsed if elapsed > 0 else 0
            live.update(Text.from_markup(
                f"  [dim]{label}... {len(tokens)} tokens ({tps:.1f} tok/s, {elapsed:.0f}s)[/dim]"
            ))

    return "".join(tokens)
