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


def stream(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.2,
    num_ctx: int = 16384,
    keep_alive: str | int | None = None,
    format: dict | str | None = None,
) -> Iterator[str]:
    """
    Stream LLM response tokens. Yields string fragments as they arrive.

    keep_alive: Ollama keep_alive parameter ("-1" = infinite, "60m", etc.).
                Prevents model unload between calls, enabling prompt caching.
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
        kwargs: dict = {
            "model": chosen_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": True,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive
        if format is not None:
            kwargs["format"] = format
        response = client.chat(**kwargs)
        for chunk in response:
            token_count += 1
            yield chunk.message.content
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
    num_ctx: int = 8192,
    keep_alive: str | int | None = None,
    format: dict | str | None = None,
) -> str:
    """
    Non-streaming completion. Returns the full response string.
    """
    return "".join(stream(system, user, model=model, temperature=temperature,
                          num_ctx=num_ctx, keep_alive=keep_alive, format=format))


def complete_with_status(
    system: str,
    user: str,
    label: str = "Generating",
    model: str | None = None,
    temperature: float = 0.1,
    num_ctx: int = 8192,
) -> str:
    """
    Like complete(), but shows a live token counter on the console while
    the LLM generates. Useful for blocking calls that would otherwise show
    no output for 30-120 seconds.

    Display: `  Generating... 847 tokens (4.2 tok/s, 32s)`

    Returns the full response string.
    """
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text

    console = Console(stderr=True)
    tokens: list[str] = []
    t0 = time.monotonic()

    with Live(console=console, refresh_per_second=2, transient=True) as live:
        for tok in stream(system, user, model=model, temperature=temperature, num_ctx=num_ctx):
            tokens.append(tok)
            elapsed = time.monotonic() - t0
            tps = len(tokens) / elapsed if elapsed > 0 else 0
            live.update(Text.from_markup(
                f"  [dim]{label}... {len(tokens)} tokens ({tps:.1f} tok/s, {elapsed:.0f}s)[/dim]"
            ))

    return "".join(tokens)
