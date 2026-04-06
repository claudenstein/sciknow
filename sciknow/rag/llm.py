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


def stream(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.2,
    num_ctx: int = 16384,
) -> Iterator[str]:
    """
    Stream LLM response tokens. Yields string fragments as they arrive.
    """
    client = _get_client()
    chosen_model = model or settings.llm_model
    t0 = time.monotonic()
    logger.info(
        f"LLM stream  model={chosen_model}  system={len(system)}c  "
        f"user={len(user)}c  temp={temperature}  ctx={num_ctx}"
    )
    token_count = 0
    try:
        response = client.chat(
            model=chosen_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            stream=True,
            options={"temperature": temperature, "num_ctx": num_ctx},
        )
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
) -> str:
    """
    Non-streaming completion. Returns the full response string.
    """
    return "".join(stream(system, user, model=model, temperature=temperature, num_ctx=num_ctx))
