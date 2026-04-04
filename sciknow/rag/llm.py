"""
Thin wrapper around the Ollama Python client.

Provides streaming and non-streaming chat, reusing a single client instance.
"""
from __future__ import annotations

from collections.abc import Iterator

import ollama

from sciknow.config import settings

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
            yield chunk.message.content
    except Exception as exc:
        msg = str(exc)
        if "not found" in msg or "404" in msg:
            raise RuntimeError(
                f"Model '{chosen_model}' not found in Ollama.\n"
                f"Pull it with:  ollama pull {chosen_model}\n"
                f"Or use the fast model:  sciknow ask question \"...\" --model {settings.llm_fast_model}"
            ) from exc
        raise


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
