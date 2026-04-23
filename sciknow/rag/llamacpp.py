"""Phase 54.6.304 — llama.cpp `llama-server` backend for the book-writer
hot path.

Mirrors `sciknow.rag.llm.stream` / `complete` so callers can swap
backends with zero call-site changes; the dispatch happens inside
`rag.llm.stream` based on `settings.llamacpp_book_writer_enabled`
+ the model name match.

Why a side process and not in-proc bindings: keeps the existing
Ollama path untouched, isolates VRAM accounting (the server owns
the GPU residency for whichever GGUF it loaded at startup), and
matches the throughput numbers people publish for `llama-server`
specifically (the published @Punch_Taylor 4090 bench used
exactly this binary).

The user is expected to start the server out-of-band — see
`docs/LLAMACPP_BOOK_WRITER.md` for the recipe. The Python side
just speaks OpenAI-compatible HTTP at `LLAMACPP_BASE_URL`
(default ``http://localhost:8080``).

Limitations vs the Ollama backend:
  - ``format=`` (JSON-schema structured output) is **not** wired.
    The book-writer hot path is plain prose; if a JSON-mode call
    ever lands here, it raises NotImplementedError so the user
    notices instead of getting silently malformed output.
  - ``keep_alive`` is N/A — llama-server holds the model resident
    for its entire lifetime; there is no eviction TTL to fight.
  - ``think`` is honored via OpenAI's ``chat_template_kwargs``
    (Qwen3.6 chat template reads ``thinking_budget``); when
    callers pass ``think=False`` we set budget=0 so the template
    skips the ``<think>...</think>`` block.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

from sciknow.config import settings

logger = logging.getLogger("sciknow.llamacpp")


def _base_url() -> str:
    url = (settings.llamacpp_base_url or "http://localhost:8080").rstrip("/")
    return url


def is_running(timeout: float = 0.5) -> bool:
    """True iff llama-server responds to /health within `timeout` seconds.

    Cheap check — used by the dispatcher in rag.llm.stream so we can
    fall back to Ollama gracefully when the side-process isn't up
    instead of failing the user's first autowrite call with a
    cryptic connection error.
    """
    try:
        req = urllib.request.Request(f"{_base_url()}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def stream(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.2,
    num_ctx: int = 16384,           # ignored — server-fixed at startup
    num_batch: int = 1024,          # ignored — server-fixed at startup
    num_predict: int | None = None,
    keep_alive: str | int | None = -1,  # ignored — N/A on llama-server
    format: dict | str | None = None,
    think: bool | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> Iterator[str]:
    """Stream tokens from llama-server's OpenAI-compatible endpoint.

    Signature is intentionally identical to ``rag.llm.stream`` so the
    dispatcher in rag.llm can route here without rewriting kwargs.
    """
    if format is not None:
        raise NotImplementedError(
            "llama.cpp backend does not implement format=json yet — "
            "the book-writer hot path is prose-only. Route JSON calls "
            "through Ollama (the dispatcher does this automatically when "
            "format != None)."
        )

    chosen_model = model or settings.llamacpp_model_alias or settings.llm_model
    payload: dict = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "stream": True,
    }
    if num_predict is not None:
        payload["max_tokens"] = num_predict
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    # Qwen3.6 chat template: the reliable no-think switch is
    # `enable_thinking=false`. Empirically (Phase 54.6.304 bench),
    # `thinking_budget=0` does NOT suppress the <think> block on the
    # Qwen3.6-27B unsloth GGUF — the model still emits reasoning tokens
    # that land in `reasoning_content`, leaving `content` empty. Using
    # enable_thinking matches Ollama's `think=False` semantics exactly.
    if think is False:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif think is True:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{_base_url()}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    t0 = time.monotonic()
    logger.info(
        f"LLM stream  backend=llamacpp  model={chosen_model}  "
        f"system={len(system)}c  user={len(user)}c  temp={temperature}"
    )
    token_count = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except Exception:
                    continue
                for choice in evt.get("choices", []) or []:
                    delta = choice.get("delta", {}) or {}
                    piece = delta.get("content") or ""
                    # Some llama-server builds also emit a "reasoning_content"
                    # field for thinking-mode models — surface it too so we
                    # don't silently drop output if think=True someday.
                    reasoning = delta.get("reasoning_content") or ""
                    full = piece + reasoning
                    if full:
                        token_count += 1
                        yield full
    except urllib.error.HTTPError as exc:
        body_msg = exc.read().decode("utf-8", errors="ignore")[:500]
        elapsed = time.monotonic() - t0
        logger.error(
            f"LLM FAILED  backend=llamacpp  model={chosen_model}  "
            f"after={elapsed:.1f}s  http={exc.code}  body={body_msg!r}"
        )
        raise RuntimeError(
            f"llama-server returned HTTP {exc.code}: {body_msg}"
        ) from exc
    except urllib.error.URLError as exc:
        elapsed = time.monotonic() - t0
        logger.error(
            f"LLM FAILED  backend=llamacpp  model={chosen_model}  "
            f"after={elapsed:.1f}s  url_error={exc!r}"
        )
        raise RuntimeError(
            f"Could not reach llama-server at {_base_url()}: {exc}.\n"
            f"Start it with: scripts/llama_server_book_writer.sh"
        ) from exc
    finally:
        elapsed = time.monotonic() - t0
        logger.info(
            f"LLM done    backend=llamacpp  model={chosen_model}  "
            f"tokens≈{token_count}  elapsed={elapsed:.1f}s"
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
    """Non-streaming completion — drains stream() into a single string."""
    return "".join(stream(
        system, user, model=model, temperature=temperature,
        num_ctx=num_ctx, num_batch=num_batch, num_predict=num_predict,
        keep_alive=keep_alive, format=format,
        think=think, top_p=top_p, top_k=top_k,
    ))
