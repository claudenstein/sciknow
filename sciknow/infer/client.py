"""HTTP client for llama-server (OpenAI-compatible).

One client module that speaks to all three roles by URL. Functions here
mirror the v1 ``rag/llm.py`` interface so consumers (book_ops,
wiki_ops, ask, etc.) need only swap import paths in Phase B+.

Per-call stats (decode/prefill timings, eval_count) are captured on a
thread-local list and drained by ``drain_call_stats()`` for the same
pulse pipeline the v1 code feeds.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx

from sciknow.config import settings

logger = logging.getLogger("sciknow.infer.client")

# ── per-thread stats side-channel (parity with v1 rag/llm.py) ────────────
_call_stats_local = threading.local()


def _record_call_stats(stats: dict) -> None:
    if not hasattr(_call_stats_local, "calls"):
        _call_stats_local.calls = []
    _call_stats_local.calls.append(stats)


def drain_call_stats() -> list[dict]:
    """Pop and return all stats accumulated on this thread since last drain."""
    calls = getattr(_call_stats_local, "calls", [])
    _call_stats_local.calls = []
    return calls


# ── shared HTTPX client per role (connection pool reuse) ─────────────────
_clients: dict[str, httpx.Client] = {}
_clients_lock = threading.Lock()


def _client_for(role: str) -> httpx.Client:
    url = {
        "writer": settings.infer_writer_url,
        "embedder": settings.infer_embedder_url,
        "reranker": settings.infer_reranker_url,
    }[role]
    with _clients_lock:
        c = _clients.get(role)
        if c is None or c.is_closed:
            c = httpx.Client(base_url=url.rstrip("/"), timeout=httpx.Timeout(
                connect=10.0, read=600.0, write=60.0, pool=10.0,
            ))
            _clients[role] = c
        return c


# ── chat / completion ────────────────────────────────────────────────────


def chat_stream(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.2,
    num_ctx: int = 16384,                # accepted for v1 parity, ignored server-side
    num_batch: int = 1024,               # accepted for v1 parity, ignored server-side
    num_predict: int | None = None,
    keep_alive: Any = None,              # accepted for v1 parity, ignored
    format: dict | str | None = None,
    think: bool | None = None,           # accepted for v1 parity, ignored
    top_p: float | None = None,
    top_k: int | None = None,
) -> Iterator[str]:
    """Stream chat completion tokens from the writer server.

    Translates the v1 ``rag/llm.py:stream`` signature to llama-server's
    OpenAI-compatible ``/v1/chat/completions`` (SSE). Yields content
    fragments as they arrive. On the final chunk, captures llama-server's
    ``timings`` payload and stashes it via ``_record_call_stats`` so the
    web pulse pipeline keeps working.

    Args mirrored from v1 — extra Ollama-specific args are accepted but
    silently ignored where llama-server doesn't expose them.
    """
    chosen_model = model or settings.writer_model_name
    client = _client_for("writer")

    body: dict[str, Any] = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "temperature": temperature,
        "cache_prompt": True,
        # Ask llama-server to attach timings on the final chunk.
        "timings_per_token": False,
        # Qwen3.x is a hybrid thinking model. We never want CoT in the
        # writer's output stream — the writer's job is to emit prose
        # the way an autowrite/wiki/ask flow can persist directly. The
        # think=True opt-in below flips this on for callers that want
        # it (verifier prompts may benefit).
        "chat_template_kwargs": {"enable_thinking": bool(think) if think is not None else False},
    }
    if num_predict is not None:
        body["max_tokens"] = num_predict
    if top_p is not None:
        body["top_p"] = top_p
    if top_k is not None:
        body["top_k"] = top_k
    if format is not None:
        # llama-server accepts response_format like OpenAI.
        if isinstance(format, dict):
            body["response_format"] = {"type": "json_object", "schema": format}
        elif format == "json":
            body["response_format"] = {"type": "json_object"}

    t0 = time.monotonic()
    token_count = 0
    last_timings: dict | None = None
    logger.info(
        "INFER stream model=%s system=%dc user=%dc temp=%s",
        chosen_model, len(system), len(user), temperature,
    )
    try:
        with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                body_txt = resp.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"llama-server /v1/chat/completions status="
                    f"{resp.status_code}: {body_txt[:500]}"
                )
            for line in resp.iter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # Capture timings if llama-server provided them on this chunk.
                if "timings" in evt:
                    last_timings = evt["timings"]
                if "usage" in evt and evt["usage"]:
                    last_timings = (last_timings or {}) | {"usage": evt["usage"]}
                choices = evt.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    token_count += 1
                    yield content
                # Some models (Qwen3.x) emit reasoning_content even when
                # thinking is disabled — surface it only if think=True
                # was explicitly requested. Otherwise drop it.
                if think:
                    rc = delta.get("reasoning_content")
                    if rc:
                        token_count += 1
                        yield rc
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error("INFER FAILED model=%s after=%.1fs error=%s: %s",
                     chosen_model, elapsed, type(exc).__name__, exc)
        raise
    finally:
        elapsed = time.monotonic() - t0
        if last_timings:
            stats = _coerce_timings_to_v1(chosen_model, last_timings)
            if stats.get("eval_count"):
                _record_call_stats(stats)
                ed = stats["eval_duration_ns"] / 1e9 if stats.get("eval_duration_ns") else 0
                ec = stats.get("eval_count") or 0
                pd = stats["prompt_eval_duration_ns"] / 1e9 if stats.get("prompt_eval_duration_ns") else 0
                pc = stats.get("prompt_eval_count") or 0
                decode_tps = (ec / ed) if ed > 0 else 0.0
                prefill_tps = (pc / pd) if pd > 0 else 0.0
                logger.info(
                    "INFER perf model=%s decode=%dtok/%.2fs (%.1f t/s) "
                    "prefill=%dtok/%.2fs (%.1f t/s)",
                    chosen_model, ec, ed, decode_tps, pc, pd, prefill_tps,
                )
        logger.info("INFER done model=%s tokens≈%d elapsed=%.1fs",
                    chosen_model, token_count, elapsed)


def _coerce_timings_to_v1(model: str, t: dict) -> dict:
    """Map llama-server's ``timings`` block to the v1 Ollama-shaped
    stats dict that the dashboard already understands.

    llama-server emits durations in **milliseconds**; we convert to
    nanoseconds for parity with the v1 wire format.
    """
    usage = t.get("usage") or {}
    prompt_n = int(t.get("prompt_n") or usage.get("prompt_tokens") or 0)
    predicted_n = int(t.get("predicted_n") or usage.get("completion_tokens") or 0)
    prompt_ms = float(t.get("prompt_ms") or 0.0)
    predicted_ms = float(t.get("predicted_ms") or 0.0)
    return {
        "model": model,
        "eval_count": predicted_n,
        "eval_duration_ns": int(predicted_ms * 1e6),
        "prompt_eval_count": prompt_n,
        "prompt_eval_duration_ns": int(prompt_ms * 1e6),
        "load_duration_ns": 0,
        "total_duration_ns": int((prompt_ms + predicted_ms) * 1e6),
    }


def chat_complete(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.1,
    num_ctx: int = 16384,
    num_batch: int = 1024,
    num_predict: int | None = None,
    keep_alive: Any = None,
    format: dict | str | None = None,
    think: bool | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> str:
    """Non-streaming completion. Concatenates chat_stream output."""
    return "".join(chat_stream(
        system, user, model=model, temperature=temperature,
        num_ctx=num_ctx, num_batch=num_batch, num_predict=num_predict,
        keep_alive=keep_alive, format=format, think=think,
        top_p=top_p, top_k=top_k,
    ))


def warm_up(model: str | None = None, num_ctx: int = 16384, num_batch: int = 1024) -> bool:
    """Pre-load the writer model with a 1-token generate.

    llama-server keeps models loaded for the process lifetime, so this
    only matters on the very first call after `infer up`. Best-effort,
    never raises.
    """
    try:
        for _ in chat_stream(
            "You are a helpful assistant.", "ok",
            model=model, temperature=0.0, num_predict=1,
        ):
            break
        return True
    except Exception as exc:
        logger.warning("INFER warm_up failed: %s", exc)
        return False


# ── embeddings ───────────────────────────────────────────────────────────


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Batch dense-embed texts via the embedder server's /v1/embeddings."""
    if not texts:
        return []
    chosen_model = model or settings.embedder_model_name
    client = _client_for("embedder")
    body = {"model": chosen_model, "input": texts}
    r = client.post("/v1/embeddings", json=body)
    if r.status_code != 200:
        raise RuntimeError(
            f"llama-server /v1/embeddings status={r.status_code}: "
            f"{r.text[:500]}"
        )
    data = r.json()
    rows = data.get("data") or []
    # Sort by index to preserve input order (OpenAI guarantees).
    rows.sort(key=lambda e: e.get("index", 0))
    return [row.get("embedding") or [] for row in rows]


# ── rerank ───────────────────────────────────────────────────────────────


def rerank(query: str, documents: list[str], model: str | None = None) -> list[float]:
    """Cross-encode (query, doc) pairs and return per-document scores.

    Uses llama-server's /v1/rerank endpoint (added upstream 2025).
    Returns scores in input order. Higher = more relevant.
    """
    if not documents:
        return []
    chosen_model = model or settings.reranker_model_name
    client = _client_for("reranker")
    body = {"model": chosen_model, "query": query, "documents": documents}
    r = client.post("/v1/rerank", json=body)
    if r.status_code != 200:
        raise RuntimeError(
            f"llama-server /v1/rerank status={r.status_code}: {r.text[:500]}"
        )
    data = r.json()
    rows = data.get("results") or []
    rows.sort(key=lambda e: e.get("index", 0))
    # llama-server emits raw cross-encoder logits in `relevance_score`.
    # Sigmoid-normalise to [0, 1] so callers see the same contract the
    # v1 FlagReranker(normalize=True) path produced.
    import math
    out: list[float] = []
    for row in rows:
        raw = row.get("relevance_score")
        if raw is None:
            raw = row.get("score") or 0.0
        try:
            x = float(raw)
        except (TypeError, ValueError):
            x = 0.0
        # Numerically stable sigmoid (overflow-safe for large |x|).
        if x >= 0:
            ex = math.exp(-x)
            out.append(1.0 / (1.0 + ex))
        else:
            ex = math.exp(x)
            out.append(ex / (1.0 + ex))
    return out


# ── slots / health (used by observability) ───────────────────────────────


def slots(role: str = "writer") -> list[dict]:
    """Return llama-server slot snapshots for diagnostics."""
    client = _client_for(role)
    try:
        r = client.get("/slots")
        if r.status_code == 200:
            d = r.json()
            return d if isinstance(d, list) else []
    except Exception:
        pass
    return []


def health(role: str = "writer") -> bool:
    client = _client_for(role)
    try:
        r = client.get("/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False
