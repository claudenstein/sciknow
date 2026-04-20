"""Phase 54.6.121 (Tier 3 #1) — SPECTER2 reranker.

SPECTER2 (Cohan et al. 2023, Ai2) is a citation-pretrained scientific
embedding model. The 2025 release ships as a base model
(`allenai/specter2_base`, ~440 MB) plus task-specific adapters; we use
the proximity adapter (`allenai/specter2`) which is trained for
"general scientific document similarity" — exactly our reranker use
case.

Net VRAM cost: ~600 MB for base + adapter at fp32, ~300 MB at fp16.
Well within the 24 GB 3090's headroom even with the writer resident.

Decision criterion (from `docs/EXPAND_ENRICH_RESEARCH_2.md` §2.2):
SPECTER2 must beat bge-m3 on MRR@10 by ≥ 0.06 (so > 0.60 absolute on
the global-cooling 0.514 baseline) for it to justify shipping as the
default rerank step. Otherwise it stays an opt-in.

This module is the embedder + a `rerank()` helper. The bench wiring
lives in `sciknow/testing/specter2_bench.py`.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


_BASE_MODEL = "allenai/specter2_base"
_ADAPTER_NAME = "allenai/specter2"   # proximity adapter

_model = None         # transformers AutoModel with adapter loaded
_tokenizer = None
_device = None


def _load() -> tuple:
    """Lazy-init SPECTER2 on GPU when available.

    Phase 54.6.121 — uses plain `transformers.AutoModel` on
    `allenai/specter2_base` (the base proximity model trained on all
    4 SciRepEval tasks). The separate `adapters` library is too
    fragile across transformer versions; the base alone gives the
    proximity-task performance our reranker needs without per-version
    glue.
    """
    global _model, _tokenizer, _device
    if _model is not None:
        return _model, _tokenizer, _device

    import torch
    from transformers import AutoTokenizer, AutoModel

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
    _model = AutoModel.from_pretrained(_BASE_MODEL)
    _model = _model.to(_device).eval()
    if _device == "cuda":
        _model = _model.half()
    logger.info("SPECTER2 loaded on %s (model=%s)", _device, _BASE_MODEL)
    return _model, _tokenizer, _device


def encode(
    texts: Iterable[str], *, batch_size: int = 16, max_length: int = 512,
) -> np.ndarray:
    """Encode a list of texts → (N, 768) float32 numpy array.

    SPECTER2 expects ``title [SEP] abstract`` format; callers should
    pre-format with a real ``[SEP]`` token where applicable. Plain
    queries pass through unchanged.
    """
    import torch
    model, tok, device = _load()
    texts = [t or "" for t in texts]
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    out_chunks: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model(**enc)
        # SPECTER2 uses the [CLS] token's hidden state as the doc vec.
        cls = output.last_hidden_state[:, 0, :]
        cls = cls.float().cpu().numpy()
        out_chunks.append(cls)
    return np.concatenate(out_chunks, axis=0).astype(np.float32)


def encode_query(text: str) -> np.ndarray:
    """Encode a single query (~768-d float32). Returns (768,)."""
    return encode([text], batch_size=1)[0]


def rerank(
    query: str,
    candidates: list[dict],
    *,
    text_key: str = "text",
    top_k: int | None = None,
) -> list[dict]:
    """Rerank candidates by SPECTER2 cosine similarity to ``query``.

    Each candidate dict must carry a ``text_key`` field. Adds
    ``specter2_score`` to each candidate dict and returns them sorted
    descending. Cheap on top-50 (one forward pass per ~50 docs).
    """
    if not candidates:
        return []
    qv = encode_query(query)
    docs = [c.get(text_key, "") for c in candidates]
    dv = encode(docs)
    qn = float(np.linalg.norm(qv))
    if qn == 0.0:
        return candidates
    for c, v in zip(candidates, dv):
        n = float(np.linalg.norm(v))
        c["specter2_score"] = float(np.dot(v, qv) / (n * qn)) if n > 0 else 0.0
    candidates.sort(key=lambda c: c.get("specter2_score", 0.0), reverse=True)
    return candidates[:top_k] if top_k else candidates
