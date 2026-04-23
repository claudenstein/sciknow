"""
Cross-encoder reranker, dispatched on the model tag.

Loaded lazily on first use and kept in memory. Two backends:

  * ``BAAI/bge-reranker-*`` (and any bge-reranker family): plain
    cross-encoder via ``FlagEmbedding.FlagReranker`` — the pre-
    54.6.274 default. Scores are normalised 0-1.
  * ``Qwen/Qwen3-Reranker-*`` (Phase 54.6.274): generative reranker
    via ``sentence_transformers.CrossEncoder``. Needs an instruction
    prefix (scientific-literature-tuned below) for best quality.
    Sigmoid activation normalises the logit output to 0-1 so the
    downstream RRF code doesn't need to know which backend ran.

Dispatch is by simple prefix match on ``settings.reranker_model``.
Adding a new backend: add another `elif model_id.startswith(...)`
branch inside ``_get_reranker`` and make it expose ``.compute_score(
pairs)``. The returned scalar must be comparable across calls (higher
= more relevant); RRF downstream does not need the absolute scale.
"""
from __future__ import annotations

from sciknow.retrieval.hybrid_search import SearchCandidate

_reranker = None


# Phase 54.6.274 — scientific-literature instruction for Qwen3-Reranker.
# The model card notes 1-5% accuracy loss without an instruction; this
# one is drawn from the Qwen3-Reranker default but retopicalised for
# our corpus (papers, not general web search).
_QWEN3_RERANK_INSTRUCTION = (
    "Given a scientific literature search query, retrieve relevant "
    "passages that answer the query"
)


class _Qwen3RerankerAdapter:
    """Phase 54.6.274 — thin adapter wrapping sentence_transformers
    ``CrossEncoder`` so the Qwen3-Reranker API matches FlagReranker's
    ``.compute_score(pairs, normalize=True)`` contract used by the
    existing ``rerank()`` call site.

    Applies the prescribed instruction prompt + sigmoid activation
    internally; caller passes plain (query, doc) pairs as before.
    """

    def __init__(self, model_id: str, devices=None, **_extra):
        """``devices`` kwarg matches the FlagReranker / sciknow
        ``load_with_cpu_fallback`` contract — it can be ``"cpu"``,
        ``"cuda"``, ``"cuda:0"``, or a list. sentence-transformers
        ``CrossEncoder`` expects a single ``device`` string, so
        normalise the first-of-list case here."""
        from sentence_transformers import CrossEncoder
        kwargs: dict = {}
        if devices is not None:
            dev = devices[0] if isinstance(devices, (list, tuple)) else devices
            kwargs["device"] = dev
        # The Qwen3-Reranker model card exposes `default_prompt_name`
        # via the `prompts` dict — see HF card "Sentence-Transformers"
        # section. This is what folds the instruction into the input
        # at predict time without the caller constructing it each call.
        self._model = CrossEncoder(
            model_id,
            prompts={"retrieval": _QWEN3_RERANK_INSTRUCTION},
            default_prompt_name="retrieval",
            **kwargs,
        )

    def compute_score(self, pairs, normalize: bool = True):
        import torch
        # sentence-transformers CrossEncoder.predict returns numpy
        # array of floats. `normalize=True` maps to sigmoid on the
        # raw logit output so scores fall in [0, 1] (FlagReranker
        # parity). `normalize=False` returns raw logits — we still
        # expose it for symmetry.
        act = torch.nn.Sigmoid() if normalize else None
        out = self._model.predict(
            pairs, activation_fn=act,
        )
        try:
            return out.tolist()
        except AttributeError:
            return list(out)


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sciknow.config import settings
        from sciknow.retrieval.device import load_with_cpu_fallback

        model_id = settings.reranker_model or ""

        # Phase 54.6.274 — dispatch by prefix. Qwen3-Reranker family
        # lives on sentence-transformers/transformers; BGE rerankers
        # stay on FlagEmbedding (fastest path for them).
        if model_id.startswith("Qwen/Qwen3-Reranker"):
            _reranker = load_with_cpu_fallback(
                _Qwen3RerankerAdapter, model_id,
            )
        else:
            from FlagEmbedding import FlagReranker
            # Phase 15.2 — CPU fallback when the GPU is full of LLM weights.
            _reranker = load_with_cpu_fallback(
                FlagReranker, model_id, use_fp16=True,
            )
    return _reranker


def release_reranker() -> None:
    """Drop the cached reranker model and free VRAM."""
    global _reranker
    if _reranker is None:
        return
    try:
        del _reranker
    finally:
        _reranker = None
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def rerank(
    query: str,
    candidates: list[SearchCandidate],
    top_k: int = 10,
) -> list[SearchCandidate]:
    """
    Score each (query, chunk) pair with the cross-encoder and return top_k results
    sorted by descending reranker score.

    The reranker score is stored back into candidate.rrf_score so callers can
    display it without knowing which scoring stage produced it.
    """
    if not candidates:
        return []

    reranker = _get_reranker()

    pairs = [[query, c.content_preview] for c in candidates]
    scores: list[float] = reranker.compute_score(pairs, normalize=True)

    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    result = []
    for score, candidate in scored[:top_k]:
        candidate.rrf_score = float(score)
        result.append(candidate)
    return result
