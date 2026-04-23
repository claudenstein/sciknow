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
    """Phase 54.6.274.3 — Qwen3-Reranker adapter using the HF model
    card's prescribed CausalLM + yes/no-token-probability scoring.

    ### Why not CrossEncoder?

    Earlier attempts used ``sentence_transformers.CrossEncoder``,
    which wraps the checkpoint as ``Qwen3ForSequenceClassification``
    — a head shape that doesn't exist in the Qwen3-Reranker
    checkpoint. transformers silently initialises ``score.weight``
    randomly and emits the warning "Some weights were not
    initialized... You should probably TRAIN this model on a
    down-stream task". The returned scores are essentially noise.
    We discovered this when the adapter scored "Berlin is the
    capital of Germany" higher than "Paris is the capital of France"
    for the query "what is the capital of france?".

    ### Correct approach (per the Qwen3-Reranker HF model card)

    1. Load as ``AutoModelForCausalLM`` with left-padding.
    2. Format each (query, doc) as::

           <Instruct>: {instruction}
           <Query>: {query}
           <Document>: {doc}

       (the model was trained with this exact template)
    3. Append the fixed assistant prefix that elicits a yes/no
       answer, get the logits at the final position.
    4. Extract logits for the "yes" and "no" tokens and compute
       ``softmax([no_logit, yes_logit])[1]`` — the probability the
       document is relevant.

    ### Batch handling

    Left padding + ``pad_token=eos_token`` + attention mask so the
    final-position logits correspond to the actual last content
    token (not a padding token) across a variable-length batch.

    Returns floats in [0, 1] when ``normalize=True`` (always, for
    this adapter — the yes-probability is already a probability).
    ``normalize=False`` returns the raw yes-minus-no logit.
    """

    # Fixed prefix/suffix that the HF model card uses. The CausalLM
    # is trained to emit "yes" or "no" after this exact template.
    _PREFIX_MSG = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on "
        "the Query and the Instruct provided. Note that the answer "
        "can only be \"yes\" or \"no\"."
        "<|im_end|>\n<|im_start|>user\n"
    )
    _SUFFIX_MSG = (
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    def __init__(self, model_id: str, devices=None, **_extra):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = "cuda"
        if devices is not None:
            dev = devices[0] if isinstance(devices, (list, tuple)) else devices

        # Left padding is critical — the yes/no prediction reads the
        # logits at the LAST token. Right-padding would put pad
        # tokens at the end and we'd read their garbage logits.
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
        ).to(dev).eval()
        self._device = dev

        # Precompute the yes/no token IDs — these are single tokens
        # in Qwen's vocab (confirmed by the HF card's example).
        self._yes_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._no_id = self._tokenizer.convert_tokens_to_ids("no")

        # Pre-tokenise the boilerplate prefix/suffix once; these
        # don't change across calls, so repeated predict() saves
        # the tokenisation cost.
        self._prefix_ids = self._tokenizer.encode(
            self._PREFIX_MSG, add_special_tokens=False,
        )
        self._suffix_ids = self._tokenizer.encode(
            self._SUFFIX_MSG, add_special_tokens=False,
        )

    def _format_user(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {_QWEN3_RERANK_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def compute_score(self, pairs, normalize: bool = True, batch_size: int = 8):
        import torch

        all_scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            user_texts = [self._format_user(q, d) for q, d in batch]

            # Tokenise user chunk, then wrap with fixed prefix/suffix
            # ids. Max length 8192 is the Qwen3-Reranker context.
            user_enc = self._tokenizer(
                user_texts, add_special_tokens=False, padding=False,
                truncation=True, max_length=8192 - len(self._prefix_ids) - len(self._suffix_ids),
            )["input_ids"]

            full_ids = [
                self._prefix_ids + ut + self._suffix_ids for ut in user_enc
            ]

            # Left-pad to the longest sequence in this batch so the
            # last token (= the position we read yes/no logits at)
            # lines up across samples.
            maxlen = max(len(x) for x in full_ids)
            padded = []
            attn = []
            for ids in full_ids:
                pad = maxlen - len(ids)
                padded.append([self._tokenizer.pad_token_id] * pad + ids)
                attn.append([0] * pad + [1] * len(ids))
            input_ids = torch.tensor(padded, device=self._device)
            attention_mask = torch.tensor(attn, device=self._device)

            with torch.no_grad():
                out = self._model(
                    input_ids=input_ids, attention_mask=attention_mask,
                )
                logits = out.logits[:, -1, :]  # [B, vocab]
                yes_logit = logits[:, self._yes_id]
                no_logit = logits[:, self._no_id]
                if normalize:
                    # Softmax over just [no, yes] so score ∈ [0, 1]
                    stacked = torch.stack([no_logit, yes_logit], dim=-1)
                    probs = torch.softmax(stacked, dim=-1)
                    scores = probs[:, 1]  # P(yes | …)
                else:
                    scores = yes_logit - no_logit
                all_scores.extend(scores.float().cpu().tolist())
        return all_scores


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
