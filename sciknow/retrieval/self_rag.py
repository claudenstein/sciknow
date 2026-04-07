"""
Self-correcting RAG — CRAG / Self-RAG pattern.

Wraps the standard retrieve → generate pipeline with two correction steps:

1. **Retrieval evaluation**: after retrieving chunks, a fast LLM check assesses
   whether the chunks are actually relevant. If relevance is low, the query is
   reformulated and retrieval is retried (max 1 retry).

2. **Answer grounding**: after generation, checks whether the answer is supported
   by the retrieved chunks. Returns a grounding score.

Based on: Self-RAG (Asai et al., ICLR 2024), CRAG framework,
Agentic RAG Survey (arxiv 2501.09136).
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger("sciknow.self_rag")


def evaluate_retrieval(
    query: str,
    results: list,
    *,
    model: str | None = None,
    threshold: float = 0.5,
) -> tuple[bool, float, str]:
    """
    Ask the LLM whether the retrieved passages are relevant to the query.

    Returns (is_relevant, score, suggested_reformulation).
    - is_relevant: True if score >= threshold
    - score: 0.0-1.0 relevance assessment
    - suggested_reformulation: alternative query if score is low
    """
    from sciknow.rag.llm import complete as llm_complete
    from sciknow.rag.prompts import format_context

    if not results:
        return False, 0.0, query

    # Use only first 3 chunks for evaluation (fast)
    context_preview = format_context(results[:3], max_chars=4000)

    system = (
        "You are a retrieval quality evaluator. Given a question and retrieved passages, "
        "assess whether the passages contain information relevant to answering the question.\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"relevance_score": 0.0-1.0, "relevant": true/false, '
        '"reason": "brief explanation", '
        '"reformulated_query": "better query if relevance is low"}'
    )
    user = f"Question: {query}\n\nRetrieved passages:\n{context_preview}\n\nEvaluate relevance."

    try:
        raw = llm_complete(system, user, model=model, temperature=0.0, num_ctx=8192)
        # Strip thinking blocks
        raw = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        first = raw.find("{")
        last = raw.rfind("}")
        if first >= 0 and last > first:
            raw = raw[first:last + 1]

        data = json.loads(raw, strict=False)
        score = float(data.get("relevance_score", 0.5))
        relevant = data.get("relevant", score >= threshold)
        reformulation = data.get("reformulated_query", query)

        return bool(relevant), score, reformulation

    except Exception as exc:
        logger.warning("Retrieval evaluation failed: %s — assuming relevant", exc)
        return True, 0.7, query


def check_grounding(
    answer: str,
    results: list,
    *,
    model: str | None = None,
) -> tuple[float, list[str]]:
    """
    Check whether the generated answer is grounded in the retrieved passages.

    Returns (grounding_score, list_of_ungrounded_claims).
    """
    from sciknow.rag.llm import complete as llm_complete
    from sciknow.rag.prompts import format_context

    if not results or not answer.strip():
        return 0.0, []

    context = format_context(results, max_chars=8000)

    system = (
        "You are a groundedness checker. Given an answer and the source passages it was "
        "based on, check whether each claim in the answer is supported by the passages.\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"grounding_score": 0.0-1.0, '
        '"ungrounded_claims": ["claim text that lacks support", ...]}'
    )
    user = f"Answer:\n{answer[:4000]}\n\nSource passages:\n{context}\n\nCheck grounding."

    try:
        raw = llm_complete(system, user, model=model, temperature=0.0, num_ctx=8192)
        raw = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        first = raw.find("{")
        last = raw.rfind("}")
        if first >= 0 and last > first:
            raw = raw[first:last + 1]

        data = json.loads(raw, strict=False)
        score = float(data.get("grounding_score", 0.5))
        ungrounded = data.get("ungrounded_claims", [])

        return score, ungrounded

    except Exception as exc:
        logger.warning("Grounding check failed: %s", exc)
        return 0.5, []
