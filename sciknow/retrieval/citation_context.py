"""Phase 54.6.113 (Tier 2 #1) — citation-context embeddings for expand.

Cohan et al. 2024 (SciRepEval) finding: a paper's abstract is often
"marketing prose" that doesn't discriminate it from topically-similar
papers. The short sentences that OTHER papers write when citing it —
Semantic Scholar's ``contexts`` field — are the community's own
description of how the paper is used, and they're much more
discriminative for relevance.

This module plugs that signal into the Phase 49 RRF ranker. Per
candidate:

    1. Fetch S2 citations (already done in Phase 49 for the
       ``isInfluential`` + ``intents`` fields — the contexts come
       in the same response, we just weren't using them).
    2. Flatten + dedupe up to N contexts (default 20).
    3. Concatenate with a separator and embed with the same bge-m3
       we already use for the `bge_m3_cosine` signal.
    4. Cosine against the corpus centroid.

Added to the RRF fusion as an 8th signal — it doesn't replace
``bge_m3_cosine`` (which uses title+abstract) because the two are
complementary: papers with few citations have no contexts, and
papers with bad abstracts have no abstract signal.

See ``docs/research/EXPAND_ENRICH_RESEARCH_2.md`` §2.1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


# Sane caps for bge-m3 and for human / API patience. 20 contexts is
# Cohan's tested sweet spot and fits comfortably in bge-m3's 8k token
# window even for long contexts.
DEFAULT_MAX_CONTEXTS: int = 20
# Short contexts are usually just "[1]" or "(Smith 2020)" markers; drop.
MIN_CONTEXT_LEN: int = 20


@dataclass(frozen=True)
class ContextBundle:
    """The citation-context embedding + provenance for one candidate."""
    n_contexts: int
    embedding: np.ndarray | None   # bge-m3 dense vec, None when no contexts
    first_preview: str = ""        # first usable context, truncated, for TSV


def extract_contexts(
    s2_citations: list[dict] | None,
    *,
    max_contexts: int = DEFAULT_MAX_CONTEXTS,
    min_len: int = MIN_CONTEXT_LEN,
) -> list[str]:
    """Flatten + dedupe S2 citation contexts for one candidate.

    Each S2 edge is ``{isInfluential, intents, contexts, citingPaper}``;
    ``contexts`` is a list of ~50-char excerpts where the citing paper
    mentioned the candidate. Returns up to ``max_contexts`` unique,
    non-trivial strings.
    """
    if not s2_citations:
        return []
    out: list[str] = []
    seen: set[str] = set()
    # Prefer contexts from influential citations first — they're more
    # likely to describe the paper's actual contribution than drive-by
    # background mentions. Then append the rest.
    def _yield(edges: Iterable[dict]) -> Iterable[str]:
        for edge in edges:
            for c in (edge.get("contexts") or []):
                if isinstance(c, str):
                    yield c

    inf_edges = [e for e in s2_citations if e.get("isInfluential")]
    other_edges = [e for e in s2_citations if not e.get("isInfluential")]

    for c in list(_yield(inf_edges)) + list(_yield(other_edges)):
        c = c.strip()
        if len(c) < min_len:
            continue
        key = c.lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_contexts:
            break
    return out


def embed_contexts(contexts: list[str]) -> np.ndarray | None:
    """Single bge-m3 dense encode over the joined contexts.

    We concatenate rather than encode each context separately and
    average: the joined text is usually 300-1200 tokens, well under
    bge-m3's 8k-token window, and one encode is cheaper than N. The
    separator mirrors bge-m3's SEP token so attention pools cleanly
    across contexts.
    """
    if not contexts:
        return None
    # Light dedup on near-identical prefixes (S2 often ships the same
    # sentence from two printings of the same paper).
    joined = " [SEP] ".join(contexts)[:4000]
    try:
        from sciknow.ingestion.embedder import _get_model
        model = _get_model()
    except Exception as exc:  # noqa: BLE001
        logger.debug("citation_context: embedder unavailable: %s", exc)
        return None
    try:
        out = model.encode(
            [joined],
            batch_size=1,
            max_length=4096,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("citation_context embed failed: %s", exc)
        return None
    vec = out.get("dense_vecs")
    if vec is None or len(vec) == 0:
        return None
    return np.asarray(vec[0], dtype=np.float32)


def context_cosine(vec: np.ndarray | None, anchor: np.ndarray | None) -> float:
    """Cosine similarity, safe on None / zero-norm inputs."""
    if vec is None or anchor is None:
        return 0.0
    n1 = float(np.linalg.norm(vec))
    n2 = float(np.linalg.norm(anchor))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(vec, anchor) / (n1 * n2))


def build_bundle(
    s2_citations: list[dict] | None,
    *,
    max_contexts: int = DEFAULT_MAX_CONTEXTS,
) -> ContextBundle:
    """Convenience: extract → embed → wrap in a ContextBundle.

    Returns an empty bundle with `embedding=None` when there are no
    usable contexts (very new / uncited papers).
    """
    ctxs = extract_contexts(s2_citations, max_contexts=max_contexts)
    emb = embed_contexts(ctxs) if ctxs else None
    preview = (ctxs[0][:140] if ctxs else "")
    return ContextBundle(
        n_contexts=len(ctxs),
        embedding=emb,
        first_preview=preview,
    )
