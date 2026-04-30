"""Phase 56.D — Per-claim retrieval engine.

For each atomic claim from 56.C, find the chunks that genuinely support
it. Pipeline:

  1. **Paraphrase** the claim into 3-5 different framings (DMQR-style,
     Yang et al. 2024).
  2. **HyDE** when the claim is short / abstract (Gao et al. 2022) —
     LLM hallucinates a plausible supporting passage; we embed and
     retrieve against it. Boosts recall when the claim's surface form
     differs from how the corpus says the same thing.
  3. **Hybrid search** per query via the existing
     ``retrieval/hybrid_search.search`` (dense + sparse + FTS, RRF-
     fused inside).
  4. **RRF fusion** across the per-paraphrase candidate pools.
  5. **MMR diversification** (Carbonell & Goldstein 1998, λ=0.65) to
     avoid intra-claim redundancy where 3 retrieval slots all point
     at the same paper.
  6. **NLI entailment scoring** of each candidate against the claim
     text. Reuses ``testing/quality.py:_nli_entail_probs``.
  7. **Entailment gate** — keep chunks with entail_prob ≥ 0.65. If
     fewer than 2 survive, the claim is **weak**: the writer (56.E)
     drops it and the coverage report (56.G) flags it.

Output: ``ClaimEvidence(claim_id, chunks, weak, weak_reason)`` per
claim. Stable shape — used by 56.E (writer) and 56.G (telemetry).
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from sciknow.core.claim_extractor import Claim

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Tuning knobs
# ──────────────────────────────────────────────────────────────────────


N_PARAPHRASES = 4
HYDE_MIN_WORDS = 12      # claims shorter than this trigger HyDE
HYBRID_K_PER_QUERY = 15  # candidates per paraphrase before fusion
RRF_K = 60               # standard RRF constant
MMR_LAMBDA = 0.65        # 1.0 = pure relevance, 0.0 = pure diversity
MMR_OUTPUT_K = 12        # candidate pool size after MMR
ENTAILMENT_FLOOR = 0.65  # NLI prob ≥ this counts as supporting
WEAK_MIN_CHUNKS = 2      # below this, claim is weak (will be dropped)


# ──────────────────────────────────────────────────────────────────────
# Output shape
# ──────────────────────────────────────────────────────────────────────


@dataclass
class EvidenceChunk:
    """One chunk that survived the retrieval + entailment gate."""
    chunk_id: str
    document_id: str
    content: str
    title: str
    doi: str | None
    year: int | None
    rrf_score: float
    nli_entail: float


@dataclass
class ClaimEvidence:
    """Output of ``retrieve_for_claim`` for one claim."""
    claim_id: str
    claim_text: str
    chunks: list[EvidenceChunk] = field(default_factory=list)
    weak: bool = False
    weak_reason: str = ""

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def summary(self) -> str:
        if self.weak:
            return f"WEAK ({self.weak_reason}) — {self.n_chunks} chunks"
        return (
            f"OK — {self.n_chunks} chunks, "
            f"top NLI {max((c.nli_entail for c in self.chunks), default=0):.2f}"
        )


# ──────────────────────────────────────────────────────────────────────
# Paraphrase + HyDE
# ──────────────────────────────────────────────────────────────────────


CLAIM_PARAPHRASE_SYSTEM = """\
You generate diverse retrieval queries for a target claim. Each
paraphrase probes a DIFFERENT facet so the retrieval covers more of
the corpus.

Emit JSON ONLY:

{
  "paraphrases": [
    "string", "string", "string", "string"
  ]
}

Rules:

1. DIVERSE FACETS, not surface rewordings. Examples for the claim
   "Solar minima coincide with surface cooling":
     ✓ "cosmogenic radionuclide records of past grand minima"
     ✓ "Maunder Minimum temperature reconstructions North Atlantic"
     ✓ "TSI variability decadal climate response"
     ✓ "11-year solar cycle and global mean temperature"
   ✗ "When solar minima happen, surface cools"  (mere wording)

2. TECHNICAL VOCABULARY — use the language a corpus paper would
   use, not the user-facing claim's language.

3. NO META-WORDS — don't include "review", "summary", "discussion".

4. JSON ONLY.
"""


HYDE_SYSTEM = """\
You write a brief hypothetical passage that COULD appear in a
research paper supporting the given claim. The passage will be
embedded and used to retrieve real corpus passages with similar
content.

Rules:
- 80-150 words.
- Technical register; named methods, datasets, time periods, units.
- Cite plausible-sounding studies inline as [N] (any number; not used).
- Output the passage text only — no preamble, no JSON.
"""


def _paraphrase_claim(
    claim: Claim,
    *,
    n: int = N_PARAPHRASES,
    model: str | None = None,
) -> list[str]:
    """Return 3-5 diverse paraphrases of the claim."""
    from sciknow.rag.llm import complete as llm_complete
    from sciknow.core.book_ops import _clean_json

    user = (
        f"Claim: {claim.text}\n"
        f"Scope: {claim.scope or '(no scope qualifier)'}\n"
        f"Target paraphrase count: {n}"
    )
    raw = llm_complete(
        CLAIM_PARAPHRASE_SYSTEM, user,
        model=model, temperature=0.4,
        num_predict=400, keep_alive=-1,
    )
    try:
        data = json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        logger.warning("paraphrase JSON parse failed, falling back to claim text only")
        return [claim.text]
    items = data.get("paraphrases") or []
    out = [str(p).strip() for p in items if str(p).strip()]
    # Always include the original claim text as one query — the writer
    # may have phrased the claim in technical terms already.
    if claim.text and claim.text not in out:
        out = [claim.text] + out
    return out[: n + 1]


def _hyde_generate(claim: Claim, *, model: str | None = None) -> str:
    """Hallucinate a plausible supporting passage for the claim."""
    from sciknow.rag.llm import complete as llm_complete
    user = f"Claim: {claim.text}\nScope: {claim.scope or '(none)'}"
    return llm_complete(
        HYDE_SYSTEM, user,
        model=model, temperature=0.5,
        num_predict=300, keep_alive=-1,
    ).strip()


def _is_short_or_abstract(claim: Claim) -> bool:
    """Heuristic for HyDE eligibility.

    Short = fewer than ``HYDE_MIN_WORDS`` words.
    Abstract = no concrete entities (no proper nouns, no numbers).
    """
    words = claim.text.split()
    if len(words) < HYDE_MIN_WORDS:
        return True
    has_proper = bool(re.search(r"\b[A-Z][a-z]+", claim.text[1:]))  # skip first cap
    has_number = bool(re.search(r"\d", claim.text))
    return not (has_proper or has_number)


# ──────────────────────────────────────────────────────────────────────
# Fusion + diversification
# ──────────────────────────────────────────────────────────────────────


def _rrf_fuse(
    pools: list[list],
    *,
    k: int = RRF_K,
    id_attr: str = "chunk_id",
) -> list[tuple]:
    """Reciprocal Rank Fusion over multiple ranked candidate lists.

    Args:
      pools: each is a ranked list of objects with ``.chunk_id``.
      k: standard RRF constant (60 is the published default).
      id_attr: attribute name to dedupe on.

    Returns: list of ``(candidate, score)`` tuples sorted by score desc.
    """
    score: dict[str, float] = {}
    pick: dict[str, object] = {}
    for pool in pools:
        for rank, c in enumerate(pool, start=1):
            cid = getattr(c, id_attr, None) or (
                c.get(id_attr) if isinstance(c, dict) else None
            )
            if not cid:
                continue
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in pick:
                pick[cid] = c
    items = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [(pick[cid], s) for cid, s in items]


def _mmr(
    candidates_with_scores: list[tuple],
    *,
    output_k: int = MMR_OUTPUT_K,
    lambda_: float = MMR_LAMBDA,
    embed_fn=None,
) -> list[tuple]:
    """Maximal Marginal Relevance reranking.

    For each step, picks the candidate that maximises::

        λ · relevance − (1 − λ) · max_sim(candidate, already_selected)

    where ``relevance`` is the candidate's existing score and ``sim``
    is cosine between candidate content embeddings.

    Args:
      candidates_with_scores: list of ``(candidate, score)`` tuples.
      output_k: stop after this many selected.
      lambda_: 1.0 = pure relevance, 0.0 = pure diversity.
      embed_fn: callable ``list[str] -> list[list[float]]`` for content
        embeddings. Defaults to ``infer.client.embed``.
    """
    if not candidates_with_scores:
        return []
    if embed_fn is None:
        from sciknow.infer.client import embed as embed_fn

    # Embed all candidate contents in one batch.
    contents: list[str] = []
    for cand, _ in candidates_with_scores:
        text = (
            getattr(cand, "content_preview", None)
            or getattr(cand, "content", None)
            or (cand.get("content_preview") if isinstance(cand, dict) else "")
            or (cand.get("content") if isinstance(cand, dict) else "")
            or ""
        )
        contents.append(text[:1500])  # cap to keep embed call cheap
    try:
        vecs = np.asarray(embed_fn(contents), dtype=np.float32)
    except Exception as exc:
        logger.warning("MMR: embedding failed (%s); falling back to relevance order", exc)
        return candidates_with_scores[:output_k]

    # Normalise so dot = cosine.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    selected: list[int] = []
    remaining = set(range(len(candidates_with_scores)))
    while remaining and len(selected) < output_k:
        best_idx, best_score = None, -math.inf
        for i in remaining:
            rel = candidates_with_scores[i][1]
            if not selected:
                score = lambda_ * rel
            else:
                # max similarity with already-selected
                sims = vecs[selected] @ vecs[i]
                max_sim = float(sims.max())
                score = lambda_ * rel - (1.0 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [candidates_with_scores[i] for i in selected]


# ──────────────────────────────────────────────────────────────────────
# Top-level entry
# ──────────────────────────────────────────────────────────────────────


def retrieve_for_claim(
    claim: Claim,
    *,
    use_hyde: bool | None = None,
    candidate_k_per_query: int = HYBRID_K_PER_QUERY,
    output_k: int = MMR_OUTPUT_K,
    entailment_floor: float = ENTAILMENT_FLOOR,
    model: str | None = None,
) -> ClaimEvidence:
    """End-to-end retrieval + filtering for one claim.

    Args:
      claim: from 56.C.
      use_hyde: force HyDE on/off. None (default) = auto by claim shape.
      candidate_k_per_query: top-K from each hybrid search call.
      output_k: candidates kept after MMR (before entailment gate).
      entailment_floor: minimum NLI entailment to keep a chunk.
      model: optional override for paraphrase / HyDE LLM calls.
    """
    from sciknow.retrieval.hybrid_search import search as hybrid_search
    from sciknow.storage.qdrant import get_client
    from sciknow.storage.db import get_session
    from sciknow.testing.quality import _nli_entail_probs

    qd = get_client()

    # 1. Build query set: paraphrases (+ original) + optional HyDE.
    queries: list[str] = _paraphrase_claim(claim, model=model)
    do_hyde = _is_short_or_abstract(claim) if use_hyde is None else use_hyde
    if do_hyde:
        try:
            queries.append(_hyde_generate(claim, model=model))
        except Exception as exc:
            logger.warning("HyDE failed for claim %s: %s", claim.claim_id, exc)

    # 2. Hybrid search per query.
    pools = []
    with get_session() as session:
        for q in queries:
            try:
                hits = hybrid_search(
                    q, qd, session, candidate_k=candidate_k_per_query,
                )
                pools.append(hits)
            except Exception as exc:
                logger.warning("hybrid_search failed on '%s...': %s",
                               q[:60], exc)

    if not pools:
        return ClaimEvidence(
            claim_id=claim.claim_id, claim_text=claim.text,
            chunks=[], weak=True,
            weak_reason="no retrieval pools succeeded",
        )

    # 3. RRF fusion (chunk_id is the dedup key — different paraphrases
    # often surface the same chunk at different ranks).
    fused = _rrf_fuse(pools, k=RRF_K, id_attr="chunk_id")
    if not fused:
        return ClaimEvidence(
            claim_id=claim.claim_id, claim_text=claim.text,
            chunks=[], weak=True, weak_reason="RRF returned no candidates",
        )

    # 4. MMR diversification.
    diversified = _mmr(fused[: candidate_k_per_query * 4],
                        output_k=output_k, lambda_=MMR_LAMBDA)

    # 5. NLI entailment scoring.
    pairs: list[tuple[str, str]] = []
    cands = []
    for cand, _score in diversified:
        text = (
            getattr(cand, "content_preview", None)
            or getattr(cand, "content", None)
            or ""
        )
        if not text:
            continue
        pairs.append((text, claim.text))
        cands.append((cand, _score, text))

    try:
        entail_probs = _nli_entail_probs(pairs)
    except Exception as exc:
        logger.warning("NLI scoring failed; keeping all candidates ungated: %s", exc)
        entail_probs = [1.0] * len(pairs)

    # 6. Build EvidenceChunk objects, gate by entailment.
    survivors: list[EvidenceChunk] = []
    for (cand, score, text), p in zip(cands, entail_probs):
        if p < entailment_floor:
            continue
        survivors.append(EvidenceChunk(
            chunk_id=str(getattr(cand, "chunk_id", "") or
                         (cand.get("chunk_id") if isinstance(cand, dict) else "")),
            document_id=str(getattr(cand, "document_id", "") or
                            (cand.get("document_id") if isinstance(cand, dict) else "")),
            content=text,
            title=str(getattr(cand, "title", "") or
                       (cand.get("title") if isinstance(cand, dict) else "") or ""),
            doi=getattr(cand, "doi", None) or (
                cand.get("doi") if isinstance(cand, dict) else None
            ),
            year=getattr(cand, "year", None) or (
                cand.get("year") if isinstance(cand, dict) else None
            ),
            rrf_score=float(score),
            nli_entail=float(p),
        ))

    # 7. Verdict.
    if len(survivors) < WEAK_MIN_CHUNKS:
        return ClaimEvidence(
            claim_id=claim.claim_id, claim_text=claim.text,
            chunks=survivors, weak=True,
            weak_reason=(
                f"only {len(survivors)} chunk(s) above entailment floor "
                f"({entailment_floor:.2f}); claim should be dropped"
            ),
        )

    return ClaimEvidence(
        claim_id=claim.claim_id, claim_text=claim.text,
        chunks=survivors,
    )


def retrieve_for_claims(
    claims: Iterable[Claim],
    **kwargs,
) -> list[ClaimEvidence]:
    """Iterator wrapper. Each claim runs sequentially today; can be
    parallelised in a follow-up patch (writer / scorer LLM calls
    block on each other anyway, so the parallelism win is bounded by
    HTTP concurrency to llama-server)."""
    return [retrieve_for_claim(c, **kwargs) for c in claims]
