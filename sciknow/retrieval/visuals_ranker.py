"""Phase 54.6.139 — 5-signal ranker for visuals-in-writer integration.

Given a draft sentence (or paragraph) and the set of papers already
cited in its vicinity, rank candidate visuals by how well each would
serve as an inline citation at that point in the prose.

Design is documented in detail in ``docs/research/RESEARCH.md §7.X`` (survey +
design brief). This module is the implementation of §3, using only
primitives sciknow already has: ``bge-reranker-v2-m3`` cross-encoder,
``visuals.ai_caption`` (Phase 54.6.72 VLM-enriched), and
``visuals.mention_paragraphs`` (Phase 54.6.138 body-text linker).

Signals (listed in design order; weights empirically tuneable):

1. **Cross-encoder(sentence, ai_caption)** — the single strongest
   signal. bge-reranker-v2-m3 score in [0, 1].
2. **Same-paper co-citation bonus** — additive boost when the visual
   belongs to a paper already cited in the sentence. Implements the
   faithfulness constraint: readers expect a referenced figure to come
   from a referenced paper.
3. **Mention-paragraph alignment** — max cross-encoder score across
   the visual's stored mention paragraphs (the author's own rhetorical
   framing of why the figure was cited). Per SciCap+ this is the
   strongest retrieval signal for matching figures to target prose.
4. **(Deferred) VLM claim-depiction faithfulness** — run the VLM on
   (sentence, image) for top-3 candidates only. Not in this module
   yet; needs an eval set to calibrate its weight and we want to see
   how far the text-only signals go first.
5. **Section-type prior** — draft section → preferred visual kinds
   (methods → figure, results → chart/table, etc.). Small boost.

Public API:
    rank_visuals(sentence, *, cited_doc_ids, section_type, top_k=5)
        -> list[RankedVisual]

Candidate pool assembly:
    - Union of all visuals from papers in ``cited_doc_ids`` (the
      writer agent passes its current citation set).
    - Top-N from ``search_visuals`` (hybrid dense+sparse) using the
      sentence as the query, filling out the pool to ``candidate_k``.

Rerank cost: O(candidate_k × (1 + mean_mentions_per_visual)) pair
evaluations with the cross-encoder. On a 15-candidate pool with ~3
mentions each that's ~60 pairs = ~0.7s on CPU (bge-reranker-v2-m3
~87 pairs/s per the Phase 54.6.136 bench). Latency is acceptable for
interactive writing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)


# ── Signal weights ──────────────────────────────────────────────────
#
# These combine into a [0, ~1.4] composite. No normalisation because
# the final score is only used to rank (not compared across queries).
# Documented here so a future eval-set-driven tuner has one place to
# mutate them.

W_CAPTION         = 0.40
W_MENTION         = 0.35
W_SAME_PAPER      = 0.20
W_SECTION_PRIOR   = 0.05


# ── Section-type prior ──────────────────────────────────────────────
#
# For each canonical draft section, the visual kinds that typically
# make sense. Matches the canonical section names sciknow's chunker
# emits (see _SECTION_PATTERNS in ingestion/chunker.py).

_SECTION_PRIORS: dict[str, set[str]] = {
    "introduction":   {"figure"},               # conceptual overview
    "methods":        {"figure"},               # schematic / diagram
    "results":        {"chart", "table", "figure"},
    "discussion":     {"chart", "figure"},
    "conclusion":     {"chart", "figure"},
    "abstract":       set(),                    # no strong prior
    "related_work":   {"figure"},
    "appendix":       {"chart", "table", "figure", "equation", "code"},
}


def _section_prior_hit(visual_kind: str, section_type: str | None) -> float:
    """1.0 if the visual's kind matches the section's preferred kinds,
    else 0.0. Neutral (0.5) when the section type is unknown / unmapped,
    so the prior neither helps nor hurts candidates in that case."""
    if not section_type:
        return 0.5
    preferred = _SECTION_PRIORS.get(section_type.lower())
    if preferred is None:
        return 0.5
    if not preferred:
        return 0.5
    return 1.0 if (visual_kind or "").lower() in preferred else 0.0


# ── Ranker output ───────────────────────────────────────────────────


@dataclass
class RankedVisual:
    """Final output row. Scores are preserved per-signal so the caller
    can display or debug which signal carried the ranking (useful both
    for the UI's "why was this suggested" tooltip and for ablation
    studies on the eval set)."""
    visual_id: str
    document_id: str
    kind: str
    figure_num: str
    ai_caption: str
    paper_title: str = ""

    # Signal breakdown
    caption_score:       float = 0.0
    mention_score:       float = 0.0
    same_paper:          bool  = False
    section_prior_hit:   float = 0.5
    composite_score:     float = 0.0

    # Best-matching mention paragraph (for display / explanation)
    best_mention_text:   str | None = None


def _fetch_visuals_by_doc_ids(session, doc_ids: Sequence[str]) -> list[dict]:
    """All visuals attached to these documents. Deduplicates on visual_id."""
    if not doc_ids:
        return []
    rows = session.execute(sql_text("""
        SELECT v.id::text AS visual_id,
               v.document_id::text AS document_id,
               v.kind,
               COALESCE(v.figure_num, '') AS figure_num,
               COALESCE(v.ai_caption, '') AS ai_caption,
               COALESCE(v.caption,    '') AS caption,
               v.mention_paragraphs,
               COALESCE(pm.title, '') AS paper_title
        FROM visuals v
        LEFT JOIN paper_metadata pm ON pm.document_id = v.document_id
        WHERE v.document_id = ANY(CAST(:docs AS uuid[]))
    """), {"docs": list(doc_ids)}).mappings().all()
    return [dict(r) for r in rows]


def _fetch_visuals_by_ids(session, visual_ids: Sequence[str]) -> list[dict]:
    if not visual_ids:
        return []
    rows = session.execute(sql_text("""
        SELECT v.id::text AS visual_id,
               v.document_id::text AS document_id,
               v.kind,
               COALESCE(v.figure_num, '') AS figure_num,
               COALESCE(v.ai_caption, '') AS ai_caption,
               COALESCE(v.caption,    '') AS caption,
               v.mention_paragraphs,
               COALESCE(pm.title, '') AS paper_title
        FROM visuals v
        LEFT JOIN paper_metadata pm ON pm.document_id = v.document_id
        WHERE v.id = ANY(CAST(:ids AS uuid[]))
    """), {"ids": list(visual_ids)}).mappings().all()
    return [dict(r) for r in rows]


# ── The ranker ─────────────────────────────────────────────────────


def rank_visuals(
    sentence: str,
    *,
    cited_doc_ids: Sequence[str] = (),
    section_type: str | None = None,
    candidate_k: int = 15,
    top_k: int = 5,
) -> list[RankedVisual]:
    """Rank candidate visuals against a draft sentence.

    Candidate pool assembly:
      - All visuals from papers in ``cited_doc_ids`` (the writer's
        current citation set).
      - Plus top-``candidate_k`` from hybrid visuals search on the
        sentence, to catch figures from papers the writer hasn't cited
        yet but whose figure fits the claim.
      - Dedup on visual_id.

    Returns the top ``top_k`` RankedVisual objects ordered by composite
    score (higher = better).
    """
    from sciknow.retrieval import reranker as _rr
    from sciknow.retrieval.visuals_search import search_visuals
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client as _qclient

    s = (sentence or "").strip()
    if not s:
        return []

    client = _qclient()
    with get_session() as session:
        # (1) Papers-cited pool — read directly from the visuals table
        cited_vis = _fetch_visuals_by_doc_ids(session, cited_doc_ids or [])
        # (2) Hybrid retrieval on the sentence — fills out the pool
        try:
            hits = search_visuals(s, client, candidate_k=candidate_k)
        except Exception as exc:
            logger.warning("search_visuals failed for ranker: %s", exc)
            hits = []

        # Dedup: cited pool first (takes precedence for metadata), then
        # add any hybrid hits not already in the cited set.
        seen: set[str] = {v["visual_id"] for v in cited_vis}
        to_fetch_ids = [h.visual_id for h in hits if h.visual_id and h.visual_id not in seen]
        extra = _fetch_visuals_by_ids(session, to_fetch_ids)
        pool = cited_vis + extra

        if not pool:
            return []

    # Guard: cap pool size so rerank latency stays bounded. Prefer the
    # cited-paper visuals (they have the co-citation bonus) and fill
    # the remainder from the hybrid hits in score order.
    pool = pool[: max(candidate_k, len(cited_vis))]

    cited_set = {str(d) for d in (cited_doc_ids or []) if d}

    # Collect all cross-encoder pairs in one batch per visual — caption
    # pair + all mention-paragraph pairs — so the reranker runs once.
    caption_pairs: list[tuple[str, str]] = []   # (sentence, caption_text)
    mention_idx: dict[int, list[int]] = {}      # visual_index → rerank-row indices for its mentions
    mention_texts: dict[int, list[str]] = {}    # visual_index → mention text strings
    all_pairs: list[list[str]] = []             # flat input to reranker

    # Build a fake "SearchCandidate"-like wrapper so we can call
    # reranker.rerank which needs content_preview.
    class _P:
        def __init__(self, text):
            self.content_preview = text
            self.rrf_score = 0.0

    caption_candidates: list[_P] = []

    for vi, v in enumerate(pool):
        # Caption: prefer ai_caption, fall back to original caption
        cap_text = (v.get("ai_caption") or v.get("caption") or "").strip()
        if not cap_text:
            # No usable text → caption score stays 0
            caption_candidates.append(_P(""))
        else:
            caption_candidates.append(_P(cap_text))

        mps = v.get("mention_paragraphs") or []
        texts = [mp.get("text", "") for mp in mps if mp.get("text")]
        mention_texts[vi] = texts

    # Score captions in a single batch
    if caption_candidates:
        try:
            scored = _rr.rerank(s, caption_candidates, top_k=len(caption_candidates))
            # After rerank, candidate.rrf_score = cross-encoder score.
            # But rerank() returns top_k sorted; we need to map back.
            score_by_obj = {id(c): c.rrf_score for c in scored}
            caption_scores = [score_by_obj.get(id(c), 0.0) for c in caption_candidates]
        except Exception as exc:
            logger.warning("caption rerank failed: %s", exc)
            caption_scores = [0.0] * len(caption_candidates)
    else:
        caption_scores = []

    # Score mention paragraphs per visual. Do one rerank call per
    # visual since mentions are visual-specific. Small batches.
    mention_scores: list[float] = [0.0] * len(pool)
    best_mention:  list[str | None] = [None] * len(pool)
    for vi, texts in mention_texts.items():
        if not texts:
            continue
        objs = [_P(t) for t in texts]
        try:
            scored = _rr.rerank(s, objs, top_k=len(objs))
            # scored sorted desc by rerank score; top gives max
            top = scored[0] if scored else None
            if top is not None:
                mention_scores[vi] = float(top.rrf_score or 0.0)
                best_mention[vi] = top.content_preview
        except Exception as exc:
            logger.debug("mention rerank failed for visual %d: %s", vi, exc)

    # Compose + sort
    out: list[RankedVisual] = []
    for vi, v in enumerate(pool):
        same = v.get("document_id") in cited_set
        section_hit = _section_prior_hit(v.get("kind", ""), section_type)
        composite = (
            W_CAPTION       * caption_scores[vi] if vi < len(caption_scores) else 0.0
        ) + (
            W_MENTION       * mention_scores[vi]
        ) + (
            W_SAME_PAPER    * (1.0 if same else 0.0)
        ) + (
            W_SECTION_PRIOR * section_hit
        )
        out.append(RankedVisual(
            visual_id=v["visual_id"],
            document_id=v["document_id"],
            kind=v.get("kind", ""),
            figure_num=v.get("figure_num", ""),
            ai_caption=(v.get("ai_caption") or v.get("caption") or "")[:500],
            paper_title=v.get("paper_title", ""),
            caption_score=round(caption_scores[vi] if vi < len(caption_scores) else 0.0, 4),
            mention_score=round(mention_scores[vi], 4),
            same_paper=same,
            section_prior_hit=section_hit,
            composite_score=round(composite, 4),
            best_mention_text=(best_mention[vi] or None),
        ))

    out.sort(key=lambda r: r.composite_score, reverse=True)
    return out[: max(top_k, 1)]
