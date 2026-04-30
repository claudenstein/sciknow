"""Phase 54.6.121 (Tier 3 #1) — SPECTER2 rerank bench.

Decision criterion (from `docs/research/EXPAND_ENRICH_RESEARCH_2.md` §2.2):
SPECTER2 rerank must beat the bge-m3 baseline on MRR@10 by ≥ 0.06
absolute (so > 0.60 on the global-cooling 0.514 baseline) for it to
justify shipping as the default rerank step. Otherwise it stays an
opt-in tool and the bge-m3 + RRF pipeline stays the default.

Reuses the 54.6.69 retrieval probe set (data/bench/retrieval_queries.jsonl).
For each probe:
  1. Run hybrid_search to get the top-50 bge-m3 candidates (baseline).
  2. Find the rank of the source chunk → baseline metric per probe.
  3. Rerank those same 50 with SPECTER2 cosine on chunk text.
  4. Find the rank of the source chunk → reranked metric per probe.

Reports both bge-m3 baseline AND SPECTER2-reranked MRR@10 / Recall@1
/ Recall@10 / NDCG@10 so the delta is decidable without re-running
the baseline separately.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Iterable

from sciknow.testing.bench import BenchMetric

logger = logging.getLogger(__name__)


_TOP_K = 10


def _ndcg_at_k(rank: int, k: int) -> float:
    if rank == 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _find_source_rank(results, source_qdrant_id: str) -> int:
    qid = str(source_qdrant_id or "")
    for i, r in enumerate(results, 1):
        cand = str(getattr(r, "chunk_id", "") or "")
        if qid and cand == qid:
            return i
        alt = str(getattr(r, "qdrant_point_id", "") or "")
        if qid and alt == qid:
            return i
    return 0


def _candidate_text(r) -> str:
    """Best-effort text extraction from a SearchCandidate. The actual
    chunk text field on the dataclass is `content_preview` (other
    field names listed for forward-compat). Falls back to title +
    section header only when no text is available — but that fallback
    is mostly meaningless and indicates a stale bench run."""
    for fname in ("content_preview", "text", "content", "snippet", "preview"):
        v = getattr(r, fname, None)
        if isinstance(v, str) and v.strip():
            return v[:1500]
    title = getattr(r, "title", "") or getattr(r, "paper_title", "") or ""
    section = getattr(r, "section_type", "") or ""
    return (title + " " + section).strip() or "—"


def b_specter2_rerank() -> Iterable[BenchMetric]:
    """Compare bge-m3 alone vs bge-m3 + SPECTER2 rerank on the
    persisted probe set. Yields baseline metrics, reranked metrics,
    and the delta + ship-decision boolean.
    """
    from sciknow.testing.retrieval_eval import load_probe_set
    from sciknow.retrieval.hybrid_search import search
    from sciknow.retrieval import specter2 as _s2
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    records = load_probe_set()
    if not records:
        yield BenchMetric("status", "no-probe-set", "",
                          note="run `sciknow bench retrieval-gen` first")
        return

    client = get_client()
    base_ranks: list[int] = []
    rer_ranks: list[int] = []
    s2_load_ms: float = 0.0
    s2_encode_ms_total: float = 0.0

    # Pre-load SPECTER2 once so the per-probe rerank cost reflects only
    # the encode pass.
    _t0 = time.monotonic()
    _s2.encode_query("warmup")
    s2_load_ms = (time.monotonic() - _t0) * 1000

    with get_session() as session:
        try:
            _ = search("warmup", client, session, candidate_k=10)
        except Exception:
            pass
        for rec in records:
            q = rec["question"]
            src = rec.get("source_qdrant_point_id", "")
            try:
                results = search(q, client, session, candidate_k=50)
            except Exception as exc:
                logger.debug("search failed %r: %s", q, exc)
                continue
            if not results:
                continue
            base_ranks.append(_find_source_rank(results, src))
            # Build a rerank pool: dict per candidate carrying the chunk_id
            # we need to find later + the text we'll feed SPECTER2.
            pool = [{
                "chunk_id": str(getattr(r, "chunk_id", "") or ""),
                "text": _candidate_text(r),
            } for r in results]
            t0 = time.monotonic()
            _s2.rerank(q, pool, text_key="text")
            s2_encode_ms_total += (time.monotonic() - t0) * 1000
            # Source rank in the reranked pool
            rerank_rank = 0
            for i, c in enumerate(pool, 1):
                if c["chunk_id"] == str(src or ""):
                    rerank_rank = i
                    break
            rer_ranks.append(rerank_rank)

    if not base_ranks:
        yield BenchMetric("status", "all-queries-failed", "")
        return

    n = len(base_ranks)

    def _stats(ranks: list[int]) -> dict:
        mrr = sum(1.0 / r for r in ranks if 1 <= r <= _TOP_K) / n
        r1 = sum(1 for r in ranks if r == 1) / n
        r10 = sum(1 for r in ranks if 1 <= r <= _TOP_K) / n
        ndcg = sum(_ndcg_at_k(r, _TOP_K) for r in ranks) / n
        return {"mrr": mrr, "r1": r1, "r10": r10, "ndcg": ndcg}

    base = _stats(base_ranks)
    rer = _stats(rer_ranks)
    delta_mrr = rer["mrr"] - base["mrr"]
    SHIP_THRESHOLD = 0.06  # decision criterion from §2.2
    ship = delta_mrr >= SHIP_THRESHOLD

    yield BenchMetric("n_queries", n, "queries")
    yield BenchMetric("baseline_mrr_at_10", round(base["mrr"], 4), "score",
                      note="bge-m3 + RRF (the existing pipeline)")
    yield BenchMetric("rerank_mrr_at_10", round(rer["mrr"], 4), "score",
                      note="bge-m3 top-50 → SPECTER2 cosine rerank")
    yield BenchMetric("delta_mrr_at_10", round(delta_mrr, 4), "delta",
                      note=f"ship if ≥ {SHIP_THRESHOLD}")
    yield BenchMetric("ship_decision", 1 if ship else 0, "bool",
                      note=("SHIP — set RERANK_BACKEND=specter2 in .env" if ship
                            else "PARK — SPECTER2 doesn't beat the threshold"))
    yield BenchMetric("baseline_recall_at_1", round(base["r1"], 4), "rate")
    yield BenchMetric("rerank_recall_at_1", round(rer["r1"], 4), "rate")
    yield BenchMetric("baseline_recall_at_10", round(base["r10"], 4), "rate")
    yield BenchMetric("rerank_recall_at_10", round(rer["r10"], 4), "rate")
    yield BenchMetric("baseline_ndcg_at_10", round(base["ndcg"], 4), "score")
    yield BenchMetric("rerank_ndcg_at_10", round(rer["ndcg"], 4), "score")
    yield BenchMetric("specter2_load_ms", round(s2_load_ms, 1), "ms")
    yield BenchMetric("specter2_encode_ms_total", round(s2_encode_ms_total, 1), "ms",
                      note=f"{round(s2_encode_ms_total/n, 1)} ms per query (50 docs each)")
