"""Phase 54.6.69 — retrieval quality benchmark (MRR / Recall / NDCG).

The prior bench layer measured retrieval *speed* (``b_retrieval_hybrid_latency``)
and *rerank impact* (``b_retrieval_rerank_mrr_shift``), but never whether
retrieval actually finds the right content. This module closes that gap
with a synthetic benchmark:

1. **One-time generation**: sample N chunks from the corpus, have the
   fast LLM write a question whose *unique* answer is in that chunk,
   persist ``(question, source_chunk_id)`` to disk.
2. **Bench function**: for every stored query, run hybrid search and
   measure where the source chunk ranks. Reports MRR@10, Recall@10,
   NDCG@10 across the whole probe set.

Why synthetic and not human-labelled: labelling is expensive, and the
*relative* signal (did this retrieval tweak help or hurt?) is all we
need for regression tracking. Absolute MRR numbers are noisy but the
delta between two runs on the same probe set is stable.

The generated probe set is persisted to
``<project>/data/bench/retrieval_queries.jsonl`` so bench runs are
reproducible across sessions. Regenerate with ``--regenerate`` when the
corpus changes materially (new papers, chunker tweaks).
"""
from __future__ import annotations

import json
import logging
import math
import random
import re
import statistics
import time
from pathlib import Path
from typing import Iterable

from sciknow.testing.bench import BenchMetric, skip

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════


DEFAULT_N_QUERIES = 200
TOP_K = 10
# Minimum chunk length to be a viable question source. Very short chunks
# produce generic, unretrievable questions.
MIN_CHUNK_CHARS = 400
# Hard cap to keep prompts cheap — the model only needs enough context
# to write a specific question, not the full 8K embedding window.
MAX_CHUNK_CHARS_IN_PROMPT = 2500


_QUESTION_SYSTEM = (
    "You are generating retrieval benchmark questions for a scientific "
    "paper corpus. Given a passage, propose ONE question whose answer is "
    "specifically in this passage and would be HARD to answer without it. "
    "The question must:\n"
    "  - be specific (names entities, numbers, mechanisms, or locations "
    "from the passage — not a generic 'what is X' prompt)\n"
    "  - be answerable ONLY from the passage or very similar passages\n"
    "  - be one sentence, 8-25 words\n"
    "  - NOT reveal the answer\n"
    "Respond with ONLY the question. No quotes, no preamble, no prefix."
)


def _bench_dir() -> Path:
    from sciknow.config import settings
    d = Path(settings.data_dir) / "bench"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _queries_path() -> Path:
    return _bench_dir() / "retrieval_queries.jsonl"


# ════════════════════════════════════════════════════════════════════════
# Generation — one-time probe-set construction
# ════════════════════════════════════════════════════════════════════════


def _fetch_sample_chunks(n: int, seed: int = 42) -> list[dict]:
    """Return n random chunks {chunk_id, document_id, content, qdrant_point_id, section_type}.

    Uses ``ORDER BY random()`` which is fine for one-time sampling;
    Postgres handles a few hundred rows out of 24k cheaply.
    """
    from sqlalchemy import text as sql_text
    from sciknow.storage.db import get_session
    with get_session() as session:
        # seed the server-side RNG so sampling is stable-ish across runs
        session.execute(sql_text(f"SELECT setseed({(seed % 1000) / 1000:.3f})"))
        rows = session.execute(sql_text(f"""
            SELECT c.id::text AS chunk_id,
                   c.document_id::text AS document_id,
                   c.content,
                   c.qdrant_point_id::text AS qdrant_point_id,
                   c.section_type
            FROM chunks c
            WHERE length(c.content) >= :min_chars
            ORDER BY random()
            LIMIT :n
        """), {"min_chars": MIN_CHUNK_CHARS, "n": n * 2}).fetchall()  # overshoot
    out: list[dict] = []
    for r in rows:
        out.append({
            "chunk_id": r[0],
            "document_id": r[1],
            "content": r[2],
            "qdrant_point_id": r[3],
            "section_type": r[4] or "unknown",
        })
        if len(out) >= n:
            break
    return out


def _question_for_chunk(chunk: dict, model: str | None = None) -> str | None:
    """Ask the fast LLM to write ONE specific question answerable from this chunk."""
    from sciknow.rag.llm import complete
    content = chunk["content"][:MAX_CHUNK_CHARS_IN_PROMPT]
    # Strip the section-type prefix chunker prepends — don't want the
    # model to copy it into the question.
    content = re.sub(r"^\[\w+\][^\n]*\n\n", "", content)
    prompt = f"Passage:\n\n{content}\n\nQuestion:"
    try:
        raw = complete(
            _QUESTION_SYSTEM, prompt,
            model=model, temperature=0.4, num_ctx=4096, keep_alive=-1,
        )
    except Exception as exc:
        logger.debug("LLM question-gen failed: %s", exc)
        return None
    q = (raw or "").strip().strip('"\'').strip()
    # Drop trailing "Answer:" echoes etc.
    q = q.split("\n", 1)[0].strip()
    if len(q) < 10 or len(q) > 300:
        return None
    # Reject questions that repeat the passage (bad generator behavior)
    if q.lower() in content.lower():
        return None
    return q


def generate_probe_set(
    n: int = DEFAULT_N_QUERIES,
    model: str | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> Path:
    """Sample n chunks, ask the fast LLM for one question per chunk, persist
    to ``<project>/data/bench/retrieval_queries.jsonl``. Returns the path.

    Safe to re-run: this overwrites any existing file (use a sentinel
    flag at the call site to skip when the file exists).
    """
    path = _queries_path()
    if verbose:
        print(f"Sampling {n} chunks from the corpus (min_chars={MIN_CHUNK_CHARS})…")
    chunks = _fetch_sample_chunks(n, seed=seed)
    if not chunks:
        raise RuntimeError(
            "No chunks available for sampling — ingest some papers first."
        )
    if verbose:
        print(f"Generating {len(chunks)} questions via LLM_FAST_MODEL…")
    records: list[dict] = []
    t0 = time.monotonic()
    for i, ch in enumerate(chunks, 1):
        q = _question_for_chunk(ch, model=model)
        if q is None:
            continue
        records.append({
            "question": q,
            "source_chunk_id": ch["chunk_id"],
            "source_document_id": ch["document_id"],
            "source_qdrant_point_id": ch["qdrant_point_id"],
            "source_section_type": ch["section_type"],
        })
        if verbose and i % 20 == 0:
            elapsed = time.monotonic() - t0
            rate = i / elapsed if elapsed else 0
            print(f"  [{i}/{len(chunks)}] kept={len(records)}  "
                  f"rate={rate:.1f}/s")
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    if verbose:
        print(f"✓ Wrote {len(records)} queries to {path}")
    return path


def load_probe_set() -> list[dict]:
    path = _queries_path()
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


# ════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════


def _ndcg_at_k(rank: int, k: int) -> float:
    """Binary-relevance NDCG@k. rank is 1-indexed; 0 means not-found."""
    if rank == 0 or rank > k:
        return 0.0
    # DCG for a single relevant doc at position r: 1/log2(r+1).
    # Ideal DCG (single relevant at position 1): 1/log2(2) = 1.
    return 1.0 / math.log2(rank + 1)


def _find_source_rank(results, source_qdrant_id: str,
                      source_chunk_id: str) -> int:
    """Return 1-indexed rank of the source chunk in retrieval results.
    0 means not found in the result set.

    ``SearchCandidate.chunk_id`` in the hybrid_search module is *the
    Qdrant point UUID*, not the PostgreSQL ``chunks.id`` primary key
    (the class docstring is explicit about this: "Qdrant point UUID /
    PostgreSQL qdrant_point_id"). So we compare the candidate's
    chunk_id against our stored ``source_qdrant_point_id``. The
    source_chunk_id path is kept as a fallback in case a future
    SearchCandidate variant exposes the PG id directly.
    """
    qid = str(source_qdrant_id or "")
    pgid = str(source_chunk_id or "")
    for i, r in enumerate(results, 1):
        cand_chunk = str(getattr(r, "chunk_id", "") or "")
        # Primary match: candidate's chunk_id == our source qdrant_point_id.
        if qid and cand_chunk == qid:
            return i
        # Fallback: some result types may expose qdrant_point_id separately.
        alt = str(getattr(r, "qdrant_point_id", "") or "")
        if qid and alt == qid:
            return i
        # Last resort: PG chunks.id if the candidate exposes it.
        pg_alt = str(getattr(r, "pg_chunk_id", "") or "")
        if pgid and pg_alt == pgid:
            return i
    return 0


# ════════════════════════════════════════════════════════════════════════
# Bench function (lives here; registered from bench.py)
# ════════════════════════════════════════════════════════════════════════


def b_retrieval_recall() -> Iterable[BenchMetric]:
    """Retrieval quality: MRR@10, Recall@10, NDCG@10 on the persisted
    synthetic probe set.

    If no probe set exists, this bench skips with instructions to run
    ``sciknow bench retrieval-gen`` (the CLI entry point that wraps
    ``generate_probe_set``). We deliberately do NOT auto-generate
    inside the bench run — generation costs ~2-5 minutes of LLM time
    and should be explicit.
    """
    records = load_probe_set()
    if not records:
        yield BenchMetric("status", "no-probe-set", "",
                          note=(f"run `sciknow bench retrieval-gen` "
                                f"to create {_queries_path().name}"))
        return

    from sciknow.retrieval.hybrid_search import search
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    client = get_client()
    ranks: list[int] = []
    latencies: list[float] = []
    n_top1 = 0
    n_top10 = 0
    n_not_found = 0
    ndcg_vals: list[float] = []

    with get_session() as session:
        # Warmup — same pattern as b_retrieval_hybrid_latency.
        try:
            _ = search("warmup probe", client, session, candidate_k=10)
        except Exception:
            pass

        for rec in records:
            q = rec["question"]
            try:
                t0 = time.monotonic()
                results = search(q, client, session, candidate_k=50)
                latencies.append((time.monotonic() - t0) * 1000)
            except Exception as exc:
                logger.debug("probe search failed %r: %s", q, exc)
                continue
            rank = _find_source_rank(
                results,
                rec.get("source_qdrant_point_id", ""),
                rec.get("source_chunk_id", ""),
            )
            ranks.append(rank)
            if rank == 1:
                n_top1 += 1
            if 1 <= rank <= TOP_K:
                n_top10 += 1
            if rank == 0:
                n_not_found += 1
            ndcg_vals.append(_ndcg_at_k(rank, TOP_K))

    if not ranks:
        yield BenchMetric("status", "all-queries-failed", "")
        return

    n = len(ranks)
    # MRR@10: 1/rank when found in top-K, 0 otherwise. Standard IR metric.
    mrr = sum(1.0 / r for r in ranks if 1 <= r <= TOP_K) / n
    recall_10 = n_top10 / n
    recall_1 = n_top1 / n
    ndcg_10 = sum(ndcg_vals) / n

    yield BenchMetric("n_queries", n, "queries")
    yield BenchMetric("mrr_at_10", round(mrr, 4), "score",
                      note="higher is better; 1.0 = always top-1")
    yield BenchMetric("recall_at_1", round(recall_1, 4), "rate",
                      note="fraction of queries whose source chunk was rank-1")
    yield BenchMetric("recall_at_10", round(recall_10, 4), "rate",
                      note="fraction of queries whose source chunk was in top-10")
    yield BenchMetric("ndcg_at_10", round(ndcg_10, 4), "score",
                      note="binary-relevance NDCG; 1.0 = always top-1, 0 = never retrieved")
    yield BenchMetric("not_found_pct",
                      round(100 * n_not_found / n, 1), "%",
                      note="fraction where source chunk not in top-50 candidates")
    if latencies:
        latencies.sort()
        yield BenchMetric("latency_p50",
                          round(latencies[len(latencies) // 2], 1), "ms")
        yield BenchMetric("latency_mean",
                          round(statistics.mean(latencies), 1), "ms")
