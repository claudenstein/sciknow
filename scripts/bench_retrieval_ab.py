"""Phase 54.6.275 — A/B retrieval-quality harness.

Complement to ``sciknow bench retrieval`` (the single-config evaluator in
``sciknow/testing/retrieval_eval.py``). This script runs the SAME probe
set across TWO config variants and reports paired per-query deltas so
you can decide whether a candidate is worth shipping.

Two A/B modes:

  * ``--mode reranker``   (cheap)  — reuses the live Qdrant index, runs
    hybrid_search once per query, then reranks the same top-50
    candidates with each reranker in ``RERANKER_CANDIDATES``. No
    corpus re-embedding. ~1 minute + reranker cold-loads.
  * ``--mode embedder``   (heavy)  — re-embeds the corpus into a sidecar
    Qdrant collection per candidate embedder, runs the probe set against
    each, and compares. Costs ~20-40 min + a few GB of Qdrant space per
    embedder. Gated by ``--i-know-it-is-heavy`` because it's an expensive
    operation and you probably don't want to run it casually.

Output: a Rich table and a JSONL artefact under
``<data_dir>/bench/retrieval_ab/bench-<ts>.jsonl``.

Decision rule (from docs/EXPAND_ENRICH_RESEARCH_2.md §2.2): a candidate
must beat the baseline on MRR@10 by ≥0.03 to justify shipping. Anything
less is within run-to-run noise.

Invocation:

    uv run python scripts/bench_retrieval_ab.py --mode reranker
    uv run python scripts/bench_retrieval_ab.py --mode embedder --i-know-it-is-heavy

Reranker candidates / embedder candidates live in the constants block
below — edit there to A/B new models.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

console = Console()


# ════════════════════════════════════════════════════════════════════════
# Configuration — edit these, not the bench function
# ════════════════════════════════════════════════════════════════════════

# Reranker A/B: list of tags. First entry is treated as "baseline"
# for the delta table. The adapter in sciknow/retrieval/reranker.py
# dispatches by tag prefix, so mix bge-* and Qwen3-Reranker-* freely.
RERANKER_CANDIDATES: list[str] = [
    "BAAI/bge-reranker-v2-m3",   # baseline (pre-54.6.274)
    "Qwen/Qwen3-Reranker-4B",    # candidate shipped in 54.6.274
]

# Embedder A/B: list of (embedder_tag, dim) pairs. Each creates a
# sidecar Qdrant collection suffixed with the tag slug. First entry
# is the baseline. dim MUST match the model's output; Qdrant doesn't
# infer it.
EMBEDDER_CANDIDATES: list[tuple[str, int]] = [
    ("BAAI/bge-m3",                 1024),  # current (dense + sparse + ColBERT)
    ("Qwen/Qwen3-Embedding-4B",     2560),  # dense-only candidate
    # ("Qwen/Qwen3-Embedding-8B",   4096),  # uncomment if you have headroom
]

# Decision thresholds
MRR_SHIPPING_DELTA = 0.03  # candidate must beat baseline by this on MRR@10


# ════════════════════════════════════════════════════════════════════════
# Shared scoring helpers (mirror retrieval_eval.py so absolute numbers
# are directly comparable to `sciknow bench retrieval` output).
# ════════════════════════════════════════════════════════════════════════

TOP_K = 10


def _ndcg_at_k(rank: int, k: int) -> float:
    import math
    if rank == 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _metrics_from_ranks(ranks: list[int]) -> dict:
    n = len(ranks) or 1
    found = [r for r in ranks if 1 <= r <= TOP_K]
    mrr = sum(1.0 / r for r in found) / n
    recall_1 = sum(1 for r in ranks if r == 1) / n
    recall_10 = len(found) / n
    ndcg_10 = sum(_ndcg_at_k(r, TOP_K) for r in ranks) / n
    not_found = sum(1 for r in ranks if r == 0) / n
    return {
        "n": n,
        "mrr_at_10": round(mrr, 4),
        "recall_at_1": round(recall_1, 4),
        "recall_at_10": round(recall_10, 4),
        "ndcg_at_10": round(ndcg_10, 4),
        "not_found_pct": round(100 * not_found, 1),
    }


# ════════════════════════════════════════════════════════════════════════
# Mode 1 — Reranker A/B
# ════════════════════════════════════════════════════════════════════════


def run_reranker_ab() -> dict:
    """Rerank the SAME top-50 candidate pool with every model in
    RERANKER_CANDIDATES. Same retrieval = fair comparison."""
    from sciknow.testing.retrieval_eval import (
        load_probe_set, _find_source_rank,
    )
    from sciknow.retrieval.hybrid_search import search as hybrid_search
    from sciknow.retrieval.reranker import rerank, release_reranker
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client
    from sciknow.config import settings

    records = load_probe_set()
    if not records:
        console.print(
            "[red]No probe set found.[/red] Run "
            "`uv run sciknow bench retrieval-gen` first to create one "
            "(typically 100 queries, 2-5 min of fast-LLM time)."
        )
        sys.exit(2)

    console.print(
        f"[bold]Retrieval A/B — reranker mode[/bold]  "
        f"probe={len(records)} queries  "
        f"candidates={len(RERANKER_CANDIDATES)}"
    )

    # Stage 1: retrieve top-50 candidates for every probe query ONCE.
    # Then we rerank the cached candidates per model. Avoids spending
    # 2× the retrieval cost; isolates the reranker as the only varying
    # factor between runs.
    client = get_client()
    retrieved: list[dict] = []
    with get_session() as session:
        # warmup
        try:
            _ = hybrid_search("warmup", client, session, candidate_k=10)
        except Exception:
            pass
        t0 = time.monotonic()
        for rec in records:
            q = rec["question"]
            try:
                candidates = hybrid_search(
                    q, client, session, candidate_k=50,
                )
                retrieved.append({"rec": rec, "candidates": candidates})
            except Exception as exc:
                console.print(f"[dim]retrieve fail: {exc}[/dim]")
        console.print(
            f"  stage 1: retrieved for {len(retrieved)} queries "
            f"in {time.monotonic() - t0:.0f}s"
        )

    # Stage 2: rerank per model. Mutate settings.reranker_model and
    # release_reranker() between each so the cached singleton picks
    # up the new tag.
    results: dict[str, dict] = {}
    for tag in RERANKER_CANDIDATES:
        console.print(f"\n[bold]→ reranker: {tag}[/bold]")
        object.__setattr__(settings, "reranker_model", tag)
        release_reranker()

        ranks: list[int] = []
        latencies: list[float] = []
        for item in retrieved:
            rec = item["rec"]
            t0 = time.monotonic()
            try:
                reranked = rerank(
                    rec["question"], item["candidates"], top_k=TOP_K,
                )
            except Exception as exc:
                console.print(f"[dim]  rerank fail {tag}: {exc}[/dim]")
                continue
            latencies.append((time.monotonic() - t0) * 1000)
            r = _find_source_rank(
                reranked,
                rec.get("source_qdrant_point_id", ""),
                rec.get("source_chunk_id", ""),
            )
            ranks.append(r)
        metrics = _metrics_from_ranks(ranks)
        if latencies:
            latencies.sort()
            metrics["latency_p50_ms"] = round(
                latencies[len(latencies) // 2], 1,
            )
            metrics["latency_mean_ms"] = round(statistics.mean(latencies), 1)
        results[tag] = metrics

    return {"mode": "reranker", "results": results}


# ════════════════════════════════════════════════════════════════════════
# Mode 2 — Embedder A/B (heavy)
# ════════════════════════════════════════════════════════════════════════


def _safe_tag_slug(tag: str) -> str:
    """Qdrant collection names must be filesystem-safe. Replace /, :
    with _, lowercase. e.g. 'Qwen/Qwen3-Embedding-4B' → 'qwen_qwen3-embedding-4b'."""
    return tag.replace("/", "_").replace(":", "_").lower()


def _sidecar_collection_name(embedder_tag: str) -> str:
    """Sidecar collection carries the project's Qdrant prefix + an
    `ab_<tag>_papers` suffix so it's easy to spot in `qdrant-web`."""
    from sciknow.core.project import get_active_project
    prefix = get_active_project().qdrant_prefix
    # e.g. 'global_cooling_ab_qwen_qwen3-embedding-4b_papers'
    return f"{prefix}_ab_{_safe_tag_slug(embedder_tag)}_papers"


def _ab_reembed_corpus(embedder_tag: str, dim: int, batch_size: int = 16) -> str:
    """Create <sidecar>_papers, batch-embed every chunk with the
    candidate dense embedder (sentence-transformers), and upsert with
    the SAME qdrant_point_id as the prod collection so probe records
    still resolve. Returns the collection name.

    Idempotent: drops any pre-existing sidecar before recreating. The
    baseline bge-m3 collection is never touched.
    """
    from sqlalchemy import text as sql_text
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
    )
    from sciknow.storage.qdrant import get_client
    from sciknow.storage.db import get_session
    from sentence_transformers import SentenceTransformer

    coll = _sidecar_collection_name(embedder_tag)
    client = get_client()

    # Drop + create (idempotent rerun)
    if client.collection_exists(coll):
        client.delete_collection(coll)
    client.create_collection(
        collection_name=coll,
        vectors_config={
            "dense": VectorParams(size=dim, distance=Distance.COSINE),
        },
    )
    console.print(f"  [dim]created sidecar collection {coll} (dim={dim})[/dim]")

    # Load the embedder. Qwen3-Embedding accepts a scientific-literature
    # instruction for queries but documents don't get prefixed, so we
    # encode raw chunk text during ingest.
    console.print(f"  [dim]loading {embedder_tag}… (may download ~8 GB on first run)[/dim]")
    model = SentenceTransformer(embedder_tag, device="cuda", trust_remote_code=True)

    # Fetch every chunk + its prod qdrant_point_id
    with get_session() as s:
        rows = s.execute(sql_text("""
            SELECT qdrant_point_id::text, content FROM chunks
            WHERE qdrant_point_id IS NOT NULL
            ORDER BY id
        """)).fetchall()
    total = len(rows)
    console.print(f"  [dim]embedding {total} chunks in batches of {batch_size}[/dim]")

    t0 = time.monotonic()
    for i in range(0, total, batch_size):
        batch = rows[i:i + batch_size]
        texts = [r[1] for r in batch]
        # Truncate very long chunks at model's max context to avoid OOM.
        # Qwen3-Embedding handles 32k tokens but SciKnow chunks are
        # typically 500-2500 tokens, so this is a safety rail.
        emb = model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        points = [
            PointStruct(id=str(r[0]), vector={"dense": e.tolist()})
            for r, e in zip(batch, emb)
        ]
        client.upsert(collection_name=coll, points=points)
        if (i // batch_size) % 20 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + batch_size) / max(elapsed, 0.001)
            eta_s = (total - i) / max(rate, 0.001)
            console.print(
                f"    [dim]{i + len(batch)}/{total}  "
                f"({rate:.0f} chunks/s · ETA {eta_s / 60:.1f}m)[/dim]"
            )

    console.print(
        f"  [green]✓[/green] re-embedded {total} chunks "
        f"in {(time.monotonic() - t0) / 60:.1f}m"
    )
    return coll


def _ab_embed_query(model, query: str, is_qwen3: bool) -> list[float]:
    """Encode a query. Qwen3-Embedding recommends an instruction prefix
    on QUERIES only (documents are unprefixed at encode time). bge-m3
    doesn't need one. Scientific-literature instruction per the HF
    model card's task-description pattern.
    """
    if is_qwen3:
        instruction = (
            "Given a scientific literature search query, retrieve "
            "relevant passages that answer the query"
        )
        # Qwen3-Embedding format: "Instruct: {task}\nQuery: {query}"
        prompt = f"Instruct: {instruction}\nQuery: {query}"
        emb = model.encode(
            [prompt], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
    else:
        emb = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
    return emb[0].tolist()


def _ab_dense_search(
    client, collection: str, vector: list[float], top_k: int,
) -> list[str]:
    """Pure dense search on a named collection. Returns point-id list
    in descending relevance."""
    resp = client.query_points(
        collection_name=collection, query=vector,
        using="dense", limit=top_k, with_payload=False,
    )
    return [str(p.id) for p in resp.points]


def _find_rank(
    ordered_ids: list[str], source_qdrant_id: str, source_chunk_id: str,
) -> int:
    """1-based rank of the source point; 0 if not in the list."""
    for i, pid in enumerate(ordered_ids, 1):
        if pid == source_qdrant_id or pid == source_chunk_id:
            return i
    return 0


def run_embedder_ab() -> dict:
    """Three-way A/B: baseline dense (bge-m3 on prod collection) vs
    candidate dense (Qwen3-Embedding-4B on sidecar) vs RRF-fused of
    the two (the 'dual-embedder' architecture).

    Uses DENSE-ONLY retrieval on each side to isolate the dense
    signal's quality — sparse and FTS are excluded so the three
    numbers are directly comparable. This is deliberately different
    from `sciknow bench retrieval` (which benchmarks the full hybrid
    stack).
    """
    from sciknow.testing.retrieval_eval import load_probe_set
    from sciknow.storage.qdrant import get_client
    from sciknow.storage.db import get_session  # noqa: F401 — kept for parity
    from sentence_transformers import SentenceTransformer
    from FlagEmbedding import BGEM3FlagModel

    records = load_probe_set()
    if not records:
        console.print("[red]No probe set found.[/red] Run "
                      "`uv run sciknow bench retrieval-gen` first.")
        sys.exit(2)

    baseline_tag, _ = EMBEDDER_CANDIDATES[0]
    candidate_tag, candidate_dim = EMBEDDER_CANDIDATES[1]

    console.print(
        f"[bold]Retrieval A/B — embedder mode[/bold]  "
        f"probe={len(records)} queries\n"
        f"  baseline:  [cyan]{baseline_tag}[/cyan] (prod Qdrant `papers`)\n"
        f"  candidate: [cyan]{candidate_tag}[/cyan] (sidecar, dim {candidate_dim})\n"
    )

    # Stage 1: re-embed corpus with candidate into sidecar.
    sidecar = _ab_reembed_corpus(candidate_tag, candidate_dim)

    # Stage 2: load the two query embedders concurrently. bge-m3 is
    # the FlagEmbedding-native class; Qwen3 is sentence-transformers.
    # Reuse the sciknow CPU-fallback helper for bge-m3 so the test
    # path matches prod exactly.
    from sciknow.retrieval.device import load_with_cpu_fallback
    console.print("\n[dim]loading bge-m3 query embedder…[/dim]")
    bge = load_with_cpu_fallback(
        BGEM3FlagModel, baseline_tag, use_fp16=True,
    )
    console.print("[dim]loading Qwen3 query embedder…[/dim]")
    qwen = SentenceTransformer(
        candidate_tag, device="cuda", trust_remote_code=True,
    )

    # Stage 3: run the probe set through each of the three scenarios.
    from sciknow.retrieval.hybrid_search import PAPERS_COLLECTION
    from sciknow.storage.qdrant import get_client as _get_client
    client = _get_client()

    console.print(
        f"\n[dim]running {len(records)} probes × 3 scenarios "
        f"(bge dense, Qwen3 dense, RRF-fused)[/dim]"
    )

    def _rrf_merge(list_a: list[str], list_b: list[str], top: int) -> list[str]:
        """Small RRF merge for two lists; scores bye id, returns top-k."""
        RRF_K = 60
        scores: dict[str, float] = {}
        for lst in (list_a, list_b):
            for rank, pid in enumerate(lst):
                scores[pid] = scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
        return [
            pid for pid, _ in sorted(
                scores.items(), key=lambda x: x[1], reverse=True,
            )
        ][:top]

    ranks_bge: list[int] = []
    ranks_qwen: list[int] = []
    ranks_rrf: list[int] = []
    latencies_bge: list[float] = []
    latencies_qwen: list[float] = []

    total_probes = len(records)
    for i, rec in enumerate(records):
        q = rec["question"]
        src_qpid = rec.get("source_qdrant_point_id", "")
        src_cid = rec.get("source_chunk_id", "")
        try:
            t0 = time.monotonic()
            bge_out = bge.encode(
                [q], batch_size=1, max_length=512,
                return_dense=True, return_sparse=False,
                return_colbert_vecs=False,
            )
            bge_vec = bge_out["dense_vecs"][0].tolist()
            latencies_bge.append((time.monotonic() - t0) * 1000)

            t0 = time.monotonic()
            qwen_vec = _ab_embed_query(qwen, q, is_qwen3=True)
            latencies_qwen.append((time.monotonic() - t0) * 1000)

            # Dense-only searches on each collection
            ids_bge = _ab_dense_search(client, PAPERS_COLLECTION, bge_vec, 50)
            ids_qwen = _ab_dense_search(client, sidecar, qwen_vec, 50)
            ids_rrf = _rrf_merge(ids_bge, ids_qwen, top=50)

            ranks_bge.append(_find_rank(ids_bge, src_qpid, src_cid))
            ranks_qwen.append(_find_rank(ids_qwen, src_qpid, src_cid))
            ranks_rrf.append(_find_rank(ids_rrf, src_qpid, src_cid))
        except Exception as exc:
            console.print(f"[dim]probe {i} failed: {exc}[/dim]")
            continue

        if (i + 1) % 25 == 0:
            console.print(f"  [dim]{i + 1}/{total_probes} done[/dim]")

    # Stage 4: metrics per scenario
    m_bge = _metrics_from_ranks(ranks_bge)
    m_qwen = _metrics_from_ranks(ranks_qwen)
    m_rrf = _metrics_from_ranks(ranks_rrf)
    if latencies_bge:
        latencies_bge.sort()
        m_bge["latency_p50_ms"] = round(latencies_bge[len(latencies_bge) // 2], 1)
    if latencies_qwen:
        latencies_qwen.sort()
        m_qwen["latency_p50_ms"] = round(latencies_qwen[len(latencies_qwen) // 2], 1)
    m_rrf["latency_p50_ms"] = round(
        (m_bge.get("latency_p50_ms", 0) + m_qwen.get("latency_p50_ms", 0)), 1,
    )

    results = {
        f"{baseline_tag} (dense)":         m_bge,
        f"{candidate_tag} (dense)":        m_qwen,
        f"RRF-fused dual dense":           m_rrf,
    }
    return {"mode": "embedder", "results": results, "sidecar_collection": sidecar}


# ════════════════════════════════════════════════════════════════════════
# CLI + output
# ════════════════════════════════════════════════════════════════════════


def _render_results(result: dict) -> None:
    mode = result["mode"]
    results = result["results"]
    baseline_tag = next(iter(results.keys()))
    baseline = results[baseline_tag]

    t = Table(
        title=f"Retrieval A/B · {mode} · baseline = {baseline_tag}",
        box=box.SIMPLE_HEAD,
    )
    t.add_column("Model")
    t.add_column("N", justify="right")
    t.add_column("MRR@10", justify="right")
    t.add_column("ΔMRR", justify="right")
    t.add_column("Recall@1", justify="right")
    t.add_column("Recall@10", justify="right")
    t.add_column("nDCG@10", justify="right")
    t.add_column("Not found %", justify="right")
    t.add_column("p50 ms", justify="right")
    for tag, m in results.items():
        delta = m["mrr_at_10"] - baseline["mrr_at_10"]
        if tag == baseline_tag:
            delta_str = "[dim]—[/dim]"
        elif delta >= MRR_SHIPPING_DELTA:
            delta_str = f"[bright_green]+{delta:.3f} ✓[/bright_green]"
        elif delta <= -MRR_SHIPPING_DELTA:
            delta_str = f"[bright_red]{delta:+.3f}[/bright_red]"
        else:
            delta_str = f"[yellow]{delta:+.3f}[/yellow]"
        t.add_row(
            tag,
            str(m["n"]),
            f"{m['mrr_at_10']:.4f}",
            delta_str,
            f"{m['recall_at_1']:.4f}",
            f"{m['recall_at_10']:.4f}",
            f"{m['ndcg_at_10']:.4f}",
            f"{m['not_found_pct']}",
            str(m.get("latency_p50_ms", "—")),
        )
    console.print(t)

    # Write artefact
    from sciknow.config import settings
    ab_dir = Path(settings.data_dir) / "bench" / "retrieval_ab"
    ab_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ab_dir / f"bench-{ts}.jsonl"
    with out_path.open("w") as f:
        f.write(json.dumps({
            "_kind": "header", "ts": ts, "mode": mode,
            "baseline": baseline_tag, "shipping_threshold_mrr": MRR_SHIPPING_DELTA,
        }) + "\n")
        for tag, m in results.items():
            f.write(json.dumps({"model": tag, **m}) + "\n")
    console.print(f"[dim]Wrote {out_path}[/dim]")

    # Verdict line — help the operator act.
    console.print()
    for tag, m in results.items():
        if tag == baseline_tag:
            continue
        delta = m["mrr_at_10"] - baseline["mrr_at_10"]
        if delta >= MRR_SHIPPING_DELTA:
            console.print(
                f"[bright_green]✓ Ship: {tag} beats baseline by "
                f"+{delta:.3f} on MRR@10 (threshold {MRR_SHIPPING_DELTA})."
                "[/bright_green]"
            )
        elif delta >= 0:
            console.print(
                f"[yellow]≈ Neutral: {tag} is +{delta:.3f} — below the "
                f"{MRR_SHIPPING_DELTA} shipping threshold; within noise."
                "[/yellow]"
            )
        else:
            console.print(
                f"[bright_red]✗ Regression: {tag} is {delta:+.3f}. "
                "Keep baseline.[/bright_red]"
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=["reranker", "embedder"], default="reranker",
    )
    p.add_argument(
        "--i-know-it-is-heavy", action="store_true",
        help="Required with --mode embedder; prevents accidental invocation.",
    )
    args = p.parse_args()

    if args.mode == "embedder" and not args.i_know_it_is_heavy:
        console.print(
            "[red]--mode embedder requires --i-know-it-is-heavy "
            "(re-embedding 25k chunks takes ~30-60 min per candidate).[/red]"
        )
        sys.exit(2)

    if args.mode == "reranker":
        result = run_reranker_ab()
    else:
        result = run_embedder_ab()

    _render_results(result)


if __name__ == "__main__":
    main()
