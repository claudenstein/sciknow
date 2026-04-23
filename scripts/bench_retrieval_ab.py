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


def run_embedder_ab() -> dict:
    """Re-embed the corpus into a sidecar Qdrant collection per
    candidate embedder, run the probe set against each, compare.

    Not fully implemented here — designed as a spec + orchestrator.
    Re-embedding 25k chunks with a 4-8B embedder is ~30-60 min; we
    stage this as explicit opt-in to avoid surprising the user.
    """
    from sciknow.testing.retrieval_eval import (
        load_probe_set, _find_source_rank,
    )
    from sciknow.storage.db import get_session

    records = load_probe_set()
    if not records:
        console.print("[red]No probe set found.[/red]")
        sys.exit(2)

    console.print(
        f"[bold yellow]Retrieval A/B — EMBEDDER mode (heavy)[/bold yellow]\n"
        f"  probe={len(records)} queries\n"
        f"  candidates={[c[0] for c in EMBEDDER_CANDIDATES]}"
    )
    console.print(
        "[yellow]Per embedder:[/yellow] create a sidecar Qdrant collection "
        "<slug>_ab_<embedder_tag>_papers, batch-embed all chunks, then run "
        "the probe set via a direct Qdrant dense query (not hybrid_search — "
        "that's wired to the prod collection name). Estimated ~30-60 min per "
        "embedder, ~2-4 GB Qdrant space per collection."
    )
    console.print(
        "[yellow]Not auto-running[/yellow] — this is a scaffold. To enable, "
        "implement `_ab_reembed_corpus(embedder_tag, dim)` and "
        "`_ab_eval_collection(collection_name, probe_records)` below per the "
        "docstring contract; they're deliberately left as stubs so the "
        "script is safe to invoke by accident. See docs/RESEARCH.md §embedder-ab."
    )
    # Intentional exit — see docstring. When someone implements the
    # stubs, flip the `NotImplementedError` to an actual call.
    raise NotImplementedError(
        "Embedder A/B orchestrator is a stub. Implement "
        "_ab_reembed_corpus + _ab_eval_collection to enable."
    )


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
