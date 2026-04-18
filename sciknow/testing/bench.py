"""End-to-end benchmarking harness for sciknow.

Companion to ``sciknow/testing/protocol.py``. Where protocol.py answers
"is the code still correct?", this module answers "how fast is it and
how good are the outputs?". Results are metrics, not pass/fail.

Design notes
------------

**Why a separate harness?** Benchmarks need a different output shape
(numbers + units, not green ticks), a different cadence (run weekly or
before a release, not every PR), and a different failure mode (a
benchmark "failure" means the metric drifted, not that the code is
broken). Forcing both concerns into the pass/fail L1/L2/L3 harness
obscures regressions of either kind.

**Layers.**

- ``fast`` — pure descriptive stats from the live DB + Qdrant. No
  embedding passes, no LLM calls. Runs in seconds. Safe to run as often
  as you want.
- ``live`` — adds one embedder pass and one hybrid_search round trip.
  Needs the bge-m3 model in VRAM (loaded on first call; ~5 s cold).
- ``llm`` — adds Ollama generation throughput. Slow (a single 200-token
  sample per model) but catches infrastructure regressions (model
  swapped out, host unreachable, 10× slower than baseline).
- ``full`` — every bench function. Order matches cost.

**Output.** Each run writes one JSONL file at
``data/bench/<UTC-ts>.jsonl`` with one record per metric, plus a rollup
``data/bench/latest.json`` so later runs can diff against it. The
Rich-rendered console table is for humans; the JSONL is for tooling.

**Why "metrics" and not "tests"?** A bench function yields one or more
``BenchMetric`` records: ``{name, value, unit, note}``. No assertions.
Regressions are detected after the fact by comparing to ``latest.json``
(``--compare`` flag). This decouples "measuring" from "judging whether
the number is bad", because the threshold for "bad" is a moving
target (e.g. upgrading the GPU will halve every latency metric, and
that's a good thing, not a regression).
"""
from __future__ import annotations

import gc
import json
import logging
import os
import random
import statistics
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger("sciknow.bench")


# ── Result types ──────────────────────────────────────────────────────


@dataclass
class BenchMetric:
    """One measurement. Emitted by bench functions; many per function is OK."""
    name: str
    value: float | int | str | None
    unit: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchResult:
    """Aggregate of one bench function's output + timing/status."""
    name: str
    category: str
    layer: str
    duration_ms: int
    metrics: list[BenchMetric] = field(default_factory=list)
    status: str = "ok"            # ok | skipped | error
    message: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "layer": self.layer,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "message": self.message,
            "metrics": [m.as_dict() for m in self.metrics],
        }


# A bench function returns nothing and appends to a context object we
# hand in, or yields BenchMetric objects. We prefer the yield pattern
# because it lets a bench function report partial results before
# crashing on a later metric.


BenchFn = Callable[[], Iterable[BenchMetric]]


# ── Runner ────────────────────────────────────────────────────────────


def _run_bench(
    fn: BenchFn,
    *,
    category: str,
    layer: str,
) -> BenchResult:
    name = fn.__name__
    t0 = time.monotonic()
    metrics: list[BenchMetric] = []
    status = "ok"
    message = ""
    try:
        for m in fn() or ():
            if not isinstance(m, BenchMetric):
                raise TypeError(
                    f"{name}: yielded {type(m).__name__}, expected BenchMetric"
                )
            metrics.append(m)
    except SkipBench as exc:
        status = "skipped"
        message = str(exc)
    except Exception as exc:
        status = "error"
        tb = traceback.format_exc().splitlines()
        short = " | ".join(line.strip() for line in tb[-3:])
        message = f"{type(exc).__name__}: {exc}  ·  {short}"
        logger.exception("bench %s failed", name)
    finally:
        gc.collect()
    elapsed = int((time.monotonic() - t0) * 1000)
    return BenchResult(
        name=name, category=category, layer=layer,
        duration_ms=elapsed, metrics=metrics,
        status=status, message=message,
    )


class SkipBench(Exception):
    """Raise from a bench function to mark it skipped with a reason."""


def skip(reason: str) -> None:
    raise SkipBench(reason)


# ── JSONL persistence ────────────────────────────────────────────────


def _bench_dir() -> Path:
    from sciknow.config import settings
    d = Path(settings.data_dir) / "bench"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_results(results: list[BenchResult], *, tag: str = "") -> Path:
    """Append one JSONL file per run + update latest.json."""
    d = _bench_dir()
    slug = _timestamp_slug()
    out = d / f"{slug}.jsonl"
    with out.open("w") as f:
        header = {
            "_kind": "header",
            "timestamp": slug,
            "tag": tag,
            "n_benchmarks": len(results),
        }
        f.write(json.dumps(header) + "\n")
        for r in results:
            f.write(json.dumps(r.as_dict()) + "\n")
    latest = d / "latest.json"
    with latest.open("w") as f:
        json.dump({
            "timestamp": slug,
            "tag": tag,
            "file": str(out.relative_to(d.parent.parent)) if _relative_safe(out) else str(out),
            "results": [r.as_dict() for r in results],
        }, f, indent=2)
    return out


def _relative_safe(p: Path) -> bool:
    try:
        p.relative_to(Path.cwd())
        return True
    except Exception:
        return False


def load_latest() -> dict | None:
    f = _bench_dir() / "latest.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def diff_against_latest(current: list[BenchResult]) -> dict[str, dict]:
    """Return {metric_key: {"new": v, "old": v, "delta": v, "delta_pct": v}}.

    Only compares numeric metrics. Missing keys on either side are
    reported as None.
    """
    latest = load_latest()
    if not latest:
        return {}
    old_metrics: dict[str, float] = {}
    for r in latest.get("results", []):
        for m in r.get("metrics", []):
            v = m.get("value")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                old_metrics[f"{r['name']}::{m['name']}"] = float(v)

    out: dict[str, dict] = {}
    for r in current:
        for m in r.metrics:
            v = m.value
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            key = f"{r.name}::{m.name}"
            old = old_metrics.get(key)
            entry: dict[str, Any] = {"new": float(v), "old": old}
            if old is not None and old != 0:
                entry["delta"] = float(v) - old
                entry["delta_pct"] = 100.0 * (float(v) - old) / old
            out[key] = entry
    return out


# ── Console rendering ────────────────────────────────────────────────


def render_report(
    results: list[BenchResult],
    *,
    diff: dict[str, dict] | None = None,
) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    # Group by category for readable output
    by_cat: dict[str, list[BenchResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    for cat, rs in by_cat.items():
        table = Table(title=f"[bold]{cat}[/bold]", box=box.SIMPLE_HEAD, expand=True)
        table.add_column("", width=2)
        table.add_column("Bench", style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Unit")
        table.add_column("Δ vs last", justify="right", style="magenta")
        table.add_column("Time", justify="right", width=8)
        table.add_column("Note", style="dim", overflow="fold")

        for r in rs:
            if r.status == "skipped":
                mark = "[yellow]~[/yellow]"
            elif r.status == "error":
                mark = "[red]✗[/red]"
            else:
                mark = "[green]✓[/green]"
            time_str = f"{r.duration_ms}ms" if r.duration_ms else "—"

            if not r.metrics:
                table.add_row(mark, r.name, "—", "—", "", "", time_str, r.message or "")
                continue
            first = True
            for m in r.metrics:
                diff_str = ""
                if diff is not None:
                    key = f"{r.name}::{m.name}"
                    d = diff.get(key)
                    if d and d.get("delta_pct") is not None:
                        sign = "+" if d["delta_pct"] >= 0 else ""
                        diff_str = f"{sign}{d['delta_pct']:.1f}%"
                v = m.value
                if isinstance(v, float):
                    v_str = f"{v:,.3f}" if abs(v) < 1000 else f"{v:,.1f}"
                elif isinstance(v, int):
                    v_str = f"{v:,}"
                else:
                    v_str = str(v)
                table.add_row(
                    mark if first else "",
                    r.name if first else "",
                    m.name, v_str, m.unit, diff_str,
                    time_str if first else "",
                    m.note,
                )
                first = False
        console.print(table)

    # Summary line
    n_ok      = sum(1 for r in results if r.status == "ok")
    n_skip    = sum(1 for r in results if r.status == "skipped")
    n_err     = sum(1 for r in results if r.status == "error")
    n_metrics = sum(len(r.metrics) for r in results)
    console.print(
        f"[bold]Total[/bold]: {n_ok}/{len(results)} ran · {n_skip} skipped · "
        f"{n_err} errored · {n_metrics} metric{'s' if n_metrics != 1 else ''}"
    )


# ── The bench functions themselves ────────────────────────────────────
#
# Each function is zero-arg and yields BenchMetric objects. Grouped by
# category below. The LAYERS dict at the bottom dispatches them.


# ════════════════════════════════════════════════════════════════════
# Category: corpus (descriptive stats about the current library)
# ════════════════════════════════════════════════════════════════════


def b_corpus_sizes() -> Iterable[BenchMetric]:
    """Top-line counts: documents, chunks, citations, wiki pages, drafts."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        n_docs     = session.execute(text("SELECT COUNT(*) FROM documents")).scalar() or 0
        n_complete = session.execute(text("SELECT COUNT(*) FROM documents WHERE ingestion_status = 'complete'")).scalar() or 0
        n_chunks   = session.execute(text("SELECT COUNT(*) FROM chunks")).scalar() or 0
        n_cits     = session.execute(text("SELECT COUNT(*) FROM citations")).scalar() or 0
        n_cits_lk  = session.execute(text("SELECT COUNT(*) FROM citations WHERE cited_document_id IS NOT NULL")).scalar() or 0
        try:
            n_wiki = session.execute(text("SELECT COUNT(*) FROM wiki_pages")).scalar() or 0
        except Exception:
            n_wiki = 0
        try:
            n_drafts = session.execute(text("SELECT COUNT(*) FROM drafts")).scalar() or 0
            n_books  = session.execute(text("SELECT COUNT(*) FROM books")).scalar() or 0
        except Exception:
            n_drafts = 0
            n_books  = 0

    yield BenchMetric("documents",     n_docs,     "papers")
    yield BenchMetric("documents_complete", n_complete, "papers",
                      note=f"{(100*n_complete/n_docs):.1f}% of library" if n_docs else "")
    yield BenchMetric("chunks",        n_chunks,   "chunks")
    yield BenchMetric("chunks_per_doc",
                      round(n_chunks / n_complete, 1) if n_complete else 0,
                      "avg",
                      note="after cleaning — excludes references + ack sections")
    yield BenchMetric("citations",     n_cits,     "edges")
    yield BenchMetric("citations_linked", n_cits_lk, "edges",
                      note=f"{(100*n_cits_lk/n_cits):.1f}% cross-linked to corpus" if n_cits else "")
    yield BenchMetric("wiki_pages",    n_wiki,     "pages")
    yield BenchMetric("books",         n_books,    "books")
    yield BenchMetric("drafts",        n_drafts,   "drafts")


def b_corpus_metadata_source_mix() -> Iterable[BenchMetric]:
    """Which layer of the 4-layer metadata cascade is populating metadata?

    Low ``crossref`` + ``arxiv`` share implies the LLM-extracted or
    embedded-PDF path is doing the work, which is slower and less
    reliable. A good-quality corpus should be majority crossref/arxiv.
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        rows = session.execute(text("""
            SELECT metadata_source, COUNT(*)
            FROM paper_metadata
            WHERE metadata_source IS NOT NULL
            GROUP BY metadata_source
            ORDER BY 2 DESC
        """)).fetchall()
        n_total = session.execute(text("SELECT COUNT(*) FROM paper_metadata")).scalar() or 0
    if n_total == 0:
        skip("no paper_metadata rows")
    for src, n in rows:
        yield BenchMetric(
            f"source_{src or 'unknown'}", n, "papers",
            note=f"{(100 * n / n_total):.1f}%",
        )
    # Derive "authoritative" = crossref + arxiv + embedded_pdf with a DOI
    auth = sum(n for s, n in rows if s in ("crossref", "arxiv"))
    yield BenchMetric(
        "authoritative_ratio", round(100 * auth / n_total, 1), "%",
        note="share of rows whose metadata came from crossref/arxiv (DOI-grounded)",
    )


def b_corpus_chunk_stats() -> Iterable[BenchMetric]:
    """Chunk-length distribution — spots under/over-long chunks that
    hurt retrieval (too short = no context; too long = diluted signal)."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        rows = session.execute(text("""
            SELECT COALESCE(content_tokens, 0)
            FROM chunks
            LIMIT 20000
        """)).fetchall()
    if not rows:
        skip("no chunks")
    vals = [r[0] for r in rows if r[0]]
    if not vals:
        skip("no token_count data")
    vals.sort()
    yield BenchMetric("sample_size",  len(vals), "chunks")
    yield BenchMetric("tokens_p50",   vals[len(vals)//2], "tokens",
                      note="median chunk length")
    yield BenchMetric("tokens_p90",   vals[int(0.9*len(vals))], "tokens")
    yield BenchMetric("tokens_p99",   vals[int(0.99*len(vals))], "tokens")
    yield BenchMetric("tokens_mean",  round(statistics.mean(vals), 1), "tokens")
    yield BenchMetric("tokens_stdev", round(statistics.stdev(vals), 1) if len(vals) > 1 else 0, "tokens")
    under_100 = sum(1 for v in vals if v < 100)
    over_1500 = sum(1 for v in vals if v > 1500)
    yield BenchMetric("pct_under_100_tok", round(100 * under_100 / len(vals), 1), "%",
                      note="very short chunks — low retrieval value")
    yield BenchMetric("pct_over_1500_tok", round(100 * over_1500 / len(vals), 1), "%",
                      note="bloated chunks — semantic dilution risk")


def b_corpus_section_coverage() -> Iterable[BenchMetric]:
    """How many papers have each canonical section type? A healthy corpus
    should have abstract / introduction / methods / results / discussion
    on >= 80% of papers."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    canonical = ["abstract", "introduction", "methods", "results", "discussion",
                 "conclusion", "related_work"]
    with get_session() as session:
        n_docs = session.execute(text("""
            SELECT COUNT(DISTINCT document_id) FROM chunks
        """)).scalar() or 0
        if n_docs == 0:
            skip("no chunks")
        for st in canonical:
            n = session.execute(text("""
                SELECT COUNT(DISTINCT document_id) FROM chunks
                WHERE section_type = :st
            """), {"st": st}).scalar() or 0
            yield BenchMetric(
                f"pct_with_{st}", round(100 * n / n_docs, 1), "%",
                note=f"{n} / {n_docs} papers have this section",
            )


def b_corpus_year_distribution() -> Iterable[BenchMetric]:
    """Recency skew. Too recent = missing foundational work; too old =
    missing SOTA. Report min/median/max year + pct since 2020."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        rows = session.execute(text("""
            SELECT year FROM paper_metadata WHERE year IS NOT NULL AND year > 0
        """)).fetchall()
    years = [r[0] for r in rows]
    if not years:
        skip("no year metadata")
    years.sort()
    yield BenchMetric("year_min",    years[0], "year")
    yield BenchMetric("year_median", years[len(years)//2], "year")
    yield BenchMetric("year_max",    years[-1], "year")
    yield BenchMetric(
        "pct_since_2020",
        round(100 * sum(1 for y in years if y >= 2020) / len(years), 1),
        "%",
    )
    yield BenchMetric(
        "pct_before_2000",
        round(100 * sum(1 for y in years if y < 2000) / len(years), 1),
        "%",
    )


# ════════════════════════════════════════════════════════════════════
# Category: qdrant (index health, not retrieval quality)
# ════════════════════════════════════════════════════════════════════


def b_qdrant_collection_stats() -> Iterable[BenchMetric]:
    """Qdrant points per collection + (dense vs sparse) coverage."""
    from sciknow.storage.qdrant import (
        get_client, papers_collection, abstracts_collection, wiki_collection,
    )
    client = get_client()
    for label, name in (
        ("papers",    papers_collection()),
        ("abstracts", abstracts_collection()),
        ("wiki",      wiki_collection()),
    ):
        try:
            info = client.get_collection(name)
            yield BenchMetric(f"{label}_points", info.points_count or 0, "pts")
            yield BenchMetric(f"{label}_indexed", info.indexed_vectors_count or 0, "pts")
        except Exception as exc:
            yield BenchMetric(f"{label}_error", str(exc)[:80], "")


def b_qdrant_raptor_levels() -> Iterable[BenchMetric]:
    """If RAPTOR is built, count summary nodes per level."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    from sciknow.storage.qdrant import get_client, papers_collection
    client = get_client()
    name = papers_collection()
    any_found = False
    for lvl in (0, 1, 2, 3, 4):
        try:
            info = client.count(
                collection_name=name,
                count_filter=Filter(must=[
                    FieldCondition(key="node_level", match=MatchValue(value=lvl))
                ]),
                exact=False,
            )
            n = info.count if hasattr(info, "count") else int(info)
            if n > 0:
                any_found = True
                yield BenchMetric(f"raptor_L{lvl}", n, "nodes")
        except Exception:
            pass
    if not any_found:
        skip("no RAPTOR summary nodes (run `sciknow catalog raptor build`)")


# ════════════════════════════════════════════════════════════════════
# Category: retrieval (hybrid_search latency + quality probes)
# ════════════════════════════════════════════════════════════════════


# A small set of domain-general probe queries — deliberately generic so
# this bench works against any sciknow project. For project-specific
# queries, a future extension could read from a fixture JSON in the
# project directory.
_PROBE_QUERIES = [
    "what are the main findings",
    "methodology used for the experiments",
    "limitations of the approach",
    "comparison with prior work",
    "dataset characteristics and size",
    "computational cost and scalability",
    "ablation study results",
    "future work directions",
]


def b_retrieval_hybrid_latency() -> Iterable[BenchMetric]:
    """Wall time for hybrid_search top-50 on a probe-query set. Measures
    the whole pipeline (dense + sparse + FTS + RRF fusion), not any one
    signal. Reports p50/p90/mean across queries.

    Warms up with one off-the-clock query so the first cold-model load
    (~5–8 s on a 3090) doesn't contaminate the mean. Phase 44.1 bench
    baseline previously recorded mean 1083 ms vs p50 68 ms purely
    because of this cold-start effect.
    """
    from sciknow.retrieval.hybrid_search import search
    from sciknow.storage.db  import get_session
    from sciknow.storage.qdrant import get_client
    client = get_client()
    timings = []
    result_counts = []
    with get_session() as session:
        # Warmup — load models, warm JIT caches. NOT counted.
        try:
            _ = search("warmup probe", client, session, candidate_k=10)
        except Exception as exc:
            logger.debug("warmup query failed: %s", exc)
        t_total_start = time.monotonic()
        for q in _PROBE_QUERIES:
            t0 = time.monotonic()
            try:
                cands = search(q, client, session, candidate_k=50)
            except Exception as exc:
                logger.warning("probe %r failed: %s", q, exc)
                continue
            dt = (time.monotonic() - t0) * 1000
            timings.append(dt)
            result_counts.append(len(cands))
    total_ms = (time.monotonic() - t_total_start) * 1000
    if not timings:
        skip("all probe queries failed")
    timings.sort()
    yield BenchMetric("queries", len(timings), "queries")
    yield BenchMetric("latency_p50", round(timings[len(timings)//2], 1), "ms")
    yield BenchMetric("latency_p90", round(timings[int(0.9*(len(timings)-1))], 1), "ms")
    yield BenchMetric("latency_mean", round(statistics.mean(timings), 1), "ms")
    yield BenchMetric("total_wall_ms", round(total_ms, 1), "ms")
    yield BenchMetric("avg_candidates", round(statistics.mean(result_counts), 1), "hits")


def b_retrieval_rerank_mrr_shift() -> Iterable[BenchMetric]:
    """How much does the cross-encoder reorder the top-10? Large shift
    = fusion is returning rough candidates that the cross-encoder
    rescues. Small shift = fusion is already near-optimal; the reranker
    is buying little."""
    from sciknow.retrieval.hybrid_search import search
    from sciknow.retrieval.reranker import rerank
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client
    client = get_client()
    mrr_shifts = []
    top1_changes = 0
    rerank_latencies = []
    n_runs = 0
    with get_session() as session:
        # Trim to 4 queries to keep rerank cost bounded (bge-reranker-v2-m3
        # on 50 candidates per query is ~1 s cold, ~200 ms hot).
        for q in _PROBE_QUERIES[:4]:
            try:
                cands = search(q, client, session, candidate_k=50)
            except Exception:
                continue
            if not cands:
                continue
            pre_order = [id(c) for c in cands[:10]]
            t0 = time.monotonic()
            reranked = rerank(q, cands, top_k=10)
            rerank_latencies.append((time.monotonic() - t0) * 1000)
            post_order = [id(c) for c in reranked]
            if pre_order and post_order and pre_order[0] != post_order[0]:
                top1_changes += 1
            # Normalized "displacement": how far each top-10 item moved
            # on average after reranking.
            disp = 0.0
            pre_idx = {x: i for i, x in enumerate(pre_order)}
            for new_i, x in enumerate(post_order):
                old_i = pre_idx.get(x, 10)
                disp += abs(new_i - old_i)
            mrr_shifts.append(disp / max(1, len(post_order)))
            n_runs += 1
    if n_runs == 0:
        skip("no candidates for any probe query")
    yield BenchMetric("queries",             n_runs, "queries")
    yield BenchMetric("avg_displacement",    round(statistics.mean(mrr_shifts), 2), "positions",
                      note="avg |new_rank - old_rank| on top-10; small = fusion ≈ optimal")
    yield BenchMetric("top1_change_rate",    round(100 * top1_changes / n_runs, 1), "%",
                      note="how often reranker picks a different #1")
    yield BenchMetric("rerank_latency_mean", round(statistics.mean(rerank_latencies), 1), "ms")


def b_retrieval_signal_overlap() -> Iterable[BenchMetric]:
    """How much does dense, sparse, and FTS agree? Low overlap means the
    three signals are genuinely complementary (RRF earns its keep).
    High overlap means you could probably drop one without loss."""
    from qdrant_client.models import SparseVector
    from sciknow.retrieval.hybrid_search import _postgres_fts
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client, papers_collection
    from sciknow.ingestion.embedder import _get_model

    model = _get_model()
    client = get_client()
    overlaps_dense_sparse = []
    overlaps_dense_fts    = []
    overlaps_sparse_fts   = []
    with get_session() as session:
        for q in _PROBE_QUERIES[:5]:
            enc = model.encode([q], batch_size=1, max_length=512,
                               return_dense=True, return_sparse=True,
                               return_colbert_vecs=False)
            dense_vec = enc["dense_vecs"][0].tolist()
            sparse_raw = enc["lexical_weights"][0]
            sv = SparseVector(
                indices=[int(k) for k in sparse_raw.keys()],
                values=[float(v) for v in sparse_raw.values()],
            )
            try:
                dense_resp = client.query_points(
                    collection_name=papers_collection(),
                    query=dense_vec, using="dense", limit=50,
                )
                sparse_resp = client.query_points(
                    collection_name=papers_collection(),
                    query=sv, using="sparse", limit=50,
                )
                fts_hits = _postgres_fts(
                    session, q, 50,
                    year_from=None, year_to=None,
                    domain=None, section=None, topic_cluster=None,
                )
            except Exception as exc:
                logger.warning("signal overlap probe failed: %s", exc)
                continue
            dense_ids  = {str(p.id) for p in dense_resp.points}
            sparse_ids = {str(p.id) for p in sparse_resp.points}
            fts_ids    = {str(pid) for pid in fts_hits}
            def jacc(a: set, b: set) -> float:
                if not a and not b:
                    return 1.0
                u = a | b
                return len(a & b) / len(u) if u else 0.0
            overlaps_dense_sparse.append(jacc(dense_ids, sparse_ids))
            overlaps_dense_fts.append(jacc(dense_ids, fts_ids))
            overlaps_sparse_fts.append(jacc(sparse_ids, fts_ids))
    if not overlaps_dense_sparse:
        skip("no probes completed")
    yield BenchMetric("jaccard_dense_sparse",
                      round(statistics.mean(overlaps_dense_sparse), 3), "",
                      note="top-50 overlap, dense vs sparse; 0=disjoint, 1=identical")
    yield BenchMetric("jaccard_dense_fts",
                      round(statistics.mean(overlaps_dense_fts), 3), "",
                      note="dense vs PostgreSQL tsvector")
    yield BenchMetric("jaccard_sparse_fts",
                      round(statistics.mean(overlaps_sparse_fts), 3), "",
                      note="sparse vs FTS — both lexical, so should be higher")
    # Complementarity score: how different are the three signals on
    # average? Lower Jaccard -> more complementary -> RRF fusion buys
    # more than any single signal.
    mean_all = statistics.mean(
        overlaps_dense_sparse + overlaps_dense_fts + overlaps_sparse_fts
    )
    yield BenchMetric("mean_overlap",  round(mean_all, 3), "",
                      note="lower is better (more complementary signals)")


# ════════════════════════════════════════════════════════════════════
# Category: models (raw speed, not quality)
# ════════════════════════════════════════════════════════════════════


def b_model_embedder_throughput() -> Iterable[BenchMetric]:
    """Raw bge-m3 encode throughput on real chunk text, bypassing
    Qdrant. Measures the VRAM-heavy part."""
    from sqlalchemy import text
    from sciknow.ingestion.embedder import _get_model
    from sciknow.storage.db import get_session
    # Sample 64 real chunks with realistic length distribution
    with get_session() as session:
        rows = session.execute(text("""
            SELECT content FROM chunks
            WHERE length(content) BETWEEN 200 AND 4000
            ORDER BY random() LIMIT 64
        """)).fetchall()
    texts = [r[0] for r in rows]
    if len(texts) < 16:
        skip(f"need 16+ sample chunks, got {len(texts)}")

    model = _get_model()
    # Warmup — first pass builds tokenizer caches
    _ = model.encode(texts[:4], batch_size=4, max_length=8192,
                     return_dense=True, return_sparse=True,
                     return_colbert_vecs=False)
    total_chars = sum(len(t) for t in texts)
    t0 = time.monotonic()
    _ = model.encode(texts, batch_size=16, max_length=8192,
                     return_dense=True, return_sparse=True,
                     return_colbert_vecs=False)
    dt = time.monotonic() - t0
    yield BenchMetric("batch_size",  16,        "")
    yield BenchMetric("total_chunks", len(texts), "")
    yield BenchMetric("wall_time",   round(dt, 2), "s")
    yield BenchMetric("chunks_per_sec",
                      round(len(texts) / dt, 1), "chunks/s",
                      note="dense + sparse in one forward pass")
    yield BenchMetric("chars_per_sec",
                      int(total_chars / dt), "char/s")


def b_model_reranker_throughput() -> Iterable[BenchMetric]:
    """bge-reranker-v2-m3 pairs/second on real (query, chunk) pairs."""
    from sqlalchemy import text
    from sciknow.retrieval.reranker import _get_reranker
    from sciknow.storage.db import get_session
    with get_session() as session:
        rows = session.execute(text("""
            SELECT content FROM chunks
            WHERE length(content) BETWEEN 200 AND 4000
            ORDER BY random() LIMIT 50
        """)).fetchall()
    chunks = [r[0] for r in rows]
    if len(chunks) < 20:
        skip("need 20+ chunks")
    reranker = _get_reranker()
    # Warmup
    _ = reranker.compute_score([["what is this", chunks[0]]], normalize=True)
    pairs = [["what are the main findings", c] for c in chunks]
    t0 = time.monotonic()
    _ = reranker.compute_score(pairs, normalize=True)
    dt = time.monotonic() - t0
    yield BenchMetric("pairs",          len(pairs), "pairs")
    yield BenchMetric("wall_time",      round(dt, 2), "s")
    yield BenchMetric("pairs_per_sec",  round(len(pairs) / dt, 1), "pairs/s")


def b_model_llm_fast_throughput() -> Iterable[BenchMetric]:
    """Ollama fast-model tokens/sec (LLM_FAST_MODEL, for metadata extract).
    A single short generation call — fast models are used inside the
    ingestion hot path so latency dominates over throughput."""
    from sciknow.config import settings
    from sciknow.rag.llm import stream
    prompt_user = (
        "Respond with 2-3 sentences on why scientific papers "
        "should have a methods section. Be concrete."
    )
    tokens: list[str] = []
    t0 = time.monotonic()
    try:
        for tok in stream("You are helpful.", prompt_user,
                          model=settings.llm_fast_model, num_ctx=2048):
            tokens.append(tok)
            if time.monotonic() - t0 > 30:
                break
    except Exception as exc:
        skip(f"fast LLM unreachable: {exc}")
    dt = time.monotonic() - t0
    yield BenchMetric("model",       settings.llm_fast_model, "")
    yield BenchMetric("tokens",      len(tokens), "")
    yield BenchMetric("wall_time",   round(dt, 2), "s")
    yield BenchMetric("tokens_per_sec", round(len(tokens) / dt, 1), "tok/s")


def b_model_llm_main_throughput() -> Iterable[BenchMetric]:
    """Ollama main-model tokens/sec (used for book writing). A single
    short generation — on a cold model this includes load time, so the
    number is pessimistic."""
    from sciknow.config import settings
    from sciknow.rag.llm import stream
    prompt_user = (
        "Write a 2-paragraph critique of the hypothesis that "
        "solar irradiance variability explains recent climate trends. "
        "Be specific about what evidence would be needed."
    )
    tokens: list[str] = []
    t0 = time.monotonic()
    try:
        for tok in stream("You are a scientific writer.", prompt_user,
                          model=settings.llm_model, num_ctx=4096):
            tokens.append(tok)
            if len(tokens) > 200:
                break
            if time.monotonic() - t0 > 120:
                break
    except Exception as exc:
        skip(f"main LLM unreachable: {exc}")
    dt = time.monotonic() - t0
    yield BenchMetric("model",          settings.llm_model, "")
    yield BenchMetric("tokens",         len(tokens), "")
    yield BenchMetric("wall_time",      round(dt, 2), "s")
    yield BenchMetric("tokens_per_sec", round(len(tokens) / dt, 1), "tok/s",
                      note="includes cold-model load on first call")


# ════════════════════════════════════════════════════════════════════
# Category: books (generation quality from existing drafts + logs)
# ════════════════════════════════════════════════════════════════════


def b_books_draft_stats() -> Iterable[BenchMetric]:
    """Distribution over draft word counts + versions — catches books
    where chapter length target is being ignored."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        rows = session.execute(text("""
            SELECT word_count, version FROM drafts
            WHERE word_count IS NOT NULL AND word_count > 0
        """)).fetchall()
    if not rows:
        skip("no drafts")
    wcs = [r[0] for r in rows]
    versions = [r[1] or 1 for r in rows]
    wcs.sort()
    yield BenchMetric("drafts",           len(wcs), "")
    yield BenchMetric("words_total",      sum(wcs), "words")
    yield BenchMetric("words_p50",        wcs[len(wcs)//2], "words")
    yield BenchMetric("words_p90",        wcs[int(0.9*(len(wcs)-1))], "words")
    yield BenchMetric("words_max",        wcs[-1], "words")
    yield BenchMetric("mean_versions",    round(statistics.mean(versions), 2), "")
    # "churn" = drafts that were revised many times
    heavy = sum(1 for v in versions if v >= 3)
    yield BenchMetric("pct_heavily_revised",
                      round(100 * heavy / len(versions), 1), "%",
                      note="drafts with version >= 3")


def b_books_groundedness_distribution() -> Iterable[BenchMetric]:
    """If custom_metadata stores verification scores, report the
    distribution. Pulls from any draft that has groundedness_score or
    a verification block."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        rows = session.execute(text("""
            SELECT custom_metadata FROM drafts
            WHERE custom_metadata IS NOT NULL
        """)).fetchall()
    scores: list[float] = []
    supported = 0
    contradicted = 0
    not_in_sources = 0
    unverifiable = 0
    for (meta,) in rows:
        if not isinstance(meta, dict):
            continue
        # Common storage shapes: meta["verification"]["groundedness_score"]
        # and meta["verification"]["claims"] with verdicts.
        v = meta.get("verification") or meta.get("verify") or {}
        if isinstance(v, dict):
            gs = v.get("groundedness_score")
            if isinstance(gs, (int, float)):
                scores.append(float(gs))
            claims = v.get("claims") or []
            for c in claims if isinstance(claims, list) else []:
                verdict = (c.get("verdict") if isinstance(c, dict) else "") or ""
                verdict = verdict.upper()
                if verdict == "SUPPORTED":
                    supported += 1
                elif verdict == "CONTRADICTED":
                    contradicted += 1
                elif verdict == "NOT_IN_SOURCES":
                    not_in_sources += 1
                elif verdict == "UNVERIFIABLE":
                    unverifiable += 1
    if not scores and (supported + contradicted + not_in_sources + unverifiable) == 0:
        skip("no verification metadata on drafts")
    if scores:
        scores.sort()
        yield BenchMetric("n_verified_drafts", len(scores), "")
        yield BenchMetric("groundedness_p50",
                          round(scores[len(scores)//2], 3), "")
        yield BenchMetric("groundedness_mean",
                          round(statistics.mean(scores), 3), "")
        yield BenchMetric("groundedness_min",
                          round(scores[0], 3), "")
    total_claims = supported + contradicted + not_in_sources + unverifiable
    if total_claims:
        yield BenchMetric("claims_total",         total_claims, "")
        yield BenchMetric("pct_supported",        round(100*supported/total_claims, 1), "%")
        yield BenchMetric("pct_contradicted",     round(100*contradicted/total_claims, 1), "%")
        yield BenchMetric("pct_not_in_sources",   round(100*not_in_sources/total_claims, 1), "%",
                          note="lower is better (reviewer said the claim isn't in the retrieved evidence)")
        yield BenchMetric("pct_unverifiable",     round(100*unverifiable/total_claims, 1), "%")


def b_books_citation_density() -> Iterable[BenchMetric]:
    """Words per inline citation. Too low = citation-spam; too high =
    ungrounded."""
    import re
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    cite_re = re.compile(r"\[\s*\d+[\s,\-\d]*\]")
    with get_session() as session:
        rows = session.execute(text("""
            SELECT content, word_count FROM drafts
            WHERE content IS NOT NULL AND word_count > 200
        """)).fetchall()
    if not rows:
        skip("no drafts with >200 words")
    densities = []
    no_cite = 0
    for content, wc in rows:
        n_cites = len(cite_re.findall(content or ""))
        if n_cites == 0:
            no_cite += 1
            continue
        densities.append((wc or 0) / n_cites)
    yield BenchMetric("drafts_sampled",  len(rows), "")
    yield BenchMetric("drafts_no_cites", no_cite, "",
                      note="drafts with zero [N]-style citations")
    if densities:
        densities.sort()
        yield BenchMetric("words_per_cite_p50",
                          round(densities[len(densities)//2], 1), "words/cite")
        yield BenchMetric("words_per_cite_mean",
                          round(statistics.mean(densities), 1), "words/cite")


def b_autowrite_log_convergence() -> Iterable[BenchMetric]:
    """Parse autowrite JSONL logs under data/autowrite/ for historical
    convergence behaviour: how many rounds until score plateau, mean
    final scores, early-stop rate."""
    from sciknow.config import settings
    log_dir = Path(settings.data_dir) / "autowrite"
    if not log_dir.exists():
        skip(f"no autowrite log dir at {log_dir}")
    files = sorted(log_dir.glob("*.jsonl"))
    if not files:
        skip("no autowrite runs")
    n_runs = 0
    rounds_until_plateau: list[int] = []
    early_stops = 0
    final_scores: list[float] = []
    for f in files[-100:]:   # last 100 runs to bound parsing cost
        scores_seen: list[float] = []
        early = False
        try:
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                kind = ev.get("kind") or ""
                if kind == "scores":
                    ov = ev.get("overall")
                    if isinstance(ov, (int, float)):
                        scores_seen.append(float(ov))
                elif kind in ("early_stop", "converged"):
                    early = True
                elif kind == "end":
                    pass
        except Exception:
            continue
        if not scores_seen:
            continue
        n_runs += 1
        if early:
            early_stops += 1
        final_scores.append(scores_seen[-1])
        # Plateau = first round i where scores[i] - scores[i-1] < 0.05
        for i in range(1, len(scores_seen)):
            if scores_seen[i] - scores_seen[i-1] < 0.05:
                rounds_until_plateau.append(i)
                break
        else:
            rounds_until_plateau.append(len(scores_seen))
    if n_runs == 0:
        skip("no autowrite runs with score events")
    yield BenchMetric("runs_parsed", n_runs, "runs")
    if final_scores:
        yield BenchMetric("final_score_mean", round(statistics.mean(final_scores), 3), "")
        yield BenchMetric("final_score_p50",  round(sorted(final_scores)[len(final_scores)//2], 3), "")
    if rounds_until_plateau:
        yield BenchMetric("rounds_to_plateau_p50",
                          round(statistics.median(rounds_until_plateau), 1), "rounds")
    yield BenchMetric("early_stop_rate", round(100 * early_stops / n_runs, 1), "%",
                      note="runs that stopped before max rounds")


# ── Phase 51.2 — new fast benches ───────────────────────────────────
# Three additions covering gaps the bench harness didn't previously
# measure: (1) KG triple health, (2) chunk-size distribution, and
# (3) PDF-converter backend mix. All are SQL-only → fast layer.


def b_kg_quality() -> Iterable[BenchMetric]:
    """Triples per paper, source-sentence coverage, predicate family mix.

    Surfaces a few known failure modes of `wiki compile`:
      - triples_per_paper very low (<3) → extraction prompt is truncating
      - source_sentence_coverage low → pre-0019 triples or model wasn't
        honouring the "copy exactly" instruction in the prompt
      - predicate diversity low → the LLM is collapsing everything into
        a generic 'related_to'
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        try:
            n_triples = session.execute(
                text("SELECT COUNT(*) FROM knowledge_graph")
            ).scalar() or 0
        except Exception:
            yield BenchMetric("kg_present", "no", note="knowledge_graph table missing")
            return
        n_papers_with_triples = session.execute(text("""
            SELECT COUNT(DISTINCT source_doc_id) FROM knowledge_graph
        """)).scalar() or 0
        n_with_sent = session.execute(text("""
            SELECT COUNT(*) FROM knowledge_graph
            WHERE source_sentence IS NOT NULL AND length(source_sentence) > 0
        """)).scalar() or 0
        n_distinct_pred = session.execute(text("""
            SELECT COUNT(DISTINCT predicate) FROM knowledge_graph
        """)).scalar() or 0

    yield BenchMetric("kg_triples_total", n_triples, "triples")
    yield BenchMetric("kg_papers_with_triples", n_papers_with_triples, "papers")
    if n_papers_with_triples:
        yield BenchMetric(
            "kg_triples_per_paper_avg",
            round(n_triples / n_papers_with_triples, 1),
            "avg",
        )
    if n_triples:
        yield BenchMetric(
            "kg_source_sentence_coverage",
            round(100 * n_with_sent / n_triples, 1),
            "%",
            note="pre-0019 rows have NULL; re-compile wiki to backfill",
        )
    yield BenchMetric("kg_distinct_predicates", n_distinct_pred, "predicates")


def b_corpus_chunk_size_distribution() -> Iterable[BenchMetric]:
    """Token-length percentiles across all chunks. Regressions here
    usually trace to changes in the per-section _PARAMS tuning — e.g.
    dropping overlap or widening the target too aggressively wrecks
    retrieval recall."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        rows = session.execute(text("""
            SELECT
                percentile_cont(0.5) WITHIN GROUP (ORDER BY content_tokens) AS p50,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY content_tokens) AS p95,
                MAX(content_tokens) AS max_tc,
                MIN(content_tokens) AS min_tc,
                COUNT(*) AS n
            FROM chunks
            WHERE content_tokens IS NOT NULL
        """)).fetchone()
    if not rows or not rows[4]:
        yield BenchMetric("chunks_with_token_count", 0, "chunks",
                          note="no token_count populated — check chunker")
        return
    yield BenchMetric("chunks_token_p50", int(rows[0] or 0), "tokens")
    yield BenchMetric("chunks_token_p95", int(rows[1] or 0), "tokens")
    yield BenchMetric("chunks_token_min", int(rows[3] or 0), "tokens")
    yield BenchMetric("chunks_token_max", int(rows[2] or 0), "tokens")


def b_pdf_backend_mix() -> Iterable[BenchMetric]:
    """Which PDF converter backend produced each complete document?
    Inferred from `mineru_output_path`: set → MinerU (primary), NULL →
    Marker (legacy fallback). A sudden swing toward Marker usually
    means MinerU is failing silently — worth surfacing weekly."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        n_mineru = session.execute(text("""
            SELECT COUNT(*) FROM documents
            WHERE ingestion_status = 'complete'
              AND mineru_output_path IS NOT NULL
        """)).scalar() or 0
        n_marker = session.execute(text("""
            SELECT COUNT(*) FROM documents
            WHERE ingestion_status = 'complete'
              AND mineru_output_path IS NULL
        """)).scalar() or 0
    total = n_mineru + n_marker or 1
    yield BenchMetric(
        "pdf_backend_mineru",
        round(100 * n_mineru / total, 1),
        "%", note=f"{n_mineru} papers",
    )
    yield BenchMetric(
        "pdf_backend_marker_fallback",
        round(100 * n_marker / total, 1),
        "%", note=f"{n_marker} papers",
    )


# ════════════════════════════════════════════════════════════════════
# Layer registry
# ════════════════════════════════════════════════════════════════════


_FAST: list[tuple[str, BenchFn]] = [
    ("corpus", b_corpus_sizes),
    ("corpus", b_corpus_metadata_source_mix),
    ("corpus", b_corpus_chunk_stats),
    ("corpus", b_corpus_chunk_size_distribution),  # Phase 51.2
    ("corpus", b_corpus_section_coverage),
    ("corpus", b_corpus_year_distribution),
    ("corpus", b_pdf_backend_mix),                  # Phase 51.2
    ("qdrant", b_qdrant_collection_stats),
    ("qdrant", b_qdrant_raptor_levels),
    ("kg",     b_kg_quality),                       # Phase 51.2
    ("books",  b_books_draft_stats),
    ("books",  b_books_groundedness_distribution),
    ("books",  b_books_citation_density),
    ("books",  b_autowrite_log_convergence),
]

def _retrieval_recall_lazy() -> Iterable[BenchMetric]:
    """Phase 54.6.69 — lazy-import wrapper for retrieval_eval.b_retrieval_recall
    so the bench module doesn't pay the import cost when no probe set exists."""
    from sciknow.testing.retrieval_eval import b_retrieval_recall as _fn
    yield from _fn()


_LIVE: list[tuple[str, BenchFn]] = [
    ("retrieval", b_retrieval_hybrid_latency),
    ("retrieval", b_retrieval_rerank_mrr_shift),
    ("retrieval", b_retrieval_signal_overlap),
    ("retrieval", _retrieval_recall_lazy),      # Phase 54.6.69
    ("models",    b_model_embedder_throughput),
    ("models",    b_model_reranker_throughput),
]

_LLM: list[tuple[str, BenchFn]] = [
    ("models", b_model_llm_fast_throughput),
    ("models", b_model_llm_main_throughput),
]

def _sweep_layer() -> list[tuple[str, BenchFn]]:
    """Import sweep benches lazily — model_sweep imports wiki_prompts
    which has heavier transitive deps than bench.py wants to pay for
    every CLI invocation."""
    from sciknow.testing import model_sweep as _sw
    return list(_sw.SWEEP_BENCHES)


def _quality_layer() -> list[tuple[str, BenchFn]]:
    """Same lazy-import pattern as _sweep_layer — quality.py pulls in
    sentence-transformers + NLI model on first call which we don't
    want for every bench invocation."""
    from sciknow.testing import quality as _q
    return list(_q.QUALITY_BENCHES)


LAYERS: dict[str, list[tuple[str, BenchFn]]] = {
    "fast":  _FAST,
    "live":  _LIVE,
    "llm":   _LLM,
    "full":  _FAST + _LIVE + _LLM,
}

# "sweep" and "quality" are pseudo-layers: not stored in LAYERS (which
# would break the invariant that every layer has >= 1 bench fn) —
# resolved lazily in run_layer() via _sweep_layer() / _quality_layer().
VALID_LAYERS: tuple[str, ...] = tuple(LAYERS.keys()) + ("sweep", "quality")


def run_layer(layer: str) -> list[BenchResult]:
    if layer == "sweep":
        benches = _sweep_layer()
    elif layer == "quality":
        benches = _quality_layer()
    else:
        benches = LAYERS.get(layer)
    if benches is None:
        raise ValueError(f"unknown bench layer {layer!r}; pick from {list(VALID_LAYERS)}")
    results: list[BenchResult] = []
    for cat, fn in benches:
        results.append(_run_bench(fn, category=cat, layer=layer))
    return results


def run(layer: str = "fast", *, tag: str = "") -> tuple[list[BenchResult], Path]:
    """Run a layer, persist results, return (results, output_path)."""
    results = run_layer(layer)
    out = write_results(results, tag=tag or layer)
    return results, out
