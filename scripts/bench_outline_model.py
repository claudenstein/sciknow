"""Phase 54.6.297 — A/B the book-outline LLM against two candidates.

Runs the same 3-candidate tournament that `sciknow book outline`
runs (same prompt, same scorer) once per model, per --runs repeat,
and reports:

  * JSON-valid candidate count per run
  * Winner chapter count (the tournament's picked best)
  * Section variance of the winner (the 54.6.65 scorer reward)
  * Wall clock per full outline (3 candidates + scoring)

Purpose: decide whether to flip ``BOOK_OUTLINE_MODEL`` to a
non-default.  Motivated by the Phase 54.6.297 ship: the writer
got its own override (``BOOK_WRITE_MODEL=qwen3.6:27b-dense``) but
outline stayed on the global ``LLM_MODEL`` because nobody had
benched it.  Run this to check.

Usage::

    uv run python scripts/bench_outline_model.py              # 3 runs, both models
    uv run python scripts/bench_outline_model.py --runs 5
    uv run python scripts/bench_outline_model.py --book "My Book"
    uv run python scripts/bench_outline_model.py --models qwen3:30b-a3b-instruct-2507-q4_K_M,qwen3.6:27b-dense

Results written as JSONL at ``data/bench/outline_ab-<ts>.jsonl``
and summarised as a Rich table at the end.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from sciknow.config import settings
from sciknow.rag import prompts as rag_prompts
from sciknow.rag.llm import complete_with_status
from sciknow.storage.db import get_session
from sqlalchemy import text

console = Console()

DEFAULT_MODELS = [
    "qwen3:30b-a3b-instruct-2507-q4_K_M",   # current default
    "qwen3.6:27b-dense",                    # candidate
]
N_CANDIDATES = 3
TEMPS = (0.5, 0.65, 0.8)


def _score_candidate(chapters: list) -> float:
    """Same shape as the `sciknow book outline` scorer (54.6.65).

    Rewards chapter count + total section count + title uniqueness
    + section-count variance.  Kept identical so bench scores are
    comparable to what the real outline command picks.
    """
    n_ch = len(chapters)
    sec_counts = [len((c.get("sections") or [])) for c in chapters]
    n_sec = sum(sec_counts)
    unique_titles = len({c.get("title", "") for c in chapters})
    if len(sec_counts) > 1:
        _m = sum(sec_counts) / len(sec_counts)
        _var = sum((s - _m) ** 2 for s in sec_counts) / len(sec_counts)
        section_variance = _var ** 0.5
    else:
        section_variance = 0.0
    return (n_ch * 0.3 + n_sec * 0.3 + unique_titles * 0.2
            + section_variance * 2.0)


def _parse_json_candidate(raw: str) -> list | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw, strict=False)
    except Exception:
        return None
    ch = data.get("chapters")
    return ch if isinstance(ch, list) and ch else None


def _run_one(model: str, book_title: str, papers: list) -> dict:
    """Run one full outline (3 candidates + pick best) and return
    a dict of measurements — no side effects on the DB."""
    system, user = rag_prompts.outline(book_title=book_title, papers=papers)
    t0 = time.monotonic()
    candidates: list[tuple[list, float]] = []
    valid = 0
    per_cand_ms: list[int] = []
    for i, temp in enumerate(TEMPS):
        ct0 = time.monotonic()
        raw = complete_with_status(
            system, user,
            label=f"  {model[:25]}  cand {i + 1}/{N_CANDIDATES}",
            model=model, temperature=temp, num_ctx=16384,
        )
        per_cand_ms.append(int((time.monotonic() - ct0) * 1000))
        ch = _parse_json_candidate(raw)
        if ch is None:
            continue
        valid += 1
        candidates.append((ch, _score_candidate(ch)))
    wall_ms = int((time.monotonic() - t0) * 1000)
    if candidates:
        candidates.sort(key=lambda c: c[1], reverse=True)
        winner, winner_score = candidates[0]
    else:
        winner, winner_score = [], 0.0
    sec_counts = [len(c.get("sections") or []) for c in winner]
    return {
        "model": model,
        "wall_ms": wall_ms,
        "per_candidate_ms": per_cand_ms,
        "valid_json_count": valid,
        "winner_chapter_count": len(winner),
        "winner_section_variance": (
            statistics.stdev(sec_counts) if len(sec_counts) > 1 else 0.0
        ),
        "winner_score": winner_score,
        "winner_titles": [c.get("title", "") for c in winner],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", type=int, default=3,
        help="Repeats per model (noise dampens over runs).",
    )
    ap.add_argument(
        "--book", type=str, default="Global Cooling: The Coming Solar Minimum",
        help="Book title handed to the prompt.  Does NOT need to "
             "correspond to a real book in the DB — the bench reads "
             "paper titles from paper_metadata regardless.",
    )
    ap.add_argument(
        "--models", type=str, default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of models to A/B.",
    )
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Same paper fetch as the real CLI + web outline paths.
    with get_session() as session:
        papers = session.execute(text(
            "SELECT title, year FROM paper_metadata "
            "WHERE title IS NOT NULL "
            "ORDER BY year DESC NULLS LAST LIMIT 200"
        )).fetchall()
    paper_list = [{"title": r[0], "year": r[1]} for r in papers if r[0]]
    console.print(
        f"[bold]outline bench[/bold] — {len(paper_list)} papers, "
        f"{args.runs} run(s) × {len(models)} model(s) "
        f"× {N_CANDIDATES} candidates"
    )

    all_results: list[dict] = []
    for model in models:
        console.rule(f"[bold]{model}[/bold]")
        for run_i in range(args.runs):
            res = _run_one(model, args.book, paper_list)
            res["run"] = run_i + 1
            res["book"] = args.book
            res["papers_total"] = len(paper_list)
            all_results.append(res)
            console.print(
                f"  run {run_i + 1}: valid {res['valid_json_count']}/{N_CANDIDATES} · "
                f"chapters {res['winner_chapter_count']} · "
                f"variance {res['winner_section_variance']:.2f} · "
                f"score {res['winner_score']:.1f} · "
                f"{res['wall_ms'] / 1000:.1f} s"
            )

    # Summary table.
    console.rule("[bold]summary (mean across runs)[/bold]")
    tbl = Table(box=box.ROUNDED)
    tbl.add_column("model", style="cyan")
    tbl.add_column("runs", justify="right")
    tbl.add_column("valid /3", justify="right")
    tbl.add_column("chapters", justify="right")
    tbl.add_column("variance", justify="right")
    tbl.add_column("score", justify="right")
    tbl.add_column("wall (s)", justify="right")
    for model in models:
        rows = [r for r in all_results if r["model"] == model]
        if not rows:
            continue
        def _mean(key):
            vals = [r.get(key) or 0 for r in rows]
            return sum(vals) / len(vals) if vals else 0.0
        tbl.add_row(
            model,
            str(len(rows)),
            f"{_mean('valid_json_count'):.1f}",
            f"{_mean('winner_chapter_count'):.1f}",
            f"{_mean('winner_section_variance'):.2f}",
            f"{_mean('winner_score'):.1f}",
            f"{_mean('wall_ms') / 1000:.1f}",
        )
    console.print(tbl)

    # Write JSONL for follow-on analysis.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("data/bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"outline_ab-{ts}.jsonl"
    with out_path.open("w") as fh:
        for r in all_results:
            fh.write(json.dumps(r, default=str) + "\n")
    console.print(
        f"[green]✓[/green] raw results written to [bold]{out_path}[/bold]"
    )

    # Tentative recommendation based on the headline numbers.
    if len(models) >= 2:
        def score(m):
            rows = [r for r in all_results if r["model"] == m]
            if not rows:
                return 0.0
            return sum(r.get("winner_score") or 0 for r in rows) / len(rows)
        best = max(models, key=score)
        if best != settings.llm_model:
            console.print(
                f"\n[bold yellow]Recommendation:[/bold yellow] "
                f"set [bold]BOOK_OUTLINE_MODEL={best}[/bold] in .env "
                f"(higher mean scorer over {args.runs} runs)."
            )
        else:
            console.print(
                f"\n[bold]Recommendation:[/bold] keep the default "
                f"(BOOK_OUTLINE_MODEL unset)."
            )


if __name__ == "__main__":
    main()
