"""Phase 54.6.240 — focused 3-way bench of the new Qwen3.6-27B-dense
vs the current LLM roles (qwen3:30b-a3b-instruct-2507 primary and
qwen3.5:27b book baseline).

Runs all three sweep bench functions (extract-kg, compile-summary,
write-section) against one fixed paper, one model at a time.
Results written as JSONL + a compact Rich summary table.

Invocation:  uv run python scripts/bench_qwen36_27b_vs_current.py

No CLI flags on purpose — this is a one-shot script. Edit the
FOCUS list below to swap in other candidates. Runs in ~20-40
minutes on a 3090 (thinking models dominate the wall clock).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from rich import box
from rich.console import Console
from rich.table import Table

from sciknow.testing import model_sweep as ms
from sciknow.testing.bench import BenchMetric

console = Console()

FOCUS = [
    "qwen3.6:27b-dense",                   # NEW — to evaluate
    "qwen3:30b-a3b-instruct-2507-q4_K_M",  # current primary (wiki/main)
    "qwen3.5:27b",                         # former book baseline (for reference)
]

import os as _os

# 54.6.241 — allow subsetting via env var, e.g.
#   BENCH_TASKS=write_section uv run python scripts/bench_qwen36_27b_vs_current.py
# Used for the post-fix re-run after qwen3.6:27b-dense's thinking
# runaway was fixed by force_no_thinking=True on this task.
_ALL_TASKS = [
    ("extract_kg",      ms.b_model_sweep_extract_kg),
    ("compile_summary", ms.b_model_sweep_compile_summary),
    ("write_section",   ms.b_model_sweep_write_section),
]
_task_filter = _os.environ.get("BENCH_TASKS", "").strip()
if _task_filter:
    wanted = {t.strip() for t in _task_filter.split(",")}
    TASKS = [(n, f) for n, f in _ALL_TASKS if n in wanted]
else:
    TASKS = _ALL_TASKS


def _consume(fn) -> list[BenchMetric]:
    """Drain a bench generator into a list, printing as it goes."""
    out: list[BenchMetric] = []
    for m in fn() or ():
        out.append(m)
    return out


def main() -> None:
    # Monkey-patch CANDIDATE_MODELS for the duration of this run.
    original = ms.CANDIDATE_MODELS
    ms.CANDIDATE_MODELS = FOCUS
    try:
        installed = ms._installed_models()
        missing = [m for m in FOCUS if m not in installed]
        if missing:
            console.print(
                f"[red]Not installed in Ollama:[/red] {missing}\n"
                f"[dim]Install or edit FOCUS in this script.[/dim]"
            )
            return

        console.print(
            f"[bold]Qwen3.6-27B-dense bench[/bold] · "
            f"paper [cyan]{ms.CANDIDATE_PAPERS[0]}[/cyan] · "
            f"{len(FOCUS)} models × {len(TASKS)} tasks"
        )
        console.print()

        all_results: dict[str, list[BenchMetric]] = {}
        t_total = time.monotonic()
        for task_name, task_fn in TASKS:
            console.print(f"[bold]── {task_name} ──[/bold]")
            t0 = time.monotonic()
            metrics = _consume(task_fn)
            all_results[task_name] = metrics
            console.print(
                f"  [dim]{len(metrics)} metrics in "
                f"{time.monotonic() - t0:.0f}s[/dim]"
            )
        elapsed = time.monotonic() - t_total
        console.print(
            f"\n[green]✓ Sweep done in {elapsed / 60:.1f}m[/green]"
        )

        # Write JSONL for later analysis
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path.home() / "Claude/sciknow/projects/global-cooling/data/bench/qwen36_focus"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"bench-{ts}.jsonl"
        with out_path.open("w") as f:
            f.write(json.dumps({
                "_kind": "header", "ts": ts, "focus": FOCUS,
                "paper": ms.CANDIDATE_PAPERS[0],
                "elapsed_s": elapsed,
            }) + "\n")
            for task, metrics in all_results.items():
                for m in metrics:
                    f.write(json.dumps({
                        "task": task, **m.as_dict(),
                    }) + "\n")
        console.print(f"[dim]Wrote {out_path}[/dim]")

        # Compact summary — one row per (model, task) with key metrics
        # (elapsed_s, eval_count, content_chars, composite quality).
        _render_summary(all_results)
    finally:
        ms.CANDIDATE_MODELS = original


def _by_model_metric(
    metrics: list[BenchMetric],
) -> dict[tuple[str, str], BenchMetric]:
    idx: dict[tuple[str, str], BenchMetric] = {}
    for m in metrics:
        # Metric name shape: "model::metric_name"
        if "::" not in m.name:
            continue
        model, rest = m.name.split("::", 1)
        idx[(model, rest)] = m
    return idx


def _render_summary(all_results: dict[str, list[BenchMetric]]) -> None:
    t = Table(
        title="Focused bench summary (key metrics per model × task)",
        box=box.SIMPLE_HEAD, expand=True,
    )
    t.add_column("Model", ratio=3)
    t.add_column("Task", ratio=2)
    t.add_column("Elapsed", justify="right", width=10)
    t.add_column("Tokens out", justify="right", width=11)
    t.add_column("Content", justify="right", width=9, style="dim")
    t.add_column("Thinking", justify="right", width=9, style="dim")
    t.add_column("Status", width=14)

    for task_name, metrics in all_results.items():
        idx = _by_model_metric(metrics)
        for model in FOCUS:
            status = idx.get((model, "status"))
            elapsed = idx.get((model, "elapsed_s"))
            eval_count = idx.get((model, "eval_count"))
            content = idx.get((model, "content_chars"))
            thinking = idx.get((model, "thinking_chars"))
            st = str(status.value) if status else "ok"
            st_colour = (
                "red" if st == "error" else
                "yellow" if st == "not-installed" else "green"
            )
            t.add_row(
                model, task_name,
                f"{elapsed.value}s" if elapsed else "—",
                f"{eval_count.value}" if eval_count else "—",
                f"{content.value}" if content else "—",
                f"{thinking.value}" if thinking else "—",
                f"[{st_colour}]{st}[/{st_colour}]",
            )
        t.add_row("", "", "", "", "", "", "")
    console.print(t)


if __name__ == "__main__":
    main()
