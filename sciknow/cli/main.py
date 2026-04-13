import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

from sciknow.cli import ask as ask_module
from sciknow.cli import book as book_module
from sciknow.cli import catalog as catalog_module
from sciknow.cli import db as db_module
from sciknow.cli import draft as draft_module
from sciknow.cli import ingest as ingest_module
from sciknow.cli import project as project_module
from sciknow.cli import search as search_module
from sciknow.cli import wiki as wiki_module
from sciknow.logging_config import setup_logging

app = typer.Typer(
    name="sciknow",
    help="Local-first scientific knowledge system.",
    no_args_is_help=True,
)
console = Console()

logger = logging.getLogger("sciknow.cli")


@app.callback()
def _startup(
    ctx: typer.Context,
    project: str = typer.Option(
        None, "--project", "-P",
        help="Override the active project for this invocation. "
             "Equivalent to setting SCIKNOW_PROJECT in the env. "
             "See `sciknow project list` for available slugs.",
    ),
) -> None:
    """Initialize logging and record the CLI invocation.

    Phase 43g — the ``--project`` root flag exports SCIKNOW_PROJECT into
    the process environment so every downstream module that reads from
    ``sciknow.core.project.get_active_project()`` picks up the override.
    Precedence (high → low): this flag → existing SCIKNOW_PROJECT env →
    ``.active-project`` file → legacy ``default`` fallback.
    """
    import os
    if project:
        # Validate eagerly so a typo fails before any subcommand runs.
        from sciknow.core.project import validate_slug
        try:
            validate_slug(project)
        except ValueError as exc:
            console.print(f"[red]--project:[/red] {exc}")
            raise typer.Exit(2)
        os.environ["SCIKNOW_PROJECT"] = project

    setup_logging()
    cmd = " ".join(sys.argv[1:]) or "(no args)"
    logger.info(f"CLI  {cmd}")


app.add_typer(catalog_module.app, name="catalog")
app.add_typer(db_module.app, name="db")
app.add_typer(ingest_module.app, name="ingest")
app.add_typer(search_module.app, name="search")
app.add_typer(ask_module.app, name="ask")
app.add_typer(book_module.app, name="book")
app.add_typer(draft_module.app, name="draft")
app.add_typer(wiki_module.app, name="wiki")
# Phase 43e — multi-project lifecycle commands.
app.add_typer(project_module.app, name="project")


@app.command(name="test")
def test_cmd(
    layer: str = typer.Option(
        "L1",
        "--layer", "-l",
        help="Which test layer to run: L1 (static), L2 (live integration), L3 (end-to-end), or all.",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Stop on the first failure instead of running every test in the layer.",
    ),
):
    """
    Run the layered testing protocol (smoke tests, not pytest).

    L1 — Static     (seconds, no deps)              imports, prompts, signatures
    L2 — Live       (tens of sec, PG + Qdrant)      hybrid_search, raptor scrolls
    L3 — End-to-end (minutes, PG + Qdrant + Ollama) one tiny LLM call, embedder

    Run L1 on every PR. Run L2 before shipping a "Phase" feature drop or after
    infrastructure changes. Run L3 after retrieval / LLM / embedder changes.
    See docs/TESTING.md for the full protocol and how to add new checks.

    Examples:

      sciknow test                  # L1 only (default — fast)
      sciknow test --layer L2       # live integration only
      sciknow test --layer all      # everything (L1 + L2 + L3)
      sciknow test -l all --fail-fast
    """
    from sciknow.testing import protocol

    layer_norm = layer.upper().strip()
    if layer_norm == "ALL":
        layers = ["L1", "L2", "L3"]
    elif layer_norm in ("L1", "L2", "L3"):
        layers = [layer_norm]
    else:
        console.print(f"[red]Unknown layer: {layer!r}. Use L1, L2, L3, or all.[/red]")
        raise typer.Exit(2)

    console.print(f"[bold]sciknow test[/bold] · layers: {', '.join(layers)}")
    console.print()
    n_failed = protocol.run_all(layers, fail_fast=fail_fast)

    console.print()
    if n_failed == 0:
        console.print("[bold green]✓ All tests passed[/bold green]")
        raise typer.Exit(0)
    else:
        console.print(f"[bold red]✗ {n_failed} test(s) failed[/bold red]")
        raise typer.Exit(1)


@app.command(name="bench")
def bench_cmd(
    layer: str = typer.Option(
        "fast", "--layer", "-l",
        help="Which bench layer to run: fast (descriptive only), live "
             "(adds hybrid_search + embedder + reranker), llm (adds "
             "Ollama throughput), or full.",
    ),
    tag: str = typer.Option(
        "", "--tag",
        help="Free-form label stamped into the output file + latest.json.",
    ),
    compare: bool = typer.Option(
        True, "--compare/--no-compare",
        help="Diff numeric metrics against the previous run's latest.json.",
    ),
):
    """Run the benchmarking harness (performance + quality metrics).

    This is separate from ``sciknow test`` — test is pass/fail for
    correctness, bench is numbers for speed/quality. Results land as
    JSONL under ``{data_dir}/bench/<ts>.jsonl`` plus a ``latest.json``
    rollup that the next run diffs against.

    \b
    Layers:
      fast   — DB + Qdrant stats, descriptive only, no model calls (~5s).
      live   — adds 1 embedder pass + hybrid_search round trip (~30s cold).
      llm    — adds Ollama fast + main model throughput (~60–180s cold).
      full   — every bench. Run before a release or after infra change.
    """
    from sciknow.testing import bench as bench_mod

    if layer not in bench_mod.LAYERS:
        console.print(f"[red]Unknown layer: {layer!r}. Use {list(bench_mod.LAYERS)}[/red]")
        raise typer.Exit(2)

    console.print(f"[bold]sciknow bench[/bold] · layer: [cyan]{layer}[/cyan]"
                  + (f" · tag: [dim]{tag}[/dim]" if tag else ""))
    console.print()
    results, out_path = bench_mod.run(layer=layer, tag=tag)

    diff = bench_mod.diff_against_latest(results) if compare else None
    bench_mod.render_report(results, diff=diff)

    console.print()
    console.print(f"[dim]Results written to[/dim] {out_path}")
    n_err = sum(1 for r in results if r.status == "error")
    if n_err:
        console.print(f"[yellow]{n_err} bench function(s) errored — check the JSONL for details.[/yellow]")
    raise typer.Exit(0 if n_err == 0 else 1)


if __name__ == "__main__":
    app()
