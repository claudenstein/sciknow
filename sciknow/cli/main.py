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
def _startup(ctx: typer.Context) -> None:
    """Initialize logging and record the CLI invocation."""
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


if __name__ == "__main__":
    app()
