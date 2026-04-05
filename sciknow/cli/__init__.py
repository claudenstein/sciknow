"""CLI subpackage for sciknow."""
from __future__ import annotations

import typer
from rich.console import Console

_console = Console(stderr=True)


def preflight(*, pg: bool = True, qdrant: bool = True) -> None:
    """
    Verify required services are reachable before a long-running command.

    Call at the top of any CLI command that touches PostgreSQL and/or Qdrant.
    Exits immediately with a clear error instead of running MinerU / embeddings
    for minutes and then crashing at the upsert or DB write.

    Usage in a typer command:
        from sciknow.cli import preflight
        preflight()              # checks both PG and Qdrant
        preflight(qdrant=False)  # PG only (e.g. db enrich)
    """
    if pg:
        from sciknow.storage.db import check_connection as pg_ok
        if not pg_ok():
            _console.print(
                "[red]✗ PostgreSQL is unreachable.[/red]\n"
                "  Check that the server is running:\n"
                "    [bold]sudo systemctl status postgresql[/bold]\n"
                "  And that .env has the correct PG_HOST/PG_PORT/PG_USER/PG_PASSWORD."
            )
            raise typer.Exit(1)

    if qdrant:
        from sciknow.storage.qdrant import check_connection as qdrant_ok
        if not qdrant_ok():
            _console.print(
                "[red]✗ Qdrant is unreachable.[/red]\n"
                "  Check that the service is running:\n"
                "    [bold]systemctl --user status qdrant[/bold]\n"
                "    [bold]systemctl --user start qdrant[/bold]\n"
                "  And that .env has the correct QDRANT_HOST/QDRANT_PORT."
            )
            raise typer.Exit(1)
