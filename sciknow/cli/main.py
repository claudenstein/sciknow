import sys
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
from sciknow.cli import search as search_module

app = typer.Typer(
    name="sciknow",
    help="Local-first scientific knowledge system.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _log_invocation(ctx: typer.Context) -> None:
    """Log every CLI invocation to data/sciknow.log before the command runs."""
    try:
        log_path = Path("data/sciknow.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cmd = " ".join(sys.argv[1:]) or "(no args)"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{ts}  {cmd}\n")
    except Exception:
        pass  # never let logging break the CLI


app.add_typer(catalog_module.app, name="catalog")
app.add_typer(db_module.app, name="db")
app.add_typer(ingest_module.app, name="ingest")
app.add_typer(search_module.app, name="search")
app.add_typer(ask_module.app, name="ask")
app.add_typer(book_module.app, name="book")
app.add_typer(draft_module.app, name="draft")


if __name__ == "__main__":
    app()
