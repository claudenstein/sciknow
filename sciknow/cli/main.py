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
from sciknow.cli import search as search_module
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


if __name__ == "__main__":
    app()
