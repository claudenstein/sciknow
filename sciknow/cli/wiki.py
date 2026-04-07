"""
sciknow wiki — Karpathy-style compiled knowledge wiki.

Commands:
  compile      Build wiki pages from ingested papers
  query        Answer questions from the compiled wiki
  lint         Check wiki health (broken links, contradictions, stale pages)
  list         List all wiki pages
  show         Display a wiki page
  synthesize   Generate a synthesis overview page on a topic
"""
from __future__ import annotations

from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

app = typer.Typer(help="Compiled knowledge wiki (Karpathy LLM-wiki pattern).")
console = Console()


def _check_wiki_table():
    """Verify the wiki_pages table exists, or exit with a helpful message."""
    from sciknow.storage.db import get_session
    from sqlalchemy import text
    try:
        with get_session() as sess:
            sess.execute(text("SELECT 1 FROM wiki_pages LIMIT 0"))
    except Exception:
        console.print(
            "[red]The wiki_pages table does not exist.[/red]\n"
            "Run the migration first:\n\n"
            "  [bold]uv run alembic upgrade head[/bold]\n"
        )
        raise typer.Exit(1)


def _consume_events(gen, console):
    """Consume events from a wiki_ops generator, printing to Rich console."""
    completed = None
    for event in gen:
        t = event.get("type")
        if t == "token":
            console.print(event["text"], end="", highlight=False)
        elif t == "progress":
            console.print(f"[dim]{event.get('detail', event.get('stage', ''))}[/dim]")
        elif t == "lint_issue":
            sev = event.get("severity", "low")
            color = "red" if sev == "high" else "yellow" if sev == "medium" else "dim"
            console.print(f"  [{color}]{event.get('type_', event.get('type', ''))}[/{color}]: {event.get('detail', '')}")
        elif t == "completed":
            completed = event
        elif t == "error":
            console.print(f"[red]Error:[/red] {event.get('message', 'unknown')}")
    return completed


@app.command()
def compile(
    doc_id: str | None = typer.Option(None, "--doc-id", "-d",
                                       help="Compile a single paper by document ID."),
    rebuild: bool = typer.Option(False, "--rebuild",
                                  help="Recompile ALL pages from scratch (destructive). Default: only compile new papers."),
    rewrite_stale: bool = typer.Option(False, "--rewrite-stale",
                                        help="Rewrite pages marked as stale."),
    model: str | None = typer.Option(None, "--model"),
):
    """
    Build wiki pages from ingested papers.

    By default, only compiles papers that don't have wiki pages yet
    (safe to re-run anytime). Use --rebuild to recompile everything.

    Examples:

      sciknow wiki compile                    # compile only new papers (safe)

      sciknow wiki compile --doc-id abc123    # compile one paper

      sciknow wiki compile --rebuild          # recompile everything from scratch
    """
    from sciknow.cli import preflight
    preflight(qdrant=True)
    _check_wiki_table()

    from sciknow.core import wiki_ops

    if doc_id:
        console.print(f"Compiling wiki page for document {doc_id[:8]}...")
        gen = wiki_ops.compile_paper_summary(doc_id, model=model, force=rebuild)
        result = _consume_events(gen, console)

        if result and not result.get("skipped"):
            console.print()
            gen2 = wiki_ops.update_concepts_for_paper(doc_id, model=model)
            result2 = _consume_events(gen2, console)
            if result2:
                console.print(f"[green]✓ Updated {result2.get('concepts_updated', 0)} concept pages[/green]")
    else:
        import time as _time
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

        gen = wiki_ops.compile_all(model=model, force=rebuild, rewrite_stale=rewrite_stale)

        result = None
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[status]}"),
            console=console,
            refresh_per_second=2,
        ) as progress:
            task_id = None
            for event in gen:
                t = event.get("type")

                if t == "compile_start":
                    total = event["total"]
                    task_id = progress.add_task(
                        "Compiling wiki", total=total, status="starting...")

                elif t == "paper_start":
                    progress.update(task_id, status=f"[dim]{event['title']}[/dim]")

                elif t == "paper_done":
                    progress.advance(task_id)
                    st = event.get("status", "")
                    c = event.get("compiled", 0)
                    s = event.get("skipped", 0)
                    f = event.get("failed", 0)
                    concepts = event.get("concepts", 0)
                    status_text = f"[green]{c} new[/green]  [dim]{s} skip[/dim]"
                    if f:
                        status_text += f"  [red]{f} fail[/red]"
                    if st == "compiled" and concepts:
                        status_text += f"  [cyan]+{concepts} concepts[/cyan]"
                    progress.update(task_id, status=status_text)

                elif t == "error":
                    console.print(f"[red]Error:[/red] {event.get('message', '')}")

                elif t == "completed":
                    result = event

        if result:
            console.print(
                f"\n[green]✓ Wiki compiled:[/green] "
                f"{result.get('compiled', 0)} new, "
                f"{result.get('skipped', 0)} skipped, "
                f"{result.get('failed', 0)} failed "
                f"/ {result.get('total', 0)} total papers"
            )


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to answer from the wiki.")],
    context_k: int = typer.Option(8, "--context-k", "-k"),
    model: str | None = typer.Option(None, "--model"),
):
    """
    Answer a question from the compiled wiki.

    Searches wiki pages (not raw paper chunks) for pre-synthesized,
    cross-referenced knowledge.

    Examples:

      sciknow wiki query "what is total solar irradiance?"

      sciknow wiki query "how do cosmic rays affect cloud formation?"
    """
    from sciknow.cli import preflight
    preflight(qdrant=True)
    _check_wiki_table()

    from sciknow.core.wiki_ops import query_wiki

    console.print()
    gen = query_wiki(question, context_k=context_k, model=model)
    result = _consume_events(gen, console)
    console.print()

    if result and result.get("sources"):
        console.print("[dim]Sources:[/dim]")
        for s in result["sources"]:
            console.print(f"  [dim]{s}[/dim]")


@app.command()
def lint(
    deep: bool = typer.Option(False, "--deep",
                               help="Run LLM-based contradiction detection (slower)."),
    model: str | None = typer.Option(None, "--model"),
):
    """
    Check wiki health: broken links, stale pages, orphaned concepts,
    missing summaries, and optionally contradictions (--deep).

    Examples:

      sciknow wiki lint

      sciknow wiki lint --deep   # includes contradiction detection
    """
    from sciknow.core.wiki_ops import lint_wiki

    console.print("Running wiki lint...\n")
    gen = lint_wiki(deep=deep, model=model)
    result = _consume_events(gen, console)

    if result:
        n = result.get("issues_count", 0)
        if n == 0:
            console.print("\n[green]✓ Wiki is clean — no issues found.[/green]")
        else:
            console.print(f"\n[yellow]{n} issues found.[/yellow]")


@app.command(name="list")
def list_pages(
    page_type: str | None = typer.Option(None, "--type", "-t",
                                          help="Filter by type: paper_summary, concept, synthesis."),
    limit: int = typer.Option(50, "--limit", "-l"),
):
    """
    List all wiki pages.

    Examples:

      sciknow wiki list

      sciknow wiki list --type concept

      sciknow wiki list --type paper_summary --limit 20
    """
    _check_wiki_table()
    from sciknow.core.wiki_ops import list_pages as _list

    pages = _list(page_type=page_type)[:limit]

    if not pages:
        console.print("[yellow]No wiki pages found.[/yellow]")
        console.print("Run [bold]sciknow wiki compile[/bold] to build the wiki.")
        return

    table = Table(title="Wiki Pages", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Slug", ratio=3)
    table.add_column("Type", style="cyan", width=16)
    table.add_column("Words", justify="right", width=7)
    table.add_column("Sources", justify="right", width=8)
    table.add_column("Updated", style="dim", width=12)

    for p in pages:
        table.add_row(
            p["slug"][:50],
            p["page_type"],
            str(p["word_count"]),
            str(p["n_sources"]),
            p["updated_at"][:10] if p["updated_at"] else "",
        )
    console.print(table)
    console.print(f"[dim]{len(pages)} pages[/dim]")


@app.command()
def show(
    slug: Annotated[str, typer.Argument(help="Wiki page slug.")],
):
    """
    Display a wiki page in the terminal.

    Examples:

      sciknow wiki show total-solar-irradiance

      sciknow wiki show a3f2b1c4-zharkova-2024-solar-eigenvectors
    """
    from sciknow.core.wiki_ops import show_page

    page = show_page(slug)
    if not page:
        console.print(f"[red]Page not found:[/red] {slug}")
        raise typer.Exit(1)

    console.print()
    console.print(Markdown(page["content"]))
    console.print()
    console.print(f"[dim]{page['path']}[/dim]")


@app.command()
def synthesize(
    topic: Annotated[str, typer.Argument(help="Topic to synthesize.")],
    model: str | None = typer.Option(None, "--model"),
):
    """
    Generate a synthesis overview page from existing wiki pages.

    Finds all paper summaries and concept pages related to the topic
    and writes a comparative analysis.

    Examples:

      sciknow wiki synthesize "solar forcing and climate"

      sciknow wiki synthesize "cosmic ray cloud nucleation"
    """
    from sciknow.cli import preflight
    preflight(qdrant=True)
    _check_wiki_table()

    from sciknow.core.wiki_ops import compile_synthesis

    console.print()
    gen = compile_synthesis(topic, model=model)
    result = _consume_events(gen, console)
    console.print()

    if result:
        console.print(
            f"\n[green]✓ Synthesis page created:[/green] [[{result.get('slug', '')}]] "
            f"({result.get('word_count', 0)} words)"
        )
