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
        from rich.live import Live
        from rich.table import Table
        from rich.text import Text
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

        gen = wiki_ops.compile_all(model=model, force=rebuild, rewrite_stale=rewrite_stale)

        result = None
        total = 0
        paper_tok_count = 0
        paper_t0 = _time.monotonic()
        compile_t0 = _time.monotonic()
        last_paper_tps = 0.0
        last_paper_title = ""
        last_paper_elapsed = 0.0
        last_paper_tokens = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=25),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[status]}"),
            console=console,
            refresh_per_second=4,
        ) as progress:
            task_id = None
            tok_task = None

            for event in gen:
                t = event.get("type")

                if t == "compile_start":
                    total = event["total"]
                    compile_t0 = _time.monotonic()
                    task_id = progress.add_task(
                        "Compiling wiki", total=total, status="starting...")
                    tok_task = progress.add_task(
                        "[dim]LLM", total=None, status="[dim]waiting...[/dim]")

                elif t == "paper_start":
                    paper_tok_count = 0
                    paper_t0 = _time.monotonic()
                    last_paper_title = event["title"]
                    progress.update(task_id,
                        status=f"[dim]{event['title'][:45]}[/dim]")
                    progress.update(tok_task,
                        status=f"[dim]{event['title'][:35]}...[/dim]")

                elif t == "token":
                    paper_tok_count += 1
                    elapsed = _time.monotonic() - paper_t0
                    tps = paper_tok_count / elapsed if elapsed > 0 else 0
                    total_toks = event.get("total_tokens", paper_tok_count)
                    progress.update(tok_task,
                        status=(
                            f"[green]{paper_tok_count} tok[/green]  "
                            f"[cyan]{tps:.1f} tok/s[/cyan]  "
                            f"[dim]{total_toks} total[/dim]"
                        ))

                elif t == "paper_done":
                    progress.advance(task_id)
                    st = event.get("status", "")
                    c = event.get("compiled", 0)
                    s = event.get("skipped", 0)
                    f = event.get("failed", 0)
                    concepts = event.get("concepts", 0)
                    p_toks = event.get("tokens", 0)
                    p_elapsed = event.get("elapsed", 0)

                    last_paper_tokens = p_toks
                    last_paper_elapsed = p_elapsed
                    last_paper_tps = p_toks / p_elapsed if p_elapsed > 0 else 0

                    # Estimate remaining time
                    done = c + s + f
                    remaining = total - done
                    if c > 0 and p_elapsed > 0:
                        total_elapsed = _time.monotonic() - compile_t0
                        avg_per_compiled = total_elapsed / max(done, 1)
                        eta_s = remaining * avg_per_compiled
                        if eta_s > 3600:
                            eta_str = f"{eta_s/3600:.1f}h"
                        elif eta_s > 60:
                            eta_str = f"{eta_s/60:.0f}m"
                        else:
                            eta_str = f"{eta_s:.0f}s"
                        eta_part = f"  [yellow]~{eta_str} left[/yellow]"
                    else:
                        eta_part = ""

                    status_text = f"[green]{c} new[/green]  [dim]{s} skip[/dim]"
                    if f:
                        status_text += f"  [red]{f} fail[/red]"
                    status_text += eta_part

                    progress.update(task_id, status=status_text)

                    if st == "compiled" and p_toks > 0:
                        progress.update(tok_task,
                            status=(
                                f"[green]{p_toks} tok in {p_elapsed:.0f}s "
                                f"({last_paper_tps:.1f} t/s)[/green]"
                                + (f"  [cyan]+{concepts} concepts[/cyan]" if concepts else "")
                            ))
                    elif st == "skipped":
                        progress.update(tok_task, status="[dim]skipped[/dim]")

                elif t == "error":
                    console.print(f"[red]Error:[/red] {event.get('message', '')}")

                elif t == "completed":
                    result = event
                    if tok_task:
                        progress.remove_task(tok_task)

        if result:
            total_elapsed = _time.monotonic() - compile_t0
            if total_elapsed > 3600:
                elapsed_str = f"{total_elapsed/3600:.1f} hours"
            elif total_elapsed > 60:
                elapsed_str = f"{total_elapsed/60:.0f} minutes"
            else:
                elapsed_str = f"{total_elapsed:.0f} seconds"

            console.print(
                f"\n[green]✓ Wiki compiled:[/green] "
                f"{result.get('compiled', 0)} new, "
                f"{result.get('skipped', 0)} skipped, "
                f"{result.get('failed', 0)} failed "
                f"/ {result.get('total', 0)} total papers "
                f"in {elapsed_str}"
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
def graph(
    entity: Annotated[str, typer.Argument(help="Entity to explore (e.g. 'solar forcing', a paper slug).")],
    depth: int = typer.Option(1, "--depth", "-d", help="Graph traversal depth (1=direct, 2=two-hop)."),
    limit: int = typer.Option(30, "--limit", "-l"),
):
    """
    Explore the knowledge graph around an entity.

    Shows all triples where the entity appears as subject or object,
    plus connected entities at the specified depth.

    Examples:

      sciknow wiki graph "solar forcing"

      sciknow wiki graph "total solar irradiance" --depth 2

      sciknow wiki graph "zharkova" --limit 50
    """
    _check_wiki_table()

    from sqlalchemy import text
    from sciknow.storage.db import get_session

    # Check KG table exists
    try:
        with get_session() as sess:
            sess.execute(text("SELECT 1 FROM knowledge_graph LIMIT 0"))
    except Exception:
        console.print(
            "[red]The knowledge_graph table does not exist.[/red]\n"
            "Run: [bold]uv run alembic upgrade head[/bold]"
        )
        raise typer.Exit(1)

    pattern = f"%{entity.lower()}%"
    with get_session() as session:
        rows = session.execute(text("""
            SELECT subject, predicate, object, source_doc_id::text
            FROM knowledge_graph
            WHERE LOWER(subject) LIKE :pat OR LOWER(object) LIKE :pat
            ORDER BY subject, predicate
            LIMIT :lim
        """), {"pat": pattern, "lim": limit}).fetchall()

    if not rows:
        console.print(f"[yellow]No triples found for:[/yellow] {entity}")
        console.print("[dim]Run `sciknow wiki compile` to extract knowledge graph triples.[/dim]")
        raise typer.Exit(0)

    table = Table(title=f"Knowledge Graph: {entity}", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Subject", ratio=2)
    table.add_column("Predicate", style="cyan", ratio=1)
    table.add_column("Object", ratio=2)
    table.add_column("Source", style="dim", width=10)

    for subj, pred, obj, src in rows:
        table.add_row(subj[:40], pred, obj[:40], (src or "")[:8])
    console.print(table)
    console.print(f"[dim]{len(rows)} triples[/dim]")

    # Depth 2: follow connected entities
    if depth >= 2:
        connected = set()
        for subj, pred, obj, _ in rows:
            connected.add(subj)
            connected.add(obj)
        connected.discard(entity.lower())

        hop2_rows = []
        for ent in list(connected)[:10]:
            with get_session() as session:
                h2 = session.execute(text("""
                    SELECT subject, predicate, object
                    FROM knowledge_graph
                    WHERE (LOWER(subject) = :ent OR LOWER(object) = :ent)
                      AND LOWER(subject) NOT LIKE :orig AND LOWER(object) NOT LIKE :orig
                    LIMIT 5
                """), {"ent": ent, "orig": pattern}).fetchall()
                hop2_rows.extend(h2)

        if hop2_rows:
            console.print(f"\n[bold]2-hop connections:[/bold]")
            for subj, pred, obj in hop2_rows[:20]:
                console.print(f"  {subj[:35]} [cyan]{pred}[/cyan] {obj[:35]}")


@app.command()
def consensus(
    topic: Annotated[str, typer.Argument(help="Topic to map consensus for.")],
    model: str | None = typer.Option(None, "--model"),
):
    """
    Map the consensus landscape for a topic.

    Uses the knowledge graph and paper summaries to identify:
    - Key claims and which papers support/contradict them
    - Consensus level (strong/moderate/weak/contested)
    - Trends (growing/stable/declining/emerging)
    - Most debated sub-topics

    Saves the result as a wiki synthesis page.

    Examples:

      sciknow wiki consensus "solar forcing and climate"

      sciknow wiki consensus "cosmic ray cloud nucleation"
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)
    _check_wiki_table()

    from sciknow.core.wiki_ops import consensus_map

    console.print()
    gen = consensus_map(topic, model=model)

    result = None
    for event in gen:
        t = event.get("type")
        if t == "progress":
            console.print(f"[dim]{event.get('detail', '')}[/dim]")
        elif t == "consensus":
            data = event["data"]
            console.print()
            console.print(Rule(f"[bold]Consensus Map: {topic}[/bold]"))
            console.print()
            console.print(f"[dim]{data.get('summary', '')}[/dim]")
            console.print()

            for c in data.get("claims", []):
                level = c.get("consensus_level", "unknown")
                color = {"strong": "green", "moderate": "cyan", "weak": "yellow", "contested": "red"}.get(level, "dim")
                console.print(f"  [{color}]{level.upper()}[/{color}]  {c.get('claim', '')}")
                sup = c.get("supporting_papers", [])
                con = c.get("contradicting_papers", [])
                if sup:
                    console.print(f"    [green]Supports ({len(sup)}):[/green] {', '.join(sup[:3])}")
                if con:
                    console.print(f"    [red]Contradicts ({len(con)}):[/red] {', '.join(con[:3])}")
                console.print()

            debated = data.get("most_debated", [])
            if debated:
                console.print("[bold]Most debated:[/bold]")
                for d in debated:
                    console.print(f"  [yellow]- {d}[/yellow]")
        elif t == "completed":
            result = event
        elif t == "error":
            console.print(f"[red]Error:[/red] {event.get('message', '')}")

    if result:
        console.print(
            f"\n[green]✓ Consensus map saved:[/green] [[{result.get('slug', '')}]] "
            f"({result.get('claims', 0)} claims)"
        )


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
