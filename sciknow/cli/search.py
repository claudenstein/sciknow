"""
sciknow search — hybrid retrieval CLI commands.
"""
from __future__ import annotations

from typing import Annotated

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

app = typer.Typer(help="Search the knowledge base.")
console = Console()


@app.command()
def query(
    q: Annotated[str, typer.Argument(metavar="QUERY", help="Search query.")],
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to return after reranking."),
    candidate_k: int = typer.Option(50, "--candidates", help="Candidates fetched before reranking."),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip reranking, return RRF results directly."),
    show_content: bool = typer.Option(False, "--show-content", "-c", help="Print full chunk text."),
    show_scores: bool = typer.Option(False, "--show-scores", help="Print relevance scores."),
    year_from: int | None = typer.Option(None, "--year-from", help="Filter: published >= year."),
    year_to: int | None = typer.Option(None, "--year-to", help="Filter: published <= year."),
    domain: str | None = typer.Option(None, "--domain", help="Filter by domain tag."),
    section: str | None = typer.Option(
        None, "--section", "-s",
        help="Filter by section type (abstract, introduction, methods, results, discussion, conclusion).",
    ),
    topic: str | None = typer.Option(None, "--topic", "-t", help="Filter by topic cluster name."),
):
    """
    Search papers using hybrid retrieval (dense + sparse + FTS) with optional reranking.

    Examples:

      sciknow search query "sea surface temperature reconstruction"

      sciknow search query "radiative forcing aerosols" --year-from 2010 --section methods

      sciknow search query "DNA replication fidelity" --no-rerank --show-content

      sciknow search query "solar forcing" --topic "Solar Irradiance"
    """
    from sciknow.retrieval import hybrid_search, reranker, context_builder
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    qdrant = get_client()

    with get_session() as session:
        with console.status("[bold green]Embedding query...", spinner="dots"):
            candidates = hybrid_search.search(
                query=q,
                qdrant_client=qdrant,
                session=session,
                candidate_k=candidate_k,
                year_from=year_from,
                year_to=year_to,
                domain=domain,
                section=section,
                topic_cluster=topic,
            )

        if not candidates:
            console.print("[yellow]No results found.[/yellow]")
            raise typer.Exit(0)

        if not no_rerank:
            with console.status("[bold green]Reranking...", spinner="dots"):
                candidates = reranker.rerank(q, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        results = context_builder.build(candidates, session)

    if not results:
        console.print("[yellow]No results after reranking.[/yellow]")
        raise typer.Exit(0)

    console.print()
    console.print(Rule(f"[bold]Results for:[/bold] {q}"))
    console.print()

    from sciknow.rag.prompts import format_authors_apa

    for result in results:
        score_str = f"  [dim]score={result.score:.4f}[/dim]" if show_scores else ""
        section_badge = f"[cyan][{result.section_type}][/cyan] " if result.section_type else ""
        title_str = result.title or "[dim](untitled)[/dim]"
        year_str = f" ({result.year})" if result.year else " (n.d.)"

        # APA-style header: rank. Authors (year). Title.
        author_str = format_authors_apa(result.authors)
        author_part = f"[dim]{author_str}[/dim] " if author_str else ""
        console.print(
            f"[bold]{result.rank}.[/bold] {author_part}"
            f"{year_str}. {section_badge}[bold]{title_str}[/bold]{score_str}"
        )

        # Journal + DOI on second line
        meta_parts = []
        if result.journal:
            meta_parts.append(f"[italic]{result.journal}[/italic]")
        if result.doi:
            meta_parts.append(f"https://doi.org/{result.doi}")
        if meta_parts:
            console.print(f"   [dim]{' · '.join(meta_parts)}[/dim]")

        # Section title if present
        if result.section_title and result.section_title != result.section_type:
            console.print(f"   [dim italic]§ {result.section_title}[/dim italic]")

        # Content
        if show_content:
            console.print()
            console.print(Panel(
                result.content,
                border_style="dim",
                padding=(0, 1),
            ))
        else:
            # Short preview — strip HTML tags and normalise whitespace
            import re as _re
            preview = _re.sub(r'<[^>]+>', '', result.content)
            preview = _re.sub(r'\s+', ' ', preview).strip()[:200]
            if len(result.content) > 200:
                preview += "…"
            console.print(f"   [dim]{preview}[/dim]")

        console.print()
