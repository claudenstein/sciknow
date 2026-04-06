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

from rich.table import Table

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
    expand: bool = typer.Option(False, "--expand/--no-expand", "-e",
                                 help="Expand query with LLM-generated synonyms/related terms before searching."),
):
    """
    Search papers using hybrid retrieval (dense + sparse + FTS) with optional reranking.

    Examples:

      sciknow search query "sea surface temperature reconstruction"

      sciknow search query "radiative forcing aerosols" --year-from 2010 --section methods

      sciknow search query "DNA replication fidelity" --no-rerank --show-content

      sciknow search query "solar forcing" --topic "Solar Irradiance"

      sciknow search query "solar forcing" --expand    # LLM expands query first
    """
    from sciknow.retrieval import hybrid_search, reranker, context_builder
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    qdrant = get_client()

    with get_session() as session:
        status_msg = "[bold green]Expanding + searching..." if expand else "[bold green]Searching..."
        with console.status(status_msg, spinner="dots"):
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
                use_query_expansion=expand,
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

        # Journal + DOI + citation count on second line
        meta_parts = []
        if result.journal:
            meta_parts.append(f"[italic]{result.journal}[/italic]")
        if result.doi:
            meta_parts.append(f"https://doi.org/{result.doi}")
        if result.citation_count > 0:
            meta_parts.append(f"cited by {result.citation_count}")
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


@app.command()
def similar(
    identifier: Annotated[str, typer.Argument(metavar="PAPER",
        help="Paper to find similar papers for. Accepts: DOI, arXiv ID, title fragment, or document UUID.")],
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of similar papers to return."),
    show_scores: bool = typer.Option(False, "--show-scores", help="Print similarity scores."),
):
    """
    Find papers with similar abstracts using the abstracts collection.

    Accepts a DOI, arXiv ID, title fragment, or document UUID. Looks up the
    paper's abstract embedding in Qdrant and returns the nearest neighbours.

    Examples:

      sciknow search similar "solar magnetic field"

      sciknow search similar 10.1093/mnras/stad1001

      sciknow search similar "2301.12345"
    """
    from sqlalchemy import text

    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import ABSTRACTS_COLLECTION, get_client

    qdrant = get_client()

    # ── Resolve identifier to a document_id ──────────────────────────────
    with get_session() as session:
        # Try DOI
        row = session.execute(
            text("SELECT d.id::text, pm.title, pm.doi, pm.year, pm.authors "
                 "FROM paper_metadata pm JOIN documents d ON d.id = pm.document_id "
                 "WHERE LOWER(pm.doi) = LOWER(:q) OR LOWER(pm.arxiv_id) = LOWER(:q) "
                 "LIMIT 1"),
            {"q": identifier.strip()},
        ).first()

        # Try title fragment
        if not row:
            row = session.execute(
                text("SELECT d.id::text, pm.title, pm.doi, pm.year, pm.authors "
                     "FROM paper_metadata pm JOIN documents d ON d.id = pm.document_id "
                     "WHERE pm.title ILIKE :pattern "
                     "ORDER BY pm.year DESC NULLS LAST LIMIT 1"),
                {"pattern": f"%{identifier.strip()}%"},
            ).first()

        # Try UUID
        if not row:
            try:
                row = session.execute(
                    text("SELECT d.id::text, pm.title, pm.doi, pm.year, pm.authors "
                         "FROM paper_metadata pm JOIN documents d ON d.id = pm.document_id "
                         "WHERE d.id::text = :q LIMIT 1"),
                    {"q": identifier.strip()},
                ).first()
            except Exception:
                pass

    if not row:
        console.print(f"[red]Paper not found:[/red] {identifier}")
        raise typer.Exit(1)

    doc_id, title, doi, year, authors = row
    console.print(f"Finding papers similar to: [bold]{title}[/bold] ({year or 'n.d.'})")

    # ── Find the abstract embedding in Qdrant ────────────────────────────
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    abstract_points = qdrant.scroll(
        collection_name=ABSTRACTS_COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=doc_id))
        ]),
        with_vectors=["dense"],
        limit=1,
    )[0]

    if not abstract_points:
        console.print("[yellow]No abstract embedding found for this paper. Was it ingested?[/yellow]")
        raise typer.Exit(1)

    query_vec = abstract_points[0].vector
    if isinstance(query_vec, dict):
        query_vec = query_vec.get("dense")

    # ── Search for nearest abstracts ─────────────────────────────────────
    results = qdrant.query_points(
        collection_name=ABSTRACTS_COLLECTION,
        query=query_vec,
        using="dense",
        limit=top_k + 1,  # +1 because the query paper itself will be in results
        with_payload=True,
    )

    # ── Display results ──────────────────────────────────────────────────
    from sciknow.rag.prompts import format_authors_apa

    console.print()
    table = Table(title=f"Papers similar to: {(title or '')[:60]}", show_lines=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Title", ratio=3)
    table.add_column("Year", width=5)
    table.add_column("Authors", ratio=2, style="dim")
    if show_scores:
        table.add_column("Sim", width=6, justify="right")

    rank = 0
    for point in results.points:
        payload = point.payload or {}
        result_doc_id = payload.get("document_id", "")
        if result_doc_id == doc_id:
            continue  # skip the query paper itself
        rank += 1
        result_title = payload.get("title") or payload.get("content_preview", "")[:80]
        result_year = str(payload.get("year") or "")
        result_authors = format_authors_apa(payload.get("authors") or []) if "authors" in payload else ""
        row = [str(rank), result_title[:80], result_year, result_authors[:60]]
        if show_scores:
            row.append(f"{point.score:.3f}")
        table.add_row(*row)
        if rank >= top_k:
            break

    console.print(table)
    if rank == 0:
        console.print("[yellow]No similar papers found.[/yellow]")
