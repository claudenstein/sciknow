"""
sciknow ask — RAG question answering and writing assistant.
"""
from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.rule import Rule

app = typer.Typer(help="Ask questions and generate text from your paper library.")
console = Console()

# ── Shared search options ──────────────────────────────────────────────────────

_CONTEXT_K   = typer.Option(8,    "--context-k",  "-k",  help="Chunks to include in LLM context after reranking.")
_CANDIDATES  = typer.Option(50,   "--candidates",        help="Candidates fetched before reranking.")
_NO_RERANK   = typer.Option(False, "--no-rerank",         help="Skip reranker, use RRF scores directly.")
_SHOW_SRC    = typer.Option(True,  "--show-sources/--no-sources", help="Print source list after the answer.")
_YEAR_FROM   = typer.Option(None, "--year-from",          help="Filter: published >= year.")
_YEAR_TO     = typer.Option(None, "--year-to",            help="Filter: published <= year.")
_DOMAIN      = typer.Option(None, "--domain",             help="Filter by domain tag.")
_SECTION     = typer.Option(None, "--section",    "-s",  help="Filter by section type.")
_MODEL       = typer.Option(None, "--model",              help="Override LLM model name (Ollama).")


def _retrieve(
    query: str,
    context_k: int,
    candidates: int,
    no_rerank: bool,
    year_from: int | None,
    year_to: int | None,
    domain: str | None,
    section: str | None,
    session,
    qdrant,
):
    from sciknow.retrieval import context_builder, hybrid_search, reranker

    candidates_list = hybrid_search.search(
        query=query,
        qdrant_client=qdrant,
        session=session,
        candidate_k=candidates,
        year_from=year_from,
        year_to=year_to,
        domain=domain,
        section=section,
    )
    if not candidates_list:
        return []

    if not no_rerank:
        candidates_list = reranker.rerank(query, candidates_list, top_k=context_k)
    else:
        candidates_list = candidates_list[:context_k]

    return context_builder.build(candidates_list, session)


def _stream_answer(system: str, user: str, model: str | None) -> None:
    from sciknow.rag.llm import stream as llm_stream

    for token in llm_stream(system, user, model=model):
        console.print(token, end="", highlight=False)
    console.print()  # final newline


# ── ask question ──────────────────────────────────────────────────────────────

@app.command()
def question(
    q: Annotated[str, typer.Argument(metavar="QUESTION", help="Question to answer.")],
    context_k: int = _CONTEXT_K,
    candidates: int = _CANDIDATES,
    no_rerank: bool = _NO_RERANK,
    show_sources: bool = _SHOW_SRC,
    year_from: int | None = _YEAR_FROM,
    year_to: int | None = _YEAR_TO,
    domain: str | None = _DOMAIN,
    section: str | None = _SECTION,
    model: str | None = _MODEL,
):
    """
    Answer a question using RAG over ingested papers.

    Examples:

      sciknow ask question "What are the main mechanisms of aerosol radiative forcing?"

      sciknow ask question "How is sea surface temperature reconstructed from proxies?" \\
          --year-from 2000 --section methods

      sciknow ask question "Explain the central dogma of molecular biology" --no-sources
    """
    from sciknow.rag import prompts
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    qdrant = get_client()

    with get_session() as session:
        with console.status("[bold green]Retrieving relevant passages...", spinner="dots"):
            results = _retrieve(q, context_k, candidates, no_rerank,
                                year_from, year_to, domain, section, session, qdrant)

        if not results:
            console.print("[yellow]No relevant passages found in the knowledge base.[/yellow]")
            raise typer.Exit(0)

        system, user = prompts.qa(q, results)

    console.print()
    console.print(Rule("[bold]Answer[/bold]"))
    console.print()

    _stream_answer(system, user, model)

    if show_sources:
        from sciknow.rag.prompts import format_sources
        console.print()
        console.print(Rule("[dim]Sources[/dim]"))
        console.print(f"[dim]{format_sources(results)}[/dim]")
    console.print()


# ── ask synthesize ────────────────────────────────────────────────────────────

@app.command()
def synthesize(
    topic: Annotated[str, typer.Argument(metavar="TOPIC", help="Topic to synthesize.")],
    context_k: int = typer.Option(12, "--context-k", "-k", help="Passages to include."),
    candidates: int = _CANDIDATES,
    no_rerank: bool = _NO_RERANK,
    show_sources: bool = _SHOW_SRC,
    year_from: int | None = _YEAR_FROM,
    year_to: int | None = _YEAR_TO,
    domain: str | None = _DOMAIN,
    model: str | None = _MODEL,
):
    """
    Write a multi-paper synthesis on a topic.

    Retrieves the most relevant passages, then asks the LLM to synthesise
    findings, methods, consensus, and open questions.

    Examples:

      sciknow ask synthesize "solar activity and climate variability"

      sciknow ask synthesize "paleoclimate proxy methods" --domain climatology --context-k 15
    """
    from sciknow.rag import prompts
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    qdrant = get_client()

    with get_session() as session:
        with console.status("[bold green]Retrieving relevant passages...", spinner="dots"):
            results = _retrieve(topic, context_k, candidates, no_rerank,
                                year_from, year_to, domain, None, session, qdrant)

        if not results:
            console.print("[yellow]No relevant passages found.[/yellow]")
            raise typer.Exit(0)

        system, user = prompts.synthesis(topic, results)

    console.print()
    console.print(Rule(f"[bold]Synthesis:[/bold] {topic}"))
    console.print()

    _stream_answer(system, user, model)

    if show_sources:
        from sciknow.rag.prompts import format_sources
        console.print()
        console.print(Rule("[dim]Sources[/dim]"))
        console.print(f"[dim]{format_sources(results)}[/dim]")
    console.print()


# ── ask write ─────────────────────────────────────────────────────────────────

_VALID_SECTIONS = {
    "abstract", "introduction", "methods", "results",
    "discussion", "conclusion", "related_work",
}

@app.command()
def write(
    topic: Annotated[str, typer.Argument(metavar="TOPIC", help="Topic for the section.")],
    section: str = typer.Option(..., "--section", "-s",
                                 help="Section type to draft (introduction, methods, results, discussion, conclusion)."),
    context_k: int = typer.Option(10, "--context-k", "-k", help="Passages to include."),
    candidates: int = _CANDIDATES,
    no_rerank: bool = _NO_RERANK,
    show_sources: bool = _SHOW_SRC,
    year_from: int | None = _YEAR_FROM,
    year_to: int | None = _YEAR_TO,
    domain: str | None = _DOMAIN,
    model: str | None = _MODEL,
):
    """
    Draft a paper section grounded in your literature library.

    Examples:

      sciknow ask write "aerosol-cloud interactions" --section introduction

      sciknow ask write "stellar population synthesis methods" --section methods \\
          --domain astrophysics --year-from 2010
    """
    if section not in _VALID_SECTIONS:
        console.print(f"[red]Unknown section type:[/red] {section}")
        console.print(f"Valid: {', '.join(sorted(_VALID_SECTIONS))}")
        raise typer.Exit(1)

    from sciknow.rag import prompts
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    qdrant = get_client()

    # For section drafting, bias retrieval toward the target section type
    search_query = f"{section} {topic}"

    with get_session() as session:
        with console.status("[bold green]Retrieving relevant passages...", spinner="dots"):
            results = _retrieve(search_query, context_k, candidates, no_rerank,
                                year_from, year_to, domain, None, session, qdrant)

        if not results:
            console.print("[yellow]No relevant passages found.[/yellow]")
            raise typer.Exit(0)

        system, user = prompts.write_section(section, topic, results)

    console.print()
    console.print(Rule(f"[bold]Draft {section.capitalize()}:[/bold] {topic}"))
    console.print()

    _stream_answer(system, user, model)

    if show_sources:
        from sciknow.rag.prompts import format_sources
        console.print()
        console.print(Rule("[dim]Sources[/dim]"))
        console.print(f"[dim]{format_sources(results)}[/dim]")
    console.print()
