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
_TOPIC       = typer.Option(None, "--topic",      "-t",  help="Filter by topic cluster name.")
_MODEL       = typer.Option(None, "--model",              help="Override LLM model name (Ollama).")
_EXPAND      = typer.Option(False, "--expand/--no-expand", "-e", help="Expand query with LLM synonyms before retrieval.")
_SELF_CORRECT = typer.Option(False, "--self-correct/--no-self-correct", help="Enable Self-RAG: evaluate retrieval relevance + check answer grounding.")


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
    topic: str | None = None,
    expand: bool = False,
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
        topic_cluster=topic,
        use_query_expansion=expand,
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
    topic: str | None = _TOPIC,
    model: str | None = _MODEL,
    expand: bool = _EXPAND,
    self_correct: bool = _SELF_CORRECT,
):
    """
    Answer a question using RAG over ingested papers.

    Use --self-correct for Self-RAG: evaluates retrieval relevance before
    answering (retries with reformulated query if poor), then checks
    answer grounding after generation.

    Examples:

      sciknow ask question "What are the main mechanisms of aerosol radiative forcing?"

      sciknow ask question "How is sea surface temperature reconstructed from proxies?" \\
          --year-from 2000 --section methods

      sciknow ask question "solar forcing" --self-correct

      sciknow ask question "solar forcing" --expand
    """
    from sciknow.rag import prompts
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    qdrant = get_client()
    effective_query = q

    with get_session() as session:
        with console.status("[bold green]Retrieving relevant passages...", spinner="dots"):
            results = _retrieve(q, context_k, candidates, no_rerank,
                                year_from, year_to, domain, section, session, qdrant, topic,
                                expand=expand)

        if not results:
            console.print("[yellow]No relevant passages found in the knowledge base.[/yellow]")
            raise typer.Exit(0)

        # Self-RAG step 1: evaluate retrieval relevance
        if self_correct:
            from sciknow.retrieval.self_rag import evaluate_retrieval
            with console.status("[bold cyan]Evaluating retrieval relevance...", spinner="dots"):
                is_relevant, rel_score, reformulated = evaluate_retrieval(
                    q, results, model=model,
                )
            console.print(f"[dim]Retrieval relevance: {rel_score:.2f}[/dim]")

            if not is_relevant and reformulated != q:
                console.print(f"[yellow]Low relevance — retrying with:[/yellow] {reformulated}")
                results = _retrieve(reformulated, context_k, candidates, no_rerank,
                                    year_from, year_to, domain, section, session, qdrant, topic,
                                    expand=True)
                effective_query = reformulated
                if not results:
                    console.print("[yellow]Still no relevant passages after reformulation.[/yellow]")
                    raise typer.Exit(0)

        system, user = prompts.qa(effective_query, results)

    console.print()
    console.print(Rule("[bold]Answer[/bold]"))
    console.print()

    # Capture answer for grounding check
    if self_correct:
        from sciknow.rag.llm import stream as llm_stream
        answer_tokens = []
        for tok in llm_stream(system, user, model=model):
            console.print(tok, end="", highlight=False)
            answer_tokens.append(tok)
        console.print()
        answer_text = "".join(answer_tokens)

        # Self-RAG step 2: check grounding
        from sciknow.retrieval.self_rag import check_grounding
        with console.status("[bold cyan]Checking answer grounding...", spinner="dots"):
            grounding_score, ungrounded = check_grounding(
                answer_text, results, model=model,
            )
        console.print()
        color = "green" if grounding_score >= 0.8 else "yellow" if grounding_score >= 0.6 else "red"
        console.print(f"[{color}]Grounding score: {grounding_score:.2f}[/{color}]")
        if ungrounded:
            console.print(f"[yellow]Ungrounded claims ({len(ungrounded)}):[/yellow]")
            for claim in ungrounded[:5]:
                console.print(f"  [dim]- {claim[:100]}[/dim]")
    else:
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
    topic_filter: str | None = _TOPIC,
    model: str | None = _MODEL,
    expand: bool = _EXPAND,
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
                                year_from, year_to, domain, None, session, qdrant, topic_filter,
                                expand=expand)

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
    topic_filter: str | None = _TOPIC,
    model: str | None = _MODEL,
    expand: bool = _EXPAND,
    save: bool = typer.Option(False, "--save", help="Save the draft to the database."),
    book: str | None = typer.Option(None, "--book", "-b", help="Book title to associate with the draft (requires --save)."),
    chapter: int | None = typer.Option(None, "--chapter", "-c", help="Chapter number to associate with the draft (requires --book)."),
):
    """
    Draft a paper section grounded in your literature library.

    Examples:

      sciknow ask write "aerosol-cloud interactions" --section introduction

      sciknow ask write "stellar population synthesis methods" --section methods \\
          --domain astrophysics --year-from 2010

      sciknow ask write "solar forcing mechanisms" --section introduction --save \\
          --book "Global Cooling" --chapter 2
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
                                year_from, year_to, domain, None, session, qdrant, topic_filter,
                                expand=expand)

        if not results:
            console.print("[yellow]No relevant passages found.[/yellow]")
            raise typer.Exit(0)

        system, user = prompts.write_section(section, topic, results)

    console.print()
    console.print(Rule(f"[bold]Draft {section.capitalize()}:[/bold] {topic}"))
    console.print()

    # Collect streamed output so we can save it
    from sciknow.rag.llm import stream as llm_stream
    tokens: list[str] = []
    for token in llm_stream(system, user, model=model):
        console.print(token, end="", highlight=False)
        tokens.append(token)
    console.print()
    content = "".join(tokens)

    if show_sources:
        from sciknow.rag.prompts import format_sources
        console.print()
        console.print(Rule("[dim]Sources[/dim]"))
        console.print(f"[dim]{format_sources(results)}[/dim]")
    console.print()

    if save:
        from sqlalchemy import text
        from sciknow.storage.db import get_session
        import uuid

        source_lines = [
            prompts._apa_citation(r, i + 1)
            for i, r in enumerate(results)
        ]
        word_count = len(content.split())

        book_id = None
        chapter_id = None

        with get_session() as session:
            if book:
                row = session.execute(
                    text("SELECT id FROM books WHERE title ILIKE :t LIMIT 1"),
                    {"t": f"%{book}%"},
                ).fetchone()
                if row:
                    book_id = str(row[0])
                else:
                    console.print(f"[yellow]Warning: book not found: {book!r}[/yellow]")

            if book_id and chapter is not None:
                row = session.execute(
                    text("SELECT id FROM book_chapters WHERE book_id = :bid AND number = :num LIMIT 1"),
                    {"bid": book_id, "num": chapter},
                ).fetchone()
                if row:
                    chapter_id = str(row[0])
                else:
                    console.print(f"[yellow]Warning: chapter {chapter} not found in book[/yellow]")

            draft_title = f"{section.capitalize()}: {topic}"
            session.execute(text("""
                INSERT INTO drafts (id, title, book_id, chapter_id, section_type, topic,
                                    content, word_count, sources, model_used)
                VALUES (:id, :title, :book_id, :chapter_id, :section_type, :topic,
                        :content, :word_count, CAST(:sources AS jsonb), :model_used)
            """), {
                "id": str(uuid.uuid4()),
                "title": draft_title,
                "book_id": book_id,
                "chapter_id": chapter_id,
                "section_type": section,
                "topic": topic,
                "content": content,
                "word_count": word_count,
                "sources": __import__("json").dumps(source_lines),
                "model_used": model or "default",
            })
            session.commit()

        console.print(f"[green]✓ Draft saved:[/green] {draft_title}  ({word_count} words)")
