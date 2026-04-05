"""
sciknow book — manage book projects, chapters, outlines, and writing.

Commands:
  create          Create a new book project
  list            List all books
  show            Show a book's chapters and draft status
  chapter add     Add a chapter to a book
  outline         Generate a chapter structure from the literature (LLM)
  write           Draft a chapter section and save it
  argue           Map the argument for a claim against the literature
  gaps            Identify what's missing in the book
  export          Compile all chapter drafts into a single document
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

app = typer.Typer(help="Manage book projects and chapter writing.")
chapter_app = typer.Typer(help="Manage chapters within a book.")
app.add_typer(chapter_app, name="chapter")

console = Console()


# ── helpers ────────────────────────────────────────────────────────────────────

def _get_book(session, title_or_id: str):
    from sqlalchemy import text
    row = session.execute(text("""
        SELECT id::text, title, description, status, created_at
        FROM books WHERE title ILIKE :q OR id::text LIKE :q
        LIMIT 1
    """), {"q": f"%{title_or_id}%"}).fetchone()
    return row


def _get_chapter(session, book_id: str, chapter_ref: str):
    """Find a chapter by number or title fragment."""
    from sqlalchemy import text
    # Try by number first
    if chapter_ref.isdigit():
        row = session.execute(text("""
            SELECT id::text, number, title, description, topic_query, topic_cluster
            FROM book_chapters WHERE book_id = :bid AND number = :num
        """), {"bid": book_id, "num": int(chapter_ref)}).fetchone()
        if row:
            return row
    # Fall back to title fragment
    return session.execute(text("""
        SELECT id::text, number, title, description, topic_query, topic_cluster
        FROM book_chapters WHERE book_id = :bid AND title ILIKE :q
        LIMIT 1
    """), {"bid": book_id, "q": f"%{chapter_ref}%"}).fetchone()


# ── create ─────────────────────────────────────────────────────────────────────

@app.command()
def create(
    title: Annotated[str, typer.Argument(help="Book title.")],
    description: str | None = typer.Option(None, "--description", "-d"),
):
    """
    Create a new book project.

    Examples:

      sciknow book create "Global Cooling: The Coming Solar Minimum"

      sciknow book create "Solar Climate" --description "The role of the sun in climate change"
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        existing = session.execute(text(
            "SELECT id FROM books WHERE title = :t"
        ), {"t": title}).fetchone()
        if existing:
            console.print(f"[yellow]Book already exists:[/yellow] {title}")
            raise typer.Exit(1)

        result = session.execute(text("""
            INSERT INTO books (title, description)
            VALUES (:title, :desc)
            RETURNING id::text
        """), {"title": title, "desc": description})
        book_id = result.fetchone()[0]
        session.commit()

    console.print(f"[green]✓ Created book:[/green] [bold]{title}[/bold]  [dim](id: {book_id[:8]})[/dim]")
    console.print(
        "\nNext steps:\n"
        f"  [bold]sciknow book outline {title!r}[/bold]   — generate a chapter structure\n"
        f"  [bold]sciknow book chapter add {title!r} \"Chapter Title\"[/bold]  — add chapters manually"
    )


# ── list ───────────────────────────────────────────────────────────────────────

@app.command(name="list")
def list_books():
    """List all book projects."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        rows = session.execute(text("""
            SELECT b.id::text, b.title, b.status, b.description,
                   COUNT(DISTINCT bc.id) as n_chapters,
                   COUNT(DISTINCT d.id)  as n_drafts
            FROM books b
            LEFT JOIN book_chapters bc ON bc.book_id = b.id
            LEFT JOIN drafts d ON d.book_id = b.id
            GROUP BY b.id, b.title, b.status, b.description
            ORDER BY b.created_at DESC
        """)).fetchall()

    if not rows:
        console.print("[yellow]No books yet.[/yellow]  Create one: [bold]sciknow book create \"Title\"[/bold]")
        raise typer.Exit(0)

    table = Table(title="Book Projects", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("ID",       style="dim",   width=8)
    table.add_column("Title",                   ratio=3)
    table.add_column("Status",   style="cyan",  width=12)
    table.add_column("Chapters", justify="right", width=9)
    table.add_column("Drafts",   justify="right", width=7)
    table.add_column("Description",             ratio=2)

    for row in rows:
        table.add_row(
            row[0][:8], row[1], row[2],
            str(row[4]), str(row[5]),
            (row[3] or "")[:60],
        )
    console.print(table)


# ── show ───────────────────────────────────────────────────────────────────────

@app.command()
def show(
    title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
):
    """Show a book's chapters, draft status, and progress."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, title)
        if not book:
            console.print(f"[red]Book not found:[/red] {title}")
            raise typer.Exit(1)

        chapters = session.execute(text("""
            SELECT bc.number, bc.title, bc.description, bc.topic_query, bc.topic_cluster,
                   COUNT(d.id) as n_drafts,
                   COALESCE(SUM(d.word_count), 0) as total_words
            FROM book_chapters bc
            LEFT JOIN drafts d ON d.chapter_id = bc.id
            WHERE bc.book_id = :bid
            GROUP BY bc.number, bc.title, bc.description, bc.topic_query, bc.topic_cluster
            ORDER BY bc.number
        """), {"bid": book[0]}).fetchall()

    console.print()
    console.print(f"[bold cyan]{book[1]}[/bold cyan]  [dim]({book[3]})[/dim]")
    if book[2]:
        console.print(f"  [dim]{book[2]}[/dim]")
    console.print()

    if not chapters:
        console.print("  [yellow]No chapters yet.[/yellow]")
        console.print(f"  Run: [bold]sciknow book outline {title!r}[/bold]")
        return

    table = Table(box=box.SIMPLE_HEAD, show_header=True, expand=True)
    table.add_column("Ch.", style="bold cyan", width=4)
    table.add_column("Title",                  ratio=2)
    table.add_column("Topic Query",  style="dim", ratio=2)
    table.add_column("Cluster",     style="dim", width=16)
    table.add_column("Drafts",      justify="right", width=7)
    table.add_column("Words",       justify="right", width=7)

    for ch in chapters:
        status_color = "green" if ch[5] > 0 else "yellow"
        table.add_row(
            str(ch[0]),
            ch[1],
            (ch[3] or "")[:40],
            (ch[4] or "")[:16],
            f"[{status_color}]{ch[5]}[/{status_color}]",
            str(ch[6]) if ch[6] else "—",
        )
    console.print(table)
    console.print()


# ── chapter add ────────────────────────────────────────────────────────────────

@chapter_app.command(name="add")
def chapter_add(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    title:      Annotated[str, typer.Argument(help="Chapter title.")],
    number:     int | None = typer.Option(None, "--number", "-n", help="Chapter number (auto if omitted)."),
    description: str | None = typer.Option(None, "--description", "-d"),
    topic_query: str | None = typer.Option(None, "--query", "-q",
                                            help="Search query to retrieve relevant papers."),
    topic_cluster: str | None = typer.Option(None, "--cluster", "-c",
                                              help="Topic cluster to scope retrieval."),
):
    """
    Add a chapter to a book.

    Examples:

      sciknow book chapter add "Global Cooling" "Solar Activity and Irradiance" --number 2

      sciknow book chapter add "Global Cooling" "Cosmic Rays and Cloud Nucleation" \\
          --query "cosmic rays cloud nucleation climate" --number 3
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

        if number is None:
            max_n = session.execute(text(
                "SELECT COALESCE(MAX(number), 0) FROM book_chapters WHERE book_id = :bid"
            ), {"bid": book[0]}).scalar()
            number = max_n + 1

        session.execute(text("""
            INSERT INTO book_chapters (book_id, number, title, description, topic_query, topic_cluster)
            VALUES (:bid, :num, :title, :desc, :tq, :tc)
        """), {
            "bid": book[0], "num": number, "title": title,
            "desc": description, "tq": topic_query, "tc": topic_cluster,
        })
        session.commit()

    console.print(
        f"[green]✓ Added Chapter {number}:[/green] [bold]{title}[/bold] "
        f"→ [bold]{book[1]}[/bold]"
    )


# ── outline ────────────────────────────────────────────────────────────────────

@app.command()
def outline(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    save:  bool = typer.Option(True,  "--save/--no-save",
                                help="Save proposed chapters to the database."),
    model: str | None = typer.Option(None, "--model", help="Override LLM model."),
):
    """
    Generate a proposed chapter structure using the LLM and your paper collection.

    Examples:

      sciknow book outline "Global Cooling"

      sciknow book outline "Global Cooling" --no-save   # preview only
    """
    import json
    from sqlalchemy import text
    from sciknow.rag import prompts
    from sciknow.rag.llm import complete
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

        papers = session.execute(text(
            "SELECT title, year FROM paper_metadata ORDER BY year DESC NULLS LAST"
        )).fetchall()

    console.print(f"Generating outline for [bold]{book[1]}[/bold] from {len(papers)} papers…")

    system, user = prompts.outline(
        book_title=book[1],
        papers=[{"title": p[0], "year": p[1]} for p in papers if p[0]],
    )

    with console.status("[bold green]Asking LLM for chapter structure…"):
        raw = complete(system, user, model=model, temperature=0.3, num_ctx=16384)

    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(raw)
        chapters = data.get("chapters", [])
    except Exception:
        console.print("[red]LLM returned invalid JSON. Raw output:[/red]")
        console.print(raw)
        raise typer.Exit(1)

    if not chapters:
        console.print("[yellow]No chapters in LLM response.[/yellow]")
        raise typer.Exit(1)

    # Display
    console.print()
    console.print(Rule(f"[bold]Proposed outline: {book[1]}[/bold]"))
    console.print()
    for ch in chapters:
        console.print(f"  [bold cyan]Ch.{ch['number']}[/bold cyan]  {ch['title']}")
        if ch.get("description"):
            console.print(f"         [dim]{ch['description']}[/dim]")
        if ch.get("topic_query"):
            console.print(f"         [dim]Query: {ch['topic_query']}[/dim]")
        console.print()

    if save:
        with get_session() as session:
            for ch in chapters:
                existing = session.execute(text("""
                    SELECT id FROM book_chapters WHERE book_id = :bid AND number = :num
                """), {"bid": book[0], "num": ch["number"]}).fetchone()
                if existing:
                    continue
                session.execute(text("""
                    INSERT INTO book_chapters (book_id, number, title, description, topic_query)
                    VALUES (:bid, :num, :title, :desc, :tq)
                """), {
                    "bid": book[0],
                    "num": ch["number"],
                    "title": ch["title"],
                    "desc": ch.get("description"),
                    "tq": ch.get("topic_query"),
                })
            session.commit()
        console.print(f"[green]✓ Saved {len(chapters)} chapters to database.[/green]")
        console.print(f"Run [bold]sciknow book show {book_title!r}[/bold] to review.")


# ── write ──────────────────────────────────────────────────────────────────────

@app.command()
def write(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    chapter:    Annotated[str, typer.Argument(help="Chapter number or title fragment.")],
    section:    str = typer.Option("introduction", "--section", "-s",
                                    help="Section type: introduction, methods, results, discussion, conclusion."),
    context_k:  int = typer.Option(12, "--context-k", "-k"),
    candidates: int = typer.Option(50, "--candidates"),
    year_from:  int | None = typer.Option(None, "--year-from"),
    year_to:    int | None = typer.Option(None, "--year-to"),
    model:      str | None = typer.Option(None, "--model"),
    no_save:    bool = typer.Option(False, "--no-save", help="Print only, don't save to DB."),
):
    """
    Draft a chapter section and save it.

    Uses the chapter's topic_query (or the chapter title) to retrieve
    the most relevant papers, then streams a draft and saves it.

    Examples:

      sciknow book write "Global Cooling" 1 --section introduction

      sciknow book write "Global Cooling" "Solar Activity" --section discussion
    """
    from sqlalchemy import text
    from sciknow.rag import prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

        ch = _get_chapter(session, book[0], chapter)
        if not ch:
            console.print(f"[red]Chapter not found:[/red] {chapter}")
            raise typer.Exit(1)

        ch_id, ch_num, ch_title, ch_desc, topic_query, topic_cluster = ch
        search_query = f"{section} {topic_query or ch_title}"

        qdrant = get_client()
        candidates_list = hybrid_search.search(
            query=search_query,
            qdrant_client=qdrant,
            session=session,
            candidate_k=candidates,
            year_from=year_from,
            year_to=year_to,
            topic_cluster=topic_cluster,
        )

        if not candidates_list:
            console.print("[yellow]No relevant passages found.[/yellow]")
            raise typer.Exit(0)

        candidates_list = reranker.rerank(search_query, candidates_list, top_k=context_k)
        results = context_builder.build(candidates_list, session)

    system, user = prompts.write_section(section, topic_query or ch_title, results)

    console.print()
    console.print(Rule(f"[bold]Ch.{ch_num}: {ch_title}[/bold] — {section.capitalize()}"))
    console.print()

    output_tokens: list[str] = []
    for token in llm_stream(system, user, model=model):
        console.print(token, end="", highlight=False)
        output_tokens.append(token)
    console.print()
    console.print()

    content = "".join(output_tokens)
    word_count = len(content.split())

    from sciknow.rag.prompts import format_sources
    source_lines = format_sources(results).splitlines()
    console.print(Rule("[dim]Sources[/dim]"))
    for line in source_lines:
        console.print(f"  [dim]{line}[/dim]")
    console.print()

    if not no_save:
        draft_title = f"Ch.{ch_num} {ch_title} — {section.capitalize()}"
        with get_session() as session:
            session.execute(text("""
                INSERT INTO drafts (title, book_id, chapter_id, section_type, topic, content,
                                    word_count, sources, model_used)
                VALUES (:title, :book_id, :chapter_id, :section, :topic, :content,
                        :wc, :sources::jsonb, :model)
            """), {
                "title":      draft_title,
                "book_id":    book[0],
                "chapter_id": ch_id,
                "section":    section,
                "topic":      topic_query or ch_title,
                "content":    content,
                "wc":         word_count,
                "sources":    __import__("json").dumps(source_lines),
                "model":      model,
            })
            session.commit()
        console.print(f"[green]✓ Saved draft:[/green] [bold]{draft_title}[/bold]  ({word_count} words)")
        console.print("View with: [bold]sciknow draft list --book " + repr(book[1]) + "[/bold]")


# ── argue ──────────────────────────────────────────────────────────────────────

@app.command()
def argue(
    claim:      Annotated[str, typer.Argument(help="The claim to map arguments for.")],
    context_k:  int = typer.Option(15, "--context-k", "-k"),
    candidates: int = typer.Option(60, "--candidates"),
    year_from:  int | None = typer.Option(None, "--year-from"),
    year_to:    int | None = typer.Option(None, "--year-to"),
    book_title: str | None = typer.Option(None, "--book", "-b",
                                           help="Scope to a book's topic clusters."),
    model:      str | None = typer.Option(None, "--model"),
    save:       bool = typer.Option(False, "--save", help="Save the argument map as a draft."),
):
    """
    Map the evidence for and against a claim from the literature.

    Retrieves the most relevant passages and classifies them as
    SUPPORTS / CONTRADICTS / NEUTRAL, then writes a structured argument map.

    Examples:

      sciknow book argue "solar activity is the primary driver of 20th century warming"

      sciknow book argue "cosmic rays modulate cloud cover" --book "Global Cooling" --save
    """
    from sqlalchemy import text
    from sciknow.rag import prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    qdrant = get_client()

    with get_session() as session:
        book_id = None
        if book_title:
            book = _get_book(session, book_title)
            if book:
                book_id = book[0]

        candidates_list = hybrid_search.search(
            query=claim,
            qdrant_client=qdrant,
            session=session,
            candidate_k=candidates,
            year_from=year_from,
            year_to=year_to,
        )
        if not candidates_list:
            console.print("[yellow]No relevant passages found.[/yellow]")
            raise typer.Exit(0)

        candidates_list = reranker.rerank(claim, candidates_list, top_k=context_k)
        results = context_builder.build(candidates_list, session)

    system, user = prompts.argue(claim, results)

    console.print()
    console.print(Rule(f"[bold]Argument Map:[/bold] {claim[:80]}"))
    console.print()

    output_tokens: list[str] = []
    for token in llm_stream(system, user, model=model, num_ctx=16384):
        console.print(token, end="", highlight=False)
        output_tokens.append(token)
    console.print()

    from sciknow.rag.prompts import format_sources
    console.print()
    console.print(Rule("[dim]Sources[/dim]"))
    source_lines = format_sources(results).splitlines()
    for line in source_lines:
        console.print(f"  [dim]{line}[/dim]")
    console.print()

    if save:
        content = "".join(output_tokens)
        draft_title = f"Argument Map: {claim[:60]}"
        with get_session() as session:
            session.execute(text("""
                INSERT INTO drafts (title, book_id, section_type, topic, content,
                                    word_count, sources, model_used)
                VALUES (:title, :book_id, 'argument_map', :topic, :content,
                        :wc, :sources::jsonb, :model)
            """), {
                "title":   draft_title,
                "book_id": book_id,
                "topic":   claim,
                "content": content,
                "wc":      len(content.split()),
                "sources": __import__("json").dumps(source_lines),
                "model":   model,
            })
            session.commit()
        console.print(f"[green]✓ Saved argument map as draft.[/green]")


# ── gaps ───────────────────────────────────────────────────────────────────────

@app.command()
def gaps(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    model: str | None = typer.Option(None, "--model"),
):
    """
    Identify gaps in the book: missing topics, weak chapters, and unwritten sections.

    Examples:

      sciknow book gaps "Global Cooling"
    """
    from sqlalchemy import text
    from sciknow.rag import prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

        chapters = session.execute(text("""
            SELECT number, title, description FROM book_chapters
            WHERE book_id = :bid ORDER BY number
        """), {"bid": book[0]}).fetchall()

        papers = session.execute(text(
            "SELECT title, year FROM paper_metadata ORDER BY year DESC NULLS LAST LIMIT 150"
        )).fetchall()

        drafts = session.execute(text("""
            SELECT d.title, d.section_type, bc.number as chapter_number
            FROM drafts d
            LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.book_id = :bid
            ORDER BY bc.number, d.created_at
        """), {"bid": book[0]}).fetchall()

    if not chapters:
        console.print("[yellow]No chapters defined yet.[/yellow]")
        console.print(f"Run [bold]sciknow book outline {book_title!r}[/bold] first.")
        raise typer.Exit(0)

    system, user = prompts.gaps(
        book_title=book[1],
        chapters=[{"number": c[0], "title": c[1], "description": c[2]} for c in chapters],
        papers=[{"title": p[0], "year": p[1]} for p in papers if p[0]],
        drafts=[{"title": d[0], "section_type": d[1], "chapter_number": d[2]} for d in drafts],
    )

    console.print()
    console.print(Rule(f"[bold]Gap Analysis:[/bold] {book[1]}"))
    console.print()

    for token in llm_stream(system, user, model=model, num_ctx=16384):
        console.print(token, end="", highlight=False)
    console.print()
    console.print()


# ── export ─────────────────────────────────────────────────────────────────────

@app.command()
def export(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    output:     Path | None = typer.Option(None, "--output", "-o",
                                            help="Output file (default: <book_title>.md)."),
    include_sources: bool = typer.Option(True, "--sources/--no-sources",
                                          help="Include source citations per chapter."),
):
    """
    Compile all chapter drafts into a single Markdown document.

    Drafts are ordered by chapter number, then by section type
    (introduction → methods → results → discussion → conclusion).

    Examples:

      sciknow book export "Global Cooling"

      sciknow book export "Global Cooling" --output my_book.md --no-sources
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    _SECTION_ORDER = {
        "introduction": 0, "methods": 1, "results": 2,
        "discussion": 3, "conclusion": 4,
    }

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

        chapters = session.execute(text("""
            SELECT id::text, number, title, description
            FROM book_chapters WHERE book_id = :bid ORDER BY number
        """), {"bid": book[0]}).fetchall()

        drafts = session.execute(text("""
            SELECT d.chapter_id::text, d.title, d.section_type, d.content,
                   d.word_count, d.sources, bc.number
            FROM drafts d
            LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.book_id = :bid
            ORDER BY bc.number, d.created_at
        """), {"bid": book[0]}).fetchall()

    if not drafts:
        console.print("[yellow]No drafts found for this book.[/yellow]")
        console.print(f"Use [bold]sciknow book write {book_title!r} <chapter>[/bold] to create drafts.")
        raise typer.Exit(0)

    # Group drafts by chapter_id
    from collections import defaultdict
    chapter_drafts: dict[str, list] = defaultdict(list)
    for d in drafts:
        chapter_drafts[d[0] or "__none__"].append(d)

    # Sort drafts within each chapter by section order
    for ch_id in chapter_drafts:
        chapter_drafts[ch_id].sort(
            key=lambda d: _SECTION_ORDER.get(d[2] or "", 99)
        )

    # Collect all sources for bibliography
    all_sources: list[str] = []
    seen_sources: set[str] = set()

    lines: list[str] = []
    lines += [f"# {book[1]}", ""]
    if book[2]:
        lines += [f"*{book[2]}*", ""]
    lines += ["---", ""]

    for ch in chapters:
        ch_id, ch_num, ch_title, ch_desc = ch
        ch_drafts = chapter_drafts.get(ch_id, [])

        lines += [f"## Chapter {ch_num}: {ch_title}", ""]
        if ch_desc:
            lines += [f"*{ch_desc}*", ""]

        if not ch_drafts:
            lines += ["*[No draft yet]*", ""]
            continue

        for draft in ch_drafts:
            _, d_title, d_section, d_content, d_words, d_sources, _ = draft
            if d_section and d_section not in ("argument_map",):
                lines += [f"### {d_section.capitalize()}", ""]
            lines += [d_content, ""]

            if include_sources:
                for s in (d_sources or []):
                    if s and s not in seen_sources:
                        seen_sources.add(s)
                        all_sources.append(s)

    # Bibliography
    if include_sources and all_sources:
        lines += ["---", "", "## Bibliography", ""]
        for i, s in enumerate(all_sources, 1):
            lines.append(f"{i}. {s}")
        lines.append("")

    # Word count
    total_words = sum(d[4] or 0 for d in drafts)
    lines += ["", f"---", f"*{total_words:,} words · {len(drafts)} sections · {len(chapters)} chapters*"]

    md = "\n".join(lines)

    if output is None:
        safe = book[1].lower().replace(" ", "_").replace("/", "-")[:50]
        output = Path(f"{safe}.md")

    output.write_text(md, encoding="utf-8")
    console.print(
        f"[green]✓ Exported [bold]{book[1]}[/bold][/green] → [bold]{output}[/bold]  "
        f"({total_words:,} words, {len(drafts)} sections)"
    )
