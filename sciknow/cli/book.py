"""
sciknow book — manage book projects, chapters, outlines, and writing.

Commands:
  create          Create a new book project
  list            List all books
  show            Show a book's chapters and draft status
  chapter add     Add a chapter to a book
  outline         Generate a chapter structure from the literature (LLM)
  plan            Generate / view / edit the book plan (thesis + scope)
  write           Draft a chapter section (with cross-chapter coherence)
  review          Run a critic pass over a saved draft
  revise          Revise a draft based on instructions or review feedback
  argue           Map the argument for a claim against the literature
  gaps            Identify + persist what's missing in the book
  export          Compile all chapter drafts into Markdown, LaTeX, or DOCX
"""
from __future__ import annotations

import json as _json
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
        SELECT id::text, title, description, status, created_at, plan
        FROM books WHERE title ILIKE :q OR id::text LIKE :q
        LIMIT 1
    """), {"q": f"%{title_or_id}%"}).fetchone()
    return row


def _get_chapter(session, book_id: str, chapter_ref: str):
    """Find a chapter by number or title fragment."""
    from sqlalchemy import text
    if chapter_ref.isdigit():
        row = session.execute(text("""
            SELECT id::text, number, title, description, topic_query, topic_cluster
            FROM book_chapters WHERE book_id = :bid AND number = :num
        """), {"bid": book_id, "num": int(chapter_ref)}).fetchone()
        if row:
            return row
    return session.execute(text("""
        SELECT id::text, number, title, description, topic_query, topic_cluster
        FROM book_chapters WHERE book_id = :bid AND title ILIKE :q
        LIMIT 1
    """), {"bid": book_id, "q": f"%{chapter_ref}%"}).fetchone()


def _get_prior_summaries(session, book_id: str, before_chapter_number: int) -> list[dict]:
    """Return summaries of all drafts from chapters before the given chapter number."""
    from sqlalchemy import text
    rows = session.execute(text("""
        SELECT bc.number, d.section_type, d.summary
        FROM drafts d
        JOIN book_chapters bc ON bc.id = d.chapter_id
        WHERE d.book_id = :bid AND bc.number < :ch_num AND d.summary IS NOT NULL
        ORDER BY bc.number, d.section_type
    """), {"bid": book_id, "ch_num": before_chapter_number}).fetchall()
    return [{"chapter_number": r[0], "section_type": r[1], "summary": r[2]} for r in rows]


def _auto_summarize(content: str, section_type: str, chapter_title: str, model: str | None = None) -> str:
    """Generate a 100-200 word summary of a draft for cross-chapter context."""
    from sciknow.rag import prompts
    from sciknow.rag.llm import complete_with_status
    system, user = prompts.draft_summary(section_type, chapter_title, content)
    try:
        return complete_with_status(
            system, user, label="Summarizing for coherence",
            model=model, temperature=0.1, num_ctx=4096,
        ).strip()
    except Exception:
        return ""


def _save_draft(session, *, title, book_id, chapter_id, section_type, topic,
                content, sources, model, summary=None, parent_draft_id=None,
                review_feedback=None, version=1):
    """Insert a draft row and return the draft_id."""
    from sqlalchemy import text
    row = session.execute(text("""
        INSERT INTO drafts (title, book_id, chapter_id, section_type, topic, content,
                            word_count, sources, model_used, version, summary,
                            parent_draft_id, review_feedback)
        VALUES (:title, :book_id, :chapter_id, :section, :topic, :content,
                :wc, :sources::jsonb, :model, :version, :summary,
                :parent_id, :review_feedback)
        RETURNING id::text
    """), {
        "title": title,
        "book_id": book_id,
        "chapter_id": chapter_id,
        "section": section_type,
        "topic": topic,
        "content": content,
        "wc": len(content.split()),
        "sources": _json.dumps(sources or []),
        "model": model,
        "version": version,
        "summary": summary,
        "parent_id": parent_draft_id,
        "review_feedback": review_feedback,
    })
    session.commit()
    return row.fetchone()[0]


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

    from sciknow.rag.llm import complete_with_status
    raw = complete_with_status(system, user, label="Generating outline", model=model, temperature=0.3, num_ctx=16384)

    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(raw, strict=False)
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
    expand:     bool = typer.Option(False, "--expand", "-e", help="Expand query with LLM synonyms before retrieval."),
    show_plan:  bool = typer.Option(False, "--plan", help="Show a sentence plan before drafting."),
    verify:     bool = typer.Option(False, "--verify", help="Run claim verification after drafting."),
    ipcc:       bool = typer.Option(False, "--ipcc", help="Add IPCC-style calibrated uncertainty language."),
):
    """
    Draft a chapter section with cross-chapter coherence.

    Injects the book plan and summaries of prior chapters into the prompt
    so the LLM maintains consistency across the book. Auto-generates a
    summary after saving for use by subsequent chapters.

    Examples:

      sciknow book write "Global Cooling" 1 --section introduction

      sciknow book write "Global Cooling" 2 --section methods --plan --verify

      sciknow book write "Global Cooling" 3 --section results --ipcc --expand
    """
    from sciknow.rag import prompts
    from sciknow.rag.llm import stream as llm_stream, complete as llm_complete, complete_with_status
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

        book_id, b_title, b_desc, b_status, b_created, b_plan = book

        ch = _get_chapter(session, book_id, chapter)
        if not ch:
            console.print(f"[red]Chapter not found:[/red] {chapter}")
            raise typer.Exit(1)

        ch_id, ch_num, ch_title, ch_desc, topic_query, topic_cluster = ch
        search_query = f"{section} {topic_query or ch_title}"

        # Retrieve prior chapter summaries for cross-chapter coherence
        prior_summaries = _get_prior_summaries(session, book_id, ch_num)
        if prior_summaries:
            console.print(f"[dim]Injecting {len(prior_summaries)} prior chapter summaries for coherence.[/dim]")

        qdrant = get_client()
        candidates_list = hybrid_search.search(
            query=search_query,
            qdrant_client=qdrant,
            session=session,
            candidate_k=candidates,
            year_from=year_from,
            year_to=year_to,
            topic_cluster=topic_cluster,
            use_query_expansion=expand,
        )

        if not candidates_list:
            console.print("[yellow]No relevant passages found.[/yellow]")
            raise typer.Exit(0)

        candidates_list = reranker.rerank(search_query, candidates_list, top_k=context_k)
        results = context_builder.build(candidates_list, session)

    # ── Optional: sentence plan ──────────────────────────────────────────
    if show_plan:
        console.print()
        console.print(Rule("[bold]Sentence Plan[/bold]"))
        console.print()
        sys_p, usr_p = prompts.sentence_plan(
            section, topic_query or ch_title, results,
            book_plan=b_plan,
            prior_summaries=prior_summaries,
        )
        for token in llm_stream(sys_p, usr_p, model=model):
            console.print(token, end="", highlight=False)
        console.print()
        console.print()

    # ── Draft with v2 prompt (book plan + prior summaries) ───────────────
    system, user = prompts.write_section_v2(
        section, topic_query or ch_title, results,
        book_plan=b_plan,
        prior_summaries=prior_summaries,
    )

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

    # ── Optional: IPCC uncertainty language ───────────────────────────────
    if ipcc:
        console.print(Rule("[dim]Applying IPCC calibrated uncertainty language[/dim]"))
        sys_i, usr_i = prompts.ipcc_uncertainty(content, results)
        content = complete_with_status(sys_i, usr_i, label="IPCC pass", model=model, temperature=0.1, num_ctx=16384)
        console.print("[green]✓ IPCC uncertainty language applied.[/green]")
        console.print()

    from sciknow.rag.prompts import format_sources
    source_lines = format_sources(results).splitlines()
    console.print(Rule("[dim]Sources[/dim]"))
    for line in source_lines:
        console.print(f"  [dim]{line}[/dim]")
    console.print()

    # ── Optional: claim verification ─────────────────────────────────────
    verify_feedback = None
    if verify:
        console.print(Rule("[dim]Verifying claims[/dim]"))
        sys_v, usr_v = prompts.verify_claims(content, results)
        try:
            raw = complete_with_status(sys_v, usr_v, label="Verifying claims", model=model, temperature=0.0, num_ctx=16384)
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            vdata = _json.loads(cleaned, strict=False)
            score = vdata.get("groundedness_score", "?")
            console.print(f"  Groundedness score: [bold]{score}[/bold]")
            for claim in vdata.get("claims", []):
                v = claim.get("verdict", "?")
                colour = "green" if v == "SUPPORTED" else "yellow" if v == "EXTRAPOLATED" else "red"
                console.print(f"  [{colour}]{v}[/{colour}]: {claim.get('text', '')[:80]}")
            unsupported = vdata.get("unsupported_claims", [])
            if unsupported:
                console.print(f"  [red]Unsupported claims ({len(unsupported)}):[/red]")
                for u in unsupported[:5]:
                    console.print(f"    [red]- {u[:80]}[/red]")
            verify_feedback = raw
        except Exception as exc:
            console.print(f"  [yellow]Verification failed: {exc}[/yellow]")
        console.print()

    # ── Save draft ───────────────────────────────────────────────────────
    if not no_save:
        # Auto-generate a summary for cross-chapter context
        with console.status("[dim]Generating summary for future chapters...[/dim]"):
            summary = _auto_summarize(content, section, ch_title, model=model)

        draft_title = f"Ch.{ch_num} {ch_title} — {section.capitalize()}"
        with get_session() as session:
            draft_id = _save_draft(
                session,
                title=draft_title,
                book_id=book_id,
                chapter_id=ch_id,
                section_type=section,
                topic=topic_query or ch_title,
                content=content,
                sources=source_lines,
                model=model,
                summary=summary,
                review_feedback=verify_feedback,
            )
        console.print(
            f"[green]✓ Saved draft:[/green] [bold]{draft_title}[/bold]  "
            f"({len(content.split())} words)"
        )


# ── plan ───────────────────────────────────────────────────────────────────────

@app.command()
def plan(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    model: str | None = typer.Option(None, "--model"),
    edit: bool = typer.Option(False, "--edit", help="Open the plan in $EDITOR for manual editing after generation."),
):
    """
    Generate or view the book plan (thesis + scope document).

    The book plan is a 200-500 word document defining the central argument,
    scope, audience, and key terms. It's injected into every `book write`
    call so all chapters stay aligned.

    If the book already has a plan, prints it. Use --edit to regenerate or
    modify it.

    Examples:

      sciknow book plan "Global Cooling"

      sciknow book plan "Global Cooling" --edit
    """
    from sqlalchemy import text
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream, complete as llm_complete
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

        book_id, b_title, b_desc, b_status, b_created, b_plan = book

        # If plan exists and not editing, just show it
        if b_plan and not edit:
            console.print()
            console.print(Rule(f"[bold]Book Plan:[/bold] {b_title}"))
            console.print()
            console.print(b_plan)
            console.print()
            console.print("[dim]Use --edit to regenerate or modify.[/dim]")
            return

        # Load chapters and papers for plan generation
        chapters = session.execute(text("""
            SELECT number, title, description FROM book_chapters
            WHERE book_id = :bid ORDER BY number
        """), {"bid": book_id}).fetchall()
        ch_list = [{"number": r[0], "title": r[1], "description": r[2] or ""} for r in chapters]

        papers = session.execute(text("""
            SELECT pm.title, pm.year FROM paper_metadata pm
            JOIN documents d ON d.id = pm.document_id
            WHERE d.ingestion_status = 'complete' AND pm.title IS NOT NULL
            ORDER BY pm.year DESC NULLS LAST
        """)).fetchall()
        paper_list = [{"title": r[0], "year": r[1]} for r in papers]

    # Generate plan
    console.print()
    console.print(Rule(f"[bold]Generating Book Plan:[/bold] {b_title}"))
    console.print()

    sys_p, usr_p = rag_prompts.book_plan(b_title, b_desc, ch_list, paper_list)
    output_tokens = []
    for token in llm_stream(sys_p, usr_p, model=model):
        console.print(token, end="", highlight=False)
        output_tokens.append(token)
    console.print()
    console.print()

    new_plan = "".join(output_tokens).strip()

    # Save
    with get_session() as session:
        session.execute(text("UPDATE books SET plan = :plan WHERE id::text = :bid"),
                        {"plan": new_plan, "bid": book_id})
        session.commit()
    console.print("[green]✓ Book plan saved.[/green]")
    console.print("[dim]This plan will be injected into all future `book write` calls.[/dim]")


# ── review ─────────────────────────────────────────────────────────────────────

@app.command()
def review(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (first 8+ chars).")],
    model: str | None = typer.Option(None, "--model"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save review feedback to the draft."),
):
    """
    Run a critic pass over a saved draft.

    Assesses groundedness, completeness, accuracy, coherence, and redundancy.
    Produces structured feedback with specific quotes and actionable suggestions.
    The feedback is saved to the draft's review_feedback field.

    Examples:

      sciknow book review 3f2a1b4c

      sciknow book review 3f2a1b4c --no-save
    """
    from sqlalchemy import text
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    with get_session() as session:
        row = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.topic, d.content,
                   d.book_id::text, d.chapter_id::text
            FROM drafts d WHERE d.id::text LIKE :q
            LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()

    if not row:
        console.print(f"[red]Draft not found:[/red] {draft_id}")
        raise typer.Exit(1)

    d_id, d_title, d_section, d_topic, d_content, d_book_id, d_chapter_id = row

    # Retrieve the same passages the draft was based on
    qdrant = get_client()
    search_query = f"{d_section or ''} {d_topic or d_title}"
    with get_session() as session:
        candidates_list = hybrid_search.search(
            query=search_query, qdrant_client=qdrant, session=session, candidate_k=50,
        )
        if candidates_list:
            candidates_list = reranker.rerank(search_query, candidates_list, top_k=12)
            results = context_builder.build(candidates_list, session)
        else:
            results = []

    console.print()
    console.print(Rule(f"[bold]Reviewing:[/bold] {d_title}"))
    console.print()

    sys_r, usr_r = rag_prompts.review(d_section, d_topic or d_title, d_content, results)
    output_tokens = []
    for token in llm_stream(sys_r, usr_r, model=model):
        console.print(token, end="", highlight=False)
        output_tokens.append(token)
    console.print()

    feedback = "".join(output_tokens).strip()

    if save:
        with get_session() as session:
            session.execute(text("UPDATE drafts SET review_feedback = :fb WHERE id::text = :did"),
                            {"fb": feedback, "did": d_id})
            session.commit()
        console.print()
        console.print("[green]✓ Review feedback saved to draft.[/green]")
        console.print("[dim]Use `sciknow book revise` to apply the feedback.[/dim]")


# ── revise ─────────────────────────────────────────────────────────────────────

@app.command()
def revise(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (first 8+ chars).")],
    instruction: str = typer.Option("", "--instruction", "-i",
                                     help="Revision instruction (or leave empty to apply saved review feedback)."),
    context_k: int = typer.Option(8, "--context-k", "-k", help="Additional passages to retrieve."),
    model: str | None = typer.Option(None, "--model"),
):
    """
    Revise a draft based on instructions or saved review feedback.

    Creates a new version (version N+1) linked to the original via
    parent_draft_id. The original is preserved. If no --instruction is
    given, uses the draft's saved review_feedback from `book review`.

    Examples:

      sciknow book revise 3f2a1b4c -i "expand the section on solar cycles"

      sciknow book revise 3f2a1b4c -i "add counterarguments from the skeptic literature"

      sciknow book revise 3f2a1b4c       # uses saved review feedback
    """
    from sqlalchemy import text
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    with get_session() as session:
        row = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.topic, d.content,
                   d.book_id::text, d.chapter_id::text, d.version,
                   d.review_feedback, d.sources
            FROM drafts d WHERE d.id::text LIKE :q
            LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()

    if not row:
        console.print(f"[red]Draft not found:[/red] {draft_id}")
        raise typer.Exit(1)

    d_id, d_title, d_section, d_topic, d_content, d_book_id, d_chapter_id, \
        d_version, d_review_feedback, d_sources = row

    # Determine revision instruction
    rev_instruction = instruction.strip()
    if not rev_instruction:
        if d_review_feedback:
            rev_instruction = f"Apply the following review feedback:\n\n{d_review_feedback}"
            console.print("[dim]Using saved review feedback as revision instruction.[/dim]")
        else:
            console.print("[red]No instruction provided and no saved review feedback.[/red]")
            console.print("Use: sciknow book revise <id> -i \"your instruction\"")
            raise typer.Exit(1)

    # Optionally retrieve additional passages for evidence
    results = []
    if context_k > 0:
        qdrant = get_client()
        search_query = f"{d_section or ''} {d_topic or d_title}"
        with get_session() as session:
            candidates_list = hybrid_search.search(
                query=search_query, qdrant_client=qdrant, session=session, candidate_k=50,
            )
            if candidates_list:
                candidates_list = reranker.rerank(search_query, candidates_list, top_k=context_k)
                results = context_builder.build(candidates_list, session)

    console.print()
    console.print(Rule(f"[bold]Revising:[/bold] {d_title} (v{d_version} → v{d_version + 1})"))
    console.print()

    sys_r, usr_r = rag_prompts.revise(d_content, rev_instruction, results or None)
    output_tokens = []
    for token in llm_stream(sys_r, usr_r, model=model):
        console.print(token, end="", highlight=False)
        output_tokens.append(token)
    console.print()

    revised_content = "".join(output_tokens).strip()

    # Auto-generate summary for the new version
    with console.status("[dim]Generating summary...[/dim]"):
        summary = _auto_summarize(revised_content, d_section or "text", d_title, model=model)

    # Save as new version
    with get_session() as session:
        new_id = _save_draft(
            session,
            title=d_title,
            book_id=d_book_id,
            chapter_id=d_chapter_id,
            section_type=d_section,
            topic=d_topic,
            content=revised_content,
            sources=_json.loads(d_sources) if isinstance(d_sources, str) else (d_sources or []),
            model=model,
            summary=summary,
            parent_draft_id=d_id,
            version=d_version + 1,
        )

    console.print()
    console.print(
        f"[green]✓ Saved as v{d_version + 1}:[/green] [bold]{d_title}[/bold]  "
        f"({len(revised_content.split())} words)"
    )
    console.print(f"[dim]Original (v{d_version}) preserved. New draft ID: {new_id[:8]}[/dim]")


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
    save: bool = typer.Option(True, "--save/--no-save",
                               help="Persist identified gaps to the book_gaps table (default: yes)."),
):
    """
    Identify + persist gaps in the book: missing topics, weak chapters, unwritten sections.

    Runs two passes: a human-readable narrative (streamed), and a structured JSON
    extraction that saves each gap to the book_gaps table for tracking. View saved
    gaps in `book show`.

    Examples:

      sciknow book gaps "Global Cooling"

      sciknow book gaps "Global Cooling" --no-save   # informational only
    """
    from sqlalchemy import text
    from sciknow.rag import prompts
    from sciknow.rag.llm import stream as llm_stream, complete as llm_complete
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

        book_id = book[0]

        chapters = session.execute(text("""
            SELECT number, title, description FROM book_chapters
            WHERE book_id = :bid ORDER BY number
        """), {"bid": book_id}).fetchall()

        papers = session.execute(text(
            "SELECT title, year FROM paper_metadata ORDER BY year DESC NULLS LAST LIMIT 150"
        )).fetchall()

        drafts_rows = session.execute(text("""
            SELECT d.title, d.section_type, bc.number as chapter_number
            FROM drafts d
            LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.book_id = :bid
            ORDER BY bc.number, d.created_at
        """), {"bid": book_id}).fetchall()

    if not chapters:
        console.print("[yellow]No chapters defined yet.[/yellow]")
        console.print(f"Run [bold]sciknow book outline {book_title!r}[/bold] first.")
        raise typer.Exit(0)

    ch_list = [{"number": c[0], "title": c[1], "description": c[2]} for c in chapters]
    p_list = [{"title": p[0], "year": p[1]} for p in papers if p[0]]
    d_list = [{"title": d[0], "section_type": d[1], "chapter_number": d[2]} for d in drafts_rows]

    # Pass 1: human-readable narrative (streamed)
    system, user = prompts.gaps(
        book_title=book[1], chapters=ch_list, papers=p_list, drafts=d_list,
    )

    console.print()
    console.print(Rule(f"[bold]Gap Analysis:[/bold] {book[1]}"))
    console.print()

    for token in llm_stream(system, user, model=model, num_ctx=16384):
        console.print(token, end="", highlight=False)
    console.print()
    console.print()

    # Pass 2: structured JSON extraction → save to book_gaps
    if save:
        from sciknow.rag.llm import complete_with_status as _cws
        sys_j, usr_j = prompts.gaps_json(
            book_title=book[1], chapters=ch_list, papers=p_list, drafts=d_list,
        )
        try:
            raw = _cws(sys_j, usr_j, label="Extracting structured gaps", model=model, temperature=0.0, num_ctx=16384)
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            gap_data = _json.loads(cleaned, strict=False)
            gap_list = gap_data.get("gaps", [])
        except Exception as exc:
            console.print(f"[yellow]Structured gap extraction failed: {exc}[/yellow]")
            gap_list = []

        if gap_list:
            # Build chapter number → chapter_id map
            ch_id_map = {}
            with get_session() as session:
                ch_rows = session.execute(text("""
                    SELECT number, id::text FROM book_chapters WHERE book_id = :bid
                """), {"bid": book_id}).fetchall()
                ch_id_map = {r[0]: r[1] for r in ch_rows}

                # Clear old gaps and insert new ones
                session.execute(text("DELETE FROM book_gaps WHERE book_id = :bid"),
                                {"bid": book_id})
                for g in gap_list:
                    ch_num = g.get("chapter_number")
                    session.execute(text("""
                        INSERT INTO book_gaps (book_id, gap_type, description, chapter_id, status)
                        VALUES (:bid, :gtype, :desc, :ch_id, 'open')
                    """), {
                        "bid": book_id,
                        "gtype": g.get("type", "topic"),
                        "desc": g.get("description", ""),
                        "ch_id": ch_id_map.get(ch_num) if ch_num else None,
                    })
                session.commit()

            console.print(f"[green]✓ Saved {len(gap_list)} gaps to book_gaps table.[/green]")
            console.print("[dim]View with `sciknow book show`.[/dim]")
        else:
            console.print("[dim]No structured gaps extracted.[/dim]")


# ── autowrite (Karpathy-loop-inspired convergence) ─────────────────────────────

_DEFAULT_SECTIONS = ["introduction", "methods", "results", "discussion", "conclusion"]


def _score_draft(draft_content, section_type, topic, session, qdrant, model=None):
    """Run the structured scoring reviewer and return parsed scores dict."""
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import complete_with_status
    from sciknow.retrieval import context_builder, hybrid_search, reranker

    search_query = f"{section_type or ''} {topic or ''}"
    candidates = hybrid_search.search(
        query=search_query, qdrant_client=qdrant, session=session, candidate_k=50,
    )
    if candidates:
        candidates = reranker.rerank(search_query, candidates, top_k=12)
    results = context_builder.build(candidates, session) if candidates else []

    sys_s, usr_s = rag_prompts.score_draft(section_type, topic, draft_content, results)
    raw = complete_with_status(
        sys_s, usr_s, label="Scoring draft",
        model=model, temperature=0.0, num_ctx=16384,
    )

    import re
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    # Strip non-JSON backslashes
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', '', cleaned)
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first >= 0 and last > first:
        cleaned = cleaned[first:last + 1]

    return _json.loads(cleaned, strict=False), results


def _autowrite_section(
    book_id, book_title, book_plan, ch_id, ch_num, ch_title, topic_query,
    topic_cluster, section, model, max_iter, target_score, auto_expand,
    ipcc, console,
):
    """Inner convergence loop for one section. Returns the final draft content."""
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream, complete_with_status
    from sciknow.retrieval import context_builder, hybrid_search, reranker
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    qdrant = get_client()
    topic = topic_query or ch_title

    # ── Step 1: Initial draft ────────────────────────────────────────────
    console.print()
    console.print(Rule(
        f"[bold]Autowrite Ch.{ch_num}: {ch_title} — {section.capitalize()}[/bold]"
    ))

    with get_session() as session:
        prior_summaries = _get_prior_summaries(session, book_id, ch_num)
        search_query = f"{section} {topic}"

        candidates = hybrid_search.search(
            query=search_query, qdrant_client=qdrant, session=session,
            candidate_k=50, topic_cluster=topic_cluster,
        )
        if not candidates:
            console.print("[yellow]No relevant passages found.[/yellow]")
            return None
        candidates = reranker.rerank(search_query, candidates, top_k=12)
        results = context_builder.build(candidates, session)

    system, user = rag_prompts.write_section_v2(
        section, topic, results,
        book_plan=book_plan,
        prior_summaries=prior_summaries,
    )

    console.print("[dim]Generating initial draft...[/dim]")
    tokens = []
    for tok in llm_stream(system, user, model=model):
        console.print(tok, end="", highlight=False)
        tokens.append(tok)
    console.print("\n")

    content = "".join(tokens)
    from sciknow.rag.prompts import format_sources
    sources = format_sources(results).splitlines()

    # ── Step 2: Score → revise → re-score loop ───────────────────────────
    history: list[dict] = []

    for iteration in range(max_iter):
        console.print(Rule(f"[dim]Iteration {iteration + 1}/{max_iter}[/dim]"))

        # Score
        with get_session() as session:
            try:
                scores, scored_results = _score_draft(
                    content, section, topic, session, qdrant, model=model,
                )
            except Exception as exc:
                console.print(f"[yellow]Scoring failed: {exc}[/yellow]")
                scores = {"overall": 0.5, "weakest_dimension": "unknown",
                          "revision_instruction": "Improve overall quality."}

        overall = scores.get("overall", 0)
        weakest = scores.get("weakest_dimension", "unknown")
        instruction = scores.get("revision_instruction", "Improve the draft.")
        missing = scores.get("missing_topics", [])

        history.append(scores)

        # Display scores
        dims = ["groundedness", "completeness", "coherence", "citation_accuracy", "overall"]
        score_line = "  ".join(
            f"[{'green' if scores.get(d, 0) >= target_score else 'yellow' if scores.get(d, 0) >= 0.7 else 'red'}]"
            f"{d[:5]}={scores.get(d, 0):.2f}[/]"
            for d in dims
        )
        console.print(f"  Scores: {score_line}")
        console.print(f"  Weakest: [bold]{weakest}[/bold]")
        console.print(f"  Instruction: [dim]{instruction}[/dim]")

        # Check convergence
        if overall >= target_score:
            console.print(f"\n[green]✓ Converged at iteration {iteration + 1} "
                          f"(overall={overall:.2f} ≥ {target_score})[/green]")
            break

        # Auto-expand if reviewer identified missing topics
        if auto_expand and missing:
            console.print(f"  [cyan]Auto-expanding: {', '.join(missing[:3])}[/cyan]")
            from sciknow.ingestion.downloader import find_and_download
            from sciknow.config import settings
            for topic_q in missing[:2]:  # limit to 2 expansions per iteration
                try:
                    from sciknow.retrieval.relevance import embed_query, score_candidates
                    # Just do a targeted search to see if we already have papers
                    with get_session() as session:
                        extra = hybrid_search.search(
                            query=topic_q, qdrant_client=qdrant, session=session,
                            candidate_k=5,
                        )
                    if len(extra) < 3:
                        console.print(f"    [dim]Few results for '{topic_q}' — would benefit from db expand[/dim]")
                except Exception:
                    pass

        # Revise
        console.print(f"\n  [dim]Revising (targeting {weakest})...[/dim]")
        sys_r, usr_r = rag_prompts.revise(content, instruction, scored_results)
        rev_tokens = []
        for tok in llm_stream(sys_r, usr_r, model=model):
            console.print(tok, end="", highlight=False)
            rev_tokens.append(tok)
        console.print("\n")

        revised = "".join(rev_tokens)

        # Re-score the revision to decide keep/discard
        with get_session() as session:
            try:
                new_scores, _ = _score_draft(
                    revised, section, topic, session, qdrant, model=model,
                )
            except Exception:
                new_scores = {"overall": overall}  # assume no change on scoring failure

        new_overall = new_scores.get("overall", 0)

        if new_overall >= overall:
            console.print(
                f"  [green]✓ KEEP[/green] v{iteration + 2}: "
                f"overall {overall:.2f} → {new_overall:.2f}"
            )
            content = revised
            overall = new_overall
        else:
            console.print(
                f"  [red]✗ DISCARD[/red] v{iteration + 2}: "
                f"overall {overall:.2f} → {new_overall:.2f} (regressed)"
            )
            # Try a different instruction on next iteration
            # The loop will re-score the original content and get a new weakest dimension

    # ── Step 3: IPCC pass (optional) ─────────────────────────────────────
    if ipcc:
        console.print(Rule("[dim]Applying IPCC uncertainty language[/dim]"))
        sys_i, usr_i = rag_prompts.ipcc_uncertainty(content, scored_results if 'scored_results' in dir() else [])
        content = complete_with_status(
            sys_i, usr_i, label="IPCC pass", model=model, temperature=0.1, num_ctx=16384,
        )

    # ── Step 4: Save ─────────────────────────────────────────────────────
    with console.status("[dim]Generating summary...[/dim]"):
        summary = _auto_summarize(content, section, ch_title, model=model)

    draft_title = f"Ch.{ch_num} {ch_title} — {section.capitalize()} (autowrite)"
    with get_session() as session:
        draft_id = _save_draft(
            session,
            title=draft_title,
            book_id=book_id,
            chapter_id=ch_id,
            section_type=section,
            topic=topic,
            content=content,
            sources=sources,
            model=model,
            summary=summary,
            version=len(history) + 1,
        )

    # Print convergence history
    console.print()
    console.print(Rule("[dim]Convergence History[/dim]"))
    for i, h in enumerate(history, 1):
        dims_str = "  ".join(f"{d[:5]}={h.get(d, 0):.2f}" for d in
                             ["groundedness", "completeness", "coherence", "citation_accuracy"])
        console.print(f"  v{i}: {dims_str}  [bold]overall={h.get('overall', 0):.2f}[/bold]")
    console.print()
    console.print(
        f"[green]✓ Saved:[/green] [bold]{draft_title}[/bold]  "
        f"({len(content.split())} words, {len(history)} iterations)"
    )

    return content


@app.command()
def autowrite(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    chapter:    str = typer.Argument(None, help="Chapter number or title fragment (omit with --full for all chapters)."),
    section:    str = typer.Option("introduction", "--section", "-s",
                                    help="Section type (or 'all' for all sections)."),
    max_iter:   int = typer.Option(3, "--max-iter", "-n", help="Max review-revise iterations per section."),
    target_score: float = typer.Option(0.85, "--target-score", "-t",
                                        help="Overall quality score to stop iterating (0.0-1.0)."),
    model:      str | None = typer.Option(None, "--model"),
    auto_expand: bool = typer.Option(False, "--auto-expand",
                                      help="Auto-run db expand when reviewer identifies missing evidence."),
    ipcc:       bool = typer.Option(False, "--ipcc", help="Apply IPCC uncertainty language on final version."),
    full:       bool = typer.Option(False, "--full",
                                     help="Write ALL chapters × ALL sections (the full autonomous pipeline)."),
):
    """
    Autonomous write → review → revise convergence loop (inspired by Karpathy's autoresearch).

    For each section, generates an initial draft, scores it on 5 quality dimensions
    (groundedness, completeness, coherence, citation accuracy, overall), then
    iteratively revises targeting the weakest dimension until the overall score
    reaches --target-score or --max-iter iterations are exhausted.

    Each iteration:
      1. Score the current draft (structured JSON reviewer)
      2. If score >= target → STOP (converged)
      3. Identify the weakest dimension
      4. Generate a targeted revision instruction
      5. Revise the draft
      6. Re-score → if improved KEEP, if regressed DISCARD

    Modes:
      Single section:    sciknow book autowrite "Book" 3 --section methods
      All sections:      sciknow book autowrite "Book" 3 --section all
      Full book:         sciknow book autowrite "Book" --full

    Examples:

      sciknow book autowrite "Global Cooling" 1 --section introduction --max-iter 5

      sciknow book autowrite "Global Cooling" 3 --section all --target-score 0.80 --ipcc

      sciknow book autowrite "Global Cooling" --full --max-iter 3 --auto-expand
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

        book_id, b_title, b_desc, b_status, b_created, b_plan = book

        if not b_plan:
            console.print(
                "[yellow]No book plan set.[/yellow] Generate one first:\n"
                f"  sciknow book plan {book_title!r}"
            )
            raise typer.Exit(1)

        chapters = session.execute(text("""
            SELECT id::text, number, title, description, topic_query, topic_cluster
            FROM book_chapters WHERE book_id = :bid ORDER BY number
        """), {"bid": book_id}).fetchall()

    if not chapters:
        console.print("[yellow]No chapters defined.[/yellow] Run `book outline` first.")
        raise typer.Exit(1)

    # Determine which chapters × sections to write
    if full:
        targets = [
            (ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], sec)
            for ch in chapters
            for sec in _DEFAULT_SECTIONS
        ]
        console.print(
            f"[bold]Autowrite FULL BOOK:[/bold] {b_title}\n"
            f"  {len(chapters)} chapters × {len(_DEFAULT_SECTIONS)} sections "
            f"= {len(targets)} total sections\n"
            f"  Max {max_iter} iterations each, target score {target_score}"
        )
    elif chapter is None:
        console.print("[red]Specify a chapter number or use --full for the entire book.[/red]")
        raise typer.Exit(1)
    else:
        with get_session() as session:
            ch = _get_chapter(session, book_id, chapter)
        if not ch:
            console.print(f"[red]Chapter not found:[/red] {chapter}")
            raise typer.Exit(1)
        ch_id, ch_num, ch_title, ch_desc, topic_query, topic_cluster = ch

        if section == "all":
            targets = [
                (ch_id, ch_num, ch_title, ch_desc, topic_query, topic_cluster, sec)
                for sec in _DEFAULT_SECTIONS
            ]
        else:
            targets = [(ch_id, ch_num, ch_title, ch_desc, topic_query, topic_cluster, section)]

        console.print(
            f"[bold]Autowrite:[/bold] {b_title} — "
            f"{len(targets)} section(s), max {max_iter} iter, target {target_score}"
        )

    # Run the convergence loop for each target
    total = len(targets)
    for i, (ch_id, ch_num, ch_title, ch_desc, topic_query, topic_cluster, sec) in enumerate(targets, 1):
        console.print(f"\n{'=' * 72}")
        console.print(f"[bold]Section {i}/{total}:[/bold] Ch.{ch_num} {ch_title} — {sec}")
        console.print(f"{'=' * 72}")

        _autowrite_section(
            book_id=book_id, book_title=b_title, book_plan=b_plan,
            ch_id=ch_id, ch_num=ch_num, ch_title=ch_title,
            topic_query=topic_query, topic_cluster=topic_cluster,
            section=sec, model=model, max_iter=max_iter,
            target_score=target_score, auto_expand=auto_expand,
            ipcc=ipcc, console=console,
        )

    console.print(f"\n[bold green]✓ Autowrite complete:[/bold green] "
                  f"{total} sections processed")


# ── export ─────────────────────────────────────────────────────────────────────

@app.command()
def export(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    output:     Path | None = typer.Option(None, "--output", "-o",
                                            help="Output file (default: <book_title>.md)."),
    include_sources: bool = typer.Option(True, "--sources/--no-sources",
                                          help="Include source citations per chapter."),
    fmt:        str = typer.Option("markdown", "--format", "-f",
                                    help="Export format: markdown, bibtex, latex, docx."),
):
    """
    Compile all chapter drafts into a single document.

    Formats:
      markdown  — Markdown with inline [N] citations + bibliography (default)
      bibtex    — .bib file from all cited papers' metadata
      latex     — Markdown → LaTeX via Pandoc (requires pandoc installed)
      docx      — Markdown → DOCX via Pandoc (requires pandoc installed)

    Examples:

      sciknow book export "Global Cooling"

      sciknow book export "Global Cooling" --format bibtex -o refs.bib

      sciknow book export "Global Cooling" --format latex -o book.tex
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

    safe = book[1].lower().replace(" ", "_").replace("/", "-")[:50]

    # ── BibTeX export ────────────────────────────────────────────────────
    if fmt == "bibtex":
        bib = _generate_bibtex(session if 'session' in dir() else None, book[0])
        bib_path = output or Path(f"{safe}.bib")
        bib_path.write_text(bib, encoding="utf-8")
        console.print(f"[green]✓ BibTeX exported:[/green] {bib_path}")
        return

    # ── Markdown (default) ───────────────────────────────────────────────
    md_path = output or Path(f"{safe}.md")
    md_path.write_text(md, encoding="utf-8")

    if fmt == "markdown":
        console.print(
            f"[green]✓ Exported [bold]{book[1]}[/bold][/green] → [bold]{md_path}[/bold]  "
            f"({total_words:,} words, {len(drafts)} sections)"
        )
        return

    # ── LaTeX / DOCX via Pandoc ──────────────────────────────────────────
    if fmt in ("latex", "docx"):
        import subprocess, shutil
        if not shutil.which("pandoc"):
            console.print(
                "[red]Pandoc not installed.[/red] Required for LaTeX/DOCX export.\n"
                "Install: [bold]sudo apt install pandoc[/bold]"
            )
            raise typer.Exit(1)

        bib = _generate_bibtex(None, book[0])
        bib_path = md_path.with_suffix(".bib")
        bib_path.write_text(bib, encoding="utf-8")

        ext = ".tex" if fmt == "latex" else ".docx"
        out_path = output or Path(f"{safe}{ext}")

        cmd = [
            "pandoc", str(md_path),
            "--citeproc", f"--bibliography={bib_path}",
            "-o", str(out_path),
        ]
        if fmt == "latex":
            cmd.extend(["--standalone"])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[red]Pandoc failed:[/red] {result.stderr[:300]}")
            raise typer.Exit(1)

        console.print(
            f"[green]✓ Exported [bold]{book[1]}[/bold][/green] → [bold]{out_path}[/bold]  "
            f"({total_words:,} words)"
        )
        console.print(f"[dim]BibTeX: {bib_path}[/dim]")
        return

    console.print(f"[red]Unknown format: {fmt}[/red]. Use: markdown, bibtex, latex, docx")


def _generate_bibtex(session, book_id: str) -> str:
    """Generate a BibTeX file from the metadata of all papers cited in the book's drafts."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as sess:
        # Get all DOIs from drafts' sources (which are stored as JSONB string arrays).
        # Also get all documents cited by any chunk that was used as a source.
        rows = sess.execute(text("""
            SELECT DISTINCT pm.doi, pm.title, pm.year, pm.authors, pm.journal,
                   pm.volume, pm.issue, pm.pages, pm.publisher
            FROM paper_metadata pm
            WHERE pm.doi IS NOT NULL
            ORDER BY pm.year DESC NULLS LAST, pm.title
        """)).fetchall()

    entries = []
    for doi, title, year, authors, journal, volume, issue, pages, publisher in rows:
        if not doi or not title:
            continue
        # Generate a citekey: AuthorYear format
        author_str = ""
        citekey_author = "Unknown"
        if authors and isinstance(authors, list) and len(authors) > 0:
            first = authors[0]
            name = first.get("name", "") if isinstance(first, dict) else str(first)
            parts = name.split()
            citekey_author = parts[-1] if parts else "Unknown"
            # Format all authors for BibTeX
            auth_list = []
            for a in authors[:10]:
                n = a.get("name", "") if isinstance(a, dict) else str(a)
                auth_list.append(n)
            author_str = " and ".join(auth_list)

        citekey = f"{citekey_author}{year or 'nd'}"

        entry = f"@article{{{citekey},\n"
        entry += f"  title = {{{title}}},\n"
        if author_str:
            entry += f"  author = {{{author_str}}},\n"
        if year:
            entry += f"  year = {{{year}}},\n"
        if journal:
            entry += f"  journal = {{{journal}}},\n"
        if volume:
            entry += f"  volume = {{{volume}}},\n"
        if issue:
            entry += f"  number = {{{issue}}},\n"
        if pages:
            entry += f"  pages = {{{pages}}},\n"
        if publisher:
            entry += f"  publisher = {{{publisher}}},\n"
        entry += f"  doi = {{{doi}}},\n"
        entry += "}\n"
        entries.append(entry)

    return "\n".join(entries)
