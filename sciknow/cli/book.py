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


def _consume_events(gen, console):
    """Consume events from a core.book_ops generator, printing to Rich console.

    Returns the final 'completed' event dict (or None if the operation failed).
    """
    from rich.rule import Rule

    completed = None
    for event in gen:
        t = event.get("type")
        if t == "token":
            console.print(event["text"], end="", highlight=False)
        elif t == "progress":
            detail = event.get("detail", event.get("stage", ""))
            console.print(f"[dim]{detail}[/dim]")
        elif t == "tree_plan":
            console.print()
            console.print(Rule("[bold]Paragraph Plan (tree)[/bold]"))
            data = event.get("data", {})
            for i, para in enumerate(data.get("paragraphs", []), 1):
                point = para.get("point", "")
                sources = ", ".join(para.get("sources", []))
                connects = para.get("connects_to", "")
                console.print(f"  [bold cyan]P{i}[/bold cyan] {point}")
                if sources:
                    console.print(f"      [dim]Sources: {sources}[/dim]")
                if connects:
                    console.print(f"      [dim]→ {connects}[/dim]")
                for child in para.get("children", []):
                    console.print(f"        [dim]- {child.get('point', '')}[/dim]")
            console.print()
        elif t == "plan":
            console.print()
            console.print(Rule("[bold]Sentence Plan[/bold]"))
            console.print(event["content"])
            console.print()
        elif t == "scores":
            scores = event["scores"]
            dims = ["groundedness", "completeness", "coherence", "citation_accuracy", "overall"]
            parts = []
            for d in dims:
                v = scores.get(d, 0)
                color = "green" if v >= 0.85 else "yellow" if v >= 0.7 else "red"
                parts.append(f"[{color}]{d[:5]}={v:.2f}[/{color}]")
            console.print(f"  Scores: {'  '.join(parts)}")
            weakest = scores.get("weakest_dimension", "")
            if weakest:
                console.print(f"  Weakest: [bold]{weakest}[/bold]")
            instruction = scores.get("revision_instruction", "")
            if instruction:
                console.print(f"  Instruction: [dim]{instruction}[/dim]")
        elif t == "iteration_start":
            console.print(Rule(f"[dim]Iteration {event['iteration']}/{event['max']}[/dim]"))
        elif t == "revision_verdict":
            icon = "\u2713" if event["action"] == "KEEP" else "\u2717"
            color = "green" if event["action"] == "KEEP" else "red"
            console.print(
                f"  [{color}]{icon} {event['action']}[/{color}]: "
                f"overall {event['old_score']:.2f} \u2192 {event['new_score']:.2f}"
            )
        elif t == "converged":
            console.print(
                f"\n[green]\u2713 Converged at iteration {event['iteration']} "
                f"(overall={event['final_score']:.2f})[/green]"
            )
        elif t == "verification":
            vdata = event["data"]
            score = vdata.get("groundedness_score", "?")
            console.print(f"  Groundedness score: [bold]{score}[/bold]")
            for claim in vdata.get("claims", []):
                v = claim.get("verdict", "?")
                color = "green" if v == "SUPPORTED" else "yellow" if v == "EXTRAPOLATED" else "red"
                console.print(f"  [{color}]{v}[/{color}]: {claim.get('text', '')[:80]}")
        elif t == "completed":
            completed = event
            wc = event.get("word_count", 0)
            did = event.get("draft_id", "")
            if wc:
                console.print(f"\n[green]\u2713 Done[/green]  ({wc} words)")
            if did:
                console.print(f"[dim]Draft ID: {did[:8]}[/dim]")
        elif t == "error":
            console.print(f"[red]Error:[/red] {event.get('message', 'unknown')}")
    return completed


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
    except Exception as exc:
        import logging
        logging.getLogger("sciknow.book").warning(
            "Auto-summarize failed for %s/%s: %s", chapter_title, section_type, exc)
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
):
    """
    Draft a chapter section with cross-chapter coherence.

    Injects the book plan and summaries of prior chapters into the prompt
    so the LLM maintains consistency across the book. Auto-generates a
    summary after saving for use by subsequent chapters.

    Examples:

      sciknow book write "Global Cooling" 1 --section introduction

      sciknow book write "Global Cooling" 2 --section methods --plan --verify

      sciknow book write "Global Cooling" 3 --section results --expand
    """
    from sciknow.core.book_ops import write_section_stream
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)
        book_id = book[0]

        ch = _get_chapter(session, book_id, chapter)
        if not ch:
            console.print(f"[red]Chapter not found:[/red] {chapter}")
            raise typer.Exit(1)
        ch_id = ch[0]

    console.print()
    gen = write_section_stream(
        book_id=book_id, chapter_id=ch_id, section_type=section,
        context_k=context_k, candidate_k=candidates,
        year_from=year_from, year_to=year_to,
        model=model, expand=expand, show_plan=show_plan,
        verify=verify, save=not no_save,
    )
    _consume_events(gen, console)


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
    from sciknow.core.book_ops import review_draft_stream

    console.print()
    gen = review_draft_stream(draft_id, model=model, save=save)
    _consume_events(gen, console)
    if save:
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
    from sciknow.core.book_ops import revise_draft_stream

    console.print()
    gen = revise_draft_stream(
        draft_id, instruction=instruction, context_k=context_k, model=model,
    )
    _consume_events(gen, console)


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
    from sciknow.core.book_ops import run_argue_stream
    from sciknow.storage.db import get_session

    bid = None
    if book_title:
        with get_session() as session:
            book = _get_book(session, book_title)
            if book:
                bid = book[0]

    console.print()
    gen = run_argue_stream(
        claim, book_id=bid, context_k=context_k, candidate_k=candidates,
        year_from=year_from, year_to=year_to, model=model, save=save,
    )
    _consume_events(gen, console)


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
    from sciknow.core.book_ops import run_gaps_stream
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)
        book_id = book[0]

    console.print()
    gen = run_gaps_stream(book_id=book_id, model=model, save=save)
    _consume_events(gen, console)


# ── serve (web reader) ─────────────────────────────────────────────────────────

@app.command()
def serve(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    port: int = typer.Option(8765, "--port", "-p", help="HTTP port."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
):
    """
    Launch a local web reader for the book.

    Opens a browser-based reading experience with sidebar navigation,
    inline editing, comments, citation links, quality scores, and
    version history. Content is served live from the database.

    Examples:

      sciknow book serve "Global Cooling"

      sciknow book serve "Global Cooling" --port 9000
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

    from sciknow.web.app import app as web_app, set_book
    set_book(book[0], book[1])

    console.print(f"\n[bold]SciKnow Book Reader[/bold]: {book[1]}")
    console.print(f"  [link=http://{host}:{port}]http://{host}:{port}[/link]")
    console.print(f"  [dim]Press Ctrl+C to stop.[/dim]\n")

    import uvicorn
    uvicorn.run(web_app, host=host, port=port, log_level="warning")


# ── autowrite (Karpathy-loop-inspired convergence) ─────────────────────────────

_DEFAULT_SECTIONS = ["introduction", "methods", "results", "discussion", "conclusion"]


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
    full:       bool = typer.Option(False, "--full",
                                     help="Write ALL chapters × ALL sections (the full autonomous pipeline)."),
    rebuild:    bool = typer.Option(False, "--rebuild",
                                     help="Rewrite sections that already have drafts (default: skip existing)."),
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

      sciknow book autowrite "Global Cooling" 3 --section all --target-score 0.80

      sciknow book autowrite "Global Cooling" --full --max-iter 3 --auto-expand
    """
    from sqlalchemy import text
    from sciknow.core.book_ops import autowrite_section_stream
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

    # Determine which chapters x sections to write
    if full:
        targets = [
            (ch[0], ch[1], ch[2], sec)
            for ch in chapters
            for sec in _DEFAULT_SECTIONS
        ]
        console.print(
            f"[bold]Autowrite FULL BOOK:[/bold] {b_title}\n"
            f"  {len(chapters)} chapters x {len(_DEFAULT_SECTIONS)} sections "
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
        ch_id, ch_num, ch_title = ch[0], ch[1], ch[2]

        if section == "all":
            targets = [(ch_id, ch_num, ch_title, sec) for sec in _DEFAULT_SECTIONS]
        else:
            targets = [(ch_id, ch_num, ch_title, section)]

        console.print(
            f"[bold]Autowrite:[/bold] {b_title} — "
            f"{len(targets)} section(s), max {max_iter} iter, target {target_score}"
        )

    # Skip sections that already have drafts (unless --rebuild)
    if not rebuild:
        with get_session() as session:
            existing = session.execute(text("""
                SELECT chapter_id::text, section_type
                FROM drafts WHERE book_id = :bid AND chapter_id IS NOT NULL
            """), {"bid": book_id}).fetchall()
        existing_set = {(r[0], r[1]) for r in existing}

        before = len(targets)
        targets = [(cid, cn, ct, sec) for cid, cn, ct, sec in targets
                    if (cid, sec) not in existing_set]
        skipped = before - len(targets)
        if skipped:
            console.print(f"[dim]Skipping {skipped} sections with existing drafts (use --rebuild to overwrite)[/dim]")
        if not targets:
            console.print("[green]All sections already have drafts.[/green]")
            raise typer.Exit(0)

    # Run the convergence loop for each target with live dashboard
    import time as _time
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    total = len(targets)
    converged = 0

    for i, (ch_id, ch_num, ch_title, sec) in enumerate(targets, 1):
        console.print(f"\n{'=' * 72}")
        console.print(
            f"[bold]Section {i}/{total}:[/bold] Ch.{ch_num} {ch_title} — {sec}"
            f"  [dim]({converged} converged so far)[/dim]"
        )
        console.print(f"{'=' * 72}")

        gen = autowrite_section_stream(
            book_id=book_id, chapter_id=ch_id, section_type=sec,
            model=model, max_iter=max_iter, target_score=target_score,
            auto_expand=auto_expand,
        )

        # Live dashboard state
        tok_count = 0
        tok_t0 = _time.monotonic()
        text_preview = ""
        current_stage = "starting"
        current_iter = 0
        iter_max = max_iter
        last_scores = {}
        score_history = []
        verdicts = []
        grounding = ""
        result = None

        def _build_display():
            tbl = Table.grid(padding=(0, 2))
            tbl.add_column(ratio=1)

            # Stage + tokens
            elapsed = _time.monotonic() - tok_t0
            tps = tok_count / elapsed if elapsed > 0 else 0
            stage_color = {"writing": "green", "scoring": "yellow", "verifying": "cyan",
                           "revising": "magenta", "saving": "blue"}.get(current_stage, "dim")
            tbl.add_row(Text.from_markup(
                f"  [{stage_color}]{current_stage.upper()}[/{stage_color}]  "
                f"[green]{tok_count} tok[/green]  "
                f"[cyan]{tps:.1f} tok/s[/cyan]  "
                f"[dim]iter {current_iter}/{iter_max}[/dim]"
            ))

            # Scores bar
            if last_scores:
                dims = ["groundedness", "completeness", "coherence", "citation_accuracy", "overall"]
                parts = []
                for d in dims:
                    v = last_scores.get(d, 0)
                    c = "green" if v >= 0.85 else "yellow" if v >= 0.7 else "red"
                    bar_len = int(v * 10)
                    bar = "█" * bar_len + "░" * (10 - bar_len)
                    parts.append(f"[{c}]{d[:5]} {bar} {v:.2f}[/{c}]")
                tbl.add_row(Text.from_markup("  " + "  ".join(parts)))

            # Verdicts
            if verdicts:
                v_text = "  ".join(verdicts[-5:])
                tbl.add_row(Text.from_markup(f"  {v_text}"))

            # Grounding
            if grounding:
                tbl.add_row(Text.from_markup(f"  {grounding}"))

            # Text preview (last 200 chars)
            if text_preview:
                preview = text_preview[-200:].replace("\n", " ").strip()
                if len(text_preview) > 200:
                    preview = "..." + preview
                tbl.add_row(Text.from_markup(f"  [dim]{preview}[/dim]"))

            return Panel(tbl, title=f"Ch.{ch_num} {sec.capitalize()}", border_style="blue")

        with Live(_build_display(), console=console, refresh_per_second=4, transient=True) as live:
            for event in gen:
                t = event.get("type")

                if t == "token":
                    tok_count += 1
                    text_preview += event["text"]
                    live.update(_build_display())

                elif t == "progress":
                    current_stage = event.get("stage", current_stage)
                    live.update(_build_display())

                elif t == "scores":
                    last_scores = event.get("scores", {})
                    score_history.append(last_scores.get("overall", 0))
                    current_iter = event.get("iteration", current_iter)
                    live.update(_build_display())

                elif t == "iteration_start":
                    current_iter = event.get("iteration", 1)
                    iter_max = event.get("max", max_iter)
                    text_preview = ""
                    tok_count = 0
                    tok_t0 = _time.monotonic()
                    live.update(_build_display())

                elif t == "revision_verdict":
                    action = event["action"]
                    icon = "\u2713" if action == "KEEP" else "\u2717"
                    color = "green" if action == "KEEP" else "red"
                    verdicts.append(
                        f"[{color}]{icon} {action} "
                        f"{event['old_score']:.2f}\u2192{event['new_score']:.2f}[/{color}]"
                    )
                    text_preview = ""
                    tok_count = 0
                    tok_t0 = _time.monotonic()
                    live.update(_build_display())

                elif t == "converged":
                    verdicts.append(
                        f"[bold green]\u2713 CONVERGED "
                        f"(iter {event['iteration']}, score {event['final_score']:.2f})[/bold green]"
                    )
                    live.update(_build_display())

                elif t == "verification":
                    vdata = event.get("data", {})
                    gs = vdata.get("groundedness_score", "?")
                    color = "green" if isinstance(gs, (int, float)) and gs >= 0.8 else "yellow"
                    grounding = f"[{color}]Groundedness: {gs}[/{color}]"
                    live.update(_build_display())

                elif t == "completed":
                    result = event

                elif t == "error":
                    console.print(f"[red]Error:[/red] {event.get('message', '')}")

        # Print final summary for this section
        if result:
            wc = result.get("word_count", 0)
            fs = result.get("final_score", 0)
            iters = result.get("iterations", 0)
            console.print(
                f"  [green]\u2713[/green] {sec.capitalize()}: "
                f"{wc} words, {iters} iterations"
                + (f", score {fs:.2f}" if fs else "")
            )
            if fs >= target_score:
                converged += 1

    console.print(f"\n[bold green]\u2713 Autowrite complete:[/bold green] "
                  f"{total} sections, {converged} converged")


# ── export ─────────────────────────────────────────────────────────────────────

@app.command()
def export(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    output:     Path | None = typer.Option(None, "--output", "-o",
                                            help="Output file (default: <book_title>.md)."),
    include_sources: bool = typer.Option(True, "--sources/--no-sources",
                                          help="Include source citations per chapter."),
    fmt:        str = typer.Option("markdown", "--format", "-f",
                                    help="Export format: markdown, html, bibtex, latex, docx."),
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
    import re as _re
    from collections import defaultdict
    chapter_drafts: dict[str, list] = defaultdict(list)
    for d in drafts:
        chapter_drafts[d[0] or "__none__"].append(d)

    # Sort drafts within each chapter by section order
    for ch_id in chapter_drafts:
        chapter_drafts[ch_id].sort(
            key=lambda d: _SECTION_ORDER.get(d[2] or "", 99)
        )

    # ── Fix 2: Global citation dedup + renumbering ───────────────────────
    # First pass: collect all unique sources across all chapters, building
    # a global bibliography with unified [1]-[N] numbering. Then remap
    # per-draft [N] references to the global numbers in the content.

    # Build per-draft source lists and collect global bibliography
    global_sources: list[str] = []      # deduplicated, ordered
    global_source_set: set[str] = set()
    draft_source_lists: dict[str, list[str]] = {}  # draft_key -> [source_str, ...]

    for ch in chapters:
        ch_id = ch[0]
        for draft in chapter_drafts.get(ch_id, []):
            _, d_title, d_section, d_content, d_words, d_sources, _ = draft
            draft_key = f"{ch_id}:{d_section}"
            local_sources = []
            for s in (d_sources or []):
                if s:
                    local_sources.append(s)
                    # Strip the "[N] " prefix for dedup (same paper may be [1] in one draft, [3] in another)
                    clean = _re.sub(r'^\[\d+\]\s*', '', s).strip()
                    if clean not in global_source_set:
                        global_source_set.add(clean)
                        global_sources.append(s)
            draft_source_lists[draft_key] = local_sources

    # Build remap: for each draft, map old [N] → new global [M]
    def _source_key(s: str) -> str:
        return _re.sub(r'^\[\d+\]\s*', '', s).strip()

    global_num_by_key = {}
    for i, s in enumerate(global_sources, 1):
        global_num_by_key[_source_key(s)] = i

    def _remap_citations(content: str, local_sources: list[str]) -> str:
        """Remap [N] citations from per-draft numbering to global numbering."""
        local_to_global = {}
        for ls in local_sources:
            m = _re.match(r'^\[(\d+)\]', ls)
            if m:
                old_num = m.group(1)
                new_num = global_num_by_key.get(_source_key(ls))
                if new_num and old_num != str(new_num):
                    local_to_global[old_num] = str(new_num)

        if not local_to_global:
            return content

        # Replace [old] with [new], using a placeholder to avoid double-replace
        for old, new in local_to_global.items():
            content = _re.sub(
                rf'\[{old}\]',
                f'[__CITE_{new}__]',
                content,
            )
        # Remove placeholders
        content = _re.sub(r'\[__CITE_(\d+)__\]', r'[\1]', content)
        return content

    # Second pass: build the markdown with remapped citations
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

            # Remap citations to global numbering
            draft_key = f"{ch_id}:{d_section}"
            remapped = _remap_citations(d_content or "", draft_source_lists.get(draft_key, []))
            lines += [remapped, ""]

    # Global bibliography with unified numbering
    if include_sources and global_sources:
        lines += ["---", "", "## Bibliography", ""]
        for i, s in enumerate(global_sources, 1):
            # Re-number the source line itself
            renumbered = _re.sub(r'^\[\d+\]', f'[{i}]', s)
            lines.append(f"{i}. {renumbered}")
        lines.append("")

    # Word count
    total_words = sum(d[4] or 0 for d in drafts)
    lines += ["", f"---", f"*{total_words:,} words · {len(drafts)} sections · {len(chapters)} chapters · {len(global_sources)} references*"]

    md = "\n".join(lines)

    safe = book[1].lower().replace(" ", "_").replace("/", "-")[:50]

    # ── HTML export (self-contained static reader) ─────────────────────
    if fmt == "html":
        from sciknow.web.app import _get_book_data, _render_book, set_book
        set_book(book[0], book[1])
        bk, chs, drs, gps, comms = _get_book_data()
        html = _render_book(bk, chs, drs, gps, comms)
        html_path = output or Path(f"{safe}.html")
        html_path.write_text(html, encoding="utf-8")
        console.print(
            f"[green]✓ HTML exported:[/green] [bold]{html_path}[/bold]  "
            f"({total_words:,} words) — open in any browser"
        )
        return

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
