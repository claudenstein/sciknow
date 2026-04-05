"""
sciknow draft — manage saved writing drafts.

Commands:
  list    Paginated table of all drafts
  show    Print a draft's full content
  delete  Remove a draft
  export  Write a draft to a file
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

app = typer.Typer(help="Manage saved writing drafts.")
console = Console()


@app.command(name="list")
def list_drafts(
    book:  str | None = typer.Option(None, "--book",  "-b", help="Filter by book title (partial)."),
    limit: int        = typer.Option(50,   "--limit", "-l", help="Max rows to show."),
    page:  int        = typer.Option(1,    "--page",  "-p", help="Page number."),
):
    """
    List all saved drafts.

    Examples:

      sciknow draft list

      sciknow draft list --book "Global Cooling"
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    conditions = []
    params: dict = {}

    if book:
        conditions.append("b.title ILIKE :book")
        params["book"] = f"%{book}%"

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    offset = (page - 1) * limit

    with get_session() as session:
        total = session.execute(text(f"""
            SELECT COUNT(*) FROM drafts d
            LEFT JOIN books b ON b.id = d.book_id
            {where}
        """), params).scalar()

        rows = session.execute(text(f"""
            SELECT d.id::text, d.title, d.section_type, d.word_count,
                   b.title as book_title,
                   bc.number as chapter_number, bc.title as chapter_title,
                   d.created_at
            FROM drafts d
            LEFT JOIN books b ON b.id = d.book_id
            LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            {where}
            ORDER BY d.created_at DESC
            LIMIT :limit OFFSET :offset
        """), {**params, "limit": limit, "offset": offset}).fetchall()

    if not rows:
        console.print("[yellow]No drafts found.[/yellow]")
        raise typer.Exit(0)

    showing = f"{offset + 1}–{offset + len(rows)} of {total}"
    table = Table(title=f"Drafts  [{showing}]", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("ID",      style="dim",  width=8)
    table.add_column("Title",               ratio=3)
    table.add_column("Section", style="cyan", width=12)
    table.add_column("Words",  justify="right", width=7)
    table.add_column("Book / Chapter",        ratio=2)
    table.add_column("Created", style="dim",  width=12)

    for row in rows:
        id_short = str(row[0])[:8]
        book_ch = ""
        if row[4]:
            book_ch = row[4]
            if row[5] is not None:
                book_ch += f"  Ch.{row[5]}: {row[6] or ''}"
        created = str(row[7])[:10] if row[7] else ""
        table.add_row(
            id_short,
            row[1] or "",
            row[2] or "",
            str(row[3]) if row[3] else "—",
            book_ch,
            created,
        )
    console.print(table)

    if total > offset + limit:
        remaining = (total - offset - limit + limit - 1) // limit
        console.print(
            f"[dim]Page {page} — use [bold]--page {page + 1}[/bold] "
            f"({remaining} more page{'s' if remaining > 1 else ''})[/dim]"
        )


@app.command()
def show(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (full UUID or first 8 chars).")],
):
    """
    Print the full content of a saved draft.

    Examples:

      sciknow draft show 3f2a1b4c

      sciknow draft show 3f2a1b4c-full-uuid
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        row = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.topic, d.content,
                   d.word_count, d.model_used, d.created_at, d.sources,
                   b.title as book_title,
                   bc.number as chapter_number, bc.title as chapter_title
            FROM drafts d
            LEFT JOIN books b ON b.id = d.book_id
            LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.id::text LIKE :id_prefix
            ORDER BY d.created_at DESC
            LIMIT 1
        """), {"id_prefix": f"{draft_id}%"}).fetchone()

    if not row:
        console.print(f"[red]No draft found with ID starting with:[/red] {draft_id}")
        raise typer.Exit(1)

    console.print()
    console.print(Rule(f"[bold]{row[1]}[/bold]"))
    console.print()

    meta_parts = []
    if row[2]:
        meta_parts.append(f"Section: {row[2]}")
    if row[9]:
        ch_str = f"  Ch.{row[10]}: {row[11]}" if row[10] is not None else ""
        meta_parts.append(f"Book: {row[9]}{ch_str}")
    if row[6]:
        meta_parts.append(f"Model: {row[6]}")
    if row[5]:
        meta_parts.append(f"Words: {row[5]}")
    if row[7]:
        meta_parts.append(f"Saved: {str(row[7])[:16]}")

    if meta_parts:
        console.print(f"  [dim]{' · '.join(meta_parts)}[/dim]")
        console.print()

    console.print(row[4])
    console.print()

    sources = row[8] or []
    if sources:
        console.print(Rule("[dim]Sources[/dim]"))
        for s in sources:
            console.print(f"  [dim]{s}[/dim]")
    console.print()


@app.command()
def delete(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (full UUID or first 8 chars).")],
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Delete a saved draft."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        row = session.execute(text("""
            SELECT id::text, title FROM drafts
            WHERE id::text LIKE :id_prefix
            LIMIT 1
        """), {"id_prefix": f"{draft_id}%"}).fetchone()

    if not row:
        console.print(f"[red]No draft found:[/red] {draft_id}")
        raise typer.Exit(1)

    if not yes:
        typer.confirm(f"Delete draft: {row[1]!r}?", abort=True)

    with get_session() as session:
        session.execute(text("DELETE FROM drafts WHERE id = :id"), {"id": row[0]})
        session.commit()

    console.print(f"[green]✓ Deleted:[/green] {row[1]}")


@app.command()
def export(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (full UUID or first 8 chars).")],
    output: Path | None = typer.Option(None, "--output", "-o",
                                        help="Output file (default: <title>.md)."),
):
    """
    Export a draft to a Markdown file.

    Examples:

      sciknow draft export 3f2a1b4c

      sciknow draft export 3f2a1b4c --output chapter1_intro.md
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        row = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.topic, d.content,
                   d.sources, d.created_at, d.model_used,
                   b.title as book_title,
                   bc.number, bc.title as chapter_title
            FROM drafts d
            LEFT JOIN books b ON b.id = d.book_id
            LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.id::text LIKE :id_prefix
            ORDER BY d.created_at DESC
            LIMIT 1
        """), {"id_prefix": f"{draft_id}%"}).fetchone()

    if not row:
        console.print(f"[red]No draft found:[/red] {draft_id}")
        raise typer.Exit(1)

    title, section, topic, content, sources = row[1], row[2], row[3], row[4], row[5] or []

    # Build Markdown
    lines = [f"# {title}", ""]
    meta = []
    if row[8]:
        ch = f"Chapter {row[9]}: {row[10]}" if row[9] is not None else ""
        meta.append(f"**Book:** {row[8]}  {ch}")
    if section:
        meta.append(f"**Section:** {section}")
    if row[7]:
        meta.append(f"**Model:** {row[7]}")
    meta.append(f"**Saved:** {str(row[6])[:16]}")
    if meta:
        lines += ["*" + " · ".join(meta) + "*", ""]
    lines += ["---", "", content, ""]
    if sources:
        lines += ["## Sources", ""]
        for s in sources:
            lines.append(f"- {s}")
        lines.append("")

    md = "\n".join(lines)

    if output is None:
        safe = title.lower().replace(" ", "_").replace("/", "-")[:50]
        output = Path(f"{safe}.md")

    output.write_text(md, encoding="utf-8")
    console.print(f"[green]✓ Exported[/green] → [bold]{output}[/bold]  ({len(content.split())} words)")
