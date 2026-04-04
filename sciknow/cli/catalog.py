"""
sciknow catalog — browse and export the paper index.

Commands:
  list    Paginated table of all papers, with filters
  show    Full record for one paper (by DOI, arXiv ID, or title fragment)
  export  Dump the catalog to CSV or JSON
  stats   Breakdown by year, journal, domain
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich import box

app = typer.Typer(help="Browse and export the paper catalog.")
console = Console()


# ── helpers ────────────────────────────────────────────────────────────────────

def _author_short(authors: list) -> str:
    if not authors:
        return ""
    first = authors[0].get("name", "")
    if not first:
        return ""
    # "Last, F." from "First Last"
    parts = first.split()
    if len(parts) >= 2:
        last = parts[-1]
        initials = "".join(p[0] + "." for p in parts[:-1] if p)
        first_fmt = f"{last}, {initials}"
    else:
        first_fmt = first
    return first_fmt + (" et al." if len(authors) > 1 else "")


def _identifier(doi: str | None, arxiv_id: str | None) -> str:
    if doi:
        return f"doi:{doi}"
    if arxiv_id:
        return f"arXiv:{arxiv_id}"
    return ""


# ── list ───────────────────────────────────────────────────────────────────────

@app.command(name="list")
def list_papers(
    author:  str | None = typer.Option(None, "--author",  "-a", help="Filter by author name (partial match)."),
    year:    int | None = typer.Option(None, "--year",    "-y", help="Filter by exact year."),
    from_year: int | None = typer.Option(None, "--from",        help="Filter: year >= value."),
    to_year:   int | None = typer.Option(None, "--to",          help="Filter: year <= value."),
    journal: str | None = typer.Option(None, "--journal", "-j", help="Filter by journal (partial match)."),
    title:   str | None = typer.Option(None, "--title",   "-t", help="Filter by title (partial match)."),
    doi:     str | None = typer.Option(None, "--doi",           help="Filter by DOI (partial match)."),
    sort:    str        = typer.Option("year", "--sort",  "-s", help="Sort by: year, title, author, journal."),
    limit:   int        = typer.Option(50,    "--limit",  "-l", help="Max rows to show (0 = all)."),
    page:    int        = typer.Option(1,     "--page",   "-p", help="Page number (used with --limit)."),
):
    """
    List papers in the catalog with optional filters.

    Examples:

      sciknow catalog list

      sciknow catalog list --author Zharkova --sort year

      sciknow catalog list --from 2015 --to 2023 --journal "Nature"

      sciknow catalog list --title "solar cycle" --limit 20
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    conditions = []
    params: dict = {}

    if author:
        conditions.append("pm.authors::text ILIKE :author")
        params["author"] = f"%{author}%"
    if year:
        conditions.append("pm.year = :year")
        params["year"] = year
    if from_year:
        conditions.append("pm.year >= :from_year")
        params["from_year"] = from_year
    if to_year:
        conditions.append("pm.year <= :to_year")
        params["to_year"] = to_year
    if journal:
        conditions.append("pm.journal ILIKE :journal")
        params["journal"] = f"%{journal}%"
    if title:
        conditions.append("pm.title ILIKE :title")
        params["title"] = f"%{title}%"
    if doi:
        conditions.append("pm.doi ILIKE :doi")
        params["doi"] = f"%{doi}%"

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sort_col = {
        "year":    "pm.year DESC NULLS LAST, pm.title",
        "title":   "pm.title",
        "author":  "pm.authors->0->>'name'",
        "journal": "pm.journal NULLS LAST, pm.year DESC",
    }.get(sort, "pm.year DESC NULLS LAST")

    offset = (page - 1) * limit if limit else 0
    limit_clause = f"LIMIT {limit} OFFSET {offset}" if limit else ""

    sql = text(f"""
        SELECT pm.title, pm.authors, pm.year, pm.journal,
               pm.doi, pm.arxiv_id, pm.metadata_source,
               d.filename
        FROM paper_metadata pm
        JOIN documents d ON d.id = pm.document_id
        {where}
        ORDER BY {sort_col}
        {limit_clause}
    """)

    count_sql = text(f"""
        SELECT COUNT(*)
        FROM paper_metadata pm
        JOIN documents d ON d.id = pm.document_id
        {where}
    """)

    with get_session() as session:
        total = session.execute(count_sql, params).scalar()
        rows = session.execute(sql, params).fetchall()

    if not rows:
        console.print("[yellow]No papers found.[/yellow]")
        raise typer.Exit(0)

    showing = f"{offset + 1}–{offset + len(rows)} of {total}"
    table = Table(
        title=f"Paper Catalog  [{showing}]",
        box=box.SIMPLE_HEAD,
        show_lines=False,
        expand=True,
    )
    table.add_column("#",        style="dim",   no_wrap=True, width=4)
    table.add_column("Year",     style="cyan",  no_wrap=True, width=6)
    table.add_column("Author(s)",               no_wrap=True, width=22)
    table.add_column("Title",                   ratio=3)
    table.add_column("Journal",                 ratio=2)
    table.add_column("Identifier", style="dim", ratio=1)

    for i, (title_val, authors, yr, journal_val, doi_val, arxiv_val, src, fname) in enumerate(rows, start=offset + 1):
        table.add_row(
            str(i),
            str(yr) if yr else "—",
            _author_short(authors or []),
            title_val or f"[dim]{fname}[/dim]",
            journal_val or "",
            _identifier(doi_val, arxiv_val),
        )

    console.print(table)

    if limit and total > offset + limit:
        remaining_pages = (total - offset - limit + limit - 1) // limit
        console.print(
            f"[dim]Page {page} — use [bold]--page {page + 1}[/bold] for next "
            f"({remaining_pages} more page{'s' if remaining_pages > 1 else ''})[/dim]"
        )


# ── show ───────────────────────────────────────────────────────────────────────

@app.command()
def show(
    query: Annotated[str, typer.Argument(help="DOI, arXiv ID, or title fragment.")],
):
    """
    Show full metadata for a single paper.

    Examples:

      sciknow catalog show 10.1093/mnras/stad1001

      sciknow catalog show "solar magnetic field eigenvectors"
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        # Try exact DOI first
        row = session.execute(text("""
            SELECT pm.*, d.filename, d.file_size_bytes, d.original_path,
                   d.ingestion_status
            FROM paper_metadata pm
            JOIN documents d ON d.id = pm.document_id
            WHERE pm.doi = :q OR pm.arxiv_id = :q
            LIMIT 1
        """), {"q": query}).fetchone()

        if not row:
            # Title fragment search
            row = session.execute(text("""
                SELECT pm.*, d.filename, d.file_size_bytes, d.original_path,
                       d.ingestion_status
                FROM paper_metadata pm
                JOIN documents d ON d.id = pm.document_id
                WHERE pm.title ILIKE :q
                ORDER BY pm.year DESC
                LIMIT 1
            """), {"q": f"%{query}%"}).fetchone()

    if not row:
        console.print(f"[yellow]No paper found matching:[/yellow] {query}")
        raise typer.Exit(1)

    mapping = row._mapping

    def _field(label: str, value, style: str = "") -> None:
        if value is None or value == [] or value == "":
            return
        val_str = str(value)
        if style:
            console.print(f"  [bold]{label}:[/bold] [{style}]{val_str}[/{style}]")
        else:
            console.print(f"  [bold]{label}:[/bold] {val_str}")

    console.print()
    console.print(f"[bold cyan]{mapping['title'] or mapping['filename']}[/bold cyan]")
    console.print()

    authors = mapping.get("authors") or []
    if authors:
        names = "; ".join(
            a.get("name", "") + (f" [{a['orcid']}]" if a.get("orcid") else "")
            for a in authors
        )
        console.print(f"  [bold]Authors:[/bold] {names}")

    _field("Year",      mapping.get("year"))
    _field("Journal",   mapping.get("journal"), "italic")
    _field("Volume",    mapping.get("volume"))
    _field("Issue",     mapping.get("issue"))
    _field("Pages",     mapping.get("pages"))
    _field("Publisher", mapping.get("publisher"))
    _field("DOI",       f"https://doi.org/{mapping['doi']}" if mapping.get("doi") else None, "blue")
    _field("arXiv",     mapping.get("arxiv_id"))
    _field("Keywords",  ", ".join(mapping.get("keywords") or []))
    _field("Domains",   ", ".join(mapping.get("domains") or []))
    _field("Metadata source", mapping.get("metadata_source"), "dim")

    abstract = mapping.get("abstract")
    if abstract:
        console.print()
        console.print("  [bold]Abstract:[/bold]")
        # Word-wrap to 100 chars
        import textwrap
        for line in textwrap.wrap(abstract, width=96):
            console.print(f"    {line}")

    console.print()
    console.print(f"  [dim]File: {mapping.get('filename')}  "
                  f"({(mapping.get('file_size_bytes') or 0) // 1024} KB)[/dim]")


# ── export ─────────────────────────────────────────────────────────────────────

@app.command()
def export(
    output:  Path = typer.Option(Path("catalog.csv"), "--output", "-o", help="Output file path."),
    fmt:     str  = typer.Option("csv", "--format",  "-f", help="Format: csv or json."),
    author:  str | None = typer.Option(None, "--author"),
    year:    int | None = typer.Option(None, "--year"),
    journal: str | None = typer.Option(None, "--journal"),
):
    """
    Export the paper catalog to CSV or JSON.

    Examples:

      sciknow catalog export --output catalog.csv

      sciknow catalog export --format json --output catalog.json

      sciknow catalog export --author Zharkova --output zharkova.csv
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    conditions = []
    params: dict = {}
    if author:
        conditions.append("pm.authors::text ILIKE :author")
        params["author"] = f"%{author}%"
    if year:
        conditions.append("pm.year = :year")
        params["year"] = year
    if journal:
        conditions.append("pm.journal ILIKE :journal")
        params["journal"] = f"%{journal}%"
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with get_session() as session:
        rows = session.execute(text(f"""
            SELECT pm.title, pm.authors, pm.year, pm.journal,
                   pm.volume, pm.issue, pm.pages, pm.publisher,
                   pm.doi, pm.arxiv_id, pm.keywords, pm.domains,
                   pm.abstract, pm.metadata_source, d.filename
            FROM paper_metadata pm
            JOIN documents d ON d.id = pm.document_id
            {where}
            ORDER BY pm.year DESC NULLS LAST, pm.title
        """), params).fetchall()

    if not rows:
        console.print("[yellow]No papers found.[/yellow]")
        raise typer.Exit(0)

    records = []
    for r in rows:
        authors_list = r[1] or []
        records.append({
            "title":       r[0],
            "authors":     "; ".join(a.get("name", "") for a in authors_list),
            "year":        r[2],
            "journal":     r[3],
            "volume":      r[4],
            "issue":       r[5],
            "pages":       r[6],
            "publisher":   r[7],
            "doi":         r[8],
            "arxiv_id":    r[9],
            "keywords":    "; ".join(r[10] or []),
            "domains":     "; ".join(r[11] or []),
            "abstract":    (r[12] or "")[:500],
            "metadata_source": r[13],
            "filename":    r[14],
        })

    output.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    console.print(f"[green]✓ Exported {len(records)} papers[/green] → [bold]{output}[/bold]")


# ── stats ──────────────────────────────────────────────────────────────────────

@app.command()
def stats():
    """Show catalog statistics: papers by year, top journals, domains."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        by_year = session.execute(text("""
            SELECT year, COUNT(*) FROM paper_metadata
            WHERE year IS NOT NULL
            GROUP BY year ORDER BY year DESC LIMIT 20
        """)).fetchall()

        by_journal = session.execute(text("""
            SELECT journal, COUNT(*) as n FROM paper_metadata
            WHERE journal IS NOT NULL AND journal != ''
            GROUP BY journal ORDER BY n DESC LIMIT 15
        """)).fetchall()

        by_source = session.execute(text("""
            SELECT metadata_source, COUNT(*) FROM paper_metadata
            GROUP BY metadata_source ORDER BY COUNT(*) DESC
        """)).fetchall()

        total = session.execute(text("SELECT COUNT(*) FROM paper_metadata")).scalar()
        with_doi = session.execute(text("SELECT COUNT(*) FROM paper_metadata WHERE doi IS NOT NULL")).scalar()
        with_abstract = session.execute(text("SELECT COUNT(*) FROM paper_metadata WHERE abstract IS NOT NULL")).scalar()
        with_authors = session.execute(text("SELECT COUNT(*) FROM paper_metadata WHERE authors != '[]'")).scalar()

    # Summary
    table = Table(title="Catalog Overview", box=box.SIMPLE_HEAD, show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value",  justify="right", style="cyan")
    table.add_row("Total papers",       str(total))
    table.add_row("With DOI",           f"{with_doi} ({100*with_doi//total}%)")
    table.add_row("With abstract",      f"{with_abstract} ({100*with_abstract//total}%)")
    table.add_row("With authors",       f"{with_authors} ({100*with_authors//total}%)")
    console.print(table)
    console.print()

    # By year (sparkline-style)
    t2 = Table(title="Papers by Year (most recent)", box=box.SIMPLE_HEAD)
    t2.add_column("Year", style="cyan", no_wrap=True)
    t2.add_column("Count", justify="right")
    t2.add_column("",      ratio=1)
    max_n = max(n for _, n in by_year) if by_year else 1
    for yr, n in by_year:
        bar = "█" * int(30 * n / max_n)
        t2.add_row(str(yr), str(n), f"[blue]{bar}[/blue]")
    console.print(t2)
    console.print()

    # By journal
    t3 = Table(title="Top Journals", box=box.SIMPLE_HEAD)
    t3.add_column("Journal", ratio=3)
    t3.add_column("Papers", justify="right", style="cyan")
    for j, n in by_journal:
        t3.add_row(j[:70], str(n))
    console.print(t3)
    console.print()

    # By metadata source
    t4 = Table(title="Metadata Source Quality", box=box.SIMPLE_HEAD)
    t4.add_column("Source",  style="bold")
    t4.add_column("Papers",  justify="right", style="cyan")
    t4.add_column("Quality", style="dim")
    quality = {
        "crossref":      "[green]High — authoritative API[/green]",
        "arxiv":         "[green]High — authoritative API[/green]",
        "llm_extracted": "[yellow]Medium — LLM fallback[/yellow]",
        "embedded_pdf":  "[yellow]Medium — PDF metadata[/yellow]",
        "unknown":       "[red]Low — extraction failed[/red]",
    }
    for src, n in by_source:
        t4.add_row(src, str(n), quality.get(src, ""))
    console.print(t4)
