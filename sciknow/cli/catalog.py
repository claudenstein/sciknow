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
               d.filename,
               COALESCE(cc.cnt, 0) AS cite_count
        FROM paper_metadata pm
        JOIN documents d ON d.id = pm.document_id
        LEFT JOIN (
            SELECT cited_document_id, COUNT(*) AS cnt
            FROM citations
            WHERE cited_document_id IS NOT NULL
            GROUP BY cited_document_id
        ) cc ON cc.cited_document_id = d.id
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
    table.add_column("Cited", style="magenta", justify="right", width=5)

    for i, row in enumerate(rows, start=offset + 1):
        title_val, authors, yr, journal_val, doi_val, arxiv_val, src, fname, cite_count = row
        table.add_row(
            str(i),
            str(yr) if yr else "—",
            _author_short(authors or []),
            title_val or f"[dim]{fname}[/dim]",
            journal_val or "",
            _identifier(doi_val, arxiv_val),
            str(cite_count) if cite_count else "",
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


# ── topics ─────────────────────────────────────────────────────────────────────

@app.command()
def topics():
    """
    List all topic clusters with paper counts.

    Examples:

      sciknow catalog topics
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        rows = session.execute(text("""
            SELECT topic_cluster, COUNT(*) as n
            FROM paper_metadata
            WHERE topic_cluster IS NOT NULL AND topic_cluster != ''
            GROUP BY topic_cluster
            ORDER BY n DESC
        """)).fetchall()

        total_clustered = session.execute(text("""
            SELECT COUNT(*) FROM paper_metadata
            WHERE topic_cluster IS NOT NULL AND topic_cluster != ''
        """)).scalar()

        total = session.execute(text("SELECT COUNT(*) FROM paper_metadata")).scalar()

    if not rows:
        console.print("[yellow]No topic clusters found. Run [bold]sciknow catalog cluster[/bold] first.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Topic Clusters  ({total_clustered}/{total} papers assigned)", box=box.SIMPLE_HEAD)
    table.add_column("Cluster",  ratio=3)
    table.add_column("Papers", justify="right", style="cyan", width=8)
    table.add_column("",        ratio=2)

    max_n = max(n for _, n in rows) if rows else 1
    for cluster_name, n in rows:
        bar = "█" * int(25 * n / max_n)
        table.add_row(cluster_name, str(n), f"[blue]{bar}[/blue]")

    console.print(table)


# ── cluster ────────────────────────────────────────────────────────────────────

@app.command()
def cluster(
    limit: int = typer.Option(0, "--limit", "-l",
                               help="Max papers to cluster (0 = all). Useful for large collections."),
    batch: int = typer.Option(50, "--batch",
                               help="Papers per LLM batch (default 50 — reliable for most models)."),
    model: str | None = typer.Option(None, "--model", help="Override LLM model name (Ollama)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print proposed clusters without saving."),
    resume: bool = typer.Option(False, "--resume", help="Only cluster papers that don't have a topic_cluster yet."),
):
    """
    Assign topic clusters to all papers using an LLM.

    The LLM proposes 6–14 named topic clusters and assigns every paper to one.
    Clusters are stored in paper_metadata.topic_cluster and can be used as a
    filter in search and ask commands (--topic).

    Incremental: each successful batch is saved immediately to the database,
    so --resume picks up where you left off if the process is interrupted.

    Examples:

      sciknow catalog cluster

      sciknow catalog cluster --batch 25 --dry-run

      sciknow catalog cluster --resume   # pick up after a partial run
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    import json as _json
    import logging as _logging
    import re as _re
    import time as _time
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream
    from sciknow.config import settings as _settings

    _log = _logging.getLogger("sciknow.cluster")

    # ── Load papers ──────────────────────────────────────────────────────
    with get_session() as session:
        where_extra = "AND pm.topic_cluster IS NULL" if resume else ""
        rows = session.execute(text(f"""
            SELECT pm.document_id::text, pm.title, pm.year
            FROM paper_metadata pm
            WHERE pm.title IS NOT NULL {where_extra}
            ORDER BY pm.year DESC NULLS LAST, pm.title
        """)).fetchall()

    papers = [{"doc_id": r[0], "title": r[1], "year": r[2]} for r in rows]
    if limit:
        papers = papers[:limit]

    if not papers:
        if resume:
            console.print("[green]All papers already have topic clusters.[/green]")
        else:
            console.print("[yellow]No papers with titles found.[/yellow]")
        raise typer.Exit(0)

    workers = max(1, _settings.llm_parallel_workers)
    total_batches = (len(papers) + batch - 1) // batch
    console.print(
        f"Clustering [bold]{len(papers)}[/bold] papers in {total_batches} "
        f"batches of {batch} ({workers} parallel LLM calls)…"
    )

    # ── JSON repair ──────────────────────────────────────────────────────

    def _sanitize_json(raw: str) -> str:
        """Aggressively repair LLM JSON output."""
        cleaned = raw.strip()

        # Strip markdown code fences
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])

        # Extract outermost { }
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1]

        # Remove control characters
        cleaned = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned)

        # Strip ALL backslash sequences that aren't valid JSON escapes.
        # This is the nuclear option: any \X where X is not one of the
        # 8 valid JSON escape chars → remove the backslash entirely.
        # This handles LaTeX \n (literal backslash + n) which is
        # indistinguishable from a JSON newline escape in broken output.
        # We do this character-by-character inside strings only.
        cleaned = _re.sub(r'\\(?!["\\/bfnrtu])', '', cleaned)

        # Now handle the tricky case: \n from LaTeX titles.
        # If a line contains what looks like a LaTeX-origin \n (preceded
        # by a letter, not whitespace), replace it with a space.
        # E.g.: "Catalogue\nanalysis" → "Catalogue analysis"
        # But preserve genuine JSON newlines in values (\n between entries).
        cleaned = _re.sub(r'([a-zA-Z])\\n([a-zA-Z])', r'\1 \2', cleaned)

        # Remove trailing commas before } and ]
        cleaned = _re.sub(r',\s*}', '}', cleaned)
        cleaned = _re.sub(r',\s*]', ']', cleaned)

        # Handle truncated JSON: close unclosed braces/brackets
        open_braces = cleaned.count('{') - cleaned.count('}')
        open_brackets = cleaned.count('[') - cleaned.count(']')
        if open_braces > 0 or open_brackets > 0:
            lines = cleaned.rstrip().rsplit('\n', 1)
            if len(lines) > 1:
                last_line = lines[-1].strip()
                if last_line and not last_line.endswith((',', '"', '}', ']')):
                    cleaned = lines[0]
            cleaned += ']' * max(0, open_brackets) + '}' * max(0, open_braces)

        return cleaned

    # ── Title matching ───────────────────────────────────────────────────

    def _norm_title(s: str) -> str:
        return _re.sub(r'\s+', ' ', _re.sub(r'[^\w\s]', '', s.lower())).strip()

    def _fuzzy_title_match(llm_title: str, title_to_doc: dict[str, str],
                           norm_cache: dict[str, str]) -> str | None:
        # Exact match
        if llm_title in title_to_doc:
            return title_to_doc[llm_title]
        # Normalized match
        norm_llm = _norm_title(llm_title)
        for paper_title, doc_id in title_to_doc.items():
            norm_paper = norm_cache.get(paper_title)
            if norm_paper is None:
                norm_paper = _norm_title(paper_title)
                norm_cache[paper_title] = norm_paper
            if norm_paper == norm_llm:
                return doc_id
        # Prefix match (for long titles)
        if len(norm_llm) >= 30:
            prefix = norm_llm[:30]
            for paper_title, doc_id in title_to_doc.items():
                if norm_cache.get(paper_title, _norm_title(paper_title)).startswith(prefix):
                    return doc_id
        return None

    # ── DB save helper ───────────────────────────────────────────────────

    def _save_assignments(assignments: dict[str, str]) -> int:
        """Save a batch of assignments to DB immediately. Returns count."""
        if not assignments:
            return 0
        with get_session() as session:
            for doc_id, cluster_name in assignments.items():
                session.execute(text("""
                    UPDATE paper_metadata SET topic_cluster = :cluster
                    WHERE document_id::text = :doc_id
                """), {"cluster": cluster_name, "doc_id": doc_id})
            session.commit()
        return len(assignments)

    # ── Core batch runner ────────────────────────────────────────────────

    _batch_state: dict[int, dict] = {}
    _state_lock = threading.Lock()

    def _run_batch(batch_num: int, batch_papers: list[dict]) -> tuple[int, dict[str, str], int]:
        """Run one batch. Returns (batch_num, {doc_id: cluster}, n_clusters)."""
        system, user = rag_prompts.cluster(batch_papers)

        with _state_lock:
            _batch_state[batch_num] = {"tokens": 0, "status": "generating",
                                       "start": _time.monotonic()}
        try:
            tokens: list[str] = []
            for tok in llm_stream(system, user, model=model, num_ctx=32768):
                tokens.append(tok)
                with _state_lock:
                    _batch_state[batch_num]["tokens"] = len(tokens)

            raw = "".join(tokens)
            with _state_lock:
                _batch_state[batch_num]["status"] = "parsing"

            if not raw.strip():
                raise ValueError("LLM returned empty response")

            cleaned = _sanitize_json(raw)
            data = _json.loads(cleaned, strict=False)
            assignments = data.get("assignments", {})
            n_clusters = len(data.get("clusters", []))

        except Exception as e:
            _log.warning("Batch %d JSON parse failed: %s", batch_num, e)
            with _state_lock:
                _batch_state[batch_num]["status"] = "failed"
            return batch_num, {}, 0

        finally:
            with _state_lock:
                if _batch_state[batch_num]["status"] not in ("done", "failed"):
                    _batch_state[batch_num]["status"] = "done"

        # Match LLM titles to doc_ids
        title_to_doc = {p["title"]: p["doc_id"] for p in batch_papers}
        # Also map sanitized titles (LLM sees stripped-LaTeX titles)
        from sciknow.rag.prompts import _strip_latex
        for p in batch_papers:
            stripped = _strip_latex(p["title"])
            if stripped != p["title"]:
                title_to_doc[stripped] = p["doc_id"]

        norm_cache: dict[str, str] = {}
        doc_assignments: dict[str, str] = {}
        unmatched = 0
        for t, c in assignments.items():
            doc_id = _fuzzy_title_match(t, title_to_doc, norm_cache)
            if doc_id:
                doc_assignments[doc_id] = c
            else:
                unmatched += 1

        if unmatched > 0:
            _log.info("Batch %d: %d/%d titles unmatched",
                      batch_num, unmatched, len(assignments))

        with _state_lock:
            _batch_state[batch_num]["status"] = "done"

        return batch_num, doc_assignments, n_clusters

    # ── Retry wrapper: shrinks batch on failure ──────────────────────────

    _MIN_BATCH = 5  # smallest batch we'll try before giving up

    def _run_with_retry(batch_num: int, batch_papers: list[dict],
                        current_size: int) -> dict[str, str]:
        """Run a batch with automatic retry at halved size on failure."""
        bn, assignments, n_clusters = _run_batch(batch_num, batch_papers)

        if assignments:
            return assignments

        # Batch failed — retry with smaller chunks
        if current_size <= _MIN_BATCH:
            _log.error("Batch %d: giving up after min batch size %d",
                       batch_num, _MIN_BATCH)
            return {}

        smaller = max(current_size // 2, _MIN_BATCH)
        _log.info("Batch %d failed, retrying %d papers in chunks of %d",
                  batch_num, len(batch_papers), smaller)

        all_sub: dict[str, str] = {}
        for i in range(0, len(batch_papers), smaller):
            sub = batch_papers[i:i + smaller]
            sub_num = batch_num * 100 + (i // smaller)
            sub_assignments = _run_with_retry(sub_num, sub, smaller)
            all_sub.update(sub_assignments)

        return all_sub

    # ── Live status printer ──────────────────────────────────────────────

    completed_batches = 0
    _status_stop = threading.Event()

    def _status_printer():
        from rich.live import Live
        from rich.text import Text
        with Live(console=console, refresh_per_second=2, transient=True) as live:
            while not _status_stop.is_set():
                with _state_lock:
                    parts = []
                    for bn in sorted(_batch_state):
                        s = _batch_state[bn]
                        elapsed = _time.monotonic() - s["start"]
                        toks = s["tokens"]
                        st = s["status"]
                        if st == "generating":
                            tps = toks / elapsed if elapsed > 0 else 0
                            parts.append(f"B{bn}:[green]{toks}tok {tps:.1f}t/s[/green]")
                        elif st == "parsing":
                            parts.append(f"B{bn}:[yellow]parsing[/yellow]")
                        elif st == "done":
                            parts.append(f"B{bn}:[dim]done[/dim]")
                        elif st == "failed":
                            parts.append(f"B{bn}:[red]failed[/red]")
                line = f"  [{completed_batches}/{total_batches} done]  " + "  ".join(parts)
                live.update(Text.from_markup(line))
                _status_stop.wait(0.5)

    status_thread = threading.Thread(target=_status_printer, daemon=True)
    status_thread.start()

    # ── Main execution ───────────────────────────────────────────────────

    all_assignments: dict[str, str] = {}
    total_saved = 0

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch_paper_map: dict[int, list[dict]] = {}
            futures = {}
            for batch_start in range(0, len(papers), batch):
                batch_papers = papers[batch_start:batch_start + batch]
                batch_num = batch_start // batch + 1
                batch_paper_map[batch_num] = batch_papers
                fut = pool.submit(_run_with_retry, batch_num, batch_papers, batch)
                futures[fut] = batch_num

            for fut in as_completed(futures):
                batch_num = futures[fut]
                try:
                    doc_assignments = fut.result()
                except Exception as exc:
                    _log.error("Batch %d raised: %s", batch_num, exc)
                    doc_assignments = {}

                all_assignments.update(doc_assignments)
                completed_batches += 1

                # Incremental save (unless dry-run)
                if not dry_run and doc_assignments:
                    saved = _save_assignments(doc_assignments)
                    total_saved += saved

                console.print(
                    f"  Batch {batch_num}/{total_batches}: "
                    f"{len(doc_assignments)}/{len(batch_paper_map[batch_num])} assigned"
                    + (f", saved {total_saved} total" if not dry_run else "")
                )
    finally:
        _status_stop.set()
        status_thread.join(timeout=2)

    # ── Summary ──────────────────────────────────────────────────────────

    unassigned = len(papers) - len(all_assignments)
    console.print(
        f"\nTotal: [bold]{len(all_assignments)}[/bold] / {len(papers)} assigned"
        + (f"  [yellow]({unassigned} unassigned)[/yellow]" if unassigned else "")
    )

    if dry_run:
        from collections import Counter
        counts = Counter(all_assignments.values())
        table = Table(title="Proposed Clusters (dry run — not saved)", box=box.SIMPLE_HEAD)
        table.add_column("Cluster", ratio=2)
        table.add_column("Papers", justify="right", style="cyan")
        for name, n in counts.most_common():
            table.add_row(name, str(n))
        console.print(table)
        return

    console.print(f"[green]✓ Saved topic_cluster for {total_saved} papers[/green]")
    if unassigned:
        console.print(f"[dim]Re-run with --resume to retry the {unassigned} unassigned papers.[/dim]")
