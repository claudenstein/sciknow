"""Phase 50.C — `sciknow spans` CLI.

Browse the `spans` table populated by sciknow.observability.tracer.
Two subcommands:

  - tail   most recent spans (one-liner per row)
  - show   full metadata dump for one trace id (with DAG indentation)

This is a debug UI, not a monitoring dashboard. For anything
serious, point a psql client at the `spans` table directly.
"""
from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Inspect local observability spans (tracer output).")
console = Console()


@app.command("tail")
def tail(
    limit: int = typer.Option(30, "--limit", "-n",
                               help="How many recent spans to show."),
    name: str = typer.Option("", "--name",
                              help="Filter by span name (exact match)."),
    trace: str = typer.Option("", "--trace",
                               help="Filter to one trace id (prefix or full)."),
):
    """Most recent spans across all traces."""
    from sciknow.cli import preflight
    preflight()
    from sqlalchemy import text as sql_text
    from sciknow.storage.db import get_session

    with get_session() as session:
        rows = session.execute(sql_text("""
            SELECT id::text, trace_id::text, name, status, duration_ms,
                   started_at, error
            FROM spans
            WHERE (:name_q IS NULL OR name = :name_q)
              AND (:trace_q IS NULL OR trace_id::text LIKE :trace_q)
            ORDER BY started_at DESC
            LIMIT :lim
        """), {
            "name_q": name or None,
            "trace_q": (f"{trace}%") if trace else None,
            "lim": max(1, min(int(limit), 500)),
        }).fetchall()
    if not rows:
        console.print("[dim](no spans recorded yet — the tracer is opt-in; "
                      "call sciknow.observability.span(name, …) from any op)[/dim]")
        raise typer.Exit(0)

    table = Table(title=f"Spans (most recent {len(rows)})")
    table.add_column("when", style="dim")
    table.add_column("trace", style="dim")
    table.add_column("name")
    table.add_column("ms", justify="right")
    table.add_column("status")
    table.add_column("error", overflow="fold", max_width=60)
    for span_id, tid, nm, status, dur, ts, err in rows:
        status_disp = (
            "[green]ok[/green]" if status == "ok" else f"[red]{status}[/red]"
        )
        dur_disp = "-" if dur is None else str(dur)
        table.add_row(
            ts.strftime("%m-%d %H:%M:%S"),
            (tid or "")[:8],
            (nm or "")[:60],
            dur_disp,
            status_disp,
            (err or "")[:80],
        )
    console.print(table)


@app.command("show")
def show(
    trace_id: str = typer.Argument(...,
        help="Full trace id (or prefix) to dump."),
):
    """Full metadata + DAG for one trace."""
    from sciknow.cli import preflight
    preflight()
    from sqlalchemy import text as sql_text
    from sciknow.storage.db import get_session

    with get_session() as session:
        rows = session.execute(sql_text("""
            SELECT id::text, parent_span_id::text, name, status,
                   duration_ms, started_at, metadata_json, error
            FROM spans
            WHERE trace_id::text LIKE :tid
            ORDER BY started_at
        """), {"tid": f"{trace_id.strip()}%"}).fetchall()
    if not rows:
        console.print(f"[red]no trace matches {trace_id!r}[/red]")
        raise typer.Exit(1)

    # Build a parent → children index for DAG rendering
    children: dict[str | None, list[tuple]] = {}
    for r in rows:
        children.setdefault(r[1], []).append(r)
    # Roots = spans whose parent isn't in this trace (or None)
    own_ids = {r[0] for r in rows}
    roots = [r for r in rows if r[1] is None or r[1] not in own_ids]

    def _render(r, depth: int) -> None:
        span_id, parent, name, status, dur, ts, meta, err = r
        indent = "  " * depth
        dur_disp = "-" if dur is None else f"{dur}ms"
        status_disp = (
            "[green]ok[/green]" if status == "ok" else f"[red]{status}[/red]"
        )
        console.print(
            f"{indent}[bold]{name}[/bold]  {dur_disp}  {status_disp}  "
            f"[dim]{ts.strftime('%H:%M:%S')}[/dim]"
        )
        if meta:
            try:
                meta_dict = meta if isinstance(meta, dict) else json.loads(meta)
            except Exception:
                meta_dict = {}
            for k, v in list(meta_dict.items())[:8]:
                console.print(f"{indent}  [dim]{k}=[/dim] {str(v)[:120]}")
        if err:
            console.print(f"{indent}  [red]err:[/red] {err[:200]}")
        for ch in children.get(span_id, []):
            _render(ch, depth + 1)

    console.print(f"[bold]Trace {rows[0][0][:8]}[/bold] · {len(rows)} span(s)")
    for r in roots:
        _render(r, 0)


@app.command("stats")
def stats():
    """Aggregate by span name — count + p50/p95 duration."""
    from sciknow.cli import preflight
    preflight()
    from sqlalchemy import text as sql_text
    from sciknow.storage.db import get_session

    with get_session() as session:
        rows = session.execute(sql_text("""
            SELECT name,
                   COUNT(*) AS n,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95,
                   MAX(duration_ms) AS max_ms
            FROM spans WHERE duration_ms IS NOT NULL
            GROUP BY name
            ORDER BY n DESC LIMIT 50
        """)).fetchall()
    if not rows:
        console.print("[dim](no spans with duration recorded)[/dim]")
        raise typer.Exit(0)
    table = Table(title="Span stats by name")
    table.add_column("name")
    table.add_column("n", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("max ms", justify="right")
    for name, n, p50, p95, mx in rows:
        table.add_row(
            (name or "")[:60], str(int(n)),
            f"{p50:.0f}" if p50 is not None else "-",
            f"{p95:.0f}" if p95 is not None else "-",
            str(int(mx)) if mx is not None else "-",
        )
    console.print(table)
