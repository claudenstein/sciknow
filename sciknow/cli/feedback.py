"""Phase 50.B — `sciknow feedback` CLI.

Capture + browse thumbs-up / thumbs-down ratings on any answer the
system produces. Stored in the `feedback` table (see migration 0020).

The point of this command isn't to be a full annotation UI — it's a
lightweight CLI hook so power users can record labels from terminal
workflows. The web reader will grow the same feature later.
"""
from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Capture + browse feedback on LLM answers (thumbs-up/down + comment).")
console = Console()


def _parse_chunks(raw: str) -> list[str]:
    return [c.strip() for c in (raw or "").split(",") if c.strip()]


_SCORE_ALIASES = {
    "up": 1, "+": 1, "+1": 1, "1": 1, "pos": 1, "positive": 1, "good": 1,
    "zero": 0, "0": 0, "neutral": 0, "mid": 0,
    "down": -1, "-": -1, "-1": -1, "neg": -1, "negative": -1, "bad": -1,
}


@app.command("add")
def add(
    score: str = typer.Argument(...,
        help="up / down / neutral (or 1 / -1 / 0 / pos / neg). Bare '-1' "
             "is rejected by Typer because it looks like a flag — quote it "
             "or use 'down'."),
    op: str = typer.Option("ask", "--op",
                            help="Which op was rated — 'ask', 'write', 'review', 'autowrite', 'kg_edge', ..."),
    query: str = typer.Option("", "--query", "-q",
                               help="The query / topic that produced the rated answer."),
    comment: str = typer.Option("", "--comment", "-c",
                                 help="Optional free-text note."),
    draft_id: str = typer.Option("", "--draft",
                                  help="If rating a draft, its UUID (full or prefix)."),
    chunk_ids: str = typer.Option("", "--chunks",
                                   help="Comma-separated chunk-id list from the answer's sources."),
    preview: str = typer.Option("", "--preview", "-p",
                                 help="Short excerpt of the rated answer for later triage."),
):
    """Record one feedback row."""
    from sciknow.cli import preflight
    preflight()
    from sqlalchemy import text as sql_text
    from sciknow.storage.db import get_session

    score_int = _SCORE_ALIASES.get((score or "").strip().lower())
    if score_int is None:
        console.print(
            f"[red]score {score!r} not recognised — use up/down/neutral or 1/-1/0.[/red]"
        )
        raise typer.Exit(2)
    chunks = _parse_chunks(chunk_ids)
    # Resolve a draft_id prefix to a full UUID (same UX as other CLI cmds)
    did: str | None = None
    if draft_id:
        with get_session() as session:
            row = session.execute(sql_text(
                "SELECT id::text FROM drafts WHERE id::text LIKE :q LIMIT 1"
            ), {"q": f"{draft_id.strip()}%"}).fetchone()
        if not row:
            console.print(f"[red]no draft matches prefix {draft_id!r}[/red]")
            raise typer.Exit(2)
        did = row[0]

    with get_session() as session:
        row = session.execute(sql_text("""
            INSERT INTO feedback (op, query, response_preview, score, comment,
                                  draft_id, chunk_ids, extras)
            VALUES (:op, :q, :preview, :score, :comment,
                    CAST(:did AS uuid), CAST(:chunks AS jsonb), CAST(:extras AS jsonb))
            RETURNING id::text, created_at
        """), {
            "op": op, "q": query or None, "preview": preview or None,
            "score": score_int, "comment": comment or None,
            "did": did, "chunks": json.dumps(chunks),
            "extras": json.dumps({}),
        })
        session.commit()
        fb_id, created_at = row.fetchone()
    sign = "+1" if score_int > 0 else ("-1" if score_int < 0 else " 0")
    console.print(
        f"[green]✓ recorded feedback[/green] {fb_id[:8]}… score={sign} "
        f"op={op} at {created_at.strftime('%Y-%m-%d %H:%M')}"
    )


@app.command("list")
def list_cmd(
    limit: int = typer.Option(20, "--limit", "-n"),
    op: str = typer.Option("", "--op", help="Filter by op ('ask', 'write', ...)."),
    score: Optional[int] = typer.Option(None, "--score",
                                          help="Filter by score (-1 / 0 / +1)."),
):
    """List recent feedback rows."""
    from sciknow.cli import preflight
    preflight()
    from sqlalchemy import text as sql_text
    from sciknow.storage.db import get_session

    with get_session() as session:
        rows = session.execute(sql_text("""
            SELECT id::text, created_at, op, score, query, comment, draft_id::text
            FROM feedback
            WHERE (:op_q IS NULL OR op = :op_q)
              AND (:score_q IS NULL OR score = :score_q)
            ORDER BY created_at DESC LIMIT :lim
        """), {
            "op_q": op or None,
            "score_q": score,
            "lim": max(1, min(int(limit), 500)),
        }).fetchall()
    if not rows:
        console.print("[dim](no feedback recorded)[/dim]")
        raise typer.Exit(0)
    table = Table(title=f"Feedback (most recent {len(rows)})")
    table.add_column("when", style="dim")
    table.add_column("score", justify="center")
    table.add_column("op")
    table.add_column("query", overflow="fold", max_width=40)
    table.add_column("comment", overflow="fold", max_width=40)
    table.add_column("draft", style="dim")
    for fb_id, ts, op_v, sc, q, cm, did in rows:
        sc_disp = "[green]👍[/green]" if sc > 0 else ("[red]👎[/red]" if sc < 0 else "○")
        table.add_row(
            ts.strftime("%m-%d %H:%M"),
            sc_disp,
            op_v,
            (q or "")[:60],
            (cm or "")[:60],
            (did or "")[:8],
        )
    console.print(table)


@app.command("stats")
def stats():
    """Summary counts by op × score."""
    from sciknow.cli import preflight
    preflight()
    from sqlalchemy import text as sql_text
    from sciknow.storage.db import get_session

    with get_session() as session:
        rows = session.execute(sql_text("""
            SELECT op, score, COUNT(*) AS n
            FROM feedback
            GROUP BY op, score
            ORDER BY op, score
        """)).fetchall()
    if not rows:
        console.print("[dim](no feedback recorded)[/dim]")
        raise typer.Exit(0)
    # Pivot to op → {pos, zero, neg}
    agg: dict[str, dict[int, int]] = {}
    for op_v, sc, n in rows:
        agg.setdefault(op_v, {-1: 0, 0: 0, 1: 0})[int(sc)] = int(n)
    table = Table(title="Feedback counts by op")
    table.add_column("op")
    table.add_column("👍", justify="right")
    table.add_column("○",  justify="right")
    table.add_column("👎", justify="right")
    table.add_column("Σ",  justify="right", style="bold")
    total_pos = total_zero = total_neg = 0
    for op_v in sorted(agg):
        d = agg[op_v]
        pos, zero, neg = d[1], d[0], d[-1]
        total_pos += pos; total_zero += zero; total_neg += neg
        table.add_row(op_v, str(pos), str(zero), str(neg), str(pos + zero + neg))
    table.add_section()
    table.add_row("total", str(total_pos), str(total_zero), str(total_neg),
                  str(total_pos + total_zero + total_neg))
    console.print(table)
