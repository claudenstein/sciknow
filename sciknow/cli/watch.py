"""``sciknow watch`` — repo-watchlist CLI (Phase 45).

Lets the user track upstream research repos for new commits + releases
without remembering to check them by hand. All state lives in
``{data_dir}/watchlist.jsonl`` per the :mod:`sciknow.core.watchlist`
module.

Subcommands:

- ``list``       — show the watchlist + what's new since last check
- ``add <url>``  — start tracking a repo (github only for now)
- ``remove …``   — stop tracking
- ``check [url]`` — fetch HEAD for one repo (or all)
- ``note <url> <text>`` — attach/change a free-form note
- ``seed``       — pre-populate with the research repos sciknow has
                   already borrowed from (karpathy/autoresearch,
                   SakanaAI/AI-Scientist, aiming-lab/AutoResearchClaw,
                   open-fars/openfars, WecoAI/aideml)
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from sciknow.core import watchlist as wl

app = typer.Typer(help="Track upstream research repos for new commits + releases.")
console = Console()


@app.command(name="list")
def list_cmd(
    check: bool = typer.Option(
        False, "--check",
        help="Hit GitHub for each watched repo first, then print the "
             "fresh state. Default behaviour prints whatever was "
             "recorded at last check (so the command is fast + offline).",
    ),
):
    """Show the watchlist. Green rows had new activity since last check."""
    wl.seed_if_empty()
    if check:
        console.print("[dim]fetching GitHub for each watched repo…[/dim]")
        for _ in wl.check_all():
            pass
    rows = wl.list_watched()
    if not rows:
        console.print("[yellow]No repos on the watchlist.[/yellow]")
        console.print("Run [bold]sciknow watch seed[/bold] to pre-populate, or "
                      "[bold]sciknow watch add <github-url>[/bold].")
        return
    table = Table(title="Watched Repos", show_lines=False, expand=True)
    table.add_column("Repo", style="bold")
    table.add_column("★",      justify="right", width=7)
    table.add_column("Pushed",   style="dim",   width=10)
    table.add_column("Δ",      justify="right", width=4)
    table.add_column("Checked",  style="dim",   width=10)
    table.add_column("Note",     overflow="fold")

    for r in rows:
        delta = r.new_commits_since_last_check or 0
        delta_str = (
            f"[green]+{delta}[/green]" if delta > 0 else
            "[dim]—[/dim]"
        )
        pushed = (r.last_pushed_at or "")[:10]
        checked = (r.last_checked_at or "")[:10]
        table.add_row(
            r.key,
            f"{r.stars:,}" if r.stars else "",
            pushed,
            delta_str,
            checked,
            (r.note or "")[:100],
        )
    console.print(table)
    if not check:
        console.print(
            "[dim]Run `sciknow watch list --check` to refresh from GitHub.[/dim]"
        )


@app.command()
def add(
    url: str = typer.Argument(..., help="GitHub URL (https://github.com/owner/repo)."),
    note: str = typer.Option("", "--note", "-n",
                              help="Free-form note about why this repo matters."),
):
    """Start tracking a repo."""
    try:
        r = wl.add(url, note=note)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    console.print(f"[green]✓ Watching[/green] {r.key}" + (f" — {note}" if note else ""))
    console.print("[dim]Run `sciknow watch check` to fetch current HEAD.[/dim]")


@app.command()
def remove(
    url_or_key: str = typer.Argument(..., help="GitHub URL or owner/repo key."),
):
    """Stop tracking a repo (keeps the log event for auditability)."""
    try:
        removed = wl.remove(url_or_key)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    if not removed:
        console.print(f"[yellow]{url_or_key!r} isn't on the watchlist.[/yellow]")
        raise typer.Exit(1)
    console.print(f"[green]✓ Removed[/green] {url_or_key}")


@app.command()
def check(
    url_or_key: str = typer.Argument(
        None,
        help="Repo to check (URL or owner/repo). Omit to check every watched repo.",
    ),
):
    """Fetch one or every watched repo and print what's new."""
    if url_or_key:
        try:
            r = wl.check(url_or_key)
        except Exception as exc:
            console.print(f"[red]check failed:[/red] {exc}")
            raise typer.Exit(1)
        delta = r.new_commits_since_last_check or 0
        mark = f"[green]+{delta} new commits[/green]" if delta > 0 else "[dim]up to date[/dim]"
        console.print(
            f"[bold]{r.key}[/bold]  ★ {r.stars:,}  pushed {r.last_pushed_at}  {mark}"
        )
        return
    table = Table(title="Check results", show_lines=False)
    table.add_column("Repo", style="bold")
    table.add_column("★", justify="right")
    table.add_column("Pushed", style="dim")
    table.add_column("Δ since last", justify="right")
    for r in wl.check_all():
        delta = r.new_commits_since_last_check or 0
        table.add_row(
            r.key,
            f"{r.stars:,}",
            r.last_pushed_at or "",
            (f"[green]+{delta}[/green]" if delta else "[dim]—[/dim]"),
        )
    console.print(table)


@app.command()
def note(
    url_or_key: str = typer.Argument(..., help="GitHub URL or owner/repo key."),
    text: str = typer.Argument(..., help="Note text."),
):
    """Attach or change the free-form note on a watched repo."""
    try:
        if "/" in url_or_key and "github.com" not in url_or_key:
            owner, repo = url_or_key.split("/", 1)
        else:
            owner, repo = wl.parse_github_url(url_or_key)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    wl._append({"kind": "note", "owner": owner, "repo": repo, "note": text})
    console.print(f"[green]✓ Updated note[/green] for {owner}/{repo}")


@app.command()
def seed():
    """Pre-populate the watchlist with sciknow's reference repos."""
    added = wl.seed_if_empty()
    if added:
        console.print(f"[green]✓ Seeded {added} repos.[/green]")
        console.print("[dim]Run `sciknow watch list` to see them.[/dim]")
    else:
        console.print("[yellow]Watchlist already has entries — nothing to seed.[/yellow]")
