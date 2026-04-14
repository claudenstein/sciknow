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
        help="Hit GitHub + HF for each watched entry first, then print "
             "the fresh state. Default behaviour prints whatever was "
             "recorded at last check (so the command is fast + offline).",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="With --check, bypass the 24h per-entry cooldown. Use "
             "sparingly — the anonymous GitHub API allows only 60 "
             "requests/hour.",
    ),
):
    """Show the watchlist (repos + HF benchmarks).

    Green rows in the repo table had new commits since last check;
    the benchmark table flags regime changes (new #1 model).
    """
    wl.seed_if_empty()
    wl.seed_benchmarks_if_missing()
    if check:
        console.print("[dim]fetching GitHub + HF for each watched entry (cooldown 24h)…[/dim]")
        n_checked, n_cached, n_err = 0, 0, 0
        for _repo, status in wl.check_all(force=force):
            if status == "checked": n_checked += 1
            elif status == "cached": n_cached += 1
            else: n_err += 1
        for _bench, status in wl.check_all_benchmarks(force=force):
            if status == "checked": n_checked += 1
            elif status == "cached": n_cached += 1
            else: n_err += 1
        if n_cached:
            console.print(
                f"[dim]{n_checked} checked, {n_cached} within 24h cooldown "
                f"(pass --force to bypass), {n_err} errored.[/dim]"
            )
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

    # Phase 46.G — render the benchmark table below the repo table.
    benches = wl.list_watched_benchmarks()
    if benches:
        btable = Table(title="Watched Benchmarks", expand=True)
        btable.add_column("Benchmark (HF dataset)", style="bold")
        btable.add_column("Top model", overflow="fold")
        btable.add_column("Score", justify="right", width=8)
        btable.add_column("Modified", style="dim", width=10)
        btable.add_column("Checked",  style="dim", width=10)
        btable.add_column("Note",     overflow="fold")
        for b in benches:
            top_name = ""
            top_score = ""
            if b.top_models:
                tm = b.top_models[0]
                top_name  = str(tm.get("name", ""))
                s = tm.get("score")
                top_score = f"{s}" if s is not None else ""
            # Regime-change badge when the #1 just moved
            if b.top_changed_since_last_check:
                top_name = f"[green]🆕 {top_name}[/green]"
            btable.add_row(
                b.dataset, top_name, top_score,
                (b.last_modified or "")[:10],
                (b.last_checked_at or "")[:10],
                (b.note or "")[:100],
            )
        console.print(btable)

    if not check:
        console.print(
            "[dim]Run `sciknow watch list --check` to refresh from GitHub + HF.[/dim]"
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


@app.command(name="add-benchmark")
def add_benchmark(
    dataset: str = typer.Argument(
        ...,
        help="HuggingFace dataset slug OR URL "
             "(e.g. 'allenai/olmOCR-bench' or the full URL).",
    ),
    note: str = typer.Option("", "--note", "-n",
                              help="Free-form note about why this benchmark matters."),
):
    """Phase 46.G — start tracking an HF benchmark leaderboard.

    Each check re-fetches the dataset's ``lastModified`` + ``sha`` and
    best-effort-parses the README for a top-5 ranked model table. When
    the #1 model changes between checks, the row is flagged on
    ``sciknow watch list``.
    """
    try:
        b = wl.add_benchmark(dataset, note=note)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    console.print(
        f"[green]✓ Watching benchmark[/green] {b.dataset}"
        + (f" — {note}" if note else "")
    )
    console.print(
        "[dim]Run `sciknow watch list --check` to fetch current top-5.[/dim]"
    )


@app.command()
def remove(
    url_or_key: str = typer.Argument(
        ...,
        help="GitHub URL / owner/repo key (repos) OR HF dataset slug (benchmarks).",
    ),
):
    """Stop tracking a repo or benchmark (keeps the log event for auditability).

    Tries repo removal first, then benchmark removal.
    """
    # Repo path
    try:
        removed = wl.remove(url_or_key)
        if removed:
            console.print(f"[green]✓ Removed repo[/green] {url_or_key}")
            return
    except ValueError:
        pass   # not a valid github URL — try benchmarks
    # Benchmark path
    if wl.remove_benchmark(url_or_key):
        console.print(f"[green]✓ Removed benchmark[/green] {url_or_key}")
        return
    console.print(f"[yellow]{url_or_key!r} isn't on the watchlist.[/yellow]")
    raise typer.Exit(1)


@app.command()
def check(
    url_or_key: str = typer.Argument(
        None,
        help="Repo to check (URL or owner/repo). Omit to check every watched repo.",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Bypass the 24h cooldown. Defaults to False — GitHub's "
             "anonymous API allows 60 req/hour and the upstream repos "
             "we watch don't ship day-over-day.",
    ),
):
    """Fetch one or every watched repo and print what's new.

    A repo is only re-checked if its last successful check was more
    than 24h ago; otherwise the cached state is shown. Use ``--force``
    to override (or set ``GITHUB_TOKEN`` in the env to raise the
    API budget to 5k/hour).
    """
    if url_or_key:
        # Route: is this a repo we're watching, or a benchmark? If it
        # matches a known bench slug exactly, go down the benchmark
        # path; otherwise try the repo path first.
        bench_keys = {b.dataset for b in wl.list_watched_benchmarks()}
        if url_or_key.strip() in bench_keys:
            try:
                b = wl.check_benchmark(url_or_key, force=force)
            except wl.RateLimited as rl:
                cached = rl.cached
                top = cached.top_models[0].get("name") if cached.top_models else "?"
                console.print(
                    f"[bold]{cached.key}[/bold]  top: {top}  modified {cached.last_modified}  "
                    f"[yellow](cached; next check in {rl.hours_remaining:.1f}h — pass --force to override)[/yellow]"
                )
                return
            except Exception as exc:
                console.print(f"[red]bench check failed:[/red] {exc}")
                raise typer.Exit(1)
            top = b.top_models[0].get("name") if b.top_models else "(no leaderboard parsed)"
            score = b.top_models[0].get("score") if b.top_models else ""
            flag = " [green](NEW #1)[/green]" if b.top_changed_since_last_check else ""
            console.print(
                f"[bold]{b.dataset}[/bold]  top: {top}"
                + (f" ({score})" if score != "" else "") + f"{flag}  modified {b.last_modified}"
            )
            return

        try:
            r = wl.check(url_or_key, force=force)
        except wl.RateLimited as rl:
            cached = rl.cached
            delta = cached.new_commits_since_last_check or 0
            mark = f"[green]+{delta} new commits[/green]" if delta > 0 else "[dim]up to date[/dim]"
            console.print(
                f"[bold]{cached.key}[/bold]  ★ {cached.stars:,}  pushed {cached.last_pushed_at}  "
                f"{mark}  [yellow](cached; next check in {rl.hours_remaining:.1f}h — pass --force to override)[/yellow]"
            )
            return
        except Exception as exc:
            console.print(f"[red]check failed:[/red] {exc}")
            raise typer.Exit(1)
        delta = r.new_commits_since_last_check or 0
        mark = f"[green]+{delta} new commits[/green]" if delta > 0 else "[dim]up to date[/dim]"
        console.print(
            f"[bold]{r.key}[/bold]  ★ {r.stars:,}  pushed {r.last_pushed_at}  {mark}"
        )
        return
    table = Table(title=f"Check results — repos (24h cooldown{' — BYPASSED' if force else ''})",
                  show_lines=False)
    table.add_column("Repo", style="bold")
    table.add_column("★", justify="right")
    table.add_column("Pushed", style="dim")
    table.add_column("Δ since last", justify="right")
    table.add_column("Status", style="dim")
    for r, status in wl.check_all(force=force):
        delta = r.new_commits_since_last_check or 0
        status_mark = (
            "[green]✓[/green]" if status == "checked" else
            "[yellow]cached[/yellow]" if status == "cached" else
            "[red]err[/red]"
        )
        table.add_row(
            r.key,
            f"{r.stars:,}" if r.stars else "",
            r.last_pushed_at or "",
            (f"[green]+{delta}[/green]" if delta else "[dim]—[/dim]"),
            status_mark,
        )
    console.print(table)

    # Phase 46.G — benchmarks table
    btable = Table(title="Check results — benchmarks", show_lines=False)
    btable.add_column("Benchmark", style="bold")
    btable.add_column("Top model", overflow="fold")
    btable.add_column("Score", justify="right")
    btable.add_column("Modified", style="dim")
    btable.add_column("Δ #1", justify="center")
    btable.add_column("Status", style="dim")
    ran_any = False
    for b, status in wl.check_all_benchmarks(force=force):
        ran_any = True
        top_name = (b.top_models[0].get("name") if b.top_models else "") or "—"
        top_score = (b.top_models[0].get("score") if b.top_models else "")
        top_score_str = f"{top_score}" if top_score != "" else ""
        delta_mark = "[green]NEW[/green]" if b.top_changed_since_last_check else "[dim]—[/dim]"
        status_mark = (
            "[green]✓[/green]" if status == "checked" else
            "[yellow]cached[/yellow]" if status == "cached" else
            "[red]err[/red]"
        )
        btable.add_row(
            b.dataset, top_name, top_score_str,
            (b.last_modified or "")[:10],
            delta_mark, status_mark,
        )
    if ran_any:
        console.print(btable)


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
