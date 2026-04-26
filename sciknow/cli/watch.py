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


# ══════════════════════════════════════════════════════════════════════
# Phase 54.6.137 — velocity query watcher
# ══════════════════════════════════════════════════════════════════════


@app.command(name="add-velocity")
def add_velocity_cmd(
    query: str = typer.Argument(
        ..., help='Semantic search string, e.g. "thermospheric cooling".',
    ),
    note: str = typer.Option(
        "", "--note", "-n", help="Free-form note displayed in the watchlist.",
    ),
    window_days: int = typer.Option(
        180, "--window-days", "-w",
        help="How many days back each check scans OpenAlex. Default 180 "
             "= ~6 months, enough to surface papers that are gathering "
             "momentum without drowning in day-one noise.",
    ),
    top_k: int = typer.Option(
        20, "--top-k", "-k",
        help="Max papers surfaced per check, ranked by citations-per-"
             "active-year. Default 20.",
    ),
):
    """Phase 54.6.137 — register a semantic query to watch for new hot papers.

    On each ``watch check``, OpenAlex is queried for papers published
    in the last ``window_days`` matching the query; results are ranked
    by citation velocity and the delta from the last check is surfaced
    as ``+N new`` so you can see what just started moving.

    Complementary to the interactive expand paths: ``expand-topic`` does
    one-shot bootstrapping, ``watch add-velocity`` subscribes you to a
    topic for ongoing delta notifications.
    """
    try:
        w = wl.add_velocity_query(
            query, note=note, window_days=window_days, top_k=top_k,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    console.print(
        f"[green]✓ Watching velocity query:[/green] {w.query!r}  "
        f"[dim](window={w.window_days}d, top_k={w.top_k})[/dim]"
    )
    console.print("[dim]Run `sciknow watch check-velocity` to get the first snapshot.[/dim]")


@app.command(name="remove-velocity")
def remove_velocity_cmd(
    query: str = typer.Argument(..., help="The exact query string."),
):
    """Stop watching a velocity query. The log entry stays for auditability."""
    if wl.remove_velocity_query(query):
        console.print(f"[green]✓ Stopped watching:[/green] {query!r}")
    else:
        console.print(f"[yellow]No velocity query matches {query!r}.[/yellow]")
        raise typer.Exit(1)


@app.command(name="check-velocity")
def check_velocity_cmd(
    query: str = typer.Argument(
        None,
        help="Query to check. Omit to check every registered velocity query.",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Bypass the 24h cooldown. OpenAlex doesn't hard-limit the "
             "anonymous API, but cooperative usage is the documented norm.",
    ),
    auto_ingest: int = typer.Option(
        0, "--auto-ingest",
        help="If > 0, after each check ingest the top-N *new* papers "
             "(those not in the previous check's DOI set) via "
             "`sciknow corpus download-dois`. 0 = off (default, user-review "
             "gate honored). Set to ~3-5 for light auto-growth on "
             "trusted queries; leave 0 for personal bibliographies.",
    ),
):
    """Phase 54.6.137 — hit OpenAlex for the registered velocity queries."""
    targets: list[wl.WatchedVelocityQuery]
    if query:
        key = wl._normalise_velocity_key(query)
        index = {w.key: w for w in wl.list_watched_velocity_queries()}
        if key not in index:
            console.print(
                f"[red]not watching[/red] {query!r}. "
                f"Run `sciknow watch add-velocity {query!r}` first."
            )
            raise typer.Exit(1)
        targets = [index[key]]
    else:
        targets = wl.list_watched_velocity_queries()
        if not targets:
            console.print("[yellow]No velocity queries on the watchlist.[/yellow]")
            console.print(
                "Run [bold]sciknow watch add-velocity \"your topic\"[/bold] to start."
            )
            return

    table = Table(
        title=f"Velocity queries — last {targets[0].window_days if len(targets) == 1 else '180'}d window",
        show_lines=False, expand=True,
    )
    table.add_column("Query", style="bold", overflow="fold")
    table.add_column("Top paper (title · year · cites/yr)", overflow="fold")
    table.add_column("+new", justify="right", width=6)
    table.add_column("Status", style="dim", width=8)

    # Track which DOIs are "new since last check" for optional auto-ingest.
    new_dois_across_all: list[str] = []

    for t in targets:
        prev_dois = set(t.last_seen_dois or [])
        try:
            got = wl.check_velocity_query(t.query, force=force)
            status = "checked"
        except wl.RateLimited as rl:
            got = rl.cached        # type: ignore[assignment]
            status = "cached"
        except Exception as exc:
            console.print(f"[red]velocity check {t.query!r} failed:[/red] {exc}")
            table.add_row(t.query[:60], "[red](error)[/red]", "—", "[red]err[/red]")
            continue

        top_paper_cell = "[dim](no papers in window)[/dim]"
        if got.last_top_papers:
            p = got.last_top_papers[0]
            title = (p.get("title") or "")[:90]
            yr = p.get("year") or "?"
            vel = p.get("velocity") or 0.0
            top_paper_cell = f"{title} · {yr} · {vel:.1f}/yr"

        # Mark newly-surfaced DOIs (not in the previous check's set).
        if status == "checked":
            fresh = [p for p in got.last_top_papers
                     if p.get("doi") and p["doi"] not in prev_dois]
            new_dois_across_all.extend(
                p["doi"] for p in fresh[: max(auto_ingest, 0)]
            )

        status_mark = (
            "[green]✓[/green]" if status == "checked"
            else "[yellow]cached[/yellow]"
        )
        delta = got.new_since_last_check or 0
        delta_cell = (
            f"[green]+{delta}[/green]" if delta > 0 else "[dim]—[/dim]"
        )
        table.add_row(got.query[:60], top_paper_cell, delta_cell, status_mark)

    console.print(table)

    # Optional auto-ingest path. Hands DOIs to the existing pipeline —
    # no per-paper manual selection, but capped by --auto-ingest N per
    # query so a hot topic doesn't dump hundreds of papers at once.
    if auto_ingest > 0 and new_dois_across_all:
        # Dedup across queries and cap.
        seen: set[str] = set()
        unique = [d for d in new_dois_across_all if not (d in seen or seen.add(d))]
        console.print(
            f"\n[bold]--auto-ingest[/bold]: queuing {len(unique)} new DOI(s) "
            "to `sciknow corpus download-dois`…"
        )
        import subprocess, sys
        for doi in unique:
            console.print(f"  {doi}")
        try:
            res = subprocess.run(
                [sys.executable, "-m", "sciknow.cli.main", "corpus",
                 "download-dois", *unique],
                check=False,
            )
            if res.returncode == 0:
                console.print("[green]✓ Queued for ingestion.[/green]")
            else:
                console.print(
                    f"[yellow]download-dois exited with code "
                    f"{res.returncode} — check the log.[/yellow]"
                )
        except Exception as exc:
            console.print(f"[red]auto-ingest failed:[/red] {exc}")


@app.command(name="list-velocity")
def list_velocity_cmd():
    """Show all registered velocity queries with their rolling state."""
    rows = wl.list_watched_velocity_queries()
    if not rows:
        console.print("[yellow]No velocity queries on the watchlist.[/yellow]")
        console.print(
            "Run [bold]sciknow watch add-velocity \"your topic\"[/bold] to start."
        )
        return
    table = Table(title="Watched Velocity Queries", show_lines=False, expand=True)
    table.add_column("Query", style="bold", overflow="fold")
    table.add_column("Window", justify="right", width=8)
    table.add_column("Top-K", justify="right", width=7)
    table.add_column("Last check", style="dim", width=12)
    table.add_column("Δ", justify="right", width=5)
    table.add_column("Note", overflow="fold")
    for w in rows:
        last = (w.last_checked_at or "")[:10]
        delta = w.new_since_last_check or 0
        delta_cell = f"[green]+{delta}[/green]" if delta > 0 else "[dim]—[/dim]"
        table.add_row(
            w.query,
            f"{w.window_days}d",
            str(w.top_k),
            last or "[dim]never[/dim]",
            delta_cell,
            (w.note or "")[:80],
        )
    console.print(table)


# ── Phase 54.6.316 — `sciknow watch study` ────────────────────────────
# Clone a watched repo into a local research scratch dir, scaffold a
# memo under `docs/research/<slug>.md`, and print a concise summary.
# Intended to be run by a human (or an agent) after `sciknow watch add`
# to kick off the actual code-reading pass. The memo is a stub + file
# inventory + detected stack — the "what can we learn" section is left
# to the reader (or the next agent hop) to fill in.


@app.command()
def study(
    url: str = typer.Argument(
        ..., help="GitHub URL OR owner/repo key already on the watchlist."
    ),
    depth: int = typer.Option(
        1, "--depth", help="git clone --depth; 1 is enough for a read-through."
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Re-clone even if the scratch dir already exists.",
    ),
):
    """Clone a watched repo + scaffold a research memo.

    Usage::

        sciknow watch add https://github.com/huggingface/ml-intern --note "..."
        sciknow watch study huggingface/ml-intern

    Writes:
      data/research/<owner>__<repo>/   — shallow clone scratch dir
      docs/research/<owner>__<repo>.md — memo stub (README excerpt +
                                         file inventory + stack notes)

    The memo is deliberately a stub — the "What could sciknow learn?"
    section is left empty for the reader (or the next agent hop) to fill
    in after reading the code.
    """
    import subprocess
    from pathlib import Path as _P
    from sciknow.config import settings

    # Resolve URL from either form
    u = url.strip()
    if u.startswith("http"):
        try:
            owner, repo = wl.parse_github_url(u)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2)
    elif "/" in u and not u.startswith("/"):
        owner, _, repo = u.partition("/")
        u = f"https://github.com/{owner}/{repo}"
    else:
        console.print("[red]Provide either a full github URL or owner/repo[/red]")
        raise typer.Exit(2)

    slug = f"{owner}__{repo}"
    scratch = settings.data_dir / "research" / slug
    memo = _P("docs/research") / f"{slug}.md"
    memo.parent.mkdir(parents=True, exist_ok=True)

    # Clone (shallow) unless dir exists
    if scratch.exists():
        if force:
            import shutil
            shutil.rmtree(scratch)
        else:
            console.print(f"[dim]scratch exists — skipping clone: {scratch}[/dim]")
    if not scratch.exists():
        scratch.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"cloning [cyan]{u}[/cyan] → {scratch}")
        r = subprocess.run(
            ["git", "clone", "--depth", str(depth), u, str(scratch)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            console.print(f"[red]git clone failed:[/red]\n{r.stderr}")
            raise typer.Exit(1)

    # Walk the tree to build an inventory (cap at 200 files so the memo
    # stays readable on huge repos).
    skip_dirs = {".git", "node_modules", "__pycache__", "dist", "build",
                 ".venv", "venv", ".next", ".cache"}
    files: list[tuple[str, int]] = []
    for p in scratch.rglob("*"):
        if not p.is_file():
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        files.append((str(p.relative_to(scratch)), size))
    files.sort()

    # Detect stack
    stack_sig = {
        "Python":  any(f.endswith(".py") for f, _ in files),
        "TypeScript/JavaScript": any(f.endswith((".ts", ".tsx", ".js", ".jsx")) for f, _ in files),
        "Rust":    any(f.endswith(".rs") for f, _ in files),
        "Go":      any(f.endswith(".go") for f, _ in files),
        "pyproject.toml": any(f == "pyproject.toml" for f, _ in files),
        "package.json": any(f == "package.json" for f, _ in files),
        "Cargo.toml": any(f == "Cargo.toml" for f, _ in files),
        "Dockerfile": any(f.endswith("Dockerfile") for f, _ in files),
    }
    detected_stack = [k for k, v in stack_sig.items() if v]

    # Read README (first one we find)
    readme_text = ""
    for candidate in ("README.md", "README.rst", "README", "readme.md"):
        p = scratch / candidate
        if p.exists():
            try:
                readme_text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                readme_text = ""
            break

    # Build the memo
    lines: list[str] = [
        f"# {owner}/{repo} — research memo",
        "",
        f"**URL**: <{u}>",
        f"**Clone scratch**: `{scratch}`",
        f"**Cloned**: {wl._ts()}",
        f"**Stack detected**: {', '.join(detected_stack) or 'unknown'}",
        f"**File count** (excl. vendor/build): {len(files)}",
        "",
        "## README (first 2000 chars)",
        "",
        "```markdown",
        (readme_text[:2000] or "(no README found)").rstrip(),
        "```",
        "",
        "## File inventory (top 50 by path)",
        "",
    ]
    for f, size in files[:50]:
        lines.append(f"- `{f}` ({size} bytes)")
    if len(files) > 50:
        lines.append(f"- … and {len(files) - 50} more files.")

    lines += [
        "",
        "## What could sciknow learn?",
        "",
        "*(To be filled in after reading the code. Candidate hooks:*",
        "*sciknow's ingest / retrieval / wiki / book-writer / autowrite /*",
        "*monitor pipelines. Look for new sources, new matching algorithms,*",
        "*new prompt patterns, new agent architectures, or new GUI ideas.)*",
        "",
        "## Relevance verdict",
        "",
        "- [ ] high — implement / port",
        "- [ ] medium — useful reference",
        "- [ ] low — skim only",
        "- [ ] not applicable",
        "",
    ]

    memo.write_text("\n".join(lines), encoding="utf-8")
    console.print(
        f"[green]✓ memo scaffolded[/green] → {memo}  "
        f"[dim]({len(files)} files, {', '.join(detected_stack) or 'no stack'})[/dim]"
    )
