"""Phase 54.6.24 — automated backup system.

CLI surface:

    sciknow backup run [--all-projects] [--no-system]  # snapshot now
    sciknow backup list                                # show history
    sciknow backup status                              # schedule + last backup info
    sciknow backup schedule [--hour HH]                # install daily crontab
    sciknow backup unschedule                          # remove crontab entry

Produces two kinds of output per run:

  * **Per-project checkpoint** (``<slug>.skproj.tar``): PG dump + Qdrant
    snapshot names + data/ dir + .env.overlay. Reuses the same staging
    logic as ``sciknow project archive``.
  * **System bundle** (``sciknow-system.tar.gz``): ``.env``, ``pyproject.toml``,
    ``uv.lock``, ``alembic.ini``, ``migrations/``, ``.active-project``,
    the ``sciknow/`` source package — everything needed to ``uv sync``
    on a fresh machine, then ``project unarchive`` each checkpoint.

Backup sets are stored under ``archives/backups/<timestamp>/`` (already
gitignored) with a ``.backup-state.json`` sidecar for history + schedule
tracking. Retention defaults to 7 sets (``BACKUP_RETAIN_COUNT`` in ``.env``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="backup",
    help="Automated backups — snapshot projects + system config.",
    no_args_is_help=True,
)
console = Console()

_BACKUP_DIR_NAME = "backups"
_STATE_FILE = ".backup-state.json"
_LOCK_FILE = ".backup.lock"
_CRON_MARKER = "# sciknow-auto-backup"


def _backup_root() -> Path:
    from sciknow.core.project import _repo_root
    return _repo_root() / "archives" / _BACKUP_DIR_NAME


def _state_path() -> Path:
    return _backup_root() / _STATE_FILE


def _read_state() -> dict:
    p = _state_path()
    if not p.exists():
        return {"backups": [], "schedule": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"backups": [], "schedule": None}


def _write_state(state: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _stage_project(project, staging: Path) -> None:
    """Stage a project into a directory: manifest + PG dump + Qdrant
    snapshot names + data copy + env overlay. Mirrors the archive
    command's staging logic but without tarring or dropping live state."""
    from sciknow.config import settings

    staging.mkdir(parents=True, exist_ok=True)

    (staging / "manifest.txt").write_text(
        f"slug: {project.slug}\n"
        f"created_at: {datetime.now(timezone.utc).isoformat()}\n"
        f"pg_database: {project.pg_database}\n"
        f"qdrant_prefix: {project.qdrant_prefix}\n"
    )

    dump_path = staging / "postgres.dump"
    env = os.environ.copy()
    env["PGPASSWORD"] = settings.pg_password
    res = subprocess.run([
        "pg_dump",
        "-h", settings.pg_host,
        "-p", str(settings.pg_port),
        "-U", settings.pg_user,
        "-F", "c", "-f", str(dump_path),
        project.pg_database,
    ], env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"pg_dump failed for {project.slug}: {res.stderr[:500]}")

    try:
        from sciknow.storage.qdrant import get_client
        client = get_client()
        snaps_dir = staging / "qdrant_snapshots"
        snaps_dir.mkdir()
        existing = {c.name for c in client.get_collections().collections}
        for coll in (project.papers_collection, project.abstracts_collection,
                     project.wiki_collection):
            if coll not in existing:
                continue
            try:
                snap = client.create_snapshot(collection_name=coll)
                (snaps_dir / f"{coll}.snapshot_name").write_text(snap.name)
            except Exception:
                pass
    except Exception:
        pass

    data_target = staging / "data"
    if project.data_dir.exists():
        shutil.copytree(project.data_dir, data_target)

    if project.env_overlay_path.exists():
        shutil.copy2(project.env_overlay_path, staging / ".env.overlay")


def _build_system_bundle(output_path: Path) -> None:
    """Build a tarball with .env, pyproject.toml, uv.lock, alembic.ini,
    migrations/, .active-project, and the sciknow/ source package."""
    from sciknow.core.project import _repo_root
    root = _repo_root()

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "sciknow-system"
        stage.mkdir()

        for name in (".env", ".env.example", "pyproject.toml", "uv.lock",
                     "alembic.ini", ".active-project"):
            src = root / name
            if src.exists():
                shutil.copy2(src, stage / name)

        mig = root / "migrations"
        if mig.exists():
            shutil.copytree(mig, stage / "migrations")

        src_pkg = root / "sciknow"
        if src_pkg.exists():
            shutil.copytree(
                src_pkg, stage / "sciknow",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

        with tarfile.open(output_path, "w:gz") as tf:
            tf.add(stage, arcname="sciknow-system")


def _parse_ts(ts: str) -> datetime | None:
    """Parse a backup-dir timestamp like ``20260415T030000Z``."""
    try:
        return datetime.strptime(ts, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _prune_old_backups(state: dict, retain: int, retain_days: int = 0) -> list[str]:
    """Remove backup sets beyond retention. Returns list of removed dir names.

    Count-based pruning drops the oldest entries until at most ``retain``
    remain. If ``retain_days > 0``, any set older than that many days is
    ALSO removed (whichever deletes more wins — the unions combine).
    """
    backups = state.get("backups", [])
    to_remove_entries: list[dict] = []

    # Count-based
    if retain > 0 and len(backups) > retain:
        to_remove_entries.extend(backups[:-retain])

    # Age-based
    if retain_days > 0:
        cutoff = datetime.now(timezone.utc).timestamp() - retain_days * 86400
        for entry in backups:
            ts = _parse_ts(entry.get("timestamp", ""))
            if ts is not None and ts.timestamp() < cutoff:
                if entry not in to_remove_entries:
                    to_remove_entries.append(entry)

    removed: list[str] = []
    for entry in to_remove_entries:
        d = _backup_root() / entry["dir"]
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        removed.append(entry["dir"])

    if removed:
        state["backups"] = [b for b in backups if b["dir"] not in set(removed)]
    return removed


@app.command(name="run")
def run_backup(
    all_projects: bool = typer.Option(True, "--all-projects/--active-only",
        help="Back up all projects (default) or just the active one."),
    no_system: bool = typer.Option(False, "--no-system",
        help="Skip the system config bundle (only back up project data)."),
):
    """Create a backup snapshot now.

    Produces one ``.skproj.tar`` per project + an optional system bundle
    under ``archives/backups/<timestamp>/``. Old backups beyond the
    retention count (``BACKUP_RETAIN_COUNT``, default 7) are pruned.
    """
    import fcntl
    import time

    from sciknow.cli import preflight
    preflight()

    from sciknow.config import settings
    from sciknow.core.project import (
        Project, get_active_project, list_projects,
    )

    backup_dir = _backup_root()
    backup_dir.mkdir(parents=True, exist_ok=True)
    lock_path = backup_dir / _LOCK_FILE
    lock_path.touch()
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        console.print("[yellow]Another backup is already running.[/yellow]")
        raise typer.Exit(1)

    try:
        t0 = time.monotonic()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = backup_dir / ts
        run_dir.mkdir(parents=True, exist_ok=True)

        if all_projects:
            projects = list_projects()
            active = get_active_project()
            if not active.is_default and not any(p.slug == active.slug for p in projects):
                projects.append(active)
            if active.is_default:
                projects.append(active)
        else:
            projects = [get_active_project()]

        files: dict[str, int] = {}
        slugs: list[str] = []
        for proj in projects:
            console.print(f"[bold]Backing up:[/bold] {proj.slug}")
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    staging = Path(tmp) / "skproj"
                    _stage_project(proj, staging)
                    tar_name = f"{proj.slug}.skproj.tar"
                    tar_path = run_dir / tar_name
                    with tarfile.open(tar_path, "w") as tf:
                        tf.add(staging, arcname="skproj")
                    files[tar_name] = tar_path.stat().st_size
                    slugs.append(proj.slug)
                console.print(f"  [green]✓[/green] {proj.slug} "
                              f"({files[tar_name] / 1024 / 1024:.1f} MB)")
            except Exception as exc:
                console.print(f"  [red]✗[/red] {proj.slug}: {exc}")

        sys_bundle = False
        if not no_system and settings.backup_include_code:
            console.print("[bold]Building system bundle…[/bold]")
            try:
                sys_name = "sciknow-system.tar.gz"
                _build_system_bundle(run_dir / sys_name)
                files[sys_name] = (run_dir / sys_name).stat().st_size
                sys_bundle = True
                console.print(f"  [green]✓[/green] system bundle "
                              f"({files[sys_name] / 1024 / 1024:.1f} MB)")
            except Exception as exc:
                console.print(f"  [red]✗[/red] system bundle: {exc}")

        elapsed = time.monotonic() - t0
        total = sum(files.values())

        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(_backup_root().parent.parent),
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            git_commit = None

        state = _read_state()
        state["backups"].append({
            "timestamp": ts,
            "dir": ts,
            "projects": slugs,
            "system_bundle": sys_bundle,
            "files": files,
            "total_bytes": total,
            "git_commit": git_commit,
            "duration_seconds": round(elapsed, 1),
        })

        pruned = _prune_old_backups(
            state,
            settings.backup_retain_count,
            retain_days=settings.backup_retain_days,
        )
        _write_state(state)

        console.print(
            f"\n[green]✓ Backup complete:[/green] {len(slugs)} project(s), "
            f"{total / 1024 / 1024:.1f} MB total, {elapsed:.1f}s"
        )
        if pruned:
            console.print(f"[dim]Pruned {len(pruned)} old backup(s).[/dim]")

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


@app.command(name="list")
def list_backups():
    """Show backup history."""
    state = _read_state()
    backups = state.get("backups", [])
    root = _backup_root()
    console.print(f"[bold]Location:[/bold] {root}")
    if not backups:
        console.print("[dim]No backups yet. Run `sciknow backup run`.[/dim]")
        return

    table = Table(title="Backup History")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Projects")
    table.add_column("Size", justify="right")
    table.add_column("System", justify="center")
    table.add_column("Duration", justify="right")

    for b in reversed(backups):
        mb = b.get("total_bytes", 0) / 1024 / 1024
        table.add_row(
            b.get("timestamp", "?"),
            ", ".join(b.get("projects", [])),
            f"{mb:.1f} MB",
            "✓" if b.get("system_bundle") else "—",
            f"{b.get('duration_seconds', 0):.0f}s",
        )
    console.print(table)
    total_mb = sum(b.get("total_bytes", 0) for b in backups) / 1024 / 1024
    console.print(f"[dim]{len(backups)} backup(s), {total_mb:.1f} MB total.[/dim]")


@app.command(name="status")
def status():
    """Show backup schedule status and last backup info."""
    state = _read_state()
    backups = state.get("backups", [])
    sched = state.get("schedule")

    if backups:
        last = backups[-1]
        console.print(f"[bold]Last backup:[/bold] {last['timestamp']}  "
                      f"({last.get('total_bytes', 0) / 1024 / 1024:.1f} MB, "
                      f"{', '.join(last.get('projects', []))})")
    else:
        console.print("[yellow]No backups yet.[/yellow]")

    if sched:
        human = sched.get("human") or sched.get("cron_expression", "?")
        console.print(f"[bold]Schedule:[/bold] {human}  "
                      f"[dim](cron: {sched.get('cron_expression', '?')}, "
                      f"installed {sched.get('installed_at', '?')})[/dim]")
    else:
        console.print("[dim]No schedule. Run `sciknow backup schedule`.[/dim]")

    from sciknow.config import settings
    ret_days = settings.backup_retain_days
    days_part = f" + older-than-{ret_days}d" if ret_days > 0 else ""
    console.print(
        f"[bold]Retention:[/bold] keep last {settings.backup_retain_count}{days_part}"
    )
    console.print(f"[bold]Location:[/bold] {_backup_root()}")


@app.command(name="schedule")
def schedule(
    frequency: str = typer.Option("daily", "--frequency", "-f",
        help="How often to run: hourly, daily, weekly, or a raw cron "
             "expression (5 fields, e.g. '*/30 * * * *'). Default: daily."),
    hour: int = typer.Option(3, "--hour",
        help="Hour of day (0-23). Used for daily + weekly. Default: 03."),
    minute: int = typer.Option(0, "--minute",
        help="Minute of hour (0-59). Default: 00."),
    weekday: int = typer.Option(0, "--weekday",
        help="Day of week for weekly (0=Sun…6=Sat). Default: 0 (Sunday)."),
):
    """Install a crontab entry for auto-backup.

    Examples:

      sciknow backup schedule                            # daily at 03:00
      sciknow backup schedule --hour 2 --minute 30       # daily at 02:30
      sciknow backup schedule --frequency hourly         # every hour at :00
      sciknow backup schedule --frequency weekly --weekday 0 --hour 4
      sciknow backup schedule --frequency "*/30 * * * *" # raw cron expr
    """
    from sciknow.core.project import _repo_root
    root = _repo_root()
    venv_bin = root / ".venv" / "bin" / "sciknow"
    if not venv_bin.exists():
        console.print(f"[red]sciknow binary not found at {venv_bin}[/red]")
        raise typer.Exit(1)

    freq = (frequency or "").strip().lower()
    if not (0 <= hour <= 23):
        console.print("[red]--hour must be 0-23[/red]"); raise typer.Exit(1)
    if not (0 <= minute <= 59):
        console.print("[red]--minute must be 0-59[/red]"); raise typer.Exit(1)
    if not (0 <= weekday <= 6):
        console.print("[red]--weekday must be 0-6[/red]"); raise typer.Exit(1)

    if freq == "hourly":
        cron_expr = f"{minute} * * * *"
        human = f"every hour at :{minute:02d}"
    elif freq == "daily":
        cron_expr = f"{minute} {hour} * * *"
        human = f"daily at {hour:02d}:{minute:02d}"
    elif freq == "weekly":
        cron_expr = f"{minute} {hour} * * {weekday}"
        names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        human = f"weekly on {names[weekday]} at {hour:02d}:{minute:02d}"
    elif freq and len(freq.split()) == 5:
        cron_expr = freq
        human = f"cron '{cron_expr}'"
    else:
        console.print(
            f"[red]Unknown frequency '{frequency}'.[/red] "
            "Use hourly, daily, weekly, or a 5-field cron expression."
        )
        raise typer.Exit(1)

    cron_line = (
        f"{cron_expr} cd {root} && {venv_bin} backup run --all-projects "
        f">> {_backup_root()}/cron.log 2>&1 {_CRON_MARKER}"
    )

    try:
        existing = subprocess.check_output(
            ["crontab", "-l"], text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        existing = ""

    lines = [l for l in existing.splitlines() if _CRON_MARKER not in l]
    lines.append(cron_line)
    new_crontab = "\n".join(lines) + "\n"

    subprocess.run(
        ["crontab", "-"], input=new_crontab, text=True, check=True,
    )

    state = _read_state()
    state["schedule"] = {
        "cron_expression": cron_expr,
        "frequency": freq,
        "human": human,
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_state(state)

    console.print(f"[green]✓ Backup scheduled {human} (local time).[/green]")
    console.print(f"[dim]Cron line: {cron_line}[/dim]")


@app.command(name="delete")
def delete_backup(
    timestamp: str = typer.Argument(
        ...,
        help="Backup timestamp/dir to delete (e.g. '20260415T030000Z'). "
             "Use 'latest' for the most recent.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y",
        help="Skip confirmation prompt."),
):
    """Delete a single backup set (both the files on disk and the state entry)."""
    state = _read_state()
    backups = state.get("backups", [])
    if not backups:
        console.print("[yellow]No backups to delete.[/yellow]")
        raise typer.Exit(0)

    if timestamp == "latest":
        entry = backups[-1]
    else:
        entry = next(
            (b for b in backups
             if b.get("dir") == timestamp or b.get("timestamp") == timestamp),
            None,
        )
    if entry is None:
        console.print(f"[red]Backup '{timestamp}' not found.[/red]")
        console.print(f"[dim]Available: {', '.join(b['dir'] for b in backups)}[/dim]")
        raise typer.Exit(1)

    mb = entry.get("total_bytes", 0) / 1024 / 1024
    projs = ", ".join(entry.get("projects", [])) or "—"
    path = _backup_root() / entry["dir"]
    console.print(f"[bold]About to delete:[/bold] {entry['dir']}  "
                  f"({mb:.1f} MB, projects: {projs})")
    console.print(f"[dim]Path: {path}[/dim]")

    if not yes:
        if not typer.confirm("Delete this backup?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    state["backups"] = [b for b in backups if b.get("dir") != entry["dir"]]
    _write_state(state)
    console.print(f"[green]✓ Deleted {entry['dir']}[/green]")


@app.command(name="purge")
def purge_backups(
    all_flag: bool = typer.Option(False, "--all",
        help="Delete ALL backup sets (the 'autodelete everything' action)."),
    older_than_days: int = typer.Option(0, "--older-than-days",
        help="Delete backup sets older than this many days."),
    yes: bool = typer.Option(False, "--yes", "-y",
        help="Skip confirmation prompt."),
):
    """Bulk-delete backups by age — or with ``--all`` wipe them all.

    Examples:

      sciknow backup purge --older-than-days 30   # drop backups older than 30d
      sciknow backup purge --all -y               # nuke everything without prompt
    """
    if not all_flag and older_than_days <= 0:
        console.print(
            "[red]Pass --all or --older-than-days N to select what to purge.[/red]"
        )
        raise typer.Exit(1)

    state = _read_state()
    backups = state.get("backups", [])
    if not backups:
        console.print("[yellow]No backups to purge.[/yellow]")
        raise typer.Exit(0)

    if all_flag:
        victims = list(backups)
        reason = "ALL backups"
    else:
        cutoff = datetime.now(timezone.utc).timestamp() - older_than_days * 86400
        victims = []
        for entry in backups:
            ts = _parse_ts(entry.get("timestamp", ""))
            if ts is not None and ts.timestamp() < cutoff:
                victims.append(entry)
        reason = f"backups older than {older_than_days}d"

    if not victims:
        console.print(f"[dim]Nothing matches ({reason}).[/dim]")
        raise typer.Exit(0)

    total_mb = sum(v.get("total_bytes", 0) for v in victims) / 1024 / 1024
    console.print(
        f"[bold]About to delete {len(victims)} backup(s) ({reason}) "
        f"— {total_mb:.1f} MB total.[/bold]"
    )
    for v in victims:
        console.print(f"  [dim]- {v['dir']}  ({v.get('total_bytes', 0)/1024/1024:.1f} MB)[/dim]")

    if not yes:
        if not typer.confirm("Proceed?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

    vset = {v["dir"] for v in victims}
    for entry in victims:
        p = _backup_root() / entry["dir"]
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    state["backups"] = [b for b in backups if b["dir"] not in vset]
    _write_state(state)
    console.print(f"[green]✓ Purged {len(victims)} backup(s).[/green]")


@app.command(name="unschedule")
def unschedule():
    """Remove the sciknow backup crontab entry."""
    try:
        existing = subprocess.check_output(
            ["crontab", "-l"], text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        console.print("[dim]No crontab entries.[/dim]")
        return

    lines = [l for l in existing.splitlines() if _CRON_MARKER not in l]
    new_crontab = "\n".join(lines) + "\n" if lines else ""

    subprocess.run(
        ["crontab", "-"], input=new_crontab, text=True, check=True,
    )

    state = _read_state()
    state["schedule"] = None
    _write_state(state)

    console.print("[green]✓ Backup schedule removed.[/green]")


@app.command(name="restore")
def restore(
    timestamp: str = typer.Argument(
        "latest",
        help="Backup timestamp to restore (e.g. '20260415T030000Z'). "
             "Use 'latest' to pick the most recent backup.",
    ),
    force: bool = typer.Option(False, "--force", "-f",
        help="Destroy existing projects before restoring (use with care)."),
    no_system: bool = typer.Option(False, "--no-system",
        help="Skip restoring the system bundle (only restore project data)."),
    project_slug: str = typer.Option("", "--project", "-p",
        help="Restore a single project from the backup set (default: all)."),
):
    """Restore projects (and optionally system config) from a backup set.

    By default restores ALL projects in the backup. Use ``--project``
    to restore just one. Refuses to overwrite existing projects unless
    ``--force`` is passed (which destroys the existing project first).

    Examples:

      sciknow backup restore                    # restore latest, all projects
      sciknow backup restore --force            # destroy + restore latest
      sciknow backup restore 20260415T030000Z   # restore a specific backup
      sciknow backup restore --project my-proj  # restore just one project
    """
    state = _read_state()
    backups = state.get("backups", [])
    if not backups:
        console.print("[red]No backups available.[/red]")
        raise typer.Exit(1)

    if timestamp == "latest":
        entry = backups[-1]
    else:
        entry = next((b for b in backups if b["dir"] == timestamp
                       or b.get("timestamp") == timestamp), None)
        if entry is None:
            console.print(f"[red]Backup '{timestamp}' not found.[/red]")
            console.print(f"[dim]Available: {', '.join(b['dir'] for b in backups)}[/dim]")
            raise typer.Exit(1)

    run_dir = _backup_root() / entry["dir"]
    if not run_dir.exists():
        console.print(f"[red]Backup directory missing: {run_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Restoring from backup:[/bold] {entry['dir']}")

    # Determine which project tars to restore
    tar_files = sorted(run_dir.glob("*.skproj.tar"))
    if project_slug:
        tar_files = [t for t in tar_files if t.stem == project_slug]
        if not tar_files:
            console.print(f"[red]No archive for project '{project_slug}' in this backup.[/red]")
            raise typer.Exit(1)

    from sciknow.cli.project import (
        _create_pg_database, _drop_pg_database, _ensure_init_dirs,
        _delete_qdrant_collections_for_project, _pg_database_exists,
    )
    from sciknow.core.project import (
        Project, get_active_project, validate_slug, write_active_slug,
    )

    restored: list[str] = []
    for tar_path in tar_files:
        console.print(f"\n[bold]Restoring:[/bold] {tar_path.name}")

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            with tarfile.open(tar_path, "r") as tf:
                tf.extractall(tmp, filter="data")
            staging = tmp / "skproj"
            if not (staging / "manifest.txt").exists():
                console.print(f"  [red]✗ Invalid archive (no manifest)[/red]")
                continue

            manifest: dict[str, str] = {}
            for line in (staging / "manifest.txt").read_text().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    manifest[k.strip()] = v.strip()
            slug = manifest.get("slug", "")
            try:
                validate_slug(slug)
            except ValueError as exc:
                console.print(f"  [red]✗ Bad slug: {exc}[/red]")
                continue

            project = Project(slug=slug, repo_root=get_active_project().repo_root)

            if project.root.exists() or _pg_database_exists(project.pg_database):
                if not force:
                    console.print(
                        f"  [yellow]⚠ Project {slug} already exists. "
                        f"Pass --force to destroy and replace.[/yellow]"
                    )
                    continue
                console.print(f"  [dim]Destroying existing {slug}…[/dim]")
                try:
                    _drop_pg_database(project.pg_database)
                except Exception:
                    pass
                try:
                    _delete_qdrant_collections_for_project(project)
                except Exception:
                    pass
                if project.root.exists():
                    shutil.rmtree(project.root)

            # PG restore
            try:
                _create_pg_database(project.pg_database)
                from sciknow.config import settings
                env = os.environ.copy()
                env["PGPASSWORD"] = settings.pg_password
                res = subprocess.run([
                    "pg_restore",
                    "-h", settings.pg_host,
                    "-p", str(settings.pg_port),
                    "-U", settings.pg_user,
                    "-d", project.pg_database,
                    str(staging / "postgres.dump"),
                ], env=env, capture_output=True, text=True)
                if res.returncode != 0:
                    console.print(f"  [red]pg_restore failed:[/red] {res.stderr[:300]}")
                    continue
                console.print(f"  [green]✓[/green] PG restored")
            except Exception as exc:
                console.print(f"  [red]✗ PG restore failed: {exc}[/red]")
                continue

            # Data directory
            _ensure_init_dirs(project)
            data_src = staging / "data"
            if data_src.exists():
                for child in data_src.iterdir():
                    dest = project.data_dir / child.name
                    if dest.exists():
                        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                    shutil.move(str(child), str(dest))
            console.print(f"  [green]✓[/green] data dir restored")

            # Env overlay
            env_src = staging / ".env.overlay"
            if env_src.exists():
                shutil.copy2(env_src, project.env_overlay_path)

            # Qdrant note
            snaps_dir = staging / "qdrant_snapshots"
            if snaps_dir.exists() and any(snaps_dir.iterdir()):
                console.print(
                    "  [yellow]![/yellow] Qdrant vectors need rebuilding: "
                    "run `sciknow db init` then "
                    "`sciknow ingest directory <data>/processed/`"
                )

            restored.append(slug)

    # System bundle
    sys_bundle = run_dir / "sciknow-system.tar.gz"
    if sys_bundle.exists() and not no_system:
        console.print(f"\n[bold]Restoring system bundle…[/bold]")
        from sciknow.core.project import _repo_root
        root = _repo_root()
        with tempfile.TemporaryDirectory() as tmp_str:
            with tarfile.open(sys_bundle, "r:gz") as tf:
                tf.extractall(Path(tmp_str), filter="data")
            sys_stage = Path(tmp_str) / "sciknow-system"
            for name in (".env", ".env.example", "pyproject.toml", "uv.lock",
                         "alembic.ini", ".active-project"):
                src = sys_stage / name
                if src.exists():
                    shutil.copy2(src, root / name)
            mig_src = sys_stage / "migrations"
            if mig_src.exists():
                mig_dst = root / "migrations"
                if mig_dst.exists():
                    shutil.rmtree(mig_dst)
                shutil.copytree(mig_src, mig_dst)
        console.print(f"  [green]✓[/green] .env, pyproject.toml, uv.lock, "
                      f"alembic.ini, migrations/ restored")
        console.print(f"  [dim]Run `uv sync` to reinstall deps if pyproject.toml changed.[/dim]")

    if restored:
        write_active_slug(restored[0])
        console.print(
            f"\n[green]✓ Restored {len(restored)} project(s): "
            f"{', '.join(restored)}[/green]"
        )
        console.print(
            f"[dim]Active project set to: {restored[0]}. "
            f"Run `sciknow db init` to rebuild Qdrant collections.[/dim]"
        )
    else:
        console.print("\n[yellow]No projects were restored.[/yellow]")
