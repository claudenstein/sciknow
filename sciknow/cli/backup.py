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


def _prune_old_backups(state: dict, retain: int) -> list[str]:
    """Remove the oldest backup sets beyond the retention count.
    Returns list of removed dir names."""
    backups = state.get("backups", [])
    if len(backups) <= retain:
        return []
    to_remove = backups[:-retain]
    removed: list[str] = []
    for entry in to_remove:
        d = _backup_root() / entry["dir"]
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        removed.append(entry["dir"])
    state["backups"] = backups[-retain:]
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

        pruned = _prune_old_backups(state, settings.backup_retain_count)
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
        console.print(f"[bold]Schedule:[/bold] {sched.get('cron_expression', '?')}  "
                      f"(installed {sched.get('installed_at', '?')})")
    else:
        console.print("[dim]No schedule. Run `sciknow backup schedule`.[/dim]")

    from sciknow.config import settings
    console.print(f"[dim]Retention: {settings.backup_retain_count} backups  "
                  f"Dir: {_backup_root()}[/dim]")


@app.command(name="schedule")
def schedule(
    hour: int = typer.Option(3, "--hour",
        help="Hour of day (0-23) for the daily backup. Default: 03:00."),
):
    """Install a daily crontab entry for auto-backup."""
    from sciknow.core.project import _repo_root
    root = _repo_root()
    venv_bin = root / ".venv" / "bin" / "sciknow"
    if not venv_bin.exists():
        console.print(f"[red]sciknow binary not found at {venv_bin}[/red]")
        raise typer.Exit(1)

    cron_line = (
        f"{0} {hour} * * * "
        f"cd {root} && {venv_bin} backup run --all-projects "
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
        "cron_expression": f"0 {hour} * * *",
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_state(state)

    console.print(f"[green]✓ Daily backup scheduled at {hour:02d}:00 local time.[/green]")
    console.print(f"[dim]Cron line: {cron_line}[/dim]")


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
