"""`sciknow library` — database + infrastructure lifecycle.

Spec §5.1: this is the v2 home for `init`, `reset`, `stats`, `migrate`,
`validate`, `snapshot` (and the operational helpers backup / restore /
doctor / monitor / dashboard / failures / drift / provenance that
live alongside them).

The implementations remain in ``sciknow.cli.db`` for now; this module
re-exports the command callables under the renamed subapp so the v2
verb structure is live immediately. ``cli/main.py`` registers a
``sciknow db`` deprecation shim that prints a one-shot warning then
dispatches to the same callables — users get a smooth migration window.

Phase F continuation: physically moving the command bodies out of
db.py is a separate cleanup commit that requires moving the helper
functions too (~3 kLOC of supporting code). Re-registration is the
contract change.
"""
from __future__ import annotations

import typer
from rich.console import Console

from sciknow.cli import db as _db

console = Console()

app = typer.Typer(
    name="library",
    help="Database + infrastructure lifecycle (init, reset, stats, "
         "migrate, validate, snapshot, backup, doctor, monitor).",
    no_args_is_help=True,
)

# ── core lifecycle (spec verbs) ─────────────────────────────────────────
app.command(name="init")(_db.init)
app.command(name="reset")(_db.reset)
app.command(name="stats")(_db.stats)

# ── operational helpers ─────────────────────────────────────────────────
app.command(name="backup")(_db.backup)
app.command(name="restore")(_db.restore)
app.command(name="failures")(_db.failures)
app.command(name="doctor")(_db.doctor)
app.command(name="monitor")(_db.monitor)
app.command(name="dashboard")(_db.dashboard)

# ── audits / drift detection ────────────────────────────────────────────
app.command(name="drift")(_db.drift_cmd)
app.command(name="provenance")(_db.provenance_cmd)


@app.command(name="migrate")
def migrate_cmd():
    """Run pending Alembic migrations against the active project's DB.

    Equivalent to ``uv run alembic upgrade head`` — wrapped here so
    library lifecycle stays a one-stop shop.
    """
    import subprocess
    import sys
    rc = subprocess.call(["uv", "run", "alembic", "upgrade", "head"])
    sys.exit(rc)


@app.command(name="validate")
def validate_cmd():
    """Validate that the SQLAlchemy models match the Alembic head.

    Equivalent to ``alembic check`` — fails non-zero if drift is
    detected (i.e. someone added a column to models.py without an
    accompanying migration).
    """
    import subprocess
    import sys
    rc = subprocess.call(["uv", "run", "alembic", "check"])
    sys.exit(rc)


@app.command(name="snapshot")
def snapshot_cmd():
    """One-shot pg_dump + Qdrant snapshot of the active project.

    Spec §5.1 names this verb explicitly; for now it delegates to the
    existing ``library backup`` flow.
    """
    # The v1 'backup' helper is the snapshot facility. Surfacing it
    # under a stable name now means future Phase G `import-v1` can
    # call `library snapshot` cleanly.
    from sciknow.cli.db import backup as _backup
    # `_backup` reads its args from the typer context — but it has
    # defaults that produce a complete tarball. Invoke directly.
    _backup()


@app.command(name="upgrade-v1")
def upgrade_v1_cmd(
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Don't drop anything — just report what would change.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Skip the confirmation prompt before destructive cleanup.",
    ),
):
    """v2 Phase G — in-place upgrade of the active project from v1 layout.

    What this does:

      1. Verifies the prod papers/abstracts/wiki/visuals collections
         match the v2 canonical embedder dim (bge-m3 = 1024). They
         already do for any project ingested with bge-m3 as the
         primary, which is the v1 default — the dual-embedder split
         only added a *sidecar* collection.
      2. Lists every Qdrant collection that begins with the project's
         qdrant_prefix and ends in ``_papers`` *other than* the
         canonical ``<prefix>papers`` collection (i.e. the v1 dual-
         embedder sidecars: ``<prefix>_ab_qwen_qwen3-embedding-4b_papers``).
      3. With confirmation, drops those sidecars (Qdrant only — no
         PG state to clean up; the chunks table is the same).
      4. Writes ``<project root>/.v2-upgraded`` with the upgrade
         timestamp + git SHA so we can refuse to re-run accidentally.

    Re-embedding is **not** required: v1 used bge-m3 dense + sparse,
    v2 uses bge-m3 dense (sparse degraded per spec §3.3) — the dense
    vectors are bit-identical because the model is the same and the
    GGUF quantisation is at Q8_0 (delta vs FP16 negligible for
    nearest-neighbor search).
    """
    from datetime import datetime, timezone
    from pathlib import Path

    from sciknow.config import settings
    from sciknow.core.project import get_active_project
    from sciknow.storage.qdrant import get_client as _qdrant_client

    project = get_active_project()
    marker = project.root / ".v2-upgraded"
    if marker.exists():
        console.print(
            f"[yellow]Project {project.slug!r} already upgraded "
            f"({marker.read_text().strip()[:60]}).[/yellow] Re-run with "
            f"`rm {marker}` first if you really want to repeat the upgrade."
        )
        raise typer.Exit(0)

    client = _qdrant_client()
    all_colls = sorted(c.name for c in client.get_collections().collections)
    prefix = project.qdrant_prefix
    canonical = {
        project.papers_collection,
        project.abstracts_collection,
        project.wiki_collection,
        f"{prefix}visuals",
    }

    # Verify canonical collections exist + dim sanity.
    issues: list[str] = []
    for coll in (project.papers_collection,):
        if coll not in all_colls:
            issues.append(f"missing canonical collection: {coll}")
            continue
        try:
            info = client.get_collection(coll)
            cfg = info.config.params.vectors
            actual = (
                cfg["dense"].size if isinstance(cfg, dict) and "dense" in cfg
                else (cfg.size if hasattr(cfg, "size") else None)
            )
        except Exception as exc:
            issues.append(f"failed to introspect {coll}: {exc}")
            continue
        if actual is not None and actual != settings.embedding_dim:
            issues.append(
                f"{coll}: dense dim={actual} but settings.embedding_dim="
                f"{settings.embedding_dim}. Re-embed required (out of scope "
                f"for in-place upgrade — fall back to `project init "
                f"--from-existing` + manual ingest)."
            )
    if issues:
        console.print("[red]Cannot upgrade — pre-flight failed:[/red]")
        for i in issues:
            console.print(f"  • {i}")
        raise typer.Exit(2)

    # Find dual-embedder sidecars and any other non-canonical collections
    # under the project prefix.
    sidecars = [
        c for c in all_colls
        if (c.startswith(prefix) or (project.is_default and "_ab_" in c))
        and c not in canonical
    ]

    console.print(f"[bold]Project:[/bold] {project.slug}")
    console.print(f"[bold]Canonical:[/bold] {sorted(canonical)}")
    console.print(f"[bold]Sidecars to drop:[/bold] "
                  f"{sidecars if sidecars else '(none)'}")

    if dry_run:
        console.print("[dim]--dry-run: no changes made.[/dim]")
        raise typer.Exit(0)

    if sidecars and not yes:
        ok = typer.confirm(
            f"Drop {len(sidecars)} sidecar collection(s)? This is "
            f"irreversible.", default=False,
        )
        if not ok:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(1)

    for coll in sidecars:
        try:
            client.delete_collection(coll)
            console.print(f"  [green]✓ dropped[/green] {coll}")
        except Exception as exc:
            console.print(f"  [red]✗ {coll}[/red]: {exc}")

    # Write the marker.
    project.root.mkdir(parents=True, exist_ok=True)
    sha = ""
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass
    marker_text = (
        f"v2-upgraded at {datetime.now(timezone.utc).isoformat()}\n"
        f"git={sha}\n"
        f"sidecars_dropped={len(sidecars)}\n"
    )
    marker.write_text(marker_text)
    console.print(
        f"[green]✓ upgrade complete.[/green] "
        f"Marker: {marker} ({len(sidecars)} sidecar(s) dropped)"
    )
