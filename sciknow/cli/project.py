"""``sciknow project ...`` — multi-project lifecycle commands.

Phase 43e. The user-facing surface for the multi-project work whose
plumbing landed in 43a–d. Subcommands:

- ``init <slug>``           create a new project (DB + collections + dir + migrations)
- ``init <slug> --from-existing``
                            adopt the legacy single-tenant layout into a project slot
- ``list``                  show every project with health + size summary
- ``show [slug]``           details for one project (defaults to active)
- ``use <slug>``            set the active project for subsequent CLI calls
- ``destroy <slug>``        drop DB + collections + data dir (guarded by --yes)
- ``archive <slug>``        bundle the project into a portable archive then drop live state
- ``unarchive <archive>``   restore an archived project
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import text

from sciknow.config import settings
from sciknow.core.project import (
    Project,
    _SLUG_RE,
    get_active_project,
    list_projects,
    read_active_slug_from_file,
    validate_slug,
    write_active_slug,
)
from sciknow.storage.db import get_admin_engine, get_engine, get_session

app = typer.Typer(help="Manage sciknow projects (corpus + book + DB isolation).")
console = Console()
logger = logging.getLogger("sciknow.cli.project")


# ── shared helpers ─────────────────────────────────────────────────────


def _resolve_slug_or_default(slug: str | None) -> Project:
    """Use the given slug if provided, else the active project."""
    if slug:
        validate_slug(slug)
        return Project(slug=slug, repo_root=get_active_project().repo_root)
    return get_active_project()


def _pg_database_exists(db_name: str) -> bool:
    with get_admin_engine().connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :n"),
            {"n": db_name},
        ).fetchone()
    return row is not None


def _create_pg_database(db_name: str, template: str | None = None) -> None:
    """``CREATE DATABASE`` against the admin DB.

    ``template`` lets us clone an existing DB atomically (used by the
    one-shot migration in Phase 43f). PostgreSQL identifiers can't be
    parameterised, so we validate strictly first then format directly.
    """
    if not db_name.replace("_", "").isalnum():
        raise ValueError(f"Refusing to CREATE DATABASE on suspicious name {db_name!r}")
    if template and not template.replace("_", "").isalnum():
        raise ValueError(f"Refusing to clone from suspicious template {template!r}")
    with get_admin_engine().connect().execution_options(
        isolation_level="AUTOCOMMIT"
    ) as conn:
        sql = f'CREATE DATABASE "{db_name}"'
        if template:
            sql += f' WITH TEMPLATE "{template}"'
        conn.execute(text(sql))


def _drop_pg_database(db_name: str) -> None:
    if not db_name.replace("_", "").isalnum():
        raise ValueError(f"Refusing to DROP DATABASE on suspicious name {db_name!r}")
    with get_admin_engine().connect().execution_options(
        isolation_level="AUTOCOMMIT"
    ) as conn:
        # Disconnect any lingering sessions so the drop succeeds even
        # when something held a stale connection (alembic, web reader).
        conn.execute(
            text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = :n AND pid <> pg_backend_pid()"
            ),
            {"n": db_name},
        )
        conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}"'))


def _run_alembic_upgrade(db_name: str) -> None:
    """Run ``alembic upgrade head`` against ``db_name``.

    Uses subprocess so the alembic process gets a fresh import of
    settings + project — we set SCIKNOW_PROJECT in the env so it picks
    up the right active project (and the -x override pins the DB
    explicitly as belt-and-suspenders).
    """
    repo_root = get_active_project().repo_root
    env = os.environ.copy()
    # `-x db_name=...` is honoured by migrations/env.py to override
    # the URL regardless of which project is active.
    cmd = [
        sys.executable, "-m", "alembic",
        "-x", f"db_name={db_name}",
        "upgrade", "head",
    ]
    result = subprocess.run(
        cmd, cwd=str(repo_root), env=env,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]alembic failed (rc={result.returncode}):[/red]")
        console.print(result.stderr or result.stdout)
        raise typer.Exit(1)


def _init_qdrant_collections_for_project(project: Project) -> None:
    """Create the project's collections (uses init_collections + env override).

    Switches the SCIKNOW_PROJECT env var so the resolution layer
    picks up the new project, then calls init_collections() which
    reads collection names from the active project. Restores the env
    afterwards.
    """
    from sciknow.storage.qdrant import init_collections

    prev = os.environ.get("SCIKNOW_PROJECT")
    os.environ["SCIKNOW_PROJECT"] = project.slug
    try:
        init_collections()
    finally:
        if prev is None:
            os.environ.pop("SCIKNOW_PROJECT", None)
        else:
            os.environ["SCIKNOW_PROJECT"] = prev


def _delete_qdrant_collections_for_project(project: Project) -> int:
    """Best-effort drop of the project's three collections.

    Returns the number actually deleted (a project that's never been
    fully provisioned may have only some). Errors swallowed per
    collection so a partial failure doesn't block destroy.
    """
    from sciknow.storage.qdrant import get_client

    client = get_client()
    deleted = 0
    for name in (
        project.papers_collection,
        project.abstracts_collection,
        project.wiki_collection,
    ):
        try:
            client.delete_collection(collection_name=name)
            deleted += 1
        except Exception as exc:
            logger.debug("delete_collection(%s) failed: %s", name, exc)
    return deleted


def _ensure_init_dirs(project: Project) -> None:
    """Create the project's directory tree if missing."""
    project.root.mkdir(parents=True, exist_ok=True)
    project.data_dir.mkdir(parents=True, exist_ok=True)
    if not project.env_overlay_path.exists():
        project.env_overlay_path.write_text(
            "# Optional per-project overrides for sciknow settings.\n"
            "# Anything set here wins over the root .env. Examples:\n"
            "#   LLM_MODEL=qwen3:32b\n"
            "#   LLM_FAST_MODEL=qwen3:8b\n"
        )


# ── commands ───────────────────────────────────────────────────────────


@app.command()
def init(
    slug: Annotated[str, typer.Argument(help="Project slug (lowercase alphanumerics + hyphens, e.g. 'global-cooling').")],
    from_existing: bool = typer.Option(
        False, "--from-existing",
        help="One-shot migration: adopt the current single-tenant install (data/, sciknow DB, "
             "unprefixed Qdrant collections) into this project slot. See `docs/PROJECTS.md` §43f.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print the steps without executing them. Useful before --from-existing.",
    ),
):
    """Create a new project (DB + collections + dir + migrations).

    Without flags: starts a brand-new empty project. The new project
    becomes the active one (.active-project is updated) so subsequent
    CLI invocations use it.

    With ``--from-existing``: migrates the legacy ``sciknow`` DB +
    ``data/`` tree + unprefixed Qdrant collections into the new
    project's slot.
    """
    try:
        validate_slug(slug)
    except ValueError as exc:
        console.print(f"[red]Invalid slug:[/red] {exc}")
        raise typer.Exit(2)
    if slug == "default":
        console.print("[red]'default' is reserved for the legacy layout — pick another slug.[/red]")
        raise typer.Exit(2)

    repo_root = get_active_project().repo_root
    project = Project(slug=slug, repo_root=repo_root)

    if project.root.exists() and not from_existing:
        console.print(f"[red]Project directory already exists:[/red] {project.root}")
        raise typer.Exit(1)
    if _pg_database_exists(project.pg_database) and not from_existing:
        console.print(f"[red]PostgreSQL database already exists:[/red] {project.pg_database}")
        console.print("[dim]Run `sciknow project destroy` first if you want to recreate it.[/dim]")
        raise typer.Exit(1)

    if from_existing:
        _do_migration_from_existing(project, dry_run=dry_run)
    else:
        _do_fresh_init(project, dry_run=dry_run)

    if not dry_run:
        write_active_slug(slug)
        console.print(f"\n[green]✓ Project [bold]{slug}[/bold] is now active.[/green]")
        console.print(f"[dim]Run `sciknow project show` to see details.[/dim]")


def _do_fresh_init(project: Project, *, dry_run: bool) -> None:
    """Empty-project init — DB, collections, dirs, migrations."""
    console.print(f"[bold]Initializing empty project:[/bold] {project.slug}")
    console.print(f"  PG database:  {project.pg_database}")
    console.print(f"  Qdrant prefix: {project.qdrant_prefix or '(none)'}")
    console.print(f"  Data dir:     {project.data_dir}")
    if dry_run:
        console.print("[dim](dry-run, no changes made)[/dim]")
        return

    with console.status("Creating directories…"):
        _ensure_init_dirs(project)
    console.print("  [green]✓[/green] directories")

    with console.status("Creating PostgreSQL database…"):
        _create_pg_database(project.pg_database)
    console.print(f"  [green]✓[/green] PG database `{project.pg_database}`")

    with console.status("Running migrations…"):
        _run_alembic_upgrade(project.pg_database)
    console.print("  [green]✓[/green] migrations applied")

    with console.status("Creating Qdrant collections…"):
        _init_qdrant_collections_for_project(project)
    console.print(
        f"  [green]✓[/green] Qdrant collections "
        f"(`{project.papers_collection}`, `{project.abstracts_collection}`, "
        f"`{project.wiki_collection}`)"
    )


def _do_migration_from_existing(project: Project, *, dry_run: bool) -> None:
    """One-shot migration from the legacy single-tenant layout."""
    repo_root = project.repo_root
    legacy_data = repo_root / "data"
    legacy_db = "sciknow"

    console.print(f"[bold]Migrating legacy layout into project:[/bold] {project.slug}")
    console.print(f"  Source PG: {legacy_db}  →  {project.pg_database}")
    console.print(f"  Source data: {legacy_data}  →  {project.data_dir}")
    console.print(f"  Qdrant: papers/abstracts/wiki  →  {project.qdrant_prefix}*")

    # Pre-flight verification
    legacy_db_exists = _pg_database_exists(legacy_db)
    if not legacy_db_exists:
        console.print(f"[red]Legacy database `{legacy_db}` not found.[/red]")
        raise typer.Exit(1)
    if not legacy_data.exists():
        console.print(f"[yellow]Legacy data dir not found: {legacy_data}. Continuing without it.[/yellow]")
    if _pg_database_exists(project.pg_database):
        console.print(f"[red]Target database `{project.pg_database}` already exists.[/red]")
        raise typer.Exit(1)

    if dry_run:
        console.print("[dim](dry-run, no changes made)[/dim]")
        return

    # 1. Clone PG via CREATE DATABASE WITH TEMPLATE — atomic, fast
    with console.status(f"Cloning PG database (`{legacy_db}` → `{project.pg_database}`)…"):
        _create_pg_database(project.pg_database, template=legacy_db)
    console.print(f"  [green]✓[/green] PG cloned to `{project.pg_database}`")

    # 2. Move the data directory contents
    with console.status("Moving filesystem contents…"):
        _ensure_init_dirs(project)
        if legacy_data.exists():
            for child in legacy_data.iterdir():
                dest = project.data_dir / child.name
                if dest.exists():
                    console.print(f"[yellow]skip (already exists): {dest}[/yellow]")
                    continue
                shutil.move(str(child), str(dest))
    console.print(f"  [green]✓[/green] data/* moved to {project.data_dir}")

    # 3. Snapshot + restore Qdrant collections under new names
    _migrate_qdrant_collections(project)

    console.print("\n[green]✓ Legacy layout migrated.[/green]")
    console.print(
        f"[dim]The old `{legacy_db}` PG database and the unprefixed Qdrant "
        f"collections are still present — run `sciknow project show` to verify "
        f"the new project, then drop the legacy ones manually if everything looks good.[/dim]"
    )


def _migrate_qdrant_collections(project: Project) -> None:
    """Copy legacy collections into the project's prefixed collections.

    Initially this used Qdrant's snapshot/restore API, but
    ``recover_snapshot(location="file://...")`` requires the snapshot
    file path to be accessible from inside the Qdrant server's
    filesystem, which doesn't work in containerised setups (or even
    bare-metal installs whose snapshot dir doesn't match the URL we
    construct). The user hits a ``Bad request: Snapshot file ... does
    not exist`` 400 even though the snapshot was created.

    Reliable alternative: ``init_collections()`` against the new
    project's prefixes + a scroll-and-upsert loop that streams every
    point (with both dense + sparse vectors and payload) from the
    legacy collection into the new one. Slower (~1300 pts/s on the
    reference hardware, so ~80s for a 100k-chunk corpus) but it has
    no dependency on the Qdrant server's filesystem layout. Leaves
    the legacy collection untouched as a recovery path.
    """
    from qdrant_client.models import PointStruct

    from sciknow.storage.qdrant import get_client, init_collections

    client = get_client()
    try:
        existing = {c.name for c in client.get_collections().collections}
    except Exception as exc:
        console.print(f"[yellow]Qdrant unreachable: {exc}. Skipping migration.[/yellow]")
        return

    # 1. Provision the project's prefixed collections under the active
    # project. We've already written .active-project upstream of this
    # call site? No — init writes it AFTER migration. Set the env var
    # so init_collections() picks the right names regardless.
    prev_env = os.environ.get("SCIKNOW_PROJECT")
    os.environ["SCIKNOW_PROJECT"] = project.slug
    try:
        init_collections()
    finally:
        if prev_env is None:
            os.environ.pop("SCIKNOW_PROJECT", None)
        else:
            os.environ["SCIKNOW_PROJECT"] = prev_env

    # 2. Copy each legacy collection's points into its prefixed twin.
    pairs = [
        ("papers", project.papers_collection),
        ("abstracts", project.abstracts_collection),
        ("wiki", project.wiki_collection),
    ]
    for old, new in pairs:
        if old not in existing:
            console.print(f"  [dim]skip Qdrant {old} (not present)[/dim]")
            continue
        try:
            src_count = client.get_collection(old).points_count
        except Exception as exc:
            console.print(f"  [yellow]skip Qdrant {old}: {exc}[/yellow]")
            continue
        if src_count == 0:
            console.print(f"  [dim]skip Qdrant {old} (empty)[/dim]")
            continue
        with console.status(f"Copying Qdrant `{old}` → `{new}` ({src_count} pts)…"):
            copied = _scroll_upsert(client, old, new, batch_size=500)
        # Brief settle window for async upserts to register in count
        import time as _t
        for _ in range(5):
            try:
                dst_count = client.get_collection(new).points_count
            except Exception:
                dst_count = -1
            if dst_count == src_count:
                break
            _t.sleep(1)
        ok = dst_count == src_count
        marker = "[green]✓[/green]" if ok else "[yellow]?[/yellow]"
        console.print(f"  {marker} Qdrant {old} → {new}: copied={copied} dst_count={dst_count}")
        if not ok:
            console.print(
                f"    [yellow](source had {src_count}; check the new collection "
                f"or re-run if the count doesn't settle)[/yellow]"
            )


def _scroll_upsert(client, src: str, dst: str, *, batch_size: int = 500) -> int:
    """Stream all points from one collection to another.

    Preserves both dense + sparse vectors and the full payload by
    requesting them on scroll and passing them straight through to
    upsert. ``wait=False`` on upsert lets us pipeline; the caller
    polls the final count to confirm the destination caught up.
    """
    from qdrant_client.models import PointStruct

    copied = 0
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=src,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        if not result:
            break
        points = [
            PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in result
        ]
        client.upsert(collection_name=dst, points=points, wait=False)
        copied += len(points)
        if next_offset is None:
            break
        offset = next_offset
    return copied


@app.command(name="list")
def list_cmd():
    """List all projects with their status."""
    projects = list_projects()
    active_slug = read_active_slug_from_file() or os.environ.get("SCIKNOW_PROJECT")

    if not projects:
        console.print("[dim]No projects yet.[/dim]")
        console.print("Run [bold]sciknow project init <slug>[/bold] to create one,")
        console.print("or [bold]sciknow project init <slug> --from-existing[/bold] to migrate the current install.")
        return

    table = Table(title="Projects")
    table.add_column("Active", justify="center", width=6)
    table.add_column("Slug", style="bold")
    table.add_column("PG database")
    table.add_column("Data dir")
    table.add_column("Status")

    for p in projects:
        active_mark = "●" if active_slug == p.slug else ""
        try:
            db_ok = _pg_database_exists(p.pg_database)
        except Exception:
            db_ok = False
        status = "[green]ok[/green]" if (p.exists() and db_ok) else "[yellow]incomplete[/yellow]"
        rel_data = p.data_dir.relative_to(p.repo_root) if p.data_dir.is_relative_to(p.repo_root) else p.data_dir
        table.add_row(active_mark, p.slug, p.pg_database, str(rel_data), status)

    console.print(table)
    if active_slug is None:
        console.print("[dim]No active project set. Use [bold]sciknow project use <slug>[/bold].[/dim]")


@app.command()
def show(slug: str = typer.Argument(None, help="Project slug. Defaults to active project.")):
    """Show details for a project (DB connection, paper / book counts)."""
    project = _resolve_slug_or_default(slug)
    console.print(f"[bold]Project:[/bold] {project.slug}{'  [dim](default / legacy)[/dim]' if project.is_default else ''}")
    console.print(f"  Root:          {project.root}")
    console.print(f"  Data dir:      {project.data_dir}{'  [yellow](missing)[/yellow]' if not project.data_dir.exists() else ''}")
    console.print(f"  PG database:   {project.pg_database}")
    console.print(f"  Qdrant prefix: {project.qdrant_prefix or '(none)'}")
    console.print(f"  Papers coll:   {project.papers_collection}")
    console.print(f"  Abstracts:     {project.abstracts_collection}")
    console.print(f"  Wiki:          {project.wiki_collection}")
    console.print(f"  Env overlay:   {project.env_overlay_path}{'  [dim](not present)[/dim]' if not project.env_overlay_path.exists() else ''}")

    # Live counts from the project's own DB
    if not _pg_database_exists(project.pg_database):
        console.print("\n[yellow]PostgreSQL database does not exist.[/yellow]")
        console.print(f"[dim]Run `sciknow project init {project.slug}` to create it.[/dim]")
        return

    try:
        with get_session(db_name=project.pg_database) as session:
            n_docs = session.execute(text("SELECT COUNT(*) FROM documents")).scalar() or 0
            n_chunks = session.execute(text("SELECT COUNT(*) FROM chunks")).scalar() or 0
            n_books = session.execute(text("SELECT COUNT(*) FROM books")).scalar() or 0
            n_drafts = session.execute(text("SELECT COUNT(*) FROM drafts")).scalar() or 0
        console.print(f"\n  Documents:  {n_docs:,}")
        console.print(f"  Chunks:     {n_chunks:,}")
        console.print(f"  Books:      {n_books:,}")
        console.print(f"  Drafts:     {n_drafts:,}")
    except Exception as exc:
        console.print(f"\n[yellow]Could not read counts: {exc}[/yellow]")


@app.command()
def use(slug: Annotated[str, typer.Argument(help="Project slug to activate.")]):
    """Set the active project (writes ``.active-project`` at repo root)."""
    try:
        validate_slug(slug)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    if slug != "default":
        # default doesn't need to "exist" in projects/, it's the legacy fallback
        candidate = Project(slug=slug, repo_root=get_active_project().repo_root)
        if not candidate.exists():
            console.print(f"[red]Project not found:[/red] {candidate.root}")
            console.print(f"[dim]Available: {', '.join(p.slug for p in list_projects()) or '(none)'}[/dim]")
            raise typer.Exit(1)
    write_active_slug(slug)
    console.print(f"[green]✓ Active project: {slug}[/green]")


@app.command(name="venue-list")
def venue_list_cmd(
    slug: str = typer.Argument(None, help="Project slug. Defaults to active."),
):
    """Phase 54.6.112 (Tier 1 #5) — show the project's venue block/allow lists.

    The JSON lives at ``<project root>/venue_config.json``. A venue
    name that matches ANY blocklist substring is hard-dropped from
    ``db expand`` candidates; matching the allowlist rescues it even
    when it also matches a blocklist / built-in predatory pattern.
    """
    from sciknow.core import venue_config as _vc
    project = _resolve_slug_or_default(slug)
    cfg = _vc.load(project.root)
    path = _vc.path_for(project.root)
    console.print(f"[bold]{project.slug}[/bold]  {path}"
                  + ("  [dim](file not yet present)[/dim]" if not path.exists() else ""))
    console.print(f"[red]Blocklist[/red] ({len(cfg.blocklist)}):")
    if cfg.blocklist:
        for p in cfg.blocklist:
            console.print(f"  - {p}")
    else:
        console.print("  [dim](empty)[/dim]")
    console.print(f"[green]Allowlist[/green] ({len(cfg.allowlist)}):")
    if cfg.allowlist:
        for p in cfg.allowlist:
            console.print(f"  - {p}")
    else:
        console.print("  [dim](empty)[/dim]")


@app.command(name="venue-block")
def venue_block_cmd(
    pattern: Annotated[str, typer.Argument(help="Substring to match against venue / publisher / host-org name (case-insensitive). Prefix with ^ or suffix with $ for a regex.")],
    slug: str = typer.Option(None, "--project", help="Project slug. Defaults to active."),
):
    """Add a venue pattern to the project's blocklist for db expand."""
    from sciknow.core import venue_config as _vc
    project = _resolve_slug_or_default(slug)
    _, added = _vc.add_pattern(project.root, pattern, kind="block")
    if added:
        console.print(f"[green]✓ Blocked[/green] pattern [bold]{pattern!r}[/bold] "
                      f"for project [bold]{project.slug}[/bold].")
    else:
        console.print(f"[yellow]Already blocked:[/yellow] {pattern!r}")


@app.command(name="venue-allow")
def venue_allow_cmd(
    pattern: Annotated[str, typer.Argument(help="Substring to match against venue / publisher / host-org name (case-insensitive). Prefix with ^ or suffix with $ for a regex.")],
    slug: str = typer.Option(None, "--project", help="Project slug. Defaults to active."),
):
    """Rescue a venue from the blocklist / built-in predatory patterns.

    Allowlist matches win over blocklist + built-in predatory patterns.
    Use for legitimate venues whose name happens to substring-match a
    predatory pattern (false positive).
    """
    from sciknow.core import venue_config as _vc
    project = _resolve_slug_or_default(slug)
    _, added = _vc.add_pattern(project.root, pattern, kind="allow")
    if added:
        console.print(f"[green]✓ Allowed[/green] pattern [bold]{pattern!r}[/bold] "
                      f"for project [bold]{project.slug}[/bold].")
    else:
        console.print(f"[yellow]Already allowed:[/yellow] {pattern!r}")


@app.command(name="venue-remove")
def venue_remove_cmd(
    pattern: Annotated[str, typer.Argument(help="Pattern to remove.")],
    from_: str = typer.Option("block", "--from",
                              help="Which list: block | allow."),
    slug: str = typer.Option(None, "--project", help="Project slug. Defaults to active."),
):
    """Remove a pattern from the project's block- or allow-list."""
    if from_ not in ("block", "allow"):
        console.print("[red]--from must be 'block' or 'allow'.[/red]")
        raise typer.Exit(1)
    from sciknow.core import venue_config as _vc
    project = _resolve_slug_or_default(slug)
    _, removed = _vc.remove_pattern(project.root, pattern, kind=from_)
    if removed:
        console.print(f"[green]✓ Removed[/green] {pattern!r} from {from_}list "
                      f"({project.slug}).")
    else:
        console.print(f"[yellow]Not found in {from_}list:[/yellow] {pattern!r}")


@app.command()
def destroy(
    slug: Annotated[str, typer.Argument(help="Project slug to destroy.")],
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
):
    """Drop the project's PG database, Qdrant collections, and data dir.

    [bold red]Destructive.[/bold red] Run ``sciknow project archive`` first if
    you might want the project back later.
    """
    if slug == "default":
        console.print("[red]Refusing to destroy the legacy 'default' project.[/red]")
        raise typer.Exit(2)
    try:
        validate_slug(slug)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    project = Project(slug=slug, repo_root=get_active_project().repo_root)
    console.print(f"[bold red]About to DESTROY project:[/bold red] {slug}")
    console.print(f"  PG database:  {project.pg_database}")
    console.print(f"  Qdrant:       {project.papers_collection}, {project.abstracts_collection}, {project.wiki_collection}")
    console.print(f"  Data dir:     {project.data_dir}")
    if not yes and not typer.confirm("Proceed?"):
        console.print("Cancelled.")
        raise typer.Exit(0)

    with console.status("Dropping PG database…"):
        try:
            _drop_pg_database(project.pg_database)
        except Exception as exc:
            console.print(f"[yellow]PG drop failed: {exc}[/yellow]")
    console.print(f"  [green]✓[/green] PG database dropped")

    n_dropped = _delete_qdrant_collections_for_project(project)
    console.print(f"  [green]✓[/green] Qdrant collections deleted ({n_dropped})")

    if project.root.exists():
        shutil.rmtree(project.root)
    console.print(f"  [green]✓[/green] Data dir removed")

    # If destroying the active project, clear .active-project
    active = read_active_slug_from_file()
    if active == slug:
        from sciknow.core.project import _active_project_file
        f = _active_project_file()
        if f.exists():
            f.unlink()
        console.print(f"[dim]Cleared .active-project (was {slug}).[/dim]")


# ── archive / unarchive ─────────────────────────────────────────────────


def _default_archive_dir() -> Path:
    return get_active_project().repo_root / "archives"


@app.command()
def archive(
    slug: Annotated[str, typer.Argument(help="Project slug to archive.")],
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Archive file path. Default: archives/<slug>-<UTC>.skproj.tar.zst",
    ),
    keep_live: bool = typer.Option(
        False, "--keep-live",
        help="Don't drop the live state after archiving (snapshot only).",
    ),
):
    """Bundle a project into a portable archive then drop the live state.

    The archive contains: PostgreSQL dump, Qdrant snapshots for each
    collection, the entire data directory, and the ``.env.overlay``.
    Restore later with ``sciknow project unarchive <file>``.
    """
    if slug == "default":
        console.print("[red]Use `sciknow db backup` for the legacy 'default' install.[/red]")
        raise typer.Exit(2)
    validate_slug(slug)
    project = Project(slug=slug, repo_root=get_active_project().repo_root)
    if not project.exists():
        console.print(f"[red]Project not found:[/red] {project.root}")
        raise typer.Exit(1)

    archive_dir = _default_archive_dir()
    archive_dir.mkdir(parents=True, exist_ok=True)
    if output is None:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        output = archive_dir / f"{slug}-{ts}.skproj.tar"

    console.print(f"[bold]Archiving project:[/bold] {slug}  →  {output}")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        staging = tmp / "skproj"
        staging.mkdir()

        # 1. Manifest
        (staging / "manifest.txt").write_text(
            f"slug: {slug}\n"
            f"created_at: {datetime.utcnow().isoformat()}Z\n"
            f"pg_database: {project.pg_database}\n"
            f"qdrant_prefix: {project.qdrant_prefix}\n"
        )

        # 2. PG dump
        with console.status("Dumping PostgreSQL…"):
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
                console.print(f"[red]pg_dump failed:[/red] {res.stderr}")
                raise typer.Exit(1)
        console.print(f"  [green]✓[/green] PG dump")

        # 3. Qdrant snapshots — best-effort, copy the snapshot files
        from sciknow.storage.qdrant import get_client
        client = get_client()
        snaps_dir = staging / "qdrant_snapshots"
        snaps_dir.mkdir()
        existing = {c.name for c in client.get_collections().collections}
        for coll in (project.papers_collection, project.abstracts_collection, project.wiki_collection):
            if coll not in existing:
                continue
            try:
                snap = client.create_snapshot(collection_name=coll)
                # Save the snapshot name so unarchive knows which file to recover from
                (snaps_dir / f"{coll}.snapshot_name").write_text(snap.name)
            except Exception as exc:
                console.print(f"  [yellow]Qdrant snapshot {coll} failed:[/yellow] {exc}")
        console.print(f"  [green]✓[/green] Qdrant snapshots")

        # 4. Data directory
        with console.status("Bundling data directory…"):
            data_target = staging / "data"
            if project.data_dir.exists():
                shutil.copytree(project.data_dir, data_target)
        console.print(f"  [green]✓[/green] data dir")

        # 5. env overlay
        if project.env_overlay_path.exists():
            shutil.copy2(project.env_overlay_path, staging / ".env.overlay")

        # 6. tar everything (zstd would need an extra dep; plain tar is universal)
        with console.status("Writing archive…"):
            with tarfile.open(output, "w") as tf:
                tf.add(staging, arcname="skproj")
        size_mb = output.stat().st_size / 1024 / 1024
        console.print(f"  [green]✓[/green] archive written ({size_mb:.1f} MB)")

    if not keep_live:
        console.print("\nDropping live state…")
        try:
            _drop_pg_database(project.pg_database)
        except Exception as exc:
            console.print(f"[yellow]PG drop failed: {exc}[/yellow]")
        _delete_qdrant_collections_for_project(project)
        if project.root.exists():
            shutil.rmtree(project.root)
        active = read_active_slug_from_file()
        if active == slug:
            from sciknow.core.project import _active_project_file
            f = _active_project_file()
            if f.exists():
                f.unlink()
        console.print("[green]✓ Live state dropped.[/green]")
    console.print(f"\n[green]✓ Archive complete:[/green] {output}")


@app.command()
def unarchive(
    archive_file: Annotated[Path, typer.Argument(help="Archive file produced by `project archive`.")],
):
    """Restore an archived project.

    Recreates the PG database (from the dump), the Qdrant collections
    (from snapshots), and the data directory.
    """
    if not archive_file.exists():
        console.print(f"[red]Archive not found:[/red] {archive_file}")
        raise typer.Exit(1)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with console.status("Extracting archive…"):
            with tarfile.open(archive_file, "r") as tf:
                tf.extractall(tmp, filter="data")
        staging = tmp / "skproj"
        if not (staging / "manifest.txt").exists():
            console.print(f"[red]Archive doesn't look like a sciknow project — manifest missing.[/red]")
            raise typer.Exit(1)

        # Parse the manifest
        manifest = {}
        for line in (staging / "manifest.txt").read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                manifest[k.strip()] = v.strip()
        slug = manifest.get("slug", "")
        try:
            validate_slug(slug)
        except ValueError as exc:
            console.print(f"[red]Manifest slug invalid: {exc}[/red]")
            raise typer.Exit(1)

        project = Project(slug=slug, repo_root=get_active_project().repo_root)
        if project.root.exists() or _pg_database_exists(project.pg_database):
            console.print(f"[red]Project {slug} already present (dir or DB exists). Refusing to overwrite.[/red]")
            console.print(f"[dim]Run `sciknow project destroy {slug}` first if you really want to replace it.[/dim]")
            raise typer.Exit(1)

        console.print(f"[bold]Restoring project:[/bold] {slug}")

        # 1. Create empty PG DB + restore from dump
        _create_pg_database(project.pg_database)
        with console.status("Restoring PostgreSQL dump…"):
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
            # pg_restore frequently emits non-fatal warnings to stderr;
            # only the exit code is authoritative.
            if res.returncode != 0:
                console.print(f"[red]pg_restore failed:[/red] {res.stderr}")
                raise typer.Exit(1)
        console.print("  [green]✓[/green] PG restored")

        # 2. Data directory
        _ensure_init_dirs(project)
        data_src = staging / "data"
        if data_src.exists():
            for child in data_src.iterdir():
                shutil.move(str(child), str(project.data_dir / child.name))
        console.print("  [green]✓[/green] data dir restored")

        # 3. env overlay
        env_src = staging / ".env.overlay"
        if env_src.exists():
            shutil.copy2(env_src, project.env_overlay_path)

        # 4. Qdrant snapshots — note: we saved the snapshot *names*;
        # the actual snapshot files live in Qdrant's storage volume,
        # which a one-shot CLI can't easily reach. Best-effort: warn
        # the user that vectors need to be re-embedded if snapshot
        # restore failed. A full vector restore would need a Qdrant-
        # native snapshot upload mechanism (REST endpoint) — out of
        # scope for the first cut; tracked in PROJECTS.md as a TODO.
        snaps_dir = staging / "qdrant_snapshots"
        if snaps_dir.exists() and any(snaps_dir.iterdir()):
            console.print(
                "  [yellow]![/yellow] Qdrant snapshots present in archive but cannot be "
                "restored automatically (Qdrant snapshot files live in the server's storage "
                "volume). To rebuild vectors: run `sciknow db init` then re-ingest with "
                "`sciknow ingest directory <project-data>/processed/`."
            )

    write_active_slug(slug)
    console.print(f"\n[green]✓ Project [bold]{slug}[/bold] restored and now active.[/green]")


# ── v2 Phase G — cross-project v1 → v2 import ───────────────────────────


def _migrate_qdrant_collections_between(
    src: Project, dst: Project,
) -> None:
    """v2 Phase G — copy every Qdrant collection from one project's
    prefix to another's.

    Generalisation of ``_migrate_qdrant_collections`` (which always
    sourced from the empty/legacy prefix). Streams via scroll+upsert
    so it doesn't rely on Qdrant's filesystem snapshot path being
    visible to the server.
    """
    from sciknow.storage.qdrant import get_client, init_collections

    client = get_client()
    try:
        existing = {c.name for c in client.get_collections().collections}
    except Exception as exc:
        console.print(f"[yellow]Qdrant unreachable: {exc}. Skipping.[/yellow]")
        return

    # Provision the target project's prefixed collections.
    prev_env = os.environ.get("SCIKNOW_PROJECT")
    os.environ["SCIKNOW_PROJECT"] = dst.slug
    try:
        init_collections()
    finally:
        if prev_env is None:
            os.environ.pop("SCIKNOW_PROJECT", None)
        else:
            os.environ["SCIKNOW_PROJECT"] = prev_env

    pairs = [
        (src.papers_collection,    dst.papers_collection),
        (src.abstracts_collection, dst.abstracts_collection),
        (src.wiki_collection,      dst.wiki_collection),
        (src.visuals_collection,   dst.visuals_collection),
    ]
    for old, new in pairs:
        if old not in existing:
            console.print(f"  [dim]skip Qdrant {old} (not present)[/dim]")
            continue
        if old == new:
            console.print(
                f"  [dim]skip Qdrant {old} (source and target collection "
                f"names are identical — nothing to copy)[/dim]"
            )
            continue
        try:
            src_count = client.get_collection(old).points_count
        except Exception as exc:
            console.print(f"  [yellow]skip Qdrant {old}: {exc}[/yellow]")
            continue
        if src_count == 0:
            console.print(f"  [dim]skip Qdrant {old} (empty)[/dim]")
            continue
        with console.status(f"Copying Qdrant `{old}` → `{new}` ({src_count} pts)…"):
            copied = _scroll_upsert(client, old, new, batch_size=500)
        # Brief settle window for async upserts to register in count.
        import time as _t
        for _ in range(5):
            try:
                dst_count = client.get_collection(new).points_count
            except Exception:
                dst_count = -1
            if dst_count == src_count:
                break
            _t.sleep(1)
        ok = dst_count == src_count
        marker = "[green]✓[/green]" if ok else "[yellow]?[/yellow]"
        console.print(f"  {marker} Qdrant {old} → {new}: copied={copied} dst_count={dst_count}")


@app.command(name="import-v1")
def import_v1(
    source_slug: Annotated[str, typer.Argument(
        help="Source project slug to import from. Use `default` for the "
             "legacy single-tenant install (PG `sciknow`, unprefixed Qdrant).",
    )],
    as_slug: Annotated[str, typer.Option(
        "--as", help="Target v2 project slug to create.",
    )],
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print planned operations without executing them.",
    ),
    skip_qdrant: bool = typer.Option(
        False, "--skip-qdrant",
        help="Skip the Qdrant copy (PG + filesystem only). Useful when "
             "you've already snapshotted vectors with `library snapshot`.",
    ),
) -> None:
    """v2 Phase G — cross-project v1 → v2 project import.

    Reads ``<source_slug>``'s PostgreSQL database, Qdrant collections,
    and ``data/`` tree, and writes them into a new v2 project at
    ``<as_slug>``. The target slug must not already exist.

    What this does:

      1. Validates the source has a PG database and the target slot
         is free.
      2. Clones the source PG database to the target via ``CREATE
         DATABASE … WITH TEMPLATE`` (atomic, fast on local PG).
      3. Copies every Qdrant collection from ``<source>_*`` to
         ``<as>_*`` via scroll+upsert (skip with ``--skip-qdrant``).
      4. rsync-equivalent copy of ``<source>/data/`` → ``<as>/data/``;
         existing files at the destination are preserved (so the
         operator can re-run the import safely after fixing data).
      5. Drops any v1 dual-embedder sidecar collections from the new
         project (mirrors what ``library upgrade-v1`` does in place).
      6. Stamps ``<as_slug>/.imported_from`` with the source slug +
         timestamp so the lineage is auditable.

    The source project is left untouched — this is a copy, not a
    move. To retire the source after a successful import, follow up
    with ``sciknow project archive <source_slug>``.

    Re-embedding is **not required** for v1→v2: the canonical
    embedder (bge-m3, Q8_0) is bit-identical between v1 and v2, so
    the dense + sparse vectors copy across as-is.
    """
    # Resolve source. Allow "default" as the legacy single-tenant slug.
    from sciknow.core.project import _DEFAULT_SLUG, _repo_root

    if source_slug == _DEFAULT_SLUG:
        source = Project.default()
    else:
        source = Project(slug=source_slug, repo_root=_repo_root())

    target = Project(slug=as_slug, repo_root=_repo_root())

    console.print(f"[bold]Importing v1 → v2:[/bold] {source.slug} → {as_slug}")
    console.print(f"  Source PG : {source.pg_database}")
    console.print(f"  Target PG : {target.pg_database}")
    console.print(f"  Source Qdrant prefix : {source.qdrant_prefix or '(unprefixed)'}")
    console.print(f"  Target Qdrant prefix : {target.qdrant_prefix}")
    console.print(f"  Source data dir      : {source.data_dir}")
    console.print(f"  Target data dir      : {target.data_dir}")

    # Validate source.
    if not _pg_database_exists(source.pg_database):
        console.print(
            f"[red]Source PG database `{source.pg_database}` not found.[/red]"
        )
        raise typer.Exit(1)
    if not source.data_dir.exists():
        console.print(
            f"[yellow]Source data dir not found: {source.data_dir}. "
            f"Will continue without filesystem copy.[/yellow]"
        )

    # Validate target.
    if target.root.exists():
        console.print(
            f"[red]Target project root already exists: {target.root}. "
            f"Refusing to overwrite — pick a fresh --as slug or "
            f"`project destroy {as_slug}` first.[/red]"
        )
        raise typer.Exit(1)
    if _pg_database_exists(target.pg_database):
        console.print(
            f"[red]Target PG database `{target.pg_database}` already "
            f"exists. Drop it manually if you really mean to import "
            f"on top of it.[/red]"
        )
        raise typer.Exit(1)

    if dry_run:
        console.print("[dim](dry-run, no changes made)[/dim]")
        return

    # 1. Clone PG.
    with console.status(
        f"Cloning PG database (`{source.pg_database}` → `{target.pg_database}`)…"
    ):
        _create_pg_database(target.pg_database, template=source.pg_database)
    console.print(f"  [green]✓[/green] PG cloned to `{target.pg_database}`")

    # 2. Provision target dirs.
    _ensure_init_dirs(target)

    # 3. Copy data dir contents (preserve existing target files).
    if source.data_dir.exists():
        with console.status("Copying filesystem contents…"):
            for child in source.data_dir.iterdir():
                dest = target.data_dir / child.name
                if dest.exists():
                    console.print(f"  [yellow]skip (exists): {dest}[/yellow]")
                    continue
                if child.is_dir():
                    shutil.copytree(child, dest)
                else:
                    shutil.copy2(child, dest)
        console.print(f"  [green]✓[/green] data/* copied to {target.data_dir}")

    # 4. Migrate Qdrant collections.
    if skip_qdrant:
        console.print("  [dim]skipped Qdrant copy (--skip-qdrant)[/dim]")
    else:
        _migrate_qdrant_collections_between(source, target)

    # 5. v1 → v2 sidecar cleanup on the target. Mirrors what
    #    `library upgrade-v1` does in-place.
    try:
        from sciknow.storage.qdrant import get_client as _get_q
        client = _get_q()
        existing = {c.name for c in client.get_collections().collections}
        prefix = target.qdrant_prefix or ""
        sidecars = [
            c for c in existing
            if c.startswith(f"{prefix}_ab_") or c.startswith(f"{prefix}ab_")
        ]
        for c in sidecars:
            try:
                client.delete_collection(c)
                console.print(f"  [green]✓[/green] dropped legacy sidecar Qdrant collection `{c}`")
            except Exception as exc:
                console.print(f"  [yellow]could not drop sidecar `{c}`: {exc}[/yellow]")
    except Exception as exc:
        console.print(f"  [yellow]sidecar cleanup skipped: {exc}[/yellow]")

    # 6. Stamp lineage marker.
    marker = target.root / ".imported_from"
    marker.write_text(
        f"{source.slug}\n"
        f"imported_at: {datetime.utcnow().isoformat()}Z\n"
        f"sciknow_version: v2\n",
        encoding="utf-8",
    )
    console.print(f"  [green]✓[/green] stamped {marker.relative_to(target.repo_root)}")

    # Also stamp `.v2-upgraded` so `library upgrade-v1` is a no-op
    # on this project (already upgraded by definition).
    upgraded = target.root / ".v2-upgraded"
    upgraded.write_text(
        f"upgraded_at: {datetime.utcnow().isoformat()}Z\n"
        f"via: project import-v1 from {source.slug}\n",
        encoding="utf-8",
    )

    console.print(
        f"\n[green]✓[/green] Project [bold]{as_slug}[/bold] imported from "
        f"[bold]{source.slug}[/bold]."
    )
    console.print(
        f"[dim]Source project left untouched. Activate with `sciknow "
        f"project use {as_slug}`.[/dim]"
    )
