import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, MofNCompleteColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

app = typer.Typer(help="Database and infrastructure management.")
console = Console()


# ── backup ─────────────────────────────────────────────────────────────────────

@app.command()
def backup(
    output: Path = typer.Option(Path("sciknow_backup.tar.gz"), "--output", "-o",
                                help="Path for the backup archive."),
    include_pdfs:    bool = typer.Option(True,  "--pdfs/--no-pdfs",
                                         help="Include original ingested PDFs (data/processed/)."),
    include_marker:  bool = typer.Option(False, "--marker/--no-marker",
                                         help="Include Marker markdown output (regenerable from PDFs)."),
    include_downloads: bool = typer.Option(True, "--downloads/--no-downloads",
                                           help="Include auto-downloaded PDFs (downloads/)."),
):
    """
    Back up the entire sciknow collection to a portable archive.

    The archive contains:

    \\b
      - PostgreSQL dump (all papers, chunks, metadata)
      - Qdrant vector snapshots (embeddings for all collections)
      - Original PDFs (data/processed/)          [--pdfs, on by default]
      - Auto-downloaded PDFs (data/downloads/)     [--downloads, on by default]
      - Marker markdown output (data/mineru_output/) [--marker, off by default]
      - .env configuration file

    To restore on a new machine:

    \\b
      sciknow db restore sciknow_backup.tar.gz

    Examples:

    \\b
      sciknow db backup
      sciknow db backup --output ~/backups/sciknow_2026-04-04.tar.gz
      sciknow db backup --no-pdfs   # metadata + vectors only (much smaller)
    """
    from sciknow.config import settings
    from sciknow.storage.qdrant import get_client

    console.print(f"[bold]Creating backup → {output}[/bold]")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "sciknow_backup"
        staging.mkdir()

        # ── 1. PostgreSQL dump ─────────────────────────────────────────────────
        with console.status("Dumping PostgreSQL…"):
            pg_dump_path = staging / "postgres.dump"
            env = os.environ.copy()
            env["PGPASSWORD"] = settings.pg_password
            result = subprocess.run(
                [
                    "pg_dump",
                    "-h", settings.pg_host,
                    "-p", str(settings.pg_port),
                    "-U", settings.pg_user,
                    "-F", "c",          # custom format (compressed, fast restore)
                    "-f", str(pg_dump_path),
                    settings.pg_database,
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                console.print(f"[red]pg_dump failed:[/red] {result.stderr}")
                raise typer.Exit(1)
        console.print(f"  [green]✓[/green] PostgreSQL dump  ({pg_dump_path.stat().st_size // 1024 // 1024} MB)")

        # ── 2. Qdrant snapshots ────────────────────────────────────────────────
        with console.status("Creating Qdrant snapshots…"):
            qdrant = get_client()
            qdrant_dir = staging / "qdrant_snapshots"
            qdrant_dir.mkdir()

            collections = [c.name for c in qdrant.get_collections().collections]
            for coll in collections:
                snap = qdrant.create_snapshot(collection_name=coll)
                # Download the snapshot file from Qdrant's storage
                import httpx
                from sciknow.config import settings as s
                snap_url = (
                    f"http://{s.qdrant_host}:{s.qdrant_port}"
                    f"/collections/{coll}/snapshots/{snap.name}"
                )
                snap_path = qdrant_dir / f"{coll}.snapshot"
                with httpx.Client(timeout=300) as client:
                    resp = client.get(snap_url)
                    resp.raise_for_status()
                    snap_path.write_bytes(resp.content)
                console.print(
                    f"  [green]✓[/green] Qdrant [bold]{coll}[/bold]  "
                    f"({snap_path.stat().st_size // 1024 // 1024} MB)"
                )

        # ── 3. PDF files ───────────────────────────────────────────────────────
        if include_pdfs and settings.processed_dir.exists():
            with console.status("Copying processed PDFs…"):
                shutil.copytree(settings.processed_dir, staging / "processed")
            n = sum(1 for _ in (staging / "processed").rglob("*.pdf"))
            console.print(f"  [green]✓[/green] Processed PDFs  ({n} files)")

        if include_downloads:
            dl_dir = Path("data/downloads")
            if dl_dir.exists():
                with console.status("Copying downloaded PDFs…"):
                    shutil.copytree(dl_dir, staging / "downloads")
                n = sum(1 for _ in (staging / "downloads").rglob("*.pdf"))
                console.print(f"  [green]✓[/green] Downloaded PDFs  ({n} files)")

        if include_marker and settings.mineru_output_dir.exists():
            with console.status("Copying Marker output…"):
                shutil.copytree(settings.mineru_output_dir, staging / "mineru_output")
            console.print("  [green]✓[/green] Marker markdown output")

        # ── 4. .env ────────────────────────────────────────────────────────────
        env_file = Path(".env")
        if env_file.exists():
            shutil.copy(env_file, staging / ".env")
            console.print("  [green]✓[/green] .env config")

        # ── 5. Write manifest ──────────────────────────────────────────────────
        import datetime
        manifest = {
            "created": datetime.datetime.utcnow().isoformat() + "Z",
            "collections": collections,
            "includes": {
                "postgres": True,
                "qdrant": True,
                "pdfs": include_pdfs,
                "downloads": include_downloads,
                "marker": include_marker,
            },
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # ── 6. Create tar.gz ───────────────────────────────────────────────────
        with console.status(f"Compressing → {output}…"):
            output.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(output, "w:gz") as tar:
                tar.add(staging, arcname="sciknow_backup")

    size_mb = output.stat().st_size // 1024 // 1024
    console.print(f"\n[bold green]✓ Backup complete[/bold green] → [bold]{output}[/bold]  ({size_mb} MB)")
    console.print(
        "\nRestore on a new machine with:\n"
        f"  [bold]sciknow db restore {output}[/bold]"
    )


# ── restore ────────────────────────────────────────────────────────────────────

@app.command()
def restore(
    archive: Path = typer.Argument(help="Path to the backup archive produced by 'sciknow db backup'."),
    skip_pdfs:    bool = typer.Option(False, "--skip-pdfs",    help="Skip restoring PDF files."),
    skip_vectors: bool = typer.Option(False, "--skip-vectors", help="Skip restoring Qdrant snapshots."),
    force:        bool = typer.Option(False, "--force",
                                      help="Drop and recreate the database before restoring (required if DB already exists)."),
):
    """
    Restore a sciknow backup on a new machine.

    Expects PostgreSQL and Qdrant to already be running (use 'sciknow db init'
    first to create the schema, or use --force to drop and recreate).

    Examples:

    \\b
      sciknow db restore sciknow_backup.tar.gz
      sciknow db restore sciknow_backup.tar.gz --force
      sciknow db restore sciknow_backup.tar.gz --skip-vectors
    """
    from sciknow.config import settings

    if not archive.exists():
        console.print(f"[red]Archive not found:[/red] {archive}")
        raise typer.Exit(1)

    console.print(f"[bold]Restoring from {archive}…[/bold]")

    with tempfile.TemporaryDirectory() as tmp:
        # ── 1. Extract archive ─────────────────────────────────────────────────
        with console.status("Extracting archive…"):
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(tmp)
        staging = Path(tmp) / "sciknow_backup"

        # Read manifest
        manifest_path = staging / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        console.print(f"  Backup created: {manifest.get('created', 'unknown')}")

        # ── 2. PostgreSQL restore ──────────────────────────────────────────────
        pg_dump_path = staging / "postgres.dump"
        if pg_dump_path.exists():
            env = os.environ.copy()
            env["PGPASSWORD"] = settings.pg_password

            if force:
                with console.status("Dropping and recreating database…"):
                    subprocess.run(
                        ["dropdb", "-h", settings.pg_host, "-p", str(settings.pg_port),
                         "-U", settings.pg_user, "--if-exists", settings.pg_database],
                        env=env, capture_output=True,
                    )
                    subprocess.run(
                        ["createdb", "-h", settings.pg_host, "-p", str(settings.pg_port),
                         "-U", settings.pg_user, settings.pg_database],
                        env=env, capture_output=True,
                    )

            with console.status("Restoring PostgreSQL…"):
                result = subprocess.run(
                    [
                        "pg_restore",
                        "-h", settings.pg_host,
                        "-p", str(settings.pg_port),
                        "-U", settings.pg_user,
                        "-d", settings.pg_database,
                        "--no-owner",
                        "--no-privileges",
                        "-1",           # single transaction
                        str(pg_dump_path),
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                )
            if result.returncode != 0:
                console.print(f"[yellow]pg_restore warnings:[/yellow] {result.stderr[:500]}")
            console.print("  [green]✓[/green] PostgreSQL restored")

        # ── 3. Qdrant snapshots ────────────────────────────────────────────────
        if not skip_vectors:
            qdrant_dir = staging / "qdrant_snapshots"
            if qdrant_dir.exists():
                import httpx
                from sciknow.storage.qdrant import get_client, init_collections
                qdrant = get_client()

                # Ensure collections exist
                init_collections()

                for snap_file in sorted(qdrant_dir.glob("*.snapshot")):
                    coll = snap_file.stem
                    with console.status(f"Uploading Qdrant snapshot [{coll}]…"):
                        snap_url = (
                            f"http://{settings.qdrant_host}:{settings.qdrant_port}"
                            f"/collections/{coll}/snapshots/upload?priority=snapshot"
                        )
                        with httpx.Client(timeout=600) as client:
                            with snap_file.open("rb") as f:
                                resp = client.post(
                                    snap_url,
                                    content=f.read(),
                                    headers={"Content-Type": "application/octet-stream"},
                                )
                            resp.raise_for_status()
                    console.print(f"  [green]✓[/green] Qdrant [{coll}] restored")

        # ── 4. PDF files ───────────────────────────────────────────────────────
        if not skip_pdfs:
            processed_src = staging / "processed"
            if processed_src.exists():
                settings.processed_dir.mkdir(parents=True, exist_ok=True)
                with console.status("Restoring processed PDFs…"):
                    shutil.copytree(processed_src, settings.processed_dir, dirs_exist_ok=True)
                n = sum(1 for _ in settings.processed_dir.rglob("*.pdf"))
                console.print(f"  [green]✓[/green] Processed PDFs  ({n} files)")

            downloads_src = staging / "downloads"
            if downloads_src.exists():
                dl_dest = Path("data/downloads")
                dl_dest.mkdir(exist_ok=True)
                with console.status("Restoring downloaded PDFs…"):
                    shutil.copytree(downloads_src, dl_dest, dirs_exist_ok=True)
                n = sum(1 for _ in dl_dest.rglob("*.pdf"))
                console.print(f"  [green]✓[/green] Downloaded PDFs  ({n} files)")

            marker_src = staging / "mineru_output"
            if marker_src.exists():
                with console.status("Restoring Marker output…"):
                    shutil.copytree(marker_src, settings.mineru_output_dir, dirs_exist_ok=True)
                console.print("  [green]✓[/green] Marker output restored")

        # ── 5. .env ────────────────────────────────────────────────────────────
        env_src = staging / ".env"
        if env_src.exists() and not Path(".env").exists():
            shutil.copy(env_src, ".env")
            console.print("  [green]✓[/green] .env restored  [dim](edit if host/credentials differ)[/dim]")

    console.print("\n[bold green]✓ Restore complete.[/bold green]")
    console.print(
        "Run [bold]sciknow db stats[/bold] to verify the collection is intact."
    )


@app.command()
def init():
    """Initialise PostgreSQL schema and Qdrant collections."""
    from alembic import command
    from alembic.config import Config as AlembicConfig

    from sciknow.storage.db import check_connection
    from sciknow.storage.qdrant import check_connection as qdrant_ok, init_collections

    console.print("[bold]Checking services...[/bold]")

    if not check_connection():
        console.print("[red]✗ PostgreSQL unreachable.[/red] Check PG_HOST/PG_USER/PG_PASSWORD in .env")
        raise typer.Exit(1)
    console.print("[green]✓ PostgreSQL[/green]")

    if not qdrant_ok():
        console.print("[red]✗ Qdrant unreachable.[/red] Check QDRANT_HOST in .env")
        raise typer.Exit(1)
    console.print("[green]✓ Qdrant[/green]")

    console.print("\n[bold]Running migrations...[/bold]")
    cfg = AlembicConfig("alembic.ini")
    command.upgrade(cfg, "head")
    console.print("[green]✓ Schema up to date[/green]")

    console.print("\n[bold]Initialising Qdrant collections...[/bold]")
    init_collections()
    console.print("[green]✓ Collections ready[/green]")

    console.print("\n[bold green]✓ Init complete.[/bold green]")


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    keep_pdfs: bool = typer.Option(True, "--keep-pdfs/--no-keep-pdfs",
                                    help="Keep PDFs in data/processed/ and downloads/ (default: yes)."),
    keep_marker: bool = typer.Option(False, "--keep-marker",
                                      help="Keep Marker output cache in data/mineru_output/ (default: no)."),
):
    """
    Wipe the entire database and vector store, then re-initialise from scratch.

    Deletes ALL ingested data: PostgreSQL tables, Qdrant collections, and
    Marker output cache. PDFs in data/processed/ and data/downloads/ are kept by
    default so you can re-ingest without downloading everything again.

    Use this before a full re-ingest (e.g. after switching to JSON output mode).

    Examples:

      sciknow db reset --yes

      sciknow db reset --yes --no-keep-pdfs   # also delete all PDFs
    """
    import shutil

    from alembic import command
    from alembic.config import Config as AlembicConfig
    from sqlalchemy import text

    from sciknow.config import settings
    from sciknow.storage.db import check_connection
    from sciknow.storage.db import engine as db_engine
    from sciknow.storage.qdrant import (
        PAPERS_COLLECTION,
        check_connection as qdrant_ok,
        init_collections,
    )
    from sciknow.storage.qdrant import get_client as get_qdrant

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------
    if not check_connection():
        console.print("[red]✗ PostgreSQL unreachable.[/red]")
        raise typer.Exit(1)
    if not qdrant_ok():
        console.print("[red]✗ Qdrant unreachable.[/red]")
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Confirm
    # ------------------------------------------------------------------
    marker_dir = settings.mineru_output_dir
    marker_size = sum(
        f.stat().st_size for f in marker_dir.rglob("*") if f.is_file()
    ) if marker_dir.exists() else 0

    console.print()
    console.print("[bold red]⚠  This will permanently delete:[/bold red]")
    console.print("  • All PostgreSQL tables (documents, chunks, metadata, books, drafts, …)")
    console.print("  • All Qdrant vector collections")
    if not keep_marker:
        console.print(
            f"  • Marker output cache ({marker_dir})  "
            f"[dim]{marker_size // 1024 // 1024} MB[/dim]"
        )
    if not keep_pdfs:
        console.print(
            f"  • PDFs in data/processed/ and data/downloads/"
        )
    console.print()

    if not yes:
        typer.confirm("Are you sure you want to reset the entire database?", abort=True)

    # ------------------------------------------------------------------
    # 1. Drop and recreate PostgreSQL schema
    # ------------------------------------------------------------------
    console.print("\n[bold]Dropping PostgreSQL schema...[/bold]")
    with db_engine.connect() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.commit()

    console.print("[bold]Re-running migrations...[/bold]")
    cfg = AlembicConfig("alembic.ini")
    command.upgrade(cfg, "head")
    console.print("[green]✓ PostgreSQL schema reset[/green]")

    # ------------------------------------------------------------------
    # 2. Drop and recreate Qdrant collections
    # ------------------------------------------------------------------
    console.print("\n[bold]Dropping Qdrant collections...[/bold]")
    qdrant = get_qdrant()
    try:
        collections = qdrant.get_collections().collections
        for col in collections:
            qdrant.delete_collection(col.name)
            console.print(f"  Deleted collection: {col.name}")
    except Exception as e:
        console.print(f"[yellow]Warning: {e}[/yellow]")

    init_collections()
    console.print("[green]✓ Qdrant collections reset[/green]")

    # ------------------------------------------------------------------
    # 3. Delete Marker output cache
    # ------------------------------------------------------------------
    if not keep_marker and marker_dir.exists():
        console.print(f"\n[bold]Deleting Marker cache ({marker_dir})...[/bold]")
        shutil.rmtree(marker_dir)
        console.print("[green]✓ Marker cache deleted[/green]")

    # ------------------------------------------------------------------
    # 4. Optionally delete PDFs
    # ------------------------------------------------------------------
    if not keep_pdfs:
        # settings.data_dir / "downloads" is already data/downloads
        for pdf_dir in [settings.processed_dir, settings.data_dir / "downloads"]:
            if pdf_dir.exists():
                console.print(f"\n[bold]Deleting {pdf_dir}...[/bold]")
                shutil.rmtree(pdf_dir)
                pdf_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    console.print()
    console.print("[bold green]✓ Reset complete.[/bold green]")
    console.print()
    console.print("Re-ingest your documents with:")
    console.print("  [bold]sciknow ingest directory data/processed/[/bold]")
    console.print("  [bold]sciknow ingest directory data/downloads/[/bold]")


@app.command()
def stats():
    """Show paper counts and ingestion status breakdown."""
    from sqlalchemy import func, text

    from sciknow.storage.db import get_session
    from sciknow.storage.models import Chunk, Document, PaperMetadata
    from sciknow.storage.qdrant import get_client

    with get_session() as session:
        total_docs = session.query(func.count(Document.id)).scalar()
        status_rows = (
            session.query(Document.ingestion_status, func.count(Document.id))
            .group_by(Document.ingestion_status)
            .all()
        )
        total_chunks = session.query(func.count(Chunk.id)).scalar()
        embedded = (
            session.query(func.count(Chunk.id))
            .filter(Chunk.qdrant_point_id.isnot(None))
            .scalar()
        )
        with_metadata = session.query(func.count(PaperMetadata.id)).scalar()

    try:
        qdrant = get_client()
        papers_info = qdrant.get_collection("papers")
        qdrant_points = papers_info.points_count
    except Exception:
        qdrant_points = "N/A"

    table = Table(title="SciKnow Stats", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total documents", str(total_docs))
    table.add_row("With metadata", str(with_metadata))
    table.add_row("Total chunks", str(total_chunks))
    table.add_row("Embedded chunks", str(embedded))
    table.add_row("Qdrant points (papers)", str(qdrant_points))
    table.add_section()

    for status, count in sorted(status_rows):
        colour = "green" if status == "complete" else "red" if status == "failed" else "yellow"
        table.add_row(f"  [{colour}]{status}[/{colour}]", str(count))

    console.print(table)


@app.command(name="refresh-metadata")
def refresh_metadata(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be updated without making changes."),
    source: str = typer.Option("all", "--source",
                               help="Which metadata sources to refresh: 'unknown', 'embedded_pdf', 'llm_extracted', or 'all'."),
):
    """
    Re-run metadata extraction for papers with poor-quality metadata.

    Targets papers where:
    - metadata_source is 'unknown' (LLM fallback failed)
    - metadata_source is 'embedded_pdf' with a garbage title (e.g. 'Microsoft Word - ...')
    - metadata_source is 'llm_extracted' (re-run with fixed LLM API)

    The markdown output from the original Marker conversion is reused — no re-conversion needed.
    """
    from sqlalchemy import text

    from sciknow.ingestion.metadata import _is_garbage_title, extract
    from sciknow.storage.db import get_session
    from sciknow.storage.models import PaperMetadata

    valid_sources = {"unknown", "embedded_pdf", "llm_extracted", "all"}
    if source not in valid_sources:
        console.print(f"[red]Invalid --source.[/red] Choose from: {', '.join(sorted(valid_sources))}")
        raise typer.Exit(1)

    with get_session() as session:
        # Find candidates
        if source == "all":
            src_filter = "pm.metadata_source IN ('unknown', 'embedded_pdf', 'llm_extracted')"
        else:
            src_filter = f"pm.metadata_source = '{source}'"

        rows = session.execute(text(f"""
            SELECT d.id::text, d.original_path, d.mineru_output_path,
                   pm.title, pm.metadata_source
            FROM documents d
            JOIN paper_metadata pm ON pm.document_id = d.id
            WHERE {src_filter}
              AND d.ingestion_status = 'complete'
        """)).fetchall()

    # Further filter embedded_pdf: only those with garbage titles
    candidates = []
    for doc_id, orig_path, marker_out, title, meta_source in rows:
        if meta_source == "embedded_pdf" and title and not _is_garbage_title(title):
            continue  # good title, skip
        candidates.append((doc_id, orig_path, marker_out, title, meta_source))

    if not candidates:
        console.print("[green]No papers need metadata refresh.[/green]")
        raise typer.Exit(0)

    console.print(f"Found [bold]{len(candidates)}[/bold] papers needing metadata refresh.")
    if dry_run:
        console.print("\n[dim]Dry run — no changes made. Papers that would be updated:[/dim]")
        for _, orig_path, _, title, src in candidates[:20]:
            console.print(f"  [dim]{src}[/dim]  {title or '(no title)'}  [dim]{Path(orig_path).name}[/dim]")
        if len(candidates) > 20:
            console.print(f"  [dim]... and {len(candidates) - 20} more[/dim]")
        raise typer.Exit(0)

    updated = skipped = failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Refreshing metadata", total=len(candidates))

        for doc_id, orig_path, marker_out, old_title, _ in candidates:
            progress.update(task, description=f"[dim]{Path(orig_path).name[:50]}[/dim]")

            # Find the markdown file produced by Marker
            md_text = ""
            if marker_out:
                marker_path = Path(marker_out)
                md_files = list(marker_path.glob("**/*.md")) if marker_path.exists() else []
                if md_files:
                    md_text = md_files[0].read_text(encoding="utf-8", errors="replace")

            if not md_text:
                skipped += 1
                progress.advance(task)
                continue

            try:
                meta = extract(Path(orig_path), md_text)

                with get_session() as session:
                    pm = session.query(PaperMetadata).filter_by(
                        document_id=doc_id
                    ).first()
                    if pm is None:
                        skipped += 1
                        progress.advance(task)
                        continue

                    pm.title           = meta.title
                    pm.abstract        = meta.abstract or pm.abstract
                    pm.year            = meta.year or pm.year
                    pm.doi             = meta.doi or pm.doi
                    pm.arxiv_id        = meta.arxiv_id or pm.arxiv_id
                    pm.journal         = meta.journal or pm.journal
                    # Always use freshly extracted authors so garbage is cleared;
                    # only fall back to old value if extraction produced nothing
                    # AND the old source was an authoritative API (crossref/arxiv).
                    if meta.authors:
                        pm.authors = meta.authors
                    elif pm.metadata_source not in ("crossref", "arxiv"):
                        pm.authors = []   # clear garbage, no reliable fallback
                    pm.keywords        = meta.keywords or pm.keywords
                    pm.metadata_source = meta.source
                    pm.crossref_raw    = meta.crossref_raw or pm.crossref_raw
                    pm.arxiv_raw       = meta.arxiv_raw or pm.arxiv_raw

                updated += 1
            except Exception as exc:
                failed += 1

            progress.advance(task)

    console.print(
        f"[green]✓ Updated {updated}[/green]  "
        f"[yellow]skipped {skipped}[/yellow]  "
        f"[red]failed {failed}[/red]"
    )


@app.command()
def enrich(
    dry_run:   bool  = typer.Option(False,  "--dry-run",   help="Show what would be updated without making changes."),
    threshold: float = typer.Option(0.85,   "--threshold", help="Minimum title-similarity score to accept a Crossref match (0–1)."),
    limit:     int   = typer.Option(0,      "--limit",     help="Max papers to process (0 = all)."),
    delay:     float = typer.Option(0.2,    "--delay",     help="Seconds between Crossref API calls (be polite)."),
):
    """
    Enrich papers that lack a DOI by searching Crossref by title.

    For each paper without a DOI, queries the Crossref title-search API,
    verifies the top result using fuzzy title matching, and — if the similarity
    score meets the threshold — updates the record with the full Crossref
    metadata (DOI, abstract, journal, volume, authors, …).

    Also runs the arXiv layer on any paper that has an arXiv ID but is missing
    title or abstract.

    Examples:

      sciknow db enrich --dry-run

      sciknow db enrich --threshold 0.90

      sciknow db enrich --limit 50 --delay 0.5
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from sqlalchemy import text

    from sciknow.config import settings
    from sciknow.ingestion.metadata import (
        _is_garbage_title,
        _layer_arxiv,
        search_crossref_by_title,
        search_openalex_by_title,
        PaperMeta,
    )
    from sciknow.storage.db import get_session
    from sciknow.storage.models import PaperMetadata

    with get_session() as session:
        rows = session.execute(text("""
            SELECT pm.id::text, pm.title, pm.authors, pm.arxiv_id, pm.metadata_source
            FROM paper_metadata pm
            WHERE pm.doi IS NULL AND pm.title IS NOT NULL
            ORDER BY pm.year DESC NULLS LAST, pm.title
        """)).fetchall()

    if limit:
        rows = rows[:limit]

    if not rows:
        console.print("[green]No papers need enrichment.[/green]")
        raise typer.Exit(0)

    workers = max(1, settings.enrich_workers)
    console.print(
        f"Found [bold]{len(rows)}[/bold] papers without a DOI. "
        f"Querying Crossref (threshold={threshold}, {workers} workers)…"
    )
    if dry_run:
        console.print("[dim]Dry run — no changes will be written.[/dim]\n")

    matched = failed = skipped = 0

    def _lookup(row) -> tuple[str, str, PaperMeta | None, str]:
        """Pure API lookup — runs in worker thread, never touches the DB."""
        pm_id, title, authors, arxiv_id, _ = row

        if _is_garbage_title(title) or len(title.strip()) < 15:
            return pm_id, title, None, "skip"

        first_author: str | None = None
        if authors:
            first_author = (authors[0] or {}).get("name")

        meta = search_crossref_by_title(title, first_author, threshold=threshold)
        if meta is None:
            meta = search_openalex_by_title(title, first_author, threshold=threshold)
        if meta is None and arxiv_id:
            stub = PaperMeta(arxiv_id=arxiv_id)
            _layer_arxiv(stub)
            if stub.title:
                meta = stub

        return pm_id, title, meta, "ok" if meta else "no_match"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Enriching", total=len(rows))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_lookup, row) for row in rows]

            for fut in as_completed(futures):
                try:
                    pm_id, title, meta, status = fut.result()
                except Exception:
                    failed += 1
                    progress.advance(task)
                    continue

                progress.update(task, description=f"[dim]{title[:55]}[/dim]")

                if status == "skip" or meta is None:
                    skipped += 1
                    progress.advance(task)
                    continue

                if dry_run:
                    doi_str = f"doi:{meta.doi}" if meta.doi else f"arXiv:{meta.arxiv_id}"
                    console.print(
                        f"  [green]✓[/green] {title[:60]}  →  {doi_str}"
                        f"  [dim](score≥{threshold})[/dim]"
                    )
                    matched += 1
                    progress.advance(task)
                    continue

                try:
                    with get_session() as session:
                        pm = session.query(PaperMetadata).filter_by(id=pm_id).first()
                        if pm is None:
                            skipped += 1
                            progress.advance(task)
                            continue

                        pm.doi             = meta.doi or pm.doi
                        pm.arxiv_id        = meta.arxiv_id or pm.arxiv_id
                        pm.title           = meta.title or pm.title
                        pm.abstract        = meta.abstract or pm.abstract
                        pm.year            = meta.year or pm.year
                        pm.journal         = meta.journal or pm.journal
                        pm.volume          = meta.volume or pm.volume
                        pm.issue           = meta.issue or pm.issue
                        pm.pages           = meta.pages or pm.pages
                        pm.publisher       = meta.publisher or pm.publisher
                        if meta.authors:
                            pm.authors = meta.authors
                        pm.keywords        = meta.keywords or pm.keywords
                        pm.metadata_source = meta.source
                        pm.crossref_raw    = meta.crossref_raw or pm.crossref_raw
                        pm.arxiv_raw       = meta.arxiv_raw or pm.arxiv_raw
                        session.commit()

                    matched += 1
                except Exception:
                    failed += 1

                progress.advance(task)

    console.print(
        f"\n[green]✓ Matched & updated {matched}[/green]  "
        f"[yellow]no match {skipped}[/yellow]  "
        f"[red]failed {failed}[/red]"
    )
    if not dry_run and matched:
        console.print(
            f"\nRun [bold]sciknow catalog stats[/bold] to see the updated coverage."
        )


@app.command()
def expand(
    download_dir: Path  = typer.Option(Path("data/downloads"), "--download-dir", "-d",
                                        help="Directory where new PDFs are saved before ingestion."),
    limit:        int   = typer.Option(0,     "--limit",     help="Max new papers to download (0 = all found)."),
    resolve:      bool  = typer.Option(False, "--resolve/--no-resolve",
                                        help="Also resolve title-only references via Crossref (slow, ~0.3s each)."),
    ingest:       bool  = typer.Option(True,  "--ingest/--no-ingest",
                                        help="Ingest downloaded PDFs immediately."),
    dry_run:      bool  = typer.Option(False, "--dry-run",   help="Show what would be downloaded without doing it."),
    delay:        float = typer.Option(0.3,   "--delay",     help="Seconds between API calls."),
):
    """
    Expand the collection by following references in existing papers.

    For every paper in the collection the command:

    \\b
      1. Extracts cited references from the Crossref reference list (if the
         paper has a DOI) and from the bibliography section of the markdown.
      2. Deduplicates against papers already in the collection (by DOI /
         arXiv ID).
      3. Optionally resolves title-only references to a DOI via Crossref
         title search (--resolve, on by default).
      4. Queries Unpaywall → arXiv → Semantic Scholar for a legal open-
         access PDF.
      5. Downloads the PDF and immediately ingests it into sciknow.

    Examples:

      sciknow db expand --dry-run

      sciknow db expand --download-dir ~/papers/auto

      sciknow db expand --limit 20 --no-ingest
    """
    import sys
    import time

    from sqlalchemy import text as sql_text

    from sciknow.config import settings
    from sciknow.ingestion.downloader import find_and_download
    from sciknow.ingestion.references import (
        extract_references_from_crossref,
        extract_references_from_markdown,
    )
    from sciknow.storage.db import get_session

    download_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: load all papers and their existing DOIs/arXiv IDs ────────────
    with get_session() as session:
        papers = session.execute(sql_text("""
            SELECT pm.doi, pm.arxiv_id, pm.title, pm.crossref_raw,
                   d.mineru_output_path
            FROM paper_metadata pm
            JOIN documents d ON d.id = pm.document_id
            WHERE d.ingestion_status = 'complete'
        """)).fetchall()

        existing_dois    = {r[0].lower() for r in papers if r[0]}
        existing_arxivs  = {r[1].lower() for r in papers if r[1]}

    console.print(f"Collection: [bold]{len(papers)}[/bold] papers, "
                  f"{len(existing_dois)} with DOI, {len(existing_arxivs)} with arXiv ID")

    # ── Step 2: extract all references ───────────────────────────────────────
    all_refs: list = []
    for doi, arxiv_id, title, crossref_raw, marker_out in papers:
        # Source A: Crossref reference list (structured, reliable)
        if crossref_raw:
            all_refs.extend(extract_references_from_crossref(crossref_raw))

        # Source B: bibliography section of the markdown
        if marker_out:
            from pathlib import Path as _Path
            mp = _Path(marker_out)
            md_files = list(mp.glob("**/*.md")) if mp.exists() else []
            if md_files:
                md_text = md_files[0].read_text(encoding="utf-8", errors="replace")
                all_refs.extend(extract_references_from_markdown(md_text))

    console.print(f"Extracted [bold]{len(all_refs)}[/bold] raw reference entries.")

    # ── Step 3: deduplicate references against each other and the collection ─
    seen: set[str] = set()
    candidates = []
    for ref in all_refs:
        key = (ref.doi or "").lower() or (ref.arxiv_id or "").lower()
        if not key and not ref.title:
            continue
        # Already in collection?
        if ref.doi and ref.doi.lower() in existing_dois:
            continue
        if ref.arxiv_id and ref.arxiv_id.lower() in existing_arxivs:
            continue
        # Deduplicate within this batch
        dedup_key = key or ref.title.lower()[:60]
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        candidates.append(ref)

    console.print(f"New references not yet in collection: [bold]{len(candidates)}[/bold]")

    if not candidates:
        console.print("[green]Collection is already up to date.[/green]")
        raise typer.Exit(0)

    # ── Step 4: filter to refs that have at least a DOI or arXiv ID ──────────
    downloadable = [r for r in candidates if r.doi or r.arxiv_id]
    console.print(
        f"Downloadable (have DOI or arXiv ID): [bold]{len(downloadable)}[/bold]"
    )

    # Apply limit early so --resolve doesn't waste time on refs we won't download
    if limit:
        # Fill up to `limit` from downloadable first; resolve title-only for remaining slots
        downloadable = downloadable[:limit]
        title_limit = max(0, limit - len(downloadable))
    else:
        title_limit = 0  # unlimited if resolve is on

    # ── Step 5: optionally resolve title-only refs to DOIs ───────────────────
    if resolve:
        from sciknow.ingestion.metadata import search_crossref_by_title
        title_only = [r for r in candidates if not r.doi and not r.arxiv_id and r.title]
        if limit:
            title_only = title_only[:title_limit]
        if title_only:
            console.print(
                f"Resolving [bold]{len(title_only)}[/bold] title-only references via Crossref…"
            )
            resolved = 0
            for ref in title_only:
                meta = search_crossref_by_title(ref.title, threshold=0.85)
                if meta and meta.doi:
                    if meta.doi.lower() not in existing_dois:
                        ref.doi = meta.doi
                        downloadable.append(ref)
                        resolved += 1
                time.sleep(delay)
            console.print(f"Resolved [bold]{resolved}[/bold] additional references to DOIs.")

    if dry_run:
        console.print("\n[dim]Dry run — no downloads. Papers that would be fetched:[/dim]")
        for ref in downloadable[:30]:
            id_str = f"doi:{ref.doi}" if ref.doi else f"arXiv:{ref.arxiv_id}"
            title_str = (ref.title or "")[:55]
            console.print(f"  {id_str}  {title_str}")
        if len(downloadable) > 30:
            console.print(f"  … and {len(downloadable) - 30} more")
        raise typer.Exit(0)

    # ── Step 6: load resume state ─────────────────────────────────────────────
    # .no_oa_cache  — DOIs/arXiv IDs confirmed to have no open-access PDF.
    #                 Skipped on future runs to avoid redundant API calls.
    # .ingest_done  — identifiers whose PDF was downloaded AND successfully
    #                 ingested. Skipped entirely on re-runs.
    no_oa_cache_file  = download_dir / ".no_oa_cache"
    ingest_done_file  = download_dir / ".ingest_done"

    no_oa_cache: set[str] = set()
    ingest_done: set[str] = set()

    if no_oa_cache_file.exists():
        no_oa_cache = set(no_oa_cache_file.read_text().splitlines())
    if ingest_done_file.exists():
        ingest_done = set(ingest_done_file.read_text().splitlines())

    # ── Step 7: download and ingest ───────────────────────────────────────────
    downloaded = skipped = ingested = failed_dl = failed_ingest = 0

    log_file = download_dir / "expand.log"
    from datetime import datetime as _dt
    _run_ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")

    def _log(line: str) -> None:
        with log_file.open("a", encoding="utf-8") as _lf:
            _lf.write(line + "\n")

    _log(f"\n{'='*72}")
    _log(f"RUN  {_run_ts}  papers={len(papers)}  candidates={len(downloadable)}")
    _log(f"{'='*72}")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from sciknow.config import settings as _settings

    dl_workers = max(1, _settings.expand_download_workers)

    # Partition refs: cached skips handled first (no work), rest fed to a
    # download pool. Ingest subprocesses stay on the main thread to avoid GPU
    # contention on the 3090 (see OPTIMIZATION.md).
    def _prep(ref):
        ref_key = (ref.doi or ref.arxiv_id or "").lower()
        safe_name = (ref.doi or ref.arxiv_id or "unknown").replace("/", "_").replace(":", "_")
        dest = download_dir / f"{safe_name}.pdf"
        title = (ref.title or "")[:80]
        return ref, ref_key, dest, title

    def _download_one(ref, ref_key, dest, title):
        """I/O-bound: runs in a worker thread. Never touches caches/logs/DB."""
        if ref_key in ingest_done or ref_key in no_oa_cache:
            return ("cached", None)
        if dest.exists():
            return ("exists", None)
        ok, source = find_and_download(
            doi=ref.doi,
            arxiv_id=ref.arxiv_id,
            dest_path=dest,
            email=settings.crossref_email,
        )
        return ("downloaded" if ok else "no_oa", source)

    def _ingest_one(dest: Path) -> tuple[bool, str]:
        result = subprocess.run(
            [sys.executable, "-m", "sciknow.cli.main", "ingest", "file", str(dest)],
            capture_output=True, text=True,
        )
        err = (result.stderr or result.stdout or "").strip()[:120]
        return result.returncode == 0, err

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            f"Downloading ({dl_workers} workers)", total=len(downloadable)
        )

        prepped = [_prep(r) for r in downloadable]

        with ThreadPoolExecutor(max_workers=dl_workers) as pool:
            future_to_info = {
                pool.submit(_download_one, *info): info for info in prepped
            }

            for fut in as_completed(future_to_info):
                ref, ref_key, dest, title = future_to_info[fut]
                label = (ref.title or ref.doi or ref.arxiv_id or "")[:50]
                progress.update(task, description=f"[dim]{label}[/dim]")

                try:
                    status, source = fut.result()
                except Exception as exc:
                    failed_dl += 1
                    _log(f"ERROR  {ref_key}  | {title}  | {exc}")
                    progress.advance(task)
                    continue

                if status == "cached":
                    if ref_key in ingest_done:
                        skipped += 1
                        _log(f"SKIP   {ref_key}  | {title}")
                    else:
                        failed_dl += 1
                        _log(f"NO_OA  {ref_key}  | {title}  (cached)")
                    progress.advance(task)
                    continue

                if status == "no_oa":
                    failed_dl += 1
                    with no_oa_cache_file.open("a") as f:
                        f.write(ref_key + "\n")
                    no_oa_cache.add(ref_key)
                    _log(f"NO_OA  {ref_key}  | {title}")
                    progress.advance(task)
                    continue

                if status == "downloaded":
                    downloaded += 1
                    progress.update(task, description=f"[green]↓ {source}[/green] {label[:40]}")
                    _log(f"DL     {ref_key}  | {title}  | source={source}")

                # status in {"downloaded", "exists"} -> optionally ingest
                if ingest:
                    ok, err = _ingest_one(dest)
                    if ok:
                        ingested += 1
                        _log(f"INGEST {ref_key}  | {title}")
                        with ingest_done_file.open("a") as f:
                            f.write(ref_key + "\n")
                        ingest_done.add(ref_key)
                    else:
                        failed_ingest += 1
                        _log(f"INGEST_FAIL {ref_key}  | {title}  | {err}")
                elif status == "exists":
                    skipped += 1
                    _log(f"SKIP   {ref_key}  | {title}  (pdf on disk, --no-ingest)")

                progress.advance(task)

    _log(
        f"SUMMARY  downloaded={downloaded}  ingested={ingested}  "
        f"skipped={skipped}  no_oa={failed_dl}  ingest_failed={failed_ingest}"
    )
    console.print(
        f"\n[bold]Summary:[/bold] "
        f"[green]↓ {downloaded} downloaded[/green]  "
        f"[green]✓ {ingested} ingested[/green]  "
        f"[yellow]⏭ {skipped} already done[/yellow]  "
        f"[red]✗ {failed_dl} no OA PDF[/red]"
        + (f"  [red]✗ {failed_ingest} ingest failed[/red]" if failed_ingest else "")
    )
    console.print(f"[dim]Run log appended to {log_file}[/dim]")
    if downloaded:
        console.print(
            f"\nNew PDFs saved to [bold]{download_dir}[/bold]. "
            "Run [bold]sciknow catalog stats[/bold] to see updated counts."
        )
    cache_size = len(no_oa_cache)
    if cache_size:
        console.print(
            f"[dim]{cache_size} DOIs cached as 'no OA PDF' — will be skipped on next run.[/dim]"
        )


@app.command()
def export(
    output: Path = typer.Option(Path("finetune_dataset.jsonl"), "--output", "-o",
                                 help="Output JSONL file path."),
    generate_qa: bool = typer.Option(False, "--generate-qa",
                                      help="Use Ollama to generate Q&A pairs per chunk (slow)."),
    limit: int = typer.Option(0, "--limit", help="Max chunks to export (0 = all)."),
    min_tokens: int = typer.Option(50, "--min-tokens",
                                   help="Skip chunks shorter than this many tokens."),
):
    """
    Export the knowledge base as a fine-tuning dataset (JSONL).

    Without --generate-qa: exports each chunk with its metadata as a context entry.
    With --generate-qa: calls Ollama on each chunk to generate a (question, answer)
    pair — useful for creating instruction-tuning data. This is slow (~5-10 s/chunk).

    Output format (both modes):

    \\b
      {
        "title": "...", "year": ..., "section": "...", "doi": "...",
        "content": "...",                    # always present
        "question": "...", "answer": "..."   # only with --generate-qa
      }
    """
    from sqlalchemy import text

    from sciknow.storage.db import get_session

    with get_session() as session:
        query = text("""
            SELECT c.id::text, c.content, c.section_type, c.content_tokens,
                   pm.title, pm.year, pm.doi, pm.authors
            FROM chunks c
            JOIN paper_metadata pm ON pm.document_id = c.document_id
            WHERE c.qdrant_point_id IS NOT NULL
              AND length(c.content) > 0
            ORDER BY pm.year DESC NULLS LAST, c.document_id, c.chunk_index
        """)
        rows = session.execute(query).fetchall()

    if not rows:
        console.print("[yellow]No embedded chunks found.[/yellow]")
        raise typer.Exit(0)

    # Filter by min_tokens
    rows = [r for r in rows if (r[3] or 0) >= min_tokens]
    if limit:
        rows = rows[:limit]

    console.print(f"Exporting [bold]{len(rows)}[/bold] chunks → [bold]{output}[/bold]")
    if generate_qa:
        console.print("[dim]--generate-qa enabled: calling Ollama per chunk (this will take a while)[/dim]")

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    def _row_to_record(row) -> dict:
        chunk_id, content, section_type, tokens, title, year, doi, authors = row
        return {
            "title":   title,
            "year":    year,
            "section": section_type,
            "doi":     doi,
            "content": content,
        }

    def _generate_qa_for(record: dict) -> dict | None:
        """Runs in a worker thread. Must not touch DB or shared files."""
        from sciknow.rag import prompts
        from sciknow.rag.llm import complete
        sys_p, usr_p = prompts.finetune_qa(
            record["title"], record["year"], record["section"], record["content"]
        )
        try:
            raw = complete(sys_p, usr_p, temperature=0.3, num_ctx=4096).strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            qa = json.loads(raw)
        except Exception:
            return None
        return {
            **record,
            "question": qa.get("question", ""),
            "answer":   qa.get("answer", ""),
        }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Exporting", total=len(rows))

        with output.open("w", encoding="utf-8") as fh:
            if not generate_qa:
                # Fast path: no LLM, just stream rows to disk.
                for row in rows:
                    fh.write(json.dumps(_row_to_record(row), ensure_ascii=False) + "\n")
                    written += 1
                    progress.advance(task)
            else:
                # Concurrent LLM calls. Ollama's server-side parallelism is
                # controlled by OLLAMA_NUM_PARALLEL (default 1, set to 4+ to
                # actually see the speedup from this client-side pool).
                from concurrent.futures import ThreadPoolExecutor, as_completed
                from sciknow.config import settings as _settings

                workers = max(1, _settings.llm_parallel_workers)
                progress.update(
                    task, description=f"Generating Q&A ({workers} parallel LLM calls)"
                )

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(_generate_qa_for, _row_to_record(row)): row
                        for row in rows
                    }
                    for fut in as_completed(futures):
                        result = fut.result()
                        if result is None:
                            skipped += 1
                        else:
                            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                            written += 1
                        progress.advance(task)

    console.print(f"[green]✓ Wrote {written} records[/green]" +
                  (f", skipped {skipped}" if skipped else "") +
                  f" → {output}")
