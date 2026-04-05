import json
import subprocess
import sys
import threading
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

app = typer.Typer(help="Ingest PDFs into the knowledge base.")
console = Console()


@app.command()
def file(
    path: Path = typer.Argument(..., help="Path to a PDF file."),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already processed."),
):
    """Ingest a single PDF file."""
    from sciknow.ingestion.pipeline import AlreadyIngested, PipelineError, ingest

    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(1)

    if path.suffix.lower() != ".pdf":
        console.print(f"[red]Not a PDF (unrecognised extension '{path.suffix}'):[/red] {path}")
        raise typer.Exit(1)

    console.print(f"Ingesting [bold]{path.name}[/bold]...")

    try:
        doc_id = ingest(path, force=force)
        console.print(f"[green]✓ Done.[/green] Document ID: [dim]{doc_id}[/dim]")
    except AlreadyIngested as e:
        console.print(f"[yellow]Already ingested[/yellow] (id={e.document_id}). Use --force to re-ingest.")
    except PipelineError as e:
        console.print(f"[red]✗ Failed:[/red] {e}")
        raise typer.Exit(1)


def _run_worker_loop(
    pdfs: list[Path],
    progress,
    task,
    results: dict,
    failed_files: list[tuple[str, str]],
) -> None:
    """
    Drive one or more worker subprocesses through the PDF list.

    Each worker runs `sciknow/ingestion/worker.py` and processes files until it
    either finishes its queue or crashes (SIGABRT / SIGSEGV / any non-zero exit).
    On crash, the in-flight file is marked failed and a fresh worker picks up
    where the previous one left off.
    """
    remaining = list(pdfs)

    while remaining:
        env = {**__import__("os").environ}
        if force:
            env["SCIKNOW_FORCE_INGEST"] = "1"

        proc = subprocess.Popen(
            [sys.executable, "-m", "sciknow.ingestion.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )

        # Write paths to the worker's stdin in a background thread to avoid
        # deadlock (pipe buffer ~64 KB; 500 paths * ~150 chars ≈ 75 KB).
        def _feed_stdin(proc, paths):
            try:
                for p in paths:
                    proc.stdin.write(str(p) + "\n")
                proc.stdin.close()
            except BrokenPipeError:
                pass  # worker died early

        feeder = threading.Thread(target=_feed_stdin, args=(proc, remaining), daemon=True)
        feeder.start()

        current_file: Path | None = None
        processed: set[Path] = set()

        for raw_line in proc.stdout:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                msg = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            file_path = Path(msg["file"])
            status = msg["status"]

            if status == "started":
                current_file = file_path
                progress.update(task, description=f"[dim]{file_path.name[:50]}[/dim]")
            elif status == "done":
                results["done"] += 1
                processed.add(file_path)
                progress.advance(task)
            elif status == "skipped":
                results["skipped"] += 1
                processed.add(file_path)
                progress.advance(task)
            elif status == "failed":
                results["failed"] += 1
                processed.add(file_path)
                failed_files.append((file_path.name, msg.get("error", "")[:120]))
                progress.advance(task)

        feeder.join()
        proc.wait()

        # If the worker crashed (non-zero exit) while processing a file,
        # that file was never reported as done/failed — mark it now.
        if proc.returncode != 0 and current_file and current_file not in processed:
            results["failed"] += 1
            failed_files.append((
                current_file.name,
                f"worker crashed (exit {proc.returncode})",
            ))
            processed.add(current_file)
            progress.advance(task)

        remaining = [p for p in remaining if p not in processed]


@app.command()
def directory(
    path: Path = typer.Argument(..., help="Directory containing PDF files."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R"),
    force: bool = typer.Option(False, "--force", "-f"),
):
    """Ingest all PDFs in a directory."""

    if not path.is_dir():
        console.print(f"[red]Not a directory:[/red] {path}")
        raise typer.Exit(1)

    glob_fn = path.rglob if recursive else path.glob
    pdfs = sorted(
        p for p in glob_fn("*")
        if p.is_file() and p.suffix.lower() == ".pdf"
    )

    if not pdfs:
        console.print("[yellow]No PDF files found.[/yellow]")
        raise typer.Exit(0)

    console.print(f"Found [bold]{len(pdfs)}[/bold] PDF(s) in {path}")

    results = {"done": 0, "skipped": 0, "failed": 0}
    failed_files: list[tuple[str, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Ingesting", total=len(pdfs))
        _run_worker_loop(pdfs, progress, task, results, failed_files)

    # Summary table
    table = Table(title="Ingestion Summary", show_header=False)
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("[green]Completed[/green]", str(results["done"]))
    table.add_row("[yellow]Skipped (duplicate)[/yellow]", str(results["skipped"]))
    table.add_row("[red]Failed[/red]", str(results["failed"]))
    console.print(table)

    if failed_files:
        console.print("\n[red]Failed files:[/red]")
        for name, err in failed_files:
            console.print(f"  [dim]{name}[/dim]: {err}")
