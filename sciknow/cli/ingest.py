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
    from sciknow.cli import preflight
    from sciknow.ingestion.pipeline import AlreadyIngested, PipelineError, ingest

    preflight()

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
    force: bool = False,
    lock: "threading.Lock | None" = None,
    ingest_source: str = "seed",
    on_file_done=None,
) -> None:
    """
    Drive one or more worker subprocesses through the PDF list.

    Each worker runs `sciknow/ingestion/worker.py` and processes files until it
    either finishes its queue or crashes (SIGABRT / SIGSEGV / any non-zero exit).
    On crash, the in-flight file is marked failed and a fresh worker picks up
    where the previous one left off.

    When called from multiple threads (parallel ingestion), pass a shared
    `lock` — it guards the results dict and failed_files list. rich.Progress
    is already thread-safe for advance/update calls.

    `ingest_source` is propagated to the worker via SCIKNOW_INGEST_SOURCE env
    var; the worker passes it to pipeline.ingest() which stamps new documents
    with it on first insert. 'seed' = manual CLI ingest, 'expand' = auto-
    discovered via `db expand`.

    `on_file_done` is an optional callback invoked from the parsing thread
    with (file_path, status, error_msg|None) for each file that reaches a
    terminal state (done / skipped / failed / crash). Lets callers attach
    per-file logging (e.g. expand.log with ref_key + title) without extending
    this function's signature further.
    """
    import contextlib
    _lock_ctx = lock if lock is not None else contextlib.nullcontext()

    def _notify(path, status, error=None):
        if on_file_done is not None:
            try:
                on_file_done(path, status, error)
            except Exception:
                pass  # never let a user callback bring down the worker loop

    remaining = list(pdfs)

    while remaining:
        env = {**__import__("os").environ}
        if force:
            env["SCIKNOW_FORCE_INGEST"] = "1"
        env["SCIKNOW_INGEST_SOURCE"] = ingest_source

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
                with _lock_ctx:
                    results["done"] += 1
                processed.add(file_path)
                progress.advance(task)
                _notify(file_path, "done", None)
            elif status == "skipped":
                with _lock_ctx:
                    results["skipped"] += 1
                processed.add(file_path)
                progress.advance(task)
                _notify(file_path, "skipped", None)
            elif status == "failed":
                err = msg.get("error", "")[:120]
                with _lock_ctx:
                    results["failed"] += 1
                    failed_files.append((file_path.name, err))
                processed.add(file_path)
                progress.advance(task)
                _notify(file_path, "failed", err)

        feeder.join()
        proc.wait()

        # If the worker crashed (non-zero exit) while processing a file,
        # that file was never reported as done/failed — mark it now.
        if proc.returncode != 0 and current_file and current_file not in processed:
            err = f"worker crashed (exit {proc.returncode})"
            with _lock_ctx:
                results["failed"] += 1
                failed_files.append((current_file.name, err))
            processed.add(current_file)
            progress.advance(task)
            _notify(current_file, "failed", err)

        remaining = [p for p in remaining if p not in processed]


def _run_parallel_workers(
    pdfs: list[Path],
    progress,
    task,
    results: dict,
    failed_files: list[tuple[str, str]],
    force: bool,
    num_workers: int,
    ingest_source: str = "seed",
    on_file_done=None,
) -> None:
    """
    Fan `pdfs` across `num_workers` concurrent worker subprocesses.

    Each bucket gets its own Python thread driving its own worker subprocess.
    Buckets are round-robin so slow PDFs distribute evenly. A single Lock
    guards shared counters and the failure list; rich.Progress is thread-safe.

    `ingest_source` and `on_file_done` are forwarded to every worker loop.
    """
    if num_workers <= 1 or len(pdfs) == 1:
        _run_worker_loop(
            pdfs, progress, task, results, failed_files,
            force=force, ingest_source=ingest_source, on_file_done=on_file_done,
        )
        return

    buckets = [pdfs[i::num_workers] for i in range(num_workers)]
    buckets = [b for b in buckets if b]

    lock = threading.Lock()
    threads: list[threading.Thread] = []
    for bucket in buckets:
        t = threading.Thread(
            target=_run_worker_loop,
            args=(bucket, progress, task, results, failed_files),
            kwargs={
                "force": force,
                "lock": lock,
                "ingest_source": ingest_source,
                "on_file_done": on_file_done,
            },
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


@app.command()
def directory(
    path: Path = typer.Argument(..., help="Directory containing PDF files."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R"),
    force: bool = typer.Option(False, "--force", "-f"),
    workers: int = typer.Option(
        0, "--workers", "-w",
        help="Parallel ingestion worker subprocesses (0 = use INGEST_WORKERS "
             "from .env, default 1). Each worker loads its own MinerU (~7GB "
             "VRAM) + bge-m3 (~2.2GB). On a 24GB GPU with an LLM resident, "
             "keep at 1. Raise to 2 only when the LLM is off-GPU.",
    ),
):
    """Ingest all PDFs in a directory."""
    from sciknow.cli import preflight
    from sciknow.config import settings

    preflight()

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

    num_workers = workers if workers > 0 else max(1, settings.ingest_workers)
    num_workers = min(num_workers, len(pdfs))

    worker_note = f" ({num_workers} workers)" if num_workers > 1 else ""
    console.print(f"Found [bold]{len(pdfs)}[/bold] PDF(s) in {path}{worker_note}")

    # Phase 54.6.31 — warm the fast model (used for metadata fallback)
    # before the ingest loop so the first paper doesn't pay cold-start.
    # Best-effort; ingest will still work if Ollama is unreachable
    # since Layer 4 (LLM) only runs when the other 3 layers miss.
    from sciknow.rag.llm import warm_up as _llm_warm_up
    _llm_warm_up(model=settings.llm_fast_model, num_ctx=4096, num_batch=1024)

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
        _run_parallel_workers(
            pdfs, progress, task, results, failed_files,
            force=force, num_workers=num_workers,
        )

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
