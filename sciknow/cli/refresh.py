"""Phase 54.6.27 — `sciknow refresh` master command.

Runs the full post-ingest pipeline in one shot when new papers are
added to the inbox. Every step is idempotent / resumable, so you can
re-run this any time; steps that have no new work to do skip quickly.

Pipeline order (matches docs/README "Full rebuild sequence"):

  1. ingest directory        — add new PDFs to Postgres + Qdrant
  2. db enrich               — DOI backfill via Crossref / OpenAlex / arXiv / Semantic Scholar
  3. db link-citations       — cross-link cited_document_id for in-corpus papers
  4. catalog cluster         — BERTopic re-cluster (includes new papers)
  5. catalog raptor build    — hierarchical summary tree rebuild
  6. db tag-multimodal       — tag chunks containing tables / equations
  7. db extract-visuals      — extract visual elements into visuals table
  8. wiki compile            — paper summaries + concept stubs + KG triples

Use ``--no-<step>`` flags to skip expensive steps you don't need this
round (e.g. ``--no-wiki`` skips the hours-long LLM compile).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.rule import Rule

console = Console()


def _sciknow_bin() -> str:
    """Absolute path to the CLI entry point in the active venv."""
    here = Path(__file__).resolve()
    # sciknow/cli/refresh.py → sciknow/cli → sciknow → repo
    repo = here.parents[2]
    venv_bin = repo / ".venv" / "bin" / "sciknow"
    if venv_bin.exists():
        return str(venv_bin)
    which = shutil.which("sciknow")
    if which:
        return which
    return "sciknow"


def _run_step(label: str, argv: list[str], optional: bool = False) -> bool:
    """Run one pipeline step, streaming output. Returns True on success."""
    console.print(Rule(f"[bold]{label}[/bold]"))
    t0 = time.monotonic()
    cmd = [_sciknow_bin()] + argv
    try:
        res = subprocess.run(cmd, check=False)
        elapsed = time.monotonic() - t0
        if res.returncode == 0:
            console.print(f"[green]✓ {label} ({elapsed:.1f}s)[/green]\n")
            return True
        if optional:
            console.print(f"[yellow]⚠ {label} failed (exit {res.returncode}) — "
                          f"continuing because step is optional[/yellow]\n")
            return True
        console.print(f"[red]✗ {label} failed (exit {res.returncode})[/red]\n")
        return False
    except Exception as exc:
        if optional:
            console.print(f"[yellow]⚠ {label} raised: {exc} — continuing[/yellow]\n")
            return True
        console.print(f"[red]✗ {label} raised: {exc}[/red]\n")
        return False


def refresh(
    papers_dir: Path = typer.Option(
        None, "--papers-dir", "-d",
        help="Directory containing PDFs to ingest. Default: "
             "projects/<active-slug>/data/inbox/ (or data/inbox/ for legacy).",
    ),
    no_ingest: bool = typer.Option(False, "--no-ingest",
        help="Skip the ingest step (use when no new PDFs, just reindex)."),
    no_enrich: bool = typer.Option(False, "--no-enrich"),
    no_citations: bool = typer.Option(False, "--no-citations"),
    no_cluster: bool = typer.Option(False, "--no-cluster",
        help="Skip BERTopic re-cluster (keeps existing topic assignments)."),
    no_raptor: bool = typer.Option(False, "--no-raptor",
        help="Skip RAPTOR tree rebuild (expensive on large corpora)."),
    no_multimodal: bool = typer.Option(False, "--no-multimodal"),
    no_visuals: bool = typer.Option(False, "--no-visuals"),
    no_wiki: bool = typer.Option(False, "--no-wiki",
        help="Skip wiki compile (the hours-long LLM step)."),
    dry_run: bool = typer.Option(False, "--dry-run",
        help="Print the steps that would run without executing them."),
):
    """Re-run the full post-ingest pipeline after adding new papers.

    Every step is idempotent / resumable. Safe to run any time.

    Examples:

      sciknow refresh                        # full pipeline, default inbox
      sciknow refresh --no-wiki              # everything except wiki compile
      sciknow refresh --papers-dir ~/pdfs    # custom ingest source
      sciknow refresh --no-ingest            # reindex only (no new PDFs)
      sciknow refresh --dry-run              # preview the plan
    """
    from sciknow.cli import preflight
    preflight()

    from sciknow.core.project import get_active_project
    active = get_active_project()

    # Resolve ingest source
    if papers_dir is None:
        papers_dir = active.data_dir / "inbox"

    console.print(f"[bold]Project:[/bold] {active.slug}")
    console.print(f"[bold]Inbox:[/bold] {papers_dir}"
                  f"{'  [dim](missing)[/dim]' if not papers_dir.exists() else ''}")
    console.print()

    # Build the plan
    steps: list[tuple[str, list[str], bool]] = []  # (label, argv, optional)

    if not no_ingest:
        if papers_dir.exists():
            n_pdfs = len(list(papers_dir.glob("**/*.pdf")))
            if n_pdfs:
                steps.append((
                    f"1. Ingest {n_pdfs} PDF(s) from {papers_dir}",
                    ["ingest", "directory", str(papers_dir)],
                    False,
                ))
            else:
                console.print(f"[dim]Inbox has no PDFs — skipping ingest[/dim]\n")
        else:
            console.print(f"[dim]Inbox missing — skipping ingest. "
                          f"Create {papers_dir} and drop PDFs there.[/dim]\n")

    if not no_enrich:
        steps.append(("2. DOI enrichment (Crossref/OpenAlex/arXiv/S2)",
                      ["db", "enrich"], True))
    if not no_citations:
        steps.append(("3. Link citations",
                      ["db", "link-citations"], True))
    if not no_cluster:
        steps.append(("4. BERTopic clustering",
                      ["catalog", "cluster"], True))
    if not no_raptor:
        steps.append(("5. RAPTOR tree build",
                      ["catalog", "raptor", "build"], True))
    if not no_multimodal:
        steps.append(("6. Tag multimodal chunks",
                      ["db", "tag-multimodal"], True))
    if not no_visuals:
        steps.append(("7. Extract visuals",
                      ["db", "extract-visuals"], True))
    if not no_wiki:
        steps.append(("8. Wiki compile (slowest)",
                      ["wiki", "compile"], True))

    if not steps:
        console.print("[yellow]No steps to run — all --no-* flags set.[/yellow]")
        return

    console.print(f"[bold]Plan: {len(steps)} step(s)[/bold]")
    for label, argv, _opt in steps:
        console.print(f"  [dim]→[/dim] {label}")
    console.print()

    if dry_run:
        console.print("[dim]Dry run — nothing executed.[/dim]")
        return

    t_total = time.monotonic()
    n_done = 0
    n_failed = 0
    for label, argv, optional in steps:
        ok = _run_step(label, argv, optional=optional)
        if ok:
            n_done += 1
        else:
            n_failed += 1
            if not optional:
                console.print(
                    f"[red]Refresh aborted at step: {label}[/red]\n"
                    f"[dim]Fix the error and re-run. Completed steps are "
                    f"idempotent and will skip on the next run.[/dim]"
                )
                raise typer.Exit(1)

    elapsed = time.monotonic() - t_total
    if elapsed > 3600:
        t_str = f"{elapsed / 3600:.1f}h"
    elif elapsed > 60:
        t_str = f"{elapsed / 60:.1f}m"
    else:
        t_str = f"{elapsed:.1f}s"
    console.print(Rule())
    console.print(
        f"[bold green]✓ Refresh complete:[/bold green] "
        f"{n_done}/{len(steps)} step(s) in {t_str}"
        + (f"  [yellow]({n_failed} optional step(s) had warnings)[/yellow]"
           if n_failed else "")
    )
