"""Phase 54.6.27 — `sciknow refresh` master command.

Runs the full post-ingest pipeline in one shot when new papers are
added to the inbox. Every step is idempotent / resumable, so you can
re-run this any time; steps that have no new work to do skip quickly.

Pipeline order (matches docs/README "Full rebuild sequence"):

  1a. ingest inbox/          — add new PDFs (your staging area)
  1b. ingest downloads/      — expand-discovered PDFs not yet processed
  1c. ingest failed/         — retry previously-failed PDFs (resume, no force)
  2. db enrich               — DOI backfill via Crossref / OpenAlex / arXiv / S2
  3. db link-citations       — cross-link cited_document_id for in-corpus papers
  4. db classify-papers      — paper_type tag per paper (peer_reviewed /
                               preprint / opinion / …) — 54.6.80
  5. catalog cluster         — BERTopic re-cluster (includes new papers)
  6. catalog raptor build    — hierarchical summary tree rebuild
  7. db tag-multimodal       — tag chunks containing tables / equations
  8. db extract-visuals      — extract visual elements into visuals table
  9. db caption-visuals      — VLM captions for figures + charts — 54.6.72
                               (needs a VLM pulled; skipped cleanly if not)
  10. db paraphrase-equations — natural-language paraphrases for retrieval
                                of LaTeX equations — 54.6.78
  11. db embed-visuals       — embed captions + paraphrases into the
                               visuals Qdrant collection — 54.6.82
  12. wiki compile           — paper summaries + concept stubs + KG triples

Every step is idempotent and **does not force rebuilds** — each one
skips rows that are already done. Pass ``--rebuild`` directly to the
underlying command if you want a force rebuild (e.g. ``sciknow wiki
compile --rebuild`` — refresh will never add that flag for you).

Use ``--no-<step>`` flags to skip expensive steps you don't need this
round (e.g. ``--no-wiki`` skips the hours-long LLM compile).

Phase 54.6.86 — added the post-ingest enrichment steps (classify,
caption, paraphrase, embed-visuals) so one `refresh` run produces a
corpus ready for retrieval + all the new 54.6.78-82 features. Prior
to this, each of those commands had to be remembered and run
separately.
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
    no_retraction_sweep: bool = typer.Option(False, "--no-retraction-sweep",
        help="Skip Crossref retraction sweep (54.6.111). Runs weekly "
             "by default on any paper not checked in the last 30 days."),
    no_citations: bool = typer.Option(False, "--no-citations"),
    no_classify: bool = typer.Option(False, "--no-classify",
        help="Skip paper-type classification (54.6.80)."),
    no_cluster: bool = typer.Option(False, "--no-cluster",
        help="Skip BERTopic re-cluster (keeps existing topic assignments)."),
    no_raptor: bool = typer.Option(False, "--no-raptor",
        help="Skip RAPTOR tree rebuild (expensive on large corpora)."),
    no_multimodal: bool = typer.Option(False, "--no-multimodal"),
    no_visuals: bool = typer.Option(False, "--no-visuals"),
    no_caption: bool = typer.Option(False, "--no-caption",
        help="Skip VLM caption-visuals (54.6.72) — skipped anyway if "
             "no vision LLM is pulled."),
    caption_model: str = typer.Option(
        "qwen2.5vl:7b", "--caption-model",
        help="Vision-LLM for the bulk caption pass. Default qwen2.5vl:7b — "
             "54.6.89 VLM sweep showed it's ~60% of the 32b judge-win "
             "quality but ~35× faster (1.85s vs 65s per caption), which "
             "matters at 9,000+ image scale (5h vs 7 days on a 3090). "
             "Passthrough to `db caption-visuals --model` — set to "
             "`qwen2.5vl:32b` for premium quality if you can afford the "
             "wall time, or to an empty string to use that CLI's default "
             "(currently qwen2.5vl:32b)."),
    no_paraphrase: bool = typer.Option(False, "--no-paraphrase",
        help="Skip equation paraphrase backfill (54.6.78)."),
    no_parse_tables: bool = typer.Option(False, "--no-parse-tables",
        help="Skip structured table parsing (54.6.106). ~2 min for 1.5k tables."),
    no_embed_visuals: bool = typer.Option(False, "--no-embed-visuals",
        help="Skip visuals Qdrant embedding (54.6.82). Only useful if "
             "caption + paraphrase populated ai_caption already."),
    no_wiki: bool = typer.Option(False, "--no-wiki",
        help="Skip wiki compile (the hours-long LLM step)."),
    dry_run: bool = typer.Option(False, "--dry-run",
        help="Print the steps that would run without executing them."),
    budget_time: str = typer.Option(
        None, "--budget-time",
        help="Phase 54.6.206 — soft wall-clock budget. Checked between "
             "steps; when exceeded, refresh finishes the current step "
             "and exits gracefully (non-zero) so scheduled overnight "
             "runs have a known cutoff. Accepts s/m/h suffix or plain "
             "seconds: --budget-time=6h, --budget-time=30m, "
             "--budget-time=3600. Already-completed steps remain "
             "idempotent; re-running picks up the unfinished plan.",
    ),
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
        # Phase 54.6.56 — also sweep downloads/ (expand-created PDFs that
        # never got moved to processed/ or failed/, usually because the
        # expand run crashed between download and ingest) and failed/
        # (previous ingestion errors — re-ingesting resumes them from
        # whatever stage they failed at, it does NOT force a re-do of
        # completed papers thanks to the SHA-256 skip in pipeline.ingest).
        def _count_pdfs(p: Path) -> int:
            if not p.exists():
                return 0
            return len(list(p.glob("**/*.pdf")))

        n_inbox = _count_pdfs(papers_dir)
        downloads_dir = active.data_dir / "downloads"
        failed_dir = active.data_dir / "failed"
        n_downloads = _count_pdfs(downloads_dir)
        n_failed = _count_pdfs(failed_dir)

        if n_inbox:
            steps.append((
                f"1a. Ingest {n_inbox} PDF(s) from inbox ({papers_dir})",
                ["ingest", "directory", str(papers_dir)],
                False,
            ))
        elif papers_dir.exists():
            console.print(f"[dim]Inbox has no PDFs — skipping inbox ingest[/dim]")
        else:
            console.print(f"[dim]Inbox missing — skipping inbox ingest. "
                          f"Create {papers_dir} and drop PDFs there.[/dim]")

        if n_downloads:
            steps.append((
                f"1b. Ingest {n_downloads} PDF(s) from downloads "
                f"(expand-discovered, not yet moved to processed/)",
                ["ingest", "directory", str(downloads_dir)],
                True,  # optional — downloads/ is auxiliary
            ))
        elif downloads_dir.exists():
            console.print(f"[dim]downloads/ has no PDFs — nothing to ingest[/dim]")

        if n_failed:
            steps.append((
                f"1c. Retry {n_failed} previously-failed PDF(s) from failed/ "
                f"(resume from last stage, no force)",
                ["ingest", "directory", str(failed_dir)],
                True,  # optional — retrying failures shouldn't block the rest
            ))
        elif failed_dir.exists():
            console.print(f"[dim]failed/ has no PDFs — nothing to retry[/dim]")

        if any([n_inbox, n_downloads, n_failed]):
            console.print()

    if not no_enrich:
        steps.append(("2. DOI enrichment (Crossref/OpenAlex/arXiv/S2)",
                      ["db", "enrich"], True))
    if not no_retraction_sweep:
        # Phase 54.6.111 — weekly-by-default retraction sweep against
        # Crossref's update-type index. Skips papers checked within 30d.
        steps.append(("2a. Retraction sweep (flag retracted/corrected)",
                      ["db", "refresh-retractions"], True))
    if not no_citations:
        steps.append(("3. Link citations",
                      ["db", "link-citations"], True))
    if not no_classify:
        # Phase 54.6.80 — skips rows that already have paper_type set,
        # so re-runs are cheap (only new/unclassified papers).
        steps.append(("4. Classify paper types (peer_reviewed / "
                      "preprint / opinion / …)",
                      ["db", "classify-papers"], True))
    if not no_cluster:
        steps.append(("5. BERTopic clustering",
                      ["catalog", "cluster"], True))
    if not no_raptor:
        steps.append(("6. RAPTOR tree build",
                      ["catalog", "raptor", "build"], True))
    if not no_multimodal:
        steps.append(("7. Tag multimodal chunks",
                      ["db", "tag-multimodal"], True))
    if not no_visuals:
        steps.append(("8. Extract visuals",
                      ["db", "extract-visuals"], True))
    if not no_caption:
        # Phase 54.6.72 — VLM captions for figures + charts. Fails
        # fast + cleanly when no VLM is pulled; optional step so
        # that's not a refresh blocker.
        # 54.6.89 — default to qwen2.5vl:7b for bulk (60% judge-win
        # quality at 35× the speed; co-resident with LLM_MODEL, no
        # model swap). Pass --caption-model "" to use the
        # caption-visuals default (qwen2.5vl:32b).
        caption_argv = ["db", "caption-visuals"]
        if caption_model and caption_model.strip():
            caption_argv += ["--model", caption_model.strip()]
        steps.append((f"9. Caption figures + charts (VLM — skipped "
                      f"if '{caption_model or 'default'}' isn't pulled)",
                      caption_argv, True))
    if not no_paraphrase:
        # Phase 54.6.78 — one-sentence prose paraphrase per equation
        # for retrieval. Skips rows that already have ai_caption set.
        steps.append(("10. Paraphrase equations (LaTeX → prose)",
                      ["db", "paraphrase-equations"], True))
    if not no_parse_tables:
        # Phase 54.6.106 — structured parse of MinerU HTML tables into
        # {title, headers, summary, n_rows, n_cols}. Powers the Visuals
        # modal table cards + downstream table retrieval.
        steps.append(("11. Parse tables (HTML → summary + headers)",
                      ["db", "parse-tables"], True))
    if not no_embed_visuals:
        # Phase 54.6.82 — embeds ai_caption into the visuals Qdrant
        # collection. Depends on caption-visuals + paraphrase-equations
        # having populated ai_caption first (same refresh pass does
        # both steps earlier).
        steps.append(("12. Embed visuals into Qdrant",
                      ["db", "embed-visuals"], True))
    if not no_wiki:
        steps.append(("13. Wiki compile (slowest)",
                      ["wiki", "compile"], True))

    if not steps:
        console.print("[yellow]No steps to run — all --no-* flags set.[/yellow]")
        return

    console.print(f"[bold]Plan: {len(steps)} step(s)[/bold]")
    for label, argv, _opt in steps:
        console.print(f"  [dim]→[/dim] {label}")
    console.print()

    # Phase 54.6.206 — parse --budget-time into seconds. Done BEFORE
    # the dry-run early return so a malformed value is caught in the
    # preview pass, not only on the real execution.
    budget_seconds: float | None = None
    if budget_time:
        try:
            bt = budget_time.strip().lower()
            if bt.endswith("h"):
                budget_seconds = float(bt[:-1]) * 3600
            elif bt.endswith("m"):
                budget_seconds = float(bt[:-1]) * 60
            elif bt.endswith("s"):
                budget_seconds = float(bt[:-1])
            else:
                budget_seconds = float(bt)
        except ValueError:
            console.print(
                f"[red]Invalid --budget-time:[/red] {budget_time!r} — "
                "expected a number with optional s/m/h suffix "
                "(e.g. 6h, 30m, 3600)."
            )
            raise typer.Exit(2)
        console.print(
            f"[dim]Budget: {budget_seconds:.0f}s "
            f"(~{budget_seconds / 3600:.1f}h). Checked between steps.[/dim]"
        )

    if dry_run:
        console.print("[dim]Dry run — nothing executed.[/dim]")
        return

    t_total = time.monotonic()
    n_done = 0
    n_failed = 0
    n_budget_skipped = 0
    for label, argv, optional in steps:
        # Budget gate runs BEFORE the step, not after, so the user
        # always gets a clean "stopped before X" message rather than
        # "already ran X and then realised the budget was blown."
        if budget_seconds is not None:
            elapsed_so_far = time.monotonic() - t_total
            if elapsed_so_far >= budget_seconds:
                remaining_steps = len(steps) - n_done - n_failed
                console.print(
                    f"[yellow]⏱ Budget exceeded "
                    f"({elapsed_so_far:.0f}s ≥ {budget_seconds:.0f}s) — "
                    f"stopping before: {label}.[/yellow]"
                )
                console.print(
                    f"[dim]Skipped {remaining_steps} remaining step(s). "
                    f"Re-run `sciknow refresh` any time — completed steps "
                    f"are idempotent; only the unfinished work replays.[/dim]"
                )
                n_budget_skipped = remaining_steps
                break
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
    if n_budget_skipped:
        console.print(
            f"[bold yellow]⏱ Refresh stopped at budget:[/bold yellow] "
            f"{n_done}/{len(steps)} step(s) completed in {t_str}, "
            f"{n_budget_skipped} skipped. Re-run to continue."
        )
        # Distinct non-zero exit for cron/scripts so "budget hit"
        # is distinguishable from "fully completed" and "hard-failed".
        raise typer.Exit(3)
    console.print(
        f"[bold green]✓ Refresh complete:[/bold green] "
        f"{n_done}/{len(steps)} step(s) in {t_str}"
        + (f"  [yellow]({n_failed} optional step(s) had warnings)[/yellow]"
           if n_failed else "")
    )
