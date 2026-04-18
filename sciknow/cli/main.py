import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

from sciknow.cli import ask as ask_module
from sciknow.cli import backup as backup_module
from sciknow.cli import book as book_module
from sciknow.cli import catalog as catalog_module
from sciknow.cli import db as db_module
from sciknow.cli import draft as draft_module
from sciknow.cli import feedback as feedback_module
from sciknow.cli import ingest as ingest_module
from sciknow.cli import project as project_module
from sciknow.cli import refresh as refresh_module
from sciknow.cli import search as search_module
from sciknow.cli import spans as spans_module
from sciknow.cli import watch as watch_module
from sciknow.cli import wiki as wiki_module
from sciknow.logging_config import setup_logging

app = typer.Typer(
    name="sciknow",
    help="Local-first scientific knowledge system.",
    no_args_is_help=True,
)
console = Console()

logger = logging.getLogger("sciknow.cli")


@app.callback()
def _startup(
    ctx: typer.Context,
    project: str = typer.Option(
        None, "--project", "-P",
        help="Override the active project for this invocation. "
             "Equivalent to setting SCIKNOW_PROJECT in the env. "
             "See `sciknow project list` for available slugs.",
    ),
) -> None:
    """Initialize logging and record the CLI invocation.

    Phase 43g — the ``--project`` root flag exports SCIKNOW_PROJECT into
    the process environment so every downstream module that reads from
    ``sciknow.core.project.get_active_project()`` picks up the override.
    Precedence (high → low): this flag → existing SCIKNOW_PROJECT env →
    ``.active-project`` file → legacy ``default`` fallback.
    """
    import os
    if project:
        # Validate eagerly so a typo fails before any subcommand runs.
        from sciknow.core.project import validate_slug
        try:
            validate_slug(project)
        except ValueError as exc:
            console.print(f"[red]--project:[/red] {exc}")
            raise typer.Exit(2)
        os.environ["SCIKNOW_PROJECT"] = project

    setup_logging()
    cmd = " ".join(sys.argv[1:]) or "(no args)"
    logger.info(f"CLI  {cmd}")


app.add_typer(backup_module.app, name="backup")
# Phase 54.6.27 — master refresh command (re-run full pipeline on new papers)
app.command(name="refresh")(refresh_module.refresh)
app.add_typer(catalog_module.app, name="catalog")
app.add_typer(db_module.app, name="db")
app.add_typer(ingest_module.app, name="ingest")
app.add_typer(search_module.app, name="search")
app.add_typer(ask_module.app, name="ask")
app.add_typer(book_module.app, name="book")
app.add_typer(draft_module.app, name="draft")
app.add_typer(wiki_module.app, name="wiki")
# Phase 43e — multi-project lifecycle commands.
app.add_typer(project_module.app, name="project")
# Phase 45 — upstream repo watchlist.
app.add_typer(watch_module.app, name="watch")
# Phase 50.B — user feedback capture (LambdaMART feedstock).
app.add_typer(feedback_module.app, name="feedback")
# Phase 50.C — local span tracer (Langfuse-pattern, no service).
app.add_typer(spans_module.app, name="spans")


@app.command(name="test")
def test_cmd(
    layer: str = typer.Option(
        "L1",
        "--layer", "-l",
        help="Which test layer to run: L1 (static), L2 (live integration), L3 (end-to-end), SMOKE (Phase 54.6.39: single-example LLM pipeline smokes only — skip the cheap L1/L2 and the utility L3 checks), or all.",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Stop on the first failure instead of running every test in the layer.",
    ),
):
    """
    Run the layered testing protocol (smoke tests, not pytest).

    L1    — Static     (seconds, no deps)              imports, prompts, signatures
    L2    — Live       (tens of sec, PG + Qdrant)      hybrid_search, raptor scrolls
    L3    — End-to-end (minutes, PG + Qdrant + Ollama) tiny LLM call, embedder +
                                                        Phase 54.6.39 single-example smokes
    SMOKE — Phase 54.6.39 single-example LLM pipeline tests ONLY (a focused
            subset of L3). Runs: num_predict cap check, extraction-model
            canary, wiki-compile-1-paper, extract-kg-1-paper, autowrite-1-iter.
            Use this after any prompt/model/num_predict change — catches the
            regressions that bulk runs would only reveal after 20-40 minutes.

    Run L1 on every PR. Run L2 before shipping a "Phase" feature drop or after
    infrastructure changes. Run L3 after retrieval / LLM / embedder changes.
    Run SMOKE after ANY change in the wiki / autowrite LLM pipelines.
    See docs/TESTING.md for the full protocol and how to add new checks.

    Examples:

      sciknow test                      # L1 only (default — fast)
      sciknow test --layer L2           # live integration only
      sciknow test --layer SMOKE        # single-example pipeline sanity (~5 min)
      sciknow test --layer all          # everything (L1 + L2 + L3)
      sciknow test -l all --fail-fast
    """
    from sciknow.testing import protocol

    layer_norm = layer.upper().strip()
    if layer_norm == "ALL":
        layers = ["L1", "L2", "L3"]
    elif layer_norm == "SMOKE":
        # Phase 54.6.39 — single-example LLM pipeline smokes only.
        # Skips the utility L3 checks (ollama reachable / llm smoke /
        # embedder loads) since they're subsumed by the pipeline tests,
        # AND skips the cheap L1/L2 checks you'd normally run first.
        layers = ["SMOKE"]
    elif layer_norm in ("L1", "L2", "L3"):
        layers = [layer_norm]
    else:
        console.print(f"[red]Unknown layer: {layer!r}. Use L1, L2, L3, SMOKE, or all.[/red]")
        raise typer.Exit(2)

    console.print(f"[bold]sciknow test[/bold] · layers: {', '.join(layers)}")
    console.print()
    n_failed = protocol.run_all(layers, fail_fast=fail_fast)

    console.print()
    if n_failed == 0:
        console.print("[bold green]✓ All tests passed[/bold green]")
        raise typer.Exit(0)
    else:
        console.print(f"[bold red]✗ {n_failed} test(s) failed[/bold red]")
        raise typer.Exit(1)


@app.command(name="bench")
def bench_cmd(
    layer: str = typer.Option(
        "fast", "--layer", "-l",
        help="Which bench layer to run: fast (descriptive only), live "
             "(adds hybrid_search + embedder + reranker), llm (adds "
             "Ollama throughput), sweep (per-model speed comparison on "
             "extract-kg / compile / write_section), quality (deep "
             "writing-quality benchmarks with NLI faithfulness + ALCE "
             "citation quality + LLM-judge pairwise), or full.",
    ),
    tag: str = typer.Option(
        "", "--tag",
        help="Free-form label stamped into the output file + latest.json.",
    ),
    compare: bool = typer.Option(
        True, "--compare/--no-compare",
        help="Diff numeric metrics against the previous run's latest.json.",
    ),
):
    """Run the benchmarking harness (performance + quality metrics).

    This is separate from ``sciknow test`` — test is pass/fail for
    correctness, bench is numbers for speed/quality. Results land as
    JSONL under ``{data_dir}/bench/<ts>.jsonl`` plus a ``latest.json``
    rollup that the next run diffs against.

    \b
    Layers:
      fast   — DB + Qdrant stats, descriptive only, no model calls (~5s).
      live   — adds 1 embedder pass + hybrid_search round trip (~30s cold).
      llm    — adds Ollama fast + main model throughput (~60–180s cold).
      sweep  — per-model comparison: runs every candidate in
               model_sweep.CANDIDATE_MODELS against extract-kg,
               compile-summary, and write-section on fixed paper 4092d6ad.
               Produces apples-to-apples metrics for picking which model
               wins each role. ~20–25 min for 6 models × 3 tasks on a 3090.
      quality — deep writing-quality benchmarks: per-model NLI-based
               faithfulness scoring, ALCE-adapted citation precision/recall,
               length + thinking-leak checks, and pairwise LLM-judge
               win-rates across 7 tasks (wiki summary, wiki polish,
               autowrite writer, book review, ask synthesize, autowrite
               scorer, wiki consensus). Uses a ~440 MB NLI cross-encoder
               (cross-encoder/nli-deberta-v3-base) for entailment,
               downloaded on first run. ~30-60 min for 3 models × 7 tasks.
      full   — every bench. Run before a release or after infra change.
    """
    from sciknow.testing import bench as bench_mod

    if layer not in bench_mod.VALID_LAYERS:
        console.print(f"[red]Unknown layer: {layer!r}. Use {list(bench_mod.VALID_LAYERS)}[/red]")
        raise typer.Exit(2)

    console.print(f"[bold]sciknow bench[/bold] · layer: [cyan]{layer}[/cyan]"
                  + (f" · tag: [dim]{tag}[/dim]" if tag else ""))
    console.print()
    results, out_path = bench_mod.run(layer=layer, tag=tag)

    diff = bench_mod.diff_against_latest(results) if compare else None
    bench_mod.render_report(results, diff=diff)

    console.print()
    console.print(f"[dim]Results written to[/dim] {out_path}")
    n_err = sum(1 for r in results if r.status == "error")
    if n_err:
        console.print(f"[yellow]{n_err} bench function(s) errored — check the JSONL for details.[/yellow]")
    raise typer.Exit(0 if n_err == 0 else 1)


@app.command(name="bench-retrieval-gen")
def bench_retrieval_gen_cmd(
    n: int = typer.Option(
        200, "-n", "--n-queries",
        help="Number of synthetic (question, source_chunk) pairs to generate.",
    ),
    model: str = typer.Option(
        None, "--model",
        help="LLM for question generation. Defaults to settings.llm_fast_model.",
    ),
    seed: int = typer.Option(
        42, "--seed",
        help="Random seed for chunk sampling (deterministic probe set).",
    ),
):
    """Phase 54.6.69 — generate the retrieval-quality probe set.

    Samples N random chunks from the corpus (length ≥ 400 chars), asks
    the fast LLM to write one specific question per chunk (answerable
    only from that passage), and persists the (question, source_chunk)
    pairs to ``<project>/data/bench/retrieval_queries.jsonl``.

    The ``b_retrieval_recall`` bench function reads this file and
    measures MRR@10 / Recall@10 / NDCG@10. Regenerate when the corpus
    changes materially (new papers, chunker version bumps). Generation
    is deterministic for a fixed ``--seed``.

    Runtime: ~2-5 min on LLM_FAST_MODEL for n=200.
    """
    from sciknow.testing import retrieval_eval
    console.print(f"[bold]Generating {n} retrieval benchmark queries…[/bold]")
    console.print(
        "[dim]This is a one-time LLM cost; the probe set is persisted "
        "and reused on every `bench --layer live` or `--layer full` "
        "run until you regenerate.[/dim]\n"
    )
    path = retrieval_eval.generate_probe_set(n=n, model=model, seed=seed)
    console.print(f"\n[green]✓ Probe set written to[/green] {path}")
    console.print(
        "[dim]Next: run `sciknow bench --layer live` to get MRR / "
        "Recall / NDCG against this probe set.[/dim]"
    )


if __name__ == "__main__":
    app()
