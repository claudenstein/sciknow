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
from sciknow.cli import corpus as corpus_module
from sciknow.cli import infer as infer_module
from sciknow.cli import ingest as ingest_module
from sciknow.cli import library as library_module
from sciknow.cli import project as project_module
from sciknow.cli import refresh as refresh_module
from sciknow.cli import search as search_module
from sciknow.cli import spans as spans_module
from sciknow.cli import watch as watch_module
from sciknow.cli import wiki as wiki_module
from sciknow.logging_config import setup_logging

app = typer.Typer(
    name="sciknow",
    help=(
        "Local-first scientific knowledge system. v2 substrate: "
        "writer + embedder + reranker on llama-server. Bring up with "
        "`sciknow infer up --role <r>`; verify with `sciknow library "
        "doctor`. See `sciknow --version` for the installed version."
    ),
    no_args_is_help=True,
)
console = Console()

logger = logging.getLogger("sciknow.cli")


def _version_callback(value: bool) -> None:
    """Eager `--version` handler — prints version + exits before any
    subcommand fires (so it works even on a broken install where
    `sciknow library doctor` would error)."""
    if value:
        from sciknow import __version__
        console.print(f"sciknow {__version__}")
        raise typer.Exit(0)


@app.callback()
def _startup(
    ctx: typer.Context,
    project: str = typer.Option(
        None, "--project", "-P",
        help="Override the active project for this invocation. "
             "Equivalent to setting SCIKNOW_PROJECT in the env. "
             "See `sciknow project list` for available slugs.",
    ),
    _version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Print the installed sciknow version and exit.",
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
# v2 Phase F — `library` + `corpus` are the spec'd v2 namespaces;
# `db` is retained as a deprecation shim that prints a one-shot
# warning then dispatches to the same callables (db_module.app is
# still the implementation source).
app.add_typer(library_module.app, name="library")
app.add_typer(corpus_module.app, name="corpus")
app.add_typer(db_module.app, name="db")
app.add_typer(ingest_module.app, name="ingest")
app.add_typer(search_module.app, name="search")
app.add_typer(ask_module.app, name="ask")
app.add_typer(book_module.app, name="book")
app.add_typer(draft_module.app, name="draft")
app.add_typer(wiki_module.app, name="wiki")
# v2 Phase A — manage llama-server (writer/embedder/reranker subprocesses).
app.add_typer(infer_module.app, name="infer")
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
    See docs/reference/TESTING.md for the full protocol and how to add new checks.

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
             "(adds hybrid_search + embedder + reranker), llm (writer "
             "throughput via the dispatch facade — backend tag in output), "
             "v2 (V2_FINAL Stage 3: Decision Gates A/B/D against the "
             "llama-server substrate), sweep (per-model speed comparison "
             "on extract-kg / compile / write_section), quality (deep "
             "writing-quality benchmarks with NLI faithfulness + ALCE "
             "citation quality + LLM-judge pairwise), vlm-sweep "
             "(54.6.74 — VLM captioning sweep with pairwise judge), "
             "or full.",
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
      llm    — fast + main model throughput via the rag.llm dispatch facade
               (routes to llama-server when USE_LLAMACPP_WRITER=True, Ollama
               otherwise). Output stamps the backend so the operator sees
               which path was timed (~60–180s cold).
      v2     — V2_FINAL Stage 3: Decision Gates A (writer tps via
               infer.client.chat_complete, appended to writer_tps.jsonl),
               B (embedder throughput via infer.client.embed), D
               (retrieval recall@10 against the synthetic probe set;
               run `sciknow bench retrieval-gen` first if missing).
               (~5–10 min for 5 writer iters + 256-chunk embed batch.)
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
      specter2 — SPECTER2 rerank measurement (54.6.121-122). Runs the
               54.6.69 retrieval probe set through hybrid_search top-50
               → SPECTER2 rerank, reports MRR@10 / Recall@1 / Recall@10
               / NDCG@10 vs the bge-m3 baseline + ship-decision verdict
               (criterion: delta ≥ +0.06 MRR). ~20 s for 200 probes
               once the SPECTER2 model is cached. Currently PARKED per
               docs/research/EXPAND_ENRICH_RESEARCH_2.md §2.2; this layer exists
               so future re-tests against new releases are one-shot.
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


# ── bench-snapshot + bench-diff (Phase 54.6.224 — roadmap 3.11.5) ────────────


def _git_head_info() -> dict[str, str | bool]:
    """Capture HEAD SHA + branch + dirty flag for bench snapshots.

    Returns all-empty strings if the repo root isn't git-managed — the
    snapshot still lands, just without the commit stamp. Branch falls
    back to 'HEAD' when detached.
    """
    import subprocess
    info = {"sha": "", "branch": "", "dirty": False}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["sha"] = sha
    except Exception:
        return info
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["branch"] = branch
    except Exception:
        pass
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["dirty"] = bool(dirty)
    except Exception:
        pass
    return info


@app.command(name="bench-snapshot")
def bench_snapshot_cmd(
    layer: str = typer.Option(
        "live", "--layer", "-l",
        help="Bench layer to snapshot. Default 'live' — adds embedder + "
             "hybrid_search + reranker on top of 'fast', which is what "
             "most retrieval regressions would show up in. Use 'full' "
             "before a release or after a model swap.",
    ),
    tag: str = typer.Option(
        "", "--tag",
        help="Free-form label (prefix of the snapshot filename).",
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir",
        help="Override the snapshot directory. Default: "
             "<data_dir>/bench/snapshots/.",
    ),
):
    """Phase 54.6.224 (roadmap 3.11.5) — per-commit bench snapshot.

    Runs a bench layer and persists the results to a stable file
    named ``<sha>[-dirty]-<timestamp>[-tag].json`` under
    ``<data_dir>/bench/snapshots/``. Enables longitudinal tracking:
    run this once per commit you care about, then
    ``sciknow bench-diff <sha_a> <sha_b>`` reports per-metric deltas
    and flags regressions.

    Difference from plain ``sciknow bench``:
      * plain bench writes a timestamped JSONL + updates latest.json
        so the NEXT run diffs against it;
      * bench-snapshot writes a git-SHA-stamped JSON file kept
        forever, so you can diff arbitrary commit pairs months later.

    Examples:

      sciknow bench-snapshot                         # 'live' on current HEAD
      sciknow bench-snapshot --layer full            # full bench
      sciknow bench-snapshot --tag pre-vlm-pro       # human label prefix
    """
    import json as _json
    from datetime import datetime, timezone
    from sciknow.testing import bench as bench_mod
    from sciknow.core.project import get_active_project

    if layer not in bench_mod.VALID_LAYERS:
        console.print(
            f"[red]Unknown layer:[/red] {layer!r}. "
            f"Use {list(bench_mod.VALID_LAYERS)}"
        )
        raise typer.Exit(2)

    head = _git_head_info()
    if not head["sha"]:
        console.print(
            "[yellow]⚠ git HEAD not available — snapshot will not "
            "carry a commit stamp.[/yellow]"
        )

    # Resolve destination dir.
    active = get_active_project()
    snap_dir = output_dir or (active.data_dir / "bench" / "snapshots")
    snap_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha_part = head["sha"] or "nosha"
    dirty_part = "-dirty" if head["dirty"] else ""
    tag_part = f"-{tag}" if tag else ""
    out_path = snap_dir / f"{sha_part}{dirty_part}-{ts}{tag_part}.json"

    console.print(
        f"[bold]sciknow bench-snapshot[/bold] · layer: "
        f"[cyan]{layer}[/cyan] · sha: [cyan]{sha_part}{dirty_part}[/cyan]"
        + (f" · tag: [dim]{tag}[/dim]" if tag else "")
    )
    console.print()

    results, _ = bench_mod.run(layer=layer, tag=tag or f"snapshot-{layer}")
    bench_mod.render_report(results)

    snapshot = {
        "schema": "sciknow-bench-snapshot/1",
        "snapshotted_at": datetime.now(timezone.utc).isoformat(),
        "git": head,
        "layer": layer,
        "tag": tag,
        "results": [r.as_dict() for r in results],
    }
    out_path.write_text(_json.dumps(snapshot, indent=2))
    console.print()
    console.print(f"[green]✓ Snapshot written →[/green] {out_path}")
    console.print(
        f"[dim]Compare with: sciknow bench-diff {out_path.name} "
        f"<other-snapshot>[/dim]"
    )

    n_err = sum(1 for r in results if r.status == "error")
    raise typer.Exit(0 if n_err == 0 else 1)


def _load_bench_snapshot(path_or_name: str) -> dict:
    """Accept either an absolute path, a relative path, or a bare
    filename under <data_dir>/bench/snapshots/. Raises typer.Exit(2)
    with a useful message on failure."""
    import json as _json
    from sciknow.core.project import get_active_project

    candidates = [Path(path_or_name)]
    snap_dir = get_active_project().data_dir / "bench" / "snapshots"
    candidates.append(snap_dir / path_or_name)
    # Allow passing just a SHA / filename prefix (partial match). We
    # ONLY do this when path_or_name is a bare name (no path
    # separators) — otherwise `snap_dir.glob(abs_path + "*.json")`
    # crashes on "Non-relative patterns are unsupported" because
    # abs_path starts with "/". This was the latent bug caught by
    # the 54.6.224 bench-diff L1 regression after Phase 54.6.230
    # brought the test surface back online.
    if (snap_dir.exists() and "/" not in path_or_name
            and not Path(path_or_name).is_absolute()):
        for p in snap_dir.glob(f"{path_or_name}*.json"):
            candidates.append(p)

    for c in candidates:
        if c.is_file():
            try:
                return _json.loads(c.read_text())
            except Exception as exc:
                console.print(
                    f"[red]Failed to parse snapshot {c}:[/red] {exc}"
                )
                raise typer.Exit(2)

    console.print(
        f"[red]No snapshot found for[/red] {path_or_name!r}. "
        f"Checked: {', '.join(str(c) for c in candidates[:3])}"
    )
    raise typer.Exit(2)


@app.command(name="bench-diff")
def bench_diff_cmd(
    snapshot_a: str = typer.Argument(
        ..., help="Earlier snapshot (path, filename, or SHA prefix).",
    ),
    snapshot_b: str = typer.Argument(
        ..., help="Later snapshot (path, filename, or SHA prefix).",
    ),
    threshold: float = typer.Option(
        0.05, "--threshold",
        help="Flag a metric as a regression when its relative "
             "worsening exceeds this fraction. Default 0.05 = 5%. "
             "'Worsening' is direction-aware: for latency-like "
             "metrics (ms / seconds / tokens_per_s:lower-better) "
             "an INCREASE is bad; for quality metrics (mrr / "
             "recall / ndcg / score / accuracy) a DECREASE is bad. "
             "Metrics with unknown direction get a delta readout "
             "but no red flag.",
    ),
    json_out: bool = typer.Option(
        False, "--json",
        help="Emit machine-readable JSON instead of a Rich table.",
    ),
):
    """Phase 54.6.224 (roadmap 3.11.5) — diff two bench snapshots.

    Loads two snapshot files (by path, filename, or SHA prefix) and
    reports per-metric deltas. Regressions beyond ``--threshold`` are
    flagged red; improvements are green; neutral moves are dim.

    Direction of "better" is inferred from the metric name and unit:

      * Quality↑ (bigger = better): mrr, recall, ndcg, score,
        accuracy, precision, f1, faithfulness, citation_precision,
        agreement, win_rate
      * Latency↓ (smaller = better): ms, seconds, _latency, tokens_per_s
        is actually better-bigger but we special-case it

    Examples:

      sciknow bench-diff abc1234 def5678
      sciknow bench-diff abc1234-dirty-20260422T120000Z.json def5678*
      sciknow bench-diff --threshold 0.10 abc1234 def5678   # 10% threshold
      sciknow bench-diff --json abc1234 def5678 | jq .regressions
    """
    import json as _json

    snap_a = _load_bench_snapshot(snapshot_a)
    snap_b = _load_bench_snapshot(snapshot_b)

    def _index(snap: dict) -> dict[str, tuple[float, str, str]]:
        """Build {fn:metric_name: (value, unit, fn_category)}."""
        idx: dict[str, tuple[float, str, str]] = {}
        for res in snap.get("results") or []:
            if res.get("status") != "ok":
                continue
            fn = res.get("name") or ""
            category = res.get("category") or ""
            for m in res.get("metrics") or []:
                v = m.get("value")
                if not isinstance(v, (int, float)):
                    continue
                key = f"{fn}:{m.get('name')}"
                idx[key] = (float(v), m.get("unit") or "",
                            category or "")
        return idx

    idx_a = _index(snap_a)
    idx_b = _index(snap_b)

    _QUALITY_UP = (
        "mrr", "recall", "ndcg", "score", "accuracy", "precision",
        "f1", "faithfulness", "citation_precision", "agreement",
        "win_rate", "tokens_per_s", "coherence", "npmi",
    )
    _LATENCY_DOWN_UNITS = {"ms", "seconds", "s"}
    _LATENCY_DOWN_KEYWORDS = ("latency", "_ms", "duration")

    def _regression_direction(name: str, unit: str) -> str | None:
        """Return 'up_is_bad' (latency-like), 'down_is_bad' (quality-
        like), or None (unknown)."""
        n = name.lower()
        u = (unit or "").lower()
        # Quality: down is bad
        if any(k in n for k in _QUALITY_UP):
            return "down_is_bad"
        # Latency: up is bad
        if u in _LATENCY_DOWN_UNITS:
            return "up_is_bad"
        if any(k in n for k in _LATENCY_DOWN_KEYWORDS):
            return "up_is_bad"
        return None

    regressions: list[dict] = []
    improvements: list[dict] = []
    neutral: list[dict] = []

    for key in sorted(set(idx_a) | set(idx_b)):
        in_a = key in idx_a
        in_b = key in idx_b
        a_val = idx_a.get(key, (None, "", ""))[0]
        b_val = idx_b.get(key, (None, "", ""))[0]
        unit = (idx_b if in_b else idx_a)[key][1]
        name = key.split(":", 1)[1] if ":" in key else key
        direction = _regression_direction(name, unit)

        record = {
            "metric": key,
            "a": a_val, "b": b_val,
            "unit": unit,
            "direction": direction,
            "only_in": None if (in_a and in_b)
                       else ("a" if in_a else "b"),
        }

        if in_a and in_b:
            delta = b_val - a_val
            pct = (delta / a_val) if a_val not in (0, None) else None
            record["delta"] = delta
            record["delta_pct"] = pct
            is_regression = False
            is_improvement = False
            if pct is not None and direction:
                if direction == "up_is_bad":
                    is_regression = pct > threshold
                    is_improvement = pct < -threshold
                else:
                    is_regression = pct < -threshold
                    is_improvement = pct > threshold
            if is_regression:
                regressions.append(record)
            elif is_improvement:
                improvements.append(record)
            else:
                neutral.append(record)
        else:
            record["delta"] = None
            record["delta_pct"] = None
            neutral.append(record)

    if json_out:
        console.print(_json.dumps({
            "a": {"git": snap_a.get("git"), "layer": snap_a.get("layer"),
                   "snapshotted_at": snap_a.get("snapshotted_at")},
            "b": {"git": snap_b.get("git"), "layer": snap_b.get("layer"),
                   "snapshotted_at": snap_b.get("snapshotted_at")},
            "threshold": threshold,
            "regressions": regressions,
            "improvements": improvements,
            "neutral": neutral,
        }, indent=2, default=str))
        raise typer.Exit(1 if regressions else 0)

    # Rich summary
    from rich.table import Table
    from rich import box as _box

    def _sha(snap):
        g = snap.get("git") or {}
        sha = g.get("sha") or "?"
        dirty = "-dirty" if g.get("dirty") else ""
        return f"{sha}{dirty}"

    console.print(
        f"[bold]Bench diff[/bold] · "
        f"A: [cyan]{_sha(snap_a)}[/cyan] "
        f"({snap_a.get('layer')}) "
        f"vs B: [cyan]{_sha(snap_b)}[/cyan] "
        f"({snap_b.get('layer')}) · threshold {threshold:.0%}"
    )
    if snap_a.get("layer") != snap_b.get("layer"):
        console.print(
            "[yellow]⚠ layers differ — metric sets may not align.[/yellow]"
        )

    t = Table(box=_box.SIMPLE_HEAD, expand=True)
    t.add_column("Metric", ratio=4)
    t.add_column("A", justify="right", width=10)
    t.add_column("B", justify="right", width=10)
    t.add_column("Δ", justify="right", width=9)
    t.add_column("Δ%", justify="right", width=8)
    t.add_column("Status", width=10)

    def _fmt_val(v):
        if v is None:
            return "—"
        if isinstance(v, float) and abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    for rec in regressions + improvements + neutral:
        delta = rec["delta"]
        pct = rec["delta_pct"]
        if rec in regressions:
            status, colour = "REGRESS", "red"
        elif rec in improvements:
            status, colour = "IMPROVE", "green"
        elif rec.get("only_in") == "a":
            status, colour = "removed", "yellow"
        elif rec.get("only_in") == "b":
            status, colour = "new", "cyan"
        else:
            status, colour = "—", "dim"
        t.add_row(
            rec["metric"],
            _fmt_val(rec["a"]),
            _fmt_val(rec["b"]),
            _fmt_val(delta),
            f"{pct * 100:+.1f}%" if pct is not None else "—",
            f"[{colour}]{status}[/{colour}]",
        )
    console.print(t)

    console.print(
        f"\n[bold]{len(regressions)}[/bold] regression(s), "
        f"[bold]{len(improvements)}[/bold] improvement(s), "
        f"[dim]{len(neutral)}[/dim] neutral"
    )
    if regressions:
        console.print(
            "[red]⚠ Regressions detected — inspect before merging.[/red]"
        )
    raise typer.Exit(1 if regressions else 0)


@app.command(name="mcp-serve")
def mcp_serve_cmd():
    """Phase 54.6.77 (#16) — run sciknow as an MCP (Model Context Protocol)
    server over stdio.

    Exposes four tools to any MCP-speaking agent (Claude Desktop,
    Claude Code, goose, etc.):
      - search_corpus       hybrid retrieval, top-k chunks
      - ask_corpus          full RAG with citations
      - list_chapters       book outline
      - get_paper_summary   compiled wiki page by slug

    No arguments — the transport is stdin/stdout. Configure Claude
    Desktop / an agent harness with:

        {
          "mcpServers": {
            "sciknow": {
              "command": "uv",
              "args": ["run", "sciknow", "mcp-serve"],
              "cwd": "/path/to/sciknow/repo"
            }
          }
        }

    The active sciknow project is whichever `.active-project` points
    at when this process starts. Restart the MCP client to pick up a
    project switch.
    """
    import asyncio
    from sciknow.mcp_server import serve_stdio
    asyncio.run(serve_stdio())


@app.command(name="bench-visuals-ranker")
def bench_visuals_ranker_cmd(
    n: int = typer.Option(
        30, "-n", "--n-items",
        help="Size of the stratified eval set (default 30 per "
             "RESEARCH.md §7.X.4). Items are mined automatically from "
             "the corpus's Phase-54.6.138 mention-paragraphs — no hand "
             "curation required.",
    ),
    seed: int = typer.Option(
        42, "--seed",
        help="Random seed for stratified sampling. Reproducible across runs.",
    ),
    mine_limit: int = typer.Option(
        500, "--mine-limit",
        help="Cap on raw items mined before stratification (default 500).",
    ),
    ablation: bool = typer.Option(
        False, "--ablate-same-paper",
        help="Run with signal 2 (same-paper bonus) ablated — drops "
             "cited_doc_ids to measure caption + mention-paragraph "
             "signals' discriminative power in isolation.",
    ),
    tag: str = typer.Option(
        "", "--tag",
        help="Free-form label stamped into the output JSONL filename.",
    ),
):
    """Phase 54.6.140 — measure the 5-signal visuals ranker.

    Mines a stratified eval set from the corpus (every stored
    ``mention_paragraph`` is a ground-truth "author said this figure
    supports this claim" pair; extracting the sentence gives an
    honest query-to-correct-figure pair without hand curation),
    runs the ranker, reports P@1 / R@3 / same-paper-rate.

    Emits JSONL to ``{data_dir}/bench/visuals_ranker-<ts>[-tag].jsonl``
    with per-item detail so regressions can be traced back to specific
    (sentence, figure) pairs that started missing.

    Examples:

      sciknow bench-visuals-ranker                  # 30 items, full ranker
      sciknow bench-visuals-ranker -n 100           # larger set
      sciknow bench-visuals-ranker --ablate-same-paper  # signal 2 off

    The eval is deterministic: same seed + same corpus state → same
    numbers. Running it before/after any ranker change produces a
    clean A/B.
    """
    from sciknow.cli import preflight
    preflight()

    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    from pathlib import Path as _P
    from sciknow.config import settings as _settings
    from sciknow.testing import visuals_eval as ve

    console.print(
        f"[bold]Mining eval items[/bold] (mine_limit={mine_limit})…"
    )
    items = ve.mine_eval_items(limit=mine_limit)
    console.print(
        f"[dim]Mined {len(items)} raw items.[/dim] "
        f"Sampling stratified {n} (seed={seed})…"
    )
    sampled = ve.sample_stratified(items, n=n, seed=seed)
    if len(sampled) < n:
        console.print(
            f"[yellow]Only {len(sampled)} items after stratification — "
            f"corpus may lack diverse mention-paragraphs. "
            f"Run `sciknow db link-visual-mentions --force` first.[/yellow]"
        )

    from collections import Counter
    type_counts = Counter(it.sentence_type for it in sampled)
    console.print(f"[dim]By sentence type: {dict(type_counts)}[/dim]\n")

    console.print(
        f"[bold]Running ranker[/bold] "
        f"({'ABLATED — signal 2 off' if ablation else 'full 5-signal'})…"
    )
    report = ve.run_eval(sampled, use_cited_doc=(not ablation))

    console.print(f"\n[bold]Results ({report.n_items} items, "
                  f"{report.elapsed_s:.1f}s total, "
                  f"{report.elapsed_s/max(report.n_items,1):.2f}s/item):[/bold]")
    console.print(f"  [green]P@1:[/green]              "
                  f"{report.p_at_1:.3f}  ({report.n_top1_correct}/{report.n_items})")
    console.print(f"  [green]R@3:[/green]              "
                  f"{report.r_at_3:.3f}  ({report.n_top3_correct}/{report.n_items})")
    console.print(f"  Same-paper top1: "
                  f"{report.same_paper_rate:.3f}  ({report.n_same_paper_top1}/{report.n_items})")
    console.print(f"  Mean composite:  "
                  f"top1={report.mean_composite_top1:.3f}  "
                  f"correct={report.mean_composite_correct:.3f}")

    # Persist
    ts = _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = f"visuals_ranker-{ts}"
    if tag:
        slug += f"-{tag}"
    if ablation:
        slug += "-ablated"
    out_dir = _P(_settings.data_dir) / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.jsonl"
    with out_path.open("w") as f:
        # Summary as line 1
        f.write(_json.dumps({
            "type": "summary",
            "n_items": report.n_items,
            "n_top1_correct": report.n_top1_correct,
            "n_top3_correct": report.n_top3_correct,
            "p_at_1": round(report.p_at_1, 4),
            "r_at_3": round(report.r_at_3, 4),
            "same_paper_rate": round(report.same_paper_rate, 4),
            "mean_composite_top1": report.mean_composite_top1,
            "mean_composite_correct": report.mean_composite_correct,
            "elapsed_s": report.elapsed_s,
            "ablation_same_paper": ablation,
            "seed": seed,
        }) + "\n")
        for item in report.per_item:
            f.write(_json.dumps({"type": "item", **item}) + "\n")
    console.print(f"\n[dim]Results written to[/dim] {out_path}")


@app.command(name="bench-autowrite-ab")
def bench_autowrite_ab_cmd(
    chapter_id: str = typer.Argument(
        ..., help="Chapter ID (UUID or 8-char prefix) to run the A/B on.",
    ),
    model: str = typer.Option(
        None, "--model",
        help="Writer model override. Defaults to settings.llm_model.",
    ),
    max_iter: int = typer.Option(
        3, "--max-iter",
        help="Autowrite max iterations per section per condition. "
             "Default 3. Lower to smoke-test, higher for research-grade.",
    ),
    include_unplanned: bool = typer.Option(
        False, "--include-unplanned",
        help="Include sections without a plan (they can only do the "
             "top-down condition; bottom-up is skipped for them). "
             "Default: only run A/B on sections with a plan.",
    ),
    output_json: bool = typer.Option(
        False, "--json",
        help="Emit machine-readable JSON instead of the Rich table.",
    ),
    tag: str = typer.Option(
        "", "--tag",
        help="Free-form label stamped into the output JSONL filename.",
    ),
):
    """Phase 54.6.161 — bottom-up vs top-down autowrite A/B.

    For each planned section in the chapter: run autowrite with the
    Phase 54.6.146 concept-density resolver fired (plan present), then
    again with the plan temporarily cleared (chapter-split fallback).
    Compare scorer dimensions — tells you whether concept-density
    actually produces better drafts on this corpus.

    The RESEARCH.md §24 gap #3: "no one has tested bottom-up section-
    first vs top-down chapter-first for LLM-assisted scientific
    writing." This CLI is the harness; the experiment design + sample
    size decision is yours.

    Cost: each section = 2 full autowrites × 3 iterations ≈
    2-6 minutes on a 3090. A 5-section chapter → 10-30 min.

    Output: per-dimension mean delta (bottom-up − top-down) + win rate.
    Persisted as JSONL under ``{data_dir}/bench/autowrite_ab-<ts>.jsonl``
    so multiple runs can be compared post-hoc.
    """
    from sciknow.cli import preflight
    preflight()

    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    from pathlib import Path as _P
    from rich.table import Table as _RT
    from sqlalchemy import text as _sql
    from sciknow.config import settings as _settings
    from sciknow.storage.db import get_session
    from sciknow.testing import autowrite_ab

    with get_session() as session:
        row = session.execute(_sql(
            "SELECT id::text, title FROM book_chapters "
            "WHERE id::text LIKE :q LIMIT 1"
        ), {"q": f"{chapter_id.strip()}%"}).fetchone()
        if not row:
            console.print(f"[red]No chapter matches {chapter_id!r}[/red]")
            raise typer.Exit(1)
        cid, ctitle = row

    console.print(
        f"[bold]Autowrite A/B on Ch.[/bold] {ctitle!r}  "
        f"[dim](id {cid[:12]}…)[/dim]\n"
    )
    try:
        report = autowrite_ab.run_ab(
            cid, model=model, max_iter=max_iter,
            only_planned=(not include_unplanned),
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]A/B failed: {exc}[/red]")
        raise typer.Exit(2)

    if output_json:
        console.print_json(_json.dumps(report.to_dict()))
    else:
        t = _RT(
            title=(f"Bottom-up vs top-down  ·  {report.n_sections} section(s)  "
                   f"·  {report.elapsed_s:.1f}s"),
            show_lines=False,
        )
        t.add_column("Dimension", style="bold")
        t.add_column("Mean Δ (BU − TD)", justify="right")
        t.add_column("BU win rate", justify="right")
        t.add_column("Verdict")
        sorted_dims = sorted(report.per_dimension_delta_mean.keys())
        for dim in sorted_dims:
            d = report.per_dimension_delta_mean[dim]
            w = report.per_dimension_win_rate.get(dim, 0.5)
            verdict = (
                "[green]BU wins[/green]" if d > 0.03 and w > 0.6
                else "[red]TD wins[/red]" if d < -0.03 and w < 0.4
                else "[dim]ambiguous[/dim]"
            )
            t.add_row(dim, f"{d:+.4f}", f"{w:.0%}", verdict)
        console.print(t)
        console.print(
            f"\n[dim]N={report.n_sections} sections. Variance on scorer "
            "dimensions is ~±0.05 across runs; verdicts require |Δ| > 0.03 "
            "AND win-rate confirmation. For tight estimates, "
            "run on multiple chapters and pool.[/dim]"
        )

    ts = _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = f"autowrite_ab-{ts}"
    if tag:
        slug += f"-{tag}"
    out_dir = _P(_settings.data_dir) / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.jsonl"
    with out_path.open("w") as f:
        f.write(_json.dumps(report.to_dict()) + "\n")
    console.print(f"\n[dim]Saved to {out_path}[/dim]")


@app.command(name="bench-idea-density")
def bench_idea_density_cmd(
    sample_per_type: int = typer.Option(
        500, "-n", "--sample-per-type",
        help="Number of sections sampled per canonical section type "
             "(abstract / introduction / methods / results / discussion "
             "/ conclusion / related_work). 500 × 7 ≈ 4-8 min wall. "
             "Drop to 100-200 for a quick smoke; raise for tighter "
             "confidence intervals.",
    ),
    output_json: bool = typer.Option(
        False, "--json",
        help="Emit machine-readable JSON instead of the Rich table.",
    ),
    tag: str = typer.Option(
        "", "--tag",
        help="Free-form label stamped into the output JSONL filename.",
    ),
):
    """Phase 54.6.160 — Brown 2008 propositional idea-density regression.

    Computes empirical words-per-concept per section type by sampling
    the corpus's paper_sections, running Brown 2008 P-density (Brown
    et al., 2008, doi:10.3758/BRM.40.2.540) via spaCy POS tagging,
    fitting ``word_count ~ n_ideas`` per section type, and reporting
    the slope as the empirical wpc.

    This is the "publishable on its own" experiment RESEARCH.md §24
    §gaps lists. Complements (does not replace) the 54.6.146
    research-grounded project-type wpc defaults — those come from a
    literature survey; this gives corpus-specific numbers.

    spaCy is required and not a default sciknow dependency:

      uv add spacy
      python -m spacy download en_core_web_sm

    Without it the command exits with a pointed install message.

    Output: per-section-type {n, mean word count, mean p-density,
    slope wpc, R², wpc IQR}. Compare `slope_wpc` and `wpc_median`
    against `sciknow book types` for the shipped defaults.

    Writes the full report to
    ``{data_dir}/bench/idea_density-<ts>[-tag].jsonl``.
    """
    from sciknow.cli import preflight
    preflight()

    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    from pathlib import Path as _P
    from rich.table import Table as _RT
    from sciknow.config import settings as _settings
    from sciknow.testing import idea_density_regression as idr

    console.print(
        f"[bold]Brown 2008 idea-density regression[/bold] "
        f"(sample={sample_per_type} per type, "
        f"{len(idr._CANONICAL_SECTIONS)} canonical types)"
    )
    console.print("[dim]spaCy POS tagging ~100 ms per section. "
                  "Wall time scales with sample size × n_types.[/dim]\n")

    try:
        report = idr.run_regression(sample_per_type=sample_per_type)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]regression failed: {exc}[/red]")
        raise typer.Exit(3)

    if output_json:
        console.print_json(_json.dumps(report.to_dict()))
    else:
        t = _RT(
            title=(f"Brown 2008 idea-density per section type  "
                   f"(n={report.n_total} sections, {report.elapsed_s:.1f}s)"),
            show_lines=False,
        )
        t.add_column("Section", style="bold")
        t.add_column("n", justify="right")
        t.add_column("Mean words", justify="right")
        t.add_column("Mean P-density", justify="right")
        t.add_column("Slope wpc", justify="right")
        t.add_column("R²", justify="right")
        t.add_column("wpc median", justify="right")
        t.add_column("wpc IQR", overflow="fold")
        for st, r in report.by_type.items():
            t.add_row(
                st,
                str(r.n_sections),
                f"{r.mean_word_count:.0f}",
                f"{r.mean_p_density:.3f}",
                f"{r.slope_wpc:.1f}",
                f"{r.r_squared:.2f}",
                f"{r.wpc_median:.0f}",
                f"{r.wpc_q1:.0f}–{r.wpc_q3:.0f}",
            )
        console.print(t)
        console.print(
            "\n[dim]slope_wpc: OLS slope of word_count ~ n_ideas; "
            "wpc_median: median of per-section word/idea ratio; "
            "compare to `sciknow book types` shipped wpc ranges.[/dim]"
        )

    # Persist to bench dir for comparability with other runs
    ts = _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = f"idea_density-{ts}"
    if tag:
        slug += f"-{tag}"
    out_dir = _P(_settings.data_dir) / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.jsonl"
    with out_path.open("w") as f:
        f.write(_json.dumps(report.to_dict()) + "\n")
    console.print(f"\n[dim]Saved to {out_path}[/dim]")


@app.command(name="bench-vlm-gen")
def bench_vlm_gen_cmd(
    n: int = typer.Option(
        15, "-n", "--n-figures",
        help="How many figures to pin for the VLM sweep (default 15).",
    ),
    seed: int = typer.Option(
        42, "--seed",
        help="Random seed for sampling — reproducible across runs.",
    ),
):
    """Phase 54.6.74 (#1b) — pin the figure set for the VLM sweep.

    Samples N figures from the corpus (figure + chart kinds with
    on-disk assets, diversified across documents), persists to
    ``<project>/data/bench/vlm_sweep_figures.json``. The
    ``bench --layer vlm-sweep`` reads this file and captions each
    figure with every installed candidate VLM.

    Deterministic for a fixed ``--seed``. Regenerate when the corpus
    changes enough that the pinned figures no longer represent it
    (e.g. after a big ingest batch).
    """
    from sciknow.testing import vlm_sweep
    console.print(f"[bold]Pinning {n} figures for VLM sweep…[/bold]")
    path = vlm_sweep.generate_figure_set(n=n, seed=seed)
    console.print(f"\n[green]✓ Figure set written to[/green] {path}")
    console.print(
        "[dim]Next: `ollama pull` any VLMs you want to bench, then "
        "`sciknow bench --layer vlm-sweep --tag <tag>` to caption + "
        "judge-rank them.[/dim]"
    )


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
