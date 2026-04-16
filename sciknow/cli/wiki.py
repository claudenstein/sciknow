"""
sciknow wiki — Karpathy-style compiled knowledge wiki.

Commands:
  compile      Build wiki pages from ingested papers
  query        Answer questions from the compiled wiki
  lint         Check wiki health (broken links, contradictions, stale pages)
  list         List all wiki pages
  show         Display a wiki page
  synthesize   Generate a synthesis overview page on a topic
"""
from __future__ import annotations

from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

app = typer.Typer(help="Compiled knowledge wiki (Karpathy LLM-wiki pattern).")
console = Console()


def _check_wiki_table():
    """Verify the wiki_pages table exists, or exit with a helpful message."""
    from sciknow.storage.db import get_session
    from sqlalchemy import text
    try:
        with get_session() as sess:
            sess.execute(text("SELECT 1 FROM wiki_pages LIMIT 0"))
    except Exception:
        console.print(
            "[red]The wiki_pages table does not exist.[/red]\n"
            "Run the migration first:\n\n"
            "  [bold]uv run alembic upgrade head[/bold]\n"
        )
        raise typer.Exit(1)


def _consume_events(gen, console):
    """Consume events from a wiki_ops generator, printing to Rich console."""
    completed = None
    for event in gen:
        t = event.get("type")
        if t == "token":
            console.print(event["text"], end="", highlight=False)
        elif t == "progress":
            console.print(f"[dim]{event.get('detail', event.get('stage', ''))}[/dim]")
        elif t == "lint_issue":
            sev = event.get("severity", "low")
            color = "red" if sev == "high" else "yellow" if sev == "medium" else "dim"
            console.print(f"  [{color}]{event.get('type_', event.get('type', ''))}[/{color}]: {event.get('detail', '')}")
        elif t == "completed":
            completed = event
        elif t == "error":
            console.print(f"[red]Error:[/red] {event.get('message', 'unknown')}")
    return completed


@app.command()
def compile(
    doc_id: str | None = typer.Option(None, "--doc-id", "-d",
                                       help="Compile a single paper by document ID."),
    rebuild: bool = typer.Option(False, "--rebuild",
                                  help="Recompile ALL pages from scratch (destructive). Default: only compile new papers."),
    rewrite_stale: bool = typer.Option(False, "--rewrite-stale",
                                        help="Rewrite pages marked as stale."),
    no_entities: bool = typer.Option(
        False, "--no-entities",
        help="Skip entity + knowledge-graph extraction (Phase 54.6.34). "
             "Useful when the structured-output extraction is failing on "
             "your corpus — you still get paper summaries written + "
             "embedded, just no KG triples. Run `sciknow wiki extract-kg` "
             "later to backfill entities when a working strategy is found.",
    ),
    model: str | None = typer.Option(None, "--model"),
):
    """
    Build wiki pages from ingested papers.

    By default, only compiles papers that don't have wiki pages yet
    (safe to re-run anytime). Use --rebuild to recompile everything.

    Examples:

      sciknow wiki compile                    # compile only new papers (safe)

      sciknow wiki compile --doc-id abc123    # compile one paper

      sciknow wiki compile --rebuild          # recompile everything from scratch

      sciknow wiki compile --no-entities      # summaries only, skip KG extraction
    """
    import warnings
    warnings.filterwarnings("ignore", message=".*urllib3.*")
    warnings.filterwarnings("ignore", message=".*charset_normalizer.*")
    warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")
    warnings.filterwarnings("ignore", message=".*fast tokenizer.*")

    from sciknow.cli import preflight
    preflight(qdrant=True)
    _check_wiki_table()

    from sciknow.core import wiki_ops

    if doc_id:
        console.print(f"Compiling wiki page for document {doc_id[:8]}...")
        gen = wiki_ops.compile_paper_summary(
            doc_id, model=model, force=rebuild, skip_entities=no_entities,
        )
        result = _consume_events(gen, console)
        if result and not result.get("skipped"):
            entities = result.get("entities", [])
            kg = result.get("kg_triples", 0)
            console.print(f"[green]✓ {len(entities)} entities, {kg} KG triples extracted[/green]")
    else:
        import time as _time
        from rich.live import Live
        from rich.table import Table
        from rich.text import Text
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

        gen = wiki_ops.compile_all(model=model, force=rebuild,
                                    rewrite_stale=rewrite_stale,
                                    skip_entities=no_entities)

        result = None
        total = 0
        paper_tok_count = 0
        paper_t0 = _time.monotonic()
        compile_t0 = _time.monotonic()
        last_paper_tps = 0.0
        last_paper_title = ""
        last_paper_elapsed = 0.0
        last_paper_tokens = 0
        workers_count = 1  # set from compile_start; used as tok/s multiplier
        # Phase 54.6.29 — rolling window of recently-seen paper titles
        # so the user can see what's being worked on across parallel
        # workers (in parallel mode, paper_start fires on replay which
        # means titles appear as papers COMPLETE, not when they start).
        recent_titles: list[str] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=25),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[status]}"),
            console=console,
            refresh_per_second=4,
        ) as progress:
            task_id = None
            tok_task = None

            for event in gen:
                t = event.get("type")

                if t == "compile_start":
                    total = event["total"]
                    workers_count = max(1, int(event.get("workers", 1)))
                    compile_t0 = _time.monotonic()
                    task_id = progress.add_task(
                        "Compiling wiki", total=total,
                        status=f"starting ({workers_count} worker(s))...")
                    tok_task = progress.add_task(
                        "[dim]LLM", total=None, status="[dim]waiting...[/dim]")

                elif t == "paper_start":
                    paper_tok_count = 0
                    paper_t0 = _time.monotonic()
                    title_full = event["title"]
                    last_paper_title = title_full
                    idx = event.get("index", 0)
                    # Phase 54.6.29 — maintain rolling window of last N
                    # papers (N = workers_count) so parallel mode shows
                    # which papers are in-flight instead of just the
                    # most recent one.
                    recent_titles.append(title_full[:40])
                    if len(recent_titles) > workers_count:
                        recent_titles.pop(0)
                    in_flight_str = " | ".join(recent_titles)
                    progress.update(task_id,
                        description=f"[bold]Wiki {idx}/{total}[/bold]",
                        status=f"[dim]{in_flight_str[:120]}[/dim]")
                    progress.update(tok_task,
                        status=f"[dim]{title_full[:45]}...[/dim]")

                elif t == "token":
                    paper_tok_count += 1
                    elapsed = _time.monotonic() - paper_t0
                    # Phase 54.6.29 — multiply by worker count in
                    # parallel mode so the displayed rate reflects
                    # aggregate GPU throughput, not the per-paper
                    # wall-time rate that underreports by a factor of
                    # ~N due to cross-worker contention.
                    raw_tps = paper_tok_count / elapsed if elapsed > 0 else 0
                    tps = raw_tps * workers_count
                    total_toks = event.get("total_tokens", paper_tok_count)
                    progress.update(tok_task,
                        status=(
                            f"[green]{paper_tok_count} tok[/green]  "
                            f"[cyan]~{tps:.0f} tok/s agg[/cyan]  "
                            f"[dim]{total_toks} total[/dim]"
                        ))

                elif t == "paper_done":
                    progress.advance(task_id)
                    st = event.get("status", "")
                    c = event.get("compiled", 0)
                    s = event.get("skipped", 0)
                    f = event.get("failed", 0)
                    concepts = event.get("concepts", 0)
                    p_toks = event.get("tokens", 0)
                    p_elapsed = event.get("elapsed", 0)

                    last_paper_tokens = p_toks
                    last_paper_elapsed = p_elapsed
                    # Per-call generation rate (not multiplied) — this
                    # is the actual per-request speed, useful for
                    # debugging whether the GPU is saturated.
                    last_paper_tps = p_toks / p_elapsed if p_elapsed > 0 else 0
                    # Aggregate throughput across workers for the final
                    # "X tok in Ys (Z t/s)" line below.
                    last_paper_agg_tps = last_paper_tps * workers_count

                    # Estimate remaining time
                    done = c + s + f
                    remaining = total - done
                    if c > 0 and p_elapsed > 0:
                        total_elapsed = _time.monotonic() - compile_t0
                        avg_per_compiled = total_elapsed / max(done, 1)
                        eta_s = remaining * avg_per_compiled
                        if eta_s > 3600:
                            eta_str = f"{eta_s/3600:.1f}h"
                        elif eta_s > 60:
                            eta_str = f"{eta_s/60:.0f}m"
                        else:
                            eta_str = f"{eta_s:.0f}s"
                        eta_part = f"  [yellow]~{eta_str} left[/yellow]"
                    else:
                        eta_part = ""

                    title_short = event.get("title", "")[:40]
                    status_text = f"[green]{c} new[/green]  [dim]{s} skip[/dim]"
                    if f:
                        status_text += f"  [red]{f} fail[/red]"
                    status_text += eta_part
                    # Phase 54.6.29 — tack on just-finished title so the
                    # user sees papers streaming past in the bar.
                    if title_short:
                        status_text += f"  [dim]last: {title_short}[/dim]"

                    progress.update(task_id, status=status_text)

                    if st == "compiled" and p_toks > 0:
                        progress.update(tok_task,
                            status=(
                                f"[green]{p_toks} tok in {p_elapsed:.0f}s "
                                f"({last_paper_tps:.1f} t/s, "
                                f"~{last_paper_agg_tps:.0f} agg)[/green]"
                            ))
                    elif st == "skipped":
                        progress.update(tok_task,
                            status=f"[dim]skip: {title_short}[/dim]")

                elif t == "error":
                    console.print(f"[red]Error:[/red] {event.get('message', '')}")

                elif t == "completed":
                    result = event
                    if tok_task:
                        progress.remove_task(tok_task)

        if result:
            total_elapsed = _time.monotonic() - compile_t0
            if total_elapsed > 3600:
                elapsed_str = f"{total_elapsed/3600:.1f} hours"
            elif total_elapsed > 60:
                elapsed_str = f"{total_elapsed/60:.0f} minutes"
            else:
                elapsed_str = f"{total_elapsed:.0f} seconds"

            console.print(
                f"\n[green]✓ Wiki compiled:[/green] "
                f"{result.get('compiled', 0)} new, "
                f"{result.get('skipped', 0)} skipped, "
                f"{result.get('failed', 0)} failed "
                f"/ {result.get('total', 0)} total papers "
                f"in {elapsed_str}"
            )


@app.command(name="repair")
def repair(
    dry_run: bool = typer.Option(False, "--dry-run",
        help="Report what would be repaired/pruned without touching anything."),
    prune: bool = typer.Option(False, "--prune",
        help="ALSO drop wiki_pages DB rows whose disk file is missing AND "
             "the page_type isn't 'concept' (i.e. paper_summary + synthesis). "
             "Pruned rows will be regenerated by the next `wiki compile` for "
             "papers, or by `wiki query` for synthesis. Concepts are always "
             "regenerated as stubs (cheap, no LLM)."),
):
    """Phase 54.6.22 — repair wiki rows whose disk file is missing.

    The wiki keeps state in two places: the ``wiki_pages`` Postgres
    table (slug + metadata + word_count + qdrant_point_id) and the
    actual markdown content under ``<data_dir>/wiki/{papers,concepts,
    synthesis}/<slug>.md``. ``show_page`` and the GUI's Browse tab read
    content FROM DISK only — so a row whose disk file got nuked
    becomes a 404 ("Wiki page <slug> not found") in the GUI.

    Common cause: ``project init <slug> --from-existing`` clones the
    Postgres DB but only ``shutil.move``-s ``data/*`` if it still
    exists. If the user previously ran ``db reset --project default``
    (which wipes ``data/wiki/``), the migration brings the DB rows
    over without the disk content. Result: orphan rows that show in
    the index but 404 on click.

    What this command does:

      * **concept** rows with no disk file → regenerate the stub on
        disk from the DB row (title + first source paper's year). No
        LLM call. Cheap.
      * **paper_summary** rows with no disk file → require an LLM
        call to recreate, so we DON'T touch them by default. With
        ``--prune``, we delete the row so the next ``wiki compile``
        run will recreate the page from scratch.
      * **synthesis** rows → same prune-only treatment as paper_summary.

    Examples:

      sciknow wiki repair --dry-run         # see what's broken
      sciknow wiki repair                   # regen concept stubs only
      sciknow wiki repair --prune           # also drop orphan paper/synth rows
    """
    from sciknow.cli import preflight
    preflight()
    _check_wiki_table()

    from sqlalchemy import text as sql_text
    from sciknow.config import settings
    from sciknow.core.wiki_ops import _ensure_wiki_dirs, _save_page
    from sciknow.storage.db import get_session

    type_to_subdir = {
        "paper_summary": "papers",
        "concept": "concepts",
        "synthesis": "synthesis",
    }

    if not dry_run:
        _ensure_wiki_dirs()

    with get_session() as session:
        rows = session.execute(sql_text("""
            SELECT wp.slug, wp.title, wp.page_type,
                   wp.source_doc_ids,
                   COALESCE((
                     SELECT pm.year FROM paper_metadata pm
                     WHERE wp.source_doc_ids IS NOT NULL
                       AND array_length(wp.source_doc_ids, 1) >= 1
                       AND pm.document_id = wp.source_doc_ids[1]
                     LIMIT 1
                   ), 0) AS year_from_first_source
            FROM wiki_pages wp
            ORDER BY wp.page_type, wp.slug
        """)).fetchall()

    missing_concepts: list[tuple] = []
    missing_papers: list[tuple] = []
    missing_synth: list[tuple] = []
    for slug, title, page_type, src_ids, year in rows:
        subdir = type_to_subdir.get(page_type)
        if subdir is None:
            continue
        path = settings.wiki_dir / subdir / f"{slug}.md"
        if path.exists():
            continue
        if page_type == "concept":
            missing_concepts.append((slug, title, year))
        elif page_type == "paper_summary":
            missing_papers.append((slug, title))
        elif page_type == "synthesis":
            missing_synth.append((slug, title))

    console.print(
        f"[bold]Wiki repair scan:[/bold] "
        f"[yellow]{len(missing_concepts)}[/yellow] missing concepts, "
        f"[yellow]{len(missing_papers)}[/yellow] missing paper_summary, "
        f"[yellow]{len(missing_synth)}[/yellow] missing synthesis"
    )
    if not (missing_concepts or missing_papers or missing_synth):
        console.print("[green]✓ All wiki_pages rows have a disk file.[/green]")
        return

    # ── Concept stubs: cheap regen ────────────────────────────────────
    regenerated = 0
    for slug, title, year in missing_concepts:
        if dry_run:
            console.print(
                f"  [dim]would regenerate concept stub:[/dim] {slug}"
            )
            regenerated += 1
            continue
        try:
            year_str = str(year) if year else "n.d."
            stub = (
                f"# {title or slug.replace('-', ' ').title()}\n\n"
                f"*Recovered stub (Phase 54.6.22 wiki repair). The original "
                f"disk content was lost, likely after a "
                f"`project init --from-existing` migration that ran AFTER a "
                f"`db reset` had wiped `data/wiki/`. Run "
                f"`sciknow wiki compile` to refill in-line concept references "
                f"as papers are recompiled.*\n\n"
                f"*Linked from {len(rows)} compiled wiki page(s); see "
                f"backlinks below.*\n"
            )
            with get_session() as session:
                _save_page(
                    session, slug=slug, title=title or slug,
                    page_type="concept", content=stub,
                    source_doc_ids=[],   # don't fabricate doc references
                    subdir="concepts",
                )
            regenerated += 1
        except Exception as exc:
            console.print(f"  [red]skip concept {slug}:[/red] {exc}")

    # ── Paper / synthesis: prune (only if --prune) ────────────────────
    pruned = 0
    if prune and (missing_papers or missing_synth):
        slugs_to_prune = [s for s, _ in missing_papers] + [s for s, _ in missing_synth]
        if dry_run:
            for s in slugs_to_prune:
                console.print(f"  [dim]would prune row:[/dim] {s}")
            pruned = len(slugs_to_prune)
        else:
            try:
                with get_session() as session:
                    placeholders = ", ".join(f":s{i}" for i, _ in enumerate(slugs_to_prune))
                    params = {f"s{i}": s for i, s in enumerate(slugs_to_prune)}
                    res = session.execute(
                        sql_text(f"DELETE FROM wiki_pages WHERE slug IN ({placeholders})"),
                        params,
                    )
                    pruned = res.rowcount or 0
                    session.commit()
            except Exception as exc:
                console.print(f"  [red]prune failed:[/red] {exc}")
    elif missing_papers or missing_synth:
        console.print(
            f"[dim]Pass --prune to drop {len(missing_papers) + len(missing_synth)} "
            f"orphan paper_summary/synthesis row(s) so the next `wiki compile` "
            f"can regenerate them.[/dim]"
        )

    verb = "Would" if dry_run else "Did"
    console.print(
        f"[bold]Repair summary:[/bold] {verb} regenerate "
        f"[green]{regenerated}[/green] concept stub(s); "
        f"{verb.lower()} prune [red]{pruned}[/red] orphan row(s)."
    )


@app.command(name="extract-kg")
def extract_kg(
    force: bool = typer.Option(False, "--force",
        help="Re-extract even for papers that already have triples. "
             "Default only backfills papers with 0 triples in knowledge_graph."),
    doc_id: str | None = typer.Option(None, "--doc-id", "-d",
        help="Extract for just one paper (matches on document id prefix). "
             "Useful for smoke-testing before firing at the whole corpus."),
    model: str | None = typer.Option(None, "--model"),
):
    """Backfill knowledge_graph triples for already-compiled wiki pages.

    Context: prior versions of ``wiki compile`` didn't run entity + KG
    extraction as part of the paper-summary pipeline. Users who built
    their wiki under those earlier versions end up with paper_summary
    pages but an empty knowledge_graph table — the KG modal shows
    "no triples match your filter" even for basic queries. This
    command walks papers that have a wiki page but zero triples and
    runs the current extraction step on them, without re-doing the
    (expensive) summary LLM call.

    Typical invocation:

      sciknow wiki extract-kg                      # backfill all orphans
      sciknow wiki extract-kg --doc-id 3f2a         # test on one paper first
      sciknow wiki extract-kg --force               # re-extract everything
    """
    from sciknow.cli import preflight
    preflight()
    _check_wiki_table()

    from sqlalchemy import text as sql_text
    from sciknow.core.wiki_ops import (
        _extract_entities_and_kg, _load_existing_slugs, _slugify,
    )
    from sciknow.storage.db import get_session

    with get_session() as session:
        if doc_id:
            rows = session.execute(sql_text("""
                SELECT d.id::text, pm.title, pm.abstract, pm.year,
                       pm.authors, pm.keywords, pm.domains
                FROM documents d
                JOIN paper_metadata pm ON pm.document_id = d.id
                WHERE d.ingestion_status = 'complete'
                  AND d.id::text LIKE :q
                LIMIT 1
            """), {"q": f"{doc_id}%"}).fetchall()
        elif force:
            rows = session.execute(sql_text("""
                SELECT d.id::text, pm.title, pm.abstract, pm.year,
                       pm.authors, pm.keywords, pm.domains
                FROM documents d
                JOIN paper_metadata pm ON pm.document_id = d.id
                WHERE d.ingestion_status = 'complete' AND pm.title IS NOT NULL
                ORDER BY pm.year DESC NULLS LAST
            """)).fetchall()
        else:
            # Find papers that have a wiki page but no triples in
            # knowledge_graph. `wiki_pages.source_doc_ids` is a uuid[]
            # array (not JSONB — see sciknow/storage/models.py:462),
            # so the previous jsonb_build_array() comparison blew up
            # with "operator does not exist: uuid[] @> jsonb" and the
            # whole command silently returned zero rows from the
            # error-recovery path. Use native array-contains.
            rows = session.execute(sql_text("""
                SELECT d.id::text, pm.title, pm.abstract, pm.year,
                       pm.authors, pm.keywords, pm.domains
                FROM documents d
                JOIN paper_metadata pm ON pm.document_id = d.id
                WHERE d.ingestion_status = 'complete' AND pm.title IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM wiki_pages wp
                      WHERE wp.source_doc_ids @> ARRAY[d.id]::uuid[]
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM knowledge_graph kg
                      WHERE kg.source_doc_id = d.id
                  )
                ORDER BY pm.year DESC NULLS LAST
            """)).fetchall()
        existing_slugs = _load_existing_slugs(session)

    if not rows:
        console.print(
            "[green]Nothing to backfill.[/green] "
            "Every wiki paper already has KG triples (or no wiki pages exist)."
        )
        return

    console.print(
        f"\nExtracting KG triples for [bold]{len(rows)}[/bold] paper(s)…\n"
    )
    from rich.progress import (
        Progress as _Progress, SpinnerColumn as _SpC,
        BarColumn as _BC, MofNCompleteColumn as _MC,
        TextColumn as _TC, TimeElapsedColumn as _TEC,
    )

    total_triples = 0
    total_entities = 0
    failed = 0
    with _Progress(
        _SpC(), _TC("[progress.description]{task.description}"),
        _BC(), _MC(), _TEC(), console=console,
    ) as progress:
        task = progress.add_task("Extracting", total=len(rows))
        for did, title, abstract, year, authors, keywords, domains in rows:
            short_title = (title or did[:8])[:60]
            progress.update(task, description=f"[dim]{short_title}[/dim]")
            try:
                # Mirror `compile_paper_summary`'s extraction-call args.
                author_str = ", ".join(
                    a.get("name", "") if isinstance(a, dict) else str(a)
                    for a in (authors or [])[:5]
                )
                kw_str = ", ".join(keywords or [])
                dom_str = ", ".join(domains or [])
                # Load sections for the extraction context — these
                # yielded better triples in the original pipeline.
                from sciknow.storage.db import get_session as _gs
                with _gs() as session:
                    secs = session.execute(sql_text("""
                        SELECT section_type, content FROM paper_sections
                        WHERE document_id::text = :did
                        ORDER BY section_index
                    """), {"did": did}).fetchall()
                section_text = "\n\n".join(
                    f"[{s[0]}]\n{s[1][:3000]}" for s in secs
                )[:12000]
                slug = _slugify(f"{did[:8]}-{title or 'untitled'}")
                entities, kg = _extract_entities_and_kg(
                    did, slug, title, author_str, str(year or "n.d."),
                    kw_str, dom_str, abstract or "", section_text,
                    existing_slugs, model=model,
                )
                total_entities += len(entities or [])
                total_triples += kg
            except Exception as exc:
                failed += 1
                console.print(f"  [red]fail {did[:8]}:[/red] {exc}")
            progress.advance(task)

    console.print(
        f"\n[bold]Summary:[/bold] "
        f"[green]{total_triples} triple(s)[/green] + "
        f"[green]{total_entities} entity mention(s)[/green] "
        f"across {len(rows) - failed}/{len(rows)} papers."
        + (f"  [red]{failed} failed[/red]" if failed else "")
    )


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to answer from the wiki.")],
    context_k: int = typer.Option(8, "--context-k", "-k"),
    model: str | None = typer.Option(None, "--model"),
):
    """
    Answer a question from the compiled wiki.

    Searches wiki pages (not raw paper chunks) for pre-synthesized,
    cross-referenced knowledge.

    Examples:

      sciknow wiki query "what is total solar irradiance?"

      sciknow wiki query "how do cosmic rays affect cloud formation?"
    """
    from sciknow.cli import preflight
    preflight(qdrant=True)
    _check_wiki_table()

    from sciknow.core.wiki_ops import query_wiki

    console.print()
    gen = query_wiki(question, context_k=context_k, model=model)
    result = _consume_events(gen, console)
    console.print()

    if result and result.get("sources"):
        console.print("[dim]Sources:[/dim]")
        for s in result["sources"]:
            console.print(f"  [dim]{s}[/dim]")


@app.command()
def lint(
    deep: bool = typer.Option(False, "--deep",
                               help="Run LLM-based contradiction detection (slower)."),
    model: str | None = typer.Option(None, "--model"),
):
    """
    Check wiki health: broken links, stale pages, orphaned concepts,
    missing summaries, and optionally contradictions (--deep).

    Examples:

      sciknow wiki lint

      sciknow wiki lint --deep   # includes contradiction detection
    """
    from sciknow.core.wiki_ops import lint_wiki

    console.print("Running wiki lint...\n")
    gen = lint_wiki(deep=deep, model=model)
    result = _consume_events(gen, console)

    if result:
        n = result.get("issues_count", 0)
        if n == 0:
            console.print("\n[green]✓ Wiki is clean — no issues found.[/green]")
        else:
            console.print(f"\n[yellow]{n} issues found.[/yellow]")


@app.command(name="list")
def list_pages(
    page_type: str | None = typer.Option(None, "--type", "-t",
                                          help="Filter by type: paper_summary, concept, synthesis."),
    limit: int = typer.Option(50, "--limit", "-l"),
):
    """
    List all wiki pages.

    Examples:

      sciknow wiki list

      sciknow wiki list --type concept

      sciknow wiki list --type paper_summary --limit 20
    """
    _check_wiki_table()
    from sciknow.core.wiki_ops import list_pages as _list

    pages = _list(page_type=page_type)[:limit]

    if not pages:
        console.print("[yellow]No wiki pages found.[/yellow]")
        console.print("Run [bold]sciknow wiki compile[/bold] to build the wiki.")
        return

    table = Table(title="Wiki Pages", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Slug", ratio=3)
    table.add_column("Type", style="cyan", width=16)
    table.add_column("Words", justify="right", width=7)
    table.add_column("Sources", justify="right", width=8)
    table.add_column("Updated", style="dim", width=12)

    for p in pages:
        table.add_row(
            p["slug"][:50],
            p["page_type"],
            str(p["word_count"]),
            str(p["n_sources"]),
            p["updated_at"][:10] if p["updated_at"] else "",
        )
    console.print(table)
    console.print(f"[dim]{len(pages)} pages[/dim]")


@app.command()
def show(
    slug: Annotated[str, typer.Argument(help="Wiki page slug.")],
):
    """
    Display a wiki page in the terminal.

    Examples:

      sciknow wiki show total-solar-irradiance

      sciknow wiki show a3f2b1c4-zharkova-2024-solar-eigenvectors
    """
    from sciknow.core.wiki_ops import show_page

    page = show_page(slug)
    if not page:
        console.print(f"[red]Page not found:[/red] {slug}")
        raise typer.Exit(1)

    console.print()
    console.print(Markdown(page["content"]))
    console.print()
    console.print(f"[dim]{page['path']}[/dim]")


@app.command()
def graph(
    entity: Annotated[str, typer.Argument(help="Entity to explore (e.g. 'solar forcing', a paper slug).")],
    depth: int = typer.Option(1, "--depth", "-d", help="Graph traversal depth (1=direct, 2=two-hop)."),
    limit: int = typer.Option(30, "--limit", "-l"),
):
    """
    Explore the knowledge graph around an entity.

    Shows all triples where the entity appears as subject or object,
    plus connected entities at the specified depth.

    Examples:

      sciknow wiki graph "solar forcing"

      sciknow wiki graph "total solar irradiance" --depth 2

      sciknow wiki graph "zharkova" --limit 50
    """
    _check_wiki_table()

    from sqlalchemy import text
    from sciknow.storage.db import get_session

    # Check KG table exists
    try:
        with get_session() as sess:
            sess.execute(text("SELECT 1 FROM knowledge_graph LIMIT 0"))
    except Exception:
        console.print(
            "[red]The knowledge_graph table does not exist.[/red]\n"
            "Run: [bold]uv run alembic upgrade head[/bold]"
        )
        raise typer.Exit(1)

    pattern = f"%{entity.lower()}%"
    with get_session() as session:
        rows = session.execute(text("""
            SELECT subject, predicate, object, source_doc_id::text
            FROM knowledge_graph
            WHERE LOWER(subject) LIKE :pat OR LOWER(object) LIKE :pat
            ORDER BY subject, predicate
            LIMIT :lim
        """), {"pat": pattern, "lim": limit}).fetchall()

    if not rows:
        console.print(f"[yellow]No triples found for:[/yellow] {entity}")
        console.print("[dim]Run `sciknow wiki compile` to extract knowledge graph triples.[/dim]")
        raise typer.Exit(0)

    table = Table(title=f"Knowledge Graph: {entity}", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Subject", ratio=2)
    table.add_column("Predicate", style="cyan", ratio=1)
    table.add_column("Object", ratio=2)
    table.add_column("Source", style="dim", width=10)

    for subj, pred, obj, src in rows:
        table.add_row(subj[:40], pred, obj[:40], (src or "")[:8])
    console.print(table)
    console.print(f"[dim]{len(rows)} triples[/dim]")

    # Depth 2: follow connected entities
    if depth >= 2:
        connected = set()
        for subj, pred, obj, _ in rows:
            connected.add(subj)
            connected.add(obj)
        connected.discard(entity.lower())

        hop2_rows = []
        for ent in list(connected)[:10]:
            with get_session() as session:
                h2 = session.execute(text("""
                    SELECT subject, predicate, object
                    FROM knowledge_graph
                    WHERE (LOWER(subject) = :ent OR LOWER(object) = :ent)
                      AND LOWER(subject) NOT LIKE :orig AND LOWER(object) NOT LIKE :orig
                    LIMIT 5
                """), {"ent": ent, "orig": pattern}).fetchall()
                hop2_rows.extend(h2)

        if hop2_rows:
            console.print(f"\n[bold]2-hop connections:[/bold]")
            for subj, pred, obj in hop2_rows[:20]:
                console.print(f"  {subj[:35]} [cyan]{pred}[/cyan] {obj[:35]}")


@app.command()
def consensus(
    topic: Annotated[str, typer.Argument(help="Topic to map consensus for.")],
    model: str | None = typer.Option(None, "--model"),
):
    """
    Map the consensus landscape for a topic.

    Uses the knowledge graph and paper summaries to identify:
    - Key claims and which papers support/contradict them
    - Consensus level (strong/moderate/weak/contested)
    - Trends (growing/stable/declining/emerging)
    - Most debated sub-topics

    Saves the result as a wiki synthesis page.

    Examples:

      sciknow wiki consensus "solar forcing and climate"

      sciknow wiki consensus "cosmic ray cloud nucleation"
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)
    _check_wiki_table()

    from sciknow.core.wiki_ops import consensus_map

    console.print()
    gen = consensus_map(topic, model=model)

    result = None
    for event in gen:
        t = event.get("type")
        if t == "progress":
            console.print(f"[dim]{event.get('detail', '')}[/dim]")
        elif t == "consensus":
            data = event["data"]
            console.print()
            console.print(Rule(f"[bold]Consensus Map: {topic}[/bold]"))
            console.print()
            console.print(f"[dim]{data.get('summary', '')}[/dim]")
            console.print()

            for c in data.get("claims", []):
                level = c.get("consensus_level", "unknown")
                color = {"strong": "green", "moderate": "cyan", "weak": "yellow", "contested": "red"}.get(level, "dim")
                console.print(f"  [{color}]{level.upper()}[/{color}]  {c.get('claim', '')}")
                sup = c.get("supporting_papers", [])
                con = c.get("contradicting_papers", [])
                if sup:
                    console.print(f"    [green]Supports ({len(sup)}):[/green] {', '.join(sup[:3])}")
                if con:
                    console.print(f"    [red]Contradicts ({len(con)}):[/red] {', '.join(con[:3])}")
                console.print()

            debated = data.get("most_debated", [])
            if debated:
                console.print("[bold]Most debated:[/bold]")
                for d in debated:
                    console.print(f"  [yellow]- {d}[/yellow]")
        elif t == "completed":
            result = event
        elif t == "error":
            console.print(f"[red]Error:[/red] {event.get('message', '')}")

    if result:
        console.print(
            f"\n[green]✓ Consensus map saved:[/green] [[{result.get('slug', '')}]] "
            f"({result.get('claims', 0)} claims)"
        )


@app.command()
def synthesize(
    topic: Annotated[str, typer.Argument(help="Topic to synthesize.")],
    model: str | None = typer.Option(None, "--model"),
):
    """
    Generate a synthesis overview page from existing wiki pages.

    Finds all paper summaries and concept pages related to the topic
    and writes a comparative analysis.

    Examples:

      sciknow wiki synthesize "solar forcing and climate"

      sciknow wiki synthesize "cosmic ray cloud nucleation"
    """
    from sciknow.cli import preflight
    preflight(qdrant=True)
    _check_wiki_table()

    from sciknow.core.wiki_ops import compile_synthesis

    console.print()
    gen = compile_synthesis(topic, model=model)
    result = _consume_events(gen, console)
    console.print()

    if result:
        console.print(
            f"\n[green]✓ Synthesis page created:[/green] [[{result.get('slug', '')}]] "
            f"({result.get('word_count', 0)} words)"
        )
