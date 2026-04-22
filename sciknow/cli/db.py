import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, MofNCompleteColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box
from rich.table import Table

app = typer.Typer(help="Database and infrastructure management.")
console = Console()


# ── Phase 49.1 — downloads/ hygiene helpers ────────────────────────────
# Every expand run deposits PDFs directly into <download_dir>/*.pdf.
# Historically they stayed there forever: successful ingests kept the
# PDF next to failed ones, users couldn't tell at a glance which were
# still actionable, and re-running expand kept re-trying PDFs whose
# ingest had already failed for intrinsic reasons (mangled PDF, image-
# only scan, MinerU timeout). These helpers move each PDF to one of
# two subfolders right after the pipeline's verdict lands, plus a
# persistent `.ingest_failed` cache so a second run skips the known-
# bad ones unless the user passes `--retry-failed`.

_PROCESSED_SUBDIR = "processed"
_FAILED_SUBDIR = "failed_ingest"


def _normalise_title_for_dedup(title: str) -> str:
    """Phase 54.6.51 — thin re-export. The actual implementation lives in
    ``sciknow.ingestion.references`` so ingestion-layer modules can use
    it without a circular import via this CLI module. L1
    ``l1_phase49_1_title_dedup_plumbing`` still grips on the old
    attribute path, hence the alias here.
    """
    from sciknow.ingestion.references import normalise_title_for_dedup as _impl
    return _impl(title)


def _move_downloaded_pdf(
    dest: Path, outcome: str, download_dir: Path, error_msg: str = ""
) -> Path | None:
    """Move a PDF into `processed/` or `failed_ingest/` based on the
    ingest outcome. Returns the new path (or None if nothing moved —
    e.g. file was already missing). Safe to call repeatedly: existing
    files at the target are overwritten. Never raises.

    `outcome` is one of: 'done', 'skipped' (already in DB), 'failed'.
    """
    if not dest.exists():
        return None
    try:
        if outcome == "failed":
            sub = download_dir / _FAILED_SUBDIR
        else:
            sub = download_dir / _PROCESSED_SUBDIR
        sub.mkdir(parents=True, exist_ok=True)
        target = sub / dest.name
        # os.replace is atomic on the same filesystem and silently
        # overwrites the target if present — perfect for this flow.
        os.replace(dest, target)
        if outcome == "failed" and error_msg:
            # Drop a sibling .error.txt so the user can see WHY the
            # ingest failed without digging through expand.log.
            try:
                (sub / (dest.stem + ".error.txt")).write_text(
                    error_msg[:4000], encoding="utf-8"
                )
            except Exception:
                pass
        return target
    except Exception:
        # File-system hiccup (cross-device rename, permission) —
        # leave the PDF where it was rather than fail the whole run.
        return None


# ── Phase 49 — RRF-fused expand ranker orchestrator ───────────────────────
# Lives at module scope so L1 tests can import it without instantiating
# the whole Typer app. See docs/EXPAND_RESEARCH.md for the design
# rationale + per-signal trade-offs; the per-signal math lives in
# sciknow/ingestion/expand_ranker.py.

def _run_rrf_ranker(
    *,
    downloadable: list,
    papers: list,
    existing_dois: set,
    budget: int,
    no_openalex: bool,
    no_s2: bool,
    dry_run: bool,
    shortlist_tsv: Path | None,
    download_dir: Path,
    console,
):
    """Upgrade the already-cosine-filtered `downloadable` list of
    `Reference` objects into a ranked, RRF-fused, hard-filtered list.

    Returns `(refs_to_download, ranked_features)`:
      - refs_to_download : Reference[]  — top-`budget` after filters + RRF
      - ranked_features  : CandidateFeatures[]  — full list with scores
                            (for --shortlist-tsv)

    Downloads / ingests are NOT done here — the caller still runs the
    existing download+ingest pipeline on the returned Reference list.
    """
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from sciknow.ingestion import expand_apis, expand_filters, expand_ranker
    from sciknow.ingestion.expand_ranker import (
        CandidateFeatures,
        apply_author_overlap,
        apply_one_timer_filter,
        bibliographic_coupling,
        compute_corpus_side_counts,
        enrich_from_openalex_work,
        local_pagerank,
        score_via_rrf,
        write_shortlist_tsv,
    )

    # Seed set = existing corpus with DOIs (needed for co-citation /
    # coupling — we need OpenAlex IDs for seeds, and we only have
    # those via DOI lookup).
    seed_dois = [p[0] for p in papers if p[0]]  # pm.doi column

    console.print(
        f"\n[bold]RRF ranker[/bold] — {len(downloadable)} candidates "
        f"→ target top {budget}"
    )

    # ── 1. Build CandidateFeatures from the cosine-prefiltered pool.
    #     Cap to 3×budget so we don't waste API quota on long tails.
    pool_cap = max(budget * 3, 60)
    cosine_pool = sorted(
        downloadable,
        key=lambda r: getattr(r, "_relevance_score", 0.0),
        reverse=True,
    )[:pool_cap]
    feats: list[CandidateFeatures] = []
    ref_by_key: dict[str, object] = {}
    for r in cosine_pool:
        f = CandidateFeatures(
            doi=(r.doi or ""),
            arxiv_id=(r.arxiv_id or ""),
            title=(r.title or ""),
            year=int(r.year or 0),
            bge_m3_cosine=float(getattr(r, "_relevance_score", 0.0) or 0.0),
        )
        feats.append(f)
        ref_by_key[f.key] = r

    # Count how many seeds reference each candidate → corpus_cite_count.
    seed_ref_counts: Counter = Counter()
    for r in cosine_pool:
        pass  # corpus_cite_count derivation is on the seed-ref side below

    # Recover corpus-cite counts from the candidate-gathering step: each
    # candidate `Reference` was already deduped across seeds but we lose
    # the count. Recompute by walking the full candidate list before
    # dedup — cheap (dict lookup) if the caller kept per-ref refs.
    # Here we approximate by counting re-appearances in `downloadable`
    # (Reference is deduped by key so this is always 1); a precise count
    # would need a refactor. Keep as 1 for now; one-timer filter still
    # fires when external_cite_count < 5.
    for f in feats:
        f.corpus_cite_count = 1

    # Phase 54.6.111 (Tier 1 #3) — populated in the OpenAlex branch;
    # stays empty in --no-openalex mode so apply_mmr short-circuits.
    concept_sets: dict[str, set[str]] = {}

    if no_openalex:
        console.print(
            "[yellow]  --no-openalex: skipping OpenAlex + PageRank + "
            "co-citation signals.[/yellow]"
        )
        # Without OpenAlex: only cosine + one-timer filter apply.
        # external_cite_count stays at 0 → one-timer drops everyone
        # with corpus_cite_count=1. Disable the one-timer filter in
        # this degraded mode by pre-populating external_cite_count=999.
        for f in feats:
            f.external_cite_count = 999
    else:
        # ── 2. Parallel OpenAlex work fetch ──────────────────────────────────
        console.print(f"  fetching OpenAlex metadata for {len(feats)} candidates…")
        oa_works = expand_ranker._parallel_openalex_works(
            [(f.doi, f.arxiv_id) for f in feats],
            max_workers=8,
        )
        # Phase 54.6.111 (Tier 1 #3) — build candidate concept sets
        # once during the OpenAlex fetch pass so apply_mmr can use them
        # as its diversity signal without a second API call.
        for f in feats:
            w = oa_works.get(f.key) or oa_works.get((f.doi or f"arxiv:{f.arxiv_id}").lower())
            enrich_from_openalex_work(f, w)
            cs = {
                (c.get("display_name") or "").lower()
                for c in ((w or {}).get("concepts") or [])
                if c.get("display_name")
            }
            if cs:
                concept_sets[f.key] = cs

        # ── 3. Hard filters (retraction / predatory / doc-type) ──────────────
        hard_dropped = 0
        for f in feats:
            w = oa_works.get(f.key)
            drop, reason = expand_filters.apply_hard_filters(w)
            if drop:
                f.hard_drop_reason = reason
                f.decisions.append(f"HARD_DROP: {reason}")
                hard_dropped += 1
        if hard_dropped:
            console.print(f"  hard filters dropped {hard_dropped} candidates")

        # ── 4. Seed enrichment: fetch OpenAlex works for seeds to get refs ──
        #     Only bother with seeds that have a DOI (needed to resolve).
        seed_dois_to_query = seed_dois[:100]  # cap for API politeness
        seed_oa_works: dict[str, dict] = {}
        if seed_dois_to_query:
            console.print(
                f"  fetching OpenAlex metadata for {len(seed_dois_to_query)} seeds…"
            )
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {
                    pool.submit(expand_apis.fetch_openalex_work, d): d
                    for d in seed_dois_to_query
                }
                for fut in as_completed(futures):
                    d = futures[fut]
                    try:
                        w = fut.result()
                        if w:
                            seed_oa_works[d.lower()] = w
                    except Exception:
                        pass

        # ── 5. Bibliographic coupling ───────────────────────────────────────
        seed_refs_union: set[str] = set()
        seed_refs_size = 0
        for w in seed_oa_works.values():
            refs = w.get("referenced_works") or []
            seed_refs_size += len(refs)
            seed_refs_union.update(refs)
        for f in feats:
            if f.hard_drop_reason:
                continue
            w = oa_works.get(f.key)
            if not w:
                continue
            cand_refs = w.get("referenced_works") or []
            f.bib_coupling = bibliographic_coupling(
                cand_refs, seed_refs_union, seed_refs_size
            )

        # ── 6. Co-citation via seed cited-by sets ──────────────────────────
        #     Cheaper to fetch once per seed than once per candidate.
        if seed_oa_works:
            console.print(f"  fetching cited-by for {len(seed_oa_works)} seeds (co-citation)…")
            forward_refs: list[list[str]] = []
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(expand_apis.fetch_openalex_cited_by, w.get("id"), per_page=100, max_pages=2)
                    for w in seed_oa_works.values()
                    if w.get("id")
                ]
                for fut in as_completed(futures):
                    try:
                        papers_citing_seed = fut.result()
                        for p in papers_citing_seed:
                            refs = p.get("referenced_works") or []
                            if refs:
                                forward_refs.append(refs)
                    except Exception:
                        pass
            if forward_refs:
                co_counts: Counter = Counter()
                cand_oa_ids = {f.openalex_id for f in feats if f.openalex_id}
                for refs in forward_refs:
                    for r in refs:
                        if r in cand_oa_ids:
                            co_counts[r] += 1
                for f in feats:
                    if f.openalex_id:
                        f.co_citation = int(co_counts.get(f.openalex_id, 0))

        # ── 7. Local PageRank on the depth-2 citation subgraph ─────────────
        #     Nodes: seed OA IDs, candidate OA IDs, their 1-hop refs.
        #     Edges: (src, tgt) where src cites tgt.
        console.print("  building depth-2 citation subgraph for PageRank…")
        edges: list[tuple[str, str]] = []
        node_set: set[str] = set()
        for w in seed_oa_works.values():
            sid = w.get("id")
            if not sid:
                continue
            node_set.add(sid)
            for ref in (w.get("referenced_works") or []):
                node_set.add(ref)
                edges.append((sid, ref))
        for f in feats:
            if not f.openalex_id:
                continue
            node_set.add(f.openalex_id)
            w = oa_works.get(f.key)
            if w:
                for ref in (w.get("referenced_works") or []):
                    node_set.add(ref)
                    edges.append((f.openalex_id, ref))
        if node_set and edges:
            pr = local_pagerank(list(node_set), edges)
            for f in feats:
                f.pagerank = float(pr.get(f.openalex_id, 0.0)) if f.openalex_id else 0.0
        else:
            console.print("  [dim]  (subgraph empty — PageRank skipped)[/dim]")

        # ── 8. Semantic Scholar isInfluential + intents + contexts ─────────
        if not no_s2:
            survivors = [f for f in feats if not f.hard_drop_reason and f.doi]
            if survivors:
                console.print(
                    f"  fetching Semantic Scholar citations for {len(survivors)} "
                    "survivors (1 RPS throttled)…"
                )
                # Phase 54.6.113 (Tier 2 #1) — compute the corpus
                # centroid once here so the citation-context cosine
                # uses the same anchor as bge_m3_cosine. Cheap after
                # compute_corpus_centroid caches; returns None when
                # the abstracts collection is empty (new project).
                from sciknow.retrieval.relevance import compute_corpus_centroid
                from sciknow.retrieval import citation_context as _cc
                base_anchor = compute_corpus_centroid()
                # Phase 54.6.115 (Tier 2 #3) — bias the anchor with
                # the project's ±marks. Tiny cost (one Qdrant scroll
                # per feedback entry, cached in-process); substantial
                # ranking benefit as labels accumulate.
                try:
                    from sciknow.core.project import get_active_project
                    from sciknow.core import expand_feedback as _fb
                    from sciknow.retrieval.feedback_anchor import bias_anchor as _bias
                    fb = _fb.load(get_active_project().root)
                    if fb.positive or fb.negative:
                        corpus_anchor, stats = _bias(
                            base_anchor, fb.positive, fb.negative,
                        )
                        if stats.get("pos") or stats.get("neg"):
                            console.print(
                                f"  [dim]  anchor biased by feedback: +{stats['pos']} / -{stats['neg']}[/dim]"
                            )
                    else:
                        corpus_anchor = base_anchor
                except Exception as exc:  # noqa: BLE001
                    logger.debug("feedback anchor bias failed: %s", exc)
                    corpus_anchor = base_anchor
                ctx_hits = 0
                for f in survivors:
                    data = expand_apis.fetch_s2_citations(f.doi)
                    f.influential_cite_count = expand_apis.count_influential_from_corpus(
                        data, existing_dois
                    )
                    # Citation-context embedding + cosine vs centroid.
                    bundle = _cc.build_bundle(data)
                    f.citation_context_n = bundle.n_contexts
                    f.citation_context_cosine = _cc.context_cosine(
                        bundle.embedding, corpus_anchor,
                    )
                    if bundle.first_preview:
                        f.decisions.append(
                            f"cite_ctx[{bundle.n_contexts}]: {bundle.first_preview[:80]}"
                        )
                    if bundle.n_contexts:
                        ctx_hits += 1
                console.print(
                    f"  [dim]  citation-context embeddings computed for "
                    f"{ctx_hits}/{len(survivors)} candidates[/dim]"
                )

        # ── 9. Author overlap ──────────────────────────────────────────────
        #     From paper_metadata (existing corpus) + OpenAlex authorships
        #     on each candidate.
        from sciknow.storage.db import get_session
        from sqlalchemy import text as sql_text
        corpus_author_counts: Counter = Counter()
        with get_session() as session:
            rows = session.execute(sql_text(
                "SELECT authors FROM paper_metadata WHERE authors IS NOT NULL"
            )).fetchall()
        for (authors,) in rows:
            # authors column is JSON array of strings (based on ingestion flow).
            try:
                if isinstance(authors, str):
                    authors_list = json.loads(authors)
                else:
                    authors_list = authors or []
            except Exception:
                authors_list = []
            for a in authors_list:
                if isinstance(a, str):
                    corpus_author_counts[a.strip().lower()] += 1
        candidate_authors: dict[str, list[str]] = {}
        for f in feats:
            w = oa_works.get(f.key)
            if not w:
                continue
            names = []
            for auth in (w.get("authorships") or []):
                display = (auth.get("author") or {}).get("display_name") or ""
                if display:
                    names.append(display.strip().lower())
            candidate_authors[f.key] = names
        apply_author_overlap(feats, dict(corpus_author_counts), candidate_authors)

        # ── 10. Corpus-side cite count derivation ──────────────────────────
        cited_by_lookup = {f.key: f.cited_by_count for f in feats}
        compute_corpus_side_counts(feats, cited_by_lookup=cited_by_lookup)

    # ── 11. One-timer filter ──────────────────────────────────────────────
    apply_one_timer_filter(feats)

    # ── 12. RRF fusion ────────────────────────────────────────────────────
    ranked = score_via_rrf(feats)
    # Phase 54.6.111 (Tier 1 #3) — MMR diversity re-rank over the top
    # of the RRF-sorted list so the round's top-N spans multiple
    # topics. lambda=0.7 keeps 70% of RRF's intent. Only active when
    # we have OpenAlex concepts (no-openalex mode skips).
    if not no_openalex and concept_sets:
        from sciknow.ingestion.expand_ranker import apply_mmr
        ranked = apply_mmr(ranked, concept_sets, lambda_=0.7, top_k=budget * 2)
    kept = [f for f in ranked if not f.hard_drop_reason]
    dropped = [f for f in ranked if f.hard_drop_reason]
    console.print(
        f"  [green]kept {len(kept)}[/green]  [red]dropped {len(dropped)}[/red]  "
        f"(→ taking top {min(budget, len(kept))} for download)"
    )

    # ── 13. Shortlist TSV for HITL review ────────────────────────────────
    tsv_path = shortlist_tsv
    if tsv_path is None and dry_run:
        tsv_path = download_dir / "expand_shortlist.tsv"
    if tsv_path:
        write_shortlist_tsv(ranked, tsv_path)
        console.print(f"  [dim]shortlist TSV written to {tsv_path}[/dim]")

    # ── 14. Return the top-`budget` Reference objects for the download phase
    top_feats = kept[:budget]
    top_refs = [ref_by_key[f.key] for f in top_feats if f.key in ref_by_key]
    return top_refs, ranked


# ── Phase 54.6.114 (Tier 2 #2) — agentic question-driven expansion ──────

def _run_agentic_expand(
    *,
    question: str,
    download_dir,
    max_rounds: int,
    budget_per_gap: int,
    doc_threshold: int,
    strategy: str,
    delay: float,
    resolve: bool,
    ingest: bool,
    dry_run: bool,
    workers: int,
    rrf_no_openalex: bool,
    rrf_no_s2: bool,
    cleanup: bool,
    retry_failed: bool,
    resume: bool = False,
) -> None:
    """Orchestrator for ``sciknow db expand --question "..."``.

    Plans → checks coverage → runs per-sub-topic expansion via the
    existing Phase 49 pipeline (``sciknow db expand --relevance-query``
    as a subprocess) → re-checks → stops when covered or max_rounds.
    """
    import subprocess
    import sys as _sys
    from sciknow.ingestion.agentic_expand import run_agentic_expansion

    console.print(f"\n[bold]Agentic expansion[/bold]  question: "
                  f"[cyan]{question[:120]}[/cyan]\n")

    def _execute_round(gaps: list[str], budget: int, round_n: int) -> dict:
        """Run ``db expand --relevance-query <gap>`` for each gap in a
        subprocess so we don't tangle Qdrant/DB/asyncio state. Per-sub-
        topic budget; one crash doesn't kill the round."""
        stats: dict = {"subtopics": [], "downloaded_total": 0}
        for i, topic in enumerate(gaps, start=1):
            console.print(f"\n  [bold]Round {round_n} sub-topic {i}/{len(gaps)}[/bold]: {topic}")
            argv = [
                _sys.executable, "-m", "sciknow.cli.main", "db", "expand",
                "--strategy", strategy,
                "--relevance-query", topic,
                "--budget", str(budget),
                "--delay", str(delay),
                "--workers", str(workers),
            ]
            if not resolve:
                argv.append("--no-resolve")
            if not ingest:
                argv.append("--no-ingest")
            if dry_run:
                argv.append("--dry-run")
            if rrf_no_openalex:
                argv.append("--no-openalex")
            if rrf_no_s2:
                argv.append("--no-semantic-scholar")
            if not cleanup:
                argv.append("--no-cleanup")
            if retry_failed:
                argv.append("--retry-failed")
            try:
                res = subprocess.run(argv, check=False)
                stats["subtopics"].append({
                    "subtopic": topic,
                    "return_code": res.returncode,
                })
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [red]sub-topic failed: {exc}[/red]")
                stats["subtopics"].append({
                    "subtopic": topic,
                    "error": str(exc)[:200],
                })
        return stats

    # Phase 54.6.124 — pass project_root so state checkpoints land in
    # the active project's data dir, and threading `resume` through so
    # --resume picks up where a prior run stopped.
    from sciknow.core.project import get_active_project
    _proj_root = get_active_project().root

    for event in run_agentic_expansion(
        question,
        max_rounds=max_rounds,
        budget_per_gap=budget_per_gap,
        doc_threshold=doc_threshold,
        execute_round_callback=_execute_round,
        project_root=_proj_root,
        resume=resume,
    ):
        t = event.get("type")
        if t == "resumed":
            console.print(
                f"\n[cyan]Resumed from round {event.get('from_round')}"
                f"[/cyan]  ({len(event.get('prior_rounds') or [])} prior round(s) on record)"
            )
            continue
        if t == "progress":
            console.print(f"  [dim]… {event.get('detail', event.get('stage', ''))}[/dim]")
        elif t == "decomp":
            subs = event.get("subtopics") or []
            label = "[Refined] " if event.get("replanned") else ""
            console.print(f"\n  {label}[bold]Sub-topics[/bold] ({len(subs)}):")
            for i, s in enumerate(subs, start=1):
                console.print(f"    {i}. {s}")
        elif t == "coverage":
            rn = event.get("round")
            console.print(f"\n  [bold]Coverage (round {rn}):[/bold]")
            for d in event.get("data") or []:
                badge = "[green]✓[/green]" if d.get("covered") else "[yellow]·[/yellow]"
                console.print(
                    f"    {badge} {d['n_papers']:>3} papers (top {d['top_score']:.2f})  {d['subtopic']}"
                )
        elif t == "round_start":
            console.print(f"\n  [bold]Expanding {len(event['gaps'])} gap sub-topic(s) "
                          f"(budget {event['budget']} each)…[/bold]")
        elif t == "round_done":
            n = sum(1 for s in (event.get("stats") or {}).get("subtopics", [])
                    if s.get("return_code") == 0)
            console.print(f"\n  [green]Round {event['round']}: "
                          f"{n} sub-topic(s) completed[/green]")
        elif t == "stopped":
            console.print(f"\n[bold green]✓ Stopped:[/bold green] {event.get('reason')}")
            fc = event.get("final_coverage") or []
            if fc:
                covered = sum(1 for c in fc if c.get("n_papers", 0) >= doc_threshold)
                console.print(f"  Final: {covered}/{len(fc)} sub-topics covered "
                              f"(≥{doc_threshold} papers each).")
        elif t == "error":
            console.print(f"\n[red]✗ {event.get('message')}[/red]")
            return


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
            # Phase 43d — project-aware data path.
            dl_dir = settings.data_dir / "downloads"
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
                # Phase 43d — restores into the active project's dir.
                dl_dest = settings.data_dir / "downloads"
                dl_dest.mkdir(parents=True, exist_ok=True)
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
def failures(
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Max error classes to show.",
    ),
    stage: str = typer.Option(
        None,
        "--stage",
        help="Filter to one stage: converting, metadata_extraction, "
        "chunking, embedding, or a refresh step name.",
    ),
    since: str = typer.Option(
        None,
        "--since",
        help="ISO8601 timestamp — only consider jobs created after this.",
    ),
):
    """Aggregate the `ingestion_jobs` table by (stage, error signature).

    Phase 54.6.205 — fulfils roadmap item 3.11.6 (failure-mode clinic).
    The `ingestion_jobs` table already records every stage outcome;
    this command surfaces it as a grouped failure class table so the
    user can see "7 papers failed metadata_extraction with an LLM
    timeout, 4 papers failed converting with MinerU OOM" at a glance.

    Error signature = the first 80 chars of the error message,
    whitespace-collapsed. Same underlying cause usually produces the
    same prefix, which is enough to cluster visually without NLP.
    """
    import re as _re
    from datetime import datetime as _dt
    from sqlalchemy import text

    from sciknow.storage.db import get_session

    params: dict = {"limit": int(limit)}
    where = ["status = 'failed'"]
    if stage:
        where.append("stage = :stage")
        params["stage"] = stage
    if since:
        try:
            _dt.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            console.print(f"[red]Invalid --since:[/red] {since!r} — expected ISO8601")
            raise typer.Exit(2)
        where.append("created_at >= CAST(:since AS timestamptz)")
        params["since"] = since
    where_sql = " AND ".join(where)

    # Extract a short signature from details->>'error'. details is
    # JSONB; `jsonb_path_query_first` gracefully handles missing keys.
    sql = text(f"""
        WITH j AS (
            SELECT
                stage,
                COALESCE(
                    substring(regexp_replace(
                        (details->>'error'),
                        '\\s+', ' ', 'g'
                    ) FROM 1 FOR 80),
                    '(no error text)'
                ) AS signature,
                created_at,
                document_id
            FROM ingestion_jobs
            WHERE {where_sql}
        )
        SELECT
            stage,
            signature,
            COUNT(*) AS n,
            MIN(created_at) AS first_seen,
            MAX(created_at) AS last_seen
        FROM j
        GROUP BY stage, signature
        ORDER BY n DESC, last_seen DESC
        LIMIT :limit
    """)

    with get_session() as session:
        rows = session.execute(sql, params).fetchall()

    if not rows:
        console.print("[green]No failed ingestion jobs[/green] match that filter.")
        return

    # Rich table
    from rich.table import Table

    t = Table(
        title=f"Ingestion failures"
        + (f" · stage={stage}" if stage else "")
        + (f" · since={since}" if since else "")
        + f" · top {len(rows)}",
        show_lines=False,
    )
    t.add_column("Stage", style="cyan")
    t.add_column("N", justify="right", style="magenta")
    t.add_column("Last seen", style="dim")
    t.add_column("Signature")
    for r in rows:
        t.add_row(
            r.stage,
            str(r.n),
            r.last_seen.strftime("%Y-%m-%d %H:%M") if r.last_seen else "",
            (r.signature or "")[:100],
        )
    console.print(t)

    total = sum(r.n for r in rows)
    console.print(
        f"\n[dim]Total failed jobs in window: {total}. "
        "Stage names match the ingestion state machine "
        "(converting / metadata_extraction / chunking / embedding) "
        "or the refresh step ('db enrich', 'catalog raptor build', "
        "'db caption-visuals', …). Pass --stage to narrow.[/dim]"
    )


def _doctor_compute(snap: dict) -> tuple[str, int, dict]:
    """Phase 54.6.253 — classify a snapshot into a doctor verdict.
    Pure function so the watch-mode re-render and the one-shot
    render share one source of truth. Returns (verdict, exit_code,
    counts)."""
    alerts = snap.get("alerts") or []
    errors = [a for a in alerts if a.get("severity") == "error"]
    warns = [a for a in alerts if a.get("severity") == "warn"]
    infos = [a for a in alerts if a.get("severity") == "info"]
    if errors:
        return ("FAIL", 2, {"error": errors, "warn": warns, "info": infos})
    if warns:
        return ("WARN", 1, {"error": errors, "warn": warns, "info": infos})
    return ("OK", 0, {"error": errors, "warn": warns, "info": infos})


def _render_doctor(snap: dict):
    """Phase 54.6.271 — render a doctor snapshot as a Rich renderable.

    Returns a ``rich.console.Group`` of Text lines so ``watch`` mode
    can repaint it via ``rich.live.Live`` without sprinkling
    ``console.print`` side-effects through the function.
    """
    from rich.console import Group
    from rich.text import Text

    verdict, _exit, counts = _doctor_compute(snap)
    errors, warns, infos = counts["error"], counts["warn"], counts["info"]

    palette = {
        "OK": ("bright_green", "✓"),
        "WARN": ("yellow", "⚠"),
        "FAIL": ("bright_red", "✗"),
    }
    colour, icon = palette[verdict]
    project_slug = (snap.get("project") or {}).get("slug") or "?"
    health = snap.get("health_score") or {}
    hs = health.get("score")
    hs_colour = ""
    if hs is not None:
        hs_colour = (
            "bright_green" if hs >= 90 else
            "yellow" if hs >= 60 else "bright_red"
        )

    lines: list = []
    head = Text()
    head.append(f"{icon} {verdict}  ", style=f"bold {colour}")
    head.append("project ", style="dim")
    head.append(project_slug, style="cyan")
    if hs is not None:
        head.append("  ·  health ", style="dim")
        head.append(f"{hs}/100", style=hs_colour)
    head.append("  ·  errors ", style="dim")
    head.append(str(len(errors)), style="red")
    head.append("  warn ", style="dim")
    head.append(str(len(warns)), style="yellow")
    head.append("  info ", style="dim")
    head.append(str(len(infos)), style="blue")
    lines.append(head)

    gpus = snap.get("gpu") or []
    host = snap.get("host") or {}
    disk_free = snap.get("disk_free") or {}
    hw_parts = []
    if gpus:
        g0 = gpus[0]
        hw_parts.append(
            f"GPU {g0.get('utilization_pct', 0)}% · "
            f"{g0.get('temperature_c', '?')}°C"
        )
    if host.get("mem_pct"):
        hw_parts.append(f"RAM {host['mem_pct']:.0f}%")
    if disk_free.get("free_mb") and disk_free.get("total_mb"):
        free_pct = (
            disk_free["free_mb"] / disk_free["total_mb"] * 100
        )
        hw_parts.append(
            f"disk {disk_free['free_mb'] / 1024:.0f}G free "
            f"({free_pct:.0f}%)"
        )
    if hw_parts:
        lines.append(Text("  ·  ".join(hw_parts), style="dim"))

    def _group(label: str, items: list, tone: str):
        if not items:
            return
        lines.append(Text(""))
        lines.append(Text(label, style=f"bold {tone}"))
        for a in items:
            ln = Text(f"  • ", style=tone)
            ln.append(f"({a.get('code')}) ", style="dim")
            ln.append(a.get("message", ""), style=tone)
            lines.append(ln)
            act = a.get("action")
            if act:
                aln = Text(f"      $ ", style="dim")
                aln.append(act, style="bright_cyan")
                lines.append(aln)

    _group("Errors",   errors, "red")
    _group("Warnings", warns,  "yellow")
    _group("Info",     infos,  "blue")

    if verdict == "OK":
        lines.append(Text(""))
        lines.append(Text("All systems green — safe to proceed.", style="green"))

    return Group(*lines)


@app.command()
def doctor(
    json_out: bool = typer.Option(
        False, "--json",
        help="Emit machine-readable JSON (same fields the CLI "
             "renderer shows). Pipe through `jq` to script against.",
    ),
    watch: int = typer.Option(
        0, "--watch",
        help="Phase 54.6.271 — repaint the verdict + alerts every "
             "N seconds (Rich Live; no alt-screen, since the doctor "
             "view is short). 0 (default) = one-shot mode. Exit "
             "code is always the final snapshot's worst-severity "
             "classification; Ctrl+C exits with code 130.",
    ),
) -> None:
    """Phase 54.6.253 — readiness / health check.

    Thin wrapper over the monitor's alert aggregator. Prints a
    single-line traffic-light verdict (ALL GREEN / N warnings / M
    errors) plus a grouped-by-severity list of every alert, then
    exits with a shell code tied to the worst severity:

      * 0  — no alerts, or only info-level
      * 1  — at least one warn-level alert
      * 2  — at least one error-level alert

    Typical uses:

      uv run sciknow db doctor                 # check before heavy job
      uv run sciknow db doctor && sciknow ingest directory ./papers/
      uv run sciknow db doctor --json | jq .
      uv run sciknow db doctor --watch 5       # live readiness tick (54.6.271)

    Runs against the same `collect_monitor_snapshot` used by
    ``sciknow db monitor`` and ``/api/monitor``; picks up every
    alert class the dashboard does (stuck_ingest, embed_drift,
    missing_model, backup_stale, config_drift, disk_critical, …).
    The `doctor` exit code is the primary signal — pipelining it
    in shell is faster than eyeballing the monitor.
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    import json as _json
    import time as _time
    from sciknow.core.monitor import collect_monitor_snapshot

    snap = collect_monitor_snapshot()
    verdict, exit_code, counts = _doctor_compute(snap)

    if json_out:
        console.print(_json.dumps({
            "verdict": verdict,
            "exit_code": exit_code,
            "counts": {
                "error": len(counts["error"]),
                "warn": len(counts["warn"]),
                "info": len(counts["info"]),
            },
            "alerts": snap.get("alerts") or [],
            "project": (snap.get("project") or {}).get("slug"),
            "snapshotted_at": snap.get("snapshotted_at"),
            "health_score": (snap.get("health_score") or {}).get("score"),
        }, indent=2, default=str))
        raise typer.Exit(exit_code)

    # Phase 54.6.271 — live watch mode. Uses Rich Live (no alt-
    # screen; doctor output is short enough to fit in the scrolling
    # region). Exits 130 on Ctrl+C per shell convention.
    if watch > 0:
        from rich.live import Live
        try:
            with Live(
                _render_doctor(snap), console=console,
                refresh_per_second=max(1, min(4, int(round(1 / max(watch, 0.25))))),
                screen=False,
            ) as live:
                while True:
                    _time.sleep(watch)
                    snap = collect_monitor_snapshot()
                    live.update(_render_doctor(snap))
        except KeyboardInterrupt:
            raise typer.Exit(130)

    console.print(_render_doctor(snap))
    raise typer.Exit(exit_code)


@app.command()
def monitor(
    days: int = typer.Option(
        14, "--days",
        help="Trailing window for throughput + LLM usage panels.",
    ),
    watch: int = typer.Option(
        0, "--watch",
        help="If >0, re-render every N seconds until Ctrl+C. "
             "Useful during active ingestion ('watch mode').",
    ),
    json_out: bool = typer.Option(
        False, "--json",
        help="Emit machine-readable JSON instead of Rich panels. "
             "Shape matches /api/monitor so scripts can consume "
             "either source interchangeably.",
    ),
    filter_str: str = typer.Option(
        "", "--filter",
        help="Phase 54.6.254 — case-insensitive substring match "
             "applied to recent_activity and top_failures rows. "
             "Use for triage: `--filter 'timeout'` to find every "
             "job that timed out, `--filter 'foo.pdf'` for a "
             "specific paper. Blank = show everything (default).",
    ),
    log_tail: int = typer.Option(
        0, "--log-tail",
        help="Phase 54.6.260 — after rendering the dashboard, "
             "print the last N lines of data_dir/sciknow.log. "
             "Useful during ingestion debugging without opening a "
             "second terminal. 0 (default) = skip.",
    ),
    alerts_md: bool = typer.Option(
        False, "--alerts-md",
        help="Phase 54.6.268 — print just the current alerts as a "
             "Markdown block (no dashboard, no JSON). Suitable for "
             "pasting into Slack / Linear / a GitHub issue. Honours "
             "--filter.",
    ),
    compact: bool = typer.Option(
        False, "--compact",
        help="Phase 54.6.270 — trimmed 1-page view: verdict + health "
             "chip + services + alerts + active jobs only. Skips the "
             "heavy panels (GPU / corpus / Qdrant / storage). Useful "
             "on narrow terminals, tmux status bars, and `watch` "
             "without losing the must-read signals.",
    ),
):
    """Phase 54.6.230 — unified live monitor.

    One top-level view that composes everything:

      * Project + corpus counts (documents / chunks / citations /
        visuals / KG triples / wiki pages / institutions).
      * Converter backend distribution (key migration signal
        post-VLM-Pro — every complete row should flip to
        `mineru-vlm-pro-vllm`).
      * Ingestion stage timing (p50 / p95 / mean), failure rate,
        trailing throughput.
      * Qdrant collection sizes + per-collection vector fields
        (catches "is ColBERT actually populated on abstracts?").
      * GPU state (memory + utilization + temperature, one row per
        GPU) from nvidia-smi.
      * Currently loaded Ollama models via `ollama ps` (with VRAM
        + keep-alive expiry per model).
      * Topic clusters + ingest sources + recent activity feed +
        LLM usage last N days.
      * Last successful `sciknow refresh` timestamp from the
        54.6.210 `.last_refresh` marker.

    Same data source (`collect_monitor_snapshot`) feeds the
    ``/api/monitor`` endpoint in the web reader, so CLI and GUI
    always agree.

    Examples:

      sciknow db monitor                    # one shot
      sciknow db monitor --watch 5          # live view, re-render every 5s
      sciknow db monitor --json | jq .      # scripted pipelines
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    import json as _json
    import time as _time
    from sciknow.core.monitor import collect_monitor_snapshot

    # Main entry
    if json_out and watch:
        console.print(
            "[red]--json and --watch together don't make sense "
            "(--json emits a single snapshot and exits).[/red]"
        )
        raise typer.Exit(2)

    def _apply_filter(s: dict) -> dict:
        """Phase 54.6.254 — substring-match filter over the rows that
        carry operator-searchable text. Applied in-place to a COPY of
        the snapshot so --watch re-renders work unchanged.

        Scope kept deliberately narrow: recent_activity (per-document
        job log), top_failures (aggregated error classes), and
        recent_activity drafts that carry title/file fields. Adding
        more tables is mechanical but we stop at the ones that
        actually have row-granular free text worth filtering.
        """
        if not filter_str:
            return s
        needle = filter_str.lower()
        def _matches(row) -> bool:
            try:
                return needle in _json.dumps(row, default=str).lower()
            except Exception:
                return True
        s = dict(s)
        pipe = dict(s.get("pipeline") or {})
        pipe["recent_activity"] = [
            r for r in (pipe.get("recent_activity") or [])
            if _matches(r)
        ]
        pipe["top_failures"] = [
            r for r in (pipe.get("top_failures") or [])
            if _matches(r)
        ]
        s["pipeline"] = pipe
        return s

    if json_out:
        snap = collect_monitor_snapshot(throughput_days=days,
                                         llm_usage_days=days)
        snap = _apply_filter(snap)
        console.print(_json.dumps(snap, indent=2, default=str))
        return

    # Phase 54.6.268 — alerts-as-markdown short-circuit. Runs after
    # filter so `--alerts-md --filter 'some_code'` would narrow if
    # we ever extend the filter to alerts (currently narrows
    # recent_activity / top_failures only — a no-op here).
    if alerts_md:
        from sciknow.core.monitor import alerts_as_markdown
        snap = collect_monitor_snapshot(throughput_days=days,
                                         llm_usage_days=days)
        snap = _apply_filter(snap)
        # Plain print (no Rich styling) so stdout is paste-ready.
        print(alerts_as_markdown(snap))
        return

    # Phase 54.6.270 — compact short-circuit. Renders a fixed minimal
    # view using the same data source as the full dashboard but only
    # the must-read panels. Still respects --watch for live updates.
    if compact:
        def _render_compact(s: dict):
            from rich.text import Text
            from rich.panel import Panel
            from rich import box as _box
            _s = _apply_filter(s)
            proj = (_s.get("project") or {}).get("slug") or "?"
            h = _s.get("health_score") or {}
            hs = h.get("score")
            verdict = h.get("verdict") or "?"
            alerts_l = _s.get("alerts") or []
            errs = sum(1 for a in alerts_l if a.get("severity") == "error")
            warns = sum(1 for a in alerts_l if a.get("severity") == "warn")
            svc = _s.get("services") or {}
            jobs = _s.get("active_jobs") or []

            head = Text()
            icon = "✓" if verdict == "healthy" else "⚠" if verdict == "degraded" else "✗"
            head_colour = "bright_green" if verdict == "healthy" else "yellow" if verdict == "degraded" else "bright_red"
            head.append(f"{icon} sciknow monitor · ", style=f"bold {head_colour}")
            head.append(proj, style="cyan")
            if hs is not None:
                hs_colour = "bright_green" if hs >= 90 else "yellow" if hs >= 60 else "bright_red"
                head.append(f"  ·  health ", style="dim")
                head.append(f"{hs}/100", style=hs_colour)
            head.append("  ·  svc ", style="dim")
            for k, short in (("postgres", "P"), ("qdrant", "Q"), ("ollama", "O")):
                info = svc.get(k) or {}
                head.append(short, style="bright_green" if info.get("up") else "bright_red")
            head.append(f"  ·  errors ", style="dim")
            head.append(str(errs), style="bright_red")
            head.append(f"  warn ", style="dim")
            head.append(str(warns), style="yellow")

            lines = [head]
            for a in alerts_l[:5]:
                sev = a.get("severity", "info")
                style = "bright_red" if sev == "error" else "yellow" if sev == "warn" else "dim"
                sev_icon = "✗" if sev == "error" else "⚠" if sev == "warn" else "ℹ"
                ln = Text(f"  {sev_icon} ", style=style)
                ln.append(a.get("message", ""), style=style)
                lines.append(ln)
                act = a.get("action")
                if act:
                    aln = Text(f"    $ ", style="dim")
                    aln.append(act, style="bright_cyan")
                    lines.append(aln)
            for j in jobs[:3]:
                elapsed = j.get("elapsed_s", 0) or 0
                el_str = f"{elapsed / 60:.1f}m" if elapsed >= 60 else f"{elapsed:.0f}s"
                ln = Text(f"  ► ", style="bright_cyan")
                ln.append(
                    f"{(j.get('id') or '?')[:8]} · {(j.get('type') or '?')[:30]} · "
                    f"{j.get('model') or '?'} · {j.get('tokens', 0)}tok @ "
                    f"{j.get('tps') or 0:.1f}t/s · {el_str}",
                    style="white",
                )
                lines.append(ln)
            if not alerts_l and not jobs:
                lines.append(Text("  All systems green · no active jobs", style="green"))

            body = Text()
            for i, line in enumerate(lines):
                if i:
                    body.append("\n")
                body.append_text(line)
            return Panel(
                body, title="[bold]compact[/bold]", title_align="left",
                border_style="bright_green", box=_box.ROUNDED, padding=(0, 1),
            )

        if watch > 0:
            from rich.live import Live
            try:
                snap = collect_monitor_snapshot(throughput_days=days, llm_usage_days=days)
                with Live(
                    _render_compact(snap), console=console,
                    refresh_per_second=max(1, min(4, int(round(1 / max(watch, 0.25))))),
                    screen=False,  # compact is 1-page, no alt-screen
                ) as live:
                    while True:
                        _time.sleep(watch)
                        snap = collect_monitor_snapshot(throughput_days=days, llm_usage_days=days)
                        live.update(_render_compact(snap))
            except KeyboardInterrupt:
                return
        else:
            snap = collect_monitor_snapshot(throughput_days=days, llm_usage_days=days)
            console.print(_render_compact(snap))
        return

    if watch > 0:
        # Phase 54.6.231 — btop-inspired in-place live view using
        # Rich's Live + Layout. Re-renders the same character cells
        # every tick instead of scrolling new frames into history.
        from rich.live import Live
        _r = max(1, min(4, int(round(1 / max(watch, 0.25)))))
        try:
            snap = collect_monitor_snapshot(
                throughput_days=days, llm_usage_days=days,
            )
            snap = _apply_filter(snap)
            with Live(
                _build_monitor_layout(snap, days=days, watch=watch),
                console=console,
                refresh_per_second=_r,
                screen=True,  # alternate-screen buffer → clean restore on exit
            ) as live:
                while True:
                    _time.sleep(watch)
                    snap = collect_monitor_snapshot(
                        throughput_days=days, llm_usage_days=days,
                    )
                    snap = _apply_filter(snap)
                    live.update(_build_monitor_layout(
                        snap, days=days, watch=watch,
                    ))
        except KeyboardInterrupt:
            return

    snap = collect_monitor_snapshot(
        throughput_days=days, llm_usage_days=days,
    )
    snap = _apply_filter(snap)
    console.print(_build_monitor_layout(snap, days=days, watch=0))

    # Phase 54.6.260 — optional log tail after the dashboard. Put it
    # last so scrolling up recovers the panels, and it doesn't burn
    # terminal rows when the operator didn't ask for it.
    if log_tail > 0:
        # Pull the cached tail from the snapshot (already computed
        # above) and extend to log_tail lines if requested > default.
        from sciknow.core.monitor import _log_tail as _lt
        from sciknow.core.project import get_active_project
        try:
            data_dir = get_active_project().data_dir
        except Exception:
            data_dir = None
        tail = _lt(data_dir, n=log_tail) if data_dir else {"lines": []}
        console.print()
        console.print(f"[bold]── last {log_tail} log lines ──[/bold]")
        for ln in tail.get("lines") or []:
            style = (
                "bright_red" if "ERROR" in ln or "CRITICAL" in ln else
                "yellow" if "WARNING" in ln else
                "dim"
            )
            console.print(f"[{style}]{ln}[/{style}]")
        if not tail.get("lines"):
            console.print("[dim](log file not found or empty)[/dim]")


# ── Phase 54.6.231 — btop-style layout builder ────────────────────────


def _build_monitor_layout(snap: dict, *, days: int, watch: int):
    """Build a Rich Layout mirroring btop's compact dashboard look.

    Structure:

        ┌─────────── header (3 rows) ───────────┐
        │  project / last refresh / timestamp   │
        ├──────────┬──────────┬─────────────────┤
        │  corpus  │  gpu     │  qdrant         │
        │          │  models  │  backends       │
        ├──────────┴──────────┴─────────────────┤
        │  stage timing (bars)                  │
        ├───────────────────────────────────────┤
        │  recent activity                      │
        └───────────────────────────────────────┘

    Rich's Layout handles term-width responsiveness — when the window
    is too narrow, columns shrink / drop. Panels use heavy-line boxes
    with accent-coloured borders for the btop aesthetic.
    """
    from datetime import datetime, timezone
    from rich import box as _box
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align

    # ── Colour palette (btop-inspired, works on dark + light terms) ──
    # 54.6.233: dim text → green4 (muted green so status indicators
    # in bright_green still stand out).
    # 54.6.234: panel borders → green.
    # 54.6.235: borders → bright_green (more vivid ANSI terminal
    # green) + box style HEAVY → ROUNDED (single-line with rounded
    # corners — half the vertical noise of HEAVY's double-line
    # look, matches the btop aesthetic).
    C_TITLE = "bold cyan"
    C_DIM = "green4"
    C_OK = "bright_green"
    C_WARN = "yellow"
    C_ERR = "bright_red"
    C_ACCENT = "bright_cyan"
    C_VALUE = "bold white"
    C_BORDER = "bright_green"
    BOX = _box.ROUNDED

    # ── Safe accessors ──────────────────────────────────────────────
    proj = snap.get("project") or {}
    corpus = snap.get("corpus") or {}
    gpus = snap.get("gpu") or []
    loaded = (snap.get("llm") or {}).get("loaded_models") or []
    qcolls = snap.get("qdrant") or []
    backends = snap.get("converter_backends") or []
    pipeline = snap.get("pipeline") or {}
    timing = pipeline.get("stage_timing") or []
    fails = pipeline.get("stage_failures") or []
    activity = pipeline.get("recent_activity") or []
    rates = pipeline.get("rates") or {}
    queue_states = pipeline.get("queue_states") or {}
    top_failures = pipeline.get("top_failures") or []
    hourly = pipeline.get("hourly_throughput") or []
    storage = snap.get("storage") or {}
    disk = storage.get("disk") or {}
    pg_mb = storage.get("pg_database_mb", 0)
    pending_dl = snap.get("pending_downloads", 0)
    host = snap.get("host") or {}
    stuck = snap.get("stuck_job") or {}
    mq = snap.get("meta_quality") or {}
    year_hist = snap.get("year_histogram") or []
    embed_cov = snap.get("embeddings_coverage") or {}
    models = snap.get("model_assignments") or {}
    cost = snap.get("cost_totals") or {}
    vcov = snap.get("visuals_coverage") or {}
    raptor = snap.get("raptor_shape") or {}
    dupe_hashes = snap.get("duplicate_hashes", 0) or 0
    bench_fresh = snap.get("bench_freshness") or {}
    # 54.6.250 — backup freshness: compact "backup Nd" marker
    backup_fresh = snap.get("backup_freshness") or {}
    # 54.6.259 — composite health score
    health = snap.get("health_score") or {}
    # 54.6.262 — services reachability (pg/qdrant/ollama up?)
    services = snap.get("services") or {}
    # 54.6.263 — snapshot self-timing
    snap_ms = snap.get("snapshot_duration_ms")
    # 54.6.237 — trend batch
    growth = snap.get("corpus_growth") or {}
    book_act = snap.get("book_activity") or {}
    bench_delta = snap.get("bench_quality_delta") or {}
    gpu_trend = snap.get("gpu_trend") or {}
    # 54.6.238 — cross-process pulses
    refresh_pulse = snap.get("refresh_pulse") or None
    # 54.6.246 — active web jobs (from cross-process pulse)
    active_jobs = snap.get("active_jobs") or []
    # 54.6.243 additions
    alerts = snap.get("alerts") or []
    inbox = snap.get("inbox") or {}
    qsig = snap.get("quality_signals") or {}
    wiki_mat = snap.get("wiki_materialization") or {}
    projects_overview = snap.get("projects_overview") or []
    # 54.6.244 additions
    funnel = snap.get("ingest_funnel") or []
    hourly_fails = snap.get("pipeline_hourly_failures") or []
    disk_free = snap.get("disk_free") or {}

    # ── Small helpers ───────────────────────────────────────────────

    def _fmt_ms(v):
        if v is None:
            return "—"
        if v >= 60_000:
            return f"{v / 60000:.1f}m"
        if v >= 1000:
            return f"{v / 1000:.1f}s"
        return f"{v:.0f}ms"

    def _fmt_mb(mb: int) -> str:
        if mb >= 1024:
            return f"{mb / 1024:.1f}G"
        return f"{mb}M"

    def _fmt_hours(hours: float | None) -> str:
        if hours is None:
            return "—"
        if hours < 1:
            return f"{int(hours * 60)}m"
        if hours < 24:
            return f"{hours:.1f}h"
        return f"{hours / 24:.1f}d"

    def _sparkline(values: list[int], width: int | None = None) -> Text:
        """Render an int sequence as a unicode-block sparkline. Zero
        values become ▁ so the baseline stays visible; the highest
        bar in the window flips to bright_red as a hot-spot signal."""
        if not values:
            return Text("—", style=C_DIM)
        if width:
            values = values[-width:]
        chars = "▁▂▃▄▅▆▇█"
        max_v = max(values) or 1
        out = Text()
        for v in values:
            if v == 0:
                out.append("▁", style=C_DIM)
                continue
            idx = min(int((v / max_v) * (len(chars) - 1)),
                      len(chars) - 1)
            pct = v / max_v
            colour = (
                C_ERR if pct >= 0.9 else
                C_WARN if pct >= 0.5 else
                C_OK
            )
            out.append(chars[idx], style=colour)
        return out

    def _bar(value: float, total: float, width: int = 14,
             palette=(C_OK, C_WARN, C_ERR)) -> Text:
        """Eighth-block bar with tri-colour palette. Colour flips on
        50% / 85% thresholds so bars get angrier as they fill up —
        the btop visual language."""
        if total <= 0:
            return Text("─" * width, style=C_DIM)
        pct = max(0.0, min(1.0, value / total))
        if pct < 0.5:
            colour = palette[0]
        elif pct < 0.85:
            colour = palette[1]
        else:
            colour = palette[2]
        # 8 sub-steps per character for smooth fill
        cells = pct * width * 8
        full = int(cells // 8)
        rem = int(cells - full * 8)
        subs = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
        bar = "█" * full
        if rem > 0:
            bar += subs[rem]
        bar = bar.ljust(width, "·")
        t = Text(bar[:width], style=colour)
        return t


    # ── Header panel ────────────────────────────────────────────────
    slug = proj.get("slug") or "(no project)"
    pg_db = proj.get("pg_database") or "?"
    last_r = snap.get("last_refresh") or "never"
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    header_text = Text()
    header_text.append("▎ sciknow monitor  ", style="bold bright_cyan")
    header_text.append(f"project ", style=C_DIM)
    header_text.append(slug, style=C_ACCENT)
    header_text.append(f"  ·  db ", style=C_DIM)
    header_text.append(pg_db, style="white")
    header_text.append(f"  ·  last refresh ", style=C_DIM)
    header_text.append(str(last_r)[:26], style="white")
    header_text.append(f"  ·  {now}", style=C_DIM)
    if watch > 0:
        header_text.append(f"  ·  refresh {watch}s  (q to quit)",
                            style=C_DIM)
    # Phase 54.6.263 — how long the snapshot took. Red when
    # >2000ms (the snapshot_slow alert threshold).
    if isinstance(snap_ms, (int, float)):
        sm_colour = (
            C_ERR if snap_ms > 2000 else
            C_WARN if snap_ms > 1000 else C_DIM
        )
        header_text.append(f"  ·  ", style=C_DIM)
        header_text.append(f"built {snap_ms}ms", style=sm_colour)
    # Phase 54.6.234 — stuck-job indicator (only shown when a stall
    # is detected — pending work with no recent job activity). Loud
    # on purpose because it's a "go check what's wrong" signal that
    # wouldn't be useful if buried.
    if stuck.get("is_stuck"):
        age_s = stuck.get("last_age_s", 0) or 0
        age_str = f"{age_s / 60:.1f}m" if age_s >= 60 else f"{age_s:.0f}s"
        header_text.append(
            f"   ⚠ STALLED — last job {age_str} ago, "
            f"{stuck.get('pending_docs', 0)} pending",
            style=C_ERR,
        )
    # Phase 54.6.238 — live refresh progress banner. When the pulse
    # is fresh, the header gets a "refresh step X/Y: <label> · Ne"
    # line so the user sees which step is running AND how long it's
    # taken. Stale pulse → show with a red STALE tag (refresh process
    # died but didn't clear the file).
    if refresh_pulse:
        step_idx = refresh_pulse.get("step_idx", 0)
        step_total = refresh_pulse.get("step_total", 0)
        label = (refresh_pulse.get("step_label") or "")[:32]
        step_elapsed = refresh_pulse.get("step_elapsed_s", 0) or 0
        elapsed = refresh_pulse.get("elapsed_s", 0) or 0
        budget = refresh_pulse.get("budget_s")

        def _fmt_s(s):
            if s >= 3600:
                return f"{s / 3600:.1f}h"
            if s >= 60:
                return f"{s / 60:.1f}m"
            return f"{s:.0f}s"

        header_text.append("   ", style=C_DIM)
        if refresh_pulse.get("is_stale"):
            header_text.append("refresh STALE ", style=C_ERR)
        else:
            header_text.append("refresh ", style=C_ACCENT)
        header_text.append(
            f"{step_idx}/{step_total} {label}  "
            f"step {_fmt_s(step_elapsed)}  total {_fmt_s(elapsed)}",
            style=C_VALUE,
        )
        if budget:
            pct = (elapsed / budget) * 100
            bc = C_WARN if pct > 80 else C_DIM
            header_text.append(
                f"  / budget {_fmt_s(budget)} ({pct:.0f}%)",
                style=bc,
            )

    # Phase 54.6.236 — bench freshness indicator. Shows a compact
    # "bench Nd" marker, green when fresh (< 7d), yellow 7-14d, red
    # > 14d. Silent when no snapshots exist (clean installs).
    bf_age = bench_fresh.get("newest_age_days")
    if bf_age is not None:
        if bf_age < 1:
            bf_colour = C_OK
            bf_str = f"bench {int(bf_age * 24)}h"
        elif bf_age < 7:
            bf_colour = C_OK
            bf_str = f"bench {bf_age:.0f}d"
        elif bf_age < 14:
            bf_colour = C_WARN
            bf_str = f"bench {bf_age:.0f}d STALE"
        else:
            bf_colour = C_ERR
            bf_str = f"bench {bf_age:.0f}d STALE"
        header_text.append(f"  ·  ", style=C_DIM)
        header_text.append(bf_str, style=bf_colour)

    # Phase 54.6.262 — services status chip. Compact three-dot
    # indicator: P (pg) Q (qdrant) O (ollama). Green up, red down.
    # Only rendered when the services dict is populated.
    if services:
        header_text.append(f"  ·  ", style=C_DIM)
        header_text.append("svc ", style=C_DIM)
        for key, short in (("postgres", "P"), ("qdrant", "Q"), ("ollama", "O")):
            info = services.get(key) or {}
            dot_colour = C_OK if info.get("up") else C_ERR
            header_text.append(short, style=dot_colour)

    # Phase 54.6.259 — composite health chip. Shows "health NN/100"
    # with a three-colour palette (green ≥90, yellow 60-89, red <60)
    # as the first telemetry chip on the header so operators read
    # the rollup before the bench/backup freshness chips.
    if health.get("score") is not None:
        hs = int(health["score"])
        if hs >= 90:
            hc = C_OK
        elif hs >= 60:
            hc = C_WARN
        else:
            hc = C_ERR
        header_text.append(f"  ·  ", style=C_DIM)
        header_text.append(f"health {hs}/100", style=hc)

    # Phase 54.6.250 — backup freshness chip. Thresholds match the
    # alert builder: green <7d, yellow 7-30d, red >30d. Silent when
    # no backup has ever run — a clean install state, not a
    # regression, so don't clutter the header.
    bk_age = backup_fresh.get("newest_age_days")
    if bk_age is not None:
        if bk_age < 1:
            bk_colour = C_OK
            bk_str = f"backup {int(bk_age * 24)}h"
        elif bk_age < 7:
            bk_colour = C_OK
            bk_str = f"backup {bk_age:.0f}d"
        elif bk_age < 30:
            bk_colour = C_WARN
            bk_str = f"backup {bk_age:.0f}d STALE"
        else:
            bk_colour = C_ERR
            bk_str = f"backup {bk_age:.0f}d STALE"
        header_text.append(f"  ·  ", style=C_DIM)
        header_text.append(bk_str, style=bk_colour)

    # Phase 54.6.246 — active web jobs banner. One compact line per
    # running job: "id · type · model · Ntok @ TPS · elapsed". Pulse-
    # level staleness means the web server crashed without clearing —
    # then render with a STALE tag in error colour so a zombie doesn't
    # look like an active run. Limited to 3 jobs to keep the header
    # at a reasonable height on small terminals.
    # Phase 54.6.247 — includes TPS (3-second rolling tokens/sec)
    # per job. TPS==0 with a fresh pulse and >5s elapsed is the
    # "model is stuck in thinking / network stall" signal.
    if active_jobs:
        for j in active_jobs[:3]:
            elapsed = j.get("elapsed_s", 0) or 0
            elapsed_str = (
                f"{elapsed / 60:.1f}m" if elapsed >= 60 else f"{elapsed:.0f}s"
            )
            model = j.get("model") or "?"
            # Strip org prefix (`BAAI/` etc.) to keep the line compact
            if "/" in str(model):
                model = str(model).split("/", 1)[1]
            tokens = j.get("tokens", 0) or 0
            tps = j.get("tps") or 0.0
            tw = j.get("target_words")
            tw_str = f"/{tw}w" if tw else ""
            task = (j.get("type") or "?")[:24]
            header_text.append("\n   ", style=C_DIM)
            if j.get("is_stale"):
                header_text.append("web STALE ", style=C_ERR)
            else:
                header_text.append("web ", style=C_ACCENT)
            # Colour TPS: red when elapsed > 5s and tps == 0 (probable
            # stall), green when > 5 tok/s, dim otherwise.
            tps_str = f"@ {tps:.1f}t/s"
            if elapsed > 5 and tps == 0:
                tps_style = f"[{C_ERR}]{tps_str}[/{C_ERR}]"
            elif tps >= 5:
                tps_style = f"[{C_OK}]{tps_str}[/{C_OK}]"
            else:
                tps_style = f"[{C_DIM}]{tps_str}[/{C_DIM}]"
            header_text.append(
                f"{j.get('id','?')[:8]} · {task} · {model[:28]} · "
                f"{tokens}{tw_str} ",
                style=C_VALUE,
            )
            header_text.append_text(Text.from_markup(tps_style))
            header_text.append(f" · {elapsed_str}", style=C_VALUE)
        if len(active_jobs) > 3:
            header_text.append(
                f"\n   [dim]…+{len(active_jobs) - 3} more jobs[/dim]",
                style=C_DIM,
            )

    header_panel = Panel(
        Align.left(header_text, vertical="middle"),
        border_style=C_BORDER, box=BOX, padding=(0, 1),
    )

    # Phase 54.6.243 — consolidated alert banner. Sits between header
    # and the top row of panels when any alert is active; errors red,
    # warnings yellow, info dim-cyan. Silent on a clean install, loud
    # when something needs attention.
    alert_panel = None
    if alerts:
        alert_tbl = Table.grid(padding=(0, 1), expand=True)
        alert_tbl.add_column(ratio=1)
        for a in alerts[:6]:  # cap to 6 so a flurry doesn't blow layout
            sev = a.get("severity", "info")
            icon, style = (
                ("✗", C_ERR) if sev == "error" else
                ("⚠", C_WARN) if sev == "warn" else
                ("ℹ", C_ACCENT)
            )
            line = Text()
            line.append(f"{icon} ", style=style)
            line.append(a.get("message", ""), style=style)
            alert_tbl.add_row(line)
            # Phase 54.6.257 — render suggested-fix command under the
            # message so operators can copy-paste. Dimmed so the
            # severity-coloured message stays the visual anchor.
            act = a.get("action")
            if act:
                aline = Text()
                aline.append("    $ ", style=C_DIM)
                aline.append(act, style=C_ACCENT)
                alert_tbl.add_row(aline)
        border = (
            C_ERR if any(a.get("severity") == "error" for a in alerts)
            else C_WARN if any(a.get("severity") == "warn" for a in alerts)
            else C_BORDER
        )
        alert_panel = Panel(
            alert_tbl, title="[bold]alerts[/bold]", title_align="left",
            border_style=border, box=BOX, padding=(0, 1),
        )

    # ── Corpus panel ────────────────────────────────────────────────
    corpus_tbl = Table.grid(padding=(0, 1), expand=True)
    corpus_tbl.add_column(justify="left", style=C_DIM, ratio=2)
    corpus_tbl.add_column(justify="right", style=C_VALUE, ratio=1)
    done = corpus.get("documents_complete", 0)
    total_docs = corpus.get("documents_total", 0)
    pct_done = (done / total_docs * 100) if total_docs else 0.0
    corpus_tbl.add_row(
        Text("documents", style=C_DIM),
        Text(f"{done:,} / {total_docs:,}", style=C_VALUE),
    )
    if total_docs:
        bar_row = Table.grid(padding=0, expand=True)
        bar_row.add_column(ratio=1)
        bar_row.add_column(justify="right", width=5)
        bar_row.add_row(
            _bar(done, total_docs, width=12),
            f" {pct_done:.0f}%",
        )
        corpus_tbl.add_row("", bar_row)
    for label, key in (
        ("chunks", "chunks"),
        ("citations", "citations"),
        ("visuals", "visuals"),
        ("kg triples", "kg_triples"),
        ("wiki pages", "wiki_pages"),
        ("institutions", "institutions"),
    ):
        v = corpus.get(key, 0)
        corpus_tbl.add_row(
            Text(label, style=C_DIM),
            Text(f"{v:,}",
                 style=C_VALUE if v else C_DIM),
        )

    # ── rates + ETA + queue summary ─────────────────────────────
    # Low-key divider + operational info block. Shows even when
    # idle so the user can see "rate is 0, no work pending" at
    # a glance.
    corpus_tbl.add_row(
        Text("─ ingest ─", style=C_DIM),
        Text(""),
    )
    rate_1h = rates.get("rate_1h") or 0
    rate_4h = rates.get("rate_4h") or 0
    rate_colour = (
        C_OK if rate_1h > 0 or rate_4h > 0 else C_DIM
    )
    corpus_tbl.add_row(
        Text("rate (1h / 4h)", style=C_DIM),
        Text(f"{rate_1h:.0f} / {rate_4h:.0f} /hr", style=rate_colour),
    )
    eta_colour = C_WARN if rates.get("eta_hours") is not None else C_DIM
    corpus_tbl.add_row(
        Text("ETA", style=C_DIM),
        Text(_fmt_hours(rates.get("eta_hours")), style=eta_colour),
    )
    # Queue summary — collapsed into one line
    qs = queue_states or {}
    if qs:
        q_str = " ".join(f"{k}:{v}" for k, v in qs.items())
        corpus_tbl.add_row(
            Text("queue", style=C_DIM),
            Text(q_str, style=C_WARN),
        )
    elif rates.get("pending_docs", 0) == 0:
        corpus_tbl.add_row(
            Text("queue", style=C_DIM),
            Text("idle", style=C_DIM),
        )
    if pending_dl:
        corpus_tbl.add_row(
            Text("pending dl", style=C_DIM),
            Text(f"{pending_dl:,}", style=C_WARN),
        )
    # Phase 54.6.243 — inbox/drop-zone waiting count
    if inbox.get("count", 0) > 0:
        age_s = inbox.get("oldest_age_s") or 0
        age_str = (
            f"{age_s / 86400:.0f}d" if age_s >= 86400 else
            f"{age_s / 3600:.0f}h" if age_s >= 3600 else
            f"{age_s / 60:.0f}m"
        )
        # Fresh drops = green (recent work to pick up); ancient = dim
        # (forgotten stash, not actionable).
        colour = C_OK if age_s < 86400 else C_DIM
        corpus_tbl.add_row(
            Text("inbox", style=C_DIM),
            Text(f"{inbox['count']} pdf · {age_str} old", style=colour),
        )

    # Phase 54.6.234 — content quality strip. Condensed because the
    # corpus panel is already dense; each row is a one-line summary
    # of a dimension that's otherwise hidden in the DB. Shows
    # immediately which stages need re-running (e.g. crossref-low =
    # re-run enrich; paper_types empty = run classify-papers).
    sources = mq.get("sources") or []
    paper_types = mq.get("paper_types") or []
    languages = mq.get("languages") or []
    if sources or paper_types or languages or mq.get("citations_total"):
        corpus_tbl.add_row(
            Text("─ quality ─", style=C_DIM),
            Text(""),
        )
    if sources:
        # Top source name + count + "/total"
        top = sources[0]
        total_src = sum(s["n"] for s in sources)
        src_str = f"{top['source']} {top['n']}/{total_src}"
        colour = (
            C_OK if top["source"] in ("crossref", "arxiv") else C_DIM
        )
        corpus_tbl.add_row(
            Text("metadata src", style=C_DIM),
            Text(src_str, style=colour),
        )
    if paper_types:
        top = paper_types[0]
        total_pt = sum(p["n"] for p in paper_types)
        corpus_tbl.add_row(
            Text("paper types", style=C_DIM),
            Text(f"{top['type']} {top['n']}/{total_pt}", style=C_DIM),
        )
    elif corpus.get("documents_complete", 0) > 0:
        # No types yet → flag as a to-do
        corpus_tbl.add_row(
            Text("paper types", style=C_DIM),
            Text("not classified", style=C_WARN),
        )
    if languages and len(languages) > 1:
        # Multi-lingual corpus — list top-3 briefly
        top_langs = ", ".join(
            f"{l['lang']}:{l['n']}" for l in languages[:3]
        )
        corpus_tbl.add_row(
            Text("languages", style=C_DIM),
            Text(top_langs, style=C_DIM),
        )
    if mq.get("citations_total"):
        xlinked_pct = mq.get("citations_crosslinked_pct", 0)
        colour = (
            C_OK if xlinked_pct >= 20 else
            C_WARN if xlinked_pct >= 5 else C_DIM
        )
        corpus_tbl.add_row(
            Text("cites xlinked", style=C_DIM),
            Text(f"{mq['citations_crosslinked']:,}/"
                 f"{mq['citations_total']:,} "
                 f"({xlinked_pct:.1f}%)", style=colour),
        )
    if mq.get("retracted"):
        corpus_tbl.add_row(
            Text("retracted", style=C_DIM),
            Text(str(mq["retracted"]), style=C_ERR),
        )
    # Phase 54.6.243 — abstract coverage (ColBERT retrieval input).
    if qsig.get("abstract_eligible"):
        pct = qsig["abstract_pct"]
        covered = qsig["abstract_covered"]
        eligible = qsig["abstract_eligible"]
        colour = (
            C_OK if pct >= 80 else
            C_WARN if pct >= 50 else C_ERR
        )
        corpus_tbl.add_row(
            Text("abstracts", style=C_DIM),
            Text(f"{covered:,}/{eligible:,} ({pct:.0f}%)", style=colour),
        )
    # Phase 54.6.243 — chunk-length distribution (splitter sanity).
    if qsig.get("chunk_p50_chars"):
        p50 = qsig["chunk_p50_chars"]
        p95 = qsig["chunk_p95_chars"] or 0
        # Healthy band is ~500-2500 at p50; outside that the splitter
        # is producing either fragment-sized or context-window-busting
        # chunks.
        colour = (
            C_OK if 500 <= p50 <= 2500 else
            C_WARN if 300 <= p50 <= 4000 else C_ERR
        )
        corpus_tbl.add_row(
            Text("chunk chars", style=C_DIM),
            Text(f"p50 {p50:,} · p95 {p95:,}", style=colour),
        )
    # Phase 54.6.243 — KG density (triples per completed doc).
    if (qsig.get("kg_triples_per_doc") or 0) > 0:
        t_per_doc = qsig["kg_triples_per_doc"]
        colour = (
            C_OK if t_per_doc >= 20 else
            C_WARN if t_per_doc >= 5 else C_DIM
        )
        corpus_tbl.add_row(
            Text("kg density", style=C_DIM),
            Text(f"{t_per_doc:.1f} triples/doc", style=colour),
        )

    # Phase 54.6.235 — embeddings coverage (silent drift detector)
    if embed_cov.get("total"):
        total = embed_cov["total"]
        embedded = embed_cov["embedded"]
        missing = embed_cov["missing"]
        pct = embed_cov["pct"]
        colour = (
            C_OK if pct >= 99 else
            C_WARN if pct >= 95 else C_ERR
        )
        lbl = f"{embedded:,}/{total:,} ({pct:.1f}%)"
        if missing:
            lbl += f"  ✗{missing:,} miss"
        corpus_tbl.add_row(
            Text("embeddings", style=C_DIM),
            Text(lbl, style=colour),
        )

    # Phase 54.6.236 — file-hash duplicate detector. Should always
    # be 0 (UNIQUE constraint on documents.file_hash), but loud if
    # it ever isn't because that implies a schema-drift bug.
    if corpus.get("documents_complete", 0) > 0:
        dupe_colour = C_OK if dupe_hashes == 0 else C_ERR
        corpus_tbl.add_row(
            Text("dupe hashes", style=C_DIM),
            Text(str(dupe_hashes), style=dupe_colour),
        )

    # Phase 54.6.237 — corpus growth rate: compact "+N /24h · +M /7d"
    # line. Shows only when 7d > 0 (keeps rows quiet on a dormant
    # install).
    if growth.get("last_7d", 0) > 0:
        corpus_tbl.add_row(
            Text("growth", style=C_DIM),
            Text(
                f"+{growth['last_24h']:,} /24h  "
                f"+{growth['last_7d']:,} /7d",
                style=C_OK,
            ),
        )

    corpus_panel = Panel(
        corpus_tbl, title="[bold]corpus · ingest · quality[/bold]",
        title_align="left",
        border_style=C_BORDER, box=BOX, padding=(0, 1),
    )

    # ── GPU + Ollama panel (combined middle column) ────────────────
    gpu_tbl = Table.grid(padding=(0, 1), expand=True)
    gpu_tbl.add_column(ratio=1)
    if gpus:
        for g in gpus:
            mu = g["memory_used_mb"]
            mt = max(g["memory_total_mb"], 1)
            gpu_header = Text()
            gpu_header.append("gpu", style=C_DIM)
            gpu_header.append(f" #{g['index']} ", style=C_VALUE)
            gpu_header.append(g["name"][:24], style=C_ACCENT)
            gpu_header.append(
                f"  {g.get('temperature_c', '?')}°C",
                style=(C_ERR if g.get("temperature_c", 0) >= 85
                       else C_WARN if g.get("temperature_c", 0) >= 75
                       else C_DIM),
            )
            gpu_tbl.add_row(gpu_header)

            # VRAM bar — show GB (compact) to save column width
            vram_row = Table.grid(padding=0, expand=True)
            vram_row.add_column(width=5, style=C_DIM)
            vram_row.add_column(ratio=1)
            vram_row.add_column(justify="right", width=11, style=C_VALUE)
            vram_row.add_row(
                "vram", _bar(mu, mt, width=14),
                f"{mu / 1024:.1f}/{mt / 1024:.0f}G",
            )
            gpu_tbl.add_row(vram_row)

            # Util bar
            util = g["utilization_pct"]
            util_row = Table.grid(padding=0, expand=True)
            util_row.add_column(width=5, style=C_DIM)
            util_row.add_column(ratio=1)
            util_row.add_column(justify="right", width=5, style=C_VALUE)
            util_row.add_row("util", _bar(util, 100, width=14),
                              f"{util}%")
            gpu_tbl.add_row(util_row)
    else:
        gpu_tbl.add_row(Text("(no GPU detected — nvidia-smi absent)",
                              style=C_DIM))

    # Phase 54.6.234 — host RAM + load average. Sits under GPU so
    # the panel covers the whole compute surface, not just the card.
    if host.get("mem_total_mb"):
        gpu_tbl.add_row("")
        gpu_tbl.add_row(Text("host", style=C_DIM))
        mem_row = Table.grid(padding=0, expand=True)
        mem_row.add_column(width=5, style=C_DIM)
        mem_row.add_column(ratio=1)
        mem_row.add_column(justify="right", width=11, style=C_VALUE)
        mem_row.add_row(
            "ram",
            _bar(host["mem_used_mb"], host["mem_total_mb"], width=14),
            f"{host['mem_used_mb'] / 1024:.1f}/"
            f"{host['mem_total_mb'] / 1024:.0f}G",
        )
        gpu_tbl.add_row(mem_row)
        # Load avg normalised against cpu_count. 1.0× load per core
        # = fully loaded; >1× = queue forming. Colour coding kicks
        # at the same 50%/85% thresholds as the rest of the UI.
        if host.get("cpu_count"):
            cores = host["cpu_count"]
            load_row = Table.grid(padding=0, expand=True)
            load_row.add_column(width=5, style=C_DIM)
            load_row.add_column(ratio=1)
            load_row.add_column(justify="right", width=11, style=C_VALUE)
            load_row.add_row(
                "load",
                _bar(host["load_1m"], cores, width=14),
                f"{host['load_1m']:.1f} ({cores}c)",
            )
            gpu_tbl.add_row(load_row)

    # Ollama models mini-list
    gpu_tbl.add_row("")
    if loaded:
        for m in loaded:
            mini = Text()
            mini.append("◉ ", style=C_OK)
            mini.append((m.get("name") or "?")[:28], style=C_ACCENT)
            mini.append(f"  {m.get('vram_mb', 0):,}MB", style=C_DIM)
            exp = m.get("expires_at") or ""
            if exp:
                # Only show the time portion of the expiry if ISO
                mini.append(f"  exp {str(exp)[-8:-3] if len(str(exp)) > 8 else str(exp)}", style=C_DIM)
            gpu_tbl.add_row(mini)
    else:
        gpu_tbl.add_row(Text("◎ ollama: no models loaded", style=C_DIM))

    # Storage block — compact one-liner rows. Shows the top-level
    # paths that can surprise you (mineru_output balloons during
    # re-ingest) plus pg DB size.
    gpu_tbl.add_row("")
    gpu_tbl.add_row(Text("storage", style=C_DIM))
    for label, key in (
        ("data", "data_dir_mb"),
        ("mineru", "mineru_output_mb"),
        ("processed", "processed_mb"),
    ):
        mb = disk.get(key, 0)
        if mb == 0 and label != "data":
            continue
        row = Table.grid(padding=0, expand=True)
        row.add_column(width=10, style=C_DIM)
        row.add_column(justify="right", style=C_VALUE)
        row.add_row(label, _fmt_mb(mb))
        gpu_tbl.add_row(row)
    pg_row = Table.grid(padding=0, expand=True)
    pg_row.add_column(width=10, style=C_DIM)
    pg_row.add_column(justify="right", style=C_VALUE)
    pg_row.add_row("pg db", _fmt_mb(pg_mb))
    gpu_tbl.add_row(pg_row)

    # Phase 54.6.244 — disk free on the data_dir filesystem. Gives
    # the capacity context the per-path sizes above don't convey.
    if disk_free.get("total_mb"):
        total_gb = disk_free["total_mb"] / 1024
        free_gb = disk_free["free_mb"] / 1024
        pct = disk_free["pct_used"]
        free_row = Table.grid(padding=0, expand=True)
        free_row.add_column(width=5, style=C_DIM)
        free_row.add_column(ratio=1)
        free_row.add_column(justify="right", width=11, style=C_VALUE)
        free_row.add_row(
            "free",
            _bar(disk_free["used_mb"], disk_free["total_mb"], width=14),
            f"{free_gb:.0f}/{total_gb:.0f}G",
        )
        gpu_tbl.add_row(free_row)

    # Phase 54.6.236 — model assignments per role. Surfaces the
    # .env-configured mapping ("which LLM is autowrite using right
    # now?"). Compact two-col: role → model (trimmed to 28 chars).
    # Phase 54.6.244 — added book writer / reviewer / autowrite
    # scorer / caption VLM rows so the whole book pipeline is
    # auditable from the dashboard. Inherited slots (override = None)
    # render with a subtle "↑llm" suffix so the user can tell
    # "explicitly set to llm_main" apart from "inherits llm_main".
    if models:
        gpu_tbl.add_row("")
        gpu_tbl.add_row(Text("models", style=C_DIM))
        llm_main = models.get("llm_main")
        rows_to_show: list[tuple[str, str | None, bool]] = [
            ("llm",      llm_main, False),
            ("fast",     models.get("llm_fast")
                         if models.get("llm_fast") != llm_main else None,
                         False),
            ("book-wr",  models.get("book_write")  or llm_main,
                         models.get("book_write") is None),
            ("book-rv",  models.get("book_review") or llm_main,
                         models.get("book_review") is None),
            ("aw-score", models.get("autowrite_scorer") or llm_main,
                         models.get("autowrite_scorer") is None),
            ("caption",  models.get("caption_vlm"), False),
            ("vlm-pro",  models.get("mineru_vlm_model"), False),
            ("embedder", models.get("embedder"), False),
            ("reranker", models.get("reranker"), False),
        ]
        for role, name, inherited in rows_to_show:
            if not name:
                continue
            display = str(name)
            # Drop `BAAI/` / `opendatalab/` org prefixes for width
            if "/" in display:
                display = display.split("/", 1)[1]
            display = display[:32]
            if inherited:
                # "↑llm" marker — muted so the model name still reads first
                display = f"{display} [dim](↑llm)[/dim]"
            row = Table.grid(padding=0, expand=True)
            row.add_column(width=9, style=C_DIM)
            row.add_column(ratio=1, style=C_ACCENT, overflow="fold")
            row.add_row(role, display)
            gpu_tbl.add_row(row)

    gpu_panel = Panel(
        gpu_tbl, title="[bold]gpu · models · storage[/bold]",
        title_align="left",
        border_style=C_BORDER, box=BOX, padding=(0, 1),
    )

    # ── Qdrant + backends panel (right column) ──────────────────────
    right_tbl = Table.grid(padding=(0, 1), expand=True)
    right_tbl.add_column(ratio=1)

    # Qdrant
    if qcolls:
        qhead = Text()
        qhead.append("qdrant", style=C_DIM)
        right_tbl.add_row(qhead)
        for c in qcolls:
            row = Table.grid(padding=(0, 1), expand=True)
            row.add_column(ratio=3)
            row.add_column(justify="right", ratio=1, style=C_VALUE)
            name = c["name"]
            # Trim project prefix for readability
            display = name
            if proj.get("slug"):
                pref = proj["slug"].replace("-", "_") + "_"
                if display.startswith(pref):
                    display = display[len(pref):]
            fields = []
            for v in c.get("vectors") or []:
                mark = "◆" if v == "colbert" else "●"
                fields.append(f"[cyan]{mark}{v}[/cyan]")
            for s in c.get("sparse_vectors") or []:
                fields.append(f"[magenta]◇{s}[/magenta]")
            label_rendered = Text.from_markup(
                f"[{C_ACCENT}]{display}[/{C_ACCENT}] "
                + " ".join(fields)
            )
            # Phase 54.6.235 — show points count + disk estimate.
            points_str = f"{c['points_count']:,}"
            disk_mb = c.get("estimated_disk_mb", 0) or 0
            if disk_mb:
                disk_str = (
                    f" [{C_DIM}]~{disk_mb / 1024:.1f}G[/{C_DIM}]"
                    if disk_mb >= 1024 else
                    f" [{C_DIM}]~{disk_mb}M[/{C_DIM}]"
                )
                points_col = Text.from_markup(points_str + disk_str)
            else:
                points_col = points_str
            row.add_row(label_rendered, points_col)
            right_tbl.add_row(row)

    # Backends
    right_tbl.add_row("")
    if backends:
        right_tbl.add_row(Text("converter backends", style=C_DIM))
        for b in backends:
            nm = b["backend"] or "(none)"
            colour = C_OK if nm.startswith("mineru-vlm-pro") else (
                C_WARN if nm == "mineru-pipeline" else C_DIM
            )
            row = Table.grid(padding=(0, 1), expand=True)
            row.add_column(ratio=3)
            row.add_column(justify="right", ratio=1, style=C_VALUE)
            row.add_row(Text(nm, style=colour), f"{b['n']:,}")
            right_tbl.add_row(row)

    # Phase 54.6.235 — top-5 topic clusters with mini bars. Uses the
    # panel's empty real estate to surface content composition at a
    # glance, bar width scaled to the largest cluster in the set.
    clusters = snap.get("topic_clusters") or []
    if clusters:
        right_tbl.add_row("")
        right_tbl.add_row(Text("top topics", style=C_DIM))
        max_n = max(c["n"] for c in clusters[:5]) or 1
        for c in clusters[:5]:
            row = Table.grid(padding=(0, 1), expand=True)
            row.add_column(ratio=2)
            row.add_column(ratio=3)
            row.add_column(justify="right", width=5, style=C_VALUE)
            name = (c["name"] or "(unnamed)")[:22]
            row.add_row(
                Text(name, style=C_ACCENT),
                _bar(c["n"], max_n, width=12,
                     palette=(C_OK, C_OK, C_OK)),  # monochrome — same colour, heat n/a here
                f"{c['n']}",
            )
            right_tbl.add_row(row)

    # Phase 54.6.243 — wiki materialization: compiled-wiki coverage
    # of the distinct topic_cluster universe. Silent pre-clustering.
    if wiki_mat.get("topics_total"):
        pct = wiki_mat["pct"]
        pages = wiki_mat["wiki_pages"]
        total = wiki_mat["topics_total"]
        colour = (
            C_OK if pct >= 80 else
            C_WARN if pct >= 20 else C_DIM
        )
        right_tbl.add_row("")
        right_tbl.add_row(
            Text("wiki materialization", style=C_DIM)
        )
        wiki_row = Table.grid(padding=(0, 1), expand=True)
        wiki_row.add_column(ratio=1)
        wiki_row.add_column(justify="right", width=12, style=C_VALUE)
        wiki_row.add_row(
            _bar(pages, total, width=14,
                 palette=(C_DIM, C_WARN, C_OK)),
            f"{pages}/{total} ({pct:.0f}%)",
        )
        right_tbl.add_row(wiki_row)

    # Phase 54.6.243 — cross-project overview. Silent when only one
    # project exists; shows compact "slug · docs · active" list
    # otherwise. Doc count of -1 signals DB unreachable / schema drift.
    if len(projects_overview) > 1:
        right_tbl.add_row("")
        right_tbl.add_row(Text("projects", style=C_DIM))
        for pov in projects_overview[:6]:
            row = Table.grid(padding=(0, 1), expand=True)
            row.add_column(ratio=3)
            row.add_column(justify="right", width=10, style=C_VALUE)
            slug = pov["slug"]
            if pov["is_active"]:
                slug_rendered = Text(f"● {slug}", style=C_OK)
            else:
                slug_rendered = Text(f"○ {slug}", style=C_DIM)
            docs = pov["docs"]
            if docs < 0:
                docs_rendered = Text("?", style=C_WARN)
            else:
                docs_rendered = Text(f"{docs:,}",
                                      style=C_VALUE if docs else C_DIM)
            row.add_row(slug_rendered, docs_rendered)
            right_tbl.add_row(row)

    # Year distribution sparkline — corpus age profile in one line.
    if year_hist:
        counts = [y["n"] for y in year_hist]
        if any(counts):
            first = year_hist[0]["year"]
            last = year_hist[-1]["year"]
            right_tbl.add_row("")
            right_tbl.add_row(
                Text(f"papers by year {first}→{last}", style=C_DIM)
            )
            right_tbl.add_row(_sparkline(counts))

    right_panel = Panel(
        right_tbl,
        title="[bold]qdrant · backends · topics · years[/bold]",
        title_align="left",
        border_style=C_BORDER, box=BOX, padding=(0, 1),
    )

    # ── Stage timing panel (with bars) ──────────────────────────────
    timing_tbl = Table.grid(padding=(0, 1), expand=True)
    timing_tbl.add_column(style=C_DIM, width=14)
    timing_tbl.add_column(justify="right", style=C_DIM, width=7)
    timing_tbl.add_column(ratio=1)
    timing_tbl.add_column(justify="right", style=C_VALUE, width=9)
    timing_tbl.add_column(justify="right", style=C_DIM, width=9)
    if timing:
        # Normalise bars against the slowest p95 in the set
        max_p95 = max(
            (row.get("p95_ms") or 0) for row in timing
        ) or 1.0
        for row in timing:
            n = row["n"]
            p50 = row.get("p50_ms")
            p95 = row.get("p95_ms")
            timing_tbl.add_row(
                row["stage"], f"n={n:,}",
                _bar(p95 or 0, max_p95, width=24,
                     palette=(C_OK, C_WARN, C_ERR)),
                _fmt_ms(p95), f"p50 {_fmt_ms(p50)}",
            )
    else:
        timing_tbl.add_row(Text("(no completed jobs yet)", style=C_DIM),
                           "", "", "", "")
    any_fail = any(f["failed"] > 0 for f in fails)
    if any_fail:
        timing_tbl.add_row("", "", "", "", "")
        for f in fails:
            if not f["failed"]:
                continue
            rate = (f["failed"] / max(f["total"], 1)) * 100
            colour = C_ERR if rate > 5 else C_WARN
            timing_tbl.add_row(
                Text(f["stage"], style=colour),
                f"fail {f['failed']}",
                _bar(f["failed"], f["total"], width=24,
                     palette=(C_WARN, C_ERR, C_ERR)),
                f"{rate:.1f}%", f"/ {f['total']}",
            )

    # 54.6.239 — build a SECOND table for content that should span
    # the full panel width (sparklines + footer lines). The 5-col
    # `timing_tbl` forces any row content into the bar column,
    # which shoves it ~22 chars right of the panel left edge —
    # visually out of place. A separate full-width grid renders
    # below the stages table inside the same Panel via Group.
    footer_tbl = Table.grid(padding=(0, 1), expand=True)
    footer_tbl.add_column(ratio=1)

    # 24h throughput sparkline — one line, left-label, right-max.
    # Makes the daily cadence (hot spots vs idle windows) visible
    # at a glance without a separate panel.
    if hourly:
        spark_row = Table.grid(padding=(0, 1), expand=True)
        spark_row.add_column(width=13, style=C_DIM)
        spark_row.add_column(ratio=1)
        spark_row.add_column(justify="right", width=18, style=C_DIM)
        peak = max(hourly) if hourly else 0
        spark_row.add_row(
            "24h docs/hr",
            _sparkline(hourly),
            f"peak {peak}  now {hourly[-1]}",
        )
        footer_tbl.add_row(spark_row)

    # Phase 54.6.244 — ingest funnel visualization. Canonical
    # pipeline stages as horizontal bars scaled against the largest
    # stage population. Unlike `queue_states` (only non-terminal),
    # the funnel shows where docs *accumulate* including "complete"
    # (the sink) — visually obvious when one stage acts as a
    # bottleneck or when docs are stuck pre-terminal.
    if funnel and any(f["n"] for f in funnel):
        max_n = max(f["n"] for f in funnel) or 1
        footer_tbl.add_row("")
        footer_tbl.add_row(Text("ingest funnel", style=C_DIM))
        for row in funnel:
            n = row["n"]
            if n == 0:
                continue  # skip stages with nothing in them
            stage = row["stage"]
            # Terminal stages are green (complete) / red (failed),
            # pre-terminal are yellow (in flight).
            palette = (
                (C_OK, C_OK, C_OK) if stage == "complete" else
                (C_ERR, C_ERR, C_ERR) if stage == "failed" else
                (C_WARN, C_WARN, C_WARN)
            )
            funnel_row = Table.grid(padding=(0, 1), expand=True)
            funnel_row.add_column(width=20, style=C_DIM)
            funnel_row.add_column(ratio=1)
            funnel_row.add_column(justify="right", width=6, style=C_VALUE)
            funnel_row.add_row(
                stage,
                _bar(n, max_n, width=18, palette=palette),
                f"{n:,}",
            )
            footer_tbl.add_row(funnel_row)

    # Phase 54.6.244 — hourly failure sparkline aligned to the 24h
    # throughput sparkline above. Same bucketing → visual pair.
    if hourly_fails and any(hourly_fails):
        spark_row = Table.grid(padding=(0, 1), expand=True)
        spark_row.add_column(width=13, style=C_DIM)
        spark_row.add_column(ratio=1)
        spark_row.add_column(justify="right", width=18, style=C_DIM)
        peak = max(hourly_fails) if hourly_fails else 0
        # Use a red-leaning palette for failures — visually distinct
        # from the green throughput sparkline even when both are
        # drawn with the same unicode blocks.
        chars = "▁▂▃▄▅▆▇█"
        max_v = max(hourly_fails) or 1
        t = Text()
        for v in hourly_fails:
            if v == 0:
                t.append("▁", style=C_DIM)
                continue
            idx = min(int((v / max_v) * (len(chars) - 1)),
                      len(chars) - 1)
            t.append(chars[idx], style=C_ERR)
        spark_row.add_row(
            "24h fails/hr",
            t,
            f"peak {peak}  now {hourly_fails[-1]}",
        )
        footer_tbl.add_row(spark_row)

    # Top failure classes — shown only when any exist in the last 24h.
    # Small by design (LIMIT 3 in the query) — summary, not clinic.
    if top_failures:
        footer_tbl.add_row("")
        for tf in top_failures:
            err = (tf.get("error") or "")[:80]
            footer_tbl.add_row(
                Text(f"✗ {tf['stage']}  ", style=C_ERR)
                + Text(err, style=C_DIM)
                + Text(f"   ×{tf['count']}", style=C_ERR),
            )

    # Phase 54.6.236 — system summary footer bundling
    # cost + visuals coverage + RAPTOR shape. Uses empty space in
    # the pipeline panel rather than adding a new layout row.
    footer_lines: list[Text] = []

    # Phase 54.6.252 — config drift footer line. Visible only when
    # non-empty (i.e., active project is overriding one or more .env
    # keys). Yellow "drift" tag so it jumps out but doesn't look
    # like an error. Truncated to 70 chars per entry to fit a 120-
    # wide terminal comfortably.
    drift = snap.get("config_drift") or []
    if drift:
        line = Text()
        line.append("drift    ", style=C_DIM)
        line.append(f"{len(drift)} override(s): ", style=C_WARN)
        line.append("; ".join(d[:70] for d in drift[:2]), style=C_VALUE)
        if len(drift) > 2:
            line.append(f"  (+{len(drift) - 2} more)", style=C_DIM)
        footer_lines.append(line)

    if cost.get("calls"):
        days = cost.get("window_days", 30)
        line = Text()
        line.append(f"llm {days}d  ", style=C_DIM)
        line.append(f"{cost['tokens']:,}t", style=C_VALUE)
        line.append(f" / {cost['seconds']:.0f}s", style=C_DIM)
        line.append(f" / {cost['calls']:,} calls", style=C_DIM)
        line.append(f" across {cost['models']} model(s)", style=C_DIM)
        footer_lines.append(line)

    if vcov.get("figures_total") or vcov.get("charts_total") \
            or vcov.get("equations_total") or vcov.get("tables_total"):
        def _cov_frag(label: str, done: int, total: int) -> Text:
            if total == 0:
                return Text("")
            pct = (done / total * 100) if total else 0
            colour = (
                C_OK if pct >= 95 else
                C_WARN if pct >= 50 else C_DIM
            )
            t = Text()
            t.append(f"{label} ", style=C_DIM)
            t.append(f"{done}/{total}", style=colour)
            t.append(f" ({pct:.0f}%)  ", style=C_DIM)
            return t

        cov_line = Text()
        cov_line.append("visuals  ", style=C_DIM)
        cov_line += _cov_frag(
            "figs capt", vcov["figures_captioned"], vcov["figures_total"]
        )
        cov_line += _cov_frag(
            "charts capt", vcov["charts_captioned"],
            vcov["charts_total"],
        )
        cov_line += _cov_frag(
            "eqs parap", vcov["equations_paraphrased"],
            vcov["equations_total"],
        )
        cov_line += _cov_frag(
            "tbls parsed", vcov["tables_parsed"], vcov["tables_total"]
        )
        if vcov.get("mentions_total_eligible"):
            cov_line += _cov_frag(
                "mentions", vcov["mentions_linked"],
                vcov["mentions_total_eligible"],
            )
        footer_lines.append(cov_line)

    if raptor.get("has_tree"):
        rline = Text()
        rline.append("raptor   ", style=C_DIM)
        rline.append(
            f"{raptor['total_nodes']} nodes", style=C_VALUE,
        )
        levels_str = " · ".join(
            f"L{l['level']}:{l['n']}" for l in raptor["levels"]
        )
        rline.append(f"  {levels_str}", style=C_DIM)
        footer_lines.append(rline)

    # Phase 54.6.237 — active book activity summary (when any book
    # exists in the DB). Compact: title, chapter completion, word count.
    if book_act.get("title"):
        chapters_total = book_act.get("chapters_total", 0) or 0
        chapters_done = book_act.get("chapters_drafted", 0) or 0
        words = book_act.get("total_words", 0) or 0
        bline = Text()
        bline.append("book     ", style=C_DIM)
        bline.append(
            (book_act["title"] or "")[:40], style=C_ACCENT,
        )
        bline.append(
            f"  {chapters_done}/{chapters_total} ch drafted",
            style=C_DIM,
        )
        bline.append(f"  {words:,}w", style=C_VALUE)
        footer_lines.append(bline)

    # Phase 54.6.237 — bench quality delta from the two newest
    # snapshots. Highlights top regression + top improvement if any.
    if bench_delta.get("have_pair") and bench_delta.get("total_metrics"):
        def _fmt_pct(p):
            return f"{p:+.1f}%" if p >= 0 else f"{p:.1f}%"

        bline = Text()
        bline.append("bench Δ  ", style=C_DIM)
        bline.append(
            f"{bench_delta.get('old_sha') or '?'[:7]} → "
            f"{bench_delta.get('new_sha') or '?'[:7]}",
            style=C_DIM,
        )
        best = (bench_delta.get("best") or [])
        worst = (bench_delta.get("worst") or [])
        if best:
            b = best[0]
            # Truncate metric name for readability
            name = b["metric"].split(":", 1)[-1][:22]
            bline.append(f"  ↑{name} ", style=C_DIM)
            bline.append(_fmt_pct(b["delta_pct"]), style=C_OK)
        if worst and worst[0]["delta_pct"] < -0.1:
            w = worst[0]
            name = w["metric"].split(":", 1)[-1][:22]
            bline.append(f"  ↓{name} ", style=C_DIM)
            bline.append(_fmt_pct(w["delta_pct"]), style=C_ERR)
        footer_lines.append(bline)

    # Phase 54.6.237 — GPU temp + util trend sparklines. Only render
    # once the ring buffer has at least 3 samples, so fresh starts
    # don't show a single-dot line.
    temp_samples = gpu_trend.get("temp_samples") or []
    util_samples = gpu_trend.get("util_samples") or []
    if len(temp_samples) >= 3:
        # Temp on its own scale (55-85°C usable), util on 0-100
        t_line = Table.grid(padding=(0, 1), expand=True)
        t_line.add_column(width=9, style=C_DIM)
        t_line.add_column(ratio=1)
        t_line.add_column(justify="right", width=6, style=C_DIM)
        t_line.add_row(
            "gpu temp", _sparkline(temp_samples),
            f"{temp_samples[-1]}°C",
        )
        footer_lines.append(t_line)
        if len(util_samples) >= 3:
            u_line = Table.grid(padding=(0, 1), expand=True)
            u_line.add_column(width=9, style=C_DIM)
            u_line.add_column(ratio=1)
            u_line.add_column(justify="right", width=6, style=C_DIM)
            u_line.add_row(
                "gpu util", _sparkline(util_samples),
                f"{util_samples[-1]}%",
            )
            footer_lines.append(u_line)

    if footer_lines:
        footer_tbl.add_row("")
        for line in footer_lines:
            footer_tbl.add_row(line)

    # Compose timing + footer into a single renderable via Group.
    # Group stacks renderables vertically — the stages table on top
    # (its own 5-col width math), the full-width footer below.
    from rich.console import Group as _Group
    timing_body = _Group(timing_tbl, footer_tbl)

    timing_panel = Panel(
        timing_body,
        title="[bold]pipeline stages[/bold] "
              "[dim](bars = p95, heat = % of slowest)[/dim]",
        title_align="left",
        border_style=C_BORDER, box=BOX, padding=(0, 1),
    )

    # ── Recent activity feed ────────────────────────────────────────
    act_tbl = Table.grid(padding=(0, 1), expand=True)
    act_tbl.add_column(style=C_DIM, width=10)
    act_tbl.add_column(style=C_ACCENT, width=12)
    act_tbl.add_column(width=12)
    act_tbl.add_column(justify="right", style=C_VALUE, width=9)
    act_tbl.add_column(style=C_DIM, ratio=1)
    if activity:
        for a in activity[:12]:
            status = a["status"] or "?"
            scolour = (
                C_OK if status in ("completed", "ok") else
                C_ERR if status == "failed" else C_WARN
            )
            ts = a["created_at"] or ""
            hms = ts.split("T")[-1][:8] if ts else "?"
            dur = a.get("duration_ms")
            dur_s = f"{dur / 1000:.1f}s" if dur is not None else "—"
            act_tbl.add_row(
                hms, a["stage"] or "?",
                Text(status, style=scolour),
                dur_s, (a["doc_id"] or "")[:8],
            )
    else:
        act_tbl.add_row(Text("(no recent jobs)", style=C_DIM),
                         "", "", "", "")
    activity_panel = Panel(
        act_tbl, title="[bold]recent activity[/bold]", title_align="left",
        border_style=C_BORDER, box=BOX, padding=(0, 1),
    )

    # ── Compose layout ─────────────────────────────────────────────
    # Sized so the stack fits a ~46-row terminal. `activity` takes
    # whatever's left so resizing the terminal grows the feed
    # instead of clipping panels. The timing panel grew in 54.6.232
    # to carry the 24h sparkline and (conditionally) top failures,
    # so its size bumped 11 → 14.
    layout = Layout()
    # Phase 54.6.243 — optional alert panel between header and top
    # row. Height scales with alert count (2 rounded borders + N
    # lines capped at 6), so the layout stays tight when clean.
    splits = [Layout(header_panel, name="header", size=3)]
    if alert_panel is not None:
        alert_lines = min(len(alerts), 6)
        splits.append(
            Layout(alert_panel, name="alerts", size=alert_lines + 2)
        )
    # 54.6.234 → 22 (host load + quality strip); 54.6.235 → 24
    # (embeds + topics + year sparkline); 54.6.236 → 26 (dupe-
    # hash row + models); 54.6.237 → 28 (growth line + room for
    # GPU trend sparklines + book activity footer); 54.6.243 → 32
    # (quality-signal rows in corpus, wiki + projects in right).
    # Activity is still ratio=1 so tall terminals absorb any further
    # growth.
    splits.extend([
        Layout(name="top", size=32),
        Layout(timing_panel, name="timing", size=26),
        Layout(activity_panel, name="activity", ratio=1,
               minimum_size=8),
    ])
    layout.split_column(*splits)
    layout["top"].split_row(
        Layout(corpus_panel, name="corpus", ratio=1),
        Layout(gpu_panel, name="gpu", ratio=1),
        Layout(right_panel, name="right", ratio=1),
    )
    return layout


@app.command()
def dashboard(
    days: int = typer.Option(
        30, "--days",
        help="Trailing window for throughput + LLM usage panels. "
             "Doesn't affect the stage-timing panel (which uses the "
             "full ingestion_jobs history).",
    ),
    json_out: bool = typer.Option(
        False, "--json",
        help="Emit machine-readable JSON instead of Rich tables.",
    ),
):
    """Phase 54.6.229 (roadmap 3.11.3) — pipeline observability snapshot.

    Four-panel read-only view that composes existing telemetry tables
    (``ingestion_jobs``, ``llm_usage_log``) into a single dashboard:

      1. **Stage timing** — count, p50, p95 duration per stage. Catches
         the "embedding suddenly got slow" regressions that
         `bench --layer live` doesn't cover because it runs synthetic
         queries rather than real ingestion.
      2. **Stage failures** — error rate per stage + top error class.
         Builds on the `sciknow db failures` clinic (54.6.205) —
         this surface is the summary view; `failures` is the deep-
         dive.
      3. **Throughput trend** — documents-per-day for the trailing
         `--days` window. Spot stalls (flat line) vs bursts.
      4. **LLM cost** — tokens + seconds per operation × model from
         `llm_usage_log`. Populated by web-UI runs; CLI LLM calls
         aren't logged here today (follow-on instrumentation needed).

    Read-only — never writes to any table. Safe to run alongside
    active ingestion.

    Examples:

      sciknow db dashboard                # default 30-day window
      sciknow db dashboard --days 7       # last week only
      sciknow db dashboard --json | jq .  # scripted pipelines
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    import json as _json
    from datetime import datetime, timezone
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    since = datetime.now(timezone.utc)
    # Best-effort — if days is bad, we'll just use "all time".
    try:
        since_iso = (since.replace(microsecond=0)
                     - __import__("datetime").timedelta(days=days)
                    ).isoformat()
    except Exception:
        since_iso = None

    with get_session() as session:
        # 1) Stage timing — percentiles on completed jobs with
        # non-null duration. Order is ascending on p95 so the
        # biggest latency offenders float down.
        timing = session.execute(text("""
            SELECT stage,
                   COUNT(*) AS n,
                   percentile_cont(0.5) WITHIN GROUP
                       (ORDER BY duration_ms) AS p50,
                   percentile_cont(0.95) WITHIN GROUP
                       (ORDER BY duration_ms) AS p95,
                   AVG(duration_ms) AS mean_ms
            FROM ingestion_jobs
            WHERE status IN ('completed', 'ok')
              AND duration_ms IS NOT NULL
            GROUP BY stage
            ORDER BY p95 DESC NULLS LAST
        """)).fetchall()

        # 2) Per-stage failure rate + top error signature.
        fails = session.execute(text("""
            SELECT stage,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)
                       AS failed,
                   COUNT(*) AS total
            FROM ingestion_jobs
            GROUP BY stage
            ORDER BY failed DESC
        """)).fetchall()

        # 3) Throughput — distinct documents touched per day in
        # the trailing window. Uses DATE() because ingestion_jobs
        # created_at is timestamptz.
        if since_iso:
            throughput = session.execute(text("""
                SELECT DATE(created_at) AS day,
                       COUNT(DISTINCT document_id) AS docs,
                       COUNT(*) AS jobs
                FROM ingestion_jobs
                WHERE created_at >= CAST(:since AS timestamptz)
                GROUP BY day
                ORDER BY day DESC
                LIMIT 30
            """), {"since": since_iso}).fetchall()
        else:
            throughput = []

        # 4) LLM usage — tokens + seconds per (operation, model).
        if since_iso:
            llm = session.execute(text("""
                SELECT operation, model_name,
                       SUM(tokens) AS tokens,
                       SUM(duration_seconds) AS seconds,
                       COUNT(*) AS calls
                FROM llm_usage_log
                WHERE started_at >= CAST(:since AS timestamptz)
                GROUP BY operation, model_name
                ORDER BY tokens DESC NULLS LAST
                LIMIT 30
            """), {"since": since_iso}).fetchall()
        else:
            llm = []

    if json_out:
        console.print(_json.dumps({
            "window_days": days,
            "since": since_iso,
            "stage_timing": [
                {"stage": t[0], "n": int(t[1]),
                 "p50_ms": float(t[2]) if t[2] is not None else None,
                 "p95_ms": float(t[3]) if t[3] is not None else None,
                 "mean_ms": float(t[4]) if t[4] is not None else None}
                for t in timing
            ],
            "stage_failures": [
                {"stage": f[0], "failed": int(f[1]),
                 "total": int(f[2]),
                 "failure_rate": (f[1] / f[2]) if f[2] else 0.0}
                for f in fails
            ],
            "throughput": [
                {"day": str(t[0]), "docs": int(t[1]),
                 "jobs": int(t[2])}
                for t in throughput
            ],
            "llm_usage": [
                {"operation": l[0], "model": l[1],
                 "tokens": int(l[2] or 0),
                 "seconds": float(l[3] or 0.0),
                 "calls": int(l[4] or 0)}
                for l in llm
            ],
        }, indent=2, default=str))
        return

    # Rich rendering
    console.print(
        f"[bold]sciknow pipeline dashboard[/bold]  "
        f"[dim](window: last {days} day(s), timing: all-time)[/dim]"
    )

    # --- Stage timing panel ---
    t1 = Table(
        title="Stage timing (completed jobs only)",
        box=box.SIMPLE_HEAD, expand=True,
    )
    t1.add_column("Stage", ratio=3)
    t1.add_column("N", justify="right", width=8, style="cyan")
    t1.add_column("p50", justify="right", width=10)
    t1.add_column("p95", justify="right", width=10)
    t1.add_column("mean", justify="right", width=10, style="dim")

    def _fmt_ms(v):
        if v is None:
            return "—"
        if v >= 60_000:
            return f"{v / 60000:.1f}m"
        if v >= 1000:
            return f"{v / 1000:.1f}s"
        return f"{v:.0f}ms"

    for stage, n, p50, p95, mean_ms in timing:
        t1.add_row(
            stage, str(n), _fmt_ms(p50), _fmt_ms(p95), _fmt_ms(mean_ms)
        )
    console.print(t1)

    # --- Failure-rate panel ---
    t2 = Table(
        title="Stage failures",
        box=box.SIMPLE_HEAD, expand=True,
    )
    t2.add_column("Stage", ratio=3)
    t2.add_column("Failed", justify="right", width=8)
    t2.add_column("Total", justify="right", width=8, style="dim")
    t2.add_column("Rate", justify="right", width=8)

    any_failures = False
    for stage, failed, total in fails:
        rate = (failed / total * 100) if total else 0.0
        colour = "dim" if failed == 0 else ("yellow" if rate < 5 else "red")
        if failed:
            any_failures = True
        t2.add_row(
            f"[{colour}]{stage}[/{colour}]",
            str(failed), str(total),
            f"[{colour}]{rate:.1f}%[/{colour}]",
        )
    console.print(t2)
    if any_failures:
        console.print(
            "[dim]Drill into top error classes with "
            "`sciknow db failures --stage <name>`.[/dim]"
        )

    # --- Throughput panel ---
    if throughput:
        t3 = Table(
            title=f"Throughput — documents touched per day (last {days}d)",
            box=box.SIMPLE_HEAD, expand=True,
        )
        t3.add_column("Day", ratio=2)
        t3.add_column("Documents", justify="right", width=11,
                      style="cyan")
        t3.add_column("Jobs", justify="right", width=8, style="dim")
        t3.add_column("", ratio=4)
        max_docs = max((r[1] for r in throughput), default=1) or 1
        for day, docs, jobs in throughput:
            bar = "█" * int(40 * docs / max_docs)
            t3.add_row(str(day), str(docs), str(jobs),
                       f"[cyan]{bar}[/cyan]")
        console.print(t3)

    # --- LLM panel ---
    if llm:
        t4 = Table(
            title=f"LLM usage (last {days}d, web-UI jobs only)",
            box=box.SIMPLE_HEAD, expand=True,
        )
        t4.add_column("Operation", ratio=2)
        t4.add_column("Model", ratio=3)
        t4.add_column("Tokens", justify="right", width=10, style="cyan")
        t4.add_column("Seconds", justify="right", width=10, style="dim")
        t4.add_column("Calls", justify="right", width=7)
        for op, model, tokens, seconds, calls in llm:
            t4.add_row(
                str(op or "—"),
                str(model or "—"),
                f"{int(tokens or 0):,}",
                f"{float(seconds or 0.0):.0f}s",
                str(int(calls or 0)),
            )
        console.print(t4)
    else:
        console.print(
            f"[dim]No LLM calls logged in the last {days}d. "
            "llm_usage_log is populated from web-UI runs only; "
            "CLI LLM calls (wiki compile, extract-kg, etc.) aren't "
            "instrumented here yet — follow-on work.[/dim]"
        )


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
        source_rows = (
            session.query(Document.ingest_source, func.count(Document.id))
            .group_by(Document.ingest_source)
            .all()
        )
        total_chunks = session.query(func.count(Chunk.id)).scalar()
        embedded = (
            session.query(func.count(Chunk.id))
            .filter(Chunk.qdrant_point_id.isnot(None))
            .scalar()
        )
        with_metadata = session.query(func.count(PaperMetadata.id)).scalar()
        total_citations = session.execute(text("SELECT COUNT(*) FROM citations")).scalar()
        linked_citations = session.execute(
            text("SELECT COUNT(*) FROM citations WHERE cited_document_id IS NOT NULL")
        ).scalar()

    try:
        from sciknow.storage.qdrant import papers_collection
        qdrant = get_client()
        papers_info = qdrant.get_collection(papers_collection())
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
    table.add_row("Citations (total)", str(total_citations))
    table.add_row("Citations (cross-linked)", str(linked_citations))
    table.add_section()

    for status, count in sorted(status_rows):
        colour = "green" if status == "complete" else "red" if status == "failed" else "yellow"
        table.add_row(f"  [{colour}]{status}[/{colour}]", str(count))

    if source_rows:
        table.add_section()
        table.add_row("[bold]Ingest source[/bold]", "")
        for source, count in sorted(source_rows):
            colour = "cyan" if source == "seed" else "magenta"
            table.add_row(f"  [{colour}]{source}[/{colour}]", str(count))

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


@app.command(name="drift")
def drift_cmd(
    n: int = typer.Option(10, "--n", help="How many recent drift records to show."),
    snapshot: bool = typer.Option(False, "--snapshot",
        help="Take a fresh drift snapshot now (not tied to an expand round)."),
    tag: str = typer.Option("manual", "--tag", help="Tag for the manual snapshot."),
    reason: str = typer.Option("", "--reason", help="Free-text reason."),
):
    """Phase 54.6.118 (Tier 4 #3) — corpus-drift log.

    Shows the last N drift records from ``<project data>/expand/drift.log``,
    or takes a new snapshot with ``--snapshot``. Drift is the cosine
    distance between the current abstract-centroid and the previous
    snapshot — a fast proxy for "this expansion changed what the
    corpus is about".
    """
    from sciknow.cli import preflight
    preflight()
    from sciknow.core.project import get_active_project
    from sciknow.retrieval.corpus_drift import record_drift, read_recent

    proj = get_active_project()
    if snapshot:
        summary = record_drift(proj.root, tag=tag, reason=reason)
        console.print(f"[green]✓ Drift snapshot recorded.[/green]")
        if summary.get("drift_delta") is not None:
            console.print(
                f"  delta={summary['drift_delta']:.4f} "
                f"(cosine={summary['drift_cosine']:.4f})"
            )
        else:
            console.print("  [dim](first snapshot — no prior centroid to diff against)[/dim]")
        return

    records = read_recent(proj.root, n=n)
    if not records:
        console.print("[yellow]No drift records yet. Run `sciknow db drift --snapshot` "
                      "or trigger an expand round.[/yellow]")
        return
    console.print(f"[bold]Recent drift ({len(records)}):[/bold]")
    for r in records:
        ts = r.get("timestamp", "?")
        tag_ = r.get("tag", "-")
        delta = r.get("delta", "None")
        reason_ = (r.get("reason") or "-")[:60]
        console.print(f"  {ts}  tag={tag_:<10}  delta={delta}  {reason_}")


@app.command(name="provenance")
def provenance_cmd(
    key: Annotated[str, typer.Argument(help="DOI, arXiv ID, or document-id prefix.")],
):
    """Phase 54.6.117 (Tier 4 #1) — show why a paper entered the corpus.

    Looks up ``documents.provenance`` by DOI / arXiv ID / document-id
    prefix and pretty-prints the structured record: source, round,
    relevance query, RRF signals at selection time, seed papers,
    and any merge history from re-discovery.
    """
    import json as _json
    from sciknow.core import provenance as _prov
    doc_id, rec = _prov.lookup(key)
    if not doc_id:
        console.print(f"[red]No document matches {key!r}.[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]document_id:[/bold] {doc_id}")
    if not rec:
        console.print("[yellow]No provenance record on this document.[/yellow]")
        console.print("[dim]Provenance writes started in Phase 54.6.117; "
                      "papers ingested before that have no record.[/dim]")
        raise typer.Exit(0)
    console.print(f"[bold]source:[/bold] {rec.get('source', '?')}")
    if "round" in rec:
        console.print(f"[bold]round:[/bold] {rec['round']}")
    if rec.get("relevance_query"):
        console.print(f"[bold]relevance_query:[/bold] {rec['relevance_query']}")
    if rec.get("question"):
        console.print(f"[bold]question:[/bold] {rec['question']}")
    if rec.get("subtopic"):
        console.print(f"[bold]subtopic:[/bold] {rec['subtopic']}")
    if rec.get("signals"):
        console.print("[bold]signals:[/bold]")
        for k, v in rec["signals"].items():
            if isinstance(v, float):
                console.print(f"  {k:<24} {v:.4f}")
            else:
                console.print(f"  {k:<24} {v}")
    if rec.get("selected_at"):
        console.print(f"[dim]selected_at: {rec['selected_at']}[/dim]")
    if rec.get("history"):
        console.print(f"\n[dim]History ({len(rec['history'])}):[/dim]")
        for h in rec["history"]:
            console.print(f"  [dim]- {h.get('source')} round={h.get('round')} "
                          f"selected_at={h.get('selected_at')}[/dim]")


@app.command(name="reconcile-preprints")
def reconcile_preprints_cmd(
    dry_run: bool = typer.Option(False, "--dry-run",
        help="Show the candidate pairs without writing canonical_document_id."),
):
    """Phase 54.6.125 (Tier 3 #3) — detect + reconcile preprint+journal pairs.

    Groups corpus DOIs by OpenAlex work_id and, for any group of ≥ 2
    papers, picks one canonical (journal > preprint; tie by chunk
    count; then year; then deterministic doc_id) and marks the others
    with ``canonical_document_id`` pointing at the canonical. The
    non-canonical rows become invisible to retrieval but are NOT
    deleted — run ``sciknow db unreconcile <doc_id>`` to reverse.

    Preprint DOI is copied onto the canonical's
    ``paper_metadata.extra.preprint_doi`` so both identifiers stay
    visible on the single canonical row.

    Examples:

      sciknow db reconcile-preprints --dry-run     # show the plan
      sciknow db reconcile-preprints               # apply
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    from sciknow.core.preprint_reconcile import (
        detect_pairs, apply_reconciliation,
    )

    t0 = time.monotonic()
    console.print("[dim]Scanning corpus DOIs against OpenAlex (cached, 1 call/DOI)…[/dim]")

    def _progress(i: int, total: int, doi: str) -> None:
        if i % 25 == 0:
            console.print(f"  [dim][{i}/{total}] {doi[:60]}[/dim]")

    pairs = detect_pairs(on_progress=_progress)
    elapsed = time.monotonic() - t0

    if not pairs:
        console.print(f"[green]No preprint+journal duplicates found.[/green] "
                      f"[dim]({elapsed:.1f}s)[/dim]")
        raise typer.Exit(0)

    console.print(f"\n[bold]{len(pairs)} reconciliation pair(s):[/bold]\n")
    for i, p in enumerate(pairs, 1):
        console.print(
            f"  [bold]{i}.[/bold]  "
            f"[green]canonical[/green] {p.canonical.doc_id[:8]}  "
            f"({p.canonical.year or '????'}, {p.canonical.n_chunks} chunks)  "
            f"{p.canonical.doi}"
        )
        console.print(
            f"      [red]non-canon[/red] {p.non_canonical.doc_id[:8]}  "
            f"({p.non_canonical.year or '????'}, {p.non_canonical.n_chunks} chunks)  "
            f"{p.non_canonical.doi}"
        )
        console.print(
            f"      [dim]title: {(p.canonical.title or '')[:80]}[/dim]"
        )
        console.print(f"      [dim]reason: {p.reason} · openalex: {p.openalex_work_id}[/dim]")

    if dry_run:
        console.print("\n[dim]Dry run — nothing written. Re-run without --dry-run to apply.[/dim]")
        raise typer.Exit(0)

    applied = 0
    for p in pairs:
        try:
            apply_reconciliation(p)
            applied += 1
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]apply failed for {p.non_canonical.doc_id[:8]}: {exc}[/red]")
    console.print(
        f"\n[green]✓ Reconciled {applied}/{len(pairs)} pair(s).[/green] "
        f"[dim]({elapsed:.1f}s total)[/dim]"
    )
    console.print(
        "[dim]Non-canonical rows are now hidden from retrieval. "
        "Run `sciknow db reconciliations` to list them, "
        "`sciknow db unreconcile <doc_id>` to undo.[/dim]"
    )


@app.command(name="reconciliations")
def reconciliations_cmd():
    """List current preprint↔journal reconciliations."""
    from sciknow.cli import preflight
    preflight(qdrant=False)
    from sciknow.core.preprint_reconcile import list_reconciliations

    pairs = list_reconciliations()
    if not pairs:
        console.print("[dim]No reconciliations on this project.[/dim]")
        raise typer.Exit(0)
    console.print(f"[bold]{len(pairs)} reconciliation(s):[/bold]\n")
    for i, p in enumerate(pairs, 1):
        console.print(
            f"  [bold]{i}.[/bold]  [green]canonical[/green] "
            f"{p['canonical_id'][:8]}  ({p['canonical_year'] or '????'})  "
            f"{p['canonical_doi']}"
        )
        console.print(
            f"      [red]non-canon[/red] {p['non_canonical_id'][:8]}  "
            f"({p['non_canonical_year'] or '????'})  {p['non_canonical_doi']}"
        )
        console.print(
            f"      [dim]{(p['canonical_title'] or '')[:90]}[/dim]"
        )


@app.command(name="unreconcile")
def unreconcile_cmd(
    doc_id: Annotated[str, typer.Argument(
        help="Non-canonical document_id (prefix OK). Clears canonical_document_id.")],
):
    """Reverse a reconciliation: clear canonical_document_id on the
    non-canonical row so it surfaces in retrieval again."""
    from sciknow.cli import preflight
    preflight(qdrant=False)
    from sqlalchemy import text as _t
    from sciknow.storage.db import get_session
    from sciknow.core.preprint_reconcile import undo_reconciliation

    # Resolve prefix to a full id
    with get_session() as session:
        row = session.execute(_t("""
            SELECT id::text FROM documents
            WHERE id::text LIKE :p || '%' AND canonical_document_id IS NOT NULL
            LIMIT 2
        """), {"p": doc_id}).fetchall()
    if len(row) == 0:
        console.print(f"[red]No non-canonical document matches {doc_id!r}.[/red]")
        raise typer.Exit(1)
    if len(row) > 1:
        console.print(f"[red]{doc_id!r} is ambiguous — matches {len(row)} documents. "
                      "Give more of the UUID.[/red]")
        raise typer.Exit(2)
    full = row[0][0]
    if undo_reconciliation(full):
        console.print(f"[green]✓ Unreconciled[/green] {full[:8]}. "
                      "Chunks will surface in retrieval again.")
    else:
        console.print(f"[yellow]No change — {full[:8]} wasn't marked non-canonical.[/yellow]")


@app.command(name="expand-inbound")
def expand_inbound_cmd(
    per_seed_cap: int = typer.Option(50, "--per-seed-cap",
        help="Max citing papers to fetch per corpus seed (OpenAlex page size)."),
    total_limit: int = typer.Option(500, "--total-limit",
        help="Hard cap on the raw candidate pool before relevance/dedup."),
    limit: int = typer.Option(20, "--limit", "-n",
        help="Max papers to download + ingest this run."),
    relevance_threshold: float = typer.Option(
        0.55, "--relevance-threshold",
        help="Drop candidates with bge-m3 cosine below this vs corpus centroid."),
    relevance_query: str = typer.Option("", "--relevance-query", "-q",
        help="Optional free-text topic anchor for the relevance filter."),
    dry_run: bool = typer.Option(False, "--dry-run",
        help="Print the shortlist without downloading."),
    retry_failed: bool = typer.Option(False, "--retry-failed",
        help="Ignore the prior no_oa / ingest_failed caches for these candidates."),
    ingest: bool = typer.Option(True, "--ingest/--no-ingest"),
):
    """Phase 54.6.123 (Tier 3 #2) — inbound "cites-me" expansion.

    Finds papers that **cite** papers already in the corpus
    (``OpenAlex filter=cites:<seed_id>``). Complementary to the Phase
    49 outbound crawl (``db expand`` follows OUR references); this
    one discovers recent papers the corpus hasn't cited yet but which
    cite us. Useful for staying current on a topic once the corpus
    is substantial.

    Applies the bge-m3 relevance filter + the existing cache-skip
    (no_oa / ingest_failed) + download-pipeline retraction/predatory
    filters, then downloads + ingests up to ``--limit`` survivors.
    Writes provenance with ``source="expand-inbound"``.

    Examples:

      sciknow db expand-inbound                             # top 20 @ 0.55 thr
      sciknow db expand-inbound -n 50 --relevance-threshold 0.6
      sciknow db expand-inbound --dry-run                   # see shortlist only
      sciknow db expand-inbound -q "climate sensitivity" -n 30
    """
    from sciknow.cli import preflight
    preflight()

    from pathlib import Path as _Path
    from sciknow.config import settings as _s
    from sciknow.core.expand_ops import find_inbound_citation_candidates
    from sciknow.core import provenance as _prov
    from sciknow.ingestion.downloader import find_and_download

    download_dir = _s.data_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"\n[bold]Inbound expansion[/bold]  "
        f"(cites-me crawl, per-seed-cap={per_seed_cap}, "
        f"total-limit={total_limit})"
    )

    t0 = time.monotonic()
    result = find_inbound_citation_candidates(
        per_seed_cap=per_seed_cap,
        total_limit=total_limit,
        relevance_query=relevance_query,
        score_relevance=True,
    )
    cands = result.get("candidates") or []
    info = result.get("info") or {}

    # Apply relevance threshold + cached-skip + final limit
    def _keep(c: dict) -> bool:
        score = c.get("relevance_score")
        if relevance_threshold > 0 and (score is None or score < relevance_threshold):
            return False
        if not retry_failed and c.get("cached_status"):
            return False
        return True

    survivors = [c for c in cands if _keep(c)]
    survivors = survivors[:limit]

    console.print(
        f"  seeds resolved: {info.get('seeds_resolved', 0)} / "
        f"{info.get('seeds_requested', 0)}\n"
        f"  raw candidates: {len(cands)}\n"
        f"  after threshold (≥{relevance_threshold}) + cache-skip: {len(survivors)}\n"
    )

    if not survivors:
        console.print("[yellow]Nothing to download.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]Top {len(survivors)}:[/bold]")
    for i, c in enumerate(survivors, 1):
        score = c.get("relevance_score")
        score_s = f"{score:.3f}" if isinstance(score, float) else "—"
        console.print(
            f"  {i:>3}. [{score_s}]  {c.get('year') or '????'}  "
            f"{(c.get('title') or '')[:100]}"
        )

    if dry_run:
        console.print("\n[dim]Dry run — nothing downloaded.[/dim]")
        raise typer.Exit(0)

    downloaded = skipped = failed_dl = 0
    paths_to_ingest: list = []
    for i, c in enumerate(survivors, 1):
        doi = (c.get("doi") or "").strip()
        if not doi:
            continue
        try:
            pdf_path = find_and_download(
                doi=doi, arxiv_id=c.get("arxiv_id"),
                download_dir=download_dir,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]✗[/red] {i:>3}. {doi}  {exc}")
            failed_dl += 1
            continue
        if pdf_path is None:
            failed_dl += 1
            console.print(f"  [yellow]⊘[/yellow] {i:>3}. {doi}  no OA PDF")
            continue
        downloaded += 1
        paths_to_ingest.append((pdf_path, c))
        console.print(f"  [green]↓[/green] {i:>3}. {doi}")

    # Ingest using the existing parallel worker pool
    ingested = 0
    if ingest and paths_to_ingest:
        from sciknow.cli.db import _run_parallel_workers
        from rich.progress import Progress

        ingest_results: list = []
        ingest_failed_files: list = []
        with Progress(console=console, transient=False) as progress:
            itask = progress.add_task("Ingesting", total=len(paths_to_ingest))
            _run_parallel_workers(
                [p for p, _ in paths_to_ingest],
                progress, itask,
                ingest_results, ingest_failed_files,
                force=False, num_workers=1,
                ingest_source="expand-inbound",
            )
        ingested = len([r for r in ingest_results if r.get("status") == "complete"])

        # Provenance: mark each ingested paper with source=expand-inbound
        for _pdf, c in paths_to_ingest:
            doi = c.get("doi")
            if not doi:
                continue
            try:
                _prov.record(
                    doi=doi, source="expand-inbound",
                    relevance_query=relevance_query or None,
                    signals={
                        "relevance_score": c.get("relevance_score"),
                        "cited_by_count": c.get("cited_by_count"),
                        "year": c.get("year"),
                    },
                )
            except Exception:  # noqa: BLE001
                pass

    console.print(
        f"\n[bold]Summary:[/bold] "
        f"[green]↓ {downloaded} downloaded[/green]  "
        f"[green]✓ {ingested} ingested[/green]  "
        f"[red]✗ {failed_dl} no OA PDF[/red]  "
        f"[dim]{time.monotonic() - t0:.1f}s wall[/dim]"
    )


@app.command(name="expand-oeuvre")
def expand_oeuvre_cmd(
    min_corpus_papers: int = typer.Option(3, "--min-corpus-papers",
        help="Author must have ≥ N papers already in the corpus to qualify."),
    per_author_limit: int = typer.Option(10, "--per-author-limit",
        help="Cap on new papers downloaded per author per run."),
    max_authors: int = typer.Option(10, "--max-authors",
        help="Upper bound on authors to expand in one run (sorted by corpus paper count, descending)."),
    dry_run: bool = typer.Option(False, "--dry-run",
        help="Show the author list + expansion plan without downloading anything."),
    relevance_query: str = typer.Option("", "--relevance-query", "-q",
        help="Free-text relevance anchor for each per-author expansion. Empty → corpus centroid."),
    strict_author: bool = typer.Option(True, "--strict-author/--no-strict-author",
        help="Pass through to db expand-author. Default ON: only OpenAlex canonical-author-ID matches."),
):
    """Phase 54.6.116 (Tier 2 #4) — auto-complete author oeuvres.

    Finds every author who already has ≥``--min-corpus-papers`` papers
    in the corpus and runs ``sciknow db expand-author`` for each,
    capping new downloads per author. Complementary to ``db expand``
    (outbound reference crawl) and ``db expand-author`` (manual single-
    author expansion): this automates the "go through the most-
    represented authors' full bibliographies" step.

    Uses existing ``search_author`` / ``expand-author`` machinery via
    subprocess so each run is fully gated by relevance filter,
    retraction / predatory / one-timer filters, MMR diversity, and
    citation-context signals (same as any other expansion). ORCIDs
    from ``paper_metadata.authors[*].orcid`` are passed through when
    present for disambiguation; falls back to strict-author name match.

    Examples:

      sciknow db expand-oeuvre                     # default: 10 authors × 10 papers
      sciknow db expand-oeuvre --min-corpus-papers 5 --max-authors 5 --dry-run
      sciknow db expand-oeuvre -q "climate sensitivity"
    """
    from sciknow.cli import preflight
    preflight()

    import subprocess
    import sys as _sys
    from sciknow.core.expand_ops import qualifying_oeuvre_authors

    qualifying = qualifying_oeuvre_authors(
        min_corpus_papers=min_corpus_papers, max_authors=max_authors,
    )

    if not qualifying:
        console.print(
            f"[yellow]No author has ≥{min_corpus_papers} corpus papers. "
            "Try lowering --min-corpus-papers or expanding the corpus first.[/yellow]"
        )
        raise typer.Exit(0)

    console.print(
        f"\n[bold]Expand oeuvre[/bold]  "
        f"{len(qualifying)} qualifying author(s) "
        f"(≥{min_corpus_papers} corpus papers), "
        f"cap {per_author_limit} new paper(s) each:\n"
    )
    for row in qualifying:
        orcid_hint = (
            f"  [dim]orcid={row['orcid']}[/dim]" if row["orcid"] else ""
        )
        console.print(
            f"  • {row['n_corpus_papers']:>3} corpus papers  {row['name']}{orcid_hint}"
        )

    if dry_run:
        console.print("\n[dim]Dry run — nothing executed.[/dim]")
        raise typer.Exit(0)

    t0 = time.monotonic()
    done = 0
    for i, row in enumerate(qualifying, start=1):
        name = row["name"]
        n = row["n_corpus_papers"]
        console.print(f"\n[bold]Author {i}/{len(qualifying)}[/bold]: {name} "
                      f"([cyan]{n}[/cyan] in corpus)")
        argv = [
            _sys.executable, "-m", "sciknow.cli.main",
            "db", "expand-author", name,
            "--limit", str(per_author_limit),
        ]
        if row["orcid"]:
            argv += ["--orcid", row["orcid"]]
        if strict_author:
            argv.append("--strict-author")
        if relevance_query:
            argv += ["--relevance-query", relevance_query]
        try:
            res = subprocess.run(argv, check=False)
            if res.returncode == 0:
                done += 1
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]author expansion failed: {exc}[/red]")

    console.print(
        f"\n[green]✓ Oeuvre expansion done:[/green] "
        f"{done}/{len(qualifying)} authors completed  "
        f"[dim]{time.monotonic() - t0:.1f}s wall[/dim]"
    )


@app.command(name="feedback-list")
def feedback_list_cmd():
    """Phase 54.6.115 (Tier 2 #3) — show the project's expand feedback.

    ±Marks from prior shortlist reviews. Positive entries bias the
    ranker's anchor vector toward similar papers; negative entries
    bias it away. File lives at ``<project root>/expand_feedback.json``.
    """
    from sciknow.core.project import get_active_project
    from sciknow.core import expand_feedback as _fb
    project = get_active_project()
    fb = _fb.load(project.root)
    path = _fb.path_for(project.root)
    console.print(f"[bold]{project.slug}[/bold]  {path}"
                  + ("  [dim](file not yet present)[/dim]" if not path.exists() else ""))
    console.print(f"[green]Positive[/green] ({len(fb.positive)}):")
    for e in fb.positive:
        who = e.doi or e.arxiv_id or e.title[:80]
        console.print(f"  + [{who}]  {e.title[:80]}"
                      + (f"  [dim]topic={e.topic}[/dim]" if e.topic else ""))
    console.print(f"[red]Negative[/red] ({len(fb.negative)}):")
    for e in fb.negative:
        who = e.doi or e.arxiv_id or e.title[:80]
        console.print(f"  - [{who}]  {e.title[:80]}"
                      + (f"  [dim]topic={e.topic}[/dim]" if e.topic else ""))


@app.command(name="feedback-add")
def feedback_add_cmd(
    kind: Annotated[str, typer.Argument(help="'positive' or 'negative' (or 'pos'/'neg').")],
    doi: str = typer.Option("", "--doi", help="DOI of the paper."),
    arxiv_id: str = typer.Option("", "--arxiv", help="arXiv ID."),
    title: str = typer.Option("", "--title", help="Paper title (for display + fallback embedding)."),
    topic: str = typer.Option("", "--topic", help="Optional free-text topic for provenance."),
):
    """Add a ±mark to the project's expand feedback.

    Examples:

      sciknow db feedback-add positive --doi 10.1029/2022JD037524 --topic "ECS"
      sciknow db feedback-add neg --title "Off-topic computer-vision paper"
    """
    kind_norm = {"pos": "positive", "positive": "positive",
                 "neg": "negative", "negative": "negative",
                 "+": "positive", "-": "negative"}.get(kind.strip().lower())
    if not kind_norm:
        console.print(f"[red]kind must be 'positive' | 'negative'.[/red]")
        raise typer.Exit(2)
    if not (doi or arxiv_id or title):
        console.print("[red]Provide at least one of --doi, --arxiv, --title.[/red]")
        raise typer.Exit(2)
    from sciknow.core.project import get_active_project
    from sciknow.core import expand_feedback as _fb
    fb, added = _fb.add_entry(
        get_active_project().root,
        kind=kind_norm, doi=doi, arxiv_id=arxiv_id,
        title=title, topic=topic,
    )
    badge = "[green]+[/green]" if kind_norm == "positive" else "[red]-[/red]"
    if added:
        console.print(f"{badge} {kind_norm} mark added. "
                      f"Total: +{len(fb.positive)} / -{len(fb.negative)}.")
    else:
        console.print("[yellow]Nothing added — no usable identifier.[/yellow]")


@app.command(name="feedback-remove")
def feedback_remove_cmd(
    key: Annotated[str, typer.Argument(help="DOI, arXiv ID, or title prefix.")],
    kind: str = typer.Option("", "--kind", help="Restrict to 'positive' or 'negative'."),
):
    """Remove a ±mark from the project's expand feedback."""
    kind_norm = None
    if kind:
        kind_norm = {"pos": "positive", "positive": "positive",
                     "neg": "negative", "negative": "negative"}.get(kind.strip().lower())
        if kind_norm is None:
            console.print("[red]--kind must be 'positive' or 'negative' if given.[/red]")
            raise typer.Exit(2)
    from sciknow.core.project import get_active_project
    from sciknow.core import expand_feedback as _fb
    fb, removed = _fb.remove_entry(
        get_active_project().root, key=key, kind=kind_norm,
    )
    if removed:
        console.print(f"[green]✓ Removed[/green] {key!r}. "
                      f"Total: +{len(fb.positive)} / -{len(fb.negative)}.")
    else:
        console.print(f"[yellow]Not found:[/yellow] {key!r}")


@app.command(name="refresh-retractions")
def refresh_retractions_cmd(
    limit: int = typer.Option(0, "--limit", "-n",
        help="Max papers to check this run (0 = all)."),
    max_age_days: int = typer.Option(30, "--max-age-days",
        help="Skip papers whose retraction_checked_at is newer than N days."),
    delay: float = typer.Option(0.1, "--delay",
        help="Seconds between Crossref API calls."),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Phase 54.6.111 (Tier 1 #2) — sweep the corpus for retractions.

    Re-checks every paper with a DOI against Crossref's
    ``update-type:retraction`` index and fills
    ``paper_metadata.retraction_status`` /
    ``paper_metadata.retraction_checked_at``. Skips papers checked
    within ``--max-age-days``; pass ``--max-age-days 0`` to force.

    Downstream effects: retracted papers are dropped by ``hybrid_search``
    (see 54.6.81 paper-type weighting — retraction is treated as a
    hard filter, not a soft weight) and flagged in the dashboard.

    Examples:

      sciknow db refresh-retractions              # all eligible
      sciknow db refresh-retractions -n 100
      sciknow db refresh-retractions --max-age-days 0   # force
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    from datetime import datetime, timedelta, timezone
    from sqlalchemy import text as sql_text
    from sciknow.storage.db import get_session
    import httpx
    from sciknow.config import settings as _s

    _ua = {"User-Agent": f"sciknow/0.1 (mailto:{_s.crossref_email})"}

    def _fetch_work(doi: str) -> dict | None:
        try:
            with httpx.Client(timeout=15) as c:
                r = c.get(f"https://api.crossref.org/works/{doi}", headers=_ua)
                if r.status_code != 200:
                    return None
                return r.json()
        except Exception:
            return None

    now = datetime.now(timezone.utc)
    max_age = timedelta(days=max_age_days) if max_age_days > 0 else None

    with get_session() as session:
        if max_age is None:
            rows = session.execute(sql_text("""
                SELECT id::text, doi FROM paper_metadata
                WHERE doi IS NOT NULL AND doi <> ''
                ORDER BY COALESCE(retraction_checked_at, '1970-01-01'::timestamptz) ASC
            """)).fetchall()
        else:
            cutoff = now - max_age
            rows = session.execute(sql_text("""
                SELECT id::text, doi FROM paper_metadata
                WHERE doi IS NOT NULL AND doi <> ''
                  AND (retraction_checked_at IS NULL OR retraction_checked_at < :cutoff)
                ORDER BY COALESCE(retraction_checked_at, '1970-01-01'::timestamptz) ASC
            """), {"cutoff": cutoff}).fetchall()

    if limit:
        rows = rows[:limit]

    total = len(rows)
    if total == 0:
        console.print("[green]Nothing to check — every paper was checked "
                      f"within the last {max_age_days} days.[/green]")
        raise typer.Exit(0)

    console.print(
        f"Checking [bold]{total}[/bold] paper(s) against Crossref retraction index"
        + (f" (max-age {max_age_days}d)" if max_age else " (forced)")
        + "…"
    )

    retracted = corrected = clean = errors = 0
    t0 = time.monotonic()

    for i, (pid, doi) in enumerate(rows, 1):
        # Crossref's update-type filter on /works/{doi} tells us whether
        # ANY retraction/withdrawal/correction has been attached. The
        # "relation" field in the work metadata surfaces these edges.
        status = "none"
        try:
            work = _fetch_work(doi)
            msg = (work or {}).get("message", {}) if work else {}
            updates = msg.get("update-to") or []
            updated_by = msg.get("updated-by") or []
            # A paper is retracted when any record "updated-by" it carries
            # update-type=retraction / withdrawal. Corrections treated
            # separately so the writer can still cite (with a flag).
            flags = [u.get("type", "").lower() for u in (updates + updated_by)]
            if any(f in ("retraction", "withdrawal") for f in flags):
                status = "retracted"
                retracted += 1
            elif any(f == "correction" for f in flags):
                status = "corrected"
                corrected += 1
            else:
                clean += 1
        except Exception as exc:  # noqa: BLE001
            status = "error"
            errors += 1
            logger.debug("refresh-retractions %s: %s", doi, exc)

        if status == "retracted":
            console.print(f"  [dim][{i}/{total}][/dim] [red]✗ RETRACTED[/red]  {doi}")
        elif status == "corrected":
            console.print(f"  [dim][{i}/{total}][/dim] [yellow]! CORRECTED[/yellow]  {doi}")
        elif status == "error":
            console.print(f"  [dim][{i}/{total}][/dim] [dim]? error[/dim]  {doi}")

        if not dry_run and status != "error":
            with get_session() as session:
                session.execute(sql_text("""
                    UPDATE paper_metadata
                    SET retraction_status = :s, retraction_checked_at = :ts
                    WHERE id::text = :pid
                """), {"s": status, "ts": now, "pid": pid})
                session.commit()

        if delay > 0 and i < total:
            time.sleep(delay)

    console.print(
        f"\n[red]Retracted: {retracted}[/red] · "
        f"[yellow]Corrected: {corrected}[/yellow] · "
        f"[green]Clean: {clean}[/green] · "
        f"[dim]Errors: {errors}[/dim] · "
        f"[dim]{time.monotonic() - t0:.1f}s wall[/dim]"
    )


@app.command()
def enrich(
    dry_run:   bool  = typer.Option(False,  "--dry-run",   help="Show what would be updated without making changes."),
    threshold: float = typer.Option(0.78,   "--threshold",
                                     help="Minimum title-similarity score to accept a single-signal match (0–1). "
                                          "Phase 51: lowered from 0.85 → 0.78 because the dual-signal author+year "
                                          "validation now covers the false-positive space a lower single-signal "
                                          "cutoff would open on its own. Bump back to 0.85 for a stricter run."),
    author_threshold: float = typer.Option(0.70, "--author-threshold",
                                            help="Minimum title similarity when author surname AND year also agree "
                                                 "— the dual-signal 'abarcative' floor. Default 0.70."),
    year_tolerance: int = typer.Option(1, "--year-tolerance",
                                        help="±years by which candidate publication_year can differ from ours "
                                             "and still count as a year-match. Default ±1."),
    shortlist_tsv: Path = typer.Option(None, "--shortlist-tsv",
                                        help="Dump every paper + its best candidate + all three signals to this "
                                             "TSV for HITL review. Useful for tuning thresholds or manual DOI entry."),
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
    from sciknow.cli import preflight
    preflight(qdrant=False)  # enrich only touches PostgreSQL

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
            SELECT pm.id::text, pm.title, pm.authors, pm.arxiv_id, pm.metadata_source,
                   pm.year, pm.abstract
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

    def _lookup(row) -> tuple[str, str, PaperMeta | None, str, int | None]:
        """Pure API lookup — runs in worker thread, never touches the DB.
        Returns (pm_id, title, meta_or_none, status, year)."""
        pm_id, title, authors, arxiv_id, _meta_src, pm_year, pm_abstract = row

        if _is_garbage_title(title) or len(title.strip()) < 15:
            return pm_id, title, None, "skip", pm_year

        first_author: str | None = None
        if authors:
            first_author = (authors[0] or {}).get("name")

        meta = search_crossref_by_title(
            title, first_author,
            threshold=threshold,
            year=pm_year,
            our_abstract=pm_abstract,
            author_threshold=author_threshold,
            year_tolerance=year_tolerance,
        )
        if meta is None:
            meta = search_openalex_by_title(
                title, first_author,
                threshold=threshold,
                year=pm_year,
                our_abstract=pm_abstract,
                author_threshold=author_threshold,
                year_tolerance=year_tolerance,
            )
        if meta is None and arxiv_id:
            stub = PaperMeta(arxiv_id=arxiv_id)
            _layer_arxiv(stub)
            if stub.title:
                meta = stub

        return pm_id, title, meta, "ok" if meta else "no_match", pm_year

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

            # Phase 51 — shortlist rows (optional HITL review output)
            shortlist_rows: list[dict] = []

            # Phase 54.6.57 — durable per-item log lines. Mirrors the
            # expand 54.6.45 pattern so the web UI log pane shows one
            # scrollable event per paper (MATCH / ARXIV / NO_MATCH /
            # SKIP / FAIL) instead of just a silent progress bar. The
            # in-place description update is kept for terminal users;
            # `progress.console.print` stacks durable lines above it.
            total_rows = len(rows)
            done_count = 0

            def _emit(mark: str, color: str, kind: str, label_: str, note: str = "") -> None:
                """Print one durable line above the live progress bar."""
                progress.console.print(
                    f"[dim][{done_count:>4d}/{total_rows}][/dim]  "
                    f"[{color}]{mark} {kind:<8}[/{color}] {label_[:70]:<70}"
                    + (f"  [dim]· {note}[/dim]" if note else "")
                )

            for fut in as_completed(futures):
                try:
                    pm_id, title, meta, status, pm_year = fut.result()
                except Exception as exc:
                    failed += 1
                    done_count += 1
                    _emit("⚠", "red", "FAIL", "(lookup exception)", str(exc)[:80])
                    progress.advance(task)
                    continue
                if shortlist_tsv:
                    shortlist_rows.append({
                        "pm_id": pm_id,
                        "title": title or "",
                        "year": pm_year,
                        "status": status,
                        "matched_doi": (meta.doi if meta else "") or "",
                        "matched_title": (meta.title if meta else "") or "",
                        "matched_year": (meta.year if meta else None),
                        "source": (meta.source if meta else "") or "",
                    })

                progress.update(task, description=f"[dim]{title[:55]}[/dim]")
                done_count += 1

                if status == "skip":
                    skipped += 1
                    _emit("⊘", "yellow", "SKIP", title or "(no title)",
                          "garbage/short title")
                    progress.advance(task)
                    continue
                if meta is None:
                    skipped += 1
                    _emit("✗", "yellow", "NO_MATCH", title or "(no title)",
                          f"below threshold={threshold}")
                    progress.advance(task)
                    continue

                if dry_run:
                    doi_str = (f"doi:{meta.doi}" if meta.doi
                               else f"arXiv:{meta.arxiv_id}")
                    kind = "DRY_ARXIV" if meta.arxiv_id and not meta.doi else "DRY_OK"
                    _emit("✓", "green", kind, title or "(no title)",
                          f"{doi_str} via {meta.source or '?'}")
                    matched += 1
                    progress.advance(task)
                    continue

                try:
                    with get_session() as session:
                        pm = session.query(PaperMetadata).filter_by(id=pm_id).first()
                        if pm is None:
                            skipped += 1
                            _emit("⊘", "yellow", "SKIP", title or "(no title)",
                                  "row disappeared during lookup")
                            progress.advance(task)
                            continue

                        # Phase 54.6.111 (Tier 1 #4) — track abstract
                        # transitions so we can re-embed the paper when
                        # enrich lands an abstract on a previously-null
                        # row (the paper's chunk embeddings were built
                        # without that abstract prefix, so retrieval was
                        # degraded).
                        abstract_was_null = not (pm.abstract or "").strip()
                        abstract_will_be_filled = bool((meta.abstract or "").strip())

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

                        # Phase 54.6.111 (Tier 1 #1) — hydrate oa_*
                        # columns from the OpenAlex raw on the match
                        # object when present. The search_openalex_…
                        # helpers already attach the raw work on meta
                        # (they keep the full response for debugging).
                        try:
                            from sciknow.ingestion.openalex_enrich import (
                                extract_openalex_enrichment,
                            )
                            oa_raw = getattr(meta, "openalex_raw", None) or {}
                            oa_updates = extract_openalex_enrichment(oa_raw)
                            for k, v in oa_updates.items():
                                setattr(pm, k, v)
                        except Exception:  # noqa: BLE001
                            pass

                        session.commit()

                        # Re-embed when a previously empty abstract just
                        # got filled. Best-effort — catches embedder
                        # import/VRAM errors so enrich doesn't abort.
                        if abstract_was_null and abstract_will_be_filled:
                            try:
                                from sciknow.ingestion.embedder import (
                                    embed_paper_abstract,
                                )
                                embed_paper_abstract(str(pm.id))
                            except Exception as exc:  # noqa: BLE001
                                logger.debug(
                                    "re-embed on enrich failed for %s: %s",
                                    pm.id, exc,
                                )

                    matched += 1
                    doi_str = (f"doi:{meta.doi}" if meta.doi
                               else f"arXiv:{meta.arxiv_id}")
                    kind = "ARXIV" if meta.arxiv_id and not meta.doi else "MATCH"
                    _emit("✓", "green", kind, title or "(no title)",
                          f"{doi_str} via {meta.source or '?'}")
                except Exception as exc:
                    failed += 1
                    _emit("⚠", "red", "FAIL", title or "(no title)",
                          f"DB write: {str(exc)[:80]}")

                progress.advance(task)

    # Phase 51 — dump the shortlist TSV if requested. Includes every
    # row (matched + rejected + skipped) so the user can grep for
    # near-misses and bump --threshold or fill a DOI by hand.
    if shortlist_tsv and shortlist_rows:
        shortlist_tsv.parent.mkdir(parents=True, exist_ok=True)
        with shortlist_tsv.open("w", encoding="utf-8") as f:
            f.write("\t".join([
                "pm_id", "status", "our_title", "our_year",
                "matched_doi", "matched_title", "matched_year", "source",
            ]) + "\n")
            for r in shortlist_rows:
                f.write("\t".join([
                    r["pm_id"],
                    r["status"],
                    (r["title"] or "").replace("\t", " ").replace("\n", " ")[:200],
                    str(r["year"] or ""),
                    r["matched_doi"],
                    (r["matched_title"] or "").replace("\t", " ")[:200],
                    str(r["matched_year"] or ""),
                    r["source"],
                ]) + "\n")
        console.print(
            f"[dim]Shortlist TSV: {shortlist_tsv}  ({len(shortlist_rows)} rows)[/dim]"
        )

    console.print(
        f"\n[green]✓ Matched & updated {matched}[/green]  "
        f"[yellow]no match {skipped}[/yellow]  "
        f"[red]failed {failed}[/red]  "
        f"[dim]thresholds: title≥{threshold:.2f} single, "
        f"≥{author_threshold:.2f} + author/year dual[/dim]"
    )
    if not dry_run and matched:
        console.print(
            f"\nRun [bold]sciknow catalog stats[/bold] to see the updated coverage."
        )


@app.command()
def expand(
    # Phase 43d — default resolved in body so the active project's
    # data_dir is used (Typer evaluates option defaults at import time,
    # which would freeze a legacy path). Pass None sentinel; resolve
    # via settings.data_dir / "downloads" below.
    download_dir: Path  = typer.Option(None, "--download-dir", "-d",
                                        help="Directory where new PDFs are saved before ingestion (default: <project data>/downloads)."),
    limit:        int   = typer.Option(0,     "--limit",     help="Max new papers to download (0 = all found)."),
    resolve:      bool  = typer.Option(False, "--resolve/--no-resolve",
                                        help="Also resolve title-only references via Crossref (slow, ~0.3s each)."),
    ingest:       bool  = typer.Option(True,  "--ingest/--no-ingest",
                                        help="Ingest downloaded PDFs immediately."),
    dry_run:      bool  = typer.Option(False, "--dry-run",   help="Show what would be downloaded without doing it."),
    delay:        float = typer.Option(0.3,   "--delay",     help="Seconds between API calls."),
    relevance:          bool  = typer.Option(True,  "--relevance/--no-relevance",
                                              help="Filter candidate references by semantic relevance to the corpus."),
    relevance_threshold: float = typer.Option(0.0, "--relevance-threshold",
                                               help="Cosine similarity threshold (0 = use EXPAND_RELEVANCE_THRESHOLD from .env, default 0.55)."),
    relevance_query:    str   = typer.Option("",    "--relevance-query", "-q",
                                              help="Free-text topic anchor for the relevance filter. If empty, the corpus centroid is used."),
    workers:            int   = typer.Option(0,     "--workers", "-w",
                                              help="Parallel ingestion worker subprocesses for the post-download ingest phase "
                                                   "(0 = use INGEST_WORKERS from .env, default 1). Each worker loads its own "
                                                   "MinerU (~7GB VRAM) + bge-m3 (~2.2GB). On a 24GB 3090 with an LLM resident, "
                                                   "keep at 1. Raise to 2 only when the LLM is off-GPU."),
    # ── Phase 49 — RRF-fused multi-signal ranker (see docs/EXPAND_RESEARCH.md)
    strategy:           str   = typer.Option("rrf", "--strategy",
                                              help="Candidate ranking strategy: 'rrf' (default — multi-signal RRF "
                                                   "fusion with hard filters, co-citation, bib coupling, PageRank, "
                                                   "influential-citation flag) or 'legacy' (pre-Phase-49 bge-m3 "
                                                   "cosine filter only)."),
    budget:             int   = typer.Option(50,    "--budget",
                                              help="Max papers to queue for download per RRF round (default 50). "
                                                   "Ignored for --strategy legacy."),
    rrf_no_openalex:    bool  = typer.Option(False, "--no-openalex",
                                              help="Skip the per-candidate OpenAlex work lookup in RRF mode. Disables "
                                                   "co-citation / bib coupling / PageRank / velocity / hard filters, "
                                                   "falling back to bge-m3 cosine + one-timer only. Use when offline."),
    rrf_no_s2:          bool  = typer.Option(False, "--no-semantic-scholar",
                                              help="Skip the Semantic Scholar citations lookup (the isInfluential + "
                                                   "intents signal). Faster by ~1 s per survivor candidate. Default "
                                                   "is to include."),
    shortlist_tsv:      Path  = typer.Option(None,  "--shortlist-tsv",
                                              help="Write the full ranked shortlist (kept + dropped, every signal) "
                                                   "to this TSV path. Implies --dry-run when set. Default in RRF dry-"
                                                   "run mode: <download_dir>/expand_shortlist.tsv."),
    # ── Phase 49.1 — downloads/ hygiene + persistent failure memory
    cleanup:      bool  = typer.Option(True,  "--cleanup/--no-cleanup",
                                        help="After each ingest, move the downloaded PDF into <download_dir>/processed/ "
                                             "(success or dedup) or <download_dir>/failed_ingest/ (ingest failed, with a "
                                             ".error.txt sibling). Keeps the root of <download_dir> empty so it's easy "
                                             "to see what's still being worked on. Default ON."),
    retry_failed: bool  = typer.Option(False, "--retry-failed",
                                        help="Ignore the .ingest_failed cache and re-try permanently-failed refs from "
                                             "a prior run. Default OFF (failed refs are skipped to save compute)."),
    # ── Phase 54.6.114 (Tier 2 #2) — agentic question-driven expansion
    question:     str   = typer.Option("", "--question",
                                        help="Agentic mode (54.6.114). Give a research question; the LLM "
                                             "decomposes it into 3-6 sub-topics, measures corpus coverage for "
                                             "each, and runs targeted expansion on the gaps. Uses the corpus "
                                             "coverage check as the stopping rule (all sub-topics ≥ N papers) "
                                             "instead of the default median-drop / novelty heuristic. Mutually "
                                             "exclusive with --relevance-query."),
    question_rounds: int = typer.Option(3, "--question-rounds",
                                         help="Max agentic-mode rounds before stopping. Default 3."),
    question_budget: int = typer.Option(10, "--question-budget",
                                         help="Per-sub-topic download budget in agentic mode. Default 10."),
    question_threshold: int = typer.Option(3, "--question-threshold",
                                            help="Corpus papers per sub-topic required to call it 'covered'. Default 3."),
    # Phase 54.6.124 (Tier 4 #2) — resume from persisted round state
    question_resume: bool = typer.Option(False, "--resume",
                                          help="Resume an in-progress agentic run. State lives at "
                                               "<project>/data/expand/agentic/<slug>-<hash>.json and is "
                                               "written after each round; same --question re-uses it."),
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

    from sciknow.cli import preflight
    preflight()

    from sqlalchemy import text as sql_text

    from sciknow.config import settings
    from sciknow.ingestion.downloader import find_and_download
    from sciknow.ingestion.references import (
        extract_references_from_crossref,
        extract_references_from_markdown,
        extract_references_from_mineru_content_list,
        fetch_openalex_references,
    )

    # Phase 43d — resolve download_dir here (Typer default was None to
    # defer active-project lookup until the command actually runs).
    if download_dir is None:
        download_dir = settings.data_dir / "downloads"
    from sciknow.storage.db import get_session

    download_dir.mkdir(parents=True, exist_ok=True)

    # Phase 54.6.114 (Tier 2 #2) — agentic question-driven mode.
    # Dispatches early so the rest of the static-expand logic below
    # can assume no --question. The agentic orchestrator calls the
    # same static expand loop per-sub-topic via a callback.
    if (question or "").strip():
        if relevance_query:
            console.print(
                "[red]--question and --relevance-query are mutually "
                "exclusive. Pick one.[/red]"
            )
            raise typer.Exit(2)
        _run_agentic_expand(
            question=question.strip(),
            download_dir=download_dir,
            max_rounds=question_rounds,
            budget_per_gap=question_budget,
            doc_threshold=question_threshold,
            resume=question_resume,
            # pass-throughs to the per-sub-topic static ranker
            strategy=strategy,
            delay=delay,
            resolve=resolve,
            ingest=ingest,
            dry_run=dry_run,
            workers=workers,
            rrf_no_openalex=rrf_no_openalex,
            rrf_no_s2=rrf_no_s2,
            cleanup=cleanup,
            retry_failed=retry_failed,
        )
        return

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
        # Phase 49.1 — title-normalised dedup. Catches duplicates where
        # the incoming reference points at the same paper via a
        # different identifier (preprint DOI vs journal DOI, arXiv id
        # vs DOI, Crossref vs OpenAlex). Built alongside the DOI/arXiv
        # sets so the dedup step downstream has all three to check.
        existing_titles_norm = {
            _normalise_title_for_dedup(r[2])
            for r in papers if r[2]
        }
        existing_titles_norm.discard("")

    console.print(f"Collection: [bold]{len(papers)}[/bold] papers, "
                  f"{len(existing_dois)} with DOI, {len(existing_arxivs)} with arXiv ID")

    # ── Step 2: extract all references ───────────────────────────────────────
    # Four sources, unioned per paper:
    #   A. Crossref-stored reference list (structured, DOI-rich)
    #   B. MinerU content_list.json (primary for MinerU-ingested papers)
    #   C. Marker markdown bibliography section (legacy fallback)
    #   D. OpenAlex referenced_works (only when A+B+C yielded few refs)
    # Phase 54.6.45 — announce each step so the user doesn't stare at a
    # silent terminal for 60+ seconds during reference extraction on a
    # large corpus.
    console.print(
        f"[dim]Scanning {len(papers)} papers for references "
        f"(Crossref + MinerU + Marker)…[/dim]"
    )
    all_refs: list = []
    source_counts = {"crossref": 0, "mineru": 0, "markdown": 0, "openalex": 0}

    # First pass: local sources (A, B, C)
    needs_openalex: list[str] = []  # DOIs of papers with <10 refs locally
    for doi, arxiv_id, title, crossref_raw, marker_out in papers:
        local_count_before = len(all_refs)

        # Source A: Crossref reference list (structured, reliable)
        if crossref_raw:
            crs = extract_references_from_crossref(crossref_raw)
            all_refs.extend(crs)
            source_counts["crossref"] += len(crs)

        if marker_out:
            from pathlib import Path as _Path
            mp = _Path(marker_out)
            if mp.exists():
                # Source B: MinerU content_list.json (primary for post-switch ingests)
                import json as _json
                content_list_candidates = list(mp.rglob("*_content_list.json"))
                if not content_list_candidates:
                    content_list_candidates = list(mp.rglob("content_list.json"))
                if content_list_candidates:
                    try:
                        cl = _json.loads(
                            content_list_candidates[0].read_text(encoding="utf-8")
                        )
                        mrefs = extract_references_from_mineru_content_list(cl)
                        all_refs.extend(mrefs)
                        source_counts["mineru"] += len(mrefs)
                    except Exception:
                        pass

                # Source C: Marker markdown bibliography (legacy)
                md_files = list(mp.rglob("*.md"))
                if md_files:
                    try:
                        md_text = md_files[0].read_text(encoding="utf-8", errors="replace")
                        mdrefs = extract_references_from_markdown(md_text)
                        all_refs.extend(mdrefs)
                        source_counts["markdown"] += len(mdrefs)
                    except Exception:
                        pass

        local_count_added = len(all_refs) - local_count_before
        # Queue for OpenAlex augmentation if we got a weak local signal.
        # Threshold of 10 is conservative — most real papers have 20-80 refs.
        if doi and local_count_added < 10:
            needs_openalex.append(doi)

    # Second pass: OpenAlex referenced_works for low-yield papers (parallel)
    if needs_openalex:
        console.print(
            f"Querying OpenAlex referenced_works for "
            f"[bold]{len(needs_openalex)}[/bold] papers with weak local ref signal…"
        )
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
        oa_workers = max(1, settings.enrich_workers)
        with ThreadPoolExecutor(max_workers=oa_workers) as pool:
            futures = {
                pool.submit(fetch_openalex_references, d, settings.crossref_email): d
                for d in needs_openalex
            }
            for fut in _as_completed(futures):
                try:
                    oa_refs = fut.result()
                except Exception:
                    continue
                all_refs.extend(oa_refs)
                source_counts["openalex"] += len(oa_refs)

    console.print(
        f"Extracted [bold]{len(all_refs)}[/bold] raw reference entries "
        f"(crossref={source_counts['crossref']}, "
        f"mineru={source_counts['mineru']}, "
        f"markdown={source_counts['markdown']}, "
        f"openalex={source_counts['openalex']})."
    )

    # ── Step 3: deduplicate references against each other and the collection ─
    # Phase 54.6.45 — progress announcement. On a corpus with 100k+ refs
    # this pass can take ~20s, which feels frozen without signal.
    console.print(f"[dim]Deduplicating {len(all_refs)} references "
                  f"against existing corpus…[/dim]")
    seen: set[str] = set()
    candidates = []
    skipped_by_title = 0
    for ref in all_refs:
        key = (ref.doi or "").lower() or (ref.arxiv_id or "").lower()
        if not key and not ref.title:
            continue
        # Already in collection?
        if ref.doi and ref.doi.lower() in existing_dois:
            continue
        if ref.arxiv_id and ref.arxiv_id.lower() in existing_arxivs:
            continue
        # Phase 49.1 — title-normalised check. Catches the same paper
        # under a different identifier (preprint DOI vs published,
        # arXiv vs Crossref). Conservative: exact match on the
        # normalised form, no fuzzy matching.
        if ref.title:
            tnorm = _normalise_title_for_dedup(ref.title)
            if tnorm and tnorm in existing_titles_norm:
                skipped_by_title += 1
                continue
        # Deduplicate within this batch
        dedup_key = key or ref.title.lower()[:60]
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        candidates.append(ref)
    if skipped_by_title:
        console.print(
            f"[dim]Skipped {skipped_by_title} reference(s) already in the corpus "
            f"under a different identifier (title-dedup).[/dim]"
        )

    console.print(f"New references not yet in collection: [bold]{len(candidates)}[/bold]")

    if not candidates:
        console.print("[green]Collection is already up to date.[/green]")
        raise typer.Exit(0)

    # ── Step 4: filter to refs that have at least a DOI or arXiv ID ──────────
    downloadable = [r for r in candidates if r.doi or r.arxiv_id]
    console.print(
        f"Downloadable (have DOI or arXiv ID): [bold]{len(downloadable)}[/bold]"
    )

    # ── Step 4b: semantic relevance filter (optional) ─────────────────────────
    # Embed candidate titles with bge-m3 and drop those that score below the
    # configured threshold against the chosen anchor (either a user query or
    # the corpus centroid). This prevents expand from dragging in unrelated
    # papers when a seed paper cites cross-disciplinary methods.
    if relevance and downloadable:
        try:
            from sciknow.retrieval.relevance import (
                compute_corpus_centroid,
                embed_query,
                score_candidates,
                score_histogram,
            )

            eff_threshold = (
                relevance_threshold if relevance_threshold > 0
                else settings.expand_relevance_threshold
            )

            if relevance_query:
                anchor_desc = f'query "{relevance_query[:60]}"'
                anchor_vec = embed_query(relevance_query)
            else:
                anchor_desc = "corpus centroid"
                anchor_vec = compute_corpus_centroid()

            if anchor_vec is None:
                console.print(
                    "[yellow]⚠ Relevance filter: no anchor available "
                    "(abstracts collection empty?). Skipping filter.[/yellow]"
                )
            else:
                titles_for_scoring = [
                    (r.title or r.raw_text or "")[:300] for r in downloadable
                ]
                console.print(
                    f"Scoring [bold]{len(downloadable)}[/bold] candidates "
                    f"against {anchor_desc} (threshold={eff_threshold:.2f})…"
                )
                scores = score_candidates(titles_for_scoring, anchor_vec)
                for ref, s in zip(downloadable, scores):
                    ref._relevance_score = s  # transient attribute; not persisted

                kept = [r for r in downloadable if r._relevance_score >= eff_threshold]
                dropped = len(downloadable) - len(kept)

                hist = score_histogram(scores, bins=10)
                if hist:
                    console.print("[dim]Relevance score distribution:[/dim]")
                    max_count = max(c for _, _, c in hist) or 1
                    for lo, hi, c in hist:
                        bar_width = int(40 * c / max_count)
                        marker = "  " if hi < eff_threshold else "▶ " if lo <= eff_threshold < hi else "  "
                        console.print(
                            f"  {marker}[dim]{lo:.2f}-{hi:.2f}[/dim] "
                            f"{'█' * bar_width} {c}"
                        )
                    console.print(
                        f"  [green]kept {len(kept)}[/green]  "
                        f"[red]dropped {dropped}[/red]  (cut at {eff_threshold:.2f})"
                    )

                downloadable = sorted(
                    kept, key=lambda r: r._relevance_score, reverse=True
                )
        except Exception as exc:
            # Most common failure mode: CUDA OOM because another model
            # (Ollama-held LLM, or a concurrent ingest) is occupying VRAM.
            # Degrade gracefully — skip the filter rather than fail the
            # whole command, since expand without a filter is still useful.
            msg = str(exc)[:160]
            console.print(
                "[yellow]⚠ Relevance filter failed, continuing without it.[/yellow]\n"
                f"  [dim]{type(exc).__name__}: {msg}[/dim]\n"
                "  [dim]Common cause: GPU OOM. Free VRAM with `ollama stop <model>` "
                "and re-run, or use [bold]--no-relevance[/bold] to skip explicitly.[/dim]"
            )

    # ── Phase 49: RRF-fused multi-signal ranker ──────────────────────────────
    # When --strategy rrf (default), run the full ranker over the candidate
    # pool: fetch OpenAlex metadata, apply hard filters, compute co-citation
    # / bib coupling / PageRank / influential-cite / author-overlap signals,
    # fuse via RRF, and cut to `budget`. Replaces `downloadable` with the
    # top-ranked survivors so the existing download flow below processes
    # exactly those. Dry-run mode writes the full shortlist TSV and exits.
    # See docs/EXPAND_RESEARCH.md for the research behind each signal.
    ranked_features: list = []
    if strategy == "rrf" and downloadable:
        downloadable, ranked_features = _run_rrf_ranker(
            downloadable=downloadable,
            papers=papers,
            existing_dois=existing_dois,
            budget=budget,
            no_openalex=rrf_no_openalex,
            no_s2=rrf_no_s2,
            dry_run=dry_run,
            shortlist_tsv=shortlist_tsv,
            download_dir=download_dir,
            console=console,
        )
        # If dry-run + RRF, the TSV has been written and we can exit early.
        if dry_run:
            raise typer.Exit(0)

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
    no_oa_cache_file   = download_dir / ".no_oa_cache"
    ingest_done_file   = download_dir / ".ingest_done"
    ingest_failed_file = download_dir / ".ingest_failed"  # Phase 49.1

    no_oa_cache: set[str] = set()
    ingest_done: set[str] = set()
    ingest_failed: set[str] = set()

    if no_oa_cache_file.exists():
        no_oa_cache = set(no_oa_cache_file.read_text().splitlines())
    if ingest_done_file.exists():
        ingest_done = set(ingest_done_file.read_text().splitlines())
    # Phase 49.1 — persistent failure memory. A previously-failed ingest
    # usually fails again for intrinsic reasons (bad PDF, image-only
    # scan, MinerU timeout). Skip unless the user explicitly retries.
    if ingest_failed_file.exists() and not retry_failed:
        ingest_failed = set(ingest_failed_file.read_text().splitlines())
        if ingest_failed:
            console.print(
                f"[dim]{len(ingest_failed)} ref(s) cached as previously failed — "
                f"will skip. Pass --retry-failed to force a retry.[/dim]"
            )

    # Phase 49.2 — also skip anything the main ingest pipeline already
    # tried and failed on (status='failed' rows in `documents`). The
    # pipeline copies/symlinks such PDFs into `data/failed/` via
    # `_archive_pdf`, which is exactly the "duplicates in the failed
    # folder" the user was seeing: they're the canonical record of a
    # prior failed attempt. Look them up by DOI/arXiv id so that
    # db expand doesn't re-download a paper the pipeline has already
    # chewed on. `--retry-failed` still bypasses this branch.
    if not retry_failed:
        with get_session() as session:
            prior_failed = session.execute(sql_text("""
                SELECT pm.doi, pm.arxiv_id
                FROM paper_metadata pm
                JOIN documents d ON d.id = pm.document_id
                WHERE d.ingestion_status = 'failed'
            """)).fetchall()
        added_from_db = 0
        for doi_val, arxiv_val in prior_failed:
            k = (doi_val or arxiv_val or "").lower().strip()
            if k and k not in ingest_failed:
                ingest_failed.add(k)
                added_from_db += 1
        if added_from_db:
            console.print(
                f"[dim]{added_from_db} additional ref(s) found in "
                f"documents table with status='failed' — will skip.[/dim]"
            )

    # ── Step 7: download phase (parallel) + ingest phase (serial, in-process) ─
    #
    # Phase split (expand v2):
    #   1. All downloads run in a thread pool (network I/O bound).
    #   2. After all downloads settle, newly-downloaded PDFs are ingested
    #      serially IN THE SAME PROCESS via pipeline.ingest(). This keeps
    #      Marker/MinerU + bge-m3 models loaded once across the whole batch
    #      instead of paying ~15-20s of per-file subprocess startup.
    #
    # Sciknow's SHA-256 hash-based dedup in pipeline.ingest() makes the old
    # `.ingest_done` file redundant — we still consult it for backward-compat
    # with pre-v2 runs, but we no longer write to it.
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

    def _prep(ref):
        ref_key = (ref.doi or ref.arxiv_id or "").lower()
        safe_name = (ref.doi or ref.arxiv_id or "unknown").replace("/", "_").replace(":", "_")
        dest = download_dir / f"{safe_name}.pdf"
        title = (ref.title or "")[:80]
        return ref, ref_key, dest, title

    def _download_one(ref, ref_key, dest, title):
        """I/O-bound: runs in a worker thread. Never touches caches/logs/DB."""
        if ref_key in no_oa_cache or ref_key in ingest_failed:
            return ("cached", None)
        # Phase 49.1 — a prior run that successfully ingested this ref
        # either wrote to the legacy `.ingest_done` cache OR moved the
        # PDF into <download_dir>/processed/. Either signal means
        # "already handled; don't re-download, don't re-ingest".
        processed_copy = download_dir / _PROCESSED_SUBDIR / dest.name
        if ref_key in ingest_done or processed_copy.exists():
            return ("already_done", None)
        if dest.exists():
            return ("exists", None)
        ok, source = find_and_download(
            doi=ref.doi,
            arxiv_id=ref.arxiv_id,
            dest_path=dest,
            email=settings.crossref_email,
        )
        return ("downloaded" if ok else "no_oa", source)

    # Phase 1: parallel downloads. Collect successful PDF paths for phase 2.
    to_ingest: list[tuple[str, str, Path]] = []  # (ref_key, title, dest)

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

            # Phase 54.6.45 — durable per-event lines. Rich.Progress
            # only updates the inline description to the *last-settled*
            # future; with 6 workers in flight, long-running downloads
            # appear to stall because no line is emitted until they
            # complete. Emit one persistent line per event (via
            # `progress.console.print`, which Rich correctly stacks
            # above the live progress bar) so the terminal log reads
            # like a change-log the user can scroll back through.
            import time as _time
            t_phase_start = _time.monotonic()
            done_count = 0
            total_dl = len(future_to_info)

            def _emit(mark: str, color: str, kind: str, label_: str, note: str = "") -> None:
                t_elapsed = _time.monotonic() - t_phase_start
                progress.console.print(
                    f"[dim][{done_count:>4d}/{total_dl}][/dim]  "
                    f"[{color}]{mark} {kind:<7}[/{color}] {label_[:70]:<70}"
                    + (f"  [dim]· {note}[/dim]" if note else "")
                )

            for fut in as_completed(future_to_info):
                ref, ref_key, dest, title = future_to_info[fut]
                label = (ref.title or ref.doi or ref.arxiv_id or "")[:70]
                progress.update(task, description=f"[dim]{label[:50]}[/dim]")
                done_count += 1

                try:
                    status, source = fut.result()
                except Exception as exc:
                    failed_dl += 1
                    _emit("✗", "red", "ERROR", label, str(exc)[:50])
                    _log(f"ERROR  {ref_key}  | {title}  | {exc}")
                    progress.advance(task)
                    continue

                if status == "cached":
                    # ingest_failed / no_oa_cache → count as non-download but
                    # not a new failure (the prior decision stands).
                    skipped += 1
                    reason = (
                        "ingest previously failed"
                        if ref_key in ingest_failed
                        else "no OA PDF"
                    )
                    _emit("⏭", "dim", "SKIP", label, f"cached: {reason}")
                    _log(f"SKIP   {ref_key}  | {title}  ({reason}, cached)")
                    progress.advance(task)
                    continue

                if status == "already_done":
                    # Phase 49.1 — prior successful ingest (either via legacy
                    # .ingest_done cache or because its PDF lives in the
                    # processed/ subfolder).
                    skipped += 1
                    _emit("⏭", "dim", "SKIP", label, "already in corpus")
                    _log(f"SKIP   {ref_key}  | {title}  (already in corpus)")
                    progress.advance(task)
                    continue

                if status == "no_oa":
                    failed_dl += 1
                    with no_oa_cache_file.open("a") as f:
                        f.write(ref_key + "\n")
                    no_oa_cache.add(ref_key)
                    # Phase 54.6.7 — also save a human-actionable row
                    # so the user can retry / manually acquire later.
                    try:
                        from sciknow.core.pending_ops import record_failure
                        record_failure(
                            doi=ref.doi or "", title=ref.title or "",
                            authors=list(ref.authors or []),
                            year=ref.year, arxiv_id=ref.arxiv_id,
                            source_method="expand",
                            reason="no_oa",
                        )
                    except Exception:
                        pass
                    _emit("✗", "yellow", "NO_OA", label, "no open-access PDF found")
                    _log(f"NO_OA  {ref_key}  | {title}")
                    progress.advance(task)
                    continue

                if status == "downloaded":
                    downloaded += 1
                    progress.update(task, description=f"[green]↓ {source}[/green] {label[:40]}")
                    _emit("✓", "green", "DL", label, f"source={source}")
                    _log(f"DL     {ref_key}  | {title}  | source={source}")
                    if ingest:
                        to_ingest.append((ref_key, title, dest))

                elif status == "exists":
                    # PDF already on disk from a prior (interrupted) run.
                    # Queue for ingestion anyway — pipeline.ingest() will
                    # hash-dedupe against the DB.
                    if ingest:
                        _emit("↺", "cyan", "EXISTS", label, "pdf on disk, queued for ingest")
                        to_ingest.append((ref_key, title, dest))
                    else:
                        skipped += 1
                        _emit("⏭", "dim", "SKIP", label, "pdf on disk, --no-ingest")
                        _log(f"SKIP   {ref_key}  | {title}  (pdf on disk, --no-ingest)")

                progress.advance(task)

    # Phase 2: parallel ingestion of everything that downloaded cleanly.
    # Each worker subprocess loads its own MinerU + bge-m3 once and processes
    # its bucket of PDFs. The main process's bge-m3 (loaded for the relevance
    # filter, if it ran) is released first so we don't keep a redundant copy
    # alongside the worker copies.
    if ingest and to_ingest:
        from sciknow.cli.ingest import _run_parallel_workers
        from sciknow.ingestion.embedder import release_model as _release_embedder

        # Resolve worker count: CLI flag wins, else INGEST_WORKERS from .env.
        ingest_workers = workers if workers > 0 else max(1, settings.ingest_workers)
        ingest_workers = min(ingest_workers, len(to_ingest))

        # Free the main-process bge-m3 so worker subprocesses can load theirs
        # without fighting the main process for VRAM.
        _release_embedder()

        # Build a path → (ref_key, title) lookup so the per-file callback can
        # write expand.log entries with the metadata workers don't know about.
        path_to_meta: dict[Path, tuple[str, str]] = {
            dest.resolve(): (ref_key, title) for ref_key, title, dest in to_ingest
        }

        # Phase 54.6.45 — progress.console.print captured below by the
        # closure so ingestion callbacks can emit durable lines too.
        _ingest_progress_ref: list = [None]  # set just before Progress ctx

        def _on_file_done(path, status, error):
            # Counters and log lines mutate nonlocal state; rich.Progress
            # handles its own threading so no extra lock is needed here.
            nonlocal ingested, failed_ingest
            ref_key, title = path_to_meta.get(path.resolve(), ("?", path.name))
            label = (title or path.name)[:70]
            prog = _ingest_progress_ref[0]
            def _say(mark, color, kind, note=""):
                if prog is None:
                    return
                prog.console.print(
                    f"  [{color}]{mark} {kind:<11}[/{color}] {label[:70]:<70}"
                    + (f"  [dim]· {note}[/dim]" if note else "")
                )
            if status == "done":
                ingested += 1
                _say("✓", "green", "INGEST")
                _log(f"INGEST {ref_key}  | {title}")
            elif status == "skipped":
                # Already in DB via SHA-256 match — count as success for UX.
                ingested += 1
                _say("⏭", "dim", "INGEST-SKIP", "already in DB")
                _log(f"INGEST {ref_key}  | {title}  (already in DB)")
            elif status == "failed":
                failed_ingest += 1
                _say("✗", "red", "INGEST-FAIL", (error or "")[:50])
                _log(f"INGEST_FAIL {ref_key}  | {title}  | {error or ''}")
                # Phase 49.1 — persist the failure so the next run
                # skips this ref by default. User can force a retry
                # with `--retry-failed`.
                try:
                    with ingest_failed_file.open("a") as _ff:
                        _ff.write(ref_key + "\n")
                    ingest_failed.add(ref_key)
                except Exception:
                    pass
            # Phase 49.1 — move the PDF into a processed/ or
            # failed_ingest/ subfolder so <download_dir> stays tidy.
            # The user can `--no-cleanup` to keep the old behaviour
            # (everything at the root of download_dir).
            if cleanup:
                _move_downloaded_pdf(
                    path, status, download_dir, error_msg=error or ""
                )

        worker_note = f" ({ingest_workers} workers)" if ingest_workers > 1 else ""
        console.print(
            f"\nIngesting [bold]{len(to_ingest)}[/bold] new PDF(s){worker_note}…"
        )

        ingest_results = {"done": 0, "skipped": 0, "failed": 0}
        ingest_failed_files: list[tuple[str, str]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            _ingest_progress_ref[0] = progress
            itask = progress.add_task("Ingesting", total=len(to_ingest))
            _run_parallel_workers(
                [dest for _, _, dest in to_ingest],
                progress, itask,
                ingest_results, ingest_failed_files,
                force=False,
                num_workers=ingest_workers,
                ingest_source="expand",
                on_file_done=_on_file_done,
            )

    # Phase 54.6.117 (Tier 4 #1) — write provenance records for every
    # paper that made it through download + ingest. ``ranked_features``
    # carries the RRF signal values per candidate; we look up by DOI
    # (or arxiv_id) against the documents table to attach them. Safe
    # to call repeatedly — record() merges with existing provenance.
    if ingested and ranked_features:
        try:
            from sciknow.core import provenance as _prov
            feats_by_doi = {(f.doi or "").lower(): f
                             for f in ranked_features if f.doi}
            feats_by_arx = {(f.arxiv_id or "").lower(): f
                             for f in ranked_features if f.arxiv_id}
            written = 0
            for f in ranked_features:
                if f.hard_drop_reason:
                    continue
                key = (f.doi or "").lower()
                if not key:
                    key = (f.arxiv_id or "").lower()
                if not key:
                    continue
                signals = {
                    "rrf_score": f.rrf_score,
                    "bge_m3_cosine": f.bge_m3_cosine,
                    "citation_context_cosine": f.citation_context_cosine,
                    "citation_context_n": f.citation_context_n,
                    "co_citation": f.co_citation,
                    "bib_coupling": f.bib_coupling,
                    "pagerank": f.pagerank,
                    "influential_cites": f.influential_cite_count,
                    "cited_by": f.cited_by_count,
                    "velocity": f.citation_velocity,
                    "author_overlap": f.author_overlap,
                    "venue": f.venue,
                    "doc_type": f.doc_type,
                }
                ok = _prov.record(
                    doi=f.doi or None,
                    source="expand-agentic" if False else "expand",
                    relevance_query=relevance_query,
                    signals=signals,
                )
                if ok:
                    written += 1
            if written:
                _log(f"PROVENANCE recorded signals for {written} ingested paper(s)")
        except Exception as exc:  # noqa: BLE001
            logger.debug("provenance write skipped: %s", exc)

    # Phase 54.6.118 (Tier 4 #3) — corpus-drift snapshot after the
    # new papers landed. Cheap (one centroid read); surfaces whether
    # the expansion pulled the corpus toward a new subtopic.
    if ingested:
        try:
            from sciknow.core.project import get_active_project
            from sciknow.retrieval.corpus_drift import record_drift
            proj = get_active_project()
            summary = record_drift(
                proj.root,
                tag="expand",
                reason=(relevance_query or "centroid")[:80],
                also_size=ingested,
            )
            if summary.get("drift_delta") is not None:
                d = summary["drift_delta"]
                colour = "dim" if d < 0.01 else ("yellow" if d < 0.05 else "red")
                console.print(
                    f"[{colour}]corpus drift: delta={d:.4f} "
                    f"(cosine={summary['drift_cosine']:.4f})[/{colour}]"
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("drift recording skipped: %s", exc)

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

    # Phase 44.1 — auto-run the citation linker at the end of expand.
    # Citations referencing newly-ingested papers don't get retroactively
    # linked by per-paper ingestion (ingest only links citations FROM the
    # paper being ingested, not citations TO it). Running the full-table
    # linker after a bulk expand closes that gap; the bench baseline found
    # 4.5% cross-link rate and a retroactive run bumped it to 6.1%.
    if ingested:
        try:
            from sqlalchemy import text as _sql_text
            from sciknow.storage.db import get_session
            with get_session() as session:
                corpus = session.execute(_sql_text("""
                    SELECT pm.doi, d.id FROM paper_metadata pm
                    JOIN documents d ON d.id = pm.document_id
                    WHERE pm.doi IS NOT NULL AND d.ingestion_status = 'complete'
                """)).fetchall()
                doi_to_doc = {(r[0] or "").lower().strip(): r[1] for r in corpus}
                unlinked = session.execute(_sql_text("""
                    SELECT id, cited_doi FROM citations
                    WHERE cited_doi IS NOT NULL AND cited_document_id IS NULL
                """)).fetchall()
                new_links = 0
                for cit_id, doi in unlinked:
                    did = doi_to_doc.get((doi or "").lower().strip())
                    if did:
                        session.execute(
                            _sql_text("UPDATE citations SET cited_document_id = :d WHERE id = :c"),
                            {"d": did, "c": cit_id},
                        )
                        new_links += 1
                session.commit()
            if new_links:
                console.print(f"[dim]✓ Backfilled {new_links} citation cross-links.[/dim]")
        except Exception as exc:
            console.print(f"[dim]citation link-backfill skipped: {exc}[/dim]")


@app.command(name="cleanup-downloads")
def cleanup_downloads(
    download_dir: Path = typer.Option(None, "--download-dir", "-d",
                                       help="Override for <download_dir> (default: <project data>/downloads)."),
    dry_run: bool = typer.Option(False, "--dry-run",
                                  help="Show what would change, don't touch any files."),
    delete_dupes: bool = typer.Option(False, "--delete-dupes",
                                       help="DELETE duplicate / already-in-DB PDFs instead of moving them "
                                            "to processed/. Saves disk but loses the audit trail — only "
                                            "use if you're sure."),
    cross_project: bool = typer.Option(True, "--cross-project/--no-cross-project",
                                        help="Also query OTHER projects' DBs for ingested file_hashes. "
                                             "Default ON — catches the common case where an expand run in "
                                             "project B re-downloaded papers already ingested in project A. "
                                             "Phase 54.6.4."),
    clean_failed: bool = typer.Option(False, "--clean-failed/--no-clean-failed",
                                       help="ALSO permanently remove failed-ingest PDFs (data/failed/ and "
                                            "downloads/failed_ingest/) plus the corresponding `documents` rows "
                                            "with ingestion_status='failed'. These are PDFs the pipeline gave "
                                            "up on; keeping them just wastes disk. Phase 54.6.19."),
):
    """Phase 49.2 + 54.6.4 — comprehensive dedup of every place a PDF can end up.

    Scans ALL of these locations for PDFs:

    \\b
      * <download_dir>/                      (loose files from interrupted expand runs)
      * <download_dir>/processed/            (Phase 49.1 success archive)
      * <download_dir>/failed_ingest/        (Phase 49.1 ingest-failed archive)
      * <data_dir>/processed/                (main pipeline success archive)
      * <data_dir>/failed/                   (main pipeline failure archive)

    For each PDF, SHA-256 its bytes. Group by hash and pick a
    canonical location (priority: data/processed > downloads/processed
    > downloads root > data/failed > downloads/failed_ingest). Every
    other copy is a duplicate and gets moved to the canonical
    subfolder (or deleted with --delete-dupes).

    With ``--cross-project`` (default ON), the command also queries every
    OTHER sciknow project's DB for ``documents.file_hash`` — so a PDF
    downloaded into project B that's already ingested in project A is
    recognised as a dupe and cleaned up. This is the common case when
    you run expand in a fresh project that overlaps with an existing
    one's corpus (the 82-PDF situation from Phase 54.6.2 repro).

    Why this matters: the main ingest pipeline's `_archive_pdf`
    already moves downloads/*.pdf into data/{processed,failed}/ on
    its own, so a successful db expand ingest leaves TWO archives —
    one via the pipeline, one via Phase 49.1's `_move_downloaded_pdf`
    — if we don't cross-reference them. This command is the unified
    cross-reference."""
    from sciknow.cli import preflight
    preflight()
    import hashlib
    from collections import defaultdict
    from sqlalchemy import create_engine, text as sql_text
    from sciknow.config import settings
    from sciknow.storage.db import get_session
    from sciknow.core.project import get_active_project, list_projects

    if download_dir is None:
        download_dir = settings.data_dir / "downloads"
    data_dir = settings.data_dir

    # Scan locations in canonical-preference order — the first one a
    # given SHA appears in is "kept", later ones are dupes.
    scan_locations: list[tuple[str, Path]] = [
        ("data/processed",            data_dir / "processed"),
        ("downloads/processed",       download_dir / _PROCESSED_SUBDIR),
        ("downloads (root)",          download_dir),
        ("data/failed",               data_dir / "failed"),
        ("downloads/failed_ingest",   download_dir / _FAILED_SUBDIR),
    ]

    def _pdfs_in(p: Path) -> list[Path]:
        if not p.exists():
            return []
        # Non-recursive — we scan each location individually.
        return sorted(
            f for f in p.iterdir()
            if f.is_file() and not f.is_symlink() and f.suffix.lower() == ".pdf"
        )

    found: list[tuple[str, Path]] = []
    for label, loc in scan_locations:
        for pdf in _pdfs_in(loc):
            found.append((label, pdf))
    if not found and not clean_failed:
        console.print("[green]No PDF files found across any archive location.[/green]")
        raise typer.Exit(0)
    if found:
        console.print(f"Scanning [bold]{len(found)}[/bold] PDF(s) across all archive locations…")
    else:
        # Phase 54.6.21 — even with no loose PDFs we still want to
        # purge `documents` rows in 'failed' status when --clean-failed
        # is set. The dedup pass becomes a no-op (empty `by_sha`) and
        # the function falls through naturally to the failed-cleanup
        # block. Just signal the user so the empty dedup summary
        # doesn't look broken.
        console.print("[dim]No PDF files in archive dirs — skipping dedup, "
                      "proceeding to documents-row purge.[/dim]")

    # Build SHA → document_row index for the corpus so we can cross-
    # reference disk content against DB state too. When --cross-project
    # is on (default), enumerate every sciknow project's DB and union
    # their file_hash columns — a PDF ingested anywhere counts.
    #
    # NOTE: we connect EXPLICITLY to active.pg_database rather than
    # relying on get_session(), because settings.pg_database is read
    # from .env at import time and may not reflect the active project
    # when the user has switched via .active-project. Direct SQL on
    # the resolved name is the only correct thing to do here.
    sha_status: dict[str, str] = {}
    sha_project: dict[str, str] = {}  # which project has it (for verbose reporting)
    active = get_active_project()
    try:
        eng_active = create_engine(
            f"postgresql://{settings.pg_user}:{settings.pg_password}"
            f"@{settings.pg_host}:{settings.pg_port}/{active.pg_database}"
        )
        with eng_active.connect() as conn:
            rows = conn.execute(sql_text(
                "SELECT file_hash, ingestion_status FROM documents "
                "WHERE file_hash IS NOT NULL"
            )).fetchall()
        for h, st in rows:
            if h:
                sha_status[h] = st
                sha_project[h] = active.slug
    except Exception as exc:
        console.print(
            f"[yellow]warn:[/yellow] could not read active project DB "
            f"({active.pg_database}): {exc}. Continuing with cross-project "
            f"lookup only."
        )

    if cross_project:
        # list_projects() only returns projects under projects/<slug>/. The
        # legacy `default` project lives at the repo root so it's NOT in
        # that list — add it manually if the active project isn't default.
        from sciknow.core.project import Project as _Project
        other_projects = [p for p in list_projects() if p.slug != active.slug]
        if not active.is_default and not any(p.is_default for p in other_projects):
            other_projects.append(_Project.default())
        for proj in other_projects:
            try:
                eng = create_engine(
                    f"postgresql://{settings.pg_user}:{settings.pg_password}"
                    f"@{settings.pg_host}:{settings.pg_port}/{proj.pg_database}"
                )
                with eng.connect() as conn:
                    prows = conn.execute(sql_text(
                        "SELECT file_hash, ingestion_status FROM documents "
                        "WHERE file_hash IS NOT NULL"
                    )).fetchall()
                    for h, st in prows:
                        if not h:
                            continue
                        # Only promote to sha_status if the cross-project hit is
                        # 'complete' — a foreign 'failed' shouldn't mask a local
                        # 'pending'. The local project's own status wins ties.
                        if h not in sha_status or (st == "complete"
                                                   and sha_status.get(h) != "complete"):
                            sha_status[h] = st
                            sha_project[h] = proj.slug
            except Exception as exc:
                console.print(f"  [dim]skip {proj.slug} DB: {exc}[/dim]")
        if other_projects:
            console.print(
                f"[dim]Cross-referenced {len(other_projects)} other project DB(s): "
                f"{', '.join(p.slug for p in other_projects)}[/dim]"
            )

    # Phase 54.6.58 — unified per-item log format, mirroring the
    # expand 54.6.45 / enrich 54.6.57 [N/M] KIND title · note pattern.
    # Gives the GUI log pane a scrollable event log per dupe action +
    # per failed-nuke file instead of free-form console.print spam.
    def _emit(done: int, total: int, mark: str, color: str,
              kind: str, label_: str, note: str = "") -> None:
        console.print(
            f"[dim][{done:>4d}/{total}][/dim]  "
            f"[{color}]{mark} {kind:<9}[/{color}] {label_[:70]:<70}"
            + (f"  [dim]· {note}[/dim]" if note else "")
        )

    # Hash every file. Group by SHA.
    by_sha: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    hash_failures: list[tuple[Path, str]] = []
    for label, pdf in found:
        try:
            h = hashlib.sha256(pdf.read_bytes()).hexdigest()
        except Exception as exc:
            hash_failures.append((pdf, str(exc)[:80]))
            continue
        by_sha[h].append((label, pdf))

    # Announce hash failures as durable lines if any (rare, but worth
    # flagging — the file stays on disk and gets skipped by dedup).
    for i, (pdf, err) in enumerate(hash_failures, 1):
        _emit(i, len(hash_failures), "⚠", "red", "HASH_FAIL",
              pdf.name, err)

    # Pre-compute the total number of dedup events we'll emit (one per
    # duplicate file, canonical files don't emit). Drives the [N/M]
    # counter so the GUI progress feel matches enrich + expand.
    total_dupes = sum(max(0, len(l) - 1) for l in by_sha.values())
    done_dupes = 0

    moved = deleted = kept = 0
    archived_orphans = 0
    foreign_ingested = 0  # 54.6.4: hits that are only ingested in OTHER projects
    for sha, locs in by_sha.items():
        # Canonical = first in scan_locations order that appears
        locs_sorted = sorted(locs, key=lambda x: [i for i, (_, p) in enumerate(scan_locations) if p == x[1].parent][0] if any(p == x[1].parent for _, p in scan_locations) else 99)
        canonical_label, canonical_path = locs_sorted[0]
        dupes = locs_sorted[1:]
        # Phase 54.6.4 — if the SHA is already ingested in ANOTHER project
        # (and not in the local one's completed set), copies sitting in
        # THIS project's downloads area are redundant. We DO NOT touch
        # the pipeline archives (data/processed/, data/failed/) because
        # those may still be referenced by their ingestion_status rows
        # via documents.original_path; the conservative thing is to
        # leave them alone and just clean the downloads clutter.
        cross_hit = (sha in sha_status
                     and sha_status[sha] == "complete"
                     and sha_project.get(sha) != active.slug)
        if cross_hit:
            # Only mark files under downloads/* as dupes — preserve the
            # pipeline archive copies to avoid breaking original_path.
            downloads_locs = [
                (lbl, p) for (lbl, p) in locs_sorted
                if lbl.startswith("downloads")
            ]
            archive_locs = [
                (lbl, p) for (lbl, p) in locs_sorted
                if not lbl.startswith("downloads")
            ]
            if downloads_locs:
                foreign_ingested += len(downloads_locs)
                dupes = downloads_locs[:]
                if archive_locs:
                    canonical_label, canonical_path = archive_locs[0]
                else:
                    canonical_label = f"(ingested in project '{sha_project[sha]}')"
                    canonical_path = None
            else:
                # Only archive copies — leave them alone.
                cross_hit = False
                kept += 1
                dupes = []
        elif sha in sha_status:
            if sha_status[sha] == "complete":
                # Corpus knows this paper — dupes can go.
                kept += 1
            else:
                # status='failed' or 'pending' — still dedupe but the
                # canonical may need to stay in data/failed for pipeline
                # retries. Prefer data/failed as canonical for failed docs.
                for i, (lbl, _p) in enumerate(locs_sorted):
                    if lbl == "data/failed":
                        canonical_label, canonical_path = lbl, _p
                        dupes = locs_sorted[:i] + locs_sorted[i+1:]
                        break
                kept += 1
        else:
            # Not in DB — might be genuinely new or orphaned.
            if len(dupes) == 0:
                kept += 1
                continue
            archived_orphans += 1

        for lbl, dupe_path in dupes:
            done_dupes += 1
            canon_name = (canonical_path.name if canonical_path is not None
                          else canonical_label)
            dup_note = f"dup of {canonical_label}/{canon_name}"
            if cross_hit:
                dup_note += f"  [cross-project: {sha_project.get(sha, '?')}]"

            if dry_run:
                _emit(done_dupes, total_dupes, "⊘", "cyan", "DRY_REM",
                      f"{lbl}/{dupe_path.name}", dup_note)
                moved += 1
                continue
            try:
                if delete_dupes or cross_hit:
                    # Cross-project hits are ALWAYS deleted — keeping a
                    # second archive of a paper that's already sitting in
                    # another project's data/processed/ is just wasted
                    # disk, and "moving to processed/" for a project that
                    # will never ingest the file is meaningless.
                    dupe_path.unlink()
                    deleted += 1
                    _emit(done_dupes, total_dupes, "✗", "red", "DELETED",
                          f"{lbl}/{dupe_path.name}", dup_note)
                else:
                    # Park in downloads/processed as the "safe keep"
                    # location. Preserves content; avoids cluttering
                    # pipeline archive dirs.
                    parked = _move_downloaded_pdf(
                        dupe_path, outcome="skipped", download_dir=download_dir,
                    )
                    # If the move target already has a same-named file
                    # (would clobber), just delete the dupe instead.
                    if parked is None and dupe_path.exists():
                        dupe_path.unlink()
                        deleted += 1
                        _emit(done_dupes, total_dupes, "✗", "red", "DELETED",
                              f"{lbl}/{dupe_path.name}",
                              f"{dup_note}  · clobber avoided")
                    else:
                        moved += 1
                        _emit(done_dupes, total_dupes, "↷", "yellow", "MOVED",
                              f"{lbl}/{dupe_path.name}",
                              f"{dup_note}  → downloads/processed/")
            except Exception as exc:
                _emit(done_dupes, total_dupes, "⚠", "red", "SKIP",
                      f"{lbl}/{dupe_path.name}", str(exc)[:80])

    verb = "Would remove" if dry_run else ("Deleted" if delete_dupes else "Moved to processed/")
    console.print(
        f"\n[bold]Summary:[/bold] "
        f"[green]✓ {moved + deleted} duplicate copies {verb.lower()}[/green]  "
        f"[yellow]↷ {kept} canonical copies kept[/yellow]"
        + (f"  [cyan]↷ {foreign_ingested} already-ingested-in-other-project[/cyan]" if foreign_ingested else "")
        + (f"  [dim]({archived_orphans} not-in-DB groups consolidated)[/dim]" if archived_orphans else "")
    )

    # Phase 54.6.19 — purge failed-ingest PDFs + their documents rows.
    # Runs AFTER the dedup pass so any failed file that was actually a
    # dupe of a complete ingest already got cleaned/moved correctly above.
    # What's left in failed dirs is the real "pipeline gave up" set.
    if clean_failed:
        failed_dirs = [
            ("data/failed",             data_dir / "failed"),
            ("downloads/failed_ingest", download_dir / _FAILED_SUBDIR),
        ]
        # Pre-count so [N/M] is accurate across both failed dirs.
        all_failed: list[tuple[str, Path]] = []
        for label, d in failed_dirs:
            if not d.exists():
                continue
            for pdf in sorted(d.iterdir()):
                if (pdf.is_file() and not pdf.is_symlink()
                        and pdf.suffix.lower() == ".pdf"):
                    all_failed.append((label, pdf))

        total_failed = len(all_failed)
        nuked_files = 0
        for idx, (label, pdf) in enumerate(all_failed, 1):
            if dry_run:
                _emit(idx, total_failed, "⊘", "cyan", "DRY_NUKE",
                      f"{label}/{pdf.name}", "would remove")
                nuked_files += 1
                continue
            try:
                pdf.unlink()
                nuked_files += 1
                _emit(idx, total_failed, "✗", "red", "NUKED",
                      f"{label}/{pdf.name}", "pipeline gave up")
            except Exception as exc:
                _emit(idx, total_failed, "⚠", "red", "SKIP",
                      f"{label}/{pdf.name}", str(exc)[:80])

        nuked_rows = 0
        orphan_wiki = 0
        try:
            eng_purge = create_engine(
                f"postgresql://{settings.pg_user}:{settings.pg_password}"
                f"@{settings.pg_host}:{settings.pg_port}/{active.pg_database}"
            )
            with eng_purge.connect() as conn:
                if dry_run:
                    res = conn.execute(sql_text(
                        "SELECT COUNT(*) FROM documents "
                        "WHERE ingestion_status = 'failed'"
                    )).scalar()
                    nuked_rows = int(res or 0)
                    # Phase 54.6.21 — count wiki_pages that WOULD be
                    # orphaned if we nuked the failed docs. wiki_pages
                    # has no FK on source_doc_ids (it's a plain UUID
                    # array), so the cascade-delete that takes care of
                    # paper_metadata / chunks / sections doesn't touch
                    # them — they'd be left pointing at dead UUIDs.
                    res2 = conn.execute(sql_text("""
                        SELECT COUNT(*) FROM wiki_pages wp
                        WHERE wp.page_type = 'paper_summary'
                          AND wp.source_doc_ids IS NOT NULL
                          AND NOT EXISTS (
                            SELECT 1 FROM documents d
                            WHERE d.id = ANY(wp.source_doc_ids)
                              AND d.ingestion_status <> 'failed'
                          )
                    """)).scalar()
                    orphan_wiki = int(res2 or 0)
                else:
                    res = conn.execute(sql_text(
                        "DELETE FROM documents WHERE ingestion_status = 'failed'"
                    ))
                    nuked_rows = res.rowcount or 0
                    # Same cleanup, post-delete: any wiki_page whose
                    # source_doc_ids no longer matches a live document
                    # is now orphaned and should go.
                    res2 = conn.execute(sql_text("""
                        DELETE FROM wiki_pages wp
                        WHERE wp.page_type = 'paper_summary'
                          AND wp.source_doc_ids IS NOT NULL
                          AND NOT EXISTS (
                            SELECT 1 FROM documents d
                            WHERE d.id = ANY(wp.source_doc_ids)
                          )
                    """))
                    orphan_wiki = res2.rowcount or 0
                    conn.commit()
        except Exception as exc:
            console.print(
                f"  [yellow]warn:[/yellow] could not purge failed documents rows: {exc}"
            )

        nuke_verb = "Would nuke" if dry_run else "Nuked"
        wiki_part = (
            f" + {orphan_wiki} orphan wiki page(s)" if orphan_wiki else ""
        )
        console.print(
            f"[bold]Failed cleanup:[/bold] "
            f"[red]✗ {nuke_verb} {nuked_files} failed PDF(s) + "
            f"{nuked_rows} documents row(s){wiki_part}[/red]"
        )


@app.command(name="repair")
def repair_cmd(
    scan:     bool = typer.Option(False, "--scan",
                                   help="Diff PG chunks vs Qdrant points; report orphans in both directions."),
    prune:    bool = typer.Option(False, "--prune",
                                   help="Delete orphan Qdrant points (safe; PG is the source of truth)."),
    rebuild_paper: str = typer.Option("", "--rebuild-paper",
                                       help="Re-chunk + re-embed one document by id / id-prefix."),
):
    """Phase 52 — surgical middle ground between `db stats` (read-only)
    and `db reset` (destructive). Picks up where `db stats` leaves off.

    Three operations; exactly one must be chosen:

    \\b
      --scan             audit PG vs Qdrant orphans
      --prune            delete orphan Qdrant points
      --rebuild-paper <id>  re-chunk + re-embed one paper
    """
    from sciknow.cli import preflight
    preflight()
    from sciknow.maintenance import repair as _repair

    flags = sum([scan, prune, bool(rebuild_paper)])
    if flags != 1:
        console.print(
            "[red]pass exactly one of --scan / --prune / --rebuild-paper <id>[/red]"
        )
        raise typer.Exit(2)

    if scan:
        report = _repair.repair_scan()
        console.print(
            f"[bold]Scan:[/bold] PG chunks = {report.pg_chunks_total:,}, "
            f"Qdrant points = {report.qdrant_points_total:,}"
        )
        console.print(
            f"  PG orphans (chunk row but no Qdrant point): [yellow]{len(report.pg_orphans)}[/yellow]"
        )
        console.print(
            f"  Qdrant orphans (point but no chunk row):    [yellow]{len(report.qdrant_orphans)}[/yellow]"
        )
        console.print(
            f"  Chunks on an older chunker_version:         [yellow]{report.stale_chunker_version}[/yellow]"
        )
        if report.ok():
            console.print("[green]✓ No repair needed.[/green]")
        else:
            console.print(
                "[dim]Next: `db repair --prune` to remove Qdrant orphans. "
                "For PG orphans, `db repair --rebuild-paper <doc_id>` "
                "re-embeds one document.[/dim]"
            )
        return

    if prune:
        report = _repair.repair_scan()
        if not report.qdrant_orphans:
            console.print("[green]✓ No Qdrant orphans to prune.[/green]")
            return
        console.print(
            f"Pruning [bold]{len(report.qdrant_orphans)}[/bold] orphan Qdrant points…"
        )
        n = _repair.repair_prune(report.qdrant_orphans)
        console.print(f"[green]✓ Deleted {n} orphan Qdrant points.[/green]")
        return

    # rebuild_paper
    try:
        n_chunks, n_vectors = _repair.rebuild_paper(rebuild_paper)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]✓ Rebuilt paper {rebuild_paper[:8]}…: "
        f"{n_chunks} chunks, {n_vectors} vectors.[/green]"
    )


@app.command(name="dedup")
def dedup_cmd(
    threshold: float = typer.Option(0.92, "--threshold",
                                     help="Cosine similarity above which chunks count as duplicates. "
                                          "Default 0.92 — tight enough that paraphrases don't collapse "
                                          "but near-identical copies do."),
    cross_document: bool = typer.Option(False, "--cross-document",
                                         help="Also scan across documents. Groups chunks by "
                                              "(section_type, first-60-chars-of-content) as a coarse "
                                              "pre-filter, then cosine-dedups within each bucket. "
                                              "Catches preprint-v1 / journal-version near-duplicates "
                                              "that SHA-256 at ingest can't. More work (more Qdrant "
                                              "hits) than the default within-document scan."),
    dry_run: bool = typer.Option(True, "--dry-run/--apply",
                                  help="Default is --dry-run: report what would be deleted. "
                                       "Pass --apply to actually remove duplicate chunks + vectors."),
    limit_docs: int = typer.Option(0, "--limit-docs",
                                    help="Only scan N documents (0 = all). Handy for a test pass."),
):
    """Phase 52 — chunk-level near-duplicate dedup.

    Catches the 'expand pulled preprint v1 + v2 + journal version'
    case where SHA-256 at ingest doesn't fire (files aren't byte-
    identical but share 95%+ of the text). Groups chunks, fetches
    their dense vectors from Qdrant, greedy-keeps the longest chunk
    in each cluster, deletes the rest.
    """
    from sciknow.cli import preflight
    preflight()
    from sciknow.maintenance import dedup as _dedup

    mode = "across documents" if cross_document else "within documents"
    verb = "Scanning (dry-run)" if dry_run else "Applying dedup"
    console.print(
        f"[bold]{verb}[/bold]  threshold={threshold}  mode={mode}"
    )
    report = _dedup.dedup_corpus(
        threshold=threshold,
        cross_document=cross_document,
        dry_run=dry_run,
        limit_docs=limit_docs,
    )
    console.print(
        f"[bold]Summary:[/bold]  groups={report.groups_seen:,}  "
        f"chunks_scanned={report.chunks_scanned:,}  "
        f"[yellow]duplicates={report.duplicates_found:,}[/yellow]  "
        + (f"[green]deleted={report.chunks_deleted:,}[/green]"
           if not dry_run else "[dim](dry-run — no deletes)[/dim]")
    )
    if dry_run and report.duplicates_found:
        console.print(
            "[dim]Pass --apply to actually delete. "
            "Run --dry-run first with --cross-document if you want to "
            "find arXiv/journal near-duplicates too.[/dim]"
        )


@app.command(name="link-citations")
def link_citations(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be linked without making changes."),
):
    """
    Cross-link the citations table so that cited_document_id is set whenever
    the cited paper is already in the corpus.

    Useful after a bulk ingest or expand: any citation whose cited_doi matches
    a corpus paper's DOI gets its cited_document_id pointer filled in. Also
    prints a summary of citation counts (how many times each paper is cited
    by other papers in the collection).

    This is the same cross-linking that pipeline.ingest() does per-paper, but
    applied in a single pass over the whole citations table — catches any gaps
    from papers ingested before the citation feature was added.
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    from sqlalchemy import text as sql_text
    from sciknow.storage.db import get_session

    with get_session() as session:
        # Build a lowercase-DOI → document_id map for all complete papers
        rows = session.execute(sql_text("""
            SELECT pm.doi, d.id
            FROM paper_metadata pm
            JOIN documents d ON d.id = pm.document_id
            WHERE pm.doi IS NOT NULL AND d.ingestion_status = 'complete'
        """)).fetchall()
        doi_to_doc = {(r[0] or "").lower().strip(): r[1] for r in rows}

        # Find unlinked citations whose cited_doi matches a corpus paper
        unlinked = session.execute(sql_text("""
            SELECT c.id, c.cited_doi
            FROM citations c
            WHERE c.cited_doi IS NOT NULL AND c.cited_document_id IS NULL
        """)).fetchall()

        linked = 0
        for cit_id, cited_doi in unlinked:
            doc_id = doi_to_doc.get((cited_doi or "").lower().strip())
            if doc_id:
                if not dry_run:
                    session.execute(
                        sql_text("UPDATE citations SET cited_document_id = :doc_id WHERE id = :cit_id"),
                        {"doc_id": doc_id, "cit_id": cit_id},
                    )
                linked += 1

        if not dry_run:
            session.commit()

        # Stats
        total_citations = session.execute(sql_text("SELECT COUNT(*) FROM citations")).scalar()
        total_linked = session.execute(
            sql_text("SELECT COUNT(*) FROM citations WHERE cited_document_id IS NOT NULL")
        ).scalar()

    action = "Would link" if dry_run else "Linked"
    console.print(
        f"[green]✓ {action} {linked} citations[/green] "
        f"(total: {total_citations}, cross-linked: {total_linked})"
    )

    if total_linked:
        # Show top-cited papers
        with get_session() as session:
            top = session.execute(sql_text("""
                SELECT pm.title, pm.doi, COUNT(*) AS cite_count
                FROM citations c
                JOIN paper_metadata pm ON pm.document_id = c.cited_document_id
                WHERE c.cited_document_id IS NOT NULL
                GROUP BY pm.title, pm.doi
                ORDER BY cite_count DESC
                LIMIT 15
            """)).fetchall()

        if top:
            table = Table(title="Most-Cited Papers in the Collection", box=box.SIMPLE_HEAD)
            table.add_column("Title", ratio=3)
            table.add_column("DOI", ratio=1, style="dim")
            table.add_column("Cited by", justify="right", style="cyan")
            for title, doi, count in top:
                table.add_row(
                    (title or "")[:70],
                    (doi or "")[:30],
                    str(count),
                )
            console.print(table)


@app.command(name="reclassify-sections")
def reclassify_sections(
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print what would change, don't write.",
    ),
):
    """Re-run the heading classifier on existing sections + chunks.

    Phase 44.1 — the ``_SECTION_PATTERNS`` in ``sciknow.ingestion.chunker``
    were broadened after bench findings showed very low hit rates on
    ``related_work`` (0.2%), ``results`` (24%), and ``abstract`` (37%).
    This command retroactively applies the new patterns to already-
    ingested papers so the corpus benefits without re-running the full
    MinerU pipeline.

    Updates:
      - ``paper_sections.section_type``  (used for ingest-time chunking params)
      - ``chunks.section_type``          (used for Qdrant filtering + retrieval)

    Idempotent — re-running against a corpus already classified with
    the current patterns is a no-op.
    """
    from collections import Counter
    from sqlalchemy import text as _text

    from sciknow.cli import preflight
    from sciknow.ingestion.chunker import _classify_heading
    from sciknow.storage.db import get_session

    preflight()

    with get_session() as session:
        rows = session.execute(_text("""
            SELECT id::text, section_title, section_type
            FROM paper_sections
            WHERE section_title IS NOT NULL
        """)).fetchall()
        if not rows:
            console.print("[yellow]No paper_sections rows — nothing to reclassify.[/yellow]")
            return

        transitions: Counter = Counter()
        to_update: list[tuple[str, str]] = []   # (section_id, new_type)
        for sid, title, old_type in rows:
            new_type = _classify_heading(title or "")
            if new_type != (old_type or "unknown"):
                transitions[f"{old_type or 'null'} → {new_type}"] += 1
                to_update.append((sid, new_type))

        console.print(f"[bold]Scanned[/bold] {len(rows):,} sections")
        if not to_update:
            console.print("[green]✓ No changes needed — classifier output matches stored types.[/green]")
            return
        console.print(f"[bold]Changes needed:[/bold] {len(to_update):,}")
        for t, n in transitions.most_common(20):
            console.print(f"  {t}: {n:,}")

        if dry_run:
            console.print("[dim](dry-run — no writes)[/dim]")
            return

        # Write paper_sections updates. Batched to keep the transaction
        # manageable on a 100k-chunk corpus.
        BATCH = 500
        for i in range(0, len(to_update), BATCH):
            batch = to_update[i:i+BATCH]
            ids_by_type: dict[str, list[str]] = {}
            for sid, nt in batch:
                ids_by_type.setdefault(nt, []).append(sid)
            for nt, ids in ids_by_type.items():
                session.execute(
                    _text("UPDATE paper_sections SET section_type = :nt "
                          "WHERE id::text = ANY(:ids)"),
                    {"nt": nt, "ids": ids},
                )
            if i % (BATCH * 10) == 0:
                session.commit()
        session.commit()
        console.print("[green]✓ paper_sections updated.[/green]")

        # Mirror into chunks via join on section_id. One UPDATE per
        # canonical type keeps the query plan simple.
        n_chunk_updates = 0
        for new_type in set(nt for _, nt in to_update):
            res = session.execute(_text("""
                UPDATE chunks c SET section_type = :nt
                FROM paper_sections ps
                WHERE c.section_id = ps.id AND ps.section_type = :nt
                  AND COALESCE(c.section_type, '') <> :nt
            """), {"nt": new_type})
            n_chunk_updates += res.rowcount or 0
        session.commit()
        console.print(f"[green]✓ chunks updated: {n_chunk_updates:,} rows.[/green]")
    console.print("[dim]Run `sciknow bench --layer fast --no-compare` to confirm new coverage percentages.[/dim]")


@app.command(name="tag-multimodal")
def tag_multimodal():
    """
    Tag chunks containing tables or equations in Qdrant.

    Scans all chunks and sets has_table=True / has_equation=True payload
    fields so they can be filtered during search (--tables / --equations).

    Run once after ingestion to enable multimodal filtering.

    Examples:

      sciknow db tag-multimodal
    """
    from sciknow.cli import preflight
    preflight(qdrant=True)

    from sciknow.retrieval.multimodal import tag_multimodal_chunks

    console.print("Scanning chunks for tables and equations...")
    result = tag_multimodal_chunks()
    console.print(
        f"[green]✓ Tagged {result['tables_tagged']} chunks with tables, "
        f"{result['equations_tagged']} with equations[/green]"
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
    from sciknow.cli import preflight
    preflight(qdrant=False)  # export reads from PostgreSQL only

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
            qa = json.loads(raw, strict=False)
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


# ── expand-author (Phase 16) ─────────────────────────────────────────────────


@app.command(name="expand-author")
def expand_author(
    name: Annotated[str, typer.Argument(help="Author name to search (display name).")],
    orcid: str = typer.Option(
        None, "--orcid",
        help="ORCID iD (preferred for common names — exact match instead of fuzzy "
             "display-name search). Format: 0000-0002-XXXX-XXXX",
    ),
    year_from: int = typer.Option(None, "--from",
        help="Inclusive earliest publication year."),
    year_to: int = typer.Option(None, "--to",
        help="Inclusive latest publication year."),
    limit: int = typer.Option(0, "--limit",
        help="Max papers to consider after dedup (0 = no limit, with safety caps in the search backends)."),
    dry_run: bool = typer.Option(False, "--dry-run",
        help="Show what would be downloaded without doing it."),
    all_matches: bool = typer.Option(False, "--all-matches",
        help="Use ALL authors with the matching surname (default: only the most-published one). "
             "Useful when 'Smith' really means every Smith and you'll filter via --relevance-query."),
    strict_author: bool = typer.Option(False, "--strict-author",
        help="Drop Crossref results entirely. Only OpenAlex's canonical-author-ID matches "
             "are kept — zero ambiguity, but smaller result set. Use when you want to be "
             "100%% sure no papers by other people with the same surname slip in."),
    relevance: bool = typer.Option(True, "--relevance/--no-relevance",
        help="Filter candidates by semantic relevance to the corpus before downloading."),
    relevance_query: str = typer.Option("", "--relevance-query", "-q",
        help="Free-text topic anchor for the relevance filter. If empty, the corpus centroid is used."),
    relevance_threshold: float = typer.Option(0.0, "--relevance-threshold",
        help="Cosine similarity threshold (0 = use EXPAND_RELEVANCE_THRESHOLD from .env, default 0.55)."),
    # Phase 43d — default resolved in body (see `expand` above).
    download_dir: Path = typer.Option(None, "--download-dir", "-d",
        help="Directory where new PDFs are saved before ingestion (default: <project data>/downloads)."),
    workers: int = typer.Option(0, "--workers", "-w",
        help="Parallel ingestion worker subprocesses (0 = use INGEST_WORKERS from .env)."),
    ingest: bool = typer.Option(True, "--ingest/--no-ingest",
        help="Ingest downloaded PDFs immediately."),
):
    """
    Expand the catalog by searching OpenAlex + Crossref for papers BY a named author.

    Different from `db expand` (which follows references in existing papers).
    Use `expand-author` when you want to add an author's full bibliography to
    your corpus regardless of whether anything you have currently cites them.

    \b
    The flow:
      1. Query OpenAlex /works for papers by the author (preferred — better
         metadata + ORCID-aware dedup across affiliations).
      2. Query Crossref /works as a fallback for anything OpenAlex missed.
      3. Merge results by DOI.
      4. Drop papers already in your corpus (by DOI).
      5. Optionally apply the relevance filter (handy for common names —
         "John Smith" + relevance against your corpus centroid filters out
         the unrelated John Smiths).
      6. Download via the existing 6-source OA discovery pipeline
         (Copernicus → arXiv → Unpaywall → OpenAlex → Europe PMC →
         Semantic Scholar).
      7. Ingest via the parallel worker pool (same as `db expand`).

    Examples:

      sciknow db expand-author "Zharkova"

      sciknow db expand-author "Zharkova" --dry-run

      sciknow db expand-author "Zharkova" --from 2015 --limit 30

      sciknow db expand-author "Zharkova" --orcid 0000-0002-0026-2725

      sciknow db expand-author "John Smith" --relevance-query "climate sensitivity"
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from sciknow.cli import preflight
    preflight()

    from sciknow.config import settings
    from sciknow.ingestion.author_search import search_author
    from sciknow.ingestion.downloader import find_and_download
    from sciknow.storage.db import get_session
    from sqlalchemy import text as sql_text

    # Phase 43d — resolve download_dir here (Typer default was None to
    # defer active-project lookup until the command actually runs).
    if download_dir is None:
        download_dir = settings.data_dir / "downloads"

    # ── Step 1: validate ─────────────────────────────────────────────────────
    if year_from is not None and year_to is not None and year_from > year_to:
        console.print(f"[red]--from {year_from} > --to {year_to}[/red]")
        raise typer.Exit(2)

    # ── Step 2: search OpenAlex + Crossref ───────────────────────────────────
    label = f"ORCID {orcid}" if orcid else f'"{name}"'
    year_range = ""
    if year_from or year_to:
        year_range = f" ({year_from or '*'}–{year_to or '*'})"

    console.print(f"\n[bold]Searching for papers by {label}{year_range}[/bold]")
    console.print("[dim]OpenAlex /works (primary) + Crossref /works (fallback)[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=None)
        try:
            candidates, info = search_author(
                name, orcid=orcid,
                year_from=year_from, year_to=year_to,
                limit=limit if limit > 0 else None,
                all_matches=all_matches,
                strict_author=strict_author,
            )
        except Exception as exc:
            console.print(f"[red]Search failed: {exc}[/red]")
            raise typer.Exit(1)
        progress.update(task, description="Done")

    # Show which author(s) OpenAlex resolved + which we picked
    if info["candidates"] and not orcid:
        picked_ids = {a["short_id"] for a in info["picked"]}
        n_picked = len(picked_ids)
        n_total = len(info["candidates"])
        if all_matches:
            console.print(
                f"\n[bold]Using all {n_total} matching author(s):[/bold]"
            )
        else:
            console.print(
                f"\n[bold]Picked top match[/bold] (out of {n_total} candidates with that surname):"
            )
        for i, a in enumerate(info["candidates"][:8]):
            mark = "[green]▶[/green]" if a["short_id"] in picked_ids else " "
            aff = " · ".join(a["affiliations"][:1]) or "(no affiliation)"
            orcid_str = (a.get("orcid") or "").replace("https://orcid.org/", "")
            orcid_short = f"  ORCID:{orcid_str}" if orcid_str else ""
            console.print(
                f"  {mark} {a['display_name']:<28} {a['works_count']:>5} works  "
                f"[dim]{aff[:45]}[/dim]{orcid_short}"
            )
        if n_total > 8:
            console.print(f"    [dim]… and {n_total - 8} more[/dim]")
        if n_picked < n_total:
            console.print(
                "\n  [dim]Wrong person? Use [bold]--orcid[/bold] for an exact match, "
                "or [bold]--all-matches[/bold] to pool all matching surnames.[/dim]"
            )

    if not candidates:
        console.print(
            "\n[yellow]No papers found. Try a different name spelling, "
            "or use --orcid for an exact match.[/yellow]"
        )
        raise typer.Exit(0)

    console.print(
        f"\nFound [bold]{info['merged']}[/bold] candidate paper(s) "
        f"([green]{info['openalex']} from OpenAlex[/green], "
        f"[cyan]{info['crossref_extra']} extra from Crossref[/cyan])"
    )
    if info.get("dropped_no_surname"):
        console.print(
            f"  [dim](dropped {info['dropped_no_surname']} where the surname "
            f"wasn't actually in the author list — defensive check)[/dim]"
        )
    if info["crossref_extra"] > info["openalex"] * 2:
        console.print(
            "  [yellow]⚠ Crossref contributed many more results than OpenAlex.[/yellow]"
            "\n    [dim]Crossref's surname search is looser than OpenAlex's canonical-author-ID match,"
            "\n    so some of these may be by different people with the same surname."
            "\n    Use [bold]--strict-author[/bold] to drop Crossref entirely (OpenAlex only).[/dim]"
        )

    # ── Step 3: dedup against existing corpus ───────────────────────────────
    with get_session() as session:
        existing = session.execute(sql_text("""
            SELECT LOWER(doi) FROM paper_metadata WHERE doi IS NOT NULL
        """)).fetchall()
        existing_dois = {r[0] for r in existing}

    before_dedup = len(candidates)
    candidates = [c for c in candidates if c.doi and c.doi.lower() not in existing_dois]
    deduped = before_dedup - len(candidates)
    if deduped:
        console.print(
            f"[dim]Skipping [bold]{deduped}[/bold] paper(s) already in the corpus.[/dim]"
        )

    if not candidates:
        console.print(
            "[yellow]All matching papers are already in your corpus. Nothing to do.[/yellow]"
        )
        raise typer.Exit(0)

    # ── Step 4: optional relevance filter ────────────────────────────────────
    # (mirrors `db expand` — same code, just on a different candidate source)
    if relevance:
        try:
            from sciknow.retrieval.relevance import (
                compute_corpus_centroid, embed_query, score_candidates,
                score_histogram,
            )
            eff_threshold = relevance_threshold if relevance_threshold > 0 else getattr(
                settings, "expand_relevance_threshold", 0.55
            )
            console.print(
                f"\n[dim]Applying relevance filter "
                f"(threshold={eff_threshold:.2f})…[/dim]"
            )
            # Phase 16.1 — fixed import names. The module exports
            # compute_corpus_centroid (not build_corpus_centroid) and
            # embed_query (not embed_anchor). The wrong names were a
            # copy-paste error in the original Phase 16 ship that the
            # graceful try/except hid until a real run surfaced it.
            anchor_vec = (
                embed_query(relevance_query) if relevance_query
                else compute_corpus_centroid()
            )
            titles = [c.title or "" for c in candidates]
            scores = score_candidates(titles, anchor_vec)

            kept_pairs = [
                (c, s) for c, s in zip(candidates, scores) if s >= eff_threshold
            ]
            dropped = len(candidates) - len(kept_pairs)

            hist = score_histogram(scores, bins=10)
            if hist:
                console.print("[dim]Relevance score distribution:[/dim]")
                max_count = max(c for _, _, c in hist) or 1
                for lo, hi, c in hist:
                    bar_width = int(40 * c / max_count)
                    marker = "  " if hi < eff_threshold else "▶ " if lo <= eff_threshold < hi else "  "
                    console.print(
                        f"  {marker}[dim]{lo:.2f}-{hi:.2f}[/dim] "
                        f"{'█' * bar_width} {c}"
                    )
                console.print(
                    f"  [green]kept {len(kept_pairs)}[/green]  "
                    f"[red]dropped {dropped}[/red]  (cut at {eff_threshold:.2f})"
                )

            candidates = [c for c, _ in sorted(kept_pairs, key=lambda x: x[1], reverse=True)]
        except Exception as exc:
            console.print(
                f"[yellow]⚠ Relevance filter failed ({type(exc).__name__}: {exc}); "
                f"continuing without it. Use --no-relevance to skip explicitly.[/yellow]"
            )

    if not candidates:
        console.print("[yellow]Nothing to download after relevance filter.[/yellow]")
        raise typer.Exit(0)

    # ── Step 5: dry-run preview ──────────────────────────────────────────────
    if dry_run:
        console.print(
            f"\n[bold]Dry run — would attempt to download {len(candidates)} paper(s):[/bold]"
        )
        for ref in candidates[:30]:
            year_str = f"({ref.year})" if ref.year else "(n.d.)"
            console.print(
                f"  [dim]{(ref.doi or '?')[:40]:<40}[/dim] "
                f"[cyan]{year_str}[/cyan] {(ref.title or '')[:60]}"
            )
        if len(candidates) > 30:
            console.print(f"  … and {len(candidates) - 30} more")
        raise typer.Exit(0)

    # ── Step 6: download phase (parallel) ────────────────────────────────────
    download_dir.mkdir(parents=True, exist_ok=True)
    no_oa_cache_file = download_dir / ".no_oa_cache"
    no_oa_cache: set[str] = (
        set(no_oa_cache_file.read_text().splitlines())
        if no_oa_cache_file.exists() else set()
    )

    dl_workers = max(1, getattr(settings, "expand_download_workers", 4))
    downloaded = failed_dl = 0
    to_ingest: list[tuple[str, str, Path]] = []  # (key, title, path)

    def _download_one(ref):
        ref_key = (ref.doi or "").lower()
        safe_name = (ref.doi or "unknown").replace("/", "_").replace(":", "_")
        dest = download_dir / f"{safe_name}.pdf"
        title = (ref.title or "")[:80]
        if ref_key in no_oa_cache:
            return ("cached", ref_key, title, dest, None)
        if dest.exists():
            return ("exists", ref_key, title, dest, None)
        ok, source = find_and_download(
            doi=ref.doi, arxiv_id=None,
            dest_path=dest, email=settings.crossref_email,
        )
        return ("downloaded" if ok else "no_oa", ref_key, title, dest, source)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Downloading ({dl_workers} workers)", total=len(candidates)
        )
        with ThreadPoolExecutor(max_workers=dl_workers) as pool:
            futures = {pool.submit(_download_one, ref): ref for ref in candidates}
            for fut in as_completed(futures):
                ref = futures[fut]
                try:
                    status, ref_key, title, dest, source = fut.result()
                except Exception as exc:
                    failed_dl += 1
                    progress.advance(task)
                    continue
                label_short = (ref.title or ref.doi or "")[:50]
                if status == "downloaded":
                    downloaded += 1
                    progress.update(task, description=f"[green]↓ {source}[/green] {label_short[:40]}")
                    to_ingest.append((ref_key, title, dest))
                elif status == "exists":
                    to_ingest.append((ref_key, title, dest))
                elif status == "no_oa":
                    failed_dl += 1
                    with no_oa_cache_file.open("a") as f:
                        f.write(ref_key + "\n")
                    # Phase 54.6.7 — record for the pending-downloads panel.
                    try:
                        from sciknow.core.pending_ops import record_failure
                        record_failure(
                            doi=ref.doi or "", title=ref.title or "",
                            authors=list(ref.authors or []),
                            year=ref.year, arxiv_id=ref.arxiv_id,
                            source_method="expand-author",
                            source_query=name,
                            reason="no_oa",
                        )
                    except Exception:
                        pass
                else:  # cached
                    failed_dl += 1
                progress.advance(task)

    # ── Step 7: ingest phase (worker pool, same as db expand) ────────────────
    ingested = failed_ingest = 0
    if ingest and to_ingest:
        from sciknow.cli.ingest import _run_parallel_workers
        from sciknow.ingestion.embedder import release_model as _release_embedder

        ingest_workers = workers if workers > 0 else max(1, settings.ingest_workers)
        ingest_workers = min(ingest_workers, len(to_ingest))
        _release_embedder()  # free main-process bge-m3 before workers fork

        path_to_meta = {dest.resolve(): (k, t) for k, t, dest in to_ingest}

        def _on_file_done(path, status, error):
            nonlocal ingested, failed_ingest
            if status in ("done", "skipped"):
                ingested += 1
            elif status == "failed":
                failed_ingest += 1

        ingest_results = {"done": 0, "skipped": 0, "failed": 0}
        ingest_failed_files: list[tuple[str, str]] = []

        worker_note = f" ({ingest_workers} workers)" if ingest_workers > 1 else ""
        console.print(f"\nIngesting [bold]{len(to_ingest)}[/bold] new PDF(s){worker_note}…")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            itask = progress.add_task("Ingesting", total=len(to_ingest))
            _run_parallel_workers(
                [dest for _, _, dest in to_ingest],
                progress, itask, ingest_results, ingest_failed_files,
                force=False, num_workers=ingest_workers,
                ingest_source="expand",
                on_file_done=_on_file_done,
            )

    # ── Summary ──────────────────────────────────────────────────────────────
    console.print(
        f"\n[bold]Summary:[/bold] "
        f"[green]↓ {downloaded} downloaded[/green]  "
        f"[green]✓ {ingested} ingested[/green]  "
        f"[red]✗ {failed_dl} no OA PDF[/red]"
        + (f"  [red]✗ {failed_ingest} ingest failed[/red]" if failed_ingest else "")
    )
    if downloaded:
        console.print(
            f"\nNew PDFs in [bold]{download_dir}[/bold]. "
            f"Run [bold]sciknow catalog stats[/bold] to see updated counts."
        )


@app.command(name="download-dois")
def download_dois(
    dois: str = typer.Option(
        "", "--dois",
        help="Comma-separated DOI list. Either this or --dois-file is required.",
    ),
    dois_file: Path = typer.Option(
        None, "--dois-file",
        help="JSON file with either a list of DOI strings OR a list of "
             "{doi, title, year} dicts. Title/year are used for progress "
             "display only — the download pipeline only needs the DOI.",
    ),
    download_dir: Path = typer.Option(
        None, "--download-dir", "-d",
        help="Directory where new PDFs are saved (default: <project data>/downloads).",
    ),
    workers: int = typer.Option(0, "--workers", "-w",
        help="Parallel ingestion workers (0 = INGEST_WORKERS from .env)."),
    ingest: bool = typer.Option(True, "--ingest/--no-ingest",
        help="Ingest downloaded PDFs immediately."),
    retry_failed: bool = typer.Option(False, "--retry-failed",
        help="Ignore the .no_oa_cache — re-attempt DOIs that returned no OA PDF "
             "on prior runs. Used when the pending-downloads panel triggers a "
             "retry (Unpaywall / S2 / Europe PMC sometimes surface a new link)."),
):
    """Download + ingest a specific list of DOIs.

    This is the primitive behind the "Download selected" button in the web
    Expand-by-Author preview modal (Phase 54.6.1). Shares the 6-source
    OA discovery pipeline and the parallel ingest worker pool with
    `db expand-author` — just skips search / dedup / relevance scoring.

    Examples:

      sciknow db download-dois --dois "10.1038/nature12345,10.1126/science.abc"

      sciknow db download-dois --dois-file selected.json --workers 4
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from sciknow.cli import preflight
    preflight()

    from sciknow.config import settings
    from sciknow.ingestion.downloader import find_and_download

    if download_dir is None:
        download_dir = settings.data_dir / "downloads"

    # ── Parse input ──────────────────────────────────────────────────────
    # Phase 54.6.51 — each entry now carries optional alternate DOIs +
    # arXiv IDs so the downloader can fall back to a preprint mirror
    # when the journal DOI's OA discovery returns nothing. Tuple layout:
    #   (doi, title, year, alternate_dois, alternate_arxiv_ids)
    doi_list: list[tuple[str, str, int | None, list[str], list[str]]] = []
    if dois_file:
        raw = json.loads(Path(dois_file).read_text())
        if not isinstance(raw, list):
            console.print("[red]--dois-file must contain a JSON list.[/red]")
            raise typer.Exit(2)
        for item in raw:
            if isinstance(item, str):
                doi_list.append((item, "", None, [], []))
            elif isinstance(item, dict) and item.get("doi"):
                doi_list.append((
                    item["doi"],
                    item.get("title", "") or "",
                    item.get("year"),
                    list(item.get("alternate_dois") or []),
                    list(item.get("alternate_arxiv_ids") or []),
                ))
            else:
                console.print(f"[yellow]Skipping malformed entry: {item!r}[/yellow]")
    if dois:
        for d in dois.split(","):
            d = d.strip()
            if d:
                doi_list.append((d, "", None, [], []))

    # dedup by DOI
    seen: set[str] = set()
    doi_list = [
        entry for entry in doi_list
        if (entry[0].lower() not in seen and not seen.add(entry[0].lower()))
    ]

    if not doi_list:
        console.print("[red]No DOIs provided (use --dois or --dois-file).[/red]")
        raise typer.Exit(2)

    console.print(
        f"[bold]Downloading {len(doi_list)} DOI(s)[/bold] into {download_dir}\n"
    )

    # ── Download phase ───────────────────────────────────────────────────
    download_dir.mkdir(parents=True, exist_ok=True)
    no_oa_cache_file = download_dir / ".no_oa_cache"
    no_oa_cache: set[str] = (
        set(no_oa_cache_file.read_text().splitlines())
        if (no_oa_cache_file.exists() and not retry_failed) else set()
    )
    if retry_failed:
        console.print("[dim]--retry-failed: ignoring .no_oa_cache for this run.[/dim]")

    dl_workers = max(1, getattr(settings, "expand_download_workers", 4))
    downloaded = failed_dl = 0
    to_ingest: list[tuple[str, str, Path]] = []

    def _download_one(item):
        doi, title, _year, alt_dois, alt_arxiv = item
        doi_key = doi.lower()
        safe_name = doi.replace("/", "_").replace(":", "_")
        dest = download_dir / f"{safe_name}.pdf"
        label = title[:80] or doi
        if doi_key in no_oa_cache:
            return ("cached", doi_key, label, dest, None)
        if dest.exists():
            return ("exists", doi_key, label, dest, None)
        ok, source = find_and_download(
            doi=doi, arxiv_id=None,
            dest_path=dest, email=settings.crossref_email,
            alternate_dois=alt_dois,
            alternate_arxiv_ids=alt_arxiv,
        )
        return ("downloaded" if ok else "no_oa", doi_key, label, dest, source)

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
            f"Downloading ({dl_workers} workers)", total=len(doi_list)
        )
        # Phase 54.6.47 — same durable-per-event pattern as db expand
        # (54.6.45). Rich's Progress bar uses \r to update in-place; when
        # download-dois is spawned by the web UI via subprocess pipe,
        # those \r updates don't produce newline-terminated log events
        # so the GUI log pane shows only the startup header and looks
        # frozen for the full duration of the download batch. Emit
        # durable per-DOI lines via progress.console.print so the web
        # SSE stream sees real events.
        done_count = 0
        total_ct = len(doi_list)

        def _emit(mark: str, color: str, kind: str, label_: str, note: str = "") -> None:
            progress.console.print(
                f"[dim][{done_count:>4d}/{total_ct}][/dim]  "
                f"[{color}]{mark} {kind:<8}[/{color}] {label_[:70]:<70}"
                + (f"  [dim]· {note}[/dim]" if note else "")
            )

        with ThreadPoolExecutor(max_workers=dl_workers) as pool:
            futures = {pool.submit(_download_one, it): it for it in doi_list}
            for fut in as_completed(futures):
                done_count += 1
                try:
                    status, doi_key, label, dest, source = fut.result()
                except Exception as exc:
                    failed_dl += 1
                    _emit("✗", "red", "ERROR", (str(exc) or "")[:70], "")
                    progress.advance(task)
                    continue
                short = label[:50]
                if status == "downloaded":
                    downloaded += 1
                    progress.update(task, description=f"[green]↓ {source}[/green] {short}")
                    _emit("✓", "green", "DL", label, f"source={source}")
                    to_ingest.append((doi_key, label, dest))
                elif status == "exists":
                    _emit("↺", "cyan", "EXISTS", label, "pdf on disk; queued for ingest")
                    to_ingest.append((doi_key, label, dest))
                elif status == "no_oa":
                    failed_dl += 1
                    _emit("✗", "yellow", "NO_OA", label, "no open-access PDF found")
                    with no_oa_cache_file.open("a") as f:
                        f.write(doi_key + "\n")
                    # Phase 54.6.7 — stash the row so it shows up in
                    # the pending-downloads panel. We match the DOI
                    # back to its full metadata from doi_list (which
                    # the download-dois CLI / web selected-download
                    # endpoint fills with title + year when it has them).
                    try:
                        from sciknow.core.pending_ops import record_failure
                        meta = next(
                            (it for it in doi_list
                             if (it[0] or "").lower() == doi_key),
                            None,
                        )
                        t, y = (meta[1], meta[2]) if meta else ("", None)
                        record_failure(
                            doi=doi_key or "", title=t or "",
                            year=y, source_method="download-dois",
                            reason="no_oa",
                        )
                    except Exception:
                        pass
                elif status == "cached":
                    _emit("⏭", "dim", "CACHED", label, "no_oa cached from prior run")
                else:
                    failed_dl += 1
                    _emit("✗", "red", "FAIL", label, f"unknown status={status}")
                progress.advance(task)

    # ── Ingest phase ─────────────────────────────────────────────────────
    ingested = failed_ingest = 0
    if ingest and to_ingest:
        from sciknow.cli.ingest import _run_parallel_workers
        from sciknow.ingestion.embedder import release_model as _release_embedder
        # Phase 54.6.48 — also release any Ollama LLM from VRAM. Pre-fix,
        # the web "Download selected" flow triggered from the wiki modal
        # would leave qwen3:30b-a3b-instruct-2507 resident (keep_alive=-1)
        # and MinerU would OOM on every PDF in the ingest phase. See
        # pipeline.py for the belt-and-braces inside the pipeline itself.
        from sciknow.rag.llm import release_llm as _release_llm
        try:
            released = _release_llm()
            if released:
                console.print(
                    f"[dim]Freed VRAM before ingest: unloaded "
                    f"{', '.join(released)}[/dim]"
                )
        except Exception as exc:
            console.print(
                f"[yellow]Warning: could not unload LLM(s) before ingest: {exc}. "
                f"MinerU may OOM on large PDFs.[/yellow]"
            )

        ingest_workers = workers if workers > 0 else max(1, settings.ingest_workers)
        ingest_workers = min(ingest_workers, len(to_ingest))
        _release_embedder()

        # Phase 54.6.47 — durable per-file ingest lines (same rationale as
        # the download phase above: \r-updated Rich progress bars get lost
        # in the web SSE stream).
        _ing_progress_ref: list = [None]
        path_to_title: dict = {dest.resolve(): label for _, label, dest in to_ingest}

        def _on_file_done(path, status, error):
            nonlocal ingested, failed_ingest
            label = (path_to_title.get(path.resolve(), path.name))[:70]
            prog = _ing_progress_ref[0]

            def _say(mark, color, kind, note=""):
                if prog is None:
                    return
                prog.console.print(
                    f"  [{color}]{mark} {kind:<11}[/{color}] {label[:70]:<70}"
                    + (f"  [dim]· {note}[/dim]" if note else "")
                )

            if status == "done":
                ingested += 1
                _say("✓", "green", "INGEST")
            elif status == "skipped":
                ingested += 1
                _say("⏭", "dim", "INGEST-SKIP", "already in DB (hash match)")
            elif status == "failed":
                failed_ingest += 1
                _say("✗", "red", "INGEST-FAIL", (error or "")[:50])

        ingest_results = {"done": 0, "skipped": 0, "failed": 0}
        ingest_failed_files: list[tuple[str, str]] = []
        worker_note = f" ({ingest_workers} workers)" if ingest_workers > 1 else ""
        console.print(f"\nIngesting [bold]{len(to_ingest)}[/bold] new PDF(s){worker_note}…")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            _ing_progress_ref[0] = progress
            itask = progress.add_task("Ingesting", total=len(to_ingest))
            _run_parallel_workers(
                [dest for _, _, dest in to_ingest],
                progress, itask, ingest_results, ingest_failed_files,
                force=False, num_workers=ingest_workers,
                ingest_source="expand",
                on_file_done=_on_file_done,
            )

    console.print(
        f"\n[bold]Summary:[/bold] "
        f"[green]↓ {downloaded} downloaded[/green]  "
        f"[green]✓ {ingested} ingested[/green]  "
        f"[red]✗ {failed_dl} no OA PDF[/red]"
        + (f"  [red]✗ {failed_ingest} ingest failed[/red]" if failed_ingest else "")
    )


# ── Phase 54.6.4 — three new expansion methods ───────────────────────
# Each is a thin wrapper that delegates to `sciknow/core/expand_ops.py`
# for the candidate-finding logic (so the web preview endpoints and
# the CLI share the same code path), then funnels the candidate DOIs
# into the existing parallel download + ingest pipeline.

def _expand_common_download_and_ingest(
    candidates: list[dict], *, download_dir,
    workers: int, ingest: bool, dry_run: bool,
    source_method: str | None = None,
    source_query: str | None = None,
) -> None:
    """Shared tail for expand-cites / expand-topic / expand-coauthors."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from sciknow.config import settings
    from sciknow.ingestion.downloader import find_and_download

    if dry_run:
        console.print(
            f"\n[bold]Dry run — would attempt to download "
            f"{len(candidates)} paper(s):[/bold]"
        )
        for c in candidates[:30]:
            year_str = f"({c.get('year')})" if c.get("year") else "(n.d.)"
            score = c.get("relevance_score")
            sc = f" score={score:.2f}" if score is not None else ""
            console.print(
                f"  [dim]{(c.get('doi') or '?')[:40]:<40}[/dim] "
                f"[cyan]{year_str}[/cyan]{sc} {(c.get('title') or '')[:60]}"
            )
        if len(candidates) > 30:
            console.print(f"  … and {len(candidates) - 30} more")
        raise typer.Exit(0)

    if download_dir is None:
        download_dir = settings.data_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    no_oa_cache_file = download_dir / ".no_oa_cache"
    no_oa_cache: set[str] = (
        set(no_oa_cache_file.read_text().splitlines())
        if no_oa_cache_file.exists() else set()
    )
    dl_workers = max(1, getattr(settings, "expand_download_workers", 4))
    downloaded = failed_dl = 0
    to_ingest = []

    def _dl(item):
        doi = item.get("doi") or ""
        title = (item.get("title") or "")[:80]
        doi_key = doi.lower()
        safe = doi.replace("/", "_").replace(":", "_")
        dest = download_dir / f"{safe}.pdf"
        if doi_key in no_oa_cache:
            return ("cached", doi_key, title, dest, None)
        if dest.exists():
            return ("exists", doi_key, title, dest, None)
        ok, source = find_and_download(
            doi=doi, arxiv_id=None, dest_path=dest,
            email=settings.crossref_email,
        )
        return ("downloaded" if ok else "no_oa", doi_key, title, dest, source)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Downloading ({dl_workers} workers)", total=len(candidates)
        )
        with ThreadPoolExecutor(max_workers=dl_workers) as pool:
            futures = {pool.submit(_dl, c): c for c in candidates}
            for fut in as_completed(futures):
                cand = futures[fut]
                try:
                    status, doi_key, title, dest, source = fut.result()
                except Exception:
                    failed_dl += 1
                    progress.advance(task)
                    continue
                short = title[:50]
                if status == "downloaded":
                    downloaded += 1
                    progress.update(task, description=f"[green]↓ {source}[/green] {short}")
                    to_ingest.append((doi_key, title, dest))
                elif status == "exists":
                    to_ingest.append((doi_key, title, dest))
                elif status == "no_oa":
                    failed_dl += 1
                    with no_oa_cache_file.open("a") as f:
                        f.write(doi_key + "\n")
                    # Phase 54.6.7 — save to pending_downloads with the
                    # richest metadata the candidate dict carries.
                    try:
                        from sciknow.core.pending_ops import record_failure
                        record_failure(
                            doi=cand.get("doi") or doi_key or "",
                            title=cand.get("title") or "",
                            authors=list(cand.get("authors") or []),
                            year=cand.get("year"),
                            arxiv_id=cand.get("arxiv_id"),
                            relevance_score=cand.get("relevance_score"),
                            source_method=source_method,
                            source_query=source_query,
                            reason="no_oa",
                        )
                    except Exception:
                        pass
                else:
                    failed_dl += 1
                progress.advance(task)

    ingested = failed_ingest = 0
    if ingest and to_ingest:
        from sciknow.cli.ingest import _run_parallel_workers
        from sciknow.ingestion.embedder import release_model as _release_embedder

        ingest_workers = workers if workers > 0 else max(1, settings.ingest_workers)
        ingest_workers = min(ingest_workers, len(to_ingest))
        _release_embedder()

        def _on_file_done(path, status, error):
            nonlocal ingested, failed_ingest
            if status in ("done", "skipped"):
                ingested += 1
            elif status == "failed":
                failed_ingest += 1

        ingest_results = {"done": 0, "skipped": 0, "failed": 0}
        ingest_failed_files = []
        worker_note = f" ({ingest_workers} workers)" if ingest_workers > 1 else ""
        console.print(f"\nIngesting [bold]{len(to_ingest)}[/bold] new PDF(s){worker_note}…")

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            console=console,
        ) as progress:
            itask = progress.add_task("Ingesting", total=len(to_ingest))
            _run_parallel_workers(
                [dest for _, _, dest in to_ingest],
                progress, itask, ingest_results, ingest_failed_files,
                force=False, num_workers=ingest_workers,
                ingest_source="expand",
                on_file_done=_on_file_done,
            )

    console.print(
        f"\n[bold]Summary:[/bold] "
        f"[green]↓ {downloaded} downloaded[/green]  "
        f"[green]✓ {ingested} ingested[/green]  "
        f"[red]✗ {failed_dl} no OA PDF[/red]"
        + (f"  [red]✗ {failed_ingest} ingest failed[/red]" if failed_ingest else "")
    )


@app.command(name="expand-cites")
def expand_cites(
    per_seed_cap: int = typer.Option(50, "--per-seed-cap",
        help="Max papers per seed (corpus paper). Default 50."),
    total_limit: int = typer.Option(500, "--total-limit",
        help="Hard cap on total candidate pool (pre-filter)."),
    relevance: bool = typer.Option(True, "--relevance/--no-relevance"),
    relevance_query: str = typer.Option("", "--relevance-query", "-q"),
    relevance_threshold: float = typer.Option(0.0, "--relevance-threshold"),
    download_dir: Path = typer.Option(None, "--download-dir", "-d"),
    workers: int = typer.Option(0, "--workers", "-w"),
    ingest: bool = typer.Option(True, "--ingest/--no-ingest"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Phase 54.6.4 — Inbound-citation discovery.

    Query OpenAlex for papers that CITE each paper in your corpus.
    Dedup, score, rank, download + ingest. Mirror of `db expand`
    (outbound) — catches forward-in-time work.
    """
    from sciknow.cli import preflight
    preflight()
    from sciknow.core.expand_ops import find_inbound_citation_candidates

    console.print("\n[bold]Expand by inbound citations[/bold]")
    console.print("[dim]OpenAlex /works?filter=cites:W… per seed[/dim]\n")
    result = find_inbound_citation_candidates(
        per_seed_cap=per_seed_cap, total_limit=total_limit,
        relevance_query=relevance_query, score_relevance=relevance,
    )
    cands = result["candidates"]
    info = result["info"]
    console.print(
        f"Resolved {info.get('seeds_resolved', 0)} seed work(s) of "
        f"{info.get('seeds_requested', 0)}; {info.get('raw', 0)} raw citers, "
        f"dedup'd {info.get('dedup_dropped', 0)}."
    )
    if relevance_threshold > 0:
        before = len(cands)
        cands = [c for c in cands if (c.get("relevance_score") or 0.0) >= relevance_threshold]
        console.print(f"Threshold {relevance_threshold:.2f} kept {len(cands)} / {before}.")
    if not cands:
        console.print("[yellow]Nothing to download.[/yellow]")
        raise typer.Exit(0)
    _expand_common_download_and_ingest(
        cands, download_dir=download_dir, workers=workers, ingest=ingest, dry_run=dry_run,
        source_method="expand-cites",
    )


@app.command(name="expand-topic")
def expand_topic(
    query: Annotated[str, typer.Argument(help="Free-text topic query.")],
    limit: int = typer.Option(500, "--limit", "-l"),
    relevance: bool = typer.Option(True, "--relevance/--no-relevance"),
    relevance_query: str = typer.Option("", "--relevance-query", "-q"),
    relevance_threshold: float = typer.Option(0.0, "--relevance-threshold"),
    download_dir: Path = typer.Option(None, "--download-dir", "-d"),
    workers: int = typer.Option(0, "--workers", "-w"),
    ingest: bool = typer.Option(True, "--ingest/--no-ingest"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Phase 54.6.4 — Topic-driven expansion.

    Free-text OpenAlex /works?search=QUERY sorted by citation count.
    Solves bootstrap / sideways-expansion `db expand` can't address.
    """
    from sciknow.cli import preflight
    preflight()
    from sciknow.core.expand_ops import find_topic_candidates

    console.print(f"\n[bold]Expand by topic search[/bold]: {query!r}")
    console.print("[dim]OpenAlex /works?search=… sort=cited_by_count:desc[/dim]\n")
    result = find_topic_candidates(
        query, limit=limit,
        relevance_query=relevance_query, score_relevance=relevance,
    )
    cands = result["candidates"]
    info = result["info"]
    console.print(
        f"Fetched {info.get('raw', 0)} candidate(s), "
        f"dedup'd {info.get('dedup_dropped', 0)} against corpus."
    )
    if relevance_threshold > 0:
        before = len(cands)
        cands = [c for c in cands if (c.get("relevance_score") or 0.0) >= relevance_threshold]
        console.print(f"Threshold {relevance_threshold:.2f} kept {len(cands)} / {before}.")
    if not cands:
        console.print("[yellow]Nothing to download.[/yellow]")
        raise typer.Exit(0)
    _expand_common_download_and_ingest(
        cands, download_dir=download_dir, workers=workers, ingest=ingest, dry_run=dry_run,
        source_method="expand-topic", source_query=query,
    )


@app.command(name="expand-coauthors")
def expand_coauthors(
    depth: int = typer.Option(1, "--depth"),
    per_author_cap: int = typer.Option(10, "--per-author-cap"),
    total_limit: int = typer.Option(500, "--total-limit"),
    relevance: bool = typer.Option(True, "--relevance/--no-relevance"),
    relevance_query: str = typer.Option("", "--relevance-query", "-q"),
    relevance_threshold: float = typer.Option(0.0, "--relevance-threshold"),
    download_dir: Path = typer.Option(None, "--download-dir", "-d"),
    workers: int = typer.Option(0, "--workers", "-w"),
    ingest: bool = typer.Option(True, "--ingest/--no-ingest"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Phase 54.6.4 — Coauthor-network snowball.

    Every OpenAlex author on any corpus paper → fetch up to
    ``--per-author-cap`` of their works. Captures the invisible
    college of same-lab researchers who don't cite each other.
    """
    from sciknow.cli import preflight
    preflight()
    from sciknow.core.expand_ops import find_coauthor_candidates

    console.print(f"\n[bold]Expand by coauthor snowball[/bold] (depth={depth})")
    console.print("[dim]Corpus authors → OpenAlex /works?filter=author.id:…[/dim]\n")
    result = find_coauthor_candidates(
        depth=depth, per_author_cap=per_author_cap, total_limit=total_limit,
        relevance_query=relevance_query, score_relevance=relevance,
    )
    cands = result["candidates"]
    info = result["info"]
    console.print(
        f"Seed authors: {info.get('seed_authors', 0)}. "
        f"Raw: {info.get('raw', 0)}, dedup'd {info.get('dedup_dropped', 0)}."
    )
    if relevance_threshold > 0:
        before = len(cands)
        cands = [c for c in cands if (c.get("relevance_score") or 0.0) >= relevance_threshold]
        console.print(f"Threshold {relevance_threshold:.2f} kept {len(cands)} / {before}.")
    if not cands:
        console.print("[yellow]Nothing to download.[/yellow]")
        raise typer.Exit(0)
    _expand_common_download_and_ingest(
        cands, download_dir=download_dir, workers=workers, ingest=ingest, dry_run=dry_run,
        source_method="expand-coauthors",
    )


# ── Phase 54.6.7 — pending_downloads sub-app ──────────────────────────
# Papers the user selected via any expand flow that couldn't be
# auto-downloaded (no OA PDF) land in the pending_downloads table. This
# sub-app lets the user view / retry / mark-done / abandon / export.

pending_app = typer.Typer(help="Manage pending_downloads table (papers "
                               "selected for ingest but no OA PDF was "
                               "found — ripe for retry or manual acquisition).")
app.add_typer(pending_app, name="pending")


@pending_app.command(name="list")
def pending_list(
    status: str = typer.Option("pending", "--status", "-s",
        help="Filter by status (pending / manual_acquired / abandoned / all). "
             "Default: pending."),
    source: str = typer.Option("", "--source",
        help="Filter by source_method (expand / expand-author / expand-cites / "
             "expand-topic / expand-coauthors / auto-expand / download-dois)."),
    limit: int = typer.Option(50, "--limit", "-l",
        help="Max rows to display."),
):
    """Show the pending-downloads table (papers waiting on a legal OA PDF)."""
    from sciknow.cli import preflight
    preflight()
    from sciknow.core.pending_ops import list_pending
    rows = list_pending(
        status=(status or None),
        source_method=(source.strip() or None),
        limit=limit,
    )
    if not rows:
        console.print("[green]No pending entries.[/green]")
        return
    table = Table(title=f"pending_downloads (status={status!r})",
                  box=box.SIMPLE_HEAD, expand=True)
    table.add_column("DOI", ratio=4, no_wrap=False)
    table.add_column("Title", ratio=6, no_wrap=False)
    table.add_column("Yr", width=4, justify="right")
    table.add_column("Src", width=14)
    table.add_column("Tries", width=5, justify="right")
    table.add_column("Last reason", ratio=3)
    for r in rows:
        table.add_row(
            (r["doi"] or "")[:40],
            (r["title"] or "")[:80],
            str(r["year"] or ""),
            (r["source_method"] or "")[:14],
            str(r["attempt_count"]),
            (r["last_failure_reason"] or "")[:30],
        )
    console.print(table)
    console.print(
        f"\n[dim]{len(rows)} row(s). Retry with "
        f"[bold]sciknow db pending retry[/bold], mark as manually acquired "
        f"with [bold]sciknow db pending mark-done <doi>[/bold], abandon "
        f"with [bold]sciknow db pending abandon <doi>[/bold].[/dim]"
    )


@pending_app.command(name="retry")
def pending_retry(
    limit: int = typer.Option(0, "--limit", "-l",
        help="Max DOIs to retry in this run (0 = all pending)."),
    download_dir: Path = typer.Option(None, "--download-dir", "-d"),
    workers: int = typer.Option(0, "--workers", "-w"),
    ingest: bool = typer.Option(True, "--ingest/--no-ingest"),
):
    """Retry every pending DOI against the 6-source OA cascade.

    Passes ``--retry-failed`` to ``download-dois`` internally so the
    .no_oa_cache is bypassed (the whole point is that one of the
    sources might have surfaced a new PDF since the last attempt).
    """
    from sciknow.cli import preflight
    preflight()
    from sciknow.core.pending_ops import list_pending

    rows = list_pending(status="pending", limit=(limit if limit > 0 else 10000))
    if not rows:
        console.print("[green]Nothing pending to retry.[/green]")
        return
    console.print(
        f"[bold]Retrying {len(rows)} pending DOI(s)…[/bold]"
    )
    # Serialise the rows as the dois-file format expected by download-dois.
    import json as _json
    import tempfile
    tmp_dir = Path(tempfile.gettempdir()) / "sciknow-pending-retry"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"retry-{os.getpid()}.json"
    tmp_path.write_text(_json.dumps([
        {"doi": r["doi"], "title": r["title"], "year": r["year"]}
        for r in rows
    ]))
    # Reuse download-dois directly — that function already records
    # failures back into pending_downloads, bumping attempt_count.
    try:
        download_dois(
            dois="", dois_file=tmp_path,
            download_dir=download_dir, workers=workers,
            ingest=ingest, retry_failed=True,
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@pending_app.command(name="mark-done")
def pending_mark_done(
    doi: Annotated[str, typer.Argument(help="DOI to mark manual_acquired.")],
    note: str = typer.Option("", "--note", "-n",
        help="Optional note (how/where acquired)."),
):
    """Mark a DOI as manually acquired (ILL / sci-hub / author email / …)."""
    from sciknow.cli import preflight
    preflight()
    from sciknow.core.pending_ops import update_status
    if update_status(doi, status="manual_acquired", notes=(note or None)):
        console.print(f"[green]✓[/green] {doi} → manual_acquired")
    else:
        console.print(f"[yellow]DOI not found in pending_downloads:[/yellow] {doi}")


@pending_app.command(name="abandon")
def pending_abandon(
    doi: Annotated[str, typer.Argument(help="DOI to abandon.")],
    note: str = typer.Option("", "--note", "-n"),
):
    """Mark a DOI as abandoned (decided it's not worth chasing)."""
    from sciknow.cli import preflight
    preflight()
    from sciknow.core.pending_ops import update_status
    if update_status(doi, status="abandoned", notes=(note or None)):
        console.print(f"[green]✓[/green] {doi} → abandoned")
    else:
        console.print(f"[yellow]DOI not found in pending_downloads:[/yellow] {doi}")


@pending_app.command(name="reopen")
def pending_reopen(
    doi: Annotated[str, typer.Argument(help="DOI to move back to 'pending'.")],
):
    """Move a manual_acquired / abandoned DOI back to pending."""
    from sciknow.cli import preflight
    preflight()
    from sciknow.core.pending_ops import update_status
    if update_status(doi, status="pending"):
        console.print(f"[green]✓[/green] {doi} → pending")
    else:
        console.print(f"[yellow]DOI not found:[/yellow] {doi}")


@pending_app.command(name="remove")
def pending_remove(
    doi: Annotated[str, typer.Argument(help="DOI to delete from the table.")],
):
    """Delete a pending row (use abandon unless you really want it gone)."""
    from sciknow.cli import preflight
    preflight()
    from sciknow.core.pending_ops import remove as _remove
    if _remove(doi):
        console.print(f"[green]✓[/green] deleted {doi}")
    else:
        console.print(f"[yellow]DOI not found:[/yellow] {doi}")


@pending_app.command(name="export")
def pending_export(
    output: Path = typer.Option(None, "--output", "-o",
        help="Output path (default: stdout)."),
    fmt: str = typer.Option("csv", "--format", "-f",
        help="csv | json."),
    status: str = typer.Option("pending", "--status", "-s"),
):
    """Dump the pending table to CSV or JSON for manual acquisition."""
    from sciknow.cli import preflight
    preflight()
    from sciknow.core.pending_ops import list_pending
    rows = list_pending(status=(status or None), limit=100000)
    if fmt.lower() == "json":
        import json as _json
        text = _json.dumps(rows, indent=2, ensure_ascii=False)
    else:
        import csv as _csv
        import io as _io
        buf = _io.StringIO()
        cols = ["doi", "title", "authors", "year", "source_method",
                "source_query", "relevance_score", "attempt_count",
                "last_attempt_at", "last_failure_reason", "status",
                "notes", "created_at"]
        w = _csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["authors"] = "; ".join(r.get("authors") or [])
            w.writerow(r)
        text = buf.getvalue()
    if output:
        output.write_text(text)
        console.print(f"[green]✓[/green] Wrote {len(rows)} row(s) → {output}")
    else:
        console.print(text)


@app.command(name="extract-visuals")
def extract_visuals_cmd(
    limit: int = typer.Option(0, "--limit", help="Process at most N papers (0 = all)."),
    force: bool = typer.Option(False, "--force",
        help="Re-extract even for papers that already have visuals rows."),
):
    """Phase 21.a — extract figures, tables, equations from content_list.json.

    Walks each ingested paper's MinerU output and creates one ``visuals``
    row per visual element (table, equation, figure, code block). No
    re-ingestion needed — reads the existing content_list.json files.

    Safe to re-run: skips papers that already have visuals unless --force.

    Examples:

      sciknow db extract-visuals                # extract for all papers
      sciknow db extract-visuals --limit 50     # first 50 only
      sciknow db extract-visuals --force        # re-extract everything
    """
    import re as _re

    from sciknow.cli import preflight
    preflight()

    from sqlalchemy import text as sql_text
    from sciknow.config import settings
    from sciknow.storage.db import get_session

    with get_session() as session:
        rows = session.execute(sql_text("""
            SELECT d.id::text, d.file_hash
            FROM documents d
            WHERE d.ingestion_status = 'complete'
            ORDER BY d.created_at
        """)).fetchall()

    total = len(rows)
    if limit > 0:
        rows = rows[:limit]

    console.print(f"Scanning [bold]{len(rows)}[/bold] of {total} papers for visuals…")

    # Pre-fetch which docs already have visuals (skip unless --force)
    done_doc_ids: set[str] = set()
    if not force:
        with get_session() as session:
            done_rows = session.execute(sql_text(
                "SELECT DISTINCT document_id::text FROM visuals"
            )).fetchall()
            done_doc_ids = {r[0] for r in done_rows}

    _FIG_NUM_RE = _re.compile(
        r'(?:Fig(?:ure)?|Table|Eq(?:uation)?)\s*\.?\s*(\d+)',
        _re.IGNORECASE,
    )

    def _join_caption(c) -> str:
        """MinerU returns captions as lists of strings. Normalize to str."""
        if c is None:
            return ""
        if isinstance(c, list):
            return " ".join(str(x) for x in c if x).strip()
        return str(c).strip()

    # Phase 54.6.213 (roadmap 3.1.6 Phase 3) — MinerU 2.5-Pro emits
    # in-table image recognition: images embedded inside table cells
    # surface either as `<img src="...">` tags in the HTML table_body
    # or as a `table_images` / `embedded_images` list on the block.
    # Pre-fix, extract-visuals only walked top-level blocks, so every
    # such image was silently dropped. Both shapes are handled here
    # to be tolerant of MinerU version drift.
    _TABLE_IMG_SRC_RE = _re.compile(
        r'<img[^>]+src=["\']([^"\']+)["\']', _re.IGNORECASE,
    )

    # Phase 54.6.214 (roadmap 3.1.6 Phase 5 / closes 3.5.2) — MinerU
    # 2.5-Pro's structured per-figure output (chart parsing, image
    # analysis) lands in one of several block fields depending on
    # version. Capture whichever is present into visuals.literal_caption
    # so the synthesis caption (ai_caption, qwen2.5vl:32b) can be
    # layered on top of a model-agnostic literal description. Pre-
    # VLM-Pro blocks have none of these keys → literal_caption stays
    # NULL and downstream consumers graceful-degrade to ai_caption.
    _LITERAL_KEYS = (
        "chart_description",   # MinerU 2.5-Pro chart parser
        "image_analysis",      # MinerU 2.5-Pro image analysis
        "image_description",   # alternate naming
        "figure_description",  # alternate naming
        "literal_caption",     # future-proofing
    )

    def _extract_literal_caption(block: dict) -> str | None:
        for key in _LITERAL_KEYS:
            val = block.get(key)
            if not val:
                continue
            if isinstance(val, list):
                joined = " ".join(str(x) for x in val if x).strip()
                if joined:
                    return joined[:4000]
            elif isinstance(val, str) and val.strip():
                return val.strip()[:4000]
        return None

    def _extract_in_table_images(block: dict) -> list[dict]:
        """Return zero or more in-table-image records for this table
        block. Each record is a partial visuals row (kind, content,
        caption, asset_path, surrounding_text — block_idx + figure_num
        are filled by the caller). The `surrounding_text` carries the
        parent table's caption so downstream caption models have
        enough context to reason about the embedded image."""
        found: list[dict] = []
        parent_caption = _join_caption(
            block.get("table_caption") or block.get("caption")
        )

        # Shape A — explicit list of in-table images.
        for key in ("table_images", "embedded_images", "inner_images"):
            items = block.get(key)
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                img_path = it.get("img_path") or it.get("path") or ""
                img_caption = _join_caption(
                    it.get("caption") or it.get("alt")
                )
                found.append({
                    "kind": "table_image",
                    "content": (img_caption or parent_caption)[:2000],
                    "caption": img_caption[:1000] if img_caption else None,
                    "asset_path": str(img_path) if img_path else None,
                    "surrounding_text": parent_caption[:500],
                })

        # Shape B — <img> tags embedded in the HTML body.
        table_html = block.get("table_body") or block.get("html") or ""
        if isinstance(table_html, str) and "<img" in table_html.lower():
            for src in _TABLE_IMG_SRC_RE.findall(table_html):
                if not src:
                    continue
                # Skip dupes against Shape A (same src already recorded).
                if any(r.get("asset_path") == src for r in found):
                    continue
                found.append({
                    "kind": "table_image",
                    "content": parent_caption[:2000],
                    "caption": None,
                    "asset_path": src,
                    "surrounding_text": parent_caption[:500],
                })

        return found

    extracted = 0
    skipped = 0
    papers_done = 0

    for doc_id, file_hash in rows:
        if doc_id in done_doc_ids:
            skipped += 1
            continue

        # Find content_list.json
        output_dir = settings.mineru_output_dir / doc_id
        if not output_dir.exists():
            continue

        content_list_path = None
        for root_d, _dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith("_content_list.json") or f == "content_list.json":
                    content_list_path = Path(root_d) / f
                    break
            if content_list_path:
                break

        if not content_list_path or not content_list_path.exists():
            continue

        try:
            content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        visuals_batch: list[dict] = []
        prev_text = ""

        for idx, block in enumerate(content_list):
            btype = block.get("type", "")

            if btype == "text":
                prev_text = (block.get("text") or "")[:500]
                continue

            if btype == "table":
                table_body = (block.get("table_body") or block.get("html") or "")
                caption = _join_caption(
                    block.get("table_caption") or block.get("caption")
                )
                fig_match = _FIG_NUM_RE.search(caption or prev_text)
                visuals_batch.append({
                    "document_id": doc_id, "kind": "table",
                    "content": str(table_body)[:10000],
                    "caption": caption[:1000],
                    "asset_path": None,
                    "block_idx": idx,
                    "figure_num": fig_match.group(0) if fig_match else None,
                    "surrounding_text": prev_text,
                })

                # Phase 54.6.213 (roadmap 3.1.6 Phase 3) — MinerU-Pro
                # surfaces images embedded inside tables. We persist
                # each as its own `table_image` row keyed to the same
                # block_idx as the parent table so retrieval can join
                # them back together. figure_num inherited from the
                # parent table's caption when present.
                for sub in _extract_in_table_images(block):
                    sub["document_id"] = doc_id
                    sub["block_idx"] = idx
                    sub["figure_num"] = (
                        fig_match.group(0) if fig_match else None
                    )
                    sub.setdefault("literal_caption", None)
                    visuals_batch.append(sub)

            elif btype == "equation":
                latex = block.get("text") or block.get("latex") or ""
                visuals_batch.append({
                    "document_id": doc_id, "kind": "equation",
                    "content": str(latex)[:5000],
                    "caption": None,
                    "asset_path": None,
                    "block_idx": idx,
                    "figure_num": None,
                    "surrounding_text": prev_text,
                })

            elif btype == "image":
                img_path = block.get("img_path") or ""
                caption = _join_caption(
                    block.get("image_caption") or block.get("caption")
                )
                fig_match = _FIG_NUM_RE.search(caption or prev_text)
                visuals_batch.append({
                    "document_id": doc_id, "kind": "figure",
                    "content": caption[:2000],
                    "caption": caption[:1000],
                    "asset_path": str(img_path) if img_path else None,
                    "block_idx": idx,
                    "figure_num": fig_match.group(0) if fig_match else None,
                    "surrounding_text": prev_text,
                    "literal_caption": _extract_literal_caption(block),
                })

            elif btype == "chart":
                # Phase 54.6.62 — MinerU 2.5 emits a distinct `chart`
                # block type for plot-like images (bar charts, line plots,
                # scatter, etc.), separate from generic `image`. Pre-fix,
                # this dispatch ignored `chart` entirely, which silently
                # dropped ~65% of visual elements on corpora dominated
                # by quantitative papers. See the 2026-04-18 audit in
                # Phase 54.6.62 writeup.
                img_path = block.get("img_path") or ""
                caption = _join_caption(
                    block.get("chart_caption") or block.get("caption")
                )
                fig_match = _FIG_NUM_RE.search(caption or prev_text)
                visuals_batch.append({
                    "document_id": doc_id, "kind": "chart",
                    "content": caption[:2000],
                    "caption": caption[:1000],
                    "asset_path": str(img_path) if img_path else None,
                    "block_idx": idx,
                    "figure_num": fig_match.group(0) if fig_match else None,
                    "surrounding_text": prev_text,
                    "literal_caption": _extract_literal_caption(block),
                })

            elif btype == "code":
                code_body = block.get("text") or block.get("code_body") or ""
                visuals_batch.append({
                    "document_id": doc_id, "kind": "code",
                    "content": str(code_body)[:10000],
                    "caption": None,
                    "asset_path": None,
                    "block_idx": idx,
                    "figure_num": None,
                    "surrounding_text": prev_text,
                })

        if visuals_batch:
            # Phase 54.6.63 — strip NUL (0x00) bytes from all text fields
            # before insert. MinerU occasionally emits \x00 inside decoded
            # LaTeX for unusual character encodings (e.g. equation content
            # that went through a corrupted font map). PostgreSQL text
            # columns cannot store \x00 (it's the C-string terminator),
            # so the whole row insert fails with "A string literal cannot
            # contain NUL (0x00) characters", which pre-54.6.63 caused
            # the entire paper's batch to be skipped — that's what the
            # 2026-04-18 audit saw on 3 papers during the chart backfill.
            for v in visuals_batch:
                for k in ("content", "caption", "asset_path",
                          "figure_num", "surrounding_text",
                          "literal_caption"):
                    val = v.get(k)
                    if isinstance(val, str) and "\x00" in val:
                        v[k] = val.replace("\x00", "")
            try:
                with get_session() as session:
                    if force:
                        session.execute(sql_text(
                            "DELETE FROM visuals WHERE document_id::text = :did"
                        ), {"did": doc_id})
                    for v in visuals_batch:
                        v.setdefault("literal_caption", None)
                        session.execute(sql_text("""
                            INSERT INTO visuals
                                (document_id, kind, content, caption, asset_path,
                                 block_idx, figure_num, surrounding_text,
                                 literal_caption)
                            VALUES
                                (CAST(:document_id AS uuid), :kind, :content,
                                 :caption, :asset_path, :block_idx, :figure_num,
                                 :surrounding_text, :literal_caption)
                        """), v)
                    session.commit()
                extracted += len(visuals_batch)
                papers_done += 1
            except Exception as exc:
                console.print(f"  [red]skip {doc_id[:8]}:[/red] {exc}")

    console.print(
        f"\n[green]✓ Extracted {extracted} visuals from {papers_done} papers[/green]"
        f"  [dim]({skipped} already done, {total - len(rows)} over limit)[/dim]"
    )


# ── link-visual-mentions (Phase 54.6.138 — RESEARCH.md §7.X signal 3) ─────────

@app.command(name="link-visual-mentions")
def link_visual_mentions_cmd(
    doc_id: str = typer.Option(
        None, "--doc-id",
        help="Restrict to one paper (UUID or prefix). Useful for smoke "
             "testing on a single document before a bulk run.",
    ),
    limit: int = typer.Option(
        0, "--limit",
        help="Process at most N papers after the doc-id filter (0 = all).",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Re-link even visuals that already have mention_paragraphs "
             "populated. Use when the regex or extraction heuristics change.",
    ),
):
    """Phase 54.6.138 — link body-text paragraphs to each visual.

    For every ``visuals`` row with a numeric ``figure_num``, scans the
    source paper's ``content_list.json`` for body paragraphs that
    reference that number (``Fig. 3``, ``Figure 3``, ``Table 2``,
    ``Eq. 5``) and persists them as JSONB on
    ``visuals.mention_paragraphs``.

    This is the infrastructure half of the visuals-in-writer feature
    (docs/RESEARCH.md §7.X, signal 3). Per SciCap+ findings, the
    mention-paragraph is the single strongest retrieval signal for
    matching a figure to target draft prose — stronger than the caption
    or the image itself — because it carries the author's rhetorical
    framing of why the figure was cited at that point.

    Idempotent: skips visuals that already have ``mention_paragraphs``
    unless ``--force``. Safe to re-run. Runs against existing
    content_list.json files — no re-ingestion needed.

    Examples:

      sciknow db link-visual-mentions                      # link all papers
      sciknow db link-visual-mentions --doc-id abc123      # smoke test one paper
      sciknow db link-visual-mentions --limit 10           # first 10 papers
      sciknow db link-visual-mentions --force              # re-link all

    After this runs, ``visuals.mention_paragraphs`` is either ``[]``
    (no body references found — e.g. decorative figures) or a list of
    ``{block_idx, text, context_before}`` entries ordered by paper
    position. Downstream: the 5-signal write-loop ranker (docs/RESEARCH.md
    §7.X.3) uses these against draft sentences as its mention-paragraph
    alignment signal.
    """
    from sciknow.cli import preflight
    preflight()
    from sciknow.core import visuals_mentions as vm

    # Resolve doc-id prefix if given
    target_doc: str | None = None
    if doc_id:
        from sqlalchemy import text as _sql
        from sciknow.storage.db import get_session as _gs
        with _gs() as session:
            row = session.execute(_sql(
                "SELECT id::text FROM documents WHERE id::text LIKE :q LIMIT 1"
            ), {"q": f"{doc_id.strip()}%"}).fetchone()
        if not row:
            console.print(f"[red]No document matches {doc_id!r}.[/red]")
            raise typer.Exit(1)
        target_doc = row[0]
        console.print(f"[dim]Resolved --doc-id to {target_doc}[/dim]\n")

    if target_doc:
        n = vm.link_visuals_for_doc(target_doc, force=force)
        console.print(
            f"[green]✓ Linked {n} visual row(s)[/green] for document "
            f"{target_doc[:12]}…"
        )
        return

    console.print(
        f"[bold]Linking mention-paragraphs[/bold] "
        f"(force={force}, limit={limit or 'all'})…\n"
    )
    total_rows = 0
    total_docs = 0
    for did, n_updated in vm.link_visuals_for_corpus(
        limit=limit, force=force,
    ):
        total_docs += 1
        total_rows += n_updated
        if n_updated:
            console.print(
                f"  [dim]{did[:12]}…[/dim]  [green]{n_updated}[/green] visual(s) linked"
            )

    console.print(
        f"\n[bold]Done.[/bold] {total_docs} paper(s) processed, "
        f"{total_rows} visual row(s) updated."
    )


# ── caption-visuals (Phase 54.6.72 — #1) ──────────────────────────────────────

@app.command(name="caption-visuals")
def caption_visuals_cmd(
    model: str = typer.Option(
        None, "--model",
        help="Vision-LLM tag to use via Ollama. Default: "
             "settings.visuals_caption_model if set (let the 54.6.74 "
             "VLM sweep winner persist via .env) else qwen2.5vl:32b. "
             "(~19 GB Q4, fits a 3090 with the main LLM unloaded — "
             "strongest open VLM that fits for document+chart quality). "
             "For faster / lower-VRAM: qwen2.5vl:7b (~6 GB, co-resident "
             "with an LLM). Other options: internvl3:14b, llama3.2-vision:11b, "
             "minicpm-v:8b. `ollama ps` to unload other models; "
             "`ollama pull <model>` to fetch.",
    ),
    kind: str = typer.Option(
        "figure,chart", "--kind",
        help="Comma-separated kinds to caption. Only image-bearing kinds "
             "(figure, chart) produce useful captions; everything else is "
             "skipped even if listed.",
    ),
    limit: int = typer.Option(
        0, "--limit", "-n",
        help="Max visuals to caption this run (0 = all pending).",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Recaption even rows that already have ai_caption set.",
    ),
    min_prob: float = typer.Option(
        0.0, "--min-prob",
        help="If set, skip rows whose existing caption is short — forces "
             "re-caption of thin placeholders without re-doing good ones.",
    ),
):
    """Phase 54.6.72 (#1) — run a vision LLM over every image visual and
    store a one-paragraph caption in the `ai_caption` column.

    Hydrates the 9,988 silent MinerU-extracted figures + charts for
    semantic retrieval and for real previews in the wiki Visuals tab.

    Model is invoked via Ollama's image endpoint. Requires the model
    to already be pulled — this command does NOT auto-pull (the pull
    is a ~6-20 GB download and we want it explicit).

    Quality note: default flipped from qwen2.5vl:7b → qwen2.5vl:32b in
    Phase 54.6.73 after the user directive "always optimize for best
    quality". On the 3090 the 32B variant (Q4 quant, ~19 GB VRAM) fits
    only when other models are unloaded (``ollama stop <current>``); it
    runs ~3-4× slower than 7B but produces materially better captions
    for scientific plots and tables (MinerU's own PDF parser is
    Qwen2-VL-derived, so Qwen2.5-VL inherits the document lineage).
    Pass ``--model qwen2.5vl:7b`` to trade quality for speed /
    co-residence with the LLM.

    Examples:

      ollama pull qwen2.5vl:32b                       # recommended
      ollama stop qwen3:30b-a3b-instruct-2507-q4_K_M  # free VRAM
      sciknow db caption-visuals                       # caption all pending
      sciknow db caption-visuals -n 20 --force         # re-caption first 20
      sciknow db caption-visuals --kind figure         # figures only
      sciknow db caption-visuals --model qwen2.5vl:7b  # faster, lower quality
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    import ollama
    from sciknow.config import settings
    from sciknow.core.visuals_caption import (
        PROMPT_SYSTEM, PROMPT_USER, resolve_asset_path,
    )
    from sciknow.storage.db import get_session
    from sqlalchemy import text as sql_text

    kinds = [k.strip() for k in kind.split(",") if k.strip()]
    if not kinds:
        console.print("[red]--kind must be a non-empty comma-separated list[/red]")
        raise typer.Exit(2)

    # Phase 54.6.74 — resolve the effective model: explicit --model
    # wins, else settings.visuals_caption_model (set by .env after
    # the VLM sweep picks a winner), else the CLI default.
    if model is None:
        model = settings.visuals_caption_model or "qwen2.5vl:32b"

    # Sanity-check model availability up front so we fail fast.
    client = ollama.Client(host=settings.ollama_host)
    try:
        installed = {m.model for m in client.list().models}
    except Exception as exc:
        console.print(f"[red]Ollama unreachable:[/red] {exc}")
        raise typer.Exit(1)
    if model not in installed:
        console.print(
            f"[red]Model {model!r} not installed.[/red] Run:\n"
            f"  [bold]ollama pull {model}[/bold]\n"
            f"and retry."
        )
        raise typer.Exit(1)

    # Fetch pending rows.
    kind_ph = ", ".join(f":k{i}" for i, _ in enumerate(kinds))
    kind_params = {f"k{i}": k for i, k in enumerate(kinds)}
    where = [f"v.kind IN ({kind_ph})", "v.asset_path IS NOT NULL"]
    if not force:
        where.append("v.ai_caption IS NULL")
    with get_session() as session:
        rows = session.execute(sql_text(f"""
            SELECT v.id::text, v.document_id::text, v.kind,
                   v.asset_path, v.caption, v.figure_num
            FROM visuals v
            WHERE {' AND '.join(where)}
            ORDER BY v.created_at
            {('LIMIT :lim' if limit else '')}
        """), {**kind_params, **({"lim": limit} if limit else {})}).fetchall()

    total = len(rows)
    if total == 0:
        console.print("[green]Nothing to caption — all matching visuals already have ai_caption.[/green]")
        return

    console.print(
        f"Captioning [bold]{total}[/bold] visual(s) with [cyan]{model}[/cyan]…"
    )

    captioned = 0
    skipped = 0
    t0 = time.monotonic()
    for idx, (vid, doc_id, vkind, asset_path, existing_caption, fig_num) in enumerate(rows, 1):
        img_path = resolve_asset_path(doc_id, asset_path)
        if img_path is None:
            skipped += 1
            console.print(f"  [dim][{idx}/{total}][/dim] "
                          f"[yellow]⊘ SKIP[/yellow]  "
                          f"{fig_num or vkind} · image file missing on disk")
            continue

        # Compose a short, targeted prompt. Existing MinerU caption is
        # usually empty or the raw "Figure 1" line — we pass it as
        # context anyway so the VLM can refine rather than ignore it.
        user_prompt = PROMPT_USER.format(
            kind=vkind,
            existing_caption=(existing_caption or "").strip() or "(none)",
        )
        try:
            resp = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user", "content": user_prompt,
                     "images": [str(img_path)]},
                ],
                options={"temperature": 0.2, "num_predict": 300},
                keep_alive=-1,
            )
            ai_caption = (resp.get("message") or {}).get("content", "").strip()
        except Exception as exc:
            skipped += 1
            console.print(f"  [dim][{idx}/{total}][/dim] "
                          f"[red]⚠ FAIL[/red]  {fig_num or vkind}  · {exc}")
            continue

        if not ai_caption or len(ai_caption) < 20:
            skipped += 1
            console.print(f"  [dim][{idx}/{total}][/dim] "
                          f"[yellow]⊘ SKIP[/yellow]  {fig_num or vkind}  · "
                          f"caption too short")
            continue

        with get_session() as session:
            session.execute(sql_text("""
                UPDATE visuals SET
                  ai_caption = :cap,
                  ai_caption_model = :mdl,
                  ai_captioned_at = now()
                WHERE id::text = :vid
            """), {"cap": ai_caption.replace("\x00", ""),
                   "mdl": model, "vid": vid})
            session.commit()
        captioned += 1
        preview = ai_caption[:70].replace("\n", " ")
        console.print(
            f"  [dim][{idx}/{total}][/dim] "
            f"[green]✓ CAP[/green]  {fig_num or vkind}  · {preview}"
        )
        if idx % 50 == 0:
            rate = idx / max(0.01, time.monotonic() - t0)
            eta = (total - idx) / max(0.01, rate)
            console.print(f"  [dim]… {rate:.2f}/s, eta {eta:.0f}s[/dim]")

    console.print(
        f"\n[green]✓ Captioned {captioned}[/green] · "
        f"[yellow]skipped {skipped}[/yellow] · "
        f"[dim]{time.monotonic() - t0:.1f}s wall[/dim]"
    )


# ── embed-visuals (Phase 54.6.82 — #11 follow-up) ────────────────────────

@app.command(name="embed-visuals")
def embed_visuals_cmd(
    kind: str = typer.Option(
        "equation,figure,chart", "--kind",
        help="Comma-separated visual kinds to embed. Default covers "
             "equation paraphrases (54.6.78) + figure/chart AI captions "
             "(54.6.72).",
    ),
    limit: int = typer.Option(
        0, "--limit", "-n",
        help="Max visuals to embed this run (0 = all pending).",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Re-embed even rows that already have qdrant_point_id set.",
    ),
):
    """Phase 54.6.82 — embed visuals' ai_caption text into the project's
    visuals Qdrant collection so paraphrases + captions become
    retrievable.

    Closes the loop on #11 (equation paraphrase) and #1 (figure captions):
    both land prose in ``visuals.ai_caption``, but nothing was indexing
    that prose for similarity search. This command embeds the caption
    with bge-m3 (dense + sparse) and upserts into the project's
    visuals Qdrant collection (created by `db init`), storing the
    resulting point ID back on the row so the API can cross-reference.

    Uses the existing bge-m3 embedder (shares the model already in
    VRAM if you've done any retrieval today). ~300-500 ms per visual
    cold, ~5-10 ms warm.

    Examples:

      sciknow db embed-visuals                        # all with ai_caption
      sciknow db embed-visuals --kind equation -n 50  # test subset
      sciknow db embed-visuals --force                # re-embed all
    """
    from sciknow.cli import preflight
    preflight()

    from sciknow.ingestion.embedder import embed_to_visuals_collection
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client as _get_qdrant
    from sqlalchemy import text as sql_text

    kinds = [k.strip() for k in kind.split(",") if k.strip()]
    if not kinds:
        console.print("[red]--kind must be a non-empty comma-separated list[/red]")
        raise typer.Exit(2)
    kind_ph = ", ".join(f":k{i}" for i, _ in enumerate(kinds))
    kind_params = {f"k{i}": k for i, k in enumerate(kinds)}

    # Phase 54.6.214 (roadmap 3.1.6 Phase 5) — admit rows that have
    # EITHER ai_caption OR literal_caption OR table_summary. Previously
    # we required ai_caption specifically, which would have left
    # literal-only rows (post-VLM-Pro ingest, before caption-visuals
    # has run) un-embedded forever.
    where = [
        f"v.kind IN ({kind_ph})",
        "(v.ai_caption IS NOT NULL "
        " OR v.literal_caption IS NOT NULL "
        " OR v.table_summary IS NOT NULL)",
        "(length(COALESCE(v.ai_caption, '')) >= 20 "
        " OR length(COALESCE(v.literal_caption, '')) >= 20 "
        " OR length(COALESCE(v.table_summary, '')) >= 20)",
    ]
    if not force:
        where.append("v.qdrant_point_id IS NULL")

    with get_session() as session:
        rows = session.execute(sql_text(f"""
            SELECT v.id::text, v.document_id::text, v.kind,
                   -- 54.6.109: for tables, prefer the parsed
                   -- table_summary over ai_caption (tables don't
                   -- get captioned).
                   -- 54.6.214: when both literal_caption (MinerU-Pro
                   -- per-figure description) and ai_caption
                   -- (qwen2.5vl:32b synthesis) are present, embed
                   -- their concatenation — literal anchors the
                   -- retrieval in what the image shows, synthesis
                   -- anchors it in what the paper *claims* about
                   -- that image. Either alone is a useful fallback.
                   CASE WHEN v.kind = 'table'
                          AND COALESCE(v.table_summary, '') <> ''
                        THEN v.table_summary
                        WHEN v.literal_caption IS NOT NULL
                          AND COALESCE(v.ai_caption, '') <> ''
                        THEN v.literal_caption || ' ' || v.ai_caption
                        WHEN v.literal_caption IS NOT NULL
                        THEN v.literal_caption
                        ELSE v.ai_caption
                   END AS embed_text,
                   v.figure_num, v.caption
            FROM visuals v
            WHERE {' AND '.join(where)}
            ORDER BY v.created_at
            {('LIMIT :lim' if limit else '')}
        """), {**kind_params, **({"lim": limit} if limit else {})}).fetchall()

    total = len(rows)
    if total == 0:
        console.print(
            "[green]Nothing to embed — every matching visual either has "
            "no ai_caption or is already in Qdrant.[/green]"
        )
        return

    console.print(f"Embedding [bold]{total}[/bold] visual(s) into the "
                  f"visuals Qdrant collection…")

    qdrant = _get_qdrant()
    done = 0
    skipped = 0
    t0 = time.monotonic()
    for idx, (vid, doc_id, vkind, ai_cap, fig_num, orig_cap) in enumerate(rows, 1):
        try:
            point_id = embed_to_visuals_collection(
                ai_cap,
                payload={
                    "visual_id": vid,
                    "document_id": doc_id,
                    "kind": vkind,
                    "figure_num": fig_num or "",
                    "original_caption": (orig_cap or "")[:500],
                    "ai_caption": ai_cap[:1000],
                },
                qdrant_client=qdrant,
            )
        except Exception as exc:
            skipped += 1
            console.print(f"  [dim][{idx}/{total}][/dim] [red]⚠ FAIL[/red]  {vkind}  · {exc}")
            continue
        if point_id is None:
            skipped += 1
            continue
        with get_session() as session:
            session.execute(sql_text("""
                UPDATE visuals SET qdrant_point_id = CAST(:pid AS uuid)
                WHERE id::text = :vid
            """), {"pid": str(point_id), "vid": vid})
            session.commit()
        done += 1
        if idx % 50 == 0:
            rate = idx / max(0.01, time.monotonic() - t0)
            eta = (total - idx) / max(0.01, rate)
            console.print(
                f"  [dim][{idx}/{total}] … {rate:.1f}/s, "
                f"eta {int(eta/60)}m {int(eta%60)}s[/dim]"
            )

    console.print(
        f"\n[green]✓ Embedded {done}[/green] · "
        f"[yellow]skipped {skipped}[/yellow] · "
        f"[dim]{time.monotonic() - t0:.1f}s wall[/dim]"
    )


# ── classify-papers (Phase 54.6.80 — #10) ────────────────────────────────

@app.command(name="classify-papers")
def classify_papers_cmd(
    model: str = typer.Option(
        None, "--model",
        help="LLM for classification. Default: settings.llm_fast_model.",
    ),
    limit: int = typer.Option(
        0, "--limit", "-n",
        help="Max papers to classify this run (0 = all pending).",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Re-classify rows that already have paper_type set.",
    ),
):
    """Phase 54.6.80 (#10) — classify each paper into one of:
    peer_reviewed / preprint / thesis / editorial / opinion / policy /
    book_chapter / unknown.

    Populates `paper_metadata.paper_type` + `.paper_type_confidence` +
    `.paper_type_model`. Enables retrieval filtering (only peer-reviewed
    for factual queries) and default downweighting for opinion / policy
    on `ask` queries.

    Uses the abstract + first 2000 chars of content + bibliographic
    metadata (journal, publisher, DOI) as classifier input. ~2-3s per
    paper on LLM_FAST_MODEL; ~30-40 min for 676 papers on a 3090.
    Interruptible — re-run picks up where you left off (skips rows
    with paper_type set unless --force).

    Examples:

      sciknow db classify-papers            # all pending
      sciknow db classify-papers -n 20      # first 20
      sciknow db classify-papers --force    # re-classify all
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    from sciknow.config import settings
    from sciknow.core.paper_type import classify_paper
    from sciknow.storage.db import get_session
    from sqlalchemy import text as sql_text

    if model is None:
        model = settings.llm_fast_model

    where = ["pm.title IS NOT NULL"]
    if not force:
        where.append("pm.paper_type IS NULL")

    with get_session() as session:
        rows = session.execute(sql_text(f"""
            SELECT pm.id::text, pm.title, pm.journal, pm.publisher,
                   pm.year, pm.doi, pm.abstract,
                   COALESCE(
                     (SELECT ps.content FROM paper_sections ps
                        WHERE ps.document_id = pm.document_id
                        ORDER BY ps.section_index LIMIT 1),
                     ''
                   ) AS first_section
            FROM paper_metadata pm
            WHERE {' AND '.join(where)}
            ORDER BY pm.year DESC NULLS LAST
            {('LIMIT :lim' if limit else '')}
        """), ({"lim": limit} if limit else {})).fetchall()

    total = len(rows)
    if total == 0:
        console.print(
            "[green]Nothing to classify — every paper already has "
            "paper_type (or pass --force).[/green]"
        )
        return

    console.print(
        f"Classifying [bold]{total}[/bold] paper(s) with [cyan]{model}[/cyan]…"
    )

    done = 0
    failed = 0
    by_type: dict[str, int] = {}
    t0 = time.monotonic()
    for idx, (pid, title, journal, publisher, year, doi, abstract, content) in enumerate(rows, 1):
        result = classify_paper(
            title=title or "",
            journal=journal or "",
            publisher=publisher or "",
            year=year,
            doi=doi or "",
            abstract=abstract or "",
            content=content or "",
            model=model,
        )
        if result is None:
            failed += 1
            console.print(
                f"  [dim][{idx}/{total}][/dim] [red]⚠ FAIL[/red]  "
                f"{(title or '(no title)')[:70]}"
            )
            continue
        with get_session() as session:
            session.execute(sql_text("""
                UPDATE paper_metadata SET
                  paper_type = :t,
                  paper_type_confidence = :c,
                  paper_type_model = :m
                WHERE id::text = :pid
            """), {"t": result.paper_type, "c": result.confidence,
                   "m": model, "pid": pid})
            session.commit()
        done += 1
        by_type[result.paper_type] = by_type.get(result.paper_type, 0) + 1
        tag_color = {
            "peer_reviewed": "green", "preprint": "cyan", "thesis": "cyan",
            "editorial": "yellow", "opinion": "yellow",
            "policy": "magenta", "book_chapter": "blue", "unknown": "dim",
        }.get(result.paper_type, "white")
        console.print(
            f"  [dim][{idx}/{total}][/dim] [{tag_color}]{result.paper_type:<14}[/{tag_color}] "
            f"({result.confidence:.2f})  {(title or '')[:65]}"
        )
        if idx % 50 == 0:
            rate = idx / max(0.01, time.monotonic() - t0)
            eta = (total - idx) / max(0.01, rate)
            console.print(
                f"  [dim]… {rate:.2f}/s, eta {int(eta/60)}m {int(eta%60)}s[/dim]"
            )

    console.print(
        f"\n[green]✓ Classified {done}[/green] · "
        f"[red]failed {failed}[/red] · "
        f"[dim]{time.monotonic() - t0:.1f}s wall[/dim]"
    )
    if by_type:
        console.print("\n[bold]By type:[/bold]")
        for t in ("peer_reviewed", "preprint", "thesis", "book_chapter",
                  "editorial", "opinion", "policy", "unknown"):
            n = by_type.get(t, 0)
            if n:
                console.print(f"  {t:<15} {n}")


# ── paraphrase-equations (Phase 54.6.78 — #11) ───────────────────────────

@app.command(name="paraphrase-equations")
def paraphrase_equations_cmd(
    model: str = typer.Option(
        None, "--model",
        help="Text LLM for paraphrasing. Default: settings.llm_fast_model "
             "(qwen3:30b-a3b-instruct-2507-q4_K_M by default).",
    ),
    limit: int = typer.Option(
        0, "--limit", "-n",
        help="Max equations to paraphrase this run (0 = all pending).",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Re-paraphrase rows that already have ai_caption set.",
    ),
):
    """Phase 54.6.78 (#11) — paraphrase MinerU-extracted equations into
    one-sentence natural-language descriptions for retrieval indexing.

    bge-m3 embeds raw LaTeX poorly because the tokenizer fragments
    commands like ``\\frac`` into characters — the resulting embedding
    drifts from the equation's meaning. A one-sentence paraphrase
    ("The slope of outgoing longwave radiation with respect to global
    surface temperature, 2.93 ± 0.3 W/m²·K") embeds far better.

    Uses LLM_FAST_MODEL (qwen3:30b-a3b-instruct-2507 by default),
    ~1-2s per equation. For 4,687 equations: ~2-3 hours on a 3090.
    Interruptible — re-run continues where you left off because the
    row's ai_caption gets populated on write. Trivial equations
    (length < 3 characters after cleanup, e.g. `a=b`) are skipped.

    Stored in the existing ai_caption column (54.6.72 migration); the
    text-LLM-vs-VLM distinction is made by `kind`: equation kind with
    ai_caption set means "paraphrased here", figure/chart means
    "image-captioned in caption-visuals".

    Examples:

      sciknow db paraphrase-equations                    # all pending
      sciknow db paraphrase-equations -n 50              # first 50
      sciknow db paraphrase-equations --force            # re-do all
      sciknow db paraphrase-equations --model gemma3:27b-it-qat
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    from sciknow.config import settings
    from sciknow.core.equation_paraphrase import paraphrase_equation
    from sciknow.storage.db import get_session
    from sqlalchemy import text as sql_text

    if model is None:
        model = settings.llm_fast_model

    where = ["v.kind = 'equation'", "v.content IS NOT NULL",
             "length(v.content) >= 5"]
    if not force:
        where.append("v.ai_caption IS NULL")

    with get_session() as session:
        rows = session.execute(sql_text(f"""
            SELECT v.id::text, v.content, COALESCE(v.surrounding_text, '')
            FROM visuals v
            WHERE {' AND '.join(where)}
            ORDER BY v.created_at
            {('LIMIT :lim' if limit else '')}
        """), ({"lim": limit} if limit else {})).fetchall()

    total = len(rows)
    if total == 0:
        console.print("[green]Nothing to paraphrase — every equation "
                      "already has ai_caption (or pass --force).[/green]")
        return

    console.print(
        f"Paraphrasing [bold]{total}[/bold] equation(s) with "
        f"[cyan]{model}[/cyan]…"
    )

    done = 0
    skipped = 0
    t0 = time.monotonic()
    for idx, (vid, latex, ctx) in enumerate(rows, 1):
        para = paraphrase_equation(latex, ctx, model=model)
        if para is None:
            skipped += 1
            console.print(
                f"  [dim][{idx}/{total}][/dim] [yellow]⊘ SKIP[/yellow]  "
                f"trivial or empty output"
            )
            continue
        with get_session() as session:
            session.execute(sql_text("""
                UPDATE visuals SET
                  ai_caption = :cap,
                  ai_caption_model = :mdl,
                  ai_captioned_at = now()
                WHERE id::text = :vid
            """), {"cap": para.replace("\x00", ""),
                   "mdl": model, "vid": vid})
            session.commit()
        done += 1
        preview = para[:80]
        console.print(
            f"  [dim][{idx}/{total}][/dim] [green]✓ PARA[/green]  {preview}"
        )
        if idx % 50 == 0:
            rate = idx / max(0.01, time.monotonic() - t0)
            eta = (total - idx) / max(0.01, rate)
            console.print(
                f"  [dim]… {rate:.2f}/s, eta {int(eta/60)}m {int(eta%60)}s[/dim]"
            )

    console.print(
        f"\n[green]✓ Paraphrased {done}[/green] · "
        f"[yellow]skipped {skipped}[/yellow] · "
        f"[dim]{time.monotonic() - t0:.1f}s wall[/dim]"
    )


# ── parse-tables (Phase 54.6.106 — #2) ─────────────────────────────────────

@app.command(name="parse-tables")
def parse_tables_cmd(
    model: str = typer.Option(
        None, "--model",
        help="Text LLM for table parsing. Default: settings.llm_fast_model.",
    ),
    limit: int = typer.Option(
        0, "--limit", "-n",
        help="Max tables to parse this run (0 = all pending).",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Re-parse rows that already have table_parsed_at set.",
    ),
):
    """Phase 54.6.106 (#2) — parse MinerU HTML tables into a semantic
    summary + column headers + shape.

    1,501 tables live as raw HTML in ``visuals.content`` and currently
    only match substring queries. This CLI asks the fast LLM to emit:

      - ``table_title``   — one-line inferred title
      - ``table_headers`` — JSONB array of column headers
      - ``table_summary`` — 1-3 sentence semantic summary
      - ``table_n_rows`` / ``table_n_cols`` — shape from the raw HTML

    Output powers: the Visuals modal's table cards (subtitle shows
    summary instead of raw HTML preview), future table-specific
    retrieval, and optional embedding of ``table_summary`` via
    ``db embed-visuals`` for semantic search.

    Examples:

      sciknow db parse-tables                  # all pending
      sciknow db parse-tables -n 20            # first 20
      sciknow db parse-tables --force          # re-parse everything
      sciknow db parse-tables --model qwen3:30b-a3b-instruct-2507-q4_K_M
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    from sciknow.core.table_parse import parse_pending_tables

    t0 = time.monotonic()

    def _progress(i: int, total: int, vid: str, outcome: str) -> None:
        short = vid[:8]
        mark = ("[green]✓[/green]" if outcome == "parsed"
                else "[yellow]⊘[/yellow]" if outcome == "empty"
                else "[red]✗[/red]")
        console.print(
            f"  [dim][{i}/{total}][/dim] {mark}  {short}  {outcome[:120]}"
        )
        if i % 25 == 0 and total > 0:
            rate = i / max(0.01, time.monotonic() - t0)
            eta = (total - i) / max(0.01, rate)
            console.print(
                f"  [dim]… {rate:.2f}/s, eta {int(eta/60)}m {int(eta%60)}s[/dim]"
            )

    stats = parse_pending_tables(
        model=model, limit=(limit or None), force=force,
        progress_callback=_progress,
    )

    console.print(
        f"\n[green]✓ Parsed {stats['parsed']}[/green] · "
        f"[red]errors {stats['errors']}[/red] · "
        f"model [cyan]{stats['model']}[/cyan] · "
        f"[dim]{time.monotonic() - t0:.1f}s wall[/dim]"
    )


# ── caption-bench (Phase 54.6.226 — roadmap 3.5.1) ───────────────────────────


_CAPTION_GRADE_PROMPT = """You are grading an AI-generated caption \
for a figure or chart from a scientific paper. The caption was \
produced by a vision-LLM (qwen2.5vl:32b) and is intended for \
retrieval + downstream writing.

Figure kind: {kind}
Paper title: "{title}" ({year})

Original caption from the paper (a short human-authored label, \
usually 1 line; may be sparse — trust it as ground truth anyway):
"{original_caption}"

Surrounding body text (the paragraph just before the figure in \
the paper; may be empty):
"{surrounding_text}"

AI-generated caption (the thing you're grading):
"{ai_caption}"

Grade three things:

1. accuracy — does the AI caption describe what the original caption \
and surrounding text suggest the figure shows?
   * yes     — captures the main subject + key elements faithfully
   * partial — mostly right but misses or garbles one element
   * no      — describes something different from what the paper's \
text indicates

2. hallucination — does the AI caption invent details (specific \
numbers, axis labels, species names, places) that aren't supported \
by the original caption or surrounding text?
   * none   — strictly derived from what's known
   * minor  — adds plausible context that might or might not be \
real; low stakes
   * severe — states specific values or entities that look invented \
(e.g. gives axis numeric ranges when the paper's caption never \
mentioned them)

3. usefulness — is the caption useful for retrieval (dense + sparse) \
and for a writer looking for a figure to cite?
   * good   — conveys what the figure shows AND why it matters in \
the paper's argument (carries the rhetorical framing)
   * ok     — describes the figure literally; helps retrieval but not \
writing
   * bland  — generic filler ("A figure showing the relationship \
between X and Y"); provides little retrieval signal

Respond in JSON ONLY, no prose:
{{"accuracy": "yes|partial|no",
  "hallucination": "none|minor|severe",
  "usefulness": "good|ok|bland",
  "reason": "<one short sentence>"}}
"""


@app.command(name="caption-bench")
def caption_bench_cmd(
    n: int = typer.Option(
        30, "--n", "-n",
        help="Random sample size. 30 is enough for ±10% precision "
             "at 95% CI on the accuracy axis.",
    ),
    kind: str = typer.Option(
        "figure,chart", "--kind",
        help="Comma-separated visual kinds to sample. Default "
             "figure,chart (the image-bearing kinds that VLM "
             "captions). Tables don't carry ai_caption at the moment.",
    ),
    judge: str = typer.Option(
        "llm", "--judge",
        help="'llm' (LLM_FAST_MODEL text-only — compares AI caption "
             "against the paper's original caption + surrounding "
             "text; no image access; cheap), 'human' (interactive "
             "prompt), or 'both'.",
    ),
    model: str | None = typer.Option(
        None, "--model",
        help="Override the judge LLM. Default: settings.llm_fast_model.",
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir",
        help="Where to write the sample JSONL. Default: "
             "<data_dir>/caption_bench/bench-<iso-timestamp>.jsonl.",
    ),
    seed: int | None = typer.Option(
        None, "--seed",
        help="Random seed for deterministic sampling (debugging only).",
    ),
):
    """Phase 54.6.226 (roadmap 3.5.1) — bench ai_caption quality on figures + charts.

    Third bench harness alongside `wiki kg-sample` (54.6.218) and
    `db equation-bench` (54.6.222). Samples N random figure/chart
    rows from the visuals table, grades their ai_caption along three
    axes, and persists results to a timestamped JSONL for
    longitudinal tracking.

    Three-axis rubric:

      * accuracy       — does the AI caption match what the paper
                         says the figure shows?  (yes/partial/no)
      * hallucination  — does it invent labels/axes/numbers?
                         (none/minor/severe)
      * usefulness     — is it retrieval- and writing-useful, or
                         generic filler?  (good/ok/bland)

    Judge is text-only by default — it compares the AI caption
    against the paper's own short caption + surrounding body text.
    That misses hallucinations that happen to look plausible in
    text, but catches the common drift cases (invented axis ranges,
    wrong subject, wrong units). A VLM-judge mode that compares
    caption to image is a potential follow-on once we're past the
    Phase 4 re-ingest.

    Output JSONL (one record per visual):

      {{
        "visual_id": "<uuid>",
        "document_id": "<uuid>", "paper_title": "...", "year": 2024,
        "kind": "figure|chart",
        "original_caption": "...",
        "surrounding_text": "...",
        "ai_caption": "...",
        "ai_caption_model": "...",
        "accuracy": "yes|partial|no",
        "hallucination": "none|minor|severe",
        "usefulness": "good|ok|bland",
        "reason": "<judge's one-sentence rationale>",
        "judge": "llm-<model>|human",
        "graded_at": "<iso timestamp>"
      }}

    Examples:

      sciknow db caption-bench                     # 30 figures+charts
      sciknow db caption-bench --n 50              # bigger sample
      sciknow db caption-bench --kind chart        # charts only
      sciknow db caption-bench --judge human       # interactive
      sciknow db caption-bench --judge both        # LLM + human confirm
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    import json as _json
    import random as _random
    from datetime import datetime, timezone
    from sqlalchemy import text as sql_text
    from sciknow.config import settings
    from sciknow.storage.db import get_session

    if judge not in ("llm", "human", "both"):
        console.print(
            f"[red]Invalid --judge:[/red] {judge!r} — must be "
            "llm / human / both."
        )
        raise typer.Exit(2)

    kinds = [k.strip() for k in kind.split(",") if k.strip()]
    if not kinds:
        console.print("[red]--kind must be non-empty[/red]")
        raise typer.Exit(2)

    from sciknow.core.project import get_active_project
    active = get_active_project()
    out_root = output_dir or (active.data_dir / "caption_bench")
    out_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_root / f"bench-{ts}.jsonl"

    if seed is not None:
        _random.seed(seed)

    # Pull candidate pool via random sample.
    kind_ph = ", ".join(f":k{i}" for i, _ in enumerate(kinds))
    kind_params = {f"k{i}": k for i, k in enumerate(kinds)}

    with get_session() as session:
        rows = session.execute(sql_text(f"""
            SELECT v.id::text, v.document_id::text, v.kind,
                   v.caption, v.surrounding_text,
                   v.ai_caption, v.ai_caption_model,
                   pm.title, pm.year
            FROM visuals v
            LEFT JOIN paper_metadata pm
                   ON pm.document_id = v.document_id
            WHERE v.kind IN ({kind_ph})
              AND v.ai_caption IS NOT NULL
              AND length(v.ai_caption) >= 20
            ORDER BY random()
            LIMIT :n
        """), {**kind_params, "n": n}).fetchall()

    if not rows:
        console.print(
            "[yellow]No captioned figures/charts match the filter.[/yellow] "
            "Run `sciknow db caption-visuals` first."
        )
        return

    console.print(
        f"[bold]Benching {len(rows)} caption(s) → {out_path}[/bold]"
    )

    _resolved_model = model or settings.llm_fast_model or settings.llm_model

    def _llm_grade(row) -> tuple[str, str, str, str]:
        import ollama as _ollama
        client = _ollama.Client(host=settings.ollama_host, timeout=60)
        prompt = _CAPTION_GRADE_PROMPT.format(
            kind=row[2],
            title=(row[7] or "(unknown)")[:150],
            year=row[8] or "n.d.",
            original_caption=(row[3] or "")[:500],
            surrounding_text=(row[4] or "")[:500],
            ai_caption=(row[5] or "")[:1500],
        )
        try:
            resp = client.chat(
                model=_resolved_model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0, "num_predict": 300},
            )
            content = (resp.get("message") or {}).get("content", "")
            data = _json.loads(content)
            acc = str(data.get("accuracy", "")).strip().lower()
            hall = str(data.get("hallucination", "")).strip().lower()
            use = str(data.get("usefulness", "")).strip().lower()
            reason = str(data.get("reason", "")).strip()
            if acc not in ("yes", "partial", "no"):
                acc = "no"
            if hall not in ("none", "minor", "severe"):
                hall = "severe"
            if use not in ("good", "ok", "bland"):
                use = "bland"
            return acc, hall, use, reason
        except Exception as exc:
            return "no", "severe", "bland", f"judge-error: {exc}"

    def _human_grade(row, sugg):
        console.print()
        console.print(
            f"[bold cyan]Paper:[/bold cyan] "
            f"\"{row[7] or '(unknown)'}\" ({row[8] or 'n.d.'})"
        )
        console.print(
            f"[bold cyan]Original caption:[/bold cyan] {(row[3] or '')[:300]}"
        )
        console.print(
            f"[bold cyan]AI caption:[/bold cyan] {(row[5] or '')[:400]}"
        )
        if sugg:
            acc_s, hall_s, use_s, reason_s = sugg
            console.print(
                f"[yellow]LLM:[/yellow] acc={acc_s}, hall={hall_s}, "
                f"use={use_s} — {reason_s}"
            )
        mapping = {
            "y": "yes", "p": "partial", "n": "no",
            "none": "none", "minor": "minor", "severe": "severe",
            "g": "good", "ok": "ok", "b": "bland",
        }
        acc = typer.prompt(
            "accuracy [y/p/n]",
            default=(sugg[0][0] if sugg else "y"),
        ).strip().lower()
        hall = typer.prompt(
            "hallucination [none/minor/severe]",
            default=(sugg[1] if sugg else "none"),
        ).strip().lower()
        use = typer.prompt(
            "usefulness [g=good/ok/b=bland]",
            default=(sugg[2][0] if sugg else "g"),
        ).strip().lower()
        reason = typer.prompt("reason (optional)", default="")
        return (
            mapping.get(acc, acc),
            mapping.get(hall, hall),
            mapping.get(use, use),
            reason,
        )

    acc_counts: dict[str, int] = {}
    hall_counts: dict[str, int] = {}
    use_counts: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            sugg = None
            if judge in ("llm", "both"):
                sugg = _llm_grade(row)
            if judge in ("human", "both"):
                acc, hall, use, reason = _human_grade(row, sugg)
                judge_tag = "human"
            else:
                acc, hall, use, reason = sugg
                judge_tag = f"llm-{_resolved_model}"

            record = {
                "visual_id": row[0],
                "document_id": row[1],
                "kind": row[2],
                "paper_title": row[7],
                "year": row[8],
                "original_caption": row[3],
                "surrounding_text": row[4],
                "ai_caption": row[5],
                "ai_caption_model": row[6],
                "accuracy": acc,
                "hallucination": hall,
                "usefulness": use,
                "reason": reason,
                "judge": judge_tag,
                "graded_at": datetime.now(timezone.utc).isoformat(),
            }
            f.write(_json.dumps(record, ensure_ascii=False) + "\n")
            acc_counts[acc] = acc_counts.get(acc, 0) + 1
            hall_counts[hall] = hall_counts.get(hall, 0) + 1
            use_counts[use] = use_counts.get(use, 0) + 1

    total = len(rows)

    def _rate(d, k):
        return d.get(k, 0) / total * 100 if total else 0.0

    console.print()
    console.print(f"[bold green]✓ Bench written → {out_path}[/bold green]")
    console.print(
        f"[bold]Accuracy[/bold]     "
        f"yes:{acc_counts.get('yes', 0):3d}  "
        f"partial:{acc_counts.get('partial', 0):3d}  "
        f"no:{acc_counts.get('no', 0):3d}   "
        f"→ {_rate(acc_counts, 'yes'):.1f}% strict"
    )
    console.print(
        f"[bold]Hallucination[/bold] "
        f"none:{hall_counts.get('none', 0):3d}  "
        f"minor:{hall_counts.get('minor', 0):3d}  "
        f"severe:{hall_counts.get('severe', 0):3d}  "
        f"→ {_rate(hall_counts, 'severe'):.1f}% severe"
    )
    console.print(
        f"[bold]Usefulness[/bold]   "
        f"good:{use_counts.get('good', 0):3d}  "
        f"ok:{use_counts.get('ok', 0):3d}  "
        f"bland:{use_counts.get('bland', 0):3d}  "
        f"→ {_rate(use_counts, 'bland'):.1f}% bland"
    )
    console.print(
        "[dim]Track these across runs to catch VLM drift or "
        "prompt regressions. A sudden jump in severe hallucinations "
        "= model upgrade misfired; jump in bland usefulness = prompt "
        "went generic.[/dim]"
    )


# ── equation-bench (Phase 54.6.222 — roadmap 3.1.2) ──────────────────────────


_EQ_GRADE_PROMPT = """You are grading an equation that was extracted \
from a scientific paper and then paraphrased into prose for retrieval.

LaTeX (what MinerU extracted from the PDF):
$$
{latex}
$$

Paraphrase (what the LLM wrote to describe the equation):
"{paraphrase}"

Context from the paper (surrounding sentence, may be empty):
"{surrounding_text}"

Grade two separate things:

1. latex_valid — Is the LaTeX syntactically correct and parseable?
   * yes     — renders cleanly, all braces balanced, macros well-formed
   * partial — renders but has minor issues (stray braces, missing
               `\\mathrm`, etc.) that don't break the meaning
   * no      — malformed, mojibake, truncated, or empty

2. paraphrase_matches — Does the paraphrase describe the same math
   as the LaTeX?
   * yes     — all variables, operators, and the overall structure are
               faithfully captured
   * partial — most of it is right but one variable or operator is
               wrong, missing, or hallucinated
   * no      — paraphrase describes different math, or is generic
               filler ("this equation relates X and Y"), or is empty
   * unclear — paraphrase is missing or the LaTeX is so broken you
               can't judge

Respond in JSON ONLY, no prose:
{{"latex_valid": "yes|partial|no",
  "paraphrase_matches": "yes|partial|no|unclear",
  "reason": "<one short sentence>"}}
"""


@app.command(name="equation-bench")
def equation_bench_cmd(
    n: int = typer.Option(
        30, "--n", "-n",
        help="Random sample size. 30 is enough for ±10% precision "
             "at 95% CI on a binary metric; raise to 50-100 for "
             "tighter intervals.",
    ),
    judge: str = typer.Option(
        "llm", "--judge",
        help="Who grades: 'llm' (LLM_FAST_MODEL auto-grades; no "
             "human input), 'human' (interactive prompt per row), "
             "or 'both' (LLM first, human confirms or overrides).",
    ),
    model: str | None = typer.Option(
        None, "--model",
        help="Override the judge LLM. Default: settings.llm_fast_model.",
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir",
        help="Where to write the sample JSONL. Default: "
             "<data_dir>/equation_bench/bench-<iso-timestamp>.jsonl.",
    ),
    seed: int | None = typer.Option(
        None, "--seed",
        help="Random seed for deterministic sampling (debugging only).",
    ),
):
    """Phase 54.6.222 (roadmap 3.1.2) — bench equation extraction + paraphrase quality.

    Picks N random equations from the visuals table (kind='equation'),
    grades each one along two axes:

      * latex_valid — did MinerU extract syntactically correct LaTeX
                      from the PDF? (measures the converter's MFR model)
      * paraphrase_matches — did the LLM paraphrase (54.6.78) faithfully
                             describe the math? (measures the paraphrase
                             prompt + model combo)

    Persists one JSONL record per equation under
    ``<data_dir>/equation_bench/bench-<iso-timestamp>.jsonl``. Track
    both metrics over time — a sudden drop in latex_valid catches
    MinerU regressions, a drop in paraphrase_matches catches LLM /
    prompt drift.

    Output shape (one JSON object per line):

      {{
        "visual_id": "<uuid>",
        "document_id": "<uuid>", "paper_title": "...", "year": 2024,
        "latex": "...", "paraphrase": "...", "surrounding_text": "...",
        "latex_valid": "yes|partial|no",
        "paraphrase_matches": "yes|partial|no|unclear",
        "reason": "<judge's one-sentence rationale>",
        "judge": "llm-<model>|human",
        "graded_at": "<iso timestamp>"
      }}

    Examples:

      sciknow db equation-bench                 # 30 equations, LLM-graded
      sciknow db equation-bench --n 50          # bigger sample
      sciknow db equation-bench --judge human   # interactive grading
      sciknow db equation-bench --judge both    # LLM + human confirm
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    import json as _json
    import random as _random
    from datetime import datetime, timezone
    from sqlalchemy import text as sql_text
    from sciknow.config import settings
    from sciknow.storage.db import get_session

    if judge not in ("llm", "human", "both"):
        console.print(
            f"[red]Invalid --judge:[/red] {judge!r} — must be "
            "llm / human / both."
        )
        raise typer.Exit(2)

    from sciknow.core.project import get_active_project
    active = get_active_project()
    out_root = output_dir or (active.data_dir / "equation_bench")
    out_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_root / f"bench-{ts}.jsonl"

    if seed is not None:
        _random.seed(seed)

    with get_session() as session:
        rows = session.execute(sql_text("""
            SELECT v.id::text, v.document_id::text, v.content,
                   v.ai_caption, v.surrounding_text, pm.title, pm.year
            FROM visuals v
            LEFT JOIN paper_metadata pm
                   ON pm.document_id = v.document_id
            WHERE v.kind = 'equation'
              AND v.content IS NOT NULL
              AND length(v.content) >= 5
              AND v.ai_caption IS NOT NULL
              AND length(v.ai_caption) >= 20
            ORDER BY random()
            LIMIT :n
        """), {"n": n}).fetchall()

    if not rows:
        console.print(
            "[yellow]No equations match the filter.[/yellow] "
            "Run `sciknow db extract-visuals` and `sciknow db "
            "paraphrase-equations` first."
        )
        return

    console.print(
        f"[bold]Benching {len(rows)} equation(s) → {out_path}[/bold]"
    )

    _resolved_model = model or settings.llm_fast_model or settings.llm_model

    def _llm_grade(row) -> tuple[str, str, str]:
        import ollama as _ollama
        client = _ollama.Client(host=settings.ollama_host, timeout=60)
        prompt = _EQ_GRADE_PROMPT.format(
            latex=(row[2] or "")[:2000],
            paraphrase=(row[3] or "")[:1500],
            surrounding_text=(row[4] or "")[:500],
        )
        try:
            resp = client.chat(
                model=_resolved_model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0, "num_predict": 250},
            )
            content = (resp.get("message") or {}).get("content", "")
            data = _json.loads(content)
            lv = str(data.get("latex_valid", "")).strip().lower()
            pm_ = str(data.get("paraphrase_matches", "")).strip().lower()
            reason = str(data.get("reason", "")).strip()
            if lv not in ("yes", "partial", "no"):
                lv = "no"
            if pm_ not in ("yes", "partial", "no", "unclear"):
                pm_ = "unclear"
            return lv, pm_, reason
        except Exception as exc:
            return "no", "unclear", f"judge-error: {exc}"

    def _human_grade(row, llm_lv, llm_pm, llm_reason):
        console.print()
        console.print(
            f"[bold cyan]LaTeX:[/bold cyan] "
            f"{(row[2] or '')[:300]}"
            + ("…" if row[2] and len(row[2]) > 300 else "")
        )
        console.print(
            f"[bold cyan]Paraphrase:[/bold cyan] {(row[3] or '')[:400]}"
        )
        console.print(
            f"[dim]Source:[/dim] \"{row[5] or '(unknown)'}\" "
            f"({row[6] or 'n.d.'})"
        )
        if llm_lv is not None:
            console.print(
                f"[yellow]LLM:[/yellow] latex={llm_lv}, "
                f"paraphrase={llm_pm} — {llm_reason}"
            )
        lv = typer.prompt(
            "latex_valid [y=yes/p=partial/n=no]",
            default=(llm_lv[0] if llm_lv else "y"),
        ).strip().lower()
        pm_ = typer.prompt(
            "paraphrase_matches [y=yes/p=partial/n=no/u=unclear]",
            default=(llm_pm[0] if llm_pm else "y"),
        ).strip().lower()
        mapping = {"y": "yes", "p": "partial", "n": "no", "u": "unclear"}
        lv_out = mapping.get(lv, lv)
        pm_out = mapping.get(pm_, pm_)
        reason = typer.prompt("reason (optional)", default="")
        return lv_out, pm_out, reason

    # Running counts
    lv_counts: dict[str, int] = {}
    pm_counts: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            lv, pm_, reason = "no", "unclear", ""
            if judge in ("llm", "both"):
                lv, pm_, reason = _llm_grade(row)
            if judge in ("human", "both"):
                if judge == "both":
                    lv, pm_, reason = _human_grade(
                        row, lv, pm_, reason,
                    )
                else:
                    lv, pm_, reason = _human_grade(row, None, None, "")
                judge_tag = "human"
            else:
                judge_tag = f"llm-{_resolved_model}"

            record = {
                "visual_id": row[0],
                "document_id": row[1],
                "paper_title": row[5],
                "year": row[6],
                "latex": row[2],
                "paraphrase": row[3],
                "surrounding_text": row[4],
                "latex_valid": lv,
                "paraphrase_matches": pm_,
                "reason": reason,
                "judge": judge_tag,
                "graded_at": datetime.now(timezone.utc).isoformat(),
            }
            f.write(_json.dumps(record, ensure_ascii=False) + "\n")
            lv_counts[lv] = lv_counts.get(lv, 0) + 1
            pm_counts[pm_] = pm_counts.get(pm_, 0) + 1

    total = len(rows)
    lv_ok = lv_counts.get("yes", 0)
    lv_ok_loose = lv_ok + lv_counts.get("partial", 0)
    pm_ok = pm_counts.get("yes", 0)
    pm_ok_loose = pm_ok + pm_counts.get("partial", 0)

    console.print()
    console.print(f"[bold green]✓ Bench written → {out_path}[/bold green]")
    console.print(
        f"[bold]LaTeX validity[/bold]   "
        f"yes:{lv_counts.get('yes', 0):3d}  "
        f"partial:{lv_counts.get('partial', 0):3d}  "
        f"no:{lv_counts.get('no', 0):3d}   "
        f"→ {(lv_ok / total * 100):.1f}% strict · "
        f"{(lv_ok_loose / total * 100):.1f}% loose"
    )
    console.print(
        f"[bold]Paraphrase match[/bold] "
        f"yes:{pm_counts.get('yes', 0):3d}  "
        f"partial:{pm_counts.get('partial', 0):3d}  "
        f"no:{pm_counts.get('no', 0):3d}  "
        f"unclear:{pm_counts.get('unclear', 0):3d}   "
        f"→ {(pm_ok / total * 100):.1f}% strict · "
        f"{(pm_ok_loose / total * 100):.1f}% loose"
    )
    console.print(
        "[dim]Track these numbers across runs under "
        f"{out_root}/ to catch regressions in the converter's MFR "
        "(drop in latex_valid) or the paraphrase LLM (drop in "
        "paraphrase_matches).[/dim]"
    )


# ── flag-self-citations (Phase 54.6.223 — roadmap 3.6.2) ─────────────────────


@app.command(name="flag-self-citations")
def flag_self_citations_cmd(
    limit: int = typer.Option(
        0, "--limit",
        help="Process at most N citations (0 = all).",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Re-classify citations that already have is_self_cite "
             "set. Use after fixing a surname-normalisation bug.",
    ),
):
    """Phase 54.6.223 (roadmap 3.6.2) — flag self-citations via author overlap.

    Walks citations that are **cross-linked in-corpus** (both citing
    and cited papers are in our ``paper_metadata`` table) and marks
    each one as self-referential when the citing paper's author list
    overlaps the cited paper's author list on at least one
    ``(surname, first_initial)`` key.

    Enables consensus auditing — "is this claim supported by
    independent work or only by the same group citing itself?" — and
    feeds the writer's groundedness / overstated-claim passes with
    the signal (``citations.is_self_cite = true`` is a trust-weight
    tag for retrieval).

    Scope (Phase 1): only cross-linked citations
    (``cited_document_id IS NOT NULL``). Non-cross-linked citations
    have ``cited_authors = NULL`` in the current schema, so we can't
    run the overlap check on them without a separate enrichment
    pass that fetches cited authors from Crossref / OpenAlex —
    that's a follow-on.

    Examples:

      sciknow db flag-self-citations              # all cross-linked
      sciknow db flag-self-citations --limit 50   # smoke test
      sciknow db flag-self-citations --force      # re-run everything
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    import json as _json
    from sqlalchemy import text as sql_text
    from sciknow.core.self_citation import detect_self_cite
    from sciknow.storage.db import get_session

    filter_sql = "" if force else "AND c.is_self_cite IS NULL"
    with get_session() as session:
        rows = session.execute(sql_text(f"""
            SELECT c.id::text,
                   citing_pm.authors,
                   cited_pm.authors
            FROM citations c
            JOIN paper_metadata citing_pm
              ON citing_pm.document_id = c.citing_document_id
            JOIN paper_metadata cited_pm
              ON cited_pm.document_id = c.cited_document_id
            WHERE c.cited_document_id IS NOT NULL
              {filter_sql}
            {('LIMIT :lim' if limit else '')}
        """), ({"lim": limit} if limit else {})).fetchall()

    if not rows:
        console.print(
            "[green]No cross-linked citations need classification.[/green] "
            "Run with --force to re-classify everything."
        )
        return

    console.print(
        f"Classifying [bold]{len(rows)}[/bold] cross-linked citation(s)…"
    )
    flagged_self = 0
    flagged_other = 0
    undecided = 0

    with get_session() as session:
        for cid, citing_authors, cited_authors in rows:
            verdict, overlap = detect_self_cite(citing_authors, cited_authors)
            # is_self_cite = NULL for undecided (author data missing);
            # self_cite_authors = [] on undecided so the UI can tell
            # "not run" from "ran but no overlap".
            session.execute(sql_text("""
                UPDATE citations
                   SET is_self_cite = :verdict,
                       self_cite_authors = CAST(:overlap AS jsonb)
                 WHERE id::text = :cid
            """), {
                "cid": cid,
                "verdict": verdict,
                "overlap": _json.dumps(overlap),
            })
            if verdict is True:
                flagged_self += 1
            elif verdict is False:
                flagged_other += 1
            else:
                undecided += 1
        session.commit()

    total = len(rows)
    self_rate = (flagged_self / total * 100) if total else 0.0
    console.print(
        f"[green]✓ {flagged_self} self-cites[/green] · "
        f"{flagged_other} independent · "
        f"[yellow]{undecided} undecided[/yellow]"
    )
    console.print(
        f"[dim]Self-cite rate on in-corpus cross-linked citations: "
        f"{self_rate:.1f}% — ballpark for academic corpora is 5-20%; "
        f"large outliers signal either an author-clustering bug or a "
        f"very insular research group.[/dim]"
    )


# ── backfill-institutions (Phase 54.6.221 — roadmap 3.2.4) ────────────────────


@app.command(name="backfill-institutions")
def backfill_institutions_cmd(
    limit: int = typer.Option(
        0, "--limit",
        help="Process at most N papers (0 = all).",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Re-query OpenAlex even for papers that already have "
             "paper_institutions rows. Use after fixing a parse bug "
             "or if OpenAlex itself updated its affiliation data.",
    ),
    delay: float = typer.Option(
        0.2, "--delay",
        help="Seconds between OpenAlex calls (be polite — 5 req/s "
             "free tier, 10 req/s with email in User-Agent).",
    ),
):
    """Phase 54.6.221 (roadmap 3.2.4) — populate paper_institutions from OpenAlex.

    For every paper with a DOI but no rows in ``paper_institutions``,
    re-queries OpenAlex by DOI, extracts the
    ``authorships[].institutions[]`` list, and writes one row per
    institution into the dedicated table. Enables institution-level
    queries the prior ``oa_institutions_ror`` column couldn't
    support (display_name / country_code / institution_type were
    discarded there).

    Idempotent — skips papers that already have at least one
    institutions row unless ``--force``. Safe to re-run after
    ingesting new papers.

    Examples:

      sciknow db backfill-institutions                  # fill all gaps
      sciknow db backfill-institutions --limit 100      # sample-size run
      sciknow db backfill-institutions --force          # full refresh
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    from sqlalchemy import text as sql_text
    from sciknow.ingestion.expand_apis import fetch_openalex_work
    from sciknow.ingestion.openalex_enrich import (
        apply_openalex_enrichment,
    )
    from sciknow.storage.db import get_session

    # Find candidate papers: have a DOI, either have no institutions
    # rows (default) or all rows (force).
    filter_sql = "" if force else (
        " AND NOT EXISTS ("
        "   SELECT 1 FROM paper_institutions pi "
        "   WHERE pi.document_id = d.id"
        " )"
    )
    with get_session() as session:
        rows = session.execute(sql_text(f"""
            SELECT pm.id::text, d.id::text, pm.doi, pm.title
            FROM paper_metadata pm
            JOIN documents d ON d.id = pm.document_id
            WHERE pm.doi IS NOT NULL
              AND d.ingestion_status = 'complete'
              {filter_sql}
            ORDER BY pm.year DESC NULLS LAST
            {('LIMIT :lim' if limit else '')}
        """), ({"lim": limit} if limit else {})).fetchall()

    if not rows:
        console.print(
            "[green]No papers need institution backfill.[/green] "
            "Run with --force to re-query every paper with a DOI."
        )
        return

    console.print(
        f"Backfilling paper_institutions for [bold]{len(rows)}[/bold] "
        f"paper(s)…"
    )
    filled = 0
    skipped = 0
    errors = 0
    t0 = time.monotonic()

    for pid, doc_id, doi, title in rows:
        try:
            work = fetch_openalex_work(doi, fields="authorships,id")
            if not work:
                skipped += 1
                continue
            with get_session() as session:
                ok = apply_openalex_enrichment(session, pid, work)
                session.commit()
            if ok:
                filled += 1
            else:
                skipped += 1
        except Exception as exc:
            errors += 1
            logger.debug("institutions backfill failed for %s: %s", pid, exc)

        # Polite delay between OpenAlex calls.
        if delay > 0:
            time.sleep(delay)

    console.print(
        f"\n[green]✓ Filled {filled}[/green] · "
        f"[yellow]skipped {skipped}[/yellow] · "
        f"[red]errors {errors}[/red] · "
        f"[dim]{time.monotonic() - t0:.1f}s wall[/dim]"
    )
