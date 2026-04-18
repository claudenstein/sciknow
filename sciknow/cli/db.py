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
        for f in feats:
            w = oa_works.get(f.key) or oa_works.get((f.doi or f"arxiv:{f.arxiv_id}").lower())
            enrich_from_openalex_work(f, w)

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

        # ── 8. Semantic Scholar isInfluential + intents ────────────────────
        if not no_s2:
            survivors = [f for f in feats if not f.hard_drop_reason and f.doi]
            if survivors:
                console.print(
                    f"  fetching Semantic Scholar citations for {len(survivors)} "
                    "survivors (1 RPS throttled)…"
                )
                for f in survivors:
                    data = expand_apis.fetch_s2_citations(f.doi)
                    f.influential_cite_count = expand_apis.count_influential_from_corpus(
                        data, existing_dois
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
                          "figure_num", "surrounding_text"):
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
                        session.execute(sql_text("""
                            INSERT INTO visuals
                                (document_id, kind, content, caption, asset_path,
                                 block_idx, figure_num, surrounding_text)
                            VALUES
                                (CAST(:document_id AS uuid), :kind, :content,
                                 :caption, :asset_path, :block_idx, :figure_num,
                                 :surrounding_text)
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
