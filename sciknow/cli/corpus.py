"""`sciknow corpus` — corpus growth + maintenance.

Spec §5.1: corpus owns ``ingest``, ``expand``, ``enrich``, ``cluster``
plus the operational siblings (refresh-metadata, repair, dedup,
reclassify-sections, link-citations, cleanup-downloads, sync-dense-
sidecar, classify-papers, all expand-* family verbs).

Implementations live in ``sciknow.cli.db`` / ``sciknow.cli.ingest`` /
``sciknow.cli.catalog``. This module is the v2 namespace; ``cli/main.py``
mounts a ``sciknow db`` deprecation shim that dispatches to the same
callables.
"""
from __future__ import annotations

import typer

from sciknow.cli import catalog as _catalog
from sciknow.cli import db as _db
from sciknow.cli import ingest as _ingest

app = typer.Typer(
    name="corpus",
    help="Corpus growth + maintenance: ingest, expand, enrich, cluster, "
         "refresh, repair, dedup, classify.",
    no_args_is_help=True,
)

# ── canonical verbs (spec §5.1) ─────────────────────────────────────────
app.add_typer(_ingest.app, name="ingest")
app.command(name="expand")(_db.expand)
app.command(name="enrich")(_db.enrich)
app.add_typer(_catalog.app, name="cluster")

# ── refresh / cleanup family ────────────────────────────────────────────
app.command(name="refresh-metadata")(_db.refresh_metadata)
app.command(name="refresh-retractions")(_db.refresh_retractions_cmd)
app.command(name="cleanup-downloads")(_db.cleanup_downloads)
app.command(name="reconcile-preprints")(_db.reconcile_preprints_cmd)
app.command(name="reconciliations")(_db.reconciliations_cmd)
app.command(name="unreconcile")(_db.unreconcile_cmd)

# ── repair / dedup / classify ───────────────────────────────────────────
app.command(name="repair")(_db.repair_cmd)
app.command(name="dedup")(_db.dedup_cmd)
app.command(name="reclassify-sections")(_db.reclassify_sections)
app.command(name="link-citations")(_db.link_citations)
app.command(name="classify-papers")(_db.classify_papers_cmd)
app.command(name="flag-self-citations")(_db.flag_self_citations_cmd)
app.command(name="sync-dense-sidecar")(_db.sync_dense_sidecar_cmd)

# ── expand family ───────────────────────────────────────────────────────
app.command(name="expand-author")(_db.expand_author)
app.command(name="expand-author-refs")(_db.expand_author_refs_cmd)
app.command(name="expand-cites")(_db.expand_cites)
app.command(name="expand-topic")(_db.expand_topic)
app.command(name="expand-coauthors")(_db.expand_coauthors)
app.command(name="expand-inbound")(_db.expand_inbound_cmd)
app.command(name="expand-oeuvre")(_db.expand_oeuvre_cmd)

# ── visuals / equations / tables ────────────────────────────────────────
app.command(name="extract-visuals")(_db.extract_visuals_cmd)
app.command(name="link-visual-mentions")(_db.link_visual_mentions_cmd)
app.command(name="caption-visuals")(_db.caption_visuals_cmd)
app.command(name="embed-visuals")(_db.embed_visuals_cmd)
app.command(name="paraphrase-equations")(_db.paraphrase_equations_cmd)
app.command(name="parse-tables")(_db.parse_tables_cmd)

# ── DOI handling ────────────────────────────────────────────────────────
app.command(name="download-dois")(_db.download_dois)
