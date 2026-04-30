# Operations Guide

[&larr; Back to README](../README.md)

---

## Backup & Restore

Move or clone the full sciknow collection to another machine.

### What gets backed up

| Component | Default | Flag to change |
|---|---|---|
| PostgreSQL database (papers, chunks, metadata) | always | — |
| Qdrant vector embeddings | always | `--skip-vectors` on restore |
| Original ingested PDFs (`data/processed/`) | on | `--no-pdfs` |
| Auto-downloaded PDFs (`data/downloads/`) | on | `--no-downloads` |
| PDF converter output (`data/mineru_output/`, used by both MinerU and Marker backends) | off | `--marker` |
| `.env` config file | always | — |

### Backup

```bash
# Full backup (PDFs + vectors + DB) — recommended
sciknow library backup --output sciknow_2026-04-04.tar.gz

# Metadata + vectors only (much smaller, no raw PDFs)
sciknow library backup --no-pdfs --no-downloads --output sciknow_metadata.tar.gz

# Include Marker output (avoids re-running OCR on restore)
sciknow library backup --marker --output sciknow_full.tar.gz
```

### Restore on a new machine

```bash
# 1. Install sciknow and start services
git clone https://github.com/you/sciknow
cd sciknow && ./scripts/setup.sh

# 2. Restore the backup
sciknow library restore sciknow_2026-04-04.tar.gz

# 3. If the DB already exists, use --force
sciknow library restore sciknow_2026-04-04.tar.gz --force

# 4. Verify
sciknow library stats
```

> After restore: edit `.env` if the new machine uses different credentials, hostnames, or model names.

---

## Reference Expansion (`db expand`)

Automatically grows the collection by following citations in existing papers. Six-source open-access discovery chain.

### How it works

1. **Reference extraction** — four sources unioned per paper: Crossref reference list, MinerU content_list.json, Marker markdown bibliography, OpenAlex referenced_works.

2. **Deduplication** — references already in the collection (by DOI or arXiv ID) are skipped.

3. **Semantic relevance filter** (on by default) — candidate titles are embedded with bge-m3 and scored against the corpus centroid or a user-provided topic query (`-q "solar forcing"`). Below threshold → dropped.

4. **Open-access PDF discovery** — six sources in priority order:
   - **Copernicus** — zero-cost URL construction for `10.5194/*` DOIs
   - **arXiv** — direct PDF for any arXiv ID
   - **Unpaywall** — largest general OA database
   - **OpenAlex** — catches preprints and institutional repos
   - **Europe PMC** — free full-text for biomedical papers
   - **Semantic Scholar** — final fallback

5. **Parallel download + batch ingest** — downloads run in a thread pool. MinerU + bge-m3 models load once and stay resident across the whole batch.

6. **Provenance tagging** — every expanded paper is tagged `ingest_source='expand'`.

```bash
sciknow corpus expand --dry-run --limit 50              # preview
sciknow corpus expand --limit 100                        # run it
sciknow corpus expand --limit 50 -q "solar irradiance"   # topic-targeted
sciknow corpus expand --relevance-threshold 0.65          # more selective
sciknow corpus expand --no-relevance --limit 100          # no filter
sciknow corpus expand --no-ingest --limit 50              # download only
sciknow corpus expand --resolve --limit 50                # resolve title-only refs
```

### Expected open-access hit rate

| Paper age | OA hit rate (Unpaywall only) | OA hit rate (v2 chain) |
|---|---|---|
| 2020-present | ~50-65% | ~65-80% |
| 2010-2019 | ~35-50% | ~45-60% |
| Pre-2010 | ~15-30% | ~20-35% |

Running `db expand` is non-destructive and idempotent.

---

## Metadata Enrichment (`db enrich`)

Finds DOIs for papers that lack them via active lookup.

1. **Crossref title search** — queries with paper title + first author
2. **Fuzzy title matching** — normalised similarity score, threshold 0.85
3. **OpenAlex fallback** — covers preprints and book chapters
4. **arXiv fallback** — for papers with arXiv ID but no metadata
5. **Full Crossref hydration** — once DOI confirmed, fetches complete record

```bash
sciknow corpus enrich                     # full run
sciknow corpus enrich --dry-run           # preview
sciknow corpus enrich --threshold 0.80    # looser matching
sciknow corpus enrich --limit 50          # bounded
```

### Why 100% DOI coverage is not achievable

| Content type | Why no DOI |
|---|---|
| Books | Have ISBNs, not DOIs |
| IPCC / UN reports | Grey literature |
| Preprints on personal sites | Never registered |
| Conference abstracts | Not always indexed |

A well-curated journal library can reach ~95%. Mixed collections plateau at 55-70%.

---

## Citation Graph

References are extracted from every paper and stored in the `citations` table. Cross-linked when both citing and cited papers are in the corpus.

- **Automatic cross-linking** during ingestion (forward + backward)
- **Batch re-linking:** `sciknow corpus link-citations` after bulk ingestion
- **Citation-boosted retrieval:** log-dampened boost via `CITATION_BOOST_FACTOR`

---

## Similar Papers

```bash
sciknow search similar "Water Vapor Feedback"
sciknow search similar 10.1038/nature12345
sciknow search similar 2301.12345
sciknow search similar "Water Vapor" --show-scores -k 20
```

Uses bge-m3 dense embeddings in the `abstracts` collection.

---

## Provenance Tracking

Every document has an `ingest_source` field:
- `seed` — manually ingested via CLI
- `expand` — auto-discovered via `db expand`

`sciknow library stats` shows an "Ingest source" breakdown.

---

## Pre-flight Checks

All long-running commands (`ingest`, `expand`, `enrich`, `export`, `catalog cluster`) verify that PostgreSQL and/or Qdrant are reachable **before** doing any work:

```
Qdrant is unreachable.
  Check that the service is running:
    systemctl --user status qdrant
    systemctl --user start qdrant
```

---

## Web reader: jobs, exports, debugging

The web reader (`sciknow book serve`) runs every LLM operation as a job tracked in `_jobs[id]` (in-memory dict, ephemeral). Two facts worth knowing for ops:

**Job GC (Phase 22).** Finished jobs are swept after 5 minutes (`_JOB_GC_AGE_SECONDS`). If you're debugging a job that "disappeared", check `data/sciknow.log` for the run id — the in-memory queue is gone but the log line is forever. Active jobs are never GC'd.

**Polling stats endpoint (Phase 32.5).** `GET /api/jobs/{job_id}/stats` returns a fixed-shape JSON snapshot: `{tokens, tps, elapsed_s, model_name, task_desc, target_words, stream_state, error_message}`. The persistent task bar polls this every 500ms. The `stream_state` field is the lifecycle marker — `streaming` / `done` / `error`. A 410 response means the job was already swept by the GC.

**Debugging a hung job.** Two channels: (1) `data/sciknow.log` for backend Python tracebacks, (2) for autowrite specifically, `data/autowrite/<run_id>.jsonl` for the per-iteration heartbeat log (Phase 24). The heartbeat fires from a side thread so even a stuck generator yields a "no progress in N seconds" line.

**Web exports (Phase 30 / 31).** `GET /api/export/{draft,chapter,book}/{id}.{ext}` produces `txt / md / html / pdf`. PDF rendering uses WeasyPrint directly off the rendered HTML. The CLI `book export` command has a different format matrix (md / html / bibtex / latex / docx via Pandoc) — see `docs/reference/BOOK.md` for the full split.

---

## Development Notes

### Adding a new section type

Edit `_SECTION_PATTERNS` in `sciknow/ingestion/chunker.py` — add a new `(canonical_type, [prefixes])` tuple. If it should not be embedded, add to `_SKIP_SECTIONS`. Add chunking parameters to `_PARAMS`.

### Changing the embedding model

1. Update `EMBEDDING_MODEL` and `EMBEDDING_DIM` in `.env`
2. Re-initialize collections: `sciknow library init`
3. Re-embed all papers: `db reset` + re-ingest

### Metadata quality

Papers are tagged with `metadata_source`: `crossref` (highest), `arxiv`, `embedded_pdf`, `llm_extracted`, `unknown`. Papers with `llm_extracted` or `unknown` are candidates for manual review.

### Resuming failed ingestion

Re-run the same `ingest` command — the pipeline detects the existing record by SHA-256 hash, resets its status, and retries from the beginning.
