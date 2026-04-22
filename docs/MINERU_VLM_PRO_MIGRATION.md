# MinerU 2.5-Pro Migration — Handoff

Phase 4 of roadmap 3.1.6. The code substrate landed across
phases 1-3 (commits 54.6.211 → 54.6.213); this document is the
**user-gated** destructive step: install the VLM extras, wipe
the corpus, re-ingest everything on the new backend.

**Nothing in this doc is reversible once started** (short of
re-ingesting from the same PDFs under the old backend, which
would take the same ~week). Read through once before running any
command.

---

## 1. Current state (2026-04-22, before migration)

- 807 documents, all `ingestion_status = complete`
- 32,992 chunks in the `papers` Qdrant collection
- `mineru` installed; `vllm` NOT installed; `transformers 4.57.6`
  installed — so VLM-Pro currently falls back to the pipeline
  backend silently (Phase 2's `_vlm_extras_missing` handler does
  the right thing; every new convert is still pipeline-mode).
- `PDF_CONVERTER_BACKEND` defaults to `auto` and
  `MINERU_VLM_BACKEND` defaults to `vllm` — both correct for the
  migration; no `.env` tweaks needed.

Verify freshly before proceeding:

```bash
uv run sciknow db stats
uv run python -c "import vllm" 2>&1 | head -3   # expect: ModuleNotFoundError
```

---

## 2. Install the VLM-Pro extras

```bash
uv add 'mineru[vllm]'
```

This pulls `vllm` + its CUDA-compatible `torch`. Expect ~2 GB of
downloads; installation takes 5-15 minutes depending on the
CUDA/driver cache. On the 3090 install box:

- vllm ≥ 0.6 supports Qwen2VL — that's what MinerU 2.5-Pro is
  built on.
- torch must match your CUDA runtime. If `uv sync` starts
  complaining about CUDA mismatches, your system `nvidia-smi`
  CUDA version and the installed `torch` CUDA wheel disagree —
  usually resolved by forcing a specific torch wheel index
  (`uv add --torch-backend cu121 torch`) or by pinning
  `torch==X.Y.Z` to match the vllm release notes.

Verify the install:

```bash
uv run python -c "import vllm; print(vllm.__version__)"
uv run python -c "import mineru; print(mineru.__version__)"
```

---

## 3. Smoke test on one paper (do NOT skip)

Pick ~one representative PDF from `projects/<slug>/data/inbox/` or
from the existing `data/processed/` tree. Run VLM-Pro explicitly:

```bash
# pick a paper already in the corpus — re-ingesting it with --force
# exercises the full pipeline without contaminating the data
DOC_ID=$(uv run sciknow db stats --json 2>/dev/null | head -1 || true)  # optional
export PDF_CONVERTER_BACKEND=mineru-vlm-pro
uv run sciknow ingest file projects/<slug>/data/processed/<paper>.pdf --force
unset PDF_CONVERTER_BACKEND
```

Watch for:

1. **First run downloads the 1.2B VLM** from HuggingFace to
   `~/.cache/modelscope` or `~/.cache/huggingface`. Allow ~5-10
   minutes on first invocation; subsequent runs are instant.
2. **vllm engine startup** logs something like `INFO: vllm is
   loading model opendatalab/MinerU2.5-Pro-2604-1.2B on device
   cuda:0`. If it says `transformers` instead, your
   `MINERU_VLM_BACKEND` setting didn't take effect — check
   `.env`.
3. **content_list.json** written under
   `data/mineru_output/<doc_id>/<stem>/vlm/`. Verify it has
   sensible `type`-keyed blocks.
4. **Converter stamp**: `uv run sciknow db stats --doc-id <prefix>`
   (or direct SQL) should show `converter_backend =
   'mineru-vlm-pro-vllm'` and `converter_version` populated.

If the smoke test passes, proceed to §4. If it fails, fix the
install issue — do NOT run `db reset` on a broken VLM-Pro setup
(you'd wipe the working pipeline-era corpus and then be unable
to rebuild).

---

## 4. Corpus-wide re-ingest

`db reset` wipes Postgres, Qdrant, `data/processed/`,
`data/downloads/`, and `mineru_output/`. It preserves
`data/inbox/` and the original PDFs under
`projects/<slug>/data/processed/` if they live there (check your
directory layout first). **Make sure the PDFs survive the reset**
— without them there's nothing to re-ingest.

```bash
# Dry-run path audit (SAFE — no writes):
uv run sciknow db stats
ls projects/<slug>/data/processed/ | wc -l    # should be ~807

# DESTRUCTIVE — only run after the smoke test in §3 passed:
uv run sciknow db reset
uv run sciknow db init        # fresh migrations + Qdrant collections
uv run sciknow ingest directory projects/<slug>/data/processed/ \
    --workers 1                # vllm serialises inference; 1 worker
uv run sciknow refresh --no-ingest
```

### Wall-clock expectation

On a 3090 with vLLM at ~0.05-0.1 pages/s (rough, no bench run):

- 807 papers × ~15 pages average = ~12,000 pages
- At 0.07 pages/s → ~48 hours of pure convert time
- Plus metadata + chunk + embed (~5-10 minutes per 100 papers) +
  full refresh (RAPTOR rebuild, wiki compile, visuals captioning)
- **Realistic estimate: 3-5 days wall-clock** on a 3090 if
  uninterrupted

Use `--budget-time` to break into sessions:

```bash
uv run sciknow refresh --no-ingest --budget-time 12h --no-wiki
# next session:
uv run sciknow refresh --no-ingest --budget-time 12h --since last-run
```

(Phase 54.6.206 + 54.6.210 were shipped on 2026-04-22 precisely
for this kind of multi-session run.)

---

## 5. Post-migration verification

After the re-ingest completes:

```bash
uv run sciknow db stats
uv run sciknow db failures         # any new failure classes?

# every document should carry the new stamp
uv run python -c "
from sciknow.storage.db import get_session
from sqlalchemy import text
with get_session() as s:
    rows = s.execute(text('''
        SELECT converter_backend, COUNT(*)
        FROM documents
        WHERE ingestion_status = 'complete'
        GROUP BY converter_backend
        ORDER BY COUNT(*) DESC
    ''')).all()
    for b, n in rows:
        print(f'{b!s:<35} {n}')
"
```

Expected: `mineru-vlm-pro-vllm` ≥ 95% of the completed rows, the
remainder on `mineru-pipeline` or `marker-json` from the
auto-dispatch fallback chain. If the VLM-Pro count is lower than
expected, inspect the failures clinic to see why.

Sanity-check a known paper's wiki + visuals to confirm:

- Tables no longer split mid-row on page breaks (cross-page
  merging working).
- In-table images appear as `kind='table_image'` rows in
  `visuals` (Phase 3 recursion working).
- Retrieval quality on a few known queries — subjective but
  should feel qualitatively tighter.

---

## 6. Rollback (if something is badly wrong)

VLM-Pro is not producing sensible output? Quick rollback:

```bash
# 1. Stop any running ingest
# 2. Pin the old backend
echo "PDF_CONVERTER_BACKEND=mineru" >> .env
# 3. Ignore the deprecation warning for now, re-ingest on pipeline
uv run sciknow db reset
uv run sciknow db init
uv run sciknow ingest directory projects/<slug>/data/processed/
# 4. Capture repro case for a bug report
```

Post-rollback, before trying VLM-Pro again, reproduce the failure
on one paper with `PDF_CONVERTER_BACKEND=mineru-vlm-pro` and file
an issue upstream with the MinerU project.

---

## 7. What comes next (Phases 5 + 6)

After the re-ingest completes and the corpus is fully on VLM-Pro:

- **Phase 5 — multi-aspect captions** (closes roadmap 3.5.2).
  Migration adds `visuals.literal_caption` + `visuals.query_caption`
  columns; `db caption-visuals` rewritten to emit all three
  (literal from MinerU-Pro content_list, synthesis from qwen2.5vl:32b,
  query from qwen2.5vl:7b). Roughly 2 days of coding.
- **Phase 6 — deprecate `PDF_CONVERTER_BACKEND=mineru`** as a
  documented option. Removes it from `.env.example` + CLAUDE.md
  + INGESTION.md options list; keeps the code path reachable
  only through auto-dispatch fallback. ~0.5 day.

After those land, §3.1.6 is fully done and the roadmap's "Next
Review" cluster (3.0.1 / 3.3.1 / 3.4.3 / 3.6.1) comes back into
play.
