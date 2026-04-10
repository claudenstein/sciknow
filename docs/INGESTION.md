# Ingestion Pipeline

[&larr; Back to README](../README.md)

---

The ingestion pipeline stages (visible in `db stats` during a run):

```
pending → converting → metadata_extraction → chunking → embedding → complete
                                                                   → failed
```

Failed PDFs are copied to `data/failed/`. Successfully processed PDFs are copied to `data/processed/`. Original files are never deleted.

---

## Stage 1 — PDF Conversion (MinerU → Marker fallback)

`sciknow/ingestion/pdf_converter.py` dispatches based on `PDF_CONVERTER_BACKEND`:

1. **MinerU 2.5 pipeline** (default) — OpenDataLab's pipeline backend. Runs a cascade of specialised models: DocLayout-YOLO for layout, MFD (Math Formula Detection) + MFR (Math Formula Recognition) for LaTeX extraction, table OCR + structure reconstruction (HTML tables), text OCR, seal detection. Scores 86.2 on [OmniDocBench v1.5](https://arxiv.org/abs/2412.07626). Runs on any GPU with ≥8 GB VRAM (Volta or newer). Models cached to `~/.cache/modelscope` on first use (~2 GB).

2. **MinerU 2.5-Pro VLM** (opt-in, Phase 21) — A 1.2B Qwen2VL model fine-tuned on MinerU's data engine, scoring **95.69 on [OmniDocBench v1.6](https://arxiv.org/abs/2604.04771)** — current SOTA among open-source PDF parsers, beating same-architecture baselines by 2.71 points and surpassing models with 200× more parameters. Set `PDF_CONVERTER_BACKEND=mineru-vlm-pro` in `.env` to enable. Requires `mineru[vlm]` extras (`uv add 'mineru[vlm]'`) and a GPU with ~4 GB free VRAM. The model identifier (`opendatalab/MinerU2.5-Pro-2604-1.2B`) is overridable via `MINERU_VLM_MODEL` if you want to pin a specific build. Slower than the pipeline backend on CPU, faster end-to-end on GPU because there's only one inference pass (vs. layout → formula → table → OCR cascade).

3. **Marker JSON** (fallback) — `marker-pdf`'s `JSONRenderer` produces a structured block tree (`SectionHeader`, `Text`, `Table`, `Equation`, `ListItem`, ...). Used automatically when MinerU fails, or when `PDF_CONVERTER_BACKEND=marker`.

4. **Marker markdown** (last resort) — if Marker's JSON path also fails, the markdown renderer runs as a final fallback.

**MinerU output format:** `content_list.json` — a flat list of typed blocks:
- `text` with `text_level` (0 = body, 1 = title, 2 = section heading, 3+ = subheading)
- `table` with HTML `table_body` + caption arrays
- `equation` with `text` containing LaTeX and `text_format: "latex"`
- `image` / `chart` / `seal` with paths + captions
- `code` with `code_body` and `sub_type` (code vs algorithm)
- `list` with `list_items` array
- Auxiliary blocks (`header`, `footer`, `page_number`, `page_footnote`, `aside_text`) — dropped by the chunker

Output lands in `data/mineru_output/{doc_id}/{stem}/auto/`. Models for both backends load once per Python process and stay resident in VRAM — batching ingestion with `--workers N` amortises the load across many PDFs.

Why MinerU as primary: Marker has a known severe performance regression on RTX 3090 ([datalab-to/marker#919](https://github.com/datalab-to/marker/issues/919) — ~0.03 pages/s with 18-19 GB of VRAM sitting idle). MinerU 2.5 on the same GPU runs at ~0.4 pages/s single-stream.

---

## Stage 2 — Metadata Extraction (4 layers)

Each layer is tried in order; later layers only run if the previous didn't fully populate the metadata:

1. **PyMuPDF** — reads embedded XMP/Info fields from the PDF. Fast but often incomplete.
2. **Crossref API** — authoritative bibliographic data by DOI. DOI is extracted from the paper text via regex. Uses the Crossref polite pool (rate limit: 50 req/s with email in User-Agent).
3. **arXiv API** — for preprints. arXiv IDs are extracted from URLs and filenames.
4. **Ollama LLM fallback** — sends the first ~3000 characters of the document text to `LLM_FAST_MODEL` with a structured JSON extraction prompt. Used when no DOI or arXiv ID is found.

Fields extracted: title, abstract, year, DOI, arXiv ID, journal, volume, issue, pages, publisher, authors (with ORCID and affiliation where available), keywords.

---

## Stage 3 — Section-Aware Chunking

The chunker has three parallel parsers, one per PDF backend, all producing the same `Section` list downstream:

**MinerU mode (primary):** `parse_sections_from_mineru(content_list)` walks the flat typed-block list. Text items with `text_level == 1 or 2` open a new top-level section; `text_level >= 3` become inline bold subheadings; `text_level == 0` (body) is accumulated. Tables are rendered to pipe-delimited plain text (with caption prepended). Equations contribute their LaTeX string. Code blocks contribute `code_body`. Lists are joined with newlines. Images/charts/seals and page-level blocks are dropped.

**Marker JSON mode (fallback):** `parse_sections_from_json(json_data)` walks Marker's nested block tree. `SectionHeader` blocks at heading level h1/h2 open new sections; h3/h4 are inlined as bold subheadings. Same implicit-heading heuristic as the MinerU path.

**Markdown mode (last resort):** `parse_sections(markdown)` detects sections from markdown headings (`#` through `####`) using regex.

### Canonical section types

| Canonical type | Detected headings |
|---|---|
| `abstract` | Abstract |
| `introduction` | Introduction, Background, Motivation, Overview |
| `methods` | Methods, Materials, Experimental Setup, Data Collection, Simulation, ... |
| `results` | Results, Findings, Evaluation, Measurements, ... |
| `discussion` | Discussion, Analysis, Interpretation, Implications |
| `conclusion` | Conclusion, Summary, Outlook, Future Work, ... |
| `related_work` | Related Work, Prior Work, Literature Review, ... |
| `references` | References, Bibliography *(not embedded — stored in citations table)* |
| `acknowledgments` | Acknowledgments, Funding *(not embedded)* |
| `appendix` | Appendix, Supplementary |

### Chunking parameters per section type

| Section | Target tokens | Overlap | Keep whole if under |
|---|---|---|---|
| abstract | 512 | 0 | 512 |
| introduction | 512 | 64 | 768 |
| methods | 512 | 128 | 768 |
| results | 512 | 128 | 768 |
| discussion | 512 | 64 | 768 |
| conclusion | 512 | 0 | 1024 |

Each chunk is prefixed with a context header:
```
[methods] Paper Title Here (2023)

<actual chunk content...>
```
This ensures the embedding captures paper identity even when a chunk is retrieved in isolation.

---

## Stage 4 — Embedding (bge-m3)

`BAAI/bge-m3` produces two vector types per chunk simultaneously:
- **Dense** (1024-dim cosine): semantic similarity search
- **Sparse** (learned lexical weights): keyword/term matching

Both are stored in Qdrant. Abstracts are also embedded separately in the `abstracts` collection for paper-level search.

---

## Idempotency and Resume

`documents` is keyed on SHA-256 hash of the file bytes, so re-running `sciknow ingest directory ...` skips completed papers and retries failed/partial ones from scratch. `data/failed/` is a **copy**; the pipeline re-reads from the original path stored in `documents.original_path`. `--force` is only for deliberately re-ingesting completed papers.

---

## Scaling

- `EMBEDDING_BATCH_SIZE=32` is the default — safe on a 24 GB 3090 with a 32B q4 LLM co-resident
- With only the embedder on the GPU (no LLM), raise to 64 for faster bulk ingestion
- Drop to 16 when running MinerU `--workers 2` simultaneously
