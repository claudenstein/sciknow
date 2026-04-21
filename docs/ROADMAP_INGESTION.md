# Ingestion + Enrichment — Research Roadmap

Living document of candidate improvements to `sciknow refresh` /
`sciknow ingest` and the downstream enrichment commands. The intent
is a menu of **research proposals** ranked by priority — not a ship
plan. Each entry is a miniature hypothesis: what's the current state,
what would change, what's the expected win, what's the cost.

Cross-references:
- `docs/INGESTION.md` — current pipeline architecture (source of truth).
- `docs/ROADMAP.md` — shipped / deferred / rejected work across the app.
- `docs/RESEARCH.md` §1-6 — ingestion research that already landed
  (MinerU, hybrid search, BERTopic, RAPTOR, wiki compile).
- `docs/BENCHMARKS.md` — retrieval + VLM sweep numbers this roadmap
  builds on.

**How to read this.** Section 3 is the full menu grouped by pipeline
stage. Section 4 is the same items collapsed into a single priority
table. Section 5 lists what we've already decided **not** to do so
further research doesn't relitigate.

---

## 1. Current pipeline (1-minute recap)

```
  inbox/ + downloads/ + failed/
         │
         ▼ sciknow ingest  (per-document state machine)
  pending → converting → metadata_extraction → chunking → embedding → complete
         │
         ▼ sciknow refresh  (13 idempotent post-ingest steps)
  1. ingest inbox  2. resume downloads  3. retry failed
  4. db enrich     5. refresh-retractions  6. link-citations
  7. classify-papers  8. catalog cluster  9. catalog raptor build
  10. tag-multimodal  11. extract-visuals  12. caption-visuals
  13. paraphrase-equations  14. parse-tables  15. embed-visuals
  16. wiki compile
```

Models in play today:
- **Dense + sparse embedder**: `BAAI/bge-m3` (1024-dim dense, lexical sparse, same pass)
- **PDF converter**: MinerU 2.5 pipeline (primary), MinerU 2.5-Pro VLM (opt-in), Marker JSON + markdown (fallback)
- **Metadata LLM fallback**: `LLM_FAST_MODEL` on first ~3000 chars
- **Visual caption VLM**: `qwen2.5vl:32b` (pinned from 54.6.74 sweep)
- **Reranker**: `BAAI/bge-reranker-v2-m3`

Idempotency + resumability: every stage keys on SHA-256 of file bytes;
re-running skips completed work; failures route to `data/failed/` with
explicit error.

---

## 2. Research axes

Every proposal below is evaluated against at least one of:

- **Q** Quality — lower hallucination, tighter section boundaries,
  better caption grounding, cleaner metadata.
- **C** Coverage — more papers resolved, fewer `metadata_source =
  unknown`, fewer empty sections, more visuals captured.
- **S** Speed — higher throughput, fewer VRAM stalls, faster enrich.
- **R** Robustness — graceful degradation, better recovery from
  partial failure, smaller blast radius.
- **O** Observability — ability to measure what's working and surface
  regressions before a user notices.

---

## 3. Proposals by pipeline stage

### 3.0 Discovery + download (pre-ingest)

#### 3.0.1 Expand OA resolver set (C)

The current download path for `sciknow db expand` etc. resolves OA
PDFs through Crossref / Unpaywall / arXiv / Semantic Scholar. Known
OA sources the expander **doesn't** consult today:

- **Europe PMC** — life-sciences + climate-adjacent, surprisingly rich
  for geochemistry, glaciology, ecology.
- **PubMed Central** — NIH-funded open-access. Partial overlap with
  Europe PMC but stricter licensing clarity.
- **CORE (core.ac.uk)** — aggregator of institutional repositories,
  good recall for preprints that never hit arXiv.
- **OSF preprints** — bioRxiv / medRxiv / EarthArXiv / SocArXiv, all
  of which sit on OSF.
- **bioRxiv / medRxiv direct API** — already partially via
  OpenAlex/S2, but going direct catches newer items faster (hours
  vs days).

Expected win: cuts `pending_downloads` rows by ~15-25% on typical
climate corpora; EarthArXiv alone would have saved ~40 papers on
global-cooling in the 54.6.51 downloader pass.

Cost: ~1 day per resolver. API schemas differ; rate limits to respect.

Open question: is the per-paper cost of querying 5 resolvers worth
the yield vs. just accepting higher `pending_downloads` counts for
manual retrieval?

#### 3.0.2 Delta velocity watcher tuning (O)

Phase 54.6.X already ships an OpenAlex "new publications" watcher.
Current frequency: on-demand. Could become scheduled + quieter.

Proposal: nightly cron → new DOIs matching the project's topic
cluster land in `pending_downloads` with a `source_method =
velocity_watcher` tag. User reviews a daily digest instead of running
expand manually.

Cost: low (~hours). Value: preserves low-friction discovery as the
corpus grows.

#### 3.0.3 Citation-graph distance gating (Q, C)

`db expand` currently follows citations one hop. A two-hop expansion
with distance-weighted scoring (hop-1 papers ranked higher than hop-2)
would widen coverage without flooding with irrelevant noise. Similar
shape to the 54.6.70 cocite-boost retrieval signal, but applied at
discovery time.

Cost: moderate (~2-3 days). Storage: `pending_downloads.distance` int.
Risk: if the scoring is loose, `pending_downloads` explodes.

---

### 3.1 PDF conversion

#### 3.1.1 Routed converter backend (S, Q)

Current dispatch is user-selected (`PDF_CONVERTER_BACKEND = auto |
mineru | mineru-vlm-pro | marker`) and applies per-file uniformly.
MinerU 2.5 (86.2 OmniDocBench) handles ~90% of papers cleanly but
loses on:
- heavily-figured papers (figure-text reading order gets tangled)
- old scans without OCR
- PDFs with embedded XObject form-fields

Proposal: add a **decision gate** before conversion that inspects:
- first-page heuristics (has machine-readable text via PyMuPDF?)
- figure density (number of images × area fraction)
- scan signature (monochrome rasters, no embedded text)

Route:
- Typical paper → MinerU 2.5 pipeline (fast path, ~0.4 pages/s)
- Complex/figure-heavy → MinerU 2.5-Pro VLM (95.69 OmniDocBench)
- Pure scan → Marker JSON with Surya OCR (slower but tolerant)

Cost: ~2 days. Benchmark set needed: curate 20 edge-case PDFs.

Expected win: quality bump on the ~10% hard tail; the 90% fast path
unchanged. Benchmark OmniDocBench difference: 86.2 → ~89+ composite
when the gate correctly picks VLM-Pro for hard cases.

#### 3.1.2 Equation + chart extraction benchmark (O, Q)

MinerU's MFD/MFR extracts LaTeX from rendered formulas. Phase 54.6.78
paraphrases equations into retrievable prose. No current benchmark of
extraction accuracy (how often is the LaTeX right? how often does the
paraphrase match the formula's meaning?).

Proposal: curate 30 equations from the corpus, have a domain expert
(or even the user) label correct/incorrect, run a retrospective
accuracy check. Establishes a baseline before we swap extractors.

Cost: ~1 day for the eval set + harness. No new extractor yet.

#### 3.1.3 Nougat / Pix2Tex comparison (Q)

Alternative LaTeX recognizers. **Meta's Nougat** (Blecher et al. 2023)
claims better formula recognition on academic papers. **Pix2Tex**
(LaTeX-OCR, open source) is a lightweight per-formula recognizer.

Proposal: after 3.1.2 establishes a baseline, try Nougat as a
parallel pipeline, see if composite OmniDocBench improves by >3
points. Only worth adopting if MinerU-MFR is a measurable bottleneck.

Cost: ~3 days. Likely only worthwhile if 3.1.2 shows >15% LaTeX
error rate on our corpus.

#### 3.1.4 Table recognition — PaddleOCR PP-StructureV3 comparison (Q)

MinerU HTML tables recover most structure but lose column types and
multi-row headers on complex stats tables. PP-StructureV3 (Baidu) is
competitive on PubTabNet. Phase 54.6.106 already downstream-parses
MinerU HTML into `{title, headers[], summary}` via LLM — if the
upstream HTML is richer, the parse step could emit real structured
rows instead of just a summary.

Cost: ~2 days benchmark. Only worth pursuing if the user starts
needing **numeric table search** (query → rows with matching cells).

#### 3.1.5 Reading-order ML model (Q)

MinerU uses heuristics (column detection, block bounding-box
ordering). Most of the time it's right; ~5% of papers get reading
order scrambled (common in two-column layouts with floating figures).

Proposal: train a lightweight reading-order classifier on
DocLayout-YOLO outputs. OR: accept MinerU's output as-is, post-process
with LLM-based reading-order fixing on the 5% where heuristics fail
(detect via section-header discontinuities).

Cost: ~1 week for training; ~1 day for LLM post-fix.

---

### 3.2 Metadata extraction

#### 3.2.1 Semantic Scholar author disambiguation (Q, C)

Currently `paper_metadata.authors` stores raw author strings from
Crossref/arXiv. S2 has author IDs with affiliation + paper history.
Hooking `corpusId → authors[].authorId` would let us:
- Detect author self-citation (for KG quality)
- Build an author-level "oeuvre" view (already partial via Phase
  54.6.X oeuvre modal)
- Disambiguate common names (J. Smith ≠ J. Smith)

Cost: ~2-3 days. S2 API rate limits are generous for this use case.

#### 3.2.2 Publisher-specific metadata resolvers (C)

For journals that don't resolve cleanly via DOI lookup (paywalled
publishers with thin Crossref metadata), consult publisher APIs:
- **Wiley Online** — open API for TDM, works for DOI lookups
- **Springer Nature** — OpenAccess API
- **Elsevier ScienceDirect** — TDM API (needs institutional token)
- **AGU / GSA / AMS** — domain-specific for earth sciences

Payoff is coverage-heavy for climate science (many papers are AGU/AMS).

Cost: ~1 day per publisher if API is friendly. Some require auth;
user can add their institutional tokens.

#### 3.2.3 Retraction + correction watcher (R, Q)

Phase 54.6.111 added one-shot retraction sweep. Could make it
continuous: a weekly cron that re-checks Crossref's retraction index
for every corpus DOI. Current state: `retraction_checked_at`
timestamp exists but is never re-checked after the initial pass.

Cost: trivial (~hours). Scheduled CronCreate.

#### 3.2.4 Affiliation + institution extraction (Q, O)

OpenAlex returns `authorships[].institutions[]` with ROR IDs. Phase
54.6.111 persists the raw OpenAlex payload; could parse institutions
into a dedicated `paper_institutions` table. Enables institution-level
queries ("show me climate papers from NOAA over the last 10 years").

Cost: ~1 day for schema + parse. Value depends on whether the user
needs institutional analysis.

#### 3.2.5 Language detection + multi-lingual FTS dictionaries (C, Q)

Ingestion assumes English. Non-English papers (e.g. Spanish-language
climate history from Latin American journals) get poorly tokenized
by `to_tsvector('english', …)`. bge-m3 IS multilingual so dense
retrieval still works, but FTS returns junk.

Proposal: add `documents.language` via `py3langid` or fasttext on the
first ~500 chars; FTS chooses its dictionary per-row via `to_tsvector(
language_config, content)`.

Cost: ~1 day. Slightly complicates the FTS index (generated column
becomes conditional).

#### 3.2.6 Keyword normalization (Q)

Different metadata sources use different keyword vocabularies (MeSH,
ACM CCS, IEEE taxonomy, author keywords, OpenAlex concepts). Today
they land in `paper_metadata.keywords` as a mixed pile. Normalizing
to a single taxonomy (probably OpenAlex concepts since they're already
there) would make topic filtering less noisy.

Cost: ~2 days. Non-trivial because OpenAlex concepts are hierarchical
and older papers have sparse coverage.

---

### 3.3 Chunking

#### 3.3.1 Semantic chunking within section (Q)

Current: token-bounded chunks with section-boundary respect. A long
"discussion" section can span 2-4 chunks with hard cut points. LangChain's
`RecursiveCharacterTextSplitter` has a semantic-cut mode using sentence
embeddings (split where cosine distance between adjacent sentences
peaks). Chroma + LlamaIndex have similar modes.

Proposal: within-section, add an optional semantic cut step — split
at topic-shift boundaries rather than token count, as long as the
resulting chunk fits the per-section `_PARAMS` budget.

Cost: ~3-5 days. Needs a small eval: does retrieval quality improve
on a 30-query test set? bge-m3 chunks are already semantically dense;
gains may be marginal.

Prior art: **Semantic Chunking** (Hongbo Liu et al., 2024 — [arXiv:
2402.05131](https://arxiv.org/abs/2402.05131)) — 2-5 pt MRR improvement
on academic corpora.

#### 3.3.2 Reference section parsing (C, Q)

Today `references` is in `_SKIP_SECTIONS` — not embedded, not
chunked. But the references list is valuable structured data:
- Author names (for coauthor graph extension)
- Cited DOIs (already extracted via `citations` table, but re-parsing
  references catches items the in-text regex missed)
- Year distribution (how recent is the bibliography?)

Proposal: run a dedicated parser over the references block per
paper — something like AnyStyle or Grobid's reference parser — and
populate a `references_parsed` table. Cross-link to existing
`citations`.

Cost: ~3-4 days. Adds a new dependency (Grobid is Dockerized; AnyStyle
is Ruby). Alternative: LLM-based reference extraction (slower but no
new runtime).

#### 3.3.3 Figure / table caption linkage (Q)

Captions currently live on `visuals` rows, separated from the
text they narratively belong to. When a paragraph says *"As shown in
Fig. 3, …"*, retrieving the paragraph **and** the figure as a joined
unit would improve writing-time grounding. Phase 54.6.138 added the
mention-paragraph linker at write time, but ingestion-time linkage
into a dedicated `paragraph_figures` edge table would enable more
downstream features.

Cost: ~2 days (re-use the mention linker, persist into a table).

#### 3.3.4 Chunk deduplication (S)

Near-duplicate chunks across papers are common: standard intros
("Climate change is one of the most pressing…"), boilerplate methods
sections, rewritten review prose. Currently indexed separately, which
inflates corpus size + dilutes retrieval.

Proposal: SimHash or MinHash-LSH on chunks during embedding; chunks
within 0.92+ Jaccard get marked `duplicate_of` pointing at the first
one. Retrieval filters out duplicates.

Cost: ~2-3 days. Risk: aggressive dedup might drop legit chunks that
just share a boilerplate sentence. Needs tuning.

---

### 3.4 Embedding

#### 3.4.1 Qwen3-Embedding comparison (Q)

New Qwen3-Embedding family (4B, 8B) is competitive with bge-m3 on
MTEB and supports 32k context (bge-m3 is 8k). For long scientific
chunks it'd handle overflow better.

Proposal: `sciknow bench --layer embedding-sweep` — encode the
standard 50-query eval set with bge-m3 + Qwen3-Embedding-4B +
Nomic-embed-v2 + GTE-Qwen2-7B, compare MRR / R@10. Swap only if Δ >
+3 points AND latency tolerable.

Cost: ~1 day to run. Switching embedders is disruptive — every chunk
needs re-embedding, and the downstream Qdrant collections need
re-creating.

#### 3.4.2 Domain adaptation via contrastive fine-tune (Q)

For the climate-science corpus specifically, fine-tuning bge-m3 on
within-corpus positive pairs (paper → its references, related papers
by cocitation) would improve retrieval on domain-specific queries.
Requires labeled positives (~2k pairs). DGX Spark-gated.

Cost: ~1 week on Spark; blocked on hardware.

Prior art: **InPars** (Bonifacio et al. 2022), **Doc2Query** —
synthetic query generation for retriever training.

#### 3.4.3 Enable ColBERT late-interaction vectors (Q, S)

bge-m3 produces dense + sparse + **colbert** vectors in a single pass.
`embedder.py:82` currently requests only dense + sparse. Enabling
colbert would add multi-vector late-interaction retrieval — better
nuance, especially for multi-concept queries.

Cost of enabling: storage + retrieval latency. Each chunk produces
~150 colbert vectors instead of 1 dense — 150× multiplier on Qdrant
storage. Retrieval becomes a re-ranking step on top-50 candidates.

Proposal: enable ONLY for the `abstracts` collection (paper-level,
fewer rows) as a cheap experiment. Measure NDCG delta.

Cost: ~2 days.

#### 3.4.4 Section-type as retrieval signal (Q)

Already present as a Qdrant payload filter (`--section methods`).
Could become a **ranking** signal rather than just a filter: queries
with method-sounding keywords ("how did they measure", "what dataset")
get methods-section chunks reranked higher. Phase 54.6.80 added the
paper-type classifier; section-type is structurally available.

Cost: ~2 days (train a simple classifier on query → preferred
section-type, apply as reranker feature).

#### 3.4.5 Rewrite query with LLM for hard queries (Q)

Step-back prompting already shipped (Phase 10 / `_retrieve_with_
step_back`). The "rejected" list explicitly says HyDE (query
expansion with a hypothetical document) was rejected — but for
**hard** queries where step-back retrieves nothing, an LLM-generated
paraphrase (not a whole hypothetical doc) might help.

Cost: ~2 days. Falls under the already-rejected direction so would
need explicit re-evaluation against Phase 34's rationale before
pursuing.

---

### 3.5 Visuals

#### 3.5.1 Caption quality audit + retry pass (Q, O)

Current `db caption-visuals` runs qwen2.5vl:32b (pinned post-54.6.74
sweep). No periodic quality audit. Captions can be bland ("A bar
chart showing …") or wrong (model hallucinates axes). A small sample
(~30) should be manually scored, then the prompt + model combo
optimized against that score.

Cost: ~1-2 days for sample + rubric; prompt tweaks are cheap once the
rubric exists.

#### 3.5.2 Multi-aspect captions (Q)

Current single caption tries to cover what/why. Proposal: three
captions per figure (stored as separate fields or JSONB):
- **Literal** — what the image shows (axes, labels, symbols)
- **Synthesis** — what the paper claims it supports
- **Query-ready** — dense search-friendly paraphrase

Increases embedding value per figure. More VLM calls (3× per figure
vs 1×) but each call is shorter.

Cost: ~2 days. Migration to add fields. Worth it when visual search
becomes a primary retrieval surface.

#### 3.5.3 Chart data extraction (Q)

Extract numerical values from bar / line charts. Tools: **PlotQA**,
**ChartQA** family models; **DePlot** (Google); **UniChart**
(HuggingFace). For climate science, this unlocks "find papers showing
CO₂ vs temperature beyond 2× pre-industrial" — currently the paper
is findable, the graph isn't queryable.

Cost: ~1 week. Research-level. Value high but scope creep risk.

#### 3.5.4 Figure-in-paragraph alignment training set (O)

Phase 54.6.138 mention-paragraph linker uses heuristics (figure
number + nearby paragraph). Could build a training set by labeling
50 (figure, paragraph) pairs, then evaluate the heuristic vs an
embedding-based alignment. Establishes ground truth before we improve
the ranker.

Cost: ~1 day labeling + harness.

---

### 3.6 Citations + reference graph

#### 3.6.1 Citation-purpose classification (Q)

Each `citations` row is "paper A cites paper B" with no semantic
context. Research shows 4-6 citation functions: **background**,
**method use**, **result comparison**, **contrast/criticism**,
**extension**, **hedging**. Labeling each citation with its function
(LLM one-shot on surrounding 2 sentences) enables:
- "Find every paper that **criticises** [PaperX]'s methods"
- "List methods papers I cite for my core approach"

Cost: ~3-4 days. LLM inference per citation is cheap; schema change
straightforward.

Prior art: **ACL-ARC** citation-function taxonomy (Jurgens et al.
2018), **SciCite** (Cohan et al. 2019) — both have published
classifiers we could port.

#### 3.6.2 Self-citation + coauthor-citation flagging (Q)

Already have coauthor graph implicitly (via `oeuvre`). Flagging
author A → paper by author A, or author A → paper by frequent
coauthor, enables consensus auditing (is this claim supported by
external work or only by the group's own?).

Cost: ~1 day. Useful for the writer's `groundedness` / `overstated`
pass.

#### 3.6.3 Forward citation counts per chunk (Q)

Currently stored at paper level. At chunk level, the same paper's
methods section might be cited 200× while its discussion is cited 3×
— very different trust signals per retrieval unit. Harder to compute
(needs citation-context → citing paper's specific chunk), but useful.

Cost: ~1 week. Not sure the win is worth it.

---

### 3.7 Knowledge graph (extract-kg)

#### 3.7.1 Entity canonicalization (Q)

KG entities today are LLM-extracted verbatim. "CO₂", "carbon
dioxide", "CO2" all yield separate nodes. Need a canonicalization
pass.

Options:
- **Rule-based** — hand-maintained alias table for common
  domain terms (CO₂, NAO, ENSO, SST, TSI, …).
- **Wikidata grounding** — query each entity against Wikidata, pick
  the highest-confidence match, link by QID.
- **Ontology** — Bind to an existing climate ontology (ECHO, ENVO) —
  over-engineered for our scale.

Recommendation: start with rule-based (~100 hand-curated aliases
from the user's papers), add Wikidata when the rule file feels
limiting.

Cost: ~2 days rules; ~1 week Wikidata.

#### 3.7.2 Relation schema stabilization (Q)

Current relations are LLM free-text ("caused by", "contributes to",
"correlates with", …). Per-paper the vocabulary is consistent; across
papers it drifts. Pick a closed vocabulary of ~20 relations (climate-
specific: FORCES, RESPONDS_TO, PROXIES_FOR, RECONSTRUCTS,
CONTRADICTS, SUPPORTS, CORRELATES_WITH, CITES_DATA, CITES_METHOD,
…), re-extract with the constrained prompt.

Cost: ~2 days. Needs user input on the vocabulary.

#### 3.7.3 KG quality sampling (O)

No current way to measure KG extraction accuracy. Sample 30 triples,
have the user (or a GPT-4-class model) grade each as
correct/plausible/wrong, track the precision over time. Catches
silent regressions when the extract model changes.

Cost: ~1 day setup, ongoing sampling essentially free.

---

### 3.8 Topic clustering (BERTopic)

#### 3.8.1 Hierarchical topics (C)

Current clusters are flat (Phase 7 shipped 6-14 named clusters).
BERTopic supports hierarchical clustering — parent topics group
related child clusters. Enables "Climate forcing → solar → grand
minima" drill-down for filtering.

Cost: ~1 day (BERTopic option toggle + UI).

#### 3.8.2 Dynamic topics over time (O, Q)

Topic prominence shifts with publication year. BERTopic's `topics_
over_time` visualizes this. Could populate a "topic velocity" metric:
which topics are accelerating (recent-years-heavy) vs declining.

Cost: ~1-2 days. Visualization heavy, mostly already-built machinery.

#### 3.8.3 Topic coherence metrics (O)

Currently trust c-TF-IDF labels. Adding UMass / NPMI coherence scores
per cluster lets us flag low-coherence topics ("catch-all" clusters
that need splitting or merging).

Cost: ~1 day.

---

### 3.9 RAPTOR

#### 3.9.1 Summary verification on rebuild (Q)

RAPTOR summaries are LLM-generated from constituent chunks. Currently
no check that the summary faithfully represents the chunk set.
Proposal: after each summary is generated, run a claim-atomization
check (existing from Phase 54.6.83) — each sentence must be supported
by at least one constituent chunk.

Cost: ~2 days. Slows rebuild (one extra verification pass).

#### 3.9.2 Adaptive tree depth (Q)

Fixed 3-level tree today. For a small corpus (<100 papers), two
levels suffice; for large (>5000) four makes chapter-length queries
more useful. Auto-pick depth based on corpus size.

Cost: ~1 day.

---

### 3.10 Wiki compile

#### 3.10.1 Incremental wiki updates (S)

`wiki compile` currently re-compiles from scratch. For a 1000-paper
corpus this is slow. Detect "changed since last compile" rows and
only re-summarize those pages.

Cost: ~3 days. Needs last-compiled-at tracking. Medium value once
corpus is stable.

#### 3.10.2 Consistency check between wiki claims and papers (Q)

If the wiki page for "NAO" makes claim X, every X should trace back
to at least one corpus paper. Auto-audit: run claim-atomization on
wiki pages, flag unsupported sentences.

Cost: ~2 days. Builds on existing claim-atomization infra.

---

### 3.11 Cross-cutting

#### 3.11.1 Parallel ingest of multiple documents (S)

Today ingest is per-document sequential. With RTX 3090 under-utilized
during metadata/chunking stages, 2-4 documents in flight would
improve throughput. Needs careful ordering around GPU stages
(converting / embedding).

Proposal: pipeline-level parallelism — N documents at different
stages simultaneously, serialized only at GPU-bound stages (converting
with MinerU, embedding with bge-m3).

Cost: ~1 week. Risk: deadlock if serialization is wrong. Value: 2-3×
throughput on typical corpus growth (+30% on all-in refresh time).

#### 3.11.2 Incremental refresh (`sciknow refresh --since=...`) (S)

Every refresh step currently scans its whole target table. Most
steps are already idempotent-by-skip (check "does this row already
have caption?"), but the scan itself takes time at 30k+ chunks.
Adding `last_updated_at` per row + a `--since=<timestamp>` or
`--new-only` flag would make incremental refresh seconds instead of
minutes.

Cost: ~2-3 days. Low risk.

#### 3.11.3 Pipeline observability dashboard (O)

`db stats` shows counts per stage. A real dashboard would show:
- ingest throughput per day (trend line)
- failure rate per stage (per-stage error histogram)
- time-per-stage distribution
- VRAM ceiling events (log when a stage failed due to OOM)
- model-call costs (tokens in/out, VLM calls)

Cost: ~3-4 days. Builds on existing `ingestion_jobs` table. UI
integration into the existing web reader.

#### 3.11.4 Budget-aware refresh (R, S)

A `--budget-time=6h` / `--budget-tokens=1M` flag that stops refresh
gracefully when hit (finishes the current doc, logs state, exits).
Useful for overnight runs with known deadlines.

Cost: ~2 days. Simple wrapper around existing per-stage loops.

#### 3.11.5 Quality regression suite (O)

A "before + after a refresh change, did things get better or worse?"
harness. Pick 50 fixed queries, measure retrieval MRR / NDCG before
and after. Already partially exists in `sciknow bench --layer
live`. Extension: snapshot the metric set per commit, alert on
regression.

Cost: ~2 days for snapshotting; ~1 day for CI integration (if we
ever have CI).

#### 3.11.6 Failure-mode clinic (R, Q)

`data/failed/` accumulates documents that couldn't ingest. Today
the `ingestion_jobs` table records the error but there's no view
that aggregates "4 papers failed because MinerU choked on nested
tables, 7 because metadata LLM timed out". A clinic view would
summarize failure classes and suggest fixes.

Cost: ~1-2 days. Build on existing telemetry.

---

## 4. Priority ranking

Scoring: **Impact** (H/M/L) × **Effort** (H/M/L) → **Verdict**.

- **Ship** — high impact + low/medium effort, low risk
- **Investigate** — needs a small benchmark before deciding
- **Defer** — good idea, bigger effort, not top of queue
- **Gated** — blocked on external (hardware, API access, user input)

| # | Proposal | Axes | Impact | Effort | Verdict | Notes |
|---|---|---|---|---|---|---|
| 3.2.3 | Retraction + correction watcher (cron) | R, Q | M | L | **Rejected** | User veto 2026-04-22: not needed |
| 3.2.5 | Language detection + multi-lingual FTS | C, Q | M | L | **Ship** | Fixes silent FTS failures on non-EN papers |
| 3.7.1 | KG entity canonicalization (rule-based) | Q | H | L | **Ship** | Hand-curated alias table first; Wikidata later |
| 3.9.1 | RAPTOR summary verification | Q | M | L | **Ship** | Reuses Phase 54.6.83 claim-atomization |
| 3.11.2 | Incremental refresh (`--since`) | S | M | L | **Ship** | 2-3 days, frequent usage win |
| 3.11.4 | Budget-aware refresh (`--budget-time`) | R, S | L | L | **Ship** | 2 days; unblocks overnight runs |
| 3.11.6 | Failure-mode clinic view | R, O | M | L | **Ship** | Surfaces data already in ingestion_jobs |
| 3.0.1 | Expand OA resolver set (Europe PMC + CORE + OSF) | C | H | M | **Next Review** | Queued for evaluation after Ship cluster lands. Pilot with Europe PMC first; measure yield lift |
| 3.1.1 | Routed converter backend (heuristic gate) | S, Q | H | M | **Next Review** | Queued for evaluation after Ship cluster lands. Need 20-PDF edge-case bench set first |
| 3.3.1 | Semantic chunking within section | Q | M | M | **Next Review** | Queued for evaluation after Ship cluster lands. Benchmark vs current MRR before committing |
| 3.4.3 | ColBERT late-interaction on abstracts collection | Q | M | M | **Next Review** | Queued for evaluation after Ship cluster lands. Cheap pilot; storage cost is the gate |
| 3.6.1 | Citation-purpose classification | Q | M | M | **Next Review** | Queued for evaluation after Ship cluster lands. Port ACL-ARC / SciCite classifier |
| 3.5.1 | Caption quality audit + retry pass | Q, O | M | L | **Investigate** | 30-sample rubric; adjust prompts iteratively |
| 3.8.1 | Hierarchical BERTopic clusters | C, Q | M | L | **Investigate** | Check: does hierarchy reveal genuine sub-topics or over-split? |
| 3.0.2 | Velocity watcher nightly cron | O | L | L | **Investigate** | User preference call — do they want a digest? |
| 3.2.1 | S2 author disambiguation | Q, C | M | M | **Defer** | Requires S2 author IDs per author; schema burden |
| 3.2.4 | Institution extraction | Q, O | L | L | **Defer** | Nice-to-have; wait for a real institutional-query use case |
| 3.3.2 | Reference section parsing (Grobid / LLM) | C, Q | M | M | **Defer** | Current citations table covers main use cases |
| 3.3.3 | Figure / table caption linkage (ingest-time table) | Q | M | M | **Defer** | Phase 54.6.138 write-time linker covers main use |
| 3.3.4 | Chunk deduplication (MinHash-LSH) | S | M | M | **Defer** | Dedup risk > current duplication cost |
| 3.4.1 | Qwen3-Embedding comparison | Q | M | L | **Defer** | Only if bench shows bge-m3 hitting a ceiling |
| 3.4.4 | Section-type as ranking signal | Q | L | M | **Defer** | Filter already exists; ranking delta likely small |
| 3.5.2 | Multi-aspect captions (literal / synthesis / search) | Q | M | M | **Defer** | VLM cost 3×; defer until visual search is a primary surface |
| 3.5.4 | Figure-paragraph alignment training set | O | L | L | **Defer** | Phase 54.6.138 heuristic is good enough for now |
| 3.6.2 | Self-citation flagging | Q | L | L | **Defer** | Piggyback on 3.2.1 |
| 3.7.2 | KG relation vocabulary constraint | Q | M | M | **Defer** | Needs user input on domain-specific relations |
| 3.7.3 | KG quality sampling | O | L | L | **Defer** | Useful when KG surface is more central |
| 3.8.2 | Dynamic topics over time | O | L | M | **Defer** | Visualization play, non-critical |
| 3.8.3 | Topic coherence metrics | O | L | L | **Defer** | Current labels are human-acceptable |
| 3.9.2 | Adaptive RAPTOR depth | Q | L | L | **Defer** | Current 3-level fine for current corpus size |
| 3.10.1 | Incremental wiki updates | S | M | M | **Defer** | Full rebuild is fine at current corpus size; worth it past 2k papers |
| 3.10.2 | Wiki-claim-to-paper consistency check | Q | M | M | **Defer** | Builds on claim-atomization; low regression risk |
| 3.11.3 | Pipeline observability dashboard | O | M | M | **Defer** | Useful polish; wait for a real incident to motivate |
| 3.11.5 | Quality regression suite | O | M | L | **Defer** | Already partly exists via `bench --layer live` |
| 3.1.2 | Equation extraction accuracy bench | O, Q | L | L | **Defer** | Run only before considering 3.1.3 |
| 3.1.5 | Reading-order ML model | Q | M | H | **Defer** | 5% of papers affected; training effort high |
| 3.2.6 | Keyword normalization to OpenAlex concepts | Q | L | M | **Defer** | User hasn't hit the noise yet |
| 3.0.3 | Citation-graph two-hop expansion | Q, C | M | M | **Defer** | Risk of `pending_downloads` explosion |
| 3.1.3 | Nougat / Pix2Tex comparison | Q | L | M | **Gated** | Depends on 3.1.2 showing a real gap |
| 3.1.4 | PaddleOCR table comparison | Q | L | M | **Gated** | Only when numeric table search becomes a primary need |
| 3.2.2 | Publisher-specific resolvers (Wiley, Springer, …) | C | M | M | **Gated** | User to provide institutional API tokens |
| 3.4.2 | Domain contrastive fine-tune of bge-m3 | Q | H | H | **Gated** | DGX Spark hardware + ≥2k labeled positive pairs |
| 3.5.3 | Chart data extraction (DePlot / UniChart) | Q | M | H | **Gated** | Research-level; DGX Spark helpful |
| 3.6.3 | Per-chunk forward citation counts | Q | L | H | **Gated** | Needs external citation-context database |
| 3.11.1 | Parallel ingest of multiple documents | S | H | M | **Gated** | Best tested on DGX Spark; RTX 3090 VRAM marginal |

---

## 5. Explicit deferrals / rejections

Do not re-propose these without surfacing new evidence against the
listed reason:

- **HyDE (hypothetical-document expansion)** — rejected in 2026-04
  lit sweep; step-back prompting (Phase 10) gave the same retrieval
  win without hallucination risk.
- **Self-RAG / CRAG (fine-tuned retrieval-aware generation)** —
  rejected; requires fine-tuned LLM, infrastructure cost outweighs
  the retrieval gain on our corpus size.
- **Dense X / Propositional Retrieval** — rejected; proposition-level
  chunking loses too much context for our long-form synthesis use
  case.
- **GraphRAG global mode** — rejected; full knowledge graph
  summarization is slow (hours) and unnecessary for the reading/
  writing loop we actually use.
- **Late Chunking (Jina)** — rejected; bge-m3's long-context
  (8k) already covers the benefit.
- **Full RST tree parsing** — rejected; PDTB-lite discourse plan
  (Phase 9) captures the rhetorical structure we actually use.
- **Full Centering Theory machinery** — rejected; Phase 8 entity-
  bridge simplification shipped the practical win.
- **FActScore online verifier** — rejected; claim-atomization
  (Phase 54.6.83) is the offline substitute.
- **ALCE citation benchmark** — rejected; our internal verify
  pipeline (Phase 54.6.145 finalize-draft) is more aligned with the
  write-time contract we enforce.
- **Paper-level FTS (title + abstract + keywords)** — rejected in
  Phase 54.6.136; signal-overlap probe caught Jaccard(sparse, FTS)
  = 0 because the tokenised text sets were structurally disjoint
  from chunk-level sparse. Chunk-level FTS (`chunks.search_vector`)
  replaced it.

---

## 6. Next action

User decision 2026-04-22: ship the cluster **minus 3.2.3** (vetoed
as unnecessary). After the cluster lands, revisit the five "Next
Review" proposals flagged in §4 (3.0.1, 3.1.1, 3.3.1, 3.4.3, 3.6.1)
and decide which ones to advance to Ship.

Shipping order (smallest → largest scope):

1. **3.11.6 Failure-mode clinic** — read-only aggregation over the
   existing `ingestion_jobs` table. No schema change.
2. **3.11.4 Budget-aware refresh (`--budget-time`)** — CLI flag +
   per-step clock check. No schema change.
3. **3.2.5 Language detection + multi-lingual FTS** — adds a
   language column + switches FTS dict per row. One migration.
4. **3.9.1 RAPTOR summary verification** — reuses claim-atomization
   (Phase 54.6.83). No schema change.
5. **3.7.1 KG entity canonicalization (rule-based)** — alias file
   shipped in-repo + normalization in extract-kg. No schema change.
6. **3.11.2 Incremental refresh (`--since`)** — touches every step's
   filter; biggest scope, best landed after the smaller ones are in.

Total time estimate for the ship cluster (minus 3.2.3): **~10
focused hours**, spread across 2-3 sessions. After shipping,
reassess the Next Review set against the new baseline.
