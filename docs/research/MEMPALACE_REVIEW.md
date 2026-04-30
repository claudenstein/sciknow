# `mempalace` — portability review

**Source**: https://github.com/mempalace/mempalace (v3.3.0; ~20 Python
modules; pytest + ruff + benchmark harness; Apache/MIT-ish; ChromaDB-
backed).
**Review date**: 2026-04-14.

Bottom line: mempalace is a **local-first verbatim chat-memory tool
for AI agents** (Claude Code / Cursor / Gemini via MCP), not a
scientific-RAG system. 80% of it (wings / rooms / drawers, the AAAK
"dialect", MCP server, `convo_miner`, method-of-loci metaphors) is
architecturally orthogonal to sciknow. But seven specific engineering
patterns inside it close concrete gaps in sciknow. Author's
philosophy — "verbatim always, never LLM-summarise the source,
incremental-only" — aligns cleanly with sciknow's own stance, and his
April 7 2026 honesty note retracting an earlier "+34% palace boost"
claim is a good pattern to emulate.

## What mempalace actually is

- ChromaDB store over a `wing/room/drawer` hierarchy of verbatim
  session transcripts + project docs.
- 29-tool MCP server so agents can read/write memories autonomously.
- Marquee numbers: LongMemEval R@5 96.6% raw, 100% with a Haiku
  reranker.
- Stack: Python 3.9+, `chromadb`, `pyyaml`, stdlib `sqlite3` for KG,
  stdlib `urllib` for an optional LLM "closet" index. No
  Qdrant/Postgres/Marker/MinerU/reranker model baked in.
- v4 alpha roadmap: pluggable backends (`pg_sorted_heap`, LanceDB),
  hybrid search, time-decay scoring, stale-index detection.

Philosophy highlights (from `CLAUDE.md` + `MISSION.md`):
  1. **Verbatim always** — never LLM-summarise user data.
  2. **Incremental only** — append, never destroy-to-rebuild.
  3. Reject cloud sync, telemetry, API-key requirements for core ops.

## Shipped patterns worth porting (priority stack)

### P1 — Query sanitiser for the web `/api/ask` (~2 h)

**What**: `query_sanitizer.py::sanitize_query` is a 4-step fallback —
passthrough → question extraction → tail sentence → tail truncation
— that defends against "system-prompt contamination". Observed in
MCP-enabled clients: a 2000-char system prompt gets concatenated in
front of a 15-char user question, the embedder drowns, recall
collapses silently.

**Why it fits**: sciknow's `/api/ask` passes whatever string the
client sent straight to bge-m3. Nothing flags the failure — just 1%
recall.

**Scope**: new `sciknow/retrieval/query_sanitizer.py` (~150 LOC);
wire in at the top of `retrieval/hybrid_search.py::search()` and
`rag/llm.py::ask_*()`. Extra 1–2 ms per query, acceptable.

**L1 test**: `sanitize_query("… long system-prompt … What is X?")`
returns method=`question_extraction` and clean_query=`"What is X?"`.

### P2 — Chunk-level near-duplicate dedup (~3 h)

**What**: `dedup.py::dedup_palace` groups chunks by source, sorts by
length desc, keep-if-cosine-distance-to-all-kept > threshold. Greedy,
O(n·k) per source, single Chroma scroll.

**Why it fits**: `db expand` pulls preprint v1, v2, and journal
versions for the same paper. SHA-256 catches byte-identical only;
arXiv v1 → v2 share ~95% text but differ by bytes. Qdrant ends up
with 3× duplicate chunks; the reranker and LLM context budget waste
slots on paraphrases.

**Scope**: new `sciknow/maintenance/dedup.py` (~150 LOC); new CLI
`sciknow corpus dedup [--dry-run] [--threshold 0.15]`. Scrolls Qdrant by
`document_id` groups, cosine-distance 0.15 (~85% similarity),
transactionally deletes points + `chunks` rows.

**L1 test**: `dedup_corpus(dry_run=True)` on a known-clean smoke
corpus returns 0 deletions.

### P3 — Chunker version stamp (~1 h)

**What**: `NORMALIZE_VERSION` integer constant stamped on every
chunk. `file_already_mined` returns False when stored-version <
current → silent resume path re-ingests automatically.

**Why it fits**: sciknow's `CLAUDE.md` explicitly warns that
`_SECTION_PATTERNS` / `_SKIP_SECTIONS` / `_PARAMS` in the chunker
are "a contract"; changing them leaves old chunks silently stale,
and the only cleanup path today is `db reset` (destructive). Current
`chunks.embedding_model` catches embedder swaps but not chunker
swaps.

**Scope**: Alembic migration adds `chunker_version INT DEFAULT 1` to
`paper_sections` (or `chunks`); new `CHUNKER_VERSION = 1` constant
at the top of `sciknow/ingestion/chunker.py`; `pipeline.py::ingest_one()`
re-chunks when version is stale. Bump the constant on any section-
pattern or parameter edit.

**L1 test**: stub `CHUNKER_VERSION=99`; call `needs_rechunk(doc)` on
a freshly-ingested v1 paper, expect True.

### P4 — Claim verification against the knowledge graph (~3 h)

**What**: `fact_checker.py::check_text` extracts (subject, predicate,
object) from prose with two regex patterns, compares against the
stored KG triples, flags `relationship_mismatch` / `invalid_as_of`.

**Why it fits**: sciknow already has (a) claim extraction in autowrite
verify, (b) a `knowledge_graph` triples table with `source_sentence`,
(c) the Phase-47 rejected-idea gate. What's missing is using the
global KG as an oracle during verify: if Chapter 3 says "Hansen 1988
used CESM1" but the KG has "Hansen 1988 used a 1-box model", we
don't currently flag the contradiction.

**Scope**: new `sciknow/verification/kg_check.py` (~120 LOC); reuse
the two regex patterns from `fact_checker._RELATIONSHIP_PATTERNS`
plus one "X <verb> Y → triple" pattern matching our predicate
vocabulary. Wire into `core/book_ops.py::_verify_draft_inner`; emit
a new `kg_contradiction` event the scorer can fold into groundedness.

**Trade-off**: regex extraction is brittle — precision > recall. Only
flag high-confidence contradictions. LLM-assisted extraction can be
a later upgrade.

**L1 test**: seed KG with `(hansen1988, used_model, box_model)`; call
`check_text("Hansen 1988 used CESM1")`; assert issues contain
`type=relationship_mismatch`.

### P5 — Temporal validity on triples (~4 h, needs sign-off)

**What**: `knowledge_graph.py::KnowledgeGraph` stores
`(subject, predicate, object, valid_from, valid_to)` quads in
SQLite; `as_of(date)` queries filter to facts valid at a point in
time; `invalidate(...)` sets `valid_to`.

**Why it fits**: niche but high-leverage. sciknow's wiki treats all
triples as timeless. For scientific claims that's mostly right, but
not always: "AlphaFold2 is SOTA" was true in 2021, contested after
ESMFold (2022), dethroned by AlphaFold3 (2024). A synthesis chapter
pulling triples without dates can confidently assert the wrong
current-state claim.

**Scope**: Alembic migration adds `valid_from DATE`, `valid_to DATE`,
`confidence REAL DEFAULT 1.0` to `knowledge_graph`. Backfill
`valid_from = paper.year` for existing rows. Wiki 3D UI adds a
timeline slider. Default query: "currently valid". ~200 LOC + UI.

**Trade-off**: automatic `valid_to` inference from "X was dethroned
by Y" language is hard — manual invalidate via CLI is the MVP.
`confidence` is free to add now, useful later for weighted graph
queries.

**L1 test**: insert triple with `valid_to='2023-01-01'`; query with
`as_of='2024-01-01'`; expect empty result.

### P6 — Maintenance: `db repair` command (~3 h, pattern only)

**What**: `repair.py::scan_palace/prune/rebuild` — surgical
detection of corrupt index entries, targeted pruning, per-paper
rebuild. Halfway-house between `db stats` (read-only) and `db reset`
(sledgehammer).

**Why it fits**: sciknow currently has no middle ground between
those two. One paper's Qdrant points can go orphan (row exists but
no vector, or vice versa) and the only known fix is `db reset`.

**Scope**: new `sciknow corpus repair [--scan|--prune|--rebuild-paper
<doc_id>]`. Scan diffs PG `chunks.qdrant_point_id` against a Qdrant
scroll; prune removes orphans; rebuild-paper re-chunks + re-embeds
one document. ~150 LOC. Don't port the Chroma code — port the
shape.

**L1 test**: plant an orphan Qdrant point; `repair_scan()` returns
it in the report dict.

### P7 — BEIR retrieval evaluation harness (~8 h, pattern only)

**What**: `benchmarks/longmemeval_bench.py` — ephemeral ChromaDB
client, fresh collection per query, NDCG@k + recall@k per question
category. Mempalace's published BENCHMARKS.md improvement narrative
(96.6 → 97.8 → 98.4 → 99.4 → 100%) is an explicitly failure-pattern-
driven story: every hop traceable to a specific bad question.

**Why it fits**: sciknow has no quantitative retrieval eval. Every
reranker/RRF/embedding tweak ships on vibes. BEIR's `scifact`,
`scidocs`, `nfcorpus` are the right analogs.

**Scope**: new `sciknow/testing/beir_eval.py` + `sciknow test --beir
scifact`. Ephemeral in-memory Qdrant (don't touch the live
collection). Adds optional `ir_datasets` dep. ~500 LOC.

**L1 test**: 10-doc synthetic corpus + 3 queries; recall@k
calculation matches hand computation.

## Explicit skip list

- **AAAK dialect** (their own dedicated query language) — mempalace's
  own benchmark shows it regresses 12.4 points vs raw. sciknow's
  bge-m3 + cross-encoder handles the equivalent.
- **Palace / wing / room / drawer hierarchy** — sciknow already has
  books → chapters → sections → chunks + topic clusters + Louvain
  communities; a second orthogonal hierarchy is a regression.
- **BM25-in-convex-combination ranking** — RRF after separate
  retrievers + cross-encoder strictly dominates for scientific text.
  Mempalace themselves are moving toward bge-large hybrid on LoCoMo.
- **4-layer "wake-up" stack** — chat-UX pattern; sciknow's book-
  reader UX is different.
- **MCP server** — sciknow is a web reader for a researcher, not an
  agent-memory tool. Wrong product shape.
- **`normalize.py` / `strip_noise` / `convo_miner`** — transcript-
  specific; sciknow ingests PDFs.
- **Claude Code save hooks** — irrelevant here.
- **`onboarding.py` wizard** — sciknow has `project init` + the
  Phase 46f setup wizard already.

## External links worth bookmarking

- **LongMemEval paper** (Wu et al. 2024) —
  https://arxiv.org/abs/2410.10813. Useful if sciknow ever wants
  long-memory eval on top of BEIR.
- **LoCoMo benchmark** — https://snap-stanford.github.io/locomo/.
  Adversarial + temporal-inference categories interesting for
  autowrite claim-verification robustness.
- **Zep Graphiti** — https://github.com/getzep/graphiti. Temporal KG
  that inspired mempalace's `knowledge_graph.py`; read its edge-
  invalidation spec before shipping P5.
- **`benchmarks/BENCHMARKS.md`** — mempalace's failure-pattern-
  driven improvement story. Worth rereading periodically as the
  mental model for "port the failure analysis, not the code".
- **`benchmarks/HYBRID_MODE.md`** — clearest single-weekend writeup
  of hybrid search with temporal-proximity boost and quoted-phrase
  boost. sciknow's `hybrid_search.py` already has most; temporal
  proximity is the gap.
- **April 7 2026 correction note in the repo README** — pattern for
  how to honestly retract an over-claim. Similar shape worth
  emulating in our own benchmark posts.

## If shipping — suggested order

1. **P1** (query sanitiser, 2 h) — smallest, closes a silent failure.
2. **P3** (chunker version stamp, 1 h) — one-line safety rail.
3. **P2** (chunk-level dedup, 3 h) — immediate wins on any corpus
   that's seen `db expand` pull preprint variants.
4. **P4** (KG claim check in verify, 3 h) — new groundedness
   dimension for autowrite.
5. **P6** (`db repair`, 3 h) — middle ground between stats and reset.
6. **P5** (temporal triples, 4 h) — needs sign-off before the
   migration touches live data.
7. **P7** (BEIR harness, 8 h) — do when reranker / RRF weight tuning
   becomes something we want to measure not guess.

Total ≈ 24 hours across ~2–3 focused sessions. Each item is
independently shippable, tested by L1, and reverts cleanly.
