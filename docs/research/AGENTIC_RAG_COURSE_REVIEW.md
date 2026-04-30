# `production-agentic-rag-course` — portability review

**Source**: https://github.com/jamwithai/production-agentic-rag-course
(on-disk branding: "Phase 1 RAG Systems — arXiv Paper Curator", 5.5k
GitHub stars, 7-week full-stack course, ~5–6k LOC excl. notebooks).
**Review date**: 2026-04-14.

Bottom line in one paragraph: this repo is a well-engineered
*teaching* artifact, not a source of novel production RAG techniques.
Sciknow already has hybrid search + RRF + cross-encoder reranker +
self-RAG grading + query reformulation + claim verification, most of
which are more sophisticated than the course equivalents. Four
patterns are genuinely portable. The rest is either already in
sciknow, actively worse than what sciknow has, or conflicts with the
local-first / no-Docker / single-user principles.

## What the course actually teaches

Week-by-week full-stack build: Docker host → OpenSearch index →
Airflow DAG for arXiv ingestion → Docling PDF parser → FastAPI +
Gradio UI → LangGraph agent state machine → Telegram bot. Tech is
**unapologetically LangChain/LangGraph-centric**, with **Jina v3**
cloud embeddings, **Langfuse** tracing, **Redis** exact-match cache,
**Ollama `llama3.2:1b`** as the local LLM default. Philosophy: "BM25
foundation before vectors, RRF fusion with k=60, LangGraph agent with
guardrail → retrieve → grade → rewrite → generate."

Positioned vs other courses: closer to a "build a SaaS in a week"
tutorial than an eval-oriented curriculum (Hamel Husain-style). Zero
RAGAS / TruLens / LLM-as-judge / annotation-workflow content; the
Langfuse integration is observability only, not evaluation.

## Already in sciknow (with variations worth noting)

- **Hybrid retrieval with RRF fusion** — course has dense+sparse via
  OpenSearch; sciknow has dense+sparse+PG-FTS *plus* a cross-encoder
  reranker. Sciknow is strictly ahead.
- **Document grading for self-RAG** — course returns binary yes/no on
  whole context; sciknow's `retrieval/self_rag.py::evaluate_retrieval`
  returns a continuous 0–1 score used to decide reformulation.
  Sciknow is richer.
- **Query reformulation** — course has a `max_retrieval_attempts=2`
  retry counter; sciknow has the same loop. One minor idea worth
  borrowing: the course's `QueryRewriteOutput(rewritten_query,
  reasoning)` stores the *reasoning* alongside the rewritten query,
  useful for debugging. 10-line add.
- **Section-based chunking** — course uses one 600-word fixed window
  with 100-word overlap; sciknow has per-type parameters + context
  headers + per-backend dispatch. Sciknow is strictly ahead.
- **Ollama structured output via Pydantic schema** — both use it.
  Sciknow calls it via `format=schema`, course via
  `llm.with_structured_output()` from LangChain (same thing
  underneath).

## Porting candidates — ranked

### 1. User feedback capture → labeled-positives feedstock
Course pattern: `POST /feedback {trace_id, score, comment}` pushes
into Langfuse. **Sciknow gap**: the parked LambdaMART learn-to-rank
(`docs/research/EXPAND_RESEARCH.md`) needs ≥500 labeled positives and has no
collection pipe. A `feedback` table keyed on `(query, chunk_ids, ts,
score, comment)` + a one-keystroke 👍/👎 on every `sciknow ask` +
book-reader answer fills that gap passively.
- Scope: table + CLI subcommand + web endpoint, ~400 LOC, 6–8 h.
- Trade-off: risk of nagging users — make it opt-in per project.
- L1 guard: insert a `(chunk_id, 1.0, "good")` row, query back,
  assert count=1 and score preserved.

### 2. Optional pre-retrieval guardrail (soft fast-path, not hard gate)
Course pattern: LLM scores query 0–100 for corpus fit; below threshold
routes to "out_of_scope". **Sciknow use case**: short-circuit a full
retrieve+rerank+generate spin when the query is obviously off-corpus
(user typed a cooking question into a climate KB). Ship as a *soft*
gate only — if score <40, skip retrieval and answer with an unsourced
LLM response marked "low corpus relevance", never refuse to answer.
Per-project `corpus_description` field feeds the scoring prompt.
- Scope: `retrieval/guardrail.py` + project config field, ~170 LOC,
  4–6 h.
- Trade-off: adds one fast-model call (~0.5–2 s) per query. Mitigate
  with a 7-day guardrail-result cache keyed on query hash.
- L2 guard: send "what is the capital of France" to a climate KB,
  assert `retrieval_attempted == False` in the response metadata.

### 3. Reasoning-steps list on the response envelope
Course pattern: every agentic response returns
`reasoning_steps: ["Validated query scope (85/100)", "Retrieved 3
relevant docs", …]`. **Sciknow gap**: the book autowrite loop already
*emits* these events via the SSE stream, but they don't persist.
Generalize to a `drafts.reasoning_trace JSON` column populated from
existing events in `_observe_event_for_stats()`; also emit on plain
`sciknow ask` output. No new LLM calls, zero extra latency.
- Scope: alembic migration + observer hook + CLI display, ~250 LOC,
  3–4 h.
- Trade-off: storage grows with log data — cap at 50 steps per draft;
  opt-in per project.
- L2 guard: run `book autowrite` on a tiny section, assert the draft's
  `reasoning_trace` contains `["retrieve", "score", "revise",
  "verify"]` in order.

### 4. Langfuse-pattern span tracing — self-hosted as a SQLite table
Course pattern: `with rag_tracer.trace_embedding(…):` context-manager
spans with start/end timestamps and input/output payloads, pushed to a
Langfuse server. **Sciknow gap**: no structured per-span timing once
DGX Spark is online and latency analysis matters. Port the
context-manager API but write to a local `spans` table keyed on
`(trace_id, span_name, start_ts, end_ts, metadata_json)`. Add
`sciknow spans tail` CLI to pretty-print recent traces. **Skip
Langfuse itself** — the service-with-Docker conflicts with
no-Docker principle; the 10% of Langfuse that matters fits in ~500
LOC locally.
- Scope: `sciknow/observability/tracer.py` + migration + CLI, ~450
  LOC, 6–10 h.
- Trade-off: reinvents 10% of Langfuse.
- L1 guard: open a span, `sleep(50ms)`, close; assert row exists with
  `end_ts - start_ts >= 50ms`.

## Explicit skip list

- **LangGraph framework itself** — sciknow's generator-based event
  pipeline in `core/book_ops.py` already does "typed events between
  stages" in ~1/3 the LOC with no dependency. Conditional-edges are
  just a switch statement.
- **LangChain `Document`, `Tool`, `with_structured_output()`** —
  sciknow uses Pydantic + `ollama.chat()` directly; Ollama's
  `format=schema` is equivalent.
- **Redis exact-match cache** — single-user local tool; exact-query
  repetition is rare, and a new service conflicts with no-Docker.
- **OpenSearch** — sciknow's Qdrant + PG-FTS + bge-m3 stack is
  strictly more capable (payload-filter indexes, sparse+dense from
  one forward pass, cross-encoder reranker). Migrating would be a
  regression.
- **Jina cloud embeddings** — breaks local-first.
- **Airflow DAGs** — `sciknow corpus expand` + `watchlist` cron is
  lighter and purpose-built.
- **Telegram bot** — sciknow is desktop; no value.
- **Gradio UI** — `sciknow book serve` is strictly more capable.
- **200-word answer cap in rag_system prompt** — scientific answers
  often need more; sciknow's prompts are appropriately unbounded.
- **Docling PDF parser** — MinerU 2.5 + Marker scores higher on
  OmniDocBench v1.6 (95.69 vs Docling's <90 typical).
- **Retrieval-decision prompt ("RETRIEVE or RESPOND")** — sciknow
  users always want sourced answers; a direct-answer path undermines
  the tool's identity.

## Worth bookmarking (links)

- **RRF original paper** (Cormack, Clarke, Büttcher 2009,
  `plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf`) —
  canonical justification of `k=60`. Cite from sciknow's
  `retrieval/hybrid_search.py` docstring.
- **Langfuse v3 SDK docs** (design reference only — do not install):
  context-manager span API (`start_as_current_span`, `update`, `end`).
- **Course author's substack** (`jamwithai.substack.com`) — skimmable
  summaries of the architectural decisions. No load-bearing content
  beyond the notebooks.

## If shipping: priority order

1. **Reasoning-steps trace** (#3) — lowest risk, no new LLM calls,
   improves debug experience on autowrite immediately.
2. **User feedback capture** (#1) — unblocks the LambdaMART upgrade
   path in `EXPAND_RESEARCH.md`.
3. **Span tracing** (#4) — start now so the data is already there when
   DGX Spark arrives and perf tuning becomes urgent.
4. **Guardrail** (#2) — last, because the signal-to-noise is weakest;
   sciknow's corpus filter already handles most out-of-scope queries
   by producing low-relevance retrievals.
