# Retrieval & RAG

[&larr; Back to README](../README.md)

---

Hybrid search combining three signal types, fused with Reciprocal Rank Fusion (RRF), reranked with a cross-encoder, and boosted by citation count.

## How Search Works

**Step 1: Query embedding**

The query is embedded with the same `BAAI/bge-m3` model used at ingest time, producing both a dense vector (1024-dim) and a sparse lexical weight vector.

**Step 2: Three parallel search legs**

| Leg | Backend | Strengths |
|---|---|---|
| Dense vector | Qdrant | Semantic similarity — finds conceptually related text even if phrasing differs |
| Sparse vector | Qdrant | Keyword matching — precise on technical terms, acronyms, species names |
| Full-text search | PostgreSQL `tsvector` | Classical BM25-style relevance, stemming, phrase proximity |

Each leg returns up to 50 candidates (configurable with `--candidates`).

**Step 3: RRF fusion**

Results from all three legs are merged using Reciprocal Rank Fusion:

```
score(chunk) = Σ weight_i / (60 + rank_i)
```

Default weights: dense=1.0, sparse=1.0, FTS=0.5. The top 50 fused candidates proceed to reranking.

**Step 4: Citation-count boost**

Papers cited by more corpus papers receive a log-dampened score boost: `score *= (1 + 0.1 * log2(1 + citation_count))`. Controlled by `CITATION_BOOST_FACTOR` (0 to disable). The boost is gentle — retrieval signals still dominate.

**Step 5: Cross-encoder reranking**

`BAAI/bge-reranker-v2-m3` scores each `(query, chunk_text)` pair directly. This is slower but much more accurate than embedding similarity alone. Returns top `--top-k` results (default 10).

**Step 6: Metadata hydration**

Full chunk content and bibliographic metadata (title, authors, year, DOI, journal, citation count) are fetched from PostgreSQL and attached to each result.

**Optional: LLM query expansion** (`--expand` / `-e`)

Before step 1, sends the query to `LLM_FAST_MODEL` to add synonyms, acronyms, and related terms. The expanded query feeds the dense + sparse legs; PostgreSQL FTS keeps the original for precision.

---

## Filters

All filters are applied before the vector search (as Qdrant pre-filters and SQL WHERE clauses), so they do not reduce recall within the matching set:

| Filter | Flag | Example |
|---|---|---|
| Year range | `--year-from`, `--year-to` | `--year-from 2015 --year-to 2023` |
| Domain tag | `--domain` | `--domain climatology` |
| Section type | `--section` | `--section methods` |
| Topic cluster | `--topic` | `--topic "Solar Irradiance"` |
| Has table | `--has-table` | Filter for chunks containing tables |
| Has equation | `--has-equation` | Filter for chunks containing equations |

---

## Question Answering (RAG)

`sciknow ask question` retrieves the most relevant passages from your library and asks the configured LLM to answer grounded strictly in those sources.

1. **Retrieval** — same hybrid search pipeline. Produces the top `--context-k` chunks (default 8).
2. **Context assembly** — chunks are numbered `[1]`, `[2]`, ... and formatted with paper title, year, and section type as a header.
3. **LLM completion** — the context + question is sent to Ollama. The response is **streamed token by token** to the terminal.
4. **Sources** — after the answer, the source list (title, year, authors, DOI) is printed.

### Self-Correcting RAG (`--self-correct`)

When enabled, adds two extra steps:

1. **Retrieval evaluation** — LLM assesses whether retrieved chunks are relevant to the question. If relevance is low, the query is reformulated and search is retried.
2. **Grounding check** — after generation, LLM evaluates whether the answer is grounded in the retrieved evidence. Reports ungrounded claims.

Based on Self-RAG and CRAG research. Achieves ~5.8% hallucination rate vs 12-14% for standard RAG.

### Context window sizing

| Model | Context (`--num-ctx`) | Max chunks |
|---|---|---|
| qwen3.5:27b | 16 384 tokens (default) | ~30 chunks |
| qwen3:8b | 8 192 tokens (default) | ~15 chunks |

Each chunk is ~512 tokens + headers. The default `--context-k 8` uses ~4 500 tokens, well within both budgets.

---

## Writing Assistant

Extends the RAG pipeline with longer-form generation tasks. All commands stream output and print sources.

### `ask synthesize`

Retrieves the most relevant passages for a topic (default `--context-k 12`) and asks the LLM to write a structured synthesis covering: key findings, methodological approaches, consensus, and open questions.

### `ask write`

Drafts a specific section on a given topic. The search query is biased toward the target section type to retrieve the most relevant content.

### `ask write --save`

Persist the draft to the database. Optionally associate it with a book and chapter:

```bash
sciknow ask write "solar activity proxies" --section introduction --save
sciknow ask write "ocean heat content trends" --section results --save \
    --book "Global Cooling" --chapter 3
```

---

## `ask` vs `wiki query` — When to Use Which

| Command | Searches | Best for |
|---|---|---|
| `ask question` | Raw paper chunks via hybrid retrieval | Specific factual questions needing exact passages |
| `wiki query` | Compiled wiki pages | Conceptual questions wanting synthesized answers |
| `ask synthesize` | Raw chunks (more context) | Topic overviews before writing |
| `book write` | Raw chunks + cross-chapter context | Structured book content |
