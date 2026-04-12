# SciKnow

A local-first scientific knowledge system that ingests papers, builds a compiled knowledge wiki, and writes grounded scientific books — all running on your own hardware. No cloud APIs.

**Ingest PDFs** (scanned or text) **&rarr; search & synthesize** across your library **&rarr; write entire books** with iterative AI review, all from the browser or CLI.

---

## Table of Contents

- [Features](#features)
- [Workflow](#workflow-from-pdfs-to-book)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Documentation](#documentation)
- [Hardware Requirements](#hardware-requirements)
- [Credits](#credits--acknowledgements)

---

## Features

**Ingestion & Library Management**
- **PDF ingestion** — MinerU 2.5 (SOTA for scientific papers) with Marker fallback. Handles scanned and text PDFs, tables, equations, figures
- **Metadata extraction** — 4-layer cascade: embedded PDF → Crossref → arXiv → LLM
- **Citation graph** — extracts references, cross-links corpus papers, boosts highly-cited papers in search
- **Collection expansion** — follows citations to discover + download open-access papers from 6 sources
- **Topic clustering** — BERTopic (UMAP + HDBSCAN + c-TF-IDF) assigns papers to named thematic clusters in seconds

**Search & Retrieval**
- **Hybrid search** — dense + sparse + full-text, fused with RRF, reranked with a cross-encoder
- **Self-correcting RAG** — evaluates retrieval quality, reformulates if poor, checks answer grounding
- **Multimodal awareness** — table/equation tagging for filtered retrieval

**Knowledge Wiki** (Karpathy LLM-wiki pattern)
- **Compiled knowledge layer** — papers synthesized into interconnected wiki pages (summaries, concept pages, synthesis overviews)
- **Knowledge graph** — entity-relationship triples extracted during compilation (GraphRAG-style)
- **Consensus mapping** — tracks agreement/disagreement across the corpus over time
- **Contradiction detection** — LLM-based lint finds disagreements between papers

**Book Writing Platform**
- **Structured projects** — book → chapter hierarchy with LLM-generated outlines and per-chapter custom sections
- **Iterative refinement** — write → review → revise loop with 5-dimension scoring and claim verification
- **Autowrite** — autonomous convergence loop: generates, scores, verifies, revises until quality target is met
- **TreeWriter planning** — hierarchical paragraph-level plans before drafting
- **Web reader** — browser-based authoring with live LLM streaming, corkboard view, chapter reader, argument maps, citation popovers, snapshots, version diffs
- **Compute dashboard** — book-level GPU compute ledger: cumulative tokens, wall time, and per-operation breakdown (write/review/revise/argue/gaps/autowrite) across every LLM call
- **Tools panel** — CLI-parity in the browser: hybrid corpus search, similarity search, multi-paper synthesis, topic-cluster browser, and one-click corpus enrich / citation expand with live log streaming
- **Multi-format export** — Markdown, HTML, BibTeX, LaTeX, DOCX with global citation dedup

**Infrastructure**
- **All local** — PostgreSQL + Qdrant + Ollama, no cloud APIs, no Docker
- **Backup & restore** — portable archives for migrating between machines

---

## Workflow: From PDFs to Book

```
 1  INGEST        PDFs → chunks in PostgreSQL + vectors in Qdrant
 2  ENRICH        Fill missing DOIs and metadata from Crossref/OpenAlex
 3  EXPAND        Follow citations → download related open-access papers
 4  CLUSTER       Group papers into thematic topics (BERTopic, seconds)
 5  WIKI          Build the compiled knowledge wiki (summaries + concepts + KG)
 6  EXPLORE       Ask questions, search, synthesize — understand your corpus
 7  BOOK          Create → outline → plan → write → review → export
```

> Steps 2-5 are optional but each improves downstream quality. You can ask questions right after step 1.

```bash
# Quick start — the essential commands
sciknow ingest directory ./papers/
sciknow db stats
sciknow catalog cluster
sciknow wiki compile
sciknow ask question "What is total solar irradiance?"
sciknow book create "My Book"
sciknow book outline "My Book"
sciknow book plan "My Book"
sciknow book serve "My Book"              # open browser, write from there
sciknow book autowrite "My Book" --full   # or let it write autonomously
sciknow book export "My Book" --format latex -o manuscript.tex
```

### When to use what

| I want to... | Use this |
|---|---|
| **Ask a quick question** about my papers | `sciknow ask question "..."` |
| **Get a pre-synthesized answer** from compiled knowledge | `sciknow wiki query "..."` |
| **Synthesize findings** across multiple papers | `sciknow ask synthesize "topic"` |
| **Write a full book** with chapters and review | `sciknow book ...` |
| **Browse compiled knowledge** about a concept | `sciknow wiki show concept-slug` |
| **Find contradictions** in your corpus | `sciknow wiki lint --deep` |
| **Map evidence for/against a claim** | `sciknow book argue "claim"` |
| **Find gaps** in a book project | `sciknow book gaps "Book"` |

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/claudenstein/sciknow
cd sciknow
bash scripts/setup.sh

# 2. Pull the main LLM
ollama pull qwen3.5:27b

# 3. Configure (set your email for Crossref polite pool)
nano .env

# 4. Initialize
sciknow db init

# 5. Ingest your papers
sciknow ingest directory ./papers/
```

See [Installation Guide](docs/INSTALLATION.md) for manual installation, Ollama performance tuning, and full configuration reference.

---

## Architecture Overview

```
PDFs ──→ MinerU 2.5 ──→ Metadata ──→ Chunker ──→ bge-m3 ──→ PostgreSQL + Qdrant
                                                                      │
Query ──→ Dense + Sparse + FTS ──→ RRF fusion ──→ Reranker ──→ Ranked results
                                                                      │
                                                              LLM (Ollama) ──→ Answer
```

Three services, all native (no Docker): **PostgreSQL 16** (relational + full-text), **Qdrant** (vectors), **Ollama** (LLM inference).

See [Architecture](docs/ARCHITECTURE.md) for the full system diagram, database schema, AI model details, and project structure.

---

## Documentation

| Document | Contents |
|---|---|
| **[Architecture](docs/ARCHITECTURE.md)** | System diagram, project structure, database schema, AI models, service layer pattern |
| **[Installation](docs/INSTALLATION.md)** | Setup script, manual install, Ollama tuning, configuration reference, hardware requirements |
| **[CLI Reference](docs/CLI.md)** | Complete command reference for all `sciknow` subcommands |
| **[Ingestion Pipeline](docs/INGESTION.md)** | PDF conversion (MinerU/Marker), metadata extraction, section-aware chunking, embedding |
| **[Retrieval & RAG](docs/RETRIEVAL.md)** | Hybrid search, RRF fusion, reranking, self-correcting RAG, writing assistant |
| **[Book Writing System](docs/BOOK.md)** | Book workflow, autowrite convergence, web reader, export formats, tips |
| **[Operations](docs/OPERATIONS.md)** | Backup/restore, reference expansion, metadata enrichment, citation graph, development notes |
| **[Testing Protocol](docs/TESTING.md)** | The 3-layer smoke harness (`sciknow test`), what each layer covers, how to add new checks |
| **[Research & Innovations](docs/RESEARCH.md)** | All implemented techniques with research basis: BERTopic, GraphRAG, Self-RAG, TreeWriter, Karpathy wiki, consensus mapping |
| **[Credits](docs/CREDITS.md)** | Open-source projects and research papers that sciknow builds on |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3090 (24 GB VRAM) | RTX 3090 + remote GPU server |
| RAM | 32 GB | 64 GB |
| Storage | 500 GB SSD | 2 TB NVMe |
| OS | Ubuntu 22.04+ | Ubuntu 22.04+ |

**VRAM budget on 3090 (24 GB):** bge-m3 (~2.2 GB) + qwen3.5:27b (~18 GB) + bge-reranker (~0.5 GB) = fits comfortably.

**Remote GPU:** Set `OLLAMA_HOST=http://your-gpu-server:11434` in `.env`. Zero code changes.

---

## Credits & Acknowledgements

sciknow builds on excellent open-source projects and research. Full details in [Credits](docs/CREDITS.md).

**Key projects:** [MinerU](https://github.com/opendatalab/MinerU), [Marker](https://github.com/VikParuchuri/marker), [Qdrant](https://github.com/qdrant/qdrant), [Ollama](https://github.com/ollama/ollama), [bge-m3](https://huggingface.co/BAAI/bge-m3), [BERTopic](https://github.com/MaartenGr/BERTopic), [FastAPI](https://github.com/fastapi/fastapi), [Typer](https://github.com/tiangolo/typer), [Rich](https://github.com/Textualize/rich), [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy), [uv](https://github.com/astral-sh/uv)

**Key research:** [RRF (SIGIR 2009)](https://dl.acm.org/doi/10.1145/1571941.1572114), [BGE M3-Embedding (2024)](https://arxiv.org/abs/2402.03216), [GraphRAG (Microsoft 2024)](https://arxiv.org/abs/2404.16130), [BERTopic (2022)](https://arxiv.org/abs/2203.05794), [Self-RAG (2023)](https://arxiv.org/abs/2310.11511), [TreeWriter (2025)](https://arxiv.org/abs/2601.12740), [OmniDocBench (2024)](https://arxiv.org/abs/2412.07626), [Karpathy LLM-wiki](https://x.com/karpathy/status/1756130027985752370)

---

*sciknow is an independent research tool. All referenced projects retain their respective licenses.*
