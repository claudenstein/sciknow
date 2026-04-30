# Credits & Acknowledgements

[&larr; Back to README](../README.md)

---

sciknow is built on the shoulders of excellent open-source projects and research. Below are the tools whose code or architecture we directly depend on, and the papers whose ideas we implemented.

## Open-Source Projects

| Project | Role in sciknow |
|---|---|
| [MinerU 2.5](https://github.com/opendatalab/MinerU) | Primary PDF converter — SOTA for scientific papers (MFD/MFR formula models, HTML tables, reading-order reconstruction) |
| [Marker](https://github.com/VikParuchuri/marker) | Fallback PDF converter — structured JSON block tree, Markdown output |
| [Qdrant](https://github.com/qdrant/qdrant) | Vector database — stores dense + sparse embeddings, payload filters, hybrid search |
| [Ollama](https://github.com/ollama/ollama) | Local LLM serving — all inference runs through the Ollama API |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | Embedding model — simultaneous dense (1024-dim) and sparse lexical vectors |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Cross-encoder reranker — final top-k selection after RRF fusion |
| [BERTopic](https://github.com/MaartenGr/BERTopic) | Topic clustering — UMAP + HDBSCAN + c-TF-IDF pipeline |
| [UMAP](https://github.com/lmcinnes/umap) | Dimensionality reduction for BERTopic |
| [hdbscan](https://github.com/scikit-learn-contrib/hdbscan) | Density-based clustering for BERTopic |
| [FastAPI](https://github.com/fastapi/fastapi) | Web framework — SSE streaming, REST API for the browser editor |
| [Typer](https://github.com/tiangolo/typer) | CLI framework — all `sciknow` subcommands |
| [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) + [Alembic](https://github.com/sqlalchemy/alembic) | ORM and schema migrations — PostgreSQL storage layer |
| [Rich](https://github.com/Textualize/rich) | Terminal UI — live dashboards, progress bars, autowrite panels |
| [PyMuPDF (fitz)](https://github.com/pymupdf/PyMuPDF) | PDF metadata extraction — embedded XMP, title/author fields |
| [python-docx](https://github.com/python-openxml/python-docx) | DOCX export |
| [WeasyPrint](https://github.com/Kozea/WeasyPrint) | HTML → PDF rendering for the Phase 31 web reader PDF export feature |
| [difflib](https://docs.python.org/3/library/difflib.html) | In-browser version diffs between draft snapshots |
| [uv](https://github.com/astral-sh/uv) | Fast Python package manager and virtual environment |

## Research Papers

These papers directly shaped the architecture, algorithms, and prompting strategies in sciknow.

### Retrieval & Search

- **Reciprocal Rank Fusion** — Cormack, Clarke & Buettcher (SIGIR 2009). [RRF outperforms Condorcet and individual rank learning methods](https://dl.acm.org/doi/10.1145/1571941.1572114). Basis for our hybrid search fusion.
- **BGE M3-Embedding** — Chen et al. (2024). [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216). The embedding model used throughout.
- **BGE Reranker** — Xiao et al. (2024). [C-Pack: Packaged Resources To Advance General Chinese Embedding](https://arxiv.org/abs/2309.07597). Basis for cross-encoder reranking.

### Knowledge Graphs & Structured Retrieval

- **GraphRAG** — Edge et al., Microsoft (2024). [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130). Concept-level entity extraction, relationship triples, and community summaries — implemented in `wiki graph`.
- **LLM-wiki (Karpathy pattern)** — Andrej Karpathy, [Twitter/X, 2023](https://x.com/karpathy/status/1756130027985752370). "Compile the knowledge, don't retrieve it." Inspires our `wiki compile` layer: synthesize papers once, query the compilation.

### Topic Modeling

- **BERTopic** — Grootendorst (2022). [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794). Our `catalog cluster` implementation.
- **UMAP** — McInnes, Healy & Melville (2018). [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426).
- **HDBSCAN** — Campello, Moulavi & Sander (2013). Density-based clustering algorithm underlying BERTopic.
- **LLM-Guided Clustering** — (ACL 2025). [LLM-Guided Semantic-Aware Clustering](https://aclanthology.org/2025.acl-long.902/). Informed the hybrid embedding + LLM naming approach.

### Self-Correcting & Grounded Generation

- **Self-RAG** — Asai et al. (2023). [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511). Inspires our retrieval evaluation and per-claim grounding checks.
- **CRAG (Corrective RAG)** — Yan et al. (2024). [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884). Query reformulation and corrective retrieval on low-relevance results.
- **Agentic RAG Survey** — (2025). [A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136). Framework for multi-step retrieval agents.

### Document Parsing

- **OmniDocBench** — He et al. (2024). [OmniDocBench: Benchmarking diverse PDF document parsing with comprehensive annotations](https://arxiv.org/abs/2412.07626). The benchmark establishing MinerU 2.5 as SOTA for scientific PDF parsing.

### Long-Form Writing

- **TreeWriter** — (arxiv, Jan 2025). [TreeWriter: Hierarchical document planning and grounded writing](https://arxiv.org/abs/2601.12740). Paragraph-level tree planning before drafting — implemented in `tree_plan()` prompt.
- **DOME** — (NAACL 2025). [Dynamic Hierarchical Outlining for Multi-document Summarization and Writing](https://aclanthology.org/2025.naacl-long.63/). Fused planning + writing stages.
- **Outline-guided Generation** — (NAACL Industry 2025). [Improving long-form generation with outline-guided hierarchical planning](https://aclanthology.org/2025.naacl-industry.20.pdf).
- **Autowrite loop** — Inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch). Generate → score → revise until quality target is met.

### Multimodal Retrieval

- **RAG-Anything** — HKUDS (2025). [RAG-Anything: All-in-One Multimodal RAG](https://github.com/HKUDS/RAG-Anything). Informed our table and equation tagging and retrieval approach.
- **ManuRAG** — (2025). [ManuRAG: Multimodal Retrieval for Scientific Manuscripts](https://arxiv.org/pdf/2601.15434). Table-aware context construction.
- **Multimodal LLM Table Understanding** — (ACL TRL 2025). [Understanding Tables with Multimodal LLMs](https://aclanthology.org/2025.trl-1.10.pdf).

### Knowledge Graph Construction

- **KG-RAG** — (Nature 2025). [Knowledge Graph-Enhanced RAG](https://www.nature.com/articles/s41598-025-21222-z). Graph-aware retrieval for scientific domains.
- **LLM-empowered KG Construction** — (2025). [Automated Knowledge Graph Construction with LLMs](https://arxiv.org/abs/2510.20345).

---

*sciknow is an independent research tool. All referenced projects retain their respective licenses.*
