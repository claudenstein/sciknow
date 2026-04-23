import logging
import os
from pathlib import Path

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_data_dir() -> Path:
    """Phase 43a — per-project default for ``data_dir``.

    Delegates to ``core.project.get_active_project()`` so each project
    gets its own ``data/`` tree. Wrapped in a factory (rather than a
    module-level ``Path`` literal) because the active project is
    resolved at settings-construction time, not at module import time.

    Env override (``DATA_DIR=...`` in .env) still wins — pydantic
    checks the env var before falling back to ``default_factory``.
    """
    from sciknow.core.project import get_active_project
    return get_active_project().data_dir


def _default_pg_database() -> str:
    """Phase 43a — per-project default for ``pg_database``.

    Same contract as ``_default_data_dir``: env override wins, else
    the active project's DB name (``sciknow`` for legacy default,
    ``sciknow_<slug>`` for real projects).
    """
    from sciknow.core.project import get_active_project
    return get_active_project().pg_database


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base data directory — Phase 43a: defaults to the active project's
    # data dir (see ``sciknow.core.project``). ``DATA_DIR`` env var still
    # wins for single-tenant / testing scenarios.
    data_dir: Path = Field(default_factory=_default_data_dir)

    # PostgreSQL
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "sciknow"
    pg_password: str = "sciknow"
    # Phase 43a: defaults to the active project's DB name
    # (``sciknow`` legacy, ``sciknow_<slug>`` per project). ``PG_DATABASE``
    # env var still wins.
    pg_database: str = Field(default_factory=_default_pg_database)

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Ollama
    ollama_host: str = "http://localhost:11434"

    # Models
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    # Phase 54.6.279 — dual-embedder split. When `dense_embedder_model`
    # is set AND different from `embedding_model`, retrieval uses the
    # named dense embedder (e.g. Qwen3-Embedding-4B) against a
    # sidecar Qdrant collection, while bge-m3 continues to provide
    # sparse + ColBERT vectors from the primary collection. Validated
    # by the comprehensive A/B bench (docs/PHASE_LOG 54.6.278) with
    # +0.035 MRR@10 full-stack lift. Unset → single-embedder mode
    # (pre-279 behaviour unchanged).
    #
    # `dense_sidecar_collection` names the Qdrant collection holding
    # Qwen3 dense vectors. When left at None, the script uses the
    # convention `<qdrant_prefix>_ab_<slug>_papers` (same as the
    # A/B harness). Explicit override wins.
    dense_embedder_model: str | None = None
    dense_embedder_dim: int = 2560
    dense_sidecar_collection: str | None = None
    llm_model: str = "qwen2.5:32b-instruct-q4_K_M"
    llm_fast_model: str = "qwen3:30b-a3b"
    # Phase 54.6.55 — optional per-role override for `book review`.
    # Falls back to llm_model when unset. Set when the global
    # LLM_MODEL is fast-but-shallow and you want a deeper critic
    # for the review pass specifically (see 2026-04-17-full bench:
    # gemma3:27b-it-qat wins book_review judge 100% / dims 5/5 vs
    # the unified qwen default's 71.4% / 3/5).
    book_review_model: str | None = None
    # Phase 54.6.243 — optional per-role override for book writing
    # (both `book write` and `book autowrite` drafting passes; does
    # NOT cover review, scoring, or verification). Falls back to
    # llm_model when unset. Added after the 54.6.243 focused 3-way
    # bench showed qwen3.6:27b-dense beats qwen3:30b-a3b-instruct
    # (the current global LLM_MODEL) on write_section by 5× on
    # citation density (7.08 vs 1.60 cites/100w) at comparable
    # wall-clock (13.8s vs 6.1s for a 150-word target). Only wire
    # it here — substituting the global LLM_MODEL would hurt
    # wiki_compile and extract_kg where qwen3:30b-a3b wins.
    book_write_model: str | None = None
    # Phase 54.6.297 — optional per-role override for `book outline`
    # (both the 3-candidate tournament in the CLI and the single-pass
    # web endpoint) AND the per-chapter `_grow_sections_llm` call.
    # Falls back to llm_model when unset.  Motivated by the 54.6.243
    # split: `book write` uses qwen3.6:27b-dense (writing wins), but
    # outline is a structured-JSON task where qwen3:30b-a3b still wins
    # the 2026-04-17 sweep (9/10 tasks including JSON-heavy ones like
    # compile_summary, extract-kg, wiki_consensus).  Keep the default
    # conservative; run `uv run python scripts/bench_outline_model.py`
    # to A/B before flipping it on your corpus.
    book_outline_model: str | None = None
    # Phase 54.6.59 — optional per-role override for the autowrite
    # score + rescore steps (NOT verify/CoVe, which stay on the writer
    # model). Falls back to llm_model when unset. The 2026-04-17-full
    # bench showed gemopus4:26b-a4b-q4_K_M gives the biggest good/bad
    # ranking gap (0.60 vs qwen's 0.27) while ranking correctly and
    # running at 6-7s per scoring call — same speed as qwen, better
    # discrimination. All other tasks on gemopus4 are broken, so wire
    # it here only — never as the global LLM_MODEL.
    autowrite_scorer_model: str | None = None
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Phase 54.6.74 (#1b) — vision-LLM default for `db caption-visuals`.
    # Overrides the CLI default when set. Lets the 54.6.74 VLM sweep
    # result (whichever model wins the judge pairwise on the user's
    # own corpus) persist without editing the CLI source. Unset →
    # CLI default (qwen2.5vl:32b) wins.
    visuals_caption_model: str | None = None

    # Phase 54.6.81 (#10 part 2) — per-paper-type retrieval weighting.
    # When True, hybrid_search multiplies each candidate's rrf_score by
    # a type-specific weight: peer_reviewed/preprint/thesis/book_chapter
    # stay at 1.0; editorial/policy drop to 0.7; unknown 0.8;
    # opinion 0.4 (hardest hit). Off by default — enable only after
    # `sciknow db classify-papers` has populated paper_type on a
    # meaningful fraction of the corpus, otherwise everything uses the
    # 'unknown' fallback.
    paper_type_weighting: bool = False
    # Optional override for the default weights — pass a JSON dict like
    # {"opinion": 0.2, "editorial": 0.5}; merged over the defaults.
    paper_type_weights: dict | None = None

    # Phase 54.6.70 (#9) — co-citation / bib-coupling retrieval boost.
    # Gentle log-dampened multiplier applied to candidates whose document
    # has citation-graph edges to the top-N anchor set from the same
    # query's retrieval. Default 0.0 (OFF) because the 54.6.70 A/B on the
    # 30-query retrieval bench regressed MRR 0.514→0.496: with the current
    # ~4.5% in-corpus citation resolution the signal is sparse and biased
    # toward popular papers rather than true-topic neighbors. Set to 0.1
    # in .env to A/B again once the corpus has denser in-corpus citations.
    cocite_boost_factor: float = 0.0

    # Crossref polite pool
    crossref_email: str = "user@example.com"

    # Phase 54.6.217 (roadmap 3.0.1 closure) — CORE (core.ac.uk) API
    # key. Free after registering at https://core.ac.uk/services/api;
    # the downloader uses it to query CORE's aggregator of
    # institutional-repository PDFs, which catches preprints that
    # never hit arXiv / OSF. Empty/missing → the CORE resolver is
    # skipped gracefully in the OA cascade (install still works,
    # just without that source).
    core_api_key: str | None = None

    # Phase 54.6.227 (roadmap 3.4.3 Phase 1) — enable ColBERT late-
    # interaction vectors on the abstracts collection. bge-m3
    # produces colbert tokens in the same forward pass as dense +
    # sparse, so the extra compute at embed time is ~0; the
    # tradeoff is storage: one abstract produces ~150 token vectors
    # × 1024 dim ≈ 600 kB, so a 1k-abstract corpus balloons by ~600
    # MB on the abstracts collection. Default False — opt-in per the
    # roadmap's "cheap experiment" framing. When True, the abstracts
    # collection is created with a `colbert` multi-vector field and
    # `embed_abstract()` populates it. Existing pre-flip abstracts
    # stay dense-only until re-embedded.
    enable_colbert_abstracts: bool = False

    # PDF converter backend.
    #
    # Values:
    #   "auto"            — try MinerU pipeline → MinerU 2.5 Pro VLM → Marker (default)
    #   "mineru"          — MinerU pipeline backend (legacy + CPU-friendly)
    #   "mineru-vlm-pro"  — MinerU 2.5-Pro VLM (95.69 on OmniDocBench v1.6,
    #                       SOTA per the Pro paper). Requires `mineru[vlm]`
    #                       extras + a GPU with ~4GB free VRAM. Slower than
    #                       pipeline on CPU.
    #   "marker"          — datalab-to/marker (legacy fallback)
    #
    # Phase 21: MinerU 2.5-Pro was initially opt-in because the 1.2B VLM
    # model was brand-new and required extra deps (vllm or transformers).
    # Phase 54.6.212 (roadmap 3.1.6 Phase 2): default flipped — "auto"
    # now dispatches VLM-Pro → pipeline → Marker JSON → Marker markdown
    # (premium-first fallback chain). The VLM-Pro attempt is silently
    # skipped if `mineru[vllm]` / `mineru[transformers]` extras aren't
    # installed, so existing installations without VLM deps continue
    # working under the pipeline fallback. Setting this to `mineru`
    # explicitly pins the deprecated pipeline backend (emits a
    # one-shot deprecation warning).
    pdf_converter_backend: str = "auto"

    # Phase 21 — explicit override for the VLM model name when running
    # MinerU 2.5-Pro. Empty string falls back to the package default
    # (currently MinerU2.5-2509-1.2B). Set to a HuggingFace identifier
    # like "opendatalab/MinerU2.5-Pro-2604-1.2B" to use Pro.
    mineru_vlm_model: str = "opendatalab/MinerU2.5-Pro-2604-1.2B"

    # Phase 54.6.211 (roadmap 3.1.6 Phase 1) — explicit choice of the
    # VLM inference backend when running MinerU 2.5-Pro. Values:
    #   "auto"         — let MinerU pick (vllm if the `mineru[vllm]`
    #                    extras are installed, transformers otherwise).
    #                    Current default for backward compatibility.
    #   "transformers" — force the HF transformers path (slower, no
    #                    vllm prerequisite; safe everywhere).
    #   "vllm"         — force the vllm-engine path (2-4× faster than
    #                    transformers on a 3090; requires `mineru[vllm]`
    #                    extras + a vllm-compatible CUDA stack).
    # Only consulted when pdf_converter_backend = "mineru-vlm-pro" or
    # when "auto" dispatch routes to VLM-Pro. Phase 54.6.212 flips
    # the default to `vllm` — the whole point of the migration is
    # to get the throughput win, and "auto" inside MinerU picks vllm
    # whenever it's installed anyway, so making this explicit is
    # clearer + surfaces a proper error if vllm fails to import.
    mineru_vlm_backend: str = "vllm"

    # Ingestion
    # Chunks per bge-m3 batch. Default 32 is safe on a 24GB GPU when the LLM is
    # also resident; raise to 64-128 for embedder-only runs, lower to 8-16 if
    # running alongside a large LLM (e.g. 32B q4) on the same GPU.
    embedding_batch_size: int = 32

    # Marker batch_multiplier (parallel_factor). Higher = more pages per Marker
    # forward pass; costs more VRAM. ~5GB peak per unit on Marker docs. Safe
    # default 2 on a 24GB GPU.
    marker_batch_multiplier: int = 2

    # SQLAlchemy connection pool. Raise when running parallel ingestion workers
    # or bulk ops (db enrich, db expand) that open many sessions concurrently.
    pg_pool_size: int = 20
    pg_max_overflow: int = 20

    # Qdrant HNSW + quantization knobs. Changes apply only on collection
    # (re)creation — existing collections keep their original params.
    qdrant_hnsw_m: int = 32
    qdrant_hnsw_ef_construct: int = 256
    qdrant_hnsw_ef: int = 128
    qdrant_scalar_quantization: bool = True

    # Bulk-op concurrency knobs
    enrich_workers: int = 8            # concurrent Crossref/OpenAlex lookups
    expand_download_workers: int = 6   # concurrent OA PDF lookups/downloads
    llm_parallel_workers: int = 4      # concurrent LLM calls (match OLLAMA_NUM_PARALLEL)
    # Phase 55.1.1 — wiki compile worker count. DELIBERATELY defaults to 1
    # rather than llm_parallel_workers, because the "parallel requests to the
    # same Ollama model speed up bulk compile" story is hardware-dependent:
    # on a 24 GB 3090 with qwen3:30b-a3b MoE (~18 GB weights) + KV cache
    # for multiple slots, VRAM headroom is tight and the MoE routing often
    # serialises on the same expert, netting out closer to 0–25 % gain than
    # the 40–60 % published numbers from dense-model / multi-GPU benches.
    # Opt in explicitly by raising this (set OLLAMA_NUM_PARALLEL to at least
    # the same value) and measure with `sciknow wiki compile` timing output.
    wiki_compile_workers: int = 1

    # Ingestion worker processes for `sciknow ingest directory`. Each worker
    # loads its own Marker (~5GB VRAM peak) + bge-m3 (~2.2GB). On a 24GB GPU
    # with an LLM actively resident (e.g. Ollama-held 32B q4 ~18GB), keep this
    # at 1. Raise to 2 only when the LLM is off-GPU (no Ollama model loaded,
    # or LLM moved to a remote host like DGX Spark).
    ingest_workers: int = 1

    # Citation-count boost factor applied to RRF scores during search.
    # Papers cited by more corpus papers rank slightly higher. Uses a log-
    # dampened multiplicative formula: score *= (1 + factor * log2(1 + count)).
    # Set to 0 to disable. Default 0.1 is a gentle nudge — retrieval signals
    # (dense + sparse + FTS) still dominate.
    citation_boost_factor: float = 0.1

    # Phase 34 — Soft RAPTOR clustering. When > 0, chunks with GMM
    # membership probability above this threshold contribute to
    # MULTIPLE cluster summaries instead of just their argmax cluster.
    # Improves recall for queries that approach a topic from different
    # angles. Set to 0 to disable (hard assignment only, Phase 12 default).
    # 0.15 is conservative — a chunk needs ≥15% probability to land in
    # a secondary cluster. Typical values: 0.10 (permissive) to 0.30
    # (restrictive). Only affects the `catalog raptor build` step.
    raptor_soft_threshold: float = 0.15

    # Phase 32.8 — Compound learning Layer 2: useful-chunk boost.
    # For each retrieved chunk, look up how often it was actually cited
    # in a finished autowrite draft (autowrite_retrievals.was_cited).
    # Same log-dampened multiplicative form as citation_boost, but
    # slightly stronger by default (0.15 vs 0.1) because useful_count
    # is a more direct signal of "this chunk helps write similar
    # sections" than passive citation popularity. Set to 0 to disable.
    useful_count_boost_factor: float = 0.15

    # `sciknow db expand` relevance filter. Cosine similarity threshold under
    # which candidate references are dropped before download. 0.55 is a sane
    # default for bge-m3 — ~80% of on-topic refs score above it on a focused
    # library, while off-topic references (statistical methods cited from a
    # climate paper, etc.) typically score 0.3-0.45.
    expand_relevance_threshold: float = 0.55

    @model_validator(mode="after")
    def _project_wins_over_env_overrides(self):
        """Phase 54.6.20 — when an explicit project is selected (via
        ``.active-project`` file or ``SCIKNOW_PROJECT``/``--project``),
        the project's ``pg_database`` and ``data_dir`` win over
        ``PG_DATABASE`` / ``DATA_DIR`` values pulled from ``.env``.

        Why: ``sciknow project use <slug>`` should actually switch the
        active project end-to-end. Without this guard, a stale
        ``PG_DATABASE`` left over in ``.env`` from the legacy
        single-tenant install silently splits state — disk writes go to
        the project's data dir but DB writes go to ``.env``'s database
        name. That's how a wiki-compile run can appear to lose work
        (the rows landed in the legacy ``sciknow`` DB, not the active
        project's DB) even though the data is intact.

        Legacy single-tenant installs (no ``.active-project`` file, no
        ``SCIKNOW_PROJECT`` set) keep working: the validator is a
        no-op, ``.env`` stays authoritative.
        """
        from sciknow.core.project import (
            get_active_project, read_active_slug_from_file,
        )

        explicit = (
            bool(os.environ.get("SCIKNOW_PROJECT", "").strip())
            or read_active_slug_from_file() is not None
        )
        if not explicit:
            # Phase 54.6.252 — always initialise the attribute so the
            # monitor helper doesn't have to guard with hasattr.
            object.__setattr__(self, "_env_overrides", [])
            return self

        active = get_active_project()
        overrides: list[str] = []
        if self.pg_database != active.pg_database:
            overrides.append(
                f"pg_database {self.pg_database!r} → {active.pg_database!r}"
            )
            object.__setattr__(self, "pg_database", active.pg_database)
        try:
            same_dir = Path(self.data_dir).resolve() == active.data_dir.resolve()
        except OSError:
            same_dir = str(self.data_dir) == str(active.data_dir)
        if not same_dir:
            overrides.append(f"data_dir {self.data_dir} → {active.data_dir}")
            object.__setattr__(self, "data_dir", active.data_dir)
        if overrides:
            logging.getLogger("sciknow.config").warning(
                "Active project %r overrides .env: %s. Drop the "
                "matching key from .env to silence this warning.",
                active.slug, "; ".join(overrides),
            )
        # Phase 54.6.252 — stash the override list on the Settings
        # instance so the monitor can surface "config drift" in the
        # dashboard. Pre-252 this was warning-only (log line once at
        # startup; users who didn't read the log missed it entirely).
        object.__setattr__(self, "_env_overrides", overrides)
        return self

    @computed_field
    @property
    def pg_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @computed_field
    @property
    def inbox_dir(self) -> Path:
        return self.data_dir / "inbox"

    @computed_field
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @computed_field
    @property
    def failed_dir(self) -> Path:
        return self.data_dir / "failed"

    @computed_field
    @property
    def mineru_output_dir(self) -> Path:
        return self.data_dir / "mineru_output"

    @computed_field
    @property
    def wiki_dir(self) -> Path:
        return self.data_dir / "wiki"

    # Phase 54.6.24 — auto-backup
    backup_retain_count: int = 7
    backup_include_code: bool = True
    # Phase 54.6.94 — age-based retention. When > 0, every `backup run`
    # also prunes any backup set older than this many days. Runs in
    # addition to `backup_retain_count` (whichever deletes more wins).
    # Set to 0 to disable age-based pruning (count-only retention).
    backup_retain_days: int = 0


def _apply_env_overlay() -> None:
    """Phase 54.6.21 — apply per-project ``.env.overlay`` BEFORE
    ``Settings()`` is instantiated.

    The overlay file is created (empty, with helpful comments) by
    ``sciknow project init`` and lives at ``<project root>/.env.overlay``.
    It's meant to layer on top of the root ``.env`` so per-project
    settings (e.g. a different ``LLM_MODEL``) can be set without
    touching the global config. Pre-Phase-54.6.21 the file was
    created, surfaced in the GUI/CLI, and bundled by archive/unarchive
    — but never actually loaded into Settings, so per-project
    overrides were silently no-ops.

    Precedence (highest first):

      1. ``os.environ`` already-set values (shell exports, --project
         CLI flag) — stay authoritative
      2. ``.env.overlay`` — per-project layer
      3. ``.env`` — global baseline (loaded by pydantic-settings)
      4. Field defaults

    Best-effort: if the overlay parse fails, we silently skip rather
    than break Settings() instantiation. Bad overlay content is
    less harmful than refusing to start.
    """
    try:
        from sciknow.core.project import get_active_project
        active = get_active_project()
        overlay = active.env_overlay_path
        if not overlay.exists():
            return
        for line in overlay.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception as exc:
        logging.getLogger("sciknow.config").warning(
            "env.overlay load skipped: %s", exc
        )


_apply_env_overlay()
settings = Settings()
