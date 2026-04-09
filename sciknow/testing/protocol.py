"""
sciknow testing protocol — layered smoke tests.

Three layers:

  L1 — Static    (seconds, no deps)              imports, prompts, signatures
  L2 — Live      (tens of seconds, PG + Qdrant)  hybrid_search, raptor scrolls
  L3 — End-to-end (minutes, PG + Qdrant + Ollama) one tiny LLM call, embedder

Each layer is a list of named test functions. A test function returns a
``TestResult`` (passed, message, duration_ms). The harness runs them, prints
a colored Rich report, and exits non-zero on any failure.

This is NOT pytest. It's a smoke-test runner that's invoked from the CLI:

    uv run sciknow test               # runs L1 only by default
    uv run sciknow test --layer L2
    uv run sciknow test --layer all

The convention is: every PR that touches synthesis pipeline, retrieval, or
ingestion should pass at least L1 before merging. L2 and L3 are run before
shipping a "Phase" feature drop or after any infrastructure change. See
docs/TESTING.md for the full protocol.

Adding a new check is one function: write it, return ``TestResult.ok(...)``
or ``TestResult.fail(...)``, then append it to the layer's list at the
bottom of this file.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Callable


# ── Result type ──────────────────────────────────────────────────────────────


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    duration_ms: int
    skipped: bool = False

    @classmethod
    def ok(cls, name: str, duration_ms: int = 0, message: str = "") -> "TestResult":
        return cls(name=name, passed=True, message=message, duration_ms=duration_ms)

    @classmethod
    def fail(cls, name: str, message: str, duration_ms: int = 0) -> "TestResult":
        return cls(name=name, passed=False, message=message, duration_ms=duration_ms)

    @classmethod
    def skip(cls, name: str, message: str) -> "TestResult":
        return cls(name=name, passed=True, message=message,
                   duration_ms=0, skipped=True)


def _run(name: str, fn: Callable[[], None]) -> TestResult:
    """Run a single test function and capture timing + exceptions."""
    t0 = time.monotonic()
    try:
        result = fn()
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        # If the test function returns a TestResult (e.g. for skip), use that
        # but stamp on the elapsed time.
        if isinstance(result, TestResult):
            result.duration_ms = elapsed_ms
            return result
        return TestResult.ok(name=name, duration_ms=elapsed_ms)
    except AssertionError as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return TestResult.fail(
            name=name, message=str(exc) or "assertion failed",
            duration_ms=elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        tb = traceback.format_exc().splitlines()
        # Keep the message short — the last 2 lines usually identify the cause.
        short = " | ".join(line.strip() for line in tb[-3:])
        return TestResult.fail(
            name=name, message=f"{type(exc).__name__}: {exc}  ·  {short}",
            duration_ms=elapsed_ms,
        )


# ════════════════════════════════════════════════════════════════════════════
# L1 — Static checks (no services required)
# ════════════════════════════════════════════════════════════════════════════


def l1_all_modules_import() -> None:
    """Every package module imports cleanly. The fastest way to catch syntax
    errors, broken imports, and circular references introduced by a recent
    change."""
    import sciknow.config            # noqa: F401
    import sciknow.core.book_ops     # noqa: F401
    import sciknow.core.wiki_ops     # noqa: F401
    import sciknow.ingestion.chunker  # noqa: F401
    import sciknow.ingestion.embedder  # noqa: F401
    import sciknow.ingestion.metadata  # noqa: F401
    import sciknow.ingestion.pdf_converter  # noqa: F401
    import sciknow.ingestion.pipeline  # noqa: F401
    import sciknow.ingestion.raptor  # noqa: F401
    import sciknow.ingestion.topic_cluster  # noqa: F401
    import sciknow.rag.llm           # noqa: F401
    import sciknow.rag.prompts       # noqa: F401
    import sciknow.rag.wiki_prompts  # noqa: F401
    import sciknow.retrieval.context_builder  # noqa: F401
    import sciknow.retrieval.hybrid_search    # noqa: F401
    import sciknow.retrieval.reranker         # noqa: F401
    import sciknow.storage.db        # noqa: F401
    import sciknow.storage.models    # noqa: F401
    import sciknow.storage.qdrant    # noqa: F401
    import sciknow.cli.main          # noqa: F401
    import sciknow.cli.book          # noqa: F401
    import sciknow.cli.catalog       # noqa: F401
    import sciknow.cli.db            # noqa: F401
    import sciknow.cli.ingest        # noqa: F401
    import sciknow.cli.search        # noqa: F401
    import sciknow.cli.wiki          # noqa: F401
    import sciknow.cli.draft         # noqa: F401
    import sciknow.web.app           # noqa: F401


def l1_prompts_phase7_hedging() -> None:
    """Phase 7 — hedging fidelity rule + OVERSTATED verdict + scoring dim."""
    from sciknow.rag import prompts
    sys_w, _ = prompts.write_section_v2("results", "x", [], book_plan=None,
                                         prior_summaries=None)
    assert "Hedging fidelity" in sys_w, "writer prompt missing hedging-fidelity rule"
    assert "may, might, could" in sys_w, "writer prompt missing BioScope cue list"
    sys_v, _ = prompts.verify_claims("draft", [])
    assert "OVERSTATED" in sys_v, "verifier missing OVERSTATED verdict"
    assert "hedging_fidelity_score" in sys_v, "verifier missing hedging_fidelity_score field"
    sys_s, _ = prompts.score_draft("results", "x", "draft", [])
    assert "hedging_fidelity" in sys_s, "scorer missing hedging_fidelity dimension"


def l1_prompts_phase8_entity_bridge() -> None:
    """Phase 8 — entity-bridge rule (Centering Theory)."""
    from sciknow.rag import prompts
    sys_w, _ = prompts.write_section_v2("results", "x", [], book_plan=None, prior_summaries=None)
    assert "entity bridge" in sys_w.lower() or "previous paragraph" in sys_w.lower(), \
        "writer prompt missing Centering entity-bridge rule"
    assert "cold-starts" in sys_w.lower() or "cold start" in sys_w.lower(), \
        "writer prompt missing 'no cold-starts' rule"


def l1_prompts_phase9_pdtb_relations() -> None:
    """Phase 9 — PDTB-lite discourse_relation enum in tree plan."""
    from sciknow.rag import prompts
    sys_t, _ = prompts.tree_plan("results", "x", [], book_plan=None, prior_summaries=None)
    assert "discourse_relation" in sys_t, "tree_plan missing discourse_relation field"
    # The 10-label PDTB-lite vocabulary should be enumerated.
    for label in ("background", "elaboration", "contrast", "concession",
                  "cause", "comparison", "exemplification", "qualification",
                  "synthesis", "evidence"):
        assert label in sys_t, f"tree_plan missing PDTB label: {label}"

    # write_section_v2 surfaces relations when paragraph_plan is provided.
    sys_w, _ = prompts.write_section_v2(
        "results", "x", [],
        paragraph_plan=[
            {"point": "p1", "discourse_relation": "background", "sources": ["[1]"]},
            {"point": "p2", "discourse_relation": "contrast", "sources": ["[2]"]},
        ],
    )
    assert "background" in sys_w.lower(), "writer prompt didn't surface background label"
    assert "contrast" in sys_w.lower(), "writer prompt didn't surface contrast label"


def l1_prompts_phase10_step_back() -> None:
    """Phase 10 — step-back retrieval prompt."""
    from sciknow.rag import prompts
    sys_p, _ = prompts.step_back("ocean heat content trends in the North Atlantic")
    assert "step-back" in sys_p.lower(), "step_back prompt missing self-reference"
    assert "abstract" in sys_p.lower() or "higher-level" in sys_p.lower(), \
        "step_back prompt should ask for an abstract reformulation"


def l1_prompts_phase11_cove() -> None:
    """Phase 11 — Chain-of-Verification questioner + answerer (decoupled)."""
    from sciknow.rag import prompts
    sys_q, _ = prompts.cove_questions("draft text")
    # The questioner sees ONLY the draft.
    assert "ONLY the draft" in sys_q or "see ONLY" in sys_q.lower(), \
        "cove_questions doesn't enforce draft-only context"
    assert "falsifiable" in sys_q.lower() or "falsifi" in sys_q.lower(), \
        "cove_questions doesn't ask for falsifiable claims"

    sys_a, _ = prompts.cove_answer("question", [])
    # The answerer sees ONLY the sources.
    assert "have NOT seen the draft" in sys_a or "NOT seen the draft" in sys_a, \
        "cove_answer doesn't enforce no-draft-access"
    assert "NOT_IN_SOURCES" in sys_a, "cove_answer missing NOT_IN_SOURCES verdict"
    assert "DIFFERENT_SCOPE" in sys_a, "cove_answer missing DIFFERENT_SCOPE verdict"


def l1_prompts_phase12_raptor() -> None:
    """Phase 12 — RAPTOR cluster summary template."""
    from sciknow.rag import prompts
    sys_r, _ = prompts.raptor_summary("chunks here", n=3)
    assert "synthesi" in sys_r.lower(), "raptor_summary doesn't ask for synthesis"
    assert "[N]" in sys_r, "raptor_summary should explicitly forbid [N] markers"
    # Hedging fidelity should be enforced at the summary level too (Phase 7
    # consistency).
    assert "epistemic" in sys_r.lower() or "hedge" in sys_r.lower(), \
        "raptor_summary doesn't enforce hedging fidelity"


def l1_book_ops_signatures() -> None:
    """All Phase 7-13 kwargs are wired through book_ops.autowrite_section_stream
    and write_section_stream."""
    import inspect
    from sciknow.core import book_ops

    aw_sig = inspect.signature(book_ops.autowrite_section_stream)
    for kw in ("use_plan", "use_step_back", "use_cove", "cove_threshold",
               "auto_expand", "max_iter", "target_score"):
        assert kw in aw_sig.parameters, f"autowrite_section_stream missing {kw}"

    ws_sig = inspect.signature(book_ops.write_section_stream)
    assert "use_step_back" in ws_sig.parameters, "write_section_stream missing use_step_back"

    save_sig = inspect.signature(book_ops._save_draft)
    assert "custom_metadata" in save_sig.parameters, \
        "_save_draft missing custom_metadata kwarg (Phase 13 score persistence)"

    # Helpers added in Phases 10/11/12/13
    assert hasattr(book_ops, "_retrieve_with_step_back")
    assert hasattr(book_ops, "_generate_step_back_query")
    assert hasattr(book_ops, "_cove_verify")


def l1_raptor_module_surface() -> None:
    """Phase 12 — raptor module exposes the public API."""
    from sciknow.ingestion import raptor
    assert hasattr(raptor, "build_raptor_tree")
    assert hasattr(raptor, "_cluster_with_gmm_bic")
    assert hasattr(raptor, "_backfill_leaf_node_level")
    assert hasattr(raptor, "_summarise_cluster")

    from sciknow.ingestion import embedder
    assert hasattr(embedder, "embed_summary_text"), \
        "embedder.embed_summary_text missing — RAPTOR build will crash"

    from sciknow.storage import qdrant
    assert hasattr(qdrant, "ensure_node_level_index"), \
        "qdrant.ensure_node_level_index missing — RAPTOR build will crash"


def l1_raptor_clustering_works() -> None:
    """RAPTOR's UMAP+GMM clustering correctly separates two well-separated
    synthetic clusters. Doesn't need any service."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="umap")

    import numpy as np
    from sciknow.ingestion import raptor

    np.random.seed(42)
    c1 = np.random.randn(20, 1024) + 5.0
    c2 = np.random.randn(20, 1024) - 5.0
    X = np.vstack([c1, c2]).astype(np.float32)
    labels, k, proba = raptor._cluster_with_gmm_bic(X, max_k=10)
    assert labels.shape == (40,), f"unexpected labels shape: {labels.shape}"
    assert k >= 2, f"GMM found only k={k} on 2-cluster data"
    half_a_majority = int(np.bincount(labels[:20]).argmax())
    half_b_majority = int(np.bincount(labels[20:]).argmax())
    assert half_a_majority != half_b_majority, \
        f"clustering failed to separate halves (both → cluster {half_a_majority})"


def l1_cli_commands_register() -> None:
    """All CLI commands shipped in Phases 7-13 are registered on their Typer
    apps."""
    from sciknow.cli import catalog as cli_catalog
    from sciknow.cli import book as cli_book

    # catalog raptor sub-Typer
    raptor_cmds = {c.name for c in cli_catalog.raptor_app.registered_commands}
    assert "build" in raptor_cmds, f"catalog raptor build missing — found: {sorted(raptor_cmds)}"
    assert "stats" in raptor_cmds, f"catalog raptor stats missing — found: {sorted(raptor_cmds)}"

    # book draft sub-Typer
    draft_cmds = {c.name for c in cli_book.draft_app.registered_commands}
    assert "scores" in draft_cmds, f"book draft scores missing — found: {sorted(draft_cmds)}"
    assert "compare" in draft_cmds, f"book draft compare missing — found: {sorted(draft_cmds)}"

    # autowrite-bench on top-level book app
    book_cmds = {c.name for c in cli_book.app.registered_commands}
    assert "autowrite-bench" in book_cmds, \
        f"autowrite-bench not on book app — found: {sorted(book_cmds)}"

    # main app composes everything
    from sciknow.cli import main as cli_main
    main_groups = {tg.name for tg in cli_main.app.registered_groups}
    for group in ("catalog", "db", "ingest", "search", "ask", "book",
                  "draft", "wiki"):
        assert group in main_groups, f"main app missing subapp: {group}"


def l1_history_shape_roundtrips() -> None:
    """The Phase 13 score history dict serialises to JSON and round-trips
    losslessly. This is what custom_metadata expects."""
    import json

    history = [
        {
            "iteration": 1,
            "scores": {
                "groundedness": 0.78, "completeness": 0.65, "coherence": 0.82,
                "citation_accuracy": 0.85, "hedging_fidelity": 0.72,
                "overall": 0.76, "weakest_dimension": "completeness",
            },
            "verification": {
                "groundedness_score": 0.78, "hedging_fidelity_score": 0.72,
                "n_supported": 8, "n_extrapolated": 2, "n_overstated": 3,
                "n_misrepresented": 0,
            },
            "cove": {
                "ran": True, "score": 0.71, "n_high_severity": 1,
                "n_medium_severity": 2, "questions_asked": 7,
            },
            "revision_verdict": "KEEP",
            "post_revision_overall": 0.83,
        }
    ]
    payload = {
        "score_history": history,
        "feature_versions": {
            "phase7_hedging_fidelity": True,
            "phase11_chain_of_verification": True,
            "phase12_raptor_retrieval": True,
        },
        "final_overall": 0.76,
        "max_iter": 3,
        "target_score": 0.85,
    }
    s = json.dumps(payload)
    parsed = json.loads(s)
    assert parsed["score_history"][0]["cove"]["ran"] is True
    assert parsed["score_history"][0]["scores"]["hedging_fidelity"] == 0.72


def l1_format_score_cell() -> None:
    """draft scores helper renders all buckets correctly."""
    from sciknow.cli.book import _format_score_cell
    assert "[green]" in _format_score_cell(0.95)
    assert "[yellow]" in _format_score_cell(0.75)
    assert "[red]" in _format_score_cell(0.50)
    assert "[dim]" in _format_score_cell(None)


def l1_web_template_has_overstated() -> None:
    """Web reader template includes the OVERSTATED verdict + CSS class."""
    import sciknow.web.app as web
    # The TEMPLATE module-level constant is the f-string render input.
    assert hasattr(web, "TEMPLATE"), "web.app.TEMPLATE missing"
    t = web.TEMPLATE
    assert "verified-overstated" in t, "OVERSTATED CSS class missing from template"
    assert "OVERSTATED" in t, "OVERSTATED string missing from template JS"
    assert "cove_verification" in t, "cove_verification SSE handler missing"
    assert "hedging_fidelity" in t, "hedging_fidelity dimension missing from web score bars"


def l1_web_template_phase14_features() -> None:
    """Phase 14 — modernized UI + new modals + score history viewer.

    Verifies that the web v2 redesign is in place: new design tokens
    (--accent: #4f46e5 indigo), modal infrastructure, the four new
    toolbar buttons, and the score history panel.
    """
    import sciknow.web.app as web
    t = web.TEMPLATE

    # New design system tokens
    assert "--font-sans:" in t, "v2 font system tokens missing from CSS"
    assert "--sp-1:" in t, "v2 spacing scale missing from CSS"
    assert "--shadow-md:" in t, "v2 shadow tokens missing from CSS"
    # Modal infrastructure
    assert ".modal-overlay" in t, "modal CSS class missing"
    assert "openModal(" in t, "openModal JS function missing"
    assert "closeModal(" in t, "closeModal JS function missing"
    # Toolbar v2: four new buttons
    assert "openWikiModal()" in t, "Wiki Query toolbar button missing"
    assert "openAskModal()" in t, "Ask Corpus toolbar button missing"
    assert "openCatalogModal()" in t, "Browse Papers toolbar button missing"
    assert "showScoresPanel()" in t, "Score History toolbar button missing"
    # Score history viewer (Phase 13 GUI integration)
    assert 'id="scores-panel"' in t, "scores-panel container missing"
    assert "scores-table" in t, "scores-table CSS class missing"
    assert "/api/draft/" in t, "draft scores API call missing from JS"
    # New modals exist
    assert 'id="wiki-modal"' in t, "wiki-modal element missing"
    assert 'id="ask-modal"' in t, "ask-modal element missing"
    assert 'id="catalog-modal"' in t, "catalog-modal element missing"
    # Stat-tile dashboard upgrade
    assert "stat-tile" in t, "stat-tile dashboard tile class missing"
    assert "raptor-bar" in t, "raptor-bar dashboard panel missing"


def l1_web_phase14_endpoints_registered() -> None:
    """Phase 14 — new FastAPI routes are registered on the app."""
    import sciknow.web.app as web
    routes = {r.path for r in web.app.routes if hasattr(r, "path")}
    expected = {
        "/api/draft/{draft_id}/scores",
        "/api/wiki/query",
        "/api/ask",
        "/api/catalog",
        "/api/stats",
        # Phase 14.3
        "/api/book",
        "/api/book/plan/generate",
    }
    missing = expected - routes
    assert not missing, f"Phase 14 routes missing from app: {missing}"


def l1_web_phase15_wiki_browse_and_stats() -> None:
    """Phase 15 — wiki browsing UI + live streaming stats helper.

    Verifies the new /api/wiki/pages and /api/wiki/page/{slug} routes are
    mounted, the wiki modal has Query + Browse tabs with all the JS
    handlers (switchWikiTab, loadWikiPages, openWikiPage), and the
    createStreamStats helper exists with the cursor helper too.
    """
    import sciknow.web.app as web

    # Routes
    routes = {r.path for r in web.app.routes if hasattr(r, "path")}
    assert "/api/wiki/pages" in routes, "/api/wiki/pages missing"
    assert "/api/wiki/page/{slug}" in routes, "/api/wiki/page/{slug} missing"

    t = web.TEMPLATE
    # Tab UI
    assert 'class="tabs"' in t, "modal tabs CSS class missing"
    assert "switchWikiTab(" in t, "switchWikiTab JS missing"
    assert 'data-tab="wiki-query"' in t, "wiki-query tab missing"
    assert 'data-tab="wiki-browse"' in t, "wiki-browse tab missing"
    # Browse view
    assert "loadWikiPages(" in t, "loadWikiPages JS missing"
    assert "openWikiPage(" in t, "openWikiPage JS missing"
    assert "closeWikiPageDetail(" in t, "closeWikiPageDetail JS missing"
    assert "wiki-page-detail" in t, "wiki-page-detail container missing"
    # Stats helper + cursor
    assert "function createStreamStats(" in t, "createStreamStats helper missing"
    assert "function setStreamCursor(" in t, "setStreamCursor helper missing"
    assert "stream-cursor" in t, "stream-cursor CSS class missing"
    # Stats footers wired in
    assert 'id="main-stream-stats"' in t, "main-stream-stats footer missing"
    assert 'id="wiki-stream-stats"' in t, "wiki-stream-stats footer missing"
    assert 'id="ask-stream-stats"' in t, "ask-stream-stats footer missing"
    assert 'id="plan-stream-stats"' in t, "plan-stream-stats footer missing"


def l1_autowrite_incremental_save() -> None:
    """Phase 15.1 — autowrite must save incrementally so cancelling
    mid-generation doesn't lose all the work.

    Verifies that autowrite_section_stream:
      - Calls _save_draft BEFORE the iteration loop (initial checkpoint)
      - Calls _update_draft_content inside the iteration loop (per-iter
        checkpoint after KEEP/DISCARD verdicts)
      - Yields 'checkpoint' events so the GUI can show feedback
      - The final save is an UPDATE not a new INSERT (no second
        _save_draft call after the iteration loop)

    And that write_section_stream:
      - Calls _save_draft right after token collection (before verify)
      - Calls _update_draft_content after verification to add the
        review feedback (so verify-failure doesn't lose the draft)

    Plus _update_draft_content helper exists.
    """
    import inspect
    from sciknow.core import book_ops

    # Helper exists
    assert hasattr(book_ops, "_update_draft_content"), \
        "_update_draft_content helper missing"

    # autowrite has incremental save
    aw_src = inspect.getsource(book_ops.autowrite_section_stream)
    assert "_save_draft" in aw_src, "autowrite never saves anything"
    assert "_update_draft_content" in aw_src, \
        "autowrite doesn't update existing draft (only one save = no incremental)"
    assert '"checkpoint"' in aw_src or "'checkpoint'" in aw_src, \
        "autowrite missing checkpoint events"
    # Make sure there's only ONE _save_draft call (the initial checkpoint),
    # not two (which would mean the old end-of-loop save still exists).
    assert aw_src.count("_save_draft(") == 1, (
        f"autowrite has {aw_src.count('_save_draft(')} _save_draft calls — "
        f"should be exactly 1 (initial checkpoint). Multiple inserts mean "
        f"the cancel-safety fix wasn't applied cleanly."
    )

    # write_section_stream has incremental save too
    ws_src = inspect.getsource(book_ops.write_section_stream)
    assert "_save_draft" in ws_src, "write_section_stream never saves"
    assert "_update_draft_content" in ws_src, \
        "write_section_stream doesn't update after verification — verify failure loses content"


def l1_writer_uses_flagship_model() -> None:
    """Phase 14.6 — assert the writing/scoring/verification path NEVER
    accidentally uses settings.llm_fast_model.

    The fast model is only allowed in:
      - step-back retrieval (utility)
      - RAPTOR cluster summaries (batch op)
      - wiki compile / metadata extraction / query expansion (utility)
      - L3 smoke test (intentional)

    All "writing-quality" call sites in book_ops.py — the actual writer,
    the scorer, the verifier, CoVe, review, revise, gaps, argue, plan
    generation — must use `model=model` (which defaults to settings.llm_model
    via rag/llm.py:48), NOT `settings.llm_fast_model`.
    """
    import inspect
    from sciknow.core import book_ops

    # The functions that drive writing/scoring quality. None of them
    # should reference llm_fast_model in their bodies.
    flagship_funcs = [
        book_ops.autowrite_section_stream,
        book_ops.write_section_stream,
        book_ops.review_draft_stream,
        book_ops.revise_draft_stream,
        book_ops._score_draft_inner,
        book_ops._verify_draft_inner,
        book_ops._cove_verify,
        book_ops.run_gaps_stream,
        book_ops.run_argue_stream,
    ]

    # Match `model=...llm_fast_model` (with optional `settings.` prefix and
    # optional `or` fallback). This is the dangerous pattern. Bare references
    # to `llm_fast_model` inside informational dicts (like the model_info
    # event payload) are allowed — they don't affect which model writes.
    import re as _re
    dangerous = _re.compile(r"model\s*=\s*[^,\)]*llm_fast_model")

    for fn in flagship_funcs:
        try:
            src = inspect.getsource(fn)
        except (TypeError, OSError):
            continue  # not introspectable, skip
        match = dangerous.search(src)
        assert not match, (
            f"{fn.__name__} passes llm_fast_model as a model= argument — "
            f"flagship-only call sites must not use the fast model. "
            f"Match: {match.group(0)!r}"
        )

    # And the model_info events should be emitted before writing.
    aw_src = inspect.getsource(book_ops.autowrite_section_stream)
    assert '"model_info"' in aw_src or "'model_info'" in aw_src, \
        "autowrite_section_stream missing model_info event for transparency"


def l1_web_phase14_4_book_sections() -> None:
    """Phase 14.4 — book-style sections, dashboard chapter editing, autosave UX.

    Verifies:
      - _DEFAULT_BOOK_SECTIONS contains the canonical book-style fallback
        and does NOT contain paper-style sections (introduction/methods/etc).
      - _chapter_sections + _normalize_section helpers exist.
      - The dashboard heatmap renders chapter titles as .ch-label.clickable
        with an openChapterModal() handler.
      - Per-chapter section template is propagated through the data pipeline:
        _get_book_data SQL selects bc.sections, chapters_json carries
        sections_template, JS reads ch.sections_template.
      - The hardcoded paper-style section list ['introduction','methods',
        'results','discussion','conclusion'] is NO LONGER in the dashboard
        section_types fallback (it was hardcoded before 14.4).
      - The autosave indicator has the new always-visible pill markup.
    """
    import sciknow.web.app as web

    # Helpers exist
    assert hasattr(web, "_DEFAULT_BOOK_SECTIONS"), "_DEFAULT_BOOK_SECTIONS missing"
    assert "overview" in web._DEFAULT_BOOK_SECTIONS, "book defaults missing 'overview'"
    assert "key_evidence" in web._DEFAULT_BOOK_SECTIONS, "book defaults missing 'key_evidence'"
    # Critically: the book defaults must NOT include paper-style sections.
    paper_sections = {"methods", "results", "discussion"}
    assert not (paper_sections & set(web._DEFAULT_BOOK_SECTIONS)), \
        "_DEFAULT_BOOK_SECTIONS leaked paper-style entries"
    assert hasattr(web, "_chapter_sections"), "_chapter_sections helper missing"
    assert hasattr(web, "_normalize_section"), "_normalize_section helper missing"

    # SQL loads bc.sections
    import inspect
    src = inspect.getsource(web._get_book_data)
    assert "bc.sections" in src, "_get_book_data SQL missing bc.sections column"

    # Template wiring
    t = web.TEMPLATE
    # Heatmap chapter title is clickable
    assert "ch-label clickable" in t, \
        "dashboard chapter title not marked clickable"
    assert "openChapterModal(&#39;' + row.id" in t or \
           'openChapterModal(' in t and "row.id" in t, \
        "dashboard chapter title doesn't open the chapter scope modal"
    # Per-chapter section template flows to the empty-state JS
    assert "sections_template" in t, "sections_template not propagated to JS"
    assert "ch.sections_template" in t, "JS doesn't read ch.sections_template"
    # Off-template heatmap cells styled
    assert ".hm-cell.off-template" in t, "off-template cell CSS missing"
    # The hardcoded paper-style mix should be gone from showChapterEmptyState
    assert "['overview', 'introduction', 'key_evidence', 'methods'" not in t, \
        "hardcoded paper-style sections still present in showChapterEmptyState"
    # Autosave indicator pill
    assert 'id="autosave-text"' in t, "autosave-text element missing"
    assert "setAutosaveState(" in t, "setAutosaveState helper missing"
    assert "Autosave on" in t, "autosave default label missing from template"
    assert ".editor-toolbar .autosave .dot" in t, "autosave dot CSS missing"


def l1_web_phase14_3_book_plan_editor() -> None:
    """Phase 14.3 — book plan + chapter scope editors are wired in.

    Verifies the toolbar button, the two new modals, the JS handlers,
    and the empty-state chapter scope card.
    """
    import sciknow.web.app as web
    t = web.TEMPLATE
    # Toolbar
    assert "openPlanModal()" in t, "Plan toolbar button missing"
    # Modal HTML
    assert 'id="plan-modal"' in t, "plan-modal element missing"
    assert 'id="chapter-modal"' in t, "chapter-modal element missing"
    assert 'id="plan-text-input"' in t, "plan textarea missing"
    assert 'id="ch-desc-input"' in t, "chapter description textarea missing"
    assert 'id="ch-tq-input"' in t, "chapter topic_query input missing"
    # JS handlers
    assert "function savePlan" in t, "savePlan() JS handler missing"
    assert "function regeneratePlan" in t, "regeneratePlan() JS handler missing"
    assert "function openChapterModal" in t, "openChapterModal() JS handler missing"
    assert "function saveChapterInfo" in t, "saveChapterInfo() JS handler missing"
    # Empty-state chapter scope card
    assert "ch-scope" in t, "ch-scope CSS class missing from empty state"
    assert "Edit chapter scope" in t, "Edit chapter scope button missing"


def l1_web_rendered_js_is_valid() -> "TestResult":
    """The rendered web template's <script> block must be syntactically
    valid JavaScript. This is the regression test for the latent
    `\\'` -> `'` Python escape bug that broke every onclick handler in
    the web reader for at least three commits — fixed by Phase 14.1.

    Strategy: render the template with placeholder values, extract the
    script block, write it to a temp file, and run `node --check` on it.
    Skips gracefully if Node isn't installed (rare on dev machines but
    possible).
    """
    import shutil, subprocess, tempfile, re as _re
    from pathlib import Path

    node_bin = shutil.which("node")
    if not node_bin:
        return TestResult.skip(
            name="l1_web_rendered_js_is_valid",
            message="node not installed — skipping JS syntax check",
        )

    from sciknow.web.app import TEMPLATE
    rendered = TEMPLATE.format(
        book_title="Test Book", search_q="", search_results_html="",
        sidebar_html="", gaps_count=0,
        active_id="abc12345-6789-0000-0000-000000000000",
        active_title="Test", active_version=1, active_words=100,
        active_chapter_id="ch1", active_section_type="intro",
        chapters_json="[]",
        content_html="<p>test</p>", sources_html="",
        review_html="", comments_html="",
    )
    m = _re.search(r"<script>(.*?)</script>", rendered, _re.DOTALL)
    if not m:
        raise AssertionError("no <script> block in rendered template")

    js = m.group(1)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8",
    ) as f:
        f.write(js)
        path = Path(f.name)

    try:
        result = subprocess.run(
            [node_bin, "--check", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            # Truncate node's error so the harness output stays readable.
            stderr_short = (result.stderr or "").strip().splitlines()[0:2]
            raise AssertionError(
                f"node --check failed: {' | '.join(stderr_short)}"
            )
    finally:
        try:
            path.unlink()
        except Exception:
            pass

    return TestResult.ok(
        name="l1_web_rendered_js_is_valid",
        message=f"node --check passed on {len(js):,} chars of rendered JS",
    )


def l1_research_doc_up_to_date() -> None:
    """docs/RESEARCH.md mentions all shipped phases."""
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "docs" / "RESEARCH.md"
    text = p.read_text()
    for phase_marker in (
        "## 13. Hedging Fidelity",
        "## 14. Local Coherence via Centering",
        "## 15. PDTB-Lite Discourse Relations",
        "## 16. Step-Back Retrieval",
        "## 18. Chain-of-Verification",
        "## 19. RAPTOR",
        "## 20. Measurement & Observability",
    ):
        assert phase_marker in text, f"docs/RESEARCH.md missing section: {phase_marker!r}"
    # Implementation timeline should mention all phases up to 13.
    for phase in ("Phase 7:", "Phase 8:", "Phase 9:", "Phase 10:",
                  "Phase 11:", "Phase 12:", "Phase 13:"):
        assert phase in text, f"docs/RESEARCH.md timeline missing {phase}"


# ════════════════════════════════════════════════════════════════════════════
# L2 — Live integration (PG + Qdrant required)
# ════════════════════════════════════════════════════════════════════════════


def l2_postgres_reachable() -> None:
    from sciknow.storage.db import check_connection
    assert check_connection(), "PostgreSQL is not reachable"


def l2_qdrant_reachable() -> None:
    from sciknow.storage.qdrant import check_connection
    assert check_connection(), "Qdrant is not reachable"


def l2_papers_collection_exists() -> None:
    from sciknow.storage.qdrant import PAPERS_COLLECTION, get_client
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    assert PAPERS_COLLECTION in existing, \
        f"Qdrant collection {PAPERS_COLLECTION!r} missing — run `sciknow db init`"


def l2_db_stats_query() -> None:
    """Confirm the documents/paper_metadata tables exist and are queryable."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        n_docs = session.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        n_meta = session.execute(text("SELECT COUNT(*) FROM paper_metadata")).scalar()
        n_chunks = session.execute(text("SELECT COUNT(*) FROM chunks")).scalar()
    assert n_docs is not None
    assert n_meta is not None
    assert n_chunks is not None
    return TestResult.ok(
        name="l2_db_stats_query",
        message=f"docs={n_docs}  meta={n_meta}  chunks={n_chunks}",
    )


def l2_ensure_node_level_index_idempotent() -> None:
    """Phase 12 — adding the node_level payload index twice should not fail."""
    from sciknow.storage.qdrant import ensure_node_level_index
    assert ensure_node_level_index() is True
    assert ensure_node_level_index() is True  # second call must be idempotent


def l2_qdrant_papers_count() -> None:
    """Read the points-count from the papers collection. Doesn't error
    on an empty collection."""
    from sciknow.storage.qdrant import PAPERS_COLLECTION, get_client
    client = get_client()
    info = client.count(collection_name=PAPERS_COLLECTION, exact=False)
    n = info.count if hasattr(info, "count") else int(info)
    return TestResult.ok(
        name="l2_qdrant_papers_count",
        message=f"~{n} points in {PAPERS_COLLECTION}",
    )


def l2_hybrid_search_smoke() -> None:
    """Phase 1 retrieval still works end-to-end against a small candidate_k.
    Skipped automatically if the corpus is empty (no chunks)."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client
    from sciknow.retrieval import hybrid_search

    with get_session() as session:
        n_chunks = session.execute(text("SELECT COUNT(*) FROM chunks")).scalar() or 0
    if n_chunks == 0:
        return TestResult.skip(
            name="l2_hybrid_search_smoke",
            message="no chunks in DB — skipping retrieval smoke",
        )

    qdrant = get_client()
    with get_session() as session:
        candidates = hybrid_search.search(
            query="climate", qdrant_client=qdrant, session=session,
            candidate_k=5,
        )
    # The query is generic; if there are any chunks at all, hybrid search
    # should return at least one candidate (even if reranking would prune it).
    assert isinstance(candidates, list), "hybrid_search.search must return a list"
    return TestResult.ok(
        name="l2_hybrid_search_smoke",
        message=f"got {len(candidates)} candidate(s) for query='climate'",
    )


# ════════════════════════════════════════════════════════════════════════════
# L3 — End-to-end (PG + Qdrant + Ollama)
# ════════════════════════════════════════════════════════════════════════════


def _ollama_reachable() -> bool:
    """Cheap check that Ollama answers /api/tags."""
    try:
        import urllib.request
        from sciknow.config import settings
        host = settings.ollama_host.rstrip("/")
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def l3_ollama_reachable() -> None:
    assert _ollama_reachable(), "Ollama not reachable on settings.ollama_host"


def l3_llm_complete_smoke() -> None:
    """One trivial llm.complete call. Uses the fast model and a tight context
    so this returns quickly even on a busy GPU."""
    from sciknow.config import settings
    from sciknow.rag.llm import complete
    out = complete(
        "You are a brief assistant.",
        "Reply with the single word: ok",
        model=settings.llm_fast_model,
        temperature=0.0,
        num_ctx=512,
    )
    assert isinstance(out, str) and len(out) > 0, f"empty LLM response: {out!r}"
    return TestResult.ok(
        name="l3_llm_complete_smoke",
        message=f"got {len(out)} chars from {settings.llm_fast_model}",
    )


def l3_embedder_loads() -> None:
    """The bge-m3 model loads and embeds a single text without crashing."""
    from sciknow.retrieval.hybrid_search import _embed_query, release_embed_model
    try:
        dense, sparse = _embed_query("smoke test")
        assert isinstance(dense, list) and len(dense) > 100
        assert sparse is not None
    finally:
        release_embed_model()
    return TestResult.ok(
        name="l3_embedder_loads",
        message=f"bge-m3 loaded, dense_dim={len(dense)}",
    )


# ════════════════════════════════════════════════════════════════════════════
# Layer registry — append new tests here.
# ════════════════════════════════════════════════════════════════════════════


L1_TESTS: list[Callable] = [
    l1_all_modules_import,
    l1_prompts_phase7_hedging,
    l1_prompts_phase8_entity_bridge,
    l1_prompts_phase9_pdtb_relations,
    l1_prompts_phase10_step_back,
    l1_prompts_phase11_cove,
    l1_prompts_phase12_raptor,
    l1_book_ops_signatures,
    l1_raptor_module_surface,
    l1_raptor_clustering_works,
    l1_cli_commands_register,
    l1_history_shape_roundtrips,
    l1_format_score_cell,
    l1_web_template_has_overstated,
    l1_web_template_phase14_features,
    l1_web_phase14_endpoints_registered,
    l1_web_phase14_3_book_plan_editor,
    l1_web_phase14_4_book_sections,
    l1_writer_uses_flagship_model,
    l1_web_phase15_wiki_browse_and_stats,
    l1_autowrite_incremental_save,
    l1_web_rendered_js_is_valid,
    l1_research_doc_up_to_date,
]

L2_TESTS: list[Callable] = [
    l2_postgres_reachable,
    l2_qdrant_reachable,
    l2_papers_collection_exists,
    l2_db_stats_query,
    l2_ensure_node_level_index_idempotent,
    l2_qdrant_papers_count,
    l2_hybrid_search_smoke,
]

L3_TESTS: list[Callable] = [
    l3_ollama_reachable,
    l3_llm_complete_smoke,
    l3_embedder_loads,
]

LAYERS: dict[str, list[Callable]] = {
    "L1": L1_TESTS,
    "L2": L2_TESTS,
    "L3": L3_TESTS,
}


# ── Runner ──────────────────────────────────────────────────────────────────


def run_layer(
    layer_name: str,
    *,
    fail_fast: bool = False,
) -> tuple[list[TestResult], int]:
    """Run all tests in a layer. Returns (results, failure_count).

    L2 and L3 are skipped wholesale if their service prerequisites aren't
    available, with a single skip result reported in place of the layer.
    """
    if layer_name == "L2":
        # Pre-check: don't try L2 if PG or Qdrant are down — return one skip.
        from sciknow.storage.db import check_connection as pg_ok
        from sciknow.storage.qdrant import check_connection as qd_ok
        if not pg_ok():
            return ([TestResult.skip("L2", "skipped — PostgreSQL not reachable")], 0)
        if not qd_ok():
            return ([TestResult.skip("L2", "skipped — Qdrant not reachable")], 0)
    if layer_name == "L3":
        if not _ollama_reachable():
            return ([TestResult.skip("L3", "skipped — Ollama not reachable")], 0)

    tests = LAYERS.get(layer_name, [])
    results: list[TestResult] = []
    n_failed = 0
    for fn in tests:
        result = _run(fn.__name__, fn)
        results.append(result)
        if not result.passed:
            n_failed += 1
            if fail_fast:
                break
    return results, n_failed


def render_report(layer_name: str, results: list[TestResult], n_failed: int) -> None:
    """Print a Rich table for one layer's results."""
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    n_total = len(results)
    n_pass = n_total - n_failed
    n_skipped = sum(1 for r in results if r.skipped)

    if n_failed == 0 and n_skipped == 0:
        title = f"[bold green]{layer_name}[/bold green]  [{n_pass}/{n_total} ✓]"
    elif n_failed == 0:
        title = f"[bold yellow]{layer_name}[/bold yellow]  [{n_pass - n_skipped}/{n_total} ✓ + {n_skipped} skipped]"
    else:
        title = f"[bold red]{layer_name}[/bold red]  [{n_pass}/{n_total} ✓  ·  {n_failed} failed]"

    table = Table(title=title, box=box.SIMPLE_HEAD, expand=True)
    table.add_column("", width=2)
    table.add_column("Test", style="bold")
    table.add_column("Time", justify="right", width=8)
    table.add_column("Note", style="dim", overflow="fold")

    for r in results:
        if r.skipped:
            mark = "[yellow]~[/yellow]"
        elif r.passed:
            mark = "[green]✓[/green]"
        else:
            mark = "[red]✗[/red]"
        time_str = f"{r.duration_ms}ms" if r.duration_ms else "—"
        note = r.message or ""
        table.add_row(mark, r.name, time_str, note[:120])

    console.print(table)


def run_all(layers: list[str], fail_fast: bool = False) -> int:
    """Run multiple layers and return the total failure count."""
    total_failed = 0
    for layer in layers:
        results, n_failed = run_layer(layer, fail_fast=fail_fast)
        render_report(layer, results, n_failed)
        total_failed += n_failed
        if fail_fast and n_failed:
            break
    return total_failed
