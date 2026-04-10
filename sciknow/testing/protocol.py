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


def l1_autowrite_streams_all_phases() -> None:
    """Phase 15.3 — non-writing phases (scoring/verification/CoVe/tree-plan)
    must stream tokens so the GUI's token counter stays alive.

    Verifies:
      - _stream_phase generator helper exists
      - _cove_verify_streaming generator exists (for streamed CoVe)
      - autowrite_section_stream uses `yield from _stream_phase` for the
        scoring + verifying + planning + rescoring phases
      - autowrite uses `yield from _cove_verify_streaming` for CoVe
      - The token events have a 'phase' field so the GUI can route them
    """
    import inspect
    from sciknow.core import book_ops

    assert hasattr(book_ops, "_stream_phase"), "_stream_phase helper missing"
    assert hasattr(book_ops, "_cove_verify_streaming"), \
        "_cove_verify_streaming helper missing"

    aw_src = inspect.getsource(book_ops.autowrite_section_stream)
    # Score / verify / planning / rescore call sites use yield from _stream_phase
    assert "yield from _stream_phase(" in aw_src, \
        "autowrite doesn't use yield from _stream_phase — non-writing phases will be silent"
    assert aw_src.count("yield from _stream_phase(") >= 4, (
        f"expected at least 4 _stream_phase yield-from sites in autowrite "
        f"(planning/scoring/verifying/rescoring), found "
        f"{aw_src.count('yield from _stream_phase(')}"
    )
    assert "yield from _cove_verify_streaming(" in aw_src, \
        "autowrite doesn't use streaming CoVe — CoVe phase will be silent"

    # Token events should carry a phase field. After Phase 19, the
    # writing/revising loops were moved into _stream_with_save which
    # tags tokens via its `phase` positional arg, so we accept either
    # the legacy inline-yield pattern OR the new _stream_with_save call.
    assert '"writing"' in aw_src or '"phase": "writing"' in aw_src, \
        "writing tokens should be tagged with phase"
    assert '"revising"' in aw_src or '"phase": "revising"' in aw_src, \
        "revising tokens should be tagged with phase"


def l1_author_search_rejects_non_author() -> None:
    """Phase 16.2 — search_author must drop papers where the searched
    surname isn't in the author list.

    The motivating risk: a future API change or off-by-one bug could
    let `search_author("Zharkova")` return a paper whose only mention
    of "Zharkova" is in the references/citations, not in the authors.
    The defensive `_surname_in_authors` last-pass check guarantees this
    can never happen silently.

    This test calls _surname_in_authors directly with crafted Reference
    objects covering:
      - paper authored by "Valentina V. Zharkova"  → KEEP
      - paper authored by "V. Zharkova"            → KEEP (token match)
      - paper authored by "Zharkova, V. V."        → KEEP (Last, First)
      - paper authored only by other people       → DROP
      - paper authored only by "Zhang"            → DROP (substring trap)
    """
    from sciknow.ingestion.author_search import _surname_in_authors
    from sciknow.ingestion.references import Reference

    keep_cases = [
        ["Valentina V. Zharkova", "Mykola Gordovskyy"],
        ["V. Zharkova", "S. Zharkov"],
        ["Zharkova, V. V."],
        ["Mykola Gordovskyy", "Valentina Zharkova"],  # second author
    ]
    for authors in keep_cases:
        ref = Reference(raw_text="t", doi="10.0/x", title="t", year=2020, authors=authors)
        assert _surname_in_authors(ref, "Zharkova"), \
            f"should KEEP paper with authors={authors}"

    drop_cases = [
        ["Smith, John", "Jones, Jane"],            # no Zharkova
        ["Stanley Ipson", "Ali Benkhalil"],        # also no Zharkova
        ["Zhang Wei", "Liu Min"],                  # 'Zhang' is not 'Zharkova'
        [],                                        # empty author list
    ]
    for authors in drop_cases:
        ref = Reference(raw_text="t", doi="10.0/x", title="t", year=2020, authors=authors)
        assert not _surname_in_authors(ref, "Zharkova"), \
            f"should DROP paper with authors={authors}"

    # Confirm the bug class is closed: a paper that "cites" Zharkova in its
    # title but is authored by other people gets dropped.
    citation_only = Reference(
        raw_text="t",
        doi="10.0/x",
        title="A Critique of Zharkova's Solar Activity Predictions",
        year=2020,
        authors=["Mark Smith", "Jane Doe"],
    )
    assert not _surname_in_authors(citation_only, "Zharkova"), \
        "papers that merely cite Zharkova in the title must be dropped"


def l1_relevance_filter_imports_resolve() -> None:
    """Phase 16.1 — every name imported from sciknow.retrieval.relevance
    inside cli/db.py must actually exist in that module.

    The motivating bug: `db expand-author` had a copy-paste error that
    imported `build_corpus_centroid` and `embed_anchor`, neither of which
    exist (the real names are compute_corpus_centroid and embed_query).
    The graceful try/except in the calling code hid the ImportError —
    the relevance filter silently never ran. Only a real e2e run
    surfaced it.

    This test parses cli/db.py with ast, finds every
    `from sciknow.retrieval.relevance import X, Y, Z` statement, and
    asserts that each imported name resolves to a real attribute on
    the relevance module. Catches the bug at L1 (zero deps) instead
    of waiting for someone to run the command.
    """
    import ast, inspect
    from sciknow.retrieval import relevance
    from sciknow.cli import db as cli_db

    available = set(dir(relevance))
    src = inspect.getsource(cli_db)
    tree = ast.parse(src)

    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "sciknow.retrieval.relevance":
            continue
        for alias in node.names:
            n = alias.name
            if n not in available:
                bad.append(n)

    assert not bad, (
        f"cli/db.py imports non-existent names from sciknow.retrieval.relevance: "
        f"{bad}. Real exports: {sorted(n for n in available if not n.startswith('_'))}"
    )


def l1_phase16_expand_author() -> None:
    """Phase 16 — `sciknow db expand-author` author search command.

    Verifies the new author_search module exposes its public API,
    the CLI command is registered, and the command's source references
    both backends + dedup logic + the existing find_and_download
    + _run_parallel_workers infrastructure (so it doesn't accidentally
    re-implement them).
    """
    import inspect

    # Module + public API
    from sciknow.ingestion import author_search
    for fn in ("search_openalex_by_author", "search_crossref_by_author", "search_author"):
        assert hasattr(author_search, fn), f"author_search.{fn} missing"

    # search_author returns (refs, source_counts) tuple
    sig = inspect.signature(author_search.search_author)
    for kw in ("orcid", "year_from", "year_to", "limit"):
        assert kw in sig.parameters, f"search_author missing kwarg: {kw}"

    # OpenAlex search supports orcid + year filters + require_doi
    sig_oa = inspect.signature(author_search.search_openalex_by_author)
    for kw in ("orcid", "year_from", "year_to", "limit", "require_doi"):
        assert kw in sig_oa.parameters, f"search_openalex_by_author missing kwarg: {kw}"

    # CLI command registered
    from sciknow.cli import db as cli_db
    cmds = {c.name for c in cli_db.app.registered_commands}
    assert "expand-author" in cmds, f"expand-author not on db app — found: {sorted(cmds)}"

    # Command source delegates to existing infrastructure
    src = inspect.getsource(cli_db.expand_author)
    assert "search_author(" in src, "command doesn't call search_author"
    assert "find_and_download(" in src, \
        "command doesn't reuse find_and_download — would re-implement OA discovery"
    assert "_run_parallel_workers(" in src, \
        "command doesn't reuse _run_parallel_workers — would re-implement ingest"
    # Dedup against existing corpus
    assert "existing_dois" in src, "command doesn't dedup against corpus DOIs"


def l1_retrieval_device_helper() -> None:
    """Phase 15.2 — bge-m3 + reranker have a CPU fallback for the case
    where the LLM has filled VRAM.

    Verifies the device helper module exists, the env var override works,
    the cache can be reset, and all three loaders (hybrid_search,
    embedder, reranker) call load_with_cpu_fallback.
    """
    import inspect, os

    # Helper module exists
    from sciknow.retrieval import device
    assert hasattr(device, "get_retrieval_device"), "get_retrieval_device missing"
    assert hasattr(device, "load_with_cpu_fallback"), "load_with_cpu_fallback missing"
    assert hasattr(device, "reset_device_cache"), "reset_device_cache missing"

    # Env var override works
    device.reset_device_cache()
    os.environ["SCIKNOW_RETRIEVAL_DEVICE"] = "cpu"
    try:
        assert device.get_retrieval_device() == "cpu", "cpu override didn't apply"
    finally:
        os.environ.pop("SCIKNOW_RETRIEVAL_DEVICE", None)
        device.reset_device_cache()

    # Auto mode returns something sensible (cpu or cuda)
    assert device.get_retrieval_device() in ("cpu", "cuda"), \
        "auto mode returned unexpected value"
    device.reset_device_cache()

    # All three loaders use the helper
    from sciknow.retrieval import hybrid_search, reranker
    from sciknow.ingestion import embedder
    for mod_name, fn in [
        ("hybrid_search", hybrid_search._get_embed_model),
        ("reranker", reranker._get_reranker),
        ("embedder", embedder._get_model),
    ]:
        src = inspect.getsource(fn)
        assert "load_with_cpu_fallback" in src, \
            f"{mod_name} loader doesn't use load_with_cpu_fallback — CUDA OOM is unguarded"


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


# ── Phase 17 — length as a scoring dimension ────────────────────────────────


def l1_phase17_prompts_length_target() -> None:
    """Phase 17 — writer/planner/scorer prompts wire a length target.

    The writer (`write_section_v2`) must produce a length instruction
    block ONLY when a positive target_words is passed. The planner
    (`tree_plan`) must derive a paragraph count from the target. The
    scorer (`score_draft`) must enumerate `length` as a dimension with
    a fixed 1.0 default so the orchestrator can inject a mechanical
    actual/target score without fighting the LLM.
    """
    from sciknow.rag import prompts

    # Writer with target — length instruction appears.
    sys_w_with, _ = prompts.write_section_v2(
        "results", "x", [],
        book_plan=None, prior_summaries=None,
        target_words=6000,
    )
    assert "Length target" in sys_w_with, (
        "write_section_v2(target_words=6000) missing 'Length target' block"
    )
    assert "approximately 6000 words" in sys_w_with, (
        "write_section_v2 doesn't embed the numeric target into the prompt"
    )

    # Writer without target — length block absent. The word "Length"
    # shouldn't leak into the prompt when we haven't asked for one.
    sys_w_without, _ = prompts.write_section_v2(
        "results", "x", [],
        book_plan=None, prior_summaries=None,
        target_words=None,
    )
    assert "Length target" not in sys_w_without, (
        "write_section_v2 should not inject a length block when target_words is None"
    )

    # Tree planner — paragraph count derived from target.
    sys_t_with, _ = prompts.tree_plan(
        "results", "x", [],
        book_plan=None, prior_summaries=None,
        target_words=6000,
    )
    assert "6000-word" in sys_t_with, (
        "tree_plan doesn't surface the word target in its length block"
    )
    # Without target, falls back to the old 5-10 paragraphs heuristic.
    sys_t_without, _ = prompts.tree_plan(
        "results", "x", [],
        book_plan=None, prior_summaries=None,
        target_words=None,
    )
    assert "5-10 paragraphs" in sys_t_without, (
        "tree_plan lost its default paragraph-count heuristic"
    )

    # Scorer — length as a 7th dimension with fixed default.
    sys_s, _ = prompts.score_draft("results", "x", "draft", [])
    assert "length" in sys_s, "SCORE_SYSTEM missing 'length' dimension"
    assert "seven dimensions" in sys_s, (
        "SCORE_SYSTEM intro still claims 'six dimensions'"
    )
    assert "actual_words / target_words" in sys_s, (
        "SCORE_SYSTEM doesn't explain the mechanical length formula"
    )


def l1_phase17_book_ops_length_helpers() -> None:
    """Phase 17 — book_ops exposes the length target helpers and threads
    target_words through write/autowrite."""
    import inspect
    from sciknow.core import book_ops

    # Public helpers exist.
    assert hasattr(book_ops, "DEFAULT_TARGET_CHAPTER_WORDS"), (
        "DEFAULT_TARGET_CHAPTER_WORDS missing — GUI + CLI would have no fallback"
    )
    assert book_ops.DEFAULT_TARGET_CHAPTER_WORDS > 0
    assert hasattr(book_ops, "LENGTH_PRIORITY_THRESHOLD")
    assert 0 < book_ops.LENGTH_PRIORITY_THRESHOLD < 1, (
        "LENGTH_PRIORITY_THRESHOLD must be in (0, 1)"
    )
    assert hasattr(book_ops, "_section_target_words")
    assert hasattr(book_ops, "_compute_length_score")
    assert hasattr(book_ops, "_get_book_length_target")
    assert hasattr(book_ops, "_get_chapter_num_sections")

    # _compute_length_score behaves correctly.
    assert book_ops._compute_length_score("word " * 3000, 6000) == 0.5
    assert book_ops._compute_length_score("word " * 9000, 6000) == 1.0, (
        "_compute_length_score should cap at 1.0 for over-target drafts"
    )
    assert book_ops._compute_length_score("anything", None) == 1.0, (
        "_compute_length_score(target=None) must return 1.0 (length not evaluated)"
    )
    assert book_ops._compute_length_score("anything", 0) == 1.0

    # _section_target_words divides sensibly and floors at 400.
    assert book_ops._section_target_words(6000, 4) == 1500
    assert book_ops._section_target_words(6000, 1) == 6000
    assert book_ops._section_target_words(1000, 10) == 400, (
        "_section_target_words must floor at 400 to keep sections book-grade"
    )

    # Signatures wire target_words through.
    aw_sig = inspect.signature(book_ops.autowrite_section_stream)
    assert "target_words" in aw_sig.parameters, (
        "autowrite_section_stream missing target_words kwarg"
    )
    ws_sig = inspect.signature(book_ops.write_section_stream)
    assert "target_words" in ws_sig.parameters, (
        "write_section_stream missing target_words kwarg"
    )


def l1_phase17_autowrite_length_loop() -> None:
    """Phase 17 — autowrite's scoring loop computes a length score,
    injects it into the scores dict, and guards against oscillation.
    """
    import inspect
    from sciknow.core import book_ops

    src = inspect.getsource(book_ops.autowrite_section_stream)

    # Length is computed and injected into scores.
    assert "_compute_length_score(content" in src, (
        "autowrite scoring loop doesn't compute a length score for the current draft"
    )
    assert 'scores["length"] = length_score' in src, (
        "autowrite doesn't inject length_score into the scores dict"
    )

    # Anti-oscillation guard: length only wins when below threshold AND
    # below all other dimensions.
    assert "LENGTH_PRIORITY_THRESHOLD" in src, (
        "autowrite doesn't reference LENGTH_PRIORITY_THRESHOLD — "
        "length would dominate the loop even for mild misses"
    )
    assert "length_score < min_other" in src, (
        "autowrite missing the 'length_score < min_other' anti-oscillation guard"
    )

    # When length wins, a targeted expand instruction is generated.
    assert 'weakest_dimension"] = "length"' in src, (
        "autowrite never sets weakest_dimension='length'"
    )
    assert "additional substantive" in src, (
        "autowrite's length revision instruction doesn't demand substantive expansion"
    )

    # Re-scoring: length is computed on the revised draft too, so a
    # successful expansion reflects in the KEEP comparison.
    assert "_compute_length_score(revised" in src, (
        "autowrite re-scoring doesn't recompute length on the revised draft — "
        "length-driven revisions would be discarded even when successful"
    )

    # target_words persisted in the checkpoint metadata so `book draft
    # scores` and the GUI can show it after the fact.
    assert '"target_words": effective_target_words' in src, (
        "autowrite doesn't persist target_words in checkpoint metadata"
    )


def l1_phase17_cli_length_flags() -> None:
    """Phase 17 — CLI exposes --target-words on write/autowrite and
    --target-chapter-words on create.
    """
    from typer.main import get_command
    from sciknow.cli import book as book_cli

    cmd = get_command(book_cli.app)
    # Walk subcommands; each is a click.Command with .params list.
    subs = {c.name: c for c in cmd.commands.values()}

    def _param_names(cmd_obj):
        return {p.name for p in getattr(cmd_obj, "params", [])}

    assert "write" in subs, "book write command missing"
    assert "target_words" in _param_names(subs["write"]), (
        "`sciknow book write` missing --target-words flag"
    )

    assert "autowrite" in subs, "book autowrite command missing"
    assert "target_words" in _param_names(subs["autowrite"]), (
        "`sciknow book autowrite` missing --target-words flag"
    )

    assert "create" in subs, "book create command missing"
    assert "target_chapter_words" in _param_names(subs["create"]), (
        "`sciknow book create` missing --target-chapter-words flag"
    )


def l1_phase17_web_length_target() -> None:
    """Phase 17 — web app exposes the length target: GET /api/book
    returns target_chapter_words, PUT accepts it, api_write /
    api_autowrite accept target_words, and the Plan modal has a
    length preset field.
    """
    import inspect
    from sciknow.web import app as web_app

    # Endpoint signatures.
    sig_put = inspect.signature(web_app.api_book_update)
    assert "target_chapter_words" in sig_put.parameters, (
        "PUT /api/book missing target_chapter_words form field"
    )

    sig_get = inspect.getsource(web_app.api_book)
    assert "target_chapter_words" in sig_get, (
        "GET /api/book doesn't return target_chapter_words"
    )
    assert "default_target_chapter_words" in sig_get, (
        "GET /api/book doesn't expose the default for the GUI to display"
    )

    sig_write = inspect.signature(web_app.api_write)
    assert "target_words" in sig_write.parameters, (
        "POST /api/write missing target_words form field"
    )

    sig_auto = inspect.signature(web_app.api_autowrite)
    assert "target_words" in sig_auto.parameters, (
        "POST /api/autowrite missing target_words form field"
    )

    # HTML template has the length field + JS helpers.
    src = inspect.getsource(web_app)
    assert "plan-target-words-input" in src, (
        "Plan modal missing target-words input — GUI users can't set length"
    )
    assert "setLengthPreset" in src, (
        "Plan modal missing setLengthPreset() helper for preset buttons"
    )
    assert "Target chapter length" in src, (
        "Plan modal missing 'Target chapter length' label"
    )


# ── Phase 18 — chapter sections + citation fix ──────────────────────────────


def l1_phase18_citation_numbering_consistent() -> None:
    """Phase 18 — format_context and format_sources must produce IDENTICAL
    [N] → paper mappings, even when the input contains duplicates.

    The bug this catches: format_context used to use `r.rank` (the
    pre-dedup retrieval rank) while format_sources used
    `enumerate(start=1)` after dedup. When dedup removed any duplicate,
    the writer's prompt contained gaps in its [N] sequence, but the
    saved sources panel had no gaps — every citation past the gap
    referenced a different paper than the panel said.
    """
    from sciknow.rag.prompts import format_context, format_sources
    from sciknow.retrieval.context_builder import SearchResult
    import re

    def _mk(rank, doc_id, title):
        return SearchResult(
            rank=rank, score=0.5, chunk_id=f"c{rank}",
            document_id=doc_id, section_type="text", section_title=None,
            content=f"This is the body of {title}.",
            title=title, year=2020, authors=[], journal=None, doi=None,
        )

    # 5 results, with results[1] being a duplicate of results[0].
    # Dedup should drop the dupe; both functions should produce a
    # 4-paper sequence numbered [1] [2] [3] [4].
    results = [
        _mk(1, "doc-a", "Paper A"),
        _mk(2, "doc-a", "Paper A"),  # duplicate of #1
        _mk(3, "doc-b", "Paper B"),
        _mk(4, "doc-c", "Paper C"),
        _mk(5, "doc-d", "Paper D"),
    ]

    ctx = format_context(results)
    src = format_sources(results)

    # Both should reference exactly [1], [2], [3], [4] — no gaps, no [5].
    ctx_nums = sorted({int(m) for m in re.findall(r'\[(\d+)\]', ctx)})
    src_nums = sorted({int(m) for m in re.findall(r'\[(\d+)\]', src)})
    assert ctx_nums == [1, 2, 3, 4], (
        f"format_context numbering wrong: {ctx_nums} (expected [1,2,3,4])"
    )
    assert src_nums == [1, 2, 3, 4], (
        f"format_sources numbering wrong: {src_nums} (expected [1,2,3,4])"
    )

    # And critically — the same [N] must reference the SAME paper in both.
    # The APA format used by _apa_citation puts the title right after
    # "(year). ", so we can pull "[N] (year). Title." with one regex
    # and build a {N: title} map for each side.
    def _build_n_to_title(text):
        out = {}
        for m in re.finditer(r'\[(\d+)\]\s*\(\d+\)\.\s*([^.]+?)\.', text):
            out[int(m.group(1))] = m.group(2).strip()
        return out

    ctx_map = _build_n_to_title(ctx)
    src_map = _build_n_to_title(src)
    assert ctx_map, f"Failed to extract any [N]→title from context: {ctx[:300]}"
    assert src_map, f"Failed to extract any [N]→title from sources: {src[:300]}"
    for n, title in src_map.items():
        assert ctx_map.get(n) == title, (
            f"Citation [{n}] references different papers: "
            f"context says {ctx_map.get(n)!r}, sources says {title!r}"
        )


def l1_phase18_chapter_sections_normalize() -> None:
    """Phase 18 — _normalize_chapter_sections accepts both legacy
    string-list and new dict-list formats and returns dicts.
    """
    from sciknow.core.book_ops import _normalize_chapter_sections

    # Empty / None → empty list
    assert _normalize_chapter_sections(None) == []
    assert _normalize_chapter_sections([]) == []

    # Legacy: list of strings
    legacy = ["overview", "key_evidence", "summary"]
    out = _normalize_chapter_sections(legacy)
    assert len(out) == 3
    assert out[0] == {"slug": "overview", "title": "Overview", "plan": ""}
    assert out[1]["slug"] == "key_evidence"
    assert out[1]["title"] == "Key Evidence"

    # New: list of dicts
    new = [
        {"slug": "solar_cycle", "title": "The 11-Year Cycle",
         "plan": "Cover sunspot counts and butterfly diagram."},
        {"title": "Geomagnetic Storms"},  # missing slug → derive from title
        {"slug": "cosmic_rays"},          # missing title → titleify slug
    ]
    out = _normalize_chapter_sections(new)
    assert len(out) == 3
    assert out[0]["slug"] == "solar_cycle"
    assert out[0]["title"] == "The 11-Year Cycle"
    assert out[0]["plan"] == "Cover sunspot counts and butterfly diagram."
    assert out[1]["slug"] == "geomagnetic_storms"
    assert out[1]["title"] == "Geomagnetic Storms"
    assert out[1]["plan"] == ""
    assert out[2]["slug"] == "cosmic_rays"
    assert out[2]["title"] == "Cosmic Rays"

    # JSON-encoded string also works
    out = _normalize_chapter_sections('["a", "b"]')
    assert len(out) == 2 and out[0]["slug"] == "a"


def l1_phase18_section_plan_threaded() -> None:
    """Phase 18 — tree_plan and write_section_v2 accept section_plan
    and inject it into the prompt when non-empty."""
    import inspect
    from sciknow.rag import prompts
    from sciknow.core import book_ops

    tp_sig = inspect.signature(prompts.tree_plan)
    assert "section_plan" in tp_sig.parameters, (
        "tree_plan missing section_plan kwarg"
    )

    ws_sig = inspect.signature(prompts.write_section_v2)
    assert "section_plan" in ws_sig.parameters, (
        "write_section_v2 missing section_plan kwarg"
    )

    # Without a section plan: no leakage
    sys_w_empty, _ = prompts.write_section_v2(
        "results", "x", [],
        book_plan=None, prior_summaries=None,
        section_plan=None,
    )
    assert "Section plan" not in sys_w_empty, (
        "write_section_v2 leaks section plan block when section_plan is None"
    )

    # With a section plan: it appears verbatim
    sys_w_filled, _ = prompts.write_section_v2(
        "results", "x", [],
        book_plan=None, prior_summaries=None,
        section_plan="Cover sunspot counts and butterfly diagram.",
    )
    assert "Section plan" in sys_w_filled
    assert "sunspot counts" in sys_w_filled

    # tree_plan: same behaviour. Note: tree_plan injects the section
    # plan into the USER message (via plan_context), not the system
    # prompt — we check the full concatenated prompt.
    sys_t_filled, usr_t_filled = prompts.tree_plan(
        "results", "x", [],
        section_plan="Discuss the 11-year cycle in detail.",
    )
    full_t = sys_t_filled + "\n" + usr_t_filled
    assert "Section plan" in full_t, (
        "tree_plan doesn't surface the section plan anywhere in its prompt"
    )
    assert "11-year cycle" in full_t

    # And without a section plan, no leakage in either part
    sys_t_empty, usr_t_empty = prompts.tree_plan(
        "results", "x", [],
        section_plan=None,
    )
    assert "Section plan — what THIS section must cover" not in (sys_t_empty + usr_t_empty), (
        "tree_plan injects an empty Section plan block when section_plan is None"
    )

    # book_ops helpers exist
    assert hasattr(book_ops, "_get_section_plan")
    assert hasattr(book_ops, "_normalize_chapter_sections")
    assert hasattr(book_ops, "_get_chapter_sections_normalized")

    # write_section_stream + autowrite_section_stream pass section_plan
    # through to write_section_v2 — verified by source inspection.
    aw_src = inspect.getsource(book_ops.autowrite_section_stream)
    assert "section_plan=section_plan" in aw_src, (
        "autowrite_section_stream doesn't pass section_plan to write_section_v2/tree_plan"
    )
    ws_src = inspect.getsource(book_ops.write_section_stream)
    assert "section_plan=section_plan" in ws_src, (
        "write_section_stream doesn't pass section_plan to write_section_v2/tree_plan"
    )


def l1_phase18_web_endpoints_and_modal() -> None:
    """Phase 18 — web app exposes the new sections CRUD endpoint and
    the chapter modal has the Sections tab + supporting JS.
    """
    import inspect
    from sciknow.web import app as web_app

    # New endpoint registered
    routes = {getattr(r, "path", ""): r for r in web_app.app.routes}
    assert "/api/chapters/{chapter_id}/sections" in routes, (
        "PUT /api/chapters/{id}/sections endpoint missing — sections CRUD won't work"
    )

    # chapter_reader returns sources_html (Phase 18 fix)
    cr_src = inspect.getsource(web_app.chapter_reader)
    assert "sources_html" in cr_src, (
        "chapter_reader doesn't return sources_html — citation links in "
        "the chapter view will be broken (no source data for popovers)"
    )
    assert "_register_source" in cr_src, (
        "chapter_reader missing global source renumbering — each draft's "
        "[N] would still collide with other drafts'"
    )

    src = inspect.getsource(web_app)

    # Chapter modal has tabs + sections editor
    assert "switchChapterTab" in src, (
        "Chapter modal missing switchChapterTab() — Sections tab is dead"
    )
    assert "ch-sections-pane" in src, "Chapter modal missing #ch-sections-pane"
    assert "ch-sections-list" in src, "Chapter modal missing #ch-sections-list container"
    assert "addSection" in src, "Chapter modal missing addSection() helper"
    assert "removeSection" in src, "Chapter modal missing removeSection() helper"
    assert "moveSection" in src, "Chapter modal missing moveSection() helper"
    assert "renderSectionEditor" in src, (
        "Chapter modal missing renderSectionEditor() — UI never paints"
    )

    # showChapterReader populates panel-sources after fetching
    assert "data.sources_html" in src and "panel-sources" in src, (
        "showChapterReader doesn't populate panel-sources — popovers "
        "won't find source data and citations will be dead"
    )

    # api_chapters surfaces sections_meta to the SPA
    api_src = inspect.getsource(web_app.api_chapters)
    assert "sections_meta" in api_src, (
        "GET /api/chapters doesn't expose sections_meta — chapter modal "
        "can't pre-fill the sections editor"
    )


# ── Phase 19 — incremental save during streaming ───────────────────────────


def l1_phase19_helpers_exist() -> None:
    """Phase 19 — _stream_with_save and _next_draft_version helpers exist
    with the expected shapes.
    """
    import inspect
    from sciknow.core import book_ops

    assert hasattr(book_ops, "_stream_with_save"), (
        "_stream_with_save helper missing — incremental save during writing/revising "
        "won't work, Stop will lose work as before"
    )
    assert hasattr(book_ops, "_next_draft_version"), (
        "_next_draft_version helper missing — new autowrite drafts won't outrank "
        "older completed drafts, refresh after Stop will show the OLD draft"
    )

    # _stream_with_save signature: takes save_callback as kwarg, accepts phase
    sig = inspect.signature(book_ops._stream_with_save)
    assert "save_callback" in sig.parameters, (
        "_stream_with_save missing save_callback parameter"
    )
    # Save interval tunables exposed at module level so a future tuning
    # change can hit a single point.
    assert hasattr(book_ops, "_STREAM_SAVE_INTERVAL_TOKENS")
    assert hasattr(book_ops, "_STREAM_SAVE_INTERVAL_SECONDS")


def l1_phase19_stream_with_save_flushes_on_close() -> None:
    """Phase 19 — _stream_with_save's try/finally MUST call save_callback
    one last time when the generator is closed, even if no save was due
    yet. This is what makes Stop preserve work-in-progress.
    """
    import sciknow.core.book_ops as book_ops

    # Fake stream of 5 tokens. Token-interval and time-interval are set
    # high so the only save that fires is the one in the finally block.
    fake_tokens = ["alpha ", "beta ", "gamma ", "delta ", "epsilon"]

    def fake_stream(*a, **kw):
        for t in fake_tokens:
            yield t

    saved: list[str] = []

    def cb(text: str) -> None:
        saved.append(text)

    # Monkey-patch llm.stream to return our fake (only inside _stream_with_save).
    import sciknow.rag.llm as llm_mod
    real_stream = llm_mod.stream
    llm_mod.stream = fake_stream
    try:
        gen = book_ops._stream_with_save(
            "sys", "usr", "test_phase",
            save_callback=cb,
            save_interval_tokens=10000,  # never trigger periodic save
            save_interval_seconds=10000.0,
        )
        # Pull 2 events, then close — simulating a Stop after 2 tokens.
        events = []
        events.append(next(gen))
        events.append(next(gen))
        gen.close()  # raises GeneratorExit at the current yield point
    finally:
        llm_mod.stream = real_stream

    # The 2 token events should have been yielded
    assert len(events) == 2, f"expected 2 token events, got {len(events)}"
    assert events[0]["type"] == "token"
    assert events[0]["phase"] == "test_phase"

    # Critically: the save_callback should have been called by the
    # finally block with the buffer accumulated so far.
    assert len(saved) >= 1, (
        "_stream_with_save didn't flush buffer on GeneratorExit — "
        "Stop will lose all in-flight tokens"
    )
    assert "alpha" in saved[-1] and "beta" in saved[-1], (
        f"flushed text doesn't contain expected tokens: {saved[-1]!r}"
    )


def l1_phase19_periodic_save_fires() -> None:
    """Phase 19 — _stream_with_save fires save_callback periodically as
    tokens accumulate, not just at the end. With 6 fake tokens and a
    save_interval_tokens of 2, we expect ~3 mid-stream saves PLUS the
    final flush.
    """
    import sciknow.core.book_ops as book_ops
    import sciknow.rag.llm as llm_mod

    fake = ["t1 ", "t2 ", "t3 ", "t4 ", "t5 ", "t6 "]

    def fake_stream(*a, **kw):
        for t in fake:
            yield t

    saved: list[str] = []
    def cb(text: str) -> None:
        saved.append(text)

    real = llm_mod.stream
    llm_mod.stream = fake_stream
    try:
        gen = book_ops._stream_with_save(
            "sys", "usr", "test",
            save_callback=cb,
            save_interval_tokens=2,
            save_interval_seconds=10000.0,
        )
        # Drain the generator naturally (no Stop)
        events = list(gen)
    finally:
        llm_mod.stream = real

    assert len(events) == 6, f"expected 6 token events, got {len(events)}"
    # Periodic saves at tokens 2, 4, 6 + final flush. Final flush may
    # be a no-op duplicate of the last periodic save (same buffer), so
    # we accept >= 3 distinct save calls.
    assert len(saved) >= 3, (
        f"expected >= 3 periodic saves with interval=2 over 6 tokens, "
        f"got {len(saved)}"
    )
    # Last save must contain the full buffer
    assert "t1" in saved[-1] and "t6" in saved[-1], (
        f"last save missing expected tokens: {saved[-1]!r}"
    )


def l1_phase19_autowrite_uses_streaming_save() -> None:
    """Phase 19 — autowrite_section_stream + write_section_stream wire
    their writing/revising loops through _stream_with_save, AND insert
    the placeholder draft BEFORE the writing loop (so periodic saves
    have a row to UPDATE).
    """
    import inspect
    from sciknow.core import book_ops

    aw_src = inspect.getsource(book_ops.autowrite_section_stream)
    ws_src = inspect.getsource(book_ops.write_section_stream)

    # autowrite uses _stream_with_save for both writing and revising
    assert aw_src.count("_stream_with_save(") >= 2, (
        f"autowrite should use _stream_with_save twice (writing + revising), "
        f"found {aw_src.count('_stream_with_save(')}"
    )

    # autowrite uses _next_draft_version so the placeholder outranks
    # older drafts in the latest-version sort
    assert "_next_draft_version(" in aw_src, (
        "autowrite doesn't bump version above existing drafts — refresh "
        "during writing will show an OLDER draft (the user's reported regression)"
    )
    # And uses draft_version (the local from _next_draft_version) when
    # bumping on KEEP / final, not the bare iteration counter
    assert "draft_version + iteration" in aw_src, (
        "autowrite KEEP path doesn't bump version relative to the starting "
        "draft_version — could roll back below an older draft"
    )

    # write_section_stream uses _stream_with_save + _next_draft_version
    assert "_stream_with_save(" in ws_src, (
        "write_section_stream's writing loop doesn't use _stream_with_save"
    )
    assert "_next_draft_version(" in ws_src, (
        "write_section_stream doesn't bump version — same regression as autowrite"
    )

    # Both functions should have the placeholder INSERT before the
    # writing call (i.e. _save_draft is called before _stream_with_save).
    # Check by line position in the source.
    aw_lines = aw_src.splitlines()
    save_draft_line = next(
        (i for i, ln in enumerate(aw_lines) if "_save_draft(" in ln),
        -1,
    )
    sws_line = next(
        (i for i, ln in enumerate(aw_lines) if "_stream_with_save(" in ln),
        -1,
    )
    assert save_draft_line >= 0 and sws_line >= 0, (
        "couldn't locate _save_draft and _stream_with_save in autowrite source"
    )
    assert save_draft_line < sws_line, (
        "autowrite's _save_draft (placeholder INSERT) must come BEFORE the "
        "first _stream_with_save call, otherwise periodic writing-saves have "
        "no row to UPDATE and Stop loses everything"
    )


# ── Phase 20 — autowrite all chapter sections from the GUI ────────────────


def l1_phase20_chapter_autowrite_helper_exists() -> None:
    """Phase 20 — autowrite_chapter_all_sections_stream generator exists
    in book_ops, takes the right kwargs, and delegates to
    autowrite_section_stream for each section.
    """
    import inspect
    from sciknow.core import book_ops

    assert hasattr(book_ops, "autowrite_chapter_all_sections_stream"), (
        "autowrite_chapter_all_sections_stream missing — toolbar Autowrite "
        "without a section selected can't iterate over the chapter's sections"
    )
    fn = book_ops.autowrite_chapter_all_sections_stream
    sig = inspect.signature(fn)
    for kw in (
        "book_id", "chapter_id", "model", "max_iter", "target_score",
        "target_words", "rebuild",
    ):
        assert kw in sig.parameters, (
            f"autowrite_chapter_all_sections_stream missing {kw} parameter"
        )

    src = inspect.getsource(fn)
    # It must call the single-section generator under the hood, NOT
    # re-implement the convergence loop (which would drift over time).
    assert "autowrite_section_stream(" in src, (
        "autowrite_chapter_all_sections_stream doesn't delegate to "
        "autowrite_section_stream — would diverge over time"
    )
    # Reads the chapter's sections list, not a hardcoded paper-style set
    assert "_get_chapter_sections_normalized(" in src, (
        "autowrite_chapter_all_sections_stream isn't reading the chapter's "
        "actual sections list"
    )
    # Skips already-drafted sections by default
    assert "existing_slugs" in src and "rebuild" in src, (
        "autowrite_chapter_all_sections_stream doesn't have skip-existing "
        "logic — would clobber previous work on every re-run"
    )
    # Has section_start / section_done envelope events for the GUI to
    # render per-section progress
    for ev in ("chapter_autowrite_start", "section_start", "section_done",
               "all_sections_complete"):
        assert ev in src, (
            f"autowrite_chapter_all_sections_stream missing {ev} event "
            f"— GUI can't show per-section progress"
        )


def l1_phase20_web_endpoint_and_router() -> None:
    """Phase 20 — POST /api/autowrite-chapter endpoint registered AND
    the JS doAutowrite() routes "no section selected" to it instead
    of defaulting to section_type='introduction'.
    """
    import inspect
    from sciknow.web import app as web_app

    routes = {getattr(r, "path", "") for r in web_app.app.routes}
    assert "/api/autowrite-chapter" in routes, (
        "POST /api/autowrite-chapter not registered — toolbar all-sections "
        "autowrite has nowhere to call"
    )

    src = inspect.getsource(web_app)
    # JS routes based on isAllSections
    assert "isAllSections" in src, (
        "doAutowrite() doesn't detect 'no section selected' — toolbar "
        "Autowrite still defaults to section_type='introduction'"
    )
    assert "/api/autowrite-chapter" in src, (
        "doAutowrite() never calls /api/autowrite-chapter"
    )
    # Handles the new envelope events
    for ev in ("chapter_autowrite_start", "section_start", "section_done",
               "all_sections_complete"):
        assert "'" + ev + "'" in src or '"' + ev + '"' in src, (
            f"JS doAutowrite() doesn't handle {ev} event"
        )


def l1_phase20_broken_citation_indicator() -> None:
    """Phase 20 — buildPopovers marks orphan/broken citations with
    .citation-broken class so the user can see which links are dead
    (instead of clicking and getting nothing).
    """
    import inspect
    from sciknow.web import app as web_app

    src = inspect.getsource(web_app)
    assert "citation-broken" in src, (
        "buildPopovers doesn't tag broken citations — orphan refs "
        "look identical to working ones, click silently does nothing"
    )
    # CSS rule exists so the broken state is visually distinct
    assert ".citation.citation-broken" in src, (
        "missing .citation.citation-broken CSS — broken state not visible"
    )
    # Click is suppressed on broken citations (no scroll-to-nowhere)
    assert "e.preventDefault" in src or "preventDefault" in src
    # Console warning for debugging
    assert "broken citation" in src.lower(), (
        "buildPopovers doesn't console.warn about broken citations"
    )


# ── Phase 21 — MinerU 2.5-Pro + sections UX + plan modal context ──────────


def l1_phase21_mineru_pro_helper() -> None:
    """Phase 21 — MinerU 2.5-Pro VLM backend opt-in plumbing.

    Verifies the helper that monkey-patches mineru.utils.enum_class.ModelPath
    to point at the Pro model, and that pdf_converter.convert() routes
    "mineru-vlm-pro" through that helper instead of the legacy pipeline
    backend.
    """
    import inspect
    from sciknow.ingestion import pdf_converter

    assert hasattr(pdf_converter, "_patch_mineru_to_pro_model"), (
        "_patch_mineru_to_pro_model helper missing — VLM Pro can't be selected"
    )
    assert hasattr(pdf_converter, "_MINERU_PRO_DEFAULT_HF"), (
        "_MINERU_PRO_DEFAULT_HF constant missing"
    )
    assert "MinerU2.5-Pro" in pdf_converter._MINERU_PRO_DEFAULT_HF, (
        f"_MINERU_PRO_DEFAULT_HF doesn't reference the Pro model: "
        f"{pdf_converter._MINERU_PRO_DEFAULT_HF!r}"
    )

    # The convert() dispatch handles "mineru-vlm-pro" explicitly
    convert_src = inspect.getsource(pdf_converter.convert)
    assert '"mineru-vlm-pro"' in convert_src, (
        "convert() doesn't handle the mineru-vlm-pro backend setting"
    )
    assert "use_vlm_pro=True" in convert_src, (
        "convert() doesn't pass use_vlm_pro=True for the Pro path"
    )

    # _convert_mineru takes the use_vlm_pro kwarg
    sig = inspect.signature(pdf_converter._convert_mineru)
    assert "use_vlm_pro" in sig.parameters, (
        "_convert_mineru missing use_vlm_pro kwarg"
    )

    # The Settings class exposes the new mineru_vlm_model field
    from sciknow.config import settings
    assert hasattr(settings, "mineru_vlm_model"), (
        "settings.mineru_vlm_model field missing"
    )

    # The patch helper actually mutates ModelPath when called
    try:
        from mineru.utils.enum_class import ModelPath
    except ImportError:
        return  # mineru not installed in this env; skip the live patch test
    original = ModelPath.vlm_root_hf
    try:
        pdf_converter._patch_mineru_to_pro_model("opendatalab/MinerU2.5-Pro-2604-1.2B")
        assert ModelPath.vlm_root_hf == "opendatalab/MinerU2.5-Pro-2604-1.2B", (
            "_patch_mineru_to_pro_model didn't update vlm_root_hf"
        )
        assert "OpenDataLab" in ModelPath.vlm_root_modelscope, (
            "_patch_mineru_to_pro_model didn't update vlm_root_modelscope"
        )
    finally:
        # Restore so other tests aren't affected
        ModelPath.vlm_root_hf = original


def l1_phase21_sidebar_renders_template_slots() -> None:
    """Phase 21 — sidebar shows ALL chapter sections (template slots),
    not just sections that already have drafts. Empty slots become
    inline "Write" CTAs; orphan drafts are visible at the end.
    """
    import inspect
    from sciknow.web import app as web_app

    src = inspect.getsource(web_app)

    # _render_sidebar handles 'empty', 'drafted', 'orphan' status values
    render_src = inspect.getsource(web_app._render_sidebar)
    assert '"empty"' in render_src or "'empty'" in render_src, (
        "_render_sidebar doesn't render empty template slots"
    )
    assert '"orphan"' in render_src or "'orphan'" in render_src, (
        "_render_sidebar doesn't render orphan drafts"
    )
    assert "writeForCell" in render_src, (
        "_render_sidebar empty slots don't trigger writeForCell"
    )

    # CSS for the new states
    assert ".sec-status-dot" in src, "section status dot CSS missing"
    assert ".sec-link.sec-empty" in src, "empty-slot sidebar CSS missing"
    assert ".sec-link.sec-orphan" in src, "orphan-draft sidebar CSS missing"

    # The _render_book sections-list builder produces entries for the
    # FULL template, not just drafted sections
    rb_src = inspect.getsource(web_app._render_book)
    assert '"status": "empty"' in rb_src, (
        "_render_book doesn't emit empty template slot entries"
    )
    assert '"status": "orphan"' in rb_src, (
        "_render_book doesn't emit orphan draft entries"
    )

    # JS rebuildSidebar mirrors the same three states so post-write
    # sidebar refreshes show empty/orphan correctly
    assert "sec-empty" in src and "sec-orphan" in src, (
        "rebuildSidebar JS doesn't render the new sec-empty/sec-orphan states"
    )


def l1_phase21_section_editor_live_slug_and_budget() -> None:
    """Phase 21 — chapter modal section editor shows live slug preview
    and per-section word budget."""
    import inspect
    from sciknow.web import app as web_app

    src = inspect.getsource(web_app)

    # Live slug derivation from title in the JS
    assert "updateSectionTitle" in src, (
        "updateSectionTitle helper missing — slug doesn't update live"
    )
    # Per-section word budget computed in renderSectionEditor
    assert "perSection" in src, (
        "renderSectionEditor doesn't compute per-section word budget"
    )
    assert "_chapterWordTarget" in src, (
        "renderSectionEditor doesn't read the chapter word target"
    )


def l1_phase21_plan_modal_context_aware() -> None:
    """Phase 21 — Plan modal has Book / Chapter / Section tabs and
    openPlanModal() accepts a context arg that auto-routes based on
    the current selection.
    """
    import inspect
    from sciknow.web import app as web_app

    src = inspect.getsource(web_app)

    # Tabs in the HTML
    assert "plan-tab-chapter" in src, "Plan modal missing Chapter tab"
    assert "plan-tab-section" in src, "Plan modal missing Section tab"
    assert "plan-chapter-pane" in src, "Plan modal missing chapter pane"
    assert "plan-section-pane" in src, "Plan modal missing section pane"

    # JS dispatch helpers
    assert "switchPlanTab" in src, "switchPlanTab JS helper missing"
    assert "populatePlanChapterTab" in src, "populatePlanChapterTab missing"
    assert "populatePlanSectionTab" in src, "populatePlanSectionTab missing"
    assert "savePlanBook" in src, "savePlanBook missing"
    assert "savePlanChapterSections" in src, "savePlanChapterSections missing"
    assert "savePlanSection" in src, "savePlanSection missing"

    # Context-aware open: detects section/chapter/book mode
    assert "_planContext" in src, "openPlanModal doesn't track context state"
    assert "currentSectionType" in src and "currentChapterId" in src, (
        "openPlanModal doesn't read currentSectionType / currentChapterId "
        "to derive context"
    )


# ── Phase 22 — XSS fixes + job cleanup + GUI improvements ────────────────


def l1_phase22_render_helpers_escape_html() -> None:
    """Phase 22 — every _render_* helper escapes user-controlled data
    so a chapter title with `<` or a comment with a `<script>` tag
    doesn't break the page or open an XSS hole.
    """
    from sciknow.web import app as web_app

    # _render_sources: APA citation strings get escaped
    out = web_app._render_sources(["[1] <script>alert(1)</script> Title."])
    assert "<script>" not in out, "_render_sources doesn't escape source strings"
    assert "&lt;script&gt;" in out, (
        "_render_sources didn't HTML-escape the < character"
    )

    # _render_search: draft title gets escaped. We check that the
    # angle brackets become entities — the literal text "onerror"
    # surviving inside an escaped run is fine because it can't fire
    # without the surrounding < tag.
    out = web_app._render_search([
        ("abc-123", "Title <img src=x onerror=alert(1)>", None, None, 100),
    ])
    assert "<img" not in out, (
        "_render_search doesn't escape angle brackets in draft titles"
    )
    assert "&lt;img" in out, "_render_search didn't HTML-escape <img"

    # _render_comments: selected text + body + cid all escaped
    out = web_app._render_comments([
        ("cid-1", "did-1", 1, "<b>selected</b>", "comment <script>x</script>",
         "open", None),
    ])
    assert "<script>" not in out, "_render_comments leaks unescaped script tags"
    assert "&lt;script&gt;" in out

    # _render_sidebar: chapter title + section titles get escaped
    items = [{
        "id": "ch-1", "num": 1, "title": "Title <img src=x>",
        "description": "", "topic_query": "",
        "sections": [
            {"id": "d1", "type": "intro", "title": "Sec <b>bold</b>",
             "plan": "", "version": 1, "words": 100, "status": "drafted"},
        ],
        "sections_template": ["intro"],
        "sections_meta": [{"slug": "intro", "title": "Sec <b>bold</b>", "plan": ""}],
    }]
    out = web_app._render_sidebar(items, "d1")
    assert "<img" not in out, "_render_sidebar doesn't escape chapter titles"
    assert "<b>bold</b>" not in out, "_render_sidebar doesn't escape section titles"
    assert "&lt;b&gt;" in out


def l1_phase22_job_cleanup() -> None:
    """Phase 22 — _gc_old_jobs removes finished jobs older than the
    cleanup window. Without this, the _jobs dict grows unbounded
    over a long writing session.
    """
    import time as _time
    from sciknow.web import app as web_app

    assert hasattr(web_app, "_gc_old_jobs"), "_gc_old_jobs helper missing"
    assert hasattr(web_app, "_JOB_GC_AGE_SECONDS")

    # Inject a fake old finished job and a fake recent finished job
    with web_app._job_lock:
        web_app._jobs["fake-old"] = {
            "queue": None, "status": "done", "type": "test",
            "cancel": None,
            "finished_at": _time.monotonic() - (web_app._JOB_GC_AGE_SECONDS + 100),
        }
        web_app._jobs["fake-recent"] = {
            "queue": None, "status": "done", "type": "test",
            "cancel": None,
            "finished_at": _time.monotonic() - 1,
        }
        web_app._jobs["fake-running"] = {
            "queue": None, "status": "running", "type": "test",
            "cancel": None, "finished_at": None,
        }
        n = web_app._gc_old_jobs()
    try:
        assert n == 1, f"_gc_old_jobs evicted {n} (expected 1)"
        assert "fake-old" not in web_app._jobs, "old finished job not evicted"
        assert "fake-recent" in web_app._jobs, "recent finished job evicted prematurely"
        assert "fake-running" in web_app._jobs, "running job evicted"
    finally:
        with web_app._job_lock:
            web_app._jobs.pop("fake-recent", None)
            web_app._jobs.pop("fake-running", None)
            web_app._jobs.pop("fake-old", None)


def l1_phase22_delete_draft_endpoint_and_orphan_cleanup() -> None:
    """Phase 22 — DELETE /api/draft/{id} endpoint registered + JS
    deleteOrphanDraft helper wires the inline X button on orphan
    drafts in the sidebar."""
    import inspect
    from sciknow.web import app as web_app

    routes = {getattr(r, "path", "") for r in web_app.app.routes}
    assert "/api/draft/{draft_id}" in routes, (
        "DELETE /api/draft/{draft_id} not registered"
    )

    # The JS template defines deleteOrphanDraft and the orphan row
    # renders the X button + click handler.
    src = inspect.getsource(web_app)
    assert "deleteOrphanDraft" in src, (
        "deleteOrphanDraft JS helper missing — orphan X button is dead"
    )
    assert "sec-orphan-delete" in src, (
        "missing sec-orphan-delete CSS class for the inline X button"
    )

    # _render_sidebar emits the X button on orphan rows
    render_src = inspect.getsource(web_app._render_sidebar)
    assert "sec-orphan-delete" in render_src, (
        "_render_sidebar doesn't emit the orphan delete button"
    )


def l1_phase22_chapter_progress_and_word_target() -> None:
    """Phase 22 — chapter completion progress bar in sidebar + word
    target progress bar in subtitle.
    """
    import inspect
    from sciknow.web import app as web_app

    src = inspect.getsource(web_app)

    # Chapter progress bar in sidebar HTML + CSS
    assert "ch-progress" in src, "missing ch-progress CSS class"
    assert "ch-progress-fill" in src, "missing ch-progress-fill bar"
    render_src = inspect.getsource(web_app._render_sidebar)
    assert "ch-progress" in render_src, (
        "_render_sidebar doesn't emit the chapter progress bar"
    )

    # Word target bar in subtitle + JS updater
    assert "word-target" in src, "missing word-target CSS class"
    assert "updateWordTargetBar" in src, (
        "updateWordTargetBar JS helper missing"
    )

    # api_section returns target_words so the JS can populate the bar
    api_src = inspect.getsource(web_app.api_section)
    assert "target_words" in api_src, (
        "api_section doesn't return target_words for the GUI bar"
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
    l1_autowrite_streams_all_phases,
    l1_retrieval_device_helper,
    l1_phase16_expand_author,
    l1_author_search_rejects_non_author,
    l1_relevance_filter_imports_resolve,
    l1_web_rendered_js_is_valid,
    l1_research_doc_up_to_date,
    # Phase 17 — length as a scoring dimension
    l1_phase17_prompts_length_target,
    l1_phase17_book_ops_length_helpers,
    l1_phase17_autowrite_length_loop,
    l1_phase17_cli_length_flags,
    l1_phase17_web_length_target,
    # Phase 18 — chapter sections as first-class entities + citation fix
    l1_phase18_citation_numbering_consistent,
    l1_phase18_chapter_sections_normalize,
    l1_phase18_section_plan_threaded,
    l1_phase18_web_endpoints_and_modal,
    # Phase 19 — incremental save during streaming
    l1_phase19_helpers_exist,
    l1_phase19_stream_with_save_flushes_on_close,
    l1_phase19_periodic_save_fires,
    l1_phase19_autowrite_uses_streaming_save,
    # Phase 20 — autowrite all chapter sections from the GUI
    l1_phase20_chapter_autowrite_helper_exists,
    l1_phase20_web_endpoint_and_router,
    l1_phase20_broken_citation_indicator,
    # Phase 21 — MinerU 2.5-Pro + sections UX + plan modal context
    l1_phase21_mineru_pro_helper,
    l1_phase21_sidebar_renders_template_slots,
    l1_phase21_section_editor_live_slug_and_budget,
    l1_phase21_plan_modal_context_aware,
    # Phase 22 — XSS fixes + job cleanup + GUI improvements
    l1_phase22_render_helpers_escape_html,
    l1_phase22_job_cleanup,
    l1_phase22_delete_draft_endpoint_and_orphan_cleanup,
    l1_phase22_chapter_progress_and_word_target,
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
