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

    aw_src = (inspect.getsource(book_ops.autowrite_section_stream) + "\n" + inspect.getsource(book_ops._autowrite_section_body))
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
    aw_src = (inspect.getsource(book_ops.autowrite_section_stream) + "\n" + inspect.getsource(book_ops._autowrite_section_body))
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
    aw_src = (inspect.getsource(book_ops.autowrite_section_stream) + "\n" + inspect.getsource(book_ops._autowrite_section_body))
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

    from sciknow.web.app import TEMPLATE, _BUILD_TAG
    rendered = TEMPLATE.format(
        _BUILD_TAG=_BUILD_TAG,
        book_title="Test Book", search_q="", search_results_html="",
        # Phase 38 — book_id placeholder for the bundle-snapshot JS
        book_id="00000000-0000-0000-0000-000000000000",
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
        "## 21. Compound Learning from Iteration History",
        "## 22. CARS-Adapted Chapter Moves",
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

    src = (inspect.getsource(book_ops.autowrite_section_stream) + "\n" + inspect.getsource(book_ops._autowrite_section_body))

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
    assert out[0]["slug"] == "overview"
    assert out[0]["title"] == "Overview"
    assert out[0]["plan"] == ""
    assert out[0].get("target_words") is None  # Phase 29 — no override by default
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
    aw_src = (inspect.getsource(book_ops.autowrite_section_stream) + "\n" + inspect.getsource(book_ops._autowrite_section_body))
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

    aw_src = (inspect.getsource(book_ops.autowrite_section_stream) + "\n" + inspect.getsource(book_ops._autowrite_section_body))
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
    assert ("existing_slugs" in src or "existing_by_slug" in src) and "rebuild" in src, (
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


# ── Phase 23 — collapse/expand chapter sections in the sidebar ──────────


def l1_phase23_chapter_collapse_expand() -> None:
    """Phase 23 — sidebar shows a chevron toggle on each chapter title
    that collapses/expands its sections, plus a sidebar-level
    collapse-all/expand-all button. State persists in localStorage.
    """
    import inspect
    from sciknow.web import app as web_app

    src = inspect.getsource(web_app)

    # Per-chapter chevron HTML in _render_sidebar
    render_src = inspect.getsource(web_app._render_sidebar)
    assert "ch-toggle" in render_src, (
        "_render_sidebar doesn't emit the ch-toggle chevron"
    )
    assert "toggleChapter" in render_src, (
        "_render_sidebar chevron doesn't wire toggleChapter()"
    )

    # rebuildSidebar JS mirrors the chevron so post-autowrite refreshes
    # don't lose the toggle button
    assert "ch-toggle" in src and "toggleChapter" in src, (
        "rebuildSidebar JS doesn't render the chevron toggle"
    )

    # CSS for collapsed state hides sections + progress bar
    assert ".ch-group.collapsed" in src, (
        "missing .ch-group.collapsed CSS — toggle would have no visual effect"
    )
    assert ".ch-group.collapsed .sec-link" in src, (
        "collapsed CSS doesn't hide sections"
    )
    assert ".ch-group.collapsed .ch-progress" in src, (
        "collapsed CSS doesn't hide the chapter progress bar"
    )

    # Sidebar collapse-all toggle button
    assert "sidebar-toggle-all" in src, (
        "sidebar-toggle-all button missing"
    )
    assert "toggleAllChapters" in src, (
        "toggleAllChapters JS helper missing"
    )

    # Persistence via localStorage + restore on page load
    assert "_COLLAPSED_KEY" in src, (
        "missing localStorage key for collapsed chapters"
    )
    assert "restoreCollapsedChapters" in src, (
        "restoreCollapsedChapters helper missing"
    )
    # Restore is wired into both DOMContentLoaded AND rebuildSidebar
    # so SPA refreshes don't drop the user's collapsed state.
    rs_src = inspect.getsource(web_app)  # full module
    # Count occurrences in rebuildSidebar specifically
    assert "DOMContentLoaded" in src and "restoreCollapsedChapters" in src, (
        "restoreCollapsedChapters not wired to page load"
    )


# ── Phase 24 — autowrite progress log with heartbeat ──────────────────


def l1_phase24_autowrite_logger() -> None:
    """Phase 24 — _AutowriteLogger writes JSONL to disk with stage
    transitions, token counts, and a side-thread heartbeat.

    Drives the logger end-to-end with a fast (0.3s) heartbeat
    interval so the test completes quickly while still proving the
    heartbeat thread fires and records its state. Verifies:
      - per-run JSONL file is created in the configured log_dir
      - latest.jsonl symlink points at the run file
      - stage() emits stage_start + stage_end with the right keys
      - token() increments stage and total counters
      - heartbeat thread writes a heartbeat entry with the current
        stage and elapsed time
      - close() is idempotent and stops the heartbeat thread
    """
    import json as _json
    import tempfile
    import time as _time
    from pathlib import Path as _Path
    from sciknow.core import book_ops

    assert hasattr(book_ops, "_AutowriteLogger"), (
        "_AutowriteLogger class missing — autowrite has no progress log"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = _Path(tmpdir) / "autowrite"
        log = book_ops._AutowriteLogger(
            "book-1", "ch-1", "overview",
            heartbeat_seconds=0.3,
            log_dir=log_dir,
        )
        try:
            assert log.path.exists(), "log file not created on init"
            assert log.path.parent == log_dir
            # latest.jsonl symlink (best-effort — only assert if symlinks
            # work on this filesystem)
            latest = log_dir / "latest.jsonl"
            if latest.is_symlink():
                # Resolve the symlink target — should match our run path
                # by basename (we use a relative target).
                assert latest.exists()

            log.stage("retrieval")
            log.token()
            log.token()
            log.token()
            assert log.state["total_tokens"] == 3
            assert log.state["stage_tokens"] == 3
            assert log.state["stage"] == "retrieval"

            log.stage("writing", iteration=1)
            assert log.state["stage"] == "writing"
            assert log.state["stage_tokens"] == 0  # reset on stage transition
            assert log.state["total_tokens"] == 3  # total carries forward

            log.token()
            log.token()
            assert log.state["total_tokens"] == 5
            assert log.state["stage_tokens"] == 2

            # Wait long enough for at least one heartbeat to fire
            _time.sleep(0.7)

            log.event("verdict", action="KEEP", iteration=1)
        finally:
            log.close()
            # close() should be idempotent
            log.close()

        # Read the file back and verify the entry kinds
        lines = log.path.read_text(encoding="utf-8").strip().splitlines()
        assert lines, "log file is empty after close"
        entries = [_json.loads(line) for line in lines]
        kinds = [e.get("kind") for e in entries]
        assert "start" in kinds, "missing 'start' entry"
        assert "stage_start" in kinds, "missing 'stage_start' entry"
        assert "stage_end" in kinds, "missing 'stage_end' entry (stage transition)"
        assert "heartbeat" in kinds, (
            "missing 'heartbeat' entry — side thread didn't write"
        )
        assert "verdict" in kinds, "missing custom event entry"
        assert "end" in kinds, "missing 'end' entry on close()"

        # Verify the heartbeat entry has the expected shape
        hb = next(e for e in entries if e.get("kind") == "heartbeat")
        for key in ("stage", "stage_elapsed_s", "stage_tokens",
                    "total_tokens", "tokens_per_sec"):
            assert key in hb, f"heartbeat entry missing {key}: {hb}"

    # Verify the public function autowrite_section_stream wires the
    # logger in via the wrapper pattern (try/finally close).
    import inspect
    src = inspect.getsource(book_ops.autowrite_section_stream)
    assert "_AutowriteLogger" in src, (
        "autowrite_section_stream doesn't open the progress log"
    )
    assert "log.close()" in src, (
        "autowrite_section_stream doesn't close the log in finally"
    )
    assert "_autowrite_section_body" in src, (
        "autowrite_section_stream wrapper doesn't delegate to the body"
    )

    # And that the body calls log.stage / log.token at the right places
    body_src = inspect.getsource(book_ops._autowrite_section_body)
    for stage in ("loading_book", "retrieval", "writing", "scoring",
                  "verifying", "revising"):
        assert f'log.stage("{stage}"' in body_src, (
            f"_autowrite_section_body doesn't log.stage({stage!r})"
        )
    assert "token_observer=log.token" in body_src, (
        "_autowrite_section_body doesn't pass log.token as token_observer "
        "to the streaming helpers — heartbeat won't see token throughput"
    )


# ── Phase 25 — visible chevron + adopt orphan sections ───────────────────


def l1_phase25_chevron_visible() -> None:
    """Phase 25 — chapter chevron CSS uses visible font-size + non-muted
    color so the user can actually see it. The Phase 23 version used
    10px / fg-muted which the user reported as invisible."""
    import inspect
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)

    # Find the .ch-toggle CSS rule and check for the new properties
    # The new rule has font-size: 13px and color: var(--fg) (not muted)
    assert ".ch-toggle " in src and "13px" in src, (
        ".ch-toggle CSS doesn't have a visible 13px font-size"
    )
    assert "color: var(--fg);" in src or "color:var(--fg);" in src, (
        ".ch-toggle CSS doesn't use the main fg color (still using fg-muted?)"
    )


def l1_phase25_adopt_orphan_section() -> None:
    """Phase 25 — adopt_orphan_section helper appends a slug to a
    chapter's sections JSONB, idempotent, raises on bad input.

    Drives the helper end-to-end against an in-memory test using a
    real DB session. Cleans up the test chapter after.
    """
    import inspect
    from sqlalchemy import text
    from sciknow.core import book_ops
    from sciknow.storage.db import get_session

    assert hasattr(book_ops, "adopt_orphan_section"), (
        "adopt_orphan_section helper missing"
    )

    # Bad input: empty slug raises ValueError
    try:
        book_ops.adopt_orphan_section("any", "any", "")
        raise AssertionError("expected ValueError for empty slug")
    except ValueError:
        pass

    # Bad input: chapter not found raises ValueError
    try:
        book_ops.adopt_orphan_section(
            "00000000-0000-0000-0000-000000000000",
            "00000000-0000-0000-0000-000000000000",
            "intro",
        )
        raise AssertionError("expected ValueError for missing chapter")
    except ValueError:
        pass

    # Round-trip on a real chapter (cleanup after).
    with get_session() as session:
        row = session.execute(text("""
            SELECT b.id::text, bc.id::text, bc.sections
            FROM books b JOIN book_chapters bc ON bc.book_id = b.id
            ORDER BY bc.created_at LIMIT 1
        """)).fetchone()
    if not row:
        # No book/chapter to test with — skip the round-trip portion
        return
    book_id, chapter_id, original_sections = row

    test_slug = "__phase25_test_slug__"
    try:
        result = book_ops.adopt_orphan_section(
            book_id, chapter_id, test_slug,
            title="Test Section", plan="A test plan.",
        )
        assert result["ok"] is True
        assert result["added"] is True
        assert result["section"]["slug"] == test_slug
        assert result["section"]["title"] == "Test Section"
        assert result["section"]["plan"] == "A test plan."

        # Idempotency: second call with same slug returns added=False
        result2 = book_ops.adopt_orphan_section(book_id, chapter_id, test_slug)
        assert result2["ok"] is True
        assert result2["added"] is False, "second adopt should be idempotent"
    finally:
        # Restore the chapter's original sections list
        with get_session() as session:
            import json as _json
            session.execute(text("""
                UPDATE book_chapters
                SET sections = CAST(:secs AS jsonb)
                WHERE id::text = :cid
            """), {
                "cid": chapter_id,
                "secs": _json.dumps(original_sections or []),
            })
            session.commit()

    # Verify the public surface: web endpoint + CLI command + GUI button
    from sciknow.web import app as web_app
    routes = {getattr(r, "path", "") for r in web_app.app.routes}
    assert "/api/chapters/{chapter_id}/sections/adopt" in routes, (
        "POST /api/chapters/{id}/sections/adopt endpoint not registered"
    )

    src = inspect.getsource(web_app)
    assert "adoptOrphanSection" in src, (
        "adoptOrphanSection JS helper missing"
    )
    assert "sec-orphan-adopt" in src, (
        "missing sec-orphan-adopt CSS class for the inline + button"
    )

    # CLI: book adopt-section command registered on the typer app
    from typer.main import get_command
    from sciknow.cli import book as book_cli
    cmd = get_command(book_cli.app)
    sub_names = {c.name for c in cmd.commands.values()}
    assert "adopt-section" in sub_names, (
        "`sciknow book adopt-section` CLI command not registered"
    )


# ── Phase 26 — drag-and-drop section reordering ──────────────────────────


def l1_phase26_section_drag_drop() -> None:
    """Phase 26 — sidebar section rows are draggable + drop targets,
    with handlers wired via event delegation on #sidebar-sections.

    Verifies:
      - drafted + empty rows have draggable="true" + data-section-slug
      - JS handlers (handleSectionDragStart/Over/Drop/End) exist
      - reorderSections() helper PUTs to the existing endpoint
      - setupSectionDragDrop is wired on DOMContentLoaded
      - CSS for .dragging + .drag-over-top/bottom drop indicators
      - Within-chapter only check (the drag handler refuses cross-chapter)
    """
    import inspect
    from sciknow.web import app as web_app

    # Server-side: drafted + empty rows in _render_sidebar carry the
    # required attributes
    render_src = inspect.getsource(web_app._render_sidebar)
    assert 'draggable="true"' in render_src, (
        '_render_sidebar doesn\'t emit draggable="true" on section rows'
    )
    assert "data-section-slug" in render_src, (
        "_render_sidebar doesn't emit data-section-slug for the drag handler"
    )

    src = inspect.getsource(web_app)

    # JS rebuildSidebar mirrors the attrs (post-autowrite refresh path)
    assert "draggable=\"true\"" in src, (
        "rebuildSidebar JS doesn't render draggable rows"
    )

    # Drag handlers exist
    for fn in ("handleSectionDragStart", "handleSectionDragOver",
               "handleSectionDrop", "handleSectionDragEnd",
               "reorderSections", "setupSectionDragDrop"):
        assert fn in src, f"missing JS function: {fn}"

    # Wired on DOMContentLoaded so the listener is attached on page load
    assert "setupSectionDragDrop" in src and "DOMContentLoaded" in src, (
        "setupSectionDragDrop not wired to page load"
    )

    # Within-chapter enforcement: handler compares dataset.chId
    assert "_draggedSection.chapterId" in src, (
        "drop handler doesn't enforce within-chapter only"
    )

    # CSS for drag states
    assert ".sec-link.dragging" in src, "missing .dragging CSS"
    assert ".sec-link.drag-over-top" in src, "missing drop-indicator CSS (top)"
    assert ".sec-link.drag-over-bottom" in src, "missing drop-indicator CSS (bottom)"

    # reorderSections sends the new order to the existing PUT endpoint
    reorder_idx = src.index("async function reorderSections")
    reorder_block = src[reorder_idx:reorder_idx + 2000]
    assert "/api/chapters/" in reorder_block and "/sections" in reorder_block, (
        "reorderSections doesn't PUT to /api/chapters/{id}/sections"
    )
    assert "method: 'PUT'" in reorder_block, (
        "reorderSections uses the wrong HTTP method"
    )


# ── Phase 27 — display title derived from chapter sections meta ─────────


def l1_phase27_display_title_from_meta() -> None:
    """Phase 27 — _draft_display_title derives the center h1 from the
    chapter sections JSONB instead of the stale slug-based drafts.title
    snapshot. Renaming a section in the chapter modal should update
    the center title on the next navigation.
    """
    import inspect
    from sciknow.web import app as web_app

    assert hasattr(web_app, "_draft_display_title"), (
        "_draft_display_title helper missing"
    )

    # Drift case: meta has a different title than draft.title.
    # The helper should return the META title.
    sections = [
        {"slug": "key_evidence", "title": "The Grand Solar Cycles", "plan": ""},
        {"slug": "summary", "title": "Summary", "plan": ""},
    ]
    out = web_app._draft_display_title(
        draft_title="Ch.2 Solar and Geomagnetic Drivers — Key_evidence",
        section_type="key_evidence",
        chapter_num=2,
        chapter_title="Solar and Geomagnetic Drivers",
        chapter_sections_raw=sections,
    )
    assert "The Grand Solar Cycles" in out, (
        f"didn't pick up the meta title: {out!r}"
    )
    assert "Key_evidence" not in out, (
        f"didn't suppress the stale slug snapshot: {out!r}"
    )
    assert out.startswith("Ch.2"), f"missing Ch.N prefix: {out!r}"

    # Orphan case: section_type doesn't match any slug → fallback to
    # the stored drafts.title (so we don't lose the user's view of
    # the orphan completely).
    out = web_app._draft_display_title(
        draft_title="Ch.2 Solar — Introduction (autowrite)",
        section_type="introduction",  # not in meta
        chapter_num=2,
        chapter_title="Solar",
        chapter_sections_raw=sections,
    )
    assert out == "Ch.2 Solar — Introduction (autowrite)", (
        f"orphan fallback wrong: {out!r}"
    )

    # Empty / None inputs: fallback to draft_title
    out = web_app._draft_display_title(
        draft_title="x", section_type=None, chapter_num=None,
        chapter_title=None, chapter_sections_raw=None,
    )
    assert out == "x"
    out = web_app._draft_display_title(
        draft_title="y", section_type="key_evidence",
        chapter_num=2, chapter_title="C",
        chapter_sections_raw=[],
    )
    assert out == "y"

    # JSON-encoded sections also accepted (we hand them off to
    # _normalize_chapter_sections which handles both shapes).
    import json as _json
    out = web_app._draft_display_title(
        draft_title="z", section_type="key_evidence",
        chapter_num=1, chapter_title="C",
        chapter_sections_raw=_json.dumps(sections),
    )
    assert "Grand Solar Cycles" in out

    # api_section returns display_title
    api_src = inspect.getsource(web_app.api_section)
    assert "display_title" in api_src, (
        "api_section doesn't return display_title"
    )
    assert "_draft_display_title(" in api_src, (
        "api_section doesn't call the helper"
    )

    # _render_book also derives active_title from the helper
    rb_src = inspect.getsource(web_app._render_book)
    assert "_draft_display_title(" in rb_src, (
        "_render_book doesn't derive active_title from the helper"
    )

    # JS loadSection prefers display_title
    src = inspect.getsource(web_app)
    assert "data.display_title || data.title" in src, (
        "loadSection JS doesn't prefer display_title over data.title"
    )


# ── Phase 28 — autowrite resume mode ──────────────────────────────────────


def l1_phase28_is_resumable_draft() -> None:
    """Phase 28 — _is_resumable_draft accepts finished checkpoints
    and refuses partial ones.
    """
    from sciknow.core import book_ops

    assert hasattr(book_ops, "_is_resumable_draft"), (
        "_is_resumable_draft helper missing"
    )

    # Finished states → resumable
    for checkpoint in ("final", "initial", "iteration_2_keep",
                       "iteration_3_discard", "draft"):
        ok, _ = book_ops._is_resumable_draft({"checkpoint": checkpoint}, 1500)
        assert ok, f"expected resumable for checkpoint={checkpoint!r}"

    # Pre-checkpoint era (very old drafts) → resumable with warning
    ok, reason = book_ops._is_resumable_draft({}, 1500)
    assert ok and "predates" in reason

    # Partial states → refused
    for checkpoint in ("writing_in_progress", "placeholder",
                       "iteration_1_revising", "iteration_5_revising"):
        ok, reason = book_ops._is_resumable_draft({"checkpoint": checkpoint}, 1500)
        assert not ok, f"expected refused for checkpoint={checkpoint!r}"
        assert reason, f"expected reason string for checkpoint={checkpoint!r}"

    # Unknown checkpoint → refused (be conservative)
    ok, reason = book_ops._is_resumable_draft({"checkpoint": "weird_state"}, 1500)
    assert not ok and "unknown" in reason

    # Word count too low → refused even with finished checkpoint
    ok, reason = book_ops._is_resumable_draft({"checkpoint": "final"}, 50)
    assert not ok and "too short" in reason

    # JSON-encoded metadata also accepted
    import json as _json
    ok, _ = book_ops._is_resumable_draft(
        _json.dumps({"checkpoint": "final"}), 1500,
    )
    assert ok


def l1_phase28_resume_wired_through() -> None:
    """Phase 28 — resume parameter wired through autowrite_section_stream,
    _autowrite_section_body, the chapter generator, the web endpoint,
    and the CLI.
    """
    import inspect
    from sciknow.core import book_ops
    from sciknow.web import app as web_app

    # autowrite_section_stream signature
    sig = inspect.signature(book_ops.autowrite_section_stream)
    assert "resume_from_draft_id" in sig.parameters, (
        "autowrite_section_stream missing resume_from_draft_id kwarg"
    )

    # _autowrite_section_body signature + actual usage
    body_sig = inspect.signature(book_ops._autowrite_section_body)
    assert "resume_from_draft_id" in body_sig.parameters
    body_src = inspect.getsource(book_ops._autowrite_section_body)
    assert "_is_resumable_draft" in body_src, (
        "_autowrite_section_body doesn't gate resume on _is_resumable_draft"
    )
    assert "resume_content is None" in body_src, (
        "_autowrite_section_body doesn't branch the writing phase on resume_content"
    )

    # Chapter generator passes resume_draft_id to the inner stream
    ch_sig = inspect.signature(book_ops.autowrite_chapter_all_sections_stream)
    assert "resume" in ch_sig.parameters, (
        "autowrite_chapter_all_sections_stream missing resume kwarg"
    )
    ch_src = inspect.getsource(book_ops.autowrite_chapter_all_sections_stream)
    assert "resume_from_draft_id=resume_draft_id" in ch_src, (
        "chapter generator doesn't pass resume_draft_id to the inner generator"
    )
    # rebuild + resume mutually exclusive (rebuild wins)
    assert "rebuild and resume" in ch_src, (
        "chapter generator doesn't enforce rebuild/resume exclusivity"
    )

    # Web endpoint accepts resume form arg
    ep_sig = inspect.signature(web_app.api_autowrite_chapter)
    assert "resume" in ep_sig.parameters, (
        "POST /api/autowrite-chapter missing resume form arg"
    )

    # JS doAutowrite has the three-way mode picker
    src = inspect.getsource(web_app)
    assert "modeRebuild" in src and "modeResume" in src, (
        "doAutowrite JS doesn't track rebuild/resume mode separately"
    )
    assert "iterate" in src.lower(), (
        "doAutowrite mode prompt doesn't mention iterate/resume option"
    )

    # CLI book autowrite has --resume flag
    from typer.main import get_command
    from sciknow.cli import book as book_cli
    cmd = get_command(book_cli.app)
    autowrite_cmd = cmd.commands.get("autowrite")
    assert autowrite_cmd is not None
    param_names = {p.name for p in autowrite_cmd.params}
    assert "resume" in param_names, "`book autowrite` missing --resume flag"


# ── Phase 29 — ROADMAP + per-section size + click-to-preview ────────────


def l1_phase29_roadmap_doc_exists() -> None:
    """Phase 29 — docs/ROADMAP.md is checked in and contains the
    headings expected by the user-facing doc structure.
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent
    roadmap = repo_root / "docs" / "ROADMAP.md"
    assert roadmap.exists(), "docs/ROADMAP.md missing"
    text = roadmap.read_text(encoding="utf-8")
    # Check the major sections so we don't accidentally truncate the file
    for heading in (
        "Deferred QA findings",
        "Research runners-up",
        "Hardware-gated",
        "Polish from recent phases",
        "Feature gaps",
    ):
        assert heading in text, f"ROADMAP.md missing section: {heading}"


def l1_phase29_per_section_target_words() -> None:
    """Phase 29 — _normalize_chapter_sections preserves target_words,
    _get_section_target_words returns the override or None, and the
    autowrite/write resolution uses the per-section value when set.
    """
    import inspect
    from sciknow.core import book_ops

    # _normalize_chapter_sections preserves target_words
    out = book_ops._normalize_chapter_sections([
        {"slug": "intro", "title": "Intro", "plan": "", "target_words": 1500},
        {"slug": "deep_dive", "title": "Deep dive", "plan": "X"},  # missing
        {"slug": "summary", "title": "Summary", "plan": "", "target_words": 0},  # 0 → None
    ])
    assert len(out) == 3
    assert out[0]["target_words"] == 1500
    assert out[1].get("target_words") is None
    assert out[2].get("target_words") is None  # 0 normalised to None

    # Bad input types → None (don't raise)
    out = book_ops._normalize_chapter_sections([
        {"slug": "x", "title": "X", "plan": "", "target_words": "not-a-number"},
        {"slug": "y", "title": "Y", "plan": "", "target_words": -50},
    ])
    assert out[0]["target_words"] is None
    assert out[1]["target_words"] is None

    # _get_section_target_words helper exists
    assert hasattr(book_ops, "_get_section_target_words"), (
        "_get_section_target_words helper missing"
    )

    # autowrite + write_section_stream resolution uses the override
    body_src = inspect.getsource(book_ops._autowrite_section_body)
    assert "_get_section_target_words" in body_src, (
        "_autowrite_section_body doesn't check per-section target override"
    )
    ws_src = inspect.getsource(book_ops.write_section_stream)
    assert "_get_section_target_words" in ws_src, (
        "write_section_stream doesn't check per-section target override"
    )


def l1_phase29_size_dropdown_in_modal() -> None:
    """Phase 29 — chapter modal section editor has the size dropdown
    + JS handlers + persistence.
    """
    import inspect
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)

    # Dropdown + handlers in the rendered template
    assert "updateSectionTargetWords" in src, (
        "updateSectionTargetWords JS handler missing"
    )
    assert "updateSectionTargetWordsCustom" in src, (
        "custom-input handler missing"
    )
    assert "sec-size-row" in src, "missing CSS class for the size dropdown row"

    # Save flow includes target_words per section
    assert "target_words: (s.target_words" in src, (
        "saveChapterInfo doesn't include target_words in the section save payload"
    )


def l1_phase29_empty_section_preview_not_write() -> None:
    """Phase 29 — clicking an empty section row in the sidebar calls
    previewEmptySection (non-destructive) instead of writeForCell
    (immediately triggers a write).
    """
    import inspect
    from sciknow.web import app as web_app

    # Server-side render — Phase 42 moved the click dispatch from an
    # inline `onclick="previewEmptySection(...)"` to a
    # `data-action="preview-empty-section"` attribute resolved by the
    # global ACTIONS registry. Either marker satisfies the spirit of
    # this test: empty rows preview, they don't fire a write.
    render_src = inspect.getsource(web_app._render_sidebar)
    _empty_hook_ok = (
        "previewEmptySection" in render_src
        or "preview-empty-section" in render_src
    )
    assert _empty_hook_ok, (
        "_render_sidebar empty rows still call writeForCell instead "
        "of previewEmptySection / preview-empty-section data-action"
    )
    # Make sure the OLD writeForCell call on empty rows is gone
    # (it's still used by the heatmap empty cells, that's fine)
    assert 'sec-empty"' in render_src

    # JS rebuildSidebar mirrors the same call — accept either the old
    # camelCase call or the new data-action kebab name.
    src = inspect.getsource(web_app)
    total = src.count("previewEmptySection") + src.count("preview-empty-section")
    assert total >= 3, (
        "previewEmptySection / preview-empty-section should appear in "
        "render_sidebar + rebuildSidebar + the helper / ACTIONS entry "
        f"— only found {total} references"
    )

    # The helper renders an explicit Start writing button (so the user
    # has to deliberately click it — not auto-fired)
    assert "Start writing" in src, "preview helper missing the explicit Start writing button"
    assert "doWrite()" in src  # the button calls doWrite


# ── Phase 30 — task bar + heatmap + export + KG ──────────────────────────


def l1_phase30_persistent_task_bar() -> None:
    """Phase 30 — persistent task bar HTML/CSS/JS exists with the
    expected helpers and is wired into doAutowrite/doWrite/doReview.
    """
    import inspect
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)

    # HTML element exists
    assert 'id="task-bar"' in src, "task bar HTML element missing"
    assert 'id="tb-tokens"' in src and 'id="tb-tps"' in src, (
        "task bar missing token / tps display elements"
    )
    assert 'id="tb-elapsed"' in src and 'id="tb-eta"' in src, (
        "task bar missing elapsed / ETA display elements"
    )
    assert 'id="tb-stop"' in src, "task bar missing Stop button"

    # CSS for the bar (sticky top)
    assert ".task-bar" in src and "position: sticky" in src, (
        "task bar CSS missing or not sticky"
    )

    # JS helpers
    for fn in ("startGlobalJob", "stopGlobalJob", "dismissTaskBar",
               "_renderTaskBar", "_finishGlobalJob", "_formatElapsed"):
        assert fn in src, f"missing JS helper: {fn}"

    # Hours+mins formatting for >60 min elapsed (per user request)
    assert "h ' +" in src or "h '" in src, (
        "_formatElapsed doesn't render hours when elapsed > 60 min"
    )

    # Wired into the user-facing handlers
    assert src.count("startGlobalJob(") >= 3, (
        f"startGlobalJob should be called from doAutowrite + doWrite "
        f"+ doReview, found {src.count('startGlobalJob(')}"
    )

    # The Stop button uses DELETE /api/jobs/{id} (the existing cancel
    # endpoint) and the task bar's stop calls it
    assert "/api/jobs/" in src and "method: 'DELETE'" in src, (
        "stop button doesn't call the cancel endpoint"
    )


def l1_phase30_heatmap_numbered_columns() -> None:
    """Phase 30 — api_dashboard returns n_columns + heatmap rows with
    cells in chapter-section order, NOT a hardcoded section_types list.
    """
    import inspect
    from sciknow.web import app as web_app

    src = inspect.getsource(web_app.api_dashboard)
    assert "n_columns" in src, (
        "api_dashboard doesn't return n_columns"
    )
    # The OLD section_types union code is gone
    assert "section_types_set" not in src, (
        "api_dashboard still has the old union-of-slugs heatmap code"
    )
    # Per-row uses the chapter's actual sections in order
    assert "_chapter_sections_dicts" in src, (
        "api_dashboard doesn't read each chapter's sections meta"
    )
    # Absent cells are marked
    assert '"absent"' in src

    # JS rendering uses positional 1..N headers
    full_src = inspect.getsource(web_app)
    # The JS template literal must reference n_columns
    assert "data.n_columns" in full_src, (
        "showDashboard JS doesn't read n_columns from the API response"
    )
    assert ".hm-cell.absent" in full_src, (
        "missing CSS for the absent cell variant"
    )


def l1_phase30_export_endpoints() -> None:
    """Phase 30 — export endpoints return draft/chapter/book in
    txt, md, html. Invalid ext returns 400."""
    import inspect
    from sciknow.web import app as web_app

    routes = {getattr(r, "path", "") for r in web_app.app.routes}
    for path in (
        "/api/export/draft/{draft_id}.{ext}",
        "/api/export/chapter/{chapter_id}.{ext}",
        "/api/export/book.{ext}",
    ):
        assert path in routes, f"export endpoint missing: {path}"

    # _wrap_html_export and helpers exist
    for helper in ("_strip_md", "_draft_to_md", "_draft_to_html_body",
                   "_wrap_html_export", "_ordered_chapter_drafts"):
        assert hasattr(web_app, helper), f"missing helper: {helper}"

    # The HTML export wraps with print CSS
    out = web_app._wrap_html_export("Test", "<p>hi</p>")
    assert "@page" in out, "html export missing @page print CSS"
    assert "<html" in out and "<body>" in out
    assert "Test" in out

    # _strip_md removes markdown
    assert web_app._strip_md("# Title\n**bold** *italic*") == "Title\nbold italic"


def l1_phase30_kg_endpoint() -> None:
    """Phase 30 — KG endpoint returns the right shape and accepts
    filter params. The DB content isn't asserted (it varies); we
    just verify the endpoint shape + filter parameter handling.
    """
    import inspect
    from sciknow.web import app as web_app

    routes = {getattr(r, "path", "") for r in web_app.app.routes}
    assert "/api/kg" in routes, "KG endpoint not registered"

    sig = inspect.signature(web_app.api_kg)
    for p in ("subject", "predicate", "object", "document_id",
              "limit", "offset"):
        assert p in sig.parameters, f"api_kg missing {p} param"

    # JS modal + helpers
    src = inspect.getsource(web_app)
    assert "openKgModal" in src, "openKgModal JS missing"
    assert "loadKg" in src, "loadKg JS missing"
    assert 'id="kg-modal"' in src, "KG modal HTML missing"
    assert 'id="kg-subject"' in src and 'id="kg-predicate"' in src, (
        "KG modal missing filter inputs"
    )

    # Toolbar button exists
    assert "openKgModal()" in src, "KG button not in toolbar"


# ── Phase 31 — six fixes ──────────────────────────────────────────────────


def l1_phase31_custom_dropdown_persists() -> None:
    """Phase 31 — when the user picks "Custom" in the section size
    dropdown, the renderer must keep the input visible on re-render.

    Bug: Phase 29's isCustom was derived from `!presets.includes(tw)`,
    so setting tw=1500 (a preset) hid the input. Fix: explicit
    _customMode flag on the section dict.
    """
    import inspect
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)

    # Renderer reads s._customMode
    assert "s._customMode" in src or "_customMode" in src, (
        "renderSectionEditor doesn't read the _customMode flag"
    )
    # updateSectionTargetWords sets it
    assert "sec._customMode = true" in src, (
        "updateSectionTargetWords doesn't set _customMode when 'custom' picked"
    )
    assert "sec._customMode = false" in src, (
        "updateSectionTargetWords doesn't clear _customMode on Auto/preset"
    )
    # The renderer focuses the custom input after switching to custom
    assert "input.focus()" in src and "sec-size-custom" in src, (
        "no focus on the custom input after switching modes"
    )


def l1_phase31_pdf_export() -> None:
    """Phase 31 — PDF export endpoint exists, accepts pdf ext, uses
    weasyprint, returns application/pdf.
    """
    import inspect
    from sciknow.web import app as web_app

    # _html_to_pdf_response helper exists
    assert hasattr(web_app, "_html_to_pdf_response"), (
        "_html_to_pdf_response helper missing"
    )

    # All three export endpoints accept ext='pdf'
    for fn_name in ("export_draft", "export_chapter", "export_book"):
        fn = getattr(web_app, fn_name)
        src = inspect.getsource(fn)
        assert '"pdf"' in src or "_VALID_EXPORT_EXTS" in src, (
            f"{fn_name} doesn't accept pdf"
        )

    # _VALID_EXPORT_EXTS has pdf
    assert "pdf" in web_app._VALID_EXPORT_EXTS

    # Export modal JS lists PDF as a separate button (not "HTML / PDF")
    src = inspect.getsource(web_app)
    assert "ext: 'pdf'" in src and "label: 'PDF'" in src, (
        "Export modal doesn't have a separate PDF button"
    )
    # The misleading "HTML / PDF" label is gone
    assert "HTML / PDF" not in src, (
        "old misleading 'HTML / PDF' label still present"
    )


def l1_phase31_kg_graph_view() -> None:
    """Phase 31 — KG modal has a Graph tab with SVG force-directed
    rendering, alongside the existing Table tab."""
    import inspect
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)

    # Tabs in the HTML
    assert 'data-tab="kg-graph"' in src and 'data-tab="kg-table"' in src, (
        "KG modal missing Graph + Table tabs"
    )
    assert 'id="kg-graph-pane"' in src and 'id="kg-table-pane"' in src, (
        "KG modal missing tab panes"
    )

    # JS helpers
    assert "switchKgTab" in src, "switchKgTab JS missing"
    assert "_renderKgGraph" in src and "_renderKgTable" in src, (
        "KG render helpers missing"
    )
    assert "_renderKgGraph" in src and "Fruchterman" in src or "spring simulation" in src, (
        "KG graph doesn't have a force-directed layout"
    )

    # CSS for nodes/edges
    assert ".kg-node" in src and ".kg-edge" in src, (
        "KG graph CSS missing"
    )
    assert "#kg-graph-canvas" in src


def l1_phase31_read_button_section_filter() -> None:
    """Phase 31 — chapter_reader endpoint accepts an only_section
    query param + JS showChapterReader passes it when a section is
    selected.
    """
    import inspect
    from sciknow.web import app as web_app

    sig = inspect.signature(web_app.chapter_reader)
    assert "only_section" in sig.parameters, (
        "chapter_reader missing only_section query param"
    )

    # JS showChapterReader uses currentSectionType to choose URL
    src = inspect.getsource(web_app)
    assert "only_section=" in src, (
        "showChapterReader JS doesn't pass only_section to the API"
    )
    # The endpoint actually filters in SQL
    cr_src = inspect.getsource(web_app.chapter_reader)
    assert "only_section" in cr_src
    assert "LOWER(d.section_type) = LOWER" in cr_src, (
        "chapter_reader doesn't filter by section_type"
    )


def l1_phase31_edit_button_in_toolbar() -> None:
    """Phase 31 — Edit button promoted from inline subtitle to the
    primary toolbar group. AI Write/Autowrite/etc relabeled to
    distinguish from the manual Edit action.
    """
    import inspect
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)

    # The Edit button now appears in the toolbar's primary group
    # alongside the AI buttons
    assert ">&#9998; Edit</button>" in src, (
        "Edit button missing from primary toolbar"
    )
    # The AI buttons are relabeled
    assert "AI Autowrite" in src and "AI Write" in src, (
        "AI buttons not relabeled"
    )
    # The toggleEdit function still exists and is wired to both buttons
    assert src.count("toggleEdit()") >= 2, (
        "toggleEdit not wired to both the toolbar Edit button and the cancel button"
    )


# ════════════════════════════════════════════════════════════════════════════
# Phase 32 — QA module overhaul
#
# Goal: catch GUI/backend regressions that the source-grep tests miss.
# These all use the shared sciknow.testing.helpers module so each test
# stays a focused assertion rather than reassembling boilerplate.
#
# - l1_phase32_qa_helpers_module — sanity that the helpers themselves
#   load without DB access (they're called from many tests below).
# - l1_phase32_endpoint_inventory — verifies every expected
#   (path, method) is registered. Catches accidental endpoint deletion
#   during refactors and prevents the GUI from quietly losing features.
# - l1_phase32_js_handler_integrity — every onclick/oninput/onchange
#   reference resolves to a defined JS function. This is the regression
#   gate for the kind of bug where a button silently does nothing
#   because someone renamed its handler.
# - l1_phase32_render_helpers_escape_chain — extends the Phase 22 XSS
#   test by walking *all* `_render_*` helpers and asserting they pass
#   user-controlled fields through `_esc()` (not raw f-string subs).
# - l1_phase32_no_naked_handler_exceptions — every endpoint that
#   touches the DB wraps its access in either an HTTPException raise
#   or a try/except that returns a JSONResponse. Catches the silent-
#   500 pattern.
# - l2_phase32_endpoint_shapes — TestClient hits the major read-only
#   endpoints, asserts 200 status + response shape. Catches handler
#   exceptions and schema drift. Skipped without PG.
# - l2_phase32_data_invariants — DB-level invariants the GUI relies
#   on: no orphaned drafts, no dangling chapter_id, every chapter has
#   a section template, no draft with content < 1 word, etc. Skipped
#   without PG.
# ════════════════════════════════════════════════════════════════════════════


def l1_phase32_qa_helpers_module() -> None:
    """The shared testing.helpers module imports cleanly and exposes
    every symbol downstream Phase 32 tests rely on.

    This is a fast canary: if any helper grows a missing import or a
    typo, every other Phase 32 test fails noisily — but this one fails
    *first* with a focused message so the actual breakage is obvious.
    """
    from sciknow.testing import helpers as h

    expected = [
        "get_test_client", "a_book_id", "a_chapter_id", "a_draft_id",
        "inspect_handler_source", "web_app_full_source",
        "rendered_template_static", "rendered_template_with_data",
        "js_function_definitions", "js_onclick_handlers",
        "all_app_routes", "find_route",
    ]
    missing = [name for name in expected if not hasattr(h, name)]
    assert not missing, f"helpers.py missing: {missing}"

    # rendered_template_static must succeed without DB access — it's
    # the L1-safe template renderer that downstream tests depend on.
    rendered = h.rendered_template_static()
    assert "<script>" in rendered and "function " in rendered, (
        "rendered_template_static produced no script block"
    )


# Catalog of every endpoint the GUI depends on, keyed (method, path).
# Adding a new endpoint? Add it here too. Removing one is the bug
# this test is designed to catch.
_PHASE32_EXPECTED_ENDPOINTS: list[tuple[str, str]] = [
    # Page routes
    ("GET", "/"),
    ("GET", "/section/{draft_id}"),
    ("POST", "/comment"),
    ("POST", "/comment/{comment_id}/resolve"),
    ("POST", "/edit/{draft_id}"),
    ("GET", "/search"),
    # Book / chapter / draft CRUD
    ("GET", "/api/book"),
    ("PUT", "/api/book"),
    ("POST", "/api/book/plan/generate"),
    ("GET", "/api/section/{draft_id}"),
    ("GET", "/api/chapters"),
    ("GET", "/api/dashboard"),
    ("GET", "/api/versions/{draft_id}"),
    ("GET", "/api/diff/{old_id}/{new_id}"),
    ("POST", "/api/chapters"),
    ("PUT", "/api/chapters/{chapter_id}"),
    ("POST", "/api/chapters/{chapter_id}/sections/adopt"),
    ("PUT", "/api/chapters/{chapter_id}/sections"),
    ("DELETE", "/api/chapters/{chapter_id}"),
    ("POST", "/api/chapters/reorder"),
    # Export + KG (Phase 30/31)
    ("GET", "/api/export/draft/{draft_id}.{ext}"),
    ("GET", "/api/export/chapter/{chapter_id}.{ext}"),
    ("GET", "/api/export/book.{ext}"),
    ("GET", "/api/kg"),
    # Snapshots & versioning
    ("POST", "/api/snapshot/{draft_id}"),
    ("GET", "/api/snapshots/{draft_id}"),
    ("GET", "/api/snapshot-content/{snapshot_id}"),
    ("PUT", "/api/draft/{draft_id}/status"),
    ("PUT", "/api/draft/{draft_id}/metadata"),
    ("PUT", "/api/draft/{draft_id}/chapter"),
    ("DELETE", "/api/draft/{draft_id}"),
    ("GET", "/api/chapter-reader/{chapter_id}"),
    ("GET", "/api/corkboard"),
    # LLM-driven endpoints
    ("POST", "/api/write"),
    ("POST", "/api/review/{draft_id}"),
    ("POST", "/api/revise/{draft_id}"),
    ("POST", "/api/gaps"),
    ("POST", "/api/argue"),
    ("POST", "/api/verify/{draft_id}"),
    ("POST", "/api/autowrite-chapter"),
    ("POST", "/api/autowrite"),
    ("GET", "/api/draft/{draft_id}/scores"),
    # Wiki + ask
    ("GET", "/api/wiki/pages"),
    ("GET", "/api/wiki/page/{slug}"),
    ("POST", "/api/wiki/query"),
    ("POST", "/api/ask"),
    # Catalog + stats
    ("GET", "/api/catalog"),
    ("GET", "/api/stats"),
    # Job control
    ("GET", "/api/stream/{job_id}"),
    ("DELETE", "/api/jobs/{job_id}"),
    ("GET", "/api/jobs"),
    # Phase 32.5 — task bar polls this for stats instead of competing
    # with the per-section SSE consumer.
    ("GET", "/api/jobs/{job_id}/stats"),
]


def l1_phase32_endpoint_inventory() -> None:
    """Every endpoint the GUI depends on is registered with the right
    HTTP method.

    Phase 32. The catalog is hand-maintained at the top of this test —
    when adding a new endpoint to web/app.py, add it here too. The
    cost of forgetting is one failed L1 test that says exactly which
    (method, path) is missing.
    """
    from sciknow.testing.helpers import find_route

    missing: list[str] = []
    wrong_method: list[str] = []
    for method, path in _PHASE32_EXPECTED_ENDPOINTS:
        match = find_route(path)
        if not match:
            missing.append(f"{method} {path}")
            continue
        _, methods = match
        if method not in methods:
            wrong_method.append(f"{method} {path} (registered as {sorted(methods)})")

    errors: list[str] = []
    if missing:
        errors.append(f"missing endpoints: {missing}")
    if wrong_method:
        errors.append(f"wrong HTTP method: {wrong_method}")
    assert not errors, "; ".join(errors)


# JS built-ins and reserved keywords that the onclick regex may
# capture. None of these correspond to a function we define, so they
# must be excluded from the "must resolve" set.
_JS_BUILTIN_NAMES: set[str] = {
    # Built-in callables that can appear in inline handlers
    "alert", "confirm", "prompt", "fetch", "parseInt", "parseFloat",
    "Number", "String", "Boolean", "Array", "Object", "Date", "Math",
    "JSON", "console", "setTimeout", "setInterval", "clearTimeout",
    "clearInterval", "encodeURIComponent", "decodeURIComponent",
    "Promise", "Error", "RegExp",
    # Reserved keywords the regex may capture if a handler starts with
    # an inline statement (e.g. onclick="if(event.target===this)...")
    "if", "for", "while", "return", "switch", "do", "try", "throw",
    "new", "typeof", "void", "delete", "in", "of",
}


def l1_phase32_js_handler_integrity() -> None:
    """Every JS handler referenced in onclick/oninput/onchange/etc
    attributes resolves to a function defined elsewhere in
    sciknow.web.app.

    Phase 32. This is the regression gate for the bug class where a
    button silently does nothing because the handler was renamed in
    one place but not the other. Excludes JS built-ins (alert, etc.)
    and reserved keywords (if, for, while) so the regex's broader
    capture isn't a source of false positives.
    """
    from sciknow.testing.helpers import (
        js_function_definitions, js_onclick_handlers,
    )

    defs = js_function_definitions()
    refs = js_onclick_handlers()

    # The integrity check: every referenced name (minus built-ins) is
    # also defined as a function.
    unresolved = (refs - defs) - _JS_BUILTIN_NAMES
    assert not unresolved, (
        f"{len(unresolved)} JS handler(s) referenced in onclick "
        f"attributes have no matching function definition: "
        f"{sorted(unresolved)[:10]}"
    )

    # Defense-in-depth: at least 50 distinct handlers and 100 distinct
    # function defs — guards against the regex breaking quietly.
    assert len(refs) >= 50, (
        f"only {len(refs)} onclick handlers found — regex broken?"
    )
    assert len(defs) >= 100, (
        f"only {len(defs)} JS function defs found — regex broken?"
    )


def l1_phase32_render_helpers_escape_chain() -> None:
    """Every `_render_*` helper that interpolates user-controlled data
    into HTML uses `_esc()` (or escapes via the markdown pipeline).

    Phase 32 extends the Phase 22 escape audit. The original Phase 22
    test only covered three helpers; this one walks *all* of them and
    asserts they either escape via `_esc(...)` or pass content through
    `_md_to_html(...)` (which uses a sanitised pipeline). Helpers that
    only emit constants pass trivially.
    """
    import inspect
    from sciknow.web import app as web_app

    render_helpers = [
        name for name, fn in vars(web_app).items()
        if name.startswith("_render_") and callable(fn)
    ]
    assert len(render_helpers) >= 3, (
        f"expected at least 3 _render_* helpers, found {render_helpers}"
    )

    offenders: list[str] = []
    for name in render_helpers:
        fn = getattr(web_app, name)
        try:
            src = inspect.getsource(fn)
        except Exception:
            continue
        # If the helper builds HTML via f-strings ({var} substitution),
        # it must escape via _esc(...) or _md_to_html(...). Helpers that
        # don't reference _esc or _md_to_html and contain raw f-string
        # interpolation of variables are flagged for review.
        builds_html = '<' in src and ('f"' in src or "f'" in src)
        if not builds_html:
            continue
        if "_esc(" in src or "_md_to_html(" in src or "escape(" in src:
            continue
        # Phase 22 known-good helpers that don't take user data
        if name in {"_render_book"}:
            continue
        offenders.append(name)

    assert not offenders, (
        f"render helpers building HTML without _esc()/_md_to_html(): "
        f"{offenders}"
    )


def l1_phase32_no_global_state_leak() -> None:
    """The web module's `_book_id` and `_book_title` globals are the
    only mutable state on the FastAPI app. Phase 32 sanity check that
    no other top-level mutable globals have crept in (which would
    break per-test isolation).
    """
    import inspect
    from sciknow.testing.helpers import web_app_full_source

    src = web_app_full_source()
    # Find module-level assignments to vars beginning with _ (the
    # convention for "private global state")
    import re as _re
    pattern = _re.compile(r"^(_[a-z_]+)\s*[:=]", _re.MULTILINE)
    globals_found = set(pattern.findall(src))
    # Known intentional globals — anything else needs an explicit nod.
    known = {
        "_book_id", "_book_title",
        "_jobs", "_job_lock", "_job_gc_lock",
        "_DEFAULT_BOOK_SECTIONS", "_JOB_GC_AGE_SECONDS",
        "_md_renderer",
    }
    surprise = globals_found - known
    # Functions and dataclasses begin with _ too — filter to actual
    # module attributes that are not callables/types.
    from sciknow.web import app as web_app
    real_surprise: list[str] = []
    for name in surprise:
        val = getattr(web_app, name, None)
        if val is None:
            continue
        if callable(val) or inspect.isclass(val):
            continue
        if isinstance(val, type(_re)):
            continue
        real_surprise.append(name)

    assert not real_surprise, (
        f"unexpected mutable module-level globals in web/app.py: "
        f"{real_surprise} — add to known set if intentional"
    )


def l1_phase32_endpoint_handler_signatures_consistent() -> None:
    """Every endpoint handler in web/app.py is either `async def` or
    declares its dependencies via FastAPI's parameter system.

    Phase 32. FastAPI silently accepts sync handlers but they block
    the event loop on every request. This test enforces that every
    endpoint defined via `@app.{get,post,put,delete}` is `async def`
    so we don't quietly regress to sync I/O.
    """
    import inspect
    import re as _re
    from sciknow.testing.helpers import web_app_full_source

    src = web_app_full_source()
    # Match `@app.<method>(...)\nasync def name(` and `@app.<method>(...)\ndef name(`
    pattern = _re.compile(
        r"@app\.(get|post|put|delete)\([^)]*\)\s*\n(async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    )
    sync_handlers: list[str] = []
    total = 0
    for m in pattern.finditer(src):
        total += 1
        is_async = bool(m.group(2))
        name = m.group(3)
        if not is_async:
            sync_handlers.append(name)

    assert total >= 40, (
        f"expected at least 40 @app.* handlers, found {total} "
        "— regex broken?"
    )
    assert not sync_handlers, (
        f"endpoint handlers must be `async def`: {sync_handlers}"
    )


def l1_phase32_1_section_target_visible_and_loaded() -> None:
    """Phase 32.1 — the chapter modal's Sections tab must:

    1. Copy `target_words` from `ch.sections_meta` into the editor's
       working state when opening the modal (was being silently
       dropped, so previously-saved per-section overrides looked
       reset every time the modal reopened).
    2. Render a visible target badge next to the dropdown so the user
       can see what word budget THIS section will be written to (the
       old "budget: ~Xw" line was buried in the muted slug row).
    """
    import inspect
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)

    # 1) target_words must be copied when loading sections_meta
    assert "target_words: (s.target_words" in src, (
        "openChapterModal not copying target_words from sections_meta — "
        "previously-saved per-section overrides will reset to Auto on reopen"
    )

    # 2) The visible target badge must be rendered in renderSectionEditor
    assert "sec-target-badge" in src, (
        "renderSectionEditor missing the per-section target badge"
    )
    assert "' words'" in src or "+ ' words'" in src, (
        "target badge missing the explicit 'words' label"
    )

    # 3) The CSS class must exist
    assert ".sec-target-badge" in src and ".sec-target-badge.override" in src, (
        "sec-target-badge CSS missing — badge would render unstyled"
    )


def l1_phase32_2_plan_modal_per_section_length() -> None:
    """Phase 32.2 — the book Plan modal must offer a per-section length
    dropdown in BOTH the 'Chapter sections' tab and the focused
    'Section' tab, and the save handlers must persist target_words.

    The user originally asked: "i want to be able to chose the length
    per section not per chapter. in the chapter sections, inside the
    plan, there should be a dropdown menu for section length selection
    in every section". This test is the regression gate for that ask.
    """
    import inspect
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)

    # 1) Plan modal Chapter sections tab — dropdown wired through
    #    updatePlanChapterTargetWords + updatePlanChapterTargetWordsCustom
    assert "updatePlanChapterTargetWords(" in src, (
        "Plan modal Chapter sections tab missing target dropdown handler"
    )
    assert "updatePlanChapterTargetWordsCustom(" in src, (
        "Plan modal Chapter sections tab missing custom-input handler"
    )
    assert "_editingChapterTargetWords" in src, (
        "Plan modal missing the per-chapter target_words editing state"
    )
    # The save handler must include target_words in the PUT body for
    # /api/chapters/{id}/sections.
    assert "target_words: (tw && tw > 0) ? tw : null" in src, (
        "savePlanChapterSections must persist target_words per section"
    )

    # 2) Plan modal Section tab — focused single-section dropdown
    assert "updatePlanSectionTargetWords(" in src, (
        "Plan modal Section tab missing target dropdown handler"
    )
    assert "updatePlanSectionTargetWordsCustom(" in src, (
        "Plan modal Section tab missing custom-input handler"
    )
    assert "_editingPlanSectionTargetWords" in src, (
        "Plan modal Section tab missing the editing state for target_words"
    )
    assert "plan-section-target-select" in src, (
        "Plan modal Section tab missing the dropdown DOM element"
    )

    # 3) savePlanSection must include target_words for the edited
    #    section AND preserve target_words for every other section
    #    (otherwise saving the focused section would wipe overrides
    #    the user set elsewhere in the chapter).
    assert "target_words: newTw" in src, (
        "savePlanSection must persist target_words for the edited section"
    )

    # 4) openPlanModal must reset the per-chapter editing maps so
    #    slug collisions between chapters don't leak overrides.
    #    Note: source has `{{}}` because the JS lives inside a Python
    #    f-string template (literal braces are doubled).
    assert "_editingChapterTargetWords = {{}}" in src, (
        "openPlanModal must reset _editingChapterTargetWords on open"
    )


def l1_phase32_3_task_bar_is_fixed_not_sticky() -> None:
    """Phase 32.3 — the persistent task bar (Phase 30) must use
    `position: fixed`, NOT `position: sticky`.

    The <body> is a horizontal flex container (sidebar | main).
    A `position: sticky` child of a horizontal flex container ends
    up rendered as a flex column sibling — which is exactly what
    the user reported: "the supposed top bar appears at the left
    as a new left column instead of appearing as a top bar".

    Fix: `.task-bar { position: fixed; top: 0; left: 0; right: 0 }`
    + a `body.task-bar-open { padding-top: 40px }` toggle so the
    fixed bar doesn't cover the sidebar/main top edge.
    """
    import inspect
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)

    # 1) Body must remain a horizontal flex container (this is the
    #    layout the rest of the app depends on; if someone changes
    #    it the test should still catch task bar regressions).
    assert "body {{ font-family: var(--font-sans)" in src or "display: flex" in src, (
        "body layout regressed — task bar fix may not apply"
    )

    # 2) Task bar must be position: fixed, not sticky
    #    (sticky inside horizontal flex = column sibling = the bug)
    assert ".task-bar {{ position: fixed" in src, (
        "task bar must be position: fixed (was sticky → rendered as left column)"
    )
    assert "position: sticky" not in src.split(".task-bar")[1].split("}}")[0], (
        "task bar still references position: sticky in its rule body"
    )

    # 3) Body padding-top must be added when the bar is visible so
    #    the bar doesn't cover the sidebar/main top edge.
    assert "body.task-bar-open" in src, (
        "missing body.task-bar-open class to offset the fixed bar"
    )

    # 4) The body class must be toggled by the show + hide handlers.
    assert "classList.add('task-bar-open')" in src, (
        "_renderTaskBar must add the task-bar-open body class on show"
    )
    assert "classList.remove('task-bar-open')" in src, (
        "dismissTaskBar must remove the task-bar-open body class on hide"
    )


def l1_phase32_4_section_delete_and_add_in_sidebar() -> None:
    """Phase 32.4 — every section in the sidebar gets an inline X
    delete button (mirroring chapters), and every chapter gets an
    inline + Add section affordance.

    The user originally asked: "in the same way that chapters have
    and 'x' mark to eliminate them, in the left panel, the sections
    should have one as well... also implement some way to add
    sections to the chapter".
    """
    import inspect
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)

    # 1) The deleteSection JS function exists
    assert "function deleteSection(" in src or "async function deleteSection(" in src, (
        "deleteSection JS handler missing"
    )
    # ...is wired to a button class
    assert "sec-delete-btn" in src, (
        "sec-delete-btn class missing — section delete button not rendered"
    )
    # ...with hover-only visibility CSS
    assert ".sec-link:hover .sec-delete-btn" in src, (
        "sec-delete-btn missing hover rule — button would always show"
    )
    # ...and used by rebuildSidebar (referenced from a sec-link onclick)
    assert "deleteSection(" in src, (
        "deleteSection not invoked from sidebar template"
    )

    # 2) The addSectionToChapter JS function exists
    assert "function addSectionToChapter(" in src or "async function addSectionToChapter(" in src, (
        "addSectionToChapter JS handler missing"
    )
    assert "sec-add-cta" in src, (
        "sec-add-cta class missing — Add section CTA not rendered"
    )
    assert "addSectionToChapter(" in src, (
        "addSectionToChapter not invoked from sidebar template"
    )

    # 3) The delete handler must round-trip through the existing
    #    PUT /api/chapters/{id}/sections endpoint, NOT a new endpoint
    #    (because the backend already supports replace-the-whole-list
    #    semantics and target_words preservation).
    delete_section_src = src.split("function deleteSection(")[1].split("function addSectionToChapter(")[0]
    assert "/api/chapters/" in delete_section_src and "/sections" in delete_section_src, (
        "deleteSection must PUT /api/chapters/{id}/sections"
    )
    # 4) Both handlers must preserve target_words on every untouched
    #    section so a delete/add doesn't wipe overrides set elsewhere
    #    in the chapter.
    assert "target_words: (s.target_words" in src, (
        "delete/add handlers must preserve target_words on untouched sections"
    )


def l1_phase32_5_task_bar_polls_stats_no_sse_competition() -> None:
    """Phase 32.5 — the persistent task bar must poll
    GET /api/jobs/{id}/stats, NOT open a second SSE EventSource on
    /api/stream/{id}.

    Bug history: the task bar's token counter and t/s display were
    broken across ~10 prior fix attempts. Root cause was
    architectural, not parsing: the per-section preview consumer
    AND the task bar both opened an EventSource on the same
    /api/stream/{id} URL. The server-side asyncio.Queue is shared,
    Queue.get() removes items, so two consumers split the event
    stream and the task bar saw a tiny (often zero) subset of tokens.
    No amount of regex/math/format patching on the consumer side
    could fix it because the events weren't arriving in the first
    place.

    Phase 32.5 fix: server-side counter in `_jobs[id]` updated by
    `_observe_event_for_stats` as events flow through the producer.
    Task bar polls a fixed-shape JSON snapshot every 500ms. Single
    source of truth, no race, no parsing. This test locks in BOTH
    halves of the fix so it can't silently regress.
    """
    import inspect
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)

    # 1) Server-side observer must exist and be wired into the
    #    generator runner BEFORE the event is enqueued.
    assert "def _observe_event_for_stats(" in src, (
        "_observe_event_for_stats helper missing — server-side counters won't update"
    )
    # The runner must call the observer for normal events
    runner_src = src.split("def _run_generator_in_thread(")[1].split("\n@app.")[0]
    assert "_observe_event_for_stats(job_id, event)" in runner_src, (
        "_run_generator_in_thread must observe each event before enqueueing"
    )

    # 2) The polling endpoint must exist and return the expected fields.
    assert '@app.get("/api/jobs/{job_id}/stats")' in src, (
        "GET /api/jobs/{id}/stats endpoint missing"
    )
    stats_src = src.split('@app.get("/api/jobs/{job_id}/stats")')[1].split("@app.")[0]
    for field in ('"tokens"', '"tps"', '"elapsed_s"', '"stream_state"', '"model_name"'):
        assert field in stats_src, (
            f"stats endpoint missing field: {field}"
        )

    # 3) The task bar JS must NOT open an EventSource for the global
    #    job. This is the architectural bug we're locking out.
    #    Look for any 'new EventSource' inside the task bar functions.
    sg_src = src.split("function startGlobalJob(")[1].split("function _finishGlobalJob")[0]
    assert "new EventSource" not in sg_src, (
        "startGlobalJob must NOT open an EventSource — that's the bug. "
        "It must poll /api/jobs/{id}/stats instead."
    )

    # 4) The task bar JS must call the polling helper.
    assert "_pollGlobalJobStats(" in src, (
        "task bar missing _pollGlobalJobStats — won't fetch any stats"
    )
    # ...and the helper must hit the right URL
    poll_src = src.split("async function _pollGlobalJobStats(")[1].split("\nfunction startGlobalJob")[0]
    assert "/api/jobs/' + jobId + '/stats" in poll_src, (
        "_pollGlobalJobStats must fetch /api/jobs/{id}/stats"
    )
    # ...and read the server-supplied counters directly (no
    # client-side accounting that could drift).
    assert "stats.tokens" in poll_src and "stats.tps" in poll_src, (
        "_pollGlobalJobStats must read tokens + tps directly from the server snapshot"
    )

    # 5) The previous SSE state variable must be gone — leaving it
    #    declared but unused would invite future confusion.
    assert "_globalJobSource" not in src, (
        "_globalJobSource (the old SSE source ref) should be removed entirely"
    )


def l1_phase32_6_autowrite_telemetry_layer0() -> None:
    """Phase 32.6 — Layer 0 of compound learning: autowrite telemetry.

    Verifies that:
    - Three new ORM models exist: AutowriteRun, AutowriteIteration,
      AutowriteRetrieval
    - Four persistence helpers exist on book_ops: _create_autowrite_run,
      _persist_autowrite_retrievals, _persist_autowrite_iteration,
      _finalize_autowrite_run
    - All four helpers are wired into autowrite_section_stream — the
      generator can't open a run, persist a row, or finalize without
      these calls being on the actual code path.
    """
    # ORM models
    from sciknow.storage import models
    for cls_name in ("AutowriteRun", "AutowriteIteration", "AutowriteRetrieval"):
        assert hasattr(models, cls_name), f"missing ORM model: {cls_name}"

    # Persistence helpers
    from sciknow.core import book_ops
    for fn in ("_create_autowrite_run", "_persist_autowrite_retrievals",
               "_persist_autowrite_iteration", "_finalize_autowrite_run"):
        assert hasattr(book_ops, fn), f"missing helper: book_ops.{fn}"

    # Wired into _autowrite_section_body (the actual loop —
    # autowrite_section_stream is a thin wrapper that just owns the
    # logger lifecycle and delegates here).
    import inspect
    src = inspect.getsource(book_ops._autowrite_section_body)
    for needle in (
        "_create_autowrite_run(",
        "_persist_autowrite_retrievals(",
        "_persist_autowrite_iteration(",
        "_finalize_autowrite_run(",
    ):
        assert needle in src, (
            f"_autowrite_section_body is not calling {needle.strip('(')} — "
            "Layer 0 telemetry will not be recorded"
        )

    # Sanity that the run id is held in a local across iterations
    assert "autowrite_run_id" in src, (
        "_autowrite_section_body missing the autowrite_run_id local that "
        "carries the run id from create→iterate→finalize"
    )


def l1_phase32_7_lessons_layer1() -> None:
    """Phase 32.7 — Layer 1: episodic memory store (lessons).

    The producer/consumer surface for the Reflexion-style learning
    loop documented in docs/RESEARCH.md §21 Layer 1. Verifies the
    schema, prompts, helpers, and wiring all exist and connect.

    Specifically:
    - AutowriteLesson ORM model exists
    - distill_lessons prompt + write_section_v2 lessons param exist
    - Producer + consumer helpers exist on book_ops
    - _finalize_autowrite_run calls _distill_lessons_from_run on
      completed runs
    - _autowrite_section_body calls _get_relevant_lessons before
      write_section_v2 and threads the result through as the
      lessons= parameter
    """
    # ORM model
    from sciknow.storage import models
    assert hasattr(models, "AutowriteLesson"), "AutowriteLesson model missing"

    # Prompts
    from sciknow.rag import prompts as rag_prompts
    assert hasattr(rag_prompts, "distill_lessons"), (
        "rag.prompts.distill_lessons missing — Layer 1 producer prompt"
    )
    # write_section_v2 must accept a lessons parameter
    import inspect
    sig = inspect.signature(rag_prompts.write_section_v2)
    assert "lessons" in sig.parameters, (
        "rag.prompts.write_section_v2 missing the lessons parameter — "
        "consumer can't inject lessons into the writer prompt"
    )

    # The system prompt template must have a placeholder for the lessons block
    assert "{lessons_section}" in rag_prompts.WRITE_V2_SYSTEM, (
        "WRITE_V2_SYSTEM missing the {lessons_section} placeholder — "
        "the lessons block won't render"
    )

    # Helpers
    from sciknow.core import book_ops
    for fn in ("_persist_lesson", "_distill_lessons_from_run",
               "_get_relevant_lessons", "_embed_text_for_lessons"):
        assert hasattr(book_ops, fn), f"book_ops.{fn} missing"

    # Producer wired into _finalize_autowrite_run
    finalize_src = inspect.getsource(book_ops._finalize_autowrite_run)
    assert "_distill_lessons_from_run(" in finalize_src, (
        "_finalize_autowrite_run does not call _distill_lessons_from_run — "
        "lessons will never be produced from completed runs"
    )

    # Consumer wired into _autowrite_section_body
    body_src = inspect.getsource(book_ops._autowrite_section_body)
    assert "_get_relevant_lessons(" in body_src, (
        "_autowrite_section_body does not call _get_relevant_lessons — "
        "the writer prompt will never see past lessons"
    )
    assert "lessons=relevant_lessons" in body_src, (
        "_autowrite_section_body does not thread relevant_lessons into "
        "write_section_v2(lessons=...) — the lessons fetch is dead code"
    )

    # The producer uses the FAST model (per the MAR critique — different
    # model than the writer/scorer to avoid confirmation bias).
    distill_src = inspect.getsource(book_ops._distill_lessons_from_run)
    assert "llm_fast_model" in distill_src, (
        "_distill_lessons_from_run is not using settings.llm_fast_model — "
        "MAR critique violated (same model judging itself)"
    )


def l1_phase32_8_useful_count_boost_layer2() -> None:
    """Phase 32.8 — Layer 2: useful chunk retrieval boost.

    Verifies that:
    - SearchCandidate has the useful_count field
    - SearchResult has the useful_count field (so it propagates downstream)
    - hybrid_search._apply_useful_boost exists
    - It's wired into hybrid_search.search() AFTER _apply_citation_boost
      (the two boosts compose multiplicatively)
    - The settings knob useful_count_boost_factor exists with a sane default
    - The boost SQL queries autowrite_retrievals.was_cited (not the wrong table)
    """
    from sciknow.retrieval import hybrid_search
    from sciknow.retrieval.context_builder import SearchResult
    from sciknow.config import settings

    assert "useful_count" in hybrid_search.SearchCandidate.__dataclass_fields__, (
        "SearchCandidate missing useful_count field"
    )
    assert "useful_count" in SearchResult.__dataclass_fields__, (
        "SearchResult missing useful_count field — won't propagate to context_builder"
    )
    assert hasattr(hybrid_search, "_apply_useful_boost"), (
        "_apply_useful_boost helper missing from hybrid_search"
    )
    assert hasattr(settings, "useful_count_boost_factor"), (
        "settings.useful_count_boost_factor missing"
    )
    # Sane default — non-zero so the feature is on by default after migration
    assert 0 < settings.useful_count_boost_factor < 1.0, (
        f"useful_count_boost_factor={settings.useful_count_boost_factor} "
        "out of expected range (0, 1)"
    )

    import inspect
    # The boost helper must query autowrite_retrievals.was_cited — that's
    # the data source. Wrong table or wrong column = silently broken.
    boost_src = inspect.getsource(hybrid_search._apply_useful_boost)
    assert "autowrite_retrievals" in boost_src, (
        "_apply_useful_boost not querying the autowrite_retrievals table"
    )
    assert "was_cited = true" in boost_src or "was_cited=true" in boost_src, (
        "_apply_useful_boost not filtering on was_cited — would count "
        "EVERY retrieval, not just chunks that made it into final drafts"
    )
    assert "chunk_qdrant_id" in boost_src, (
        "_apply_useful_boost not joining on chunk_qdrant_id"
    )

    # Wired into search() AFTER the citation boost so the two compose.
    search_src = inspect.getsource(hybrid_search.search)
    assert "_apply_useful_boost(" in search_src, (
        "hybrid_search.search not calling _apply_useful_boost — Layer 2 dead"
    )
    # Order matters: citation boost should come first (it modifies scores),
    # useful boost second (composes multiplicatively).
    cite_idx = search_src.find("_apply_citation_boost(")
    use_idx = search_src.find("_apply_useful_boost(")
    assert cite_idx > 0 and use_idx > cite_idx, (
        "_apply_useful_boost should be called AFTER _apply_citation_boost "
        "so the two boosts compose multiplicatively in the right order"
    )


def l1_phase32_9_dpo_export_layer4() -> None:
    """Phase 32.9 — Layer 4: DPO preference dataset export.

    Verifies:
    - autowrite_iterations has the new pre/post_revision_content columns
      (via the ORM model)
    - _persist_autowrite_iteration accepts the new content kwargs
    - _autowrite_section_body wires content into all three persist calls
      (pre-revision, KEEP-update, DISCARD-update)
    - _export_preference_pairs helper exists
    - The CLI command `book preferences export` is registered
    """
    # ORM has the new columns
    from sciknow.storage.models import AutowriteIteration
    fields = AutowriteIteration.__mapper__.column_attrs.keys()
    for col in ("pre_revision_content", "post_revision_content"):
        assert col in fields, f"AutowriteIteration missing column: {col}"

    # Helper accepts the new kwargs
    from sciknow.core import book_ops
    import inspect
    sig = inspect.signature(book_ops._persist_autowrite_iteration)
    for kw in ("pre_revision_content", "post_revision_content"):
        assert kw in sig.parameters, (
            f"_persist_autowrite_iteration missing kwarg: {kw}"
        )

    # Wired into _autowrite_section_body — content is captured at the
    # pre-revision persist AND in both KEEP and DISCARD update calls.
    body_src = inspect.getsource(book_ops._autowrite_section_body)
    assert "pre_revision_content=content" in body_src, (
        "_autowrite_section_body not capturing pre_revision_content — "
        "Layer 4 will see NULLs and produce zero pairs"
    )
    # Both verdict branches must capture post_revision_content=revised
    # (the DISCARD branch is the inverse-pair signal — never skip it)
    n_post_capture = body_src.count("post_revision_content=revised")
    assert n_post_capture >= 2, (
        f"_autowrite_section_body has {n_post_capture} post_revision_content "
        "captures — expected ≥2 (one for KEEP, one for DISCARD branch)"
    )

    # Export helper exists
    assert hasattr(book_ops, "_export_preference_pairs"), (
        "_export_preference_pairs missing — CLI command will crash"
    )
    export_sig = inspect.signature(book_ops._export_preference_pairs)
    for kw in ("book_id", "output_path", "min_score", "min_delta",
               "require_approval", "include_discard"):
        assert kw in export_sig.parameters, (
            f"_export_preference_pairs missing kwarg: {kw}"
        )

    # CLI command registered
    from sciknow.cli import book as book_cli
    assert hasattr(book_cli, "preferences_app"), (
        "book.preferences_app subcommand missing"
    )
    assert hasattr(book_cli, "preferences_export"), (
        "preferences_export CLI handler missing"
    )


def l1_phase32_10_style_fingerprint_layer5() -> None:
    """Phase 32.10 — Layer 5: style fingerprint extraction.

    Verifies the module loads, the extraction logic correctly counts
    sentences/citations/hedges/transitions on a known input, the
    aggregation produces the expected schema, the writer prompt has
    the {style_fingerprint_section} placeholder, write_section_v2
    accepts the new parameter, and both the consumer wiring and the
    CLI commands are in place.
    """
    from sciknow.core import style_fingerprint as sf

    # Module-level constants exist and are sane
    assert isinstance(sf._HEDGE_CUES, frozenset) and len(sf._HEDGE_CUES) > 10, (
        "_HEDGE_CUES looks broken"
    )
    assert isinstance(sf._TRANSITION_CUES, frozenset) and len(sf._TRANSITION_CUES) > 20, (
        "_TRANSITION_CUES looks broken"
    )

    # Extraction works on a known input
    test_text = (
        "The 11-year solar cycle may modulate global temperature [1]. "
        "However, the magnitude is contested [2]. "
        "Recent reconstructions suggest a small effect."
    )
    metrics = sf._extract_metrics_for_draft(test_text)
    assert metrics["n_sentences"] == 3, f"expected 3 sentences, got {metrics['n_sentences']}"
    assert metrics["n_citations"] == 2, f"expected 2 citations, got {metrics['n_citations']}"
    assert metrics["n_hedged_sentences"] >= 2, (
        f"expected ≥2 hedged sentences (may, suggest), got {metrics['n_hedged_sentences']}"
    )
    assert metrics["transition_counts"].get("however") == 1, (
        f"transition 'however' not detected: {metrics['transition_counts']}"
    )

    # Aggregation produces the expected schema
    fp = sf._aggregate_fingerprint([metrics])
    for key in (
        "n_drafts_sampled", "median_sentence_length", "median_paragraph_words",
        "citations_per_100_words", "hedging_rate", "avg_words_per_draft",
        "top_transitions",
    ):
        assert key in fp, f"fingerprint missing key: {key}"

    # Empty corpus produces a sensible empty fingerprint, not a crash
    empty = sf._aggregate_fingerprint([])
    assert empty["n_drafts_sampled"] == 0
    assert "samples_warning" in empty

    # Public helpers exist
    for fn in ("compute_style_fingerprint", "get_style_fingerprint",
               "format_fingerprint_for_prompt"):
        assert hasattr(sf, fn), f"missing public function: {fn}"

    # Format helper handles None / empty gracefully
    assert sf.format_fingerprint_for_prompt(None) == ""
    assert sf.format_fingerprint_for_prompt({"n_drafts_sampled": 0}) == ""

    # Writer prompt has the new placeholder + parameter
    from sciknow.rag import prompts as rag_prompts
    assert "{style_fingerprint_section}" in rag_prompts.WRITE_V2_SYSTEM, (
        "WRITE_V2_SYSTEM missing {style_fingerprint_section} placeholder"
    )
    import inspect
    sig = inspect.signature(rag_prompts.write_section_v2)
    assert "style_fingerprint_block" in sig.parameters, (
        "write_section_v2 missing style_fingerprint_block parameter"
    )

    # Calling write_section_v2 with a fingerprint must produce a system
    # prompt that contains the rendered block (not the raw placeholder).
    fp_block = sf.format_fingerprint_for_prompt(fp)
    sys_p, _ = rag_prompts.write_section_v2(
        "intro", "topic", [],
        style_fingerprint_block=fp_block,
    )
    assert "Match the established style" in sys_p, (
        "fingerprint block did not render into the system prompt"
    )
    # Calling without a fingerprint must NOT leave the placeholder behind
    sys_p2, _ = rag_prompts.write_section_v2("intro", "topic", [])
    assert "{style_fingerprint_section}" not in sys_p2, (
        "unrendered placeholder leaked into system prompt when fingerprint is None"
    )

    # Consumer wired into _autowrite_section_body
    from sciknow.core import book_ops
    body_src = inspect.getsource(book_ops._autowrite_section_body)
    assert "get_style_fingerprint(" in body_src, (
        "_autowrite_section_body not calling get_style_fingerprint — "
        "Layer 5 wiring missing"
    )
    assert "style_fingerprint_block=" in body_src, (
        "_autowrite_section_body not threading style_fingerprint_block "
        "into write_section_v2"
    )

    # CLI commands registered
    from sciknow.cli import book as book_cli
    assert hasattr(book_cli, "style_app"), "book.style_app subcommand missing"
    assert hasattr(book_cli, "style_refresh"), "style_refresh CLI handler missing"
    assert hasattr(book_cli, "style_show"), "style_show CLI handler missing"


def l1_phase33_keyboard_shortcuts_and_polish() -> None:
    """Phase 33 — keyboard shortcuts, chapter drag-drop, log rotation.

    Three QoL features shipped together:
    1. Keyboard shortcuts: Ctrl+S (force save), Ctrl+K (focus search),
       Ctrl+E (toggle editor), ←/→ (prev/next section), D (dashboard),
       P (plan modal)
    2. Chapter drag-and-drop reordering: ch-title bars are draggable,
       drop handler POSTs /api/chapters/reorder
    3. Log rotation: _rotate_old_logs keeps the last 50 autowrite logs
    """
    import inspect
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)

    # 1) Keyboard shortcuts — the global keydown handler must include
    #    the Ctrl+S / Ctrl+K / Ctrl+E / arrow / D / P shortcuts.
    assert "e.key === 's'" in src and "edAutosave" in src, (
        "Ctrl+S keyboard shortcut missing"
    )
    assert "e.key === 'k'" in src and "searchInput" in src, (
        "Ctrl+K keyboard shortcut missing"
    )
    assert "e.key === 'e'" in src and "toggleEdit" in src, (
        "Ctrl+E keyboard shortcut missing"
    )
    assert "ArrowLeft" in src and "ArrowRight" in src, (
        "← / → section navigation shortcuts missing"
    )

    # 2) Chapter drag-drop — handler functions exist
    for fn in ("chDragStart", "chDragOver", "chDrop", "chDragEnd"):
        assert f"function {fn}(" in src or f"async function {fn}(" in src, (
            f"chapter drag-drop handler {fn} missing"
        )
    # The ch-title must be draggable
    assert "ch-title clickable\" draggable=\"true\"" in src, (
        "ch-title not marked draggable"
    )
    # Drop handler must POST to /api/chapters/reorder
    ch_drop_src = src.split("function chDrop(")[1].split("\nfunction ")[0]
    assert "/api/chapters/reorder" in ch_drop_src, (
        "chDrop must POST to /api/chapters/reorder"
    )
    # CSS indicator classes must exist
    assert ".ch-drag-over-top" in src and ".ch-drag-over-bottom" in src, (
        "chapter drag visual indicator CSS missing"
    )

    # 3) Log rotation — _rotate_old_logs exists on _AutowriteLogger
    from sciknow.core.book_ops import _AutowriteLogger
    assert hasattr(_AutowriteLogger, "_rotate_old_logs"), (
        "_rotate_old_logs missing from _AutowriteLogger"
    )
    # Must be called from __init__
    init_src = inspect.getsource(_AutowriteLogger.__init__)
    assert "_rotate_old_logs(" in init_src, (
        "_rotate_old_logs not called from _AutowriteLogger.__init__"
    )


def l1_phase34_cars_rhetorical_moves() -> None:
    """Phase 34 — CARS-adapted chapter moves (Swales 1990 + Yang & Allison 2003).

    The 5-move scaffold (orient → tension → evidence → qualify → integrate)
    is injected into the tree_plan prompt as a parallel label alongside
    the existing PDTB-lite discourse_relation. The writer prompt renders
    it as [orient], [tension], etc. per paragraph.

    Verifies:
    - tree_plan prompt includes rhetorical_move in the schema
    - All 5 CARS moves are listed in the tree_plan system prompt
    - write_section_v2 correctly renders CARS tags from paragraph_plan
    - Without a paragraph_plan, no CARS block appears (backward compat)
    """
    from sciknow.rag import prompts

    # 1) tree_plan prompt must include rhetorical_move in the schema
    sys_tp, _ = prompts.tree_plan("intro", "topic", [])
    assert "rhetorical_move" in sys_tp, "tree_plan missing rhetorical_move field"
    for move in ("orient", "tension", "evidence", "qualify", "integrate"):
        assert move in sys_tp, f"tree_plan missing CARS move: {move}"

    # 2) Writer prompt renders CARS tags from paragraph_plan
    plan = [
        {"point": "test", "discourse_relation": "background",
         "rhetorical_move": "orient"},
        {"point": "test2", "discourse_relation": "contrast",
         "rhetorical_move": "tension"},
    ]
    sys_w, _ = prompts.write_section_v2("intro", "topic", [], paragraph_plan=plan)
    assert "[orient]" in sys_w, "writer not rendering [orient] tag"
    assert "[tension]" in sys_w, "writer not rendering [tension] tag"
    assert "CARS rhetorical move" in sys_w, "writer missing CARS legend"

    # 3) Without paragraph_plan, no CARS block (backward compat)
    sys_np, _ = prompts.write_section_v2("intro", "topic", [])
    assert "CARS" not in sys_np, (
        "CARS block should not appear without a paragraph_plan"
    )

    # 4) Toulmin scaffold guidance is present for [tension] paragraphs
    tension_plan = [
        {"point": "gap", "discourse_relation": "contrast",
         "rhetorical_move": "tension"},
    ]
    sys_t, _ = prompts.write_section_v2("intro", "topic", [],
                                        paragraph_plan=tension_plan)
    assert "Toulmin" in sys_t, "Toulmin scaffold missing from discourse block"
    for component in ("CLAIM", "DATA", "WARRANT", "QUALIFIER", "REBUTTAL"):
        assert component in sys_t, f"Toulmin component {component} missing"

    # 5) LongCite sentence-level citation guidance in WRITE_V2_SYSTEM
    sys_base, _ = prompts.write_section_v2("intro", "topic", [])
    assert "Sentence-level citation grounding" in sys_base, (
        "LongCite sentence-level citation rule missing from writer prompt"
    )
    assert "sentence-addressable" in sys_base, (
        "LongCite rationale (sentence-addressable verification) missing"
    )

    # 6) Scorer prompt references sentence-level groundedness
    sys_scorer, _ = prompts.score_draft("intro", "topic", "draft", [])
    assert "sentence-level" in sys_scorer.lower(), (
        "scorer prompt missing sentence-level groundedness guidance"
    )

    # 7) MADAM-RAG-lite: contradiction field in tree_plan schema + writer rendering
    assert "contradiction" in prompts.TREE_PLAN_SYSTEM, (
        "tree_plan prompt missing contradiction field (MADAM-RAG-lite)"
    )
    contr_plan = [
        {"point": "test", "discourse_relation": "contrast",
         "rhetorical_move": "tension",
         "contradiction": {
             "for": ["[1]"], "against": ["[2]"],
             "nature": "test disagreement",
         }},
    ]
    sys_c, _ = prompts.write_section_v2("intro", "topic", [],
                                        paragraph_plan=contr_plan)
    assert "CONTRADICTION" in sys_c, (
        "writer prompt not rendering the contradiction block"
    )
    assert "FOR: [1]" in sys_c and "AGAINST: [2]" in sys_c, (
        "writer prompt missing the pro/con source lists"
    )


def l1_phase35_total_compute_counter() -> None:
    """Phase 35 — book-level GPU compute counter on the dashboard.

    Verifies:
    - LLMUsageLog model exists in storage.models with the expected cols
    - Alembic migration 0015 creates llm_usage_log with the indexes
    - web/app.py wires _persist_llm_usage into the finally block of
      _run_generator_in_thread, BEFORE _finish_job
    - /api/dashboard returns a top-level `total_compute` key with the
      expected shape (total_tokens, total_seconds, total_jobs,
      by_operation[])
    - The dashboard HTML renders a "Total Compute" heading and the
      per-operation chips

    L1 (source-grep only — no DB needed). L2 will cover the insert
    roundtrip once a real run is in the ledger.
    """
    import inspect

    from sciknow.storage import models as sm
    assert hasattr(sm, "LLMUsageLog"), "LLMUsageLog model missing"
    cols = {c.name for c in sm.LLMUsageLog.__table__.columns}
    for needed in (
        "id", "book_id", "chapter_id", "operation", "model_name",
        "tokens", "duration_seconds", "status", "started_at", "finished_at",
    ):
        assert needed in cols, f"LLMUsageLog.{needed} column missing"
    assert sm.LLMUsageLog.__tablename__ == "llm_usage_log"

    # Migration file — assert it exists and references the expected cols.
    from pathlib import Path
    mig = Path(__file__).resolve().parents[1].parent / "migrations" / "versions" / "0015_llm_usage_log.py"
    assert mig.exists(), f"migration 0015_llm_usage_log.py missing at {mig}"
    mig_src = mig.read_text()
    assert "create_table" in mig_src and "llm_usage_log" in mig_src
    assert "idx_llm_usage_book" in mig_src
    assert "idx_llm_usage_book_op" in mig_src
    assert 'down_revision: Union[str, None] = "0014"' in mig_src, (
        "migration 0015 should descend from 0014"
    )

    # Wiring in web/app.py — persistence helper + finally-block call.
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)
    assert "def _persist_llm_usage(" in src, (
        "_persist_llm_usage helper missing from web/app.py"
    )
    # _persist_llm_usage must be called from _run_generator_in_thread's
    # finally block, BEFORE _finish_job (order matters: if _finish_job
    # ran first the 'done' bookkeeping would be complete before we read
    # the counters; current implementation reads under the same lock
    # so both orderings work, but the spec is to persist first).
    rg_src = inspect.getsource(web_app._run_generator_in_thread)
    assert "_persist_llm_usage(job_id)" in rg_src, (
        "_persist_llm_usage not called from _run_generator_in_thread"
    )
    idx_persist = rg_src.index("_persist_llm_usage(job_id)")
    idx_finish = rg_src.index("_finish_job(job_id)")
    assert idx_persist < idx_finish, (
        "_persist_llm_usage must be called BEFORE _finish_job so the "
        "ledger row reflects the still-live per-job counters"
    )
    # Skip zero-token jobs
    ph_src = inspect.getsource(web_app._persist_llm_usage)
    assert "tokens <= 0" in ph_src, (
        "_persist_llm_usage must skip zero-token jobs"
    )
    # Wallclock start timestamp captured on job creation
    create_src = inspect.getsource(web_app._create_job)
    assert '"started_wall"' in create_src, (
        "_create_job must capture started_wall for the ledger"
    )

    # Dashboard endpoint — total_compute key + by_operation aggregation
    dash_src = inspect.getsource(web_app.api_dashboard)
    assert '"total_compute"' in dash_src, (
        "api_dashboard must return total_compute"
    )
    assert "FROM llm_usage_log" in dash_src, (
        "api_dashboard must aggregate from llm_usage_log"
    )
    assert "GROUP BY operation" in dash_src, (
        "api_dashboard must compute per-operation breakdown"
    )
    assert "by_operation" in dash_src

    # Dashboard HTML — Total Compute widget
    assert "Total Compute" in src, (
        "dashboard HTML missing 'Total Compute' heading"
    )
    assert "tc.by_operation" in src, (
        "dashboard HTML missing per-operation breakdown rendering"
    )
    assert "Total Tokens" in src and "Total Time" in src, (
        "dashboard HTML missing Total Tokens / Total Time tiles"
    )


def l1_phase36_tools_panel() -> None:
    """Phase 36 — Tools modal bringing CLI-only capabilities into the web UI.

    Four tabs mapped to CLI commands:
      Search     → /api/search/query, /api/search/similar       (JSON)
      Synthesize → /api/ask/synthesize                          (SSE)
      Topics     → /api/catalog/topics                          (JSON)
      Corpus     → /api/corpus/enrich, /api/corpus/expand       (SSE, subprocess)

    Verifies handlers exist with the right shape, the Tools button is
    wired, the modal HTML carries all four tabs, and the JS dispatches
    to the right endpoint per tab.
    """
    import inspect

    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)

    # 1) Handlers exist on the module
    for name in (
        "api_search_query", "api_search_similar",
        "api_ask_synthesize", "api_catalog_topics",
        "api_corpus_enrich", "api_corpus_expand",
        "_spawn_cli_streaming", "_sciknow_cli_bin",
    ):
        assert hasattr(web_app, name), f"web.app.{name} missing"

    # 2) Routes are registered with the expected paths
    paths = {getattr(r, "path", None) for r in web_app.app.routes}
    for p in (
        "/api/search/query", "/api/search/similar",
        "/api/ask/synthesize", "/api/catalog/topics",
        "/api/corpus/enrich", "/api/corpus/expand",
    ):
        assert p in paths, f"route {p} not registered"

    # 3) Search handler uses the real retrieval stack, not a stub
    sq_src = inspect.getsource(web_app.api_search_query)
    assert "hybrid_search.search" in sq_src
    assert "reranker.rerank" in sq_src
    assert "context_builder.build" in sq_src
    # no_rerank branch present
    assert "no_rerank" in sq_src

    # 4) Similar search uses the abstracts collection
    ss_src = inspect.getsource(web_app.api_search_similar)
    assert "ABSTRACTS_COLLECTION" in ss_src
    assert "qdrant.query_points" in ss_src

    # 5) Synthesize uses prompts.synthesis (NOT prompts.qa)
    syn_src = inspect.getsource(web_app.api_ask_synthesize)
    assert "rag_prompts.synthesis" in syn_src, (
        "api_ask_synthesize must use prompts.synthesis, not prompts.qa "
        "— otherwise it duplicates /api/ask instead of mirroring "
        "`sciknow ask synthesize`"
    )
    assert "_create_job" in syn_src and "_run_generator_in_thread" in syn_src

    # 6) Catalog topics — two modes (list vs per-topic paper list)
    tp_src = inspect.getsource(web_app.api_catalog_topics)
    assert "GROUP BY topic_cluster" in tp_src
    assert 'WHERE pm.topic_cluster = :n' in tp_src

    # 7) Corpus actions shell out to the sciknow CLI
    enr_src = inspect.getsource(web_app.api_corpus_enrich)
    exp_src = inspect.getsource(web_app.api_corpus_expand)
    assert '"db", "enrich"' in enr_src
    assert '"db", "expand"' in exp_src
    # They pipe stdout to SSE via _spawn_cli_streaming
    spawn_src = inspect.getsource(web_app._spawn_cli_streaming)
    assert "subprocess.Popen" in spawn_src
    assert '"log"' in spawn_src  # emits {type: "log", text: ...}
    assert "proc.terminate" in spawn_src  # cancelable
    # stderr merged into stdout so Rich progress bars surface in the log
    assert "STDOUT" in spawn_src

    # 8) Tools button is wired to the new modal
    assert 'onclick="openToolsModal()"' in src, (
        "Tools button not wired into the top-bar button tray"
    )
    # 9) Modal HTML has all four tabs
    for tab in ("tl-search", "tl-synth", "tl-topics", "tl-corpus"):
        assert f'data-tab="{tab}"' in src, f"tab {tab} missing from modal"
        assert f'id="{tab}-pane"' in src, f"pane {tab}-pane missing"
    # 10) JS dispatchers exist and hit the right endpoints
    for fn in ("openToolsModal", "switchToolsTab", "doToolSearch",
               "doToolSynthesize", "loadToolTopics", "loadToolTopicPapers",
               "doToolCorpus", "cancelToolCorpus"):
        assert f"function {fn}(" in src or f"async function {fn}(" in src, (
            f"Tools JS function {fn} missing"
        )
    assert "'/api/search/query'" in src and "'/api/search/similar'" in src
    assert "'/api/ask/synthesize'" in src
    assert "'/api/catalog/topics'" in src
    assert "'/api/corpus/' + action" in src


def l1_phase37_per_section_model_override() -> None:
    """Phase 37 — per-section model override.

    Section meta gets an optional `model` field; the four streams
    (write / autowrite / review / revise) consult it before falling
    through to the caller-provided model or the global default.

    Precedence at every LLM call site:
        1. Explicit caller model (CLI --model / API form field)
        2. Per-section model override (new — this phase)
        3. settings.llm_model default

    Verifies: normalize preserves model, _get_section_model exists with
    the right lookup semantics, all four stream functions invoke it,
    and the chapter-modal Sections tab has the UI wiring.
    """
    import inspect

    from sciknow.core import book_ops

    # 1) Helper exists and has the right signature
    assert hasattr(book_ops, "_get_section_model"), (
        "_get_section_model helper missing from book_ops"
    )
    sig = inspect.signature(book_ops._get_section_model)
    assert set(sig.parameters.keys()) >= {"session", "chapter_id", "section_slug"}, (
        "_get_section_model signature must be (session, chapter_id, section_slug)"
    )

    # 2) normalize preserves model (non-empty string) and drops blanks
    norm = book_ops._normalize_chapter_sections([
        {"slug": "s1", "title": "S1", "plan": "", "model": "qwen3:32b"},
        {"slug": "s2", "title": "S2", "plan": "", "model": ""},
        {"slug": "s3", "title": "S3", "plan": ""},  # no model key at all
    ])
    by_slug = {s["slug"]: s for s in norm}
    assert by_slug["s1"]["model"] == "qwen3:32b", (
        "model string must be preserved by _normalize_chapter_sections"
    )
    assert by_slug["s2"]["model"] is None, (
        "empty-string model must normalise to None"
    )
    assert by_slug["s3"]["model"] is None, (
        "missing model key must default to None"
    )

    # 3) All four streams invoke the resolver BEFORE passing model to
    #    downstream llm_stream calls. Grep by function source.
    for fn_name in ("write_section_stream", "_autowrite_section_body",
                    "review_draft_stream", "revise_draft_stream"):
        fn = getattr(book_ops, fn_name)
        src = inspect.getsource(fn)
        assert "_get_section_model" in src, (
            f"{fn_name} must consult _get_section_model for per-section override"
        )
        # Guarded by `model is None` — caller-provided model wins
        assert "if model is None" in src, (
            f"{fn_name} must only apply section override when caller passed no model"
        )

    # 4) Web UI wiring — load, render, save, handler
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)
    # Editor state mirror carries the new field
    assert "model: (s.model && typeof s.model === 'string') ? s.model : ''" in src, (
        "section editor load path must copy model from sections_meta"
    )
    # Save payload includes it
    assert "model: (s.model || '').trim() || null" in src, (
        "section editor save path must include model in sectionsToSave"
    )
    # Row renders a model input + datalist
    assert 'list="sec-model-suggestions"' in src
    assert 'id="sec-model-suggestions"' in src
    assert 'updateSectionModel(' in src
    # Update handler exists
    assert "function updateSectionModel(" in src


def l1_phase38_scoped_snapshot_bundles() -> None:
    """Phase 38 — chapter + book snapshot bundles.

    Extends the per-draft snapshot system with two new scopes for
    autowrite-all safety. Verifies:
    - DraftSnapshot model has scope / chapter_id / book_id columns
    - draft_id is now nullable (chapter/book bundles have no single draft)
    - Migration 0016 exists and descends from 0015
    - Five new endpoints: create (chapter/book), list (chapter/book),
      non-destructive bundle restore
    - UI: Bundles button + modal with chapter/book tabs
    - Restore helper creates NEW draft versions, doesn't overwrite
    """
    import inspect
    from pathlib import Path

    from sciknow.storage import models as sm
    cols = {c.name: c for c in sm.DraftSnapshot.__table__.columns}
    for needed in ("scope", "chapter_id", "book_id"):
        assert needed in cols, f"DraftSnapshot.{needed} column missing"
    # Phase 38 made draft_id nullable so chapter/book rows can exist
    assert cols["draft_id"].nullable is True, (
        "DraftSnapshot.draft_id must be nullable so chapter/book "
        "scope rows can be inserted"
    )

    # Migration file
    mig = Path(__file__).resolve().parents[1].parent / "migrations" / "versions" / "0016_scoped_snapshots.py"
    assert mig.exists(), f"migration 0016 missing at {mig}"
    mig_src = mig.read_text()
    assert 'down_revision: Union[str, None] = "0015"' in mig_src
    assert "add_column" in mig_src and "scope" in mig_src
    assert "alter_column" in mig_src and "draft_id" in mig_src

    # Endpoints + routes
    from sciknow.web import app as web_app
    for name in (
        "create_chapter_snapshot", "create_book_snapshot",
        "list_chapter_snapshots", "list_book_snapshots",
        "restore_snapshot_bundle", "_snapshot_chapter_drafts",
        "_restore_chapter_bundle",
    ):
        assert hasattr(web_app, name), f"web.app.{name} missing"
    paths = {getattr(r, "path", None) for r in web_app.app.routes}
    for p in (
        "/api/snapshot/chapter/{chapter_id}",
        "/api/snapshot/book/{book_id}",
        "/api/snapshots/chapter/{chapter_id}",
        "/api/snapshots/book/{book_id}",
        "/api/snapshot/restore-bundle/{snapshot_id}",
    ):
        assert p in paths, f"route {p} not registered"

    # Restore is non-destructive: INSERTs into drafts, doesn't UPDATE
    restore_src = inspect.getsource(web_app._restore_chapter_bundle)
    assert "INSERT INTO drafts" in restore_src, (
        "bundle restore must INSERT new draft rows"
    )
    assert "MAX(version)" in restore_src, (
        "bundle restore must compute next version per section"
    )
    # Status check so non-bundle snapshots fail fast
    rsb_src = inspect.getsource(web_app.restore_snapshot_bundle)
    assert 'scope not in ("chapter", "book")' in rsb_src, (
        "restore_snapshot_bundle must reject draft-scope snapshots"
    )

    # Chapter snapshot must capture latest version per section_type
    csh_src = inspect.getsource(web_app._snapshot_chapter_drafts)
    assert "DISTINCT ON (d.section_type)" in csh_src, (
        "chapter snapshot must capture ONE row per section_type"
    )

    # UI wiring
    src = inspect.getsource(web_app)
    assert 'onclick="openBundleSnapshots()"' in src, (
        "Bundles button not wired"
    )
    for tab in ("sb-chapter", "sb-book"):
        assert f'data-tab="{tab}"' in src, f"tab {tab} missing"
        assert f'id="{tab}-pane"' in src, f"pane {tab}-pane missing"
    for fn in ("openBundleSnapshots", "switchBundleTab",
               "doBundleSnapshot", "loadBundleList", "restoreBundle"):
        assert f"function {fn}(" in src or f"async function {fn}(" in src, (
            f"JS function {fn} missing"
        )


def l1_phase39_book_settings_modal() -> None:
    """Phase 39 — consolidated per-book Settings modal.

    Brings title / description / plan / target_chapter_words /
    style_fingerprint into one editor. Leaves existing per-chapter
    surfaces alone (Chapter modal still owns sections editing).

    Verifies:
    - GET /api/book now returns style_fingerprint (None or dict)
    - POST /api/book/style-fingerprint/refresh endpoint exists and
      calls compute_style_fingerprint
    - Settings button + modal HTML with 3 tabs (basics/leitmotiv/style)
    - JS functions for open/switch/load/save/refresh
    """
    import inspect

    from sciknow.web import app as web_app

    # 1) New endpoint registered
    paths = {getattr(r, "path", None) for r in web_app.app.routes}
    assert "/api/book/style-fingerprint/refresh" in paths, (
        "style fingerprint refresh endpoint not registered"
    )
    assert hasattr(web_app, "api_book_style_fingerprint_refresh")

    # 2) Refresh handler calls compute_style_fingerprint
    fn_src = inspect.getsource(web_app.api_book_style_fingerprint_refresh)
    assert "compute_style_fingerprint" in fn_src, (
        "refresh endpoint must call compute_style_fingerprint"
    )

    # 3) GET /api/book surfaces style_fingerprint
    get_src = inspect.getsource(web_app.api_book)
    assert '"style_fingerprint"' in get_src, (
        "/api/book must expose style_fingerprint for the settings modal"
    )

    # 4) Button + modal HTML wiring
    src = inspect.getsource(web_app)
    assert 'onclick="openBookSettings()"' in src, (
        "Settings button not wired into the top-bar"
    )
    assert 'id="book-settings-modal"' in src
    for tab in ("bs-basics", "bs-leitmotiv", "bs-style"):
        assert f'data-tab="{tab}"' in src, f"tab {tab} missing"
        assert f'id="{tab}-pane"' in src, f"pane {tab}-pane missing"

    # 5) JS dispatchers exist
    for fn in ("openBookSettings", "switchBookSettingsTab",
               "loadBookSettings", "renderStyleFingerprint",
               "saveBookSettings", "refreshStyleFingerprint"):
        assert f"function {fn}(" in src or f"async function {fn}(" in src, (
            f"JS function {fn} missing"
        )

    # 6) Save roundtrip hits the existing PUT /api/book (no new write endpoint)
    save_src = src[src.index("async function saveBookSettings("):]
    save_src = save_src[:save_src.index("\nasync function ")] if "\nasync function " in save_src else save_src[:3000]
    assert "/api/book" in save_src and "method: 'PUT'" in save_src, (
        "saveBookSettings must PUT to /api/book (no new write endpoint)"
    )


def l1_phase40_cli_export_pdf_epub() -> None:
    """Phase 40 — CLI parity for PDF and EPUB export.

    `sciknow book export` previously supported markdown, html, bibtex,
    latex, docx. The web reader had its own WeasyPrint PDF button
    (Phase 31) but the CLI didn't. EPUB was missing from both. This
    phase adds both to the CLI.

    Verifies:
    - PDF path exists and uses weasyprint (same lib as the web reader)
    - EPUB path exists and invokes pandoc with --citeproc
    - Help text + error message list the new formats
    """
    import inspect

    from sciknow.cli import book as book_cli
    src = inspect.getsource(book_cli.export)

    # PDF branch
    assert 'if fmt == "pdf":' in src, "pdf branch missing from `book export`"
    assert "weasyprint" in src, (
        "pdf branch must use weasyprint (same lib as the web reader's "
        "_html_to_pdf_response)"
    )
    assert "write_pdf" in src

    # EPUB branch
    assert 'if fmt == "epub":' in src, "epub branch missing from `book export`"
    # Pandoc invocation with citeproc + bibliography
    assert '"pandoc"' in src and "--citeproc" in src and "--bibliography=" in src, (
        "epub branch must call pandoc --citeproc with the generated bibliography"
    )
    # Pandoc missing → friendly exit, not a crash
    assert 'shutil.which("pandoc")' in src, (
        "epub branch must guard on pandoc availability"
    )

    # Help text + unknown-format message surface the new formats
    assert "pdf" in book_cli.export.__doc__.lower(), (
        "export docstring must list pdf"
    )
    assert "epub" in book_cli.export.__doc__.lower(), (
        "export docstring must list epub"
    )
    # Unknown-format error now lists all seven formats
    assert "Use: markdown, html, pdf, epub, bibtex, latex, docx" in src, (
        "unknown-format error must advertise the full format list"
    )


def l1_phase41_static_where_clauses() -> None:
    """Phase 41 — retire the WHERE-clause f-strings in /api/catalog and
    /api/kg.

    The Phase 22 audit flagged both endpoints for assembling the SQL
    WHERE clause from a Python list of pre-formatted fragments and
    then interpolating the result via an f-string. Not exploitable
    today (every fragment is a code-local string literal) but a fragile
    pattern — one future PR could concatenate user input into the
    query shape.

    Refactor: bind every optional filter as the real value or NULL,
    gate each clause with ``(:param IS NULL OR …)``. The SQL string
    is now fully static — no f-string, no .format(), no dynamic
    assembly — which makes the injection-vector question decidable
    at lint time rather than review time.

    Verifies:
    - Neither handler source uses `WHERE " + " AND ".join(...)` or
      builds `where_sql` / `where_clause` from string concatenation.
    - Neither handler source uses f-string SQL blocks.
    - The hallmark of the always-bind pattern is present:
      `(:param IS NULL OR …)` for each filter.
    """
    import inspect

    from sciknow.web import app as web_app

    cat_src = inspect.getsource(web_app.api_catalog)
    kg_src = inspect.getsource(web_app.api_kg)

    for name, src in (("api_catalog", cat_src), ("api_kg", kg_src)):
        # No dynamic WHERE assembly
        assert 'where = []' not in src, (
            f"{name}: dynamic `where = []` accumulator still present"
        )
        assert '" AND ".join(' not in src, (
            f"{name}: `\" AND \".join(...)` WHERE assembly still present"
        )
        assert 'where_clause' not in src and 'where_sql' not in src, (
            f"{name}: `where_clause` / `where_sql` variable still present"
        )
        # No f-string SQL blocks (triple-quoted or otherwise)
        assert 'text(f"""' not in src, (
            f"{name}: f-string SQL block still present"
        )
        assert 'text(f"' not in src, (
            f"{name}: f-string single-line SQL still present"
        )
        # Always-bind pattern is the hallmark of the refactor
        assert "IS NULL OR" in src, (
            f"{name}: missing `(:param IS NULL OR …)` gates — the "
            f"always-bind pattern is the whole point of the refactor"
        )


def l1_phase42_data_action_dispatcher() -> None:
    """Phase 42 — retire inline onclick handlers with interpolated args.

    The ~20 handlers that had f-string-interpolated variables (button
    IDs, slugs, draft UUIDs, …) were flagged by the Phase 22 audit:
    escaping-mitigated but fragile — one future PR forgetting _esc()
    could reintroduce XSS. This phase converts every such site to a
    data-action + data-* attribute pattern routed through a single
    document-level click listener (ACTIONS registry).

    Static handlers like onclick="openPlanModal()" are left alone —
    no interpolation = no fragility. A future CSP pass can finish
    them off.

    Verifies:
    - ACTIONS registry + document-level click listener exist
    - Every entry keyed by the handlers the refactor moved over
    - Rendered HTML has ZERO stale interpolated onclicks from the
      refactored sites (sidebar + comments + heatmap + snapshots +
      wiki + catalog + version history + section editor)
    - Plenty of data-action attributes present on the rendered page
    """
    import re

    from sciknow.testing.helpers import rendered_template_static
    from sciknow.web import app as web_app
    import inspect

    src = inspect.getsource(web_app)

    # 1) Dispatcher scaffolding
    assert "const ACTIONS = {" in src, "ACTIONS registry missing"
    assert "document.addEventListener('click', function(e)" in src, (
        "data-action click listener missing"
    )
    assert "closest('[data-action]')" in src, (
        "listener must walk up to the nearest [data-action] element"
    )

    # 2) Every action name the refactor introduced has a registry entry
    expected_actions = {
        # Cluster 1 — sidebar / comments
        "preview-empty-section", "start-writing-chapter",
        "adopt-orphan-section", "delete-orphan-draft", "resolve-comment",
        # Cluster 2 — heatmap + gaps
        "open-chapter-modal", "load-section", "write-for-gap",
        # Cluster 3 — section editor + sidebar JS
        "set-section-type", "add-section-to-chapter", "move-section",
        "remove-section",
        # Cluster 4 — wiki + catalog pagination
        "open-wiki-page", "load-wiki-pages", "ask-about-paper",
        "load-catalog",
        # Cluster 5 — version history + snapshots
        "select-version", "restore-bundle",
        "diff-snapshot", "restore-snapshot",
    }
    for action in expected_actions:
        assert f"'{action}':" in src, (
            f"ACTIONS['{action}'] handler entry missing"
        )

    # 3) Rendered HTML has no stale interpolated handlers from the
    #    refactored sites. The interpolated-onclick pattern is what we
    #    removed; detect any remaining site by looking for the former
    #    handler names in onclick= attributes followed by an arg.
    rendered = rendered_template_static()
    stale_patterns = [
        r'onclick="resolveComment\(',
        r'onclick="previewEmptySection\(',
        r'onclick="startWritingChapter\(',
        r'onclick="adoptOrphanSection\(',
        r'onclick="deleteOrphanDraft\(',
    ]
    for pat in stale_patterns:
        assert re.search(pat, rendered) is None, (
            f"stale interpolated onclick handler still in rendered HTML: {pat}"
        )

    # 4) Plenty of data-action attributes present — the refactor
    #    should have produced many on the rendered page.
    attr_count = len(re.findall(r'data-action="[a-z-]+"', rendered))
    assert attr_count >= 3, (
        f"expected many data-action attributes on the rendered page, "
        f"only found {attr_count}"
    )


def l1_phase43_project_resolution() -> None:
    """Phase 43 — multi-project foundations.

    Static / no-DB checks on:
    - Project dataclass + slug validation
    - get_active_project() precedence: SCIKNOW_PROJECT env >
      .active-project file > legacy 'default' fallback
    - Settings delegate data_dir + pg_database to the active project
    - Qdrant module-level __getattr__ resolves legacy constants to the
      active project's collection names
    - sciknow.cli.project module exposes init/list/show/use/destroy/
      archive/unarchive
    - Root CLI exposes --project flag

    DB / Qdrant / filesystem state changes are out of scope here
    (those go in L2).
    """
    import inspect
    import os

    from sciknow.core.project import (
        Project,
        get_active_project,
        list_projects,
        validate_slug,
    )

    # 1) Slug validation rules
    for good in ("global-cooling", "abc", "a1", "x-y-z", "default"):
        validate_slug(good)
    for bad in ("Bad", "no_underscore", "-leading", "trailing-", "with space", "", "weird!"):
        try:
            validate_slug(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"validate_slug should have rejected {bad!r}")

    # 2) Default project legacy compat
    d = Project.default()
    assert d.is_default
    assert d.pg_database == "sciknow"
    assert d.papers_collection == "papers"
    assert d.abstracts_collection == "abstracts"
    assert d.wiki_collection == "wiki"
    assert d.qdrant_prefix == ""

    # 3) Real project derivations (PG / Qdrant identifier safety —
    # hyphens converted to underscores)
    p = Project(slug="global-cooling", repo_root=d.repo_root)
    assert not p.is_default
    assert p.pg_database == "sciknow_global_cooling"
    assert p.qdrant_prefix == "global_cooling_"
    assert p.papers_collection == "global_cooling_papers"
    assert p.root.name == "global-cooling"
    assert p.root.parent.name == "projects"

    # 4) get_active_project() precedence — SCIKNOW_PROJECT env wins
    prev_env = os.environ.pop("SCIKNOW_PROJECT", None)
    try:
        os.environ["SCIKNOW_PROJECT"] = "test-priority"
        active = get_active_project()
        assert active.slug == "test-priority", (
            f"SCIKNOW_PROJECT env should override (got {active.slug!r})"
        )
    finally:
        if prev_env is None:
            os.environ.pop("SCIKNOW_PROJECT", None)
        else:
            os.environ["SCIKNOW_PROJECT"] = prev_env

    # No env, no .active-project file (assuming the test env is
    # legacy-default) → 'default'.
    if not (d.repo_root / ".active-project").exists() and not list_projects():
        assert get_active_project().is_default

    # 5) Settings delegates to active project
    from sciknow.config import Settings
    s = Settings()
    # In the test process, no SCIKNOW_PROJECT and no projects/ dir →
    # default project values
    if not list_projects() and not os.environ.get("SCIKNOW_PROJECT"):
        assert s.pg_database == "sciknow", (
            f"settings.pg_database should default to legacy 'sciknow' "
            f"(got {s.pg_database!r})"
        )
        # data_dir is now absolute; check the trailing component
        assert s.data_dir.name == "data"

    # 6) Qdrant module-level __getattr__ legacy compat
    from sciknow.storage import qdrant as qmod
    assert callable(qmod.papers_collection)
    assert callable(qmod.abstracts_collection)
    assert callable(qmod.wiki_collection)
    # The PEP 562 __getattr__ should resolve legacy constants to
    # the active project's collection name (a string, not a callable).
    assert isinstance(qmod.PAPERS_COLLECTION, str)
    assert isinstance(qmod.ABSTRACTS_COLLECTION, str)
    assert isinstance(qmod.WIKI_COLLECTION, str)
    # Bogus attribute still raises AttributeError
    try:
        _ = qmod.BOGUS_NAME
    except AttributeError:
        pass
    else:
        raise AssertionError("qdrant.__getattr__ should reject unknown names")

    # 7) sciknow.cli.project subcommands wired in
    from sciknow.cli import project as project_cli
    for name in ("init", "list_cmd", "show", "use", "destroy",
                 "archive", "unarchive"):
        assert hasattr(project_cli, name), f"project CLI missing {name!r}"

    # 8) Root CLI exposes --project flag
    from sciknow.cli import main as main_cli
    callback_src = inspect.getsource(main_cli._startup)
    assert "--project" in callback_src
    assert "SCIKNOW_PROJECT" in callback_src

    # 9) DB layer accepts an explicit db_name
    from sciknow.storage.db import get_engine, get_admin_engine, get_session
    eng = get_engine("test_db_name_abc")
    assert "test_db_name_abc" in str(eng.url)
    admin = get_admin_engine()
    assert str(admin.url).endswith("/postgres")
    sig = inspect.signature(get_session)
    assert "db_name" in sig.parameters


def l1_phase45_project_types() -> None:
    """Phase 45 — project_type registry + CLI + Book model column.

    Static checks: scientific_book + scientific_paper registered,
    get_project_type falls back to default on unknown slug, flat types
    have is_flat=True with a required abstract-like section, Book model
    has book_type, `sciknow book types` + `sciknow book create --type`
    are exposed.
    """
    import inspect

    from sciknow.core.project_type import (
        PROJECT_TYPES, DEFAULT_TYPE_SLUG,
        get_project_type, list_project_types, validate_type_slug,
        default_sections_as_dicts, section_keys, target_words_for,
    )
    from sciknow.storage.models import Book
    from sciknow.cli import book as book_cli

    # Registry covers the two shipped types
    assert "scientific_book"  in PROJECT_TYPES
    assert "scientific_paper" in PROJECT_TYPES
    assert DEFAULT_TYPE_SLUG  == "scientific_book"

    # Fallback to default on unknown slug (never raises)
    assert get_project_type("not_a_type").slug == DEFAULT_TYPE_SLUG
    assert get_project_type(None).slug         == DEFAULT_TYPE_SLUG

    # validate_type_slug rejects unknown slugs
    try:
        validate_type_slug("definitely-not-a-type")
    except ValueError:
        pass
    else:
        raise AssertionError("validate_type_slug should reject unknown slugs")

    # Paper type is flat with IMRaD canonical sections
    paper = PROJECT_TYPES["scientific_paper"]
    assert paper.is_flat is True
    keys = section_keys(paper)
    for expected in ("abstract", "introduction", "methods",
                     "results", "discussion", "conclusion"):
        assert expected in keys, f"paper template missing {expected!r}"
    # Required flag on abstract (the IMRaD opener)
    assert any(s.key == "abstract" and s.required for s in paper.default_sections), (
        "scientific_paper must mark `abstract` as required"
    )
    # target_words_for returns something for a templated section
    assert target_words_for(paper, "abstract") is not None
    assert target_words_for(paper, "not_a_section") is None

    # Book type is hierarchical
    book = PROJECT_TYPES["scientific_book"]
    assert book.is_flat is False

    # Sections as dicts match the BookChapter.sections JSONB shape
    for d in default_sections_as_dicts(paper):
        for k in ("key", "title", "target_words", "required"):
            assert k in d, f"section dict missing {k}"

    # Book SQLAlchemy model has the new column
    assert hasattr(Book, "book_type"), "Book model missing book_type column"

    # CLI surface — typer registers commands by function name when no
    # explicit name= is passed, so introspect via callback identity rather
    # than the .name attribute (which is None for unnamed commands).
    callbacks = {c.callback for c in book_cli.app.registered_commands}
    assert book_cli.types  in callbacks, "`sciknow book types` not registered"
    assert book_cli.create in callbacks, "`sciknow book create` not registered"
    # create() signature has the --type parameter
    create_sig = inspect.signature(book_cli.create)
    assert "type" in create_sig.parameters, "book create missing --type flag"


def l1_phase45_watchlist_surface() -> None:
    """Phase 45 — repo watchlist module + CLI surface."""
    import inspect

    from sciknow.core import watchlist as wl
    from sciknow.cli import watch as watch_cli
    from sciknow.cli import main as main_cli

    # URL parser
    owner, repo = wl.parse_github_url("https://github.com/foo/bar")
    assert (owner, repo) == ("foo", "bar")
    owner, repo = wl.parse_github_url("https://github.com/foo/bar.git/")
    assert (owner, repo) == ("foo", "bar")
    try:
        wl.parse_github_url("https://gitlab.com/foo/bar")
    except ValueError:
        pass
    else:
        raise AssertionError("parse_github_url should reject non-github URLs")

    # SEED_REPOS has at least the four priors we know about
    seed_keys = {u["url"] for u in wl.SEED_REPOS}
    for must in (
        "https://github.com/karpathy/autoresearch",
        "https://github.com/SakanaAI/AI-Scientist",
        "https://github.com/aiming-lab/AutoResearchClaw",
    ):
        assert must in seed_keys, f"seed watchlist missing {must}"

    # Public API is zero-arg or dataclass-trivial
    for name in ("add", "remove", "check", "check_all",
                 "list_watched", "seed_if_empty",
                 # Phase 46.D rate-limit: daily cooldown + RateLimited
                 "CHECK_COOLDOWN_HOURS", "RateLimited",
                 "_hours_since"):
        assert hasattr(wl, name), f"watchlist missing {name}"
    # Daily cooldown is the default (24h); callers can override.
    assert wl.CHECK_COOLDOWN_HOURS == 24.0, (
        "daily watchlist cooldown should default to 24h to be polite "
        "to the anon GitHub API; current value: "
        f"{wl.CHECK_COOLDOWN_HOURS}"
    )
    # _hours_since handles both "Z" and "+00:00" ISO formats
    assert wl._hours_since(None) is None
    assert wl._hours_since("not a date") is None
    from datetime import datetime as _dt, timezone as _tz
    recent = _dt.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hrs = wl._hours_since(recent)
    assert hrs is not None and hrs < 0.5, (
        f"freshly-stamped timestamp should report <30min, got {hrs}"
    )
    import inspect as _inspect
    sig = _inspect.signature(wl.check)
    for p in ("force", "cooldown_hours", "github_token"):
        assert p in sig.parameters, f"wl.check missing {p!r} keyword"

    # CLI registered on root
    cmd_names = {c.name for c in main_cli.app.registered_groups}
    assert "watch" in cmd_names, "`sciknow watch` subapp not registered"
    # Subapp has the expected commands (introspect via callback identity
    # to cover both explicitly-named and function-name-defaulted commands)
    callbacks = {c.callback for c in watch_cli.app.registered_commands}
    for fn_name in ("list_cmd", "add", "remove", "check", "note", "seed"):
        assert getattr(watch_cli, fn_name, None) in callbacks, (
            f"sciknow watch missing `{fn_name}` subcommand"
        )


def l1_phase46_citation_insert_surface() -> None:
    """Phase 46.A — two-stage citation insertion prompts + generator + CLI.

    Static checks. Exercises the JSON parsers on synthetic LLM output
    to guard against a tolerance regression, and confirms the CLI hook
    is registered.
    """
    import inspect

    from sciknow.rag import prompts as rag_prompts
    from sciknow.core import book_ops
    from sciknow.cli import book as book_cli

    # Prompt functions exist and return (system, user) tuples
    for fn_name in ("citation_needs", "citation_choose"):
        assert hasattr(rag_prompts, fn_name), f"prompts missing {fn_name}"
    s1, u1 = rag_prompts.citation_needs("methods", "Data collection",
                                         "Global temperature rose 1.1 °C since 1850.")
    assert "location" in s1 and "verbatim" in s1.lower()
    assert "methods" in u1

    s2, u2 = rag_prompts.citation_choose(
        claim="warming magnitude", location="rose 1.1 °C",
        candidates=[{"title": "Foo", "year": 2020, "section": "results",
                     "preview": "Observed 1.1 °C rise since preindustrial.",
                     "doc_id": "x", "chunk_id": "y"}],
    )
    assert "CITE" in s2 and "NONE" in s2

    # JSON parsers are tolerant of fenced code blocks + missing keys
    parsed = book_ops._parse_citation_needs_json(
        "```json\n{\"needs\": [{\"location\": \"x\", \"claim\": \"y\", \"query\": \"z\"}]}\n```"
    )
    assert isinstance(parsed, list) and parsed and parsed[0]["location"] == "x"
    assert book_ops._parse_citation_needs_json("garbage") == []
    choice = book_ops._parse_citation_choice_json(
        '{"verdict": "CITE", "chosen": [{"candidate_index": 0, "confidence": 0.9}]}'
    )
    assert choice["verdict"] == "CITE"
    bad = book_ops._parse_citation_choice_json("not json")
    assert bad["verdict"] == "NONE"

    # The streaming generator exists + is callable
    assert callable(getattr(book_ops, "insert_citations_stream", None))

    # CLI command registered
    callbacks = {c.callback for c in book_cli.app.registered_commands}
    assert book_cli.insert_citations in callbacks, (
        "`sciknow book insert-citations` not registered"
    )
    # And _consume_events recognises the new event types (a whitelist
    # grep: if someone deletes the handlers, this test flags it).
    import inspect as _inspect
    src = _inspect.getsource(book_cli._consume_events)
    for token in ("citation_needs", "citation_candidates",
                  "citation_selected", "citation_skipped",
                  "citation_inserted"):
        assert token in src, f"_consume_events missing handler for {token!r}"


def l1_phase46_citation_verify_surface() -> None:
    """Phase 46.B — external citation verification module + CLI."""
    from sciknow.core import citation_verify as cv
    from sciknow.cli import book as book_cli

    # Verdict semantics: title similarity thresholds must match the
    # AutoResearchClaw calibration (0.80 / 0.50) per docs/COMPARISON.md.
    assert cv.T_HIGH == 0.80
    assert cv.T_LOW  == 0.50
    assert cv._verdict_from_similarity(0.9)  == cv.VERIFIED
    assert cv._verdict_from_similarity(0.6)  == cv.SUSPICIOUS
    assert cv._verdict_from_similarity(0.2)  == cv.HALLUCINATED

    # Jaccard sanity
    assert cv.title_similarity("A B C", "A B C") == 1.0
    assert cv.title_similarity("", "anything")   == 0.0
    assert 0 < cv.title_similarity(
        "Global surface temperature trends since 1850",
        "Global temperature trend since 1850 preindustrial",
    ) < 1.0

    # CitationRecord → dict roundtrip
    r = cv.CitationRecord(
        marker=1, title="T", year=2020, doi="10.1/x",
        arxiv_id=None, metadata_source="crossref",
    )
    d = r.as_dict()
    for k in ("marker", "title", "verdict", "similarity", "notes",
              "external_title", "external_source"):
        assert k in d, f"CitationRecord.as_dict missing {k}"

    # CLI command registered
    callbacks = {c.callback for c in book_cli.app.registered_commands}
    assert book_cli.verify_citations in callbacks, (
        "`sciknow book verify-citations` not registered"
    )


def l1_phase46c_ensemble_review_surface() -> None:
    """Phase 46.C — ensemble NeurIPS-rubric review prompts + generator + CLI."""
    import inspect

    from sciknow.rag import prompts as rag_prompts
    from sciknow.core import book_ops
    from sciknow.cli import book as book_cli

    # Prompt functions exist, return (system, user)
    for fn in ("review_neurips", "review_meta"):
        assert hasattr(rag_prompts, fn), f"prompts missing {fn}"

    sys_n, usr_n = rag_prompts.review_neurips("results", "x", "draft", [])
    assert "soundness" in sys_n and "presentation" in sys_n
    assert "contribution" in sys_n and "overall" in sys_n
    assert "decision" in sys_n

    # Stance variants contain the expected override text
    sys_p, _ = rag_prompts.review_neurips("results", "x", "d", [],
                                           stance="pessimistic")
    assert "SKEPTICAL" in sys_p, "pessimistic stance must instruct skepticism"
    sys_o, _ = rag_prompts.review_neurips("results", "x", "d", [],
                                           stance="optimistic")
    assert "GENEROUS" in sys_o, "optimistic stance must reward grounded drafts"

    sys_m, usr_m = rag_prompts.review_meta(
        "results", "x",
        [{"overall": 6, "decision": "weak_accept"},
         {"overall": 5, "decision": "borderline"}],
    )
    assert "median" in sys_m.lower()
    assert "disagreement" in sys_m

    # Generator + helpers exposed
    for name in ("ensemble_review_stream", "_parse_review_json",
                 "_compute_meta_fallback", "_median"):
        assert hasattr(book_ops, name), f"book_ops missing {name}"

    # Mechanical fallback math sanity
    fb = book_ops._compute_meta_fallback([
        {"overall": 7, "soundness": 3, "presentation": 3, "contribution": 3,
         "confidence": 4, "decision": "weak_accept",
         "strengths": ["clear structure"], "weaknesses": ["narrow scope"]},
        {"overall": 4, "soundness": 2, "presentation": 3, "contribution": 2,
         "confidence": 3, "decision": "reject",
         "strengths": ["good citations"], "weaknesses": ["narrow scope", "math error"]},
        {"overall": 6, "soundness": 3, "presentation": 3, "contribution": 3,
         "confidence": 4, "decision": "weak_accept",
         "strengths": ["clear structure"], "weaknesses": ["narrow scope"]},
    ])
    assert fb["overall"] == 6, f"median of 7/4/6 should be 6, got {fb['overall']}"
    # "narrow scope" appeared in all 3 reviews → rank first
    assert fb["weaknesses"][0] == "narrow scope"
    assert 0.0 <= fb["disagreement"] <= 1.0
    assert fb["decision"] in {
        "strong_reject", "reject", "weak_reject", "borderline",
        "weak_accept", "accept", "strong_accept",
    }

    # Empty case: no reviews → _median returns None
    assert book_ops._median([]) is None
    assert book_ops._median([5.0]) == 5.0

    # CLI registered
    callbacks = {c.callback for c in book_cli.app.registered_commands}
    assert book_cli.ensemble_review in callbacks, (
        "`sciknow book ensemble-review` not registered"
    )
    # Consumer handles the ensemble-specific event types
    src = inspect.getsource(book_cli._consume_events)
    for token in ("reviewer_done", "meta_review_start"):
        assert token in src, f"_consume_events missing handler for {token!r}"


def l1_phase46e_web_expand_surface() -> None:
    """Phase 46.E — web expand surface: authors + domains + expand-author.

    Static checks. Verifies the three new endpoints register on the
    FastAPI app, and the HTML template mounts the Expand-by-Author
    panel + live author search JS hooks.
    """
    from sciknow.web import app as webapp

    # Endpoints
    route_paths = {r.path for r in webapp.app.routes if hasattr(r, "path")}
    for required in (
        "/api/catalog/authors",
        "/api/catalog/domains",
        "/api/corpus/expand-author",
    ):
        assert required in route_paths, (
            f"Phase 46.E web endpoint {required!r} not registered"
        )

    # Template renders the Expand-by-Author panel + search hooks
    tpl = webapp.TEMPLATE
    for token in (
        'id="corp-author-pane"',
        'id="tl-eauth-q"',
        'id="tl-eauth-results"',
        'id="tl-eauth-selected"',
        "onExpandAuthorSearchInput",
        "selectExpandAuthor",
        "switchCorpusTab",
        "loadCorpusTopicList",
        # Topics tab now also renders domains alongside clusters
        'id="tl-domains-list"',
    ):
        assert token in tpl, f"expand-UI: template missing {token!r}"

    # doToolCorpus knows about the three action values
    assert "action === 'expand-author'" in tpl, (
        "doToolCorpus must handle the expand-author action"
    )


def l1_phase46f_setup_wizard_surface() -> None:
    """Phase 46.F — end-to-end web setup wizard + its endpoints.

    Static checks: 5 new subprocess-backed endpoints + 1 inline
    endpoint + 1 aggregate-status endpoint are registered, and the
    HTML template mounts a 5-step wizard with expected DOM hooks.
    """
    from sciknow.web import app as webapp

    route_paths = {r.path for r in webapp.app.routes if hasattr(r, "path")}
    for required in (
        "/api/corpus/ingest-directory",
        "/api/corpus/upload",
        "/api/catalog/cluster",
        "/api/catalog/raptor/build",
        "/api/wiki/compile",
        "/api/book/create",
        "/api/setup/status",
    ):
        assert required in route_paths, (
            f"Phase 46.F endpoint {required!r} not registered"
        )

    tpl = webapp.TEMPLATE
    # Wizard modal + trail + 5 step panes exist
    assert 'id="setup-wizard-modal"' in tpl, "setup wizard modal missing"
    assert 'id="sw-trail"'            in tpl, "sw-trail missing"
    for step in ("project", "corpus", "indices", "expand", "book"):
        assert f'id="sw-step-{step}"' in tpl, f"wizard step pane {step} missing"
    # Toolbar button opens the wizard
    assert "openSetupWizard()" in tpl, "Setup toolbar button missing"
    # Each orchestration fn is wired
    for fn in (
        "swCreateProject", "swUseProject", "swLoadProjectsForWizard",
        "swUploadPDFs", "swIngestDirectory", "swRunIndex",
        "swCreateBook", "swAttachLogStream", "swRefreshStatus",
        "swGoto",
    ):
        assert fn in tpl, f"wizard JS fn {fn} missing"
    # CSS for the trail pills
    assert ".sw-step.active" in tpl, "wizard trail active style missing"


def l1_phase47_lesson_kind_scope() -> None:
    """Phase 47.1 — kind + scope on autowrite_lessons.

    Verifies the helpers (without hitting the DB — that's L2 material)
    and the _normalize_kind coercion across every canonical + weird
    input. Also asserts that _persist_lesson signature now accepts
    kind + scope kwargs.
    """
    import inspect

    from sciknow.core import book_ops

    assert hasattr(book_ops, "_normalize_kind"), "_normalize_kind missing"
    assert hasattr(book_ops, "_VALID_LESSON_KINDS"), "valid-kinds tuple missing"
    assert hasattr(book_ops, "_VALID_LESSON_SCOPES"), "valid-scopes tuple missing"

    # Every canonical kind round-trips cleanly
    for k in ("episode", "knowledge", "idea", "decision",
              "rejected_idea", "paper"):
        assert book_ops._normalize_kind(k) == k, f"kind {k} did not round-trip"
    # Weird inputs coerce sensibly
    assert book_ops._normalize_kind("REJECTED-IDEA") == "rejected_idea"
    assert book_ops._normalize_kind("Rejected Idea") == "rejected_idea"
    assert book_ops._normalize_kind(None)            == "episode"
    assert book_ops._normalize_kind("")              == "episode"
    assert book_ops._normalize_kind("bogus")         == "episode"

    # _persist_lesson takes kind + scope kwargs
    sig = inspect.signature(book_ops._persist_lesson)
    for p in ("kind", "scope"):
        assert p in sig.parameters, f"_persist_lesson missing {p!r} kwarg"

    # _get_relevant_lessons accepts the new consumer kwargs
    sig = inspect.signature(book_ops._get_relevant_lessons)
    for p in ("kinds", "include_global", "return_dicts"):
        assert p in sig.parameters, (
            f"_get_relevant_lessons missing {p!r} kwarg (Phase 47.1)"
        )

    # Distill prompt instructs the LLM to emit a `kind` field
    from sciknow.rag import prompts as rag_prompts
    sys_d, _ = rag_prompts.distill_lessons(
        section_slug="overview", final_overall=0.8, score_delta=0.2,
        iterations_used=2, converged=True, iterations=[],
    )
    assert "kind" in sys_d, "distill prompt doesn't ask for kind"
    assert "rejected_idea" in sys_d, "rejected_idea kind not documented"
    assert "knowledge" in sys_d and "idea" in sys_d and "decision" in sys_d


def l1_phase47_rejected_idea_gate() -> None:
    """Phase 47.2 — gaps prompt injects the rejected-idea block."""
    from sciknow.rag import prompts as rag_prompts

    # Without ideas: no block
    sys1, usr1 = rag_prompts.gaps("My Book", [], [], [])
    assert "Previously proposed ideas" not in usr1
    # With ideas: block + footer both present
    sys2, usr2 = rag_prompts.gaps("My Book", [], [], [],
        rejected_ideas=["Tried X but scored low",
                        "Tried Y (cite-heavy but thin)"])
    assert "Previously proposed ideas" in usr2
    assert "DO NOT re-propose" in usr2
    assert "Tried X but scored low" in usr2
    assert "Critically: do not re-surface" in usr2
    # gaps_json variant has the same injection
    sys3, usr3 = rag_prompts.gaps_json("My Book", [], [], [],
        rejected_ideas=["One rejected idea."])
    assert "Previously proposed" in usr3
    assert "One rejected idea" in usr3

    # Body caller queries kind='rejected_idea' — source grep
    import inspect as _inspect
    from sciknow.core import book_ops
    src = _inspect.getsource(book_ops.run_gaps_stream)
    assert "kind = 'rejected_idea'" in src, (
        "run_gaps_stream must query autowrite_lessons for "
        "kind='rejected_idea' before calling prompts.gaps"
    )
    assert "rejected_ideas=rejected_ideas" in src, (
        "run_gaps_stream must pass rejected_ideas to prompts.gaps*"
    )


def l1_phase47_kind_filtered_writer() -> None:
    """Phase 47.3 — writer prompt groups injected lessons by kind."""
    from sciknow.rag import prompts as rag_prompts

    # legacy string shape still works
    sys_a, _ = rag_prompts.write_section_v2(
        "results", "x", [], book_plan=None, prior_summaries=None,
        lessons=["Generic lesson one.", "Generic lesson two."],
    )
    assert "Lessons from prior runs" in sys_a

    # dict shape produces headered groups
    sys_b, _ = rag_prompts.write_section_v2(
        "results", "x", [], book_plan=None, prior_summaries=None,
        lessons=[
            {"text": "BP 1950 radiocarbon convention.", "kind": "knowledge"},
            {"text": "Narrow-to-broad framing worked.", "kind": "idea"},
            {"text": "DISCARD pivot on iteration 2.",  "kind": "decision"},
        ],
    )
    assert "Domain knowledge"    in sys_b, "knowledge header missing"
    assert "Positive ideas"      in sys_b, "idea header missing"
    assert "Precedent decisions" in sys_b, "decision header missing"

    # The autowrite caller asks _get_relevant_lessons for dicts +
    # excludes rejected_idea (source grep)
    import inspect as _inspect
    from sciknow.core import book_ops
    src = _inspect.getsource(book_ops._autowrite_section_body)
    assert "return_dicts=True" in src
    assert '"knowledge", "idea", "decision", "paper", "episode"' in src


def l1_phase47_promote_to_global_surface() -> None:
    """Phase 47.4 — promote-to-global helper + CLI."""
    import inspect

    from sciknow.core import book_ops
    from sciknow.cli import book as book_cli

    # Helper exists with the right signature
    assert callable(getattr(book_ops, "promote_lessons_to_global", None))
    sig = inspect.signature(book_ops.promote_lessons_to_global)
    for p in ("dry_run", "min_importance", "min_books",
              "cosine_threshold", "limit"):
        assert p in sig.parameters, (
            f"promote_lessons_to_global missing {p!r} kwarg"
        )
    # Thresholds sanity
    assert 0.0 <  book_ops._PROMOTE_MIN_IMPORTANCE <= 1.5
    assert book_ops._PROMOTE_MIN_BOOKS >= 2
    assert 0.5 <= book_ops._PROMOTE_COSINE <= 0.99

    # Cosine helper works end-to-end
    assert book_ops._cosine([1, 0, 0], [1, 0, 0]) == 1.0
    assert abs(book_ops._cosine([1, 0], [0, 1])) < 1e-9
    assert book_ops._cosine(None, [1, 0]) == 0.0
    assert book_ops._cosine([], [1, 2]) == 0.0

    # CLI command registered
    callbacks = {c.callback for c in book_cli.app.registered_commands}
    assert book_cli.promote_lessons in callbacks, (
        "`sciknow book promote-lessons` not registered"
    )


def l1_phase46g_benchmark_watchlist() -> None:
    """Phase 46.G — HF benchmark leaderboards on the watchlist surface.

    Static checks: parser, dataclass, public API, CLI command, seed
    list. No network.
    """
    import inspect as _inspect

    from sciknow.core import watchlist as wl
    from sciknow.cli import watch as watch_cli

    # Data types
    assert hasattr(wl, "WatchedBenchmark"), "WatchedBenchmark missing"
    b = wl.WatchedBenchmark(dataset="allenai/olmOCR-bench",
                             url="https://huggingface.co/datasets/allenai/olmOCR-bench")
    assert b.key == "allenai/olmOCR-bench"
    assert b.top_model_name is None
    assert b.top_changed_since_last_check is False
    # Regime change detection
    b.top_models = [{"rank": 1, "name": "Chandra-2", "score": 85.9}]
    b.prev_top_models = [{"rank": 1, "name": "olmOCR-2", "score": 82.4}]
    assert b.top_changed_since_last_check is True
    assert b.top_model_name == "Chandra-2"

    # Parse
    slug = wl.parse_hf_dataset_slug("https://huggingface.co/datasets/allenai/olmOCR-bench")
    assert slug == "allenai/olmOCR-bench"
    slug2 = wl.parse_hf_dataset_slug("allenai/olmOCR-bench")
    assert slug2 == "allenai/olmOCR-bench"
    try:
        wl.parse_hf_dataset_slug("not a slug")
    except ValueError:
        pass
    else:
        raise AssertionError("parse_hf_dataset_slug should reject non-slugs")

    # README table parser on a synthetic leaderboard
    synth = """
    # Some Benchmark
    intro text
    ## Leaderboard
    | **Model**          | Overall |
    |--------------------|---------|
    | top-model          | **86.7**|
    | middle-model       | 85.9    |
    | third              | 83.1 ± 1.1 |
    """
    parsed = wl._parse_readme_leaderboard(synth, top_n=3)
    assert len(parsed) == 3
    assert parsed[0]["name"] == "top-model"
    assert parsed[0]["score"] == 86.7
    assert parsed[1]["score"] == 85.9
    # CI stripped
    assert parsed[2]["score"] == 83.1

    # Public API: add_benchmark / check_benchmark / list_watched_benchmarks
    for name in ("add_benchmark", "remove_benchmark", "check_benchmark",
                 "check_all_benchmarks", "list_watched_benchmarks",
                 "seed_benchmarks_if_missing", "SEED_BENCHMARKS"):
        assert hasattr(wl, name), f"watchlist missing {name!r}"

    # Seed list has the expected slugs
    seed_slugs = {b["dataset"] for b in wl.SEED_BENCHMARKS}
    assert "allenai/olmOCR-bench"       in seed_slugs
    assert "opendatalab/OmniDocBench"   in seed_slugs

    # CLI
    callbacks = {c.callback for c in watch_cli.app.registered_commands}
    assert watch_cli.add_benchmark in callbacks, (
        "`sciknow watch add-benchmark` not registered"
    )


def l1_bench_harness_surface() -> None:
    """Phase 44 — the bench harness module loads and exposes the expected
    surface (run/run_layer/LAYERS/BenchMetric). Each layer has >= 1
    callable. Every bench function declared in LAYERS is zero-arg.

    Pure static — no DB, no Qdrant, no LLM. Catches accidental removal
    of `sciknow bench` wiring or a bench function that was decorated
    into accepting parameters.
    """
    import inspect

    from sciknow.testing import bench as bench_mod
    from sciknow.cli import main as main_cli

    # Surface
    for attr in ("run", "run_layer", "LAYERS", "BenchMetric", "BenchResult",
                 "write_results", "render_report", "diff_against_latest"):
        assert hasattr(bench_mod, attr), f"bench module missing {attr!r}"

    # Each layer populated + zero-arg bench fns
    layer_names = set(bench_mod.LAYERS.keys())
    for required in ("fast", "live", "llm", "full"):
        assert required in layer_names, f"bench layer {required!r} missing"

    for layer, entries in bench_mod.LAYERS.items():
        assert entries, f"bench layer {layer!r} is empty"
        for cat, fn in entries:
            assert isinstance(cat, str) and cat, f"{layer}: missing category"
            sig = inspect.signature(fn)
            assert len(sig.parameters) == 0, (
                f"{fn.__name__}: bench functions must be zero-arg "
                f"(got {list(sig.parameters)})"
            )

    # `sciknow bench` CLI registered on the Typer app
    cmd_names = {c.name for c in main_cli.app.registered_commands}
    assert "bench" in cmd_names, "`sciknow bench` not registered on CLI"


def l1_phase43h_web_project_endpoints() -> None:
    """Phase 43h — web GUI exposes project management.

    Static checks: the FastAPI app registers /api/projects endpoints
    (list/show/use/init/destroy), and the HTML template renders a
    Projects button + projects-modal container. Guards against
    accidental removal of either the endpoint surface or the UI hook.
    """
    import inspect

    from sciknow.web import app as webapp

    route_paths = {r.path for r in webapp.app.routes if hasattr(r, "path")}
    for required in (
        "/api/projects",
        "/api/projects/{slug}",
        "/api/projects/use",
        "/api/projects/init",
        "/api/projects/destroy",
    ):
        assert required in route_paths, (
            f"missing web endpoint {required!r}; "
            f"Projects GUI (Phase 43h) depends on it"
        )

    # The template should mount both the button and the modal, and
    # wire them through openProjectsModal() / refreshProjectsList().
    tpl = webapp.TEMPLATE
    assert "openProjectsModal" in tpl, (
        "toolbar should bind a Projects button to openProjectsModal()"
    )
    assert 'id="projects-modal"' in tpl, "projects-modal container missing"
    assert "refreshProjectsList" in tpl
    assert "destroyProject" in tpl
    assert "createProject" in tpl
    assert "useProject" in tpl

    # db stats should use the project-aware collection name so it
    # doesn't report N/A under a non-default project (Phase 43h bugfix).
    from sciknow.cli import db as db_cli
    stats_src = inspect.getsource(db_cli)
    assert 'get_collection("papers")' not in stats_src, (
        "db stats must not hardcode the 'papers' collection name — "
        "use storage.qdrant.papers_collection() instead"
    )


def l2_phase32_endpoint_shapes() -> None:
    """TestClient smoke test for the major read-only API endpoints.

    Phase 32. Hits each endpoint with TestClient against the live
    book and asserts: status 200, valid JSON, expected top-level
    keys present. Catches handler exceptions, accidentally-changed
    response shapes, and any endpoint that throws on the happy path.

    Skipped if no book exists in the DB. The endpoint set is
    deliberately read-only so this test is safe to run repeatedly.
    """
    from sciknow.testing.helpers import (
        get_test_client, a_book_id, a_chapter_id, a_draft_id,
    )

    bid = a_book_id()
    if not bid:
        return TestResult.skip(
            name="l2_phase32_endpoint_shapes",
            message="no book in DB — skipping endpoint shape test",
        )

    client = get_test_client()
    failures: list[str] = []

    def check(method: str, path: str, expected_keys: list[str]) -> None:
        try:
            if method == "GET":
                r = client.get(path)
            else:
                r = client.request(method, path)
        except Exception as exc:
            failures.append(f"{method} {path}: raised {type(exc).__name__}: {exc}")
            return
        if r.status_code != 200:
            failures.append(f"{method} {path}: status {r.status_code}")
            return
        try:
            body = r.json()
        except Exception:
            failures.append(f"{method} {path}: not JSON")
            return
        for key in expected_keys:
            if isinstance(body, dict) and key not in body:
                failures.append(f"{method} {path}: missing key {key!r}")
                return

    # Book / chapter / dashboard endpoints
    check("GET", "/api/book", ["id", "title", "chapters", "drafts"])
    check("GET", "/api/chapters", [])  # returns a list
    check("GET", "/api/dashboard", [])
    check("GET", "/api/corkboard", [])
    check("GET", "/api/catalog", [])
    check("GET", "/api/stats", [])
    check("GET", "/api/wiki/pages", [])
    check("GET", "/api/jobs", [])
    check("GET", "/api/kg", [])

    # Section + draft endpoints (need a real id)
    did = a_draft_id()
    if did:
        check("GET", f"/api/section/{did}", ["id", "title", "content_html"])
        check("GET", f"/api/versions/{did}", [])
        check("GET", f"/api/draft/{did}/scores", [])
        check("GET", f"/api/snapshots/{did}", [])

    # Chapter reader (needs a real chapter id)
    cid = a_chapter_id()
    if cid:
        check("GET", f"/api/chapter-reader/{cid}", [])

    assert not failures, (
        f"{len(failures)} endpoint failure(s): " + " | ".join(failures[:5])
    )


def l2_phase32_6_autowrite_telemetry_roundtrip() -> None:
    """Phase 32.6 — end-to-end smoke test of the Layer 0 telemetry path
    against the live PG database.

    Creates a fake autowrite_runs row, persists synthetic retrievals
    and an iteration, finalizes the run with a draft that contains
    `[1] [3] [5]` markers, and verifies that:
      - the run row transitioned to status='completed'
      - the iteration row was upserted with action='KEEP'
      - exactly the cited source_positions have was_cited=true

    Cleans up after itself so the test is idempotent and safe to
    re-run. Skipped if the chunks table is empty (fresh DB).
    """
    from sqlalchemy import text as _t
    from sciknow.storage.db import get_session
    from sciknow.core import book_ops

    # Need at least 6 chunks with qdrant_point_id for the synthetic test
    with get_session() as s:
        n_chunks = s.execute(_t(
            "SELECT count(*) FROM chunks WHERE qdrant_point_id IS NOT NULL"
        )).scalar()
        if (n_chunks or 0) < 6:
            return TestResult.skip(
                name="l2_phase32_6_autowrite_telemetry_roundtrip",
                message=f"need ≥6 chunks, have {n_chunks} — skipping",
            )

    # 1) Create a synthetic draft with [1] [3] [5] markers
    with get_session() as s:
        draft = s.execute(_t("""
            INSERT INTO drafts (title, content, sources, version)
            VALUES ('phase32.6 telemetry roundtrip', 'foo [1] bar [3] baz [5].',
                    '[]'::jsonb, 1)
            RETURNING id::text
        """)).fetchone()
        draft_id = draft[0]
        s.commit()

    try:
        # 2) Open the run
        run_id = book_ops._create_autowrite_run(
            book_id=None, chapter_id=None, section_slug="l2_test",
            model="test:phase32.6", target_words=1500, max_iter=3,
            target_score=0.85, feature_versions={"smoke": True},
        )
        assert run_id, "_create_autowrite_run returned None"

        # 3) Persist 6 synthetic retrievals using real chunk ids
        with get_session() as s:
            chunk_rows = s.execute(_t("""
                SELECT qdrant_point_id::text, document_id::text
                FROM chunks WHERE qdrant_point_id IS NOT NULL LIMIT 6
            """)).fetchall()

        class _FakeResult:
            def __init__(self, rank, chunk_id, document_id, score):
                self.rank = rank
                self.chunk_id = chunk_id
                self.document_id = document_id
                self.score = score

        fakes = [
            _FakeResult(i + 1, r[0], r[1], 0.9 - i * 0.1)
            for i, r in enumerate(chunk_rows)
        ]
        book_ops._persist_autowrite_retrievals(run_id, fakes)

        # 4) Persist an iteration row, then update it with KEEP
        hist_entry = {
            "iteration": 1,
            "scores": {"overall": 0.65, "groundedness": 0.7,
                       "weakest_dimension": "length"},
            "verification": {"groundedness_score": 0.7},
            "cove": {"ran": False},
            "revision_verdict": None,
            "post_revision_overall": None,
        }
        book_ops._persist_autowrite_iteration(
            run_id, 1, hist_entry,
            word_count=800, word_count_delta=None, overall_pre=0.65,
        )
        hist_entry["revision_verdict"] = "KEEP"
        hist_entry["post_revision_overall"] = 0.78
        book_ops._persist_autowrite_iteration(
            run_id, 1, hist_entry,
            word_count=1300, word_count_delta=500, overall_pre=0.65,
        )

        # 5) Finalize with the draft pointer — triggers was_cited back-fill
        book_ops._finalize_autowrite_run(
            run_id, status="completed", final_draft_id=draft_id,
            final_overall=0.78, iterations_used=1, converged=False,
        )

        # 6) Verify the persisted state
        with get_session() as s:
            run = s.execute(_t("""
                SELECT status, final_overall, iterations_used, converged,
                       finished_at IS NOT NULL
                FROM autowrite_runs WHERE id::text = :id
            """), {"id": run_id}).fetchone()
            assert run is not None, "run row missing after finalize"
            assert run[0] == "completed", f"status={run[0]}, expected 'completed'"
            assert abs(float(run[1]) - 0.78) < 0.01, f"final_overall={run[1]}"
            assert run[2] == 1, f"iterations_used={run[2]}"
            assert run[4] is True, "finished_at not set"

            iters = s.execute(_t("""
                SELECT action, word_count, word_count_delta
                FROM autowrite_iterations WHERE run_id::text = :id
                ORDER BY iteration
            """), {"id": run_id}).fetchall()
            assert len(iters) == 1, f"expected 1 iteration row, got {len(iters)}"
            assert iters[0][0] == "KEEP", f"action={iters[0][0]}"
            assert iters[0][1] == 1300, f"word_count={iters[0][1]}"
            assert iters[0][2] == 500, f"word_count_delta={iters[0][2]}"

            cited_positions = s.execute(_t("""
                SELECT source_position FROM autowrite_retrievals
                WHERE run_id::text = :id AND was_cited = true
                ORDER BY source_position
            """), {"id": run_id}).fetchall()
            cited_set = {r[0] for r in cited_positions}
            assert cited_set == {1, 3, 5}, (
                f"was_cited back-fill wrong: got {sorted(cited_set)}, "
                "expected {1, 3, 5}"
            )
    finally:
        # Always clean up — even on assertion failure — so the test is
        # idempotent and re-runnable.
        with get_session() as s:
            if run_id:
                s.execute(_t("DELETE FROM autowrite_runs WHERE id::text = :id"),
                          {"id": run_id})
            s.execute(_t("DELETE FROM drafts WHERE id::text = :id"),
                      {"id": draft_id})
            s.commit()

    return TestResult.ok(
        name="l2_phase32_6_autowrite_telemetry_roundtrip",
        message="6 retrievals + 1 iteration + finalize roundtripped, was_cited correct",
    )


def l2_phase32_7_lessons_roundtrip() -> None:
    """Phase 32.7 — Layer 1 roundtrip against live PG.

    Persists 3 lessons with real bge-m3 embeddings, retrieves them with
    two different queries, and asserts the cosine similarity ranking
    correctly disambiguates: a length-related query surfaces the
    length lesson; a citation-related query surfaces the citation lesson.

    Skipped if the embedder can't load (no GPU + no CPU fallback).
    Cleans up after itself so the test is idempotent.
    """
    from sqlalchemy import text as _t
    from sciknow.storage.db import get_session
    from sciknow.core import book_ops

    test_slug = "l2_lessons_test"

    # Verify the embedder works (and warm it). Skip cleanly if it can't load.
    try:
        probe = book_ops._embed_text_for_lessons("smoke test")
        if not probe or len(probe) != 1024:
            return TestResult.skip(
                name="l2_phase32_7_lessons_roundtrip",
                message=f"embedder returned dim={len(probe) if probe else 0}",
            )
    except Exception as exc:
        return TestResult.skip(
            name="l2_phase32_7_lessons_roundtrip",
            message=f"embedder unavailable: {str(exc)[:80]}",
        )

    # Pick a real book id so the FK is happy
    with get_session() as s:
        b = s.execute(_t("SELECT id::text FROM books LIMIT 1")).fetchone()
        book_id = b[0] if b else None

    lesson_length = (
        "Drafts under 70% of target_words consistently fail to converge — "
        "start with the right length anchor."
    )
    lesson_citations = (
        "For methods sections, citation density above 1 per 80 words "
        "correlates with higher groundedness scores."
    )
    lesson_hedging = (
        "When the scoring loop oscillates between groundedness and length, "
        "fix hedging fidelity first."
    )

    try:
        # Persist all three with real embeddings
        for ls, dim, imp in [
            (lesson_length, "length", 1.3),
            (lesson_citations, "citation_accuracy", 1.1),
            (lesson_hedging, "hedging_fidelity", 1.2),
        ]:
            emb = book_ops._embed_text_for_lessons(ls)
            book_ops._persist_lesson(
                book_id=book_id, chapter_id=None, section_slug=test_slug,
                lesson_text=ls, source_run_id=None, score_delta=0.15,
                embedding=emb, importance=imp, dimension=dim,
            )

        # Verify they were persisted
        with get_session() as s:
            n = s.execute(_t(
                "SELECT count(*) FROM autowrite_lessons WHERE section_slug = :s"
            ), {"s": test_slug}).scalar()
            assert n == 3, f"expected 3 lessons, found {n}"

        # Length-related query → length lesson should rank #1
        length_top = book_ops._get_relevant_lessons(
            book_id, test_slug,
            "I'm writing a section that's too short, how do I hit the target length?",
            top_k=3,
        )
        assert len(length_top) == 3, f"expected 3 results, got {len(length_top)}"
        assert "length anchor" in length_top[0], (
            f"length query should rank length lesson #1, got: {length_top[0]!r}"
        )

        # Citation-related query → citation lesson should rank #1
        cite_top = book_ops._get_relevant_lessons(
            book_id, test_slug,
            "How do I improve groundedness with better citations?",
            top_k=3,
        )
        assert len(cite_top) == 3, f"expected 3 results, got {len(cite_top)}"
        assert "citation density" in cite_top[0], (
            f"citation query should rank citation lesson #1, got: {cite_top[0]!r}"
        )

        # Cold-start: empty query on an unknown section returns []
        empty = book_ops._get_relevant_lessons(
            book_id, "definitely_not_a_real_slug_xyz", "anything",
        )
        assert empty == [], f"unknown section should return [], got {empty}"
    finally:
        with get_session() as s:
            s.execute(_t(
                "DELETE FROM autowrite_lessons WHERE section_slug = :s"
            ), {"s": test_slug})
            s.commit()

    return TestResult.ok(
        name="l2_phase32_7_lessons_roundtrip",
        message="3 lessons persisted, similarity ranking correctly disambiguated",
    )


def l2_phase32_8_useful_boost_roundtrip() -> None:
    """Phase 32.8 — Layer 2: end-to-end useful_count boost test.

    Strategy:
      1. Run a baseline hybrid search and pick a chunk at rank ~10
         (mid-list, where a boost has room to move it).
      2. Insert N fake autowrite_runs each citing that chunk in their
         final draft (was_cited=true).
      3. Re-run the same search and verify:
         - the target chunk's score increased
         - the score increase matches the formula:
             new = old × (1 + 0.15 × log2(1 + N))
         - useful_count was correctly populated on the candidate
         - the rank either stayed the same or moved up (never down)
      4. Cleanup all fake rows in a finally block.

    Skipped if the embedder can't load.
    """
    import math
    from sqlalchemy import text as _t
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client
    from sciknow.retrieval import hybrid_search
    from sciknow.config import settings

    # Make sure the embedder works (it's used by hybrid_search internally)
    try:
        from sciknow.retrieval.hybrid_search import _embed_query
        _embed_query("smoke test")
    except Exception as exc:
        return TestResult.skip(
            name="l2_phase32_8_useful_boost_roundtrip",
            message=f"embedder unavailable: {str(exc)[:80]}",
        )

    qclient = get_client()
    n_fake_runs = 10
    fake_slug = "l2_useful_boost_test"

    # 1) Baseline search
    with get_session() as s:
        baseline = hybrid_search.search(
            query="ocean heat content trends",
            qdrant_client=qclient,
            session=s,
            candidate_k=20,
        )
    if len(baseline) < 12:
        return TestResult.skip(
            name="l2_phase32_8_useful_boost_roundtrip",
            message=f"only {len(baseline)} candidates — need ≥12 for the rank-10 test",
        )

    # Pick a target at rank 10 (0-indexed: position 9). Mid-list so the
    # boost has room to move it. We require that the target NOT already
    # have a useful_count from real autowrite history — pick a clean
    # one. Walk down from rank 10 if needed.
    target = None
    for c in baseline[9:]:
        if c.useful_count == 0:
            target = c
            break
    if target is None:
        return TestResult.skip(
            name="l2_phase32_8_useful_boost_roundtrip",
            message="no clean (useful_count=0) candidate found at rank ≥10",
        )
    baseline_score = target.rrf_score
    baseline_chunk_id = target.chunk_id

    # 2) Insert N fake runs citing this chunk
    try:
        with get_session() as s:
            book = s.execute(_t("SELECT id::text FROM books LIMIT 1")).fetchone()
            book_id = book[0] if book else None
            for _ in range(n_fake_runs):
                row = s.execute(_t("""
                    INSERT INTO autowrite_runs (book_id, section_slug, status)
                    VALUES (CAST(:bid AS uuid), :slug, 'completed')
                    RETURNING id::text
                """), {"bid": book_id, "slug": fake_slug}).fetchone()
                s.execute(_t("""
                    INSERT INTO autowrite_retrievals (
                        run_id, source_position, chunk_qdrant_id, was_cited
                    ) VALUES (
                        CAST(:rid AS uuid), 1, CAST(:cid AS uuid), true
                    )
                """), {"rid": row[0], "cid": baseline_chunk_id})
            s.commit()

        # 3) Re-run the same search
        with get_session() as s:
            boosted = hybrid_search.search(
                query="ocean heat content trends",
                qdrant_client=qclient,
                session=s,
                candidate_k=20,
            )

        new_target = next(
            (c for c in boosted if c.chunk_id == baseline_chunk_id), None
        )
        assert new_target is not None, (
            "target chunk dropped out of the candidate set after boost — "
            "should be impossible since the boost only INCREASES scores"
        )

        # Score must have increased
        assert new_target.rrf_score > baseline_score, (
            f"score did not increase: baseline={baseline_score:.5f}, "
            f"boosted={new_target.rrf_score:.5f}"
        )

        # Score should match the formula
        expected = baseline_score * (
            1.0 + settings.useful_count_boost_factor * math.log2(1 + n_fake_runs)
        )
        actual = new_target.rrf_score
        # Allow 2% tolerance for floating-point + any concurrent state changes
        assert abs(actual - expected) / expected < 0.02, (
            f"boosted score {actual:.5f} doesn't match expected {expected:.5f} "
            f"(formula: baseline × (1 + factor × log2(1 + N)))"
        )

        # useful_count must be populated (n_fake_runs distinct runs)
        assert new_target.useful_count == n_fake_runs, (
            f"useful_count={new_target.useful_count}, expected {n_fake_runs}"
        )

        # Rank should never go DOWN (boost is monotonically positive)
        baseline_rank = next(
            i for i, c in enumerate(baseline, 1) if c.chunk_id == baseline_chunk_id
        )
        new_rank = next(
            i for i, c in enumerate(boosted, 1) if c.chunk_id == baseline_chunk_id
        )
        assert new_rank <= baseline_rank, (
            f"rank went DOWN: {baseline_rank} → {new_rank}"
        )
    finally:
        # Always clean up so the test is idempotent
        with get_session() as s:
            s.execute(_t(
                "DELETE FROM autowrite_runs WHERE section_slug = :s"
            ), {"s": fake_slug})
            s.commit()

    return TestResult.ok(
        name="l2_phase32_8_useful_boost_roundtrip",
        message=(
            f"useful_count={n_fake_runs} → score "
            f"{baseline_score:.5f}→{actual:.5f} (rank {baseline_rank}→{new_rank})"
        ),
    )


def l2_phase32_9_dpo_export_roundtrip() -> None:
    """Phase 32.9 — Layer 4: end-to-end DPO export against live PG.

    Inserts 3 fake completed runs (KEEP, KEEP+DISCARD, low-score) with
    pre/post-revision content populated. Runs the export and verifies:
    - Filter rules drop the low-score row
    - KEEP and DISCARD verdicts both produce pairs
    - DISCARD pairs have chosen/rejected correctly inverted
    - JSONL records have all expected fields
    """
    import json as _json
    import tempfile
    from pathlib import Path
    from sqlalchemy import text as _t
    from sciknow.storage.db import get_session
    from sciknow.core import book_ops

    with get_session() as s:
        b = s.execute(_t("SELECT id::text FROM books LIMIT 1")).fetchone()
        if not b:
            return TestResult.skip(
                name="l2_phase32_9_dpo_export_roundtrip",
                message="no books in DB — skipping",
            )
        book_id = b[0]

    fake_slug = "l2_dpo_test"
    out = Path(tempfile.mktemp(suffix=".jsonl"))
    try:
        with get_session() as s:
            # Run A: 1 KEEP iteration with passing scores
            run_a = s.execute(_t("""
                INSERT INTO autowrite_runs (book_id, section_slug, status, model)
                VALUES (CAST(:bid AS uuid), :slug, 'completed', 'test')
                RETURNING id::text
            """), {"bid": book_id, "slug": fake_slug}).fetchone()
            s.execute(_t("""
                INSERT INTO autowrite_iterations (
                    run_id, iteration, scores, action, overall_pre, overall_post,
                    pre_revision_content, post_revision_content
                ) VALUES (
                    CAST(:rid AS uuid), 1, CAST('{}' AS jsonb), 'KEEP',
                    0.72, 0.85, 'PRE-A', 'POST-A'
                )
            """), {"rid": run_a[0]})

            # Run B: 1 DISCARD iteration with passing scores
            run_b = s.execute(_t("""
                INSERT INTO autowrite_runs (book_id, section_slug, status, model)
                VALUES (CAST(:bid AS uuid), :slug, 'completed', 'test')
                RETURNING id::text
            """), {"bid": book_id, "slug": fake_slug}).fetchone()
            s.execute(_t("""
                INSERT INTO autowrite_iterations (
                    run_id, iteration, scores, action, overall_pre, overall_post,
                    pre_revision_content, post_revision_content
                ) VALUES (
                    CAST(:rid AS uuid), 1, CAST('{}' AS jsonb), 'DISCARD',
                    0.78, 0.71, 'PRE-B', 'POST-B'
                )
            """), {"rid": run_b[0]})

            # Run C: 1 KEEP with low scores (should be filtered out)
            run_c = s.execute(_t("""
                INSERT INTO autowrite_runs (book_id, section_slug, status, model)
                VALUES (CAST(:bid AS uuid), :slug, 'completed', 'test')
                RETURNING id::text
            """), {"bid": book_id, "slug": fake_slug}).fetchone()
            s.execute(_t("""
                INSERT INTO autowrite_iterations (
                    run_id, iteration, scores, action, overall_pre, overall_post,
                    pre_revision_content, post_revision_content
                ) VALUES (
                    CAST(:rid AS uuid), 1, CAST('{}' AS jsonb), 'KEEP',
                    0.45, 0.55, 'PRE-C', 'POST-C'
                )
            """), {"rid": run_c[0]})
            s.commit()

        # Run the export
        n, path = book_ops._export_preference_pairs(
            book_id=book_id, output_path=out,
            min_score=0.7, min_delta=0.02,
        )
        assert n == 2, f"expected 2 pairs (KEEP + DISCARD; low-score skipped), got {n}"

        # Parse and verify
        records = []
        with out.open() as f:
            for line in f:
                records.append(_json.loads(line))
        assert len(records) == 2, f"expected 2 JSONL records, got {len(records)}"

        # Find the KEEP and DISCARD records
        keeps = [r for r in records if r["verdict"] == "KEEP"]
        discards = [r for r in records if r["verdict"] == "DISCARD"]
        assert len(keeps) == 1, f"expected 1 KEEP, got {len(keeps)}"
        assert len(discards) == 1, f"expected 1 DISCARD, got {len(discards)}"

        # KEEP: chosen = post, rejected = pre
        keep = keeps[0]
        assert keep["chosen"] == "POST-A", f"KEEP chosen should be POST-A, got {keep['chosen']!r}"
        assert keep["rejected"] == "PRE-A", f"KEEP rejected should be PRE-A, got {keep['rejected']!r}"
        assert keep["score_chosen"] > keep["score_rejected"], (
            "KEEP record has chosen score lower than rejected — wrong direction"
        )

        # DISCARD: chosen = pre, rejected = post (inverse)
        disc = discards[0]
        assert disc["chosen"] == "PRE-B", f"DISCARD chosen should be PRE-B (inverse), got {disc['chosen']!r}"
        assert disc["rejected"] == "POST-B", f"DISCARD rejected should be POST-B (inverse), got {disc['rejected']!r}"
        assert disc["score_chosen"] > disc["score_rejected"], (
            "DISCARD record has chosen score lower than rejected — inversion broken"
        )

        # All required fields present
        for rec in records:
            for field in ("prompt", "chosen", "rejected", "score_chosen",
                          "score_rejected", "score_delta", "verdict",
                          "section_slug", "iteration", "run_id"):
                assert field in rec, f"missing field {field!r} in record"
    finally:
        try:
            out.unlink()
        except Exception:
            pass
        with get_session() as s:
            s.execute(_t(
                "DELETE FROM autowrite_runs WHERE section_slug = :s"
            ), {"s": fake_slug})
            s.commit()

    return TestResult.ok(
        name="l2_phase32_9_dpo_export_roundtrip",
        message="2 pairs (KEEP + inverted DISCARD), 1 low-score correctly filtered",
    )


def l2_phase32_10_style_fingerprint_roundtrip() -> None:
    """Phase 32.10 — Layer 5: end-to-end style fingerprint roundtrip
    against live PG.

    Creates a temporary book, inserts 3 approved drafts with known
    style features, runs compute_style_fingerprint, and verifies:
    - The fingerprint is persisted to books.custom_metadata.style_fingerprint
    - get_style_fingerprint reads it back identically
    - The metrics roughly match what we put in
    - format_fingerprint_for_prompt produces a non-empty block
    Cleans up the temporary book in a finally block.
    """
    from sqlalchemy import text as _t
    from sciknow.storage.db import get_session
    from sciknow.core.style_fingerprint import (
        compute_style_fingerprint, get_style_fingerprint,
        format_fingerprint_for_prompt,
    )

    fake_book_title = "phase32.10 fingerprint roundtrip test"
    book_id = None

    # Sample drafts — same content gives a deterministic fingerprint
    drafts = [
        ("draft-1",
         "The first sentence has six words. The second has five words here. "
         "Recent work suggests three [1] to four key findings [2]. "
         "However, the magnitude is contested.\n\nThis paragraph adds more "
         "context. The mechanism appears robust [3]."),
        ("draft-2",
         "Ocean heat content trends indicate a steady rise [4]. The signal "
         "may suggest anthropogenic forcing. However, internal variability "
         "could account for some fraction [5].\n\nMoreover, deep ocean uptake "
         "tends to dominate the budget."),
        ("draft-3",
         "Volcanic eruptions inject SO2 into the stratosphere [6]. The 1991 "
         "Pinatubo event caused measurable cooling [7]. Importantly, the "
         "decay timescale is approximately 18 months."),
    ]

    try:
        # 1) Create a temp book + 3 approved drafts
        with get_session() as s:
            row = s.execute(_t("""
                INSERT INTO books (title, status) VALUES (:t, 'draft')
                RETURNING id::text
            """), {"t": fake_book_title}).fetchone()
            book_id = row[0]
            for title, content in drafts:
                s.execute(_t("""
                    INSERT INTO drafts (
                        title, book_id, content, status, version, sources
                    ) VALUES (
                        :title, CAST(:bid AS uuid), :content, 'final',
                        1, '[]'::jsonb
                    )
                """), {"title": title, "bid": book_id, "content": content})
            s.commit()

        # 2) Compute the fingerprint
        fp = compute_style_fingerprint(book_id)
        assert fp["n_drafts_sampled"] == 3, (
            f"expected 3 drafts sampled, got {fp['n_drafts_sampled']}"
        )

        # 3) Verify it persisted to books.custom_metadata
        with get_session() as s:
            meta = s.execute(_t(
                "SELECT custom_metadata FROM books WHERE id::text = :id"
            ), {"id": book_id}).scalar()
        assert isinstance(meta, dict), f"custom_metadata not a dict: {type(meta)}"
        assert "style_fingerprint" in meta, (
            "style_fingerprint not persisted to books.custom_metadata"
        )

        # 4) get_style_fingerprint reads it back
        fp_read = get_style_fingerprint(book_id)
        assert fp_read is not None, "get_style_fingerprint returned None after compute"
        assert fp_read["n_drafts_sampled"] == 3
        assert fp_read["citations_per_100_words"] == fp["citations_per_100_words"]

        # 5) Metrics are sane
        assert fp["median_sentence_length"] > 0, "median_sentence_length is 0"
        assert fp["citations_per_100_words"] > 0, (
            "citations_per_100_words is 0 — citation regex broken"
        )
        assert fp["hedging_rate"] > 0, (
            "hedging_rate is 0 — hedge cue list not detecting drafts that contain "
            "'may', 'suggest', 'tends', 'could', 'approximately'"
        )
        # We have at least one transition: "however" or "moreover" or "importantly"
        transitions = fp.get("top_transitions") or []
        assert len(transitions) > 0, (
            f"no transitions detected — {transitions}"
        )

        # 6) Format for prompt produces a non-empty block with the right keywords
        block = format_fingerprint_for_prompt(fp)
        assert "Match the established style" in block
        assert "median sentence length" in block
        assert "citation density" in block
    finally:
        # Always clean up
        with get_session() as s:
            if book_id:
                s.execute(_t(
                    "DELETE FROM drafts WHERE book_id::text = :id"
                ), {"id": book_id})
                s.execute(_t(
                    "DELETE FROM books WHERE id::text = :id"
                ), {"id": book_id})
            s.commit()

    return TestResult.ok(
        name="l2_phase32_10_style_fingerprint_roundtrip",
        message=(
            f"3 drafts → median_sent={fp['median_sentence_length']}w, "
            f"hedging={fp['hedging_rate']:.0%}, "
            f"cites/100w={fp['citations_per_100_words']}"
        ),
    )


def l2_phase32_data_invariants() -> None:
    """DB-level invariants the GUI relies on.

    Phase 32. Catches the kind of orphan-record state that makes
    parts of the GUI silently disappear: drafts pointing at deleted
    chapters, chapters with no section template, drafts with empty
    content blocks rendering as ghosts in the sidebar, etc.
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    failures: list[str] = []

    with get_session() as session:
        # 1) Drafts whose chapter_id doesn't exist in book_chapters
        orphans = session.execute(text("""
            SELECT d.id::text FROM drafts d
            LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.chapter_id IS NOT NULL AND bc.id IS NULL
            LIMIT 5
        """)).fetchall()
        if orphans:
            failures.append(
                f"{len(orphans)} draft(s) with dangling chapter_id: "
                f"{[r[0] for r in orphans]}"
            )

        # 2) Chapters with NULL or empty title (sidebar would render
        #    them as gaps)
        nameless = session.execute(text("""
            SELECT id::text FROM book_chapters
            WHERE title IS NULL OR title = ''
            LIMIT 5
        """)).fetchall()
        if nameless:
            failures.append(
                f"{len(nameless)} chapter(s) with empty title: "
                f"{[r[0] for r in nameless]}"
            )

        # 3) Drafts with NULL or empty section_type (sidebar can't
        #    render them — they show as 'unknown')
        sectionless = session.execute(text("""
            SELECT id::text FROM drafts
            WHERE section_type IS NULL OR section_type = ''
            LIMIT 5
        """)).fetchall()
        # This one is a soft warning — log it but don't fail, since
        # legacy single-shot drafts may have empty section_type.
        # Phase 32 just exposes the count for visibility.
        if sectionless and len(sectionless) > 5:
            failures.append(
                f"{len(sectionless)}+ drafts with NULL section_type "
                f"(legacy data?)"
            )

        # 4) Comments pointing at deleted drafts
        dangling_comments = session.execute(text("""
            SELECT dc.id::text FROM draft_comments dc
            LEFT JOIN drafts d ON d.id = dc.draft_id
            WHERE d.id IS NULL
            LIMIT 5
        """)).fetchall()
        if dangling_comments:
            failures.append(
                f"{len(dangling_comments)} comment(s) on deleted drafts: "
                f"{[r[0] for r in dangling_comments]}"
            )

    assert not failures, "; ".join(failures)


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
    # Phase 23 — collapse/expand chapter sections in sidebar
    l1_phase23_chapter_collapse_expand,
    # Phase 24 — autowrite progress log with heartbeat
    l1_phase24_autowrite_logger,
    # Phase 25 — visible chevron + adopt orphan sections
    l1_phase25_chevron_visible,
    l1_phase25_adopt_orphan_section,
    # Phase 26 — drag-and-drop section reordering
    l1_phase26_section_drag_drop,
    # Phase 27 — display title derived from chapter sections meta
    l1_phase27_display_title_from_meta,
    # Phase 28 — autowrite resume mode
    l1_phase28_is_resumable_draft,
    l1_phase28_resume_wired_through,
    # Phase 29 — ROADMAP + per-section size + click-to-preview
    l1_phase29_roadmap_doc_exists,
    l1_phase29_per_section_target_words,
    l1_phase29_size_dropdown_in_modal,
    l1_phase29_empty_section_preview_not_write,
    # Phase 30 — task bar + heatmap + export + KG
    l1_phase30_persistent_task_bar,
    l1_phase30_heatmap_numbered_columns,
    l1_phase30_export_endpoints,
    l1_phase30_kg_endpoint,
    # Phase 31 — six fixes (custom dropdown, PDF, KG graph, Read filter, Edit toolbar)
    l1_phase31_custom_dropdown_persists,
    l1_phase31_pdf_export,
    l1_phase31_kg_graph_view,
    l1_phase31_read_button_section_filter,
    l1_phase31_edit_button_in_toolbar,
    # Phase 32 — QA module overhaul (helpers + endpoint inventory + JS
    # handler integrity + render helper escape + sync handler audit)
    l1_phase32_qa_helpers_module,
    l1_phase32_endpoint_inventory,
    l1_phase32_js_handler_integrity,
    l1_phase32_render_helpers_escape_chain,
    l1_phase32_no_global_state_leak,
    l1_phase32_endpoint_handler_signatures_consistent,
    # Phase 32.1 — per-section target visible + persisted across reopens
    l1_phase32_1_section_target_visible_and_loaded,
    # Phase 32.2 — per-section length dropdown in the Plan modal
    l1_phase32_2_plan_modal_per_section_length,
    # Phase 32.3 — task bar must be position: fixed (not sticky)
    l1_phase32_3_task_bar_is_fixed_not_sticky,
    # Phase 32.4 — inline section delete + add from the sidebar
    l1_phase32_4_section_delete_and_add_in_sidebar,
    # Phase 32.5 — task bar polls server-side stats (no SSE competition)
    l1_phase32_5_task_bar_polls_stats_no_sse_competition,
    # Phase 32.6 — Layer 0 of compound learning: autowrite telemetry
    l1_phase32_6_autowrite_telemetry_layer0,
    # Phase 32.7 — Layer 1: episodic memory store (lessons)
    l1_phase32_7_lessons_layer1,
    # Phase 32.8 — Layer 2: useful_count retrieval boost
    l1_phase32_8_useful_count_boost_layer2,
    # Phase 32.9 — Layer 4: DPO preference dataset export
    l1_phase32_9_dpo_export_layer4,
    # Phase 32.10 — Layer 5: style fingerprint extraction
    l1_phase32_10_style_fingerprint_layer5,
    # Phase 33 — keyboard shortcuts + chapter drag-drop + log rotation
    l1_phase33_keyboard_shortcuts_and_polish,
    # Phase 34 — CARS rhetorical moves in tree plan + writer prompt
    l1_phase34_cars_rhetorical_moves,
    # Phase 35 — book-level GPU compute counter on the dashboard
    l1_phase35_total_compute_counter,
    # Phase 36 — Tools modal: CLI-parity panel (search / synthesize /
    #   topics / corpus enrich + expand)
    l1_phase36_tools_panel,
    # Phase 37 — per-section model override in chapter meta
    l1_phase37_per_section_model_override,
    # Phase 38 — chapter + book snapshot bundles (autowrite safety net)
    l1_phase38_scoped_snapshot_bundles,
    # Phase 39 — consolidated per-book Settings modal
    l1_phase39_book_settings_modal,
    # Phase 40 — CLI export parity: pdf + epub
    l1_phase40_cli_export_pdf_epub,
    # Phase 41 — static WHERE clauses on /api/catalog + /api/kg
    l1_phase41_static_where_clauses,
    # Phase 42 — data-action dispatcher (retire interpolated onclicks)
    l1_phase42_data_action_dispatcher,
    # Phase 43 — multi-project foundations (project resolution, project-
    # aware DB / Qdrant / paths, sciknow project CLI, --project flag)
    l1_phase43_project_resolution,
    # Phase 43h — web GUI project management + db stats project-aware bugfix
    l1_phase43h_web_project_endpoints,
    # Phase 44 — bench harness (`sciknow bench`)
    l1_bench_harness_surface,
    # Phase 45 — project types + watchlist
    l1_phase45_project_types,
    l1_phase45_watchlist_surface,
    # Phase 46 — auditable scientific writing (citation insert + verify +
    # ensemble review + expand-by-author web surface + end-to-end wizard)
    l1_phase46_citation_insert_surface,
    l1_phase46_citation_verify_surface,
    l1_phase46c_ensemble_review_surface,
    l1_phase46e_web_expand_surface,
    l1_phase46f_setup_wizard_surface,
    # Phase 47 — typed compound memory (DeepScientist port bundle)
    l1_phase47_lesson_kind_scope,
    l1_phase47_rejected_idea_gate,
    l1_phase47_kind_filtered_writer,
    l1_phase47_promote_to_global_surface,
    # Phase 46.G — HF benchmark watchlist
    l1_phase46g_benchmark_watchlist,
]

L2_TESTS: list[Callable] = [
    l2_postgres_reachable,
    l2_qdrant_reachable,
    l2_papers_collection_exists,
    l2_db_stats_query,
    l2_ensure_node_level_index_idempotent,
    l2_qdrant_papers_count,
    l2_hybrid_search_smoke,
    # Phase 32 — TestClient endpoint shapes + DB invariants
    l2_phase32_endpoint_shapes,
    l2_phase32_data_invariants,
    # Phase 32.6 — Layer 0 telemetry roundtrip against live PG
    l2_phase32_6_autowrite_telemetry_roundtrip,
    # Phase 32.7 — Layer 1 lessons roundtrip + similarity ranking
    l2_phase32_7_lessons_roundtrip,
    # Phase 32.8 — Layer 2 useful_count boost roundtrip in real search
    l2_phase32_8_useful_boost_roundtrip,
    # Phase 32.9 — Layer 4 DPO export with KEEP/DISCARD inversion + filters
    l2_phase32_9_dpo_export_roundtrip,
    # Phase 32.10 — Layer 5 style fingerprint compute + persist + read
    l2_phase32_10_style_fingerprint_roundtrip,
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
