"""
Phase 32 — shared helpers for the L1/L2/L3 testing protocol.

The QA protocol started as ad-hoc grep-based tests that lived inline
in `protocol.py`. As coverage grew (60+ tests by Phase 31) the
inline-import-import-everywhere pattern was making each test ~30
lines of boilerplate before the actual assertion. This module
extracts the common pieces:

- ``get_test_client(book_id=...)`` — TestClient with the global
  book wired up so endpoints that read ``_book_id`` see the right
  one. Cached at module level so multiple tests share one client.
- ``a_book_id()`` / ``a_chapter_id()`` / ``a_draft_id()`` — return
  identifiers for the first book/chapter/draft in the live DB.
  Tests can use these to hit endpoints without hardcoding UUIDs.
- ``inspect_handler_source(name)`` — return the source of any
  named function from ``sciknow.web.app``. Centralised to avoid
  20 ``import inspect; from sciknow.web import app as web_app;``
  blocks.
- ``rendered_template()`` — return the full rendered HTML+JS+CSS
  template once, cached, so a test can grep it without rendering it
  freshly each time.
- ``js_function_definitions()`` / ``js_onclick_handlers()`` — parse
  the rendered template into the set of defined and referenced JS
  function names so handler-integrity tests can diff them.

None of these import services that need PG/Qdrant/Ollama at module
load time — they're all callable from L1 tests.
"""
from __future__ import annotations

import inspect
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional


# ── Test client + identifiers ────────────────────────────────────────


@lru_cache(maxsize=1)
def get_test_client():
    """Return a FastAPI TestClient with the global book id wired up.

    Cached so the same client is reused across all tests in a single
    `sciknow test` invocation. The client is read-only — tests should
    not POST destructive endpoints without cleaning up after.
    """
    from fastapi.testclient import TestClient
    from sqlalchemy import text
    from sciknow.web.app import app, set_book
    from sciknow.storage.db import get_session

    with get_session() as session:
        row = session.execute(
            text("SELECT id::text, title FROM books LIMIT 1")
        ).fetchone()
    if row:
        set_book(row[0], row[1])
    return TestClient(app)


@lru_cache(maxsize=1)
def a_book_id() -> Optional[str]:
    """Return the id of the first book in the DB, or None."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        row = session.execute(
            text("SELECT id::text FROM books LIMIT 1")
        ).fetchone()
    return row[0] if row else None


@lru_cache(maxsize=1)
def a_chapter_id() -> Optional[str]:
    """Return the id of the first chapter in the DB, or None."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        row = session.execute(text("""
            SELECT id::text FROM book_chapters
            ORDER BY created_at LIMIT 1
        """)).fetchone()
    return row[0] if row else None


@lru_cache(maxsize=1)
def a_draft_id() -> Optional[str]:
    """Return the id of the first draft with content > 100 words."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        row = session.execute(text("""
            SELECT id::text FROM drafts
            WHERE word_count > 100
            ORDER BY created_at DESC LIMIT 1
        """)).fetchone()
    return row[0] if row else None


# ── Source inspection helpers ────────────────────────────────────────


def inspect_handler_source(name: str) -> str:
    """Return the source of a named function from sciknow.web.app.

    Used by tests that want to assert specific lines/strings exist
    in a handler without 5 lines of boilerplate. Raises AttributeError
    if the function doesn't exist (which is itself a useful test
    failure mode).
    """
    from sciknow.web import app as web_app
    fn = getattr(web_app, name)
    return inspect.getsource(fn)


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def web_app_full_source() -> str:
    """Return the full source of sciknow.web.app as a single string.

    Cached. Used by tests that grep across the whole module instead
    of a specific function (template content, CSS rules, etc).

    v2 Phase E: as the 31 kLOC f-string template is decomposed into
    Jinja2 partials + static assets, this helper appends the contents
    of every static CSS / JS file under ``sciknow/web/static/`` so
    grep-style tests for CSS class names + JS function names still
    see the same surface they did pre-refactor. The contract that
    matters ("does the page emit class X / function Y") is preserved.
    """
    from sciknow.web import app as web_app
    src = inspect.getsource(web_app)
    chunks = [src]

    # CSS + JS under web/static/ — re-doubled to mimic the f-string
    # source form (see comment block above each rewrite).
    static_dir = Path(__file__).resolve().parents[1] / "web" / "static"
    if static_dir.exists():
        for ext in ("css", "js"):
            for p in sorted(static_dir.rglob(f"*.{ext}")):
                try:
                    body = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                # v1-era L1 tests greped the f-string template, so
                # CSS + JS braces appear doubled (`{{` / `}}`). Re-double
                # when injecting the static file content so those
                # assertions keep passing without per-test edits.
                # (v2 Phase E — JS extraction shipped 2026-04-25; without
                # this re-double, ~29 plan-modal / monitor tests asserting
                # `{{}}` source-form would fail against the now-rendered
                # `{}` in /static/js/sciknow.js.)
                if ext in ("css", "js"):
                    body = body.replace("{", "{{").replace("}", "}}")
                # JS extraction also collapsed `\\` → `\` (the rendered
                # browser form). Tests that asserted the source-form
                # double-backslash (e.g. `[.\\\\)]` regex char-class
                # markers, `\\'plan\\'` quoted strings) need that form
                # back. Re-double after the brace fix so test grep
                # patterns still hit.
                if ext == "js":
                    body = body.replace("\\", "\\\\")
                chunks.append(
                    f"\n# --- web/static/{p.relative_to(static_dir)} ---\n" + body
                )

    # HTML templates under web/templates/ — v2 Phase E extracted the
    # 3.8 kLOC TEMPLATE string into book_reader.html. The template
    # uses Python `.format()`, so `{var}` placeholders and `{{`/`}}`
    # literal braces in the *source* form are preserved verbatim in
    # the file. But Python source escapes (`\\` → `\`, `\'` → `'`)
    # were applied during extraction. Tests that grep for the source
    # form (e.g. raw assertions checking for `\\'foo\\'` patterns in
    # rendered onclick handlers) need those re-doubled — same logic
    # as the JS extraction above.
    templates_dir = Path(__file__).resolve().parents[1] / "web" / "templates"
    if templates_dir.exists():
        for p in sorted(templates_dir.rglob("*.html")):
            try:
                body = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            # Re-double backslashes to mirror Python source-form so
            # tests that asserted `\\'plan\\'`-style patterns still
            # match. Braces in templates are already in source form
            # (the file is read verbatim, .format() never applied),
            # so don't touch those — leave `{var}` and `{{`/`}}` as-is.
            body = body.replace("\\", "\\\\")
            chunks.append(
                f"\n# --- web/templates/{p.relative_to(templates_dir)} ---\n" + body
            )

    # Resource-scoped route modules under web/routes/ — v2 Phase E
    # split. Tests that grep for `@app.get("/api/X")` etc. need the
    # extracted handlers visible too. We append the source verbatim
    # (no re-escape) so any `@router.X` patterns are findable, and
    # tests that look for both forms (`@app.X` OR `@router.X`) keep
    # working.
    routes_dir = Path(__file__).resolve().parents[1] / "web" / "routes"
    if routes_dir.exists():
        for p in sorted(routes_dir.rglob("*.py")):
            if p.name == "__init__.py":
                continue
            try:
                body = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            chunks.append(
                f"\n# --- web/routes/{p.relative_to(routes_dir)} ---\n" + body
            )

    return "".join(chunks)


# ── Rendered template helpers ────────────────────────────────────────


@lru_cache(maxsize=1)
def rendered_template_static() -> str:
    """Render the TEMPLATE f-string with placeholder values. No DB.

    L1-friendly: substitutes safe placeholder values for every
    interpolation slot so the template renders without touching
    Postgres/Qdrant/Ollama. Used by JS-integrity tests that need the
    expanded `{var}` substitutions but don't care about real data.

    v2 Phase E: appends the contents of every static CSS / JS file
    under ``sciknow/web/static/`` so source-grep tests for CSS class
    names continue to find them after the templates were
    externalised. The contract is "does the page emit X" — the
    static link tag plus the file content satisfy it.
    """
    from sciknow.web.app import TEMPLATE, _BUILD_TAG
    rendered = TEMPLATE.format(
        _BUILD_TAG=_BUILD_TAG,
        book_title="Test Book",
        # Phase 38 — `book_id` placeholder for the bundle-snapshot JS
        # that references `/api/snapshot/book/{book_id}` directly in
        # the rendered client code.
        book_id="00000000-0000-0000-0000-000000000000",
        search_q="",
        search_results_html="",
        sidebar_html="",
        gaps_count=0,
        active_id="abc12345-6789-0000-0000-000000000000",
        active_title="Test",
        active_version=1,
        active_words=100,
        active_chapter_id="ch1",
        active_section_type="intro",
        chapters_json="[]",
        content_html="<p>test</p>",
        sources_html="",
        review_html="",
        comments_html="",
        # Phase 54.6.178 — routed-views auto-open script slot.
        auto_open_script="",
    )
    static_dir = Path(__file__).resolve().parents[1] / "web" / "static"
    if static_dir.exists():
        parts = [rendered]
        for ext in ("css", "js"):
            for p in sorted(static_dir.rglob(f"*.{ext}")):
                try:
                    body = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                # CSS braces re-doubled to satisfy v1-era greps that
                # expected the f-string template syntax. See the
                # web_app_full_source() docstring.
                if ext == "css":
                    body = body.replace("{", "{{").replace("}", "}}")
                parts.append(
                    f"\n<!-- static/{p.relative_to(static_dir)} -->\n" + body
                )
        return "".join(parts)
    return rendered


@lru_cache(maxsize=1)
def rendered_template_with_data() -> str:
    """Render the full HTML+JS+CSS template with real DB data. L2 only.

    Cached. Used by tests that need the dynamically-built sidebar/
    heatmap/source chunks rendered with real book data. Requires PG.
    """
    from sciknow.web.app import _render_book, _get_book_data, set_book

    bid = a_book_id()
    if bid:
        # Need a title too — fetch directly to avoid coupling
        from sqlalchemy import text
        from sciknow.storage.db import get_session
        with get_session() as session:
            row = session.execute(text(
                "SELECT id::text, title FROM books WHERE id::text = :bid"
            ), {"bid": bid}).fetchone()
        if row:
            set_book(row[0], row[1])

    book, chapters, drafts, gaps, comments = _get_book_data()
    return _render_book(book, chapters, drafts, gaps, comments)


# ── JS-template parsers ──────────────────────────────────────────────


# Function definitions: `function name(` or `async function name(`
_FN_DEF_RE = re.compile(r"\b(?:async\s+)?function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
# onclick handlers: `onclick="name(` or `onclick='name('
# (we accept both quote styles plus the &quot; entity form)
_ONCLICK_RE = re.compile(
    r'on(?:click|input|change|keydown|submit|mouseenter|mouseleave|dragstart|dragover|drop|dragend|dragleave)='
    r'["\'](?:event\.[a-zA-Z]+\(\);)*\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\('
)


@lru_cache(maxsize=1)
def js_function_definitions() -> set[str]:
    """Return the set of JS function names defined anywhere in the
    sciknow.web.app module source.

    Operates on the module source (not a rendered template) so it's
    L1-safe — no DB needed. The TEMPLATE f-string is part of the
    source, so all template-defined functions are included. Helpers
    that build HTML chunks (e.g. _render_sources) are also covered.
    """
    src = web_app_full_source()
    return set(_FN_DEF_RE.findall(src))


@lru_cache(maxsize=1)
def js_onclick_handlers() -> set[str]:
    """Return the set of JS function names referenced from
    on{click,input,change,keydown,submit,mouseenter,mouseleave,
    dragstart,dragover,drop,dragend,dragleave} attributes anywhere
    in the sciknow.web.app module source.

    L1-safe (no DB). Operates on the inspect-based source so it
    catches handlers in TEMPLATE, in `_render_*` helpers, and in any
    other HTML-building function. The regex skips known leading
    event.X(); calls (e.g. event.stopPropagation();) so the captured
    name is the actual handler being called, not the event method.
    """
    src = web_app_full_source()
    return set(_ONCLICK_RE.findall(src))


# ── Endpoint inventory ───────────────────────────────────────────────


@lru_cache(maxsize=1)
def all_app_routes() -> list[tuple[str, set[str]]]:
    """Return [(path, methods), ...] for every registered FastAPI
    route in sciknow.web.app. Includes the SSE stream endpoint
    and any websocket if added later.
    """
    from sciknow.web.app import app
    out = []
    for r in app.routes:
        path = getattr(r, "path", None)
        methods = getattr(r, "methods", None)
        if path and methods:
            out.append((path, set(methods)))
    return out


def find_route(path: str) -> Optional[tuple[str, set[str]]]:
    """Look up registered routes by exact path. Returns
    (path, union_of_methods) or None.

    FastAPI registers each (path, method) pair as its own route entry,
    so a single endpoint like ``/api/book`` may appear once for GET and
    once for PUT. This helper unions methods across all matching
    entries so callers can ask "is this method registered for this
    path?" with a single lookup.
    """
    matched_methods: set[str] = set()
    for p, m in all_app_routes():
        if p == path:
            matched_methods |= m
    if not matched_methods:
        return None
    return (path, matched_methods)
