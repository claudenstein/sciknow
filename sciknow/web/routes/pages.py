"""``sciknow.web.routes.pages`` — SPA-routed page endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. 17 page
handlers that share a common pattern: each emits the standard book
reader template with one specific modal pre-opened (so the URL is
shareable and browser back/forward traverses modal state).

Cross-module deps (resolved via lazy `_app` shim per handler):
  `_app._get_book_data`, `_app._render_book`, `_app._routed_view`.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


def _route_modal(modal_id: str) -> HTMLResponse:
    """Lazy `_app._routed_view` wrapper. Each modal route just calls
    this with its modal id."""
    from sciknow.web import app as _app
    return _app._routed_view(modal_id)


@router.get("/section/{draft_id}", response_class=HTMLResponse)
async def section(draft_id: str):
    from sciknow.web import app as _app
    book, chapters, drafts, gaps, comments = _app._get_book_data()
    return HTMLResponse(_app._render_book(book, chapters, drafts, gaps, comments, focus_draft=draft_id))


@router.get("/plan", response_class=HTMLResponse)
async def route_plan():
    return _route_modal("plan-modal")


@router.get("/settings", response_class=HTMLResponse)
async def route_settings():
    return _route_modal("book-settings-modal")


@router.get("/wiki", response_class=HTMLResponse)
async def route_wiki():
    return _route_modal("wiki-modal")


@router.get("/bundles", response_class=HTMLResponse)
async def route_bundles():
    return _route_modal("bundles-modal")


@router.get("/tools", response_class=HTMLResponse)
async def route_tools():
    return _route_modal("tools-modal")


@router.get("/projects", response_class=HTMLResponse)
async def route_projects():
    return _route_modal("projects-modal")


@router.get("/catalog", response_class=HTMLResponse)
async def route_catalog():
    return _route_modal("catalog-modal")


@router.get("/export", response_class=HTMLResponse)
async def route_export():
    return _route_modal("export-modal")


@router.get("/corpus", response_class=HTMLResponse)
async def route_corpus():
    return _route_modal("corpus-modal")


@router.get("/visualize", response_class=HTMLResponse)
async def route_visualize():
    return _route_modal("viz-modal")


@router.get("/kg", response_class=HTMLResponse)
async def route_kg():
    return _route_modal("kg-modal")


@router.get("/ask", response_class=HTMLResponse)
async def route_ask():
    return _route_modal("ask-modal")


@router.get("/setup", response_class=HTMLResponse)
async def route_setup():
    return _route_modal("setup-modal")


@router.get("/backups", response_class=HTMLResponse)
async def route_backups():
    return _route_modal("backups-modal")


@router.get("/visuals", response_class=HTMLResponse)
async def route_visuals():
    return _route_modal("visuals-modal")


@router.get("/help", response_class=HTMLResponse)
async def route_help():
    return _route_modal("ai-help-modal")
