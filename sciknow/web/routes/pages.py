"""``sciknow.web.routes.pages`` — pages endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. Behaviour
unchanged. Cross-module deps resolved via the standard lazy
`from sciknow.web import app as _app` shim.
"""
from __future__ import annotations

import asyncio
import json
import threading

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response, StreamingResponse, PlainTextResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

router = APIRouter()


@router.get("/section/{draft_id}", response_class=HTMLResponse)
async def section(draft_id: str):
    from sciknow.web import app as _app
    book, chapters, drafts, gaps, comments = _app._get_book_data()
    return HTMLResponse(_app._render_book(book, chapters, drafts, gaps, comments, focus_draft=draft_id))


@router.get("/plan",      response_class=HTMLResponse)
async def route_plan():      return _routed_view("plan-modal")


@router.get("/settings",  response_class=HTMLResponse)
async def route_settings():  return _routed_view("book-settings-modal")


@router.get("/wiki",      response_class=HTMLResponse)
async def route_wiki():      return _routed_view("wiki-modal")


@router.get("/bundles",   response_class=HTMLResponse)
async def route_bundles():   return _routed_view("bundles-modal")


@router.get("/tools",     response_class=HTMLResponse)
async def route_tools():     return _routed_view("tools-modal")


@router.get("/projects",  response_class=HTMLResponse)
async def route_projects():  return _routed_view("projects-modal")


@router.get("/catalog",   response_class=HTMLResponse)
async def route_catalog():   return _routed_view("catalog-modal")


@router.get("/export",    response_class=HTMLResponse)
async def route_export():    return _routed_view("export-modal")


@router.get("/corpus",    response_class=HTMLResponse)
async def route_corpus():    return _routed_view("corpus-modal")


@router.get("/visualize", response_class=HTMLResponse)
async def route_visualize(): return _routed_view("viz-modal")


@router.get("/kg",        response_class=HTMLResponse)
async def route_kg():        return _routed_view("kg-modal")


@router.get("/ask",       response_class=HTMLResponse)
async def route_ask():       return _routed_view("ask-modal")


@router.get("/setup",     response_class=HTMLResponse)
async def route_setup():     return _routed_view("setup-modal")


@router.get("/backups",   response_class=HTMLResponse)
async def route_backups():   return _routed_view("backups-modal")


@router.get("/visuals",   response_class=HTMLResponse)
async def route_visuals():   return _routed_view("visuals-modal")


@router.get("/help",      response_class=HTMLResponse)
async def route_help():      return _routed_view("ai-help-modal")
