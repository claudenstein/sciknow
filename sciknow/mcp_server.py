"""Phase 54.6.77 (#16) — MCP (Model Context Protocol) server.

Exposes the sciknow corpus as a set of MCP tools so external LLM
agents (Claude Desktop, Claude Code, goose, etc.) can query it
directly. Runs over stdio — the caller launches this process and
speaks MCP over its stdin/stdout.

Tools exposed
-------------

``search_corpus(query, k)``
    Hybrid search (dense + sparse + FTS, RRF-fused). Returns the top-k
    chunks with chunk id, document id, section, and a short preview.

``ask_corpus(question, k)``
    Full RAG — retrieve + prompt the local LLM, return the answer
    with citations. Uses settings.llm_model.

``list_chapters(book_title_prefix)``
    Book outline so a calling agent can orient itself.

``get_paper_summary(slug_prefix)``
    Fetches a compiled wiki paper-summary page by slug.

The server is registered as ``sciknow mcp-serve`` in the CLI. Configure
in Claude Desktop / etc. by pointing ``command`` at the uv entry:

    {
      "mcpServers": {
        "sciknow": {
          "command": "uv",
          "args": ["run", "sciknow", "mcp-serve"],
          "cwd": "/path/to/sciknow/repo"
        }
      }
    }
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Tool implementations (small wrappers around existing sciknow helpers)
# ════════════════════════════════════════════════════════════════════════


def _search_corpus(query: str, k: int = 10) -> dict:
    from sciknow.retrieval.hybrid_search import search as _search
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client as _get_qdrant

    client = _get_qdrant()
    with get_session() as session:
        cands = _search(query, client, session, candidate_k=max(20, k))
    cands = cands[:k]

    hits: list[dict] = []
    for c in cands:
        hits.append({
            "chunk_id": getattr(c, "chunk_id", ""),
            "document_id": getattr(c, "document_id", ""),
            "section": getattr(c, "section_type", "") or "",
            "title": getattr(c, "title", "") or "",
            "year": getattr(c, "year", None),
            "score": round(float(getattr(c, "rrf_score", 0.0)), 4),
            "preview": (getattr(c, "content_preview", "") or "")[:400],
        })
    return {"query": query, "k": k, "hits": hits}


def _ask_corpus(question: str, k: int = 12) -> dict:
    from sciknow.retrieval.hybrid_search import search as _search
    from sciknow.rag.llm import complete as _complete
    from sciknow.rag import prompts as _prompts
    from sciknow.retrieval.context_builder import build as _build_ctx
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client as _get_qdrant

    client = _get_qdrant()
    with get_session() as session:
        cands = _search(question, client, session, candidate_k=max(30, k * 2))
        if not cands:
            return {"question": question, "answer": "", "sources": [],
                    "note": "no retrieval hits"}
        results = _build_ctx(cands[:k], session)
    sys_p, usr_p = _prompts.ask(question, results)
    try:
        raw = _complete(sys_p, usr_p, temperature=0.1, num_ctx=16384,
                        keep_alive=-1)
    except Exception as exc:
        return {"question": question, "answer": "",
                "sources": [], "error": str(exc)}
    sources = [
        {
            "n": i + 1,
            "title": getattr(r, "title", "") or "",
            "year": getattr(r, "year", None),
            "doi": getattr(r, "doi", "") or "",
            "section": getattr(r, "section_type", "") or "",
        }
        for i, r in enumerate(results)
    ]
    return {"question": question, "answer": raw, "sources": sources}


def _list_chapters(book_title_prefix: str = "") -> dict:
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        if book_title_prefix.strip():
            bk = session.execute(text(
                "SELECT id::text, title FROM books "
                "WHERE title ILIKE :p LIMIT 1"
            ), {"p": f"%{book_title_prefix.strip()}%"}).fetchone()
            if not bk:
                return {"error": f"book not found matching {book_title_prefix!r}"}
            book_id = bk[0]
            book_title = bk[1]
        else:
            bk = session.execute(text(
                "SELECT id::text, title FROM books "
                "ORDER BY created_at DESC LIMIT 1"
            )).fetchone()
            if not bk:
                return {"error": "no books exist yet"}
            book_id = bk[0]
            book_title = bk[1]
        rows = session.execute(text("""
            SELECT number, title, description, topic_query
            FROM book_chapters
            WHERE book_id::text = :bid ORDER BY number
        """), {"bid": book_id}).fetchall()
    return {
        "book": book_title,
        "chapters": [
            {"number": r[0], "title": r[1] or "",
             "description": r[2] or "", "topic_query": r[3] or ""}
            for r in rows
        ],
    }


def _get_paper_summary(slug_prefix: str) -> dict:
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    from sciknow.core.wiki_ops import show_page
    slug_prefix = (slug_prefix or "").strip()
    if not slug_prefix:
        return {"error": "slug_prefix is required"}
    with get_session() as session:
        row = session.execute(text("""
            SELECT slug, title, page_type FROM wiki_pages
            WHERE slug LIKE :q LIMIT 1
        """), {"q": f"{slug_prefix}%"}).fetchone()
    if not row:
        return {"error": f"no wiki page matching {slug_prefix!r}"}
    try:
        page = show_page(row[0])
    except Exception as exc:
        return {"error": str(exc)}
    if not page:
        return {"error": "wiki page content missing"}
    return {
        "slug": row[0], "title": row[1] or "",
        "page_type": row[2] or "",
        "content": (page.get("content") or "")[:8000],
    }


# ════════════════════════════════════════════════════════════════════════
# Tool schema registry
# ════════════════════════════════════════════════════════════════════════


_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_corpus",
        "description": (
            "Hybrid search across the sciknow paper corpus "
            "(dense + sparse + full-text, RRF-fused). Returns the top-k "
            "most relevant chunks with paper title, year, section, and "
            "preview. Use when you need retrieval-quality passages "
            "rather than generated prose."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Search query."},
                "k": {"type": "integer", "default": 10,
                      "minimum": 1, "maximum": 50,
                      "description": "How many hits to return."},
            },
            "required": ["query"],
        },
        "handler": lambda args: _search_corpus(
            args.get("query", ""), int(args.get("k", 10))
        ),
    },
    {
        "name": "ask_corpus",
        "description": (
            "Ask a question and get a RAG-generated answer backed by the "
            "sciknow corpus. Uses hybrid retrieval + the local LLM. "
            "Returns the answer with numbered citations you can "
            "cross-reference via search_corpus."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "k": {"type": "integer", "default": 12,
                      "minimum": 3, "maximum": 30,
                      "description": "How many passages to give the LLM."},
            },
            "required": ["question"],
        },
        "handler": lambda args: _ask_corpus(
            args.get("question", ""), int(args.get("k", 12))
        ),
    },
    {
        "name": "list_chapters",
        "description": (
            "List the chapter outline for a book in this sciknow "
            "project. Empty book_title_prefix returns the most "
            "recently-created book."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "book_title_prefix": {"type": "string", "default": ""},
            },
        },
        "handler": lambda args: _list_chapters(
            args.get("book_title_prefix", "")
        ),
    },
    {
        "name": "get_paper_summary",
        "description": (
            "Fetch a compiled wiki paper-summary (or concept, or "
            "synthesis) page by slug prefix. Use after search_corpus "
            "points at an interesting paper and you want the LLM-"
            "compiled digest instead of the raw chunks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "slug_prefix": {"type": "string"},
            },
            "required": ["slug_prefix"],
        },
        "handler": lambda args: _get_paper_summary(args.get("slug_prefix", "")),
    },
]


# ════════════════════════════════════════════════════════════════════════
# Server bootstrap
# ════════════════════════════════════════════════════════════════════════


def _build_mcp_server():
    """Construct the MCP server with all tools registered."""
    from mcp.server import Server
    from mcp.types import Tool, TextContent

    server = Server("sciknow")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in _TOOLS
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[TextContent]:
        tool = next((t for t in _TOOLS if t["name"] == name), None)
        if tool is None:
            return [TextContent(type="text",
                                text=json.dumps({"error": f"unknown tool {name!r}"}))]
        try:
            result = tool["handler"](arguments or {})
        except Exception as exc:
            logger.exception("tool %r failed", name)
            result = {"error": str(exc)}
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str),
        )]

    return server


async def serve_stdio() -> None:
    """Run the MCP server over stdio until the caller closes the pipes."""
    from mcp.server.stdio import stdio_server
    server = _build_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            server.create_initialization_options(),
        )
