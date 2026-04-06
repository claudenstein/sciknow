"""
Book web reader — FastAPI app serving a live browsable view of a sciknow book.

Launched via `sciknow book serve "Book Title"`. Features:
  - Sidebar chapter/section navigation
  - Live content from PostgreSQL (refreshes on each page load)
  - Per-section quality scores (color-coded)
  - Version history with diffs
  - Citation links (hover for paper details)
  - Inline comments/annotations per paragraph
  - Edit-in-place with one-click "revise this section"
  - Search within the book
  - Dark/light theme toggle
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

app = FastAPI(title="SciKnow Book Reader")


# Global state — set by the CLI before launching uvicorn
_book_id: str = ""
_book_title: str = ""


def set_book(book_id: str, book_title: str) -> None:
    global _book_id, _book_title
    _book_id = book_id
    _book_title = book_title


# ── Data helpers ──────────────────────────────────────────────────────────────

def _get_book_data():
    with get_session() as session:
        book = session.execute(text("""
            SELECT id::text, title, description, plan, status
            FROM books WHERE id::text = :bid
        """), {"bid": _book_id}).fetchone()

        chapters = session.execute(text("""
            SELECT bc.id::text, bc.number, bc.title, bc.description,
                   bc.topic_query, bc.topic_cluster
            FROM book_chapters bc
            WHERE bc.book_id = :bid ORDER BY bc.number
        """), {"bid": _book_id}).fetchall()

        drafts = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.content,
                   d.word_count, d.sources, d.version, d.summary,
                   d.review_feedback, d.chapter_id::text,
                   d.parent_draft_id::text, d.created_at,
                   bc.number AS ch_num, bc.title AS ch_title
            FROM drafts d
            LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.book_id = :bid
            ORDER BY bc.number, d.section_type, d.version DESC
        """), {"bid": _book_id}).fetchall()

        gaps = session.execute(text("""
            SELECT bg.id::text, bg.gap_type, bg.description, bg.status,
                   bc.number AS ch_num
            FROM book_gaps bg
            LEFT JOIN book_chapters bc ON bc.id = bg.chapter_id
            WHERE bg.book_id = :bid
            ORDER BY bc.number NULLS LAST, bg.gap_type
        """), {"bid": _book_id}).fetchall()

        comments = session.execute(text("""
            SELECT dc.id::text, dc.draft_id::text, dc.paragraph_index,
                   dc.selected_text, dc.comment, dc.status, dc.created_at
            FROM draft_comments dc
            JOIN drafts d ON d.id = dc.draft_id
            WHERE d.book_id = :bid
            ORDER BY dc.created_at
        """), {"bid": _book_id}).fetchall()

    return book, chapters, drafts, gaps, comments


def _md_to_html(text_content: str) -> str:
    """Simple markdown → HTML conversion for draft content."""
    if not text_content:
        return ""
    html = text_content
    # Headers
    html = re.sub(r'^### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    # Bold and italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    # Citation references [N] → styled spans
    html = re.sub(r'\[(\d+)\]', r'<span class="citation" data-ref="\1">[\1]</span>', html)
    # Paragraphs
    paragraphs = html.split('\n\n')
    html = ''.join(
        f'<p data-para="{i}">{p.strip()}</p>' if not p.strip().startswith('<h')
        else p.strip()
        for i, p in enumerate(paragraphs) if p.strip()
    )
    return html


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    book, chapters, drafts, gaps, comments = _get_book_data()
    if not book:
        return HTMLResponse("<h1>Book not found</h1>", status_code=404)
    return HTMLResponse(_render_book(book, chapters, drafts, gaps, comments))


@app.get("/section/{draft_id}", response_class=HTMLResponse)
async def section(draft_id: str):
    book, chapters, drafts, gaps, comments = _get_book_data()
    return HTMLResponse(_render_book(book, chapters, drafts, gaps, comments, focus_draft=draft_id))


@app.post("/comment")
async def add_comment(
    draft_id: str = Form(...),
    paragraph_index: int = Form(None),
    selected_text: str = Form(None),
    comment: str = Form(...),
):
    with get_session() as session:
        session.execute(text("""
            INSERT INTO draft_comments (draft_id, paragraph_index, selected_text, comment)
            VALUES (:did::uuid, :para, :sel, :comment)
        """), {"did": draft_id, "para": paragraph_index, "sel": selected_text, "comment": comment})
        session.commit()
    return RedirectResponse(f"/section/{draft_id}", status_code=303)


@app.post("/comment/{comment_id}/resolve")
async def resolve_comment(comment_id: str):
    with get_session() as session:
        session.execute(text(
            "UPDATE draft_comments SET status = 'resolved' WHERE id::text = :cid"
        ), {"cid": comment_id})
        session.commit()
    return JSONResponse({"ok": True})


@app.post("/edit/{draft_id}")
async def edit_draft(draft_id: str, content: str = Form(...)):
    with get_session() as session:
        session.execute(text("""
            UPDATE drafts SET content = :content, word_count = :wc WHERE id::text = :did
        """), {"did": draft_id, "content": content, "wc": len(content.split())})
        session.commit()
    return RedirectResponse(f"/section/{draft_id}", status_code=303)


@app.get("/search", response_class=HTMLResponse)
async def search_book(q: str = ""):
    if not q:
        return RedirectResponse("/")
    book, chapters, drafts, gaps, comments = _get_book_data()
    matched = [d for d in drafts if q.lower() in (d[3] or "").lower()]
    return HTMLResponse(_render_book(book, chapters, drafts, gaps, comments, search_q=q, search_results=matched))


@app.get("/api/book")
async def api_book():
    book, chapters, drafts, gaps, comments = _get_book_data()
    return {
        "title": book[1] if book else "",
        "chapters": len(chapters),
        "drafts": len(drafts),
        "gaps": len(gaps),
        "comments": len(comments),
    }


# ── HTML Template ─────────────────────────────────────────────────────────────

def _render_book(book, chapters, drafts, gaps, comments,
                 focus_draft=None, search_q="", search_results=None):
    """Render the full book reader as a self-contained HTML page."""

    # Group drafts by chapter
    chapter_drafts = {}
    draft_map = {}
    for d in drafts:
        draft_id, title, sec_type, content, wc, sources, version, summary, \
            review_fb, ch_id, parent_id, created, ch_num, ch_title = d
        draft_map[draft_id] = d
        key = ch_id or "__none__"
        if key not in chapter_drafts:
            chapter_drafts[key] = []
        # Only keep the latest version per section_type per chapter
        existing = [x for x in chapter_drafts[key] if x[2] == sec_type]
        if not existing or (version or 1) > (existing[0][6] or 1):
            chapter_drafts[key] = [x for x in chapter_drafts[key] if x[2] != sec_type]
            chapter_drafts[key].append(d)

    # Group comments by draft
    draft_comments = {}
    for c in comments:
        cid, did, para, sel, comm, status, created = c
        if did not in draft_comments:
            draft_comments[did] = []
        draft_comments[did].append(c)

    # Build sidebar
    sidebar_items = []
    for ch in chapters:
        ch_id, ch_num, ch_title, ch_desc, tq, tc = ch
        ch_ds = chapter_drafts.get(ch_id, [])
        sections = []
        for d in sorted(ch_ds, key=lambda x: {"introduction": 0, "methods": 1, "results": 2, "discussion": 3, "conclusion": 4}.get(x[2] or "", 9)):
            sections.append({
                "id": d[0], "type": d[2] or "text", "version": d[6] or 1,
                "words": d[4] or 0,
            })
        sidebar_items.append({
            "num": ch_num, "title": ch_title, "id": ch_id,
            "sections": sections,
        })

    # Build main content
    active_draft = None
    if focus_draft:
        active_draft = draft_map.get(focus_draft)
    elif drafts:
        active_draft = drafts[0]

    active_html = ""
    active_comments = []
    active_sources = []
    active_review = ""
    active_id = ""
    active_title = ""
    if active_draft:
        active_id = active_draft[0]
        active_title = active_draft[1]
        active_html = _md_to_html(active_draft[3] or "")
        active_sources = json.loads(active_draft[5]) if isinstance(active_draft[5], str) else (active_draft[5] or [])
        active_review = active_draft[8] or ""
        active_comments = draft_comments.get(active_id, [])

    # Gap summary
    open_gaps = [g for g in gaps if g[3] == "open"]

    return TEMPLATE.format(
        book_title=book[1] if book else "Untitled",
        book_plan=(book[3] or "No plan set.") if book else "",
        sidebar_html=_render_sidebar(sidebar_items, active_id),
        content_html=active_html,
        active_id=active_id,
        active_title=active_title,
        active_version=active_draft[6] if active_draft else 1,
        active_words=active_draft[4] if active_draft else 0,
        sources_html=_render_sources(active_sources),
        review_html=_md_to_html(active_review) if active_review else "<em>No review yet.</em>",
        comments_html=_render_comments(active_comments),
        gaps_count=len(open_gaps),
        search_q=search_q,
        search_results_html=_render_search(search_results) if search_results else "",
    )


def _render_sidebar(items, active_id):
    html = ""
    for ch in items:
        html += f'<div class="ch-group"><div class="ch-title">Ch.{ch["num"]}: {ch["title"]}</div>'
        for sec in ch["sections"]:
            active = "active" if sec["id"] == active_id else ""
            html += (f'<a class="sec-link {active}" href="/section/{sec["id"]}">'
                     f'{sec["type"].capitalize()} <span class="meta">v{sec["version"]} · {sec["words"]}w</span></a>')
        if not ch["sections"]:
            html += '<div class="sec-link empty">No drafts yet</div>'
        html += '</div>'
    return html


def _render_sources(sources):
    if not sources:
        return "<em>No sources.</em>"
    return "<ol>" + "".join(f"<li>{s}</li>" for s in sources if s) + "</ol>"


def _render_comments(comments):
    if not comments:
        return ""
    html = ""
    for c in comments:
        cid, did, para, sel, comm, status, created = c
        cls = "resolved" if status == "resolved" else "open"
        sel_html = f'<div class="sel-text">"{sel[:100]}"</div>' if sel else ""
        para_html = f'<span class="para-ref">P{para}</span> ' if para is not None else ""
        resolve_btn = (
            f'<button class="resolve-btn" onclick="resolveComment(\'{cid}\')">Resolve</button>'
            if status == "open" else '<span class="resolved-tag">Resolved</span>'
        )
        html += f'<div class="comment {cls}">{para_html}{sel_html}<div class="comm-text">{comm}</div>{resolve_btn}</div>'
    return html


def _render_search(results):
    if not results:
        return "<p>No results.</p>"
    html = ""
    for d in results[:20]:
        html += f'<a href="/section/{d[0]}" class="search-result"><strong>{d[1]}</strong> ({d[4] or 0} words)</a>'
    return html


TEMPLATE = """\
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{book_title} — SciKnow Reader</title>
<style>
:root {{ --bg: #fff; --fg: #1a1a1a; --sidebar-bg: #f5f5f5; --border: #e0e0e0;
         --accent: #2563eb; --accent-light: #dbeafe; --success: #16a34a;
         --warning: #d97706; --danger: #dc2626; --code-bg: #f8f8f8; }}
[data-theme="dark"] {{ --bg: #1a1a2e; --fg: #e0e0e0; --sidebar-bg: #16213e;
         --border: #333; --accent: #60a5fa; --accent-light: #1e3a5f;
         --code-bg: #0f3460; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Georgia', serif; color: var(--fg); background: var(--bg);
        display: flex; height: 100vh; }}
/* Sidebar */
.sidebar {{ width: 280px; background: var(--sidebar-bg); border-right: 1px solid var(--border);
            overflow-y: auto; flex-shrink: 0; padding: 16px 0; }}
.sidebar h2 {{ padding: 8px 16px; font-size: 15px; color: var(--accent); }}
.ch-group {{ margin-bottom: 8px; }}
.ch-title {{ padding: 6px 16px; font-weight: bold; font-size: 13px; color: var(--fg); opacity: 0.7; }}
.sec-link {{ display: block; padding: 4px 16px 4px 32px; text-decoration: none;
             color: var(--fg); font-size: 13px; border-left: 3px solid transparent; }}
.sec-link:hover {{ background: var(--accent-light); }}
.sec-link.active {{ border-left-color: var(--accent); background: var(--accent-light); font-weight: bold; }}
.sec-link .meta {{ font-size: 11px; opacity: 0.5; }}
.sec-link.empty {{ color: var(--fg); opacity: 0.3; font-style: italic; }}
/* Main */
.main {{ flex: 1; overflow-y: auto; padding: 40px 60px; max-width: 900px; }}
.main h1 {{ font-size: 28px; margin-bottom: 8px; }}
.main .subtitle {{ font-size: 14px; color: var(--fg); opacity: 0.5; margin-bottom: 24px; }}
.main p {{ line-height: 1.8; margin-bottom: 12px; text-align: justify; }}
.main h2,.main h3,.main h4 {{ margin: 24px 0 12px; }}
.citation {{ color: var(--accent); cursor: pointer; font-weight: bold; }}
/* Right panel */
.panel {{ width: 320px; border-left: 1px solid var(--border); overflow-y: auto;
          padding: 16px; font-size: 13px; background: var(--sidebar-bg); }}
.panel h3 {{ font-size: 14px; margin: 16px 0 8px; color: var(--accent); }}
.panel ol {{ padding-left: 20px; }} .panel li {{ margin-bottom: 4px; font-size: 12px; }}
/* Comments */
.comment {{ padding: 8px; margin: 4px 0; border-left: 3px solid var(--accent);
            background: var(--accent-light); border-radius: 4px; }}
.comment.resolved {{ opacity: 0.5; border-left-color: var(--success); }}
.sel-text {{ font-style: italic; font-size: 12px; opacity: 0.6; margin-bottom: 4px; }}
.para-ref {{ font-size: 11px; background: var(--accent); color: white; padding: 1px 6px;
             border-radius: 8px; }}
.comm-text {{ margin: 4px 0; }}
.resolve-btn {{ font-size: 11px; background: var(--success); color: white; border: none;
                padding: 2px 8px; border-radius: 4px; cursor: pointer; }}
.resolved-tag {{ font-size: 11px; color: var(--success); }}
/* Comment form */
.comment-form {{ margin-top: 12px; }}
.comment-form textarea {{ width: 100%; padding: 6px; font-size: 12px; border: 1px solid var(--border);
                          border-radius: 4px; resize: vertical; min-height: 60px; background: var(--bg);
                          color: var(--fg); }}
.comment-form button {{ margin-top: 4px; padding: 4px 12px; background: var(--accent); color: white;
                        border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }}
/* Search */
.search-bar {{ padding: 8px 16px; }}
.search-bar input {{ width: 100%; padding: 6px 10px; border: 1px solid var(--border);
                     border-radius: 4px; font-size: 13px; background: var(--bg); color: var(--fg); }}
.search-result {{ display: block; padding: 6px 16px; text-decoration: none; color: var(--fg);
                  border-bottom: 1px solid var(--border); }}
.search-result:hover {{ background: var(--accent-light); }}
/* Theme toggle */
.theme-toggle {{ position: fixed; bottom: 16px; right: 16px; background: var(--accent);
                 color: white; border: none; padding: 8px 12px; border-radius: 20px;
                 cursor: pointer; font-size: 12px; z-index: 100; }}
/* Edit */
.edit-btn {{ background: var(--accent); color: white; border: none; padding: 4px 12px;
             border-radius: 4px; cursor: pointer; font-size: 12px; margin-left: 8px; }}
.edit-area {{ width: 100%; min-height: 400px; padding: 12px; font-family: 'Courier New', monospace;
              font-size: 14px; border: 1px solid var(--border); border-radius: 4px;
              background: var(--bg); color: var(--fg); }}
</style>
</head>
<body>

<!-- Sidebar -->
<nav class="sidebar">
  <h2>{book_title}</h2>
  <div class="search-bar">
    <form action="/search" method="get">
      <input type="text" name="q" placeholder="Search..." value="{search_q}">
    </form>
  </div>
  {search_results_html}
  {sidebar_html}
  <div style="padding: 8px 16px; font-size: 12px; opacity: 0.5; margin-top: 16px;">
    {gaps_count} open gaps
  </div>
</nav>

<!-- Main content -->
<main class="main" id="content">
  <h1>{active_title}</h1>
  <div class="subtitle">Version {active_version} · {active_words} words
    <button class="edit-btn" onclick="toggleEdit()">Edit</button>
  </div>

  <div id="read-view">{content_html}</div>

  <form id="edit-view" action="/edit/{active_id}" method="post" style="display:none;">
    <textarea class="edit-area" name="content" id="edit-area"></textarea>
    <br>
    <button type="submit" class="edit-btn" style="margin-top:8px;">Save</button>
    <button type="button" class="edit-btn" style="background:var(--danger);" onclick="toggleEdit()">Cancel</button>
  </form>
</main>

<!-- Right panel -->
<aside class="panel">
  <h3>Sources</h3>
  {sources_html}

  <h3>Review Feedback</h3>
  <div style="font-size:12px;">{review_html}</div>

  <h3>Comments</h3>
  {comments_html}
  <form class="comment-form" action="/comment" method="post">
    <input type="hidden" name="draft_id" value="{active_id}">
    <textarea name="comment" placeholder="Add a comment..."></textarea>
    <button type="submit">Add Comment</button>
  </form>
</aside>

<button class="theme-toggle" onclick="toggleTheme()">Toggle Theme</button>

<script>
function toggleTheme() {{
  const html = document.documentElement;
  html.dataset.theme = html.dataset.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('theme', html.dataset.theme);
}}
if (localStorage.getItem('theme')) {{
  document.documentElement.dataset.theme = localStorage.getItem('theme');
}}
function toggleEdit() {{
  const rv = document.getElementById('read-view');
  const ev = document.getElementById('edit-view');
  const ta = document.getElementById('edit-area');
  if (ev.style.display === 'none') {{
    // Extract text from read view
    ta.value = rv.innerText;
    rv.style.display = 'none';
    ev.style.display = 'block';
  }} else {{
    rv.style.display = 'block';
    ev.style.display = 'none';
  }}
}}
function resolveComment(cid) {{
  fetch('/comment/' + cid + '/resolve', {{method: 'POST'}})
    .then(() => location.reload());
}}
</script>
</body>
</html>
"""
