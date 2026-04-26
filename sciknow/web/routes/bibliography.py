"""``sciknow.web.routes.bibliography`` — bibliography endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. Behaviour
unchanged. The only cross-module dep is `_app._get_book_data` (resolved
lazily inside each handler via the standard `from sciknow.web import
app as _app` shim — defers the import until call-time so the route
file imports cleanly while app.py is mid-load).
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from sciknow.storage.db import get_session

router = APIRouter()


@router.get("/api/bibliography/audit")
async def api_bibliography_audit():
    """Per-draft sanity check for the book-wide bibliography.

    For each draft, reports:
      - broken_refs: citation numbers used in the body that have NO
                     matching entry in the draft's sources array.
      - orphan_sources: sources listed on the draft but never cited
                        in its body text.
      - duplicate_keys: the same publication appearing twice at
                        different local numbers (after strip ``[N]``).

    The user hits "Sanity check" once after a bibliography churn to see
    if the rewrite left any draft with references that point at a
    non-existent paper, which is almost always what "the citations look
    changed" means in practice.
    """
    from sciknow.web import app as _app
    book, _chapters, _drafts, _gaps, _comments = _app._get_book_data()
    if not book:
        return JSONResponse({"rows": [], "note": "No active book."})

    import re as _re
    rows: list[dict] = []
    totals = {"drafts_checked": 0, "broken": 0, "orphans": 0, "dupes": 0}

    with get_session() as session:
        draft_rows = session.execute(text("""
            SELECT d.id::text, d.title, d.section_type, d.content, d.sources,
                   d.version, d.chapter_id::text, d.custom_metadata,
                   COALESCE(bc.number, 999999) AS ch_num,
                   bc.title AS ch_title
            FROM drafts d
            LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.book_id = :bid
            ORDER BY ch_num, d.section_type, d.version DESC
        """), {"bid": str(book[0])}).fetchall()

    # Collapse to the active draft per (chapter, section) pair.
    def _meta_dict(raw):
        if isinstance(raw, dict): return raw
        if isinstance(raw, str) and raw:
            try: return json.loads(raw) or {}
            except Exception: return {}
        return {}

    # Phase 54.6.321 — same three-tier pick as core.bibliography:
    # is_active wins, then prefer non-empty content, then highest version.
    best: dict[tuple, tuple] = {}
    for r in draft_rows:
        did, title, sec, content, sources, ver, ch_id, meta, ch_num, ch_title = r
        key = (ch_id, sec)
        is_active = bool(_meta_dict(meta).get("is_active"))
        has_content = bool((content or "").strip())
        cur = best.get(key)
        if cur is None:
            best[key] = (r, is_active, has_content); continue
        cur_row, cur_active, cur_content = cur
        if is_active and not cur_active:
            best[key] = (r, True, has_content); continue
        if cur_active and not is_active:
            continue
        if has_content and not cur_content:
            best[key] = (r, is_active, True); continue
        if cur_content and not has_content:
            continue
        if (ver or 1) > (cur_row[5] or 1):
            best[key] = (r, is_active, has_content)

    for r, _is_a, _has_c in best.values():
        did, title, sec, content, sources, ver, ch_id, meta, ch_num, ch_title = r
        src_list = sources if isinstance(sources, list) else []
        src_by_num: dict[int, str] = {}
        for s in src_list:
            m = _re.match(r"^\s*\[(\d+)\]\s*(.*)$", s or "")
            if not m: continue
            src_by_num[int(m.group(1))] = m.group(2).strip()
        cited_nums = set()
        for m in _re.finditer(r"\[(\d+)\]", content or ""):
            cited_nums.add(int(m.group(1)))
        broken = sorted([n for n in cited_nums if n not in src_by_num])
        orphans = sorted([n for n in src_by_num.keys() if n not in cited_nums])
        # Duplicate detection: same normalized source body under two nums.
        seen_bodies: dict[str, list[int]] = {}
        for n, body in src_by_num.items():
            key2 = _re.sub(r"\s+", " ", body).lower().strip()[:300]
            seen_bodies.setdefault(key2, []).append(n)
        dupes = [nums for nums in seen_bodies.values() if len(nums) > 1]
        if broken or orphans or dupes:
            rows.append({
                "draft_id": did,
                "title": title,
                "section_type": sec,
                "chapter_num": int(ch_num) if ch_num and ch_num != 999999 else None,
                "chapter_title": ch_title,
                "broken_refs": broken,
                "orphan_sources": orphans,
                "duplicate_groups": dupes,
            })
            totals["broken"] += len(broken)
            totals["orphans"] += len(orphans)
            totals["dupes"] += len(dupes)
        totals["drafts_checked"] += 1

    return JSONResponse({"rows": rows, "totals": totals})


@router.post("/api/bibliography/sort")
async def api_bibliography_sort():
    """Flatten the global bibliography numbering INTO the stored draft
    content so the editor, markdown, and reader all agree on the same
    ``[N]`` values.

    Without this, the reader shows the global numbering (via
    ``_app.BookBibliography.remap_content``) but the raw markdown in the
    editor still has the original local numbers. Running "Sort" once
    rewrites each draft's content + sources list so ``[N]`` everywhere
    means the same paper. Idempotent — re-running after new sections
    are drafted re-numbers them into the book's reading order.
    """
    from sciknow.web import app as _app
    book, _chapters, _drafts, _gaps, _comments = _app._get_book_data()
    if not book:
        return JSONResponse({"ok": False, "note": "No active book."})

    import re as _re
    from sciknow.core.bibliography import BookBibliography, _renumber_source_line

    updated = 0
    with get_session() as session:
        bib = _app.BookBibliography.from_book(session, book[0])
        for did, lmap in bib.draft_local_to_global.items():
            if not lmap: continue
            row = session.execute(text(
                "SELECT content, sources FROM drafts WHERE id::text = :did"
            ), {"did": did}).fetchone()
            if not row: continue
            content, sources_raw = row
            # Rewrite body [N] → [G] via a placeholder two-pass.
            def _to_ph(match):
                n = int(match.group(1))
                g = lmap.get(n)
                return f"[__CITE_{g}__]" if g is not None else match.group(0)
            new_content = _re.sub(r"\[(\d+)\]", _to_ph, content or "")
            new_content = _re.sub(r"\[__CITE_(\d+)__\]", r"[\1]", new_content)
            # Rewrite the sources list to carry the new global numbers,
            # sorted ascending so the draft's local sources tab reads
            # 1,2,3… of the paper's actual global order.
            src_in = sources_raw if isinstance(sources_raw, list) else []
            new_sources: list[tuple[int, str]] = []
            for s in src_in:
                m = _re.match(r"^\s*\[(\d+)\]\s*", s or "")
                if not m: continue
                n = int(m.group(1))
                g = lmap.get(n)
                if g is None: continue
                new_sources.append((g, _renumber_source_line(s, g)))
            new_sources.sort(key=lambda t: t[0])
            if new_content != (content or "") or [s for _, s in new_sources] != (src_in or []):
                session.execute(text("""
                    UPDATE drafts SET content = :c, sources = CAST(:s AS jsonb)
                     WHERE id::text = :did
                """), {"c": new_content,
                       "s": json.dumps([s for _, s in new_sources]),
                       "did": did})
                updated += 1
        session.commit()

    return JSONResponse({"ok": True, "drafts_updated": updated,
                          "total_global_sources": len(bib.global_sources)})


@router.get("/api/bibliography/citation/{global_num}")
async def api_bibliography_citation(global_num: int):
    """Return full metadata for a global citation number — title,
    authors, year, journal, doi, abstract, best-open-URL — so the
    reader can render a click-to-preview popover richer than the hover
    tooltip in ``buildPopovers``.
    """
    from sciknow.web import app as _app
    book, _chapters, _drafts, _gaps, _comments = _app._get_book_data()
    if not book:
        raise HTTPException(404, "No active book.")
    from sciknow.core.bibliography import BookBibliography, _source_key
    import re as _re
    with get_session() as session:
        bib = _app.BookBibliography.from_book(session, book[0])
        idx = int(global_num) - 1
        if idx < 0 or idx >= len(bib.global_sources):
            raise HTTPException(404, "Citation out of range.")
        source_line = bib.global_sources[idx]
        # Attempt to map the source line back to a paper_metadata row.
        key = _source_key(source_line)
        # Best-effort: extract the title inside ... (year). Title. journal
        m = _re.search(r"\(\d{4}(?:-\d{2}-\d{2})?\)\.\s+([^.]+)\.", source_line)
        title_guess = m.group(1).strip() if m else None
        meta: dict | None = None
        if title_guess:
            mrow = session.execute(text("""
                SELECT document_id::text, title, authors, year, journal, doi, abstract,
                       open_access_url, url
                FROM paper_metadata
                WHERE lower(title) = lower(:t)
                LIMIT 1
            """), {"t": title_guess}).fetchone()
            if mrow:
                did, title, authors, year, journal, doi, abstract, oa_url, url = mrow
                meta = {
                    "document_id": did,
                    "title": title,
                    "authors": authors or [],
                    "year": year,
                    "journal": journal,
                    "doi": doi,
                    "abstract": (abstract or "")[:1200],
                    "open_access_url": oa_url or url or (f"https://doi.org/{doi}" if doi else None),
                }
    return JSONResponse({
        "global_num": int(global_num),
        "source_line": source_line,
        "key": key,
        "metadata": meta,
    })
