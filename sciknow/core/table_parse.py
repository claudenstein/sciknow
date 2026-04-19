"""Phase 54.6.106 (#2) — MinerU HTML tables → structured metadata.

MinerU emits tables as a single ``<table>`` blob with heavy rowspan /
colspan merging. Getting clean ``{headers, rows, units}`` without an
LLM is unreliable because MinerU often merges the title row INTO the
table header row (see 54.6.106 sample).

This module produces a lightweight structured summary, not a fully
normalized relation — enough to:

  * show a readable subtitle in the Visuals gallery ("Proxy sites with
    latitude, sample resolution, …")
  * embed for semantic retrieval over tables (future companion to
    ``db embed-visuals`` on figures/charts)
  * label the table with inferred column headers so GUI users don't
    have to squint at raw MinerU output

Output shape written back to the ``visuals`` row:

    table_title    TEXT       — inferred caption/title (one line)
    table_headers  JSONB      — array of column header strings
    table_summary  TEXT       — 1-3 sentence semantic summary
    table_n_rows   INT        — observed row count from HTML
    table_n_cols   INT        — observed column count from HTML
    table_parsed_at TIMESTAMP — parse timestamp (idempotency key)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from sqlalchemy import text

from sciknow.config import settings
from sciknow.rag.llm import complete
from sciknow.storage.db import get_session

logger = logging.getLogger("sciknow.core.table_parse")


_SYS_PROMPT = """You parse HTML tables into a compact JSON structure.

Return ONLY valid JSON with this exact shape (no prose, no code fences):

  {
    "title": "<one-line title, null if unclear>",
    "headers": ["<col1>", "<col2>", ...],
    "summary": "<1-3 sentences describing what the table contains>"
  }

Rules:
- "headers" must be an array of short column-label strings (no cell
  values). If the table has merged header rows, produce the most useful
  flat list (prefer the most specific level).
- "title" is the table's caption/title as a short phrase. Null if the
  input is pure data with no title row.
- "summary" describes what ROWS the table contains, what the columns
  represent, and any units observed. Scientific, concise, factual.
- If the input is not actually a table (garbled HTML, empty, etc),
  return {"title": null, "headers": [], "summary": "unparsable"}.
"""


@dataclass
class ParsedTable:
    title: str | None
    headers: list[str]
    summary: str
    n_rows: int
    n_cols: int


def _count_rows_cols(html: str) -> tuple[int, int]:
    """Cheap regex-based row/col count — no DOM needed. MinerU emits
    flat ``<tr>…<td>…</td></tr>`` blocks, so the first row's ``<td>``
    count is a decent col-count approximation."""
    rows = re.findall(r"<tr\b[^>]*>", html, flags=re.IGNORECASE)
    n_rows = len(rows)
    if n_rows == 0:
        return 0, 0
    # First <tr>…</tr> block
    m = re.search(r"<tr\b[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return n_rows, 0
    first = m.group(1)
    cells = re.findall(r"<t[hd]\b", first, flags=re.IGNORECASE)
    return n_rows, len(cells)


def parse_table(html: str, model: str | None = None) -> ParsedTable:
    """Run the fast LLM on the table HTML and return a ParsedTable."""
    n_rows, n_cols = _count_rows_cols(html or "")
    # Guard against absurd inputs — cap at 12 KB of HTML (MinerU tables
    # tend to be 2-8 KB; tail truncation loses data rows but keeps the
    # headers we care about).
    html_trim = (html or "")[:12000]
    if not html_trim.strip():
        return ParsedTable(
            title=None, headers=[], summary="unparsable",
            n_rows=n_rows, n_cols=n_cols,
        )
    chosen = model or settings.llm_fast_model
    usr = f"Input HTML:\n\n{html_trim}\n\nRespond with the JSON now."
    try:
        raw = complete(
            _SYS_PROMPT, usr,
            model=chosen, temperature=0.1, num_ctx=8192, num_predict=1500,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("parse_table LLM call failed: %s", exc)
        return ParsedTable(
            title=None, headers=[], summary=f"LLM error: {exc}",
            n_rows=n_rows, n_cols=n_cols,
        )

    # Strip thinking tags + JSON fences.
    txt = raw.strip()
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL | re.IGNORECASE).strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(txt, strict=False)
    except Exception:
        # Try to locate a {…} JSON object somewhere in the reply.
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return ParsedTable(
                title=None, headers=[], summary="LLM JSON parse failed",
                n_rows=n_rows, n_cols=n_cols,
            )
        try:
            data = json.loads(m.group(0), strict=False)
        except Exception as exc:  # noqa: BLE001
            return ParsedTable(
                title=None, headers=[], summary=f"LLM JSON parse failed: {exc}",
                n_rows=n_rows, n_cols=n_cols,
            )

    title = data.get("title")
    if isinstance(title, str):
        title = title.strip()[:400] or None
    else:
        title = None
    headers = data.get("headers") or []
    if not isinstance(headers, list):
        headers = []
    headers = [str(h).strip()[:200] for h in headers if h]
    summary = str(data.get("summary") or "").strip()[:2000]
    return ParsedTable(
        title=title, headers=headers, summary=summary,
        n_rows=n_rows, n_cols=n_cols,
    )


def parse_pending_tables(
    model: str | None = None, limit: int | None = None, force: bool = False,
    progress_callback=None,
) -> dict:
    """Parse every unparsed (or all, with ``force``) kind='table' row.

    Returns a dict with counters: ``{parsed, skipped, errors, total}``.
    ``progress_callback(i, total, vid, outcome)`` is called after each
    row so the CLI can render a live tally.
    """
    where = "kind = 'table' AND content IS NOT NULL AND content <> ''"
    if not force:
        where += " AND table_parsed_at IS NULL"
    with get_session() as session:
        rows = session.execute(text(
            f"SELECT id::text, content FROM visuals WHERE {where} "
            f"ORDER BY created_at ASC"
            f"{' LIMIT :lim' if limit else ''}"
        ), {"lim": limit} if limit else {}).fetchall()

    total = len(rows)
    parsed = skipped = errors = 0
    chosen_model = model or settings.llm_fast_model

    for i, (vid, html) in enumerate(rows, start=1):
        try:
            result = parse_table(html, model=chosen_model)
            outcome = "parsed" if result.headers or result.summary else "empty"
            with get_session() as session:
                session.execute(text("""
                    UPDATE visuals SET
                        table_title = :title,
                        table_headers = CAST(:headers AS jsonb),
                        table_summary = :summary,
                        table_n_rows = :n_rows,
                        table_n_cols = :n_cols,
                        table_parsed_at = NOW()
                    WHERE id::text = :vid
                """), {
                    "title": result.title,
                    "headers": json.dumps(result.headers),
                    "summary": result.summary,
                    "n_rows": result.n_rows,
                    "n_cols": result.n_cols,
                    "vid": vid,
                })
                session.commit()
            parsed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("parse_pending_tables: row %s failed: %s", vid, exc)
            outcome = f"error: {exc}"
            errors += 1
        if progress_callback is not None:
            progress_callback(i, total, vid, outcome)

    return {
        "parsed": parsed, "skipped": skipped, "errors": errors,
        "total": total, "model": chosen_model,
    }
