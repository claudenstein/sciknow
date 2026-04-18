"""Phase 54.6.76 (#15) — GPU-time ledger per draft / chapter / book.

Rolls up `autowrite_runs` telemetry (started_at, finished_at,
tokens_used, model) into aggregates that answer:

  - *How much GPU-wall time did this one draft consume?*
  - *Which chapter ate my afternoon?*
  - *Per-book totals so I can see drift as the corpus grows.*

All time is **wall time per autowrite run** — includes retrieval,
scoring, verification, CoVe, and idle gaps inside a run. This is the
right number to optimize against because it's what the user actually
waits for. If you want pure LLM time only, look at the per-model
throughput bench instead.

Reads are read-only — no new tables, no new columns. Just a set of
aggregation queries exposed both as a Python helper and a
`sciknow book ledger` CLI.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LedgerRow:
    """One row of the ledger (draft / chapter / book / section level)."""
    scope: str              # 'draft' | 'chapter' | 'book' | 'section'
    label: str              # human-readable label (title, section name, ...)
    n_runs: int             # how many autowrite runs contributed
    wall_seconds: float     # total autowrite wall-time
    tokens: int             # sum of autowrite_runs.tokens_used
    started_first: str      # first run's started_at (display-ready)
    finished_last: str      # last run's finished_at

    @property
    def tokens_per_second(self) -> float:
        if self.wall_seconds <= 0:
            return 0.0
        return self.tokens / self.wall_seconds


def _fetch_rows(session, where_sql: str, params: dict) -> list[dict]:
    """Fetch autowrite_runs rows filtered by the caller's WHERE clause
    and return a uniform shape. Runs without finished_at are treated
    as wall_seconds=0 (still ongoing or crashed mid-run — those don't
    count toward the ledger until they finish)."""
    from sqlalchemy import text
    rows = session.execute(text(f"""
        SELECT r.id::text,
               r.book_id::text,
               r.chapter_id::text,
               r.section_slug,
               r.final_draft_id::text,
               r.model,
               r.tokens_used,
               r.started_at,
               r.finished_at,
               COALESCE(
                   EXTRACT(EPOCH FROM (r.finished_at - r.started_at)),
                   0
               ) AS wall_seconds
        FROM autowrite_runs r
        WHERE {where_sql}
        ORDER BY r.started_at
    """), params).fetchall()
    return [
        {
            "run_id": r[0], "book_id": r[1], "chapter_id": r[2],
            "section_slug": r[3], "final_draft_id": r[4],
            "model": r[5], "tokens": int(r[6] or 0),
            "started_at": str(r[7]) if r[7] else "",
            "finished_at": str(r[8]) if r[8] else "",
            "wall_seconds": float(r[9] or 0),
        }
        for r in rows
    ]


# ════════════════════════════════════════════════════════════════════════
# Per-scope rollups
# ════════════════════════════════════════════════════════════════════════


def ledger_for_draft(session, draft_id: str) -> LedgerRow | None:
    """Ledger for one draft. Matches on either the draft's own id
    prefix OR the autowrite run that produced it (final_draft_id)."""
    from sqlalchemy import text
    d = session.execute(text(
        "SELECT id::text, title FROM drafts WHERE id::text LIKE :q LIMIT 1"
    ), {"q": f"{draft_id}%"}).fetchone()
    if not d:
        return None
    rows = _fetch_rows(
        session,
        "r.final_draft_id::text = :did",
        {"did": d[0]},
    )
    return _aggregate("draft", d[1] or d[0][:8], rows)


def ledger_for_chapter(session, chapter_id: str) -> LedgerRow | None:
    """Ledger for one chapter — sums every autowrite run that touched
    any section in the chapter."""
    from sqlalchemy import text
    ch = session.execute(text(
        "SELECT id::text, number, title FROM book_chapters "
        "WHERE id::text = :cid LIMIT 1"
    ), {"cid": chapter_id}).fetchone()
    if not ch:
        return None
    rows = _fetch_rows(
        session,
        "r.chapter_id::text = :cid",
        {"cid": ch[0]},
    )
    label = f"Ch.{ch[1]} {ch[2] or ''}".strip()
    return _aggregate("chapter", label, rows)


def ledger_for_book(session, book_id: str) -> LedgerRow | None:
    """Ledger for the whole book — sums every run whose
    autowrite_runs.book_id matches."""
    from sqlalchemy import text
    b = session.execute(text(
        "SELECT id::text, title FROM books WHERE id::text = :bid LIMIT 1"
    ), {"bid": book_id}).fetchone()
    if not b:
        return None
    rows = _fetch_rows(session, "r.book_id::text = :bid", {"bid": b[0]})
    return _aggregate("book", b[1] or b[0][:8], rows)


def ledger_per_chapter(session, book_id: str) -> list[LedgerRow]:
    """One row per chapter in a book — for the book-detail view."""
    from sqlalchemy import text
    chapters = session.execute(text(
        "SELECT id::text, number, title FROM book_chapters "
        "WHERE book_id::text = :bid ORDER BY number"
    ), {"bid": book_id}).fetchall()
    out: list[LedgerRow] = []
    for ch in chapters:
        rows = _fetch_rows(
            session, "r.chapter_id::text = :cid", {"cid": ch[0]},
        )
        if not rows:
            continue
        label = f"Ch.{ch[1]} {ch[2] or ''}".strip()
        lg = _aggregate("chapter", label, rows)
        if lg:
            out.append(lg)
    return out


def ledger_per_section(session, chapter_id: str) -> list[LedgerRow]:
    """One row per section in a chapter — for the chapter-detail view."""
    rows = _fetch_rows(
        session, "r.chapter_id::text = :cid", {"cid": chapter_id},
    )
    by_slug: dict[str, list[dict]] = {}
    for r in rows:
        s = r["section_slug"] or "(unknown)"
        by_slug.setdefault(s, []).append(r)
    out: list[LedgerRow] = []
    for slug, subset in sorted(by_slug.items()):
        lg = _aggregate("section", slug, subset)
        if lg:
            out.append(lg)
    return out


def _aggregate(scope: str, label: str, rows: list[dict]) -> LedgerRow | None:
    if not rows:
        return LedgerRow(
            scope=scope, label=label, n_runs=0, wall_seconds=0.0,
            tokens=0, started_first="", finished_last="",
        )
    wall = sum(r["wall_seconds"] for r in rows)
    tokens = sum(r["tokens"] for r in rows)
    started = min((r["started_at"] for r in rows if r["started_at"]),
                  default="")
    finished = max((r["finished_at"] for r in rows if r["finished_at"]),
                   default="")
    return LedgerRow(
        scope=scope, label=label, n_runs=len(rows),
        wall_seconds=wall, tokens=tokens,
        started_first=started, finished_last=finished,
    )


# ════════════════════════════════════════════════════════════════════════
# Display helpers (used by both CLI + web)
# ════════════════════════════════════════════════════════════════════════


def format_wall(seconds: float) -> str:
    """Human-readable duration: `12s`, `3m 45s`, `1h 12m`."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m"


def ledger_as_dict(row: LedgerRow) -> dict:
    """Serializable form for JSON responses."""
    return {
        "scope": row.scope,
        "label": row.label,
        "n_runs": row.n_runs,
        "wall_seconds": round(row.wall_seconds, 1),
        "wall_human": format_wall(row.wall_seconds),
        "tokens": row.tokens,
        "tokens_per_second": round(row.tokens_per_second, 1),
        "started_first": row.started_first,
        "finished_last": row.finished_last,
    }
