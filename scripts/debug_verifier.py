"""Verifier MISREPRESENTED diagnostic (Phase 55.V18 follow-up).

Re-runs claim-verification against an existing saved draft so we can
inspect the per-claim verdicts and reasons WITHOUT spending another
10 min on a full autowrite cycle.

The bug surfaced in slate #2: every cited [N] claim in the writer's
draft for `the_science_of_sunspots` was marked MISREPRESENTED by the
verifier (51 of 51 on the q4_0 cell's 993-word draft), forcing
groundedness=0.0 and triggering the revision-compression cascade.

Pre-Phase-55.S1 the verifier was silently no-op'ing (returned
groundedness=None) so books shipped without the verifier's harsh
second opinion ever running. With the dedicated scorer role
working, verification now runs reliably and dominates the score.

This script:
  1. Pulls a saved draft by id (default: the 993-word q4_0 sunspots
     draft, hash 4cb268b2)
  2. Re-runs hybrid_search + rerank + context_builder with the same
     query autowrite uses (`f"{section_type} {topic}"`)
  3. Calls `_verify_draft_inner(draft, results, model=scorer_model)`
  4. Pretty-prints the per-claim verdicts so we can manually compare
     5-10 MISREPRESENTED claims to the cited chunks

Usage:
  uv run python scripts/debug_verifier.py
  uv run python scripts/debug_verifier.py --draft-id 4cb268b2
  uv run python scripts/debug_verifier.py --draft-id <prefix> --top-k 8

Diagnostic outputs:
  - For each cited [N] in the draft, the verifier's verdict
    (SUPPORTED / EXTRAPOLATED / OVERSTATED / MISREPRESENTED / MISSING)
  - The verifier's reason
  - The actual cited chunk text so a human can manually adjudicate
    "writer hallucinated" vs "verifier was too strict"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import text as sql_text

from sciknow.config import settings
from sciknow.storage.db import get_session
from sciknow.storage.qdrant import get_client as get_qdrant
from sciknow.core.book_ops import _retrieve_with_step_back, _verify_draft_inner

console = Console()


def _load_draft(draft_prefix: str) -> dict:
    """Pull the draft's content + section_type + chapter context.

    Uses the SAME `topic` resolution as `_autowrite_section_body`:
    chapter.topic_query OR chapter.title (NOT book.description).
    """
    with get_session() as s:
        row = s.execute(sql_text("""
            SELECT d.id::text, d.content, d.section_type, d.chapter_id::text,
                   d.book_id::text, d.word_count
            FROM drafts d
            WHERE d.id::text LIKE :pfx
            ORDER BY d.created_at DESC
            LIMIT 1
        """), {"pfx": f"{draft_prefix}%"}).fetchone()
        if not row:
            raise SystemExit(f"No draft matching prefix {draft_prefix!r}")
        did, content, section_type, chapter_id, book_id, words = row

        ch = s.execute(sql_text("""
            SELECT title, topic_query FROM book_chapters WHERE id::text = :cid
        """), {"cid": chapter_id}).fetchone()
        chapter_title = ch[0] if ch else "(unknown chapter)"
        chapter_topic_query = ch[1] if ch and len(ch) > 1 else None

        bk = s.execute(sql_text("""
            SELECT title FROM books WHERE id::text = :bid
        """), {"bid": book_id}).fetchone()
        book_title = bk[0] if bk else "(unknown book)"

    # Same resolution as autowrite.py:299 — `topic = topic_query or ch_title`.
    topic = chapter_topic_query or chapter_title

    return {
        "draft_id": did, "content": content, "section_type": section_type,
        "chapter_id": chapter_id, "chapter_title": chapter_title,
        "book_id": book_id, "book_title": book_title,
        "topic": topic, "word_count": words,
    }


def _print_chunk_summary(results: list, max_chunks_to_show: int = 6) -> None:
    tbl = Table(title="Retrieved chunks (top-k after rerank)", show_lines=False)
    tbl.add_column("[N]", justify="right")
    tbl.add_column("doc")
    tbl.add_column("section")
    tbl.add_column("snippet")
    for i, r in enumerate(results[:max_chunks_to_show], start=1):
        title = (r.title or "(untitled)")[:30]
        section = (r.section_type or "?")[:12]
        snippet = (r.content or "")[:80].replace("\n", " ")
        tbl.add_row(f"[{i}]", title, section, snippet + "...")
    console.print(tbl)
    if len(results) > max_chunks_to_show:
        console.print(f"[dim]... and {len(results) - max_chunks_to_show} more chunks not shown[/dim]")


def _print_verdicts(vdata: dict, results: list, max_claims_to_show: int = 10) -> None:
    claims = vdata.get("claims") or []
    if not claims:
        console.print("[red]Verifier returned no `claims` list — unparseable JSON or empty draft.[/red]")
        return

    by_verdict = {}
    for c in claims:
        v = c.get("verdict") or "UNKNOWN"
        by_verdict.setdefault(v, []).append(c)

    counts_tbl = Table(title="Verdict summary", show_lines=False)
    counts_tbl.add_column("verdict")
    counts_tbl.add_column("count", justify="right")
    for v in ("SUPPORTED", "OVERSTATED", "EXTRAPOLATED", "MISREPRESENTED", "UNKNOWN"):
        counts_tbl.add_row(v, str(len(by_verdict.get(v, []))))
    console.print(counts_tbl)

    for verdict_kind in ("MISREPRESENTED", "EXTRAPOLATED", "OVERSTATED"):
        for c in by_verdict.get(verdict_kind, [])[:max_claims_to_show]:
            cite = c.get("citation") or "[?]"
            text = (c.get("text") or "")[:300]
            reason = (c.get("reason") or "")[:300]

            try:
                idx = int(cite.strip("[]")) - 1
            except (ValueError, AttributeError):
                idx = None
            cited_chunk = ""
            if idx is not None and 0 <= idx < len(results):
                cited_chunk = (results[idx].content or "")[:500].replace("\n", " ")

            console.print(Panel(
                f"[bold]{verdict_kind}[/bold] {cite}\n\n"
                f"[bold]claim:[/bold] {text}\n\n"
                f"[bold]verifier reason:[/bold] {reason}\n\n"
                f"[bold]cited chunk text (first 500 chars):[/bold] {cited_chunk}",
                title=f"Claim flagged {verdict_kind}",
                border_style="red" if verdict_kind == "MISREPRESENTED" else "yellow",
            ))


def main() -> int:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--draft-id", default="4cb268b2",
                   help="Prefix of the saved draft to verify (default: 4cb268b2 — q4_0 sunspots cell with 51 MISREPRESENTED claims)")
    p.add_argument("--top-k", type=int, default=8,
                   help="Number of chunks to retrieve (matches autowrite default).")
    p.add_argument("--max-claims-to-show", type=int, default=8,
                   help="How many flagged claims to print per verdict kind.")
    p.add_argument("--save-json", default=None,
                   help="Optional path to dump full verifier output as JSON.")
    args = p.parse_args()

    info = _load_draft(args.draft_id)
    console.rule(f"[bold]Verifier diagnostic: draft {info['draft_id'][:8]}[/bold]")
    console.print(f"  book: {info['book_title']!r}")
    console.print(f"  chapter: {info['chapter_title']!r}")
    console.print(f"  section: [bold]{info['section_type']}[/bold]")
    console.print(f"  draft length: {info['word_count']} words / {len(info['content'])} chars")
    console.print()

    query = f"{info['section_type']} {info['topic']}"
    console.print(f"[dim]Retrieving with query: {query!r}[/dim]")

    qdrant = get_qdrant()
    with get_session() as session:
        results, sources = _retrieve_with_step_back(
            session, qdrant, query,
            topic_cluster=None, model=None, use_step_back=False,
        )
    if not results:
        console.print("[red]Retrieval returned no chunks. Aborting.[/red]")
        return 2
    console.print(f"[dim]Got {len(results)} chunks after rerank[/dim]\n")
    _print_chunk_summary(results)

    scorer_model = (settings.scorer_model_name
                    if getattr(settings, "use_llamacpp_scorer", False)
                    else settings.book_write_model)
    console.print(f"\n[dim]Calling _verify_draft_inner with model={scorer_model!r}...[/dim]")
    vdata = _verify_draft_inner(info["content"], results, model=scorer_model)
    console.print(f"[dim]Verifier returned: groundedness_score="
                  f"{vdata.get('groundedness_score')}  "
                  f"hedging_fidelity_score={vdata.get('hedging_fidelity_score')}  "
                  f"claims={len(vdata.get('claims') or [])}[/dim]\n")

    _print_verdicts(vdata, results, max_claims_to_show=args.max_claims_to_show)

    if args.save_json:
        outp = Path(args.save_json)
        outp.write_text(json.dumps({
            "draft_info": {k: v for k, v in info.items() if k != "content"},
            "draft_content": info["content"],
            "retrieved_chunks": [
                {"index": i + 1, "title": r.title, "section": r.section_type,
                 "year": r.year, "content": r.content}
                for i, r in enumerate(results)
            ],
            "verifier_output": vdata,
        }, indent=2, default=str))
        console.print(f"\n[green]Full output saved to {outp}[/green]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
