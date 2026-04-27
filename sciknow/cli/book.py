"""
sciknow book — manage book projects, chapters, outlines, and writing.

Commands:
  create          Create a new book project
  list            List all books
  show            Show a book's chapters and draft status
  chapter add     Add a chapter to a book
  outline         Generate a chapter structure from the literature (LLM)
  plan            Generate / view / edit the book plan (thesis + scope)
  write           Draft a chapter section (with cross-chapter coherence)
  review          Run a critic pass over a saved draft
  revise          Revise a draft based on instructions or review feedback
  argue           Map the argument for a claim against the literature
  gaps            Identify + persist what's missing in the book
  export          Compile all chapter drafts into Markdown, LaTeX, or DOCX
"""
from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

app = typer.Typer(help="Manage book projects and chapter writing.")
chapter_app = typer.Typer(help="Manage chapters within a book.")
app.add_typer(chapter_app, name="chapter")

# Track A measurement & observability commands.
# `book draft scores <id>`, `book draft compare <a> <b>` — read autowrite
# score history persisted to drafts.custom_metadata. `book autowrite-bench`
# is a top-level command since it's a multi-run experiment, not a draft op.
draft_app = typer.Typer(help="Inspect and compare saved drafts (Track A measurement).")
app.add_typer(draft_app, name="draft")

# Phase 32.9 — Compound learning Layer 4: DPO preference dataset export.
# `book preferences export` walks autowrite_iterations and writes a
# JSONL file in the standard {prompt, chosen, rejected} shape, ready
# for DPO fine-tuning when the DGX Spark arrives (Layer 6). Both KEEP
# and DISCARD verdicts produce pairs (DISCARDs as inverse pairs).
preferences_app = typer.Typer(help="Export DPO preference pairs from autowrite history (Layer 4).")
app.add_typer(preferences_app, name="preferences")

# Phase 32.10 — Compound learning Layer 5: style fingerprint.
# `book style refresh` extracts the user's writing style from approved
# drafts (status in final/reviewed/revised) and persists to
# books.custom_metadata.style_fingerprint. The autowrite loop reads
# this and injects it into the writer system prompt as a style anchor.
style_app = typer.Typer(help="Manage the per-book writing style fingerprint (Layer 5).")
app.add_typer(style_app, name="style")

console = Console()


# ── helpers ────────────────────────────────────────────────────────────────────

def _get_book(session, title_or_id: str):
    """Fetch a book row.

    Returns a tuple ``(id, title, description, status, created_at, plan, book_type)``.
    Phase 45 appended ``book_type`` at the end so every caller can
    branch on project type without rewriting their indexed access.
    Older callers that slice ``row[:6]`` keep working unchanged.
    """
    from sqlalchemy import text
    row = session.execute(text("""
        SELECT id::text, title, description, status, created_at, plan, book_type
        FROM books WHERE title ILIKE :q OR id::text LIKE :q
        LIMIT 1
    """), {"q": f"%{title_or_id}%"}).fetchone()
    return row


def _get_chapter(session, book_id: str, chapter_ref: str):
    """Find a chapter by number or title fragment."""
    from sqlalchemy import text
    if chapter_ref.isdigit():
        row = session.execute(text("""
            SELECT id::text, number, title, description, topic_query, topic_cluster
            FROM book_chapters WHERE book_id = :bid AND number = :num
        """), {"bid": book_id, "num": int(chapter_ref)}).fetchone()
        if row:
            return row
    return session.execute(text("""
        SELECT id::text, number, title, description, topic_query, topic_cluster
        FROM book_chapters WHERE book_id = :bid AND title ILIKE :q
        LIMIT 1
    """), {"bid": book_id, "q": f"%{chapter_ref}%"}).fetchone()


def _get_prior_summaries(session, book_id: str, before_chapter_number: int) -> list[dict]:
    """Return summaries of all drafts from chapters before the given chapter number."""
    from sqlalchemy import text
    rows = session.execute(text("""
        SELECT bc.number, d.section_type, d.summary
        FROM drafts d
        JOIN book_chapters bc ON bc.id = d.chapter_id
        WHERE d.book_id = :bid AND bc.number < :ch_num AND d.summary IS NOT NULL
        ORDER BY bc.number, d.section_type
    """), {"bid": book_id, "ch_num": before_chapter_number}).fetchall()
    return [{"chapter_number": r[0], "section_type": r[1], "summary": r[2]} for r in rows]


def _consume_events(gen, console):
    """Consume events from a core.book_ops generator, printing to Rich console.

    Returns the final 'completed' event dict (or None if the operation failed).
    """
    from rich.rule import Rule

    completed = None
    for event in gen:
        t = event.get("type")
        if t == "token":
            console.print(event["text"], end="", highlight=False)
        elif t == "progress":
            detail = event.get("detail", event.get("stage", ""))
            console.print(f"[dim]{detail}[/dim]")
        elif t == "tree_plan":
            console.print()
            console.print(Rule("[bold]Paragraph Plan (tree)[/bold]"))
            data = event.get("data", {})
            for i, para in enumerate(data.get("paragraphs", []), 1):
                point = para.get("point", "")
                sources = ", ".join(para.get("sources", []))
                connects = para.get("connects_to", "")
                console.print(f"  [bold cyan]P{i}[/bold cyan] {point}")
                if sources:
                    console.print(f"      [dim]Sources: {sources}[/dim]")
                if connects:
                    console.print(f"      [dim]→ {connects}[/dim]")
                for child in para.get("children", []):
                    console.print(f"        [dim]- {child.get('point', '')}[/dim]")
            console.print()
        elif t == "plan":
            console.print()
            console.print(Rule("[bold]Sentence Plan[/bold]"))
            console.print(event["content"])
            console.print()
        elif t == "scores":
            scores = event["scores"]
            dims = ["groundedness", "completeness", "coherence", "citation_accuracy", "overall"]
            parts = []
            for d in dims:
                v = scores.get(d, 0)
                color = "green" if v >= 0.85 else "yellow" if v >= 0.7 else "red"
                parts.append(f"[{color}]{d[:5]}={v:.2f}[/{color}]")
            console.print(f"  Scores: {'  '.join(parts)}")
            weakest = scores.get("weakest_dimension", "")
            if weakest:
                console.print(f"  Weakest: [bold]{weakest}[/bold]")
            instruction = scores.get("revision_instruction", "")
            if instruction:
                console.print(f"  Instruction: [dim]{instruction}[/dim]")
        elif t == "iteration_start":
            console.print(Rule(f"[dim]Iteration {event['iteration']}/{event['max']}[/dim]"))
        elif t == "revision_verdict":
            icon = "\u2713" if event["action"] == "KEEP" else "\u2717"
            color = "green" if event["action"] == "KEEP" else "red"
            console.print(
                f"  [{color}]{icon} {event['action']}[/{color}]: "
                f"overall {event['old_score']:.2f} \u2192 {event['new_score']:.2f}"
            )
        elif t == "converged":
            console.print(
                f"\n[green]\u2713 Converged at iteration {event['iteration']} "
                f"(overall={event['final_score']:.2f})[/green]"
            )
        elif t == "verification":
            vdata = event["data"]
            score = vdata.get("groundedness_score", "?")
            console.print(f"  Groundedness score: [bold]{score}[/bold]")
            for claim in vdata.get("claims", []):
                v = claim.get("verdict", "?")
                color = "green" if v == "SUPPORTED" else "yellow" if v == "EXTRAPOLATED" else "red"
                console.print(f"  [{color}]{v}[/{color}]: {claim.get('text', '')[:80]}")
        elif t == "citation_needs":
            # Phase 46.A — two-stage citation insertion: pass-1 output.
            console.print()
            console.print(Rule(f"[bold]Citation opportunities[/bold]  ({event.get('count', 0)} found)"))
            for i, n in enumerate(event.get("needs", []), 1):
                loc = (n.get("location") or "")[:90]
                q   = n.get("query") or ""
                console.print(f"  [cyan]{i}.[/cyan] {loc}")
                console.print(f"      [dim]query: {q}[/dim]")
            console.print()
        elif t == "citation_candidates":
            # Don't print the full candidate list per need — it's noisy;
            # just show the count so the user sees progress.
            console.print(
                f"  [dim]→ retrieved {len(event.get('candidates', []))} candidates "
                f"for need #{event.get('index', '?') + 1 if isinstance(event.get('index'), int) else '?'}[/dim]"
            )
        elif t == "citation_selected":
            m = event.get("marker", "?")
            src = event.get("source", {})
            conf = event.get("confidence", 0.0)
            console.print(
                f"  [green]+[/green] [{m}] {src.get('title', '')[:70]} "
                f"({src.get('year', '')}) [dim]conf={conf:.2f}[/dim]"
            )
        elif t == "citation_skipped":
            r = event.get("reason", "")
            console.print(f"  [yellow]~[/yellow] skipped: {r}")
        elif t == "citation_inserted":
            console.print(
                f"\n[bold]Inserted {event.get('count', 0)} citation(s)[/bold] "
                f"([dim]{event.get('skipped', 0)} skipped[/dim])"
            )
        elif t == "reviewer_done":
            # Phase 46.C — ensemble review, one reviewer's result
            status = event.get("status", "")
            idx   = event.get("index", "?")
            stance = event.get("stance", "")
            if status == "ok":
                overall = event.get("overall")
                decision = event.get("decision", "")
                conf = event.get("confidence")
                ov = f"{overall}" if overall is not None else "—"
                cf = f"conf={conf}" if conf is not None else ""
                console.print(
                    f"  [green]✓[/green] Reviewer {idx} ({stance}): "
                    f"overall={ov}  decision={decision}  {cf}"
                )
            else:
                console.print(
                    f"  [yellow]~[/yellow] Reviewer {idx} ({stance}): {status} "
                    f"{event.get('message', '')}"
                )
        elif t == "meta_review_start":
            n = event.get("n_reviewers", 0)
            scores = event.get("overall_scores", [])
            scores_str = ", ".join(str(s) if s is not None else "?" for s in scores)
            console.print(
                f"\n[bold]Fusing {n} reviewer(s)[/bold]  [dim](overall: {scores_str})[/dim]"
            )
        elif t == "completed":
            completed = event
            wc = event.get("word_count", 0)
            did = event.get("draft_id", "")
            if wc:
                console.print(f"\n[green]\u2713 Done[/green]  ({wc} words)")
            if "n_inserted" in event:
                # Phase 46.A — citation-insert completion summary
                ni = event.get("n_inserted", 0)
                ns = event.get("n_skipped", 0)
                nn = event.get("n_needs", 0)
                saved = " [dim](dry-run)[/dim]" if event.get("saved") is False else ""
                console.print(
                    f"\n[green]\u2713[/green] Citation pass done: "
                    f"{ni} inserted / {nn} identified / {ns} skipped{saved}"
                )
            if "meta" in event and isinstance(event.get("meta"), dict):
                # Phase 46.C — ensemble review meta summary
                meta = event["meta"]
                dec  = meta.get("decision", "?")
                ov   = meta.get("overall", "?")
                disagreement = meta.get("disagreement", 0.0) or 0.0
                color = (
                    "green"  if isinstance(ov, (int, float)) and ov >= 7 else
                    "yellow" if isinstance(ov, (int, float)) and ov >= 5 else
                    "red"
                )
                console.print()
                console.print(Rule("[bold]Meta-reviewer verdict[/bold]"))
                console.print(
                    f"  overall=[bold {color}]{ov}[/bold {color}]  "
                    f"soundness={meta.get('soundness', '?')}  "
                    f"presentation={meta.get('presentation', '?')}  "
                    f"contribution={meta.get('contribution', '?')}  "
                    f"confidence={meta.get('confidence', '?')}  "
                    f"decision=[bold]{dec}[/bold]  "
                    f"disagreement={disagreement:.2f}"
                )
                strengths = meta.get("strengths") or []
                weaknesses = meta.get("weaknesses") or []
                if strengths:
                    console.print("  [green]Strengths[/green]:")
                    for s in strengths[:5]:
                        console.print(f"    [green]+[/green] {s[:160]}")
                if weaknesses:
                    console.print("  [red]Weaknesses[/red]:")
                    for w in weaknesses[:5]:
                        console.print(f"    [red]-[/red] {w[:160]}")
                r = meta.get("rationale") or ""
                if r:
                    console.print(f"  [dim]{r[:300]}[/dim]")
            if did:
                console.print(f"[dim]Draft ID: {did[:8]}[/dim]")
        elif t == "error":
            console.print(f"[red]Error:[/red] {event.get('message', 'unknown')}")
    return completed


def _auto_summarize(content: str, section_type: str, chapter_title: str, model: str | None = None) -> str:
    """Generate a 100-200 word summary of a draft for cross-chapter context."""
    from sciknow.rag import prompts
    from sciknow.rag.llm import complete_with_status
    system, user = prompts.draft_summary(section_type, chapter_title, content)
    try:
        return complete_with_status(
            system, user, label="Summarizing for coherence",
            model=model, temperature=0.1, num_ctx=4096,
        ).strip()
    except Exception as exc:
        import logging
        logging.getLogger("sciknow.book").warning(
            "Auto-summarize failed for %s/%s: %s", chapter_title, section_type, exc)
        return ""


def _save_draft(session, *, title, book_id, chapter_id, section_type, topic,
                content, sources, model, summary=None, parent_draft_id=None,
                review_feedback=None, version=1):
    """Insert a draft row and return the draft_id."""
    from sqlalchemy import text
    row = session.execute(text("""
        INSERT INTO drafts (title, book_id, chapter_id, section_type, topic, content,
                            word_count, sources, model_used, version, summary,
                            parent_draft_id, review_feedback)
        VALUES (:title, :book_id, :chapter_id, :section, :topic, :content,
                :wc, CAST(:sources AS jsonb), :model, :version, :summary,
                :parent_id, :review_feedback)
        RETURNING id::text
    """), {
        "title": title,
        "book_id": book_id,
        "chapter_id": chapter_id,
        "section": section_type,
        "topic": topic,
        "content": content,
        "wc": len(content.split()),
        "sources": _json.dumps(sources or []),
        "model": model,
        "version": version,
        "summary": summary,
        "parent_id": parent_draft_id,
        "review_feedback": review_feedback,
    })
    session.commit()
    return row.fetchone()[0]


# ── create ─────────────────────────────────────────────────────────────────────

@app.command()
def create(
    title: Annotated[str, typer.Argument(help="Book title.")],
    description: str | None = typer.Option(None, "--description", "-d"),
    type: str = typer.Option(
        "scientific_book", "--type", "-t",
        help=(
            "Project type. Drives section defaults, length targets, "
            "prompt conditioning, and export defaults. Run "
            "`sciknow book types` to list available types."
        ),
    ),
    target_chapter_words: int | None = typer.Option(
        None, "--target-chapter-words",
        help=(
            "Target words per chapter for autowrite/write. Defaults "
            "to the project type's default (book: 6400, paper: 4000). "
            "Sections get a proportional share of this budget."
        ),
    ),
    bootstrap: bool = typer.Option(
        True, "--bootstrap/--no-bootstrap",
        help=(
            "For flat project types (e.g. scientific_paper), auto-create "
            "a single chapter with the type's canonical section template. "
            "Has no effect on hierarchical types like scientific_book."
        ),
    ),
):
    """
    Create a new project.

    Phase 45 — projects now carry a **type**. The type decides the default
    section set, length targets, and downstream prompt conditioning:

      - ``scientific_book``  (default)  hierarchical book → chapters → sections
      - ``scientific_paper``            flat IMRaD paper (one chapter, canonical sections)

    Run ``sciknow book types`` for the full list + each type's template.

    Examples:

      sciknow book create "Global Cooling: The Coming Solar Minimum"

      sciknow book create "Climate-Volcano Coupling" --type scientific_paper

      sciknow book create "Solar Climate" --description "The role of the sun in climate change"

      sciknow book create "Long Book" --target-chapter-words 10000
    """
    import json as _json
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    from sciknow.core.project_type import (
        default_sections_as_dicts, get_project_type, validate_type_slug,
    )

    try:
        validate_type_slug(type)
    except ValueError as exc:
        console.print(f"[red]--type:[/red] {exc}")
        raise typer.Exit(2)
    pt = get_project_type(type)

    with get_session() as session:
        existing = session.execute(text(
            "SELECT id FROM books WHERE title = :t"
        ), {"t": title}).fetchone()
        if existing:
            console.print(f"[yellow]Book already exists:[/yellow] {title}")
            raise typer.Exit(1)

        # Phase 54.6.158 — only write target_chapter_words when the
        # user explicitly passed --target-chapter-words. Pre-54.6.158
        # this always wrote ``pt.default_target_chapter_words`` into
        # custom_metadata, which froze the creation-time default and
        # shadowed the Level-3 fallback forever — including the
        # research-grounded 54.6.146 updates that bumped
        # scientific_book from 6400 to 8000 (etc.). Every existing
        # book now needs `book set-target --unset` to unfreeze; new
        # books just inherit whichever project-type default is current.
        custom_meta: dict = {}
        if target_chapter_words is not None:
            if target_chapter_words <= 0:
                console.print("[red]--target-chapter-words must be positive[/red]")
                raise typer.Exit(1)
            custom_meta["target_chapter_words"] = target_chapter_words

        result = session.execute(text("""
            INSERT INTO books (title, description, book_type, custom_metadata)
            VALUES (:title, :desc, :btype, CAST(:meta AS jsonb))
            RETURNING id::text
        """), {
            "title": title, "desc": description, "btype": pt.slug,
            "meta": _json.dumps(custom_meta),
        })
        book_id = result.fetchone()[0]
        session.commit()

        # Bootstrap a single chapter with the type's section template
        # for flat types — this is the "one project = one document"
        # shape for papers, literature reviews, grant proposals, etc.
        if bootstrap and pt.is_flat:
            sections_json = _json.dumps(default_sections_as_dicts(pt))
            session.execute(text("""
                INSERT INTO book_chapters
                  (book_id, number, title, description, sections)
                VALUES
                  (CAST(:book_id AS uuid), 1, :ch_title, :ch_desc,
                   CAST(:sections AS jsonb))
            """), {
                "book_id": book_id,
                "ch_title": title,
                "ch_desc": description or f"{pt.display_name} — {title}",
                "sections": sections_json,
            })
            session.commit()

    console.print(
        f"[green]✓ Created {pt.display_name}:[/green] [bold]{title}[/bold] "
        f"[dim](type={pt.slug}, id={book_id[:8]})[/dim]"
    )
    if pt.is_flat and bootstrap:
        console.print(
            "[dim]  Auto-created one chapter with sections: "
            + ", ".join(s.key for s in pt.default_sections) + "[/dim]"
        )
        console.print(
            "\nNext steps:\n"
            f"  [bold]sciknow book serve {title!r}[/bold]  — open in the browser\n"
            f"  [bold]sciknow book autowrite {title!r} --chapter 1 --full[/bold]  — write every section"
        )
    else:
        console.print(
            "\nNext steps:\n"
            f"  [bold]sciknow book outline {title!r}[/bold]   — generate a chapter structure\n"
            f"  [bold]sciknow book chapter add {title!r} \"Chapter Title\"[/bold]  — add chapters manually"
        )


@app.command(name="types")
def types():
    """List available project types with their research-grounded length ranges.

    Phase 45 shipped the registry; Phase 54.6.146 added concept-density
    metadata; Phase 54.6.147 surfaces it here. Columns:

      Chapter default   — books.custom_metadata.target_chapter_words fallback
      Section (mid)     — what Level-0 concept-density will target for
                          a plan with 3-4 bullets (range = low × wpc_mid,
                          high × wpc_mid)
      Concepts/sec      — concepts_per_section_range from ProjectType
                          (Cowan 2001 novel-chunk bound)
      Words/concept     — words_per_concept_range from ProjectType
                          (derived from literature per genre)

    See docs/RESEARCH.md §24 for the research justification behind
    every number.
    """
    from rich.table import Table as _RT
    from sciknow.core.project_type import list_project_types

    table = _RT(title="Project Types (with Phase 54.6.146 concept-density metadata)")
    table.add_column("Slug", style="bold", overflow="fold")
    table.add_column("Display", style="cyan", overflow="fold")
    table.add_column("Flat", justify="center", width=4)
    table.add_column("Chapter\ndefault", justify="right", width=8)
    table.add_column("Section\n(mid)",   justify="right", overflow="fold")
    table.add_column("Concepts\n/section", justify="center", width=9)
    table.add_column("Words\n/concept",    justify="center", width=10)
    table.add_column("Description", style="dim", overflow="fold")
    for pt in list_project_types():
        clo, chi = pt.concepts_per_section_range
        wlo, whi = pt.words_per_concept_range
        wmid = (wlo + whi) // 2
        slo, shi = clo * wmid, chi * wmid
        table.add_row(
            pt.slug,
            pt.display_name,
            "●" if pt.is_flat else "",
            f"{pt.default_target_chapter_words:,}",
            f"{slo:,}–{shi:,}",
            f"{clo}–{chi}",
            f"{wlo}–{whi}",
            pt.description,
        )
    console.print(table)
    console.print(
        "[dim]Concept-density sizing: sections with a bullet plan "
        "auto-size as N × (wpc midpoint). See `docs/RESEARCH.md §24`.[/dim]"
    )


# ── set-target (Phase 54.6.143) ────────────────────────────────────────────────

@app.command(name="set-target")
def set_target(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    words: int = typer.Option(
        None, "--words", "-w",
        help="Target words for a chapter. Omit with --unset to clear.",
    ),
    chapter: str = typer.Option(
        None, "--chapter", "-c",
        help=(
            "Chapter number or title fragment. When set, only this chapter's "
            "target is changed. When omitted, the book-level default "
            "(applies to every chapter without its own override) is changed."
        ),
    ),
    unset: bool = typer.Option(
        False, "--unset",
        help="Clear the target back to the inherited default. Requires "
             "--chapter (use the book's book_type default for the whole book).",
    ),
):
    """Phase 54.6.143 — set the chapter-level word target.

    Resolution during autowrite (highest → lowest priority):

      1. ``--target-words`` passed to ``book autowrite``
      2. per-section override (chapter modal's Sections tab)
      3. ``book_chapters.target_words``       (this command, per-chapter)
      4. ``books.custom_metadata.target_chapter_words`` (this command, book-wide)
      5. ``project_type.default_target_chapter_words`` (from ``book types``)
      6. hardcoded 6000 fallback

    Examples:

      # Set every chapter of a textbook to 15000 words
      sciknow book set-target "Intro ML Textbook" --words 15000

      # Override just chapter 3 to 8000 words (dense methods chapter)
      sciknow book set-target "Intro ML Textbook" --chapter 3 --words 8000

      # Clear the per-chapter override, let chapter 3 inherit again
      sciknow book set-target "Intro ML Textbook" --chapter 3 --unset
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    from sciknow.core.project_type import get_project_type

    if unset and words is not None:
        console.print("[red]--unset and --words are mutually exclusive.[/red]")
        raise typer.Exit(2)
    if not unset and (words is None or words <= 0):
        console.print("[red]--words must be a positive integer (or use --unset).[/red]")
        raise typer.Exit(2)

    with get_session() as session:
        # Resolve book
        row = session.execute(text("""
            SELECT id::text, title, book_type, COALESCE(custom_metadata, '{}'::jsonb)
            FROM books WHERE title ILIKE :q OR id::text LIKE :q
            ORDER BY updated_at DESC LIMIT 1
        """), {"q": f"%{book_title}%"}).fetchone()
        if not row:
            console.print(f"[red]No book matches {book_title!r}[/red]")
            raise typer.Exit(1)
        book_id, b_title, b_type, meta = row[0], row[1], row[2], row[3]

        if chapter:
            ch_row = session.execute(text("""
                SELECT id::text, number, title FROM book_chapters
                WHERE book_id = CAST(:bid AS uuid)
                  AND (CAST(number AS text) = :q OR title ILIKE :qt)
                ORDER BY number LIMIT 1
            """), {"bid": book_id, "q": chapter.strip(),
                   "qt": f"%{chapter.strip()}%"}).fetchone()
            if not ch_row:
                console.print(
                    f"[red]No chapter matches {chapter!r} in {b_title!r}[/red]"
                )
                raise typer.Exit(1)
            ch_id, ch_num, ch_title = ch_row
            if unset:
                session.execute(text(
                    "UPDATE book_chapters SET target_words = NULL "
                    "WHERE id = CAST(:cid AS uuid)"
                ), {"cid": ch_id})
                session.commit()
                console.print(
                    f"[green]✓ Cleared per-chapter target[/green] for "
                    f"Ch.{ch_num} {ch_title!r} — will inherit from book."
                )
                return
            session.execute(text(
                "UPDATE book_chapters SET target_words = :w "
                "WHERE id = CAST(:cid AS uuid)"
            ), {"w": int(words), "cid": ch_id})
            session.commit()
            pt = get_project_type(b_type)
            console.print(
                f"[green]✓ Set[/green] Ch.{ch_num} {ch_title!r} "
                f"target to [bold]{words:,}[/bold] words  "
                f"[dim](book default: {pt.default_target_chapter_words:,}, "
                f"book type: {b_type})[/dim]"
            )
            return

        # Book-level default — Phase 54.6.158 now supports --unset
        # for dropping the custom_metadata.target_chapter_words key so
        # the Level-3 project-type default takes over. This is the
        # unfreezer for pre-54.6.158 books that carry a creation-time
        # default as a stale override (see `book create` changelog).
        if isinstance(meta, str):
            import json as _json
            meta = _json.loads(meta or "{}")
        meta = dict(meta or {})
        pt = get_project_type(b_type)
        if unset:
            if "target_chapter_words" not in meta:
                console.print(
                    f"[yellow]No book-level target set on {b_title!r} — "
                    f"already inheriting the project-type default "
                    f"({pt.default_target_chapter_words:,} for {b_type}).[/yellow]"
                )
                return
            prior = meta.pop("target_chapter_words", None)
            session.execute(text(
                "UPDATE books SET custom_metadata = CAST(:m AS jsonb) "
                "WHERE id = CAST(:bid AS uuid)"
            ), {"m": __import__("json").dumps(meta), "bid": book_id})
            session.commit()
            console.print(
                f"[green]✓ Cleared book-level target[/green] for {b_title!r}  "
                f"[dim](was {int(prior):,}; now inherits project-type default "
                f"{pt.default_target_chapter_words:,} for {b_type})[/dim]"
            )
            return
        meta["target_chapter_words"] = int(words)
        session.execute(text(
            "UPDATE books SET custom_metadata = CAST(:m AS jsonb) "
            "WHERE id = CAST(:bid AS uuid)"
        ), {"m": __import__("json").dumps(meta), "bid": book_id})
        session.commit()
        console.print(
            f"[green]✓ Set book-level target[/green] for {b_title!r} "
            f"to [bold]{words:,}[/bold] words  "
            f"[dim](project-type default: {pt.default_target_chapter_words:,}, "
            f"book type: {b_type})[/dim]"
        )


# ── length-report (Phase 54.6.153) ─────────────────────────────────────────────

@app.command(name="length-report")
def length_report(
    book_title: Annotated[str, typer.Argument(
        help="Book title or ID fragment.",
    )],
    output_json: bool = typer.Option(
        False, "--json",
        help="Emit machine-readable JSON instead of the Rich table. "
             "Useful for piping into jq or another script.",
    ),
):
    """Phase 54.6.153 — whole-book projected length report.

    Walks every chapter × every section, runs the same resolver chain
    autowrite uses (explicit override → concept-density → chapter-split),
    and prints a table showing per-section target + level, chapter
    totals, and the whole-book total. Answers "how long is my book
    going to be?" before you start autowriting.

    Notes:

    - Targets shown are the pre-widener values. The Phase 54.6.150
      retrieval-density widener would normally nudge ±wpc_range based
      on actual retrieved-chunk counts, but firing retrieval for every
      section just to preview targets is too expensive. Final
      autowrite-time targets are typically within ±50% of what this
      reports.
    - Section levels:
        `override`        — per-section target_words set explicitly
        `concept_density` — bottom-up from section plan (54.6.146)
        `chapter_split`   — top-down chapter target ÷ num sections

    Examples:

      sciknow book length-report "Global Cooling"
      sciknow book length-report "Global Cooling" --json | jq .total_words
    """
    from rich.table import Table as _RT
    from rich import box as _rbox
    from sqlalchemy import text
    from sciknow.core.length_report import walk_book_lengths
    from sciknow.storage.db import get_session

    # Resolve book
    with get_session() as session:
        row = session.execute(text("""
            SELECT id::text, title FROM books
            WHERE title ILIKE :q OR id::text LIKE :q
            ORDER BY updated_at DESC LIMIT 1
        """), {"q": f"%{book_title}%"}).fetchone()
    if not row:
        console.print(f"[red]No book matches {book_title!r}[/red]")
        raise typer.Exit(1)
    book_id, resolved_title = row

    try:
        report = walk_book_lengths(book_id)
    except Exception as exc:
        console.print(f"[red]length-report failed: {exc}[/red]")
        raise typer.Exit(2)

    if output_json:
        import json as _json
        console.print_json(_json.dumps(report.to_dict()))
        return

    # Rich table: sections nested under chapters, with running totals
    console.print(
        f"\n[bold]{report.title}[/bold]  ·  type: [cyan]{report.book_type}[/cyan]  ·  "
        f"{report.n_chapters} chapter(s), {report.n_sections} section(s)\n"
    )

    for ch in report.chapters:
        ch_pct = 100.0 * ch.total_words / max(1, report.total_words)
        console.print(
            f"[bold]Ch.{ch.number}[/bold] {ch.title!r}  ·  "
            f"chapter target [cyan]{ch.chapter_target:,}[/cyan] "
            f"[dim]({ch.chapter_level})[/dim]  ·  "
            f"section sum [yellow]{ch.total_words:,}[/yellow] words "
            f"[dim]({ch_pct:.1f}% of book)[/dim]"
        )
        t = _RT(box=_rbox.MINIMAL, show_header=True, pad_edge=False,
                show_lines=False)
        t.add_column("  §", width=4, style="dim")
        t.add_column("Title", overflow="fold")
        t.add_column("Target", justify="right", width=9)
        t.add_column("Level", style="dim")
        t.add_column("Notes", style="dim", overflow="fold")
        for s in ch.sections:
            level_colour = {
                "explicit_section_override": "[green]override[/green]",
                "concept_density":           "[cyan]concept-density[/cyan]",
                "chapter_split":             "[dim]chapter-split[/dim]",
            }.get(s.level, s.level)
            t.add_row(
                f"  {s.slug[:20]}",
                s.title[:40],
                f"{s.target:,}",
                level_colour,
                s.explanation[:60],
            )
        console.print(t)
        console.print("")

    # Footer summary
    hist = report.level_histogram()
    bits = []
    for lvl, count in hist.items():
        bits.append(f"{count} {lvl}")
    hist_str = "  ·  ".join(bits) if bits else "(no sections)"
    console.print(
        f"[bold]Book total:[/bold] [yellow]{report.total_words:,}[/yellow] words across "
        f"{report.n_chapters} chapter(s) and {report.n_sections} section(s)"
    )
    console.print(f"[dim]Section levels: {hist_str}[/dim]")
    console.print(
        "[dim]Pre-widener values. Phase 54.6.150 retrieval-density widener "
        "adjusts ±wpc_range at autowrite time based on retrieved-chunk count.[/dim]"
    )


# ── plan-sections (Phase 54.6.154) ─────────────────────────────────────────────

@app.command(name="plan-sections")
def plan_sections(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    chapter: str = typer.Option(
        None, "--chapter", "-c",
        help="Chapter number or title fragment. Omit to plan every chapter.",
    ),
    model: str = typer.Option(
        None, "--model",
        help="LLM to use. Defaults to LLM_FAST_MODEL (structured output, "
             "no need for the flagship writer).",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Overwrite existing plans. Default: skip sections that "
             "already have a plan.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show which sections would be planned without calling the "
             "LLM or writing to the DB.",
    ),
):
    """Phase 54.6.154 — auto-generate per-section concept plans via LLM.

    For each section in the (filtered) chapters, asks the LLM for a
    3-4 bullet concept list framed by the chapter scope and section
    title. The resulting plan is written into ``book_chapters.sections[i].plan``,
    activating the Phase 54.6.146 bottom-up concept-density resolver
    the next time autowrite runs on that section.

    Resolver impact: sections that were previously on the
    ``chapter_split`` fallback (top-down chapter_target ÷ num_sections)
    flip to ``concept_density`` (N bullets × wpc_midpoint). Run
    ``sciknow book length-report`` before and after to see the shift.

    Examples:

      sciknow book plan-sections "Global Cooling"                  # whole book
      sciknow book plan-sections "Global Cooling" --chapter 3      # one chapter
      sciknow book plan-sections "Global Cooling" --force          # overwrite
      sciknow book plan-sections "Global Cooling" --dry-run        # preview
    """
    from sqlalchemy import text
    from sciknow.core.book_ops import generate_section_plan, _count_plan_concepts
    from sciknow.storage.db import get_session

    with get_session() as session:
        row = session.execute(text("""
            SELECT id::text, title FROM books
            WHERE title ILIKE :q OR id::text LIKE :q
            ORDER BY updated_at DESC LIMIT 1
        """), {"q": f"%{book_title}%"}).fetchone()
        if not row:
            console.print(f"[red]No book matches {book_title!r}[/red]")
            raise typer.Exit(1)
        book_id, resolved_title = row

        chapter_filter = ""
        params: dict = {"bid": book_id}
        if chapter:
            chapter_filter = "AND (CAST(number AS text) = :q OR title ILIKE :qt)"
            params["q"] = chapter.strip()
            params["qt"] = f"%{chapter.strip()}%"
        ch_rows = session.execute(text(f"""
            SELECT id::text, number, title, sections
            FROM book_chapters
            WHERE book_id = CAST(:bid AS uuid) {chapter_filter}
            ORDER BY number
        """), params).fetchall()

    if not ch_rows:
        console.print(
            f"[yellow]No chapters match {chapter!r} in {resolved_title!r}[/yellow]"
        )
        raise typer.Exit(1)

    console.print(
        f"\n[bold]{resolved_title}[/bold]  ·  "
        f"planning {len(ch_rows)} chapter(s)  "
        f"[dim](force={force}, dry_run={dry_run})[/dim]\n"
    )

    total_planned = 0
    total_skipped = 0
    total_failed = 0
    for ch_id, ch_num, ch_title, raw_sections in ch_rows:
        from sciknow.core.book_ops import _normalize_chapter_sections
        sections = _normalize_chapter_sections(raw_sections)
        if not sections:
            console.print(
                f"[dim]Ch.{ch_num} {ch_title!r} — no sections; skipping.[/dim]"
            )
            continue
        console.print(
            f"[bold]Ch.{ch_num}[/bold] {ch_title!r}  ·  "
            f"{len(sections)} section(s)"
        )
        for s in sections:
            slug = s.get("slug", "")
            prior = (s.get("plan") or "").strip()
            has_prior = bool(prior) and _count_plan_concepts(prior) > 0
            if has_prior and not force:
                console.print(
                    f"  [dim]{slug[:24]:24s}[/dim] "
                    f"[dim]→ already planned ({_count_plan_concepts(prior)} "
                    f"bullets), skipping[/dim]"
                )
                total_skipped += 1
                continue
            if dry_run:
                console.print(
                    f"  [cyan]{slug[:24]:24s}[/cyan] "
                    f"[dim]→ WOULD plan (currently "
                    f"{'overwriting existing' if has_prior else 'empty'})[/dim]"
                )
                continue
            console.print(
                f"  [cyan]{slug[:24]:24s}[/cyan] "
                f"[dim]→ calling LLM…[/dim]",
                end="\r",
            )
            try:
                result = generate_section_plan(
                    book_id, ch_id, slug, model=model, force=force,
                )
            except Exception as exc:
                console.print(
                    f"  [red]{slug[:24]:24s}[/red] "
                    f"[red]→ FAIL: {str(exc)[:80]}[/red]"
                )
                total_failed += 1
                continue
            if result["wrote"]:
                n = result["n_concepts"]
                first_bullet = (result["new_plan"].splitlines() or [""])[0][:70]
                console.print(
                    f"  [green]{slug[:24]:24s}[/green] "
                    f"→ [bold]{n} concepts[/bold]  [dim]{first_bullet}[/dim]"
                )
                total_planned += 1
            else:
                reason = result.get("skipped_reason") or "unknown"
                console.print(
                    f"  [yellow]{slug[:24]:24s}[/yellow] "
                    f"[yellow]→ skipped ({reason})[/yellow]"
                )
                total_skipped += 1

    console.print(
        f"\n[bold]Done.[/bold] planned: [green]{total_planned}[/green]  "
        f"skipped: [yellow]{total_skipped}[/yellow]  "
        f"failed: [red]{total_failed}[/red]"
    )
    if total_planned and not dry_run:
        console.print(
            "[dim]Next: run [bold]sciknow book length-report[/bold] to see "
            "how many sections flipped from chapter-split to "
            "concept-density.[/dim]"
        )


# ── list ───────────────────────────────────────────────────────────────────────

@app.command(name="list")
def list_books():
    """List all book projects."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        rows = session.execute(text("""
            SELECT b.id::text, b.title, b.status, b.description, b.book_type,
                   COUNT(DISTINCT bc.id) as n_chapters,
                   COUNT(DISTINCT d.id)  as n_drafts
            FROM books b
            LEFT JOIN book_chapters bc ON bc.book_id = b.id
            LEFT JOIN drafts d ON d.book_id = b.id
            GROUP BY b.id, b.title, b.status, b.description, b.book_type
            ORDER BY b.created_at DESC
        """)).fetchall()

    if not rows:
        console.print("[yellow]No projects yet.[/yellow]  Create one: [bold]sciknow book create \"Title\"[/bold]")
        raise typer.Exit(0)

    table = Table(title="Projects", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("ID",       style="dim",   width=8)
    table.add_column("Title",                   ratio=3)
    table.add_column("Type",     style="magenta", width=18)
    table.add_column("Status",   style="cyan",  width=10)
    table.add_column("Chapters", justify="right", width=9)
    table.add_column("Drafts",   justify="right", width=7)
    table.add_column("Description",             ratio=2)

    for row in rows:
        btype = (row[4] or "scientific_book").replace("scientific_", "")
        table.add_row(
            row[0][:8], row[1], btype, row[2],
            str(row[5]), str(row[6]),
            (row[3] or "")[:60],
        )
    console.print(table)


# ── show ───────────────────────────────────────────────────────────────────────

@app.command()
def show(
    title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
):
    """Show a book's chapters, draft status, and progress."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, title)
        if not book:
            console.print(f"[red]Book not found:[/red] {title}")
            raise typer.Exit(1)

        chapters = session.execute(text("""
            SELECT bc.number, bc.title, bc.description, bc.topic_query, bc.topic_cluster,
                   COUNT(d.id) as n_drafts,
                   COALESCE(SUM(d.word_count), 0) as total_words
            FROM book_chapters bc
            LEFT JOIN drafts d ON d.chapter_id = bc.id
            WHERE bc.book_id = :bid
            GROUP BY bc.number, bc.title, bc.description, bc.topic_query, bc.topic_cluster
            ORDER BY bc.number
        """), {"bid": book[0]}).fetchall()

    console.print()
    console.print(f"[bold cyan]{book[1]}[/bold cyan]  [dim]({book[3]})[/dim]")
    if book[2]:
        console.print(f"  [dim]{book[2]}[/dim]")
    console.print()

    if not chapters:
        console.print("  [yellow]No chapters yet.[/yellow]")
        console.print(f"  Run: [bold]sciknow book outline {title!r}[/bold]")
        return

    table = Table(box=box.SIMPLE_HEAD, show_header=True, expand=True)
    table.add_column("Ch.", style="bold cyan", width=4)
    table.add_column("Title",                  ratio=2)
    table.add_column("Topic Query",  style="dim", ratio=2)
    table.add_column("Cluster",     style="dim", width=16)
    table.add_column("Drafts",      justify="right", width=7)
    table.add_column("Words",       justify="right", width=7)

    for ch in chapters:
        status_color = "green" if ch[5] > 0 else "yellow"
        table.add_row(
            str(ch[0]),
            ch[1],
            (ch[3] or "")[:40],
            (ch[4] or "")[:16],
            f"[{status_color}]{ch[5]}[/{status_color}]",
            str(ch[6]) if ch[6] else "—",
        )
    console.print(table)
    console.print()


# ── chapter add ────────────────────────────────────────────────────────────────

@chapter_app.command(name="add")
def chapter_add(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    title:      Annotated[str, typer.Argument(help="Chapter title.")],
    number:     int | None = typer.Option(None, "--number", "-n", help="Chapter number (auto if omitted)."),
    description: str | None = typer.Option(None, "--description", "-d"),
    topic_query: str | None = typer.Option(None, "--query", "-q",
                                            help="Search query to retrieve relevant papers."),
    topic_cluster: str | None = typer.Option(None, "--cluster", "-c",
                                              help="Topic cluster to scope retrieval."),
):
    """
    Add a chapter to a book.

    Examples:

      sciknow book chapter add "Global Cooling" "Solar Activity and Irradiance" --number 2

      sciknow book chapter add "Global Cooling" "Cosmic Rays and Cloud Nucleation" \\
          --query "cosmic rays cloud nucleation climate" --number 3
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

        if number is None:
            max_n = session.execute(text(
                "SELECT COALESCE(MAX(number), 0) FROM book_chapters WHERE book_id = :bid"
            ), {"bid": book[0]}).scalar()
            number = max_n + 1

        session.execute(text("""
            INSERT INTO book_chapters (book_id, number, title, description, topic_query, topic_cluster)
            VALUES (:bid, :num, :title, :desc, :tq, :tc)
        """), {
            "bid": book[0], "num": number, "title": title,
            "desc": description, "tq": topic_query, "tc": topic_cluster,
        })
        session.commit()

    console.print(
        f"[green]✓ Added Chapter {number}:[/green] [bold]{title}[/bold] "
        f"→ [bold]{book[1]}[/bold]"
    )


# ── outline ────────────────────────────────────────────────────────────────────

@app.command()
def outline(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    save:  bool = typer.Option(True,  "--save/--no-save",
                                help="Save proposed chapters to the database."),
    model: str | None = typer.Option(None, "--model", help="Override LLM model."),
    overwrite: str | None = typer.Option(
        None, "--overwrite",
        help="Re-outline an already-outlined book. One of: 'archive' "
             "(drop chapters, KEEP drafts as orphans — recoverable via the "
             "sidebar's + adopt button or via snapshot-restore), or 'hard' "
             "(drop chapters AND delete drafts; irreversible without a "
             "snapshot). Both modes auto-snapshot first as a safety net "
             "(skip with --no-snapshot). Without this flag, existing "
             "chapters are preserved and the LLM's new chapters are "
             "merged in by `number` (the legacy behaviour).",
    ),
    snapshot_first: bool = typer.Option(
        True, "--snapshot/--no-snapshot",
        help="With --overwrite, take a book-wide draft snapshot before "
             "wiping anything. Default ON; pass --no-snapshot to skip "
             "(only sensible when you've just snapshotted by hand).",
    ),
):
    """
    Generate a proposed chapter structure using the LLM and your paper collection.

    Examples:

      sciknow book outline "Global Cooling"

      sciknow book outline "Global Cooling" --no-save                # preview only
      sciknow book outline "Global Cooling" --overwrite=archive      # drafts → orphans
      sciknow book outline "Global Cooling" --overwrite=hard         # delete drafts too
    """
    import json
    from sqlalchemy import text
    from sciknow.config import settings
    from sciknow.rag import prompts
    from sciknow.rag.llm import complete
    from sciknow.storage.db import get_session

    from sciknow.core.project_type import get_project_type

    if overwrite is not None and overwrite not in ("archive", "hard"):
        console.print(
            f"[red]--overwrite must be 'archive' or 'hard'; got {overwrite!r}.[/red]"
        )
        raise typer.Exit(2)

    # Phase 54.6.297 — per-role outline model override.  CLI `--model`
    # arg wins > BOOK_OUTLINE_MODEL env > LLM_MODEL (default in the
    # llm.stream helper).  Resolve once so both the tournament loop
    # and the fallback single-pass path use the same choice.
    effective_model = (
        model or getattr(settings, "book_outline_model", None) or None
    )

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

        # Flat project types (scientific_paper) don't have a chapter
        # outline — the one chapter + canonical sections was bootstrapped
        # at `book create` time. Refuse to overwrite that shape here.
        pt = get_project_type(book[6] if len(book) > 6 else None)
        if pt.is_flat:
            console.print(
                f"[yellow]Skipping outline:[/yellow] project type [bold]{pt.slug}[/bold] "
                "is flat (one chapter, canonical sections auto-created at project init)."
            )
            console.print(
                f"  To customize sections, edit chapter 1 directly: "
                f"[bold]sciknow book chapter show {book_title!r} 1[/bold]"
            )
            raise typer.Exit(0)

        papers = session.execute(text(
            "SELECT title, year FROM paper_metadata ORDER BY year DESC NULLS LAST"
        )).fetchall()

        # Phase 54.6.x — gather topic-cluster catalogue with
        # per-cluster representative abstracts so the LLM sees what
        # the corpus actually contains. Mirrors the web flow in
        # web/routes/book.py:api_book_outline_generate.
        cluster_rows = session.execute(text("""
            SELECT pm.topic_cluster, COUNT(*) AS n
            FROM paper_metadata pm
            JOIN documents d ON d.id = pm.document_id
            WHERE d.ingestion_status = 'complete'
              AND pm.topic_cluster IS NOT NULL
              AND pm.topic_cluster != ''
            GROUP BY pm.topic_cluster
            HAVING COUNT(*) >= 2
            ORDER BY COUNT(*) DESC
        """)).fetchall()
        cluster_catalogue: list[dict] = []
        for cname, n in cluster_rows:
            rep_rows = session.execute(text("""
                SELECT pm.title, pm.year, pm.abstract
                FROM paper_metadata pm
                JOIN documents d ON d.id = pm.document_id
                WHERE d.ingestion_status = 'complete'
                  AND pm.topic_cluster = :tc
                  AND pm.title IS NOT NULL
                ORDER BY pm.year DESC NULLS LAST,
                         LENGTH(COALESCE(pm.abstract, '')) DESC
                LIMIT 3
            """), {"tc": cname}).fetchall()
            cluster_catalogue.append({
                "name": cname,
                "count": int(n or 0),
                "papers": [
                    {
                        "title": r[0],
                        "year": r[1],
                        "abstract": (
                            (r[2] or "").strip().replace("\n", " ")[:280]
                        ),
                    }
                    for r in rep_rows
                ],
            })

    n_clusters = len(cluster_catalogue)
    cluster_msg = f" + {n_clusters} topic clusters" if n_clusters else ""
    console.print(
        f"Generating outline for [bold]{book[1]}[/bold] from "
        f"{len(papers)} papers{cluster_msg}…"
    )

    system, user = prompts.outline(
        book_title=book[1],
        papers=[{"title": p[0], "year": p[1]} for p in papers if p[0]],
        plan=book[5] if len(book) > 5 else None,
        clusters=cluster_catalogue,
    )

    from sciknow.rag.llm import complete_with_status

    # Phase 54.6.26 — tree-search outline generation (from AI-Scientist v2).
    # Generate 3 candidate outlines at higher temperature, then pick the
    # best one by scoring each candidate against corpus coverage. This
    # explores breadth before committing to depth — the same idea as
    # AI-Scientist's agentic tree search, applied at the outline level.
    N_CANDIDATES = 3
    candidates: list[tuple[list, str, float]] = []  # (chapters, raw, score)
    for ci in range(N_CANDIDATES):
        console.print(f"  [dim]Generating candidate {ci + 1}/{N_CANDIDATES}…[/dim]")
        raw_i = complete_with_status(
            system, user,
            label=f"Candidate {ci + 1}/{N_CANDIDATES}",
            model=effective_model, temperature=0.5 + ci * 0.15,  # 0.5, 0.65, 0.8
            num_ctx=16384,
        )
        raw_i = raw_i.strip()
        if raw_i.startswith("```"):
            raw_i = raw_i.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            data_i = json.loads(raw_i, strict=False)
            ch_i = data_i.get("chapters", [])
        except Exception:
            continue
        if not ch_i:
            continue
        # Score: reward breadth (chapter count + unique titles + total
        # sections) but also reward VARIANCE in per-chapter section
        # counts (Phase 54.6.65). Pre-fix the scorer was monotonic in
        # total-section-count, which biased the tournament toward
        # candidates that gave every chapter the max section count —
        # producing uniform outlines where 8 chapters × 5 sections = 40
        # always beat 8 chapters × (3..7 varied) = 38.
        n_ch = len(ch_i)
        sec_counts = [len(c.get("sections", [])) for c in ch_i]
        n_sec = sum(sec_counts)
        unique_titles = len(set(c.get("title", "") for c in ch_i))
        # Stddev of section counts, 0 when all equal. Scale up so a
        # varied candidate (stddev ≈ 1.5) adds ≈3 to the score —
        # enough to outrank a flat-max candidate with 2 extra sections.
        if len(sec_counts) > 1:
            _m = sum(sec_counts) / len(sec_counts)
            _var = sum((s - _m) ** 2 for s in sec_counts) / len(sec_counts)
            section_variance = _var ** 0.5
        else:
            section_variance = 0.0
        score_i = (n_ch * 0.3 + n_sec * 0.3 + unique_titles * 0.2
                   + section_variance * 2.0)
        candidates.append((ch_i, raw_i, score_i))

    if not candidates:
        console.print("[red]All candidate outlines failed. Trying single pass…[/red]")
        raw = complete_with_status(system, user, label="Generating outline",
                                   model=effective_model, temperature=0.3, num_ctx=16384)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(raw, strict=False)
            chapters = data.get("chapters", [])
        except Exception:
            console.print("[red]LLM returned invalid JSON. Raw output:[/red]")
            console.print(raw)
            raise typer.Exit(1)
    else:
        candidates.sort(key=lambda c: c[2], reverse=True)
        chapters, raw, best_score = candidates[0]
        console.print(
            f"  [green]✓ Picked best of {len(candidates)} candidates "
            f"(score {best_score:.1f}, {len(chapters)} chapters)[/green]"
        )

    if not chapters:
        console.print("[yellow]No chapters in LLM response.[/yellow]")
        raise typer.Exit(1)

    # Phase 54.6.65 — resize sections to match corpus evidence density.
    # One hybrid-retrieval per chapter on the chapter's topic_query;
    # section lists get trimmed to a bucket derived from the number of
    # distinct papers retrieved. Runs only when the retrieval stack is
    # reachable; offline invocations fall through silently.
    from sciknow.core.book_ops import resize_sections_by_density as _resize
    console.print("  [dim]Resizing sections by corpus evidence density…[/dim]")
    chapters = _resize(chapters, model=effective_model)

    # Phase 54.6.x — DEEP outline post-pass. Mirror of the web flow:
    # for every chapter × section, run hybrid retrieval, build a
    # SECTION_PLAN prompt grounded in leitmotiv + evidence + earlier
    # chapters, and attach the resulting bullet list + target_words
    # to each section so it's saved as a full
    # {slug, title, plan, target_words} dict.
    from sciknow.core.book_ops import deep_plan_outline_chapters as _deep
    book_type = (book[6] if len(book) > 6 else None) or "scientific_book"
    book_plan_text = book[5] if len(book) > 5 else None
    console.print(
        "  [dim]Deep section planning "
        "(per-section retrieval + leitmotiv-grounded concept lists)…[/dim]"
    )
    n_planned_total = 0
    n_failed_total = 0
    for evt in _deep(
        chapters,
        book_title=book[1],
        book_type=book_type,
        book_plan=book_plan_text,
        model=effective_model,
    ):
        et = evt.get("type")
        if et == "deep_plan_section_start":
            console.print(
                f"     [dim]Ch.{evt['chapter_index']}/{evt['chapter_total']} "
                f"§{evt['section_index']}/{evt['section_total']} — "
                f"{evt['section_title']}[/dim]"
            )
        elif et == "deep_plan_section_done":
            if evt.get("error"):
                n_failed_total += 1
            else:
                n_planned_total += 1
        elif et == "deep_plan_complete":
            n_planned_total = evt.get("n_planned", n_planned_total)
            n_failed_total = evt.get("n_failed", n_failed_total)
    console.print(
        f"  [dim]Deep planning: "
        f"{n_planned_total} section(s) planned, "
        f"{n_failed_total} failed.[/dim]"
    )

    def _section_label(s) -> str:
        if isinstance(s, dict):
            return s.get("title") or s.get("slug") or "?"
        return str(s)

    # Display
    console.print()
    console.print(Rule(f"[bold]Proposed outline: {book[1]}[/bold]"))
    console.print()
    for ch in chapters:
        console.print(f"  [bold cyan]Ch.{ch['number']}[/bold cyan]  {ch['title']}")
        if ch.get("description"):
            console.print(f"         [dim]{ch['description']}[/dim]")
        if ch.get("topic_query"):
            console.print(f"         [dim]Query: {ch['topic_query']}[/dim]")
        if ch.get("sections"):
            console.print(
                f"         Sections: "
                f"{' → '.join(_section_label(s) for s in ch['sections'])}"
            )
        di = ch.get("_density_info")
        if di:
            note = f"{di['n_papers']} papers → target {di['target']}"
            action = di.get("action", "")
            if action and action != "kept":
                note += (f" ({action}"
                         f" from {di['original_count']} → {di['final_count']})")
            console.print(f"         [dim]Density: {note}[/dim]")
        console.print()

    if save:
        with get_session() as session:
            # Inspect current chapter/draft counts so we can prompt the
            # user when they're about to clobber something.
            n_existing_chapters = session.execute(text(
                "SELECT COUNT(*) FROM book_chapters WHERE book_id::text = :bid"
            ), {"bid": book[0]}).scalar() or 0
            n_existing_drafts = session.execute(text(
                "SELECT COUNT(*) FROM drafts WHERE book_id::text = :bid"
            ), {"bid": book[0]}).scalar() or 0

            # Refuse merge-mode (--overwrite unset) when the book already
            # has an outline AND would silently re-skip every proposed
            # chapter. The legacy "INSERT only when number doesn't exist"
            # path stays for the rare case where a user manually deleted
            # one chapter and wants a top-up — that produces non-zero
            # inserts even with chapters present, so we only block when
            # EVERY proposed number collides.
            if overwrite is None and n_existing_chapters > 0:
                proposed_nums = {int(ch["number"]) for ch in chapters}
                existing_nums = {
                    int(r[0]) for r in session.execute(text(
                        "SELECT number FROM book_chapters "
                        "WHERE book_id::text = :bid"
                    ), {"bid": book[0]}).fetchall()
                }
                if proposed_nums.issubset(existing_nums):
                    console.print(
                        f"[yellow]Book already has {n_existing_chapters} "
                        f"chapter(s); every proposed number collides.[/yellow]"
                    )
                    console.print(
                        "Re-run with one of:\n"
                        "  [bold]--overwrite=archive[/bold]  drop chapters, "
                        "keep drafts as orphans (recoverable)\n"
                        "  [bold]--overwrite=hard[/bold]     drop chapters "
                        "AND delete drafts (irreversible without snapshot)\n"
                        "Both modes auto-snapshot first."
                    )
                    raise typer.Exit(1)

            # Overwrite path — snapshot then wipe.
            if overwrite is not None and n_existing_chapters > 0:
                if snapshot_first and n_existing_drafts > 0:
                    from sciknow.web.app import _snapshot_book_drafts
                    snap_id = _snapshot_book_drafts(
                        session, book[0],
                        name=f"pre-reoutline-{overwrite}",
                    )
                    console.print(
                        f"[dim]Snapshot {str(snap_id)[:8]} captured "
                        f"({n_existing_drafts} drafts). Restore with: "
                        f"sciknow book snapshot-restore {str(snap_id)[:8]}[/dim]"
                    )
                if overwrite == "hard":
                    n_drafts_deleted = session.execute(text(
                        "DELETE FROM drafts WHERE book_id::text = :bid"
                    ), {"bid": book[0]}).rowcount or 0
                    console.print(
                        f"[yellow]Deleted {n_drafts_deleted} draft(s).[/yellow]"
                    )
                else:  # archive
                    n_archived = session.execute(text(
                        "UPDATE drafts SET chapter_id = NULL "
                        "WHERE book_id::text = :bid AND chapter_id IS NOT NULL"
                    ), {"bid": book[0]}).rowcount or 0
                    console.print(
                        f"[dim]Unlinked {n_archived} draft(s) from chapters "
                        f"(now visible as orphans in the sidebar).[/dim]"
                    )
                n_chapters_dropped = session.execute(text(
                    "DELETE FROM book_chapters WHERE book_id::text = :bid"
                ), {"bid": book[0]}).rowcount or 0
                console.print(
                    f"[yellow]Dropped {n_chapters_dropped} chapter(s).[/yellow]"
                )

            for ch in chapters:
                existing = session.execute(text("""
                    SELECT id FROM book_chapters WHERE book_id = :bid AND number = :num
                """), {"bid": book[0], "num": ch["number"]}).fetchone()
                if existing:
                    continue
                sections_json = _json.dumps(ch.get("sections", []))
                session.execute(text("""
                    INSERT INTO book_chapters (book_id, number, title, description, topic_query, sections)
                    VALUES (:bid, :num, :title, :desc, :tq, CAST(:secs AS jsonb))
                """), {
                    "bid": book[0],
                    "num": ch["number"],
                    "title": ch["title"],
                    "desc": ch.get("description"),
                    "tq": ch.get("topic_query"),
                    "secs": sections_json,
                })
            session.commit()
        console.print(f"[green]✓ Saved {len(chapters)} chapters to database.[/green]")
        console.print(f"Run [bold]sciknow book show {book_title!r}[/bold] to review.")


# ── adopt-section ─────────────────────────────────────────────────────────────

@app.command(name="adopt-section")
def adopt_section(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    chapter: Annotated[str, typer.Argument(help="Chapter number or title fragment.")],
    section_slug: Annotated[str, typer.Argument(
        help="Section slug to adopt (the section_type of the orphan draft).",
    )],
    title: str = typer.Option(
        None, "--title", "-t",
        help="Display title for the section. Defaults to titleified slug.",
    ),
    plan: str = typer.Option(
        None, "--plan", "-p",
        help="Section plan text. Defaults to empty.",
    ),
):
    """Phase 25 — adopt an orphan draft's section_type into a chapter's
    sections list, so it appears as a regular drafted section instead
    of "orphan" in the GUI.

    The motivating scenario: you defined 5 sections on a chapter AFTER
    autowriting an introduction draft. The introduction draft's
    section_type doesn't match any of the new slugs, so the GUI shows
    it with a red "orphan" dot. This command appends the slug to the
    chapter's sections list — the draft content stays unchanged but
    is re-classified from "orphan" to "drafted" on the next refresh.

    Idempotent: if the slug is already in the chapter's sections,
    nothing happens.

    Examples:

      sciknow book adopt-section "The Global Cooling" 2 introduction

      sciknow book adopt-section "The Global Cooling" 2 introduction \\
          --title "Introduction" --plan "Brief framing of solar drivers."
    """
    from sciknow.core.book_ops import adopt_orphan_section
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)
        ch = _get_chapter(session, book[0], chapter)
        if not ch:
            console.print(f"[red]Chapter not found:[/red] {chapter}")
            raise typer.Exit(1)
        book_id = book[0]
        chapter_id = ch[0]
        ch_num = ch[1]
        ch_title = ch[2]

    try:
        result = adopt_orphan_section(
            book_id, chapter_id, section_slug,
            title=title, plan=plan,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    sec = result["section"]
    if result["added"]:
        console.print(
            f"[green]✓ Adopted[/green] section [bold]{sec['slug']}[/bold] "
            f"({sec['title']}) into Ch.{ch_num}: {ch_title}"
        )
        console.print(f"  Total sections in chapter: {len(result['sections'])}")
        console.print(
            "[dim]Refresh the web reader (Ctrl+Shift+R) to see the orphan "
            "draft re-classified as drafted.[/dim]"
        )
    else:
        console.print(
            f"[yellow]No change.[/yellow] Section [bold]{sec['slug']}[/bold] "
            f"is already in Ch.{ch_num}: {ch_title}"
        )


# ── write ──────────────────────────────────────────────────────────────────────

@app.command()
def write(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    chapter:    Annotated[str, typer.Argument(help="Chapter number or title fragment.")],
    section:    str = typer.Option("introduction", "--section", "-s",
                                    help="Section type: introduction, methods, results, discussion, conclusion."),
    context_k:  int = typer.Option(12, "--context-k", "-k"),
    candidates: int = typer.Option(50, "--candidates"),
    year_from:  int | None = typer.Option(None, "--year-from"),
    year_to:    int | None = typer.Option(None, "--year-to"),
    model:      str | None = typer.Option(None, "--model"),
    no_save:    bool = typer.Option(False, "--no-save", help="Print only, don't save to DB."),
    expand:     bool = typer.Option(False, "--expand", "-e", help="Expand query with LLM synonyms before retrieval."),
    show_plan:  bool = typer.Option(False, "--plan", help="Show a sentence plan before drafting."),
    verify:     bool = typer.Option(False, "--verify", help="Run claim verification after drafting."),
    target_words: int | None = typer.Option(
        None, "--target-words",
        help=(
            "Target words for THIS section. Overrides the book-level "
            "target_chapter_words setting. Default: chapter target / "
            "number of sections in the chapter."
        ),
    ),
):
    """
    Draft a chapter section with cross-chapter coherence.

    Injects the book plan and summaries of prior chapters into the prompt
    so the LLM maintains consistency across the book. Auto-generates a
    summary after saving for use by subsequent chapters.

    Examples:

      sciknow book write "Global Cooling" 1 --section introduction

      sciknow book write "Global Cooling" 2 --section methods --plan --verify

      sciknow book write "Global Cooling" 3 --section results --expand
    """
    from sciknow.core.book_ops import write_section_stream
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)
        book_id = book[0]

        ch = _get_chapter(session, book_id, chapter)
        if not ch:
            console.print(f"[red]Chapter not found:[/red] {chapter}")
            raise typer.Exit(1)
        ch_id = ch[0]

    console.print()
    gen = write_section_stream(
        book_id=book_id, chapter_id=ch_id, section_type=section,
        context_k=context_k, candidate_k=candidates,
        year_from=year_from, year_to=year_to,
        model=model, expand=expand, show_plan=show_plan,
        verify=verify, save=not no_save,
        target_words=target_words,
    )
    _consume_events(gen, console)


# ── plan ───────────────────────────────────────────────────────────────────────

@app.command()
def plan(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    model: str | None = typer.Option(None, "--model"),
    edit: bool = typer.Option(False, "--edit", help="Open the plan in $EDITOR for manual editing after generation."),
):
    """
    Generate or view the book plan (thesis + scope document).

    The book plan is a 200-500 word document defining the central argument,
    scope, audience, and key terms. It's injected into every `book write`
    call so all chapters stay aligned.

    If the book already has a plan, prints it. Use --edit to regenerate or
    modify it.

    Examples:

      sciknow book plan "Global Cooling"

      sciknow book plan "Global Cooling" --edit
    """
    from sqlalchemy import text
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import stream as llm_stream, complete as llm_complete
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

        book_id, b_title, b_desc, b_status, b_created, b_plan, _b_type = book

        # If plan exists and not editing, just show it
        if b_plan and not edit:
            console.print()
            console.print(Rule(f"[bold]Book Plan:[/bold] {b_title}"))
            console.print()
            console.print(b_plan)
            console.print()
            console.print("[dim]Use --edit to regenerate or modify.[/dim]")
            return

        # Load chapters and papers for plan generation
        chapters = session.execute(text("""
            SELECT number, title, description FROM book_chapters
            WHERE book_id = :bid ORDER BY number
        """), {"bid": book_id}).fetchall()
        ch_list = [{"number": r[0], "title": r[1], "description": r[2] or ""} for r in chapters]

        papers = session.execute(text("""
            SELECT pm.title, pm.year FROM paper_metadata pm
            JOIN documents d ON d.id = pm.document_id
            WHERE d.ingestion_status = 'complete' AND pm.title IS NOT NULL
            ORDER BY pm.year DESC NULLS LAST
        """)).fetchall()
        paper_list = [{"title": r[0], "year": r[1]} for r in papers]

    # Generate plan
    console.print()
    console.print(Rule(f"[bold]Generating Book Plan:[/bold] {b_title}"))
    console.print()

    sys_p, usr_p = rag_prompts.book_plan(b_title, b_desc, ch_list, paper_list)
    output_tokens = []
    for token in llm_stream(sys_p, usr_p, model=model):
        console.print(token, end="", highlight=False)
        output_tokens.append(token)
    console.print()
    console.print()

    new_plan = "".join(output_tokens).strip()

    # Save
    with get_session() as session:
        session.execute(text("UPDATE books SET plan = :plan WHERE id::text = :bid"),
                        {"plan": new_plan, "bid": book_id})
        session.commit()
    console.print("[green]✓ Book plan saved.[/green]")
    console.print("[dim]This plan will be injected into all future `book write` calls.[/dim]")


# ── insert-citations (Phase 46.A — two-stage citation loop) ─────────────────────

@app.command(name="insert-citations")
def insert_citations(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (first 8+ chars).")],
    model: str | None = typer.Option(None, "--model"),
    candidate_k: int = typer.Option(8, "--candidate-k", "-k",
                                     help="Top-K hybrid-search candidates to show pass-2 per claim."),
    max_needs: int | None = typer.Option(None, "--max-needs",
                                          help="Cap on citation opportunities to process (saves LLM calls)."),
    dry_run: bool = typer.Option(False, "--dry-run",
                                  help="Don't write a new version — print what would change."),
    save: bool = typer.Option(True, "--save/--no-save",
                               help="Persist a new draft version with inserted [N] markers."),
):
    """
    Phase 46.A — two-stage citation insertion (AI-Scientist pattern).

    Pass 1: LLM sees only the draft, identifies where a citation is
            needed, emits {location, claim, query} records.
    Pass 2: For each need, hybrid search retrieves top-K candidates,
            LLM picks (or rejects) with confidence scores.
    Apply:  Deterministic single-pass rewrite inserts [N] markers at
            the identified locations; sources are merged into the
            draft's JSONB ``sources`` column. Saves as a new version.

    This is strictly better than writing-with-citations in one shot
    because placement is auditable, retrieval is budget-bounded, and
    "no good citation available" is an explicit output.

    Examples:

      sciknow book insert-citations 3f2a1b4c

      sciknow book insert-citations 3f2a1b4c --dry-run   # preview

      sciknow book insert-citations 3f2a1b4c --max-needs 5 -k 12
    """
    from sciknow.core.book_ops import insert_citations_stream

    console.print()
    gen = insert_citations_stream(
        draft_id, model=model, candidate_k=candidate_k,
        max_needs=max_needs, dry_run=dry_run, save=save,
    )
    _consume_events(gen, console)


# ── finalize-draft (Phase 54.6.145 — L3 VLM verify pre-export) ──────────────────

@app.command(name="finalize-draft")
def finalize_draft(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (first 8+ chars).")],
    vlm_model: str = typer.Option(
        None, "--vlm-model",
        help="VLM to use. Defaults to settings.visuals_caption_model "
             "(the same model the caption-visuals pipeline uses).",
    ),
    flag_threshold: int = typer.Option(
        4, "--flag-threshold",
        help="VLM scores below this trigger a flag (0-10 scale). "
             "Default 4 — the 5-6 band is 'consistent but not clearly "
             "demonstrated'; below 4 is 'topically wrong or unrelated'.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Write the full JSON report to this file in addition to "
             "the console table.",
    ),
):
    """Phase 54.6.145 — L3 claim-depiction verification for a final draft.

    Per the Q2 design decision (see PHASE_LOG 54.6.142), L3 verification
    runs a vision-language model on every ``[Fig. N]`` marker in the
    draft, checking whether the cited image actually **depicts** the
    claim. It's too expensive for every autowrite iteration (3-10s per
    marker × N markers × M iterations) so it lives here — run once
    before export when the draft is stable.

    Exit code 0 if every marker passes, 1 if any are flagged. The per-
    marker table shows the claim sentence, the VLM's 0-10 score, and
    its one-sentence justification so you can decide: replace the
    figure, drop the citation, or keep it.

    Tables and equations are L3-skipped (no raster image to show the
    VLM) — they trust the L2 entailment check that ran during
    autowrite.

    Examples:

      sciknow book finalize-draft 3f2a1b4c
      sciknow book finalize-draft 3f2a1b4c --flag-threshold 6
      sciknow book finalize-draft 3f2a1b4c --vlm-model minicpm-v:8b
      sciknow book finalize-draft 3f2a1b4c -o finalize.json
    """
    import json as _json
    from rich.table import Table as _RT
    from sciknow.core.finalize_draft import verify_draft_figures_l3

    try:
        report = verify_draft_figures_l3(
            draft_id=draft_id,
            vlm_model=vlm_model,
            flag_threshold=flag_threshold,
            on_progress=lambda i, total, marker: console.print(
                f"  [dim][{i}/{total}][/dim] verifying {marker}…"
            ),
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]finalize-draft failed:[/red] {exc}")
        raise typer.Exit(3)

    if report.n_markers == 0:
        console.print(
            "[yellow]No `[Fig. N]` / `[Table N]` / `[Eq. N]` markers "
            "found in the draft.[/yellow] Nothing to verify."
        )
        raise typer.Exit(0)

    # Results table
    table = _RT(
        title=f"L3 verify · draft {report.draft_id[:12]}… · "
              f"model {report.vlm_model} · {report.elapsed_s:.1f}s",
        show_lines=False, expand=True,
    )
    table.add_column("Marker", style="bold", width=12)
    table.add_column("Kind", width=8)
    table.add_column("Resolved", width=9)
    table.add_column("Score", justify="right", width=6)
    table.add_column("Verdict", width=8)
    table.add_column("Justification / claim", overflow="fold")

    for v in report.verdicts:
        if not v.resolved:
            verdict = "[red]HALLUC[/red]"
            score_s = "—"
            justif = f"Marker doesn't resolve to any visual. Claim: \"{v.claim_sentence[:140]}…\""
        elif v.vlm_score is None:
            verdict = "[yellow]SKIP[/yellow]"
            score_s = "—"
            justif = v.vlm_justification or ""
        else:
            verdict = ("[green]PASS[/green]" if v.passes
                       else "[red]FLAG[/red]")
            score_s = f"{v.vlm_score}/10"
            justif = v.vlm_justification or ""
        table.add_row(
            v.marker, v.kind, "●" if v.resolved else "○",
            score_s, verdict, justif,
        )
    console.print(table)

    # Summary
    console.print(
        f"\n[bold]{report.n_passing}/{report.n_markers} pass[/bold]"
        + (f"  ·  [red]{report.n_flagged} flagged[/red]"
           if report.n_flagged else "")
        + f"  ·  L1 resolved: {report.n_resolved}/{report.n_markers}"
        + f"  ·  pass rate {report.pass_rate:.1%}"
    )

    if output:
        serialised = {
            "draft_id": report.draft_id,
            "vlm_model": report.vlm_model,
            "flag_threshold": flag_threshold,
            "n_markers": report.n_markers,
            "n_resolved": report.n_resolved,
            "n_passing": report.n_passing,
            "n_flagged": report.n_flagged,
            "pass_rate": report.pass_rate,
            "elapsed_s": report.elapsed_s,
            "verdicts": [v.__dict__ for v in report.verdicts],
        }
        output.write_text(_json.dumps(serialised, indent=2))
        console.print(f"[dim]Full JSON report written to[/dim] {output}")

    # Non-zero exit so CI / scripts can gate export on clean verify
    raise typer.Exit(0 if report.n_flagged == 0 else 1)


# ── verify-citations (Phase 46.B — external citation verification) ──────────────

@app.command(name="verify-citations")
def verify_citations(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (first 8+ chars).")],
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Write the full JSON report to this file (default: print summary only).",
    ),
    show_records: bool = typer.Option(
        True, "--records/--no-records",
        help="Print the per-citation verdict table.",
    ),
):
    """
    Phase 46.B — external citation verification (AutoResearchClaw pattern).

    Walks the draft's sources and cross-checks each cited paper against
    external registries:

      1. arXiv ID    → Semantic Scholar arXiv endpoint (JSON).
      2. DOI         → Crossref canonical record.
      3. Title only  → OpenAlex search; top hit.

    Each cited source gets a verdict based on title-Jaccard to the
    authoritative record:

      VERIFIED      ≥ 0.80
      SUSPICIOUS    0.50 – 0.80
      HALLUCINATED  < 0.50 or no match at all
      SKIPPED       no identifier AND no title to query with

    Complementary to the in-corpus ``book review`` grounding pass —
    that checks whether claims match the retrieved evidence; this
    checks whether the cited sources are real published papers.

    Examples:

      sciknow book verify-citations 3f2a1b4c

      sciknow book verify-citations 3f2a1b4c -o verification.json
    """
    import json as _json
    from sciknow.core.citation_verify import (
        HALLUCINATED, SKIPPED, SUSPICIOUS, VERIFIED, verify_draft,
    )

    console.print(f"[bold]Verifying citations[/bold] on draft {draft_id}…")
    report = verify_draft(draft_id)
    if report.n_citations == 0:
        console.print("[yellow]No citations on this draft (nothing to verify).[/yellow]")
        raise typer.Exit(0)

    # Summary line with colored counts
    counts = [
        (f"[green]✓ verified {report.n_verified}[/green]", report.n_verified),
        (f"[yellow]? suspicious {report.n_suspicious}[/yellow]", report.n_suspicious),
        (f"[red]✗ hallucinated {report.n_hallucinated}[/red]", report.n_hallucinated),
        (f"[dim]- skipped {report.n_skipped}[/dim]", report.n_skipped),
    ]
    console.print(
        f"[bold]{report.n_citations}[/bold] citations: "
        + "  ".join(label for label, _ in counts)
    )

    if show_records:
        table = Table(title=f"Citation Verdicts — draft {report.draft_id[:8]}",
                      box=box.SIMPLE_HEAD, expand=True)
        table.add_column("[N]", justify="right", width=4)
        table.add_column("Verdict", width=14)
        table.add_column("Sim", justify="right", width=5)
        table.add_column("Source", width=10)
        table.add_column("Title",  ratio=3)
        table.add_column("DOI / arXiv", ratio=1, style="dim")
        table.add_column("Notes", ratio=1, style="dim", overflow="fold")

        for r in report.records:
            color = (
                "green"  if r.verdict == VERIFIED else
                "yellow" if r.verdict == SUSPICIOUS else
                "red"    if r.verdict == HALLUCINATED else
                "dim"
            )
            ident = r.doi or (f"arXiv:{r.arxiv_id}" if r.arxiv_id else "—")
            table.add_row(
                str(r.marker if r.marker is not None else "—"),
                f"[{color}]{r.verdict}[/{color}]",
                f"{r.similarity:.2f}" if r.similarity else "—",
                r.external_source or "—",
                (r.title or "")[:80],
                ident[:40],
                r.notes[:60],
            )
        console.print(table)

    if output is not None:
        output.write_text(_json.dumps(report.as_dict(), indent=2))
        console.print(f"[dim]Full report written to {output}[/dim]")


# ── ensemble-review (Phase 46.C — NeurIPS-rubric panel + meta-reviewer) ─────────

@app.command(name="ensemble-review")
def ensemble_review(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (first 8+ chars).")],
    n: int = typer.Option(3, "--n", "-n", min=1, max=9,
                           help="Number of independent reviewers (3 is NeurIPS default)."),
    temperature: float = typer.Option(0.75, "--temperature", "-T",
                                       help="Per-reviewer sampling temperature (0.75 NeurIPS convention)."),
    context_k: int = typer.Option(12, "--context-k", "-k",
                                   help="Passages retrieved for each reviewer to see."),
    model: str | None = typer.Option(None, "--model"),
    save: bool = typer.Option(True, "--save/--no-save",
                               help="Persist panel + meta to drafts.custom_metadata.ensemble_review."),
):
    """
    Phase 46.C — ensemble NeurIPS-rubric review with meta-reviewer fusion.

    Runs N independent reviewers over one draft section (default 3), each
    with temperature 0.75 and a rotating stance (neutral/pessimistic/
    optimistic — positivity-bias mitigation from AI-Scientist v1 §4).
    A meta-reviewer then fuses their numeric scores (median over the
    rubric) and unions their free-text lists, weighted by agreement.

    Rubric (NeurIPS 2024 form):
      soundness, presentation, contribution   (1–4)
      overall                                  (1–10)
      confidence                               (1–5)
      decision  ∈ strong_reject … strong_accept

    Complementary to `book review` (single-pass critic) and
    `book verify-citations` (external citation check). Ensemble review
    is higher-variance-reducing but more expensive (~N× the LLM cost).

    Examples:

      sciknow book ensemble-review 3f2a1b4c

      sciknow book ensemble-review 3f2a1b4c -n 5 -T 0.8

      sciknow book ensemble-review 3f2a1b4c --no-save
    """
    from sciknow.core.book_ops import ensemble_review_stream

    console.print()
    gen = ensemble_review_stream(
        draft_id, n_reviewers=n, temperature=temperature,
        context_k=context_k, model=model, save=save,
    )
    _consume_events(gen, console)


# ── ledger (Phase 54.6.76 — #15) ───────────────────────────────────────────

@app.command()
def ledger(
    book_title: Annotated[str, typer.Argument(help="Book title or id-prefix.")],
    chapter: int = typer.Option(
        0, "--chapter", "-c",
        help="Chapter number. Default 0 = book-level + per-chapter breakdown. "
             "Positive N = one chapter's section-level breakdown.",
    ),
    draft_id: str = typer.Option(
        "", "--draft", "-d",
        help="Draft id-prefix. Overrides --chapter; shows one draft's "
             "autowrite wall-time + tokens.",
    ),
):
    """Phase 54.6.76 (#15) — GPU-time + token ledger per draft, chapter,
    or book, sourced from the autowrite_runs telemetry table.

    Shows total wall-seconds (retrieval + scoring + verification + CoVe +
    idle) and total tokens. Helps spot a runaway CoVe loop or a
    chapter that's disproportionately expensive.

    Examples:

      sciknow book ledger "Global Cooling"                 # book + per-chapter
      sciknow book ledger "Global Cooling" --chapter 3     # sections in ch.3
      sciknow book ledger "Global Cooling" --draft 3f2a1b  # one draft
    """
    from sciknow.core import gpu_ledger
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)
        book_id = book[0]

        if draft_id:
            row = gpu_ledger.ledger_for_draft(session, draft_id)
            if not row or row.n_runs == 0:
                console.print(
                    f"[yellow]No autowrite runs found for draft "
                    f"{draft_id!r}.[/yellow]"
                )
                raise typer.Exit(0)
            _print_ledger_row(row)
            return

        if chapter > 0:
            from sqlalchemy import text as _t
            ch = session.execute(_t(
                "SELECT id::text FROM book_chapters "
                "WHERE book_id::text = :bid AND number = :n LIMIT 1"
            ), {"bid": book_id, "n": chapter}).fetchone()
            if not ch:
                console.print(f"[red]Chapter {chapter} not found in {book[1]!r}.[/red]")
                raise typer.Exit(1)
            header = gpu_ledger.ledger_for_chapter(session, ch[0])
            sections = gpu_ledger.ledger_per_section(session, ch[0])
            if header:
                _print_ledger_row(header, prefix="[bold]")
            if sections:
                console.print("  [dim]sections:[/dim]")
                for s in sections:
                    _print_ledger_row(s, indent="    ")
            return

        header = gpu_ledger.ledger_for_book(session, book_id)
        per_ch = gpu_ledger.ledger_per_chapter(session, book_id)
        if header:
            _print_ledger_row(header, prefix="[bold]")
        if per_ch:
            console.print("  [dim]chapters:[/dim]")
            for c in per_ch:
                _print_ledger_row(c, indent="    ")


def _print_ledger_row(row, indent: str = "", prefix: str = "") -> None:
    from sciknow.core.gpu_ledger import format_wall
    label = f"{prefix}{row.label}[/bold]" if prefix else row.label
    if row.n_runs == 0:
        console.print(f"{indent}{label}  [dim]— no autowrite runs yet —[/dim]")
        return
    tps = f"{row.tokens_per_second:.1f} tok/s" if row.wall_seconds > 0 else "—"
    console.print(
        f"{indent}{label}  "
        f"[cyan]{format_wall(row.wall_seconds)}[/cyan]  "
        f"[dim]{row.tokens:,} tok · {tps} · "
        f"{row.n_runs} run{'s' if row.n_runs != 1 else ''}[/dim]"
    )


# ── snapshot / snapshots / snapshot-restore (Phase 54.6.75 — #13) ──────────

@app.command()
def snapshot(
    book_title: Annotated[str, typer.Argument(
        help="Book title (or book-id prefix) to snapshot. Pass '-' "
             "with --draft to skip the book lookup (the draft id "
             "carries enough context).")],
    chapter: int = typer.Option(
        0, "--chapter", "-c",
        help="Chapter number to snapshot (default 0 = whole book).",
    ),
    draft: str | None = typer.Option(
        None, "--draft", "-d",
        help="Section-scope snapshot: snapshot just this single draft's "
             "current content. Accepts a full draft UUID or any prefix "
             "(min 6 chars). Produces a scope='draft' row that the web's "
             "/api/snapshot-content/<id> endpoint and `book snapshot-restore` "
             "(coming Phase 3) consume identically to legacy per-draft "
             "snapshots. Mutex with --chapter.",
    ),
    name: str = typer.Option("", "--name", help="Optional snapshot label."),
):
    """Snapshot a book / chapter / single section so you can roll back.

    Mirrors the web reader's snapshot buttons. Captures the latest
    draft per (chapter, section_type) into one ``draft_snapshots``
    row with the appropriate scope. Non-destructive — nothing is
    overwritten.

    Phase 54.6.328 — every snapshot now carries a diff brief in the
    ``meta`` column, computed at create time vs the prior snapshot
    in the same scope. Renders as a "Δ +1,247/-380w · 4¶" line on
    save and in `book snapshots` listings.

    Examples:

      sciknow book snapshot "Global Cooling"                    # whole book
      sciknow book snapshot "Global Cooling" --chapter 3        # one chapter
      sciknow book snapshot - --draft cd7cdee2 --name "pre-revise"
                                                                # one section
    """
    import datetime as _dt
    from sqlalchemy import text as _text
    from sciknow.storage.db import get_session
    from sciknow.core.snapshot_diff import (
        compute_bundle_brief, compute_prose_diff,
        render_brief_one_line,
    )
    from sciknow.web.app import (
        _snapshot_chapter_drafts, _snapshot_book_drafts,
    )
    from sciknow.web.routes.snapshots import _prev_bundle_content

    if draft is not None and chapter > 0:
        console.print("[red]--draft and --chapter are mutually exclusive.[/red]")
        raise typer.Exit(2)

    # Section-scope path — short-circuits the book lookup so callers
    # that only have a draft id (the GUI's editor toolbar) can snapshot
    # without naming the enclosing book.
    if draft is not None:
        if len(draft) < 6:
            console.print(
                "[red]--draft id too short[/red] (min 6 chars to disambiguate)."
            )
            raise typer.Exit(2)
        with get_session() as session:
            row = session.execute(_text("""
                SELECT id::text, title, content, word_count
                FROM drafts WHERE id::text LIKE :q LIMIT 2
            """), {"q": f"{draft}%"}).fetchall()
            if not row:
                console.print(f"[red]No draft matches {draft!r}.[/red]")
                raise typer.Exit(1)
            if len(row) > 1:
                console.print(
                    f"[red]Ambiguous draft prefix {draft!r}[/red] — "
                    f"matches {len(row)} rows."
                )
                raise typer.Exit(1)
            did, dtitle, dcontent, dwc = row[0]
            prev = session.execute(_text("""
                SELECT content FROM draft_snapshots
                WHERE draft_id::text = :did
                ORDER BY created_at DESC LIMIT 1
            """), {"did": did}).fetchone()
            prev_content = prev[0] if prev else ""
            meta = compute_prose_diff(prev_content, dcontent or "")
            snap_name = name.strip() or (
                f"{(dtitle or 'draft')[:40]} — "
                f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            session.execute(_text("""
                INSERT INTO draft_snapshots
                    (draft_id, scope, name, content, word_count, meta)
                VALUES
                    (CAST(:did AS uuid), 'draft', :name, :content, :wc,
                     CAST(:meta AS jsonb))
            """), {"did": did, "name": snap_name,
                   "content": dcontent or "", "wc": dwc or 0,
                   "meta": __import__("json").dumps(meta)})
            session.commit()
        console.print(
            f"[green]✓ Snapshot saved:[/green] {snap_name}\n"
            f"  scope=draft  draft={did[:8]}  words={dwc or 0:,}\n"
            f"  diff: {render_brief_one_line(meta)}"
        )
        return

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)
        book_id = book[0]

        if chapter > 0:
            ch = session.execute(_text("""
                SELECT id::text, title FROM book_chapters
                WHERE book_id::text = :bid AND number = :n LIMIT 1
            """), {"bid": book_id, "n": chapter}).fetchone()
            if not ch:
                console.print(f"[red]Chapter {chapter} not found in {book[1]!r}.[/red]")
                raise typer.Exit(1)
            bundle = _snapshot_chapter_drafts(session, ch[0])
            if not bundle["drafts"]:
                console.print(f"[yellow]Chapter has no drafts to snapshot.[/yellow]")
                raise typer.Exit(0)
            snap_name = name.strip() or (
                f"{book[1]} · Ch.{chapter} — "
                f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            total_words = sum(d.get("word_count") or 0 for d in bundle["drafts"])
            prev_bundle = _prev_bundle_content(
                session, scope="chapter", container_id=ch[0],
            )
            meta = compute_bundle_brief(bundle, prev_bundle)
            session.execute(_text("""
                INSERT INTO draft_snapshots
                    (chapter_id, scope, name, content, word_count, meta)
                VALUES
                    (CAST(:cid AS uuid), 'chapter', :name, :content, :wc,
                     CAST(:meta AS jsonb))
            """), {"cid": ch[0], "name": snap_name,
                   "content": _json.dumps(bundle), "wc": total_words,
                   "meta": _json.dumps(meta)})
            session.commit()
            console.print(
                f"[green]✓ Snapshot saved:[/green] {snap_name}\n"
                f"  scope=chapter  drafts={len(bundle['drafts'])}  "
                f"words={total_words:,}\n"
                f"  diff: {render_brief_one_line(meta)}"
            )
            return

        # Whole-book path — delegate to the shared helper which handles
        # bundle assembly + brief computation in one call.
        snap_id = _snapshot_book_drafts(session, book_id, name=name.strip())
        if not snap_id:
            console.print(f"[yellow]Book has no drafts to snapshot.[/yellow]")
            raise typer.Exit(0)
        session.commit()
        # Read back the persisted row so we can render the same brief
        # the helper computed.
        row = session.execute(_text(
            "SELECT name, word_count, meta FROM draft_snapshots "
            "WHERE id::text = :sid"
        ), {"sid": snap_id}).fetchone()
        if row:
            console.print(
                f"[green]✓ Snapshot saved:[/green] {row[0]}\n"
                f"  scope=book  words={row[1] or 0:,}\n"
                f"  diff: {render_brief_one_line(row[2] or {})}"
            )


@app.command()
def snapshots(
    book_title: Annotated[str, typer.Argument(help="Book title.")],
    chapter: int = typer.Option(
        0, "--chapter", "-c",
        help="Chapter number to list (default 0 = list book + every "
             "chapter's snapshots).",
    ),
):
    """Phase 54.6.75 (#13) — list saved snapshots for a book / chapter.

    Useful for finding a snapshot ID to pass to `snapshot-restore`.

    Examples:

      sciknow book snapshots "Global Cooling"                 # book + all chapters
      sciknow book snapshots "Global Cooling" --chapter 3     # one chapter
    """
    from sqlalchemy import text as _text
    from sciknow.storage.db import get_session
    from sciknow.core.snapshot_diff import render_brief_one_line

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)
        book_id = book[0]
        if chapter > 0:
            ch = session.execute(_text("""
                SELECT id::text, title FROM book_chapters
                WHERE book_id::text = :bid AND number = :n LIMIT 1
            """), {"bid": book_id, "n": chapter}).fetchone()
            if not ch:
                console.print(f"[red]Chapter {chapter} not found.[/red]")
                raise typer.Exit(1)
            rows = session.execute(_text("""
                SELECT id::text, name, word_count, created_at, scope, meta,
                       NULL AS ch_num
                FROM draft_snapshots
                WHERE chapter_id::text = :cid AND scope = 'chapter'
                ORDER BY created_at DESC
            """), {"cid": ch[0]}).fetchall()
        else:
            rows = session.execute(_text("""
                SELECT s.id::text, s.name, s.word_count, s.created_at, s.scope,
                       s.meta, bc.number
                FROM draft_snapshots s
                LEFT JOIN book_chapters bc ON bc.id = s.chapter_id
                WHERE s.book_id::text = :bid
                   OR s.chapter_id IN (
                       SELECT id FROM book_chapters WHERE book_id::text = :bid
                   )
                ORDER BY s.created_at DESC
            """), {"bid": book_id}).fetchall()
    if not rows:
        console.print("[dim]No snapshots yet.[/dim]")
        return
    console.print()
    for r in rows:
        sid, nm, wc, ts, scope = r[0], r[1], r[2] or 0, str(r[3] or ""), r[4]
        meta = r[5] if isinstance(r[5], dict) else {}
        ch_num = r[6] if len(r) > 6 and r[6] is not None else "—"
        scope_tag = (f"[dim]scope=book[/dim]" if scope == "book"
                     else f"[dim]scope=chapter Ch.{ch_num}[/dim]")
        brief = render_brief_one_line(meta)
        console.print(
            f"  [cyan]{sid[:8]}[/cyan]  {ts[:19]}  "
            f"{scope_tag}  words={wc:,}  {nm}\n"
            f"           [dim]Δ {brief}[/dim]"
        )


@app.command(name="snapshot-restore")
def snapshot_restore(
    snapshot_id: Annotated[str, typer.Argument(
        help="Snapshot ID (first 8+ chars from `sciknow book snapshots`).")],
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Report what would be restored; do NOT insert new drafts.",
    ),
):
    """Phase 54.6.75 (#13) — restore a snapshot by inserting NEW draft
    versions (non-destructive; existing drafts are kept).

    Each section in the snapshot gets a new `drafts` row at
    `version = max(current_version) + 1`, so the restored text
    becomes the new "latest" while the pre-restore drafts remain in
    version history. Undo = snapshot → restore an earlier snapshot.

    Only bundle-scope snapshots (chapter, book) are accepted here.
    Per-draft restore still runs through the web reader today.

    Examples:

      sciknow book snapshot-restore 3a9e1b2c
      sciknow book snapshot-restore 3a9e1b2c --dry-run
    """
    from sqlalchemy import text as _text
    from sciknow.storage.db import get_session
    from sciknow.web.app import _restore_chapter_bundle

    with get_session() as session:
        row = session.execute(_text(
            "SELECT id::text, scope, content, name FROM draft_snapshots "
            "WHERE id::text LIKE :q LIMIT 1"
        ), {"q": f"{snapshot_id}%"}).fetchone()
        if not row:
            console.print(f"[red]Snapshot not found:[/red] {snapshot_id}")
            raise typer.Exit(1)
        sid, scope, content, nm = row
        if scope not in ("chapter", "book"):
            console.print(
                f"[red]Snapshot scope is {scope!r}.[/red] Only chapter + "
                f"book bundles can be restored via this command."
            )
            raise typer.Exit(1)
        try:
            payload = _json.loads(content)
        except Exception as exc:
            console.print(f"[red]Malformed snapshot bundle:[/red] {exc}")
            raise typer.Exit(1)

        if dry_run:
            if scope == "chapter":
                n = len(payload.get("drafts") or [])
                console.print(
                    f"[dim][dry-run] would insert {n} new draft version(s) "
                    f"from '{nm}'.[/dim]"
                )
            else:
                n_ch = len(payload.get("chapters") or [])
                n_d = sum(len(c.get("drafts") or []) for c in
                          payload.get("chapters") or [])
                console.print(
                    f"[dim][dry-run] would restore {n_ch} chapter(s) / "
                    f"{n_d} draft(s) from '{nm}'.[/dim]"
                )
            raise typer.Exit(0)

        total = 0
        chapters_touched = 0
        if scope == "chapter":
            total = _restore_chapter_bundle(session, payload)
            chapters_touched = 1
        else:
            for ch_bundle in payload.get("chapters") or []:
                total += _restore_chapter_bundle(session, ch_bundle)
                chapters_touched += 1
        session.commit()

    console.print(
        f"[green]✓ Restored {total} draft(s) across "
        f"{chapters_touched} chapter(s) from snapshot '{nm}'.[/green]\n"
        f"  [dim]Existing drafts are preserved as prior versions — "
        f"use `book snapshots` to snapshot before trying another restore.[/dim]"
    )


# ── history / diff (Phase 54.6.328 snapshot-versioning Phase 3) ───────────

@app.command(name="history")
def history(
    target: Annotated[str, typer.Argument(
        help="What to walk: '<chapter>:<section>' for one section's "
             "version chain (e.g. '3:solar_dynamo'), or just a book "
             "title for a book-level rollup.")],
    book_title: str | None = typer.Option(
        None, "--book", "-b",
        help="Book title (only needed when target is just <chapter>:<section>; "
             "the active book is used otherwise).",
    ),
    chapter: int = typer.Option(
        0, "--chapter", "-c",
        help="With a book title target, scope the rollup to one chapter "
             "instead of the whole book.",
    ),
    limit: int = typer.Option(
        50, "--limit", "-n",
        help="Cap rows printed. Default 50; section history is rarely "
             "longer in practice.",
    ),
):
    """Walk a section / chapter / book version timeline with diff briefs.

    For a section target ('<ch-num>:<section-slug>'), prints every
    drafts.version row interleaved with any draft-scope snapshots,
    each with a one-line diff brief vs the immediately-prior entry.

    For a book or chapter target, rolls up to the latest active draft
    per section + the most recent chapter/book snapshots.

    Examples:

      sciknow book history 3:solar_dynamo_behavior_over_millennia
      sciknow book history "Global Cooling" --chapter 3
      sciknow book history "Global Cooling"
    """
    from sqlalchemy import text as _text
    from sciknow.storage.db import get_session
    from sciknow.core.snapshot_diff import render_brief_one_line
    from sciknow.core.version_history import list_section_history

    # Section view: target shape ch:slug
    if ":" in target and not target.endswith(":"):
        try:
            ch_part, slug = target.split(":", 1)
            ch_num = int(ch_part)
        except ValueError:
            console.print(
                "[red]Invalid section target.[/red] Use '<chapter>:<section-slug>'."
            )
            raise typer.Exit(2)
        with get_session() as session:
            ch = session.execute(_text("""
                SELECT bc.id::text, bc.title FROM book_chapters bc
                JOIN books b ON b.id = bc.book_id
                WHERE bc.number = :n
                ORDER BY b.created_at DESC LIMIT 1
            """), {"n": ch_num}).fetchone()
            if not ch:
                console.print(f"[red]No chapter with number {ch_num}.[/red]")
                raise typer.Exit(1)
            entries = list_section_history(
                session, chapter_id=ch[0], section_slug=slug,
            )
        if not entries:
            console.print(f"[dim]No history for {ch_num}:{slug}.[/dim]")
            return
        console.print()
        console.print(
            f"  [bold]Ch.{ch_num}[/bold] · {ch[1] or '?'} · "
            f"[cyan]{slug}[/cyan]  ({len(entries)} entries)"
        )
        console.print()
        for e in entries[:limit]:
            kind_tag = (
                "[bold]✓ active[/bold]" if e.is_active
                else f"[dim]{e.kind}[/dim]"
            )
            score = e.extra.get("final_overall")
            score_tag = (
                f" score={score:.2f}" if isinstance(score, (int, float))
                else ""
            )
            label = e.label
            if e.kind == "draft":
                label = f"v{e.version or 0}"
            ts = (e.created_at or "")[:19]
            console.print(
                f"  [cyan]{e.id[:8]}[/cyan]  {label:<10s}  {ts}  "
                f"{e.word_count:>6,d}w  {kind_tag}{score_tag}"
            )
            console.print(
                f"           [dim]Δ {render_brief_one_line(e.meta)}[/dim]"
            )
        return

    # Book / chapter rollup view.
    with get_session() as session:
        book = _get_book(session, target)
        if not book:
            console.print(f"[red]Book not found:[/red] {target}")
            raise typer.Exit(1)
        book_id = book[0]

        if chapter > 0:
            ch = session.execute(_text("""
                SELECT id::text, title FROM book_chapters
                WHERE book_id::text = :bid AND number = :n LIMIT 1
            """), {"bid": book_id, "n": chapter}).fetchone()
            if not ch:
                console.print(f"[red]Chapter {chapter} not found.[/red]")
                raise typer.Exit(1)
            ch_filter = "WHERE chapter_id::text = :cid"
            params = {"cid": ch[0]}
            scope_label = f"Ch.{chapter} {ch[1] or ''}"
        else:
            ch_filter = (
                "WHERE chapter_id IN ("
                "  SELECT id FROM book_chapters WHERE book_id::text = :bid"
                ")"
            )
            params = {"bid": book_id}
            scope_label = book[1]

        # Latest active draft per (chapter, section_type), with brief
        # = diff vs the prior version in that chain.
        sec_rows = session.execute(_text(f"""
            SELECT DISTINCT ON (chapter_id, section_type)
                id::text, chapter_id::text, section_type, version,
                word_count, content, created_at,
                COALESCE((custom_metadata->>'is_active')::boolean, FALSE) AS active
            FROM drafts
            {ch_filter}
            ORDER BY chapter_id, section_type,
                     COALESCE((custom_metadata->>'is_active')::boolean, FALSE) DESC,
                     CASE WHEN content IS NULL OR LENGTH(content) < 50
                          THEN 1 ELSE 0 END,
                     version DESC
        """), params).fetchall()

        # Bundle snapshots in scope.
        snap_q = (
            "SELECT id::text, name, word_count, created_at, scope, meta "
            "FROM draft_snapshots "
        )
        if chapter > 0:
            snap_q += "WHERE chapter_id::text = :cid AND scope='chapter' "
            snap_params = {"cid": ch[0]}
        else:
            snap_q += (
                "WHERE book_id::text = :bid OR chapter_id IN ("
                "  SELECT id FROM book_chapters WHERE book_id::text = :bid"
                ") "
            )
            snap_params = {"bid": book_id}
        snap_q += "ORDER BY created_at DESC LIMIT 20"
        snap_rows = session.execute(_text(snap_q), snap_params).fetchall()

    console.print()
    console.print(f"  [bold]{scope_label}[/bold]  ({len(sec_rows)} sections)")
    console.print()
    for r in sec_rows[:limit]:
        did, _cid, slug, ver, wc, _content, ts, active = r
        active_tag = "[bold]✓[/bold]" if active else " "
        console.print(
            f"  {active_tag}  [cyan]{did[:8]}[/cyan]  "
            f"v{ver or 0:<3d}  {(slug or '')[:50]:<50s}  "
            f"{wc or 0:>6,d}w  [dim]{str(ts or '')[:19]}[/dim]"
        )
    if snap_rows:
        console.print()
        console.print("  [dim]bundle snapshots[/dim]")
        for s in snap_rows:
            sid, nm, wc, ts, scope, meta = s
            meta = meta if isinstance(meta, dict) else {}
            console.print(
                f"     [cyan]{sid[:8]}[/cyan]  "
                f"[dim]scope={scope:<7s}[/dim]  "
                f"{wc or 0:>6,d}w  {(nm or '')[:50]}"
            )
            console.print(
                f"              [dim]Δ {render_brief_one_line(meta)} · "
                f"{str(ts or '')[:19]}[/dim]"
            )


@app.command(name="diff")
def diff(
    ref_a: Annotated[str, typer.Argument(
        help="First version ref (older). Accepts a draft/snapshot id "
             "or prefix (≥6 chars), or '<ch>:<slug>:vN' / "
             "'<ch>:<slug>:latest'.")],
    ref_b: Annotated[str, typer.Argument(
        help="Second version ref (newer). Same shapes as ref_a.")],
    json_out: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead "
        "of unified diff text.",
    ),
    context: int = typer.Option(
        3, "--context", "-U",
        help="Lines of context for the unified diff (default 3).",
    ),
):
    """Word-level diff between any two versions.

    Examples:

      sciknow book diff 3a9e1b2c d7c3   # two draft prefixes
      sciknow book diff 3:solar_dynamo:v6 3:solar_dynamo:latest
      sciknow book diff 3a9e1b2c 3:solar_dynamo:latest --json
    """
    import difflib as _difflib
    import json as _json
    from sqlalchemy import text as _text  # noqa: F401
    from sciknow.storage.db import get_session
    from sciknow.core.snapshot_diff import (
        compute_prose_diff, compute_outline_structural_diff,
        _outline_chapters,
    )
    from sciknow.core.version_history import resolve_version_ref

    with get_session() as session:
        a = resolve_version_ref(session, ref_a)
        b = resolve_version_ref(session, ref_b)
    if not a:
        console.print(f"[red]Couldn't resolve {ref_a!r}.[/red]")
        raise typer.Exit(1)
    if not b:
        console.print(f"[red]Couldn't resolve {ref_b!r}.[/red]")
        raise typer.Exit(1)

    brief = compute_prose_diff(a["content"], b["content"])

    # Phase 54.6.328 (snapshot-versioning Phase 6) — when both refs are
    # book-scope snapshots, parse out the chapter shape and add the
    # structural diff so the user sees outline-level change first.
    structural = None
    if a.get("kind") == "snapshot" and b.get("kind") == "snapshot":
        try:
            ab = _json.loads(a["content"]) if a["content"] else {}
            bb = _json.loads(b["content"]) if b["content"] else {}
            if isinstance(ab, dict) and "chapters" in ab \
                    and isinstance(bb, dict) and "chapters" in bb:
                structural = compute_outline_structural_diff(
                    _outline_chapters(ab.get("chapters") or []),
                    _outline_chapters(bb.get("chapters") or []),
                )
        except Exception:
            structural = None

    if json_out:
        out = {
            "a": {k: v for k, v in a.items() if k != "content"},
            "b": {k: v for k, v in b.items() if k != "content"},
            "brief": brief,
        }
        if structural is not None:
            out["structural"] = structural
        console.print(_json.dumps(out, indent=2))
        return

    console.print()
    console.print(f"  [dim]A:[/dim] {a['label']}  ({a['word_count']:,}w)")
    console.print(f"  [dim]B:[/dim] {b['label']}  ({b['word_count']:,}w)")
    console.print(
        f"  [dim]Δ +{brief['words_added']:,}/-{brief['words_removed']:,}w · "
        f"+{brief['paragraphs_added']}/-{brief['paragraphs_removed']}¶ · "
        f"+{brief['citations_added']}/-{brief['citations_removed']}cite[/dim]"
    )
    if structural is not None and any(
        structural.get(k) for k in (
            "added_chapters", "removed_chapters",
            "renamed_chapters", "section_changes",
        )
    ):
        console.print()
        console.print("  [bold]Outline changes[/bold]")
        for c in structural.get("added_chapters") or []:
            console.print(
                f"    [green]+ Ch.{c.get('number')}[/green]  "
                f"{c.get('title', '')}"
            )
        for c in structural.get("removed_chapters") or []:
            console.print(
                f"    [red]- Ch.{c.get('number')}[/red]  "
                f"{c.get('title', '')}"
            )
        for c in structural.get("renamed_chapters") or []:
            kind = c.get("kind", "renamed")
            if kind == "renumbered":
                console.print(
                    f"    [yellow]~ Ch.{c.get('from_number')} → "
                    f"Ch.{c.get('to_number')}[/yellow]  {c.get('title', '')}"
                )
            else:
                console.print(
                    f"    [yellow]~ Ch.{c.get('number')}[/yellow]  "
                    f"{c.get('from_title', '')!r} → "
                    f"{c.get('to_title', '')!r}"
                )
        for ch in structural.get("section_changes") or []:
            n = ch.get("chapter_number")
            t = ch.get("chapter_title", "")
            added = ch.get("added") or []
            removed = ch.get("removed") or []
            console.print(
                f"    [dim]Ch.{n}[/dim] {t!r}: "
                f"[green]+{len(added)}[/green]/[red]-{len(removed)}[/red] sections"
            )
            for s in added:
                console.print(f"        [green]+ {s}[/green]")
            for s in removed:
                console.print(f"        [red]- {s}[/red]")
    console.print()
    diff_text = "\n".join(_difflib.unified_diff(
        (a["content"] or "").splitlines(),
        (b["content"] or "").splitlines(),
        fromfile=a["label"],
        tofile=b["label"],
        n=context,
        lineterm="",
    ))
    if not diff_text:
        console.print("[dim]No textual differences.[/dim]")
        return
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            console.print(f"[bold]{line}[/bold]")
        elif line.startswith("+"):
            console.print(f"[green]{line}[/green]")
        elif line.startswith("-"):
            console.print(f"[red]{line}[/red]")
        elif line.startswith("@@"):
            console.print(f"[cyan]{line}[/cyan]")
        else:
            console.print(line)


# ── review ─────────────────────────────────────────────────────────────────────

@app.command()
def review(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (first 8+ chars).")],
    model: str | None = typer.Option(None, "--model"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save review feedback to the draft."),
):
    """
    Run a critic pass over a saved draft.

    Assesses groundedness, completeness, accuracy, coherence, and redundancy.
    Produces structured feedback with specific quotes and actionable suggestions.
    The feedback is saved to the draft's review_feedback field.

    Examples:

      sciknow book review 3f2a1b4c

      sciknow book review 3f2a1b4c --no-save
    """
    from sciknow.core.book_ops import review_draft_stream

    console.print()
    gen = review_draft_stream(draft_id, model=model, save=save)
    _consume_events(gen, console)
    if save:
        console.print("[dim]Use `sciknow book revise` to apply the feedback.[/dim]")


# ── Phase 54.6.14 — BMAD-inspired critic skills ───────────────────────────────

@app.command(name="adversarial-review")
def adversarial_review_cmd(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (first 8+ chars).")],
    model: str | None = typer.Option(None, "--model"),
):
    """Cynical critic pass — finds ≥10 concrete issues per draft.

    Complements `book review` (graded, 5-dim) with an exhaustive
    criticism pass. Output is a numbered markdown list; nothing is
    persisted (doesn't overwrite review_feedback).
    """
    from sciknow.core.book_ops import adversarial_review_stream
    console.print()
    gen = adversarial_review_stream(draft_id, model=model)
    _consume_events(gen, console)


@app.command(name="edge-cases")
def edge_cases_cmd(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (first 8+ chars).")],
    model: str | None = typer.Option(None, "--model"),
):
    """Exhaustive path enumeration — reports only unhandled cases.

    For every claim in the draft, walks scope boundaries, counter-cases,
    causal alternatives, quantitative limits, extrapolations, and
    missing controls. Returns structured findings with location +
    trigger + consequence + severity.
    """
    from sciknow.core.book_ops import edge_case_hunter_stream
    console.print()
    gen = edge_case_hunter_stream(draft_id, model=model)
    _consume_events(gen, console)


# ── verify-draft (Phase 54.6.83 — #8) ──────────────────────────────────

@app.command(name="verify-draft")
def verify_draft_cmd(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (first 8+ chars).")],
    allow_llm: bool = typer.Option(
        True, "--llm-atomize/--no-llm-atomize",
        help="Use LLM to atomize complex sentences the regex heuristic "
             "missed. Disable for purely-mechanical splits (faster, "
             "may miss mixed-truth sentences with subtle compound clauses).",
    ),
    model: str = typer.Option(
        None, "--model",
        help="LLM for atomization fallback. Default: LLM_FAST_MODEL.",
    ),
    limit: int = typer.Option(
        0, "--limit", "-n",
        help="Show at most N mixed-truth sentences in detail (0 = all).",
    ),
):
    """Phase 54.6.83 (#8) — offline claim-atomization + NLI verification.

    Splits each sentence in the draft into atomic sub-claims
    (heuristic first, LLM fallback for complex compound sentences),
    scores each sub-claim's NLI entailment against the draft's source
    chunks, and reports sentences where sub-claims split between
    supported and unsupported — the mixed-truth failure mode that the
    existing sentence-level verifier misses.

    Reads from the draft's persisted ``sources`` field (populated by
    `book write` / `autowrite`). Read-only — no database writes.

    Examples:

      sciknow book verify-draft 3f2a1b4c
      sciknow book verify-draft 3f2a1b4c --no-llm-atomize
      sciknow book verify-draft 3f2a1b4c -n 5
    """
    import json as _json
    from sqlalchemy import text
    from sciknow.core.claim_atomize import verify_draft
    from sciknow.storage.db import get_session

    with get_session() as session:
        row = session.execute(text("""
            SELECT id::text, title, content, sources
            FROM drafts WHERE id::text LIKE :q LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()
    if not row:
        console.print(f"[red]Draft not found:[/red] {draft_id}")
        raise typer.Exit(1)
    d_id, d_title, d_content, d_sources = row
    sources = (_json.loads(d_sources) if isinstance(d_sources, str)
               else (d_sources or []))
    if not sources:
        console.print(
            f"[red]Draft {d_id[:8]} has no stored sources.[/red] "
            f"verify-draft needs the retrieval context. Re-generate "
            f"the draft via `book write` / `autowrite` so sources "
            f"get persisted."
        )
        raise typer.Exit(2)
    if not d_content:
        console.print(f"[red]Draft {d_id[:8]} has empty content.[/red]")
        raise typer.Exit(2)

    console.print(
        f"Verifying [bold]{d_title}[/bold] "
        f"([dim]{d_id[:8]}[/dim], {len(sources)} sources)…"
    )
    result = verify_draft(
        d_content, sources,
        model=model, allow_llm_atomize=allow_llm,
    )
    console.print(f"  {result.summary()}")

    if result.mixed_truth_count == 0:
        console.print(
            "\n[green]No mixed-truth sentences — every atomized "
            "sub-claim is consistent with the sources.[/green]"
        )
        return

    console.print(f"\n[bold yellow]{result.mixed_truth_count} mixed-truth "
                  f"sentence(s) — single-NLI would have averaged these away:"
                  f"[/bold yellow]\n")
    shown = 0
    for sv in result.sentences:
        if not sv.mixed_truth:
            continue
        if limit > 0 and shown >= limit:
            break
        shown += 1
        console.print(f"[dim]Sentence:[/dim] {sv.sentence}")
        for sc in sv.sub_claims:
            color = "green" if sc.supported else "red"
            flag = "✓" if sc.supported else "✗"
            console.print(
                f"  [{color}]{flag} {sc.entailment:.3f}[/{color}]  {sc.text}"
            )
        console.print()


# ── align-citations (Phase 54.6.71) ────────────────────────────────────

@app.command(name="align-citations")
def align_citations_cmd(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (first 8+ chars).")],
    save: bool = typer.Option(True, "--save/--no-save",
        help="Write the remapped content back to the draft row."),
    low_threshold: float = typer.Option(0.5, "--low-threshold",
        help="Only remap when claimed chunk's entailment is below this."),
    win_margin: float = typer.Option(0.15, "--win-margin",
        help="Top chunk must beat claimed chunk's entailment by this much."),
):
    """Phase 54.6.71 — align citation markers with the chunks that best
    entail each sentence.

    For every [N] marker in the draft, score the sentence's entailment
    against every retrieval source chunk and (conservatively) remap N
    when the claimed source is wrong. Reuses the NLI cross-encoder that
    the quality bench already loads for faithfulness scoring.

    Requires the draft to have a non-empty ``sources`` field (populated
    automatically by `book write` / `book autowrite`). Drafts without
    stored sources error out — this post-pass has nothing to align against.

    Examples:

      sciknow book align-citations 3f2a1b4c

      sciknow book align-citations 3f2a1b4c --no-save   # dry-run

      sciknow book align-citations 3f2a1b4c --low-threshold 0.4 --win-margin 0.2
    """
    import json as _json
    from sqlalchemy import text
    from sciknow.core.citation_align import align_citations
    from sciknow.storage.db import get_session

    with get_session() as session:
        row = session.execute(text("""
            SELECT id::text, title, content, sources
            FROM drafts WHERE id::text LIKE :q LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()
    if not row:
        console.print(f"[red]Draft not found:[/red] {draft_id}")
        raise typer.Exit(1)
    d_id, d_title, d_content, d_sources = row
    sources = (_json.loads(d_sources) if isinstance(d_sources, str)
               else (d_sources or []))
    if not sources:
        console.print(
            f"[red]Draft {d_id[:8]} has no stored sources.[/red] "
            f"This alignment pass needs the retrieval context the writer "
            f"used. Re-generate the draft via `book write` / `autowrite` "
            f"so sources get persisted, then retry."
        )
        raise typer.Exit(2)
    if not d_content:
        console.print(f"[red]Draft {d_id[:8]} has empty content.[/red]")
        raise typer.Exit(2)

    console.print(
        f"Aligning citations for [bold]{d_title}[/bold] "
        f"([dim]{d_id[:8]}[/dim], {len(sources)} sources)…"
    )
    result = align_citations(d_content, sources,
                             low_threshold=low_threshold,
                             win_margin=win_margin)
    console.print(f"  {result.summary()}")
    if result.n_remapped == 0:
        console.print("[green]No remaps needed — citations align with sources.[/green]")
        raise typer.Exit(0)

    # Show each remap for review
    for ev in result.remaps:
        console.print(
            f"  [yellow]·[/yellow] [dim]{ev.sentence_preview}[/dim]"
        )
        console.print(
            f"    [{ev.claimed_n}] ({ev.claimed_score:.3f})  →  "
            f"[{ev.new_n}] ({ev.new_score:.3f})"
        )

    if save:
        with get_session() as session:
            session.execute(text(
                "UPDATE drafts SET content = :c, "
                "word_count = :wc, updated_at = now() WHERE id::text = :did"
            ), {"c": result.new_text,
                "wc": len(result.new_text.split()),
                "did": d_id})
            session.commit()
        console.print(f"\n[green]✓ Saved {result.n_remapped} remaps to the draft.[/green]")
    else:
        console.print(f"\n[dim]--no-save: draft not modified.[/dim]")


# ── revise ─────────────────────────────────────────────────────────────────────

@app.command()
def revise(
    draft_id: Annotated[str, typer.Argument(help="Draft ID (first 8+ chars).")],
    instruction: str = typer.Option("", "--instruction", "-i",
                                     help="Revision instruction (or leave empty to apply saved review feedback)."),
    context_k: int = typer.Option(8, "--context-k", "-k", help="Additional passages to retrieve."),
    model: str | None = typer.Option(None, "--model"),
):
    """
    Revise a draft based on instructions or saved review feedback.

    Creates a new version (version N+1) linked to the original via
    parent_draft_id. The original is preserved. If no --instruction is
    given, uses the draft's saved review_feedback from `book review`.

    Examples:

      sciknow book revise 3f2a1b4c -i "expand the section on solar cycles"

      sciknow book revise 3f2a1b4c -i "add counterarguments from the skeptic literature"

      sciknow book revise 3f2a1b4c       # uses saved review feedback
    """
    from sciknow.core.book_ops import revise_draft_stream

    console.print()
    gen = revise_draft_stream(
        draft_id, instruction=instruction, context_k=context_k, model=model,
    )
    _consume_events(gen, console)


# ── argue ──────────────────────────────────────────────────────────────────────

@app.command()
def argue(
    claim:      Annotated[str, typer.Argument(help="The claim to map arguments for.")],
    context_k:  int = typer.Option(15, "--context-k", "-k"),
    candidates: int = typer.Option(60, "--candidates"),
    year_from:  int | None = typer.Option(None, "--year-from"),
    year_to:    int | None = typer.Option(None, "--year-to"),
    book_title: str | None = typer.Option(None, "--book", "-b",
                                           help="Scope to a book's topic clusters."),
    model:      str | None = typer.Option(None, "--model"),
    save:       bool = typer.Option(False, "--save", help="Save the argument map as a draft."),
):
    """
    Map the evidence for and against a claim from the literature.

    Retrieves the most relevant passages and classifies them as
    SUPPORTS / CONTRADICTS / NEUTRAL, then writes a structured argument map.

    Examples:

      sciknow book argue "solar activity is the primary driver of 20th century warming"

      sciknow book argue "cosmic rays modulate cloud cover" --book "Global Cooling" --save
    """
    from sciknow.core.book_ops import run_argue_stream
    from sciknow.storage.db import get_session

    bid = None
    if book_title:
        with get_session() as session:
            book = _get_book(session, book_title)
            if book:
                bid = book[0]

    console.print()
    gen = run_argue_stream(
        claim, book_id=bid, context_k=context_k, candidate_k=candidates,
        year_from=year_from, year_to=year_to, model=model, save=save,
    )
    _consume_events(gen, console)


# ── gaps ───────────────────────────────────────────────────────────────────────

@app.command()
def gaps(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    model: str | None = typer.Option(None, "--model"),
    save: bool = typer.Option(True, "--save/--no-save",
                               help="Persist identified gaps to the book_gaps table (default: yes)."),
):
    """
    Identify + persist gaps in the book: missing topics, weak chapters, unwritten sections.

    Runs two passes: a human-readable narrative (streamed), and a structured JSON
    extraction that saves each gap to the book_gaps table for tracking. View saved
    gaps in `book show`.

    Examples:

      sciknow book gaps "Global Cooling"

      sciknow book gaps "Global Cooling" --no-save   # informational only
    """
    from sciknow.core.book_ops import run_gaps_stream
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)
        book_id = book[0]

    console.print()
    gen = run_gaps_stream(book_id=book_id, model=model, save=save)
    _consume_events(gen, console)


@app.command(name="auto-expand")
def auto_expand(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    per_gap_limit: int = typer.Option(100, "--per-gap-limit",
        help="Max OpenAlex results per gap. Lower = faster + noisier."),
    relevance_threshold: float = typer.Option(0.55, "--relevance-threshold",
        help="Drop candidates below this cosine similarity to corpus centroid."),
    download_dir: Path = typer.Option(None, "--download-dir", "-d"),
    workers: int = typer.Option(0, "--workers", "-w"),
    ingest: bool = typer.Option(True, "--ingest/--no-ingest"),
    dry_run: bool = typer.Option(False, "--dry-run",
        help="Show the ranked candidate list + exit (no download)."),
):
    """Phase 54.6.5 — Auto-expand from this book's open gaps.

    For every open ``BookGap`` of type 'topic' or 'evidence' on the
    book, run ``db expand-topic`` with the gap's description as the
    query; merge across gaps (each candidate remembers which gaps it
    addresses); score the merged list against the corpus centroid;
    filter, download + ingest.

    Composition of ``book gaps`` + ``db expand-topic`` so you don't
    have to manually transcribe gap descriptions into search queries.

    Examples:

      sciknow book auto-expand "Global Cooling"

      sciknow book auto-expand "Global Cooling" --relevance-threshold 0.65 --dry-run
    """
    from sciknow.cli import preflight
    preflight()
    from sciknow.core.expand_ops import find_candidates_for_book_gaps
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)
        book_id = book[0]

    console.print(f"\n[bold]Auto-expand for book[/bold] {book_title!r}")
    console.print("[dim]Querying OpenAlex for each open topic/evidence gap…[/dim]\n")
    result = find_candidates_for_book_gaps(
        book_id, per_gap_limit=per_gap_limit, score_relevance=True,
    )
    cands = result["candidates"]
    info = result["info"]
    console.print(
        f"Processed {info.get('gaps_processed', 0)} of "
        f"{info.get('gaps_total', 0)} open gap(s). "
        f"Raw {info.get('raw', 0)} → merged {info.get('merged', 0)}, "
        f"dedup'd {info.get('dedup_dropped', 0)}."
    )
    if not cands:
        console.print("[yellow]Nothing to download.[/yellow]")
        raise typer.Exit(0)

    if relevance_threshold > 0:
        before = len(cands)
        cands = [c for c in cands
                 if (c.get("relevance_score") or 0.0) >= relevance_threshold]
        console.print(f"Relevance ≥ {relevance_threshold:.2f} kept {len(cands)} / {before}.")
    if not cands:
        console.print("[yellow]Nothing above the threshold.[/yellow]")
        raise typer.Exit(0)

    # Show gap-coverage preview — papers that close multiple gaps are
    # prioritised in the helper's sort, so the top of the list should
    # have the most-valuable items.
    console.print("\n[bold]Top candidates by gap-coverage × relevance:[/bold]")
    for c in cands[:10]:
        n_gaps = len(c.get("gap_ids") or [])
        year_str = f"({c.get('year')})" if c.get("year") else "(n.d.)"
        score = c.get("relevance_score")
        sc = f"  s={score:.2f}" if score is not None else ""
        console.print(
            f"  [cyan]{n_gaps}g[/cyan] {(c.get('doi') or '?')[:38]:<38} "
            f"{year_str}{sc} {(c.get('title') or '')[:60]}"
        )

    # Delegate to the shared download+ingest tail (lives in db.py).
    from sciknow.cli.db import _expand_common_download_and_ingest
    _expand_common_download_and_ingest(
        cands, download_dir=download_dir, workers=workers,
        ingest=ingest, dry_run=dry_run,
        source_method="auto-expand", source_query=book_title,
    )


# ── serve (web reader) ─────────────────────────────────────────────────────────


def _kill_stale_serve(host: str, port: int, *, wait_s: float = 5.0) -> None:
    """Phase 54.6.x — preflight: if another sciknow book-serve already
    holds the bind port, SIGTERM it cleanly so this invocation can
    proceed without the cryptic "[Errno 98] address already in use"
    crash + manual `lsof + kill` ritual.

    Safety rails:
      1. Only kills processes whose cmdline contains both `sciknow`
         and `book serve` (or `serve`) — never random listeners.
      2. Skips our own PID.
      3. SIGTERM first, escalate to SIGKILL only after `wait_s`
         seconds.
      4. Polls socket bind to confirm the port actually freed before
         returning; raises typer.Exit if it didn't.

    No-op when the port is free.
    """
    import socket as _socket
    import signal as _signal
    import time as _time
    import os as _os

    # Probe: try to bind. If it fails with EADDRINUSE, find the
    # holder. Otherwise return immediately.
    try:
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            s.bind((host, int(port)))
        return  # port is free
    except OSError:
        pass

    # Port is busy. Identify the holder via /proc (no lsof dep).
    holder_pid: int | None = None
    try:
        # /proc/net/tcp shows local_address as hex IP:PORT.
        with open("/proc/net/tcp", "r") as fh:
            for line in fh.readlines()[1:]:
                parts = line.split()
                if len(parts) < 4:
                    continue
                local = parts[1]  # "0100007F:223D" → 127.0.0.1:8765
                state = parts[3]  # "0A" = LISTEN
                if state != "0A":
                    continue
                try:
                    _hex_ip, hex_port = local.split(":")
                    if int(hex_port, 16) != int(port):
                        continue
                except Exception:  # noqa: BLE001
                    continue
                inode = parts[9]
                # Walk /proc/*/fd to find the PID owning this socket.
                for entry in _os.listdir("/proc"):
                    if not entry.isdigit():
                        continue
                    fd_dir = f"/proc/{entry}/fd"
                    try:
                        for fd in _os.listdir(fd_dir):
                            try:
                                tgt = _os.readlink(f"{fd_dir}/{fd}")
                                if tgt == f"socket:[{inode}]":
                                    holder_pid = int(entry)
                                    break
                            except OSError:
                                continue
                    except (PermissionError, FileNotFoundError):
                        continue
                    if holder_pid is not None:
                        break
                if holder_pid is not None:
                    break
    except FileNotFoundError:
        # Non-Linux fallback: no /proc. Refuse rather than killing
        # blind — the user can clean up manually.
        console.print(
            f"[yellow]Port {port} is in use, but I can't identify the "
            f"holder on this OS. Stop it manually and retry.[/yellow]"
        )
        raise typer.Exit(1)

    if holder_pid is None or holder_pid == _os.getpid():
        console.print(
            f"[yellow]Port {port} is in use but no PID found via /proc. "
            f"Stop the process holding it and retry.[/yellow]"
        )
        raise typer.Exit(1)

    # Confirm cmdline looks like sciknow.
    cmdline = ""
    try:
        with open(f"/proc/{holder_pid}/cmdline", "rb") as fh:
            cmdline = fh.read().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            )
    except Exception:  # noqa: BLE001
        cmdline = ""
    looks_like_sciknow = (
        "sciknow" in cmdline and ("book serve" in cmdline or " serve " in cmdline)
    )
    if not looks_like_sciknow:
        console.print(
            f"[red]Port {port} is held by PID {holder_pid} "
            f"(not a sciknow process):[/red] {cmdline.strip()[:180]}\n"
            f"  Refusing to kill — stop it manually and retry."
        )
        raise typer.Exit(1)

    console.print(
        f"[dim]Port {port} is held by stale sciknow process "
        f"PID {holder_pid}; sending SIGTERM…[/dim]"
    )
    try:
        _os.kill(holder_pid, _signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        console.print(
            f"[red]Cannot SIGTERM PID {holder_pid} (permission denied). "
            f"Stop it manually and retry.[/red]"
        )
        raise typer.Exit(1)

    # Wait for the port to free.
    deadline = _time.monotonic() + wait_s
    while _time.monotonic() < deadline:
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
                s.bind((host, int(port)))
            console.print(f"[green]Port {port} freed; starting server.[/green]")
            return
        except OSError:
            _time.sleep(0.2)

    # Last resort: SIGKILL.
    console.print(
        f"[yellow]PID {holder_pid} did not exit after {wait_s}s; "
        f"sending SIGKILL.[/yellow]"
    )
    try:
        _os.kill(holder_pid, _signal.SIGKILL)
    except ProcessLookupError:
        pass

    deadline = _time.monotonic() + 3.0
    while _time.monotonic() < deadline:
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
                s.bind((host, int(port)))
            return
        except OSError:
            _time.sleep(0.1)

    console.print(
        f"[red]Could not free port {port} after SIGKILL. "
        f"Please investigate manually.[/red]"
    )
    raise typer.Exit(1)


@app.command()
def serve(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    port: int = typer.Option(8765, "--port", "-p", help="HTTP port."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
):
    """
    Launch a local web reader for the book.

    Opens a browser-based reading experience with sidebar navigation,
    inline editing, comments, citation links, quality scores, and
    version history. Content is served live from the database.

    Examples:

      sciknow book serve "Global Cooling"

      sciknow book serve "Global Cooling" --port 9000
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

    from sciknow.web.app import app as web_app, set_book
    set_book(book[0], book[1])

    # Phase 43g — surface which project is being served so the user
    # knows which DB / data dir / collections are in play.
    from sciknow.core.project import get_active_project
    active = get_active_project()
    project_label = (
        f"[dim]project: default / legacy[/dim]"
        if active.is_default
        else f"[dim]project: {active.slug}[/dim]"
    )

    # Phase 54.6.x — pre-flight port check. If another sciknow book
    # serve is already bound to this port, SIGTERM it before binding
    # ourselves. The "address already in use" uvicorn error otherwise
    # exits with no useful guidance and the user has to hunt the PID
    # by hand. We only kill processes that look like sciknow's own
    # `sciknow book serve` — never anything else (Brave, postgres,
    # etc.). Confirmation via the cmdline match.
    _kill_stale_serve(host, port)

    console.print(f"\n[bold]SciKnow Book Reader[/bold]: {book[1]}")
    console.print(f"  {project_label}")
    console.print(f"  [link=http://{host}:{port}]http://{host}:{port}[/link]")
    console.print(f"  [dim]Press Ctrl+C to stop.[/dim]\n")

    # Phase 54.6.308 — pipe uvicorn into sciknow.log so every GUI
    # HTTP hit is captured alongside CLI invocations, LLM summaries,
    # and error tracebacks.  ``log_config=None`` is the key bit:
    # without it, uvicorn.run() applies its built-in LOGGING_CONFIG
    # at startup and blows away any handlers we attach beforehand.
    # With log_config=None uvicorn trusts whatever logging is already
    # wired, so our rotating-file handler on uvicorn.* loggers sticks
    # and every HTTP request lands in <project>/data/sciknow.log.
    from sciknow.logging_config import attach_uvicorn_to_sciknow_log
    attach_uvicorn_to_sciknow_log()
    import uvicorn
    uvicorn.run(web_app, host=host, port=port, log_level="info",
                log_config=None, access_log=True)


# ── autowrite (Karpathy-loop-inspired convergence) ─────────────────────────────

_DEFAULT_SECTIONS = ["overview", "key_evidence", "current_understanding", "open_questions", "summary"]


def _get_chapter_sections(session, chapter_id: str) -> list[str]:
    """Get per-chapter sections from DB, or fall back to defaults."""
    from sqlalchemy import text
    row = session.execute(text(
        "SELECT sections FROM book_chapters WHERE id::text = :cid"
    ), {"cid": chapter_id}).fetchone()
    if row and row[0] and isinstance(row[0], list) and len(row[0]) > 0:
        # Normalize section names to lowercase slugs
        return [s.lower().replace(" ", "_") for s in row[0]]
    return _DEFAULT_SECTIONS


@app.command()
def autowrite(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    chapter:    str = typer.Argument(None, help="Chapter number or title fragment (omit with --full for all chapters)."),
    section:    str = typer.Option("introduction", "--section", "-s",
                                    help="Section type (or 'all' for all sections)."),
    max_iter:   int = typer.Option(3, "--max-iter", "-n", help="Max review-revise iterations per section."),
    target_score: float = typer.Option(0.85, "--target-score", "-t",
                                        help="Overall quality score to stop iterating (0.0-1.0)."),
    model:      str | None = typer.Option(None, "--model"),
    auto_expand: bool = typer.Option(False, "--auto-expand",
                                      help="Auto-run db expand when reviewer identifies missing evidence."),
    full:       bool = typer.Option(False, "--full",
                                     help="Write ALL chapters × ALL sections (the full autonomous pipeline)."),
    rebuild:    bool = typer.Option(False, "--rebuild",
                                     help="Rewrite sections that already have drafts (default: skip existing)."),
    resume:     bool = typer.Option(
        False, "--resume",
        help=(
            "Phase 28 — for sections that already have a finished "
            "draft, load it as the starting content and run more "
            "iterations on it instead of skipping. Refuses to resume "
            "from drafts in a partial state (writing_in_progress, "
            "iteration_*_revising, placeholder) — use --rebuild to "
            "overwrite those. Mutually exclusive with --rebuild "
            "(rebuild wins if both are set)."
        ),
    ),
    target_words: int | None = typer.Option(
        None, "--target-words",
        help=(
            "Target words per section for this run. Overrides the "
            "book-level target_chapter_words. Default: chapter target "
            "/ number of sections in the chapter. Length becomes a "
            "scoring dimension in the autowrite loop."
        ),
    ),
    include_visuals: bool = typer.Option(
        False, "--include-visuals",
        help=(
            "Phase 54.6.142 — run the 5-signal visuals ranker on each "
            "section's topic and offer the writer a shortlist of "
            "figures/tables it may cite as `[Fig. N]`. Adds a "
            "`visual_citation` scoring dimension (mechanically computed: "
            "0.0 on any hallucinated marker, 1.0 on correct use, 0.5 "
            "when good candidates were available but none were cited). "
            "Level-1 + Level-2 verify run per iteration. Level-3 VLM "
            "claim-depiction verification is deferred to a future "
            "`book finalize-draft` pre-export pass. Default off — "
            "existing workflows are untouched."
        ),
    ),
    flexible_length: bool | None = typer.Option(
        None, "--flexible-length/--no-flexible-length",
        help=(
            "Phase 54.6.x — set the per-chapter flexible_length flag "
            "before running. When ON, autowrite is permitted to extend "
            "up to 2× the section's target_words IF the retrieval pool "
            "is rich enough (≥24 chunks). Persists to "
            "book_chapters.flexible_length so future runs inherit it. "
            "With --full this applies to every chapter; with a single "
            "chapter argument, only that chapter is toggled. Omit to "
            "leave per-chapter flags untouched."
        ),
    ),
):
    """
    Autonomous write → review → revise convergence loop (inspired by Karpathy's autoresearch).

    For each section, generates an initial draft, scores it on 5 quality dimensions
    (groundedness, completeness, coherence, citation accuracy, overall), then
    iteratively revises targeting the weakest dimension until the overall score
    reaches --target-score or --max-iter iterations are exhausted.

    Each iteration:
      1. Score the current draft (structured JSON reviewer)
      2. If score >= target → STOP (converged)
      3. Identify the weakest dimension
      4. Generate a targeted revision instruction
      5. Revise the draft
      6. Re-score → if improved KEEP, if regressed DISCARD

    Modes:
      Single section:    sciknow book autowrite "Book" 3 --section methods
      All sections:      sciknow book autowrite "Book" 3 --section all
      Full book:         sciknow book autowrite "Book" --full

    Examples:

      sciknow book autowrite "Global Cooling" 1 --section introduction --max-iter 5

      sciknow book autowrite "Global Cooling" 3 --section all --target-score 0.80

      sciknow book autowrite "Global Cooling" --full --max-iter 3 --auto-expand
    """
    from sqlalchemy import text
    from sciknow.core.book_ops import autowrite_section_stream
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

        book_id, b_title, b_desc, b_status, b_created, b_plan, _b_type = book

        if not b_plan:
            console.print(
                "[yellow]No book plan set.[/yellow] Generate one first:\n"
                f"  sciknow book plan {book_title!r}"
            )
            raise typer.Exit(1)

        chapters = session.execute(text("""
            SELECT id::text, number, title, description, topic_query, topic_cluster
            FROM book_chapters WHERE book_id = :bid ORDER BY number
        """), {"bid": book_id}).fetchall()

    if not chapters:
        console.print("[yellow]No chapters defined.[/yellow] Run `book outline` first.")
        raise typer.Exit(1)

    # Determine which chapters x sections to write
    if full:
        targets = []
        with get_session() as session:
            for ch in chapters:
                ch_secs = _get_chapter_sections(session, ch[0])
                for sec in ch_secs:
                    targets.append((ch[0], ch[1], ch[2], sec))
        console.print(
            f"[bold]Autowrite FULL BOOK:[/bold] {b_title}\n"
            f"  {len(chapters)} chapters, {len(targets)} total sections\n"
            f"  Max {max_iter} iterations each, target score {target_score}"
        )
    elif chapter is None:
        console.print("[red]Specify a chapter number or use --full for the entire book.[/red]")
        raise typer.Exit(1)
    else:
        with get_session() as session:
            ch = _get_chapter(session, book_id, chapter)
            if not ch:
                console.print(f"[red]Chapter not found:[/red] {chapter}")
                raise typer.Exit(1)
            ch_id, ch_num, ch_title = ch[0], ch[1], ch[2]

            if section == "all":
                ch_secs = _get_chapter_sections(session, ch_id)
                targets = [(ch_id, ch_num, ch_title, sec) for sec in ch_secs]
            else:
                targets = [(ch_id, ch_num, ch_title, section)]

        console.print(
            f"[bold]Autowrite:[/bold] {b_title} — "
            f"{len(targets)} section(s), max {max_iter} iter, target {target_score}"
        )

    # Phase 54.6.x — apply the --flexible-length toggle before running.
    # The flag is persisted on book_chapters.flexible_length so the
    # autowrite engine's _get_chapter_flexible_length() picks it up
    # via the same code path used by the GUI checkbox. Scope:
    #   - --full: every chapter in the book
    #   - single chapter argument: only that chapter
    if flexible_length is not None:
        with get_session() as session:
            if full:
                n = session.execute(text("""
                    UPDATE book_chapters SET flexible_length = :flex
                    WHERE book_id::text = :bid
                """), {"flex": bool(flexible_length), "bid": book_id}).rowcount
            else:
                # Targets list contains one or more (ch_id, ch_num, …)
                # tuples for the resolved chapter; we only need to flip
                # that single chapter id.
                target_ch_ids = sorted({t[0] for t in targets})
                n = 0
                for cid in target_ch_ids:
                    n += session.execute(text("""
                        UPDATE book_chapters SET flexible_length = :flex
                        WHERE id::text = :cid
                    """), {"flex": bool(flexible_length), "cid": cid}).rowcount
            session.commit()
        state = "ON" if flexible_length else "OFF"
        console.print(
            f"[dim]flexible_length={state} on {n} chapter(s) "
            f"(persisted to book_chapters.flexible_length).[/dim]"
        )

    # Phase 28 — rebuild and resume are mutually exclusive
    if rebuild and resume:
        console.print("[yellow]Both --rebuild and --resume given; --rebuild wins.[/yellow]")
        resume = False

    # Build a slug → latest draft id map. Used by:
    #   - default mode: skip sections that already have a draft
    #   - resume mode: pass the draft id to autowrite_section_stream
    #     so it loads the existing content as the iteration starting point
    existing_by_key: dict[tuple[str, str], dict] = {}
    if not rebuild:
        with get_session() as session:
            existing_rows = session.execute(text("""
                SELECT DISTINCT ON (chapter_id, section_type)
                    chapter_id::text, section_type, id::text,
                    custom_metadata, word_count
                FROM drafts
                WHERE book_id = :bid AND chapter_id IS NOT NULL
                ORDER BY chapter_id, section_type, version DESC, created_at DESC
            """), {"bid": book_id}).fetchall()
        for r in existing_rows:
            existing_by_key[(r[0], r[1])] = {
                "draft_id": r[2],
                "custom_metadata": r[3],
                "word_count": r[4] or 0,
            }

        if not resume:
            # Default: skip existing drafts
            before = len(targets)
            targets = [(cid, cn, ct, sec) for cid, cn, ct, sec in targets
                        if (cid, sec) not in existing_by_key]
            skipped = before - len(targets)
            if skipped:
                console.print(f"[dim]Skipping {skipped} sections with existing drafts (use --rebuild to overwrite or --resume to iterate further)[/dim]")
            if not targets:
                console.print("[green]All sections already have drafts.[/green]")
                raise typer.Exit(0)

    # Run the convergence loop for each target with live dashboard
    import time as _time
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from sciknow.core.book_ops import _is_resumable_draft

    total = len(targets)
    converged = 0

    for i, (ch_id, ch_num, ch_title, sec) in enumerate(targets, 1):
        console.print(f"\n{'=' * 72}")
        # Phase 28 — resolve the resume_draft_id for this section.
        # In resume mode, refuse partial states with a clear message
        # and skip to the next section instead of clobbering them.
        resume_draft_id: str | None = None
        if resume:
            existing = existing_by_key.get((ch_id, sec))
            if existing:
                ok, reason = _is_resumable_draft(
                    existing["custom_metadata"], existing["word_count"],
                )
                if not ok:
                    console.print(
                        f"[bold]Section {i}/{total}:[/bold] Ch.{ch_num} {ch_title} — {sec}\n"
                        f"  [yellow]Skipping resume: {reason}[/yellow]"
                    )
                    console.print(f"{'=' * 72}")
                    continue
                resume_draft_id = existing["draft_id"]

        mode_label = " [dim](resume)[/dim]" if resume_draft_id else ""
        console.print(
            f"[bold]Section {i}/{total}:[/bold] Ch.{ch_num} {ch_title} — {sec}{mode_label}"
            f"  [dim]({converged} converged so far)[/dim]"
        )
        console.print(f"{'=' * 72}")

        gen = autowrite_section_stream(
            book_id=book_id, chapter_id=ch_id, section_type=sec,
            model=model, max_iter=max_iter, target_score=target_score,
            auto_expand=auto_expand,
            target_words=target_words,
            resume_from_draft_id=resume_draft_id,
            include_visuals=include_visuals,
        )

        # Live dashboard state
        tok_count = 0
        tok_t0 = _time.monotonic()
        text_preview = ""
        current_stage = "starting"
        current_iter = 0
        iter_max = max_iter
        last_scores = {}
        score_history = []
        verdicts = []
        grounding = ""
        result = None

        def _build_display():
            tbl = Table.grid(padding=(0, 2))
            tbl.add_column(ratio=1)

            # Stage + tokens
            elapsed = _time.monotonic() - tok_t0
            tps = tok_count / elapsed if elapsed > 0 else 0
            stage_color = {"writing": "green", "scoring": "yellow", "verifying": "cyan",
                           "revising": "magenta", "saving": "blue"}.get(current_stage, "dim")
            tbl.add_row(Text.from_markup(
                f"  [{stage_color}]{current_stage.upper()}[/{stage_color}]  "
                f"[green]{tok_count} tok[/green]  "
                f"[cyan]{tps:.1f} tok/s[/cyan]  "
                f"[dim]iter {current_iter}/{iter_max}[/dim]"
            ))

            # Scores bar
            if last_scores:
                dims = ["groundedness", "completeness", "coherence", "citation_accuracy", "overall"]
                parts = []
                for d in dims:
                    v = last_scores.get(d, 0)
                    c = "green" if v >= 0.85 else "yellow" if v >= 0.7 else "red"
                    bar_len = int(v * 10)
                    bar = "█" * bar_len + "░" * (10 - bar_len)
                    parts.append(f"[{c}]{d[:5]} {bar} {v:.2f}[/{c}]")
                tbl.add_row(Text.from_markup("  " + "  ".join(parts)))

            # Verdicts
            if verdicts:
                v_text = "  ".join(verdicts[-5:])
                tbl.add_row(Text.from_markup(f"  {v_text}"))

            # Grounding
            if grounding:
                tbl.add_row(Text.from_markup(f"  {grounding}"))

            # Text preview (last 200 chars)
            if text_preview:
                preview = text_preview[-200:].replace("\n", " ").strip()
                if len(text_preview) > 200:
                    preview = "..." + preview
                tbl.add_row(Text.from_markup(f"  [dim]{preview}[/dim]"))

            return Panel(tbl, title=f"Ch.{ch_num} {sec.capitalize()}", border_style="blue")

        with Live(_build_display(), console=console, refresh_per_second=4, transient=True) as live:
            for event in gen:
                t = event.get("type")

                if t == "token":
                    tok_count += 1
                    text_preview += event["text"]
                    live.update(_build_display())

                elif t == "progress":
                    current_stage = event.get("stage", current_stage)
                    live.update(_build_display())

                elif t == "scores":
                    last_scores = event.get("scores", {})
                    score_history.append(last_scores.get("overall", 0))
                    current_iter = event.get("iteration", current_iter)
                    live.update(_build_display())

                elif t == "iteration_start":
                    current_iter = event.get("iteration", 1)
                    iter_max = event.get("max", max_iter)
                    text_preview = ""
                    tok_count = 0
                    tok_t0 = _time.monotonic()
                    live.update(_build_display())

                elif t == "revision_verdict":
                    action = event["action"]
                    icon = "\u2713" if action == "KEEP" else "\u2717"
                    color = "green" if action == "KEEP" else "red"
                    verdicts.append(
                        f"[{color}]{icon} {action} "
                        f"{event['old_score']:.2f}\u2192{event['new_score']:.2f}[/{color}]"
                    )
                    text_preview = ""
                    tok_count = 0
                    tok_t0 = _time.monotonic()
                    live.update(_build_display())

                elif t == "converged":
                    verdicts.append(
                        f"[bold green]\u2713 CONVERGED "
                        f"(iter {event['iteration']}, score {event['final_score']:.2f})[/bold green]"
                    )
                    live.update(_build_display())

                elif t == "verification":
                    vdata = event.get("data", {})
                    gs = vdata.get("groundedness_score", "?")
                    color = "green" if isinstance(gs, (int, float)) and gs >= 0.8 else "yellow"
                    grounding = f"[{color}]Groundedness: {gs}[/{color}]"
                    live.update(_build_display())

                elif t == "completed":
                    result = event

                elif t == "error":
                    console.print(f"[red]Error:[/red] {event.get('message', '')}")

        # Print final summary for this section
        if result:
            wc = result.get("word_count", 0)
            fs = result.get("final_score", 0)
            iters = result.get("iterations", 0)
            console.print(
                f"  [green]\u2713[/green] {sec.capitalize()}: "
                f"{wc} words, {iters} iterations"
                + (f", score {fs:.2f}" if fs else "")
            )
            if fs >= target_score:
                converged += 1

    console.print(f"\n[bold green]\u2713 Autowrite complete:[/bold green] "
                  f"{total} sections, {converged} converged")


# ── export ─────────────────────────────────────────────────────────────────────

@app.command()
def export(
    book_title: Annotated[str, typer.Argument(help="Book title or ID fragment.")],
    output:     Path | None = typer.Option(None, "--output", "-o",
                                            help="Output file (default: <book_title>.md)."),
    include_sources: bool = typer.Option(True, "--sources/--no-sources",
                                          help="Include source citations per chapter."),
    fmt:        str = typer.Option("markdown", "--format", "-f",
                                    help="Export format: markdown, html, pdf, epub, bibtex, latex, docx."),
):
    """
    Compile all chapter drafts into a single document.

    Formats:
      markdown  — Markdown with inline [N] citations + bibliography (default)
      html      — self-contained static reader (same look as the web reader)
      pdf       — HTML → PDF via weasyprint (Phase 40 — parity with the web export)
      epub      — Markdown → EPUB via Pandoc (Phase 40; requires pandoc installed)
      bibtex    — .bib file from all cited papers' metadata
      latex     — Markdown → LaTeX via Pandoc (requires pandoc installed)
      docx      — Markdown → DOCX via Pandoc (requires pandoc installed)

    Examples:

      sciknow book export "Global Cooling"

      sciknow book export "Global Cooling" --format pdf -o book.pdf

      sciknow book export "Global Cooling" --format epub -o book.epub

      sciknow book export "Global Cooling" --format bibtex -o refs.bib
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    _SECTION_ORDER = {
        "introduction": 0, "methods": 1, "results": 2,
        "discussion": 3, "conclusion": 4,
    }

    with get_session() as session:
        book = _get_book(session, book_title)
        if not book:
            console.print(f"[red]Book not found:[/red] {book_title}")
            raise typer.Exit(1)

        chapters = session.execute(text("""
            SELECT id::text, number, title, description
            FROM book_chapters WHERE book_id = :bid ORDER BY number
        """), {"bid": book[0]}).fetchall()

        drafts = session.execute(text("""
            SELECT d.chapter_id::text, d.title, d.section_type, d.content,
                   d.word_count, d.sources, bc.number
            FROM drafts d
            LEFT JOIN book_chapters bc ON bc.id = d.chapter_id
            WHERE d.book_id = :bid
            ORDER BY bc.number, d.created_at
        """), {"bid": book[0]}).fetchall()

    if not drafts:
        console.print("[yellow]No drafts found for this book.[/yellow]")
        console.print(f"Use [bold]sciknow book write {book_title!r} <chapter>[/bold] to create drafts.")
        raise typer.Exit(0)

    # Group drafts by chapter_id
    import re as _re
    from collections import defaultdict
    chapter_drafts: dict[str, list] = defaultdict(list)
    for d in drafts:
        chapter_drafts[d[0] or "__none__"].append(d)

    # Sort drafts within each chapter by section order
    for ch_id in chapter_drafts:
        chapter_drafts[ch_id].sort(
            key=lambda d: _SECTION_ORDER.get(d[2] or "", 99)
        )

    # ── Fix 2: Global citation dedup + renumbering ───────────────────────
    # First pass: collect all unique sources across all chapters, building
    # a global bibliography with unified [1]-[N] numbering. Then remap
    # per-draft [N] references to the global numbers in the content.

    # Build per-draft source lists and collect global bibliography
    global_sources: list[str] = []      # deduplicated, ordered
    global_source_set: set[str] = set()
    draft_source_lists: dict[str, list[str]] = {}  # draft_key -> [source_str, ...]

    for ch in chapters:
        ch_id = ch[0]
        for draft in chapter_drafts.get(ch_id, []):
            _, d_title, d_section, d_content, d_words, d_sources, _ = draft
            draft_key = f"{ch_id}:{d_section}"
            local_sources = []
            for s in (d_sources or []):
                if s:
                    local_sources.append(s)
                    # Strip the "[N] " prefix for dedup (same paper may be [1] in one draft, [3] in another)
                    clean = _re.sub(r'^\[\d+\]\s*', '', s).strip()
                    if clean not in global_source_set:
                        global_source_set.add(clean)
                        global_sources.append(s)
            draft_source_lists[draft_key] = local_sources

    # Build remap: for each draft, map old [N] → new global [M]
    def _source_key(s: str) -> str:
        return _re.sub(r'^\[\d+\]\s*', '', s).strip()

    global_num_by_key = {}
    for i, s in enumerate(global_sources, 1):
        global_num_by_key[_source_key(s)] = i

    def _remap_citations(content: str, local_sources: list[str]) -> str:
        """Remap [N] citations from per-draft numbering to global numbering."""
        local_to_global = {}
        for ls in local_sources:
            m = _re.match(r'^\[(\d+)\]', ls)
            if m:
                old_num = m.group(1)
                new_num = global_num_by_key.get(_source_key(ls))
                if new_num and old_num != str(new_num):
                    local_to_global[old_num] = str(new_num)

        if not local_to_global:
            return content

        # Replace [old] with [new], using a placeholder to avoid double-replace
        for old, new in local_to_global.items():
            content = _re.sub(
                rf'\[{old}\]',
                f'[__CITE_{new}__]',
                content,
            )
        # Remove placeholders
        content = _re.sub(r'\[__CITE_(\d+)__\]', r'[\1]', content)
        return content

    # Second pass: build the markdown with remapped citations
    lines: list[str] = []
    lines += [f"# {book[1]}", ""]
    if book[2]:
        lines += [f"*{book[2]}*", ""]
    lines += ["---", ""]

    for ch in chapters:
        ch_id, ch_num, ch_title, ch_desc = ch
        ch_drafts = chapter_drafts.get(ch_id, [])

        lines += [f"## Chapter {ch_num}: {ch_title}", ""]
        if ch_desc:
            lines += [f"*{ch_desc}*", ""]

        if not ch_drafts:
            lines += ["*[No draft yet]*", ""]
            continue

        for draft in ch_drafts:
            _, d_title, d_section, d_content, d_words, d_sources, _ = draft
            if d_section and d_section not in ("argument_map",):
                lines += [f"### {d_section.capitalize()}", ""]

            # Remap citations to global numbering
            draft_key = f"{ch_id}:{d_section}"
            remapped = _remap_citations(d_content or "", draft_source_lists.get(draft_key, []))
            lines += [remapped, ""]

    # Global bibliography with unified numbering
    if include_sources and global_sources:
        lines += ["---", "", "## Bibliography", ""]
        for i, s in enumerate(global_sources, 1):
            # Re-number the source line itself
            renumbered = _re.sub(r'^\[\d+\]', f'[{i}]', s)
            lines.append(f"{i}. {renumbered}")
        lines.append("")

    # Word count
    total_words = sum(d[4] or 0 for d in drafts)
    lines += ["", f"---", f"*{total_words:,} words · {len(drafts)} sections · {len(chapters)} chapters · {len(global_sources)} references*"]

    md = "\n".join(lines)

    safe = book[1].lower().replace(" ", "_").replace("/", "-")[:50]

    # ── HTML export (self-contained static reader) ─────────────────────
    if fmt == "html":
        from sciknow.web.app import _get_book_data, _render_book, set_book
        set_book(book[0], book[1])
        bk, chs, drs, gps, comms = _get_book_data()
        html = _render_book(bk, chs, drs, gps, comms)
        html_path = output or Path(f"{safe}.html")
        html_path.write_text(html, encoding="utf-8")
        console.print(
            f"[green]✓ HTML exported:[/green] [bold]{html_path}[/bold]  "
            f"({total_words:,} words) — open in any browser"
        )
        return

    # ── PDF export — Phase 40, parity with the web reader's export ─────
    # Uses weasyprint on the rendered HTML. No pandoc dependency; the
    # only requirement is weasyprint, which the web reader already
    # depends on for its PDF button (pyproject.toml).
    if fmt == "pdf":
        try:
            from weasyprint import HTML as _WPHTML
        except ImportError as exc:
            console.print(
                f"[red]PDF export requires weasyprint ({exc}).[/red]\n"
                f"Install: [bold]uv add weasyprint[/bold]"
            )
            raise typer.Exit(1)
        from sciknow.web.app import _get_book_data, _render_book, set_book
        set_book(book[0], book[1])
        bk, chs, drs, gps, comms = _get_book_data()
        html = _render_book(bk, chs, drs, gps, comms)
        pdf_path = output or Path(f"{safe}.pdf")
        try:
            _WPHTML(string=html).write_pdf(str(pdf_path))
        except Exception as exc:
            console.print(f"[red]PDF render failed:[/red] {exc}")
            raise typer.Exit(1)
        console.print(
            f"[green]✓ PDF exported:[/green] [bold]{pdf_path}[/bold]  "
            f"({total_words:,} words)"
        )
        return

    # ── EPUB export — Phase 40, Markdown → EPUB via pandoc ─────────────
    # Piggybacks on the same markdown generation path used by latex/
    # docx above. Pandoc's built-in EPUB writer is the pragmatic
    # choice — no extra Python dep, and it already does citeproc
    # against the generated BibTeX.
    if fmt == "epub":
        import subprocess, shutil
        if not shutil.which("pandoc"):
            console.print(
                "[red]Pandoc not installed.[/red] Required for EPUB export.\n"
                "Install: [bold]sudo apt install pandoc[/bold]"
            )
            raise typer.Exit(1)
        md_path = Path(f"{safe}.md")
        md_path.write_text(md, encoding="utf-8")
        bib = _generate_bibtex(None, book[0])
        bib_path = md_path.with_suffix(".bib")
        bib_path.write_text(bib, encoding="utf-8")
        epub_path = output or Path(f"{safe}.epub")
        cmd = [
            "pandoc", str(md_path),
            "--citeproc", f"--bibliography={bib_path}",
            "--metadata", f"title={book[1]}",
            "-o", str(epub_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[red]Pandoc failed:[/red] {result.stderr[:300]}")
            raise typer.Exit(1)
        console.print(
            f"[green]✓ EPUB exported:[/green] [bold]{epub_path}[/bold]  "
            f"({total_words:,} words)"
        )
        console.print(f"[dim]BibTeX: {bib_path}[/dim]")
        return

    # ── BibTeX export ────────────────────────────────────────────────────
    if fmt == "bibtex":
        bib = _generate_bibtex(session if 'session' in dir() else None, book[0])
        bib_path = output or Path(f"{safe}.bib")
        bib_path.write_text(bib, encoding="utf-8")
        console.print(f"[green]✓ BibTeX exported:[/green] {bib_path}")
        return

    # ── Markdown (default) ───────────────────────────────────────────────
    md_path = output or Path(f"{safe}.md")
    md_path.write_text(md, encoding="utf-8")

    if fmt == "markdown":
        console.print(
            f"[green]✓ Exported [bold]{book[1]}[/bold][/green] → [bold]{md_path}[/bold]  "
            f"({total_words:,} words, {len(drafts)} sections)"
        )
        return

    # ── LaTeX / DOCX via Pandoc ──────────────────────────────────────────
    if fmt in ("latex", "docx"):
        import subprocess, shutil
        if not shutil.which("pandoc"):
            console.print(
                "[red]Pandoc not installed.[/red] Required for LaTeX/DOCX export.\n"
                "Install: [bold]sudo apt install pandoc[/bold]"
            )
            raise typer.Exit(1)

        bib = _generate_bibtex(None, book[0])
        bib_path = md_path.with_suffix(".bib")
        bib_path.write_text(bib, encoding="utf-8")

        ext = ".tex" if fmt == "latex" else ".docx"
        out_path = output or Path(f"{safe}{ext}")

        cmd = [
            "pandoc", str(md_path),
            "--citeproc", f"--bibliography={bib_path}",
            "-o", str(out_path),
        ]
        if fmt == "latex":
            cmd.extend(["--standalone"])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[red]Pandoc failed:[/red] {result.stderr[:300]}")
            raise typer.Exit(1)

        console.print(
            f"[green]✓ Exported [bold]{book[1]}[/bold][/green] → [bold]{out_path}[/bold]  "
            f"({total_words:,} words)"
        )
        console.print(f"[dim]BibTeX: {bib_path}[/dim]")
        return

    console.print(f"[red]Unknown format: {fmt}[/red]. Use: markdown, html, pdf, epub, bibtex, latex, docx")


def _generate_bibtex(session, book_id: str) -> str:
    """Generate a BibTeX file from the metadata of all papers cited in the book's drafts."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as sess:
        # Get all DOIs from drafts' sources (which are stored as JSONB string arrays).
        # Also get all documents cited by any chunk that was used as a source.
        rows = sess.execute(text("""
            SELECT DISTINCT pm.doi, pm.title, pm.year, pm.authors, pm.journal,
                   pm.volume, pm.issue, pm.pages, pm.publisher
            FROM paper_metadata pm
            WHERE pm.doi IS NOT NULL
            ORDER BY pm.year DESC NULLS LAST, pm.title
        """)).fetchall()

    entries = []
    for doi, title, year, authors, journal, volume, issue, pages, publisher in rows:
        if not doi or not title:
            continue
        # Generate a citekey: AuthorYear format
        author_str = ""
        citekey_author = "Unknown"
        if authors and isinstance(authors, list) and len(authors) > 0:
            first = authors[0]
            name = first.get("name", "") if isinstance(first, dict) else str(first)
            parts = name.split()
            citekey_author = parts[-1] if parts else "Unknown"
            # Format all authors for BibTeX
            auth_list = []
            for a in authors[:10]:
                n = a.get("name", "") if isinstance(a, dict) else str(a)
                auth_list.append(n)
            author_str = " and ".join(auth_list)

        citekey = f"{citekey_author}{year or 'nd'}"

        entry = f"@article{{{citekey},\n"
        entry += f"  title = {{{title}}},\n"
        if author_str:
            entry += f"  author = {{{author_str}}},\n"
        if year:
            entry += f"  year = {{{year}}},\n"
        if journal:
            entry += f"  journal = {{{journal}}},\n"
        if volume:
            entry += f"  volume = {{{volume}}},\n"
        if issue:
            entry += f"  number = {{{issue}}},\n"
        if pages:
            entry += f"  pages = {{{pages}}},\n"
        if publisher:
            entry += f"  publisher = {{{publisher}}},\n"
        entry += f"  doi = {{{doi}}},\n"
        entry += "}\n"
        entries.append(entry)

    return "\n".join(entries)


# ── Track A: draft inspection (scores, compare) ───────────────────────────────


def _load_draft_metadata(draft_id: str) -> dict | None:
    """Fetch a draft's row + custom_metadata. Accepts a UUID prefix."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        row = session.execute(text("""
            SELECT id::text, title, section_type, topic, content, word_count,
                   custom_metadata, created_at, version, model_used,
                   book_id::text, chapter_id::text
            FROM drafts WHERE id::text LIKE :q
            ORDER BY created_at DESC
            LIMIT 1
        """), {"q": f"{draft_id}%"}).fetchone()
    if not row:
        return None
    return {
        "id": row[0], "title": row[1], "section_type": row[2],
        "topic": row[3], "content": row[4], "word_count": row[5],
        "custom_metadata": row[6] or {}, "created_at": row[7],
        "version": row[8], "model_used": row[9],
        "book_id": row[10], "chapter_id": row[11],
    }


def _format_score_cell(value, threshold_good: float = 0.85, threshold_mid: float = 0.7) -> str:
    """Render a 0-1 score with color tags for Rich tables."""
    if value is None or not isinstance(value, (int, float)):
        return "[dim]—[/dim]"
    color = "green" if value >= threshold_good else "yellow" if value >= threshold_mid else "red"
    return f"[{color}]{value:.2f}[/{color}]"


@draft_app.command(name="scores")
def draft_scores(
    draft_id: Annotated[str, typer.Argument(help="Draft ID or prefix.")],
):
    """
    Show the iteration-by-iteration score history for an autowrite draft.

    Reads custom_metadata.score_history (populated by autowrite_section_stream)
    and prints a Rich table with one row per iteration showing all six scoring
    dimensions, verification flag counts (extrapolated/overstated/missing),
    Chain-of-Verification score if it ran, and the keep/discard verdict.

    Drafts created by `book write` (not autowrite) won't have a score history —
    only autowrite persists this data.

    Example:

      sciknow book draft scores 3f2a1b4c
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    d = _load_draft_metadata(draft_id)
    if not d:
        console.print(f"[red]Draft not found: {draft_id}[/red]")
        raise typer.Exit(1)

    meta = d["custom_metadata"] or {}
    history = meta.get("score_history") or []
    feature_versions = meta.get("feature_versions") or {}

    console.print(Rule(f"[bold]{d['title']}[/bold]"))
    console.print(f"[dim]ID: {d['id']}  ·  v{d['version']}  ·  {d['word_count']} words  ·  {d['section_type']}[/dim]")
    if feature_versions:
        active = [k.replace("phase", "P").replace("_", " ")
                  for k, v in feature_versions.items()
                  if k.startswith("phase") and v]
        if active:
            console.print(f"[dim]Features active: {', '.join(active)}[/dim]")

    if not history:
        console.print(
            "\n[yellow]No score history persisted on this draft.[/yellow]\n"
            "[dim]Only autowrite drafts have a score history. Drafts created by "
            "`book write` (without autowrite) record only the final state.[/dim]"
        )
        return

    # Score table — one row per iteration.
    table = Table(
        title=f"Convergence trajectory · {len(history)} iteration(s)",
        box=box.SIMPLE_HEAD,
        expand=True,
    )
    table.add_column("Iter", style="bold", width=4)
    table.add_column("Ground", justify="right", width=7)
    table.add_column("Compl", justify="right", width=7)
    table.add_column("Coher", justify="right", width=7)
    table.add_column("CitAcc", justify="right", width=7)
    table.add_column("HedFid", justify="right", width=7)
    table.add_column("Overall", justify="right", width=8, style="bold")
    table.add_column("Weakest", style="cyan", width=14)
    table.add_column("Verify", style="dim", width=18)
    table.add_column("CoVe", style="dim", width=12)
    table.add_column("Verdict", width=10)

    for h in history:
        s = h.get("scores") or {}
        v = h.get("verification") or {}
        cove = h.get("cove") or {}
        verify_short = ""
        if v.get("n_extrapolated") or v.get("n_overstated") or v.get("n_misrepresented"):
            parts = []
            if v.get("n_extrapolated"):
                parts.append(f"[yellow]{v['n_extrapolated']}E[/yellow]")
            if v.get("n_overstated"):
                parts.append(f"[orange1]{v['n_overstated']}O[/orange1]")
            if v.get("n_misrepresented"):
                parts.append(f"[red]{v['n_misrepresented']}M[/red]")
            verify_short = " ".join(parts)
        elif v.get("n_supported"):
            verify_short = f"[green]{v['n_supported']}S[/green]"
        cove_short = ""
        if cove.get("ran"):
            score = cove.get("score")
            n_high = cove.get("n_high_severity", 0)
            n_med = cove.get("n_medium_severity", 0)
            score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
            cove_short = score_s
            if n_high or n_med:
                cove_short += f" [red]{n_high}H[/red]/[yellow]{n_med}M[/yellow]"
        verdict = h.get("revision_verdict") or ""
        verdict_color = "green" if verdict == "KEEP" else "red" if verdict == "DISCARD" else "dim"
        weakest = s.get("weakest_dimension", "")
        table.add_row(
            str(h.get("iteration", "?")),
            _format_score_cell(s.get("groundedness")),
            _format_score_cell(s.get("completeness")),
            _format_score_cell(s.get("coherence")),
            _format_score_cell(s.get("citation_accuracy")),
            _format_score_cell(s.get("hedging_fidelity")),
            _format_score_cell(s.get("overall")),
            weakest[:14],
            verify_short or "[dim]—[/dim]",
            cove_short or "[dim]—[/dim]",
            f"[{verdict_color}]{verdict or '—'}[/{verdict_color}]",
        )

    console.print(table)

    # Improvement summary across iterations.
    if len(history) >= 2:
        first = history[0].get("scores") or {}
        last = history[-1].get("scores") or {}
        deltas = []
        for dim in ("groundedness", "completeness", "coherence", "citation_accuracy",
                    "hedging_fidelity", "overall"):
            f, l = first.get(dim), last.get(dim)
            if isinstance(f, (int, float)) and isinstance(l, (int, float)):
                d = l - f
                arrow = "↑" if d > 0.001 else "↓" if d < -0.001 else "="
                color = "green" if d > 0.001 else "red" if d < -0.001 else "dim"
                deltas.append(f"[{color}]{dim[:8]}: {f:.2f} {arrow} {l:.2f}[/{color}]")
        if deltas:
            console.print()
            console.print("[bold]Δ first → last:[/bold]  " + "  ·  ".join(deltas))

    final = meta.get("final_overall")
    if final is not None:
        console.print(f"\n[dim]Final overall: {final:.3f}  ·  target: {meta.get('target_score', '?')}  ·  max_iter: {meta.get('max_iter', '?')}[/dim]")


@draft_app.command(name="compare")
def draft_compare(
    draft_a: Annotated[str, typer.Argument(help="First draft ID or prefix.")],
    draft_b: Annotated[str, typer.Argument(help="Second draft ID or prefix.")],
    rescore: bool = typer.Option(
        False, "--rescore",
        help="Re-run the scorer + verifier against fresh retrieval. Slow but accurate.",
    ),
    model: str | None = typer.Option(None, "--model", help="LLM for rescoring."),
):
    """
    Compare two drafts side-by-side.

    By default reads the persisted score history from custom_metadata
    (no LLM calls). With --rescore, re-runs the scorer + verifier against
    fresh retrieval for both drafts so the comparison reflects the CURRENT
    rubric (useful for comparing pre- and post-Phase-7 drafts).

    Example:

      sciknow book draft compare 3f2a1b 8d4c2f          # fast, persisted only
      sciknow book draft compare 3f2a1b 8d4c2f --rescore  # accurate, runs LLM
    """
    from sciknow.cli import preflight
    preflight(qdrant=False)

    a = _load_draft_metadata(draft_a)
    b = _load_draft_metadata(draft_b)
    if not a:
        console.print(f"[red]Draft A not found: {draft_a}[/red]")
        raise typer.Exit(1)
    if not b:
        console.print(f"[red]Draft B not found: {draft_b}[/red]")
        raise typer.Exit(1)

    def _final_scores(d: dict) -> dict:
        meta = d.get("custom_metadata") or {}
        history = meta.get("score_history") or []
        if history:
            return (history[-1] or {}).get("scores") or {}
        return {}

    if rescore:
        preflight(qdrant=True)
        from sciknow.core import book_ops
        from sciknow.rag import prompts as rag_prompts
        from sciknow.rag.llm import complete as llm_complete
        from sciknow.storage.db import get_session
        from sciknow.storage.qdrant import get_client
        qdrant = get_client()

        def _rescore(d: dict) -> dict:
            console.print(f"[dim]Rescoring {d['title'][:60]}...[/dim]")
            with get_session() as session:
                results, _sources = book_ops._retrieve(
                    session, qdrant,
                    f"{d['section_type'] or ''} {d['topic'] or d['title']}",
                    candidate_k=50, context_k=12,
                )
            if not results:
                return {}
            sys_s, usr_s = rag_prompts.score_draft(
                d['section_type'] or "text", d['topic'] or d['title'],
                d['content'], results,
            )
            try:
                raw = llm_complete(sys_s, usr_s, model=model, temperature=0.0, num_ctx=16384, keep_alive=-1)
                return _json.loads(book_ops._clean_json(raw), strict=False)
            except Exception as exc:
                console.print(f"[red]Rescoring failed: {exc}[/red]")
                return {}

        a_scores = _rescore(a)
        b_scores = _rescore(b)
    else:
        a_scores = _final_scores(a)
        b_scores = _final_scores(b)
        if not a_scores and not b_scores:
            console.print(
                "[yellow]Neither draft has persisted score history.[/yellow]  "
                "Re-run with [bold]--rescore[/bold] to evaluate them with the current rubric."
            )
            return

    console.print(Rule("[bold]Draft comparison[/bold]"))
    table = Table(box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Dimension", style="bold", width=18)
    table.add_column(f"A: {a['title'][:28]}", justify="right")
    table.add_column(f"B: {b['title'][:28]}", justify="right")
    table.add_column("Δ (B − A)", justify="right")

    for dim in ("groundedness", "completeness", "coherence", "citation_accuracy",
                "hedging_fidelity", "overall"):
        va = a_scores.get(dim)
        vb = b_scores.get(dim)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = vb - va
            color = "green" if d > 0.001 else "red" if d < -0.001 else "dim"
            arrow = "↑" if d > 0.001 else "↓" if d < -0.001 else "="
            delta_str = f"[{color}]{arrow} {d:+.3f}[/{color}]"
        else:
            delta_str = "[dim]—[/dim]"
        table.add_row(
            dim,
            _format_score_cell(va),
            _format_score_cell(vb),
            delta_str,
        )

    console.print(table)
    console.print(
        f"\n[dim]A: {a['id'][:8]} v{a['version']} · {a['word_count']} words · {a['model_used'] or '?'}[/dim]"
    )
    console.print(
        f"[dim]B: {b['id'][:8]} v{b['version']} · {b['word_count']} words · {b['model_used'] or '?'}[/dim]"
    )

    # Feature-version delta if both have it
    a_meta = a.get("custom_metadata") or {}
    b_meta = b.get("custom_metadata") or {}
    a_feat = a_meta.get("feature_versions") or {}
    b_feat = b_meta.get("feature_versions") or {}
    if a_feat or b_feat:
        diffs = []
        for k in sorted(set(a_feat.keys()) | set(b_feat.keys())):
            if a_feat.get(k) != b_feat.get(k):
                diffs.append(f"{k}: A={a_feat.get(k)} → B={b_feat.get(k)}")
        if diffs:
            console.print(f"\n[dim]Feature differences:[/dim]")
            for d in diffs:
                console.print(f"  [dim]· {d}[/dim]")


@app.command(name="promote-lessons")
def promote_lessons(
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print what would be promoted; don't write. Recommended "
             "on the first run after accumulating lessons from >=3 books.",
    ),
    min_importance: float = typer.Option(
        0.8, "--min-importance",
        help="Minimum per-lesson importance to consider.",
    ),
    min_books: int = typer.Option(
        3, "--min-books",
        help="Minimum distinct books a lesson must appear in (via "
             "embedding similarity) to be promoted.",
    ),
    cosine_threshold: float = typer.Option(
        0.85, "--cosine",
        help="Cosine similarity threshold for bucketing lessons as "
             "'the same' across books.",
    ),
    limit: int = typer.Option(
        50, "--limit",
        help="Max lessons to promote in this run.",
    ),
):
    """Phase 47.4 — promote book-scoped autowrite_lessons to the global pool.

    Iterates high-importance lessons, buckets them by embedding
    similarity, and promotes any bucket that spans N distinct books
    into scope='global' (cross-project). Subsequent autowrite runs in
    ANY project will retrieve from both scopes.

    Safe to run repeatedly — lessons already present at global scope
    (via embedding near-match) are skipped.

    Examples:

      sciknow book promote-lessons --dry-run
      sciknow book promote-lessons
      sciknow book promote-lessons --min-books 2   # looser for small libraries
    """
    from sciknow.core.book_ops import promote_lessons_to_global

    console.print(
        f"[bold]Promote lessons to global[/bold] "
        f"(min_importance={min_importance}, min_books={min_books}, "
        f"cosine>={cosine_threshold}, limit={limit})"
    )
    if dry_run:
        console.print("[dim]dry-run — no writes[/dim]")

    summary = promote_lessons_to_global(
        dry_run=dry_run,
        min_importance=min_importance,
        min_books=min_books,
        cosine_threshold=cosine_threshold,
        limit=limit,
    )

    console.print(
        f"\n  candidates:              {summary.get('candidates', 0)}"
        f"\n  buckets:                 {summary.get('buckets', 0)}"
        f"\n  promoted:                [green]{summary.get('promoted', 0)}[/green]"
        f"\n  skipped — too few books: {summary.get('skipped_too_few_books', 0)}"
        f"\n  skipped — already global:{summary.get('skipped_already_global', 0)}"
    )
    details = summary.get("details") or []
    if details:
        console.print()
        console.print(f"[bold]{'Would promote' if dry_run else 'Promoted'}:[/bold]")
        for i, d in enumerate(details, 1):
            console.print(
                f"  [cyan]{i}.[/cyan] [{d['kind']}/{d['dimension']}] "
                f"(bucket={d['bucket_size']}, books={d['n_books']})"
            )
            console.print(f"      [dim]{d['text']}[/dim]")
    if "error" in summary:
        console.print(f"[red]Error:[/red] {summary['error']}")


@app.command(name="autowrite-bench")
def autowrite_bench(
    book_ref: Annotated[str, typer.Argument(help="Book title fragment or ID.")],
    chapter_ref: Annotated[str, typer.Argument(help="Chapter number or title fragment.")],
    section_type: Annotated[str, typer.Argument(help="Section type (e.g. 'overview', 'key_evidence').")],
    runs: int = typer.Option(3, "--runs", "-n", help="How many independent autowrite runs."),
    max_iter: int = typer.Option(2, "--max-iter", help="max_iter passed to each autowrite run."),
    target_score: float = typer.Option(0.85, "--target-score", help="target_score passed to each run."),
    model: str | None = typer.Option(None, "--model", help="LLM model override."),
    use_cove: bool = typer.Option(True, "--cove/--no-cove"),
    use_step_back: bool = typer.Option(True, "--step-back/--no-step-back"),
    use_plan: bool = typer.Option(True, "--plan/--no-plan"),
):
    """
    Run autowrite N times under identical conditions and report variance.

    Useful for measuring scorer/verifier stability and validating that the
    convergence loop is well-calibrated. Each run's score history is saved
    to data/bench/<timestamp>/run_N.json so you can inspect them later.

    The output is a Rich table with mean ± std for each scoring dimension
    across runs, plus per-run wall time. If the std is high relative to
    the mean, the loop is noisy — either bump max_iter or accept the noise
    as the floor for A/B comparisons.

    Example:

      sciknow book autowrite-bench "Global Cooling" 3 overview --runs 5
      sciknow book autowrite-bench "Global Cooling" 3 overview --no-cove
    """
    from sciknow.cli import preflight
    preflight()

    from datetime import datetime
    from sciknow.config import settings  # Phase 43d — for project data dir
    from sciknow.core import book_ops
    from sciknow.storage.db import get_session

    with get_session() as session:
        book = _get_book(session, book_ref)
        if not book:
            console.print(f"[red]Book not found: {book_ref}[/red]")
            raise typer.Exit(1)
        ch = _get_chapter(session, book[0], chapter_ref)
        if not ch:
            console.print(f"[red]Chapter not found: {chapter_ref}[/red]")
            raise typer.Exit(1)
        book_id = book[0]
        ch_id = ch[0]
        ch_num = ch[1]
        ch_title = ch[2]

    # Phase 43d — project-aware bench output.
    bench_dir = settings.data_dir / "bench" / datetime.now().strftime("%Y%m%d-%H%M%S")
    bench_dir.mkdir(parents=True, exist_ok=True)

    console.print(Rule(f"[bold]autowrite-bench[/bold] · Ch.{ch_num} {ch_title} — {section_type}"))
    console.print(f"[dim]{runs} runs · max_iter={max_iter} · target={target_score} · cove={use_cove} · step_back={use_step_back} · plan={use_plan}[/dim]")
    console.print(f"[dim]Output: {bench_dir}[/dim]\n")

    import time as _time
    runs_data = []

    for run_idx in range(1, runs + 1):
        console.print(f"[bold cyan]Run {run_idx}/{runs}[/bold cyan]")
        t0 = _time.monotonic()
        run_history = []
        final_scores = None
        final_overall = None
        last_error = None

        try:
            for ev in book_ops.autowrite_section_stream(
                book_id=book_id,
                chapter_id=ch_id,
                section_type=section_type,
                model=model,
                max_iter=max_iter,
                target_score=target_score,
                use_plan=use_plan,
                use_step_back=use_step_back,
                use_cove=use_cove,
                auto_expand=False,
            ):
                t = ev.get("type")
                if t == "scores":
                    final_scores = ev.get("scores", {})
                    final_overall = final_scores.get("overall")
                elif t == "completed":
                    run_history = ev.get("history") or []
                    final_overall = ev.get("final_score") or final_overall
                elif t == "error":
                    last_error = ev.get("message", "unknown")
                    console.print(f"  [red]Error: {last_error}[/red]")
                    break
        except Exception as exc:
            last_error = str(exc)
            console.print(f"  [red]Run {run_idx} crashed: {exc}[/red]")

        elapsed = _time.monotonic() - t0
        run_data = {
            "run_idx": run_idx,
            "elapsed_sec": round(elapsed, 1),
            "final_scores": final_scores,
            "final_overall": final_overall,
            "n_iterations": len(run_history),
            "history": run_history,
            "error": last_error,
        }
        runs_data.append(run_data)

        # Persist this run.
        out_file = bench_dir / f"run_{run_idx:02d}.json"
        out_file.write_text(_json.dumps(run_data, indent=2, default=str))
        if final_overall is not None:
            console.print(f"  [green]→ overall {final_overall:.3f} in {elapsed:.0f}s ({len(run_history)} iter)[/green]")
        elif last_error:
            console.print(f"  [red]→ failed in {elapsed:.0f}s[/red]")

    # Summary table — mean ± std across runs
    import statistics
    successful = [r for r in runs_data if r.get("final_scores")]
    if not successful:
        console.print("\n[red]All runs failed.[/red]")
        return

    console.print()
    table = Table(title=f"Bench summary · {len(successful)}/{runs} runs successful",
                  box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Dimension", style="bold")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for dim in ("groundedness", "completeness", "coherence", "citation_accuracy",
                "hedging_fidelity", "overall"):
        vals = [r["final_scores"].get(dim) for r in successful if r["final_scores"]]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            table.add_row(dim, "—", "—", "—", "—")
            continue
        mean_v = statistics.mean(vals)
        std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
        table.add_row(
            dim,
            _format_score_cell(mean_v),
            f"{std_v:.3f}",
            _format_score_cell(min(vals)),
            _format_score_cell(max(vals)),
        )

    console.print(table)

    elapsed_total = sum(r["elapsed_sec"] for r in runs_data)
    avg_elapsed = elapsed_total / max(len(runs_data), 1)
    console.print(
        f"\n[dim]Total wall time: {elapsed_total:.0f}s  ·  avg per run: {avg_elapsed:.0f}s  ·  results in {bench_dir}[/dim]"
    )


# ── Phase 32.9 — Layer 4: preference pair export ──────────────────────────────


@preferences_app.command(name="export")
def preferences_export(
    book: str = typer.Argument(
        None,
        help="Book title or ID fragment (omit to export from all books)",
    ),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Output JSONL path (defaults to data/preferences/<book>.jsonl)",
    ),
    min_score: float = typer.Option(
        0.7, "--min-score",
        help="Drop pairs where the higher score is below this threshold (low signal on both sides)",
    ),
    min_delta: float = typer.Option(
        0.02, "--min-delta",
        help="Drop pairs where the score gap is below this (too noisy to learn from)",
    ),
    require_approval: bool = typer.Option(
        False, "--require-approval",
        help="Only include pairs from runs whose final draft has custom_metadata.preference_approved=true (human-in-the-loop bias mitigation)",
    ),
    no_discard: bool = typer.Option(
        False, "--no-discard",
        help="Skip DISCARD-verdict pairs (the inverse signal). KEEP-only by default keeps things conservative.",
    ),
    stats: bool = typer.Option(
        False, "--stats",
        help="Show counts only, don't write the file",
    ),
):
    """Phase 32.9 — Export Layer 4 DPO preference pairs as JSONL.

    Walks `autowrite_iterations` and turns every revision attempt into a
    `{prompt, chosen, rejected}` record. Both KEEP and DISCARD verdicts
    produce pairs:

      KEEP    → chosen = post-revision, rejected = pre-revision
      DISCARD → chosen = pre-revision,  rejected = post-revision (inverse signal)

    Output format is the standard DPO shape, ready for HuggingFace TRL
    or any other DPO trainer. Use `--require-approval` for the human-
    in-the-loop bias mitigation: only pairs from drafts where the user
    explicitly marked the final result as good get exported.

    Examples:

      sciknow book preferences export                            # all books
      sciknow book preferences export "Global Cooling"           # one book
      sciknow book preferences export "Global Cooling" --stats   # count only
      sciknow book preferences export "Global Cooling" -o /tmp/dataset.jsonl
      sciknow book preferences export "Global Cooling" --require-approval
    """
    from sciknow.core.book_ops import _export_preference_pairs
    from sciknow.storage.db import get_session

    book_id = None
    if book:
        with get_session() as session:
            row = _get_book(session, book)
        if not row:
            console.print(f"[red]Book not found:[/red] {book}")
            raise typer.Exit(code=1)
        book_id = row[0]
        console.print(f"[dim]exporting preferences for book: {row[1]} ({book_id[:8]}...)[/dim]")
    else:
        console.print("[dim]exporting preferences for ALL books[/dim]")

    if stats:
        # Run the export to a tempfile, count, then delete it.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            n, _ = _export_preference_pairs(
                book_id=book_id, output_path=tmp_path,
                min_score=min_score, min_delta=min_delta,
                require_approval=require_approval,
                include_discard=not no_discard,
            )
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass
        console.print(f"\n[bold]{n}[/bold] preference pairs would be exported")
        console.print(f"  filters: min_score={min_score}, min_delta={min_delta}, "
                      f"approval={'required' if require_approval else 'off'}, "
                      f"discards={'excluded' if no_discard else 'included'}")
        return

    n, path = _export_preference_pairs(
        book_id=book_id, output_path=output,
        min_score=min_score, min_delta=min_delta,
        require_approval=require_approval,
        include_discard=not no_discard,
    )
    console.print(f"\n[green]✓[/green] exported [bold]{n}[/bold] preference pairs to [cyan]{path}[/cyan]")
    if n == 0:
        console.print(
            "[dim]No pairs exported. Possible reasons:\n"
            "  - No autowrite runs have completed since Phase 32.9 shipped\n"
            "    (older runs don't have pre/post-revision content captured)\n"
            "  - All pairs were filtered out by --min-score / --min-delta\n"
            "  - --require-approval is on but no drafts are marked approved[/dim]"
        )
    else:
        # Show a small summary of what's in the file
        import json as _json
        verdicts: dict[str, int] = {}
        with path.open() as f:
            for line in f:
                rec = _json.loads(line)
                verdicts[rec["verdict"]] = verdicts.get(rec["verdict"], 0) + 1
        verdict_str = ", ".join(f"{k}={v}" for k, v in sorted(verdicts.items()))
        console.print(f"[dim]breakdown: {verdict_str}[/dim]")


# ── Phase 32.10 — Layer 5: style fingerprint CLI ──────────────────────────────


@style_app.command(name="refresh")
def style_refresh(
    book: str = typer.Argument(..., help="Book title or ID fragment"),
):
    """Phase 32.10 — Recompute the book's style fingerprint from the
    user's approved drafts (status in final/reviewed/revised).

    The fingerprint is persisted to books.custom_metadata.style_fingerprint
    and read by the autowrite loop on the next run, injecting a style
    anchor into the writer system prompt.

    Run this whenever you've marked new drafts as final/reviewed/revised
    and want the autowrite voice to start matching them. The computation
    is pure Python over draft text — no LLM cost, runs in milliseconds.
    """
    from sciknow.core.style_fingerprint import compute_style_fingerprint
    from sciknow.storage.db import get_session

    with get_session() as session:
        row = _get_book(session, book)
    if not row:
        console.print(f"[red]Book not found:[/red] {book}")
        raise typer.Exit(code=1)
    book_id, title = row[0], row[1]
    console.print(f"[dim]computing style fingerprint for {title} ({book_id[:8]}...)[/dim]")

    fp = compute_style_fingerprint(book_id)
    n_drafts = fp.get("n_drafts_sampled", 0)

    if n_drafts == 0:
        console.print(
            "\n[yellow]No approved drafts found.[/yellow] "
            "Mark drafts as final/reviewed/revised in the web reader "
            "(Status dropdown in the toolbar) and re-run this command."
        )
        return

    console.print(f"\n[green]✓[/green] fingerprint computed from [bold]{n_drafts}[/bold] approved draft(s)")
    console.print(f"  median sentence length: [bold]{fp.get('median_sentence_length', 0)}[/bold] words")
    console.print(f"  median paragraph length: [bold]{fp.get('median_paragraph_words', 0)}[/bold] words")
    console.print(f"  typical section length: [bold]~{int(fp.get('avg_words_per_draft', 0) or 0)}[/bold] words")
    console.print(f"  citation density: [bold]{fp.get('citations_per_100_words', 0)}[/bold] per 100 words")
    console.print(f"  hedging rate: [bold]{fp.get('hedging_rate', 0):.0%}[/bold] of sentences")
    transitions = fp.get("top_transitions") or []
    if transitions:
        t_str = ", ".join(f"\"{t['word']}\" ({t['count']})" for t in transitions[:5])
        console.print(f"  top transitions: {t_str}")
    if "samples_warning" in fp:
        console.print(f"\n[dim]{fp['samples_warning']}[/dim]")
    console.print(
        "\n[dim]The next autowrite run on this book will inject this "
        "fingerprint into the writer system prompt as a style anchor.[/dim]"
    )


@style_app.command(name="show")
def style_show(
    book: str = typer.Argument(..., help="Book title or ID fragment"),
):
    """Phase 32.10 — Display the persisted style fingerprint for a book.

    Reads books.custom_metadata.style_fingerprint without recomputing.
    Use `style refresh` to regenerate from the current draft state.
    """
    from sciknow.core.style_fingerprint import (
        get_style_fingerprint, format_fingerprint_for_prompt,
    )
    from sciknow.storage.db import get_session

    with get_session() as session:
        row = _get_book(session, book)
    if not row:
        console.print(f"[red]Book not found:[/red] {book}")
        raise typer.Exit(code=1)
    book_id, title = row[0], row[1]

    fp = get_style_fingerprint(book_id)
    if not fp:
        console.print(
            f"[yellow]No style fingerprint set for[/yellow] {title}.\n"
            f"Run [cyan]sciknow book style refresh \"{title}\"[/cyan] to compute one."
        )
        return

    console.print(f"\n[bold]Style fingerprint for {title}[/bold]")
    console.print(format_fingerprint_for_prompt(fp))
    if "samples_warning" in fp:
        console.print(f"[dim]{fp['samples_warning']}[/dim]")
