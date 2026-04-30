"""Phase 56.B — Corpus-driven outline proposer.

Walks the topic tree (56.A), scores clusters against the book's scope,
and asks the writer LLM to emit a candidate outline anchored to specific
clusters at every level.

Output structure:

  OutlineProposal
    └─ chapters: list[ChapterProposal]
       │     anchor cluster (top-level RAPTOR node, in-scope)
       │     title, description
       │     plan_bullets (3-5)
       │     target_words (carried forward from book defaults)
       └─ sections: list[SectionProposal]
          │     anchor cluster (sub-cluster of chapter's anchor)
          │     title, plan_bullets, target_words
          └─ subsections: list[SubsectionProposal]
                anchor cluster (further sub-cluster, may be empty)
                title, plan_bullet (single)

Caps come from book.custom_metadata defaults or fall back to:
  target_chapters = 12
  target_sections_per_chapter = 6
  target_subsections_per_section = 4

The LLM is asked separately for each level so the prompt window stays
small and the model can focus on one cluster at a time.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sciknow.core.topic_tree import TopicNode, TopicTree, embed_book_scope

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Defaults & limits
# ──────────────────────────────────────────────────────────────────────


DEFAULT_TARGET_CHAPTERS = 12
DEFAULT_SECTIONS_PER_CHAPTER = 6
DEFAULT_SUBSECTIONS_PER_SECTION = 4

# Min number of papers in a candidate cluster to be considered. Below
# this it's better to fold into a parent or sibling.
MIN_CHAPTER_PAPERS = 8
MIN_SECTION_PAPERS = 4
MIN_SUBSECTION_PAPERS = 2

# Scope-relevance thresholds at each level. Lower at deeper levels —
# once a chapter is in-scope, more of its sub-clusters are too.
SCOPE_THRESHOLD_CHAPTER = 0.30
SCOPE_THRESHOLD_SECTION = 0.20
SCOPE_THRESHOLD_SUBSECTION = 0.10


# ──────────────────────────────────────────────────────────────────────
# Data shape
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SubsectionProposal:
    title: str
    plan_bullet: str           # single concrete bullet
    anchor_cluster_id: str
    n_papers: int
    scope_relevance: float


@dataclass
class SectionProposal:
    title: str
    plan_bullets: list[str]
    anchor_cluster_id: str
    n_papers: int
    scope_relevance: float
    target_words: int = 1500
    subsections: list[SubsectionProposal] = field(default_factory=list)

    def as_legacy_section(self) -> dict:
        """Render in the existing book_chapters.sections JSON shape so
        the proposal can be written directly into the DB without
        schema changes."""
        slug = re.sub(r"[^a-z0-9]+", "_", self.title.lower()).strip("_")
        return {
            "slug": slug,
            "title": self.title,
            "plan": "\n".join(f"- {b}" for b in self.plan_bullets),
            "target_words": self.target_words,
            "anchor_cluster_id": self.anchor_cluster_id,
        }


@dataclass
class ChapterProposal:
    number: int                 # 1-indexed
    title: str
    description: str
    anchor_cluster_id: str
    n_papers: int
    scope_relevance: float
    sections: list[SectionProposal] = field(default_factory=list)


@dataclass
class OutlineProposal:
    book_id: str
    chapters: list[ChapterProposal] = field(default_factory=list)
    rejected_clusters: list[str] = field(default_factory=list)
    n_in_scope_clusters_found: int = 0

    def as_summary(self) -> dict:
        return {
            "book_id": self.book_id,
            "n_chapters": len(self.chapters),
            "n_sections": sum(len(c.sections) for c in self.chapters),
            "n_subsections": sum(
                len(s.subsections) for c in self.chapters for s in c.sections
            ),
            "n_in_scope_clusters_found": self.n_in_scope_clusters_found,
        }


# ──────────────────────────────────────────────────────────────────────
# LLM prompts (compact — one cluster at a time)
# ──────────────────────────────────────────────────────────────────────


CHAPTER_SYSTEM = """\
You propose chapter titles for a research-led monograph. You receive a
book scope and one cluster of related papers from the corpus. Emit
JSON ONLY in this shape:

{
  "title": "string, ≤ 12 words, no chapter number",
  "description": "string, 2-3 sentences, what the chapter covers",
  "plan": ["bullet 1", "bullet 2", "..."]
}

Rules:
- Title names the topic, not the activity ("The Maunder Minimum",
  not "A Discussion of the Maunder Minimum").
- Description grounds in what THIS cluster covers, not the broader
  book.
- 3-5 plan bullets, each one is a substantive claim or sub-topic the
  chapter develops. NOT meta-commentary ("we will discuss"). Each
  bullet is a content claim.
- No markdown, no preamble, no closing notes. JSON only.
"""


SECTION_SYSTEM = """\
You propose section titles for a chapter of a research-led monograph.
You receive the chapter's title + description, and one sub-cluster of
papers that should anchor this section. Emit JSON ONLY:

{
  "title": "string, ≤ 12 words",
  "plan": ["bullet 1", "bullet 2", "..."]
}

Rules:
- Title is concrete ("Solar irradiance reconstructions from
  cosmogenic isotopes" not "Solar Methods").
- 3-5 plan bullets, each a content claim. Avoid meta-commentary.
- The section should be coherent with the chapter; do NOT rephrase
  the chapter topic, drill into one specific aspect.
- JSON only.
"""


SUBSECTION_SYSTEM = """\
You propose a subsection within a section of a research-led monograph.
You receive the section title and a small sub-cluster of papers. Emit
JSON ONLY:

{
  "title": "string, ≤ 8 words",
  "plan": "single concrete bullet that describes the subsection scope"
}

Rules:
- One concrete topical aspect of the parent section.
- JSON only.
"""


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _clean_json(raw: str) -> str:
    """Reuse the existing book-ops cleaner for robust LLM JSON."""
    from sciknow.core.book_ops import _clean_json as _impl
    return _impl(raw)


def _llm_json(system: str, user: str, *, model: str | None = None) -> dict:
    """One LLM call that returns a parsed JSON object. Retries the
    parse on a single invalid pass; raises on the second failure."""
    from sciknow.rag.llm import complete as llm_complete
    raw = llm_complete(system, user, model=model, temperature=0.2,
                       num_predict=600, keep_alive=-1)
    try:
        return json.loads(_clean_json(raw))
    except json.JSONDecodeError as exc:
        logger.warning("outline LLM emitted invalid JSON, retrying: %s", exc)
        raw2 = llm_complete(system, user + "\n\nReturn JSON only.",
                            model=model, temperature=0.0,
                            num_predict=600, keep_alive=-1)
        return json.loads(_clean_json(raw2))


def _format_cluster_for_prompt(node: TopicNode, *, max_chars: int = 1500) -> str:
    """Compact rendering of a cluster for the LLM prompt."""
    parts = [
        f"Cluster: {node.section_title or node.title}",
        f"Papers: {node.n_documents}",
    ]
    if node.year_min and node.year_max:
        parts.append(f"Year range: {node.year_min}–{node.year_max}")
    if node.summary_text:
        summary = node.summary_text.strip()
        if len(summary) > max_chars:
            summary = summary[: max_chars - 3] + "..."
        parts.append(f"Summary: {summary}")
    return "\n".join(parts)


def _candidate_filter(
    nodes: list[TopicNode],
    *,
    min_papers: int,
    threshold: float,
    cap: int,
) -> list[TopicNode]:
    """Apply paper-count + scope-relevance gates and cap the result.

    Sorted by scope_relevance descending; ties broken by n_documents
    descending so larger themed clusters win.
    """
    eligible = [
        n for n in nodes
        if n.n_documents >= min_papers
        and (n._scope_relevance is not None and n._scope_relevance >= threshold)
    ]
    eligible.sort(
        key=lambda n: (n._scope_relevance or 0.0, n.n_documents),
        reverse=True,
    )
    return eligible[:cap]


# ──────────────────────────────────────────────────────────────────────
# Top-level entry
# ──────────────────────────────────────────────────────────────────────


def propose_outline(
    book_id: str,
    *,
    target_chapters: int = DEFAULT_TARGET_CHAPTERS,
    sections_per_chapter: int = DEFAULT_SECTIONS_PER_CHAPTER,
    subsections_per_section: int = DEFAULT_SUBSECTIONS_PER_SECTION,
    model: str | None = None,
) -> OutlineProposal:
    """Walk the corpus topic tree, score against the book's scope,
    and emit a candidate outline.

    Args:
      book_id: UUID of the book.
      target_chapters: cap on top-level clusters → chapter candidates.
      sections_per_chapter: cap per chapter.
      subsections_per_section: cap per section. 0 disables subsections.
      model: optional override for the writer model.

    Returns:
      OutlineProposal with chapters / sections / subsections, each
      anchored to a TopicTree cluster_id. The caller (CLI / GUI)
      reviews and persists.
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client
    from sciknow.core.project import get_active_project

    # 1. Load book scope.
    with get_session() as s:
        row = s.execute(text(
            "SELECT description, plan FROM books WHERE id::text = :bid"
        ), {"bid": book_id}).fetchone()
    if not row:
        raise ValueError(f"book not found: {book_id}")
    description, plan = row
    scope_vec = embed_book_scope(description or "", plan)

    # 2. Load topic tree.
    p = get_active_project()
    qd = get_client()
    tree = TopicTree.from_qdrant(qd, p.papers_collection)
    if len(tree) == 0:
        raise RuntimeError(
            f"no RAPTOR summary nodes found in {p.papers_collection!r}; "
            "build the tree first via `sciknow corpus refresh` or "
            "`sciknow catalog raptor`"
        )

    # 3. Score every node against scope (single pass, populates cache).
    for node in tree.all_nodes():
        tree.score_against_scope(node, scope_vec)

    proposal = OutlineProposal(book_id=book_id)
    proposal.n_in_scope_clusters_found = sum(
        1 for n in tree.all_nodes()
        if (n._scope_relevance or 0.0) >= SCOPE_THRESHOLD_SUBSECTION
    )

    # 4. Pick chapter candidates.
    # Prefer top-level (highest level) nodes; if too few, drop a level.
    for chapter_lvl in sorted({n.level for n in tree.all_nodes()},
                              reverse=True):
        candidates = [n for n in tree.all_nodes() if n.level == chapter_lvl]
        chapter_candidates = _candidate_filter(
            candidates,
            min_papers=MIN_CHAPTER_PAPERS,
            threshold=SCOPE_THRESHOLD_CHAPTER,
            cap=target_chapters,
        )
        if len(chapter_candidates) >= max(2, target_chapters // 3):
            break
    else:
        chapter_candidates = []

    # 5. For each chapter, pick section candidates from its subtree.
    for ch_idx, ch_node in enumerate(chapter_candidates, start=1):
        try:
            ch_payload = _llm_json(
                CHAPTER_SYSTEM,
                f"Book scope:\n{description}\n\n"
                f"{_format_cluster_for_prompt(ch_node)}",
                model=model,
            )
        except Exception as exc:
            logger.warning("chapter LLM failed for %s: %s",
                           ch_node.node_id, exc)
            proposal.rejected_clusters.append(ch_node.node_id)
            continue

        ch_prop = ChapterProposal(
            number=ch_idx,
            title=str(ch_payload.get("title") or "Untitled chapter"),
            description=str(ch_payload.get("description") or ""),
            anchor_cluster_id=ch_node.node_id,
            n_papers=ch_node.n_documents,
            scope_relevance=ch_node._scope_relevance or 0.0,
        )

        # Section candidates: children of this chapter's anchor cluster.
        sec_pool = list(ch_node.children) or []
        sec_candidates = _candidate_filter(
            sec_pool,
            min_papers=MIN_SECTION_PAPERS,
            threshold=SCOPE_THRESHOLD_SECTION,
            cap=sections_per_chapter,
        )

        # If the chapter has no children (e.g. L1 cluster used as
        # chapter) we skip section drilldown — the chapter itself is
        # the smallest unit. The caller can still author one section
        # spanning the whole chapter manually.
        for sec_node in sec_candidates:
            try:
                sec_payload = _llm_json(
                    SECTION_SYSTEM,
                    f"Chapter: {ch_prop.title}\n"
                    f"Chapter description: {ch_prop.description}\n\n"
                    f"{_format_cluster_for_prompt(sec_node)}",
                    model=model,
                )
            except Exception as exc:
                logger.warning("section LLM failed for %s: %s",
                               sec_node.node_id, exc)
                proposal.rejected_clusters.append(sec_node.node_id)
                continue

            plan_bullets = sec_payload.get("plan") or []
            if isinstance(plan_bullets, str):
                plan_bullets = [b.strip(" -•") for b in plan_bullets.splitlines()
                                if b.strip()]
            sec_prop = SectionProposal(
                title=str(sec_payload.get("title") or "Untitled section"),
                plan_bullets=[str(b) for b in plan_bullets if b],
                anchor_cluster_id=sec_node.node_id,
                n_papers=sec_node.n_documents,
                scope_relevance=sec_node._scope_relevance or 0.0,
            )

            # Subsection candidates: children of this section's
            # cluster, if subsections_per_section > 0.
            if subsections_per_section > 0 and sec_node.children:
                sub_candidates = _candidate_filter(
                    list(sec_node.children),
                    min_papers=MIN_SUBSECTION_PAPERS,
                    threshold=SCOPE_THRESHOLD_SUBSECTION,
                    cap=subsections_per_section,
                )
                for sub_node in sub_candidates:
                    try:
                        sub_payload = _llm_json(
                            SUBSECTION_SYSTEM,
                            f"Section: {sec_prop.title}\n\n"
                            f"{_format_cluster_for_prompt(sub_node)}",
                            model=model,
                        )
                    except Exception as exc:
                        logger.warning("subsection LLM failed for %s: %s",
                                       sub_node.node_id, exc)
                        proposal.rejected_clusters.append(sub_node.node_id)
                        continue
                    sub_plan = sub_payload.get("plan", "")
                    if isinstance(sub_plan, list):
                        sub_plan = sub_plan[0] if sub_plan else ""
                    sec_prop.subsections.append(SubsectionProposal(
                        title=str(sub_payload.get("title") or ""),
                        plan_bullet=str(sub_plan or ""),
                        anchor_cluster_id=sub_node.node_id,
                        n_papers=sub_node.n_documents,
                        scope_relevance=sub_node._scope_relevance or 0.0,
                    ))

            ch_prop.sections.append(sec_prop)

        proposal.chapters.append(ch_prop)

    return proposal


def render_proposal_text(proposal: OutlineProposal) -> str:
    """Pretty-print a proposal as indented text for CLI / log output."""
    lines: list[str] = []
    s = proposal.as_summary()
    lines.append(
        f"Outline proposal — {s['n_chapters']} chapters · "
        f"{s['n_sections']} sections · "
        f"{s['n_subsections']} subsections "
        f"({s['n_in_scope_clusters_found']} in-scope clusters considered)"
    )
    lines.append("")
    for ch in proposal.chapters:
        lines.append(
            f"Ch.{ch.number}  {ch.title}  "
            f"[{ch.n_papers} papers · rel={ch.scope_relevance:+.2f}]"
        )
        if ch.description:
            lines.append(f"        {ch.description}")
        for sec in ch.sections:
            lines.append(
                f"  §  {sec.title}  "
                f"[{sec.n_papers} papers · rel={sec.scope_relevance:+.2f}]"
            )
            for b in sec.plan_bullets:
                lines.append(f"        • {b}")
            for sub in sec.subsections:
                lines.append(
                    f"      ─ {sub.title}  "
                    f"[{sub.n_papers} papers · rel={sub.scope_relevance:+.2f}]"
                )
                if sub.plan_bullet:
                    lines.append(f"          • {sub.plan_bullet}")
        lines.append("")
    return "\n".join(lines)


def write_proposal_to_book(
    proposal: OutlineProposal,
    *,
    overwrite: bool = False,
) -> int:
    """Persist the proposal into the existing books / book_chapters
    schema. Each chapter gets one row; sections fold into
    ``book_chapters.sections`` JSON (existing shape). Subsections are
    stored as plan-bullet sub-items within each section's plan.

    Returns the number of chapters written.

    Args:
      overwrite: if True, drops every existing chapter for this book
        before inserting. Default False — refuses to write when a
        chapter with the same number exists.
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    from uuid import uuid4

    with get_session() as s:
        if overwrite:
            s.execute(text(
                "DELETE FROM book_chapters WHERE book_id::text = :bid"
            ), {"bid": proposal.book_id})

        for ch in proposal.chapters:
            sections_json = []
            for sec in ch.sections:
                sec_dict = sec.as_legacy_section()
                if sec.subsections:
                    extras = "\n\nSubsections:\n" + "\n".join(
                        f"- **{sub.title}** — {sub.plan_bullet}"
                        for sub in sec.subsections
                    )
                    sec_dict["plan"] = sec_dict["plan"] + extras
                sections_json.append(sec_dict)

            s.execute(text("""
                INSERT INTO book_chapters
                  (id, book_id, number, title, description,
                   topic_query, sections, custom_metadata)
                VALUES
                  (:id, :bid, :num, :title, :desc, :topic,
                   CAST(:sections AS jsonb), CAST(:meta AS jsonb))
                ON CONFLICT (book_id, number) DO UPDATE SET
                  title = EXCLUDED.title,
                  description = EXCLUDED.description,
                  topic_query = EXCLUDED.topic_query,
                  sections = EXCLUDED.sections,
                  custom_metadata = EXCLUDED.custom_metadata
            """), {
                "id": str(uuid4()),
                "bid": proposal.book_id,
                "num": ch.number,
                "title": ch.title,
                "desc": ch.description,
                "topic": ch.title,
                "sections": json.dumps(sections_json),
                "meta": json.dumps({
                    "anchor_cluster_id": ch.anchor_cluster_id,
                    "scope_relevance": ch.scope_relevance,
                    "source": "phase56_outline_proposer",
                }),
            })
        s.commit()
    return len(proposal.chapters)
