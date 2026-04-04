"""
Prompt templates for all RAG and writing tasks.

All templates take a dict of variables and return a (system, user) tuple.
"""
from __future__ import annotations
import re

from sciknow.retrieval.context_builder import SearchResult


# ── APA-style citation helpers ─────────────────────────────────────────────────

def _apa_author(author: dict) -> str:
    """
    Format one author dict as "Last, F.F." (APA style).
    Handles both "First Last" and "Last, First" name strings.
    """
    name = author.get("name", "").strip()
    if not name:
        return ""

    # Already "Last, First" format
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        last = parts[0]
        given = parts[1] if len(parts) > 1 else ""
    else:
        # "First [Middle] Last" — split on spaces
        tokens = name.split()
        if len(tokens) == 1:
            return tokens[0]
        last = tokens[-1]
        given = " ".join(tokens[:-1])

    # Build initials from given name(s)
    initials = "".join(
        p[0].upper() + "."
        for p in re.split(r"[\s\-]+", given)
        if p and p[0].isalpha()
    )
    return f"{last}, {initials}" if initials else last


def format_authors_apa(authors: list[dict], max_authors: int = 3) -> str:
    """
    Render an author list in APA style.
      1 author  → "Last, F."
      2-3       → "Last, F., & Last2, F."
      4+        → "Last, F., et al."
    """
    if not authors:
        return ""
    formatted = [_apa_author(a) for a in authors[:max_authors + 1] if a.get("name")]
    formatted = [f for f in formatted if f]
    if not formatted:
        return ""

    if len(authors) > max_authors:
        return formatted[0] + ", et al."
    if len(formatted) == 1:
        return formatted[0]
    return ", ".join(formatted[:-1]) + ", & " + formatted[-1]


# ── Context formatting ─────────────────────────────────────────────────────────

def _apa_citation(r: SearchResult, number: int) -> str:
    """
    Format one result as an APA-style citation line:
      [N] Last, F., et al. (year). Title. Journal. doi:...
    """
    author_str = format_authors_apa(r.authors)
    year_str   = f"({r.year})" if r.year else "(n.d.)"
    title      = r.title or "(untitled)"
    # Truncate very long titles for display
    if len(title) > 120:
        title = title[:117] + "..."

    parts = [f"[{number}]"]
    if author_str:
        parts.append(author_str)
    parts.append(year_str + ".")
    parts.append(title + ".")
    if r.journal:
        parts.append(r.journal + ".")
    if r.doi:
        parts.append(f"https://doi.org/{r.doi}")

    return " ".join(parts)


def _norm_title(title: str | None) -> str:
    """Normalise a title for deduplication (lowercase, collapse whitespace, strip punctuation)."""
    if not title:
        return ""
    return re.sub(r'[^a-z0-9]', '', title.lower())


def _dedup(results: list[SearchResult]) -> list[SearchResult]:
    """
    Remove duplicate results, keeping the highest-ranked occurrence.
    Deduplicates by document_id first, then by normalised title
    (catches the same paper ingested twice from different files).
    """
    seen_docs: set[str] = set()
    seen_titles: set[str] = set()
    out = []
    for r in results:
        nt = _norm_title(r.title)
        if r.document_id in seen_docs:
            continue
        if nt and nt in seen_titles:
            continue
        seen_docs.add(r.document_id)
        if nt:
            seen_titles.add(nt)
        out.append(r)
    return out


def format_context(results: list[SearchResult], max_chars: int = 24000) -> str:
    """
    Render a numbered list of passages for insertion into a prompt.
    Truncates at max_chars to stay within the model's context window.
    Deduplicates by document_id and normalised title.
    """
    deduped = _dedup(results)

    parts: list[str] = []
    total = 0
    for r in deduped:
        section = r.section_type or "text"
        header = _apa_citation(r, r.rank) + f" [{section}]"
        block = f"{header}\n{r.content}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


def format_sources(results: list[SearchResult]) -> str:
    """APA-style source list for printing after the answer. Deduplicates by document and title."""
    lines = []
    for rank, r in enumerate(_dedup(results), start=1):
        lines.append(_apa_citation(r, rank))
    return "\n".join(lines)


# ── Q&A ───────────────────────────────────────────────────────────────────────

QA_SYSTEM = """\
You are a scientific research assistant with access to a curated library of papers.
Answer the user's question using ONLY the provided context passages.
Each passage is numbered [N] with the paper title, year, and section type.

Rules:
- Cite every factual claim with [N] inline.
- If the context is insufficient, say so clearly — do not guess or fabricate.
- Be concise and precise. Prefer technical accuracy over simplicity.
- If multiple passages contradict each other, note the disagreement."""

QA_USER = """\
Context passages:

{context}

---

Question: {question}"""


def qa(question: str, results: list[SearchResult]) -> tuple[str, str]:
    return QA_SYSTEM, QA_USER.format(
        context=format_context(results),
        question=question,
    )


# ── Synthesis ─────────────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM = """\
You are a scientific research assistant skilled at synthesising findings across papers.
Write a structured synthesis based ONLY on the provided passages.
Cite sources inline as [N]. Do not introduce information not present in the context."""

SYNTHESIS_USER = """\
Passages from the literature:

{context}

---

Write a synthesis on the following topic. Cover key findings, methodological approaches, \
areas of consensus, and open questions where evident from the passages.

Topic: {topic}"""


def synthesis(topic: str, results: list[SearchResult]) -> tuple[str, str]:
    return SYNTHESIS_SYSTEM, SYNTHESIS_USER.format(
        context=format_context(results),
        topic=topic,
    )


# ── Section drafting ──────────────────────────────────────────────────────────

WRITE_SYSTEM = """\
You are a scientific writing assistant. Draft a {section} section in academic style \
based ONLY on the provided literature passages.
Cite sources inline as [N]. Use precise, formal language. \
Do not introduce claims beyond what the passages support."""

WRITE_USER = """\
Literature passages:

{context}

---

Draft a {section} section on the following topic.

Topic: {topic}"""


def write_section(
    section: str,
    topic: str,
    results: list[SearchResult],
) -> tuple[str, str]:
    return (
        WRITE_SYSTEM.format(section=section),
        WRITE_USER.format(
            context=format_context(results),
            section=section,
            topic=topic,
        ),
    )


# ── Fine-tuning Q&A generation ────────────────────────────────────────────────

FINETUNE_QA_SYSTEM = """\
You are a scientific question generator. Given a passage from a scientific paper, \
generate one specific, answerable question whose answer is directly contained in the passage. \
Then provide the answer.

Respond in this exact JSON format:
{"question": "...", "answer": "..."}

The question should be specific enough that it could only be answered from this passage."""

FINETUNE_QA_USER = """\
Paper: {title} ({year})
Section: {section}

Passage:
{content}"""


def finetune_qa(
    title: str | None,
    year: int | None,
    section: str | None,
    content: str,
) -> tuple[str, str]:
    return FINETUNE_QA_SYSTEM, FINETUNE_QA_USER.format(
        title=title or "Unknown",
        year=year or "Unknown",
        section=section or "text",
        content=content[:2000],
    )
