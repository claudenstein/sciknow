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


# ── Book outline generation ───────────────────────────────────────────────────

OUTLINE_SYSTEM = """\
You are an expert scientific book editor. Given a list of scientific paper titles \
and their publication years, propose a structured book outline.

Rules:
- Propose 6–12 chapters that logically build the book's argument
- Each chapter should have a clear title and a 1–2 sentence description of what it covers
- Order chapters so the argument flows from foundational evidence to conclusions
- Respond ONLY with valid JSON in the exact format shown"""

OUTLINE_USER = """\
Book title: {book_title}

Available papers in the collection ({n_papers} total):
{paper_list}

Propose a chapter structure for this book. Return JSON:
{{
  "chapters": [
    {{"number": 1, "title": "...", "description": "...", "topic_query": "..."}},
    ...
  ]
}}

The "topic_query" should be a short search phrase (3–6 words) that retrieves the most \
relevant papers from the collection for that chapter."""


def outline(book_title: str, papers: list[dict]) -> tuple[str, str]:
    """papers: list of {title, year}"""
    lines = []
    for i, p in enumerate(papers[:150], 1):  # cap at 150 to fit context
        yr = f" ({p['year']})" if p.get("year") else ""
        lines.append(f"{i}. {p['title']}{yr}")
    paper_list = "\n".join(lines)
    return OUTLINE_SYSTEM, OUTLINE_USER.format(
        book_title=book_title,
        n_papers=len(papers),
        paper_list=paper_list,
    )


# ── Topic clustering ──────────────────────────────────────────────────────────

CLUSTER_SYSTEM = """\
You are a scientific librarian. Given a list of paper titles, group them into \
thematic clusters.

Rules:
- Identify 6–14 distinct topic clusters appropriate for this collection
- Assign EVERY paper to exactly one cluster
- Use short, descriptive cluster names (2–4 words, e.g. "Solar Irradiance", "Ocean Cycles")
- Respond ONLY with valid JSON"""

CLUSTER_USER = """\
Papers ({n} total):
{paper_list}

Group these papers into thematic clusters. Return JSON:
{{
  "clusters": ["Cluster Name 1", "Cluster Name 2", ...],
  "assignments": {{"Paper title": "Cluster Name", ...}}
}}"""


def cluster(papers: list[dict]) -> tuple[str, str]:
    """papers: list of {title, year}"""
    lines = []
    for p in papers[:200]:
        yr = f" ({p['year']})" if p.get("year") else ""
        lines.append(f"- {p['title']}{yr}")
    return CLUSTER_SYSTEM, CLUSTER_USER.format(
        n=len(papers),
        paper_list="\n".join(lines),
    )


# ── Argument mapping ──────────────────────────────────────────────────────────

ARGUE_SYSTEM = """\
You are a scientific argument analyst. Given a claim and a set of literature passages, \
map the argument by classifying each source.

For each passage classify it as:
- SUPPORTS: directly supports the claim with evidence or reasoning
- CONTRADICTS: presents evidence or reasoning against the claim
- NEUTRAL: methodological, background, or tangentially related

Then write a structured argument map with:
1. A summary of the evidence FOR the claim
2. A summary of the evidence AGAINST (counterarguments to address)
3. Key methodological papers
4. An overall assessment of how well the literature supports the claim

Cite passages inline as [N]. Be analytically rigorous."""

ARGUE_USER = """\
Claim: {claim}

Literature passages:

{context}

---

Map the argument for this claim. Structure your response as:

## Evidence Supporting the Claim
[passages that support it, with [N] citations]

## Counterarguments & Contradicting Evidence
[passages that contradict or complicate the claim]

## Key Methodological Context
[methods papers needed to evaluate the evidence]

## Assessment
[overall strength of the evidence base for this claim]"""


def argue(claim: str, results: list) -> tuple[str, str]:
    return ARGUE_SYSTEM, ARGUE_USER.format(
        claim=claim,
        context=format_context(results),
    )


# ── Gap analysis ──────────────────────────────────────────────────────────────

GAPS_SYSTEM = """\
You are a scientific book editor reviewing a work in progress. Given a book's chapter \
outline and the available literature, identify gaps that should be addressed."""

GAPS_USER = """\
Book: {book_title}

Chapter outline:
{chapter_list}

Papers available in the collection ({n_papers} total):
{paper_list}

Existing draft sections:
{draft_list}

---

Identify the most important gaps in this book project:

1. **Topic gaps**: important aspects of "{book_title}" not covered by any chapter
2. **Evidence gaps**: chapters with weak paper support (suggest search terms to find more papers)
3. **Argument gaps**: logical steps missing between chapters
4. **Draft gaps**: chapters that have no drafts yet (highest writing priority)

Be specific and actionable."""


def gaps(
    book_title: str,
    chapters: list[dict],
    papers: list[dict],
    drafts: list[dict],
) -> tuple[str, str]:
    chapter_list = "\n".join(
        f"  Ch.{c['number']}: {c['title']} — {c.get('description', '')}"
        for c in chapters
    )
    paper_lines = "\n".join(
        f"  - {p['title']} ({p.get('year', '?')})"
        for p in papers[:100]
    )
    draft_lines = "\n".join(
        f"  - [{d['section_type']}] {d['title']} (Ch.{d.get('chapter_number', '?')})"
        for d in drafts
    ) or "  (none yet)"
    return GAPS_SYSTEM, GAPS_USER.format(
        book_title=book_title,
        chapter_list=chapter_list,
        n_papers=len(papers),
        paper_list=paper_lines,
        draft_list=draft_lines,
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
