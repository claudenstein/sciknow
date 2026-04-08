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
and their publication years, propose a structured book outline for a scientific \
divulgation book (not a research paper).

Rules:
- Propose 6–12 chapters that logically build the book's narrative
- Each chapter should have a clear title and a 1–2 sentence description
- Order chapters so the argument flows from foundational concepts to conclusions
- For EACH chapter, propose 3–6 sections appropriate to that chapter's content. \
  Use section names that fit a book (NOT paper-style like "methods" or "results"). \
  Good section names: "Historical Context", "Key Evidence", "The Debate", \
  "Mechanisms", "Observations", "Implications", "Current Understanding", \
  "Unresolved Questions", "Future Outlook", etc.
- Respond ONLY with valid JSON in the exact format shown"""

OUTLINE_USER = """\
Book title: {book_title}

Available papers in the collection ({n_papers} total):
{paper_list}

Propose a chapter structure for this book. Return JSON:
{{
  "chapters": [
    {{
      "number": 1,
      "title": "...",
      "description": "...",
      "topic_query": "...",
      "sections": ["Section Name 1", "Section Name 2", "Section Name 3"]
    }},
    ...
  ]
}}

The "topic_query" should be a short search phrase (3–6 words) that retrieves the most \
relevant papers from the collection for that chapter.
The "sections" should be 3–6 section names appropriate for a scientific book chapter \
(NOT paper-style — no "methods", "results", "discussion"). Think of how a popular \
science book structures its chapters."""


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
  "assignments": {{"P1": "Cluster Name", "P2": "Cluster Name", ...}}
}}

IMPORTANT: In the "assignments" dict, use the paper ID (P1, P2, ...) as the key, NOT the title."""


def _strip_latex(title: str) -> str:
    """Remove LaTeX math and commands from a title to prevent JSON escape issues.

    Paper titles often contain ``$\\mathsf{R}$`` or ``\\textbf{}``,
    which produce invalid JSON escapes (``\\m``, ``\\t``, ``\\n``) when
    the LLM echoes them back inside a JSON string.  Stripping before
    prompting is more reliable than post-hoc repair.
    """
    # Remove inline math: $...$ (non-greedy)
    title = re.sub(r'\$[^$]+\$', '', title)
    # Remove common LaTeX commands: \command{...} or \command
    title = re.sub(r'\\(?:mathsf|mathrm|mathbf|textbf|textit|emph|textrm|text)\b\{?([^}]*)\}?', r'\1', title)
    # Remove remaining bare backslashes (e.g. \n, \%, \\)
    title = re.sub(r'\\[a-zA-Z]+', '', title)
    title = title.replace('\\', '')
    # Collapse whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def cluster(papers: list[dict]) -> tuple[str, str]:
    """papers: list of {title, year, doc_id}

    Uses P1, P2, ... IDs so the LLM returns IDs (not titles) in the
    assignments dict — avoids fuzzy title matching failures.
    """
    lines = []
    for i, p in enumerate(papers[:200], 1):
        yr = f" ({p['year']})" if p.get("year") else ""
        clean_title = _strip_latex(p['title'])
        lines.append(f"P{i}: {clean_title}{yr}")
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


# ── Book plan generation ─────────────────────────────────────────────────────

PLAN_SYSTEM = """\
You are a scientific book editor. Write a concise book plan (200–500 words) that serves \
as the thesis statement and scope document for a scientific book.

The plan should define:
1. Central argument or thesis of the book
2. Scope: what is covered and what is explicitly excluded
3. Intended audience
4. How the evidence will be structured to build the argument
5. Key terms and definitions that will be used consistently"""

PLAN_USER = """\
Book title: {book_title}
{description_line}

Chapter outline:
{chapter_list}

Available papers ({n_papers} total, showing top 50):
{paper_list}

---

Write the book plan."""


def book_plan(
    book_title: str,
    description: str | None,
    chapters: list[dict],
    papers: list[dict],
) -> tuple[str, str]:
    chapter_list = "\n".join(
        f"  Ch.{c['number']}: {c['title']} — {c.get('description', '')}"
        for c in chapters
    ) or "  (no chapters yet)"
    paper_lines = "\n".join(
        f"  - {p['title']} ({p.get('year', '?')})"
        for p in papers[:50]
    )
    desc = f"\nDescription: {description}" if description else ""
    return PLAN_SYSTEM, PLAN_USER.format(
        book_title=book_title,
        description_line=desc,
        chapter_list=chapter_list,
        n_papers=len(papers),
        paper_list=paper_lines,
    )


# ── Draft summary generation ─────────────────────────────────────────────────

SUMMARY_SYSTEM = """\
Summarise the following draft section in 100–200 words. Focus on: key claims made, \
evidence cited, conclusions drawn. This summary will be used as context when writing \
subsequent chapters to maintain cross-chapter coherence. Do NOT include meta-commentary \
about the writing — only summarise the substantive content."""

SUMMARY_USER = """\
Section type: {section_type}
Chapter: {chapter_title}

Draft content:
{content}"""


def draft_summary(
    section_type: str,
    chapter_title: str,
    content: str,
) -> tuple[str, str]:
    return SUMMARY_SYSTEM, SUMMARY_USER.format(
        section_type=section_type or "text",
        chapter_title=chapter_title or "Unknown",
        content=content[:8000],
    )


# ── Enhanced write_section with cross-chapter context ────────────────────────

WRITE_V2_SYSTEM = """\
You are an expert scientific writer drafting a section of a book. Write in a formal, \
precise academic style appropriate for a scientific monograph.

Rules:
- Base ALL claims on the provided literature passages — cite as [N]
- Never invent references or cite papers not in the provided context
- Use precise, quantitative language; avoid vague generalisations
- Maintain consistency with the book plan and prior chapter summaries below
- Do not repeat content already covered in prior chapters — reference it instead

{book_plan_section}
{prior_summaries_section}"""

WRITE_V2_USER = """\
Literature passages:

{context}

---

Draft a {section} section on the following topic.

Topic: {topic}"""


def write_section_v2(
    section: str,
    topic: str,
    results: list,
    book_plan: str | None = None,
    prior_summaries: list[dict] | None = None,
) -> tuple[str, str]:
    plan_block = ""
    if book_plan:
        plan_block = f"\nBook plan:\n{book_plan}\n"

    summaries_block = ""
    if prior_summaries:
        lines = []
        for s in prior_summaries:
            lines.append(f"Ch.{s.get('chapter_number', '?')} [{s.get('section_type', 'text')}]: {s['summary']}")
        summaries_block = "\nPrior chapter summaries:\n" + "\n".join(lines) + "\n"

    return (
        WRITE_V2_SYSTEM.format(
            book_plan_section=plan_block,
            prior_summaries_section=summaries_block,
        ),
        WRITE_V2_USER.format(
            context=format_context(results),
            section=section,
            topic=topic,
        ),
    )


# ── Review (critic agent) ───────────────────────────────────────────────────

REVIEW_SYSTEM = """\
You are a rigorous scientific peer reviewer evaluating a draft book section. \
Assess the draft on these dimensions and provide structured feedback:

1. **Groundedness**: Is every claim supported by a cited source [N]? Flag any unsupported claims.
2. **Completeness**: Are there obvious gaps — topics the section should cover but doesn't?
3. **Accuracy**: Based on the provided source passages, are any claims misrepresented?
4. **Coherence**: Does the argument flow logically? Are there contradictions?
5. **Redundancy**: Is anything repeated unnecessarily?
6. **Suggestions**: Concrete, actionable improvements (max 5).

Be specific. Quote the problematic text. Reference [N] citations where relevant."""

REVIEW_USER = """\
Section type: {section_type}
Topic: {topic}

Draft to review:
{draft_content}

---

Source passages the draft was based on:
{context}

---

Provide your review."""


def review(
    section_type: str,
    topic: str,
    draft_content: str,
    results: list,
) -> tuple[str, str]:
    return REVIEW_SYSTEM, REVIEW_USER.format(
        section_type=section_type or "text",
        topic=topic or "",
        draft_content=draft_content[:12000],
        context=format_context(results),
    )


# ── Revise (apply instruction to existing draft) ────────────────────────────

REVISE_SYSTEM = """\
You are a scientific writer revising a draft section based on specific feedback. \
Apply the requested change while preserving the rest of the text, its citations, \
and its academic tone. Output the COMPLETE revised section, not just the changed parts."""

REVISE_USER = """\
Current draft:
{draft_content}

---

Revision instruction: {instruction}

---

Additional literature passages (if the revision needs new evidence):
{context}

---

Output the complete revised section."""


def revise(
    draft_content: str,
    instruction: str,
    results: list | None = None,
) -> tuple[str, str]:
    ctx = format_context(results) if results else "(no additional passages)"
    return REVISE_SYSTEM, REVISE_USER.format(
        draft_content=draft_content[:12000],
        instruction=instruction,
        context=ctx,
    )


# ── Claim verification ──────────────────────────────────────────────────────

VERIFY_SYSTEM = """\
You are a fact-checker for scientific text. Given a draft and its source passages, \
check whether each citation [N] in the draft actually supports the claim it's attached to.

For each citation, classify it as:
- SUPPORTED: the claim accurately reflects what the cited passage says
- EXTRAPOLATED: the claim goes beyond what the passage states
- MISREPRESENTED: the claim contradicts or misrepresents the passage
- MISSING: the claim has no citation but should have one

Respond in JSON format:
{{"claims": [
  {{"text": "the claim text", "citation": "[N]", "verdict": "SUPPORTED|EXTRAPOLATED|MISREPRESENTED", "reason": "brief explanation"}},
  ...
],
"unsupported_claims": ["claim text without citation that needs one", ...],
"groundedness_score": 0.85
}}

The groundedness_score is the fraction of cited claims that are SUPPORTED (0.0–1.0)."""

VERIFY_USER = """\
Draft:
{draft_content}

---

Source passages:
{context}

---

Verify the claims."""


def verify_claims(
    draft_content: str,
    results: list,
) -> tuple[str, str]:
    return VERIFY_SYSTEM, VERIFY_USER.format(
        draft_content=draft_content[:12000],
        context=format_context(results),
    )


# ── Sentence planning ───────────────────────────────────────────────────────

SENTENCE_PLAN_SYSTEM = """\
You are a scientific writing planner. Given a section type, topic, and retrieved \
literature passages, create a detailed paragraph-by-paragraph plan for the section.

For each paragraph specify:
- The main point it should make
- Which source(s) [N] it should cite
- How it connects to the previous paragraph

Output as a numbered list. Each entry is one paragraph. Keep it to 5–10 paragraphs. \
This plan will be used by the writer to draft the actual section."""

SENTENCE_PLAN_USER = """\
Section type: {section_type}
Topic: {topic}

Literature passages:
{context}

{plan_context}
---

Create a paragraph-by-paragraph plan for this section."""


# ── Hierarchical tree planning (TreeWriter pattern) ──────────────────────

TREE_PLAN_SYSTEM = """\
You are a scientific writing architect. Given a section type, topic, and \
retrieved literature, create a hierarchical paragraph plan as a JSON tree.

Each node represents a paragraph with:
- "point": the main argument or claim this paragraph makes
- "sources": which [N] citations support it
- "connects_to": how it links to the next paragraph
- "children": optional sub-points (for complex paragraphs)

The tree should flow logically: each paragraph builds on the previous one.

Respond ONLY with valid JSON:
{{
  "section_title": "...",
  "paragraphs": [
    {{
      "point": "Main argument of this paragraph",
      "sources": ["[1]", "[3]"],
      "connects_to": "How this leads to the next paragraph",
      "children": []
    }},
    ...
  ]
}}

Aim for 5-10 paragraphs. Each paragraph should make exactly one clear point."""

TREE_PLAN_USER = """\
Section type: {section_type}
Topic: {topic}

Literature passages:
{context}

{plan_context}
---

Create a hierarchical paragraph plan for this section."""


def tree_plan(
    section_type: str,
    topic: str,
    results: list,
    book_plan: str | None = None,
    prior_summaries: list[dict] | None = None,
) -> tuple[str, str]:
    plan_ctx = ""
    if book_plan:
        plan_ctx += f"\nBook plan:\n{book_plan}\n"
    if prior_summaries:
        lines = [f"Ch.{s.get('chapter_number','?')} [{s.get('section_type','text')}]: {s['summary']}"
                 for s in prior_summaries]
        plan_ctx += "\nPrior chapter summaries:\n" + "\n".join(lines) + "\n"
    return TREE_PLAN_SYSTEM, TREE_PLAN_USER.format(
        section_type=section_type or "text",
        topic=topic or "",
        context=format_context(results),
        plan_context=plan_ctx,
    )


def sentence_plan(
    section_type: str,
    topic: str,
    results: list,
    book_plan: str | None = None,
    prior_summaries: list[dict] | None = None,
) -> tuple[str, str]:
    plan_ctx = ""
    if book_plan:
        plan_ctx += f"\nBook plan:\n{book_plan}\n"
    if prior_summaries:
        lines = [f"Ch.{s.get('chapter_number','?')} [{s.get('section_type','text')}]: {s['summary']}"
                 for s in prior_summaries]
        plan_ctx += "\nPrior chapter summaries:\n" + "\n".join(lines) + "\n"
    return SENTENCE_PLAN_SYSTEM, SENTENCE_PLAN_USER.format(
        section_type=section_type or "text",
        topic=topic or "",
        context=format_context(results),
        plan_context=plan_ctx,
    )


# ── Structured review scoring (for autowrite convergence) ────────────────

SCORE_SYSTEM = """\
You are a scientific peer reviewer scoring a draft section. Evaluate on these \
five dimensions, each scored 0.0–1.0:

1. **groundedness** — What fraction of claims cite a source [N] that actually supports them?
2. **completeness** — Does the section cover all major aspects of the topic given the available evidence?
3. **coherence** — Does the argument flow logically? Are transitions smooth? No contradictions?
4. **citation_accuracy** — Are the [N] references used correctly and not misrepresenting their sources?
5. **overall** — Holistic quality score considering all dimensions.

Also identify the **weakest_dimension** (the one with the lowest score) and provide \
a specific **revision_instruction** (1–2 sentences) that would most improve the draft \
if applied as a targeted revision.

If evidence is missing on a topic, include it in **missing_topics** (list of short phrases \
that could be used as search queries to find relevant papers).

Respond ONLY with valid JSON:
{
  "groundedness": 0.85,
  "completeness": 0.72,
  "coherence": 0.90,
  "citation_accuracy": 0.88,
  "overall": 0.84,
  "weakest_dimension": "completeness",
  "revision_instruction": "Add discussion of proxy calibration methods, citing Smith2020 and Jones2019.",
  "missing_topics": ["proxy calibration uncertainty", "tree ring standardization methods"]
}"""

SCORE_USER = """\
Section type: {section_type}
Topic: {topic}

Draft to score:
{draft_content}

---

Source passages the draft was based on:
{context}

---

Score the draft."""


def score_draft(
    section_type: str,
    topic: str,
    draft_content: str,
    results: list,
) -> tuple[str, str]:
    return SCORE_SYSTEM, SCORE_USER.format(
        section_type=section_type or "text",
        topic=topic or "",
        draft_content=draft_content[:12000],
        context=format_context(results),
    )


# ── Structured gap analysis (JSON) ──────────────────────────────────────────

GAPS_JSON_SYSTEM = """\
You are a scientific book editor. Analyse a book project and return structured \
gaps in JSON format.

Respond ONLY with valid JSON:
{{"gaps": [
  {{"type": "topic|evidence|argument|draft",
    "description": "specific gap description",
    "chapter_number": null or int,
    "priority": "high|medium|low",
    "suggested_action": "concrete action to address this gap"}}
]}}"""

GAPS_JSON_USER = """\
Book: {book_title}

Chapter outline:
{chapter_list}

Papers available ({n_papers} total):
{paper_list}

Existing drafts:
{draft_list}

---

Identify all gaps. Return JSON."""


def gaps_json(
    book_title: str,
    chapters: list[dict],
    papers: list[dict],
    drafts: list[dict],
) -> tuple[str, str]:
    chapter_list = "\n".join(
        f"  Ch.{c['number']}: {c['title']} — {c.get('description', '')}"
        for c in chapters
    ) or "  (no chapters yet)"
    paper_lines = "\n".join(
        f"  - {p['title']} ({p.get('year', '?')})"
        for p in papers[:100]
    )
    draft_lines = "\n".join(
        f"  - [{d['section_type']}] {d['title']} (Ch.{d.get('chapter_number', '?')})"
        for d in drafts
    ) or "  (none yet)"
    return GAPS_JSON_SYSTEM, GAPS_JSON_USER.format(
        book_title=book_title,
        chapter_list=chapter_list,
        n_papers=len(papers),
        paper_list=paper_lines,
        draft_list=draft_lines,
    )
