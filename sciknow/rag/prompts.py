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

    Phase 18 — uses sequential `enumerate(start=1)` numbering on the
    deduped list, IDENTICAL to format_sources(). The writer prompt and
    the sources panel must use the same [N] → paper mapping or
    citations link to the wrong paper (or to nothing at all). Earlier
    versions used `r.rank` here, which is the pre-dedup retrieval
    rank; if dedup removed any duplicate, the prompt's [N] sequence
    had gaps while the sources panel was renumbered sequentially,
    causing every citation past the gap to mismatch.
    """
    deduped = _dedup(results)

    parts: list[str] = []
    total = 0
    for rank, r in enumerate(deduped, start=1):
        section = r.section_type or "text"
        header = _apa_citation(r, rank) + f" [{section}]"
        block = f"{header}\n{r.content}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


def format_sources(results: list[SearchResult]) -> str:
    """APA-style source list for printing after the answer. Deduplicates by
    document and title.

    Numbering MUST stay aligned with format_context() — both use
    `enumerate(_dedup(results), start=1)`. See the Phase 18 note in
    format_context for the bug this prevents.
    """
    lines = []
    for rank, r in enumerate(_dedup(results), start=1):
        lines.append(_apa_citation(r, rank))
    return "\n".join(lines)


# ── RAPTOR cluster summarisation (Sarthi et al., ICLR 2024) ─────────────────
#
# Used by sciknow.ingestion.raptor to compress a cluster of related chunks
# into a single retrievable summary. The summary becomes a level-N node in
# the same Qdrant collection as the leaf chunks; the writer's retriever
# then sees a mix of fine-grained chunks AND mid-level summaries.

RAPTOR_SUMMARY_SYSTEM = """\
You are a scientific knowledge synthesizer. You receive a cluster of {n} text \
excerpts that have been grouped because they are semantically related. Write ONE \
compact synthesis (250-450 words) that captures what these excerpts collectively say.

Rules:
- Open with one sentence stating the central topic of the cluster ("These N \
excerpts cover [topic]." or similar — no preamble).
- Identify the recurring claims, methods, datasets, and findings. Group related \
claims together.
- Where excerpts disagree or qualify each other, note BOTH positions explicitly \
("Some studies argue X, while others find Y under different conditions.").
- Preserve scope qualifiers verbatim — regions ("North Atlantic", "tropical \
Pacific"), time periods ("since 1979", "the satellite era"), datasets ("CMIP6", \
"ERA5"), conditions ("under high-emission scenarios"). Do NOT generalise past \
what the excerpts say.
- Preserve epistemic strength: if the source says "suggests", "may", "is associated \
with", "appears to", use the same words. NEVER upgrade to "proves", "causes", \
"demonstrates".
- Use concrete technical vocabulary, named methods, named datasets, named effects, \
and specific numbers where present. This summary will be embedded and added to a \
search index — keywords matter for retrieval.
- Do NOT use [N] citation markers — the writer who later retrieves this summary \
has no numbering context. Instead, refer to source studies in prose: "Studies of \
ENSO using CMIP6 …", "Recent satellite observations of stratospheric water vapor …", \
"Several palaeoclimate reconstructions …".
- Do NOT include filler like "it is interesting to note that" or "this raises the \
question of". Just report the synthesis.

This summary will be embedded by bge-m3 and stored in the search index alongside \
the original excerpts. When a future writer retrieves it, they will use it as a \
higher-level synthesis that complements the individual chunks."""

RAPTOR_SUMMARY_USER = """\
Cluster of {n} excerpts from the corpus:

{chunks_text}

---

Write the synthesis."""


def raptor_summary(chunks_text: str, n: int) -> tuple[str, str]:
    return (
        RAPTOR_SUMMARY_SYSTEM.format(n=n),
        RAPTOR_SUMMARY_USER.format(n=n, chunks_text=chunks_text),
    )


# ── Step-back query reformulation (Zheng et al., ICLR 2024) ─────────────────

STEP_BACK_SYSTEM = """\
You are a scientific research assistant. Given a concrete query about a specific \
section of a scientific topic, produce a single more abstract / higher-level \
"step-back" question that, if answered, would provide useful background, mechanism, \
or principle context for answering the original question.

Rules:
- Keep the step-back question short (5–12 words).
- It should be ABSTRACT, not a paraphrase. If the original is "ocean heat content \
trends in the North Atlantic since 1990", a good step-back is "mechanisms of ocean \
heat uptake and transport"; a bad step-back is "ocean heat content in the Atlantic".
- Do NOT add citations, formatting, or explanation. Return ONLY the step-back \
question on a single line."""

STEP_BACK_USER = """\
Original query: {query}

Step-back question:"""


def step_back(query: str) -> tuple[str, str]:
    return STEP_BACK_SYSTEM, STEP_BACK_USER.format(query=query)


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

{rejected_block}---

Identify the most important gaps in this book project:

1. **Topic gaps**: important aspects of "{book_title}" not covered by any chapter
2. **Evidence gaps**: chapters with weak paper support (suggest search terms to find more papers)
3. **Argument gaps**: logical steps missing between chapters
4. **Draft gaps**: chapters that have no drafts yet (highest writing priority)

Be specific and actionable.
{rejected_footer}"""


def _format_rejected_block(rejected_ideas: list[str] | None) -> tuple[str, str]:
    """Phase 47.2 — format the rejected-ideas block that goes INTO the
    user prompt + the footer line that reminds the LLM of the gate.

    Returns ``("" , "")`` when there's nothing to inject so the base
    prompt is unchanged for cold-start books. Produced by querying
    ``autowrite_lessons WHERE kind='rejected_idea'`` upstream — see
    ``sciknow.core.book_ops.analyze_gaps_stream``.
    """
    ideas = [str(x).strip() for x in (rejected_ideas or []) if x and str(x).strip()]
    if not ideas:
        return "", ""
    lines = "\n".join(f"  - {t[:240]}" for t in ideas[:10])
    block = (
        f"Previously proposed ideas that were tried and scored poorly "
        f"— DO NOT re-propose these or close variants:\n{lines}\n\n"
    )
    footer = (
        "\nCritically: do not re-surface any of the rejected-ideas "
        "listed above. If a gap you would have proposed is similar to "
        "one of them, propose a DIFFERENT angle or skip it."
    )
    return block, footer


def gaps(
    book_title: str,
    chapters: list[dict],
    papers: list[dict],
    drafts: list[dict],
    *,
    rejected_ideas: list[str] | None = None,
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
    rejected_block, rejected_footer = _format_rejected_block(rejected_ideas)
    return GAPS_SYSTEM, GAPS_USER.format(
        book_title=book_title,
        chapter_list=chapter_list,
        n_papers=len(papers),
        paper_list=paper_lines,
        draft_list=draft_lines,
        rejected_block=rejected_block,
        rejected_footer=rejected_footer,
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

Sentence-level citation grounding (Phase 34 — LongCite pattern):
- Every sentence that makes a factual claim MUST end with at least one [N] \
citation. Do NOT cluster citations at the end of a paragraph — distribute \
them to the specific sentences that draw on each source.
- A sentence that introduces a concept, defines a term, or makes a purely \
logical transition does NOT need a citation. But any sentence containing \
a datum, measurement, model result, empirical finding, or attributed claim \
MUST cite its source inline.
- When multiple sources support the same sentence, cite them all: [1][3][5]. \
When a sentence synthesizes two opposing findings, cite both sides: \
"A suggests X [1], while B finds Y [3]."
- This makes post-hoc groundedness verification sentence-addressable: \
the scorer can check each sentence against its cited passage independently.

Hedging fidelity (CRITICAL — preserves scientific integrity):
- For every claim drawn from a [N] source, transfer the source's epistemic strength \
verbatim at the lexical level. If the source says *suggests*, *indicates*, *is associated \
with*, *may*, *appears to*, write exactly the same strength — never upgrade to *shows*, \
*proves*, *causes*, *demonstrates*, *establishes*.
- If the source qualifies its claim ("in the tropics", "on decadal timescales", \
"under high-emission scenarios", "for the period studied"), carry the qualifier into \
your sentence. Do NOT generalise beyond the source's stated scope.
- Preserve hedges from this cue list when present in the source: may, might, could, \
suggest(s), indicate(s), appear(s), seem(s), likely, possibly, probably, tend to, \
consistent with, associated with, evidence for, in line with, support(s) the view \
that, point(s) to, hint(s) at.
- Do NOT *add* hedges the source did not use; only *preserve* the ones it did.
- Boosters (clearly, undoubtedly, obviously, definitively) should be used at most \
two or three times per section, and only when the source itself is similarly emphatic.

Local coherence (entity bridge):
- The first sentence of each new paragraph must explicitly name at least one entity \
(concept, mechanism, region, dataset, paper, or quantity) that was the grammatical \
subject or most salient noun phrase of the *last* sentence of the previous paragraph. \
No cold-starts.
- If you must genuinely shift topic, open with a short bridge clause that names both \
the prior topic and the new one ("Beyond the ocean heat content discussed above, the \
atmospheric branch …").

{length_target_section}
{style_fingerprint_section}
{lessons_section}
{section_plan_section}
{discourse_relation_section}
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
    paragraph_plan: list[dict] | None = None,
    target_words: int | None = None,
    section_plan: str | None = None,
    lessons: list[str] | None = None,
    style_fingerprint_block: str | None = None,
) -> tuple[str, str]:
    # Phase 32.7 — Layer 1 episodic memory. `lessons` is a list of 1-3
    # sentence concrete reflections distilled from past autowrite runs
    # on similar sections (see core.book_ops._distill_lessons_from_run).
    # Injected as a "Lessons from prior runs" block in the system prompt
    # so the writer is conditioned by what worked / didn't work last time.
    # Empty list = no block (cold-start, fresh book).
    #
    # Phase 47.3 — each lesson can be a STRING (legacy) or a DICT with
    # ``{"text", "kind", "dimension", "scope"}`` (Phase 47.1 consumer
    # with ``return_dicts=True``). When dicts are supplied we group the
    # injection by ``kind`` so the writer handles them differently:
    # a "knowledge" fact is prepended like a reminder, a "rejected_idea"
    # is a hard negative, a "decision" is precedent, an "idea" is
    # a positive suggestion. Same 5-item cap in total (ERL anti-bloat).
    lessons_block = ""
    if lessons:
        capped = lessons[:5]
        # Partition into (kind, text) tuples. Strings default to episode.
        partitioned: dict[str, list[str]] = {}
        for ls in capped:
            if isinstance(ls, dict):
                t = (ls.get("text") or "").strip()
                k = (ls.get("kind") or "episode").strip().lower()
            else:
                t = (ls or "").strip()
                k = "episode"
            if not t:
                continue
            partitioned.setdefault(k, []).append(t)
        if partitioned:
            # Render in a stable, task-meaningful order: what we know →
            # what to reuse → what not to do → what to remember from
            # past decisions → leftover episodes.
            _KIND_ORDER = [
                ("knowledge",
                 "Domain knowledge from prior runs (internalize these):"),
                ("idea",
                 "Positive ideas that worked before (consider reusing):"),
                ("rejected_idea",
                 "Ideas that were tried and scored poorly (do NOT reuse):"),
                ("decision",
                 "Precedent decisions from prior runs on this section:"),
                ("paper",
                 "Papers flagged as especially useful for this section:"),
                ("episode",
                 "General lessons from prior iteration trajectories:"),
            ]
            parts = []
            for kind, header in _KIND_ORDER:
                items = partitioned.get(kind) or []
                if not items:
                    continue
                bullets = "\n".join(f"  - {t}" for t in items)
                parts.append(f"{header}\n{bullets}")
            if parts:
                lessons_block = (
                    "\nLessons from prior runs on similar sections "
                    "(grouped by kind; heed each group's intent):\n\n"
                    + "\n\n".join(parts) + "\n"
                )

    plan_block = ""
    if book_plan:
        plan_block = f"\nBook plan:\n{book_plan}\n"

    # Phase 18 — per-section plan. The user fills this in via the
    # chapter modal's Sections tab; it tells the writer what THIS
    # specific section should cover, scoped narrower than the
    # chapter-level description. Distinct from book_plan (whole-book
    # leitmotiv) and from `topic` (the retrieval query).
    section_plan_block = ""
    if section_plan and section_plan.strip():
        section_plan_block = (
            f"\nSection plan — what THIS section must cover:\n"
            f"{section_plan.strip()}\n"
        )

    summaries_block = ""
    if prior_summaries:
        lines = []
        for s in prior_summaries:
            lines.append(f"Ch.{s.get('chapter_number', '?')} [{s.get('section_type', 'text')}]: {s['summary']}")
        summaries_block = "\nPrior chapter summaries:\n" + "\n".join(lines) + "\n"

    # Optional discourse-relation guidance from a tree plan (PDTB-lite)
    # + Phase 34 CARS rhetorical-move guidance (Swales 1990 + Yang & Allison 2003).
    # If the planner produced a paragraph_plan list with discourse_relation
    # and/or rhetorical_move fields, surface both so the writer knows:
    #   (a) how to open each paragraph (discourse connective)
    #   (b) what rhetorical function the paragraph serves (CARS move)
    discourse_block = ""
    if paragraph_plan:
        relation_lines = []
        for i, p in enumerate(paragraph_plan, 1):
            rel = (p.get("discourse_relation") or "").strip().lower()
            move = (p.get("rhetorical_move") or "").strip().lower()
            point = (p.get("point") or p.get("main_point") or "").strip()
            if (rel or move) and point:
                parts = []
                if rel:
                    parts.append(rel)
                if move:
                    parts.append(f"[{move}]")
                line = f"  Paragraph {i} — {' '.join(parts)}: {point}"
                # Phase 34 — MADAM-RAG-lite: if the planner tagged this
                # paragraph with a contradiction, add explicit pro/con
                # source guidance so the writer presents both sides.
                contr = p.get("contradiction")
                if isinstance(contr, dict) and contr.get("for") and contr.get("against"):
                    for_srcs = ", ".join(str(s) for s in contr["for"])
                    against_srcs = ", ".join(str(s) for s in contr["against"])
                    nature = contr.get("nature") or "opposing findings"
                    line += (
                        f"\n    ⚡ CONTRADICTION: {nature}"
                        f"\n       FOR: {for_srcs}  |  AGAINST: {against_srcs}"
                        f"\n       → Present BOTH sides fairly. Use Toulmin structure"
                        f" if this is a [tension] paragraph."
                    )
                relation_lines.append(line)
        if relation_lines:
            discourse_block = (
                "\nParagraph plan with discourse relations (PDTB-lite) and CARS "
                "rhetorical moves. For each paragraph:\n"
                "  Discourse relation → open with an appropriate connective:\n"
                "    background → 'Historically …', 'Earlier work …'\n"
                "    elaboration → 'More specifically …', 'In detail …'\n"
                "    evidence → 'Supporting this …', 'Direct measurements show …'\n"
                "    contrast → 'However …', 'By contrast …', 'Yet …'\n"
                "    concession → 'Although …', 'While X holds, …'\n"
                "    cause → 'As a result …', 'Because of this …'\n"
                "    comparison → 'Similarly …', 'Like X, Y …'\n"
                "    exemplification → 'For example …', 'A case in point …'\n"
                "    qualification → 'Within these limits …', 'For the period studied …'\n"
                "    synthesis → 'Taken together …', 'These lines of evidence converge on …'\n"
                "  CARS rhetorical move [in brackets] → shapes the paragraph's purpose:\n"
                "    [orient] — define scope, frame concepts, set expectations\n"
                "    [tension] — identify gaps, contradictions, open questions\n"
                "    [evidence] — present specific data, measurements, observations\n"
                "    [qualify] — hedge, state limitations, scope conditions\n"
                "    [integrate] — synthesize into a conclusion connecting to the broader argument\n"
                "  Toulmin scaffold for [tension] paragraphs (Phase 34):\n"
                "    When a paragraph is marked [tension], structure it using the Toulmin\n"
                "    model (Toulmin 1958) to give it genuine argumentative depth:\n"
                "      1. CLAIM — state the contested assertion or open question\n"
                "      2. DATA — present the evidence for and/or against\n"
                "      3. WARRANT — explain the reasoning connecting data to claim\n"
                "      4. QUALIFIER — state the conditions/scope under which the claim holds\n"
                "      5. REBUTTAL — acknowledge the strongest counter-argument\n"
                "    This turns a flat 'the topic is debated' into a structured argument.\n"
                "    Non-tension paragraphs should NOT use the Toulmin scaffold — it would\n"
                "    be forced and unnatural for orient/evidence/qualify/integrate moves.\n\n"
                + "\n".join(relation_lines) + "\n"
            )

    # Phase 17 — length target. The writer needs an explicit word-count
    # anchor or it defaults to the model's idea of "appropriate" length
    # (typically 400-800 words for a "section", which is way too short
    # for a book chapter). With a target it produces drafts close to
    # the requested length on the first try.
    length_block = ""
    if target_words and target_words > 0:
        # Reasonable paragraph count derived from the target: ~150 words
        # per paragraph for academic prose, with a floor of 4 and a
        # ceiling of 20. Lets the model size paragraphs naturally.
        n_paragraphs = max(4, min(20, target_words // 150))
        length_block = (
            f"\nLength target — IMPORTANT:\n"
            f"- Aim for approximately {target_words} words.\n"
            f"- Plan for around {n_paragraphs} paragraphs of substantive content.\n"
            f"- The corpus has plenty of source material; cover the topic "
            f"comprehensively at this length rather than producing a brief summary.\n"
            f"- Pull genuinely new claims and quantitative detail from the "
            f"provided passages — do not pad with filler phrases like \"it is "
            f"worth noting\" or \"this raises the question\".\n"
        )

    return (
        WRITE_V2_SYSTEM.format(
            length_target_section=length_block,
            style_fingerprint_section=(style_fingerprint_block or ""),
            lessons_section=lessons_block,
            section_plan_section=section_plan_block,
            discourse_relation_section=discourse_block,
            book_plan_section=plan_block,
            prior_summaries_section=summaries_block,
        ),
        WRITE_V2_USER.format(
            context=format_context(results),
            section=section,
            topic=topic,
        ),
    )


# ── Lesson distillation (Phase 32.7 — Layer 1 producer side) ──────────

DISTILL_LESSONS_SYSTEM = """\
You are a senior writing coach analyzing an autowrite iteration history \
to extract concrete, actionable lessons that would improve future writing \
of similar sections.

Your job is NOT to praise or summarize. Your job is to identify SPECIFIC, \
TRANSFERABLE patterns from this run that the writer should remember next \
time it drafts a section like this one.

Rules:
- Output 1 to 3 lessons. Three is the absolute maximum.
- Each lesson must be 1-2 sentences. No paragraphs.
- Each lesson must be CONCRETE — refer to a specific dimension (e.g. \
"groundedness") or a specific revision pattern (e.g. "the length \
revision worked when it was paired with adding 2-3 new substantive \
paragraphs"), not vague advice like "write better".
- Each lesson must be TRANSFERABLE — applicable to other sections of \
the same kind, not just this exact draft. Don't reference specific \
claims or papers.
- If the run was uneventful (converged on iteration 1, no improvements \
needed), output an empty list. Don't manufacture lessons that aren't \
there.
- Tag each lesson with a `dimension` field naming which scoring \
dimension it's about: groundedness | completeness | coherence | \
citation_accuracy | hedging_fidelity | length | general.
- Tag each lesson with a `kind` field — what KIND OF THING it is:
  * `knowledge`     — a domain fact / convention / nomenclature to \
remember across runs (e.g. "BP dates are measured from 1950")
  * `idea`          — a framing or sub-topic that worked well and \
should be considered again for similar sections
  * `rejected_idea` — a framing or sub-topic that was tried and \
scored poorly; the gap-finder should NOT re-propose it
  * `decision`      — a pivot / rollback / revise verdict made during \
this run that the writer should internalize
  * `episode`       — default bucket for per-run lessons that don't fit \
the categories above (use sparingly — prefer a specific kind)

Output STRICT JSON in this shape:
{{"lessons": [
  {{"text": "...", "dimension": "...", "kind": "..."}},
  ...
]}}

If there are no lessons worth recording, output: {{"lessons": []}}"""

DISTILL_LESSONS_USER = """\
Section: {section_slug}
Final overall score: {final_overall:.2f}
Score delta (final - first iteration): {score_delta:+.2f}
Iterations used: {iterations_used}
Converged: {converged}

Iteration trajectory (each row is one iteration's pre-revision state):

{iterations_block}

Extract 1-3 transferable lessons from this trajectory, OR output \
{{"lessons": []}} if there's nothing worth recording."""


def distill_lessons(
    section_slug: str,
    final_overall: float,
    score_delta: float,
    iterations_used: int,
    converged: bool,
    iterations: list[dict],
) -> tuple[str, str]:
    """Phase 32.7 — Layer 1 producer prompt. Used by
    core.book_ops._distill_lessons_from_run to extract 1-3 concrete
    lessons from a completed autowrite run's trajectory.

    `iterations` is a list of dicts shaped like:
        {iteration, scores, action, weakest_dimension, revision_instruction,
         word_count, word_count_delta}

    Note: this prompt is intentionally fed to the FAST model
    (settings.llm_fast_model), not the writer/scorer. Per the MAR
    critique (arXiv:2512.20845), letting the same model that
    produced the verdicts also write the reflections leads to
    confirmation bias and repeated reasoning errors.
    """
    rows = []
    for it in iterations:
        scores = it.get("scores") or {}
        line = (
            f"  Iter {it.get('iteration', '?')} — "
            f"overall={scores.get('overall', 0):.2f} "
            f"weakest={(it.get('weakest_dimension') or scores.get('weakest_dimension') or 'unknown')} "
            f"action={it.get('action') or 'KEEP'} "
            f"words={it.get('word_count', 0)}"
            + (f" (Δ{it.get('word_count_delta'):+d})" if it.get('word_count_delta') is not None else "")
        )
        instr = it.get("revision_instruction") or scores.get("revision_instruction")
        if instr:
            # Truncate long revision instructions to keep the prompt short
            line += f"\n    revision: {str(instr)[:240]}"
        rows.append(line)
    iterations_block = "\n".join(rows) if rows else "  (no iterations recorded)"

    return (
        DISTILL_LESSONS_SYSTEM,
        DISTILL_LESSONS_USER.format(
            section_slug=section_slug,
            final_overall=float(final_overall or 0.0),
            score_delta=float(score_delta or 0.0),
            iterations_used=iterations_used,
            converged="yes" if converged else "no",
            iterations_block=iterations_block,
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
check whether each citation [N] in the draft actually supports the claim it's attached to, \
AND whether the claim's epistemic strength matches the source.

For each citation, classify it as:
- SUPPORTED: the claim accurately reflects what the cited passage says, at the same \
epistemic strength
- EXTRAPOLATED: the claim goes beyond what the passage states (additional facts, \
extended scope, generalisation past the source's scope conditions)
- OVERSTATED: the underlying fact IS in the source, but the draft has *strengthened* \
the epistemic modality. Examples: source says "suggests a link", draft says "proves a \
link"; source says "is associated with", draft says "causes"; source says "may", draft \
says "does". The fact is right; the certainty is wrong.
- MISREPRESENTED: the claim contradicts or fundamentally misrepresents the passage
- MISSING: the claim has no citation but should have one

Respond in JSON format:
{{"claims": [
  {{"text": "the claim text", "citation": "[N]", "verdict": "SUPPORTED|EXTRAPOLATED|OVERSTATED|MISREPRESENTED", "reason": "brief explanation"}},
  ...
],
"unsupported_claims": ["claim text without citation that needs one", ...],
"groundedness_score": 0.85,
"hedging_fidelity_score": 0.92
}}

The groundedness_score is the fraction of cited claims that are SUPPORTED or OVERSTATED \
(i.e., factually present in the source) — out of 0.0–1.0.
The hedging_fidelity_score is the fraction of cited claims that are SUPPORTED (i.e., \
the modality is also right) — out of 0.0–1.0. A draft can be 1.0 grounded but 0.6 \
hedging-faithful if it consistently strengthens *suggests* into *proves*."""

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


# ── Chain-of-Verification (Dhuliawala et al., Findings of ACL 2024) ─────────
#
# CoVe decouples fact-checking from drafting: the answerer for each
# verification question never sees the draft, only the source passages.
# This breaks the anchoring bias where a single-window verifier rubber-
# stamps claims because it's reading the draft and the evidence in the
# same context.

COVE_QUESTIONS_SYSTEM = """\
You are a fact-checker preparing verification questions for a draft scientific \
section. You see ONLY the draft, not the sources. Your job is to identify the \
most falsifiable factual claims in the draft and turn each into a precise \
question whose answer would either confirm or refute the claim.

Rules:
- Pick 5-8 claims that are FALSIFIABLE: specific numbers, mechanisms, dates, \
attributions, named effects, scope conditions, magnitudes, causal directions.
- Skip vague or aesthetic claims ("the issue is complex", "researchers have \
studied this for decades").
- Each question should be answerable by a yes/no, a number, a date, or a short \
factual phrase — NOT an essay.
- For each question, quote the EXACT draft sentence the claim comes from.
- Note the citation marker [N] from the draft, if any.
- Bias toward claims that look CONFIDENT or use STRONG modal verbs (proves, \
demonstrates, causes, establishes) — those are the highest-risk overstatements.

Respond ONLY with valid JSON:
{
  "questions": [
    {
      "question": "What equilibrium climate sensitivity range does CMIP6 report in [3]?",
      "draft_claim": "CMIP6 models report an equilibrium climate sensitivity range of 2.5-4.0 K [3].",
      "citation": "[3]"
    },
    ...
  ]
}"""

COVE_QUESTIONS_USER = """\
Draft section:
{draft_content}

---

Generate verification questions."""


def cove_questions(draft_content: str) -> tuple[str, str]:
    return COVE_QUESTIONS_SYSTEM, COVE_QUESTIONS_USER.format(
        draft_content=draft_content[:12000],
    )


COVE_ANSWER_SYSTEM = """\
You are a careful scientific researcher answering a single factual question \
STRICTLY from the provided source passages. You have NOT seen the draft that \
motivated this question — answer based only on what the passages say.

Rules:
- If the passages directly answer the question, give the precise answer with \
the citation marker [N] from the passages.
- If the passages contain partial information, give the partial answer and \
note what is missing.
- If the passages do not address the question at all, set verdict to \
NOT_IN_SOURCES and answer "Not addressed in the provided passages".
- If the passages address the question but at a DIFFERENT scope, period, \
or with different qualifiers than the question implies (e.g. question asks \
about "global" but the source only studies "the North Atlantic"), set verdict \
to DIFFERENT_SCOPE and give the source's answer faithfully along with the \
scope difference.
- Match the source's epistemic strength — do NOT strengthen *suggests* into \
*proves*, *associated with* into *causes*, or *may* into *does*. If the \
source hedges, your answer must hedge.

Respond ONLY with valid JSON:
{
  "answer": "...",
  "verdict": "CONFIRMED|PARTIAL|NOT_IN_SOURCES|DIFFERENT_SCOPE",
  "citation": "[N]" or null,
  "notes": "any caveats, scope differences, or epistemic-strength notes"
}"""

COVE_ANSWER_USER = """\
Source passages:

{context}

---

Question: {question}"""


def cove_answer(question: str, results: list) -> tuple[str, str]:
    return COVE_ANSWER_SYSTEM, COVE_ANSWER_USER.format(
        context=format_context(results),
        question=question,
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
- "discourse_relation": how this paragraph relates to the PREVIOUS one — pick \
exactly one from this fixed PDTB-lite vocabulary:
    * background — sets up context, history, definitions
    * elaboration — adds more detail to a point already made
    * evidence — provides supporting data or measurements for the previous claim
    * contrast — presents an opposing view or counter-evidence
    * concession — acknowledges a limitation or qualification
    * cause — explains a mechanism or reason
    * comparison — draws a parallel with another case
    * exemplification — gives a concrete example
    * qualification — narrows scope or adds caveats
    * synthesis — integrates multiple prior threads (typically near the end)
  The FIRST paragraph's discourse_relation should be "background".
- "rhetorical_move": the CARS-adapted chapter move this paragraph serves — \
pick exactly one from this 5-move vocabulary (Swales 1990, Yang & Allison 2003):
    * orient — establish the terrain: define key concepts, frame the scope, \
set up the reader's expectations for what this section covers
    * tension — identify gaps, contradictions, open questions, or unresolved \
debates in the literature. This is where the section becomes argumentative.
    * evidence — present and evaluate the empirical evidence. Cite specific \
data, measurements, model results, or observations.
    * qualify — hedge, scope qualifications, limitations, conditions under \
which findings hold. Mirror the sources' epistemic strength.
    * integrate — synthesize the preceding points into a coherent conclusion \
that connects to the broader chapter/book argument.
  Guidelines: a well-structured section has AT LEAST one "orient" paragraph \
(typically first), at least one "evidence" paragraph (the core), and \
ideally one "integrate" paragraph (near the end). "tension" and "qualify" \
add argumentative texture — use them when the literature genuinely has \
gaps or caveats, not as filler. Do NOT make every paragraph "evidence".
- "connects_to": one short sentence on how it leads to the next paragraph
- "contradiction": (optional, Phase 34 MADAM-RAG-lite) — when the paragraph's \
point involves OPPOSING findings across sources, include this object:
    {{
      "for": ["[1]", "[3]"],       // sources supporting the claim
      "against": ["[4]", "[7]"],   // sources opposing or complicating it
      "nature": "..."              // one-sentence description of the disagreement
    }}
  Set to null for paragraphs that don't involve a genuine contradiction. This \
field is read by the writer prompt, which renders explicit pro/con guidance so \
the paragraph presents both sides fairly (inspired by MADAM-RAG, Wang et al. \
COLM 2025).
- "children": optional sub-points (for complex paragraphs)

The tree should flow logically: each paragraph builds on the previous one. Do NOT \
make every paragraph "elaboration" — vary the discourse relations to give the section \
real argumentative texture (some contrast, some concession, some synthesis).

Respond ONLY with valid JSON:
{{
  "section_title": "...",
  "paragraphs": [
    {{
      "point": "Main argument of this paragraph",
      "sources": ["[1]", "[3]"],
      "discourse_relation": "background",
      "rhetorical_move": "orient",
      "contradiction": null,
      "connects_to": "How this leads to the next paragraph",
      "children": []
    }},
    ...
  ]
}}

{length_target_section}Each paragraph should make exactly one clear point."""

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
    target_words: int | None = None,
    section_plan: str | None = None,
) -> tuple[str, str]:
    plan_ctx = ""
    if book_plan:
        plan_ctx += f"\nBook plan:\n{book_plan}\n"
    # Phase 18 — per-section plan. Sits between the book plan and the
    # prior chapter summaries so the planner has both the global
    # leitmotiv and the local "this section must cover X" instruction.
    if section_plan and section_plan.strip():
        plan_ctx += (
            f"\nSection plan — what THIS section must cover:\n"
            f"{section_plan.strip()}\n"
        )
    if prior_summaries:
        lines = [f"Ch.{s.get('chapter_number','?')} [{s.get('section_type','text')}]: {s['summary']}"
                 for s in prior_summaries]
        plan_ctx += "\nPrior chapter summaries:\n" + "\n".join(lines) + "\n"
    # Phase 17 — derive paragraph count from length target so the plan
    # has the right shape for the writer to fill at the right size.
    if target_words and target_words > 0:
        n_paragraphs = max(4, min(20, target_words // 150))
        length_block = (
            f"Aim for around {n_paragraphs} paragraphs to fill an "
            f"approximately {target_words}-word section. "
        )
    else:
        length_block = "Aim for 5-10 paragraphs. "
    return TREE_PLAN_SYSTEM.format(length_target_section=length_block), TREE_PLAN_USER.format(
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
seven dimensions, each scored 0.0–1.0:

1. **groundedness** — What fraction of factual-claim sentences have an inline [N] citation \
that actually supports the claim? (Phase 34 — sentence-level: check each sentence individually, \
not just paragraph-level citation presence. A paragraph with 4 claim sentences and 1 citation \
at the end has groundedness ~0.25, not 1.0.)
2. **completeness** — Does the section cover all major aspects of the topic given the available evidence?
3. **coherence** — Does the argument flow logically? Are transitions smooth? No contradictions? \
Does each new paragraph open by naming an entity from the previous paragraph (no cold-starts)?
4. **citation_accuracy** — Are the [N] references used correctly and not misrepresenting their sources?
5. **hedging_fidelity** — Does the draft preserve the source's epistemic strength? \
Penalise upgrades from *suggests* → *proves*, *associated with* → *causes*, *may* → *does*. \
Penalise dropped scope qualifiers ("in the tropics", "on decadal timescales"). \
Reward drafts that carry hedges and qualifiers verbatim from sources. This is the lexical-level \
counterpart to groundedness.
6. **length** — Is the draft close to its target word count? The orchestrator computes this \
mechanically as min(1.0, actual_words / target_words) and injects it into your scores — \
always return 1.0 here; the orchestrator will overwrite it. Do NOT penalise short drafts \
under any other dimension: brevity is handled exclusively by this length score.
7. **overall** — Holistic quality score considering all dimensions.

Also identify the **weakest_dimension** (the one with the lowest score) and provide \
a specific **revision_instruction** (1–2 sentences) that would most improve the draft \
if applied as a targeted revision. If length is the weakest dimension, the instruction \
should tell the writer to expand with additional substantive paragraphs pulling new \
claims and quantitative detail from the provided passages — not to pad with filler.

If evidence is missing on a topic, include it in **missing_topics** (list of short phrases \
that could be used as search queries to find relevant papers).

Respond ONLY with valid JSON:
{
  "groundedness": 0.85,
  "completeness": 0.72,
  "coherence": 0.90,
  "citation_accuracy": 0.88,
  "hedging_fidelity": 0.78,
  "length": 1.0,
  "overall": 0.83,
  "weakest_dimension": "hedging_fidelity",
  "revision_instruction": "Soften three overstated claims: change 'proves' to 'suggests' for [2], restore the 'in the North Atlantic' qualifier on the AMOC claim citing [4], and replace 'causes' with 'is associated with' for [7].",
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

{rejected_block}---

Identify all gaps. Return JSON.{rejected_footer}"""


def gaps_json(
    book_title: str,
    chapters: list[dict],
    papers: list[dict],
    drafts: list[dict],
    *,
    rejected_ideas: list[str] | None = None,
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
    rejected_block, rejected_footer = _format_rejected_block(rejected_ideas)
    return GAPS_JSON_SYSTEM, GAPS_JSON_USER.format(
        book_title=book_title,
        chapter_list=chapter_list,
        n_papers=len(papers),
        paper_list=paper_lines,
        draft_list=draft_lines,
        rejected_block=rejected_block,
        rejected_footer=rejected_footer,
    )


# ── Phase 46 — Two-stage citation insertion (AI-Scientist pattern) ────
#
# Split the "where does a citation belong?" decision from the "what goes
# there?" decision. Pass 1 sees only the draft + the canonical section
# taxonomy and emits {location, claim, query} records. Pass 2 sees each
# location + the top-K hybrid-search candidates and picks (or rejects).
#
# The gain over one-shot write-with-citations:
# - Placement becomes auditable (each [N] has an explicit rationale)
# - Retrieval is budget-bounded per claim, not dependent on whatever
#   happened to be in the writer's context window
# - "No good citation available" is an explicit output, so hallucinated
#   [N] markers drop sharply


CITATION_NEEDS_SYSTEM = """\
You are a citation editor for a scientific manuscript. You see only the \
draft text — no sources. Your job: identify the passages where an inline \
citation is needed (missing) or where the existing citation looks weak \
(e.g. a sweeping claim propped up by one reference), and emit a precise \
search query that a paper-retrieval system could use to find supporting \
evidence.

Rules:
- Focus on FALSIFIABLE factual claims: specific numbers, mechanisms, \
attributions, named effects, scope conditions, causal directions, \
historical statements, named benchmarks or datasets.
- Skip aesthetic / opinion / scene-setting sentences.
- Skip claims that ALREADY carry a citation unless the claim looks \
under-supported ("the literature shows [3]" with just one [3] is a red \
flag).
- The query should be 4–12 words, technical, concrete. Think "what you \
would type into Google Scholar." Not a question — a noun phrase.
- The location must be a VERBATIM substring of the draft (a sentence or \
clause). Quote it exactly, including punctuation. Do not paraphrase.
- If the section has no claims that warrant a citation, return an empty \
list. Do NOT pad.

Respond ONLY with valid JSON in this shape:

{
  "needs": [
    {
      "location": "Global surface temperature has risen by about 1.1 °C since 1850.",
      "claim": "magnitude of warming since preindustrial baseline",
      "query": "preindustrial global mean surface temperature change 1850 present",
      "reason": "specific number with no citation",
      "existing_citations": []
    },
    {
      "location": "Aerosol forcing remains the largest source of uncertainty in climate projections [3].",
      "claim": "aerosol forcing dominates projection uncertainty",
      "query": "anthropogenic aerosol radiative forcing uncertainty climate projections",
      "reason": "sweeping claim supported by only one reference",
      "existing_citations": ["[3]"]
    }
  ]
}"""


CITATION_NEEDS_USER = """\
Section: {section_type} — {section_title}

Draft:
---
{draft_content}
---

Identify places that need (additional) citations. Return JSON only."""


def citation_needs(
    section_type: str,
    section_title: str,
    draft_content: str,
) -> tuple[str, str]:
    return CITATION_NEEDS_SYSTEM, CITATION_NEEDS_USER.format(
        section_type=section_type or "unknown",
        section_title=section_title or "",
        draft_content=draft_content[:12000],
    )


CITATION_CHOOSE_SYSTEM = """\
You are selecting a citation for a single claim in a scientific draft. \
You see the claim + the top-K hybrid-search candidates from the author's \
paper corpus. Your job: pick AT MOST TWO candidates that directly \
support the claim, or explicitly return NONE if no candidate is a good fit.

Rules:
- Prefer DIRECT support over tangential support. A paper that measures \
the claimed quantity beats a paper that mentions it in passing.
- Prefer PRIMARY sources over reviews unless the claim is itself \
"the literature agrees that…" — then a review is appropriate.
- Prefer RECENT papers over older ones when the claim concerns current \
understanding (post-2015 for contemporary work); prefer ORIGINAL papers \
when the claim is historical.
- If none of the candidates is a good fit, return "verdict": "NONE" with \
a short reason. DO NOT force a citation.
- Quality > quantity. One excellent citation beats three mediocre ones.

Respond ONLY with valid JSON:

{
  "verdict": "CITE" | "NONE",
  "chosen": [
    {"candidate_index": 2, "confidence": 0.9, "why": "reports the exact number"},
    {"candidate_index": 5, "confidence": 0.6, "why": "provides the mechanism"}
  ],
  "reason": "plain-English explanation (1 sentence)"
}"""


CITATION_CHOOSE_USER = """\
Claim: {claim}
Draft context (the sentence where the citation goes):
  "{location}"

Top {k} candidates from the corpus:
{candidates}

Pick up to 2 that directly support the claim, or return NONE.
Return JSON only."""


def citation_choose(
    claim: str,
    location: str,
    candidates: list,
) -> tuple[str, str]:
    # Format candidates like the retrieval context but trimmed —
    # one numbered block per candidate so the LLM can refer by index.
    blocks = []
    for i, c in enumerate(candidates):
        title  = (c.get("title")   if isinstance(c, dict) else getattr(c, "title",   "")) or ""
        year   = (c.get("year")    if isinstance(c, dict) else getattr(c, "year",    "")) or ""
        section = (c.get("section") if isinstance(c, dict) else getattr(c, "section_type", "")) or ""
        preview = (c.get("preview") if isinstance(c, dict) else getattr(c, "content_preview", "")) or ""
        blocks.append(
            f"[{i}] ({year}, {section}) {title}\n    {preview[:350]}"
        )
    return CITATION_CHOOSE_SYSTEM, CITATION_CHOOSE_USER.format(
        claim=claim,
        location=location,
        k=len(candidates),
        candidates="\n".join(blocks) or "(no candidates)",
    )


# ── Phase 46.C — Ensemble NeurIPS-rubric review + meta-reviewer ────────
#
# Replaces (or augments) the single-pass `review_draft_stream` with N
# independent reviewers at temperature 0.75 (NeurIPS paper-review
# convention) + a meta-reviewer that fuses their numeric rubric scores
# and synthesises the free-text fields.
#
# The rubric is the NeurIPS 2024 reviewer form (sections, ranges, and
# field names preserved verbatim where possible) so a sciknow draft
# can be judged on the same yardstick as a real submission.
#
# Positivity-bias mitigation (documented in AI-Scientist v1 paper §4):
# half the reviewers run a pessimistic variant of the system prompt
# ("if unsure, recommend reject"). The meta-reviewer sees both. This
# matches `reviewer_system_prompt_neg` in AI-Scientist/perform_review.py.


REVIEW_NEURIPS_BASE = """\
You are a reviewer for a top-tier machine learning / applied science \
conference. You are reading ONE section of a larger work-in-progress \
manuscript. Your job is to produce a structured review using the \
conference's standard form, based ONLY on the draft text and the \
retrieved source passages the author used. Be specific, cite evidence \
from the draft, and never invent detail.

Score each dimension on its documented scale. Err toward harshness for \
vague claims, uncited specific numbers, and hedging that doesn't match \
the source's confidence. Err toward generosity only for sections that \
are structurally sound and evidentially grounded — a good draft in a \
weak genre (e.g. a summary-heavy section) is still a good draft.

Respond ONLY with valid JSON matching this schema:

{
  "summary": "3-5 sentences summarising what the section argues and how.",
  "strengths": ["…", "…"],
  "weaknesses": ["…", "…"],
  "questions": ["…", "…"],
  "limitations": ["…", "…"],
  "ethical_concerns": false,
  "soundness": 3,
  "presentation": 3,
  "contribution": 3,
  "overall": 6,
  "confidence": 3,
  "decision": "weak_accept",
  "rationale": "1-2 sentences tying the numeric scores to the text."
}

Scoring scales (NeurIPS 2024 form):
  soundness:     1 (poor) | 2 (fair) | 3 (good) | 4 (excellent)
  presentation:  1 (poor) | 2 (fair) | 3 (good) | 4 (excellent)
  contribution:  1 (poor) | 2 (fair) | 3 (good) | 4 (excellent)
  overall:       1-10   (10 = top 2%)
  confidence:    1-5    (5 = absolutely certain)
  decision:      "strong_reject" | "reject" | "weak_reject" |
                 "borderline" | "weak_accept" | "accept" | "strong_accept"
"""


REVIEW_NEURIPS_PESSIMISTIC = REVIEW_NEURIPS_BASE + (
    "\n\nIMPORTANT — you are the SKEPTICAL reviewer on this panel. If "
    "you are unsure about a claim, RECOMMEND REJECT. Your job is to "
    "find every hole. Do not soften scores to be polite. AI-Scientist "
    "and follow-up work document a systematic positivity bias in "
    "LLM reviewers; your role is the counterweight."
)


REVIEW_NEURIPS_OPTIMISTIC = REVIEW_NEURIPS_BASE + (
    "\n\nIMPORTANT — you are the GENEROUS reviewer on this panel. "
    "Reward sections that are structurally sound, well-grounded, and "
    "clearly written, even if narrow in scope. A focused contribution "
    "beats a sprawling one. Do not reject on stylistic grounds alone."
)


REVIEW_NEURIPS_USER = """\
Section type: {section_type}
Topic: {topic}

Draft:
---
{draft_content}
---

Source passages the author used:
{context}

---

Return your structured review as JSON only."""


def review_neurips(
    section_type: str,
    topic: str,
    draft_content: str,
    results: list,
    *,
    stance: str = "neutral",
) -> tuple[str, str]:
    """One reviewer's prompt. ``stance`` ∈ {neutral, pessimistic, optimistic}.

    The caller runs this N times with distinct stances + temperature
    0.75 to get an ensemble. See ``core.book_ops.ensemble_review_stream``.
    """
    if stance == "pessimistic":
        sys = REVIEW_NEURIPS_PESSIMISTIC
    elif stance == "optimistic":
        sys = REVIEW_NEURIPS_OPTIMISTIC
    else:
        sys = REVIEW_NEURIPS_BASE
    return sys, REVIEW_NEURIPS_USER.format(
        section_type=section_type or "text",
        topic=topic or "",
        draft_content=draft_content[:12000],
        context=format_context(results),
    )


REVIEW_META_SYSTEM = """\
You are the meta-reviewer (area chair) fusing N independent reviews of \
one section of a scientific manuscript. You see every reviewer's full \
review as JSON. Your job: produce a single synthesis review that:

- Numeric scores are the MEDIAN of the individual reviewers' scores \
(not the mean — median is robust to one outlier reviewer).
- `decision` is chosen by applying the median overall score to the \
standard NeurIPS cutoffs: <5 = reject family, 5-6 = borderline, \
7 = weak_accept, >=8 = accept family.
- `summary` should capture WHAT the reviewers converged on.
- `strengths` / `weaknesses` / `questions` / `limitations` are the \
UNION of individual reviewers' lists, deduplicated and sorted by how \
many reviewers raised each point (highest-agreement items first).
- `rationale` should call out explicit disagreement: "reviewer 2 is \
the only one who flagged X, but they had low confidence; taking the \
majority view that…"
- `confidence` is the max of the individual reviewer confidences \
(someone in the pool is probably right about what they know well).

Respond with JSON matching the same schema the individual reviewers \
used, plus one extra field: `disagreement` = a float in [0, 1] \
measuring how much the reviewers diverged on `overall` (0 = unanimous, \
1 = wildly split, computed as stdev/range). High disagreement is a \
signal that the draft is borderline."""


REVIEW_META_USER = """\
Section: {section_type} — {topic}

The {n} independent reviews (each a JSON object):

{reviews_json}

---

Produce the fused meta-review as JSON only."""


def review_meta(
    section_type: str,
    topic: str,
    reviews: list[dict],
) -> tuple[str, str]:
    import json as _json
    return REVIEW_META_SYSTEM, REVIEW_META_USER.format(
        section_type=section_type or "text",
        topic=topic or "",
        n=len(reviews),
        reviews_json=_json.dumps(reviews, indent=2)[:24000],
    )
