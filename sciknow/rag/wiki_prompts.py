"""
Prompt templates for the wiki knowledge layer.

All templates follow the same (system, user) -> tuple[str, str] pattern
as prompts.py.
"""
from __future__ import annotations


# ── Paper summary ────────────────────────────────────────────────────────────

PAPER_SUMMARY_SYSTEM = """\
You are a scientific knowledge curator. Given a paper's metadata and section \
contents, write a structured wiki summary page in Markdown.

Format:
# {title} ({year})

**Authors:** ...
**Journal:** ...

## Key Findings
- Bullet points of the main results and conclusions.

## Methods
Brief description of the methodology.

## Contributions
What this paper adds to the field.

## Limitations
Acknowledged limitations or caveats.

## Related Concepts
Link related topics using [[concept-slug]] notation.

Rules:
- Be precise and concise — aim for 300–600 words.
- Use [[concept-slug]] links for key concepts, methods, and datasets.
- If a concept slug is provided in the user message, reuse it. Otherwise propose new ones in lowercase-hyphenated form."""

PAPER_SUMMARY_USER = """\
Paper: {title}
Authors: {authors}
Year: {year}
Journal: {journal}
DOI: {doi}
Keywords: {keywords}
Domains: {domains}

Abstract:
{abstract}

Section contents:
{sections}

Existing wiki concepts (reuse these slugs where applicable): {existing_slugs}

---

Write the wiki summary page for this paper."""


def wiki_paper_summary(
    title: str, authors: str, year: str, journal: str, doi: str,
    keywords: str, domains: str, abstract: str, sections: str,
    existing_slugs: list[str],
) -> tuple[str, str]:
    slug_str = ", ".join(existing_slugs[:200]) if existing_slugs else "(none yet)"
    return (
        PAPER_SUMMARY_SYSTEM.format(
            title=title or "Untitled", year=year or "n.d.",
        ),
        PAPER_SUMMARY_USER.format(
            title=title or "Untitled", authors=authors or "Unknown",
            year=year or "n.d.", journal=journal or "Unknown",
            doi=doi or "N/A", keywords=keywords or "N/A",
            domains=domains or "N/A",
            abstract=(abstract or "N/A")[:3000],
            sections=(sections or "")[:12000],
            existing_slugs=slug_str,
        ),
    )


# ── Entity extraction ────────────────────────────────────────────────────────

EXTRACT_ENTITIES_SYSTEM = """\
You are a scientific knowledge extractor. Given a paper's metadata, extract \
structured entities AND knowledge graph triples in a single pass.

Rules:
- Extract 3–8 key concepts (scientific phenomena, theories, metrics)
- Extract 1–4 methods (techniques, models, algorithms)
- Extract 0–3 datasets (named datasets or data sources)
- Extract 5–15 knowledge graph triples (subject, predicate, object, source_sentence)
- For each triple, also capture "source_sentence": the single verbatim
  sentence from the paper text above that evidences the claim. Copy it
  exactly (≤ 300 chars). If no single sentence supports the triple,
  use an empty string — do NOT paraphrase or synthesize.
- Reuse existing concept names provided in the user message where applicable
- Use lowercase-hyphenated slug format for new concepts (e.g. "total-solar-irradiance")
- Respond ONLY with valid JSON."""

EXTRACT_ENTITIES_USER = """\
Paper slug: {slug}
Title: {title}
Authors: {authors}
Year: {year}
Keywords: {keywords}
Domains: {domains}
Abstract: {abstract}

Key sections:
{sections}

Existing wiki concepts (reuse these where applicable): {existing_slugs}

Return JSON:
{{
  "concepts": ["concept-slug-1", "concept-slug-2", ...],
  "methods": ["method-slug-1", ...],
  "datasets": ["dataset-slug-1", ...],
  "triples": [
    {{"subject": "...", "predicate": "uses_method|studies|finds|supports|contradicts|related_to", "object": "...", "source_sentence": "verbatim sentence from the paper that evidences the triple"}},
    ...
  ]
}}"""


def _head_tail_slice(text: str, total_budget: int = 2000) -> str:
    """Phase 55 — when the full `sections` blob exceeds our token
    budget, split the budget between the head (methods / intro
    opening) and the tail (results / conclusion). Covers both the
    claims triple-extraction wants from the top of the paper and
    the findings triple-extraction wants from the bottom. The
    middle is dropped with a visible marker so the model doesn't
    stitch two disjoint passages into one sentence.

    If the text already fits in the budget, it's returned
    unchanged."""
    t = text or ""
    if len(t) <= total_budget:
        return t
    half = total_budget // 2
    head = t[:half]
    tail = t[-half:]
    return f"{head}\n\n[…section body omitted; {len(t) - total_budget} chars…]\n\n{tail}"


def wiki_extract_entities(
    title: str, authors: str, year: str, keywords: str, domains: str,
    abstract: str, existing_slugs: list[str],
    slug: str = "", sections: str = "",
) -> tuple[str, str]:
    slug_str = ", ".join(existing_slugs[:300]) if existing_slugs else "(none yet)"
    # Phase 55 — sections budget shrunk from 6000 → 2000 chars via
    # head+tail slicing. See docs/WIKI_COMPILE_SPEED.md: BioREx §4.2 +
    # LangExtract chunked-extraction guidance both confirm that for
    # triple extraction specifically, abstract + conclusion + first
    # methods paragraph carry ≥90% of the signal. Head+tail (not
    # head-only) preserves methods opening AND findings closure.
    return (
        EXTRACT_ENTITIES_SYSTEM,
        EXTRACT_ENTITIES_USER.format(
            slug=slug or "unknown",
            title=title or "Untitled", authors=authors or "Unknown",
            year=year or "n.d.", keywords=keywords or "N/A",
            domains=domains or "N/A",
            abstract=(abstract or "N/A")[:2000],
            sections=_head_tail_slice(sections or "", total_budget=2000),
            existing_slugs=slug_str,
        ),
    )


# ── Concept page (full write or update) ──────────────────────────────────────

CONCEPT_PAGE_SYSTEM = """\
You are a scientific encyclopedia editor. Write or update a wiki concept page \
that synthesizes what multiple papers say about this concept.

Rules:
- If updating an existing page, preserve and build on existing content. Add \
new information from the new paper seamlessly.
- Use [[concept-slug]] links for related concepts.
- Available concepts: {existing_slugs}
- Cite papers as (Author, Year) and link to paper summaries using [[paper-slug]].
- Structure the page clearly with ## headings.
- Aim for 200–500 words per concept page."""

CONCEPT_PAGE_USER = """\
Concept: {concept_name}

Existing page content:
{existing_content}

New paper contributing to this concept:
Title: {paper_title} ({paper_year})
Relevant passages: {passages}

Other papers already on this page: {source_papers}

---

Write the complete updated concept page."""


def wiki_concept_page(
    concept_name: str, existing_content: str,
    paper_title: str, paper_year: str, passages: str,
    source_papers: str, existing_slugs: list[str],
) -> tuple[str, str]:
    slug_str = ", ".join(existing_slugs[:200]) if existing_slugs else "(none yet)"
    return (
        CONCEPT_PAGE_SYSTEM.format(existing_slugs=slug_str),
        CONCEPT_PAGE_USER.format(
            concept_name=concept_name,
            existing_content=existing_content or "(new page — no existing content)",
            paper_title=paper_title or "Unknown",
            paper_year=paper_year or "n.d.",
            passages=(passages or "")[:8000],
            source_papers=source_papers or "(first paper)",
        ),
    )


# ── Concept page append (fast model, incremental update) ─────────────────────

CONCEPT_APPEND_SYSTEM = """\
You are a scientific wiki editor. Append a brief update (2–4 sentences) to an \
existing concept page based on a newly ingested paper. Do NOT rewrite the page — \
only add what is new."""

CONCEPT_APPEND_USER = """\
Concept: {concept_name}
New paper: {paper_title} ({paper_year})
Relevant passage: {passage}

Write a brief addition (2–4 sentences) describing what this paper adds to our \
understanding of {concept_name}. Start with "**{paper_title} ({paper_year})** "."""


def wiki_concept_append(
    concept_name: str, paper_title: str, paper_year: str, passage: str,
) -> tuple[str, str]:
    return (
        CONCEPT_APPEND_SYSTEM,
        CONCEPT_APPEND_USER.format(
            concept_name=concept_name,
            paper_title=paper_title or "Unknown",
            paper_year=paper_year or "n.d.",
            passage=(passage or "")[:3000],
        ),
    )


# ── Synthesis page ───────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM = """\
You are a scientific research synthesist. Write a high-level overview page \
that compares and contrasts multiple papers on a broad topic.

Structure:
# {topic}: State of Research

## Overview
Brief introduction to the topic and why it matters.

## Key Findings Across Studies
Synthesize the main results, noting agreements and differences.

## Areas of Consensus
What do most papers agree on?

## Open Questions and Disagreements
Where do findings conflict? What remains unresolved?

## Timeline of Key Developments
Chronological progression of understanding.

Rules:
- Cite papers as (Author, Year) with [[paper-slug]] links.
- Link concepts using [[concept-slug]].
- Available concepts: {existing_slugs}
- Be analytical, not just descriptive — synthesize, don't summarize."""

SYNTHESIS_USER = """\
Topic: {topic}

Paper summaries to synthesize:
{paper_summaries}

Related concept pages:
{concept_pages}

---

Write the synthesis page."""


def wiki_synthesis(
    topic: str, paper_summaries: str, concept_pages: str,
    existing_slugs: list[str],
) -> tuple[str, str]:
    slug_str = ", ".join(existing_slugs[:200]) if existing_slugs else "(none yet)"
    return (
        SYNTHESIS_SYSTEM.format(topic=topic, existing_slugs=slug_str),
        SYNTHESIS_USER.format(
            topic=topic,
            paper_summaries=(paper_summaries or "")[:16000],
            concept_pages=(concept_pages or "")[:8000],
        ),
    )


# ── Lint: contradiction detection ────────────────────────────────────────────

LINT_CONTRADICTIONS_SYSTEM = """\
You are a scientific fact-checker. Given a concept and claims from multiple \
papers about it, identify any contradictions, disagreements, or inconsistencies.

For each contradiction found, specify:
- The conflicting claims
- Which papers make each claim
- Whether this is a genuine disagreement or a difference in scope/methodology

Respond ONLY with valid JSON:
{{
  "contradictions": [
    {{
      "claim_a": "...",
      "paper_a": "...",
      "claim_b": "...",
      "paper_b": "...",
      "severity": "high|medium|low",
      "explanation": "..."
    }}
  ],
  "consistent": true|false
}}

If no contradictions are found, return {{"contradictions": [], "consistent": true}}."""

LINT_CONTRADICTIONS_USER = """\
Concept: {concept_name}

Claims from papers:
{claims}

---

Check for contradictions."""


def wiki_lint_contradictions(
    concept_name: str, claims: str,
) -> tuple[str, str]:
    return (
        LINT_CONTRADICTIONS_SYSTEM,
        LINT_CONTRADICTIONS_USER.format(
            concept_name=concept_name,
            claims=(claims or "")[:12000],
        ),
    )


# ── Knowledge graph triple extraction ────────────────────────────────────────

KG_EXTRACT_SYSTEM = """\
You are a scientific knowledge engineer. Given a paper's title, abstract, and \
key sections, extract entity-relationship triples for a knowledge graph.

Triple types to extract:
- (Paper, uses_method, Method) — research methods and techniques used
- (Paper, studies, Phenomenon) — what the paper investigates
- (Paper, finds, Finding) — key results or conclusions
- (Paper, supports, Claim) — claims the paper provides evidence for
- (Paper, contradicts, Claim) — claims the paper provides counter-evidence for
- (Concept, related_to, Concept) — conceptual relationships
- (Method, applied_to, Domain) — method-domain connections
- (Dataset, measures, Variable) — what datasets track

Rules:
- Extract 5-15 triples per paper
- Use normalized entity names (lowercase, consistent naming)
- The paper itself is always referred to by its short slug
- Respond ONLY with valid JSON"""

KG_EXTRACT_USER = """\
Paper slug: {slug}
Title: {title}
Year: {year}
Abstract: {abstract}

Key sections:
{sections}

Return JSON:
{{
  "triples": [
    {{"subject": "...", "predicate": "...", "object": "..."}},
    ...
  ]
}}"""


# ── Consensus mapping ────────────────────────────────────────────────────────

CONSENSUS_SYSTEM = """\
You are a scientific evidence analyst. Given a topic and claims from multiple \
papers, map the consensus landscape:

1. Identify the key claims or positions on this topic
2. For each claim, list which papers support it and which contradict it
3. Note how consensus has shifted over time (if year information is available)
4. Identify the most debated/contested sub-topics

Respond ONLY with valid JSON:
{{
  "topic": "...",
  "claims": [
    {{
      "claim": "description of the claim",
      "supporting_papers": ["paper1 (year)", "paper2 (year)"],
      "contradicting_papers": ["paper3 (year)"],
      "neutral_papers": ["paper4 (year)"],
      "consensus_level": "strong|moderate|weak|contested",
      "trend": "growing|stable|declining|emerging"
    }}
  ],
  "most_debated": ["sub-topic 1", "sub-topic 2"],
  "summary": "brief overall assessment of the state of consensus"
}}"""

CONSENSUS_USER = """\
Topic: {topic}

Knowledge graph triples related to this topic:
{triples}

Paper summaries mentioning this topic:
{summaries}

---

Map the consensus landscape for this topic."""


def wiki_consensus(
    topic: str, triples: str, summaries: str,
) -> tuple[str, str]:
    return (
        CONSENSUS_SYSTEM,
        CONSENSUS_USER.format(
            topic=topic,
            triples=(triples or "")[:8000],
            summaries=(summaries or "")[:12000],
        ),
    )


def kg_extract_triples(
    slug: str, title: str, year: str, abstract: str, sections: str,
) -> tuple[str, str]:
    """Standalone KG extraction (unused — entity+KG merged into wiki_extract_entities)."""
    return (
        KG_EXTRACT_SYSTEM,
        KG_EXTRACT_USER.format(
            slug=slug, title=title or "Untitled",
            year=year or "n.d.",
            abstract=(abstract or "")[:2000],
            sections=(sections or "")[:8000],
        ),
    )
