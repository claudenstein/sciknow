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
You are a scientific knowledge extractor. Output VALID JSON only.

CRITICAL TYPE CONSTRAINTS — read these twice:

WRONG — do not do this:
{"concepts": [{"name": "climate-change", "description": "..."}]}

RIGHT — concepts is a flat list of slug strings:
{"concepts": ["climate-change", "greenhouse-gas-emissions"]}

Same rule for "methods" and "datasets": arrays of STRING slugs, \
never objects with "name"/"description" keys.

Required output shape (JSON):
{
  "concepts":  [<slug-str>, ...]   // 3-8 items, scientific phenomena / theories / metrics
  "methods":   [<slug-str>, ...]   // 1-4 items, techniques or algorithms
  "datasets":  [<slug-str>, ...]   // 0-3 items, named datasets
  "triples":   [
    {"subject": "<slug>", "predicate": "<snake_verb>", "object": "<slug>", "source_sentence": "<verbatim ≤ 300 chars>"},
    ...   // EXACTLY 6-12 items, not fewer
  ]
}

Slug format: lowercase-hyphenated (e.g. "atmospheric-co2-rise"), \
never Title Case, never spaces.
Predicates: short snake_case verbs like uses_method, studies, finds, \
supports, contradicts, measures, causes, correlates_with, related_to.
source_sentence: verbatim from the paper text, or "" if none. \
NEVER paraphrase or invent a sentence.
Reuse existing concept slugs from the provided list when they match.

Respond ONLY with the JSON object — no preamble, no fences, no commentary.
DO NOT copy placeholder strings like "concept-slug-1" from any example; \
fill in the real values extracted from THIS paper."""

# Phase 54.6.36 — few-shot example moved to system prompt (concrete values,
# not placeholders) so the model sees what good output LOOKS like without
# being tempted to echo fake placeholder strings back.
# Phase 54.6.40 — switched to explicit wrong-vs-right schema enforcement.
# Rationale: qwen2.5:32b-instruct was returning `[{"name": ..., "description":
# ...}]` for concepts/methods/datasets regardless of prose rules in the
# prompt. Adding a 6-triple example did not move the needle (measured
# live: identical output before/after). Showing a WRONG pattern next to
# a RIGHT pattern flipped behavior: 4 concepts as flat slugs, 6 triples
# (up from 3), and 6/6 with source_sentence (up from 0/3). The
# "EXACTLY 6-12 items, not fewer" line is also load-bearing — the model
# was anchoring to the example count and producing one-for-one matches
# with whatever the example showed.

EXTRACT_ENTITIES_EXAMPLE = """\

Example (solar-physics paper titled "Solar cycle 24 anomalies"):
{
  "concepts": ["solar-minimum", "sunspot-number", "geomagnetic-activity", "extended-minimum", "coronal-holes"],
  "methods": ["wavelet-analysis", "cross-correlation", "superposed-epoch-analysis"],
  "datasets": ["sidc-sunspot-catalog", "omni-solar-wind-database"],
  "triples": [
    {"subject": "solar-cycle-24", "predicate": "studies", "object": "sunspot-number", "source_sentence": "We analyzed the sunspot-number record from 2008 to 2019 to characterize the depth of the cycle 24 minimum."},
    {"subject": "cycle-24-minimum", "predicate": "finds", "object": "reduced-geomagnetic-activity", "source_sentence": "The geomagnetic Ap index reached a 100-year low during the extended minimum of 2008-2009."},
    {"subject": "wavelet-analysis", "predicate": "measures", "object": "sunspot-number", "source_sentence": "Morlet wavelet analysis was applied to the daily sunspot-number series to isolate decadal modulations."},
    {"subject": "extended-minimum", "predicate": "correlates_with", "object": "coronal-holes", "source_sentence": "Persistent low-latitude coronal holes accompanied the extended minimum, mediating solar-wind outflow."},
    {"subject": "solar-cycle-24", "predicate": "contradicts", "object": "dynamo-prediction-2007", "source_sentence": "The observed polar-field strength fell well below the 2007 dynamo prediction of a strong cycle."},
    {"subject": "superposed-epoch-analysis", "predicate": "uses_method", "object": "omni-solar-wind-database", "source_sentence": "Superposed-epoch composites were built from OMNI high-resolution solar-wind data."}
  ]
}"""


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

Extract the concepts, methods, datasets, and triples from THIS paper as a single JSON object. Begin immediately with {{ — no preamble, no code fence."""


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
    # Phase 54.6.42 — existing_slugs cap 300 → 100. With 300 slugs
    # the user prompt balloons to ~27 KB and qwen3:30b-a3b-instruct-2507
    # emits 3000+ tokens of verbose output (40+ concepts, long source
    # sentences) that blow past any num_predict cap mid-JSON. 100
    # slugs is enough for the "reuse where applicable" hint without
    # encouraging the model to match every one.
    slug_str = ", ".join(existing_slugs[:100]) if existing_slugs else "(none yet)"
    # Phase 55.2 — rollback. The Phase 55 head+tail cut to 2 KB was
    # too aggressive — for multi-section papers it shaved the middle
    # methodology/results paragraphs that *do* carry triple-extraction
    # signal. Restore a simple linear slice at 8 KB (up from the
    # pre-Phase-55 default of 6 KB) so the model sees enough context
    # to extract a full set of triples. `_head_tail_slice` stays
    # available for callers that explicitly want it.
    # Phase 54.6.40 — EXAMPLE deliberately omitted. Live test showed
    # qwen2.5:32b-instruct regresses to verbose {"name":..., "description":...}
    # objects whenever the long few-shot example is appended, even with
    # the wrong-vs-right schema framing in SYSTEM. Without the EXAMPLE,
    # the same paper yields flat slug strings + 6 triples with source
    # sentences. The inline mini-example inside SYSTEM is retained.
    return (
        EXTRACT_ENTITIES_SYSTEM,
        EXTRACT_ENTITIES_USER.format(
            slug=slug or "unknown",
            title=title or "Untitled", authors=authors or "Unknown",
            year=year or "n.d.", keywords=keywords or "N/A",
            domains=domains or "N/A",
            abstract=(abstract or "N/A")[:2000],
            sections=(sections or "")[:8000],
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

Preferred predicate vocabulary (Phase 54.6.220, roadmap 3.7.2). Emit \
these canonical forms when possible — a downstream canonicaliser \
maps common synonyms to them, but using the canonical form directly \
produces cleaner triples:

  Causal / forcing:       forces, responds_to, correlates_with
  Evidence / claims:      supports, contradicts, neutral_on
  Proxy / reconstruction: proxies_for, reconstructs
  Measurement:            measures, observes
  Methods:                uses_method, applied_to, predicts
  Structural:             part_of, related_to, has_property
  Meta / citations:       cites_data, cites_method
  Paper actions:          studies, finds

Typical triple shapes:
- (Paper, uses_method, Method)       — research methods and techniques used
- (Paper, studies, Phenomenon)       — what the paper investigates
- (Paper, finds, Finding)            — key results or conclusions
- (Paper, supports, Claim)           — claims the paper provides evidence for
- (Paper, contradicts, Claim)        — claims the paper provides counter-evidence for
- (Concept, related_to, Concept)     — conceptual relationships
- (Method, applied_to, Domain)       — method-domain connections
- (Dataset, measures, Variable)      — what datasets track
- (Proxy, proxies_for, Target)       — "tree rings proxies_for temperature"
- (Forcing, forces, Response)        — "solar_irradiance forces surface_temperature"

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


# ── Phase 54.6.26 — Multi-perspective pre-research (from Stanford STORM) ────

PERSPECTIVES_SYSTEM = """\
You are a scientific research panel. Given a paper's metadata, generate \
3 brief expert perspectives that would ask different probing questions \
about this paper. Each perspective is 1-2 sentences identifying what \
angle that expert would focus on and what question they'd ask.

Respond as a numbered list:
1. **[Role]**: [Question]
2. **[Role]**: [Question]
3. **[Role]**: [Question]

Be specific to the paper's actual content — no generic placeholders."""

PERSPECTIVES_USER = """\
Title: {title}
Authors: {authors}
Year: {year}
Abstract: {abstract}

Key sections:
{sections}"""


def wiki_perspectives(
    title: str, authors: str, year: str, abstract: str, sections: str,
) -> tuple[str, str]:
    """Generate multi-perspective expert questions before writing."""
    return (
        PERSPECTIVES_SYSTEM,
        PERSPECTIVES_USER.format(
            title=title or "Untitled", authors=authors or "Unknown",
            year=year or "n.d.",
            abstract=(abstract or "N/A")[:1500],
            sections=(sections or "")[:3000],
        ),
    )


# ── Phase 54.6.26 — Wiki polishing pass (from STORM Article Polishing) ──────

POLISH_SYSTEM = """\
You are a scientific wiki editor performing a final polish pass. Given a \
draft wiki summary page, improve it by:

1. Removing any redundant/repeated sentences or bullet points
2. Smoothing transitions between sections
3. Ensuring [[concept-slug]] links are used for key terms
4. Fixing any formatting inconsistencies
5. Ensuring the page stays under 600 words

Return the polished page in the same Markdown format. Do NOT add new \
factual claims — only clean up the existing content."""

POLISH_USER = """\
{content}

---

Polish this wiki page. Return ONLY the improved Markdown, no commentary."""


def wiki_polish(content: str) -> tuple[str, str]:
    """Polish a wiki page — dedup, transitions, formatting."""
    return (POLISH_SYSTEM, POLISH_USER.format(content=content[:8000]))
