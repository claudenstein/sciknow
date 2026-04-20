"""Phase 54.6.114 (Tier 2 #2) — question-driven agentic expansion.

A thin orchestrator on top of the Phase 49 RRF ranker that lets the
user hand sciknow a *research question* rather than a seed corpus and
a relevance query. An LLM plans, executes, and re-plans against a
corpus-coverage stopping rule.

Flow:

1. **Decompose** the question into 3-6 sub-topics (LLM_FAST_MODEL).
2. **Coverage** — for each sub-topic, run hybrid search over the
   current corpus and count unique papers. Sub-topics with ≥3
   papers are "covered"; those with 0-2 are "gaps".
3. **Targeted expand** — for each gap sub-topic, run
   ``_run_rrf_ranker`` with that sub-topic as ``--relevance-query``
   and a per-sub-topic budget (default 10). Download + ingest.
4. **Re-check coverage** after ingestion.
5. **Stop** when all sub-topics are covered, OR coverage hasn't
   increased in a full round, OR max_rounds reached.
6. **Re-plan** between rounds — if the LLM's original decomposition
   turned out to be too vague (gaps still empty after a round), give
   it the current coverage snapshot and ask for a refined
   decomposition.

Complements rather than replaces ``db expand``:

* Static ``db expand`` → ranker does all the work, user picks seeds
  manually (or the whole corpus).
* ``db expand --question`` → LLM picks the seeds and stopping rule.

Design constraints:

- Uses the EXISTING Phase 49 RRF ranker — we're not re-ranking, we're
  re-driving. Every downloaded paper still goes through the same
  retraction / predatory / one-timer / MMR / citation-context pipeline.
- Coverage check uses ``hybrid_search`` (dense + sparse + FTS + RRF)
  the corpus already relies on, not a bespoke topic classifier. If
  the hybrid search can answer the sub-topic, it's covered.
- Provenance: every round records which sub-topic it served and why.
  For now logs-only; Tier 4 #1 will persist to ``documents.provenance``.

See ``docs/EXPAND_ENRICH_RESEARCH_2.md`` §3.1.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)


# ── LLM prompts ─────────────────────────────────────────────────────

_DECOMP_SYSTEM = """You are a research-librarian assistant. Given one research
question, break it into 3-6 narrow, orthogonal sub-topics that together cover
the question. Each sub-topic must be a SHORT search-query-like noun phrase
(NOT a full sentence), under 10 words, specific enough that a paper's title
+ abstract can be judged relevant to it in under a minute.

Return ONLY valid JSON with this shape (no prose, no code fences):

  {"subtopics": ["<phrase 1>", "<phrase 2>", ...]}

Avoid overlapping sub-topics. Avoid overly-broad ones ("climate change" is
too broad; "equilibrium climate sensitivity from paleoclimate constraints"
is right). Avoid method labels unless the question names a method."""

_REFINE_SYSTEM = """You previously decomposed a research question into
sub-topics. Corpus coverage for each was measured. Sub-topics with 0-2 papers
are "gaps". Produce a REFINED decomposition that:

1. Keeps sub-topics with good coverage as-is.
2. Splits overly-broad gap sub-topics into narrower ones.
3. Adds NEW sub-topics that might reveal a missed angle.

Return ONLY valid JSON:
  {"subtopics": ["<phrase 1>", ...]}"""


@dataclass
class SubtopicCoverage:
    subtopic: str
    n_papers: int          # distinct document_ids in top hybrid-search hits
    top_score: float       # top RRF score from hybrid_search
    sample_titles: list[str] = field(default_factory=list)


@dataclass
class RoundPlan:
    round_n: int
    subtopics: list[str]
    coverage: dict[str, SubtopicCoverage]
    gap_subtopics: list[str]        # those needing expansion this round
    budget_per_gap: int


# ── Decomposition + refinement ──────────────────────────────────────

def _strip_think_and_fences(raw: str) -> str:
    """Drop <think>...</think> and ```json``` fences so JSON parsers
    see just the object. Shared utility pattern from table_parse + others."""
    txt = raw.strip()
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL | re.IGNORECASE).strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return txt


def decompose_question(
    question: str,
    *,
    model: str | None = None,
    coverage: dict[str, SubtopicCoverage] | None = None,
) -> list[str]:
    """LLM-decompose a research question into 3-6 sub-topics.

    When ``coverage`` is supplied, calls the REFINE prompt so the LLM
    re-plans against observed gaps.
    """
    from sciknow.config import settings
    from sciknow.rag.llm import complete

    chosen = model or settings.llm_fast_model
    if coverage:
        cov_json = json.dumps(
            [
                {"subtopic": sc.subtopic, "n_papers": sc.n_papers}
                for sc in coverage.values()
            ],
            ensure_ascii=False,
        )
        usr = (
            f"Original question:\n\n{question.strip()}\n\n"
            f"Previous decomposition + coverage:\n{cov_json}\n\n"
            "Return the refined JSON decomposition now."
        )
        sys = _REFINE_SYSTEM
    else:
        usr = (
            f"Research question:\n\n{question.strip()}\n\n"
            "Return the decomposition JSON now."
        )
        sys = _DECOMP_SYSTEM

    try:
        raw = complete(
            sys, usr,
            model=chosen, temperature=0.3, num_ctx=4096, num_predict=600,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("decompose_question LLM call failed: %s", exc)
        return []

    txt = _strip_think_and_fences(raw)
    try:
        data = json.loads(txt, strict=False)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0), strict=False)
        except Exception:
            return []

    subs = data.get("subtopics") or []
    out: list[str] = []
    seen: set[str] = set()
    for s in subs:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if not s or len(s) > 200:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    # Clamp to 3..6 like the prompt asked; if the LLM gave < 3 we
    # still return them — the caller can deal.
    return out[:6]


# ── Coverage check via hybrid_search ────────────────────────────────

def check_coverage(
    subtopics: list[str],
    *,
    candidate_k: int = 30,
    doc_threshold: int = 3,       # papers needed to count as covered
    score_floor: float = 0.15,    # RRF score floor for "relevant"
) -> dict[str, SubtopicCoverage]:
    """Run hybrid search over the corpus for each sub-topic and
    compute coverage. ``doc_threshold`` is the count of distinct
    document_ids required to call a sub-topic "covered".
    """
    from sciknow.retrieval.hybrid_search import search
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    client = get_client()
    out: dict[str, SubtopicCoverage] = {}
    with get_session() as session:
        for topic in subtopics:
            try:
                hits = search(topic, client, session, candidate_k=candidate_k)
            except Exception as exc:  # noqa: BLE001
                logger.warning("coverage check failed for %r: %s", topic, exc)
                out[topic] = SubtopicCoverage(subtopic=topic, n_papers=0, top_score=0.0)
                continue
            # Count distinct papers whose top hit beats the floor.
            per_doc_best: dict[str, float] = {}
            titles: dict[str, str] = {}
            for h in hits:
                doc_id = str(getattr(h, "document_id", "") or "")
                if not doc_id:
                    continue
                score = float(getattr(h, "rrf_score", 0.0) or getattr(h, "score", 0.0))
                if score < score_floor:
                    continue
                if score > per_doc_best.get(doc_id, -1.0):
                    per_doc_best[doc_id] = score
                    t = getattr(h, "paper_title", None) or getattr(h, "title", None) or ""
                    titles[doc_id] = t
            ranked = sorted(per_doc_best.items(), key=lambda kv: -kv[1])
            top_titles = [titles.get(d, "") for d, _ in ranked[:5] if titles.get(d, "")]
            out[topic] = SubtopicCoverage(
                subtopic=topic,
                n_papers=len(per_doc_best),
                top_score=(ranked[0][1] if ranked else 0.0),
                sample_titles=top_titles,
            )
    return out


# ── Orchestrator ────────────────────────────────────────────────────

def run_agentic_expansion(
    question: str,
    *,
    max_rounds: int = 3,
    budget_per_gap: int = 10,
    doc_threshold: int = 3,
    model: str | None = None,
    execute_round_callback=None,
    progress_callback=None,
) -> Iterator[dict]:
    """Drive the plan → check → expand → re-check loop.

    Yields event dicts the CLI / web layer can stream:

    - ``{"type": "decomp", "subtopics": [...]}``
    - ``{"type": "coverage", "round": N, "data": [{subtopic, n_papers, ...}]}``
    - ``{"type": "round_start", "round": N, "gaps": [...], "budget": N}``
    - ``{"type": "round_done", "round": N, "stats": {...}}``
    - ``{"type": "stopped", "reason": "...", "final_coverage": [...]}``

    ``execute_round_callback(gap_subtopics, budget, round_n)`` is invoked
    with the list of gap sub-topics to expand in this round; return a
    stats dict. The callback owns the actual ranker + download + ingest
    work so this module stays I/O-lean.
    """
    # 1. Initial decomposition
    yield {"type": "progress", "stage": "decomposing",
           "detail": "LLM decomposing the research question into sub-topics…"}
    subtopics = decompose_question(question, model=model)
    if not subtopics:
        yield {"type": "error",
               "message": "LLM failed to decompose the question. Try rephrasing."}
        return
    yield {"type": "decomp", "subtopics": subtopics}

    prev_coverage_n: dict[str, int] = {}
    last_gap_ids: set[str] = set()

    for round_n in range(1, max_rounds + 1):
        # 2. Coverage
        yield {"type": "progress", "stage": "coverage",
               "detail": f"Round {round_n}: measuring corpus coverage for {len(subtopics)} sub-topics…"}
        coverage = check_coverage(subtopics, doc_threshold=doc_threshold)
        yield {
            "type": "coverage", "round": round_n,
            "data": [
                {
                    "subtopic": sc.subtopic,
                    "n_papers": sc.n_papers,
                    "top_score": round(sc.top_score, 3),
                    "sample_titles": sc.sample_titles[:3],
                    "covered": sc.n_papers >= doc_threshold,
                }
                for sc in coverage.values()
            ],
        }

        # 3. Identify gaps
        gaps = [sc.subtopic for sc in coverage.values() if sc.n_papers < doc_threshold]
        if not gaps:
            yield {"type": "stopped",
                   "reason": f"all {len(subtopics)} sub-topics covered (≥{doc_threshold} papers each)",
                   "final_coverage": [
                       {"subtopic": sc.subtopic, "n_papers": sc.n_papers}
                       for sc in coverage.values()
                   ]}
            return

        # Check if this round would repeat the same gap set with no progress
        gap_set = frozenset(gaps)
        current_gap_counts = {k: v.n_papers for k, v in coverage.items()}
        if round_n > 1 and set(gap_set) == last_gap_ids:
            # Did any gap sub-topic's count increase?
            progressed = any(
                current_gap_counts.get(t, 0) > prev_coverage_n.get(t, 0)
                for t in gap_set
            )
            if not progressed:
                yield {"type": "progress", "stage": "replanning",
                       "detail": f"Round {round_n}: no progress on gaps; asking LLM to refine the decomposition…"}
                refined = decompose_question(question, model=model, coverage=coverage)
                if refined and refined != subtopics:
                    subtopics = refined
                    yield {"type": "decomp", "subtopics": subtopics,
                           "replanned": True}
                    # Re-measure on the refined set next iteration
                    continue
                else:
                    yield {"type": "stopped",
                           "reason": "no coverage progress + LLM could not refine further",
                           "final_coverage": [
                               {"subtopic": sc.subtopic, "n_papers": sc.n_papers}
                               for sc in coverage.values()
                           ]}
                    return

        # 4. Execute expand per gap sub-topic via the callback
        yield {"type": "round_start", "round": round_n,
               "gaps": gaps, "budget": budget_per_gap}
        if execute_round_callback is not None:
            try:
                stats = execute_round_callback(gaps, budget_per_gap, round_n) or {}
            except Exception as exc:  # noqa: BLE001
                yield {"type": "error",
                       "message": f"round {round_n} callback crashed: {exc}"}
                return
            yield {"type": "round_done", "round": round_n, "stats": stats}
        else:
            # Dry-run / test mode — no actual expansion executed
            yield {"type": "round_done", "round": round_n,
                   "stats": {"mode": "dry-run"}}

        prev_coverage_n = current_gap_counts
        last_gap_ids = set(gap_set)

    yield {"type": "stopped",
           "reason": f"reached max_rounds={max_rounds}",
           "final_coverage": [
               {"subtopic": sc.subtopic, "n_papers": sc.n_papers}
               for sc in coverage.values()
           ]}
