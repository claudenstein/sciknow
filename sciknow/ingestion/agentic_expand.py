"""Phase 54.6.114 (Tier 2 #2) — question-driven agentic expansion.
Phase 54.6.124 (Tier 4 #2) — resume / checkpointing.

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

See ``docs/research/EXPAND_ENRICH_RESEARCH_2.md`` §3.1.
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


# ── Checkpointing (Tier 4 #2, 54.6.124) ─────────────────────────────

def _state_path(project_root, question: str):
    """Slug the question into a deterministic state-file path so the
    same question re-uses its state on resume. Keeps runs isolated by
    a short hash suffix; collisions between two different questions
    with the same first 48 chars are fine (the full question is
    stored inside the JSON and verified on load)."""
    import hashlib
    from pathlib import Path
    slug = "".join(c if (c.isalnum() or c in "-_") else "-"
                   for c in question[:48].strip().lower())
    slug = slug.strip("-") or "question"
    h = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]
    d = Path(project_root) / "data" / "expand" / "agentic"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{slug}-{h}.json"


def load_state(project_root, question: str) -> dict | None:
    """Return the persisted state for this question, or None if absent."""
    import json as _json
    p = _state_path(project_root, question)
    if not p.exists():
        return None
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("agentic state unreadable at %s: %s", p, exc)
        return None
    if data.get("question") != question:
        # Hash collision — don't silently clobber a different question.
        logger.warning("agentic state question mismatch at %s — ignoring",
                       p)
        return None
    return data


def save_state(project_root, question: str, state: dict) -> None:
    """Overwrite the state file atomically-ish (write-then-rename)."""
    import json as _json
    p = _state_path(project_root, question)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(
        _json.dumps({"question": question, **state}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(p)


def clear_state(project_root, question: str) -> None:
    p = _state_path(project_root, question)
    if p.exists():
        p.unlink()


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
    candidate_k: int = 15,
    doc_threshold: int = 3,       # papers needed to count as covered
    score_floor: float = 0.15,    # cross-encoder score floor for "relevant" (0..1)
) -> dict[str, SubtopicCoverage]:
    """Run hybrid search + cross-encoder rerank for each sub-topic and
    compute coverage. ``doc_threshold`` is the count of distinct
    document_ids required to call a sub-topic "covered".

    Phase 54.6.134 — uses the reranker for the relevance floor.
    Raw RRF scores from ``hybrid_search`` plateau at ~0.04 regardless
    of semantic match (rank-based fusion of dense/sparse/FTS), so a
    static floor on them cannot discriminate covered vs gap — the
    original implementation silently marked every sub-topic as a gap
    because 0.04 < 0.15. The cross-encoder ``bge-reranker-v2-m3``
    produces normalized [0..1] similarity scores that cleanly separate
    matches (~0.9) from noise (~0.01), which is what the 0.15 floor
    was always meant to operate on.
    """
    from sciknow.retrieval.hybrid_search import search
    from sciknow.retrieval import reranker as _reranker
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client

    client = get_client()
    out: dict[str, SubtopicCoverage] = {}
    with get_session() as session:
        for topic in subtopics:
            try:
                hits = search(topic, client, session, candidate_k=candidate_k)
                if hits:
                    hits = _reranker.rerank(topic, hits, top_k=candidate_k)
            except Exception as exc:  # noqa: BLE001
                logger.warning("coverage check failed for %r: %s", topic, exc)
                out[topic] = SubtopicCoverage(subtopic=topic, n_papers=0, top_score=0.0)
                continue
            # Count distinct papers whose top hit beats the floor.
            # After rerank(), h.rrf_score holds the cross-encoder score.
            per_doc_best: dict[str, float] = {}
            titles: dict[str, str] = {}
            for h in hits:
                doc_id = str(getattr(h, "document_id", "") or "")
                if not doc_id:
                    continue
                score = float(getattr(h, "rrf_score", 0.0) or 0.0)
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


# ── Phase 54.6.132 → 54.6.133 — RRF-parity preview gathering ───────


def _parse_shortlist_tsv(tsv_path) -> list[dict]:
    """Parse a Phase-49 ``expand_shortlist.tsv`` into the candidate
    dict shape eapRender expects. Mirrors the parser in the
    ``/api/corpus/expand/preview/.../candidates`` web endpoint so the
    agentic preview produces the same row shape as the non-agentic
    expand preview."""
    import csv

    out: list[dict] = []

    def _f(row, k):
        v = row.get(k)
        if not v or v in ("None", "nan"):
            return None
        try:
            return float(v)
        except Exception:
            return None

    def _i(row, k):
        v = row.get(k)
        if not v or v in ("None", "nan"):
            return None
        try:
            return int(float(v))
        except Exception:
            return None

    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            decision = (row.get("decision") or "").upper()
            # Preview surfaces only KEEP rows — the dropped ones were
            # filtered out by the same hard filters auto-mode applies
            # (retraction / predatory / one-timer) and would never be
            # downloaded. Showing them just clutters the cherry-pick UI.
            if decision != "KEEP":
                continue
            doi = (row.get("doi") or "").strip() or None
            out.append({
                "doi": doi,
                "arxiv_id": (row.get("arxiv_id") or "").strip() or None,
                "title": (row.get("title") or "").strip(),
                "authors": [],
                "year": _i(row, "year"),
                "relevance_score": _f(row, "bge_m3_cosine"),
                "rrf_score": _f(row, "rrf_score"),
                "decision": decision,
                "drop_reason": None,
                "signals": {
                    "co_citation":       _f(row, "co_citation"),
                    "bib_coupling":      _f(row, "bib_coupling"),
                    "pagerank":          _f(row, "pagerank"),
                    "influential_cites": _i(row, "influential_cites"),
                    "cited_by":          _i(row, "cited_by"),
                    "velocity":          _f(row, "velocity"),
                    "concept_overlap":   _f(row, "concept_overlap"),
                    "venue":             (row.get("venue") or "").strip() or None,
                },
            })
    return out


def gather_candidates_for_gaps(
    gaps: list[str],
    *,
    budget_per_gap: int = 10,
    on_progress=None,
    project_root=None,
) -> dict:
    """Phase 54.6.133 — HONEST preview: per gap, spawn the same
    ``db expand --relevance-query <gap>`` pipeline auto-mode runs,
    but in dry-run mode that emits a shortlist TSV instead of
    downloading. Parse each TSV into candidate dicts annotated with
    ``_agentic_subtopic`` so the GUI shows them grouped, dedup by
    DOI across gaps, and return the merged list.

    Why subprocess + TSV: ``_run_rrf_ranker`` is wired into the CLI
    ``expand`` command's reference-extraction + cosine-prefiltering
    pipeline; calling it in-process would require reproducing a few
    hundred lines of upstream setup. The subprocess + TSV mechanism
    is already proven (used by ``/api/corpus/expand/preview``) and
    gives true parity: the candidates surfaced here are exactly the
    ones auto-mode would download, just deferred to user approval.

    Cost: ~30-90s per gap subprocess (bge-m3 cold-load + reference
    extraction + RRF). For 5 gaps that's typically 3-7 min wall.
    Worth it because the preview now matches reality.
    """
    import os
    import subprocess
    import sys as _sys
    import tempfile
    import time as _time
    from pathlib import Path as _Path

    out: list[dict] = []
    seen_dois: set[str] = set()
    per_gap_stats: list[dict] = []
    total_kept = 0
    cross_gap_dups = 0

    tmp_dir = _Path(tempfile.mkdtemp(prefix="sciknow-agentic-preview-"))
    try:
        for i, gap in enumerate(gaps, start=1):
            if on_progress:
                try:
                    on_progress(i, len(gaps), gap)
                except Exception:
                    pass
            tsv_path = tmp_dir / f"gap-{i}.tsv"
            argv = [
                _sys.executable, "-m", "sciknow.cli.main",
                "db", "expand",
                "--strategy", "rrf",
                "--relevance-query", gap,
                "--budget", str(budget_per_gap),
                "--dry-run",
                "--shortlist-tsv", str(tsv_path),
                "--no-resolve",   # skip slow Crossref title lookup
                "--workers", "0",
            ]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            t0 = _time.monotonic()
            try:
                res = subprocess.run(
                    argv, capture_output=True, text=True,
                    env=env, timeout=600,
                )
                elapsed = _time.monotonic() - t0
            except subprocess.TimeoutExpired:
                per_gap_stats.append({
                    "subtopic": gap, "n_candidates": 0,
                    "error": "timeout (600s) — corpus too large or LLM stuck?",
                })
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("RRF preview subprocess failed for gap %r: %s", gap, exc)
                per_gap_stats.append({
                    "subtopic": gap, "n_candidates": 0,
                    "error": str(exc)[:200],
                })
                continue
            if not tsv_path.exists():
                stderr_tail = (res.stderr or "")[-300:].strip()
                per_gap_stats.append({
                    "subtopic": gap, "n_candidates": 0,
                    "error": (
                        "no shortlist produced (no qualifying refs?). "
                        f"stderr tail: {stderr_tail}"
                    )[:300],
                })
                continue
            kept = 0
            for c in _parse_shortlist_tsv(tsv_path):
                doi = (c.get("doi") or "").lower()
                if doi and doi in seen_dois:
                    cross_gap_dups += 1
                    continue
                if doi:
                    seen_dois.add(doi)
                c["_agentic_subtopic"] = gap
                out.append(c)
                kept += 1
            total_kept += kept
            per_gap_stats.append({
                "subtopic": gap, "n_candidates": kept,
                "elapsed_s": round(elapsed, 1),
            })
    finally:
        # Best-effort cleanup of the per-gap TSVs + tmp dir.
        try:
            import shutil as _sh
            _sh.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    # Phase 54.6.135 — annotate cached_status so the web UI's
    # "Hide cached" checkbox actually filters agentic-preview rows.
    # The TSV shortlist format doesn't carry cache state, and the
    # preview subprocess runs with `--dry-run` (so the downloader's
    # skip-cached logic never gets a chance to flag them), which means
    # candidates reach eapRender with cached_status unset — the filter
    # sees `!undefined === true` and lets every row through.
    try:
        from sciknow.core.expand_ops import _load_download_caches
        no_oa_keys, fail_keys = _load_download_caches()
        for c in out:
            doi_k = (c.get("doi") or "").lower()
            arx_k = (c.get("arxiv_id") or "").lower()
            cs = None
            if doi_k and doi_k in no_oa_keys:       cs = "no_oa"
            elif arx_k and arx_k in no_oa_keys:     cs = "no_oa"
            elif doi_k and doi_k in fail_keys:      cs = "ingest_failed"
            elif arx_k and arx_k in fail_keys:      cs = "ingest_failed"
            c["cached_status"] = cs
    except Exception as exc:  # noqa: BLE001
        logger.warning("agentic preview cache annotation failed: %s", exc)

    return {
        "candidates": out,
        "gaps": per_gap_stats,
        "info": {
            "n_gaps": len(gaps),
            "raw_candidates": total_kept + cross_gap_dups,
            "merged_candidates": len(out),
            "cross_gap_duplicates": cross_gap_dups,
            "budget_per_gap": budget_per_gap,
            "ranker": "rrf",  # Phase 54.6.133 — parity with auto-mode
        },
    }


def run_preview_round(
    question: str,
    *,
    budget_per_gap: int = 10,
    doc_threshold: int = 3,
    model: str | None = None,
    on_progress=None,
) -> dict:
    """Phase 54.6.132 — single-round preview orchestrator. Decompose
    question → measure coverage → identify gaps → gather candidates
    per gap. Returns everything the GUI needs to render an interactive
    cherry-pick step before any download. Intentionally synchronous
    and stateless: each call advances ONE round, the user picks +
    downloads, then re-calls to do the next round (so coverage is
    re-measured against the new corpus state).
    """
    if on_progress:
        on_progress(0, 4, "decomposing question")
    subtopics = decompose_question(question, model=model)
    if not subtopics:
        return {
            "subtopics": [], "coverage": [], "gaps": [],
            "candidates": [], "info": {
                "error": "LLM decomposition returned no sub-topics. "
                         "Try rephrasing the question.",
            },
        }

    if on_progress:
        on_progress(1, 4, "measuring corpus coverage")
    coverage = check_coverage(subtopics, doc_threshold=doc_threshold)
    coverage_rows = [
        {
            "subtopic": sc.subtopic, "n_papers": sc.n_papers,
            "top_score": sc.top_score, "covered": sc.n_papers >= doc_threshold,
            "sample_titles": sc.sample_titles[:3],
        }
        for sc in coverage.values()
    ]
    gaps = [sc.subtopic for sc in coverage.values()
            if sc.n_papers < doc_threshold]

    if not gaps:
        return {
            "subtopics": subtopics,
            "coverage": coverage_rows,
            "gaps": [], "candidates": [],
            "info": {
                "all_covered": True,
                "doc_threshold": doc_threshold,
                "message": (
                    f"All {len(subtopics)} sub-topics already have "
                    f"≥{doc_threshold} papers. No expansion needed."
                ),
            },
        }

    if on_progress:
        on_progress(2, 4, f"gathering candidates for {len(gaps)} gap(s)")
    gathered = gather_candidates_for_gaps(
        gaps, budget_per_gap=budget_per_gap,
        on_progress=lambda i, n, g: (
            on_progress(2 + (i / max(n, 1)) * 2, 4, f"gathering: {g}")
            if on_progress else None
        ),
    )
    if on_progress:
        on_progress(4, 4, "done")
    return {
        "subtopics": subtopics,
        "coverage": coverage_rows,
        "gaps": gathered["gaps"],
        "candidates": gathered["candidates"],
        "info": {
            **gathered["info"],
            "doc_threshold": doc_threshold,
            "all_covered": False,
        },
    }


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
    project_root=None,
    resume: bool = False,
) -> Iterator[dict]:
    """Drive the plan → check → expand → re-check loop.

    Yields event dicts the CLI / web layer can stream:

    - ``{"type": "decomp", "subtopics": [...]}``
    - ``{"type": "coverage", "round": N, "data": [{subtopic, n_papers, ...}]}``
    - ``{"type": "round_start", "round": N, "gaps": [...], "budget": N}``
    - ``{"type": "round_done", "round": N, "stats": {...}}``
    - ``{"type": "stopped", "reason": "...", "final_coverage": [...]}``
    - ``{"type": "resumed", "from_round": N, "prior_rounds": [...]}``  (54.6.124)

    ``execute_round_callback(gap_subtopics, budget, round_n)`` is invoked
    with the list of gap sub-topics to expand in this round; return a
    stats dict. The callback owns the actual ranker + download + ingest
    work so this module stays I/O-lean.

    Phase 54.6.124 — when ``project_root`` is supplied, state (subtopics,
    completed rounds, coverage history) is persisted to
    ``<project>/data/expand/agentic/<slug>-<hash>.json`` between rounds.
    ``resume=True`` loads that state so a crash / manual stop between
    rounds can be picked up with ``db expand --question "..." --resume``.
    """
    # 1. Resume or decompose
    state: dict = {}
    starting_round = 1
    if resume and project_root is not None:
        prior = load_state(project_root, question)
        if prior:
            state = prior
            subtopics = list(prior.get("subtopics") or [])
            starting_round = int(prior.get("next_round", 1))
            yield {"type": "resumed",
                   "from_round": starting_round,
                   "prior_rounds": prior.get("rounds") or [],
                   "subtopics": subtopics}
        else:
            yield {"type": "progress", "stage": "resume_miss",
                   "detail": "No prior state for this question — starting fresh."}
            subtopics = None
    else:
        subtopics = None

    if not subtopics:
        yield {"type": "progress", "stage": "decomposing",
               "detail": "LLM decomposing the research question into sub-topics…"}
        subtopics = decompose_question(question, model=model)
        if not subtopics:
            yield {"type": "error",
                   "message": "LLM failed to decompose the question. Try rephrasing."}
            return
        yield {"type": "decomp", "subtopics": subtopics}
        state = {
            "subtopics": list(subtopics),
            "rounds": [],
            "next_round": 1,
            "max_rounds": max_rounds,
            "budget_per_gap": budget_per_gap,
            "doc_threshold": doc_threshold,
        }
        if project_root is not None:
            save_state(project_root, question, state)

    prev_coverage_n: dict[str, int] = {}
    last_gap_ids: set[str] = set()
    # On resume, seed the progress trackers from the last recorded round
    # so the "no-progress → replan" logic doesn't fire spuriously.
    if state.get("rounds"):
        last_record = state["rounds"][-1]
        prev_coverage_n = {
            e["subtopic"]: e.get("n_papers", 0)
            for e in (last_record.get("coverage") or [])
        }
        last_gap_ids = set(last_record.get("gaps") or [])

    for round_n in range(starting_round, max_rounds + 1):
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
            # 54.6.124 — clean state on early success too.
            if project_root is not None:
                clear_state(project_root, question)
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
                # 54.6.124 — persist the CURRENT state so `--resume`
                # can retry from this round after the user fixes the
                # issue. `next_round` stays at round_n so the round
                # is not marked complete.
                if project_root is not None:
                    state["next_round"] = round_n
                    state.setdefault("rounds", []).append({
                        "round": round_n,
                        "coverage": [
                            {"subtopic": sc.subtopic, "n_papers": sc.n_papers}
                            for sc in coverage.values()
                        ],
                        "gaps": list(gaps),
                        "error": f"{type(exc).__name__}: {exc}"[:400],
                    })
                    save_state(project_root, question, state)
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

        # 54.6.124 — checkpoint after the round lands cleanly. Advances
        # next_round so a subsequent `--resume` skips this one.
        if project_root is not None:
            state.setdefault("rounds", []).append({
                "round": round_n,
                "coverage": [
                    {"subtopic": sc.subtopic, "n_papers": sc.n_papers}
                    for sc in coverage.values()
                ],
                "gaps": list(gaps),
            })
            state["next_round"] = round_n + 1
            save_state(project_root, question, state)

    yield {"type": "stopped",
           "reason": f"reached max_rounds={max_rounds}",
           "final_coverage": [
               {"subtopic": sc.subtopic, "n_papers": sc.n_papers}
               for sc in coverage.values()
           ]}
    # Clean state on a fully-completed run so `--resume` doesn't loop.
    if project_root is not None:
        clear_state(project_root, question)
