"""Phase 49 — RRF-fused multi-signal ranker for `db expand`.

Upgrades the Phase 44 single-signal bge-m3 cosine filter to a proper
ranker that considers citation-graph structure + semantic similarity
+ negative signals, then fuses them via Reciprocal Rank Fusion
(Cormack, Clarke & Buettcher 2009) into a single sortable score.

The audit + research that justify this design live in
`docs/research/EXPAND_RESEARCH.md`. The short version:

- **Hard filters** (retraction, predatory venue, editorial/erratum,
  short proceedings, one-timer) drop candidates before they reach the
  ranker. Implemented in `expand_filters.py`; the one-timer check
  lives here because it needs corpus-side counts.
- **Signals** are deliberately weak individually — log-normalised
  citation count, co-citation strength, bibliographic coupling, local
  PageRank, bge-m3 cosine, author overlap, concept overlap, venue
  prior, influential-citation count, citation velocity. None is
  reliable alone.
- **RRF** fuses rankings across signals. Scale-agnostic (PageRank ~
  1e-5 vs cosine ~ 0.7 vs citation count ~ 500 would wreck any
  weighted sum); only the per-signal ranks matter.
- **Best-first frontier** with a budget per round + stopping rule
  (median-score drop OR novelty-ratio floor). The orchestrator in
  `cli/db.py` uses these; this module just produces the ranked list.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

logger = logging.getLogger("sciknow.expand_ranker")


# ── Candidate feature vector ────────────────────────────────────────

@dataclass
class CandidateFeatures:
    """Every signal we want to consider for one candidate paper. All
    numeric fields default to 0.0/0 so the ranker can score a candidate
    even when some signals couldn't be computed (API timeout,
    missing DOI, etc)."""

    # Identity
    doi: str = ""
    arxiv_id: str = ""
    title: str = ""
    year: int = 0
    openalex_id: str = ""
    venue: str = ""
    doc_type: str = ""

    # Citation-graph signals (higher = better)
    co_citation: int = 0
    bib_coupling: float = 0.0      # Salton-normalized, in [0, 1]
    pagerank: float = 0.0          # local depth-2 subgraph
    influential_cite_count: int = 0  # corpus → candidate with isInfluential
    cited_by_count: int = 0
    citation_velocity: float = 0.0  # cites / active year

    # Semantic / content signals (higher = better)
    bge_m3_cosine: float = 0.0
    # Phase 54.6.113 (Tier 2 #1) — Cohan 2024 SciRepEval. Cosine of
    # "how OTHER papers describe this paper" (S2 citation contexts,
    # bge-m3-embedded) against the corpus centroid. 0.0 when the
    # candidate has no citations yet or S2 was skipped.
    citation_context_cosine: float = 0.0
    citation_context_n: int = 0     # count of unique contexts used
    concept_overlap: float = 0.0    # Jaccard against corpus concept histogram
    author_overlap: int = 0         # #authors with ≥2 corpus papers
    venue_weight: float = 0.0       # corpus-coverage of the venue

    # Corpus-side counts (used by one-timer filter)
    corpus_cite_count: int = 0
    external_cite_count: int = 0

    # Hard-drop reason (set by filters; empty = kept)
    hard_drop_reason: str = ""

    # Final fused score (filled after RRF)
    rrf_score: float = 0.0

    # Human-readable provenance — every ranker decision appends here
    decisions: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable identifier for ranking. DOI preferred; fall back to
        arXiv-id then to title prefix."""
        return (
            (self.doi.lower() if self.doi else "")
            or (self.arxiv_id.lower() if self.arxiv_id else "")
            or (self.title or "").lower()[:60]
        )

    def is_one_timer(self, *, corpus_min: int = 1, external_min: int = 5) -> bool:
        """Cited exactly once by our corpus and ≤ external_min times
        anywhere else — almost certainly a drive-by mention, not a
        canonical reference. Default (1, 5) per the research doc."""
        return (
            self.corpus_cite_count <= corpus_min
            and self.external_cite_count < external_min
        )


# ── Reciprocal Rank Fusion ──────────────────────────────────────────

def rrf_fuse(
    rankings: Iterable[list[str]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion (Cormack et al. 2009). Each ranking is
    an ordered list of candidate keys (best first). The fused score
    for candidate C is Σ 1/(k + rank_i(C)), where rank_i is 1-indexed
    and k=60 is the paper's default.

    Scale-agnostic — ignores the underlying signal magnitudes, which
    is exactly right when they span many orders of magnitude."""
    scores: dict[str, float] = defaultdict(float)
    for rank in rankings:
        for i, cand_key in enumerate(rank):
            scores[cand_key] += 1.0 / (k + i + 1)
    return dict(scores)


def _rank_by(
    candidates: list[CandidateFeatures],
    getter,
    *,
    reverse: bool = True,
) -> list[str]:
    """Sort candidates by `getter(c)` (desc by default) and return
    their keys. Ties are stable (Python's sort)."""
    return [c.key for c in sorted(candidates, key=getter, reverse=reverse)]


# ── Local PageRank on a depth-2 citation subgraph ──────────────────
# scipy sparse CSR for the transition matrix so the math scales to
# tens of thousands of nodes (the depth-2 subgraph around a 100-seed
# corpus can touch a few thousand works). Dense numpy would need N²
# floats, which blows past 1 GB around N ≈ 12000.

def local_pagerank(
    nodes: list[str],
    edges: list[tuple[str, str]],
    *,
    damping: float = 0.85,
    iters: int = 30,
) -> dict[str, float]:
    """Power-iteration PageRank on the subgraph (nodes, edges). Edges
    are directed (src, tgt) where src cites tgt. Nodes without an
    outlink are dangling and teleport uniformly. Returns normalised
    scores summing to 1. Empty graph → empty dict."""
    n = len(nodes)
    if n == 0:
        return {}
    idx = {node: i for i, node in enumerate(nodes)}

    # Collect clean edges (both endpoints in the node set, no self-loops)
    rows: list[int] = []
    cols: list[int] = []
    out_deg = np.zeros(n, dtype=np.int64)
    for src, tgt in edges:
        s = idx.get(src)
        t = idx.get(tgt)
        if s is None or t is None or s == t:
            continue
        rows.append(t)  # column-stochastic: M[t, s] = 1/out_deg[s]
        cols.append(s)
        out_deg[s] += 1

    from scipy.sparse import csr_matrix  # local import; scipy is a transitive dep
    if rows:
        # Weight each entry by 1 / out_deg of its source column
        weights = np.array([1.0 / out_deg[s] for s in cols], dtype=np.float64)
        M = csr_matrix((weights, (rows, cols)), shape=(n, n))
    else:
        M = csr_matrix((n, n), dtype=np.float64)

    dangling = (out_deg == 0).astype(np.float64) / n
    rank = np.ones(n, dtype=np.float64) / n
    teleport = (1.0 - damping) / n
    for _ in range(iters):
        rank = damping * (M.dot(rank) + dangling * rank.sum()) + teleport
    return {nodes[i]: float(rank[i]) for i in range(n)}


# ── Corpus-side derivations used by signals ─────────────────────────

def bibliographic_coupling(
    candidate_refs: Iterable[str],
    seed_refs_union: set[str],
    seed_refs_size: int,
) -> float:
    """Salton-normalised bibliographic coupling between a candidate
    and the seed set: |refs(candidate) ∩ refs(seeds)| /
    sqrt(|refs(candidate)| · |refs(seeds)|). Zero when either side is
    empty."""
    cand = {r for r in candidate_refs if r}
    if not cand or not seed_refs_union or seed_refs_size == 0:
        return 0.0
    overlap = len(cand & seed_refs_union)
    denom = (len(cand) * seed_refs_size) ** 0.5
    return overlap / denom if denom > 0 else 0.0


def compute_co_citation(
    candidate_openalex_id: str,
    forward_citers_refs: list[list[str]],
) -> int:
    """How many papers in the forward-citation set of the seeds also
    cite the candidate? This is the Connected Papers core metric.

    `forward_citers_refs` is a list where each element is the
    `referenced_works` of one paper that cites at least one seed."""
    if not candidate_openalex_id:
        return 0
    return sum(
        1 for refs in forward_citers_refs if candidate_openalex_id in refs
    )


def citation_velocity(counts_by_year: list[dict] | None, window: int = 3) -> float:
    """Mean citations/year over the most recent `window` years. Uses
    the OpenAlex `counts_by_year` array. Zero when unavailable."""
    if not counts_by_year:
        return 0.0
    recent = sorted(counts_by_year, key=lambda d: -(d.get("year") or 0))[:window]
    if not recent:
        return 0.0
    return sum(d.get("cited_by_count") or 0 for d in recent) / len(recent)


# ── Top-level scoring ───────────────────────────────────────────────

def apply_one_timer_filter(
    candidates: list[CandidateFeatures],
    *,
    corpus_min: int = 1,
    external_min: int = 5,
) -> None:
    """Mutate: mark one-timers as hard-dropped. Separate step so the
    caller can report how many were dropped for this specific
    reason."""
    for c in candidates:
        if c.hard_drop_reason:
            continue
        if c.is_one_timer(corpus_min=corpus_min, external_min=external_min):
            c.hard_drop_reason = (
                f"one_timer(corpus={c.corpus_cite_count},external={c.external_cite_count})"
            )
            c.decisions.append(c.hard_drop_reason)


def apply_mmr(
    ranked: list[CandidateFeatures],
    concept_sets: dict[str, set[str]],
    *,
    lambda_: float = 0.7,
    top_k: int | None = None,
) -> list[CandidateFeatures]:
    """Phase 54.6.111 (Tier 1 #3) — Maximal Marginal Relevance diversity
    re-rank over a score-sorted candidate list.

    Penalises each pick by its max concept-set Jaccard similarity to
    already-picked candidates, then re-orders. ``lambda_=0.7`` keeps
    70% of the original RRF signal and 30% of the diversity signal —
    enough to break monoculture without destroying the ranker's intent.

    ``concept_sets`` maps candidate.key → set of OpenAlex concept
    display_names (already computed for the ``concept_overlap`` signal).
    Candidates absent from ``concept_sets`` are treated as "no concepts"
    and Jaccard collapses to 0 — neutral, never penalized.

    When ``top_k`` is set, only the first ``top_k`` survive MMR; the
    tail keeps its RRF order. Hard-dropped candidates (rrf_score == 0)
    are skipped to preserve the dry-run TSV's "dropped-first" section.
    """
    kept = [c for c in ranked if not c.hard_drop_reason]
    tail = [c for c in ranked if c.hard_drop_reason]
    if len(kept) <= 1:
        return kept + tail
    limit = top_k if (top_k is not None and top_k > 0) else len(kept)
    limit = min(limit, len(kept))

    selected: list[CandidateFeatures] = []
    selected_keys: set[str] = set()
    pool = kept.copy()

    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    # Normalize the RRF score to [0, 1] for the MMR convex combination.
    max_rrf = max((c.rrf_score for c in kept), default=1.0) or 1.0
    while pool and len(selected) < limit:
        best = None
        best_score = -1e18
        for c in pool:
            relevance = c.rrf_score / max_rrf
            if selected:
                c_set = concept_sets.get(c.key, set())
                max_sim = max(
                    _jaccard(c_set, concept_sets.get(s.key, set()))
                    for s in selected
                )
            else:
                max_sim = 0.0
            mmr = lambda_ * relevance - (1.0 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best = c
        assert best is not None
        best.decisions.append(f"mmr_pick@{len(selected)+1}: score={best_score:+.3f}")
        selected.append(best)
        selected_keys.add(best.key)
        pool.remove(best)

    # Appending the remaining pool in their original RRF order.
    rest = [c for c in kept if c.key not in selected_keys]
    return selected + rest + tail


def score_via_rrf(
    candidates: list[CandidateFeatures],
    *,
    k: int = 60,
) -> list[CandidateFeatures]:
    """Apply RRF over the eight signals and return candidates sorted
    by fused score (desc). Hard-dropped candidates are excluded from
    ranking but retained in the returned list at the end (score 0.0)
    so the dry-run TSV can include them for HITL review."""
    kept = [c for c in candidates if not c.hard_drop_reason]
    dropped = [c for c in candidates if c.hard_drop_reason]
    if not kept:
        for c in dropped:
            c.rrf_score = 0.0
        return dropped

    rankings = [
        _rank_by(kept, lambda c: c.co_citation),
        _rank_by(kept, lambda c: c.bib_coupling),
        _rank_by(kept, lambda c: c.pagerank),
        _rank_by(kept, lambda c: c.bge_m3_cosine),
        # Phase 54.6.113 — citation-context cosine as a distinct signal
        # from bge_m3_cosine (title+abstract). They're complementary:
        # new papers have no contexts, old papers with bad abstracts
        # have no abstract signal.
        _rank_by(kept, lambda c: c.citation_context_cosine),
        _rank_by(kept, lambda c: c.influential_cite_count),
        _rank_by(kept, lambda c: c.citation_velocity),
        _rank_by(kept, lambda c: c.concept_overlap),
        _rank_by(kept, lambda c: c.author_overlap),
    ]
    fused = rrf_fuse(rankings, k=k)
    for c in kept:
        c.rrf_score = fused.get(c.key, 0.0)
    for c in dropped:
        c.rrf_score = 0.0
    kept.sort(key=lambda c: c.rrf_score, reverse=True)
    return kept + dropped


# ── Stopping rule for the best-first frontier ──────────────────────

def _parallel_openalex_works(
    dois_and_ids: list[tuple[str, str]],
    *,
    max_workers: int = 8,
) -> dict[str, dict | None]:
    """Fetch OpenAlex works for a batch of (doi, arxiv_id) pairs in
    parallel (I/O bound — threads are fine). Key in the returned
    dict is the DOI in lower-case (or "arxiv:<id>" when only an
    arXiv id is available). Missing / failed fetches → None."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from sciknow.ingestion.expand_apis import fetch_openalex_work

    def _fetch(doi: str, arxiv_id: str) -> tuple[str, dict | None]:
        ident = doi or arxiv_id
        key = (doi or f"arxiv:{arxiv_id}").lower()
        if not ident:
            return key, None
        return key, fetch_openalex_work(ident)

    out: dict[str, dict | None] = {}
    if not dois_and_ids:
        return out
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch, d, a) for d, a in dois_and_ids]
        for fut in as_completed(futures):
            try:
                key, work = fut.result()
                out[key] = work
            except Exception as exc:
                logger.debug("OpenAlex fetch future failed: %s", exc)
    return out


def enrich_from_openalex_work(
    feat: CandidateFeatures, work: dict | None,
) -> None:
    """Populate the OpenAlex-derived fields on a CandidateFeatures
    from a fetched `/works/{id}` response."""
    if not work:
        return
    feat.openalex_id = work.get("id") or feat.openalex_id
    feat.cited_by_count = int(work.get("cited_by_count") or 0)
    feat.citation_velocity = citation_velocity(work.get("counts_by_year"))
    feat.doc_type = (work.get("type") or "").lower()
    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    feat.venue = (source.get("display_name") or "")[:120]
    if not feat.year:
        feat.year = int(work.get("publication_year") or 0)
    # Fall back to OpenAlex title when the seed reference-extractor
    # didn't capture one (common for plain-text bibliography lines).
    if not feat.title:
        feat.title = (work.get("title") or work.get("display_name") or "")[:280]


def compute_corpus_side_counts(
    candidates: list[CandidateFeatures],
    *,
    cited_by_lookup: dict[str, int],
) -> None:
    """Derive `external_cite_count` from the OpenAlex total minus the
    per-candidate corpus cites. `corpus_cite_count` is expected to
    already be set (computed from the seed reference extraction)."""
    for c in candidates:
        total = cited_by_lookup.get(c.key, c.cited_by_count)
        c.external_cite_count = max(0, total - c.corpus_cite_count)


def apply_author_overlap(
    candidates: list[CandidateFeatures],
    corpus_author_counts: dict[str, int],
    candidate_authors: dict[str, list[str]],
    *,
    min_corpus_papers: int = 2,
) -> None:
    """For each candidate, count authors who have ≥2 papers already
    in the corpus. candidate_authors keyed by candidate.key →
    list[author_display_name_lower]. corpus_author_counts maps
    normalised author name → number of corpus papers."""
    for c in candidates:
        names = candidate_authors.get(c.key) or []
        c.author_overlap = sum(
            1 for n in names if corpus_author_counts.get(n, 0) >= min_corpus_papers
        )


def write_shortlist_tsv(
    ranked: list[CandidateFeatures],
    path,
) -> None:
    """Dump the full feature breakdown + RRF score to a TSV for
    human-in-the-loop review. Each row is one candidate; kept and
    dropped rows are both included so the user can see what was
    filtered and why. Format is stable (new columns appended) so
    downstream tooling can parse older outputs."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "rank", "decision", "drop_reason", "rrf_score",
        "doi", "arxiv_id", "title", "year", "venue", "doc_type",
        "co_citation", "bib_coupling", "pagerank",
        "influential_cites", "cited_by", "velocity",
        "bge_m3_cosine", "citation_context_cosine", "citation_context_n",
        "concept_overlap", "author_overlap", "venue_weight",
        "corpus_cites", "external_cites", "openalex_id",
    ]
    with p.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for i, c in enumerate(ranked, start=1):
            decision = "DROP" if c.hard_drop_reason else "KEEP"
            row = [
                str(i),
                decision,
                c.hard_drop_reason or "",
                f"{c.rrf_score:.6f}",
                c.doi or "",
                c.arxiv_id or "",
                (c.title or "").replace("\t", " ").replace("\n", " ")[:200],
                str(c.year or ""),
                (c.venue or "").replace("\t", " ")[:120],
                c.doc_type or "",
                str(c.co_citation),
                f"{c.bib_coupling:.4f}",
                f"{c.pagerank:.6f}",
                str(c.influential_cite_count),
                str(c.cited_by_count),
                f"{c.citation_velocity:.2f}",
                f"{c.bge_m3_cosine:.4f}",
                f"{c.citation_context_cosine:.4f}",
                str(c.citation_context_n),
                f"{c.concept_overlap:.4f}",
                str(c.author_overlap),
                f"{c.venue_weight:.4f}",
                str(c.corpus_cite_count),
                str(c.external_cite_count),
                c.openalex_id or "",
            ]
            f.write("\t".join(row) + "\n")


def should_stop_expansion(
    round_scores: list[float],
    first_round_scores: list[float] | None,
    novelty_ratio: float,
    *,
    median_drop_threshold: float = 0.3,
    novelty_floor: float = 0.3,
) -> tuple[bool, str]:
    """Return (stop?, reason). Two OR'd conditions:
    - median(round_N) < (1 - median_drop_threshold) * median(round_1)
    - novelty_ratio < novelty_floor (fraction of round-N top-K that
      are actually new — not already in the corpus from prior rounds)
    """
    if not round_scores:
        return True, "empty_round"
    if novelty_ratio < novelty_floor:
        return True, f"novelty_ratio={novelty_ratio:.2f}<{novelty_floor:.2f}"
    if first_round_scores:
        median_n = float(np.median(round_scores))
        median_1 = float(np.median(first_round_scores))
        if median_1 > 0 and median_n < (1.0 - median_drop_threshold) * median_1:
            return True, (
                f"median_score_drop={1 - median_n/median_1:.2f}"
                f">={median_drop_threshold:.2f}"
            )
    return False, ""
