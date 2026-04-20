"""Phase 54.6.140 — ground-truth eval set for the visuals ranker.

Mines eval items from the corpus itself. Every stored ``mention_paragraph``
from Phase 54.6.138 is literally an "author said this figure supports
this claim" pair — the source paper's body text explicitly cites the
figure for a specific point. Extracting the sentence that does the
citing gives a query-to-correct-figure pair without manual curation,
and at corpus scale (16 k+ mentions on global-cooling) we can sample
a reliable stratified eval rather than hand-curating 30 items.

Design:

1. **Candidate selection**: visuals with ≥1 mention paragraph whose
   text is specific enough to be a meaningful claim (contains one of
   the verbs in ``_SPECIFIC_VERBS``: "shows", "illustrates", etc.).
2. **Sentence extraction**: pull the sentence containing the figure
   reference from the mention paragraph. Strip the ``Fig. N`` /
   ``Figure N`` text so the query simulates the writer's draft prose
   (which would use a bare claim, not a figure reference).
3. **Stratification**: by visual kind (figure vs chart vs table vs
   equation) and by sentence-type heuristics (evidentiary / method /
   illustrative), so ``sample_eval_set(n=30)`` gets a balanced set.
4. **Ground truth**: the visual_id the mention paragraph was stored
   against. Single-correct-answer eval (no P@K ambiguity, though
   ``R@k`` can still be evaluated since often multiple figures in the
   same paper would be partially-correct).

Public API:

    mine_eval_items(limit=200, min_sentence_chars=60, ...) -> list[EvalItem]
    sample_stratified(items, n=30, seed=0) -> list[EvalItem]
    run_eval(items, top_k=5, candidate_k=15) -> EvalReport

Run once to produce an eval file, then re-run the ranker against that
file as the code evolves. Mining is deterministic when seeded (stable
sort + seeded sample) so a bench run is reproducible.
"""
from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Iterable

from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)


# ── Heuristic filters ──────────────────────────────────────────────

# Verbs that typically appear in a specific, evidence-bearing mention.
# "Fig. 3 is a schematic" is weaker evidence than "Fig. 3 shows the
# trend…" — the former doesn't commit the author to a specific claim.
_SPECIFIC_VERBS = (
    r"shows?", r"illustrates?", r"depicts?", r"reveals?", r"displays?",
    r"presents?", r"demonstrates?", r"summariz(?:es?|ed)", r"plots?",
    r"indicates?", r"reports?", r"compar(?:es?|ed)", r"gives?",
)

_SPECIFIC_RE = re.compile(
    r"\b(?:" + "|".join(_SPECIFIC_VERBS) + r")\b",
    re.IGNORECASE,
)

# Fig/Table/Eq reference — keep in sync with visuals_mentions._REF_RE
# but a simpler variant for *stripping* (which doesn't care about
# precision-vs-recall trade-offs that matter during linking).
_STRIP_REF_RE = re.compile(
    r"\b(?:Fig(?:ure)?s?|Tab(?:le)?s?|Eq(?:uation)?s?)"
    r"\s*\.?\s*\d+(?:\s*[a-z])?(?:\s*\([a-z,\-– ]+\))?\b",
    re.IGNORECASE,
)


# Sentence splitter — deliberately simple. Scientific prose has lots
# of abbreviations that break naive sentence splitting (e.g. "Fig. 3.");
# we split on the ``. `` / ``.\n`` / ``! `` / ``? `` bigrams, which
# catches most cases without splitting on inline abbreviations.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


# Sentence-type classifier. Used for stratified sampling so the final
# 30-item set isn't dominated by one mode. Cheap regex check — the
# point is to diversify the sample, not to label-train anything.

_METHODS_HINTS = re.compile(
    r"\b(?:we\s+(?:use|apply|run|fit|compute|calculate|perform)|"
    r"method|methodology|protocol|procedure|algorithm|setup|"
    r"technique|approach|workflow)\b",
    re.IGNORECASE,
)
_ILLUSTRATIVE_HINTS = re.compile(
    r"\b(?:schematic|diagram|overview|cartoon|illustration|"
    r"conceptual|representation|sketch|flowchart)\b",
    re.IGNORECASE,
)


def _classify_sentence(sentence: str) -> str:
    """One of {"methods", "illustrative", "evidentiary"}. Default is
    evidentiary — most body-text figure mentions are in Results or
    Discussion, asserting that the figure demonstrates some finding."""
    if _METHODS_HINTS.search(sentence):
        return "methods"
    if _ILLUSTRATIVE_HINTS.search(sentence):
        return "illustrative"
    return "evidentiary"


# ── Eval item + report ─────────────────────────────────────────────


@dataclass
class EvalItem:
    """One ground-truth (sentence → correct_visual_id) pair.

    ``raw_sentence`` preserves the original body-text sentence (with
    the Fig. N reference) for audit + display; ``query_sentence`` is
    the stripped version passed to the ranker to simulate a writer's
    draft claim.
    """
    document_id: str
    visual_id: str
    visual_kind: str
    visual_figure_num: str
    raw_sentence: str
    query_sentence: str
    sentence_type: str             # "evidentiary" | "methods" | "illustrative"
    paper_title: str = ""


@dataclass
class EvalReport:
    """Aggregate + per-item results from running the ranker on a set."""
    n_items: int
    n_top1_correct: int
    n_top3_correct: int
    n_same_paper_top1: int           # top-1 from the same paper (sanity)
    mean_composite_top1: float
    mean_composite_correct: float    # composite on items where the correct answer was found somewhere in top_k
    per_item: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def p_at_1(self) -> float:
        return self.n_top1_correct / self.n_items if self.n_items else 0.0

    @property
    def r_at_3(self) -> float:
        return self.n_top3_correct / self.n_items if self.n_items else 0.0

    @property
    def same_paper_rate(self) -> float:
        return self.n_same_paper_top1 / self.n_items if self.n_items else 0.0


# ── Mining ─────────────────────────────────────────────────────────


def mine_eval_items(
    limit: int = 500,
    min_sentence_chars: int = 60,
    max_sentence_chars: int = 350,
) -> list[EvalItem]:
    """Scan the corpus's linked visuals, extract sentences that cite a
    specific figure via a specific verb, return as EvalItem list.

    ``limit`` caps the number of items mined (stability; avoid a
    full-corpus sweep when we only need a few hundred). ``min_`` and
    ``max_sentence_chars`` drop extreme sentences: too short → probably
    a TOC line or heading fragment, too long → multiple claims bundled.
    """
    from sciknow.storage.db import get_session
    from sciknow.core import visuals_mentions as vm
    out: list[EvalItem] = []

    with get_session() as session:
        rows = session.execute(sql_text("""
            SELECT v.id::text AS visual_id,
                   v.document_id::text AS document_id,
                   v.kind AS visual_kind,
                   COALESCE(v.figure_num, '') AS visual_figure_num,
                   v.mention_paragraphs,
                   COALESCE(pm.title, '') AS paper_title
            FROM visuals v
            LEFT JOIN paper_metadata pm ON pm.document_id = v.document_id
            WHERE v.mention_paragraphs IS NOT NULL
              AND jsonb_array_length(v.mention_paragraphs) > 0
            ORDER BY v.id
        """)).mappings().all()

    for r in rows:
        if len(out) >= limit:
            break
        target_num = vm._parse_figure_number(r["visual_figure_num"])
        if target_num is None:
            continue
        mps = r["mention_paragraphs"] or []
        for mp in mps:
            text_block = (mp.get("text") or "").strip()
            if not text_block:
                continue
            # Find the sentence that explicitly cites this figure via
            # a specific verb.
            sentences = _SENTENCE_SPLIT_RE.split(text_block)
            for raw in sentences:
                raw = raw.strip()
                if not (min_sentence_chars <= len(raw) <= max_sentence_chars):
                    continue
                if not _SPECIFIC_RE.search(raw):
                    continue
                # Must reference this particular figure number
                # (the mention paragraph could contain refs to several)
                if not _has_ref_for_num(raw, target_num, r["visual_kind"]):
                    continue
                stripped = _STRIP_REF_RE.sub("", raw)
                stripped = re.sub(r"\(\s*\)", "", stripped)  # clean "()" leftovers
                stripped = re.sub(r"\s+", " ", stripped).strip().strip(",.")
                if len(stripped) < 40:
                    continue
                out.append(EvalItem(
                    document_id=r["document_id"],
                    visual_id=r["visual_id"],
                    visual_kind=r["visual_kind"],
                    visual_figure_num=r["visual_figure_num"],
                    raw_sentence=raw,
                    query_sentence=stripped,
                    sentence_type=_classify_sentence(stripped),
                    paper_title=r["paper_title"],
                ))
                break  # one item per mention paragraph
            if len(out) >= limit:
                break
    return out


def _has_ref_for_num(text: str, target_num: int, visual_kind: str) -> bool:
    """Does this sentence reference figure number ``target_num`` with a
    keyword that matches the visual's kind?"""
    from sciknow.core import visuals_mentions as vm
    for m in vm._REF_RE.finditer(text):
        ref_kind = m.group("kind")
        if not vm._kind_matches(visual_kind, ref_kind):
            continue
        try:
            if int(m.group("num")) == target_num:
                return True
        except (TypeError, ValueError):
            continue
    return False


# ── Stratified sampling ────────────────────────────────────────────


def sample_stratified(
    items: list[EvalItem],
    n: int = 30,
    seed: int = 0,
) -> list[EvalItem]:
    """Sample ``n`` items with roughly equal counts across sentence
    types. Deterministic for a given seed.

    When one stratum is thin (e.g. the corpus has few "methods"
    figures), falls back to filling the shortfall with items from the
    largest bucket rather than returning fewer than ``n``.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[EvalItem]] = {}
    for it in items:
        by_type.setdefault(it.sentence_type, []).append(it)
    # Stable order before shuffling
    for k in by_type:
        by_type[k].sort(key=lambda it: (it.document_id, it.visual_id))
        rng.shuffle(by_type[k])

    buckets = list(by_type.keys()) or ["evidentiary"]
    base = n // max(len(buckets), 1)
    picked: list[EvalItem] = []
    for t in buckets:
        picked.extend(by_type.get(t, [])[:base])
    # Fill any shortfall from the largest bucket
    while len(picked) < n:
        largest = max(buckets, key=lambda k: len(by_type.get(k, [])))
        remaining = [it for it in by_type.get(largest, []) if it not in picked]
        if not remaining:
            break
        picked.append(remaining[0])
    return picked[:n]


# ── Harness ────────────────────────────────────────────────────────


def run_eval(
    items: list[EvalItem],
    *,
    top_k: int = 5,
    candidate_k: int = 15,
    use_cited_doc: bool = True,
) -> EvalReport:
    """Run the ranker on every eval item, compare to ground truth.

    ``use_cited_doc=True`` passes the source paper's document_id as
    ``cited_doc_ids`` — this is the realistic case (the writer is
    citing the paper the figure came from). ``use_cited_doc=False``
    ablates signal 2 (same-paper bonus) to isolate the caption +
    mention-paragraph signals' discriminative power.
    """
    from sciknow.retrieval.visuals_ranker import rank_visuals

    n_items = len(items)
    if n_items == 0:
        return EvalReport(0, 0, 0, 0, 0.0, 0.0, [], 0.0)

    n_top1 = 0
    n_top3 = 0
    n_same_paper_top1 = 0
    sum_composite_top1 = 0.0
    sum_composite_correct = 0.0
    per_item: list[dict] = []

    t0 = time.monotonic()
    for it in items:
        section_for_prior = {
            "methods": "methods",
            "illustrative": "introduction",
            "evidentiary": "results",
        }.get(it.sentence_type, "results")

        ranked = rank_visuals(
            it.query_sentence,
            cited_doc_ids=[it.document_id] if use_cited_doc else [],
            section_type=section_for_prior,
            candidate_k=candidate_k,
            top_k=top_k,
        )
        correct_rank = None
        for i, r in enumerate(ranked):
            if r.visual_id == it.visual_id:
                correct_rank = i + 1
                break

        top1 = ranked[0] if ranked else None
        is_top1_correct = (correct_rank == 1)
        is_top3_correct = (correct_rank is not None and correct_rank <= 3)
        same_paper_top1 = bool(top1 and top1.document_id == it.document_id)

        if is_top1_correct:
            n_top1 += 1
        if is_top3_correct:
            n_top3 += 1
        if same_paper_top1:
            n_same_paper_top1 += 1
        if top1:
            sum_composite_top1 += top1.composite_score
        if correct_rank is not None:
            sum_composite_correct += ranked[correct_rank - 1].composite_score

        per_item.append({
            "visual_id": it.visual_id,
            "figure_num": it.visual_figure_num,
            "sentence_type": it.sentence_type,
            "query": it.query_sentence[:200],
            "correct_rank": correct_rank,
            "top1_vis": top1.visual_id if top1 else None,
            "top1_fig": top1.figure_num if top1 else None,
            "top1_composite": (top1.composite_score if top1 else 0.0),
            "same_paper_top1": same_paper_top1,
        })

    elapsed = time.monotonic() - t0
    return EvalReport(
        n_items=n_items,
        n_top1_correct=n_top1,
        n_top3_correct=n_top3,
        n_same_paper_top1=n_same_paper_top1,
        mean_composite_top1=round(sum_composite_top1 / n_items, 4),
        mean_composite_correct=round(sum_composite_correct / max(n_top3, 1), 4),
        per_item=per_item,
        elapsed_s=round(elapsed, 1),
    )
