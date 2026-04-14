"""Phase 53 — "four conditions" refinement gate for autowrite.

Port of autoreason (Nous Research 2026) Section 2's most transferable
claim: iterative refinement loops are only useful when the task has
all four of these properties:

  1. **external verification**    — something beyond the model's priors
                                    can tell right from wrong (for
                                    sciknow, this is retrieval hits).
  2. **constrained scope**        — the target has a clear finite shape
                                    (for sciknow, target_words).
  3. **structured reasoning**     — the task has compositional structure
                                    the model can grip (for sciknow,
                                    an explicit outline / section type).
  4. **sufficient decision space** — there's meaningful room between
                                    candidate answers (too-short sections
                                    collapse to one-right-answer).

When even one condition fails, autoreason Table 12 shows that iterative
refinement can *degrade* output (59–70% word-count loss in their Haiku
3.5 baselines). Sciknow's `autowrite_section_stream` currently runs
the same expensive loop on every section regardless, which wastes
LLM compute and sometimes makes things worse.

This module is a *predicate*. It returns (should_refine, reason) so
the caller can:

  - emit a warning event and proceed (warning-only mode), or
  - skip the revision loop entirely (hard-skip mode).

Initial rollout is warning-only — we log and let the loop run — so we
can observe the gate's recommendations against real autowrite
trajectories before hard-skipping.
"""
from __future__ import annotations

from dataclasses import dataclass


# The section types where unbounded scope is most dangerous. Discussion
# and conclusion are the classic "keep talking" failure cases in LLM
# writing — if target_words is None *and* section_type is one of
# these, refinement almost certainly adds bloat rather than content.
_UNBOUNDED_SECTION_TYPES: frozenset[str] = frozenset({
    "discussion", "conclusion",
})

# Below this many retrieval hits the system has essentially no
# external verification signal — `_verify_draft_inner` can't ground
# claims, `_score_draft_inner`'s groundedness dimension collapses to
# judging the prose in isolation.
_MIN_RETRIEVAL_HITS_FOR_REFINEMENT = 3


@dataclass(frozen=True)
class GateDecision:
    """Result of a gate call. `recommend_refinement` is authoritative;
    `reasons` is a list of {condition_name: failure_message} pairs for
    any failed conditions, empty when all four pass."""
    recommend_refinement: bool
    reasons: list[tuple[str, str]]

    def summary(self) -> str:
        if self.recommend_refinement:
            return "refinement recommended — all four conditions met"
        parts = [f"{name}: {msg}" for name, msg in self.reasons]
        return "refinement not recommended — " + "; ".join(parts)


def should_run_refinement(
    *,
    section_type: str | None,
    target_words: int | None,
    num_retrieval_hits: int,
    has_explicit_outline: bool,
) -> GateDecision:
    """Evaluate the four autoreason conditions against a single
    autowrite section about to enter its review/revise loop.

    All four conditions must pass for refinement to be recommended.
    Failures are collected so the caller can log the full picture
    (useful in the warning-only rollout phase).
    """
    failures: list[tuple[str, str]] = []

    # 1. external verification — enough retrieval hits that groundedness
    # / claim verification have something to compare against.
    if num_retrieval_hits < _MIN_RETRIEVAL_HITS_FOR_REFINEMENT:
        failures.append((
            "external_verification",
            f"only {num_retrieval_hits} retrieval hit(s) — need "
            f"≥ {_MIN_RETRIEVAL_HITS_FOR_REFINEMENT} for groundedness "
            "scoring to have signal",
        ))

    # 2. constrained scope — target_words is set, OR the section is a
    # naturally-bounded type (abstract, methods). Unbounded discussion
    # / conclusion sections without a target_words hit this branch.
    sec = (section_type or "").lower()
    if target_words is None and sec in _UNBOUNDED_SECTION_TYPES:
        failures.append((
            "constrained_scope",
            f"section_type={sec!r} has no natural length bound and "
            "target_words is unset — the loop will keep expanding",
        ))

    # 3. structured reasoning — there's an explicit outline / plan the
    # writer is executing. Sciknow captures this via the section having
    # a plan (TreeWriter sentence-level or chapter-level outline).
    if not has_explicit_outline:
        failures.append((
            "structured_reasoning",
            "no explicit outline / plan — refinement has nothing "
            "compositional to grip; iteration tends to drift",
        ))

    # 4. sufficient decision space — the target is large enough that
    # there's room for the revision to meaningfully differ from the
    # incumbent. Below ~120 words, revisions converge to a single
    # "right" wording and the loop runs unproductively. Skip this
    # check when target_words is None (covered by #2).
    if target_words is not None and target_words < 120:
        failures.append((
            "decision_space",
            f"target_words={target_words} is small — revisions collapse "
            "to one wording, iterating can't meaningfully improve",
        ))

    return GateDecision(
        recommend_refinement=not failures,
        reasons=failures,
    )
