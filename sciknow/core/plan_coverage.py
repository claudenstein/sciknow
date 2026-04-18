"""Phase 54.6.79 (#6) — plan-coverage score for the autowrite loop.

Until today, the autowrite scorer judged a draft along dimensions
(groundedness, citation accuracy, coherence, hedging, length) but not
*"did this draft cover what the section plan actually asked for?"*.
A draft can score 0.9 on every prose dimension while silently
skipping the third of four bullets in its plan — and the loop happily
terminates.

This module supplies that missing dimension. Given the section
plan text (free-form paragraph the user wrote in the Plan modal) and
the draft prose, it:

1. **Atomizes** the plan into sentence-level "bullets" — cheap split
   on ``[.!?]`` boundaries with a fallback to the whole paragraph
   when the plan is a single sentence.
2. **Entails** each bullet against the draft via the same NLI
   cross-encoder (``cross-encoder/nli-deberta-v3-base``) used by
   faithfulness scoring and citation alignment — no new model load.
3. Returns a **coverage fraction** (bullets entailed at ≥ 0.5),
   a list of per-bullet scores, and the set of *missed* bullets
   (entailment < 0.5) so the revision prompt can target them.

Integration path: fold the coverage float into the autowrite
``scores`` dict as a new dimension. The existing KEEP/DISCARD +
weakest-dimension logic in ``_autowrite_section_body`` already picks
up any dimension below ``target_score`` and triggers a targeted
revision — no changes needed to the termination decision itself.
The scorer prompt doesn't need to know about coverage at all; we
compute it separately and merge.

Reference: CovScore (Zhang et al., 2024) for the entailment-coverage
approach on summarization. We're applying the same idea to section
drafts.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Thresholds
# ════════════════════════════════════════════════════════════════════════


DEFAULT_ENTAIL_THRESHOLD = 0.5   # per-bullet; entailed iff prob >= this
MIN_BULLET_LENGTH = 15           # chars — skip fragments like "etc." or "i.e."


# ════════════════════════════════════════════════════════════════════════
# Plan atomization
# ════════════════════════════════════════════════════════════════════════


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
# Also split on clear list separators so "covers A; B; C" becomes 3 bullets.
_LIST_SPLIT = re.compile(r"\s*[;•]\s+|\s+\|\s+")


def atomize_plan(plan_text: str) -> list[str]:
    """Return the plan's atomic bullets. Empty list if the plan text is
    blank / too short to form a single bullet.

    Strategy: try sentence split first. If that yields <2 bullets but
    the text contains semicolons or pipes, split on those instead. A
    single remaining short bullet falls back to returning the whole
    plan as one bullet so coverage still gets a score.
    """
    t = (plan_text or "").strip()
    if not t or len(t) < MIN_BULLET_LENGTH:
        return []

    parts = [p.strip() for p in _SENT_SPLIT.split(t) if p.strip()]
    if len(parts) < 2:
        # Try list-style splits
        alt = [p.strip() for p in _LIST_SPLIT.split(t) if p.strip()]
        if len(alt) >= 2:
            parts = alt
    # Drop fragments; always keep at least one bullet.
    parts = [p for p in parts if len(p) >= MIN_BULLET_LENGTH]
    return parts or [t]


# ════════════════════════════════════════════════════════════════════════
# NLI-based coverage score
# ════════════════════════════════════════════════════════════════════════


@dataclass
class CoverageResult:
    coverage: float                   # fraction in [0, 1]
    n_bullets: int
    n_covered: int
    per_bullet: list[dict]            # [{bullet, score, covered}]
    missed_bullets: list[str]         # bullets below threshold

    def as_dict(self) -> dict:
        return {
            "coverage": round(self.coverage, 3),
            "n_bullets": self.n_bullets,
            "n_covered": self.n_covered,
            "per_bullet": self.per_bullet,
            "missed_bullets": self.missed_bullets,
        }


def _nli_entail_probs(pairs: list[tuple[str, str]]) -> list[float]:
    """Delegate to the NLI helper from the quality bench. Lazy imports
    so callers don't pay the 440 MB model load unless coverage is
    actually computed."""
    if not pairs:
        return []
    from sciknow.testing.quality import _nli_entail_probs as _fn
    try:
        return _fn(pairs)
    except Exception as exc:
        logger.warning("NLI unavailable — plan coverage disabled: %s", exc)
        return [0.0] * len(pairs)


def compute_coverage(
    draft_text: str,
    plan_text: str,
    *,
    threshold: float = DEFAULT_ENTAIL_THRESHOLD,
) -> CoverageResult:
    """Return coverage of `plan_text`'s atomic bullets by `draft_text`.

    Coverage = fraction of bullets whose NLI-entailment probability
    against the full draft is >= ``threshold`` (default 0.5, same as
    the faithfulness rule of thumb).
    """
    bullets = atomize_plan(plan_text)
    if not bullets or not (draft_text or "").strip():
        return CoverageResult(
            coverage=1.0,           # vacuous pass — nothing to cover
            n_bullets=0, n_covered=0, per_bullet=[], missed_bullets=[],
        )

    # NLI pairs are (premise=draft, hypothesis=bullet). For a long
    # draft we don't chunk — the NLI cross-encoder pads/truncates to
    # 512 tokens by default, which means very long drafts clip their
    # tail. That's acceptable for a first version; if it matters we'll
    # slide a window like faithfulness_score does.
    pairs = [(draft_text, b) for b in bullets]
    probs = _nli_entail_probs(pairs)

    per_bullet: list[dict] = []
    missed: list[str] = []
    n_covered = 0
    for b, p in zip(bullets, probs):
        covered = bool(p >= threshold)
        if covered:
            n_covered += 1
        else:
            missed.append(b)
        per_bullet.append({
            "bullet": b[:200],
            "score": round(float(p), 3),
            "covered": covered,
        })
    coverage = n_covered / len(bullets)
    return CoverageResult(
        coverage=coverage,
        n_bullets=len(bullets),
        n_covered=n_covered,
        per_bullet=per_bullet,
        missed_bullets=missed,
    )


# ════════════════════════════════════════════════════════════════════════
# Revision-instruction helper (for missed bullets)
# ════════════════════════════════════════════════════════════════════════


def revision_hint_for_misses(missed: list[str]) -> str:
    """Render a revision instruction that names the missed bullets so
    the writer can expand the draft to cover them. Returns empty
    string when there are no misses — caller uses that to decide
    whether to trigger a plan-coverage-driven revision round."""
    if not missed:
        return ""
    if len(missed) == 1:
        return (
            "The section plan calls for this point that the current "
            f"draft doesn't cover: {missed[0]!r}. Expand the draft to "
            "address it grounded in the retrieved sources."
        )
    points = "; ".join(f"'{m}'" for m in missed[:5])
    more = "" if len(missed) <= 5 else f" (+{len(missed) - 5} more)"
    return (
        "The section plan calls for these points that the current draft "
        f"doesn't cover: {points}{more}. Expand the draft to address "
        "each, grounded in the retrieved sources."
    )
