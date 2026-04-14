"""Phase 53 — bootstrap CIs + McNemar's test for bench / HITL evals.

Port of autoreason's `experiments/v2/compute_stats.py`. Keeps the
interface minimal — two functions the rest of `sciknow/testing/` and
`db expand`'s HITL path can call without pulling in a stats framework.

Why we want it: `sciknow test` / `sciknow bench` currently report raw
pass/fail counts with no uncertainty. For A/B comparisons (e.g. "is
bge-reranker-v2-m3 actually better than no rerank?", "did the Phase 51
multi-signal enrich matcher raise the DOI-match rate?") we want:

  1. a confidence interval on each arm's success rate, and
  2. a paired significance test when the same items were run through
     both arms.

Bootstrap CIs handle (1) for any resample-able statistic. McNemar's
handles (2) for binary paired outcomes.

Zero new dependencies — scipy is already transitive via the numpy
stack (FlagEmbedding, transformers, umap-learn).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence


# ── Bootstrap CI ────────────────────────────────────────────────────


@dataclass
class BootstrapResult:
    """Result of a bootstrap CI estimate."""
    mean: float
    lo: float  # lower CI bound at 1 - confidence
    hi: float  # upper CI bound at 1 - confidence
    n_samples: int
    confidence: float


def bootstrap_ci(
    data: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> BootstrapResult:
    """Percentile-method bootstrap confidence interval.

    Default statistic is the mean — use `statistic=median` or any other
    Sequence→float for different estimands. Resamples with replacement
    `n_resamples` times and takes the empirical (α/2, 1-α/2) quantiles
    as the interval bounds.

    Args:
        data: the observations to resample. Numeric; for binary
            success/failure data pass 0s and 1s.
        statistic: defaults to arithmetic mean.
        n_resamples: bootstrap replicate count. 1000 is the standard
            entry point; 10_000 gives tighter quantile estimates but
            scales linearly.
        confidence: two-sided level; 0.95 → 95% CI.
        seed: RNG seed for determinism in tests and regression runs.
            Pass None for fresh randomness.

    Returns:
        BootstrapResult with (mean, lo, hi, n, confidence). Invariant:
        lo ≤ mean ≤ hi (within float rounding).
    """
    if not data:
        return BootstrapResult(0.0, 0.0, 0.0, 0, confidence)
    stat = statistic or (lambda xs: sum(xs) / len(xs))
    mean = stat(data)
    rng = random.Random(seed)
    n = len(data)
    reps: list[float] = []
    for _ in range(n_resamples):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        reps.append(stat(sample))
    reps.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = int(alpha * len(reps))
    hi_idx = int((1.0 - alpha) * len(reps)) - 1
    lo_idx = max(0, min(lo_idx, len(reps) - 1))
    hi_idx = max(0, min(hi_idx, len(reps) - 1))
    return BootstrapResult(
        mean=float(mean),
        lo=float(reps[lo_idx]),
        hi=float(reps[hi_idx]),
        n_samples=n,
        confidence=confidence,
    )


# ── McNemar's test ──────────────────────────────────────────────────


@dataclass
class McNemarResult:
    """Result of McNemar's paired test. `b` = items A got wrong that B
    got right; `c` = items A got right that B got wrong. `b - c` is
    the raw signed improvement; the test asks whether it's bigger
    than chance under H0 (no difference)."""
    b: int
    c: int
    statistic: float      # χ² (or z² under continuity correction)
    p_value: float
    direction: str        # 'A_better' | 'B_better' | 'tie'


def mcnemar_test(
    paired_outcomes: Sequence[tuple[bool, bool]],
    *,
    continuity_correction: bool = True,
) -> McNemarResult:
    """Exact or asymptotic McNemar's test on paired binary outcomes.

    Args:
        paired_outcomes: list of (A_correct, B_correct) — one tuple
            per item that both arms were evaluated on. Order is
            irrelevant.
        continuity_correction: apply Edwards' continuity correction
            (recommended when b+c is small; default True).

    Returns:
        McNemarResult. Note: when `b + c < 10` the asymptotic χ²
        approximation is shaky; prefer `scipy.stats.binomtest(b, b+c,
        0.5)` for an exact test. We use the asymptotic form here to
        stay dependency-minimal, with the continuity correction as a
        mitigation.
    """
    b = sum(1 for a, bi in paired_outcomes if (not a) and bi)
    c = sum(1 for a, bi in paired_outcomes if a and (not bi))
    if b + c == 0:
        return McNemarResult(b=0, c=0, statistic=0.0, p_value=1.0, direction="tie")
    if continuity_correction:
        stat = ((abs(b - c) - 1) ** 2) / (b + c)
    else:
        stat = ((b - c) ** 2) / (b + c)
    # χ² with 1 dof — use scipy if available; fall back to a simple
    # survival-function approximation otherwise.
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(stat, df=1))
    except Exception:
        # Rough approximation: chi2(1) is the square of a standard
        # normal; P(Z² > s) = 2·Φ(−√s).
        import math
        p = 2.0 * _phi_complement(math.sqrt(max(stat, 0.0)))
    direction = "A_better" if c > b else ("B_better" if b > c else "tie")
    return McNemarResult(
        b=b, c=c, statistic=float(stat), p_value=float(p),
        direction=direction,
    )


def _phi_complement(z: float) -> float:
    """1 - Φ(z) via erfc — used only when scipy is missing."""
    import math
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# ── Convenience: paired success-rate summary ─────────────────────────


@dataclass
class PairedComparison:
    """Bundle of the two statistics the caller usually wants at once:
    a per-arm bootstrap CI + the paired McNemar verdict. Used by
    `sciknow test --compare` and the `db expand` HITL review path."""
    a_rate: BootstrapResult
    b_rate: BootstrapResult
    mcnemar: McNemarResult


def compare_paired_binary(
    paired_outcomes: Sequence[tuple[bool, bool]],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int | None = 42,
) -> PairedComparison:
    """One-stop shop: bootstrap CI on each arm's success rate +
    McNemar's paired test on the pairs. Input is a list of
    (A_correct, B_correct) tuples."""
    a_data = [1.0 if a else 0.0 for a, _ in paired_outcomes]
    b_data = [1.0 if b else 0.0 for _, b in paired_outcomes]
    return PairedComparison(
        a_rate=bootstrap_ci(
            a_data, n_resamples=n_resamples, confidence=confidence, seed=seed,
        ),
        b_rate=bootstrap_ci(
            b_data, n_resamples=n_resamples, confidence=confidence, seed=seed,
        ),
        mcnemar=mcnemar_test(paired_outcomes),
    )
