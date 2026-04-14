"""Phase 53 — length-controlled pairwise evaluation.

Port of autoreason's Section-7.7 / Table-17 technique: when
comparing multiple candidate drafts or answer versions, truncate
them all to the median word count *before* handing them to the
judge, so any quality preference can't be explained by mere
verbosity.

Why this matters for sciknow: the current scorer in
`sciknow/core/book_ops.py::_score_draft_inner` can trivially
prefer longer drafts because longer drafts cover more topics →
higher completeness score. That's a known failure mode of
LLM-as-judge (length bias — see e.g. Zheng et al. 2023 arXiv
2306.05685 and the pair of 2407.01085 / 2310.10076 that
autoreason cites). Length-controlled eval is the standard
antidote: before you ask the judge which candidate is better,
equalise their lengths.

This module is a `bench`-time / debug utility — it's NOT
called from the autowrite hot path. Call it when you want to
do a rigorous head-to-head comparison (e.g. "does revision-v2
actually beat revision-v1 once we control for length?").
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass


def _word_count(text: str) -> int:
    """Whitespace-split word count — same convention as the rest of
    sciknow's length accounting."""
    return len([w for w in (text or "").split() if w])


def truncate_to_words(text: str, n_words: int) -> str:
    """Keep the first `n_words` whitespace-separated tokens of `text`.
    Joins back with single spaces — paragraph breaks are lost but
    that's fine for judge-pairwise scoring where formatting isn't
    what's being evaluated."""
    if n_words <= 0:
        return ""
    tokens = (text or "").split()
    if len(tokens) <= n_words:
        return text or ""
    return " ".join(tokens[:n_words])


@dataclass
class LengthControlledPair:
    """One candidate after length-control preprocessing."""
    label: str
    original_words: int
    clipped_words: int
    clipped_text: str


def equalize_lengths(candidates: list[tuple[str, str]]) -> list[LengthControlledPair]:
    """Truncate every candidate to the *median* word count of the set.

    Median (not min) is intentional — using the min would penalise
    the longest candidate disproportionately if there's one outlier
    much shorter than the rest. Median is the standard choice in the
    autoreason Table-17 methodology.

    Args:
        candidates: list of (label, text) tuples. Order is preserved
            in the output.

    Returns:
        list of LengthControlledPair, one per input, in the same
        order, with `clipped_text` guaranteed to have ≤ median words.
    """
    if not candidates:
        return []
    counts = [_word_count(t) for _, t in candidates]
    median = int(statistics.median(counts))
    out: list[LengthControlledPair] = []
    for (label, text), n in zip(candidates, counts):
        clipped = truncate_to_words(text, median) if n > median else text
        out.append(LengthControlledPair(
            label=label,
            original_words=n,
            clipped_words=_word_count(clipped),
            clipped_text=clipped,
        ))
    return out


@dataclass
class LengthControlReport:
    """Summary of a length-control pass — what was trimmed and by
    how much. The caller does the actual pairwise scoring (on the
    `clipped_text` fields) separately; this module only normalises
    the inputs."""
    median_words: int
    entries: list[LengthControlledPair]
    max_original_words: int
    min_original_words: int
    trimmed_count: int  # how many candidates were actually shortened


def compare_at_matched_length(
    candidates: list[tuple[str, str]],
) -> LengthControlReport:
    """Build a report suitable for handing downstream to a scorer /
    judge panel. Does NOT call any LLM — the actual judging is a
    separate concern the caller can do via `sciknow ask` / book_ops
    / a local judge prompt."""
    entries = equalize_lengths(candidates)
    if not entries:
        return LengthControlReport(0, [], 0, 0, 0)
    originals = [e.original_words for e in entries]
    trimmed = sum(1 for e in entries if e.clipped_words < e.original_words)
    return LengthControlReport(
        median_words=int(statistics.median(originals)),
        entries=entries,
        max_original_words=max(originals),
        min_original_words=min(originals),
        trimmed_count=trimmed,
    )
