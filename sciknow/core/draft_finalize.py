"""Phase 56.H — Section-ending safety net.

Always-on post-stream cleanup. Catches mid-sentence / mid-word /
orphan-`[` output regardless of which writer engine produced the
draft. Cheap (regex-only), bounded (15% retention floor on the lossy
slice fallback), and idempotent (clean text round-trips unchanged).

Documented in ``docs/research/SECTION_ENDING_RESEARCH.md`` Tier 0.5.
The companion Tier 1 (LLM continuation pass) and Tier 2/3/4 (decode-
time and prompt-level length controls) are deferred to Phase 57; this
module is the cheapest correct fix and runs unconditionally on every
writer output.

Order of operations:
  1. Strip a trailing dangling citation opener (``[``, ``[3``, ``[3,``)
     — incomplete citation markers that biber can't parse.
  2. If the result still doesn't end on a sentence terminator, slice
     back to the last terminator that's followed by whitespace or
     end-of-string.
  3. If the slice would lose more than 15% of the body, keep the
     mid-word original — drastically lossy is worse than visible-but-
     repairable.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# Sentence-terminating characters. Includes Unicode close-quotes so
# ``...statement.”`` is recognised as clean.
_SENT_END_CHARS = frozenset({".", "!", "?", "\"", "'", ")", "]", "”", "’", "»"})


# Trailing dangling citation patterns:
#   ``[``           — empty opener
#   ``[3``          — opener with one number, no close
#   ``[3, 5``       — opener with multiple numbers, no close
#   ``[3,``         — opener with trailing comma
# All of these are syntactically invalid citation markers that biber
# rejects and the GUI styles as broken. Strip them at the very end of
# the body.
_DANGLING_CITE_RE = re.compile(r"\[[\d,\s]*\Z")


# Sentence-terminator at end-of-string OR followed by whitespace/quote.
# Used by _slice_to_last_sentence to find the rightmost legitimate stop.
_SENT_BOUNDARY_RE = re.compile(r"[.!?](?=[\s\"'”’)»]|\Z)")


# Minimum retention fraction for the slice fallback. If slicing would
# remove more than (1 - this) of the body, we keep the mid-word
# original because the user-facing damage of an empty section is
# worse than a visible-but-fixable cutoff.
_RETENTION_FLOOR = 0.85


def _ends_cleanly(text: str) -> bool:
    """True iff the body ends on a sentence terminator (or is empty).

    Walks back over trailing whitespace before looking at the last
    glyph, so ``"...statement.\\n\\n"`` counts as clean.
    """
    if not text:
        return True
    s = text.rstrip()
    if not s:
        return True
    return s[-1] in _SENT_END_CHARS


def _strip_dangling_citation_opener(text: str) -> str:
    """Drop a trailing ``[`` / ``[N`` / ``[N,`` from the very end.

    Only removes the dangling opener; whitespace before it is
    preserved on purpose so a subsequent slice has something to walk.
    Idempotent on text without a dangling opener.
    """
    if not text:
        return text
    s = text.rstrip()
    m = _DANGLING_CITE_RE.search(s)
    if m and m.start() > 0:
        return s[: m.start()].rstrip()
    return s


def _slice_to_last_sentence(text: str) -> str:
    """Trim back to the rightmost sentence terminator that's at end-
    of-string or followed by whitespace/quote.

    Returns ``text`` unchanged if no boundary is found at all (better
    to keep a partial than emit empty). Trailing close-quote /
    close-paren are included in the cut so ``...statement.” `` keeps
    the closing quote.
    """
    if not text:
        return text
    s = text.rstrip()
    # Find every sentence boundary; take the last one.
    best = None
    for m in _SENT_BOUNDARY_RE.finditer(s):
        best = m
    if best is None:
        return text  # no boundary; keep as-is
    end_idx = best.end()  # one past the terminator
    # Include immediate trailing close-quote / close-paren.
    while end_idx < len(s) and s[end_idx] in {"\"", "'", ")", "]", "”", "’", "»"}:
        end_idx += 1
    return s[:end_idx]


def ensure_clean_ending(content: str) -> str:
    """Two-pass cleanup that returns content ending on a sentence
    terminator, or the original (mid-word / mid-sentence) if cleaning
    would lose more than 15% of the body.

    Sequence:
      1. ``_strip_dangling_citation_opener`` — invalid ``[`` / ``[N`` /
         ``[N,`` openers are always safe to drop.
      2. If still not clean, ``_slice_to_last_sentence``.
      3. If the slice retention falls below the floor, return the
         partially-cleaned content (dangling cite removed, but
         possibly mid-word).
    """
    if not content:
        return content

    cleaned = _strip_dangling_citation_opener(content)
    if _ends_cleanly(cleaned):
        return cleaned

    sliced = _slice_to_last_sentence(cleaned)
    if not sliced:
        return cleaned
    # Retention floor: keep the slice only if it preserves enough body.
    if len(sliced) >= int(len(cleaned) * _RETENTION_FLOOR):
        return sliced
    return cleaned
