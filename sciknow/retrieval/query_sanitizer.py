"""Phase 52 — query sanitiser for the retrieval path.

Defends against "system-prompt contamination" on ``/api/ask`` and
``sciknow ask``: an MCP-enabled client can accidentally concatenate a
2000-char system prompt in front of the user's 15-char question, and
the embedder drowns. Recall collapses silently — no error, just
garbage retrievals.

The pattern is ported from mempalace/mempalace's ``query_sanitizer.py``
(Issue #333 there). We keep the API shape (``SanitizeResult`` +
``sanitize_query``) and the 4-step fallback so the tuning from their
longmemeval bench carries over.

Four steps, tried in order:

1. **passthrough** — short clean queries (≤ 400 chars, no system-prompt
   fingerprints) flow through unchanged.
2. **question_extraction** — when the input contains a ``?``, back-walk
   from the last one to the previous sentence boundary and return that
   sentence. Handles the classic "long preamble + actual question"
   shape.
3. **tail_sentence** — take the last 1-2 sentences if no ``?`` was
   present but the string is long. A long input that's purely
   declarative is probably "context + statement of what to find".
4. **tail_truncate** — hard cut to the last ``max_len`` chars. Last
   resort; never leave an absurdly long query going to the embedder.

`method` is always one of those four names so downstream code can log
which path fired (useful for the spans tracer to see whether the
sanitiser is actually saving queries or passing them through).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Fingerprints that identify "this looks like a leaked system prompt"
# even when the text is short. Lower-cased substring match.
_SYSTEM_PROMPT_HINTS: tuple[str, ...] = (
    "you are a",
    "your task is",
    "your role is",
    "<|system|>",
    "### instruction",
    "### system",
    "<system>",
    "assistant:",
    "human:",
)

# Upper bound for "clean" passthrough. Longer than a normal
# conversational query but short enough that if it's still all
# relevant, the embedder handles it fine.
_PASSTHROUGH_MAX_LEN = 400

# Hard ceiling for the final emit. Even after sanitisation, never hand
# the embedder more than this many chars.
_HARD_MAX_LEN = 500


@dataclass(frozen=True)
class SanitizeResult:
    """Return value from ``sanitize_query``.

    * ``clean_query``  — the string to feed to the embedder / retriever.
    * ``method``       — which of the four strategies produced it
                         ("passthrough", "question_extraction",
                         "tail_sentence", or "tail_truncate").
    * ``original_len`` — length of the input we received.
    * ``clean_len``    — length of the output.
    """
    clean_query: str
    method: str
    original_len: int
    clean_len: int


def _looks_like_system_prompt(s: str) -> bool:
    low = s.lower()
    return any(hint in low for hint in _SYSTEM_PROMPT_HINTS)


def _find_last_question(s: str) -> str | None:
    """Return the last question sentence in `s` (from the sentence-
    boundary before the last '?' to that '?'). None if no '?' is
    present or the result is uselessly short."""
    last_q = s.rfind("?")
    if last_q == -1:
        return None
    # Walk backward from last_q to the last sentence boundary
    # (period / newline / bullet / question / exclamation mark).
    start = max(
        s.rfind(".", 0, last_q),
        s.rfind("\n", 0, last_q),
        s.rfind("!", 0, last_q),
        s.rfind("?", 0, last_q),
    )
    start = 0 if start == -1 else start + 1
    candidate = s[start:last_q + 1].strip()
    return candidate if len(candidate) >= 5 else None


def _tail_sentences(s: str, n: int = 2) -> str:
    """Last `n` sentences of `s`, crudely split on .!? boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", s.strip())
    tail = " ".join(parts[-n:]).strip()
    return tail


def sanitize_query(
    query: str,
    *,
    max_len: int = _HARD_MAX_LEN,
) -> SanitizeResult:
    """Run the 4-step fallback. Never raises; always returns a
    SanitizeResult. An empty input yields an empty passthrough."""
    q = (query or "").strip()
    orig_len = len(q)
    if not q:
        return SanitizeResult("", "passthrough", orig_len, 0)

    # Step 1 — passthrough for short, benign queries
    if orig_len <= _PASSTHROUGH_MAX_LEN and not _looks_like_system_prompt(q):
        return SanitizeResult(q, "passthrough", orig_len, orig_len)

    # Step 2 — question extraction
    qs = _find_last_question(q)
    if qs and len(qs) <= max_len:
        return SanitizeResult(qs, "question_extraction", orig_len, len(qs))

    # Step 3 — tail sentences (2 by default)
    tail = _tail_sentences(q, n=2)
    if tail and len(tail) <= max_len:
        return SanitizeResult(tail, "tail_sentence", orig_len, len(tail))

    # Step 4 — hard truncation (last N chars)
    trunc = q[-max_len:].strip()
    return SanitizeResult(trunc, "tail_truncate", orig_len, len(trunc))
