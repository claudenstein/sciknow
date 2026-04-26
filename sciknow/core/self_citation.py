"""Phase 54.6.223 — roadmap 3.6.2: self-citation detection.

Pure-Python helpers for deciding whether one paper's authors overlap
with another's — the signal that tags a citation as a *self-citation*.
No LLM, no API, no fuzzy matching beyond surname + first-initial
normalisation.

Used by ``sciknow corpus flag-self-citations`` (and callable directly
from anywhere that has two author lists in hand, e.g. the
groundedness / overstated-claim passes in the writer).

The detection is deliberately conservative: we match on
``(surname, first_initial)`` pairs so "J. Smith" vs "John Smith" vs
"Smith, J" all collapse to the same key, but "J. Smith" vs "J.A.
Smith" stay distinct (matching too loosely would inflate the
self-cite rate on common surnames). Anyone who wants looser matching
can post-process the ``self_cite_authors`` list.

See ``docs/ROADMAP_INGESTION.md`` §3.6.2.
"""
from __future__ import annotations

import re
import unicodedata


# Accepts "James K. Whittaker", "Whittaker, James K.", "J. K. Whittaker",
# "Whittaker JK", "J.K. Whittaker", "Whittaker J K". Splits on commas
# FIRST to isolate the "Surname, Given" form, then falls back to
# "last token = surname" for "Given Surname" order.
_SPACE_RE = re.compile(r"\s+")
_NONALPHA_HYPHEN_RE = re.compile(r"[^a-z\-]")


def _normalize(name: str) -> str:
    """NFKC + lowercase + strip surrounding whitespace. ASCII-fold
    accents so "Müller" and "Muller" match under the same key."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    # Drop combining marks (accents). ASCII-only survives.
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def surname_key(name: str) -> str | None:
    """Normalise an author-name string to a `(surname, first_initial)`
    key used for self-citation overlap detection.

    Returns None when the input is empty, whitespace-only, or clearly
    not a human name (single character, initials-only, etc.).

    Examples:
      >>> surname_key("James K. Whittaker")
      'whittaker,j'
      >>> surname_key("Whittaker, J. K.")
      'whittaker,j'
      >>> surname_key("Whittaker")
      'whittaker,'
      >>> surname_key("")
      >>> surname_key("et al.")
    """
    if not name:
        return None
    n = _normalize(name)
    # Drop common suffixes / trailing punctuation so "Smith et al." and
    # "Smith, J. et al." both reduce cleanly.
    n = re.sub(r"\bet\.?\s*al\.?\b", "", n).strip()
    n = n.strip(" .,;:-")
    if not n:
        return None

    # Surname, Given form.
    if "," in n:
        surname, given = n.split(",", 1)
    else:
        tokens = _SPACE_RE.split(n)
        if len(tokens) == 1:
            # Single token — surname-only. Rare but possible.
            surname, given = tokens[0], ""
        else:
            # Given Surname — last token = surname.
            surname = tokens[-1]
            given = " ".join(tokens[:-1])

    # Clean surname: lowercase alpha + hyphen only.
    surname = _NONALPHA_HYPHEN_RE.sub("", surname.strip())
    if not surname or len(surname) < 2:
        return None

    # Extract first initial from given (first alphabetic character).
    initial = ""
    for ch in given:
        if ch.isalpha():
            initial = ch
            break

    return f"{surname},{initial}"


def surname_keys_from_authors(authors: list | None) -> set[str]:
    """Extract the set of surname keys from an authors JSONB list.

    Each element is expected to be a dict with a ``name`` field
    (sciknow persists authors as ``[{name, orcid, affiliation}, …]``)
    but plain-string elements are tolerated for robustness.
    """
    if not authors or not isinstance(authors, list):
        return set()
    keys: set[str] = set()
    for a in authors:
        if isinstance(a, dict):
            name = a.get("name") or ""
        else:
            name = str(a)
        k = surname_key(name)
        if k:
            keys.add(k)
    return keys


def detect_self_cite(
    citing_authors: list | None,
    cited_authors: list | None,
) -> tuple[bool | None, list[str]]:
    """Compare the two author lists; return (verdict, overlap_keys).

    Verdict is:
      * True   — at least one (surname, initial) key appears on both sides
      * False  — both lists parsed to non-empty sets and they don't
                 overlap
      * None   — one or both sides yielded an empty key set, so the
                 classifier can't decide (author list missing, all
                 entries unparseable, or "et al." only)

    The returned overlap list preserves insertion order (stable across
    runs for identical inputs).
    """
    citing = surname_keys_from_authors(citing_authors)
    cited = surname_keys_from_authors(cited_authors)
    if not citing or not cited:
        return None, []
    overlap = sorted(citing & cited)
    return (bool(overlap), overlap)
