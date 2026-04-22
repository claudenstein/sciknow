"""Phase 54.6.220 — roadmap 3.7.2: KG relation vocabulary stabilization.

Collapses free-text LLM-extracted predicates onto a closed vocabulary
of ~18 relations. Complements the entity canonicalization shipped in
54.6.209 (``kg_canonicalize.py``): one stabilises node names, the
other stabilises edge types. Together they turn the
``knowledge_graph`` table from "500 unique predicate strings" into
"20 buckets" — queryable by relation type, consistent across papers,
and interpretable in the graph UI.

Design choices:

* **Closed vocabulary, not free-text.** Per-paper the extractor
  already produces fairly consistent vocab; the drift happens across
  papers because "causes" / "caused by" / "leads to" / "drives" /
  "triggers" are all the same relation at any useful level of
  analysis. We pick one canonical name per bucket and map aliases.
* **Climate-science-first, not generic-ontology.** The corpus is
  climate research; relations like ``PROXIES_FOR`` and
  ``RECONSTRUCTS`` matter here but wouldn't in a generic ontology.
  Generic fallbacks (``ASSOCIATED_WITH``) catch the tail.
* **Lowercase storage.** Matches ``wiki_ops.py`` existing
  normalisation at line 773 (``.lower()``). Canonical names are
  stored ``snake_case`` for consistency.
* **Pass-through on no match.** Unknown predicates round-trip
  unchanged rather than collapsing to a generic fallback bucket —
  losing the original signal would be worse than carrying a
  low-frequency tail.

See ``docs/ROADMAP_INGESTION.md`` §3.7.2.
"""
from __future__ import annotations

import re
import unicodedata

# ── Canonical vocabulary ────────────────────────────────────────────────
#
# Keys are the canonical relation names; values are every alias we want
# to collapse INTO that canonical name. All strings lowercase; the
# canonical name itself is included in its own alias list so a paper
# that already uses the canonical form round-trips losslessly.
#
# When adding a new relation, also consider updating
# ``rag/wiki_prompts.py::KG_EXTRACT_SYSTEM`` so the extractor is
# aware of it at prompt time — otherwise the LLM keeps emitting
# synonyms that hit the alias table rather than the canonical name,
# and the alias list just grows.

_CANONICAL: dict[str, tuple[str, ...]] = {
    # ── Causal / forcing ───────────────────────────────────────────────
    "forces": (
        "forces", "forcing", "drives", "causes", "caused by",
        "causes of", "triggers", "induces", "leads to", "results in",
        "produces",
    ),
    "responds_to": (
        "responds to", "response to", "affected by", "influenced by",
        "sensitive to", "depends on",
    ),
    "correlates_with": (
        "correlates with", "correlated with", "correlation with",
        "co-varies with", "covaries with", "associated with",
        "linked to", "co-occurs with", "coincides with",
    ),
    # ── Evidence + claims ──────────────────────────────────────────────
    "supports": (
        "supports", "provides evidence for", "evidence for",
        "consistent with", "confirms", "agrees with", "validates",
    ),
    "contradicts": (
        "contradicts", "refutes", "inconsistent with", "disagrees with",
        "challenges", "counters", "counter-evidence for",
        "contradicted by",
    ),
    "neutral_on": (
        "neutral on", "mentions", "discusses", "references",
        "touches on",
    ),
    # ── Proxy / reconstruction (climate-specific) ──────────────────────
    "proxies_for": (
        "proxies for", "proxy for", "is proxy for", "serves as proxy",
        "indicates", "represents",
    ),
    "reconstructs": (
        "reconstructs", "reconstruction of", "reconstructed from",
        "derived from proxy", "inferred from",
    ),
    # ── Measurement / data ─────────────────────────────────────────────
    "measures": (
        "measures", "measurement of", "quantifies",
        "records", "monitors",
    ),
    "observes": (
        "observes", "observations of", "detects", "sees",
    ),
    # ── Methods ────────────────────────────────────────────────────────
    "uses_method": (
        "uses_method", "uses method", "applies", "employs",
        "uses the", "using", "adopts", "based on",
    ),
    "applied_to": (
        "applied to", "applied on", "used on", "used for",
    ),
    "predicts": (
        "predicts", "forecasts", "projects", "simulates",
        "estimates", "models",
    ),
    # ── Structural / ontological ───────────────────────────────────────
    "part_of": (
        "part of", "component of", "belongs to", "subsystem of",
        "member of", "includes", "includes the", "contains",
    ),
    "related_to": (
        "related to", "relation to", "relates to",
        "linked to concept", "connected to",
    ),
    "has_property": (
        "has property", "characterised by", "characterized by",
        "properties", "property of",
    ),
    # ── Meta / citations ───────────────────────────────────────────────
    "cites_data": (
        "cites data", "uses data from", "dataset from", "data from",
    ),
    "cites_method": (
        "cites method", "methodology from", "method from",
    ),
    # ── Paper-level actions (from the current extraction prompt) ───────
    "studies": (
        "studies", "investigates", "examines", "analyses",
        "analyzes", "explores",
    ),
    "finds": (
        "finds", "reports", "concludes", "demonstrates", "shows",
        "result", "finding",
    ),
}


# Build the reverse index once at import time. `_ALIAS_TO_CANONICAL`
# maps each normalised alias → the canonical form. O(1) lookup in the
# normalization loop.
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in _CANONICAL.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias.lower()] = _canonical


# ── Normalisation ──────────────────────────────────────────────────────


_WS_RE = re.compile(r"\s+")
_UNDERSCORE_RE = re.compile(r"[_]+")


def _normalize(raw: str) -> str:
    """Light pre-processing before alias lookup.

    Lowercases, NFKC-normalises unicode (rare for predicates but
    matches entity-canonicalizer behaviour for symmetry), strips
    surrounding whitespace, collapses internal whitespace runs to
    single spaces, preserves underscores so ``uses_method`` matches
    ``uses_method`` directly without needing an alias entry.
    """
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", str(raw)).strip().lower()
    s = _WS_RE.sub(" ", s)
    return s


def canonicalize_relation(raw: str) -> str:
    """Map a free-text predicate to its canonical form.

    Returns the canonical relation name when the alias table has a
    match, otherwise returns the input normalised (lowercased,
    whitespace-collapsed) so unknown predicates still round-trip
    consistently without getting dropped. Callers who want to
    distinguish "was canonical" from "was passed through unchanged"
    can compare against the original input.

    Examples:

      >>> canonicalize_relation("leads to")
      'forces'
      >>> canonicalize_relation("Inconsistent With")
      'contradicts'
      >>> canonicalize_relation("unknown-relation-XYZ")
      'unknown-relation-xyz'
    """
    norm = _normalize(raw)
    if not norm:
        return ""
    # Exact alias hit (fast path).
    hit = _ALIAS_TO_CANONICAL.get(norm)
    if hit:
        return hit
    # Underscore-insensitive retry: "uses_method" → "uses method"
    hit2 = _ALIAS_TO_CANONICAL.get(_UNDERSCORE_RE.sub(" ", norm))
    if hit2:
        return hit2
    # No match — pass through normalised. Preserves the original
    # signal; `wiki kg-sample` can catch low-frequency tails later.
    return norm


def canonical_relations() -> tuple[str, ...]:
    """Sorted tuple of canonical relation names.

    Public API for the KG_EXTRACT_SYSTEM prompt + any UI that wants
    to enumerate the valid relation buckets.
    """
    return tuple(sorted(_CANONICAL.keys()))
