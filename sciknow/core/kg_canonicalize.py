"""Phase 54.6.209 — roadmap 3.7.1: rule-based KG entity canonicalization.

Loads the ``kg_aliases.yaml`` alias table shipped next to this module
and exposes a single ``canonicalize(name)`` function used by the
`db extract-kg` / `wiki compile` path to collapse surface variants
("CO2", "CO₂", "carbon dioxide", "Carbon Dioxide") onto one canonical
form before insertion into the ``knowledge_graph`` table.

Deliberately cheap and deterministic — no LLM, no web lookup, no
outbound I/O. The rule file is user-editable in the repo; after
adding new aliases, re-run ``sciknow wiki compile --rebuild`` to
re-populate ``knowledge_graph`` with the updated mapping.

See ``docs/roadmap/ROADMAP_INGESTION.md`` §3.7.1 for the research context
and next-step Wikidata grounding option.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_ALIASES_FILE = Path(__file__).with_suffix(".yaml").with_stem("kg_aliases")


# ── Normalisation ────────────────────────────────────────────────────


_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[\s.,;:!?\-_]+$")


def _normalize(name: str) -> str:
    """Light pre-processing before alias lookup.

    Lowercases, NFKC-normalises unicode (so ``CO₂`` becomes ``CO2``
    in the lookup key — the alias file also lists both), collapses
    whitespace, strips trailing punctuation/hyphens. Empty input
    maps to empty output; the caller decides whether to drop or keep
    an empty entity.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", str(name)).strip().lower()
    s = _WS_RE.sub(" ", s)
    s = _TRAILING_PUNCT_RE.sub("", s)
    return s


# ── Alias loader (cached — the YAML parse is the only non-trivial cost) ────


@lru_cache(maxsize=1)
def _load_alias_map() -> dict[str, str]:
    """Return ``{variant → canonical}`` as a flat dict.

    The YAML is keyed by canonical form; this loader flattens each
    variant (including the canonical form itself) onto the canonical
    form so a single dict lookup covers both. Keys are stored
    normalised via ``_normalize``.
    """
    try:
        import yaml
    except ImportError:
        logger.warning(
            "pyyaml not installed — KG entity canonicalization will "
            "act as an identity function. Run `uv add pyyaml` to "
            "enable the alias table."
        )
        return {}
    if not _ALIASES_FILE.exists():
        logger.warning(
            "kg_aliases.yaml missing at %s — canonicalization disabled.",
            _ALIASES_FILE,
        )
        return {}
    try:
        with _ALIASES_FILE.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning(
            "Failed to load %s (%s) — canonicalization disabled.",
            _ALIASES_FILE, exc,
        )
        return {}
    out: dict[str, str] = {}
    for canonical, variants in raw.items():
        canon_norm = _normalize(canonical)
        if not canon_norm:
            continue
        # The canonical form matches itself.
        out[canon_norm] = canon_norm
        if not isinstance(variants, list):
            continue
        for v in variants:
            key = _normalize(v)
            if key:
                out[key] = canon_norm
    return out


# ── Public API ───────────────────────────────────────────────────────


def canonicalize(name: str) -> str:
    """Return the canonical form of ``name`` if the alias file knows
    it, otherwise the original string lower-cased + whitespace-
    normalised. Empty input → empty output.
    """
    norm = _normalize(name)
    if not norm:
        return ""
    return _load_alias_map().get(norm, norm)


def canonicalize_count() -> int:
    """How many entries are in the loaded alias map. Useful for
    smoke tests + observability — a zero value means the YAML
    didn't load (missing file, pyyaml not installed, parse error)
    so canonicalize() is effectively an identity function.
    """
    return len(_load_alias_map())


def reload_aliases() -> None:
    """Drop the cached alias map so the next ``canonicalize`` call
    re-reads the YAML. Primarily for tests and for the user who
    just edited the file in a long-running REPL.
    """
    _load_alias_map.cache_clear()
