"""Phase 54.6.112 (Tier 1 #5) — per-project venue block/allow lists.

A project-scoped JSON file (``<project root>/venue_config.json``)
carrying two lists of case-insensitive substring patterns:

* ``blocklist`` — extends the built-in predatory/hijacked pattern set
  in ``sciknow/ingestion/expand_filters.py``. A candidate whose
  OpenAlex ``primary_location.source.host_organization_name`` or
  ``publisher`` matches ANY blocklist pattern is hard-dropped from
  the ``db expand`` RRF pool, same as a seed-list predatory hit.

* ``allowlist`` — an escape hatch for legitimate venues whose names
  happen to match a blocklist pattern (e.g. "Frontiers in X" is
  sometimes-predatory, sometimes-legitimate). A candidate whose
  venue matches an allowlist pattern wins over a blocklist hit and
  is kept.

Design rationale (see ``docs/research/EXPAND_ENRICH_RESEARCH_2.md`` §4.4):

- JSON, not the project's ``.env.overlay``, so the data structure is
  richer than string=value AND so it's safe to commit via
  ``sciknow project archive`` without leaking env secrets.
- Per-project, not global, because predatory / "is this venue ok?"
  judgments are field-specific — climate research's dodgy venues
  are not the same as biomedical's.
- Substring-match (case-insensitive) rather than ISSN or regex so the
  user can type ``"david publishing"`` and it Just Works. Regex is
  available as an escape hatch via a ``^…$``-prefixed pattern.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


_FILENAME = "venue_config.json"


@dataclass
class VenueConfig:
    blocklist: list[str] = field(default_factory=list)
    allowlist: list[str] = field(default_factory=list)

    def has_block(self, pattern: str) -> bool:
        p = _normalise(pattern)
        return any(_normalise(x) == p for x in self.blocklist)

    def has_allow(self, pattern: str) -> bool:
        p = _normalise(pattern)
        return any(_normalise(x) == p for x in self.allowlist)

    def matches_block(self, venue: str | None) -> str | None:
        """Return the matching blocklist pattern, or None."""
        return _first_match(venue, self.blocklist)

    def matches_allow(self, venue: str | None) -> str | None:
        """Return the matching allowlist pattern, or None."""
        return _first_match(venue, self.allowlist)


def _normalise(s: str) -> str:
    return (s or "").strip().lower()


def _first_match(venue: str | None, patterns: Iterable[str]) -> str | None:
    v = _normalise(venue or "")
    if not v:
        return None
    for p in patterns:
        np = _normalise(p)
        if not np:
            continue
        # Regex escape hatch: anchored patterns
        if np.startswith("^") or np.endswith("$"):
            try:
                if re.search(np, v):
                    return p
            except re.error:
                continue
            continue
        if np in v:
            return p
    return None


def path_for(project_root: Path) -> Path:
    return Path(project_root) / _FILENAME


def load(project_root: Path) -> VenueConfig:
    p = path_for(project_root)
    if not p.exists():
        return VenueConfig()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("venue_config.json unreadable at %s: %s", p, exc)
        return VenueConfig()
    if not isinstance(data, dict):
        logger.warning("venue_config.json is not a JSON object at %s", p)
        return VenueConfig()
    b = data.get("blocklist") or []
    a = data.get("allowlist") or []
    if not isinstance(b, list):
        b = []
    if not isinstance(a, list):
        a = []
    return VenueConfig(
        blocklist=[str(x) for x in b if isinstance(x, str)],
        allowlist=[str(x) for x in a if isinstance(x, str)],
    )


def save(project_root: Path, cfg: VenueConfig) -> Path:
    p = path_for(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "blocklist": sorted(set(cfg.blocklist), key=str.lower),
        "allowlist": sorted(set(cfg.allowlist), key=str.lower),
    }
    p.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return p


def add_pattern(project_root: Path, pattern: str, *, kind: str) -> tuple[VenueConfig, bool]:
    """Append a pattern to blocklist / allowlist; return (new_cfg, was_added)."""
    if kind not in ("block", "allow"):
        raise ValueError(f"kind must be 'block' or 'allow', got {kind!r}")
    cfg = load(project_root)
    target = cfg.blocklist if kind == "block" else cfg.allowlist
    existing = {_normalise(x) for x in target}
    if _normalise(pattern) in existing:
        return cfg, False
    target.append(pattern)
    save(project_root, cfg)
    return cfg, True


def remove_pattern(project_root: Path, pattern: str, *, kind: str) -> tuple[VenueConfig, bool]:
    """Remove a pattern from blocklist / allowlist; return (new_cfg, was_removed)."""
    if kind not in ("block", "allow"):
        raise ValueError(f"kind must be 'block' or 'allow', got {kind!r}")
    cfg = load(project_root)
    target = cfg.blocklist if kind == "block" else cfg.allowlist
    target_norm = _normalise(pattern)
    remaining = [x for x in target if _normalise(x) != target_norm]
    if len(remaining) == len(target):
        return cfg, False
    if kind == "block":
        cfg.blocklist = remaining
    else:
        cfg.allowlist = remaining
    save(project_root, cfg)
    return cfg, True
