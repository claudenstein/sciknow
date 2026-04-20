"""Phase 49 — hard filters for `db expand` candidate gating.

Runs *before* any ranking so noise candidates (retractions, predatory
venues, editorials, errata, conference abstracts) never make it into
the RRF pool. Each filter is cheap (inspects the OpenAlex work dict
that the ranker already fetches for every candidate) and returns a
reason string so the dry-run TSV can attribute every drop.

See `docs/EXPAND_RESEARCH.md` for the design rationale and the per-
signal trade-offs. Filters here are hard-drop (the candidate never
reaches the RRF); signals that *penalise* without dropping live on
`CandidateFeatures` in `expand_ranker.py` instead.
"""
from __future__ import annotations

from typing import Iterable

# ── Document-type blocklist ──────────────────────────────────────────
# Types that are never primary research literature. OpenAlex's `type`
# field follows Crossref's CRMF. `peer-review` is Crossref's type for
# standalone peer-review reports (we want the reviewed paper itself,
# not the review document). `letter` covers Letters to the Editor.
DROP_DOC_TYPES: frozenset[str] = frozenset({
    "editorial",
    "erratum",
    "correction",
    "letter",
    "peer-review",
    "grant",
})

# ── Predatory / hijacked venue heuristics ────────────────────────────
# Tiny seed list of the most-cited predatory publishers (by
# partial-name match against OpenAlex `primary_location.source.host_
# organization_name` or `publisher`). Keep this CONSERVATIVE — false
# positives here silently delete papers the user probably wanted. A
# bigger curated list can be loaded from a file at runtime; see
# `load_extra_predatory_patterns` below.
#
# Sources: Beall's archived list + Retraction Watch's Hijacked Journal
# Checker (both widely referenced). We keep only patterns that are
# unambiguous publisher-name substrings (not ISSNs, which change).
_DEFAULT_PREDATORY_PATTERNS: tuple[str, ...] = (
    "scientific research publishing",   # SCIRP
    "omics international",
    "academia publishing",
    "david publishing",
    "science publishing group",
    "bentham science",
    "scholarena",
    "juniper publishers",
    "open access pub",
    "pushpa publishing",
    "sciencepg",
)

# Mutable module-level set so a user can extend at runtime via
# `load_extra_predatory_patterns`. Lower-cased for case-insensitive
# substring matching.
PREDATORY_PUBLISHER_PATTERNS: set[str] = {p.lower() for p in _DEFAULT_PREDATORY_PATTERNS}


def load_extra_predatory_patterns(patterns: Iterable[str]) -> None:
    """Extend the predatory-pattern set (used by config loaders / tests)."""
    for p in patterns:
        if p and p.strip():
            PREDATORY_PUBLISHER_PATTERNS.add(p.strip().lower())


# ── Individual filter predicates ─────────────────────────────────────

def is_retracted(work: dict | None) -> bool:
    """OpenAlex exposes Crossref's Retraction Watch flag as
    `is_retracted`. Conservative: treat missing field as not retracted
    so an API hiccup doesn't mass-delete candidates."""
    if not work:
        return False
    return bool(work.get("is_retracted") or False)


def _venue_names(work: dict | None) -> list[str]:
    """Return the venue-identifying strings to match against block /
    allow / predatory patterns. Extends the Phase 49 match surface
    (host_organization_name + publisher) to also include the source's
    display_name so a venue like "Journal of Multidisciplinary X"
    matches on its actual name, not just its parent publisher."""
    if not work:
        return []
    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    out: list[str] = []
    for key in ("host_organization_name", "publisher", "display_name"):
        v = source.get(key)
        if isinstance(v, str) and v:
            out.append(v)
    # Top-level alternate fields OpenAlex occasionally surfaces
    for key in ("host_venue_display_name", "host_venue_publisher"):
        v = work.get(key)
        if isinstance(v, str) and v:
            out.append(v)
    return out


def is_predatory_venue(work: dict | None) -> bool:
    """Substring match of the candidate's publisher / host org /
    source-display-name against `PREDATORY_PUBLISHER_PATTERNS`.
    Case-insensitive. Extended in 54.6.112 to consume the per-project
    ``venue_config.json`` blocklist + allowlist; the allowlist wins
    when both match so users can rescue legitimate venues that happen
    to hit a substring pattern."""
    venues = _venue_names(work)
    if not venues:
        return False
    # Per-project overrides (Phase 54.6.112). Allowlist wins even if a
    # built-in predatory pattern would have matched.
    try:
        from sciknow.core.project import get_active_project
        from sciknow.core import venue_config as _vc
        cfg = _vc.load(get_active_project().root)
        for v in venues:
            if cfg.matches_allow(v):
                return False
            if cfg.matches_block(v):
                return True
    except Exception:  # noqa: BLE001
        # Config load failures must not break expand — silently fall
        # through to the built-in list.
        pass
    # Built-in predatory patterns — keep as the last check so an
    # allowlist entry can rescue a venue from them.
    for val in venues:
        v = val.lower()
        for pat in PREDATORY_PUBLISHER_PATTERNS:
            if pat in v:
                return True
    return False


def drop_reason_by_doc_type(work: dict | None) -> str:
    """If the candidate's type is blocklisted, return the type string
    so the caller can log it. Empty string = keep. Also catches
    too-short proceedings entries (< 4 pages) which are almost always
    abstracts-only, not full papers."""
    if not work:
        return ""
    t = (work.get("type") or "").strip().lower()
    if t in DROP_DOC_TYPES:
        return t
    # Short proceedings articles are conference abstracts in practice.
    # OpenAlex stores page info in `biblio.first_page` / `last_page`.
    if t == "proceedings-article":
        bib = work.get("biblio") or {}
        try:
            fp = int(str(bib.get("first_page") or "0").split("-")[0])
            lp = int(str(bib.get("last_page") or "0").split("-")[-1])
            if 0 < fp <= lp and (lp - fp) < 3:
                return "short_proceedings"
        except (ValueError, TypeError):
            pass
    return ""


def apply_hard_filters(work: dict | None) -> tuple[bool, str]:
    """Run all hard filters in order. Returns (should_drop, reason)
    where `reason` is a short stable token for TSV logging."""
    if is_retracted(work):
        return True, "retracted"
    if is_predatory_venue(work):
        return True, "predatory_venue"
    t_drop = drop_reason_by_doc_type(work)
    if t_drop:
        return True, f"doc_type:{t_drop}"
    return False, ""
