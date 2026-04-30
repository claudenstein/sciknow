"""Phase 54.6.115 (Tier 2 #3) — per-project expand feedback (HITL).

Stores the user's +/- marks on expansion shortlist candidates so the
ranker can bias the next round toward similar papers (positive) and
away from rejected ones (negative). File lives at
``<project root>/expand_feedback.json``.

Two consumers:

1. **Anchor bias** — the ranker's corpus centroid (used by the
   `bge_m3_cosine` and `citation_context_cosine` signals) is adjusted
   by subtracting a weighted mean of the negative-paper embeddings
   and adding a weighted mean of the positive ones. Survivors
   described similarly to rejected papers rank lower; those similar
   to accepted ones rank higher. See
   ``docs/research/EXPAND_ENRICH_RESEARCH_2.md`` §2.4.

2. **Future LambdaMART training data** — once ≥500 labels have
   accumulated, the JSON feeds a LightGBM train run (the Phase 49
   parked item). The current module just persists; training stays
   parked until label volume justifies it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


_FILENAME = "expand_feedback.json"


@dataclass
class FeedbackEntry:
    doi: str = ""
    arxiv_id: str = ""
    title: str = ""
    topic: str = ""
    added_at: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackEntry":
        return cls(
            doi=(d.get("doi") or "").strip().lower(),
            arxiv_id=(d.get("arxiv_id") or "").strip().lower(),
            title=(d.get("title") or "").strip(),
            topic=(d.get("topic") or "").strip(),
            added_at=(d.get("added_at") or "").strip(),
        )

    def to_dict(self) -> dict:
        out = {
            "doi": self.doi, "arxiv_id": self.arxiv_id,
            "title": self.title, "topic": self.topic,
            "added_at": self.added_at,
        }
        return {k: v for k, v in out.items() if v}

    @property
    def key(self) -> str:
        """Stable identity key — DOI preferred, then arXiv id, then
        normalized title prefix."""
        return self.doi or self.arxiv_id or (self.title[:80].lower().strip())


@dataclass
class Feedback:
    positive: list[FeedbackEntry] = field(default_factory=list)
    negative: list[FeedbackEntry] = field(default_factory=list)

    def all_keys(self, kind: str) -> set[str]:
        target = self.positive if kind == "positive" else self.negative
        return {e.key for e in target if e.key}

    def n_total(self) -> int:
        return len(self.positive) + len(self.negative)


def path_for(project_root: Path) -> Path:
    return Path(project_root) / _FILENAME


def load(project_root: Path) -> Feedback:
    p = path_for(project_root)
    if not p.exists():
        return Feedback()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("expand_feedback.json unreadable at %s: %s", p, exc)
        return Feedback()
    if not isinstance(data, dict):
        return Feedback()
    pos = [FeedbackEntry.from_dict(x) for x in (data.get("positive") or [])]
    neg = [FeedbackEntry.from_dict(x) for x in (data.get("negative") or [])]
    return Feedback(
        positive=[e for e in pos if e.key],
        negative=[e for e in neg if e.key],
    )


def save(project_root: Path, fb: Feedback) -> Path:
    p = path_for(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "positive": [e.to_dict() for e in fb.positive],
        "negative": [e.to_dict() for e in fb.negative],
    }
    p.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return p


def add_entry(
    project_root: Path,
    *,
    kind: str,
    doi: str = "",
    arxiv_id: str = "",
    title: str = "",
    topic: str = "",
) -> tuple[Feedback, bool]:
    """Add a +/- entry. Returns (feedback, was_added).

    If the same key already exists in the OPPOSITE list, it's moved
    rather than duplicated — the most recent verdict wins.
    """
    if kind not in ("positive", "negative"):
        raise ValueError("kind must be 'positive' or 'negative'")
    entry = FeedbackEntry(
        doi=(doi or "").strip().lower(),
        arxiv_id=(arxiv_id or "").strip().lower(),
        title=(title or "").strip(),
        topic=(topic or "").strip(),
        added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if not entry.key:
        return load(project_root), False

    fb = load(project_root)
    # Remove from both lists first (moving if it existed on the other side)
    fb.positive = [e for e in fb.positive if e.key != entry.key]
    fb.negative = [e for e in fb.negative if e.key != entry.key]
    target = fb.positive if kind == "positive" else fb.negative
    target.append(entry)
    save(project_root, fb)
    return fb, True


def remove_entry(
    project_root: Path,
    *,
    key: str,
    kind: str | None = None,
) -> tuple[Feedback, bool]:
    """Remove by DOI / arXiv ID / normalised-title key. When ``kind``
    is ``None`` removes from whichever list it's in; otherwise only
    removes from the specified list."""
    if not key:
        return load(project_root), False
    fb = load(project_root)
    k = key.strip().lower()
    removed = False
    if kind in (None, "positive"):
        before = len(fb.positive)
        fb.positive = [e for e in fb.positive if e.key != k]
        removed = removed or (len(fb.positive) != before)
    if kind in (None, "negative"):
        before = len(fb.negative)
        fb.negative = [e for e in fb.negative if e.key != k]
        removed = removed or (len(fb.negative) != before)
    if removed:
        save(project_root, fb)
    return fb, removed
