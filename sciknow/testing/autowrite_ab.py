"""Phase 54.6.161 — bottom-up vs top-down autowrite A/B harness.

RESEARCH.md §24 §gaps #3: "Bottom-up vs top-down authoring A/B — no one
in the educational-technology literature has tested bottom-up section-
first vs top-down chapter-first generation for LLM-assisted scientific
writing. sciknow's autowrite pipeline is the ideal testbed."

This module provides the scaffolding, not a full experimental run — a
proper study needs tens of sections × 2 conditions × 3 iterations each,
which is hours of LLM compute. The infra lets a future researcher run
paired trials and get structured output; they still own the actual
experiment design + execution.

**Protocol**:

1. Pick a chapter whose sections all have a plan (prereq for the
   bottom-up condition via the 54.6.146 concept-density resolver).
2. For each section, snapshot the existing plan + target_words.
3. Run autowrite in **bottom-up** mode (plan present → concept-density
   resolver fires → target = n_concepts × wpc_mid). Save the final
   draft's scorer dimensions.
4. Temporarily clear the plan, run autowrite in **top-down** mode
   (no plan → chapter_split fallback → target = chapter_target / n).
   Save dimensions.
5. Restore the plan.
6. Emit a paired-difference report per scorer dimension +
   visual_citation + length_score.

**Caveats** (documented in the CLI):

- Autowrite costs ~60-180 s per section on a 3090. A 5-section
  chapter × 2 modes = ~10-30 min wall.
- Drafts are written to the DB twice per section; the A/B harness
  doesn't clean them up — post-hoc analysis can use `drafts.version`
  to tell A from B.
- Scorer variance is real (~±0.05 on each dimension across runs on
  the same input); N=5 per chapter gives rough signal only.

**Public API**::

    run_ab(chapter_id, *, model=None, max_iter=3) -> ABReport

Reports `{per_section, per_dimension_delta_mean, win_rate}`.
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SectionTrial:
    section_slug: str
    # Bottom-up (plan present, concept-density resolver fires)
    bu_draft_id: str | None = None
    bu_target_words: int = 0
    bu_scores: dict = field(default_factory=dict)
    bu_elapsed_s: float = 0.0
    # Top-down (plan cleared, chapter-split fallback)
    td_draft_id: str | None = None
    td_target_words: int = 0
    td_scores: dict = field(default_factory=dict)
    td_elapsed_s: float = 0.0
    error: str | None = None


@dataclass
class ABReport:
    chapter_id: str
    chapter_title: str
    n_sections: int
    trials: list[SectionTrial] = field(default_factory=list)
    # Aggregates
    per_dimension_delta_mean: dict[str, float] = field(default_factory=dict)
    # For each dimension, % of sections where bottom-up > top-down
    per_dimension_win_rate: dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chapter_id": self.chapter_id,
            "chapter_title": self.chapter_title,
            "n_sections": self.n_sections,
            "elapsed_s": self.elapsed_s,
            "per_dimension_delta_mean": {
                k: round(v, 4) for k, v in self.per_dimension_delta_mean.items()
            },
            "per_dimension_win_rate": {
                k: round(v, 3) for k, v in self.per_dimension_win_rate.items()
            },
            "trials": [
                {
                    "section_slug": t.section_slug,
                    "bu_draft_id": t.bu_draft_id, "td_draft_id": t.td_draft_id,
                    "bu_target": t.bu_target_words, "td_target": t.td_target_words,
                    "bu_scores": t.bu_scores, "td_scores": t.td_scores,
                    "bu_elapsed_s": round(t.bu_elapsed_s, 1),
                    "td_elapsed_s": round(t.td_elapsed_s, 1),
                    "error": t.error,
                }
                for t in self.trials
            ],
        }


# ── Helpers ─────────────────────────────────────────────────────────


def _run_autowrite_and_collect(
    book_id: str,
    chapter_id: str,
    section_slug: str,
    *,
    model: str | None,
    max_iter: int,
) -> tuple[str | None, int, dict, float]:
    """Drive ``autowrite_section_stream`` to completion, returning
    ``(draft_id, effective_target_words, final_scores, elapsed_s)``.

    Collects the last ``scores`` event + the ``completed`` event's
    draft_id, ignores token-level events to keep the harness tight.
    """
    from sciknow.core.book_ops import autowrite_section_stream
    t0 = time.monotonic()
    draft_id = None
    effective_target = 0
    final_scores: dict = {}
    for evt in autowrite_section_stream(
        book_id=book_id, chapter_id=chapter_id,
        section_type=section_slug,
        model=model, max_iter=max_iter,
    ):
        kind = evt.get("type", "")
        if kind == "length_target":
            effective_target = int(evt.get("target_words") or 0)
        elif kind == "scores":
            sc = evt.get("scores") or evt.get("data") or {}
            if isinstance(sc, dict):
                final_scores = sc
        elif kind == "completed":
            draft_id = evt.get("draft_id")
        elif kind == "error":
            raise RuntimeError(evt.get("message") or "autowrite error")
    return draft_id, effective_target, final_scores, time.monotonic() - t0


def _with_plan_temporarily_cleared(chapter_id: str, section_slug: str):
    """Context manager: temporarily set the section's plan to empty,
    restore on exit. Runs the top-down condition on a chapter whose
    sections have plans without permanently destroying them.
    """
    from contextlib import contextmanager
    from sqlalchemy import text
    from sciknow.core.book_ops import (
        _get_chapter_sections_normalized, _normalize_chapter_sections,
    )
    from sciknow.storage.db import get_session
    import json as _json

    @contextmanager
    def _ctx():
        with get_session() as session:
            sections = _get_chapter_sections_normalized(session, chapter_id)
        original_plan = None
        target_idx = None
        for i, s in enumerate(sections):
            if s.get("slug", "") == section_slug:
                original_plan = s.get("plan", "")
                target_idx = i
                break
        if target_idx is None:
            raise ValueError(f"section {section_slug!r} not in chapter")
        # Write the cleared-plan version
        sections[target_idx] = {**sections[target_idx], "plan": ""}
        with get_session() as session:
            session.execute(text("""
                UPDATE book_chapters SET sections = CAST(:secs AS jsonb)
                WHERE id::text = :cid
            """), {"cid": chapter_id, "secs": _json.dumps(sections)})
            session.commit()
        try:
            yield
        finally:
            # Restore
            sections[target_idx] = {**sections[target_idx], "plan": original_plan}
            with get_session() as session:
                session.execute(text("""
                    UPDATE book_chapters SET sections = CAST(:secs AS jsonb)
                    WHERE id::text = :cid
                """), {"cid": chapter_id, "secs": _json.dumps(sections)})
                session.commit()

    return _ctx()


# ── Public API ──────────────────────────────────────────────────────


def run_ab(
    chapter_id: str,
    *,
    model: str | None = None,
    max_iter: int = 3,
    only_planned: bool = True,
) -> ABReport:
    """Run the A/B on every section of a chapter and return the report.

    ``only_planned=True`` (default) skips sections without a plan —
    those can't do the bottom-up condition.
    """
    from sqlalchemy import text
    from sciknow.core.book_ops import (
        _get_chapter_sections_normalized, _count_plan_concepts,
    )
    from sciknow.storage.db import get_session

    t_all0 = time.monotonic()
    with get_session() as session:
        row = session.execute(text("""
            SELECT book_id::text, title FROM book_chapters
            WHERE id::text = :cid LIMIT 1
        """), {"cid": chapter_id}).fetchone()
        if not row:
            raise ValueError(f"no chapter {chapter_id!r}")
        book_id, chapter_title = row
        sections = _get_chapter_sections_normalized(session, chapter_id)

    trials: list[SectionTrial] = []
    for s in sections:
        slug = s.get("slug", "")
        plan = s.get("plan") or ""
        if only_planned and _count_plan_concepts(plan) < 1:
            logger.info("A/B skipping %s (no plan)", slug)
            continue

        trial = SectionTrial(section_slug=slug)

        # Bottom-up condition (plan present → concept-density)
        try:
            bu_draft, bu_tw, bu_scores, bu_dt = _run_autowrite_and_collect(
                book_id, chapter_id, slug, model=model, max_iter=max_iter,
            )
            trial.bu_draft_id = bu_draft
            trial.bu_target_words = bu_tw
            trial.bu_scores = bu_scores
            trial.bu_elapsed_s = bu_dt
        except Exception as exc:  # noqa: BLE001
            trial.error = f"bottom-up failed: {exc}"
            trials.append(trial)
            continue

        # Top-down condition (plan cleared → chapter-split)
        try:
            with _with_plan_temporarily_cleared(chapter_id, slug):
                td_draft, td_tw, td_scores, td_dt = _run_autowrite_and_collect(
                    book_id, chapter_id, slug, model=model, max_iter=max_iter,
                )
            trial.td_draft_id = td_draft
            trial.td_target_words = td_tw
            trial.td_scores = td_scores
            trial.td_elapsed_s = td_dt
        except Exception as exc:  # noqa: BLE001
            trial.error = (trial.error or "") + f" · top-down failed: {exc}"

        trials.append(trial)

    # Aggregates
    per_dim_deltas: dict[str, list[float]] = {}
    for t in trials:
        if not t.bu_scores or not t.td_scores:
            continue
        for dim in set(t.bu_scores) | set(t.td_scores):
            bu_v = t.bu_scores.get(dim)
            td_v = t.td_scores.get(dim)
            if not (isinstance(bu_v, (int, float)) and isinstance(td_v, (int, float))):
                continue
            per_dim_deltas.setdefault(dim, []).append(float(bu_v) - float(td_v))

    mean_deltas = {dim: statistics.fmean(vs) for dim, vs in per_dim_deltas.items() if vs}
    win_rates = {
        dim: sum(1 for v in vs if v > 0) / len(vs)
        for dim, vs in per_dim_deltas.items() if vs
    }

    return ABReport(
        chapter_id=chapter_id,
        chapter_title=chapter_title or "",
        n_sections=len(trials),
        trials=trials,
        per_dimension_delta_mean=mean_deltas,
        per_dimension_win_rate=win_rates,
        elapsed_s=round(time.monotonic() - t_all0, 1),
    )
