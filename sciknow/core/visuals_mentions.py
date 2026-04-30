"""Phase 54.6.138 — mention-paragraph extraction for visuals.

For each ``visuals`` row with a numeric ``figure_num``, scan the source
paper's ``content_list.json`` for body-text paragraphs that reference
that number (``Fig. 3``, ``Figure 3``, ``Table 2``, ``Eq. 5``) and
persist them as JSONB on ``visuals.mention_paragraphs``.

Why this matters (docs/research/RESEARCH.md §7.X, signal 3): SciCap+ (Yang et
al., 2023) showed that the mention-paragraph — the author's own prose
framing of why the figure is cited at that point — is a stronger
retrieval signal for matching a figure to target draft prose than
either the caption or the image itself. Captions are notoriously terse
(~50% near-useless per SciCap); the mention-paragraph carries the
rhetorical framing the writer agent needs.

Idempotent: existing non-null ``mention_paragraphs`` rows are skipped
unless ``force=True``. Safe to re-run; the linker is read-only against
the filesystem content_list.json.

Public API:
- ``link_visuals_for_doc(doc_id, *, force=False)`` → int
- ``link_visuals_for_corpus(limit=0, force=False, on_progress=None)`` →
    yields (doc_id, n_linked) per paper
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from sqlalchemy import text as sql_text

from sciknow.config import settings
from sciknow.storage.db import get_session

logger = logging.getLogger(__name__)


# ── Figure-number parsing ────────────────────────────────────────────
#
# Matches: "Fig. 3", "Fig 3", "Figure 3", "Figure 3a", "Fig. 3(a)",
# "Figs. 3–5" (captured once per range endpoint), "Tables 2 and 3".
# Deliberately generous; precision beats recall for this pass because
# false-positive paragraphs dilute the signal. A bare "3" is NEVER a
# match — it must be preceded by the figure/table/equation keyword.
_REF_RE = re.compile(
    r"""
    \b
    (?P<kind>Fig(?:ure)?s?|Tab(?:le)?s?|Eq(?:uation)?s?)
    \s*\.?\s*
    (?P<num>\d+)
    (?!\.\d)                     # NOT a hierarchical sub-figure label
                                 # ("Fig. 2.1" is a different figure,
                                 #  not a reference to figure 2)
    (?:\s*[-–—]\s*\d+)?          # optional range ("3-5")
    (?:\s*[a-z])?                # optional subpanel letter ("3a")
    (?:\s*\([a-z](?:[-–—,]?[a-z])*\))?  # or "3(a)", "3(a,b)", "3(a-c)"
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


_FIGNUM_NUMERIC_RE = re.compile(r"(\d+)")


def _parse_figure_number(fig_num: str | None) -> int | None:
    """Extract the numeric core of a figure_num string.

    ``figure_num`` is free-form because MinerU captions vary ("Figure 3",
    "Fig. 3.5", "Table IIIa"). We accept any string that contains a
    leading integer and use that as the canonical ID for matching.
    Returns None when no numeric prefix exists (e.g. roman numerals,
    caption-less figures).
    """
    if not fig_num:
        return None
    m = _FIGNUM_NUMERIC_RE.search(fig_num)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _kind_matches(visual_kind: str, ref_kind: str) -> bool:
    """Does a reference's keyword (``Fig``/``Table``/``Eq``) apply to
    this visual's kind? Figures/charts share the "Fig" family; tables
    match "Table"; equations match "Eq". Returns True on an ambiguous
    mismatch so callers can decide (currently we require exact family
    match to keep precision high)."""
    vk = (visual_kind or "").lower()
    rk = (ref_kind or "").lower()
    if rk.startswith("fig"):
        return vk in ("figure", "chart", "image")
    if rk.startswith("tab"):
        return vk == "table"
    if rk.startswith("eq"):
        return vk == "equation"
    return False


# ── Mention extraction ───────────────────────────────────────────────


@dataclass
class MentionParagraph:
    block_idx: int
    text: str
    context_before: str | None = None

    def to_dict(self) -> dict:
        return {
            "block_idx": self.block_idx,
            "text": self.text,
            "context_before": self.context_before,
        }


def _find_content_list(mineru_dir: Path) -> Path | None:
    """Locate the content_list.json under a doc's mineru_output/{doc_id}/ tree."""
    if not mineru_dir.exists():
        return None
    for root_d, _dirs, files in os.walk(mineru_dir):
        for f in files:
            if f.endswith("_content_list.json") or f == "content_list.json":
                return Path(root_d) / f
    return None


def _extract_mentions_for_number(
    content_list: list,
    target_num: int,
    target_kind: str,   # "figure" | "table" | "equation"
    *,
    visual_block_idx: int | None,
    max_paragraphs: int = 8,
    paragraph_max_chars: int = 600,
) -> list[MentionParagraph]:
    """Walk the content_list, return text blocks that reference
    ``target_num`` via a keyword in ``target_kind``'s family.

    Skips the visual's own block (``visual_block_idx``) so captions
    don't self-reference. Truncates each paragraph to ``paragraph_max_chars``
    so the JSONB payload stays bounded on papers that reference the
    same figure 20+ times.
    """
    out: list[MentionParagraph] = []
    for idx, block in enumerate(content_list):
        if visual_block_idx is not None and idx == visual_block_idx:
            continue
        btype = (block.get("type") or "").lower()
        if btype != "text":
            continue
        txt = (block.get("text") or "").strip()
        if not txt:
            continue
        # Only consider references whose keyword family matches the visual's kind
        for m in _REF_RE.finditer(txt):
            ref_kind = m.group("kind")
            if not _kind_matches(target_kind, ref_kind):
                continue
            try:
                ref_num = int(m.group("num"))
            except (TypeError, ValueError):
                continue
            if ref_num != target_num:
                continue
            # Capture the sentence preceding the match as context_before
            span_start = m.start()
            # Walk back to the previous sentence end
            prior = txt[:span_start]
            # Prefer the last sentence break; fall back to 120 chars
            sentence_end = max(prior.rfind("."), prior.rfind("!"), prior.rfind("?"))
            if sentence_end >= 0 and span_start - sentence_end <= 300:
                context_before = prior[sentence_end + 1 :].strip() or None
            else:
                context_before = prior[-120:].strip() or None
            out.append(MentionParagraph(
                block_idx=idx,
                text=txt[:paragraph_max_chars],
                context_before=context_before[:200] if context_before else None,
            ))
            break  # one hit per paragraph; avoid duplicate captures of "Fig. 3 and Fig. 3b"
        if len(out) >= max_paragraphs:
            break
    return out


# ── Public API ───────────────────────────────────────────────────────


def link_visuals_for_doc(doc_id: str, *, force: bool = False) -> int:
    """Populate ``mention_paragraphs`` for every ``visuals`` row of this
    document. Returns the number of rows updated (0 means either no
    visuals or all already had mention_paragraphs and force=False).
    """
    mineru_dir = settings.mineru_output_dir / doc_id
    content_list_path = _find_content_list(mineru_dir)
    if content_list_path is None:
        logger.debug("no content_list.json for %s", doc_id)
        return 0
    try:
        content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("parse content_list.json for %s failed: %s", doc_id, exc)
        return 0
    if not isinstance(content_list, list):
        return 0

    updated = 0
    with get_session() as session:
        # Pull only visuals that could have mentions (numeric figure_num)
        rows = session.execute(sql_text("""
            SELECT id::text, kind, figure_num, block_idx, mention_paragraphs
            FROM visuals
            WHERE document_id = CAST(:doc AS uuid)
        """), {"doc": doc_id}).fetchall()

        for vis_id, kind, figure_num, block_idx, existing in rows:
            if existing is not None and not force:
                continue
            target_num = _parse_figure_number(figure_num)
            if target_num is None:
                # Mark as linked (empty list) so we don't re-scan every run
                session.execute(sql_text("""
                    UPDATE visuals SET mention_paragraphs = CAST('[]' AS jsonb)
                    WHERE id = CAST(:id AS uuid)
                """), {"id": vis_id})
                updated += 1
                continue
            mentions = _extract_mentions_for_number(
                content_list, target_num, kind or "",
                visual_block_idx=block_idx,
            )
            payload = json.dumps([m.to_dict() for m in mentions])
            session.execute(sql_text("""
                UPDATE visuals
                SET mention_paragraphs = CAST(:p AS jsonb)
                WHERE id = CAST(:id AS uuid)
            """), {"p": payload, "id": vis_id})
            updated += 1
        session.commit()
    return updated


def link_visuals_for_corpus(
    *,
    limit: int = 0,
    force: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> Iterable[tuple[str, int]]:
    """Yield ``(doc_id, n_rows_updated)`` per paper. Iterates documents
    that have at least one visuals row; skips papers whose visuals
    already all carry mention_paragraphs unless ``force=True``."""
    with get_session() as session:
        if force:
            doc_ids = [r[0] for r in session.execute(sql_text("""
                SELECT DISTINCT document_id::text FROM visuals
                ORDER BY document_id
            """)).fetchall()]
        else:
            doc_ids = [r[0] for r in session.execute(sql_text("""
                SELECT DISTINCT document_id::text FROM visuals
                WHERE id IN (
                    SELECT id FROM visuals WHERE mention_paragraphs IS NULL
                )
                ORDER BY document_id
            """)).fetchall()]

    if limit > 0:
        doc_ids = doc_ids[:limit]

    total = len(doc_ids)
    for i, doc_id in enumerate(doc_ids, start=1):
        if on_progress:
            try:
                on_progress(i, total, doc_id)
            except Exception:
                pass
        try:
            n = link_visuals_for_doc(doc_id, force=force)
        except Exception as exc:  # noqa: BLE001
            logger.warning("link_visuals_for_doc(%s) failed: %s", doc_id, exc)
            n = 0
        yield doc_id, n
