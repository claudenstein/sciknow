"""Phase 32.10 — Compound learning Layer 5: style fingerprint extraction.

Extracts a small JSON fingerprint of the user's writing style from drafts
they have explicitly approved (status in {final, reviewed, revised}). The
fingerprint is stored in `books.custom_metadata.style_fingerprint` and
injected into the writer system prompt as a style anchor so the autowrite
loop produces drafts that match the user's voice over time.

Why this is independent of Layers 1-4: it doesn't read from the autowrite
telemetry tables at all. It reads from `drafts.content` directly, looking
only at drafts the user actively touched. So Layer 5 can ship in parallel
with everything else and starts producing value as soon as ONE draft gets
marked as reviewed/revised/final.

Metrics extracted:
- median_sentence_length    — words per sentence (typographical fingerprint)
- median_paragraph_words    — paragraph length distribution
- citations_per_100_words   — citation density (literal [N] markers)
- hedging_rate              — fraction of sentences containing a hedge cue
                              from the BioScope-derived list (Phase 7)
- top_transitions           — sentence-initial transition words the user
                              actually uses, ranked by frequency
- avg_words_per_draft       — typical section length the user accepts
- n_drafts_sampled          — how many drafts went into this fingerprint

The fingerprint is recomputed on demand via
`compute_style_fingerprint(book_id)` and called manually from the new CLI
command `sciknow book style refresh`. There's no auto-refresh hook —
recomputing after every status change would be wasteful and the user
explicitly triggers refresh when they're ready.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Iterable

logger = logging.getLogger("sciknow.core.style_fingerprint")


# ── Hedge cues mirroring Phase 7's BioScope-derived list ────────────────
# Kept in sync with the prompt rule in rag/prompts.py:704-707. If you
# change the prompt cue list, change this constant too.
_HEDGE_CUES: frozenset[str] = frozenset({
    "may", "might", "could", "suggest", "suggests", "indicate", "indicates",
    "appear", "appears", "seem", "seems", "likely", "possibly", "probably",
    "tend", "tends", "consistent", "associated", "evidence", "support",
    "supports", "point", "points", "hint", "hints", "perhaps", "potentially",
    "approximately", "roughly",
})

# Sentence-initial transitions worth tracking. Curated from common
# academic prose connectives. The fingerprint reports the top-N the
# user actually uses so the writer can mirror them.
_TRANSITION_CUES: frozenset[str] = frozenset({
    "however", "moreover", "furthermore", "therefore", "thus", "hence",
    "consequently", "nevertheless", "nonetheless", "additionally",
    "similarly", "conversely", "specifically", "notably", "importantly",
    "interestingly", "remarkably", "indeed", "instead", "meanwhile",
    "subsequently", "previously", "historically", "recently", "first",
    "second", "third", "finally", "alternatively", "accordingly",
    "in contrast", "by contrast", "in addition", "in particular",
    "for example", "for instance", "in summary", "in conclusion",
    "taken together", "more recently",
})

# Status values that count as "user-approved" (vs the autowrite default
# 'drafted'). Phase 5 web reader exposes these as the status dropdown.
_APPROVED_STATUSES: tuple[str, ...] = ("final", "reviewed", "revised")

# Floor for the number of drafts needed before a fingerprint is meaningful.
# Below this we still compute, but flag the fingerprint as "preliminary"
# in `samples_warning`.
_MIN_SAMPLES_FOR_RELIABLE = 5


def _split_sentences(text: str) -> list[str]:
    """Best-effort sentence segmentation. Splits on `.`, `!`, `?` followed
    by whitespace, ignoring decimals and abbreviations roughly via a
    look-behind on a non-digit. Adequate for fingerprint stats — not
    a replacement for spaCy.
    """
    if not text:
        return []
    # Strip citation markers so they don't break sentence boundaries
    cleaned = re.sub(r"\[\d+\]", "", text)
    # Strip code blocks (rare in book drafts but possible)
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    # Split on sentence-final punctuation followed by whitespace
    # The look-behind avoids splitting "e.g." / "i.e." / "Fig. 3" etc.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", cleaned)
    return [s.strip() for s in parts if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


def _count_words(s: str) -> int:
    return len([w for w in s.split() if w])


def _median(values: Iterable[float]) -> float:
    vs = sorted(values)
    n = len(vs)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return float(vs[n // 2])
    return (vs[n // 2 - 1] + vs[n // 2]) / 2.0


def _extract_metrics_for_draft(content: str) -> dict:
    """Compute the per-draft metrics that aggregate into the book
    fingerprint. Returns a dict of raw counts; the aggregator turns
    these into percentiles."""
    sentences = _split_sentences(content)
    paragraphs = _split_paragraphs(content)
    words = _count_words(content)
    citations = len(re.findall(r"\[\d+\]", content))

    # Per-sentence length distribution
    sentence_lengths = [_count_words(s) for s in sentences]
    paragraph_lengths = [_count_words(p) for p in paragraphs]

    # Hedge cue rate: fraction of sentences containing at least one hedge.
    # Word-boundary match, case-insensitive, on lowercased tokens.
    n_hedged_sentences = 0
    transition_counts: Counter[str] = Counter()
    for s in sentences:
        # Tokenize once per sentence; cheap.
        toks = re.findall(r"[a-zA-Z]+(?:'[a-z]+)?", s.lower())
        if any(t in _HEDGE_CUES for t in toks):
            n_hedged_sentences += 1
        # Sentence-initial transition: first 1-3 tokens
        if toks:
            first_word = toks[0]
            if first_word in _TRANSITION_CUES:
                transition_counts[first_word] += 1
            if len(toks) >= 2:
                two = f"{toks[0]} {toks[1]}"
                if two in _TRANSITION_CUES:
                    transition_counts[two] += 1
            if len(toks) >= 3:
                three = f"{toks[0]} {toks[1]} {toks[2]}"
                if three in _TRANSITION_CUES:
                    transition_counts[three] += 1

    return {
        "n_words": words,
        "n_sentences": len(sentences),
        "n_paragraphs": len(paragraphs),
        "n_citations": citations,
        "n_hedged_sentences": n_hedged_sentences,
        "sentence_lengths": sentence_lengths,
        "paragraph_lengths": paragraph_lengths,
        "transition_counts": dict(transition_counts),
    }


def _aggregate_fingerprint(per_draft: list[dict]) -> dict:
    """Take a list of per-draft metric dicts and produce the final
    aggregated fingerprint that lands in books.custom_metadata.
    """
    if not per_draft:
        return {
            "n_drafts_sampled": 0,
            "samples_warning": "No approved drafts (status in final/reviewed/revised) — fingerprint empty.",
        }

    all_sent_lens = [
        l for m in per_draft for l in m["sentence_lengths"] if l > 0
    ]
    all_para_lens = [
        l for m in per_draft for l in m["paragraph_lengths"] if l > 0
    ]
    total_words = sum(m["n_words"] for m in per_draft)
    total_sentences = sum(m["n_sentences"] for m in per_draft)
    total_citations = sum(m["n_citations"] for m in per_draft)
    total_hedged = sum(m["n_hedged_sentences"] for m in per_draft)

    # Aggregate transition counts across all drafts
    agg_transitions: Counter[str] = Counter()
    for m in per_draft:
        agg_transitions.update(m["transition_counts"])
    top_transitions = [
        {"word": w, "count": c}
        for w, c in agg_transitions.most_common(8)
    ]

    fingerprint = {
        "n_drafts_sampled": len(per_draft),
        "median_sentence_length": round(_median(all_sent_lens), 1),
        "median_paragraph_words": round(_median(all_para_lens), 1),
        "citations_per_100_words": (
            round(100.0 * total_citations / total_words, 2)
            if total_words > 0 else 0.0
        ),
        "hedging_rate": (
            round(total_hedged / total_sentences, 3)
            if total_sentences > 0 else 0.0
        ),
        "avg_words_per_draft": (
            round(total_words / len(per_draft), 0)
            if per_draft else 0.0
        ),
        "top_transitions": top_transitions,
    }
    if len(per_draft) < _MIN_SAMPLES_FOR_RELIABLE:
        fingerprint["samples_warning"] = (
            f"Only {len(per_draft)} approved draft(s) — "
            f"fingerprint is preliminary (≥{_MIN_SAMPLES_FOR_RELIABLE} recommended)."
        )
    return fingerprint


def compute_style_fingerprint(book_id: str) -> dict:
    """Compute and persist the style fingerprint for a book.

    Walks all drafts in the book whose status is in _APPROVED_STATUSES,
    extracts per-draft metrics, aggregates them, and writes the result
    to `books.custom_metadata.style_fingerprint`.

    Returns the fingerprint dict (also persisted as a side effect).
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as session:
        rows = session.execute(text("""
            SELECT content
            FROM drafts
            WHERE book_id::text = :bid
              AND status = ANY(:statuses)
              AND content IS NOT NULL
              AND length(content) > 100
        """), {
            "bid": book_id,
            "statuses": list(_APPROVED_STATUSES),
        }).fetchall()

    per_draft = [_extract_metrics_for_draft(r[0]) for r in rows]
    fingerprint = _aggregate_fingerprint(per_draft)

    # Persist to books.custom_metadata.style_fingerprint via JSONB merge
    import json
    with get_session() as session:
        session.execute(text("""
            UPDATE books SET
                custom_metadata = COALESCE(custom_metadata, CAST('{}' AS jsonb))
                                  || jsonb_build_object('style_fingerprint',
                                       CAST(:fp AS jsonb))
            WHERE id::text = :bid
        """), {
            "bid": book_id,
            "fp": json.dumps(fingerprint),
        })
        session.commit()

    logger.info(
        "computed style fingerprint for book %s: %d drafts sampled, %d sentences, "
        "%d hedged, %d citations",
        book_id, fingerprint.get("n_drafts_sampled", 0),
        sum(m["n_sentences"] for m in per_draft),
        sum(m["n_hedged_sentences"] for m in per_draft),
        sum(m["n_citations"] for m in per_draft),
    )
    return fingerprint


def get_style_fingerprint(book_id: str) -> dict | None:
    """Read the persisted style fingerprint without recomputing.

    Returns None if the book doesn't exist or has no fingerprint yet.
    Used by the autowrite loop to inject the fingerprint into the
    writer prompt without paying the recomputation cost.
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    try:
        with get_session() as session:
            row = session.execute(text("""
                SELECT custom_metadata FROM books WHERE id::text = :bid LIMIT 1
            """), {"bid": book_id}).fetchone()
        if not row:
            return None
        meta = row[0] or {}
        if isinstance(meta, str):
            import json
            try:
                meta = json.loads(meta)
            except Exception:
                return None
        fp = meta.get("style_fingerprint") if isinstance(meta, dict) else None
        if not fp or not isinstance(fp, dict):
            return None
        if fp.get("n_drafts_sampled", 0) == 0:
            return None
        return fp
    except Exception as exc:
        logger.warning("style fingerprint read failed: %s", exc)
        return None


def format_fingerprint_for_prompt(fp: dict | None) -> str:
    """Render a style fingerprint as a compact prompt block. Returns
    empty string if no fingerprint or insufficient samples.
    """
    if not fp or fp.get("n_drafts_sampled", 0) == 0:
        return ""
    sentence_target = fp.get("median_sentence_length", 0)
    para_target = fp.get("median_paragraph_words", 0)
    cite_density = fp.get("citations_per_100_words", 0)
    hedging = fp.get("hedging_rate", 0)
    avg_words = int(fp.get("avg_words_per_draft", 0) or 0)
    transitions = fp.get("top_transitions") or []
    transition_str = ", ".join(
        f"\"{t['word']}\"" for t in transitions[:5]
    ) if transitions else "(none observed)"

    block = (
        "\nMatch the established style of THIS book "
        f"(extracted from {fp['n_drafts_sampled']} approved draft(s)):\n"
        f"- median sentence length: {sentence_target} words\n"
        f"- median paragraph length: {para_target} words\n"
        f"- typical section length: ~{avg_words} words\n"
        f"- citation density: {cite_density} citations per 100 words\n"
        f"- hedging rate: {hedging:.0%} of sentences contain a hedge marker\n"
        f"- preferred transitions: {transition_str}\n"
    )
    return block
