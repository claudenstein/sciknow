"""Phase 54.6.71 — citation marker → chunk alignment post-pass (#7).

Rationale
---------

The 2026-04-17-full quality bench showed `autowrite_writer` citation
precision at 11% for gemma3 and 60% for the winning qwen model —
meaning 40–90% of emitted citation markers don't point at a source
chunk that actually supports the sentence they live in. That's the
ALCE citation_precision failure mode.

This module runs as a post-pass AFTER the writer produces a draft but
BEFORE the scorer grades it (or on-demand via `sciknow book
align-citations <draft_id>`). For every sentence in the draft that
contains an `[N]` citation marker, it scores the sentence's entailment
against every source chunk available to the writer. If the claimed
source (chunk N) is NOT the top entailer AND the top entailer's score
exceeds the claimed source's by a configurable margin, the marker is
remapped to the true top-entailing chunk.

The alignment is **conservative by design** — it only remaps when:
  - the claimed chunk's entailment is below an absolute threshold
    (default 0.5), AND
  - the top chunk's entailment beats it by the margin (default 0.15)

That twin gate prevents the post-pass from thrashing on borderline
cases where the writer's citation is plausible — we only intervene
when it's clearly wrong by the model's own evaluation.

Reference: Zhang et al., *LongCite*, 2024 — sentence-level citation
grounding via entailment. Our implementation is prompt-free (just the
cross-encoder NLI model already loaded for faithfulness scoring), so
it composes with the existing verify + CoVe stack without new LLM cost.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Defaults / thresholds
# ════════════════════════════════════════════════════════════════════════


DEFAULT_LOW_THRESHOLD = 0.5      # claimed chunk entailment must be below this
DEFAULT_WIN_MARGIN = 0.15        # and top chunk must beat it by at least this


# Pattern for citation markers. Matches [1], [12], [1,2], [1, 2, 3].
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class RemapEvent:
    """One citation remap. Logged for observability / user review."""
    sentence_preview: str          # first 80 chars of the sentence
    original_marker: str           # e.g. "[3]"
    claimed_n: int                 # original cited chunk number
    new_n: int                     # new top-entailing chunk number
    claimed_score: float           # entailment for chunk N
    new_score: float               # entailment for the remapped chunk


@dataclass
class AlignmentResult:
    """Aggregate outcome of running align_citations on one draft."""
    new_text: str                  # draft with citation markers rewritten
    n_sentences_scanned: int       # sentences containing ≥1 [N]
    n_citations_checked: int       # total [N] markers examined
    n_remapped: int                # how many markers were rewritten
    remaps: list[RemapEvent]       # detail for each remap

    def summary(self) -> str:
        if self.n_citations_checked == 0:
            return "0 citations found to align"
        pct = 100.0 * self.n_remapped / self.n_citations_checked
        return (
            f"scanned {self.n_sentences_scanned} cited sentences · "
            f"{self.n_citations_checked} markers · "
            f"remapped {self.n_remapped} ({pct:.1f}%)"
        )


# ════════════════════════════════════════════════════════════════════════
# Source-chunk normalization
# ════════════════════════════════════════════════════════════════════════


def _normalize_sources(source_chunks: list) -> list[dict]:
    """Return a uniform [{n, content}] list from heterogeneous sources input.

    sciknow has multiple source-shape conventions depending on the call
    site: SearchResult dataclass instances, raw dicts from drafts.sources,
    and tuples. We accept all three and return the minimal shape needed
    for NLI scoring — a 1-indexed number (matches the writer's [N]
    convention) and the full chunk text.
    """
    out: list[dict] = []
    for i, s in enumerate(source_chunks, 1):
        if s is None:
            continue
        if hasattr(s, "content"):
            # SearchResult-like dataclass
            text = getattr(s, "content", "") or ""
        elif isinstance(s, dict):
            text = (s.get("content") or s.get("text") or
                    s.get("chunk_text") or "")
        elif isinstance(s, str):
            text = s
        else:
            text = ""
        text = text.strip()
        if not text:
            continue
        out.append({"n": i, "content": text})
    return out


# ════════════════════════════════════════════════════════════════════════
# Sentence + citation parsing (kept local to avoid cross-module coupling)
# ════════════════════════════════════════════════════════════════════════


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])")


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return [(start, end, sentence_text), ...] with offsets into the
    original text. We need offsets because the remap has to rewrite
    markers in-place without re-splitting the draft.
    """
    out: list[tuple[int, int, str]] = []
    # Walk sentence spans via the split regex.
    pos = 0
    for m in _SENT_SPLIT.finditer(text):
        end = m.start()
        sent = text[pos:end].strip()
        if sent:
            out.append((pos, end, sent))
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        out.append((pos, len(text), tail))
    return out


def _markers_in_sentence(sentence_text: str) -> list[tuple[int, int, list[int]]]:
    """Return [(match_start_in_sentence, match_end, [n1, n2, ...])] for
    every citation marker in the sentence. Handles both `[1]` and
    `[1, 2, 3]` forms.
    """
    out: list[tuple[int, int, list[int]]] = []
    for m in _CITE.finditer(sentence_text):
        nums_str = m.group(1)
        nums = [int(x.strip()) for x in nums_str.split(",") if x.strip().isdigit()]
        out.append((m.start(), m.end(), nums))
    return out


# ════════════════════════════════════════════════════════════════════════
# NLI scoring
# ════════════════════════════════════════════════════════════════════════


def _nli_score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    """Delegate to the NLI helper already loaded for the quality bench.

    We import lazily so core/book_ops imports of this module don't pull
    sentence-transformers + a 440 MB model onto every CLI invocation.
    The model is cached after first use, so autowrite's inline call-site
    pays the load cost once and amortizes across all iterations.
    """
    if not pairs:
        return []
    from sciknow.testing.quality import _nli_entail_probs
    # _nli_entail_probs can raise skip() on model-load failure — it's a
    # bench helper. Catch and degrade to no-op so production autowrite
    # doesn't blow up when NLI is unavailable (e.g. no network for the
    # first download).
    try:
        return _nli_entail_probs(pairs)
    except Exception as exc:
        logger.warning("NLI unavailable — citation alignment disabled: %s", exc)
        return [0.0] * len(pairs)


# ════════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════════


def align_citations(
    draft_text: str,
    source_chunks: list,
    *,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    win_margin: float = DEFAULT_WIN_MARGIN,
) -> AlignmentResult:
    """Rewrite draft_text so citation markers point at the chunk that
    best entails the sentence they live in.

    Parameters
    ----------
    draft_text
        Full draft content with `[N]` citation markers, where N is the
        1-indexed position of the source chunk in ``source_chunks``.
    source_chunks
        List of sources in the same order the writer saw them. Each
        item can be a SearchResult-like object (with ``.content``), a
        dict with ``content``/``text``/``chunk_text``, or a plain
        string. The n=1 source is index 0 etc.
    low_threshold
        Only consider remapping when the CLAIMED chunk's entailment
        probability is below this value. Default 0.5.
    win_margin
        Only remap if the TOP chunk's entailment exceeds the claimed
        chunk's by this much. Default 0.15. Together with
        ``low_threshold`` this forms the conservative remap gate.

    Returns
    -------
    AlignmentResult with the rewritten text, counts, and per-remap log.
    """
    sources = _normalize_sources(source_chunks)
    if not sources or not draft_text or not draft_text.strip():
        return AlignmentResult(draft_text or "", 0, 0, 0, [])

    # {n: content} for O(1) lookup (sources are 1-indexed).
    by_n = {s["n"]: s["content"] for s in sources}
    all_n = sorted(by_n.keys())

    # Step 1: walk the draft, collect every sentence that has at least one
    # citation marker, along with the (start_in_draft, end_in_draft,
    # [ns_cited]) for each marker.
    sentences = _split_sentences(draft_text)
    cited_sentences: list[dict] = []
    total_citations = 0
    for s_start, s_end, s_text in sentences:
        markers = _markers_in_sentence(s_text)
        if not markers:
            continue
        total_citations += sum(len(ns) for _, _, ns in markers)
        cited_sentences.append({
            "s_start": s_start,
            "s_end": s_end,
            "text": s_text,
            "markers": markers,
        })
    if not cited_sentences:
        return AlignmentResult(draft_text, 0, 0, 0, [])

    # Step 2: batch NLI. For each cited sentence, we score it against
    # every source chunk. Flatten the (sentence, source) pairs.
    pairs: list[tuple[str, str]] = []
    for sd in cited_sentences:
        for n in all_n:
            pairs.append((by_n[n], sd["text"]))
    probs = _nli_score_pairs(pairs)
    # probs is (sentence_i × source_j) flat in the order we enqueued.
    n_src = len(all_n)

    # Step 3: for each cited sentence, figure out remaps.
    remaps: list[RemapEvent] = []
    for si, sd in enumerate(cited_sentences):
        window = probs[si * n_src : (si + 1) * n_src]
        scores_by_n = dict(zip(all_n, window))
        top_n = max(scores_by_n, key=scores_by_n.get) if scores_by_n else None
        top_score = scores_by_n.get(top_n, 0.0) if top_n else 0.0

        for (m_start, m_end, ns) in sd["markers"]:
            # We remap PER-NUMBER inside the marker, not per-marker — a
            # `[1, 5]` where 1 is right but 5 is wrong should remap 5 only.
            remapped_ns: list[int] = []
            for n in ns:
                claimed_score = scores_by_n.get(n, 0.0)
                if (claimed_score < low_threshold
                        and top_n is not None
                        and top_n != n
                        and (top_score - claimed_score) >= win_margin):
                    remapped_ns.append(top_n)
                    remaps.append(RemapEvent(
                        sentence_preview=sd["text"][:80],
                        original_marker=f"[{','.join(map(str, ns))}]",
                        claimed_n=n,
                        new_n=top_n,
                        claimed_score=round(float(claimed_score), 3),
                        new_score=round(float(top_score), 3),
                    ))
                else:
                    remapped_ns.append(n)
            # Record the substitution on the sentence. We collect all
            # subs first, then apply in reverse-order so offsets stay
            # stable.
            sd.setdefault("subs", []).append(
                (m_start, m_end, remapped_ns, ns)
            )

    # Step 4: rewrite. Walk sentences in original-draft order and splice.
    out = []
    last = 0
    for sd in cited_sentences:
        # Everything between last-end and this sentence's start, verbatim.
        out.append(draft_text[last : sd["s_start"]])
        # Build the rewritten sentence from in-sentence markers sorted.
        subs = sorted(sd.get("subs", []), key=lambda x: x[0])
        s_text = sd["text"]
        rewritten_parts: list[str] = []
        cursor = 0
        for m_start, m_end, new_ns, orig_ns in subs:
            rewritten_parts.append(s_text[cursor:m_start])
            # Deduplicate while preserving order.
            seen: set[int] = set()
            uniq: list[int] = []
            for x in new_ns:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            rewritten_parts.append("[" + ", ".join(str(x) for x in uniq) + "]")
            cursor = m_end
        rewritten_parts.append(s_text[cursor:])
        rewritten = "".join(rewritten_parts)
        # Preserve whatever separator was between this sentence and the
        # next (spaces / newlines sit in draft_text[s_end:next_s_start]).
        out.append(rewritten)
        last = sd["s_end"]
    out.append(draft_text[last:])
    new_text = "".join(out)

    return AlignmentResult(
        new_text=new_text,
        n_sentences_scanned=len(cited_sentences),
        n_citations_checked=total_citations,
        n_remapped=len(remaps),
        remaps=remaps,
    )
