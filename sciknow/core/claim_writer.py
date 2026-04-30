"""Phase 56.E — Claim-driven writer.

Replaces the section-shot writer with per-claim micro-generations.
Each claim becomes 1-3 sentences supported by its pinned chunks. The
writer never picks a citation marker — it emits ``<C:claim_id>``
placeholders that the assembler (56.F) resolves to deterministic
``[N]`` once the per-section source list is fixed.

Output shape:

  ClaimSentences(claim_id, text, hedge_used, n_words, n_chunks_pinned)

The writer is intentionally *short-context*: 2-3 chunks per claim
rather than 12 chunks per section. EOS-bias mid-sentence cutoffs
(documented in docs/research/SECTION_ENDING_RESEARCH.md) are rare on
short generations, so the safety net (56.H) almost never fires here.

Hedging fidelity is structural: the writer prompt is parameterised by
``hedge_strength``, and the prompt enforces vocabulary appropriate to
each level:
  - strong:      "shows", "establishes", "demonstrates"
  - qualified:   "suggests", "indicates", "is associated with"
  - speculative: "may", "could", "tend to"
The scorer (56.J) verifies deterministically — pick the cue word
in the rendered sentence, check it falls in the claim's hedge bucket.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sciknow.core.claim_extractor import Claim, HedgeStrength
from sciknow.retrieval.claim_retrieve import ClaimEvidence, EvidenceChunk

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Tuning
# ──────────────────────────────────────────────────────────────────────


MAX_CHUNKS_PER_CLAIM = 3   # how many supporting chunks the prompt shows
MAX_TOKENS_PER_CLAIM = 450 # output budget (1-3 sentences = ~150-300)


# ──────────────────────────────────────────────────────────────────────
# Output shape
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ClaimSentences:
    """One claim's writer output: prose with embedded citation
    placeholder.

    The prose contains exactly one ``<C:claim_id>`` placeholder that
    the assembler later replaces with the per-section global ``[N]``
    citation marker. ``hedge_used`` records the cue word the writer
    actually chose (for the deterministic scorer in 56.J).
    """
    claim_id: str
    text: str                  # rendered prose, with <C:claim_id> placeholder
    hedge_used: HedgeStrength  # the hedge bucket actually emitted
    n_words: int
    n_chunks_pinned: int


# ──────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────


_HEDGE_VOCAB = {
    HedgeStrength.STRONG: (
        "shows / establishes / demonstrates / proves / confirms / "
        "rules out"
    ),
    HedgeStrength.QUALIFIED: (
        "suggests / indicates / is associated with / appears to / "
        "is consistent with / supports"
    ),
    HedgeStrength.SPECULATIVE: (
        "may / might / could / possibly / tend to / hint at"
    ),
}


CLAIM_WRITER_SYSTEM_TMPL = """\
You write 1-3 sentences asserting ONE claim. Cite ONLY the chunks
provided. End the cited sentence(s) with the placeholder marker
``<C:{claim_id}>`` — do NOT use [N] markers; the assembler resolves
``<C:…>`` to the correct global number after.

CLAIM
-----
{claim_text}

SCOPE
-----
{scope}

HEDGE STRENGTH (REQUIRED — do not upgrade or downgrade)
------------------------------------------------------
Use vocabulary from this set: {hedge_vocab}.
DO NOT use stronger or weaker cue words than this set permits.
If the supporting chunks contradict this hedge level, prefer the
hedge level given here (the planner already calibrated it).

SUPPORTING CHUNKS
-----------------
{chunks_block}

OUTPUT RULES
------------
- 1-3 sentences. No headers. No preamble. No closing notes.
- Every cited factual sentence ends with ``<C:{claim_id}>`` (even when
  multiple sentences cite the same chunks; the assembler dedupes).
- Carry over scope qualifiers ("in the North Atlantic", "post-1500",
  "for decadal timescales") VERBATIM — do not generalise.
- Do not introduce numbers, dates, or named entities not present in
  the chunks.
- Use precise, formal academic register. Avoid filler.

Begin:
"""


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _format_chunks_block(chunks: list[EvidenceChunk], *, max_total: int = 6000) -> str:
    """Render up to MAX_CHUNKS_PER_CLAIM chunks in a compact form,
    capped at ``max_total`` chars overall.
    """
    parts: list[str] = []
    total = 0
    for c in chunks[:MAX_CHUNKS_PER_CLAIM]:
        title = c.title or "(untitled)"
        body = c.content.strip()
        # Cap each chunk body so a single 8000-char chunk doesn't
        # dominate.
        if len(body) > max_total - total - 200:
            body = body[: max_total - total - 200] + "…"
        block = f"--- {title} ---\n{body}"
        if total + len(block) > max_total:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def _detect_hedge_used(text: str) -> HedgeStrength:
    """Detect which hedge bucket the writer's emitted prose actually
    uses, by cue lookup. Reuses the same cue tables as the extractor
    for symmetry."""
    from sciknow.core.claim_extractor import infer_hedge_from_text
    return infer_hedge_from_text(text)


# ──────────────────────────────────────────────────────────────────────
# Top-level entry
# ──────────────────────────────────────────────────────────────────────


def write_claim(
    claim: Claim,
    evidence: ClaimEvidence,
    *,
    model: str | None = None,
) -> ClaimSentences | None:
    """Generate prose for one claim. Returns None if the claim is weak.

    Weak claims (insufficient evidence above the entailment floor)
    must NOT be written — the section drops them and the coverage
    feedback loop (56.G) surfaces the gap. This is the core
    "drop, don't fabricate" guarantee.
    """
    if evidence.weak:
        logger.info(
            "claim %s skipped (WEAK: %s)",
            claim.claim_id, evidence.weak_reason,
        )
        return None
    if not evidence.chunks:
        return None

    from sciknow.rag.llm import complete as llm_complete
    from sciknow.core.draft_finalize import ensure_clean_ending

    chunks_block = _format_chunks_block(evidence.chunks)
    prompt = CLAIM_WRITER_SYSTEM_TMPL.format(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        scope=claim.scope or "(no scope qualifier — assert unconditionally)",
        hedge_vocab=_HEDGE_VOCAB[claim.hedge_strength],
        chunks_block=chunks_block,
    )

    raw = llm_complete(
        prompt, "",
        model=model,
        temperature=0.2,
        num_predict=MAX_TOKENS_PER_CLAIM,
        keep_alive=-1,
    )
    text = ensure_clean_ending(raw.strip())
    if not text:
        return None

    # Guarantee the placeholder shows up at least once. Models
    # occasionally drop it; we append on the trailing sentence as a
    # last resort so the assembler can still hook this claim into
    # the section's source list.
    placeholder = f"<C:{claim.claim_id}>"
    if placeholder not in text:
        # Insert before the trailing terminator if any; else append.
        for term in (".", "?", "!"):
            if text.rstrip().endswith(term):
                text = text.rstrip()[:-1] + f" {placeholder}" + term
                break
        else:
            text = text.rstrip() + f" {placeholder}."

    return ClaimSentences(
        claim_id=claim.claim_id,
        text=text,
        hedge_used=_detect_hedge_used(text),
        n_words=len(text.split()),
        n_chunks_pinned=min(len(evidence.chunks), MAX_CHUNKS_PER_CLAIM),
    )


def write_claims(
    claims: list[Claim],
    evidence_by_claim: dict[str, ClaimEvidence],
    *,
    model: str | None = None,
) -> list[ClaimSentences]:
    """Run write_claim across a list. Returns only the non-None
    results (weak claims dropped). Order matches the input."""
    out: list[ClaimSentences] = []
    for c in claims:
        ev = evidence_by_claim.get(c.claim_id)
        if ev is None:
            continue
        s = write_claim(c, ev, model=model)
        if s is not None:
            out.append(s)
    return out
