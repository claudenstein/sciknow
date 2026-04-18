"""Phase 54.6.83 (#8) — claim-atomization for deeper faithfulness audit.

Sentence-level NLI (as used by the quality bench's ``faithfulness_score``
and by the autowrite verifier) has a well-known blindspot: a sentence
like *"X causes Y, and Greenland observations from 2010-2020 confirm it"*
gets one entailment score, but the X-causes-Y claim and the
Greenland-observations claim are independent and the source may
support one but not the other. The single score averages them away.

This module splits each sentence into **atomic sub-claims** before
NLI, then reports per-sub-claim + per-sentence verdicts. It's
scoped as an **offline** diagnostic — a new ``sciknow book
verify-draft <id>`` CLI runs it on a saved draft, producing a report
that highlights mixed-truth sentences. Deliberately not wired into
the quality bench (doing so would multiply NLI cost by 2-3× across
every bench run, with marginal signal).

**Atomization strategy.** Heuristic-first, LLM-fallback:

1. Cheap regex splits on coordinating conjunctions (``, and ``,
   ``; ``, ``. ``) — catches ~70% of compound sentences for free.
2. For sentences that stay complex after the heuristic (>30 words,
   multiple clauses, or coordinating conjunctions the regex missed),
   fall back to an LLM atomizer.

This keeps the average cost near zero for clean prose and only pays
the LLM tax on the sentences that genuinely need it.

Reference: FActScore (Min et al., 2023) for the atomize-then-verify
approach — RESEARCH.md §526 explicitly keeps the *offline*
evaluator-slice open while rejecting the online variant.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Thresholds
# ════════════════════════════════════════════════════════════════════════


COMPLEX_SENTENCE_WORDS = 30   # trigger LLM atomizer for sentences above this
MAX_SUB_CLAIMS = 4            # cap per sentence — don't explode on runaways
MIN_SUB_CLAIM_WORDS = 3       # shorter than 3 words is almost never a claim
ENTAIL_THRESHOLD = 0.5        # per-sub-claim entailment gate for "supported"


# ════════════════════════════════════════════════════════════════════════
# Heuristic atomization
# ════════════════════════════════════════════════════════════════════════


# Split points that almost always mark an independent clause boundary.
# Order matters — we try these in order, preferring the safer ones first.
_SPLIT_PATTERNS = [
    # Strong: semicolons + em-dashes almost always separate independent
    # clauses in scientific prose.
    re.compile(r"\s*[;]\s+"),
    re.compile(r"\s+—\s+"),
    # Medium: "; and" / ", and" / "; but" / ", but" when the subject
    # repeats or the tense shifts (hard to detect without a parser —
    # we accept some noise here).
    re.compile(r",\s+and\s+(?=[A-Z]|[a-z]+\s+(?:the|a|an|has|have|is|are|was|were|will|can|may))"),
    re.compile(r",\s+but\s+(?=[A-Z]|[a-z]+\s+(?:the|a|an|has|have|is|are|was|were))"),
]


def _word_count(s: str) -> int:
    return len([w for w in (s or "").split() if w.strip()])


def atomize_heuristic(sentence: str) -> list[str]:
    """Split on coordinating-conjunction / punctuation boundaries that
    almost always separate independent clauses. Returns [sentence]
    unchanged when no split applies."""
    s = (sentence or "").strip()
    if not s:
        return []
    parts = [s]
    for pat in _SPLIT_PATTERNS:
        new_parts: list[str] = []
        for p in parts:
            splits = [x.strip() for x in pat.split(p) if x.strip()]
            # Only accept the split if every resulting piece is a real
            # clause — otherwise we're fragmenting on a coincidental
            # comma in a number list or a quoted phrase.
            if (len(splits) >= 2
                    and all(_word_count(x) >= MIN_SUB_CLAIM_WORDS for x in splits)):
                new_parts.extend(splits)
            else:
                new_parts.append(p)
        parts = new_parts
        if len(parts) >= MAX_SUB_CLAIMS:
            parts = parts[:MAX_SUB_CLAIMS]
            break
    return [p for p in parts if _word_count(p) >= MIN_SUB_CLAIM_WORDS] or [s]


def needs_llm_atomizer(sentence: str) -> bool:
    """True when heuristic atomization left a sentence that looks
    compound (long + multiple conjunctions) — triggers the LLM pass."""
    s = (sentence or "").strip()
    if _word_count(s) <= COMPLEX_SENTENCE_WORDS:
        return False
    # Count conjunctions NOT already caught by the heuristic. If a
    # sentence has ≥2 conjunctions AND is long, the heuristic missed
    # at least one of them.
    n_conj = len(re.findall(
        r"\b(?:and|but|whereas|while|however|although|because)\b", s,
        flags=re.IGNORECASE,
    ))
    return n_conj >= 2


# ════════════════════════════════════════════════════════════════════════
# LLM atomizer (fallback)
# ════════════════════════════════════════════════════════════════════════


_ATOMIZE_SYSTEM = (
    "You split one scientific sentence into atomic sub-claims. Each "
    "sub-claim must be an independent factual statement that can be "
    "verified on its own. Output a JSON array of strings. Rules:\n"
    "  - Between 1 and 4 sub-claims.\n"
    "  - Each sub-claim is a complete sentence (subject + verb + "
    "predicate).\n"
    "  - Do NOT add content the original doesn't state; do NOT drop "
    "quantitative details / citations.\n"
    "  - If the sentence is already atomic (one independent factual "
    "statement), return a single-element array with the sentence "
    "unchanged.\n"
    "Respond with ONLY the JSON array, no preamble."
)


def atomize_llm(sentence: str, model: str | None = None) -> list[str]:
    """LLM fallback for sentences the heuristic couldn't split. Returns
    [sentence] on any parsing failure — failing closed avoids losing
    the sentence from the verification set."""
    import json
    from sciknow.rag.llm import complete as _complete

    user = f"Sentence:\n{sentence}\n\nSub-claims:"
    try:
        raw = _complete(
            _ATOMIZE_SYSTEM, user,
            model=model, temperature=0.0, num_ctx=4096, keep_alive=-1,
        )
    except Exception as exc:
        logger.debug("atomize_llm failed: %s", exc)
        return [sentence]
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        arr = json.loads(raw, strict=False)
    except Exception:
        # Try to extract the first [...] block.
        m = re.search(r"\[[\s\S]*?\]", raw)
        if not m:
            return [sentence]
        try:
            arr = json.loads(m.group(0), strict=False)
        except Exception:
            return [sentence]
    if not isinstance(arr, list):
        return [sentence]
    out: list[str] = []
    for item in arr[:MAX_SUB_CLAIMS]:
        t = str(item).strip().strip('"\'')
        if t and _word_count(t) >= MIN_SUB_CLAIM_WORDS:
            out.append(t)
    return out or [sentence]


def atomize(sentence: str, model: str | None = None,
            *, allow_llm: bool = True) -> list[str]:
    """Top-level: heuristic first, LLM fallback for complex survivors."""
    heur = atomize_heuristic(sentence)
    if not allow_llm:
        return heur
    # Only ask the LLM if heuristic left behind ONE long compound sentence.
    if len(heur) == 1 and needs_llm_atomizer(heur[0]):
        return atomize_llm(heur[0], model=model)
    return heur


# ════════════════════════════════════════════════════════════════════════
# Full-draft verification
# ════════════════════════════════════════════════════════════════════════


@dataclass
class SubClaimVerdict:
    text: str
    entailment: float
    supported: bool


@dataclass
class SentenceVerdict:
    sentence: str
    atomized: bool                     # did we actually split
    sub_claims: list[SubClaimVerdict] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        if not self.sub_claims:
            return 0.0
        return sum(1 for s in self.sub_claims if s.supported) / len(self.sub_claims)

    @property
    def mixed_truth(self) -> bool:
        """True when at least one sub-claim is supported AND at least
        one isn't — the failure mode the single-sentence NLI misses."""
        if len(self.sub_claims) < 2:
            return False
        truth = {s.supported for s in self.sub_claims}
        return truth == {True, False}


@dataclass
class DraftVerification:
    n_sentences: int
    n_atomized: int                    # sentences that got split
    n_sub_claims: int
    n_supported: int
    mean_entailment: float
    mixed_truth_count: int             # SentenceVerdict.mixed_truth == True
    sentences: list[SentenceVerdict] = field(default_factory=list)

    def summary(self) -> str:
        atom_pct = 100.0 * self.n_atomized / max(1, self.n_sentences)
        sup_pct = 100.0 * self.n_supported / max(1, self.n_sub_claims)
        return (
            f"{self.n_sentences} sentences → {self.n_sub_claims} sub-claims "
            f"({atom_pct:.0f}% atomized) · "
            f"{sup_pct:.1f}% supported · "
            f"mean entailment {self.mean_entailment:.3f} · "
            f"{self.mixed_truth_count} mixed-truth sentences"
        )


def _split_sentences(text: str) -> list[str]:
    """Light wrapper — reuse the same splitter as quality.py."""
    from sciknow.testing.quality import _split_sentences as _split
    return _split(text or "")


def verify_draft(
    draft_text: str,
    source_chunks: list,
    *,
    model: str | None = None,
    allow_llm_atomize: bool = True,
) -> DraftVerification:
    """Atomize every sentence, score each sub-claim's entailment
    against the source chunks via NLI, aggregate.

    ``source_chunks`` accepts the same heterogeneous shape as
    ``citation_align.align_citations`` — SearchResult-like objects,
    dicts with ``content``/``text``/``chunk_text``, or strings.
    """
    from sciknow.core.citation_align import _normalize_sources
    from sciknow.testing.quality import _nli_entail_probs

    sentences = _split_sentences(draft_text)
    sources = _normalize_sources(source_chunks)
    if not sentences or not sources:
        return DraftVerification(
            n_sentences=0, n_atomized=0, n_sub_claims=0, n_supported=0,
            mean_entailment=0.0, mixed_truth_count=0, sentences=[],
        )

    # Phase 1: atomize.
    per_sentence: list[tuple[str, list[str]]] = []
    for s in sentences:
        subs = atomize(s, model=model, allow_llm=allow_llm_atomize)
        per_sentence.append((s, subs))

    # Phase 2: for each sub-claim, compute max-over-sources entailment.
    all_sub_claims = [sc for _, subs in per_sentence for sc in subs]
    # Batch (source, sub_claim) pairs flat.
    pairs: list[tuple[str, str]] = []
    n_src = len(sources)
    for sc in all_sub_claims:
        for src in sources:
            pairs.append((src["content"], sc))
    probs = _nli_entail_probs(pairs) if pairs else []

    # Reduce to max-over-sources per sub-claim.
    per_sub_score: list[float] = []
    for i in range(len(all_sub_claims)):
        window = probs[i * n_src: (i + 1) * n_src] if n_src else []
        per_sub_score.append(max(window) if window else 0.0)

    # Assemble verdicts.
    verdicts: list[SentenceVerdict] = []
    n_atomized = 0
    n_sub_claims = 0
    n_supported = 0
    cursor = 0
    total_entail = 0.0
    mixed_count = 0
    for sent, subs in per_sentence:
        sub_verdicts: list[SubClaimVerdict] = []
        for sub in subs:
            p = per_sub_score[cursor]; cursor += 1
            supported = bool(p >= ENTAIL_THRESHOLD)
            if supported:
                n_supported += 1
            total_entail += p
            sub_verdicts.append(SubClaimVerdict(
                text=sub, entailment=round(float(p), 3), supported=supported,
            ))
            n_sub_claims += 1
        atomized = len(subs) > 1
        if atomized:
            n_atomized += 1
        sv = SentenceVerdict(sentence=sent, atomized=atomized,
                              sub_claims=sub_verdicts)
        if sv.mixed_truth:
            mixed_count += 1
        verdicts.append(sv)

    mean = (total_entail / max(1, n_sub_claims))
    return DraftVerification(
        n_sentences=len(sentences), n_atomized=n_atomized,
        n_sub_claims=n_sub_claims, n_supported=n_supported,
        mean_entailment=round(mean, 3),
        mixed_truth_count=mixed_count, sentences=verdicts,
    )
