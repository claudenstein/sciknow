"""Phase 54.6.80 (#10) — paper-type classifier.

Your corpus mixes peer-reviewed papers, preprints, theses, policy
briefs, opinion pieces, editorials, and book chapters — currently
retrieved at equal weight. For factual queries an opinion piece is a
liability; for rhetorical queries it's relevant. A one-pass LLM
classifier tags each paper, and the retrieval side (hybrid_search) can
then filter or down-weight by type.

Classification signal: the paper's abstract (if populated) plus the
first ~2000 chars of its content are usually enough to distinguish a
peer-reviewed article from a policy brief. Journal name + publisher
help too — passed as context to the LLM. The classifier returns:

  {
    "type": "peer_reviewed" | "preprint" | "thesis" | "editorial" |
            "opinion" | "policy" | "book_chapter" | "unknown",
    "confidence": 0.0 - 1.0,
    "evidence": "short rationale"  (debug only, not persisted)
  }

Uses LLM_FAST_MODEL (qwen3:30b-a3b-instruct-2507 by default) at
temperature 0 for stability — classification is a retrieval-governance
decision, not a creative task, so we want deterministic output.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


VALID_TYPES: tuple[str, ...] = (
    "peer_reviewed", "preprint", "thesis", "editorial",
    "opinion", "policy", "book_chapter", "unknown",
)


CLASSIFIER_SYSTEM = (
    "You are classifying scientific documents for a corpus retrieval "
    "system. Given the first part of a document plus its bibliographic "
    "metadata, output a strict JSON object with three fields: type, "
    "confidence (0.0-1.0), evidence (one short sentence).\n\n"
    "Allowed type values (pick exactly one):\n"
    "  - peer_reviewed: published in a peer-reviewed journal or "
    "conference (Nature, Science, IEEE, PNAS, etc.)\n"
    "  - preprint: arXiv / bioRxiv / SSRN / ResearchGate unreviewed\n"
    "  - thesis: PhD, MSc, or undergraduate thesis / dissertation\n"
    "  - editorial: journal editorial / letter to the editor / "
    "commentary piece written by journal staff\n"
    "  - opinion: opinion piece, op-ed, blog post, advocacy essay, "
    "non-peer-reviewed popular-science article\n"
    "  - policy: government report, IPCC / UN / agency white paper, "
    "think-tank brief, policy analysis\n"
    "  - book_chapter: chapter in an edited volume or monograph\n"
    "  - unknown: not enough information to decide\n\n"
    "Heuristics: journal names like 'Nature', 'Science', 'IEEE', "
    "'PNAS', 'Climate Dynamics' → peer_reviewed. 'arXiv' in DOI → "
    "preprint. 'thesis' / 'dissertation' in title → thesis. 'IPCC' / "
    "'EPA' / government author → policy. Short opinion-style prose "
    "with no methods section → opinion / editorial. When unsure, "
    "prefer unknown over guessing.\n\n"
    "Respond with ONLY the JSON object, no preamble."
)


CLASSIFIER_USER = (
    "Title: {title}\n"
    "Journal / Venue: {journal}\n"
    "Publisher: {publisher}\n"
    "Year: {year}\n"
    "DOI: {doi}\n\n"
    "Abstract:\n{abstract}\n\n"
    "First 2000 characters of content:\n{content}\n\n"
    "Output JSON now."
)


@dataclass
class ClassificationResult:
    paper_type: str
    confidence: float
    evidence: str


def _strip_latex(text: str) -> str:
    """Drop common LaTeX fragments that corrupt JSON escapes when the
    LLM echoes them back in the evidence field."""
    t = text or ""
    t = re.sub(r"\$[^$]+\$", "", t)
    t = re.sub(r"\\[a-zA-Z]+\s*\{[^}]*\}", "", t)
    t = t.replace("\\", "")
    return t


def classify_paper(
    *,
    title: str = "",
    journal: str = "",
    publisher: str = "",
    year: int | str | None = None,
    doi: str = "",
    abstract: str = "",
    content: str = "",
    model: str | None = None,
) -> ClassificationResult | None:
    """Classify one paper. Returns None on LLM/parsing failure (caller
    leaves the row's paper_type NULL so future runs can retry).

    Content can be long — we trim to 2000 chars to keep the prompt
    cheap; that's usually enough to see the methods/results structure
    that separates peer-reviewed from opinion.
    """
    from sciknow.rag.llm import complete as _complete

    user = CLASSIFIER_USER.format(
        title=_strip_latex(title or "(untitled)")[:300],
        journal=journal or "(unknown)",
        publisher=publisher or "(unknown)",
        year=str(year) if year else "(unknown)",
        doi=doi or "(none)",
        abstract=_strip_latex(abstract or "(missing)")[:1500],
        content=_strip_latex(content or "")[:2000] or "(missing)",
    )

    try:
        raw = _complete(
            CLASSIFIER_SYSTEM, user,
            model=model, temperature=0.0, num_ctx=8192, keep_alive=-1,
        )
    except Exception as exc:
        logger.warning("paper-type classify failed: %s", exc)
        return None

    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(raw, strict=False)
    except Exception:
        # The model sometimes wraps in a sentence. Try to extract the
        # first {...} block.
        m = re.search(r"\{[\s\S]*?\}", raw)
        if not m:
            logger.warning("paper-type: no JSON in response: %s", raw[:200])
            return None
        try:
            data = json.loads(m.group(0), strict=False)
        except Exception:
            return None

    ptype = str(data.get("type") or "").strip().lower()
    if ptype not in VALID_TYPES:
        # Map a few common misspellings before giving up.
        alias = {
            "peer-reviewed": "peer_reviewed",
            "peer_review": "peer_reviewed",
            "research_article": "peer_reviewed",
            "article": "peer_reviewed",
            "review": "peer_reviewed",
            "dissertation": "thesis",
            "whitepaper": "policy",
            "white_paper": "policy",
            "commentary": "editorial",
            "op-ed": "opinion",
            "blog": "opinion",
        }
        ptype = alias.get(ptype, "unknown")

    try:
        conf = float(data.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    evidence = str(data.get("evidence") or "").strip()[:300]

    return ClassificationResult(
        paper_type=ptype, confidence=conf, evidence=evidence,
    )
