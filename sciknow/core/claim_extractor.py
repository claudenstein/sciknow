"""Phase 56.C — Atomic claim extraction.

For each leaf section in an accepted outline, extract N atomic claims
that the section will assert. The claim is the smallest commit unit
the writer (56.E) will make against the corpus — one assertion, one
hedge level, one supporting evidence pool.

Design:

  Section (with anchor cluster) → cluster summary + top-N chunks
                                → LLM emits 8-15 atomic claims
                                → each claim has:
                                    text                (one assertion)
                                    scope               (qualifier — period, region, conditions)
                                    hedge_strength      (strong | qualified | speculative)
                                    anchor_cluster_id   (the section's cluster)
                                    candidate_chunk_ids (rough pre-filter for 56.D)

Hedging is inferred from cue words in the supporting chunks:
  - ``demonstrates``, ``shows``, ``establishes``, ``proves``  → strong
  - ``suggests``, ``indicates``, ``is associated with``       → qualified
  - ``may``, ``might``, ``could``, ``possibly``, ``tend to``  → speculative

Phase 56 makes hedging a structural property of the *claim* — the
writer (56.E) transcribes hedge_strength into prose; the scorer
verifies deterministically (cue lookup, not LLM judgment). This is
the structural fix for the v55 regression where ``hedging_fidelity``
was always the bottom-ranked dimension.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data shape
# ──────────────────────────────────────────────────────────────────────


class HedgeStrength(str, Enum):
    """Three-bucket epistemic strength scale.

    The writer's prompt and the scorer's verifier consume this; keep
    string values stable across components.
    """
    STRONG = "strong"
    QUALIFIED = "qualified"
    SPECULATIVE = "speculative"


@dataclass
class Claim:
    """One atomic assertion the section will make.

    ``candidate_chunk_ids`` is a soft pre-filter — the per-claim
    retrieval engine (56.D) will run a real query expansion + entailment
    pass against the full corpus, so this list is only a hint of which
    chunks the LLM was looking at when it emitted the claim.
    """
    claim_id: str                    # uuid; assigned at extraction time
    text: str
    scope: str                       # qualifier; may be empty
    hedge_strength: HedgeStrength
    anchor_cluster_id: str
    candidate_chunk_ids: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "scope": self.scope,
            "hedge_strength": self.hedge_strength.value,
            "anchor_cluster_id": self.anchor_cluster_id,
            "candidate_chunk_ids": self.candidate_chunk_ids,
        }


# ──────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────


CLAIM_EXTRACTION_SYSTEM = """\
You decompose a section's topical scope into ATOMIC CLAIMS — the
smallest assertion units the section will make. Emit JSON ONLY.

Schema:

{
  "claims": [
    {
      "text": "Single declarative assertion. One subject, one
               predicate, one quantifier or qualifier. ≤ 30 words.",
      "scope": "Period / region / conditions under which the claim
                holds. Empty string if the claim is unconditional.",
      "hedge_strength": "strong" | "qualified" | "speculative",
      "candidate_chunk_ids": ["chunk_uuid_1", "chunk_uuid_2"]
    },
    ...
  ]
}

Rules:

1. ATOMICITY — each claim is exactly ONE assertion. Wrong:
   "Solar minima cause cooling AND historical records confirm it."
   Right (two separate claims):
     "Solar minima coincide with cooling at the surface."
     "Cosmogenic radionuclide records confirm 11 grand minima since 800 AD."

2. SCOPE — preserve qualifiers verbatim from source chunks.
   "in the North Atlantic", "post-1500", "under high-emission
   scenarios", "for decadal timescales" are all valid scope strings.
   Do NOT generalise past what the chunks say.

3. HEDGE_STRENGTH — derive from cue words in the chunks supporting
   the claim:
     strong       — chunks use "shows", "establishes", "demonstrates"
     qualified    — chunks use "suggests", "indicates", "is
                    associated with", "appears to"
     speculative  — chunks use "may", "might", "could", "possibly"
   When chunks disagree, take the LOWEST strength (be conservative).

4. CANDIDATE_CHUNK_IDS — the chunk_ids you read this claim from.
   You will see chunks formatted as `[CHUNK <id>]\\n<content>`; copy
   the ids verbatim. 1-3 chunks per claim is normal.

5. NO META-COMMENTARY — do NOT emit claims like "this section
   discusses X" or "we will examine Y". Each claim is content.

6. TARGET 8-15 CLAIMS for a typical section. Drop a claim rather
   than restate something already covered.

7. JSON ONLY. No markdown, no preamble, no closing notes.
"""


# ──────────────────────────────────────────────────────────────────────
# Cue-word lookup (also used post-hoc for verification)
# ──────────────────────────────────────────────────────────────────────


_HEDGE_CUES = {
    HedgeStrength.STRONG: {
        "shows", "show", "demonstrates", "demonstrate", "establishes",
        "establish", "proves", "prove", "confirms", "confirm",
        "rules out", "ruled out",
    },
    HedgeStrength.QUALIFIED: {
        "suggests", "suggest", "indicates", "indicate",
        "is associated with", "associated with", "appears to",
        "appear to", "consistent with", "in line with", "supports",
        "support", "points to", "points towards",
    },
    HedgeStrength.SPECULATIVE: {
        "may", "might", "could", "possibly", "probably",
        "tend to", "tends to", "perhaps", "hypothesise",
        "hint at", "hints at", "speculate",
    },
}


def infer_hedge_from_text(text: str) -> HedgeStrength:
    """Best-guess hedge strength from cue words in a passage.

    Returns ``QUALIFIED`` when no cues match — the safe default for
    scientific prose. Lowest-strength cue wins on disagreement.
    """
    if not text:
        return HedgeStrength.QUALIFIED
    t = text.lower()
    has_strong = any(c in t for c in _HEDGE_CUES[HedgeStrength.STRONG])
    has_qual = any(c in t for c in _HEDGE_CUES[HedgeStrength.QUALIFIED])
    has_spec = any(c in t for c in _HEDGE_CUES[HedgeStrength.SPECULATIVE])
    if has_spec:
        return HedgeStrength.SPECULATIVE
    if has_qual:
        return HedgeStrength.QUALIFIED
    if has_strong:
        return HedgeStrength.STRONG
    return HedgeStrength.QUALIFIED


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _format_chunks_for_prompt(chunks: list[dict], *, max_chars: int = 12000) -> str:
    """Render chunks as ``[CHUNK <id>]\\n<content>`` blocks separated
    by ``\\n---\\n``. Truncates at max_chars total."""
    parts: list[str] = []
    total = 0
    for c in chunks:
        cid = str(c.get("chunk_id") or c.get("id") or "")
        content = (c.get("content") or "").strip()
        if not cid or not content:
            continue
        block = f"[CHUNK {cid}]\n{content}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


def _get_top_chunks_for_cluster(
    cluster_id: str,
    *,
    n: int = 6,
) -> list[dict]:
    """Pull the top-N chunks for a cluster, ordered by centrality.

    Centrality proxy = chunk's payload.score relative to cluster
    centroid. We use Qdrant's vector search with the cluster's
    centroid as the query — the existing point's vector matches the
    cluster summary, so this approximates "most representative
    chunks of this cluster".
    """
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    from sciknow.storage.qdrant import get_client
    from sciknow.core.project import get_active_project

    p = get_active_project()
    qd = get_client()

    # Step 1: fetch the cluster centroid.
    points = qd.retrieve(
        collection_name=p.papers_collection,
        ids=[cluster_id],
        with_payload=True,
        with_vectors=True,
    )
    if not points:
        logger.warning("cluster %s not found in qdrant", cluster_id)
        return []
    pt = points[0]
    centroid = pt.vector
    if isinstance(centroid, dict):
        centroid = centroid.get("dense") or next(iter(centroid.values()))

    document_ids = (pt.payload or {}).get("document_ids") or []
    if not document_ids:
        return []

    # Step 2: search leaf chunks (node_level == 0) restricted to the
    # cluster's document set, ranked by similarity to the centroid.
    flt = Filter(
        must=[
            FieldCondition(key="node_level", match=MatchValue(value=0)),
        ]
    )
    try:
        # Qdrant's modern API renamed `search` to `query_points`; fall
        # back to `search` on older clients.
        if hasattr(qd, "query_points"):
            results = qd.query_points(
                collection_name=p.papers_collection,
                query=centroid,
                query_filter=flt,
                limit=n * 4,   # over-fetch then doc-filter
                with_payload=True,
            ).points
        else:
            results = qd.search(
                collection_name=p.papers_collection,
                query_vector=centroid,
                query_filter=flt,
                limit=n * 4,
                with_payload=True,
            )
    except Exception as exc:
        logger.warning("centroid search failed for cluster %s: %s",
                       cluster_id, exc)
        return []

    # Step 3: keep only chunks whose document is in the cluster.
    docs = set(document_ids)
    out: list[dict] = []
    for r in results:
        payload = r.payload or {}
        if payload.get("document_id") not in docs:
            continue
        out.append({
            "chunk_id": str(r.id),
            "content": payload.get("content")
                or payload.get("content_preview") or "",
            "title": payload.get("title") or "",
            "document_id": payload.get("document_id"),
        })
        if len(out) >= n:
            break
    return out


# ──────────────────────────────────────────────────────────────────────
# Top-level entry
# ──────────────────────────────────────────────────────────────────────


def extract_claims_for_section(
    section_title: str,
    section_plan: str,
    cluster_id: str,
    cluster_summary: str,
    *,
    n_chunks: int = 6,
    model: str | None = None,
) -> list[Claim]:
    """Extract atomic claims for one section.

    Args:
      section_title: the section's title from the outline.
      section_plan: bullet list (one per line) the section will cover.
      cluster_id: anchor cluster from the outline (RAPTOR node id).
      cluster_summary: the RAPTOR summary text for that cluster.
      n_chunks: how many representative chunks to feed the LLM.
      model: optional override for the writer model.

    Returns: list[Claim] (typically 8-15).
    """
    from uuid import uuid4
    from sciknow.rag.llm import complete as llm_complete
    from sciknow.core.book_ops import _clean_json

    chunks = _get_top_chunks_for_cluster(cluster_id, n=n_chunks)
    chunk_block = _format_chunks_for_prompt(chunks)

    user = (
        f"Section title: {section_title}\n"
        f"Section plan:\n{section_plan}\n\n"
        f"Cluster summary:\n{cluster_summary}\n\n"
        f"Representative chunks (with their chunk_ids):\n{chunk_block}\n"
    )

    raw = llm_complete(
        CLAIM_EXTRACTION_SYSTEM, user,
        model=model, temperature=0.2,
        num_predict=2400, keep_alive=-1,
    )
    try:
        data = json.loads(_clean_json(raw))
    except json.JSONDecodeError as exc:
        logger.warning("claim extractor invalid JSON, retrying once: %s", exc)
        raw2 = llm_complete(
            CLAIM_EXTRACTION_SYSTEM, user + "\n\nReturn JSON only.",
            model=model, temperature=0.0,
            num_predict=2400, keep_alive=-1,
        )
        data = json.loads(_clean_json(raw2))

    raw_claims = data.get("claims") or []
    out: list[Claim] = []
    for rc in raw_claims:
        text = str(rc.get("text") or "").strip()
        if not text:
            continue
        scope = str(rc.get("scope") or "").strip()
        hs_raw = str(rc.get("hedge_strength") or "qualified").lower().strip()
        try:
            hedge = HedgeStrength(hs_raw)
        except ValueError:
            hedge = HedgeStrength.QUALIFIED
        cand = rc.get("candidate_chunk_ids") or []
        if not isinstance(cand, list):
            cand = []
        out.append(Claim(
            claim_id=str(uuid4()),
            text=text,
            scope=scope,
            hedge_strength=hedge,
            anchor_cluster_id=cluster_id,
            candidate_chunk_ids=[str(x) for x in cand],
        ))
    return out


def claims_to_jsonable(claims: list[Claim]) -> list[dict]:
    return [c.to_jsonable() for c in claims]


def claims_from_jsonable(rows: list[dict]) -> list[Claim]:
    """Round-trip helper. Used when reading a persisted claim plan
    from the drafts table back into Claim objects."""
    out: list[Claim] = []
    for r in rows or []:
        try:
            hs = HedgeStrength(str(r.get("hedge_strength", "qualified")).lower())
        except ValueError:
            hs = HedgeStrength.QUALIFIED
        out.append(Claim(
            claim_id=str(r.get("claim_id") or ""),
            text=str(r.get("text") or ""),
            scope=str(r.get("scope") or ""),
            hedge_strength=hs,
            anchor_cluster_id=str(r.get("anchor_cluster_id") or ""),
            candidate_chunk_ids=[str(x) for x in (r.get("candidate_chunk_ids") or [])],
        ))
    return out
