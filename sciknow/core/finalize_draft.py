"""Phase 54.6.145 — L3 (VLM) claim-depiction verification for final drafts.

Per the Q2 design decision (docs/roadmap/PHASE_LOG.md 54.6.142), the per-iteration
autowrite loop runs L1 (resolution) + L2 (cross-encoder entailment)
verification on `[Fig. N]` markers. L3 — running a vision-language model
on `(claim_sentence, image)` pairs to check whether the figure actually
*depicts* the claim — is too expensive for every iteration (3-10s per
marker, 4-32 min per chapter), so it lives here in a one-shot finalise
pass meant to run once before export.

Design:

    1. Load the draft content from the ``drafts`` table.
    2. Derive the candidate visual set from ``drafts.sources`` — every
       paper the draft cites is a valid source of figures, and the
       visual's ``figure_num`` can be matched against each paper's
       internal figure numbering. Unlike the autowrite-time ranker,
       we aren't re-running retrieval; we just match markers against
       the cited papers' figures.
    3. For each `[Fig. N]` / `[Table N]` / `[Eq. N]` marker:
        a. L1: does the marker resolve to a visual in the candidate
           pool? (Same check as 54.6.142's _verify_figure_refs.)
        b. L3: run the VLM on ``(containing_sentence, image_path)``
           with a prompt asking for a 0-10 depiction score +
           one-sentence justification.
    4. Flag markers with L1 failures OR L3 score below
       ``flag_threshold`` (default 4/10) for user review.

Public API:

    verify_draft_figures_l3(
        draft_id: str,
        *,
        vlm_model: str | None = None,
        flag_threshold: int = 4,
        on_progress: Callable | None = None,
    ) -> FinalizeReport

The report carries per-marker verdicts so the CLI can render them in
a table and the caller can decide whether to block export.

VRAM note: the VLM is typically *not* resident during drafting (the
text LLM is). Running this finalise pass triggers a model swap
(~10-30 s cold load). Ollama's ``keep_alive=-1`` pins the model
during the run so subsequent markers don't re-swap, then releases on
process exit.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)


# ── Prompts ─────────────────────────────────────────────────────────

L3_VERIFY_SYSTEM = """\
You are a scientific editor verifying that a figure cited in a draft \
actually depicts the claim it is cited for. You are given one image \
and one claim sentence drawn from a scientific draft. Your job is to \
judge whether the image *demonstrates* the claim — not whether the \
image is topically related, but whether a reader looking at the image \
would see the specific evidence the claim asserts.

Scoring rubric (0-10):

  0-2  Image does not depict the claim at all. Wrong topic, wrong \
variable, or the figure shows something completely unrelated.
  3-4  Image is topically related but doesn't show the specific \
claim. E.g. the claim is about sea level trends and the figure \
shows a temperature map.
  5-6  Image is consistent with the claim but doesn't clearly \
demonstrate it. The reader would need additional context to see \
the evidence.
  7-8  Image reasonably supports the claim. A reader can see the \
evidence with a small amount of interpretation.
  9-10 Image clearly and directly demonstrates the claim. The \
specific evidence is visible and unambiguous.

Return ONLY a JSON object with two keys:
  {"score": <int 0-10>, "justification": "<one-sentence reason>"}
No other text, no markdown, no thinking blocks.
"""


L3_VERIFY_USER = """\
Claim sentence: {claim}

Does the provided image depict the claim? Return your JSON verdict."""


# ── Data classes ────────────────────────────────────────────────────


@dataclass
class FigureVerdict:
    marker: str                     # e.g. "[Fig. 3]"
    kind: str                       # "figure" | "table" | "equation"
    num: int
    resolved: bool                  # L1: did it map to a visual in pool?
    visual_id: str | None = None
    figure_num: str | None = None
    document_id: str | None = None
    asset_path: str | None = None
    claim_sentence: str = ""
    # L3 results — None when unresolved or image missing
    vlm_score: int | None = None
    vlm_justification: str | None = None
    # Final verdict — True = looks good, False = needs user review
    passes: bool = False


@dataclass
class FinalizeReport:
    draft_id: str
    n_markers: int
    n_resolved: int
    n_passing: int
    n_flagged: int
    verdicts: list[FigureVerdict] = field(default_factory=list)
    vlm_model: str = ""
    elapsed_s: float = 0.0

    @property
    def pass_rate(self) -> float:
        return self.n_passing / self.n_markers if self.n_markers else 1.0


# ── Marker extraction reuses the 54.6.142 regex ─────────────────────


def _extract_figure_refs_with_sentences(draft: str) -> list[dict]:
    """Walk the draft, return one dict per `[Fig. N]`-style marker
    with the containing sentence.

    Reuses the Phase 54.6.142 extractor but also picks up the
    surrounding sentence — the L3 prompt needs the claim text, not
    just the marker number.
    """
    from sciknow.core.book_ops import _extract_figure_refs, _sentence_for_marker
    markers = _extract_figure_refs(draft or "")
    out = []
    for kind, num, raw in markers:
        out.append({
            "marker": raw,
            "kind": kind,
            "num": num,
            "sentence": _sentence_for_marker(draft, raw),
        })
    return out


def _extract_cited_document_ids(sources: list) -> list[str]:
    """Extract document UUIDs from a draft's ``sources`` JSONB column.

    The sources list contains heterogeneous shapes across phases — we
    defensively try every known shape and return whatever document_ids
    we can find. Empty list on any failure mode.
    """
    if not sources:
        return []
    out: list[str] = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        did = s.get("document_id") or s.get("doc_id") or s.get("paper_id")
        if did:
            out.append(str(did))
    return out


# ── L3 entry point ──────────────────────────────────────────────────


def verify_draft_figures_l3(
    draft_id: str,
    *,
    vlm_model: str | None = None,
    flag_threshold: int = 4,
    on_progress: Callable | None = None,
) -> FinalizeReport:
    """Run VLM claim-depiction verification on every bracketed figure
    citation in a draft. One VLM call per resolved marker.

    ``vlm_model`` defaults to ``settings.visuals_caption_model`` (the
    same model the Phase-54.6.72 captioning pipeline uses, kept on
    disk after the first caption-visuals run).
    """
    import json as _json
    import time as _time

    from sciknow.config import settings as _settings
    from sciknow.core.book_ops import _extract_figure_refs
    from sciknow.core.visuals_caption import resolve_asset_path
    from sciknow.core.visuals_mentions import _parse_figure_number, _kind_matches
    from sciknow.storage.db import get_session

    use_v2_vlm = bool(getattr(_settings, "use_llamacpp_vlm", True))
    if use_v2_vlm:
        # v2 substrate path — vlm role on llama-server :8093 (Qwen3-VL).
        # Caller can override via the explicit vlm_model arg; otherwise
        # we use the configured logical name (label only — actual GGUF
        # is set by VLM_MODEL_GGUF in config.py).
        model = vlm_model or getattr(_settings, "vlm_model_name", None) or "qwen3-vl-30b-a3b"
    else:
        # v1 rollback — Ollama VLM (typically qwen2.5vl:32b).
        model = (
            vlm_model
            or getattr(_settings, "visuals_caption_model", None)
            or "qwen2.5vl:32b"
        )
    t0 = _time.monotonic()

    # ── Load the draft
    with get_session() as session:
        row = session.execute(sql_text("""
            SELECT id::text, content, sources
            FROM drafts WHERE id::text = :id OR id::text LIKE :pfx
            ORDER BY version DESC LIMIT 1
        """), {"id": draft_id, "pfx": f"{draft_id}%"}).fetchone()
        if not row:
            raise ValueError(f"no draft matches {draft_id!r}")
        did, content, sources = row[0], row[1], row[2]

    markers_with_sent = _extract_figure_refs_with_sentences(content)
    if not markers_with_sent:
        return FinalizeReport(
            draft_id=did, n_markers=0, n_resolved=0,
            n_passing=0, n_flagged=0, vlm_model=model,
            elapsed_s=round(_time.monotonic() - t0, 1),
        )

    # ── Build the candidate visual pool from cited documents
    cited_docs = _extract_cited_document_ids(sources or [])
    by_key: dict[tuple[str, int], dict] = {}
    if cited_docs:
        with get_session() as session:
            vrows = session.execute(sql_text("""
                SELECT v.id::text, v.document_id::text, v.kind,
                       v.figure_num, v.asset_path
                FROM visuals v
                WHERE v.document_id = ANY(CAST(:docs AS uuid[]))
                  AND v.figure_num IS NOT NULL
            """), {"docs": cited_docs}).fetchall()
            for vr in vrows:
                fn = _parse_figure_number(vr[3] or "")
                if fn is None:
                    continue
                vk = (vr[2] or "").lower()
                if vk in ("figure", "chart", "image"):
                    family = "figure"
                elif vk == "table":
                    family = "table"
                elif vk == "equation":
                    family = "equation"
                else:
                    continue
                # First hit wins per (family, number) to avoid shadowing
                by_key.setdefault((family, fn), {
                    "visual_id": vr[0],
                    "document_id": vr[1],
                    "kind": vk,
                    "figure_num": vr[3] or "",
                    "asset_path": vr[4] or "",
                })

    # ── Resolve the VLM caller. v2 path goes through the llama-server
    # vlm role; v1 fallback uses Ollama. If the vlm role isn't running
    # (writer typically holds the GPU), degrade gracefully to L2 — the
    # user explicitly starts vlm + reruns finalize when they want full
    # L3 figure verification.
    client = None  # v1 ollama client
    use_v2 = use_v2_vlm
    if use_v2:
        try:
            from sciknow.infer import server as _infer_server
            if not _infer_server.health("vlm"):
                logger.warning(
                    "vlm role on :%s not healthy — L3 figure verification "
                    "will be skipped (start with `sciknow infer up --role vlm` "
                    "or run `sciknow corpus caption-visuals` which auto-swaps).",
                    _infer_server.ROLE_DEFAULTS.get("vlm", {}).get("port", 8093),
                )
                use_v2 = False  # neither path active → degrade to L2
        except Exception as exc:  # noqa: BLE001
            logger.warning("vlm role check failed: %s", exc)
            use_v2 = False
    if not use_v2:
        # v1 fallback — try Ollama. May still be unreachable, in which
        # case we degrade like the v2 path.
        try:
            import ollama as _ollama
            client = _ollama.Client(host=_settings.ollama_host)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ollama import failed — L3 verify cannot run: %s", exc)
            client = None

    verdicts: list[FigureVerdict] = []
    for i, mws in enumerate(markers_with_sent, start=1):
        if on_progress:
            try:
                on_progress(i, len(markers_with_sent), mws["marker"])
            except Exception:
                pass

        kind = mws["kind"]
        num = mws["num"]
        sent = mws["sentence"] or ""

        v = by_key.get((kind, num))
        if v is None:
            verdicts.append(FigureVerdict(
                marker=mws["marker"], kind=kind, num=num,
                resolved=False, claim_sentence=sent, passes=False,
            ))
            continue

        # Resolve the image file on disk
        img_path = None
        if v.get("asset_path"):
            p = resolve_asset_path(v["document_id"], v["asset_path"])
            if p and p.exists():
                img_path = p

        # Tables + equations have no raster image to show the VLM; they
        # get L1-only — resolved = pass (we trust the caption match
        # from L2 that already ran during autowrite).
        no_vlm = (not use_v2) and (client is None)
        if kind in ("table", "equation") or img_path is None or no_vlm:
            verdicts.append(FigureVerdict(
                marker=mws["marker"], kind=kind, num=num,
                resolved=True,
                visual_id=v["visual_id"], figure_num=v["figure_num"],
                document_id=v["document_id"],
                asset_path=(str(img_path) if img_path else v.get("asset_path")),
                claim_sentence=sent,
                vlm_score=None,
                vlm_justification=(
                    "L3 skipped — "
                    + ("no raster image to verify"
                       if kind in ("table", "equation")
                       else "image file not found on disk"
                       if img_path is None
                       else "VLM client unavailable")
                ),
                passes=True,
            ))
            continue

        # ── Level 3: one VLM call (llama-server v2 or Ollama v1)
        user_prompt = L3_VERIFY_USER.format(claim=sent[:600])
        score: int | None = None
        justification = ""
        try:
            if use_v2:
                from sciknow.infer import client as _infer_client
                raw = _infer_client.chat_with_image(
                    system=L3_VERIFY_SYSTEM,
                    user=user_prompt,
                    image_paths=[str(img_path)],
                    model=model,
                    temperature=0.2,
                    num_predict=200,
                )
            else:
                resp = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": L3_VERIFY_SYSTEM},
                        {"role": "user", "content": user_prompt,
                         "images": [str(img_path)]},
                    ],
                    options={"temperature": 0.2, "num_predict": 200},
                    keep_alive=-1,
                )
                raw = (resp.get("message") or {}).get("content", "").strip()
            # Model may return ```json fenced — strip fences
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            # Also strip thinking blocks if a thinking model slipped in
            raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
            parsed = _json.loads(raw)
            sc = parsed.get("score")
            if isinstance(sc, (int, float)):
                score = int(round(sc))
            justification = str(parsed.get("justification") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("L3 VLM call failed for %s: %s", mws["marker"], exc)
            justification = f"L3 VLM call errored: {str(exc)[:120]}"

        passes = score is not None and score >= flag_threshold
        verdicts.append(FigureVerdict(
            marker=mws["marker"], kind=kind, num=num,
            resolved=True,
            visual_id=v["visual_id"], figure_num=v["figure_num"],
            document_id=v["document_id"],
            asset_path=str(img_path),
            claim_sentence=sent,
            vlm_score=score,
            vlm_justification=justification,
            passes=passes,
        ))

    n_resolved = sum(1 for v in verdicts if v.resolved)
    n_passing = sum(1 for v in verdicts if v.passes)
    n_flagged = len(verdicts) - n_passing

    return FinalizeReport(
        draft_id=did,
        n_markers=len(verdicts),
        n_resolved=n_resolved,
        n_passing=n_passing,
        n_flagged=n_flagged,
        verdicts=verdicts,
        vlm_model=model,
        elapsed_s=round(_time.monotonic() - t0, 1),
    )
