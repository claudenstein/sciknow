"""Phase 54.6.72 (#1) — helpers for vision-LLM captioning of figures/charts.

Kept in a standalone module because (a) the CLI command imports it and
(b) the web API's future "recaption this single figure" endpoint will
too. Centralizing the prompts + path resolver also makes them easy to
A/B tune without editing scattered call sites.
"""
from __future__ import annotations

from pathlib import Path


# ════════════════════════════════════════════════════════════════════════
# VLM prompts
# ════════════════════════════════════════════════════════════════════════


PROMPT_SYSTEM = (
    "You are captioning a scientific figure or chart for a research-paper "
    "retrieval index. Your caption will be embedded and used to surface "
    "this image when someone queries the corpus.\n\n"
    "Rules:\n"
    "  - One paragraph, 2-4 sentences, 40-120 words.\n"
    "  - Lead with what the image SHOWS (type of plot, axes, units, "
    "compared variables). Then what it COMMUNICATES (the finding or "
    "pattern visible in the data).\n"
    "  - Use concrete domain language when visible (e.g. 'solar irradiance "
    "(W/m²)', 'outgoing longwave radiation', 'Pleistocene glaciation'), "
    "not generic phrasing.\n"
    "  - NO preamble like 'This image shows' — just start with the content.\n"
    "  - NO speculation beyond what the image + existing caption make "
    "explicit. If a feature is ambiguous, say so.\n"
    "  - NO markdown, NO lists, NO headers. Plain prose only."
)


PROMPT_USER = (
    "This is a {kind} extracted from a scientific paper. "
    "The paper's own caption was: {existing_caption}\n\n"
    "Write a retrieval-ready description of the image."
)


# ════════════════════════════════════════════════════════════════════════
# Disk path resolution
# ════════════════════════════════════════════════════════════════════════


def resolve_asset_path(doc_id: str, asset_path: str) -> Path | None:
    """Resolve a visual's `asset_path` against the active project's
    mineru_output subtree.

    MinerU writes images to one of three layouts depending on backend:

      * ``<slug>/auto/images/<sha>.jpg``   — pipeline mode (MinerU 2.0)
      * ``<slug>/vlm/images/<sha>.jpg``    — VLM-Pro mode (MinerU 2.5+, current default)
      * ``<slug>/images/<sha>.jpg``        — bare layout (older outputs)

    The row stores the relative path ``images/<sha>.jpg``, so we probe
    each candidate infix in priority order until one resolves. This
    mirrors the fix shipped in Phase 54.6.303 to ``/api/visuals/image``
    in the web app — without it, every visual ingested with VLM-Pro
    silently failed to resolve, leaving caption-visuals and
    finalize_draft.verify_draft_figures_l3 unable to find ANY figure
    file on disk (the L3 verifier degrades to L2; caption-visuals
    skips with "image file missing on disk").

    Returns None if the file is missing (e.g. a mineru_output cleanup
    deleted the image after the row was created). Callers should skip
    gracefully rather than error.

    Path-traversal guard: the resolved path must be under the doc's
    mineru_output dir (same guard as /api/visuals/image — see 54.6.61).
    """
    from sciknow.config import settings

    doc_dir = Path(settings.data_dir) / "mineru_output" / str(doc_id)
    if not doc_dir.is_dir():
        return None
    doc_real = doc_dir.resolve()
    # Probe each backend's layout. Order matters only for performance
    # — once a candidate hits, return it.
    infixes = ("auto", "vlm", None)
    for sub in doc_dir.iterdir():
        if not sub.is_dir():
            continue
        for infix in infixes:
            candidate = (sub / infix / asset_path) if infix else (sub / asset_path)
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved.is_file() and doc_real in resolved.parents:
                return resolved
    return None
