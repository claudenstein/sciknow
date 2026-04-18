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

    MinerU writes images to
    ``data/mineru_output/<doc_id>/<slug>/auto/images/<sha>.jpg`` and
    the row stores the relative path ``images/<sha>.jpg`` — so we
    resolve against each ``<slug>/auto/`` subfolder until one hits.

    Returns None if the file is missing (e.g. a mineru_output cleanup
    deleted the image after the row was created). Callers should skip
    gracefully rather than error — a missing file is a valid skip
    reason, not a bug.

    Path-traversal guard: the resolved path must be under the doc's
    mineru_output dir (same guard as /api/visuals/image in the web
    app — see 54.6.61).
    """
    from sciknow.config import settings

    doc_dir = Path(settings.data_dir) / "mineru_output" / str(doc_id)
    if not doc_dir.is_dir():
        return None
    for sub in doc_dir.iterdir():
        if not sub.is_dir():
            continue
        # Primary layout: <slug>/auto/images/<sha>.jpg
        primary = (sub / "auto" / asset_path).resolve()
        if primary.is_file() and doc_dir.resolve() in primary.parents:
            return primary
        # Fallback: no auto/ (older MinerU outputs)
        fallback = (sub / asset_path).resolve()
        if fallback.is_file() and doc_dir.resolve() in fallback.parents:
            return fallback
    return None
