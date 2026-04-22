"""
PDF to Markdown/JSON converter.

Supports two backends, chosen via settings.pdf_converter_backend:

  - **mineru** (default, primary) — OpenDataLab MinerU 2.5 pipeline backend.
    Best quality on scientific papers per OmniDocBench: dedicated MFD/MFR models
    for formulas, structured table reconstruction, consistent reading-order on
    multi-column layouts. Output is MinerU's content_list.json (a flat list of
    typed blocks: text with text_level, table with HTML body, equation with
    LaTeX, image, code, list, …).

  - **marker** (fallback) — datalab-to/marker with Surya OCR + layout models.
    Kept as a fallback for documents where MinerU fails (rare). Produces
    Marker's block-tree JSON or, if that also fails, plain markdown.

`convert()` dispatches based on settings.pdf_converter_backend:
  - "mineru"  → MinerU only; raises ConversionError on failure
  - "marker"  → Marker JSON → Marker markdown (legacy behaviour)
  - "auto"    → MinerU → Marker JSON → Marker markdown (default)

The returned ConversionResult carries:
  - backend      : which backend produced the result
  - json_path    : Marker JSON file (marker backend only)
  - json_data    : parsed Marker JSON tree (marker backend only)
  - content_list : MinerU content_list.json as a list of dicts (mineru only)
  - content_list_path : path to the saved content_list.json (mineru only)
  - md_path      : markdown file (marker markdown fallback only)
  - text         : plain-text representation for metadata extraction (always set)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Backend = Literal["mineru", "marker_json", "marker_md"]


class ConversionError(Exception):
    pass


def _marker_version() -> str | None:
    """Return the marker-pdf package version for the converter stamp."""
    try:
        import marker as _marker_pkg
        return f"marker-pdf {getattr(_marker_pkg, '__version__', '?')}"
    except Exception:
        return None


@dataclass
class ConversionResult:
    """Returned by convert(). Carries all forms of the converted document."""
    # Which backend produced this result.
    backend: Backend = "marker_json"

    # MinerU fields (backend="mineru")
    content_list: list[dict] | None = None
    content_list_path: Path | None = None

    # Marker JSON fields (backend="marker_json")
    json_path: Path | None = None
    json_data: dict | None = None

    # Marker markdown fallback fields (backend="marker_md")
    md_path: Path | None = None

    # Always set: plain text for metadata extraction
    text: str = ""

    # Phase 54.6.211 (roadmap 3.1.6 Phase 1) — free-form provenance
    # for the stamp on documents.converter_backend / converter_version.
    # `converter_mode` is the finer variant of `backend`:
    #   "mineru-pipeline"          — MinerU 2.0 stack (legacy)
    #   "mineru-vlm-pro-vllm"      — MinerU 2.5-Pro VLM via vllm
    #   "mineru-vlm-pro-transformers" — MinerU 2.5-Pro VLM via HF
    #   "mineru-vlm-auto"          — MinerU VLM with auto-engine
    #   "marker-json" / "marker-md"
    # `converter_version` is the producer's version string
    # (e.g. "mineru 3.0.9", "marker-pdf 1.6.2"); None if unavailable.
    converter_mode: str | None = None
    converter_version: str | None = None

    @property
    def is_json(self) -> bool:
        """True if we have structured output (either backend). The chunker
        uses this to decide between structure-aware parsing and markdown regex."""
        return self.backend in ("mineru", "marker_json")


# ---------------------------------------------------------------------------
# Plain-text extraction from Marker JSON
# ---------------------------------------------------------------------------

_HTML_ENTITIES = {
    '&amp;': '&', '&lt;': '<', '&gt;': '>',
    '&quot;': '"', '&#39;': "'", '&nbsp;': ' ',
}
_HTML_ENTITY_RE = re.compile('|'.join(re.escape(k) for k in _HTML_ENTITIES))


def _strip_html(html: str) -> str:
    """Remove all HTML tags and decode common entities, collapsing whitespace."""
    text = re.sub(r'<[^>]+>', ' ', html or '')
    text = _HTML_ENTITY_RE.sub(lambda m: _HTML_ENTITIES[m.group(0)], text)
    return re.sub(r'\s+', ' ', text).strip()


def _table_to_text(html: str) -> str:
    """
    Convert an HTML table to a plain-text pipe-delimited representation.
    Falls back to stripped HTML if parsing fails.
    """
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html or '', re.DOTALL | re.IGNORECASE)
    if not rows:
        return _strip_html(html)

    lines = []
    for row_html in rows:
        cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, re.DOTALL | re.IGNORECASE)
        cells = [_strip_html(c) for c in cells]
        if any(cells):
            lines.append(' | '.join(cells))

    return '\n'.join(lines) if lines else _strip_html(html)


# Block types whose HTML content should be collected as paragraph text
_TEXT_BLOCKS = {
    'Text', 'ListItem', 'Caption', 'Footnote', 'Code',
    'Handwriting', 'TextInlineMath',
}
# Block types that are structural containers (recurse into children)
_CONTAINER_BLOCKS = {
    'Page', 'ListGroup', 'FigureGroup', 'PictureGroup', 'TableGroup',
}
# Block types to skip entirely
_SKIP_BLOCKS = {
    'PageHeader', 'PageFooter', 'Figure', 'Picture', 'Form',
    'ComplexRegion', 'TableOfContents', 'Reference',
}


def extract_text_from_json(json_data: dict) -> str:
    """
    Walk the Marker JSON tree and produce a clean plain-text string.
    Used to feed the metadata extraction layer (which expects a text document).
    """
    parts: list[str] = []

    def _walk(node: dict) -> None:
        bt = node.get('block_type', '')
        html = node.get('html') or ''
        children = node.get('children') or []

        if bt in _SKIP_BLOCKS:
            return
        if bt == 'SectionHeader':
            heading = _strip_html(html)
            if heading:
                parts.append(f'\n\n## {heading}\n')
        elif bt == 'Table':
            table_text = _table_to_text(html)
            if table_text:
                parts.append(table_text)
        elif bt == 'Equation':
            eq = _strip_html(html)
            if eq:
                parts.append(eq)
        elif bt in _TEXT_BLOCKS:
            text = _strip_html(html)
            if text:
                parts.append(text)
        elif bt in _CONTAINER_BLOCKS:
            for child in children:
                _walk(child)
        # Any other block type: recurse into children
        else:
            for child in children:
                _walk(child)

    for page in json_data.get('children', []):
        _walk(page)

    return '\n\n'.join(p.strip() for p in parts if p.strip())


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

def _load_marker():
    """Lazy-load Marker models. Raises ConversionError if not installed."""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        return PdfConverter, create_model_dict
    except ImportError:
        raise ConversionError(
            "marker-pdf not installed. Run:\n"
            "  uv pip install marker-pdf"
        )


# Module-level model cache so models are loaded once per process
_models: dict | None = None


def _get_models() -> dict:
    global _models
    if _models is None:
        _, create_model_dict = _load_marker()
        _models = create_model_dict()
    return _models


def _marker_config() -> dict:
    """Marker runtime tuning knobs driven by settings.

    `batch_multiplier` multiplies Marker/Surya's internal batch sizes. Each
    unit is ~5GB VRAM peak at the OCR stage. Conservative default is 2.
    """
    from sciknow.config import settings
    cfg: dict = {}
    mult = getattr(settings, "marker_batch_multiplier", 1)
    if mult and mult > 1:
        cfg["batch_multiplier"] = mult
    return cfg


# ---------------------------------------------------------------------------
# MinerU plain-text extraction (for the metadata stage)
# ---------------------------------------------------------------------------

# Content_list item types that contribute to the plain-text rendering.
# (image/chart/seal/page_number/header/footer are skipped as noise.)
_MINERU_TEXTISH_TYPES = {"text", "equation", "code", "list"}


def extract_text_from_content_list(content_list: list[dict]) -> str:
    """
    Flatten MinerU's content_list.json into a plain-text string for the
    metadata extraction stage. Preserves heading markers so the first few
    hundred chars still look like a well-structured document.
    """
    parts: list[str] = []

    for item in content_list or []:
        itype = item.get("type", "")

        if itype == "text":
            text = (item.get("text") or "").strip()
            if not text:
                continue
            level = item.get("text_level") or 0
            if level and level >= 1:
                # Use markdown-ish heading for downstream text consumers
                hashes = "#" * min(level, 4)
                parts.append(f"\n\n{hashes} {text}\n")
            else:
                parts.append(text)

        elif itype == "equation":
            eq = (item.get("text") or "").strip()
            if eq:
                parts.append(eq)

        elif itype == "code":
            code = (item.get("code_body") or "").strip()
            if code:
                parts.append(code)

        elif itype == "list":
            for li in item.get("list_items") or []:
                if isinstance(li, str) and li.strip():
                    parts.append(f"- {li.strip()}")
                elif isinstance(li, dict):
                    txt = (li.get("text") or "").strip()
                    if txt:
                        parts.append(f"- {txt}")

        elif itype == "table":
            body = item.get("table_body") or ""
            if body:
                parts.append(_table_to_text(body))

        # image/chart/seal/header/footer/page_number/page_footnote/aside_text → skip

    return "\n\n".join(p.strip() for p in parts if p.strip())


# ---------------------------------------------------------------------------
# MinerU backend
# ---------------------------------------------------------------------------

def _load_mineru():
    """Lazy-load MinerU. Raises ConversionError if not installed."""
    try:
        from mineru.cli.common import do_parse, read_fn
        return do_parse, read_fn
    except ImportError as exc:
        raise ConversionError(
            f"mineru not installed ({exc}). Run:\n"
            f"  uv add 'mineru[core]'"
        )


# Phase 21 — MinerU 2.5-Pro VLM model identifiers. The Pro variant
# (95.69 on OmniDocBench v1.6) was published 2026-04-09 and uses the
# same Qwen2VL 1.2B architecture as the base 2.5 — only the training
# data was upgraded. Drop-in compatible with vlm-auto-engine.
_MINERU_PRO_DEFAULT_HF = "opendatalab/MinerU2.5-Pro-2604-1.2B"


def _patch_mineru_to_pro_model(model_name_hf: str) -> None:
    """Phase 21 — monkey-patch mineru.utils.enum_class.ModelPath so the
    VLM backend pulls Pro instead of the base 2.5 weights.

    The mineru pip package (3.0.9) hardcodes the VLM model name as a
    class attribute on ModelPath, with no env var or config-file
    override. Until upstream exposes this, we patch the constant
    in-place before any do_parse call. Same Qwen2VL architecture so
    no other changes are needed.

    Idempotent — safe to call multiple times.
    """
    try:
        from mineru.utils.enum_class import ModelPath
    except ImportError:
        return  # mineru not installed; nothing to patch
    if getattr(ModelPath, "vlm_root_hf", None) == model_name_hf:
        return  # already patched
    ModelPath.vlm_root_hf = model_name_hf
    # ModelScope variant: capitalise the org name to match upstream's
    # convention (opendatalab → OpenDataLab on ModelScope).
    ModelPath.vlm_root_modelscope = model_name_hf.replace(
        "opendatalab/", "OpenDataLab/"
    )


def _convert_mineru(
    pdf_path: Path,
    output_dir: Path,
    *,
    use_vlm_pro: bool = False,
    vlm_model_name: str | None = None,
) -> ConversionResult:
    """
    Run MinerU on a single PDF and return a ConversionResult.

    Two modes:

    - **pipeline (default)** — legacy MinerU 2.0 stack: layout (PP-DocLayoutV2)
      + formula (UniMERNet) + table (SLANet+) + OCR (paddleocr_torch). CPU
      friendly, ~6GB peak VRAM with all models loaded. OmniDocBench v1.5: 86.2.

    - **vlm-pro (opt-in)** — MinerU 2.5-Pro VLM (Qwen2VL 1.2B fine-tuned on
      MinerU's data engine). OmniDocBench v1.6: 95.69 (Phase 21). Requires a
      GPU with ~4GB free VRAM and `mineru[vlm]` extras. Set
      settings.pdf_converter_backend = "mineru-vlm-pro" to enable.

    Both backends emit the same content_list.json schema, so the chunker's
    parse_sections_from_mineru handles them identically.

    MinerU writes files to `{output_dir}/{pdf_stem}/{pipeline|vlm}/`:
        - content_list.json  ← primary structured output (what we parse)
        - middle.json        ← intermediate rep with bboxes (kept for debugging)
        - {stem}.md          ← disabled (we parse content_list directly)
        - *_layout.pdf       ← disabled (visualisation, we don't need)
        - *_span.pdf         ← disabled (same)

    Model weights download on first call (a few GB for VLM, ~2GB for
    pipeline); subsequent calls reuse the cache. Models stay resident in
    VRAM for the Python process lifetime — batching matters for throughput.
    """
    do_parse, read_fn = _load_mineru()

    # Phase 54.6.211 (roadmap 3.1.6 Phase 1) — honour the
    # `mineru_vlm_backend` setting when VLM-Pro is selected. The
    # MinerU CLI accepts three variants of the VLM engine name:
    #   auto         → let MinerU pick (vllm if available)
    #   vllm         → vllm-engine (fast path, requires mineru[vllm])
    #   transformers → transformers path (safe fallback)
    # Passed through as do_parse(backend=...); MinerU maps these to
    # its internal engine selector. Pipeline mode ignores the setting
    # entirely because it doesn't use a VLM.
    if use_vlm_pro:
        _patch_mineru_to_pro_model(vlm_model_name or _MINERU_PRO_DEFAULT_HF)
        from sciknow.config import settings as _s
        vlm_backend_pref = (_s.mineru_vlm_backend or "auto").strip().lower()
        if vlm_backend_pref == "vllm":
            chosen_backend = "vlm-vllm-engine"
            converter_mode = "mineru-vlm-pro-vllm"
        elif vlm_backend_pref == "transformers":
            chosen_backend = "vlm-transformers"
            converter_mode = "mineru-vlm-pro-transformers"
        else:
            chosen_backend = "vlm-auto-engine"
            converter_mode = "mineru-vlm-auto"
    else:
        chosen_backend = "pipeline"
        converter_mode = "mineru-pipeline"

    # Capture MinerU version for the documents.converter_version stamp.
    try:
        import mineru as _mineru_pkg
        mineru_version = f"mineru {getattr(_mineru_pkg, '__version__', '?')}"
    except Exception:
        mineru_version = None

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    pdf_bytes = read_fn(str(pdf_path))

    # Run MinerU — chosen backend, English, with formula + table enabled.
    # Disable all the visualisation/dump outputs we don't use to keep the
    # per-document footprint light.
    try:
        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[stem],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["en"],
            backend=chosen_backend,
            parse_method="auto",
            formula_enable=True,
            table_enable=True,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=False,
            f_dump_middle_json=True,   # small + useful for debugging
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,  # this is the file we consume
        )
    except (ImportError, ModuleNotFoundError) as exc:
        # The vlm-auto-engine path needs `mineru[vlm]` extras (vllm or
        # transformers). Surface a useful install hint instead of a
        # cryptic ModuleNotFoundError.
        if use_vlm_pro:
            raise ConversionError(
                f"MinerU VLM backend missing dependencies ({exc}).\n"
                f"Install with: uv add 'mineru[vlm]'\n"
                f"Or fall back to the pipeline backend: set\n"
                f"  PDF_CONVERTER_BACKEND=mineru\n"
                f"in your .env file."
            )
        raise

    # MinerU's output dir layout depends on the backend:
    #   pipeline → {stem}/auto/  or  {stem}/pipeline/
    #   vlm      → {stem}/vlm/   (or under /auto/ in older builds)
    content_list_path = output_dir / stem / "auto" / f"{stem}_content_list.json"
    if not content_list_path.exists():
        for sub in ("pipeline", "vlm"):
            alt = output_dir / stem / sub / f"{stem}_content_list.json"
            if alt.exists():
                content_list_path = alt
                break
    if not content_list_path.exists():
        # Last-resort glob: MinerU versions move things around.
        candidates = list((output_dir / stem).rglob("*content_list*.json"))
        if candidates:
            content_list_path = candidates[0]

    if not content_list_path.exists():
        raise ConversionError(
            f"MinerU ran but content_list.json not found under {output_dir / stem}"
        )

    content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
    text = extract_text_from_content_list(content_list)

    if not text.strip():
        raise ConversionError("MinerU produced empty text")

    return ConversionResult(
        backend="mineru",
        content_list=content_list,
        content_list_path=content_list_path,
        text=text,
        converter_mode=converter_mode,
        converter_version=mineru_version,
    )


def _convert_marker_json(pdf_path: Path, output_dir: Path) -> ConversionResult:
    """Marker structured JSON path (legacy primary, now fallback)."""
    PdfConverter, _ = _load_marker()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    paper_out_dir = output_dir / stem
    paper_out_dir.mkdir(parents=True, exist_ok=True)

    json_path = paper_out_dir / f"{stem}.json"
    models = _get_models()
    converter = PdfConverter(
        config=_marker_config(),
        artifact_dict=models,
        renderer='marker.renderers.json.JSONRenderer',
    )
    rendered = converter(str(pdf_path))

    raw = rendered.model_dump_json(exclude=['metadata'])
    json_path.write_text(raw, encoding='utf-8')

    json_data = json.loads(raw)
    text = extract_text_from_json(json_data)

    if not text.strip():
        raise ConversionError("Marker JSON conversion produced empty text")

    return ConversionResult(
        backend="marker_json",
        json_path=json_path,
        json_data=json_data,
        text=text,
        converter_mode="marker-json",
        converter_version=_marker_version(),
    )


def _convert_marker_markdown(pdf_path: Path, output_dir: Path) -> ConversionResult:
    """Marker markdown path (last-ditch fallback)."""
    PdfConverter, _ = _load_marker()
    from marker.output import text_from_rendered
    from marker.config.parser import ConfigParser

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    paper_out_dir = output_dir / stem
    paper_out_dir.mkdir(parents=True, exist_ok=True)

    md_path = paper_out_dir / f"{stem}.md"

    models = _get_models()
    config_parser = ConfigParser({
        'output_format': 'markdown',
        **_marker_config(),
    })
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=models,
    )
    rendered = converter(str(pdf_path))
    text, _, images = text_from_rendered(rendered)

    md_path.write_text(text, encoding='utf-8')

    if images:
        img_dir = paper_out_dir / 'images'
        img_dir.mkdir(exist_ok=True)
        for img_name, img in images.items():
            img.save(img_dir / img_name)

    return ConversionResult(
        backend="marker_md",
        md_path=md_path,
        text=text,
        converter_mode="marker-md",
        converter_version=_marker_version(),
    )


_MINERU_DEPRECATION_WARNED: bool = False


def _warn_mineru_pipeline_deprecated() -> None:
    """One-shot deprecation warning for PDF_CONVERTER_BACKEND=mineru.

    Phase 54.6.212 (roadmap 3.1.6 Phase 2) flipped the default to
    VLM-Pro; the pipeline-mode backend is retained as fallback only.
    Users who explicitly pinned `PDF_CONVERTER_BACKEND=mineru` get
    one warning per process so the signal isn't buried in verbose
    ingest output, but every run surfaces it once.
    """
    global _MINERU_DEPRECATION_WARNED
    if _MINERU_DEPRECATION_WARNED:
        return
    _MINERU_DEPRECATION_WARNED = True
    import warnings
    warnings.warn(
        "PDF_CONVERTER_BACKEND=mineru pins the deprecated pipeline "
        "backend (54.6.212, roadmap 3.1.6). OmniDocBench v1.6 scores: "
        "pipeline 86.2 vs VLM-Pro 95.69. Drop the setting to use "
        "`auto` (premium-first fallback chain: VLM-Pro → pipeline → "
        "Marker) or set `mineru-vlm-pro` for VLM-Pro only.",
        DeprecationWarning,
        stacklevel=2,
    )


def _vlm_extras_missing(exc: BaseException) -> bool:
    """True if the exception indicates mineru[vlm]/[vllm] extras are
    not installed. Used by auto-dispatch to silently fall through to
    pipeline mode on installs without the VLM dependencies, while
    still propagating genuine per-PDF conversion errors."""
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return True
    msg = str(exc).lower()
    return "missing dependencies" in msg or "mineru[vlm]" in msg


def convert(pdf_path: Path, output_dir: Path) -> ConversionResult:
    """
    Convert a PDF to structured output. Dispatches based on
    settings.pdf_converter_backend.

    Backends:
      - "auto"            : **default post-54.6.212.** Premium-first
                            fallback chain: VLM-Pro → pipeline →
                            Marker JSON → Marker markdown. VLM-Pro
                            is silently skipped if `mineru[vllm]` /
                            `mineru[transformers]` extras aren't
                            installed (install detection via
                            `_vlm_extras_missing`), so existing
                            installations keep working under
                            pipeline while new ones automatically
                            pick up the quality bump.
      - "mineru-vlm-pro"  : MinerU 2.5-Pro VLM only (raises on
                            failure; requires the VLM extras +
                            GPU). Use to verify VLM-Pro is healthy.
      - "mineru"          : **DEPRECATED** (54.6.212) — pipeline
                            MinerU only (raises on failure). Emits
                            a one-shot DeprecationWarning. Retained
                            as fallback within "auto" but no longer
                            a recommended pin.
      - "marker"          : Marker JSON → Marker markdown (legacy,
                            handy for pure scans).

    Always returns a ConversionResult with `.backend` set to the
    backend that actually produced the result, `.converter_mode` set
    to the finer variant (pipeline / vlm-pro-vllm / vlm-pro-
    transformers / vlm-auto / marker-json / marker-md), and `.text`
    populated for metadata extraction.
    """
    from sciknow.config import settings

    backend_setting = getattr(settings, "pdf_converter_backend", "auto")
    vlm_model_name = getattr(settings, "mineru_vlm_model", None)
    errors: list[str] = []

    # ---- 0. MinerU 2.5-Pro VLM (explicit opt-in) ----
    if backend_setting == "mineru-vlm-pro":
        try:
            return _convert_mineru(
                pdf_path, output_dir,
                use_vlm_pro=True, vlm_model_name=vlm_model_name,
            )
        except Exception as exc:
            # Explicit Pro mode: do not silently fall back to a worse
            # backend, the user opted in to Pro for a reason.
            raise ConversionError(f"MinerU 2.5-Pro VLM: {exc}") from exc

    # ---- 1. MinerU 2.5-Pro VLM (primary in "auto" post-54.6.212) ----
    if backend_setting == "auto":
        try:
            return _convert_mineru(
                pdf_path, output_dir,
                use_vlm_pro=True, vlm_model_name=vlm_model_name,
            )
        except Exception as exc:
            if _vlm_extras_missing(exc):
                # Silent fall-through — install doesn't have the VLM
                # deps yet; pipeline mode can still handle this PDF.
                # We do NOT log an error here because this is the
                # expected path on boxes where the user hasn't run
                # `uv add 'mineru[vllm]'` yet.
                errors.append(f"VLM-Pro: extras not installed ({exc})")
            else:
                # Genuine convert error (OOM, malformed PDF, model
                # download failure). Log and fall through to pipeline
                # since auto mode promises a fallback chain.
                errors.append(f"VLM-Pro: {exc}")

    # ---- 2. MinerU pipeline (fallback in "auto"; pinned in "mineru") ----
    if backend_setting in ("auto", "mineru"):
        if backend_setting == "mineru":
            _warn_mineru_pipeline_deprecated()
        try:
            return _convert_mineru(pdf_path, output_dir, use_vlm_pro=False)
        except Exception as exc:
            msg = f"MinerU: {exc}"
            errors.append(msg)
            if backend_setting == "mineru":
                # Explicit MinerU-only mode: do not fall back.
                raise ConversionError(msg) from exc

    # ---- 3. Marker JSON (primary in "marker", fallback in "auto") ----
    if backend_setting in ("auto", "marker"):
        try:
            return _convert_marker_json(pdf_path, output_dir)
        except Exception as exc:
            errors.append(f"Marker JSON: {exc}")

    # ---- 4. Marker markdown (last resort for "auto" and "marker") ----
    if backend_setting in ("auto", "marker"):
        try:
            return _convert_marker_markdown(pdf_path, output_dir)
        except Exception as exc:
            errors.append(f"Marker markdown: {exc}")

    raise ConversionError(
        f"All configured PDF converter backends failed for {pdf_path.name}:\n  "
        + "\n  ".join(errors)
    )
