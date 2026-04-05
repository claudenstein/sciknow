"""
PDF to Markdown/JSON converter using Marker (marker-pdf).

Conversion modes:
  - JSON (default): uses Marker's structured JSONRenderer, which preserves
    block types (SectionHeader, Text, Table, Equation, …) with heading levels
    and section hierarchy. Preferred for new ingestion — gives the section-aware
    chunker exact structural signals instead of regex heuristics.

  - Markdown (legacy): used automatically as a fallback if JSON conversion fails.

The returned ConversionResult carries:
  - json_path  : Path to the saved .json file (None in markdown fallback)
  - md_path    : Path to the saved .md file (None in JSON mode)
  - text       : Plain-text representation for metadata extraction (always set)
  - json_data  : Parsed dict from the JSON file (None in markdown fallback)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


class ConversionError(Exception):
    pass


@dataclass
class ConversionResult:
    """Returned by convert(). Carries all forms of the converted document."""
    # JSON mode fields
    json_path: Path | None = None
    json_data: dict | None = None

    # Markdown fallback fields
    md_path: Path | None = None

    # Always set: plain text for metadata extraction
    text: str = ""

    @property
    def is_json(self) -> bool:
        return self.json_data is not None


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


def convert(pdf_path: Path, output_dir: Path) -> ConversionResult:
    """
    Convert a PDF to structured JSON (preferred) with plain-text fallback.

    Output layout:
        output_dir/
          <stem>/
            <stem>.json     ← Marker JSON (block-structured)
            <stem>.md       ← only written on markdown fallback
            images/         ← extracted images (markdown mode only)

    Returns a ConversionResult; always sets .text for metadata extraction.
    """
    PdfConverter, _ = _load_marker()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    paper_out_dir = output_dir / stem
    paper_out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Primary path: JSON output
    # ------------------------------------------------------------------
    json_path = paper_out_dir / f"{stem}.json"
    try:
        models = _get_models()
        converter = PdfConverter(
            config=_marker_config(),
            artifact_dict=models,
            renderer='marker.renderers.json.JSONRenderer',
        )
        rendered = converter(str(pdf_path))

        # rendered is a JSONOutput Pydantic model
        raw = rendered.model_dump_json(exclude=['metadata'])
        json_path.write_text(raw, encoding='utf-8')

        json_data = json.loads(raw)
        text = extract_text_from_json(json_data)

        if not text.strip():
            raise ConversionError("JSON conversion produced empty text")

        return ConversionResult(
            json_path=json_path,
            json_data=json_data,
            text=text,
        )

    except ConversionError:
        raise
    except Exception as exc:
        # Fall through to markdown fallback
        _fallback_reason = str(exc)

    # ------------------------------------------------------------------
    # Fallback: markdown output (keeps legacy behaviour)
    # ------------------------------------------------------------------
    md_path = paper_out_dir / f"{stem}.md"
    try:
        from marker.output import text_from_rendered
        from marker.config.parser import ConfigParser

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
            md_path=md_path,
            text=text,
        )

    except Exception as exc:
        raise ConversionError(
            f"Both JSON and markdown conversion failed for {pdf_path.name}.\n"
            f"JSON error: {_fallback_reason}\n"
            f"Markdown error: {exc}"
        ) from exc
