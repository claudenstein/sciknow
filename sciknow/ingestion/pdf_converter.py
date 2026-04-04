"""
PDF to Markdown converter using Marker (marker-pdf).

Marker handles both text-based and scanned PDFs, runs models on GPU,
and produces clean markdown with LaTeX math and table support.
"""
import sys
from pathlib import Path


class ConversionError(Exception):
    pass


def convert(pdf_path: Path, output_dir: Path) -> Path:
    """
    Convert a PDF to markdown using Marker.
    Returns the path to the produced .md file.

    Output layout:
        output_dir/
          <stem>/
            <stem>.md
            images/
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        from marker.config.parser import ConfigParser
    except ImportError:
        raise ConversionError(
            "marker-pdf not installed. Run:\n"
            "  uv pip install marker-pdf"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem
    paper_out_dir = output_dir / stem
    paper_out_dir.mkdir(parents=True, exist_ok=True)
    md_path = paper_out_dir / f"{stem}.md"

    try:
        config_parser = ConfigParser({"output_format": "markdown"})
        models = create_model_dict()
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=models,
        )
        rendered = converter(str(pdf_path))
        text, _, images = text_from_rendered(rendered)

        # Write markdown
        md_path.write_text(text, encoding="utf-8")

        # Save extracted images
        if images:
            img_dir = paper_out_dir / "images"
            img_dir.mkdir(exist_ok=True)
            for img_name, img in images.items():
                img.save(img_dir / img_name)

    except Exception as exc:
        raise ConversionError(f"Marker failed for {pdf_path.name}: {exc}") from exc

    return md_path
