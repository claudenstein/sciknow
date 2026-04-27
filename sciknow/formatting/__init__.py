"""Professional LaTeX/PDF formatting pipeline.

Markdown drafts → typed IR → Jinja2-rendered LaTeX → latexmk → PDF.

Public entry points::

    from sciknow.formatting import build_book_pdf, build_book_tex_bundle

    pdf_bytes, log = build_book_pdf(book_id, options=ExportOptions(...))

The IR layer (``ir.py``) is the contract every other module follows.
Templates live under ``templates/<project_type>/``; each owns a
``main.tex.j2`` and ``preamble.tex``.

GPU-free by design: no model weights are touched. latexmk and pandoc
are pure-CPU. Safe to run alongside ongoing inference jobs.
"""
from __future__ import annotations

from sciknow.formatting.options import ExportOptions, list_templates_for_type
from sciknow.formatting.build import (
    build_book_pdf,
    build_book_tex_bundle,
    build_wiki_pdf,
)
from sciknow.formatting.compile import LatexCompileError

__all__ = [
    "ExportOptions",
    "list_templates_for_type",
    "build_book_pdf",
    "build_book_tex_bundle",
    "build_wiki_pdf",
    "LatexCompileError",
]
