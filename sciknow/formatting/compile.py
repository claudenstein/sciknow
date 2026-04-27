"""latexmk subprocess wrapper.

Runs ``latexmk`` with the right engine flag, captures the full log,
returns ``(pdf_bytes, log)``.

The compile happens in a temporary directory which the caller manages
(see ``build.py``). Figure assets and the .bib file are copied in by
the build orchestrator before this is invoked.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class LatexCompileError(RuntimeError):
    """Raised when latexmk exits non-zero. ``log_text`` carries the full
    log so the GUI/CLI can surface it to the user."""

    def __init__(self, message: str, log_text: str = "", tex_source: str = ""):
        super().__init__(message)
        self.log_text = log_text
        self.tex_source = tex_source


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def compile_tex(
    tex_source: str,
    workdir: Path,
    *,
    engine: str = "lualatex",
    bib_backend: str = "biber",
    timeout_seconds: int = 240,
    extra_files: dict[str, bytes] | None = None,
    main_filename: str = "main.tex",
) -> tuple[bytes, str]:
    """Compile a LaTeX source to PDF.

    Args:
      tex_source: full ``main.tex`` contents.
      workdir: directory to compile in. Must already exist; figure
        assets and ``refs.bib`` must be in place before calling.
      engine: ``lualatex`` (default), ``pdflatex``, or ``xelatex``.
      timeout_seconds: hard cap on the latexmk run.
      extra_files: optional ``{filename: bytes}`` to drop into workdir.
      main_filename: name to write the .tex file as (default ``main.tex``).

    Returns:
      ``(pdf_bytes, log_text)``.

    Raises:
      LatexCompileError if compilation fails.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    main_tex = workdir / main_filename
    main_tex.write_text(tex_source, encoding="utf-8")

    if extra_files:
        for name, data in extra_files.items():
            (workdir / name).write_bytes(data)

    if not _which("latexmk"):
        raise LatexCompileError(
            "latexmk not found on PATH. Install texlive-full "
            "(sudo apt install texlive-full).",
            tex_source=tex_source,
        )
    if not _which(engine):
        raise LatexCompileError(
            f"LaTeX engine {engine!r} not found. "
            f"Install texlive-luatex / texlive-latex-extra.",
            tex_source=tex_source,
        )

    engine_flag = {
        "lualatex": "-lualatex",
        "pdflatex": "-pdf",
        "xelatex":  "-xelatex",
    }.get(engine, "-lualatex")

    cmd = [
        "latexmk",
        engine_flag,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "-no-shell-escape",
    ]
    # Tell latexmk explicitly which bib backend to use. With biblatex+biber
    # (kaobook, classicthesis), biber runs automatically. With natbib+bibtex
    # (elsarticle, IEEEtran), we force the legacy bibtex path.
    if bib_backend == "bibtex":
        cmd.append("-bibtex")
    cmd.append(main_filename)

    log.info("compiling LaTeX in %s with %s", workdir, " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise LatexCompileError(
            f"latexmk timed out after {timeout_seconds}s",
            log_text=str(e),
            tex_source=tex_source,
        )

    log_path = workdir / (Path(main_filename).stem + ".log")
    log_text = ""
    if log_path.exists():
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            log_text = "(could not read .log)"
    full_log = (
        "── latexmk stdout ──\n" + (proc.stdout or "")
        + "\n── latexmk stderr ──\n" + (proc.stderr or "")
        + "\n── .log ──\n" + log_text
    )

    pdf_path = workdir / (Path(main_filename).stem + ".pdf")
    # The PDF's existence is the source of truth: latexmk often returns
    # non-zero on bibtex warnings, undefined references on first pass,
    # or font substitutions, while still producing a perfectly valid
    # PDF. Only fail if no PDF was emitted at all.
    if not pdf_path.exists():
        raise LatexCompileError(
            f"latexmk failed (rc={proc.returncode}); no PDF produced. See log.",
            log_text=full_log,
            tex_source=tex_source,
        )
    return pdf_path.read_bytes(), full_log
