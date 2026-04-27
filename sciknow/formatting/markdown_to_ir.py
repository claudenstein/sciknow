"""Markdown → IR conversion.

Uses ``markdown-it-py`` (CommonMark + tables + strikethrough). Citation
markers ``[N]``, ``[N, M]``, ``[1,2,3]`` are recognised in inline text
and replaced with ``Citation`` nodes whose ``citekeys`` are looked up
via the caller-supplied ``cite_map: dict[int, str]`` (global N →
BibTeX citekey).

Math: dollar-delimited math is supported when the dollarmath plugin
is loaded (``mdit_py_plugins.dollarmath``). Falls back gracefully if
the plugin isn't installed — the math expression remains as raw
``$…$`` text and the LaTeX renderer passes it through unmodified.

HTML tables (from MinerU) are detected as ``html_block`` tokens and
converted via pandoc fragment shellout. If pandoc fails, the raw HTML
is dropped with a warning so the document still compiles.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

from sciknow.formatting.ir import (
    Block,
    CodeBlock,
    CodeInline,
    Citation,
    Emph,
    EquationBlock,
    FigureBlock,
    Heading,
    Inline,
    Link,
    ListBlock,
    MathInline,
    Paragraph,
    QuoteBlock,
    RuleBlock,
    Strong,
    TableBlock,
    TextRun,
)

log = logging.getLogger(__name__)


# ── Citation marker recognition ───────────────────────────────────────
#
# Matches: [3], [3,4], [1, 2, 3]. Strictly digits + commas + spaces
# inside the brackets so we don't false-match ``[link](url)`` (which
# the markdown parser turns into a Link node before this regex runs
# on the leftover text fragments anyway, but we belt-and-brace).
_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _split_citations(text: str, cite_map: dict[int, str]) -> list[Inline]:
    """Split a plain text run into TextRun / Citation inlines."""
    out: list[Inline] = []
    i = 0
    for m in _CITE_RE.finditer(text):
        if m.start() > i:
            out.append(TextRun(text=text[i:m.start()]))
        nums = [int(n.strip()) for n in m.group(1).split(",")]
        keys = [cite_map[n] for n in nums if n in cite_map]
        if keys:
            out.append(Citation(citekeys=keys))
        else:
            # Unknown citation — preserve the original marker so authors
            # can spot the dangling reference in the rendered PDF.
            out.append(TextRun(text=m.group(0)))
        i = m.end()
    if i < len(text):
        out.append(TextRun(text=text[i:]))
    return out


# ── Inline conversion ─────────────────────────────────────────────────


def _inline_from_node(node: SyntaxTreeNode, cite_map: dict[int, str]) -> list[Inline]:
    """Walk an ``inline`` node's children and convert to Inline list."""
    out: list[Inline] = []
    for child in node.children:
        t = child.type
        if t == "text":
            out.extend(_split_citations(child.content or "", cite_map))
        elif t == "softbreak" or t == "hardbreak":
            out.append(TextRun(text="\n"))
        elif t == "code_inline":
            out.append(CodeInline(code=child.content or ""))
        elif t == "em":
            out.append(Emph(inlines=_inline_from_node(child, cite_map)))
        elif t == "strong":
            out.append(Strong(inlines=_inline_from_node(child, cite_map)))
        elif t == "s":   # strikethrough — render as text (LaTeX has no canonical strike)
            out.extend(_inline_from_node(child, cite_map))
        elif t == "link":
            href = child.attrs.get("href", "") or ""
            out.append(Link(href=href, inlines=_inline_from_node(child, cite_map)))
        elif t == "image":
            # Inline image inside a paragraph: emit as text fallback —
            # block-level figure handling sees image_paragraph instead.
            alt = (child.attrs.get("alt") or "").strip()
            if alt:
                out.append(TextRun(text=f"[image: {alt}]"))
        elif t == "math_inline":
            out.append(MathInline(tex=child.content or ""))
        elif t == "html_inline":
            # Drop raw inline HTML — almost always a stray <br> or
            # <span> from MinerU prose; can't reliably translate.
            pass
        else:
            # Unknown inline type — fall back to its concatenated text
            # content so we don't lose information silently.
            txt = child.content or ""
            if txt:
                out.extend(_split_citations(txt, cite_map))
    return out


# ── Block conversion ──────────────────────────────────────────────────


def _table_node_to_latex(node: SyntaxTreeNode, cite_map: dict[int, str]) -> TableBlock:
    """Convert a markdown-table SyntaxTreeNode to a TableBlock.

    We render rows as ``cell & cell \\\\`` lines (LaTeX tabular body)
    and infer the column spec from the first row. The renderer wraps
    in ``\\begin{table}\\begin{tabular}{...}...\\end{tabular}\\end{table}``.
    """
    rows: list[list[list[Inline]]] = []
    n_cols = 0
    aligns: list[str] = []   # 'l', 'c', 'r'
    for section in node.children:
        if section.type not in ("thead", "tbody"):
            continue
        for tr in section.children:
            if tr.type != "tr":
                continue
            row: list[list[Inline]] = []
            for cell in tr.children:
                if cell.type not in ("th", "td"):
                    continue
                style = cell.attrs.get("style") or ""
                if not aligns or len(aligns) < n_cols + 1:
                    if "text-align:right" in style:
                        aligns.append("r")
                    elif "text-align:center" in style:
                        aligns.append("c")
                    else:
                        aligns.append("l")
                # Inline content of a cell
                cell_inlines: list[Inline] = []
                for child in cell.children:
                    if child.type == "inline":
                        cell_inlines.extend(_inline_from_node(child, cite_map))
                row.append(cell_inlines)
            n_cols = max(n_cols, len(row))
            rows.append(row)

    if not rows:
        return TableBlock(latex_body="", column_spec="")

    # Build a body where cells are placeholders to be rendered by the
    # template's inline filter. We emit a marker that the renderer can
    # split on; simpler is to render to LaTeX here directly.
    from sciknow.formatting.latex_renderer import inlines_to_latex
    body_lines: list[str] = []
    for r_idx, row in enumerate(rows):
        cells = [inlines_to_latex(c).strip() for c in row]
        # Pad if a row is short
        while len(cells) < n_cols:
            cells.append("")
        body_lines.append(" & ".join(cells) + r" \\")
        if r_idx == 0 and len(rows) > 1:
            body_lines.append(r"\midrule")
    column_spec = "".join(aligns) if aligns else ("l" * n_cols)
    return TableBlock(latex_body="\n".join(body_lines), column_spec=column_spec)


def _html_block_table_to_latex(html: str) -> Optional[TableBlock]:
    """Try to convert a raw HTML <table> block to LaTeX via pandoc.

    Returns ``None`` if pandoc isn't available, the HTML doesn't look
    like a table, or conversion fails.
    """
    s = (html or "").strip()
    if not s.lower().startswith("<table"):
        return None
    if not shutil.which("pandoc"):
        log.warning("pandoc not found; HTML table dropped")
        return None
    try:
        proc = subprocess.run(
            ["pandoc", "-f", "html", "-t", "latex"],
            input=s, capture_output=True, text=True, timeout=15, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.warning("pandoc table conversion failed: %s", e)
        return None
    out = proc.stdout
    # Pandoc typically wraps tables in \begin{longtable}{...}…\end{longtable}
    # or \begin{table}\begin{tabular}{...}…\end{tabular}\end{table}.
    # We pass through verbatim — `latex_body` of TableBlock tolerates a
    # pre-formatted complete table via a bypass marker.
    return TableBlock(latex_body=out, column_spec="__verbatim__")


def _block_from_node(
    node: SyntaxTreeNode,
    cite_map: dict[int, str],
    figure_assets: dict[str, Path],
    image_root: Optional[Path],
) -> list[Block]:
    """Walk a tree node and return zero or more Block instances."""
    t = node.type
    if t == "paragraph":
        # Special-case: a paragraph that contains a single image becomes
        # a FigureBlock so the renderer can use \begin{figure}.
        inline_nodes = [c for c in node.children if c.type == "inline"]
        if (
            inline_nodes
            and len(inline_nodes[0].children) == 1
            and inline_nodes[0].children[0].type == "image"
        ):
            img = inline_nodes[0].children[0]
            src = img.attrs.get("src") or ""
            alt = (img.attrs.get("alt") or "").strip()
            title = (img.attrs.get("title") or "").strip()
            caption = title or alt
            if src and image_root is not None:
                p = Path(src)
                if not p.is_absolute():
                    p = image_root / src
                if p.exists():
                    figure_assets[p.name] = p
                    return [FigureBlock(image_path=p, caption=caption)]
            # Image not found: fall back to caption text
            return [Paragraph(inlines=[TextRun(text=f"[missing figure: {caption or src}]")])]
        # Regular paragraph
        inlines: list[Inline] = []
        for c in node.children:
            if c.type == "inline":
                inlines.extend(_inline_from_node(c, cite_map))
        return [Paragraph(inlines=inlines)]

    if t == "heading":
        depth = int(node.tag[1]) if node.tag and node.tag.startswith("h") else 1
        inlines: list[Inline] = []
        for c in node.children:
            if c.type == "inline":
                inlines.extend(_inline_from_node(c, cite_map))
        # All depths are passed through; caller normalises depth offset.
        return [Heading(inlines=inlines, depth=depth)]

    if t == "code_block" or t == "fence":
        info = (node.info or "").strip() if hasattr(node, "info") else ""
        return [CodeBlock(code=node.content or "", language=info or None)]

    if t == "blockquote":
        children: list[Block] = []
        for c in node.children:
            children.extend(_block_from_node(c, cite_map, figure_assets, image_root))
        return [QuoteBlock(blocks=children)]

    if t == "bullet_list" or t == "ordered_list":
        items: list[list[Block]] = []
        for li in node.children:
            if li.type != "list_item":
                continue
            item_blocks: list[Block] = []
            for c in li.children:
                item_blocks.extend(_block_from_node(c, cite_map, figure_assets, image_root))
            items.append(item_blocks)
        return [ListBlock(ordered=(t == "ordered_list"), items=items)]

    if t == "hr":
        return [RuleBlock()]

    if t == "table":
        return [_table_node_to_latex(node, cite_map)]

    if t == "math_block":
        return [EquationBlock(tex=node.content or "")]

    if t == "html_block":
        # Try table conversion; otherwise drop (raw HTML rarely renders
        # cleanly in LaTeX and the body text usually says the same thing).
        tbl = _html_block_table_to_latex(node.content or "")
        if tbl:
            return [tbl]
        return []

    # Unknown block — recurse into children to avoid silent drops on
    # plugin-added types we haven't handled yet.
    out: list[Block] = []
    for c in node.children or []:
        out.extend(_block_from_node(c, cite_map, figure_assets, image_root))
    return out


# ── Public entry point ────────────────────────────────────────────────


def _build_md_parser() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"breaks": False, "html": True}).enable("table").enable("strikethrough")
    # Try to enable dollar-math; it's optional and graceful if missing.
    try:
        from mdit_py_plugins.dollarmath import dollarmath_plugin
        md.use(dollarmath_plugin, allow_labels=True, allow_space=True, allow_digits=True, double_inline=False)
    except Exception:
        log.debug("dollarmath plugin unavailable; $…$ math will render as text")
    return md


def markdown_to_blocks(
    text: str,
    cite_map: dict[int, str],
    *,
    figure_assets: dict[str, Path],
    image_root: Optional[Path] = None,
) -> list[Block]:
    """Convert one markdown string to a flat list of IR Block objects.

    ``cite_map`` is the global ``[N] → citekey`` mapping (already
    remapped to global numbers by ``BookBibliography.remap_content``).

    ``figure_assets`` is mutated in place: every successfully-resolved
    figure path is recorded under its filename so the build orchestrator
    can copy assets into the compile dir.
    """
    md = _build_md_parser()
    tokens = md.parse(text or "")
    if not tokens:
        return []
    tree = SyntaxTreeNode(tokens)
    blocks: list[Block] = []
    for child in tree.children:
        blocks.extend(_block_from_node(child, cite_map, figure_assets, image_root))
    return blocks


def cite_map_from_global_sources(
    global_sources: Iterable[str],
    citekeys_by_source_key: dict[str, str],
) -> dict[int, str]:
    """Build the global-N → citekey map from BookBibliography output.

    ``global_sources`` is the ``BookBibliography.global_sources`` list
    (each line begins with ``[N]``). ``citekeys_by_source_key`` maps
    the dedup key (= source line minus its ``[N]`` prefix) to the
    BibTeX citekey produced by ``bibtex.build_bibliography``.
    """
    out: dict[int, str] = {}
    for line in global_sources:
        m = re.match(r"^\s*\[(\d+)\]\s*(.*)$", line)
        if not m:
            continue
        n = int(m.group(1))
        key = m.group(2).strip()
        ck = citekeys_by_source_key.get(key)
        if ck:
            out[n] = ck
    return out
