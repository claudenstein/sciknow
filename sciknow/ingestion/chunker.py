"""
Section-aware chunker for scientific paper documents.

Two entry points:
  parse_sections_from_json(json_data)  — preferred; reads Marker's structured
                                         JSON output for exact block-type-based
                                         section detection and heading levels.
  parse_sections(markdown)             — legacy fallback; regex over plain
                                         markdown headings.

Both return list[Section], consumed by chunk_document().
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Section type classification  (shared by both entry points)
# ---------------------------------------------------------------------------

# (canonical_type, list of lowercase prefixes to match heading text against)
_SECTION_PATTERNS: list[tuple[str, list[str]]] = [
    ("abstract",       ["abstract"]),
    ("introduction",   ["introduction", "background", "motivation", "overview"]),
    ("related_work",   ["related work", "prior work", "literature review",
                        "previous work", "state of the art"]),
    ("methods",        ["method", "material", "experiment", "data and method",
                        "approach", "model", "dataset", "proposed",
                        "framework", "architecture", "observation",
                        "instrument", "data collection", "simulation"]),
    ("results",        ["result", "finding", "performance", "evaluation",
                        "experiment", "measurement", "observation"]),
    ("discussion",     ["discussion", "analysis", "interpretation", "implication"]),
    ("conclusion",     ["conclusion", "summary", "outlook", "future work",
                        "closing", "concluding"]),
    ("acknowledgments",["acknowledgment", "acknowledgement", "funding", "support"]),
    ("references",     ["reference", "bibliography"]),
    ("appendix",       ["appendix", "supplement"]),
]

_SKIP_SECTIONS = {"references", "acknowledgments"}


def _classify_heading(heading: str) -> str:
    """Map a heading string to a canonical section type."""
    normalised = re.sub(r'^[\d.]+\s*', '', heading.lower().strip())
    normalised = re.sub(r'[^\w\s]', '', normalised).strip()
    for section_type, prefixes in _SECTION_PATTERNS:
        for prefix in prefixes:
            if normalised.startswith(prefix):
                return section_type
    return "unknown"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Section:
    section_type: str
    section_title: str
    section_index: int
    content: str

    @property
    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class Chunk:
    section_type: str
    section_title: str
    section_index: int
    chunk_index: int        # global index within the document
    content: str            # includes context header
    raw_content: str        # content without header
    content_tokens: int
    char_start: int
    char_end: int


# ---------------------------------------------------------------------------
# Chunking parameters per section type
# ---------------------------------------------------------------------------

@dataclass
class ChunkParams:
    target_tokens: int
    overlap_tokens: int
    keep_whole_if_under: int


_PARAMS: dict[str, ChunkParams] = {
    "abstract":        ChunkParams(512,  0,   512),
    "introduction":    ChunkParams(512,  64,  768),
    "related_work":    ChunkParams(512,  64,  768),
    "methods":         ChunkParams(512,  128, 768),
    "results":         ChunkParams(512,  128, 768),
    "discussion":      ChunkParams(512,  64,  768),
    "conclusion":      ChunkParams(512,  0,   1024),
    "appendix":        ChunkParams(512,  64,  768),
    "unknown":         ChunkParams(512,  64,  768),
}
_DEFAULT_PARAMS = _PARAMS["unknown"]


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


# ---------------------------------------------------------------------------
# JSON entry point  (preferred)
# ---------------------------------------------------------------------------

# Block types whose text content goes into the running section body
_BODY_BLOCKS = {
    'Text', 'ListItem', 'Caption', 'Footnote', 'Code',
    'Handwriting', 'TextInlineMath',
}
# Block types that are structural containers (recurse into children)
_CONTAINER_BLOCKS = {
    'Page', 'ListGroup', 'FigureGroup', 'PictureGroup', 'TableGroup',
}
# Block types that are completely ignored
_IGNORE_BLOCKS = {
    'PageHeader', 'PageFooter', 'Figure', 'Picture', 'Form',
    'ComplexRegion', 'TableOfContents', 'Reference',
}


_HTML_ENTITIES = {
    '&amp;': '&', '&lt;': '<', '&gt;': '>',
    '&quot;': '"', '&#39;': "'", '&nbsp;': ' ',
}
_HTML_ENTITY_RE = re.compile('|'.join(re.escape(k) for k in _HTML_ENTITIES))


def _strip_html(html: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', html or '')
    text = _HTML_ENTITY_RE.sub(lambda m: _HTML_ENTITIES[m.group(0)], text)
    return re.sub(r'\s+', ' ', text).strip()


def _table_to_text(html: str) -> str:
    """Convert an HTML table to a pipe-delimited plain-text representation."""
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html or '', re.DOTALL | re.IGNORECASE)
    if not rows:
        return _strip_html(html)
    lines = []
    for row_html in rows:
        cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_html,
                           re.DOTALL | re.IGNORECASE)
        cells = [_strip_html(c) for c in cells]
        if any(cells):
            lines.append(' | '.join(cells))
    return '\n'.join(lines) if lines else _strip_html(html)


def _heading_level_from_html(html: str) -> int:
    """Extract numeric heading level (1–4) from a SectionHeader HTML string."""
    m = re.search(r'<h(\d)', html or '')
    return int(m.group(1)) if m else 2


# Known canonical section words that should open a new section even when
# Marker classifies the block as Text (common in older or simply-formatted PDFs).
_KNOWN_SECTION_WORDS: frozenset[str] = frozenset(
    prefix
    for _, prefixes in _SECTION_PATTERNS
    for prefix in prefixes
)


def _is_implicit_section_header(html: str) -> bool:
    """
    Return True if a Text block looks like an undetected section heading.

    Criteria:
      - Short (≤ 6 words)
      - Content matches a known section keyword exactly after normalisation
    """
    text = _strip_html(html)
    if not text or len(text.split()) > 6:
        return False
    normalised = re.sub(r'^[\d.]+\s*', '', text.lower().strip())
    normalised = re.sub(r'[^\w\s]', '', normalised).strip()
    # Must match one of the known prefixes exactly
    return any(normalised == kw or normalised.startswith(kw + ' ')
               for kw in _KNOWN_SECTION_WORDS)


def parse_sections_from_json(json_data: dict) -> list[Section]:
    """
    Build a list of Sections by walking Marker's JSON block tree.

    Algorithm:
      - Walk all blocks across all pages in document order.
      - SectionHeader blocks open a new section.
      - Text/ListItem/Table/Equation/… blocks accumulate into the current
        section's content.
      - PageHeader, PageFooter, Figure, Picture and similar noise are skipped.
      - Only top-level sections (heading level 1–2) open new sections;
        deeper headings (h3–h4) are inlined into their parent section's text.
        This avoids fragmenting papers that use subsections heavily.
    """
    # ------------------------------------------------------------------
    # Step 1: flatten all blocks across pages in document order
    # ------------------------------------------------------------------
    flat: list[dict] = []

    def _collect(node: dict) -> None:
        bt = node.get('block_type', '')
        if bt in _IGNORE_BLOCKS:
            return
        if bt in _CONTAINER_BLOCKS:
            for child in (node.get('children') or []):
                _collect(child)
        else:
            flat.append(node)
            # Recurse for groups that have both inline html and children
            for child in (node.get('children') or []):
                _collect(child)

    for page in json_data.get('children', []):
        _collect(page)

    # ------------------------------------------------------------------
    # Step 2: group into sections
    # ------------------------------------------------------------------
    sections: list[Section] = []
    current_type = "unknown"
    current_title = ""
    current_parts: list[str] = []
    section_index = 0
    seen_section_ids: set[str] = set()  # avoid duplicate block traversal

    def _flush() -> None:
        nonlocal section_index
        if not current_parts:
            return
        content = '\n\n'.join(p for p in current_parts if p.strip())
        if not content.strip():
            return
        sections.append(Section(
            section_type=current_type,
            section_title=current_title,
            section_index=section_index,
            content=content,
        ))
        section_index += 1

    for block in flat:
        bt = block.get('block_type', '')
        html = block.get('html') or ''
        bid = block.get('id', '')

        # Avoid processing the same block twice (container recursion artefact)
        if bid and bid in seen_section_ids:
            continue
        if bid:
            seen_section_ids.add(bid)

        if bt == 'SectionHeader':
            level = _heading_level_from_html(html)
            heading_text = _strip_html(html)

            if level <= 2:
                # Open a new top-level section
                _flush()
                current_parts = []
                current_title = heading_text
                current_type = _classify_heading(heading_text)
            else:
                # Sub-section: keep as a bold separator within current section
                if heading_text:
                    current_parts.append(f'**{heading_text}**')

        elif bt in _BODY_BLOCKS and _is_implicit_section_header(html):
            # Some papers render section headings as plain Text blocks
            # (common in older PDFs where heading formatting is lost).
            # Treat them as top-level section openers.
            heading_text = _strip_html(html)
            _flush()
            current_parts = []
            current_title = heading_text
            current_type = _classify_heading(heading_text)

        elif bt == 'Table':
            table_text = _table_to_text(html)
            if table_text.strip():
                current_parts.append(table_text)

        elif bt == 'Equation':
            eq = _strip_html(html)
            if eq:
                current_parts.append(eq)

        elif bt in _BODY_BLOCKS:
            text = _strip_html(html)
            if text:
                current_parts.append(text)

        # All other block types are ignored (Figure, Picture, …)

    _flush()  # last section

    # ------------------------------------------------------------------
    # Step 3: handle documents with no section headers at all
    # ------------------------------------------------------------------
    if not sections:
        all_text = '\n\n'.join(current_parts)
        if all_text.strip():
            sections.append(Section(
                section_type="unknown",
                section_title="",
                section_index=0,
                content=all_text.strip(),
            ))

    return sections


# ---------------------------------------------------------------------------
# Markdown entry point  (legacy fallback)
# ---------------------------------------------------------------------------

def parse_sections(markdown: str) -> list[Section]:
    """
    Split markdown into sections by heading detection.
    Used as fallback when JSON conversion is unavailable.
    """
    heading_re = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
    matches = list(heading_re.finditer(markdown))

    if not matches:
        return [Section(
            section_type="unknown",
            section_title="",
            section_index=0,
            content=markdown.strip(),
        )]

    sections: list[Section] = []

    preamble = markdown[:matches[0].start()].strip()
    if preamble:
        sections.append(Section(
            section_type="abstract",
            section_title="preamble",
            section_index=0,
            content=preamble,
        ))

    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()
        section_type = _classify_heading(heading_text)
        content_start = match.end()
        content_end = (matches[i + 1].start()
                       if i + 1 < len(matches) else len(markdown))
        content = markdown[content_start:content_end].strip()
        if not content:
            continue
        sections.append(Section(
            section_type=section_type,
            section_title=heading_text,
            section_index=len(sections),
            content=content,
        ))

    return sections


# ---------------------------------------------------------------------------
# Chunking  (shared by both entry points)
# ---------------------------------------------------------------------------

def chunk_document(
    sections: list[Section],
    paper_title: str,
    paper_year: int | None,
) -> list[Chunk]:
    """Convert sections into retrieval-ready chunks with context headers."""
    chunks: list[Chunk] = []
    global_chunk_index = 0
    char_offset = 0

    year_str = str(paper_year) if paper_year else "n.d."
    title_str = paper_title or "Unknown Title"

    for section in sections:
        if section.section_type in _SKIP_SECTIONS:
            continue

        params = _PARAMS.get(section.section_type, _DEFAULT_PARAMS)
        section_chunks = _chunk_section(section.content, params)

        for raw_content in section_chunks:
            header = f"[{section.section_type}] {title_str} ({year_str})\n\n"
            full_content = header + raw_content
            tokens = _count_tokens(full_content)
            char_end = char_offset + len(raw_content)

            chunks.append(Chunk(
                section_type=section.section_type,
                section_title=section.section_title,
                section_index=section.section_index,
                chunk_index=global_chunk_index,
                content=full_content,
                raw_content=raw_content,
                content_tokens=tokens,
                char_start=char_offset,
                char_end=char_end,
            ))
            global_chunk_index += 1
            char_offset = char_end

    return chunks


def _chunk_section(content: str, params: ChunkParams) -> list[str]:
    """Chunk a single section's text into overlapping windows."""
    total_tokens = _count_tokens(content)
    if total_tokens <= params.keep_whole_if_under:
        return [content.strip()]

    paragraphs = [p.strip() for p in re.split(r'\n\n+', content) if p.strip()]
    if not paragraphs:
        return [content.strip()]

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para)
        if para_tokens > params.target_tokens:
            if buffer:
                chunks.append('\n\n'.join(buffer))
                buffer, buffer_tokens = _apply_overlap(buffer, params.overlap_tokens)
            sentence_chunks = _chunk_by_sentences(para, params)
            chunks.extend(sentence_chunks[:-1])
            if sentence_chunks:
                last = sentence_chunks[-1]
                buffer.append(last)
                buffer_tokens += _count_tokens(last)
            continue

        if buffer_tokens + para_tokens > params.target_tokens and buffer:
            chunks.append('\n\n'.join(buffer))
            buffer, buffer_tokens = _apply_overlap(buffer, params.overlap_tokens)

        buffer.append(para)
        buffer_tokens += para_tokens

    if buffer:
        chunks.append('\n\n'.join(buffer))

    return [c for c in chunks if c.strip()]


def _apply_overlap(
    buffer: list[str], overlap_tokens: int
) -> tuple[list[str], int]:
    if overlap_tokens == 0:
        return [], 0
    overlap_buf: list[str] = []
    overlap_count = 0
    for para in reversed(buffer):
        t = _count_tokens(para)
        if overlap_count + t > overlap_tokens:
            break
        overlap_buf.insert(0, para)
        overlap_count += t
    return overlap_buf, overlap_count


def _chunk_by_sentences(text: str, params: ChunkParams) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0
    for sent in sentences:
        t = _count_tokens(sent)
        if buffer_tokens + t > params.target_tokens and buffer:
            chunks.append(' '.join(buffer))
            buffer, buffer_tokens = _apply_overlap(buffer, params.overlap_tokens)
        buffer.append(sent)
        buffer_tokens += t
    if buffer:
        chunks.append(' '.join(buffer))
    return chunks
