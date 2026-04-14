"""
Section-aware chunker for scientific paper documents.

Three entry points, one per PDF backend:
  parse_sections_from_mineru(content_list) — primary; reads MinerU 2.5
                                             content_list.json (flat list of
                                             typed blocks with text_level for
                                             headings).
  parse_sections_from_json(json_data)      — fallback; reads Marker's nested
                                             block tree (SectionHeader/Text/
                                             Table/Equation/…).
  parse_sections(markdown)                 — last resort; regex over plain
                                             markdown headings.

All three return list[Section], consumed by chunk_document().
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# Phase 52 — stamp every chunk with this integer. Bump when
# `_SECTION_PATTERNS`, `_SKIP_SECTIONS`, `_PARAMS`, or the chunking
# functions change in a way that invalidates stored output. Pipeline
# and `db repair --rebuild-paper` read this to detect staleness.
#
# Version history:
#   0 = pre-Phase-52 (server_default for existing rows post-migration 0022)
#   1 = Phase 52 initial stamp. No semantic change vs 0; the bump
#       exercises the staleness-detection path end-to-end.
CHUNKER_VERSION: int = 1


def needs_rechunk(stored_version: int | None) -> bool:
    """True if a chunk stored with `stored_version` is older than the
    code's current `CHUNKER_VERSION`. Treats None / 0 as stale."""
    return (stored_version or 0) < CHUNKER_VERSION


# ---------------------------------------------------------------------------
# Section type classification  (shared by both entry points)
# ---------------------------------------------------------------------------

# (canonical_type, list of lowercase prefixes to match heading text against)
#
# Phase 44.1 — broadened after bench findings showed very low hit rates on
# `related_work` (0.2%), `results` (24%), and `abstract` (37%). Two changes:
# (1) expand each list with domain-common synonyms the original regex missed
#     (e.g. "findings" for results, "review" for related_work, "extended
#     abstract" for abstract), and (2) switch `_classify_heading` from
#     first-match-wins to longest-prefix-wins so overlapping keywords
#     ("experiment" in methods, "experimental results" in results) resolve
#     to the more specific category.
#
# Overlapping keywords that previously caused misclassification:
#   - "experiment*" in methods preempted "experimental result/finding"
#   - "observation" in methods preempted "observed results"
# Both fixed by longest-prefix-wins.

_SECTION_PATTERNS: list[tuple[str, list[str]]] = [
    ("abstract",       ["abstract", "extended abstract", "synopsis",
                        "plain language summary", "plain-language summary",
                        "highlights"]),
    ("introduction",   ["introduction", "background", "motivation", "overview",
                        "preface", "preamble"]),
    ("related_work",   ["related work", "prior work", "literature review",
                        "previous work", "previous research", "previous studies",
                        "state of the art", "state-of-the-art",
                        "review of", "historical perspective",
                        "research context", "theoretical background"]),
    ("methods",        ["method", "material", "data and method",
                        "approach", "model description", "model",
                        "framework", "architecture",
                        "instrument", "data collection",
                        "experimental setup", "experimental design",
                        "experimental method", "experimental procedure",
                        "experiment", "simulation setup", "numerical setup",
                        "simulation", "observation", "dataset",
                        "study area", "study site", "study region",
                        "sample preparation", "sampling",
                        "proposed"]),
    ("results",        ["result", "finding", "main finding", "key finding",
                        "performance", "evaluation",
                        "experimental result", "experimental finding",
                        "observed result", "simulation result",
                        "main outcome", "outcome",
                        "empirical result", "empirical finding",
                        "measurement result", "measurement",
                        "observations and measurements",
                        # Combined headings stay in `results` rather than
                        # `discussion`: users filtering --section results
                        # expect findings; a "Results and Discussion" section
                        # contains findings-as-primary-content with analysis
                        # threaded in. Classifying it as discussion hid 118
                        # papers' findings under a less-specific bucket
                        # (post-bench Phase 44.1 finding). Prefix is longer
                        # than plain "discussion" so longest-prefix wins.
                        "results and discussion"]),
    ("discussion",     ["discussion", "analysis", "interpretation", "implication",
                        "general discussion"]),
    ("conclusion",     ["conclusion", "concluding remark", "concluding",
                        "summary", "summary and conclusion", "closing",
                        "outlook", "future work", "future research",
                        "future direction", "perspectives and outlook"]),
    ("acknowledgments",["acknowledgment", "acknowledgement", "funding",
                        "support", "author contribution",
                        "conflict of interest", "competing interest",
                        "ethics statement"]),
    ("references",     ["reference", "bibliography", "works cited",
                        "cited literature"]),
    ("appendix",       ["appendix", "supplement", "supplementary",
                        "supporting information", "code availability",
                        "data availability"]),
]

_SKIP_SECTIONS = {"references", "acknowledgments"}


# Longest-prefix-wins classifier, built once at import time.
# Each entry is (prefix, section_type), sorted by prefix length DESC so the
# first startswith match is also the longest. This gets us both "experimental
# result" → results and "experimental setup" → methods without the old
# first-match-wins ordering footgun.
_CLASSIFY_TABLE: list[tuple[str, str]] = sorted(
    ((prefix, section_type)
     for section_type, prefixes in _SECTION_PATTERNS
     for prefix in prefixes),
    key=lambda pair: len(pair[0]),
    reverse=True,
)


def _classify_heading(heading: str) -> str:
    """Map a heading string to a canonical section type (longest-prefix wins)."""
    normalised = re.sub(r'^[\d.]+\s*', '', heading.lower().strip())
    normalised = re.sub(r'[^\w\s]', '', normalised).strip()
    for prefix, section_type in _CLASSIFY_TABLE:
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


def _mineru_list_items_to_text(list_items: list) -> str:
    """MinerU list_items is either list[str] or list[dict{text: str}]."""
    out: list[str] = []
    for li in list_items or []:
        if isinstance(li, str):
            s = li.strip()
            if s:
                out.append(f"- {s}")
        elif isinstance(li, dict):
            txt = (li.get("text") or "").strip()
            if txt:
                out.append(f"- {txt}")
    return "\n".join(out)


def parse_sections_from_mineru(content_list: list[dict]) -> list[Section]:
    """
    Build a list of Sections by walking MinerU's content_list.json.

    Input format (MinerU 2.5 pipeline backend):
      Flat list of typed items:
        {type: "text",     text: "...", text_level: 0|1|2|3, bbox, page_idx}
        {type: "table",    table_body: "<html>...</html>", table_caption: [...], ...}
        {type: "equation", text: "<latex>", text_format: "latex"}
        {type: "code",     code_body: "...", code_caption: [...], sub_type: "code"|"algorithm"}
        {type: "list",     list_items: [...] }
        {type: "image",    img_path, image_caption, image_footnote}
        {type: "chart"|"seal"}             — treated like image (skipped)
        {type: "header"|"footer"|"page_number"|"page_footnote"|"aside_text"} — skipped

    Algorithm:
      - text with text_level == 0 (or missing) → body content for current section
      - text with text_level == 1 or 2         → open a new top-level section
      - text with text_level >= 3              → inline bold subheading within current section
      - table/equation/code/list               → body content
      - image/chart/seal/auxiliary             → skipped
      - Implicit-heading heuristic: short Text blocks whose body matches a known
        section word are promoted to section headers (same as the Marker path).
    """
    sections: list[Section] = []
    current_type = "unknown"
    current_title = ""
    current_parts: list[str] = []
    section_index = 0

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

    # Auxiliary / noise types that never contribute to section content
    NOISE_TYPES = {"image", "chart", "seal", "header", "footer",
                   "page_number", "page_footnote", "aside_text"}

    for item in content_list or []:
        itype = item.get("type", "")

        if itype in NOISE_TYPES:
            continue

        if itype == "text":
            text = (item.get("text") or "").strip()
            if not text:
                continue
            level = item.get("text_level") or 0

            if level in (1, 2):
                # Top-level section heading
                _flush()
                current_parts = []
                current_title = text
                current_type = _classify_heading(text)
                continue

            if level and level >= 3:
                # Subheading — inline bold marker, keep within current section
                current_parts.append(f"**{text}**")
                continue

            # level 0 (body text) — but check implicit section heuristic first.
            # Short body-level blocks that look exactly like a canonical section
            # word get promoted to section headings. This handles older PDFs
            # where MinerU's layout model didn't emit a text_level.
            if len(text.split()) <= 6:
                normalised = re.sub(r'^[\d.]+\s*', '', text.lower().strip())
                normalised = re.sub(r'[^\w\s]', '', normalised).strip()
                if any(normalised == kw or normalised.startswith(kw + ' ')
                       for kw in _KNOWN_SECTION_WORDS):
                    _flush()
                    current_parts = []
                    current_title = text
                    current_type = _classify_heading(text)
                    continue

            current_parts.append(text)

        elif itype == "table":
            body = item.get("table_body") or ""
            if body:
                table_text = _table_to_text(body)
                if table_text.strip():
                    # Include caption if present for better retrieval signal
                    caption_parts = item.get("table_caption") or []
                    caption = " ".join(c for c in caption_parts if isinstance(c, str)).strip()
                    if caption:
                        current_parts.append(f"{caption}\n{table_text}")
                    else:
                        current_parts.append(table_text)

        elif itype == "equation":
            eq = (item.get("text") or "").strip()
            if eq:
                current_parts.append(eq)

        elif itype == "code":
            code = (item.get("code_body") or "").strip()
            if code:
                current_parts.append(code)

        elif itype == "list":
            list_text = _mineru_list_items_to_text(item.get("list_items") or [])
            if list_text:
                current_parts.append(list_text)

        # Unknown types are ignored rather than raising — MinerU may introduce
        # new types across versions, and silently dropping them is safer than
        # breaking ingestion on upgrade.

    _flush()  # last section

    # Handle documents with no section headers at all
    if not sections and current_parts:
        all_text = '\n\n'.join(current_parts)
        if all_text.strip():
            sections.append(Section(
                section_type="unknown",
                section_title="",
                section_index=0,
                content=all_text.strip(),
            ))

    return sections


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
