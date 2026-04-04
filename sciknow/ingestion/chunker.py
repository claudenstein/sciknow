"""
Section-aware chunker for scientific paper markdown.

Detects sections by heading patterns, classifies them into canonical types,
then chunks each section using type-specific rules.
"""
import re
from dataclasses import dataclass

import tiktoken

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Section type classification
# ---------------------------------------------------------------------------

# Each entry: (canonical_type, list of prefixes to match against lowercased heading)
_SECTION_PATTERNS: list[tuple[str, list[str]]] = [
    ("abstract",       ["abstract"]),
    ("introduction",   ["introduction", "background", "motivation", "overview"]),
    ("related_work",   ["related work", "prior work", "literature review", "previous work", "state of the art"]),
    ("methods",        ["method", "material", "experiment", "data and method", "approach",
                        "model", "dataset", "proposed", "framework", "architecture",
                        "observation", "instrument", "data collection", "simulation"]),
    ("results",        ["result", "finding", "performance", "evaluation", "experiment",
                        "measurement", "observation"]),
    ("discussion",     ["discussion", "analysis", "interpretation", "implication"]),
    ("conclusion",     ["conclusion", "summary", "outlook", "future work", "closing",
                        "concluding"]),
    ("acknowledgments",["acknowledgment", "acknowledgement", "funding", "support"]),
    ("references",     ["reference", "bibliography"]),
    ("appendix",       ["appendix", "supplement"]),
]

# Sections not embedded into the vector store
_SKIP_SECTIONS = {"references", "acknowledgments"}


def _classify_heading(heading: str) -> str:
    normalized = re.sub(r'^[\d.]+\s*', '', heading.lower().strip())
    normalized = re.sub(r'[^\w\s]', '', normalized).strip()
    for section_type, prefixes in _SECTION_PATTERNS:
        for prefix in prefixes:
            if normalized.startswith(prefix):
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
    chunk_index: int       # global index within the document
    content: str           # includes context header
    raw_content: str       # content without header
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
    keep_whole_if_under: int  # keep section as single chunk if <= this many tokens


_PARAMS: dict[str, ChunkParams] = {
    "abstract":        ChunkParams(target_tokens=512,  overlap_tokens=0,   keep_whole_if_under=512),
    "introduction":    ChunkParams(target_tokens=512,  overlap_tokens=64,  keep_whole_if_under=768),
    "related_work":    ChunkParams(target_tokens=512,  overlap_tokens=64,  keep_whole_if_under=768),
    "methods":         ChunkParams(target_tokens=512,  overlap_tokens=128, keep_whole_if_under=768),
    "results":         ChunkParams(target_tokens=512,  overlap_tokens=128, keep_whole_if_under=768),
    "discussion":      ChunkParams(target_tokens=512,  overlap_tokens=64,  keep_whole_if_under=768),
    "conclusion":      ChunkParams(target_tokens=512,  overlap_tokens=0,   keep_whole_if_under=1024),
    "appendix":        ChunkParams(target_tokens=512,  overlap_tokens=64,  keep_whole_if_under=768),
    "unknown":         ChunkParams(target_tokens=512,  overlap_tokens=64,  keep_whole_if_under=768),
}

_DEFAULT_PARAMS = _PARAMS["unknown"]


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_sections(markdown: str) -> list[Section]:
    """Split markdown into sections by heading detection."""
    # Match headings: # ## ### #### (up to 4 levels)
    heading_re = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)

    matches = list(heading_re.finditer(markdown))

    if not matches:
        # No headings found — treat entire document as one unknown section
        return [Section(
            section_type="unknown",
            section_title="",
            section_index=0,
            content=markdown.strip(),
        )]

    sections: list[Section] = []

    # Text before first heading (often title block / authors)
    preamble = markdown[:matches[0].start()].strip()
    if preamble:
        sections.append(Section(
            section_type="abstract",  # treat as abstract-like for chunking
            section_title="preamble",
            section_index=0,
            content=preamble,
        ))

    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()
        section_type = _classify_heading(heading_text)

        # Content runs from end of this heading line to start of next heading
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
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

    # Split on paragraph boundaries (double newline)
    paragraphs = [p.strip() for p in re.split(r'\n\n+', content) if p.strip()]

    if not paragraphs:
        return [content.strip()]

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        # If a single paragraph exceeds target, split it by sentences
        if para_tokens > params.target_tokens:
            if buffer:
                chunks.append('\n\n'.join(buffer))
                buffer, buffer_tokens = _apply_overlap(buffer, params.overlap_tokens)
            sentence_chunks = _chunk_by_sentences(para, params)
            chunks.extend(sentence_chunks[:-1])
            # Keep last sentence chunk in buffer
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


def _apply_overlap(buffer: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Slide the window back by keeping trailing paragraphs within overlap budget."""
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
    """Last-resort: split oversized paragraph by sentences."""
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
