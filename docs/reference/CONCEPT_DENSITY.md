# Concept-density length sizing

[&larr; Back to README](../README.md) · [Research brief (§24)](RESEARCH.md#24-concept-density-length-sizing-research-brief-2026-04-20) · [Phase log](../roadmap/PHASE_LOG.md)

This document ties together the concept-density work that shipped across Phases 54.6.146–162. It's the "how do I use this" reference; the *why* is in [`RESEARCH.md §24`](RESEARCH.md#24-concept-density-length-sizing-research-brief-2026-04-20) and the per-phase implementation detail is in [`PHASE_LOG.md`](../roadmap/PHASE_LOG.md).

## The idea

Traditional autowrite took a top-down word budget: *book target ÷ chapters = chapter target*; *chapter target ÷ sections = section target*. That's a page-count model, not a pedagogy model.

The concept-density model, grounded in Cowan (2001), Mayer (2021), and Wiggins & McTighe (2005), inverts the direction: **size the section around the concepts it introduces, then let chapter and book length emerge from the sum**. Each section targets 3–4 novel chunks × per-genre words-per-concept.

## Two knobs, one per book

Every sciknow book has a **project type** (`sciknow book types`) which fixes two research-grounded ranges:

| Type | Chapter default | Concepts/section | Words/concept | Section @ midpoint |
|---|---:|---:|---:|---:|
| `scientific_paper` | 4,000 | 2–4 | 200–400 | 600–1,200 |
| `review_article` | 5,000 | 3–5 | 250–450 | 1,050–1,750 |
| `popular_science` | 6,500 | 3–4 | 400–700 | 1,650–2,200 |
| `scientific_book` | 8,000 | 3–4 | 500–800 | 1,950–2,600 |
| `instructional_textbook` | 4,500 | 3–4 | 400–700 | 1,650–2,200 |
| `academic_monograph` | 15,000 | 4–5 | 600–1,000 | 3,200–4,000 |

Concept counts derived from Cowan 2001; words-per-concept derived from the literature survey in [RESEARCH.md §24](RESEARCH.md#24-concept-density-length-sizing-research-brief-2026-04-20). Run `sciknow book types` for the live table with descriptions.

Change a book's type post-creation via the Book Settings modal's Basics tab dropdown, or `sciknow book set-target "<title>"` (clears the book-level override so the type default takes over).

## How a section target gets picked

Four-level resolver fires at the moment autowrite decides how many words to target for a section:

1. **Caller override** — `autowrite --target-words N` or per-call API argument.
2. **Per-section override** — explicit `target_words` in the Chapter modal Sections tab.
3. **Concept-density** — if the section's plan has N bullets, `target = N × wpc_midpoint`. **This is where the bottom-up magic happens.** Logs a soft warning when N > 4 (Cowan cap).
4. **Chapter split** — chapter target ÷ number of sections (top-down fallback for sections without a plan).

After retrieval runs, the **retrieval-density widener** (RESEARCH.md §24 guideline 4) nudges the Level-3 wpc toward the high end when retrieval returned >25 chunks, or the low end when <10. Bounded by the project-type `words_per_concept_range` so the adjustment is never runaway.

After the widener, if the final target exceeds the **Delgado 2018 digital-section ceiling** (3,000 words for most types, 5,000 for `academic_monograph`), a non-blocking `section_length_warning` fires in the autowrite stream. Autowrite still produces prose at the requested target — the warning is education, not enforcement.

## Workflows

### New book, use it right away

```bash
uv run sciknow book create "My Topic" --type scientific_book
# → 8,000 words/chapter by default (no custom_metadata override)
```

Don't pass `--target-chapter-words` unless you actively want to override the project-type default. Leaving it unset lets you absorb future research-grounded default updates automatically.

### Auto-plan every empty section

The concept-density path only fires when a section has a plan. sciknow can generate the plans for you:

**Per chapter (GUI):** Chapter modal → Sections tab → **🧠 Auto-plan sections** button.

**Whole book (GUI):** Book Settings → Basics tab → **🧠 Auto-plan entire book** button. Streams progress into an inline log (~5–10s per empty section; 48-section book ≈ 4–8 min).

**CLI:**
```bash
uv run sciknow book plan-sections "My Topic"                 # whole book
uv run sciknow book plan-sections "My Topic" --chapter 3     # one chapter
uv run sciknow book plan-sections "My Topic" --force         # overwrite
```

The LLM uses `LLM_FAST_MODEL` (structured-output task; flagship writer would waste VRAM and latency). Output is 3–4 bullets per section, framed by the chapter scope and section title.

### Adjust plans manually

Chapter modal → Sections tab → each textarea shows a **live concept-count readout** below it:

> **3 concepts** × 650 wpc = **~1,950 words**

Updates as you type. If you exceed 4 bullets, an inline warning cites Cowan 2001.

### See the whole-book projection

**GUI:** Book Settings → Basics tab → **📄 Projected length report** → refresh. Expandable per-chapter view with every section's target + resolver level.

**CLI:**
```bash
uv run sciknow book length-report "My Topic"
uv run sciknow book length-report "My Topic" --json | jq .total_words
```

Shows chapter and book totals plus the level histogram ("42 concept-density, 6 chapter-split"). Pre-widener values; the retrieval-density adjustment fires only at autowrite time.

### Validate the shipped numbers against your corpus

Two benchmarks validate the shipped project-type numbers against your actual data:

**Section-length distribution (fast, always safe):**
```bash
uv run sciknow bench --layer fast
# Look for the iqr_<section_type> rows. Each carries an
# alignment tag against RESEARCH.md §24's PubMed reference.
```

Also surfaced in Book Settings as the **📊 Corpus section-length distribution** panel — no CLI needed.

**Brown 2008 idea-density regression (slow, optional spaCy):**
```bash
uv add spacy
python -m spacy download en_core_web_sm
uv run sciknow bench-idea-density -n 500
```

Fits `word_count ~ n_ideas` per section type and reports the **empirical** wpc — compare to the shipped project-type midpoints. If your corpus's slope_wpc diverges materially from the shipped default, that's grounds for either a corpus-specific override (`book set-target --words N`) or a project-type swap.

**Autowrite bottom-up vs top-down A/B:**
```bash
uv run sciknow bench-autowrite-ab <chapter-id> --max-iter 3
```

Takes each planned section in a chapter, runs autowrite twice (plan present vs plan cleared), and reports paired scorer deltas. Cost: ~10–30 min per chapter. Run on a few chapters before drawing conclusions — scorer variance is ~±0.05.

### Pre-export verify (visuals)

If you're using `autowrite --include-visuals` and have `[Fig. N]` markers in your drafts, run the Level-3 VLM pass before exporting:

**GUI:** Verify dropdown → **💾 Finalize Draft (L3 VLM verify)**.

**CLI:**
```bash
uv run sciknow book finalize-draft <draft-id>
uv run sciknow book finalize-draft <draft-id> --flag-threshold 6
```

Every `[Fig. N]` marker gets scored 0–10 on claim-depiction match by the VLM. Exit code 0 on clean, 1 on any flagged (so CI can gate export).

## When to NOT use concept-density

- **Single-paper IMRaD write-ups** (`scientific_paper` type). Sections are short and dense; plans are usually overkill. Top-down chapter-split works fine.
- **Resume an existing book with a frozen target.** If a pre-54.6.146 book has `custom_metadata.target_chapter_words` set, Level-2 shadows Level-3 until you run `sciknow book set-target "<title>" --unset`.
- **Sections without a clear concept structure** (e.g. an acknowledgements page). The resolver falls back to chapter-split automatically.

## See also

- `docs/research/RESEARCH.md §24` — full research brief with Cowan / Mayer / Wiggins & McTighe / Brown 2008 citations and grade table for each guideline.
- `docs/benchmarks/BENCHMARKS.md` — corpus section-length validation data.
- `docs/roadmap/PHASE_LOG.md` — per-phase implementation detail (search for `54.6.146` through `54.6.162`).
- `sciknow/core/project_type.py` — the source of truth for the per-type defaults and ranges.
- `sciknow/core/book_ops.py` — `_get_section_concept_density_target`, `_adjust_target_for_retrieval_density`, etc.
