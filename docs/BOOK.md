# Book Writing System

[&larr; Back to README](../README.md)

---

A structured writing pipeline for long-form scientific books, reports, and review articles. Built on the RAG pipeline with iterative refinement, cross-chapter coherence, and autonomous convergence.

## Data Model

```
Book  (title, description, plan, status)
  ├── BookChapter  (number, title, description, topic_query, topic_cluster, sections)
  │     └── Draft  (section_type, content, sources, summary, version,
  │                  parent_draft_id, review_feedback, model_used, status, custom_metadata)
  └── BookGap  (gap_type, description, chapter_id, status, resolved_draft_id)
```

Key fields:
- `books.plan` — thesis statement + scope (200-500 words), injected into every write call
- `book_chapters.sections` — per-chapter JSONB list of section names (e.g. `["overview", "key_evidence", "current_understanding", "open_questions", "summary"]`)
- `drafts.summary` — auto-generated 100-200 word summary for cross-chapter context
- `drafts.parent_draft_id` — links revisions to their parent (version chain: v1 → v2 → v3)
- `drafts.review_feedback` — the reviewer agent's structured assessment
- `drafts.status` — workflow status (to_do, drafted, reviewed, revised, final)
- `book_gaps` — persistent gap tracking with type, priority, and resolution status

---

## Cross-Chapter Coherence

The biggest quality improvement over simple per-section generation. When writing Chapter N, the prompt automatically includes:
1. **Book plan** — the thesis statement and scope document (generated once with `book plan`)
2. **Prior chapter summaries** — auto-generated 100-200 word summaries of every draft from chapters 1 through N-1

This prevents contradictions, repeated explanations, and inconsistent terminology across chapters.

---

## The Write → Review → Revise Loop

1. **Write** (`book write`) — RAG-grounded draft with cross-chapter context
2. **Review** (`book review`) — LLM critic assesses groundedness, completeness, accuracy, coherence, redundancy. Saves structured feedback.
3. **Revise** (`book revise`) — applies review feedback (or a custom instruction) to produce version N+1. The original is preserved.
4. Repeat 2-3 until satisfied.

### Quality flags on `book write`

| Flag | What it does |
|---|---|
| `--plan` | Hierarchical tree plan (TreeWriter pattern): JSON paragraph skeleton with main point, sources, and transitions before drafting |
| `--verify` | Post-generation claim verification — checks each [N] citation against its source passage, reports a groundedness score |
| `--expand` | LLM query expansion before retrieval |

---

## Autowrite: Autonomous Convergence

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — propose → evaluate → keep/discard loops.

**How it works:**

1. Generates an initial draft (with book plan + cross-chapter summaries for coherence)
2. Scores the draft on 5 dimensions (0.0-1.0): groundedness, completeness, coherence, citation accuracy, overall
3. If overall ≥ target score → **converged**, stop
4. Identifies the **weakest dimension** and generates a targeted revision instruction
5. Revises the draft targeting that specific weakness
6. Re-scores the revision: if improved → **keep**, if regressed → **discard**
7. Repeats until converged or max iterations exhausted

**Convergence example:**
```
v1: groun=0.65  compl=0.60  coher=0.80  citat=0.70  overall=0.69
    Weakest: completeness → "Add discussion of proxy calibration methods"
v2: groun=0.72  compl=0.78  coher=0.82  citat=0.75  overall=0.77  KEEP
    Weakest: groundedness → "Cite primary sources for claims in paragraph 3"
v3: groun=0.85  compl=0.80  coher=0.85  citat=0.82  overall=0.83  KEEP
v4: groun=0.88  compl=0.83  coher=0.87  citat=0.85  overall=0.86  CONVERGED
```

**Modes:**
- Single section: `book autowrite "Book" 3 --section methods`
- All sections of one chapter: `book autowrite "Book" 3 --section all`
- Full book: `book autowrite "Book" --full`

**Flags:**
- `--max-iter N` — max iterations per section (default 3)
- `--target-score 0.85` — quality threshold to stop (default 0.85)
- `--auto-expand` — when the reviewer identifies missing evidence, checks corpus coverage and flags topics for expansion
- `--rebuild` — overwrite existing drafts (default: auto-resume, skipping sections with drafts)

**Estimated times (qwen3.5:27b on 3090):**

| Mode | Sections | Time |
|---|---|---|
| One section (3 iterations) | 1 | ~5-8 min |
| One chapter (all sections) | 5 | ~30-40 min |
| Full 10-chapter book | 50 | ~4-6 hours (unattended) |

---

## Web Reader (`book serve`)

`sciknow book serve "Global Cooling"` launches a local web application at `http://127.0.0.1:8765` with:

- **Sidebar navigation** — SPA-style chapter/section navigation without page reloads, with word counts and version numbers
- **Action toolbar** — Write, Review, Revise, Autowrite, Argue, and Gaps buttons directly in the browser. Every operation streams LLM output live via SSE (Server-Sent Events)
- **Live streaming** — tokens appear in the browser in real-time. Autowrite shows iteration scores, keep/discard decisions, and convergence progress
- **Book dashboard** — completion heatmap (chapters x sections, color-coded), stats cards, actionable gap list
- **Version history + diffs** — see all versions of a section (v1 → v2 → v3), word-level diff with red/green highlighting
- **Chapter management** — add/delete/reorder chapters from the sidebar
- **Enhanced editor** — split-pane markdown editor with live preview, toolbar for bold/italic/headings/citations, auto-saves every 5 seconds
- **Autowrite dashboard** — live SVG convergence chart, color-coded score bars, decision log, stop button
- **Citation popovers** — hover over `[N]` to see paper title, authors, year, journal
- **Claim verification** — green/yellow/red per-citation indicators with groundedness score
- **Argument map** — SVG diagram: central claim node, green (supports), red (contradicts), gray (neutral)
- **Corkboard view** — Scrivener-inspired card layout, color-coded by status
- **Chapter reader** — all sections concatenated as one continuous scroll
- **Snapshots** — save/restore named copies of drafts
- **Custom status labels** — To Do, Drafted, Reviewed, Revised, Final
- **Comments/annotations** — per-section, resolvable
- **Dark/light theme** — persists to localStorage
- **No external dependencies** — pure HTML/CSS/JS, no npm, no build step

### Web API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/write` | POST | Start a section draft |
| `/api/review/{draft_id}` | POST | Start a critic pass |
| `/api/revise/{draft_id}` | POST | Start a revision |
| `/api/autowrite` | POST | Start the convergence loop |
| `/api/argue` | POST | Map evidence for/against a claim |
| `/api/verify/{draft_id}` | POST | Run claim verification |
| `/api/gaps` | POST | Run gap analysis |
| `/api/stream/{job_id}` | GET | SSE endpoint for live streaming |
| `/api/jobs/{job_id}` | DELETE | Cancel a running job |
| `/api/section/{draft_id}` | GET | Section data as JSON |
| `/api/chapters` | GET/POST | Chapter list / add chapter |
| `/api/chapters/{id}` | PUT/DELETE | Update / delete chapter |
| `/api/chapters/reorder` | POST | Reorder chapters |
| `/api/dashboard` | GET | Completion heatmap, stats, gaps |
| `/api/versions/{draft_id}` | GET | Version chain |
| `/api/diff/{old_id}/{new_id}` | GET | Word-level diff as HTML |
| `/api/corkboard` | GET | Card data for corkboard view |
| `/api/chapter-reader/{id}` | GET | All sections concatenated |
| `/api/snapshot/{draft_id}` | POST | Save a named snapshot |
| `/api/snapshots/{draft_id}` | GET | List snapshots |
| `/api/draft/{id}/status` | PUT | Update draft status |
| `/api/draft/{id}/metadata` | PUT | Merge custom metadata |

---

## Export Formats

| Format | Command | Notes |
|---|---|---|
| Markdown | `book export "..." -o book.md` | Default. Inline [N] citations + bibliography |
| HTML | `book export "..." --format html -o book.html` | Self-contained reader with sidebar + theme |
| BibTeX | `book export "..." --format bibtex -o refs.bib` | From paper_metadata |
| LaTeX | `book export "..." --format latex -o book.tex` | Via Pandoc + `--citeproc` |
| DOCX | `book export "..." --format docx -o book.docx` | Via Pandoc |

Export deduplicates citations globally across all chapters — `[1]` in Ch.1 and `[3]` in Ch.5 pointing to the same paper become a unified `[N]` with a single bibliography entry.

---

## Writing Workflow: Step by Step

### 1. Cluster your papers
```bash
sciknow catalog cluster
sciknow catalog topics
```

### 2. Create the book and generate structure
```bash
sciknow book create "Global Cooling" \
    --description "Evidence for solar-driven climate variability"
sciknow book outline "Global Cooling"
sciknow book show "Global Cooling"
```

### 3. Generate the book plan
```bash
sciknow book plan "Global Cooling"
```
The thesis statement that anchors every chapter. Read carefully. Regenerate with `--edit` if needed.

### 4. Check gaps before writing
```bash
sciknow book gaps "Global Cooling"
```
If gaps are severe, expand the corpus: `sciknow db expand --limit 50 -q "your topic"`

### 5. Write chapter by chapter
Start from chapter 1 — each chapter gets summaries of all prior chapters:
```bash
sciknow book write "Global Cooling" 1 --section overview --plan --verify
sciknow book write "Global Cooling" 1 --section key_evidence
sciknow book write "Global Cooling" 2 --section overview --plan
```

### 6. Review and revise
```bash
sciknow book review <draft_id>
sciknow book revise <draft_id>
sciknow book revise <draft_id> -i "expand the discussion of Maunder Minimum evidence"
```

### 7. Argument mapping for contested claims
```bash
sciknow book argue "solar activity is the primary driver of 20th century warming" --save
```

### 8. Re-check gaps, iterate
```bash
sciknow book gaps "Global Cooling"
sciknow book show "Global Cooling"
```

### 9. Export
```bash
sciknow book export "Global Cooling" --format latex -o manuscript.tex
sciknow book export "Global Cooling" --format bibtex -o refs.bib
```

---

## Tips for Effective Book Writing

- **Write sequentially** (ch1 → ch2 → ch3). Each chapter's prompt includes summaries of all prior chapters.
- **Use `--plan` on first drafts.** Shows the paragraph skeleton before the full draft streams.
- **Use `--verify` on important sections.** Catches hallucinated citations early.
- **Argue before discussion sections.** Run `book argue` on key claims first, then write nuanced discussion.
- **Expand targetedly when gaps appear.** Run `db expand -q "your topic"` to grow corpus in weak areas.
- **Review every section before moving on.** The review → revise loop catches issues early.
- **Use autowrite for hands-off convergence.** `book autowrite "Book" --full` handles everything unattended.
- **Use the web reader for everything.** The browser drives the entire workflow with live streaming.

### Browser-First Workflow

```
sciknow book serve "Global Cooling"
→ opens http://localhost:8765

In the browser:
  1. Click "Write" on any empty section → tokens stream live
  2. Click "Review" → critic feedback streams into the panel
  3. Click "Revise" → revised version appears with live progress
  4. Click "Autowrite" → convergence loop with live scores
  5. Click "Argue" → evidence map for any claim
  6. Click "Gaps" → identifies missing topics
  7. Edit any section inline, add comments, resolve them
```
