# Formatting / Professional PDF export

Phase 55 (`v2-llamacpp` branch) — `sciknow.formatting` is a self-contained
pipeline that compiles books and papers to journal-grade PDFs through
real LaTeX templates. It runs alongside the existing pandoc/weasyprint
exports without replacing them.

## Why a second export path?

The pre-Phase-55 path was Markdown → pandoc → LaTeX → pdflatex (or
HTML → weasyprint). It worked, but:

- Pandoc owns the entire LaTeX preamble; you can't pick a class or a font.
- Tables, equations, figures degrade to pandoc's lowest-common defaults.
- No bibliography styling beyond what citeproc+CSL produces.
- No support for academic book classes (kaobook, classicthesis, memoir).

Phase 55 inverts the responsibility: sciknow owns the IR + templates, and
shells out to `latexmk` only for the final compile. Each template can
fully customise its preamble, packages, fonts, bibliography style,
chapter style, etc.

## Pipeline

```
PostgreSQL drafts (markdown + sources JSONB)
    │
    ▼
[1] BookBibliography.from_book           ── global [N] dedup across the book
    │
    ▼
[2] bibtex.build_bibliography            ── stable cite keys (LastnameYear_hash)
    │
    ▼
[3] markdown_to_blocks                   ── markdown-it-py + dollarmath plugin
    │                                       inline [N] → \cite{key}
    │                                       fenced code → CodeBlock
    │                                       $$…$$ → EquationBlock
    │                                       md tables → TableBlock
    │                                       html tables (MinerU) → pandoc shell-out
    │
    ▼
[4] IR Document                           ── chapters → sections → blocks
    │
    ▼
[5] latex_renderer.render_document        ── Jinja2 template per project type
    │                                       (LaTeX-safe delimiters: ((* *)) and ((( ))))
    │
    ▼
[6] compile.compile_tex                   ── latexmk -lualatex (or pdflatex)
    │                                       in tempdir, with figures + refs.bib
    │
    ▼
PDF bytes + log_text + tex_source
```

GPU-free by design. lualatex/pdflatex are pure CPU; pandoc fragment
shellouts (HTML tables) are pure CPU. Safe to run alongside ongoing
inference jobs on the GPU.

## Module layout

```
sciknow/formatting/
├── __init__.py             # public API: build_book_pdf, build_book_tex_bundle
├── ir.py                   # dataclass IR (Document, Chapter, Section, Block, …)
├── markdown_to_ir.py       # markdown-it-py + citation rewriter
├── bibtex.py               # APA → BibTeX (stable cite keys)
├── latex_renderer.py       # IR + Jinja2 → main.tex string
├── compile.py              # latexmk wrapper, capture stdout/log/.log
├── build.py                # end-to-end orchestrator
├── options.py              # ExportOptions, template registry, font/bib registries
└── templates/
    ├── _shared/
    │   └── preamble_common.tex
    ├── elsarticle/         # natbib + bibtex
    ├── elsarticle-review/
    ├── ieeetran/           # natbib + bibtex
    ├── revtex/             # natbib + bibtex
    ├── article/
    ├── kaobook/            # vendored class — biblatex + biber
    │   ├── kaobook.cls
    │   ├── kao*.sty
    │   └── LICENSE         # LPPL
    ├── memoir/
    ├── memoir-narrative/   # popular-science: veelo + lettrine + accent colour
    ├── memoir-textbook/    # instructional: example/exercise/checkpoint boxes
    ├── scrbook/
    ├── classicthesis/      # André Miede style on top of scrbook
    └── tufte-book/         # natbib (tufte-book hard-wires it)
```

## Template registry (project_type → templates)

| Project type             | Default            | Alternates                              |
|--------------------------|--------------------|-----------------------------------------|
| `scientific_paper`       | `elsarticle`       | `ieeetran`, `revtex`, `article`         |
| `scientific_book`        | `kaobook` (vendored) | `memoir`, `scrbook`                   |
| `popular_science`        | `memoir-narrative` | `tufte-book`, `kaobook`                 |
| `instructional_textbook` | `kaobook`          | `memoir-textbook`                       |
| `academic_monograph`     | `classicthesis`    | `kaobook`, `memoir`                     |
| `review_article`         | `elsarticle-review`| `article`                               |

Cross-type selection is allowed — pick `kaobook` for a paper or
`elsarticle` for a book if you want; the template's `is_flat` branch
handles the layout switch.

## Citation handling

Inline `[N]` markers in draft markdown are recognised by a strict regex
that won't false-match `[link](url)`. The pipeline:

1. `BookBibliography.from_book` resolves local `[N]` per-draft to global
   numbers, dedup'd across all chapters.
2. `bibtex.build_bibliography` parses each global APA line, extracts the
   DOI, looks up `paper_metadata`, and emits a stable `@article` entry
   with key `<LastnameYear>_<sha1[:4]>`.
3. `markdown_to_blocks` uses the global-N → citekey map to rewrite
   `[3]` → `\cite{smith2022_a3f1}`.

When zero citations are detected, `bibliography` is suppressed at the
document level — natbib/biblatex would otherwise produce broken `.bbl`
files (IEEEtran's bst is particularly strict about empty bibliographies).

## Bib backend per template

| Template             | `bib_system` | `bib_backend` | Notes                        |
|----------------------|--------------|---------------|------------------------------|
| elsarticle           | natbib       | bibtex        | elsarticle hardwires natbib  |
| elsarticle-review    | natbib       | bibtex        |                              |
| ieeetran             | natbib       | bibtex        | uses `IEEEtran.bst`          |
| revtex               | natbib       | bibtex        | uses `apsrev4-2.bst`         |
| tufte-book           | natbib       | bibtex        | tufte-book includes bibentry |
| (everything else)    | biblatex     | biber         | modern, rich features        |

`compile_tex` passes `-bibtex` to latexmk for natbib templates so the
right backend runs.

## Engines

`lualatex` is the universal default. We picked it over pdflatex because:

- Full Unicode (climate papers contain Greek letters, accents, CJK names).
- System fonts via fontspec (no .pfb juggling).
- Robust against memoir's deep package interactions.
- Same speed as pdflatex on books > 100 pages once cached.

Templates can override via `TemplateSpec.engine`. Currently every
template uses lualatex.

## Code blocks

`listings` package, no `--shell-escape`. Decision: reliability over
beauty. `minted` looks better but requires both `--shell-escape` (lets
LaTeX execute arbitrary shell during compile) and the `pygments` Python
package — two extra failure modes for marginal aesthetic gain. The
shared preamble defines a `sciknowcode` colour theme that's clean and
readable.

## CLI

```bash
sciknow book export "Global Cooling" --format pdf-pro --template kaobook
sciknow book export "Global Cooling" --format pdf-pro \
    --template classicthesis \
    --font ebgaramond --font-size 11 \
    --bib-style authoryear

# .tex source bundle (.zip) for offline editing or co-author handoff
sciknow book export "Global Cooling" --format tex-bundle -o gc-source.zip
```

On compile failure, two artifacts are written to the working directory:

- `<book>-latex.log` — full latexmk + .log capture
- `<book>-failed.tex` — the rendered `main.tex` for inspection

## Web GUI

The Export modal (Book ▾ → Export, or `/export` deeplink) has two
panels:

1. **Quick export** (Phase 30) — markdown / HTML / weasyprint PDF /
   text, per draft / chapter / book. One-click downloads.
2. **Professional PDF (LaTeX)** (Phase 55) — full template + font +
   bib-style + ToC + cover + author/affiliation/abstract overrides.
   Compiles asynchronously in a thread, polls `/api/export/pro/job/{id}`
   for status, auto-downloads on success. On failure the compile log
   is shown inline.

Backend endpoints:

- `POST /api/export/pro/build` → `{job_id, status: "pending"}`
- `GET  /api/export/pro/job/{id}` → status JSON
- `GET  /api/export/pro/job/{id}/download` → PDF or .zip
- `GET  /api/export/pro/templates` → registry snapshot for the picker

Jobs live in-process with a 30-minute TTL — no Postgres state, no
Redis. They're meant to back a UI session, not durable workflow.

## Adding a new template

1. Create `sciknow/formatting/templates/<slug>/main.tex.j2`.
2. Optionally `preamble.tex` (for class-specific tweaks beyond
   `_shared/preamble_common.tex`).
3. If the class isn't in `texlive-full`, vendor `*.cls`/`*.sty` next to
   `main.tex.j2` along with the upstream LICENSE.
4. Register the slug in `options.TEMPLATES[<project_type>]` with the
   right `bib_system` / `bib_backend` / `engine`.
5. Write a smoke compile in `testing/protocol.py` (see
   `l1_formatting_template_registry_complete`) to gate regressions.

## Smoke tests

`l1_formatting_renders_minimal_doc`, `l1_formatting_markdown_citations`,
`l1_formatting_handles_rich_blocks`, and
`l1_formatting_template_registry_complete` are wired into `sciknow test
--layer L1`. They exercise the IR + renderer + markdown parsing without
a Postgres connection or a latexmk run, so they stay under a few hundred
ms total.

A heavier smoke check (build a real book with each template) is
intentionally NOT in L1 because each compile is 5–15 seconds. Run it
manually via `sciknow book export "<title>" --format pdf-pro --template
<slug>` against any populated book whenever you change a template.
