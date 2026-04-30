# Wiki browsing UX — design research

**Review date**: 2026-04-14. Same style as `KG_RESEARCH.md` /
`EXPAND_RESEARCH.md` / `MEMPALACE_REVIEW.md` / `AUTOREASON_REVIEW.md`
— concrete, priority-stacked, explicit skip list.

## Current state (audited from the code)

From `sciknow/web/app.py`, `sciknow/core/wiki_ops.py`,
`sciknow/storage/models.py::WikiPage`, and the `wiki compile` CLI:

- **Entry point**: a single toolbar "Wiki Query" button opens a modal
  with two tabs (Query / Browse). No dedicated page. No deep-linkable
  URL for any individual wiki page.
- **Data**: `wiki_pages` table with
  `(slug, title, page_type, source_doc_ids, qdrant_point_id,
   needs_rewrite, word_count, …)`. Page types: `paper_summary`,
  `concept`, `synthesis`. Markdown on disk under
  `data/wiki/{papers,concepts,synthesis}/{slug}.md`. Separate
  `knowledge_graph` table for (subject, predicate, object,
  source_sentence) triples.
- **Rendering**: `_md_to_html()` at `app.py:438-459` handles headers,
  bold/italic, paragraphs, and `[N]` → `<span class="citation"
  data-ref>` spans. **Doesn't render `[[wiki-links]]`**, no headings
  get IDs, no TOC, no tables, no math, no images, no citation popover
  on the wiki view.
- **Navigation**: list ↔ detail only. No breadcrumbs, no hyperlinks
  between pages, no history, no "related pages", no backlinks.
- **Search**: a page-type filter and a separate RAG-style ask
  ("answer my question from the wiki"). No live full-text search
  over wiki pages, no fuzzy title autocomplete, no command palette.
- **KG integration**: none. The 3D orbit graph and the wiki live in
  separate modals, don't cross-link. Concept pages don't show their
  related triples.
- **Corpus integration**: one-way. Paper ingest produces a summary
  page; nothing jumps back from page to source PDF.
- **Known gaps from the code itself**: concept pages are stubs unless
  `wiki compile --rewrite-stale` is run; lint finds broken
  `[[links]]` but can't fix them in the UI; three CLI commands
  (`wiki lint`, `wiki synthesize`, `wiki consensus`) have no web
  surface.

## What good scientific-wiki browsing looks like in 2025

Across every mature knowledge-base UI, four affordances do almost
all the work and everything else is decoration:

1. **Backlinks as the primary "what connects to this" surface** —
   Obsidian / Roam / Logseq / Dendron / Foam all expose a persistent
   "Linked mentions" panel under the page. The 2025 academic-Obsidian
   consensus is that backlinks + a global quick-switcher replace
   folder hierarchy entirely.
2. **Ego-expansion graph navigation**, not global-graph browsing —
   Connected Papers, Inciteful, ResearchRabbit all start from one
   seed and grow by click because global-graph views of >200 nodes
   hairball. Sciknow already got this right in the KG modal; the
   lesson is to *not* promote the global graph to be the primary
   wiki nav.
3. **Command palette keyed on page titles (`⌘K` / `Ctrl-K`)** — the
   single highest-leverage power-user affordance across Notion,
   Linear, Slack, VS Code, and every modern KB since 2021. One
   keystroke → type → Enter. Users reach for this 50× per session.
4. **Structured entity "infobox"** — Wikipedia's right-rail or
   Scholia's auto-generated entity sections (timeline, co-authors,
   topics, citing works). This is what distinguishes a *wiki* from
   an aimless blob of prose.

Two non-obvious constraints sharpen the picture:

- **Graph views are a navigation trap.** The 2025 Obsidian-academia
  consensus is that graph view is "fun to explore" but users
  navigate by search + backlinks + recent in practice. Sciknow's
  rich KG modal is the right home for the graph; don't promote it
  to the primary reading surface.
- **Typography is load-bearing, not decoration.** Every serious
  reader tool — Readwise Reader, Pocket, Instapaper, Matter — lands
  on ~65–75 ch columns, serif body, 1.6 line-height. Prose quality
  *is* the read experience.

What explicitly **doesn't** port: Notion/Capacities "database-as-
page" (overkill for read-heavy compiled output), Roam-style block
references (assumes continuous editing; sciknow's wiki is LLM-
compiled), Dendron hierarchical filename schemas (solves a problem
flat-slug+tags don't have), Logseq daily notes (wrong product
shape), big infobox templates à la MediaWiki (overengineering for
our scale).

## Recommended priority stack

Top-first by experience-gain per hour of work. All items are
additive — none break existing surfaces.

### #1 — SPA route for the wiki (leave the modal) (~300 LOC, 4 h)

**Why this is #1.** The current UX funnels users through three taps
(toolbar → modal → list → click → inner pane) to read one page.
Every other item on this list is gated by "are users actually
reading the pages?" — and they won't when the page is a modal-in-
modal with cramped prose. This is **not a shallow cosmetic change**.
A stable DOM surface is prerequisite for the TOC, citation popovers,
backlinks, hash permalinks, and keyboard shortcuts to be anywhere to
attach. Turn `wiki-browse-pane` into a full-page SPA route at
`#wiki` and `#wiki/<slug>`, apply the serif 65–72 ch column, keep
the *query* tab as its own modal-launched feature.

**Scope**: extract the wiki HTML out of `wiki-modal` into a top-level
section matching the book-reader layout (`app.py:5925-5972` area);
register `#wiki` / `#wiki/<slug>` in the existing hash-nav dispatcher;
widen `.wiki-page-content` to `max-width: 72ch`, `line-height: 1.65`,
serif body. The right panel (sources / comments) remains useful as-
is — they follow the active wiki page.

**Trade-offs**: one route conflict to resolve (wiki active ⇄ book
chapter active); existing query tab migration takes care itself.

**L1 test**: grep `web_app_full_source()` for a `#wiki/` hash-route
handler and `max-width: 72ch` on `.wiki-page-content`.

### #2 — Render `[[wiki-slug]]` links as real hyperlinks (~100 LOC, 2 h)

**Why**. Pages on disk already use `[[slug]]` liberally; the renderer
silently drops them so users see literal text like "see
[[anthropogenic-forcing]]". One regex pass in `_md_to_html()`, an
existence check against the slug set (cached per render), two CSS
states — live link + greyed dead link like MediaWiki — unlocks *all*
traversal-based browsing. Everything downstream (backlinks, command
palette, keyboard nav) composes on top.

**Scope**: extend `_md_to_html()` with
`[[slug]]` / `[[slug|alt text]]` regex → `<a class="wiki-link"
href="#wiki/{slug}">text</a>` for existing slugs, `wiki-link-dead`
span for missing. Front-end hash-change handler opens the target.

**L1 test**: `_md_to_html("see [[foo-bar]] there")` with a mocked
slug existence check returns `<a class="wiki-link"` when the slug
exists, `wiki-link-dead` otherwise.

### #3 — Command palette (`⌘K` / `Ctrl-K`) with fuzzy page-title search (~200 LOC, 2 h)

**Why**. Single keystroke → type → Enter. The highest-ROI navigation
affordance in modern KBs. Zero backend complexity beyond a simple
title-index endpoint. Becomes load-bearing once the wiki has ≥ 50
pages and users stop remembering slugs.

**Scope**: new `wiki-palette` modal + JS module with fzf-style
scoring (~40 LOC, no dep); new `/api/wiki/titles` endpoint returning
`[{slug, title, page_type}]` for client-side fuzzy match. Bind
`⌘K`/`Ctrl-K` globally. Recent-history as the empty state (ring-
buffered in localStorage). Arrow-key nav + Enter opens.

**L1 test**: inspect handler source for a keydown listener on
`key === "k"` with `(metaKey || ctrlKey)`.

### #4 — Backlinks panel + "Related pages" on every wiki page (~250 LOC + migration, 4 h)

**Why**. This is what turns a list of generated pages into a *wiki*.
For a scientific corpus it's *more* valuable than in generic note-
taking — you want to see that `total-solar-irradiance` is referenced
by 18 paper summaries and jump into them. Related-pages-by-embedding
is the low-friction spatial companion, and we already have the
WIKI-collection vectors.

**Scope**:
- Migration adding a `wiki_backlinks (from_slug, to_slug,
  anchor_text)` table; populated at the end of `wiki compile` by
  scanning stored markdown for `[[slug]]` occurrences and by joining
  `knowledge_graph` rows where the object slug matches. Cheap O(N)
  per compile.
- `/api/wiki/page/{slug}/backlinks` + `/related` endpoints (related =
  Qdrant ANN top-5 in WIKI collection, `exclude=self`).
- Right-sidebar renders both as collapsible sections under the TOC.

**Trade-offs**: computing backlinks at query time would be O(n) in
page count — acceptable below 500 pages, unacceptable above. Ship
with materialised table.

**L1 test**: insert `[[concept-x]]` into one page via `_save_page`,
hit `/api/wiki/page/concept-x/backlinks`, assert 1 row.

### #5 — Auto-TOC + heading anchors + KaTeX math (~200 LOC, 3 h)

**Why**. Three polish items that compound. Auto-TOC makes long
synthesis pages navigable; heading anchors unlock deep-link sharing
(`#wiki/foo#results`); KaTeX renders the `$\Delta T$` and
$CO_2$ that scientific pages are full of. Without math rendering,
climate / physics / chemistry pages look broken.

**Scope**: wrap `#/##/###` in `<h2 id="slug-text">` etc. (collision-
safe via counter) in `_md_to_html()`. New left-rail TOC component
with `IntersectionObserver` highlighting the active item on scroll.
Vendor KaTeX (~900 KB, local-first principle beats CDN), call
`renderMathInElement()` after the markdown injects.

**Trade-offs**: MathJax is richer and 5× bigger — pick KaTeX.

**L1 test**: rendered HTML contains `<nav class="wiki-toc">` with
one `<a href="#...">` per `<h2>`; math spans get the expected
`class="katex"` wrapper after init.

### #6 — "Ask this page" inline RAG (~200 LOC, 3 h)

**Why**. Repurposes the existing Ask / SSE job machinery, scoped to
the current page's `source_doc_ids`. The *right* scope for most
questions a reader forms mid-read. Closes the "wait, but what about
X?" loop without making the user leave the page.

**Scope**: new `/api/wiki/page/{slug}/ask` that calls
`hybrid_search(..., doc_id_filter=source_doc_ids)` then streams via
the existing SSE job pool (`_run_generator_in_thread`). Inline chat
component at the bottom of each wiki page. Offer a "broaden to full
corpus" toggle for cross-paper answers.

**L1 test**: mock retrieve, call the endpoint, assert
`doc_id_filter` is populated from the page's `source_doc_ids`.

### #7 — Inline KG preview on concept pages (~250 LOC, 3 h)

**Why**. Concept pages are stubs today. We already have KG triples
keyed by subject/object, the 3D KG modal, and shareable `#kg=...`
URLs from Phase 48d. Render a 200×200 SVG showing the concept node
+ top-6 neighbors at the top of every concept page; click opens the
KG modal ego-focused on that node. Turns stub concept pages into
useful summaries *without needing `--rewrite-stale`*. Also: render
a "Facts from the corpus" section listing all triples where the
concept's slug matches subject or object, coloured by predicate
family.

**Scope**: extend `/api/wiki/page/{slug}` to include
`related_triples` when the slug looks like a concept; render a small
server-side SVG (or a frozen-tick client-side render) for the ego
preview; click-through composes `#kg=` with the concept pinned.

**L1 test**: render a concept page with ≥3 KG triples, assert
`<svg class="ego-preview"` + `<a href="#kg="` in output, plus a
"Facts from the corpus" heading with the triples.

### #8 — Lead-paragraph typography + staleness banner + "My take" annotation (~200 LOC + migration, 3 h)

**Why**. Three small upgrades that shift the perceived quality of
the reading surface. Wikipedia's lead-paragraph idiom — first
paragraph in slightly larger type, subtle left-border accent — is
the single biggest discovery accelerator in a general-purpose wiki;
for paper summaries the lead *is* the TL;DR. A staleness banner
surfaces the `needs_rewrite` flag the model has but the UI doesn't.
"My take" is the most-requested annotation idiom in Obsidian/Logseq
academic workflows and costs one small table.

**Scope**: CSS class `wiki-lead` on the first `<p>`; banner when
`needs_rewrite=true`; migration adding `wiki_annotations (slug PK,
body TEXT, updated_at)` + `PUT /api/wiki/page/{slug}/annotation`
endpoint rendering below the content.

**L1 tests**: `wiki-lead` on the first rendered `<p>`; PUT→GET
roundtrip on the annotation; stale-page banner renders when the DB
flag is set.

### #9 — Keyboard shortcuts + permalink with scroll position (~100 LOC, 1 h)

**Why**. Together with #3 this makes the wiki keyboard-driven. `j/k`
for next/previous page in the current list, `/` focus search, `g w`
wiki, `g h` home, `?` shows a cheatsheet. Users who discover `j/k`
never go back. `#wiki/foo#scroll=h-id` in the fragment restores
scroll position on load — zero server change.

**L1 test**: key-router handles `j`, `k`, `g` then `h`/`w`, and `/`.

## Explicit skip list

- **MathJax** — KaTeX wins on size and speed. Skip.
- **Syntax highlighting for code blocks** — scientific wikis rarely
  contain code. Skip Prism / highlight.js; revisit if someone
  complains.
- **Global graph view as primary navigator** — already have the KG
  modal; don't promote it. The 2025 academic-Obsidian consensus is
  that global-graph navigation is decorative.
- **Roam-style block references** — sciknow pages are LLM-compiled,
  not hand-edited at block granularity.
- **Database-as-page views (Notion / Capacities)** — overkill for a
  read-heavy surface.
- **Dendron hierarchical filename schemas** — solves a problem
  `slug + tags` doesn't have.
- **Inline persistent highlights with char-offset storage** —
  offsets rot when `wiki compile --rebuild` overwrites content.
  Use "My take" (#8) instead.
- **Figure / table / image carry-through from source PDFs** — high
  value but a deep change touching MinerU → chunker → wiki compile
  → renderer. Defer until #1–#5 ship and users still want it.
- **Diff between page versions** — requires a versioning table and
  a mental model that conflicts with "`wiki compile` is canonical".
  Skip unless a user explicitly asks.
- **Real-time collaborative editing** — wrong product. Single-user
  local.
- **Big MediaWiki-style infobox templates (`{{infobox paper}}`)** —
  overengineering. Server-rendered sections keyed on `page_type`
  cover 95% of what we'd use this for.

## Shipping order

Minimum-viable upgrade ≈ **9 hours** (#1 + #2 + #3 + #5): SPA route
+ clickable wiki links + command palette + TOC/math. Takes the wiki
from "modal afterthought" to "real reading surface with
hyperlinks, math, navigable long pages, keyboard-driven jump-to-
anywhere."

Full stack ≈ **23 hours** for #1–#9 (~1500 LOC + 2 migrations,
spread across 5 modules). Suggested order:

1. **#1** SPA route (4 h) — foundation.
2. **#2** wiki-link rendering (2 h) — biggest perceived jump; turns
   every "see [[foo]]" into a real navigation.
3. **#3** command palette (2 h) — power-user force multiplier, and
   needed *before* the wiki gets large enough for slugs to be
   forgotten.
4. **#5** TOC + anchors + KaTeX (3 h) — makes long pages legible.
5. **#4** backlinks + related (4 h) — turns the pile into a *wiki*.
6. **#6** "ask this page" RAG (3 h) — closes the mid-read question
   loop.
7. **#7** inline KG preview + Facts section (3 h) — rescues stub
   concept pages without needing `--rewrite-stale`.
8. **#8** lead + staleness banner + My take (3 h) — polish + one
   annotation primitive.
9. **#9** keyboard shortcuts + scroll-permalinks (1 h) — last.

## External resources worth keeping

- **Emile van Krieken, "How I use Obsidian for academic work" (2025)**
  — the clearest 2025 writeup of academic-wiki habits. Confirms
  backlinks + typed links + flat folder + quick-switcher as the
  load-bearing stack.
- **Maggie Appleton, "Command K Bars"** — canonical design brief for
  the Ctrl+K pattern.
- **Mobbin: Command Palette** — concrete UI variants with examples.
- **Aaron Tay, "ResearchRabbit's 2025 revamp"** — explicit argument
  for ego-expansion over global-graph UIs in paper discovery.
- **Effortless Academic, "Litmaps vs ResearchRabbit vs Connected
  Papers (2025)"** — head-to-head; notes Litmaps' year/citation-axis
  timeline as the feature users reach for (worth cribbing later as a
  `timeline` page-type for synthesis pages).
- **LMU Library, "The Changing Landscape of Academic Search and
  Connected Papers" (2025)** — consolidation context; lens for why
  ego-expansion beats global browsing.
- **WDscholia/scholia** — the Wikidata-backed scholarly profile
  reference implementation; lift their per-entity section template
  (publications → timeline → co-authors → topics → citing).
- **Obsidian Help: Graph view** — official notes on why global-graph
  browsing is decorative for serious use.
- **Obsidian forum, "Tags and links in scientific literature note
  taking"** — prevailing hub-and-spoke atomic-note + typed-links
  pattern for academic KBs.
- **KaTeX docs (katex.org)** — `katex.render()` + `auto-render`
  extension is a 3-line integration.
- **MediaWiki rendering rules** — canonical reference for wiki-link
  grammar and `See also` / lead-paragraph conventions.

## Bottom-line recommendation

The wiki surface is undersold, not underbuilt. The data is there
(paper summaries, concept pages, KG triples, embeddings, source-doc
references); the reading UI swallows them inside a two-tab modal.
**Ship #1 + #2 + #3 + #5 — four changes, nine hours — and wiki
browsing goes from "I don't go there" to "I open wiki pages as
often as the book reader."** Everything past #5 in the stack is
additive and can be sequenced by whoever's using the wiki heaviest
at the time.
