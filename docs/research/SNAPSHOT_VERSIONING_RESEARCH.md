# Snapshot & Version Management — Research + Proposal

**Status**: research note · not yet implemented · 2026-04-26
**Companions**: [`docs/reference/BOOK.md`](../reference/BOOK.md), [`docs/reference/BOOK_ACTIONS.md`](../reference/BOOK_ACTIONS.md)
**Author**: post-soak design pass for v2.1 / v2.2 polish

This doc surveys how sciknow currently handles snapshots + draft versions
across the four user-visible granularities (book outline, chapter,
section, single draft), compares against peer tools (Google Docs,
Notion, Overleaf, Manuskript), identifies the concrete gaps, and
proposes a unified model with a copy-pasteable CLI + GUI spec.

The motivation: re-running `book outline --overwrite` (just shipped),
running `book autowrite-all`, or accepting a `revise` pass should all
leave the user with an obvious "go back to before this" affordance and
a clear *summary of what changed* — not just a snapshot row buried in
a list with a name + word count.

---

## TL;DR

Sciknow has the *plumbing* for multi-level snapshots (PG schema,
restore endpoints, history panel). What's missing is **a unified
timeline UX**, **diff briefs**, and **automatic snapshots before
destructive multi-section ops**. The recommended shape:

1. One conceptual model: **everything a user can be afraid to lose
   has a snapshot timeline**. Section drafts already do (the `version`
   chain in the `drafts` table). Chapter and book don't — they have
   *manual* snapshots only.
2. Add **automatic snapshots** before any LLM operation that touches
   ≥1 section's content (autowrite, autowrite-all, finalize-draft
   rewrites, revise-with-figure-fix). Free; sciknow already snapshots
   before book-outline overwrite.
3. **Diff briefs** as the universal currency:
   `Δ +1,247 words / -380 words · 4 paragraphs added / 1 removed · 2 new citations`.
   Computed once on snapshot-create, cached on the row, rendered
   identically in CLI and GUI.
4. **CLI**: extend the existing `book snapshot{s,-restore}` family
   with `--diff` and a new `book history <chapter-or-section>` verb
   that walks the version chain.
5. **GUI**: a single "Timeline" panel (replaces the current History
   panel) that pivots between draft / chapter / book scopes via tabs;
   each row carries the diff brief; hovering opens a floating compare
   pane; clicking restores.

---

## Current state inventory (v2.0.0rc2)

### Schema

| Table | Granularity | Trigger | Storage | Restore semantics |
|---|---|---|---|---|
| `drafts` (`version`, `parent_draft_id`, `is_active`) | section | every autowrite iteration + every manual `book write` + every `revise` produces a new row | full content per row | activate via `is_active` flag (Phase 54.6.309) |
| `draft_snapshots` (`scope='draft'`) | one section's content frozen | manual `POST /api/snapshot/<draft_id>` | raw text | overwrites current content |
| `draft_snapshots` (`scope='chapter'`) | bundle of one chapter's drafts | manual `book snapshot --chapter N` | JSON bundle | inserts NEW draft versions per section (non-destructive) |
| `draft_snapshots` (`scope='book'`) | bundle of every chapter | manual `book snapshot` (or `book outline --overwrite` since today) | JSON of chapter bundles | per-chapter restore loop |

**Live source**: `sciknow/storage/models.py:430-492`

### CLI surface today

```
sciknow book snapshot "Title"                 # whole-book bundle
sciknow book snapshot "Title" --chapter 3     # one chapter
sciknow book snapshots "Title"                # list (book + chapter rows)
sciknow book snapshot-restore <id>            # restore a bundle
sciknow book draft scores <draft-id>          # autowrite score per version
```

There is **no CLI verb** for:

- listing the version chain of a single draft (drafts table) at scope=section
- showing what changed between two snapshots
- previewing the contents of a snapshot before restore
- per-section snapshot at the user level (drafts table is per-section but
  the user thinks in terms of named saves, not sequential version numbers)

### GUI surface today

| Panel / control | What it does | File:line |
|---|---|---|
| **History** modal | Lists `drafts` rows for the current section: word count, score, model, "Make active" button | `sciknow.js:3774`, `showVersions()` |
| **Save as new version…** toolbar | Manually create a labelled `drafts` row pointing at the current section | `sciknow.js` (Phase 54.6.312) |
| **Snapshots** modal (Book menu) | List + restore book/chapter snapshots | `sciknow.js`, `routes/snapshots.py` |
| `diffSnapshot()` | Half-built stub — fetches snapshot content but doesn't actually compute a diff | `sciknow.js:15012` |
| `restoreSnapshot()` | Overwrites current draft content with snapshot text | `sciknow.js:15034` |

The History modal is *good* for section-level navigation (Phase
54.6.309 + .312 polish made it competent) but **isolated** — it
doesn't surface chapter or book scope. The Snapshots modal is the
opposite: book/chapter, no section drill-down.

### Existing autowrite checkpointing

Each autowrite iteration produces a fresh `drafts` row with
`version = max(version) + 1` and `parent_draft_id` pointing back. So
autowrite is *already* preserving every iteration as a snapshot —
just not labelled as such. The `final_overall` autowrite score is
stamped on each iteration row's `custom_metadata`.

This is great per-section. It's **invisible at chapter/book scope** —
running `autowrite-all` over 8 chapters × 5 sections × 4 iterations
creates 160 draft rows with no enclosing "session" boundary.

---

## Gaps

1. **No "session" concept.** A user runs `autowrite-all` overnight,
   gets back 160 new draft rows. There's no single rollback target
   — you'd have to flip `is_active` on every section by hand.
   Mitigation today: snapshot the chapter manually before running.
   But that's a discipline tax.

2. **No diff briefs.** Snapshot list shows
   `name · scope · words · created_at`. The user can't tell which
   snapshot is "the one before I rewrote chapter 3" without restoring
   each in turn.

3. **No cross-snapshot compare.** Restore is the only verb. You can't
   pick two snapshots and ask "what changed?" Peer tools (Google Docs,
   Overleaf, Notion) all support compare-any-two.

4. **No structural diff for book scope.** A book snapshot's diff is
   meaningful in two dimensions: prose changes per section AND
   structural changes (chapters added / renumbered / removed,
   section lists altered). Today there's no representation of either.

5. **Section-level snapshot UX is split brain.** `drafts.version` is
   the snapshot timeline at section scope; `draft_snapshots` is for
   bundles. The user has to learn two mental models. Most prose tools
   unify them: every save is a checkpoint, with rollup by container.

6. **No automatic safety nets for non-outline destructive ops.**
   `book outline --overwrite` snapshots automatically. `autowrite-all`,
   `book revise --all-sections`, `book finalize-draft --regenerate`,
   etc. don't.

7. **Diff stub is unfinished.** `diffSnapshot()` in `sciknow.js:15012`
   exists but only renders the snapshot content; no actual word-level
   diff is computed.

---

## Peer-tool survey

Each tool taught the prose-software industry one specific lesson;
combining the lessons gives a sound design budget.

| Tool | Pattern worth borrowing | Pattern worth NOT borrowing |
|---|---|---|
| **Google Docs** | "Version history" sidebar with **named** + **unnamed** revisions; click a row → diff highlighted in the document; restore = creates a new revision (non-destructive). **Named revisions** persist forever; unnamed ones get pruned automatically. | Auto-saves at every keystroke generate hundreds of rows; aggressive pruning is required. |
| **Notion** | "Page history" with date · author · word delta brief. **Daily rollups**: every revision in a calendar day collapses to one row by default; expand to see fine-grained ones. | Restore is whole-page only; no per-block restore. |
| **Overleaf** | **Compare any two revisions**. Author colours. Named labels (`v1-submission`, `pre-review`). | Linear timeline with no scope rollup — fine for one paper, awkward for a multi-chapter book. |
| **Manuskript / yWriter** (novel-writing apps) | Per-scene + per-chapter **snapshots** with **restore-just-this-scene** from a book-wide snapshot. Distinguishes between *scene revisions* (linear chain) and *backups* (named, persistent). | Their UI is clunky; a tree of scenes is hard to scan past 30 entries. |
| **Git** | The mental model that every state is hashed and addressable; you can compare arbitrary two commits; merging is possible. | Most prose users do NOT want to think in commits and branches; the file-system metaphor leaks. |
| **VS Code "Timeline"** | Per-file timeline auto-rolled-up by source (git, local-history). | Timeline is per-file only; no project-wide view. |
| **Microsoft Word "Track Changes"** | Per-paragraph attribution + accept/reject per change. | Hard to reason about at scale; cumulative track-changes load makes documents slow. |

**Synthesis**: the standard prose pattern is

- **Linear version chain** at the smallest unit (section / scene)
- **Bundle snapshots** at higher containers (chapter / book) at user-named or tool-triggered moments
- **Diff brief** at every entry: words ± / paragraphs ± / structural Δ
- **Compare-any-two** + **restore-this-only** semantics
- **Rollup by trigger / day** in the UI to keep the list scannable

Sciknow's existing schema already supports the first two. The work
ahead is the diff brief, the compare verb, the rollup UI, and a few
more auto-snapshot triggers.

---

## Proposed unified model

### Granularity & trigger matrix

| Scope | Created by | Auto-trigger | Prune policy |
|---|---|---|---|
| **section/draft** (`drafts.version`) | autowrite iter, manual `book write`, manual `revise`, "Save as new version" | every iteration | keep all (cheap, full content per row) — already the case |
| **section/snapshot** (`draft_snapshots.scope='draft'`) | "Snapshot this section" GUI button, `book snapshot --section <id>` (NEW) | before any single-section LLM op (NEW: revise, finalize, …) | rolling 30-day; named ones never prune |
| **chapter** (`draft_snapshots.scope='chapter'`) | `book snapshot --chapter N`, GUI Snapshots modal | before `autowrite-chapter`, before any multi-section LLM op (NEW) | named: never; unnamed: rolling 14-day |
| **book** (`draft_snapshots.scope='book'`) | `book snapshot`, `book outline --overwrite` (auto), GUI | before `autowrite-all`, before `book outline --overwrite` (already auto) | named: never; unnamed: rolling 7-day |

The "scope='draft'" row is the only new conceptual addition — and it's
already in the schema. We just need the CLI verb + GUI control.

### Diff brief format

A row in any timeline (CLI or GUI) carries this 5-tuple:

```
<scope>  <name>  <date>  <size>  <Δ-brief>  [trigger]
```

Where `<Δ-brief>` is computed once at snapshot-create time and
persisted to a new `meta JSONB` column on `draft_snapshots`. Three
flavours:

**Prose diff** (sections, draft-scope):

```
Δ +1,247 / -380 words · 4¶ added · 1¶ removed · 2 new citations
```

Implementation: word-level diff via `difflib.ndiff` or the Python
`diff_match_patch` lib (~5ms for a 5kw section). Paragraph count
change is `len(after.split('\n\n')) - len(before.split('\n\n'))`.
Citation count is the change in `[N]` markers. All linear in section
size; comfortable to compute synchronously.

**Bundle prose diff** (chapter, book): aggregate of the per-section
prose diffs in the bundle, plus structural deltas:

```
chapter 3:  Δ +2,840 words across 4 sections · 1 section added
            (long_term_solar_modulation_patterns)
            highest churn: solar_dynamo_behavior_over_millennia (+1,400w)
```

**Structural diff** (book outline level):

```
2 chapters added (10. The Tipping Points · 11. Future Outlook)
1 chapter removed (Climate Models and the IPCC Narrative)
3 chapters renamed
17 sections added · 4 removed
```

The structural diff diff'ing two book bundles compares
`{(ch.number, ch.title): [section.slug, ...]}` dicts.

### Compare semantics

- **Same scope, two snapshots** → side-by-side prose / structural diff.
- **Snapshot vs current** → same shape (current = synthesised on the fly).
- **Across scopes** (e.g. compare a section snapshot to a book snapshot)
  → not supported; the UI hides the option to keep semantics clean.

### Restore semantics

Inherits from the existing scope-keyed restore (`scope='draft'` overwrites,
`scope='chapter'` and `scope='book'` insert new versions). One addition:

**Selective restore from a bundle**: pick any subset of sections inside
a chapter or book bundle and restore just those. Today it's all-or-nothing.
UI: tree view inside the snapshot row with section-level checkboxes.

---

## CLI surface — concrete proposal

```bash
# Existing — unchanged shape, just add --diff:
sciknow book snapshot "Title"                          # whole book
sciknow book snapshot "Title" --chapter 3              # one chapter
sciknow book snapshots "Title"                         # list
sciknow book snapshot-restore <id>                     # restore bundle
sciknow book snapshot-restore <id> --diff              # NEW: dry-run + diff
sciknow book snapshot-restore <id> --sections a,b,c    # NEW: selective restore

# NEW — section-scope snapshots:
sciknow book snapshot --draft <draft-id> --name "pre-revise"
sciknow book snapshot --section <chapter-num>:<section-slug>   # convenience

# NEW — section history timeline (drafts.version chain + diffs):
sciknow book history <chapter-num>:<section-slug>      # versions + briefs
sciknow book history "Title" --chapter 3               # all sections in ch3
sciknow book history "Title"                           # whole book rollup

# NEW — explicit compare:
sciknow book diff <id-or-version-A> <id-or-version-B>
sciknow book diff <draft-id> --vs current              # snapshot vs live

# NEW — flip an existing draft version active:
sciknow book activate <draft-id>                       # already in web; CLI mirror
```

Output of `book history <ch>:<section>` (the new flagship):

```
draft a3c9e1d2  ch3.solar_dynamo_behavior_over_millennia

  v8  ✓ active   2026-04-22 14:33  1,621w  autowrite-iter-3 (score 0.81)
                  Δ +127 / -83 words · 1¶ added
  v7              2026-04-22 14:31  1,577w  autowrite-iter-2 (score 0.76)
                  Δ +445 / -2 words · 3¶ added · 1 new citation
  v6              2026-04-22 14:29  1,134w  autowrite-iter-1 (score 0.61)
                  Δ from blank · 11¶ added · 7 citations
  v5              2026-04-21 09:12  1,012w  manual revise
  v4   "pre-revise"  2026-04-21 09:08  1,440w  named save
  v3              2026-04-20 19:45  1,440w  book write
  ...
```

Output of `book history "Title" --chapter 3` is the same with one row
per section, rolled up to "latest active version" plus a
chapter-level brief at the top.

---

## GUI surface — concrete proposal

Replace the current History modal with a **unified Timeline panel**
that has three tabs:

```
┌─ Timeline ────────────────────────────────────────────────────────┐
│  [ Section ]  [ Chapter ]  [ Book ]                  [↻ refresh] │
├──────────────────────────────────────────────────────────────────┤
│  Section: Ch.3 / solar_dynamo_behavior_over_millennia            │
│                                                                   │
│  ◉ v8     2026-04-22 14:33   1,621w   autowrite iter 3 (0.81)   │
│  ◉ v7     14:31              1,577w   autowrite iter 2  (0.76)   │
│         Δ +445 / -2 · 3¶ added · 1 new cite                       │
│  ◉ v6     14:29              1,134w   autowrite iter 1  (0.61)   │
│  ◉ v5     2026-04-21 09:12   1,012w   manual revise               │
│  ◉ v4     09:08 "pre-revise" 1,440w   ← named                     │
│         Δ from v3: +0w (note save)                                │
│  ─────                                                            │
│  ⊞ 2026-04-20 (3 versions, expand)                                │
│  ⊞ 2026-04-15 (2 versions, expand)                                │
│                                                                   │
│  Right-click row → [ Restore | Compare with active | Compare    │
│                     with… | Rename | Delete ]                    │
└──────────────────────────────────────────────────────────────────┘
```

**Section tab**: walks `drafts.version` for the current section, with
day-rollup (≥1 day older = collapsed by default). Hover a row → side
panel shows the prose diff vs the active version.

**Chapter tab**: lists `draft_snapshots.scope='chapter'` rows AND
auto-rollups of section-level versions ("autowrite-chapter session
2026-04-22, 5 sections, +6,340w"). Hover → shows the per-section
brief.

**Book tab**: lists `draft_snapshots.scope='book'` rows + the same
auto-rollup pattern for any multi-chapter operation. Hover → shows
structural diff for outline-level bundles.

### Compare floating pane

Click "Compare with…" → docked side panel:

```
┌─ Compare ───────────────────────────────────────────────────────┐
│  v4 "pre-revise" (2026-04-21 09:08)   →   v8 active             │
│                                                                  │
│  Δ +181 / -469 words · 4¶ added · 3¶ rewritten · 2 new cites    │
│                                                                  │
│  ¶ Para 1  unchanged                                             │
│  ¶ Para 2  rewritten:                                            │
│           - the Maunder Minimum "shut down" the dynamo entirely  │
│           + the Maunder Minimum suppressed but did not stop the  │
│             solar dynamo, with reduced amplitude oscillations    │
│           citations: was [12], now [12,40]                       │
│  ¶ Para 3  added (+312 words on solar grand minima recurrence)   │
│  ...                                                             │
│                                                                  │
│  [ Restore v4 ]  [ Restore v4 + keep v8 as orphan ]   [ Close ] │
└──────────────────────────────────────────────────────────────────┘
```

### Auto-snapshot trigger UI

Anywhere a destructive multi-section LLM op is about to run (autowrite-
chapter, autowrite-all, book outline --overwrite, etc.), the
confirmation prompt mentions the auto-snapshot:

```
┌─ Run autowrite over Chapter 3? ──────────────────────────────────┐
│  6 sections · estimated wall ≈ 18 min                            │
│                                                                   │
│  Auto-snapshot before running:  [✓] enabled (default)            │
│  Snapshot name: pre-autowrite-2026-04-26                         │
│  Restore later via: Timeline → Chapter tab                       │
│                                                                   │
│              [ Cancel ]   [ Run autowrite ]                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Implementation phases

Smallest-shippable-slice first. Each phase is independently mergeable.

### Phase 1 — diff brief on existing snapshots (1 day)

- Add `meta JSONB` column to `draft_snapshots` (Alembic migration).
- Compute the diff brief at snapshot-create time. For chapter/book
  scope, the brief is per-section-rolled-up.
- Render the brief in `book snapshots` CLI output.
- Render the brief in the existing Snapshots modal (one extra column).
- L1 contract test for the brief shape.
- **No new CLI verbs.** Existing surfaces just get richer.

### Phase 2 — section-scope snapshot CLI + auto-trigger (1-2 days)

- New verb `book snapshot --draft <id> --name "..."` (mirrors the
  existing web `POST /api/snapshot/<draft_id>`).
- Auto-snapshot before `book revise`, `book finalize-draft`, and any
  other single-section LLM op. Default ON, `--no-snapshot` to skip.
- L1 contract test pinning the auto-trigger sites.

### Phase 3 — `book history` + `book diff` CLI (2 days)

- New verb `book history <chapter-num>:<slug>` walks `drafts.version`
  chain (+ any draft-scope snapshots interleaved by created_at) with
  the diff brief per row.
- New verb `book diff <id-A> <id-B>` (or `<id> --vs current`) prints
  a word-level diff. Use `difflib.unified_diff` for stdout; emit JSON
  with `--json` for the GUI to consume.
- L2 roundtrip test.

### Phase 4 — auto-snapshot for `autowrite-chapter` / `autowrite-all` (1 day)

- Add a chapter snapshot at the start of every autowrite-chapter run
  (named `pre-autowrite-{date}`).
- Surface in the GUI confirmation modal.
- Auto-prune unnamed snapshots: keep last 14 per chapter, last 7 per
  book.

### Phase 5 — Timeline GUI panel (3-4 days)

- New unified modal replacing the current History + Snapshots modals.
- Three tabs (Section / Chapter / Book) with day-rollup.
- Compare floating pane with paragraph-level diff highlighting.
- Selective restore via section checkboxes inside chapter/book bundles.

### Phase 6 — structural outline diff (1-2 days)

- Diff two book-scope snapshots' chapter-list shapes.
- Render in the Book-tab compare pane and in `book diff --book`.

Total estimate: ~10 working days of engineering, four ship boundaries
that each leave the system better than before.

---

## Open questions

1. **How aggressively should we prune?** The user has 814 papers ×
   typical 4-iteration autowrite ≈ thousands of draft rows. None
   pruned today. A `book history --prune-unnamed --keep 30` verb would
   probably be welcome eventually but isn't critical until storage hurts.

2. **Should chapter/book snapshots be tagged with the operation that
   triggered them?** The current `name` column is free-form. Adding
   a `trigger` enum column (`manual`, `autowrite-chapter`,
   `outline-overwrite`, `revise-all`, …) would let the timeline filter
   "show me only autowrite checkpoints" cleanly.

3. **Does the GUI need an export-snapshot-as-zip affordance?** Useful
   for sharing a specific book state with a collaborator. Probably yes,
   not urgent.

4. **Do we want to introduce branching?** Right now every section's
   `parent_draft_id` chain is linear. Allowing two children of the
   same parent would let users explore alternative revisions. The
   schema supports it (`parent_draft_id` is a many-to-one column).
   The UX is the hard part — most prose users don't want git-style
   branches. Defer.

5. **Cross-project snapshot transfer?** The DB has per-project
   isolation (Phase 43+); a snapshot is bound to its project. Moving
   a chapter from project A to B is currently not possible without
   a full export. Out of scope for this proposal.

---

## What we're NOT proposing

To stay scoped:

- No git-style branching / merge UI. Linear chain stays linear.
- No real-time collaboration / presence — single-user app.
- No storage-format changes beyond adding a meta column. Full
  content per row stays the storage model. Compression / delta
  encoding can be a v3 problem.
- No undo/redo in the active editor (that's a different feature —
  in-session keystrokes vs persisted snapshots).
- No "revert chapter to 30 days ago" calendar selector. Manual
  date filtering inside the timeline list is enough.

---

## Appendix: file refs that informed this doc

- `sciknow/storage/models.py:407-492` — drafts + draft_snapshots schema
- `sciknow/web/routes/snapshots.py` — server endpoints (~350 LoC)
- `sciknow/cli/book.py:2110-2280` — snapshot CLI verbs
- `sciknow/web/static/js/sciknow.js:3774-3920` — `showVersions()` History modal
- `sciknow/web/static/js/sciknow.js:15012-15043` — `diffSnapshot()` stub +
  `restoreSnapshot()` (proof that the GUI half-built the diff but never
  finished it)
- `sciknow/core/book_ops.py` — autowrite engine that creates the section
  version chain (Phase C re-export shim)

The implementation phases above are designed to be shippable inside the
v2.1 / v2.2 windows — none require a substrate change, and Phase 1 is
small enough to ship alongside other v2.1 cleanup work.
