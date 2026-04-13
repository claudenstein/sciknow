# Strategy: Human + Autowrite Division of Labor

*A direct answer to "should I just autowrite the whole book?"*

---

## The direct answer: no, don't autowrite the whole book

The pipeline is excellent at what it does, but it cannot do the thing a
scientific book *is*. The intuition that autowrite's fact-checking is
better and less biased than a human's is partly right and partly wrong,
and the reasons matter for the strategy.

## What the pipeline genuinely wins on

- **Consistency across 100k+ words.** Humans drift; the style
  fingerprint (Layer 5) doesn't.
- **Citation density.** Humans get lazy around paragraph-end; LongCite
  enforces per-sentence grounding.
- **Groundedness on well-established facts.** CoVe + verification
  catches ~95% of "the draft says X, the cited source says ~X" drift.
- **Rapid first drafts.** Gets you to *critique mode* faster, which is
  where the real work happens.
- **Mechanical consistency.** It will not accidentally contradict a
  previous chapter; PDTB-lite + Centering keep it honest.
- **Non-selective summarization of what's IN the corpus.** No personal
  stake, no favorite papers.

These are real wins. For sections that are fundamentally summarization
(background, methods overviews, consensus syntheses), autowrite is
already better than most humans writing under deadline pressure.

## Where the pipeline fails, and will keep failing

1. **Fact-check ≠ truth-check.** Groundedness confirms "the draft
   matches the source." It doesn't confirm "the source is correct,"
   "the source is the strongest evidence," or "the source hasn't been
   retracted." A paper from a lab you distrust scores identically to a
   landmark replication.

2. **Corpus bias is invisible to the writer.** The autowrite will
   faithfully reflect whatever you ingested. If your corpus is 80%
   climate-sensitivity papers from one research tradition, the book
   will be too. The LLM cannot audit what's *missing* from the corpus
   — only you can.

3. **No novel argument.** LLMs compose, they don't originate. If your
   book's value-add is *"what I already know that no one else has said
   yet,"* autowrite erases that. It produces the average of the
   corpus, not your reading of it. A literature review on steroids.

4. **No domain-specific error detection.** A non-trivial fraction of
   papers in any field have methodological issues that an expert sees
   immediately and an LLM never will. (Proxy calibration errors,
   cherry-picked time windows, conflated variables.) The pipeline
   won't flag these; it'll cite them.

5. **Editorial judgment under uncertainty.** "This is contested" /
   "this is cutting-edge" / "this is orthodoxy" are social knowledge
   claims. The LLM sees text; you see lab politics, retraction
   histories, methodological skepticism.

6. **"Less biased" is partly a mirage.** You replaced LLM-prior bias
   (which is real — training data skew) with corpus-selection bias
   (also real — you chose what to ingest). The latter is more
   controllable but doesn't vanish just because the words came through
   a retrieval pipeline.

## What a book-autowritten-end-to-end actually reads like

Blunt version. It would read as:

- Correct within the corpus
- Well-cited per sentence
- Consistent in style and terminology
- Structurally conventional (CARS moves make it so)
- Appropriately hedged
- **Unoriginal.** No thesis emerges from pure synthesis.
- **Invisible.** Reads like an LLM wrote it, because one did.
- **Safe to boring.** Hedging fidelity is a two-edged sword:
  well-calibrated sections lose their ability to argue forcefully.
- **Unable to commit.** When the corpus disagrees, MADAM-RAG-lite
  presents both sides; a human author takes a position.

That's a useful artifact — it's a very good annotated literature
review. But it's not a scientific *book*. What makes a scientific book
valuable, beyond a review paper, is the author's argument: a position
the corpus doesn't make for itself, defended against the corpus's own
counterevidence.

---

## The ultimate strategy — principled division of labor

End-to-end workflow optimized for the current pipeline.

### 1. Corpus curation — **you-heavy**

The single most important decision. Your corpus IS your bias.

- Actively hunt papers that *disagree* with your thesis. `sciknow db
  expand` follows citations; run it on dissenting papers too, not just
  ones you agree with.
- Deliberate diversity: journals (not just top tier), methods (not
  just your methods), eras (decade-balanced), schools of thought.
- **Self-check**: run `ask synthesize "the strongest counterargument
  to <your thesis>"`. If the output sounds weaker than you know the
  real counterargument to be, your corpus is imbalanced — add papers.
- Target: 3000–5000 papers is usually enough. Quality of selection
  matters more than count.

### 2. Thesis & leitmotiv — **you-only**

Only you know what this book is arguing that isn't already said. No
LLM can produce this. The `books.plan` field is where your voice
lives.

- Write the leitmotiv yourself. 300–500 words. Don't regenerate it —
  the LLM-generated plan will drift toward "this is a balanced survey
  of X," which is the opposite of a thesis.
- Test the thesis: `ask synthesize "evidence against <your claim>"`.
  If the evidence is strong, either revise the thesis or plan to
  confront it head-on in specific chapters.

### 3. Chapter structure — **mixed**

- Let `book outline` propose. It's good at conventional structure.
- You arbitrate. Ask: does this structure serve *my argument* or is it
  generic? Reorder, merge, split.
- Add at least one chapter that is explicitly *"confronting
  counterevidence"* — makes MADAM-RAG-lite + Toulmin scaffolds
  actually earn their keep.

### 4. Topic clustering — **tool-heavy, you-guided**

- `catalog cluster` organizes. Review the cluster names.
- False distinctions (two clusters that should be one): rename one,
  merge intent in the plan.
- Missing distinctions: you know which papers belong together even
  when embeddings don't. Create a `--topic` override if needed.

### 5. Per-section planning — **you-heavy, 3 sentences each**

This is the most underused knob.

- For each section, write 2–3 sentences in `sections_meta[].plan`:
  *what angle does THIS section take, what are the key claims, what
  should it NOT do?*
- This is where you inject original framing. Without it, the writer
  reverts to genre conventions.
- Set per-section `model` overrides (Phase 37): flagship on the
  argumentative sections, fast on the descriptive ones.

### 6. Drafting — **tool-heavy, autowrite with convergence**

- Snapshot chapter (Phase 38) before running autowrite-all.
- Autowrite with `max_iter=3, target_score=0.85`. Let CoVe fire on
  anything below 0.85 groundedness.
- Watch the compute counter (Phase 35). If a section costs more than
  5× your average, the plan is probably inconsistent with the corpus
  — refine the plan rather than burning tokens.
- Monitor `autowrite/latest.jsonl` via `tail -f ... | jq` for stalls.

### 7. Human review — **you-heavy, ~40–50% of total project time**

This is where the book becomes yours. Read every section. For each,
flag three things:

1. **Miscalibrated confidence** — hedging fidelity catches obvious
   overreach; you catch the subtle kind.
2. **Weak citations** — the LLM can't distinguish a preprint from a
   top-tier paper evidentiarily. You can.
3. **Missing perspective** — what *didn't* it say that you'd expect
   to see? That's usually your thesis trying to surface.

Use `review` + `revise` with specific instructions, not generic
"improve this." E.g.: *"Add a counterargument in para 3, citing
Schmidt 2023 [N]. Strengthen the claim in para 5 — the hedge is too
soft, the evidence is overwhelming."*

### 8. Preference marking — **you, while reviewing**

- Every section you KEEP vs DISCARD becomes a DPO preference pair
  (Phase 32.9, Layer 4).
- Mark approvals explicitly via `preference_approved`. Two years from
  now on DGX Spark, these become training data for a writer
  fine-tuned on *your* voice (Phase 6 = Layer 6).
- **This is the compound-learning payoff.** Your reviewing effort
  doesn't just improve this book — it improves the tool for your
  next book.

### 9. Cross-chapter coherence — **you-only**

The pipeline handles section-level coherence (Centering, PDTB-lite).
It does not handle chapter-level narrative arc. That's the one thing
no algorithm touches.

- Chapter transitions, the opening, the closing — write these
  yourself.
- Reread the book top to bottom at least twice. Look for argument
  drift.

### 10. Export & iterate

- Phase 40's CLI PDF/EPUB for external reviewers.
- Real external review (colleagues, not the LLM) catches things no
  automated system can.

---

## The honest bottom line

**Autowrite produces first drafts. You produce the book.**

The right split, for a serious scientific book:

- **Tool does**: 70% of total wordcount generation, 90% of
  fact-grounding, 95% of style consistency, 100% of citation
  mechanics.
- **You do**: 100% of thesis, 100% of argument structure, 100% of
  counterevidence confrontation, 50% of every section's final form
  through review/revise.

The "ultimate strategy" isn't automation. It's **leverage**: autowrite
frees you from the drudgery of summarizing what you already know, so
you can spend that time on what only you can do — which is decide what
this book *means*.

A book that could have been fully autowritten didn't need to be
written. A book worth writing can't be fully autowritten.

---

## Related

- `docs/RESEARCH.md` — the literature + design decisions behind each
  synthesis technique in use.
- `docs/ROADMAP.md` — what's shipped and what's deferred.
- `docs/BOOK.md` — operational walkthrough of the book commands.
