# `db expand` + `db enrich` — 2026-04-19 research sweep

Follow-on to `docs/EXPAND_RESEARCH.md` (Phase 49, shipped 2026-04-14).
That doc's seven-item priority stack is all implemented and status-
confirmed; everything here is **net-new** work and is organized by
expected impact per hour. Grouped into:

1. **Enrich** — what we're not pulling today and should be
2. **Expand** — new signals / new capabilities since Phase 49
3. **Agentic / LLM-driven** — a genuinely new capability class
4. **Quality-of-life** — reliability, provenance, HITL polish
5. **Priority stack** — what to ship first

The 2026-04-14 rationale (RRF because signals are weak individually,
best-first because BFS explodes, HITL shortlist for provenance) still
holds. What changed: the corpus is bigger, ≥50-label LambdaMART
threshold is closer, and the 2024-2026 literature added three concrete
techniques we don't have (citation-context embeddings, agentic query
planning, diversity sampling) that complement — not replace — Phase 49.

---

## 1. `db enrich` — the under-powered command

Today `enrich` is a thin Crossref-title-search + arXiv fallback that
backfills DOI + basic metadata when `paper_metadata.doi IS NULL`. It
does one pass, never revisits, and stops at title/abstract/authors/year.
OpenAlex is only used as a title-search alternative and the richer
fields are discarded. Gaps:

### 1.1 Missing metadata we're already paying for

The OpenAlex `/works/{id}` response carries fields we drop on the floor.
Add one column per to `paper_metadata` (or a sibling `paper_enrichment`
table) and hydrate at enrich time. Zero additional API calls — the
response already contains them:

- **`concepts[]`** — OpenAlex's hierarchical concept tree with per-concept
  scores. Already leveraged at candidate-ranking time in expand (concept
  overlap signal); we don't persist them. Persisting unlocks:
  (a) post-hoc cluster analysis (see papers tagged "radiative forcing"
  but NOT tagged "stratospheric chemistry"), (b) the paper-type
  classifier's eval signal (Phase 54.6.80 gate), (c) a "find related
  papers in my corpus" feature without re-hitting OpenAlex.
- **`funders[]`** + **`grants[]`** — for climate/policy research the
  funder reveals stance: a paper funded by EPA reads differently from
  one funded by API (American Petroleum Institute). Persist as JSONB.
  Unlocks a "show me everything CESM-2 grantees published on this
  topic" query.
- **`authorships[].institutions[].ror`** — institutional affiliation
  via ROR (Research Organization Registry) IDs. Deduplicates
  Harvard / Harvard University / Harvard College. Unlocks co-
  institutional expansion ("expand to other NOAA/NCAR papers").
- **`referenced_works_count`** + **`cited_by_count`** + **`counts_by_year`**
  — paper-level citation velocity that we already recompute expand-
  side. Persisting turns it into a corpus-wide "what's cooling/heating
  up" view.
- **`primary_location.source.issn_l`** + **`host_organization`** —
  venue-level metadata used by the Phase 49 venue_weight signal. We
  pull it but don't store it; next expand call re-hits the API.
- **`biblio`** — volume / issue / first_page / last_page. Useful for
  proper bibliography export (Phase 40 EPUB / LaTeX export currently
  emits minimum-viable citations).

**Effort:** half-day. One migration, one adapter pass over the already-
fetched OpenAlex work.

### 1.2 Retraction + correction sweep on existing papers

Today the retraction filter is checked **only when expand considers a
new candidate**. Papers already in the corpus never get re-checked, so
a paper that gets retracted after ingestion stays in the corpus flagged
as authoritative. Add:

- **`sciknow db refresh-retractions`** — iterate `paper_metadata.doi`,
  batch-query Crossref's `works?filter=doi:…&filter=update-type:retraction`,
  mark hits in a new `paper_metadata.retraction_status` column
  (`retracted | corrected | none`) with a `retraction_checked_at`
  timestamp. Run weekly or on demand.
- **Downstream consumers** should refuse to cite retracted papers: the
  writer prompt block should include a "DO NOT cite retracted papers"
  rule, `hybrid_search` should drop retracted chunks with a hard filter.
  **Effort:** half-day + a small rule pass through the writer prompt.

### 1.3 Preprint → journal version reconciliation

Many climate papers arrive as arXiv preprints and later get a journal
DOI. Today these are two separate `documents` rows. OpenAlex's
`ids.mag`, `ids.pmid`, `ids.pmcid`, `ids.doi` for the same work are
linked — we can reconcile. Add a `paper_metadata.preprint_doi` field
and merge chunks / citations between preprint-row and published-row
without losing provenance. Reduces noise in the `expand` one-timer
filter.

### 1.4 Keywords + MeSH + JEL code enrichment

`paper_metadata.keywords` exists as `list[str]` but is usually populated
from PDF XMP (unreliable). OpenAlex ships author-supplied keywords;
PubMed ships MeSH codes for biomedical papers. Adding these turns
"search keyword" queries into hard filters + unlocks a keyword-cloud
browser in the GUI. **Effort:** 1-2h (extract-only, no schema change —
use existing `keywords` field).

### 1.5 Full-text detection signal

OpenAlex exposes `has_fulltext: bool` + `open_access.is_oa: bool`.
Today `db expand` uses these internally; `paper_metadata` doesn't
persist them. Persist so the user can `db status --no-fulltext` to
find the 60% of corpus papers for which we have metadata but no
ingested body — a useful "what's missing" view we don't expose today.

---

## 2. `db expand` — new signals since Phase 49

Phase 49 shipped co-citation + bib coupling + PageRank + isInfluential
+ one-timer filter + RRF. What the 2024-2026 literature added:

### 2.1 Citation-context embeddings (Cohan et al. 2024, SciRepEval)

**The single biggest research idea we don't use.** Instead of embedding
a candidate's title+abstract, embed the **contexts in which OTHER
papers cite it**. Semantic Scholar's `/paper/{id}/citations?fields=contexts`
returns the ~50-char sentence each citer wrote about the cited paper;
concatenate 10-20 of these, embed that, cosine against the corpus
centroid. Papers are more discriminatively described by how they're
used than by their own abstract — abstracts are marketing.

**Why bge-m3 underperforms here**: "climate sensitivity estimate" is
topical to half the corpus; "ECS of 3.2 K constrained by the last
glacial maximum" is specific. Citation contexts ARE the specific
description.

**Cost:** one S2 call per candidate (we already make one for the
isInfluential field), so a tight fold-in. The embedding is the same
bge-m3 we already load.

**Effort:** 3-4h. Biggest win in this section.

### 2.2 SPECTER2 rerank (PARKED — measured 2026-04-19, decisive negative)

Originally parked in Phase 49 as "marginal improvement, 440 MB VRAM
cost." Re-tested in Phase 54.6.121 against the 54.6.69 retrieval
probe set (200 queries, global-cooling corpus). Result was **−0.37
MRR@10 vs the bge-m3 + RRF baseline** — not marginal improvement
but catastrophic degradation:

| metric | bge-m3 baseline | SPECTER2 rerank | delta |
| --- | --- | --- | --- |
| MRR@10 | 0.576 | 0.206 | **−0.37** |
| Recall@1 | 41.0% | 13.5% | **−27.5pp** |
| Recall@10 | 87.5% | 44.0% | **−43.5pp** |
| NDCG@10 | 0.649 | 0.260 | **−0.39** |

Why: SPECTER2 was trained on **title + abstract pairs**. Our chunks
are ~512-token paragraphs of mid-paper prose with section headers
and citation markers — out-of-distribution for SPECTER2's training
contract. bge-m3 is purpose-trained for hybrid dense+sparse retrieval
on arbitrary passage shapes.

**Decision: permanent PARK as a chunk-level reranker.** SPECTER2 is
*not* a drop-in replacement for our bge-m3 + RRF + cross-encoder
pipeline.

**Where SPECTER2 might still help:** PAPER-level retrieval (rerank
the `abstracts` Qdrant collection against a query). That's a separate
use case from chunk retrieval and a future bench when we have a
paper-level eval set. Not on the current roadmap.

The bench code stays at `sciknow/testing/specter2_bench.py` so
future re-tests against new releases / new corpora are one-shot.

### 2.3 Diversity sampling (MMR / DPP)

RRF returns the top-K by fused score. The top-50 tends to cluster
topically — a corpus expansion round that adds 50 papers on the same
narrow subtopic is less useful than one that adds 40 on-topic plus 10
adjacent. Apply **Maximal Marginal Relevance** (Carbonell & Goldstein
1998) post-RRF: for each pick, discount by its cosine similarity to
papers already in the round's pick list. λ=0.7 keeps 70% of the ranker's
signal. Alternative: **Determinantal Point Processes** (Kulesza & Taskar
2012) — probabilistic diverse subset selection, strictly better but
harder to tune.

**Effort:** half-day for MMR, 1 day for DPP.

### 2.4 Negative-example learning

The HITL shortlist TSV today only captures "kept/dropped reason". Let
the user mark rows as `+` (want more like this) or `−` (avoid like
this); on the next round:
- Add the `+` papers as additional seeds alongside the corpus seeds
- Subtract the `−` paper embeddings from the query (paper2vec style)
- Use the marks as labels for a future LambdaMART model (Phase 49's
  parked item, now with data)

**Effort:** 1h for the column + subtraction trick. The LambdaMART
piece stays parked until ≥500 labels accumulate.

### 2.5 Inbound "cites me" crawl — SHIPPED 54.6.123

"What papers cite a seed" complements "what seed cites". New CLI:

```
sciknow db expand-inbound [-n LIMIT] [--relevance-threshold T] \
                          [--per-seed-cap N] [--total-limit N] \
                          [-q "topic"] [--dry-run] [--retry-failed]
```

Reuses `find_inbound_citation_candidates` (which was the backing for
the existing preview-only UI), adds relevance threshold + cached-skip
+ download/ingest + provenance writes with `source="expand-inbound"`.
Complementary to `db expand` (forward-in-time vs backward).

GUI: Inbound-cites tab in the Corpus modal gained **Expand now** button
next to the existing Preview button.

### 2.6 Author oeuvre completion

If author X has ≥3 papers in the corpus AND ≥10 total papers per
OpenAlex, auto-expand to the missing ones (after the normal relevance
filter). Today author_overlap is a **ranking signal** but never a
**seed source**. Complements the outbound-ref crawl with "go through
the most-represented authors' full bibliographies."

**Effort:** 3-4h. Useful even on its own; combines well with HITL
(user endorses the author before the crawl).

### 2.7 Funder / grant cohort expansion

For policy-relevant fields, papers from the same grant cohort are
topically coherent. OpenAlex `/works?filter=grants.award_id:<id>` fans
out cheaply. Example: for the climate corpus, NOAA CVP grant
NA19OAR4310279 produced 14 papers; 6 in our corpus, 8 not. Auto-expand
the missing 8 (subject to relevance filter). Similar for funders at the
DOE / NSF / EPA level.

**Effort:** 2h once §1.1 persists `funders + grants`.

### 2.8 OpenReview integration (conference preprints)

NeurIPS, ICLR, ACL, MICCAI, etc. papers exist on OpenReview before a
journal DOI — even before arXiv. For methods-heavy subfields we miss a
~6-month window by waiting for the conference DOI. `openreview.net`
exposes JSON API for public papers. **Relevant for** climate-ML, RL,
NLP-in-science papers that the user might want to track.

**Effort:** 4-6h (new adapter + schema for conference-preprint origin).
Low priority for pure climate corpus, high for ML-adjacent subfields.

---

## 3. Agentic / LLM-driven expansion — a new capability class

**What sciknow has today:** a static pipeline. User picks seeds (the
whole corpus, or `--relevance-query "X"`), the ranker does its work,
user reviews the shortlist, accepts or rejects.

**What the 2024-2026 literature shows:** LLM-driven expansion that
plans, executes, critiques, and iterates — **agentic RAG** (PaperQA2,
OpenScholar, SciAgent, ScholarCopilot). The LLM doesn't replace the
ranker; it replaces the static query specification AND the decision of
"did we find enough?"

### 3.1 Question-driven expansion

**`sciknow db expand --question "What is the current best estimate of
equilibrium climate sensitivity?"`**

Flow:
1. LLM (fast model) breaks the question into sub-topics (ECS
   definition, constraints from paleo / observational / modelling,
   distribution shapes, uncertainty sources).
2. For each sub-topic, run a corpus search; find the top 5 in-corpus
   papers that cover it.
3. For each sub-topic where in-corpus coverage is thin
   (≤2 papers), run expand with the sub-topic as `--relevance-query`
   and a small budget (5-10 per sub-topic).
4. After downloads + ingest, run `book gaps` equivalent to check if
   the question is now answerable; if not, the LLM refines sub-topics
   and loops.

**Why this is new:** the user doesn't need to decompose the question
into queries. The LLM does it, and **the corpus-coverage check is the
stopping rule** — far better than "novelty ratio < 30%".

**Why it's feasible:** every component exists (LLM for decomposition,
`hybrid_search` for corpus-side check, `db expand` for targeted
expansion, `book gaps` for coverage verification). It's an orchestrator
that chains them.

**Effort:** 1-2 days for a minimum viable version.

### 3.2 Iterative HITL with LLM-generated critiques

Today the shortlist TSV is raw rows. With an LLM critique pass, every
`±` decision generates a single sentence ("rejected because: method
doesn't apply to climate; mostly on machine vision") that seeds the
next round's ranker with **learned weights per sub-topic** — "paper X
similar to previously-rejected Y on feature Z". This is LambdaMART
with an LLM as feature engineer.

**Effort:** 1 day.

### 3.3 Query-plan provenance

Every expand round records **what question** was being answered and
**which sub-queries** drove which candidates. Turns expand from a
black box into a reviewable audit log. Lets the user say "I want to
redo sub-topic 3 with a tighter relevance threshold" without re-doing
the whole round.

**Effort:** 4-6h; natural outcome of §3.1 being implemented first.

---

## 4. Quality-of-life / reliability

### 4.1 Resume / checkpointing for agentic expansion — SHIPPED 54.6.124

Agentic runs (`db expand --question "..."`) can be multi-hour. Each
round now writes state to `<project>/data/expand/agentic/<slug>-<hash>.json`;
`db expand --question "..." --resume` loads that state, skips
completed rounds, and picks up at the last `next_round`. Crash
mid-round persists the error on the partial round record so resume
can show where it failed before retrying.

GUI: new **resume** checkbox in the Corpus modal's Agentic tab.

Static (non-agentic) `db expand` resume stays open — its single-shot
shape needs different state (per-candidate download progress, not
per-round coverage) and isn't as painful since individual rounds are
typically 5-30 min, not hours.

### 4.2 Per-paper provenance

The `documents` table today only records `original_path`. We don't
know *why* a paper was added — expand (which round? which signals?),
enrich (which layer won?), manual `ingest file`? Add
`documents.provenance JSONB`: `{source: "expand" | "ingest" | ...,
round: 2, signals: {co_cite: 12, bib_coupling: 0.08, rrf_score: 0.31},
seed_paper_ids: [...]}`. Powers a "why is this paper here?" tooltip
and ablation analyses ("drop everything with rrf_score < 0.25, re-run
autowrite, did scores change?").

**Effort:** 1 day for the backfill pass on already-ingested papers
(best-effort from `expand.log`) plus the forward-write hook.

### 4.3 Corpus-drift detection

After each expand round, compute the centroid shift of corpus
abstract-embeddings vs. before. If the cosine drop exceeds 0.05 in one
round, flag it — the expansion has pulled the corpus toward a new
subtopic. Might be intentional; should be explicit. Emit to
`data/expand/drift.log` + surface in the `db stats` output.

**Effort:** 2h.

### 4.4 Venue blacklist/whitelist CLI

`sciknow project config venue-block "Journal of Multidisciplinary
Frontiers"` + venue-list in the `.env.overlay`. Today the predatory
list is hard-coded in `expand_filters.py:29` (`load_extra_predatory_patterns`
exists but isn't wired to CLI). Expose via:

- `sciknow project config venue-block <pattern>` — append to overlay
- `sciknow project config venue-allow <pattern>` — override blocklist
  for a specific venue that falls foul of the predatory pattern match
  (some legitimate venues hit false-positive)

**Effort:** 2h.

### 4.5 Incremental re-embed after enrich

Today `db enrich` updates title/abstract but the paper's chunks +
abstract embedding stay stale if the enrich fills in a previously-empty
abstract. Trigger an `embedder.embed_paper(paper_id)` on enrich where
the abstract column transitions from NULL to non-NULL.

**Effort:** 1h.

---

## 5. Priority stack

Ship in this order; each line is independently shippable.

### 5a. Tier 1 — demonstrable user-visible wins, ≤ half-day each

1. **Persist OpenAlex metadata we already fetch** (§1.1) — `concepts`,
   `funders`, `grants`, `ror_ids`, `counts_by_year`, `biblio`. Zero
   additional API load; migration only.
2. **Retraction sweep on existing papers** (§1.2) — `db
   refresh-retractions` + a hard-filter rule in the writer prompt.
3. **Diversity sampling (MMR) post-RRF** (§2.3) — surfaces topical
   variety inside each expansion round.
4. **Incremental re-embed after enrich** (§4.5) — corrects a silent
   data-quality gap.
5. **Venue block/allow CLI** (§4.4) — small but the user has already
   hit the predatory-false-positive case.

### 5b. Tier 2 — research upgrades, 1-2 days each

1. **Citation-context embeddings** (§2.1) — the biggest research idea
   we're missing. Mechanics match §5a so it slots in naturally after.
2. **Question-driven expansion** (§3.1) — new capability class, not
   an incremental upgrade. Start with a minimum viable version
   (LLM→sub-topics→per-sub-topic expand→gap check), polish over time.
3. **Negative-example learning** (§2.4) — wires the HITL loop into
   the ranker. Unblocks the LambdaMART roadmap eventually.
4. **Author oeuvre completion** (§2.6) — new seed source alongside
   outbound-ref crawl.

### 5c. Tier 3 — evaluate first, then decide

1. **SPECTER2 rerank revisit** (§2.2) — contingent on 54.6.69
   retrieval-bench delta. If SPECTER2 > 0.60 MRR@10, ship; else re-park.
2. **Inbound "cites me" crawl** (§2.5) — complementary to §2.6;
   easier to ship but less unique.
3. **Preprint ↔ journal reconciliation** (§1.3) — moderate effort,
   modest payoff until the corpus has many preprint/journal pairs.

### 5d. Tier 4 — infrastructure + observability

1. **Per-paper provenance** (§4.2) — pays off repeatedly (why-here
   tooltip, ablation analyses, expand debugging).
2. **Resume / checkpointing for expand** (§4.1) — worth it after §3.1
   multiplies the rounds-per-run count.
3. **Corpus-drift detection** (§4.3) — low effort, unlocks a "how has
   the corpus evolved?" retrospective.

---

## Anti-patterns / rejected ideas

Not worth pursuing (documenting so we don't relitigate):

- **Scite.ai paid API** — redundant with Semantic Scholar's free
  `isInfluential` + `intents`. Documented in original EXPAND_RESEARCH.md.
- **Web of Science / Scopus** — paid, closed; no path to a reproducible
  pipeline.
- **Full SPECTER2 model (non-adapter)** — too big for the 3090 alongside
  the writer. Adapter variant only; see §2.2 trigger.
- **Train LambdaMART now** — needs ≥500 labeled positives. Same
  conclusion as 2026-04-14. Reopen when the negative-example learning
  HITL (§2.4) has accumulated enough data — the user has full control
  over the pace of label accumulation, so it's a matter of use, not
  research.
- **LLM-as-sole-ranker** — no "delete the ranker, LLM decides." The
  ranker provides provenance and reproducibility; the LLM layer
  augments it (§3.x) but doesn't replace it.
- **Dense X / Propositional retrieval** for expand — wrong layer;
  expand is about finding candidate PAPERS, not chunks. The
  proposition-embedding work belongs to `hybrid_search`, where it's
  already in the RESEARCH.md rejected list.

---

## Sources consulted (2024-2026)

- **Cohan et al. 2024** — *SciRepEval: A Multi-Format Benchmark for Scientific Document Representations*, EMNLP. Citation-context embeddings (§2.1).
- **Sebok et al. 2024** — *OpenScholar: Synthesizing Scientific Literature with Retrieval-Augmented Language Models*, Ai2 preprint. End-to-end agentic literature synthesis; the architecture we're inching toward (§3.1).
- **Aha-Skarlatidis et al. 2024** — *PaperQA2: Agentic Retrieval Over Scientific Literature*, Future House preprint. Self-critique + citation verification; complements our existing `book review` / `verify-draft` flows.
- **Carbonell & Goldstein 1998** — *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.* SIGIR. Still the reference for §2.3.
- **Kulesza & Taskar 2012** — *Determinantal Point Processes for Machine Learning.* Foundations and Trends in ML. Alternative diversity method.
- **SPECTER2 Adapters 2025 release** (Cohan et al., Ai2 model hub) — climate-science adapter at 110 MB. Motivates §2.2 revisit.
- **Bornmann & Daniel 2010** — *Citation Speed as a Measure to Predict the Attention an Article Receives.* Journal of Informetrics. Velocity signal, already in Phase 49; referenced here for citation-intent filtering context.

Referenced-but-already-implemented in Phase 49: Cormack/Clarke/Buettcher (RRF), Valenzuela/Ha/Etzioni (citation intents), Connected Papers, Inciteful.
