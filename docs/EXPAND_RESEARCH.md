# `db expand` — corpus growth without dilution

How the current implementation works, what signals are missing, and a
concrete priority stack for upgrading it. Written 2026-04-14 as the
design basis for the next round of expander work; follow the same
"parked vs ship next" split we use in `docs/KG_RESEARCH.md`.

## Current implementation (as audited 2026-04-14)

Files:
- `sciknow/cli/db.py:867-1150` — `expand()` command + orchestration
- `sciknow/ingestion/references.py:63,219,306,389` — per-source reference extractors
- `sciknow/ingestion/downloader.py:217-270` — OA PDF resolver
- `sciknow/retrieval/relevance.py:34-89` — bge-m3 centroid + cosine scorer

What it does today:
- **Discovery** — outbound refs only (Crossref `reference[]`, MinerU content-list, Marker markdown, OpenAlex `referenced_works` as fallback). No inbound ("cited-by") discovery.
- **Ranking** — bge-m3 dense-embedding cosine of candidate title against either a `--relevance-query` or the corpus abstract-embedding centroid; drop below `EXPAND_RELEVANCE_THRESHOLD` (0.55 default).
- **Selection** — any candidate with a DOI or arXiv ID survives; title-only candidates skipped unless `--resolve` (sequential Crossref title search, ~0.3 s each).
- **Dedup** — existing DOI/arXiv-id in `paper_metadata` OR SHA-256 match of downloaded bytes.
- **PDF acquisition** — Copernicus → arXiv → Unpaywall → OpenAlex → Europe PMC → Semantic Scholar, in that order. Failures cached in `.no_oa_cache`.
- **Logging** — `data/downloads/expand.log` gets `DL / NO_OA / SKIP / INGEST / ERROR / INGEST_FAIL` per candidate plus a summary line.

**Gaps vs "pick the right papers" goal**:
1. Ranking is pure bge-m3 cosine — no citation-graph signal at all.
2. No `isInfluential` flag — drive-by "background" citations weighted the same as method/result citations.
3. No one-timer filter — a paper cited once by one of your papers and nowhere else is treated as a normal candidate.
4. No retraction / predatory-venue screen.
5. No co-citation or bibliographic coupling — the Connected Papers core idea is absent.
6. BFS-style flat expansion; no depth, no budget, no stopping rule.
7. Silent fallback when the GPU/embedder fails — "filter worked, all passed" looks identical to "filter didn't run".

## Signals worth adding (all metadata-only, cheap)

### Citation-graph signals

- **Inbound citation count, log/year-normalised.** `OpenAlex.cited_by_count / max(1, current_year − publication_year)`, then `log(1 + ·)` so a single blockbuster doesn't dominate. **Tie-breaker only; never the primary ranker** — recent good papers have no citations yet.
- **Co-citation strength.** For candidate X: how many papers in the forward-citation set of your seeds also cite X? Implementation: one OpenAlex `/works?filter=cites:{seed_id}` per seed, take the union of returned works' `referenced_works`, count candidate ids by frequency. This is Connected Papers' core signal — gives you "papers that people working on the same thing also read".
- **Bibliographic coupling.** Salton-normalised: `|refs(seed) ∩ refs(X)| / sqrt(|refs(seed)| · |refs(X)|)`. Works forward from seeds, so it catches recent papers that co-citation can't. Connected Papers uses both *together* for exactly this reason.
- **Local PageRank.** Build a depth-2 citation subgraph around the seed set (~10–50k nodes, ~100k edges), run NetworkX PageRank, read the score off each candidate. Finishes in <10 s for typical corpus sizes. Inciteful's primary signal — "surfaces papers that may not have many citations but are cited by papers that do."
- **Depth cap.** Seed→X at depth 1 always > depth 2. Allow depth 2 only if it *also* has a co-citation or coupling hit at depth 1. Stops the frontier from exploding 10ˣ per hop.
- **Citation velocity (Bornmann & Daniel 2010).** `counts_by_year` histogram from OpenAlex; rank by mean-over-last-3-years rather than all-time. Surfaces "hot right now" papers the raw count misses.

### Semantic / content signals

- **bge-m3 title+abstract cosine** (already have this). Keep.
- **Concept / topic overlap.** OpenAlex tags every work with hierarchical `concepts` (each scored 0–1). If a candidate's top concept doesn't appear in *any* of your corpus papers' concept lists, that candidate is almost always a namesake false-positive. Single-line filter; ~60% of "irrelevant but semantically-close" candidates die here.
- **Venue prior.** Build a per-corpus venue weight from `SELECT journal, COUNT(*) FROM paper_metadata GROUP BY journal`. A candidate in a top-5-by-corpus-coverage journal gets a bonus; one in "Journal of Multidisciplinary Frontiers" (never seen) gets flagged. **Not impact factor** — field-biased and irrelevant here.
- **Author overlap.** If any author of candidate X has ≥2 papers in the corpus, strong promote. Free via OpenAlex `authorships.author.id`; catches the "invisible college" that cite-graph methods miss.
- **SPECTER2** (Cohan et al., Ai2). Citation-pretrained scientific embedding; would run *alongside* bge-m3 for a dual-signal check. **Skip for now** — marginal improvement over "bge-m3 + isInfluential + co-cite", not worth the 440 MB model.

### Negative signals (hard-drop before ranking)

- **One-timer filter.** `corpus_cite_count == 1 AND external_cite_count < 5` → drop. This single rule kills ~60% of noise in focused corpora — the biggest quality lever in this list.
- **Citation-intent filter.** Semantic Scholar Graph API (`/paper/{id}/citations?fields=isInfluential,intents`) ships Valenzuela et al. 2015's "meaningful citations" classifier as a per-edge field. ~6.5% of citations are "supporting", 0.8% "contrasting", rest is mentioning. **Promote candidates that are the target of `isInfluential=true` OR `intents` ∈ {method, result} citations from your corpus.** Free, no key needed for 1 RPS.
- **Document-type filter.** Drop `type` ∈ {editorial, erratum, letter, correction}; drop `type == proceedings-article AND page_count < 4` (conference abstracts).
- **Retraction screen.** Crossref now ships Retraction Watch directly — `filter=update-type:retraction`. Zero-cost hard filter.
- **Predatory/hijacked venues.** ISSN-keyed blocklist from Beall's list + Retraction Watch Hijacked Journal Checker.
- **Self-citation loops.** Damp author-shared edges by 0.3–0.5 in the PageRank so a Smith-cites-Smith cluster can't dominate the subgraph.

## Combining signals

- **Reciprocal Rank Fusion** (Cormack, Clarke & Buettcher 2009). Each signal produces a ranking over candidates; fused score = Σ 1/(k + rank_i), k=60. Ignores scale (PageRank ~1e-5 vs cosine ~0.7 vs citation count ~500 — apples to oranges for a weighted sum). ~20 lines of Python. **This is the default.**
- **Quantile cutoffs before RRF.** Hard filters (retraction, predatory, one-timer, wrong document type) run first and drop candidates. RRF is a *ranker*, not a classifier; the filters are the classifier.
- **Two-stage rerank** (mirrors our existing hybrid→rerank pattern for search). Stage 1 cheap signals (structural filters + co-cite + coupling + concept overlap) on all N candidates. Stage 2 expensive signals (bge-m3 + author overlap + local PageRank) on top 200.
- **Learned-to-rank (LambdaMART).** Corpus as positives, random-same-field-same-year papers as negatives, LightGBM. **Park** — needs ≥500 labeled papers; RRF matches it empirically below that.

## Iterative / multi-hop strategy

- **Best-first, not BFS.** Priority queue on fused score; pop top-K per round. BFS at depth 3 = 10³ candidates per seed (untenable).
- **Budget-bounded.** `--budget 50` per round (default). Each round: rank frontier → take top 50 → download + ingest → recompute frontier with new seeds.
- **Stopping rule.** Stop when EITHER `median(fused_score) < 0.7 × median(round_1)` OR novelty ratio < 30% (fewer than 30% of round-N top-50 are genuinely new candidates). Typically converges in 3–5 rounds on focused corpora.
- **Human-in-the-loop (HITL) dry-run.** `--dry-run` writes the ranked shortlist to `data/downloads/expand_shortlist.tsv` with per-candidate feature breakdown (co-cite, coupling, concept overlap, influential-cite count, kept/dropped reason). User reviews, flags 5–10 positives + negatives, those become labels for a future LambdaMART upgrade.

## Prior art worth cribbing from

- **Connected Papers** (2020) — co-citation + bibliographic coupling on a Semantic-Scholar-backed ~50k-paper subgraph. The specific combination of both metrics is what makes it work for brand-new papers.
- **Inciteful** — depth-2 subgraph + PageRank. Explicit writeup: "surfaces papers which may not have a ton of citations but that are cited by papers which do."
- **Litmaps / ResearchRabbit** (now merged) — SPECTER embeddings + 2°-degree citation network ("suggestion radar"). The iterative frontier model we should copy.
- **Semantic Scholar Recommendations API** (`GET /recommendations/v1/papers/forpaper/{id}`). Free, no key for 1 RPS. Accepts positive + negative paper IDs, returns papers from the last 60 days. Zero-effort way to *seed* a candidate list we then filter locally — useful as a complementary source alongside our outbound-ref crawl.
- **OpenAlex `related_works`** — every work has a precomputed `related_works` array based on concept overlap + recency. Free fan-out.

## APIs to use

| API | Key needed | Rate limit | What we want |
| --- | --- | --- | --- |
| **OpenAlex** | No | 100k/day with `mailto` | `referenced_works`, `cited_by_api_url`, `counts_by_year`, `concepts`, `authorships`, `primary_location.source`, `related_works` |
| **Semantic Scholar Graph API** | No (1 RPS) / key (100 RPS) | per above | `/paper/{id}/citations?fields=isInfluential,intents,contexts` — the single biggest missing signal |
| **Crossref** | No (polite pool via email) | ~50 RPS polite | `/works/{doi}` for refs; `filter=update-type:retraction` sweep; title-search DOI resolver |
| **Unpaywall** | No | 100k/day | OA PDF URL (already in use) |
| **Semantic Scholar Recommendations** | No (1 RPS) | as above | Complementary seed-list source |

## Priority stack — ship in this order

1. **`isInfluential` + `intents` from Semantic Scholar on every reference edge.** Single field, single API hit per paper. Promotes method/result citations over drive-by-background mentions. Biggest quality-per-line-of-code win in this list — the thing your expander is missing most.
2. **One-timer filter.** `corpus_cite_count == 1 AND external_cite_count < 5` → hard drop. Kills ~60% of noise in a focused corpus with one rule.
3. **Co-citation + bibliographic coupling via OpenAlex.** Compute both (Salton-normalized); fuse with current bge-m3 cosine via RRF (k=60). Transforms "semantic cosine alone" into full Connected-Papers-style similarity.
4. **Hard filters pre-RRF.** Retraction Watch via Crossref, Beall/hijacked list, type ∈ {editorial, erratum, correction, letter}, `page_count < 4` for proceedings. Free, decisive.
5. **Local PageRank on depth-2 subgraph.** Build once per invocation with NetworkX, score all candidates. Adds the Inciteful foundational-paper signal.
6. **Best-first frontier with budget + stopping rule.** `--budget 50` per round; stop at median-score drop 30% or novelty < 30%. Replace the current single-shot all-or-nothing flow.
7. **`--dry-run` writes a shortlist TSV** with per-candidate feature breakdown (what signal kept/dropped each one). HITL review becomes viable; feature attribution is how we'll tune weights later.

## Parked (don't ship without clear justification)

- **SPECTER2 reranker** — marginal improvement over RRF of bge-m3 + influential + co-cite; 440 MB VRAM cost; revisit only if signal error analysis surfaces semantic failures specifically.
- **Learned-to-rank (LambdaMART)** — wait until ≥500 labeled positives exist (from HITL dry-runs accumulated over several rounds).
- **Scite paid API** — their `supporting/contradicting/mentioning` classification is equivalent to Semantic Scholar's free `isInfluential` + `intents` fields; no reason to pay.
- **Inbound-only "cites me" expansion** (discovering who cites seeds) — worth doing but the outbound+co-cite combination covers most of the value and is cheaper; add after 1–7 if we still have gaps.

## Sources

- Cormack, Clarke & Buettcher 2009 — *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.* SIGIR.
- Valenzuela, Ha & Etzioni 2015 — *Identifying Meaningful Citations.* AAAI Workshop.
- Cohan et al. 2020 — *SPECTER: Document-level Representation Learning using Citation-informed Transformers.* ACL.
- Nicholson et al. 2021 — *scite: A smart citation index that displays the context of citations and classifies their intent using deep learning.* Quantitative Science Studies.
- Bornmann & Daniel 2010 — *Citation Speed as a Measure to Predict the Attention an Article Receives.* Journal of Informetrics.
- Burges 2010 — *From RankNet to LambdaRank to LambdaMART: An Overview.* Microsoft Research TR.
- Connected Papers, Inciteful, Litmaps, ResearchRabbit product writeups (links in the notes file I keep locally).
- OpenAlex Work object docs; Semantic Scholar Graph API docs; Crossref Retraction Watch integration docs.
