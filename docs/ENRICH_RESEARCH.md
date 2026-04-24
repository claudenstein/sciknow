# `db enrich` — Research Memo on DOI/ISBN Recovery Improvements

**Status**: research only, not yet implemented.
**Generated**: 2026-04-24.
**Scope**: raising the match rate for the ~40-50% of publications that still
lack a DOI or ISBN after the current 4-layer cascade (Crossref title →
OpenAlex title → arXiv ID → LLM over first 3k chars).

Current pipeline lives in `sciknow/cli/db.py::enrich` (line 5104) and
`sciknow/ingestion/metadata.py`. This memo evaluates additional sources,
smarter matching, and neglected PDF signals, then recommends a
one-evening highest-ROI patch.

## 1. Article-level metadata APIs (beyond Crossref + OpenAlex + arXiv)

| Source | What it adds | Auth | Rate limit | Verdict |
|---|---|---|---|---|
| **Semantic Scholar Graph API** — `/graph/v1/paper/search/match` takes a title, returns the single best-match paper with `externalIds.DOI`, `arXivId`, `PMID`, `MAG`. Covers ~200M including grey literature + preprints OpenAlex misses. | Free API key via email form; unauth still works. | 1 RPS with key; unauth shares a global pool — 429s in bulk are common. Always request the key. | **Top-tier add.** Purpose-built for this exact task. |
| **Europe PMC** — PMID/DOI/PMCID conversion + title search over 33M life-sciences records (PubMed, Agricola, patents). Fills NIH-deposited works. | None | ~10 RPS polite | **High value for climate↔health overlap.** |
| **NCBI PMC ID Converter** — batch convert PMID/PMCID/DOI, 200 per call. | `tool=`+`email=` params | 3 RPS unauth, 10 RPS with API key. | Pair with PubMed ESearch (title → PMID) then this for PMID → DOI. |
| **DataCite REST** — 38M dataset/preprint/software DOIs. Relevant climate prefixes: `10.5281` (Zenodo), `10.5067` (NASA), `10.1594` (PANGAEA). | None for read | Low-hundreds/sec tiers; polite use is fine. | **Ship it for climate corpora** — many DataCite-only DOIs Crossref ignores. |
| **OpenAIRE Graph API v2** | Aggregated EU repos: theses, reports, project deliverables. | None for search; OAuth only for quotas. | Anonymous OK for moderate use. | Often duplicates OpenAlex; worthwhile for EU-funded grey lit. |
| **Fatcat (Internet Archive)** — lookup by DOI/arXiv/PMID/PMCID/ISBN plus `refcat` reference-matching for partial citation strings. | None | Undocumented, generous | Complements Semantic Scholar. |
| **CORE API v3** | 290M+ OA papers including repository-only preprints. | Free registered key | ~10 tokens/min free tier. | Low priority — coverage mostly ⊂ OpenAlex + S2. |
| **DBLP** | Only valuable for CS/ML conference papers without DOIs (many IEEE workshops, CEUR-WS). | None | Polite use | Only if CS content is non-trivial. |
| **NASA ADS** | Astrophysics/earth-sci gold standard; bibcode→DOI is often the only way to link pre-2000 or non-Crossref'd astro papers. | Free token | 5000 calls/day default | Only if corpus touches astro/geo. |

## 2. Book / ISBN sources

| Source | Notes |
|---|---|
| **OpenLibrary Search** `/search.json?title=...&author=...` | 1 RPS unauth, 3 RPS with identifying `User-Agent: sciknow/x.y (email)`. Returns ISBN-10/13, OCLC, LCCN, subjects. **Start here.** |
| **Google Books** `/volumes?q=intitle:...+inauthor:...` | 100 req/day free tier unless a GCP key (then ~1000/day). ISBN in `industryIdentifiers`. Blocker for bulk. |
| **WorldCat xISBN** | **Dead.** Legacy xID deprecated; WorldCat Search API v1 retired 2024-12-31. v2 needs institutional OCLC auth. |
| **ISBNdb** | Paid only — $14.99/mo minimum. Skip without budget. |
| **Library of Congress SRU** `http://lx2.loc.gov:210/LCDB?...` | Free, no auth. CQL indexes: `bath.isbn`, `dc.title`, `dc.author`. ~10 RPM polite ceiling. Returns MARCXML with LCCN + LC subject headings — most authoritative free subject classification. |
| **LibraryThing ThingISBN** | Free; returns related/alternate ISBNs for a given ISBN — canonicalising editions, not first-hit recovery. |

Trust ranking for ISBN-13 correctness: **OpenLibrary ≈ LoC > Google Books** (Google occasionally returns editions with fabricated or supplier-internal ISBNs).
For subjects: **LoC (LCSH) > OpenLibrary > Google**.

## 3. Fuzzy-matching improvements

The current single-threshold-on-title-similarity has two weaknesses: brittleness under punctuation/LaTeX, and no corroborating signal when titles are generic ("Deep Learning", "A Review of X"). Layered recommendation:

1. **Normalise first** — NFKD casefold, strip LaTeX, collapse whitespace, drop trailing `.` and subtitle after `:`.
2. **rapidfuzz `token_set_ratio`** as primary gate (reorder/dup/punct tolerant). Accept ≥ 90; corroborate 80–89.
3. **Author-surname intersection** — cheapest false-positive killer. Require ≥ 1 surname overlap (casefolded, unicode-normalised). A single surname agreement + title ≥ 85 beats title-only ≥ 95 on real corpora.
4. **Year agreement ±1** — journal version may lag preprint by one year.
5. **3-gram character shingle Jaccard** — tiebreaker between multiple candidates with similar token_set scores (resolves "Deep Learning" vs "Deep Learning for X" collisions).
6. **First-3-page PDF text regex** for `\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b` — catches DOIs printed in footers but absent from XMP. Validate via Crossref `/works/{doi}` and require returned title token_set ≥ 80.

## 4. Signals the current pipeline likely misses

- **PDF XMP packet (not just Info dict)** — PyMuPDF's `doc.metadata` reads the Info dict only. The XMP packet (`doc.xref_xml_metadata()` → `doc.xref_stream(xref)`) contains publisher-stamped `prism:doi`, `prism:url`, `dc:identifier`. Parse with `lxml`. High-signal, near-zero false positives for Elsevier/Wiley/Springer/IOP PDFs.
- **Crossref prefix validation** — once any DOI string is found in the PDF, validate it resolves via `/works/{doi}` AND verify the prefix is registered via `/prefixes/{prefix}`. Rejects OCR-mangled / typo DOIs (e.g. `10.10116/…`).
- **Unpaywall `/v2/{doi}`** — not for *finding*, for *disambiguating*. Fetch the OA URL, grep `<meta name="citation_doi">` for a ground-truth echo. 100k/day free, `?email=` param.
- **Combined Crossref `query.author` + `query.title`** — the current pipeline probably only sends one; combining narrows the candidate pool at no quota cost.
- **arXiv OAI-PMH title search** — the current arXiv layer only fires when an arXiv ID already exists. Adding `export.arxiv.org/api/query?search_query=ti:"..."+AND+au:"..."` catches preprints where the PDF lacks the arXiv-ID stamp.

## 5. Recommended highest-ROI one-evening patch (ranked)

1. **Semantic Scholar `/graph/v1/paper/search/match`**
   `https://api.semanticscholar.org/graph/v1/paper/search/match?query=<title>&fields=externalIds,title,authors,year`
   Single endpoint designed for this problem, returns DOI+arXiv+PMID+MAG. Request a free API key (email form) for the 1 RPS guarantee.

2. **PDF XMP + footer-regex DOI scrape**
   Extract XMP via PyMuPDF, parse with `lxml`, look for `prism:doi`. On miss, regex the first 3 pages of extracted text for `10.\d{4,9}/\S+` and validate each candidate via Crossref `/works/{doi}` (50 RPS polite with `mailto=`). No external quota — fixes "the DOI was printed right there" misses.

3. **OpenLibrary Search for books**
   `https://openlibrary.org/search.json?title=<t>&author=<a>&fields=key,isbn,lccn,subject`
   Free, no key, 3 RPS with an identifying `User-Agent: sciknow/x.y (email)`. Returns edition ISBNs + LC subjects. Trustworthy.

**Bonus if climate content dominates**: DataCite as a 4th layer —
`https://api.datacite.org/dois?query=titles.title:"<t>"&page[size]=5`.
No auth. Covers PANGAEA and Zenodo DOIs Crossref/OpenAlex often don't index.

## Proposed ordering for a new cascade

```
PDF XMP → first-page regex+Crossref-validate → Crossref title(+author)
  → Semantic Scholar /match → OpenAlex → Europe PMC (life-sci)
  → DataCite (climate) → arXiv title-search → LLM fallback
```

## Books (separate cascade)

```
PDF XMP (dc:identifier) → OpenLibrary → LoC SRU → Google Books (quota-limited)
```

## Key files to touch when this work is picked up

- `sciknow/ingestion/metadata.py` — add new layer functions beside `search_crossref_by_title` / `search_openalex_by_title`.
- `sciknow/cli/db.py::enrich` (line 5104) — extend the `_lookup` cascade.
- `sciknow/config.py` — settings for `SEMANTIC_SCHOLAR_API_KEY`, `UNPAYWALL_EMAIL`, new rate-limit knobs.
- New module `sciknow/ingestion/xmp_doi.py` for the PDF XMP + footer-regex layer.

## Sources

- [Semantic Scholar Graph API docs](https://api.semanticscholar.org/api-docs/)
- [Semantic Scholar API release notes](https://github.com/allenai/s2-folks/blob/main/API_RELEASE_NOTES.md)
- [Europe PMC RESTful Web Service](https://europepmc.org/RestfulWebService)
- [NCBI PMC ID Converter](https://pmc.ncbi.nlm.nih.gov/tools/id-converter-api/)
- [DataCite REST API](https://support.datacite.org/docs/api)
- [DataCite rate limits](https://support.datacite.org/docs/rate-limit)
- [OpenAIRE Graph API](https://graph.openaire.eu/docs/apis/graph-api/)
- [Fatcat Cookbook](https://guide.fatcat.wiki/cookbook.html)
- [CORE API documentation](https://core.ac.uk/documentation/api)
- [DBLP search API FAQ](https://dblp.org/faq/How+to+use+the+dblp+search+API.html)
- [NASA ADS via astroquery](https://astroquery.readthedocs.io/en/stable/nasa_ads/nasa_ads.html)
- [OpenLibrary Search API](https://openlibrary.org/dev/docs/api/search)
- [OpenLibrary rate limits issue](https://github.com/internetarchive/openlibrary/issues/10585)
- [Google Books API v1](https://developers.google.com/books/docs/v1/using)
- [WorldCat Search API transition](https://www.oclc.org/developer/support/worldcat-search-transition.en.html)
- [ISBNdb pricing](https://isbndb.com/isbn-database)
- [Library of Congress APIs](https://www.loc.gov/apis/)
- [Unpaywall API](https://unpaywall.org/products/api)
- [PyMuPDF XMP metadata guide](https://medium.com/@pymupdf/mastering-metadata-and-table-of-contents-manipulation-with-pymupdf-b9099b64b17b)
- [Crossref getPrefixPublisher API](https://crossref.gitlab.io/knowledge_base/docs/services/get-prefix-publisher/)
- [RapidFuzz](https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html)
