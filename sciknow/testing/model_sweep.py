"""Model sweep benchmark — which Ollama model is best at each sciknow LLM role?

This module exists because sciknow has three distinct LLM roles
(extract-kg JSON, wiki compile prose, book autowrite long-form) and
each has failed in its own way when we naively swap models. Rather
than debug-by-session, this harness produces a comparable table:
every candidate model × every task × a fixed input, with uniform
metrics so picks are data-driven.

Design choices
--------------

**Why not inside ``protocol.py``?** Those are pass/fail correctness
tests. A model sweep emits numbers that invite comparison — "model X
was 2× faster but produced 30% fewer triples". That's a metric, not
a test.

**Why hook into ``bench.py``?** It already has ``BenchMetric``/
``BenchResult``, JSONL persistence, a ``latest.json`` rollup, and
Rich rendering. Reinventing those inside a standalone script
defeats the existing "compare against last run" workflow.

**Why fixed input papers, not a sampled sweep?** Variance between
papers dominates variance between models, so comparing model A on
paper X against model B on paper Y is noise. Lock the papers so
every row of the table is apples-to-apples. Two papers are chosen
on purpose: one math-heavy pathological case (4092d6ad, known to
break structured-output on thinking models) and one clean
descriptive paper (631fd2ea). Expand CANDIDATE_PAPERS to taste.

**Why one shot per (model, task)?** LLM variance at temperature=0
is small on modern llama.cpp builds, and 3-shot × 6 models × 3
tasks × 90s = 50 minutes. Single-shot keeps the harness usable;
if a result looks fishy re-run with a different seed by changing
the tag.

**What NOT to measure here.** This module does not cover: end-to-end
pipeline wiring (that's SMOKE's job), correctness under code changes
(L1/L2), or resource metrics like VRAM (covered by `nvidia-smi`
externally). Stay focused on model-comparable output quality.

Runtime
-------

Full sweep: roughly ``#models × #tasks × 60-90s`` plus cold-load
overhead (~15s/model). With 6 models × 3 tasks that's ~20-25 min on
a 3090 after models are pulled. If you just want one task, edit
``SWEEP_BENCHES`` below.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Iterable

from sciknow.testing.bench import BenchMetric, skip

# ════════════════════════════════════════════════════════════════════════
# Configuration — edit these, not the bench functions
# ════════════════════════════════════════════════════════════════════════

# Candidate models. Order defines table column order. Add or remove
# as you pull/drop models; a model that isn't installed yields a
# skip metric rather than failing the whole sweep.
CANDIDATE_MODELS: list[str] = [
    # Current production picks (baselines to beat)
    "qwen2.5:32b-instruct-q4_K_M",       # extract-kg baseline
    "qwen3:30b-a3b",                     # compile baseline (thinking)
    "qwen3.5:27b",                       # autowrite baseline (thinking)

    # New candidates the user is pulling 2026-04-16
    "qwen3:30b-a3b-instruct-2507-q4_K_M",  # official non-thinking 30B-A3B
    "gemma3:27b-it-qat",                    # Google non-thinking, QAT
    "command-r:35b",                        # Cohere RAG specialist
]

# Fixed test paper prefixes (8-char document_id prefix). The first is
# math-heavy (LaTeX equations, known pathological for thinking models);
# the second is descriptive prose. Sweep runs against the FIRST one
# only to keep runtime sane — add more via --paper if you want.
CANDIDATE_PAPERS: list[str] = [
    "4092d6ad",  # Nature Controls the CO2 Increase II — math-heavy
    # "631fd2ea",  # Sun Reversed Decades-long Weakening Trend — descriptive
]

# Per-task generation budgets. Match production defaults so results
# reflect real-world use. If a thinking model burns all of these in
# <think>, that's a legitimate failure signal for the sweep.
BUDGETS = {
    "extract_kg":      {"num_ctx": 8192,  "num_predict": 2048, "temperature": 0.0},
    "compile_summary": {"num_ctx": 8192,  "num_predict": 1500, "temperature": 0.3},
    "write_section":   {"num_ctx": 4096,  "num_predict": 600,  "temperature": 0.4},
}

# Slug format used in the extract-kg prompt — lowercase, hyphenated,
# no spaces or title-case. Used for predicate/subject-shape scoring.
_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")
_SLUG       = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_THINK_TAG  = re.compile(r"<think>|</think>", re.IGNORECASE)


# ════════════════════════════════════════════════════════════════════════
# Helpers — DB fetch + uniform scoring
# ════════════════════════════════════════════════════════════════════════


@dataclass
class PaperCtx:
    doc_id: str
    title: str
    authors: str
    year: str
    keywords: str
    domains: str
    abstract: str
    sections_text: str
    source_text: str   # full body for verbatim-sentence grounding check


def _load_paper(prefix: str) -> PaperCtx:
    """Load title + metadata + section text for one test paper. Raises
    SkipBench if no match — lets the whole sweep skip cleanly when
    the user's corpus doesn't contain the pinned test paper."""
    from sqlalchemy import text
    from sciknow.storage.db import get_session

    with get_session() as s:
        row = s.execute(text("""
            SELECT d.id::text, pm.title, pm.authors, pm.year,
                   pm.keywords, pm.domains, pm.abstract
            FROM documents d JOIN paper_metadata pm ON pm.document_id = d.id
            WHERE d.id::text LIKE :p
              AND d.ingestion_status = 'complete'
            LIMIT 1
        """), {"p": f"{prefix}%"}).fetchone()
        if not row:
            skip(f"test paper {prefix} not in corpus — skipping sweep")
        doc_id, title, authors, year, keywords, domains, abstract = row

        secs = s.execute(text("""
            SELECT section_type, content FROM paper_sections
            WHERE document_id::text = :did
            ORDER BY section_index
        """), {"did": doc_id}).fetchall()

    sections_text = "\n\n".join(
        f"[{r[0]}]\n{(r[1] or '')[:3000]}" for r in secs
    )[:12000]
    source_text = "\n".join((r[1] or "") for r in secs)
    # Handle NULL authors / year safely
    authors_str = authors if isinstance(authors, str) else ", ".join(authors or [])
    kw_str  = keywords if isinstance(keywords, str) else ", ".join(keywords or [])
    dom_str = domains  if isinstance(domains, str)  else ", ".join(domains  or [])
    return PaperCtx(
        doc_id=doc_id, title=title or "Untitled",
        authors=authors_str or "Unknown",
        year=str(year or "n.d."),
        keywords=kw_str or "", domains=dom_str or "",
        abstract=abstract or "",
        sections_text=sections_text,
        source_text=source_text,
    )


def _installed_models() -> set[str]:
    """Names of models currently loaded into Ollama (not just pulled —
    any model that responds to /api/tags). Used to skip candidates
    the user hasn't pulled yet, rather than timing out on them."""
    try:
        import ollama
        client = ollama.Client()
        resp = client.list()
        return {m.get("name") or m.get("model") or "" for m in resp.get("models", [])}
    except Exception:
        return set()


def _call_model_raw(system: str, user: str, model: str, budget: dict) -> dict:
    """One chat call. Returns {content, thinking, elapsed_s, eval_count,
    done_reason, error}. We go through the raw ollama client instead of
    sciknow.rag.llm.complete so we can capture ``thinking`` separately
    — without that, a thinking-runaway is indistinguishable from a hang.
    """
    import ollama
    client = ollama.Client()
    t0 = time.monotonic()
    try:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options={
                "temperature": budget["temperature"],
                "num_ctx":     budget["num_ctx"],
                "num_predict": budget["num_predict"],
                "num_batch":   1024,
            },
            keep_alive=0,   # unload after so the next candidate gets fresh VRAM
        )
    except Exception as exc:
        return {"error": str(exc), "elapsed_s": time.monotonic() - t0,
                "content": "", "thinking": "", "eval_count": 0,
                "done_reason": "exception"}
    msg = resp.get("message") or {}
    return {
        "error": None,
        "elapsed_s": time.monotonic() - t0,
        "content":   msg.get("content", "") or "",
        "thinking":  msg.get("thinking", "") or "",
        "eval_count":  resp.get("eval_count", 0) or 0,
        "done_reason": resp.get("done_reason", ""),
    }


# ════════════════════════════════════════════════════════════════════════
# Task 1 — extract-kg (JSON extraction)
# ════════════════════════════════════════════════════════════════════════


def _score_extract_kg(result: dict, paper: PaperCtx) -> dict:
    """Convert one extract-kg model response into a metrics dict.
    Keys: content_chars, json_ok, shape, n_concepts, n_methods, n_datasets,
    n_triples, sent_present_pct, sent_verbatim_pct, pred_snake_pct,
    slug_ok_pct, thinking_chars."""
    from sciknow.core.wiki_ops import _find_json_block, _strip_thinking

    raw = result.get("content") or ""
    cleaned = _strip_thinking(raw).strip()
    out = {
        "content_chars":     len(raw),
        "thinking_chars":    len(result.get("thinking") or ""),
        "json_ok":           0,
        "shape":             "empty",
        "n_concepts":        0,
        "n_methods":         0,
        "n_datasets":        0,
        "n_triples":         0,
        "sent_present_pct":  0.0,
        "sent_verbatim_pct": 0.0,
        "pred_snake_pct":    0.0,
        "slug_ok_pct":       0.0,
    }
    if not cleaned:
        return out

    block = _find_json_block(cleaned)
    if not block:
        out["shape"] = "no-json"
        return out
    try:
        data = json.loads(block, strict=False)
    except Exception:
        out["shape"] = "bad-json"
        return out
    out["json_ok"] = 1

    concepts = data.get("concepts") or []
    methods  = data.get("methods")  or []
    datasets = data.get("datasets") or []
    triples  = data.get("triples")  or []
    all_ent  = concepts + methods + datasets

    n_str  = sum(1 for x in all_ent if isinstance(x, str))
    n_dict = sum(1 for x in all_ent if isinstance(x, dict))
    if all_ent == []:
        out["shape"] = "empty-entities"
    elif n_dict == 0 and n_str > 0:
        out["shape"] = "flat"
    elif n_dict > 0 and n_str == 0:
        out["shape"] = "dict"
    else:
        out["shape"] = "mixed"

    out["n_concepts"] = len(concepts)
    out["n_methods"]  = len(methods)
    out["n_datasets"] = len(datasets)
    out["n_triples"]  = len(triples)

    # Slug compliance on entity-name strings (dict entries don't count)
    slug_candidates = [x for x in all_ent if isinstance(x, str)]
    if slug_candidates:
        out["slug_ok_pct"] = 100.0 * sum(
            1 for x in slug_candidates if _SLUG.match(x)
        ) / len(slug_candidates)

    # Triple-level scoring. Skip non-dict triples (some models emit
    # tuples or strings under pressure; those count as zero).
    if triples:
        dict_triples = [t for t in triples if isinstance(t, dict)]
        if dict_triples:
            # snake_case predicate share
            preds = [
                (t.get("predicate") or "") if isinstance(t.get("predicate"), str)
                else ""
                for t in dict_triples
            ]
            out["pred_snake_pct"] = 100.0 * sum(
                1 for p in preds if _SNAKE_CASE.match(p)
            ) / len(dict_triples)

            # source_sentence present + verbatim
            sents = []
            for t in dict_triples:
                s = t.get("source_sentence")
                sents.append(s if isinstance(s, str) else "")
            present = sum(1 for s in sents if s.strip())
            out["sent_present_pct"] = 100.0 * present / len(dict_triples)

            # verbatim: check whether each non-empty sentence appears in
            # the source paper text. Normalize whitespace to forgive
            # minor quoting differences; anything below 60% is a strong
            # hallucination signal.
            src_norm = re.sub(r"\s+", " ", paper.source_text).lower()
            verbatim = 0
            for s in sents:
                s_norm = re.sub(r"\s+", " ", s).strip().lower()
                if len(s_norm) >= 20 and s_norm in src_norm:
                    verbatim += 1
            if present:
                out["sent_verbatim_pct"] = 100.0 * verbatim / present
    return out


def b_model_sweep_extract_kg() -> Iterable[BenchMetric]:
    """Per-model extract-kg JSON extraction quality + speed.

    Emits one group of metrics per candidate model. Missing models
    (not pulled locally) are emitted as a single skip metric.
    """
    from sciknow.rag import wiki_prompts

    paper = _load_paper(CANDIDATE_PAPERS[0])
    sys_p, usr_p = wiki_prompts.wiki_extract_entities(
        title=paper.title, authors=paper.authors, year=paper.year,
        keywords=paper.keywords, domains=paper.domains,
        abstract=paper.abstract, existing_slugs=[],
        slug=f"sweep-{paper.doc_id[:8]}", sections=paper.sections_text,
    )
    installed = _installed_models()

    for model in CANDIDATE_MODELS:
        if installed and model not in installed:
            yield BenchMetric(f"{model}::status", "not-installed", "",
                              note="pull with `ollama pull " + model + "`")
            continue
        resp = _call_model_raw(sys_p, usr_p, model, BUDGETS["extract_kg"])
        if resp.get("error"):
            yield BenchMetric(f"{model}::status", "error", "",
                              note=resp["error"][:80])
            yield BenchMetric(f"{model}::elapsed_s",
                              round(resp["elapsed_s"], 1), "s")
            continue

        scored = _score_extract_kg(resp, paper)
        yield BenchMetric(f"{model}::elapsed_s",
                          round(resp["elapsed_s"], 1), "s",
                          note=f"done={resp.get('done_reason','')}")
        yield BenchMetric(f"{model}::eval_count", resp["eval_count"], "tok")
        yield BenchMetric(f"{model}::content_chars",
                          scored["content_chars"], "chars")
        yield BenchMetric(f"{model}::thinking_chars",
                          scored["thinking_chars"], "chars",
                          note="high + content=0 ⇒ thinking runaway")
        yield BenchMetric(f"{model}::json_ok", scored["json_ok"], "bool")
        yield BenchMetric(f"{model}::shape", scored["shape"], "")
        yield BenchMetric(f"{model}::n_triples", scored["n_triples"], "triples")
        yield BenchMetric(f"{model}::n_entities",
                          scored["n_concepts"] + scored["n_methods"] + scored["n_datasets"],
                          "items",
                          note=f"c={scored['n_concepts']} m={scored['n_methods']} d={scored['n_datasets']}")
        yield BenchMetric(f"{model}::slug_ok_pct",
                          round(scored["slug_ok_pct"], 1), "%",
                          note="entity strings matching lowercase-hyphenated slug")
        yield BenchMetric(f"{model}::pred_snake_pct",
                          round(scored["pred_snake_pct"], 1), "%",
                          note="triple predicates in snake_case")
        yield BenchMetric(f"{model}::sent_present_pct",
                          round(scored["sent_present_pct"], 1), "%",
                          note="triples with non-empty source_sentence")
        yield BenchMetric(f"{model}::sent_verbatim_pct",
                          round(scored["sent_verbatim_pct"], 1), "%",
                          note="sentences that are verbatim in the paper (anti-hallucination)")


# ════════════════════════════════════════════════════════════════════════
# Task 2 — wiki compile (free-form scientific prose)
# ════════════════════════════════════════════════════════════════════════


def _score_compile(text: str, paper: PaperCtx) -> dict:
    # Word count: split on whitespace after stripping markdown headers
    body = re.sub(r"^#+ .*$", "", text, flags=re.MULTILINE)
    words = len(body.split())
    out = {
        "words":          words,
        "on_target":      1 if 200 <= words <= 800 else 0,
        "thinking_leaked": 1 if _THINK_TAG.search(text) else 0,
        "title_in_output": 1 if paper.title[:30].lower() in text.lower() else 0,
    }
    return out


def b_model_sweep_compile_summary() -> Iterable[BenchMetric]:
    """Per-model wiki paper-summary quality + speed.

    Uses ``wiki_paper_summary`` — the production prompt — against a
    fixed paper. Measures length hit (200-800 target), thinking-tag
    leakage, and whether the model grounds the summary on the paper
    (title must appear, or at minimum a recognizable name match)."""
    from sciknow.rag import wiki_prompts

    paper = _load_paper(CANDIDATE_PAPERS[0])
    # wiki_paper_summary expects journal/doi which aren't in PaperCtx;
    # pass empty strings — the summary quality doesn't hinge on them.
    sys_p, usr_p = wiki_prompts.wiki_paper_summary(
        title=paper.title, authors=paper.authors, year=paper.year,
        journal="", doi="",
        keywords=paper.keywords, domains=paper.domains,
        abstract=paper.abstract, sections=paper.sections_text,
        existing_slugs=[],
    )
    installed = _installed_models()

    for model in CANDIDATE_MODELS:
        if installed and model not in installed:
            yield BenchMetric(f"{model}::status", "not-installed", "")
            continue
        resp = _call_model_raw(sys_p, usr_p, model, BUDGETS["compile_summary"])
        if resp.get("error"):
            yield BenchMetric(f"{model}::status", "error", "",
                              note=resp["error"][:80])
            continue

        content = resp["content"]
        scored = _score_compile(content, paper)
        yield BenchMetric(f"{model}::elapsed_s",
                          round(resp["elapsed_s"], 1), "s",
                          note=f"done={resp.get('done_reason','')}")
        yield BenchMetric(f"{model}::eval_count", resp["eval_count"], "tok")
        yield BenchMetric(f"{model}::content_chars", len(content), "chars")
        yield BenchMetric(f"{model}::thinking_chars",
                          len(resp.get("thinking") or ""), "chars")
        yield BenchMetric(f"{model}::words", scored["words"], "words")
        yield BenchMetric(f"{model}::on_target", scored["on_target"], "bool",
                          note="1 if words ∈ [200, 800]")
        yield BenchMetric(f"{model}::thinking_leaked",
                          scored["thinking_leaked"], "bool",
                          note="<think> tag in final output (bad)")
        yield BenchMetric(f"{model}::title_in_output",
                          scored["title_in_output"], "bool",
                          note="cheap grounding proxy")


# ════════════════════════════════════════════════════════════════════════
# Task 3 — book autowrite (long-form grounded writing)
# ════════════════════════════════════════════════════════════════════════


_CITE_MARK = re.compile(r"\[\d+\]|\[@[\w-]+\]|\((?:[A-Z][a-z]+ (?:et al\.,? )?\d{4})\)")


def _score_write_section(text: str) -> dict:
    body = re.sub(r"^#+ .*$", "", text, flags=re.MULTILINE)
    words = len(body.split())
    cite_count = len(_CITE_MARK.findall(text))
    return {
        "words":           words,
        "on_target":       1 if 80 <= words <= 300 else 0,  # target was 150
        "thinking_leaked": 1 if _THINK_TAG.search(text) else 0,
        "cite_marks":      cite_count,
        "cites_per_100w":  round(100.0 * cite_count / max(words, 1), 2),
    }


def b_model_sweep_write_section() -> Iterable[BenchMetric]:
    """Per-model write_section_v2 quality + speed.

    Uses the same trivial context as the Phase 54.6.39 autowrite smoke
    test so this sweeps a realistic writer call without needing the
    full retrieval stack to be populated. target_words=150 is small on
    purpose — keeps runtime down and focuses the metric on prompt
    compliance rather than long-form fatigue."""
    from sciknow.rag import prompts
    from sciknow.retrieval.context_builder import SearchResult

    # Build two real ``SearchResult`` objects — write_section_v2 walks
    # ``r.title`` / ``r.year`` / ``r.content`` / ``r.authors`` etc.,
    # so dict literals crash in _norm_title and the context formatter.
    # Production feeds this prompt with SearchResult instances
    # (hydrated from SearchCandidate + full chunk content from PG),
    # so the sweep does the same to stay faithful.
    results = [
        SearchResult(
            rank=1, score=0.1, chunk_id="sweep-a", document_id="sweep-doc-a",
            section_type="results", section_title="Findings",
            content="Climate sensitivity remains debated. The IPCC reports a likely range of 2.5-4°C per doubling of CO2. Recent observational studies suggest the distribution may be skewed, with a long tail toward higher sensitivities.",
            title="Sample source A", year=2022,
            authors=[{"name": "A. Researcher"}], journal="Journal of Climate", doi=None,
        ),
        SearchResult(
            rank=2, score=0.09, chunk_id="sweep-b", document_id="sweep-doc-b",
            section_type="discussion", section_title="Discussion",
            content="Observational estimates of climate sensitivity tend toward the lower end of the likely range, though paleoclimate constraints push back against the lowest values.",
            title="Sample source B", year=2020,
            authors=[{"name": "B. Scientist"}], journal="Nature Climate", doi=None,
        ),
    ]
    sys_p, usr_p = prompts.write_section_v2(
        section="introduction",
        topic="Scientific uncertainty in climate sensitivity estimates",
        results=results,
        book_plan="A book on climate sensitivity uncertainty.",
        prior_summaries=None, paragraph_plan=None,
        target_words=150, section_plan=None,
        lessons=None, style_fingerprint_block=None,
    )
    installed = _installed_models()

    for model in CANDIDATE_MODELS:
        if installed and model not in installed:
            yield BenchMetric(f"{model}::status", "not-installed", "")
            continue
        resp = _call_model_raw(sys_p, usr_p, model, BUDGETS["write_section"])
        if resp.get("error"):
            yield BenchMetric(f"{model}::status", "error", "",
                              note=resp["error"][:80])
            continue

        content = resp["content"]
        scored = _score_write_section(content)
        yield BenchMetric(f"{model}::elapsed_s",
                          round(resp["elapsed_s"], 1), "s",
                          note=f"done={resp.get('done_reason','')}")
        yield BenchMetric(f"{model}::eval_count", resp["eval_count"], "tok")
        yield BenchMetric(f"{model}::content_chars", len(content), "chars")
        yield BenchMetric(f"{model}::words", scored["words"], "words")
        yield BenchMetric(f"{model}::on_target", scored["on_target"], "bool",
                          note="1 if words ∈ [80, 300] for target=150")
        yield BenchMetric(f"{model}::cite_marks", scored["cite_marks"], "refs",
                          note="[N] / (Author YYYY) / [@key] style markers")
        yield BenchMetric(f"{model}::cites_per_100w",
                          scored["cites_per_100w"], "/100w")
        yield BenchMetric(f"{model}::thinking_leaked",
                          scored["thinking_leaked"], "bool")


# ════════════════════════════════════════════════════════════════════════
# Registry — imported by bench.py LAYERS
# ════════════════════════════════════════════════════════════════════════


SWEEP_BENCHES: list[tuple[str, callable]] = [
    ("sweep", b_model_sweep_extract_kg),
    ("sweep", b_model_sweep_compile_summary),
    ("sweep", b_model_sweep_write_section),
]
