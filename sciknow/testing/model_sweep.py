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
    # Phase 54.6.53 — full local-install sweep for the 2026-04-17 bench.
    # Listed in name-sort order; not priority. Any model not in
    # ``ollama list`` is gracefully skipped with a "not-installed"
    # status metric rather than failing the whole sweep.
    "gemma3:27b-it-qat",                      # current book-writing baseline
    "gemma4:26b-a4b-it-q4_K_M",
    "gemma4:31b",                             # may fail load on Ollama < 0.20.0
    "gemopus4:26b-a4b-q4_K_M",                # gemma4 MoE community tune
    "mn-darkest-universe:29b-q4_K_M",         # mistral-nemo variant
    "nemotron-cascade-2:30b",                 # NVIDIA
    "ornstein3.6:35b-a3b-q4_K_S",             # qwen35moe community tune
    "qwen2.5:32b-instruct-q4_K_M",            # former extract-kg baseline
    "qwen3:30b-a3b-instruct-2507-q4_K_M",     # current compile + extract-kg
    "qwen3.5:27b",                            # former book baseline (thinking)
    "qwen3.6:27b-dense",                      # 54.6.240 new dense candidate
    "qwen3.6:35b-a3b-q4_K_M",                 # thinking variant
    "qwen3.6:35b-a3b-ud-q4_K_S",              # unsloth UD quant (thinking)
    "supergemma4:26b-uncensored-q4_K_M",      # gemma4 dense community tune
    "supergemma4:31b-abliterated-q4_K_M",     # broken output (for reference)
]

# Fixed test paper prefixes (8-char document_id prefix). Replaced
# post-54.6.210 re-ingest (DB reset invalidated the old doc_ids).
# Sweep runs against the FIRST one only to keep runtime sane.
CANDIDATE_PAPERS: list[str] = [
    # Phase 54.6.240 — refreshed for the VLM-Pro-era corpus.
    "2b7a96c0",  # historic datasets open solar flux — descriptive prose, 75 chunks
    "0a325b78",  # Carrington Events — rich abstract + body
    "b76e2c8a",  # ION-CAGE — numerical model, equations
]

# ════════════════════════════════════════════════════════════════════════
# Per-task budgets — Phase 54.6.85 methodology overhaul
# ════════════════════════════════════════════════════════════════════════
#
# Pre-54.6.85, every model ran at ``temperature=0`` and ``num_predict``
# between 600 and 2048. This was methodologically unfair to Qwen's
# thinking variants:
#
#   - Qwen's own docs recommend ``num_predict`` **16k–32k** for
#     thinking mode; CoT commonly spans 4-16k tokens before a single
#     answer token emerges. At 2048 the CoT is truncated mid-thought,
#     producing 0-word outputs that misleadingly marked the model as
#     "broken".
#   - Qwen explicitly discourages ``temperature=0`` — causes repetition
#     loops, especially with thinking on. Recommended: 0.7 non-thinking,
#     1.0 thinking.
#   - Ollama has a native ``think`` boolean that can force-disable
#     thinking on hybrid (3.5 / 3.6) models. Pre-54.6.85 the bench
#     never used it.
#
# Fix: budgets are now a BASE that gets scaled by the model's profile.
# Thinking models get 8× the predict budget and Qwen-recommended
# sampling; non-thinking stay near the previous numbers. Each model
# also gets its sampling params (temp, top_p, top_k) from
# ``profile_for()`` below.
#
# Sources for the fix (see docs/PHASE_LOG.md 54.6.85 entry):
#   - Qwen3-30B-A3B-Instruct-2507 HF card: temp 0.7, top_p 0.8, top_k 20
#   - Qwen3.5 / 3.6 model cards: temp 1.0 thinking / 0.6 coding,
#     top_p 0.95, top_k 20
#   - Ollama thinking docs: native ``think`` boolean flag
#   - Qwen3 blog: CoT budget recommendations
BUDGETS = {
    "extract_kg":      {"num_ctx": 8192,  "num_predict": 2048, "temperature": 0.2},
    "compile_summary": {"num_ctx": 8192,  "num_predict": 1500, "temperature": 0.5},
    "write_section":   {"num_ctx": 4096,  "num_predict": 800,  "temperature": 0.6},
}


# Thinking-mode multiplier for ``num_predict``. 8× covers Qwen's
# recommended 16k floor when the base budget is 2048.
THINKING_PREDICT_MULT = 8
# Minimum budget floor for thinking models — regardless of base.
# 12k covers typical CoT (4-10k) + an answer of a few hundred words.
THINKING_MIN_PREDICT = 12288


@dataclass
class ModelProfile:
    """How to invoke a given model fairly.

    ``thinks_by_default`` — True for hybrid Qwen3.5/3.6 that emit
    ``<think>…</think>`` unless told otherwise; drives the budget
    multiplier and sampling recommendations.
    ``can_disable_thinking`` — True when the Ollama ``think`` flag
    works (3.5+ support it, thinking-only 2507 variants don't).
    ``temperature`` / ``top_p`` / ``top_k`` — thinking-mode sampling
    per model-family doc.
    ``nt_*`` — instruct (non-thinking) sampling. When the bench
    forces no-thinking on a hybrid model, use these instead.
    Qwen3.6 README prescribes temp=0.7, top_p=0.80, top_k=20,
    min_p=0.0, presence_penalty=1.5 for instruct mode — radically
    different from the thinking-mode recipe — and using the
    thinking-mode recipe with thinking off is what made the model
    look worse than qwen3:30b-a3b-instruct in 54.6.240-242.
    """
    thinks_by_default: bool
    can_disable_thinking: bool
    # thinking-mode sampling
    temperature: float
    top_p: float
    top_k: int
    # instruct-mode (non-thinking) sampling — per Qwen3.6 README
    nt_temperature: float = 0.7
    nt_top_p: float = 0.8
    nt_top_k: int = 20
    nt_presence_penalty: float = 0.0
    nt_min_p: float = 0.0


def profile_for(model: str) -> ModelProfile:
    """Heuristically infer a model's profile from its tag. Hybrid
    Qwen 3.5/3.6 get thinking treatment; everything else stays at
    non-thinking defaults.

    See docs/PHASE_LOG.md 54.6.85 for the source-of-truth per family.
    Add new entries here when new candidates join ``CANDIDATE_MODELS``.
    """
    m = (model or "").lower()
    # Qwen 3.5 / 3.6 — hybrid thinking, soft-switchable. Instruct-mode
    # sampling per Qwen3.6 README: temp=0.7, top_p=0.80, top_k=20,
    # min_p=0.0, presence_penalty=1.5 (see
    # huggingface/unsloth-Qwen3.6-27B-GGUF/README.md "Best Practices").
    if "qwen3.5" in m or "qwen3.6" in m or "ornstein3.6" in m:
        return ModelProfile(
            thinks_by_default=True, can_disable_thinking=True,
            temperature=1.0, top_p=0.95, top_k=20,
            nt_temperature=0.7, nt_top_p=0.80, nt_top_k=20,
            nt_presence_penalty=1.5, nt_min_p=0.0,
        )
    # Qwen 3 Thinking-2507 — thinking-only, no soft-switch
    if ("qwen3-thinking" in m or "qwen3:thinking" in m
            or "qwen3-30b-a3b-thinking" in m):
        return ModelProfile(
            thinks_by_default=True, can_disable_thinking=False,
            temperature=1.0, top_p=0.95, top_k=20,
        )
    # Qwen 3 Instruct-2507 — strictly non-thinking
    if "instruct-2507" in m or "qwen3:30b-a3b-instruct" in m:
        return ModelProfile(
            thinks_by_default=False, can_disable_thinking=False,
            temperature=0.7, top_p=0.8, top_k=20,
        )
    # Qwen 2.5 instruct — non-thinking, classic sampling
    if "qwen2.5" in m and "instruct" in m:
        return ModelProfile(
            thinks_by_default=False, can_disable_thinking=False,
            temperature=0.7, top_p=0.8, top_k=20,
        )
    # Default: assume non-thinking; use the task temperature as-is.
    return ModelProfile(
        thinks_by_default=False, can_disable_thinking=False,
        temperature=0.7, top_p=0.9, top_k=40,
    )


def effective_budget(
    task: str, model: str, *, force_no_thinking: bool = False,
) -> dict:
    """Return a budget dict scaled for the model's profile.

    Default (``force_no_thinking=False``): thinking models get an 8×
    num_predict multiplier with a 12k floor, and thinking-mode
    sampling.

    ``force_no_thinking=True``: if the model supports the soft switch
    (hybrid Qwen3.5/3.6), return the instruct-mode sampling recipe
    (temp=0.7, top_p=0.80, presence_penalty=1.5 …) and base
    num_predict — no thinking multiplier needed. Phase 54.6.243
    added this path after the 54.6.240 bench showed qwen3.6 looking
    worse than qwen3:30b-a3b-instruct only because it was compared
    in thinking mode against non-thinking baselines; with this
    budget the same model beats them on a probe run.
    """
    base = BUDGETS[task]
    p = profile_for(model)
    use_nt = force_no_thinking and p.can_disable_thinking
    predict = base["num_predict"]
    if p.thinks_by_default and not use_nt:
        predict = max(THINKING_MIN_PREDICT, predict * THINKING_PREDICT_MULT)
    if use_nt:
        return {
            "num_ctx":     base["num_ctx"],
            "num_predict": predict,
            "temperature": p.nt_temperature,
            "top_p":       p.nt_top_p,
            "top_k":       p.nt_top_k,
            "min_p":       p.nt_min_p,
            "presence_penalty": p.nt_presence_penalty,
            "thinks_by_default": False,
        }
    return {
        "num_ctx":     base["num_ctx"],
        "num_predict": predict,
        "temperature": p.temperature,
        "top_p":       p.top_p,
        "top_k":       p.top_k,
        "thinks_by_default": p.thinks_by_default,
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
    # Handle NULL authors / year safely. Authors is JSONB list of
    # dicts {name, orcid, affiliation} — extract `name` before joining.
    def _stringify(v):
        if isinstance(v, str):
            return v
        if not v:
            return ""
        out = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(str(item.get("name") or ""))
        return ", ".join(x for x in out if x)

    authors_str = _stringify(authors)
    kw_str = _stringify(keywords)
    dom_str = _stringify(domains)
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


def _call_model_raw(
    system: str, user: str, model: str, budget: dict,
    *, force_no_thinking: bool = False,
) -> dict:
    """One chat call. Returns {content, thinking, elapsed_s, eval_count,
    done_reason, error, budget_predict, temp}. We go through the raw
    ollama client so we can capture ``thinking`` separately — without
    that, a thinking-runaway is indistinguishable from a hang.

    Phase 54.6.85 — budget includes Qwen-recommended sampling (top_p,
    top_k) and ``thinks_by_default`` signal. By default the caller
    leaves Ollama to decide whether to think (no explicit ``think``
    flag) so the "default behavior" comparison stays pure.

    Phase 54.6.241 — ``force_no_thinking=True`` explicitly passes
    ``think=False`` to Ollama. Only meaningful on hybrid models
    where ``profile_for(model).can_disable_thinking`` is true;
    non-hybrid models ignore the flag. The write_section task in
    particular uses this because a short grounded writing prompt
    doesn't benefit from CoT, and we observed qwen3.6:27b-dense
    burning the whole num_predict budget inside <think>…</think>
    with an empty final answer when thinking was on.
    """
    import ollama
    client = ollama.Client()
    t0 = time.monotonic()
    options = {
        "temperature": budget["temperature"],
        "num_ctx":     budget["num_ctx"],
        "num_predict": budget["num_predict"],
        "num_batch":   1024,
    }
    if budget.get("top_p") is not None:
        options["top_p"] = budget["top_p"]
    if budget.get("top_k") is not None:
        options["top_k"] = budget["top_k"]
    # Phase 54.6.243 — forward instruct-mode penalties when the
    # budget carries them (set only by force_no_thinking on hybrid
    # Qwen3.5/3.6). The Qwen3.6 README prescribes presence_penalty=1.5
    # for non-thinking mode specifically because without it the model
    # repeats itself; omitting it was one of the silent fairness
    # bugs behind the "qwen3.6 looks worse than qwen3:30b-a3b" result
    # on 54.6.240.
    if budget.get("min_p") is not None:
        options["min_p"] = budget["min_p"]
    if budget.get("presence_penalty") is not None:
        options["presence_penalty"] = budget["presence_penalty"]
    chat_kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        options=options,
        keep_alive=0,   # unload after so the next candidate gets fresh VRAM
    )
    # Phase 54.6.241 — explicit think flag for hybrid models. Only
    # applied when force_no_thinking is set AND the model supports
    # the switch (non-hybrid models either error or no-op on the
    # flag; we skip silently). Passing `think=False` on a model
    # that supports it suppresses the <think> block entirely and
    # produces a direct answer — fixes the write_section runaway
    # where qwen3.6:27b-dense produced 3970 thinking tokens + 0
    # content chars.
    if force_no_thinking:
        prof = profile_for(model)
        if prof.can_disable_thinking:
            chat_kwargs["think"] = False
    try:
        resp = client.chat(**chat_kwargs)
    except Exception as exc:
        return {"error": str(exc), "elapsed_s": time.monotonic() - t0,
                "content": "", "thinking": "", "eval_count": 0,
                "done_reason": "exception",
                "budget_predict": budget.get("num_predict", 0),
                "temp": budget.get("temperature", 0),
                "thinks_by_default": budget.get("thinks_by_default", False)}
    msg = resp.get("message") or {}
    return {
        "error": None,
        "elapsed_s": time.monotonic() - t0,
        "content":   msg.get("content", "") or "",
        "thinking":  msg.get("thinking", "") or "",
        "eval_count":  resp.get("eval_count", 0) or 0,
        "done_reason": resp.get("done_reason", ""),
        "budget_predict": budget.get("num_predict", 0),
        "temp":        budget.get("temperature", 0),
        "thinks_by_default": budget.get("thinks_by_default", False),
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
        # Phase 54.6.243 — extract_kg is structured JSON; hybrid models
        # should run this task in instruct (non-thinking) mode for an
        # apples-to-apples comparison vs qwen3:30b-a3b-instruct. Pre-243,
        # Qwen3.6 emitted ~16k chars of <think> and truncated to 7 triples
        # vs qwen3's 12 triples in 15s — that made the model look worse
        # when it's actually faster + cleaner when allowed to skip CoT.
        resp = _call_model_raw(
            sys_p, usr_p, model,
            effective_budget("extract_kg", model, force_no_thinking=True),
            force_no_thinking=True,
        )
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
    # Phase 54.6.90 — strip leaked <think>…</think> blocks BEFORE the
    # word count so the on-target verdict measures the actual answer,
    # not answer+thinking. Pre-54.6.90 qwen3.6-ud was marked over-target
    # (1527 words) even though its real answer was ~300-500; the rest
    # was raw <think> content that Ollama's native tag-splitter failed
    # to hoist out. thinking_leaked stays as a diagnostic flag —
    # computed on the RAW text so we still surface the format bug.
    from sciknow.core.wiki_ops import _strip_thinking
    leaked = bool(_THINK_TAG.search(text))
    cleaned = _strip_thinking(text)
    body = re.sub(r"^#+ .*$", "", cleaned, flags=re.MULTILINE)
    words = len(body.split())
    out = {
        "words":            words,
        "words_raw":        len(text.split()),   # answer + leaked thinking
        "on_target":        1 if 200 <= words <= 800 else 0,
        "thinking_leaked":  1 if leaked else 0,
        "title_in_output":  1 if paper.title[:30].lower() in cleaned.lower() else 0,
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
        # Phase 54.6.243 — same fairness fix as extract_kg: compile_summary
        # is a direct-output prose task, not a reasoning task. qwen3:30b-a3b
        # runs it in 13s non-thinking; qwen3.6 in thinking took 125s with
        # a 12.5k-char CoT preamble. Force no-thinking + instruct sampling.
        resp = _call_model_raw(
            sys_p, usr_p, model,
            effective_budget("compile_summary", model, force_no_thinking=True),
            force_no_thinking=True,
        )
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
    # Phase 54.6.90 — same fix as _score_compile: strip leaked
    # <think>…</think> before counting so on_target measures the
    # answer, not answer+thinking. Citation counts also use the
    # stripped text — a [N] inside a <think> block isn't a real
    # citation in the final output.
    from sciknow.core.wiki_ops import _strip_thinking
    leaked = bool(_THINK_TAG.search(text))
    cleaned = _strip_thinking(text)
    body = re.sub(r"^#+ .*$", "", cleaned, flags=re.MULTILINE)
    words = len(body.split())
    cite_count = len(_CITE_MARK.findall(cleaned))
    return {
        "words":           words,
        "words_raw":       len(text.split()),
        "on_target":       1 if 80 <= words <= 300 else 0,  # target was 150
        "thinking_leaked": 1 if leaked else 0,
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
        # Phase 54.6.243 — aligned with extract_kg / compile_summary:
        # hybrid models run this direct-writing task in instruct mode
        # with instruct sampling (temp=0.7 / top_p=0.80 / top_k=20 /
        # presence_penalty=1.5), matching the Qwen3.6 README recipe.
        # 54.6.242's "use thinking defaults" rollback was wrong: the
        # probe in Phase 54.6.243 confirms thinking mode burns 80 %+
        # of the budget inside <think> for a 150-word writing prompt,
        # so instruct sampling is the right default for this task.
        resp = _call_model_raw(
            sys_p, usr_p, model,
            effective_budget("write_section", model, force_no_thinking=True),
            force_no_thinking=True,
        )
        if resp.get("error"):
            yield BenchMetric(f"{model}::status", "error", "",
                              note=resp["error"][:80])
            continue

        content = resp["content"]
        thinking = resp.get("thinking", "") or ""
        scored = _score_write_section(content)
        yield BenchMetric(f"{model}::elapsed_s",
                          round(resp["elapsed_s"], 1), "s",
                          note=f"done={resp.get('done_reason','')}")
        yield BenchMetric(f"{model}::eval_count", resp["eval_count"], "tok")
        yield BenchMetric(f"{model}::content_chars", len(content), "chars")
        # 54.6.241 — expose thinking_chars so a silent runaway is
        # distinguishable from an empty-response failure. Should be
        # 0 when force_no_thinking sticks; a positive number means
        # the model still emitted a <think> block despite the flag.
        yield BenchMetric(f"{model}::thinking_chars", len(thinking), "chars",
                          note="should be 0 with force_no_thinking=True")
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
