"""Writing-quality benchmark harness for sciknow's LLM-driven features.

Companion to ``sciknow/testing/model_sweep.py`` (which benchmarks raw
speed + JSON-output quality for extract-kg). This module goes deeper on
prose generation: faithfulness, citation quality, coherence, and
discipline-specific writing properties.

Design
------

**Scope.** Every sciknow feature that emits prose (wiki compile, wiki
polish, book write, book autowrite, book review, ask synthesize, etc.)
should have a quality bench here. Structured-output tasks (scorer,
verifier, consensus) get lighter benches focused on validity +
schema compliance.

**Metric families (what + why)**

1. ``faithfulness`` — NLI-based entailment between each output
   sentence and the retrieved source chunks. Local cross-encoder model
   (``cross-encoder/nli-deberta-v3-base``, ~440 MB, ~50 ms per pair on
   CPU). Mean entailment probability + fraction of ungrounded
   sentences (below threshold). AlignScore-style, but cheaper.

2. ``citation_quality`` — ALCE-adapted for our ``[N]`` / ``[@key]`` /
   ``(Author YYYY)`` citation formats. ``citation_recall`` = fraction
   of sentences where at least one emitted citation's referent chunk
   entails the sentence. ``citation_precision`` = fraction of emitted
   citations that are non-"irrelevant" per ALCE's two conditions.

3. ``length`` — words within a sensible band for the task. Binary hit
   metric.

4. ``coherence_pairwise`` — LLM-as-judge pairwise. For each pair of
   candidate models on the same input, a JUDGE model (always a
   different family than either tested model) picks A/B/TIE.
   Randomized order to mitigate position bias. Win-rate is reported.

5. ``task_specific`` — JSON validity, snake_case predicate %, topic
   grounding (title in output), thinking-tag leakage, etc. One or two
   per task.

**Why these and not others**

- BLEU/ROUGE are included only as sanity checks because they reward
  surface overlap, which correlates poorly with writing quality for
  scientific prose (you want the same meaning expressed clearly, not
  the same tokens).
- MAUVE (used by ALCE for fluency) needs a big text-completion LM to
  compare distributions; skipped for local-first reasons.
- FactScore is claim-atomization + LLM-judge; redundant with our
  NLI-based faithfulness + LLM-judge pair.
- BERTScore would duplicate what bge-m3 cosine already does; skipped.

**Judge-model rotation**

Self-enhancement bias (an LLM rating its own output higher) is a
real risk (see G-Eval + LLM-as-a-judge surveys). The ``_pick_judge``
helper always returns a model from a different family than the one
under test — concretely, qwen judges gemma and vice versa.

**Fixed fixtures**

Three papers are pinned by doc_id prefix. Three topics are pinned
for ask_synthesize / wiki_consensus. These are locked so every
bench run is apples-to-apples. See CANDIDATE_PAPERS / CANDIDATE_TOPICS.

Runtime
-------

Per-task cost ≈ #candidates × (gen_time ~30-90s) + #pairs ×
judge_time ~10s + NLI scoring ~5s. Full quality layer with 3
candidates × 7 tasks ≈ 30-60 min on a 3090. Expect to run weekly
or after any prompt/model change to writing-quality features.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, Callable, Iterable

from sciknow.testing.bench import BenchMetric, skip

logger = logging.getLogger("sciknow.testing.quality")


# ════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════


# Candidates for writing-quality benchmarks. Pinned to the current
# production picks + an alternative per role so we can detect drift
# when a prompt or model swap changes quality.
CANDIDATE_MODELS: list[str] = [
    # Phase 54.6.53 — full sweep across every locally-installed model
    # for the 2026-04-17 bench. Not-installed candidates are skipped
    # gracefully. Order is stable for diff-ability across runs.
    "gemma3:27b-it-qat",                      # current book-writing baseline
    "gemma4:26b-a4b-it-q4_K_M",
    "gemma4:31b",                             # may fail load on Ollama < 0.20.0
    "gemopus4:26b-a4b-q4_K_M",
    "mn-darkest-universe:29b-q4_K_M",
    "nemotron-cascade-2:30b",
    "ornstein3.6:35b-a3b-q4_K_S",
    "qwen2.5:32b-instruct-q4_K_M",
    "qwen3:30b-a3b-instruct-2507-q4_K_M",     # current compile + extract-kg
    "qwen3.5:27b",
    "qwen3.6:35b-a3b-q4_K_M",
    "qwen3.6:35b-a3b-ud-q4_K_S",
    "supergemma4:26b-uncensored-q4_K_M",
    "supergemma4:31b-abliterated-q4_K_M",     # broken output — keeps for ref
]

# Judge-model roster. Used by _pick_judge to pick a model from a
# different family than the candidate pair being compared. Ordered
# by preference; the first non-conflict is chosen. With 14 candidates
# spanning qwen/gemma/mistral-nemo/nvidia-nemotron families, most pairs
# have a non-conflict judge available; the remaining pairs fall through
# to self-judge with a logged warning.
JUDGE_MODEL_ROSTER: list[str] = [
    "qwen3:30b-a3b-instruct-2507-q4_K_M",     # fast non-thinking qwen
    "gemma3:27b-it-qat",                      # non-thinking gemma
    "qwen2.5:32b-instruct-q4_K_M",            # reliable non-thinking fallback
    "mn-darkest-universe:29b-q4_K_M",         # mistral-nemo, different family
    "nemotron-cascade-2:30b",                 # nvidia, different family
]

# Test papers (pinned by 8-char document_id prefix). One math-heavy
# pathological case, one descriptive prose paper, one review-style
# for variety.
CANDIDATE_PAPERS: list[str] = [
    "4092d6ad",   # Nature Controls CO2 II — math-heavy
    "631fd2ea",   # Sun Reversed Weakening Trend — descriptive solar physics
]

# Topics for ask_synthesize / wiki_consensus. These are corpus-dependent
# — each must have wiki pages or retrieval hits in the user's project.
# Picked from topics that produced real consensus output during
# interactive testing.
CANDIDATE_TOPICS: list[str] = [
    "climate sensitivity",
    "solar variability and climate",
]

# Generation budgets per task (match production defaults). A thinking
# model that exhausts these in ``<think>`` alone is a legitimate
# quality failure for the sweep.
BUDGETS = {
    # Phase 54.6.91 — budgets raised AND made model-aware via
    # effective_budget() below. Pre-54.6.91, every model got the same
    # 1000-2000 num_predict / temp 0-0.4, which (per 54.6.85 and
    # BENCH_METHODOLOGY R2) is architecturally unfair to thinking
    # models that need 16k+ budget. The quality bench was fixed in a
    # separate commit from the sweep — don't let this drift again.
    "wiki_summary":     {"num_ctx": 8192,  "num_predict": 2000, "temperature": 0.3},
    "wiki_polish":      {"num_ctx": 8192,  "num_predict": 2000, "temperature": 0.2},
    "autowrite_writer": {"num_ctx": 4096,  "num_predict": 2000, "temperature": 0.4},
    "book_review":      {"num_ctx": 8192,  "num_predict": 2000, "temperature": 0.3},
    "ask_synthesize":   {"num_ctx": 16384, "num_predict": 2500, "temperature": 0.3},
    "autowrite_scorer": {"num_ctx": 8192,  "num_predict": 1500, "temperature": 0.2},
    "judge":            {"num_ctx": 8192,  "num_predict": 400,  "temperature": 0.1},
}


def effective_budget(task: str, model: str) -> dict:
    """Phase 54.6.91 — quality-bench analog of model_sweep.effective_budget.

    Scales `num_predict` by 8× for thinking models (min 12k floor) so
    they have room for CoT + answer. Overrides sampling with Qwen-
    recommended per-family parameters instead of the one-size-fits-all
    budget default. Mirrors sweep exactly so both benches apply the
    same methodology to the same candidates.
    """
    from sciknow.testing.model_sweep import (
        profile_for, THINKING_PREDICT_MULT, THINKING_MIN_PREDICT,
    )
    base = BUDGETS[task]
    p = profile_for(model)
    predict = base["num_predict"]
    if p.thinks_by_default:
        predict = max(THINKING_MIN_PREDICT, predict * THINKING_PREDICT_MULT)
    return {
        "num_ctx":     base["num_ctx"],
        "num_predict": predict,
        "temperature": p.temperature,
        "top_p":       p.top_p,
        "top_k":       p.top_k,
        "thinks_by_default": p.thinks_by_default,
    }

# NLI model used for faithfulness + citation quality. This is the
# standard cross-encoder NLI model from sentence-transformers — trained
# on SNLI+MultiNLI+FEVER+ANLI. ~440 MB. Runs fast on CPU, faster on GPU.
# Output class order is [contradiction, entailment, neutral] after
# softmax; we take P(entailment).
NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"

# Entailment threshold: sentence is considered grounded if P(entailment)
# against best-matching chunk > this. 0.5 is the standard decision
# boundary for binary NLI predictions.
FAITHFULNESS_THRESHOLD = 0.5

# Hedging words — used as a scientific-writing style marker. Higher
# density in results/discussion sections is generally better.
_HEDGE_WORDS = re.compile(
    r"\b(may|might|could|suggest|suggests|suggested|likely|possibly|"
    r"appears?|appear|seems?|seem|tend|tends|indicate|indicates|"
    r"indicative|implies?|consistent|hypothes|posit)\b",
    re.IGNORECASE,
)

# Citation marker extractor (same regex used in model_sweep.py).
_CITE_MARK = re.compile(
    r"\[(\d+)\]|\[@([\w-]+)\]|\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?,?|and\s+\w+))?)\s+(\d{4})\)"
)
_THINK_TAG = re.compile(r"<think>|</think>", re.IGNORECASE)


def _clean_and_count(raw: str) -> tuple[str, bool, int]:
    """Phase 54.6.90 — centralize the "did thinking leak → strip it →
    count words on the stripped output" accounting so every quality
    task reads the SAME metric family.

    Returns (cleaned_text, thinking_leaked_bool, words_in_cleaned).

    Pre-54.6.90 every task counted words in the raw text, which
    included any <think>…</think> blocks Ollama's native splitter
    failed to hoist out. That inflated "words", broke on_target, and
    diluted citations_per_100w — exactly the bug we caught on
    qwen3.6-ud in the sweep re-run.
    """
    from sciknow.core.wiki_ops import _strip_thinking
    leaked = bool(_THINK_TAG.search(raw or ""))
    cleaned = _strip_thinking(raw or "")
    body = re.sub(r"^#+\s.*$", "", cleaned, flags=re.MULTILINE)
    words = len(body.split())
    return cleaned, leaked, words


# ════════════════════════════════════════════════════════════════════════
# Lazy-loaded NLI model (shared across benches)
# ════════════════════════════════════════════════════════════════════════


_NLI_MODEL = None  # type: ignore


def _get_nli():
    """Return the NLI cross-encoder, lazy-loaded on first use.

    Using sentence-transformers CrossEncoder because it handles
    tokenization + batching + GPU detection for us. The model emits
    3-class logits (contradiction / entailment / neutral); we softmax
    and take the entailment probability."""
    global _NLI_MODEL
    if _NLI_MODEL is not None:
        return _NLI_MODEL
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        skip(f"sentence_transformers unavailable: {exc}")
    try:
        # predict_fn default handles softmax; we just need the
        # right output labels
        _NLI_MODEL = CrossEncoder(NLI_MODEL_ID, max_length=512)
    except Exception as exc:
        skip(f"could not load NLI model {NLI_MODEL_ID}: {exc}")
    return _NLI_MODEL


def _nli_entail_probs(premise_hyp_pairs: list[tuple[str, str]]) -> list[float]:
    """Batch-score (premise, hypothesis) pairs. Returns P(entailment)
    in [0, 1] per pair.

    The cross-encoder output has 3 classes in order
    [contradiction, entailment, neutral]. We softmax and take column 1."""
    if not premise_hyp_pairs:
        return []
    nli = _get_nli()
    import numpy as np
    logits = nli.predict(premise_hyp_pairs, convert_to_numpy=True,
                         show_progress_bar=False, batch_size=16)
    # Some model configs output 2 classes (not_entail / entail) instead
    # of 3. Handle both.
    if logits.ndim == 1:
        # single-value output — interpret as logit of entailment
        return [float(1.0 / (1.0 + np.exp(-x))) for x in logits]
    if logits.shape[1] == 3:
        # [contradiction, entailment, neutral]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return [float(p) for p in probs[:, 1]]
    if logits.shape[1] == 2:
        # [not_entailment, entailment]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return [float(p) for p in probs[:, 1]]
    # Unknown shape — fall back to sigmoid of first column
    return [float(1.0 / (1.0 + np.exp(-x))) for x in logits[:, 0]]


# ════════════════════════════════════════════════════════════════════════
# Faithfulness scoring (sentence vs best-matching chunk of source)
# ════════════════════════════════════════════════════════════════════════


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Drops short fragments (< 10 chars)."""
    # Strip markdown headers + list markers
    body = re.sub(r"^#+\s.*$", "", text or "", flags=re.MULTILINE)
    body = re.sub(r"^\s*[-*]\s+", "", body, flags=re.MULTILINE)
    sents = [s.strip() for s in _SENT_SPLIT.split(body) if s.strip()]
    return [s for s in sents if len(s) >= 10]


def _chunk_text(text: str, chunk_chars: int = 1500) -> list[str]:
    """Slice long source text into overlapping chunks for NLI premise.
    350-token chunks are what AlignScore uses; we approximate at
    ~1500 chars (roughly 400 tokens) with no overlap for simplicity."""
    if not text:
        return []
    return [text[i:i+chunk_chars] for i in range(0, len(text), chunk_chars)]


def faithfulness_score(output: str, source_text: str) -> dict:
    """Per-sentence entailment score of output against source_text.

    Returns:
        mean: average P(entailment) across all sentences
        min: worst sentence's score
        hallucinated_pct: fraction of sentences below threshold
        n_sentences: how many scored
    """
    sentences = _split_sentences(output)
    chunks = _chunk_text(source_text)
    if not sentences or not chunks:
        return {"mean": 0.0, "min": 0.0, "hallucinated_pct": 0.0,
                "n_sentences": len(sentences)}

    # For each (chunk, sentence) pair we want max-over-chunks P(entail).
    # Batch the pairs flat, then reduce.
    pairs: list[tuple[str, str]] = []
    for s in sentences:
        for c in chunks:
            pairs.append((c, s))
    probs = _nli_entail_probs(pairs)

    # Reduce: max across chunks per sentence.
    n_chunks = len(chunks)
    per_sent: list[float] = []
    for i in range(len(sentences)):
        window = probs[i * n_chunks : (i + 1) * n_chunks]
        per_sent.append(max(window) if window else 0.0)

    mean = sum(per_sent) / len(per_sent)
    mn = min(per_sent)
    halluc = sum(1 for p in per_sent if p < FAITHFULNESS_THRESHOLD)
    return {
        "mean":             round(mean, 3),
        "min":              round(mn, 3),
        "hallucinated_pct": round(100.0 * halluc / len(per_sent), 1),
        "n_sentences":      len(per_sent),
    }


# ════════════════════════════════════════════════════════════════════════
# Citation quality (ALCE-adapted)
# ════════════════════════════════════════════════════════════════════════


def _extract_citations(output: str) -> list[tuple[int, int, str]]:
    """Extract citation markers from text. Returns list of
    (start, end, marker_text) tuples ordered by position."""
    out: list[tuple[int, int, str]] = []
    for m in _CITE_MARK.finditer(output):
        out.append((m.start(), m.end(), m.group(0)))
    return out


def _group_citations_per_sentence(output: str, citations: list[tuple[int, int, str]]) -> list[tuple[str, list[str]]]:
    """Walk the output once; for each sentence emit (sentence_text,
    [citation_markers_inside_it]). A marker is attributed to whichever
    sentence contains its end position."""
    # Build sentence-offset map
    sentences = _split_sentences(output)
    result: list[tuple[str, list[str]]] = []
    # Simpler approach: re-scan output linearly and track current
    # sentence buffer. Good enough for short-form science prose.
    # Fall back to dumping all citations into the last sentence if
    # parsing fails.
    cur_start = 0
    cite_idx = 0
    for s in sentences:
        pos = output.find(s, cur_start)
        if pos < 0:
            continue
        sent_end = pos + len(s)
        mrks = []
        while cite_idx < len(citations) and citations[cite_idx][1] <= sent_end + 2:
            mrks.append(citations[cite_idx][2])
            cite_idx += 1
        result.append((s, mrks))
        cur_start = sent_end
    return result


def citation_quality_alce(output: str, source_chunks: list[str]) -> dict:
    """Adapted from ALCE (Gao et al., EMNLP 2023).

    Citations in the output reference retrieved chunks by position
    (``[1]`` → source_chunks[0], etc.). For each sentence with at
    least one citation:
      - recall = 1 iff concat(all cited chunks) entails the sentence
    For each emitted citation c_ij referencing chunk k:
      - "irrelevant" iff (a) chunk[k] alone does NOT entail s_i AND
                          (b) removing chunk[k] from C_i does NOT
                              break entailment (i.e. redundant).
      - precision = 1 if recall_i = 1 AND NOT irrelevant
    """
    if not source_chunks:
        return {"citation_recall": None, "citation_precision": None,
                "n_cited_sentences": 0, "n_total_citations": 0}

    citations = _extract_citations(output)
    if not citations:
        return {"citation_recall": None, "citation_precision": None,
                "n_cited_sentences": 0, "n_total_citations": 0}

    # Per-sentence grouping
    grouped = _group_citations_per_sentence(output, citations)

    def marker_to_chunk_idx(marker: str) -> int | None:
        """Extract the 1-based index from a numeric marker."""
        m = re.match(r"\[(\d+)\]", marker)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(source_chunks):
                return idx
        # Non-numeric citations (@key, (Author YYYY)) — we can't map
        # these to chunks without a bibliography lookup; treat as
        # non-evaluable (neither precision nor recall impact).
        return None

    cited_sents: list[tuple[str, list[int]]] = []
    for sent, mrks in grouped:
        chunk_ids: list[int] = []
        for m in mrks:
            idx = marker_to_chunk_idx(m)
            if idx is not None:
                chunk_ids.append(idx)
        if chunk_ids:
            cited_sents.append((sent, chunk_ids))

    if not cited_sents:
        return {"citation_recall": None, "citation_precision": None,
                "n_cited_sentences": 0, "n_total_citations": 0}

    # Recall: for each cited sentence, does concat(cited chunks) entail it?
    recall_pairs = [
        ("\n".join(source_chunks[k] for k in ks), s)
        for s, ks in cited_sents
    ]
    recall_probs = _nli_entail_probs(recall_pairs)
    recall_per_sent = [p >= FAITHFULNESS_THRESHOLD for p in recall_probs]
    recall = 100.0 * sum(recall_per_sent) / len(recall_per_sent)

    # Precision: per-citation.
    # Build the pairs we need:
    #   (a) chunk[k] vs s_i  — is this citation supportive alone?
    #   (b) concat(chunks excluding k) vs s_i — is it redundant?
    alone_pairs: list[tuple[str, str]] = []
    except_pairs: list[tuple[str, str]] = []
    citation_to_sent: list[int] = []
    for si, (s, ks) in enumerate(cited_sents):
        for k in ks:
            citation_to_sent.append(si)
            alone_pairs.append((source_chunks[k], s))
            others = [source_chunks[o] for o in ks if o != k]
            except_pairs.append((("\n".join(others) if others else ""), s))

    alone_probs = _nli_entail_probs(alone_pairs) if alone_pairs else []
    except_probs = _nli_entail_probs(except_pairs) if except_pairs else []

    prec_per_cite: list[bool] = []
    for i, si in enumerate(citation_to_sent):
        # Sentence must have recall=1 for any of its citations to count as precision=1
        if not recall_per_sent[si]:
            prec_per_cite.append(False)
            continue
        alone_supports = alone_probs[i] >= FAITHFULNESS_THRESHOLD
        except_still_supports = (except_probs[i] >= FAITHFULNESS_THRESHOLD
                                 if except_pairs[i][0] else False)
        # ALCE: irrelevant iff (NOT alone) AND (except-still-supports)
        irrelevant = (not alone_supports) and except_still_supports
        prec_per_cite.append(not irrelevant)

    precision = 100.0 * sum(prec_per_cite) / len(prec_per_cite) if prec_per_cite else 0.0

    return {
        "citation_recall":    round(recall, 1),
        "citation_precision": round(precision, 1),
        "n_cited_sentences":  len(cited_sents),
        "n_total_citations":  len(prec_per_cite),
    }


# ════════════════════════════════════════════════════════════════════════
# LLM-as-judge pairwise
# ════════════════════════════════════════════════════════════════════════


def _pick_judge(*candidates_under_test: str) -> str:
    """Pick the first judge model from the roster that isn't one of
    the models being compared. Mitigates self-enhancement bias."""
    tested = set(candidates_under_test)
    for m in JUDGE_MODEL_ROSTER:
        if m not in tested:
            return m
    # Fallback: reuse the first roster entry (will self-judge; flag in logs)
    logger.warning("No judge model differs from candidates %s — self-judge bias possible",
                   tested)
    return JUDGE_MODEL_ROSTER[0]


_JUDGE_SYSTEM = """\
You are an expert reviewer comparing two candidate pieces of scientific writing.
Evaluate ONLY on: (1) factual grounding (does it stick to the source material?), \
(2) coherence and logical flow, (3) scientific clarity and precision.
Output ONLY JSON matching this schema:
{"winner": "A" | "B" | "TIE", "reason": "<=25 words>"}
Start your response immediately with the JSON object. No preamble."""

_JUDGE_USER_TEMPLATE = """\
Task / Topic: {topic}

Source material excerpt:
---
{source_preview}
---

Candidate A:
---
{output_a}
---

Candidate B:
---
{output_b}
---

Which candidate is better? Begin with {{."""


def _llm_judge_pairwise(
    topic: str, source_preview: str, output_a: str, output_b: str,
    judge_model: str,
) -> dict:
    """Ask judge which candidate is better. Returns
    {winner: "A"|"B"|"TIE", reason: str, elapsed_s: float}."""
    from sciknow.rag.llm import complete as llm_complete

    # Randomize A/B to mitigate position bias (we un-randomize the
    # winner label before returning).
    if random.random() < 0.5:
        swapped = False
        a, b = output_a, output_b
    else:
        swapped = True
        a, b = output_b, output_a

    prompt = _JUDGE_USER_TEMPLATE.format(
        topic=topic[:200],
        source_preview=source_preview[:2000],
        output_a=a[:3500],
        output_b=b[:3500],
    )
    t0 = time.monotonic()
    try:
        raw = llm_complete(
            _JUDGE_SYSTEM, prompt, model=judge_model,
            temperature=BUDGETS["judge"]["temperature"],
            num_ctx=BUDGETS["judge"]["num_ctx"],
            num_predict=BUDGETS["judge"]["num_predict"],
            keep_alive=-1,
        ) or ""
    except Exception as exc:
        return {"winner": "ERROR", "reason": f"judge call failed: {exc}"[:100],
                "elapsed_s": time.monotonic() - t0}

    elapsed = time.monotonic() - t0

    # Parse JSON
    try:
        from sciknow.core.wiki_ops import _find_json_block, _strip_thinking
        cleaned = _strip_thinking(raw).strip()
        block = _find_json_block(cleaned) or "{}"
        data = json.loads(block, strict=False)
    except Exception:
        return {"winner": "PARSE_ERR", "reason": (raw[:80] or "no output"),
                "elapsed_s": elapsed}

    winner = (data.get("winner") or "").upper()
    if winner not in ("A", "B", "TIE"):
        return {"winner": "PARSE_ERR", "reason": f"bad winner={winner!r}",
                "elapsed_s": elapsed}

    # Un-swap if the order was flipped before showing to the judge
    if swapped and winner in ("A", "B"):
        winner = "B" if winner == "A" else "A"

    return {"winner": winner, "reason": data.get("reason", "")[:120],
            "elapsed_s": elapsed}


# ════════════════════════════════════════════════════════════════════════
# Helpers — DB fetch + uniform call wrapper
# ════════════════════════════════════════════════════════════════════════


def _load_paper_ctx(prefix: str):
    """Reuse model_sweep's loader. Keeps fixture handling in one place."""
    from sciknow.testing.model_sweep import _load_paper
    return _load_paper(prefix)


def _installed_models() -> set[str]:
    """Names of models the user has pulled locally."""
    try:
        import ollama
        client = ollama.Client()
        resp = client.list()
        return {m.get("name") or m.get("model") or "" for m in resp.get("models", [])}
    except Exception:
        return set()


def _call_model_raw(system: str, user: str, model: str, budget: dict) -> dict:
    """Single chat call with thinking captured separately. Same shape as
    model_sweep._call_model_raw but local to this module so the two
    benches can diverge."""
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
                **({"top_p": budget["top_p"]}
                   if budget.get("top_p") is not None else {}),
                **({"top_k": budget["top_k"]}
                   if budget.get("top_k") is not None else {}),
            },
            keep_alive=0,
        )
    except Exception as exc:
        return {"error": str(exc), "elapsed_s": time.monotonic() - t0,
                "content": "", "thinking": "", "eval_count": 0,
                "done_reason": "exception"}
    msg = resp.get("message") or {}
    return {
        "error":       None,
        "elapsed_s":   time.monotonic() - t0,
        "content":     msg.get("content", "") or "",
        "thinking":    msg.get("thinking", "") or "",
        "eval_count":  resp.get("eval_count", 0) or 0,
        "done_reason": resp.get("done_reason", ""),
    }


def _skip_if_model_missing(model: str, installed: set[str]) -> bool:
    """Helper: when set, skip this candidate."""
    return bool(installed) and model not in installed


def _pairwise_win_rates(
    outputs: dict[str, str], topic: str, source_preview: str,
) -> dict[str, dict]:
    """For each pair of candidate outputs, run one pairwise judge. Return
    {model: {"wins": n, "losses": n, "ties": n, "win_rate": float}}."""
    models = [m for m, o in outputs.items() if o]
    results = {m: {"wins": 0, "losses": 0, "ties": 0, "errors": 0} for m in models}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            judge = _pick_judge(a, b)
            verdict = _llm_judge_pairwise(
                topic, source_preview, outputs[a], outputs[b], judge,
            )
            w = verdict["winner"]
            if w == "A":
                results[a]["wins"] += 1; results[b]["losses"] += 1
            elif w == "B":
                results[b]["wins"] += 1; results[a]["losses"] += 1
            elif w == "TIE":
                results[a]["ties"] += 1; results[b]["ties"] += 1
            else:
                results[a]["errors"] += 1; results[b]["errors"] += 1

    # Win rate = wins / (wins + losses), undefined on all ties.
    for m, r in results.items():
        decisive = r["wins"] + r["losses"]
        r["win_rate"] = round(100.0 * r["wins"] / decisive, 1) if decisive else None
    return results


# ════════════════════════════════════════════════════════════════════════
# Task 1 — wiki compile summary (prose)
# ════════════════════════════════════════════════════════════════════════


def b_quality_wiki_summary() -> Iterable[BenchMetric]:
    """Per-model wiki summary quality: faithfulness, length hit,
    thinking leakage, pairwise judge win-rate against other candidates."""
    from sciknow.rag import wiki_prompts

    paper = _load_paper_ctx(CANDIDATE_PAPERS[0])
    sys_p, usr_p = wiki_prompts.wiki_paper_summary(
        title=paper.title, authors=paper.authors, year=paper.year,
        journal="", doi="",
        keywords=paper.keywords, domains=paper.domains,
        abstract=paper.abstract, sections=paper.sections_text,
        existing_slugs=[],
    )
    installed = _installed_models()
    outputs: dict[str, str] = {}

    for model in CANDIDATE_MODELS:
        if _skip_if_model_missing(model, installed):
            yield BenchMetric(f"{model}::status", "not-installed", "")
            continue
        resp = _call_model_raw(sys_p, usr_p, model, effective_budget("wiki_summary", model))
        if resp.get("error"):
            yield BenchMetric(f"{model}::status", "error", "",
                              note=resp["error"][:80])
            continue

        content_raw = resp["content"]
        # Phase 54.6.90 — strip any leaked <think>…</think> before scoring
        # so qwen3.6-ud-style format bugs don't pollute the word / hedge /
        # faithfulness metrics. Feed the CLEANED text to faithfulness too.
        content, thinking, words = _clean_and_count(content_raw)
        outputs[model] = content
        hedge_n = len(_HEDGE_WORDS.findall(content))

        yield BenchMetric(f"{model}::elapsed_s",
                          round(resp["elapsed_s"], 1), "s",
                          note=f"done={resp.get('done_reason','')}")
        yield BenchMetric(f"{model}::words", words, "words")
        yield BenchMetric(f"{model}::on_target",
                          1 if 200 <= words <= 800 else 0, "bool",
                          note="1 if words in [200,800]")
        yield BenchMetric(f"{model}::thinking_leaked",
                          1 if thinking else 0, "bool")
        yield BenchMetric(f"{model}::hedging_per_100w",
                          round(100.0 * hedge_n / max(words, 1), 2), "/100w",
                          note="may/might/suggest/... — appropriate for science")

        # Faithfulness — NLI entailment vs source text
        faith = faithfulness_score(content, paper.source_text)
        yield BenchMetric(f"{model}::faithfulness_mean",
                          faith["mean"], "prob",
                          note=f"P(entail), over {faith['n_sentences']} sentences")
        yield BenchMetric(f"{model}::faithfulness_min",
                          faith["min"], "prob",
                          note="worst sentence")
        yield BenchMetric(f"{model}::hallucinated_pct",
                          faith["hallucinated_pct"], "%",
                          note=f"sentences with P(entail) < {FAITHFULNESS_THRESHOLD}")

    # Pairwise judge round-robin
    if len([o for o in outputs.values() if o]) >= 2:
        judgments = _pairwise_win_rates(
            outputs, topic=f"Wiki summary of '{paper.title[:80]}'",
            source_preview=paper.source_text,
        )
        for model, r in judgments.items():
            yield BenchMetric(f"{model}::judge_wins", r["wins"], "pairs")
            yield BenchMetric(f"{model}::judge_losses", r["losses"], "pairs")
            yield BenchMetric(f"{model}::judge_ties", r["ties"], "pairs")
            if r["win_rate"] is not None:
                yield BenchMetric(f"{model}::judge_win_rate",
                                  r["win_rate"], "%",
                                  note="wins / (wins+losses) excluding ties")


# ════════════════════════════════════════════════════════════════════════
# Task 2 — wiki polish (prose refinement)
# ════════════════════════════════════════════════════════════════════════


_POLISH_SEED = """\
# Solar Cycle 24 Anomalies and Climate Response

Solar cycle 24 exhibited unusual characteristics. Solar cycle 24 exhibited unusual characteristics. The sunspot count peaked at substantially lower levels than cycle 23, and the minimum between cycles was extended by approximately two years. Some researchers have suggested [1] that this extended minimum may signal the onset of a Grand Solar Minimum, comparable to the Maunder Minimum of the 17th century. However other authors point out that the available data are insufficient. That these data are insufficient, and more cycles are needed to make any firm conclusion.

The geomagnetic Ap index reached a 100-year low during the extended minimum of 2008–2009 [2]. This is suggestive of reduced solar wind pressure and a weakened interplanetary magnetic field. Such conditions have been hypothesized to affect cosmic ray flux at Earth's atmosphere, which in turn some authors link to low-cloud-formation rates via the Svensmark mechanism [3]. But the statistical evidence for this connection remains highly contested; remains highly contested in the literature.
"""


def b_quality_wiki_polish() -> Iterable[BenchMetric]:
    """Polish a deliberately rough draft. Good models should: remove
    duplicates, fix clunky phrasing, preserve citations, preserve
    approximate length."""
    from sciknow.rag import wiki_prompts

    sys_p, usr_p = wiki_prompts.wiki_polish(_POLISH_SEED)
    installed = _installed_models()
    outputs: dict[str, str] = {}

    seed_words = len(_POLISH_SEED.split())
    seed_cites = len(_extract_citations(_POLISH_SEED))

    for model in CANDIDATE_MODELS:
        if _skip_if_model_missing(model, installed):
            yield BenchMetric(f"{model}::status", "not-installed", "")
            continue
        resp = _call_model_raw(sys_p, usr_p, model, effective_budget("wiki_polish", model))
        if resp.get("error"):
            yield BenchMetric(f"{model}::status", "error", "",
                              note=resp["error"][:80])
            continue

        content_raw = resp["content"]
        # Phase 54.6.90 — strip leaked <think>…</think> before scoring.
        content, leaked_polish, out_words = _clean_and_count(content_raw)
        outputs[model] = content
        out_cites = len(_extract_citations(content))
        # Did it remove the duplicated sentence? Look for "exhibited unusual characteristics" count
        dup_count_before = _POLISH_SEED.count("exhibited unusual characteristics")
        dup_count_after = content.count("exhibited unusual characteristics")
        dup_count_after2 = content.count("highly contested")  # second duplicate
        dedup_ok = (dup_count_after < dup_count_before and dup_count_after2 < 2)

        yield BenchMetric(f"{model}::elapsed_s",
                          round(resp["elapsed_s"], 1), "s")
        yield BenchMetric(f"{model}::words", out_words, "words")
        yield BenchMetric(f"{model}::length_preserved",
                          1 if 0.5 * seed_words <= out_words <= 1.8 * seed_words else 0,
                          "bool",
                          note=f"output ∈ [0.5×, 1.8×] of seed ({seed_words}w)")
        yield BenchMetric(f"{model}::citations_preserved",
                          1 if out_cites >= seed_cites - 1 else 0, "bool",
                          note=f"seed had {seed_cites} citations, output has {out_cites}")
        yield BenchMetric(f"{model}::deduplicated",
                          1 if dedup_ok else 0, "bool",
                          note="the deliberately-duplicated phrases were fixed")
        yield BenchMetric(f"{model}::thinking_leaked",
                          1 if leaked_polish else 0, "bool")

    if len([o for o in outputs.values() if o]) >= 2:
        judgments = _pairwise_win_rates(
            outputs, topic="Polished version of a rough scientific prose draft",
            source_preview=_POLISH_SEED,
        )
        for model, r in judgments.items():
            if r["win_rate"] is not None:
                yield BenchMetric(f"{model}::judge_win_rate",
                                  r["win_rate"], "%")


# ════════════════════════════════════════════════════════════════════════
# Task 3 — book autowrite writer (prose + citations)
# ════════════════════════════════════════════════════════════════════════


def _build_fake_results():
    """Fabricate SearchResults with realistic scientific content + a
    clear mapping from each [N] marker to source_chunks[N-1]."""
    from sciknow.retrieval.context_builder import SearchResult
    items = [
        ("Climate sensitivity remains debated. The IPCC reports a likely range of "
         "2.5-4 °C per doubling of CO2 based on a synthesis of paleoclimate, "
         "observational, and modeling lines of evidence.",
         "Sherwood et al. 2020", 2020, "Reviews of Geophysics"),
        ("Observational constraints from the instrumental warming record tend to "
         "indicate climate sensitivity near the lower end of the IPCC range, "
         "though paleoclimate reconstructions push back against the lowest values.",
         "Lewis and Curry 2018", 2018, "Journal of Climate"),
        ("Cloud feedbacks are the single largest source of uncertainty in climate "
         "sensitivity estimates. Recent high-resolution GCM studies suggest that "
         "low-cloud cover over subtropical oceans decreases under warming, "
         "producing a positive feedback and pushing sensitivity upward.",
         "Bony et al. 2015", 2015, "Nature Geoscience"),
    ]
    return [
        SearchResult(
            rank=i + 1, score=1.0 - 0.05 * i,
            chunk_id=f"q-{i}", document_id=f"q-doc-{i}",
            section_type="results", section_title=f"Source {i+1}",
            content=content,
            title=title, year=year,
            authors=[{"name": title.split()[0] + " et al."}],
            journal=journal, doi=None,
        )
        for i, (content, title, year, journal) in enumerate(items)
    ]


def b_quality_autowrite_writer() -> Iterable[BenchMetric]:
    """Write a 150-word introduction section about climate sensitivity
    using 3 pre-formatted SearchResult sources. Measure: length hit,
    citation precision/recall (ALCE), faithfulness (NLI), thinking
    leakage, pairwise judge."""
    from sciknow.rag import prompts

    results = _build_fake_results()
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
    outputs: dict[str, str] = {}
    source_chunks = [r.content for r in results]
    source_preview = "\n---\n".join(source_chunks)

    for model in CANDIDATE_MODELS:
        if _skip_if_model_missing(model, installed):
            yield BenchMetric(f"{model}::status", "not-installed", "")
            continue
        resp = _call_model_raw(sys_p, usr_p, model, effective_budget("autowrite_writer", model))
        if resp.get("error"):
            yield BenchMetric(f"{model}::status", "error", "",
                              note=resp["error"][:80])
            continue

        content_raw = resp["content"]
        # Phase 54.6.90 — strip leaked <think> before scoring.
        content, leaked_aw, words = _clean_and_count(content_raw)
        outputs[model] = content

        yield BenchMetric(f"{model}::elapsed_s",
                          round(resp["elapsed_s"], 1), "s")
        yield BenchMetric(f"{model}::words", words, "words")
        yield BenchMetric(f"{model}::on_target",
                          1 if 80 <= words <= 300 else 0, "bool",
                          note="target=150, ok if in [80, 300]")
        yield BenchMetric(f"{model}::thinking_leaked",
                          1 if leaked_aw else 0, "bool")

        # Faithfulness — grounded in the 3 source chunks
        faith = faithfulness_score(content, "\n".join(source_chunks))
        yield BenchMetric(f"{model}::faithfulness_mean",
                          faith["mean"], "prob",
                          note=f"over {faith['n_sentences']} sentences")
        yield BenchMetric(f"{model}::hallucinated_pct",
                          faith["hallucinated_pct"], "%")

        # Citation quality — ALCE-adapted
        cq = citation_quality_alce(content, source_chunks)
        if cq["citation_recall"] is not None:
            yield BenchMetric(f"{model}::citation_recall",
                              cq["citation_recall"], "%",
                              note=f"over {cq['n_cited_sentences']} cited sentences")
            yield BenchMetric(f"{model}::citation_precision",
                              cq["citation_precision"], "%",
                              note=f"over {cq['n_total_citations']} total [N] citations")
        else:
            yield BenchMetric(f"{model}::citation_recall", 0, "%",
                              note="no parseable [N] citations in output")
            yield BenchMetric(f"{model}::citation_precision", 0, "%")

    if len([o for o in outputs.values() if o]) >= 2:
        judgments = _pairwise_win_rates(
            outputs,
            topic="Book introduction section on climate sensitivity uncertainty",
            source_preview=source_preview,
        )
        for model, r in judgments.items():
            if r["win_rate"] is not None:
                yield BenchMetric(f"{model}::judge_win_rate",
                                  r["win_rate"], "%")


# ════════════════════════════════════════════════════════════════════════
# Task 4 — book review (critical prose)
# ════════════════════════════════════════════════════════════════════════


_REVIEW_DRAFT = """\
# Introduction — Climate Sensitivity Uncertainty

The climate sensitivity — the equilibrium temperature response to a doubling of CO2 — is a central uncertainty in climate projections [1]. The IPCC AR6 assessment places the likely range at 2.5–4.0 °C. This range is narrower than in AR5 because multiple independent lines of evidence have converged.

Observational estimates from the historical warming record alone tend to fall toward the lower end [2], while paleoclimate reconstructions push back against values below 2 °C. Climate model projections span the full range and depend critically on low-cloud feedback, which remains the largest single source of uncertainty [3]. This is a big problem.

Policy implications of this uncertainty are substantial. A 4 °C world looks qualitatively different from a 2 °C world, both in terms of regional climate impacts and in terms of the adaptation pathways available to human societies. Therefore understanding what drives the spread of sensitivity estimates is of direct societal relevance.
"""


def b_quality_book_review() -> Iterable[BenchMetric]:
    """Review a fixed draft. Good reviews should: identify genuine
    weaknesses, be structured, be actionable."""
    from sciknow.rag import prompts

    results = _build_fake_results()
    sys_p, usr_p = prompts.review(
        section_type="introduction",
        topic="Scientific uncertainty in climate sensitivity estimates",
        draft_content=_REVIEW_DRAFT,
        results=results,
    )
    installed = _installed_models()
    outputs: dict[str, str] = {}

    for model in CANDIDATE_MODELS:
        if _skip_if_model_missing(model, installed):
            yield BenchMetric(f"{model}::status", "not-installed", "")
            continue
        resp = _call_model_raw(sys_p, usr_p, model, effective_budget("book_review", model))
        if resp.get("error"):
            yield BenchMetric(f"{model}::status", "error", "",
                              note=resp["error"][:80])
            continue

        content_raw = resp["content"]
        # Phase 54.6.90 — strip leaked <think> before scoring.
        content, leaked_br, words = _clean_and_count(content_raw)
        outputs[model] = content
        # Does the review mention "big problem" weakness (the
        # deliberately vague sentence in the draft)? This is a planted
        # signal — a good reviewer notices vague language.
        catches_vague = ("big problem" in content.lower()
                         or "vague" in content.lower()
                         or "unspecific" in content.lower())
        # Does it reference the target dimensions?
        mentions_dims = sum(
            1 for d in ("groundedness", "completeness", "coherence",
                         "citation", "accuracy")
            if d.lower() in content.lower()
        )

        yield BenchMetric(f"{model}::elapsed_s",
                          round(resp["elapsed_s"], 1), "s")
        yield BenchMetric(f"{model}::words", words, "words")
        yield BenchMetric(f"{model}::thinking_leaked",
                          1 if leaked_br else 0, "bool")
        yield BenchMetric(f"{model}::catches_vague_sentence",
                          1 if catches_vague else 0, "bool",
                          note="noticed the planted 'big problem' weakness")
        yield BenchMetric(f"{model}::dimensions_covered",
                          mentions_dims, "dims",
                          note="mentions groundedness/completeness/coherence/citation/accuracy")

    if len([o for o in outputs.values() if o]) >= 2:
        judgments = _pairwise_win_rates(
            outputs,
            topic="Reviewing a scientific introduction section",
            source_preview=_REVIEW_DRAFT,
        )
        for model, r in judgments.items():
            if r["win_rate"] is not None:
                yield BenchMetric(f"{model}::judge_win_rate",
                                  r["win_rate"], "%")


# ════════════════════════════════════════════════════════════════════════
# Task 5 — autowrite scorer (structured JSON)
# ════════════════════════════════════════════════════════════════════════


_GOOD_DRAFT = _REVIEW_DRAFT  # well-cited, 3 citations, topical
_BAD_DRAFT = """\
# Introduction

Climate change is a very important topic. Scientists talk about it a lot. \
Temperature goes up when CO2 goes up, that's basically the main idea. \
There are several theories. Some say the sensitivity is high, others say it is low. \
Nobody really knows. The models disagree with each other. More research is needed.
"""


def b_quality_autowrite_scorer() -> Iterable[BenchMetric]:
    """The autowrite scorer returns JSON with 5 dimensions + overall.
    Good scorers should (1) always emit valid JSON, (2) rank the
    obviously-bad draft below the obviously-good one, (3) flag the
    weakest dimension accurately on the bad draft."""
    from sciknow.rag import prompts

    results = _build_fake_results()
    installed = _installed_models()

    for model in CANDIDATE_MODELS:
        if _skip_if_model_missing(model, installed):
            yield BenchMetric(f"{model}::status", "not-installed", "")
            continue

        sys_p_good, usr_p_good = prompts.score_draft(
            section_type="introduction",
            topic="Climate sensitivity uncertainty",
            draft_content=_GOOD_DRAFT, results=results,
        )
        sys_p_bad, usr_p_bad = prompts.score_draft(
            section_type="introduction",
            topic="Climate sensitivity uncertainty",
            draft_content=_BAD_DRAFT, results=results,
        )

        good = _call_model_raw(sys_p_good, usr_p_good, model, effective_budget("autowrite_scorer", model))
        bad  = _call_model_raw(sys_p_bad,  usr_p_bad,  model, effective_budget("autowrite_scorer", model))
        if good.get("error") or bad.get("error"):
            yield BenchMetric(f"{model}::status", "error", "",
                              note=(good.get("error") or bad.get("error") or "")[:80])
            continue

        # Parse both
        from sciknow.core.wiki_ops import _find_json_block, _strip_thinking
        def _parse(resp):
            cleaned = _strip_thinking(resp["content"]).strip()
            block = _find_json_block(cleaned)
            if not block:
                return None
            try:
                return json.loads(block, strict=False)
            except Exception:
                return None

        data_good = _parse(good)
        data_bad  = _parse(bad)

        good_overall = data_good.get("overall", -1) if data_good else -1
        bad_overall  = data_bad.get("overall", -1)  if data_bad  else -1
        json_valid_both = (data_good is not None) and (data_bad is not None)
        ranks_correctly = (json_valid_both
                           and isinstance(good_overall, (int, float))
                           and isinstance(bad_overall, (int, float))
                           and good_overall > bad_overall)

        yield BenchMetric(f"{model}::elapsed_s_good",
                          round(good["elapsed_s"], 1), "s")
        yield BenchMetric(f"{model}::elapsed_s_bad",
                          round(bad["elapsed_s"], 1), "s")
        yield BenchMetric(f"{model}::json_valid_both",
                          1 if json_valid_both else 0, "bool")
        yield BenchMetric(f"{model}::good_overall",
                          good_overall if isinstance(good_overall, (int, float)) else 0,
                          "score")
        yield BenchMetric(f"{model}::bad_overall",
                          bad_overall if isinstance(bad_overall, (int, float)) else 0,
                          "score")
        yield BenchMetric(f"{model}::ranks_correctly",
                          1 if ranks_correctly else 0, "bool",
                          note="good draft scored higher than bad draft")
        if json_valid_both and isinstance(good_overall, (int, float)) and isinstance(bad_overall, (int, float)):
            yield BenchMetric(f"{model}::ranking_gap",
                              round(float(good_overall) - float(bad_overall), 3),
                              "Δscore",
                              note="bigger = clearer discrimination")


# ════════════════════════════════════════════════════════════════════════
# Task 6 — ask synthesize (multi-paper prose)
# ════════════════════════════════════════════════════════════════════════


def b_quality_ask_synthesize() -> Iterable[BenchMetric]:
    """Multi-paper synthesis on a pinned topic. Uses the full retrieval
    + synthesis pipeline so retrieval quality is controlled for across
    models (only the LLM differs)."""
    from sciknow.rag.prompts import synthesis
    from sciknow.retrieval import context_builder, hybrid_search
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import get_client as _qdrant_client

    topic = CANDIDATE_TOPICS[0]

    # Retrieve once; reuse for every candidate to isolate LLM quality.
    # The real retrieval pipeline takes (query, qdrant_client, session)
    # — same shape as `sciknow ask` uses.
    with get_session() as session:
        qdrant = _qdrant_client()
        candidates = hybrid_search.search(
            query=topic, qdrant_client=qdrant, session=session,
            candidate_k=50,
        )
        if not candidates:
            skip(f"no retrieval hits for '{topic}' — seed the corpus first")
        results = context_builder.build(candidates[:8], session)

    sys_p, usr_p = synthesis(topic=topic, results=results)
    installed = _installed_models()
    outputs: dict[str, str] = {}
    source_text = "\n".join(r.content for r in results)

    for model in CANDIDATE_MODELS:
        if _skip_if_model_missing(model, installed):
            yield BenchMetric(f"{model}::status", "not-installed", "")
            continue
        resp = _call_model_raw(sys_p, usr_p, model, effective_budget("ask_synthesize", model))
        if resp.get("error"):
            yield BenchMetric(f"{model}::status", "error", "",
                              note=resp["error"][:80])
            continue

        content_raw = resp["content"]
        # Phase 54.6.90 — strip leaked <think> before scoring; cascade
        # the cleaned text into faithfulness + ALCE citation quality
        # below so the metrics reflect the actual answer.
        content, leaked_as, words = _clean_and_count(content_raw)
        outputs[model] = content

        yield BenchMetric(f"{model}::elapsed_s",
                          round(resp["elapsed_s"], 1), "s")
        yield BenchMetric(f"{model}::words", words, "words")
        yield BenchMetric(f"{model}::thinking_leaked",
                          1 if leaked_as else 0, "bool")

        faith = faithfulness_score(content, source_text)
        yield BenchMetric(f"{model}::faithfulness_mean",
                          faith["mean"], "prob")
        yield BenchMetric(f"{model}::hallucinated_pct",
                          faith["hallucinated_pct"], "%")

        source_chunks = [r.content for r in results]
        cq = citation_quality_alce(content, source_chunks)
        if cq["citation_recall"] is not None:
            yield BenchMetric(f"{model}::citation_recall",
                              cq["citation_recall"], "%")
            yield BenchMetric(f"{model}::citation_precision",
                              cq["citation_precision"], "%")

    if len([o for o in outputs.values() if o]) >= 2:
        judgments = _pairwise_win_rates(
            outputs, topic=f"Synthesis of corpus evidence on '{topic}'",
            source_preview=source_text,
        )
        for model, r in judgments.items():
            if r["win_rate"] is not None:
                yield BenchMetric(f"{model}::judge_win_rate",
                                  r["win_rate"], "%")


# ════════════════════════════════════════════════════════════════════════
# Task 7 — wiki consensus (structured JSON)
# ════════════════════════════════════════════════════════════════════════


def b_quality_wiki_consensus() -> Iterable[BenchMetric]:
    """wiki consensus: JSON validity, claim count range, presence of
    supporting + contradicting paper lists. No faithfulness (the output
    IS derived from the retrieved triples/summaries, not prose about
    source chunks)."""
    from sciknow.core.wiki_ops import consensus_map

    topic = CANDIDATE_TOPICS[0]
    installed = _installed_models()

    for model in CANDIDATE_MODELS:
        if _skip_if_model_missing(model, installed):
            yield BenchMetric(f"{model}::status", "not-installed", "")
            continue

        t0 = time.monotonic()
        consensus_event: dict | None = None
        completed_event: dict | None = None
        error: str | None = None
        try:
            for ev in consensus_map(topic, model=model):
                if ev.get("type") == "consensus":
                    consensus_event = ev
                elif ev.get("type") == "completed":
                    completed_event = ev
                elif ev.get("type") == "error":
                    error = ev.get("message", "")
                    break
        except Exception as exc:
            error = str(exc)
        elapsed = time.monotonic() - t0

        if error and not consensus_event:
            yield BenchMetric(f"{model}::status", "error", "", note=error[:80])
            yield BenchMetric(f"{model}::elapsed_s", round(elapsed, 1), "s")
            continue

        data = (consensus_event or {}).get("data", {}) or {}
        claims = data.get("claims") or []
        # Quality checks
        valid_shape = (
            isinstance(data, dict)
            and isinstance(claims, list)
            and all(isinstance(c, dict) for c in claims)
        )
        has_supporting = all(isinstance(c.get("supporting_papers"), list) for c in claims)
        has_contradicting = all(isinstance(c.get("contradicting_papers"), list) for c in claims)
        n_consensus_levels = len({c.get("consensus_level") for c in claims})

        yield BenchMetric(f"{model}::elapsed_s", round(elapsed, 1), "s")
        yield BenchMetric(f"{model}::n_claims", len(claims), "claims")
        yield BenchMetric(f"{model}::claims_in_range",
                          1 if 2 <= len(claims) <= 8 else 0, "bool",
                          note="ok if claims ∈ [2, 8]")
        yield BenchMetric(f"{model}::json_valid", 1 if valid_shape else 0, "bool")
        yield BenchMetric(f"{model}::has_sup_con_fields",
                          1 if (has_supporting and has_contradicting) else 0,
                          "bool")
        yield BenchMetric(f"{model}::n_consensus_levels",
                          n_consensus_levels, "levels",
                          note="strong/moderate/weak/contested variety")
        yield BenchMetric(f"{model}::has_summary",
                          1 if isinstance(data.get("summary"), str) and data["summary"] else 0,
                          "bool")


# ════════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════════


QUALITY_BENCHES: list[tuple[str, Callable[[], Iterable[BenchMetric]]]] = [
    # Writing-quality (prose) tasks — the headline focus
    ("quality", b_quality_wiki_summary),
    ("quality", b_quality_wiki_polish),
    ("quality", b_quality_autowrite_writer),
    ("quality", b_quality_book_review),
    ("quality", b_quality_ask_synthesize),
    # Structured-output tasks that gate writing-quality
    ("quality", b_quality_autowrite_scorer),
    ("quality", b_quality_wiki_consensus),
]
