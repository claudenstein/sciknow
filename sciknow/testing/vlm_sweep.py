"""Phase 54.6.74 (#1b) — vision-LLM sweep bench.

Mirror of ``model_sweep.py`` for the VLM role (figure + chart captioning).
Same shape: fixed inputs, every candidate model runs each input once,
emits a comparable metrics row per model. Also does a text-LLM
pairwise judge so we get a win-rate column that's directly comparable
to the text-LLM quality bench.

Scoring axes specific to VLM captioning for a scientific corpus:
 - ``elapsed_s`` / ``words`` — raw speed and verbosity
 - ``specificity`` — regex count of concrete domain terms
   (unit tokens like "W/m²", "ppm", "mg/m³", chart-type names like
   "bar chart", "scatter plot", axis labels)
 - ``hedging_per_100w`` — same "may/might/suggest" counter used in
   quality.py; high hedging indicates the VLM is unsure of what the
   image shows
 - ``judge_win_rate`` — pairwise text-LLM judge asks *"which caption
   is more specific and useful for retrieving this paper later?"* The
   judge never sees the image — it only compares caption quality for
   the SAME figure across models, so higher specificity + better
   domain vocab wins.

Runtime
-------
With 5 VLMs × 15 figures × ~20s/caption + ~15s cold-load/model =
~30-40 min captioning + ~5-10 min pairwise judge = ~45 min total.
"""
from __future__ import annotations

import json
import logging
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sciknow.testing.bench import BenchMetric, skip

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════


CANDIDATE_VLMS: list[str] = [
    # Order is display order, not priority. Any model not in
    # ``ollama list`` is gracefully reported as ``not-installed`` and
    # contributes no scoring rows.
    "qwen2.5vl:7b",          # ~6 GB Q4  — fast baseline, co-resident
    "minicpm-v:8b",          # ~5 GB Q4  — efficient, chart-strong
    "llama3.2-vision:11b",   # ~8 GB Q4  — Meta, general
    "internvl3:14b",         # ~15 GB Q4 — document-focused
    "qwen2.5vl:32b",         # ~19 GB Q4 — MinerU-lineage, needs LLM unloaded
]


DEFAULT_N_FIGURES = 15
MIN_CAPTION_WORDS = 20
MAX_CAPTION_WORDS = 200


# Regexes for the specificity counter. These are *crude* — they favor
# recall over precision — but the relative counts across models on the
# same figure set are the signal we care about, not absolute specificity.
_UNIT_PATTERN = re.compile(
    # Captures unit tokens with or without a numeric prefix — presence
    # alone is informative ("outgoing longwave radiation (W/m²)" gets a
    # unit-hit even though no specific value is mentioned).
    r"(?:W/m[²2]|W\s*m-?[²2]|ppm|ppb|ppt|mg/m[³3]|kg/m[³3]|°C|°F|"
    r"hPa|kPa|Pa|m/s|km/s|mm/yr|m/yr|Sv\b|TSI|SST|OLR|ECS|TCR)",
    re.IGNORECASE,
)
_PLOT_TYPE_PATTERN = re.compile(
    r"\b(bar\s+chart|line\s+(?:plot|graph)|scatter\s*(?:plot)?|"
    r"heat\s?map|heatmap|histogram|box\s?plot|time[\s-]?series|"
    r"contour|stacked\s+bar|pie\s+chart|log[- ]?log|semi[- ]?log|"
    r"error\s+bar|confidence\s+interval)\b",
    re.IGNORECASE,
)
_AXIS_PATTERN = re.compile(
    r"\b(x[-\s]?axis|y[-\s]?axis|horizontal\s+axis|vertical\s+axis|"
    r"\baxis\b|legend|units?\s*(?:are|:|in)|measured\s+in)\b",
    re.IGNORECASE,
)
_HEDGE_PATTERN = re.compile(
    r"\b(may|might|could|suggest|appear|seem|possibly|perhaps|"
    r"unclear|uncertain|likely|probably|seems|appears)\b",
    re.IGNORECASE,
)


# ════════════════════════════════════════════════════════════════════════
# Figure-set generation (one-time)
# ════════════════════════════════════════════════════════════════════════


def _bench_dir() -> Path:
    from sciknow.config import settings
    d = Path(settings.data_dir) / "bench"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _figures_path() -> Path:
    return _bench_dir() / "vlm_sweep_figures.json"


def _captions_path() -> Path:
    return _bench_dir() / "vlm_sweep_captions.json"


def _fetch_sample_figures(n: int, seed: int = 42) -> list[dict]:
    """Return ``n`` figure / chart rows with resolvable on-disk assets.

    We prefer diversity: figures AND charts from different documents,
    with non-empty captions where possible. Skips rows whose
    ``asset_path`` no longer resolves (MinerU cleanup etc.).
    """
    from sqlalchemy import text as sql_text
    from sciknow.storage.db import get_session
    from sciknow.core.visuals_caption import resolve_asset_path

    with get_session() as session:
        session.execute(sql_text(f"SELECT setseed({(seed % 1000) / 1000:.3f})"))
        rows = session.execute(sql_text("""
            SELECT v.id::text, v.document_id::text, v.kind, v.asset_path,
                   v.caption, v.figure_num, pm.title, pm.year
            FROM visuals v
            JOIN paper_metadata pm ON pm.document_id = v.document_id
            WHERE v.asset_path IS NOT NULL
              AND v.kind IN ('figure', 'chart')
            ORDER BY random()
            LIMIT :over
        """), {"over": n * 5}).fetchall()

    out: list[dict] = []
    seen_docs: set[str] = set()
    for r in rows:
        img_path = resolve_asset_path(r[1], r[3])
        if img_path is None:
            continue
        # Prefer diversity across documents — only let a doc contribute
        # two figures max (avoid sampling 15 figures all from one paper).
        if list(seen_docs).count(r[1]) >= 2:
            continue
        seen_docs.add(r[1])
        out.append({
            "visual_id": r[0],
            "document_id": r[1],
            "kind": r[2],
            "asset_path": r[3],
            "resolved_path": str(img_path),
            "caption": r[4] or "",
            "figure_num": r[5] or "",
            "paper_title": r[6] or "",
            "paper_year": r[7],
        })
        if len(out) >= n:
            break
    return out


def generate_figure_set(n: int = DEFAULT_N_FIGURES, seed: int = 42,
                        verbose: bool = True) -> Path:
    """One-time: sample ``n`` figures from the corpus and persist the
    pinned set. Overwrites any existing file (use the ``bench
    vlm-sweep`` command path to avoid regenerating inadvertently)."""
    path = _figures_path()
    figures = _fetch_sample_figures(n, seed=seed)
    if not figures:
        raise RuntimeError(
            "No figures with on-disk images found — run "
            "`sciknow corpus extract-visuals` first and make sure "
            "mineru_output/ is intact."
        )
    with path.open("w") as f:
        json.dump(figures, f, indent=2)
    if verbose:
        print(f"✓ Pinned {len(figures)} figures to {path}")
    return path


def load_figure_set() -> list[dict]:
    path = _figures_path()
    if not path.exists():
        return []
    return json.loads(path.read_text())


# ════════════════════════════════════════════════════════════════════════
# Metric helpers
# ════════════════════════════════════════════════════════════════════════


def _count_words(text: str) -> int:
    return len([w for w in (text or "").split() if w.strip()])


def _specificity_score(text: str) -> tuple[int, int, int, int]:
    """Return (unit_hits, plot_type_hits, axis_hits, total_specific) for
    one caption. The counters overlap conceptually (a caption can mention
    both axes and units) — totals are capped per-category to avoid
    rewarding repetition."""
    t = text or ""
    u = min(5, len(_UNIT_PATTERN.findall(t)))
    p = min(3, len(_PLOT_TYPE_PATTERN.findall(t)))
    a = min(4, len(_AXIS_PATTERN.findall(t)))
    return u, p, a, u + p + a


def _hedging_per_100w(text: str) -> float:
    wc = _count_words(text)
    if wc == 0:
        return 0.0
    hedges = len(_HEDGE_PATTERN.findall(text or ""))
    return round(100.0 * hedges / wc, 2)


# ════════════════════════════════════════════════════════════════════════
# VLM captioning
# ════════════════════════════════════════════════════════════════════════


def _vlm_sampling_for(model: str) -> dict:
    """Phase 54.6.85 — per-VLM sampling + num_predict recommended by
    the model's own HF card. VLMs don't emit CoT, so there's no
    thinking-budget issue here — but temperature / top_p / top_k
    still matter for description quality.

    Sources (match the 54.6.85 text-LLM research methodology):
      - Qwen2.5-VL HF card: temp 0.2-0.3 for description, top_p 0.9.
      - InternVL3 paper + model card: temp 0.3, top_p 0.9, top_k 50.
      - Llama-3.2-Vision HF card: temp 0.6, top_p 0.9.
      - MiniCPM-V card: temp 0.7, top_p 0.8.

    All VLMs get num_predict=500 — a paragraph-length caption
    (120-200 words) plus headroom. Pre-54.6.85 was 300 which
    occasionally truncated mid-sentence.
    """
    m = (model or "").lower()
    if "qwen2.5vl" in m or "qwen2.5-vl" in m or "qwen2vl" in m:
        return {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "num_predict": 500}
    if "internvl" in m:
        return {"temperature": 0.3, "top_p": 0.9, "top_k": 50, "num_predict": 500}
    if "llama3.2-vision" in m or "llama-3.2-vision" in m:
        return {"temperature": 0.6, "top_p": 0.9, "top_k": 50, "num_predict": 500}
    if "minicpm-v" in m or "minicpm" in m:
        return {"temperature": 0.7, "top_p": 0.8, "top_k": 40, "num_predict": 500}
    # Default
    return {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "num_predict": 500}


def _caption_one(
    client, model: str, image_path: str, kind: str, existing_caption: str
) -> tuple[str | None, float]:
    """Return (caption_or_None, elapsed_s). None on any error.

    Phase 54.6.85 — sampling now per-model via ``_vlm_sampling_for``
    instead of a hardcoded temp=0.2/num_predict=300. The latter
    truncated Qwen2.5-VL-32B's detailed captions mid-sentence.
    """
    from sciknow.core.visuals_caption import PROMPT_SYSTEM, PROMPT_USER
    user_prompt = PROMPT_USER.format(
        kind=kind,
        existing_caption=(existing_caption or "").strip() or "(none)",
    )
    sampling = _vlm_sampling_for(model)
    t0 = time.monotonic()
    try:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": user_prompt,
                 "images": [image_path]},
            ],
            options={
                "temperature": sampling["temperature"],
                "top_p":       sampling["top_p"],
                "top_k":       sampling["top_k"],
                "num_predict": sampling["num_predict"],
            },
            keep_alive=-1,
        )
        cap = (resp.get("message") or {}).get("content", "").strip()
        if not cap:
            return None, time.monotonic() - t0
        return cap, time.monotonic() - t0
    except Exception as exc:
        logger.warning("VLM %s caption failed: %s", model, exc)
        return None, time.monotonic() - t0


# ════════════════════════════════════════════════════════════════════════
# Pairwise judge (text LLM, doesn't see image)
# ════════════════════════════════════════════════════════════════════════


_JUDGE_SYSTEM = (
    "You are comparing two captions written for the SAME scientific "
    "figure. You have not seen the figure. Your job: pick which caption "
    "would be more useful if someone were searching a corpus for this "
    "specific paper by query.\n\n"
    "Prefer the caption that is more SPECIFIC (names concrete axes, "
    "units, plot type, domain concepts) and more GROUNDED (fewer "
    "'may/might/seems' hedges without justification).\n\n"
    "Respond with exactly one token: A, B, or TIE."
)


def _judge_pair(client, judge_model: str, caption_a: str,
                caption_b: str) -> str:
    """Return 'A', 'B', or 'TIE' (uppercase)."""
    try:
        user = (
            "Caption A:\n"
            + caption_a.strip()
            + "\n\nCaption B:\n"
            + caption_b.strip()
            + "\n\nYour answer (A, B, or TIE):"
        )
        resp = client.chat(
            model=judge_model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0.0, "num_predict": 8},
            keep_alive=-1,
        )
        out = (resp.get("message") or {}).get("content", "").strip().upper()
        # First letter of response.
        for ch in out:
            if ch in ("A", "B"):
                return ch
            if ch == "T":
                return "TIE"
        return "TIE"
    except Exception as exc:
        logger.warning("judge call failed: %s", exc)
        return "TIE"


# ════════════════════════════════════════════════════════════════════════
# Bench function — the one everything goes through
# ════════════════════════════════════════════════════════════════════════


def b_vlm_sweep() -> Iterable[BenchMetric]:
    """Run every locally-installed VLM candidate over the pinned figure
    set, collect metrics, and compute pairwise judge-model win rates.
    Persists captions to ``data/bench/vlm_sweep_captions.json`` for
    inspection.

    Skips (with a clean metric row) when the figure set hasn't been
    generated yet.
    """
    figures = load_figure_set()
    if not figures:
        yield BenchMetric("status", "no-figure-set", "",
                          note=(f"run `sciknow bench-vlm-gen` to create "
                                f"{_figures_path().name}"))
        return

    import ollama
    from sciknow.config import settings
    client = ollama.Client(host=settings.ollama_host)

    try:
        installed = {m.model for m in client.list().models}
    except Exception as exc:
        yield BenchMetric("status", "ollama-unreachable", "",
                          note=str(exc))
        return

    # ── 1. Per-VLM captioning ─────────────────────────────────────────
    captions_by_vlm: dict[str, dict[str, str]] = {}
    metrics_by_vlm: dict[str, dict] = {}

    for vlm in CANDIDATE_VLMS:
        if vlm not in installed:
            yield BenchMetric(f"{vlm}::status", "not-installed", "",
                              note=f"`ollama pull {vlm}` to include")
            continue
        captions_by_vlm[vlm] = {}
        metrics_by_vlm[vlm] = {
            "elapsed_s": [],
            "words": [],
            "unit_hits": [],
            "plot_type_hits": [],
            "axis_hits": [],
            "specificity_total": [],
            "hedging_per_100w": [],
            "empty_count": 0,
        }
        for fig in figures:
            cap, dt = _caption_one(
                client, vlm, fig["resolved_path"], fig["kind"],
                fig["caption"],
            )
            if cap is None:
                metrics_by_vlm[vlm]["empty_count"] += 1
                captions_by_vlm[vlm][fig["visual_id"]] = ""
                continue
            captions_by_vlm[vlm][fig["visual_id"]] = cap
            u, p, a, tot = _specificity_score(cap)
            metrics_by_vlm[vlm]["elapsed_s"].append(dt)
            metrics_by_vlm[vlm]["words"].append(_count_words(cap))
            metrics_by_vlm[vlm]["unit_hits"].append(u)
            metrics_by_vlm[vlm]["plot_type_hits"].append(p)
            metrics_by_vlm[vlm]["axis_hits"].append(a)
            metrics_by_vlm[vlm]["specificity_total"].append(tot)
            metrics_by_vlm[vlm]["hedging_per_100w"].append(
                _hedging_per_100w(cap)
            )

    # Persist all captions for inspection.
    _captions_path().write_text(json.dumps({
        "figures": figures,
        "captions": captions_by_vlm,
    }, indent=2))

    # ── 2. Pairwise judging via text LLM ──────────────────────────────
    judge_model = settings.llm_fast_model
    alive_vlms = [v for v in CANDIDATE_VLMS if v in captions_by_vlm]
    wins: dict[str, int] = {v: 0 for v in alive_vlms}
    losses: dict[str, int] = {v: 0 for v in alive_vlms}
    ties: dict[str, int] = {v: 0 for v in alive_vlms}
    pair_count: dict[str, int] = {v: 0 for v in alive_vlms}

    if judge_model and judge_model in installed and len(alive_vlms) >= 2:
        rng = random.Random(42)
        for fig in figures:
            fid = fig["visual_id"]
            # Build per-figure model-caption pool.
            pool = [(v, captions_by_vlm[v].get(fid, "")) for v in alive_vlms]
            pool = [(v, c) for v, c in pool if c]
            if len(pool) < 2:
                continue
            # All unique pairs
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    va, ca = pool[i]
                    vb, cb = pool[j]
                    # Randomize A/B order to cancel position bias.
                    if rng.random() < 0.5:
                        va, vb = vb, va
                        ca, cb = cb, ca
                    verdict = _judge_pair(client, judge_model, ca, cb)
                    pair_count[va] += 1
                    pair_count[vb] += 1
                    if verdict == "A":
                        wins[va] += 1
                        losses[vb] += 1
                    elif verdict == "B":
                        wins[vb] += 1
                        losses[va] += 1
                    else:
                        ties[va] += 1
                        ties[vb] += 1
    else:
        yield BenchMetric("judge", "skipped", "",
                          note=(f"judge {judge_model} unavailable or "
                                f"<2 VLMs responded"))

    # ── 3. Emit per-VLM metric rows ───────────────────────────────────
    for vlm in alive_vlms:
        m = metrics_by_vlm[vlm]
        n_cap = len(m["elapsed_s"])
        if n_cap == 0:
            yield BenchMetric(f"{vlm}::status", "all-empty", "",
                              note="VLM returned no captions")
            continue
        yield BenchMetric(f"{vlm}::elapsed_mean_s",
                          round(statistics.mean(m["elapsed_s"]), 2), "s")
        yield BenchMetric(f"{vlm}::words_mean",
                          round(statistics.mean(m["words"]), 1), "words")
        yield BenchMetric(f"{vlm}::specificity_mean",
                          round(statistics.mean(m["specificity_total"]), 2), "hits",
                          note="sum of unit + plot-type + axis regex hits per caption")
        yield BenchMetric(f"{vlm}::hedging_per_100w",
                          round(statistics.mean(m["hedging_per_100w"]), 2), "/100w")
        yield BenchMetric(f"{vlm}::empty_captions",
                          m["empty_count"], "count")
        if pair_count.get(vlm, 0) > 0:
            wr = 100.0 * wins[vlm] / pair_count[vlm]
            yield BenchMetric(f"{vlm}::judge_win_rate",
                              round(wr, 1), "%",
                              note=(f"{wins[vlm]}W / {losses[vlm]}L / "
                                    f"{ties[vlm]}T out of {pair_count[vlm]}"))

    yield BenchMetric("total_figures", len(figures), "figures")
    yield BenchMetric("captions_persisted", "yes", "",
                      note=f"inspect {_captions_path()}")


SWEEP_BENCHES: list[tuple[str, callable]] = [
    ("vlm", b_vlm_sweep),
]
