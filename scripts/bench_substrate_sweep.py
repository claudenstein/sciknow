"""Phase 55.V10/V15 — llama-server flag-sweep bench harness.

Single-knob sweeps + a 2-knob matrix mode. Default baseline matches
the expert's recommended Qwen3.6-27B 24-GB-3090 config (Q4_K_M,
ctx 262144, np 1, flash-attn on, q4_0 KV cache); sweeps move
**from** that baseline so we test against a known-good starting
point rather than fresh-from-defaults.

Per (role, knob, value) we capture:

  - prompt-eval TPS  (server-reported)
  - decode TPS       (server-reported)
  - peak VRAM (MiB)  (nvidia-smi sample at end of generation)
  - first-token latency (s, wall)
  - cold-load time   (s, wall — process spawn → /health=200)
  - failure mode     (OOM / context-overflow / crash / OK)

Discipline: ONE knob per **single-mode** run. The 2-knob matrix
mode is explicit (--matrix-knob) and produces a full grid.

Three workloads:
  - typical:           ~5K input + 1500 output (autowrite write call)
  - large:             ~18K input + 1500 output (push ctx upper bound)
  - autowrite-section: real autowrite_section_stream over a fixed
                        seed section — slow (~10 min) but captures
                        the autowrite scorer's overall + groundedness
                        so we can answer "does this knob change
                        autowrite quality, not just throughput?"

Results land in ``data/bench/substrate_sweep/<timestamp>.jsonl``
plus a Rich-rendered summary on stdout. Re-runs append.

Usage examples:

  # Phase 55.V15 priority #1 — verify the q4_0 KV-cache claim.
  # The expert claim: q4_0 KV at 262K = 21 GB total at 40 t/s flat.
  # Sweeping cache-type at the expert's full 262K context isolates
  # the cache-type effect at the working-extreme — exactly where it
  # matters and where the q8 "3× slower" claim should manifest if true.
  uv run python scripts/bench_substrate_sweep.py \\
      --role writer --knob cache-type \\
      --values f16,q8_0,q4_0 \\
      --baseline expert --workload typical

  # Phase 55.V15 priority #2 — paired (cache-type × ctx-size) matrix.
  # Locates the speed cliffs (if any) per cache type across the
  # context-window range we actually use → 262K.
  uv run python scripts/bench_substrate_sweep.py \\
      --role writer \\
      --matrix-knob cache-type --matrix-values f16,q8_0,q4_0 \\
      --knob ctx-size --values 16384,24576,65536,131072,262144 \\
      --baseline expert --workload typical

  # Phase 55.V15 priority #3 — quality probe. Slow (~10 min/run);
  # answers "does q4_0 KV cost autowrite groundedness".
  uv run python scripts/bench_substrate_sweep.py \\
      --role writer --knob cache-type \\
      --values f16,q8_0,q4_0 \\
      --workload autowrite-section --reps 1 \\
      --autowrite-section-slug the_engine_of_the_sun

  # Single-knob sweep starting from current production defaults
  # (writer ctx 24576, fp16 KV) — baseline=current.
  uv run python scripts/bench_substrate_sweep.py \\
      --role writer --knob ctx-size \\
      --values 16384,20480,24576,32768,65536 \\
      --baseline current --workload large

Read scripts/bench_substrate_sweep.py top-of-file for the full
methodology; docs/BENCH_OPTIMIZATION_PLAN.md captures the planning
doc and how to interpret the table.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import httpx

# Ensure the project root is on sys.path so `uv run python scripts/...`
# can find `sciknow.*` regardless of cwd.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rich.console import Console
from rich.table import Table

from sciknow.config import settings
from sciknow.infer import server as srv

console = Console()


# ── Synthetic workloads ────────────────────────────────────────────────────────
#
# Single fixed seed across runs so the only thing that changes is the
# llama-server flag. Two shapes:
#   - "typical": ~5K-token prompt + 1500-token completion. Mirrors the
#     autowrite write/revise call.
#   - "large":   ~18K-token prompt + 1500-token completion. Pushes the
#     ctx-size upper bound; relevant for the recently-fixed
#     "request exceeds ctx" failures on text-heavy sections.

_SYS_PROMPT = (
    "You are a scientific writer drafting one section of a research book "
    "on global cooling and solar minima. Write in a measured, evidence-"
    "grounded tone with appropriate hedging. Output prose only — no "
    "headers, no bullet lists."
)

_TOPIC = "the_role_of_the_maunder_minimum_in_long_term_climate_forcing"

# ~360-token chunk repeated to reach the target prompt size. Content is
# real enough to look like retrieval evidence, but doesn't need to be
# scientifically accurate — we're benching wall-clock, not quality.
_CHUNK_TEMPLATE = (
    "[chunk {i}] (Eddy 1976, climate variability)\n"
    "The Maunder Minimum, conventionally dated 1645–1715, was a 70-year "
    "interval of vanishing sunspot activity coinciding with the coldest "
    "phase of the Little Ice Age in the Northern Hemisphere. Reconstructed "
    "10Be cosmogenic isotope flux peaks in this window indicate weakened "
    "solar magnetic shielding. Tree-ring derived 14C records corroborate "
    "the magnetic reduction. Ground-based temperature reconstructions from "
    "the Central England series and Alpine glacier advance show winter-"
    "season anomalies of −0.5 to −1.2°C relative to the 20th-century mean. "
    "Ocean-atmosphere coupling via the AMO is hypothesized to amplify the "
    "small radiative forcing into a regional climate signal of larger "
    "magnitude. Model-based attribution remains contested.\n\n"
)


def _build_prompt(approx_input_tokens: int) -> str:
    """Stuff `_CHUNK_TEMPLATE` until the prompt reaches the target size.

    Phase 55.V19 — empirical token count of `_CHUNK_TEMPLATE` is **175**
    tokens (Qwen2.5 tokenizer; verified 2026-04-28). Earlier estimate
    of 90 was off by ~2× and produced prompts much larger than
    advertised — `large` workload claimed 18K tokens but emitted
    ~35.6K, overflowing ctx=24576/32768 cells with HTTP 400.
    """
    chunks_needed = max(1, approx_input_tokens // 175)
    body = "".join(_CHUNK_TEMPLATE.format(i=i) for i in range(chunks_needed))
    return (
        f"Topic: {_TOPIC}\n\n"
        f"Retrieved evidence (use only what is supported below):\n\n"
        f"{body}\n\n"
        "Task: write a ~1500-word section grounded in the evidence above. "
        "Cite chunks as [chunk N]. Begin now.\n"
    )


WORKLOADS = {
    "typical": {
        # ~5K input tokens + 1500 output → matches autowrite write/revise.
        "approx_input_tokens": 5000,
        "max_predict": 1500,
        "kind": "synthetic",
    },
    "large": {
        # ~18K input tokens — pushes ctx-size near 24K limit.
        "approx_input_tokens": 18000,
        "max_predict": 1500,
        "kind": "synthetic",
    },
    "autowrite-section": {
        # Real autowrite_section_stream over a fixed-seed section.
        # SLOW (~10 min/run), but captures the autowrite scorer's
        # overall + groundedness so we can answer "does this knob
        # change quality, not just throughput?". Pair with --reps 1
        # unless you have GPU time to burn.
        "kind": "autowrite",
    },
}


# ── Expert / current baselines ─────────────────────────────────────────────────
#
# Phase 55.V15 — when sweeping a single knob, every OTHER role param
# should be set to a defensible, known-good baseline. Two presets:
#
#   "expert"  — community-recommended Qwen3.6-27B 24 GB 3090 config.
#               Comes from a power-user posting (also matches the
#               @Punch_Taylor 4090 bench config we cited in the
#               Phase 54.6.303 commit). The CLAIM is q4_0 KV at 262K
#               fits in 21 GB at 40 t/s flat curve. The bench's job
#               is to verify or falsify on this hardware + this
#               GGUF (Q4_K_XL vs Q4_K_M is slightly bigger; that's
#               the only knob we don't sweep).
#
#   "current" — whatever's currently in ROLE_DEFAULTS at the moment
#               the script starts. Useful for "would this delta
#               regress my production config?" sweeps.
#
# Default: --baseline expert. Verify-the-claim sweeps start there
# and only change the swept knob; if the expert claim is right, the
# unmoved-rows are at the working-extreme already.
_EXPERT_BASELINE = {
    "writer": {
        "ctx_size": 262144,            # 262K — the headline claim
        "n_gpu_layers": 999,           # -ngl 99 (offload everything)
        "parallel": 1,                 # -np 1 (single user slot)
        "extra_flags": [
            "--cont-batching",
            "--flash-attn", "on",      # -fa on
            "--cache-type-k", "q4_0",  # the load-bearing flag
            "--cache-type-v", "q4_0",
        ],
    },
    "scorer": {
        # Same shape applied to gemma-4-31B (~18 GB Q4_1 on disk).
        # Headroom math: gemma 18 GB + 262K@q4_0 KV ~5 GB = ~23 GB.
        # Tighter than the writer; expect 262K to OOM on the 3090
        # for the scorer. The bench will confirm where it cliffs.
        "ctx_size": 262144,
        "n_gpu_layers": 999,
        "parallel": 1,
        "extra_flags": [
            "--cont-batching",
            "--flash-attn", "on",
            "--cache-type-k", "q4_0",
            "--cache-type-v", "q4_0",
        ],
    },
}


def _apply_baseline(role: str, baseline: str) -> None:
    """Mutate ROLE_DEFAULTS[role] in place to the chosen baseline."""
    if baseline == "current":
        return  # whatever ROLE_DEFAULTS happens to have wins
    if baseline != "expert":
        raise ValueError(f"unknown baseline {baseline!r}")
    base = _EXPERT_BASELINE.get(role)
    if base is None:
        return
    cfg = srv.ROLE_DEFAULTS[role]
    cfg["ctx_size"] = base["ctx_size"]
    cfg["n_gpu_layers"] = base["n_gpu_layers"]
    cfg["parallel"] = base["parallel"]
    cfg["extra_flags"] = list(base["extra_flags"])


# ── llama-server config patching ───────────────────────────────────────────────


def _patch_role_for_run(role: str, knob: str, value: str) -> dict:
    """Mutate ROLE_DEFAULTS[role] in place for the upcoming run.
    Returns a backup dict so the caller can restore on tear-down.

    Handled knobs (one --knob per single-mode script invocation; the
    matrix mode applies _two_ patches in sequence around the same
    backup):
      ctx-size       → cfg["ctx_size"]
      batch-size     → extra_flags --batch-size <v> --ubatch-size <v/4>
      cache-type     → extra_flags --cache-type-k <v> --cache-type-v <v>
      n-gpu-layers   → cfg["n_gpu_layers"]
      flash-attn     → extra_flags --flash-attn <on|off>
      parallel       → cfg["parallel"]
    """
    backup = {
        "ctx_size": srv.ROLE_DEFAULTS[role].get("ctx_size"),
        "n_gpu_layers": srv.ROLE_DEFAULTS[role].get("n_gpu_layers"),
        "parallel": srv.ROLE_DEFAULTS[role].get("parallel"),
        "extra_flags": list(srv.ROLE_DEFAULTS[role].get("extra_flags") or []),
    }

    cfg = srv.ROLE_DEFAULTS[role]

    if knob == "ctx-size":
        cfg["ctx_size"] = int(value)
    elif knob == "n-gpu-layers":
        cfg["n_gpu_layers"] = int(value)
    elif knob == "parallel":
        cfg["parallel"] = int(value)
    elif knob == "batch-size":
        bs = int(value)
        ubs = max(bs // 4, 512)
        # Strip any prior batch-size / ubatch-size + append our values.
        flags = [f for f in cfg["extra_flags"]
                 if not f.startswith("--batch-size") and not f.startswith("--ubatch-size")]
        # Filter out any values that immediately follow those flags too.
        cleaned: list[str] = []
        skip_next = False
        for f in cfg["extra_flags"]:
            if skip_next:
                skip_next = False
                continue
            if f in ("--batch-size", "--ubatch-size"):
                skip_next = True
                continue
            cleaned.append(f)
        cleaned += ["--batch-size", str(bs), "--ubatch-size", str(ubs)]
        cfg["extra_flags"] = cleaned
    elif knob == "cache-type":
        # Phase 55.V16 — accept either a single quant (paired K=V, the
        # standard pattern) or `K:V` for asymmetric K/V quants.
        # Examples: "q4_0", "q8_0", "f16", "q8_0:q4_0", "f16:q8_0".
        # The asymmetric pattern is the GGML-discussion-#5932 sweet
        # spot for Qwen-family GQA: keep key precision (matters for
        # retrieval-style attention) while halving V cache.
        if ":" in str(value):
            k_quant, v_quant = str(value).split(":", 1)
        else:
            k_quant = v_quant = str(value)
        cleaned: list[str] = []
        skip_next = False
        for f in cfg["extra_flags"]:
            if skip_next:
                skip_next = False
                continue
            if f in ("--cache-type-k", "--cache-type-v"):
                skip_next = True
                continue
            cleaned.append(f)
        cleaned += ["--cache-type-k", k_quant.strip(),
                    "--cache-type-v", v_quant.strip()]
        cfg["extra_flags"] = cleaned
    elif knob == "flash-attn":
        cleaned: list[str] = []
        skip_next = False
        for f in cfg["extra_flags"]:
            if skip_next:
                skip_next = False
                continue
            if f == "--flash-attn":
                skip_next = True
                continue
            cleaned.append(f)
        cleaned += ["--flash-attn", str(value)]
        cfg["extra_flags"] = cleaned
    else:
        raise ValueError(f"Unknown knob: {knob!r}")

    return backup


def _restore_role(role: str, backup: dict) -> None:
    cfg = srv.ROLE_DEFAULTS[role]
    cfg["ctx_size"] = backup["ctx_size"]
    cfg["n_gpu_layers"] = backup["n_gpu_layers"]
    cfg["parallel"] = backup["parallel"]
    cfg["extra_flags"] = list(backup["extra_flags"])


# ── Workload runner ────────────────────────────────────────────────────────────


def _vram_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return -1


def _down_all_roles(grace_s: float = 2.0) -> None:
    """Tear down every role for a clean slate."""
    for r in ("writer", "embedder", "reranker", "scorer", "vlm"):
        try:
            srv.down(r, timeout=10.0)
        except Exception:
            pass
    time.sleep(grace_s)


def _run_single_request(
    role: str, prompt: str, max_predict: int, *,
    timeout_s: float = 300.0,
) -> dict:
    """Issue one /v1/chat/completions and capture timings."""
    url = srv._resolve_url(role).rstrip("/") + "/v1/chat/completions"
    body = {
        "model": getattr(settings,
                          "writer_model_name" if role == "writer" else
                          ("scorer_model_name" if role == "scorer" else
                           f"{role}_model_name"), role),
        "messages": [
            {"role": "system", "content": _SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": max_predict,
        "temperature": 0.2,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=timeout_s) as c:
            r = c.post(url, json=body)
        wall = time.monotonic() - t0
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        d = r.json()
        timings = d.get("timings") or {}
        usage = d.get("usage") or {}
        return {
            "wall_s": wall,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_eval_tps": timings.get("prompt_per_second"),
            "decode_tps": timings.get("predicted_per_second"),
            "prompt_ms": timings.get("prompt_ms"),
            "predicted_ms": timings.get("predicted_ms"),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _run_autowrite_section_workload(
    *, book_id: str, chapter_id: str, section_type: str,
) -> dict:
    """Phase 55.V15 — autowrite-section quality probe.

    Runs ``autowrite_section_stream`` once with --rebuild semantics
    (so we always start from scratch — bench results have to be
    independent of pre-existing draft state). Captures wall time +
    final overall + groundedness + any verification flags from the
    `completed` event.

    The autowrite engine drives its own substrate swaps via Phase
    55.V1's ``activate_phase`` calls — this runner does NOT
    pre-bring-up roles. ROLE_DEFAULTS patching survives the swaps
    because the engine reads ROLE_DEFAULTS at each up() call-site.
    """
    # Import via the book_ops re-export — book_ops.py:5412 deliberately
    # re-exports the autowrite engine entry points so external callers
    # don't trip the autowrite ↔ book_ops circular import (autowrite.py:42
    # has a top-level `from book_ops import ...`; book_ops.py loads
    # autowrite at the bottom only after its own names are all bound).
    from sciknow.core.book_ops import autowrite_section_stream

    t0 = time.monotonic()
    completed: dict | None = None
    section_error: dict | None = None
    iter_scores: list[dict] = []
    plan_coverage: list[dict] = []
    try:
        for evt in autowrite_section_stream(
            book_id=book_id, chapter_id=chapter_id,
            section_type=section_type,
            max_iter=2, target_score=0.99,  # force the loop to run all iters
            auto_expand=False, use_plan=True,
            use_step_back=True, use_cove=True,
            cove_threshold=0.85,
            target_words=None,
            resume_from_draft_id=None,        # rebuild semantics
            include_visuals=False,
            force_resume=False,
        ):
            t = (evt or {}).get("type")
            if t == "completed":
                completed = evt
            elif t == "section_error":
                section_error = evt
            elif t == "scores":
                iter_scores.append(evt)
            elif t == "plan_coverage":
                plan_coverage.append(evt)
    except Exception as exc:
        return {
            "wall_s": time.monotonic() - t0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    wall = time.monotonic() - t0
    return {
        "wall_s": wall,
        "completed": completed,
        "section_error": section_error,
        "final_score": (completed or {}).get("final_score"),
        "iterations": (completed or {}).get("iterations"),
        "word_count": (completed or {}).get("word_count"),
        "iter_scores": iter_scores,
        "plan_coverage": plan_coverage,
    }


def _run_one_config(
    role: str, knob: str, value: str, workload: str,
    *, reps: int = 3, predict: int = 1500,
    matrix_knob: str | None = None, matrix_value: str | None = None,
    autowrite_section_args: dict | None = None,
    baseline: str = "expert",
    ctx_override: int | None = None,
) -> dict:
    """Patch → up → run reps → down → restore. Returns averaged dict.

    Phase 55.V15 — applies ``baseline`` first (default expert), then
    patches the swept knob, then optionally a matrix knob (for the
    paired sweep). All rolled back at tear-down.

    Phase 55.V16b — `ctx_override` pins ctx_size after the baseline is
    applied (and before the swept knob), so a quality probe can hold
    ctx constant at a fittable value (e.g. 65536) while sweeping
    cache-type. Incompatible with `knob == "ctx-size"` (caller checks).
    """
    wcfg = WORKLOADS[workload]
    is_autowrite = wcfg.get("kind") == "autowrite"
    if is_autowrite:
        prompt = ""
        max_predict = predict
    else:
        prompt = _build_prompt(wcfg["approx_input_tokens"])
        max_predict = predict or wcfg["max_predict"]

    _down_all_roles()
    # Apply the chosen baseline FIRST so non-swept params start from a
    # known config, not whatever ROLE_DEFAULTS was at script start.
    _apply_baseline(role, baseline)
    if ctx_override is not None:
        srv.ROLE_DEFAULTS[role]["ctx_size"] = int(ctx_override)
    backup = _patch_role_for_run(role, knob, value)
    if matrix_knob and matrix_value is not None:
        # Stack a second patch on top of the same backup so one
        # restore unwinds both. This is the matrix-mode path.
        m_backup = _patch_role_for_run(role, matrix_knob, matrix_value)
        # Merge backups: earlier _patch_role_for_run already captured
        # the post-baseline state; the second backup captures the
        # post-knob state. We need to restore to the FIRST one, so
        # discard m_backup.
        _ = m_backup
    cold_t0 = time.monotonic()
    cold_s = None
    failure = None
    runs: list[dict] = []
    peak_vram = -1
    try:
        if is_autowrite:
            # Engine handles its own up()/swap. We just patch and run.
            cold_s = 0.0
            for _ in range(reps):
                aw_args = autowrite_section_args or {}
                r = _run_autowrite_section_workload(**aw_args)
                runs.append(r)
                v = _vram_used_mib()
                if v > peak_vram:
                    peak_vram = v
                if "error" in r:
                    failure = r["error"]
                    break
        else:
            srv.up(role, wait=True)
            cold_s = time.monotonic() - cold_t0
            for _ in range(reps):
                r = _run_single_request(role, prompt, max_predict)
                runs.append(r)
                v = _vram_used_mib()
                if v > peak_vram:
                    peak_vram = v
                if "error" in r:
                    failure = r["error"]
                    break
    except Exception as exc:
        failure = f"up_failed: {type(exc).__name__}: {exc}"
    finally:
        _down_all_roles()
        _restore_role(role, backup)

    # Aggregate runs.
    ok_runs = [r for r in runs if "error" not in r]
    def _stat(field):
        vals = [r.get(field) for r in ok_runs if r.get(field) is not None]
        if not vals:
            return None
        return {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }

    out = {
        "role": role,
        "knob": knob,
        "value": value,
        "matrix_knob": matrix_knob,
        "matrix_value": matrix_value,
        "baseline": baseline,
        "ctx_override": ctx_override,
        "workload": workload,
        "approx_input_tokens": wcfg.get("approx_input_tokens"),
        "max_predict": max_predict,
        "reps_attempted": reps,
        "reps_ok": len(ok_runs),
        "cold_load_s": cold_s,
        "peak_vram_mib": peak_vram,
        "wall_s": _stat("wall_s"),
        "prompt_eval_tps": _stat("prompt_eval_tps"),
        "decode_tps": _stat("decode_tps"),
        "prompt_tokens": _stat("prompt_tokens"),
        "completion_tokens": _stat("completion_tokens"),
        "failure": failure,
        "raw_runs": runs,
    }
    # Autowrite-section workload also surfaces the scorer-judged
    # quality so we can detect quality regressions, not just throughput.
    if is_autowrite:
        out["final_score"] = _stat("final_score")
        out["iterations"] = _stat("iterations")
        out["word_count"] = _stat("word_count")
    return out


def _resolve_autowrite_section(slug: str | None) -> dict:
    """Resolve --autowrite-section-slug into book_id + chapter_id.

    If slug is None, picks the first section of the active book. If
    slug is provided, finds the chapter that owns it. Raises if no
    match.
    """
    from sqlalchemy import text as sql_text
    from sciknow.storage.db import get_session
    with get_session() as session:
        book = session.execute(sql_text("""
            SELECT id::text FROM books LIMIT 1
        """)).fetchone()
        if not book:
            raise RuntimeError("no book in active project")
        book_id = book[0]
        if slug:
            ch_rows = session.execute(sql_text("""
                SELECT id::text, sections FROM book_chapters
                WHERE book_id::text = :bid
                ORDER BY number
            """), {"bid": book_id}).fetchall()
            for cid, secs in ch_rows:
                if isinstance(secs, str):
                    try:
                        secs = json.loads(secs)
                    except Exception:
                        secs = []
                if not isinstance(secs, list):
                    continue
                for s in secs:
                    if isinstance(s, dict) and s.get("slug") == slug:
                        return {
                            "book_id": book_id,
                            "chapter_id": cid,
                            "section_type": slug,
                        }
            raise RuntimeError(f"no section with slug {slug!r} in book {book_id}")
        # Default: first section of first chapter.
        ch = session.execute(sql_text("""
            SELECT id::text, sections FROM book_chapters
            WHERE book_id::text = :bid ORDER BY number LIMIT 1
        """), {"bid": book_id}).fetchone()
        if not ch:
            raise RuntimeError("no chapters in active book")
        cid, secs = ch
        if isinstance(secs, str):
            try:
                secs = json.loads(secs)
            except Exception:
                secs = []
        if not isinstance(secs, list) or not secs:
            raise RuntimeError(f"chapter {cid} has no sections")
        first = secs[0]
        if not isinstance(first, dict) or not first.get("slug"):
            raise RuntimeError("first section has no slug")
        return {
            "book_id": book_id, "chapter_id": cid,
            "section_type": first["slug"],
        }


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--role", required=True,
                   choices=("writer", "scorer"),
                   help="Which big-model role to bench.")
    p.add_argument("--knob", required=True,
                   choices=("ctx-size", "batch-size", "cache-type",
                            "n-gpu-layers", "flash-attn", "parallel"),
                   help="Single flag to sweep. ONE per run by design.")
    p.add_argument("--values", required=True,
                   help="Comma-separated values to test (e.g. 16384,24576,32768).")
    p.add_argument("--matrix-knob", default=None,
                   choices=(None, "ctx-size", "batch-size", "cache-type",
                            "n-gpu-layers", "flash-attn", "parallel"),
                   help="Phase 55.V15 — secondary knob for the 2-knob matrix "
                        "mode. Each --value is paired with each --matrix-value, "
                        "producing a full grid (#values × #matrix-values cells). "
                        "Use sparingly; explicit by design.")
    p.add_argument("--matrix-values", default=None,
                   help="Comma-separated values for --matrix-knob. Required "
                        "when --matrix-knob is set.")
    p.add_argument("--workload", default="typical",
                   choices=tuple(WORKLOADS.keys()),
                   help="'typical'/'large' = synthetic prompt; "
                        "'autowrite-section' = real autowrite_section_stream "
                        "with scorer-judged final_score (slow ~10 min/run).")
    p.add_argument("--baseline", default="expert",
                   choices=("expert", "current"),
                   help="Phase 55.V15 — non-swept params start from this "
                        "baseline. 'expert' = community Qwen3.6 24GB config "
                        "(ctx 262144, q4_0 KV, fa on, np 1, ngl 99). "
                        "'current' = whatever ROLE_DEFAULTS holds at start. "
                        "Default: expert.")
    p.add_argument("--ctx-override", type=int, default=None,
                   help="Phase 55.V16b — pin ctx_size after the baseline is "
                        "applied. Use to hold ctx constant while sweeping a "
                        "different knob (e.g. cache-type quality probe at a "
                        "ctx where ALL candidates fit on the GPU). "
                        "Incompatible with --knob ctx-size and "
                        "--matrix-knob ctx-size.")
    p.add_argument("--autowrite-section-slug", default=None,
                   help="Slug of the section to drive the autowrite-section "
                        "workload. Default: first section of the active book.")
    p.add_argument("--reps", type=int, default=3,
                   help="Repetitions per (knob, value). Default 3.")
    p.add_argument("--predict", type=int, default=1500,
                   help="Output token cap per run. Default 1500.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write the JSONL log. Default "
                        "<data_dir>/bench/substrate_sweep/.")
    args = p.parse_args()

    values = [v.strip() for v in args.values.split(",") if v.strip()]
    if not values:
        console.print("[red]--values produced an empty list.[/red]")
        return 2

    matrix_values: list[str] = []
    if args.matrix_knob:
        if not args.matrix_values:
            console.print("[red]--matrix-knob requires --matrix-values.[/red]")
            return 2
        if args.matrix_knob == args.knob:
            console.print("[red]--matrix-knob must differ from --knob.[/red]")
            return 2
        matrix_values = [v.strip() for v in args.matrix_values.split(",")
                         if v.strip()]
        if not matrix_values:
            console.print("[red]--matrix-values produced an empty list.[/red]")
            return 2

    if args.ctx_override is not None:
        if args.knob == "ctx-size" or args.matrix_knob == "ctx-size":
            console.print("[red]--ctx-override is incompatible with sweeping "
                          "ctx-size (it would silently shadow the swept value).[/red]")
            return 2

    out_dir = args.out_dir or Path(settings.data_dir) / "bench" / "substrate_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = (f"-{args.matrix_knob}" if args.matrix_knob else "")
    out_file = out_dir / f"{ts}-{args.role}-{args.knob}{suffix}.jsonl"

    aw_args = None
    if args.workload == "autowrite-section":
        aw_args = _resolve_autowrite_section(args.autowrite_section_slug)
        console.print(
            f"[dim]autowrite-section bound to "
            f"book_id={aw_args['book_id'][:8]}… "
            f"section={aw_args['section_type']}[/dim]"
        )

    console.print(
        f"\n[bold]Substrate sweep[/bold] · role={args.role} · "
        f"knob={args.knob} · values={values} · workload={args.workload} · "
        f"baseline={args.baseline} · reps={args.reps}"
        + (f" · matrix={args.matrix_knob}={matrix_values}"
           if args.matrix_knob else "")
        + f"\n[dim]Writing: {out_file}[/dim]\n"
    )

    # In matrix mode iterate the cartesian product; otherwise a single
    # axis. The output shape (rows list) is the same — matrix_knob /
    # matrix_value fields are populated only in matrix mode.
    cells: list[tuple[str, str | None]] = []
    if args.matrix_knob:
        for v in values:
            for mv in matrix_values:
                cells.append((v, mv))
    else:
        cells = [(v, None) for v in values]

    rows: list[dict] = []
    with out_file.open("a") as fh:
        for (v, mv) in cells:
            label = f"{args.knob}={v}"
            if mv is not None:
                label += f"  {args.matrix_knob}={mv}"
            console.print(f"  ▶ {label}")
            r = _run_one_config(
                args.role, args.knob, v, args.workload,
                reps=args.reps, predict=args.predict,
                matrix_knob=args.matrix_knob, matrix_value=mv,
                autowrite_section_args=aw_args,
                baseline=args.baseline,
                ctx_override=args.ctx_override,
            )
            r["timestamp"] = datetime.now(timezone.utc).isoformat()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
            if r["failure"]:
                console.print(f"    [red]✗[/red] {r['failure'][:120]}")
            else:
                if args.workload == "autowrite-section":
                    fs = (r.get("final_score") or {}).get("mean")
                    iters = (r.get("iterations") or {}).get("mean")
                    wall = (r.get("wall_s") or {}).get("mean") or 0
                    console.print(
                        f"    [green]✓[/green] final_score="
                        f"{(fs or 0):.2f} iter={iters or '?'} "
                        f"wall={wall:.0f}s peak_vram={r['peak_vram_mib']}MiB"
                    )
                else:
                    d = (r["decode_tps"] or {}).get("mean") or 0
                    pe = (r["prompt_eval_tps"] or {}).get("mean") or 0
                    console.print(
                        f"    [green]✓[/green] decode={d:.1f}t/s "
                        f"prompt_eval={pe:.0f}t/s "
                        f"peak_vram={r['peak_vram_mib']}MiB "
                        f"cold={(r['cold_load_s'] or 0):.1f}s"
                    )

    # Render summary table — different columns for autowrite-section
    # (quality probe) vs synthetic (throughput).
    title_extra = f" × {args.matrix_knob}" if args.matrix_knob else ""
    if args.workload == "autowrite-section":
        tbl = Table(
            title=f"Sweep (autowrite-section): {args.role}.{args.knob}{title_extra}",
            show_lines=True,
        )
        tbl.add_column(args.knob)
        if args.matrix_knob:
            tbl.add_column(args.matrix_knob)
        tbl.add_column("final_score", justify="right")
        tbl.add_column("iter", justify="right")
        tbl.add_column("words", justify="right")
        tbl.add_column("peak VRAM", justify="right")
        tbl.add_column("wall (s)", justify="right")
        tbl.add_column("status")
        for r in rows:
            row_cells = [str(r["value"])]
            if args.matrix_knob:
                row_cells.append(str(r.get("matrix_value") or ""))
            if r["failure"]:
                row_cells += ["—", "—", "—", "—", "—",
                              f"[red]{r['failure'][:50]}[/red]"]
            else:
                fs = (r.get("final_score") or {}).get("mean") or 0
                iters = (r.get("iterations") or {}).get("mean") or 0
                wc = (r.get("word_count") or {}).get("mean") or 0
                wall = (r.get("wall_s") or {}).get("mean") or 0
                row_cells += [
                    f"{fs:.2f}",
                    f"{iters:.0f}",
                    f"{wc:.0f}",
                    f"{r['peak_vram_mib']} MiB",
                    f"{wall:.1f}",
                    f"[green]ok[/green] ({r['reps_ok']}/{r['reps_attempted']})",
                ]
            tbl.add_row(*row_cells)
    else:
        tbl = Table(
            title=f"Sweep: {args.role}.{args.knob}{title_extra}",
            show_lines=True,
        )
        tbl.add_column(args.knob)
        if args.matrix_knob:
            tbl.add_column(args.matrix_knob)
        tbl.add_column("decode TPS", justify="right")
        tbl.add_column("prompt-eval TPS", justify="right")
        tbl.add_column("peak VRAM", justify="right")
        tbl.add_column("cold load", justify="right")
        tbl.add_column("wall (s)", justify="right")
        tbl.add_column("status")
        for r in rows:
            row_cells = [str(r["value"])]
            if args.matrix_knob:
                row_cells.append(str(r.get("matrix_value") or ""))
            if r["failure"]:
                row_cells += ["—", "—", "—", "—", "—",
                              f"[red]{r['failure'][:50]}[/red]"]
            else:
                d = (r["decode_tps"] or {}).get("mean") or 0
                pe = (r["prompt_eval_tps"] or {}).get("mean") or 0
                wall = (r["wall_s"] or {}).get("mean") or 0
                row_cells += [
                    f"{d:.1f}",
                    f"{pe:.0f}",
                    f"{r['peak_vram_mib']} MiB",
                    f"{(r['cold_load_s'] or 0):.1f}s",
                    f"{wall:.1f}",
                    f"[green]ok[/green] ({r['reps_ok']}/{r['reps_attempted']})",
                ]
            tbl.add_row(*row_cells)
    console.print(tbl)
    console.print(f"\n[dim]Full log: {out_file}[/dim]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
