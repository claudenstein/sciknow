"""Phase 55.V10 — llama-server flag-sweep bench harness.

For one llama-server flag at a time, sweep N values against the writer
or scorer role and measure:

  - prompt-eval TPS  (server-reported)
  - decode TPS       (server-reported)
  - peak VRAM (MiB)  (nvidia-smi sample at end of generation)
  - first-token latency (s, wall)
  - cold-load time   (s, wall — process spawn → /health=200)
  - failure mode     (OOM / context-overflow / crash / OK)

Discipline: ONE knob per run. The script enforces it — pass --knob
exactly once. Re-run with a different --knob to compare.

Each (role, knob, value) is timed three times and averaged with min/max
preserved so noise is visible. Results land in
``data/bench/substrate_sweep/<timestamp>.jsonl`` + a Rich-rendered
summary table on stdout. Re-runs append; the JSONL is the authoritative
record.

Usage examples:

  # Sweep writer ctx-size, the immediate question.
  uv run python scripts/bench_substrate_sweep.py \\
      --role writer --knob ctx-size \\
      --values 16384,20480,24576,28672,32768

  # Sweep scorer ctx-size (gemma 4 31B).
  uv run python scripts/bench_substrate_sweep.py \\
      --role scorer --knob ctx-size \\
      --values 16384,20480,24576

  # Paired KV-cache quant (K and V together — the standard pattern).
  uv run python scripts/bench_substrate_sweep.py \\
      --role writer --knob cache-type \\
      --values f16,q8_0,q4_0

  # Batch / ubatch — affects long-prompt prefill speed.
  uv run python scripts/bench_substrate_sweep.py \\
      --role writer --knob batch-size \\
      --values 1024,2048,4096,8192

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
    """Stuff `_CHUNK_TEMPLATE` until the prompt reaches the target size."""
    chunks_needed = max(1, approx_input_tokens // 90)  # ~90 tokens/chunk
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
    },
    "large": {
        # ~18K input tokens — pushes ctx-size near 24K limit.
        "approx_input_tokens": 18000,
        "max_predict": 1500,
    },
}


# ── llama-server config patching ───────────────────────────────────────────────


def _patch_role_for_run(role: str, knob: str, value: str) -> dict:
    """Mutate ROLE_DEFAULTS[role] in place for the upcoming run.
    Returns a backup dict so the caller can restore on tear-down.

    Handled knobs (one --knob per script invocation, by design):
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
        # Same value for K and V (the standard paired pattern).
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
        cleaned += ["--cache-type-k", str(value), "--cache-type-v", str(value)]
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


def _run_one_config(
    role: str, knob: str, value: str, workload: str,
    *, reps: int = 3, predict: int = 1500,
) -> dict:
    """Patch → up → run reps → down → restore. Returns averaged dict."""
    wcfg = WORKLOADS[workload]
    prompt = _build_prompt(wcfg["approx_input_tokens"])
    max_predict = predict or wcfg["max_predict"]

    _down_all_roles()
    backup = _patch_role_for_run(role, knob, value)
    cold_t0 = time.monotonic()
    cold_s = None
    failure = None
    runs: list[dict] = []
    peak_vram = -1
    try:
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

    return {
        "role": role,
        "knob": knob,
        "value": value,
        "workload": workload,
        "approx_input_tokens": wcfg["approx_input_tokens"],
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
    p.add_argument("--workload", default="typical",
                   choices=tuple(WORKLOADS.keys()),
                   help="Synthetic prompt size.")
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

    out_dir = args.out_dir or Path(settings.data_dir) / "bench" / "substrate_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"{ts}-{args.role}-{args.knob}.jsonl"

    console.print(
        f"\n[bold]Substrate sweep[/bold] · role={args.role} · "
        f"knob={args.knob} · values={values} · workload={args.workload} · "
        f"reps={args.reps}\n[dim]Writing: {out_file}[/dim]\n"
    )

    rows: list[dict] = []
    with out_file.open("a") as fh:
        for v in values:
            console.print(f"  ▶ {args.knob}={v}")
            r = _run_one_config(
                args.role, args.knob, v, args.workload,
                reps=args.reps, predict=args.predict,
            )
            r["timestamp"] = datetime.now(timezone.utc).isoformat()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
            if r["failure"]:
                console.print(f"    [red]✗[/red] {r['failure'][:120]}")
            else:
                d = (r["decode_tps"] or {}).get("mean")
                pe = (r["prompt_eval_tps"] or {}).get("mean")
                console.print(
                    f"    [green]✓[/green] decode={d:.1f}t/s "
                    f"prompt_eval={pe:.0f}t/s "
                    f"peak_vram={r['peak_vram_mib']}MiB "
                    f"cold={r['cold_load_s']:.1f}s"
                )

    # Render summary table.
    tbl = Table(title=f"Sweep: {args.role}.{args.knob}", show_lines=True)
    tbl.add_column("value")
    tbl.add_column("decode TPS", justify="right")
    tbl.add_column("prompt-eval TPS", justify="right")
    tbl.add_column("peak VRAM", justify="right")
    tbl.add_column("cold load", justify="right")
    tbl.add_column("wall (s)", justify="right")
    tbl.add_column("status")
    for r in rows:
        if r["failure"]:
            tbl.add_row(str(r["value"]), "—", "—", "—", "—", "—",
                        f"[red]{r['failure'][:50]}[/red]")
            continue
        d = (r["decode_tps"] or {}).get("mean") or 0
        pe = (r["prompt_eval_tps"] or {}).get("mean") or 0
        wall = (r["wall_s"] or {}).get("mean") or 0
        tbl.add_row(
            str(r["value"]),
            f"{d:.1f}",
            f"{pe:.0f}",
            f"{r['peak_vram_mib']} MiB",
            f"{r['cold_load_s']:.1f}s",
            f"{wall:.1f}",
            f"[green]ok[/green] ({r['reps_ok']}/{r['reps_attempted']})",
        )
    console.print(tbl)
    console.print(f"\n[dim]Full log: {out_file}[/dim]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
