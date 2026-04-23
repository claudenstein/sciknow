"""Phase 54.6.304 — book-writer tok/s bench across Ollama vs llama.cpp.

Drives the SAME autowrite-shaped prompt against either backend and
reports decode tok/s, prompt-eval tok/s, and total wall-clock. Goal
is to compare apples to apples for "how fast does a section get
written" — independent of sciknow's wrapper code.

The prompt is a fixed fixture (no retrieval, no DB) so successive
runs are bit-identical. ~3500 input tokens, target 1500 output
tokens, /no_think to force non-thinking mode (matches autowrite's
``force_no_thinking=True`` setting for qwen3.6).

Usage:
  uv run python scripts/bench_writer_tps.py --backend ollama
  uv run python scripts/bench_writer_tps.py --backend llamacpp --base-url http://localhost:8080
  uv run python scripts/bench_writer_tps.py --backend ollama --runs 3 --warmup 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Long, realistic section-write user prompt. ~3500 tokens of input,
# autowrite-shaped: a section title + topic + several "context"
# excerpts in the same flavour as our retrieval pipeline.
_SYSTEM = (
    "You are a scientific writer drafting a chapter section for an "
    "evidence-grounded book on climate intervention. Write in clear, "
    "engaging prose at the level of a Scientific American feature. "
    "Cite each substantive claim with bracketed paper IDs from the context. "
    "Aim for ~1500 words. Do not invent references. Do not include a heading."
)

_CONTEXT_BLOCK = (
    "[paper-001] Stratospheric aerosol injection (SAI) involves the deliberate "
    "release of reflective particles into the lower stratosphere to scatter a "
    "small fraction of incoming sunlight back to space. Modelling studies "
    "consistently find that SAI could partially offset the radiative forcing "
    "of greenhouse gases on a global-mean basis, but introduce regional "
    "perturbations to precipitation patterns, particularly affecting monsoon "
    "circulation in South Asia and West Africa.\n\n"
    "[paper-002] Marine cloud brightening exploits the Twomey effect: adding "
    "small sea-salt aerosols to low marine stratus clouds increases their "
    "albedo by a measurable amount. Field-scale tests are challenging because "
    "the resulting cloud changes are entangled with natural variability on "
    "the spatial scales we can observe with current satellites.\n\n"
    "[paper-003] Cirrus cloud thinning, in contrast, targets the longwave "
    "side of the radiation budget rather than shortwave reflectivity. The "
    "proposed mechanism injects ice nuclei into thin cirrus to accelerate "
    "ice particle settling, reducing cloud cover. The forcing leverage per "
    "unit aerosol is large but the operational uncertainties — particularly "
    "the risk of overseeding — remain unresolved.\n\n"
    "[paper-004] Direct ocean alkalinity enhancement relies on dissolving "
    "alkaline minerals (typically olivine or hydrated lime) into surface "
    "waters to shift the carbonate equilibrium and increase dissolved "
    "inorganic carbon storage. Recent shipboard demonstrations show the "
    "chemistry works as predicted; the open question is how to industrialise "
    "the supply chain at gigaton scale without competing with cement "
    "production for limestone.\n\n"
    "[paper-005] Iron fertilisation of high-nutrient low-chlorophyll regions "
    "enhances primary productivity transiently, but the export efficiency to "
    "the deep ocean — the metric that matters for climate — has been "
    "consistently lower than initial enthusiasm suggested. The 13 in-situ "
    "experiments to date show a wide spread, with median export of "
    "approximately 1 gC per gFe added.\n\n"
    "[paper-006] Governance of solar geoengineering research is proceeding "
    "asynchronously across jurisdictions. The 2021 Harvard SCoPEx field test "
    "was suspended after Saami council objections; the 2022 Mexican "
    "moratorium on outdoor experimentation followed a Make Sunsets balloon "
    "release. Calls for a non-use agreement now sit alongside calls for "
    "structured research programs at the same UN bodies.\n\n"
    "[paper-007] Termination shock — the rapid warming that would follow an "
    "abrupt halt to a sustained SAI program — is the most-cited risk in the "
    "literature. Model studies estimate decadal warming rates 5-10 times "
    "faster than current trends if a 1.5 K SAI offset were withdrawn over a "
    "single year, with corresponding ecological disruption.\n\n"
    "[paper-008] Coupling carbon dioxide removal with solar radiation "
    "modification is increasingly framed as a complementary rather than "
    "substitute strategy. The 'peak shaving' framing uses SRM transiently "
    "to suppress overshoot temperatures while CDR scales up to reverse the "
    "underlying CO2 forcing on multi-decadal timescales.\n"
)

_USER = (
    "Section title: Mechanisms and feasibility of solar radiation management\n"
    "Chapter topic: Climate intervention strategies and their tradeoffs\n\n"
    "Context excerpts:\n\n"
    f"{_CONTEXT_BLOCK}\n\n"
    "Write the section now. Cite bracketed IDs inline."
)


@dataclass
class RunResult:
    backend: str
    model: str
    elapsed_s: float
    output_tokens: int
    prompt_tokens: int | None
    decode_tps: float
    prompt_tps: float | None
    first_token_s: float
    output_chars: int


def _bench_ollama(model: str, num_ctx: int, num_predict: int) -> RunResult:
    import ollama
    client = ollama.Client(host="http://localhost:11434")
    t0 = time.monotonic()
    first_token_t: float | None = None
    pieces: list[str] = []
    last_chunk = None
    for chunk in client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER},
        ],
        stream=True,
        keep_alive=-1,
        think=False,                     # autowrite contract for qwen3.6
        options={
            "temperature": 0.2, "num_ctx": num_ctx,
            "num_batch": 1024, "num_predict": num_predict,
        },
    ):
        # qwen3.6 hybrid: with think=False the content channel carries
        # the visible output. Defensive: also pick up thinking tokens
        # if a future ollama version routes them differently.
        piece = (getattr(chunk.message, "content", "") or "") + \
                (getattr(chunk.message, "thinking", "") or "")
        if first_token_t is None and piece:
            first_token_t = time.monotonic()
        pieces.append(piece)
        last_chunk = chunk
    elapsed = time.monotonic() - t0
    output = "".join(pieces)
    # Native Ollama metrics (ns) — most accurate for decode tps.
    eval_count = getattr(last_chunk, "eval_count", None) or 0
    eval_dur = getattr(last_chunk, "eval_duration", None) or 0
    prompt_count = getattr(last_chunk, "prompt_eval_count", None)
    prompt_dur = getattr(last_chunk, "prompt_eval_duration", None) or 0
    decode_tps = eval_count / (eval_dur / 1e9) if eval_dur else 0.0
    prompt_tps = (
        prompt_count / (prompt_dur / 1e9) if prompt_dur and prompt_count else None
    )
    return RunResult(
        backend="ollama", model=model, elapsed_s=elapsed,
        output_tokens=eval_count, prompt_tokens=prompt_count,
        decode_tps=decode_tps, prompt_tps=prompt_tps,
        first_token_s=(first_token_t - t0) if first_token_t else elapsed,
        output_chars=len(output),
    )


def _bench_llamacpp(base_url: str, model: str,
                    num_ctx: int, num_predict: int) -> RunResult:
    import urllib.request
    import urllib.error
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER},
        ],
        "temperature": 0.2,
        "max_tokens": num_predict,
        "stream": True,
        # Qwen3.6 chat template: disable thinking to match the Ollama
        # baseline (which passes think=False above). thinking_budget=0
        # is NOT honored on this template — use enable_thinking=false.
        "chat_template_kwargs": {"enable_thinking": False},
        # llama-server bundles timing stats on the final chunk only
        # when this flag is set; otherwise we only see usage counts.
        "timings_per_token": True,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    first_token_t: float | None = None
    pieces: list[str] = []
    output_tokens = 0
    prompt_tokens: int | None = None
    final_timings: dict | None = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                evt = json.loads(data)
            except Exception:
                continue
            # llama-server packs final timings in the last chunk's
            # "timings" key (non-standard but widely used).
            if "timings" in evt:
                final_timings = evt["timings"]
            if "usage" in evt and evt["usage"]:
                u = evt["usage"]
                if u.get("prompt_tokens"):
                    prompt_tokens = u["prompt_tokens"]
                if u.get("completion_tokens"):
                    output_tokens = u["completion_tokens"]
            for choice in evt.get("choices", []) or []:
                delta = choice.get("delta", {}) or {}
                content = delta.get("content") or ""
                if content and first_token_t is None:
                    first_token_t = time.monotonic()
                if content:
                    pieces.append(content)
    elapsed = time.monotonic() - t0
    output = "".join(pieces)
    if not output_tokens:
        # Fall back to whitespace token estimate.
        output_tokens = max(1, len(output.split()))
    decode_tps: float
    prompt_tps: float | None
    if final_timings:
        decode_tps = float(final_timings.get("predicted_per_second")
                           or (output_tokens / elapsed))
        prompt_tps = final_timings.get("prompt_per_second")
        if prompt_tokens is None:
            prompt_tokens = final_timings.get("prompt_n")
    else:
        decode_tps = output_tokens / elapsed if elapsed else 0.0
        prompt_tps = None
    return RunResult(
        backend="llamacpp", model=model, elapsed_s=elapsed,
        output_tokens=output_tokens, prompt_tokens=prompt_tokens,
        decode_tps=decode_tps, prompt_tps=prompt_tps,
        first_token_s=(first_token_t - t0) if first_token_t else elapsed,
        output_chars=len(output),
    )


def _print_result(r: RunResult, label: str = "") -> None:
    head = f"  [{label}] " if label else "  "
    pt = f"{r.prompt_tokens}" if r.prompt_tokens else "—"
    pps = f"{r.prompt_tps:.1f}" if r.prompt_tps else "—"
    print(
        f"{head}{r.backend:8s} model={r.model[:32]:32s} "
        f"out={r.output_tokens:5d} tok ({r.output_chars:5d} chars)  "
        f"decode={r.decode_tps:6.2f} t/s  prompt={pps:>6s} t/s "
        f"(prompt_n={pt})  ttft={r.first_token_s:5.2f}s  "
        f"wall={r.elapsed_s:6.2f}s"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ollama", "llamacpp"], required=True)
    p.add_argument("--model", default="qwen3.6:27b-dense",
                   help="Ollama model name OR llama.cpp served-model alias")
    p.add_argument("--base-url", default="http://localhost:8080",
                   help="llama-server base URL (llamacpp backend only)")
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--num-predict", type=int, default=1500)
    p.add_argument("--warmup", type=int, default=1,
                   help="warmup runs (not reported)")
    p.add_argument("--runs", type=int, default=3,
                   help="reported runs")
    p.add_argument("--out", default=None,
                   help="optional JSONL output path")
    args = p.parse_args()

    fn = _bench_ollama if args.backend == "ollama" else _bench_llamacpp
    fn_args: tuple
    if args.backend == "ollama":
        fn_args = (args.model, args.num_ctx, args.num_predict)
    else:
        fn_args = (args.base_url, args.model, args.num_ctx, args.num_predict)

    print(f"== bench {args.backend} model={args.model} "
          f"runs={args.runs} warmup={args.warmup} "
          f"num_ctx={args.num_ctx} target_out={args.num_predict} ==")

    # warmup
    for i in range(args.warmup):
        try:
            r = fn(*fn_args)
            _print_result(r, label=f"warmup {i+1}")
        except Exception as e:
            print(f"  warmup failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

    results: list[RunResult] = []
    for i in range(args.runs):
        try:
            r = fn(*fn_args)
        except Exception as e:
            print(f"  run {i+1} failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        results.append(r)
        _print_result(r, label=f"run    {i+1}")

    if results:
        avg_decode = sum(r.decode_tps for r in results) / len(results)
        avg_prompt_vals = [r.prompt_tps for r in results if r.prompt_tps]
        avg_prompt = sum(avg_prompt_vals) / len(avg_prompt_vals) if avg_prompt_vals else None
        avg_tokens = sum(r.output_tokens for r in results) / len(results)
        avg_wall = sum(r.elapsed_s for r in results) / len(results)
        print()
        print(f"  AVERAGE  decode={avg_decode:.2f} t/s  "
              f"prompt={(f'{avg_prompt:.1f}' if avg_prompt else '—'):>6s} t/s  "
              f"out~{avg_tokens:.0f} tok  wall~{avg_wall:.2f}s")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a") as f:
            for r in results:
                f.write(json.dumps(asdict(r)) + "\n")
        print(f"  wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
