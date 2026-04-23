# llama.cpp `llama-server` for the book-writer hot path

Phase 54.6.304 — opt-in side-process backend for `sciknow book write` /
`sciknow book autowrite`. Routes the book-writer model's streaming calls
through a standalone `llama-server` process instead of Ollama; every
other LLM role (outline, scorer, verifier, wiki compile, extract-kg,
…) keeps using Ollama unchanged.

**Why:** on this 3090, the llama.cpp `llama-server` binary is about **12%
faster per token** than Ollama for the same quant at the same context
size. Measured via `scripts/bench_writer_tps.py` on Qwen3.6-27B Q4_K_XL
at 32 768 ctx with an autowrite-shaped prompt:

| backend  | decode tok/s | prompt tok/s | wall (~1 500 out tok) |
|----------|-------------:|-------------:|----------------------:|
| Ollama   |        32.03 |          979 |                 48.1s |
| llama.cpp|    **35.95** |         1136 |                 42.3s |

See `data/bench/writer_tps.jsonl` for the raw JSONL and
`data/bench/tune_llama_server.log` for the tuning sweep.

## Ceiling notes

Qwen3.6-27B at Q4_K_XL is 17.6 GB of weights. The 3090 has 936 GB/s of
memory bandwidth, so the theoretical decode ceiling is ~53 t/s. Under
load the GPU pins at 99% util / 415 W / 9501 MHz mem clock — we're
fully bandwidth-bound, not compute-bound. **35.95 t/s is 68% of
peak**, same efficiency as the @Punch_Taylor 4090 number (43.1 t/s
on Q4_K_M = 68% of that card's theoretical peak).

Crossing 40 t/s on this quant is unreachable on this GPU. Options if
you need to push past it later:

  - **IQ4_XS / Q4_K_M GGUF** (~15.4 / 16.8 GB). Expected ~40 / ~38 t/s.
    Quality within ~1% of Q4_K_XL on 27B-class models.
  - **UD-Q3_K_XL** (14.5 GB). Expected ~42-44 t/s with a ~3-5% quality
    dip on hard tasks; still fine for prose writing.
  - **Speculative decoding** with a Qwen3-0.6B draft model. 1.5-2×
    speedup with no quality loss, at the cost of more VRAM and a
    second model download.

## Prerequisites

  - `llama-server` binary built with CUDA:
    `~/Claude/llama.cpp-build/llama.cpp/build/bin/llama-server`
  - Qwen3.6-27B GGUF at:
    `~/Claude/huggingface/unsloth-Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf`

Override either with env vars (`LLAMA_BIN=...`, `GGUF=...`) on the
startup script.

## Start / stop

```bash
# foreground (ctrl-C to stop)
scripts/llama_server_book_writer.sh

# background, with log
nohup scripts/llama_server_book_writer.sh > /tmp/llama-server.log 2>&1 &

# stop
pkill -f 'llama-server.*Qwen3.6-27B'
```

The script reads these env vars (defaults in parens):

| env                 | default     | meaning                                    |
|---------------------|-------------|--------------------------------------------|
| `LLAMACPP_PORT`     | 8080        | bind port                                  |
| `LLAMACPP_HOST`     | 127.0.0.1   | bind host                                  |
| `LLAMACPP_CTX`      | 32768       | context window                             |
| `LLAMACPP_ALIAS`    | qwen3.6:27b-dense | OpenAI-API model name it registers   |
| `LLAMACPP_FA`       | on          | flash attention (`on` / `off`)             |
| `LLAMACPP_CACHE_K`  | q8_0        | KV-K type: `f16` / `q8_0` / `q4_0`         |
| `LLAMACPP_CACHE_V`  | q8_0        | KV-V type (same domain)                    |
| `LLAMACPP_BATCH`    | 2048        | `--batch-size` (prompt prefill)            |
| `LLAMACPP_UBATCH`   | 512         | `--ubatch-size`                            |
| `LLAMACPP_NO_MMAP`  | 1           | pass `--no-mmap`                           |

The defaults came out of `scripts/tune_llama_server.sh` (4-combo sweep,
fa-q8q8 won by 0.07 t/s over fa-q4q4 and ships ~500 MB more KV headroom
for long contexts).

## Tune your own flags

```bash
scripts/tune_llama_server.sh short      # 4 combos, ~6 min
scripts/tune_llama_server.sh full       # 8 combos, ~12 min
```

Writes a leaderboard to `data/bench/tune_llama_server.log`. The driver
stops any running server, sweeps each combo (1 warmup + 2 bench runs
at 800 output tokens), and restarts nothing at the end — you pick the
winner and restart manually (`scripts/llama_server_book_writer.sh`).

## Wire it into sciknow

In `.env` (or `.env.overlay`):

```bash
LLAMACPP_BOOK_WRITER_ENABLED=true
LLAMACPP_BASE_URL=http://localhost:8080
LLAMACPP_MODEL_ALIAS=qwen3.6:27b-dense
BOOK_WRITE_MODEL=qwen3.6:27b-dense
```

The dispatcher in `sciknow/rag/llm.py` routes to the llama.cpp backend
iff **all** of these are true for a given call:

  1. `llamacpp_book_writer_enabled` is `true`
  2. `book_write_model` is set
  3. The call's `model=` matches `book_write_model`
  4. `format=` is `None` (JSON-schema calls stay on Ollama)
  5. `llama-server` answers `GET /health` inside 500 ms

Any miss falls through to Ollama. If (5) fails while (1)-(4) succeed,
you get a one-line WARNING in the log so the "server forgot to start"
case is obvious instead of silent.

## Benchmark

```bash
# Ollama baseline (stop llama-server first to free VRAM)
pkill -f 'llama-server.*Qwen3.6-27B'
uv run python scripts/bench_writer_tps.py --backend ollama \
    --model qwen3.6:27b-dense --num-ctx 32768 --num-predict 1500 \
    --warmup 1 --runs 3

# llama.cpp
scripts/llama_server_book_writer.sh &
uv run python scripts/bench_writer_tps.py --backend llamacpp \
    --base-url http://localhost:8080 \
    --model qwen3.6:27b-dense --num-ctx 32768 --num-predict 1500 \
    --warmup 1 --runs 3
```

The bench prompt is ~3 500 input tokens of realistic autowrite
context + `think=False` / `enable_thinking=false` (no CoT) to match
the autowrite contract.

## Gotchas

  - **`thinking_budget: 0` does not suppress the `<think>` block on the
    Qwen3.6-27B unsloth GGUF.** The reliable switch on this chat
    template is `chat_template_kwargs: {enable_thinking: false}`. The
    `sciknow.rag.llamacpp` backend sends this automatically when
    `think=False`.
  - **First token latency is high if the server just started** — model
    load takes ~5 s after the process is up. Subsequent calls are
    fast. Warm up with `curl /health` + a throwaway `/v1/chat/completions`
    if you care.
  - **VRAM headroom matters.** llama-server holds the model resident
    for its entire lifetime (~18.5 GB on this setup). Keeping Ollama
    loaded alongside a second 27B+ model OOMs the card. The dispatcher's
    fallback is designed for exactly this case — stop `llama-server`
    and the book-writer will transparently use Ollama.
  - **`format=json_schema` goes to Ollama, not llama-server.** The
    llama.cpp backend doesn't implement structured output (guarded by
    `NotImplementedError`). All prose writing is JSON-free, so this
    only affects JSON-heavy flows like outline generation — which
    are routed by model, not by the dispatcher.
