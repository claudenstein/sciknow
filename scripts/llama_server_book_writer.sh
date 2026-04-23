#!/usr/bin/env bash
# Phase 54.6.304 — start llama-server for the book-writer hot path.
#
# Flags follow the @Punch_Taylor 4090 throughput recipe (43.1 t/s on
# Qwen3.6-27B Q4_K_M) adapted for the local 3090:
#   -ngl 99               offload all layers
#   -c 32768              32K context — matches sciknow's autowrite num_ctx
#                         envelope. Bumped to 65536 / 131072 only if you
#                         actually feed long contexts; q4_0 KV is the
#                         compression knob that lets you go past 32K
#                         on a 24 GB card without OOM.
#   --batch-size 2048     prefill chunk; matches Ollama's OLLAMA_NUM_BATCH
#                         and is the @Punch_Taylor default for prompt-eval
#                         throughput on this class of card.
#   --ubatch-size 512     micro-batch — keeps cuBLAS kernels filled.
#   -np 1                 single-slot serving (autowrite is sequential).
#   -fa on                flash attention (the @Punch_Taylor 4090 number
#                         requires this — Ollama silently disabled it for
#                         us via a typo'd env var; we don't make that
#                         mistake here).
#   --cache-type-k/v q4_0 KV-cache quantization (matches the bench).
#                         q8_0 is the safer default if you see grounding
#                         regressions; q4_0 reclaims ~50% of KV VRAM.
#   --alias qwen3.6:27b-dense
#                         the OpenAI-API model name. Must match
#                         settings.llamacpp_model_alias and book_write_model
#                         so sciknow's dispatcher routes correctly.
#
# Usage (foreground):  scripts/llama_server_book_writer.sh
# Usage (background):  nohup scripts/llama_server_book_writer.sh > /tmp/llama-server.log 2>&1 &
# Stop:                pkill -f 'llama-server.*Qwen3.6-27B'

set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-$HOME/Claude/llama.cpp-build/llama.cpp/build/bin/llama-server}"
GGUF="${GGUF:-$HOME/Claude/huggingface/unsloth-Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf}"
PORT="${LLAMACPP_PORT:-8080}"
HOST="${LLAMACPP_HOST:-127.0.0.1}"
CTX="${LLAMACPP_CTX:-32768}"
ALIAS="${LLAMACPP_ALIAS:-qwen3.6:27b-dense}"

# Tuning knobs — the tuning sweep (scripts/tune_llama_server.sh) overrides
# these via env to compare flag combinations without editing the script.
# Defaults are the Phase 54.6.304 tuned values for the local 3090.
FA="${LLAMACPP_FA:-on}"                        # flash attention: on/off
CACHE_K="${LLAMACPP_CACHE_K:-q8_0}"             # KV-K type: f16 | q8_0 | q4_0
CACHE_V="${LLAMACPP_CACHE_V:-q8_0}"             # KV-V type: f16 | q8_0 | q4_0
BATCH="${LLAMACPP_BATCH:-2048}"
UBATCH="${LLAMACPP_UBATCH:-512}"
NO_MMAP="${LLAMACPP_NO_MMAP:-1}"                # 1 = pass --no-mmap, 0 = don't

if [ ! -x "$LLAMA_BIN" ]; then
  echo "llama-server binary not found at $LLAMA_BIN" >&2
  exit 1
fi
if [ ! -f "$GGUF" ]; then
  echo "GGUF not found at $GGUF" >&2
  exit 1
fi

echo "Starting llama-server"
echo "  binary : $LLAMA_BIN"
echo "  model  : $GGUF"
echo "  alias  : $ALIAS"
echo "  port   : $PORT"
echo "  ctx    : $CTX"
echo "  fa=$FA kv=$CACHE_K/$CACHE_V batch=$BATCH/$UBATCH no-mmap=$NO_MMAP"

ARGS=(
  -m "$GGUF"
  --alias "$ALIAS"
  --host "$HOST" --port "$PORT"
  -ngl 99
  -c "$CTX"
  --batch-size "$BATCH"
  --ubatch-size "$UBATCH"
  -np 1
  -fa "$FA"
  --cache-type-k "$CACHE_K"
  --cache-type-v "$CACHE_V"
)
if [ "$NO_MMAP" = "1" ]; then
  ARGS+=(--no-mmap)
fi
exec "$LLAMA_BIN" "${ARGS[@]}"
