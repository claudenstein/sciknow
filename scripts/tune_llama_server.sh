#!/usr/bin/env bash
# Phase 54.6.304 — tune llama-server flags for book-writer decode tok/s
# on the local 3090. Starts llama-server with each flag combo, runs
# scripts/bench_writer_tps.py, records the average decode t/s, tears
# down and tries the next combo. Prints a final leaderboard.
#
# Usage:  scripts/tune_llama_server.sh [short|full]
#   short (default) — 4 combos, ~6 min
#   full            — 8 combos, ~12 min
#
# The winner is written to data/bench/tune_llama_server.log along with
# full per-run metrics; copy its flags into scripts/llama_server_book_writer.sh
# (or leave them as env defaults) to adopt the tune.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p data/bench
LOG="data/bench/tune_llama_server.log"
: > "$LOG"

MODE="${1:-short}"

# Each combo: "label|FA|CACHE_K|CACHE_V|BATCH|UBATCH"
if [ "$MODE" = "full" ]; then
  COMBOS=(
    "fa-q4q4|on|q4_0|q4_0|2048|512"
    "fa-q8q8|on|q8_0|q8_0|2048|512"
    "fa-f16f16|on|f16|f16|2048|512"
    "nofa-f16f16|off|f16|f16|2048|512"
    "fa-q8q4|on|q8_0|q4_0|2048|512"
    "fa-f16q8|on|f16|q8_0|2048|512"
    "fa-q4q4-big|on|q4_0|q4_0|4096|1024"
    "fa-q8q8-small|on|q8_0|q8_0|1024|256"
  )
else
  COMBOS=(
    "fa-q4q4|on|q4_0|q4_0|2048|512"
    "fa-q8q8|on|q8_0|q8_0|2048|512"
    "fa-f16f16|on|f16|f16|2048|512"
    "fa-q8q4|on|q8_0|q4_0|2048|512"
  )
fi

stop_server() {
  pkill -f 'llama-server.*Qwen3.6-27B' 2>/dev/null || true
  for _ in $(seq 1 30); do
    pgrep -f 'llama-server.*Qwen3.6-27B' >/dev/null 2>&1 || return 0
    sleep 0.5
  done
  pkill -9 -f 'llama-server.*Qwen3.6-27B' 2>/dev/null || true
  sleep 1
}

start_server() {
  local fa="$1" ck="$2" cv="$3" bs="$4" ub="$5"
  env LLAMACPP_FA="$fa" LLAMACPP_CACHE_K="$ck" LLAMACPP_CACHE_V="$cv" \
      LLAMACPP_BATCH="$bs" LLAMACPP_UBATCH="$ub" \
      nohup bash "$ROOT/scripts/llama_server_book_writer.sh" \
      >"/tmp/llama-server.tune.log" 2>&1 &
  disown
  for _ in $(seq 1 60); do
    if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "server never came up" >&2
  tail -30 /tmp/llama-server.tune.log >&2
  return 1
}

echo "== llama-server tuning sweep — $(date -Is) ==" | tee -a "$LOG"
stop_server

declare -a RESULTS
for combo in "${COMBOS[@]}"; do
  IFS='|' read -r label fa ck cv bs ub <<<"$combo"
  echo | tee -a "$LOG"
  echo "---- $label (fa=$fa kv=$ck/$cv batch=$bs/$ub) ----" | tee -a "$LOG"
  stop_server
  if ! start_server "$fa" "$ck" "$cv" "$bs" "$ub"; then
    RESULTS+=("$label|FAIL|FAIL")
    continue
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$LOG"
  # 1 warmup + 2 runs @ 800 output tokens. Decode t/s stabilizes fast;
  # 800 tokens * ~25 s is plenty to average out noise.
  out=$(uv run python scripts/bench_writer_tps.py \
          --backend llamacpp --model qwen3.6:27b-dense \
          --base-url http://localhost:8080 \
          --num-ctx 32768 --num-predict 800 \
          --warmup 1 --runs 2 2>&1 || echo "BENCH FAILED")
  echo "$out" | tee -a "$LOG"
  avg=$(echo "$out" | awk '/AVERAGE/ {print $2}' | sed 's/decode=//')
  RESULTS+=("$label|${avg:-?}|fa=$fa kv=$ck/$cv batch=$bs/$ub")
done

stop_server

echo | tee -a "$LOG"
echo "==== LEADERBOARD ====" | tee -a "$LOG"
printf "%-20s  %-10s  %s\n" "label" "decode t/s" "flags" | tee -a "$LOG"
printf '%s\n' "${RESULTS[@]}" | awk -F'|' '{printf "%-20s  %-10s  %s\n", $1, $2, $3}' \
  | sort -k2 -r | tee -a "$LOG"
echo | tee -a "$LOG"
echo "log: $LOG"
