#!/bin/bash
# omxpert run: Qwen3.8-Flash-Next (Qwen 4 preview architecture, 125B MoE + 51B n-gram table) on 2x RTX 3090 + 64 GB RAM + NVMe,
# with the night-5 exact stack. Output is the same model's output (exact tricks only), served at http://127.0.0.1:${PORT:-8080}
# (OpenAI-compatible /v1). Ideas credited to 0xSero's glm-flash-lite (read pool, layer-ahead prefetch, staged prefill).
#   omxpert/run.sh            start the server
#   omxpert/run.sh smoke      start, ask one coding question, print the answer and speed, stop
#   MTP=1 omxpert/run.sh      faster answers, slower prompts (see below)
# The first request after start is slow while RAM fills (smoke test: 8.9 tok/s); warmed it runs at the speeds in the report.
set -e
D="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="${LLAMA:-$D/llama.cpp/build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-GGUF}"
GG=$(ls "$MODEL_DIR"/UD-Q4_K_XL/*00001-of-*.gguf 2>/dev/null | head -1)
PORT="${PORT:-8080}"
if [ -z "$GG" ]; then
  echo "model missing: downloading UD-Q4_K_XL (111 GB) into $MODEL_DIR"
  HF_HUB_DISABLE_XET=1 hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" --include "mtp-Qwen3.8-Flash-Next-Q8_0.gguf" --local-dir "$MODEL_DIR"
  GG=$(ls "$MODEL_DIR"/UD-Q4_K_XL/*00001-of-*.gguf | head -1)
fi
# --- the exact stack (results/night5/stack.json): same output as stock ---
#  -lzm off       the 28.8 GB n-gram table is mapped like any weight and read in at load (bit-exact; steady page faults)
#  MTP=1          also load Qwen's draft head: answers ~1.36x faster (measured), but its ~4 GB of VRAM pushes experts into RAM
#                 and prompts got 3-4x slower on this 2x3090 + 62 GB box. Use it for chat with short prompts, not for agents.
MTP_FILE="$MODEL_DIR/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
if [ "${MTP:-0}" = "1" ]; then DEF_ARGS="-lzm off -md $MTP_FILE --spec-type draft-mtp --spec-draft-n-max 2"; else DEF_ARGS="-lzm off"; fi
DEF_ENV=""
# --- night 6: STACK=pinned, the learned stack (results/night6, docs/report_night6.md) ---
#  your profile (omxpert/profile.sh) picks which experts live in VRAM, across both GPUs, the rest stay in RAM, never both (0xSero's
#  exclusive tiers); the CPU computes the cold experts while the GPUs compute the hot ones (FreeToken's hybrid idea, EDX_OVERLAP);
#  prompt batches run each layer on the GPU that holds its pins (EDX_PREFILL_LOCAL, EDX_XDEV). PIN_MIB: VRAM for pins (2x3090: 36500).
if [ "${STACK:-}" = "pinned" ]; then
  PIN_FILE="${PIN_FILE:-$D/omxpert/pins/profile.bin}"
  [ -f "$PIN_FILE" ] || { echo "no profile at $PIN_FILE: run omxpert/profile.sh first (it learns your experts from your own text)"; exit 1; }
  DEF_ARGS="-fit off -ngl 99 --n-cpu-moe 48 --moe-cache-mib ${PIN_MIB:-36500} -lzm off"
  DEF_ENV="EDX_PIN=$PIN_FILE EDX_PAGEOUT=1 EDX_XDEV=1 EDX_OVERLAP=1 EDX_PREFILL_LOCAL=1"
fi
STACK_ARGS="${STACK_ARGS-$DEF_ARGS}"
STACK_ENV="${STACK_ENV-$DEF_ENV}"
# ---------------------------------------------------------------------------------
cmd=(env $STACK_ENV nice -n 10 "$LLAMA" -m "$GG" --host 127.0.0.1 --port "$PORT" -c "${CTX:-16384}" -np 1 $STACK_ARGS)
if [ "$1" != "smoke" ]; then
  exec "${cmd[@]}"
fi
"${cmd[@]}" > /tmp/omxpert_smoke.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
for i in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/health" | grep -q ok && break
  kill -0 $PID 2>/dev/null || { echo "server died, see /tmp/omxpert_smoke.log"; exit 1; }
  sleep 2
done
curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a Python function that returns the n-th Fibonacci number iteratively.\n\n```python\n","n_predict":96,"temperature":0}' \
  | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['content']); t=r['timings']; print(f\"\n[decode {t['predicted_per_second']:.1f} tok/s, prompt {t['prompt_per_second']:.1f} tok/s]\")"
