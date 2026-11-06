#!/bin/bash
# Night 2 Phase 2: decode 30 held-out coding prompts x 256 tokens under three placements, with per-token ids + timestamps.
cd ~/expert-dispatch; while [ ! -f n2_phase1.done ]; do sleep 30; done
(cd llama.cpp && nice -n 10 cmake --build build --target llama-completion llama-perplexity llama-speculative-simple llama-server -j 8 2>&1 | grep -E " error") && { echo "- $(date -Is) PHASE2 build FAILED" >> NIGHT2_LOG.md; exit 1; }
scripts/wait_quiet.sh; echo "- $(date -Is) PHASE2 start" >> NIGHT2_LOG.md
O=results/night2/phase2; mkdir -p $O/prompts
pixi run python - <<'PY'
from pathlib import Path
W=Path.home()/"expert-dispatch"; t=(W/"corpus/coding_heldout.txt").read_text(); n=30; step=len(t)//n
for i in range(n): (W/f"results/night2/phase2/prompts/p{i:02d}.txt").write_text(t[i*step:i*step+2000])
PY
M=~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf; R=results/Qwen3.6-35B-A3B; X=2800; : > $O/times.log; : > $O/traced_times.log
for cfg in "pin_coding EDX_PIN=$R/pin_coding.bin --n-cpu-moe=40 --moe-cache-mib=$X" "pin_mixed EDX_PIN=$R/pin_mixed.bin --n-cpu-moe=40 --moe-cache-mib=$X" "layers X=1 --n-cpu-moe=34"; do
  set -- $cfg; name=$1; env1=$2; shift 2; args=$(echo "$@" | tr '=' ' ')
  for p in $O/prompts/p*.txt; do
    b=$(basename $p .txt); tr=""; [ $name != layers ] && tr="EDX_TRACE_DECODE=$O/${name}_$b.trace"
    run() { env $env1 "$@" CUDA_VISIBLE_DEVICES=0 timeout 400 nice -n 5 llama.cpp/build/bin/llama-completion -m $M -f $p -n 256 -no-cnv --temp 0 -ngl 99 -c 4096 --no-display-prompt -v $args 2>&1 >/dev/null | grep "  eval time" | sed -E 's/.*\(\s*([0-9.]+) ms per token.*/\1/'; }
    l0=$(cut -d' ' -f1 /proc/loadavg); ms=$(run X=1)          # timed run, no tracer (the tracer reads ids back every token)
    echo "$name $b ms=$ms load=$l0" >> $O/times.log
    [ -n "$tr" ] && echo "$name $b traced_ms=$(run $tr)" >> $O/traced_times.log
  done
done
echo "- $(date -Is) PHASE2 decode done" >> NIGHT2_LOG.md
pixi run python scripts/n2_p1_analysis.py >> NIGHT2_LOG.md 2>&1
touch n2_phase2.done
