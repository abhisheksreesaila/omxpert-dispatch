#!/bin/bash
# Night 2: real decode speed on Qwen3.5-122B-A10B Q4_K_M, 2x3090, experts of the
# first NCPU layers in RAM, with and without llama.cpp's SLRU GPU expert cache.
cd ~/expert-dispatch
M=$(ls ~/models/Qwen3.5-122B-A10B-GGUF/*/*00001-of-*.gguf ~/models/Qwen3.5-122B-A10B-GGUF/*00001-of-*.gguf 2>/dev/null | head -1)
NCPU=${NCPU:-34}; mkdir -p results/qwen122b
avail=$(free -g | awk '/Mem:/{print $7}'); [ "$avail" -lt 50 ] && { echo "- $(date -Is) BENCH122: RAM avail ${avail}G < 50, not starting" >> NIGHT1_LOG.md; exit 1; }
pixi run python - <<'PY'
from pathlib import Path
W=Path.home()/"expert-dispatch"; out=W/"results/qwen122b"
for tag in ("coding","design_chat"):
    t=(W/f"corpus/{tag}_heldout.txt").read_text()
    for i in range(3):
        (out/f"prompt_{tag}_{i}.txt").write_text(t[i*50000:i*50000+3000])
PY
run(){ # name, extra args
  for p in results/qwen122b/prompt_*.txt; do
    nice -n 5 llama.cpp/build/bin/llama-completion -m "$M" -f $p -n 200 -no-cnv --temp 0 -ngl 99 --n-cpu-moe $NCPU -ts 4,1 -lv 3 \
      -c 4096 --no-display-prompt $2 2>&1 | grep -E "eval time|hit rate|model buffer size" | sed "s|^|$1 $(basename $p) |"
  done
}
{ run nocache ""; run slru8g "--moe-cache-mib 8000"; } > results/qwen122b/bench.log
echo "- $(date -Is) BENCH122 done (NCPU=$NCPU)" >> NIGHT1_LOG.md
touch bench122.done
