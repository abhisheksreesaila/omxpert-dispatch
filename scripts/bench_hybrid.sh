#!/bin/bash
# Idea 20: F layers fully on GPU (the last F, as --n-cpu-moe keeps the first N on CPU) + the rest of the VRAM budget pinned.
cd ~/expert-dispatch; while [ ! -f coder.done ]; do sleep 30; done; scripts/wait_quiet.sh
M=~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf; B=llama.cpp/build/bin/llama-completion; R=results/Qwen3.6-35B-A3B; OUT=$R/bench_hybrid.log; : > $OUT
for X in 5600 11200; do
  LT=$(( X / 466 ))
  for F in 0 4 8 12 16 20 24; do
    [ $F -gt $LT ] && continue
    C=$(( X - F * 466 )); args="--n-cpu-moe $(( 40 - F ))"; env="EDX_PIN=$R/pin_coding.bin"
    [ $C -ge 300 ] && args="$args --moe-cache-mib $C" || env="X=1"
    for p in results/qwen122b/prompt_coding_*.txt; do
      ms=$(env $env CUDA_VISIBLE_DEVICES=0 timeout 600 nice -n 5 $B -m $M -f $p -n 120 -no-cnv --temp 0 -ngl 99 -c 4096 --no-display-prompt -v $args 2>&1 >/dev/null | grep "  eval time" | sed -E 's/.*\(\s*([0-9.]+) ms per token.*/\1/')
      echo "budget=$X full_layers=$F pinned_mib=$C $(basename $p .txt) ms=$ms" >> $OUT
    done
  done
done
echo "- $(date -Is) hybrid bench done" >> NIGHT1_LOG.md; touch hybrid.done
