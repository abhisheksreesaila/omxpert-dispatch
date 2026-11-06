#!/bin/bash
# Controlled test, no SSD: Qwen3.6-35B-A3B fully in RAM, one GPU, equal VRAM for experts.
cd ~/expert-dispatch; M=~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf; B=llama.cpp/build/bin/llama-completion
PIN=results/Qwen3.6-35B-A3B/pin_coding.bin; OUT=results/Qwen3.6-35B-A3B/bench35.log; : > $OUT
run(){ # label, env, args
  for p in results/qwen122b/prompt_*.txt; do
    r=$(env $2 CUDA_VISIBLE_DEVICES=0 timeout 600 nice -n 5 $B -m $M -f $p -n 120 -no-cnv --temp 0 -ngl 99 -c 4096 --no-display-prompt -v $3 2>&1 >/dev/null | grep -E "  eval time|hit rate = " | sed -E 's/.*\(\s*([0-9.]+) ms per token.*/ms=\1/; s/.*hit rate = ([0-9.]+)%.*/hit=\1/' | tr '\n' ' ')
    echo "$1 $(basename $p .txt) $r" >> $OUT
  done
}
run allcpu   "X=1" "--n-cpu-moe 40"
for X in 2800 5600; do
  L=$(( X / 466 ))
  run "layers_${X}"  "X=1"                         "--n-cpu-moe $(( 40 - L ))"
  run "slru_${X}"    "X=1"                         "--n-cpu-moe 40 --moe-cache-mib $X"
  run "pinned_${X}"  "EDX_PIN=$PIN EDX_PIN_STATS=1" "--n-cpu-moe 40 --moe-cache-mib $X"
  run "pinnedNS_${X}" "EDX_PIN=$PIN"               "--n-cpu-moe 40 --moe-cache-mib $X"
done
echo "- $(date -Is) bench35 done" >> NIGHT1_LOG.md; touch bench35.done
