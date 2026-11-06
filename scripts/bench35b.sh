#!/bin/bash
# Per-use-case profiles: each prompt type with its own profile, a mixed profile, and bigger budgets.
cd ~/expert-dispatch
M=~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf; B=llama.cpp/build/bin/llama-completion; R=results/Qwen3.6-35B-A3B
OUT=$R/bench35b.log; : > $OUT
run(){ # label, prompt glob, env, args
  for p in $2; do
    r=$(env $3 CUDA_VISIBLE_DEVICES=0 timeout 600 nice -n 5 $B -m $M -f $p -n 120 -no-cnv --temp 0 -ngl 99 -c 4096 --no-display-prompt -v $4 2>&1 >/dev/null | grep -E "  eval time" | sed -E 's/.*\(\s*([0-9.]+) ms per token.*/ms=\1/')
    echo "$1 $(basename $p .txt) $r" >> $OUT
  done
}
C="results/qwen122b/prompt_coding_*.txt"; D="results/qwen122b/prompt_design_chat_*.txt"
for X in 2800 5600 11200; do
  L=$(( X / 466 ))
  run "layers_${X}"      "$C $D" "X=1" "--n-cpu-moe $(( 40 - L ))"
  run "slru_${X}"        "$C $D" "X=1" "--n-cpu-moe 40 --moe-cache-mib $X"
  run "pin-coding_${X}"  "$C"    "EDX_PIN=$R/pin_coding.bin" "--n-cpu-moe 40 --moe-cache-mib $X"
  run "pin-chat_${X}"    "$D"    "EDX_PIN=$R/pin_chat.bin"   "--n-cpu-moe 40 --moe-cache-mib $X"
  run "pin-mixed_${X}"   "$C $D" "EDX_PIN=$R/pin_mixed.bin"  "--n-cpu-moe 40 --moe-cache-mib $X"
done
echo "- $(date -Is) bench35b done" >> NIGHT1_LOG.md; touch bench35b.done
