#!/bin/bash
cd ~/expert-dispatch; GG=$(ls ~/models/Qwen3-Coder-30B-A3B-GGUF/*Q4_K_M*.gguf | head -1); R=results/Qwen3-Coder-30B-A3B; OUT=$R/bench_prompt.log; : > $OUT
for X in 2500 5011; do
for p in results/qwen122b/prompt_*.txt; do
  pin=$R/pin_coding.bin; case $p in *chat*) pin=$R/pin_chat.bin;; esac
  for cfg in "profile X=1" "blend0.8 EDX_PIN_PROMPT=0.8" "blend0.5 EDX_PIN_PROMPT=0.5" "prompt_only EDX_PIN_PROMPT=0.0"; do set -- $cfg
    ms=$(env $2 EDX_PIN=$pin CUDA_VISIBLE_DEVICES=0 timeout 400 nice -n 5 llama.cpp/build/bin/llama-completion -m $GG -f $p -n 200 -no-cnv --temp 0 -ngl 99 --n-cpu-moe 48 --moe-cache-mib $X -c 4096 --no-display-prompt -v 2>&1 >/dev/null | grep "  eval time" | sed -E 's/.*\(\s*([0-9.]+) ms per token.*/\1/')
    hit=$(env $2 EDX_HOSTMAP=1 EDX_PIN_STATS=1 EDX_PIN=$pin CUDA_VISIBLE_DEVICES=0 timeout 400 nice -n 5 llama.cpp/build/bin/llama-completion -m $GG -f $p -n 200 -no-cnv --temp 0 -ngl 99 --n-cpu-moe 48 --moe-cache-mib $X -c 4096 --no-display-prompt -v 2>&1 >/dev/null | grep "hit rate = " | sed -E 's/.*hit rate = ([0-9.]+)%.*/\1/')
    echo "mib=$X $1 $(basename $p .txt) ms=$ms hit=$hit" >> $OUT
  done
done; done
echo "- $(date -Is) prompt-lineup bench done" >> NIGHT1_LOG.md; touch bench_prompt.done
