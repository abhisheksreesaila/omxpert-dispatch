#!/bin/bash
# Soft steering toward pinned experts (EDX_STEER=lambda): perplexity, decode speed, hit rate. Qwen3-Coder.
cd ~/expert-dispatch; GG=$(ls ~/models/Qwen3-Coder-30B-A3B-GGUF/*Q4_K_M*.gguf | head -1); R=results/Qwen3-Coder-30B-A3B; OUT=$R/steer.log; : > $OUT
B=llama.cpp/build/bin
for t in coding chat; do
  pin=$R/pin_coding.bin; P="results/qwen122b/prompt_coding_*.txt"; [ $t = chat ] && { pin=$R/pin_chat.bin; P="results/qwen122b/prompt_design_chat_*.txt"; }
  for X in 2500 5011; do
    for cfg in "pinned X=1" "pinned_min16 EDX_PIN_MINPL=16" "steer0.5 EDX_PIN_MINPL=16,EDX_STEER=0.5" "steer1 EDX_PIN_MINPL=16,EDX_STEER=1" "steer2 EDX_PIN_MINPL=16,EDX_STEER=2"; do
      set -- $cfg; E=$(echo $2 | tr ',' ' ')
      ppl=$(env $E EDX_PIN=$pin CUDA_VISIBLE_DEVICES=0 timeout 1200 nice -n 5 $B/llama-perplexity -m $GG -f /tmp/edx_ppl_${t}_s.txt -c 512 -b 512 -ub 8 -ngl 99 --n-cpu-moe 48 --moe-cache-mib $X 2>&1 | grep "Final estimate" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')
      ms=0; n=0; hs=0
      for p in $P; do
        m=$(env $E EDX_PIN=$pin CUDA_VISIBLE_DEVICES=0 timeout 400 nice -n 5 $B/llama-completion -m $GG -f $p -n 160 -no-cnv --temp 0 -ngl 99 --n-cpu-moe 48 --moe-cache-mib $X -c 4096 --no-display-prompt -v 2>&1 >/dev/null | grep "  eval time" | sed -E 's/.*\(\s*([0-9.]+) ms per token.*/\1/')
        h=$(env $E EDX_HOSTMAP=1 EDX_PIN_STATS=1 EDX_PIN=$pin CUDA_VISIBLE_DEVICES=0 timeout 400 nice -n 5 $B/llama-completion -m $GG -f $p -n 160 -no-cnv --temp 0 -ngl 99 --n-cpu-moe 48 --moe-cache-mib $X -c 4096 --no-display-prompt -v 2>&1 >/dev/null | grep "hit rate = " | sed -E 's/.*hit rate = ([0-9.]+)%.*/\1/')
        ms=$(python3 -c "print($ms+${m:-0})"); hs=$(python3 -c "print($hs+${h:-0})"); n=$((n+1))
      done
      echo "$t mib=$X $1 ppl=$ppl ms=$(python3 -c "print(round($ms/$n,1))") hit=$(python3 -c "print(round($hs/$n,1))")" >> $OUT
    done
  done
done
echo "- $(date -Is) steer bench done" >> NIGHT1_LOG.md; touch steer.done
