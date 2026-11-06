#!/bin/bash
# Fast-lane quality: perplexity with routing restricted to pinned experts (EDX_DRAFT_ALL) vs the full router.
cd ~/expert-dispatch; GG=$(ls ~/models/Qwen3-Coder-30B-A3B-GGUF/*Q4_K_M*.gguf | head -1); R=results/Qwen3-Coder-30B-A3B; OUT=$R/fastlane_ppl.log; : > $OUT
ppl(){ env $1 CUDA_VISIBLE_DEVICES=0 timeout 1500 nice -n 5 llama.cpp/build/bin/llama-perplexity -m $GG -f $2 -c 512 -b 512 -ub 8 -ngl 99 --n-cpu-moe 48 --moe-cache-mib $3 2>&1 | grep -E "Final estimate" | sed -E 's/.*PPL = ([0-9.]+).*/\1/'; }
for t in coding chat; do
  pin=$R/pin_coding.bin; [ $t = chat ] && pin=$R/pin_chat.bin
  echo "$t full $(ppl "EDX_PIN=$pin" /tmp/edx_ppl_$t.txt 5011)" >> $OUT
  for X in 2500 5011 10022; do
    echo "$t fast_own_mib$X $(ppl "EDX_PIN=$pin EDX_DRAFT_ALL=1" /tmp/edx_ppl_$t.txt $X)" >> $OUT
  done
  other=$R/pin_chat.bin; [ $t = chat ] && other=$R/pin_coding.bin
  echo "$t fast_wrongprofile_mib5011 $(ppl "EDX_PIN=$other EDX_DRAFT_ALL=1" /tmp/edx_ppl_$t.txt 5011)" >> $OUT
done
echo "- $(date -Is) fast-lane ppl done" >> NIGHT1_LOG.md; touch fastlane.done
