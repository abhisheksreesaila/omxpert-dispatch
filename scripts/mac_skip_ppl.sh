#!/bin/bash
# P6 skip-the-miss quality on Qwen3-Coder-30B (Mac): coding held-out perplexity, hot set = coding profile 6 GB (steer-only, no copies)
cd ~/expert-dispatch; B=llama.cpp/build/bin; M=models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf; R=results; O=results/mac
C="-nr -fit off -lm mmap -ngl 99 --n-cpu-moe 48 --moe-cache-mib 6144"
H="EDX_PIN=$R/pin_coding.bin EDX_STEER_ONLY=1 EDX_PIN_MINPL=16"
ppl() { name=$1; shift
  out=$(env $H "$@" perl -e 'alarm shift; exec @ARGV' 2400 $B/llama-perplexity -m $M -f $R/coding_heldout.txt -c 512 --chunks 12 -b 8 -ub 8 $C 2>&1)
  echo "$name ppl=$(echo "$out" | perl -ne 'print $1 if /Final estimate: PPL = ([0-9.]+)/') $(echo "$out" | perl -ne 'print "s_per_pass=$1" if /([0-9.]+) seconds per pass/')" >> $O/ppl_skip.log; }
echo "# $(date) start (during the 122B copy: s/pass not comparable)" >> $O/ppl_skip.log
ppl skip_top16mass EDX_SKIP=16
ppl skip_top8mass  EDX_SKIP=8
ppl skip_survivors EDX_SKIP=s
ppl steer1_skip16  EDX_SKIP=16 EDX_STEER=1
echo "# $(date) done" >> $O/ppl_skip.log; touch $O/ppl_skip.done
