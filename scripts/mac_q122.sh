#!/bin/bash
# Qwen3.5-122B-A10B Q4_K_M (72 GB, 4.5x the Mac's 16 GB) on the M4: does steering make a model this far over RAM usable?
# Experts read from the mmap (SSD + page cache); attention/shared on Metal. Prompts: first 1200 chars of 3 coding prompts, 64 tokens.
cd ~/expert-dispatch; B=llama.cpp/build/bin; M=models/q122/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf; R=results/q122; O=results/mac
C="-nr -fit off -lm mmap -ngl 99 --n-cpu-moe 48 -c 2048"
for i in 0 1 2; do head -c 1200 $R/prompt_coding_$i.txt > $R/short_$i.txt; done
run() { name=$1; shift
  cm=""; [[ " $* " == *" EDX_PIN="* ]] && cm="--moe-cache-mib 6144"   # the cache flag only marks the hot set (steer-only)
  for i in 0 1 2; do
    out=$(env "$@" X=1 perl -e 'alarm shift; exec @ARGV' 2400 $B/llama-completion -m $M -f $R/short_$i.txt -n 64 -no-cnv --temp 0 --no-display-prompt -v $C $cm 2>&1 >/dev/null)
    ms=$(echo "$out" | perl -ne 'print $1 if /^\S+ I common_perf_print:\s+eval time.*\(\s*([0-9.]+) ms per token/')
    pp=$(echo "$out" | perl -ne 'print $1 if /prompt eval time.*\(\s*([0-9.]+) ms per token/')
    echo "$name short_$i ms=$ms pp_ms=$pp" >> $O/q122.log
    [ -z "$ms" ] && echo "$out" | tail -20 > $O/fail_q122_${name}_$i.log
  done; }
H="EDX_PIN=$R/pin_coding.bin EDX_STEER_ONLY=1 EDX_PIN_MINPL=16"
echo "# $(date) start" >> $O/q122.log
run baseline X=1
run steer1  $H EDX_STEER=1
run steer2  $H EDX_STEER=2
run steer1_skip16 $H EDX_STEER=1 EDX_SKIP=16
# "batch the work": N independent jobs decoding together share each layer's expert reads from the SSD (exact model, no steering)
echo "# $(date) batched (exact model): aggregate tok/s at 1/2/4/8 parallel jobs" >> $O/q122.log
perl -e 'alarm shift; exec @ARGV' 5400 $B/llama-batched-bench -m $M $C -c 4096 -b 512 -ub 512 -npp 128 -ntg 32 -npl 1,2,4,8 2>&1 | grep -E "^\|" >> $O/q122.log
echo "# $(date) done" >> $O/q122.log; touch $O/q122.done
