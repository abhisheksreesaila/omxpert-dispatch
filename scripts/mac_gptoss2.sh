#!/bin/bash
# gpt-oss-120b (MXFP4 as released, 63 GB = 4x the Mac's RAM): exact model vs steering toward its coding profile; batching; quality.
cd ~/expert-dispatch; B=llama.cpp/build/bin; M=models/gptoss/gpt-oss-120b-MXFP4.gguf; R=results; O=results/mac
C="-nr -fit off -lm mmap -ngl 99 --n-cpu-moe 36"
for i in 0 1 2; do head -c 1200 $R/q122/prompt_coding_$i.txt > $R/short_$i.txt 2>/dev/null || head -c 1200 $R/prompts/p0$i.txt > $R/short_$i.txt; done
run() { name=$1; shift; cm=""; [[ " $* " == *" EDX_PIN="* ]] && cm="--moe-cache-mib 6144"
  for i in 0 1 2; do
    out=$(env "$@" X=1 perl -e 'alarm shift; exec @ARGV' 2400 $B/llama-completion -m $M -f $R/short_$i.txt -n 96 -no-cnv --temp 0 --no-display-prompt -v $C -c 4096 $cm 2>&1 >/dev/null)
    ms=$(echo "$out" | perl -ne 'print $1 if /^\S+ I common_perf_print:\s+eval time.*\(\s*([0-9.]+) ms per token/')
    pp=$(echo "$out" | perl -ne 'print $1 if /prompt eval time.*\(\s*([0-9.]+) ms per token/')
    echo "$name short_$i ms=$ms pp_ms=$pp" >> $O/gptoss2.log
    [ -z "$ms" ] && echo "$out" | tail -20 > $O/fail_gptoss_${name}_$i.log
  done; }
H="EDX_PIN=$R/pin_gptoss_coding.bin EDX_STEER_ONLY=1 EDX_PIN_MINPL=8"
ppl() { name=$1; shift; cm=""; [[ " $* " == *" EDX_PIN="* ]] && cm="--moe-cache-mib 6144"
  out=$(env "$@" X=1 perl -e 'alarm shift; exec @ARGV' 5400 $B/llama-perplexity -m $M -f $R/coding_heldout.txt -c 512 --chunks 4 -b 8 -ub 8 $C $cm 2>&1)
  echo "ppl $name $(echo "$out" | perl -ne 'print $1 if /Final estimate: PPL = ([0-9.]+)/')" >> $O/gptoss2.log; }
echo "# $(date) start: steering fixed for logit routers (logit - lambda)" >> $O/gptoss2.log
run steer0.5 $H EDX_STEER=0.5
run steer1 $H EDX_STEER=1
run steer2 $H EDX_STEER=2
ppl steer1 $H EDX_STEER=1
ppl steer2 $H EDX_STEER=2
ppl steer0.5 $H EDX_STEER=0.5
echo "# $(date) batched, steer λ=1" >> $O/gptoss2.log
env $H EDX_STEER=1 perl -e 'alarm shift; exec @ARGV' 5400 $B/llama-batched-bench -m $M $C --moe-cache-mib 6144 -c 8192 -b 512 -ub 512 -npp 128 -ntg 32 -npl 1,2,4,8 2>&1 | grep -E "^\|" >> $O/gptoss2.log
echo "# $(date) done" >> $O/gptoss2.log; touch $O/gptoss2.done
