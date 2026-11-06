#!/bin/bash
# Mac run 2: steer-only (no expert copies; the profile only marks the hot set) vs run-1 winners; then quality (perplexity, -ub 8 so steering applies).
cd ~/expert-dispatch; B=llama.cpp/build/bin; M=models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf; R=results; O=results/mac; mkdir -p $O
C="-nr -fit off -lm mmap -ngl 99 --n-cpu-moe 48"
num() { perl -ne 'print "$1" if /$ARGV[0]/' ; }
run() { name=$1; shift; envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  for p in p09 p00 p01 p02 p03; do
    out=$(env "${envs[@]}" X=1 perl -e 'alarm shift; exec @ARGV' 900 $B/llama-completion -m $M -f $R/prompts/$p.txt -n 128 -no-cnv --temp 0 -c 4096 --no-display-prompt -v $C "$@" 2>&1 >/dev/null)
    ms=$(echo "$out" | perl -ne 'print $1 if /^\S+ I common_perf_print:\s+eval time.*\(\s*([0-9.]+) ms per token/')
    echo "$name $p ms=$ms" >> $O/bench2.log
    [ -z "$ms" ] && echo "$out" | tail -15 > $O/fail2_${name}_$p.log
  done; }
ppl() { name=$1; shift; envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  out=$(env "${envs[@]}" X=1 perl -e 'alarm shift; exec @ARGV' 2400 $B/llama-perplexity -m $M -f $R/coding_heldout.txt -c 512 --chunks 12 -b 8 -ub 8 $C "$@" 2>&1)
  echo "$name ppl=$(echo "$out" | perl -ne 'print $1 if /Final estimate: PPL = ([0-9.]+)/') $(echo "$out" | perl -ne 'print "s_per_pass=$1" if /([0-9.]+) seconds per pass/')" >> $O/ppl2.log; }
echo "# $(date) start" >> $O/bench2.log
run experts_cpu -- 
run steeronly4g_l1  EDX_PIN=$R/pin_coding.bin EDX_STEER_ONLY=1 EDX_STEER=1 EDX_PIN_MINPL=16 -- --moe-cache-mib 4096
run steeronly6g_l1  EDX_PIN=$R/pin_coding.bin EDX_STEER_ONLY=1 EDX_STEER=1 EDX_PIN_MINPL=16 -- --moe-cache-mib 6144
run steeronly8g_l1  EDX_PIN=$R/pin_coding.bin EDX_STEER_ONLY=1 EDX_STEER=1 EDX_PIN_MINPL=16 -- --moe-cache-mib 8192
run steeronly6g_l0.5 EDX_PIN=$R/pin_coding.bin EDX_STEER_ONLY=1 EDX_STEER=0.5 EDX_PIN_MINPL=16 -- --moe-cache-mib 6144
run steeronly6g_l2  EDX_PIN=$R/pin_coding.bin EDX_STEER_ONLY=1 EDX_STEER=2 EDX_PIN_MINPL=16 -- --moe-cache-mib 6144
run pin6g_steer1    EDX_PIN=$R/pin_coding.bin EDX_PAGEOUT=1 EDX_STEER=1 EDX_PIN_MINPL=16 -- --moe-cache-mib 6144
echo "# $(date) speed done" >> $O/bench2.log
ppl plain --
ppl steeronly6g_l0.5 EDX_PIN=$R/pin_coding.bin EDX_STEER_ONLY=1 EDX_STEER=0.5 EDX_PIN_MINPL=16 -- --moe-cache-mib 6144
ppl steeronly6g_l1  EDX_PIN=$R/pin_coding.bin EDX_STEER_ONLY=1 EDX_STEER=1 EDX_PIN_MINPL=16 -- --moe-cache-mib 6144
ppl steeronly6g_l2  EDX_PIN=$R/pin_coding.bin EDX_STEER_ONLY=1 EDX_STEER=2 EDX_PIN_MINPL=16 -- --moe-cache-mib 6144
echo "# $(date) done" >> $O/bench2.log; touch $O/bench2.done
