#!/bin/bash
# Work lane: gpt-oss-120b serving 8 coding jobs at once; exact vs steered. Graded by hidden tests.
cd ~/expert-dispatch; B=llama.cpp/build/bin; M=models/gptoss/gpt-oss-120b-MXFP4.gguf; R=results; L=results/mac/work.log
C="-nr -fit off -lm mmap -ngl 99 --n-cpu-moe 36 -np 8 -c 24576 -b 512 -ub 512 --port 8099 --jinja"
H="EDX_PIN=$R/pin_gptoss_coding.bin EDX_STEER_ONLY=1 EDX_PIN_MINPL=8"
one() { name=$1; shift; cm=""; [[ " $* " == *" EDX_PIN="* ]] && cm="--moe-cache-mib 6144"
  env "$@" X=1 $B/llama-server -m $M $C $cm > results/mac/work_server_$name.log 2>&1 & sp=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8099/health >/dev/null && break; sleep 5; done
  perl -e 'alarm shift; exec @ARGV' 7200 python3 scripts/mac_work_tasks.py $name 8099 >> $L 2>> results/mac/work.err
  kill $sp; wait $sp 2>/dev/null; sleep 5; }
echo "# $(date) start" >> $L
one steer1 $H EDX_STEER=1
one steer0.5 $H EDX_STEER=0.5
one exact
echo "# $(date) done" >> $L; touch results/mac/work.done
