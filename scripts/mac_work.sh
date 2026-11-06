#!/bin/bash
# Work lane on real text: gpt-oss-120b via llama-server, 8 coding tasks graded by hidden tests. NP = parallel slots.
cd ~/expert-dispatch; B=llama.cpp/build/bin; M=models/gptoss/gpt-oss-120b-MXFP4.gguf; R=results; L=results/mac/work.log
H="EDX_PIN=$R/pin_gptoss_coding.bin EDX_STEER_ONLY=1 EDX_PIN_MINPL=8"
one() { name=$1; np=$2; shift 2; cm=""; [[ " $* " == *" EDX_PIN="* ]] && cm="--moe-cache-mib 6144"
  env "$@" X=1 $B/llama-server -m $M -nr -fit off -lm mmap -ngl 99 --n-cpu-moe 36 -np $np -c $((np*3072)) -b 512 -ub 512 --port 8099 --jinja $cm > results/mac/work_server_$name.log 2>&1 & sp=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8099/health >/dev/null && break; sleep 5; done
  perl -e 'alarm shift; exec @ARGV' 5400 python3 scripts/mac_work_tasks.py $name 8099 >> $L 2>> results/mac/work.err
  kill $sp; wait $sp 2>/dev/null; sleep 5; }
echo "# $(date) start (real text; batched-bench used random tokens)" >> $L
one steer1_np1 1 $H EDX_STEER=1
one steer1_np4 4 $H EDX_STEER=1
one steer0.5_np1 1 $H EDX_STEER=0.5
echo "# $(date) done" >> $L; touch results/mac/work.done
