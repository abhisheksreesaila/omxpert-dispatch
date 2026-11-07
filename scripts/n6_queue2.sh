#!/bin/bash
# Night 6 queue 2: same-method (server timings, one server per config) comparisons of fit vs the full pinned stack, interleaved;
# per-request A/Bs inside the pinned stack (prefetch prev vs off; MTP on vs off); coding-prompt runs for the quality check.
cd ~/expert-dispatch
FIT="-lzm off"
PIN="-fit off -ngl 99 --n-cpu-moe 48 --moe-cache-mib 36500 -lzm off"
PENV="EDX_PIN=results/night6/pins/chat.bin EDX_PAGEOUT=1 EDX_XDEV=1 EDX_OVERLAP=1 EDX_PREFILL_LOCAL=1"
MTPF=$HOME/models/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
ab() { timeout 3000 pixi run -q python scripts/n6_ab.py "$@"; }
export PROMPTS=chat ROUNDS=4 WARM=8
echo off > /tmp/n6pf.ctl
for i in 1 2; do
  STACK_ARGS="$FIT" STACK_ENV="" ab results/night6/ab_fit_$i.json base=-
  STACK_ARGS="$PIN" STACK_ENV="$PENV EDX_PF_CTL=/tmp/n6pf.ctl" CTL=/tmp/n6pf.ctl ab results/night6/ab_pin_$i.json pfprev=prev pfoff=off
done
STACK_ARGS="-fit off -ngl 99 --n-cpu-moe 48 --moe-cache-mib 33000 -lzm off -md $MTPF --spec-type draft-mtp --spec-draft-n-max 2" STACK_ENV="$PENV" \
  ab results/night6/ab_pinmtp.json 'mtp=-:{"speculative.n_max":2}' 'nomtp=-:{"speculative.n_max":0}'
export PROMPTS=coding ROUNDS=1 WARM=4
STACK_ARGS="$FIT" STACK_ENV="" ab results/night6/q_fit_a.json base=-
STACK_ARGS="$PIN" STACK_ENV="$PENV" ab results/night6/q_pin.json base=-
STACK_ARGS="$FIT" STACK_ENV="" ab results/night6/q_fit_b.json base=-
STACK_ARGS="$PIN" STACK_ENV="$PENV EDX_STEER=0.5" ab results/night6/q_pinsteer.json base=-
echo QUEUE2_DONE
