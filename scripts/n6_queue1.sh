#!/bin/bash
# Night 6 queue 1: pinned chat profile + cross-device prefill copies, + overlap, vs llama.cpp's own fit, on held-out chat prompts
cd ~/expert-dispatch
PIN="-fit off -ngl 99 --n-cpu-moe 48 --moe-cache-mib 34000 -lzm off"
PENV="EDX_PIN=results/night6/pins/chat.bin EDX_PAGEOUT=1 EDX_XDEV=1"
export PROMPTS=chat WARM=8 ROUNDS=2
STACK_ARGS="$PIN" STACK_ENV="$PENV" timeout 3000 scripts/n6_llama.sh pinx_chat
STACK_ARGS="$PIN" STACK_ENV="$PENV EDX_OVERLAP=1" timeout 3000 scripts/n6_llama.sh pinxo_chat
STACK_ARGS="-lzm off" STACK_ENV="" timeout 3000 scripts/n6_llama.sh base_chat
echo QUEUE1_DONE
