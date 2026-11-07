#!/bin/bash
# Night 6 queue 3 (after queue 2): MTP on the pinned stack with the draft head on GPU 0 and fewer pins; launcher smoke test
cd ~/expert-dispatch
until grep -q QUEUE2_DONE results/night6/queue2.log; do sleep 20; done
PENV="EDX_PIN=results/night6/pins/chat.bin EDX_PAGEOUT=1 EDX_XDEV=1 EDX_OVERLAP=1 EDX_PREFILL_LOCAL=1"
MTPF=$HOME/models/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
PROMPTS=chat ROUNDS=4 WARM=8 STACK_ARGS="-fit off -ngl 99 --n-cpu-moe 48 --moe-cache-mib 30000 -lzm off -md $MTPF -devd CUDA0 --spec-type draft-mtp --spec-draft-n-max 2" \
  STACK_ENV="$PENV" timeout 3000 pixi run -q python scripts/n6_ab.py results/night6/ab_pinmtp.json 'mtp=-:{"speculative.n_max":2}' 'nomtp=-:{"speculative.n_max":0}'
STACK=pinned PORT=8090 timeout 1500 omxpert/run.sh smoke > results/night6/smoke_pinned.txt 2>&1; echo "smoke exit $?" >> results/night6/smoke_pinned.txt
echo QUEUE3_DONE
