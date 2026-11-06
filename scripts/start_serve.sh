#!/bin/bash
# usage: start_serve.sh [bias.npy]   — MAX serve gpt-oss-20b with Expert Dispatch hooks
cd ~/expert-dispatch/max
export EDX_HOOK=1 EDX_OUT=$HOME/expert-dispatch/traces/max EDX_ACT_GIB=${EDX_ACT_GIB:-9}
[ -n "$1" ] && export EDX_BIAS=$1 || unset EDX_BIAS
pixi run max serve --model openai/gpt-oss-20b --custom-architectures $HOME/expert-dispatch/max:edx_gptoss \
  --devices ${EDX_DEVICES:-gpu:0,1} --max-length 4096 --max-batch-size 2 --max-batch-input-tokens ${EDX_BIT:-2048} \
  --device-memory-utilization ${EDX_UTIL:-0.7} --port 8000 > serve.log 2>&1 &
for i in $(seq 300); do grep -qE "Model worker ready|Traceback|RuntimeError" serve.log && break; sleep 3; done
grep -E "Model worker ready|Error" serve.log | tail -3; cat $EDX_OUT/IMPORTED.log 2>/dev/null | tail -4
