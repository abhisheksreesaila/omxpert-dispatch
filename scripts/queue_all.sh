#!/bin/bash
# Run when the machine is free. Strictly one GPU job at a time.
cd ~/expert-dispatch
scripts/max_full.sh > max_full.log 2>&1
for mu in 1.0 3.0; do
  CUDA_VISIBLE_DEVICES=0 pixi run python scripts/learn_cache.py 16 $mu 120000 200 2>&1 | grep -E "^\{|^step|Error|error" > learn_cache_mu$mu.log
done
CUDA_VISIBLE_DEVICES=0 pixi run python scripts/torch_bias_ppl.py design_chat 40000 2>&1 | grep -E "^baseline|^\{" > bias_chat.log
echo QUEUE DONE > queue.done
