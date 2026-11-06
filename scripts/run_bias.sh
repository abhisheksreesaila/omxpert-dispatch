#!/bin/bash
# after imatrix frees GPU1: cache-aware routing quality test on OLMoE
cd ~/expert-dispatch
# (sequenced by master.sh)
for tag in coding design_chat; do
  CUDA_VISIBLE_DEVICES=${GPU:-0} pixi run python scripts/torch_bias_ppl.py $tag 40000 2>&1 | grep -v "Loading\|Warning"
done
echo BIAS DONE
