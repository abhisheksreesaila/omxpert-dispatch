#!/bin/bash
cd ~/expert-dispatch
for tag in coding design_chat tool_calls docs math; do
  CUDA_VISIBLE_DEVICES=1 pixi run python scripts/torch_trace.py allenai/OLMoE-1B-7B-0125-Instruct $tag train 300000 2>&1 | grep -v "Loading\|Warning"
  CUDA_VISIBLE_DEVICES=1 pixi run python scripts/torch_trace.py allenai/OLMoE-1B-7B-0125-Instruct $tag heldout 100000 2>&1 | grep -v "Loading\|Warning"
done
echo ALL DONE
