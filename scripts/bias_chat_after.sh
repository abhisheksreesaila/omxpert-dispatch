#!/bin/bash
cd ~/expert-dispatch
while pgrep -f "llama-imatrix|run_imatrix" >/dev/null; do sleep 20; done
CUDA_VISIBLE_DEVICES=1 pixi run python scripts/torch_bias_ppl.py design_chat 40000 2>&1 | grep -E "^baseline|^\{" > bias_chat.log
echo DONE >> bias_chat.log
