#!/bin/bash
cd ~/expert-dispatch
HF_HUB_DISABLE_XET=1 nice -n 10 pixi run python -c "
from huggingface_hub import snapshot_download
p=snapshot_download('unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF', allow_patterns=['*Q4_K_M*'], local_dir='$HOME/models/Qwen3-Coder-30B-A3B-GGUF')
print('done', p)"
echo "- $(date -Is) Qwen3-Coder-30B download exit $?" >> NIGHT1_LOG.md
