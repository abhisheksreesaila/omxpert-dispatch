#!/bin/bash
cd ~/expert-dispatch
HF_HUB_DISABLE_XET=1 pixi run python -c "
from huggingface_hub import snapshot_download
p=snapshot_download('unsloth/Qwen3.5-122B-A10B-GGUF', allow_patterns=['*Q4_K_M*'], local_dir='$HOME/models/Qwen3.5-122B-A10B-GGUF')
print('done', p)"
echo "- $(date -Iseconds) 122B download exit $?" >> NIGHT1_LOG.md
