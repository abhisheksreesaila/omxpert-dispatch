#!/bin/bash
cd ~/expert-dispatch; M2=Qwen3.5-122B-A10B
GG=~/models/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf
for f in corpus/imx122/coding_heldout.txt corpus/imx122/coding_train.txt corpus/imx122/design_chat_heldout.txt corpus/imx122/design_chat_train.txt; do
  n=$(basename $f .txt); [ -f traces/$M2/$n.npz ] && continue
  nice -n 5 llama.cpp/build/bin/llama-edx-trace -m $GG -f $f -o traces/$M2/$n.bin -ngl 99 --n-cpu-moe 34 -ts 5,1 -c 2048 -b 256 -ub 256 2>&1 | grep -E "EDX|failed| E "
  pixi run python -c "
import numpy as np; a=np.fromfile('traces/$M2/$n.bin',dtype=np.int32).reshape(-1,48,8)
np.savez('traces/$M2/$n.npz', topk=a.astype(np.int16)); print('$n', a.shape)" && rm traces/$M2/$n.bin
done
echo "- $(date -Is) 122B traces done" >> NIGHT1_LOG.md; touch trace122.done
