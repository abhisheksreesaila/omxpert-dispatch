#!/bin/bash
# Night 6: routing traces of Qwen3.8-Flash-Next on the owner's chat history (train + held-out). Traces hold expert ids only.
cd ~/expert-dispatch; MD=Qwen3.8-Flash-Next; mkdir -p traces/$MD
GG=$(ls ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/*00001-of-*.gguf)
for n in chat_train chat_heldout; do
  [ -f traces/$MD/$n.npz ] && continue
  C=${CHARS:-240000}; [ $n = chat_heldout ] && C=100000
  head -c $C private/chat/$n.txt > private/chat/_$n.txt
  CUDA_VISIBLE_DEVICES=0,1 nice -n 10 llama.cpp/build/bin/llama-edx-trace -m $GG -f private/chat/_$n.txt -o traces/$MD/$n.bin -c 2048 -b 512 -ub 512 2>&1 | grep -E "EDX|failed| E "
  pixi run python -c "
import numpy as np; a=np.fromfile('traces/$MD/$n.bin',dtype=np.int32).reshape(-1,48,10)
np.savez('traces/$MD/$n.npz', topk=a.astype(np.int16)); print('$n', a.shape)" && rm traces/$MD/$n.bin
done
echo TRACE_DONE
