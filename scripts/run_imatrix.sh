#!/bin/bash
cd ~/expert-dispatch; mkdir -p imatrix/qwen36
M=~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
for f in corpus/imx/*.txt; do
  n=$(basename $f .txt)
  [ -f imatrix/qwen36/$n.gguf ] && continue
  CUDA_VISIBLE_DEVICES=${GPU:-1} llama.cpp/build/bin/llama-imatrix -m $M -f $f -o imatrix/qwen36/$n.gguf \
    -ngl 99 --no-ppl -c 2048 -b 2048 2>&1 | tail -2
done
pixi run python scripts/read_counts.py > imatrix/read_counts.log 2>&1; echo IMATRIX DONE
