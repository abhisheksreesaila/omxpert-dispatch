#!/bin/bash
# Per-token Qwen3.6-35B-A3B routing traces via llama.cpp (tools/edx-trace), then analysis.
cd ~/expert-dispatch
M=Qwen3.6-35B-A3B; mkdir -p traces/$M
for f in corpus/imx/*_train.txt corpus/imx/*_heldout.txt; do
  n=$(basename $f .txt); [ -f traces/$M/$n.npz ] && continue
  nice -n 10 llama.cpp/build/bin/llama-edx-trace -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -f $f -o traces/$M/$n.bin \
    -ngl 99 -c 2048 -b 512 -ub 512 2>&1 | grep -E "EDX_TRACE_DONE|failed|error"
  pixi run python -c "
import numpy as np; a=np.fromfile('traces/$M/$n.bin',dtype=np.int32).reshape(-1,40,8)
np.savez('traces/$M/$n.npz', topk=a.astype(np.int16)); print('$n', a.shape)" && rm traces/$M/$n.bin
done
echo "- $(date -Iseconds) Qwen per-token traces done (llama.cpp edx-trace)" >> NIGHT1_LOG.md
nice -n 10 pixi run python scripts/analyze.py $M > qwen_analyze.log 2>&1
echo "- $(date -Iseconds) Qwen analysis exit $?" >> NIGHT1_LOG.md
touch qwen_trace.done
