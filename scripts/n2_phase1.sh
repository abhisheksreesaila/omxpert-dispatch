#!/bin/bash
# Night 2 Phase 1: measure PCIe upload BW, CPU expert cost, per-layer CPU-hop (sync) cost.
cd ~/expert-dispatch; while [ ! -f steer.done ]; do sleep 30; done; scripts/wait_quiet.sh
O=results/night2; echo "- $(date -Is) PHASE1 start" >> NIGHT2_LOG.md
CUDA_VISIBLE_DEVICES=0 tools/pcie_bw > $O/pcie.jsonl
M=~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf; : > $O/decode_grid.log
for k in 8 4; do for N in 0 10 20 30 40; do for p in results/qwen122b/prompt_coding_0.txt results/qwen122b/prompt_coding_1.txt; do
  ms=$(CUDA_VISIBLE_DEVICES=0 timeout 400 nice -n 5 llama.cpp/build/bin/llama-completion -m $M -f $p -n 128 -no-cnv --temp 0 -ngl 99 --n-cpu-moe $N -c 4096 --no-display-prompt --override-kv qwen35moe.expert_used_count=int:$k -v 2>&1 >/dev/null | grep "  eval time" | sed -E 's/.*\(\s*([0-9.]+) ms per token.*/\1/')
  echo "k=$k ncpu=$N $(basename $p .txt) ms=$ms" >> $O/decode_grid.log
done; done; done
pixi run python scripts/n2_fit_costs.py
echo "- $(date -Is) PHASE1 done: $(cat $O/costs.json | tr -d '\n' | cut -c1-300)" >> NIGHT2_LOG.md; touch n2_phase1.done
