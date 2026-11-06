#!/bin/bash
cd ~/expert-dispatch; while [ ! -f n2_tail.done ]; do sleep 30; done
echo "- $(date -Is) IDEA26 speculative-experts trace" >> NIGHT2_LOG.md
head -c 8000 corpus/coding_heldout.txt > /tmp/edx_spec.txt
EDX_TRACE_HIDDEN=results/night2/hidden_coding.f16 CUDA_VISIBLE_DEVICES=0 nice -n 5 llama.cpp/build/bin/llama-edx-trace -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -f /tmp/edx_spec.txt -o /tmp/edx_spec_ids.bin -ngl 99 -c 2048 -b 512 -ub 512 2>&1 | grep EDX_TRACE_DONE >> NIGHT2_LOG.md
nice -n 15 pixi run python scripts/spec_experts_sim.py > results/night2/spec_experts.txt 2>&1; sed 's/^/  /' results/night2/spec_experts.txt >> NIGHT2_LOG.md
echo "- $(date -Is) IDEA26 done" >> NIGHT2_LOG.md; touch n2_spec.done
