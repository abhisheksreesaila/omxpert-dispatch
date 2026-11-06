#!/bin/bash
# Night 2 tail: probability trace -> P4 sim -> P5 sim
cd ~/expert-dispatch; while [ ! -f n2_rest.done ]; do sleep 30; done
echo "- $(date -Is) PHASE5 probs trace" >> NIGHT2_LOG.md
head -c 40000 corpus/coding_heldout.txt > /tmp/edx_p4.txt
EDX_TRACE_PROBS=results/night2/probs_coding_heldout.f16 CUDA_VISIBLE_DEVICES=0 nice -n 5 llama.cpp/build/bin/llama-edx-trace -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -f /tmp/edx_p4.txt -o /tmp/edx_p4.bin -ngl 99 -c 2048 -b 512 -ub 512 2>&1 | grep EDX_TRACE_DONE >> NIGHT2_LOG.md
pixi run python -c "
import numpy as np; a=np.fromfile('/tmp/edx_p4.bin',dtype=np.int32).reshape(-1,40,8); np.savez('results/night2/ids_coding_heldout.npz', topk=a.astype(np.int16)); print(a.shape)"
(cd scripts && nice -n 15 pixi run python n2_p4_sim.py) > results/night2/p4_sim.txt 2>&1; sed 's/^/  /' results/night2/p4_sim.txt >> NIGHT2_LOG.md
echo "- $(date -Is) PHASE6 P5 sim" >> NIGHT2_LOG.md
nice -n 15 pixi run python scripts/n2_p5_sim.py Qwen3.6-35B-A3B coding > results/night2/p5_sim.txt 2>&1; sed 's/^/  /' results/night2/p5_sim.txt >> NIGHT2_LOG.md
echo "- $(date -Is) night2 tail done" >> NIGHT2_LOG.md; touch n2_tail.done
