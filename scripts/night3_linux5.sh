#!/bin/bash
# Round 5: redo 4-of-64 on OLMoE with unloaded slots dropped; then the whole pipeline on a second model (Granite 3.1 3B-A800M, 40 experts, top-8).
cd ~/expert-dispatch; L=results/night3_linux.log; export CUDA_VISIBLE_DEVICES=0 LR=3e-6
run() { echo "# $(date) $*" >> $L; nice -n 10 timeout 5400 pixi run python "$@" 2>&1 | grep -E "^\{|^round|^start|Error|error" >> $L; }
echo "# $(date) ROUND 5 (fixed: chosen-but-unloaded experts get weight 0)" >> $L
run scripts/swapsearch_olmoe.py 4 10 6
RES=search run scripts/adapter_olmoe.py 4 inf 32 480000 1800
run scripts/adapter_olmoe.py 4 inf 32 480000 1800
run scripts/eval_bundles.py
export MODEL=ibm-granite/granite-3.1-3b-a800m-instruct; G=results/granite-3.1-3b-a800m-instruct; mkdir -p $G
SKIP_LEARN=1 run scripts/residency_olmoe.py 8 120000 0
cp $G/residency_S8_t120000.pt $G/residency_S16_t120000.pt; cp $G/residency_S8_t120000.pt $G/residency_S4_t120000.pt
run scripts/swapsearch_olmoe.py 16 3 4
run scripts/swapsearch_olmoe.py 8 10 6
run scripts/swapsearch_olmoe.py 4 10 6
RES=search run scripts/adapter_olmoe.py 16 inf 32 240000 900
RES=search run scripts/adapter_olmoe.py 8 inf 32 480000 1800
RES=search run scripts/adapter_olmoe.py 4 inf 32 480000 1800
run scripts/eval_bundles.py
echo "# $(date) done5" >> $L; touch results/night3_linux5.done
