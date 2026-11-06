#!/bin/bash
# Round 4: co-optimise the S=4 bundle: search the set with the adapter active, then retrain the adapter on the new set.
cd ~/expert-dispatch; L=results/night3_linux.log; export CUDA_VISIBLE_DEVICES=0 LR=3e-6; R=results/OLMoE-1B-7B-0125-Instruct
run() { echo "# $(date) $*" >> $L; nice -n 10 timeout 5400 pixi run python "$@" 2>&1 | grep -E "^\{|^round|^start|Error|error" >> $L; }
TAG=_co START=$R/swapsearch_S4.pt ADAPTER=$R/adapter_coding_search_S4_linf_r32_t480000.pt run scripts/swapsearch_olmoe.py 4 4 6
cp $R/swapsearch_S4.pt $R/swapsearch_S4_plain.pt; cp $R/swapsearch_S4_co.pt $R/swapsearch_S4.pt
RES=search run scripts/adapter_olmoe.py 4 inf 32 480000 1800
cp $R/swapsearch_S4_plain.pt $R/swapsearch_S4.pt
echo "# $(date) done4" >> $L; touch results/night3_linux4.done
