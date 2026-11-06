#!/bin/bash
# Follow-up: a longer swap search at S=8, then the adapter on the searched set (hard mask).
cd ~/expert-dispatch; L=results/night3_linux.log; export CUDA_VISIBLE_DEVICES=0 LR=3e-6
until [ -f results/night3_linux.done ]; do sleep 30; done
run() { echo "# $(date) $*" >> $L; nice -n 10 timeout 5400 pixi run python "$@" 2>&1 | grep -E "^\{|^round|^start|Error|error" >> $L; }
cp results/OLMoE-1B-7B-0125-Instruct/swapsearch_S8.json results/OLMoE-1B-7B-0125-Instruct/swapsearch_S8_r3.json
cp results/OLMoE-1B-7B-0125-Instruct/swapsearch_S8.pt results/OLMoE-1B-7B-0125-Instruct/swapsearch_S8_r3.pt
run scripts/swapsearch_olmoe.py 8 10 6
RES=search run scripts/adapter_olmoe.py 8 inf 32 240000 900
RES=search run scripts/adapter_olmoe.py 16 inf 32 240000 900
echo "# $(date) done2" >> $L; touch results/night3_linux2.done
