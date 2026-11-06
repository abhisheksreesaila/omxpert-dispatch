#!/bin/bash
# Night 3, Linux GPU 0 queue (OLMoE): learn residency by search, extreme residency + adapter, small adapter, best combination.
cd ~/expert-dispatch; L=results/night3_linux.log; export CUDA_VISIBLE_DEVICES=0 LR=3e-6
run() { echo "# $(date) $*" >> $L; nice -n 10 timeout 3600 pixi run python "$@" 2>&1 | grep -E "^\{|^round|^start|Error|error" >> $L; }
run scripts/swapsearch_olmoe.py 8 3 4
run scripts/swapsearch_olmoe.py 16 3 4
run scripts/adapter_olmoe.py 4 inf 32 240000 900
RES=gate run scripts/adapter_olmoe.py 8 inf 8 240000 900
RES=gate run scripts/adapter_olmoe.py 8 2 32 240000 900
RES=gate run scripts/adapter_olmoe.py 8 inf 32 480000 1800
echo "# $(date) done" >> $L; touch results/night3_linux.done
