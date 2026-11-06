#!/bin/bash
cd ~/expert-dispatch; L=results/night3_linux.log; export CUDA_VISIBLE_DEVICES=0 LR=3e-6 MODEL=ibm-granite/granite-3.1-3b-a800m-instruct
run() { echo "# $(date) $*" >> $L; nice -n 10 timeout 3600 pixi run python "$@" 2>&1 | grep -E "^\{|Error|error" >> $L; }
RES=search run scripts/adapter_olmoe.py 16 inf 32 240000 900
RES=search run scripts/adapter_olmoe.py 8 inf 32 480000 1800
RES=search run scripts/adapter_olmoe.py 4 inf 32 480000 1800
run scripts/eval_bundles.py
echo "# $(date) done7" >> $L; touch results/night3_linux7.done
