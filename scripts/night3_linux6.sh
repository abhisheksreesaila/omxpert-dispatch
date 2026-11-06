#!/bin/bash
cd ~/expert-dispatch; L=results/night3_linux.log; export CUDA_VISIBLE_DEVICES=0
until [ -f results/night3_linux5.done ]; do sleep 30; done
echo "# $(date) scripts/eval_bundles.py (OLMoE, fixed mask)" >> $L
MODEL=allenai/OLMoE-1B-7B-0125-Instruct nice -n 10 timeout 3600 pixi run python scripts/eval_bundles.py 2>&1 | grep -E "^\{|Error|error" >> $L
echo "# $(date) done6" >> $L; touch results/night3_linux6.done
