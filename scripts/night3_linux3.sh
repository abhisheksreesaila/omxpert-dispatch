#!/bin/bash
# Round 3: learn the 4-per-layer set, then the strongest bundles on learned sets.
cd ~/expert-dispatch; L=results/night3_linux.log; export CUDA_VISIBLE_DEVICES=0 LR=3e-6
run() { echo "# $(date) $*" >> $L; nice -n 10 timeout 5400 pixi run python "$@" 2>&1 | grep -E "^\{|^round|^start|Error|error" >> $L; }
python3 - <<'PY'
# swapsearch needs the gate scores for S=4: reuse the S=8 file (same gate scores, any S)
import shutil, pathlib; R = pathlib.Path.home()/"expert-dispatch/results/OLMoE-1B-7B-0125-Instruct"
shutil.copy(R/"residency_S8_t120000.pt", R/"residency_S4_t120000.pt")
PY
run scripts/swapsearch_olmoe.py 4 10 6
RES=search run scripts/adapter_olmoe.py 4 inf 32 480000 1800
RES=search run scripts/adapter_olmoe.py 8 inf 32 480000 1800
RES=search run scripts/adapter_olmoe.py 8 2 32 240000 900
echo "# $(date) done3" >> $L; touch results/night3_linux3.done
