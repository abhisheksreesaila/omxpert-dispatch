#!/bin/bash
cd ~/expert-dispatch
for t in coding design_chat; do CUDA_VISIBLE_DEVICES=1 nice -n 10 pixi run python scripts/spec_accept.py $t 30000 > spec_$t.log 2>&1; done
touch spec.done
