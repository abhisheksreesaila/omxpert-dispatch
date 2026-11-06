#!/bin/bash
# Detached pipeline: MAX (both GPUs) -> imatrix on GPU1 || bias test on GPU0
cd ~/expert-dispatch
echo "- $(date -Is) master start (T3 restart killed previous jobs)" >> NIGHT1_LOG.md
scripts/run_max.sh > max_run.log 2>&1
echo "- $(date -Is) MAX stage exit $?" >> NIGHT1_LOG.md
GPU=1 scripts/run_imatrix.sh > imatrix_run.log 2>&1 &
GPU=0 scripts/run_bias.sh > bias_run.log 2>&1 &
wait
echo "- $(date -Is) master done" >> NIGHT1_LOG.md
echo MASTER DONE > master.done
