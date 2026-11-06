#!/bin/bash
# wait for queue_all, then retry MAX with more memory headroom
cd ~/expert-dispatch
while [ ! -f queue.done ]; do sleep 30; done
echo "- $(date -Iseconds) MAX retry: util 0.92, act 4 GiB" >> NIGHT1_LOG.md
EDX_UTIL=0.92 EDX_ACT_GIB=4 scripts/max_full.sh > max_retry.log 2>&1
echo "- $(date -Iseconds) MAX retry exit $?" >> NIGHT1_LOG.md
touch max_retry.done
