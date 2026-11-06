#!/bin/bash
cd ~/expert-dispatch
EDX_UTIL=0.95 EDX_ACT_GIB=4 EDX_BIT=1024 scripts/max_full.sh > max_go.log 2>&1
echo "- $(date -Iseconds) MAX full run exit $?" >> NIGHT1_LOG.md
touch max_go.done
