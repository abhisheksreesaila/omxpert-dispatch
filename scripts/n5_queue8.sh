#!/bin/bash
# Night 5 queue 8 (after queue 7): the learned-residency test without the table confound. All runs -lzm off (table populated at
# load), interleaved: llama.cpp fit vs coding profile pinned per expert vs the profile + steering lambda 0.5.
cd ~/expert-dispatch; R=results/night5
until [ -f $R/queue7.done ]; do sleep 30; done
S="pixi run -q python scripts/night5_server.py"
log() { echo "- $(date +%H:%M) $*" >> docs/journal_night5.md; }
run() { [ -f $R/$1.json ] && return; log "run $1 ($ARGS | $EDXENV)"; $S $1 $R/$1.json > $R/$1.out 2>&1 || log "FAILED $1: $(tail -2 $R/$1.out | tr '\n' ' ')"; }
B=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{s+=$1-5200} END {print s}')
P="-fit off -ngl 99 --n-cpu-moe 48 --moe-cache-mib $B -lzm off"
PE="EDX_PIN=$R/pins/q38_coding.bin EDX_PAGEOUT=1"
ARGS="-lzm off" EDXENV="" run fit_lzm_1
ARGS="$P" EDXENV="$PE" run pin_lzm_1
ARGS="$P" EDXENV="$PE EDX_STEER=0.5" run pin_lzm_steer05_1
ARGS="-lzm off" EDXENV="" run fit_lzm_2
ARGS="$P" EDXENV="$PE" run pin_lzm_2
ARGS="$P" EDXENV="$PE EDX_STEER=0.5" run pin_lzm_steer05_2
log "queue8 done"; touch $R/queue8.done
