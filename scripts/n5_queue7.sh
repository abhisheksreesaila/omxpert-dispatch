#!/bin/bash
# Night 5 queue 7 (after queue 9): hand-off test, contiguous layout fixed. With -ts 53,47 GPU0 holds layers 0-24 and GPU1 25-47;
# putting CPU experts at the start of each GPU's range (0-11, 25-37) keeps ~4 GPU<->CPU switches per token and balances VRAM.
cd ~/expert-dispatch; R=results/night5
until [ -f $R/queue9.done ]; do sleep 30; done
S="pixi run -q python scripts/night5_server.py"
log() { echo "- $(date +%H:%M) $*" >> docs/journal_night5.md; }
run() { [ -f $R/$1.json ] && return; log "run $1 ($ARGS)"; $S $1 $R/$1.json > $R/$1.out 2>&1 || log "FAILED $1: $(tail -2 $R/$1.out | tr '\n' ' ')"; }
H="-fit off -ngl 99 -lzm off -ts 53,47"
C="$(seq -s'|' 0 11)|$(seq -s'|' 25 37)"
ARGS="$H -ot blk\.($C)\.ffn_.*_exps\.=CPU" EDXENV="" run handoff_contig2_1
ARGS="$H -ot blk\.(0|$(seq -s'|' 1 2 47))\.ffn_.*_exps\.=CPU" EDXENV="" run handoff_alt_3
ARGS="$H -ot blk\.($C)\.ffn_.*_exps\.=CPU" EDXENV="" run handoff_contig2_2
log "queue7 done"; touch $R/queue7.done
