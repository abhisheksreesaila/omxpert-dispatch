#!/bin/bash
# Night 5 queue 9 (after queue 6): the MTP prompt fix. The self-contained draft head loaded after the fit had filled both GPUs, so
# it ran on the CPU, including its 248k-word output layer for every prompt token. Use the shared head (borrows the main model's
# embeddings/output, 2.8 GB), leave it room on GPU0 (-fitt 3600,1024) and put it there (-devd CUDA0 -ngld all).
cd ~/expert-dispatch; R=results/night5
until [ -f $R/queue6.done ]; do sleep 30; done
S="pixi run -q python scripts/night5_server.py"
log() { echo "- $(date +%H:%M) $*" >> docs/journal_night5.md; }
run() { [ -f $R/$1.json ] && return; log "run $1 ($ARGS)"; $S $1 $R/$1.json > $R/$1.out 2>&1 || log "FAILED $1: $(tail -2 $R/$1.out | tr '\n' ' ')"; }
SH=$HOME/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
SA="-lzm off -fitt 3600,1024 -md $SH --spec-type draft-mtp --spec-draft-n-max 2 -devd CUDA0 -ngld all"
ARGS="$SA" EDXENV="" run stack_shared_1
ARGS="$SA" EDXENV="" run stack_shared_2
echo "$SA" > $R/stack_shared_args.txt
STACK_ARGS="$SA" STACK_ENV="" PORT=18090 omxpert/run.sh smoke > $R/smoke_shared.log 2>&1 && log "smoke_shared ok: $(tail -1 $R/smoke_shared.log)" || log "smoke_shared FAILED"
log "queue9 done"; touch $R/queue9.done
