#!/bin/bash
# Night 5 queue 6 (after queue 5): the final exact stack = n-gram table populated at load (-lzm off) + MTP drafting,
# run twice, its KL (MTP is exact by construction, so the KL run checks -lzm off), and the launcher smoke test with it
cd ~/expert-dispatch; R=results/night5
until [ -f $R/queue5.done ]; do sleep 30; done
GG=$(ls ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/*00001-of-*.gguf)
MTP=$HOME/models/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
S="pixi run -q python scripts/night5_server.py"
log() { echo "- $(date +%H:%M) $*" >> docs/journal_night5.md; }
run() { [ -f $R/$1.json ] && return; log "run $1 ($ARGS)"; $S $1 $R/$1.json > $R/$1.out 2>&1 || log "FAILED $1: $(tail -2 $R/$1.out | tr '\n' ' ')"; }
SA="-lzm off -md $MTP --spec-type draft-mtp --spec-draft-n-max 2"
ARGS="$SA" EDXENV="" run stack_final_1
ARGS="$SA" EDXENV="" run stack_final_2
echo "$SA" > $R/stack_final_args.txt
head -c 60000 corpus/coding_heldout.txt > /tmp/n5_kl.txt
[ -f $R/kl_stack_final.log ] || { CUDA_VISIBLE_DEVICES=0,1 nice -n 10 llama.cpp/build/bin/llama-perplexity -m $GG -f /tmp/n5_kl.txt -c 512 --chunks 8 --kl-divergence-base $R/kl_base.bin --kl-divergence -lzm off > $R/kl_stack_final.log 2>&1; log "KL stack_final: $(grep -E 'Mean +KLD|Same top' $R/kl_stack_final.log | tr -s ' ' | tr '\n' ' ')"; }
STACK_ARGS="$SA" STACK_ENV="" PORT=18090 omxpert/run.sh smoke > $R/smoke_final.log 2>&1 && log "smoke_final ok: $(tail -1 $R/smoke_final.log)" || log "smoke_final FAILED"
log "queue6 done"; touch $R/queue6.done
