#!/bin/bash
# Night 5 queue 4 (after queue 3): the KL noise floor. Stock llama.cpp with only the batch size changed is still "the same model",
# yet different batch shapes take different kernels; these KLs say how much drift is normal before calling a trick lossy.
cd ~/expert-dispatch; R=results/night5
until [ -f $R/queue3.done ]; do sleep 30; done
GG=$(ls ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/*00001-of-*.gguf)
log() { echo "- $(date +%H:%M) $*" >> docs/journal_night5.md; }
S="pixi run -q python scripts/night5_server.py"
run() { [ -f $R/$1.json ] && return; log "run $1 ($ARGS | $EDXENV)"; $S $1 $R/$1.json > $R/$1.out 2>&1 || log "FAILED $1: $(tail -2 $R/$1.out | tr '\n' ' ')"; }
# hand-off test: the same 25 layers' experts on the CPU, contiguous (~2 GPU<->CPU hand-offs per token) vs alternating (~50)
ODD=$(seq -s'|' 1 2 47); FIRST=$(seq -s'|' 0 24)
H="-fit off -ngl 99 -lzm off -ts 53,47"   # GPU1 also drives the display (~2.4 GB)
ARGS="$H -ot blk\.(0|$ODD)\.ffn_.*_exps\.=CPU" EDXENV="" run handoff_alt_1
ARGS="$H -ot blk\.($FIRST)\.ffn_.*_exps\.=CPU" EDXENV="" run handoff_contig_1
ARGS="$H -ot blk\.(0|$ODD)\.ffn_.*_exps\.=CPU" EDXENV="" run handoff_alt_2
ARGS="$H -ot blk\.($FIRST)\.ffn_.*_exps\.=CPU" EDXENV="" run handoff_contig_2
head -c 60000 corpus/coding_heldout.txt > /tmp/n5_kl.txt
kl() { [ -f $R/kl_$1.log ] && return; n=$1; a=$2; shift 2
  env CUDA_VISIBLE_DEVICES=0,1 "$@" nice -n 10 llama.cpp/build/bin/llama-perplexity -m $GG -f /tmp/n5_kl.txt -c 512 --chunks 8 --kl-divergence-base $R/kl_base.bin --kl-divergence $a > $R/kl_$n.log 2>&1
  log "KL $n: $(grep -E 'Mean +KLD|Same top' $R/kl_$n.log | tr -s ' ' | tr '\n' ' ')"; }
kl repeat_base "-ub 512 -b 512"
kl floor_ub256 "-ub 256 -b 256"
kl lzm_off "-lzm off"
kl floor_ub8 "-ub 8 -b 8"
log "queue4 done"; touch $R/queue4.done
