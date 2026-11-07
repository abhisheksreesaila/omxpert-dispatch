#!/bin/bash
# Night 5 queue 2 (after queue 1): staged prefill, then the learned tricks: per-expert profile residency, steering; KL for each
cd ~/expert-dispatch; R=results/night5
until [ -f $R/queue1.done ]; do sleep 30; done
GG=$(ls ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/*00001-of-*.gguf)
S="pixi run -q python scripts/night5_server.py"
log() { echo "- $(date +%H:%M) $*" >> docs/journal_night5.md; }
run() { [ -f $R/$1.json ] && return; log "run $1 ($ARGS | $EDXENV)"; $S $1 $R/$1.json > $R/$1.out 2>&1 || log "FAILED $1: $(tail -2 $R/$1.out | tr '\n' ' ')"; }
kl() {  # kl <name> <args> <env...>: KL to the baseline logits on the same text
  [ -f $R/kl_$1.log ] && return; n=$1; a=$2; shift 2
  env CUDA_VISIBLE_DEVICES=0,1 "$@" nice -n 10 llama.cpp/build/bin/llama-perplexity -m $GG -f /tmp/n5_kl.txt -c 512 --chunks 8 --kl-divergence-base $R/kl_base.bin --kl-divergence $a > $R/kl_$n.log 2>&1
  log "KL $n: $(grep -E 'Mean +KLD|Same top' $R/kl_$n.log | tr -s ' ' | tr '\n' ' ')"
}
head -c 60000 corpus/coding_heldout.txt > /tmp/n5_kl.txt
# staged prefill: stream each CPU layer's experts once per 2048-token batch instead of once per 512
ARGS="-ub 2048 -b 2048" EDXENV="" run staged_prefill
kl staged_prefill "-ub 2048 -b 2048"
# learned residency: coding profile pinned per expert across both GPUs, the rest computed from RAM
[ -f $R/pins/q38_coding.bin ] || { mkdir -p $R/pins; EDX_E=512 pixi run -q python scripts/make_pin.py Qwen3.8-Flash-Next $R/pins/q38_coding.bin coding_train; }
B=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{s+=$1-5200} END {print s}')
P="-fit off -ngl 99 --n-cpu-moe 48 --moe-cache-mib $B"
ARGS="$P" EDXENV="EDX_PIN=$R/pins/q38_coding.bin EDX_PAGEOUT=1" run pin
ARGS="$P -ub 2048 -b 2048" EDXENV="EDX_PIN=$R/pins/q38_coding.bin EDX_PAGEOUT=1" run pin_staged
ARGS="$P" EDXENV="EDX_PIN=$R/pins/q38_coding.bin EDX_PAGEOUT=1 EDX_STEER=0.5" run pin_steer05
ARGS="$P" EDXENV="EDX_PIN=$R/pins/q38_coding.bin EDX_PAGEOUT=1 EDX_STEER=1" run pin_steer1
# KL: steering only acts on <= 8-token batches, so score it with 8-token batches (decode-like)
kl pin_ub8 "$P -ub 8 -b 8" EDX_PIN=$R/pins/q38_coding.bin
kl pin_steer05 "$P -ub 8 -b 8" EDX_PIN=$R/pins/q38_coding.bin EDX_STEER=0.5
kl pin_steer1 "$P -ub 8 -b 8" EDX_PIN=$R/pins/q38_coding.bin EDX_STEER=1
log "queue2 done"; touch $R/queue2.done
