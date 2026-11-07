#!/bin/bash
# Night 5 queue 1: baseline, KL reference, traces + learned table, then the exact tricks one at a time (same harness)
cd ~/expert-dispatch; R=results/night5; mkdir -p $R
GG=$(ls ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/*00001-of-*.gguf)
S="pixi run -q python scripts/night5_server.py"
log() { echo "- $(date +%H:%M) $*" >> docs/journal_night5.md; }
guard() { a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); d=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ $d -lt 20 ] && { log "STOP: disk ${d}G"; exit 1; }; }
run() { guard; [ -f $R/$1.json ] && return; log "run $1 ($ARGS | $EDXENV)"; $S $1 $R/$1.json > $R/$1.out 2>&1 || log "FAILED $1: $(tail -2 $R/$1.out | tr '\n' ' ')"; }

ARGS="" EDXENV="" run base
if [ ! -f $R/kl_base.bin ]; then
  head -c 60000 corpus/coding_heldout.txt > /tmp/n5_kl.txt
  CUDA_VISIBLE_DEVICES=0,1 nice -n 10 llama.cpp/build/bin/llama-perplexity -m $GG -f /tmp/n5_kl.txt -c 512 --chunks 8 --kl-divergence-base $R/kl_base.bin > $R/kl_base.log 2>&1 && log "KL base saved" || log "KL base FAILED"
fi
scripts/n5_trace.sh > $R/trace.log 2>&1; log "traces: $(grep -h coding $R/trace.log | tr '\n' ' ')"
[ -f $R/pred_q38.bin ] || pixi run -q python scripts/n5_predictor.py Qwen3.8-Flash-Next coding_train coding_heldout $R/pred_q38.bin $R/predictor_offline.json > $R/predictor.log 2>&1
ARGS="" EDXENV="EDX_PF_MODE=stats" run pf_stats
ARGS="" EDXENV="EDX_PF_MODE=same" run pf_same
ARGS="" EDXENV="EDX_PF_MODE=prev" run pf_prev
ARGS="" EDXENV="EDX_PF_MODE=markov EDX_PF_PRED=$R/pred_q38.bin" run pf_markov
ARGS="" EDXENV="EDX_PF_MODE=both EDX_PF_PRED=$R/pred_q38.bin" run pf_both
ARGS="-md $HOME/models/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2" EDXENV="" run mtp
ARGS="-lzm off" EDXENV="" run engram_in_ram
log "queue1 done"; touch $R/queue1.done
