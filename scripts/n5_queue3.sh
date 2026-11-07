#!/bin/bash
# Night 5 queue 3 (after queue 2): in-process A/B of the decode-time tricks, exclusive tiers A/B, a second baseline for drift,
# the combined exact stack, its KL, and the launcher smoke test
cd ~/expert-dispatch; R=results/night5
until [ -f $R/queue2.done ]; do sleep 30; done
GG=$(ls ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/*00001-of-*.gguf)
S="pixi run -q python scripts/night5_server.py"
log() { echo "- $(date +%H:%M) $*" >> docs/journal_night5.md; }
run() { [ -f $R/$1.json ] && return; log "run $1 ($ARGS | $EDXENV)"; $S $1 $R/$1.json > $R/$1.out 2>&1 || log "FAILED $1: $(tail -2 $R/$1.out | tr '\n' ' ')"; }
[ -f $R/pf_ab.json ] || { log "run pf_ab (in-process A/B, 7 arms x 7 rounds)"; pixi run -q python scripts/night5_ab.py $R/pf_ab.json > $R/pf_ab.out 2>&1 || log "FAILED pf_ab"; }
[ -f $R/mtp_ab.json ] || { log "run mtp_ab (in-process A/B, MTP on/off, 8 paired rounds)"; ARMS=mtp ROUNDS=8 pixi run -q python scripts/night5_ab.py $R/mtp_ab.json > $R/mtp_ab.out 2>&1 || log "FAILED mtp_ab"; }
B=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{s+=$1-5200} END {print s}')
ARGS="-fit off -ngl 99 --n-cpu-moe 48 --moe-cache-mib $B" EDXENV="EDX_PIN=$R/pins/q38_coding.bin" run pin_nopageout
ARGS="" EDXENV="" run base2
ARGS="-lzm off -ub 2048 -b 2048" EDXENV="" run staged_lzm
IFS='|' read -r SA _ _ SE _ _ WHY <<< "$(pixi run -q python scripts/n5_pick_stack.py)"
log "stack picked: [$WHY] args=[$SA] env=[$SE]"
echo "$SA" > $R/stack_args.txt; echo "$SE" > $R/stack_env.txt; echo "$WHY" > $R/stack_why.txt
ARGS="$SA" EDXENV="$SE" run stack
ARGS="$SA" EDXENV="$SE" run stack_repeat
head -c 60000 corpus/coding_heldout.txt > /tmp/n5_kl.txt
SA_KL=$(echo "$SA" | sed -E 's/-md [^ ]+//; s/--spec-type [^ ]+//; s/--spec-draft-n-max [^ ]+//')
env CUDA_VISIBLE_DEVICES=0,1 $SE nice -n 10 llama.cpp/build/bin/llama-perplexity -m $GG -f /tmp/n5_kl.txt -c 512 --chunks 8 --kl-divergence-base $R/kl_base.bin --kl-divergence $SA_KL > $R/kl_stack.log 2>&1
log "KL stack: $(grep -E 'Mean +KLD|Same top' $R/kl_stack.log | tr -s ' ' | tr '\n' ' ')"
STACK_ARGS="$SA" STACK_ENV="$SE" PORT=18090 omxpert/run.sh smoke > $R/smoke.log 2>&1 && log "smoke ok" || log "smoke FAILED"
log "queue3 done"; touch $R/queue3.done
