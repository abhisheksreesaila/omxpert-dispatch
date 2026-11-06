#!/bin/bash
# One job at a time. Refuses to start if the machine is busy.
cd ~/expert-dispatch
log(){ echo "- $(date -Is) MAXFULL: $*" | tee -a NIGHT1_LOG.md; }
avail=$(free -g | awk '/Mem:/{print $7}')
g0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); g1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
if [ "$avail" -lt 24 ] || [ "$g0" -gt 2000 ] || [ "$g1" -gt 2500 ]; then log "machine busy (RAM avail ${avail}G, GPU ${g0}/${g1} MiB) - not starting"; exit 1; fi
rm -rf traces/max/*/ ; : > traces/max/IMPORTED.log
scripts/start_serve.sh || exit 1
pixi run python scripts/max_drive.py prefill coding train 4000
cat traces/max/IMPORTED.log
ls traces/max/coding_train/*.npz >/dev/null 2>&1 || { log "no trace shards; hooks did not fire"; scripts/stop_serve.sh; exit 1; }
log "hooks fire in MAX worker"; rm -rf traces/max/coding_train
for tag in coding design_chat tool_calls docs math; do
  pixi run python scripts/max_drive.py prefill $tag train 200000
  pixi run python scripts/max_drive.py prefill $tag heldout 60000
done
pixi run python scripts/max_drive.py decode 30
pixi run python scripts/max_drive.py gen results/max_gptoss_gen_base.json 30
scripts/stop_serve.sh
pixi run python scripts/max_collect.py && pixi run python scripts/analyze.py max_gptoss && log "traces+analysis done"
for cfg in "8 1.0" "16 1.0" "8 0.5"; do
  set -- $cfg; b=$(pixi run python scripts/max_bias_eval.py make_bias $1 $2)
  scripts/start_serve.sh $b
  pixi run python scripts/max_drive.py gen results/max_gptoss_gen_S$1_l$2.json 30
  scripts/stop_serve.sh
  r=$(pixi run python scripts/max_bias_eval.py compare results/max_gptoss_gen_base.json results/max_gptoss_gen_S$1_l$2.json)
  log "steer S=$1/32 lam=$2: $r"
done
log "done"
