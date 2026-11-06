#!/bin/bash
# Waits until GPU0 is free (other session's tests), then: MAX serve with hooks -> prefill traces
# for every tag -> decode sample -> collect -> analyze. Logs to max_run.log.
cd ~/expert-dispatch
log(){ echo "- $(date -Is) MAX: $*" | tee -a NIGHT1_LOG.md; }
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 2000 ] \
   && ! pgrep -f "torch_trace|torch_bias|llama-imatrix|run_imatrix|run_bias" >/dev/null; do sleep 30; done
log "GPU0 free, starting serve"
rm -rf traces/max/*/ ; : > traces/max/IMPORTED.log
scripts/start_serve.sh
grep -q "first execute\|graph.output" traces/max/IMPORTED.log || log "WARNING hooks not seen in worker yet"
pixi run python scripts/max_drive.py prefill coding train 4000 && cat traces/max/IMPORTED.log
ls traces/max/coding_train >/dev/null 2>&1 || { log "FAILED: no trace shards written; see max/serve.log"; scripts/stop_serve.sh; exit 1; }
for tag in coding design_chat tool_calls docs math; do
  [ "$tag" = coding ] || pixi run python scripts/max_drive.py prefill $tag train 300000
  pixi run python scripts/max_drive.py prefill $tag heldout 100000
done
pixi run python scripts/max_drive.py decode 30
log "traces done"
rm -rf traces/max/coding_train && pixi run python scripts/max_drive.py prefill coding train 300000
pixi run python scripts/max_collect.py
pixi run python scripts/analyze.py max_gptoss && log "analysis done"
scripts/stop_serve.sh
