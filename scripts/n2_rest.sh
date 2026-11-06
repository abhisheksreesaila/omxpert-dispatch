#!/bin/bash
# Night 2: Phase 3 validation -> Phase 4 sim -> real runs of the best plans.
cd ~/expert-dispatch; while [ ! -f n2_phase2.done ]; do sleep 30; done
echo "- $(date -Is) PHASE3 validation" >> NIGHT2_LOG.md
(cd scripts && nice -n 15 pixi run python sim_critical_path.py validate) >> NIGHT2_LOG.md 2>&1
echo "- $(date -Is) PHASE4 sim" >> NIGHT2_LOG.md
(cd scripts && nice -n 15 pixi run python n2_phase4_sim.py) > results/night2/phase4_sim.txt 2>&1; tail -6 results/night2/phase4_sim.txt | sed 's/^/  /' >> NIGHT2_LOG.md
scripts/wait_quiet.sh; echo "- $(date -Is) PHASE4 real runs" >> NIGHT2_LOG.md
M=~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf; R=results/Qwen3.6-35B-A3B; O=results/night2; : > $O/phase4_real.log
run(){ # name, env, args...
  name=$1; env1=$2; shift 2
  for p in $O/phase2/prompts/p0[0-9].txt; do
    ms=$(env $env1 CUDA_VISIBLE_DEVICES=0 timeout 400 nice -n 5 llama.cpp/build/bin/llama-completion -m $M -f $p -n 256 -no-cnv --temp 0 -ngl 99 -c 4096 --no-display-prompt -v "$@" 2>&1 >/dev/null | grep "  eval time" | sed -E 's/.*\(\s*([0-9.]+) ms per token.*/\1/')
    echo "$name $(basename $p .txt) ms=$ms" >> $O/phase4_real.log
  done
}
for frac in 15 60; do
  X=$(( 466 * 40 * frac / 100 )); NW=$(( 40 * frac / 100 ))
  run "freq_pins_$frac" "EDX_PIN=$R/pin_coding.bin" --n-cpu-moe 40 --moe-cache-mib $X
  run "whole_layers_$frac" "X=1" --n-cpu-moe $(( 40 - NW ))
  run "no_gpu_experts_$frac" "X=1" --n-cpu-moe 40
  read pinf cmib ot <<< "$(pixi run python scripts/n2_plan_args.py $O/plans/plan_always_$frac.json)"
  eval "run dp_hybrid_$frac \"EDX_PIN=$pinf\" $ot --moe-cache-mib $cmib"
done
echo "- $(date -Is) PHASE4 real runs done" >> NIGHT2_LOG.md; touch n2_rest.done
