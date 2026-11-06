#!/bin/bash
# Resume the night-2 pipeline from the start of the queue (run only when the machine is dedicated to this).
cd ~/expert-dispatch
rm -f steer.done n2_phase1.done n2_phase2.done n2_rest.done n2_tail.done n2_spec.done
setsid nohup bash -c 'scripts/wait_quiet.sh; scripts/bench_steer_chat.sh' >/dev/null 2>&1 </dev/null &
for s in n2_phase1 n2_phase2 n2_rest n2_tail n2_spec_experts; do setsid nohup scripts/$s.sh > $s.out 2>&1 </dev/null & done
echo "night-2 pipeline queued; progress in NIGHT2_LOG.md"
