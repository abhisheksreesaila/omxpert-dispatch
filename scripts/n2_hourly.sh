#!/bin/bash
# Owner's request: check every hour; start the night-2 pipeline only when the machine is quiet.
# Quiet = for 10 consecutive minutes: load1 < 3, RAM available >= 30 GB, GPU0 < 2000 MiB, GPU1 < 2500 MiB used.
cd ~/expert-dispatch
quiet10(){
  for i in $(seq 10); do
    l=$(cut -d' ' -f1 /proc/loadavg); a=$(free -g | awk '/Mem:/{print $7}')
    g0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); g1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    if ! awk "BEGIN{exit !($l < 3)}" || [ "$a" -lt 30 ] || [ "$g0" -gt 2000 ] || [ "$g1" -gt 2500 ]; then
      echo "- $(date -Is) hourly check: busy (load $l, RAM avail ${a}G, GPU ${g0}/${g1} MiB)" >> NIGHT2_LOG.md; return 1
    fi
    sleep 60
  done
  return 0
}
for h in $(seq 12); do
  if quiet10; then
    echo "- $(date -Is) hourly check: quiet for 10 min, starting pipeline" >> NIGHT2_LOG.md
    scripts/n2_resume.sh >> NIGHT2_LOG.md 2>&1; exit 0
  fi
  sleep 3000   # ~1 hour between checks (10 min window + 50 min wait)
done
echo "- $(date -Is) hourly check: never quiet in 12 h, not started" >> NIGHT2_LOG.md
