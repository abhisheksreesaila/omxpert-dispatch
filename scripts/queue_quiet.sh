#!/bin/bash
cd ~/expert-dispatch
scripts/wait_quiet.sh; echo "- $(date -Is) machine quiet; bench_spec" >> NIGHT1_LOG.md; scripts/bench_spec.sh
scripts/wait_quiet.sh; echo "- $(date -Is) machine quiet; bench35b" >> NIGHT1_LOG.md
sed -i 's/^cd ~\/expert-dispatch; while .*$/cd ~\/expert-dispatch/' scripts/bench35b.sh; scripts/bench35b.sh
touch queue_quiet.done
