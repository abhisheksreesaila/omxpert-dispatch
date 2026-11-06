#!/bin/bash
# when the Mac is back: make sure exactly one clean run 2 is going (kill duplicates/contaminated, restart once)
K="-i $HOME/.ssh/edx_mac -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"; H=${MAC_HOST:?set MAC_HOST=user@host}
until ssh $K $H true 2>/dev/null; do sleep 60; done
ssh $K $H 'cd ~/expert-dispatch; n=$(ps -axo command | grep -c "^/bin/bash ./[m]ac_bench2.sh")
  if [ -f results/mac/bench2.clean ] && [ "$n" -ge 1 ]; then echo "clean run already going"; exit 0; fi
  for p in $(ps -axo pid,command | awk "/[m]ac_bench2.sh|[l]lama-completion|[l]lama-perplexity/ {print \$1}"); do kill $p; done; sleep 3
  [ -f results/mac/bench2.log ] && mv results/mac/bench2.log results/mac/bench2_contaminated_$(date +%H%M).log; rm -f results/mac/fail2_* results/mac/ppl2.log results/mac/bench2.done
  touch results/mac/bench2.started results/mac/bench2.clean
  S=mac_bench2; nohup caffeinate -dims ./$S.sh > results/mac_bench2.out 2>&1 < /dev/null & sleep 2
  echo "restarted clean: $(ps -axo command | grep -c "^/bin/bash ./[m]ac_bench2.sh") bench process(es)"'
