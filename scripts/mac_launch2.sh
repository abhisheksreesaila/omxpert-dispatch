#!/bin/bash
# wait until the Mac answers, sync the patched source, rebuild, smoke-test steer-only, then run mac_bench2.sh under caffeinate
K="-i $HOME/.ssh/edx_mac -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10"; H=${MAC_HOST:?set MAC_HOST=user@host}
for try in 1 2 3 4 5 6; do
until ssh $K $H true 2>/dev/null; do sleep 60; done
cd ~/expert-dispatch
rsync -a --exclude build --exclude .git -e "ssh $K" llama.cpp $H:expert-dispatch/
scp -q $K scripts/mac_bench2.sh $H:expert-dispatch/
ssh $K $H 'export PATH=$HOME/.pixi/bin:$PATH; cd ~/expert-dispatch; caffeinate -i cmake --build llama.cpp/build --target llama-completion llama-perplexity -j 10 2>&1 | grep -E " error" ; chmod +x mac_bench2.sh
  EDX_PIN=results/pin_coding.bin EDX_STEER_ONLY=1 EDX_STEER=1 EDX_PIN_MINPL=16 llama.cpp/build/bin/llama-completion -m models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf -p "def fib(n):" -n 16 -no-cnv --temp 0 -c 1024 -nr -fit off -lm mmap -ngl 99 --n-cpu-moe 48 --moe-cache-mib 6144 -v 2>&1 | grep -E "MoE cache size|EDX pinned|eval time|error" > results/mac/smoke2.log
  test -f results/mac/bench2.started && { echo already started; exit 0; }; touch results/mac/bench2.started; test -f results/mac/bench2.done && { echo already done; exit 0; }
  nohup caffeinate -dims ./mac_bench2.sh > results/mac_bench2.out 2>&1 < /dev/null & echo launched'
[ $? -eq 0 ] && break; sleep 60; done
