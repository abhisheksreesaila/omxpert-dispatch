#!/bin/bash
cd ~/expert-dispatch; while [ ! -f bench_prompt.done ]; do sleep 30; done; scripts/wait_quiet.sh
GG=$(ls ~/models/Qwen3-Coder-30B-A3B-GGUF/*Q4_K_M*.gguf | head -1); R=results/Qwen3-Coder-30B-A3B; OUT=$R/bench_spec_devmap.log; : > $OUT
for p in results/qwen122b/prompt_coding_*.txt; do
  ms=$(EDX_PIN=$R/pin_coding.bin CUDA_VISIBLE_DEVICES=0 timeout 400 nice -n 5 llama.cpp/build/bin/llama-completion -m $GG -f $p -n 120 -no-cnv --temp 0 -ngl 99 --n-cpu-moe 48 --moe-cache-mib 5011 -c 4096 --no-display-prompt -v 2>&1 >/dev/null | grep "  eval time" | sed -E 's/.*,\s+([0-9.]+) tokens per second.*/\1/')
  echo "plain $(basename $p .txt) tps=$ms" >> $OUT
  dr=$(EDX_DRAFT_ALL=1 EDX_PIN=$R/pin_coding.bin CUDA_VISIBLE_DEVICES=0 timeout 400 nice -n 5 llama.cpp/build/bin/llama-completion -m $GG -f $p -n 120 -no-cnv --temp 0 -ngl 99 --n-cpu-moe 48 --moe-cache-mib 5011 -c 4096 --no-display-prompt -v 2>&1 >/dev/null | grep "  eval time" | sed -E 's/.*,\s+([0-9.]+) tokens per second.*/\1/')
  echo "draft_alone $(basename $p .txt) tps=$dr" >> $OUT
  for k in 2 3 4; do
    EDX_PIN=$R/pin_coding.bin EDX_DRAFT=1 CUDA_VISIBLE_DEVICES=0 timeout 400 nice -n 5 llama.cpp/build/bin/llama-speculative-simple -m $GG -md $GG --spec-type draft-simple -f $p -n 120 --temp 0 -ngl 99 -ngld 99 --n-cpu-moe 48 -ncmoed 48 --moe-cache-mib 5011 --spec-draft-n-max $k -c 4096 -v > /dev/null 2>/tmp/edx_sd.log
    r=$(python3 - <<'PY'
import re
rounds=[];pending=None;tps='?'
for l in open('/tmp/edx_sd.log'):
    m=re.search(r'gen = (\d+)',l)
    if 'called impl draft-simple' in l and m: pending=int(m.group(1)); continue
    m=re.search(r'partial acceptance: (\d+) < (\d+)',l) or re.search(r'accepted (\d+)/(\d+) draft tokens',l)
    if m and pending: rounds.append((int(m.group(1)),pending)); pending=None
    m=re.search(r'decoded\s+\d+ tokens in\s+[0-9.]+ seconds, speed:\s+([0-9.]+)',l)
    if m: tps=m.group(1)
a=sum(x for x,_ in rounds); g=sum(y for _,y in rounds)
print(f"tps={tps} accept={100*a/max(g,1):.1f}% tok_per_round={(a+len(rounds))/max(len(rounds),1):.2f}")
PY
)
    echo "spec_k$k $(basename $p .txt) $r" >> $OUT
  done
done
echo "- $(date -Is) spec devmap bench done" >> NIGHT1_LOG.md
nice -n 19 pixi run python scripts/admission_sim.py Qwen3.6-35B-A3B coding > results/Qwen3.6-35B-A3B/admission_coding.txt 2>&1
echo "- $(date -Is) admission sim done" >> NIGHT1_LOG.md; touch after_prompt.done
