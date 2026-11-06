#!/bin/bash
# Self-speculative decoding vs plain pinned, 35B coding prompts, 30% VRAM for experts.
cd ~/expert-dispatch; M=~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf; R=results/Qwen3.6-35B-A3B; OUT=$R/bench_spec.log; : > $OUT
for p in results/qwen122b/prompt_coding_*.txt; do
  ms=$(EDX_PIN=$R/pin_coding.bin CUDA_VISIBLE_DEVICES=0 timeout 600 nice -n 5 llama.cpp/build/bin/llama-completion -m $M -f $p -n 120 -no-cnv --temp 0 -ngl 99 --n-cpu-moe 40 --moe-cache-mib 5600 -c 4096 --no-display-prompt -v 2>&1 >/dev/null | grep "  eval time" | sed -E 's/.*,\s+([0-9.]+) tokens per second.*/\1/')
  echo "plain_pinned $(basename $p .txt) tps=$ms" >> $OUT
  for k in 2 3 4 6; do
    EDX_PIN=$R/pin_coding.bin EDX_DRAFT=1 CUDA_VISIBLE_DEVICES=0 timeout 600 nice -n 5 llama.cpp/build/bin/llama-speculative-simple -m $M -md $M --spec-type draft-simple -f $p -n 120 --temp 0 -ngl 99 -ngld 99 --n-cpu-moe 40 -ncmoed 40 --moe-cache-mib 5600 --spec-draft-n-max $k -c 4096 -v > /tmp/edx_spec_run.txt 2>/tmp/edx_spec_run.log
    r=$(python3 - <<'PY'
import re
rounds=[];pending=None
for l in open('/tmp/edx_spec_run.log'):
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
touch bench_spec.done
