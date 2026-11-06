#!/bin/bash
# Qwen3-Coder-30B-A3B (pure attention MoE): trace -> profile -> placement bench -> self-speculative bench
cd ~/expert-dispatch
while [ ! -f queue_quiet.done ] || ! grep -q "Qwen3-Coder-30B download exit" NIGHT1_LOG.md; do sleep 30; done
GG=$(ls ~/models/Qwen3-Coder-30B-A3B-GGUF/*Q4_K_M*.gguf | head -1); MD=Qwen3-Coder-30B-A3B; R=results/$MD; mkdir -p traces/$MD $R
scripts/wait_quiet.sh
echo "- $(date -Is) CODER: tracing" >> NIGHT1_LOG.md
for f in corpus/imx/coding_train.txt corpus/imx/coding_heldout.txt corpus/imx/design_chat_train.txt corpus/imx/design_chat_heldout.txt; do
  n=$(basename $f .txt); [ -f traces/$MD/$n.npz ] && continue
  CUDA_VISIBLE_DEVICES=0 nice -n 5 llama.cpp/build/bin/llama-edx-trace -m $GG -f $f -o traces/$MD/$n.bin -ngl 99 -c 2048 -b 512 -ub 512 2>&1 | grep -E "EDX_TRACE_DONE|failed"
  pixi run python -c "
import numpy as np, re
L=int(open('/dev/stdin').read().split('L=')[1].split()[0]) if False else None
a=np.fromfile('traces/$MD/$n.bin',dtype=np.int32); K=8
for L in (48,40,36,32,24,94,64,60):
    if a.size % (L*K)==0 and a.reshape(-1,L,K).max()<512: break
a=a.reshape(-1,L,K); np.savez('traces/$MD/$n.npz', topk=a.astype(np.int16)); print('$n', a.shape)" && rm traces/$MD/$n.bin
done
EDX_E=128 pixi run python scripts/make_pin.py $MD $R/pin_coding.bin coding_train
EDX_E=128 pixi run python scripts/make_pin.py $MD $R/pin_chat.bin design_chat_train
HOSTMIB=$(CUDA_VISIBLE_DEVICES=0 timeout 300 llama.cpp/build/bin/llama-completion -m $GG -p hi -n 2 -no-cnv -ngl 99 --n-cpu-moe 99 --moe-cache-mib 500 -v 2>&1 | grep -m1 "MoE cache size" | sed -E 's/.*for ([0-9.]+) MiB.*/\1/')
NL=$(python3 -c "import numpy as np;print(np.load('traces/$MD/coding_train.npz')['topk'].shape[1])")
PERL=$(python3 -c "print(int(float('$HOSTMIB')/$NL))"); echo "- $(date -Is) CODER: host experts ${HOSTMIB} MiB, ${NL} layers, ${PERL} MiB/layer" >> NIGHT1_LOG.md
OUT=$R/bench.log; : > $OUT
B=llama.cpp/build/bin/llama-completion
one(){ env $2 CUDA_VISIBLE_DEVICES=0 timeout 600 nice -n 5 $B -m $GG -f $3 -n 120 -no-cnv --temp 0 -ngl 99 -c 4096 --no-display-prompt -v $4 2>&1 >/dev/null | grep -E "  eval time" | sed -E 's/.*\(\s*([0-9.]+) ms per token.*/ms=\1/'; }
scripts/wait_quiet.sh
for share in 15 30; do
  X=$(( PERL * NL * share / 100 )); LG=$(( NL * share / 100 ))
  for p in results/qwen122b/prompt_coding_*.txt results/qwen122b/prompt_design_chat_*.txt; do
    pin=$R/pin_coding.bin; case $p in *chat*) pin=$R/pin_chat.bin;; esac
    echo "vram${share} layers $(basename $p .txt) $(one layers X=1 $p "--n-cpu-moe $(( NL - LG ))")" >> $OUT
    echo "vram${share} slru $(basename $p .txt) $(one slru X=1 $p "--n-cpu-moe $NL --moe-cache-mib $X")" >> $OUT
    echo "vram${share} pinned_own $(basename $p .txt) $(one pin EDX_PIN=$pin $p "--n-cpu-moe $NL --moe-cache-mib $X")" >> $OUT
  done
done
echo "- $(date -Is) CODER: placement bench done" >> NIGHT1_LOG.md
# self-speculative on coding prompts, 30% VRAM
X=$(( PERL * NL * 30 / 100 ))
for p in results/qwen122b/prompt_coding_*.txt; do
  for k in 2 3 4 6; do
    EDX_PIN=$R/pin_coding.bin EDX_DRAFT=1 CUDA_VISIBLE_DEVICES=0 timeout 400 nice -n 5 llama.cpp/build/bin/llama-speculative-simple -m $GG -md $GG --spec-type draft-simple -f $p -n 120 --temp 0 -ngl 99 -ngld 99 --n-cpu-moe $NL -ncmoed $NL --moe-cache-mib $X --spec-draft-n-max $k -c 4096 -v > /tmp/edx_cspec.txt 2>/tmp/edx_cspec.log
    r=$(python3 - <<'PY'
import re
rounds=[];pending=None;tps='?'
for l in open('/tmp/edx_cspec.log'):
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
echo "- $(date -Is) CODER: all done" >> NIGHT1_LOG.md; touch coder.done
