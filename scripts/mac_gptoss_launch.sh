#!/bin/bash
# wait for the gpt-oss trace on the Mac, build its coding profile here, push it, start mac_gptoss.sh
K="-i $HOME/.ssh/edx_mac -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"; H=${MAC_HOST:?set MAC_HOST=user@host}
cd ~/expert-dispatch; mkdir -p results/mac results/gptoss
until ssh $K $H 'test -f ~/expert-dispatch/results/mac/chain3.done' 2>/dev/null; do sleep 120; done
scp -q $K $H:expert-dispatch/results/gptoss_trace.bin $H:expert-dispatch/results/mac/batch.log $H:expert-dispatch/results/mac/ppl_skip.log results/gptoss/ || exit 1
pixi run python - <<'PY' || exit 1
import numpy as np
from pathlib import Path
W = Path.home()/"expert-dispatch/results/gptoss"; L, K, E = 36, 4, 128
t = np.fromfile(W/"gptoss_trace.bin", np.int32); T = t.size//(L*K); t = t[:T*L*K].reshape(T, L, K)
score = np.stack([np.bincount(t[:, l].ravel(), minlength=E) for l in range(L)]).astype(np.float32)/T
with open(W/"pin_gptoss_coding.bin", "wb") as f: np.array([L, E], np.int32).tofile(f); score.tofile(f)
cov = [np.sort(score[l])[::-1][:32].sum()/K for l in range(L)]
print(f"gpt-oss trace: {T} tokens; top-32 of 128 experts cover {100*np.mean(cov):.1f}% of routings (mean over layers)")
PY
scp -q $K results/gptoss/pin_gptoss_coding.bin scripts/mac_gptoss.sh $H:expert-dispatch/ && ssh $K $H 'cd ~/expert-dispatch; mv pin_gptoss_coding.bin results/; chmod +x mac_gptoss.sh; S=mac_gptoss; nohup caffeinate -dims ./$S.sh > results/mac_gptoss.out 2>&1 < /dev/null & echo gptoss_started'
