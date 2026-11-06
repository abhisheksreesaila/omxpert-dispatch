"""Turn a Phase-4 plan into llama.cpp settings: a pin file that ranks the chosen experts first, the -ot rules
for whole layers, and the cache size. prints: <pinfile> <cache_mib> <ot_args...>"""
import json, sys
from pathlib import Path
import numpy as np
W = Path.home() / "expert-dispatch"; plan = json.load(open(sys.argv[1]))
L, E = 40, 256; EXPERT_MIB = 1.82
score = np.zeros((L, E), np.float32); whole = []; slots = 0
cnt = np.stack([np.bincount(np.load(W / "traces/Qwen3.6-35B-A3B/coding_train.npz")["topk"][:, l].astype(np.int64).ravel(), minlength=E) for l in range(L)]).astype(np.float32)
score[:] = cnt / cnt.max() * 1e-3            # fallback order for any spare slots
for row in plan["layers"]:
    if row["whole"]:
        whole.append(row["layer"])
    for e in row["experts"]:
        score[row["layer"], e] = 1.0 + cnt[row["layer"], e] / cnt.max(); slots += 1
out = W / f"results/night2/plans/{Path(sys.argv[1]).stem}.pin"
with open(out, "wb") as f:
    np.array([L, E], np.int32).tofile(f); score.tofile(f)
cache_mib = int(slots * EXPERT_MIB * 1.06) + 64   # small margin: groups split the budget by size
ot = []
if whole:
    ot += ["-ot", "blk\\.(" + "|".join(map(str, whole)) + ")\\.ffn_.*_exps\\.weight=CUDA0"]
ot += ["-ot", "ffn_.*_exps\\.weight=CPU"]
print(out, cache_mib, " ".join(f"'{a}'" for a in ot))
