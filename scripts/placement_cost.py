"""Same VRAM, three ways to spend it, scored in estimated ms/token on held-out traces.

  per_layer      llama.cpp today (--n-cpu-moe N): the last L-N layers' experts fully in VRAM,
                 the rest computed on the CPU from RAM.
  profile        the hottest (layer, expert) pairs from the train split (global knapsack,
                 equal expert sizes); misses computed on the CPU from RAM, never uploaded.
  profile_even   same, but the same number of slots in every layer.
  lru_upload     llama.cpp --moe-cache-mib style: per-layer LRU, every miss uploaded over PCIe.

Cost model (constants measured on this box with Qwen3.5-122B Q4_K_M, see NIGHT1_LOG):
  ms/token = T_GPU + C_CPU * (expert reads served from RAM) + C_UP * (uploads)
usage: placement_cost.py <model_dir> <tag> [T_GPU C_CPU C_UP]
"""
import json, sys
from pathlib import Path
import numpy as np
from numba import njit

W = Path.home() / "expert-dispatch"
name, tag = sys.argv[1], sys.argv[2]
T_GPU, C_CPU, C_UP = (float(x) for x in sys.argv[3:6]) if len(sys.argv) > 5 else (8.0, 0.24, 0.8)
tr = np.load(W / f"traces/{name}/{tag}_train.npz")["topk"].astype(np.int64)
ho = np.load(W / f"traces/{name}/{tag}_heldout.npz")["topk"].astype(np.int64)
T, L, K = ho.shape
E = int(max(tr.max(), ho.max())) + 1
cnt = np.stack([np.bincount(tr[:, l].ravel(), minlength=E) for l in range(L)]).astype(float)
ho_cnt = np.stack([np.bincount(ho[:, l].ravel(), minlength=E) for l in range(L)]).astype(float)
reads = T * L * K


@njit
def lru(topk, slots):  # slots[l] experts per layer; returns uploads
    T, L, K = topk.shape
    last = np.full((L, 1024), -1, np.int64)
    res = np.zeros((L, 1024), np.bool_)
    n = np.zeros(L, np.int64)
    up = 0
    for t in range(T):
        for l in range(L):
            S = slots[l]
            for j in range(K):
                e = topk[t, l, j]
                if S == 0:
                    up += 1  # nothing cacheable: still has to be uploaded in this mode
                    continue
                if not res[l, e]:
                    up += 1
                    if n[l] < S:
                        n[l] += 1
                    else:
                        v = -1; best = 1 << 62
                        for c in range(1024):
                            if res[l, c] and last[l, c] < best and last[l, c] != t:
                                best = last[l, c]; v = c
                        res[l, v] = False
                    res[l, e] = True
                last[l, e] = t
    return up


def ms(cpu_reads, uploads):
    return T_GPU + (C_CPU * cpu_reads + C_UP * uploads) / T


out = {"model": name, "tag": tag, "tokens": T, "consts": [T_GPU, C_CPU, C_UP], "rows": []}
for n_gpu_layers in (6, 10, 14, 18, 24):
    budget = n_gpu_layers * E  # same VRAM as n_gpu_layers full layers
    pl = ho_cnt[L - n_gpu_layers:].sum()
    order = np.argsort(-cnt.ravel())[:budget]
    mask = np.zeros(L * E, bool); mask[order] = True
    pg = ho_cnt.ravel()[mask].sum()
    per = budget // L
    pe = sum(ho_cnt[l][np.argsort(-cnt[l])[:per]].sum() for l in range(L))
    up = lru(ho, np.full(L, per, np.int64))
    row = {"vram_layers_equiv": n_gpu_layers, "vram_share": n_gpu_layers / L,
           "per_layer": {"hit": pl / reads, "ms": ms(reads - pl, 0)},
           "profile": {"hit": pg / reads, "ms": ms(reads - pg, 0)},
           "profile_even": {"hit": pe / reads, "ms": ms(reads - pe, 0)},
           "lru_upload": {"hit": 1 - up / reads, "ms": ms(0, up)}}
    out["rows"].append(row)
    print(n_gpu_layers, " ".join(f"{k}: {v['hit']*100:.1f}% {1000/v['ms']:.1f}tok/s" for k, v in row.items() if isinstance(v, dict)), flush=True)
(W / f"results/{name}").mkdir(parents=True, exist_ok=True)
json.dump(out, open(W / f"results/{name}/placement_cost_{tag}.json", "w"), indent=1)
