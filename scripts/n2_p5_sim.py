"""Idea 17: pinned profile + a small adaptive pool with upload admission.

VRAM = S_total slots per layer. A share is pinned to the profile (never moves);
the rest is an LRU pool that admits an expert on a miss only if a rule says
its near-term reuse pays for the upload. Misses that are not admitted run on
the CPU from RAM. Scored in estimated ms/token with constants fit on the clean
35B runs (bench35.log): T0 + C_CPU * cpu_reads + C_UP * uploads.

Rules: never (= pinned only), always (= LRU pool), recent>=n (used >= n times
in the last H tokens), oracle>=n (will be used >= n times in the next H tokens,
the ceiling a learned predictor could reach).
usage: admission_sim.py <model_dir> <tag>
"""
import json, sys
from pathlib import Path
import numpy as np
from numba import njit

W = Path.home() / "expert-dispatch"
name, tag = sys.argv[1], sys.argv[2]
C = json.load(open(W / "results/night2/costs.json"))
T0, C_CPU, C_UP, SYNC = C["gpu_base_ms_k8"], C["c_cpu_ms_per_expert"], C["upload_ms_per_expert_pageable"], C["sync_ms_per_cpu_layer"]
tr = np.load(W / f"traces/{name}/{tag}_train.npz")["topk"].astype(np.int64)
ho = np.load(W / f"traces/{name}/{tag}_heldout.npz")["topk"].astype(np.int64)[:30000]
T, L, K = ho.shape
E = 256
cnt = np.stack([np.bincount(tr[:, l].ravel(), minlength=E) for l in range(L)])


@njit
def run(topk, pinned, S_pool, rule, n_thr, H):
    # rule: 0 never, 1 always, 2 recent, 3 oracle, 4 recent with a set-completion bonus (needs n_thr-1 if it is the only miss)
    T, L, K = topk.shape
    E = pinned.shape[1]
    inpool = np.zeros((L, E), np.bool_)
    last = np.full((L, E), -1, np.int64)
    npool = np.zeros(L, np.int64)
    hist = np.zeros((L, E, 64), np.int64)  # ring of last use times (H <= 64)
    hn = np.zeros((L, E), np.int64)
    # next-use lists for the oracle: count future uses within H via a sliding count
    fut = np.zeros((L, E), np.int64)
    if rule == 3:
        for t in range(min(H, T)):
            for l in range(L):
                for j in range(K):
                    fut[l, topk[t, l, j]] += 1
    cpu = 0; up = 0; hit = 0
    for t in range(T):
        if rule == 3:  # window [t, t+H): drop t-1, add t+H-1
            if t > 0:
                for l in range(L):
                    for j in range(K):
                        fut[l, topk[t - 1, l, j]] -= 1
            if t + H - 1 < T and t > 0:
                for l in range(L):
                    for j in range(K):
                        fut[l, topk[t + H - 1, l, j]] += 1
        for l in range(L):
            nm = 0
            for j in range(K):
                if not (pinned[l, topk[t, l, j]] or inpool[l, topk[t, l, j]]):
                    nm += 1
            for j in range(K):
                e = topk[t, l, j]
                if pinned[l, e] or inpool[l, e]:
                    hit += 1
                else:
                    admit = False
                    if S_pool > 0:
                        if rule == 1:
                            admit = True
                        elif rule == 2 or rule == 4:
                            c = 0
                            for q in range(min(hn[l, e], 64)):
                                if t - hist[l, e, q] <= H:
                                    c += 1
                            admit = c >= n_thr if rule == 2 else (c >= n_thr or (nm == 1 and c >= n_thr - 1))
                        elif rule == 3:
                            admit = fut[l, e] - 1 >= n_thr  # uses after this one
                    if admit:
                        up += 1
                        if npool[l] < S_pool:
                            npool[l] += 1
                        else:
                            v = -1; b = 1 << 62
                            for c2 in range(E):
                                if inpool[l, c2] and last[l, c2] < b and last[l, c2] != t:
                                    b = last[l, c2]; v = c2
                            inpool[l, v] = False
                        inpool[l, e] = True
                    else:
                        cpu += 1
                last[l, e] = t
                hist[l, e, hn[l, e] % 64] = t
                hn[l, e] += 1
    return hit, cpu, up


out = {"model": name, "tag": tag, "consts": [T0, C_CPU, C_UP], "rows": []}
for share in (0.15, 0.30):
    S = int(E * share)
    for pool_frac in (0.0, 0.1):
        Sp = S - int(S * pool_frac); Sa = S - Sp
        pinned = np.zeros((L, E), np.bool_)
        for l in range(L):
            pinned[l, np.argsort(-cnt[l])[:Sp]] = True
        rules = [("never", 0, 0, 1)] if Sa == 0 else \
            [("admit-on-miss", 1, 0, 1)] + [(f"reused>={n}/H32", 2, n, 32) for n in (1, 2, 3, 4, 6, 8)] + \
            [(f"reused>={n}+setbonus/H32", 4, n, 32) for n in (2, 3, 4, 6)] + [(f"oracle>={n}/H32", 3, n, 32) for n in (2, 3, 4)]
        for rname, rule, n, h in rules:
            hit, cpu, up = run(ho, pinned, Sa, rule, n, h)
            ms = T0 + (C_CPU * cpu + C_UP * up) / T + SYNC * L
            row = {"vram": share, "pool": pool_frac, "rule": rname, "hit": hit / (T * L * K), "ms": ms}
            out["rows"].append(row)
            print(f"vram {share:.0%} pool {pool_frac:.0%} {rname:16s} hit {100*hit/(T*L*K):5.1f}% uploads/tok {up/T:6.1f} cpu/tok {cpu/T:6.1f}  {1000/ms:5.1f} tok/s", flush=True)
json.dump(out, open(W / "results/night2/p5_sim.json", "w"), indent=1)
