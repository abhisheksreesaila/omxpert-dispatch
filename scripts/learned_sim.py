"""Learnable expert cache, simulated on real per-token traces.

Each layer keeps S resident experts. On a miss the expert is fetched and the
resident expert with the lowest score is evicted. The score is learned:
    score = w_rec * exp(-age / tau) + w_freq * freq_ema + w_prior * profile + w_xl * xlayer
  - age:     tokens since last use (w_rec alone == LRU)
  - freq:    decayed use count
  - profile: train-split frequency (the night-1 profile)
  - xlayer:  expected use at this layer predicted from this token's experts at
             the previous layer (transition table learned on train)
Optional cross-layer prefetch: before layer l runs, fetch the top-m predicted
experts that are not resident (counted as prefetch traffic, assumed overlapped).

Weights are fit by random search on the train split, scored on held-out.
usage: learned_sim.py <model_dir> [tags...]
"""
import json, sys, time
from pathlib import Path
import numpy as np
from numba import njit

W = Path.home() / "expert-dispatch"
name = sys.argv[1]
TR = W / "traces" / name
tags = sys.argv[2:] or ["coding", "design_chat", "docs", "math", "tool_calls"]
SIZES = [16, 32, 64]
FIT_TOKENS, EVAL_TOKENS = 20_000, 60_000


@njit(cache=True)
def sim(topk, S, prior, X, w_rec, tau, w_freq, fdecay, w_prior, w_xl, m_pref):
    T, L, K = topk.shape
    E = prior.shape[1]
    res = np.zeros((L, E), np.bool_)
    nres = np.zeros(L, np.int64)
    last = np.full((L, E), -1e9)
    freq = np.zeros((L, E))
    xs = np.zeros(E)
    score = np.zeros(E)
    need = np.zeros(E, np.bool_)
    hits = 0; acc = 0; pref = 0; pref_used = 0
    pref_flag = np.zeros((L, E), np.bool_)
    for t in range(T):
        for l in range(L):
            # cross-layer prediction for this layer from previous layer's experts
            xs[:] = 0.0
            if l > 0:
                for j in range(K):
                    e0 = topk[t, l - 1, j]
                    for e in range(E):
                        xs[e] += X[l - 1, e0, e]
            for e in range(E):
                freq[l, e] *= fdecay
                score[e] = (w_rec * np.exp(-(t - last[l, e]) / tau) + w_freq * freq[l, e]
                            + w_prior * prior[l, e] + w_xl * xs[e])
            for e in range(E):
                need[e] = False
            for j in range(K):
                need[topk[t, l, j]] = True
            # prefetch top-m predicted non-resident experts
            if m_pref > 0 and l > 0:
                order = np.argsort(-xs)
                got = 0
                for q in range(E):
                    if got >= m_pref:
                        break
                    e = order[q]
                    if res[l, e]:
                        continue
                    got += 1
                    pref += 1
                    if nres[l] < S:
                        res[l, e] = True; nres[l] += 1
                    else:
                        v = -1; vs = 1e300
                        for c in range(E):
                            if res[l, c] and score[c] < vs and xs[c] < xs[e]:
                                vs = score[c]; v = c
                        if v < 0:
                            continue
                        res[l, v] = False; res[l, e] = True
                    pref_flag[l, e] = True
                    score[e] = 1e200  # just fetched: don't evict it this layer
            for j in range(K):
                e = topk[t, l, j]
                acc += 1
                if res[l, e]:
                    hits += 1
                    if pref_flag[l, e]:
                        pref_used += 1
                else:
                    if nres[l] < S:
                        res[l, e] = True; nres[l] += 1
                    else:
                        v = -1; vs = 1e300
                        for c in range(E):
                            if res[l, c] and not need[c] and score[c] < vs:
                                vs = score[c]; v = c
                        res[l, v] = False; res[l, e] = True
                pref_flag[l, e] = False
                last[l, e] = t
                freq[l, e] += 1.0
                score[e] = 1e200
    return hits / acc, pref / (T * L), pref_used


def load(tag, split):
    return np.load(TR / f"{tag}_{split}.npz")["topk"].astype(np.int64)


def tables(tr, E):
    T, L, K = tr.shape
    prior = np.zeros((L, E))
    for l in range(L):
        prior[l] = np.bincount(tr[:, l].ravel(), minlength=E)
    prior /= prior.mean(1, keepdims=True)
    X = np.zeros((L - 1, E, E))  # X[l, e0, e] = P(e used at l+1 | e0 used at l) / K
    for l in range(L - 1):
        a, b = tr[:, l], tr[:, l + 1]
        idx = (a[:, :, None] * E + b[:, None, :]).ravel()
        X[l] = np.bincount(idx, minlength=E * E).reshape(E, E)
        X[l] /= np.maximum(np.bincount(a.ravel(), minlength=E), 1)[:, None]
    X /= K
    return prior, X


LRU = dict(w_rec=1.0, tau=50.0, w_freq=0.0, fdecay=0.9, w_prior=0.0, w_xl=0.0, m_pref=0)


def run(topk, S, prior, X, p):
    return sim(topk, S, prior, X, p["w_rec"], p["tau"], p["w_freq"], p["fdecay"], p["w_prior"], p["w_xl"], p["m_pref"])


rng = np.random.default_rng(0)
out = {}
for tag in tags:
    tr, ho = load(tag, "train"), load(tag, "heldout")[:EVAL_TOKENS]
    E = int(max(tr.max(), ho.max())) + 1
    prior, X = tables(tr[:-FIT_TOKENS], E)  # tables from older train, fit on the newest train slice
    fit = tr[-FIT_TOKENS:]
    out[tag] = {}
    for S in SIZES:
        t0 = time.time()
        best, bp = run(fit, S, prior, X, LRU)[0], dict(LRU)
        lru_fit = best
        for it in range(40):  # random search around the best so far
            p = dict(bp) if it > 8 else dict(LRU)
            p["tau"] = float(np.exp(rng.uniform(np.log(2), np.log(2000))))
            p["w_freq"] = float(rng.choice([0, rng.exponential(0.05)]))
            p["fdecay"] = float(1 - np.exp(rng.uniform(np.log(1e-3), np.log(0.3))))
            p["w_prior"] = float(rng.choice([0, rng.exponential(0.1)]))
            p["w_xl"] = float(rng.choice([0, rng.exponential(1.0)]))
            h = run(fit, S, prior, X, p)[0]
            if h > best:
                best, bp = h, p
        prior_full, X_full = tables(tr, E)
        r = {"lru": run(ho, S, prior_full, X_full, LRU)[0]}
        r["learned_evict"] = run(ho, S, prior_full, X_full, bp)[0]
        r["params"] = bp
        r["fit_gain"] = best - lru_fit
        for m in (2, 4, 8):
            p = dict(bp, m_pref=m)
            h, traffic, _ = run(ho, S, prior_full, X_full, p)
            r[f"learned+prefetch{m}"] = h
            r[f"prefetch{m}_per_layer_token"] = traffic
        out[tag][S] = r
        print(tag, S, {k: (round(v * 100, 1) if isinstance(v, float) else v) for k, v in r.items() if k != "params"},
              f"{time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(W / f"results/{name}/learned_sim.json", "w"), indent=1)
print("done")
