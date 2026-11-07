"""Night 5: learned next-layer expert predictor (the "learned prefetcher") and an offline test of every guesser.
Trains on routing traces [T, L, K]; for each held-out token and layer l, guesses N experts of layer l+1 before l+1 runs.
Guessers:
  profile : the N most used experts of layer l+1 in the train traces (a static hot set)
  prev    : the previous token's picks at layer l+1 (recency)
  markov  : learned table: experts that most often follow this token's picks at layer l (co-firing across layers)
  both    : learned score + a bonus for the previous token's picks, top N
Recall = share of layer l+1's real picks that were guessed (what a prefetch would have had in RAM in time).
usage: n5_predictor.py <model_dir> <train_tag> <heldout_tag> <out_table.bin> <out.json>
"""
import json, sys
from pathlib import Path
import numpy as np

W = Path.home() / "expert-dispatch"
md, tr_tag, ho_tag, out_bin, out_json = sys.argv[1:6]
tr = np.load(W / f"traces/{md}/{tr_tag}.npz")["topk"].astype(np.int64)
ho = np.load(W / f"traces/{md}/{ho_tag}.npz")["topk"].astype(np.int64)
T, L, K = tr.shape
E = max(int(tr.max()), int(ho.max())) + 1
M = 32

# co-firing counts C[l, e, e'] = how often e at layer l is followed by e' at layer l+1 in the same token
succ = np.full((L - 1, E, M), -1, np.int16)
for l in range(L - 1):
    C = np.zeros((E, E), np.int32)
    a, b = tr[:, l, :], tr[:, l + 1, :]
    np.add.at(C, (np.repeat(a, K, axis=1).ravel(), np.tile(b, (1, K)).ravel()), 1)
    order = np.argsort(-C, axis=1)[:, :M]
    cnt = np.take_along_axis(C, order, axis=1)
    succ[l] = np.where(cnt > 0, order, -1)
with open(out_bin, "wb") as f:
    np.array([L, E, M], np.int32).tofile(f); succ.tofile(f)

freq = np.stack([np.bincount(tr[:, l].ravel(), minlength=E) for l in range(L)])


def markov_scores(picks, l):
    s = np.zeros(E)
    for e in picks:
        row = succ[l, e]
        ok = row >= 0
        s[row[ok]] += 1.0 / (1 + np.arange(M))[ok]
    return s


res = {"model": md, "train_tokens": int(T), "heldout_tokens": int(ho.shape[0]), "L": int(L), "E": int(E), "K": int(K), "recall": {}}
Ho = ho[:3000]
for N in (10, 16, 24, 32):
    hits = {"profile": 0, "prev": 0, "markov": 0, "both": 0}; tot = 0
    for t in range(1, len(Ho)):
        for l in range(L - 1):
            real = set(Ho[t, l + 1].tolist()); tot += K
            prof = set(np.argsort(-freq[l + 1])[:N].tolist())
            prev = list(Ho[t - 1, l + 1].tolist())
            ms = markov_scores(Ho[t, l].tolist(), l)
            mk = list(np.argsort(-ms)[:N])
            sb = ms.copy(); sb[prev] += 1.0   # recency bonus on top of the learned co-firing score
            both = list(np.argsort(-sb)[:N])
            hits["profile"] += len(real & prof); hits["prev"] += len(real & set(prev[:N]))
            hits["markov"] += len(real & set(mk)); hits["both"] += len(real & set(both))
    res["recall"][N] = {k: round(v / tot, 3) for k, v in hits.items()}
    print(N, res["recall"][N], flush=True)
Path(out_json).write_text(json.dumps(res, indent=1))
