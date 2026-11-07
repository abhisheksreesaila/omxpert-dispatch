"""Night 6: held-out hit curve of a profile built from the owner's chat history, against the night-5 coding profile, on held-out
chat sessions (new conversations). Hit = share of the held-out expert picks whose expert is in the resident set.
usage: n6_hitcurve.py <out.json>"""
import json, sys
from pathlib import Path
import numpy as np

W = Path.home() / "expert-dispatch"; T = W / "traces/Qwen3.8-Flash-Next"; E = 512
ld = lambda n: np.load(T / f"{n}.npz")["topk"].astype(np.int64)
ctr, cho, kod = ld("chat_train"), ld("chat_heldout"), ld("coding_train")
L = ctr.shape[1]
cnt = lambda a: np.stack([np.bincount(a[:, l].ravel(), minlength=E) for l in range(L)]).astype(float)
c_chat, c_code = cnt(ctr), cnt(kod)
h = cnt(cho)


def curve(c):
    order = np.argsort(-c.ravel()); out = {}
    for f in (0.1, 0.2, 0.3, 0.45, 0.56, 0.7):
        keep = np.zeros(L * E, bool); keep[order[:int(f * L * E)]] = True
        out[f] = round(float(h.ravel()[keep].sum() / h.sum()), 3)
    return out


def curve_per_layer_budget(c):   # same number of experts in every layer (what a per-layer cache can hold)
    out = {}
    for f in (0.1, 0.2, 0.3, 0.45, 0.56, 0.7):
        k = int(f * E); hit = sum(h[l, np.argsort(-c[l])[:k]].sum() for l in range(L))
        out[f] = round(float(hit / h.sum()), 3)
    return out


# how quickly the profile settles: train on the first N tokens of chat history
learn = {}
for n in (2000, 5000, 10000, 20000, len(ctr)):
    n = min(n, len(ctr)); learn[n] = curve(cnt(ctr[:n]))[0.3]
res = {"source": "owner_chat_history", "corpus_tokens": int(len(ctr)), "heldout_tokens": int(len(cho)),
       "hit_curve": curve(c_chat), "hit_curve_per_layer_budget": curve_per_layer_budget(c_chat),
       "coding_profile_on_chat": curve(c_code), "oracle_on_chat": curve(h), "uniform": {f: f for f in (0.1, 0.2, 0.3, 0.45, 0.56, 0.7)},
       "learning_curve_at_30pct": learn,
       "note": "hit = share of held-out chat expert picks served by the resident set; oracle = the held-out set's own counts (ceiling)"}
Path(sys.argv[1]).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
np.save(W / "results/night6/profile_chat_counts.npy", c_chat)
