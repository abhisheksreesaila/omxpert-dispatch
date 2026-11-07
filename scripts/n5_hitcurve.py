"""Held-out hit rate of a per-expert coding profile vs whole-layer placement, by share of experts resident (Qwen3.8).
usage: n5_hitcurve.py <model_dir> <out.json>"""
import json, sys
from pathlib import Path
import numpy as np
W = Path.home() / "expert-dispatch"; md, out = sys.argv[1:3]
tr = np.load(W / f"traces/{md}/coding_train.npz")["topk"].astype(np.int64)
ho = np.load(W / f"traces/{md}/coding_heldout.npz")["topk"].astype(np.int64)
T, L, K = tr.shape; E = 512
cnt = np.stack([np.bincount(tr[:, l].ravel(), minlength=E) for l in range(L)]).astype(float)
hcnt = np.stack([np.bincount(ho[:, l].ravel(), minlength=E) for l in range(L)]).astype(float)
order = np.argsort(-cnt.ravel())
res = {"per_expert": {}, "per_layer": {}}
for f in (0.1, 0.2, 0.3, 0.45, 0.56, 0.7, 0.85):
    n = int(f * L * E); keep = np.zeros(L * E, bool); keep[order[:n]] = True
    res["per_expert"][f] = round(float(hcnt.ravel()[keep].sum() / hcnt.sum()), 3)
    res["per_layer"][f] = round(f, 3)   # whole layers: hit share = resident share (every layer is used equally)
res["top_share"] = {p: round(float(np.sort(hcnt.ravel())[::-1][:int(p * L * E)].sum() / hcnt.sum()), 3) for p in (0.1, 0.25, 0.5)}
Path(out).write_text(json.dumps(res, indent=1)); print(json.dumps(res))
