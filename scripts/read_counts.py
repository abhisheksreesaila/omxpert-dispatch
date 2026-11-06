"""Read per-expert routing counts from llama-imatrix GGUF output and analyze them.

usage: read_counts.py   (reads imatrix/qwen36/*.gguf, writes counts/qwen36_*.npy and results/qwen36/metrics.json)

imatrix stores, per MoE tensor blk.N.ffn_down_exps.weight, a `.counts` array with
the number of tokens routed to each expert.
"""
import json, re
from pathlib import Path
import numpy as np
from gguf import GGUFReader

W = Path.home() / "expert-dispatch"
IM, OUT = W / "imatrix/qwen36", W / "results/qwen36"
OUT.mkdir(parents=True, exist_ok=True)


def load(p):
    r = GGUFReader(str(p))
    rows = {}
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.ffn_down_exps\.weight\.counts", t.name)
        if m:
            rows[int(m.group(1))] = np.asarray(t.data, dtype=np.float64).ravel()
    L = max(rows) + 1
    return np.stack([rows[i] for i in range(L)])


C = {p.stem: load(p) for p in sorted(IM.glob("*.gguf"))}
for k, v in C.items():
    np.save(W / "counts" / f"qwen36_{k}.npy", v)
L, E = next(iter(C.values())).shape
tags = sorted({k.rsplit("_", 1)[0] for k in C if k.endswith("_train")})


def cov(c, f):
    s = np.sort(c, 1)[:, ::-1]
    return s[:, : int(E * f)].sum(1) / s.sum(1)


def top(c, n):
    return [set(np.argsort(-c[l])[:n]) for l in range(L)]


def static_hit(prof, ho, S):
    return float(sum(ho[l][list(top(prof, S)[l])].sum() for l in range(L)) / ho.sum())


m = {"model": "Qwen3.6-35B-A3B UD-Q4_K_M (llama.cpp imatrix counts)", "layers": L, "experts": E,
     "routings_per_layer": {k: float(v[0].sum()) for k, v in C.items()}}
m["coverage"] = {t: {f"top{int(f*100)}%": {"mean": float(cov(C[t+'_train'], f).mean()),
                                            "min": float(cov(C[t+'_train'], f).min())}
                     for f in (.10, .25, .50)} for t in tags}
q = E // 4
J = {x: {y: float(np.mean([len(a & b) / len(a | b) for a, b in zip(top(C[x+'_train'], q), top(C[y+'_train'], q))]))
         for y in tags} for x in tags}
m["overlap_jaccard_top25"] = J
m["overlap_mean_offdiag"] = float(np.mean([J[x][y] for x in tags for y in tags if x != y]))
m["overlap_random_baseline"] = 0.25 / 1.75
if all(f"{t}_trainA" in C for t in tags):
    m["churn_top25"] = {t: float(np.mean([1 - len(a & b) / q for a, b in zip(top(C[t+'_trainA'], q), top(C[t+'_trainB'], q))]))
                        for t in tags}
sizes = [E // 16, E // 8, E // 4, E // 2]
m["heldout_static_hit"] = {}
for t in tags:
    if f"{t}_heldout" not in C:
        continue
    other = sum(C[o + "_train"] for o in tags if o != t)
    m["heldout_static_hit"][t] = {
        "own_profile": {S: static_hit(C[t + "_train"], C[t + "_heldout"], S) for S in sizes},
        "other_tags_profile": {S: static_hit(other, C[t + "_heldout"], S) for S in sizes}}
json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
print(json.dumps(m, indent=1)[:3000])
