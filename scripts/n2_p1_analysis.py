"""P1: does per-token decode time track layers-with-any-miss better than total misses?"""
import json, struct, glob
from pathlib import Path
import numpy as np
W = Path.home() / "expert-dispatch"; O = W / "results/night2/phase2"

def read_trace(path):
    recs = []; b = open(path, "rb").read(); i = 0
    while i + 16 <= len(b):
        t, nt, nr = struct.unpack_from("<qii", b, i); i += 16; layers = {}
        for _ in range(nr):
            il, n = struct.unpack_from("<ii", b, i); i += 8
            layers[il] = np.frombuffer(b, np.int32, n, i); i += 4 * n
        recs.append((t, nt, layers))
    return recs

def slots(path):
    b = open(path, "rb").read(); L, E = struct.unpack_from("<ii", b, 0)
    return np.frombuffer(b, np.int32, L * E, 8).reshape(L, E)

out = {}
for name in ("pin_coding", "pin_mixed"):
    X = []; Y = []
    for tp in sorted(glob.glob(str(O / f"{name}_p*.trace"))):
        recs = read_trace(tp); sm = slots(tp + ".slots")
        recs = [r for r in recs if r[1] == 1]  # single-token decode steps
        for (t0, _, _), (t1, _, lay) in zip(recs, recs[1:]):
            dt = (t1 - t0) / 1e3
            miss = [int((sm[il][ids] < 0).sum()) for il, ids in lay.items() if sm[il][0] != -2]
            X.append([sum(miss), sum(m > 0 for m in miss), max(miss) if miss else 0]); Y.append(dt)
    X = np.array(X, float); Y = np.array(Y)
    keep = Y < np.percentile(Y, 99)  # drop stalls (page faults, scheduler hiccups)
    X, Y = X[keep], Y[keep]
    def r2(cols):
        A = np.c_[X[:, cols], np.ones(len(Y))]; coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        res = Y - A @ coef; return float(1 - res.var() / Y.var()), coef.tolist()
    o = {"tokens": int(len(Y)), "mean_ms": float(Y.mean()),
         "mean_misses": float(X[:, 0].mean()), "mean_layers_with_miss": float(X[:, 1].mean()), "mean_max_miss": float(X[:, 2].mean())}
    for nm, cols in (("misses", [0]), ("layers_with_any_miss", [1]), ("max_miss", [2]), ("all", [0, 1, 2])):
        o[f"R2_{nm}"], o[f"coef_{nm}"] = r2(cols)
    out[name] = o
json.dump(out, open(W / "results/night2/p1.json", "w"), indent=1)
for k, v in out.items():
    print(f"- P1 {k}: n={v['tokens']} mean {v['mean_ms']:.2f} ms; misses/tok {v['mean_misses']:.1f}, layers-with-miss {v['mean_layers_with_miss']:.1f}/40; "
          f"R2 misses {v['R2_misses']:.3f}, layers-with-any-miss {v['R2_layers_with_any_miss']:.3f}, max-miss {v['R2_max_miss']:.3f}, all {v['R2_all']:.3f}")
