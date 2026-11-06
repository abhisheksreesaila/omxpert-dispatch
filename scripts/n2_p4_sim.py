"""Night 2 Phase 5 (P4): complete-the-set nudge vs global nudge, in simulation, with real router probabilities.

Input: held-out coding traces with router probabilities (EDX_TRACE_PROBS, f16 [T, L, E]) and top-k ids [T, L, K].
Resident set: the frequency pins (night-1 profile) at a given VRAM share.
Policies:
  none         the model's own top-k
  global(λ)    top-k of p * (resident ? 1 : e^-λ)   (EDX_STEER, already in llama.cpp)
  finish(r)    only token-layers with 1-2 misses: replace each missed expert by the best resident expert outside
               the chosen set, if its probability >= r * the missed expert's probability
Metrics: share of token-layers fully resident, simulated ms/token (always / skip runtime models), and the
routing probability mass changed (sum of p over experts dropped from the model's own top-k; quality proxy).
"""
import json, sys
from pathlib import Path
import numpy as np
import sim_critical_path as S

W = Path.home() / "expert-dispatch"
P = np.fromfile(W / "results/night2/probs_coding_heldout.f16", np.float16)
ids = np.load(W / "results/night2/ids_coding_heldout.npz")["topk"].astype(np.int64)
T, L, K = ids.shape; E = 256
P = P[: T * L * E].reshape(T, L, E).astype(np.float32)
tr = np.load(W / "traces/Qwen3.6-35B-A3B/coding_train.npz")["topk"].astype(np.int64)
cnt = np.stack([np.bincount(tr[:, l].ravel(), minlength=E) for l in range(L)]).astype(float)


def evaluate(sel, res_mask):
    """sel [T, L, K] chosen experts; returns metrics."""
    miss = (~res_mask[np.arange(L)[None, :, None], sel]).sum(2)          # [T, L]
    full = float((miss == 0).mean())
    out = {"fully_resident": full, "misses_per_tok": float(miss.sum(1).mean())}
    for always in (True, False):
        has = np.ones_like(miss, bool) if always else miss > 0
        t = S.BASE + (S.C_GPU * (K - miss) + np.where(has, S.C_CPU * np.maximum(1, miss) + S.SYNC, 0.0)).sum(1)
        out["ms_" + ("always" if always else "skip")] = float(t.mean())
    return out


def mass_changed(sel):
    own = np.sort(ids, 2); new = np.sort(sel, 2)
    dropped = 0.0
    for k in range(K):
        e = ids[:, :, k]
        kept = (sel == e[:, :, None]).any(2)
        dropped += np.where(kept, 0.0, np.take_along_axis(P, e[:, :, None], 2)[:, :, 0]).sum()
    return float(dropped / (T * L))


results = {}
for frac in (0.15, 0.30):
    S_slots = int(frac * L * E)
    res_mask = np.stack(S.frequency_pins(cnt, S_slots, min_per_layer=16))
    r = {"none": {**evaluate(ids, res_mask), "mass_changed": 0.0}}
    for lam in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        f = np.where(res_mask[None], 1.0, np.exp(-lam))
        sel = np.argsort(-(P * f), 2)[:, :, :K]
        r[f"global_{lam}"] = {**evaluate(sel, res_mask), "mass_changed": mass_changed(sel)}
    miss_mask = ~res_mask[np.arange(L)[None, :, None], ids]                # [T, L, K]
    nmiss = miss_mask.sum(2)
    for rr in (0.3, 0.5, 0.7):
        sel = ids.copy()
        cand = np.where(res_mask[None], P, -1.0)                           # resident experts' probabilities
        np.put_along_axis(cand, ids, -1.0, 2)                               # outside the chosen set
        order = np.argsort(-cand, 2)[:, :, :2]                              # best two resident alternatives
        for t, l in zip(*np.where((nmiss >= 1) & (nmiss <= 2))):
            alt = list(order[t, l]); ai = 0
            for k in np.where(miss_mask[t, l])[0]:
                if ai >= len(alt):
                    break
                a = alt[ai]
                if cand[t, l, a] >= rr * P[t, l, ids[t, l, k]]:
                    sel[t, l, k] = a; ai += 1
        r[f"finish_{rr}"] = {**evaluate(sel, res_mask), "mass_changed": mass_changed(sel)}
    results[f"{int(frac*100)}"] = r
    for k, v in r.items():
        print(f"VRAM {frac:.0%} {k:12s} full {100*v['fully_resident']:5.1f}%  misses/tok {v['misses_per_tok']:6.1f}  "
              f"tok/s always {1000/v['ms_always']:5.1f} skip {1000/v['ms_skip']:5.1f}  mass changed {v['mass_changed']:.4f}", flush=True)
json.dump(results, open(W / "results/night2/p4_sim.json", "w"), indent=1)
