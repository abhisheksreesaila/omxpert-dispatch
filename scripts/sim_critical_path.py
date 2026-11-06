"""Night 2 Phase 3: critical-path simulator for MoE decode with experts split between VRAM and RAM.

Per token, per MoE layer:
  model "max":  max(gpu_part, cpu_part) + sync * [layer has a CPU part]       (the runbook's theory)
  model "sum":  gpu_part + cpu_part + sync * [layer has a CPU part]           (llama.cpp's sequential splits)
with
  gpu_part = c_gpu * resident chosen experts            (VRAM reads, small)
  cpu_part = c_cpu * max(1, missed chosen) if the layer's CPU pass runs      (EDX pinned mode always runs it,
             using a dummy expert when nothing misses; "skip" mode drops the pass when nothing misses)
Whole layers on the GPU never have a CPU part.

Placements are described per layer: "whole" (all experts on GPU), "cpu" (none), or a resident set.
usage: imported by n2_phase3/4 scripts; `python sim_critical_path.py validate` checks against Phase-2 measurements.
"""
import json, sys, struct, glob, re
from pathlib import Path
import numpy as np

W = Path.home() / "expert-dispatch"
C = json.load(open(W / "results/night2/costs.json"))
C_CPU = C["c_cpu_ms_per_expert"]
SYNC = C["sync_ms_per_cpu_layer"]
BASE = C["gpu_base_ms_k8"]          # attention, shared experts, router, output: everything not routed-expert
C_GPU = 0.004                        # ms per resident expert read (1.8 MB at ~500 GB/s effective); tiny either way


def layer_cost(n_res, n_miss, mode, cpu_pass_always=True):
    has_cpu = n_miss > 0 or cpu_pass_always
    gpu = C_GPU * n_res
    cpu = C_CPU * max(1, n_miss) if has_cpu else 0.0
    sync = SYNC if has_cpu else 0.0
    return (max(gpu, cpu) if mode == "max" else gpu + cpu) + sync


def token_times(topk, placement, mode="sum", cpu_pass_always=True):
    """topk [T, L, K]; placement: list of per-layer entries: 'whole' | 'cpu' | bool mask [E] of resident experts."""
    T, L, K = topk.shape
    out = np.full(T, BASE)
    for l in range(L):
        p = placement[l]
        if isinstance(p, str) and p == "whole":
            out += C_GPU * K
            continue
        if isinstance(p, str) and p == "cpu":
            res = np.zeros(T, int)
        else:
            res = p[topk[:, l]].sum(1)
        miss = K - res
        has_cpu = (miss > 0) | cpu_pass_always
        gpu = C_GPU * res
        cpu = np.where(has_cpu, C_CPU * np.maximum(1, miss), 0.0)
        sync = np.where(has_cpu, SYNC, 0.0)
        out += (np.maximum(gpu, cpu) if mode == "max" else gpu + cpu) + sync
    return out


def placement_from_slots(sm):
    """slot map [L, E] from a trace (.slots): -2 = layer not cached (treated as whole if on GPU, decided by caller)."""
    return [m >= 0 for m in sm]


def frequency_pins(counts, budget_slots, min_per_layer=0):
    """global knapsack by frequency (the night-1 profile), optional per-layer minimum."""
    L, E = counts.shape
    order = np.dstack(np.unravel_index(np.argsort(-counts.ravel()), counts.shape))[0]
    masks = np.zeros((L, E), bool); taken = np.zeros(L, int); n = 0
    if min_per_layer:
        for l in range(L):
            masks[l, np.argsort(-counts[l])[:min_per_layer]] = True; n += min_per_layer
    for l, e in order:
        if n >= budget_slots:
            break
        if not masks[l, e]:
            masks[l, e] = True; n += 1
    return [m for m in masks]


def read_trace(path):
    recs = []; b = open(path, "rb").read(); i = 0
    while i + 16 <= len(b):
        t, nt, nr = struct.unpack_from("<qii", b, i); i += 16; layers = {}
        for _ in range(nr):
            il, n = struct.unpack_from("<ii", b, i); i += 8
            layers[il] = np.frombuffer(b, np.int32, n, i).copy(); i += 4 * n
        recs.append((t, nt, layers))
    return recs


def trace_topk(path, L=40, K=8):
    recs = [r for r in read_trace(path) if r[1] == 1]
    tk = np.zeros((len(recs), L, K), np.int64)
    for i, (_, _, lay) in enumerate(recs):
        for il, ids in lay.items():
            tk[i, il] = ids[:K]
    return tk


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "validate":
    O = W / "results/night2/phase2"
    meas = {}
    for l in open(O / "times.log"):
        p = l.split()
        if len(p) >= 3 and p[2].startswith("ms=") and p[2][3:] and (len(p) < 4 or float(p[3][5:]) < 4.0):  # drop runs started under foreign load
            meas.setdefault(p[0], []).append(float(p[2][3:]))
    tks = [trace_topk(t) for t in sorted(glob.glob(str(O / "pin_coding_p*.trace")))]
    topk = np.concatenate(tks)
    sm = {}
    for name in ("pin_coding", "pin_mixed"):
        f = sorted(glob.glob(str(O / f"{name}_p*.trace.slots")))[0]
        b = open(f, "rb").read(); L, E = struct.unpack_from("<ii", b, 0)
        sm[name] = np.frombuffer(b, np.int32, L * E, 8).reshape(L, E)
    L = topk.shape[1]
    places = {"pin_coding": placement_from_slots(sm["pin_coding"]), "pin_mixed": placement_from_slots(sm["pin_mixed"]),
              "layers": ["cpu"] * (L - 6) + ["whole"] * 6}   # --n-cpu-moe 34: first 34 layers' experts on CPU
    res = {}
    for mode in ("sum", "max"):
        for name, pl in places.items():
            always = name != "layers"
            sim = token_times(topk, pl, mode, cpu_pass_always=always).mean()
            m = float(np.mean(meas.get(name, [np.nan])))
            res[f"{mode}:{name}"] = {"sim_ms": round(float(sim), 2), "measured_ms": round(m, 2), "err_%": round(100 * (sim / m - 1), 1)}
    json.dump(res, open(W / "results/night2/sim_validation.json", "w"), indent=1)
    for k, v in res.items():
        print(f"- SIM {k}: sim {v['sim_ms']} ms vs measured {v['measured_ms']} ms ({v['err_%']:+.1f}%)")
