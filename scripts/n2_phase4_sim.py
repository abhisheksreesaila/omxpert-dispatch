"""Night 2 Phase 4 (simulation): P2 set-completing pins and P3 per-layer whole-vs-pins, vs baselines.

Per layer l and slot count S, cost_l(S) = mean simulated layer time on TRAIN traces with the layer's best
S-expert set (frequency order, or greedy set-completion), and cost_l(whole) for a fully resident layer.
A DP over layers splits the VRAM budget (in expert slots, granularity G) to minimise total time.
Evaluated on HELD-OUT traces at 15/30/60% of expert slots. Two runtime models:
  always: the CPU pass runs whenever a layer has any CPU-placed expert (llama.cpp static graph, EDX today)
  skip:   the CPU pass runs only when the token misses in that layer (a dynamic runtime; upper bound)
writes results/night2/phase4_sim.json and the chosen plans (for real runs) to results/night2/plans/.
"""
import json
from pathlib import Path
import numpy as np
import sim_critical_path as S

W = Path.home() / "expert-dispatch"
O = W / "results/night2"; (O / "plans").mkdir(parents=True, exist_ok=True)
tr = np.load(W / "traces/Qwen3.6-35B-A3B/coding_train.npz")["topk"].astype(np.int64)[-60000:]
ho = np.load(W / "traces/Qwen3.6-35B-A3B/coding_heldout.npz")["topk"].astype(np.int64)[:30000]
T, L, K = tr.shape; E = 256; G = 8
cnt = np.stack([np.bincount(tr[:, l].ravel(), minlength=E) for l in range(L)]).astype(float)


def layer_cost(topk_l, mask, always, mode="sum"):
    res = mask[topk_l].sum(1); miss = K - res
    has = np.ones_like(miss, bool) if always else miss > 0
    gpu = S.C_GPU * res; cpu = np.where(has, S.C_CPU * np.maximum(1, miss), 0.0); sync = np.where(has, S.SYNC, 0.0)
    return float(((np.maximum(gpu, cpu) if mode == "max" else gpu + cpu) + sync).mean())


def freq_order(l):
    return np.argsort(-cnt[l])


def setcomplete_order(l, always, max_s=256):
    """greedy: add the expert that most reduces the layer cost on train (favours experts that complete 8-sets)."""
    x = tr[:, l]; mask = np.zeros(E, bool); order = []
    # cheap incremental version: track per-token miss counts; gain of adding e = sum over tokens containing e of
    # (cost(miss) - cost(miss-1)); cost has a cliff at miss 1->0 in the skip model (sync + cpu floor saved).
    miss = np.full(len(x), K)
    tok_of = [np.where((x == e).any(1))[0] for e in range(E)]
    def c(m):
        has = (m > 0) | always
        return np.where(has, S.C_CPU * np.maximum(1, m) + S.SYNC, 0.0)
    for _ in range(max_s):
        best, be = -1.0, -1
        for e in range(E):
            if mask[e] or len(tok_of[e]) == 0:
                continue
            t = tok_of[e]; g = (c(miss[t]) - c(miss[t] - 1)).sum()
            if g > best:
                best, be = g, e
        if be < 0:
            rest = [e for e in range(E) if not mask[e]]
            order += rest; break
        mask[be] = True; order.append(be); miss[tok_of[be]] -= 1
    return np.array(order[:max_s])


def cost_curve(l, order, always, data):
    xs = list(range(0, E + 1, G)); out = []
    for s in xs:
        m = np.zeros(E, bool); m[order[:s]] = True
        out.append(layer_cost(data[:, l], m, always))
    return xs, out


def dp(curves, whole_cost, budget):
    """curves[l] = (xs, costs) over slot counts; option 'whole' costs whole_cost at E slots. Minimise sum."""
    nb = budget // G; INF = 1e18
    best = np.full(nb + 1, INF); best[0] = 0.0; choice = []
    for l in range(L):
        xs, cs = curves[l]; nbest = np.full(nb + 1, INF); arg = np.zeros((nb + 1, 2), int)
        opts = [(x // G, c, 0) for x, c in zip(xs, cs)] + [(E // G, whole_cost[l], 1)]
        for b in range(nb + 1):
            if best[b] >= INF:
                continue
            for w, c, is_whole in opts:
                if b + w <= nb and best[b] + c < nbest[b + w]:
                    nbest[b + w] = best[b] + c; arg[b + w] = (w, is_whole)
        choice.append(arg); best = nbest
    b = int(np.argmin(best)); plan = []
    for l in range(L - 1, -1, -1):
        w, is_whole = choice[l][b]; plan.append((int(w * G), bool(is_whole))); b -= w
    return plan[::-1]


def evaluate(plan_masks, always):
    return float(S.token_times(ho, plan_masks, "sum", cpu_pass_always=always).mean())


results = {}
for always in (True, False):
    tag = "always" if always else "skip"
    orders = {"freq": [freq_order(l) for l in range(L)]}
    orders["setc"] = [setcomplete_order(l, always) for l in range(L)]
    whole_cost = [S.C_GPU * K] * L
    for frac in (0.15, 0.30, 0.60):
        B = int(frac * L * E); r = {}
        # baselines
        fp = S.frequency_pins(cnt, B)
        r["freq_pins_global"] = evaluate(fp, always)
        nwhole = B // E
        r["whole_layers"] = evaluate(["cpu"] * (L - nwhole) + ["whole"] * nwhole, always)
        r["no_gpu_experts"] = evaluate(["cpu"] * L, always)
        for oname, order in orders.items():
            curves = [cost_curve(l, order[l], always, tr) for l in range(L)]
            plan = dp(curves, whole_cost, B)
            masks = []
            for l, (s, whole) in enumerate(plan):
                if whole:
                    masks.append("whole")
                elif s == 0:
                    masks.append("cpu")
                else:
                    m = np.zeros(E, bool); m[order[l][:s]] = True; masks.append(m)
            r[f"dp_hybrid_{oname}"] = evaluate(masks, always)
            r[f"dp_hybrid_{oname}_whole_layers"] = int(sum(w for _, w in plan))
            # pins-only DP (P2 without P3)
            plan2 = dp(curves, [1e9] * L, B)
            masks2 = []
            for l, (s2, _) in enumerate(plan2):
                if s2 == 0:
                    masks2.append("cpu")
                else:
                    m = np.zeros(E, bool); m[order[l][:s2]] = True; masks2.append(m)
            r[f"dp_pins_{oname}"] = evaluate(masks2, always)
            if oname == ("setc" if not always else "freq"):
                json.dump({"frac": frac, "model": tag, "order": oname,
                           "layers": [{"layer": l, "whole": w, "slots": s, "experts": [int(e) for e in order[l][:s]] if not w else []}
                                      for l, (s, w) in enumerate(plan)]},
                          open(O / f"plans/plan_{tag}_{int(frac*100)}.json", "w"), indent=1)
        results[f"{tag}_{int(frac*100)}"] = r
        print(tag, f"{frac:.0%}", {k: (round(1000 / v, 1) if isinstance(v, float) else v) for k, v in r.items()}, "(tok/s)", flush=True)
json.dump(results, open(O / "phase4_sim.json", "w"), indent=1)
