"""Auto-pick the expert lineup from the request ("captain reads the pitch").
For held-out windows: prompt = P tokens, answer = the next A tokens.
Lineups of S experts/layer, scored by hit rate on the answer:
  own     profile of the window's true use case
  other   the other use case's profile
  auto    the profile whose expert histogram is closest (cosine) to the prompt's
  prompt  the prompt's own top-S experts
  blend   a*profile(auto) + (1-a)*prompt, both normalised per layer
usage: autopick_sim.py <model_dir> [P A]"""
import sys, json
from pathlib import Path
import numpy as np
W = Path.home() / "expert-dispatch"; name = sys.argv[1]
P, A = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (512, 256)
tags = ["coding", "design_chat"]
ld = lambda t, s: np.load(W / f"traces/{name}/{t}_{s}.npz")["topk"].astype(np.int64)
tr = {t: ld(t, "train") for t in tags}; ho = {t: ld(t, "heldout") for t in tags}
T, L, K = tr["coding"].shape; E = int(max(a.max() for a in tr.values())) + 1
hist = lambda x: np.stack([np.bincount(x[:, l].ravel(), minlength=E) for l in range(L)]).astype(float)
prof = {t: hist(tr[t]) for t in tags}
norm = lambda h: h / np.maximum(h.sum(1, keepdims=True), 1)
def hit(lineup_score, ans, S):
    top = np.argsort(-lineup_score, 1)[:, :S]; m = np.zeros((L, E), bool); np.put_along_axis(m, top, True, 1)
    return m[np.arange(L)[None, :, None], ans].mean()
out = {"P": P, "A": A, "rows": {}}
for frac in (0.15, 0.30):
    S = max(1, int(E * frac)); acc = {k: [] for k in ["own", "other", "auto", "prompt", "blend0.5", "blend0.8"]}; correct = []
    for t in tags:
        x = ho[t]
        for s in range(0, len(x) - P - A, P + A):
            pr, an = x[s:s + P], x[s + P:s + P + A]; hp = norm(hist(pr))
            sims = {u: (norm(prof[u]) * hp).sum() / (np.linalg.norm(norm(prof[u])) * np.linalg.norm(hp)) for u in tags}
            pick = max(sims, key=sims.get); correct.append(pick == t)
            o = [u for u in tags if u != t][0]
            acc["own"].append(hit(prof[t], an, S)); acc["other"].append(hit(prof[o], an, S))
            acc["auto"].append(hit(prof[pick], an, S)); acc["prompt"].append(hit(hp + 1e-9 * norm(prof[pick]), an, S))
            for a in (0.5, 0.8):
                acc[f"blend{a}"].append(hit(a * norm(prof[pick]) + (1 - a) * hp, an, S))
    r = {k: float(np.mean(v)) for k, v in acc.items()}; r["auto_pick_accuracy"] = float(np.mean(correct)); r["windows"] = len(correct)
    out["rows"][frac] = r
    print(f"VRAM {frac:.0%}: " + "  ".join(f"{k} {v*100:.1f}" for k, v in r.items() if k != "windows"), f"(n={len(correct)})", flush=True)
json.dump(out, open(W / f"results/{name}/autopick.json", "w"), indent=1)
