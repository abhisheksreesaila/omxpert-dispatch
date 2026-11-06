"""Phase 4 analysis. usage: analyze.py <model_dir_name>   (e.g. OLMoE-1B-7B-0125-Instruct, max)

Reads traces/<model>/<tag>_<split>.npz (topk [T, L, k]), writes
results/<model>/metrics.json and PNG plots. Cache hit rates come from the Mojo
simulator (mojo/cachesim).
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = Path.home() / "expert-dispatch"
name = sys.argv[1]
TR, OUT = W / "traces" / name, W / "results" / name
OUT.mkdir(parents=True, exist_ok=True)
SIM = W / "mojo/cachesim"

traces = {p.stem: np.load(p)["topk"].astype(np.int32) for p in sorted(TR.glob("*.npz"))}
tags = sorted({k.rsplit("_", 1)[0] for k in traces if k.endswith("_train")})
T0, L, K = traces[f"{tags[0]}_train"].shape
E = int(max(t.max() for t in traces.values())) + 1
E = 1 << (E - 1).bit_length()  # round to power of two (32, 64, 128, 256)


def counts(tr):
    c = np.zeros((L, E), np.int64)
    for l in range(L):
        c[l] = np.bincount(tr[:, l].ravel(), minlength=E)
    return c


def coverage(c, frac):
    s = np.sort(c, 1)[:, ::-1]
    n = max(1, int(round(E * frac)))
    return s[:, :n].sum(1) / s.sum(1)


def topset(c, frac):
    n = int(round(E * frac))
    return [set(np.argsort(-c[l])[:n]) for l in range(L)]


m = {"model": name, "layers": L, "experts": E, "k": K, "tokens": {k: len(v) for k, v in traces.items()}}
C = {t: counts(traces[f"{t}_train"]) for t in tags}
for t in tags:
    (W / "counts").mkdir(exist_ok=True)
    np.save(W / "counts" / f"{name}_{t}.npy", C[t])

# 1. coverage
m["coverage"] = {t: {f"top{int(f*100)}%": {"mean": float(coverage(C[t], f).mean()),
                                            "min": float(coverage(C[t], f).min()),
                                            "max": float(coverage(C[t], f).max())}
                     for f in (0.10, 0.25, 0.50)} for t in tags}
m["uniform_baseline_top25%"] = 0.25

# 2. cross-tag overlap (Jaccard of top-25% sets, layer mean)
sets = {t: topset(C[t], 0.25) for t in tags}
J = np.array([[np.mean([len(a & b) / len(a | b) for a, b in zip(sets[x], sets[y])]) for y in tags] for x in tags])
m["overlap_jaccard_top25"] = {x: {y: float(J[i, j]) for j, y in enumerate(tags)} for i, x in enumerate(tags)}
m["overlap_mean_offdiag"] = float(J[~np.eye(len(tags), dtype=bool)].mean()) if len(tags) > 1 else None
# random top-25% sets would give Jaccard ≈ 0.25/(2-0.25)
m["overlap_random_baseline"] = 0.25 / 1.75

# 3. held-out hit rates via Mojo simulator
sizes = [s for s in (E // 32, E // 16, E // 8, E // 4, E // 2, 3 * E // 4) if s >= 2]
m["cache_sizes"] = sizes
m["hit_rate"] = {}
for t in tags:
    ho = traces.get(f"{t}_heldout")
    if ho is None:
        continue
    ho = np.ascontiguousarray(ho[:60000])  # cap replay length
    hb, pb = OUT / f"_{t}_ho.bin", OUT / f"_{t}_prof.bin"
    ho.astype(np.int32).tofile(hb)
    C[t].astype(np.int32).tofile(pb)
    csv = subprocess.run([str(SIM), str(hb), str(len(ho)), str(L), str(K), str(E), str(pb),
                          ",".join(map(str, sizes))], capture_output=True, text=True, check=True).stdout
    agg = {}
    for line in csv.strip().splitlines()[1:]:
        pol, S, _, h, n = line.split(",")
        a = agg.setdefault(pol, {}).setdefault(int(S), [0, 0])
        a[0] += int(h); a[1] += int(n)
    m["hit_rate"][t] = {pol: {S: v[0] / v[1] for S, v in d.items()} for pol, d in agg.items()}
    # cross-profile control: serve this tag's held-out with *another* tag's profile
    hb.unlink(); pb.unlink()

# 3b. prompt-conditioned predictor ("look at the task, then pick the experts"):
# per 2048-token chunk, choose top-S from profile + routing of the first 128 tokens,
# score hits on the rest of the chunk.
def session_pred(t, S, warm=128, w_prof=0.5):
    ho = traces[f"{t}_heldout"]
    prof = C[t] / C[t].sum(1, keepdims=True)
    hits = tot = 0
    for s in range(0, len(ho) - 2048 + 1, 2048):
        ch = ho[s:s + 2048]
        for l in range(L):
            first = np.bincount(ch[:warm, l].ravel(), minlength=E)
            score = w_prof * prof[l] + (1 - w_prof) * first / first.sum()
            keep = np.zeros(E, bool); keep[np.argsort(-score)[:S]] = True
            rest = ch[warm:, l].ravel()
            hits += keep[rest].sum(); tot += rest.size
    return hits / max(tot, 1)

for t in m["hit_rate"]:
    m["hit_rate"][t]["session_predictor"] = {S: float(session_pred(t, S)) for S in sizes}
    # generic-profile control: a profile built from *other* tags only
    if len(tags) < 2:
        continue
    other = sum(C[o] for o in tags if o != t)
    ho = traces[f"{t}_heldout"][:60000]
    m["hit_rate"][t]["other_tags_profile"] = {}
    for S in sizes:
        h = 0
        for l in range(L):
            keep = np.zeros(E, bool); keep[np.argsort(-other[l])[:S]] = True
            h += keep[ho[:, l].ravel()].sum()
        m["hit_rate"][t]["other_tags_profile"][S] = float(h / ho[:, :].size)

# 4. decode vs prefill
if "decode_sample" in {k for k in traces}:
    d = traces["decode_sample"]
    cd = counts(d)
    ref = "design_chat" if "design_chat" in tags else tags[0]
    m["decode_vs_prefill"] = {"decode_coverage25": float(coverage(cd, .25).mean()),
                              "prefill_heldout_coverage25": float(coverage(counts(traces[f'{ref}_heldout']), .25).mean())}
else:
    m["decode_vs_prefill"] = "not measured (no decode trace for this model)"

# 5. churn: oldest half vs newer half of train
m["churn_top25"] = {}
for t in tags:
    tr = traces[f"{t}_train"]; h = len(tr) // 2
    a, b = topset(counts(tr[:h]), .25), topset(counts(tr[h:]), .25)
    m["churn_top25"][t] = float(np.mean([1 - len(x & y) / len(x) for x, y in zip(a, b)]))

# 6. speed estimate (planning only): Qwen3.5-122B-A10B, ~5 GB routed + 1.3 GB always-on bytes/token,
# experts that fit in ~40 GB VRAM ≈ 55% of routed experts; RAM 70 GB/s, VRAM 936 GB/s, 60% efficiency.
def tps(hit):
    t = 1.3 / 936 + 5 * hit / 936 + 5 * (1 - hit) / 70
    return 0.6 / t
c = "coding" if "coding" in m["hit_rate"] else next(iter(m["hit_rate"]))
S55 = min(sizes, key=lambda s: abs(s - 0.55 * E))
hr = m["hit_rate"][c]
m["speed_estimate_122B"] = {"note": "ESTIMATE, bandwidth model only; hit rates borrowed from this model",
                            "S_used": S55,
                            **{pol: {"hit": hr[pol][S55], "tok_s": tps(hr[pol][S55])}
                               for pol in ("static_profile", "lru", "profile_plus_lru", "ema_predictor", "session_predictor")},
                            "all_in_RAM_tok_s": tps(0.0)}

json.dump(m, open(OUT / "metrics.json", "w"), indent=1, default=float)

# plots
fig, ax = plt.subplots(figsize=(6, 4))
xs = np.linspace(0, 1, 41)[1:]
for t in tags:
    ax.plot(xs * 100, [coverage(C[t], x).mean() * 100 for x in xs], label=t)
ax.plot(xs * 100, xs * 100, "k:", label="uniform")
ax.set(xlabel="% hottest experts kept (per layer)", ylabel="% routings covered", title=f"Coverage — {name}")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "coverage.png", dpi=120)

fig, ax = plt.subplots(figsize=(6, 4))
for pol, d in hr.items():
    ax.plot([s / E * 100 for s in sizes], [d[s] * 100 for s in sizes], marker="o", label=pol)
ax.set(xlabel="cache size (% of experts per layer)", ylabel="held-out hit rate %", title=f"Hit rate vs cache size — {c}, {name}")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "hitrate.png", dpi=120)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(J, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(len(tags)), tags, rotation=45, ha="right"); ax.set_yticks(range(len(tags)), tags)
for i in range(len(tags)):
    for j in range(len(tags)):
        ax.text(j, i, f"{J[i,j]:.2f}", ha="center", va="center", color="w", fontsize=8)
ax.set_title("Top-25% expert overlap (Jaccard)"); fig.colorbar(im); fig.tight_layout(); fig.savefig(OUT / "overlap.png", dpi=120)

fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(range(L), coverage(C[c], .25) * 100); ax.axhline(25, color="k", ls=":")
ax.set(xlabel="layer", ylabel="coverage@25% (%)", title=f"Per-layer concentration — {c}")
fig.tight_layout(); fig.savefig(OUT / "per_layer.png", dpi=120)
print(json.dumps({k: m[k] for k in ("coverage", "overlap_mean_offdiag", "churn_top25")}, indent=1)[:2000])
