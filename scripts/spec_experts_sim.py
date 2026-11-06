"""Idea 26, speculative experts: predict layer l+1's experts from layer l's router input.
Prediction: top-m of  W_{l+1} . h_l  (the next layer's router applied to this layer's attn_post_norm).
Sanity: W_{l+1} . h_{l+1} must reproduce the real top-8. Baselines: previous token's experts at layer l+1,
and the static profile's top-m. Also: for a fixed pinned set, how many speculative uploads are useful."""
import json
from pathlib import Path
import numpy as np
from gguf import GGUFReader
W = Path.home() / "expert-dispatch"; O = W / "results/night2"
ids = np.fromfile("/tmp/edx_spec_ids.bin", np.int32).reshape(-1, 40, 8)
T = ids.shape[0]; L = 40; D = 2048; E = 256
H = np.fromfile(O / "hidden_coding.f16", np.float16)[: T * L * D].reshape(T, L, D).astype(np.float32)
r = GGUFReader(str(Path.home() / "models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"))
Wr = {}
for t in r.tensors:
    if t.name.endswith("ffn_gate_inp.weight"):
        Wr[int(t.name.split(".")[1])] = np.array(t.data, dtype=np.float32).reshape(E, D)
tr = np.load(W / "traces/Qwen3.6-35B-A3B/coding_train.npz")["topk"].astype(np.int64)
cnt = np.stack([np.bincount(tr[:, l].ravel(), minlength=E) for l in range(L)]).astype(float)
def recall(pred, actual):  # pred [T, m], actual [T, 8]
    return float(np.mean([len(set(p) & set(a)) / 8 for p, a in zip(pred, actual)]))
res = {"tokens": int(T)}
sanity = np.mean([recall(np.argsort(-(H[:, l] @ Wr[l].T), 1)[:, :8], ids[:, l]) for l in range(L)])
res["sanity_same_layer"] = float(sanity)
for m in (8, 12, 16, 24):
    xl = np.mean([recall(np.argsort(-(H[:, l] @ Wr[l + 1].T), 1)[:, :m], ids[:, l + 1]) for l in range(L - 1)])
    prev = np.mean([recall(np.concatenate([ids[:-1, l + 1], np.argsort(-cnt[l + 1])[None, :max(0, m - 8)].repeat(T - 1, 0)], 1)[:, :m], ids[1:, l + 1]) for l in range(L - 1)])
    prof = np.mean([recall(np.argsort(-cnt[l + 1])[None, :m].repeat(T, 0), ids[:, l + 1]) for l in range(L - 1)])
    res[f"recall@{m}"] = {"next_router_on_this_hidden": float(xl), "previous_token": float(prev), "static_profile": float(prof)}
# usefulness of speculative uploads on top of a 15% frequency pin set
pins = np.zeros((L, E), bool)
for l in range(L): pins[l, np.argsort(-cnt[l])[:38]] = True
for m in (8, 12, 16):
    useful = up = covered = missed = 0
    for l in range(L - 1):
        pred = np.argsort(-(H[:, l] @ Wr[l + 1].T), 1)[:, :m]
        for p, a in zip(pred, ids[:, l + 1]):
            cold_pred = [e for e in p if not pins[l + 1, e]]; cold_act = [e for e in a if not pins[l + 1, e]]
            up += len(cold_pred); useful += len(set(cold_pred) & set(cold_act)); covered += len(set(cold_pred) & set(cold_act)); missed += len(cold_act)
    res[f"spec_upload_top{m}"] = {"uploads_per_layer_token": up / (T * (L - 1)), "useful_fraction": useful / max(up, 1),
                                  "misses_covered": covered / max(missed, 1)}
json.dump(res, open(O / "spec_experts.json", "w"), indent=1); print(json.dumps(res, indent=1))
