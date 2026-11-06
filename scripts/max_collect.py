"""Merge MAX hook shards traces/max/<tag>/*.npz into traces/max_gptoss/<tag>.npz, topk [T, L, k]."""
from pathlib import Path
import numpy as np
W = Path.home() / "expert-dispatch"
L, K = 24, 4  # gpt-oss-20b: 24 MoE layers, top-4 of 32
out = W / "traces/max_gptoss"; out.mkdir(parents=True, exist_ok=True)
for d in sorted(p for p in (W / "traces/max").iterdir() if p.is_dir() and p.name not in ("smoke", "untagged")):
    parts = [np.load(f)["topk"] for f in sorted(d.glob("*.npz"))]
    if not parts:
        continue
    topk = np.concatenate(parts).reshape(-1, L, K).astype(np.int16)
    np.savez(out / f"{d.name}.npz", topk=topk)
    print(d.name, topk.shape)
