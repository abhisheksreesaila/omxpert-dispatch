"""Write an EDX_PIN profile for llama.cpp: int32 n_layer, int32 n_expert, float32 score[n_layer][n_expert].
Score = how often each (layer, expert) was routed in the given train traces.
usage: make_pin.py <model_dir> <out.bin> <tag_train> [<tag_train> ...]"""
import sys
from pathlib import Path
import numpy as np
W = Path.home() / "expert-dispatch"
name, out, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
tr = np.concatenate([np.load(W / f"traces/{name}/{t}.npz")["topk"].astype(np.int64) for t in tags])
T, L, K = tr.shape; E = int(tr.max()) + 1 if len(sys.argv) < 0 else max(int(tr.max()) + 1, int(__import__("os").environ.get("EDX_E", "0")))
score = np.stack([np.bincount(tr[:, l].ravel(), minlength=E) for l in range(L)]).astype(np.float32) / T
with open(out, "wb") as f:
    np.array([L, E], np.int32).tofile(f); score.tofile(f)
print(out, L, E, T, "tokens")
