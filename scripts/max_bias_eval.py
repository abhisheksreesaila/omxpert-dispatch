"""Compare greedy continuations: unsteered vs steered (router bias) gpt-oss on MAX.
usage: max_bias_eval.py make_bias <S> <lam>          -> results/max_gptoss/bias_S<S>_l<lam>.npy
       max_bias_eval.py compare <base.json> <steered.json>
"""
import json, sys
from pathlib import Path
import numpy as np
W = Path.home() / "expert-dispatch"
if sys.argv[1] == "make_bias":
    S, lam = int(sys.argv[2]), float(sys.argv[3])
    c = np.load(W / "counts/max_gptoss_coding.npy")          # [L, E] from MAX traces
    b = np.full(c.shape, -lam, np.float32)
    for l in range(c.shape[0]):
        b[l, np.argsort(-c[l])[:S]] = 0.0
    out = W / f"results/max_gptoss/bias_S{S}_l{lam}.npy"; np.save(out, b); print(out)
else:
    a, b = (json.loads(Path(p).read_text()) for p in sys.argv[2:4])
    same_prefix, exact = [], 0
    for x, y in zip(a, b):
        n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        same_prefix.append(n / max(len(x), 1)); exact += x == y
    print(json.dumps({"prompts": len(a), "exact_match": exact / len(a),
                      "mean_shared_prefix_frac": float(np.mean(same_prefix))}))
