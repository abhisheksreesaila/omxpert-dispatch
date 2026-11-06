"""Fallback tracer for MoE models MAX can't run (OLMoE). Prefill only.

Writes traces/<model>/<tag>_<split>.npz with topk int16 [n_tokens, n_layers, k]
(same layout the MAX hooks produce after concatenation).

usage: torch_trace.py <hf_model_id> <tag> <split> <max_tokens>
"""
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

W = Path.home() / "expert-dispatch"
mid, tag, split, max_tok = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
CTX = 2048

tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
k = model.config.num_experts_per_tok

routers = [m for n, m in model.named_modules() if type(m).__name__.endswith("TopKRouter")]
sink = []
for r in routers:
    r.register_forward_hook(lambda m, i, o: sink.append(o[2].to(torch.int16).cpu()))

ids = tok((W / f"corpus/{tag}_{split}.txt").read_text(), return_tensors="pt").input_ids[0][:max_tok]
out = []
with torch.no_grad():
    for s in range(0, len(ids), CTX):
        sink.clear()
        model(ids[s:s + CTX][None].cuda())
        out.append(torch.stack([x.reshape(-1, k) for x in sink], 1).numpy())  # [T, L, k]
topk = np.concatenate(out)
d = W / "traces" / mid.split("/")[-1]
d.mkdir(parents=True, exist_ok=True)
np.savez(d / f"{tag}_{split}.npz", topk=topk, chunk_id=np.arange(len(topk)) // CTX)
print(tag, split, topk.shape)
