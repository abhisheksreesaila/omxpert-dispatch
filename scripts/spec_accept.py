"""Self-speculative decoding with resident experts (OLMoE, torch).

Draft = the same model with routing restricted/steered to the profile's top-S experts
per layer (the fast lane, all in VRAM). Target = the full router. Teacher-forced on
held-out text, the acceptance rate is how often the draft's greedy next token equals
the target's. Expected tokens per verify pass with draft length k: (1-a^(k+1))/(1-a).

usage: spec_accept.py <tag> <n_tokens> [profile_tag]
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

W = Path.home() / "expert-dispatch"
MID = "allenai/OLMoE-1B-7B-0125-Instruct"
tag, ntok = sys.argv[1], int(sys.argv[2])
ptag = sys.argv[3] if len(sys.argv) > 3 else tag
prof = np.load(W / f"counts/OLMoE-1B-7B-0125-Instruct_{ptag}.npy")
L, E = prof.shape
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda").eval()
routers = [m for n, m in model.named_modules() if type(m).__name__.endswith("TopKRouter")]
st = {"bias": None}


def hook(l):
    def fwd(m, args, out):
        logits, scores, idx = out
        if st["bias"] is None:
            return out
        probs = F.softmax(logits.float(), -1)
        idx = torch.topk(logits.float() + st["bias"][l], m.top_k, -1).indices
        top = probs.gather(-1, idx)
        if getattr(m, "norm_topk_prob", False):
            top = top / top.sum(-1, keepdim=True)
        return logits, top.to(scores.dtype), idx
    return fwd


for l, r in enumerate(routers):
    r.register_forward_hook(hook(l))
ids = tok((W / f"corpus/{tag}_heldout.txt").read_text(), return_tensors="pt").input_ids[0][:ntok]


@torch.no_grad()
def argmaxes():
    out = []
    for s in range(0, len(ids) - 1, 1024):
        x = ids[s:s + 1024][None].cuda()
        out.append(model(x).logits[0].argmax(-1).cpu())
    return torch.cat(out)


st["bias"] = None
full = argmaxes()
res = {"tag": tag, "profile": ptag, "tokens": int(len(ids)), "runs": []}
for S in (E // 2, E // 4, E // 8):
    resident = torch.zeros(L, E, dtype=torch.bool, device="cuda")
    for l in range(L):
        resident[l, torch.as_tensor(np.argsort(-prof[l])[:S].copy())] = True
    for lam in (1.0, 2.0, float("inf")):
        st["bias"] = torch.where(resident, 0.0, -lam if lam != float("inf") else -1e9).float()
        a = float((argmaxes() == full).float().mean())
        r = {"S": S, "lambda": lam, "accept": a,
             **{f"tok_per_verify_k{k}": (1 - a ** (k + 1)) / (1 - a) for k in (2, 4, 8)}}
        res["runs"].append(r); print(r, flush=True)
json.dump(res, open(W / f"results/OLMoE-1B-7B-0125-Instruct/spec_accept_{tag}_prof-{ptag}.json", "w"), indent=1)
