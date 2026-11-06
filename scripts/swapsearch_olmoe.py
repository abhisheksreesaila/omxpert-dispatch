"""Learn the resident set by search (OLMoE): start from the gate-weighted set, then per layer try swapping the weakest
residents for the strongest outsiders; keep a swap only if KL to the full model drops on a train slice (hard mask).
Honest and non-differentiable: every decision is scored by a real forward pass. Evaluated on held-out text after.

usage: swapsearch_olmoe.py <S> <rounds> <cand>
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

W = Path.home() / "expert-dispatch"
MID = __import__("os").environ.get("MODEL", "allenai/OLMoE-1B-7B-0125-Instruct"); MNAME = MID.split("/")[-1]
S, rounds, cand = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
NSLICE, NEVAL = 4096, 40_000

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda").eval()
routers = [m for n, m in model.named_modules() if type(m).__name__.endswith("TopKRouter")]
def moe_weights(m, logits, idx, keep=None):
    """Mixing weights for the chosen experts, the way this model computes them. keep: bool [E] of loaded experts;
    a chosen expert that is not loaded gets weight 0 (it only happens when fewer experts are loaded than top-k)."""
    if type(m).__name__.startswith("GraniteMoe"):          # softmax over the chosen logits only
        z = logits.float().gather(-1, idx)
        if keep is not None:
            z = z.masked_fill(~keep[idx], float("-inf"))
        return F.softmax(z, -1)
    top = F.softmax(logits.float(), -1).gather(-1, idx)    # OLMoE: full-softmax probabilities
    if getattr(m, "norm_topk_prob", False):
        top = top / top.sum(-1, keepdim=True)
    return top * keep[idx] if keep is not None else top

L, E = len(routers), getattr(model.config, "num_experts", None) or model.config.num_local_experts
state = {"mask": None}


def hook(l):
    def fwd(m, args, out):
        if state["mask"] is None:
            return out
        logits, scores, idx = out
        idx = torch.topk(logits.float() + torch.where(state["mask"][l], 0.0, -1e9), m.top_k, dim=-1).indices
        top = moe_weights(m, logits, idx, state["mask"][l])
        return logits, top.to(scores.dtype), idx
    return fwd


for l, r in enumerate(routers):
    r.register_forward_hook(hook(l))
import os, torch.nn as nn
if os.environ.get("ADAPTER"):
    H = model.config.hidden_size
    class Adapter(nn.Module):
        def __init__(self):
            super().__init__(); self.down = nn.Linear(H, 32, bias=False); self.up = nn.Linear(32, H, bias=False)
        def forward(self, h):
            h = h.float(); return self.up(self.down(h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)))
    ads = nn.ModuleList([Adapter() for _ in range(L)]).cuda(); ads.load_state_dict(torch.load(os.environ["ADAPTER"]))
    blocks = [m for n, m in model.named_modules() if (type(m).__name__.endswith("SparseMoeBlock") or type(m).__name__ == "GraniteMoeMoE")]
    def bh(l):
        def f(m, a, out):
            if state["mask"] is None:
                return out
            add = ads[l](a[0]); return (out[0] + add.to(out[0].dtype),) + tuple(out[1:]) if isinstance(out, tuple) else out + add.to(out.dtype)
        return f
    for l, b_ in enumerate(blocks):
        b_.register_forward_hook(bh(l))
ids_of = lambda name, n: tok((W / f"corpus/{name}.txt").read_text(), return_tensors="pt").input_ids[0][:n]
slice_ = ids_of("coding_train", 200_000)[-NSLICE:]          # a slice the counts were not built from
xs = [slice_[s:s + 1024][None].cuda() for s in range(0, NSLICE, 1024)]
with torch.no_grad():
    state["mask"] = None
    teach = [F.log_softmax(model(x).logits.float(), -1) for x in xs]


@torch.no_grad()
def kl(mask):
    state["mask"] = mask
    return sum(float(F.kl_div(F.log_softmax(model(x).logits.float(), -1), t, log_target=True, reduction="sum")) for x, t in zip(xs, teach)) / NSLICE


@torch.no_grad()
def ppl(ids, mask):
    state["mask"] = mask
    nll, n = 0.0, 0
    for s in range(0, len(ids) - 1, 1024):
        x = ids[s:s + 1024][None].cuda()
        nll += F.cross_entropy(model(x).logits[0, :-1].float(), x[0, 1:], reduction="sum").item(); n += x.shape[1] - 1
    return float(np.exp(nll / n))


gate = torch.load(W / "results" / MNAME / f"residency_S{S}_t120000.pt")["gate"].cuda()
order = torch.argsort(gate, dim=1, descending=True)
mask = torch.zeros(L, E, dtype=torch.bool, device="cuda"); mask.scatter_(1, order[:, :S], True)
if os.environ.get("START"):
    mask = torch.load(os.environ["START"]).cuda()
start = mask.clone()
best = kl(mask); t0 = time.time(); kept = 0
print(f"start KL {best:.4f}", flush=True)
for r in range(rounds):
    for l in range(L):
        res_ = [e for e in order[l].tolist() if mask[l, e]][::-1][:cand]          # weakest residents by gate score
        out_ = [e for e in order[l].tolist() if not mask[l, e]][:cand]            # strongest outsiders
        for a in res_:
            for b in out_:
                if not mask[l, a] or mask[l, b]:
                    continue
                m2 = mask.clone(); m2[l, a] = False; m2[l, b] = True
                k = kl(m2)
                if k < best - 1e-4:
                    best, mask, kept = k, m2, kept + 1
                    break
    print(f"round {r+1} KL {best:.4f} swaps kept {kept} {time.time()-t0:.0f}s", flush=True)
ev = {"coding": ids_of("coding_heldout", NEVAL), "design_chat": ids_of("design_chat_heldout", NEVAL)}
res = {"S": S, "rounds": rounds, "cand": cand, "swaps_kept": kept, "search_s": time.time() - t0, "slice_kl_start": None, "slice_kl_end": best,
       "changed_share": int((mask & ~start).sum()) / (L * S)}
for k, ids in ev.items():
    res[f"{k}_full"] = ppl(ids, None); res[f"{k}_gate"] = ppl(ids, start); res[f"{k}_search"] = ppl(ids, mask)
print(json.dumps(res), flush=True)
out = W / "results" / MNAME / f"swapsearch_S{S}{os.environ.get('TAG', '')}.json"
json.dump(res, open(out, "w"), indent=1); torch.save(mask.cpu(), out.with_suffix(".pt"))
