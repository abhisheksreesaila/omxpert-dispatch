"""Readable scorecard for tonight's bundles (OLMoE): next-token top-1 / top-5 accuracy and perplexity on held-out
coding and chat, for the full model and for each resident set (hard mask), with and without its adapter.
Agreement = how often the bundle's top-1 token equals the full model's top-1 token.
usage: eval_bundles.py
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

W = Path.home() / "expert-dispatch"
MID = __import__("os").environ.get("MODEL", "allenai/OLMoE-1B-7B-0125-Instruct"); MNAME = MID.split("/")[-1]; NEVAL = 40_000
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

blocks = [m for n, m in model.named_modules() if (type(m).__name__.endswith("SparseMoeBlock") or type(m).__name__ == "GraniteMoeMoE")]
L, E, H = len(routers), getattr(model.config, "num_experts", None) or model.config.num_local_experts, model.config.hidden_size
state = {"mask": None, "ad": None}


class Adapter(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.down = nn.Linear(H, rank, bias=False); self.up = nn.Linear(rank, H, bias=False)

    def forward(self, h):
        h = h.float(); return self.up(self.down(h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)))


def rhook(l):
    def f(m, a, out):
        if state["mask"] is None:
            return out
        logits, scores, idx = out
        probs = F.softmax(logits.float(), -1)
        idx = torch.topk(logits.float() + torch.where(state["mask"][l], 0.0, -1e9), m.top_k, dim=-1).indices
        return logits, moe_weights(m, logits, idx, state["mask"][l]).to(scores.dtype), idx
    return f


def bhook(l):
    def f(m, a, out):
        if state["ad"] is None:
            return out
        add = state["ad"][l](a[0])
        return (out[0] + add.to(out[0].dtype),) + tuple(out[1:]) if isinstance(out, tuple) else out + add.to(out.dtype)
    return f


for l in range(L):
    routers[l].register_forward_hook(rhook(l)); blocks[l].register_forward_hook(bhook(l))
ids_of = lambda n: tok((W / f"corpus/{n}.txt").read_text(), return_tensors="pt").input_ids[0][:NEVAL]
ev = {"coding": ids_of("coding_heldout"), "chat": ids_of("design_chat_heldout")}
full_top1 = {}


@torch.no_grad()
def score(k, ids):
    nll = n = c1 = c5 = agree = 0; tops = []
    for s in range(0, len(ids) - 1, 1024):
        x = ids[s:s + 1024][None].cuda(); y = x[0, 1:]
        lg = model(x).logits[0, :-1].float()
        nll += F.cross_entropy(lg, y, reduction="sum").item(); n += len(y)
        t5 = lg.topk(5, -1).indices; c1 += (t5[:, 0] == y).sum().item(); c5 += (t5 == y[:, None]).any(-1).sum().item()
        tops.append(t5[:, 0])
    tops = torch.cat(tops)
    if k not in full_top1:
        full_top1[k] = tops
    return {"ppl": float(np.exp(nll / n)), "top1": c1 / n, "top5": c5 / n, "agree_full": (tops == full_top1[k]).float().mean().item()}


def mask_from(score_or_mask, S):
    t = torch.as_tensor(score_or_mask).float().cuda()
    m = torch.zeros(L, E, dtype=torch.bool, device="cuda"); m.scatter_(1, torch.topk(t, S, 1).indices, True); return m


def adapter(path, rank=32):
    ad = nn.ModuleList([Adapter(rank) for _ in range(L)]).cuda(); ad.load_state_dict(torch.load(path)); return ad


R = W / "results" / MNAME
counts = np.load(W / "counts/OLMoE-1B-7B-0125-Instruct_coding.npy") if "OLMoE" in MID else torch.load(R / "residency_S8_t120000.pt")["counts"].numpy()
configs = [("full model", None, None)]
for S in (16, 8, 4):
    configs.append((f"{S}/{E} counted", mask_from(counts, S), None))
    configs.append((f"{S}/{E} learned set", mask_from(torch.load(R / f"swapsearch_S{S}.pt"), S), None))
    ad = {16: "adapter_coding_search_S16_linf_r32_t240000.pt", 8: "adapter_coding_search_S8_linf_r32_t480000.pt", 4: "adapter_coding_search_S4_linf_r32_t480000.pt"}[S]
    configs.append((f"{S}/{E} learned set + adapter", mask_from(torch.load(R / f"swapsearch_S{S}.pt"), S), adapter(R / ad)))
rows = []
for name, m, ad in configs:
    state.update(mask=m, ad=ad)
    row = {"config": name}
    for k, ids in ev.items():
        for kk, v in score(k, ids).items():
            row[f"{k}_{kk}"] = v
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(R / "bundles_scorecard.json", "w"), indent=1)
