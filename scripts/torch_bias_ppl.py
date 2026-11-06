"""Cache-aware routing quality test (OLMoE, torch fallback).

For each cache size S (experts kept resident per layer, chosen by the coding
profile), add a penalty -lam to the router logits of non-resident experts and
measure held-out perplexity. lam=inf means "resident experts only".
Also measures the hit rate the bias achieves (share of routings landing on
resident experts).

usage: torch_bias_ppl.py <tag> <n_tokens>
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
prof = np.load(W / f"counts/OLMoE-1B-7B-0125-Instruct_{tag}.npy")  # [L, E] from train
L, E = prof.shape

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda").eval()
routers = [m for n, m in model.named_modules() if type(m).__name__.endswith("TopKRouter")]
state = {"bias": None, "hit": 0, "tot": 0}


def make_hook(l):
    def pre(m, args):
        return None
    def fwd(m, args, out):
        logits, scores, idx = out
        if state["bias"] is None:
            return out
        probs = F.softmax(logits.float(), dim=-1)          # weights stay the model's own
        idx = torch.topk(logits.float() + state["bias"][l], m.top_k, dim=-1).indices  # only selection is biased
        top = probs.gather(-1, idx)
        if getattr(m, "norm_topk_prob", False):
            top = top / top.sum(-1, keepdim=True)
        res = state["resident"][l]
        state["hit"] += int(res[idx].sum()); state["tot"] += idx.numel()
        return logits, top.to(scores.dtype), idx
    return fwd


for l, r in enumerate(routers):
    r.register_forward_hook(make_hook(l))

ids = tok((W / f"corpus/{tag}_heldout.txt").read_text(), return_tensors="pt").input_ids[0][:ntok]


@torch.no_grad()
def ppl():
    nll, n = 0.0, 0
    for s in range(0, len(ids) - 1, 1024):
        x = ids[s:s + 1024][None].cuda()
        lg = model(x).logits[0, :-1].float()
        nll += F.cross_entropy(lg, x[0, 1:], reduction="sum").item(); n += x.shape[1] - 1
    return float(np.exp(nll / n))


res = {"tag": tag, "tokens": int(len(ids)), "baseline_ppl": ppl(), "runs": []}
print("baseline", res["baseline_ppl"], flush=True)
for S in (E // 2, E // 4, E // 8):
    resident = torch.zeros(L, E, dtype=torch.bool, device="cuda")
    for l in range(L):
        resident[l, torch.as_tensor(np.argsort(-prof[l])[:S].copy())] = True
    state["resident"] = resident
    for lam in (0.5, 1.0, 2.0, float("inf")):
        state["bias"] = torch.where(resident, 0.0, -lam if lam != float("inf") else -1e9).float()
        state["hit"] = state["tot"] = 0
        p = ppl()
        run = {"S": S, "lambda": lam, "ppl": p, "ppl_increase_%": 100 * (p / res["baseline_ppl"] - 1),
               "hit_rate": state["hit"] / state["tot"]}
        res["runs"].append(run); print(run, flush=True)
    # unbiased hit rate at this S for reference
    state["bias"] = torch.zeros(L, E, device="cuda"); state["hit"] = state["tot"] = 0
    ppl(); res["runs"].append({"S": S, "lambda": 0.0, "ppl": res["baseline_ppl"], "ppl_increase_%": 0.0,
                               "hit_rate": state["hit"] / state["tot"]})
state["bias"] = None
out = W / "results/OLMoE-1B-7B-0125-Instruct" / f"bias_ppl_{tag}.json"
json.dump(res, open(out, "w"), indent=1)
