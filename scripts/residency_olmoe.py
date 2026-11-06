"""Learnable residency (the spec's "learnable cache", OLMoE, torch): which S experts per layer should stay resident?

Compares three ways to pick the resident set, each evaluated with routing restricted to it (hard mask, zero misses):
  counts : how often each expert was routed on coding train (what we used so far)
  gate   : router-weighted frequency (sum of the top-k gate weights), the gate-value half of REAP
  learned: one trainable score per (layer, expert); during training the router sees logits + log sigmoid(g) for both
           selection and weights (so g gets a gradient), the loss is KL to the full model plus a penalty that keeps
           sum sigmoid(g) = S per layer; at eval the top-S scores per layer form the hard mask.

usage: residency_olmoe.py <S> <train_tokens> <steps>
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

W = Path.home() / "expert-dispatch"
MID = "allenai/OLMoE-1B-7B-0125-Instruct"
S, ntrain, steps_max = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
SEQ, NEVAL = 512, 40_000
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda").eval()
for p in model.parameters():
    p.requires_grad_(False)
routers = [m for n, m in model.named_modules() if type(m).__name__.endswith("TopKRouter")]
L = len(routers)
E = model.config.num_experts
state = {"mode": "full", "mask": None, "g": None, "count": None}


def router_hook(l):
    def fwd(m, args, out):
        logits, scores, idx = out
        mode = state["mode"]
        if mode == "count":   # gather routing statistics on the unrestricted model
            probs = F.softmax(logits.float(), -1)
            state["count"][0][l] += torch.bincount(idx.reshape(-1), minlength=E).float()
            state["count"][1][l].index_add_(0, idx.reshape(-1), probs.gather(-1, idx).reshape(-1))
            return out
        if mode == "full":
            return out
        if mode == "taylor":
            sc = state["s"][l][idx]
            return logits, (scores.float() * sc).to(scores.dtype) if scores.dtype != torch.float32 else scores * sc, idx
        if mode in ("mask", "maskn"):   # hard mask: only resident experts. mask: the model's raw weights; maskn: renormalised over residents
            z = logits.float() + torch.where(state["mask"][l], 0.0, -1e9)
        else:                 # "soft": learned residency scores enter selection and weights
            z = logits.float() + F.logsigmoid(state["g"][l])
        probs = F.softmax(logits.float(), -1) if mode == "mask" else F.softmax(z, -1)   # maskn and soft: renormalised
        idx = torch.topk(z, m.top_k, dim=-1).indices
        top = probs.gather(-1, idx)
        if getattr(m, "norm_topk_prob", False):
            top = top / top.sum(-1, keepdim=True)
        return logits, top.to(scores.dtype), idx
    return fwd


for l, r in enumerate(routers):
    r.register_forward_hook(router_hook(l))


def ids_of(name, n):
    return tok((W / f"corpus/{name}.txt").read_text(), return_tensors="pt").input_ids[0][:n]


@torch.no_grad()
def ppl(ids, mode, mask=None):
    state.update(mode=mode, mask=mask)
    nll, n = 0.0, 0
    for s in range(0, len(ids) - 1, 1024):
        x = ids[s:s + 1024][None].cuda()
        lg = model(x).logits[0, :-1].float()
        nll += F.cross_entropy(lg, x[0, 1:], reduction="sum").item(); n += x.shape[1] - 1
    return float(np.exp(nll / n))


def topS(score):
    m = torch.zeros(L, E, dtype=torch.bool, device="cuda")
    m.scatter_(1, torch.topk(score, S, dim=1).indices, True)
    return m


train = ids_of("coding_train", ntrain)
ev = {"coding": ids_of("coding_heldout", NEVAL), "design_chat": ids_of("design_chat_heldout", NEVAL)}

# routing statistics on the train split (same text the learner sees)
state["count"] = (torch.zeros(L, E, device="cuda"), torch.zeros(L, E, device="cuda"))
with torch.no_grad():
    state["mode"] = "count"
    for s in range(0, len(train) - 1, 1024):
        model(train[s:s + 1024][None].cuda())
counts, gatew = state["count"]
masks = {"counts": topS(counts), "gate": topS(gatew)}

# learned importance: first-order loss increase if an expert were removed (Taylor), from next-token loss on train text
state["s"] = torch.ones(L, E, device="cuda", requires_grad=True)
state["mode"] = "taylor"; imp = torch.zeros(L, E, device="cuda")
for s0 in range(0, len(train) - 1, 1024):
    x = train[s0:s0 + 1024][None].cuda()
    lossv = F.cross_entropy(model(x).logits[0, :-1].float(), x[0, 1:])
    gs, = torch.autograd.grad(lossv, state["s"])
    imp += -gs          # removing the expert moves s 1 -> 0, so the loss changes by about -dL/ds; big = important
masks["taylor"] = topS(imp)
if __import__("os").environ.get("SKIP_LEARN"):
    steps_max = 0

# learned residency: init from counts (top-S at +2, rest at -2), then train g only
g = torch.where(masks["counts"], 2.0, -2.0).float().requires_grad_(True)
state["g"] = g
opt = torch.optim.Adam([g], lr=float(__import__("os").environ.get("GLR", "0.05")))
chunks = [train[s:s + SEQ] for s in range(0, len(train) - SEQ, SEQ)]
mu = float(__import__("os").environ.get("MU", "1.0"))
t0, step, log = time.time(), 0, []
while step < steps_max and steps_max > 0:
    for j in torch.randperm(len(chunks)).tolist():
        x = chunks[j][None].cuda()
        with torch.no_grad():
            state["mode"] = "full"; teacher = F.log_softmax(model(x).logits.float(), -1)
        state["mode"] = "soft"
        student = F.log_softmax(model(x).logits.float(), -1)
        kl = F.kl_div(student, teacher, log_target=True, reduction="batchmean") / SEQ
        budget = ((torch.sigmoid(g).sum(1) - S) / S).pow(2).mean()
        sg = torch.sigmoid(g)
        binar = (sg * (1 - sg)).mean()                       # 0 when every gate is fully on or off
        nu = float(__import__("os").environ.get("NU", "8")) * min(1.0, step / (0.6 * steps_max))   # ramp up
        loss = kl + mu * budget + nu * binar
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        step += 1; log.append((kl.item(), budget.item()))
        if step % 50 == 0:
            a = np.mean(log[-50:], 0); print(f"step {step} kl {a[0]:.4f} budget {a[1]:.4f} binar {float(binar):.4f} on {int((torch.sigmoid(g)>.5).sum())} {time.time()-t0:.0f}s", flush=True)
        if step >= steps_max:
            break
if steps_max > 0:
    masks["learned"] = topS(g.detach())

res = {"S": S, "train_tokens": ntrain, "steps": step, "train_s": time.time() - t0,
       "overlap_learned_vs_counts": int((masks["learned"] & masks["counts"]).sum()) / (L * S) if "learned" in masks else None,
       "overlap_gate_vs_counts": int((masks["gate"] & masks["counts"]).sum()) / (L * S),
       "overlap_taylor_vs_counts": int((masks["taylor"] & masks["counts"]).sum()) / (L * S)}
for k, ids in ev.items():
    res[f"{k}_full"] = ppl(ids, "full")
    for name, m in masks.items():
        res[f"{k}_{name}"] = ppl(ids, "mask", m)

print(json.dumps(res), flush=True)
out = W / "results/OLMoE-1B-7B-0125-Instruct" / f"residency_S{S}_t{ntrain}.json"
json.dump(res, open(out, "w"), indent=1)
torch.save({"g": g.detach().cpu(), "counts": counts.cpu(), "gate": gatew.cpu(), "taylor": imp.cpu()}, out.with_suffix(".pt"))
