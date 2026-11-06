"""Steering-aware adapter (OLMoE, torch): can a tiny learned correction win back the quality lost by routing only
to resident experts?

Resident set: the coding profile's top-S experts per layer. Steering: -lam on non-resident router logits
(selection only; mixing weights stay the model's own). Adapter: per MoE block, a low-rank residual
out += up(down(h)), trained by distillation (KL to the unsteered full model) on coding train text.
Base model frozen. Evaluates held-out perplexity on coding (in-domain) and design_chat (out of domain).

usage: adapter_olmoe.py <S> <lam> <rank> <train_tokens> [steps]
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

W = Path.home() / "expert-dispatch"
MID = __import__("os").environ.get("MODEL", "allenai/OLMoE-1B-7B-0125-Instruct"); MNAME = MID.split("/")[-1]
S, lam, rank, ntrain = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
steps_max = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
SEQ, NEVAL = 512, 40_000
torch.manual_seed(0)

PROF = __import__("os").environ.get("PROFILE", "coding")
prof = np.load(W / f"counts/OLMoE-1B-7B-0125-Instruct_{PROF}.npy") if "OLMoE" in MID else None  # [L, E]; other models: counts from residency_olmoe.py
RES = __import__("os").environ.get("RES", "counts")   # counts: the profile npy; gate: router-weighted frequency saved by residency_olmoe.py
if prof is None and RES == "counts":
    prof = torch.load(W / "results" / MNAME / f"residency_S8_t120000.pt")["counts"].numpy()
if RES == "search":   # the mask learned by swapsearch_olmoe.py
    prof = torch.load(W / "results" / MNAME / f"swapsearch_S{S}.pt").float().numpy()
elif RES != "counts":
    prof = torch.load(W / "results" / MNAME / f"residency_S{S}_t120000.pt")[RES].numpy()
L, E = prof.shape
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda").eval()
for p in model.parameters():
    p.requires_grad_(False)
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
assert len(routers) == L and len(blocks) == L, (len(routers), len(blocks))
H = model.config.hidden_size

resident = torch.zeros(L, E, dtype=torch.bool, device="cuda")
for l in range(L):
    resident[l, torch.as_tensor(np.argsort(-prof[l])[:S].copy())] = True
bias = torch.where(resident, 0.0, -lam if math.isfinite(lam) else -1e9).float()
state = {"steer": False, "adapt": False, "hit": 0, "tot": 0}


def router_hook(l):
    def fwd(m, args, out):
        if not state["steer"]:
            return out
        logits, scores, idx = out
        idx = torch.topk(logits.float().detach() + bias[l], m.top_k, dim=-1).indices
        top = moe_weights(m, logits, idx, resident[l] if not math.isfinite(lam) else None)
        state["hit"] += int(resident[l][idx].sum()); state["tot"] += idx.numel()
        return logits, top.to(scores.dtype), idx
    return fwd


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(H, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, H, bias=False, dtype=torch.float32)
        nn.init.normal_(self.down.weight, std=H ** -0.5); nn.init.zeros_(self.up.weight)

    def forward(self, h):
        h = h.float()
        return self.up(self.down(h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)))


adapters = nn.ModuleList([Adapter() for _ in range(L)]).cuda()


def block_hook(l):
    def fwd(m, args, out):
        if not state["adapt"]:
            return out
        h = args[0]
        if isinstance(out, tuple):
            return (out[0] + adapters[l](h).to(out[0].dtype),) + tuple(out[1:])
        return out + adapters[l](h).to(out.dtype)
    return fwd


for l in range(L):
    routers[l].register_forward_hook(router_hook(l))
    blocks[l].register_forward_hook(block_hook(l))


def ids_of(name, n):
    return tok((W / f"corpus/{name}.txt").read_text(), return_tensors="pt").input_ids[0][:n]


@torch.no_grad()
def ppl(ids, steer, adapt):
    state.update(steer=steer, adapt=adapt, hit=0, tot=0)
    nll, n = 0.0, 0
    for s in range(0, len(ids) - 1, 1024):
        x = ids[s:s + 1024][None].cuda()
        lg = model(x).logits[0, :-1].float()
        nll += F.cross_entropy(lg, x[0, 1:], reduction="sum").item(); n += x.shape[1] - 1
    return float(np.exp(nll / n)), (state["hit"] / state["tot"] if state["tot"] else None)


ev = {"coding": ids_of("coding_heldout", NEVAL), "design_chat": ids_of("design_chat_heldout", NEVAL)}
res = {"profile": PROF, "residency": RES, "S": S, "lambda": lam, "rank": rank, "train_tokens": ntrain, "adapter_params": sum(p.numel() for p in adapters.parameters())}
for k, ids in ev.items():
    res[f"{k}_full"] = ppl(ids, False, False)[0]
    res[f"{k}_steer"], res[f"{k}_hit"] = ppl(ids, True, False)
print(json.dumps(res), flush=True)

train = ids_of(f"{PROF}_train", ntrain)
chunks = [train[s:s + SEQ] for s in range(0, len(train) - SEQ, SEQ)]
opt = torch.optim.AdamW(adapters.parameters(), lr=float(__import__('os').environ.get('LR', '3e-4')), weight_decay=0.0)
t0, step, losses = time.time(), 0, []
for epoch in range(2):
    for j in torch.randperm(len(chunks)).tolist():
        x = chunks[j][None].cuda()
        with torch.no_grad():
            state.update(steer=False, adapt=False)
            teacher = F.log_softmax(model(x).logits.float(), -1)
        state.update(steer=True, adapt=True)
        student = F.log_softmax(model(x).logits.float(), -1)
        loss = F.kl_div(student, teacher, log_target=True, reduction="batchmean") / SEQ
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(adapters.parameters(), 1.0); opt.step()
        losses.append(loss.item()); step += 1
        if step % 50 == 0 or step in (1, 5, 10, 20):
            print(f"step {step} kl {np.mean(losses[-50:]):.4f} {time.time()-t0:.0f}s", flush=True)
        if step >= steps_max:
            break
    if step >= steps_max:
        break
res["train_steps"], res["train_s"] = step, time.time() - t0
for k, ids in ev.items():
    res[f"{k}_steer_adapter"] = ppl(ids, True, True)[0]
    res[f"{k}_full_adapter"] = ppl(ids, False, True)[0]   # adapter without steering: should stay near full if it only corrects steering
    with torch.no_grad():   # held-out KL to the full model, per token (what training minimised)
        x = ids[:1024][None].cuda(); state.update(steer=False, adapt=False); tl = F.log_softmax(model(x).logits.float(), -1)
        for tag, st, ad in (("steer", True, False), ("steer_adapter", True, True)):
            state.update(steer=st, adapt=ad); sl = F.log_softmax(model(x).logits.float(), -1)
            res[f"{k}_kl_{tag}"] = float(F.kl_div(sl, tl, log_target=True, reduction="sum") / x.shape[1])
    gap = res[f"{k}_steer"] - res[f"{k}_full"]
    res[f"{k}_gap_closed_%"] = 100 * (res[f"{k}_steer"] - res[f"{k}_steer_adapter"]) / gap if gap > 0 else None
print(json.dumps(res), flush=True)
out = W / "results" / MNAME / f"adapter_{PROF}_{RES}_S{S}_l{lam}_r{rank}_t{ntrain}.json"
json.dump(res, open(out, "w"), indent=1)
torch.save(adapters.state_dict(), out.with_suffix(".pt"))
