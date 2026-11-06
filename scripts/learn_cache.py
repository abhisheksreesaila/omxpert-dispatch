"""Learnable expert cache (OLMoE, frozen model).

Learns a router bias B[L, E] under a fixed VRAM budget of S resident experts per
layer. Objective on the owner's train split:
    loss = NLL + mu * expected non-resident routing mass
Selection uses top-k of (logits + B); mixing weights use softmax(logits + B), so
NLL gradients reach B (straight-through on the discrete top-k).
Every `refresh` steps the resident set is re-picked as the S experts the *biased*
router uses most (alternating route <-> residency, like EM).

Compared on held-out against the hand-set bias (lambda=1 on non-resident).

usage: learn_cache.py <S> <mu> [train_tokens] [steps]
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

W = Path.home() / "expert-dispatch"
MID = "allenai/OLMoE-1B-7B-0125-Instruct"
S, mu = int(sys.argv[1]), float(sys.argv[2])
ntrain = int(sys.argv[3]) if len(sys.argv) > 3 else 120_000
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 200
SEQ, REFRESH = 512, 50

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda").eval()
for p in model.parameters():
    p.requires_grad_(False)
model.gradient_checkpointing_enable()
routers = [m for n, m in model.named_modules() if type(m).__name__.endswith("TopKRouter")]
L, E = len(routers), routers[0].weight.shape[0]
prof = np.load(W / "counts/OLMoE-1B-7B-0125-Instruct_coding.npy")

B = torch.zeros(L, E, device="cuda", requires_grad=True)
resident = torch.zeros(L, E, dtype=torch.bool, device="cuda")
for l in range(L):
    resident[l, torch.as_tensor(np.argsort(-prof[l])[:S].copy())] = True
st = {"mode": "learn", "miss": [], "use": torch.zeros(L, E, device="cuda"), "hit": 0, "tot": 0, "fixed": None}


def hook(l):
    def fwd(m, args, out):
        logits, scores, idx = out
        lg = logits.float()
        if st["mode"] == "base":
            return out
        bias = B[l] if st["mode"] == "learn" else st["fixed"][l]
        z = lg + bias
        idx = torch.topk(z.detach(), m.top_k, dim=-1).indices
        if st["mode"] == "learn":
            p = F.softmax(z, -1)
            top = p.gather(-1, idx)                                   # grads reach B
            st["miss"].append((p * (~resident[l]).float()).sum(-1).mean())
            st["use"][l] += torch.bincount(idx.flatten(), minlength=E).float()
        else:
            top = F.softmax(lg, -1).gather(-1, idx)                   # eval: model's own weights
        st["hit"] += int(resident[l][idx].sum()); st["tot"] += idx.numel()
        return logits, top.to(scores.dtype), idx
    return fwd


for l, r in enumerate(routers):
    r.register_forward_hook(hook(l))

tr = tok((W / "corpus/coding_train.txt").read_text()[-ntrain * 5:], return_tensors="pt").input_ids[0][-ntrain:]
ho = tok((W / "corpus/coding_heldout.txt").read_text(), return_tensors="pt").input_ids[0][:40_000]


@torch.no_grad()
def evaluate(mode, fixed=None):
    st.update(mode=mode, fixed=fixed, hit=0, tot=0)
    nll = n = 0
    for s in range(0, len(ho) - 1, 1024):
        x = ho[s:s + 1024][None].cuda()
        lg = model(x).logits[0, :-1].float()
        nll += F.cross_entropy(lg, x[0, 1:], reduction="sum").item(); n += x.shape[1] - 1
    return {"ppl": float(np.exp(nll / n)), "hit": st["hit"] / max(st["tot"], 1)}


res = {"S": S, "E": E, "mu": mu, "steps": steps}
res["baseline"] = evaluate("base")
hand = torch.where(resident, 0.0, -1.0)
res["hand_lambda1"] = evaluate("fixed", hand)
print(res, flush=True)

opt = torch.optim.Adam([B], lr=0.05)
starts = torch.randint(0, len(tr) - SEQ - 1, (steps,))
model.train()  # needed for checkpointing; OLMoE has no dropout
for step in range(steps):
    st.update(mode="learn", miss=[])
    x = tr[starts[step]:starts[step] + SEQ][None].cuda()
    out = model(x, labels=x)
    miss = torch.stack(st["miss"]).mean()
    loss = out.loss + mu * miss
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 0:
        print(f"step {step} nll {out.loss.item():.3f} miss_mass {miss.item():.3f}", flush=True)
    if (step + 1) % REFRESH == 0:  # re-pick resident set from biased routing
        with torch.no_grad():
            for l in range(L):
                resident[l] = False
                resident[l, torch.topk(st["use"][l], S).indices] = True
            st["use"].zero_()
model.eval()
res["learned"] = evaluate("fixed", B.detach())
res["hand_lambda1_after_reselect"] = evaluate("fixed", torch.where(resident, 0.0, -1.0))
out = W / f"results/OLMoE-1B-7B-0125-Instruct/learned_cache_S{S}_mu{mu}.json"
json.dump(res, open(out, "w"), indent=1)
torch.save({"B": B.detach().cpu(), "resident": resident.cpu()}, out.with_suffix(".pt"))
print(json.dumps(res), flush=True)
