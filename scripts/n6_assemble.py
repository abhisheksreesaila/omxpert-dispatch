"""Night 6: assemble combined.json (every trick: measured or not built), quality.json (greedy outputs vs stock llama.cpp) from the runs.
usage: n6_assemble.py"""
import json, statistics
from pathlib import Path

R = Path.home() / "expert-dispatch/results/night6"


def ld(n):
    p = R / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def arm(d, a):
    return d["arms"][a] if d and "arms" in d and a in d["arms"] else None


T = {}
# FreeToken's engine, as is, and its idea ported into llama.cpp
fs, base = ld("ft_stock"), ld("baseline"); ov = ld("ab_overlap")
T["freetoken_hybrid"] = {"status": "measured", "from": "FreeToken (FlashML), CPU computes misses while the GPU computes hits",
    "engine_as_is": {"freetoken_kai_tok_s": fs and fs["decode_tok_s"], "freetoken_prefill": fs and fs["prefill_tok_s"],
                     "llamacpp_tok_s": base and base["decode_tok_s"], "llamacpp_prefill": base and base["prefill_tok_s"],
                     "why": "64 GB desktop: ~25 GB of RAM free for its expert bank, VRAM cache can only hold RAM-resident experts, most picks fault in from NVMe"},
    "ported_to_llamacpp": {"on": arm(ov, "on") and arm(ov, "on")["median_tok_s"], "off": arm(ov, "off") and arm(ov, "off")["median_tok_s"]},
    "paired": arm(ov, "on") and arm(ov, "on")["paired_ratios"], "note": "in-process A/B, order alternated; round 0 is the cold first round"}
fits = [ld(f"ab_fit_{i}") for i in (1, 2)]; pins = [ld(f"ab_pin_{i}") for i in (1, 2)]
fit_t = [arm(f, "base")["median_tok_s"] for f in fits if arm(f, "base")]
pin_t = [arm(p, "pfoff")["median_tok_s"] for p in pins if arm(p, "pfoff")]
fit_p = [arm(f, "base")["prefill_median"] for f in fits if arm(f, "base")]
pin_p = [arm(p, "pfoff")["prefill_median"] for p in pins if arm(p, "pfoff")]
T["profile_residency"] = {"status": "measured" if fit_t and pin_t else "not_built", "from": "omxpert: profile from the owner's chat history picks the VRAM experts",
    "vs": {"fit_tok_s": fit_t, "pinned_stack_tok_s": pin_t, "fit_prefill": fit_p, "pinned_prefill": pin_p},
    "reason": None if fit_t and pin_t else "runs missing",
    "note": "separate servers interleaved fit, pinned, fit, pinned; same prompts (held-out chat sessions) and server timings"}
pf = [arm(p, "pfprev") for p in pins if arm(p, "pfprev")]
T["layer_ahead_prefetch"] = {"status": "measured" if pf else "not_built", "from": "0xSero (layer-ahead), our CPU-side reader",
    "paired": [r for a in pf for r in a["paired_ratios"]], "vs": {"pfprev": [a["median_tok_s"] for a in pf], "pfoff": pin_t},
    "reason": None if pf else "runs missing", "note": "previous token's cold picks of the next layer, madvise WILLNEED by 8 reader threads"}
T["predictor_prefetch"] = {"status": "not_built", "from": "omxpert learned guesser + Fate's cross-layer gate idea",
    "offline": ld("predictor_chat"),
    "reason": "learned table guesses pinned experts too and would pull their RAM back (pinned mode pages them out); needs a pin filter in ggml-cpu. Offline on chat: 44.5% of next-layer picks with 16 guesses (49.1% with recency). Cross-layer gate needs hidden states in the CPU op (not built)."}
T["read_pool"] = {"status": "measured" if pf else "not_built", "from": "0xSero", "paired": T["layer_ahead_prefetch"]["paired"],
    "reason": None if pf else "runs missing", "note": "the reader pool only does work when prefetch is on: same arm as layer_ahead_prefetch"}
mt = ld("ab_pinmtp")
T["mtp"] = {"status": "measured" if arm(mt, "mtp") else "not_built", "from": "Qwen / llama.cpp",
    "paired": arm(mt, "mtp") and arm(mt, "mtp")["paired_ratios"],
    "vs": {"mtp": arm(mt, "mtp") and arm(mt, "mtp")["median_tok_s"], "nomtp": arm(mt, "nomtp") and arm(mt, "nomtp")["median_tok_s"],
           "mtp_prefill": arm(mt, "mtp") and arm(mt, "mtp")["prefill_median"]},
    "reason": None if arm(mt, "mtp") else "MTP run failed or missing (see results/night6/llama_ab_pinmtp.log)", "note": "pinned stack with 33 GB of pins (the draft head takes VRAM)"}
T["staged_prefill"] = {"status": "not_built", "from": "0xSero",
    "reason": "night 5 measured no gain on ~550-token prompts (22.9/149 vs 22.6/150); tonight's prompt fix is per-layer placement (EDX_PREFILL_LOCAL) instead"}
pc, px, pxl = ld("pin_chat"), ld("pinx_chat"), ld("pinxol_chat")
T["exclusive_tiers"] = {"status": "measured" if pc and px else "not_built", "from": "0xSero (exclusive VRAM/RAM tiers) + fixes tonight",
    "vs": {"night5_code_tok_s": pc and pc["decode_tok_s"], "night5_code_prefill": pc and pc["prefill_tok_s"], "night5_code_flt": pc and pc.get("majflt_per_token"),
           "xdev_tok_s": px and px["decode_tok_s"], "xdev_prefill": px and px["prefill_tok_s"], "xdev_flt": px and px.get("majflt_per_token"),
           "xdev_overlap_local_tok_s": pxl and pxl["decode_tok_s"], "xdev_overlap_local_prefill": pxl and pxl["prefill_tok_s"], "xdev_overlap_local_flt": pxl and pxl.get("majflt_per_token")},
    "reason": None if pc and px else "runs missing", "note": "separate runs, chat prompts, client timings; pinned experts' RAM is given back (EDX_PAGEOUT)"}
qs, qp = ld("q_pinsteer"), ld("q_pin")
T["steering"] = {"status": "measured" if arm(qs, "base") and arm(qp, "base") else "not_built", "from": "omxpert (swap a pick for a VRAM expert; Skliar et al.'s cache-conditional routing)",
    "vs": {"steer05_tok_s": arm(qs, "base") and arm(qs, "base")["median_tok_s"], "pinned_tok_s": arm(qp, "base") and arm(qp, "base")["median_tok_s"]},
    "reason": None if arm(qs, "base") else "run missing", "note": "lossy: quality in quality.json"}
json.dump({"tricks": T}, open(R / "combined.json", "w"), indent=1)

# quality: greedy coding outputs vs stock llama.cpp (q_fit_a); floor = q_fit_b (same settings, separate process)
def texts(n):
    d = ld(n)
    return d["texts"]["base"] if d and "texts" in d and "base" in d["texts"] else None


def cmp(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    wa, wb = a.split(), b.split()
    return {"same_prefix_chars": n, "len": [len(a), len(b)], "word_match": round(sum(x == y for x, y in zip(wa, wb)) / max(1, max(len(wa), len(wb))), 3)}


ref = texts("q_fit_a"); q = {"reference": "stock llama.cpp fit, -lzm off (q_fit_a), 3 held-out coding prompts x 128 greedy tokens"}
if ref:
    def sc(n):
        t = texts(n)
        if not t:
            return None
        rows = {k: cmp(ref[k], t[k]) for k in ref if k in t}
        return {"per_prompt": rows, "word_match": round(statistics.mean(r["word_match"] for r in rows.values()), 3)}
    def sc_bench(n):   # n6_bench files keep texts as {"0": ...}
        d = ld(n)
        if not d or "texts" not in d:
            return None
        rows = {k: cmp(ref[k], d["texts"][k]) for k in ref if k in d["texts"]}
        return {"per_prompt": rows, "word_match": round(statistics.mean(r["word_match"] for r in rows.values()), 3)}
    q["floor_exact"] = sc("q_fit_b")            # stock vs stock, separate process: identical settings
    q["floor"] = sc_bench("ft_stock")           # another engine on the same GGUF (FreeToken): same model, different arithmetic
    q["floor_note"] = "greedy text diverges at the first close call under any change of arithmetic; FreeToken is the same model with other kernels"
    q["pinned_stack"] = sc("q_pin"); q["pinned_steer05"] = sc("q_pinsteer")
    q["token_match"] = {k: (q[k] or {}).get("word_match") for k in ("floor_exact", "floor", "pinned_stack", "pinned_steer05")}
    fl = (q["floor"] or {}).get("word_match") or 0; pm = (q["pinned_stack"] or {}).get("word_match") or 0
    q["verdict"] = ("not bit-exact; diverges from stock about as much as another engine on the same model does (weak test: 3 prompts x 128 tokens)"
                    if pm >= fl - 0.05 else "diverges from stock more than another engine on the same model does (weak test: 3 prompts x 128 tokens)")
json.dump(q, open(R / "quality.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("status", "vs", "paired")} for k, v in T.items()}, indent=1)[:3000])
print(json.dumps(q.get("token_match")), q.get("verdict"))
