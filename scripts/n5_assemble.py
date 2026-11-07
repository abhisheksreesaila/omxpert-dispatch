"""Night 5: gather the run files into the milestone files (baseline, exact, learned, stack). Numbers only come from runs;
a trick that was not run says so and why."""
import json, re
from pathlib import Path

W = Path.home() / "expert-dispatch"; R = W / "results/night5"
def j(n):
    p = R / f"{n}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return None if "skipped" in d else d


def kl(name):
    p = R / f"kl_{name}.log"
    if not p.exists():
        return None
    t = p.read_text(errors="replace")
    m = re.search(r"Mean\s+KLD:\s+([-\d.]+)", t); s = re.search(r"Same top p:\s+([\d.]+)", t)
    return {"kl": float(m.group(1)), "same_top_pct": float(s.group(1)) if s else None} if m else None


def row(name, **extra):
    r = j(name)
    if not r:
        return {"status": "missing_run", "run": name, **extra}
    out = {"status": "measured", "run": name, "decode_tok_s": r["tok_s"], "decode_runs": r["decode_tok_s"],
           "prefill_tok_s": r["prefill_tok_s_median"], "majflt_per_token": r["majflt_per_token"]}
    for k in ("draft_accept", "edx_pf", "rss_gb"):
        if k in r:
            out[k] = r[k]
    out.update(extra)
    return out


base = j("base")
if base:
    (R / "baseline.json").write_text(json.dumps({
        "model": "Qwen3.8-Flash-Next UD-Q4_K_XL (qwen4exp), 111 GB incl. 28.8 GB IQ4_NL n-gram table",
        "box": "2x RTX 3090 (NVLink), Ryzen 5900X 12C, 62 GB RAM (~35 GB free), Samsung 990 PRO on btrfs/dm-crypt",
        "config": "stock llama-server, llama.cpp's own fit over both GPUs, mmap, lazy n-gram table (defaults)",
        "decode_tok_s": base["tok_s"], "decode_runs": base["decode_tok_s"], "prefill_tok_s": base["prefill_tok_s_median"],
        "ram_used_gb": base["rss_gb"], "majflt_per_token": base["majflt_per_token"], "vram_used_mib": base["used_vram_mib"],
        "kl_reference": "results/night5/kl_base.bin (this config's logits on 4,096 held-out coding tokens)"}, indent=1))

E = "engine-level: llama.cpp's scheduler runs a layer's GPU part and CPU part one after another and syncs with the host between splits; "
exact = {
    "staged_prefill": row("staged_prefill", what="prompt in 2048-token batches: each CPU layer's experts stream to the GPU once per 2048 tokens, not per 512", kl=kl("staged_prefill")),
    "exclusive_tiers": row("pin", what="experts pinned in VRAM have their RAM pages given back (EDX_PAGEOUT), so RAM only holds experts that are not on a GPU",
                           compare_to=row("pin_nopageout")),
    "cpu_lane_overlap": {"status": "not_built", "reason": E + "running the CPU's experts while the GPU works needs a split that runs both at once (an engine change, weeks). Costed in leverage.json."},
    "read_pool": row("pf_same", what="8 reader threads fetch the picked experts' other matrices (up/down) while the first one computes"),
    "layer_ahead_prefetch": row("pf_prev", what="read pool + next layer's experts guessed from the previous token's picks (his trick guesses with the next router; ours below)",
                                learned_variant=row("pf_markov"), combined=row("pf_both"),
                                his_router_variant="not built: needs the router input at run time inside the CPU path; offline recall of our guessers in predictor_offline.json"),
    "no_host_sync": {"status": "not_built", "reason": E + "removing the per-layer sync means GPU kernels that poll for experts landing (mapped flags), i.e. replacing ggml's scheduler for MoE layers."},
    "mtp": row("mtp", what="Qwen's own 1-layer draft head guesses 2 tokens ahead, the full model checks them in one pass (exact)",
               ab=(j("mtp_ab") or {}).get("arms")),
    "engram_ssd": row("engram_in_ram", what="the 28.8 GB n-gram table loaded into RAM (-lzm off) vs read row by row from the SSD (stock default, the baseline)",
                      note="stock llama.cpp already keeps tables > 4 GiB on disk and reads the ~16 rows a token needs"),
}
(R / "exact.json").write_text(json.dumps({"baseline": {"decode_tok_s": base and base["tok_s"], "prefill_tok_s": base and base["prefill_tok_s_median"]}, "tricks": exact}, indent=1))

pfab = (j("pf_ab") or {}).get("arms")
for k, arm in (("read_pool", "same"), ("layer_ahead_prefetch", "prev")):
    if pfab and exact[k].get("status") != "not_built":
        exact[k]["status"] = "measured"; exact[k]["ab_vs_off"] = pfab.get(arm); exact[k]["ab_off"] = pfab.get("off")
        exact[k]["ab_note"] = "in-process A/B, all arms with MTP drafting on (the per-request switch was ignored in that run)"
(R / "exact.json").write_text(json.dumps({"baseline": {"decode_tok_s": base and base["tok_s"], "prefill_tok_s": base and base["prefill_tok_s_median"]}, "tricks": exact}, indent=1))
pred = json.loads((R / "predictor_offline.json").read_text()) if (R / "predictor_offline.json").exists() else None
learned = {
    "profile_residency": row("pin", what="coding profile (from 10k traced tokens) decides which experts sit in VRAM, per expert instead of per layer",
                             kl=kl("pin_ub8"), staged=row("pin_staged")),
    "learned_prefetcher": ({"status": "measured", "exact": True, "what": "table learned from traces: experts at layer l -> likely experts at l+1",
                            "offline": pred, "ab_vs_off": pfab.get("markov"), "ab_both": pfab.get("both"), "ab_off": pfab.get("off")} if pfab else
                           row("pf_markov", exact=True, offline=pred)),
    "steering": row("pin_steer05", what="router nudged toward VRAM experts (lambda 0.5 and 1)", kl=kl("pin_steer05"),
                    strong=row("pin_steer1", kl=kl("pin_steer1"))),
    "adapter": {"status": "not_built", "reason": "a steering-aware adapter needs thousands of forward+backward passes of a 125B model; this box does ~30 tok/s of prompt. Shown on OLMoE and Granite (night 3)."},
}
for k in ("profile_residency", "steering"):
    if learned[k].get("status") == "measured" and learned[k].get("kl") is None:
        learned[k]["kl_missing"] = True
(R / "learned.json").write_text(json.dumps({"tricks": learned}, indent=1))

sh_log = R / "smoke_shared.log"
shared = j("stack_shared_1") is not None and sh_log.exists() and "tok/s" in sh_log.read_text(errors="replace")
final = shared or j("stack_final_1") is not None
st, st2 = ((j("stack_shared_1"), j("stack_shared_2")) if shared else (j("stack_final_1"), j("stack_final_2")) if final else (j("stack"), j("stack_repeat")))
if st and base:
    k = kl("stack_final" if final else "stack")
    sm = R / ("smoke_shared.log" if shared else "smoke_final.log" if final else "smoke.log")
    smoke = sm.read_text(errors="replace") if sm.exists() else ""
    runs = st["decode_tok_s"] + (st2["decode_tok_s"] if st2 else [])
    b2 = j("base2")
    (R / "stack.json").write_text(json.dumps({
        "tricks": ("ngram_table_populated_at_load (-lzm off), mtp (shared head on GPU0)" if shared else "ngram_table_populated_at_load (-lzm off), mtp") if final else (R / "stack_why.txt").read_text().strip(),
        "args": st["args"], "env": st["edxenv"], "which_runs": (["stack_shared_1", "stack_shared_2"] if shared else ["stack_final_1", "stack_final_2"]) if final else ["stack", "stack_repeat"],
        "baseline_tok_s": base["tok_s"], "baseline_prefill_tok_s": base["prefill_tok_s_median"],
        "baseline_warm_tok_s": b2 and b2["tok_s"], "baseline_warm_prefill_tok_s": b2 and b2["prefill_tok_s_median"],
        "exact_stack_tok_s": round(sorted(runs)[len(runs) // 2], 2), "exact_stack_runs": runs,
        "exact_stack_prefill_tok_s": st["prefill_tok_s_median"], "exact_stack_prefill_runs": st["prefill_tok_s"] + (st2["prefill_tok_s"] if st2 else []),
        "exact_stack_draft_accept": st.get("draft_accept"),
        "exact_stack_kl": k and k["kl"], "exact_stack_same_top_pct": k and k["same_top_pct"],
        "kl_note": "MTP output is exact by construction (verified by the full model); the KL run checks the rest of the stack against the baseline logits. Batch-shape floor ~0.026.",
        "smoke_ok": "tok/s" in smoke and "server died" not in smoke, "smoke_tail": smoke[-400:]}, indent=1))
print("assembled")
