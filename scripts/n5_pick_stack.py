"""Pick the exact stack from measured runs. Decode tricks from the in-process A/B (pf_ab.json): a prefetch mode and/or MTP
joins if its median beats 'off' by >5% AND it beat 'off' in most paired rounds. Staged prefill joins if its prompt speed
beat both baseline runs. Prints ARGS|||EDXENV|||why."""
import json
from pathlib import Path
R = Path.home() / "expert-dispatch/results/night5"
MTP = str(Path.home() / "models/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q8_0.gguf")


def j(n):
    p = R / f"{n}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return None if "skipped" in d else d


args, env, why = [], [], []
bases = [b for b in (j("base"), j("base2")) if b]
eg = j("engram_in_ram")   # the n-gram table through the OS page cache (-lzm off) instead of llama.cpp's lazy reader
if eg and bases and all(eg["prefill_tok_s_median"] > b["prefill_tok_s_median"] * 1.5 for b in bases) and eg["tok_s"] >= min(b["tok_s"] for b in bases):
    args += ["-lzm", "off"]; why.append("ngram_table_via_page_cache")
st = j("staged_prefill"); sl = j("staged_lzm")
if "-lzm" in args and sl and eg:   # judge staged prefill on top of the page-cache table: staged_lzm vs engram_in_ram
    if sl["prefill_tok_s_median"] > eg["prefill_tok_s_median"] * 1.05:
        args += ["-ub", "2048", "-b", "2048"]; why.append("staged_prefill")
elif st and bases and all(st["prefill_tok_s_median"] > b["prefill_tok_s_median"] * 1.05 for b in bases):
    args += ["-ub", "2048", "-b", "2048"]; why.append("staged_prefill")
ab = j("pf_ab")
if ab and "arms" in ab:
    A = ab["arms"]
    wins = lambda k: A[k]["vs_off"] > 1.05 and sum(r > 1.0 for r in A[k]["paired_ratios"]) > len(A[k]["paired_ratios"]) / 2
    pf = [k for k in ("same", "prev", "markov", "both") if wins(k)]
    best_pf = max(pf, key=lambda k: A[k]["median_tok_s"]) if pf else None
    # MTP from its own paired A/B (in pf_ab every arm drafted: the per-request switch was ignored until the server patch)
    mab = j("mtp_ab")
    use_mtp = bool(mab and "arms" in mab and mab["arms"]["off+mtp"]["vs_off"] > 1.05
                   and sum(r > 1.0 for r in mab["arms"]["off+mtp"]["paired_ratios"]) > len(mab["arms"]["off+mtp"]["paired_ratios"]) / 2)
    if best_pf:
        env += [f"EDX_PF_MODE={best_pf}"] + ([f"EDX_PF_PRED={R / 'pred_q38.bin'}"] if best_pf in ("markov", "both") else [])
        why.append(f"prefetch_{best_pf}")
    if use_mtp:
        args += ["-md", MTP, "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]; why.append("mtp")
print(" ".join(args) + "|||" + " ".join(env) + "|||" + (",".join(why) or "none"))
