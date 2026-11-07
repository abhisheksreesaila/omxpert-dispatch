"""Night 5 leverage model: what each added part (or software change) would buy on Qwen3.8-Flash-Next on this box.
Components MEASURED in one warmed server (mtp_ab.json, MTP-off arm, n-gram table via page cache): total ms/token and the CPU
expert ms/token (timer around llama.cpp's CPU expert op); the rest is GPU work + hand-offs + sampling. llama.cpp's own fit puts
24 of 48 layers' experts on the CPU (op counts). The GPU<->CPU hand-off cost comes from the hand-off test (same 25 CPU layers,
alternating vs contiguous). Checked on the contiguous runs (separate processes, 25 CPU layers), which it was not fitted to.
usage: n5_leverage.py <out.json>"""
import json, statistics, sys
from pathlib import Path

W = Path.home() / "expert-dispatch"; R = W / "results/night5"


def ld(n):
    p = R / n
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return None if "skipped" in d else d


ab, hit = ld("mtp_ab.json"), ld("hitcurve_q38.json")
A = ab["arms"]
T_tot = 1000 / A["off"]["median_tok_s"]
T_cpu = A["off"]["cpu_expert_ms_per_tok"]
T_other = T_tot - T_cpu
mtp_gain = A["off+mtp"]["median_tok_s"] / A["off"]["median_tok_s"]
L, K, E = 48, 10, 512
CPU_LAYERS = 24                       # measured: 73.3 CPU expert ops per decoded token / 3 matrices (pf_stats run)
EXP_MB = 3.07                          # 1.573 GB per layer / 512 (gguf)
TOTAL_EXP_GB = L * E * EXP_MB / 1e3
ms_per_cpu_layer = T_cpu / CPU_LAYERS  # all 10 picks of one layer on the CPU


def hitf(res):   # held-out hit rate of the coding profile at a resident share (linear between measured points)
    pts = sorted([(0.0, 0.0), (1.0, 1.0)] + [(float(k), v) for k, v in hit["per_expert"].items()])
    for (a, ha), (b, hb) in zip(pts, pts[1:]):
        if a <= res <= b:
            return ha + (hb - ha) * (res - a) / (b - a)


# hand-off cost: alternating (25 CPU layers interleaved, ~50 GPU<->CPU switches/token) vs contiguous (~2)
alt = [ld(f"handoff_alt_{i}.json") for i in (1, 2, 3)]; con = [ld(f"handoff_contig2_{i}.json") for i in (1, 2)]
alt = [x["tok_s"] for x in alt if x and "tok_s" in x]; con = [x["tok_s"] for x in con if x and "tok_s" in x]
handoff_ms = None
if alt and con:
    handoff_ms = max(0.0, (1000 / statistics.mean(alt) - 1000 / statistics.mean(con)) / (50 - 4))   # contig2 has ~4 switches


def decode(cpu_picks_share, handoffs=2, cpu_speed=1.0, overlap=False, mtp=False):
    """cpu_picks_share: share of the 480 picks per token computed by the CPU (llama.cpp fit today: 0.5)."""
    tc = T_cpu * (cpu_picks_share / 0.5) / cpu_speed
    th = (handoff_ms or 0.0) * max(0, handoffs - 2)
    t = (max(tc, T_other) if overlap else tc + T_other) + (0.0 if overlap else th)   # hand-off cost 0 until measured (contig2 runs)
    return round(1000 / t * (mtp_gain if mtp else 1.0), 1)


out = {"measured": {"total_ms": round(T_tot, 1), "cpu_expert_ms": T_cpu, "other_ms": round(T_other, 1), "cpu_layers": CPU_LAYERS,
                    "ms_per_cpu_layer": round(ms_per_cpu_layer, 2), "off_tok_s": A["off"]["median_tok_s"], "mtp_tok_s": A["off+mtp"]["median_tok_s"],
                    "mtp_gain": round(mtp_gain, 3), "ram_hit_off": A["off"].get("ram_hit"), "handoff_ms": handoff_ms and round(handoff_ms, 3),
                    "handoff_runs": {"alternating": alt, "contiguous": con}}}
check = {"note": "hand-off runs missing"}
chk = con or alt
if chk:
    pred = decode(25 / L)
    check = {"config": "25 CPU layers, -lzm off, no MTP, separate processes (" + ("handoff_contig2" if con else "handoff_alt") + ")", "pred_tok_s": pred,
             "meas_tok_s": round(statistics.mean(chk), 2), "meas_runs": chk, "err": round((pred - statistics.mean(chk)) / statistics.mean(chk), 3), "fitted": False}
out["check"] = check

vram_exp_gb = (L - CPU_LAYERS) * E * EXP_MB / 1e3
res_now = vram_exp_gb / TOTAL_EXP_GB
egpu_res = min(1.0, (vram_exp_gb + 22) / TOTAL_EXP_GB)
wf = {}
wf["now"] = {"label": "This box, llama.cpp fit, page-cache table (no MTP)", "decode_tok_s": decode(0.5), "why": "measured split: half of the 480 expert picks per token on the CPU"}
wf["now_mtp"] = {"label": "Same + MTP drafting (measured)", "decode_tok_s": decode(0.5, mtp=True), "why": f"MTP multiplies decode by {mtp_gain:.2f} in the paired A/B"}
pins = [ld(f"pin_lzm_{i}.json") for i in (1, 2)]; pins = [x["tok_s"] for x in pins if x and "tok_s" in x]
fits = [ld(f"fit_lzm_{i}.json") for i in (1, 2)]; fits = [x["tok_s"] for x in fits if x and "tok_s" in x]
pred_pin = decode(1 - hitf(res_now))
wf["profile_llamacpp"] = {"label": "Software: profile picks VRAM experts, inside llama.cpp (MEASURED)", "decode_tok_s": round(statistics.mean(pins), 1) if pins else pred_pin,
                          "why": f"measured {pins} vs llama.cpp fit {fits} (same flags, interleaved); the model predicted {pred_pin}: llama.cpp's pinned path costs something this model does not see"}
out["profile_miss"] = {"pred_tok_s": pred_pin, "meas_tok_s": pins, "fit_tok_s": fits,
                       "note": "CPU time does not scale with CPU picks in llama.cpp's pinned mode; until that cost is found, the 'engine' rows are optimistic"}
wf["profile_engine"] = {"label": "Software: profile + his engine (overlap, no per-layer sync), ESTIMATE, optimistic", "decode_tok_s": decode(1 - hitf(res_now), overlap=True),
                        "why": "the CPU's few cold picks run while the GPU works; no hand-off stalls"}
wf["overlap"] = {"label": "Software: CPU works while the GPU works (engine change only)", "decode_tok_s": decode(0.5, overlap=True), "why": "time = max(CPU, GPU) instead of their sum"}
wf["second_nvme"] = {"label": "Second NVMe (RAID0)", "decode_tok_s": decode(0.5), "prefill": "faster model load (111 GB) and cold starts",
                     "why": f"no decode gain: {A['off'].get('ram_hit', 1)*100:.2f}% of expert pages are already in RAM once the table goes through the page cache"}
wf["ram_128gb"] = {"label": "128 GB RAM (+64 GB DDR4, 4 slots)", "decode_tok_s": decode(0.5), "prefill": "no SSD at all; room for the lossless 102 GB n-gram table or 8-bit experts",
                   "why": "decode is CPU-bound, not RAM-bound; RAM buys quality headroom (lossless table, Q8 experts) and a second model, not speed"}
wf["egpu"] = {"label": "eGPU 24 GB + your profile, ESTIMATE (needs the engine)", "decode_tok_s": decode(1 - hitf(egpu_res), handoffs=96),
              "why": f"{egpu_res*100:.0f}% of experts in VRAM serve {hitf(egpu_res)*100:.0f}% of coding picks; hidden states cross the x4 link (~10 KB per hop)"}
wf["egpu_layers"] = {"label": "eGPU as 7 more whole layers (works in llama.cpp today)", "decode_tok_s": decode((CPU_LAYERS - 7) / L), "why": "llama.cpp layer split, 17 CPU layers left"}
wf["cpu_16core"] = {"label": "16-core CPU (5950X drop-in)", "decode_tok_s": decode(0.5, cpu_speed=16 / 12), "why": "upper bound: assumes the CPU expert math scales with cores; it is partly RAM-bandwidth bound"}
wf["npu"] = {"label": "NPU", "decode_tok_s": "no gain", "why": "this desktop has none; an NPU sharing the same DDR4 reads experts no faster than the CPU (memory-bound); a separate NPU box is a network hop like the Mac"}
wf["mac"] = {"label": "M4 Mac (16 GB) over Tailscale", "decode_tok_s": "background lane only", "why": "measured round trip 53 ms average (7.6-302 ms): one hop costs more than a whole token here; give it its own jobs instead"}
wf["dream"] = {"label": "Profile + engine + eGPU + MTP, ESTIMATE, optimistic", "decode_tok_s": decode(1 - hitf(egpu_res), overlap=True, mtp=True), "why": "GPU-bound: what is left is the GPUs' own time"}
out["whatif"] = wf
out["note"] = "decode estimates built from measured components (MTP-off arm unless marked); 'profile_engine', 'overlap' and 'dream' need an engine that does not exist yet"
print(json.dumps(out, indent=1))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
