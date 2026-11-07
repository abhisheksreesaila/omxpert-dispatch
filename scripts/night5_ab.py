"""Night 5 in-process A/B: ONE warmed llama-server, the decode-time tricks switched per request, arms taken in turn over
several rounds so page-cache drift hits every arm alike (separate server runs of one config swung 11-19 tok/s here).
Arms: prefetch mode (EDX_PF_CTL file, re-read every 64 ops) x MTP drafting on/off (per-request speculative.type).
usage: ARGS=... EDXENV=... night5_ab.py <out.json>
"""
import json, os, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

W = Path.home() / "expert-dispatch"; out = Path(sys.argv[1]); R = W / "results/night5"
M = sorted((Path.home() / "models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL").glob("*00001-of-*.gguf"))[0]
MTP = Path.home() / "models/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
PORT = 18082; URL = f"http://127.0.0.1:{PORT}"
CTL = "/tmp/edx_pf_ctl"; Path(CTL).write_text("off\n")
ROUNDS = int(os.environ.get("ROUNDS", "7"))
ARMS = [("off", False), ("same", False), ("prev", False), ("markov", False), ("both", False), ("off", True), ("both", True)]
if os.environ.get("ARMS") == "mtp":   # MTP on/off only (needs the server patch honouring speculative.n_max = 0)
    ARMS = [("off", False), ("off", True)]

env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", EDX_PF_CTL=CTL, EDX_PF_PRED=str(R / "pred_q38.bin"))
for kv in os.environ.get("EDXENV", "").split():
    k, v = kv.split("=", 1); env[k] = v
args = [str(W / "llama.cpp/build/bin/llama-server"), "-m", str(M), "--host", "127.0.0.1", "--port", str(PORT), "-c", "4096", "-np", "1",
        "--no-webui", "-lv", "3", "-md", str(MTP), "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"] + os.environ.get("ARGS", "-lzm off").split()   # -lzm off: 4x faster prompts, far fewer faults (engram_in_ram run)
log = open(out.with_suffix(".server.log"), "w")
srv = subprocess.Popen(["nice", "-n", "10"] + args, env=env, stdout=log, stderr=subprocess.STDOUT)


def post(body):
    req = urllib.request.Request(URL + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def majflt():
    return int(open(f"/proc/{srv.pid}/stat").read().rsplit(")", 1)[1].split()[9])


def pfstats():   # cumulative per-mode counters written by edx-prefetch: mode ops pages resident ns
    d = {}
    try:
        for line in Path(CTL + ".stats").read_text().split("\n"):
            f = line.split()
            if len(f) == 5:
                d[int(f[0])] = [int(x) for x in f[1:]]
    except Exception:
        pass
    return d


MODE_IDX = {"off": 1, "same": 2, "prev": 3, "markov": 4, "both": 5}


def gen(p, n, mtp):
    return post({"prompt": p, "n_predict": n, "temperature": 0, "cache_prompt": False, "speculative.n_max": 2 if mtp else 0})


res = {"args": os.environ.get("ARGS", "-lzm off"), "edxenv": os.environ.get("EDXENV", ""), "rounds": ROUNDS, "samples": []}
try:
    t0 = time.time()
    while True:
        try:
            if json.loads(urllib.request.urlopen(URL + "/health", timeout=5).read()).get("status") == "ok":
                break
        except Exception:
            pass
        if srv.poll() is not None or time.time() - t0 > 1800:
            raise SystemExit("server did not start")
        time.sleep(3)
    text = (W / "corpus/coding_heldout.txt").read_text()
    prompts = [text[i * 40000: i * 40000 + 1200] for i in range(max(len(ARMS), ROUNDS) + 4)]
    warm = []
    for i in range(6):
        warm.append(round(gen(prompts[-1 - i % 4], 48, False)["timings"]["predicted_per_second"], 2))
    res["warmup_tok_s"] = warm
    for rnd in range(ROUNDS):
        order = list(enumerate(ARMS)) if rnd % 2 == 0 else list(enumerate(ARMS))[::-1]   # alternate order: the 2nd slot of a round ran ~5% faster (mtp_ab_invalid)
        for a, (mode, mtp) in order:
            Path(CTL).write_text(mode + "\n"); time.sleep(0.2)
            p = prompts[rnd % ROUNDS] if len(ARMS) < 4 else prompts[(a + rnd) % len(ARMS)]   # paired: same prompt for both arms in a round (2 arms) / Latin square (7 arms)
            f0 = majflt(); s0 = pfstats(); r = gen(p, 128, mtp); time.sleep(0.3); f1 = majflt(); s1 = pfstats(); t = r["timings"]
            i = MODE_IDX[mode]; d = [b - a for a, b in zip(s0.get(i, [0] * 4), s1.get(i, [0] * 4))]
            s = {"round": rnd, "mode": mode, "mtp": mtp, "decode_tok_s": round(t["predicted_per_second"], 2),
                 "prefill_tok_s": round(t["prompt_per_second"], 1), "n": t["predicted_n"], "majflt": f1 - f0,
                 "cpu_expert_ms_per_tok": round(d[3] / 1e6 / max(t["predicted_n"], 1), 2), "ram_hit": round(d[2] / d[1], 4) if d[1] else None}
            if "draft_n" in t:
                s["draft_n"] = t["draft_n"]; s["draft_acc"] = t.get("draft_n_accepted", 0)
            res["samples"].append(s); print(s, flush=True)
    arms = {}
    for mode, mtp in ARMS:
        ss = [s for s in res["samples"] if s["mode"] == mode and s["mtp"] == mtp]
        key = mode + ("+mtp" if mtp else "")
        arms[key] = {"median_tok_s": round(statistics.median(s["decode_tok_s"] for s in ss), 2), "runs": [s["decode_tok_s"] for s in ss],
                     "majflt_per_tok": round(sum(s["majflt"] for s in ss) / max(1, sum(s["n"] for s in ss)), 1),
                     "cpu_expert_ms_per_tok": round(statistics.median(s["cpu_expert_ms_per_tok"] for s in ss), 2),
                     "ram_hit": round(statistics.median(s["ram_hit"] for s in ss if s["ram_hit"] is not None), 4) if any(s["ram_hit"] is not None for s in ss) else None}
        if mtp:
            dn = sum(s.get("draft_n", 0) for s in ss); arms[key]["draft_accept"] = round(sum(s.get("draft_acc", 0) for s in ss) / max(dn, 1), 3)
    off = arms["off"]["median_tok_s"]
    for k in arms:
        arms[k]["vs_off"] = round(arms[k]["median_tok_s"] / off, 3)
        # paired: same round, ratio to that round's 'off' sample (removes drift between rounds)
        pr = []
        for rnd in range(ROUNDS):
            o = [s for s in res["samples"] if s["round"] == rnd and s["mode"] == "off" and not s["mtp"]][0]["decode_tok_s"]
            m, mt = (k.replace("+mtp", ""), k.endswith("+mtp"))
            x = [s for s in res["samples"] if s["round"] == rnd and s["mode"] == m and s["mtp"] == mt][0]["decode_tok_s"]
            pr.append(round(x / o, 3))
        arms[k]["paired_ratios"] = pr
    res["arms"] = arms
    try:
        res["pf_stats_by_mode"] = Path(CTL + ".stats").read_text().split("\n")
    except Exception:
        pass
finally:
    srv.terminate()
    try:
        srv.wait(120)
    except Exception:
        srv.kill()
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res.get("arms", {}), indent=1))
