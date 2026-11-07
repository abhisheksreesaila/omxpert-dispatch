"""Night 5 harness: one long-running llama-server per config on Qwen3.8-Flash-Next, warmed until decode is steady, then
measured on held-out coding prompts. Same method as night 4 (per-process benchmarks at the RAM edge measure page-cache
churn, not the config).
usage: ARGS="<extra llama-server args>" EDXENV="K=V K=V" night5_server.py <name> <out.json>
Records decode/prefill tok/s, major page faults per generated token (reads that went to the SSD), RAM/VRAM, the EDX_PF
stats line and speculative acceptance when drafting is on.
"""
import json, os, re, subprocess, sys, threading, time, urllib.request
from pathlib import Path

W = Path.home() / "expert-dispatch"
name, out = sys.argv[1], Path(sys.argv[2])
M = Path(os.environ.get("MODEL", str(sorted((Path.home() / "models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL").glob("*00001-of-*.gguf"))[0])))
PORT = int(os.environ.get("PORT", "18081"))
URL = f"http://127.0.0.1:{PORT}"
NPRED = int(os.environ.get("NPRED", "128"))


def free_vram():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()
    return [int(x) for x in q]


def used_vram():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()
    return [int(x) for x in q]


def avail_ram_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20


def proc_stat(pid):   # (major faults, RSS GB) of the server
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return int(f[9]), int(f[21]) * 4096 / 1e9


env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1")
for kv in os.environ.get("EDXENV", "").split():
    k, v = kv.split("=", 1); env[k] = v
args = [str(W / "llama.cpp/build/bin/llama-server"), "-m", str(M), "--host", "127.0.0.1", "--port", str(PORT),
        "-c", "4096", "-np", "1", "--no-webui", "-lv", "3"] + os.environ.get("ARGS", "").split()
log_path = out.with_suffix(".server.log")
log = open(log_path, "w")
state0 = {"name": name, "args": os.environ.get("ARGS", ""), "edxenv": os.environ.get("EDXENV", ""),
          "avail_ram_gb_start": round(avail_ram_gb(), 1), "free_vram_start": free_vram()}
srv = subprocess.Popen(["nice", "-n", "10"] + args, env=env, stdout=log, stderr=subprocess.STDOUT)


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


res = dict(state0)
try:
    t0 = time.time()
    while True:
        try:
            if json.loads(urllib.request.urlopen(URL + "/health", timeout=5).read()).get("status") == "ok":
                break
        except Exception:
            pass
        if srv.poll() is not None or time.time() - t0 > 1800:
            raise SystemExit("server did not start, see " + str(log_path))
        time.sleep(3)
    res["load_s"] = round(time.time() - t0, 1)
    text = (W / "corpus/coding_heldout.txt").read_text()
    prompts = [text[i * 40000: i * 40000 + 2000] for i in range(10)]

    def gen(p, n):
        r = post("/completion", {"prompt": p, "n_predict": n, "temperature": 0, "cache_prompt": False})
        return r

    warm = []
    for i in range(8):
        warm.append(round(gen(prompts[3 + i % 7], 48)["timings"]["predicted_per_second"], 2))
        if len(warm) >= 3 and abs(warm[-1] - warm[-2]) / warm[-1] < 0.10:
            break
    res["warmup_tok_s"] = warm
    f0, _ = proc_stat(srv.pid)
    runs = [gen(p, NPRED) for p in prompts[:3]]
    f1, rss = proc_stat(srv.pid)
    tm = [r["timings"] for r in runs]
    ntok = sum(t["predicted_n"] for t in tm) + sum(t["prompt_n"] for t in tm)
    res["decode_tok_s"] = [round(t["predicted_per_second"], 2) for t in tm]
    res["prefill_tok_s"] = [round(t["prompt_per_second"], 1) for t in tm]
    res["tok_s"] = round(sorted(res["decode_tok_s"])[1], 2)
    res["prefill_tok_s_median"] = round(sorted(res["prefill_tok_s"])[1], 1)
    res["majflt_per_token"] = round((f1 - f0) / max(ntok, 1), 1)
    res["ssd_MB_per_token_est"] = round((f1 - f0) * 4096 / 1e6 / max(ntok, 1), 2)   # lower bound: one 4 KiB page per fault
    if "draft_n" in tm[0]:
        dn = sum(t.get("draft_n", 0) for t in tm); da = sum(t.get("draft_n_accepted", 0) for t in tm)
        res["draft_accept"] = round(da / max(dn, 1), 3); res["draft_n"] = dn
    res["texts"] = [r["content"][:400] for r in runs]
    res["rss_gb"] = round(rss, 1); res["used_vram_mib"] = used_vram(); res["avail_ram_gb_end"] = round(avail_ram_gb(), 1)
finally:
    srv.terminate()
    try:
        srv.wait(120)
    except Exception:
        srv.kill()
    log.close()
    for line in open(log_path, errors="replace"):
        if line.startswith("EDX_PF "):
            res["edx_pf"] = line.strip()
        m = re.search(r"(CUDA\d|CPU_Mapped|CPU) model buffer size =\s+([\d.]+) MiB", line)
        if m:
            res.setdefault("buffers_mib", {})[m.group(1)] = res.get("buffers_mib", {}).get(m.group(1), 0) + float(m.group(2))
    print(json.dumps({k: v for k, v in res.items() if k != "texts"}), flush=True)
    out.write_text(json.dumps(res, indent=1))
