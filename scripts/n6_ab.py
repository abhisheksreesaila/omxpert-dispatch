"""Night 6 in-process A/B: one warmed llama.cpp server, arms switched per request, order alternated each round (night 5's method:
separate runs on this shared desktop swing 11-23 tok/s, and the second arm of a round runs ~5% faster).
Arms: name=ctl_value[:json_extra]   ctl_value is written to $CTL before the arm's requests ('-' leaves it alone).
usage: STACK_ARGS=... STACK_ENV=... CTL=/tmp/n6ov.ctl n6_ab.py <out.json> on=1 off=0
env: PROMPTS (chat|coding) ROUNDS (6) NPRED (128) WARM (6)"""
import json, os, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

W = Path.home() / "expert-dispatch"
out = Path(sys.argv[1]); arms = []
for a in sys.argv[2:]:
    name, rest = a.split("=", 1)
    ctl, _, extra = rest.partition(":")
    arms.append((name, ctl, json.loads(extra) if extra else {}))
ROUNDS, NPRED, WARM = int(os.environ.get("ROUNDS", 6)), int(os.environ.get("NPRED", 128)), int(os.environ.get("WARM", 6))
PROMPTS = os.environ.get("PROMPTS", "chat"); CTL = os.environ.get("CTL", "/tmp/n6ov.ctl")
text = (W / ("private/chat/chat_heldout.txt" if PROMPTS == "chat" else "corpus/coding_heldout.txt")).read_text()
off = 20000 if PROMPTS == "chat" else 0
prompts = [text[i * 40000 + off: i * 40000 + off + 2000] for i in range(10)]
PORT = 18082; URL = f"http://127.0.0.1:{PORT}"
env = dict(os.environ, PORT=str(PORT), CTX="4096")
log = open(W / f"results/night6/llama_{out.stem}.log", "w")
srv = subprocess.Popen([str(W / "omxpert/run.sh")], env=env, stdout=log, stderr=subprocess.STDOUT)


def post(body):
    r = urllib.request.urlopen(urllib.request.Request(URL + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600)
    return json.loads(r.read())


def gen(p, n, extra):
    b = {"prompt": p, "n_predict": n, "temperature": 0, "cache_prompt": False}; b.update(extra)
    r = post(b); t = r["timings"]
    return {"decode": round(t["predicted_per_second"], 2), "prefill": round(t["prompt_per_second"], 1),
            "draft_acc": (t.get("draft_n_accepted", 0) / t["draft_n"]) if t.get("draft_n") else None, "content": r["content"]}


try:
    t0 = time.time()
    while True:
        try:
            if json.loads(urllib.request.urlopen(URL + "/health", timeout=5).read()).get("status") == "ok":
                break
        except Exception:
            pass
        if srv.poll() is not None:
            raise SystemExit("server died")
        time.sleep(3)
    res = {"stack_args": os.environ.get("STACK_ARGS"), "stack_env": os.environ.get("STACK_ENV"), "prompts": PROMPTS, "load_s": round(time.time() - t0)}
    res["warm"] = [gen(prompts[3 + i % 7], 48, {})["decode"] for i in range(WARM)]
    per = {a[0]: [] for a in arms}; pre = {a[0]: [] for a in arms}; texts = {a[0]: {} for a in arms}; rounds = []
    for r in range(ROUNDS):
        order = arms if r % 2 == 0 else arms[::-1]; rr = {}
        for name, ctl, extra in order:
            if ctl != "-":
                Path(CTL).write_text(ctl); time.sleep(0.2)
            gs = [gen(p, NPRED, extra) for p in prompts[:3]]
            d = statistics.mean(g["decode"] for g in gs); rr[name] = d
            per[name] += [g["decode"] for g in gs]; pre[name] += [g["prefill"] for g in gs]
            if r == 0 and PROMPTS == "coding":
                texts[name] = {str(i): g["content"] for i, g in enumerate(gs)}
        rounds.append(rr)
    base = arms[-1][0] if len(arms) == 2 else arms[0][0]
    res["arms"] = {}
    for name, _, extra in arms:
        res["arms"][name] = {"extra": extra, "median_tok_s": round(statistics.median(per[name]), 2), "prefill_median": round(statistics.median(pre[name]), 1),
                             "paired_ratios": [round(x[name] / x[base], 3) for x in rounds]}
        res["arms"][name]["vs_" + base] = round(res["arms"][name]["median_tok_s"] / statistics.median(per[base]), 3)
    res["rounds"] = rounds; res["base_arm"] = base
    if PROMPTS == "coding":
        res["texts"] = texts
    out.write_text(json.dumps(res, indent=1)); print(json.dumps({k: (v["median_tok_s"], v["prefill_median"], v["paired_ratios"]) for k, v in res["arms"].items()}))
finally:
    srv.terminate()
    try:
        srv.wait(timeout=60)
    except subprocess.TimeoutExpired:   # EDX_PF reader threads can hold up exit
        srv.kill(); srv.wait()
