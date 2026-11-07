"""Night 6 shared client: times any OpenAI-compatible server (llama.cpp or FreeToken) the same way.
Raw completions (no chat template), greedy, streamed. Prompt tok/s = prompt tokens / time to first token; decode tok/s =
(completion tokens - 1) / (last token - first token). Prompts: held-out coding text, as in night 5 (3 x ~2000 chars).
Saves the generated text too, for the greedy token-match quality check.
usage: n6_bench.py <base_url e.g. http://127.0.0.1:18081> <name> <out.json>   env: ROUNDS (2) NPRED (128) WARM (5) EXTRA (json merged into each request)"""
import json, os, statistics, sys, time, urllib.request
from pathlib import Path

W = Path.home() / "expert-dispatch"
base, name, out = sys.argv[1].rstrip("/"), sys.argv[2], Path(sys.argv[3])
ROUNDS, NPRED, WARM = int(os.environ.get("ROUNDS", 2)), int(os.environ.get("NPRED", 128)), int(os.environ.get("WARM", 5))
EXTRA = json.loads(os.environ.get("EXTRA", "{}"))
PROMPTS = os.environ.get("PROMPTS", "coding")   # coding: night-5 held-out code; chat: held-out sessions of the owner's chat (texts not saved)
text = (W / ("private/chat/chat_heldout.txt" if PROMPTS == "chat" else "corpus/coding_heldout.txt")).read_text()
prompts = [text[i * 40000 + (20000 if PROMPTS == "chat" else 0): i * 40000 + 2000 + (20000 if PROMPTS == "chat" else 0)] for i in range(10)]


def model_id():
    try:
        return json.loads(urllib.request.urlopen(base + "/v1/models", timeout=30).read())["data"][0]["id"]
    except Exception:
        return "default"


MODEL = model_id()


def gen(p, n):
    body = {"model": MODEL, "prompt": p, "max_tokens": n, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "cache_prompt": False, "ignore_eos": True}
    body.update(EXTRA)
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); t_first = t_last = None; pieces = []; usage = None; nchunks = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                if c.get("text"):
                    now = time.time()
                    t_first = t_first or now; t_last = now; nchunks += 1
                    pieces.append(c["text"])
    pt = (usage or {}).get("prompt_tokens"); ct = (usage or {}).get("completion_tokens") or nchunks
    ttft = (t_first or time.time()) - t0
    dec = (ct - 1) / (t_last - t_first) if t_first and t_last and t_last > t_first and ct > 1 else None
    return {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": round(ttft, 3),
            "prefill_tok_s": round(pt / ttft, 1) if pt else None, "decode_tok_s": round(dec, 2) if dec else None,
            "text": "".join(pieces)}


res = {"name": name, "base": base, "model": MODEL, "extra": EXTRA, "npred": NPRED, "t": time.strftime("%H:%M:%S")}
res["warm"] = [gen(prompts[3 + i % 7], 48)["decode_tok_s"] for i in range(WARM)]
runs = []
for r in range(ROUNDS):
    for i, p in enumerate(prompts[:3]):
        g = gen(p, NPRED); g["round"], g["prompt"] = r, i; runs.append(g)
res["runs"] = runs
dec = [g["decode_tok_s"] for g in runs if g["decode_tok_s"]]; pre = [g["prefill_tok_s"] for g in runs if g["prefill_tok_s"]]
res["decode_tok_s"] = round(statistics.median(dec), 2) if dec else 0
res["prefill_tok_s"] = round(statistics.median(pre), 1) if pre else 0
res["prompts"] = PROMPTS
if PROMPTS == "chat":
    for g in runs: g.pop("text", None)
else:
    res["texts"] = {str(g["prompt"]): g["text"] for g in runs if g["round"] == 0}
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("name", "decode_tok_s", "prefill_tok_s", "warm")}))
