"""Drive a running `max serve` (with edx_gptoss hooks) over the corpus.

prefill: each corpus chunk is sent as a raw completion prompt with max_tokens=1,
so the trace captures the routing of every corpus token.
decode:  real prompts from held-out chats/code, 256 generated tokens each.

usage: max_drive.py prefill <tag> <split> <max_tokens_total>
       max_drive.py decode <n_prompts>
       max_drive.py gen <outfile.json> <n_prompts>   (greedy continuations, for bias quality checks)
"""
import json, sys, time, random, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

W = Path.home() / "expert-dispatch"
TAGF = W / "traces/max/TAG"
URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "openai/gpt-oss-20b"
CHARS = 6000  # ~1.5K tokens per request


def post(prompt, max_tokens, temperature=0.0):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": temperature}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["text"]


def chunks(tag, split, limit_chars):
    text = (W / f"corpus/{tag}_{split}.txt").read_text()
    out = [text[i:i + CHARS] for i in range(0, min(len(text), limit_chars), CHARS)]
    return [c for c in out if len(c) > 200]


def set_tag(t):
    TAGF.parent.mkdir(parents=True, exist_ok=True)
    TAGF.write_text(t)


def prompts(n, seed=0):
    random.seed(seed)
    ps = []
    for tag in ("design_chat", "coding"):
        cs = chunks(tag, "heldout", 10 ** 9)
        random.shuffle(cs)
        ps += [c[:2000] for c in cs[: n // 2]]
    return ps


if __name__ == "__main__":
    mode = sys.argv[1]
    t0 = time.time()
    if mode == "prefill":
        tag, split, lim = sys.argv[2], sys.argv[3], int(sys.argv[4])
        cs = chunks(tag, split, lim * 4)
        set_tag(f"{tag}_{split}")
        with ThreadPoolExecutor(1) as ex:
            list(ex.map(lambda c: post(c, 1), cs))
        print(f"{tag}_{split}: {len(cs)} chunks in {time.time() - t0:.0f}s")
    elif mode == "decode":
        ps = prompts(int(sys.argv[2]))
        set_tag("decode_sample")
        for p in ps:
            post(p, 256)
        print(f"decode: {len(ps)} prompts in {time.time() - t0:.0f}s")
    elif mode == "gen":
        ps = prompts(int(sys.argv[3]), seed=1)
        set_tag("gen_" + Path(sys.argv[2]).stem)  # keep routing so steered hit rate can be measured
        outs = [post(p, 128) for p in ps]
        Path(sys.argv[2]).write_text(json.dumps(outs))
        print(f"gen: {len(ps)} in {time.time() - t0:.0f}s")
    set_tag("off")
