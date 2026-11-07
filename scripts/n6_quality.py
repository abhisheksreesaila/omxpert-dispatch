"""Night 6 quality check: greedy outputs of a stack vs the reference (stock llama.cpp, same GGUF) on the same coding prompts.
Reports per prompt the length of the identical prefix (characters) and the share of matching whitespace-separated words
position by position; the floor is the reference against a second reference run (identical settings, separate process).
usage: n6_quality.py <out.json> <ref.json> <ref2.json> <name=stack.json> [<name=stack.json> ...]"""
import json, sys


def cmp(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    wa, wb = a.split(), b.split()
    m = sum(x == y for x, y in zip(wa, wb)) / max(1, max(len(wa), len(wb)))
    return {"same_prefix_chars": n, "len": [len(a), len(b)], "word_match": round(m, 3)}


def score(ref, other):
    rows = {k: cmp(ref["texts"][k], other["texts"][k]) for k in ref["texts"] if k in other.get("texts", {})}
    return {"per_prompt": rows, "word_match": round(sum(r["word_match"] for r in rows.values()) / max(1, len(rows)), 3),
            "identical": all(r["same_prefix_chars"] == max(r["len"]) for r in rows.values())}


out, ref, ref2 = sys.argv[1], json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
res = {"reference": sys.argv[2], "floor": score(ref, ref2), "stacks": {}}
for a in sys.argv[4:]:
    name, path = a.split("=", 1)
    res["stacks"][name] = score(ref, json.load(open(path)))
res["token_match"] = {k: v["word_match"] for k, v in res["stacks"].items()}
print(json.dumps({"floor": res["floor"]["word_match"], "floor_identical": res["floor"]["identical"], **res["token_match"]}, indent=1))
json.dump(res, open(out, "w"), indent=1)
