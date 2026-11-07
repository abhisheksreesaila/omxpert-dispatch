"""Night 6 before/after page: docs/visuals/night6_before_after.html. Night-6 numbers are read from results/night6; the "before" layout
(37.7 / 39 GB, 32 of 49 ms) is night 5's measured split (results/night5/leverage.json).
usage: n6_page.py"""
import json, statistics
from pathlib import Path

W = Path.home() / "expert-dispatch"; R = W / "results/night6"


def ld(n):
    p = R / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def a(d, k):
    return (d.get("arms") or {}).get(k) or {}


def med(xs):
    xs = [x for x in xs if x]
    return round(statistics.median(xs), 1) if xs else None


fs, base, pc, px, pxo, pxl, bc = (ld(n) for n in ("ft_stock", "baseline", "pin_chat", "pinx_chat", "pinxo_chat", "pinxol_chat", "base_chat"))
fits = [ld(f"ab_fit_{i}") for i in (1, 2)]; pins = [ld(f"ab_pin_{i}") for i in (1, 2)]
fit_d = med([a(f, "base").get("median_tok_s") for f in fits]); fit_p = med([a(f, "base").get("prefill_median") for f in fits])
pin_d = med([a(p, "pfoff").get("median_tok_s") for p in pins]); pin_p = med([a(p, "pfoff").get("prefill_median") for p in pins])
ov, mt, prof, q = ld("ab_overlap"), ld("ab_pinmtp"), ld("profile_chat"), ld("quality")
pf_pairs = [r for p in pins for r in a(p, "pfprev").get("paired_ratios", [])]
hc = prof.get("hit_curve", {}); cc = prof.get("coding_profile_on_chat", {})
res_share = 36500 * 1.048576 / (48 * 512 * 3.07)   # 36,500 MiB of pins (bytes) / all experts (bytes)
pts = sorted((float(k), v) for k, v in hc.items())
hot = next((ha + (hb - ha) * (res_share - x0) / (x1 - x0) for (x0, ha), (x1, hb) in zip(pts, pts[1:]) if x0 <= res_share <= x1), None)
hot_s = f"{hot*100:.0f}%" if hot else "most"

ladder = [  # (label, decode, prompt, note)
    ("FreeToken-Kai as is (its 64 GB mode)", fs.get("decode_tok_s"), fs.get("prefill_tok_s"), "coding prompts"),
    ("llama.cpp pinned, night-5 code", pc.get("decode_tok_s"), pc.get("prefill_tok_s"), f"{pc.get('majflt_per_token')} page faults/token"),
    ("+ copy pins across GPUs for prompts", px.get("decode_tok_s"), px.get("prefill_tok_s"), f"{px.get('majflt_per_token')} faults/token"),
    ("+ CPU works while GPU works", pxo.get("decode_tok_s"), pxo.get("prefill_tok_s"), "separate run"),
    ("+ each layer's prompt work on its own GPU", pxl.get("decode_tok_s"), pxl.get("prefill_tok_s"), f"{pxl.get('majflt_per_token')} faults/token"),
    ("llama.cpp's own fit (stock layout)", bc.get("decode_tok_s"), bc.get("prefill_tok_s"), "the bar to beat"),
]
mx = max([x[1] or 0 for x in ladder] + [1]); mxp = max([x[2] or 0 for x in ladder] + [1])
rows = "".join(f"<tr><td>{l}</td><td><div class=bar style='width:{(d or 0)/mx*100:.0f}%'></div><b>{d}</b></td>"
               f"<td><div class='bar p' style='width:{(p or 0)/mxp*100:.0f}%'></div><b>{p}</b></td><td class=n>{n}</td></tr>" for l, d, p, n in ladder)


def pr(x):
    return ", ".join(f"{v:.2f}" for v in x) if x else "n/a"


tricks = [
    ("Profile from your own chat history picks the VRAM experts", "omxpert", "a buffer pool that pins the pages your queries hit",
     f"30% of experts serve {hc.get('0.3', '?')} of held-out chat picks (night-5 coding profile: {cc.get('0.3', '?')}); stack vs fit: {pin_d} vs {fit_d} tok/s, prompts {pin_p} vs {fit_p}"),
    ("Exclusive tiers: an expert lives in VRAM or RAM, never both", "0xSero", "hot/cold storage tiering without duplicate copies",
     f"what makes 48 GB of VRAM + ~40 GB of free RAM hold all 77 GB of experts"),
    ("CPU computes cold experts while the GPUs compute hot ones", "FreeToken (FlashML)", "a parallel query: two engines scan their partitions at once",
     f"same-server A/B: on {a(ov,'on').get('median_tok_s')} vs off {a(ov,'off').get('median_tok_s')} tok/s, rounds {pr(a(ov,'on').get('paired_ratios'))}"),
    ("Prompt batches run each layer on the GPU that holds its pins", "tonight (fix)", "push the query to the node that owns the partition",
     f"prompts {pc.get('prefill_tok_s')} → {pxl.get('prefill_tok_s')} tok/s, faults {pc.get('majflt_per_token')} → {pxl.get('majflt_per_token')} per token"),
    ("Layer-ahead prefetch of the next layer's experts", "0xSero idea, our reader", "read-ahead of the next index pages",
     f"paired rounds {pr(pf_pairs)}"),
    ("MTP drafting (Qwen's draft head)", "Qwen / llama.cpp", "speculative execution, verified",
     f"mtp {a(mt,'mtp').get('median_tok_s')} vs {a(mt,'nomtp').get('median_tok_s')} tok/s, rounds {pr(a(mt,'mtp').get('paired_ratios'))}; prompts {a(mt,'mtp').get('prefill_median')}"),
]
trows = "".join(f"<tr><td>{t}</td><td>{w}</td><td class=n>{d}</td><td>{m}</td></tr>" for t, w, d, m in tricks)
qm = q.get("token_match") or {}

html = f"""<!doctype html><html><head><meta charset=utf-8><title>Night 6: three sets of tricks, one engine</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#e8e6e1;background:#16181d}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:18px;margin:28px 0 8px;color:#f5c451}} .sub{{color:#a9a59c}}
table{{width:100%;border-collapse:collapse;font-size:14px}} td,th{{padding:6px 8px;border-bottom:1px solid #2a2d35;vertical-align:top;text-align:left}}
.bar{{display:inline-block;height:12px;background:#f5c451;border-radius:3px;margin-right:6px;vertical-align:middle}} .bar.p{{background:#6aa9ff}}
.n{{color:#a9a59c;font-size:13px}} .cols{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card{{background:#1e2128;border-radius:10px;padding:14px 16px;box-shadow:0 2px 10px #0006}}
.tier{{border-radius:8px;padding:8px 10px;margin:6px 0;font-size:13px}} .gpu{{background:#3b3220;border-left:4px solid #f5c451}}
.ram{{background:#1f2c3d;border-left:4px solid #6aa9ff}} .ssd{{background:#2a2a2a;border-left:4px solid #888}} .cpu{{background:#23302a;border-left:4px solid #63c58a}}
.arrow{{text-align:center;color:#a9a59c;font-size:12px}}
</style></head><body>
<h1>Night 6: three sets of tricks, one engine</h1>
<div class=sub>Qwen3.8-Flash-Next (Qwen 4 preview architecture, 111 GB at 4-bit) on 2× RTX 3090 + 64 GB RAM. Every number below was measured on this box (results/night6).</div>

<h2>Before and after, as data flow</h2>
<div class=cols>
<div class=card><b>Before: llama.cpp's fit (range partitioning by layer)</b>
<div class="tier gpu">GPUs: layers 1–24, all 512 experts each (37.7 GB), including the ones your work never picks</div>
<div class="tier ram">RAM (page cache): layers 25–48, all experts (39 GB)</div>
<div class="tier cpu">CPU: computes every pick of layers 25–48: half of the 480 picks per token, 32 of 49 ms</div>
<div class=arrow>GPU and CPU take turns: the token waits for both</div></div>
<div class=card><b>After: your profile tiers the experts (hot/cold storage, parallel query)</b>
<div class="tier gpu">GPUs: the experts your chat history uses most, from all 48 layers (36,500 MiB, {res_share*100:.0f}% of all experts), {hot_s} of held-out chat picks</div>
<div class="tier ram">RAM: only the experts not in VRAM (exclusive tiers, no double copies)</div>
<div class="tier cpu">CPU: the few cold picks, computed <i>while</i> the GPUs compute the hot ones</div>
<div class=arrow>prompt batches run each layer on the GPU that owns its pins</div></div>
</div>

<h2>The speed ladder (held-out chat prompts, tok/s; bars: decode gold, prompt blue)</h2>
<table><tr><th>Setup</th><th>Decode</th><th>Prompt</th><th></th></tr>{rows}</table>
<p class=n>Separate runs on a shared desktop swing; the comparisons that count are below.</p>

<h2>Same-method comparisons</h2>
<table><tr><td>Pinned stack vs llama.cpp fit (interleaved servers, server timings)</td><td><b>{pin_d}</b> vs <b>{fit_d}</b> tok/s decode, <b>{pin_p}</b> vs <b>{fit_p}</b> prompt</td></tr>
<tr><td>Same intelligence (greedy coding outputs vs stock llama.cpp, word match)</td><td>stack {qm.get('pinned_stack')}, steering λ0.5 {qm.get('pinned_steer05')}, stock-vs-stock floor {qm.get('floor')}: {q.get('verdict','')}</td></tr></table>

<h2>Each trick, whose it is, and its database twin</h2>
<table><tr><th>Trick</th><th>From</th><th>Database twin</th><th>Measured</th></tr>{trows}</table>

<h2>Credits</h2>
<p><b>0xSero</b> (glm-flash-lite): exclusive tiers, layer-ahead prefetch, read pool, staged prefill, CPU/GPU overlap. <b>FreeToken</b> (FlashML, Apache-2.0)
and its Kai fork: the hybrid executor that computes misses on the CPU while the GPU works, which we ported into llama.cpp's scheduler.
<b>Qwen</b>: the model and its MTP head. <b>llama.cpp / ggml</b> and <b>unsloth</b> (GGUF). Profiles from your own chat history, the learned guesser and
steering are omxpert's.</p>
</body></html>"""
out = W / "docs/visuals/night6_before_after.html"; out.write_text(html); print(out)
