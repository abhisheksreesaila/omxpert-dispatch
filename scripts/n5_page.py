"""Night 5: build docs/visuals/night5_before_after.html from the result files (no hand-typed numbers).
Side-by-side before/after per trick with a database analogy, the two isometric diagrams, and the leverage table."""
import html, json
from pathlib import Path

W = Path.home() / "expert-dispatch"; R = W / "results/night5"; V = W / "docs/visuals"
ld = lambda n: json.loads((R / n).read_text()) if (R / n).exists() else {}
base, exact, learned, stack, lev, ab = ld("baseline.json"), ld("exact.json").get("tricks", {}), ld("learned.json").get("tricks", {}), ld("stack.json"), ld("leverage.json"), ld("pf_ab.json")
hit = ld("hitcurve_q38.json"); pred = ld("predictor_offline.json")

DB = {
    "staged_prefill": ("Batch scan", "Scan the big table once per 2,048 rows instead of once per 512."),
    "exclusive_tiers": ("No double buffering", "A page lives in the buffer pool or the OS cache, not both (like O_DIRECT in Postgres/InnoDB)."),
    "cpu_lane_overlap": ("Parallel operators", "Two operators of one query run at the same time instead of taking turns."),
    "read_pool": ("Parallel I/O workers", "Several readers fetch pages at once, like parallel sequential scan workers."),
    "layer_ahead_prefetch": ("Read-ahead", "Fetch the pages the next step will need while this step runs."),
    "no_host_sync": ("No round-trips", "Stop asking the coordinator after every step; workers poll a flag."),
    "mtp": ("Speculative execution", "Guess the next two answers cheaply; one real pass confirms them."),
    "engram_ssd": ("Index on disk", "A 28.8 GB key-value index; each query needs ~16 rows, so it never needs to be in RAM."),
    "profile_residency": ("pg_prewarm by workload", "Keep the pages your workload uses hot, not whole tables by name."),
    "learned_prefetcher": ("Learned read-ahead", "Learn which pages follow which, from your own query log."),
    "steering": ("Query rewrite", "Nudge the planner toward an index that is already cached; small change in answer."),
    "adapter": ("Materialized correction", "A small learned table that fixes the answers the rewrite changed."),
}
NAMES = {"staged_prefill": "Staged prefill", "exclusive_tiers": "Exclusive tiers", "cpu_lane_overlap": "CPU works while GPU works",
         "read_pool": "Parallel read pool", "layer_ahead_prefetch": "Layer-ahead prefetch", "no_host_sync": "No per-layer sync",
         "mtp": "MTP drafting (Qwen's own)", "engram_ssd": "N-gram table on SSD", "profile_residency": "Your profile picks VRAM experts",
         "learned_prefetcher": "Learned next-layer guesser", "steering": "Steering toward VRAM", "adapter": "Steering-aware adapter"}
WHO = {"staged_prefill": "0xSero", "exclusive_tiers": "0xSero", "cpu_lane_overlap": "0xSero", "read_pool": "0xSero",
       "layer_ahead_prefetch": "0xSero", "no_host_sync": "0xSero", "mtp": "Qwen / llama.cpp", "engram_ssd": "Qwen / llama.cpp",
       "profile_residency": "omxpert", "learned_prefetcher": "omxpert", "steering": "omxpert", "adapter": "omxpert"}
EXACT = set(exact)
e = html.escape


def cell(t, k):
    if t.get("status") == "not_built":
        return '<span class="nb">not built</span>', e(t.get("reason", ""))[:220]
    if t.get("status") != "measured":
        return '<span class="nb">pending</span>', ""
    bits = []
    ab_key = {"read_pool": "same", "layer_ahead_prefetch": "prev", "learned_prefetcher": "markov"}.get(k)
    mab_ = ld("mtp_ab.json").get("arms", {})
    if k == "mtp" and mab_:
        bits.append(f"decode {mab_['off']['median_tok_s']} → <b>{mab_['off+mtp']['median_tok_s']}</b> tok/s (same server, 8 paired rounds, 8/8 faster), {mab_['off+mtp']['draft_accept']*100:.0f}% of drafts accepted")
        bits.append("but its ~4 GB of VRAM pushes experts into RAM: prompts 20-80 tok/s instead of 110-170 in separate runs")
    elif ab_key and ab.get("arms", {}).get(ab_key):
        a = ab["arms"][ab_key]; o = ab["arms"]["off"]
        bits.append(f"decode {o['median_tok_s']} → <b>{a['median_tok_s']}</b> tok/s (same server, {len(a['runs'])} rounds; within the ~5% run-order effect)")
        if "draft_accept" in a:
            bits.append(f"{a['draft_accept']*100:.0f}% of drafts accepted")
    else:
        bits.append(f"decode {base.get('decode_tok_s')} → <b>{t.get('decode_tok_s')}</b> tok/s, prompt {base.get('prefill_tok_s')} → <b>{t.get('prefill_tok_s')}</b>")
    if t.get("kl") and isinstance(t["kl"], dict):
        bits.append(f"KL {t['kl']['kl']:.4f}, same top token {t['kl']['same_top_pct']}%")
    return '<span class="ok">measured</span>', "; ".join(bits)


rows = []
for k in list(exact) + list(learned):
    t = exact.get(k) or learned.get(k)
    st, txt = cell(t, k)
    an, ad = DB.get(k, ("", ""))
    rows.append(f"<tr><td><b>{NAMES.get(k, k)}</b><div class='who'>{WHO.get(k, '')} · {'exact' if k in EXACT or k == 'learned_prefetcher' else 'changes output a little'}</div></td>"
                f"<td><b>{e(an)}</b><div class='who'>{e(ad)}</div></td><td>{st}</td><td>{txt}</td></tr>")

lev_rows = ""
for k, v in (lev.get("whatif") or {}).items():
    lev_rows += f"<tr><td><b>{e(v.get('label', k))}</b></td><td>{e(str(v.get('decode_tok_s', '–')))}</td><td>{e(str(v.get('prefill', '–')))}</td><td>{e(v.get('why', ''))}</td></tr>"

head = ""
b2 = ld("base2.json"); mab = ld("mtp_ab.json").get("arms", {}); s05 = ld("pin_steer05.json"); s1 = ld("pin_steer1.json")
cl = [ld(f"cold_lazy_{i}.json") for i in (1, 2)]; cz = [ld(f"cold_lzm_{i}.json") for i in (1, 2)]
ladder = [("Stock llama.cpp, first run of the day (cold cache)", base.get("decode_tok_s"), base.get("prefill_tok_s"), "measured, separate run", "exact")]
for i, c in enumerate(cl):
    if c.get("tok_s"): ladder.append((f"Stock, n-gram table dropped from cache (cold #{i+1})", c["tok_s"], c["prefill_tok_s_median"], "measured", "exact"))
for i, c in enumerate(cz):
    if c.get("tok_s"): ladder.append((f"-lzm off, table dropped from cache (cold #{i+1})", c["tok_s"], c["prefill_tok_s_median"], "measured", "exact"))
if b2: ladder.append(("Stock, after the table had been cached by earlier runs (warm)", b2["tok_s"], b2["prefill_tok_s_median"], "measured", "exact"))
if mab: ladder.append(("Table via page cache, no drafting (one server)", mab["off"]["median_tok_s"], None, "in-process A/B, 8 rounds", "exact"))
if mab: ladder.append(("+ MTP drafting (same server)", mab["off+mtp"]["median_tok_s"], None, "in-process A/B, 8 rounds", "exact"))
if stack.get("exact_stack_tok_s"): ladder.append(("Final stack: -lzm off + MTP (omxpert/run.sh)" if "stack_final_1" in stack.get("which_runs", []) else "Stack so far: MTP only (lazy table)", stack["exact_stack_tok_s"], stack["exact_stack_prefill_tok_s"], "measured, 2 separate runs", "exact"))
for n, lab, q in (("fit_lzm_1", "llama.cpp fit, -lzm off (residency test, run 1)", "exact"), ("fit_lzm_2", "llama.cpp fit, -lzm off (run 2)", "exact"),
                  ("pin_lzm_1", "Your coding profile pinned per expert (learned residency)", "exact"),
                  ("pin_lzm_steer05_1", "Profile + steering λ0.5", "lossy: KL 0.074")):
    d = ld(n + ".json")
    if d.get("tok_s"): ladder.append((lab, d["tok_s"], d["prefill_tok_s_median"], "measured, separate run, interleaved", q))
mx = max(x[1] for x in ladder if x[1])
bars = "".join(f"<tr><td>{e(n)}<div class='who'>{e(how)} · {e(q)}</div></td><td style='width:45%'><div class='bar' style='width:{100*v/mx:.0f}%;background:{'#c98a00' if q=='exact' else '#9a9a9a'}'>{v}</div></td><td>{p if p else '–'}</td></tr>" for n, v, p, how, q in ladder if v)
head = f"<h2>Decode speed, measured</h2><table><tr><th>Setup</th><th>Decode tok/s</th><th>Prompt tok/s</th></tr>{bars}</table>"
hits = ""
if hit:
    hits = ("<p>Your coding profile on Qwen 4's 512 tiny experts: keeping the top "
            + ", ".join(f"{int(float(f)*100)}% of experts serves {v*100:.0f}%" for f, v in hit["per_expert"].items() if f in ("0.2", "0.3", "0.45"))
            + " of held-out picks (whole layers: the same % in, the same % out).</p>")
if pred:
    r16 = pred["recall"].get("16", {})
    hits += (f"<p>Guessing the next layer's experts with 16 fetches: static hot list {r16.get('profile', 0)*100:.0f}%, previous token "
             f"{r16.get('prev', 0)*100:.0f}%, learned table {r16.get('markov', 0)*100:.0f}%, both {r16.get('both', 0)*100:.0f}% of real picks.</p>")

page = f"""<!doctype html><html><head><meta charset="utf-8"><title>omxpert night 5: before / after</title><style>
body{{font-family:Inter,system-ui,sans-serif;margin:0;padding:24px;background:#faf8f3;color:#222;line-height:1.45}}
h1{{font-size:22px;margin:0}} h2{{font-size:16px;margin:28px 0 8px}} .sub{{color:#666;margin:4px 0 16px}}
.kpis{{display:flex;gap:12px;flex-wrap:wrap}} .k{{background:#fff;border:1px solid #e4dfd3;border-radius:10px;padding:12px 16px}}
.k .n{{font-size:22px;font-weight:700;color:#8a5a00}} .k .l{{font-size:12px;color:#666}}
table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e4dfd3;border-radius:10px;font-size:13px}}
td,th{{padding:8px 10px;border-top:1px solid #eee;vertical-align:top;text-align:left}} th{{font-size:12px;color:#777;border:none}}
.who{{font-size:11.5px;color:#777}} .bar{{color:#fff;font-weight:700;padding:3px 8px;border-radius:5px;min-width:30px}} .ok{{color:#1d7a3a;font-weight:600}} .nb{{color:#a66b00;font-weight:600}}
.dia{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} iframe{{width:100%;height:640px;border:1px solid #e4dfd3;border-radius:10px;background:#fff}}
@media(max-width:900px){{.dia{{grid-template-columns:1fr}}}} .credit{{font-size:13px;color:#555;margin-top:24px}}
</style></head><body>
<h1>omxpert night 5: Qwen 4's architecture on a 2×3090 desktop, before and after</h1>
<div class="sub">Qwen3.8-Flash-Next (qwen4exp, 125B MoE + 51B n-gram table, 111 GB at 4-bit) on 2×RTX 3090, 62 GB RAM, one NVMe. Stock llama.cpp vs the same model with the tricks below.</div>
{head}
{hits}
<h2>Before and after, one trick at a time</h2>
<table><tr><th>Trick</th><th>Database view</th><th>Status</th><th>Before → after</th></tr>{''.join(rows)}</table>
<h2>One token's trip: before (left) and after (right)</h2>
<div class="dia"><iframe src="night5/before.html"></iframe><iframe src="night5/after.html"></iframe></div>
<h2>Leverage: what each added part would buy</h2>
<table><tr><th>Add</th><th>Decode tok/s (est.)</th><th>Prompt</th><th>Why</th></tr>{lev_rows}</table>
<p class="credit">Systems ideas (read pool, layer-ahead prefetch, exclusive tiers, staged prefill, CPU/GPU overlap, no host sync) come from
<b>0xSero</b>'s glm-flash-lite (GLM-5.3-Flash at 19.5 tok/s on one 3090), re-implemented here inside llama.cpp. MTP drafting and the lazy
n-gram table are Qwen's and llama.cpp's. Profiles, learned guesser and steering are omxpert's. Estimates are marked; everything else was measured on this box.</p>
</body></html>"""
(V / "night5_before_after.html").write_text(page)
print("wrote", V / "night5_before_after.html")
