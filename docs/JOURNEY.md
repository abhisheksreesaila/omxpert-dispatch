# omxpert dispatch: the research journey, day by day

> **An omxlabs research project.** The same page on the site: [journey.html](https://abhisheksreesaila.github.io/omxpert-dispatch/journey.html). Cover page: [index.html](https://abhisheksreesaila.github.io/omxpert-dispatch/).

## The idea

A mixture-of-experts model wakes only a few experts per token, and one person's work keeps waking the same ones. The idea: **put a small learned model next to the big one** that predicts which experts your work will use. Keep those in fast memory and leave the rest in RAM or on disk. A huge model on a desktop, at full intelligence, with less memory.

This page is how that idea held up, one day at a time. Each day says what we tried, what worked, what failed, what we took back, and the one lesson we kept.

- 🧪 Every number comes from the lab notes. Estimates and simulations are marked.
- ❌ A dead end is a result. It goes in the log with the number that killed it.
- ↩️ When a published number turned out wrong, we corrected it before re-measuring.

Key: ✅ worked · ❌ failed · ↩️ retracted · 💡 the lesson · 📄 lab notes

---

## Day 1 · Nov 3–4 (night 1) 🔍 Is the way you use experts personal at all?

**Tried.** Traced which experts fire on the owner's own work (coding, chat, math, tool calls, docs) on three models: OLMoE-1B-7B in PyTorch, Qwen3.6-35B-A3B in llama.cpp, and gpt-oss-20b in Modular MAX. Replayed caches in Mojo to compare a fixed profile with LRU and with the Belady ceiling. Then nudged the router toward experts already loaded (steering).

✅ **Worked**
- **Profiles are personal on fine-grained models.** Qwen3.6-35B (256 experts per layer): your own coding profile beats one built from your other work by **+11 points** at a 25% cache (66.7% vs 55.3%). On OLMoE (64 experts) the gap was only 1.8.
- **Steering is the biggest lever.** OLMoE, 25% of experts resident: hits go from 69% to **94% for +3.2% perplexity**.
- Pinned profile vs llama.cpp's layer split, 35B, 30% of experts in VRAM: **19.1 vs 24.7 ms** per token, 23% faster.
- Live profile swap mid-generation in **447 ms**, output coherent.

❌ **Failed**
- The go test "profile ≥ LRU + 10 points" **failed by −12 to −28 points**. Consecutive tokens reuse experts, so recency beats a long-run profile on hit rate.
- A hard "resident only" mask: +45% perplexity at 25% resident. A soft nudge works; a hard wall doesn't.
- A learned bias per expert landed on the same trade-off curve as one hand-set λ.
- Self-speculative decoding on Qwen3.6: 85–99% accepted, but 30–41 tok/s vs 46–54 plain.
- MAX: tracing hooks fixed and proven, but gpt-oss-20b ran out of memory at load on 2×3090.

↩️ **Retracted**
- Nothing published was taken back. One benchmark was thrown away before it counted: another session pushed the load average to 21, so its timings were invalid.

💡 **The lesson:** don't just cache, steer. Steering moved hit rate by 15–25 points; the best cache policy moved it by 1.5–3.

📄 [NIGHT1_MORNING_REPORT.md](logs/NIGHT1_MORNING_REPORT.md) · [NIGHT1_LOG.md](logs/NIGHT1_LOG.md)

---

## Day 2 · Nov 4–5 (night 2 + the Mac) 🌙 Does the slowest expert set the pace? And does a 120B fit on a 16 GB laptop?

**Tried.** Night 2's runbook predicted that decode time tracks the number of layers with any miss (P1), plus four placement ideas (P2–P5). The machine was busy with other sessions all evening (load average up to 43), so the owner stood the work down at 22:41, and the hourly checker stopped silently after 00:44. The morning report said "not measured". At 08:51 the owner said run now. Then the work moved to a 16 GB MacBook (Apple M4) running gpt-oss-120b, a 63 GB model, four times its memory.

✅ **Worked**
- **Costs measured:** one CPU expert read 0.081 ms; an uploaded expert must be reused ≥2.9 times to beat a CPU read.
- Steering in llama.cpp, Qwen3-Coder-30B, λ = 1: **30–37% faster** than llama.cpp's layer split for +1.2–1.7% perplexity.
- **gpt-oss-120b on the 16 GB Mac:** exact 1.04 tok/s; steering λ = 0.5: **1.46× for +2.0%** perplexity; λ = 1: **2.85× for +8.3%**.

❌ **Failed**
- **P1 not supported.** Per-token time tracks total misses (R² 0.875), not layers with a miss (0.288).
- P2, set-completing pins, equalled plain frequency pins in simulation: no gain.
- Speculative decoding on the Mac (EAGLE3 head): **0.27–0.75×** of plain decoding. Each token's experts barely overlap, so checking k tokens costs about k tokens.
- Plain pinning on the Mac lost to macOS's own page cache.

↩️ **Retracted**
- **Batching "7.0 tok/s for 8 jobs"** came from a benchmark that feeds random tokens. Eight real coding jobs ran at ~1.0 tok/s total: different jobs touch more experts than fit in RAM. Corrected on the site the same day.
- The night-2 morning report ("not measured") was overtaken by the run later that morning; the log has the real P1 verdict.

💡 **The lesson:** measure on real text, on a quiet machine. A random-token benchmark, or a run next to someone else's build, gives a number that isn't yours.

📄 [NIGHT2_MORNING_REPORT.md](logs/NIGHT2_MORNING_REPORT.md) · [NIGHT2_LOG.md](logs/NIGHT2_LOG.md) · [MAC_LOG.md](logs/MAC_LOG.md) · [IDEAS.md](logs/IDEAS.md)

---

## Day 3 · Nov 5–6 (night 3, autopilot) 🧠 Can the expert set be learned instead of counted?

**Tried.** On small models we can train, two learned steps: a swap search that keeps a swap only if the model gets closer to the full model (KL), then a 2M-parameter adapter trained for that set. Routing never leaves the loaded experts. On the Mac, a sweep of how many experts per layer the 120B needs, and real graded coding tasks.

✅ **Worked**
- **OLMoE, 8 of 64 experts:** counted set 47.8% next token right → learned set 57.9% → learned set + adapter **66.2%** (full model 69.1%).
- **Second model family,** Granite 3B, 8 of 40: 33.7% → 43.8% → **57.6%** (full 68.7%).
- A usable bundle in **~90 s** of compute.
- 120B on the Mac: 8 experts per layer at λ = 1, **2.45× for +5.4%**. The right profile costs +8.7%, the wrong one +34.5%. 5,000 tokens of history (~4 min of profiling) is enough.
- Real graded coding tasks, steered 120B: **7/8 and 8/8** pass hidden tests.

❌ **Failed**
- Gradient-learned gates: relaxed gates never fully close, and the strict set they pick collapses.
- Taylor importance ignores that masked tokens re-route; much worse.
- Co-optimising the search with the adapter found no improving swap. Search first, then adapt.
- Renormalising weights over loaded experts: coding perplexity 47.2 vs 8.7.
- 4 real jobs at once were slower than one at a time (1.00 vs 1.36 tok/s).

↩️ **Retracted**
- **The first 4-of-64 numbers leaked.** With fewer loaded experts than the model picks (8), the mask filled the gap with unloaded experts at full weight. Found at 03:00, the live site was corrected at once, the code fixed, and 4-of-64 re-measured: **62.3%, not 64.0%**. 8 and 16 of 64 were unaffected.
- Still open, not hidden: the adapter also lowers perplexity below the full model's, and we can't yet explain why.

💡 **The lesson:** correct the public number first, then re-measure. And check that a mask really masks.

📄 [NIGHT3_REPORT.md](logs/NIGHT3_REPORT.md) · [NIGHT3_LOG.md](logs/NIGHT3_LOG.md) · `decisions_night3.md` (not public yet)

---

## Day 4 · Nov 6 (daytime) 🏗️ Could a ~300B model run on this desktop, with a fast lane and a background lane?

**Tried.** Qwen3.5-122B as test material on 2×3090: per-expert placement vs llama.cpp's per-layer placement, two lanes on one model, the SSD tier, and a cost model for GLM-5.3-Flash checked against the 122B. The bar: 0xSero runs GLM-5.3-Flash at 19.5 tok/s on one 3090.

✅ **Worked**
- **The second GPU as a compute tier** (35B): 2.3× faster than experts in RAM. Pinned uploads: 1.35× over no cache.
- **Two lanes on one 122B:** fast lane alone 11.2 tok/s; when the background job yields in 16-token chunks, the fast lane keeps **12.3**, the same as alone.
- SSD measured: 2.8 GB/s one read at a time, 4.9 GB/s with 4+ readers; llama.cpp's page faults deliver ~2.5 GB/s.
- Coding profile on the 122B: 56% of experts resident serve 84% of picks (whole layers: 56%).

❌ **Failed**
- **Per-expert placement lost 2.4–3.6× by default.** Big prompt batches read every host expert: 67 GB through ~31 GB of free RAM. With 8-token batches it only tied (8.49 vs 8.14 tok/s).
- Three measurement attempts failed before the fourth worked.
- 0xSero's image couldn't run here (needs root and ~90 GB more disk), so his tricks were measured as costs.
- Projection, an estimate: his style of engine here ~8 tok/s on one GPU, ~11 on two; llama.cpp today ~0.6.

↩️ **Retracted**
- The cost model's page-fault read rate was a guess (0.95 GB/s) until capped-RAM runs measured 2.5 GB/s. The 122B check passed only after that change, and the change is logged.
- ~10 btrfs "corrupt" counts were false alarms from our own SSD test (two direct reads into one buffer). The flagged regions re-read cleanly.

💡 **The lesson:** after two failures, test the premise. All three bad attempts assumed a fresh process per prompt measures placement. At the RAM edge it measures page churn; a long-running, warmed server measures placement.

📄 `docs/report_night4.md` · `docs/journal_night4.md` · `docs/decisions_night4.md` (not public yet)

---

## Day 5 · Nov 6 (16:00–21:00) 🚀 Does Qwen 4's architecture run on a desktop, and which tricks actually help?

**Tried.** Qwen3.8-Flash-Next (the Qwen 4 preview architecture: a 125B MoE plus a 51B-value n-gram table, 111 GB at 4-bit) on 2×3090 with ~35 GB of free RAM. Every trick from 0xSero and from us, measured against stock llama.cpp on the same box; lossy ones scored with KL.

✅ **Worked**
- **It runs bit-exact** to stock llama.cpp at **~21–24 tok/s** decode and ~130–170 tok/s prompt. The first run of the day was 11 / 28.
- **MTP drafting** (Qwen's own draft head): 20.3 → 27.6 tok/s, ×1.36, 8/8 paired rounds, 86% of drafts accepted. It costs prompt speed (20–80 tok/s instead of 110–170), so it's opt-in.
- Reading the n-gram table at load (`-lzm off`): bit-exact, low page faults in 8/8 runs (stock 4/11).
- **The profile concentrates:** 30% of experts serve 80% of coding picks.
- Learned next-layer guesser, offline: 52% of next-layer picks with 16 guesses (fixed profile 26%, previous token 36%).
- A launcher people can run: [`omxpert/run.sh`](../omxpert/run.sh).

❌ **Failed**
- **Pinning the coding profile per expert inside llama.cpp:** 20.3 vs 21–22 tok/s, and prompts 4–5× slower.
- Read pool, layer-ahead prefetch, and our learned guesser online: all within noise. 99.97% of expert pages were already in RAM.
- Steering: ~+10% (23.9 vs 21–22 tok/s) for KL 0.074 against a 0.027 floor. Small, and lossy.
- Our cost model over-predicted the pinned run by 2×.

↩️ **Retracted**
- **"Steering is 2× faster":** compared against a page-fault-heavy first run. Clean reruns: ~+10%.
- **"`-lzm off` = 4× faster prompts":** warm stock reaches the same speed. Kept as the robust default, claim withdrawn.
- The first MTP A/B was invalid: the server ignored the per-request setting, so both arms drafted. Fixed and rerun.
- Prefetch "wins" of +1–3% sat inside a run-order effect: the second arm won 8/8 rounds even when both arms were identical.

💡 **The lesson:** separate runs lie: the same stock setup measured 11 tok/s at 16:06 and 23 at 18:34. Use one warmed server, switch tricks per request, alternate the order. And on this box the bottleneck is the CPU's expert math (32 of 49 ms per token), not the disk.

📄 [report_night5.md](logs/report_night5.md) · [journal_night5.md](logs/journal_night5.md) · [decisions_night5.md](logs/decisions_night5.md) · [GOAL_NIGHT5.md](logs/GOAL_NIGHT5.md) · [before and after, visually](https://abhisheksreesaila.github.io/omxpert-dispatch/night5_before_after.html)

---

## Where it stands

- **It runs, exactly.** Qwen3.8-Flash-Next (Qwen 4 preview architecture, 111 GB at 4-bit) runs bit-exact on 2× RTX 3090 + 64 GB RAM at ~21–24 tok/s decode in llama.cpp.
- **The learning works.** 30% of the experts serve 80% of coding picks.
- **But it isn't faster yet.** Inside llama.cpp, pinning those experts did not make it faster (20.3 vs 21–22 tok/s). It needs an engine where the CPU computes the cold experts while the GPU computes the hot ones, at the same time.
- **Day 6, next:** that engine, built on [FreeToken](https://github.com/FlashML-org/FreeToken)'s hybrid CPU+GPU engine (FlashML-org/FreeToken, Apache-2.0), with our profile deciding which experts sit on the GPU.
- **Prior art.** Expert offloading, profile-driven placement, learned prefetch and router bias toward cached experts all have prior work, listed in [PRIOR_ART.md](PRIOR_ART.md). Our angle is narrower: a personal, per-task profile plus learned prediction, tested end to end on consumer hardware, and measured honestly, including what failed.

## Credits

- **[0xSero](https://github.com/0xSero), glm-flash-lite.** 0xSero runs GLM-5.3-Flash at 19.5 tok/s on one RTX 3090. The systems tricks we re-implemented inside llama.cpp are his: the parallel read pool, layer-ahead prefetch, exclusive VRAM/RAM tiers, staged prefill, CPU/GPU overlap, and no per-layer host sync. Two of them (overlap, no host sync) need an engine change and are not built yet, so his core speedups are still untested here. **The systems tricks are his; the profiling, the learned prediction (expert sets, adapters, next-layer guesser) and the steering are ours.**
- **The Qwen team:** the model, its MTP draft head and the n-gram table.
- **[llama.cpp and ggml](https://github.com/ggml-org/llama.cpp):** the engine every run used.
- **[unsloth](https://huggingface.co/unsloth):** the GGUF quants.
- **FlashML / FreeToken authors:** the hybrid CPU+GPU engine the next experiment builds on.
- **The prior-art authors** in [PRIOR_ART.md](PRIOR_ART.md): ProMoE, MoE-Infinity, PowerInfer, KTransformers, Cache-Prior, ReMoE, DoMoE and the rest.
