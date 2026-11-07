# Night 5 (daytime autopilot, 2026-11-06 from 16:00): his tricks + ours on Qwen 4's architecture

**End goal (owner's words, condensed):** take all the tricks learned from 0xSero and from us, try them tonight on the NVIDIA
box with Qwen3.8-Flash-Next ("that's the four"), and show before/after on the exact model, so people can download it and
use it reliably; then find where added hardware (RAM, SSD, GPU, NPU, Mac) would give leverage: the seed of a
"Hyper-V for inference" that pools commodity hardware. Easiest thing that proves the hypothesis; Mojo comes later.

## Milestones (each checked by `scripts/check_night5.sh`)
1. **Baseline:** stock llama.cpp runs Qwen3.8-Flash-Next (UD-Q4_K_XL, 111 GB) on 2×3090 + RAM; decode, prefill, RAM
   recorded; KL reference logits saved. (`results/night5/baseline.json`)
2. **Exact tricks** (output unchanged): every trick from his list and ours has a row: measured, or not built with a reason;
   at least 3 measured against the baseline on the same server harness. (`results/night5/exact.json`)
3. **Learned tricks** (output may change): profile residency, learned prefetcher, steering, adapter; at least 3 measured,
   each lossy one with KL to the baseline. (`results/night5/learned.json`)
4. **The stack people can run:** best exact-only stack vs baseline (speed + KL) and `omxpert/run.sh` that starts it;
   smoke test answers a prompt. (`results/night5/stack.json`)
5. **Leverage:** cost model checked on the baseline (≤30% error), then what each added part buys: second NVMe, 128 GB
   RAM, eGPU, NPU, the Mac. (`results/night5/leverage.json`)
6. **Report:** `docs/report_night5.md` (credits 0xSero), before/after visual `docs/visuals/night5_before_after.html`,
   IDEAS entry 58+.

**Done means:** `scripts/check_night5.sh` prints PASS ×6.

## Out of bounds
No deleting models or caches (the 122B stays unless the owner says so), no sudo, no paid services, no secrets, nothing
pushed or published (the public repo gets a local commit only), nothing outside ~/expert-dispatch and ~/models.
Shared box: `nice -n 10`, keep RAM available ≥ 8 GB, stop if free disk < 20 GB or a GPU > 85 °C for 5 min.

## Assumptions (instead of asking)
- Model: Qwen3.8-Flash-Next UD-Q4_K_XL with the bundled IQ4_NL table (fits disk; lossless BF16 table needs the 122B
  deleted). Fallback Qwen3.6-35B if it won't run.
- "Full intelligence" reference = the same Q4 file run by stock llama.cpp; exact tricks must match it (KL ≈ 0).
- Tools: our llama.cpp tree (EDX patch) + small patches; no 0xSero code (his image needs root + disk), his ideas
  re-implemented and credited.
- Engine-level tricks that llama.cpp's scheduler can't express in hours (true CPU/GPU overlap within a layer, no
  per-layer host sync) are recorded as "not built" with what it would take, and costed in the model.

## Budget
80 turns or 8 hours, whichever first.
