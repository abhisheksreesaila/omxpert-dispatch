# Expert Dispatch on Apple M4 (16 GB unified memory)
Model: Qwen3-Coder-30B-A3B Q4_K_M (18.6 GB > RAM, so experts page from the SSD). llama.cpp + EDX patch, Metal, `-nr -fit off -lm mmap`.
Each config: 1 warm-up prompt (dropped) + 4 held-out coding prompts x 128 tokens. Script: scripts/mac_bench.sh; raw: results/mac/bench.log.

## 2026-11-05 run 1 (11:35-12:20)
| Config | ms/token | tok/s |
|---|---|---|
| pinned coding 6 GB + pageout + steer λ=1 | **115.7** | **8.6** |
| all experts on CPU (mmap, macOS LRU) | 172.3 | 5.8 |
| pinned coding 4 GB + pageout | 213.8 | 4.7 |
| pinned coding 6 GB + pageout | 214.9 | 4.7 |
| pinned coding 4 GB | 219.2 | 4.6 |
| pinned CHAT profile 6 GB + pageout (wrong profile) | 336.5 | 3.0 |
| 24 of 48 layers' experts on Metal (llama.cpp usual) | 342.4 | 2.9 |
| llama.cpp LRU expert cache 4 GB | 559.2 | 1.8 |
| everything on Metal | failed (all runs) | - |

Findings:
- On unified memory, pinning a hot set into a Metal buffer does NOT beat letting macOS cache the mmap (214 vs 172 ms). The copy takes RAM from the page cache and the hot/cold split runs both passes in sequence.
- Steering toward the pinned set is the win: 1.49x faster than the best plain layout, 3.0x faster than the usual layer split. Quality on the Mac not measured yet (Linux: +1.2-1.7% ppl at λ=1).
- The right profile matters: coding profile 215 vs chat profile 337 ms (1.57x).
- llama.cpp's LRU expert cache is the worst option here (memcpy churn).
- EDX_PAGEOUT: no measurable effect (214 vs 219).
- Mac went to sleep after the run (caffeinate ended), so the next runs need it awake.
- 2026-11-05 12:56 Mac run 2: double launch at 12:39/12:40 (first launcher had launched before its SSH dropped); Mac then flapping/offline (SSH connects, then times out mid-command: looks like sleep with lid closed or on battery). Run-2 state on the Mac unknown; scripts/mac_fix2.sh restarts exactly one clean run when it stays reachable.
- 2026-11-05 13:17 Run 2 steer-only runs crashed: copy_experts (large-batch expert copy) built a view over the 1-slot bank. Fixed (copy_experts returns 0 in steer-only), not yet rebuilt on the Mac. Killing the run also killed its caffeinate, and the Mac went to sleep / unreachable at 13:14. Lesson: start a standalone caffeinate before stopping a job that holds the only one.

## 2026-11-05 run 2 (clean, 13:20-14:10): steering without copies, and quality
Steer-only (`EDX_STEER_ONLY=1`): the profile only marks the hot set (by `--moe-cache-mib`), nothing is copied, every expert reads from the mmap. `EDX_PIN_MINPL=16`. 4 coding prompts x 128 tokens.

| Config | ms/token | tok/s | vs all-experts-from-file |
|---|---|---|---|
| steer-only 6 GB, λ=2 | 99.3 | 10.1 | 2.03x |
| pinned 6 GB copy + steer λ=1 | 110.5 | 9.1 | 1.82x |
| steer-only 6 GB, λ=1 | 122.9 | 8.1 | 1.64x |
| steer-only 4 GB, λ=1 | 125.1 | 8.0 | 1.61x |
| steer-only 8 GB, λ=1 | 130.2 | 7.7 | 1.55x |
| steer-only 6 GB, λ=0.5 | 156.3 | 6.4 | 1.29x |
| all experts from file (baseline) | 201.3 | 5.0 | 1.00x |

Quality, coding held-out perplexity (12 x 512 tokens, -ub 8 so steering applies):
| Config | ppl | change | s/pass |
|---|---|---|---|
| plain | 3.138 | - | 51.5 |
| steer-only λ=0.5 | 3.158 | +0.6% | 33.8 |
| steer-only λ=1 | 3.172 | +1.1% | 23.2 |
| steer-only λ=2 | 3.327 | +6.0% | 17.4 |

Findings:
- Steering is the lever on unified memory: λ=1 is 1.64x faster for +1.1% perplexity; λ=0.5 1.29x for +0.6%; λ=2 2.0x for +6%.
- Steer-only (no copy) is NOT better than copy+steer: 122.9 vs 110.5 ms at λ=1. With steering on, the copied hot set does pay (~10%), unlike plain pinning in run 1. Idea 34 rejected as a speed win; it is the zero-memory option.
- Hot-set size barely matters at λ=1 (4/6/8 GB: 125/123/130 ms): the nudge, not the size, makes the gain.
- Baseline drifted between runs (172 run 1 vs 201 run 2, ~15%): compare within a run.
- pin6g+steer1 perplexity not measured separately; same hot set and λ as steer-only λ=1, so routing is the same (pinned mode matched plain ppl on Linux).

## 2026-11-05 run 3: skip-the-miss quality (P6) and the batching prototype (Coder-30B)
P6 skip quality (coding perplexity, 12 x 512, -ub 8; hot set = coding profile 6 GB; plain = 3.138; steer λ=1 = 3.172):
| Denominator | ppl | change |
|---|---|---|
| top-8 (selected) mass | 3.170 | +1.0% |
| survivors' mass | 3.188 | +1.6% |
| top-16 mass (handbook's pick) | 3.247 | +3.5% |
| steer λ=1 + skip (top-16) | 3.553 | +13.2% |
Here (drop only 1-2 misses, never the top-2) the plain top-8 denominator is best, unlike the handbook's static top-4 setting. Stacking skip on steering hurts. Skip speed not measured cleanly (ran during a 63 GB download).

Batching (owner's idea 38): llama-batched-bench, 128-token prompts, 64 generated tokens per job.
| Jobs at once | exact: gen tok/s total | steer-only λ=1: gen tok/s total |
|---|---|---|
| 1 | 6.8 | 10.0 |
| 2 | 9.6 | 11.6 |
| 4 | 10.5 | 18.0 |
| 8 | 14.0 | 26.6 |
- 8 jobs at once: 2.1x the throughput of one job (exact), 2.65x with steering. Steering + batching = 3.9x the exact single-job speed, on a 16 GB Mac with a model larger than RAM.
- Prompt processing does not batch-scale past 4 jobs here (27 -> 23 tok/s at 8).

## 2026-11-05 run 4: gpt-oss-120b (MXFP4 as released, 63 GB, 36 layers x 128 experts, top-4) on the 16 GB M4
Coding profile: traced on the Mac, 25,363 coding tokens in 20 min; the top 32 of 128 experts per layer cover 74.6% of routings.
3 coding prompts (1,200 chars), 96 tokens; 6 GB hot set (steer-only), `EDX_PIN_MINPL=8`.
| Config | ms/token | tok/s | speed-up | coding ppl (4 x 512) |
|---|---|---|---|---|
| exact model | 963 | 1.04 | 1.0x | 3.787 |
| steer λ=1 (BUGGY for this router) | 347 | 2.88 | 2.8x | 4.262 (+12.6%) |
| steer λ=2 (BUGGY) | 171 | 5.8 | 5.6x | not measured |
Batching, exact model (gen tok/s total): 1 job 1.99, 2 jobs 2.21, 4 jobs 2.87, 8 jobs 2.77: only 1.44x, it plateaus at 4 (top-4 of 128 experts: 8 jobs share fewer experts than on Qwen).

BUG found: gpt-oss selects experts on raw router logits (softmax_weight gating). EDX_STEER multiplied the scores by e^-λ, which raises negative logits, so it partly steered toward the wrong experts. Fixed: for logit routers, subtract λ from non-pinned logits (the same nudge in log space). Qwen (softmax gating) results are unaffected. Run 5 re-measures gpt-oss steering with the fix.

## 2026-11-05 run 5: gpt-oss-120b, steering fixed (logit - λ)
| Config | ms/token | tok/s | speed-up | coding ppl (4 x 512) |
|---|---|---|---|---|
| exact model (run 4) | 963 | 1.04 | 1.0x | 3.787 |
| steer λ=0.5 | 661 | 1.51 | 1.46x | 3.862 (+2.0%) |
| steer λ=1 | 338 | 2.96 | 2.85x | 4.100 (+8.3%) |
| steer λ=2 | 148 | 6.76 | 6.5x | 7.046 (+86%, unusable) |
Batching with λ=1 (gen tok/s total): 1 job 3.0, 2 jobs 4.2, 4 jobs 5.9, 8 jobs 7.0 = 3.5x the exact single-job rate (1.99).
- The fix helped quality at λ=1 (4.262 -> 4.100) at the same speed.
- gpt-oss is far more sensitive than Qwen: top-4 routing means each swapped expert is a quarter of the layer (Qwen top-8: an eighth). λ must be set per model: λ=0.5 here ~ λ=1 on Qwen in quality cost.
- Perplexity on 4 chunks only (2,048 tokens): treat ±1-2% as noise.

## 2026-11-05 run 6: speculative decoding on gpt-oss-120b (lossless)
Smoke test, EAGLE3 head (eagle3-gpt-oss-120b-Q8_0, 0.85 GB), draft 4, plain-text prompt, cold cache, 35 tokens:
decode 0.558 tok/s (vs 1.04 exact), accept 35.7% (20 of 56 drafted), ~2.5 tokens per verify step, ~4.5 s per step.
A 5-token verify cost ~4.5x one token: consecutive tokens of one sequence share few of their top-4 experts, so verifying does not share SSD reads the way 8 parallel jobs do.
Run 6 grid (`mac_spec.sh`, chat-format prompt `results/harmony_code.txt`, 96 tokens, warm cache): results below when done.
Run 6 grid results (19:20-19:59, chat-format coding prompt, 52-token prompt, ~100 tokens generated, temp 0):
| Config | decode tok/s | vs exact | accept |
|---|---|---|---|
| exact (3 runs: 1124, 1127, 1103 ms/token) | 0.89 | 1.0x | - |
| EAGLE3 draft 2 | 0.663 | 0.75x | 68.7% |
| EAGLE3 draft 4 | 0.540 | 0.61x | 50.0% |
| EAGLE3 draft 8 | 0.237 | 0.27x | 20.6% |
| steer λ=0.5 alone | 1.02 (980 ms) | 1.15x | - |
| EAGLE3 draft 4 + steer λ=0.5 | 0.552 | 0.62x | 42.9% |
| EAGLE3 draft 2 + steer λ=1 | 0.898 | 1.0x (steer λ=1 alone was 2.96 in run 5) | 48.5% |
| gpt-oss-20b as draft, 4 | FAILED: Metal out of memory (kIOGPUCommandBufferCallbackErrorOutOfMemory) | | |
- Speculative decoding LOSES on this Mac at every setting, even with 69% acceptance. Draft 2: ~41 verify steps of 3 tokens at ~3.6 s each = ~3.2x the cost of one token. Verifying k tokens costs about k tokens (no shared reads: consecutive tokens pick different top-4 experts), so there is nothing to win.
- Steering does not rescue it: steering + EAGLE3 is slower than steering alone.
- Two models at once does not fit: 120B + 20B ran Metal out of memory. A chat lane must take turns with the 120B, not run beside it.
- steer λ=0.5 gave only 1.15x on this short chat prompt (1.46x in run 5 on 1,200-byte coding prompts): the speed-up depends on the prompt.
