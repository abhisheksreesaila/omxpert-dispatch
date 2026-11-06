# Expert Dispatch: idea log

Every idea tried or queued, with status and the number that decided it. Newest at the bottom of each section.
Status: TESTED (with result), RUNNING, QUEUED, PARKED (why).

## Placement
1. **Static profile in VRAM, judged by hit rate**: TESTED. Loses to LRU by 12–28 points on Qwen (fine-grained), ties on OLMoE. Hit rate turned out to be the wrong metric (see 3).
2. **llama.cpp built-in SLRU cache (`--moe-cache-mib`)**: TESTED on 122B. 58% hits at 8 GB, but **slower** than no cache (~10 vs ~13 tok/s): every miss is uploaded over PCIe (~0.8 ms/expert), more expensive than computing it on the CPU from RAM.
3. **Pinned profile, misses computed in RAM, never upload** (`EDX_PIN`, llama.cpp patch): TESTED on 122B. 70.9% VRAM hits, 0 uploads, correct output, but ~15 tok/s vs ~14 today: no clear win. The SSD paging noise on 122B (62 GB RAM for a 72 GB model) hides the effect. RUNNING: clean test on the 35B (all in RAM).
4. **Release pinned experts' RAM pages** (`madvise(MADV_PAGEOUT)` after pinning): BUILT, part of the 35B run. It frees page cache for the cold experts, which matters most at the RAM edge.
5. **No dummy expert read**: hot rows in the CPU pass are routed to a cold expert the token already uses. BUILT.

## Routing
6. **Router steering toward resident experts (−λ on non-resident)**: TESTED on OLMoE. Coding: 69 → 94% hits for +3.2% perplexity. Chat costs ~3× more; λ = 0.5 is the safe setting.
7. **Learned router bias per expert** (`learn_cache.py`): TESTED. Same trade-off curve as a single hand-set λ, so no win. Only re-picking the resident set helped (+2.7% perplexity at 93% hits).

## Prediction
8. **Session predictor (first 128 tokens)**: TESTED, no gain over the profile.
9. **Profile + recency EMA**: TESTED, best online policy on OLMoE, below LRU on Qwen.
10. **Cross-layer prefetch from a co-occurrence table**: TESTED in simulation. +3.8 points of hits, but only 11% of prefetches get used. Too blunt; it needs the real next-layer router on the current hidden state (Fate).

## Results from the pinned-profile benchmarks (clean, 35B fully in RAM)
11. **Pinned profile vs llama.cpp, equal VRAM, coding prompts**: TESTED.
    - 15% of experts in VRAM: pinned 22.4 ms vs whole layers 26.5 vs built-in cache 32.0.
    - 30% of experts: pinned 19.1 vs 24.7 vs 23.1, so **23% faster than today's layer split and 17% faster than the built-in cache.**
    - With the wrong profile (coding profile on chat): no gain. The profile has to match the use case.
12. **Profile per use case, loaded at session start** (owner's idea): QUEUED (`bench35b.sh`). Chat profile on chat, coding on coding, a mixed profile on both, at 15/30/60% VRAM.

## Self-speculative decoding (owner's idea)
13. **Fast lane drafts, full model verifies**: the draft is the same model with routing kept to VRAM-resident experts; the full router checks k drafted tokens in one batch. Output stays exactly the full model's. TESTED on OLMoE coding (acceptance = draft's greedy token equals the full model's):

    | Resident experts | λ = 1 (soft) | λ = 2 | resident only |
    |---|---|---|---|
    | 50% | 92.3% | 87.7% | 85.9% |
    | 25% | 90.3% | 82.2% | 76.0% |
    | 12.5% | 91.0% | 83.1% | 52.4% |

    - At 90% acceptance, k = 4 gives ~4.1 tokens per verify pass. The verify pass costs one batched read of the union of experts for 5 tokens.
    - NEXT: the cost model for a verify pass from Qwen traces (union of experts over k+1 consecutive tokens, minus the pinned set), then a llama.cpp prototype (two contexts on one model, the draft context with pinned-only routing).

## Pooled RAM across machines (owner's idea, for later)
14. **Pool cheap DDR4 boxes as the slow tier** (512 GB–1 TB of used server RAM is cheap). Key insight from tonight: **compute where the weights live.** Don't ship expert weights over the network, ship the activations: a few KB per token per layer. Each remote box holds cold experts in its RAM and runs them on its own CPU.
    - The network then carries KB, not MB, per expert. The cost is latency per layer hop (TCP ~50–100 µs, RDMA ~5–10 µs), not bandwidth.
    - Prototype path: llama.cpp `rpc-server` on the other box plus `-ot` to place chosen experts' tensors on the RPC device. A later version uses per-expert placement (the pinned patch generalized to N tiers: GPU, local RAM, remote RAM).
    - Deeper: RDMA (used ConnectX-3/4 cards are cheap), batching the hops of the slow lane, NUMA-aware placement, huge pages, `io_uring` for SSD tiers.
    - Fits the two lanes: the fast lane stays local; the slow lane (overnight agents) can fan out to the RAM pool.
    - Can test hop overhead today with `rpc-server` on localhost.
15. **Self-speculative decoding, built in llama.cpp** (`EDX_DRAFT=1`): the draft is the same GGUF loaded again (mmap, so no extra RAM), with routing masked to pinned experts. It never runs an expert on the CPU. TESTED on 35B coding at 30% VRAM:
    - **real acceptance 76.7%, 4.07 tokens per round** with k = 4. llama.cpp's own "accept = 100%" counter is wrong for hybrid models: it counts after checkpoint restore.
    - Qwen3.6 has recurrent layers, so each rejection costs a checkpoint restore.
    - Speed RUNNING (`bench_spec.sh`, k = 2/3/4/6 vs plain pinned, behind a "machine is quiet" gate).
    - OLMoE chat acceptance is much lower (32% vs 76% at 25% resident), so speculation suits the coding/background lane.

## Next ideas (queued, not run)
16. **Swap the profile at session start or task switch.** Re-pinning 5.6 GB over PCIe takes ~0.5 s and 20 GB ~1.6 s: cheap at a session boundary. Implementation: re-read `EDX_PIN` on a signal (SIGUSR1) or a server endpoint, then re-upload the slots. That is the owner's "set of experts per use case, loaded per session".
17. **Upload admission by expected reuse.** Uploading costs about 3–10× a CPU read, so admit an expert into VRAM only when its predicted near-term uses outweigh the upload. That's a small learned predictor, Belady-style. It's the learnable part on top of the pinned profile: keep most slots pinned and a few adaptive.
18. **Cheaper draft:** top-4 instead of top-8 experts in the draft, or skip some layers (early exit). The draft costs ~8 ms per token on the 35B now.
19. **Speculation on the 122B:** CPU reads dominate there, so batching the verify pass pays more. Test when RAM allows.
20. **Mix whole layers with pinned experts:** each CPU-touching layer has a fixed hop cost of ~0.15 ms, so the best plan probably puts a few hottest layers fully on GPU and pins experts in the rest. It's a knapsack with a per-layer cost.

## Results, evening run 2 (35B, ms/token, mean of 3 prompts)
- **Use-case profile vs mixed profile vs llama.cpp** (the owner's per-use-case idea):

  | VRAM | Prompts | Whole layers | Built-in cache | Own profile pinned | Mixed profile |
  |---|---|---|---|---|---|
  | 15% | coding | 28.4 | 34.1 | **21.5** | 25.9 |
  | 15% | chat | 29.4 | 36.8 | **26.6** | 26.6 |
  | 30% | coding | 21.3 | 21.1 | **18.9** | 19.4 |
  | 30% | chat | 21.4 | **19.7** | 20.2 | 22.3 |
  | 60% | both | **14.6** | 16.5 | 16.4 | 16.7 |

  - Use-case profiles win on small GPUs: coding −24% at 15% VRAM. The task-specific profile beats a mixed one.
  - At large VRAM, whole layers win, because of the fixed CPU-hop cost per layer. → idea 20 (hybrid) RUNNING.
- **Speculation on Qwen3.6 (hybrid architecture)**: acceptance 85–99%, but 30–41 tok/s vs 46–54 plain. Each partial accept restores a recurrent-state checkpoint and the target redoes the work. k = 6 hangs in llama.cpp's checkpoint loop (a bug). → retest on pure-attention Qwen3-Coder-30B-A3B (RUNNING).

## Results, night run 3
- **Qwen3-Coder-30B-A3B (pure attention, 48 layers × 128 experts), ms/token:**

  | VRAM | Prompts | Whole layers | Built-in cache | Own profile pinned |
  |---|---|---|---|---|
  | 15% | coding | 37.4 | 58.2 | **34.2** |
  | 15% | chat | 37.4 | 52.4 | **30.7** |
  | 30% | coding | 32.2 | 31.2 | **26.4** |
  | 30% | chat | 34.3 | 28.8 | **25.1** |

  A second model confirms the result: −9% to −27% vs today's layer split.
- **Speculation on Qwen3-Coder:** 29–37 tok/s vs ~38 plain, acceptance 58–89%. Still a loss, so the draft is too expensive. Suspect: host→GPU copy of the slot map per layer (48 stalls per draft token). FIX WRITTEN: device-resident slot maps.
- **21. Read the pitch: pick the lineup from the prompt** (simulation, `autopick_sim.py`):
  - The prompt's expert histogram identifies the use case **99–100%** of the time (cosine to the profiles).
  - A lineup from the prompt's own experts beats the use-case profile on the answer: Coder 15% VRAM 52.7 vs 44.8, 30% 75.2 vs 68.0; 35B +5 to +6 points.
  - The wrong profile is a disaster: 18% hits.
  - BUILT in llama.cpp (`EDX_PIN_PROMPT=<profile weight>`): the prompt's expert ids become graph outputs; after the prompt, re-pin to a blend before decode.
- **22. Live lineup swap** (`SIGUSR1` + new profile file): **0.45 s** mid-generation (44% of experts shared and kept in place), output coherent. Moved into an after-compute hook.
- **21 (measured). Read-the-pitch lineup, real llama.cpp, Qwen3-Coder, ms/token (hits):**

  | VRAM | Prompts | Saved profile | Blend 0.5 | Prompt only |
  |---|---|---|---|---|
  | 15% | coding | 34.1 (36%) | 31.4 (51%) | **30.1 (53%)** |
  | 15% | chat | 31.4 (45%) | **28.9 (52%)** | 29.1 (53%) |
  | 30% | coding | 26.0 (62%) | **24.7 (74%)** | 25.4 (72%) |
  | 30% | chat | 23.9 (70%) | 23.8 (71%) | 24.2 (68%) |

  - The re-pin after a prompt costs ~0.2 s (~25% of the lineup changes).
  - The gain is largest on small GPUs. Vs llama.cpp's layer split: coding at 15% −20%, chat at 30% −31%.
  - A 50/50 blend is the safe default.
- **23. Device-resident slot maps:** identical output, ~3% faster (removes a host→device copy per layer).
- **20 (measured). Hybrid placement** (hottest whole layers + pinned experts) beats both pure layouts: 30% VRAM 18.3 vs 20.4 / 21.2 ms; 60% VRAM 13.5 vs 16.3 / 14.6.
- **15 (final). Self-speculative decoding, REJECTED for RAM-offloaded MoE:**
  - The draft alone (pinned experts only, slot map on the GPU) runs at **162 tok/s**, vs 39 for the full model.
  - But verifying k+1 tokens costs ~3.3× a single token: each token routes to different experts, so the batch reads ~2.5× more cold experts on the CPU.
  - Net 32–35 tok/s vs 38–40 plain, at every k.
  - Structural: speculation needs verification to be cheap, and here it isn't.
- **24. Fast lane = the pinned-only model on its own** (from 15): **162 tok/s (4× faster).** Its greedy next token agrees with the full model's 77–89% of the time on coding. That's the owner's two-lane design made concrete: live chat on the fast lane, background agents on the full model. NEXT: measure its quality directly (perplexity and task evals vs the full model), and route math and reasoning to the full lane.
- **17 (simulated). Learned upload admission** (35B, 30% VRAM, est. tok/s):
  - Pinned only: 58.5.
  - Upload every miss into a 10–40% adaptive pool: 41–45 (worse).
  - "Used ≥3 times in the last 16 tokens": 60.7–61.6 (+4–5%).
  - Perfect foresight (≥2 uses in the next 16): 62–63 (+7–8%, the ceiling for a learned predictor).
  - Verdict: real but modest. Worth it only after the bigger wins.
- **24 (measured). Fast lane as a hard mask (pinned experts only): TOO LOSSY.**
  - Coding perplexity: 3.65 (full) → 10.96 at 30% VRAM (3×); 15% collapses (2,052) because the global ranking leaves some layers with fewer than 8 pinned experts. Wrong profile: 21.7.
  - Chat: 22.7 → 31.2.
  - Same lesson as OLMoE: a hard mask breaks, a soft nudge doesn't.
- **25. Soft steering in llama.cpp** (`EDX_STEER=λ`): multiply non-pinned experts' selection probability by e^-λ, keep the mixing weights. Picked cold experts still run on the CPU. Plus `EDX_PIN_MINPL=m` (each layer gets at least m pinned experts). RUNNING: λ 0/0.5/1/2 × 15%/30% × coding/chat, with perplexity, speed and hits.
- **Correctness check:** pinned mode (hot/cold split) gives the same perplexity as the plain model (3.51–3.54 vs 3.53) for batches ≤ 8. Larger batches crash a CUDA mul_mat_id path, so the pinned path is limited to ≤ 8 tokens (decode).
- **26. Speculative experts** (owner's idea): guess the next layer's experts, start uploading them during the current layer's compute, and let the router confirm in the normal pass. A wrong guess just runs on the CPU, so output is unchanged.
  - The point: uploads hurt only on the critical path (~3× a CPU read); a speculative upload overlapped with compute is nearly free. It needs high guess quality.
  - The co-occurrence guesser was weak (11% useful).
  - QUEUED (`spec_experts_sim.py`): apply layer l+1's router to layer l's hidden state (Fate-style) and measure recall@8/12/16 vs a previous-token baseline, plus how many speculative uploads would be useful on top of the pinned set.
  - If recall is high: a patch with a copy stream that uploads predicted cold experts one layer ahead.

## Night 2 (2026-11-05): measured on Linux
- **P1 (measured): "layers with any miss" is NOT the right metric.** Per-token time tracks total misses (R² 0.88) far better than layers-with-any-miss (0.29). Phase 1 found the per-layer CPU handoff cost ≈ 0 on this box (fit −0.05 ms); the cost is the CPU expert reads themselves (0.081 ms each, ~27 GB/s).
- **Upload threshold measured:** an uploaded expert must be reused ≥2.9 times to beat a CPU read (PCIe 9.4 GB/s pageable, 13.0 pinned).
- **P2 (sim): set-completing pins = frequency pins.** No gain at 15/30/60%. Rejected.
- **P3 (sim): per-layer whole-vs-pins helps only at 60% VRAM** (+5%, ~11 whole layers); at 15–30% the optimizer picks pins only. Matches the earlier hand-picked hybrid (13.5 vs 14.6 ms at 60%).

## Mac (Apple M4, 16 GB unified memory): new ideas
- **27 (measured). Turn off weight repacking on ARM (`-nr`).** By default llama.cpp repacks CPU experts into anonymous RAM: an 18.6 GB model on a 16 GB Mac then swaps (1.2 s/token). With `-nr` the experts stay file-backed mmap pages that macOS can drop and re-read from the SSD.
- **28 (measured, first run). On unified memory, pin, don't cache.** With 4 GB pinned: the coding profile gives 245 ms/token vs 654 for llama.cpp's LRU cache (2.7×). The LRU "uploads" 24 GB of memcpy for 63 tokens and its copies fight the page cache.
- **29. Drop the duplicate pages (`EDX_PAGEOUT`).** A pinned expert exists twice on a Mac: in the Metal buffer and in the page cache. Paging the mmap copy out frees RAM for the cold experts. Being measured.
- **30. Zero-copy pinning on unified memory.** Instead of copying hot experts into a Metal buffer, `mlock` their file pages and let Metal read them in place (a no-copy buffer over the mmap). Same pinning, zero duplicates, no copy at load.
- **31. Reorder the GGUF by profile.** Rewrite the model file so each use case's hot experts are contiguous. SSD readahead then fetches useful neighbours, and the hot set can be locked with one `mlock` range.
- **32. SSD-aware steering.** On the Mac the cold tier is the SSD (~100× slower than RAM), so the nudge (λ) should be stronger there. A per-tier λ: weak for RAM-resident cold experts, strong for experts that would page in from the SSD (`mincore()` tells which).
- **33. Session lineup as a file hint.** At session start, `madvise(WILLNEED)` the use case's next-most-likely experts so the SSD prefetches them in the background before the first token.
- **Mac run 1 result (see MAC_LOG.md):** steering 115.7 ms vs 172.3 (CPU/mmap) vs 342.4 (layer split); plain pinning loses to the OS page cache on unified memory. Next: **34. Steer-only mode for unified memory** — keep every expert on the CPU/mmap path (no Metal copy, no split), use the profile only to nudge the router toward the hot set so it stays in the page cache. Expected to beat 115.7 ms.
- **34 (measured, Mac run 2). Steer-only:** 1.64x faster than the baseline at λ=1 for +1.1% ppl, using ~5 MB extra memory. Copy+steer is ~10% faster still (110.5 vs 122.9 ms). Hot-set size 4-8 GB hardly matters; λ does.
- **35. Tune λ per tier on the Mac** (idea 32 made concrete): λ=1 costs +1.1% ppl, λ=2 +6% for another 1.24x. A per-layer λ (strong where the router is flat, weak where it is sharp) may buy λ=2 speed at λ=1 quality.
- **36. Steering speeds prompt processing too:** perplexity passes at -ub 8 went 51.5 → 23.2 s at λ=1 (2.2x). Worth testing steering in larger prompt batches now that steer-only has no 8-token limit from the split.

## Owner's two-lane design (2026-11-05)
- **37. Fast chat lane, slow work lane.** Chat only needs to understand and plan, so it must be fast; the work needs the full model's intelligence and can take 5-7 hours in the background. Chat lane: a small model or a steered "chatter" lineup kept entirely in RAM (Qwen3-Coder-30B steered: ~10 tok/s on the M4). Work lane: the big model, exact, on the SSD.
- **38. Batch the work lane (owner's idea).** Autoregressive decoding can't skip ahead within one sequence, but N independent jobs (files, tests, sub-tasks the orchestrator splits off) can decode together: each layer's experts are read from the SSD once and serve all N. Expected: aggregate tok/s grows with N until compute-bound, because the union of experts per layer grows slower than N. Test queued: llama-batched-bench on the 122B at 1/2/4/8 parallel jobs.
- **39. The orchestrator writes batches, not turns.** The chat lane turns a request into independent work items up front (a plan with N parallel tasks), so the work lane always runs full batches. Same trick as offline batch inference APIs, run locally overnight.
- **40. Re-test speculative decoding on the SSD-bound Mac.** On the 3090s verify cost ~3.3x a token (union of cold CPU experts) and speculation lost. On the Mac, where a token costs seconds of SSD reads, a k-token verify that shares expert reads may win. The chat lineup doubles as the draft.
- **38 (measured, Coder-30B on the M4).** 8 parallel jobs: 2.1x total throughput exact, 2.65x steered; steered + batched = 26.6 tok/s total vs 6.8 for one exact job (3.9x).
- **P6 skip-the-miss (measured, quality only):** top-8 denominator +1.0% ppl, survivors +1.6%, top-16 +3.5%; steer + skip +13% (don't stack).
- **gpt-oss-120b on the 16 GB M4 (measured):** exact 1.04 tok/s; steering λ=0.5 1.46x for +2.0% ppl, λ=1 2.85x for +8.3%, λ=2 6.5x but +86%. Batched (8 jobs) + λ=1: 7.0 tok/s total. Found and fixed a steering bug for logit routers (gpt-oss): subtract λ, don't multiply.
- **41. Calibrate λ per model (and per layer) automatically.** Top-4 models need ~half the λ of top-8 models. A quick auto-tune: pick the largest λ whose perplexity on a small held-out set stays within a budget (say +2%).

### 42. Speculative decoding on SSD-bound MoE (measured 2026-11-05, Mac run 6): REJECTED
EAGLE3 head on gpt-oss-120b: best 0.75x of exact (draft 2, 69% accept); worse with longer drafts; no better with steering. Verifying k tokens costs ~k tokens because each token's top-4 experts barely overlap, so the SSD reads are not shared. Lesson: on an expert-read-bound machine, only *independent parallel jobs* share reads (and only under steering, which herds them onto the same experts); speculative decoding of one sequence does not. gpt-oss-20b as a draft beside the 120B ran Metal out of memory.

### 43. Chat lane takes turns with the work lane (follows from 42)
Two models cannot share 16 GB live. Design: chat lane (small model) runs while the owner talks; on hand-off it writes a batch file of independent work items, exits, and the 120B runs the batch steered + batched (7 tok/s total measured). Session-level routing: the chat model labels the task type, which picks the 120B's expert profile and λ; prefetch (idea 33) warms those experts before the batch starts. To test: switch-over cost (unload 20B, first-token latency of 120B on a warm vs cold cache).
