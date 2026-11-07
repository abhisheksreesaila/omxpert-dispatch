# Night 5 report: Qwen 4's architecture on a 2×3090 desktop (2026-11-06, daytime autopilot, 16:00-21:00)

**Goal (owner's words, condensed):** take the tricks learned from 0xSero and from us, try them on the NVIDIA box with
Qwen3.8-Flash-Next ("that's the four"), show before/after on the exact model so people can download and use it, then find where
added hardware would give leverage: the seed of a "Hyper-V for inference".

**Checker** (`scripts/check_night5.sh`): PASS ×6 (baseline, exact tricks, learned tricks, runnable stack, leverage, report).

## The answer in four lines
1. **Big model, full intelligence, less RAM: yes.** Qwen3.8-Flash-Next (Qwen 4 preview architecture: 125B MoE + a 51B-value
   n-gram table, 111 GB at 4-bit) runs on 2×3090 with ~35 GB of free RAM at **~21-24 tok/s decode and ~130-170 tok/s prompt**,
   bit-exact to stock llama.cpp. The first run of the day was 11 / 28 tok/s; the slow runs all had hundreds of page faults per token.
2. **Exact tricks:** MTP drafting (Qwen's own draft head) ×1.36 decode, 8/8 paired rounds, but its 4 GB of VRAM costs prompt
   speed here (20-80 tok/s). `-lzm off` is bit-exact and kept page faults low in 8/8 runs. His read pool and layer-ahead prefetch,
   and our learned next-layer guesser, give nothing on this box: 99.97% of expert pages are already in RAM.
3. **Learned tricks:** Qwen 4's tiny experts concentrate (30% of experts serve 80% of coding picks), but inside llama.cpp pinning
   them per expert did not beat llama.cpp's own placement (20.3 vs 21-22 tok/s) and made prompts 4-5× slower. Steering added ~10%
   for a measurable quality cost (KL 0.074 vs a 0.027 floor).
4. **The bottleneck is the CPU's expert math (32 of 49 ms per token).** More RAM or a second NVMe buys no decode speed; moving
   expert work to a GPU does: an eGPU as 7 more layers ~25 tok/s in llama.cpp today; with an engine that runs CPU and GPU together
   more is possible, but our model of that is unproven (it over-predicted the pinned run by 2×).

## What each trick did (measured on this box)
| Trick | From | Result | Verdict |
|---|---|---|---|
| N-gram table mapped and read at load (`-lzm off`) | Qwen/llama.cpp flag | KL 0 (bit-exact); low faults 8/8 runs vs stock 4/11; same speed as stock once stock settles | keep (robustness) |
| MTP drafting | Qwen/llama.cpp | 20.3 → 27.6 tok/s, 8/8 paired, 86% drafts accepted; prompts 20-80 tok/s instead of 110-170 | opt-in for chat (`MTP=1`) |
| Staged prefill (2048-token batches) | 0xSero | 22.9 / 149 vs 22.6 / 150 without it; nothing to gain on ~550-token prompts | not needed here |
| Parallel read pool | 0xSero | +2.8% in one server, within the ~5% run-order effect | no gain (nothing misses RAM) |
| Layer-ahead prefetch (previous-token guess) | 0xSero idea | +1.5%, within noise | no gain here |
| Learned next-layer guesser | omxpert | offline: 52% of next-layer picks with 16 guesses (static 26%, previous token 36%); online +1.7%, within noise | pays only when experts miss RAM |
| Exclusive tiers (page out pinned experts' RAM) | 0xSero | pin 13.4 vs no pageout 21.1, both confounded runs | inconclusive |
| CPU works while GPU works | 0xSero | not built (engine change) | the next step |
| No per-layer host sync | 0xSero | not built; but GPU↔CPU switches measured cheap: 50 per token ≈ 4 per token (22.0 vs 19.7 tok/s) | low priority here |
| Coding profile picks VRAM experts | omxpert | 20.3 vs 21-22 tok/s; prompts 32 vs 143-149 tok/s | doesn't pay inside llama.cpp |
| Steering toward VRAM experts | omxpert | λ0.5: 23.9 vs 21-22 tok/s, KL 0.074 (floor 0.027) | small gain, lossy |
| Steering-aware adapter | omxpert | not built (needs thousands of 125B forward+backward passes) | shown on small models (night 3) |

## What we learned that we did not expect
1. **Page faults decide everything, and separate runs lie.** The same stock config measured 11 tok/s at 16:06 and 23 tok/s at
   18:34. Inside one server, the second arm of a round ran ~5% faster even when identical. Method that held: one warmed server,
   tricks switched per request, order alternated. A first wave of results (steering "2×", "`-lzm off` = 4× prompts") came from
   comparing against the fault-heavy first run; both were retracted after clean reruns (D-11, D-14).
2. **llama.cpp's "exact" isn't bit-exact across batch shapes.** Changing only the batch size gives KL ~0.025 (94% same top token)
   on this MoE, because routing flips. Identical settings give KL 0. Lossy tricks are judged against that floor.
3. **This box is CPU-bound for Qwen 4.** Half of the 480 expert picks per token run on the 12-core CPU: 32 of 49 ms.
4. **MTP works here, unlike on the 16 GB Mac** (idea 42): the experts are in RAM and consecutive tokens share most of them.

## Before and after, visually
`docs/visuals/night5_before_after.html`: the measured speed ladder, each trick next to its database analogy, the leverage table,
and two isometric diagrams of one token's trip (`docs/visuals/night5/before.html`, `after.html`: click blocks, press Play).

## Leverage: what each added part would buy (decode, from measured components)
The model predicted 19.7 tok/s for a different layer layout and measured 19.7 there, but it over-predicted the pinned profile by 2×, so its software-engine rows are ceilings, not promises.

| Add | Decode tok/s | Why |
|---|---|---|
| Nothing (today, no MTP) | 20.3 | half the expert picks on the CPU |
| MTP drafting (measured) | 27.6 | ×1.36, costs prompt speed |
| Second NVMe | 20.3 | 99.97% of expert pages already in RAM; helps load time only |
| 128 GB RAM | 20.3 | decode is CPU-bound; buys the lossless 102 GB table, 8-bit experts or a second model |
| 16-core CPU (5950X) | ≤24.2 | upper bound; the expert math is partly RAM-bandwidth bound |
| eGPU 24 GB as 7 more layers (llama.cpp today) | 25.0 | 17 CPU layers left |
| NPU | no gain | none in this desktop; on the same DDR4 it reads no faster |
| M4 Mac over Tailscale | background lane only | round trip 53 ms average (7.6-302): a hop costs more than a token |
| Engine: CPU and GPU at once | 31.4 (estimate) | max instead of sum |
| Engine + profile + eGPU + MTP | 77.8 (estimate, optimistic) | the model over-predicted the pinned run 2×; treat as a ceiling |

## Decisions worth a second look
- **D-13:** the launcher defaults to `-lzm off` without MTP (steady, fast prompts); `MTP=1` trades prompt speed for ~1.36× faster
  answers. Your call which default fits a video.
- **D-05/D-14:** all decode comparisons that matter come from one-server A/Bs or interleaved reruns; the first separate-run
  numbers stay in the logs but are not used.
- **D-03:** his ideas were rebuilt inside llama.cpp, not run in his engine. The two that need an engine (overlap, no host sync)
  were not built, so his core speedups are untested here.

## Needs you
- Nothing to run it: `omxpert/run.sh` (smoke test passed: it answered a coding question). The model is in `~/models`.
- Disk: 34 GB free now (111 GB model + 2.8 GB draft head added). The lossless n-gram table (102 GB) would need the 122B deleted.
- Nothing was pushed: the public repo and site are untouched (a local commit is ready in `~/expert-dispatch-public`).

## Credits
The systems ideas (parallel read pool, layer-ahead prefetch, exclusive tiers, staged prefill, CPU/GPU overlap, no per-layer host
sync) are **0xSero**'s, from glm-flash-lite (GLM-5.3-Flash at 19.5 tok/s on one RTX 3090), re-implemented here inside llama.cpp.
MTP drafting and the n-gram table loader are Qwen's and llama.cpp's. Profiles, the learned guesser and steering are omxpert's.

## Next
Build the one piece llama.cpp can't express: an expert layer where the CPU's cold experts run while the GPU runs the hot ones
(his overlap). That's where the profile's 89% GPU hit rate could turn into speed, and the natural first Mojo component.
