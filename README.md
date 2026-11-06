# Expert Dispatch

**Cover page: https://abhisheksreesaila.github.io/expert-dispatch/**

**Run a mixture-of-experts model far bigger than your memory by loading only the experts *your* work uses.**

A 63 GB model (gpt-oss-120b, as released, MXFP4) on a 16 GB MacBook Pro (Apple M4). Exact model: 1.0 tok/s. With the three levers below: up to 7.0 tok/s for batched background work.

This is a research log, not a product. Every number here was measured on real hardware unless marked *simulated*. Failures are reported alongside wins.

---

## The idea

A mixture-of-experts (MoE) model has, say, 128 experts per layer, but each token only uses 4 or 8 of them. People don't use a model for everything; most of what one person does is a handful of task types (coding, web search, tool use, docs). So:

1. **Profile** your own sessions and log which experts light up for each task type: profile 1 (coding), profile 2 (chat), profile 3 (...).
2. **Classify** a new session (from the prompt alone: 99–100% accurate in our simulations) and load *that* profile's experts into the fastest memory. A few minutes of setup, then the session runs hot.
3. **Tier the rest:** hot experts in GPU memory, warm ones in RAM, cold ones on SSD. Go to SSD only when you must.
4. **When an expert is missing, use a nearby one that is loaded.** A small, tunable nudge to the router (λ) is enough. It's cheaper than waiting, and better than skipping the expert.

You don't need the whole model in fast memory. You need the slice your session uses.

## What we found: Pin, Steer, Batch

Three levers did almost all the work:

| Lever | What it does | Best measured result |
|---|---|---|
| **Pin** | Keep the session profile's experts in the fastest memory | Linux, 2×RTX 3090: **32% faster** than llama.cpp's default placement, exact output |
| **Steer** | Nudge the router toward experts already loaded (`logit − λ` for non-resident experts) | Mac M4 16 GB, Qwen3-Coder-30B: **1.64×** for +1.1% perplexity. gpt-oss-120b: **1.46×** for +2.0% (λ=0.5), **2.85×** for +8.3% (λ=1) |
| **Batch** | Run independent jobs together so they share each layer's expert reads | gpt-oss-120b, steered, 8 jobs: **1.0 → 7.0 tok/s total**. Qwen3-Coder-30B: **26.6 tok/s total** |

Steering is what makes batching pay off on a top-4 model: without it, 8 jobs share few experts (only 1.44×). Steering herds them onto the same loaded experts.

### Results on the 16 GB Mac (Apple M4, unified memory, SSD as the cold tier)

gpt-oss-120b MXFP4 (63 GB, 4× the RAM), coding:

| Config | tok/s | vs exact | Coding perplexity |
|---|---|---|---|
| Exact model | 1.04 | 1.0× | 3.787 |
| Steer λ=0.5 | 1.51 | 1.46× | 3.862 (+2.0%) |
| Steer λ=1 | 2.96 | 2.85× | 4.100 (+8.3%) |
| Steer λ=2 | 6.76 | 6.5× | 7.046 (+86%, unusable) |
| Steer λ=1, 8 jobs batched | **7.0 total** | 6.7× | same as λ=1 |
| Speculative decoding (EAGLE3 head) | 0.24–0.66 | 0.27–0.75× | lossless, but slower |

Practical check (8 real coding tasks, answers graded by running hidden tests): see [`results/mac/work.log`](results/mac/work.log) and the section below.

Qwen3-Coder-30B-A3B (18.6 GB, top-8 of 128): steering λ=1 is 1.64× for +1.1% perplexity; 8 batched jobs give 2.1× (exact) and 2.65× (steered).

### What did not work

| Tried | Result | Why |
|---|---|---|
| Speculative decoding on the Mac (EAGLE3 head, gpt-oss-20b draft) | 0.27–0.75× of plain decoding, even at 69% acceptance | Verifying k tokens costs ~k tokens: consecutive tokens pick different experts, so no SSD reads are shared |
| Plain pinning on the Mac | Lost to macOS's own page cache | Unified memory: the OS already keeps hot pages resident |
| Two models live in 16 GB (chat model + 120B) | Metal out of memory | They must take turns |
| Skip the missing expert (drop and rescale) | Same quality cost as steering, no extra speed; worse when combined with steering | Using a nearby loaded expert beats using none |
| A learned per-expert router bias | No better than one hand-set λ | A constant can't beat a constant by much; learning *which* experts stay resident helped a little |
| Static profile vs a recency (LRU) cache, hit-rate only | LRU wins by 12–28 points on a 256-expert model | Consecutive tokens reuse experts; profiles shine at very small caches (≤6%) and as a warm start |
| Per-layer upload tricks (night 2, P1/P2) | No gain | Decode time tracks total misses (R² 0.88), not layers-with-a-miss (0.29) |
| Batching without steering on gpt-oss | 1.44× only | Top-4 of 128: independent jobs rarely share experts |

### Personal profiles are real

On Qwen3.6-35B-A3B (256 experts per layer), a profile built from your own coding beats one built from your other work by **+11 points** of hit rate at a 25% cache, and is nearly **2×** better at 6%. Coarse models (64 experts) barely specialize (+1.8 points). Profiles go stale (25–28% of the hot set changed within weeks), so refresh them.

## The design this points to: two lanes

```
 you ──► fast chat lane (small model)        background work lane (big MoE, full intelligence)
         • understands, plans                • loads the profile's experts (pin)
         • classifies the task type ───────► • steers toward them (λ per model)
         • writes a batch of work items      • runs the batch together (batch)
         • steps aside (16 GB can't hold both)  ~7 tok/s total on gpt-oss-120b
```

## Open directions (not done)

- **Steering-aware adapter.** λ=1 is 2.85× faster but costs 8.3% quality. Treat each expert as an "agent" and train a small adapter that corrects for a substituted expert; recover the quality, keep the speed. The most promising next step.
- **Onboarding: profile from your history.** Most people already have chat sessions to export. A short guided session (~15 minutes) plus a classifier picks the experts per task type, which removes the cold start.
- **An encoder-only classifier** (a decoder-stripped LLM used as a universal classifier) to route sessions cheaply.
- **Auto-calibrate λ per model and per layer.** gpt-oss (top-4) needs about half the λ of a top-8 model.
- **Switch-over cost:** time from chat model exit to the 120B's first token, warm vs cold.

## Repository map

| Path | What |
|---|---|
| [`patch/llama.cpp-edx.patch`](patch/llama.cpp-edx.patch) | Our llama.cpp changes: pinned hot set, steering (`EDX_STEER`), steer-only mode, skip-the-miss (`EDX_SKIP`), decode tracer. Base commit `bbb02c9` |
| [`patch/edx-trace/`](patch/edx-trace/) | Per-token expert tracer (~2,000 tok/s) |
| [`scripts/`](scripts/) | Profiling, simulators, benchmarks (`mac_*.sh` are the Mac runs) |
| [`mojo/cachesim.mojo`](mojo/cachesim.mojo) | Cache replay simulator |
| [`results/`](results/) | Metrics, plots, profiles (`pin_*.bin`), Mac logs |
| [`docs/index.html`](docs/index.html) | The cover page: https://abhisheksreesaila.github.io/expert-dispatch/ |
| [`docs/vision.md`](docs/vision.md) | The original idea |
| [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) | Related work (ProMoE, MoE-Infinity, Cache-Prior, ReMoE, DoMoE, llama.cpp `--moe-cache-mib`, ...) |
| [`docs/logs/`](docs/logs/) | Night-by-night lab notes, the Mac log, and the idea log (`IDEAS.md`, 40+ ideas) |

## Reproduce (Mac)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout bbb02c9
git apply ../patch/llama.cpp-edx.patch && cp -r ../patch/edx-trace tools/
cmake -B build && cmake --build build -j
# exact vs steered, experts left on SSD via mmap:
C="-nr -fit off -lm mmap -ngl 99 --n-cpu-moe 36"
EDX_PIN=results/gptoss/pin_gptoss_coding.bin EDX_STEER_ONLY=1 EDX_PIN_MINPL=8 EDX_STEER=0.5 \
  build/bin/llama-completion -m gpt-oss-120b-MXFP4.gguf $C -f prompt.txt -n 96
```

`-nr` (no repack) matters on Apple Silicon: repacking copies the experts into RAM and the model swaps (1.2 s/token).

## Caveats

- Perplexity on 4 × 512 tokens: treat ±1–2% as noise.
- Profiles were built from one person's work; the "personal" gain is measured on that one person.
- Linux timings ran on a shared workstation; some runs were under load and are marked in the logs.

## Prior art

Biasing the router toward cached experts was published by Skliar et al. ([Cache-Prior](https://arxiv.org/abs/2412.00099), 2024); domain routing tables by DoMoE (2026); profile-driven placement by PowerInfer and KTransformers. What this repo adds: measurements of pin + steer + batch together, on a model 4× larger than unified memory, with the SSD as the cold tier, and the negative results (speculative decoding, skip, learned bias) that rule out the obvious alternatives. Full table in [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md).

## License

MIT for the scripts and docs. The llama.cpp patch follows llama.cpp's MIT license.
