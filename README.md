# omxpert dispatch

> **An omxlabs research project.** Status: research, not a product.

**Cover page: https://abhisheksreesaila.github.io/omxpert-dispatch/**

Research journey, day by day: [docs/JOURNEY.md](docs/JOURNEY.md) →

## Vision

Big open models are now mixtures of experts, and running them at home is hard with RAM and GPUs scarce. Instead of tuning a whole model or agent to you, we're testing one level deeper: **can we learn which experts *your* work uses, from your own session history, keep those in fast memory, and keep re-learning them as your work changes?** Then we stack tricks others have shared on top and measure each one. An ongoing local-AI experiment.

## Setup

- **Profile:** run the model over your past sessions and count which experts fire (counts only, never the text). 2,000–5,000 tokens were enough.
- **Machine 1:** 16 GB MacBook (M4), gpt-oss-120b (63 GB, 4× the RAM), llama.cpp + our patch, SSD as the cold tier.
- **Machine 2:** Linux, 2× RTX 3090 + 64 GB RAM, Qwen3.8-Flash-Next (Qwen 4 preview, 111 GB), llama.cpp + our patch; FreeToken tried.

## What worked

**Machine 1: gpt-oss-120b on a 16 GB MacBook** (decode tok/s, one job)

| | Step | From | tok/s | Quality |
|---|---|---|---|---|
| 0 | Baseline: exact model, experts read from the SSD | llama.cpp | 1.04 | exact |
| 1 | Borrow a loaded expert, gently (λ 0.5) | idea: Cache-Prior (Skliar et al.) | 1.51 | +2.0% perplexity |
| 2 | Borrow firmly (λ 1), toward your coding profile | omxpert | **2.96** (2.85×) | +8.3%; graded coding tasks 7/8 (8/8 at λ 0.5) |

**Machine 2: Qwen3.8-Flash-Next on 2× RTX 3090 + 64 GB** (decode tok/s, held-out chat prompts)

| | Step | From | tok/s |
|---|---|---|---|
| 0 | Baseline: stock llama.cpp, its own split by layer | llama.cpp | 17.7 (prompts 108) |
| 1 | Your profile picks the VRAM experts; each expert in VRAM or RAM, never both | omxpert + 0xSero's exclusive tiers | 8.9 (first try: prompts re-read experts from disk, 3,148 page faults/token) |
| 2 | Fix the prompt path (copy experts across GPUs, run each layer on its GPU) | omxpert | 21.2 (faults → 48/token) |
| 3 | CPU computes cold experts while GPUs compute hot ones | FreeToken (FlashML), ported into llama.cpp | +2–5% |
| = | All together vs stock, side by side (two interleaved server pairs) | | **22.2 / 21.1 vs 17.7 / 15.7**; prompts 218 / 202 vs 108 / 83 |

Rows 1–3 are separate runs on a shared desktop; the last row is the fair comparison. 30% of experts served 79% of picks in new chats (someone else's profile: 44%). Not bit-exact on machine 2 (greedy word match with stock 0.49; another engine on the same model 0.35). Side experiment on small models: learning *which* 8 of 64 experts to keep plus a tiny adapter kept 66% of next code tokens right vs 69% for the full model (counting alone: 48%).

## What didn't work

| Tried | Result |
|---|---|
| FreeToken's engine as is, 64 GB of RAM (its GPU cache only holds RAM-resident experts) | 1.4 tok/s |
| Your profile on unlike work (chat profile, coding prompts) | 10.0 vs 14.9 stock: re-profile when work changes |
| Speculative decoding on the Mac | 0.27–0.75× |
| Batching several real jobs on the Mac | slower than one at a time |
| Layer-ahead prefetch and read pool (0xSero's) on the desktop | no gain (prefetch ~6% slower) |
| Qwen's MTP draft head on the pinned stack | 21.7 vs 23.0 (×1.36 on stock llama.cpp) |
| Plain pinning on the Mac, skipping missing experts, a learned per-expert bias | no gain |

## It carries on: contribute

Next: the expert layer in Mojo (hot experts on GPUs, cold on CPU, one pass per layer, SIMD) and a pass@1 quality test. Try it with your own history (`omxpert/profile.sh`, then `STACK=pinned omxpert/run.sh`), share numbers or suggest a trick: [open an issue](https://github.com/abhisheksreesaila/omxpert-dispatch/issues).

Borrowed with thanks: [0xSero](https://github.com/0xSero) (exclusive tiers, prefetch, read pool), FreeToken by FlashML and the FreeToken-Kai fork, Skliar et al.'s [Cache-Prior](https://arxiv.org/abs/2412.00099), Qwen, llama.cpp / ggml, unsloth.

---

# Lab detail (earlier write-ups)

## The idea

Mixture-of-experts models are where open AI is going ("Dense is basically over", Thom Wolf, as Hugging Face released its MoE training handbook). They are huge, but sparse: gpt-oss-120b wakes 4 of its 128 experts per token, and one person's work keeps waking the same few.

Engines already offload experts to RAM and SSD and cache them reactively, and that works well. The question here: **what if your expert set were something learned?**

1. **Profile** your own sessions and record which experts each kind of task uses (coding, search, docs, tools).
2. **Classify** a new session from its first prompt (99–100% in our simulations) and **predict** its experts.
3. **Place** them before the first token: the core on the GPU, the neighbours in RAM, the rest on SSD. Picked, not cached.
4. **Re-profile** when your work shifts (this week Python, the weekend Mojo). Same model, new expert set. No quantization, no retraining.
5. When a rare token wants an expert that isn't loaded, **borrow a loaded neighbour** (a small router nudge, λ) instead of waiting on the SSD.

## New (2026-11-06): learned expert sets and adapters

On a small MoE (OLMoE-1B-7B, 64 experts, 8 per token), routing strictly to a few loaded experts per layer, scored on held-out coding text:

| Experts kept per layer | Counted set | Learned set (swap search) | Learned set + adapter (2M params, ~5 min) | Full model |
|---|---|---|---|---|
| 16 of 64 | 64.7% next token right | 65.7% | **68.0%** | 69.1% |
| 8 of 64 | 47.8% | 57.9% | **66.2%** | 69.1% |
| 4 of 64 | 8.0% | 19.6% | **62.3%** | 69.1% |

- **Second model, same recipe** (IBM Granite 3.1 3B-A800M, 40 experts, 8 per token; full model 68.7%): 16 of 40: counted 42.7% → learned 51.7% → + adapter 63.3%. 8 of 40: 33.7% → 43.8% → 57.6%. 4 of 40: 1.5% → 20.3% → 45.0%.
- **Learned set:** start from the most-used experts and try swaps layer by layer; keep a swap only if the model gets closer to the full model (KL) on training text. Gradient-learned gates and first-order importance both failed (logged).
- **Adapter:** a low-rank residual per MoE block, trained by distillation against the full model. Set + adapter = one bundle per profile; the matching bundle wins on its own kind of work.
- **Correction (03:15):** the first 4-of-64 numbers were wrong. The model picks 8 experts per token, so with only 4 loaded the strict mask still filled 4 slots with unloaded experts at full weight. 8 and 16 of 64 are unaffected. Re-measured with the unloaded slots dropped: 62.3% (the faulty run said 64.0%).
- Caveats: one small model; the bundle agrees with the full model's top guess 71–82% of the time; the adapter also lowers perplexity on its own (unexplained, see `docs/logs/NIGHT3_LOG.md`). Not yet on the 120B.
- gpt-oss-120b on the 16 GB Mac (steered, real graded coding tasks): 7/8 and 8/8 tests passed; running 4 real jobs at once was slower than one at a time.

## What we've seen so far: Pin, Steer, Batch

Pin and steer did the work; batch is still open (see the correction below):

| Lever | What it does | Best measured result |
|---|---|---|
| **Pin** | Keep the session profile's experts in the fastest memory | Linux, 2×RTX 3090: **32% faster** than llama.cpp's default placement, exact output |
| **Steer** | Nudge the router toward experts already loaded (`logit − λ` for non-resident experts) | Mac M4 16 GB, Qwen3-Coder-30B: **1.64×** for +1.1% perplexity. gpt-oss-120b: **1.46×** for +2.0% (λ=0.5), **2.85×** for +8.3% (λ=1) |
| **Batch** | Run independent jobs together so they share each layer's expert reads | Benchmark only: 7.0 tok/s total (gpt-oss-120b, 8 jobs) and 26.6 (Qwen3-Coder-30B), **measured on random tokens**. Eight real coding jobs: **~1.0 tok/s total** (RAM thrash). Open |

**Correction (2026-11-05):** `llama-batched-bench` feeds random tokens, not text. On real coding jobs, 8 at once ran at ~1.0 tok/s total on the 16 GB Mac, worse than one steered job (2.96): eight different jobs touch more experts than fit in RAM, and macOS pages them in and out. Batching on real text with fewer jobs (1 vs 4) is being measured.

### Results on the 16 GB Mac (Apple M4, unified memory, SSD as the cold tier)

gpt-oss-120b MXFP4 (63 GB, 4× the RAM), coding:

| Config | tok/s | vs exact | Coding perplexity |
|---|---|---|---|
| Exact model | 1.04 | 1.0× | 3.787 |
| Steer λ=0.5 | 1.51 | 1.46× | 3.862 (+2.0%) |
| Steer λ=1 | 2.96 | 2.85× | 4.100 (+8.3%) |
| Steer λ=2 | 6.76 | 6.5× | 7.046 (+86%, unusable) |
| Steer λ=1, 8 jobs, random-token benchmark | 7.0 total | not real text | same as λ=1 |
| Steer λ=1, 8 real coding jobs (llama-server) | ~1.0 total | worse than one job | same as λ=1 |
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

## A design we're considering: two lanes

```
 you ──► fast chat lane (small model)        background work lane (big MoE, full intelligence)
         • understands, plans                • loads the profile's experts (pin)
         • classifies the task type ───────► • steers toward them (λ per model)
         • writes a batch of work items      • runs the batch together (batch)
         • steps aside (16 GB can't hold both)  ~3 tok/s per job on gpt-oss-120b
```

## Open directions (not done; what would you try?)

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
| [`docs/index.html`](docs/index.html) | The cover page: https://abhisheksreesaila.github.io/omxpert-dispatch/ |
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
