# omxpert dispatch

> **An omxlabs research project.** Status: research, not a product.

**Cover page: https://abhisheksreesaila.github.io/omxpert-dispatch/**

Research journey, day by day: [docs/JOURNEY.md](docs/JOURNEY.md) →

**An open experiment: can we learn, from your own sessions, which experts of a mixture-of-experts model you use, keep re-learning them as your needs change, and combine that with tricks others have shared (FreeToken, 0xSero's) into a better way to run big models locally?**

We don't know yet. Below is the story so far, then the results, wins and failures. Ideas and criticism are welcome: [open an issue](https://github.com/abhisheksreesaila/omxpert-dispatch/issues).

A 63 GB model (gpt-oss-120b, as released, MXFP4) on a 16 GB MacBook Pro (Apple M4). Exact model: 1.04 tok/s. Steered toward the coding profile: 2.96 tok/s on real text (λ=1, +8.3% perplexity).

This is a research log, not a product. Every number here was measured on real hardware unless marked *simulated*. Failures are reported alongside wins.

---

## The idea

Mixture-of-experts models are where open AI is going ("Dense is basically over", Thom Wolf, as Hugging Face released its MoE training handbook). They are huge, but sparse: gpt-oss-120b wakes 4 of its 128 experts per token, and one person's work keeps waking the same few.

Engines already offload experts to RAM and SSD and cache them reactively, and that works well. The question here: **what if your expert set were something learned?**

1. **Profile** your own sessions and record which experts each kind of task uses (coding, search, docs, tools).
2. **Classify** a new session from its first prompt (99–100% in our simulations) and **predict** its experts.
3. **Place** them before the first token: the core on the GPU, the neighbours in RAM, the rest on SSD. Picked, not cached.
4. **Re-profile** when your work shifts (this week Python, the weekend Mojo). Same model, new expert set. No quantization, no retraining.
5. When a rare token wants an expert that isn't loaded, **borrow a loaded neighbour** (a small router nudge, λ) instead of waiting on the SSD.

## New (2026-11-07): trying it on Qwen 4's architecture, on one desktop

Qwen3.8-Flash-Next (Qwen 4 preview architecture: 125B MoE + 51B n-gram table, 111 GB at 4-bit) on 2× RTX 3090 + 64 GB RAM. An early result: one machine, one person's history.
We tried three sets of tricks in one engine: **your profile** (from your own chat history) picks the VRAM experts, **0xSero's exclusive
tiers** keep each expert in VRAM or RAM but never both, and **FreeToken's CPU-while-GPU** hybrid, ported into llama.cpp's scheduler.

| Your held-out chat prompts (2 interleaved server pairs) | Decode tok/s | Prompt tok/s |
|---|---|---|
| FreeToken as is (its 64 GB mode) | 1.4 | 17 |
| Stock llama.cpp (its own fit) | 17.7 / 15.7 | 108 / 83 |
| **Combined stack** (`STACK=pinned omxpert/run.sh`) | **22.2 / 21.1** | **218 / 202** |

- **The profile is personal:** 30% of the experts serve 79% of picks in new conversations (ceiling 83%); someone else's coding profile serves 44%. 2,000 tokens of history are enough. Learn yours with `omxpert/profile.sh` (counts only, no text kept).
- **Night 5's "pinning doesn't pay" was a bug:** prompt batches re-read GPU 1's pinned experts from disk (3,148 page faults per token). `EDX_XDEV` + `EDX_PREFILL_LOCAL` took it to 48.
- **Small or no gain:** CPU-while-GPU (`EDX_OVERLAP`) +2–5% (the profile already puts ~94% of picks on the GPUs); MTP and layer-ahead prefetch none on the pinned stack.
- **Costs:** on text unlike your profile it was slower than stock (coding prompts with a chat profile: 10.0 vs 14.9 tok/s; steering λ0.5 recovers 16.9). Not bit-exact: greedy word match with stock 0.49, vs 0.35 for FreeToken on the same model. A pass@1 coding test is next. First request after start is slow (2.7 tok/s) while RAM fills.
- Night 5 (same box, exact stack `-lzm off` + MTP): 21–25 tok/s bit-exact; the CPU's expert math is the bottleneck (32 of 49 ms).
- Details: [`docs/logs/report_night6.md`](docs/logs/report_night6.md), [before/after page](docs/night6_before_after.html), [journey Day 6](docs/JOURNEY.md). **Next:** the expert layer in Mojo.

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
