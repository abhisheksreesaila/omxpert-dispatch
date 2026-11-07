# Night 6 report: three sets of tricks, one engine (2026-11-06 22:50 → 2026-11-07, autopilot)

**Goal (owner's words, condensed):** combine the tricks from all three of us (0xSero's systems tricks, FreeToken's engine where the
CPU works while the GPU works, and our profiling from the owner's own chat history plus prediction) on Qwen3.8-Flash-Next, the only
model kept, the fastest way possible, llama.cpp or FreeToken, so it can be published when Qwen 4 ships. Mojo after.

**Checker** (`scripts/check_night6.sh`): see the end of this report.

## The answer in four lines
1. **The combined stack beats stock llama.cpp on the same box and model:** decode **22.2 / 21.1 tok/s vs 17.7 / 15.7**, prompts
   **218 / 202 vs 108 / 83 tok/s** (two interleaved pairs of servers, the owner's held-out chat prompts, server timings).
   That is about +30% decode and 2× prompt reading, **on the kind of text the profile was learned from**. On other text (coding
   prompts, one round each) the same stack ran slower than stock (10.0 vs 14.9 / 13.8 tok/s): the profile has to match your work.
   Outputs are not bit-exact (greedy text diverges like it does on another engine; see "Same intelligence?").
2. **The learning is real and personal:** a profile from the owner's own chat history puts 30% of the experts in front of 79% of the
   picks in *new* conversations (ceiling 83%); a profile from someone else's work (night 5's coding corpus) reaches 44%. 2,000 tokens
   of history are enough.
3. **FreeToken's engine, as is, loses here (1.4 tok/s)**, because a 64 GB desktop leaves it ~25 GB of RAM for experts and its VRAM
   cache can only hold experts that are also in RAM. So its idea went into llama.cpp instead, where tiers are already exclusive.
4. **What made the difference was not the overlap itself (+2-5%) but fixing how pinned mode reads prompts:** night 5's pinned run
   was losing to 3,148 page faults per token, all from prompt batches; three fixes took it to 48.

## The stack, as data flow (database terms)
| | Before: llama.cpp's fit | After: the learned stack (`STACK=pinned omxpert/run.sh`) |
|---|---|---|
| Partitioning | by layer (range partitioning): layers 1-24 on the GPUs with all 512 experts each, 25-48 in RAM | by access frequency (hot/cold tiering): your profile's hottest experts of all 48 layers on the GPUs (36,500 MiB), the rest in RAM, never both |
| Who computes a token | GPU for half the layers, CPU for the other half (32 of 49 ms), in turn | GPUs compute ~94% of held-out chat picks (from the hit curve); the CPU computes the cold ones *at the same time* (parallel query) |
| Prompts | each GPU computes its own layers | each layer runs on the GPU that holds its pins; pinned experts are copied GPU-side, only cold ones come from RAM |

## What each trick did (measured on this box)
| Trick | From | Result | Verdict |
|---|---|---|---|
| Profile from your own chat history picks the VRAM experts | omxpert | 30% of experts → 78.9% of held-out chat picks (coding profile 43.7%, ceiling 82.9%); with the fixes below, stack vs fit +26% / +35% decode, ~2× prompts | **keep: the core of the stack** |
| Exclusive tiers (pinned experts' RAM given back) | 0xSero | what lets 48 GB VRAM + ~40 GB free RAM hold all 77 GB of experts; needed the three prompt fixes to stop faulting | keep |
| CPU works while GPU works (`EDX_OVERLAP`) | FreeToken (FlashML) | same-server A/B 24.42 vs 24.21 tok/s, won 5/6 rounds, ~+2-5% after run order | keep (small here: little CPU work left to hide) |
| Copy pins across GPUs for prompts (`EDX_XDEV`) | tonight | faults 3,148 → 189 per token, prompts 9.8 → 21.2 tok/s | keep |
| Each layer's prompt work on its own GPU (`EDX_PREFILL_LOCAL`) | tonight | faults → 48 per token, prompts → 92.8 (client) / 202-218 (server timing) | keep |
| FreeToken engine as is (Kai fork, 64 GB mode) | FreeToken | 1.37 tok/s decode, 17.3 prompt; ordered by our profile ~0.8-1 (stopped early) | not on a 64 GB desktop |
| Layer-ahead prefetch (previous token's cold picks) | 0xSero idea, our reader | 20.63 vs 22.21 and 19.74 vs 21.09 tok/s (paired rounds 0.90-1.26) | off: costs ~6% here |
| Learned next-layer guesser | omxpert | offline on chat: 44.5% of next-layer picks with 16 guesses (49.1% with recency); not wired into pinned mode (it would pull pinned experts back into RAM) | not built in the stack |
| MTP drafting on the stack | Qwen / llama.cpp | same-server A/B (30,000 MiB of pins, draft head on GPU 0): 21.74 vs 23.0 tok/s, paired 0.79 / 0.92 / 1.30 / 1.07 | off on this stack (night 5: ×1.36 on the fit layout) |
| Steering λ0.5 (swap a pick for a VRAM expert) | omxpert (cf. Skliar et al.) | on coding prompts (profile mismatch): 16.9 vs 10.0 tok/s without it (stock 14.9 / 13.8); greedy word match vs stock 0.53 (stack 0.49) | opt-in when your work drifts from your profile; lossy |
| Staged prefill | 0xSero | not rerun (night 5: no gain on ~550-token prompts) | not needed |
| Hot/cold split for prompt batches too (`EDX_PIN_MAXTOK`) | tonight | CUDA illegal memory access in the quantized expert kernel (as night 5) | not built |

## What we learned that we did not expect
1. **The profile is personal after all.** Night 1 (OLMoE) said hot experts were generic. On Qwen 4's 512 small experts per layer,
   someone else's profile is barely better than random on your work, and your own is 79%.
2. **FreeToken's design assumes RAM ≥ experts.** Its VRAM cache is a cache of RAM. On 128 GB it is fast (the Kai fork reports 18-20
   tok/s on two 12 GB cards); on this 64 GB desktop with a desktop session running it is 15× slower than llama.cpp.
3. **Night 5's "pinning doesn't pay" was a prompt-path bug, not a property of pinning.** Prompt batches all ran on GPU 0 and
   re-read GPU 1's pinned experts from disk.
4. **A profile is a bet on your work.** On the owner's chat it pays ~30%; on coding prompts it cost ~30% (one round). Re-profile
   when the work changes; steering softens a mismatch.
5. **The overlap is worth little once the profile is good.** With ~90% of picks on the GPUs, the CPU's share is a few experts per
   token; hiding it saves 2-5%. The big lever is *which* experts sit in VRAM.

## Same intelligence?
Greedy outputs on 3 held-out coding prompts × 128 tokens, compared word by word with stock llama.cpp (`results/night6/quality.json`):

| Run | Word match with stock |
|---|---|
| stock again (separate process) | 1.00 (identical) |
| FreeToken on the same GGUF (another engine, same model) | 0.35 |
| the pinned stack | 0.49 |
| the pinned stack + steering λ0.5 | 0.53 |

The stack is **not bit-exact**: the hot experts run on the GPU and the cold ones on the CPU, so the sums round differently and a
close call between two tokens can flip (prompt 0's first token: stock `null;`, the stack and FreeToken `;`). After a flip greedy
text goes its own way, so word match is a harsh measure. It diverges less than another engine on the same model does. This is not a
quality benchmark; a pass@1 on real coding tasks is the honest next test.

## Decisions worth a second look
- **D-03:** pivoted from FreeToken to llama.cpp after two FreeToken runs at ~1 tok/s. Porting exclusive tiers into FreeToken's
  CUDA-graph slot cache is the other path; it would keep FreeToken's faster kernels and its prompt path.
- **D-05:** "same intelligence" is a greedy word-match on 3 coding prompts, a weak test; llama-perplexity's KL can't see the pinned
  path (it only runs on ≤8-token batches).
- **D-06:** your chat profile stays on this machine; the public launcher ships `omxpert/profile.sh` so each user learns their own.

## Run it
```bash
omxpert/profile.sh                   # learn your experts from your chat history (minutes; counts only, no text kept)
STACK=pinned omxpert/run.sh smoke    # start, ask one coding question, print answer + speed, stop
STACK=pinned omxpert/run.sh          # serve on http://127.0.0.1:8080/v1
```
Smoke test (`STACK=pinned omxpert/run.sh smoke`): answered with a correct iterative Fibonacci function. **The first request after start
is slow (2.7 tok/s decode)** while the cold experts come off the disk into RAM; warm it with one request before recording a video.
If your work changes (new language, new kind of task), run `omxpert/profile.sh` again: on text unlike the profile the stack was
slower than stock (10.0 vs 14.9 / 13.8 tok/s on coding prompts with the chat profile), and steering (`STACK_ENV` + `EDX_STEER=0.5`)
recovered it to 16.9 at a small quality cost.

## Needs you
- Nothing was pushed: night 6 is local only (`~/expert-dispatch`; patch, scripts and results ready to copy to the public repo).
- A yes to publish night 6 (public repo + journey page Day 6).

## Credits
**0xSero** (glm-flash-lite): exclusive tiers, layer-ahead prefetch, read pool, staged prefill, CPU/GPU overlap. **FreeToken** (FlashML,
Apache-2.0) and yuuki-net's **FreeToken-Kai** fork: the hybrid executor (CPU computes misses while the GPU computes hits), its 64 GB
bank mode, and the measurements that pointed at RAM; its idea is re-implemented in llama.cpp's scheduler here. **Qwen**: model, MTP
head, n-gram table. **llama.cpp / ggml**, **unsloth** (GGUF). Profiles from your own chat history, the learned guesser, steering and
the prompt-path fixes are omxpert's.

## Next
Mojo: the expert layer itself (hot experts on the GPUs, cold on the CPU, one kernel launch per layer), where llama.cpp's per-split
scheduling and the >8-token kernel crash go away; then feed the learned guesser hidden states (Fate's cross-layer gate) for the
cold picks.

## Checker
```
PASS 1 baseline
PASS 2 freetoken runs
PASS 3 chat profile
PASS 4 combined
PASS 5 quality
PASS 6 run + report
```
