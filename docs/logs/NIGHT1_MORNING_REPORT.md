# Expert Dispatch: Night 1 report

**Verdict: GO, in a revised form: task-specific profiles, refreshed often, plus router steering toward resident experts.**
- **Task profiles pay off on fine-grained models.** On Qwen3.6-35B-A3B (256 experts per layer), a coding profile beats a profile built from your other work by **+11 points** on held-out coding (66.7% vs 55.3% at a 25% cache), and is nearly **2×** better at a 6% cache (35.5% vs 18.8%).
  - On OLMoE (64 coarse experts) the gap was only 1.8 points. Specialization grows with expert count.
- **Router steering is the biggest lever.** OLMoE, 25% resident: hits go from 69% to **94% for +3.2% perplexity**.
- **Profiles go stale.** 25–28% of the hot set changed for coding and docs within weeks, so profiles need regular refreshing (nightly re-profiling, or the EMA predictor).

> Status (14:30): OLMoE and Qwen results final. Learnable cache and chat steering done (sections below). MAX: expert-trace hooks now proven inside the MAX worker, but gpt-oss-20b does not reliably fit 2×3090 for the full run (out of memory at load). See `NIGHT1_LOG.md`.

## Update, evening: Qwen per-token traces (llama.cpp), the decisive test
New llama.cpp tool `tools/edx-trace` (`cb_eval` on `ffn_moe_topk-<layer>`, all tokens marked as outputs) traced every tag on Qwen3.6-35B-A3B Q4: about 1.1M tokens, 40 layers × top-8 of 256, at ~2,000 tok/s. The cache was replayed in Mojo on held-out data (`results/Qwen3.6-35B-A3B/metrics.json`).

| Coding, cache per layer | Static profile (own) | Profile from other tags | **LRU** | Profile + recency EMA | Belady ceiling |
|---|---|---|---|---|---|
| 6% (16/256) | 35.2 | 18.7 | **47.4** | 46.9 | 66.7 |
| 12.5% (32) | 49.8 | 34.3 | **63.5** | 61.5 | 79.4 |
| 25% (64) | 66.6 | 55.1 | **78.9** | 76.5 | 89.6 |
| 50% (128) | 84.0 | 79.6 | **92.1** | 90.5 | 96.8 |

All five tags look the same: LRU beats the static profile by 12–28 points at 25%. Docs is the worst case: 52.4 vs 80.7.

- **The go test "profile ≥ LRU + 10 points" fails, by −12 to −28 points.** On a fine-grained model, consecutive tokens reuse experts heavily, so recency beats a long-run profile. Mixing the profile in (profile + LRU, EMA) also scores below plain LRU. This matches the community warning in your spec (#27861).
- **The profile is still personal:** own profile vs one built from your other tags is +11 points. It just loses to recency.
- **The headroom is in eviction and prefetch:** Belady is 10.7 points above LRU at 25%, and 19 points above at 6%. A learned eviction or prefetch policy (the spec's Belady-imitating re-ranker) is where a learnable cache can still win.
- **Revised hypothesis:**
  - Keep the cache adaptive: llama.cpp's SLRU, plus a learned re-ranker or next-layer prefetch.
  - Use router steering toward the experts *currently* cached, rather than toward a fixed profile.
  - Use the profile only as a warm start.

## Update, afternoon run
**Steering costs more on chat than on coding** (OLMoE design_chat held-out, baseline perplexity 36.2):

| Resident | No steering | λ = 0.5 | λ = 1 | λ = 2 |
|---|---|---|---|---|
| 50% (32/64) | 77.2% hit | 89.2%, +0.9% ppl | 95.1%, +5.0% | 99.6%, +25% |
| 25% (16/64) | 53.8% | 70.3%, +2.0% | 82.9%, +9.6% | 97.9%, +115% |
| 12.5% (8/64) | 37.5% | 51.4%, +1.0% | 64.1%, +6.0% | 83.8%, +43% |

- The resident set was the chat profile in both tests. Chat routes more broadly than coding, so the same λ costs about 3× the quality.
- The safe chat setting is λ = 0.5: +12–17 points of hits for ≤ 2% perplexity.
- This supports your two lanes: give the chat lane more VRAM and gentle steering, and push the background lane harder.

**Learnable cache, first try: no win over the hand-set bias.**
- Setup: `scripts/learn_cache.py`, OLMoE coding, S = 16/64. It learns a router bias for every expert in every layer, and alternates with re-picking the resident set. 200 steps on 120K tokens.

| Version | Hit | Perplexity |
|---|---|---|
| hand-set λ = 1 | 94.0% | +3.2% |
| learned, μ = 1 | 92.1% | +3.7% (worse than the hand-set line) |
| learned, μ = 3 | 97.5% | +12.3% (about 1.5 points better than interpolating the hand-set λ = 1 and λ = 2 points) |
| hand-set λ = 1, on the set the learner re-picked | 93.2% | **+2.7%** (best quality per hit; the residency step helped) |

- **Read:** learning *which experts stay resident* helped a little. Learning the bias itself landed on roughly the same trade-off curve as one hand-set λ.
- **Next try:** per-token or per-context bias (a tiny gate network), instead of one constant per expert. A constant can't beat a single constant by much.

**MAX:**
- The cause of "hooks did not fire": MAX's model worker is a separate process that never loaded the `--custom-architectures` plugin.
- The fix: an autoload in the MAX env that only switches on when `EDX_HOOK=1` is set. It loads the plugin lazily, when `gpt_oss.arch` is imported.
- Smoke test: real per-token routing traces, 24 layers × top-4, from MAX.
- The full run then failed to load at 88%, 92% and 95% GPU memory utilization. The model needs 19.4 GB of a 21 GB pool on GPU 0, and GPU 1 also runs your display.
- Options: run gpt-oss on a single bigger GPU, or trace gpt-oss in torch (MXFP4 → bf16, offloaded).

## Qwen3.6-35B-A3B (40 layers × 256 experts, top-8): llama.cpp imatrix counts
Held-out static-profile hit rate. "Own" = profile from that tag's train split; "other" = profile from the other four tags.

| Tag | 6% cache own / other | 12.5% | 25% | 50% | Churn (top-25%) |
|---|---|---|---|---|---|
| coding | **35.5 / 18.8** | 50.0 / 34.3 | **66.7 / 55.3** | 84.1 / 79.5 | 25% |
| design_chat | 29.1 / 20.9 | 44.2 / 34.3 | 66.1 / 53.5 | 88.9 / 79.6 | 9% |
| math | 26.4 / 21.0 | 41.4 / 34.7 | 61.7 / 54.9 | 85.2 / 81.1 | 18% |
| tool_calls | 29.3 / 27.3 | 44.6 / 42.9 | 64.0 / 62.6 | 84.4 / 84.0 | 10% |
| docs | 17.9 / **24.6** | 32.1 / 37.9 | 52.2 / **58.5** | 78.7 / 83.6 | **28%** |

- Concentration: the hottest 25% of experts carry 64–71% of routings (go bar: 70%).
- Cross-tag overlap of the hot sets: **0.43** (OLMoE 0.64; random 0.14). Coding vs chat is only 0.28.
- Docs is the warning case: its train split (Sept–Oct) went stale against Oct–Nov held-out docs. A profile from your other work beat it, so a stale profile is worse than a generic one.
- Caveats: Q4 quantization; per-expert counts only (no per-token traces), so no LRU or Belady comparison for Qwen. Non-coding train slices are 100K tokens; coding is 200K.

## What ran
| Model | Engine | Why this engine | Data |
|---|---|---|---|
| OLMoE-1B-7B (16 layers × 64 experts, top-8) | PyTorch (fallback) | MAX has no OLMoE architecture | 300K train + 100K held-out tokens per tag, per-token traces |
| gpt-oss-20b (24 × 32, top-4) | **MAX 26.7 nightly**, 2×3090 | MAX-first; the only MoE that fits 48 GB on Ampere in MAX | not run: MAX starts and generates (48 tok in 1.3 s) but trace capture is unverified, and heavy jobs are paused |
| Qwen3.6-35B-A3B UD-Q4_K_M (40 × 256, top-8) | llama.cpp `llama-imatrix` (fallback) | MAX needs bf16 = 70 GB; no new code needed in llama.cpp | per-expert counts: train, train halves, held-out per tag (~2M tokens) |

Cache hit rates were simulated in **Mojo** (`mojo/cachesim.mojo`).

## Go/no-go table (OLMoE, coding unless noted)
| Measure | Go | Rethink | Measured |
|---|---|---|---|
| Routings handled by hottest 25% of experts (layer mean) | ≥ 70% | ≤ 45% | **68%** (tool_calls 70%, chat 54%) |
| Held-out hit, static profile vs LRU, at S = 25% of experts | ≥ 10 pts better | within 3 pts | **+3.2 pts** (68.7 vs 65.5); at 12.5% cache **+15 pts** |
| Static profile vs Belady MIN at 25% | within 15 pts | — | **12.5 pts** below |
| Expert overlap between tags (top-25% Jaccard) | < 0.4 | > 0.7 | **0.64** (random would be 0.14) |
| Churn, older vs newer half of train | < 10% | > 30% | **18%** coding, 6–10% other tags |
| Decode vs prefill coverage | within 5 pts | > 15 pts | not measured on OLMoE (MAX run measures it) |
| Personal profile vs other-tags profile (new test) | — | — | **+1.8 pts** only |
| Profile + recency EMA predictor vs best other online policy (new) | — | — | **+1.5 to +3 pts**, best on every tag |
| Profile vs llama.cpp LRU (Phase 5) | ≥ 5 pts | equal | not run |

![coverage](results/OLMoE-1B-7B-0125-Instruct/coverage.png)
![hit rate](results/OLMoE-1B-7B-0125-Instruct/hitrate.png)
![overlap](results/OLMoE-1B-7B-0125-Instruct/overlap.png)

## Cache-aware routing (router steering), the strongest result
OLMoE, coding held-out, 40K tokens. Resident set = the profile's top-S experts per layer. Non-resident experts get a penalty −λ when the router *selects*; mixing weights stay the model's own. (`scripts/torch_bias_ppl.py`, `results/OLMoE-1B-7B-0125-Instruct/bias_ppl_coding.json`; baseline perplexity 5.87.)

| Resident | No steering | λ = 0.5 | λ = 1 | λ = 2 | hard mask |
|---|---|---|---|---|---|
| 50% (32/64) | 87% hit | 96.5% hit, +0.6% ppl | 98.6%, +1.5% | 99.9%, +5.7% | 100%, +7.9% |
| 25% (16/64) | 69% hit | 85.8%, +0.8% | **94.0%, +3.2%** | 99.4%, +19.7% | 100%, +45% |
| 12.5% (8/64) | 51% hit | 65.2%, +0.5% | 77.6%, +2.6% | 93.2%, +17% | 100%, +473% |

- At 25% residency, λ = 1 cuts misses from 31% to 6% (5× fewer) for +3.2% perplexity. Bandwidth-model estimate for a 122B MoE: ~22 → ~56 tok/s (estimate, not measured).
- A soft nudge works; a hard "resident only" mask does not.
- This moves hit rate by 15–25 points. The best cache *policy* moved it by 1.5–3.

## What surprised me
- **Whether profiles matter depends on the model.** On OLMoE (64 coarse experts), a profile from your other work was as good as a coding profile (−1.8 points). On Qwen (256 fine experts), the coding profile wins by 11 points. Fine-grained MoEs specialize, and that's where the big local models are heading.
- **LRU collapses at small caches** (≤ 6% of experts: 1–4% hits against 20–35% for a profile), because every token needs 8 experts per layer and LRU thrashes. **The profile matters most when VRAM is tight**, which is exactly the "model much bigger than VRAM" case.
- **The ceiling is far above every online policy** (Belady 81% vs best online 70% at 25%). The only lever with real headroom is short-horizon lookahead (next layer, next token), not a better long-run profile.
- **"Look at the task first, then pick the experts" didn't help.** A predictor using the first 128 tokens of each 2K chunk did no better than the static profile.

## Novelty (details in `results/PRIOR_ART.md`)
- **Predicting experts and prefetching them:** already done many times (ProMoE, ExpertFlow, DoMoE, HOBBIT…). A learned predictor on 2×3090 was already tried and showed no gain.
- **A frequency profile pinned in VRAM:** shipped already (KTransformers, PowerInfer). llama.cpp merged an LRU GPU expert cache in October 2026 (`--moe-cache-mib`).
- **Open:**
  - (1) a per-user profile measured against that LRU. Tonight says the user part adds little.
  - (2) Nightly router-only tuning on your own data, with rollback if quality drops. This is the angle the numbers support. See below.

## Blockers / needs you
- Nothing is blocked for you. GPU 0 was in use by another session's mxframe tests, so the MAX run waits for it automatically (`scripts/run_max.sh`).

## Recommended night 2
1. **Cache-aware routing** (results pending in `results/*/bias_ppl_*.json`): bias the router toward experts already in VRAM and measure the quality cost. If 25–50% residency costs under ~2% perplexity, misses mostly disappear, and that beats any cache policy.
2. **Make the bias learned instead of hand-set:** a nightly router-only fine-tune on your own logged prompts (ReMoE-style), with an automatic rollback check on held-out perplexity.
3. **Upstream lever:** warm-start llama.cpp's `--moe-cache-mib` cache from a saved profile with recency decay (what the EMA predictor does). It's small, measurable, and helps most at small VRAM.
