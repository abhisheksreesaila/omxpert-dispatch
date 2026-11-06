# Night 3 (2026-11-05/06): pushing learnable experts

## Real-text work lane (Mac, gpt-oss-120b, llama-server, 8 graded coding tasks)
- steer λ=1, one job at a time: **7/8 tasks pass hidden tests**, 2,364 tokens in 1,736 s = 1.36 tok/s end to end (prompt processing included).
- (8 jobs at once, earlier: ~1.0 tok/s total, RAM thrash; batched-bench's 7.0 was random tokens.)
- Profiling cost on the Mac: the tracer read 25k coding tokens of gpt-oss-120b in ~20 min (21 tok/s, batch prefill).
- A profile from only 1,000 tokens: its top-16 experts per layer cover 65.9% of routings.

## Steering-aware adapter (Linux GPU 0, OLMoE-1B-7B, S=16 of 64 resident, coding profile)
Low-rank residual per MoE block (rank 32, 2.1M params, base frozen), trained by distillation (KL to the unsteered model) on coding train text.
- Adam lr 2e-3 and 3e-5 diverged (ppl 800k; or ppl *below* the full model with KL to it rising: a broken optimum, discarded). Gradient check: the gradient is right, the surface is sharp (SGD 1e-4 descends cleanly; Adam 3e-6 stable).
- λ=1, Adam 3e-6, 400 steps (80 s): KL to full model coding 0.103 -> 0.098, chat 0.319 -> 0.242. Perplexity: coding steer 6.056 -> 5.605 (full 5.868), chat 41.50 -> 36.37 (full 36.25).
- CAVEAT: the adapter also lowers perplexity *without* steering (coding 5.48, chat 33.3, below the full model) while KL to the full model barely moves. So the ppl gain is not only "correcting steering"; unexplained. KL to the full model is the honest metric.
- **Hard mask (only resident experts, 100% hits, zero SSD reads), S=16/64, 900 steps on 240k tokens (~3 min):** coding ppl 8.531 -> **5.833** (full model 5.868); KL to full 1.09 -> 0.52 (halved). Chat (out of domain): 173.3 -> 54.3 (full 36.2), KL 2.62 -> 1.37.
- λ=2: coding 7.026 -> 5.608, KL 0.519 -> 0.361; chat 94.6 -> 45.6.
- Read: a 2.1M-param adapter trained ~3 min on your coding text recovers most of the loss from never touching 75% of the experts, in domain. Out of domain it helps but does not close the gap, so an adapter belongs to its profile (coding set + coding adapter).
- Sweep (hard mask, coding ppl; full 5.868): S=8/64 33.63 -> **6.63** (KL 2.92 -> 0.79); S=16 rank 8 (0.5M params) 5.93; rank 32 5.83; rank 128 5.62 (KL 0.48); S=16 rank 32 on only 30k tokens / 116 steps / **19 s**: 6.30 (KL 0.65).
- Bundles (hard mask, S=16, rank 32): coding set + coding adapter: coding 5.83, chat 54.3. Chat set + chat adapter: coding 6.91, chat **42.7** (full 36.2; unadapted 200.8). The matching bundle wins on its own work both ways: profile = expert set + adapter.
- Oddity: on chat text, the chat-profile mask (200.8) is worse than the coding-profile mask (173.3) before adapting; unexplained (chat train/held-out drift?).
- Leakage check: 0.6% of held-out coding 64-char windows appear verbatim in train (chat 15.3%), so leakage does not explain the adapter's below-full perplexity; still unexplained.
- S=8/64 with soft steering + adapter: λ=2: hits 93.2%, coding 6.874 -> **5.532**, KL 0.422 -> 0.277. λ=1: hits 77.6%, 6.022 -> 5.556.
- Why the adapter beats the full model: on held-out coding, adding the adapter to the unsteered model gains 0.092 nats/token, KL(full || full+adapter) 0.068; the gain is spread over ordinary coding tokens ('\n' 2.8%, 'tools', '-', '.', 'import', 'const'...), not one token. It behaves like a light adaptation to the owner's coding style, even though it was trained only to imitate the full model. Mechanism still unproven.

## The spec's "learnable cache": which experts should be resident? (OLMoE, hard mask, coding / chat held-out ppl)
| Resident set | S=16 of 64 | S=8 of 64 |
|---|---|---|
| counts (routing frequency, what we used) | 8.73 / 167 | 39.1 / 1922 |
| **gate** (router-weighted frequency, REAP's gate half) | **8.43 / 152** | **31.6 / 1827** |
| taylor (first-order loss rise if removed) | 30.2 / 236 | 154 / 982 |
| learned gates (hard-concrete-style, KL + budget) | 27,510 (no binarisation); 4,012 (with binarisation penalty) | not run |
- Learned gates failed: the soft relaxation keeps partly-open gates, so the trained soft model leans on experts the hard mask removes (the spec's soft-to-hard warning). A hard mask gives no gradient for "bring this expert in".
- Renormalising weights over residents is far worse than keeping the model's raw weights (coding 47.2 vs 8.7 at S=16), matching the handbook's "never rescale over survivors".
- Taylor (per-expert removal cost) ignores that masked tokens re-route to other experts; much worse.
- Gate-weighted set + adapter (hard mask, rank 32, 3 min): S=8 coding 6.566 (counts+adapter 6.634), S=16 5.756 (5.833). Small extra gain; the adapter does most of the work.
- **Swap search learns a better resident set (S=8, 3 rounds, 4 candidates, 2.6 min):** start from the gate set, keep a swap only if KL to the full model drops on a 4k-token train slice. Coding ppl 31.6 -> **18.1**, chat 1,827 -> **591** (generalises to text it never searched on); 16% of the set changed; KL 3.86 -> 2.27 and still falling. First learned residency method that beats counting.
- Queued (Linux, unattended): S=16 search; adapter at S=4; rank-8 adapter at S=8; S=8 λ=2 + adapter; longer adapter training; then a 10-round search at S=8 and the adapter on the searched sets (results/night3_linux.log).
- Round 1–2 results (hard mask unless noted; coding / chat ppl; full 5.868 / 36.25):
  - S=16: gate 8.43 / 152 → search (3 rounds, 8% changed) **7.53 / 115.8**; + adapter **5.67** / 51.8.
  - S=8: gate 31.6 / 1,827 → search 3 rounds 18.1 / 591 → 10 rounds (converged at round 7, 23% changed, 24 min) **14.75 / 520**; + adapter 6.44 / 71.8 (gate + adapter 6.57; gate + adapter on 2× data 6.28).
  - S=4 (6%): counts 2,791 / 10,022 → + adapter **8.31** / 102.
  - S=8 λ=2 (93% hits) + adapter: 5.51 / 39.1. Rank-8 adapter at S=8: 6.79.
  - Read: the learned set does most of its work before the adapter (31.6 → 14.75); the adapter then closes most of the rest. On top of a searched set the adapter's extra gain is smaller (6.57 → 6.44): the two partly overlap.
- Round 3: S=4 (6%) search converged at round 5 (42% of the set changed, 16 min): coding gate 2,422 → search **130.7**; + adapter (2× data, 5 min) **7.22** / chat 76.1. S=8 search + adapter (2× data) **6.25** / 68.4. S=8 search λ=2 + adapter 5.53 / 39.0.

## Readable scorecard (held-out next-token accuracy, hard mask; `results/OLMoE-1B-7B-0125-Instruct/bundles_scorecard.json`)
| Config | code ppl | code top-1 | agrees with full | chat ppl | chat top-1 |
|---|---|---|---|---|---|
| full model | 5.87 | 69.1% | 100% | 36.2 | 41.5% |
| 16/64 counted | 8.53 | 64.7% | 75.3% | 173.3 | 30.0% |
| 16/64 learned set | 7.53 | 65.7% | 76.4% | 115.8 | 32.7% |
| 16/64 learned + adapter | 5.67 | **68.0%** | 81.9% | 51.8 | 37.5% |
| 8/64 counted | 33.63 | 47.8% | 51.4% | 1540 | 11.9% |
| 8/64 learned set | 14.75 | 57.9% | 63.4% | 520.4 | 20.8% |
| 8/64 learned + adapter | 6.25 | **66.2%** | 76.7% | 68.4 | 34.5% |
| 4/64 counted | 2791 | 9.2% | 9.6% | 10022 | 2.9% |
| 4/64 learned set | 130.7 | 30.1% | 31.4% | 2240 | 7.9% |
| 4/64 learned + adapter | 7.22 | **64.0%** | 71.3% | 76.1 | 33.5% |
