# Night 3 report (autopilot, 2026-11-06 01:30 → 06:00)

**Goal:** push learnable experts as far as possible. **Checker:** `scripts/check_night3.sh` (output at the end).

## The headline
**Learn the experts, don't count them.** On two small MoE models, a learned *bundle* (a searched expert set + a 2M-parameter adapter) runs on a fraction of the experts at close to full quality, scored on your held-out code:

| | Counted set | Learned set | Learned set + adapter | Full model |
|---|---|---|---|---|
| OLMoE, 8 of 64 experts | 47.8% next token right | 57.9% | **66.2%** | 69.1% |
| OLMoE, 4 of 64 | 8.0% | 19.6% | **62.3%** | 69.1% |
| Granite 3B, 8 of 40 | 33.7% | 43.8% | **57.6%** | 68.7% |

A usable bundle takes ~90 s of compute (1 search round + a short adapter).

## On the 120B (Mac, 16 GB)
- 8 experts per layer at λ=1: **2.45× faster for +5.4%**; 32 per layer at λ=0.5: 1.35× for +0.5%.
- The right profile matters: steering coding work to the chat profile is slower and costs +34.5% (vs +8.7%).
- 5,000 tokens of your history (~4 min of profiling) is enough to learn the profile.
- Real graded coding tasks: 7/8 and 8/8 pass; running 4 jobs at once is slower than one at a time.

## Dead ends (logged)
Gradient-learned gates (soft-to-hard gap), Taylor importance (ignores re-routing), co-optimising search with the adapter (no improving swap), renormalising weights over loaded experts (much worse).

## A mistake, caught
The first 4-of-64 numbers leaked: with fewer loaded experts than the model picks (8), the mask filled the gap with unloaded experts at full weight. Found at 03:00, the live site was corrected immediately, the code fixed (unloaded slots get weight 0), and 4-of-64 re-measured: 62.3%, not 64.0%. 8- and 16-of-64 were unaffected.

## Three decisions worth a second look
- D-01: OLMoE/Granite on Linux for anything trained or searched; the 120B only for inference sweeps (the Mac can't do thousands of forward passes).
- D-04: the site shows next-token accuracy, not perplexity (readable); perplexity stays in the logs.
- D-05: correct the public site the moment a number is found wrong, before re-measuring.

## Next
Bring bundles to the 120B: a llama.cpp hook for the adapter residual, plus a bigger training box (or distil on gpt-oss-20b first).
