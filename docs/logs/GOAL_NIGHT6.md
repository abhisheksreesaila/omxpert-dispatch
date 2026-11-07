# Night 6 (autopilot, 2026-11-06 evening): three sets of tricks, one engine, Qwen3.8-Flash-Next

**End goal (owner's words, condensed):** combine the tricks from all three of us (0xSero's systems tricks, FreeToken's
engine where the CPU works while the GPU works, and our profiling from the owner's own chat history plus prediction) on
Qwen3.8-Flash-Next, the only model kept, and implement it the fastest way possible, llama.cpp or FreeToken, so that when
Qwen 4 ships this can be published. Mojo comes after.

## Milestones (each checked by `scripts/check_night6.sh`)
1. **Baseline:** our night-5 llama.cpp stack measured by one client shared with every engine tonight (same prompts,
   streaming, decode + prompt tok/s, ≥2 runs). (`results/night6/baseline.json`)
2. **FreeToken runs it:** FreeToken (upstream or the Kai fork) serves Qwen3.8-Flash-Next on 2×3090 + 64 GB RAM, answers
   the smoke prompt, measured by the same client. (`results/night6/ft_stock.json`)
3. **Profile from the owner's chat history:** expert trace of the model on the owner's own chat history (local only, text
   never committed), held-out hit curve with ≥3 points. (`results/night6/profile_chat.json`)
4. **Combined:** every trick has a row (FreeToken hybrid, profile-ordered residency, predictor prefetch, MTP, 0xSero's
   read pool / layer-ahead / staged prefill / exclusive tiers, steering): measured or not built with a reason; ≥3
   measured with paired comparisons on one engine. (`results/night6/combined.json`)
5. **Same intelligence:** the final stack's answers checked against the reference (KL, or greedy token match on fixed
   prompts vs the llama.cpp reference); lossy tricks flagged. (`results/night6/quality.json`)
6. **Run it + report:** `omxpert/run.sh` starts the fastest stack (smoke ok, `results/night6/stack.json`);
   `docs/report_night6.md` credits 0xSero and FreeToken; before/after visual `docs/visuals/night6_before_after.html`.

**Done means:** `scripts/check_night6.sh` prints PASS ×6.

## Out of bounds
No sudo, no paid services, no secrets. Nothing pushed (night 6 stays local until the owner says yes). The owner's chat
text never leaves this machine and is never committed: only expert counts are saved. Delete nothing outside
~/expert-dispatch, ~/models and FreeToken's own caches. Shared box: `nice -n 10`, keep RAM available ≥ 8 GB, stop if
free disk < 20 GB or a GPU > 85 °C for 5 min.

## Assumptions (instead of asking)
- Model: the UD-Q4_K_XL GGUF already on disk (111 GB). If FreeToken can't serve it, download its known-good NVFP4
  checkpoint (disk now 154 GB free) and say so.
- Engine: FreeToken-Kai fork first (it has the 64 GB-RAM mode, `--moe-bank-ram`, and the two-GPU split); upstream if
  the fork fails. Installed in its own pixi environment (no pip).
- "Same intelligence" reference: our llama.cpp stack on the same GGUF (bit-exact to stock). Different engines and
  kernels are never bit-exact; judge by KL / token match against a measured floor, as in night 5.
- Chat history = the owner's Claude Code transcripts on this machine (their prompts and the answers), the closest thing
  to "our own chat history".
- Video default (steady vs MTP) is decided by measurement in the final stack, not by taste.

## Budget
120 turns or 8 hours, whichever first.
