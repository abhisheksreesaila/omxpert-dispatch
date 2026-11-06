# Expert Dispatch: Night 2 report ("the slowest expert sets the pace")

**Verdict: NOT MEASURED.** None of the night-2 runbook ran.
- The workstation was busy with other sessions all evening: load average up to 43, with mxapp Mojo builds, mxframe agents and Playwright browsers. The owner stood the work down at 22:41 to avoid crashing the machine.
- The hourly quiet-checker logged two busy checks (23:54: load 6.7, GPUs 8.5/9.0 GB in use; 00:44: load 13.6). It then stopped without logging anything further. The cause is unknown: no error was written, and the machine did not reboot.
- At 07:30 the machine is quiet (load 0.5), but this check was told not to start heavy jobs.
- So there is no night-2 number to report, and none is invented below.

## P1–P5

| Prediction | Measured | Number | Verdict |
|---|---|---|---|
| P1: decode time tracks layers-with-any-miss better than total misses | not measured | — | pipeline never started (machine busy, then the checker stopped) |
| P2: set-completing pins beat frequency pins | not measured | — | same |
| P3: per-layer choice of whole layer vs pins beats both pure layouts | **partly, from the evening run before night 2** | 35B coding at 30% VRAM: 4 whole layers + pins **18.3 ms** vs pins only 20.4 vs whole layers only 21.2; at 60%: 20 whole layers + pins **13.5** vs 16.3 vs 14.6 | supported in that hand-picked sweep; the runbook's per-layer optimizer was not run |
| P4: nudge only to finish the set | not measured | — | needs the router-probability trace (tracer ready, not run) |
| P5: upload only if it pays | not measured with night-2 costs | earlier sim (night-1 constants, 35B, 30% VRAM): a simple "used ≥3 times in the last 16 tokens" rule +4–5% over pinned only; a perfect-foresight rule +7–8%; admit-on-miss −25% | direction supported by the earlier sim; the runbook's version (measured costs, set-completion bonus) was not run |

## Costs (`costs.json`)
Not measured; the Phase 1 script never ran. Earlier estimates, from the clean 35B runs on night 1:
- CPU expert read: ~0.056 ms.
- Upload: ~0.167 ms per 1.8 MB expert, i.e. ~3× a CPU read.
- Fixed GPU part: ~11.9 ms per token.
- Hop cost: ~0.15 ms per CPU-touching layer.

These are fitted estimates, not the runbook's direct measurements.

## Real-run table at 15% and 60%
Not run with the night-2 placements. The closest real measurements (35B, coding prompts, ms/token, night 1 / evening):

| VRAM | Frequency pins | Whole layers | No experts on GPU | Best new placement |
|---|---|---|---|---|
| 15% | 21.5 | 28.4 | 29.8 | not run (DP optimizer) |
| 60% | 16.4 | 14.6 | — | 13.5 (hand-picked 20 whole layers + pins; not the optimizer's plan) |

## Speculative experts (idea 26)
Not measured. The hidden-state tracer (`EDX_TRACE_HIDDEN`) and the predictor test (`scripts/spec_experts_sim.py`) are built but never ran.

## What did complete (before the stand-down)
**Soft steering toward pinned experts** (`EDX_STEER=λ`), Qwen3-Coder-30B, coding. Perplexity on held-out coding (lower is better); "whole layers" is llama.cpp today:

| VRAM | Plain pinned | λ = 0.5 | λ = 1 | λ = 2 | Whole layers |
|---|---|---|---|---|---|
| 15% | 36.8 ms, ppl 2.917 | 30.8 ms, 2.931 | **26.2 ms, 2.965** | 21.3 ms, 3.352 | 37.4 ms |
| 30% | 26.5 ms, 2.917 | 22.8 ms, 2.943 | **20.3 ms, 2.951** | 18.9 ms, 3.305 | 32.2 ms |

- **λ = 1 decodes 30–37% faster than llama.cpp's layer split for +1.2–1.7% perplexity.**
- Chat half: only plain pinned at 15% finished (33.1 ms, ppl 20.10; with a 16-per-layer minimum, 31.5 ms, 20.14). The rest was stopped by the stand-down.
- The hit rates printed as 0.0 for the steered configs are a stats-path bug in the tool, not a measurement; ignore them.

## Blockers
- **A dedicated machine.** Every timing in this project swings ±20% when other sessions run, and heavy runs alongside them risk a hang.
- **The hourly checker stopped silently after 00:44.** It needs a sturdier launcher, e.g. a systemd user timer or the T3 scheduler, instead of a background shell.

## Night 3 (or a dedicated slot)
1. Run `scripts/n2_resume.sh` with nothing else on the machine. All phases are written; about 2.5–3 hours.
2. Finish the chat half of the steering sweep, then make λ = 1 plus the per-use-case profile the default "fast lane" config.
3. The Mac test (M4 Pro, 16 GB unified memory): profile pinned in RAM (`mlock`) vs the OS page cache, with the SSD as the slow tier. It waits on the owner approving the machine link.
