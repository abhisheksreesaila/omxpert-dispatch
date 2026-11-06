# Night 2 log: "the slowest expert sets the pace"

- 2026-11-04T22:17:55-08:00 START. Runbook: expert-dispatch-night2.md. Model: Qwen3.6-35B-A3B (UD-Q4_K_M).
- llama.cpp commit: bbb02c9 + EDX patches (pinned split, devmap, steer, prompt lineup, re-pin)
- GPUs: 0, NVIDIA GeForce RTX 3090, 3739 MiB, 24576 MiB;1, NVIDIA GeForce RTX 3090, 981 MiB, 24576 MiB;
- RAM: 62 GB total, 48 GB avail; disk free MODEL_DIR: 183G
- Note: Qwen3-Coder steering sweep (bench_steer.sh) still running; not disturbed. GPU timing phases wait for it.
- Design caveat recorded: llama.cpp runs a layer's GPU split and CPU split sequentially (scheduler), and EDX pinned mode always runs the CPU pass (dummy expert when no misses). The runbook's max() model will be tested against a sum() model in Phase 3.
- 2026-11-04T22:28:09-08:00 Paused: other sessions (mxapp mojo builds, mxframe, playwright) drove load avg to 43; stopped steer bench (chat half), requeued behind wait_quiet. Downstream phases gated on quiet.
- 2026-11-04T22:41:41-08:00 STAND-DOWN at owner's request: machine busy with other sessions tonight (risk of crash). All EDX jobs stopped, nothing queued. Resume with scripts/n2_resume.sh when the machine is dedicated.
- 2026-11-04T23:53:06-08:00 Scheduled night-2 pipeline to start at ~03:53 (owner's request), gated on quiet machine.
- 2026-11-04T23:54:33-08:00 hourly check: busy (load 6.70, RAM avail 38G, GPU 8524/9042 MiB)
- 2026-11-04T23:54:35-08:00 Replaced fixed 03:53 start with hourly quiet checks (owner's request).
- 2026-11-05T00:44:33-08:00 hourly check: busy (load 13.56, RAM avail 35G, GPU 8524/9179 MiB)
- 2026-11-05T07:31:12-08:00 MORNING CHECK: pipeline never started (busy at 23:54 and 00:44, then hourly checker stopped silently). NIGHT2_MORNING_REPORT.md written: night-2 predictions not measured. Scheduled check disabled.
- 2026-11-05 08:51 owner said run now; load 0.57, GPUs 55/974 MiB, 48 GB RAM free; launching n2_resume.sh in systemd user scope edx-night2
- 2026-11-05T09:22:12-08:00 NOTE: mxapp mojo builds (5 parallel, other session) pushed load to ~25 from ~09:05; steer chat sweep ran during it, so its ms numbers are load-contaminated (ppl still valid). Phase 1 is gated on wait_quiet and will wait.
- 2026-11-05T09:43:04-08:00 PHASE1 start
- 2026-11-05T09:45:43-08:00 PHASE1 done: { "model": "Qwen3.6-35B-A3B UD-Q4_K_M", "expert_MiB": 2.062, "c_cpu_ms_per_expert": 0.0982, "cpu_expert_GBps_eff": 22.0, "sync_ms_per_cpu_layer": -0.1831, "gpu_base_ms_k8": 5.6, "gpu_base_ms_k4": 8.44, "per_cpu_layer_ms_k8": 0.6025, "per_cpu_layer_ms_k4": 0.2097, "pcie_GBps_pageable_2MiB": 9.36, "pc
- 2026-11-05T09:52:45-08:00 costs refit: first fit gave sync -0.18 ms (outlier k=4 ncpu=10 22.47 ms under load); refit on fastest of each pair: {"model":"Qwen3.6-35B-A3BUD-Q4_K_M","expert_MiB":2.062,"c_cpu_ms_per_expert":0.0807,"cpu_expert_GBps_eff":26.8,"sync_ms_per_cpu_layer":0.0,"sync_ms_raw_fit":-0.0488,"gpu_base_ms_k8":5.53,"gpu_base_ms_k4":5.62,"per_cpu_layer_ms_k8":0.5968,"per_cpu_layer_ms_k4":
- 2026-11-05T10:14:06-08:00 PHASE2 start
- 2026-11-05T10:26:01-08:00 PHASE2 aborted at 10:22: every pinned run crashed (decode tracer: atof(NULL) in pin_init via prompt-blend path; and graph reuse dropped traced ids after 2 steps). Fixed in llama-moe-cache.cpp/llama-context.cpp, verified 63/63 records on a 64-token run. Phase 2 now times each run untraced (ms + start load) and traces separately (traced_times.log); validation drops runs started at load >= 4. Relaunched.
- 2026-11-05T10:45:02-08:00 PHASE2 start
- 2026-11-05T11:12:52-08:00 PHASE2 decode done
- P1 pin_coding: n=6861 mean 24.03 ms; misses/tok 176.5, layers-with-miss 38.7/40; R2 misses 0.875, layers-with-any-miss 0.288, max-miss 0.222, all 0.883
- P1 pin_mixed: n=6861 mean 26.19 ms; misses/tok 189.6, layers-with-miss 39.2/40; R2 misses 0.390, layers-with-any-miss 0.104, max-miss 0.066, all 0.391
- 2026-11-05T11:13:04-08:00 PHASE3 validation
- SIM sum:pin_coding: sim 20.45 ms vs measured 24.9 ms (-17.9%)
- SIM sum:pin_mixed: sim 21.58 ms vs measured nan ms (+nan%)
- SIM sum:layers: sim 27.67 ms vs measured nan ms (+nan%)
- SIM max:pin_coding: sim 19.87 ms vs measured 24.9 ms (-20.2%)
- SIM max:pin_mixed: sim 21.07 ms vs measured nan ms (+nan%)
- SIM max:layers: sim 27.67 ms vs measured nan ms (+nan%)
- 2026-11-05T11:13:05-08:00 PHASE4 sim
- 2026-11-05T11:14:26-08:00 P1 verdict: NOT SUPPORTED. Per-token time tracks total misses (R2 0.875 pin_coding, 0.390 pin_mixed) far better than layers-with-any-miss (0.288 / 0.104); consistent with Phase-1 sync ~0. Caveat: phase 2 ran under foreign load (mean load 8-13; only 2/90 runs started below 4), so validation used medians: measured pin_coding 23.4 < pin_mixed 25.0 < layers 26.6 ms vs sim(sum) 20.5 < 21.6 < 27.7: ranking matches, abs error -13% to +4%.
  always 15% {'freq_pins_global': 56.1, 'whole_layers': 36.1, 'no_gpu_experts': 31.9, 'dp_hybrid_freq': 56.1, 'dp_hybrid_freq_whole_layers': 0, 'dp_pins_freq': 56.1, 'dp_hybrid_setc': 56.0, 'dp_hybrid_setc_whole_layers': 0, 'dp_pins_setc': 56.0} (tok/s)
  always 30% {'freq_pins_global': 70.1, 'whole_layers': 41.7, 'no_gpu_experts': 31.9, 'dp_hybrid_freq': 70.1, 'dp_hybrid_freq_whole_layers': 0, 'dp_pins_freq': 70.1, 'dp_hybrid_setc': 70.1, 'dp_hybrid_setc_whole_layers': 0, 'dp_pins_setc': 70.1} (tok/s)
  always 60% {'freq_pins_global': 91.2, 'whole_layers': 60.1, 'no_gpu_experts': 31.9, 'dp_hybrid_freq': 95.7, 'dp_hybrid_freq_whole_layers': 11, 'dp_pins_freq': 91.3, 'dp_hybrid_setc': 95.7, 'dp_hybrid_setc_whole_layers': 10, 'dp_pins_setc': 91.8} (tok/s)
  skip 15% {'freq_pins_global': 56.7, 'whole_layers': 36.1, 'no_gpu_experts': 31.9, 'dp_hybrid_freq': 56.7, 'dp_hybrid_freq_whole_layers': 0, 'dp_pins_freq': 56.7, 'dp_hybrid_setc': 56.7, 'dp_hybrid_setc_whole_layers': 0, 'dp_pins_setc': 56.7} (tok/s)
  skip 30% {'freq_pins_global': 73.1, 'whole_layers': 41.7, 'no_gpu_experts': 31.9, 'dp_hybrid_freq': 73.1, 'dp_hybrid_freq_whole_layers': 0, 'dp_pins_freq': 73.1, 'dp_hybrid_setc': 73.1, 'dp_hybrid_setc_whole_layers': 0, 'dp_pins_setc': 73.1} (tok/s)
  skip 60% {'freq_pins_global': 108.7, 'whole_layers': 60.1, 'no_gpu_experts': 31.9, 'dp_hybrid_freq': 108.6, 'dp_hybrid_freq_whole_layers': 0, 'dp_pins_freq': 108.6, 'dp_hybrid_setc': 108.6, 'dp_hybrid_setc_whole_layers': 0, 'dp_pins_setc': 108.6} (tok/s)
- 2026-11-05T11:39:50-08:00 STOPPED Linux pipeline at owner's request (after Phase-4 sim; real Phase-4 runs, P4/P5 sims, spec-experts not run). Work moves exclusively to the Mac.
