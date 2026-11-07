#!/bin/bash
# Qwen3.8-Flash-Next (qwen4exp) UD-Q4_K_XL with its bundled IQ4_NL engram table + MTP drafter (Q8_0)
# xet transfer ran at ~250 KB/s on 2026-11-06 while plain HTTPS gave 114 MB/s, so xet is off
cd ~/models/Qwen3.8-Flash-Next-GGUF
export HF_HUB_DISABLE_XET=1
HF=~/expert-dispatch/.pixi/envs/default/bin/hf
nice -n 10 $HF download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" --include "mtp-Qwen3.8-Flash-Next-Q8_0.gguf" --local-dir . && touch ~/expert-dispatch/dl_q38.done
