# omxpert: Qwen 4's architecture on a desktop

Run **Qwen3.8-Flash-Next** (the Qwen 4 preview architecture, `qwen4exp`: 125B-parameter MoE plus a 51B n-gram table,
111 GB at 4-bit) on a desktop with **2× RTX 3090, 64 GB RAM and one NVMe**. You get the same model's output, faster than a stock
llama.cpp setup on the same box.

```bash
omxpert/run.sh          # starts an OpenAI-compatible server on http://127.0.0.1:8080/v1
omxpert/run.sh smoke    # starts it, asks one coding question, prints the answer and the speed, stops
```

It needs our llama.cpp build (the night-5 patch adds the expert prefetcher; the other flags work on stock llama.cpp builds that
support `qwen4exp` and MTP drafting) and the model in `~/models/Qwen3.8-Flash-Next-GGUF` (the script downloads it if missing).

## What it changes, and why
See `docs/report_night5.md` and `docs/visuals/night5_before_after.html` for the measured before/after of every trick.

## Credits
The systems ideas (parallel read pool, layer-ahead prefetch, exclusive tiers, staged prefill, CPU/GPU overlap, no per-layer
host sync) come from **0xSero**'s glm-flash-lite, which runs GLM-5.3-Flash at 19.5 tok/s on one RTX 3090. MTP drafting and the
lazy n-gram table are Qwen's and llama.cpp's. Usage profiles, the learned next-layer guesser and steering are omxpert's.
