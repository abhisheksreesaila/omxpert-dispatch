# omxpert: Qwen 4's architecture on a desktop

Run **Qwen3.8-Flash-Next** (the Qwen 4 preview architecture, `qwen4exp`: 125B-parameter MoE plus a 51B n-gram table,
111 GB at 4-bit) on a desktop with **2× RTX 3090, 64 GB RAM and one NVMe**.

```bash
omxpert/run.sh                      # steady stack (night 5): OpenAI-compatible server on http://127.0.0.1:8080/v1
omxpert/profile.sh                  # learn YOUR experts from your own chat history (or: omxpert/profile.sh my_text.txt)
STACK=pinned omxpert/run.sh         # learned stack (night 6): your profile's experts in VRAM, the rest in RAM, CPU and GPU at once
STACK=pinned omxpert/run.sh smoke   # start it, ask one coding question, print the answer and the speed, stop
MTP=1 omxpert/run.sh                # steady stack + Qwen's draft head: faster answers, slower prompts
```

## The learned stack (`STACK=pinned`), in one picture
- **Profile** (`omxpert/profile.sh`): runs the model over your own text and counts which of its 24,576 experts fire. On the
  owner's chat history, 30% of the experts served 79% of the picks in new conversations, and 2,000 tokens were enough to learn it.
  A profile from someone else's work (a coding corpus) served 44%: the profile has to be yours.
- **Exclusive tiers** (0xSero's idea): your hottest experts live in VRAM across both GPUs, every other expert in RAM, never both,
  so 48 GB of VRAM and ~40 GB of free RAM hold all 77 GB of experts.
- **CPU works while the GPU works** (FreeToken's hybrid executor, ported into llama.cpp's scheduler as `EDX_OVERLAP`): the few cold
  experts are computed on the CPU at the same time as the GPUs compute the hot ones.
- **Prompts on the right GPU** (`EDX_PREFILL_LOCAL`, `EDX_XDEV`): long prompts run each layer on the GPU that holds its experts.

Measured numbers and every trick's before/after: `docs/report_night6.md`, `docs/visuals/night6_before_after.html`.

It needs our llama.cpp build (`patch/llama.cpp-edx.patch`) and the model in `~/models/Qwen3.8-Flash-Next-GGUF` (the script
downloads it if missing). `PIN_MIB` sets the VRAM for pinned experts (36500 on 2× 24 GB).

## Credits
**0xSero**'s glm-flash-lite (GLM-5.3-Flash at 19.5 tok/s on one RTX 3090): exclusive tiers, layer-ahead prefetch, read pool, staged
prefill, CPU/GPU overlap. **FreeToken** (FlashML, Apache-2.0) and its Kai fork: the hybrid CPU/GPU expert executor. **Qwen**: the
model, MTP head and n-gram table. **llama.cpp / ggml**, **unsloth** (GGUF). Usage profiles, the learned next-layer guesser and
steering are omxpert's.
