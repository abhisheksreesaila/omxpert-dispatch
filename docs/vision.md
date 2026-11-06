# Expert Dispatch: vision (owner's words, condensed, 2026-11-04)

**Core idea: a learnable expert cache.** Which experts sit in VRAM, and how the router leans toward them, are trainable parameters: a small neural network inside the neural network. They are trained on the owner's own tasks, so the model uses only the experts it needs for ~80–90% of the work, with the rest kept as backup in RAM, SSD or another machine. Caches follow fixed rules; this one learns.

**Two lanes.**
- *Fast lane*: live chat, strategy, design. Always served from GPU-resident experts; never waits on a fetch.
- *Slow lane*: background and overnight work. It can pull experts from RAM, SSD or the network, because nobody is waiting on it. Local AI doesn't have to run at full speed 24/7.

**Overnight self-tuning.** While the owner sleeps, re-profile the day's work, retrain the residency and router bias, check quality on held-out data, roll back if it got worse.

**Borrow from serving systems:**
- speculative decoding: the fast lane drafts, the full model verifies;
- paged attention: experts as pages;
- prefix caching: per-task expert profiles;
- prefill/decode disaggregation;
- data, hybrid and expert parallelism;
- RDMA sharding across machines for the slow lane.

The software abstracts the complexity.

## Evidence so far (MORNING_REPORT.md)
- A hand-set router bias (one λ) already moves the hit rate 69% → 94% for +3.2% perplexity (OLMoE, 25% resident). The learnable version is `scripts/learn_cache.py`.
- Fine-grained MoEs specialize per task (Qwen 256 experts: +11 points for a task profile), and profiles drift within weeks, so nightly retraining is justified.

## Closest prior art (results/PRIOR_ART.md)
- ReMoE (router fine-tune for expert reuse)
- Cache-Prior (inference-time cache bias)
- CAR-MoE (RL-tuned routing)
- KTransformers/PowerInfer (frequency placement)
- llama.cpp `--moe-cache-mib` (LRU)

None jointly learns residency and bias under a VRAM budget, per user and per task, overnight, with fast/slow lanes.
