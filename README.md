# omxpert dispatch

> **An omxlabs research project.** An open experiment, not a product.

**Can a 125B MoE run well on a desktop?** Can we optimise Qwen3.8-Flash-Next (125B mixture of experts, the Qwen 4 preview architecture) for local AI on 2× RTX 3090 (48 GB of GPU memory) + 64 GB RAM, and could smaller setups work too?

**The idea.** Mixture of experts is the baseline now: nearly every new model is one, and each token wakes only a few of its thousands of experts. So why carry every generic expert at full cost? Choose our own experts: keep the ones my work actually uses, specialised and personal, close to the GPU, and make the rest cheap.

## What we tried (latest five days)

| Day | What we tried | Result | Verdict |
|---|---|---|---|
| [Day 8](docs/JOURNEY.md) · Nov 18 | Plan: prune, repair, reward-trained router, lookups, a nightly gate | no result yet | next |
| [Day 7](docs/JOURNEY.md) · Nov 16–17 | Expert layer in Mojo, on llama.cpp's own CUDA stream | 31.9 → 49–54 tok/s | kept |
| [Day 6](docs/JOURNEY.md) · Nov 6–7 | Profile from my chats + exclusive tiers + CPU while GPU | 17.7 → 22.2 tok/s vs stock | kept |
| [Day 5](docs/JOURNEY.md) · Nov 6 | Qwen 4 architecture on 2× 3090, every trick measured | 21–24 tok/s, bit-exact | kept |
| [Day 4](docs/JOURNEY.md) · Nov 6 | Per-expert placement and two lanes on a 122B | 2.4–3.6× slower by default | dropped |

[All days →](docs/JOURNEY.md)

## Biggest gains so far

- **79% vs 44%:** a profile from my own chats; its top 30% of experts serve 79% of picks in new conversations, someone else's profile 44% (Day 6).
- **31.9 → 49–54 tok/s decode:** the Mojo expert layer, back to back against my pinned llama.cpp stack (Day 7).
- **17.7 → 22.2 tok/s decode:** profile + exclusive tiers + CPU while GPU, against stock llama.cpp on held-out chat prompts (Day 6).

Each number compares against its own reference on the same day.

## Learn more

[Cover page](https://abhisheksreesaila.github.io/omxpert-dispatch/) · [Storyboard](https://abhisheksreesaila.github.io/omxpert-dispatch/journey.html#story) · [Day-by-day journey](docs/JOURNEY.md) · [Mojo explainer](https://abhisheksreesaila.github.io/omxpert-mojo/) · [Lab detail and how to reproduce](docs/LAB_DETAIL.md) · [Prior art](docs/PRIOR_ART.md) · [Open an issue](https://github.com/abhisheksreesaila/omxpert-dispatch/issues)

Run it with your own history: `omxpert/profile.sh`, then `STACK=pinned omxpert/run.sh`. Details in [LAB_DETAIL.md](docs/LAB_DETAIL.md).

## License

MIT
