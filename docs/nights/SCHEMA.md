# Nightly storyboard frames

Each morning the nightly loop of [omxpert-evolve](../journey.html#nightly) writes one file here, `YYYY-MM-DD.json`, and adds its name to `index.json`. `journey.html` reads them and draws one frame per night. Numbers only: no chat text ever lands here (`publish_night.py` refuses free text beyond two short fields).

```json
{
  "date": "2026-11-19",
  "night": 1,
  "champion_before": "mojo-stream-full",
  "champion_after": "skip-0.02",
  "decode_tps_before": 49.3,
  "decode_tps_after": 52.0,
  "tried": [
    {"name": "skip-0.02", "verdict": "promoted", "kl": 0.031, "top1": 0.93, "pass1": 0.82,
     "decode_tps": 52.0, "gpu_hit": 0.92, "note": "drops cold picks with gate < 0.02"},
    {"name": "reap-25", "verdict": "rejected", "kl": 0.081, "top1": 0.86, "pass1": 0.75,
     "decode_tps": 55.1, "gpu_hit": 0.96, "note": "over the KL gate"}
  ],
  "sources": [{"title": "REAP", "url": "https://arxiv.org/abs/2510.13999"}],
  "picture": "One sentence: what the night looked like."
}
```

The values above illustrate the shape only; they are not a measurement.

| field | type | rule |
|---|---|---|
| `date` | string | `YYYY-MM-DD`, also the file name |
| `night` | integer | 1, 2, 3, ... |
| `champion_before`, `champion_after`, `tried[].name` | string | ≤ 60 chars, letters, digits, `space . _ + - /` |
| `decode_tps_before`, `decode_tps_after`, `tried[].decode_tps` | number | tok/s decode, the gate's paired measurement |
| `tried[].verdict` | string | `promoted` or `rejected` |
| `tried[].kl` | number | KL to the untouched model on the FROZEN set |
| `tried[].top1`, `tried[].pass1`, `tried[].gpu_hit` | number | fractions 0 to 1 |
| `tried[].note`, `picture` | string | ≤ 200 chars, written by the loop, never copied from a chat |
| `sources[]` | object | `title` ≤ 120 chars, `url` starting `https://` |
