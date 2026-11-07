"""Night 5: write docs/visuals/night5/after.json from the result files, then build it with isoarch."""
import json, subprocess
from pathlib import Path
W = Path.home() / "expert-dispatch"; R = W / "results/night5"; D = W / "docs/visuals/night5"
ld = lambda n: json.loads((R / n).read_text()) if (R / n).exists() else {}
mab, pfab, eg, st, lev = ld("mtp_ab.json"), ld("pf_ab.json"), ld("engram_in_ram.json"), ld("stack.json"), ld("leverage.json")
off = mab.get("arms", {}).get("off", {}); on = mab.get("arms", {}).get("off+mtp", {})
cpu_ms = off.get("cpu_expert_ms_per_tok", "?"); tot = round(1000 / off["median_tok_s"], 1) if off else "?"
acc = round(100 * on.get("draft_accept", 0)); mtp_on = on.get("median_tok_s", "?"); mtp_off = off.get("median_tok_s", "?")
pre = eg.get("prefill_tok_s_median", "?")
spec = {
  "title": "After: the same model with the night-5 exact stack",
  "subtitle": "Same weights, same outputs: the n-gram table goes through the OS page cache, Qwen's draft head guesses ahead, and the CPU is now the only real wait",
  "style": "fieldnotes",
  "floors": [
    {"id": "compute", "label": "compute", "note": "who does the math", "nodes": ["you", "router", "mtp", "gpu0", "gpu1", "cpu"]},
    {"id": "memory", "label": "memory", "note": "the buffer pool", "nodes": ["vram0", "vram1", "ram", "pool"]},
    {"id": "disk", "label": "disk", "note": "the table on disk", "nodes": ["experts", "ngram"]}
  ],
  "nodes": [
    {"id": "you", "label": "You", "kind": "client", "at": [0, 0], "tech": "chat / coding agent"},
    {"id": "router", "label": "Router", "kind": "ai", "at": [6, 0], "tech": "48 layers × pick 10 of 512", "desc": "Unchanged: the same experts are picked, so the output is the model's own."},
    {"id": "mtp", "label": "Draft head", "kind": "ai", "at": [6, -5], "tech": "Qwen's MTP layer, 4 GB", "desc": f"Guesses the next 2 tokens; the full model checks all 3 in one pass. {acc}% of guesses accepted.",
     "notes": "Database view: speculative execution. Checking 3 tokens costs about one token's expert reads, because consecutive tokens reuse most experts."},
    {"id": "gpu0", "label": "GPU 0", "kind": "service", "at": [12, -5], "tech": "RTX 3090"},
    {"id": "gpu1", "label": "GPU 1", "kind": "service", "at": [12, 0], "tech": "RTX 3090, NVLink"},
    {"id": "cpu", "label": "CPU", "kind": "worker", "at": [12, 5], "tech": "Ryzen 5900X, 12 cores", "desc": f"Still computes 24 layers' experts: {cpu_ms} ms of every {tot} ms token. This is now the bottleneck.",
     "notes": "Next lever: move this work to a GPU (your profile + an engine that overlaps CPU and GPU), or add an eGPU. See the leverage table."},
    {"id": "vram0", "label": "VRAM 0", "kind": "cache", "at": [12, -5], "tech": "24 GB"},
    {"id": "vram1", "label": "VRAM 1", "kind": "cache", "at": [12, 0], "tech": "24 GB"},
    {"id": "ram", "label": "RAM page cache", "kind": "cache", "at": [12, 5], "tech": "~35 GB free", "desc": f"{off.get('ram_hit', 0)*100:.2f}% of the expert pages the CPU needs are already here: decode no longer waits on the SSD."},
    {"id": "pool", "label": "Read pool", "kind": "worker", "at": [18, 5], "tech": "8 readers + learned guesser", "desc": "Fetches next layer's likely experts ahead (his idea, our learned table). Measured: no gain here, because nothing misses RAM any more. It pays on SSD-bound boxes (the Mac, bigger models)."},
    {"id": "experts", "label": "Expert file", "kind": "storage", "at": [12, 5], "tech": "77 GB, NVMe"},
    {"id": "ngram", "label": "N-gram table", "kind": "db", "at": [12, 0], "tech": "28.8 GB, page cache", "desc": "Mapped like any other weight and read in at load (-lzm off). Bit-exact. Low page faults (~10 per token) in 8 of 8 runs; stock lazy mode was erratic (high faults in 7 of 11 runs).",
     "notes": "Database view: let the OS buffer cache handle an index lookup instead of a hand-rolled random reader."}
  ],
  "edges": [
    {"from": "you", "to": "router", "label": "prompt"},
    {"from": "router", "to": "mtp", "label": "guess 2"},
    {"from": "router", "to": "gpu0", "label": "12 layers"},
    {"from": "router", "to": "gpu1", "label": "12 layers"},
    {"from": "router", "to": "cpu", "label": "24 layers"},
    {"from": "gpu0", "to": "vram0", "label": "hit"},
    {"from": "gpu1", "to": "vram1", "label": "hit"},
    {"from": "cpu", "to": "ram", "label": "99.99% hit"},
    {"from": "pool", "to": "ram", "label": "prefetch", "style": "async"},
    {"from": "ram", "to": "experts", "label": "rare", "style": "data"},
    {"from": "router", "to": "ngram", "label": "16 rows", "style": "data"}
  ],
  "flows": [
    {"name": "Generating (after)", "desc": f"Decode: {mtp_off} tok/s without drafting, {mtp_on} with, in the same server.", "steps": [
      {"from": "you", "to": "router", "note": "The last token goes in; the router picks the same 10 experts per layer as before."},
      {"from": "router", "to": "ngram", "note": "Its n-gram rows come from the OS page cache, read in at load."},
      {"from": "router", "to": "mtp", "note": "Qwen's draft head guesses the next two tokens."},
      {"from": "router", "to": "gpu0", "note": "The GPUs run their 24 layers for all 3 tokens at once."},
      {"from": "router", "to": "cpu", "note": "The CPU runs the other 24 layers for the 3 tokens; it reads about as much as for one."},
      {"from": "cpu", "to": "ram", "note": "Practically every expert is already in RAM: no SSD waits."},
      {"at": "cpu", "note": f"The CPU's {cpu_ms} ms is now most of the token: that is what to attack next."}
    ]},
    {"name": "Reading your prompt (after)", "desc": "Prefill: ~110-170 tok/s in every -lzm off run (stock: 17-175 depending on the run).", "steps": [
      {"from": "you", "to": "router", "note": "A prompt arrives in 512-token batches, as before."},
      {"from": "router", "to": "ngram", "note": "8,000+ n-gram rows per batch come from cached pages."},
      {"from": "router", "to": "cpu", "note": "The CPU-tier experts still stream to a GPU once per batch: the next limit for long prompts."}
    ]}
  ]
}
(D / "after.json").write_text(json.dumps(spec, indent=1))
print(subprocess.run(["python3", str(Path.home() / ".claude/skills/isoarch/scripts/build.py"), str(D / "after.json"), str(D / "after.html")], capture_output=True, text=True).stdout)
