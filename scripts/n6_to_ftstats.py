"""Night 6: write our chat-history profile (expert pick counts per layer, from llama.cpp traces) as a FreeToken
--moe-stats-out file, so FreeToken's --moe-bank-stats orders its RAM bank by the owner's own usage.
usage: n6_to_ftstats.py <counts.npy> <out.json>"""
import json, sys
import numpy as np
c = np.load(sys.argv[1]).astype(int)
json.dump({"decode_freq": c.tolist(), "layer_range": [0, int(c.shape[0])], "source": "omxpert chat-history profile (llama.cpp traces)"},
          open(sys.argv[2], "w"))
print(c.shape, int(c.sum()))
