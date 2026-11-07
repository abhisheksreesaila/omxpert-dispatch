#!/bin/bash
# omxpert profile: learn which of Qwen3.8-Flash-Next's 24,576 experts YOUR work uses, from your own text, and write the pin file
# that `STACK=pinned omxpert/run.sh` keeps in VRAM. Nothing leaves the machine; the pin file holds expert counts, not text.
#   omxpert/profile.sh                       use your Claude Code chat history (~/.claude/projects), 240k characters
#   omxpert/profile.sh my_notes.txt          use any text file (code, chats, docs: whatever you will ask the model about)
# Output: omxpert/pins/profile.bin  (int32 n_layer, int32 n_expert, float32 share[n_layer][n_expert])
# Measured on the owner's chat history: 2,000 tokens already give the full hit rate (30% of experts serve 79% of picks).
set -e
D="$(cd "$(dirname "$0")/.." && pwd)"
TRACE="${TRACE:-$D/llama.cpp/build/bin/llama-edx-trace}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-GGUF}"
GG=$(ls "$MODEL_DIR"/UD-Q4_K_XL/*00001-of-*.gguf | head -1)
mkdir -p "$D/omxpert/pins"; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT; chmod 700 "$TMP"
if [ -n "$1" ]; then
  head -c "${CHARS:-240000}" "$1" > "$TMP/corpus.txt"
else
  echo "reading your Claude Code chat history (prompts and visible replies only; stays in $TMP, deleted at exit)"
  python3 "$D/scripts/n6_chat_corpus.py" "$TMP/chat" > /dev/null
  head -c "${CHARS:-240000}" "$TMP/chat/chat_train.txt" > "$TMP/corpus.txt"
fi
echo "tracing the model's expert picks on $(wc -c < "$TMP/corpus.txt") characters (a few minutes)"
nice -n 10 "$TRACE" -m "$GG" -f "$TMP/corpus.txt" -o "$TMP/trace.bin" -c 2048 -b 512 -ub 512 2>&1 | grep -E "EDX_TRACE_DONE|failed" || true
python3 - "$TMP/trace.bin" "$D/omxpert/pins/profile.bin" <<'EOF'
import struct, sys, array
L, K, E = 48, 10, 512
a = array.array("i"); a.frombytes(open(sys.argv[1], "rb").read())
cnt = [[0] * E for _ in range(L)]
for i, e in enumerate(a):
    cnt[(i // K) % L][e] += 1
tot = float(sum(map(sum, cnt))) or 1.0
with open(sys.argv[2], "wb") as f:
    f.write(struct.pack("ii", L, E))
    for row in cnt:
        f.write(struct.pack(f"{E}f", *[c / tot for c in row]))
print(f"wrote {sys.argv[2]} from {len(a) // (L * K)} tokens")
EOF
