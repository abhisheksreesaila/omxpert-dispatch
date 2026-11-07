"""Night 6: build the owner's chat-history corpus from their Claude Code transcripts (local only; never committed or published).
Keeps the human prompts and the assistant's visible text replies (no tool output, no file dumps, no thinking), drops
anything that looks like a secret. Splits by session: sessions are shuffled with a fixed seed, ~80% train / 20% held out,
so the hit curve measures new conversations, not new lines of the same one.
usage: n6_chat_corpus.py <out_dir>   (writes chat_train.txt, chat_heldout.txt, chat_stats.json)"""
import json, random, re, sys
from pathlib import Path

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|"
                    r"(?i:api[_-]?key|token|password|secret)\s*[:=]\s*\S{8,}|xox[bp]-[A-Za-z0-9-]{10,})")
SKIP = ("<system-reminder>", "<command-", "<local-command", "Caveat:", "[Request interrupted", "<task-notification>")


def texts(msg):
    c = msg.get("content")
    if isinstance(c, str):
        yield c
    elif isinstance(c, list):
        for b in c:
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
                yield b["text"]


sessions = {}
for f in sorted(Path.home().glob(".claude/projects/*/*.jsonl")):
    parts = []
    try:
        for line in f.open(errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") not in ("user", "assistant") or d.get("isSidechain") or d.get("isMeta"):
                continue
            for t in texts(d.get("message") or {}):
                t = t.strip()
                if len(t) < 20 or any(t.startswith(s) for s in SKIP) or SECRET.search(t):
                    continue
                parts.append(("User: " if d["type"] == "user" else "Assistant: ") + t[:6000])
    except Exception:
        continue
    if parts:
        sessions[f.stem] = "\n\n".join(parts)

keys = sorted(sessions); random.Random(6).shuffle(keys)
cut = int(len(keys) * 0.8)
tr = "\n\n".join(sessions[k] for k in keys[:cut]); ho = "\n\n".join(sessions[k] for k in keys[cut:])
(out / "chat_train.txt").write_text(tr); (out / "chat_heldout.txt").write_text(ho)
st = {"sessions": len(keys), "train_sessions": cut, "train_chars": len(tr), "heldout_chars": len(ho)}
(out / "chat_stats.json").write_text(json.dumps(st)); print(st)
