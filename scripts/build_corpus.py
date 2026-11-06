"""Phase 1: build the owner's corpus, tagged by kind, split by time (oldest 80% train).

Sources: git-tracked files in ~/Projects/* (minus third-party clones), Claude Code
transcripts in ~/.claude/projects. Nothing leaves the machine.
"""
import json, os, re, subprocess, random
from pathlib import Path

HOME = Path.home()
OUT = HOME / "expert-dispatch/corpus"
SKIP_REPOS = {"t3code", "archive", "localv2", "dev-tools"}
SKIP_PARTS = {"node_modules", "vendor", ".git", "dist", "build", ".pixi", ".venv",
              "site-packages", "third_party", "__pycache__", ".next", "coverage"}
CODE_EXT = {".py", ".ts", ".tsx", ".js", ".mjs", ".mojo", ".rs", ".go", ".swift",
            ".sh", ".css", ".html", ".sql", ".toml", ".c", ".cpp", ".h", ".kt"}
DOC_EXT = {".md", ".txt", ".rst"}
SKIP_NAMES = re.compile(r"(lock|\.min\.|\.map$|\.env)", re.I)
CAP_CHARS = 4_000_000  # ~1M tokens per tag
CHUNK = 6000

SECRET = re.compile(
    r"(sk-[A-Za-z0-9_\-]{20,}|ghp_[A-Za-z0-9]{30,}|github_pat_\w{30,}|AKIA[0-9A-Z]{16}|"
    r"xox[abpr]-[\w-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY|eyJ[\w-]{20,}\.[\w-]{20,}\.|"
    r"(?i:(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+]{16,}))")
MATH = re.compile(r"[=+\-*/^]\s*\d|\\frac|\\sum|\d+\s*[x×*]\s*\d+|\$[^$]+\$")

dropped = 0
items = []  # (tag, mtime, text, source)


def add(tag, mtime, text, src):
    global dropped
    text = text.strip()
    if len(text) < 40:
        return
    if SECRET.search(text):
        dropped += 1
        return
    for i in range(0, len(text), CHUNK):
        piece = text[i:i + CHUNK]
        t = tag
        if tag in ("docs", "design_chat") and len(MATH.findall(piece)) > len(piece) / 120:
            t = "math"
        items.append((t, mtime, piece, src))


# ---- repos
for repo in sorted((HOME / "Projects").iterdir()):
    if not repo.is_dir() or repo.name in SKIP_REPOS or repo.name.endswith(".worktrees"):
        continue
    try:
        files = subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True,
                               text=True, timeout=60).stdout.split()
    except Exception:
        files = []
    for rel in files:
        p = repo / rel
        if set(Path(rel).parts) & SKIP_PARTS or SKIP_NAMES.search(p.name):
            continue
        ext = p.suffix.lower()
        if ext not in CODE_EXT | DOC_EXT:
            continue
        try:
            st = p.stat()
            if st.st_size > 1_000_000 or st.st_size == 0:
                continue
            text = p.read_text(errors="strict")
        except Exception:
            continue
        add("coding" if ext in CODE_EXT else "docs", st.st_mtime, text, f"{repo.name}/{rel}")

# ---- transcripts
from datetime import datetime
for f in (HOME / ".claude/projects").glob("*/*.jsonl"):
    try:
        lines = f.read_text(errors="ignore").splitlines()
    except Exception:
        continue
    for ln in lines:
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if d.get("type") not in ("user", "assistant"):
            continue
        ts = d.get("timestamp")
        try:
            mt = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except Exception:
            mt = f.stat().st_mtime
        c = (d.get("message") or {}).get("content")
        if isinstance(c, str):
            if "<system-reminder>" not in c[:200]:
                add("design_chat", mt, c, f.name)
            continue
        for b in c or []:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                add("design_chat", mt, b.get("text", ""), f.name)
            elif bt == "tool_use":
                add("tool_calls", mt, json.dumps({"tool": b.get("name"), "input": b.get("input")})[:CHUNK], f.name)
            elif bt == "tool_result":
                cc = b.get("content")
                if isinstance(cc, list):
                    cc = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
                add("tool_calls", mt, str(cc)[:CHUNK], f.name)

# ---- split + cap + write
OUT.mkdir(parents=True, exist_ok=True)
random.seed(0)
manifest = ["# Corpus manifest", "", f"Secret-pattern chunks dropped: {dropped}", "",
            "| tag | split | chunks | ~tokens | date range | sources |", "|---|---|---|---|---|---|"]
for tag in sorted({t for t, *_ in items}):
    rows = sorted([x for x in items if x[0] == tag], key=lambda x: x[1])
    total = sum(len(x[2]) for x in rows)
    if total > CAP_CHARS * 1.25:  # downsample uniformly in time, keep order
        keep = CAP_CHARS * 1.25 / total
        rows = [x for x in rows if random.random() < keep]
    cut = int(len(rows) * 0.8)
    for split, part in (("train", rows[:cut]), ("heldout", rows[cut:])):
        with open(OUT / f"{tag}_{split}.txt", "w") as fh:
            for x in part:
                fh.write(x[2] + "\n\n")
        ch = sum(len(x[2]) for x in part)
        if part:
            dr = f"{datetime.fromtimestamp(part[0][1]):%Y-%m-%d} → {datetime.fromtimestamp(part[-1][1]):%Y-%m-%d}"
        else:
            dr = "-"
        flag = " ⚠ tiny" if ch / 4 < 10_000 else ""
        manifest.append(f"| {tag} | {split} | {len(part)} | {ch // 4:,}{flag} | {dr} | {len({x[3] for x in part})} |")
(OUT / "MANIFEST.md").write_text("\n".join(manifest) + "\n")
print("\n".join(manifest))
