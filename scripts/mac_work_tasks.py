# Work-lane practicality test: 8 coding tasks sent at once to llama-server; answers graded by running hidden tests.
import json, re, subprocess, sys, tempfile, threading, time, urllib.request
TASKS = [
 ("parse_duration", "Write a Python function parse_duration(s: str) -> int that converts strings like '1h30m15s', '45m', '2h', '10s' into total seconds. Raise ValueError on empty or invalid input.",
  "assert parse_duration('1h30m15s')==5415\nassert parse_duration('45m')==2700\nassert parse_duration('2h')==7200\nassert parse_duration('10s')==10\nfor bad in ['', 'abc', '5x']:\n    try:\n        parse_duration(bad); raise SystemExit('no error for '+bad)\n    except ValueError: pass"),
 ("roman_to_int", "Write a Python function roman_to_int(s: str) -> int that converts a valid Roman numeral (I..MMMCMXCIX) to an integer.",
  "assert roman_to_int('III')==3\nassert roman_to_int('IV')==4\nassert roman_to_int('MCMXCIV')==1994\nassert roman_to_int('MMMCMXCIX')==3999\nassert roman_to_int('XLII')==42"),
 ("merge_intervals", "Write a Python function merge_intervals(intervals: list[list[int]]) -> list[list[int]] that merges overlapping or touching closed intervals and returns them sorted by start.",
  "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]\nassert merge_intervals([[5,7],[1,2]])==[[1,2],[5,7]]"),
 ("top_k_words", "Write a Python function top_k_words(text: str, k: int) -> list[str] that returns the k most frequent words (case-insensitive, words are runs of letters a-z), most frequent first, ties broken alphabetically.",
  "assert top_k_words('the cat and the hat and the bat', 2)==['the','and']\nassert top_k_words('B a b A c', 3)==['a','b','c']\nassert top_k_words('', 3)==[]"),
 ("flatten_dict", "Write a Python function flatten_dict(d: dict, sep: str = '.') -> dict that flattens nested dicts into dotted keys, e.g. {'a': {'b': 1}, 'c': 2} -> {'a.b': 1, 'c': 2}. Empty nested dicts are dropped.",
  "assert flatten_dict({'a':{'b':1},'c':2})=={'a.b':1,'c':2}\nassert flatten_dict({'x':{'y':{'z':3}}}, sep='/')=={'x/y/z':3}\nassert flatten_dict({'a':{}})=={}\nassert flatten_dict({})=={}"),
 ("is_balanced", "Write a Python function is_balanced(s: str) -> bool that checks whether the brackets (), [], {} in s are balanced and properly nested; other characters are ignored.",
  "assert is_balanced('a(b[c]{d})e')\nassert not is_balanced('(]')\nassert not is_balanced('(()')\nassert is_balanced('')\nassert not is_balanced('}{')"),
 ("LRUCache", "Write a Python class LRUCache with __init__(self, capacity: int), get(self, key) -> value or -1 if missing, and put(self, key, value). When full, put evicts the least recently used key; get counts as a use.",
  "c=LRUCache(2)\nc.put(1,1); c.put(2,2)\nassert c.get(1)==1\nc.put(3,3)\nassert c.get(2)==-1\nc.put(4,4)\nassert c.get(1)==-1\nassert c.get(3)==3 and c.get(4)==4\nc.put(3,30); assert c.get(3)==30"),
 ("run_length", "Write a Python function run_length(s: str) -> list[tuple[str, int]] that run-length encodes a string, e.g. 'aaabcc' -> [('a', 3), ('b', 1), ('c', 2)].",
  "assert run_length('aaabcc')==[('a',3),('b',1),('c',2)]\nassert run_length('')==[]\nassert run_length('z')==[('z',1)]\nassert run_length('aabaa')==[('a',2),('b',1),('a',2)]"),
]
URL = "http://127.0.0.1:%s/v1/chat/completions" % (sys.argv[2] if len(sys.argv) > 2 else "8099")
def ask(i, out):
    name, prompt, _ = TASKS[i]
    body = {"messages": [{"role": "user", "content": prompt + " Reply with the code in one ```python block, no tests."}],
            "max_tokens": 1500, "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "low"}}
    t0 = time.time()
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=7200))
        out[i] = (r["choices"][0]["message"].get("content") or "", r["usage"]["completion_tokens"], time.time() - t0)
    except Exception as e:
        out[i] = ("", 0, time.time() - t0); print("ERR", name, e, file=sys.stderr)
def grade(text, tests):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    if not blocks: return False
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(blocks[-1] + "\n\n" + tests + "\n"); p = f.name
    try: return subprocess.run([sys.executable, "-I", p], capture_output=True, timeout=20).returncode == 0
    except subprocess.TimeoutExpired: return False
if __name__ == "__main__":
    label = sys.argv[1]; out = [None] * len(TASKS); t0 = time.time()
    th = [threading.Thread(target=ask, args=(i, out)) for i in range(len(TASKS))]
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.time() - t0; toks = sum(o[1] for o in out)
    passed = [grade(o[0], TASKS[i][2]) for i, o in enumerate(out)]
    json.dump({"label": label, "answers": [o[0] for o in out]}, open("results/mac/work_%s.json" % label, "w"), indent=1)
    print("%s pass=%d/8 [%s] tokens=%d wall_s=%.0f tok_s_total=%.2f" % (label, sum(passed), "".join("1" if p else "0" for p in passed), toks, wall, toks / wall))
