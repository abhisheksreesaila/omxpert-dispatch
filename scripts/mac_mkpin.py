"""Build an EDX_PIN profile from an edx-trace file (int32 [T][L][K]), pure Python (no numpy on the Mac).
usage: mac_mkpin.py <trace.bin> <out.bin> [max_tokens]"""
import array, struct, sys
L, K, E = 36, 4, 128
t = array.array("i"); t.frombytes(open(sys.argv[1], "rb").read())
T = len(t) // (L * K)
if len(sys.argv) > 3: T = min(T, int(sys.argv[3]))
cnt = [[0] * E for _ in range(L)]
for i in range(T):
    b = i * L * K
    for l in range(L):
        row = cnt[l]
        for k in range(K): row[t[b + l * K + k]] += 1
with open(sys.argv[2], "wb") as f:
    f.write(struct.pack("ii", L, E))
    for l in range(L): f.write(struct.pack(f"{E}f", *[c / T for c in cnt[l]]))
cov = sum(sum(sorted(cnt[l], reverse=True)[:16]) for l in range(L)) / (T * L * K)
print(sys.argv[2], T, "tokens; top-16 per layer cover %.1f%% of routings" % (100 * cov))
