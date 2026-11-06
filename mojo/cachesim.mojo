# Expert-cache simulator. Replays a routing trace layer by layer and reports
# hit rate per policy and cache size S (experts resident per layer).
#
# usage: cachesim <heldout.bin> <T> <L> <K> <E> <profile.bin> <S1,S2,...>
#   heldout.bin : int32 [T, L, K] routed expert ids, token order
#   profile.bin : int32 [L, E]    routing counts from the train split
# prints CSV: policy,S,layer,hits,accesses

from std.sys import argv


def read_i32(path: String) raises -> List[Int32]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    var n = len(b) // 4
    var p = b.unsafe_ptr().unsafe_bitcast[Int32]()
    var out = List[Int32](capacity=n)
    for i in range(n):
        out.append(p[i])
    return out^


def top_s(counts: List[Float64], s: Int) -> List[Bool]:
    # mark the s highest-count experts (ties broken by lower id)
    var e = len(counts)
    var res = List[Bool](length=e, fill=False)
    for _ in range(s):
        var best = -1
        var bv = -1.0e300
        for j in range(e):
            if not res[j] and counts[j] > bv:
                bv = counts[j]
                best = j
        if best >= 0:
            res[best] = True
    return res^


def main() raises:
    var a = argv()
    var trace = read_i32(String(a[1]))
    var T = Int(String(a[2]))
    var L = Int(String(a[3]))
    var K = Int(String(a[4]))
    var E = Int(String(a[5]))
    var prof = read_i32(String(a[6]))
    var sizes = List[Int]()
    for part in String(a[7]).split(","):
        sizes.append(Int(String(part)))

    print("policy,S,layer,hits,accesses")
    var n = T * K
    for l in range(L):
        # flatten this layer's access sequence
        var seq = List[Int](capacity=n)
        for t in range(T):
            for k in range(K):
                seq.append(Int(trace[(t * L + l) * K + k]))
        var pc = List[Float64](length=E, fill=0.0)
        var ptot = 0.0
        for j in range(E):
            pc[j] = Float64(prof[l * E + j])
            ptot += pc[j]
        # next-use index for Belady
        var nxt = List[Int](length=n, fill=n + 1)
        var last = List[Int](length=E, fill=n + 1)
        for i in range(n - 1, -1, -1):
            nxt[i] = last[seq[i]]
            last[seq[i]] = i

        for S in sizes:
            if S > E:
                continue
            # --- static profile
            var st = top_s(pc, S)
            var h = 0
            for i in range(n):
                if st[seq[i]]:
                    h += 1
            print("static_profile,", S, ",", l, ",", h, ",", n, sep="")

            # --- LRU (and profile + LRU: S/2 pinned, S/2 LRU)
            for variant in range(2):
                var pinned = List[Bool](length=E, fill=False)
                var cap = S
                if variant == 1:
                    pinned = top_s(pc, S // 2)
                    cap = S - S // 2
                var res = List[Bool](length=E, fill=False)
                var stamp = List[Int](length=E, fill=-1)
                var used = 0
                h = 0
                for i in range(n):
                    var x = seq[i]
                    if pinned[x]:
                        h += 1
                        continue
                    if res[x]:
                        h += 1
                    else:
                        if used == cap:
                            var victim = -1
                            var vs = n + 10
                            for j in range(E):
                                if res[j] and stamp[j] < vs:
                                    vs = stamp[j]
                                    victim = j
                            res[victim] = False
                            used -= 1
                        res[x] = True
                        used += 1
                    stamp[x] = i
                if variant == 0:
                    print("lru,", S, ",", l, ",", h, ",", n, sep="")
                else:
                    print("profile_plus_lru,", S, ",", l, ",", h, ",", n, sep="")

            # --- Belady MIN (offline optimum)
            var res = List[Bool](length=E, fill=False)
            var nu = List[Int](length=E, fill=n + 1)
            var used = 0
            h = 0
            for i in range(n):
                var x = seq[i]
                if res[x]:
                    h += 1
                else:
                    if used == S:
                        var victim = -1
                        var far = -1
                        for j in range(E):
                            if res[j] and nu[j] > far:
                                far = nu[j]
                                victim = j
                        res[victim] = False
                        used -= 1
                    res[x] = True
                    used += 1
                nu[x] = nxt[i]
            print("belady_min,", S, ",", l, ",", h, ",", n, sep="")

            # --- learned frequency predictor: score = prior (profile) + EMA of
            # recent routing; cache = top-S by score, refreshed every 32 tokens.
            # This is the "predict the experts" policy without lookahead.
            var score = List[Float64](length=E, fill=0.0)
            for j in range(E):
                score[j] = pc[j] / (ptot + 1.0) * 64.0
            var cache = top_s(score, S)
            h = 0
            var decay = 0.98
            for t in range(T):
                for k in range(K):
                    var x = seq[t * K + k]
                    if cache[x]:
                        h += 1
                for j in range(E):
                    score[j] *= decay
                for k in range(K):
                    score[seq[t * K + k]] += 1.0
                if t % 32 == 31:
                    # prior keeps a floor so the cache does not forget the profile
                    var mix = List[Float64](length=E, fill=0.0)
                    for j in range(E):
                        mix[j] = score[j] + pc[j] / (ptot + 1.0) * 16.0
                    cache = top_s(mix, S)
            print("ema_predictor,", S, ",", l, ",", h, ",", n, sep="")
