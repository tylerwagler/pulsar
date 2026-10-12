"""Replay draft schedules over unstopped traces with a cost model: ms/round = BASE + ROW * verified drafts
+ STEP * drafted-but-dropped drafts.  Rules: tau (per-draft, pulsar: stopper verified), tf (per-draft, stopper
dropped unless first), path (chain survival = prod conf; the draft that takes it under thr is verified, then stop),
pathdrop (that draft dropped).

  policy.py BASE ROW STEP TRACE...     (traces from unstopped runs: PULSAR_SPEC_TAU=off / TF conf 1e-9)

L288 2026-10-12 (td305, nsys over unstopped depth 2 / 6): BASE 28.75, ROW 4.45 = verify row 2.49 + MTP draft step
1.74 (of which the 82,196-row MXFP8 draft head 0.98); TensorFold 0.6.6: BASE 29.2, ROW 3.43."""
import sys
BASE, ROW, STEP = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
traces = sys.argv[4:]

def load(p):
    return [(int(l.split()[0]), int(l.split()[1]), [float(x) for x in l.split()[2:]]) for l in open(p) if l.strip()]

def chain(conf, depth, rule, thr):
    """(verified drafts L, dropped steps)"""
    s = 1.0
    for j in range(depth):
        c = conf[j]
        s *= c
        if rule == "tau" and c < thr: return j + 1, 0
        if rule == "tf" and c < thr: return (1, 0) if j == 0 else (j, 1)
        if rule == "path" and s < thr: return j + 1, 0
        if rule == "pathdrop" and s < thr: return (1, 0) if j == 0 else (j, 1)
    return depth, 0

def run(rows, depth, rule, thr):
    tok = ms = n = 0
    for nv, nk, conf in rows:
        if nv < depth: continue
        L, drop = chain(conf, depth, rule, thr)
        tok += min(nk, L) + 1
        ms += BASE + ROW * L + STEP * drop
        n += 1
    return tok / n, ms / n, ms / tok

data = {p: load(p) for p in traces}
D = min(max(nv for nv, _, _ in r) for r in data.values())
configs = [(4, "tau", 0.7), (6, "tau", 0.7), (6, "tf", 0.7)] + \
          [(d, r, t) for d in (5, 6, 8, 10) if d <= D for r in ("path", "pathdrop") for t in (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)]
print(f"cost: base {BASE} + {ROW}/verified draft + {STEP}/dropped draft step; depth available {D}")
print(f"{'depth rule thr':20s} " + " ".join(f"{p.split('/')[-1][:24]:>26s}" for p in traces) + "   mean ms/tok")
for d, r, t in configs:
    if d > D: continue
    cells, tot = [], 0
    for p in traces:
        tk, ms, mpt = run(data[p], d, r, t)
        cells.append(f"{tk:4.2f}t {ms:5.1f}ms {mpt:5.2f}")
        tot += mpt
    print(f"{d:2d} {r:8s} {t:4.2f}     " + " ".join(f"{c:>26s}" for c in cells) + f"   {tot / len(traces):6.2f}")
