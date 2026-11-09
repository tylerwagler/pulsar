"""L288 acceptance curves from round traces (tf_trace.py / qwen_generate QWEN_SPEC_TRACE): one line a round,
`n_verified n_kept conf_1 .. conf_m` (m >= n_verified: TF records the confidence of a draft its stop rule dropped).

  curves.py TRACE [TRACE ...]            per-depth conditional acceptance, calibration, policy replay
"""
import sys
from collections import defaultdict

BINS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]


def load(path):
    rows = []
    for line in open(path):
        p = line.split()
        if not p:
            continue
        nv, nk = int(p[0]), int(p[1])
        conf = [float(x) for x in p[2:]]
        rows.append((nv, nk, conf))
    return rows


def replay(rows, depth, tau, verify_stopper):
    """Replay a draft schedule over UNSTOPPED rounds (each round's chain known to its full depth): rows verified and
    tokens kept per round if the chain had been cut by (depth, tau).  verify_stopper: pulsar's rule (the draft under
    tau is verified, then the chain ends); else TF's (the first draft always, a later one under tau not drafted)."""
    tok = rws = n = 0
    for nv, nk, conf in rows:
        if nv < depth:          # a round cut by max_tokens: not comparable
            continue
        L = depth
        for j in range(depth):
            if conf[j] < tau:
                L = j + 1 if (verify_stopper or j == 0) else j
                break
        tok += min(nk, L) + 1
        rws += L + 1
        n += 1
    return tok / n, rws / n, n


def report(path):
    rows = load(path)
    R = len(rows)
    toks = sum(nk + 1 for _, nk, _ in rows)
    print(f"== {path}: {R} rounds, {toks / R:.3f} tokens/round, {sum(nv for nv, _, _ in rows) / R:.2f} drafts verified/round")
    # conditional acceptance a_d = P(draft d kept | drafts 1..d-1 kept, draft d verified)
    reach, acc = defaultdict(int), defaultdict(int)
    for nv, nk, _ in rows:
        for d in range(1, nv + 1):
            if nk >= d - 1:
                reach[d] += 1
                acc[d] += nk >= d
    print("  depth d        : " + " ".join(f"{d:6d}" for d in sorted(reach)))
    print("  P(acc d|reach) : " + " ".join(f"{acc[d] / reach[d]:6.3f}" for d in sorted(reach)))
    print("  P(kept >= d)   : " + " ".join(f"{sum(1 for nv, nk, _ in rows if nk >= d and nv >= d) / R:6.3f}" for d in sorted(reach)))
    print("  reached        : " + " ".join(f"{reach[d]:6d}" for d in sorted(reach)))
    # stop reasons (as run): chain ended by a rejection, by the depth, or by the confidence rule
    rej = sum(1 for nv, nk, _ in rows if nk < nv)
    full = sum(1 for nv, nk, _ in rows if nk == nv)
    print(f"  rounds ending in a rejected draft {rej / R:.3f}, all verified drafts kept {full / R:.3f}")
    # calibration: P(draft kept | reached, conf bin) pooled over depth, and by depth 1 / 2+
    for label, sel in (("all", lambda d: True), ("d=1", lambda d: d == 1), ("d>=2", lambda d: d >= 2)):
        hit, cnt = defaultdict(int), defaultdict(int)
        for nv, nk, conf in rows:
            for d in range(1, nv + 1):
                if nk >= d - 1 and sel(d):
                    c = conf[d - 1]
                    b = max(i for i in range(len(BINS) - 1) if c >= BINS[i])
                    cnt[b] += 1
                    hit[b] += nk >= d
        print(f"  calib {label:5s}: " + " ".join(
            f"[{BINS[b]:.2f},{BINS[b + 1]:.2f}) {hit[b] / cnt[b]:.2f} n={cnt[b]}" for b in sorted(cnt)))
    return rows


def main():
    runs = [(p, report(p)) for p in sys.argv[1:]]
    print("\n== policy replay over the unstopped rounds (tokens/round, rows/round): depth x tau, pulsar rule / TF rule")
    for p, rows in runs:
        if not rows or max(nv for nv, _, _ in rows) < 4:
            continue
        D = max(nv for nv, _, _ in rows)
        print(f"-- {p}")
        for depth in range(2, D + 1):
            line = []
            for tau in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
                t1, r1, _ = replay(rows, depth, tau, True)
                t2, r2, _ = replay(rows, depth, tau, False)
                line.append(f"tau {tau:.1f}: {t1:.2f}/{r1:.2f} | {t2:.2f}/{r2:.2f}")
            print(f"  depth {depth}: " + "   ".join(line))


if __name__ == "__main__":
    main()
