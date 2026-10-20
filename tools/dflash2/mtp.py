"""MTP side of the gate from qwen_generate QWEN_SPEC_TRACE files: P(chain kept >= d) (curves.py's definition) and
tokens/round as served, plus the fixed-depth-k replay (tokens = min(kept, k) + 1) over unstopped rounds."""
import sys


def load(path):
    rows = []
    for line in open(path):
        p = line.split()
        if p:
            rows.append((int(p[0]), int(p[1])))
    return rows


for path in sys.argv[1:]:
    rows = load(path)
    R = len(rows)
    D = max(nv for nv, _ in rows)
    keep = [sum(1 for nv, nk in rows if nk >= d and nv >= d) / R for d in range(1, D + 1)]
    full = [r for r in rows if r[0] == D]
    fixed = {k: sum(min(nk, k) + 1 for _, nk in full) / len(full) for k in range(1, D + 1)}
    print(f"{path}: {R} rounds, served {sum(nk + 1 for _, nk in rows) / R:.3f} tok/round, full-depth rounds {len(full)}")
    print("  P(kept>=d) " + " ".join(f"{x:.3f}" for x in keep))
    print("  fixed-k replay tok/round " + " ".join(f"k{k} {v:.2f}" for k, v in fixed.items()))
