#!/usr/bin/env python3
"""ref_kl.py -- grade engine logit rows against the reference anchors by KL (L255).

    python3 ref_kl.py ANCHORS_DIR DUMP_DIR [DUMP_DIR ...]

For each prompt P in ANCHORS_DIR (P.ref.json + P.ref.bin, qwen_anchors.py) and each DUMP_DIR written by
tests/qwen_ref_gate with PULSAR_REF_DUMP (P.d<depth>.eng.f32), prints per depth KL(ref || engine) in nats over the
full vocabulary, whether the argmax agrees, and the reference's probability of the engine's top-1; then the mean
KL per prompt and dump.  Several DUMP_DIRs (e.g. the EXL3 container and a Tessera-layer container, same build)
print side by side.
"""
import json
import os
import sys

import numpy as np

HDR = 88   # qwen_ref_gate.cpp QWEN_REF_HDR


def softmax_log(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def main():
    anchors, dumps = sys.argv[1], sys.argv[2:]
    if not dumps:
        raise SystemExit(__doc__)
    prompts = sorted(f[:-len(".ref.json")] for f in os.listdir(anchors) if f.endswith(".ref.json"))
    for p in prompts:
        meta = json.load(open(os.path.join(anchors, f"{p}.ref.json")))
        W, rows = int(meta["width"]), meta["rows"]
        ref = np.fromfile(os.path.join(anchors, f"{p}.ref.bin"), dtype=np.float32, offset=HDR).reshape(-1, W)
        print(f"== {p}  (width {W}, {len(rows)} depths)   " + "   ".join(f"[{os.path.basename(d.rstrip('/'))}]"
                                                                         for d in dumps))
        sums = [[] for _ in dumps]
        for r, row in enumerate(rows):
            d = int(row["depth"])
            lr = softmax_log(ref[r])
            pr = np.exp(lr)
            cells = []
            for i, dd in enumerate(dumps):
                path = os.path.join(dd, f"{p}.d{d}.eng.f32")
                if not os.path.exists(path):
                    cells.append("   (no row)            ")
                    continue
                le = softmax_log(np.fromfile(path, dtype=np.float32))
                kl = float((pr * (lr - le)).sum())
                top = int(le.argmax())
                sums[i].append(kl)
                cells.append(f"KL {kl:8.5f} {'=' if top == int(lr.argmax()) else 'X'} p_ref(top) {pr[top]:.3f}")
            print(f"  d={d:<6} ref H {row.get('entropy_nats', float('nan')):6.3f}  " + "   ".join(cells))
        print("  mean KL  " + "   ".join(f"{np.mean(s):.5f} over {len(s)}" if s else "-" for s in sums))


if __name__ == "__main__":
    main()
