#!/usr/bin/env python3
"""Compare two pulsar-bench full-vocab frontier-logit dumps (the exact methodology
that produced the MXFP4-attention 33% / MXFP8-head 1.87% / NVFP4-head 6.76%
calibration points).

Metrics per frontier:
  - RMS(diff) as % of baseline logit RMS   (<3% promising, 3-7% borderline, >10% dead)
  - max single-logit |shift|
  - argmax hold
  - top-10 overlap (and whether the top-10 ORDER holds)

Usage: compare_frontier_logits.py BASE_DIR TEST_DIR [--label NAME]
Pairs up frontier_*.logits.json files present in both dirs.
"""
import argparse
import glob
import json
import os

import numpy as np


def load(fn):
    with open(fn) as f:
        d = json.load(f)
    a = np.array(d["logits"], dtype=np.float64)
    return a, d.get("argmax_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir")
    ap.add_argument("test_dir")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    base_files = {os.path.basename(p): p
                  for p in glob.glob(os.path.join(args.base_dir, "frontier_*.logits.json"))}
    test_files = {os.path.basename(p): p
                  for p in glob.glob(os.path.join(args.test_dir, "frontier_*.logits.json"))}
    common = sorted(set(base_files) & set(test_files))
    if not common:
        raise SystemExit(f"no common frontier dumps between {args.base_dir} and {args.test_dir}")

    print(f"== {args.label or args.test_dir} vs {args.base_dir} ==")
    print(f"{'frontier':>10} {'rms_drift%':>10} {'max_shift':>10} {'argmax':>8} "
          f"{'top10':>6} {'top10_order':>11}")
    for name in common:
        b, b_arg = load(base_files[name])
        t, t_arg = load(test_files[name])
        assert b.shape == t.shape, (name, b.shape, t.shape)
        d = t - b
        rms_b = float(np.sqrt(np.mean(b * b)))
        rms_d = float(np.sqrt(np.mean(d * d)))
        pct = 100.0 * rms_d / rms_b
        mx = float(np.abs(d).max())
        top_b = np.argsort(-b, kind="stable")[:10]
        top_t = np.argsort(-t, kind="stable")[:10]
        overlap = len(set(top_b.tolist()) & set(top_t.tolist()))
        order = "hold" if np.array_equal(top_b, top_t) else "reorder"
        argmax_hold = "hold" if int(np.argmax(b)) == int(np.argmax(t)) else \
            f"FLIP {int(np.argmax(b))}->{int(np.argmax(t))}"
        frontier = name.split("_")[1].split(".")[0].lstrip("0") or "0"
        print(f"{frontier:>10} {pct:>10.3f} {mx:>10.3f} {argmax_hold:>8} "
              f"{overlap:>5}/10 {order:>11}")
        # context for the calibration table
        print(f"           (baseline logit RMS {rms_b:.3f}; baseline argmax {b_arg}, "
              f"test argmax {t_arg})")


if __name__ == "__main__":
    main()
