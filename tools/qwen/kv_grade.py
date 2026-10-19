#!/usr/bin/env python3
"""L291: grade QSA K/V formats from tests/qwen_kv_walk score outputs -- per depth, each arm against the reference
arm (bf16 K/V) whose logits the walk read.

  kv_grade.py ARM=out.tsv [ARM=out.tsv ...] [--block 64] [--boot 2000]

Every TSV must score the same windows (manifest) against the same reference dump; positions are checked to match.
Per depth bucket (the manifest's window depth) and arm: KL(ref || arm) mean, top-1 agreement with the reference,
dNLL = mean(NLL_arm - NLL_ref) of the true token with a 95% block-bootstrap CI (blocks of --block consecutive
positions inside one window; windows are independent documents), and the reference's own NLL.  An arm named
'floor' is read as the same-instrument perturbation floor (a quality-neutral change of the reference arm): an arm
whose KL / top-1 sit inside the floor's and whose dNLL CI covers 0 is not distinguishable from the reference.
"""
import argparse
import sys

import numpy as np


def load(path):
    a = np.genfromtxt(path, delimiter='\t', names=True)
    if a.size == 0:
        raise SystemExit(f'{path}: empty')
    return a


def boot_ci(x, blk, R, seed=0):
    """95% CI of mean(x) resampling whole blocks."""
    ids, inv = np.unique(blk, return_inverse=True)
    s = np.bincount(inv, weights=x)
    c = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(ids), (R, len(ids)))
    m = s[i].sum(1) / c[i].sum(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('arms', nargs='+', metavar='ARM=TSV')
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    arms = {}
    for spec in a.arms:
        name, path = spec.split('=', 1)
        arms[name] = load(path)
    names = list(arms)
    base = arms[names[0]]
    for n in names[1:]:
        t = arms[n]
        if len(t) != len(base) or (t['win'] != base['win']).any() or (t['pos'] != base['pos']).any():
            raise SystemExit(f'{n}: positions differ from {names[0]}')
        if np.abs(t['nll_ref'] - base['nll_ref']).max() > 1e-6:
            raise SystemExit(f'{n}: scored against a different reference dump')
    depths = sorted(set(int(d) for d in base['depth']))
    print(f'| depth | windows | positions | ref NLL | arm | KL(ref||arm) mean | median | top-1 vs ref | '
          f'dNLL vs ref [95% CI] |')
    print('|---:|---:|---:|---:|---|---:|---:|---:|---|')
    for d in depths + ['all']:
        sel = np.ones(len(base), bool) if d == 'all' else base['depth'] == d
        win = base['win'][sel].astype(np.int64)
        pos = base['pos'][sel].astype(np.int64)
        blk = win * 10_000_000 + (pos - pos.min()) // a.block
        for n in names:
            t = arms[n]
            kl = t['kl'][sel]
            dn = (t['nll_arm'] - t['nll_ref'])[sel]
            lo, hi = boot_ci(dn, blk, a.boot)
            print(f'| {d} | {len(np.unique(win))} | {sel.sum()} | {t["nll_ref"][sel].mean():.4f} | {n} | '
                  f'{kl.mean():.3e} | {np.median(kl):.3e} | {100 * t["top1"][sel].mean():.2f} % | '
                  f'{dn.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] |')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
