#!/usr/bin/env python3
"""L288 qwen-exl3-head: the SHORT grade of head formats -- every held-out row its own context (reset per row), its first
~2048 tokens, every position scored, every head on the same head inputs, against a same-instrument perturbation floor.

head_grade.py grades heads over one 25k-token concatenated walk; this grades them at the depths L251's instrument used
(rows reset at <= 2048 tokens) and puts a trunk perturbation beside them, so a head's error reads against the response
of the same instrument to a quality-neutral change.

Per row i the engine walks row i alone (--kl-tokens DIR/row{i}.json, prefix 32) twice under the QHEAD_WALK capture
(~/qhead/cap.patch): the base build (DIR/base{i}.rows, plus a sparse logits dump DIR/base{i}.klrf for the alignment
check) and the floor build (DIR/{floor}{i}.rows: the base with the GR W_down sum x (1 + 1e-6), ~/qcal/p2.patch).  In a
teacher-forced walk a head change never reaches the trunk, so each head's logits are its weight times the base rows;
the floor is the served MXFP8 head on the perturbed trunk's rows.

  head_short.py --dir DIR --docs N --source DIR --quant k4=Q4.pt k5=Q5.pt [--floor p2] [--block 64]

Reported per candidate (mxfp8, the quants, the floor): KL(BF16 || c) and KL(MXFP8 || c) mean / median, top-1 agreement
with each, NLL of the true token, dNLL vs the MXFP8 head with a block-bootstrap 95% CI (blocks of --block positions
inside one row), and the per-row NLL table.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import head_exl3 as HX   # noqa: E402
from head_grade import bf16_to_f32, klrf, mxfp8   # noqa: E402

V, E = HX.N_VOCAB, HX.N_EMBD


def rows_of(path, n):
    have = os.path.getsize(path) // (2 * E)
    if have < n:
        raise SystemExit(f'{path}: {have} head calls, the row needs {n}')
    return bf16_to_f32(np.memmap(path, dtype=np.uint16, mode='r', shape=(have, E))[:n])


def ratio_ci(s, c, R=2000, seed=0):
    """95% bootstrap CI of sum(s) / sum(c) over resampled blocks (s: per-block sums, c: per-block counts)."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(s), (R, len(s)))
    return np.percentile(s[i].sum(1) / c[i].sum(1), [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--docs', type=int, required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--quant', nargs='*', default=[])
    ap.add_argument('--floor', default='p2')
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--out', help='save per-position arrays (npz)')
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())

    W = {'bf16': bf16_to_f32(HX.read_bf16(a.source, 'lm_head.weight', (V, E)))}
    W['mxfp8'] = mxfp8(W['bf16'])
    for spec in a.quant:
        name, path = spec.split('=', 1)
        q = torch.load(path, weights_only=False)
        if q['draft'] or q['n'] != V:
            raise SystemExit(f'{path}: not an lm_head quant')
        W[name] = q['rec'].float()
        print(f'{name}: K={q["k"]} proxy err {q["proxy_err"]:.4e} rel Frobenius {q["rel_frob"]:.4e}', flush=True)
    heads = list(W)
    cand = [k for k in heads if k != 'bf16'] + [a.floor]

    def logp(z):
        z = z.double()
        return z - torch.logsumexp(z, 1, keepdim=True)

    acc = {k: {m: [] for m in ('nll', 'klbf', 'klmx', 't1bf', 't1mx')} for k in ['bf16'] + cand}
    doc_of, ind_of = [], []
    for d in range(a.docs):
        toks = np.array(json.load(open(f'{a.dir}/row{d}.json')))
        ref, stride, pre = klrf(f'{a.dir}/base{d}.klrf')
        n = len(toks) - pre                      # head call j predicts toks[pre + j]
        X = rows_of(f'{a.dir}/base{d}.rows', n)
        XF = rows_of(f'{a.dir}/{a.floor}{d}.rows', n)
        tt = torch.from_numpy(toks[pre:pre + n]).long()
        align = []
        for s in range(0, n, a.chunk):
            r = slice(s, min(n, s + a.chunk))
            L = {k: X[r] @ W[k].T for k in heads}
            L[a.floor] = XF[r] @ W['mxfp8'].T
            P = {k: logp(v) for k, v in L.items()}
            pb, pm = P['bf16'].exp(), P['mxfp8'].exp()
            ab, am = L['bf16'].argmax(1), L['mxfp8'].argmax(1)
            for k, lp in P.items():
                acc[k]['nll'].append(-lp.gather(1, tt[r][:, None])[:, 0].numpy())
                acc[k]['klbf'].append((pb * (P['bf16'] - lp)).sum(1).numpy())
                acc[k]['klmx'].append((pm * (P['mxfp8'] - lp)).sum(1).numpy())
                acc[k]['t1bf'].append((L[k].argmax(1) == ab).numpy())
                acc[k]['t1mx'].append((L[k].argmax(1) == am).numpy())
            # the engine's own MXFP8 logits at its dumped positions: the rows are aligned with the walk or the grade is void
            for j in range(s, r.stop):
                if j % stride == 0 and j // stride < ref.shape[0]:
                    e_ = torch.from_numpy(np.asarray(ref[j // stride]))
                    align.append(float((L['mxfp8'][j - s] - e_).abs().max() / e_.abs().max()))
        mx = max(align) if align else float('nan')
        print(f'row {d}: {n} positions, engine MXFP8 vs offline at {len(align)} dumped positions: max rel {mx:.2e} '
              f'({"ALIGNED" if mx < 1e-4 else "NOT ALIGNED -- void"})', flush=True)
        if not mx < 1e-4:
            raise SystemExit('alignment failed')
        doc_of.append(np.full(n, d))
        ind_of.append(np.arange(n) + pre)
    doc, ind = np.concatenate(doc_of), np.concatenate(ind_of)
    M = {k: {m: np.concatenate(v) for m, v in acc[k].items()} for k in acc}
    blk = doc * 100000 + (ind // a.block)
    _, bid = np.unique(blk, return_inverse=True)
    cnt = np.bincount(bid).astype(float)

    print(f'\n== SHORT grade: {a.docs} rows, each reset, first <= {ind.max() + 1} tokens, {len(doc)} positions scored '
          f'(every position); floor = {a.floor} (MXFP8 head on the perturbed trunk)')
    print(f'  {"cand":8s} {"KL(bf16||c) mean / median":>27s} {"top1 bf16":>9s} {"KL(mx||c) mean / median":>25s} '
          f'{"top1 mx":>8s} {"NLL":>8s}  dNLL vs mxfp8 [95% block CI]   dNLL vs bf16 [95% CI]')
    for k in ['bf16'] + cand:
        m = M[k]
        d_mx = m['nll'] - M['mxfp8']['nll']
        d_bf = m['nll'] - M['bf16']['nll']
        lo, hi = ratio_ci(np.bincount(bid, d_mx), cnt)
        lo2, hi2 = ratio_ci(np.bincount(bid, d_bf), cnt)
        print(f'  {k:8s} {m["klbf"].mean():.4e} / {np.median(m["klbf"]):.2e}   {100 * m["t1bf"].mean():6.2f}%  '
              f'{m["klmx"].mean():.4e} / {np.median(m["klmx"]):.2e}  {100 * m["t1mx"].mean():6.2f}%  '
              f'{m["nll"].mean():.5f}  {d_mx.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]  {d_bf.mean():+.5f} [{lo2:+.5f}, {hi2:+.5f}]')
    print('\n== per row: NLL (nats / token)   and KL(bf16 || c) mean')
    print('  row  n     ' + ' '.join(f'{k:>9s}' for k in ['bf16'] + cand) + '   ' + ' '.join(f'{"KL " + k:>12s}' for k in cand))
    for d in range(a.docs):
        s = doc == d
        print(f'  {d:3d} {s.sum():5d} ' + ' '.join(f'{M[k]["nll"][s].mean():9.5f}' for k in ['bf16'] + cand) + '   ' +
              ' '.join(f'{M[k]["klbf"][s].mean():12.3e}' for k in cand))
    if a.out:
        np.savez(a.out, doc=doc, ind=ind, **{f'{m}_{k}': M[k][m] for k in M for m in M[k]})


if __name__ == '__main__':
    main()
