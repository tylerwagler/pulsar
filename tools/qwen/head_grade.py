#!/usr/bin/env python3
"""L288 qwen-exl3-head: grade head formats on a teacher-forced walk, every head on the SAME head inputs.

In a teacher-forced walk the tokens are fixed, so a head-only change never reaches the trunk: the walk's head inputs
(the mixer's bf16 rows, one per position, captured by the engine's QHEAD_WALK instrument during the base walk) are
the same for every head, and each format's logits are those rows times its weight.  So BF16 -- which the engine no
longer runs -- is graded on the walk exactly, and the per-position KL is the head's own error (no state divergence:
the walk's 0.21-0.23 "reassociation floor" of a TRUNK change does not apply to a head change, and the head's floor is
the f32 summation order, ~1e-12).

  head_grade.py --rows WALK.bf16 --klrf BASE.klrf --source DIR --quant k3=Q3.pt k4=Q4.pt k5=Q5.pt
                [--engine k5=K5.klrf ...] [--block 64]

  --klrf     the engine's walk dump of the same run (DS4KLRF1, every `stride`-th position; its logits are the served
             MXFP8 head's): the MXFP8 reference, the tokens (.tokens.json), and the alignment check -- the offline
             MXFP8 emulation (the producers' encoder: per-32 E8M0 = floor(log2 amax) - 7, E4M3 RNE) must equal it to
             f32 summation order, or the rows are not aligned with the dump.
  --engine   engine walk dumps of builds serving an EXL3 head: each must equal its offline reconstruction to the
             kernel's f32 class (the engine computes what is graded here).

Per pair (reference P, candidate Q): KL(P||Q) mean / median / p99, top-1 agreement, NLL of the true next token
under each, dNLL = NLL(Q) - NLL(P) with a block-bootstrap 95% CI (blocks of --block scored positions), short-depth
rows (the first 256 / 1024 tokens of each document) and the deep tail (in-document depth >= 4096).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import head_exl3 as HX   # noqa: E402

V, E = HX.N_VOCAB, HX.N_EMBD


def klrf(path):
    h = np.fromfile(path, dtype=np.uint32, count=6)
    v, stride, pre, n = int(h[2]), int(h[3]), int(h[4]), int(h[5])
    return np.memmap(path, dtype=np.float32, mode='r', offset=24, shape=(n, v)), stride, pre


def bf16_to_f32(a):
    return torch.from_numpy((np.asarray(a).astype(np.uint32) << 16).view(np.float32))


def mxfp8(w):
    """The producers' MXFP8 of a bf16 [out][in] matrix, decoded back to f32 (exact values the engine multiplies)."""
    out = torch.empty_like(w)
    for s in range(0, w.shape[0], 8192):
        b = w[s:s + 8192].reshape(-1, E // 32, 32)
        amax = b.abs().amax(-1, keepdim=True)
        e = torch.where(amax > 0, torch.floor(torch.log2(amax)) - 7, torch.zeros_like(amax)).clamp(-127, 127)
        sc = torch.exp2(e)
        q = (b / sc).to(torch.float8_e4m3fn).float() * sc
        out[s:s + 8192] = q.reshape(-1, E)
    return out


def boot(x, B, R=2000, seed=0):
    rng = np.random.default_rng(seed)
    nb = len(x) // B
    xb = x[:nb * B].reshape(nb, B).mean(1)
    return np.percentile(xb[rng.integers(0, nb, (R, nb))].mean(1), [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', required=True)
    ap.add_argument('--klrf', required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--quant', nargs='*', default=[])
    ap.add_argument('--engine', nargs='*', default=[])
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--chunk', type=int, default=128)
    ap.add_argument('--out', help='save per-position arrays (npz)')
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())

    ref, stride, pre = klrf(a.klrf)
    toks = np.array(json.load(open(a.klrf + '.tokens.json')))
    n_calls = os.path.getsize(a.rows) // (2 * E)
    rows = np.memmap(a.rows, dtype=np.uint16, mode='r', shape=(n_calls, E))
    n = ref.shape[0]
    calls = stride * np.arange(n)                   # scored k = the head call at position pre - 1 + k * stride
    if calls[-1] >= n_calls:
        raise SystemExit(f'{a.rows}: {n_calls} head calls, the dump needs {calls[-1] + 1}')
    X = bf16_to_f32(rows[calls])                    # [n][E]
    tt = toks[pre + stride * np.arange(n)]

    W = {'bf16': bf16_to_f32(HX.read_bf16(a.source, 'lm_head.weight', (V, E)))}
    W['mxfp8'] = mxfp8(W['bf16'])
    for spec in a.quant:
        name, path = spec.split('=', 1)
        q = torch.load(path, weights_only=False)
        if q['draft'] or q['n'] != V:
            raise SystemExit(f'{path}: not an lm_head quant')
        W[name] = q['rec'].float()
        print(f'{name}: K={q["k"]} proxy err {q["proxy_err"]:.4e} rel Frobenius {q["rel_frob"]:.4e}', flush=True)
    eng = {}
    for spec in a.engine:
        name, path = spec.split('=', 1)
        if name not in W:
            raise SystemExit(f'--engine {name}: no --quant {name} to compare it with')
        eng[name], s2, p2 = klrf(path)
        if (s2, p2) != (stride, pre) or eng[name].shape != ref.shape:
            raise SystemExit(f'{path}: a different walk')

    names = list(W)
    cols = names + ['engine_mx'] + ['engine_' + k for k in eng]
    # per position: NLL of the true token, KL vs bf16 and vs the engine's MXFP8, top-1 agreement with each
    nll = {k: np.empty(n) for k in cols}
    kl_bf, kl_mx, t1_bf, t1_mx = ({k: np.empty(n) for k in cols} for _ in range(4))
    align = np.empty(n)
    eng_dev = {k: np.empty(n) for k in eng}

    def logp(z):
        z = z.double()
        return z - torch.logsumexp(z, 1, keepdim=True)

    for s in range(0, n, a.chunk):
        x = X[s:s + a.chunk]
        idx = torch.from_numpy(tt[s:s + a.chunk]).long()
        L = {k: x @ W[k].T for k in names}
        L['engine_mx'] = torch.from_numpy(np.asarray(ref[s:s + a.chunk]))
        for k in eng:
            L['engine_' + k] = torch.from_numpy(np.asarray(eng[k][s:s + a.chunk]))
        P = {k: logp(v) for k, v in L.items()}
        pb, pm = P['bf16'].exp(), P['engine_mx'].exp()
        r = slice(s, s + len(x))
        for k, lpk in P.items():
            nll[k][r] = -lpk.gather(1, idx[:, None])[:, 0].numpy()
            kl_bf[k][r] = (pb * (P['bf16'] - lpk)).sum(1).numpy()
            kl_mx[k][r] = (pm * (P['engine_mx'] - lpk)).sum(1).numpy()
            t1_bf[k][r] = (L[k].argmax(1) == L['bf16'].argmax(1)).numpy()
            t1_mx[k][r] = (L[k].argmax(1) == L['engine_mx'].argmax(1)).numpy()
        align[r] = ((L['mxfp8'] - L['engine_mx']).abs().amax(1) / L['engine_mx'].abs().amax(1)).numpy()
        for k in eng:   # the engine's EXL3 head vs its offline reconstruction
            e_ = L['engine_' + k]
            eng_dev[k][r] = ((e_ - L[k]).abs().amax(1) / e_.abs().amax(1)).numpy()
        print(f'  {s + len(x)}/{n}', end='\r', flush=True)
    print()
    print(f'walk: {n} scored positions (stride {stride}, prefix {pre}), {n_calls} head calls captured')
    print(f'alignment: offline MXFP8 vs the engine dump, max |diff| / max|logit| per row: max {align.max():.2e} '
          f'median {np.median(align):.2e}  ({"ALIGNED" if align.max() < 1e-4 else "NOT ALIGNED -- the grade is void"})')
    for k in eng:
        print(f'engine {k} head vs its offline reconstruction, max |diff| / max|logit| per row: max {eng_dev[k].max():.2e} '
              f'median {np.median(eng_dev[k]):.2e}')

    pos = pre + stride * np.arange(n)
    starts = [i for i in range(len(toks) - 1) if toks[i] == 248045 and toks[i + 1] == toks[1]] + [len(toks)]
    doc = np.searchsorted(starts, pos, side='right') - 1
    ind = pos - np.array(starts)[doc]

    def table(title, refk, KL, T1):
        print(f'\n== vs {title}: per candidate  KL mean / median / p99   top-1   NLL   dNLL [95% block CI]   '
              f'first-256 KL / top-1   first-1024 KL / top-1   depth>=4096 KL / top-1')
        for k in [c for c in nll if c != refk]:
            d = nll[k] - nll[refk]
            lo, hi = boot(d, a.block)
            m256, m1k, mdeep = ind < 256, ind < 1024, ind >= 4096
            print(f'  {k:10s} {KL[k].mean():.4e} / {np.median(KL[k]):.2e} / {np.percentile(KL[k], 99):.2e}  '
                  f'{100 * T1[k].mean():6.2f}%  {nll[k].mean():.5f}  {d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]  '
                  f'{KL[k][m256].mean():.2e} / {100 * T1[k][m256].mean():.2f}%  '
                  f'{KL[k][m1k].mean():.2e} / {100 * T1[k][m1k].mean():.2f}%  '
                  f'{KL[k][mdeep].mean() if mdeep.any() else float("nan"):.2e} / '
                  f'{100 * T1[k][mdeep].mean() if mdeep.any() else float("nan"):.2f}%')
        print(f'  ({title} NLL {nll[refk].mean():.5f})')

    table('BF16 (the checkpoint\'s head)', 'bf16', kl_bf, t1_bf)
    table('the engine\'s MXFP8 head', 'engine_mx', kl_mx, t1_mx)
    if a.out:
        np.savez(a.out, pos=pos, ind=ind, doc=doc, **{f'nll_{k}': v for k, v in nll.items()},
                 **{f'klbf_{k}': v for k, v in kl_bf.items()}, **{f'klmx_{k}': v for k, v in kl_mx.items()})


if __name__ == '__main__':
    main()
