#!/usr/bin/env python3
"""L288 qwen-exl3-head: Qwen3.8-Flash-Next's heads in EXL3 -- the Hessian, the quantization, the container.

The engine reads a head in whatever format the container stores it (family_qwen_s4.cpp admit_head): lm_head.weight
bf16 (encoded to MXFP8 on the device, L251) or EXL3 at a dense-arm rate (the dense bf16-rows arm as stored), and the
MTP drafter's own head mtp.draft_head.weight + mtp.draft_head.ids when the container carries them.  This tool makes
those tensors from the checkpoint's BF16 lm_head with exllamav3's quantizer, calibrated the way U-e4-d5 was (L251:
the 418 calibration rows of calib-qwen38-v1.jsonl, <= 2048 tokens each; mul1 codebook, apply_out_scales, sigma_reg
default), on the head's own input -- the trunk mixer's bf16 rows.

  hessian  --rows F [F ...] --out H.pt
      H = sum x x^T over every row of the capture files (bf16 [n][2560], the head input, written by the engine's
      QHEAD capture instrument during a prefill of the calibration rows); saved with the row count.
  quant    --source DIR --hessian H.pt --k K [--draft] --seed S --out Q.pt
      lm_head (or its draft-vocabulary rows, --draft) [in 2560, out N] -> exllamav3 quantize_exl3 on the GPU; saves the
      packed trellis / suh / svh, the reconstruction (bf16, for grading), the proxy error and the ids (--draft).
  emit     --base DIR --out DIR [--head Q.pt] [--draft Q.pt]
      a container that IS `base` (every file a symlink) except: --head drops the bf16 lm_head from the shard that held
      it (that shard rewritten, its other tensors verbatim) and adds lm_head.weight in EXL3; --draft adds
      mtp.draft_head.weight + .ids.  The new tensors go in model-qhead.safetensors.  `base` is never written.

The draft head's rows are the engine's draft vocabulary in the engine's order (qwen_draft_vocab, family_qwen.cpp): ids
[0, QWEN_DRAFT_PREFIX), the first `extra` of src/engine/qwen_draft_vocab.inc's ranked list sorted ascending, then
[eos_id, n_vocab).  EXL3 needs the row count % 128 == 0, so `extra` is the default rounded up to make it so (the
engine derives `extra` from the container's row count and checks the stated ids).
"""
import argparse
import json
import os
import re
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INC = os.path.join(HERE, '..', '..', 'src', 'engine', 'qwen_draft_vocab.inc')
N_EMBD, N_VOCAB, EOS_ID = 2560, 248320, 248044          # Flash-Next (PULSAR_QWEN_SHAPE_FLASH_NEXT)
EXL3_LAYOUT = {3: 'exl3m_k3', 4: 'exl3m_k4', 5: 'exl3m_k5'}
ALIGN = 32


# ---- safetensors (header-level; payloads by offset) ---------------------------------------------------------------
def st_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def find_tensor(d, name):
    for fn in sorted(os.listdir(d)):
        if fn.endswith('.safetensors'):
            h, off = st_header(os.path.join(d, fn))
            if name in h:
                return os.path.join(d, fn), h, off
    raise SystemExit(f'{d}: no shard holds {name}')


def read_bf16(d, name, shape):
    path, h, off = find_tensor(d, name)
    e = h[name]
    if e['dtype'] != 'BF16' or e['shape'] != list(shape):
        raise SystemExit(f'{name}: {e["dtype"]} {e["shape"]}, expected BF16 {list(shape)}')
    a, b = e['data_offsets']
    return np.memmap(path, dtype=np.uint16, mode='r', offset=off + a, shape=tuple(shape))


def write_shard(path, entries, meta):
    """entries: [(name, dtype, shape, bytes-like)] -> a safetensors file, payloads 32-byte aligned (each entry's
    size is a multiple of 32 or it goes last)."""
    entries = sorted(entries, key=lambda e: e[3].nbytes % ALIGN != 0)
    header, cur = {}, 0
    for name, dt, shape, data in entries:
        if cur % ALIGN:
            raise SystemExit(f'{name}: offset {cur} breaks the {ALIGN}-byte alignment')
        header[name] = {'dtype': dt, 'shape': list(shape), 'data_offsets': [cur, cur + data.nbytes]}
        cur += data.nbytes
    hj = json.dumps({**header, '__metadata__': meta}, separators=(',', ':'), sort_keys=True).encode()
    hj += b' ' * (-(8 + len(hj)) % ALIGN)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(struct.pack('<Q', len(hj)))
        f.write(hj)
        for _, _, _, data in entries:
            f.write(memoryview(data).cast('B'))
    os.replace(tmp, path)


# ---- the draft vocabulary (the engine's rule) ---------------------------------------------------------------------
def draft_ids(extra=None):
    src = open(INC).read()
    prefix = int(re.search(r'#define QWEN_DRAFT_PREFIX (\d+)u', src).group(1))
    default = int(re.search(r'#define QWEN_DRAFT_EXTRA_DEFAULT (\d+)u', src).group(1))
    ranked = [int(x) for x in re.findall(r'-?\d+', src.split('k_qwen_draft_extra[] = {', 1)[1].split('};', 1)[0])]
    if extra is None:
        n = prefix + default + (N_VOCAB - EOS_ID)
        extra = default + (-n % 128)                    # EXL3: rows % 128 == 0
    if extra > len(ranked):
        raise SystemExit(f'extra {extra} > the {len(ranked)} ranked ids')
    ids = list(range(prefix)) + sorted(ranked[:extra]) + list(range(EOS_ID, N_VOCAB))
    return np.array(ids, dtype=np.int32), extra


# ---- hessian ------------------------------------------------------------------------------------------------------
def cmd_hessian(a):
    import torch
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    H = torch.zeros(N_EMBD, N_EMBD, dtype=torch.float64, device=dev)
    count = 0
    for fn in a.rows:
        n = os.path.getsize(fn) // (2 * N_EMBD)
        if os.path.getsize(fn) != n * 2 * N_EMBD:
            raise SystemExit(f'{fn}: not whole bf16 [n][{N_EMBD}] rows')
        mm = np.memmap(fn, dtype=np.uint16, mode='r', shape=(n, N_EMBD))
        for s in range(0, n, 16384):
            # each chunk's sum in f32 (16k terms), the chunks summed in f64
            x = torch.from_numpy((mm[s:s + 16384].astype(np.uint32) << 16).view(np.float32)).to(dev)
            if not torch.isfinite(x).all():
                raise SystemExit(f'{fn}: non-finite rows near {s}')
            H += (x.T @ x).double()
        count += n
        print(f'{fn}: {n:,} rows', flush=True)
    H = H.cpu()
    torch.save({'H': H.float(), 'count': count, 'files': a.rows}, a.out)
    d = torch.diagonal(H) / count
    print(f'H over {count:,} rows -> {a.out}: diag mean {d.mean():.4e} min {d.min():.3e} max {d.max():.3e}')


# ---- quant --------------------------------------------------------------------------------------------------------
def cmd_quant(a):
    import torch
    from exllamav3.modules.quant.exl3_lib import quantize as XQ
    hd = torch.load(a.hessian)
    W = read_bf16(a.source, 'lm_head.weight', (N_VOCAB, N_EMBD))
    ids, extra = draft_ids() if a.draft else (None, None)
    rows = np.asarray(W[ids] if a.draft else W)
    n = rows.shape[0]
    w = torch.from_numpy((rows.astype(np.uint32) << 16).view(np.float32))      # [out, in] f32, exact
    dev = torch.device('cuda:0')
    qa = {'seed': a.seed, 'K': a.k, 'devices': [0], 'device_ratios': None, 'apply_out_scales': True,
          'debug_dir': None, 'mul1': True}
    H = {'H': hd['H'].to(dev).clone(), 'first_key': 'lm_head', 'count': hd['count'], 'finalized': False,
         'num_total': hd['count'] * N_EMBD, 'inf_nan': torch.zeros(2), 'device': dev}
    print(f'quantizing {"draft head" if a.draft else "lm_head"} [{N_EMBD} -> {n}] at K={a.k} (H over {hd["count"]:,} '
          f'rows, seed {a.seed})', flush=True)
    wq, perr, out = XQ.quantize_exl3(w.t().contiguous(), H, qa, True)
    tr, suh, svh = out['trellis'], out['suh'], out['svh']
    if 'mul1' not in out or tuple(tr.shape) != (N_EMBD // 16, n // 16, 16 * a.k) or tr.dtype != torch.int16:
        raise SystemExit(f'unexpected quantizer output: {sorted(out)} trellis {tuple(tr.shape)} {tr.dtype}')
    rec = wq.t().contiguous()                                             # [out, in]
    rel = ((rec.float().cpu() - w).norm() / w.norm()).item()
    torch.save({'k': a.k, 'draft': a.draft, 'n': n, 'extra': extra, 'ids': ids, 'proxy_err': float(perr),
                'rel_frob': rel, 'q_fallback': bool(qa.get('q_fallback', False)),
                'trellis': tr.cpu(), 'suh': suh.cpu(), 'svh': svh.cpu(), 'rec': rec.to(torch.bfloat16).cpu(),
                'seed': a.seed, 'hessian': a.hessian, 'source': a.source}, a.out)
    print(f'K={a.k} {"draft" if a.draft else "lm_head"}: proxy err {perr:.4e}, rel Frobenius {rel:.4e}, '
          f'fallback {qa.get("q_fallback")} -> {a.out}')


def exl3_blob(q):
    """The engine's byte model (exl3_expert_layout): trellis I16 [in/16][out/16][16K] | suh F16 [in] | svh F16 [out]."""
    tr = q['trellis'].numpy().astype('<i2', copy=False)
    suh = q['suh'].numpy().astype('<f2', copy=False)
    svh = q['svh'].numpy().astype('<f2', copy=False)
    if tr.shape != (N_EMBD // 16, q['n'] // 16, 16 * q['k']) or suh.shape != (N_EMBD,) or svh.shape != (q['n'],):
        raise SystemExit('the packed tensors do not have the layout')
    return np.concatenate([tr.reshape(-1).view(np.uint8), suh.view(np.uint8), svh.view(np.uint8)])


# ---- emit ---------------------------------------------------------------------------------------------------------
def cmd_emit(a):
    import torch
    base, out = os.path.abspath(a.base), os.path.abspath(a.out)
    if out == base or os.path.exists(out):
        raise SystemExit(f'{out} exists or is the base -- a container copy is made fresh, never over another')
    if not a.head and not a.draft:
        raise SystemExit('nothing to change: give --head and / or --draft')
    os.makedirs(out)
    top = None
    tensors, entries = {}, []
    if a.head:
        q = torch.load(a.head, weights_only=False)
        if q['draft'] or q['n'] != N_VOCAB:
            raise SystemExit(f'{a.head} is not an lm_head quant')
        blob = exl3_blob(q)
        entries.append(('lm_head.weight', 'U8', [len(blob)], blob))
        tensors['lm_head.weight'] = {'layout': EXL3_LAYOUT[q['k']], 'gguf_name': 'lm_head.weight',
                                     'dims_ne': [N_EMBD, N_VOCAB]}
        top, h, off = find_tensor(base, 'lm_head.weight')
    if a.draft:
        q = torch.load(a.draft, weights_only=False)
        if not q['draft']:
            raise SystemExit(f'{a.draft} is not a draft-head quant')
        ids, _ = draft_ids(q['extra'])
        if not np.array_equal(ids, q['ids']):
            raise SystemExit(f'{a.draft}: its ids are not the draft vocabulary\'s')
        blob = exl3_blob(q)
        entries.append(('mtp.draft_head.weight', 'U8', [len(blob)], blob))
        entries.append(('mtp.draft_head.ids', 'I32', [q['n']], ids.astype('<i4')))
        tensors['mtp.draft_head.weight'] = {'layout': EXL3_LAYOUT[q['k']], 'gguf_name': 'mtp.draft_head.weight',
                                            'dims_ne': [N_EMBD, q['n']]}
        tensors['mtp.draft_head.ids'] = {'layout': 'i32', 'gguf_name': 'mtp.draft_head.ids', 'dims_ne': [q['n']]}
    for fn in sorted(os.listdir(base)):
        if fn == 'model.safetensors.index.json' or fn == 'SHA256SUMS':
            continue                                    # describe the base's shards, not this container's
        src = os.path.realpath(os.path.join(base, fn))
        if top and src == os.path.realpath(top):
            # the shard that held the bf16 lm_head, rewritten without it: every other tensor verbatim
            meta = dict(h['__metadata__'])
            pt = json.loads(meta['pulsar.tensors'])
            pt.pop('lm_head.weight')
            meta['pulsar.tensors'] = json.dumps(pt, separators=(',', ':'), sort_keys=True)
            raw = np.memmap(src, dtype=np.uint8, mode='r')
            keep = [(k, v['dtype'], v['shape'], raw[off + v['data_offsets'][0]: off + v['data_offsets'][1]])
                    for k, v in h.items() if k not in ('__metadata__', 'lm_head.weight')]
            write_shard(os.path.join(out, fn), keep, meta)
            print(f'{fn}: rewritten without lm_head.weight ({len(keep)} tensors kept verbatim)')
        else:
            os.symlink(src, os.path.join(out, fn))
    write_shard(os.path.join(out, 'model-qhead.safetensors'), entries,
                {'format': 'pt', 'pulsar.format': 'pulsar-safetensors-v1', 'pulsar.shard_key': 'qhead',
                 'pulsar.tensors': json.dumps(tensors, separators=(',', ':'), sort_keys=True),
                 'pulsar.qhead': json.dumps({'base': base, 'head': a.head, 'draft': a.draft})})
    print(f'{out}: {", ".join(f"{k} {v["layout"]}" for k, v in tensors.items())}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('hessian')
    p.add_argument('--rows', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p = sp.add_parser('quant')
    p.add_argument('--source', required=True, help='a container (or checkpoint dir) holding the BF16 lm_head.weight')
    p.add_argument('--hessian', required=True)
    p.add_argument('--k', type=int, required=True, choices=(3, 4, 5))
    p.add_argument('--draft', action='store_true', help='the draft-vocabulary rows instead of the whole head')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', required=True)
    p = sp.add_parser('emit')
    p.add_argument('--base', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--head')
    p.add_argument('--draft')
    a = ap.parse_args()
    {'hessian': cmd_hessian, 'quant': cmd_quant, 'emit': cmd_emit}[a.cmd](a)


if __name__ == '__main__':
    main()
