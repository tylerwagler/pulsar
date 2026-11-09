#!/usr/bin/env python3
"""Compare a batched step's per-stage dumps against per-bank solo steps, row by
row, and name the first (layer, stage) where each row diverges.

Dumps come from PULSAR_CUDA_GRAPH_DUMP_PREFIX (gpu_diag.cpp): raw f32 files
named <prefix>_<stage>-<layer>_pos<pos>.bin, each [rows x width].  The gates
switch the prefix per step:
  mixed_neutrality_gate (L152_DUMP_DIR): batched under bat_f, bank k solo
      under solo<k>          -> default prefixes
  dspark_batch_gate (L150_DUMP_DIR):     batched under t0_bat, bank b
      serialized under t0_ser_b<b>  -> --batched t0_bat --solo 't0_ser_b{k}'

usage: dumpcmp_rows.py DIR rows_bank0 rows_bank1 [...] [--batched PFX] [--solo FMT]
       [--detail-layers N]   (print every differing (stage,row) up to layer N; default 2)
The batched step's rows are the banks' rows in issue order."""
import sys, glob, re, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('dir')
ap.add_argument('rows', nargs='+', type=int, help='rows per bank, in issue order')
ap.add_argument('--batched', default='bat_f')
ap.add_argument('--solo', default='solo{k}')
ap.add_argument('--detail-layers', type=int, default=2)
a = ap.parse_args()

rpb = a.rows
off = [sum(rpb[:k]) for k in range(len(rpb))]
total = sum(rpb)
pat = re.compile(r'/' + re.escape(a.batched) + r'_([a-z_0-9]+)-(\d+)_pos(\d+)\.bin$')

keys = sorted({(int(m.group(2)), m.group(1), m.group(3))
               for f in glob.glob(f'{a.dir}/{a.batched}_*.bin')
               for m in [pat.search(f)] if m})
if not keys:
    sys.exit(f'no {a.batched}_*.bin under {a.dir}')

first = {}
for il, name, pos in keys:
    bat = np.fromfile(f'{a.dir}/{a.batched}_{name}-{il}_pos{pos}.bin', dtype=np.float32)
    if bat.size % total:
        continue                      # not a per-row tensor
    width = bat.size // total
    brows = bat.reshape(total, width)
    for k in range(len(rpb)):
        sf = glob.glob(f'{a.dir}/{a.solo.format(k=k)}_{name}-{il}_pos*.bin')
        if not sf:
            continue
        ser = np.fromfile(sf[0], dtype=np.float32)
        if ser.size != rpb[k] * width:
            print(f'{name} L{il} bank {k}: solo size {ser.size} != {rpb[k]}x{width}')
            continue
        ser = ser.reshape(rpb[k], width)
        mine = brows[off[k]:off[k] + rpb[k]]
        diff = ser != mine
        for r in range(rpb[k]):
            if not diff[r].any():
                continue
            n, mx = int(diff[r].sum()), float(np.abs(ser[r] - mine[r]).max())
            first.setdefault((k, r), (il, name, n, mx))
            if il <= a.detail_layers:
                print(f'L{il} {name:<22} bank {k} row {r}: DIFF {n}/{width} elems, max |d| {mx:.4g}')

print('--- first divergence per row ---')
for k in range(len(rpb)):
    for r in range(rpb[k]):
        v = first.get((k, r))
        print(f'bank {k} row {r}: ' + (f'L{v[0]} {v[1]} ({v[2]} elems, max |d| {v[3]:.4g})' if v
                                       else 'identical at every dumped stage'))
