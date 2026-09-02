#!/usr/bin/env python3
"""L152: compare GATE 5R's per-stage dumps of the batched step (bat_f, rows in
issue order: bank 0's rpb0 rows then bank 1's rpb1 rows) against each bank's
solo step (solo<k>, rows [0, rpb_k)), per stage and layer, per row.  Prints the
FIRST (layer, stage) at which each row differs and the layer-0..2 detail.

usage: l152_dumpcmp.py DIR rpb0 rpb1 [rpb2 ...]"""
import sys, glob, re
import numpy as np

d = sys.argv[1]
rpb = [int(x) for x in sys.argv[2:]]
off = [sum(rpb[:k]) for k in range(len(rpb))]
total = sum(rpb)

def stage_key(f):
    m = re.search(r'/bat_f_([a-z_0-9]+)-(\d+)_pos(\d+)\.bin$', f)
    return (int(m.group(2)), m.group(1), m.group(3)) if m else None

files = sorted({stage_key(f) for f in glob.glob(f'{d}/bat_f_*.bin') if stage_key(f)})
first = {}   # (bank,row) -> (layer, stage)
for il, name, pos in files:
    bat = np.fromfile(f'{d}/bat_f_{name}-{il}_pos{pos}.bin', dtype=np.float32)
    if bat.size % total:
        continue           # not a per-row tensor (indexer scores etc.)
    width = bat.size // total
    brows = bat.reshape(total, width)
    for k in range(len(rpb)):
        sf = glob.glob(f'{d}/solo{k}_{name}-{il}_pos*.bin')
        if not sf:
            continue
        ser = np.fromfile(sf[0], dtype=np.float32)
        if ser.size != rpb[k] * width:
            print(f'{name} L{il} bank {k}: solo size {ser.size} != {rpb[k]}x{width}')
            continue
        ser = ser.reshape(rpb[k], width)
        diff = ser != brows[off[k]:off[k] + rpb[k]]
        for r in range(rpb[k]):
            if diff[r].any():
                if (k, r) not in first:
                    first[(k, r)] = (il, name, int(diff[r].sum()), float(np.abs(ser[r] - brows[off[k] + r]).max()))
                if il <= 2:
                    print(f'L{il} {name:<22} bank {k} row {r}: DIFF {int(diff[r].sum())}/{width} elems, '
                          f'max |d| {float(np.abs(ser[r] - brows[off[k] + r]).max()):.4g}')
print('--- first divergence per row ---')
for k in range(len(rpb)):
    for r in range(rpb[k]):
        v = first.get((k, r))
        print(f'bank {k} row {r}: ' + (f'L{v[0]} {v[1]} ({v[2]} elems, max |d| {v[3]:.4g})' if v else 'identical at every dumped stage'))
