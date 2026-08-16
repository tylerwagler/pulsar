#!/usr/bin/env python3
"""Every tensor we store WIDER than the checkpoint does, and what it costs.

Above-source is not a fidelity bug -- f32 represents every bf16 value exactly,
and f16 represents every e4m3 value exactly -- so nothing here is wrong output.
It is pure waste: bytes on disk and in VRAM carrying no information.

Covers the DRAFTER as well as the main model. The drafter is not a separate
model: our dspark.* tensors are the checkpoint's mtp.* module (3 stages, 4705
tensors upstream), so they have source counterparts and belong in this audit.
Leaving them out understates the total by more than an order of magnitude --
mtp.2.markov_head alone is two [129280, 256] tensors we hold f32 against a bf16
source.

Usage: audit_above_source.py
"""
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_main_template import HFCheckpoint, LAYER_MAP, TOP_MAP  # noqa: E402
from gguf_hdr import scan  # noqa: E402

ART = '/mnt/models/ds4flash-v5mx4-0731-reap25-mxfp8head-dspark0731-v3.gguf'
HF = '/mnt/models/dsv4-flash-0731'

TY = {0: 'F32', 1: 'F16', 16: 'IQ2_XXS', 26: 'I32', 30: 'BF16', 38: 'FP8_E4M3',
      39: 'MXFP4', 40: 'CUTLASS_MXFP4', 41: 'MXFP8_LT', 43: 'IQ2_XXS_MMQ'}
# bytes per element. MXFP8 (38/41) is E4M3 + a per-32 E8M0 scale == 33 B / 32.
WIDTH = {'F32': 4.0, 'F16': 2.0, 'BF16': 2.0, 'FP8_E4M3': 33.0 / 32.0,
         'MXFP8_LT': 33.0 / 32.0}
SRCNORM = {'F8_E4M3': 'FP8_E4M3'}

# dspark.* IS mtp.* -- same module, our name vs theirs.
DSPARK_EXTRA = {
    'main_norm.weight': 'main_norm.weight',
    'main_proj.weight': 'main_proj.weight',
    'markov_head.markov_w1.weight': 'markov_head.markov_w1.weight',
    'markov_head.markov_w2.weight': 'markov_head.markov_w2.weight',
    'final_norm.weight': 'norm.weight',
    'norm.weight': 'norm.weight',
    'confidence_head.proj.weight': 'confidence_head.proj.weight',
    # upstream drops the .weight suffix on the hc_head trio
    'hc_head_base.weight': 'hc_head_base',
    'hc_head_fn.weight': 'hc_head_fn',
    'hc_head_scale.weight': 'hc_head_scale',
}


def hf_name(ds4):
    if ds4.startswith('blk.'):
        _, layer, suf = ds4.split('.', 2)
        return f'layers.{layer}.{LAYER_MAP[suf]}' if suf in LAYER_MAP else None
    if ds4.startswith('dspark.'):
        rest = ds4[len('dspark.'):]
        head = rest.split('.', 1)[0]
        if head.isdigit():
            suf = rest.split('.', 1)[1]
            if suf in LAYER_MAP:
                return f'mtp.{head}.{LAYER_MAP[suf]}'
            if suf in DSPARK_EXTRA:
                return f'mtp.{head}.{DSPARK_EXTRA[suf]}'
            return None
        # unstaged: ours carries one, source carries one per stage -- stage 0
        return f'mtp.0.{DSPARK_EXTRA[rest]}' if rest in DSPARK_EXTRA else None
    return TOP_MAP.get(ds4)


def main():
    ck = HFCheckpoint(HF)
    tensors, _, _ = scan(ART)
    rows, unmapped = OrderedDict(), []

    for t in tensors:
        ours = TY.get(t['type'])
        hf = hf_name(t['name'])
        if hf is None or not ck.has(hf):
            if ours in WIDTH:
                unmapped.append(t['name'])
            continue
        src = SRCNORM.get(ck.dtype(hf), ck.dtype(hf))
        if ours not in WIDTH or src not in WIDTH:
            continue
        if WIDTH[ours] <= WIDTH[src]:
            continue
        fam = t['name'].split('.', 2)[-1] if t['name'].startswith('blk.') else t['name']
        if t['name'].startswith('dspark.'):
            p = t['name'].split('.')
            fam = 'dspark.' + ('.'.join(p[2:]) if p[1].isdigit() else '.'.join(p[1:]))
        r = rows.setdefault((fam, ours, src), [0, 0.0, 0.0])
        r[0] += 1
        r[1] += t['ne'] * WIDTH[ours]
        r[2] += t['ne'] * WIDTH[src]

    W = '%-40s %-9s %-9s %5s %11s %11s'
    print(W % ('family', 'ours', 'source', 'n', 'now', 'saved'))
    print('-' * 92)
    tn = tc = tsv = 0
    for (fam, ours, src), (n, cur, at) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
        print(W % (fam[:40], ours, src, n, '%.2f MB' % (cur / 1e6), '%.2f MB' % ((cur - at) / 1e6)))
        tn += n
        tc += cur
        tsv += cur - at
    print('-' * 92)
    print(W % ('TOTAL', '', '', tn, '%.2f MB' % (tc / 1e6), '%.2f MB' % (tsv / 1e6)))
    sz = os.path.getsize(ART)
    print('\nartifact %.1f GB; reclaiming all of it is %.3f%%' % (sz / 1e9, 100.0 * tsv / sz))
    if unmapped:
        print('\nunmapped (no source name known, NOT audited): %d' % len(unmapped))
        for n in unmapped[:10]:
            print('   %s' % n)


if __name__ == '__main__':
    main()
