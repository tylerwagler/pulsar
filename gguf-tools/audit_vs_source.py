#!/usr/bin/env python3
"""Every ds4 tensor family: what we SHIP, what we'd BUILD now, what SOURCE has.

Main model and drafter both. The drafter is not a separate model -- our dspark.*
is the checkpoint's mtp.* module -- so it belongs in the same table; leaving it
out is what hid a 132 MB item.

The BUILD column is read out of the live type policy (build_main_template's
suffix_type for the main model, dspark_type_flags.txt for the drafter), not from
a list maintained here, so this map cannot drift from what the tools actually
emit.

Three distinctions a plain dtype diff gets wrong:

  - LAYOUT REPACKS ARE NOT FORMAT CHANGES. MXFP8_LT is FP8_E4M3 pre-swizzled,
    CUTLASS_MXFP4 is MXFP4 pre-swizzled: same values, different byte order.
    They read AT source.
  - THE CONTAINER IS NOT THE FORMAT. Routed experts ship as safetensors I8 plus
    an F8_E8M0 scale, two fp4 per byte. config.json calls that fp4, and that is
    what it is compared against.
  - ABOVE SOURCE IS NOT A DEFECT. f32 holds every bf16 value exactly and f16
    holds every e4m3 code exactly, so those rows are correct output in too many
    bytes -- a size problem, not a fidelity one. Only BELOW source loses
    information.

Usage: audit_vs_source.py
"""
import os
import re
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_main_template import HFCheckpoint, LAYER_MAP, TOP_MAP, suffix_type  # noqa: E402
from gguf_hdr import scan  # noqa: E402

# NOTE this is the pre-repack artifact, the newest one present on the dev box.
# The SERVING file is v5mx4-0731-ltdraft.gguf (sparky), where the 2026-08-12
# repack already moved 25 drafter dense tensors FP8_E4M3 -> MXFP8_LT. Those 25
# rows therefore show as "changing" here and are already done in production;
# point ART at the ltdraft file when auditing what actually serves.
ART = '/mnt/models/ds4flash-v5mx4-0731-reap25-mxfp8head-dspark0731-v3.gguf'
HF = '/mnt/models/dsv4-flash-0731'
FLAGS = os.path.join(HERE, 'dspark_type_flags.txt')
# The main model's attention and expert types come from the quantizer's
# --format-map, NOT from the template's suffix_type -- reading only the template
# reports the attention stack as FP8_E4M3 when the build actually emits the
# pre-swizzled MXFP8_LT twin, which looks like a regression that is not there.
FORMAT_MAP = os.path.join(HERE, 'prisma', 'v5mx4-format-map.json')

TY = {0: 'F32', 1: 'F16', 16: 'IQ2_XXS', 26: 'I32', 30: 'BF16', 38: 'FP8_E4M3',
      39: 'MXFP4', 40: 'CUTLASS_MXFP4', 41: 'MXFP8_LT', 42: 'IQ2_XXS_SOA',
      43: 'IQ2_XXS_MMQ'}
BYNAME = {'f32': 'F32', 'f16': 'F16', 'bf16': 'BF16', 'fp8_e4m3': 'FP8_E4M3',
          'mxfp8_lt': 'MXFP8_LT', 'cutlass_mxfp4': 'CUTLASS_MXFP4',
          'iq2_xxs_mmq': 'IQ2_XXS_MMQ'}
REPACK = {'MXFP8_LT': 'FP8_E4M3', 'CUTLASS_MXFP4': 'MXFP4',
          'IQ2_XXS_SOA': 'IQ2_XXS', 'IQ2_XXS_MMQ': 'IQ2_XXS'}
SRCNORM = {'F8_E4M3': 'FP8_E4M3', 'I8': 'MXFP4'}
EXPERTS = {'ffn_gate_exps.weight': 'w1', 'ffn_up_exps.weight': 'w3',
           'ffn_down_exps.weight': 'w2'}
DSPARK_EXTRA = {
    'main_norm.weight': 'main_norm.weight', 'main_proj.weight': 'main_proj.weight',
    'markov_head.markov_w1.weight': 'markov_head.markov_w1.weight',
    'markov_head.markov_w2.weight': 'markov_head.markov_w2.weight',
    'confidence_head.proj.weight': 'confidence_head.proj.weight',
    'norm.weight': 'norm.weight', 'hc_head_base.weight': 'hc_head_base',
    'hc_head_fn.weight': 'hc_head_fn', 'hc_head_scale.weight': 'hc_head_scale',
}


def hf_name(ds4):
    if ds4.startswith('blk.'):
        _, layer, suf = ds4.split('.', 2)
        if suf in EXPERTS:
            return f'layers.{layer}.ffn.experts.0.{EXPERTS[suf]}.weight'
        return f'layers.{layer}.{LAYER_MAP[suf]}' if suf in LAYER_MAP else None
    if ds4.startswith('dspark.'):
        rest = ds4[len('dspark.'):]
        head = rest.split('.', 1)[0]
        if head.isdigit():
            suf = rest.split('.', 1)[1]
            if suf in EXPERTS:
                return f'mtp.{head}.ffn.experts.0.{EXPERTS[suf]}.weight'
            if suf in LAYER_MAP:
                return f'mtp.{head}.{LAYER_MAP[suf]}'
            return f'mtp.{head}.{DSPARK_EXTRA[suf]}' if suf in DSPARK_EXTRA else None
        return f'mtp.0.{DSPARK_EXTRA[rest]}' if rest in DSPARK_EXTRA else None
    return TOP_MAP.get(ds4)


def load_flags():
    out = {}
    txt = open(FLAGS).read()
    for name, ty in re.findall(r'--tensor-type\s+(\S+)=(\S+)', txt):
        out[name] = BYNAME.get(ty, ty)
    return out


def load_format_map():
    import json
    try:
        return json.load(open(FORMAT_MAP))
    except OSError:
        return {}


def build_type(ds4, dims, flags, fmap):
    """What the CURRENT policy would emit: quantizer overrides first, then the
    template default -- the same precedence the build itself uses."""
    if ds4.startswith('dspark.'):
        return flags.get(ds4)          # None -> template default, shown as '?'
    if ds4 in fmap:
        return fmap[ds4]
    try:
        return TY.get(suffix_type(ds4, len(dims)))
    except Exception:
        return None


def verdict(ours, src):
    if ours is None or src == '-':
        return '?'
    base = REPACK.get(ours, ours)
    tag = ' (repack)' if base != ours else ''
    if base == src:
        return 'AT source' + tag
    order = {'IQ2_XXS': 0, 'MXFP4': 1, 'FP8_E4M3': 2, 'BF16': 3, 'F16': 3, 'F32': 4}
    a, b = order.get(base, 9), order.get(src, 9)
    if base == 'F16' and src == 'BF16':
        return 'WRONG 16-bit'
    if a < b:
        return 'BELOW source'
    if a > b:
        return 'ABOVE source'
    return f'{base} vs {src}'


def main():
    ck = HFCheckpoint(HF)
    flags = load_flags()
    fmap = load_format_map()
    tensors, _, _ = scan(ART)
    rows = OrderedDict()
    for t in tensors:
        nm = t['name']
        hf = hf_name(nm)
        src = SRCNORM.get(ck.dtype(hf), ck.dtype(hf)) if (hf and ck.has(hf)) else '-'
        ours = TY.get(t['type'], str(t['type']))
        new = build_type(nm, t['dims'], flags, fmap)
        if nm.startswith('blk.'):
            fam = nm.split('.', 2)[-1]
        elif nm.startswith('dspark.'):
            p = nm.split('.')
            fam = 'dspark.' + ('.'.join(p[2:]) if p[1].isdigit() else '.'.join(p[1:]))
        else:
            fam = nm
        rows.setdefault((fam, ours, new, src), [0])[0] += 1

    W = '%-38s %-13s %-13s %-9s %-16s %4s'
    changed = [(k, v) for k, v in rows.items() if k[1] != k[2] and k[2]]
    same = [(k, v) for k, v in rows.items() if not (k[1] != k[2] and k[2])]
    for grp, title in ((changed, 'CHANGING'), (same, 'UNCHANGED')):
        if not grp:
            continue
        print('\n=== %s ===' % title)
        print(W % ('family', 'SHIP NOW', 'BUILD NOW', 'SOURCE', 'after', 'n'))
        print('-' * 98)
        for (fam, ours, new, src), (n,) in sorted(grp):
            print(W % (fam[:38], ours, new or '?', src, verdict(new or ours, src), n))


if __name__ == '__main__':
    main()
