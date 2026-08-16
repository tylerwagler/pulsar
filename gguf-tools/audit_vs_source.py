"""Audit every ds4 tensor family against the upstream checkpoint.

Answers one question per family: is what we ship AT the source format, ABOVE it
(holding more bits than the source carries), or BELOW it (a real fidelity loss)?

Three things this gets right that a naive dtype diff does not:

  - LAYOUT REPACKS ARE NOT FORMAT CHANGES. MXFP8_LT is FP8_E4M3 pre-swizzled and
    CUTLASS_MXFP4 is MXFP4 pre-swizzled -- same values, different byte order --
    so they read AT source, not as a difference.
  - THE CONTAINER IS NOT THE FORMAT. Routed experts ship as safetensors I8 with
    an F8_E8M0 companion scale, two fp4 per byte; config.json calls that fp4 and
    that is what it is compared against.
  - SCALE GRANULARITY IS REPORTED, because it is a separate axis from width. The
    fp8 dense weights carry 128x128 block scales; the fp4 experts carry per-32,
    which is OCP MX granularity and exactly what our MXFP4 uses.

Usage: audit_vs_source.py   (paths are pinned below)
"""
import sys
from collections import OrderedDict

sys.path.insert(0, '/home/claude/Projects/pulsar/.claude/worktrees/dsml-truncated-toolcall/gguf-tools')
from build_main_template import HFCheckpoint
from gguf_hdr import scan
from splice_bf16_plain import hf_name_for, suffix_of

ART = '/mnt/models/ds4flash-v5mx4-0731-reap25-mxfp8head-dspark0731-v3.gguf'
TY = {0: 'F32', 1: 'F16', 16: 'IQ2_XXS', 26: 'I32', 30: 'BF16', 38: 'FP8_E4M3',
      39: 'MXFP4', 40: 'CUTLASS_MXFP4', 41: 'MXFP8_LT', 42: 'IQ2_XXS_SOA', 43: 'IQ2_XXS_MMQ'}
# ours -> the format it is a pure LAYOUT repack of (same values, different bytes)
REPACK = {'MXFP8_LT': 'FP8_E4M3', 'CUTLASS_MXFP4': 'MXFP4',
          'IQ2_XXS_SOA': 'IQ2_XXS', 'IQ2_XXS_MMQ': 'IQ2_XXS'}
# safetensors spelling -> ours
NORM = {'F8_E4M3': 'FP8_E4M3', 'I8': 'MXFP4'}   # routed experts: I8 pack + E8M0 == fp4

EXPERTS = {'ffn_gate_exps.weight': 'w1', 'ffn_up_exps.weight': 'w3',
           'ffn_down_exps.weight': 'w2'}

ck = HFCheckpoint('/mnt/models/dsv4-flash-0731')
tensors, _, _ = scan(ART)


def src_of(name):
    """(source format, scale granularity) for a ds4 tensor, normalised to our names."""
    fam = suffix_of(name)
    if fam in EXPERTS and name.startswith('blk.'):
        layer = name.split('.')[1]
        w = EXPERTS[fam]
        hf = f'layers.{layer}.ffn.experts.0.{w}.weight'
        if not ck.has(hf):
            return '-', ''
        d, s = ck.shape(hf), ck.shape(f'layers.{layer}.ffn.experts.0.{w}.scale')
        # I8 holds two fp4 per byte, so the logical row is twice the stored one
        return 'MXFP4', 'per-%d' % ((d[1] * 2) // s[1])
    hf = hf_name_for(name)
    if hf is None or not ck.has(hf):
        return '-', ''
    dt = NORM.get(ck.dtype(hf), ck.dtype(hf))
    gran = ''
    if dt == 'FP8_E4M3':
        sc = hf.rsplit('.', 1)[0] + '.weight_scale_inv'
        for cand in (hf.replace('.weight', '.scale'), sc):
            if ck.has(cand):
                d, s = ck.shape(hf), ck.shape(cand)
                gran = '%dx%d' % (d[0] // s[0], d[1] // s[1])
                break
    return dt, gran


PENDING = {'attn_compressor_ape.weight', 'attn_compressor_gate.weight',
           'attn_compressor_kv.weight', 'ffn_gate_inp.weight', 'indexer.proj.weight',
           'indexer_compressor_ape.weight', 'indexer_compressor_gate.weight',
           'indexer_compressor_kv.weight', 'hc_attn_fn.weight', 'hc_ffn_fn.weight',
           'output_hc_fn.weight'}


def verdict(ours, src):
    base = REPACK.get(ours, ours)
    tag = ' (repack)' if base != ours else ''
    if src == '-':
        return 'no source counterpart'
    if base == src:
        return 'AT source' + tag
    order = {'IQ2_XXS': 0, 'MXFP4': 1, 'FP8_E4M3': 2, 'F16': 3, 'BF16': 3, 'F32': 4}
    a, b = order.get(base, 9), order.get(src, 9)
    if base == 'F16' and src == 'BF16':
        return 'WRONG 16-bit: flushes small values'
    if base == 'F16' and src == 'F32':
        return 'NARROWED badly: deletes ~11%'
    if base == 'BF16' and src == 'F32':
        return 'narrowed, full range kept'
    if a < b:
        return 'BELOW source'
    if a > b:
        return 'ABOVE source (wider than needed)'
    return f'{base} vs {src}'


rows, draft = OrderedDict(), 0
for t in tensors:
    if t['name'].startswith('dspark.'):
        draft += 1
        continue
    ours = TY.get(t['type'], str(t['type']))
    src, gran = src_of(t['name'])
    rows.setdefault((suffix_of(t['name']), ours, src, gran), [0])[0] += 1

W = '%-31s %-14s %-9s %-6s %-34s %4s'
print(W % ('ds4 tensor family', 'WE SHIP', 'SOURCE', 'blk', 'relationship to source', 'n'))
print('-' * 104)
for (fam, ours, src, gran), (n,) in sorted(rows.items()):
    v = verdict(ours, src)
    if fam in PENDING:
        v = verdict('BF16', src) + '  [after splice]'
    print(W % (fam[:31], ours, src, gran or '-', v, n))
print('\n(%d drafter dspark.* tensors omitted -- separate model, no source counterpart)' % draft)
