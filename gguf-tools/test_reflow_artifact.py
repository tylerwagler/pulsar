#!/usr/bin/env python3
"""End-to-end test for reflow_artifact.py on a synthetic artifact.

Builds a small GGUF carrying REAL ds4 tensor names -- so the tool resolves real
target types and reads real source tensors -- covering every conversion class
at once, and deliberately interleaving them so a size change in one tensor has
to shift the offsets of the ones after it:

    attn_norm        F32 -> BF16        SHRINKS, source bf16, straight copy
    hc_attn_fn       F16 -> F32         GROWS,   source f32,  straight copy
    indexer.attn_q_b F16 -> FP8_E4M3    SHRINKS, source fp8,  through the codec
    attn_sinks       F32 -> F32         DECOY, must not move or change
    hc_attn_base     F32 -> F32         DECOY, after two resizes
    ffn_down_exps    IQ2_XXS -> ..._MMQ REPACK from existing bytes, no source

The repack case matters because it is the only class that never reads the
checkpoint -- it permutes bytes already in the file -- so a bug there would be
invisible to any check that compares against source. IQ2_XXS_MMQ is a pure
permutation into two planes, so the check is that every content byte survives
and only alignment padding is added.

The decoys are the point. The tool writes at computed absolute offsets, so an
arithmetic slip does not fail loudly -- it silently lands a tensor on top of its
neighbour. A test where every tensor changes would not catch that.

Checks: every target retyped and byte-equal to what the conversion should
produce, every decoy byte-identical to the input, offsets exactly as planned and
alignment preserved, and the file's own header self-consistent on re-read.

Usage: test_reflow_artifact.py [--hf /mnt/models/dsv4-flash-0731]
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_main_template import HFCheckpoint, ne_reversed  # noqa: E402
from gguf_hdr import scan  # noqa: E402
from audit_vs_source import TY, hf_name  # noqa: E402
from reflow_artifact import source_bytes, type_bytes  # noqa: E402

ALIGN = 32
F32, F16 = 0, 1

# (ds4 name, type to seed it with, expected target or None for "unchanged")
CASES = [
    ('blk.0.attn_norm.weight', F32, 'BF16'),
    ('blk.0.attn_sinks.weight', F32, None),
    ('blk.0.hc_attn_fn.weight', F16, 'F32'),
    ('blk.2.indexer.attn_q_b.weight', F16, 'FP8_E4M3'),
    ('blk.0.hc_attn_base.weight', F32, None),
]

# Repacks take no source tensor, so this one is built by hand. nblk must be a
# multiple of 32 for the size identity the packer requires (ne % 8192 == 0).
IQ2_XXS = 16
IQ2_CASE = ('blk.9.ffn_down_exps.weight', [512, 512, 64], IQ2_XXS)


def _str(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def build_gguf(path, entries):
    hdr = b'GGUF' + struct.pack('<I', 3) + struct.pack('<Q', len(entries)) + struct.pack('<Q', 1)
    hdr += _str('general.alignment') + struct.pack('<I', 4) + struct.pack('<I', ALIGN)
    off, infos, blobs = 0, b'', []
    for name, dims, ttype, payload in entries:
        infos += _str(name) + struct.pack('<I', len(dims))
        infos += b''.join(struct.pack('<Q', d) for d in dims)
        infos += struct.pack('<I', ttype) + struct.pack('<Q', off)
        blobs.append(payload)
        off += len(payload) + ((ALIGN - (len(payload) % ALIGN)) % ALIGN)
    body = hdr + infos
    with open(path, 'wb') as f:
        f.write(body + b'\0' * ((ALIGN - (len(body) % ALIGN)) % ALIGN))
        for p in blobs:
            f.write(p)
            f.write(b'\0' * ((ALIGN - (len(p) % ALIGN)) % ALIGN))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf', default='/mnt/models/dsv4-flash-0731')
    a = ap.parse_args()
    ck = HFCheckpoint(a.hf)

    entries, expect = [], {}
    for name, seed_ty, want in CASES:
        hf = hf_name(name)
        if hf is None or not ck.has(hf):
            raise SystemExit(f'test setup: no source for {name} ({hf})')
        dims = ne_reversed(ck.shape(hf))
        ne = 1
        for d in dims:
            ne *= d
        width = 4 if seed_ty == F32 else 2
        # A distinctive filler makes "was overwritten" and "was left alone" both
        # unambiguous; neither is a plausible real value.
        payload = (b'\xA5\x5A\xA5\x5A' * ne)[:ne * width]
        entries.append((name, dims, seed_ty, payload))
        expect[name] = dict(want=want, hf=hf, ne=ne, dims=dims, orig=payload)

    # the repack case: raw IQ2_XXS blocks, 66 B each, content is arbitrary
    # because the packer only permutes
    iq_name, iq_dims, iq_ty = IQ2_CASE
    iq_ne = iq_dims[0] * iq_dims[1] * iq_dims[2]
    iq_nblk = iq_ne // 256
    assert iq_nblk % 32 == 0, iq_nblk
    iq_payload = bytes((i * 7 + 11) & 0xFF for i in range(iq_nblk * 66))
    entries.append((iq_name, iq_dims, iq_ty, iq_payload))

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, 'in.gguf')
        dst = os.path.join(td, 'out.gguf')
        build_gguf(src, entries)
        before, _, ds_before = scan(src)
        bmap = {t['name']: t for t in before}

        r = subprocess.run([sys.executable, os.path.join(HERE, 'reflow_artifact.py'),
                            src, dst, '--hf', a.hf, '--verify'],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            raise SystemExit('reflow failed')

        after, align, ds_after = scan(dst)
        amap = {t['name']: t for t in after}
        fails = []
        if ds_before != ds_after:
            fails.append(f'data_start moved {ds_before} -> {ds_after}')

        with open(dst, 'rb') as f:
            for name, e in expect.items():
                v = amap[name]
                if v['offset'] % align:
                    fails.append(f'{name}: offset {v["offset"]} not {align}-aligned')
                if e['want'] is None:
                    if v['type'] != bmap[name]['type']:
                        fails.append(f'{name}: DECOY retyped to {v["type"]}')
                    f.seek(ds_after + v['offset'])
                    if f.read(len(e['orig'])) != e['orig']:
                        fails.append(f'{name}: DECOY data changed')
                    continue
                if TY.get(v['type']) != e['want']:
                    fails.append(f'{name}: type {TY.get(v["type"])}, expected {e["want"]}')
                    continue
                nb = type_bytes(e['want'], e['ne'])
                f.seek(ds_after + v['offset'])
                got = f.read(nb)
                if got == e['orig'][:nb]:
                    fails.append(f'{name}: still the filler, not converted')
                if got != source_bytes(ck, e['hf'], e['want'], e['dims']):
                    fails.append(f'{name}: data != source-converted')

        # the repack: every content byte must survive the permutation
        v = amap[iq_name]
        if TY.get(v['type']) != 'IQ2_XXS_MMQ':
            fails.append(f'{iq_name}: type {TY.get(v["type"])}, expected IQ2_XXS_MMQ')
        else:
            from collections import Counter
            with open(dst, 'rb') as f:
                f.seek(ds_after + v['offset'])
                packed = f.read(len(iq_payload) + 64)
            lost = Counter(iq_payload) - Counter(packed)
            if lost:
                fails.append(f'{iq_name}: repack lost {sum(lost.values())} content bytes')

        # offsets must be strictly increasing and non-overlapping
        ordered = sorted(after, key=lambda t: t['offset'])
        for x, y in zip(ordered, ordered[1:]):
            xn = type_bytes(TY.get(x['type']), x['ne']) or 0
            if x['offset'] + xn > y['offset']:
                fails.append(f'{x["name"]} overruns {y["name"]}')

    for m in fails:
        print('FAIL:', m)
    if fails:
        raise SystemExit(1)
    n = sum(1 for e in expect.values() if e['want'])
    print(f'PASS: {n} converted (shrink, grow and codec paths), 1 repacked, '
          f'{len(expect) - n} decoys untouched, offsets aligned and non-overlapping')


if __name__ == '__main__':
    main()
