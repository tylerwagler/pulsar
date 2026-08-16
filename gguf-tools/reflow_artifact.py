#!/usr/bin/env python3
"""Rewrite an artifact to the current type policy without re-quantizing it.

WHY THIS IS NOT A RE-QUANTIZE, AND NOT A SPLICE EITHER
------------------------------------------------------
The in-place splice (splice_bf16_plain.py) works only when a tensor's byte size
is unchanged -- F16 -> BF16, both 2 bytes. The current policy also moves types
that change WIDTH: f32 norms to bf16, f16 hc_*_fn to f32, the fp8 head to bf16,
the f16 indexer q_b to MXFP8. Once any tensor resizes, every later data offset
moves and the file has to be rewritten.

Rewritten, not recomputed. NOTHING here needs quantization work:

    bf16 target, bf16 source     straight copy of the source bytes
    f32  target, f32  source     straight copy
    bf16 target, f32  source     truncate to bf16, round-to-nearest-even,
                                 bit-identical to ds4q_f32_to_bf16_row
    fp8  target, f8_e4m3 source  dequantize and re-block through the shipping
                                 codec. MEASURED exact on the real tensors --
                                 8388364/8388608 values bit-identical, every
                                 difference a -0 becoming +0 -- because per-32
                                 is strictly finer than the source's 128x128,
                                 so each group's amax can only be <= the
                                 block's and every mantissa lands on the same
                                 e4m3 code.

So the experts -- the expensive part, and the part we are deliberately NOT
changing -- are byte-copied straight across. That is the main reason to prefer
this over re-running the quantizer: it keeps the 91 IQ2 stacks and the
CUTLASS_MXFP4 tensors provably identical to what serves today, so if anything
moves afterwards it cannot be the experts. A requant would redo them from the
imatrix and survivor set and put that beyond proof.

A convenient property of GGUF here: tensor NAMES and DIMS do not change, and
type and offset are fixed-width fields, so the header's SIZE is unchanged and
data_start does not move. Only the offsets inside the data region shift.

USAGE
    reflow_artifact.py IN.gguf OUT.gguf --hf /mnt/models/dsv4-flash-0731
                       [--dry-run] [--verify] [--limit N]

Needs the quant venv (numpy) and builds a small shared object from the codec
sources on first use so the fp8 path is the SHIPPING code, not a reimplementation.
"""
import argparse
import ctypes
import hashlib
import os
import re
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_main_template import HFCheckpoint, suffix_type  # noqa: E402
from gguf_hdr import scan  # noqa: E402
from audit_vs_source import (BYNAME, TY, hf_name, load_flags,  # noqa: E402
                             load_format_map)

TYPE_ID = {v: k for k, v in TY.items()}
ALIGN_DEFAULT = 32
CODEC_SRC = ['quants_fp.c', 'quants_common.c', 'quants_kquants.c']


# --------------------------------------------------------------------------
# sizes
# --------------------------------------------------------------------------
def type_bytes(tname, ne):
    """On-disk bytes for `ne` elements of a ds4 type, or None if we do not
    produce that type here (unchanged tensors are copied, never sized)."""
    if tname in ('F32',):
        return ne * 4
    if tname in ('F16', 'BF16'):
        return ne * 2
    if tname == 'FP8_E4M3':
        if ne % 32:
            raise SystemExit(f'MXFP8 needs a multiple of 32 elements, got {ne}')
        return (ne // 32) * 33
    return None


# --------------------------------------------------------------------------
# the shipping fp8 codec, via ctypes -- never a reimplementation
# --------------------------------------------------------------------------
_LIB = None


def codec():
    global _LIB
    if _LIB is not None:
        return _LIB
    so = os.path.join(HERE, 'libds4q_reflow.so')
    srcs = [os.path.join(HERE, s) for s in CODEC_SRC]
    if not os.path.exists(so) or any(os.path.getmtime(s) > os.path.getmtime(so) for s in srcs):
        cmd = ['gcc', '-O2', '-fPIC', '-shared', '-I', HERE, '-o', so] + srcs + ['-lm']
        print('-- building the codec shared object:\n   %s' % ' '.join(cmd))
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(so)
    lib.ds4q_quantize_fp8_e4m3.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    lib.ds4q_quantize_fp8_e4m3.restype = ctypes.c_size_t
    lib.ds4q_pack_iq2_xxs_mmq.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    lib.ds4q_iq2_xxs_mmq_bytes.argtypes = [ctypes.c_int64]
    lib.ds4q_iq2_xxs_mmq_bytes.restype = ctypes.c_size_t
    lib.ds4q_pack_mxfp8_lt.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_int64, ctypes.c_int64]
    lib.ds4q_mxfp8_lt_bytes.argtypes = [ctypes.c_int64, ctypes.c_int64]
    lib.ds4q_mxfp8_lt_bytes.restype = ctypes.c_size_t
    _LIB = lib
    return lib


_E4M3 = None


def e4m3_table():
    global _E4M3
    import numpy as np
    if _E4M3 is None:
        t = np.zeros(256, dtype=np.float32)
        for b in range(256):
            s = -1.0 if b & 0x80 else 1.0
            e, m = (b >> 3) & 0xF, b & 0x7
            if e == 0xF and m == 0x7:
                t[b] = np.nan
            elif e == 0:
                t[b] = s * (m / 8.0) * 2.0 ** -6
            else:
                t[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
        _E4M3 = t
    return _E4M3


# --------------------------------------------------------------------------
# LAYOUT REPACKS: a permutation of the tensor's EXISTING bytes into the
# pre-swizzled twin the engine wants. No checkpoint read, no arithmetic, no
# value change -- MXFP8_LT is FP8_E4M3 de-interleaved, IQ2_XXS_MMQ is IQ2_XXS
# split into two planes. Kept distinct from the source-fed conversions below
# because these cannot lose anything even in principle.
# --------------------------------------------------------------------------
REPACK_OF = {('IQ2_XXS', 'IQ2_XXS_MMQ'), ('FP8_E4M3', 'MXFP8_LT')}


def repack_size(lib, cur, want, dims, ne):
    if want == 'IQ2_XXS_MMQ':
        return lib.ds4q_iq2_xxs_mmq_bytes(ne // 256)
    ncols = int(dims[0])
    return lib.ds4q_mxfp8_lt_bytes(ne // ncols, ncols)


def repack_bytes(cur, want, raw, dims, ne):
    import numpy as np
    lib = codec()
    src = np.frombuffer(raw, dtype=np.uint8)
    out = np.empty(repack_size(lib, cur, want, dims, ne), dtype=np.uint8)
    if want == 'IQ2_XXS_MMQ':
        lib.ds4q_pack_iq2_xxs_mmq(src.ctypes.data, out.ctypes.data, ne // 256)
    else:
        # ds4 dims are [in, out, ...]; the packer takes [nrows=out, ncols=in].
        ncols = int(dims[0])
        lib.ds4q_pack_mxfp8_lt(src.ctypes.data, out.ctypes.data, ne // ncols, ncols)
    return out.tobytes()


# --------------------------------------------------------------------------
# producing one tensor's new bytes from the checkpoint
# --------------------------------------------------------------------------
def source_bytes(ck, hf, target, dims):
    import numpy as np
    src = ck.dtype(hf)
    raw = ck.raw(hf)

    if target == 'BF16' and src == 'BF16':
        return raw
    if target == 'F32' and src == 'F32':
        return raw
    if target == 'F16' and src == 'F16':
        return raw

    if target == 'BF16' and src == 'F32':
        # identical to ds4q_f32_to_bf16_row (verified over 2.5M bit patterns)
        u = np.frombuffer(raw, dtype='<u4')
        out = ((u.astype('<u8') + (0x7FFF + ((u >> 16) & 1))) >> 16).astype('<u2')
        nan = (u & 0x7FFFFFFF) > 0x7F800000
        if nan.any():
            print('   !! %s: %d NaNs, quieted (matches the quantizer)' % (hf, int(nan.sum())))
            out = np.where(nan, ((u >> 16) | 64).astype('<u2'), out).astype('<u2')
        return out.tobytes()

    if target == 'FP8_E4M3' and src == 'F8_E4M3':
        scale_name = hf.rsplit('.', 1)[0] + '.scale'
        if not ck.has(scale_name):
            raise SystemExit(f'{hf}: fp8 source with no companion .scale')
        shp = ck.shape(hf)
        d = np.frombuffer(raw, dtype=np.uint8).reshape(shp)
        s = np.frombuffer(ck.raw(scale_name), dtype=np.uint8).reshape(ck.shape(scale_name))
        rb, cb = shp[0] // s.shape[0], shp[1] // s.shape[1]
        val = e4m3_table()[d] * np.exp2(s.astype(np.int32) - 127).repeat(rb, 0).repeat(cb, 1)
        val = np.ascontiguousarray(val.astype('<f4'))
        nrows, ncols = val.shape
        out = np.empty((nrows * ncols // 32) * 33, dtype=np.uint8)
        codec().ds4q_quantize_fp8_e4m3(val.ctypes.data, out.ctypes.data, 0, nrows, ncols)
        return out.tobytes()

    raise SystemExit(f'{hf}: no rule for source {src} -> target {target}')


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--hf', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--verify', action='store_true',
                    help='re-read the output and check every tensor')
    ap.add_argument('--limit', type=int, help='stop after N tensors (smoke test)')
    ap.add_argument('--only', action='append', default=[], metavar='FAMILY',
                    help='change ONLY these families; hold every other family at '
                         'its current type. The complement of --pin, and the '
                         'right shape for a bisect: the output differs from the '
                         'input in exactly the named group, so a gate verdict '
                         'attributes to that group and nothing else.')
    ap.add_argument('--pin', action='append', default=[], metavar='FAMILY',
                    help='hold this family at its CURRENT type instead of the '
                         'policy target. Repeatable. Used to build one-variable '
                         'variants for bisecting a gate failure: pinning a '
                         'family makes it byte-copy from the input, so the '
                         'variant differs from the full reflow in exactly that '
                         'family and nothing else.')
    a = ap.parse_args()

    ck = HFCheckpoint(a.hf)
    flags, fmap = load_flags(), load_format_map()
    tensors, alignment, data_start = scan(a.src)
    tensors.sort(key=lambda t: t['offset'])
    src_size = os.path.getsize(a.src)

    # Input span per tensor: offsets are ascending and each is padded up to
    # `alignment`, so [offset, next_offset) is the tensor plus its padding.
    # Copying the span verbatim reproduces both.
    for i, t in enumerate(tensors):
        end = tensors[i + 1]['offset'] if i + 1 < len(tensors) else (src_size - data_start)
        t['in_span'] = end - t['offset']

    plan, n_changed, n_copied, n_repack = [], 0, 0, 0
    for t in tensors:
        cur = TY.get(t['type'], str(t['type']))
        want = None
        nm = t['name']
        if nm.startswith('dspark.'):
            want = flags.get(nm)
        elif nm in fmap:
            want = fmap[nm]
        else:
            try:
                want = TY.get(suffix_type(nm, len(t['dims'])))
            except Exception:
                want = None
        fam_now = nm.split('.', 2)[-1] if nm.startswith('blk.') else nm
        if nm.startswith('dspark.'):
            pp = nm.split('.')
            fam_now = 'dspark.' + ('.'.join(pp[2:]) if pp[1].isdigit() else '.'.join(pp[1:]))
        if a.only and not (fam_now in a.only or nm in a.only):
            want = cur          # not under test: hold it at the input's type
        if fam_now in a.pin or nm in a.pin:
            want = cur          # hold it where it is; falls into byte-copy below
        if want is None or want == cur:
            plan.append((t, None, t['in_span']))
            n_copied += 1
            continue
        if (cur, want) in REPACK_OF:
            plan.append((t, want, repack_size(codec(), cur, want, t['dims'], t['ne'])))
            n_repack += 1
            continue
        nb = type_bytes(want, t['ne'])
        if nb is None:
            raise SystemExit(f'{nm}: cannot produce type {want} here')
        hf = hf_name(nm)
        if hf is None or not ck.has(hf):
            raise SystemExit(f'{nm}: wants {want} but has no source tensor ({hf})')
        plan.append((t, want, nb))
        n_changed += 1

    def pad(n):
        return (alignment - (n % alignment)) % alignment

    new_off, total = [], 0
    for _, want, nb in plan:
        new_off.append(total)
        total += nb + pad(nb)

    if a.only:
        print('-- ONLY these families change: %s' % ', '.join(a.only))
    if a.pin:
        print('-- PINNED (held at their current type): %s' % ', '.join(a.pin))
    print('-- %d tensors: %d byte-copied, %d repacked in place, '
          '%d regenerated from source'
          % (len(plan), n_copied, n_repack, n_changed))
    print('-- data region %.2f GB -> %.2f GB   file %.2f GB -> %.2f GB'
          % ((src_size - data_start) / 1e9, total / 1e9,
             src_size / 1e9, (data_start + total) / 1e9))
    # Keep these apart. A repack is a permutation of bytes we already have and
    # cannot change a value; only the source-fed set involves reading the
    # checkpoint at all. Lumping them together reports the 91 IQ2 stacks as
    # "regenerated", which is exactly the impression this tool exists to avoid.
    rp = sum(nb for t, want, nb in plan
             if want and (TY.get(t['type'], str(t['type'])), want) in REPACK_OF)
    regen = sum(nb for t, want, nb in plan
                if want and (TY.get(t['type'], str(t['type'])), want) not in REPACK_OF)
    print('-- repacked from existing bytes: %8.1f MB (%.2f%% of the file)'
          % (rp / 1e6, 100.0 * rp / src_size))
    print('-- read from the checkpoint:     %8.1f MB (%.2f%% of the file)'
          % (regen / 1e6, 100.0 * regen / src_size))
    if a.dry_run:
        for t, want, nb in plan[:10]:
            if want:
                print('     %-44s %-9s -> %-9s %10d B' % (
                    t['name'][:44], TY.get(t['type'], t['type']), want, nb))
        print('-- dry run, nothing written')
        return

    print('-- writing %s' % a.dst)
    with open(a.src, 'rb') as fin, open(a.dst, 'wb') as fout:
        fin.seek(0)
        header = fin.read(data_start)
        fout.write(header)
        done = 0
        for (t, want, nb), off in zip(plan, new_off):
            fout.seek(data_start + off)
            if want is None:
                fin.seek(data_start + t['offset'])
                left = nb
                while left:
                    chunk = fin.read(min(left, 1 << 24))
                    if not chunk:
                        raise SystemExit(f"{t['name']}: short read from input")
                    fout.write(chunk)
                    left -= len(chunk)
            else:
                cur = TY.get(t['type'], str(t['type']))
                if (cur, want) in REPACK_OF:
                    fin.seek(data_start + t['offset'])
                    buf = repack_bytes(cur, want, fin.read(t['in_span']), t['dims'], t['ne'])
                else:
                    buf = source_bytes(ck, hf_name(t['name']), want, t['dims'])
                if len(buf) != nb:
                    raise SystemExit(f"{t['name']}: produced {len(buf)} B, planned {nb}")
                fout.write(buf)
                fout.write(b'\0' * pad(nb))
            done += 1
            if a.limit and done >= a.limit:
                print('-- stopped after %d tensors (--limit)' % done)
                break
            if done % 200 == 0:
                print('   %d/%d' % (done, len(plan)))

        # patch the header in place: same field widths, so nothing reflows
        for (t, want, _), off in zip(plan, new_off):
            if want is not None:
                fout.seek(t['type_pos'])
                fout.write(struct.pack('<I', TYPE_ID[want]))
            fout.seek(t['type_pos'] + 4)
            fout.write(struct.pack('<Q', off))
        fout.flush()
        os.fsync(fout.fileno())

    print('-- wrote %s (%.2f GB)' % (a.dst, os.path.getsize(a.dst) / 1e9))

    if a.verify:
        verify(a, plan, new_off, data_start, ck)


def expected_bytes(fin, data_start, ck, t, want):
    cur = TY.get(t['type'], str(t['type']))
    if (cur, want) in REPACK_OF:
        fin.seek(data_start + t['offset'])
        return repack_bytes(cur, want, fin.read(t['in_span']), t['dims'], t['ne'])
    return source_bytes(ck, hf_name(t['name']), want, t['dims'])


def verify(a, plan, new_off, data_start, ck):
    """Re-read the output independently. The unchanged tensors are the point:
    the whole argument for reflowing over re-quantizing is that they come
    through untouched, so that is what gets checked, not assumed."""
    out_t, _, out_ds = scan(a.dst)
    by = {t['name']: t for t in out_t}
    bad = 0
    with open(a.src, 'rb') as fin, open(a.dst, 'rb') as fout:
        for (t, want, nb), off in zip(plan, new_off):
            v = by[t['name']]
            if v['offset'] != off:
                print('   !! %s: offset %d, expected %d' % (t['name'], v['offset'], off))
                bad += 1
                continue
            fout.seek(out_ds + off)
            got = fout.read(nb)
            if want is None:
                fin.seek(data_start + t['offset'])
                if got != fin.read(nb):
                    print('   !! %s: COPIED tensor differs from the input' % t['name'])
                    bad += 1
            else:
                if TY.get(v['type']) != want:
                    print('   !! %s: type %s, expected %s' % (t['name'], v['type'], want))
                    bad += 1
                elif hashlib.sha256(got).digest() != hashlib.sha256(
                        expected_bytes(fin, data_start, ck, t, want)).digest():
                    print('   !! %s: data does not match source' % t['name'])
                    bad += 1
    if bad:
        raise SystemExit('-- VERIFY FAILED on %d tensors' % bad)
    print('-- verified %d tensors: copies byte-identical to input, '
          'regenerated ones match source, all offsets as planned' % len(plan))


if __name__ == '__main__':
    main()
