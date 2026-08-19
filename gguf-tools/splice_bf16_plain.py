#!/usr/bin/env python3
"""Move build_main_template.BF16_GROUP from F16 to BF16 in an existing artifact.

WHY THIS IS NOT A RE-QUANTIZE
-----------------------------
F16 and BF16 are BOTH 2 bytes per element, so every affected tensor's byte size
is IDENTICAL and no data offset in the file moves.  That turns what sounds like
a rebuild into a copy plus 337 small writes -- 780 MB touched out of a 92 GB
artifact (0.84%).  This copies the file and rewrites only

  (a) the type field of the targeted tensors in the header (F16 1 -> BF16 30), and
  (b) those tensors' data regions, in place, at their existing offsets.

Everything else -- experts, attention, norms, KV -- is byte-copied untouched.
Same in-place retype-and-rewrite technique as repack_iq2_mmq.py, which relies
on the same size-identity.

WHY THE DATA COMES FROM SOURCE, NOT FROM THE EXISTING F16
---------------------------------------------------------
It would be far cheaper to widen the F16 already in the file (f16 -> f32 ->
bf16) and never open the checkpoint.  That would be WRONG, and silently so.
The reason these tensors are moving is that f16 storage already destroyed them
at the SMALL end: f16 flushes everything under 5.96e-8 to exactly zero, and
these tensors run down to 2.7e-15 -- 11.07% of layers.0.hc_attn_fn's nonzero
weights are gone, not approximated.  Widening a zero gives you a wider zero.
Only re-reading the checkpoint recovers the values, which is the whole point.

TWO SOURCE DTYPES, TWO DIFFERENT CLAIMS
---------------------------------------
The group is not uniform, so do not describe the output as one thing:

  188 tensors are BF16 upstream -> straight memcpy, and the result is a
      LOSSLESS copy of what the checkpoint holds.
  149 tensors are F32 upstream -> truncated to bf16 with round-to-nearest-even.
      That is a deliberate NARROWING, chosen because f16 is worse for this data
      (it deletes 11% of the small weights; bf16 keeps f32's range and pays in
      mantissa bits these values do not use).

Either way the source is row-major and so is ds4 -- ne_reversed() flips the
DIMENSION LIST for GGUF's convention, it does not transpose data -- so no
re-layout is involved.  We verify every tensor against source after writing
rather than trusting any of the above.

USAGE
    splice_bf16_plain.py IN.gguf OUT.gguf --hf /mnt/models/dsv4-flash-0731
                         [--match SUBSTR] [--dry-run]

Run it with the quant venv (/home/claude/.venvs/pulsar-quant/bin/python): the
F32 half needs numpy.  The BF16-only half does not.

NOTE the engine must accept PULSAR_TENSOR_BF16 in tensor_expect_plain_or_mxfp8
(weights.cpp) before the spliced artifact will load; it dies on anything but
F16/FP8_E4M3 today.  Deliberately -- see the comment there.
"""
import argparse
import os
import shutil
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_main_template import BF16_GROUP, HFCheckpoint, LAYER_MAP, TOP_MAP  # noqa: E402
from gguf_hdr import scan  # noqa: E402

F16 = 1
BF16 = 30


def bf16_bytes(ckpt, hf, src_dtype):
    """The tensor as BF16 bytes, ready to write into the artifact.

    BF16 source is a straight memcpy -- the bytes are already what we want.
    F32 source is truncated to the top 16 bits with round-to-nearest-even,
    which is exactly what bf16 IS, so no range is lost and only mantissa bits
    below bf16's 7 are dropped.
    """
    buf = ckpt.raw(hf)
    if src_dtype == 'BF16':
        return buf

    import numpy as np  # only the F32 path needs it
    u = np.frombuffer(buf, dtype='<u4')
    # This MUST agree bit-for-bit with ds4q_f32_to_bf16_row (quants_common.c),
    # or a spliced artifact stops being equivalent to a rebuilt one and the two
    # paths quietly diverge. Both branches below are transcribed from it.
    rounded = ((u.astype('<u8') + (0x7FFF + ((u >> 16) & 1))) >> 16).astype('<u2')
    is_nan = (u & 0x7FFFFFFF) > 0x7F800000
    if is_nan.any():
        # The C quiets NaN rather than rounding it; match that, but say so --
        # NaN in a weight tensor is a checkpoint problem, not a splice problem.
        print(f'   !! {hf}: {int(is_nan.sum())} NaNs in source, quieted (matches quantizer)')
        rounded = np.where(is_nan, ((u >> 16) | 64).astype('<u2'), rounded)
    return rounded.astype('<u2').tobytes()


def hf_name_for(ds4_name):
    """ds4 tensor name -> upstream HF tensor name, or None if not mapped."""
    if ds4_name.startswith('blk.'):
        _, layer, suffix = ds4_name.split('.', 2)
        if suffix not in LAYER_MAP:
            return None
        return f'layers.{layer}.{LAYER_MAP[suffix]}'
    return TOP_MAP.get(ds4_name)


def suffix_of(ds4_name):
    return ds4_name.split('.', 2)[-1] if ds4_name.startswith('blk.') else ds4_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--hf', required=True, help='upstream HF checkpoint dir')
    ap.add_argument('--match', help='only splice tensors whose name contains this')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    ckpt = HFCheckpoint(args.hf)
    tensors, alignment, data_start = scan(args.src)

    todo, skipped = [], []
    for t in tensors:
        if suffix_of(t['name']) not in BF16_GROUP:
            continue
        if args.match and args.match not in t['name']:
            continue
        if t['type'] == BF16:
            skipped.append((t['name'], 'already BF16'))
            continue
        if t['type'] != F16:
            skipped.append((t['name'], f"type {t['type']}, not F16"))
            continue
        hf = hf_name_for(t['name'])
        if hf is None or not ckpt.has(hf):
            skipped.append((t['name'], f'no source tensor ({hf})'))
            continue
        dt = ckpt.dtype(hf)
        if dt not in ('BF16', 'F32'):
            skipped.append((t['name'], f'source dtype {dt}, expected BF16 or F32'))
            continue
        want = t['ne'] * 2
        todo.append(dict(t, hf=hf, nbytes=want, src_dtype=dt))

    if skipped:
        print(f'-- skipped {len(skipped)}:')
        for name, why in skipped[:12]:
            print(f'     {name}: {why}')
        if len(skipped) > 12:
            print(f'     ... and {len(skipped) - 12} more')

    n_copy = sum(1 for t in todo if t['src_dtype'] == 'BF16')
    total = sum(t['nbytes'] for t in todo)
    print(f'-- splicing {len(todo)} tensors, {total / 1e6:.1f} MB '
          f'({100.0 * total / max(os.path.getsize(args.src), 1):.3f}% of the artifact)')
    print(f'     {n_copy} from a BF16 source (lossless copy), '
          f'{len(todo) - n_copy} narrowed from F32')
    for t in todo[:6]:
        print(f'     {t["name"]:40s} <- {t["hf"]:40s} {t["src_dtype"]:>5s} {t["nbytes"]:>9d} B')
    if len(todo) > 6:
        print(f'     ... and {len(todo) - 6} more')

    if args.dry_run:
        print('-- dry run, nothing written')
        return
    if not todo:
        print('-- nothing to do')
        return

    print(f'-- copying {args.src} -> {args.dst}')
    shutil.copyfile(args.src, args.dst)

    with open(args.dst, 'r+b') as f:
        for t in todo:
            buf = bf16_bytes(ckpt, t['hf'], t['src_dtype'])
            # The size identity is the whole basis for splicing in place. If it
            # ever fails we would corrupt every tensor after this one, so treat
            # it as fatal rather than clamping or padding.
            if len(buf) != t['nbytes']:
                raise SystemExit(
                    f'{t["name"]}: source is {len(buf)} B but the artifact '
                    f'reserves {t["nbytes"]} B -- shapes disagree, refusing to splice')
            f.seek(data_start + t['offset'])
            f.write(buf)
            f.seek(t['type_pos'])
            f.write(struct.pack('<I', BF16))
        f.flush()
        os.fsync(f.fileno())

    # Read back independently: confirm every target now declares BF16 and holds
    # bytes identical to source. Cheap next to the copy, and it catches an
    # offset mistake that would otherwise surface as garbage weights at serve.
    tensors2, _, data_start2 = scan(args.dst)
    by_name = {t['name']: t for t in tensors2}
    bad = 0
    with open(args.dst, 'rb') as f:
        for t in todo:
            v = by_name[t['name']]
            if v['type'] != BF16:
                print(f'   !! {t["name"]}: type is {v["type"]}, not BF16')
                bad += 1
                continue
            f.seek(data_start2 + v['offset'])
            if f.read(t['nbytes']) != bf16_bytes(ckpt, t['hf'], t['src_dtype']):
                print(f'   !! {t["name"]}: data does not match source')
                bad += 1
    if bad:
        raise SystemExit(f'-- VERIFY FAILED on {bad} tensors')
    print(f'-- verified {len(todo)} tensors: type BF16, bytes identical to source')
    print(f'-- wrote {args.dst}')


if __name__ == '__main__':
    main()
