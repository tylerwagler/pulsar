#!/usr/bin/env python3
"""verify_model_mmq.py -- validate a repack_iq2_mmq.py output against its source.

Three levels, all byte-exact:

  1. HEADER.  Identical file size, tensor count, names, dims and DATA OFFSETS
     (if an offset moved the header was reflowed and every reader is wrong);
     every type unchanged EXCEPT the converted ones, which must go 16 -> 43 and
     nothing else; each converted tensor's byte size unchanged.

  2. EVERYTHING ELSE IS UNTOUCHED.  Streams both whole files and memcmps them
     while skipping only (a) the 4-byte type field of each converted tensor and
     (b) each converted tensor's data span.  This is stronger than a per-tensor
     walk: it also covers metadata, padding and any region no tensor claims.

  3. PAYLOAD.  Emits a manifest of the converted spans for verify_model_mmq
     (CUDA), which device-derepacks each one and compares it to the source raw
     bytes -- proving the shipped artifact inverts to the original model.

Usage: verify_model_mmq.py SRC.gguf DST.gguf MANIFEST
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_iq2_mmq import scan, aligned_bytes, IQ2_XXS, IQ2_XXS_MMQ, QK_K, BLK_BYTES

src, dst, manifest = sys.argv[1], sys.argv[2], sys.argv[3]

s_sz, d_sz = os.path.getsize(src), os.path.getsize(dst)
print("src size = %d" % s_sz)
print("dst size = %d" % d_sz)
if s_sz != d_sz:
    raise SystemExit("FAIL: file size changed by %d bytes" % (d_sz - s_sz))
print("size: OK (identical, growth 0)")

st, sal, sds = scan(src)
dt, dal, dds = scan(dst)
if len(st) != len(dt):
    raise SystemExit("FAIL: tensor count %d vs %d" % (len(st), len(dt)))
if sds != dds:
    raise SystemExit("FAIL: data_start moved %d -> %d" % (sds, dds))
print("tensors=%d data_start=%d alignment=%d (both files agree)" % (len(st), sds, sal))

skip = []        # (start, end) byte ranges excluded from the whole-file diff
spans = []       # converted payload spans, for the CUDA verifier
unchanged = 0
for a, b in zip(st, dt):
    if a["name"] != b["name"]:
        raise SystemExit("FAIL: name mismatch %s vs %s" % (a["name"], b["name"]))
    if a["dims"] != b["dims"]:
        raise SystemExit("FAIL: %s dims %s vs %s" % (a["name"], a["dims"], b["dims"]))
    if a["offset"] != b["offset"]:
        raise SystemExit("FAIL: %s offset %d vs %d" % (a["name"], a["offset"], b["offset"]))
    if a["type"] == b["type"]:
        unchanged += 1
        continue
    if not (a["type"] == IQ2_XXS and b["type"] == IQ2_XXS_MMQ):
        raise SystemExit("FAIL: %s type %d -> %d (only 16 -> 43 is allowed)"
                         % (a["name"], a["type"], b["type"]))
    nblk = a["ne"] // QK_K
    nbytes = nblk * BLK_BYTES
    if aligned_bytes(nblk) != nbytes:
        raise SystemExit("FAIL: %s aligned size %d != raw %d"
                         % (a["name"], aligned_bytes(nblk), nbytes))
    off = sds + a["offset"]
    spans.append((a["name"], off, nbytes, a["dims"]))
    skip.append((a["type_pos"], a["type_pos"] + 4))
    skip.append((off, off + nbytes))

print("converted 16 -> 43 : %d tensors" % len(spans))
print("type unchanged     : %d tensors" % unchanged)
if len(spans) + unchanged != len(st):
    raise SystemExit("FAIL: accounting")
print("HEADER: PASS")

# ---- level 2: whole-file diff outside the converted regions ----------------
skip.sort()
merged = []
for a0, a1 in skip:
    if merged and a0 <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], a1)
    else:
        merged.append([a0, a1])
keep = []
pos = 0
for a0, a1 in merged:
    if a0 > pos:
        keep.append((pos, a0))
    pos = a1
if pos < s_sz:
    keep.append((pos, s_sz))
total_keep = sum(b - a for a, b in keep)
print("comparing %d untouched regions, %d bytes (%.2f GB)"
      % (len(keep), total_keep, total_keep / 1e9))

CH = 1 << 26
done = 0
with open(src, "rb") as fa, open(dst, "rb") as fb:
    for a0, a1 in keep:
        p = a0
        while p < a1:
            n = min(CH, a1 - p)
            fa.seek(p); ba = fa.read(n)
            fb.seek(p); bb = fb.read(n)
            if len(ba) != n or len(bb) != n:
                raise SystemExit("FAIL: short read at %d" % p)
            if ba != bb:
                x = np.frombuffer(ba, dtype=np.uint8)
                y = np.frombuffer(bb, dtype=np.uint8)
                i = int(np.argmax(x != y))
                raise SystemExit("FAIL: untouched region differs at byte %d "
                                 "(%02x vs %02x)" % (p + i, x[i], y[i]))
            p += n
            done += n
        print("  ... %.1f%%" % (100.0 * done / total_keep), end="\r", flush=True)
print()
print("UNTOUCHED REGIONS: PASS (%d bytes identical)" % total_keep)

with open(manifest, "w") as f:
    for name, off, nbytes, dims in spans:
        f.write("%s %d %d %d %d %d\n" % (name, off, nbytes, dims[0], dims[1], dims[2]))
print("wrote manifest %s (%d converted spans)" % (manifest, len(spans)))
