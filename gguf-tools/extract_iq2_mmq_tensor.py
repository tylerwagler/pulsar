#!/usr/bin/env python3
"""Extract one IQ2_XXS routed-expert tensor's RAW block stream from a GGUF and
produce the ALIGNED artifact for it using the exact code path repack_iq2_mmq.py
uses on the real model (repack_tensor, in place, descending chunks).

Emits <outdir>/raw.bin and <outdir>/aligned.bin plus a geometry line the CUDA
round-trip harness parses.  Keeping the producer identical to the shipping tool
is the point: the harness then compares the TOOL's bytes against the device
repack, not a re-implementation of it.

Usage: extract_iq2_mmq_tensor.py GGUF TENSOR_NAME OUTDIR
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_iq2_mmq import scan, check_geometry, repack_tensor, aligned_bytes

gguf, name, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outdir, exist_ok=True)

tensors, alignment, data_start = scan(gguf)
t = next((x for x in tensors if x["name"] == name), None)
if t is None:
    raise SystemExit("tensor not found: %s" % name)
if t["type"] != 16:
    raise SystemExit("%s is type %d, not IQ2_XXS(16)" % (name, t["type"]))
nblk, raw_bytes = check_geometry(t)

raw_path = os.path.join(outdir, "raw.bin")
ali_path = os.path.join(outdir, "aligned.bin")
base = data_start + t["offset"]
with open(gguf, "rb") as f, open(raw_path, "wb") as g:
    f.seek(base)
    left = raw_bytes
    while left:
        n = min(left, 1 << 26)
        b = f.read(n)
        if len(b) != n:
            raise SystemExit("short read")
        g.write(b)
        left -= n

shutil.copyfile(raw_path, ali_path)
with open(ali_path, "r+b") as f:
    repack_tensor(f, 0, nblk, 1 << 20)

print("name=%s" % name)
print("dims=%s" % t["dims"])
print("nblk=%d" % nblk)
print("raw_bytes=%d" % raw_bytes)
print("aligned_bytes=%d" % aligned_bytes(nblk))
print("in_dim=%d out_dim=%d group_count=%d" % (t["dims"][0], t["dims"][1], t["dims"][2]))
print("raw=%s aligned=%s" % (raw_path, ali_path))
