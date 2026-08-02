#!/usr/bin/env python3
"""repack_iq2_soa.py -- rewrite IQ2_XXS (16) expert tensors as IQ2_XXS_SOA (42).

WHY THIS IS NOT A RE-QUANTIZE.  Type 42 holds exactly the same 66 bytes per
block as type 16, with the two planes split so the code stream is load-aligned
(block_iq2_xxs puts qs[] at offset 2, forcing 2-byte LDG.E.U16 loads).  It is a
pure permutation: same values, same block count, same tensor byte size.  So the
artifact can be produced from any existing GGUF by shuffling bytes -- no HF
source, no imatrix, no quantizer, and no numerical change whatsoever.

Because the size is identical, every tensor's data offset is unchanged too, so
this copies the file and rewrites only (a) the type field of the targeted
tensors in the header and (b) those tensors' data regions in place.

Layout, per tensor of nblk = ne/256 blocks -- must match
ds4q_iq2_xxs_soa_repack() in gguf-tools/quants_common.c:
    [0,        nblk*64) q plane : block b at b*64  (uint16 qs[32])
    [nblk*64, +nblk*2)  d plane : block b at nblk*64 + b*2 (uint16 d)

Usage:
  repack_iq2_soa.py IN.gguf OUT.gguf [--match SUBSTR] [--dry-run]

--match limits which tensor names are converted (default: every IQ2_XXS
tensor).  --dry-run reports what would change and exits without writing.
"""
import argparse
import os
import shutil
import struct
import sys

import numpy as np

IQ2_XXS = 16
IQ2_XXS_SOA = 42
QK_K = 256
BLK_BYTES = 66


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _skip_value(f, t):
    if t in (0, 1, 7):   f.read(1)
    elif t in (2, 3):    f.read(2)
    elif t in (4, 5, 6): f.read(4)
    elif t in (10, 11, 12): f.read(8)
    elif t == 8:         _read_str(f)
    elif t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _skip_value(f, et)
    else:
        raise SystemExit(f"unknown GGUF value type {t}")


def scan(path):
    """Return (tensors, alignment, data_start). Each tensor records where its
    TYPE field sits in the header so it can be patched without reflowing."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit("not a GGUF")
        struct.unpack("<I", f.read(4))
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        alignment = 32
        for _ in range(n_kv):
            key = _read_str(f)
            (vt,) = struct.unpack("<I", f.read(4))
            if key == "general.alignment" and vt == 4:
                (alignment,) = struct.unpack("<I", f.read(4))
            else:
                _skip_value(f, vt)
        tensors = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            type_pos = f.tell()
            (ttype,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            ne = 1
            for d in dims:
                ne *= d
            tensors.append(dict(name=name, dims=dims, type=ttype, type_pos=type_pos,
                                offset=offset, ne=ne))
        data_start = f.tell()
    pad = (alignment - (data_start % alignment)) % alignment
    return tensors, alignment, data_start + pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--match", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tensors, alignment, data_start = scan(a.src)
    targets = [t for t in tensors
               if t["type"] == IQ2_XXS and (not a.match or a.match in t["name"])]
    total = sum(t["ne"] // QK_K * BLK_BYTES for t in targets)
    print(f"tensors={len(tensors)} alignment={alignment} data_start={data_start}")
    print(f"IQ2_XXS targets={len(targets)}  bytes={total/1e9:.2f} GB")
    for t in targets[:3]:
        print(f"  e.g. {t['name']} dims={t['dims']} nblk={t['ne']//QK_K}")
    if not targets:
        raise SystemExit("no IQ2_XXS tensors matched -- nothing to do")
    for t in targets:
        if t["ne"] % QK_K:
            raise SystemExit(f"{t['name']}: ne={t['ne']} not a multiple of {QK_K}")
    if a.dry_run:
        print("dry-run: no output written")
        return

    print(f"copying {a.src} -> {a.dst} ...", flush=True)
    shutil.copyfile(a.src, a.dst)

    done = 0
    with open(a.dst, "r+b") as f:
        for t in targets:
            nblk = t["ne"] // QK_K
            base = data_start + t["offset"]
            # IN-PLACE ORDERING, and it is load-bearing:
            #   q plane block b -> base + b*64   (packed block b was at b*66)
            #   d plane block b -> base + nblk*64 + b*2
            # The q write for chunk i lands at i*CH*64, strictly BELOW the next
            # read at (i+1)*CH*66, so the write pointer always trails the read
            # pointer and streaming in place is safe.
            # The d plane is NOT safe that way: it starts at nblk*64, which is
            # inside the packed region (~97% in), so writing d during the pass
            # would clobber packed blocks not yet read.  Buffer d (2 B/block,
            # ~17 MB for the largest shipped tensor) and flush it after.
            CH = 1 << 20  # blocks per chunk (~66 MB read)
            dbuf = np.empty((nblk, 2), dtype=np.uint8)
            for b0 in range(0, nblk, CH):
                nb = min(CH, nblk - b0)
                f.seek(base + b0 * BLK_BYTES)
                raw = np.frombuffer(f.read(nb * BLK_BYTES), dtype=np.uint8)
                arr = raw.reshape(nb, BLK_BYTES)
                dbuf[b0:b0 + nb] = arr[:, :2]
                q = np.ascontiguousarray(arr[:, 2:])          # (nb, 64)
                f.seek(base + b0 * 64)
                f.write(q.tobytes())
            f.seek(base + nblk * 64)
            f.write(dbuf.tobytes())
            f.seek(t["type_pos"])
            f.write(struct.pack("<I", IQ2_XXS_SOA))
            done += 1
            if done % 10 == 0 or done == len(targets):
                print(f"  repacked {done}/{len(targets)}", flush=True)
    print(f"wrote {a.dst}")


if __name__ == "__main__":
    main()
