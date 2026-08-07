#!/usr/bin/env python3
"""repack_iq2_mmq.py -- rewrite IQ2_XXS (16) routed-expert tensors as
IQ2_XXS_MMQ (43): llama.cpp-MMQ's *aligned SoA* artifact layout.

WHY.  block_iq2_xxs is 66 bytes with qs[] at offset 2, so a raw block stream is
only 2-byte aligned and nvcc emits LDG.E.U16 -- two 16-bit loads per 32-bit
weight word.  MMQ is weight-read-bound, so this is expensive.  Measured at the
production shape (K=4096, M=2048, 192 experts, 6 used, 4096 tokens): raw blocks
44.704 ms/pair-call vs aligned SoA 18.654 ms/pair-call = 2.40x.

NOT A RE-QUANTIZE.  Type 43 holds exactly the same 66 bytes per block as type
16, permuted into two planes.  Same values, same block count, same tensor byte
size -- so this is produced by shuffling bytes in a copy of any existing GGUF:
no HF source, no imatrix, no quantizer, and no numerical change whatsoever.

NOT THE SAME AS TYPE 42.  PULSAR_TENSOR_IQ2_XXS_SOA (42) is pulsar's own
Phase-0 split (q plane first, then d plane, no padding).  Type 43 is the layout
the vendored llama.cpp MMQ adapter's kernels read, which puts the d plane FIRST
and 64B-aligns the code plane.  The two are NOT interchangeable; they are
different permutations of the same bytes.

Layout, per tensor of nblk = ne/256 blocks -- must match
ds4_repack_iq2_aligned_device() / repack_iq2_xxs_aligned_kernel() in
src/cuda/mmq/ds4_repack.cu, and ds4_mmq_iq2_xxs_aligned_bytes() in ds4_mmq.h:

    dq_bytes = align_up(nblk * 2, 64)
    [0,        nblk*2)          d plane : block b's __half d at b*2
    [nblk*2,   dq_bytes)        zero pad to a 64B boundary
    [dq_bytes, +nblk*64)        q plane : block b's uint2 qs[8] at dq_bytes + b*64

Block linear order equals the raw tensor byte order (expert-major, then row,
then block), so nblk = dims[2] * dims[1] * (dims[0]/256).

SIZE IDENTITY.  aligned == raw requires align_up(nblk*2,64) + nblk*64 ==
nblk*66, i.e. nblk % 32 == 0.  The tool ASSERTS this per tensor and refuses to
convert a tensor that would grow; that assertion is the load-bearing check that
the dims were read correctly.  For the shipped v5mx4 shapes (192 experts x 2048
x 4096 and 192 x 4096 x 2048) nblk = 6291456, dq_bytes = 12582912 with zero
padding, and the artifact is exactly 415236096 bytes -- byte-identical to the
raw stream.

Because the size is identical, every tensor's data offset is unchanged, so this
copies the file and rewrites only (a) the type field of the targeted tensors in
the header and (b) those tensors' data regions in place.

Usage:
  repack_iq2_mmq.py IN.gguf OUT.gguf [--match SUBSTR] [--dry-run]
                                     [--no-copy] [--chunk-blocks N]

--match limits which tensor names are converted, comma-separated (default:
every 3-D IQ2_XXS routed-expert tensor).  Converting a SUBSET is legitimate:
the vendored adapter's own runtime repack candidate predicate
(ds4_repack_iq2_candidate) only ever covers ffn_gate_exps and ffn_up_exps, so
--match ffn_gate_exps,ffn_up_exps reproduces exactly the set the runtime cache
would have built, which is what an offline-vs-runtime logits parity run needs.
gate and up must be converted TOGETHER -- the fused gate+up kernels read one
layout and a half-repacked pair is rejected.

--no-copy rewrites OUT.gguf in place (it must already be a copy of IN.gguf);
useful to resume after an interrupted run without re-copying 86 GB.
"""
import argparse
import os
import shutil
import struct

import numpy as np

IQ2_XXS = 16
IQ2_XXS_MMQ = 43
QK_K = 256
BLK_BYTES = 66
Q_BYTES = 64
D_BYTES = 2
ALIGN = 64

EXPERT_SUFFIXES = (".ffn_gate_exps.weight", ".ffn_up_exps.weight",
                   ".ffn_down_exps.weight")


def align_up(x, a):
    return (x + a - 1) // a * a


def aligned_bytes(nblk):
    """Mirror of ds4_mmq_iq2_xxs_aligned_bytes(M, K, n_experts)."""
    return align_up(nblk * D_BYTES, ALIGN) + nblk * Q_BYTES


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
        raise SystemExit("unknown GGUF value type %d" % t)


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


def is_candidate(t, pats):
    """3-D IQ2_XXS routed-expert stack, optionally name-filtered."""
    if t["type"] != IQ2_XXS or len(t["dims"]) != 3:
        return False
    if not any(t["name"].endswith(s) for s in EXPERT_SUFFIXES):
        return False
    if pats and not any(p in t["name"] for p in pats):
        return False
    return True


def check_geometry(t):
    """Raise unless the tensor's aligned artifact is byte-identical in size to
    the raw block stream.  This is the assertion that catches wrong dims."""
    name, dims, ne = t["name"], t["dims"], t["ne"]
    if ne % QK_K:
        raise SystemExit("%s: ne=%d not a multiple of %d" % (name, ne, QK_K))
    if dims[0] % QK_K:
        raise SystemExit("%s: dims[0]=%d not a multiple of %d" % (name, dims[0], QK_K))
    nblk = ne // QK_K
    raw = nblk * BLK_BYTES
    ali = aligned_bytes(nblk)
    if ali != raw:
        raise SystemExit(
            "%s: aligned size %d != raw size %d (delta %d); nblk=%d is not a "
            "multiple of 32 -- refusing to grow the file"
            % (name, ali, raw, ali - raw, nblk))
    return nblk, raw


def repack_tensor(f, base, nblk, chunk_blocks):
    """Permute one tensor's data region in place, raw -> aligned.

    ORDERING IS LOAD-BEARING.  The q plane write for block b lands at
    nblk*2 + b*64, which is always AT OR ABOVE that block's raw read position
    b*66 -- so a forward pass would overwrite blocks it has not read yet.
    Walking chunks in DESCENDING order fixes this: while processing chunk
    [b0, b1) the still-unread raw bytes are [0, 66*b0), and the write region
    starts at nblk*2 + 64*b0 >= 66*b0 for every b0 <= nblk.  Already-written q
    data for blocks >= b1 starts at nblk*2 + 64*b1, above our write end, so it
    is untouched too.

    The d plane is NOT safe that way: it occupies [0, nblk*2), which sits
    inside the first ~3% of the packed region, so writing it during the pass
    would clobber packed blocks not yet read.  Buffer d (2 B/block, ~12 MB for
    the shipped tensors) and flush it after the pass.
    """
    dq_bytes = align_up(nblk * D_BYTES, ALIGN)
    dbuf = np.empty((nblk, D_BYTES), dtype=np.uint8)
    starts = list(range(0, nblk, chunk_blocks))
    for b0 in reversed(starts):
        nb = min(chunk_blocks, nblk - b0)
        f.seek(base + b0 * BLK_BYTES)
        raw = f.read(nb * BLK_BYTES)
        if len(raw) != nb * BLK_BYTES:
            raise SystemExit("short read at block %d" % b0)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(nb, BLK_BYTES)
        dbuf[b0:b0 + nb] = arr[:, :D_BYTES]
        q = np.ascontiguousarray(arr[:, D_BYTES:])
        f.seek(base + dq_bytes + b0 * Q_BYTES)
        f.write(q.tobytes())
    f.seek(base)
    f.write(dbuf.tobytes())
    if dq_bytes > nblk * D_BYTES:
        f.write(b"\0" * (dq_bytes - nblk * D_BYTES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--match", default="",
                    help="comma-separated substrings; convert if ANY matches")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-copy", action="store_true",
                    help="dst already is a copy of src; rewrite it in place")
    ap.add_argument("--chunk-blocks", type=int, default=1 << 20,
                    help="blocks per streamed chunk (default 1Mi ~ 66 MB)")
    a = ap.parse_args()

    tensors, alignment, data_start = scan(a.src)
    pats = [x for x in a.match.split(",") if x]
    targets = [t for t in tensors if is_candidate(t, pats)]

    n_iq2 = sum(1 for t in tensors if t["type"] == IQ2_XXS)
    print("tensors=%d alignment=%d data_start=%d" % (len(tensors), alignment, data_start))
    print("IQ2_XXS(16) total=%d  selected targets=%d" % (n_iq2, len(targets)))
    if not targets:
        raise SystemExit("no IQ2_XXS routed-expert tensors matched -- nothing to do")

    total_raw = 0
    by_shape = {}
    for t in targets:
        nblk, raw = check_geometry(t)
        t["nblk"] = nblk
        t["bytes"] = raw
        total_raw += raw
        by_shape.setdefault(tuple(t["dims"]), []).append(t["name"])
    print("all %d targets pass the aligned==raw size assertion" % len(targets))
    for dims, names in sorted(by_shape.items()):
        t0 = next(t for t in targets if t["name"] == names[0])
        print("  dims=%s n=%d nblk=%d bytes=%d (aligned=%d)"
              % (list(dims), len(names), t0["nblk"], t0["bytes"],
                 aligned_bytes(t0["nblk"])))
    print("total converted bytes = %d (%.2f GB), growth = 0" % (total_raw, total_raw / 1e9))

    if a.dry_run:
        print("dry-run: no output written")
        return

    if not a.no_copy:
        print("copying %s -> %s ..." % (a.src, a.dst), flush=True)
        shutil.copyfile(a.src, a.dst)
        print("copy done", flush=True)

    done = 0
    with open(a.dst, "r+b") as f:
        for t in targets:
            repack_tensor(f, data_start + t["offset"], t["nblk"], a.chunk_blocks)
            f.seek(t["type_pos"])
            f.write(struct.pack("<I", IQ2_XXS_MMQ))
            done += 1
            print("  repacked %d/%d %s" % (done, len(targets), t["name"]), flush=True)
        f.flush()
        os.fsync(f.fileno())
    print("wrote %s" % a.dst)


if __name__ == "__main__":
    main()
