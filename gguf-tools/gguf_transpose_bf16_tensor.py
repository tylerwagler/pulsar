#!/usr/bin/env python3
"""In-place, size-preserving TRANSPOSE of one 2D BF16 tensor in a GGUF.

L213: the engine stores the DSpark drafter's markov_w2 k-major (dims (vocab,
256), element (v, i) at i*vocab + v) so the markov kernels' warps read
contiguous memory.  The quantizer emits that layout for a fresh build
(quantize/dsq_generate.c, is_kmajor_tensor); this tool migrates an EXISTING
artifact without re-quantising 92 GB: copy the file, swap the tensor's two
header dims, permute its data block.  A bf16 transpose is exact and the byte
count is unchanged, so no offsets move and every other byte of the file is
untouched.

  gguf_transpose_bf16_tensor.py SRC DST --tensor NAME [--verify]

--verify re-reads DST and proves two things: the named tensor equals the
transpose of SRC's, and every byte outside the two rewritten ranges (16 header
bytes, one data block) is identical to SRC.  Refuses a square tensor (the
loader could not tell the layouts apart) and refuses SRC == DST.
"""
import argparse, os, shutil, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_hdr import scan  # noqa: E402

GGML_BF16 = 30
CHUNK = 64 << 20


def find(tensors, name):
    for t in tensors:
        if t["name"] == name:
            return t
    sys.exit("tensor %r not in file" % name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if os.path.realpath(a.src) == os.path.realpath(a.dst):
        sys.exit("refusing: SRC and DST are the same file")
    import numpy as np

    tensors, alignment, data_start = scan(a.src)
    t = find(tensors, a.tensor)
    ne = list(t["dims"])              # scan(): dims = the list, ne = element COUNT
    if t["type"] != GGML_BF16:
        sys.exit("refusing: %s is type %d, not BF16 (%d)" % (a.tensor, t["type"], GGML_BF16))
    if len(ne) != 2:
        sys.exit("refusing: %s is %dD, need 2D" % (a.tensor, len(ne)))
    if ne[0] == ne[1]:
        sys.exit("refusing: square tensor -- the two layouts are indistinguishable by dims")
    ne0, ne1 = ne                      # ne0 is the fastest dim
    nbytes = ne0 * ne1 * 2
    dims_pos = t["type_pos"] - 16      # GGUF tensor info: name, n_dims(u32), dims[](u64 each), type(u32), offset(u64)
    data_pos = data_start + t["offset"]

    print("src %s\n  %s type %d ne=(%d, %d)  header dims @%d  data @%d (%d bytes)"
          % (a.src, a.tensor, t["type"], ne0, ne1, dims_pos, data_pos, nbytes))

    # sanity: the header really holds (ne0, ne1) where we think it does
    with open(a.src, "rb") as f:
        f.seek(dims_pos)
        h0, h1 = struct.unpack("<QQ", f.read(16))
    if (h0, h1) != (ne0, ne1):
        sys.exit("header dims at %d read (%d, %d), scan says (%d, %d) -- layout assumption wrong, refusing"
                 % (dims_pos, h0, h1, ne0, ne1))

    print("copying %s -> %s" % (a.src, a.dst))
    shutil.copyfile(a.src, a.dst)

    with open(a.dst, "r+b") as f:
        f.seek(data_pos)
        raw = f.read(nbytes)
        if len(raw) != nbytes:
            sys.exit("short read of tensor data")
        # memory is [ne1 rows][ne0 cols]; the transpose is [ne0 rows][ne1 cols], i.e. new ne = (ne1, ne0)
        m = np.frombuffer(raw, dtype=np.uint16).reshape(ne1, ne0)
        mt = np.ascontiguousarray(m.T)
        f.seek(data_pos)
        f.write(mt.tobytes())
        f.seek(dims_pos)
        f.write(struct.pack("<QQ", ne1, ne0))
    print("wrote transposed data (%d bytes) and dims (%d, %d) -> (%d, %d)" % (nbytes, ne0, ne1, ne1, ne0))

    if not a.verify:
        return

    print("verifying ...")
    tensors2, alignment2, data_start2 = scan(a.dst)
    t2 = find(tensors2, a.tensor)
    if list(t2["dims"]) != [ne1, ne0] or t2["type"] != GGML_BF16 or data_start2 != data_start \
            or t2["offset"] != t["offset"] or alignment2 != alignment:
        sys.exit("VERIFY FAIL: dst header does not describe the transposed tensor in place")
    with open(a.src, "rb") as fs, open(a.dst, "rb") as fd:
        fs.seek(data_pos); fd.seek(data_pos)
        src_m = np.frombuffer(fs.read(nbytes), dtype=np.uint16).reshape(ne1, ne0)
        dst_m = np.frombuffer(fd.read(nbytes), dtype=np.uint16).reshape(ne0, ne1)
        if not np.array_equal(dst_m, src_m.T):
            sys.exit("VERIFY FAIL: dst tensor is not the transpose of src")
        # every byte outside the two rewritten ranges must be identical
        size = os.path.getsize(a.src)
        if os.path.getsize(a.dst) != size:
            sys.exit("VERIFY FAIL: size changed")
        skip = [(dims_pos, dims_pos + 16), (data_pos, data_pos + nbytes)]
        fs.seek(0); fd.seek(0)
        pos = 0
        while pos < size:
            bs = fs.read(CHUNK); bd = fd.read(CHUNK)
            if len(bs) != len(bd):
                sys.exit("VERIFY FAIL: read length differs at %d" % pos)
            if bs != bd:
                # EVERY differing byte must fall inside one of the two rewritten ranges
                diff = pos + np.flatnonzero(np.frombuffer(bs, dtype=np.uint8) != np.frombuffer(bd, dtype=np.uint8))
                inside = np.zeros(diff.shape, dtype=bool)
                for lo, hi in skip:
                    inside |= (diff >= lo) & (diff < hi)
                if not inside.all():
                    sys.exit("VERIFY FAIL: byte %d differs outside the rewritten ranges" % int(diff[~inside][0]))
            pos += len(bs)
    print("VERIFY OK: %s is the exact transpose; all %d other bytes identical" % (a.tensor, size - nbytes - 16))


if __name__ == "__main__":
    main()
