#!/usr/bin/env python3
"""Reference IQ2_XXS (16) -> IQ2_XXS_MMQ (43) permutation, for byte-comparing
against the native ds4q_pack_iq2_xxs_mmq(). Driven by `make test-iq2-mmq`.

This mirrors the transform in gguf-tools/repack_iq2_mmq.py, which is the
layout's reference implementation (and matches ds4_repack_iq2_aligned_device()
in src/cuda/mmq/ds4_repack.cu). Kept deliberately independent and stdlib-only:
the point of the test is that two separately-written implementations agree, so
importing the other one would defeat it.

  raw block b: [2 B __half d][64 B qs]           (66 B, 2-byte aligned)
  mmq:         [0,  nblk*2)   d plane
               [.., dq)       zero pad to 64 B
               [dq, +nblk*64) q plane            (dq = align_up(nblk*2, 64))
"""
import sys


def pack(raw: bytes, nblk: int) -> bytes:
    if nblk % 32:
        raise SystemExit("nblk %d is not a multiple of 32 (layout would grow)" % nblk)
    dq = (nblk * 2 + 63) // 64 * 64
    out = bytearray(dq + nblk * 64)
    for b in range(nblk):
        src = b * 66
        out[b * 2:b * 2 + 2] = raw[src:src + 2]
        q = dq + b * 64
        out[q:q + 64] = raw[src + 2:src + 66]
    return bytes(out)


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: %s NBLK RAW16_FILE OUT_FILE" % sys.argv[0])
    nblk = int(sys.argv[1])
    raw = open(sys.argv[2], "rb").read()
    if len(raw) != nblk * 66:
        raise SystemExit("raw is %d bytes, expected %d for nblk=%d"
                         % (len(raw), nblk * 66, nblk))
    out = pack(raw, nblk)
    if len(out) != len(raw):
        raise SystemExit("size identity broken: %d != %d" % (len(out), len(raw)))
    open(sys.argv[3], "wb").write(out)
    print("ref: nblk=%d mmq=%d" % (nblk, len(out)))


if __name__ == "__main__":
    main()
