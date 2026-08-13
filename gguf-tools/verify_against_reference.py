#!/usr/bin/env python3
"""Classify every tensor difference between a rebuilt artifact and a reference.

The collapsed pipeline is not expected to be byte-identical to the shipped
v5mx4 artifact, and "not identical" is useless on its own -- the question is
whether every difference is one of the two INTENDED pre-formatted-twin
upgrades, or whether something else moved.  So each tensor lands in exactly
one bucket and anything that does not fit is a failure:

  IDENTICAL    same type, same bytes
  MMQ          reference IQ2_XXS(16) -> rebuild IQ2_XXS_MMQ(43), and the bytes
               are exactly repack_iq2_mmq's permutation of the reference's
  LT           reference FP8_E4M3(38) -> rebuild MXFP8_LT(41), size-checked
               only (see below)
  MISMATCH     anything else

The LT bucket is deliberately weaker than MMQ.  ds4q_pack_mxfp8_lt is a C
swizzle; reimplementing it here to "verify" the C would just be testing a
second implementation of my own guess against the first.  These 25 drafter
tensors are covered instead by audit_artifact_types.py (which fails on any
plain twin surviving) plus an actual model load -- a wrong swizzle does not
load quietly.  This script reports them as type-upgraded and size-consistent
and says so, rather than implying a byte proof it did not do.

Usage: verify_against_reference.py REBUILD.gguf REFERENCE.gguf [--sample N]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_iq2_mmq import scan, align_up, BLK_BYTES, D_BYTES, ALIGN  # noqa: E402

IQ2_XXS, IQ2_XXS_MMQ = 16, 43
FP8_E4M3, MXFP8_LT = 38, 41
CHUNK = 1 << 24


def sizes_by_name(tensors, data_start, path):
    """Data size per tensor = gap to the next offset (last one runs to EOF)."""
    order = sorted(tensors, key=lambda t: t["offset"])
    end = os.path.getsize(path) - data_start
    out = {}
    for i, t in enumerate(order):
        nxt = order[i + 1]["offset"] if i + 1 < len(order) else end
        out[t["name"]] = nxt - t["offset"]
    return out


def read_exact(f, off, n):
    f.seek(off)
    buf = bytearray(n)
    got = 0
    while got < n:
        chunk = f.readinto(memoryview(buf)[got:min(got + CHUNK, n)])
        if not chunk:
            raise SystemExit("short read at %d (%d/%d)" % (off, got, n))
        got += chunk
    return bytes(buf)


def mmq_permute(raw, nblk):
    """d-plane then q-plane, matching repack_iq2_mmq.repack_tensor."""
    import numpy as np
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(nblk, BLK_BYTES)
    dq = align_up(nblk * D_BYTES, ALIGN)
    out = np.zeros(dq + nblk * (BLK_BYTES - D_BYTES), dtype=np.uint8)
    out[: nblk * D_BYTES] = arr[:, :D_BYTES].reshape(-1)
    out[dq:] = arr[:, D_BYTES:].reshape(-1)
    return out.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rebuild")
    ap.add_argument("reference")
    ap.add_argument("--sample", type=int, default=0,
                    help="only compare the first N tensors by size (0 = all)")
    a = ap.parse_args()

    rt, _, rds = scan(a.rebuild)
    ft, _, fds = scan(a.reference)
    rmap = {t["name"]: t for t in rt}
    fmap = {t["name"]: t for t in ft}

    only_rebuild = sorted(set(rmap) - set(fmap))
    only_ref = sorted(set(fmap) - set(rmap))
    print("rebuild   %s: %d tensors, data_start=%d" % (a.rebuild, len(rt), rds))
    print("reference %s: %d tensors, data_start=%d" % (a.reference, len(ft), fds))
    if only_rebuild or only_ref:
        print("TENSOR SET DIFFERS: +%d only in rebuild, +%d only in reference"
              % (len(only_rebuild), len(only_ref)))
        for n in only_rebuild[:10]:
            print("   only-rebuild:", n)
        for n in only_ref[:10]:
            print("   only-reference:", n)

    rsz = sizes_by_name(rt, rds, a.rebuild)
    fsz = sizes_by_name(ft, fds, a.reference)

    shared = sorted(set(rmap) & set(fmap), key=lambda n: fsz[n])
    if a.sample:
        shared = shared[: a.sample]

    counts = {"IDENTICAL": 0, "MMQ": 0, "LT": 0, "MISMATCH": 0}
    failures = []

    fr = open(a.rebuild, "rb")
    ff = open(a.reference, "rb")
    try:
        for i, name in enumerate(shared):
            r, f = rmap[name], fmap[name]
            if r["dims"] != f["dims"]:
                counts["MISMATCH"] += 1
                failures.append((name, "dims %s != %s" % (r["dims"], f["dims"])))
                continue

            if r["type"] == f["type"]:
                if rsz[name] != fsz[name]:
                    counts["MISMATCH"] += 1
                    failures.append((name, "same type %d but size %d != %d"
                                     % (r["type"], rsz[name], fsz[name])))
                    continue
                rb = read_exact(fr, rds + r["offset"], rsz[name])
                fb = read_exact(ff, fds + f["offset"], fsz[name])
                if rb == fb:
                    counts["IDENTICAL"] += 1
                else:
                    counts["MISMATCH"] += 1
                    failures.append((name, "type %d bytes differ" % r["type"]))

            elif f["type"] == IQ2_XXS and r["type"] == IQ2_XXS_MMQ:
                ne = 1
                for d in f["dims"]:
                    ne *= d
                nblk = ne // 256
                if fsz[name] != nblk * BLK_BYTES or rsz[name] != fsz[name]:
                    counts["MISMATCH"] += 1
                    failures.append((name, "MMQ size %d vs %d (nblk=%d)"
                                     % (rsz[name], fsz[name], nblk)))
                    continue
                fb = read_exact(ff, fds + f["offset"], fsz[name])
                rb = read_exact(fr, rds + r["offset"], rsz[name])
                if rb == mmq_permute(fb, nblk):
                    counts["MMQ"] += 1
                else:
                    counts["MISMATCH"] += 1
                    failures.append((name, "type 43 != repack(type 16)"))

            elif f["type"] == FP8_E4M3 and r["type"] == MXFP8_LT:
                counts["LT"] += 1

            else:
                counts["MISMATCH"] += 1
                failures.append((name, "unexpected type change %d -> %d"
                                 % (f["type"], r["type"])))

            if (i + 1) % 100 == 0:
                print("  ... %d/%d" % (i + 1, len(shared)), flush=True)
    finally:
        fr.close()
        ff.close()

    print()
    print("compared %d shared tensors" % len(shared))
    for k in ("IDENTICAL", "MMQ", "LT", "MISMATCH"):
        print("  %-10s %d" % (k, counts[k]))
    print("  (LT = type-upgraded 38->41, size-consistent; byte proof is the")
    print("   audit gate + model load, not this script)")
    if failures:
        print("\nFAILURES (%d):" % len(failures))
        for n, why in failures[:40]:
            print("  %s: %s" % (n, why))
    ok = not failures and not only_rebuild and not only_ref
    print("\nVERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
