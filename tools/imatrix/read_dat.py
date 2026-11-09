#!/usr/bin/env python3
"""Read a llama.cpp legacy importance-matrix `.dat` (what `pulsar --imatrix-out` writes, for DeepSeek and Qwen)
and check it: the frame parses to the last byte, every value is finite and non-negative, and each entry's length
is a whole number of vectors.  Prints one line per entry, or only the totals with --quiet.

usage: read_dat.py FILE [--quiet] [--expect-substr NAME ...]
  --expect-substr  fail unless some entry name contains NAME (repeatable)
"""
import argparse, math, struct, sys


def read(path):
    data = open(path, "rb").read()
    off = 0

    def i32():
        nonlocal off
        v = struct.unpack_from("<i", data, off)[0]
        off += 4
        return v

    n = i32()
    entries = []
    for _ in range(n):
        ln = i32()
        name = data[off:off + ln].decode()
        off += ln
        ncall = i32()
        nval = i32()
        vals = struct.unpack_from("<%df" % nval, data, off)
        off += 4 * nval
        entries.append((name, ncall, vals))
    chunks = i32()
    dl = i32()
    dataset = data[off:off + dl].decode()
    off += dl
    if off != len(data):
        raise SystemExit("%s: %d trailing bytes after the frame" % (path, len(data) - off))
    return entries, chunks, dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--expect-substr", action="append", default=[])
    a = ap.parse_args()
    entries, chunks, dataset = read(a.file)
    bad = 0
    for name, ncall, vals in entries:
        nonfinite = sum(1 for v in vals if not math.isfinite(v) or v < 0)
        bad += nonfinite
        if not a.quiet:
            print("%-72s ncall=%d nval=%d mean=%.4g max=%.4g%s" % (
                name, ncall, len(vals), sum(vals) / max(1, len(vals)), max(vals) if vals else 0.0,
                "  NONFINITE/NEGATIVE=%d" % nonfinite if nonfinite else ""))
    missing = [s for s in a.expect_substr if not any(s in e[0] for e in entries)]
    print("%s: %d entries, %d values, chunks=%d, dataset=%s, %d bad values%s" % (
        a.file, len(entries), sum(len(e[2]) for e in entries), chunks, dataset, bad,
        ", MISSING " + ",".join(missing) if missing else ""))
    sys.exit(1 if bad or missing or not entries else 0)


if __name__ == "__main__":
    main()
