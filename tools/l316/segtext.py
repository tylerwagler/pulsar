#!/usr/bin/env python3
"""segtext.py -- L316: the chain texts a disk-KV segment store holds (the conversation as the bank rendered it).

    python3 tools/l316/segtext.py KVDIR [--deepest] [--out FILE]

Each <key>.seg is a header (152 bytes), its text, the payload and an optional trailer (src/lib/pulsar_segstore.cpp).
A key names the whole text from 0 to its G; this walks the parents to rebuild every leaf's full text and lists the
leaves.  --deepest writes the deepest leaf's text (stdout or --out)."""
import argparse, os, struct, sys

HDR = 152   # src/lib/pulsar_segstore.cpp HDR
ap = argparse.ArgumentParser()
ap.add_argument("kvdir")
ap.add_argument("--deepest", action="store_true")
ap.add_argument("--out")
a = ap.parse_args()
segs = {}
for f in os.listdir(a.kvdir):
    if not f.endswith(".seg"):
        continue
    with open(os.path.join(a.kvdir, f), "rb") as fp:
        h = fp.read(HDR)
        if len(h) < HDR or struct.unpack_from("<I", h, 0)[0] != 0x31475350:
            continue
        key = h[16:56].decode()
        parent = h[56:96].rstrip(b"\0").decode()
        g_prev, g = struct.unpack_from("<II", h, 96)
        (tlen,) = struct.unpack_from("<Q", h, 104)
        segs[key] = (parent, g_prev, g, fp.read(tlen))
parents = {v[0] for v in segs.values()}


def full(k):
    out = []
    while k:
        p, _, _, t = segs[k]
        out.append(t)
        k = p
    return b"".join(reversed(out))


leaves = sorted((v[2], k) for k, v in segs.items() if k not in parents)
for g, k in leaves:
    print("leaf G=%d key=%s text=%d bytes" % (g, k[:12], len(full(k))), file=sys.stderr)
if a.deepest and leaves:
    t = full(leaves[-1][1])
    (open(a.out, "wb") if a.out else sys.stdout.buffer).write(t)
