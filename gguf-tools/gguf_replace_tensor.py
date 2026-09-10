#!/usr/bin/env python3
"""Replace ONE tensor's payload (and its type / dims) in a GGUF, reflowing the
tensor directory -- for a replacement whose byte size CHANGES, which the
size-preserving in-place tools (gguf_transpose_bf16_tensor.py, the F16->BF16
splice) cannot do.  Every other tensor is streamed byte-for-byte from SRC.

Type-agnostic on purpose: a tensor's payload length is taken from the file
itself (the gap to the next tensor's offset, or to end-of-file), never from a
per-type size table, so a type this script has never heard of copies fine.

L213 step 2: splice the blob dsq_requant_tensor produced for the drafter's
markov_w2 (type 45 int8+rowscale or type 38 MXFP8) into a copy of the shipped
artifact.

  gguf_replace_tensor.py SRC DST --tensor NAME --blob FILE --type T --dims D0,D1[,...] [--verify]

--verify re-reads DST and proves: the directory describes the replacement as
requested; every OTHER tensor's dims, type and payload bytes equal SRC's; and
the replaced tensor's payload equals the blob.
"""
import argparse, os, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_hdr import scan  # noqa: E402

CHUNK = 64 << 20


def pad_to(v, a): return (v + a - 1) // a * a


def dir_start_of(path, t0):
    """Byte position where the tensor directory begins = tensor 0's name-length
    field; derived from scan()'s type_pos and asserted against the file."""
    name_b = t0["name"].encode("utf-8")
    start = t0["type_pos"] - 8 * len(t0["dims"]) - 4 - len(name_b) - 8
    with open(path, "rb") as f:
        f.seek(start)
        (n,) = struct.unpack("<Q", f.read(8))
        if n != len(name_b) or f.read(n) != name_b:
            sys.exit("directory start derivation failed (name-length field mismatch)")
    return start


def slots(tensors, data_start, file_size):
    """Payload slot length of each tensor from the file layout alone."""
    order = sorted(tensors, key=lambda t: t["offset"])
    out = {}
    for k, t in enumerate(order):
        end = order[k + 1]["offset"] if k + 1 < len(order) else file_size - data_start
        out[t["name"]] = end - t["offset"]
    return order, out


def copy_range(src, dst, pos, n):
    src.seek(pos)
    left = n
    while left:
        b = src.read(min(CHUNK, left))
        if not b: sys.exit("short read at %d" % pos)
        dst.write(b); left -= len(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--blob", required=True)
    ap.add_argument("--type", type=int, required=True)
    ap.add_argument("--dims", required=True, help="new dims, fastest first, e.g. 129280,256")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if os.path.realpath(a.src) == os.path.realpath(a.dst): sys.exit("refusing: SRC and DST are the same file")
    new_dims = [int(x) for x in a.dims.split(",")]
    blob_size = os.path.getsize(a.blob)

    tensors, alignment, data_start = scan(a.src)
    file_size = os.path.getsize(a.src)
    byname = {t["name"]: t for t in tensors}
    if a.tensor not in byname: sys.exit("tensor %r not in %s" % (a.tensor, a.src))
    old = byname[a.tensor]
    if len(new_dims) != len(old["dims"]):
        sys.exit("refusing: new dims rank %d != old rank %d (directory size would change)" % (len(new_dims), len(old["dims"])))
    order, slot = slots(tensors, data_start, file_size)
    dir_start = dir_start_of(a.src, order[0] if order[0]["type_pos"] < tensors[0]["type_pos"] else tensors[0])
    # tensors[] is in DIRECTORY order (scan reads it in order); order[] is in DATA order

    print("src %s: %d tensors, alignment %d, data_start %d" % (a.src, len(tensors), alignment, data_start))
    print("  %s: type %d dims %s slot %d bytes  ->  type %d dims %s blob %d bytes (padded %d)"
          % (a.tensor, old["type"], old["dims"], slot[a.tensor], a.type, new_dims, blob_size, pad_to(blob_size, alignment)))

    # new offsets in DATA order
    new_off = {}
    pos = 0
    for t in order:
        new_off[t["name"]] = pos
        n = pad_to(blob_size, alignment) if t["name"] == a.tensor else slot[t["name"]]
        pos += n
    new_payload_total = pos

    with open(a.src, "rb") as fs, open(a.blob, "rb") as fb, open(a.dst, "wb") as fd:
        # header + KVs verbatim (tensor count unchanged)
        copy_range(fs, fd, 0, dir_start)
        # directory, in the original directory order
        for t in tensors:
            nb = t["name"].encode("utf-8")
            fd.write(struct.pack("<Q", len(nb))); fd.write(nb)
            dims = new_dims if t["name"] == a.tensor else t["dims"]
            fd.write(struct.pack("<I", len(dims)))
            for d in dims: fd.write(struct.pack("<Q", d))
            fd.write(struct.pack("<I", a.type if t["name"] == a.tensor else t["type"]))
            fd.write(struct.pack("<Q", new_off[t["name"]]))
        here = fd.tell()
        new_data_start = pad_to(here, alignment)
        fd.write(b"\0" * (new_data_start - here))
        if new_data_start != data_start:
            print("  note: data_start moved %d -> %d (directory size changed)" % (data_start, new_data_start))
        # payloads, in data order
        for t in order:
            if t["name"] == a.tensor:
                fb.seek(0)
                left = blob_size
                while left:
                    b = fb.read(min(CHUNK, left)); fd.write(b); left -= len(b)
                fd.write(b"\0" * (pad_to(blob_size, alignment) - blob_size))
            else:
                copy_range(fs, fd, data_start + t["offset"], slot[t["name"]])
        fd.flush(); os.fsync(fd.fileno())
    print("wrote %s: payload %d -> %d bytes" % (a.dst, file_size - data_start, new_payload_total))

    if not a.verify: return
    print("verifying ...")
    t2s, al2, ds2 = scan(a.dst)
    by2 = {t["name"]: t for t in t2s}
    if len(t2s) != len(tensors) or al2 != alignment: sys.exit("VERIFY FAIL: tensor count or alignment changed")
    order2, slot2 = slots(t2s, ds2, os.path.getsize(a.dst))
    r = by2[a.tensor]
    if list(r["dims"]) != new_dims or r["type"] != a.type or slot2[a.tensor] != pad_to(blob_size, alignment):
        sys.exit("VERIFY FAIL: replaced tensor's directory entry is not as requested")
    with open(a.src, "rb") as fs, open(a.dst, "rb") as fd, open(a.blob, "rb") as fb:
        fd.seek(ds2 + r["offset"])
        left = blob_size
        while left:
            n = min(CHUNK, left)
            if fd.read(n) != fb.read(n): sys.exit("VERIFY FAIL: replaced payload != blob")
            left -= n
        for t in tensors:
            if t["name"] == a.tensor: continue
            u = by2[t["name"]]
            if list(u["dims"]) != list(t["dims"]) or u["type"] != t["type"] or slot2[t["name"]] != slot[t["name"]]:
                sys.exit("VERIFY FAIL: %s directory entry changed" % t["name"])
            fs.seek(data_start + t["offset"]); fd.seek(ds2 + u["offset"])
            left = slot[t["name"]]
            while left:
                n = min(CHUNK, left)
                if fs.read(n) != fd.read(n): sys.exit("VERIFY FAIL: %s payload differs" % t["name"])
                left -= n
    print("VERIFY OK: %s replaced (type %d, dims %s, %d bytes); all %d other tensors byte-identical"
          % (a.tensor, a.type, new_dims, blob_size, len(tensors) - 1))


if __name__ == "__main__":
    main()
