#!/usr/bin/env python3
"""Append KV records to a GGUF by rewriting its header and streaming the data.

WHY A REWRITE.  A GGUF header has no slack: the tensor data starts at the first
alignment boundary after the last tensor-info record, so adding even one KV
moves data_start and every byte after it.  Tensor records (names, dims, types,
offsets) are copied verbatim -- offsets are data_start-relative, so they do
not change -- and the data region is streamed straight across.  Nothing is
re-quantized, nothing is reinterpreted: the output's data region is the
input's, byte for byte, which the tool proves by hashing both (the input's
while copying, the output's in a verify pass) and refusing to report success
on a mismatch.

The KV records come from ANOTHER GGUF (typically the template the artifact was
built from) as raw bytes, the way the quantizer's merge takes them
(gguf_take_kvs in dsq_gguf_io.c): the key, its type and its value travel
together, so the tool cannot mistype a value.  A key already present in the
input is refused -- this appends, it does not edit.

USAGE
    gguf_add_kvs.py IN.gguf OUT.gguf --from TEMPLATE.gguf KEY [KEY ...] [--no-verify]
"""
import argparse
import hashlib
import os
import struct
import sys

CHUNK = 64 << 20


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def skip_value(f, t):
    if t in (0, 1, 7):
        f.read(1)
    elif t in (2, 3):
        f.read(2)
    elif t in (4, 5, 6):
        f.read(4)
    elif t in (10, 11, 12):
        f.read(8)
    elif t == 8:
        read_str(f)
    elif t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            skip_value(f, et)
    else:
        raise SystemExit(f"unknown GGUF value type {t}")


def scan_header(path):
    """(version, n_tensors, kv_records{key: raw bytes}, kv_order, tensor_info raw bytes,
    alignment, data_start)"""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF")
        (version,) = struct.unpack("<I", f.read(4))
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        alignment = 32
        kvs, order = {}, []
        for _ in range(n_kv):
            start = f.tell()
            key = read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            if key == "general.alignment" and t == 4:
                (alignment,) = struct.unpack("<I", f.read(4))
            else:
                skip_value(f, t)
            end = f.tell()
            f.seek(start)
            kvs[key] = f.read(end - start)
            order.append(key)
        tinfo_start = f.tell()
        for _ in range(n_tensors):
            read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            f.read(8 * nd + 4 + 8)
        tinfo_end = f.tell()
        f.seek(tinfo_start)
        tinfo = f.read(tinfo_end - tinfo_start)
        data_start = (tinfo_end + alignment - 1) // alignment * alignment
    return version, n_tensors, kvs, order, tinfo, alignment, data_start


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--from", dest="src", required=True, help="GGUF holding the KV records to append")
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--no-verify", action="store_true", help="skip the read-back hash of the output data region")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")

    version, n_tensors, kvs, order, tinfo, alignment, data_start = scan_header(a.inp)
    _, _, src_kvs, _, _, _, _ = scan_header(a.src)
    for k in a.keys:
        if k in kvs:
            raise SystemExit(f"{a.inp} already has {k}; this tool appends, it does not edit")
        if k not in src_kvs:
            raise SystemExit(f"{a.src} has no {k}")
    in_size = os.path.getsize(a.inp)
    data_bytes = max(0, in_size - data_start)   # a template ends with its tensor infos: no data region

    header = bytearray(b"GGUF")
    header += struct.pack("<I", version)
    header += struct.pack("<Q", n_tensors)
    header += struct.pack("<Q", len(order) + len(a.keys))
    for k in order:
        header += kvs[k]
    for k in a.keys:
        header += src_kvs[k]
    header += tinfo
    new_data_start = (len(header) + alignment - 1) // alignment * alignment
    header += b"\0" * (new_data_start - len(header))
    print(f"header: {data_start} -> {new_data_start} B (+{len(a.keys)} KVs: {', '.join(a.keys)}); "
          f"data region {data_bytes / 2**30:.2f} GiB", flush=True)

    h_in = hashlib.sha256()
    with open(a.inp, "rb") as fi, open(a.out, "wb") as fo:
        fo.write(header)
        fi.seek(data_start)
        done = 0
        while True:
            buf = fi.read(CHUNK)
            if not buf:
                break
            h_in.update(buf)
            fo.write(buf)
            done += len(buf)
            if done % (16 << 30) < CHUNK:
                print(f"  copied {done / 2**30:.0f} / {data_bytes / 2**30:.0f} GiB", flush=True)
    out_size = os.path.getsize(a.out)
    if data_bytes and out_size != new_data_start + data_bytes:
        raise SystemExit(f"size mismatch: wrote {out_size}, expected {new_data_start + data_bytes}")
    print(f"input data sha256 {h_in.hexdigest()}", flush=True)
    if a.no_verify:
        print("verify SKIPPED (--no-verify)")
        return
    h_out = hashlib.sha256()
    with open(a.out, "rb") as fo:
        fo.seek(new_data_start)
        while True:
            buf = fo.read(CHUNK)
            if not buf:
                break
            h_out.update(buf)
    if h_out.hexdigest() != h_in.hexdigest():
        raise SystemExit(f"VERIFY FAILED: output data sha256 {h_out.hexdigest()} != input {h_in.hexdigest()}")
    print(f"verify OK: output data region byte-identical ({h_out.hexdigest()})")


if __name__ == "__main__":
    main()
