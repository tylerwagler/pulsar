#!/usr/bin/env python3
"""unmerge_dspark_gguf.py — inverse of merge_dspark_gguf.py.

Strips the appended dspark.* tensors and the dspark KV entries from a merged
ds4 GGUF, recovering the pre-merge main file. Main tensor offsets are relative
to data_start and are preserved untouched; the drafter data region (appended
after all main data) is simply not copied.

Because merge preserves the main file byte-for-byte, unmerge + re-merge is the
cheap path for drafter-only fixes — no re-quantization of the main model.

Usage: unmerge_dspark_gguf.py MERGED.gguf OUT.gguf
"""
import struct
import sys

import merge_dspark_gguf as mdg

MERGE_KEYS = set(mdg.MERGE_KEYS)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    merged_path, out_path = sys.argv[1:3]
    r = mdg.Reader(merged_path)
    sizes = mdg.tensor_sizes(r)

    keep_kv = [(k, raw) for k, raw in r.kv if k not in MERGE_KEYS]
    keep_t, drop_t = [], []
    for i, (name, dims, ttype, off, raw) in enumerate(r.tensors):
        (drop_t if name.startswith("dspark.") else keep_t).append((i, name, off, raw))
    assert drop_t, "no dspark tensors present"
    assert len(r.kv) - len(keep_kv) == len(MERGE_KEYS & {k for k, _ in r.kv}), "kv drop mismatch"

    dspark_min = min(off for _, _, off, _ in drop_t)
    main_end = max(off + sizes[i] for i, _, off, _ in keep_t)
    assert main_end <= dspark_min, f"main data ({main_end}) overlaps drafter region ({dspark_min})"

    out = open(out_path, "wb")
    out.write(b"GGUF" + struct.pack("<I", 3))
    out.write(struct.pack("<Q", len(keep_t)) + struct.pack("<Q", len(keep_kv)))
    for _, raw in keep_kv:
        out.write(raw)
    for _, _, _, raw in keep_t:
        out.write(raw)
    pos = out.tell()
    out.write(b"\x00" * (mdg.align_up(pos, r.alignment) - pos))

    r.f.seek(r.data_pos)
    remaining = dspark_min  # copy [0, dspark_min) of the data region
    while remaining:
        chunk = r.f.read(min(64 << 20, remaining))
        assert chunk, "short read"
        out.write(chunk)
        remaining -= len(chunk)
    out.close()
    print(f"unmerged: kept {len(keep_t)} tensors / {len(keep_kv)} kv, "
          f"dropped {len(drop_t)} dspark tensors, data {dspark_min} bytes")


if __name__ == "__main__":
    main()
