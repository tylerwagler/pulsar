#!/usr/bin/env python3
"""merge_dspark_gguf.py — splice the DSpark drafter into the main ds4 GGUF.

Appends the support model's `dspark.*` tensors and the metadata keys the
engine's dspark_weights_bind() reads (`deepseek_v4_dspark.embedding_length`,
`dspark.target_layer_ids.{0,1,2}`) to the main model file, producing one
self-contained release artifact. The engine auto-enables DSpark when it sees
`dspark.main_proj.weight` in the main model.

Main-file tensor offsets are preserved byte-for-byte (the data blob is copied
verbatim); drafter tensors are appended with fresh 32-byte-aligned offsets.

Usage:
  merge_dspark_gguf.py MAIN.gguf DSPARK.gguf OUT.gguf
"""
import struct
import sys

ALIGN = 32

# kv value type ids -> fixed size (bytes); 8=string, 9=array handled apart
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

MERGE_KEYS = (
    "deepseek_v4_dspark.embedding_length",
    "dspark.target_layer_ids.0",
    "dspark.target_layer_ids.1",
    "dspark.target_layer_ids.2",
)


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")
        assert self.f.read(4) == b"GGUF", f"{path}: not a GGUF"
        self.version = self.u32()
        assert self.version == 3, f"{path}: GGUF v{self.version} unsupported"
        self.n_tensors = self.u64()
        self.n_kv = self.u64()
        self.kv = []       # (key, raw_bytes_of_entire_entry)
        for _ in range(self.n_kv):
            start = self.f.tell()
            key = self.s()
            t = self.u32()
            self.skip_val(t)
            end = self.f.tell()
            self.f.seek(start)
            self.kv.append((key, self.f.read(end - start)))
        self.tensors = []  # (name, dims, type, offset, raw_info_bytes)
        for _ in range(self.n_tensors):
            start = self.f.tell()
            name = self.s()
            nd = self.u32()
            dims = [self.u64() for _ in range(nd)]
            ttype = self.u32()
            off = self.u64()
            end = self.f.tell()
            self.f.seek(start)
            self.tensors.append((name, dims, ttype, off, self.f.read(end - start)))
        self.alignment = ALIGN
        for key, raw in self.kv:
            if key == "general.alignment":
                # last 4 bytes of a u32 kv entry are the value
                self.alignment = struct.unpack("<I", raw[-4:])[0]
        self.data_pos = align_up(self.f.tell(), self.alignment)
        self.f.seek(0, 2)
        self.file_size = self.f.tell()

    def u32(self): return struct.unpack("<I", self.f.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.f.read(8))[0]
    def s(self):   return self.f.read(self.u64()).decode()

    def skip_val(self, t):
        if t == 8:
            self.f.read(self.u64())
        elif t == 9:
            et = self.u32()
            n = self.u64()
            if et == 8:
                for _ in range(n):
                    self.f.read(self.u64())
            else:
                self.f.read(SCALAR[et] * n)
        else:
            self.f.read(SCALAR[t])


def align_up(v, a):
    return (v + a - 1) // a * a


def enc_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def tensor_sizes(r):
    """Size of each tensor's data = gap to the next offset (last: to EOF)."""
    order = sorted(range(len(r.tensors)), key=lambda i: r.tensors[i][3])
    sizes = [0] * len(r.tensors)
    for k, i in enumerate(order):
        off = r.tensors[i][3]
        nxt = r.tensors[order[k + 1]][3] if k + 1 < len(order) else r.file_size - r.data_pos
        sizes[i] = nxt - off
    return sizes


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main_path, dspark_path, out_path = sys.argv[1:4]
    m = Reader(main_path)
    d = Reader(dspark_path)
    assert m.alignment == ALIGN and d.alignment == ALIGN, "unexpected alignment"

    if any(name.startswith("dspark.") for name, *_ in m.tensors):
        sys.exit(f"{main_path} already contains dspark.* tensors")
    kv_add = []
    have = {k for k, _ in m.kv}
    for want in MERGE_KEYS:
        raw = next((raw for k, raw in d.kv if k == want), None)
        if raw is None:
            # No silent skip: a merged artifact missing target_layer_ids fails
            # dspark_weights_bind() at load. build_dspark_template emits them.
            sys.exit(f"{dspark_path}: missing required kv {want}")
        if want in have:
            sys.exit(f"{main_path} already has kv {want}")
        kv_add.append(raw)

    d_sizes = tensor_sizes(d)
    main_data_size = m.file_size - m.data_pos

    # lay out drafter tensors after the main blob, in original offset order
    d_order = sorted(range(len(d.tensors)), key=lambda i: d.tensors[i][3])
    new_off = {}
    cur = align_up(main_data_size, ALIGN)
    for i in d_order:
        new_off[i] = cur
        cur = align_up(cur + d_sizes[i], ALIGN)

    out = open(out_path, "wb")
    out.write(b"GGUF")
    out.write(struct.pack("<I", 3))
    out.write(struct.pack("<Q", m.n_tensors + d.n_tensors))
    out.write(struct.pack("<Q", m.n_kv + len(kv_add)))
    for _, raw in m.kv:
        out.write(raw)
    for raw in kv_add:
        out.write(raw)
    for _, _, _, _, raw in m.tensors:
        out.write(raw)
    for i, (name, dims, ttype, _off, _raw) in enumerate(d.tensors):
        out.write(enc_str(name))
        out.write(struct.pack("<I", len(dims)))
        for dim in dims:
            out.write(struct.pack("<Q", dim))
        out.write(struct.pack("<I", ttype))
        out.write(struct.pack("<Q", new_off[i]))
    pad = align_up(out.tell(), ALIGN) - out.tell()
    out.write(b"\0" * pad)

    # main data blob, verbatim
    m.f.seek(m.data_pos)
    copied = 0
    while copied < main_data_size:
        chunk = m.f.read(min(64 << 20, main_data_size - copied))
        assert chunk
        out.write(chunk)
        copied += len(chunk)
    # drafter tensors at their new aligned offsets
    data_start_check = out.tell() - main_data_size
    for i in d_order:
        want_pos = data_start_check + new_off[i]
        out.write(b"\0" * (want_pos - out.tell()))
        d.f.seek(d.data_pos + d.tensors[i][3])
        left = d_sizes[i]
        while left:
            chunk = d.f.read(min(64 << 20, left))
            assert chunk
            out.write(chunk)
            left -= len(chunk)
    out.close()
    print(f"wrote {out_path}: {m.n_tensors}+{d.n_tensors} tensors, "
          f"{m.n_kv}+{len(kv_add)} kv, {cur / 1e9:.2f} GB data")


if __name__ == "__main__":
    main()
