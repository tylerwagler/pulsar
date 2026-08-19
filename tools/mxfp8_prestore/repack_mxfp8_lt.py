#!/usr/bin/env python3
"""Offline pre-store of MXFP8 weights (type 38) into MXFP8_LT (type 41).

Task #43: the runtime otherwise de-interleaves + swizzles every type-38 weight
into a cudaMalloc'd device buffer at first use (g_fp8_mx_by_offset), a ~6.4 GiB
double-store of the whole MXFP8 set. This tool bakes that exact device layout
into a copy of the gguf so the engine can point cuBLASLt straight at the mmap.

For each type-38 tensor [out, in] (33 B blocks: [E8M0][32 x E4M3]) it emits, at
the SAME file offset and byte size:

    [ E4M3 data, de-interleaved to [in,out] col-major == [out,in] row-major ]
    [ E8M0 scale, mx_sfoff()-swizzled, zero-padded to rup(out,128) x KBp     ]

byte-for-byte what mxfp8_weight_convert_kernel builds today (verified against a
CUDA dump). The tensor's type field is patched 38 -> 41 in the header. Every
other byte (header, offsets, non-fp8 data) is untouched, so the size is
unchanged and offsets stay 32 B aligned.

Layout equality holds exactly when out % 128 == 0 and (in/32) % 4 == 0 (true for
every shipped weight); the tool asserts it so a bad shape can't silently grow.

Usage:
  repack_mxfp8_lt.py SRC.gguf --list
  repack_mxfp8_lt.py SRC.gguf OUT.gguf            # full repack (copy + patch)
  repack_mxfp8_lt.py SRC.gguf --emit-one NAME D S # dump one tensor's LT bytes
                                                  # (data->D, scale->S) for the
                                                  # CUDA byte-identity check
"""
import argparse
import os
import shutil
import struct
import sys
import time

import numpy as np

ALIGN = 32
T_MXFP8 = 38
T_MXFP8_LT = 41


def parse_gguf(path):
    """Return (list of (name, typ, dims, rel_off, type_field_pos), data_start)."""
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", "not a gguf"
    struct.unpack("<I", f.read(4))                      # version
    nt, = struct.unpack("<Q", f.read(8))
    nkv, = struct.unpack("<Q", f.read(8))

    def rs():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n)

    def rv(t):
        if t == 8:
            return rs()
        if t == 9:
            et, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            return [rv(et) for _ in range(n)]
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        fm = {0: "<b", 1: "<B", 2: "<h", 3: "<H", 4: "<i", 5: "<I", 6: "<f",
              7: "<?", 10: "<q", 11: "<Q", 12: "<d"}
        return struct.unpack(fm[t], f.read(sz[t]))[0]

    for _ in range(nkv):
        rs()
        t, = struct.unpack("<I", f.read(4))
        rv(t)
    tens = []
    for _ in range(nt):
        nb, = struct.unpack("<Q", f.read(8))
        name = f.read(nb).decode()
        nd, = struct.unpack("<I", f.read(4))
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        type_pos = f.tell()                             # <-- file pos of type u32
        typ, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        tens.append((name, typ, dims, off, type_pos))
    data_start = (f.tell() + ALIGN - 1) // ALIGN * ALIGN
    f.close()
    return tens, data_start


def mx_rup(x, n):
    return (x + n - 1) // n * n


def lt_layout(in_dim, out_dim):
    KB = in_dim // 32
    KBp = mx_rup(KB, 4)
    data_bytes = in_dim * out_dim
    scale_bytes = mx_rup(out_dim, 128) * KBp
    old_bytes = out_dim * KB * 33
    return KB, KBp, data_bytes, scale_bytes, old_bytes


def build_scale_dest_index(out_dim, KB, KBp):
    """Vectorized mx_sfoff(o, kb, KBp) for the full (o in [0,out), kb in [0,KB))
    grid -> flat destination index into the zero-padded scale buffer. Mirrors the
    __device__ mx_sfoff in src/cuda/pulsar_cuda_matmul.cu exactly."""
    o = np.arange(out_dim, dtype=np.int64)[:, None]     # (out,1)
    kb = np.arange(KB, dtype=np.int64)[None, :]         # (1,KB)
    return (((o // 128) * (KBp // 4) + (kb // 4)) * 512
            + (o % 32) * 16 + ((o % 128) // 32) * 4 + (kb % 4))  # (out,KB)


def repack_tensor(raw_bytes, in_dim, out_dim):
    """raw_bytes: the type-38 tensor payload (out*KB*33 B). Returns the LT
    payload (data || scale) as a single contiguous bytes object of equal size."""
    KB, KBp, data_bytes, scale_bytes, old_bytes = lt_layout(in_dim, out_dim)
    assert len(raw_bytes) == old_bytes, (len(raw_bytes), old_bytes)
    assert out_dim % 128 == 0 and KB % 4 == 0, (
        f"shape [{in_dim},{out_dim}] is not 128-aligned; LT size would differ "
        f"from type-38 size (out%128={out_dim % 128}, KB%4={KB % 4})")
    assert data_bytes + scale_bytes == old_bytes, (data_bytes, scale_bytes, old_bytes)

    blocks = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(out_dim, KB, 33)
    # data: E4M3 bytes 1..33 of each block -> [out, in] row-major (== [in,out] col)
    data = np.ascontiguousarray(blocks[:, :, 1:]).reshape(out_dim, in_dim)
    # scale: E8M0 byte 0 of each block scattered to its swizzled slot; pad = 0
    scale = np.zeros(scale_bytes, dtype=np.uint8)
    dest = build_scale_dest_index(out_dim, KB, KBp)     # (out,KB)
    scale[dest.reshape(-1)] = blocks[:, :, 0].reshape(-1)
    return data.tobytes() + scale.tobytes()


def dims_in_out(dims):
    assert len(dims) == 2, dims
    return dims[0], dims[1]                              # ne0=in (contig), ne1=out


# ONLY the workhorse weights that route through the LT-aware cuda_fp8_mx_weight
# resolver (src/engine/weights.c tensor_expect_mxfp8): attn q/kv/output, shared
# experts, the output head, and the DSpark support weights (main_proj + its attn
# and shared experts). Any OTHER type-38 tensor (e.g. a future FP8-quantized
# hc_attn_fn / ffn_gate_inp / compressor / indexer projection) takes the PLAIN
# matmul path, which has no MXFP8_LT branch — repacking it would decode to
# garbage, so it must stay type-38. Match on the base name (second-to-last dotted
# token) so both blk.N.* and dspark.N.* are covered.
_WORKHORSE_BASES = frozenset({
    "attn_q_a", "attn_q_b", "attn_kv", "attn_output_a", "attn_output_b",
    "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
    "output", "main_proj",
})


def is_workhorse(name):
    parts = name.split(".")
    base = parts[-2] if len(parts) >= 2 else parts[0]
    return base in _WORKHORSE_BASES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--emit-one", nargs=3, metavar=("NAME", "DATAF", "SCALEF"))
    args = ap.parse_args()

    tens, data_start = parse_gguf(args.src)
    all_fp8 = [(n, t, d, o, tp) for (n, t, d, o, tp) in tens if t == T_MXFP8]
    fp8 = [r for r in all_fp8 if is_workhorse(r[0])]
    skipped = [r for r in all_fp8 if not is_workhorse(r[0])]
    if skipped:
        print(f"NOTE: leaving {len(skipped)} non-workhorse type-38 tensor(s) as "
              f"type-38 (they take the plain matmul path, not MXFP8_LT):")
        for n, t, d, o, tp in skipped:
            print(f"  - {n} dims={d}")

    if args.list:
        tot = 0
        shapes = {}
        for n, t, d, o, tp in fp8:
            in_dim, out_dim = dims_in_out(d)
            _, _, db, sb, ob = lt_layout(in_dim, out_dim)
            tot += ob
            shapes[(in_dim, out_dim)] = shapes.get((in_dim, out_dim), 0) + 1
        print(f"data_start={data_start}  type-38 tensors={len(fp8)}  "
              f"total={tot/1073741824:.3f} GiB")
        for s, c in sorted(shapes.items()):
            print(f"  in={s[0]:6d} out={s[1]:6d}  x{c}")
        return

    if args.emit_one:
        name, dataf, scalef = args.emit_one
        rec = next((r for r in fp8 if r[0] == name), None)
        assert rec, f"{name} not a type-38 tensor"
        n, t, d, o, tp = rec
        in_dim, out_dim = dims_in_out(d)
        _, _, db, sb, ob = lt_layout(in_dim, out_dim)
        with open(args.src, "rb") as f:
            f.seek(data_start + o)
            raw = f.read(ob)
        payload = repack_tensor(raw, in_dim, out_dim)
        with open(dataf, "wb") as f:
            f.write(payload[:db])
        with open(scalef, "wb") as f:
            f.write(payload[db:])
        print(f"emit-one {name}: data={db} scale={sb} bytes")
        return

    assert args.out, "need OUT.gguf (or --list)"
    total = sum(lt_layout(*dims_in_out(d))[4] for _, _, d, _, _ in fp8)
    print(f"repacking {len(fp8)} type-38 tensors ({total/1073741824:.2f} GiB) "
          f"-> MXFP8_LT (type 41)", flush=True)

    if not os.path.exists(args.out):
        print(f"copying {args.src} -> {args.out} ...", flush=True)
        t0 = time.time()
        shutil.copyfile(args.src, args.out)
        print(f"  copied in {time.time()-t0:.0f}s", flush=True)
    else:
        print(f"OUT exists, patching in place: {args.out}", flush=True)
    assert os.path.getsize(args.src) == os.path.getsize(args.out)

    t0 = time.time()
    with open(args.src, "rb") as fin, open(args.out, "r+b") as fout:
        for i, (name, t, d, o, tp) in enumerate(fp8):
            in_dim, out_dim = dims_in_out(d)
            _, _, db, sb, ob = lt_layout(in_dim, out_dim)
            fin.seek(data_start + o)
            raw = fin.read(ob)
            payload = repack_tensor(raw, in_dim, out_dim)
            assert len(payload) == ob
            fout.seek(data_start + o)
            fout.write(payload)
            fout.seek(tp)                                # patch type 38 -> 41
            fout.write(struct.pack("<I", T_MXFP8_LT))
            if (i + 1) % 40 == 0 or i + 1 == len(fp8):
                print(f"  [{i+1}/{len(fp8)}] {name:44s} {ob/1e6:8.1f} MB "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
