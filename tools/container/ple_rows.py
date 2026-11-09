#!/usr/bin/env python3
"""ple_rows.py -- the Qwen3.8-Flash-Next PLE n-gram table as ONE disk row file (L251 S6; the L242 path).

    ple_rows.py build  --hf DIR --out DIR            # streams the 128 shards once; ~102.4 GB out
    ple_rows.py verify --hf DIR --out DIR [--sample N] [--full]

WHAT IT WRITES
  <out>/ple-l<L>.rows   a 64-byte header, then n_rows records of `row_bytes` bytes: row r of the table the model
                        indexes.  qwen4_exp stores that table as `split_ngram_parts` (128) row-contiguous BF16
                        tensors `layers.<L>.ple.ple_embedding.ngram_embedding.shard_<s>.weight` [2,500,012, 160];
                        transformers' conversion (Concatenate(dim=0), shard_i in NUMERIC order) makes global row
                        r = shard r // rows_per_shard, local row r % rows_per_shard.  The hashed n-gram id the model
                        computes (head offset + remainder mod the head's prime) IS that global row, so the file is the
                        shards' payloads concatenated in numeric order -- a copy, no conversion (native-format
                        transport).  The rows past sum(head vocab sizes) are the checkpoint's padding to
                        make_ngram_vocab_size_divisible_by; they are copied too (never addressed).
  <out>/ple-l<L>.json   the manifest: header fields, the head table (offsets / prime sizes / layer multipliers, from
                        the checkpoint's own I64 buffers), the source snapshot + every shard's tensor offset, and the
                        SHA-256 of every shard's payload as read at build time and of the whole row file.

THE HEADER (64 bytes, little endian) -- the L242 PENGRAM1 layout, version 2:
    0  char[8]  "PENGRAM1"
    8  u32      version            2 (v1 = V4.1 Engram: E4M3 values + E8M0 scales, 264-byte records)
   12  u32      layer              the model layer the table serves (0-based; Qwen PLE = 1)
   16  u64      n_rows
   24  u32      row_bytes          dim * 2 = 320
   28  u32      dim                160 values per row (hidden 2560 / 16 heads)
   32  u32      n_scale            0 (no block scales: BF16 values)
   36  u32      value_dtype        1 = BF16  (v1 files: 0 = E4M3 + E8M0)
   40  u32      n_heads            16 (bigram + trigram x heads_per_ngram 8); the head table is in the manifest
   44  u32      rows_per_part      2,500,012 (the source's split, for provenance)
   48  u8[16]   zero
  A v1 reader (src/engine/engram.cpp) refuses this file by its version -- fail closed, never misread.

VERIFY (the byte check against the SOURCE rows, never against the file's own manifest):
  - header fields vs the checkpoint (shapes, dtype, part count, head table re-read);
  - every shard: SHA-256 of the source payload (re-read from the checkpoint) == SHA-256 of the file's segment
    (--full; ~2 x 102 GB of reads), else the manifest's build-time digests are re-checked against the file only;
  - N random global rows (default 65,536, seeded) read through an independent path -- the safetensors header's
    data_offsets for that row's shard, one pread per row -- byte-equal to the file's record, plus the first and
    last row of every shard;
  - file size == 64 + n_rows * row_bytes.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_source import HFCheckpoint  # noqa: E402

MAGIC = b"PENGRAM1"
VERSION = 2
HDR_BYTES = 64
VALUE_DTYPE_BF16 = 1
CHUNK = 256 << 20


def resolve_hf(path):
    """An HF cache repo dir (models--X) resolves to its single snapshot; a plain checkpoint dir is itself."""
    snaps = glob.glob(os.path.join(path, "snapshots", "*"))
    if snaps:
        if len(snaps) != 1:
            raise SystemExit(f"{path}: {len(snaps)} snapshots, name one")
        return snaps[0]
    return path


class PleSource:
    """The PLE table as the checkpoint stores it, read by shard header."""

    def __init__(self, hf_dir):
        self.hf = HFCheckpoint(resolve_hf(hf_dir))
        cfg = self.hf.config
        if cfg.get("model_type") not in ("qwen4_exp", "qwen4_exp_text"):
            raise SystemExit(f"model_type {cfg.get('model_type')!r}: the PLE row file is a qwen4_exp table")
        ids = cfg["ple_layer_ids"]
        if len(ids) != 1:
            raise SystemExit(f"ple_layer_ids {ids}: one PLE layer expected")
        self.layer = int(ids[0]) - 1                         # 1-based in the config (`layer_idx + 1 in ple_layer_ids`)
        self.n_parts = int(cfg["split_ngram_parts"])
        pre = f"model.language_model.layers.{self.layer}.ple.ple_embedding."
        self.pre = pre
        pat = re.compile(re.escape(pre) + r"ngram_embedding\.shard_(\d+)\.weight$")
        parts = sorted((int(m.group(1)), n) for n in self.hf.names() if (m := pat.match(n)))
        if [p for p, _ in parts] != list(range(self.n_parts)):
            raise SystemExit(f"{pre}ngram_embedding: parts {[p for p, _ in parts][:5]}... are not 0..{self.n_parts - 1}")
        self.parts = [n for _, n in parts]
        shapes = {tuple(self.hf.shape(n)) for n in self.parts}
        dtypes = {self.hf.dtype(n) for n in self.parts}
        if len(shapes) != 1 or dtypes != {"BF16"}:
            raise SystemExit(f"PLE parts: shapes {shapes} dtypes {dtypes}; expected one BF16 shape")
        self.rows_per_part, self.dim = shapes.pop()
        self.n_rows = self.rows_per_part * self.n_parts
        self.row_bytes = self.dim * 2
        n_heads = (int(cfg["ngram_size"]) - 1) * int(cfg["heads_per_ngram"])
        if self.dim * n_heads != int(cfg["ple_embed_dim"]):
            raise SystemExit(f"PLE dim {self.dim} x {n_heads} heads != ple_embed_dim {cfg['ple_embed_dim']}")
        self.n_heads = n_heads
        self.offsets = self._i64(pre + "ngram_heads_offsets")
        self.sizes = self._i64(pre + "ngram_heads_vocab_sizes")
        self.multipliers = self._i64(pre + "layer_multipliers")
        if len(self.offsets) != n_heads or len(self.sizes) != n_heads:
            raise SystemExit(f"head table: {len(self.offsets)} offsets / {len(self.sizes)} sizes for {n_heads} heads")
        run = 0
        for o, s in zip(self.offsets, self.sizes):
            if o != run:
                raise SystemExit(f"head offsets {self.offsets} are not the running sum of the sizes {self.sizes}")
            run += s
        if run > self.n_rows:
            raise SystemExit(f"head table addresses {run} rows; the parts hold {self.n_rows}")
        self.addressed_rows = run

    def _i64(self, name):
        if self.hf.dtype(name) != "I64":
            raise SystemExit(f"{name}: dtype {self.hf.dtype(name)}, expected I64")
        raw = self.hf.raw(name)
        return list(struct.unpack(f"<{len(raw) // 8}q", raw))

    def header(self):
        h = bytearray(HDR_BYTES)
        h[0:8] = MAGIC
        struct.pack_into("<IIQIIIIII", h, 8, VERSION, self.layer, self.n_rows, self.row_bytes, self.dim, 0,
                         VALUE_DTYPE_BF16, self.n_heads, self.rows_per_part)
        return bytes(h)

    def span(self, part):
        path, off, n = self.hf.span(self.parts[part])
        if n != self.rows_per_part * self.row_bytes:
            raise SystemExit(f"{self.parts[part]}: {n} bytes, expected {self.rows_per_part * self.row_bytes}")
        return path, off, n


def paths(out_dir, layer):
    return os.path.join(out_dir, f"ple-l{layer}.rows"), os.path.join(out_dir, f"ple-l{layer}.json")


def copy_hash(src_path, off, n, dst, h_part, h_file):
    with open(src_path, "rb") as f:
        f.seek(off)
        left = n
        while left:
            b = f.read(min(left, CHUNK))
            if not b:
                raise SystemExit(f"{src_path}: short read at {off + n - left}")
            h_part.update(b)
            h_file.update(b)
            if dst is not None:
                dst.write(b)
            left -= len(b)


def cmd_build(a):
    src = PleSource(a.hf)
    os.makedirs(a.out, exist_ok=True)
    rows_p, man_p = paths(a.out, src.layer)
    hdr = src.header()
    h_file = hashlib.sha256(hdr)
    part_sha, part_src = [], []
    t0 = time.time()
    with open(rows_p + ".tmp", "wb") as out:
        out.write(hdr)
        for p in range(src.n_parts):
            path, off, n = src.span(p)
            hp = hashlib.sha256()
            copy_hash(path, off, n, out, hp, h_file)
            part_sha.append(hp.hexdigest())
            part_src.append({"tensor": src.parts[p], "file": os.path.basename(path), "offset": off, "bytes": n})
            done = (p + 1) * n
            el = time.time() - t0
            print(f"  part {p + 1:3d}/{src.n_parts}  {done / 1e9:7.1f} GB  {done / el / 1e6:6.0f} MB/s", flush=True)
        out.flush()
        os.fsync(out.fileno())
    size = os.path.getsize(rows_p + ".tmp")
    want = HDR_BYTES + src.n_rows * src.row_bytes
    if size != want:
        raise SystemExit(f"{rows_p}.tmp: {size} bytes, expected {want}")
    os.replace(rows_p + ".tmp", rows_p)
    man = {
        "format": "pulsar-ple-rows", "magic": MAGIC.decode(), "version": VERSION, "header_bytes": HDR_BYTES,
        "layer": src.layer, "n_rows": src.n_rows, "row_bytes": src.row_bytes, "dim": src.dim, "n_scale": 0,
        "value_dtype": "bf16", "n_heads": src.n_heads, "rows_per_part": src.rows_per_part, "n_parts": src.n_parts,
        "addressed_rows": src.addressed_rows, "padding_rows": src.n_rows - src.addressed_rows,
        "head_offsets": src.offsets, "head_vocab_sizes": src.sizes, "layer_multipliers": src.multipliers,
        "rows_bytes": size, "sha256": h_file.hexdigest(), "part_sha256": part_sha,
        "source": {"dir": resolve_hf(a.hf), "parts": part_src},
    }
    with open(man_p, "w") as f:
        json.dump(man, f, indent=1)
        f.write("\n")
    print(f"ple_rows: layer {src.layer}: {src.n_rows:,} rows x {src.row_bytes} B = {size / 1e9:.2f} GB "
          f"in {time.time() - t0:.0f} s; sha256 {h_file.hexdigest()}")
    return 0


def cmd_verify(a):
    src = PleSource(a.hf)
    rows_p, man_p = paths(a.out, src.layer)
    man = json.load(open(man_p))
    fails = []

    def check(ok, what):
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}", flush=True)
        if not ok:
            fails.append(what)

    size = os.path.getsize(rows_p)
    check(size == HDR_BYTES + src.n_rows * src.row_bytes, f"file size {size} == 64 + {src.n_rows} x {src.row_bytes}")
    with open(rows_p, "rb") as f:
        hdr = f.read(HDR_BYTES)
    check(hdr == src.header(), "header == the one the checkpoint implies")
    for k, v in (("n_rows", src.n_rows), ("row_bytes", src.row_bytes), ("dim", src.dim), ("layer", src.layer),
                 ("head_offsets", src.offsets), ("head_vocab_sizes", src.sizes),
                 ("layer_multipliers", src.multipliers)):
        check(man.get(k) == v, f"manifest {k} == checkpoint")

    # sampled rows through an independent path: the shard's safetensors header, one pread per row
    rng = random.Random(a.seed)
    want = sorted({rng.randrange(src.n_rows) for _ in range(a.sample)}
                  | {p * src.rows_per_part for p in range(src.n_parts)}
                  | {(p + 1) * src.rows_per_part - 1 for p in range(src.n_parts)}
                  | set(o for o in src.offsets))
    bad = 0
    fds = {}
    with open(rows_p, "rb") as rf:
        for r in want:
            p, local = divmod(r, src.rows_per_part)
            name = src.parts[p]
            spath = os.path.join(src.hf.dir, src.hf.weight_map[name])
            if spath not in fds:
                fds[spath] = open(spath, "rb")
            sf = fds[spath]
            sf.seek(0)
            (n,) = struct.unpack("<Q", sf.read(8))
            sh = json.loads(sf.read(n))[name]
            if sh["dtype"] != "BF16" or sh["shape"] != [src.rows_per_part, src.dim]:
                bad += 1
                continue
            sf.seek(8 + n + sh["data_offsets"][0] + local * src.row_bytes)
            srow = sf.read(src.row_bytes)
            rf.seek(HDR_BYTES + r * src.row_bytes)
            if rf.read(src.row_bytes) != srow:
                bad += 1
    for fh in fds.values():
        fh.close()
    check(bad == 0, f"{len(want):,} rows (random + every part's first/last + every head's first) byte-equal "
                    f"to the source ({bad} differ)")

    if a.full:
        t0 = time.time()
        h_file = hashlib.sha256(hdr)
        nb = 0
        with open(rows_p, "rb") as rf:
            rf.seek(HDR_BYTES)
            for p in range(src.n_parts):
                path, off, n = src.span(p)
                hs, hf_ = hashlib.sha256(), hashlib.sha256()
                copy_hash(path, off, n, None, hs, hashlib.sha256())
                left = n
                while left:
                    b = rf.read(min(left, CHUNK))
                    hf_.update(b)
                    h_file.update(b)
                    left -= len(b)
                if hs.digest() != hf_.digest() or hs.hexdigest() != man["part_sha256"][p]:
                    nb += 1
                    print(f"  FAIL  part {p}: source {hs.hexdigest()[:16]} file {hf_.hexdigest()[:16]} "
                          f"manifest {man['part_sha256'][p][:16]}", flush=True)
                if p % 16 == 15:
                    print(f"    {p + 1}/{src.n_parts} parts hashed, {time.time() - t0:.0f} s", flush=True)
        check(nb == 0, f"all {src.n_parts} parts: SHA-256(source payload) == SHA-256(file segment) == manifest")
        check(h_file.hexdigest() == man["sha256"], "whole-file SHA-256 == manifest")
    print("VERIFY", "PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("build", cmd_build), ("verify", cmd_verify)):
        p = sub.add_parser(name)
        p.add_argument("--hf", required=True, help="the BF16 qwen4_exp checkpoint (dir or HF cache repo dir)")
        p.add_argument("--out", required=True)
        if name == "verify":
            p.add_argument("--sample", type=int, default=65536)
            p.add_argument("--seed", type=int, default=251)
            p.add_argument("--full", action="store_true", help="hash every part on both sides (~2 x 102 GB read)")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
