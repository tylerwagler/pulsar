#!/usr/bin/env python3
"""Release gate: no tensor may ship in a PLAIN type that has a pre-formatted twin.

Three of our quant types are pure byte permutations of a plain predecessor --
same dims, same byte count, bit-identical content, but laid out the way the
device wants it:

    16 IQ2_XXS   -> 42 IQ2_XXS_SOA / 43 IQ2_XXS_MMQ
    38 FP8_E4M3  -> 41 MXFP8_LT      (workhorse weights only)
    39 MXFP4     -> 40 CUTLASS_MXFP4 (routed experts)

Shipping the plain form is never a correctness bug, which is exactly why it
survives review: the engine converts at first use instead. The cost is silent --
a cudaMalloc'd second copy of every affected tensor alongside the mmap. On
2026-08-12 the drafter was found shipping all 25 of its dense weights as plain
type 38, a 0.429 GiB double-store that had been resident since the drafter was
first built, because the drafter is built from a separate pinned type table
(gguf-tools/dspark_type_flags.txt) that nobody re-checked when the main model
moved to MXFP8_LT.

This gate exists so that class of omission fails a release instead of being
found by a stranger reading a type census a year later.

Exit 0 = clean, 1 = a plain twin is present (message names the tensors).

  audit_artifact_types.py MODEL.gguf            # gate (quiet unless it fails)
  audit_artifact_types.py MODEL.gguf --census   # full type breakdown
"""
import argparse
import collections
import struct
import sys

ALIGN = 32

# kv value type ids -> fixed size (bytes); 8=string, 9=array handled apart
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def align_up(v, a):
    return (v + a - 1) // a * a


class Reader:
    """Full GGUF reader keeping raw kv/info bytes. Moved here from the retired
    merge_dspark_gguf.py (superseded by deepseek4-quantize --dspark-template);
    this gate was its last consumer."""

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

T_IQ2_XXS, T_FP8_E4M3, T_MXFP4 = 16, 38, 39
T_CUTLASS_MXFP4, T_MXFP8_LT, T_IQ2_SOA, T_IQ2_MMQ = 40, 41, 42, 43

NAMES = {0: "F32", 1: "F16", 16: "IQ2_XXS(plain)", 26: "I32", 30: "BF16",
         38: "FP8_E4M3(plain)", 39: "MXFP4(plain)", 40: "CUTLASS_MXFP4",
         41: "MXFP8_LT", 42: "IQ2_XXS_SOA", 43: "IQ2_XXS_MMQ"}

# Mirrors _WORKHORSE_BASES in tools/mxfp8_prestore/repack_mxfp8_lt.py and
# is_mxfp8_lt_workhorse() in gguf-tools/quantize/dsq_names.c. Keep all three
# in sync -- a name missing here makes this gate silently permissive.
_WORKHORSE_BASES = frozenset({
    "attn_q_a", "attn_q_b", "attn_kv", "attn_output_a", "attn_output_b",
    "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
    "output", "main_proj",
})


def base_of(name):
    parts = name.split(".")
    return parts[-2] if len(parts) >= 2 else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--census", action="store_true", help="print the full type breakdown")
    a = ap.parse_args()

    r = Reader(a.gguf)
    counts = collections.Counter()
    offenders = []
    for name, dims, typ, off, _ in r.tensors:
        counts[typ] += 1
        if typ == T_FP8_E4M3 and base_of(name) in _WORKHORSE_BASES:
            offenders.append((name, typ, T_MXFP8_LT))
        elif typ == T_MXFP4:
            offenders.append((name, typ, T_CUTLASS_MXFP4))
        elif typ == T_IQ2_XXS:
            offenders.append((name, typ, T_IQ2_MMQ))

    if a.census:
        print("%-20s %6s" % ("type", "count"))
        for t, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            print("%-20s %6d" % (NAMES.get(t, "type%d" % t), c))
        print()

    if not offenders:
        print("artifact-type audit: PASS (%d tensors, no plain twins)" % len(r.tensors))
        return 0

    print("artifact-type audit: FAIL -- %d tensor(s) ship a PLAIN type that has a "
          "pre-formatted twin.\nEach one costs a runtime convert plus a second "
          "device copy beside the mmap." % len(offenders), file=sys.stderr)
    by_kind = collections.Counter((o[1], o[2]) for o in offenders)
    for (got, want), n in by_kind.items():
        print("  %-18s -> %-16s  x%d" % (NAMES.get(got), NAMES.get(want), n), file=sys.stderr)
    for name, got, want in offenders[:10]:
        print("      %s" % name, file=sys.stderr)
    if len(offenders) > 10:
        print("      ... and %d more" % (len(offenders) - 10), file=sys.stderr)
    print("\nFix: repack with tools/mxfp8_prestore/repack_mxfp8_lt.py (type 38 -> 41),\n"
          "or rebuild with the correct --tensor-type pins "
          "(gguf-tools/dspark_type_flags.txt for the drafter).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
