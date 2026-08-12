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
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_dspark_gguf import Reader

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
