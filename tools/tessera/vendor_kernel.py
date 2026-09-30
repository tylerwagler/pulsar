#!/usr/bin/env python3
"""vendor_kernel.py -- regenerate src/cuda/mmq/tessera_routed_fused_window.cuh from Tessera's source (L255).

    python3 vendor_kernel.py TESSERA_CHECKOUT/src/tessera/serving/csrc/routed_fused_window.cu --rev SHORT_SHA

Keeps every device function byte for byte and replaces only the torch host glue (see VENDOR-TESSERA.md).  Each
replaced line is asserted first, so when upstream moves a seam this fails by name instead of vendoring a
half-edited file; update the line numbers here after reading the new upstream.

Made with Tessera by Robert Tand - https://github.com/RobTand/tessera
"""
import argparse
import datetime
import hashlib
import os
import subprocess

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "cuda", "mmq",
                   "tessera_routed_fused_window.cuh")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("--rev", required=True, help="the upstream commit the file comes from (short sha)")
    ap.add_argument("--date", default=None, help="the commit's date, YYYY-MM-DD (default: git log of src)")
    a = ap.parse_args()
    raw = open(a.src, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    date = a.date or subprocess.run(["git", "-C", os.path.dirname(a.src), "log", "-1", "--format=%cs", a.rev],
                                    capture_output=True, text=True, check=True).stdout.strip()
    src = raw.decode().split("\n")

    def lines(first, last):   # 1-based, inclusive
        return src[first - 1:last]

    def expect(n, text):
        if not src[n - 1].strip().startswith(text):
            raise SystemExit(f"upstream line {n} is {src[n - 1]!r}, expected it to start {text!r}: the seam moved")

    expect(58, "#include <torch/extension.h>")
    expect(61, "#include <c10/cuda/CUDAException.h>")
    expect(1107, "int max_dynamic_smem_bytes")
    expect(1109, "C10_CUDA_CHECK(cudaDeviceGetAttribute")
    expect(1118, "C10_CUDA_CHECK(cudaFuncSetAttribute")
    expect(1119, "cudaFuncAttributeMaxDynamicSharedMemorySize")
    expect(1123, "C10_CUDA_KERNEL_LAUNCH_CHECK();")
    expect(1153, 'TORCH_CHECK(false, "the ", (MODE == 2')
    expect(1156, "}")
    expect(1178, "// DENSE && SPLIT")
    expect(1206, "}")

    out = [
        "// VENDORED from Tessera (https://github.com/RobTand/tessera), src/tessera/serving/csrc/routed_fused_window.cu",
        f"// at upstream {a.rev} ({date}), file sha256 {digest}.",
        "// Made with Tessera by Robert Tand - https://github.com/RobTand/tessera",
        "// License: MIT + Tessera Attribution Addendum 1.0 (LicenseRef-Tessera-Attribution-1.0); see src/cuda/mmq/VENDOR-TESSERA.md.",
        "//",
        "// pulsar edits (host glue only -- every device function is byte-for-byte upstream):",
        "//   * the torch/c10 #includes (upstream lines 58-61) are dropped;",
        "//   * the three c10 CUDA checks and the one TORCH_CHECK in max_dynamic_smem_bytes / launch_pair / launch call",
        "//     the includer's hooks TESSERA_HOST_CUDA(expr) and TESSERA_HOST_REFUSE_PAIR(mode, r_lo, two, tile_words, K);",
        "//   * the torch host entries and PYBIND module (upstream lines 1158-1177 and 1208-1520) are dropped; pulsar's",
        "//     launcher (pulsar_tessera.cu) replaces them.  dense_reduce_kernel (upstream 1178-1206) is kept verbatim.",
        "// The includer defines TESSERA_ROUTED_FUSED_FP8 and the two hooks before including this file, and compiles the",
        "// translation unit WITHOUT --use_fast_math (upstream builds -O3 -lineinfo; the SwiGLU epilogue calls expf and",
        "// divides, which fast-math would approximate).",
        "",
    ]
    out += lines(1, 57)
    out += ["// (upstream lines 58-61: the torch / c10 includes -- dropped, see the header above)"]
    out += lines(62, 1108)
    out += ["    TESSERA_HOST_CUDA(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));"]
    out += lines(1110, 1117)
    out += ["        TESSERA_HOST_CUDA(cudaFuncSetAttribute(routed_fused_kernel<FP8, MODE, DENSE, SPLIT, RL, TWO>,",
            src[1118].replace(" " * 36, " " * 47, 1)]
    out += lines(1120, 1122)
    out += ["    TESSERA_HOST_CUDA(cudaGetLastError());"]
    out += lines(1124, 1152)
    out += ["    TESSERA_HOST_REFUSE_PAIR(MODE, k.r_lo, k.two, p.tile_words, p.K);"]
    out += lines(1156, 1156)
    out += [""]
    out += lines(1178, 1206)
    out += ["", "}  // namespace", ""]
    with open(OUT, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {os.path.normpath(OUT)}: {len(out)} lines from {a.rev} ({digest[:12]})")


if __name__ == "__main__":
    main()
