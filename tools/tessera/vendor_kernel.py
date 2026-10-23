#!/usr/bin/env python3
"""vendor_kernel.py -- regenerate src/cuda/mmq/tessera_routed_fused_window.cuh from Tessera's source (L255).

    python3 vendor_kernel.py TESSERA_CHECKOUT/src/tessera/serving/csrc/routed_fused_window.cu --rev SHORT_SHA

Keeps every device function byte for byte and replaces only the torch host glue (see VENDOR-TESSERA.md).  Each
replaced line is asserted first, so when upstream moves a seam this fails by name instead of vendoring a
half-edited file; update the line numbers here after reading the new upstream.

Seams as of upstream 4897b634 (2026-10-23, contract v66; the file is 3779 lines, four families;
the v64->v66 drift is the BF16 value family's row scale moving from a folded pre-dot multiply
to an fp32 epilogue -- device code inside the kept region, reflected by no mirror here):
  68-72      the torch / c10 / ATen includes                                  -> dropped
  1964-1968  max_dynamic_smem_bytes: C10_CUDA_CHECK                           -> TESSERA_HOST_CUDA
  1970-1982  launch_variant: C10_CUDA_CHECK(cudaFuncSetAttribute), LAUNCH_CHECK -> TESSERA_HOST_CUDA
  2014-2026  launch: the three piece-major TORCH_CHECKs                        -> TESSERA_HOST_CHECK
  2052-2055  launch: the run-pair refusal TORCH_CHECK                          -> TESSERA_HOST_REFUSE_PAIR
  2061-2078  check_words / check_slot (torch::Tensor)                          -> dropped
  2079-2110  dense_reduce_kernel (grew +3: the BF16 epilogue scale)            kept verbatim
  2111-2132  i32_ptr / f32_ptr / TABLE_DTYPE / check_run_tables (torch)        -> dropped
  2133-3048  the E2M1 (FP4) family, under #if TESSERA_ROUTED_FUSED_FP4         -> dropped (we build FP4=0)
  3050       the anonymous namespace's close                                   kept
  3052-end   the torch host entries and the PYBIND module                      -> dropped (pulsar_tessera.cu)

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

    expect(68, "#include <torch/extension.h>")
    expect(72, "#include <c10/cuda/CUDAException.h>")
    expect(73, "#include <cuda_runtime.h>")
    expect(1964, "int max_dynamic_smem_bytes")
    expect(1966, "C10_CUDA_CHECK(cudaDeviceGetAttribute")
    expect(1975, "C10_CUDA_CHECK(cudaFuncSetAttribute(routed_fused_kernel<FP8, MODE, DENSE, SPLIT, RL, TWO, BMT, PM, PAIRED>,")
    expect(1976, "cudaFuncAttributeMaxDynamicSharedMemorySize, smem));")
    expect(1980, "C10_CUDA_KERNEL_LAUNCH_CHECK();")
    expect(2014, "TORCH_CHECK(!p.piece_major || (FAMILY_MMA8 && FP8),")
    expect(2015, '"the piece-major reader is the E4M3 MMA reader only");')
    expect(2018, "TORCH_CHECK(!k.two && k.r_lo == 4,")
    expect(2019, '"the piece-major reader is the one-run rate-4 routed body only");')
    expect(2024, 'TORCH_CHECK(false, "no piece-major rate-4 launch for this mode/width");')
    expect(2052, 'TORCH_CHECK(false, "the ", (MODE == 2')
    expect(2055, '(DENSE ? RATE_MAX : ROUTED_RATE_MAX)')
    expect(2056, "}")
    expect(2061, "void check_words(const torch::Tensor& words")
    expect(2079, "// DENSE && SPLIT")
    expect(2083, "template <bool FP8>")
    expect(2084, "__global__ void dense_reduce_kernel")
    expect(2110, "}")
    expect(2112, "const int32_t* i32_ptr(const torch::Tensor& t)")
    expect(2133, "#if TESSERA_ROUTED_FUSED_FP4")
    expect(3048, "#endif  // TESSERA_ROUTED_FUSED_FP4")
    expect(3050, "}  // namespace")
    expect(3056, "void routed_fused_forward(")

    out = [
        "// VENDORED from Tessera (https://github.com/RobTand/tessera), src/tessera/serving/csrc/routed_fused_window.cu",
        f"// at upstream {a.rev} ({date}), file sha256 {digest}.",
        "// Made with Tessera by Robert Tand - https://github.com/RobTand/tessera",
        "// License: MIT + Tessera Attribution Addendum 1.0 (LicenseRef-Tessera-Attribution-1.0); see src/cuda/mmq/VENDOR-TESSERA.md.",
        "//",
        "// pulsar edits (host glue only -- every device function is byte-for-byte upstream):",
        "//   * the torch / c10 / ATen #includes (upstream lines 68-72) are dropped;",
        "//   * the c10 CUDA checks in max_dynamic_smem_bytes / launch_variant call the includer's TESSERA_HOST_CUDA(expr);",
        "//   * the three piece-major TORCH_CHECKs in launch call TESSERA_HOST_CHECK(cond, msg), and its run-pair refusal",
        "//     calls TESSERA_HOST_REFUSE_PAIR(mode, r_lo, two, tile_words, K);",
        "//   * check_words / check_slot / i32_ptr / f32_ptr / TABLE_DTYPE / check_run_tables (torch::Tensor helpers of the",
        "//     host entries, upstream 2061-2078 and 2111-2132), the E2M1 family (2133-3048, built only at",
        "//     TESSERA_ROUTED_FUSED_FP4=1) and the torch host entries + PYBIND module (3052-end) are dropped; pulsar's",
        "//     launcher (pulsar_tessera.cu) replaces the entries.  dense_reduce_kernel (upstream 2079-2110) is kept verbatim.",
        "// The includer defines TESSERA_ROUTED_FUSED_FP8 and the three hooks before including this file, and compiles the",
        "// translation unit WITHOUT --use_fast_math (upstream builds -O3 -lineinfo; the SwiGLU epilogue calls expf and",
        "// divides, which fast-math would approximate).",
        "",
    ]
    out += lines(1, 67)
    out += ["// (upstream lines 68-72: the torch / c10 / ATen includes -- dropped, see the header above)"]
    out += lines(73, 1965)
    out += ["    TESSERA_HOST_CUDA(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));"]
    out += lines(1967, 1974)
    out += ["        TESSERA_HOST_CUDA(cudaFuncSetAttribute(routed_fused_kernel<FP8, MODE, DENSE, SPLIT, RL, TWO, BMT, PM, PAIRED>,",
            "   " + src[1975]]
    out += lines(1977, 1979)
    out += ["    TESSERA_HOST_CUDA(cudaGetLastError());"]
    out += lines(1981, 2013)
    out += ['    TESSERA_HOST_CHECK(!p.piece_major || (FAMILY_MMA8 && FP8), "the piece-major reader is the E4M3 MMA reader only");']
    out += lines(2016, 2017)
    out += ['            TESSERA_HOST_CHECK(!k.two && k.r_lo == 4, "the piece-major reader is the one-run rate-4 routed body only");']
    out += lines(2020, 2023)
    out += ['            TESSERA_HOST_CHECK(false, "no piece-major rate-4 launch for this mode/width");']
    out += lines(2025, 2051)
    out += ["    TESSERA_HOST_REFUSE_PAIR(MODE, k.r_lo, k.two, p.tile_words, p.K);"]
    out += lines(2056, 2056)
    out += [""]
    out += lines(2079, 2110)
    out += ["", "}  // namespace", ""]
    with open(OUT, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {os.path.normpath(OUT)}: {len(out)} lines from {a.rev} ({digest[:12]})")


if __name__ == "__main__":
    main()
