#!/usr/bin/env python3
"""vendor_kernel.py -- regenerate src/cuda/mmq/tessera_routed_fused_window.cuh from Tessera's source (L255).

    python3 vendor_kernel.py TESSERA_CHECKOUT/src/tessera/serving/csrc/routed_fused_window.cu --rev SHORT_SHA

Keeps every device function byte for byte and replaces only the torch host glue (see VENDOR-TESSERA.md).  Each
replaced line is asserted first, so when upstream moves a seam this fails by name instead of vendoring a
half-edited file; update the line numbers here after reading the new upstream.

Seams as of upstream 37742e0f (2026-10-08, contract v64; the file is 3778 lines, four families):
  68-72      the torch / c10 / ATen includes                                  -> dropped
  1966-1970  max_dynamic_smem_bytes: C10_CUDA_CHECK                           -> TESSERA_HOST_CUDA
  1972-1983  launch_variant: C10_CUDA_CHECK(cudaFuncSetAttribute), LAUNCH_CHECK -> TESSERA_HOST_CUDA
  2016-2026  launch: the three piece-major TORCH_CHECKs                        -> TESSERA_HOST_CHECK
  2054-2057  launch: the run-pair refusal TORCH_CHECK                          -> TESSERA_HOST_REFUSE_PAIR
  2063-2080  check_words / check_slot (torch::Tensor)                          -> dropped
  2081-2109  dense_reduce_kernel                                               kept verbatim
  2110-2131  i32_ptr / f32_ptr / TABLE_DTYPE / check_run_tables (torch)        -> dropped
  2132-3048  the E2M1 (FP4) family, under #if TESSERA_ROUTED_FUSED_FP4         -> dropped (we build FP4=0)
  3049       the anonymous namespace's close                                   kept
  3051-end   the torch host entries and the PYBIND module                      -> dropped (pulsar_tessera.cu)

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
    expect(1966, "int max_dynamic_smem_bytes")
    expect(1968, "C10_CUDA_CHECK(cudaDeviceGetAttribute")
    expect(1977, "C10_CUDA_CHECK(cudaFuncSetAttribute(routed_fused_kernel<FP8, MODE, DENSE, SPLIT, RL, TWO, BMT, PM, PAIRED>,")
    expect(1978, "cudaFuncAttributeMaxDynamicSharedMemorySize, smem));")
    expect(1982, "C10_CUDA_KERNEL_LAUNCH_CHECK();")
    expect(2016, "TORCH_CHECK(!p.piece_major || (FAMILY_MMA8 && FP8),")
    expect(2017, '"the piece-major reader is the E4M3 MMA reader only");')
    expect(2020, "TORCH_CHECK(!k.two && k.r_lo == 4,")
    expect(2021, '"the piece-major reader is the one-run rate-4 routed body only");')
    expect(2026, 'TORCH_CHECK(false, "no piece-major rate-4 launch for this mode/width");')
    expect(2054, 'TORCH_CHECK(false, "the ", (MODE == 2')
    expect(2057, '(DENSE ? RATE_MAX : ROUTED_RATE_MAX)')
    expect(2058, "}")
    expect(2063, "void check_words(const torch::Tensor& words")
    expect(2081, "// DENSE && SPLIT")
    expect(2085, "template <bool FP8>")
    expect(2086, "__global__ void dense_reduce_kernel")
    expect(2109, "}")
    expect(2111, "const int32_t* i32_ptr(const torch::Tensor& t)")
    expect(2132, "#if TESSERA_ROUTED_FUSED_FP4")
    expect(3047, "#endif  // TESSERA_ROUTED_FUSED_FP4")
    expect(3049, "}  // namespace")
    expect(3055, "void routed_fused_forward(")

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
        "//     host entries, upstream 2063-2080 and 2110-2131), the E2M1 family (2132-3048, built only at",
        "//     TESSERA_ROUTED_FUSED_FP4=1) and the torch host entries + PYBIND module (3051-end) are dropped; pulsar's",
        "//     launcher (pulsar_tessera.cu) replaces the entries.  dense_reduce_kernel (upstream 2081-2109) is kept verbatim.",
        "// The includer defines TESSERA_ROUTED_FUSED_FP8 and the three hooks before including this file, and compiles the",
        "// translation unit WITHOUT --use_fast_math (upstream builds -O3 -lineinfo; the SwiGLU epilogue calls expf and",
        "// divides, which fast-math would approximate).",
        "",
    ]
    out += lines(1, 67)
    out += ["// (upstream lines 68-72: the torch / c10 / ATen includes -- dropped, see the header above)"]
    out += lines(73, 1967)
    out += ["    TESSERA_HOST_CUDA(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));"]
    out += lines(1969, 1976)
    out += ["        TESSERA_HOST_CUDA(cudaFuncSetAttribute(routed_fused_kernel<FP8, MODE, DENSE, SPLIT, RL, TWO, BMT, PM, PAIRED>,",
            "   " + src[1977]]
    out += lines(1979, 1981)
    out += ["    TESSERA_HOST_CUDA(cudaGetLastError());"]
    out += lines(1983, 2015)
    out += ['    TESSERA_HOST_CHECK(!p.piece_major || (FAMILY_MMA8 && FP8), "the piece-major reader is the E4M3 MMA reader only");']
    out += lines(2018, 2019)
    out += ['            TESSERA_HOST_CHECK(!k.two && k.r_lo == 4, "the piece-major reader is the one-run rate-4 routed body only");']
    out += lines(2022, 2025)
    out += ['            TESSERA_HOST_CHECK(false, "no piece-major rate-4 launch for this mode/width");']
    out += lines(2027, 2053)
    out += ["    TESSERA_HOST_REFUSE_PAIR(MODE, k.r_lo, k.two, p.tile_words, p.K);"]
    out += lines(2058, 2058)
    out += [""]
    out += lines(2081, 2109)
    out += ["", "}  // namespace", ""]
    with open(OUT, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {os.path.normpath(OUT)}: {len(out)} lines from {a.rev} ({digest[:12]})")


if __name__ == "__main__":
    main()
