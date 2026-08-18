# Vendored llama.cpp mmq kernels

> ## ⚠ 2026-08-18: THE VENDORED KERNELS ARE GONE. READ THIS FIRST.
>
> **10,135 lines across 17 files were deleted** (ledger L066).  Everything below
> that describes `mmq.cuh`, `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh`,
> `vecdotq.cuh`, `mmvq.*`, `quantize.*` or the `mmq-config-*` family is
> HISTORICAL — those files no longer exist.
>
> **Why.** The routed experts moved to E4M3 activations, and the IQ2 GEMM now
> unpacks weights to E4M3 in-register and issues a block-scaled MXFP8 MMA
> (`ds4_mmq_d2r.cu`, ours).  The path that could still have reached upstream's
> int8 `mul_mat_q` makes D2R a hard requirement and *fails closed*, so the
> vendored kernel family became unreachable **by construction**.  Verified
> before deletion with the linker's own view: `mul_mat_q_case` had 3 definitions
> and **0 undefined references**, as did `ds4_mmq_q8_0_aligned_dense_vec`,
> `ds4_mmq_iq2_xxs_aligned_bytes` and `quantize_mmq_q8_1_cuda`.
>
> **What is still vendored, and it is now a short list:**
>
> | file | why it survives |
> |---|---|
> | `mma.cuh` | `ggml_cuda_mma::tile` / `load_ldmatrix`, used by our D2R GEMM |
> | `mmid.cu` + `mmid.cuh` | `#include`d by `ds4_mmid.cu` because upstream's `launch_mm_ids_helper<N>` is file-static |
> | `ggml-common.h` | `block_iq2_xxs` and friends -- the on-disk quant layouts |
> | `ggml*.h`, `vendors/` | redirect headers the above need |
>
> **`common.cuh` and `unary.cuh` are GONE (2026-08-18, L066 step 3.)**  `unary.cuh`
> was 114 lines of declarations for functions we never implemented or called --
> its only consumer, `mmvq.cu`, was deleted in step 2, and after that nothing but
> this document referenced it.  `common.cuh` was 1,669 lines held open by
> **nineteen** symbols, and the two that mattered were `ggml_backend_cuda_context`
> and `ggml_cuda_pool_alloc`: the MoE entries took per-call scratch from the ggml
> pool.  They now take it from the engine's own arena on a dedicated scratch slot
> (`CUDA_SCRATCH_MMQ` -- the slot exists because the MoE launcher holds an arena
> across the call, so sharing one region would alias live buffers).  With the pool
> gone the remaining surface is small enough to state: device info, `CUDA_CHECK`,
> five constants, two arch predicates, two warp reduces and `memcpy_1` -- and it
> now lives in `ds4_cuda_env.cuh` (256 lines), which is still derived work and
> still tracked below.
>
> **Constants that outlived their files** now live in `ds4_act_block.cuh`:
> `block_mx_act_mmq` (renamed from `block_q8_1_mmq`; it holds E4M3, not q8_1),
> `DS4_ACT_BLOCK_VALS` (was `QK8_1_MMQ`), `DS4_ACT_QUANT_BLOCK` (was
> `CUDA_QUANTIZE_BLOCK_SIZE_MMQ`) and `MMQ_DP4A_MAX_BATCH_SIZE`, which is still
> read by the live `ds4_mmq_should_use` gate and is kept at upstream's value so
> the routing decision is unchanged by the deletion.
>
> **A re-vendor is now much smaller, and the rename conflict noted below is
> moot** for `mmq.cuh` and `quantize.cu` — both are deleted.

This directory contains source files copied verbatim from
[llama.cpp's `ggml-cuda` backend](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda),
plus a thin ds4-side adapter (`ds4_ggml_stubs.{h,cu}` and `ds4_mmq.{h,cu}`)
that lets the templated CUDA kernels compile and link without the full ggml
runtime.

## Why these files are vendored, not submoduled

ds4 is a flat, self-contained C/CUDA codebase with no third-party build
dependencies. A submodule would force consumers to clone, configure, and
build all of llama.cpp just to use a few quantized-matmul kernels. Vendoring
keeps ds4 self-contained at the cost of a periodic re-sync.

## Upstream pin

| Field         | Value                                                                                                              |
|---------------|--------------------------------------------------------------------------------------------------------------------|
| Source        | https://github.com/ggml-org/llama.cpp                                                                              |
| Commit        | `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`                                                                         |
| Commit date   | 2026-05-14                                                                                                         |
| Commit title  | `ggml-hexagon: cpy: add contiguous fast-path in reshape copy (#23076)`                                             |
| License       | MIT (`LICENSE` at the repository root, copyright "2023-2026 The ggml authors"). Compatible with ds4's MIT license. |

## File inventory

> ⚠ The `Status` column below is **as-imported and partly stale**: `mmq.cuh`,
> `mmid.cu` and `mmid.cuh` are marked verbatim but carry ds4 patches.  See
> **ds4 local modifications (measured 2026-08-13)** further down for the
> figures that were actually diffed.

| File                  | Origin in llama.cpp                          | Status                                                                   | Lines |
|-----------------------|----------------------------------------------|--------------------------------------------------------------------------|-------|
| `mmq.cuh`             | `ggml/src/ggml-cuda/mmq.cuh`                 | verbatim                                                                 |  4176 |
| `mma.cuh`             | `ggml/src/ggml-cuda/mma.cuh`                 | verbatim                                                                 |  1456 |
| `vecdotq.cuh`         | `ggml/src/ggml-cuda/vecdotq.cuh`             | verbatim                                                                 |  1317 |
| `quantize.cuh`        | `ggml/src/ggml-cuda/quantize.cuh`            | verbatim                                                                 |    41 |
| `quantize.cu`         | `ggml/src/ggml-cuda/quantize.cu`             | verbatim                                                                 |   443 |
| `mmid.cuh`            | `ggml/src/ggml-cuda/mmid.cuh`                | verbatim                                                                 |     5 |
| `mmid.cu`             | `ggml/src/ggml-cuda/mmid.cu`                 | verbatim                                                                 |   164 |
| `mmvq.cuh`            | `ggml/src/ggml-cuda/mmvq.cuh`                | patched (Step 6): `mul_mat_vec_q_switch_type` proto exposed; ggml-tensor entries gated on `DS4_MMVQ_INCLUDE_GGML_ENTRIES` | ~36 |
| `mmvq.cu`             | `ggml/src/ggml-cuda/mmvq.cu`                 | patched: `mul_mat_vec_q_switch_type` promoted from `static`; `ggml_cuda_mul_mat_vec_q` + `ggml_cuda_op_mul_mat_vec_q` gated on `DS4_MMVQ_INCLUDE_GGML_ENTRIES` | 1163 |
| ~~`unary.cuh`~~       | ~~`ggml/src/ggml-cuda/unary.cuh`~~           | **DELETED 2026-08-18**: its consumer `mmvq.cu` went in L066 step 2, after which nothing included it | — |
| ~~`common.cuh`~~      | `ggml/src/ggml-cuda/common.cuh`              | **DELETED 2026-08-18**: replaced by `ds4_cuda_env.cuh`, which carries the nineteen live symbols. Upstream's NVIDIA arms verbatim; the AMD/HIP/MUSA halves dropped as unreachable (`cc` is read from the device, and we run GB10) | — |
| `ds4_cuda_env.cuh`    | `ggml/src/ggml-cuda/common.cuh` (extract)    | the reachable subset of `common.cuh`; see its header for exactly what was dropped | 256 |
| `ggml-common.h`       | `ggml/src/ggml-common.h` (extract)           | **TRIMMED 2026-08-18**: an empty-shim compile of every built MMQ TU found exactly two things referenced -- `QK8_1` and the `iq2xxs_grid` table. The table is carried BYTE-IDENTICAL (diffed); the other ~1,600 lines were block layouts and grids for quant types this engine does not run. ⚠ the table is EMITTED, not declared -- it needs `GGML_COMMON_IMPL_CUDA` defined before the include, and vanishes silently without it |   103 |
| `vendors/cuda.h`      | `ggml/src/ggml-cuda/vendors/cuda.h`          | verbatim                                                                 |    28 |
| `ggml.h`              | (new)                                        | redirect to `ds4_ggml_stubs.h`                                           |     5 |
| `ggml-impl.h`         | (new)                                        | redirect to `ds4_ggml_stubs.h`                                           |     5 |
| `ggml-cuda.h`         | (new)                                        | redirect to `ds4_ggml_stubs.h`                                           |     5 |
| `ds4_ggml_stubs.h`    | (new)                                        | shim: ggml_type enum, macros, info struct, type_size lookups             | ~280 |
| `ds4_ggml_stubs.cu`   | (new)                                        | shim impls: `ggml_cuda_info`, naive pool, `ggml_backend_cuda_context::*` | ~110 |
| `ds4_mmq.h`           | (new)                                        | public C ABI for ds4 to call                                             |  ~70 |
| `ds4_mmq.cu`          | (new)                                        | host wrappers; Phase 0 instantiates `mul_mat_q_case<Q8_0>` only          | ~120 |

**Total vendored:** ~11,000 lines of CUDA. **Total shim/adapter:** ~600 lines.

## What llama.cpp's `mmq.cu` does that we don't vendor

The upstream `ggml/src/ggml-cuda/mmq.cu` (372 lines) is the ggml-backend
dispatch entry. It talks to `ggml_tensor` / `ggml_backend_buffer_get_usage`,
does its own activation Q8_1 quantization, and wires into the ggml op graph.
We **don't vendor it.** Instead `ds4_mmq.cu` provides ds4-style entries
that bypass the ggml graph and call the per-type `mul_mat_q_case<T>` directly.

## Symbol-resolution table

Symbols the vendored files reference, and how they resolve in this directory:

| Symbol category                                | Resolution                                                                                       |
|------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `GGML_ASSERT`, `GGML_ABORT`, `GGML_UNUSED`     | Macros in `ds4_ggml_stubs.h`                                                                     |
| `GGML_PAD`, `GGML_UNUSED_VARS`                 | Macros in `ds4_ggml_stubs.h`                                                                     |
| `GGML_CUDA_CC_*` constants                     | Defined in `ds4_cuda_env.cuh` (the five still referenced)                                        |
| `GGML_TYPE_*` enum values                      | Defined in `ds4_ggml_stubs.h::enum ggml_type`                                                    |
| `GGML_CUDA_MAX_DEVICES`, `GGML_CUDA_NAME`      | Macros in `ds4_ggml_stubs.h`                                                                     |
| `GGML_MAX_DIMS`, `GGML_MAX_SRC`                | Macros in `ds4_ggml_stubs.h`                                                                     |
| `block_q8_0`, `block_q2_K`, `block_iq2_xxs`    | Defined in vendored `ggml-common.h` (gated by `GGML_COMMON_IMPL_CUDA`)                           |
| `ggml_half`, `ggml_half2`                      | `uint16_t` typedefs in `ds4_ggml_stubs.h`                                                        |
| `ggml_type_size`, `ggml_blck_size`             | Inline implementations in `ds4_ggml_stubs.h`                                                     |
| `ggml_cuda_info()`                             | Singleton in `ds4_ggml_stubs.cu`, populated via `cudaGetDeviceProperties`                        |
| `ggml_cuda_get_device`, `ggml_cuda_set_device` | Thin wrappers in `ds4_ggml_stubs.cu`                                                             |
| `ggml_time_us`                                 | `std::chrono::steady_clock` in `ds4_ggml_stubs.cu`                                               |
| ~~`ggml_cuda_pool`, `ggml_cuda_pool_alloc`~~   | **GONE 2026-08-18**: MMQ scratch is the `CUDA_SCRATCH_MMQ` arena slot; `ds4_naive_pool` deleted   |
| ~~`ggml_backend_cuda_context`~~                | **GONE 2026-08-18**: existed only to hand out `pool()`                                            |
| `ggml_tensor`                                  | Forward declaration in `ds4_ggml_stubs.h`. Never dereferenced - only held as pointer.            |
| `ggml_glu_op`                                  | Enum stub in `ds4_ggml_stubs.h` (unused by ds4; kept so the stub header compiles)                |
| `CUDA_CHECK`, `CUBLAS_CHECK`                   | Macros in `ds4_ggml_stubs.h`                                                                     |
| `ggml_cuda_launch_mm_ids_helper`               | Defined in vendored `mmid.cu`                                                                    |
| `ggml_cuda_should_use_mmq`                     | Defined in vendored `mmq.cu` - **we re-implement in `ds4_mmq.cu`** since we don't vendor mmq.cu  |
| `ggml_cuda_mul_mat_q*`                         | Upstream's host entry. **Replaced** by `ds4_mmq_q8_0_dense` and family.                          |

## Things we deliberately do NOT support

- **CUDA graphs.** ds4 manages its own streams; `USE_CUDA_GRAPH` is undefined.
- **HIP / MUSA / AMD backends.** vendored code's HIP/MUSA `#ifdef` branches are dead in our build but kept for upstream diff-cleanliness.
- **The full `ggml_tensor` type.** No tensor introspection - shapes and strides come in via raw arguments to `ds4_mmq_*`.
- **`ggml_op` graph evaluation.** We call kernels directly.

## ds4 local modifications (measured 2026-08-13 against the pin)

⚠ **The "verbatim" labels in the file inventory above are not all accurate.**
Diffing every vendored file against pinned `5c0e946` shows we carry
**+480 / −24 lines** of ds4 patches across five files.  Only the `mmvq`
pair was previously documented; the `mmq.cuh` and `mmid` patches were not,
which is exactly how a re-vendor turns into a surprise.

⚠ **`block_q8_1_mmq` is renamed `block_mx_act_mmq` throughout (2026-08-18).**
A pure rename — same fields, same 144-byte layout — but it WILL conflict on
every re-vendor, so it is listed per file below.  It is worth the conflict: the
routed-expert path stopped filling that struct with int8 q8_1 and now fills it
with e4m3 bit patterns plus a ue8m0 byte in `d4[]` (see `ds4_quantize_e4m3.cu`),
and the stale name cost real time — it was read as evidence that the MoE still
ran q8_1 activations, and a whole scoping verdict was written on that mistake
before Tyler caught it (ledger L065).  Upstream's name is correct upstream,
where MMQ really does quantise activations to q8_1; in this fork it is not.

| File            | ds4 delta vs pin | What it is |
|-----------------|------------------|------------|
| `mmq.cuh`       | +255 / −18 | **(a)** `load_tiles_q2_K_soa` and `load_tiles_iq2_xxs_soa` — aligned-SoA twins of upstream's `load_tiles_q2_K` / `load_tiles_iq2_xxs`, reading the weight-server artifact (P4 Inc3).  **(b)** the `x_soa` / `soa_blocks` fields on `mmq_args` plus the `if constexpr (type == …) { if (x_soa != nullptr) … }` dispatch that selects them.  **(c)** a `DS4_CUDA_MMQ_X_MAX` env clip (Step-4 experiment hook).  **(d)** `block_q8_1_mmq` RENAMED to `block_mx_act_mmq` (2026-08-18, rename only — layout and size are byte-identical). |
| `quantize.cu`   | rename only | same `block_q8_1_mmq` → `block_mx_act_mmq` rename; `quantize_mmq_q8_1_cuda` itself has no live caller in this build. |
| `mmid.cu`       | +177 / −2  | ds4 expert-routing entries |
| `mmvq.cu`       | +26 / −4   | `mul_mat_vec_q_switch_type` promoted from `static`; ggml-tensor entries gated on `DS4_MMVQ_INCLUDE_GGML_ENTRIES` |
| `mmvq.cuh`      | +18 / −0   | matching prototype exposure + the same gate |
| `mmid.cuh`      | +4 / −0    | prototype exposure for the above |

Byte-identical to the pin, and therefore free to replace outright:
`mma.cuh`, `ggml-common.h`, `vendors/cuda.h`, and the `ds4_cuda_env.cuh` extract.

## Re-syncing with upstream

> **Do NOT `cp` the files over.**  An earlier revision of this document
> recommended exactly that.  It predates the patches above and would silently
> discard all five of them — including the entire aligned-SoA path, which is
> the type-43 performance work.  The `ds4_mmq.cu` side would fail to compile
> (it references `x_soa`), but the `mmid`/`mmvq` patches can be lost quietly.

Re-vendoring is a **patch-carrying merge**, not a copy.  The procedure:

```sh
# 1. Establish the three-way base: the pinned upstream revision.
PIN=5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a
# 2. For each patched file, extract our delta against the PIN (not master):
#      diff <pinned copy> <our copy>  > ds4-<file>.patch
# 3. Drop in the new upstream revision for every file.
# 4. Re-apply each ds4 patch, re-homing it if upstream moved the code.
# 5. Re-run the symbol sweep for shim gaps:
grep -hoE 'GGML_[A-Z_]+|ggml_[a-z_]+' *.cuh *.cu *.h | sort -u > /tmp/symbols.new
diff /tmp/symbols.last /tmp/symbols.new
# 6. Update the pin in this file, rebuild, and run the bit-exactness gates
#    (make cuda-prefill-gate) — a silently-dropped SoA path still produces
#    plausible numbers, so compiling is not evidence of correctness.
```

New `GGML_CUDA_CC_*`-style constants usually arrive via `common.cuh`, from which we now extract (`ds4_cuda_env.cuh`) and which we
vendor, and need no shim change.  New `ggml_*` host helpers need
`ds4_ggml_stubs.h` extended.

### State of upstream as of 2026-08-13 (measured, master vs our pin)

Upstream **split `mmq.cuh` apart**: our single 189,612-byte file corresponds to
upstream's `mmq.cuh` (71,514 B) plus two extracted files, `mmq-load-tiles.cuh`
(80,567 B) and `mmq-vec-dot.cuh` (49,070 B), plus eight per-architecture
`mmq-config-*.cuh` headers.  All ten are `#include`d unconditionally by
`mmq.cuh`, so all ten must be vendored even though the CDNA/RDNA ones are dead
weight in a CUDA-only fork — patching the includes out would create fresh
divergence and defeat the point.

Two facts make this cheaper than the file count suggests:

- **The adapter contract survives.**  `mul_mat_q_case<type>(ggml_backend_cuda_context&, const mmq_args&, cudaStream_t)`
  still exists with an unchanged signature, so `ds4_mmq.cu`'s dispatch does not
  need rewriting.  `mmq_args` itself did change: upstream dropped
  `use_stream_k` and added `const float * y_scale`, so our two trailing fields
  must be re-added and the adapter's initializers checked.
- **Our patches partition along the seam upstream chose.**  Both SoA loaders
  are twins of `load_tiles_*` functions that now live in `mmq-load-tiles.cuh`,
  so patch (a) relocates to that file next to its counterparts and only the
  `x_soa` dispatch hook (b) stays in `mmq.cuh`.  This is mechanical
  re-homing, not a redesign.

`mmq-config-blackwell.cuh` is new since our pin and is the reason to care about
GB10 specifically — but note it configures **MXFP4 and NVFP4 only**, and falls
back to `ggml_cuda_mmq_get_config_ampere` for everything else.  There is still
**no upstream Blackwell tuning for 2-bit types** (IQ2_XXS, Q2_K), so this sync
does not by itself move our dominant gate/up stage.  It does bring Blackwell
MXFP4 config (we run MXFP4 on 13 of 43 layers) and NVFP4 support we do not
have at all.

## Testing matrix

| Test                                          | Status                       | Phase     |
|-----------------------------------------------|------------------------------|-----------|
| `nvcc -c cuda/mmq/ds4_mmq.cu` builds cleanly  | **passes** (sm_120, nvcc 13) | Phase 0   |
| Q8_0 dense parity vs CPU reference            | **passes** (4 shapes)        | Phase 2   |
| Q2_K dense parity                             | **passes** (4 shapes)        | Phase 3   |
| IQ2_XXS dense parity                          | **passes** (4 shapes)        | Phase 3   |
| MoE `_id` Q8_0 parity                         | **passes** (3 shapes)        | Phase 4   |
| MoE `_id` Q2_K parity                         | **passes** (3 shapes)        | Phase 4   |
| MoE `_id` IQ2_XXS parity                      | **passes** (4 shapes)        | Phase 4   |
| `make ds4-bench` with mmq integration         | **builds and runs**          | Phase 5/6 |
| Frontier sweep, ctx 2k-16k, V4 Flash IQ2XXS   | **see results below**        | Phase 7   |

## Validated performance

PRO 6000 Blackwell (sm_120), CUDA 13.0, V4 Flash IQ2_XXS GGUF (86.7 GB),
default `DS4_CUDA_MMQ_MOE_MIN_TOKENS=2` (legacy decode preserved):

| ctx    | baseline pf t/s | mmq pf t/s | speedup    | baseline gen t/s | mmq gen t/s | gen ratio |
|--------|-----------------|------------|------------|------------------|-------------|-----------|
|  2048  | 373.21          | 1033.42    | **2.77x**  | 40.48            | 39.33       | 0.972x    |
|  4096  | 366.25          | 1041.39    | **2.84x**  | 39.64            | 38.64       | 0.975x    |
|  6144  | 364.81          | 1025.24    | **2.81x**  | 39.50            | 38.55       | 0.976x    |
|  8192  | 364.01          | 1026.71    | **2.82x**  | 38.81            | 37.88       | 0.976x    |
| 10240  | 361.75          | 1019.53    | **2.82x**  | 38.45            | 37.57       | 0.977x    |
| 12288  | 360.52          | 1013.17    | **2.81x**  | 38.31            | 37.22       | 0.972x    |
| 14336  | 359.15          | 1004.45    | **2.80x**  | 38.09            | 37.18       | 0.976x    |
| 16384  | 357.99          | 1001.29    | **2.80x**  | 38.71            | 37.86       | 0.978x    |

Sustained ~2.80x prefill speedup across the swept context range; gen
within 2.5% of baseline (run-to-run variance).  See `local/docs/`
(auto-round companion repo) for the full Phase 0-7 execution log,
parity-test output, and detailed plan.

---

## pulsar-side provenance (2026-08-07)

This tree entered pulsar via **Palaferri's** ds4 fork
(`xangel82/DS4-GB10-GX10-DSpark-CUDA` @ `302f517`, MIT), which is itself the
llama.cpp vendor documented above plus the `ds4_*` adapter. The upstream pin,
file inventory and licence statements above are **unchanged and still apply** —
the llama.cpp files are verbatim at commit `5c0e946`.

### Local modifications (pulsar, NOT upstream and NOT Palaferri)

| File | Change |
|---|---|
| `ds4_mmq_d2r.cu` | added `ds4_mmq_iq2_xxs_moe_d2r_single_launch` |
| `ds4_mmq_d2r.cuh` | its declaration |
| `ds4_mmq.cu` | IQ2_XXS twin of the existing Q2_K single-tensor D2R dispatch in `ds4_mmq_moe_impl`; `ds4_mmq_iq2_xxs_moe_soa` |
| `ds4_mmq.h` | declaration of `ds4_mmq_iq2_xxs_moe_soa` |

**Why.** Upstream ships a Q2_K single-tensor D2R (`down_q2k_d2r_kernel`) and an
IQ2 *pair* D2R, but no IQ2 **single**. A routed DOWN whose tensor is IQ2 rather
than Q2_K — which is v5mx, our model has no Q2_K at all — therefore fell back to
stock `mul_mat_q` at 33.24 ms/layer against Entrpi's 9.78 ms Q2_K D2R. That was
the entire remaining MoE gap.

**No new kernel was written.** `gateup_iq2_d2r_pair_kernel` already separates the
two projections along `blockIdx.z` (`W_soa = leg == 0 ? gate_soa : up_soa`), so
launching it with `gridDim.z = 1` pins leg to 0 and computes exactly one tensor.
The single launcher is that launch geometry plus the pair launch's own
`soa_blocks` validation; the dispatch mirrors the Q2_K block beside it.

Result: routed IQ2 down 33.24 -> 8.98 ms/layer, and all IQ2 MoE moved from
4055 ms to 783 ms at a 4096-token prefill.

### Re-syncing

The llama.cpp files may be refreshed from upstream normally. The four files
above carry pulsar-only additions that are **purely additive** (new symbols and
one new dispatch branch); re-apply them after any re-sync.
`gguf-tools/d2r_iq2_single_launch.inc` and `gguf-tools/patch_iq2_d2r_single.py`
reproduce both mechanically.
