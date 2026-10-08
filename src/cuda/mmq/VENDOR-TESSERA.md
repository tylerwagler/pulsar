# Vendored Tessera kernel (L255)

**Made with Tessera by Robert Tand - https://github.com/RobTand/tessera**

`tessera_routed_fused_window.cuh` is Tessera's fused window kernel,
`src/tessera/serving/csrc/routed_fused_window.cu`, copied from upstream
**`37742e0f`** (2026-10-08, contract v64; file sha256 `4e93959a2a26…`).
First vendored from `1381c3b7` (2026-09-29); re-vendored 2026-10-08 -- the upstream
file grew from 1,520 to 3,778 lines (four families: value, E4M3, E4M3-MMA, E2M1;
dense rates 9-14; the piece-major E4M3 layout; the paired-K32 schedule), the value
family's `Params` gained `piece_major`, `fixup`, `tile_sem`, `roles` (all zero here:
pulsar never requests piece-major and launches no multi-role dense), and `table0/1`
became `const void*`.  Byte parity with Tessera's own build at this revision is
**NOT YET re-run** (the gate needs sparky); the 1381c3b7 gate was 17/17. Tessera is
licensed **MIT + Attribution Addendum 1.0** (`LicenseRef-Tessera-Attribution-1.0`);
the full text is `LICENSE-TESSERA` in this directory, verbatim (addendum A3
requires it to travel with every copy).

## What the addendum asks of us

- **A2**: any model or artifact we publish, distribute or offer as a service
  that is *built or served with* this kernel must say, in its model card or
  README: `Made with Tessera by Robert Tand - https://github.com/RobTand/tessera`.
  That includes the Elytron API if it serves a Tessera-format model.
- **A3/A4**: keep `LICENSE-TESSERA` and the copyright notice with the code; never
  present it as ours. The three pulsar files that touch it carry the credit line.
- **A5**: "Tessera" names Rob's project only, never a pulsar product or lane.

## What was changed (host glue only)

Every device function is upstream byte for byte. The header's own banner
lists the edits; in short:

| upstream (37742e0f) | here |
| --- | --- |
| lines 68-72, the torch / c10 / ATen `#include`s | dropped |
| `C10_CUDA_CHECK` in `max_dynamic_smem_bytes`, `launch_variant`; `C10_CUDA_KERNEL_LAUNCH_CHECK` in `launch_variant` | `TESSERA_HOST_CUDA(expr)`, defined by the includer |
| the three piece-major `TORCH_CHECK(cond, msg)` in `launch` | `TESSERA_HOST_CHECK(cond, msg)`, defined by the includer |
| `TORCH_CHECK(false, …)` at the end of `launch` (the run-pair refusal) | `TESSERA_HOST_REFUSE_PAIR(…)`, defined by the includer |
| lines 2063-2080 and 2110-2131: `check_words`, `check_slot`, `i32_ptr`, `f32_ptr`, `TABLE_DTYPE`, `check_run_tables` (torch::Tensor) | dropped |
| lines 2081-2109, `dense_reduce_kernel` | kept verbatim |
| lines 2132-3048, the E2M1 family (`#if TESSERA_ROUTED_FUSED_FP4`, which we build at 0) | dropped |
| lines 3051-end: the torch host entries and the PYBIND module | dropped; `pulsar_tessera.cu` replaces them |

`pulsar_tessera.cu` fills `Params` exactly as upstream's `dense_forward` /
`routed_fused_forward` do, replaces `token_sum`'s host entry, and replaces the
Python owner's routing prep (`FusedRoutedWindowMoE._routing`: counts, the stable
argsort, the two cumsums) with three small kernels. It builds the value family
only (`TESSERA_ROUTED_FUSED_FP8=0`): the Qwen lane's activations are BF16.

**Build flags matter.** Upstream compiles `-O3 -lineinfo` without fast-math;
the SwiGLU epilogue calls `expf` and divides. The Makefile compiles
`pulsar_tessera.o` with pulsar's flags minus `--use_fast_math`.

## The gate

`make tessera-kernel-gate TESSERA_FIXTURE=…` runs `tests/tessera_kernel_gate`
against a fixture from `tools/tessera/kernel_fixture.py`, which runs **Tessera's
own build** of this kernel (its torch extension) on Qwen3.8-Flash-Next layer-12
weights. The gate passes only when every output byte matches and pulsar's
split-K choice equals Tessera's `dense_k_split`.

## Re-syncing

1. `python3 tools/tessera/vendor_kernel.py <tessera>/src/tessera/serving/csrc/routed_fused_window.cu --rev <sha>`
   regenerates the header with the same cuts; it asserts each replaced line
   first, so a moved seam fails by name (update its line numbers after reading
   the new upstream).
2. Diff the device code against the previous vendor: any change to `Params`,
   the host entries' field assignments, `dense_k_split` or `_routing` must be
   mirrored in `pulsar_tessera.cu`.
3. Regenerate the fixture with the new Tessera and run the gate.
