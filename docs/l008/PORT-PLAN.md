# L008 — MMQ vendor re-sync: measured port plan

State of branch `l008-mmq-vendor` as of 2026-08-13. **Incomplete and does not
compile** — the vendored tree is upstream master, but `ds4_mmq.cu` still expects
the old `mmq_args` shape and the SoA threading that has not been re-applied yet.
`dev` is unaffected.

## What is already done here

* 21 upstream files vendored from llama.cpp **master `a94d563e`** (pin was
  `5c0e9468`, 2026-05-14). New since the pin: `mmq-load-tiles.cuh`,
  `mmq-vec-dot.cuh`, and 8 `mmq-config-*.cuh`, all unconditionally included.
* `mmid.cuh`, `mmvq.cuh`, `mmid.cu`, `mmvq.cu` restored to **vanilla** master.
* Our five pin-era patches extracted verbatim into `docs/l008/*.pin-delta.patch`
  and the two SoA loader bodies into `soa_loaders_from_pin.txt`.

## Measured delta (this is the useful part)

Our patch vs the pin was **+480/−24 over 5 files**. Upstream moved a long way:
`mmq.cuh` alone changed **4255 lines** and was split into 10 files.

| file | ours vs pin | upstream pin→master | disposition |
|---|---|---|---|
| `mmq.cuh` | 275 | 4255 | **~63 lines must stay**; 172 move out; 20 dropped |
| `mmid.cu` | 181 | 33 | **9 lines stay**; 168 move to a ds4 file |
| `mmvq.cu` | 32 | 256 | stays (correctness + visibility) |
| `mmvq.cuh` | 20 | 4 | **vanilla** — replaced by an include-shim |
| `mmid.cuh` | 6 | 4 | **vanilla** |
| `mma.cuh`, `vecdotq.cuh`, `quantize.{cu,cuh}`, `common.cuh`, `unary.cuh` | 0 | 0–578 | take upstream verbatim |

**Target: 514 changed lines across 5 upstream files → ~85 across 3.**

## The three patches that must remain, and why

1. **`mmq.cuh` (~63 lines)** — threading `x_soa`/`soa_blocks` through the kernel
   chain plus the `if constexpr (type == GGML_TYPE_Q2_K)` SoA dispatch. Upstream's
   loader typedef is `(x, x_tile, kbx0, i_max, stride)` with **no** slot for the
   pair count, so the extra argument cannot be passed without widening it. The
   172 lines of loader *bodies* do NOT need to be here — move them to a ds4-owned
   header and leave only the plumbing.
2. **`mmid.cu` (9 lines)** — a `__syncwarp()` between the shared-memory `store[]`
   writes and the cross-lane reads in upstream's `mm_ids_helper`. Post-Volta
   independent thread scheduling makes this a genuine RAW hazard; on GB10 it
   realized as nondeterministic MoE routing and BOS spam (compute-sanitizer
   racecheck). **This is an upstream bug — contributing it retires the patch.**
3. **`mmvq.cu`** — the `-1` router-id guard (task #23): our router's NaN path
   emits `-1` expert ids, upstream casts to `uint32`, and the resulting wild
   weight-channel offset is a cuda-gdb-convicted illegal address. Also an
   **upstream bug candidate.** Plus the `static` → external promotion of
   `mul_mat_vec_q_switch_type`.

## What to DROP rather than carry

* `DS4_CUDA_MMQ_X_MAX` (20 of the 275 `mmq.cuh` lines) — a tile-width sweep hook
  with no caller. Carrying an experiment knob across every future re-sync is
  exactly the cost L008 exists to remove.

## What moves to ds4-owned files

* `mm_ids_helper_global` + its launcher dispatch and the `case 1:` fast path
  (168 lines) — our adapter already checks the smem cap (`ds4_mmq.cu:694`), so it
  can call our kernel directly and leave upstream's launcher untouched.
  ⚠ Chunking under the cap is NOT a viable alternative: the helper builds a
  compacted structure with global `expert_bounds`, so per-chunk merging just
  reimplements the two-pass variant.
* The two SoA loader bodies (172 lines) into e.g. `ds4_mmq_soa_tiles.cuh`.

## The hard part, honestly

**21 of 22 `mmq.cuh` hunks fail to apply** (`patch -F5`); only the pure-addition
loader hunk lands. This is a hand-port, not a merge. Each of the 20 threading
edits must be re-placed against master's restructured kernels, which now use
`ggml_cuda_mmq_config`, template `<ggml_type type, int J, bool fallback>`, and
`ggml_cuda_mmq_get_I(...)` where we used `mmq_y` / `need_check` /
`mmq_get_nwarps_device()`. The loader *bodies* port cleanly under that renaming;
the plumbing does not.

`ds4_mmq.cu` also needs adapting: `mmq_args` **dropped `use_stream_k`** and
**gained `y_scale`**, and our `x_soa`/`soa_blocks` fields must be re-added.

## Verification bar (do not skip)

`make cuda-prefill-gate`. **Compiling proves nothing**: if the SoA dispatch is
dropped the code still runs and still produces plausible numbers — it just
silently stops using the aligned artifact. The gate is what catches that.
