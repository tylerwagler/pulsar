# Agent Notes

Pulsar (`pulsar`) is a DeepSeek V4 Flash specific inference engine, not a
generic GGUF runner. This tree is the **CUDA-only fork** of antirez's upstream
project, targeting the NVIDIA DGX Spark (GB10, `sm_121`, ~121 GB usable
unified memory). The Metal, ROCm, and CPU inference backends were fully
removed; `cuda` is the only backend. The engine still contains shared
host-side math (`attention.cpp`, `hc.cpp`, `layers.cpp`, `moe.cpp`,
`experts.cpp`) that the live GPU path calls — it is not dead CPU-backend code.

## Goals

- Keep the production path as whole-model CUDA graph inference on GB10.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF. The model
  must be fully resident: a GGUF that does not fit is rejected at load, never
  partially streamed.
- Preserve correctness before speed. Do not keep a faster path with
  unexplained attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Keep the implementation small, sharp, easy to understand. Don't introduce
  slop: fragile case-patching, dead code, or needless complexity.
- Comment inference code where model mechanics, cache lifetime, memory policy,
  or API orchestration are not obvious locally. Prefer comments beside the
  implementation over separate design documents.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- The tree is C++ (`src/**/*.cpp`, `src/cuda/*.cu`); there are no `.c` TUs
  left. C++ is no longer confined to the CUTLASS TU — templates, RAII and
  headers-with-inline are fair game anywhere. The small/sharp rule above still
  governs: use C++ where it removes duplication or makes an invariant
  checkable at compile time (the MoE kernels' `Dot8`/`STAGED`-style template
  dispatch is the house pattern), not for abstraction on principle.

## Layout

Public headers: `src/pulsar.h` (engine API) and `src/pulsar_gpu.h` (GPU graph API).

- `src/engine/` — 30 TUs + `pulsar_engine_internal.h`: GGUF parsing, tokenizer,
  weight binder (`weights.cpp`), quant format/kernel tables, GPU graph
  orchestration (`gpu_graph_alloc.cpp`, `gpu_graph_state.cpp`,
  `gpu_prefill.cpp`, `gpu_decode.cpp`, `gpu_diag.cpp`), sessions + KV payload
  serialization, imatrix
  collection, steering, and the shared host math TUs.
- `src/cuda/` — 9 kernel TUs + `pulsar_cuda_internal.h`: runtime/memory
  (`pulsar_cuda_runtime.cu`), matmuls incl. the cuBLASLt MXFP8 path
  (`pulsar_cuda_matmul.cu`), attention, MoE, indexer, norm/KV, HC router; plus
  `pulsar_mxfp4_cutlass.cu` (CUTLASS grouped GEMM for MXFP4 experts) and
  `pulsar_iq2_tables_cuda.inc`.
- `src/server/` — 14 TUs + `pulsar_server_internal.h`: HTTP server,
  OpenAI/Responses/Anthropic endpoints and streaming, prompt rendering, disk
  KV cache, exact-DSML tool replay.
- `src/agent/` — 20 TUs + `pulsar_agent_internal.h`: native coding agent (tools,
  terminal UI, sessions, compaction).
- `src/cli/` — `pulsar_cli.cpp`, `pulsar_bench.cpp`, `pulsar_eval.cpp` entry points.
- `src/lib/` — shared pieces: help text, kvstore.
- `src/vendor/` — linenoise, rax.
- `tests/` — test runners; `pulsar_test.cpp` and `pulsar_agent_test.cpp` are unity builds
  that `#include` the server/agent source lists.
- `cutlass/` — git submodule (v4.5.2), **required** for the MXFP4 expert path.
- `gguf-tools/` — offline quantization/imatrix tooling that produces the GGUFs
  this fork loads.

Internal-header convention: a symbol is de-static'd and declared in the
module's `pulsar_*_internal.h` only when another TU of the same module needs it.
Everything else stays `static`. The cross-module surface is `src/pulsar.h`,
`src/pulsar_gpu.h`, and the `src/lib/*.h` headers only.

## Build

```sh
git submodule update --init cutlass   # once
make cuda-spark          # DGX Spark / GB10 (CUDA_ARCH=sm_120f)
make cuda-generic        # other local CUDA GPUs (CUDA_ARCH=native)
make cuda CUDA_ARCH=sm_N # explicit -arch, e.g. cross-builds
```

Binaries land in the repo root: `pulsar`, `pulsar-server`, `pulsar-agent`, `pulsar-bench`,
`pulsar-eval`. Note `pulsar_mxfp4_cutlass.cu` needs the `sm_120f` family for the
mxf4 block-scale MMA; the Makefile handles its flags.

## Testing

- `make test` runs `./pulsar-eval --self-test-extractors`, `./pulsar_agent_test`,
  and `./pulsar_test`. The eval self-test and agent test need no model;
  `pulsar_test` loads a model (`PULSAR_TEST_MODEL`, default `./ds4flash.gguf`).
- `pulsar_test` distinguishes **gating** internal-correctness tests (any failure
  fails the suite) from **informational** ones: `logprob-vectors` compares
  the 2-bit production model against full-precision official-API logprobs (a
  drift dashboard, expected mismatches), and `think-tool-recovery` is
  run-to-run flaky from the nondeterministic batched-prefill float-atomic
  down-sum. Informational mismatches are reported but exit 0.
- `make cuda-regression` runs `tests/cuda_long_context_smoke`: GPU kernel
  smoke tests, no model required.
- `./pulsar_test --logprob-vectors` compares against official-API vectors and
  pins `PULSAR_CUDA_PREFILL_CHUNK=2048`.
- imatrix collection (`--imatrix-dataset` / `--imatrix-out`) requires `--cuda`.

## Validation Culture

- **Refactors** (code movement, TU splits, renames) must produce byte-identical
  greedy output versus the pre-change binary.
- **Numerics changes** (kernels, quantization, activation formats) are
  perplexity-gated (`--perplexity-file`) plus the deterministic `pulsar-eval`
  q1..q4 token-count gate; recent reference point on the 97 GB zero-Q8 Flash
  oracle: ppl 7.3216, decode 12.35 t/s, long-prompt prefill 162 t/s.
- **One pulsar process at a time on the GB10.** Two ~97 GB model mappings OOM the
  box; the instance lock (`PULSAR_LOCK_FILE`) is intentional.

## Supported Weight Formats (binder-enforced)

| Tensor group | Accepted formats |
| --- | --- |
| Attention projections, shared experts | MXFP8 (FP8 E4M3 + per-32 E8M0 scales) |
| Routed experts gate/up/down | exactly `IQ2_XXS`/`IQ2_XXS`/`Q2_K`, or all three `MXFP4`; `IQ2_XXS_SOA` (42) is accepted anywhere `IQ2_XXS` (16) is |
| Output head | `BF16` or MXFP8 |
| Norms, embeddings, indexer, HC | `F32`/`F16` |

Legacy `Q4_K`/`Q8_0` weights are rejected at load with one clear error
(`weights_reject_unsupported_types` in `src/engine/weights.cpp`). `Q8_K` exists
only as *activation* quantization inside the routed-expert (MoE) kernels.

## Compute Paths

- **Prefill:** dense matmuls run cuBLASLt block-scaled MXFP8×MXFP8
  (`VEC32_UE8M0`) on tensor cores, including the attention-output projections.
  MXFP4 expert prefill runs the CUTLASS mxf4×mxf4 block-scaled grouped GEMM
  (GB10 / `sm_120f` family).
- **Decode:** custom fused kernels, memory-bound; activations stay raw f32
  except inside MoE (Q8_K).
- **FP8 KV cache**: the compressed cache is packed E4M3+scale storage by
  default (`PULSAR_ATTN_PACK=1`); the raw window is F16 (`PULSAR_RAW_F16=1`).
  Decode attention reads the packed cache natively. Prefill attention
  consumes a per-chunk dequantized **f32 shadow**
  (`gpu_graph_attn_comp_read_cache`) — deliberately: native packed prefill
  reads were tried 2026-08-02 and measured slower at every depth through
  131k (see the Deferred Work NO-GO entry below for why).

## Environment Variables

All runtime tuning/diagnostic gates use the `PULSAR_CUDA_*` prefix (this fork
renamed every `PULSAR_METAL_*` gate; there are no compatibility aliases).
Also: `PULSAR_TEST_MODEL`, `PULSAR_LOCK_FILE`, `PULSAR_GGUF_DIR` (download script).

## Deferred Work

- ~~Native packed-KV reads in the prefill attention kernels~~ — **MEASURED
  NO-GO 2026-08-02** (tried and reverted the same day; see the PR #9 trail).
  Routing the packed comp cache into the indexed prefill kernel (its pack
  branch is bit-exact; gate PASSED) measured −1.6% @8k, −1.9% @16k and
  −2.7% @131k vs the f32 shadow. The bandwidth thesis fails because DSA's
  top-k keeps the attention working set L2-hot at any depth (512 gathered
  rows/token, heavy overlap between neighbors), so the 4x byte saving buys
  nothing while the per-(token,row) e4m3 decode tax is unconditional. The
  shadow (decode once per chunk, share across tokens) is the measured-optimal
  amortization — do not retry without a design that decodes at most once per
  (chunk, row).
- Move MoE decode off Q8_K activation quantization.
