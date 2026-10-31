# Agent Notes

Pulsar (`pulsar`) is a CUDA inference engine for a fixed set of model
families, not a generic model runner. Each family is a plugin behind one
session core (`src/engine/family.h`; the registry is `k_families` in
`src/engine/family.cpp`, keyed by the container's `general.architecture`):

| Family (`general.architecture`) | Models | Module |
| --- | --- | --- |
| `deepseek4` | DeepSeek V4 Flash (0731 / Vision-Exp) and V4.1 | `src/engine/family_deepseek.cpp` + the DeepSeek forward |
| `qwen4_exp` | Qwen3.8-Flash-Next | `src/engine/family_qwen*.cpp` |
| `inkling` | Inkling | `src/families/inkling/` |
| `nemotron_h` | Nemotron-3.5 Super | `src/families/nemotron_h/` |
| `nemotron_enc` | Nemotron-3.5 Embed-8B / Rerank-8B (encoders) | `src/families/nemotron_enc/` |
| `nemotron_parse` | Nemotron-3.5 Parse (page parser) | `src/families/nemotron_parse/` |

What each family implements is its `PULSAR_FAMILY_CAP_*` mask (banks, spec,
payload, rewind, vision, TP, imatrix, chat, segments, mixed prefill, encode,
parse); an operation a family lacks refuses by name (`pulsar_family_require`).

This tree began as the **CUDA-only fork** of antirez's upstream ds4 project.
The production target is the NVIDIA DGX Spark (GB10, `sm_121`, built as
`sm_120f`, ~121 GB usable unified memory). Older targets build too (L321:
sm_75 serves the Nemotron encoder and Parse families); a lane whose
instructions the compile target lacks refuses by name
(`src/cuda/pulsar_cuda_arch.cuh`). The Metal, ROCm, and CPU inference backends
were fully removed; `cuda` is the only backend. The engine still contains
shared host-side math (`attention.cpp`, `layers.cpp`, `moe.cpp`) that the live
GPU path calls — it is not dead CPU-backend code.

## Goals

- Keep the production path as whole-model CUDA graph inference on GB10.
- Keep model loading mmap-backed; do not eagerly copy the weights. The model
  must be fully resident: a checkpoint that does not fit is rejected at load,
  never partially streamed. The exceptions are tables a format declares
  disk-resident and reads by row on purpose: Qwen's PLE n-gram row file
  (opened in `src/engine/family_qwen_s4.cpp`, the next prompt chunk's rows
  read ahead, L329b) and DeepSeek V4.1's Engram table (L242). Such tables
  must sit on local NVMe.
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

- `src/engine/` — 58 TUs + `pulsar_engine_internal.h`: the model layer and the
  safetensors container reader (`model.cpp`, `safetensors.cpp`, `cursor.cpp`),
  the family registry and the DeepSeek / Qwen families (`family*.cpp`),
  tokenizers, weight binders (`weights.cpp`, `role_bind.cpp`), quant
  format/kernel tables, the DeepSeek GPU graph orchestration
  (`gpu_graph_alloc.cpp`, `gpu_graph_state.cpp`, `gpu_prefill.cpp`,
  `gpu_decode.cpp`, `gpu_diag.cpp`), the family-neutral prefill loop
  (`prefill_loop.cpp`) and row-step driver (`step_driver.cpp`), sessions,
  banks, grid checkpoints and KV payload serialization (`session*.cpp`,
  `checkpoint.cpp`), the speculation lane and drafters (`spec_*.cpp`), images
  (`image_*.cpp`, `vision*.cpp`), TP slicing (`tp_*.cpp`), imatrix collection,
  steering, and the shared host math TUs.
- `src/families/` — the family modules added as plugins (L295): `inkling/`,
  `nemotron_h/`, `nemotron_enc/`, `nemotron_parse/`. A new family adds a
  directory; the Makefile globs it.
- `src/cuda/` — 43 kernel TUs + `pulsar_cuda_internal.h`: runtime/memory
  (`pulsar_cuda_runtime.cu`), matmuls incl. the cuBLASLt MXFP8 path
  (`pulsar_cuda_matmul.cu`), attention (MLA, GQA, QSA, GDN, Mamba-2), MoE,
  indexer, KV rows, drafters, sampler, vision, TP; plus
  `pulsar_mxfp4_cutlass.cu` (CUTLASS grouped GEMM for MXFP4 experts),
  `pulsar_cuda_fmha.cu` (over the vendored CUTLASS FMHA in `fmha/`), the
  `mmq/` directory (the D2R 2-bit MoE kernels, the EXL3 kernels, the vendored
  Tessera kernel) and `pulsar_iq2_tables_cuda.inc`.
- `src/server/` — 28 TUs + `pulsar_server_internal.h`: HTTP server,
  OpenAI/Responses/Anthropic endpoints and streaming, per-family chat
  rendering and output parsing, the scheduler (banks, preemption, admission),
  disk KV cache, structured output (`constrain.cpp`), the encoder/parser
  endpoints (`encode_server.cpp`), the stall watchdog (`watchdog.cpp`),
  exact-DSML tool replay.
- `src/tp/` — the tensor-parallel transport (TCP / RDMA), gate scheduler and
  verify.
- `src/agent/` — 19 TUs + `pulsar_agent_internal.h`: native coding agent (tools,
  terminal UI, sessions, compaction).
- `src/cli/` — `pulsar_cli.cpp`, `pulsar_bench.cpp`, `pulsar_eval.cpp` entry points.
- `src/lib/` — shared pieces: help text, the disk KV segment store and chains
  (`pulsar_segstore`, `pulsar_kvchain`), the grammar boundary
  (`pulsar_grammar.cpp`), the BPE tokenizer and the per-family chat /
  output-parser frontends, JSON, strong unit types (`pulsar_units.h`).
- `src/vendor/` — linenoise; `xgrammar/` (L303: xgrammar's C++ core, verbatim, pinned in its `UPSTREAM`).
  The tree builds `-fno-exceptions`; xgrammar throws, so its TUs and the ONE boundary TU that calls it
  (`src/lib/pulsar_grammar.cpp`, which catches everything into return values) are the only ones built
  `-fexceptions` (Makefile `GRAMMAR_CXXFLAGS`).
- `tests/` — test runners; `pulsar_test.cpp` and `pulsar_agent_test.cpp` are unity builds
  that `#include` the server/agent source lists.
- `cutlass/` — EXTERNAL header-only dependency, **required** for the MXFP4
  expert path. Deliberately **not** a git submodule (see `cutlass.pin` for the
  rationale and the pinned sha); `CUTLASS_DIR` may point anywhere.
- `tools/container/` — the direct artifact builder (HF checkpoint(s) -> the
  declared-layout safetensors checkpoint the engine serves, for every family;
  L247, one recipe schema since L279).  No GGUF anywhere: that tooling was
  archived 2026-09-24 at tag `archive/gguf-tooling-2026-09-24`.
  `tools/imatrix/` is the calibration corpus
  and collector docs, `tools/reap/` the router audit the battery runs,
  `tools/quality-testing/` the official-continuation scorer.

Internal-header convention: a symbol is de-static'd and declared in the
module's `pulsar_*_internal.h` only when another TU of the same module needs it.
Everything else stays `static`. The cross-module surface is `src/pulsar.h`,
`src/pulsar_gpu.h`, and the `src/lib/*.h` headers only.

## Build

```sh
make cutlass                          # once: clone the pinned CUTLASS
                                      # (or set CUTLASS_DIR=/path/to/cutlass)
make cuda-spark                          # DGX Spark / GB10; rebuilds
                                         # pulsar-server at CUDA_ARCH=sm_120f
make pulsar-server CUDA_ARCH=native      # other local CUDA GPUs
make pulsar-server CUDA_ARCH=sm_N        # explicit -arch, e.g. cross-builds
```

Binaries land in the repo root: `pulsar`, `pulsar-server`, `pulsar-agent`, `pulsar-bench`,
`pulsar-eval`. Note `pulsar_mxfp4_cutlass.cu` needs the `sm_120f` family for the
mxf4 block-scale MMA; the Makefile handles its flags.

## Testing

- `make gates` is the full battery and `make gates-dev` the iteration tier
  (CLAUDE.md, `docs/ENGINEERING-RULES.md` §10, `docs/QA_BEFORE_RELEASES.md`).
- `make test` runs `./pulsar_test` and the seam check. `pulsar_test` loads a
  model (`PULSAR_TEST_MODEL`; the battery passes `FRONTIER_MODEL` through). The
  eval extractor
  self-test (`./pulsar-eval --self-test-extractors`) and `./pulsar_agent_test`
  are separate targets; see `docs/QA_BEFORE_RELEASES.md`.
- `pulsar_test` distinguishes **gating** internal-correctness tests (any failure
  fails the suite) from **informational** ones: only `logprob-vectors`, which
  compares the production model against full-precision official-API logprobs
  (a drift dashboard, expected mismatches). Informational mismatches are
  reported but exit 0; `pulsar_test --gating` (the battery's
  `unit-test-gate`) skips them.
- `make cuda-regression` runs the model-free GPU kernel gates
  (`CUDA_REGRESSION_BINS` in the Makefile: `tests/cuda_long_context_smoke` and
  the family kernel gates), concurrently; no model required.
- `./pulsar_test --logprob-vectors` compares against official-API vectors and
  pins `PULSAR_CUDA_PREFILL_CHUNK=2048`.
- imatrix collection uses `--imatrix-dataset` / `--imatrix-out` (CUDA-only, like all inference).

## Validation Culture

- **Refactors** (code movement, TU splits, renames) must produce byte-identical
  greedy output versus the pre-change binary.
- **Numerics changes** (kernels, quantization, activation formats) are graded
  against the B300 reference (`make cuda-reference-gate`, `PULSAR_REF_DIR`),
  never argued (`docs/ENGINEERING-RULES.md` §7).
- **One pulsar process at a time on the GB10.** Two large model mappings OOM
  the box; the instance lock (`PULSAR_LOCK_FILE`, default `/tmp/ds4.lock`) is
  intentional.

## Supported Weight Formats (binder-enforced)

A checkpoint declares each tensor's storage by NAME; the accepted names are
`native` plus the table in `src/engine/model.cpp`:

| Layout | What it is |
| --- | --- |
| `native` + dtype | `F32` / `BF16` / `F16` / `I32` as stored (norms, embeddings, BF16 heads and linears, misc) |
| `mxfp8_lt` | pre-stored MXFP8 (E4M3 + per-32 E8M0, cuBLASLt swizzle) |
| `cutlass_mxfp4` | MXFP4 routed experts in the CUTLASS block-scaled layout |
| `iq2_xxs_mmq_k` | 2-bit IQ2_XXS routed experts for the D2R MMQ kernels |
| `fp8_e4m3_soa_k` | the DSpark drafter's markov table (E8M0 scale plane + E4M3 payload) |
| `exl3m_k2`, `exl3m_k2h`, `exl3m_k3` .. `exl3m_k6`, `exl3m_k8` | EXL3 trellis weights at that rate (routed experts and dense linears; L245) |
| `nvfp4` | NVFP4 W4A16 rows (the LiLiCorr drafter, L298) |

Which layout a given tensor may take is its family binder's decision
(DeepSeek: `src/engine/weights.cpp`; the others through `role_bind` and their
`*_weights.cpp`). For DeepSeek V4: attention projections and shared experts
are `mxfp8_lt`; routed gate/up must match, and each of gate/up and down is
independently `iq2_xxs_mmq_k`, `cutlass_mxfp4` or EXL3 (V4.1), per layer; the
output head is `BF16` or MXFP8.

An unknown or retired name is refused at load by name, once (`st_layout_type`
in `src/engine/safetensors.cpp`). The legacy interleaved MXFP8 layout is
refused there with the repack named, and unsupported types are refused again by
`weights_reject_unsupported_types` in `src/engine/weights.cpp`.

## Compute Paths (DeepSeek V4)

- **Prefill:** dense matmuls run cuBLASLt block-scaled MXFP8×MXFP8
  (`VEC32_UE8M0`) on tensor cores, including the attention-output projections.
  MXFP4 expert prefill runs the CUTLASS mxf4×mxf4 block-scaled grouped GEMM
  (GB10 / `sm_120f` family).
- **Decode:** custom fused kernels, memory-bound. Activations are STAGED in
  f32 and encoded to E4M3 where they feed a block-scaled GEMM (the encoding is
  cached per activation buffer — see `pulsar_gpu_mxfp8_act_cache_*` in
  `src/pulsar_gpu.h`). The MoE path stages
  E4M3 too: its activation block is `block_mx_act_mmq`, renamed from upstream's
  `block_q8_1_mmq` precisely because it no longer holds q8_1. There is no Q8_K
  anywhere: the type and its blocks are deleted.
- **KV cache**: one row style per profile (`pulsar_kv_row_style` in
  `src/pulsar_gpu.h`). **UNIFIED** (0731 / Vision-Exp): one **NVFP4 384-B
  row** at head_dim 512 / n_rot 64 for every non-index buffer — raw ring, comp
  pool, drafter ring and the current chunk (224 B of e2m1 nibbles + 28 B of
  E4M3 scale codes + a 4-B f32 row scale + 128 B of bf16 rope; the block
  comment in `src/cuda/pulsar_cuda_internal.h`). **CSA2** (V4.1, L218): a
  528-B E4M3 WINDOW row in every sliding-window ring and a 288-B E2M1 MAIN row
  in each compressed pool ("THE TWO KV ROWS" in `src/pulsar_gpu.h`). Both carry
  the FP4 index-K row. There is no conversion path between row formats — stale
  payloads refuse to load through the payload / segment versions.
  `PULSAR_ATTN_PACK` and `PULSAR_RAW_F16` are no longer switches: the first
  names the format, the second is gone.
  Quantize EXACTLY ONCE (attn_pack_store_kernel); re-encoding decoded FP4
  misrounds ~33% of blocks, so every later move is a byte move.
  Decode attention reads the packed cache natively. Prefill attention reads
  the same packed pool through the fp16 tensor-core tier (L166, `comp_kv`):
  one arm for every entry, no shadow. A native packed read was once measured
  slower than a per-chunk f32 shadow; the f32 kernels and that shadow were
  deleted together, so do not reconstruct them (see the Deferred Work entry
  below for the measured history).

## Environment Variables

44 `PULSAR_*` names are read under `src/` (2026-10-30); only five carry the
`PULSAR_CUDA_*` prefix (`PULSAR_CUDA_PREFILL_CHUNK` and the four
`PULSAR_CUDA_GRAPH_DUMP_*`). Engine and server knobs sit directly under
`PULSAR_` — `PULSAR_MSEQ_BANKS`, `PULSAR_OVERCOMMIT`, `PULSAR_MIXED_BATCH`,
`PULSAR_STALL_WATCHDOG_S`, the `PULSAR_SERVER_*` and `PULSAR_TP_*` families,
`PULSAR_SPEC_DEPTH` / `PULSAR_SPEC_TAU`, the `PULSAR_DSPARK_*` diagnostics, and
others. The ones an operator sets are documented in `docs/SERVER.md`
(Environment variables).

To enumerate them rather than trust this list:

    grep -rhoE '"PULSAR_[A-Z0-9_]+"' src/ | tr -d '"' | sort -u

⚠ Several once-documented switches no longer exist and now do NOTHING if set:
`PULSAR_RAW_F16`, `PULSAR_FP8_NO_MXCORE`, `PULSAR_GGUF_DIR`,
`PULSAR_CUDA_INDEXER_MXFP4`, `PULSAR_CUDA_NO_INDEXED_DECODE_HEADS8`,
`PULSAR_DUMP_ATTN`, `PULSAR_SPLITKV_DEBUG`, `PULSAR_EVAL_PIN`,
`PULSAR_ALLOC_GUARD`, `PULSAR_IDX_FP4`, `PULSAR_ATTN_PACK`,
`PULSAR_WARM_FORK`. Most were retired in the v0.5.0 switch audit, which
deletes any opt-out with no real caller. Setting a deleted flag fails
silently, so check the grep above before relying on one.

## Deferred Work

- ~~Native packed-KV reads in the prefill attention kernels~~ — **MEASURED
  NO-GO 2026-08-02** against the then-current per-(token,row) decode (tried and
  reverted the same day; see the PR #9 trail). Routing the packed comp cache
  into the indexed prefill kernel (its pack branch is bit-exact; gate PASSED)
  measured −1.6% @8k, −1.9% @16k and −2.7% @131k vs the f32 chunk-shadow. The
  bandwidth thesis fails because DSA's top-k keeps the attention working set
  L2-hot at any depth (512 gathered rows/token, heavy overlap between
  neighbors), so the 4x byte saving bought nothing while the per-(token,row)
  e4m3 decode tax was unconditional. L166 later deleted the f32 kernels and the
  shadow together; the packed pool is now the only prefill KV source. Do not
  retry a design that decodes a row more than once per (chunk, row).
