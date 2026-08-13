# Engine code review — 2026-08-13 (static)

> **STATUS (updated 2026-08-13, after the findings were acted on).** This is the
> review as written — a static read with no GPU. Several findings have since been
> measured or landed, so do not act on a Part A/B item without checking here first:
>
> - **Part A (A1-A6): all landed** — `0ab4b0e` (spill durability + startup sweep),
>   `8ce9ba9` (hot-path getenv), `36541ae` (sampler scratch), `414247c` (dspark conf
>   scratch), `cde10f4` (redundant `project_main_x`), `c2d3af1` (kvstore
>   supersedes-once). GPU-validated by `cuda-spec-sampling-gate` and `cuda-fork-gate`.
>   A6 is HALF done: the `refresh()` directory-rescan half was deliberately skipped
>   (caching it needs invalidation on four mutation paths; getting it wrong serves a
>   stale index).
> - **B5: MEASURED WORTHLESS, then fixed anyway.** `attn_pack_dequant_kernel` was
>   0.00% of GPU time. Fixed for cleanliness (`671c1ba`, `fe7dd1b`, `92bb9e0`):
>   launches 229 -> 61. The residual 61 are a **kernel capability gap**, not filth —
>   `prefill_static_mixed_heads_tensor` has no `comp_kv_pack` parameter and
>   genuinely requires the f32 shadow.
> - **B6 <= 4.1%, B7 ~ 2.2%, B4 bounded by 4.7%** from an 8k nsys trace; B6+B7
>   together ~6.3% is the real fusion prize against a 13.7% total producer class.
> - **B1/B2/B3 NOT priced, and cannot be by `pulsar-bench`** — it has no speculation
>   support, so those per-spec-step paths never execute. Needs pulsar-server profiled
>   under speculative load.
> - **The §8 MXFP8 re-quantization audit was independently resolved as ABSENT** in
>   our pipeline (ledger L022).
>
> Headline the review did not predict: prefill measures **935 t/s at 8k**, against a
> working figure of 376 — see ledger L017.

Static four-scope review of the engine + server, no GPU available: decode/spec
hot path, prefill producer fusion, server/KV, and the MXFP8 weight-converter
audit. Every finding carries file:line evidence from a code read; anything
requiring a real model is marked **needs measurement** and belongs in the
pending nsys prefill audit.

## Settled ground (do not re-open)

Decode is at the bandwidth roofline (projections measured 97-111% of
achievable); split-KV decode attention, head-grouping, f16 prefill attention,
the indexer MXFP4 scorer, and the D2R/MMQ MoE arc are shipped. CUDA graphs
were measured dead (98.5% GPU-busy). Q8_K activations in the routed MoE are
permanent. Spec draft-3 is the measured optimum. This review only reports
things outside that ground.

## MXFP8 weight re-quantization audit — ABSENT

The b12x §8 bug class (round an arbitrary fp32 128x128 block scale to UE8M0
while keeping the original FP8 values; ~2.7x excess weight error) does NOT
exist in our pipeline:

- The HF checkpoint's scale tensors are already F8_E8M0 (e.g.
  `layers.0.attn.wo_a.scale` = F8_E8M0 [64,32]); the (w_fp8, fp32 scale) pair
  never enters the converter. An fp32 scale tensor dies at load
  (`gguf-tools/quantize/dsq_codecs.c:60,70`).
- Every in-repo scale transform either re-derives values under the rounded
  scale (`ds4q_quantize_fp8_e4m3`, `gguf-tools/quantize/quants_fp.c:54-78`) or
  copies value bytes verbatim with a scale-only swizzle
  (`ds4q_pack_mxfp8_lt`, quants_fp.c:225-254). The runtime type-38->device and
  type-41 MXFP8_LT paths copy bytes verbatim, zero-copy
  (`src/cuda/pulsar_cuda_matmul.cu:529-601`).

Two residuals, both cheap:

1. E8M0 byte 255 decodes to +Inf unguarded at runtime (`dsq_codecs.c:8`).
   Writers clamp to <=254, but a foreign or corrupt GGUF can carry 255 —
   worth a load-time die, same class as the NaN policy.
2. No byte-exact parity test exists against DeepGEMM `per_block_cast_to_fp8`
   semantics; `--compare-tensor` only validates pipeline self-consistency.

## Part A — cheap fixes (no GPU; host-only or bit-exact)

### A1. [ROBUSTNESS] Guard-spill file has no durability, then the only copy is freed

`src/server/server_sched.cpp:1428-1457`: the spill path writes with plain
`fopen("wb")`+`fclose` — no fsync, no atomic rename — and then calls
`bank_free_physical` on the source. The spill is the ONLY physical copy after
that point. A crash or power loss leaves a truncated spill and that
conversation permanently 500s (bank load refuses it). The disk-KV cache store
received exactly this durability fix earlier; the spill path never did. Fix:
write to `.tmp.<pid>`, fsync, rename, then free.

### A2. [WASTE] getenv on hot paths — two live violators

- `src/engine/gpu_prefill.cpp:2744`: `getenv("PULSAR_CUDA_GRAPH_TOKEN_PROFILE")`
  inside `gpu_graph_eval_token_raw_swa`, called per token.
- `src/cuda/pulsar_cuda_moe.cu:1412`: `PULSAR_CUDA_MOE_PROFILE` read on every
  type-43 MoE launch (every layer, every decode token).
- Per-step dump-hook getenvs: `src/engine/session_spec.cpp:375-378,1007`.

All violate the no-hot-path-flags rule; hoist to cached statics like the
existing `gpu_graph_env_flag` pattern.

### A3. [WASTE] Sampler bypasses its own scratch

`src/engine/tokenizer.cpp:1185-1213`: `sample_full_vocab` mallocs cand/keys/
tmp up to ~5 MB per sampled token (top_p<1, no top_k — the common client
shape), while the reusable `pulsar_sample_scratch` exists and this path
bypasses it.

### A4. [WASTE] Per-step device alloc/free in the spec path

`src/engine/session_spec.cpp:1073-1074` (twin 1373-1374): `conf_dev`/`tok_dev`
are `cudaMalloc`'d and freed every fused spec step; `tokens_t` alloc/free per
draft step (`src/engine/gpu_decode.cpp:1780,1795`). The tree itself documents
this pattern as device-serializing where it was fixed elsewhere
(`gpu_decode.cpp:1676-1678`, `src/cuda/pulsar_cuda_dspark.cu:218-223`).
Persist the scratch.

### A5. [WASTE] Redundant project_main_x

`src/engine/session_spec.cpp:893` re-projects `main_x` after the seeding loop
already left it at the same value (comment admits "for clarity"): 3 sync
copies + gemv + norm wasted per step. Verify the invariant, drop the
redundant pass.

### A6. [WASTE] kvstore text-path rescan + O(n^2) eviction re-hash

`src/lib/pulsar_kvstore.cpp:424` `refresh()` (opendir + stat/read-header of
every `.kv` file) runs per request via `find_text_prefix` (:600) and per
store via `evict()` (:456); a single-process server invalidates nothing
between calls. And `:461-496` re-hashes the incoming text (SHA1 of up to
~100 KB) per CONTINUED entry per eviction pass — invariant across passes.
Index the directory once; hoist the incoming hash.

## Part B — measurement-gated (sparky; price in one nsys session)

### B1. DSpark seeding copy-storm

`pulsar_gpu_tensor_copy` is a synchronous D2D `cudaMemcpy`
(`src/cuda/pulsar_cuda_runtime.cu:1358-1370`). The per-step seed path
(`session_spec.cpp:445-458` -> `gpu_decode.cpp:1723-1758`) fires ~12 blocking
copies + ~20 tiny launches per seeded row, up to 4 rows per fused step, after
the verify drain. Candidate: one batched seed kernel. Distinct from the
prompt-window seeding question (one-time TTFT ~40 ms); this is per-step.

### B2. Stage-B comp saves are unconditional

`src/engine/gpu_prefill.cpp:906-913`: 2 D2D copies per compressed layer
(~86 launches/step) every fused step, consumed only on partial accepts.
Candidate: save lazily on the partial-accept path.

### B3. Triple device drain per spec step

Snapshot sync (`session_spec.cpp:315`) + verify sync after the layer sweep
(`src/engine/imatrix.cpp:951`) + after the head encode (:1002). The
mid-verify drain is removable: the head encode can enqueue on the same stream
before one drain.

### B4. Act cache not armed for batch_ffn_norm

The MXFP8 activation cache is armed only for `batch_attn_norm`
(`gpu_prefill.cpp:576`). The FFN section runs 3 MXFP8 quants + 1 f16
conversion of `batch_ffn_norm` per layer (`gpu_prefill.cpp:2524-2551`), and
the verify/draft forward runs gate/up as two separate GEMMs with a standalone
swiglu, where decode already ships a fused `shared_gate_up_swiglu_mxfp8`
kernel. Same coherence invariant as the shipped attn_norm arm — bit-exact.

### B5. Prefill shadow dequant rebuilt per slice span

`attn_pack_dequant_kernel` re-dequants the full compressed cache per
`PULSAR_PREFILL_SLICE` span (`gpu_prefill.cpp:1924/2091`): 8 spans x all 1024
comp rows ~ 22 MB/layer/chunk vs 2.7 MB once. A valid-flag on
`attn_comp_dequant` is bit-exact: already-stored packed rows are read-only
between spans. Distinct from the measured packed-KV-reads NO-GO — this is
redundant dequant work, not a format change.

### B6. flat_hc norm -> f16 triple-emit

`rms_norm_plain` -> f32 `batch_flat_hc` (hc_dim 16384, 64 KB/token) ->
`f32_to_f16_kernel`, twice per layer (`gpu_prefill.cpp:498,2411`). The norm
producer could emit f16 with identical RNE in one pass — the producer-fusion
class (norm emits f32 + f16 + quantized), bit-exact.

### B7. MoE gather + pack fusion

`pack_act_e4m3_rowmajor_vec` re-reads the gathered activation (96 KB/token)
that the gather kernel just wrote (`src/cuda/pulsar_cuda_moe.cu:412` vs
`src/cuda/pulsar_mxfp4_cutlass.cu:665`). Pack-during-gather is bit-exact
(same codes).

## Caveats

- All file:line refs are from a static read; re-verify at implementation time.
- A1-A6 need no model; each is host-only or byte-identical by construction.
- B1-B7 need measurement — decode and prefill are near their rooflines, so
  price each before spending.
- Spec-path cleanup (A4/A5, B1-B3) was previously parked on "speculation
  doesn't pay"; DSpark now wins at draft-2/3, so that gate is void.

## Order of work

1. A1 (correctness), A2, A3, A4, A5, A6 — fix batch, each gated.
2. One nsys session prices B1-B7 against the prefill/decode maps.
3. Fold survivors into the prefill/decode plans.
