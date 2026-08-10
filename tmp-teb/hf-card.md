---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-DSpark-0731
pipeline_tag: text-generation
tags:
  - gguf
  - deepseek
  - moe
  - quantized
  - 2-bit
  - mxfp4
  - speculative-decoding
  - gb10
---

# DeepSeek-V4-Flash 0731 — IQ2_XXS_MMQ · MXFP4 · MXFP8 · DSpark (ds4 GGUF)

A 92 GB single-file build of
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark-0731)
(284B MoE, 13B active, 1M context) sized to run **fully resident on one NVIDIA
GB10 (128 GB unified memory)** with room for very deep context — the 0731
DSpark speculative-decoding drafter is merged into the same file and
auto-enables on load.

This repo hosts the **latest release artifact only** (`ds4flash-v4.gguf`).
The repo name carries the earlier v3 line's REAP-25 branding; **the v4 line
is not expert-pruned** — it ships the full 256-expert set.  The engine's MLA
KV cache is compact enough (~4 KB per token end-to-end) that the pruning
headroom is no longer needed: measured, the full model serves twelve
concurrent 32k sessions or a 384k-token window with room to spare, and holds
full tool-calling coherence at ~295k live tokens.

Runs **only** on the [pulsar engine](https://github.com/tylerwagler/pulsar).
The file uses custom tensor formats (IQ2_XXS_MMQ aligned pre-store,
MXFP4/MXFP8 microscaling, embedded drafter) that llama.cpp and other GGUF
loaders will not accept.

## What's in the file

The routed-expert formats were chosen by a **measured-KL allocation** — not a
hand rule: per-tensor reconstruction error against the FP8/FP4 QAT source,
Fisher-weighted, allocated under the byte budget by an exact knapsack.

| Component | Treatment |
|---|---|
| Routed experts | **IQ2_XXS floor** (2.06 bpw, ~35 GiB) with **MXFP4** (4.25 bpw, byte-lossless, ~42 GiB) promoted on the quality-sensitive layers — the early stack (layers 0–11) plus four sensitive mid layers. MXFP4 layers run the CUTLASS tensor-core type-40 W4A8 grouped GEMM |
| 2-bit storage | **IQ2_XXS_MMQ (type 43) aligned-SoA pre-store**: the MMQ tensor-core tile layout is baked at quantize time — same 66 bytes per block as IQ2_XXS, byte-identical logits, and the engine starts serving in ~21 s with **no boot-time repack** |
| Expert count | **Full 256 routed experts** — this line is not pruned |
| Attention, shared experts, dense | **MXFP8 (E4M3)** — byte-lossless re-encode of the FP8 QAT source, type-41 MXFP8_LT swizzle, loaded zero-copy |
| LM head | MXFP8 (E4M3), MXFP8_LT swizzle |
| DSpark drafter | The **0731** drafter, embedded in the same GGUF; auto-detected and enabled at load |
| KV cache (runtime) | Packed MXFP8 attention KV + MXFP4 indexer cache — bit-exact with the QAT cache format |

Why MXFP4 for the promoted layers instead of a k-quant: MXFP4 **is** the
checkpoint's source encoding, so it is a byte-lossless re-encode (zero
reconstruction error) at 4.25 bpw. No k-quant can be both lossless and this
small.

## Requirements

- NVIDIA GB10 (SM 12.1, 128 GB unified memory)
- pulsar engine with type-43 (`IQ2_XXS_MMQ`) support, built with
  `CUDA_ARCH=sm_120f`

## Download

Use the engine repo's `./download_model.sh v4`, or:

```sh
hf download twaggs88/DeepSeek-V4-Flash-REAP25-DSpark-ds4-GGUF ds4flash-v4.gguf --local-dir gguf
```
