# Vendored: CUTLASS's memory-efficient attention (examples/41_fused_multi_head_attention)

The forward kernel of xformers' memory-efficient attention as CUTLASS ships it in its examples, **verbatim** at the
CUTLASS pin (`cutlass.pin`, v4.8.1 `2c85f497`; byte-identical to v4.8.0 `098de2a6`, where it was first copied): `kernel_forward.h` and the include closure it
needs (`gemm/`, `iterators/`, `epilogue/`, `transform/`, `gemm_kernel_utils.h`, `debug_utils.h`).  BSD-3-Clause,
NVIDIA's header in every file.  Not part of CUTLASS's `include/` tree, which is all the pin's build reads -- hence a
copy here, as `src/cuda/mmq/` holds the llama.cpp pieces.

Used by `src/cuda/pulsar_cuda_fmha.cu` (L321): the Sm75 instance (fp16, 32 x 128 blocks, head dims up to 128) for
Nemotron Parse's RADIO ViT -- one bidirectional run over a page's ~13k patches.

Refresh: re-copy the same file list from the new pin's `examples/41_fused_multi_head_attention/` (the closure of
`kernel_forward.h`'s quoted includes), then `make fmha-gate`.  Do not edit these files; a change goes in the wrapper.
