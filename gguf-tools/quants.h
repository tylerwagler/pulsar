#ifndef DS4_QUANTS_H
#define DS4_QUANTS_H

/*
 * Narrow quantization API used by the DS4 GGUF writer.
 *
 * The enum values reuse the GGUF/GGML type-ID numbering so template metadata
 * can be copied without translation, but this is a PRIVATE fork namespace:
 * most IDs (MXFP8, MXFP4, CUTLASS_MXFP4, ...) do not exist upstream, and even
 * the shared ones are written with DS4-specific layouts.  Files produced by
 * these tools load ONLY in the DwarfStar engine, not in llama.cpp/GGML.
 * Only the formats used by the DS4 Flash quantization recipes are implemented
 * as output targets.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4Q_MAX_DIMS 4

typedef enum {
    DS4Q_TYPE_F32     = 0,
    DS4Q_TYPE_F16     = 1,
    DS4Q_TYPE_Q4_0    = 2,
    DS4Q_TYPE_Q4_1    = 3,
    DS4Q_TYPE_Q5_0    = 6,
    DS4Q_TYPE_Q5_1    = 7,
    DS4Q_TYPE_Q8_0    = 8,
    DS4Q_TYPE_Q8_1    = 9,
    DS4Q_TYPE_Q2_K    = 10,
    DS4Q_TYPE_Q3_K    = 11,
    DS4Q_TYPE_Q4_K    = 12,
    DS4Q_TYPE_Q5_K    = 13,
    DS4Q_TYPE_Q6_K    = 14,
    DS4Q_TYPE_Q8_K    = 15,
    DS4Q_TYPE_IQ2_XXS = 16,
    DS4Q_TYPE_IQ2_XS  = 17,
    DS4Q_TYPE_IQ3_XXS = 18,
    DS4Q_TYPE_IQ1_S   = 19,
    DS4Q_TYPE_IQ4_NL  = 20,
    DS4Q_TYPE_IQ3_S   = 21,
    DS4Q_TYPE_IQ2_S   = 22,
    DS4Q_TYPE_IQ4_XS  = 23,
    DS4Q_TYPE_I8      = 24,
    DS4Q_TYPE_I16     = 25,
    DS4Q_TYPE_I32     = 26,
    DS4Q_TYPE_I64     = 27,
    DS4Q_TYPE_F64     = 28,
    DS4Q_TYPE_IQ1_M   = 29,
    DS4Q_TYPE_BF16    = 30,
    DS4Q_TYPE_TQ1_0   = 34,
    DS4Q_TYPE_TQ2_0   = 35,
    DS4Q_TYPE_FP8_E4M3 = 38,   /* MXFP8: E4M3 + per-32 E8M0 block scale (33 B/32) */
    DS4Q_TYPE_MXFP4   = 39,
    DS4Q_TYPE_CUTLASS_MXFP4 = 40, /* per-expert CUTLASS B layout: E2M1 data
                                   * (K-major, byte-verbatim from source) +
                                   * Blackwell 128x4 swizzled E8M0 SF tile */
    DS4Q_TYPE_MXFP8_LT = 41,   /* pre-swizzled FP8 workhorse layout: the type-38
                                * E4M3 weights + E8M0 scales stored in the exact
                                * device layout the engine builds at first use
                                * (de-interleaved [in,out] col-major E4M3 data +
                                * mx_sfoff-swizzled E8M0 scale). Wire-matches the
                                * engine's DS4_TENSOR_MXFP8_LT. */
    DS4Q_TYPE_IQ2_XXS_SOA = 42, /* pre-stored SoA twin of IQ2_XXS (16): the same
                                 * 66 B/block content with the two planes split
                                 * so the weight stream is load-aligned.  See
                                 * ds4q_iq2_xxs_soa_repack() for the byte spec.
                                 * SAME total bytes as type 16 (64 + 2 == 66),
                                 * so dims and byte accounting are unchanged --
                                 * this is a pure permutation, exactly as
                                 * MXFP8_LT (41) is to FP8_E4M3 (38). */
    DS4Q_TYPE_IQ2_XXS_MMQ = 43, /* the OTHER pre-stored twin of IQ2_XXS (16):
                                 * llama.cpp-MMQ's aligned-SoA layout (d plane
                                 * first, q plane 64 B-aligned).  Distinct from
                                 * 42 -- different permutation of the same
                                 * bytes, not interchangeable.  Matches the
                                 * engine's PULSAR_TENSOR_IQ2_XXS_MMQ.  Was
                                 * engine-only until 2026-08-12, which is why
                                 * repack_iq2_mmq.py existed as a separate
                                 * post-pass over the whole artifact. */
    DS4Q_TYPE_COUNT   = 44,
} ds4q_type;

static inline size_t ds4q_pad(size_t x, size_t n) {
    return ((x + n - 1) / n) * n;
}

const char *ds4q_type_name(ds4q_type type);
bool ds4q_can_quantize(ds4q_type type);
int64_t ds4q_block_size(ds4q_type type);
size_t ds4q_row_size(ds4q_type type, int64_t ne);
bool ds4q_requires_imatrix(ds4q_type type);
void ds4q_dequantize_iq2_xxs(const void *blocks, float *out, int64_t n);

/* ---- IQ2_XXS_SOA (type 42) -------------------------------------------------
 * WHY: block_iq2_xxs is 66 bytes with qs[] at offset 2, so a block's code
 * stream is only 2-byte aligned and nvcc must emit LDG.E.U16 -- two 16-bit
 * loads per 32-bit weight word (verified in SASS on the shipped object: the
 * gate/up kernel has NO 64/128-bit global loads at all).  Splitting the block
 * into two planes makes qs 64 B-aligned per block and uint2/uint4-loadable at
 * the SAME byte count.  Measured +7.6% on the gate/up kernel at the production
 * shape, bit-identical output (tests/iq2_soa_bench.cu).
 *
 * LAYOUT, for a tensor of nblk = ne/256 blocks:
 *     [0,            nblk*64) : q plane  -- block b at b*64, the 32 uint16 qs
 *     [nblk*64, nblk*64+nblk*2): d plane -- block b at nblk*64 + b*2, uint16 d
 * Total nblk*66 bytes, identical to type 16.  The q plane leads so that its
 * per-block stride is a power of two from offset 0.
 *
 * Both directions are byte-exact and self-inverse; the engine's device loader
 * and any host consumer must agree with these two functions. */
size_t ds4q_iq2_xxs_soa_qplane_bytes(int64_t ne);
void ds4q_iq2_xxs_soa_repack(const void *packed, void *soa, int64_t ne);
void ds4q_iq2_xxs_soa_unpack(const void *soa, void *packed, int64_t ne);
void ds4q_dequantize_q2_k(const void *blocks, float *out, int64_t n);
void ds4q_dequantize_fp8_e4m3(const void *blocks, float *out, int64_t n);
void ds4q_dequantize_mxfp4(const void *blocks, float *out, int64_t n);

/* CUTLASS_MXFP4 (type 40) helpers.  One expert of shape [nrows=N(out),
 * ncols=K(in)] packs as data (N*K/2 bytes, the source E2M1 [N,K/2] array
 * verbatim) followed by the swizzled scale-factor tile (one E8M0 byte per
 * 32-elem K-block; rows padded to 128, K-blocks padded to 4).  Matches the
 * engine's DS4_TENSOR_CUTLASS_MXFP4 and the CUTLASS Sm1xx SFB atom layout;
 * validated byte-identical to the mxfp4_pack_source_cli splice output. */
size_t ds4q_cutlass_mxfp4_sf_bytes(int64_t nrows, int64_t ncols);
size_t ds4q_cutlass_mxfp4_bytes(int64_t nrows, int64_t ncols);
void ds4q_pack_cutlass_mxfp4(const uint8_t *e2m1, const uint8_t *e8m0,
                             void *dst, int64_t nrows, int64_t ncols);

/* MXFP8_LT (type 41) helpers.  One workhorse tensor of shape [nrows=out(N),
 * ncols=in(K)] packs as E4M3 data (nrows*ncols bytes, de-interleaved from the
 * type-38 [E8M0][32 x E4M3] blocks to [out,in] row-major) followed by the
 * mx_sfoff-swizzled E8M0 scale (rows padded to 128, K-blocks padded to 4).
 * Byte-identical to repack_tensor() in tools/mxfp8_prestore/repack_mxfp8_lt.py
 * and to the engine's DS4_TENSOR_MXFP8_LT device layout. */
size_t ds4q_mxfp8_lt_sf_bytes(int64_t nrows, int64_t ncols);
size_t ds4q_mxfp8_lt_bytes(int64_t nrows, int64_t ncols);
void ds4q_pack_mxfp8_lt(const uint8_t *fp8_blocks, void *dst,
                        int64_t nrows, int64_t ncols);

/* IQ2_XXS_MMQ (type 43) helpers.  A pure permutation of the raw IQ2_XXS (16)
 * block stream -- same 66 B/block of content, same total size -- into the two
 * planes the vendored llama.cpp MMQ kernels read.  Raw blocks are
 * [__half d][32 x uint16 qs] = 2 + 64 B, only 2-byte aligned, which costs MMQ
 * two 16-bit loads per 32-bit weight word (measured 44.704 -> 18.654 ms per
 * pair-call at the production shape, 2.40x).  Layout, nblk = ne/256:
 *     dq = align_up(nblk*2, 64)
 *     [0,  nblk*2)   d plane, block b's half at b*2
 *     [.., dq)       zero pad to 64 B
 *     [dq, +nblk*64) q plane, block b's 64 qs bytes at dq + b*64
 * Size identity (and therefore an in-place layout change that preserves every
 * gguf data offset) requires nblk % 32 == 0; the packer refuses otherwise.
 * Byte-identical to gguf-tools/repack_iq2_mmq.py (the offline producer of this
 * layout; the on-device ds4_repack_iq2_aligned_device()/ds4_repack.cu twin was
 * removed -- the device side reads this layout directly through the MMQ SoA loaders).
 * NOT interchangeable with IQ2_XXS_SOA (42), which is a different permutation
 * (q plane first, no padding). */
size_t ds4q_iq2_xxs_mmq_bytes(int64_t nblk);
void ds4q_pack_iq2_xxs_mmq(const uint8_t *iq2_blocks, void *dst, int64_t nblk);

/* The canonical Blackwell 128x4 scale-factor swizzle, shared by the CUTLASS
 * MXFP4 and MXFP8_LT packers.  Byte-identical to the __device__ mx_sfoff in
 * src/cuda/ds4_cuda_matmul.cu and build_scale_dest_index() in
 * tools/mxfp8_prestore/repack_mxfp8_lt.py. */
size_t ds4q_mx_sfoff(int64_t row, int64_t kb, int64_t kbp);
void ds4q_quantize_init(ds4q_type type);
size_t ds4q_quantize_chunk(ds4q_type type, const float *src, void *dst,
                           int64_t start, int64_t nrows, int64_t ncols,
                           const float *imatrix);

float ds4q_f16_to_f32(uint16_t bits);
float ds4q_bf16_to_f32(uint16_t bits);
void ds4q_f32_to_f16_row(const float *src, uint16_t *dst, int64_t n);
void ds4q_f32_to_bf16_row(const float *src, uint16_t *dst, int64_t n);

#endif
