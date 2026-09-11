/* pulsar_cuda_internal.h — internal shared declarations for the cuda/ translation units.
 * Produced by the multi-TU split of pulsar_cuda.cu; edit freely (the
 * generator is not part of the build).
 *
 * No -rdc: __global__/__device__/__constant__ symbols never cross TU
 * boundaries. The shared __device__ helpers below are static
 * __forceinline__, so each TU gets its own copy. */
#ifndef PULSAR_CUDA_INTERNAL_H
#define PULSAR_CUDA_INTERNAL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cub/block/block_radix_sort.cuh>

#include <stdint.h>
#include <type_traits>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pulsar_gpu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256

enum {
    /* The decode kernels' per-token raw-row scratch: a raw window wider than
     * this is refused at the descriptor check (attention.cu). */
    PULSAR_CUDA_ATTENTION_RAW_SCORE_CAP = 256u,
    PULSAR_CUDA_TOPK_MERGE_GROUP = 8u
};

/* The indexer scorer tier is tiled for exactly this many heads (V4.1: 32,
 * L218; V4 was 64).  Both entry gates and the MXFP4 kernel read THIS. */
#define PULSAR_IDX_MXFP4_HEADS 32u

/*
 * Microscaling (MX / OCP) compressed-KV storage.  One E8M0 (power-of-two)
 * scale byte per BLOCK=32 elements, laid out per row as [data ...][scales ...]:
 *   MXFP8 (E4M3 data): HD data bytes  + NBLK scale bytes  (HD=512 -> 528 B/row)
 *   MXFP4 (E2M1 data): HD/2 nibble bytes + NBLK scale bytes (HD=512 -> 272 B/row)
 * This is the CUTLASS-consumable layout (float_ue8m0_t scales, block 32); the
 * GEMM path re-tiles the scales into CUTLASS's swizzled SF layout at use time.
 */
/* There was a format SELECTOR here (NONE/FP4 plus a rowbytes switch) from when
 * the indexer cache could be f32 or FP4.  It cannot: the cache is FP4, every
 * reader decodes it in place, and the f32 arm had no caller left. */

/*
 * The two KV rows (L218) -- WINDOW (the rings) and MAIN (the kv sources'
 * pools) -- are laid out and documented in src/pulsar_gpu.h, the one
 * definition both sides of the seam read.  Their decoders live below
 * (winkv_row_ld4 / mainkv_row_ld4), the packers in pulsar_cuda_kvrows.cu.
 * Quantise once, move bytes after; there is no other KV row format.
 */

/* Stored Q element type; pairs with PULSAR_Q_ELT_SIZE in pulsar_gpu.h.
 * __half since the L045 flip. */
typedef __half pulsar_q_t;

/* The typedef and the host-side byte macro live in different headers and are
 * both authorities on the same fact.  HC already has this bridge (below); Q --
 * the buffer whose width mismatch actually shipped as defect nine -- did not:
 * changing either alone compiled clean. */
static_assert(PULSAR_Q_ELT_FMT == PULSAR_ELT_F16 &&
              std::is_same<pulsar_q_t, __half>::value,
              "q format tag and pulsar_q_t state the same type; move both or neither");
static_assert(sizeof(pulsar_q_t) == PULSAR_Q_ELT_SIZE,
              "pulsar_q_t and PULSAR_Q_ELT_SIZE state the same width; move both or neither");

/* The cudaDataType the stored Q presents to cuBLAS.  Derived from pulsar_q_t
 * rather than written out, so a GEMM's operand type can never drift from the
 * buffer it reads -- that failure mode is a silent wrong answer, not a fault.
 * Compile-time constant: it folds, so this is not a hot-path branch. */
static const cudaDataType_t PULSAR_Q_CUDA_TYPE =
        (sizeof(pulsar_q_t) == sizeof(float)) ? CUDA_R_32F : CUDA_R_16F;

/* ---- Q-buffer element access (L045) -------------------------------------
 *
 * batch_q is migrating f32 -> f16.  The STORAGE narrows; the ARITHMETIC does
 * not.  Every kernel templated on QT below loads to f32, accumulates in f32,
 * and narrows only at the store -- so the <float> instantiation is bit-identical
 * to the code that shipped, and the <__half> one differs by exactly one
 * round-to-nearest-even per stored element.
 *
 * That distinction is the whole reason this is templated rather than rewritten:
 * the f32 arm must remain provable by the byte-exact prefill gate while the f16
 * arm is graded by cuda-reference-gate, which is a different contract.
 *
 * ⚠ The head RMS norm reads this buffer to form a SUM OF SQUARES over head_dim.
 * Under q_prep_active batch_q holds UNNORMALISED q_b output -- the widest
 * magnitudes in the activation path -- so the <__half> arm narrows the inputs
 * to that reduction, not merely its result.  Measured neutral at mid depths
 * (L045 stage 1), but it is a fidelity change and must never be described as
 * bit-exact. */
template <typename QT>
__device__ __forceinline__ float q_load(const QT *p, uint64_t i);
template <>
__device__ __forceinline__ float q_load<float>(const float *p, uint64_t i) { return p[i]; }
template <>
__device__ __forceinline__ float q_load<__half>(const __half *p, uint64_t i) { return __half2float(p[i]); }

template <typename QT>
__device__ __forceinline__ void q_store(QT *p, uint64_t i, float v);
template <>
__device__ __forceinline__ void q_store<float>(float *p, uint64_t i, float v) { p[i] = v; }
template <>
__device__ __forceinline__ void q_store<__half>(__half *p, uint64_t i, float v) { p[i] = __float2half(v); }

/* e4m3 byte * scale by pure bit math -- bit-identical to the textbook
 * decode (1 + mant/8)*2^(exp-7) * scale with the sign applied (normals become
 * that exact float built directly from its bit pattern;
 * subnormals use the same mant*2^-9 product; scale is an exact power of two,
 * and (-v)*s == -(v*s) in IEEE), but with no exp2f in the inner loops. */
__device__ static inline float pulsar_e4m3_times(uint32_t b, float scale) {
    const uint32_t e = (b >> 3) & 15u;
    const uint32_t m = b & 7u;
    const float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20))
                      : (float)m * 0.001953125f;
    const float sv = v * scale;
    return (b & 0x80u) ? -sv : sv;
}

/* E2M1 magnitude decode: 3-bit code -> value, kept as bit math (not a memory
 * table) so the inner attention loops pay ALU, not LDC traffic. */
__device__ static inline float pulsar_e2m1_times(uint32_t nib, float scale) {
    const uint32_t c = nib & 7u;
    /* codes 0..7 = 0, 0.5, 1, 1.5, 2, 3, 4, 6.  Codes >= 2 are normals
     * (exponent 126 + c>>1, mantissa bit c&1); 0 and 0.5 are the format's
     * zero and subnormal, handled as the exact halves they are. */
    const float v = (c >= 2u) ? __uint_as_float(((126u + (c >> 1)) << 23) | ((c & 1u) << 22))
                              : (float)c * 0.5f;
    const float sv = v * scale;
    return (nib & 8u) ? -sv : sv;
}

/* E8M0 scale byte -> 2^(b - 127) (b == 255 is the format's NaN; never stored). */
__device__ static inline float pulsar_e8m0_scale(uint32_t b) {
    return __uint_as_float(b << 23);
}

/* The reference's fast_round_scale (kernel.py): the E8M0 byte for scale
 * 2^ceil(log2(y)), y = amax * fp_max_inv already formed in fp32, read off y's
 * IEEE fields -- exponent field plus one when the mantissa is non-zero.  NOT
 * log2f/ceilf, which round differently at the boundaries; every E8M0 scale
 * the engine writes (window KV rows, indexer rows) comes from here. */
__device__ static inline uint32_t pulsar_e8m0_round_up(float y) {
    const uint32_t b = __float_as_uint(y);
    const uint32_t e8 = ((b >> 23) & 0xFFu) + (((b & 0x7FFFFFu) != 0u) ? 1u : 0u);
    return e8 > 254u ? 254u : e8;
}

/* The OCP E2M1 nibble codec: [sign:1][magnitude:3], magnitudes 0, 0.5, 1,
 * 1.5, 2, 3, 4, 6 -- CUTLASS float_e2m1_t.  Encode is round-to-nearest with
 * ties to the even code, i.e. hardware RNE. */
__device__ static inline float dsv4_e2m1fn_value_dev(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}
__device__ static inline uint8_t dsv4_e2m1fn_encode_dev(float x) {
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_dev(0));
    for (int i = 1; i < 8; i++) {
        float diff = fabsf(ax - dsv4_e2m1fn_value_dev(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return (uint8_t)((best & 7) | ((x < 0.0f) ? 0x8u : 0u));
}
__device__ static inline float dsv4_e2m1fn_decode_dev(uint8_t nib, float scale) {
    return pulsar_e2m1_times(nib, scale);
}

/* WHERE A BATCH ROW LANDS IN THE RING.  One definition, deliberately: the
 * window packer (quantise then store) and the scatter (copy packed bytes)
 * write the same ring from the same batch, and a disagreement puts a token's
 * KV in the wrong slot -- a position-dependent wrong answer, not a crash.
 * raw_cap == 0: consecutive rows from out_row0 (a pack buffer, not a ring).
 * Returns PULSAR_KV_RING_DEAD_ROW when seq_id puts the row outside the pool;
 * the decision depends only on the row, so a block-uniform early-out. */
#define PULSAR_KV_RING_DEAD_ROW (~0ull)
__device__ __forceinline__ static uint64_t pulsar_kv_ring_slot(
        uint32_t row, uint32_t out_row0, uint32_t raw_cap, uint32_t n_banks,
        const int32_t *__restrict__ positions, const int32_t *__restrict__ seq_id) {
    if (!raw_cap) return (uint64_t)(out_row0 + row);
    if (seq_id && (uint32_t)seq_id[row] >= n_banks) return PULSAR_KV_RING_DEAD_ROW;
    const uint32_t pos = positions ? (uint32_t)positions[row] : out_row0 + row;
    return (uint64_t)(seq_id ? (uint32_t)seq_id[row] * raw_cap : 0u) + pos % raw_cap;
}

/* The sanctioned element reads of the two KV rows (every KV buffer -- see
 * pulsar_gpu.h).  The opaque carriers pulsar_winkv_row_t / pulsar_mainkv_row_t
 * are declared there. */
__device__ static inline float winkv_ld(const pulsar_winkv_row_t *kv, uint64_t row, uint32_t d, uint32_t head_dim) {
    const uint8_t *r = (const uint8_t *)kv + row * PULSAR_WINKV_ROWBYTES(head_dim);
    return pulsar_e4m3_times(r[d], pulsar_e8m0_scale(r[head_dim + d / PULSAR_WINKV_BLOCK]));
}
__device__ static inline float mainkv_ld(const pulsar_mainkv_row_t *kv, uint64_t row, uint32_t d, uint32_t head_dim) {
    const uint8_t *r = (const uint8_t *)kv + row * PULSAR_MAINKV_ROWBYTES(head_dim);
    const float scale = pulsar_e4m3_times(r[head_dim / 2u + d / PULSAR_MAINKV_BLOCK], 1.0f);
    const uint32_t nib = (r[d >> 1] >> ((d & 1u) * 4u)) & 0xFu;
    return pulsar_e2m1_times(nib, scale);
}

/* Four consecutive dims [c4*4, c4*4+4) from a ROW POINTER -- global or the
 * smem copy a cp.async stage filled; byte offsets are row-relative either way.
 * Four dims never straddle a scale block, so one scale each. */
__device__ static inline float4 winkv_row_ld4(const uint8_t *pr, uint32_t c4, uint32_t head_dim) {
    const uint32_t base = c4 << 2;
    const float scale = pulsar_e8m0_scale(pr[head_dim + base / PULSAR_WINKV_BLOCK]);
    const uint32_t w = *(const uint32_t *)(pr + base);
    float4 v;
    v.x = pulsar_e4m3_times(w & 0xFFu, scale);
    v.y = pulsar_e4m3_times((w >> 8) & 0xFFu, scale);
    v.z = pulsar_e4m3_times((w >> 16) & 0xFFu, scale);
    v.w = pulsar_e4m3_times(w >> 24, scale);
    return v;
}
__device__ static inline float4 mainkv_row_ld4(const uint8_t *pr, uint32_t c4, uint32_t head_dim) {
    const uint32_t base = c4 << 2;
    const float scale = pulsar_e4m3_times(pr[head_dim / 2u + base / PULSAR_MAINKV_BLOCK], 1.0f);
    const uint32_t b0 = pr[base >> 1], b1 = pr[(base >> 1) + 1u];
    float4 v;
    v.x = pulsar_e2m1_times(b0 & 0xFu, scale);
    v.y = pulsar_e2m1_times(b0 >> 4, scale);
    v.z = pulsar_e2m1_times(b1 & 0xFu, scale);
    v.w = pulsar_e2m1_times(b1 >> 4, scale);
    return v;
}

/** A device buffer plus the metadata needed to interpret its bytes.
 *
 * Deliberately NOT a shaped tensor: no dimensions, no strides. Shape lives at
 * the call site, which is what lets one allocation be viewed several ways
 * across a layer. What the buffer must carry is how wide an element is and
 * what format it is in -- see `esz` and `fmt`, both of which exist because
 * leaving that knowledge to consumers produced silent, type-legal bugs.
 */
struct pulsar_gpu_tensor {
    void *ptr;        ///< device pointer
    uint64_t bytes;   ///< allocation size
    int owner;        ///< non-zero when this struct owns `ptr` and must free it
    /** Non-zero for a placeholder handed out while pulsar_gpu_tensor_dry_begin
     * is in force: nothing is behind `ptr`, and every device operation on the
     * tensor (fill, write, read, copy, free) is a no-op that reports success.
     * Exists so the allocation code can be run to PRICE a session without
     * touching the device -- one authority for what a session costs. */
    int dry;
    /** Bytes per element.  0 = unspecified, which reads as f32: that keeps
     * every pre-existing alloc meaning exactly what it meant, so only a
     * buffer that is NOT f32 has to say so, once, where it is created.
     *
     * This field exists because the alternative -- every consumer taking an
     * out_f16/q_f16 flag, every bound writing sizeof(float), every view
     * restating the stride -- produced eight distinct defects in one
     * narrowing, all of them type-legal and all of them silent. */
    uint32_t esz;
    /** Element FORMAT (L106 K15).  0 (PULSAR_ELT_F32) for every plain alloc,
     * so pre-existing zero-initialised tensors keep meaning what they meant;
     * set from the *_ELT_FMT authority at alloc_elt, inherited by views. */
    uint32_t fmt;
};

/* The element size to actually use.  Never read t->esz directly. */
static inline uint32_t pulsar_tensor_esz(const pulsar_gpu_tensor *t) {
    return (t && t->esz) ? t->esz : 4u;
}

/* The element FORMAT to actually use (L106 K15).  Never read t->fmt directly.
 * A tensor with no stated format is f32, matching pulsar_tensor_esz's default
 * -- the two defaults must stay in agreement. */
static inline pulsar_elt_fmt pulsar_tensor_fmt(const pulsar_gpu_tensor *t) {
    return (pulsar_elt_fmt)(t ? t->fmt : (uint32_t)PULSAR_ELT_F32);
}

/* Stack sub-view that INHERITS the base's element size.  The hand-rolled
 * aggregate initialisers this replaces dropped it, which silently retyped a
 * narrowed buffer as f32 halfway through a split. */
static inline pulsar_gpu_tensor pulsar_tensor_subview(const pulsar_gpu_tensor *b,
                                                     uint64_t off, uint64_t bytes) {
    pulsar_gpu_tensor t;
    t.ptr = (char *)b->ptr + off;
    t.bytes = bytes;
    t.owner = 0;
    t.fmt = b ? b->fmt : 0u;
    t.esz = b->esz;
    return t;
}

/* Constant-fill of an f32 device buffer.  This lived TWICE, byte-identical and
 * `static`, in pulsar_cuda_norm_kv.cu and pulsar_cuda_runtime.cu -- one of the
 * two duplicate kernel definitions the 2026-08-17 inventory turned up.  It is
 * still `static`, so each TU that uses it still gets its own copy in the binary
 * (cross-TU __global__ linkage would need -rdc=true, which this build does not
 * use); what the move buys is ONE definition to keep correct instead of two
 * that can drift apart silently. */
__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}


/* Hyper-connection residual-stream stored sample type (task #62; see the
 * PULSAR_HC_ELT_SIZE note in pulsar_gpu.h). pulsar_hc_t is the on-device STORAGE type of
 * the six swap-coupled HC residual carriers; loads promote to f32 and stores
 * round from f32 so all in-kernel accumulation stays f32 (torch semantics:
 * bf16 storage, f32 math). Only the carriers use this — flat_hc (RMSNorm out)
 * and hc_mix/hc_split (Sinkhorn control weights) stay f32. sizeof(pulsar_hc_t)
 * MUST equal PULSAR_HC_ELT_SIZE. */
typedef __nv_bfloat16 pulsar_hc_t;
__device__ __forceinline__ static float pulsar_hc_load(const pulsar_hc_t *p, uint64_t i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ static void  pulsar_hc_store(pulsar_hc_t *p, uint64_t i, float v) { p[i] = __float2bfloat16(v); }
static_assert(PULSAR_HC_ELT_FMT == PULSAR_ELT_BF16 &&
              std::is_same<pulsar_hc_t, __nv_bfloat16>::value,
              "hc format tag and pulsar_hc_t state the same type; move both or neither");
static_assert(sizeof(pulsar_hc_t) == PULSAR_HC_ELT_SIZE, "pulsar_hc_t size must match PULSAR_HC_ELT_SIZE");

/* Stored attention-output (heads) element type; pairs with
 * PULSAR_HEADS_ELT_SIZE in pulsar_gpu.h.  Same contract as pulsar_hc_t: the
 * STORAGE narrows, every kernel still loads to f32 and accumulates in f32.
 *
 * FLIPPED to __nv_bfloat16 2026-08-24, after increments 1-7 landed the whole
 * plumbing inert at f32 and the full suite went green on it byte-exact.  The
 * static_assert below is the whole point of declaring the two together: the
 * Q buffer's width mismatch shipped as a real defect because changing either
 * side alone compiled clean. */
typedef __nv_bfloat16 pulsar_heads_t;
/* Both casts are valid for float AND __nv_bfloat16 (the bf16 type carries a
 * float conversion operator and a float constructor), so the flip really is
 * one line in pulsar_gpu.h plus one in the typedef above. */
__device__ __forceinline__ static float heads_load(const pulsar_heads_t *p, uint64_t i) {
    return (float)p[i];
}
__device__ __forceinline__ static void heads_store(pulsar_heads_t *p, uint64_t i, float v) {
    p[i] = (pulsar_heads_t)v;
}
/** The value heads_store(v) leaves in the buffer, as a float: what every E4M3
 * encoder of the heads must quantise (L195: the fused attention epilogue and
 * the rope tail encoded their f32 registers, the read-back encoder the stored
 * bf16 -- one operand, two encodings, different attn-out bytes per arm). */
__device__ __forceinline__ static float heads_round(float v) {
    return (float)(pulsar_heads_t)v;
}
static_assert(PULSAR_HEADS_ELT_FMT == PULSAR_ELT_BF16 &&
              std::is_same<pulsar_heads_t, __nv_bfloat16>::value,
              "heads format tag and pulsar_heads_t state the same type; move both or neither");
static_assert(sizeof(pulsar_heads_t) == PULSAR_HEADS_ELT_SIZE,
              "pulsar_heads_t and PULSAR_HEADS_ELT_SIZE state the same width; move both or neither");

/* A weight tensor stored EITHER f32 or bf16, decided per tensor at load time
 * rather than per build (unlike pulsar_hc_t above, which is one compile-time
 * choice for the whole engine).  Several families moved to bf16 storage to stop
 * paying f32 bytes for values the checkpoint only ever held in bf16; the
 * drafter's f32-source tensors, and any artifact built before that, still
 * carry f32.  Both must work in one binary, so the storage is a template
 * parameter: the inner loop keeps a plain indexed load and there is no branch
 * per element.  Math stays f32 either way -- bf16 is storage, not precision. */
/* NAME THE OTHER HALF. These take a void* and pick the decode from a bool, so
 * nothing in the type system stops you calling the wrong one -- and the false
 * branch is a DIFFERENT WIDTH in each (4 bytes vs 2). Passing an f16 tensor to
 * the f32 variant reads twice the bytes and runs off the end of the tensor,
 * which faults asynchronously and surfaces as a failure in whatever CUDA call
 * comes next. That is exactly what happened on 2026-08-15 with token_embd.
 * The suffix is the contract: check it against the tensor's actual type. */
template <bool BF16>
__device__ __forceinline__ static float pulsar_w_load_f32_or_bf16(const void *w, uint64_t i) {
    if constexpr (BF16) return __bfloat162float(((const __nv_bfloat16 *)w)[i]);
    else                return ((const float *)w)[i];
}

/* Typed weight load: the POINTER carries the storage, so a mismatched width is
 * a compile error rather than a silent wrong-width read. Prefer these over the
 * bool-templated void* pair above for anything newly written -- that pair is
 * what let the embed kernels read an f16 table as f32 on 2026-08-15. */
__device__ __forceinline__ static float pulsar_wt_load(const float *p, uint64_t i) { return p[i]; }
__device__ __forceinline__ static float pulsar_wt_load(const __nv_bfloat16 *p, uint64_t i) { return __bfloat162float(p[i]); }


/* pulsar_w_load_f16_or_bf16<BF16> lived here: the false arm read __half. It had
 * no callers, and the artifact has had ZERO F16 tensors since the source-format
 * migration -- the type survives only as an enum value with no loader and no
 * validator that accepts it. Deleted 2026-08-17 with the __half overload of
 * pulsar_wt_load, for the same reason. */

/* Bytes per element for the above.  Every bounds check on such a weight must
 * use THIS, not sizeof(float): a bf16 tensor is half the bytes, and validating
 * it against the f32 size rejects a legitimate tensor sitting near the end of
 * the file. */
__host__ __device__ __forceinline__ static uint64_t pulsar_w_elt_bytes(int w_bf16) {
    return w_bf16 ? 2u : 4u;
}

/* cuda_block_q8_K was here: the Q8_K activation block, int8 quants with an f32
 * scale and per-16 partial sums.  Zero references in the tree -- the int8
 * activation arms it belonged to are gone and every expert GEMM stages E4M3. */

/* ---- shared types ---- */

/** A span of the memory-mapped model file, and where it lives on the device.
 *
 * A range can reach the GPU two ways: copied into the arena, or the host pages
 * registered so the device addresses them directly. The `registered_*` fields
 * describe the second case and the `arena_allocated` flag the first; they are
 * mutually exclusive, and the flags are what tell teardown which cleanup a
 * given range needs.
 */
struct cuda_model_range {
    const void *host_base;          ///< mapping base this range is an offset into
    uint64_t offset;                ///< byte offset of the range within the mapping
    uint64_t bytes;                 ///< range length
    char *device_ptr;               ///< device address of the range's first byte
    void *registered_base;          ///< page-aligned host base actually registered
    char *registered_device_base;   ///< device address corresponding to registered_base
    uint64_t registered_bytes;      ///< bytes registered (>= `bytes`, page-rounded)
    int host_registered;            ///< the host pages are registered and must be unregistered
    int arena_allocated;            ///< the bytes were copied into the arena instead
};

/** Bump allocator holding model weights on the device. One allocation, carved
 * by offset -- weights are written once at load and never freed individually,
 * so nothing more is needed. */
struct cuda_model_arena {
    char *device_ptr;  ///< base of the arena allocation
    uint64_t bytes;    ///< total arena size
    uint64_t used;     ///< bytes handed out so far; the bump pointer
};

/** An MXFP8 weight matrix on the device: E4M3 values with their block scales
 * kept in a separate table. */
struct fp8_mx_weight {
    const void *host_base;   ///< mapping base the source bytes came from
    uint64_t offset,         ///< byte offset within that mapping
             in_dim,         ///< input width (K)
             out_dim;        ///< output width (N)
    __nv_fp8_e4m3 *data;     ///< device E4M3 values
    unsigned char *scale;    ///< device block scales, one per 32 values
};

/* ---- shared host globals ---- */

extern cublasLtHandle_t g_cublaslt;
extern std::unordered_set<uint64_t> g_fp8_offsets;

/* ---- once-per-shape bookkeeping that never goes quiet (L189) ----
 *
 * The announce and refusal lines that prove which arm ran (rule 5) and which
 * precondition refused (rule 9) are printed once per distinct shape.  Until
 * L189 every such site kept `static uint64_t seen[16]` and stopped RECORDING
 * at the 17th shape, so a new shape after sixteen was neither printed nor
 * counted -- a refusal in silence, forever, at exactly the instrument
 * ENGINEERING-RULES.md section 1 names as the enforcement mechanism.
 *
 * This table records every distinct key it is given (open-addressed, 256
 * slots, one probe on the hot repeat) and counts EVERY call.  When it is full
 * it says so once, with the running call count, and keeps counting; the
 * caller's own return value (refuse / launch) is never affected. */
#define PULSAR_SHAPE_ONCE_CAP 256u
struct pulsar_shape_once {
    uint64_t keys[PULSAR_SHAPE_ONCE_CAP];   ///< 0 = empty; stored keys carry bit 63 so none is 0
    uint32_t n_keys;                         ///< distinct shapes recorded
    uint64_t n_calls;                        ///< every call, listed or not
    int      full_said;                      ///< the "further shapes not listed" line printed
};

/** The (a, b) shape pair as a table key -- (in_dim, out_dim), (q_n, 0), ... */
static inline uint64_t pulsar_shape_key(uint64_t a, uint64_t b) {
    return (a << 32) ^ b;
}

/** Count this call; return 1 exactly once per distinct `key` (the caller then
 * prints its line, which may quote t->n_calls).  A full table prints once, with
 * `what` and the call count, that further shapes are not listed, and returns 0. */
static inline int pulsar_shape_once_first(pulsar_shape_once *t, uint64_t key, const char *what) {
    t->n_calls++;
    const uint64_t k = key | (1ull << 63);
    uint32_t i = (uint32_t)((k ^ (k >> 29) ^ (k >> 47)) & (PULSAR_SHAPE_ONCE_CAP - 1u));
    for (uint32_t probe = 0; probe < PULSAR_SHAPE_ONCE_CAP; probe++) {
        const uint64_t cur = t->keys[i];
        if (cur == k) return 0;
        if (cur == 0) {
            t->keys[i] = k;
            t->n_keys++;
            return 1;
        }
        i = (i + 1u) & (PULSAR_SHAPE_ONCE_CAP - 1u);
    }
    if (!t->full_said) {
        t->full_said = 1;
        fprintf(stderr, "pulsar: %s: %u distinct shapes recorded, table full -- further shapes are NOT "
                        "listed (calls so far: %llu; every call is still counted)\n",
                what, t->n_keys, (unsigned long long)t->n_calls);
    }
    return 0;
}

/* ---- shared host functions ---- */

#include "pulsar_cuda_scratch.h"   /* cuda_tmp_alloc + the slotted bump arena */
void cuda_fp8_weight_cache_clear(void);
/** L188: read-and-clear the routed experts' non-finite flag (pulsar_cuda_moe.cu).
 * Returns 1 and names the first (layer, arm) whose per-token reduction summed a
 * non-finite value, 0 when the flag is clear, -1 on a CUDA error.  Called once
 * per step at the stream drain (pulsar_gpu_end_commands); a 1 fails the step. */
int pulsar_gpu_routed_moe_nonfinite_take(uint32_t *layer_index, const char **arm);
const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what);
int cuda_ok(cudaError_t err, const char *what);

/* ---- shared __device__ inline helpers (per-TU copies; no -rdc) ---- */

__device__ static __forceinline__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

#endif /* PULSAR_CUDA_INTERNAL_H */
