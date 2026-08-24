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
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context.  11712 fits under the
     * GB10 48 KB shared-memory limit. */
    PULSAR_CUDA_ATTENTION_SCORE_CAP = 11712u,
    PULSAR_CUDA_ATTENTION_RAW_SCORE_CAP = 256u,
    PULSAR_CUDA_TOPK_MERGE_GROUP = 8u
};

#define PULSAR_FP8_KV_BLOCK 64u
#define PULSAR_FP8_KV_NBLK(HD) (((HD) + PULSAR_FP8_KV_BLOCK - 1u) / PULSAR_FP8_KV_BLOCK)
#define PULSAR_FP8_KV_ROWBYTES(HD) ((HD) + PULSAR_FP8_KV_NBLK(HD) * sizeof(float))

/*
 * Microscaling (MX / OCP) compressed-KV storage.  One E8M0 (power-of-two)
 * scale byte per BLOCK=32 elements, laid out per row as [data ...][scales ...]:
 *   MXFP8 (E4M3 data): HD data bytes  + NBLK scale bytes  (HD=512 -> 528 B/row)
 *   MXFP4 (E2M1 data): HD/2 nibble bytes + NBLK scale bytes (HD=512 -> 272 B/row)
 * This is the CUTLASS-consumable layout (float_ue8m0_t scales, block 32); the
 * GEMM path re-tiles the scales into CUTLASS's swizzled SF layout at use time.
 */
#define PULSAR_MXKV_BLOCK 32u
#define PULSAR_MXKV_NBLK(HD) (((HD) + PULSAR_MXKV_BLOCK - 1u) / PULSAR_MXKV_BLOCK)
#define PULSAR_MXKV_FP4_ROWBYTES(HD) (((HD) + 1u) / 2u + PULSAR_MXKV_NBLK(HD))
/* There was a format SELECTOR here (NONE/FP4 plus a rowbytes switch) from when
 * the indexer cache could be f32 or FP4.  It cannot: the cache is FP4, every
 * reader decodes it in place, and the f32 arm had no caller left. */

/*
 * PULSAR_ATTN_PACK compressed-KV storage (value-preserving).  A comp row today is
 * f32 with the nope dims already fp8-roundtripped in place (fp8_kv_quantize:
 * per-64 block scale = exp2f(ceilf(log2f(fmaxf(amax,1e-4)/448))), e4m3
 * clamp-roundtrip) and the n_rot rope tail bf16-roundtripped in place.  The
 * packed row stores exactly those values:
 *   [n_nope e4m3 bytes][n_nope/64 E8M0 scale bytes][pad to 4B][n_rot bf16]
 * head_dim 512 / n_rot 64 -> 448 + 7 + 1 + 128 = 584 B/row (vs 2048 f32).
 * The E8M0 byte is the scale exponent + 127 (power-of-two by construction),
 * so decode (e4m3 value * 2^(e8-127); rope __bfloat162float) is bit-identical
 * to the f32 cache.  Must stay in sync with PULSAR_ENGINE_ATTN_PACK_ROWBYTES.
 *
 * This row is BYTE-IDENTICAL to vLLM's `fp8_ds_mla` DSv4 cache line (448 e4m3 +
 * 7 ue8m0-per-64 + 1 pad + 64 bf16 = 584 B).  The rope tail was f32 (712 B/row)
 * until 2026-08-11; the source model runs a BF16 residual, so f32 there stored
 * precision the model never had.  Keep the two layouts in step: a divergence
 * costs the cross-engine KV diff that validates this cache against a stock vLLM
 * run.  Bumping this row layout MUST bump PULSAR_SESSION_PAYLOAD_VERSION.
 */
#define PULSAR_ATTN_PACK_NROT 64u
#define PULSAR_ATTN_PACK_NOPE(HD) ((HD) - PULSAR_ATTN_PACK_NROT)
#define PULSAR_ATTN_PACK_SCALES_PAD(HD) \
    ((PULSAR_ATTN_PACK_NOPE(HD) / PULSAR_FP8_KV_BLOCK + 3u) & ~3u)
#define PULSAR_ATTN_PACK_ROWBYTES(HD) \
    ((uint64_t)PULSAR_ATTN_PACK_NOPE(HD) + PULSAR_ATTN_PACK_SCALES_PAD(HD) + \
     (uint64_t)PULSAR_ATTN_PACK_NROT * 2u)

/* Default-ON env gate for the measured fast tiers.  Suite-v1 KL cleared the
 * fp16 attention and MXFP4 indexer tiers on 2026-08-08 (exact per-position
 * KL(off||on) p95 well inside the quant's own divergence-from-reference
 * budget; docs/engine-perf-map.md, "fidelity ledger"), so presence-means-on
 * flipped to on-unless-"0".  Setting the variable to "0" opts out; any other
 * value (or unset) leaves the tier enabled. */
static inline int pulsar_env_tier_on(const char *name) {
    const char *v = getenv(name);
    return !(v && v[0] == '0' && v[1] == '\0');
}

/* Stored Q element type; pairs with PULSAR_Q_ELT_SIZE in pulsar_gpu.h.
 * __half since the L045 flip. */
typedef __half pulsar_q_t;

/* The typedef and the host-side byte macro live in different headers and are
 * both authorities on the same fact.  HC already has this bridge (below); Q --
 * the buffer whose width mismatch actually shipped as defect nine -- did not:
 * changing either alone compiled clean. */
static_assert(sizeof(pulsar_q_t) == PULSAR_Q_ELT_SIZE,
              "pulsar_q_t and PULSAR_Q_ELT_SIZE state the same width; move both or neither");

uint32_t pulsar_gpu_act_f32_first_present_row(const void *ptr, uint64_t n_tok,
                                              uint64_t in_dim);

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

/* Four consecutive Q values as a float4, indexed in units of FOUR ELEMENTS.
 *
 * The heads8 kernels give each lane dims 4*lane + 128k .. +3 via q4[lane+32k].
 * Keeping that index in element-quads means the f16 arm reads the SAME dims as
 * the f32 arm -- an 8-byte load instead of 16 -- so the lane->dim mapping and
 * the shared-memory K layout are untouched.  Values are unpacked to f32 on the
 * spot, so the dot products and their register footprint are unchanged. */
template <typename QT>
__device__ __forceinline__ float4 q_load4(const QT *p, uint32_t i4);
template <>
__device__ __forceinline__ float4 q_load4<float>(const float *p, uint32_t i4) {
    return ((const float4 *)p)[i4];
}
template <>
__device__ __forceinline__ float4 q_load4<__half>(const __half *p, uint32_t i4) {
    const uint2 raw = ((const uint2 *)p)[i4];          /* 4 halves, 8 B */
    const __half2 lo = *(const __half2 *)&raw.x;
    const __half2 hi = *(const __half2 *)&raw.y;
    const float2 a = __half22float2(lo);
    const float2 b = __half22float2(hi);
    return make_float4(a.x, a.y, b.x, b.y);
}

template <typename QT>
__device__ __forceinline__ void q_store(QT *p, uint64_t i, float v);
template <>
__device__ __forceinline__ void q_store<float>(float *p, uint64_t i, float v) { p[i] = v; }
template <>
__device__ __forceinline__ void q_store<__half>(__half *p, uint64_t i, float v) { p[i] = __float2half(v); }

/* e4m3 byte * scale by pure bit math — bit-identical to the textbook
 * decode (1 + mant/8)*2^(exp-7) * scale with the sign applied (normals become
 * that exact float built directly from its bit pattern;
 * subnormals use the same mant*2^-9 product; scale is an exact power of two,
 * and (-v)*s == -(v*s) in IEEE), but with no exp2f in the inner loops. */
__device__ static inline float attn_pack_e4m3(uint32_t b, float scale) {
    const uint32_t e = (b >> 3) & 15u;
    const uint32_t m = b & 7u;
    const float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20))
                      : (float)m * 0.001953125f;
    const float sv = v * scale;
    return (b & 0x80u) ? -sv : sv;
}

/* The opaque packed-row carriers (pulsar_attn_pack_t / pulsar_mxkv_pack_t)
 * are declared in pulsar_gpu.h (L092); the accessors below are the only
 * sanctioned element reads. */
__device__ static inline float attn_comp_pack_ld(const pulsar_attn_pack_t *comp_kv, uint64_t row, uint32_t d, uint32_t head_dim) {
    const uint32_t n_nope = head_dim - PULSAR_ATTN_PACK_NROT;
    const uint8_t *r = (const uint8_t *)comp_kv + row * PULSAR_ATTN_PACK_ROWBYTES(head_dim);
    if (d < n_nope) {
        const float scale = __uint_as_float((uint32_t)r[n_nope + (d / PULSAR_FP8_KV_BLOCK)] << 23);
        return attn_pack_e4m3(r[d], scale);
    }
    return __bfloat162float(((const __nv_bfloat16 *)(r + n_nope +
                             PULSAR_ATTN_PACK_SCALES_PAD(head_dim)))[d - n_nope]);
}

struct pulsar_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
    /* Bytes per element.  0 = unspecified, which reads as f32: that keeps
     * every pre-existing alloc meaning exactly what it meant, so only a
     * buffer that is NOT f32 has to say so, once, where it is created.
     *
     * This field exists because the alternative -- every consumer taking an
     * out_f16/q_f16 flag, every bound writing sizeof(float), every view
     * restating the stride -- produced eight distinct defects in one
     * narrowing, all of them type-legal and all of them silent. */
    uint32_t esz;
};

/* The element size to actually use.  Never read t->esz directly. */
static inline uint32_t pulsar_tensor_esz(const pulsar_gpu_tensor *t) {
    return (t && t->esz) ? t->esz : 4u;
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

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

/* ---- shared types ---- */

struct cuda_model_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered;
    int arena_allocated;
};

struct cuda_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};

struct fp8_mx_weight { const void *host_base; uint64_t offset, in_dim, out_dim; __nv_fp8_e4m3 *data; unsigned char *scale; };

/* ---- shared host globals ---- */

extern cublasHandle_t g_cublas;
extern int g_cublas_ready;
extern cublasLtHandle_t g_cublaslt;
extern std::unordered_set<uint64_t> g_fp8_offsets;

/* ---- shared host functions ---- */

#include "pulsar_cuda_scratch.h"   /* cuda_tmp_alloc + the slotted bump arena */
void cuda_fp8_weight_cache_clear(void);
int cuda_attention_score_buffer_fits(uint32_t n_comp);
const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what);
int cuda_ok(cudaError_t err, const char *what);
int cublas_ok(cublasStatus_t st, const char *what);
int cuda_matmul_fp8_hc_expand_tensor_labeled(
        pulsar_gpu_tensor       *out_hc,
        pulsar_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        const pulsar_gpu_tensor *block_add,
        const pulsar_gpu_tensor *residual_hc,
        const pulsar_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label);

/* ---- shared __device__ inline helpers (per-TU copies; no -rdc) ---- */

__device__ static __forceinline__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static __forceinline__ float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

#endif /* PULSAR_CUDA_INTERNAL_H */
