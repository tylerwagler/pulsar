// SPDX-License-Identifier: MIT
/** @file
 *  BF16 weights against raw bf16 rows: the dense Linear and the routed experts (L295 F7). */
// L295 F7: BF16 weights against raw bf16 rows -- the dense Linear and the routed experts (contract and arms in
// pulsar_cuda_bf16_rows.cuh).
//
// The ROWS kernel.  A CTA owns 128 weight rows (8 warps x the MMA's M of 16) and up to 8 * NT activation rows (the
// MMA's N, NT n8 tiles); it walks K in chunks, staging the chunk's activation rows in shared memory (a row stride of
// 16 mod 128 bytes: a quarter-warp's 16-byte loads hit distinct bank groups), and each warp streams its 16 weight
// rows once.  The k order inside a 64-wide step is permuted -- the same permutation on both operands is the same dot
// product -- so lane (g, c) loads 16 CONTIGUOUS weights of each of its two rows (two uint4) and the matching 16 bf16
// of its activation row: logical k {2c, 2c+1 | 2c+8, 2c+9} of 16-k sub-step s is actual k 16c + 4s + {0, 1 | 2, 3}
// (the mxfp8 rows kernel's mapping, pulsar_cuda_qwen_gr.cu).  bf16 x bf16 is exact in f32; each 16-k MMA goes into a
// fresh fragment that is added into the row's f32 accumulator in k order, so a row's bytes are a function of its own
// activation and the weight alone -- not of NT, the chunking, or the other rows of the call.
//
// The routed arm runs the same kernel per work item: an expert's assignments [start, start + count) of the
// expert-sorted schedule (count <= 64), the activation rows gathered by the schedule's source index, the outputs
// scattered by its destination (the pair).  Gate and up are one launch (gridDim.z), the down another.

#include "pulsar_cuda_bf16_rows.cuh"
#include "pulsar_cuda_mx.cuh"   /* pulsar_swiglu_elem: the fold's expression */
#ifdef PULSAR_HAVE_MMQ
#include "mmq/ds4_mmid.cuh"
#include "mmq/ds4_mmq.h"
#include "pulsar_cuda_scratch.h"
#include "pulsar_cuda_arch.cuh"
#endif

#include <cuda_bf16.h>
#include <stdio.h>

namespace {

constexpr int kWarps = 8;              ///< 16 weight rows each: 128 outputs a CTA
/// Weight rows (outputs) one CTA owns: 128.
constexpr int kOutPerCta = 16 * kWarps;
constexpr int kStageBytes = 32768;     ///< the staged activation chunk, at most (no opt-in shared memory)
constexpr int kRoutedNT = 8;           ///< the routed arm's n8 tiles: 64 assignments a work item
constexpr int kMaxE = 1024;            ///< experts the one-block plan kernel scans

/** Elements of K a chunk stages for NT n8 tiles: kStageBytes over 8 NT rows, a multiple of 64. */
/// @tparam NT n8 activation tiles per CTA (1, 2 or kRoutedNT).
/// @return K elements per staged chunk.
template <int NT> __host__ __device__ constexpr int chunk_k() { return kStageBytes / (8 * NT) / 2 / 64 * 64; }
/// @tparam NT n8 activation tiles per CTA.
/// @return Bytes between staged activation rows: chunk_k * 2 + 16, i.e. 16 mod 128 (bank-group spread).
template <int NT> __host__ __device__ constexpr int stage_stride() { return chunk_k<NT>() * 2 + 16; }   /* 16 mod 128 */
/// @tparam NT n8 activation tiles per CTA.
/// @return Dynamic shared memory of one CTA's staged chunk: 8 NT rows x stage_stride.
template <int NT> __host__ __device__ constexpr size_t stage_bytes() { return (size_t)8 * NT * stage_stride<NT>(); }
static_assert(stage_bytes<1>() <= 48 * 1024 && stage_bytes<2>() <= 48 * 1024 && stage_bytes<kRoutedNT>() <= 48 * 1024,
              "the staged chunk must fit the default dynamic shared memory");


/** One CTA's 128 outputs [o0, o0 + 128) of W [out][in] for nr <= 8 NT activation rows: row i of the call is
 *  X[xs(i)] (xs = xidx ? xidx[start + i] : start + i) and lands in Y[yd(i)] (yd likewise from yidx), ld out. */
/// Launch contract: 256 threads (8 warps x 16 weight rows), `hx` = stage_bytes<NT>() of 16-byte-aligned dynamic
/// shared memory; `in` a multiple of 64 and W / X rows 16-byte aligned.
/// @tparam NT n8 activation tiles (nr <= 8 NT).
/// @param W the weight, bf16 row-major [out][in].
/// @param out weight rows (output width).
/// @param in the reduction width K.
/// @param o0 the CTA's first output.
/// @param X the activation rows, bf16 [*][in].
/// @param xidx source row map (NULL: identity from `start`).
/// @param yidx destination row map (NULL: identity from `start`).
/// @param start the work item's first schedule index.
/// @param nr rows in the work item.
/// @param Y the output, f32 [*][out].
/// @param hx the staged activation chunk (dynamic shared memory).
template <int NT>
__device__ __forceinline__ void rows_tile(const __nv_bfloat16 *__restrict__ W, int out, int in, int o0,
                                          const __nv_bfloat16 *__restrict__ X, const int32_t *__restrict__ xidx,
                                          const int32_t *__restrict__ yidx, int start, int nr, float *__restrict__ Y,
                                          uint8_t *hx) {
    constexpr int KC = chunk_k<NT>(), STRIDE = stage_stride<NT>();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int r0 = o0 + warp * 16 + g, r1 = r0 + 8;
    const bool v0 = r0 < out, v1 = r1 < out;
    const int nt = (nr + 7) >> 3;   /* the n8 tiles holding rows: warp-uniform */
    const __nv_bfloat16 *w0 = W + (size_t)(v0 ? r0 : 0) * in + 16 * c;
    const __nv_bfloat16 *w1 = W + (size_t)(v1 ? r1 : 0) * in + 16 * c;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
    for (int k0 = 0; k0 < in; k0 += KC) {
        const int kc = in - k0 < KC ? in - k0 : KC, q16 = kc / 8;
        __syncthreads();   /* the previous chunk is consumed */
        for (int i = threadIdx.x; i < nt * 8 * q16; i += blockDim.x) {
            const int t = i / q16, q = i - t * q16;
            uint4 v = z4;
            if (t < nr) {
                const int src = xidx ? xidx[start + t] : start + t;
                v = *reinterpret_cast<const uint4 *>(X + (size_t)src * in + k0 + q * 8);
            }
            *reinterpret_cast<uint4 *>(hx + (size_t)t * STRIDE + q * 16) = v;
        }
        __syncthreads();
        for (int kk = 0; kk < kc; kk += 64) {
            const uint4 *p0 = reinterpret_cast<const uint4 *>(w0 + k0 + kk);
            const uint4 *p1 = reinterpret_cast<const uint4 *>(w1 + k0 + kk);
            const uint4 a0l = v0 ? __ldcs(p0) : z4, a0h = v0 ? __ldcs(p0 + 1) : z4;
            const uint4 a1l = v1 ? __ldcs(p1) : z4, a1h = v1 ? __ldcs(p1 + 1) : z4;
            /* [2s] = k 4s, 4s+1; [2s+1] = k 4s+2, 4s+3 of the lane's 16 */
            const uint32_t A0[8] = {a0l.x, a0l.y, a0l.z, a0l.w, a0h.x, a0h.y, a0h.z, a0h.w};
            const uint32_t A1[8] = {a1l.x, a1l.y, a1l.z, a1l.w, a1h.x, a1h.y, a1h.z, a1h.w};
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                if (j >= nt) break;
                const uint8_t *xr = hx + (size_t)(j * 8 + g) * STRIDE + (size_t)(kk + 16 * c) * 2;
                const uint4 q0 = *reinterpret_cast<const uint4 *>(xr);
                const uint4 q1 = *reinterpret_cast<const uint4 *>(xr + 16);
                const uint32_t B[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const uint32_t af[4] = {A0[2 * s], A1[2 * s], A0[2 * s + 1], A1[2 * s + 1]};
                    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    pulsar_mma_bf16_16816(d, af, B[2 * s], B[2 * s + 1]);
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[j][e] += d[e];
                }
            }
        }
    }
    /* acc[j][0..1] = row r0, call rows 8j + 2c, +1; acc[j][2..3] = row r1 */
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int t = 8 * j + 2 * c + (e & 1), row = e < 2 ? r0 : r1;
            if (t < nr && (e < 2 ? v0 : v1)) {
                const int dst = yidx ? yidx[start + t] : start + t;
                Y[(size_t)dst * out + row] = acc[j][e];
            }
        }
}

/* The dense Linear: one work item, the call's rows 0 .. nr - 1. */
/// Grid ceil(out / 128), 256 threads, stage_bytes<NT>() dynamic shared memory: Y [nr][out] f32 = X W^T.
/// @tparam NT n8 activation tiles (nr <= 8 NT).
/// @param W the weight, bf16 [out][in].
/// @param out output width.
/// @param in input width (a multiple of 64).
/// @param X the rows, bf16 [nr][in].
/// @param nr rows.
/// @param Y the output, f32 [nr][out].
template <int NT>
__global__ void __launch_bounds__(32 * kWarps)
bf16_rows_dense_kernel(const __nv_bfloat16 *__restrict__ W, int out, int in, const __nv_bfloat16 *__restrict__ X,
                       int nr, float *__restrict__ Y) {
    extern __shared__ __align__(16) uint8_t hx[];
    rows_tile<NT>(W, out, in, (int)blockIdx.x * kOutPerCta, X, nullptr, nullptr, 0, nr, Y, hx);
}

#ifdef PULSAR_HAVE_MMQ
/** One routed projection of a launch: its stack [n_expert][out][in] and its output [pairs][out]. */
struct routed_proj {
    const __nv_bfloat16 *stack;
    float *y;
};

/* The routed projections (gridDim.z of them, one shape): CTA (x, item, z) runs work item `item` of the plan over
 * outputs [128 x, +128) of projection z. */
__global__ void __launch_bounds__(32 * kWarps)
bf16_rows_routed_kernel(routed_proj p0, routed_proj p1, int out, int in, const __nv_bfloat16 *__restrict__ X,
                        const int32_t *__restrict__ ids_src, const int32_t *__restrict__ ids_dst,
                        const int4 *__restrict__ work, const int *__restrict__ n_work) {
    extern __shared__ __align__(16) uint8_t hx[];
    if ((int)blockIdx.y >= *n_work) return;
    const int4 it = work[blockIdx.y];   /* expert, start, count */
    const routed_proj &p = blockIdx.z == 0 ? p0 : p1;
    rows_tile<kRoutedNT>(p.stack + (size_t)it.x * out * in, out, in, (int)blockIdx.x * kOutPerCta, X, ids_src,
                         ids_dst, it.y, it.z, p.y, hx);
}

/* The work list: expert e's assignments [bounds[e], bounds[e+1]) in pieces of 8 * kRoutedNT, in expert order -- one
 * block of >= E threads, an inclusive scan over the per-expert piece counts.  `cap` is the list's length (the launch
 * sizes it to pairs / 64 + E + 1, which no routing exceeds). */
__global__ void bf16_moe_plan_kernel(const int32_t *__restrict__ bounds, int E, int4 *__restrict__ work,
                                     int *__restrict__ n_work, int cap) {
    constexpr int kItem = 8 * kRoutedNT;
    __shared__ int s[kMaxE];
    const int e = threadIdx.x;
    const int n = e < E ? bounds[e + 1] - bounds[e] : 0;
    const int mb = (n + kItem - 1) / kItem;
    s[e] = mb;
    __syncthreads();
    for (int off = 1; off < (int)blockDim.x; off <<= 1) {
        const int v = e >= off ? s[e - off] : 0;
        __syncthreads();
        s[e] += v;
        __syncthreads();
    }
    const int at = s[e] - mb;
    for (int j = 0; j < mb; ++j)
        if (at + j < cap) work[at + j] = make_int4(e, bounds[e] + j * kItem, min(kItem, n - j * kItem), 0);
    if (e == (int)blockDim.x - 1) *n_work = s[e] < cap ? s[e] : cap;
}

/* The fold: each scheduled pair's activation with its route weight folded in, rounded to the bf16 row the down
 * reads -- SWIGLU pulsar_swiglu_elem (the EXL3 fold's and the shared expert's expression), RELU2 pulsar_relu2_elem
 * (relu(u)^2 * w in f32, a NaN kept; the non-gated shared expert's expression too; g unread).  A pair with no expert (sel < 0) is not in the schedule: its row is not written and nothing reads it. */
template <int ACT>
__global__ void __launch_bounds__(256)
bf16_moe_fold_kernel(const float *__restrict__ g, const float *__restrict__ u, const int32_t *__restrict__ sel,
                     const float *__restrict__ wts, float clamp, int64_t pairs, int mid, __nv_bfloat16 *__restrict__ h) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs * mid) return;
    const int64_t p = i / mid;
    if (sel[p] < 0) return;
    if constexpr (ACT == PULSAR_FFN_ACT_RELU2) {
        /* pulsar_relu2_elem: F7 18f2278f's NaN-keeping relu^2 (x <= 0 ? 0 : x, then r * r * w), the one expression the
         * non-gated shared expert's producer reads too */
        h[i] = __float2bfloat16(pulsar_relu2_elem(u[i], wts[p]));
    } else {
        h[i] = __float2bfloat16(pulsar_swiglu_elem(g[i], u[i], wts[p], clamp));
    }
}

/* The sum: out[t][o] = the row's scheduled slots' down outputs, slots in order; a non-finite sum records nf_code. */
__global__ void __launch_bounds__(256)
bf16_moe_sum_kernel(const float *__restrict__ dz, const int32_t *__restrict__ sel, int T, int top_k, int out,
                    float *__restrict__ y, uint32_t *__restrict__ nf_flag, uint32_t nf_code) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)T * out) return;
    const int64_t t = i / out, o = i - t * out;
    float acc = 0.0f;
    for (int j = 0; j < top_k; ++j)
        if (sel[t * top_k + j] >= 0) acc += dz[(t * top_k + j) * out + o];
    y[i] = acc;
    if (!isfinite(acc)) atomicCAS(nf_flag, 0u, nf_code);
}
#endif

/// Checks the last launch, naming a failure on stderr.
/// @param what the launch's name.
/// @return true when no error is pending.
bool launch_ok(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: BF16 rows %s: launch failed: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

/// @param p a pointer.
/// @return true when p is 16-byte aligned.
bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

} // namespace

int pulsar_bf16_rows_linear_launch(const void *w, const uint16_t *x_bf16, int rows, int in, int out, float *y,
                                   bool prompt, cudaStream_t stream) {
    PULSAR_REQUIRE_CC(80, "the bf16 rows linear (bf16 m16n8k16 / cuBLASLt bf16)", -1);   /* L321: refuse below the lane's arch, before its kernels trap */
    if (!w || !x_bf16 || !y || rows <= 0 || in <= 0 || out <= 0 || in % 64 || !aligned16(w) || !aligned16(x_bf16) ||
        !aligned16(y)) {
        fprintf(stderr, "pulsar: BF16 linear (bf16 rows): %d -> %d at %d rows needs in %% 64 and 16-byte aligned "
                        "weight, rows and output -- refusing\n", in, out, rows);
        return -1;
    }
    if (prompt) {
        static bool announced = false;
        if (!__atomic_exchange_n(&announced, true, __ATOMIC_RELAXED))
            fprintf(stderr, "pulsar: BF16 linear (bf16 rows), prompt chunks = cuBLASLt, one fixed-order kernel per shape "
                            "(first: %d -> %d at %d rows)\n", in, out, rows);
        if (!pulsar_gpu_bf16_rows_gemm(y, (const uint16_t *)w, x_bf16, (uint64_t)in, (uint64_t)out, (uint64_t)rows,
                                       stream)) {
            fprintf(stderr, "pulsar: BF16 linear (bf16 rows): the GEMM %d -> %d at %d rows refused -- no fallback\n", in,
                    out, rows);
            return -1;
        }
        return 0;
    }
    static bool announced = false;
    if (!__atomic_exchange_n(&announced, true, __ATOMIC_RELAXED))
        fprintf(stderr, "pulsar: BF16 linear (bf16 rows), decode rows = the tensor-core rows kernel, %d-row blocks, "
                        "M-independent at every width (first: %d -> %d at %d rows)\n", PULSAR_BF16_ROWS_DECODE_MAX, in,
                out, rows);
    const dim3 grid((unsigned)((out + kOutPerCta - 1) / kOutPerCta));
    const __nv_bfloat16 *W = static_cast<const __nv_bfloat16 *>(w), *X = reinterpret_cast<const __nv_bfloat16 *>(x_bf16);
    /* decode rows past the kernel's 16 run 16-row blocks in turn: each block is the call it is alone (as the EXL3 and
     * mxfp8 decode arms do, L288 C1) */
    for (int r0 = 0; r0 < rows; r0 += PULSAR_BF16_ROWS_DECODE_MAX) {
        const int m = rows - r0 < PULSAR_BF16_ROWS_DECODE_MAX ? rows - r0 : PULSAR_BF16_ROWS_DECODE_MAX;
        if (m > 8)
            bf16_rows_dense_kernel<2><<<grid, 32 * kWarps, stage_bytes<2>(), stream>>>(W, out, in, X + (size_t)r0 * in, m,
                                                                                      y + (size_t)r0 * out);
        else
            bf16_rows_dense_kernel<1><<<grid, 32 * kWarps, stage_bytes<1>(), stream>>>(W, out, in, X + (size_t)r0 * in, m,
                                                                                      y + (size_t)r0 * out);
    }
    return launch_ok("dense") ? 0 : -3;
}

int pulsar_bf16_moe_routed_launch(const pulsar_bf16_moe &w, const int32_t *sel, const float *wts, int T, int top_k,
                                  float clamp, const uint16_t *x_bf16, float *gate_z, float *up_z, uint16_t *mid,
                                  float *down_z, float *out, uint32_t *nf_flag, uint32_t nf_code, cudaStream_t stream) {
    PULSAR_REQUIRE_CC(80, "the bf16 routed MoE (bf16 m16n8k16)", -1);   /* L321: refuse below the lane's arch, before its kernels trap */
    const char *tag = "pulsar: routed MoE (bf16 rows, BF16 experts)";
    /* the activation names the stacks: SWIGLU gate + up, RELU2 up alone (a non-gated expert: no gate stack) */
    const bool gated = w.act == PULSAR_FFN_ACT_SWIGLU;
    if ((w.act != PULSAR_FFN_ACT_SWIGLU && w.act != PULSAR_FFN_ACT_RELU2) || gated != (w.gate != nullptr) ||
        (gated && (!gate_z || !aligned16(w.gate))) || !w.up || !w.down || !sel || !wts || !x_bf16 || !up_z || !mid ||
        !down_z || !out || !nf_flag || T <= 0 || top_k <= 0 || w.n_experts <= 0 || w.n_experts > kMaxE ||
        w.in_dim <= 0 || w.mid_dim <= 0 || w.out_dim <= 0 || w.in_dim % 64 || w.mid_dim % 64 || !aligned16(w.up) ||
        !aligned16(w.down) || !aligned16(x_bf16) || !aligned16(mid)) {
        fprintf(stderr, "%s: the stacks the activation reads (act %d: SWIGLU gate + up, RELU2 up alone), a down, every "
                        "buffer, %d -> %d -> %d (in and mid %% 64), 1..%d experts and 16-byte alignment -- refusing\n",
                tag, (int)w.act, w.in_dim, w.mid_dim, w.out_dim, kMaxE);
        return -1;
    }
#ifndef PULSAR_HAVE_MMQ
    (void)clamp; (void)nf_code; (void)stream;
    fprintf(stderr, "%s: built without the MMQ drivers whose schedule it rides -- refusing\n", tag);
    return -1;
#else
    static int mmq_ready = -1;   /* the schedule's device-info singleton */
    if (mmq_ready < 0) {
        int dev = 0;
        (void)cudaGetDevice(&dev);
        mmq_ready = ds4_mmq_init(dev) == 0 ? 1 : 0;
    }
    if (!mmq_ready) {
        fprintf(stderr, "%s: the MMQ drivers are unavailable on this device -- refusing\n", tag);
        return -1;
    }
    const int64_t pairs = (int64_t)T * top_k;
    const int E = w.n_experts;
    const int cap = (int)(pairs / (8 * kRoutedNT)) + E + 1;
    if (pairs > INT32_MAX || cap > 65535) {
        fprintf(stderr, "%s: %lld assignments are past the work list's grid -- refusing\n", tag, (long long)pairs);
        return -1;
    }
    static bool announced[2] = {false, false};
    if (!__atomic_exchange_n(&announced[gated], true, __ATOMIC_RELAXED))
        fprintf(stderr, "%s = the tensor-core rows kernel over the expert-sorted schedule (%d assignments a work item), "
                        "%s, bf16 %s rows, down, slot-ordered sum (%d -> %d -> %d, top-%d of %d)\n", tag,
                8 * kRoutedNT, gated ? "gate + up in one launch" : "up alone", gated ? "SwiGLU" : "relu^2", w.in_dim,
                w.mid_dim, w.out_dim, top_k, E);
    const uint64_t ids_b = (uint64_t)pairs * sizeof(int32_t), eb_b = (uint64_t)(E + 1) * sizeof(int32_t);
    const uint64_t work_b = (uint64_t)cap * sizeof(int4);
    cuda_arena ar;
    if (!cuda_arena_begin_slot(&ar, CUDA_SCRATCH_MMQ, 2 * ids_b + eb_b + work_b + 4 * 256 + 256, tag)) return -1;
    int32_t *ids_src = (int32_t *)cuda_arena_take(&ar, ids_b, 256);
    int32_t *ids_dst = (int32_t *)cuda_arena_take(&ar, ids_b, 256);
    int32_t *bounds = (int32_t *)cuda_arena_take(&ar, eb_b, 256);
    int4 *work = (int4 *)cuda_arena_take(&ar, work_b, 256);
    int *n_work = (int *)cuda_arena_take(&ar, sizeof(int), 256);
    if (!n_work) return -1;   /* take() latches */
    ds4_launch_mm_ids_helper(sel, ids_src, ids_dst, bounds, E, T, top_k, /*nchannels_y=*/1, /*si1=*/top_k, /*sis1=*/1,
                             stream);
    if (!launch_ok("schedule")) return -3;
    bf16_moe_plan_kernel<<<1, (E + 31) / 32 * 32, 0, stream>>>(bounds, E, work, n_work, cap);
    if (!launch_ok("plan")) return -3;
    const __nv_bfloat16 *X = reinterpret_cast<const __nv_bfloat16 *>(x_bf16);
    const int H = w.in_dim, M = w.mid_dim, O = w.out_dim;
    const size_t smem = stage_bytes<kRoutedNT>();
    /* up (and gate): the source row is the token, the destination the pair; projection z 0 is up, z 1 the gate */
    const routed_proj pu{static_cast<const __nv_bfloat16 *>(w.up), up_z};
    const routed_proj pg = gated ? routed_proj{static_cast<const __nv_bfloat16 *>(w.gate), gate_z} : routed_proj{nullptr, nullptr};
    bf16_rows_routed_kernel<<<dim3((unsigned)((M + kOutPerCta - 1) / kOutPerCta), (unsigned)cap, gated ? 2u : 1u),
                              32 * kWarps, smem, stream>>>(pu, pg, M, H, X, ids_src, ids_dst, work, n_work);
    if (!launch_ok("gate / up")) return -3;
    const int64_t nm = pairs * M;
    __nv_bfloat16 *hm = reinterpret_cast<__nv_bfloat16 *>(mid);
    if (gated)
        bf16_moe_fold_kernel<PULSAR_FFN_ACT_SWIGLU><<<(unsigned)((nm + 255) / 256), 256, 0, stream>>>(
            gate_z, up_z, sel, wts, clamp, pairs, M, hm);
    else
        bf16_moe_fold_kernel<PULSAR_FFN_ACT_RELU2><<<(unsigned)((nm + 255) / 256), 256, 0, stream>>>(
            nullptr, up_z, sel, wts, clamp, pairs, M, hm);
    if (!launch_ok("fold")) return -3;
    /* the down: its source row is the pair (the fold wrote one row per pair) */
    bf16_rows_routed_kernel<<<dim3((unsigned)((O + kOutPerCta - 1) / kOutPerCta), (unsigned)cap, 1), 32 * kWarps, smem,
                              stream>>>(routed_proj{static_cast<const __nv_bfloat16 *>(w.down), down_z},
                                        routed_proj{nullptr, nullptr}, O, M, reinterpret_cast<const __nv_bfloat16 *>(mid),
                                        ids_dst, ids_dst, work, n_work);
    if (!launch_ok("down")) return -3;
    const int64_t no = (int64_t)T * O;
    bf16_moe_sum_kernel<<<(unsigned)((no + 255) / 256), 256, 0, stream>>>(down_z, sel, T, top_k, O, out, nf_flag,
                                                                         nf_code);
    return launch_ok("sum") ? 0 : -3;
#endif
}
