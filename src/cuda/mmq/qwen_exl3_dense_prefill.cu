// SPDX-License-Identifier: MIT
/** @file
 *  The dense EXL3 arm for prompt chunks: one arithmetic, as a fixed-algorithm cuBLASLt GEMM on the weight decoded to
 *  fp16, or (small chunks, L381) a kernel that reads the trellis in place and writes the same bytes. */
// The dense EXL3 arm for PROMPT CHUNKS (L251): every prompt chunk of ds4_exl3_dense_launch (decode rows keep the GEMV).
//
// The decode arm (ds4_exl3_dense.cu) is a GEMV: every CTA decodes its trellis tiles for 16 rows and
// reads the activation planes for 32 outputs, so a 4096-row prompt re-decodes each weight 256 times
// and re-reads each activation N/32 times.  This is the GEMM, TensorFold's shape for the same format
// (tensorfold/cuda/exl3/prefill.py: "W_q decoded once a chunk into a ... fp16 GEMM"):
//
//   1. DECODE  the trellis ONCE into an fp16 K x N matrix W_q (row-major), with the GEMV's own tile
//              decode (mul1_pair: the same fp16 values, bit for bit).
//   2. PREP    per row: x * suh, H128 per 128-block, then ONE power-of-two prescale for the whole row
//              (so a GEMM can sum across blocks), rounded to one fp16 plane: 11 significant bits,
//              8x finer than the bf16 row it came from (exllamav3's and TensorFold's operand; a second
//              "lo" plane doubled the GEMM for precision the input does not carry).
//   3. GEMM    one cuBLASLt tensor-core GEMM, Z = X W_q, f32 accumulate and output -- a row's bytes
//              independent of how many rows share the call (L266): the fixed-algorithm GEMM
//              (pulsar_cuda_lt_fixed.h, L321: the contract now shared with the f16 rows arm) -- the
//              reduction scheme pinned to NONE and ONE algorithm per (K, N), chosen at the production
//              prefill chunk's row count and checked to support each other row count.
//              cublasGemmEx(CUBLAS_GEMM_DEFAULT) sat here and picked a kernel (and split-K) by M, so a
//              prompt cut at G got different bytes from the same prompt prefilled whole --
//              the chunk-neutrality gate (now tests/session_contract_gate) measured 1-2 nats of logit difference.  DeepSeek's bf16
//              GEMM met the same fact in L183 (pulsar_cuda_matmul.cu bf16_lt_matmul).
//   4. EPILOGUE per row: times 2^-P, the output H128 per 128-block, times svh.
//
// A prefilled row and a decoded row agree to rounding, not to the bit (Tyler 2026-09-29, "Don't use
// a decode kernel for prefill"); graded against double by tests/exl3_dense_gate at every rate.
// The scratch (W_q, the planes, the row scales) is one device buffer grown to the largest call.
//
// THE TWO KERNELS OF THE ARM (L381).  Steps 1 and 3 cost the same at 5 rows as at 4096 (the whole weight to fp16,
// then a GEMM over it): two thirds of a short turn's prefill chunk.  A launch of few rows runs instead
//
//   PREP              step 2, the same kernel: the same plane and row scales
//   IN PLACE          dense_inplace_kernel: steps 1, 3 and 4 in one CTA per (128 stored outputs, 64 rows).  A warp
//                     owns a 16-output tile: it decodes its trellis words in registers, a 128-k block at a time, to
//                     the fp16 values step 1 writes (the same mul1_pair on the same states) as mma B fragments,
//                     and applies them to the plane's rows (staged by cp.async a block ahead) in ONE ascending chain
//                     of m16n8k16 accumulates an output, k-tile 0 .. K/16 - 1 into one f32 accumulator -- the order
//                     the fixed cuBLASLt GEMM follows on the devices the rule names (tests/lt_order_probe) -- then
//                     step 4's f32 operations on the block (times 2^-P, H128, svh) and one store of y.
//
// with the GEMM arm's bytes, so a prompt row's bytes still do not depend on the chunk it rides in (reuse == cold, the
// prefix-cache gates) and no prefill byte moved when it landed.  Which kernel a launch takes:
//
//   the rule   ds4_exl3_dense_prompt_arm (ds4_exl3_dense.cuh): in place iff the device is sm_12x and M <= 256 rows,
//              with the measurement beside it.  Everything else is the GEMM arm, the code above, untouched.
//   the check  the equality is the VENDOR kernel's property -- a (compute capability, cuBLASLt version, shape) --
//              so it is never assumed: the FIRST launch the rule gives the in-place kernel for a (device, K, N,
//              rate) in a process runs BOTH kernels on that launch's own operands (the in-place one into scratch,
//              the GEMM arm into y), waits, and compares every output byte.  Equal: the shape is recorded and its
//              later launches run in place.  Not equal: ONE loud line (the shape, the device, the cuBLASLt version,
//              the first differing output) and the in-place kernel is off for EVERY shape for the life of the
//              process; that launch and all later ones return the GEMM arm's bytes.  So a cuBLASLt or driver change
//              on a serving host cannot split prompt bytes between small and large chunks: the worst case is
//              today's speed.  A multi launch checks each projection as its own shape.  The cost: one GEMM-arm
//              launch, one stream wait and two copies of M x N f32 to the host, once a shape a process.
//              The checked shapes are a small lock-free table (a CAS a slot; the launches of one process come from
//              a few long-lived GPU threads).  A launch under a stream capture is not checked and takes the GEMM
//              (no prompt step is captured today: step_driver.cpp graphs pure decode and verify steps only).
//
// tests/exl3_dense_gate section 10 holds the equality byte for byte at every rate and width with each kernel forced
// (ds4_exl3_dense_prompt_force), the rule's table, and the check's mismatch path (ds4_exl3_dense_prompt_check_fault).

#include "ds4_exl3_dense.cuh"
#include "ds4_exl3_dev.cuh"
#include "engine/exl3_trellis.h"

#include <cstdio>
#include <cstring>
#include <cublasLt.h>   /* cublasLtGetVersion: the check's announce lines */
#include "cuda/pulsar_cuda_lt_fixed.h"
#include "cuda/pulsar_cuda_arch.cuh"   /* pulsar_cuda_smem_opt_in */
#include "pulsar_gpu.h"   /* pulsar_gpu_seg_grow */
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kHiExp = 14;   ///< the row's largest |value| lands just under 2^14 (the GEMV's prescale)

/// 2^e as an f32 built from its exponent bits.
/// @param e  the exponent, in [-126, 127]
/// @return   2^e
__device__ __forceinline__ float pow2f(int e) { return __int_as_float((127 + e) << 23); }

/* 1. one warp per 16 x 16 tile (kt, nt): lane t holds positions 8t..8t+7 -- k rows (t%4)*2 + {0,1,8,9}
 * of columns t/4 (positions 8t..8t+3) and t/4 + 8 (8t+4..8t+7), the mma B-fragment order. */
/// DECODE at rate K2: the trellis into fp16 W_q.  Grid ceil(K/16 x N/16 / 8), 256 threads: one warp per 16 x 16
/// tile, tiles in the trellis's (kt, nt) order.
/// @param trellis  the weight's trellis words, device
/// @param wq       out: W_q fp16 [K][N], row-major
/// @param K        input width (a multiple of 128)
/// @param N        stored output width (a multiple of 128)
template <int K2>
__global__ void __launch_bounds__(256)
dense_decode_kernel(const uint32_t *__restrict__ trellis, __half *__restrict__ wq, int K, int N) {
    constexpr int W32 = exl3dev::Rate<K2>::words32;
    const int64_t tile = (int64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int ntn = N >> 4;
    if (tile >= (int64_t)(K >> 4) * ntn) return;
    const int kt = (int)(tile / ntn), nt = (int)(tile % ntn);
    const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const uint32_t *t = trellis + (size_t)tile * W32;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        int lo, hi, sh;
        exl3dev::run4_window<K2>(8 * lane + 4 * h, lo, hi, sh);
        uint32_t st[4];
        exl3dev::run4_split<K2>(EXL3_RUN4_JOIN(t[lo], t[hi], sh), 8 * lane + 4 * h, st);
        const uint32_t b0 = exl3dev::mul1_pair(st[0], st[1]);   /* k rows 2c, 2c+1 */
        const uint32_t b1 = exl3dev::mul1_pair(st[2], st[3]);   /* k rows 2c+8, 2c+9 */
        const int col = nt * 16 + g + 8 * h, k0 = kt * 16 + 2 * c;
        const __half2 p0 = *reinterpret_cast<const __half2 *>(&b0), p1 = *reinterpret_cast<const __half2 *>(&b1);
        wq[(size_t)(k0 + 0) * N + col] = __low2half(p0);
        wq[(size_t)(k0 + 1) * N + col] = __high2half(p0);
        wq[(size_t)(k0 + 8) * N + col] = __low2half(p1);
        wq[(size_t)(k0 + 9) * N + col] = __high2half(p1);
    }
}

/* 2. one CTA (8 warps) per row: the rotated row into shared memory block by block, the row amax,
 * then the fp16 plane and the row's inverse prescale. */
/// PREP: x * suh, H128 per block, one power-of-two prescale per row, one fp16 plane.  Grid M, 256 threads,
/// K * 4 bytes of dynamic smem (the rotated row; the launcher opts in past 48 KiB).
/// @param x    bf16 input rows [M][K] (8-byte aligned rows)
/// @param suh  the input sign/scale vector, fp16 [K]
/// @param xhi  out: the fp16 operand plane [M][K]
/// @param inv  out: each row's inverse prescale 2^-P, f32 [M]
/// @param K    input width (a multiple of 128)
__global__ void __launch_bounds__(256)
dense_prep_kernel(const __nv_bfloat16 *__restrict__ x, const __half *__restrict__ suh,
                  __half *__restrict__ xhi, float *__restrict__ inv, int K) {
    extern __shared__ float s_row[];              /* [K] */
    __shared__ float s_amax[8];
    const int m = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float amax = 0.0f;
    for (int blk = warp; blk < (K >> 7); blk += 8) {
        const int k = blk * 128 + lane * 4;
        const uint2 xw = *reinterpret_cast<const uint2 *>(x + (size_t)m * K + k);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw.y));
        const uint2 sw = *reinterpret_cast<const uint2 *>(suh + k);
        const __half2 s01 = *reinterpret_cast<const __half2 *>(&sw.x), s23 = *reinterpret_cast<const __half2 *>(&sw.y);
        float v[4] = {a.x * __low2float(s01), a.y * __high2float(s01), b.x * __low2float(s23), b.y * __high2float(s23)};
        exl3dev::had128(v);
#pragma unroll
        for (int i = 0; i < 4; ++i) { s_row[k + i] = v[i]; amax = fmaxf(amax, fabsf(v[i])); }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) s_amax[warp] = amax;
    __syncthreads();
    float rmax = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) rmax = fmaxf(rmax, s_amax[w]);
    int p = rmax > 0.0f ? kHiExp - (((__float_as_int(rmax) >> 23) & 0xff) - 126) : 0;
    p = max(-100, min(100, p));
    const float up = pow2f(p);
    for (int k = threadIdx.x * 2; k < K; k += 512) {
        const float a0 = s_row[k] * up, a1 = s_row[k + 1] * up;
        *reinterpret_cast<__half2 *>(xhi + (size_t)m * K + k) = __floats2half2_rn(a0, a1);
    }
    if (threadIdx.x == 0) inv[m] = pow2f(-p);
}

/* 4. one warp per (row, 128-block of outputs), in place on y (row width N, the logical outputs).  The blocks past
 * n_main -- one, the stored width's last when N is not a 128 multiple (exl3_out_stored, L295 F7) -- come from `tail`
 * [M][128] instead, and only their columns below N are written. */
/// EPILOGUE: times 2^-P, the output H128 and svh, in place on y.  Grid ceil(M x stored N / 128 / 8), 256 threads.
/// @param y       f32 [M][N], the GEMM's columns below n_main; rewritten finished
/// @param tail    the last stored 128-block's GEMM output f32 [M][128] when N is not a 128 multiple, else NULL
/// @param inv     each row's inverse prescale [M]
/// @param svh     the output sign/scale vector, fp16 [stored N]
/// @param M       rows
/// @param N       logical output width (y's row)
/// @param n_main  N rounded down to 128: the columns read from y
__global__ void __launch_bounds__(256)
dense_epilogue_kernel(float *__restrict__ y, const float *__restrict__ tail, const float *__restrict__ inv,
                      const __half *__restrict__ svh, int M, int N, int n_main) {
    const int64_t task = (int64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int n_blk = (int)(exl3_out_stored((uint64_t)N) >> 7);
    if (task >= (int64_t)M * n_blk) return;
    const int m = (int)(task / n_blk), blk = (int)(task % n_blk), lane = threadIdx.x & 31, n0 = blk * 128 + lane * 4;
    float4 *p = reinterpret_cast<float4 *>(y + (size_t)m * N + n0);
    const float4 q = n0 < n_main ? *p : *reinterpret_cast<const float4 *>(tail + (size_t)m * 128 + (n0 - n_main));
    const float s = inv[m];
    float z[4] = {q.x * s, q.y * s, q.z * s, q.w * s};
    exl3dev::had128(z);
    const __half2 v01 = *reinterpret_cast<const __half2 *>(svh + n0), v23 = *reinterpret_cast<const __half2 *>(svh + n0 + 2);
    if (n0 < N)
        *p = make_float4(z[0] * __low2float(v01), z[1] * __high2float(v01), z[2] * __low2float(v23), z[3] * __high2float(v23));
}

/* ---- the in-place kernel (L381) -------------------------------------------------------------------------------- */

constexpr int kInRows    = 64;        ///< rows a CTA of the in-place kernel: four m16 tiles
constexpr int kInApad    = 128 + 8;   ///< fp16 per staged plane row: 16-byte rows, conflict-free ldmatrix
constexpr int kInThreads = 256;       ///< its 8 warps: the eight 16-output tiles of a 128-output block

/// The in-place kernel's dynamic shared memory: the plane's current and next 128-k block of the CTA's rows, then
/// (the k loop done) the block's 64 x 128 f32 accumulators for the epilogue.
struct inplace_smem {
    __half a[2][kInRows][kInApad];   ///< [buffer][row][k of the block]
};
static_assert(sizeof(inplace_smem) >= (size_t)kInRows * 128 * sizeof(float), "the epilogue's block reuses the A buffers");

/// Four 8x8 b16 matrices from shared memory into an MMA A fragment (ldmatrix .x4).
/// @param a  out: the fragment
/// @param p  the lane's row of the staged tile (16-byte aligned)
__device__ __forceinline__ void inplace_ldmatrix(uint32_t (&a)[4], const __half *p) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}

/// Block blk of the plane's rows [a0, a0 + mr) into buffer buf by cp.async, rows mr .. arows - 1 zero-filled (the
/// last m16 tile's rows past the chunk: zero operands, and their accumulators are never read).
/// @param S      the CTA's shared memory
/// @param buf    the buffer, 0 or 1
/// @param xhi    the plane, fp16 [M][K]
/// @param a0     the CTA's first row
/// @param mr     its real rows, 1..kInRows
/// @param arows  its staged rows: mr rounded up to 16
/// @param K      input width
/// @param blk    the 128-k block
/// @param tid    the thread
__device__ __forceinline__ void inplace_stage(inplace_smem &S, int buf, const __half *__restrict__ xhi, int a0, int mr,
                                              int arows, int K, int blk, int tid) {
    for (int c = tid; c < arows * 16; c += kInThreads) {
        const int r = c >> 4, ch = c & 15;
        const bool real = r < mr;
        const size_t off = (size_t)(a0 + (real ? r : 0)) * K + (size_t)blk * 128 + ch * 8;
        pulsar_cp_async16(&S.a[buf][r][ch * 8], xhi + off, real ? 16 : 0);
    }
    pulsar_cp_async_commit();
}

/// IN PLACE at rate K2 (the file's header): steps 1, 3 and 4 of the arm for a chunk of few rows, the GEMM arm's
/// bytes.  Grid (stored N / 128, ceil(M / kInRows)), kInThreads threads, sizeof(inplace_smem) dynamic smem; two CTAs
/// an SM.  CTA (x, y) = stored outputs [128 x, 128 x + 128) of rows [64 y, 64 y + 64); warp w = its 16-output tile w.
/// A lane holds positions 8 l .. 8 l + 7 of every tile -- the mma B fragment (dense_decode_kernel's lane order) --
/// so its two runs of a tile decode straight into the two B registers of each n8 half; the words of a 128-k block's
/// eight tiles are loaded a block ahead, the plane's rows staged a block ahead.  acc[m][h][i] is row 16 m + g + 8
/// (i >> 1), output 16 w + 8 h + 2 c + (i & 1): ONE accumulator an output, the k-tiles in ascending order.
/// @param tr    the weight's trellis words, device
/// @param xhi   dense_prep_kernel's plane, fp16 [M][K]
/// @param inv   its inverse prescales 2^-P, f32 [M]
/// @param svh   the output sign/scale vector, fp16 [stored N]
/// @param y     out: f32 [M][Ny], finished
/// @param M     rows
/// @param K     input width (a multiple of 128)
/// @param N     stored output width (a multiple of 128)
/// @param Ny    logical output width (y's row; <= N, a multiple of 4): a stored pad column is never written
template <int K2>
__global__ void __launch_bounds__(kInThreads, 2)
dense_inplace_kernel(const uint32_t *__restrict__ tr, const __half *__restrict__ xhi, const float *__restrict__ inv,
                     const __half *__restrict__ svh, float *__restrict__ y, int M, int K, int N, int Ny) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    inplace_smem &S = *reinterpret_cast<inplace_smem *>(smem_raw);
    const int a0 = (int)blockIdx.y * kInRows, mr = min(kInRows, M - a0), mt_n = (mr + 15) >> 4, arows = mt_n * 16;
    const int n0 = (int)blockIdx.x * 128, nblk = K >> 7;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int W32 = exl3dev::Rate<K2>::words32;
    const int ntn = N >> 4, nt = (n0 >> 4) + warp;
    int lo[2], hi[2], sh[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) exl3dev::run4_window<K2>(8 * lane + 4 * h, lo[h], hi[h], sh[h]);
    const int g = lane >> 2, c2 = (lane & 3) * 2, arow = lane & 15, ahalf = lane >> 4;
    float acc[4][2][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[m][h][i] = 0.0f;

    /* the lane's two windows (two words each) of the block's eight tiles of this warp's column */
    auto load_words = [&](int blk, uint32_t (&wd)[8][4]) {
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            const uint32_t *t = tr + ((size_t)(blk * 8 + kt) * (size_t)ntn + (size_t)nt) * W32;
            wd[kt][0] = t[lo[0]]; wd[kt][1] = t[hi[0]];
            wd[kt][2] = t[lo[1]]; wd[kt][3] = t[hi[1]];
        }
    };
    uint32_t wcur[8][4], wnxt[8][4];
    inplace_stage(S, 0, xhi, a0, mr, arows, K, 0, tid);
    load_words(0, wcur);
    for (int blk = 0; blk < nblk; ++blk) {
        const int buf = blk & 1;
        const bool more = blk + 1 < nblk;
        if (more) {
            inplace_stage(S, buf ^ 1, xhi, a0, mr, arows, K, blk + 1, tid);
            load_words(blk + 1, wnxt);
            pulsar_cp_async_wait<1>();
        } else {
            pulsar_cp_async_wait<0>();
        }
        __syncthreads();
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            uint2 bw[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t st[4];
                exl3dev::run4_split<K2>(EXL3_RUN4_JOIN(wcur[kt][2 * h], wcur[kt][2 * h + 1], sh[h]), 8 * lane + 4 * h, st);
                bw[h] = make_uint2(exl3dev::mul1_pair(st[0], st[1]), exl3dev::mul1_pair(st[2], st[3]));
            }
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (m < mt_n) {                        /* CTA-uniform */
                    uint32_t ah[4];
                    inplace_ldmatrix(ah, &S.a[buf][m * 16 + arow][kt * 16 + ahalf * 8]);
#pragma unroll
                    for (int h = 0; h < 2; ++h) pulsar_mma_f16_16816(acc[m][h], ah, bw[h].x, bw[h].y);   /* the chain */
                }
            }
        }
        __syncthreads();                               /* buf is refilled two blocks on */
        if (more) {
#pragma unroll
            for (int kt = 0; kt < 8; ++kt)
#pragma unroll
                for (int i = 0; i < 4; ++i) wcur[kt][i] = wnxt[kt][i];
        }
    }
    /* 4. the epilogue, in the CTA: the block's accumulators to shared memory, then dense_epilogue_kernel's f32
     * operations a (row, lane) -- times 2^-P, H128 over the whole stored block, times svh -- and one store */
    float *zb = reinterpret_cast<float *>(smem_raw);
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = m * 16 + g + 8 * (i >> 1);
                if (r < mr) zb[r * 128 + 16 * warp + 8 * h + c2 + (i & 1)] = acc[m][h][i];
            }
    __syncthreads();
    const int nq = n0 + lane * 4;
    for (int r = warp; r < mr; r += 8) {               /* whole warps: H128 needs the block's 32 lanes */
        const float4 q = *reinterpret_cast<const float4 *>(zb + r * 128 + lane * 4);
        const float s = inv[a0 + r];
        float z[4] = {q.x * s, q.y * s, q.z * s, q.w * s};
        exl3dev::had128(z);
        const __half2 v01 = *reinterpret_cast<const __half2 *>(svh + nq), v23 = *reinterpret_cast<const __half2 *>(svh + nq + 2);
        if (nq < Ny)
            *reinterpret_cast<float4 *>(y + (size_t)(a0 + r) * (size_t)Ny + nq) =
                make_float4(z[0] * __low2float(v01), z[1] * __high2float(v01), z[2] * __low2float(v23), z[3] * __high2float(v23));
    }
}

/* The arm's device scratch (W_q, the plane, the row scales, the tail), grown to the largest call; never freed.  ONE
 * PER (DEVICE, STREAM).  Per device since L335 (it was one process-wide buffer -- a TP rank on device 1 was handed
 * device 0's memory); per stream since L375 (O60): launches on two streams of one device run concurrently -- a
 * riding shared expert's on its lane's stream (exl3_moe_shared_fork) beside the block's on the caller's -- and one
 * buffer would have the second launch's decode overwrite the W_q the first one's GEMM is still reading.  A grow
 * follows the one rule (pulsar_gpu_seg_grow): an open capture is broken off first (a free / malloc cannot be
 * recorded), and a regrow drops the captured graphs (one that baked the old pointer would replay into freed memory). */
/// Streams of one device that may hold a scratch (the caller's and its lanes'; past it a launch refuses).
constexpr int kScratchStreams = 8;
/// One (device, stream)'s scratch.
struct stream_scratch {
    cudaStream_t stream;   ///< the launch stream it serves (0: the per-thread default, as the caller names it)
    uint8_t *base;         ///< the scratch, or nullptr before its first call
    size_t bytes;          ///< its bytes
};
/// Per device: its streams' scratches, [0, g_scratch_n[dev]) in use.
stream_scratch g_scratch[PULSAR_CUDA_DEVICE_SLOTS][kScratchStreams] = {};
/// Per device: entries of g_scratch in use.
int g_scratch_n[PULSAR_CUDA_DEVICE_SLOTS] = {};

/// @param dev     the device's slot (pulsar_cuda_device_slot)
/// @param stream  the launch stream
/// @return        the (device, stream)'s scratch entry, made on first use; nullptr when the device's
///                kScratchStreams are taken by other streams
stream_scratch *scratch_for(int dev, cudaStream_t stream) {
    for (int i = 0; i < g_scratch_n[dev]; ++i)
        if (g_scratch[dev][i].stream == stream) return &g_scratch[dev][i];
    if (g_scratch_n[dev] == kScratchStreams) return nullptr;
    stream_scratch *e = &g_scratch[dev][g_scratch_n[dev]++];
    *e = stream_scratch{stream, nullptr, 0};
    return e;
}

/// @param b  bytes
/// @return   b rounded up to a multiple of 256
size_t up256(size_t b) { return (b + 255u) & ~(size_t)255u; }

/// The (device, stream)'s scratch grown to `need` bytes (the one rule for a grow: the comment above).
/// @param tag     the caller's name for a refusal
/// @param stream  the launch stream
/// @param need    bytes
/// @return        the scratch, or nullptr (refused, said)
uint8_t *scratch_take(const char *tag, cudaStream_t stream, size_t need) {
    const int dev = pulsar_cuda_device_slot();
    if (dev < 0) return nullptr;
    stream_scratch *const sc = scratch_for(dev, stream);
    if (!sc) {
        fprintf(stderr, "%s: %d streams of this device already hold the arm's scratch; a ninth has none -- refusing\n",
                tag, kScratchStreams);
        return nullptr;
    }
    if (need > sc->bytes) {
        if (!pulsar_gpu_seg_grow("the qwen exl3 dense prefill scratch", sc->base != nullptr)) {
            fprintf(stderr, "%s: the scratch must grow to %zu bytes and the open capture could not be broken off -- "
                            "refusing\n", tag, need);
            return nullptr;
        }
        if (sc->base) (void)cudaFree(sc->base);
        sc->base = nullptr;
        sc->bytes = 0;
        if (cudaMalloc(&sc->base, need) != cudaSuccess) {
            (void)cudaGetLastError();
            sc->base = nullptr;
            fprintf(stderr, "%s: scratch of %zu bytes refused -- refusing\n", tag, need);
            return nullptr;
        }
        sc->bytes = need;
    }
    return sc->base;
}

/// Launch step 2, the PREP, for M rows (both kernels of the arm take its plane and row scales).
/// @param tag     the caller's name for a refusal
/// @param x       bf16 input rows [M][K]
/// @param suh     the input sign/scale vector, fp16 [K]
/// @param xhi     out: the fp16 operand plane [M][K]
/// @param inv     out: each row's inverse prescale [M]
/// @param M       rows
/// @param K       input width
/// @param stream  the launch stream
/// @return        true; false when the device refuses the kernel's shared memory (said)
bool prep_launch(const char *tag, const void *x, const __half *suh, __half *xhi, float *inv, int M, int K,
                 cudaStream_t stream) {
    /* the prep holds one row's K f32 in shared memory: past the 48 KiB a launch gets by default (K > 12288 -- L295:
     * Inkling's dense down, K 16384) the kernel opts in once to the larger carve-out.  The arithmetic is unchanged. */
    const size_t prep_smem = (size_t)K * sizeof(float);
    static int prep_smem_set[PULSAR_CUDA_DEVICE_SLOTS] = {};   /* per device (L335: process-wide before) */
    if (prep_smem > 48u * 1024u && !pulsar_cuda_smem_opt_in((const void *)dense_prep_kernel, prep_smem_set, prep_smem)) {
        fprintf(stderr, "%s: the prep's %zu B of shared memory (K=%d) is refused by the device -- refusing\n", tag,
                prep_smem, K);
        return false;
    }
    dense_prep_kernel<<<M, 256, prep_smem, stream>>>(static_cast<const __nv_bfloat16 *>(x), suh, xhi, inv, K);
    return true;
}

/// The GEMM arm's scratch bytes for a launch: W_q, the plane, the row scales, the tail.
/// @param M   rows
/// @param K   input width
/// @param Ny  logical output width
/// @return    bytes
size_t gemm_bytes(int M, int K, int Ny) {
    const int N = (int)exl3_out_stored((uint64_t)Ny);
    return up256((size_t)K * N * sizeof(__half)) + up256((size_t)M * K * sizeof(__half)) + up256((size_t)M * sizeof(float)) +
           (Ny == N ? 0 : up256((size_t)M * 128 * sizeof(float)));
}

/// The GEMM arm (the file's steps 1-4): the launch of every prompt chunk the rule does not give the in-place kernel.
/// @param tag            the caller's name for a refusal
/// @param w              the [trellis | suh | svh] slice
/// @param k2             rate
/// @param x_bf16         bf16 input rows [M][K]
/// @param y              out: f32 [M][Ny]
/// @param M              rows
/// @param K              input width
/// @param Ny             logical output width
/// @param trellis_bytes  bytes of the trellis plane
/// @param stream         the launch stream
/// @return               0, -1 refused, -3 launch failure
int gemm_arm(const char *tag, const void *w, int k2, const void *x_bf16, float *y, int M, int K, int Ny,
             uint64_t trellis_bytes, cudaStream_t stream) {
    /* N: the stored width the trellis and svh carry; Ny: the logical one, y's row (L295 F7) */
    const int N = (int)exl3_out_stored((uint64_t)Ny), n_main = Ny / 128 * 128;
    const size_t wq_b = up256((size_t)K * N * sizeof(__half)), pl_b = up256((size_t)M * K * sizeof(__half));
    uint8_t *const scr = scratch_take(tag, stream, gemm_bytes(M, K, Ny));
    if (!scr) return -1;
    __half *wq = reinterpret_cast<__half *>(scr);
    __half *xhi = reinterpret_cast<__half *>(scr + wq_b);
    float *inv = reinterpret_cast<float *>(scr + wq_b + pl_b);
    float *tail = Ny != N ? reinterpret_cast<float *>(scr + wq_b + pl_b + up256((size_t)M * sizeof(float))) : nullptr;
    const uint32_t *tr = static_cast<const uint32_t *>(w);
    const __half *suh = reinterpret_cast<const __half *>(static_cast<const uint8_t *>(w) + trellis_bytes);
    const __half *svh = suh + K;

    const int64_t tiles = (int64_t)(K >> 4) * (N >> 4);
    const unsigned dgrid = (unsigned)((tiles + 7) / 8);
    switch (k2) {
    case 4:  dense_decode_kernel<4><<<dgrid, 256, 0, stream>>>(tr, wq, K, N); break;
    case 6:  dense_decode_kernel<6><<<dgrid, 256, 0, stream>>>(tr, wq, K, N); break;
    case 8:  dense_decode_kernel<8><<<dgrid, 256, 0, stream>>>(tr, wq, K, N); break;
    case 10: dense_decode_kernel<10><<<dgrid, 256, 0, stream>>>(tr, wq, K, N); break;
    case 12: dense_decode_kernel<12><<<dgrid, 256, 0, stream>>>(tr, wq, K, N); break;
    case 16: dense_decode_kernel<16><<<dgrid, 256, 0, stream>>>(tr, wq, K, N); break;
    default:
        fprintf(stderr, "%s: rate k2=%d has no instance (K = 2, 3, 4, 5, 6, 8) -- refusing\n", tag, k2);
        return -1;
    }
    if (!prep_launch(tag, x_bf16, suh, xhi, inv, M, K, stream)) return -1;
    /* the whole GEMM, or (a logical width short of the stored one) its 128-multiple head into y and its last stored
     * block into the tail */
    if (n_main) {
        const int rc = pulsar_lt_fixed_gemm(PULSAR_LT_A_KN, PULSAR_LT_PROMPT, wq, K, N, 0, n_main, xhi, M, y, Ny, stream,
                                            tag);
        if (rc) return rc;
    }
    if (Ny != N) {
        const int rc = pulsar_lt_fixed_gemm(PULSAR_LT_A_KN, PULSAR_LT_PROMPT, wq, K, N, n_main, 128, xhi, M, tail, 128,
                                            stream, tag);
        if (rc) return rc;
    }
    const int64_t etasks = (int64_t)M * (N >> 7);
    dense_epilogue_kernel<<<(unsigned)((etasks + 7) / 8), 256, 0, stream>>>(y, tail, inv, svh, M, Ny, n_main);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

/* ---- the in-place kernel's launch, the rule's state and the check (L381) ---------------------------------------- */

/// The in-place kernel's signature.
typedef void (*inplace_fn)(const uint32_t *, const __half *, const float *, const __half *, float *, int, int, int, int);

/// The in-place kernel's instance at rate k2: the one authority (the launch and the description ask it).
/// @param k2  rate (2 x bits per weight)
/// @return    the kernel, or nullptr for a rate without an instance
inplace_fn inplace_instance(int k2) {
    return k2 == 4  ? dense_inplace_kernel<4>
         : k2 == 6  ? dense_inplace_kernel<6>
         : k2 == 8  ? dense_inplace_kernel<8>
         : k2 == 10 ? dense_inplace_kernel<10>
         : k2 == 12 ? dense_inplace_kernel<12>
         : k2 == 16 ? dense_inplace_kernel<16>
                    : nullptr;
}

/// The in-place kernel's scratch bytes for a launch: the plane and the row scales.
/// @param M  rows
/// @param K  input width
/// @return   bytes
size_t inplace_bytes(int M, int K) { return up256((size_t)M * K * sizeof(__half)) + up256((size_t)M * sizeof(float)); }

/// The in-place launch: the PREP into `scr`, then dense_inplace_kernel into `out`.
/// @param tag            the caller's name for a refusal
/// @param w              the [trellis | suh | svh] slice
/// @param k2             rate
/// @param x_bf16         bf16 input rows [M][K]
/// @param out            out: f32 [M][Ny]
/// @param M              rows
/// @param K              input width
/// @param Ny             logical output width
/// @param trellis_bytes  bytes of the trellis plane
/// @param scr            inplace_bytes(M, K) of the stream's scratch
/// @param stream         the launch stream
/// @return               0, -1 refused, -3 launch failure
int inplace_launch(const char *tag, const void *w, int k2, const void *x_bf16, float *out, int M, int K, int Ny,
                   uint64_t trellis_bytes, uint8_t *scr, cudaStream_t stream) {
    const inplace_fn kern = inplace_instance(k2);
    if (!kern) {
        fprintf(stderr, "%s: rate k2=%d has no instance (K = 2, 3, 4, 5, 6, 8) -- refusing\n", tag, k2);
        return -1;
    }
    const int N = (int)exl3_out_stored((uint64_t)Ny);
    __half *xhi = reinterpret_cast<__half *>(scr);
    float *inv = reinterpret_cast<float *>(scr + up256((size_t)M * K * sizeof(__half)));
    const __half *suh = reinterpret_cast<const __half *>(static_cast<const uint8_t *>(w) + trellis_bytes);
    if (!prep_launch(tag, x_bf16, suh, xhi, inv, M, K, stream)) return -1;
    kern<<<dim3((unsigned)(N >> 7), (unsigned)((M + kInRows - 1) / kInRows)), kInThreads, sizeof(inplace_smem), stream>>>(
        static_cast<const uint32_t *>(w), xhi, inv, suh + K, out, M, K, N, Ny);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: in-place launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

/// Shapes the check can record (a process serves a few dozen; a table that is full leaves further shapes on the GEMM).
constexpr int kCheckedSlots = 128;
/// A slot's bit that says the shape's first in-place launch was announced.
constexpr uint64_t kServedBit = 1ull << 62;
/// The checked shapes: 0 = free, else shape_key() | kServedBit.  Filled in order by CAS, never emptied in a serving
/// process, read without a lock.
uint64_t g_checked[kCheckedSlots] = {};
/// A check found a mismatch: the in-place kernel is off for every shape for the life of the process.
bool g_inplace_off = false;
/// An instrument's forced kernel (ds4_exl3_dense_prompt_force): 0 = the rule's.  The engine never sets it.
int g_prompt_force = 0;
/// An instrument's fault in the check (ds4_exl3_dense_prompt_check_fault).  The engine never sets it.
bool g_check_fault = false;

/// A shape's key: the device slot, the rate, K and the logical N; never 0, never with kServedBit.
/// @param dev  the device's slot
/// @param k2   rate
/// @param K    input width
/// @param Ny   logical output width
/// @return     the key, or 0 for a shape the key cannot hold (it keeps the GEMM)
uint64_t shape_key(int dev, int k2, int K, int Ny) {
    if (dev < 0 || dev >= 16 || k2 < 1 || k2 >= 32 || K < 1 || K >= (1 << 24) || Ny < 1 || Ny >= (1 << 28)) return 0;
    return (uint64_t)K | (uint64_t)Ny << 24 | (uint64_t)k2 << 52 | (uint64_t)dev << 57;
}

/// What the table says of a key.
enum checked_state { CHECKED_NO = 0, CHECKED_YES = 1, CHECKED_FULL = 2 };

/// Looks a shape up.
/// @param key   shape_key()
/// @param slot  out (may be nullptr): its slot when found
/// @return      CHECKED_YES, CHECKED_NO (a free slot is left), or CHECKED_FULL (absent and no slot is free)
checked_state checked_find(uint64_t key, uint64_t **slot) {
    for (uint64_t &s : g_checked) {
        const uint64_t v = __atomic_load_n(&s, __ATOMIC_ACQUIRE);
        if (!v) return CHECKED_NO;
        if ((v & ~kServedBit) == key) {
            if (slot) *slot = &s;
            return CHECKED_YES;
        }
    }
    return CHECKED_FULL;
}

/// Records a shape that passed its check (twice is once: two threads may check one shape).
/// @param key  shape_key()
void checked_add(uint64_t key) {
    for (uint64_t &s : g_checked) {
        uint64_t v = 0;
        if (__atomic_compare_exchange_n(&s, &v, key, false, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)) return;
        if ((v & ~kServedBit) == key) return;
    }
}

/// The kernel a prompt launch takes now: the forced one, else the rule's subject to the check's state.
/// @param dev  the device's slot
/// @param k2   rate
/// @param M    rows
/// @param K    input width
/// @param Ny   logical output width
/// @param key  out: the shape's key when the answer is _INPLACE or _CHECK by the rule (0 when forced)
/// @return     DS4_EXL3_DENSE_PROMPT_GEMM, _INPLACE or _CHECK
int prompt_pick(int dev, int k2, int M, int K, int Ny, uint64_t *key) {
    *key = 0;
    if (g_prompt_force) return g_prompt_force;
    if (ds4_exl3_dense_prompt_arm(M, pulsar_cuda_device_cc()) != DS4_EXL3_DENSE_PROMPT_INPLACE ||
        __atomic_load_n(&g_inplace_off, __ATOMIC_ACQUIRE))
        return DS4_EXL3_DENSE_PROMPT_GEMM;
    const uint64_t k = shape_key(dev, k2, K, Ny);
    const checked_state st = k ? checked_find(k, nullptr) : CHECKED_FULL;
    if (st == CHECKED_FULL) {
        static bool announced = false;
        if (!__atomic_exchange_n(&announced, true, __ATOMIC_RELAXED))
            fprintf(stderr, "pulsar: EXL3 dense prompt rows: %d shapes are checked for the in-place kernel and the table "
                            "is full (or K=%d N=%d has no key): further shapes keep the fixed GEMM, the same bytes\n",
                    kCheckedSlots, K, Ny);
        return DS4_EXL3_DENSE_PROMPT_GEMM;
    }
    *key = k;
    return st == CHECKED_YES ? DS4_EXL3_DENSE_PROMPT_INPLACE : DS4_EXL3_DENSE_PROMPT_CHECK;
}

/// THE CHECK (the file's header): both kernels on this launch's operands, every output byte compared; y holds the
/// GEMM arm's bytes whatever the verdict.  Equal: the shape is recorded.  Not: one loud line, the in-place kernel off.
/// @param tag            the caller's name for a refusal
/// @param key            the shape's key
/// @param w              the [trellis | suh | svh] slice
/// @param k2             rate
/// @param x_bf16         bf16 input rows [M][K]
/// @param y              out: f32 [M][Ny]
/// @param M              rows
/// @param K              input width
/// @param Ny             logical output width
/// @param trellis_bytes  bytes of the trellis plane
/// @param stream         the launch stream
/// @return               0, -1 refused, -3 launch failure
int checked_launch(const char *tag, uint64_t key, const void *w, int k2, const void *x_bf16, float *y, int M, int K,
                   int Ny, uint64_t trellis_bytes, cudaStream_t stream) {
    const cudaStream_t s = stream ? stream : cudaStreamPerThread;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
        (void)cudaGetLastError();
        /* a capture cannot hold the check's wait and copies: this launch is the GEMM arm, the shape stays unchecked */
        return gemm_arm(tag, w, k2, x_bf16, y, M, K, Ny, trellis_bytes, stream);
    }
    /* the in-place kernel into the scratch past the GEMM arm's bytes (its plane where the GEMM arm's W_q will be: it
     * is done, in stream order, before the decode writes there), then the GEMM arm into y */
    const size_t g_b = gemm_bytes(M, K, Ny), out_b = (size_t)M * Ny * sizeof(float);
    uint8_t *const scr = scratch_take(tag, stream, g_b + up256(out_b));
    if (!scr) return -1;
    float *const d_in = reinterpret_cast<float *>(scr + g_b);
    int rc = inplace_launch(tag, w, k2, x_bf16, d_in, M, K, Ny, trellis_bytes, scr, stream);
    if (!rc) rc = gemm_arm(tag, w, k2, x_bf16, y, M, K, Ny, trellis_bytes, stream);   /* the same scratch: no grow */
    if (rc) return rc;
    float *const h = static_cast<float *>(malloc(2 * out_b));
    bool ok = h && cudaStreamSynchronize(s) == cudaSuccess;
    if (ok && g_check_fault) {   /* an instrument's fault: one bit of the in-place kernel's first output */
        uint32_t v = 0;
        ok = cudaMemcpy(&v, d_in, sizeof v, cudaMemcpyDeviceToHost) == cudaSuccess;
        v ^= 1u;
        ok = ok && cudaMemcpy(d_in, &v, sizeof v, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    ok = ok && cudaMemcpy(h, d_in, out_b, cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(h + (size_t)M * Ny, y, out_b, cudaMemcpyDeviceToHost) == cudaSuccess;
    if (!ok) {
        (void)cudaGetLastError();
        free(h);
        fprintf(stderr, "%s: the in-place kernel's check could not read its outputs back (K=%d N=%d M=%d) -- refusing\n",
                tag, K, Ny, M);
        return -3;
    }
    const float *const a = h, *const b = h + (size_t)M * Ny;
    const size_t n_out = (size_t)M * Ny;
    size_t first = n_out, n_diff = 0;
    if (memcmp(a, b, out_b))
        for (size_t i = 0; i < n_out; ++i)
            if (memcmp(a + i, b + i, sizeof(float))) {
                if (first == n_out) first = i;
                ++n_diff;
            }
    const int cc = pulsar_cuda_device_cc(), ltv = (int)cublasLtGetVersion();
    if (!n_diff) {
        checked_add(key);
        fprintf(stderr, "pulsar: EXL3 dense prompt rows K=%d N=%d k2=%d: the in-place kernel wrote the fixed GEMM's bytes "
                        "(%d rows x %d outputs compared; sm_%d, cuBLASLt %d) -- this shape's launches of up to %d rows read "
                        "the trellis in place from here (L381)\n", K, Ny, k2, M, Ny, cc, ltv,
                DS4_EXL3_DENSE_PROMPT_INPLACE_ROWS);
    } else if (!__atomic_exchange_n(&g_inplace_off, true, __ATOMIC_ACQ_REL)) {
        uint32_t ua, ub;
        memcpy(&ua, a + first, sizeof ua);
        memcpy(&ub, b + first, sizeof ub);
        fprintf(stderr, "pulsar: *** EXL3 dense prompt rows K=%d N=%d k2=%d: the in-place kernel's bytes are NOT the fixed "
                        "GEMM's on this device (sm_%d, cuBLASLt %d): %zu of %zu outputs differ, the first at row %zu column "
                        "%zu (in place %.9g = 0x%08x, GEMM %.9g = 0x%08x).  The in-place kernel is OFF for every shape "
                        "in this process: every prompt chunk takes the fixed GEMM, this launch included, so no prompt "
                        "byte moved.  Run tests/lt_order_probe on this device and library. ***\n", K, Ny, k2, cc, ltv,
                n_diff, n_out, first / (size_t)Ny, first % (size_t)Ny, (double)a[first], ua, (double)b[first], ub);
    }
    free(h);
    return 0;
}

} // namespace

int ds4_exl3_dense_prompt_arm(int M, int cc) {
    return cc / 10 == 12 && M >= 1 && M <= DS4_EXL3_DENSE_PROMPT_INPLACE_ROWS ? DS4_EXL3_DENSE_PROMPT_INPLACE
                                                                             : DS4_EXL3_DENSE_PROMPT_GEMM;
}

int ds4_exl3_dense_prompt_force(int arm) {
    if (arm != 0 && arm != DS4_EXL3_DENSE_PROMPT_GEMM && arm != DS4_EXL3_DENSE_PROMPT_INPLACE) return -1;
    g_prompt_force = arm;
    return 0;
}

void ds4_exl3_dense_prompt_check_fault(bool on) { g_check_fault = on; }

void ds4_exl3_dense_prompt_check_reset(void) {
    for (uint64_t &s : g_checked) __atomic_store_n(&s, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_inplace_off, false, __ATOMIC_RELEASE);
}

int ds4_exl3_dense_prompt_describe(int k2, int M, int K, int N, ds4_exl3_dense_prompt_desc *d) {
    const inplace_fn kern = inplace_instance(k2);
    const int dev = pulsar_cuda_device_slot();
    if (!d || !kern || dev < 0 || M < 1 || K < 1 || K % EXL3_HAD_BLOCK || N < 1 || N % 4) return -1;
    cudaFuncAttributes fa;
    int occ = 0;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, kInThreads, sizeof(inplace_smem)) != cudaSuccess) {
        (void)cudaGetLastError();
        return -1;
    }
    uint64_t key = 0;
    const int arm = prompt_pick(dev, k2, M, K, N, &key);
    const uint64_t sk = shape_key(dev, k2, K, N);
    *d = ds4_exl3_dense_prompt_desc{arm, __atomic_load_n(&g_inplace_off, __ATOMIC_ACQUIRE) ? 1 : 0,
                                    sk && checked_find(sk, nullptr) == CHECKED_YES ? 1 : 0,
                                    (int)(exl3_out_stored((uint64_t)N) >> 7) * ((M + kInRows - 1) / kInRows), kInThreads,
                                    fa.numRegs, (int)(sizeof(inplace_smem) + fa.sharedSizeBytes), occ};
    return 0;
}

int qwen_exl3_dense_prefill_launch(const void *w, int k2, const void *x_bf16, float *y, int M, int K, int Ny,
                                   uint64_t trellis_bytes, cudaStream_t stream) {
    const char *tag = "qwen_exl3_dense_prefill_launch";
    if (!w || !x_bf16 || !y || M <= 0 || Ny <= 0 || Ny % 4 || K % EXL3_HAD_BLOCK) {
        fprintf(stderr, "%s: bad arguments (M=%d K=%d N=%d) -- refusing\n", tag, M, K, Ny);
        return -1;
    }
    const int dev = pulsar_cuda_device_slot();
    if (dev < 0) return -1;
    uint64_t key = 0;
    const int arm = prompt_pick(dev, k2, M, K, Ny, &key);
    if (arm == DS4_EXL3_DENSE_PROMPT_GEMM) return gemm_arm(tag, w, k2, x_bf16, y, M, K, Ny, trellis_bytes, stream);
    if (arm == DS4_EXL3_DENSE_PROMPT_CHECK) return checked_launch(tag, key, w, k2, x_bf16, y, M, K, Ny, trellis_bytes, stream);
    /* in place: the shape passed its check (or an instrument forces the kernel); said once a shape */
    uint64_t *slot = nullptr;
    const bool forced = !key;
    if (forced ? true : checked_find(key, &slot) == CHECKED_YES) {
        static bool forced_said = false;
        const bool first = forced ? !__atomic_exchange_n(&forced_said, true, __ATOMIC_RELAXED)
                                  : !(__atomic_fetch_or(slot, kServedBit, __ATOMIC_ACQ_REL) & kServedBit);
        if (first)
            fprintf(stderr, "pulsar: EXL3 dense prompt launch K=%d N=%d k2=%d of up to %d rows = the prep + the in-place "
                            "kernel on the trellis, no fp16 weight (L381; first call: M=%d%s)\n", K, Ny, k2,
                    DS4_EXL3_DENSE_PROMPT_INPLACE_ROWS, M, forced ? ", FORCED by an instrument at every M" : "");
    }
    uint8_t *const scr = scratch_take(tag, stream, inplace_bytes(M, K));
    if (!scr) return -1;
    return inplace_launch(tag, w, k2, x_bf16, y, M, K, Ny, trellis_bytes, scr, stream);
}
