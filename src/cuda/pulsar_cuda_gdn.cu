/* Gated DeltaNet kernels (L251, Qwen3.8-Flash-Next) -- the contract, the
 * reference math and the state layout are in pulsar_cuda_gdn.h. */
#include "pulsar_cuda_gdn.h"
#include <cuda_bf16.h>
#include "pulsar_cuda_mx.cuh"

#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

namespace {

constexpr int NK = PULSAR_GDN_NK, NV = PULSAR_GDN_NV, DK = PULSAR_GDN_DK, DV = PULSAR_GDN_DV;
constexpr int QKV = PULSAR_GDN_QKV_DIM, VDIM = PULSAR_GDN_V_DIM;
constexpr int Q_OFF = 0, K_OFF = NK * DK, V_OFF = 2 * NK * DK;
constexpr int GQA = NV / NK;                 /* 3 V heads per K head */
constexpr int GB = 2 * NV;                   /* scratch per row: decay[48], beta[48] */
constexpr float EPS = 1e-6f;                 /* l2norm and the gated norm (rms_norm_eps) */

static_assert(DK == 128 && DV == 128, "one 128-thread CTA per conv head");
static_assert(NV % NK == 0, "GQA");
static_assert(PULSAR_GDN_KCONV == 4, "the conv window is three registers");
static_assert(QKV == 10240 && VDIM == 6144, "Qwen3.8-Flash-Next");

/* ---- kernel 1: conv + SiLU + L2 norms + gates -------------------------------
 * CTA (sequence x tile, head): 128 threads = the head's 128 channels; heads
 * 0..15 are q, 16..31 k, 32..79 v.  A thread slides its channel's 4-tap window
 * over the tile's tokens.  Every per-token value is computed the same way
 * whatever tile it lands in (the window comes from the rows before it, or the
 * conv state before row 0), so the result is independent of the tiling. */
constexpr int T1 = 16;                       /* tokens per kernel-1 tile */
constexpr int CONV_HEADS = QKV / 128;        /* 80 */

__global__ void __launch_bounds__(128)
gdn_conv_prep_kernel(const float *__restrict__ qkv, int ld_qkv,
                     const float *__restrict__ a, int ld_a,
                     const float *__restrict__ b, int ld_b,
                     const uint16_t *__restrict__ conv_w,
                     const uint16_t *__restrict__ A_log,
                     const uint16_t *__restrict__ dt_bias,
                     int seq_rows,
                     const int32_t *__restrict__ row_slot,
                     float *__restrict__ conv_state,
                     float *__restrict__ qkvn,       /* scratch [rows][QKV] */
                     float *__restrict__ gb,         /* scratch [rows][GB]  */
                     int tiles_grid) {
    __shared__ float part[T1][4];
    const int seq  = blockIdx.x / tiles_grid;
    const int tg   = blockIdx.x % tiles_grid;
    const int head = blockIdx.y;
    const int tid  = threadIdx.x;
    const int ch   = head * 128 + tid;
    const int r0   = seq * seq_rows;
    const int T    = seq_rows;
    float *cs = conv_state + (size_t)row_slot[r0] * PULSAR_GDN_CONV_STATE_FLOATS;

    /* bf16 -> f32 is a left shift of the 16 stored bits (exact). */
    const uint2 cwr = reinterpret_cast<const uint2 *>(conv_w)[ch];
    const float4 cw = make_float4(__uint_as_float((uint32_t)(uint16_t)cwr.x << 16),
                                  __uint_as_float(cwr.x & 0xFFFF0000u),
                                  __uint_as_float((uint32_t)(uint16_t)cwr.y << 16),
                                  __uint_as_float(cwr.y & 0xFFFF0000u));
    const bool is_qk = head < 2 * NK;
    const bool is_q  = head < NK;
    const int  hv    = head - 2 * NK;        /* V head for the gate columns */

    /* The old conv state: only the tile-0 CTA reads it (the first three
     * tokens' windows reach into it) and only that CTA writes it, below. */
    float old0 = 0.f, old1 = 0.f, old2 = 0.f;
    if (tg == 0) { old0 = cs[0 * QKV + ch]; old1 = cs[1 * QKV + ch]; old2 = cs[2 * QKV + ch]; }

    {                                         /* the grid has exactly ceil(T / T1) tiles */
        const int t0 = tg * T1;
        const int tn = min(T1, T - t0);
        float w0, w1, w2;
        if (t0 == 0) { w0 = old0; w1 = old1; w2 = old2; }
        else {
            w0 = qkv[(size_t)(r0 + t0 - 3) * ld_qkv + ch];
            w1 = qkv[(size_t)(r0 + t0 - 2) * ld_qkv + ch];
            w2 = qkv[(size_t)(r0 + t0 - 1) * ld_qkv + ch];
        }
        float y[T1];
        #pragma unroll
        for (int j = 0; j < T1; j++) {
            y[j] = 0.f;
            if (j < tn) {
                const float x = qkv[(size_t)(r0 + t0 + j) * ld_qkv + ch];
                float acc = cw.x * w0;
                acc = fmaf(cw.y, w1, acc);
                acc = fmaf(cw.z, w2, acc);
                acc = fmaf(cw.w, x, acc);
                y[j] = acc / (1.f + __expf(-acc));           /* SiLU */
                w0 = w1; w1 = w2; w2 = x;
            }
        }
        if (is_qk) {                                          /* block-uniform */
            #pragma unroll
            for (int j = 0; j < T1; j++) {
                float ss = y[j] * y[j];
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
                if ((tid & 31) == 0) part[j][tid >> 5] = ss;
            }
            __syncthreads();
            #pragma unroll
            for (int j = 0; j < T1; j++) {
                if (j < tn) {
                    const float ss = ((part[j][0] + part[j][1]) + part[j][2]) + part[j][3];
                    float v = y[j] * rsqrtf(ss + EPS);
                    if (is_q) v *= 0.08838834764831845f;      /* 128^-0.5 */
                    y[j] = v;
                }
            }
        } else if (tid < tn) {
            const int row = r0 + t0 + tid;
            /* decay in DOUBLE: it multiplies the state every token, so its
             * error compounds over the head's memory (~1/(1-decay) tokens, 5e4
             * at the slowest heads); fast-math __expf is ~2 ulp.  48 per row. */
            const double sp_in = (double)(a[(size_t)row * ld_a + hv] + __uint_as_float((uint32_t)dt_bias[hv] << 16));
            const double sp = sp_in > 20.0 ? sp_in : log1p(exp(sp_in));      /* F.softplus */
            gb[(size_t)row * GB + hv]      = (float)exp(-exp((double)__uint_as_float((uint32_t)A_log[hv] << 16)) * sp);
            gb[(size_t)row * GB + NV + hv] = 1.f / (1.f + __expf(-b[(size_t)row * ld_b + hv]));
        }
        #pragma unroll
        for (int j = 0; j < T1; j++)
            if (j < tn) qkvn[(size_t)(r0 + t0 + j) * QKV + ch] = y[j];
    }

    /* Advance the conv state: the last three inputs of the sequence, reaching
     * into the old state when the sequence is shorter than three.  Only the
     * tile-0 CTA reads the old state, and it read it above. */
    if (tg == 0) {
        /* position p of the new state is T-3+j; p < 0 is old-state index T+j */
        const float n0 = T >= 3 ? qkv[(size_t)(r0 + T - 3) * ld_qkv + ch] : (T == 1 ? old1 : old2);
        const float n1 = T >= 2 ? qkv[(size_t)(r0 + T - 2) * ld_qkv + ch] : old2;
        const float n2 = qkv[(size_t)(r0 + T - 1) * ld_qkv + ch];
        cs[0 * QKV + ch] = n0;
        cs[1 * QKV + ch] = n1;
        cs[2 * QKV + ch] = n2;
    }
}

/* ---- the verify capture (L251 MTP): the conv state after each row r < T - 1 of ONE sequence --------
 * The state after row r is the inputs at call-relative positions r - 2, r - 1, r, reaching into the
 * OLD state for negative positions -- the same rule the conv kernel's advance applies to the whole
 * call.  Launched before that kernel, so the old state is still in the pool.  grid (T - 1, 80). */
__global__ void __launch_bounds__(128)
gdn_conv_rows_kernel(const float *__restrict__ qkv, int ld_qkv, const int32_t *__restrict__ row_slot,
                     const float *__restrict__ conv_state, float *__restrict__ conv_rows) {
    const int r = blockIdx.x, ch = blockIdx.y * 128 + threadIdx.x;
    const float *cs = conv_state + (size_t)row_slot[0] * PULSAR_GDN_CONV_STATE_FLOATS;
    float *dst = conv_rows + (size_t)r * PULSAR_GDN_CONV_STATE_FLOATS;
    #pragma unroll
    for (int j = 0; j < 3; j++) {
        const int p = r - 2 + j;
        dst[j * QKV + ch] = p >= 0 ? qkv[(size_t)p * ld_qkv + ch] : cs[(3 + p) * QKV + ch];
    }
}

/* ---- kernel 2: recurrence + gated RMSNorm + the out_proj activation ---------
 * A cluster of 4 CTAs per (sequence, V head); CTA r owns columns v = 32r..32r+31.
 * 256 threads: warp w owns state rows k = 16w..16w+15, lane l owns column
 * 32r + l: 16 floats of S per thread, in registers for the whole sequence.
 * Per token, one reduction over the 8 warps gives, for the thread's column,
 *   SK = S^T k,  SQ = S^T q,  and  KQ = k . q          (S before this token)
 * and then, with S' = decay S + k delta^T:
 *   delta = beta (v - decay SK),   o = S'^T q = decay SQ + delta KQ.
 * One __syncthreads per token (the partials are double-buffered by parity).
 * Every CTA of the cluster computes KQ from the same values in the same order,
 * so the four agree bit for bit.
 *
 * The gated norm needs mean(o^2) over the head's 128 columns: each CTA reduces
 * its 32 (one warp per token), the cluster exchanges the four partials through
 * distributed shared memory and every CTA sums them in rank order.  The CTA's
 * 32 columns are one MX block, so the E4M3 emission is the canonical warp
 * encoder. */
constexpr int T2 = 8;                        /* tokens staged per tile */
constexpr int RW = 8;                        /* warps */
constexpr int RPW = DK / RW;                 /* 16 state rows per warp */
constexpr int VQ = 4;                        /* CTAs (column quarters) per V head */
static_assert(DV == VQ * 32, "a CTA's columns are one warp wide");

__global__ void __cluster_dims__(1, 1, VQ) __launch_bounds__(256)
gdn_recur_norm_kernel(const float *__restrict__ qkvn,
                      const float *__restrict__ gb,
                      const float *__restrict__ z, int ld_z,
                      const uint16_t *__restrict__ norm_w,
                      int seq_rows,
                      const int32_t *__restrict__ row_slot,
                      float *__restrict__ rec_state,
                      float *__restrict__ rec_rows,
                      float *__restrict__ out_f32,
                      __nv_bfloat16 *__restrict__ out_bf16,
                      __nv_fp8_e4m3 *__restrict__ out_q,
                      unsigned char *__restrict__ out_s, int kbp) {
    __shared__ float4 sq[T2][DK / 4], sk[T2][DK / 4];
    __shared__ float4 sv4[T2][8];
    __shared__ float  so[T2][32];
    __shared__ float  sdec[T2], sbeta[T2];
    __shared__ float2 red[2][RW][32];
    __shared__ float  redKQ[2][RW];
    __shared__ float  ssp[2][T2];            /* this CTA's sum of o^2, read by the cluster */

    cg::cluster_group cluster = cg::this_cluster();
    const int seq  = blockIdx.x;
    const int h    = blockIdx.y;
    const int qr   = (int)cluster.block_rank();          /* == blockIdx.z */
    const int kh   = h / GQA;
    const int tid  = threadIdx.x;
    const int w    = tid >> 5;
    const int lane = tid & 31;
    const int col  = qr * 32 + lane;                      /* column within the head */
    const int r0   = seq * seq_rows;
    const int T    = seq_rows;

    float *S = rec_state + (size_t)row_slot[r0] * PULSAR_GDN_REC_STATE_FLOATS + (size_t)h * DK * DV;
    float s[RPW];
    #pragma unroll
    for (int i = 0; i < RPW; i++) s[i] = S[(w * RPW + i) * DV + col];

    const float nw = __uint_as_float((uint32_t)norm_w[col] << 16);
    int par = 0, tpar = 0;

    for (int t0 = 0; t0 < T; t0 += T2) {
        const int tn = min(T2, T - t0);
        /* Stage the tile: q^, k^ of K head kh; this CTA's 32 v; decay, beta. */
        for (int i = tid; i < tn * 72; i += 256) {
            const int j = i / 72, r = i % 72;
            const float *row = qkvn + (size_t)(r0 + t0 + j) * QKV;
            if (r < 32)      sq[j][r]       = reinterpret_cast<const float4 *>(row + Q_OFF + kh * DK)[r];
            else if (r < 64) sk[j][r - 32]  = reinterpret_cast<const float4 *>(row + K_OFF + kh * DK)[r - 32];
            else             sv4[j][r - 64] = reinterpret_cast<const float4 *>(row + V_OFF + h * DV + qr * 32)[r - 64];
        }
        if (tid < tn) {
            sdec[tid]  = gb[(size_t)(r0 + t0 + tid) * GB + h];
            sbeta[tid] = gb[(size_t)(r0 + t0 + tid) * GB + NV + h];
        }
        __syncthreads();

        for (int j = 0; j < tn; j++) {
            float kr[RPW], qv[RPW];
            #pragma unroll
            for (int i = 0; i < RPW / 4; i++) {
                const float4 k4 = sk[j][w * (RPW / 4) + i], q4 = sq[j][w * (RPW / 4) + i];
                kr[4 * i] = k4.x; kr[4 * i + 1] = k4.y; kr[4 * i + 2] = k4.z; kr[4 * i + 3] = k4.w;
                qv[4 * i] = q4.x; qv[4 * i + 1] = q4.y; qv[4 * i + 2] = q4.z; qv[4 * i + 3] = q4.w;
            }
            float pk = 0.f, pq = 0.f, kq = 0.f;
            #pragma unroll
            for (int i = 0; i < RPW; i++) {
                pk = fmaf(s[i], kr[i], pk);
                pq = fmaf(s[i], qv[i], pq);
                kq = fmaf(kr[i], qv[i], kq);
            }
            red[par][w][lane] = make_float2(pk, pq);
            if (lane == 0) redKQ[par][w] = kq;
            __syncthreads();

            float2 acc = red[par][0][lane];
            float KQ = redKQ[par][0];
            #pragma unroll
            for (int u = 1; u < RW; u++) {
                const float2 p2 = red[par][u][lane];
                acc.x += p2.x; acc.y += p2.y;
                KQ += redKQ[par][u];
            }
            const float dec = sdec[j];
            const float vv = reinterpret_cast<const float *>(sv4[j])[lane];
            const float d = sbeta[j] * (vv - dec * acc.x);
            if (w == 0) so[j][lane] = fmaf(d, KQ, dec * acc.y);
            #pragma unroll
            for (int i = 0; i < RPW; i++) s[i] = fmaf(kr[i], d, dec * s[i]);
            if (rec_rows && t0 + j < T - 1) {               /* the verify capture: S after this row */
                float *R = rec_rows + (size_t)(t0 + j) * PULSAR_GDN_REC_STATE_FLOATS + (size_t)h * DK * DV;
                #pragma unroll
                for (int i = 0; i < RPW; i++) R[(w * RPW + i) * DV + col] = s[i];
            }
            par ^= 1;
        }
        __syncthreads();                                   /* so[] complete */

        /* This CTA's share of mean(o^2), one warp per token, then the cluster
         * exchange -- double-buffered by tile parity: a peer reads buffer tpar
         * before it arrives at the NEXT tile's cluster barrier, and this CTA
         * rewrites buffer tpar only after that barrier. */
        for (int j = w; j < tn; j += RW) {
            float ss = so[j][lane] * so[j][lane];
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
            if (lane == 0) ssp[tpar][j] = ss;
        }
        cluster.sync();
        for (int j = w; j < tn; j += RW) {
            float ss = 0.f;
            #pragma unroll
            for (int r = 0; r < VQ; r++) ss += cluster.map_shared_rank(&ssp[tpar][0], r)[j];
            const int row = r0 + t0 + j;
            const float inv = rsqrtf(ss * (1.f / DV) + EPS);
            const float zz = z[(size_t)row * ld_z + h * DV + col];
            const float y = nw * (so[j][lane] * inv) * (1.f / (1.f + __expf(-zz)));
            if (out_f32) out_f32[(size_t)row * VDIM + h * DV + col] = y;
            /* L251 / ac69748f: the GDN output's bf16 row is what the out_proj reads, so the op
             * emits it here rather than an E4M3 slot (rule 3: the producer decides the format). */
            if (out_bf16) out_bf16[(size_t)row * VDIM + h * DV + col] = __float2bfloat16(y);
            if (out_q) pulsar_mx_emit_block(y, (uint32_t)(h * DV + col), (uint32_t)row, VDIM, kbp, out_q, out_s);
        }
        tpar ^= 1;
        __syncthreads();                                   /* tiles and so[] reused */
    }

    #pragma unroll
    for (int i = 0; i < RPW; i++) S[(w * RPW + i) * DV + col] = s[i];
    cluster.sync();                                        /* no CTA leaves while a peer may read its ssp */
}

bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

int refuse(const char *what) {
    fprintf(stderr, "pulsar: gdn: refused -- %s\n", what);
    return -1;
}

} // namespace

extern "C" size_t pulsar_gdn_scratch_bytes(int rows) {
    if (rows < 1) return 0;
    return (size_t)rows * (QKV + GB) * sizeof(float);
}

extern "C" int pulsar_gdn_forward(const pulsar_gdn_weights *w, const pulsar_gdn_call *c, cudaStream_t stream) {
    if (!w || !c) return refuse("null weights or call");
    if (!w->conv_w || !w->A_log || !w->dt_bias || !w->norm_w) return refuse("a weight pointer is null");
    if (((uintptr_t)w->conv_w & 7u) != 0) return refuse("conv_w not 8-byte aligned (a bf16 uint2 row)");
    if (c->n_seq < 1 || c->seq_rows < 1 || (int64_t)c->n_seq * c->seq_rows > (1 << 24))
        return refuse("n_seq / seq_rows out of range");
    const int rows = c->n_seq * c->seq_rows;
    if (!c->row_slot || !c->conv_state || !c->rec_state) return refuse("row_slot or a state pointer is null");
    if (!c->qkv || !c->z || !c->a || !c->b) return refuse("an input pointer is null");
    if (!aligned16(c->qkv) || !aligned16(c->z) || !aligned16(c->rec_state) || !aligned16(c->scratch))
        return refuse("qkv / z / rec_state / scratch not 16-byte aligned");
    if (c->ld_qkv < QKV || c->ld_z < VDIM || c->ld_a < NV || c->ld_b < NV || (c->ld_qkv & 3) || (c->ld_z & 3))
        return refuse("an input pitch is short or not a multiple of 4 floats");
    if (!c->scratch || c->scratch_bytes < pulsar_gdn_scratch_bytes(rows)) return refuse("scratch missing or short");
    if (!c->out_f32 && !c->out_e4m3 && !c->out_bf16)
        return refuse("no output (out_f32, out_e4m3 and out_bf16 all NULL)");
    if (c->out_bf16 && ((uintptr_t)c->out_bf16 & 1u)) return refuse("out_bf16 not 2-byte aligned");
    if (c->out_f32 && !aligned16(c->out_f32)) return refuse("out_f32 not 16-byte aligned");
    if (c->out_e4m3 && (!c->out_scale || c->out_kbp != pulsar_mx_kbp(VDIM) || ((uintptr_t)c->out_e4m3 & 3u)))
        return refuse("A8 slot: scale missing, kbp != pulsar_mx_kbp(6144), or data not 4-byte aligned");

    float *qkvn = static_cast<float *>(c->scratch);
    float *gb   = qkvn + (size_t)rows * QKV;
    const int tiles = (c->seq_rows + T1 - 1) / T1;
    if ((c->conv_rows == nullptr) != (c->rec_rows == nullptr) || (c->conv_rows && c->n_seq != 1))
        return refuse("the verify capture needs both row buffers and one sequence");
    if (c->conv_rows && c->seq_rows > 1)
        gdn_conv_rows_kernel<<<dim3((unsigned)(c->seq_rows - 1), CONV_HEADS), 128, 0, stream>>>(
            c->qkv, c->ld_qkv, c->row_slot, c->conv_state, c->conv_rows);

    gdn_conv_prep_kernel<<<dim3((unsigned)(c->n_seq * tiles), CONV_HEADS), 128, 0, stream>>>(
        c->qkv, c->ld_qkv, c->a, c->ld_a, c->b, c->ld_b, w->conv_w, w->A_log, w->dt_bias,
        c->seq_rows, c->row_slot, c->conv_state, qkvn, gb, tiles);
    gdn_recur_norm_kernel<<<dim3((unsigned)c->n_seq, NV, VQ), 256, 0, stream>>>(
        qkvn, gb, c->z, c->ld_z, w->norm_w, c->seq_rows, c->row_slot, c->rec_state, c->rec_rows,
        c->out_f32, static_cast<__nv_bfloat16 *>(c->out_bf16),
        static_cast<__nv_fp8_e4m3 *>(c->out_e4m3), static_cast<unsigned char *>(c->out_scale), c->out_kbp);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: gdn: launch failed: %s\n", cudaGetErrorString(e));
        return -3;
    }
    return 0;
}
