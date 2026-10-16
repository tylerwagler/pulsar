/* Mamba-2 (SSD) mixer block (L296 N2) -- the contract, the math and the state
 * layout are in pulsar_cuda_mamba2.h.
 *
 * The launches of a call:
 *   1. the x | B | C convolution: E3's block (pulsar_sconv_forward) with the
 *      layer's bias and SiLU, into the call's scratch (bf16 [rows][CC]);
 *   2. dt / dA per (row, head) in double (mamba2_dt), into the scratch --
 *      only when a sequence runs more than one tile of rows (L296 perf: a
 *      decode-width call computes them in the scan's staging instead, one
 *      launch fewer; a long run keeps the parallel launch, since the double
 *      transcendentals would sit on every tile's serial path).  One device
 *      function either way: the same bytes;
 *   3. the scan: one CTA per (sequence, head, half of the head's P rows),
 *      128 threads; thread (q, p) holds S[p][32q .. 32q+31] in registers for
 *      the whole sequence.  Per row it applies the update to its 32 columns
 *      and sums its share of y[p] = S[p] . C; the four shares of y[p] are added
 *      in a fixed tree after the tile.  The update never reads y, so a tile of
 *      rows runs with no barrier between rows;
 *   4. the gated grouped RMSNorm: one CTA per (row, norm group).
 * The thread that computes an element is fixed by its (head, p, n), never by
 * the call's shape, so every schedule gives the same bytes. */
#include "pulsar_cuda_mamba2.h"
#include "pulsar_cuda_sconv.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

namespace {

constexpr int P = 64;            /* head_dim (the built instance) */
constexpr int N = 128;           /* d_state */
constexpr int NQ = 4;            /* column quarters: a thread's state is one p row's N / NQ columns */
constexpr int NPT = N / NQ;      /* 32 floats of state a thread */
constexpr int PS = 2;            /* CTAs a head (halves of its p rows) */
constexpr int PC = P / PS;       /* p rows a CTA */
constexpr int NT = PC * NQ;      /* 128 threads: warp q owns column quarter q of all PC rows */
constexpr int TT = 16;           /* rows a tile */
static_assert(PC == 32, "a warp is one column quarter of the CTA's p rows");
static_assert(NPT % 8 == 0, "the y share is four chains of NPT / 4");

__device__ __forceinline__ float bf2f(uint16_t b) { return pulsar_sconv_bf2f(b); }
__device__ __forceinline__ uint16_t f2bf(float f) { return pulsar_sconv_f2bf(f); }

/* ---- dt and dA of (row, head), double: softplus(dt_raw + dt_bias) and exp(dt * A), A = -exp(A_log) ------ */
__device__ __forceinline__ void mamba2_dt(uint16_t dt_raw, uint16_t dt_bias, uint16_t A_log, float *dt_out,
                                          float *dA_out) {
    const float s = __fadd_rn(bf2f(dt_raw), bf2f(dt_bias));
    const float dt = s > 20.0f ? s : (float)log1p(exp((double)s));            /* softplus, torch's threshold 20 */
    const double A = -exp((double)bf2f(A_log));
    *dt_out = dt;
    *dA_out = (float)exp((double)dt * A);
}

constexpr int DT_THREADS = 256;

/* ---- 2. dt and dA per (row, head) into the scratch (a call with a run longer than a tile) --------------- */
__global__ void __launch_bounds__(DT_THREADS)
mamba2_dt_kernel(const uint16_t *__restrict__ dt_raw, uint64_t ld_dt, const uint16_t *__restrict__ dt_bias,
                 const uint16_t *__restrict__ A_log, uint32_t H, uint32_t rows, float *__restrict__ dt_out,
                 float *__restrict__ dA_out) {
    const uint64_t i = (uint64_t)blockIdx.x * DT_THREADS + threadIdx.x;
    if (i >= (uint64_t)rows * H) return;
    const uint32_t row = (uint32_t)(i / H), h = (uint32_t)(i % H);
    mamba2_dt(dt_raw[(uint64_t)row * ld_dt + h], dt_bias[h], A_log[h], &dt_out[i], &dA_out[i]);
}

/* ---- 3. the scan ------------------------------------------------------------------------------------- */
struct scan_args {
    const uint16_t *conv_out;      /* [rows][CC] */
    const float *dt, *dA;          /* [rows][H] from launch 2, or NULL: the scan stages them itself */
    const uint16_t *dt_raw;        /* row r's head h at dt_raw[r * ld_dt + h] */
    uint64_t ld_dt;
    const uint16_t *dt_bias, *A_log;
    const uint16_t *D;
    const int32_t *seq_first, *seq_slot;
    float *ssm;                    /* pool */
    float *cap;                    /* DELTA records, or NULL */
    uint16_t *y;                   /* [rows][H*P] */
    uint32_t H, G, CC, upd;
};

__global__ void __launch_bounds__(NT) mamba2_scan_kernel(const scan_args a) {
    __shared__ __align__(16) float sdb[TT][N];   /* dt * B, per row of the tile */
    __shared__ __align__(16) float sc[TT][N];    /* C */
    __shared__ float sx[TT][PC];
    __shared__ float sdt[TT], sda[TT];
    __shared__ float part[TT][NQ][PC];

    const uint32_t seq = blockIdx.x, h = blockIdx.y, ph = blockIdx.z;
    const int tid = threadIdx.x, q = tid / PC, pl = tid % PC;
    const uint32_t HP = a.H * P, GN = a.G * N;
    const uint32_t g = h / (a.H / a.G);
    const int32_t r0 = a.seq_first[seq];
    const int32_t T = a.seq_first[seq + 1] - r0;
    const uint64_t slot = (uint64_t)a.seq_slot[seq];
    float *S = a.ssm + slot * HP * N + (uint64_t)h * P * N + (uint64_t)(ph * PC + pl) * N + q * NPT;
    const bool cap_b = a.cap && ph == 0 && h % (a.H / a.G) == 0;   /* the CTA that records its group's B */

    float s[NPT];
#pragma unroll
    for (int i = 0; i < NPT / 4; ++i) {
        const float4 v = reinterpret_cast<const float4 *>(S)[i];
        s[4 * i] = v.x; s[4 * i + 1] = v.y; s[4 * i + 2] = v.z; s[4 * i + 3] = v.w;
    }
    const float Dh = bf2f(a.D[h]);

    for (int32_t t0 = 0; t0 < T; t0 += TT) {
        const int tn = min(TT, (int)(T - t0));
        /* stage: B and C of group g, the CTA's x, dt and dA */
        for (int i = tid; i < tn * (N / 8); i += NT) {
            const int j = i / (N / 8), n8 = (i % (N / 8)) * 8;
            const uint64_t row = (uint64_t)(r0 + t0 + j);
            const uint16_t *cr = a.conv_out + row * a.CC + HP + (uint64_t)g * N + n8;
            const uint4 bv = *reinterpret_cast<const uint4 *>(cr);
            const uint4 cv = *reinterpret_cast<const uint4 *>(cr + GN);
            const uint32_t bu[4] = {bv.x, bv.y, bv.z, bv.w}, cu[4] = {cv.x, cv.y, cv.z, cv.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                sdb[j][n8 + 2 * k] = bf2f((uint16_t)(bu[k] & 0xffffu));
                sdb[j][n8 + 2 * k + 1] = bf2f((uint16_t)(bu[k] >> 16));
                sc[j][n8 + 2 * k] = bf2f((uint16_t)(cu[k] & 0xffffu));
                sc[j][n8 + 2 * k + 1] = bf2f((uint16_t)(cu[k] >> 16));
            }
            if (cap_b) {
                float *U = a.cap + row * a.upd + HP + (uint64_t)g * N + n8;
#pragma unroll
                for (int k = 0; k < 8; ++k) U[k] = sdb[j][n8 + k];
            }
        }
        for (int i = tid; i < tn * PC; i += NT) {
            const int j = i / PC, pp = i % PC;
            sx[j][pp] = bf2f(a.conv_out[(uint64_t)(r0 + t0 + j) * a.CC + (uint64_t)h * P + ph * PC + pp]);
        }
        if (tid < tn) {
            const uint64_t row = (uint64_t)(r0 + t0 + tid);
            if (a.dt) {
                sdt[tid] = a.dt[row * a.H + h];
                sda[tid] = a.dA[row * a.H + h];
            } else {
                mamba2_dt(a.dt_raw[row * a.ld_dt + h], a.dt_bias[h], a.A_log[h], &sdt[tid], &sda[tid]);
            }
        }
        __syncthreads();
        for (int i = tid; i < tn * N; i += NT) {
            const int j = i / N, n = i % N;
            sdb[j][n] = __fmul_rn(sdt[j], sdb[j][n]);
        }
        __syncthreads();

        /* the rows: thread (q, pl) updates its 32 columns and sums its share of y */
        for (int j = 0; j < tn; ++j) {
            const float xv = sx[j][pl], da = sda[j];
            const float4 *db4 = reinterpret_cast<const float4 *>(&sdb[j][q * NPT]);
            const float4 *c4 = reinterpret_cast<const float4 *>(&sc[j][q * NPT]);
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int i = 0; i < NPT / 4; ++i) {
                const float4 db = db4[i], cc = c4[i];
                float *si = &s[4 * i];
                si[0] = __fmaf_rn(db.x, xv, __fmul_rn(da, si[0]));
                si[1] = __fmaf_rn(db.y, xv, __fmul_rn(da, si[1]));
                si[2] = __fmaf_rn(db.z, xv, __fmul_rn(da, si[2]));
                si[3] = __fmaf_rn(db.w, xv, __fmul_rn(da, si[3]));
                float &ac = acc[i / (NPT / 16)];   /* chain k sums columns [8k, 8k + 8) of the thread's 32 */
                ac = __fmaf_rn(si[0], cc.x, ac);
                ac = __fmaf_rn(si[1], cc.y, ac);
                ac = __fmaf_rn(si[2], cc.z, ac);
                ac = __fmaf_rn(si[3], cc.w, ac);
            }
            part[j][q][pl] = __fadd_rn(__fadd_rn(acc[0], acc[1]), __fadd_rn(acc[2], acc[3]));
        }
        __syncthreads();

        /* y = D x + the four shares, in a fixed tree; the DELTA record's x, dt, dA */
        for (int i = tid; i < tn * PC; i += NT) {
            const int j = i / PC, pp = i % PC;
            const uint64_t row = (uint64_t)(r0 + t0 + j);
            const float sum = __fadd_rn(__fadd_rn(part[j][0][pp], part[j][1][pp]), __fadd_rn(part[j][2][pp], part[j][3][pp]));
            const uint32_t col = h * P + ph * PC + pp;
            a.y[row * HP + col] = f2bf(__fmaf_rn(Dh, sx[j][pp], sum));
            if (a.cap) a.cap[row * a.upd + col] = sx[j][pp];
        }
        if (a.cap && ph == 0 && tid < tn) {
            float *U = a.cap + (uint64_t)(r0 + t0 + tid) * a.upd + HP + GN;
            U[h] = sdt[tid];
            U[a.H + h] = sda[tid];
        }
        __syncthreads();   /* the staging buffers are rewritten by the next tile */
    }

    if (!a.cap) {
#pragma unroll
        for (int i = 0; i < NPT / 4; ++i)
            reinterpret_cast<float4 *>(S)[i] = make_float4(s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]);
    }
}

/* ---- the DELTA commit: the scan's own update on the recorded values, row by row ----------------------- */
__global__ void __launch_bounds__(NT)
mamba2_commit_kernel(float *__restrict__ ssm, const float *__restrict__ upd, uint32_t keep, uint32_t H, uint32_t G,
                     uint32_t upd_floats) {
    const uint32_t h = blockIdx.x, ph = blockIdx.y;
    const int tid = threadIdx.x, q = tid / PC, pl = tid % PC;
    const uint32_t HP = H * P, GN = G * N, g = h / (H / G), p = ph * PC + pl;
    float *S = ssm + (uint64_t)h * P * N + (uint64_t)p * N + q * NPT;
    float s[NPT];
#pragma unroll
    for (int i = 0; i < NPT / 4; ++i) {
        const float4 v = reinterpret_cast<const float4 *>(S)[i];
        s[4 * i] = v.x; s[4 * i + 1] = v.y; s[4 * i + 2] = v.z; s[4 * i + 3] = v.w;
    }
    for (uint32_t r = 0; r < keep; ++r) {
        const float *U = upd + (uint64_t)r * upd_floats;
        const float xv = U[h * P + p], dt = U[HP + GN + h], da = U[HP + GN + H + h];
        const float4 *b4 = reinterpret_cast<const float4 *>(U + HP + (uint64_t)g * N + q * NPT);
#pragma unroll
        for (int i = 0; i < NPT / 4; ++i) {
            const float4 b = b4[i];
            float *si = &s[4 * i];
            si[0] = __fmaf_rn(__fmul_rn(dt, b.x), xv, __fmul_rn(da, si[0]));
            si[1] = __fmaf_rn(__fmul_rn(dt, b.y), xv, __fmul_rn(da, si[1]));
            si[2] = __fmaf_rn(__fmul_rn(dt, b.z), xv, __fmul_rn(da, si[2]));
            si[3] = __fmaf_rn(__fmul_rn(dt, b.w), xv, __fmul_rn(da, si[3]));
        }
    }
#pragma unroll
    for (int i = 0; i < NPT / 4; ++i)
        reinterpret_cast<float4 *>(S)[i] = make_float4(s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]);
}

/* ---- 4. the gated grouped RMSNorm: one CTA per (row, group), 8 channels a thread ---------------------- */
__global__ void __launch_bounds__(1024)
mamba2_norm_kernel(const uint16_t *__restrict__ y, uint32_t HP, const uint16_t *__restrict__ z, uint64_t ld_z,
                   const uint16_t *__restrict__ w, uint32_t ng, float eps, uint16_t *__restrict__ out, uint64_t ld_out) {
    __shared__ float wsum[32];
    __shared__ float total;
    const uint64_t row = blockIdx.x;
    const uint32_t c0 = blockIdx.y * ng + threadIdx.x * 8;
    const uint4 yv = *reinterpret_cast<const uint4 *>(y + row * HP + c0);
    const uint4 zv = *reinterpret_cast<const uint4 *>(z + row * ld_z + c0);
    const uint4 wv = *reinterpret_cast<const uint4 *>(w + c0);
    const uint32_t yu[4] = {yv.x, yv.y, yv.z, yv.w}, zu[4] = {zv.x, zv.y, zv.z, zv.w}, wu[4] = {wv.x, wv.y, wv.z, wv.w};
    float v[8], ss = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float yy = bf2f((uint16_t)(yu[k / 2] >> (16 * (k & 1)))), zz = bf2f((uint16_t)(zu[k / 2] >> (16 * (k & 1))));
        v[k] = __fmul_rn(yy, zz / (1.0f + __expf(-zz)));
        ss = __fmaf_rn(v[k], v[k], ss);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = wsum[0];
        for (uint32_t i = 1; i < blockDim.x / 32; ++i) t = __fadd_rn(t, wsum[i]);
        total = t;
    }
    __syncthreads();
    const float r = __frsqrt_rn(__fadd_rn(__fdiv_rn(total, (float)ng), eps));
    uint32_t o[4];
#pragma unroll
    for (int k = 0; k < 8; k += 2) {
        const float w0 = bf2f((uint16_t)(wu[k / 2] & 0xffffu)), w1 = bf2f((uint16_t)(wu[k / 2] >> 16));
        const uint16_t lo = f2bf(__fmul_rn(w0, bf2f(f2bf(__fmul_rn(v[k], r)))));
        const uint16_t hi = f2bf(__fmul_rn(w1, bf2f(f2bf(__fmul_rn(v[k + 1], r)))));
        o[k / 2] = (uint32_t)lo | ((uint32_t)hi << 16);
    }
    *reinterpret_cast<uint4 *>(out + row * ld_out + c0) = make_uint4(o[0], o[1], o[2], o[3]);
}

int refuse(const char *what) {
    fprintf(stderr, "pulsar: mamba2: refused -- %s\n", what);
    return -1;
}

bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

size_t round16(size_t b) { return (b + 15u) & ~(size_t)15u; }

/* the geometry the built kernels run, or the reason it is refused */
const char *geom_refusal(const pulsar_mamba2_geom *g) {
    if (!g) return "no geometry";
    if (g->head_dim != (uint32_t)P || g->d_state != (uint32_t)N)
        return "head_dim x d_state is not the built instance (64 x 128)";
    if (g->n_heads == 0 || g->n_groups == 0 || g->n_heads % g->n_groups)
        return "n_heads must be a positive multiple of n_groups";
    if (g->n_heads > 65535u) return "n_heads above one grid column (65535)";
    if (g->conv_kernel < 2 || g->conv_kernel > PULSAR_SCONV_KMAX) return "conv_kernel outside 2 .. PULSAR_SCONV_KMAX";
    if (g->norm_group == 0 || g->norm_group % 256 || g->norm_group > 8192 || (g->n_heads * P) % g->norm_group)
        return "norm_group must be a multiple of 256, at most 8192, dividing n_heads * head_dim";
    if (!(g->norm_eps > 0.0f)) return "norm_eps must be positive";
    return nullptr;
}

/* [lo, hi) bytes of `rows` rows of `C` bf16 at stride ld */
void span(const uint16_t *p, uint64_t ld, uint32_t rows, uint64_t C, uintptr_t *lo, uintptr_t *hi) {
    *lo = (uintptr_t)p;
    *hi = (uintptr_t)(p + (uint64_t)(rows - 1) * ld + C);
}

} // namespace

extern "C" uint32_t pulsar_mamba2_conv_channels(const pulsar_mamba2_geom *g) {
    return g ? g->n_heads * g->head_dim + 2u * g->n_groups * g->d_state : 0;
}
extern "C" uint64_t pulsar_mamba2_conv_state_elems(const pulsar_mamba2_geom *g) {
    return g && g->conv_kernel >= 1 ? (uint64_t)(g->conv_kernel - 1) * pulsar_mamba2_conv_channels(g) : 0;
}
extern "C" uint64_t pulsar_mamba2_ssm_state_floats(const pulsar_mamba2_geom *g) {
    return g ? (uint64_t)g->n_heads * g->head_dim * g->d_state : 0;
}
extern "C" uint64_t pulsar_mamba2_upd_floats(const pulsar_mamba2_geom *g) {
    if (!g) return 0;
    const uint64_t f = (uint64_t)g->n_heads * g->head_dim + (uint64_t)g->n_groups * g->d_state + 2u * g->n_heads;
    return (f + 3u) / 4u * 4u;
}
extern "C" size_t pulsar_mamba2_scratch_bytes(const pulsar_mamba2_geom *g, uint32_t rows) {
    if (!g) return 0;
    return round16((size_t)rows * pulsar_mamba2_conv_channels(g) * 2u) +
           round16((size_t)rows * g->n_heads * g->head_dim * 2u) + 2u * round16((size_t)rows * g->n_heads * 4u);
}

extern "C" int pulsar_mamba2_forward(const pulsar_mamba2_layer *L, const pulsar_mamba2_call *c, cudaStream_t stream) {
    if (!L || !c) return refuse("no layer or call");
    const pulsar_mamba2_geom *g = &L->geom;
    if (const char *why = geom_refusal(g)) return refuse(why);
    if (!L->conv_w || !L->A_log || !L->dt_bias || !L->D || !L->norm_w)
        return refuse("a parameter is missing (conv_w, A_log, dt_bias, D and norm_w are required; conv_b may be NULL)");
    if (!aligned16(L->conv_w) || !aligned16(L->norm_w) || (L->conv_b && !aligned16(L->conv_b)))
        return refuse("conv_w, conv_b and norm_w must be 16-byte aligned");
    if (c->n_seqs == 0 || c->n_rows == 0 || c->max_seq_rows == 0 || c->max_seq_rows > c->n_rows ||
        c->n_seqs > c->n_rows || !c->seq_first || !c->seq_slot)
        return refuse("the call's rows: no sequences or rows, a longest run past the rows, or no row descriptors");
    const uint32_t H = g->n_heads, HP = H * P, CC = pulsar_mamba2_conv_channels(g), rows = c->n_rows;
    if (!c->z || !c->xbc || !c->dt || !c->out || !aligned16(c->z) || !aligned16(c->xbc) || !aligned16(c->out))
        return refuse("z, xbc, dt and out must be set; z, xbc and out 16-byte aligned");
    if (c->ld_z < HP || c->ld_z % 8 || c->ld_xbc < CC || c->ld_xbc % 8 || c->ld_out < HP || c->ld_out % 8 || c->ld_dt < H)
        return refuse("a row stride is short (z / out >= H*P, xbc >= CC, dt >= H) or not a multiple of 8 (z, xbc, out)");
    if (!c->conv_state || !c->ssm_state || !aligned16(c->conv_state) || !aligned16(c->ssm_state))
        return refuse("the conv and SSM state pools must be set and 16-byte aligned");
    if ((c->conv_capture == nullptr) != (c->ssm_capture == nullptr))
        return refuse("a verify captures both states (conv SNAPSHOT and SSM DELTA) or neither");
    if (c->ssm_capture && (!aligned16(c->ssm_capture) || !aligned16(c->conv_capture)))
        return refuse("the captures must be 16-byte aligned");
    if (!c->scratch || !aligned16(c->scratch) || c->scratch_bytes < pulsar_mamba2_scratch_bytes(g, rows))
        return refuse("scratch missing, misaligned or short of pulsar_mamba2_scratch_bytes");
    {
        uintptr_t ol, oh, il, ih;
        span(c->out, c->ld_out, rows, HP, &ol, &oh);
        const struct { const uint16_t *p; uint64_t ld, C; } in[3] = {{c->z, c->ld_z, HP}, {c->xbc, c->ld_xbc, CC}, {c->dt, c->ld_dt, H}};
        for (const auto &x : in) {
            span(x.p, x.ld, rows, x.C, &il, &ih);
            if (ol < ih && il < oh) return refuse("the output overlaps an input");
        }
    }

    uint8_t *sp = static_cast<uint8_t *>(c->scratch);
    uint16_t *conv_out = reinterpret_cast<uint16_t *>(sp);
    sp += round16((size_t)rows * CC * 2u);
    uint16_t *y = reinterpret_cast<uint16_t *>(sp);
    sp += round16((size_t)rows * HP * 2u);
    float *dt = reinterpret_cast<float *>(sp);
    sp += round16((size_t)rows * H * 4u);
    float *dA = reinterpret_cast<float *>(sp);

    /* 1. the convolution (E3) -- it refuses before launching anything, so a refusal here launches nothing */
    const pulsar_sconv_decl cd{CC, g->conv_kernel, 0, L->conv_w, L->conv_b, PULSAR_SCONV_ACT_SILU};
    const uint64_t rec = pulsar_mamba2_conv_state_elems(g);
    pulsar_sconv_stream cs{};
    cs.decl = &cd;
    cs.x = c->xbc; cs.ld_x = c->ld_xbc;
    cs.y = conv_out; cs.ld_y = CC;
    cs.state = c->conv_state; cs.state_ld = CC; cs.state_stride = rec;
    cs.capture = c->conv_capture; cs.capture_ld = CC; cs.capture_stride = rec;
    const pulsar_sconv_rows sr{c->n_seqs, rows, c->max_seq_rows, c->seq_first, c->seq_slot};
    const int rc = pulsar_sconv_forward(&cs, 1, &sr, stream);
    if (rc) return rc;

    /* 2. dt, dA -- a run longer than one tile only; a decode-width call stages them in the scan */
    const bool dt_launch = c->max_seq_rows > (uint32_t)TT;
    if (dt_launch) {
        const uint64_t n_dt = (uint64_t)rows * H;
        mamba2_dt_kernel<<<(unsigned)((n_dt + DT_THREADS - 1) / DT_THREADS), DT_THREADS, 0, stream>>>(
            c->dt, c->ld_dt, L->dt_bias, L->A_log, H, rows, dt, dA);
    }

    /* 3. the scan */
    scan_args a{};
    a.dt = dt_launch ? dt : nullptr;
    a.dA = dt_launch ? dA : nullptr;
    a.conv_out = conv_out; a.dt_raw = c->dt; a.ld_dt = c->ld_dt; a.dt_bias = L->dt_bias; a.A_log = L->A_log;
    a.D = L->D;
    a.seq_first = c->seq_first; a.seq_slot = c->seq_slot;
    a.ssm = c->ssm_state; a.cap = c->ssm_capture; a.y = y;
    a.H = H; a.G = g->n_groups; a.CC = CC; a.upd = (uint32_t)pulsar_mamba2_upd_floats(g);
    mamba2_scan_kernel<<<dim3(c->n_seqs, H, PS), NT, 0, stream>>>(a);

    /* 4. the gated norm */
    mamba2_norm_kernel<<<dim3(rows, HP / g->norm_group), g->norm_group / 8, 0, stream>>>(
        y, HP, c->z, c->ld_z, L->norm_w, g->norm_group, g->norm_eps, c->out, c->ld_out);

    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: mamba2: launch failed: %s\n", cudaGetErrorString(e));
        return -3;
    }
    return 0;
}

extern "C" int pulsar_mamba2_commit(const pulsar_mamba2_geom *g, float *ssm_state, uint32_t slot, const float *upd,
                                    uint32_t keep, cudaStream_t stream) {
    if (const char *why = geom_refusal(g)) return refuse(why);
    if (!ssm_state || !upd || !aligned16(ssm_state) || !aligned16(upd) || keep < 1)
        return refuse("commit: a null or misaligned pointer, or keep 0");
    mamba2_commit_kernel<<<dim3(g->n_heads, PS), NT, 0, stream>>>(
        ssm_state + (uint64_t)slot * pulsar_mamba2_ssm_state_floats(g), upd, keep, g->n_heads, g->n_groups,
        (uint32_t)pulsar_mamba2_upd_floats(g));
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "pulsar: mamba2: commit launch failed: %s\n", cudaGetErrorString(e));
        return -3;
    }
    return 0;
}
