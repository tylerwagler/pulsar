/* EXL3 MoE PREFILL GATE (L287): the routed experts' prompt-chunk arm (src/cuda/mmq/exl3_moe_prefill.cu),
 * one arm for every family, against an f32-weight reference.
 *
 * Model-free.  Random trellis words (any word is a valid tile), suh / svh in the checkpoint's distribution,
 * ragged expert loads (a Zipf-like router: experts with hundreds of assignments -- several CTA work items --
 * beside empty ones), and the activation in BOTH producer formats the arm reads: the DeepSeek family's E4M3
 * slot (per-32 ue8m0 plane, pulsar_mx_sfoff) and the Qwen family's bf16 rows.  The reference is built from
 * src/engine/exl3_trellis.h alone (the host tile decode, byte-exact to exllamav3) -- W_hat in f32, the
 * input x (* suh, Sylvester H128 / sqrt 128 when the projection rotates it) in double, a double-accumulated
 * GEMM on the device over sampled assignments:
 *
 *   1. shapes: DeepSeek V4 Vision-Exp (4096 -> 2048 x 256 experts, top-6), V4.1 (5120 -> 2304 x 384,
 *      top-6), Qwen3.8-Flash-Next (2560 -> 1280 fused gate_up / 640 -> 2560 down x 512, top-10); the
 *      gate (rotating) and the down (pre-rotated input) projection of each.
 *   2. rates: every k2 the arm instantiates (K = 2, 2.5, 3, 4, 5, 6) at 512 tokens.
 *   3. rows: 1, 2, 5, 16, 27, 64, 512, 4096, 8192 tokens at the production rates.
 *   4. chunk invariance: a prompt cut into chunks of 1 / 3 / 17 / 64 / 200 / ... tokens gives every
 *      (token, slot) output the SAME BITS as the prompt in one call.
 *   5. refusals: a rate without an instance, a short workspace, an E4M3 slot without its scales, M % 128.
 * Error per assignment row: rel Frobenius ||z - ref|| / ||ref|| and max |z - ref| / max |ref|; reported as the
 * max and mean over the sampled rows.  The operand is ONE fp16 plane per (row, 128-block) prescale: an
 * input the arm rotates rounds once to 11 significant bits (row error ~2^-12 / sqrt 3), an E4M3 value
 * that is not rotated is exact (f32-accumulation error only).
 *
 * usage: ./tests/exl3_moe_prefill_gate
 */
#include "cuda/mmq/exl3_moe_prefill.cuh"
#include "cuda/pulsar_cuda_mx.cuh"
#include "engine/exl3_trellis.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "EXL3-MOE-PREFILL FAIL: " __VA_ARGS__); \
                                       fprintf(stderr, "\n"); g_fail = 1; } } while (0)
#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "EXL3-MOE-PREFILL FAIL: %s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static uint32_t g_rng = 0x9E3779B9u;
static uint32_t rnd(void) { uint32_t x = g_rng; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return g_rng = x; }
static float rndf(void) { return (float)(rnd() >> 8) * (1.0f / 16777216.0f); }

/* host E4M3 (OCP): 1-4-3, bias 7, subnormals at exp 0, NaN = exp 15 mant 7 */
static float e4m3_to_f32(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 0) v = (float)m / 8.0f * exp2f(-6.0f);
    else v = (1.0f + (float)m / 8.0f) * exp2f((float)(e - 7));
    return s ? -v : v;
}
static uint8_t rnd_e4m3(void) { uint8_t b; do { b = (uint8_t)rnd(); } while ((b & 0x7fu) == 0x7fu); return b; }
static uint16_t rnd_suh(void) { return exl3_f32_to_f16((0.5f + rndf()) * 0.025f * ((rnd() & 1u) ? 1.0f : -1.0f)); }
static uint16_t rnd_svh(void) { return exl3_f32_to_f16((0.75f + 0.5f * rndf()) * ((rnd() & 1u) ? 1.0f : -1.0f)); }

static float g_mul1[65536];   /* exl3_mul1_decode as f32, every state */

/* A pool of distinct expert-projections at one (K, N, rate): the device slices, the table of E entries
 * (entry e reads slice e % P -- a real table's shape over a small pool) and W_hat of each slice in f32. */
struct pool {
    int K = 0, N = 0, k2 = 0, P = 0, E = 0;
    uint64_t stride = 0, trellis = 0;
    uint8_t *d_arena = nullptr;
    const void **d_table = nullptr;
    float *d_w = nullptr;            /* [P][K][N] */
    void release() { cudaFree(d_arena); cudaFree(d_table); cudaFree(d_w); }
};
static pool make_pool(int K, int N, int k2, int P, int E) {
    pool p;
    p.K = K; p.N = N; p.k2 = k2; p.P = P; p.E = E;
    uint64_t scales = 0;
    if (!exl3_expert_layout(K, N, k2, &p.trellis, &scales, &p.stride)) { fprintf(stderr, "layout refused\n"); exit(1); }
    std::vector<uint8_t> bytes(p.stride * P);
    std::vector<float> w((size_t)P * K * N);
    const int words = exl3_words_per_tile(k2);
    for (int b = 0; b < P; b++) {
        uint8_t *s = bytes.data() + b * p.stride;
        uint16_t *t = (uint16_t *)s;
        for (uint64_t i = 0; i < p.trellis / 2; i++) t[i] = (uint16_t)rnd();
        uint16_t *sc = (uint16_t *)(s + p.trellis);
        for (int i = 0; i < K; i++) sc[i] = rnd_suh();
        for (int i = 0; i < N; i++) sc[K + i] = rnd_svh();
        float *wb = w.data() + (size_t)b * K * N;
        for (int kt = 0; kt < K / 16; kt++)
            for (int nt = 0; nt < N / 16; nt++) {
                const uint16_t *tile = t + ((size_t)kt * (N / 16) + nt) * words;
                for (int q = 0; q < EXL3_TILE_WEIGHTS; q++) {
                    int r, c;
                    exl3_tile_position(q, &r, &c);
                    wb[(size_t)(kt * 16 + r) * N + nt * 16 + c] = g_mul1[exl3_tile_state(tile, k2, q)];
                }
            }
    }
    CUDA_OK(cudaMalloc(&p.d_arena, bytes.size()));
    CUDA_OK(cudaMemcpy(p.d_arena, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&p.d_w, w.size() * sizeof(float)));
    CUDA_OK(cudaMemcpy(p.d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<const void *> tab(2 * (size_t)E);
    for (int e = 0; e < E; e++) {
        const uint8_t *s = p.d_arena + (e % P) * p.stride;
        tab[2 * e] = s;
        tab[2 * e + 1] = s + p.trellis;
    }
    CUDA_OK(cudaMalloc(&p.d_table, tab.size() * sizeof(void *)));
    CUDA_OK(cudaMemcpy(p.d_table, tab.data(), tab.size() * sizeof(void *), cudaMemcpyHostToDevice));
    return p;
}

/* One routing: n_tok tokens x topk distinct experts each, Zipf-like (expert ~ E u^3: a handful of experts
 * take hundreds of assignments, the tail is empty), with the per-pair choices as the router emits them. */
static std::vector<int32_t> make_selection(int n_tok, int topk, int E) {
    std::vector<int32_t> sel((size_t)n_tok * topk);
    for (int t = 0; t < n_tok; t++)
        for (int s = 0; s < topk; s++) {
            int e;
            bool dup;
            do {
                const float u = rndf();
                e = (rnd() & 3u) ? (int)((float)E * u * u * u) : (int)(rnd() % (uint32_t)E);
                if (e >= E) e = E - 1;
                dup = false;
                for (int q = 0; q < s; q++) dup |= sel[(size_t)t * topk + q] == e;
            } while (dup);
            sel[(size_t)t * topk + s] = e;
        }
    return sel;
}

/* The expert-sorted schedule mm_ids_helper lays out for pairs [p0, p1): within an expert, pair order.
 * ids_src = the activation row (token for gate / up, pair for a down whose input is per pair). */
struct sched {
    std::vector<int32_t> ids_dst, ids_src, bounds;
};
static sched make_sched(const std::vector<int32_t> &sel, int topk, int E, int p0, int p1, bool per_pair) {
    sched s;
    s.bounds.assign(E + 1, 0);
    std::vector<std::vector<int32_t>> by(E);
    for (int p = p0; p < p1; p++) by[sel[p]].push_back(p);
    for (int e = 0; e < E; e++) {
        s.bounds[e] = (int32_t)s.ids_dst.size();
        for (int p : by[e]) {
            s.ids_dst.push_back(p - p0);
            s.ids_src.push_back(per_pair ? p - p0 : p / topk - p0 / topk);
        }
    }
    s.bounds[E] = (int32_t)s.ids_dst.size();
    return s;
}

/* The activation: rows x K in the producer's format, plus its exact value as f32 (E4M3 x 2^e and bf16 are
 * both exact in f32). */
struct act_rows {
    int format = 0, rows = 0, K = 0, kbp = 0;
    std::vector<uint8_t> q;          /* E4M3 [rows][K] or bf16 bytes [rows][K][2] */
    std::vector<uint8_t> sf;         /* E4M3: the ue8m0 slab (pulsar_mx_sfoff) */
    std::vector<float> v;            /* [rows][K] exact value */
};
static act_rows make_act(int format, int rows, int K) {
    act_rows a;
    a.format = format; a.rows = rows; a.K = K;
    a.v.resize((size_t)rows * K);
    if (format == EXL3_MOE_ACT_E4M3_SLOT) {
        a.kbp = pulsar_mx_kbp(K);
        a.q.resize((size_t)rows * K);
        a.sf.assign(pulsar_mx_sf_slab_bytes(rows, a.kbp), 0);
        for (int r = 0; r < rows; r++)
            for (int g = 0; g < K / 32; g++) {
                const uint8_t sb = (uint8_t)(118 + rnd() % 12);
                a.sf[pulsar_mx_sfoff(r, g, a.kbp)] = sb;
                for (int j = 0; j < 32; j++) {
                    const size_t i = (size_t)r * K + g * 32 + j;
                    a.q[i] = rnd_e4m3();
                    a.v[i] = ldexpf(e4m3_to_f32(a.q[i]), (int)sb - 127);
                }
            }
    } else {
        a.q.resize((size_t)rows * K * 2);
        for (size_t i = 0; i < (size_t)rows * K; i++) {
            const float x = (rndf() * 2.0f - 1.0f) * exp2f((float)(int)(rnd() % 9) - 4.0f);
            const __nv_bfloat16 b = __float2bfloat16(x);
            memcpy(&a.q[2 * i], &b, 2);
            a.v[i] = __bfloat162float(b);
        }
    }
    return a;
}

/* Reference rotation: xr[s][k] = (H128 (x * suh)) / sqrt 128 per block, or x, in double. */
__global__ void ref_input_kernel(const float *__restrict__ xv, const int32_t *__restrict__ rows,
                                 const int32_t *__restrict__ blobs, const void *const *__restrict__ table,
                                 int K, bool rotate, double *__restrict__ xr) {
    const int s = blockIdx.x, blk = blockIdx.y, i = threadIdx.x;
    __shared__ double xs[128];
    const int k = blk * 128 + i;
    double x = (double)xv[(size_t)rows[s] * K + k];
    if (rotate) x *= (double)__half2float(reinterpret_cast<const __half *>(table[2 * (size_t)blobs[s] + 1])[k]);
    xs[i] = x;
    __syncthreads();
    double y = x;
    if (rotate) {
        y = 0.0;
        for (int j = 0; j < 128; j++) y += (__popc(i & j) & 1) ? -xs[j] : xs[j];
        y *= 0.08838834764831845;   /* 1 / sqrt 128 */
    }
    xr[(size_t)s * K + k] = y;
}
/* Reference GEMM: ref[s][n] = sum_k xr[s][k] W_hat[k][n], double accumulation. */
__global__ void ref_gemm_kernel(const double *__restrict__ xr, const float *__restrict__ w, const int32_t *__restrict__ wblob,
                                int K, int N, double *__restrict__ ref) {
    const int s = blockIdx.x, n = blockIdx.y * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float *wb = w + (size_t)wblob[s] * K * N;
    const double *x = xr + (size_t)s * K;
    double acc = 0.0;
    for (int k = 0; k < K; k++) acc += x[k] * (double)wb[(size_t)k * N + n];
    ref[(size_t)s * N + n] = acc;
}

/* The device side of one activation. */
struct dev_act {
    void *x = nullptr, *sf = nullptr;
    float *v = nullptr;
    exl3_moe_act desc{};
    void release() { cudaFree(x); cudaFree(sf); cudaFree(v); }
};
static dev_act upload_act(const act_rows &a) {
    dev_act d;
    CUDA_OK(cudaMalloc(&d.x, a.q.size()));
    CUDA_OK(cudaMemcpy(d.x, a.q.data(), a.q.size(), cudaMemcpyHostToDevice));
    if (!a.sf.empty()) {
        CUDA_OK(cudaMalloc(&d.sf, a.sf.size()));
        CUDA_OK(cudaMemcpy(d.sf, a.sf.data(), a.sf.size(), cudaMemcpyHostToDevice));
    }
    CUDA_OK(cudaMalloc(&d.v, a.v.size() * sizeof(float)));
    CUDA_OK(cudaMemcpy(d.v, a.v.data(), a.v.size() * sizeof(float), cudaMemcpyHostToDevice));
    d.desc = exl3_moe_act{a.format, d.x, d.sf, a.kbp};
    return d;
}

/* Run the arm over pairs [p0, p1) of sel; out_host gets [p1 - p0][N] in pair order. */
static int run_arm(const pool &pl, bool rotate, const exl3_moe_act &act, const std::vector<int32_t> &sel, int topk,
                   int p0, int p1, bool per_pair, std::vector<float> &out_host, sched *keep = nullptr,
                   float *ms = nullptr) {
    sched s = make_sched(sel, topk, pl.E, p0, p1, per_pair);
    const int64_t n_assign = p1 - p0;
    int32_t *d_dst, *d_src, *d_b;
    float *d_out;
    void *ws;
    const size_t wsb = exl3_moe_prefill_ws_bytes(n_assign, pl.K, pl.E);
    CUDA_OK(cudaMalloc(&d_dst, n_assign * 4));
    CUDA_OK(cudaMalloc(&d_src, n_assign * 4));
    CUDA_OK(cudaMalloc(&d_b, (pl.E + 1) * 4));
    CUDA_OK(cudaMalloc(&d_out, (size_t)n_assign * pl.N * 4));
    CUDA_OK(cudaMalloc(&ws, wsb));
    CUDA_OK(cudaMemcpy(d_dst, s.ids_dst.data(), n_assign * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_src, s.ids_src.data(), n_assign * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_b, s.bounds.data(), (pl.E + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_out, 0xff, (size_t)n_assign * pl.N * 4));   /* NaN: an unwritten output cannot pass */
    int rc = exl3_moe_prefill_launch(pl.d_table, pl.k2, rotate, act, d_dst, d_src, d_b, d_out, pl.N, pl.K,
                                     n_assign, pl.E, ws, wsb, 0);
    CUDA_OK(cudaDeviceSynchronize());
    if (rc == 0 && ms) {
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < 5; i++)
            rc |= exl3_moe_prefill_launch(pl.d_table, pl.k2, rotate, act, d_dst, d_src, d_b, d_out, pl.N, pl.K,
                                          n_assign, pl.E, ws, wsb, 0);
        cudaEventRecord(b);
        CUDA_OK(cudaEventSynchronize(b));
        cudaEventElapsedTime(ms, a, b);
        *ms /= 5.0f;
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    out_host.resize((size_t)n_assign * pl.N);
    CUDA_OK(cudaMemcpy(out_host.data(), d_out, out_host.size() * 4, cudaMemcpyDeviceToHost));
    cudaFree(d_dst); cudaFree(d_src); cudaFree(d_b); cudaFree(d_out); cudaFree(ws);
    if (keep) *keep = s;
    return rc;
}

struct errs { double frob_max = 0, frob_mean = 0, elem_max = 0; int rows = 0; };

/* Grade sampled pairs of one whole-prompt run against the reference. */
static errs grade(const pool &pl, bool rotate, const dev_act &da, const std::vector<int32_t> &sel, int topk,
                  int n_pairs, bool per_pair, const std::vector<float> &out, int max_samples) {
    std::vector<int32_t> pairs;
    if (n_pairs <= max_samples) for (int p = 0; p < n_pairs; p++) pairs.push_back(p);
    else for (int i = 0; i < max_samples; i++) pairs.push_back((int)(rnd() % (uint32_t)n_pairs));
    const int S = (int)pairs.size();
    std::vector<int32_t> rows(S), blobs(S), eidx(S);
    for (int i = 0; i < S; i++) {
        rows[i] = per_pair ? pairs[i] : pairs[i] / topk;
        eidx[i] = sel[pairs[i]];
        blobs[i] = eidx[i] % pl.P;
    }
    int32_t *d_rows, *d_e, *d_blob;
    double *d_xr, *d_ref;
    CUDA_OK(cudaMalloc(&d_rows, S * 4)); CUDA_OK(cudaMalloc(&d_e, S * 4)); CUDA_OK(cudaMalloc(&d_blob, S * 4));
    CUDA_OK(cudaMalloc(&d_xr, (size_t)S * pl.K * 8)); CUDA_OK(cudaMalloc(&d_ref, (size_t)S * pl.N * 8));
    CUDA_OK(cudaMemcpy(d_rows, rows.data(), S * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_e, eidx.data(), S * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_blob, blobs.data(), S * 4, cudaMemcpyHostToDevice));
    ref_input_kernel<<<dim3(S, pl.K / 128), 128>>>(da.v, d_rows, d_e, pl.d_table, pl.K, rotate, d_xr);
    ref_gemm_kernel<<<dim3(S, (pl.N + 127) / 128), 128>>>(d_xr, pl.d_w, d_blob, pl.K, pl.N, d_ref);
    CUDA_OK(cudaGetLastError());
    std::vector<double> ref((size_t)S * pl.N);
    CUDA_OK(cudaMemcpy(ref.data(), d_ref, ref.size() * 8, cudaMemcpyDeviceToHost));
    cudaFree(d_rows); cudaFree(d_e); cudaFree(d_blob); cudaFree(d_xr); cudaFree(d_ref);
    errs e;
    e.rows = S;
    for (int i = 0; i < S; i++) {
        double num = 0, den = 0, emax = 0, rmax = 0;
        for (int n = 0; n < pl.N; n++) {
            const double r = ref[(size_t)i * pl.N + n], g = out[(size_t)pairs[i] * pl.N + n];
            const double d = std::isfinite(g) ? g - r : INFINITY;
            num += d * d; den += r * r;
            emax = fmax(emax, fabs(d)); rmax = fmax(rmax, fabs(r));
        }
        const double f = sqrt(num / den);
        e.frob_max = fmax(e.frob_max, f);
        e.frob_mean += f / S;
        e.elem_max = fmax(e.elem_max, emax / rmax);
    }
    return e;
}

struct shape { const char *what; int H, mid, E, topk; bool fused; int k2_gate, k2_down, fmt; };

/* The bound per (format, rotation): a rotated operand rounds once to 11 significant bits per element
 * (rel Frobenius ~1.4e-4 per row); an unrotated E4M3 or bf16 value is exact in the plane (f32 accumulate
 * only).  Measured (L287): rotated worst 2.35e-4 (bound 1e-3), unrotated 9.6e-7 (bound 5e-6); a format
 * error reads ~1e-1. */
static double frob_bound(bool rotate) { return rotate ? 1e-3 : 5e-6; }

static void section_case(const shape &sh, bool gate, int k2, int n_tok, int fmt, int max_samples, bool chunks,
                         int P, bool timed) {
    const int K = gate ? sh.H : sh.mid;
    const int N = gate ? (sh.fused ? 2 * sh.mid : sh.mid) : sh.H;
    const bool rotate = gate;
    const bool per_pair = !gate;
    pool pl = make_pool(K, N, k2, P, sh.E);
    const std::vector<int32_t> sel = make_selection(n_tok, sh.topk, sh.E);
    const int n_pairs = n_tok * sh.topk;
    act_rows a = make_act(fmt, per_pair ? n_pairs : n_tok, K);
    dev_act da = upload_act(a);
    std::vector<float> whole;
    sched s;
    float ms = 0.0f;
    int rc = run_arm(pl, rotate, da.desc, sel, sh.topk, 0, n_pairs, per_pair, whole, &s, timed ? &ms : nullptr);
    CHECK(rc == 0, "%s %s K=%g T=%d: launch rc=%d", sh.what, gate ? "gate" : "down", k2 / 2.0, n_tok, rc);
    int maxload = 0, empty = 0;
    for (int e = 0; e < sh.E; e++) {
        const int c = s.bounds[e + 1] - s.bounds[e];
        maxload = c > maxload ? c : maxload;
        empty += c == 0;
    }
    const errs er = grade(pl, rotate, da, sel, sh.topk, n_pairs, per_pair, whole, max_samples);
    const double bound = frob_bound(rotate);
    CHECK(er.frob_max < bound, "%s %s K=%g T=%d: worst row rel Frobenius %.3e >= %.1e", sh.what,
          gate ? "gate" : "down", k2 / 2.0, n_tok, er.frob_max, bound);
    printf("  %-10s %-4s %-5s K=%-3g T=%-5d %5d x %-5d | load max %4d, %3d of %d experts empty | rows %4d: "
           "rel Frob max %.2e mean %.2e, max|err|/max|ref| %.2e",
           sh.what, gate ? "gate" : "down", fmt == EXL3_MOE_ACT_E4M3_SLOT ? "e4m3" : "bf16", k2 / 2.0, n_tok, K, N,
           maxload, empty, sh.E, er.rows, er.frob_max, er.frob_mean, er.elem_max);
    if (timed) printf(" | %.3f ms, %.1f TFLOP/s", ms, 2.0 * n_pairs * (double)K * N / (ms * 1e-3) / 1e12);
    printf("\n");
    if (chunks) {
        /* the same prompt in chunks: every (token, slot) output keeps its bits */
        static const int cuts[] = {1, 3, 17, 64, 200, 31, 2, 129};
        std::vector<float> got((size_t)n_pairs * N);
        int t0 = 0, ci = 0, nchunks = 0;
        while (t0 < n_tok) {
            const int len = cuts[ci++ % 8];
            const int t1 = t0 + len < n_tok ? t0 + len : n_tok;
            std::vector<float> part;
            /* the chunk's own activation rows: a chunk is its own call, its rows start at 0 */
            act_rows ac;
            ac.format = a.format; ac.K = K;
            const int r0 = per_pair ? t0 * sh.topk : t0, r1 = per_pair ? t1 * sh.topk : t1;
            ac.rows = r1 - r0;
            ac.v.assign(a.v.begin() + (size_t)r0 * K, a.v.begin() + (size_t)r1 * K);
            if (fmt == EXL3_MOE_ACT_E4M3_SLOT) {
                ac.kbp = a.kbp;
                ac.q.assign(a.q.begin() + (size_t)r0 * K, a.q.begin() + (size_t)r1 * K);
                ac.sf.assign(pulsar_mx_sf_slab_bytes(ac.rows, ac.kbp), 0);
                for (int r = 0; r < ac.rows; r++)
                    for (int g = 0; g < K / 32; g++)
                        ac.sf[pulsar_mx_sfoff(r, g, ac.kbp)] = a.sf[pulsar_mx_sfoff(r0 + r, g, a.kbp)];
            } else {
                ac.q.assign(a.q.begin() + (size_t)r0 * K * 2, a.q.begin() + (size_t)r1 * K * 2);
            }
            dev_act dc = upload_act(ac);
            rc = run_arm(pl, rotate, dc.desc, sel, sh.topk, t0 * sh.topk, t1 * sh.topk, per_pair, part);
            dc.release();
            CHECK(rc == 0, "%s chunk [%d, %d) rc=%d", sh.what, t0, t1, rc);
            memcpy(&got[(size_t)t0 * sh.topk * N], part.data(), part.size() * 4);
            t0 = t1;
            nchunks++;
        }
        size_t diff = 0;
        for (size_t i = 0; i < got.size(); i++) diff += memcmp(&got[i], &whole[i], 4) != 0;
        CHECK(diff == 0, "%s %s K=%g T=%d: %zu of %zu outputs differ between %d chunks and one call", sh.what,
              gate ? "gate" : "down", k2 / 2.0, n_tok, diff, got.size(), nchunks);
        printf("    chunk invariance: %d chunks (1..200 tokens) vs one call: %s (%zu outputs)\n", nchunks,
               diff ? "DIFFER" : "bit-identical", got.size());
    }
    da.release();
    pl.release();
}

static void section_refusals(void) {
    printf("5. refusals\n");
    pool pl = make_pool(256, 256, 6, 1, 4);
    act_rows a = make_act(EXL3_MOE_ACT_E4M3_SLOT, 8, 256);
    dev_act da = upload_act(a);
    const std::vector<int32_t> sel = make_selection(8, 2, 4);
    sched s = make_sched(sel, 2, 4, 0, 16, false);
    int32_t *d_dst, *d_src, *d_b;
    float *d_out;
    void *ws;
    const size_t wsb = exl3_moe_prefill_ws_bytes(16, 256, 4);
    CUDA_OK(cudaMalloc(&d_dst, 64)); CUDA_OK(cudaMalloc(&d_src, 64)); CUDA_OK(cudaMalloc(&d_b, 20));
    CUDA_OK(cudaMalloc(&d_out, 16 * 256 * 4)); CUDA_OK(cudaMalloc(&ws, wsb));
    CUDA_OK(cudaMemcpy(d_dst, s.ids_dst.data(), 64, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_src, s.ids_src.data(), 64, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_b, s.bounds.data(), 20, cudaMemcpyHostToDevice));
    const int ok = exl3_moe_prefill_launch(pl.d_table, 6, true, da.desc, d_dst, d_src, d_b, d_out, 256, 256, 16, 4, ws, wsb, 0);
    CHECK(ok == 0, "the well-formed call refused (rc=%d)", ok);
    const int r_rate = exl3_moe_prefill_launch(pl.d_table, 7, true, da.desc, d_dst, d_src, d_b, d_out, 256, 256, 16, 4, ws, wsb, 0);
    const int r_ws = exl3_moe_prefill_launch(pl.d_table, 6, true, da.desc, d_dst, d_src, d_b, d_out, 256, 256, 16, 4, ws, wsb - 256, 0);
    exl3_moe_act nosf = da.desc;
    nosf.sf = nullptr;
    const int r_sf = exl3_moe_prefill_launch(pl.d_table, 6, true, nosf, d_dst, d_src, d_b, d_out, 256, 256, 16, 4, ws, wsb, 0);
    const int r_m = exl3_moe_prefill_launch(pl.d_table, 6, true, da.desc, d_dst, d_src, d_b, d_out, 192, 256, 16, 4, ws, wsb, 0);
    CHECK(r_rate != 0 && r_ws != 0 && r_sf != 0 && r_m != 0, "refusals: rate %d, workspace %d, E4M3 without scales %d, "
          "M %% 128 %d (all must be nonzero)", r_rate, r_ws, r_sf, r_m);
    printf("  k2=7 rc=%d, workspace 256 B short rc=%d, E4M3 slot without its scales rc=%d, M=192 rc=%d\n",
           r_rate, r_ws, r_sf, r_m);
    CUDA_OK(cudaDeviceSynchronize());
    cudaFree(d_dst); cudaFree(d_src); cudaFree(d_b); cudaFree(d_out); cudaFree(ws);
    da.release();
    pl.release();
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    for (uint32_t s = 0; s < 65536; s++) g_mul1[s] = exl3_f16_to_f32(exl3_mul1_decode(s));
    const shape v4  = {"V4-vexp", 4096, 2048, 256, 6, false, 6, 6, EXL3_MOE_ACT_E4M3_SLOT};
    const shape v41 = {"V4.1", 5120, 2304, 384, 6, false, 6, 6, EXL3_MOE_ACT_E4M3_SLOT};
    const shape qw  = {"Qwen3.8FN", 2560, 640, 512, 10, true, 8, 10, EXL3_MOE_ACT_BF16_ROWS};
    printf("exl3-moe-prefill-gate: the EXL3 routed prefill arm vs the f32-weight reference\n");
    printf("1-2. every rate the arm instantiates, 512 tokens, both projections, each family's format\n");
    static const int k2s[] = {4, 5, 6, 8, 10, 12};
    for (const shape *sh : {&v4, &v41, &qw})
        for (int k2 : k2s)
            for (int gate = 1; gate >= 0; gate--)
                section_case(*sh, gate, k2, 512, sh->fmt, 256, false, 2, false);
    printf("   the other producer format on the same shapes (the arm reads either)\n");
    section_case(v4, true, 6, 512, EXL3_MOE_ACT_BF16_ROWS, 256, false, 2, false);
    section_case(v4, false, 6, 512, EXL3_MOE_ACT_BF16_ROWS, 256, false, 2, false);
    section_case(qw, true, 8, 512, EXL3_MOE_ACT_E4M3_SLOT, 256, false, 2, false);
    section_case(qw, false, 10, 512, EXL3_MOE_ACT_E4M3_SLOT, 256, false, 2, false);
    printf("3-4. rows 1..8192 at the production rates (DeepSeek K3, Qwen K4 / K5), chunk invariance up to 1024\n");
    static const int toks[] = {1, 2, 5, 16, 27, 64, 512, 1024, 4096, 8192};
    for (const shape *sh : {&v4, &v41, &qw})
        for (int t : toks)
            for (int gate = 1; gate >= 0; gate--)
                section_case(*sh, gate, gate ? sh->k2_gate : sh->k2_down, t, sh->fmt, t >= 4096 ? 192 : 256,
                             t > 1 && t <= 1024, 4, t >= 512);
    section_refusals();
    printf(g_fail ? "EXL3-MOE-PREFILL GATE FAIL\n" : "EXL3-MOE-PREFILL GATE PASS\n");
    return g_fail;
}
