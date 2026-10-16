/* EXL3 MoE PREFILL GATE (L287, task #46): the routed experts' tensor-core arms (src/cuda/mmq/exl3_moe_prefill.cu)
 * -- the prompt-chunk GEMM and the decode GEMM -- one arm each for every family, against an f32-weight reference.
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
 * and the DECODE arm (task #46; exl3_moe_decode_launch, its split partials summed in split order as the fold
 * and the sum add them):
 *   6-7. every rate x shape x projection x format at 16 tokens, against the same reference and bounds;
 *   8. rows 1, 2, 3, 5, 8, 16, 40 at the production rates (incl. turboderp's K3 Qwen pair), and BATCH
 *      invariance: the rows cut into calls of 1 / 3 / 17 / ... tokens give every output the SAME BITS as one
 *      call -- the arm's promise (its split count is a function of K alone; a row's m16 tile, k order and
 *      warp / split order do not depend on its batchmates);
 *   9. a gate + up pair in one launch == the two projections launched alone, bit for bit; the refusals.
 *  10. (L295 F7) the pair at K5 at Inkling's routed shape (4096 -> 2048 -> 4096, 256 experts, top-6, bf16 rows):
 *      prompt rows 1..4096 and decode rows 1..40 of both projections, and the pair launch, as above.
 *  11. (L297) the decode arm at a verify's widths on Nemotron's up-only shape (1024 -> 2688 -> 1024, 512 experts,
 *      top-22, K4): rows 1..40, every row == the same row through the 1-row path (each row its own call), and the
 *      mixed cuts; Qwen's shape every row alone at 8 and 16 rows.
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
#include <algorithm>
#include <thread>
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
    for (int b = 0; b < P; b++) {   /* the random words and scales: one stream, in order */
        uint8_t *s = bytes.data() + b * p.stride;
        uint16_t *t = (uint16_t *)s;
        for (uint64_t i = 0; i < p.trellis / 2; i++) t[i] = (uint16_t)rnd();
        uint16_t *sc = (uint16_t *)(s + p.trellis);
        for (int i = 0; i < K; i++) sc[i] = rnd_suh();
        for (int i = 0; i < N; i++) sc[K + i] = rnd_svh();
    }
    /* W_hat: a pure function of the words, decoded on every core (2026-10-14: this host decode was ~all of the
     * gate's 114 s, single-threaded; each thread owns whole 16-row tile bands, so the bytes are the same) */
    const int bands = P * (K / 16);
    const unsigned nt_threads = std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
    std::vector<std::thread> th;
    for (unsigned w0 = 0; w0 < nt_threads; w0++)
        th.emplace_back([&, w0]() {
            for (int band = (int)w0; band < bands; band += (int)nt_threads) {
                const int b = band / (K / 16), kt = band % (K / 16);
                const uint16_t *t = (const uint16_t *)(bytes.data() + b * p.stride);
                float *wb = w.data() + (size_t)b * K * N;
                for (int nt = 0; nt < N / 16; nt++) {
                    const uint16_t *tile = t + ((size_t)kt * (N / 16) + nt) * words;
                    for (int q = 0; q < EXL3_TILE_WEIGHTS; q++) {
                        int r, c;
                        exl3_tile_position(q, &r, &c);
                        wb[(size_t)(kt * 16 + r) * N + nt * 16 + c] = g_mul1[exl3_tile_state(tile, k2, q)];
                    }
                }
            }
        });
    for (std::thread &x : th) x.join();
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

/* The arm under test: false = the prompt-chunk GEMM, true = the decode GEMM (task #46). */
static bool g_decode = false;
/* L297: the invariance check cuts the rows into calls of ONE token each (every row through the 1-row path) instead of
 * the mixed 1 / 3 / 17 / ... cuts. */
static bool g_cut_one = false;

/* One launch of the arm under test; for the decode arm `out` is unused and the split partials are summed in
 * split order (the fold's / sum's order) into *dec_out [n_assign][N] f32 on the host. */
static int launch_arm(const pool &pl, bool rotate, const exl3_moe_act &act, const int32_t *d_dst, const int32_t *d_src,
                      const int32_t *d_b, float *d_out, int64_t n_assign, void *ws, size_t wsb,
                      std::vector<float> *dec_out) {
    if (!g_decode)
        return exl3_moe_prefill_launch(pl.d_table, pl.k2, rotate, act, d_dst, d_src, d_b, d_out, pl.N, pl.K, n_assign,
                                       pl.E, ws, wsb, 0);
    exl3_moe_partials zp{};
    const int rc = exl3_moe_decode_launch(pl.d_table, nullptr, pl.k2, rotate, act, d_dst, d_src, d_b, pl.N, pl.K,
                                          n_assign, pl.E, ws, wsb, &zp, 0);
    if (rc || !dec_out) return rc;
    std::vector<float> part((size_t)zp.splits * zp.split_stride);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(part.data(), zp.z[0], part.size() * 4, cudaMemcpyDeviceToHost));
    dec_out->assign(part.begin(), part.begin() + zp.split_stride);
    for (int sp = 1; sp < zp.splits; sp++)
        for (int64_t i = 0; i < zp.split_stride; i++) (*dec_out)[i] += part[(size_t)sp * zp.split_stride + i];
    return 0;
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
    const size_t wsb = g_decode ? exl3_moe_decode_ws_bytes(1, n_assign, pl.K, pl.N)
                                : exl3_moe_prefill_ws_bytes(n_assign, pl.K, pl.E);
    CUDA_OK(cudaMalloc(&d_dst, n_assign * 4));
    CUDA_OK(cudaMalloc(&d_src, n_assign * 4));
    CUDA_OK(cudaMalloc(&d_b, (pl.E + 1) * 4));
    CUDA_OK(cudaMalloc(&d_out, (size_t)n_assign * pl.N * 4));
    CUDA_OK(cudaMalloc(&ws, wsb));
    CUDA_OK(cudaMemcpy(d_dst, s.ids_dst.data(), n_assign * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_src, s.ids_src.data(), n_assign * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_b, s.bounds.data(), (pl.E + 1) * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_out, 0xff, (size_t)n_assign * pl.N * 4));   /* NaN: an unwritten output cannot pass */
    CUDA_OK(cudaMemset(ws, 0xff, wsb));                              /* (the decode arm's partials live here) */
    std::vector<float> dec;
    int rc = launch_arm(pl, rotate, act, d_dst, d_src, d_b, d_out, n_assign, ws, wsb, &dec);
    CUDA_OK(cudaDeviceSynchronize());
    if (rc == 0 && ms) {
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < 5; i++) rc |= launch_arm(pl, rotate, act, d_dst, d_src, d_b, d_out, n_assign, ws, wsb, nullptr);
        cudaEventRecord(b);
        CUDA_OK(cudaEventSynchronize(b));
        cudaEventElapsedTime(ms, a, b);
        *ms /= 5.0f;
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    out_host.resize((size_t)n_assign * pl.N);
    if (g_decode && rc == 0) out_host = dec;
    else CUDA_OK(cudaMemcpy(out_host.data(), d_out, out_host.size() * 4, cudaMemcpyDeviceToHost));
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
    CHECK(rc == 0, "%s %s %s K=%g T=%d: launch rc=%d", g_decode ? "decode" : "prefill", sh.what, gate ? "gate" : "down",
          k2 / 2.0, n_tok, rc);
    int maxload = 0, empty = 0;
    for (int e = 0; e < sh.E; e++) {
        const int c = s.bounds[e + 1] - s.bounds[e];
        maxload = c > maxload ? c : maxload;
        empty += c == 0;
    }
    const errs er = grade(pl, rotate, da, sel, sh.topk, n_pairs, per_pair, whole, max_samples);
    const double bound = frob_bound(rotate);
    CHECK(er.frob_max < bound, "%s %s %s K=%g T=%d: worst row rel Frobenius %.3e >= %.1e",
          g_decode ? "decode" : "prefill", sh.what,
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
            const int len = g_cut_one ? 1 : cuts[ci++ % 8];
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
        CHECK(diff == 0, "%s %s %s K=%g T=%d: %zu of %zu outputs differ between %d calls and one call",
              g_decode ? "decode" : "prefill", sh.what,
              gate ? "gate" : "down", k2 / 2.0, n_tok, diff, got.size(), nchunks);
        printf("    %s invariance: %d calls (%s) vs one call: %s (%zu outputs)\n", g_decode ? "batch" : "chunk",
               nchunks, g_cut_one ? "1 token each" : "1..200 tokens", diff ? "DIFFER" : "bit-identical", got.size());
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

/* 9. the decode arm's pair launch (two slices, one PREP + one GEMM launch) gives each projection exactly the bits
 * of that projection launched alone; and its refusals. */
static void section_decode_pair(const char *what = "V4", int k2 = 6, int fmt = EXL3_MOE_ACT_E4M3_SLOT,
                                bool refusals = true) {
    if (refusals) printf("9. decode pair launch == two single launches; decode refusals\n");
    const int K = 4096, N = 2048, E = 8, topk = 6, n_tok = 7;
    pool g = make_pool(K, N, k2, 2, E), u = make_pool(K, N, k2, 2, E);
    act_rows a = make_act(fmt, n_tok, K);
    dev_act da = upload_act(a);
    const std::vector<int32_t> sel = make_selection(n_tok, topk, E);
    sched s = make_sched(sel, topk, E, 0, n_tok * topk, false);
    const int64_t n = n_tok * topk;
    int32_t *d_dst, *d_src, *d_b;
    CUDA_OK(cudaMalloc(&d_dst, n * 4)); CUDA_OK(cudaMalloc(&d_src, n * 4)); CUDA_OK(cudaMalloc(&d_b, (E + 1) * 4));
    CUDA_OK(cudaMemcpy(d_dst, s.ids_dst.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_src, s.ids_src.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_b, s.bounds.data(), (E + 1) * 4, cudaMemcpyHostToDevice));
    const size_t w2 = exl3_moe_decode_ws_bytes(2, n, K, N), w1 = exl3_moe_decode_ws_bytes(1, n, K, N);
    void *ws;
    CUDA_OK(cudaMalloc(&ws, w2));
    auto read = [&](const exl3_moe_partials &zp, int j) {
        std::vector<float> v((size_t)zp.splits * zp.split_stride);
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaMemcpy(v.data(), zp.z[j], v.size() * 4, cudaMemcpyDeviceToHost));
        return v;
    };
    exl3_moe_partials zp{}, z1{};
    int rc = exl3_moe_decode_launch(g.d_table, u.d_table, k2, true, da.desc, d_dst, d_src, d_b, N, K, n, E, ws, w2, &zp, 0);
    CHECK(rc == 0 && zp.z[1] && zp.splits >= 1, "%s pair launch rc=%d", what, rc);
    const std::vector<float> pg = read(zp, 0), pu = read(zp, 1);
    rc = exl3_moe_decode_launch(g.d_table, nullptr, k2, true, da.desc, d_dst, d_src, d_b, N, K, n, E, ws, w1, &z1, 0);
    const std::vector<float> sg = read(z1, 0);
    rc |= exl3_moe_decode_launch(u.d_table, nullptr, k2, true, da.desc, d_dst, d_src, d_b, N, K, n, E, ws, w1, &z1, 0);
    const std::vector<float> su = read(z1, 0);
    CHECK(rc == 0 && pg.size() == sg.size() && pu.size() == su.size(), "%s single launches rc=%d", what, rc);
    size_t diff = 0;
    for (size_t i = 0; i < pg.size() && i < sg.size(); i++) diff += memcmp(&pg[i], &sg[i], 4) != 0;
    for (size_t i = 0; i < pu.size() && i < su.size(); i++) diff += memcmp(&pu[i], &su[i], 4) != 0;
    CHECK(diff == 0, "%s pair launch differs from the single launches in %zu partials", what, diff);
    printf("  %s gate + up pair K=%g (%s rows), %lld assignments, %d splits: %s (%zu partials)\n", what, k2 / 2.0,
           fmt == EXL3_MOE_ACT_E4M3_SLOT ? "e4m3" : "bf16", (long long)n, zp.splits,
           diff ? "DIFFER" : "bit-identical to two single launches", pg.size() + pu.size());
    if (!refusals) {
        CUDA_OK(cudaDeviceSynchronize());
        cudaFree(d_dst); cudaFree(d_src); cudaFree(d_b); cudaFree(ws);
        da.release(); g.release(); u.release();
        return;
    }
    const int r_rate = exl3_moe_decode_launch(g.d_table, nullptr, 7, true, da.desc, d_dst, d_src, d_b, N, K, n, E, ws, w1, &z1, 0);
    const int r_ws = exl3_moe_decode_launch(g.d_table, u.d_table, 6, true, da.desc, d_dst, d_src, d_b, N, K, n, E, ws, w1, &z1, 0);
    exl3_moe_act nosf = da.desc;
    nosf.sf = nullptr;
    const int r_sf = exl3_moe_decode_launch(g.d_table, nullptr, 6, true, nosf, d_dst, d_src, d_b, N, K, n, E, ws, w1, &z1, 0);
    const int r_m = exl3_moe_decode_launch(g.d_table, nullptr, 6, true, da.desc, d_dst, d_src, d_b, 2000, K, n, E, ws, w1, &z1, 0);
    const int r_pr = exl3_moe_decode_launch(g.d_table, u.d_table, 6, false, da.desc, d_dst, d_src, d_b, N, K, n, E, ws, w2, &z1, 0);
    CHECK(r_rate && r_ws && r_sf && r_m && r_pr, "decode refusals: rate %d, workspace %d, E4M3 without scales %d, M %% 128 %d, "
          "unrotated pair %d (all must be nonzero)", r_rate, r_ws, r_sf, r_m, r_pr);
    printf("  k2=7 rc=%d, a pair in a one-projection workspace rc=%d, E4M3 slot without its scales rc=%d, M=2000 rc=%d, "
           "an unrotated pair rc=%d\n", r_rate, r_ws, r_sf, r_m, r_pr);
    CUDA_OK(cudaDeviceSynchronize());
    cudaFree(d_dst); cudaFree(d_src); cudaFree(d_b); cudaFree(ws);
    da.release(); g.release(); u.release();
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

    g_decode = true;
    printf("6-7. the DECODE arm (task #46): every rate, both projections, each family's format, 16 tokens (Zipf "
           "routing: experts with several 16-row tiles beside single rows), batch invariance\n");
    for (const shape *sh : {&v4, &v41, &qw})
        for (int k2 : k2s)
            for (int gate = 1; gate >= 0; gate--)
                section_case(*sh, gate, k2, 16, sh->fmt, 160, k2 == sh->k2_gate || k2 == sh->k2_down, 2, false);
    printf("   the other producer format on the same shapes\n");
    section_case(v4, true, 6, 16, EXL3_MOE_ACT_BF16_ROWS, 96, true, 2, false);
    section_case(v4, false, 6, 16, EXL3_MOE_ACT_BF16_ROWS, 96, true, 2, false);
    section_case(qw, true, 8, 16, EXL3_MOE_ACT_E4M3_SLOT, 160, true, 2, false);
    section_case(qw, false, 10, 16, EXL3_MOE_ACT_E4M3_SLOT, 160, true, 2, false);
    printf("8. decode rows 1..40 at the production rates (DeepSeek K3, Qwen K4 / K5 and turboderp's K3 pair), "
           "batch invariance: a row's bits never depend on how many rows ride with it\n");
    static const int dtoks[] = {1, 2, 3, 5, 8, 16, 40};
    const shape qwp = {"Qwen-td305", 2560, 640, 512, 10, false, 6, 6, EXL3_MOE_ACT_BF16_ROWS};
    for (const shape *sh : {&v4, &v41, &qw, &qwp})
        for (int t : dtoks)
            for (int gate = 1; gate >= 0; gate--)
                section_case(*sh, gate, gate ? sh->k2_gate : sh->k2_down, t, sh->fmt, 160, t > 1, 4, t == 5 || t == 40);
    section_decode_pair();
    /* 10. L295 F7: the pair arm at K5 -- Inkling's pair build (routed 4096 -> 2048 -> 4096, 256 experts, top-6, bf16
     * rows; gate and up split, each rotating its own input).  The tensor-core arms instantiate K5 for every routed
     * kind already (EXL3_ARM_MOE_PREFILL), so the pair's K5 is the rate table's line, and this proves the arms
     * serve it at Inkling's shapes: the gate (rotating) and the down, prompt rows 1..4096 (chunk invariance to
     * 1024) and decode rows 1..40 (batch invariance), graded against the f32-weight reference; the pair in one
     * decode launch == its two single launches, bit for bit.  Last, so the sections above keep their draws. */
    printf("10. Inkling's pair at K5 (L295 F7): prompt rows 1..4096 and decode rows 1..40, both projections, bf16 "
           "rows; the pair launch\n");
    const shape ink = {"Inkling-K5", 4096, 2048, 256, 6, false, 10, 10, EXL3_MOE_ACT_BF16_ROWS};
    g_decode = false;
    for (int t : {1, 5, 16, 64, 512, 4096})
        for (int gate = 1; gate >= 0; gate--)
            section_case(ink, gate, 10, t, ink.fmt, t >= 4096 ? 192 : 256, t > 1 && t <= 1024, 4, t >= 512);
    g_decode = true;
    for (int t : dtoks)
        for (int gate = 1; gate >= 0; gate--)
            section_case(ink, gate, 10, t, ink.fmt, 160, t > 1, 4, t == 5 || t == 40);
    section_decode_pair("Inkling", 10, EXL3_MOE_ACT_BF16_ROWS, false);
    CHECK(exl3_arm_has_rate(EXL3_ARM_PAIR, 10), "the pair arm's rate table has no K5");
    /* 11. L297: the decode arm at a verify's widths for Nemotron's up-only routed shape (latent 1024 -> 2688 -> 1024, 512
     * experts, top-22, K4, bf16 rows; the up rotates its input as a gate does) -- graded against the f32-weight
     * reference, and every width's rows == the same rows through the 1-row path (each row its own call) and == the
     * mixed cuts, byte for byte: an R-row step reads an expert's tile once for all of its rows (the plan's 16-row
     * tiles; hot experts carry several) and gives each row the 1-row decode's bits.  Qwen's shape the same at 8 and
     * 16 rows.  Last, so the sections above keep their draws. */
    printf("11. the decode arm at verify widths (L297): Nemotron's up-only shape (1024 -> 2688 -> 1024, 512 experts, "
           "top-22, K4) rows 1..40, every row == its 1-row call; Qwen's at 8 / 16 rows\n");
    const shape nem = {"Nemotron-K4", 1024, 2688, 512, 22, false, 8, 8, EXL3_MOE_ACT_BF16_ROWS};
    for (int t : {1, 2, 3, 4, 6, 8, 16, 22, 40})
        for (int gate = 1; gate >= 0; gate--) {
            g_cut_one = true;
            section_case(nem, gate, 8, t, nem.fmt, 160, t > 1, 4, t == 6 || t == 40);
            g_cut_one = false;
            if (t >= 6) section_case(nem, gate, 8, t, nem.fmt, 32, true, 4, false);
        }
    g_cut_one = true;
    for (int t : {8, 16})
        for (int gate = 1; gate >= 0; gate--)
            section_case(qw, gate, gate ? qw.k2_gate : qw.k2_down, t, qw.fmt, 64, true, 4, false);
    g_cut_one = false;
    printf(g_fail ? "EXL3-MOE-PREFILL GATE FAIL\n" : "EXL3-MOE-PREFILL GATE PASS\n");
    return g_fail;
}
