/* EXL3 MoE EPILOGUE GATE (L245 step 3; task #46): the routed arm's projections and epilogues against the host
 * authority.
 *
 * Model-free.  Random trellis words (any 16-bit word is a valid tile), random
 * fp16 scales in the real checkpoint's distribution (suh ~0.025 with signs,
 * svh ~+-1), a random E4M3 activation slot -- graded against a reference built
 * ONLY from src/engine/exl3_trellis.h (the host dequant that
 * tests/exl3_dequant_gate.cpp holds byte-exact to exllamav3) and double-precision
 * arithmetic, the z each projection feeds the epilogues coming from the DECODE
 * arm (exl3_moe_decode_launch, its K-split partials summed in split order):
 *
 *   1. gate / up: z = W_hat^T (H128(x * suh) / sqrt128) per assignment, per
 *      projection, at the V4.1 expert shape (5120 -> 2304), every rate the
 *      DeepSeek stacks carry (K = 2, 2.5, 3).  The operand is the arm's one fp16
 *      plane (11 significant bits per (row, 128-block)): max |err| / max |z|
 *      under 1e-3 (measured ~2e-4).
 *   2. the down (no in-kernel rotation) at 2304 -> 5120 on the fold's own E4M3
 *      mid, which the plane holds exactly: f32 accumulation only, 2e-5.
 *   3. the fold: svh * H(z) on both projections, swiglu with the router weight,
 *      the down input rotation H(v * suh_d), the E4M3 emit in 32-groups --
 *      decoded back and compared within one E4M3 ulp at the group's scale.
 *   4. the sum: slot-ordered sum of svh_d * H(z_d), and the non-finite flag.
 *   4b. the split partials: the fold and the sum fed the decode arm's partials
 *      (adding the splits themselves) == fed the pre-summed z, bit for bit.
 *   5. a mutation: one flipped trellis bit must move the projection's output.
 * Then Qwen3.8-Flash-Next's fused gate_up / down (exl3_moe_epilogue_gate_qwen.inc).
 * The decode arm's own grade (every rate, shape, format, 1..40 rows, batch
 * invariance) is tests/exl3_moe_prefill_gate.cu.
 *
 * usage: ./tests/exl3_moe_epilogue_gate
 */
#include "../src/cuda/mmq/exl3_moe_epilogue.cu"
#include "../src/cuda/mmq/exl3_moe_prefill.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "EXL3-MOE-EPILOGUE FAIL: " __VA_ARGS__); \
                                       fprintf(stderr, "\n"); g_fail = 1; } } while (0)
#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "EXL3-MOE-EPILOGUE FAIL: %s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

static uint32_t g_rng = 0x2545F491u;
static uint32_t rnd(void) { uint32_t x = g_rng; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return g_rng = x; }
static float rndf(void) { return (float)(rnd() >> 8) * (1.0f / 16777216.0f); }

/* host E4M3 (OCP): 1-4-3, bias 7, subnormals at exp 0, NaN = exp 15 mant 7 */
static float e4m3_to_f32(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 0) v = (float)m / 8.0f * exp2f(-6.0f);
    else if (e == 15 && m == 7) v = NAN;
    else v = (1.0f + (float)m / 8.0f) * exp2f((float)(e - 7));
    return s ? -v : v;
}
static uint8_t rnd_e4m3(void) { uint8_t b; do { b = (uint8_t)rnd(); } while ((b & 0x7fu) == 0x7fu || (b & 0x7fu) < 0x08u); return b; }

/* the checkpoint's own distributions: suh carries the per-row scale, svh ~ +-1 */
static uint16_t rnd_suh(void) { return exl3_f32_to_f16((0.5f + rndf()) * 0.025f * ((rnd() & 1u) ? 1.0f : -1.0f)); }
static uint16_t rnd_svh(void) { return exl3_f32_to_f16((0.75f + 0.5f * rndf()) * ((rnd() & 1u) ? 1.0f : -1.0f)); }

/* One expert-projection in pulsar's container layout, on the host. */
struct expert_blob {
    std::vector<uint8_t> bytes;    /* [trellis | suh | svh] */
    uint64_t trellis = 0, scales = 0, stride = 0;
    const uint16_t *words() const { return (const uint16_t *)bytes.data(); }
    const uint16_t *suh() const { return (const uint16_t *)(bytes.data() + trellis); }
    const uint16_t *svh(int k) const { return suh() + k; }
};
static expert_blob make_expert(int k, int n, int k2) {
    expert_blob b;
    if (!exl3_expert_layout(k, n, k2, &b.trellis, &b.scales, &b.stride)) { fprintf(stderr, "layout refused\n"); exit(1); }
    b.bytes.resize(b.stride);
    uint16_t *w = (uint16_t *)b.bytes.data();
    for (uint64_t i = 0; i < b.trellis / 2; i++) w[i] = (uint16_t)rnd();
    uint16_t *s = (uint16_t *)(b.bytes.data() + b.trellis);
    for (int i = 0; i < k; i++) s[i] = rnd_suh();
    for (int i = 0; i < n; i++) s[k + i] = rnd_svh();
    return b;
}
/* W_hat (k, n) as double, from the host authority */
static void dequant_expert(const expert_blob &b, int k, int n, int k2, std::vector<double> &w) {
    const int words = exl3_words_per_tile(k2);
    w.assign((size_t)k * n, 0.0);
    uint16_t tile[256];
    for (int kt = 0; kt < k / 16; kt++)
        for (int nt = 0; nt < n / 16; nt++) {
            exl3_tile_dequant(b.words() + ((size_t)kt * (n / 16) + nt) * words, k2, tile);
            for (int r = 0; r < 16; r++)
                for (int c = 0; c < 16; c++)
                    w[(size_t)(kt * 16 + r) * n + nt * 16 + c] = exl3_f16_to_f32(tile[r * 16 + c]);
        }
}
static void had_blocks(std::vector<double> &v) {
    for (size_t i = 0; i + 128 <= v.size(); i += 128) exl3_had128(v.data() + i);
}

/* A device pointer table [E][2] over a device arena of E expert slices. */
struct dev_stack {
    uint8_t *arena = nullptr;
    const void **table = nullptr;
    uint64_t stride = 0, split = 0;
};
static int upload_stack(const std::vector<expert_blob> &ex, dev_stack &d) {
    d.stride = ex[0].stride; d.split = ex[0].trellis;
    CUDA_OK(cudaMalloc((void **)&d.arena, d.stride * ex.size()));
    for (size_t e = 0; e < ex.size(); e++)
        CUDA_OK(cudaMemcpy(d.arena + e * d.stride, ex[e].bytes.data(), d.stride, cudaMemcpyHostToDevice));
    std::vector<const void *> t(ex.size() * 2);
    for (size_t e = 0; e < ex.size(); e++) { t[2 * e] = d.arena + e * d.stride; t[2 * e + 1] = d.arena + e * d.stride + d.split; }
    CUDA_OK(cudaMalloc((void **)&d.table, t.size() * sizeof(void *)));
    CUDA_OK(cudaMemcpy(d.table, t.data(), t.size() * sizeof(void *), cudaMemcpyHostToDevice));
    return 0;
}

/* The routing of one step: n_tok tokens x n_slot slots over E experts, sorted
 * by expert exactly as the driver's mm_ids_helper lays it out. */
struct routing {
    int n_tok, n_slot, E, n_assign;
    std::vector<int32_t> selected;       /* [pair] expert, pair = tok*n_slot + slot */
    std::vector<int32_t> ids_dst;        /* [sorted col] -> pair */
    std::vector<int32_t> ids_tok;        /* [sorted col] -> token */
    std::vector<int32_t> bounds;         /* [E+1] */
    std::vector<float>   weights;        /* [pair] */
};
static routing make_routing(int n_tok, int n_slot, int E) {
    routing r; r.n_tok = n_tok; r.n_slot = n_slot; r.E = E; r.n_assign = n_tok * n_slot;
    r.selected.resize(r.n_assign); r.weights.resize(r.n_assign);
    for (int t = 0; t < n_tok; t++) {
        for (int s = 0; s < n_slot; s++) {
            int e;
            bool dup;
            do { e = (int)(rnd() % (uint32_t)E); dup = false; for (int q = 0; q < s; q++) dup |= r.selected[t * n_slot + q] == e; } while (dup);
            r.selected[t * n_slot + s] = e;
            r.weights[t * n_slot + s] = 0.05f + 0.3f * rndf();
        }
    }
    r.bounds.assign(E + 1, 0);
    for (int e = 0; e < E; e++) {
        r.bounds[e] = (int32_t)r.ids_dst.size();
        for (int p = 0; p < r.n_assign; p++) if (r.selected[p] == e) { r.ids_dst.push_back(p); r.ids_tok.push_back(p / n_slot); }
    }
    r.bounds[E] = r.n_assign;
    return r;
}

/* Activations: the producer's E4M3 slot -- per token a random E4M3 row [K] and its ue8m0 group bytes in the
 * plane's pulsar_mx_sfoff layout -- which the arm reads in place through ids_src (task #44: nothing stages it). */
struct acts {
    std::vector<uint8_t> q, sf;             /* [rows][K] E4M3, the ue8m0 plane */
    int kbp = 0;
    std::vector<double> x;                  /* [rows][K] decoded */
};
static acts make_acts(const routing &r, int K) {
    acts a;
    a.kbp = pulsar_mx_kbp(K);
    a.q.assign((size_t)r.n_tok * K, 0);
    a.sf.assign(pulsar_mx_sf_slab_bytes(r.n_tok, a.kbp), 0);
    a.x.assign((size_t)r.n_tok * K, 0.0);
    std::vector<uint8_t> sc((size_t)K / 32);
    for (int t = 0; t < r.n_tok; t++) {
        for (int g = 0; g < K / 32; g++) {
            sc[g] = (uint8_t)(120 + rnd() % 8);
            a.sf[pulsar_mx_sfoff(t, g, a.kbp)] = sc[g];
        }
        for (int k = 0; k < K; k++) {
            a.q[(size_t)t * K + k] = rnd_e4m3();
            a.x[(size_t)t * K + k] = (double)e4m3_to_f32(a.q[(size_t)t * K + k]) * exp2((double)sc[k / 32] - 127.0);
        }
    }
    return a;
}
/* A device slot's rows decoded on the host (the fold's mid, read back). */
static acts decode_slot(const std::vector<uint8_t> &q, const std::vector<uint8_t> &sf, int rows, int K, int kbp) {
    acts a;
    a.q = q; a.sf = sf; a.kbp = kbp;
    a.x.assign((size_t)rows * K, 0.0);
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < K; k++)
            a.x[(size_t)r * K + k] = (double)e4m3_to_f32(q[(size_t)r * K + k]) * exp2((double)sf[pulsar_mx_sfoff(r, k / 32, kbp)] - 127.0);
    return a;
}
/* the slot on the device, as the arm reads it; free with free_act */
static exl3_moe_act upload_act(const acts &a) {
    void *q = nullptr, *sf = nullptr;
    if (cudaMalloc(&q, a.q.size()) != cudaSuccess || cudaMalloc(&sf, a.sf.size()) != cudaSuccess) { fprintf(stderr, "alloc\n"); exit(1); }
    cudaMemcpy(q, a.q.data(), a.q.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(sf, a.sf.data(), a.sf.size(), cudaMemcpyHostToDevice);
    return exl3_moe_act{EXL3_MOE_ACT_E4M3_SLOT, q, sf, a.kbp};
}
static void free_act(const exl3_moe_act &a) { cudaFree(const_cast<void *>(a.x)); cudaFree(const_cast<void *>(a.sf)); }
static int32_t *upload_ids(const std::vector<int32_t> &v) {
    int32_t *d = nullptr;
    if (cudaMalloc((void **)&d, v.size() * sizeof(int32_t)) != cudaSuccess) { fprintf(stderr, "alloc\n"); exit(1); }
    cudaMemcpy(d, v.data(), v.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    return d;
}
/* z = the split partials summed in split order (the fold's and the sum's own order), [rows][M] */
__global__ void sum_splits_kernel(const float *__restrict__ part, int splits, int64_t stride, float *__restrict__ z) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < stride; i += (int64_t)gridDim.x * blockDim.x) {
        float v = part[i];
        for (int sp = 1; sp < splits; sp++) v += part[sp * stride + i];
        z[i] = v;
    }
}
/* The decode arm's launch of one projection (or a gate + up pair), its partials summed into z (/ zu); *keep gets
 * the partials themselves, valid until the next launch. */
static void *g_ws = nullptr;
static size_t g_ws_bytes = 0;
static int dec_launch(const void *t, const void *ut, bool rotate, int k2, const exl3_moe_act &a, const int32_t *ids,
                      const int32_t *src, const int32_t *bounds, float *z, float *zu, int M, int K, int n, int E,
                      exl3_moe_partials *keep = nullptr) {
    const size_t need = exl3_moe_decode_ws_bytes(ut ? 2 : 1, n, K, M);
    if (need > g_ws_bytes) {
        cudaFree(g_ws);
        if (cudaMalloc(&g_ws, need) != cudaSuccess) return -9;
        g_ws_bytes = need;
    }
    exl3_moe_partials zp{};
    const int rc = exl3_moe_decode_launch(t, ut, k2, rotate, a, ids, src, bounds, M, K, n, E, g_ws, g_ws_bytes, &zp, 0);
    if (rc) return rc;
    sum_splits_kernel<<<64, 256>>>(zp.z[0], zp.splits, zp.split_stride, z);
    if (ut) sum_splits_kernel<<<64, 256>>>(zp.z[1], zp.splits, zp.split_stride, zu);
    if (keep) *keep = zp;
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}
static int gemv_pair(const void *gt, const void *ut, int k2, const exl3_moe_act &a, const int32_t *ids, const int32_t *src,
                     const int32_t *bounds, float *zg, float *zu, int M, int K, int n, int E,
                     exl3_moe_partials *keep = nullptr) {
    return dec_launch(gt, ut, true, k2, a, ids, src, bounds, zg, zu, M, K, n, E, keep);
}
static int gemv_one(int kind, const void *t, int k2, const exl3_moe_act &a, const int32_t *ids, const int32_t *src,
                    const int32_t *bounds, float *z, int M, int K, int n, int E, exl3_moe_partials *keep = nullptr) {
    if (!exl3_arm_has_rate(kind, k2)) return -1;   /* the kind's admission table (the front door's check) */
    return dec_launch(t, nullptr, kind != EXL3_ARM_DOWN, k2, a, ids, src, bounds, z, nullptr, M, K, n, E, keep);
}

/* host reference z for one assignment: W_hat^T applied to the rotated (or
 * plain) activation */
static void ref_gemv(const std::vector<double> &what, int K, int M, const double *x, const uint16_t *suh,
                     bool rotate, std::vector<double> &z) {
    std::vector<double> xr(x, x + K);
    if (rotate) {
        for (int k = 0; k < K; k++) xr[k] *= exl3_f16_to_f32(suh[k]);
        had_blocks(xr);
    }
    z.assign(M, 0.0);
    for (int k = 0; k < K; k++) {
        const double xk = xr[k];
        const double *wr = what.data() + (size_t)k * M;
        for (int n = 0; n < M; n++) z[n] += wr[n] * xk;
    }
}
static double max_rel(const std::vector<float> &got, const std::vector<double> &want, size_t off, size_t n, double *scale_out) {
    double mx = 0, sc = 0;
    for (size_t i = 0; i < n; i++) { sc = fmax(sc, fabs(want[i])); }
    for (size_t i = 0; i < n; i++) mx = fmax(mx, fabs((double)got[off + i] - want[i]));
    *scale_out = sc;
    return sc > 0 ? mx / sc : mx;
}
/* FNV-1a over a buffer's bytes: the DeepSeek sections print it so a change to
 * the kernel TUs can be proven byte-neutral by diffing two gate logs. */
static uint64_t fnv64(const std::vector<float> &v) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t *b = (const uint8_t *)v.data();
    for (size_t i = 0; i < v.size() * sizeof(float); i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static uint64_t fnv64b(const std::vector<uint8_t> &v) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t b : v) { h ^= b; h *= 1099511628211ull; }
    return h;
}
static float swiglu_host(float g, float u, float wv, float clamp) {
    if (clamp > 1.0e-6f) { g = fminf(g, clamp); u = fminf(fmaxf(u, -clamp), clamp); }
    const float s = g / (1.0f + expf(-g));
    return s * u * wv;
}

#include "exl3_moe_epilogue_gate_qwen.inc"

int main(void) {
    const int K = 5120, M = 2304;      /* the V4.1 expert: in 5120, mid 2304 */
    const int E = 4, n_tok = 2, n_slot = 3;
    printf("exl3-moe-epilogue-gate: the EXL3 routed arm vs src/engine/exl3_trellis.h, %d -> %d, %d experts, %d assignments\n",
           K, M, E, n_tok * n_slot);

    const routing r = make_routing(n_tok, n_slot, E);
    int32_t *d_ids = nullptr, *d_bounds = nullptr, *d_sel = nullptr;
    float *d_w = nullptr;
    CUDA_OK(cudaMalloc((void **)&d_ids, r.n_assign * sizeof(int32_t)));
    CUDA_OK(cudaMalloc((void **)&d_bounds, (E + 1) * sizeof(int32_t)));
    CUDA_OK(cudaMalloc((void **)&d_sel, r.n_assign * sizeof(int32_t)));
    CUDA_OK(cudaMalloc((void **)&d_w, r.n_assign * sizeof(float)));
    CUDA_OK(cudaMemcpy(d_ids, r.ids_dst.data(), r.n_assign * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_bounds, r.bounds.data(), (E + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_sel, r.selected.data(), r.n_assign * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_w, r.weights.data(), r.n_assign * sizeof(float), cudaMemcpyHostToDevice));

    const acts a = make_acts(r, K);
    const exl3_moe_act d_act = upload_act(a);
    int32_t *d_src = upload_ids(r.ids_tok);

    float *d_zg = nullptr, *d_zu = nullptr;
    CUDA_OK(cudaMalloc((void **)&d_zg, (size_t)r.n_assign * M * sizeof(float)));
    CUDA_OK(cudaMalloc((void **)&d_zu, (size_t)r.n_assign * M * sizeof(float)));
    std::vector<float> zg((size_t)r.n_assign * M), zu((size_t)r.n_assign * M);

    /* 1. the decode arm's gate + up pair at every rate */
    std::vector<expert_blob> gate, up;
    std::vector<std::vector<double>> what_g(E), what_u(E);
    for (int k2 : {4, 5, 6}) {
        gate.clear(); up.clear();
        for (int e = 0; e < E; e++) { gate.push_back(make_expert(K, M, k2)); up.push_back(make_expert(K, M, k2)); }
        dev_stack sg, su;
        if (upload_stack(gate, sg) || upload_stack(up, su)) return 1;
        CUDA_OK(cudaMemset(d_zg, 0, (size_t)r.n_assign * M * sizeof(float)));
        const int rc = gemv_pair(sg.table, su.table, k2, d_act, d_ids, d_src, d_bounds, d_zg, d_zu, M, K, r.n_assign, E);
        CHECK(rc == 0, "pair launch k2=%d rc=%d", k2, rc);
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaMemcpy(zg.data(), d_zg, zg.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(zu.data(), d_zu, zu.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int e = 0; e < E; e++) { dequant_expert(gate[e], K, M, k2, what_g[e]); dequant_expert(up[e], K, M, k2, what_u[e]); }
        double worst = 0, scale = 0;
        for (int col = 0; col < r.n_assign; col++) {
            const int pair = r.ids_dst[col], e = r.selected[pair], t = r.ids_tok[col];
            std::vector<double> z;
            ref_gemv(what_g[e], K, M, a.x.data() + (size_t)t * K, gate[e].suh(), true, z);
            double s; worst = fmax(worst, max_rel(zg, z, (size_t)pair * M, M, &s)); scale = fmax(scale, s);
            ref_gemv(what_u[e], K, M, a.x.data() + (size_t)t * K, up[e].suh(), true, z);
            worst = fmax(worst, max_rel(zu, z, (size_t)pair * M, M, &s));
        }
        CHECK(worst < 1e-3, "gate / up pair K=%g: max rel error %.3e vs the host authority (limit 1e-3)", k2 / 2.0, worst);
        printf("  gate / up pair K=%g: %d assignments x %d outputs, max rel %.2e (|z| up to %.3g); bytes fnv %016llx\n",
               k2 / 2.0, r.n_assign, M, worst, scale, (unsigned long long)(fnv64(zg) ^ (fnv64(zu) * 3u)));

        if (k2 == 6) {
            /* 3. the fold, on this K=3 gate/up and a K=3 down */
            std::vector<expert_blob> down;
            for (int e = 0; e < E; e++) down.push_back(make_expert(M, K, 6));
            dev_stack sd;
            if (upload_stack(down, sd)) return 1;
            const int kbp = pulsar_mx_kbp(M);
            const size_t sf_bytes = pulsar_mx_sf_slab_bytes(r.n_assign, kbp);
            uint8_t *d_mq = nullptr, *d_msf = nullptr;
            CUDA_OK(cudaMalloc((void **)&d_mq, (size_t)r.n_assign * M));
            CUDA_OK(cudaMalloc((void **)&d_msf, sf_bytes));
            CUDA_OK(cudaMemset(d_msf, 0, sf_bytes));
            const int frc = ds4_exl3_moe_fold_launch(d_zg, d_zu, d_sel, d_w, sg.table, su.table, sd.table, K, M, r.n_assign, 0.0f, d_mq, d_msf, kbp, 0);
            CHECK(frc == 0, "fold launch rc=%d", frc);
            CUDA_OK(cudaDeviceSynchronize());
            std::vector<uint8_t> mq((size_t)r.n_assign * M), msf(sf_bytes);
            CUDA_OK(cudaMemcpy(mq.data(), d_mq, mq.size(), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(msf.data(), d_msf, sf_bytes, cudaMemcpyDeviceToHost));
            /* host: from the DEVICE z (the fold's own input), the rest in double */
            double fold_worst = 0;
            size_t n_bad = 0;
            for (int pair = 0; pair < r.n_assign; pair++) {
                const int e = r.selected[pair];
                std::vector<double> g(M), u(M), t(M);
                for (int n = 0; n < M; n++) { g[n] = zg[(size_t)pair * M + n]; u[n] = zu[(size_t)pair * M + n]; }
                had_blocks(g); had_blocks(u);
                for (int n = 0; n < M; n++) {
                    const float yg = (float)(g[n] * exl3_f16_to_f32(gate[e].svh(K)[n]));
                    const float yu = (float)(u[n] * exl3_f16_to_f32(up[e].svh(K)[n]));
                    t[n] = (double)swiglu_host(yg, yu, r.weights[pair], 0.0f) * exl3_f16_to_f32(down[e].suh()[n]);
                }
                had_blocks(t);
                for (int grp = 0; grp < M / 32; grp++) {
                    /* decode the device's group with ITS scale byte; the shared exponent
                     * is amax-derived so a last-bit difference in amax can shift it by
                     * one -- compare values, one E4M3 ulp at the group's scale */
                    const int se = (int)msf[pulsar_mx_sfoff(pair, grp, kbp)] - 127;
                    double amax = 0;
                    for (int j = 0; j < 32; j++) amax = fmax(amax, fabs(t[grp * 32 + j]));
                    const double ulp = exp2((double)se) * 0.125 * 2.0;   /* one e4m3 step at the top binade, x2 for a shifted exponent */
                    for (int j = 0; j < 32; j++) {
                        const double got = (double)e4m3_to_f32(mq[(size_t)pair * M + grp * 32 + j]) * exp2((double)se);
                        const double err = fabs(got - t[grp * 32 + j]);
                        const double lim = fmax(ulp, fabs(t[grp * 32 + j]) * (1.0 / 8.0)) + 1e-30;
                        if (err > lim) n_bad++;
                        if (amax > 0) fold_worst = fmax(fold_worst, err / amax);
                    }
                }
            }
            CHECK(n_bad == 0, "fold: %zu of %zu mid values outside one E4M3 ulp of the host chain", n_bad, (size_t)r.n_assign * M);
            printf("  fold: %d pairs x %d, worst |err|/amax %.2e, %zu outside tolerance; bytes fnv %016llx\n",
                   r.n_assign, M, fold_worst, n_bad, (unsigned long long)(fnv64b(mq) ^ (fnv64b(msf) * 3u)));

            /* 4. the sum on random z_d [pair][K] (out_dim = K here) */
            std::vector<float> zd((size_t)r.n_assign * K);
            for (auto &v : zd) v = rndf() * 2.0f - 1.0f;
            float *d_zd = nullptr, *d_out = nullptr;
            uint32_t *d_nf = nullptr;
            CUDA_OK(cudaMalloc((void **)&d_zd, zd.size() * sizeof(float)));
            CUDA_OK(cudaMalloc((void **)&d_out, (size_t)n_tok * K * sizeof(float)));
            CUDA_OK(cudaMalloc((void **)&d_nf, sizeof(uint32_t)));
            CUDA_OK(cudaMemcpy(d_zd, zd.data(), zd.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemset(d_nf, 0, sizeof(uint32_t)));
            int src_rc = ds4_exl3_moe_sum_launch(d_out, d_zd, d_sel, sd.table, M, K, n_slot, n_tok, d_nf, 0x501u, 0);
            CHECK(src_rc == 0, "sum launch rc=%d", src_rc);
            CUDA_OK(cudaDeviceSynchronize());
            std::vector<float> out((size_t)n_tok * K);
            uint32_t nf = 7;
            CUDA_OK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(&nf, d_nf, sizeof nf, cudaMemcpyDeviceToHost));
            CHECK(nf == 0, "sum flagged a non-finite value on finite input (0x%x)", nf);
            double sum_worst = 0;
            for (int t = 0; t < n_tok; t++) {
                std::vector<double> acc(K, 0.0);
                for (int s = 0; s < n_slot; s++) {
                    const int pair = t * n_slot + s, e = r.selected[pair];
                    std::vector<double> z(zd.begin() + (size_t)pair * K, zd.begin() + (size_t)(pair + 1) * K);
                    had_blocks(z);
                    for (int o = 0; o < K; o++) acc[o] += z[o] * exl3_f16_to_f32(down[e].svh(M)[o]);
                }
                double s; sum_worst = fmax(sum_worst, max_rel(out, acc, (size_t)t * K, K, &s));
            }
            CHECK(sum_worst < 1e-5, "sum: max rel %.3e (limit 1e-5)", sum_worst);
            const float nan_v = NAN;
            CUDA_OK(cudaMemcpy(d_zd + 5, &nan_v, sizeof nan_v, cudaMemcpyHostToDevice));
            src_rc = ds4_exl3_moe_sum_launch(d_out, d_zd, d_sel, sd.table, M, K, n_slot, n_tok, d_nf, 0x501u, 0);
            CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(&nf, d_nf, sizeof nf, cudaMemcpyDeviceToHost));
            CHECK(src_rc == 0 && nf == 0x501u, "sum did not flag a NaN input (flag 0x%x)", nf);
            printf("  sum: %d tokens x %d, max rel %.2e; NaN flagged 0x%x\n", n_tok, K, sum_worst, nf);

            /* 2. the down: 2304 -> 5120, input = the fold's mid, read in its E4M3 slot (row = pair) */
            {
                const acts am = decode_slot(mq, msf, r.n_assign, M, kbp);
                const exl3_moe_act d_am{EXL3_MOE_ACT_E4M3_SLOT, d_mq, d_msf, kbp};
                float *d_zdown = nullptr;
                CUDA_OK(cudaMalloc((void **)&d_zdown, (size_t)r.n_assign * K * sizeof(float)));
                const int drc = gemv_one(EXL3_ARM_DOWN, sd.table, 6, d_am, d_ids, d_ids, d_bounds, d_zdown, K, M, r.n_assign, E);
                CHECK(drc == 0, "down launch rc=%d", drc);
                CUDA_OK(cudaDeviceSynchronize());
                std::vector<float> zdown((size_t)r.n_assign * K);
                CUDA_OK(cudaMemcpy(zdown.data(), d_zdown, zdown.size() * sizeof(float), cudaMemcpyDeviceToHost));
                std::vector<double> what_d;
                double dworst = 0, dscale = 0;
                for (int col = 0; col < r.n_assign; col++) {
                    const int pair = r.ids_dst[col], e = r.selected[pair];
                    dequant_expert(down[e], M, K, 6, what_d);
                    std::vector<double> z;
                    ref_gemv(what_d, M, K, am.x.data() + (size_t)pair * M, nullptr, false, z);
                    double s; dworst = fmax(dworst, max_rel(zdown, z, (size_t)pair * K, K, &s)); dscale = fmax(dscale, s);
                }
                CHECK(dworst < 2e-5, "down: max rel %.3e (limit 2e-5)", dworst);
                printf("  down K=3: %d assignments x %d outputs, max rel %.2e (|z| up to %.3g); bytes fnv %016llx\n",
                       r.n_assign, K, dworst, dscale, (unsigned long long)fnv64(zdown));

                cudaFree(d_zdown);
            }
            /* 4b. the split partials: the fold and the sum fed the decode arm's partials (adding the splits
             * themselves) write exactly what they write fed the pre-summed z */
            {
                exl3_moe_partials zp{};
                int prc = gemv_pair(sg.table, su.table, 6, d_act, d_ids, d_src, d_bounds, d_zg, d_zu, M, K, r.n_assign, E, &zp);
                CHECK(prc == 0 && zp.splits > 1, "partials: launch rc=%d, %d splits (the V4.1 gate must split)", prc, zp.splits);
                const size_t mb = (size_t)r.n_assign * M, sfb = pulsar_mx_sf_slab_bytes(r.n_assign, kbp);
                std::vector<uint8_t> q0(mb), q1(mb), f0(sfb), f1(sfb);
                CUDA_OK(cudaMemset(d_msf, 0, sfb));
                prc |= ds4_exl3_moe_fold_launch(d_zg, d_zu, d_sel, d_w, sg.table, su.table, sd.table, K, M, r.n_assign, 0.0f, d_mq, d_msf, kbp, 0);
                CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(q0.data(), d_mq, mb, cudaMemcpyDeviceToHost));
                CUDA_OK(cudaMemcpy(f0.data(), d_msf, sfb, cudaMemcpyDeviceToHost));
                CUDA_OK(cudaMemset(d_msf, 0, sfb));
                prc |= ds4_exl3_moe_fold_launch(zp.z[0], zp.z[1], d_sel, d_w, sg.table, su.table, sd.table, K, M, r.n_assign, 0.0f, d_mq, d_msf, kbp, 0,
                                                nullptr, zp.splits, zp.split_stride);
                CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(q1.data(), d_mq, mb, cudaMemcpyDeviceToHost));
                CUDA_OK(cudaMemcpy(f1.data(), d_msf, sfb, cudaMemcpyDeviceToHost));
                CHECK(prc == 0 && q0 == q1 && f0 == f1, "fold: the split partials and the summed z emit different mids (rc=%d)", prc);
                /* the sum, on the down's partials vs their pre-summed z */
                exl3_moe_partials dp{};
                const exl3_moe_act d_am2{EXL3_MOE_ACT_E4M3_SLOT, d_mq, d_msf, kbp};
                float *d_zd2 = nullptr, *d_o0 = nullptr, *d_o1 = nullptr;
                CUDA_OK(cudaMalloc((void **)&d_zd2, (size_t)r.n_assign * K * sizeof(float)));
                CUDA_OK(cudaMalloc((void **)&d_o0, (size_t)n_tok * K * sizeof(float)));
                CUDA_OK(cudaMalloc((void **)&d_o1, (size_t)n_tok * K * sizeof(float)));
                prc |= gemv_one(EXL3_ARM_DOWN, sd.table, 6, d_am2, d_ids, d_ids, d_bounds, d_zd2, K, M, r.n_assign, E, &dp);
                prc |= ds4_exl3_moe_sum_launch(d_o0, d_zd2, d_sel, sd.table, M, K, n_slot, n_tok, d_nf, 0x501u, 0);
                prc |= ds4_exl3_moe_sum_launch(d_o1, dp.z[0], d_sel, sd.table, M, K, n_slot, n_tok, d_nf, 0x501u, 0, nullptr,
                                               nullptr, dp.splits, dp.split_stride);
                std::vector<float> o0((size_t)n_tok * K), o1(o0.size());
                CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(o0.data(), d_o0, o0.size() * 4, cudaMemcpyDeviceToHost));
                CUDA_OK(cudaMemcpy(o1.data(), d_o1, o1.size() * 4, cudaMemcpyDeviceToHost));
                CHECK(prc == 0 && memcmp(o0.data(), o1.data(), o0.size() * 4) == 0,
                      "sum: the split partials and the summed z give different outputs (rc=%d)", prc);
                printf("  split partials: fold (%d splits) and sum (%d splits) == fed the summed z, bit for bit\n",
                       zp.splits, dp.splits);
                cudaFree(d_zd2); cudaFree(d_o0); cudaFree(d_o1);
            }
            /* 5. the mutation: one flipped trellis bit in expert 0's gate must move z */
            {
                uint8_t byte = 0;
                CUDA_OK(cudaMemcpy(&byte, sg.arena + 1000, 1, cudaMemcpyDeviceToHost));
                byte ^= 0x10u;
                CUDA_OK(cudaMemcpy(sg.arena + 1000, &byte, 1, cudaMemcpyHostToDevice));
                std::vector<float> zg2(zg.size());
                gemv_pair(sg.table, su.table, 6, d_act, d_ids, d_src, d_bounds, d_zg, d_zu, M, K, r.n_assign, E);
                CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(zg2.data(), d_zg, zg2.size() * sizeof(float), cudaMemcpyDeviceToHost));
                size_t moved = 0;
                for (size_t i = 0; i < zg.size(); i++) moved += zg[i] != zg2[i];
                CHECK(moved > 0, "a flipped trellis bit did not change the arm's output -- the comparison is degenerate");
                printf("  mutation: one flipped bit moved %zu outputs\n", moved);
            }
            cudaFree(d_mq); cudaFree(d_msf); cudaFree(d_zd); cudaFree(d_out); cudaFree(d_nf);
            cudaFree(sd.arena); cudaFree((void *)sd.table);
        }
        cudaFree(sg.arena); cudaFree((void *)sg.table); cudaFree(su.arena); cudaFree((void *)su.table);
    }
    if (qwen_section()) g_fail = 1;
    printf(g_fail ? "EXL3-MOE-EPILOGUE GATE FAIL\n" : "EXL3-MOE-EPILOGUE GATE PASS\n");
    return g_fail;
}
