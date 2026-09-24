/* EXL3 GEMV GATE (L245 step 3): the device arm against the host authority.
 *
 * Model-free.  Random trellis words (any 16-bit word is a valid tile), random
 * fp16 scales in the real checkpoint's distribution (suh ~0.025 with signs,
 * svh ~+-1), random E4M3 activation blocks -- and every kernel of the EXL3 arm
 * is graded against a reference built ONLY from src/engine/exl3_trellis.h
 * (the host dequant that tests/exl3_dequant_gate.cpp holds byte-exact to
 * exllamav3) and double-precision arithmetic:
 *
 *   1. the gate/up GEMV: z = W_hat^T (H128(x * suh) / sqrt128) per assignment,
 *      per projection, at the V4.1 expert shape (5120 -> 2304), for every rate
 *      the arm instantiates (K = 2, 2.5, 3).  Tolerance = f32 fma order.
 *   2. the down GEMV (single, no in-kernel rotation) at 2304 -> 5120.
 *   3. the fold: svh * H(z) on both projections, swiglu with the router weight,
 *      the down input rotation H(v * suh_d), the E4M3 emit in 32-groups --
 *      decoded back and compared within one E4M3 ulp at the group's scale.
 *   4. the sum: slot-ordered sum of svh_d * H(z_d), and the non-finite flag.
 *   5. a mutation: one flipped trellis bit must move the GEMV's output.
 *   6. the numbers: GB/s of the pair and single GEMVs at the expert shape,
 *      6 assignments (a decode step's routing), weights streamed from DRAM.
 *
 * usage: ./tests/exl3_gemv_gate
 */
#include "../src/cuda/mmq/ds4_exl3_gemv.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "EXL3-GEMV FAIL: " __VA_ARGS__); \
                                       fprintf(stderr, "\n"); g_fail = 1; } } while (0)
#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "EXL3-GEMV FAIL: %s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

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

/* Activations: per token a random E4M3 row [K] with ue8m0 group bytes; staged
 * per sorted assignment as block_mx_act_mmq [K/128][n_assign]. */
struct acts {
    /* block_mx_act_mmq is not copyable (its union), so the staging lives in bytes */
    std::vector<uint8_t> raw;               /* [K/128][n_assign] blocks */
    std::vector<double> x;                  /* [n_tok][K] decoded */
    void resize(size_t n) { raw.assign(n * sizeof(block_mx_act_mmq), 0); }
    block_mx_act_mmq &blk(size_t i) { return *reinterpret_cast<block_mx_act_mmq *>(raw.data() + i * sizeof(block_mx_act_mmq)); }
    size_t bytes() const { return raw.size(); }
};
static acts make_acts(const routing &r, int K) {
    acts a;
    const int nb = K / 128;
    std::vector<uint8_t> q((size_t)r.n_tok * K);
    std::vector<uint8_t> sc((size_t)r.n_tok * K / 32);
    a.x.assign((size_t)r.n_tok * K, 0.0);
    for (int t = 0; t < r.n_tok; t++) {
        for (int g = 0; g < K / 32; g++) sc[(size_t)t * (K / 32) + g] = (uint8_t)(120 + rnd() % 8);
        for (int k = 0; k < K; k++) {
            q[(size_t)t * K + k] = rnd_e4m3();
            a.x[(size_t)t * K + k] = (double)e4m3_to_f32(q[(size_t)t * K + k]) * exp2((double)sc[(size_t)t * (K / 32) + k / 32] - 127.0);
        }
    }
    a.resize((size_t)nb * r.n_assign);
    for (int col = 0; col < r.n_assign; col++) {
        const int t = r.ids_tok[col];
        for (int b = 0; b < nb; b++) {
            block_mx_act_mmq &blk = a.blk((size_t)b * r.n_assign + col);
            for (int g = 0; g < 4; g++) blk.d4[g] = (float)sc[(size_t)t * (K / 32) + b * 4 + g];
            for (int j = 0; j < 128; j++) blk.qs[j] = (int8_t)q[(size_t)t * K + b * 128 + j];
        }
    }
    return a;
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
static float swiglu_host(float g, float u, float wv, float clamp) {
    if (clamp > 1.0e-6f) { g = fminf(g, clamp); u = fminf(fmaxf(u, -clamp), clamp); }
    const float s = g / (1.0f + expf(-g));
    return s * u * wv;
}

int main(void) {
    const int K = 5120, M = 2304;      /* the V4.1 expert: in 5120, mid 2304 */
    const int E = 4, n_tok = 2, n_slot = 3;
    printf("exl3-gemv-gate: the EXL3 arm vs src/engine/exl3_trellis.h, %d -> %d, %d experts, %d assignments\n",
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
    block_mx_act_mmq *d_act = nullptr;
    CUDA_OK(cudaMalloc((void **)&d_act, a.bytes()));
    CUDA_OK(cudaMemcpy(d_act, a.raw.data(), a.bytes(), cudaMemcpyHostToDevice));

    float *d_zg = nullptr, *d_zu = nullptr;
    CUDA_OK(cudaMalloc((void **)&d_zg, (size_t)r.n_assign * M * sizeof(float)));
    CUDA_OK(cudaMalloc((void **)&d_zu, (size_t)r.n_assign * M * sizeof(float)));
    std::vector<float> zg((size_t)r.n_assign * M), zu((size_t)r.n_assign * M);

    /* 1. the pair GEMV at every rate */
    std::vector<expert_blob> gate, up;
    std::vector<std::vector<double>> what_g(E), what_u(E);
    for (int k2 : {4, 5, 6}) {
        gate.clear(); up.clear();
        for (int e = 0; e < E; e++) { gate.push_back(make_expert(K, M, k2)); up.push_back(make_expert(K, M, k2)); }
        dev_stack sg, su;
        if (upload_stack(gate, sg) || upload_stack(up, su)) return 1;
        CUDA_OK(cudaMemset(d_zg, 0, (size_t)r.n_assign * M * sizeof(float)));
        const int rc = ds4_exl3_moe_gemv_pair_launch(sg.table, su.table, k2, d_act, d_ids, d_bounds, d_zg, d_zu, M, K, r.n_assign, E, 0);
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
        CHECK(worst < 2e-5, "pair GEMV K=%g: max rel error %.3e vs the host authority (limit 2e-5)", k2 / 2.0, worst);
        printf("  pair GEMV K=%g: %d assignments x %d outputs, max rel %.2e (|z| up to %.3g)\n", k2 / 2.0, r.n_assign, M, worst, scale);

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
            printf("  fold: %d pairs x %d, worst |err|/amax %.2e, %zu outside tolerance\n", r.n_assign, M, fold_worst, n_bad);

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

            /* 2. the down GEMV: 2304 -> 5120, input = the fold's mid (E4M3) re-staged */
            {
                acts am;
                const int nbm = M / 128;
                am.resize((size_t)nbm * r.n_assign);
                am.x.assign((size_t)r.n_assign * M, 0.0);
                for (int col = 0; col < r.n_assign; col++) {
                    const int pair = r.ids_dst[col];
                    for (int b = 0; b < nbm; b++) {
                        block_mx_act_mmq &blk = am.blk((size_t)b * r.n_assign + col);
                        for (int g = 0; g < 4; g++) {
                            const uint8_t sb = msf[pulsar_mx_sfoff(pair, b * 4 + g, kbp)];
                            blk.d4[g] = (float)sb;
                            for (int j = 0; j < 32; j++) {
                                const uint8_t q = mq[(size_t)pair * M + b * 128 + g * 32 + j];
                                blk.qs[g * 32 + j] = (int8_t)q;
                                am.x[(size_t)pair * M + b * 128 + g * 32 + j] = (double)e4m3_to_f32(q) * exp2((double)sb - 127.0);
                            }
                        }
                    }
                }
                block_mx_act_mmq *d_am = nullptr;
                float *d_zdown = nullptr;
                CUDA_OK(cudaMalloc((void **)&d_am, am.bytes()));
                CUDA_OK(cudaMemcpy(d_am, am.raw.data(), am.bytes(), cudaMemcpyHostToDevice));
                CUDA_OK(cudaMalloc((void **)&d_zdown, (size_t)r.n_assign * K * sizeof(float)));
                const int drc = ds4_exl3_moe_gemv_single_launch(sd.table, 6, d_am, d_ids, d_bounds, d_zdown, K, M, r.n_assign, E, 0);
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
                CHECK(dworst < 2e-5, "down GEMV: max rel %.3e (limit 2e-5)", dworst);
                printf("  down GEMV K=3: %d assignments x %d outputs, max rel %.2e (|z| up to %.3g)\n", r.n_assign, K, dworst, dscale);

                /* 6. the numbers, on a stack that does NOT fit in L2 (GB10: 24 MiB):
                 * 12 experts, 12 assignments to 12 distinct experts (two decode
                 * tokens x 6 slots), so every byte the GEMV reads is a DRAM byte. */
                {
                    const int Eb = 12;
                    routing rb; rb.n_tok = 2; rb.n_slot = 6; rb.E = Eb; rb.n_assign = 12;
                    rb.selected.resize(12); rb.weights.assign(12, 0.1f); rb.bounds.assign(Eb + 1, 0);
                    for (int p2 = 0; p2 < 12; p2++) { rb.selected[p2] = p2; rb.ids_dst.push_back(p2); rb.ids_tok.push_back(p2 / 6); }
                    for (int e = 0; e <= Eb; e++) rb.bounds[e] = e;
                    std::vector<expert_blob> bg, bu, bd;
                    for (int e = 0; e < Eb; e++) { bg.push_back(make_expert(K, M, 6)); bu.push_back(make_expert(K, M, 6)); bd.push_back(make_expert(M, K, 6)); }
                    dev_stack sbg, sbu, sbd;
                    if (upload_stack(bg, sbg) || upload_stack(bu, sbu) || upload_stack(bd, sbd)) return 1;
                    const acts ab = make_acts(rb, K), abm = make_acts(rb, M);
                    block_mx_act_mmq *d_ab = nullptr, *d_abm = nullptr;
                    int32_t *d_bids = nullptr, *d_bb = nullptr;
                    float *d_bzg = nullptr, *d_bzu = nullptr, *d_bzd = nullptr;
                    CUDA_OK(cudaMalloc((void **)&d_ab, ab.bytes()));
                    CUDA_OK(cudaMemcpy(d_ab, ab.raw.data(), ab.bytes(), cudaMemcpyHostToDevice));
                    CUDA_OK(cudaMalloc((void **)&d_abm, abm.bytes()));
                    CUDA_OK(cudaMemcpy(d_abm, abm.raw.data(), abm.bytes(), cudaMemcpyHostToDevice));
                    CUDA_OK(cudaMalloc((void **)&d_bids, 12 * sizeof(int32_t)));
                    CUDA_OK(cudaMemcpy(d_bids, rb.ids_dst.data(), 12 * sizeof(int32_t), cudaMemcpyHostToDevice));
                    CUDA_OK(cudaMalloc((void **)&d_bb, (Eb + 1) * sizeof(int32_t)));
                    CUDA_OK(cudaMemcpy(d_bb, rb.bounds.data(), (Eb + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
                    CUDA_OK(cudaMalloc((void **)&d_bzg, (size_t)12 * M * sizeof(float)));
                    CUDA_OK(cudaMalloc((void **)&d_bzu, (size_t)12 * M * sizeof(float)));
                    CUDA_OK(cudaMalloc((void **)&d_bzd, (size_t)12 * K * sizeof(float)));
                    cudaEvent_t t0, t1;
                    CUDA_OK(cudaEventCreate(&t0)); CUDA_OK(cudaEventCreate(&t1));
                    const int iters = 40;
                    float ms = 0;
                    for (int i = 0; i < 3; i++) ds4_exl3_moe_gemv_pair_launch(sbg.table, sbu.table, 6, d_ab, d_bids, d_bb, d_bzg, d_bzu, M, K, 12, Eb, 0);
                    CUDA_OK(cudaEventRecord(t0));
                    for (int i = 0; i < iters; i++) ds4_exl3_moe_gemv_pair_launch(sbg.table, sbu.table, 6, d_ab, d_bids, d_bb, d_bzg, d_bzu, M, K, 12, Eb, 0);
                    CUDA_OK(cudaEventRecord(t1)); CUDA_OK(cudaEventSynchronize(t1));
                    CUDA_OK(cudaEventElapsedTime(&ms, t0, t1)); ms /= iters;
                    const double pair_bytes = 2.0 * 12 * (double)bg[0].stride;
                    printf("  pair GEMV K=3 5120->2304 x 12 assignments (12 experts, DRAM): %.1f us, %.0f GB/s (%.1f MB; %.1f us per matrix)\n",
                           ms * 1000.0, pair_bytes / (ms * 1e-3) / 1e9, pair_bytes / 1e6, ms * 1000.0 / 24.0);
                    for (int i = 0; i < 3; i++) ds4_exl3_moe_gemv_single_launch(sbd.table, 6, d_abm, d_bids, d_bb, d_bzd, K, M, 12, Eb, 0);
                    CUDA_OK(cudaEventRecord(t0));
                    for (int i = 0; i < iters; i++) ds4_exl3_moe_gemv_single_launch(sbd.table, 6, d_abm, d_bids, d_bb, d_bzd, K, M, 12, Eb, 0);
                    CUDA_OK(cudaEventRecord(t1)); CUDA_OK(cudaEventSynchronize(t1));
                    CUDA_OK(cudaEventElapsedTime(&ms, t0, t1)); ms /= iters;
                    const double down_bytes = 12.0 * (double)bd[0].stride;
                    printf("  down GEMV K=3 2304->5120 x 12 assignments (12 experts, DRAM): %.1f us, %.0f GB/s (%.1f MB; %.1f us per matrix)\n",
                           ms * 1000.0, down_bytes / (ms * 1e-3) / 1e9, down_bytes / 1e6, ms * 1000.0 / 12.0);
                    /* the same at ONE assignment (the single-stream decode step of one expert) */
                    for (int i = 0; i < 3; i++) ds4_exl3_moe_gemv_pair_launch(sbg.table, sbu.table, 6, d_ab, d_bids, d_bb, d_bzg, d_bzu, M, K, 1, Eb, 0);
                    CUDA_OK(cudaEventRecord(t0));
                    for (int i = 0; i < iters; i++) ds4_exl3_moe_gemv_pair_launch(sbg.table, sbu.table, 6, d_ab, d_bids, d_bb, d_bzg, d_bzu, M, K, 1, Eb, 0);
                    CUDA_OK(cudaEventRecord(t1)); CUDA_OK(cudaEventSynchronize(t1));
                    CUDA_OK(cudaEventElapsedTime(&ms, t0, t1)); ms /= iters;
                    printf("  pair GEMV K=3 x 1 assignment (L2-warm after the first rep): %.1f us, %.0f GB/s equivalent\n",
                           ms * 1000.0, 2.0 * bg[0].stride / (ms * 1e-3) / 1e9);
                    cudaFree(d_ab); cudaFree(d_abm); cudaFree(d_bids); cudaFree(d_bb); cudaFree(d_bzg); cudaFree(d_bzu); cudaFree(d_bzd);
                    cudaFree(sbg.arena); cudaFree((void *)sbg.table); cudaFree(sbu.arena); cudaFree((void *)sbu.table);
                    cudaFree(sbd.arena); cudaFree((void *)sbd.table);
                }
                cudaFree(d_am); cudaFree(d_zdown);
            }
            /* 5. the mutation: one flipped trellis bit in expert 0's gate must move z */
            {
                uint8_t byte = 0;
                CUDA_OK(cudaMemcpy(&byte, sg.arena + 1000, 1, cudaMemcpyDeviceToHost));
                byte ^= 0x10u;
                CUDA_OK(cudaMemcpy(sg.arena + 1000, &byte, 1, cudaMemcpyHostToDevice));
                std::vector<float> zg2(zg.size());
                ds4_exl3_moe_gemv_pair_launch(sg.table, su.table, 6, d_act, d_ids, d_bounds, d_zg, d_zu, M, K, r.n_assign, E, 0);
                CUDA_OK(cudaDeviceSynchronize());
                CUDA_OK(cudaMemcpy(zg2.data(), d_zg, zg2.size() * sizeof(float), cudaMemcpyDeviceToHost));
                size_t moved = 0;
                for (size_t i = 0; i < zg.size(); i++) moved += zg[i] != zg2[i];
                CHECK(moved > 0, "a flipped trellis bit did not change the GEMV output -- the comparison is degenerate");
                printf("  mutation: one flipped bit moved %zu outputs\n", moved);
            }
            cudaFree(d_mq); cudaFree(d_msf); cudaFree(d_zd); cudaFree(d_out); cudaFree(d_nf);
            cudaFree(sd.arena); cudaFree((void *)sd.table);
        }
        cudaFree(sg.arena); cudaFree((void *)sg.table); cudaFree(su.arena); cudaFree((void *)su.table);
    }
    printf(g_fail ? "EXL3-GEMV GATE FAIL\n" : "EXL3-GEMV GATE PASS\n");
    return g_fail;
}
