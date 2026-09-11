/* Single-pass mHC hand-over gate (L218, DeepSeek-V4.1 Block.forward).
 *
 * The fused split+collapse+norm kernel must collapse a row with the pre it
 * was HANDED (the previous sublayer's, or the identity before layer 0) and
 * leave its OWN pre behind for the next sublayer.  A kernel that collapsed
 * with its own pre (0731's hc_pre) or forgot to write the carry would run
 * clean and produce plausible activations -- this gate is the only thing that
 * tells the two apart before the reference comparison.
 *
 * Self-contained: #includes the shipped TU so it drives the REAL kernel, links
 * nothing else (sparky is aarch64; the dev box has no GPU).  Two sublayer
 * calls chained on one carry buffer, each checked against a host oracle of
 * the reference math (hc_split_sinkhorn for the coefficients, hc_pre for the
 * collapse, RMSNorm); the carry after each call must be that call's own pre,
 * and the split's pre slots must be the same bytes. */
#include "../src/cuda/pulsar_cuda_hc_router.cu"

int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "cuda: %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}
/* the "model mapping" is one device buffer here; offsets are plain byte offsets */
const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    (void)bytes; (void)what;
    return (const char *)model_map + offset;
}

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>

static const uint32_t N_ROWS = 6u, N_EMBD = 5120u, N_HC = 4u, MIX_HC = 24u, ITERS = 20u;
static const float HC_EPS = 1e-6f, NORM_EPS = 1e-6f;

static float bf16r(float x) {
    uint32_t u; memcpy(&u, &x, 4); u += 0x7fffu + ((u >> 16) & 1u); u &= 0xffff0000u;
    float f; memcpy(&f, &u, 4); return f;
}
static float sigmoidf(float z) { return 1.0f / (1.0f + expf(-z)); }

/* hc_split_sinkhorn (kernel.py) on one row */
static void oracle_split(const float *mix, const float *scale, const float *base, float *pre, float *post, float *comb) {
    for (uint32_t j = 0; j < N_HC; j++) pre[j] = sigmoidf(mix[j] * scale[0] + base[j]) + HC_EPS;
    for (uint32_t j = 0; j < N_HC; j++) post[j] = 2.0f * sigmoidf(mix[j + N_HC] * scale[1] + base[j + N_HC]);
    for (uint32_t j = 0; j < N_HC; j++)
        for (uint32_t k = 0; k < N_HC; k++)
            comb[j * N_HC + k] = mix[j * N_HC + k + 2u * N_HC] * scale[2] + base[j * N_HC + k + 2u * N_HC];
    for (uint32_t j = 0; j < N_HC; j++) {
        float m = -INFINITY; for (uint32_t k = 0; k < N_HC; k++) m = fmaxf(m, comb[j * N_HC + k]);
        float ss = 0.0f;
        for (uint32_t k = 0; k < N_HC; k++) { comb[j * N_HC + k] = expf(comb[j * N_HC + k] - m); ss += comb[j * N_HC + k]; }
        for (uint32_t k = 0; k < N_HC; k++) comb[j * N_HC + k] = comb[j * N_HC + k] / ss + HC_EPS;
    }
    for (uint32_t k = 0; k < N_HC; k++) {
        float ss = 0.0f; for (uint32_t j = 0; j < N_HC; j++) ss += comb[j * N_HC + k];
        for (uint32_t j = 0; j < N_HC; j++) comb[j * N_HC + k] /= (ss + HC_EPS);
    }
    for (uint32_t it = 1; it < ITERS; it++) {
        for (uint32_t j = 0; j < N_HC; j++) {
            float ss = 0.0f; for (uint32_t k = 0; k < N_HC; k++) ss += comb[j * N_HC + k];
            for (uint32_t k = 0; k < N_HC; k++) comb[j * N_HC + k] /= (ss + HC_EPS);
        }
        for (uint32_t k = 0; k < N_HC; k++) {
            float ss = 0.0f; for (uint32_t j = 0; j < N_HC; j++) ss += comb[j * N_HC + k];
            for (uint32_t j = 0; j < N_HC; j++) comb[j * N_HC + k] /= (ss + HC_EPS);
        }
    }
}

static pulsar_gpu_tensor *dalloc(uint64_t bytes) {
    pulsar_gpu_tensor *t = (pulsar_gpu_tensor *)calloc(1, sizeof *t);
    if (cudaMalloc(&t->ptr, bytes) != cudaSuccess) { fprintf(stderr, "cudaMalloc failed\n"); exit(1); }
    t->bytes = bytes; return t;
}

static double rel_l2(const std::vector<float> &a, const std::vector<float> &b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); i++) { const double d = (double)a[i] - (double)b[i]; num += d * d; den += (double)b[i] * (double)b[i]; }
    return sqrt(num / (den > 0 ? den : 1.0));
}

int main(void) {
    printf("HC CARRY KERNEL TEST: %u rows x %u, hc %u, %u sinkhorn iters\n", N_ROWS, N_EMBD, N_HC, ITERS);
    std::mt19937_64 rng(20260911);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    /* weights: [scale 3][base 24][norm_w n_embd f32] in one device "mapping" */
    std::vector<float> wbuf(3 + MIX_HC + N_EMBD);
    wbuf[0] = 0.7f; wbuf[1] = 0.9f; wbuf[2] = 1.1f;
    for (uint32_t i = 0; i < MIX_HC; i++) wbuf[3 + i] = 0.3f * nd(rng);
    for (uint32_t i = 0; i < N_EMBD; i++) wbuf[3 + MIX_HC + i] = 1.0f + 0.1f * nd(rng);
    const uint64_t scale_off = 0, base_off = 3 * sizeof(float), norm_off = (3 + MIX_HC) * sizeof(float);
    float *dw; cudaMalloc(&dw, wbuf.size() * sizeof(float));
    cudaMemcpy(dw, wbuf.data(), wbuf.size() * sizeof(float), cudaMemcpyHostToDevice);

    const uint64_t hc_elems = (uint64_t)N_ROWS * N_HC * N_EMBD;
    pulsar_gpu_tensor *res = dalloc(hc_elems * sizeof(pulsar_hc_t));
    pulsar_gpu_tensor *mix = dalloc((uint64_t)N_ROWS * MIX_HC * sizeof(float));
    pulsar_gpu_tensor *split = dalloc((uint64_t)N_ROWS * MIX_HC * sizeof(float));
    pulsar_gpu_tensor *pre = dalloc((uint64_t)N_ROWS * N_HC * sizeof(float));
    pulsar_gpu_tensor *norm_out = dalloc((uint64_t)N_ROWS * N_EMBD * sizeof(float));
    pulsar_gpu_tensor *out = dalloc((uint64_t)N_ROWS * N_EMBD * sizeof(float));

    /* the identity pre-mix, through the shipped fill */
    if (!pulsar_gpu_hc_pre_identity_tensor(pre, N_ROWS, N_HC)) { printf("identity fill FAILED\n"); return 1; }
    std::vector<float> pre_host((size_t)N_ROWS * N_HC);
    cudaMemcpy(pre_host.data(), pre->ptr, pre_host.size() * 4, cudaMemcpyDeviceToHost);
    for (uint32_t t = 0; t < N_ROWS; t++)
        for (uint32_t h = 0; h < N_HC; h++)
            if (pre_host[t * N_HC + h] != (h == 0 ? 1.0f : 0.0f)) { printf("identity fill wrong at %u,%u\n", t, h); return 1; }

    int fails = 0;
    double worst_norm = 0.0, worst_pre = 0.0, worst_comb = 0.0;
    /* two chained sublayers: each collapses with what the one before left */
    for (int call = 0; call < 2; call++) {
        std::vector<float> xh(hc_elems), mh((size_t)N_ROWS * MIX_HC);
        std::vector<pulsar_hc_t> xs(hc_elems);
        for (size_t i = 0; i < hc_elems; i++) { xh[i] = bf16r(nd(rng) * (0.5f + 0.25f * (float)(i % N_HC))); xs[i] = __float2bfloat16(xh[i]); }
        for (auto &v : mh) v = 1.5f * nd(rng);
        cudaMemcpy(res->ptr, xs.data(), hc_elems * sizeof(pulsar_hc_t), cudaMemcpyHostToDevice);
        cudaMemcpy(mix->ptr, mh.data(), mh.size() * 4, cudaMemcpyHostToDevice);
        cudaMemset(split->ptr, 0xFF, split->bytes);

        const std::vector<float> pre_in = pre_host;   /* what this call is handed */
        if (!pulsar_gpu_hc_split_weighted_sum_norm_f16_tensor(out, norm_out, NULL, NULL, 0, NULL, 0u,
                                                             split, pre, mix, res, dw, wbuf.size() * 4,
                                                             scale_off, base_off, norm_off,
                                                             N_ROWS, N_EMBD, N_HC, ITERS, HC_EPS, NORM_EPS, 0) ||
            cudaDeviceSynchronize() != cudaSuccess) {
            printf("call %d: launch FAILED\n", call); return 1;
        }
        std::vector<float> got_norm((size_t)N_ROWS * N_EMBD), got_out((size_t)N_ROWS * N_EMBD),
                           got_split((size_t)N_ROWS * MIX_HC);
        cudaMemcpy(got_norm.data(), norm_out->ptr, got_norm.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(got_out.data(), out->ptr, got_out.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(got_split.data(), split->ptr, got_split.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(pre_host.data(), pre->ptr, pre_host.size() * 4, cudaMemcpyDeviceToHost);

        /* oracle */
        std::vector<float> ref_norm((size_t)N_ROWS * N_EMBD), ref_out((size_t)N_ROWS * N_EMBD),
                           ref_pre((size_t)N_ROWS * N_HC), ref_comb((size_t)N_ROWS * N_HC * N_HC), post(N_HC);
        for (uint32_t t = 0; t < N_ROWS; t++) {
            oracle_split(&mh[t * MIX_HC], &wbuf[0], &wbuf[3], &ref_pre[t * N_HC], post.data(), &ref_comb[t * N_HC * N_HC]);
            double ss = 0.0;
            for (uint32_t d = 0; d < N_EMBD; d++) {
                float acc = 0.0f;
                for (uint32_t h = 0; h < N_HC; h++) acc += xh[((uint64_t)t * N_HC + h) * N_EMBD + d] * pre_in[t * N_HC + h];
                acc = bf16r(acc);   /* hc_pre: y.to(bf16) */
                ref_out[(size_t)t * N_EMBD + d] = acc; ss += (double)acc * acc;
            }
            const float sc = 1.0f / sqrtf((float)(ss / N_EMBD) + NORM_EPS);
            for (uint32_t d = 0; d < N_EMBD; d++)   /* RMSNorm: (weight * x).to(bf16) */
                ref_norm[(size_t)t * N_EMBD + d] = bf16r(ref_out[(size_t)t * N_EMBD + d] * sc * wbuf[3 + MIX_HC + d]);
        }
        const double e_out = rel_l2(got_out, ref_out), e_norm = rel_l2(got_norm, ref_norm);
        const double e_pre = rel_l2(pre_host, ref_pre);
        std::vector<float> got_comb((size_t)N_ROWS * N_HC * N_HC), got_pre_slots((size_t)N_ROWS * N_HC);
        for (uint32_t t = 0; t < N_ROWS; t++) {
            for (uint32_t i = 0; i < N_HC * N_HC; i++) got_comb[t * N_HC * N_HC + i] = got_split[t * MIX_HC + 2 * N_HC + i];
            for (uint32_t h = 0; h < N_HC; h++) got_pre_slots[t * N_HC + h] = got_split[t * MIX_HC + h];
        }
        const double e_comb = rel_l2(got_comb, ref_comb);
        const bool carry_is_own_pre = memcmp(got_pre_slots.data(), pre_host.data(), pre_host.size() * 4) == 0;
        /* the decisive check: the collapse used pre_in, not this call's pre */
        std::vector<float> wrong_out((size_t)N_ROWS * N_EMBD);
        for (uint32_t t = 0; t < N_ROWS; t++)
            for (uint32_t d = 0; d < N_EMBD; d++) {
                float acc = 0.0f;
                for (uint32_t h = 0; h < N_HC; h++) acc += xh[((uint64_t)t * N_HC + h) * N_EMBD + d] * ref_pre[t * N_HC + h];
                wrong_out[(size_t)t * N_EMBD + d] = bf16r(acc);
            }
        const double e_wrong = rel_l2(got_out, wrong_out);
        printf("call %d (%s): collapse rel L2 %.2e vs handed pre (%.2e vs own pre -- must be far), norm %.2e, "
               "carry vs oracle pre %.2e, comb %.2e, split pre slots == carry: %s\n",
               call, call == 0 ? "identity in" : "previous pre in", e_out, e_wrong, e_norm, e_pre, e_comb,
               carry_is_own_pre ? "yes" : "NO");
        worst_norm = fmax(worst_norm, e_norm); worst_pre = fmax(worst_pre, e_pre); worst_comb = fmax(worst_comb, e_comb);
        if (e_out > 1e-5 || e_norm > 1e-4 || e_pre > 1e-5 || e_comb > 1e-4 || !carry_is_own_pre || e_wrong < 1e-2) fails++;
    }
    printf("HC CARRY KERNEL TEST: %s (worst norm %.2e, pre %.2e, comb %.2e)\n", fails ? "FAIL" : "PASS",
           worst_norm, worst_pre, worst_comb);
    return fails ? 1 : 0;
}
