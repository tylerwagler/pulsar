/* Correctness gate for the CSA2 compressor kernels (L218, DeepSeek-V4.1).
 *
 * #includes the shipped .cu so it drives the REAL kernels, not a copy (the
 * pattern tests/idx_mxfp4_kernel_test.cu set).  The oracle is the reference's
 * Compressor.forward + RMSNorm.forward written out in C, on the same fp32 rows
 * the kernels read: softmax over a group's scores with the weights normalised
 * first, the weighted sum rounded to bf16, RMSNorm in fp32 with the weighted
 * result rounded to bf16; at ratio 1 the projection rounded to bf16 and normed.
 *
 * What is asserted:
 *   ratio 2, prefill:  every emitted latent within 2 bf16 ulps of the oracle
 *                      (exp/rsqrt/reduction-order noise inside one rounding
 *                      step; the bf16 rounding points themselves are exact and
 *                      a wrong rounding point shows as >= 1e-2 relative), and
 *                      the trailing partial group left in the state, the rest
 *                      of the state empty (kv 0, score -inf).
 *   ratio 2, update:   the per-position path over the same tokens emits the
 *                      SAME latents (bit-identical to the prefill path -- same
 *                      kernel, same rows, same order) and leaves the state
 *                      empty after each emit.
 *   ratio 1:           latent = norm(bf16(kv)), no state touched.
 *   store:             a slot store lands at pos %% ratio and nothing else.
 *
 * build+run (on the GPU box):
 *   nvcc -O3 -arch=sm_120f -Isrc -Isrc/cuda -o /tmp/csa2_kt tests/csa2_compressor_kernel_test.cu
 *   /tmp/csa2_kt
 */
#include "../src/cuda/pulsar_cuda_csa2.cu"

#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* cuda_ok / cuda_model_range_ptr live in pulsar_cuda_runtime.cu; this TU links
 * neither the engine nor the rest of the backend, so provide the two symbols
 * the launchers need.  The "model map" here is a plain device buffer, so a
 * range pointer is base + offset. */
int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    return 0;
}
const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    (void)bytes; (void)what;
    return (const char *)model_map + offset;
}

static float bf16r(float v) { return __bfloat162float(__float2bfloat16(v)); }

static pulsar_gpu_tensor *dev_tensor(size_t bytes) {
    pulsar_gpu_tensor *t = (pulsar_gpu_tensor *)calloc(1, sizeof(*t));
    if (cudaMalloc(&t->ptr, bytes) != cudaSuccess) { fprintf(stderr, "cudaMalloc %zu\n", bytes); exit(1); }
    t->bytes = bytes;
    return t;
}
static void dev_write(pulsar_gpu_tensor *t, const void *h, size_t bytes) { cudaMemcpy(t->ptr, h, bytes, cudaMemcpyHostToDevice); }
static void dev_read(void *h, const pulsar_gpu_tensor *t, size_t bytes) { cudaMemcpy(h, t->ptr, bytes, cudaMemcpyDeviceToHost); }

static float frand(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((*s >> 40) & 0xFFFFFF) / 16777216.0f * 2.0f - 1.0f;
}

/* The reference: pooled = sum_r kv_r * softmax(score)_r (weights first), then
 * bf16, then RMSNorm(eps) with a bf16 weight, then bf16. */
static void oracle_latent(const float *kv, const float *sc, uint32_t ratio, uint32_t D,
                          const float *w, float eps, float *out) {
    std::vector<float> pooled(D);
    for (uint32_t d = 0; d < D; d++) {
        float v;
        if (ratio == 1u) {
            v = bf16r(kv[d]);
        } else {
            float m = -INFINITY;
            for (uint32_t r = 0; r < ratio; r++) m = fmaxf(m, sc[r * D + d]);
            float den = 0.0f;
            for (uint32_t r = 0; r < ratio; r++) den += expf(sc[r * D + d] - m);
            float acc = 0.0f;
            for (uint32_t r = 0; r < ratio; r++) acc += kv[r * D + d] * (expf(sc[r * D + d] - m) / den);
            v = bf16r(acc);
        }
        pooled[d] = v;
    }
    double ss = 0.0;
    for (uint32_t d = 0; d < D; d++) ss += (double)pooled[d] * pooled[d];
    const float inv = 1.0f / sqrtf((float)(ss / D) + eps);
    for (uint32_t d = 0; d < D; d++) out[d] = bf16r(w[d] * (pooled[d] * inv));
}

/* bf16 ulps between two bf16-exact values */
static int bf16_ulps(float a, float b) {
    const int ia = (int)(__bfloat16_as_ushort(__float2bfloat16(a)));
    const int ib = (int)(__bfloat16_as_ushort(__float2bfloat16(b)));
    if (a == b) return 0;
    if ((a < 0) != (b < 0)) return 1 << 15;
    return abs(ia - ib);
}

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

int main(void) {
    const uint32_t D = 512u;
    const float eps = 1e-20f;
    uint64_t seed = 0x5eed5eedull;
    /* norm weight, bf16 in the artifact */
    std::vector<__nv_bfloat16> w_b(D); std::vector<float> w_f(D);
    for (uint32_t d = 0; d < D; d++) { w_b[d] = __float2bfloat16(0.5f + frand(&seed)); w_f[d] = __bfloat162float(w_b[d]); }
    pulsar_gpu_tensor *w_dev = dev_tensor(D * sizeof(__nv_bfloat16));
    dev_write(w_dev, w_b.data(), D * sizeof(__nv_bfloat16));

    /* ---- ratio 2: 35 tokens = 17 groups + 1 pending, from position 64 ---- */
    {
        const uint32_t ratio = 2u, n_tok = 35u, pos0 = 64u, n_groups = n_tok / ratio;
        std::vector<float> kv(n_tok * D), sc(n_tok * D);
        for (auto &v : kv) v = 3.0f * frand(&seed);
        for (auto &v : sc) v = 4.0f * frand(&seed);
        pulsar_gpu_tensor *kv_d = dev_tensor(kv.size() * 4), *sc_d = dev_tensor(sc.size() * 4);
        dev_write(kv_d, kv.data(), kv.size() * 4); dev_write(sc_d, sc.data(), sc.size() * 4);
        pulsar_gpu_tensor *lat_d = dev_tensor(n_groups * D * 4);
        pulsar_gpu_tensor *st_kv = dev_tensor(ratio * D * 4), *st_sc = dev_tensor(ratio * D * 4);
        CHECK(pulsar_gpu_csa2_compressor_prefill_tensor(lat_d, kv_d, sc_d, st_kv, st_sc, w_dev->ptr, D * 2, 0, 30u,
                                                        D, ratio, pos0, n_tok, eps), "ratio-2 prefill launch");
        cudaDeviceSynchronize();
        std::vector<float> lat(n_groups * D), ref(D);
        dev_read(lat.data(), lat_d, lat.size() * 4);
        int worst = 0; double worst_rel = 0.0;
        for (uint32_t gi = 0; gi < n_groups; gi++) {
            oracle_latent(&kv[gi * ratio * D], &sc[gi * ratio * D], ratio, D, w_f.data(), eps, ref.data());
            for (uint32_t d = 0; d < D; d++) {
                const int u = bf16_ulps(lat[gi * D + d], ref[d]);
                if (u > worst) worst = u;
                const double rel = fabs((double)lat[gi * D + d] - ref[d]) / (fabs((double)ref[d]) + 1e-6);
                if (rel > worst_rel) worst_rel = rel;
                CHECK(lat[gi * D + d] == bf16r(lat[gi * D + d]), "ratio-2 latent[%u][%u] is not bf16-exact", gi, d);
            }
        }
        CHECK(worst <= 2, "ratio-2 prefill latent vs oracle: %d bf16 ulps (rel %.3g)", worst, worst_rel);
        printf("ratio-2 prefill: %u latents, worst %d bf16 ulp, worst rel %.3g vs the reference oracle\n", n_groups, worst, worst_rel);
        /* state: slot 0 = token 34 (pending), slot 1 empty */
        std::vector<float> skv(ratio * D), ssc(ratio * D);
        dev_read(skv.data(), st_kv, skv.size() * 4); dev_read(ssc.data(), st_sc, ssc.size() * 4);
        int bad = 0;
        for (uint32_t d = 0; d < D; d++) {
            if (skv[d] != kv[(n_tok - 1u) * D + d] || ssc[d] != sc[(n_tok - 1u) * D + d]) bad++;
            if (skv[D + d] != 0.0f || ssc[D + d] != -INFINITY) bad++;
        }
        CHECK(bad == 0, "ratio-2 prefill state: %d elements wrong (slot 0 must be token %u, slot 1 empty)", bad, n_tok - 1u);

        /* per-position path over the same tokens: same latents, empty state after each emit */
        pulsar_gpu_tensor *row_d = dev_tensor(D * 4);
        std::vector<float> row(D);
        int mism = 0, emits = 0, state_bad = 0;
        cudaMemset(st_kv->ptr, 0, ratio * D * 4);
        std::vector<float> ninf(ratio * D, -INFINITY); dev_write(st_sc, ninf.data(), ninf.size() * 4);
        for (uint32_t t = 0; t < n_tok; t++) {
            pulsar_gpu_tensor kvv = *kv_d, scv = *sc_d;
            kvv.ptr = (char *)kv_d->ptr + (size_t)t * D * 4; kvv.bytes = D * 4;
            scv.ptr = (char *)sc_d->ptr + (size_t)t * D * 4; scv.bytes = D * 4;
            int emitted = 0;
            CHECK(pulsar_gpu_csa2_compressor_update_tensor(row_d, &kvv, &scv, st_kv, st_sc, w_dev->ptr, D * 2, 0, 30u,
                                                           D, ratio, pos0 + t, eps, &emitted), "ratio-2 update launch t=%u", t);
            cudaDeviceSynchronize();
            const int want_emit = ((pos0 + t + 1u) % ratio) == 0u;
            CHECK(emitted == want_emit, "ratio-2 update t=%u emitted %d want %d", t, emitted, want_emit);
            if (emitted) {
                dev_read(row.data(), row_d, D * 4);
                const uint32_t gi = t / ratio;
                for (uint32_t d = 0; d < D; d++) if (row[d] != lat[gi * D + d]) mism++;
                emits++;
                dev_read(skv.data(), st_kv, skv.size() * 4); dev_read(ssc.data(), st_sc, ssc.size() * 4);
                for (uint32_t i = 0; i < ratio * D; i++) if (skv[i] != 0.0f || ssc[i] != -INFINITY) state_bad++;
            }
        }
        CHECK(emits == (int)n_groups, "ratio-2 update emitted %d groups, want %u", emits, n_groups);
        CHECK(mism == 0, "ratio-2 update vs prefill latents: %d elements differ (must be bit-identical)", mism);
        CHECK(state_bad == 0, "ratio-2 update: %d state elements not empty after an emit", state_bad);
        printf("ratio-2 update: %d emits, %d mismatches vs prefill, state empty after emit: %s\n",
               emits, mism, state_bad ? "NO" : "yes");
        /* store: the last position's slot */
        {
            pulsar_gpu_tensor kvv = *kv_d, scv = *sc_d;
            kvv.ptr = (char *)kv_d->ptr + (size_t)(n_tok - 1u) * D * 4; kvv.bytes = D * 4;
            scv.ptr = (char *)sc_d->ptr + (size_t)(n_tok - 1u) * D * 4; scv.bytes = D * 4;
            CHECK(pulsar_gpu_csa2_compressor_store_tensor(&kvv, &scv, st_kv, st_sc, D, ratio, pos0 + n_tok - 1u), "store launch");
            cudaDeviceSynchronize();
            dev_read(skv.data(), st_kv, skv.size() * 4); dev_read(ssc.data(), st_sc, ssc.size() * 4);
            int sbad = 0;
            const uint32_t slot = (pos0 + n_tok - 1u) % ratio;
            for (uint32_t d = 0; d < D; d++) {
                if (skv[slot * D + d] != kv[(n_tok - 1u) * D + d]) sbad++;
                if (skv[(1u - slot) * D + d] != 0.0f) sbad++;
            }
            CHECK(sbad == 0, "store: %d elements wrong (slot %u)", sbad, slot);
        }
        /* an unaligned start refuses */
        CHECK(!pulsar_gpu_csa2_compressor_prefill_tensor(lat_d, kv_d, sc_d, st_kv, st_sc, w_dev->ptr, D * 2, 0, 30u,
                                                         D, ratio, pos0 + 1u, n_tok, eps),
              "ratio-2 prefill at an odd position was not refused");
    }
    /* ---- ratio 1: 7 tokens, no state ---- */
    {
        const uint32_t ratio = 1u, n_tok = 7u, pos0 = 9u;
        std::vector<float> kv(n_tok * D);
        for (auto &v : kv) v = 3.0f * frand(&seed);
        pulsar_gpu_tensor *kv_d = dev_tensor(kv.size() * 4);
        dev_write(kv_d, kv.data(), kv.size() * 4);
        pulsar_gpu_tensor *lat_d = dev_tensor(n_tok * D * 4);
        CHECK(pulsar_gpu_csa2_compressor_prefill_tensor(lat_d, kv_d, NULL, NULL, NULL, w_dev->ptr, D * 2, 0, 30u,
                                                        D, ratio, pos0, n_tok, eps), "ratio-1 prefill launch");
        cudaDeviceSynchronize();
        std::vector<float> lat(n_tok * D), ref(D);
        dev_read(lat.data(), lat_d, lat.size() * 4);
        int worst = 0;
        for (uint32_t t = 0; t < n_tok; t++) {
            oracle_latent(&kv[t * D], NULL, ratio, D, w_f.data(), eps, ref.data());
            for (uint32_t d = 0; d < D; d++) { const int u = bf16_ulps(lat[t * D + d], ref[d]); if (u > worst) worst = u; }
        }
        CHECK(worst <= 2, "ratio-1 prefill latent vs oracle: %d bf16 ulps", worst);
        pulsar_gpu_tensor *row_d = dev_tensor(D * 4);
        std::vector<float> row(D);
        int mism = 0;
        for (uint32_t t = 0; t < n_tok; t++) {
            pulsar_gpu_tensor kvv = *kv_d; kvv.ptr = (char *)kv_d->ptr + (size_t)t * D * 4; kvv.bytes = D * 4;
            int emitted = 0;
            CHECK(pulsar_gpu_csa2_compressor_update_tensor(row_d, &kvv, NULL, NULL, NULL, w_dev->ptr, D * 2, 0, 30u,
                                                           D, ratio, pos0 + t, eps, &emitted), "ratio-1 update launch");
            cudaDeviceSynchronize();
            CHECK(emitted == 1, "ratio-1 update t=%u did not emit", t);
            dev_read(row.data(), row_d, D * 4);
            for (uint32_t d = 0; d < D; d++) if (row[d] != lat[t * D + d]) mism++;
        }
        CHECK(mism == 0, "ratio-1 update vs prefill: %d elements differ", mism);
        printf("ratio-1: %u latents, worst %d bf16 ulp vs the oracle, update == prefill: %s\n", n_tok, worst, mism ? "NO" : "yes");
    }
    printf("CSA2 COMPRESSOR KERNEL TEST: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
