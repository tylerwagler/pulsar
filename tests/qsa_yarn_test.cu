/* L280: pulsar_qsa_rope_init graded element-wise against transformers' own rope init for Qwen3.8-Flash-Next.
 *
 * tests/qsa_yarn_golden.h is tools/qwen/yarn_golden.py's output: for the native rope (Qwen4ExpTextRotaryEmbedding.
 * compute_default_rope_parameters) and the model card's YaRN (`_compute_yarn_parameters`, factor 4 over 262144),
 *   *_f64     transformers' function evaluated with its torch dtypes mapped to float64, each frequency rounded once
 *             to f32 -- the exact value, which the engine's table must equal BIT FOR BIT
 *   *_hf_f32  the function as shipped (torch float32: pow, both divisions and the blend each round): the engine's
 *             distance from it is reported in ulps and bounded at 2
 * plus the attention factor (Python double): the engine's mscale must be its f32 rounding.
 *
 * Host only (no kernel runs): pulsar_qsa_rope_init is host code of pulsar_cuda_qsa.cu. */
#include "pulsar_gpu.h"
#include "qsa_yarn_golden.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail = 1; printf("  FAIL  "); } else { printf("  ok    "); } \
                              printf(__VA_ARGS__); printf("\n"); } while (0)

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static void grade(const char *name, const pulsar_qsa_rope &r, const uint32_t *f64, const uint32_t *hf32, double att) {
    constexpr uint32_t N = PULSAR_QSA_ROT_DIM / 2;
    uint32_t exact = 0, max_ulp = 0, off = 0;
    for (uint32_t i = 0; i < N; i++) {
        const uint32_t b = bits(r.inv_freq[i]);
        exact += b == f64[i];
        const uint32_t ulp = b > hf32[i] ? b - hf32[i] : hf32[i] - b;   /* positive finite floats: bits are monotone */
        off += ulp != 0;
        if (ulp > max_ulp) max_ulp = ulp;
        if (b != f64[i]) printf("        pair %2u: engine %08x, transformers f64 %08x\n", i, b, f64[i]);
    }
    CHECK(exact == N, "%s: %u of %u pairs equal transformers' f64 table rounded once to f32", name, exact, N);
    CHECK(max_ulp <= 2, "%s: %u of %u pairs differ from transformers' shipped f32 table, by at most %u ulp", name, off, N,
          max_ulp);
    CHECK(bits(r.mscale) == bits((float)att), "%s: mscale %.9g == f32(transformers' attention factor %.17g)", name,
          (double)r.mscale, att);
}

int main() {
    pulsar_qsa_rope nat, yarn;
    CHECK(pulsar_qsa_rope_init(&nat, NULL) == 1, "the native rope");
    const pulsar_qsa_yarn y{qsa_golden_yarn_factor, qsa_golden_yarn_orig_ctx, 32.0, 1.0};
    CHECK(pulsar_qsa_rope_init(&yarn, &y) == 1, "YaRN x%g over %u", y.factor, y.orig_ctx);
    grade("native", nat, qsa_golden_native_f64, qsa_golden_native_hf_f32, qsa_golden_native_attention_factor);
    grade("yarn x4", yarn, qsa_golden_yarn_f64, qsa_golden_yarn_hf_f32, qsa_golden_yarn_attention_factor);
    /* the ramp's ends, from the table itself: the pairs past high are the native frequency / 4 exactly, the pairs
     * up to low are the native frequency -- one blend, no third behaviour */
    uint32_t kept = 0, quartered = 0;
    for (uint32_t i = 0; i < PULSAR_QSA_ROT_DIM / 2; i++) {
        kept += bits(yarn.inv_freq[i]) == bits(nat.inv_freq[i]);
        quartered += bits(yarn.inv_freq[i]) == bits(nat.inv_freq[i] * 0.25f);
    }
    CHECK(kept == 15 && quartered == 10, "pairs 0..14 extrapolated (%u), 22..31 interpolated (%u): the ramp is [14, 22]",
          kept, quartered);
    printf("\nqsa_yarn_test: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
