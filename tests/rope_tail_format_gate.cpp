/* Device gate: the tail rope (pulsar_gpu_rope_tail_tensor / _mx_tensor) on every stored element format (L269).
 *
 * The rope picks its kernel instantiation from the buffer.  It used to key on the element WIDTH alone, and f16 and
 * bf16 are both two bytes: V4.1's prefill Q (PULSAR_Q_ELT_FMT = f16) -- roped by the standalone kernel whenever a
 * "Qcur" dump is armed, and the DSpark drafter's Q on every V4.1 draft -- went through the bf16 instantiation, which
 * read and wrote f16 bit patterns as bf16.  Position 0 survived (a rotation by zero round-trips the bits), every
 * later position came back scaled and skewed (|q| +9-20% on the V4.1 fixture), and the "Qcur" dump moved the
 * fixture's teacher-forced KL against DeepSeek's reference from 0.43 to 1.23.
 *
 * What is graded, at V4.1's Q shape (64 heads x 512, 64-dim rope tail) under both of V4.1's rope parameter sets
 * (the uncompressed layers' theta 10000 and the compressed layers' theta 160000 with YaRN), forward and inverse:
 *   - the f16 and bf16 instantiations equal the f32 instantiation's result rounded to their format (the kernel's
 *     contract: operands widen to f32, one f32 rotation core, only the store narrows) -- within half a unit in the
 *     last place of the format, every element;
 *   - the nope dims [0, head_dim - n_rot) are untouched, bit for bit, in every format;
 *   - a two-byte buffer whose format is not f16 or bf16 (PULSAR_ELT_BYTES) is refused, not guessed at.
 * Inputs are k * 2^-6 for |k| <= 127, exactly representable in f32, f16 and bf16, so all three runs start from the
 * same values.
 *
 * No model needed.  build+run: make rope-tail-format-gate CUDA_ARCH=sm_120f
 */
#include "pulsar_gpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

static const uint32_t N_TOK = 5u, N_HEAD = 64u, HEAD_DIM = 512u, N_ROT = 64u;

/* Exact encodings of values that ARE representable (k * 2^-6, |k| <= 127): no rounding is involved, so a plain
 * field copy is the conversion. */
static uint16_t bf16_bits(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof u);
    return (uint16_t)(u >> 16);
}
static uint16_t f16_bits(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof u);
    const uint32_t sign = (u >> 16) & 0x8000u;
    if ((u & 0x7fffffffu) == 0u) return (uint16_t)sign;
    const int exp = (int)((u >> 23) & 0xffu) - 127 + 15;     /* every input here is a normal f16 */
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((u >> 13) & 0x3ffu));
}

struct rope_args { const char *name; float fb, fs, ef, af; uint32_t n_ctx; bool inverse; };

static bool run_one(pulsar_gpu_tensor *t, uint32_t pos0, const rope_args &a) {
    return pulsar_gpu_rope_tail_tensor(t, N_TOK, N_HEAD, HEAD_DIM, N_ROT, pos0, a.n_ctx, a.inverse, a.fb, a.fs,
                                       a.ef, a.af, 32.0f, 1.0f, NULL) != 0;
}

int main(void) {
    const size_t n = (size_t)N_TOK * N_HEAD * HEAD_DIM;
    std::vector<float> x(n);
    uint64_t s = 0x9e3779b97f4a7c15ull;
    for (auto &v : x) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        v = (float)((int)((s >> 40) % 255u) - 127) * 0.015625f;
    }
    std::vector<uint16_t> xh(n), xb(n);
    for (size_t i = 0; i < n; i++) { xh[i] = f16_bits(x[i]); xb[i] = bf16_bits(x[i]); }

    /* V4.1's two rope parameter sets (gpu_prefill.cpp's freq_base / freq_scale / ext_factor / attn_factor): the
     * uncompressed layers (n_ctx_orig 0) and the compressed layers (theta 160000, YaRN factor 16 over 65536). */
    const float yarn_af = 1.0f / (1.0f + 0.1f * logf(16.0f));
    const rope_args sets[] = {
        { "theta 10000, uncompressed", 10000.0f, 1.0f / 16.0f, 1.0f, yarn_af, 0u, false },
        { "theta 160000, YaRN 16/65536", 160000.0f, 1.0f / 16.0f, 1.0f, yarn_af, 65536u, false },
        { "theta 160000, YaRN, inverse", 160000.0f, 1.0f / 16.0f, 1.0f, yarn_af, 65536u, true },
    };
    const uint32_t n_nope = HEAD_DIM - N_ROT;
    for (const rope_args &a : sets) {
        const uint32_t pos0 = 7u;
        pulsar_gpu_tensor *tf = pulsar_gpu_tensor_alloc_elt(n, 4u, PULSAR_ELT_F32);
        pulsar_gpu_tensor *th = pulsar_gpu_tensor_alloc_elt(n, 2u, PULSAR_ELT_F16);
        pulsar_gpu_tensor *tb = pulsar_gpu_tensor_alloc_elt(n, 2u, PULSAR_ELT_BF16);
        CHECK(tf && th && tb, "alloc");
        if (!tf || !th || !tb) return 1;
        CHECK(pulsar_gpu_tensor_write(tf, 0, x.data(), n * sizeof(float)), "write f32");
        CHECK(pulsar_gpu_tensor_write(th, 0, xh.data(), n * sizeof(uint16_t)), "write f16");
        CHECK(pulsar_gpu_tensor_write(tb, 0, xb.data(), n * sizeof(uint16_t)), "write bf16");
        CHECK(run_one(tf, pos0, a) && run_one(th, pos0, a) && run_one(tb, pos0, a), "%s: rope launch", a.name);
        pulsar_gpu_synchronize();
        std::vector<float> rf(n), rh(n), rb(n);
        CHECK(pulsar_gpu_tensor_read_f32(tf, 0, rf.data(), n) && pulsar_gpu_tensor_read_f32(th, 0, rh.data(), n) &&
              pulsar_gpu_tensor_read_f32(tb, 0, rb.data(), n), "%s: read back", a.name);

        /* half an ulp of the format at |r| (f16: 10 stored mantissa bits, bf16: 7), with the f16 subnormal floor */
        uint32_t bad_h = 0, bad_b = 0, nope_bad = 0;
        double worst_h = 0.0, worst_b = 0.0;
        for (size_t i = 0; i < n; i++) {
            const uint32_t d = (uint32_t)(i % HEAD_DIM);
            if (d < n_nope) {
                nope_bad += (rf[i] != x[i]) + (rh[i] != x[i]) + (rb[i] != x[i]);
                continue;
            }
            const double r = rf[i], ar = fabs(r);
            const double eh = fabs((double)rh[i] - r), eb = fabs((double)rb[i] - r);
            const double tol_h = ldexp(ar, -11) + ldexp(1.0, -25), tol_b = ldexp(ar, -8) + 1e-38;
            bad_h += eh > tol_h;
            bad_b += eb > tol_b;
            if (ar > 1e-3) { worst_h = fmax(worst_h, eh / ar); worst_b = fmax(worst_b, eb / ar); }
        }
        CHECK(nope_bad == 0, "%s: %u nope elements moved", a.name, nope_bad);
        CHECK(bad_h == 0, "%s: f16 tail != the f32 rotation rounded to f16 on %u elements (worst rel %.3g)",
              a.name, bad_h, worst_h);
        CHECK(bad_b == 0, "%s: bf16 tail != the f32 rotation rounded to bf16 on %u elements (worst rel %.3g)",
              a.name, bad_b, worst_b);
        printf("%-30s f16 worst rel %.3g (half-ulp 4.9e-4), bf16 worst rel %.3g (half-ulp 3.9e-3), nope untouched\n",
               a.name, worst_h, worst_b);
        pulsar_gpu_tensor_free(tf);
        pulsar_gpu_tensor_free(th);
        pulsar_gpu_tensor_free(tb);
    }

    /* a two-byte buffer that is neither f16 nor bf16 has no arm: refused, never run as one of them */
    {
        pulsar_gpu_tensor *tx = pulsar_gpu_tensor_alloc_elt(n, 2u, PULSAR_ELT_BYTES);
        CHECK(tx != NULL, "alloc bytes");
        if (tx) {
            const bool refused = !run_one(tx, 0u, sets[0]);
            CHECK(refused, "a 2-byte PULSAR_ELT_BYTES buffer was roped instead of refused");
            if (refused) printf("refusal: a 2-byte buffer of format BYTES is refused\n");
            pulsar_gpu_tensor_free(tx);
        }
    }

    printf("ROPE TAIL FORMAT GATE: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
