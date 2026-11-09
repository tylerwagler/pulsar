/* Host-side KV row fixture helpers, shared by the attention gates and the row
 * pack gate: exact host replicas of the two V4.1 row packers
 * (src/cuda/pulsar_cuda_kvrows.cu) so a gate can build rows, know their
 * decoded values, and pin the device packer byte for byte.
 *
 * WHY A FIXTURE ENCODES RATHER THAN DRAWS BYTES.  When the raw KV caches moved
 * to a packed row (2026-08-17), these gates were updated to build packed rows
 * by filling the E4M3 payload with uniform random BYTES.  That is not a
 * uniform distribution of values: a uniform byte is uniform in EXPONENT, so
 * the decoded magnitudes span 2^-9..448 and are dominated by the large end.  A
 * 512-wide q.k dot over such rows produces scores tens of thousands apart, and
 * the softmax collapses to one-hot.
 *
 * A one-hot softmax silently removes what these gates exist to measure:
 *   - attn_f16_kernel_test: median per-(tok,head) rel L2 fell to 4e-12 (exact
 *     agreement on a degenerate input) while a few sink-dominated heads with
 *     |ref| ~ 3e-4 reported 6.4e-2 and failed the gate on nothing but a
 *     small-norm denominator.
 *   - the split-KV merge gate (deleted with the f32 decode kernel, L166):
 *     residuals of 1.3e-49 and 0.0e+00, because when one split held all the
 *     weight there was no reassociation to measure and the softmax MERGE --
 *     the entire subject of that test -- never ran.
 *
 * Encoding a draw fixes both: the decoded magnitude tracks the DRAW, and the
 * scale byte follows the data instead of the other way round.
 *
 * THE RECIPES, mirrored from the device (any divergence is a gate failure,
 * not slop -- the packers use bit-field E8M0 rounding and IEEE division, so
 * there is no fast-math budget any more):
 *   WINDOW  v = bf16(x); per 32: a = max(amax, 1e-4); e8 = round_up(a * (1/448));
 *           code = e4m3_rne_sat(v / 2^(e8-127)); row = [HD codes][HD/32 e8].
 *   MAIN    v = bf16(x); per 16: a = max(amax, 6*2^-9); se = e4m3_rne_sat(a / 6);
 *           nib = e2m1_rne(clamp(v / e4m3(se), +-6)); row = [HD/2 nibble bytes,
 *           low nibble = even dim][HD/16 se].
 */
#ifndef PULSAR_TESTS_KV_ROW_FIXTURE_H
#define PULSAR_TESTS_KV_ROW_FIXTURE_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

/* ---- element codecs (inverse searches of the value tables, so encode and
 * decode cannot drift; fixture code -- clarity over speed) ---------------- */
static inline float host_e4m3_mag(uint8_t code) {
    const uint32_t e = (code >> 3) & 15u, m = code & 7u;
    if (e == 15u && m == 7u) return NAN;
    if (e == 0u) return (float)m * 0.001953125f;
    return std::ldexp(1.0f + (float)m / 8.0f, (int)e - 7);
}
/* RNE, saturating at 448 -- what __nv_fp8_e4m3(float) does on finite input.
 * Ties go to the even CODE, which is the even mantissa (code LSB == mantissa LSB). */
static inline uint8_t host_e4m3_encode_pos(float x) {
    if (x >= 448.0f) return 126u;
    uint32_t best = 0u; float bd = std::fabs(x - host_e4m3_mag(0));
    for (uint32_t c = 1u; c <= 126u; c++) {
        const float d = std::fabs(x - host_e4m3_mag((uint8_t)c));
        if (d < bd || (d == bd && (c & 1u) == 0u && (best & 1u) != 0u)) { best = c; bd = d; }
    }
    return (uint8_t)best;
}
static inline uint8_t host_e4m3_encode(float x) {
    return (uint8_t)(host_e4m3_encode_pos(std::fabs(x)) | (std::signbit(x) ? 0x80u : 0u));
}
/* the device's pulsar_e4m3_times: magnitude * scale, sign applied last */
static inline float host_e4m3_times(uint8_t code, float scale) {
    const float sv = host_e4m3_mag((uint8_t)(code & 0x7Fu)) * scale;
    return (code & 0x80u) ? -sv : sv;
}
static inline float host_e2m1_value(uint32_t c) {
    static const float t[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return t[c & 7u];
}
/* the device's dsv4_e2m1fn_encode_dev: nearest code, ties to even, sign from x < 0 */
static inline uint8_t host_e2m1_encode(float x) {
    const float ax = std::fmin(std::fabs(x), 6.0f);
    uint32_t best = 0u; float bd = std::fabs(ax - host_e2m1_value(0));
    for (uint32_t i = 1u; i < 8u; i++) {
        const float d = std::fabs(ax - host_e2m1_value(i));
        if (d < bd || (d == bd && (i & 1u) == 0u && (best & 1u) != 0u)) { best = i; bd = d; }
    }
    return (uint8_t)(best | (x < 0.0f ? 0x8u : 0u));
}
static inline float host_e2m1_times(uint8_t nib, float scale) {
    const float sv = host_e2m1_value(nib & 7u) * scale;
    return (nib & 8u) ? -sv : sv;
}
static inline float host_bf16r(float x) {
    uint32_t u; std::memcpy(&u, &x, sizeof u);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float f; std::memcpy(&f, &u, sizeof f); return f;
}
/* fast_round_scale's E8M0 byte (the device's pulsar_e8m0_round_up) */
static inline uint32_t host_e8m0_round_up(float y) {
    uint32_t b; std::memcpy(&b, &y, sizeof b);
    const uint32_t e8 = ((b >> 23) & 0xFFu) + (((b & 0x7FFFFFu) != 0u) ? 1u : 0u);
    return e8 > 254u ? 254u : e8;
}
static inline float host_e8m0_scale(uint32_t e8) {
    const uint32_t u = e8 << 23; float f; std::memcpy(&f, &u, sizeof f); return f;
}

/* ---- the two row packers ------------------------------------------------ */
/* Pack one head_dim-wide float row as a WINDOW row; `dec` (optional) receives
 * the decoded values so an oracle and the kernel look at the same numbers. */
static inline void host_winkv_pack_row(const float *vals, uint8_t *row, float *dec, uint32_t head_dim) {
    for (uint32_t b = 0; b < head_dim / 32u; b++) {
        float amax = 0.0f;
        for (uint32_t d = b * 32u; d < (b + 1u) * 32u; d++) amax = std::fmax(amax, std::fabs(host_bf16r(vals[d])));
        const uint32_t e8 = host_e8m0_round_up(std::fmax(amax, 1e-4f) * (1.0f / 448.0f));
        const float s = host_e8m0_scale(e8);
        row[head_dim + b] = (uint8_t)e8;
        for (uint32_t d = b * 32u; d < (b + 1u) * 32u; d++) {
            const float q = std::fmin(448.0f, std::fmax(-448.0f, host_bf16r(vals[d]) / s));
            row[d] = host_e4m3_encode(q);
            if (dec) dec[d] = host_e4m3_times(row[d], s);
        }
    }
}
/* Pack one head_dim-wide float row as a MAIN row. */
static inline void host_mainkv_pack_row(const float *vals, uint8_t *row, float *dec, uint32_t head_dim) {
    for (uint32_t b = 0; b < head_dim / 16u; b++) {
        float amax = 0.0f;
        for (uint32_t d = b * 16u; d < (b + 1u) * 16u; d++) amax = std::fmax(amax, std::fabs(host_bf16r(vals[d])));
        const uint8_t se = host_e4m3_encode_pos(std::fmax(amax, 6.0f * 0.001953125f) / 6.0f);
        const float s = host_e4m3_mag(se);
        row[head_dim / 2u + b] = se;
        for (uint32_t d = b * 16u; d < (b + 1u) * 16u; d += 2u) {
            const uint8_t n0 = host_e2m1_encode(std::fmin(6.0f, std::fmax(-6.0f, host_bf16r(vals[d]) / s)));
            const uint8_t n1 = host_e2m1_encode(std::fmin(6.0f, std::fmax(-6.0f, host_bf16r(vals[d + 1u]) / s)));
            row[d >> 1] = (uint8_t)(n0 | (n1 << 4));
            if (dec) { dec[d] = host_e2m1_times(n0, s); dec[d + 1u] = host_e2m1_times(n1, s); }
        }
    }
}

#ifdef __CUDACC__   /* the Q upload needs the CUDA types; host-only gates take the codecs alone */
/* ---- Q upload in the engine's stored element type -------------------------
 *
 * Fixtures must NOT assume Q is f32.  pulsar_q_t narrowed to __half (L045), and
 * a fixture that cudaMemcpys floats into that buffer keeps compiling and
 * running while handing the kernel garbage.  The quiet variant is worse: the
 * decode kernels are TEMPLATED on the Q type, so a `float *` argument deduces
 * QT=float and the gate certifies an instantiation the engine never launches --
 * green, and measuring nothing.
 *
 * Going through here ties the fixture to pulsar_q_t by construction. */
static inline void fixture_q_set(float &d, float v)  { d = v; }
static inline void fixture_q_set(__half &d, float v) { d = __float2half(v); }

static inline pulsar_q_t *fixture_upload_q(const std::vector<float> &q) {
    pulsar_q_t *d = NULL;
    if (cudaMalloc(&d, q.size() * sizeof(pulsar_q_t)) != cudaSuccess) return NULL;
    std::vector<pulsar_q_t> h(q.size());
    for (size_t i = 0; i < q.size(); i++) fixture_q_set(h[i], q[i]);
    if (cudaMemcpy(d, h.data(), h.size() * sizeof(pulsar_q_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        return NULL;
    }
    return d;
}

#endif /* __CUDACC__ */

#endif /* PULSAR_TESTS_KV_ROW_FIXTURE_H */
